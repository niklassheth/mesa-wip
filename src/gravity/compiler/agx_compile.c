/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "gravity/debug.h"
#include "agx_compile.h"
#include "gravity/clc/gravity_clc.h"
#include "gravity/layout/layout.h"
#include "gravity/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "glsl_types.h"
#include "nir.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "shader_enums.h"

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"shaders",   AGX_DBG_SHADERS,	"Dump shaders in NIR and AIR"},
   {"shaderdb",  AGX_DBG_SHADERDB,	"Print statistics"},
   {"verbose",   AGX_DBG_VERBOSE,	"Disassemble verbosely"},
   {"internal",  AGX_DBG_INTERNAL,	"Dump even internal shaders"},
   {"novalidate",AGX_DBG_NOVALIDATE,"Skip IR validation in debug builds"},
   {"noopt",     AGX_DBG_NOOPT,     "Disable backend optimizations"},
   {"wait",      AGX_DBG_WAIT,      "Wait after all async instructions"},
   {"nopreamble",AGX_DBG_NOPREAMBLE,"Do not use shader preambles"},
   {"demand",    AGX_DBG_DEMAND,    "Bound tightly to register demand"},
   {"nosched",   AGX_DBG_NOSCHED,   "Do not schedule the shader"},
   {"spill",     AGX_DBG_SPILL,     "Spill (almost) everything"},
   {"nopromote", AGX_DBG_NOPROMOTE, "Do not promote constants to uniforms"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

DEBUG_GET_ONCE_FLAGS_OPTION(agx_compiler_debug, "AGX_MESA_DEBUG",
                            agx_debug_options, 0)

bool
agx_apple9_trace_enabled(void)
{
   /* Any value, including an empty string or "0", enables tracing. */
   return agx_apple9_debug_enabled(AGX_A9_DBG_TRACE);
}

uint64_t
agx_get_compiler_debug(void)
{
   return debug_get_option_agx_compiler_debug();
}

/* Table describing the relationship between register pressure and thread
 * count on Apple9 (G16G B1 / T8132 / M4), empirically measured on silicon.
 * Each entry describes a maximum number of half-registers (2 * 32-bit GPRs)
 * and the associated maximum resident thread count per core.
 *
 * Dual 160 KiB/core register banks scale occupancy up to 2048 threads/core
 * (64 warps) below 48 GPRs and stay flat at 832 threads/core (26 warps)
 * across 45..96 GPRs. Sorted in ascending order of maximum registers.
 */
static const struct agx_occupancy occupancies[] = {
   { 36, 2048 }, /* <= 18 GPRs: 64 warps/core */
   { 48, 1664 }, /* <= 24 GPRs: 52 warps/core */
   { 56, 1536 }, /* <= 28 GPRs: 48 warps/core */
   { 64, 1280 }, /* <= 32 GPRs: 40 warps/core */
   { 72, 1152 }, /* <= 36 GPRs: 36 warps/core */
   { 80,  960 }, /* <= 40 GPRs: 30 warps/core */
   { 88,  896 }, /* <= 44 GPRs: 28 warps/core */
   { 192, 832 }, /* 45..96 GPRs: 26 warps/core */
};

struct agx_occupancy
agx_occupancy_for_register_count(unsigned halfregs)
{
   for (unsigned i = 0; i < ARRAY_SIZE(occupancies); ++i) {
      unsigned max = occupancies[i].max_registers;
      assert((i == 0 || max > occupancies[i - 1].max_registers) && "ascending");

      if (halfregs <= max)
         return occupancies[i];
   }

   /* Clamp pre-spill estimates above Apple9's 96-GPR register file to the
    * floor occupancy tier rather than aborting. */
   return occupancies[ARRAY_SIZE(occupancies) - 1];
}

static bool
agx_lower_sincos_filter(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fsin || alu->op == nir_op_fcos;
}

static nir_def *
agx_lower_sincos_impl(struct nir_builder *b, nir_instr *instr, UNUSED void *_)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], 1);
   nir_def *turns = nir_fmul_imm(b, x, M_1_PI * 0.5f);

   if (alu->op == nir_op_fcos)
      turns = nir_fadd_imm(b, turns, 0.25f);

   nir_def *quadrants = nir_fmul_imm(b, nir_ffract(b, turns), 4.0);
   return nir_fsin_agx(b, quadrants);
}

static bool
agx_lower_sincos(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader, agx_lower_sincos_filter,
                                        agx_lower_sincos_impl, NULL);
}

static bool
agx_lower_front_face(struct nir_builder *b, nir_intrinsic_instr *intr,
                     UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_front_face)
      return false;

   nir_def *def = &intr->def;
   assert(def->bit_size == 1);

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(def, nir_inot(b, nir_load_back_face_agx(b, 1)));
   return true;
}

/*
 * Standard NIR optimization loop run in agx_preprocess_nir.
 */
static void
agx_optimize_loop_nir(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

bool
agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                     unsigned bit_size, unsigned num_components,
                     int64_t hole_size, nir_intrinsic_instr *low,
                     nir_intrinsic_instr *high, void *data)
{
   if (hole_size > 0)
      return false;

   /* Must be aligned to the size of the load */
   unsigned align = nir_combined_align(align_mul, align_offset);
   if ((bit_size / 8) > align)
      return false;

   if (num_components > 4)
      return false;

   if (bit_size > 32)
      return false;

   return true;
}

static bool
lower_sink_address(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ro_sink_address_poly:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      return true;
   case nir_intrinsic_ro_to_rw_poly: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(
         &intr->def,
         nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, AGX_ZERO_PAGE_ADDRESS),
                   nir_imm_int64(b, AGX_SCRATCH_PAGE_ADDRESS),
                   intr->src[0].ssa));
      return true;
   }
   default:
      break;
   }

   return false;
}

bool
agx_nir_lower_sink_address(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_sink_address,
                                     nir_metadata_control_flow, NULL);
}

/*
 * The hardware frcp instruction is sometimes off by 1 ULP. For correctly
 * rounded frcp, a refinement step is required.
 */
static nir_def *
libagx_frcp(nir_builder *b, nir_def *x)
{
   nir_def *u = nir_frcp(b, x);
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *u_2 = nir_ffma(b, nir_ffma(b, nir_fneg(b, x), u, one), u, u);
   return nir_bcsel(b, nir_fisnan(b, u_2), u, u_2);
}

static bool
agx_nir_lower_fdiv(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_frcp || !nir_alu_instr_is_exact(alu) ||
       alu->def.bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, libagx_frcp(b, nir_ssa_for_alu_src(b, alu, 0)));
   return true;
}

bool
agx_nir_lower_accurate_frcp(nir_shader *nir)
{
   return nir_shader_alu_pass(nir, agx_nir_lower_fdiv,
                              nir_metadata_control_flow, NULL);
}

/* Preprocess NIR independent of shader state */
void
agx_preprocess_nir(nir_shader *nir)
{
   if (!nir)
      return;

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Lower large arrays to scratch and small arrays to csel */
   NIR_PASS(_, nir, nir_lower_vars_to_scratch, 256,
            glsl_get_natural_size_align_bytes, glsl_get_word_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_var_copies);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, agx_nir_lower_frag_sidefx);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }

   /* Clean up deref gunk after lowering I/O */
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_frexp);
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_flrp, 16 | 32 | 64, false);
   NIR_PASS(_, nir, agx_lower_sincos);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_lower_front_face,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_all_phis_to_scalar);

   agx_optimize_loop_nir(nir);

   /* Lower idiv after an optimization loop so we can constant fold more before
    * nir_opt_idiv_const.
    */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   nir_lower_idiv_options idiv_options = {
      .allow_fp16 = true,
   };

   NIR_PASS(_, nir, nir_lower_idiv, &idiv_options);

   /* Has to run after nir_lower_idiv */
   NIR_PASS(_, nir, agx_nir_lower_accurate_frcp);

   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* We're lowered away all variables. Remove them all for smaller shaders. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);
   nir->info.io_lowered = true;

   /* Move before lowering */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
}

static bool
lower_printf_buffer(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *_)
{
   uint64_t value = 0;
   if (intr->intrinsic == nir_intrinsic_load_printf_buffer_address)
      value = LIBAGX_PRINTF_BUFFER_ADDRESS;
   else if (intr->intrinsic == nir_intrinsic_load_printf_buffer_size)
      value = LIBAGX_PRINTF_BUFFER_SIZE;
   else
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

bool
agx_nir_lower_printf_buffer(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_printf_buffer,
                                     nir_metadata_control_flow, NULL);
}
