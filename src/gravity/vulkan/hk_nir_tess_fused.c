/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <float.h>
#include <math.h>
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "hk_apple9_mesh.h"
#include "hk_cmd_buffer.h"
#include "hk_nir_fused_vs.h"
#include "hk_shader.h"

/* =========================================================================
 * Zero-PREGFX Fused OBJECT(VS + TCS) -> MESH(TES) HW-Mesh Tessellation
 *
 * Eliminates the PREGFX CDM compute pass, intermediate patch/vertex DRAM ring,
 * tessellator count/scan/emit dispatches, and HK_READS_INTERNAL tiling barrier
 * for hardware-eligible direct and indexed tessellation draws:
 *
 * - OBJECT(VS + TCS) runs 1 parent workgroup per input patch (physical T32),
 *   fetches VBO/IBO vertices in parallel via the shared fused-VS helper, stages
 *   live VS->TCS outputs in threadgroup shared memory, runs TCS with top-level
 *   barriers preserved, culls patches with any outer <= 0 or NaN via
 *   EmitMeshTasksEXT(0, 0, 0), writes a compact 40 B header + tightly packed
 *   live TCS outputs directly into per-parent GET_SR 0xe0/0xe1 task payload,
 *   and launches the exact child MESH workgroup count.
 *
 * - MESH(TES) reads the 40 B header and live patch CPs directly from
 *   GET_SR 0xe0/0xe1, generates (u, v) domain coordinates and shared-vertex
 *   meshlet topology procedurally in registers (Quads and Triangles; Equal,
 *   FractionalOdd, and FractionalEven spacing; CW and CCW winding + dynamic
 *   domain origin), evaluates ONE shared copy of the TES body in parallel
 *   across unique meshlet vertices, and publishes directly to the Apple9
 *   hardware mesh output ring.
 * ========================================================================= */

#define HK_TESS_PAYLOAD_HEADER_B 40u
#define HK_TESS_SHM_LEVELS_B     24u /* 6 x float: outer[0..3], inner[0..1] */

#define HK_TESS_FLAG_FLIP_WINDING (1u << 0)
#define HK_TESS_FLAG_ALL_ONES     (1u << 1)
#define HK_TESS_FLAG_WHOLE_PATCH  (1u << 2)

/* Minimum positive 16.16 fixed-point epsilon matching poly_tessellator.h. */
#define HK_TESS_ODD_MIN_PLUS_EPS  (1.0f + 1.0f / 65536.0f)

struct tess_meshlet_shape {
   unsigned max_v;
   unsigned max_p;
   unsigned tile_w;
   unsigned tile_h;
   unsigned p_strip;
   bool allow_whole_patch;
};

static struct tess_meshlet_shape
choose_meshlet_shape(unsigned vertex_components)
{
   unsigned planes = ALIGN_POT(vertex_components, 4);
   if (planes <= 4) {
      /* Fits tess4 tri (V19/P24) and tess4 quad (V25/P32) in one child MESH WG. */
      return (struct tess_meshlet_shape){
         .max_v = 32,
         .max_p = 42,
         .tile_w = 5,
         .tile_h = 4,
         .p_strip = 30,
         .allow_whole_patch = true,
      };
   }
   if (planes <= 68) {
      /* Safe-stride U0_B unlocks wg_rows == 32 and wg_rows > 40 across all
       * planes <= 68 (3.78x faster on T8132 silicon at comps=32 vs V4/P2). */
      return (struct tess_meshlet_shape){
         .max_v = 16,
         .max_p = 18,
         .tile_w = 3,
         .tile_h = 3,
         .p_strip = 14,
         .allow_whole_patch = true,
      };
   }
   /* Very wide varying payloads (planes 72..128): V8/P6 keeps safe-stride
    * ring occupancy compact while still 2.6x+ wider than V4/P2. */
   return (struct tess_meshlet_shape){
      .max_v = 8,
      .max_p = 6,
      .tile_w = 3,
      .tile_h = 1,
      .p_strip = 6,
      .allow_whole_patch = true,
   };
}

static bool
tes_output_slot_ok(unsigned loc, unsigned comp, unsigned num_comps,
                   unsigned clip_size, unsigned cull_size)
{
   if (comp + num_comps > 4)
      return false;
   switch (loc) {
   case VARYING_SLOT_POS:
      return true;
   case VARYING_SLOT_PSIZ:
      return comp == 0 && num_comps == 1;
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
      return 4 * (loc - VARYING_SLOT_CLIP_DIST0) + comp + num_comps <= 8;
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
      return clip_size + 4 * (loc - VARYING_SLOT_CULL_DIST0) + comp +
                num_comps <=
             8 &&
             cull_size > 0;
   default:
      return loc >= VARYING_SLOT_VAR0 && loc < VARYING_SLOT_VAR0 + 32;
   }
}

static bool
is_control_barrier(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(intr) != SCOPE_NONE;
}

static bool
validate_tcs_barriers(nir_shader *tcs, const char **reason)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(tcs);
   nir_foreach_block(block, impl) {
      bool top_level = block->cf_node.parent == &impl->cf_node;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_barrier)
            continue;
         if (!top_level && nir_intrinsic_execution_scope(intr) != SCOPE_NONE) {
            if (reason)
               *reason = "fused tess requires top-level TCS control barriers";
            return false;
         }
      }
   }
   return true;
}

static void
scan_tes_clip_cull_sizes(const nir_shader *tes, unsigned *clip_out,
                         unsigned *cull_out)
{
   unsigned clip_size = tes->info.clip_distance_array_size;
   unsigned cull_size = tes->info.cull_distance_array_size;
   if (clip_size || cull_size) {
      *clip_out = clip_size;
      *cull_out = cull_size;
      return;
   }
   nir_foreach_function_impl(impl, (nir_shader *)tes) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output ||
                !nir_src_is_const(intr->src[1]))
               continue;
            uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned wmask = nir_intrinsic_write_mask(intr);
            if (!wmask)
               continue;
            unsigned hi = nir_intrinsic_component(intr) + util_last_bit(wmask);
            if (loc == VARYING_SLOT_CLIP_DIST0 || loc == VARYING_SLOT_CLIP_DIST1) {
               unsigned end = 4u * (loc - VARYING_SLOT_CLIP_DIST0) + hi;
               if (end > clip_size)
                  clip_size = end;
            } else if (loc == VARYING_SLOT_CULL_DIST0 ||
                       loc == VARYING_SLOT_CULL_DIST1) {
               unsigned end = 4u * (loc - VARYING_SLOT_CULL_DIST0) + hi;
               if (end > cull_size)
                  cull_size = end;
            }
         }
      }
   }
   *clip_out = clip_size;
   *cull_out = cull_size;
}

static bool
validate_stages(const nir_shader *vs, const nir_shader *tcs,
                const nir_shader *tes,
                const struct hk_tess_fused_options *opts,
                enum tess_primitive_mode *out_domain,
                enum gl_tess_spacing *out_spacing, bool *out_ccw,
                unsigned *out_tcs_verts, const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (!vs || !tcs || !tes || vs->info.stage != MESA_SHADER_VERTEX ||
       !vs->info.io_lowered || tcs->info.stage != MESA_SHADER_TESS_CTRL ||
       !tcs->info.io_lowered || tes->info.stage != MESA_SHADER_TESS_EVAL ||
       !tes->info.io_lowered)
      REJECT("fused tess requires lowered VS, TCS, and TES NIR");

   enum tess_primitive_mode domain =
      tes->info.tess._primitive_mode != TESS_PRIMITIVE_UNSPECIFIED
         ? tes->info.tess._primitive_mode
         : tcs->info.tess._primitive_mode;
   enum gl_tess_spacing spacing =
      tes->info.tess.spacing != TESS_SPACING_UNSPECIFIED
         ? tes->info.tess.spacing
         : tcs->info.tess.spacing;
   if (spacing == TESS_SPACING_UNSPECIFIED)
      spacing = TESS_SPACING_EQUAL;
   bool point_mode = tes->info.tess.point_mode || tcs->info.tess.point_mode;
   bool ccw = tes->info.tess.ccw || tcs->info.tess.ccw;
   unsigned out_v = tcs->info.tess.tcs_vertices_out;

   if (domain != TESS_PRIMITIVE_TRIANGLES && domain != TESS_PRIMITIVE_QUADS)
      REJECT("fused tess supports triangles and quads domains only");
   if (spacing != TESS_SPACING_EQUAL &&
       spacing != TESS_SPACING_FRACTIONAL_ODD &&
       spacing != TESS_SPACING_FRACTIONAL_EVEN)
      REJECT("unsupported tessellation spacing mode");
   if (point_mode)
      REJECT("fused tess does not support point_mode");
   if (!out_v || out_v > 32)
      REJECT("fused tess requires 1..32 TCS output vertices");

   if (tcs->xfb_info || tes->xfb_info || tcs->info.writes_memory ||
       tes->info.writes_memory || tcs->info.outputs_written_16bit ||
       tcs->info.inputs_read_16bit || tes->info.outputs_written_16bit ||
       tes->info.inputs_read_16bit || tcs->info.shared_size ||
       tes->info.shared_size)
      REJECT("fused tess excludes XFB, memory writes, shared memory, and 16-bit IO");

   if (vs->info.float_controls_execution_mode !=
       tcs->info.float_controls_execution_mode)
      REJECT("fused tess requires matching VS and TCS float execution controls");

   unsigned clip_size = 0, cull_size = 0;
   scan_tes_clip_cull_sizes(tes, &clip_size, &cull_size);
   if (clip_size > 8 || cull_size > 8 || clip_size + cull_size > 8)
      REJECT("fused tess supports at most 8 clip plus cull distances");
   if (!(tes->info.outputs_written & VARYING_BIT_POS))
      REJECT("fused tess requires Position output in TES");

   if (!hk_nir_fused_validate_vs(vs, opts ? &opts->vertex : NULL, reason))
      return false;

   /* Validate TCS instructions. */
   unsigned tcs_impls = 0;
   nir_foreach_function_impl(impl, tcs) {
      if (++tcs_impls > 1)
         REJECT("fused tess requires single-entrypoint TCS");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("fused tess requires inlined TCS calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_per_vertex_input: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused tess requires constant TCS input slot offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported TCS per-vertex input load");
               break;
            }
            case nir_intrinsic_store_per_vertex_output: {
               if (!nir_src_is_const(intr->src[2]))
                  REJECT("fused tess requires constant TCS output slot offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[2]);
               if (loc >= VARYING_SLOT_MAX || intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                         intr->src[0].ssa->num_components >
                      4)
                  REJECT("unsupported TCS per-vertex output store");
               break;
            }
            case nir_intrinsic_store_output: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused tess requires constant TCS patch output offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               unsigned max_c = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 2 : 4;
               if ((loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
                    loc != VARYING_SLOT_TESS_LEVEL_INNER &&
                    !(loc >= VARYING_SLOT_PATCH0 &&
                      loc < VARYING_SLOT_PATCH0 + 32)) ||
                   intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                         intr->src[0].ssa->num_components >
                      max_c)
                  REJECT("unsupported TCS patch output store");
               break;
            }
            case nir_intrinsic_load_output: {
               if (!nir_src_is_const(intr->src[0]))
                  REJECT("fused tess requires constant TCS load_output offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[0]);
               unsigned max_c = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 2 : 4;
               if ((loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
                    loc != VARYING_SLOT_TESS_LEVEL_INNER) ||
                   intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components >
                      max_c)
                  REJECT("fused tess does not support non-level TCS output reads");
               break;
            }
            case nir_intrinsic_load_per_vertex_output:
               REJECT("fused tess falls back on cross-invocation TCS output reads");
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT("fused tess requires lowered external TCS dereferences");
               break;
            case nir_intrinsic_barrier:
            case nir_intrinsic_load_invocation_id:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_patch_vertices_in:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_draw_id:
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_push_constant:
            case nir_intrinsic_load_root_agx:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_ro_sink_address_poly:
            case nir_intrinsic_load_constant:
            case nir_intrinsic_image_load_apple9_agx:
               break;
            default:
               REJECT("unsupported TCS intrinsic in fused tess");
            }
         }
      }
   }

   /* Validate TES instructions. */
   unsigned tes_impls = 0;
   nir_foreach_function_impl(impl, tes) {
      if (++tes_impls > 1)
         REJECT("fused tess requires single-entrypoint TES");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("fused tess requires inlined TES calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused tess requires constant TES output slot offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->src[0].ssa->bit_size != 32 ||
                   !tes_output_slot_ok(loc, nir_intrinsic_component(intr),
                                       intr->src[0].ssa->num_components,
                                       clip_size, cull_size))
                  REJECT("unsupported TES output store in fused tess");
               break;
            }
            case nir_intrinsic_load_per_vertex_input: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused tess requires constant TES per-vertex slot offsets");
               if (nir_src_is_const(intr->src[0]) &&
                   nir_src_as_uint(intr->src[0]) >= out_v)
                  REJECT("TES per-vertex input index out of range");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported TES per-vertex input load");
               break;
            }
            case nir_intrinsic_load_input: {
               if (!nir_src_is_const(intr->src[0]))
                  REJECT("fused tess requires constant TES patch input offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[0]);
               unsigned max_c = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 2 : 4;
               if ((loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
                    loc != VARYING_SLOT_TESS_LEVEL_INNER &&
                    !(loc >= VARYING_SLOT_PATCH0 &&
                      loc < VARYING_SLOT_PATCH0 + 32)) ||
                   intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components >
                      max_c)
                  REJECT("unsupported TES patch input load");
               break;
            }
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT("fused tess requires lowered external TES dereferences");
               break;
            case nir_intrinsic_load_tess_coord:
            case nir_intrinsic_load_tess_coord_xy:
            case nir_intrinsic_load_tess_level_outer:
            case nir_intrinsic_load_tess_level_inner:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_patch_vertices_in:
            case nir_intrinsic_load_draw_id:
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_push_constant:
            case nir_intrinsic_load_root_agx:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_ro_sink_address_poly:
            case nir_intrinsic_load_constant:
            case nir_intrinsic_image_load_apple9_agx:
               break;
            default:
               REJECT("unsupported TES intrinsic in fused tess");
            }
         }
      }
   }

   *out_domain = domain;
   *out_spacing = spacing;
   *out_ccw = ccw;
   *out_tcs_verts = out_v;
   return true;
#undef REJECT
}

/* =========================================================================
 * CPU-Side Reference Helpers for Static Constant Tess-Factor Detection
 * ========================================================================= */

static float
cpu_clamp_factor(float f, enum gl_tess_spacing spacing, float lower_override)
{
   float lo = spacing == TESS_SPACING_FRACTIONAL_EVEN ? 2.0f : 1.0f;
   float hi = spacing == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f;
   if (lower_override > lo)
      lo = lower_override;
   if (!(f >= lo))
      f = lo;
   else if (f > hi)
      f = hi;
   if (spacing == TESS_SPACING_EQUAL)
      f = ceilf(f);
   return f;
}

static unsigned
cpu_seg_count(float clamped, enum gl_tess_spacing spacing)
{
   if (spacing == TESS_SPACING_EQUAL)
      return (unsigned)clamped;
   /* Parity-round on the clamped binary32 float before 16.16 quantization so
    * factors immediately above a parity boundary (e.g. 0x40000001 for even or
    * 0x3f800001 for odd) retain their required segment count. Both (clamped - 1)
    * in [0, 62] and * 0.5f are exact in IEEE-754 binary32. */
   if (spacing == TESS_SPACING_FRACTIONAL_ODD)
      return 2u * (unsigned)ceilf((clamped - 1.0f) * 0.5f) + 1u;
   return 2u * (unsigned)ceilf(clamped * 0.5f);
}

static void
cpu_patch_counts(enum tess_primitive_mode domain, enum gl_tess_spacing spacing,
                 const float raw_outer[4], const float raw_inner[2],
                 const struct tess_meshlet_shape *shape, unsigned *out_wgs,
                 unsigned *out_prims)
{
   unsigned n_edges = domain == TESS_PRIMITIVE_QUADS ? 4 : 3;
   for (unsigned e = 0; e < n_edges; ++e) {
      if (!(raw_outer[e] > 0.0f)) {
         *out_wgs = 0;
         *out_prims = 0;
         return;
      }
   }

   float f_out[4] = {1, 1, 1, 1};
   float f_in[2] = {1, 1};
   float maxf = 0.0f;
   for (unsigned e = 0; e < n_edges; ++e) {
      f_out[e] = cpu_clamp_factor(raw_outer[e], spacing, 0.0f);
      if (f_out[e] > maxf)
         maxf = f_out[e];
   }
   unsigned n_inner = domain == TESS_PRIMITIVE_QUADS ? 2 : 1;
   for (unsigned a = 0; a < n_inner; ++a) {
      float raw_i = raw_inner[a];
      if (raw_i > maxf)
         maxf = raw_i;
   }
   float in_lo = (spacing == TESS_SPACING_FRACTIONAL_ODD && maxf > 1.0f)
                    ? HK_TESS_ODD_MIN_PLUS_EPS
                    : 0.0f;
   for (unsigned a = 0; a < n_inner; ++a)
      f_in[a] = cpu_clamp_factor(raw_inner[a], spacing, in_lo);

   unsigned n_out[4] = {1, 1, 1, 1};
   unsigned n_in[2] = {1, 1};
   bool all_ones = spacing != TESS_SPACING_FRACTIONAL_EVEN;
   for (unsigned e = 0; e < n_edges; ++e) {
      n_out[e] = cpu_seg_count(f_out[e], spacing);
      if (n_out[e] != 1)
         all_ones = false;
   }
   for (unsigned a = 0; a < n_inner; ++a) {
      n_in[a] = cpu_seg_count(f_in[a], spacing);
      if (n_in[a] != 1)
         all_ones = false;
   }

   if (all_ones) {
      *out_wgs = 1;
      *out_prims = domain == TESS_PRIMITIVE_QUADS ? 2 : 1;
      return;
   }

   unsigned min_in = spacing == TESS_SPACING_FRACTIONAL_ODD ? 3 : 2;
   for (unsigned a = 0; a < n_inner; ++a) {
      if (n_in[a] < min_in)
         n_in[a] = min_in;
   }

   if (domain == TESS_PRIMITIVE_QUADS) {
      unsigned outer_sum = n_out[0] + n_out[1] + n_out[2] + n_out[3];
      unsigned v_patch = outer_sum + (n_in[0] - 1) * (n_in[1] - 1);
      unsigned p_patch = outer_sum + 2 * (n_in[0] - 2) + 2 * (n_in[1] - 2) +
                         2 * (n_in[0] - 2) * (n_in[1] - 2);
      *out_prims = p_patch;
      if (shape->allow_whole_patch && v_patch <= shape->max_v &&
          p_patch <= shape->max_p) {
         *out_wgs = 1;
         return;
      }
      unsigned w = 0;
      for (unsigned e = 0; e < 4; ++e) {
         unsigned n_is = ((e & 1) ? n_in[0] : n_in[1]) - 2;
         w += DIV_ROUND_UP(n_out[e] + n_is, shape->p_strip);
      }
      unsigned qu = n_in[0] - 2, qv = n_in[1] - 2;
      if (qu && qv)
         w += DIV_ROUND_UP(qu, shape->tile_w) * DIV_ROUND_UP(qv, shape->tile_h);
      *out_wgs = w;
   } else {
      unsigned outer_sum = n_out[0] + n_out[1] + n_out[2];
      unsigned ni = n_in[0];
      unsigned m = (ni - 1) >> 1;
      unsigned even = (ni & 1) == 0 ? 1 : 0;
      unsigned v_patch = outer_sum + 3 * m * (m + even) + even;
      unsigned p_patch =
         outer_sum + 3 * (ni - 2) + ((3 * (ni - 2) * (ni - 2)) >> 1);
      *out_prims = p_patch;
      if (shape->allow_whole_patch && v_patch <= shape->max_v &&
          p_patch <= shape->max_p) {
         *out_wgs = 1;
         return;
      }
      unsigned w = 0;
      for (unsigned e = 0; e < 3; ++e)
         w += DIV_ROUND_UP(n_out[e] + (ni - 2), shape->p_strip);
      unsigned rings = ni >> 1;
      for (unsigned r = 1; r < rings; ++r)
         w += 3 * DIV_ROUND_UP(2 * (ni - 2 * r - 1), shape->p_strip);
      if (ni & 1)
         w += 1;
      *out_wgs = w;
   }
}

static bool
resolve_root_const_offset(nir_def *def, unsigned depth, bool *has_root,
                          uint64_t *out_offset)
{
   if (depth > 32)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      *has_root = false;
      *out_offset = nir_src_as_uint(nir_src_for_ssa(def));
      return true;
   }
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_root_agx ||
          (intr->intrinsic == nir_intrinsic_load_ubo &&
           nir_src_is_const(intr->src[0]) &&
           nir_src_as_uint(intr->src[0]) == 0 &&
           nir_src_is_const(intr->src[1]) &&
           nir_src_as_uint(intr->src[1]) ==
              offsetof(struct hk_root_descriptor_table, root_desc_addr))) {
         *has_root = true;
         *out_offset = 0;
         return true;
      }
      return false;
   }
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op == nir_op_mov || alu->op == nir_op_u2u64)
      return resolve_root_const_offset(alu->src[0].src.ssa, depth + 1, has_root,
                                       out_offset);
   if (alu->op == nir_op_iadd) {
      bool r0 = false, r1 = false;
      uint64_t o0 = 0, o1 = 0;
      if (!resolve_root_const_offset(alu->src[0].src.ssa, depth + 1, &r0, &o0) ||
          !resolve_root_const_offset(alu->src[1].src.ssa, depth + 1, &r1, &o1) ||
          (r0 && r1))
         return false;
      *has_root = r0 || r1;
      *out_offset = (o0 + o1) & BITFIELD64_MASK(def->bit_size);
      return true;
   }
   return false;
}

static bool
resolve_tess_level_scalar(nir_def *def, unsigned comp,
                          struct hk_apple9_tess_level_src *out)
{
   nir_scalar s = nir_scalar_resolved(def, comp);
   if (nir_scalar_is_const(s)) {
      *out = (struct hk_apple9_tess_level_src){
         .kind = HK_APPLE9_TESS_LEVEL_CONST,
         .const_value = nir_scalar_as_float(s),
      };
      return true;
   }

   unsigned extra_byte_off = 0;
   nir_def *ld = s.def;
   unsigned ld_comp = s.comp;
   nir_instr *instr = nir_def_instr(ld);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_unpack_64_2x32_split_x ||
          alu->op == nir_op_unpack_64_2x32_split_y) {
         extra_byte_off = alu->op == nir_op_unpack_64_2x32_split_y ? 4u : 0u;
         nir_scalar s64 =
            nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[ld_comp]);
         ld = s64.def;
         ld_comp = s64.comp;
         instr = nir_def_instr(ld);
      }
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned elem_bytes = ld->bit_size / 8;
   if (elem_bytes != 4 && elem_bytes != 8)
      return false;

   uint64_t push_off = UINT64_MAX;
   const uint64_t push_start = hk_root_descriptor_offset(push);
   const uint64_t push_end = push_start + HK_MAX_PUSH_SIZE;
   if (intr->intrinsic == nir_intrinsic_load_push_constant) {
      if (!nir_src_is_const(intr->src[0]))
         return false;
      push_off = (uint64_t)nir_intrinsic_base(intr) +
                 nir_src_as_uint(intr->src[0]) + (uint64_t)ld_comp * elem_bytes +
                 extra_byte_off;
   } else if (intr->intrinsic == nir_intrinsic_load_global_constant ||
              intr->intrinsic == nir_intrinsic_load_global_constant_offset) {
      bool has_root = false;
      uint64_t root_off = 0;
      if (!resolve_root_const_offset(intr->src[0].ssa, 0, &has_root,
                                     &root_off) ||
          !has_root)
         return false;
      if (intr->intrinsic == nir_intrinsic_load_global_constant_offset) {
         if (!nir_src_is_const(intr->src[1]))
            return false;
         root_off += nir_src_as_uint(intr->src[1]);
      }
      root_off += (uint64_t)ld_comp * elem_bytes + extra_byte_off;
      if (root_off >= push_start && root_off <= push_end - 4)
         push_off = root_off - push_start;
   } else if (intr->intrinsic == nir_intrinsic_load_ubo) {
      if (!nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) != 0 ||
          !nir_src_is_const(intr->src[1]))
         return false;
      uint64_t root_off = (uint64_t)nir_src_as_uint(intr->src[1]) +
                          (uint64_t)ld_comp * elem_bytes + extra_byte_off;
      if (root_off >= push_start && root_off <= push_end - 4)
         push_off = root_off - push_start;
   } else {
      return false;
   }

   if (push_off > HK_MAX_PUSH_SIZE - 4 || (push_off & 3u) != 0)
      return false;

   *out = (struct hk_apple9_tess_level_src){
      .kind = HK_APPLE9_TESS_LEVEL_PUSH,
      .push_offset = (uint16_t)push_off,
   };
   return true;
}

static bool
tess_level_src_equal(const struct hk_apple9_tess_level_src *a,
                     const struct hk_apple9_tess_level_src *b)
{
   if (a->kind != b->kind)
      return false;
   if (a->kind == HK_APPLE9_TESS_LEVEL_CONST)
      return memcmp(&a->const_value, &b->const_value, sizeof(float)) == 0;
   if (a->kind == HK_APPLE9_TESS_LEVEL_PUSH)
      return a->push_offset == b->push_offset;
   return true;
}

/* Classify whether every active tess level written by TCS comes from a single
 * compile-time constant or 32-bit push-constant float. Sets *out_all_const when
 * every active level is compile-time constant (enabling exact compile-time
 * bounds), and returns true whenever all active outer/inner levels are const or
 * push-constant (enabling exact draw-time TVB/ring bounds from push memory). */
static bool
detect_tess_level_sources(const nir_shader *tcs,
                          enum tess_primitive_mode domain,
                          struct hk_apple9_tess_level_src out_levels[6],
                          bool *out_all_const)
{
   const unsigned n_edges = domain == TESS_PRIMITIVE_QUADS ? 4 : 3;
   const unsigned n_inner = domain == TESS_PRIMITIVE_QUADS ? 2 : 1;
   uint8_t outer_seen = 0, inner_seen = 0;
   memset(out_levels, 0, 6 * sizeof(out_levels[0]));
   *out_all_const = false;

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)tcs);
   nir_foreach_block(block, impl) {
      bool in_loop = false;
      for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
         if (n->type == nir_cf_node_loop)
            in_loop = true;
      }
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
             loc != VARYING_SLOT_TESS_LEVEL_INNER)
            continue;
         if (in_loop)
            return false;
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         u_foreach_bit(i, wmask) {
            unsigned dst = comp + i;
            bool is_outer = loc == VARYING_SLOT_TESS_LEVEL_OUTER;
            if ((is_outer && dst >= n_edges) || (!is_outer && dst >= n_inner))
               continue;
            struct hk_apple9_tess_level_src src = {0};
            if (!resolve_tess_level_scalar(intr->src[0].ssa, i, &src))
               return false;
            unsigned slot = is_outer ? dst : 4u + dst;
            uint8_t *seen = is_outer ? &outer_seen : &inner_seen;
            if ((*seen & BITFIELD_BIT(dst)) &&
                !tess_level_src_equal(&out_levels[slot], &src))
               return false;
            out_levels[slot] = src;
            *seen |= BITFIELD_BIT(dst);
         }
      }
   }
   /* Require every active outer level to be written to a single const/push
    * source (unwritten inner defaults to 0.0f -> sound lower bound). */
   if (outer_seen != BITFIELD_MASK(n_edges))
      return false;

   bool all_const = true;
   for (unsigned i = 0; i < 6; ++i) {
      if (out_levels[i].kind == HK_APPLE9_TESS_LEVEL_PUSH)
         all_const = false;
   }
   *out_all_const = all_const;
   return true;
}

/* =========================================================================
 * Conservative Fused VS + TCS NIR Float Interval Upper-Bound Analysis
 *
 * Computes sound IEEE-754 float intervals [lo, hi] (tracking NaN possibility
 * and optional push-constant upper bounds) across VS outputs and TCS
 * tessellation-level stores to tighten static max_child_wgs_per_patch and
 * max_prims_per_patch whenever TCS tess levels are clamped or bounded.
 * ========================================================================= */

struct tess_finterval {
   float lo;
   float hi;
   bool may_nan;
   /* When true: whenever push[push_offset] (f32) is non-NaN at draw time, the
    * runtime value is guaranteed non-NaN and <= max(push[push_offset], 0),
    * with push <= 0 implying val <= 0 (so culled outer / min inner). */
   bool has_push_cap;
   uint16_t push_offset;
};

static inline struct tess_finterval
finterval_top(void)
{
   return (struct tess_finterval){
      .lo = -INFINITY,
      .hi = INFINITY,
      .may_nan = true,
   };
}

static inline float
finterval_ftz_lo(float f)
{
   return (f > 0.0f && f < FLT_MIN) ? 0.0f : f;
}

static inline float
finterval_ftz_hi(float f)
{
   return (f < 0.0f && f > -FLT_MIN) ? 0.0f : f;
}

static inline float
finterval_d_to_f_down(double d)
{
   if (isnan(d))
      return -INFINITY;
   if (d < -(double)FLT_MAX)
      return -INFINITY;
   if (d > (double)FLT_MAX)
      return FLT_MAX;
   float f = (float)d;
   if ((double)f > d)
      f = nextafterf(f, -INFINITY);
   return finterval_ftz_lo(f);
}

static inline float
finterval_d_to_f_up(double d)
{
   if (isnan(d))
      return INFINITY;
   if (d > (double)FLT_MAX)
      return INFINITY;
   if (d < -(double)FLT_MAX)
      return -FLT_MAX;
   float f = (float)d;
   if ((double)f < d)
      f = nextafterf(f, INFINITY);
   return finterval_ftz_hi(f);
}

static inline struct tess_finterval
finterval_const(float c)
{
   if (isnan(c))
      return finterval_top();
   return (struct tess_finterval){
      .lo = finterval_ftz_lo(c),
      .hi = finterval_ftz_hi(c),
      .may_nan = false,
   };
}

static inline bool
finterval_is_finite_const(struct tess_finterval a, float *out)
{
   if (!a.may_nan && isfinite(a.lo) && a.lo == a.hi) {
      *out = a.lo;
      return true;
   }
   return false;
}

static inline struct tess_finterval
finterval_union(struct tess_finterval a, struct tess_finterval b)
{
   struct tess_finterval r = {
      .lo = fminf(a.lo, b.lo),
      .hi = fmaxf(a.hi, b.hi),
      .may_nan = a.may_nan || b.may_nan,
   };
   if (a.has_push_cap && b.has_push_cap && a.push_offset == b.push_offset) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (a.has_push_cap && !b.may_nan && b.hi <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (b.has_push_cap && !a.may_nan && a.hi <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = b.push_offset;
   }
   return r;
}

static inline struct tess_finterval
finterval_neg(struct tess_finterval a)
{
   return (struct tess_finterval){
      .lo = finterval_ftz_lo(-a.hi),
      .hi = finterval_ftz_hi(-a.lo),
      .may_nan = a.may_nan,
   };
}

static inline struct tess_finterval
finterval_abs(struct tess_finterval a)
{
   if (a.lo >= 0.0f)
      return (struct tess_finterval){
         .lo = finterval_ftz_lo(a.lo),
         .hi = finterval_ftz_hi(a.hi),
         .may_nan = a.may_nan,
      };
   if (a.hi <= 0.0f)
      return (struct tess_finterval){
         .lo = finterval_ftz_lo(-a.hi),
         .hi = finterval_ftz_hi(-a.lo),
         .may_nan = a.may_nan,
      };
   return (struct tess_finterval){
      .lo = 0.0f,
      .hi = finterval_ftz_hi(fmaxf(-a.lo, a.hi)),
      .may_nan = a.may_nan,
   };
}

static inline struct tess_finterval
finterval_max(struct tess_finterval a, struct tess_finterval b)
{
   /* NIR nir_op_fmax is IEEE maxNum: fmax(NaN, y) = y, fmax(x, NaN) = x. */
   struct tess_finterval r;
   if (!a.may_nan && !b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fmaxf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(fmaxf(a.hi, b.hi)),
         .may_nan = false,
      };
   } else if (a.may_nan && !b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(b.lo),
         .hi = finterval_ftz_hi(fmaxf(a.hi, b.hi)),
         .may_nan = false,
      };
   } else if (!a.may_nan && b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(a.lo),
         .hi = finterval_ftz_hi(fmaxf(a.hi, b.hi)),
         .may_nan = false,
      };
   } else {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fminf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(fmaxf(a.hi, b.hi)),
         .may_nan = true,
      };
   }
   if (a.has_push_cap && b.has_push_cap && a.push_offset == b.push_offset) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (a.has_push_cap && !b.may_nan && b.hi <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (b.has_push_cap && !a.may_nan && a.hi <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = b.push_offset;
   }
   return r;
}

static inline struct tess_finterval
finterval_min(struct tess_finterval a, struct tess_finterval b)
{
   /* NIR nir_op_fmin is IEEE minNum: fmin(NaN, y) = y, fmin(x, NaN) = x. */
   struct tess_finterval r;
   if (!a.may_nan && !b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fminf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(fminf(a.hi, b.hi)),
         .may_nan = false,
      };
   } else if (a.may_nan && !b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fminf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(b.hi),
         .may_nan = false,
      };
   } else if (!a.may_nan && b.may_nan) {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fminf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(a.hi),
         .may_nan = false,
      };
   } else {
      r = (struct tess_finterval){
         .lo = finterval_ftz_lo(fminf(a.lo, b.lo)),
         .hi = finterval_ftz_hi(fmaxf(a.hi, b.hi)),
         .may_nan = true,
      };
   }
   /* If a's push cap holds whenever push[off] is non-NaN, a is non-NaN in that
    * state, so fmin(a, anything) is non-NaN and <= a. */
   if (a.has_push_cap) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (b.has_push_cap) {
      r.has_push_cap = true;
      r.push_offset = b.push_offset;
   }
   return r;
}

static inline struct tess_finterval
finterval_add(struct tess_finterval a, struct tess_finterval b)
{
   bool may_nan = a.may_nan || b.may_nan ||
                  (a.lo == -INFINITY && b.hi == INFINITY) ||
                  (a.hi == INFINITY && b.lo == -INFINITY);
   float lo;
   if ((a.lo == -INFINITY && b.lo == INFINITY) ||
       (a.lo == INFINITY && b.lo == -INFINITY))
      lo = -INFINITY;
   else if (a.lo == -INFINITY || b.lo == -INFINITY)
      lo = -INFINITY;
   else if (a.lo == INFINITY || b.lo == INFINITY)
      lo = INFINITY;
   else
      lo = finterval_d_to_f_down((double)a.lo + (double)b.lo);

   float hi;
   if ((a.hi == -INFINITY && b.hi == INFINITY) ||
       (a.hi == INFINITY && b.hi == -INFINITY))
      hi = INFINITY;
   else if (a.hi == INFINITY || b.hi == INFINITY)
      hi = INFINITY;
   else if (a.hi == -INFINITY || b.hi == -INFINITY)
      hi = -INFINITY;
   else
      hi = finterval_d_to_f_up((double)a.hi + (double)b.hi);

   struct tess_finterval r = {
      .lo = lo,
      .hi = hi,
      .may_nan = may_nan,
   };
   float c = 0.0f;
   if (a.has_push_cap && finterval_is_finite_const(b, &c) && c <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   } else if (b.has_push_cap && finterval_is_finite_const(a, &c) && c <= 0.0f) {
      r.has_push_cap = true;
      r.push_offset = b.push_offset;
   }
   return r;
}

static inline struct tess_finterval
finterval_square(struct tess_finterval a)
{
   float lo, hi;
   if (a.lo >= 0.0f) {
      lo = isinf(a.lo) ? INFINITY : finterval_d_to_f_down((double)a.lo * (double)a.lo);
      hi = isinf(a.hi) ? INFINITY : finterval_d_to_f_up((double)a.hi * (double)a.hi);
   } else if (a.hi <= 0.0f) {
      lo = isinf(a.hi) ? INFINITY : finterval_d_to_f_down((double)a.hi * (double)a.hi);
      hi = isinf(a.lo) ? INFINITY : finterval_d_to_f_up((double)a.lo * (double)a.lo);
   } else {
      float m = fmaxf(-a.lo, a.hi);
      lo = 0.0f;
      hi = isinf(m) ? INFINITY : finterval_d_to_f_up((double)m * (double)m);
   }
   return (struct tess_finterval){
      .lo = lo,
      .hi = hi,
      .may_nan = a.may_nan,
   };
}

static inline struct tess_finterval
finterval_mul(struct tess_finterval a, struct tess_finterval b)
{
   float c = 0.0f;
   if (finterval_is_finite_const(b, &c)) {
      struct tess_finterval tmp = a;
      a = b;
      b = tmp;
   }
   if (finterval_is_finite_const(a, &c)) {
      if (c == 0.0f) {
         return (struct tess_finterval){
            .lo = 0.0f,
            .hi = 0.0f,
            .may_nan = b.may_nan || isinf(b.lo) || isinf(b.hi),
         };
      }
      struct tess_finterval r;
      if (c > 0.0f) {
         r = (struct tess_finterval){
            .lo = isinf(b.lo) ? b.lo : finterval_d_to_f_down((double)c * (double)b.lo),
            .hi = isinf(b.hi) ? b.hi : finterval_d_to_f_up((double)c * (double)b.hi),
            .may_nan = b.may_nan,
         };
         if (b.has_push_cap && c <= 1.0f) {
            r.has_push_cap = true;
            r.push_offset = b.push_offset;
         }
      } else {
         r = (struct tess_finterval){
            .lo = isinf(b.hi) ? -b.hi : finterval_d_to_f_down((double)c * (double)b.hi),
            .hi = isinf(b.lo) ? -b.lo : finterval_d_to_f_up((double)c * (double)b.lo),
            .may_nan = b.may_nan,
         };
      }
      return r;
   }

   if (isfinite(a.lo) && isfinite(a.hi) && isfinite(b.lo) && isfinite(b.hi)) {
      double p0 = (double)a.lo * (double)b.lo;
      double p1 = (double)a.lo * (double)b.hi;
      double p2 = (double)a.hi * (double)b.lo;
      double p3 = (double)a.hi * (double)b.hi;
      double mn = fmin(fmin(p0, p1), fmin(p2, p3));
      double mx = fmax(fmax(p0, p1), fmax(p2, p3));
      return (struct tess_finterval){
         .lo = finterval_d_to_f_down(mn),
         .hi = finterval_d_to_f_up(mx),
         .may_nan = a.may_nan || b.may_nan,
      };
   }

   bool a_inf = isinf(a.lo) || isinf(a.hi);
   bool b_inf = isinf(b.lo) || isinf(b.hi);
   bool may_nan = a.may_nan || b.may_nan ||
                  (a_inf && b.lo <= 0.0f && b.hi >= 0.0f) ||
                  (b_inf && a.lo <= 0.0f && a.hi >= 0.0f);
   if (a.lo >= 0.0f && b.lo >= 0.0f)
      return (struct tess_finterval){.lo = 0.0f, .hi = INFINITY, .may_nan = may_nan};
   if (a.hi <= 0.0f && b.hi <= 0.0f)
      return (struct tess_finterval){.lo = 0.0f, .hi = INFINITY, .may_nan = may_nan};
   if ((a.lo >= 0.0f && b.hi <= 0.0f) || (a.hi <= 0.0f && b.lo >= 0.0f))
      return (struct tess_finterval){.lo = -INFINITY, .hi = 0.0f, .may_nan = may_nan};
   return (struct tess_finterval){.lo = -INFINITY, .hi = INFINITY, .may_nan = may_nan};
}

/* Float32 ULP at real value |r| per Vulkan SPIR-V precision rules
 * (2^(floor(log2(|r|)) - 23), clamped below at subnormal 2^-149). */
static inline double
finterval_f32_ulp(double r)
{
   double a = fabs(r);
   if (a == 0.0 || !isfinite(a))
      return 0.0;
   int exp = 0;
   frexp(a, &exp); /* 2^(exp-1) <= a < 2^exp */
   int ulp_exp = exp - 24;
   if (ulp_exp < -149)
      ulp_exp = -149;
   return ldexp(1.0, ulp_exp);
}

static inline struct tess_finterval
finterval_sqrt(struct tess_finterval a)
{
   if (a.hi < 0.0f)
      return finterval_top();
   bool may_nan = a.may_nan || a.lo < 0.0f;
   /* Apple9 lowers fsqrt(x) as x * factor(x) where factor approximates
    * 1/sqrt(x) within 2 ULP (Vulkan InverseSqrt). Propagating 2 * x * ulp(1/sqrt(x))
    * <= 2^-22 * sqrt(x) < 4 * ulp(sqrt(x)) plus 0.5 ULP multiply rounding yields
    * < 4.5 ULP; 5.0 * ulp(d) in double before directed rounding encloses all
    * float32 inputs including power-of-two boundaries and Vulkan's 3-ULP Sqrt. */
   float lo;
   if (a.lo <= 0.0f) {
      lo = 0.0f;
   } else if (isinf(a.lo)) {
      lo = INFINITY;
   } else {
      double d = sqrt((double)a.lo);
      double err = 5.0 * finterval_f32_ulp(d);
      lo = fmaxf(0.0f, finterval_d_to_f_down(d - err));
   }
   float hi;
   if (a.hi == 0.0f) {
      hi = 0.0f;
   } else if (isinf(a.hi)) {
      hi = INFINITY;
   } else {
      double d = sqrt((double)a.hi);
      double err = 5.0 * finterval_f32_ulp(d);
      hi = finterval_d_to_f_up(d + err);
   }
   return (struct tess_finterval){.lo = lo, .hi = hi, .may_nan = may_nan};
}

static inline struct tess_finterval
finterval_rsq(struct tess_finterval a)
{
   /* IEEE 1/sqrt(-0.0) = -Inf and 1/sqrt(+0.0) = +Inf; whenever a.lo <= 0.0f,
    * -0.0f is representable in the interval so the lower bound can reach -Inf. */
   if (a.lo <= 0.0f)
      return (struct tess_finterval){
         .lo = -INFINITY,
         .hi = INFINITY,
         .may_nan = a.may_nan || a.lo < 0.0f,
      };
   /* Vulkan InverseSqrt permits 2 ULP defined at the infinitely precise result;
    * compute absolute error in double before directed float32 rounding so
    * power-of-two boundary crossings are enclosed soundly. */
   float lo;
   if (isinf(a.hi)) {
      lo = 0.0f;
   } else {
      double d = 1.0 / sqrt((double)a.hi);
      double err = 2.001 * finterval_f32_ulp(d);
      lo = fmaxf(0.0f, finterval_d_to_f_down(d - err));
   }
   float hi;
   if (isinf(a.lo)) {
      hi = 0.0f;
   } else {
      double d = 1.0 / sqrt((double)a.lo);
      double err = 2.001 * finterval_f32_ulp(d);
      hi = finterval_d_to_f_up(d + err);
   }
   return (struct tess_finterval){.lo = lo, .hi = hi, .may_nan = a.may_nan};
}

static inline struct tess_finterval
finterval_rcp(struct tess_finterval a)
{
   /* Strictly exclude +/-0.0f so 1/a stays finite and sign-preserving.
    * Widen in double by 2.501 * ulp(exact) to cover Vulkan's 2.5-ULP 1/x
    * precision rule across power-of-two boundaries. */
   if (a.lo > 0.0f) {
      float lo;
      if (isinf(a.hi)) {
         lo = 0.0f;
      } else {
         double d = 1.0 / (double)a.hi;
         lo = fmaxf(0.0f, finterval_d_to_f_down(d - 2.501 * finterval_f32_ulp(d)));
      }
      float hi;
      if (isinf(a.lo)) {
         hi = 0.0f;
      } else {
         double d = 1.0 / (double)a.lo;
         hi = finterval_d_to_f_up(d + 2.501 * finterval_f32_ulp(d));
      }
      return (struct tess_finterval){.lo = lo, .hi = hi, .may_nan = a.may_nan};
   }
   if (a.hi < 0.0f) {
      float lo;
      if (isinf(a.hi)) {
         lo = 0.0f;
      } else {
         double d = 1.0 / (double)a.hi;
         lo = finterval_d_to_f_down(d - 2.501 * finterval_f32_ulp(d));
      }
      float hi;
      if (isinf(a.lo)) {
         hi = 0.0f;
      } else {
         double d = 1.0 / (double)a.lo;
         hi = fminf(0.0f, finterval_d_to_f_up(d + 2.501 * finterval_f32_ulp(d)));
      }
      return (struct tess_finterval){.lo = lo, .hi = hi, .may_nan = a.may_nan};
   }
   return finterval_top();
}

static inline struct tess_finterval
finterval_div(struct tess_finterval a, struct tess_finterval b)
{
   /* AGX enables lower_fdiv (a / b -> a * frcp(b)), so bound via a * rcp(b)
    * with outward-widened rcp(b), which conservatively encloses both exact
    * IEEE division and reciprocal-multiply lowering. */
   struct tess_finterval rcp = finterval_rcp(b);
   if (rcp.lo == -INFINITY && rcp.hi == INFINITY)
      return finterval_top();
   struct tess_finterval r = finterval_mul(a, rcp);
   if (a.has_push_cap && !rcp.may_nan && rcp.lo > 0.0f && rcp.hi <= 1.0f) {
      r.has_push_cap = true;
      r.push_offset = a.push_offset;
   }
   return r;
}

/* IEEE-754 binary32 round-to-nearest-even implemented purely on bit patterns
 * so interval analysis never depends on the calling thread's fegetround() mode. */
static float
finterval_round_even(float x)
{
   uint32_t bits;
   memcpy(&bits, &x, sizeof(bits));
   uint32_t sign = bits & 0x80000000u;
   int exp = (int)((bits >> 23) & 0xffu) - 127;
   if (exp >= 23)
      return x;
   if (exp < 0) {
      uint32_t abs_bits = bits & 0x7fffffffu;
      uint32_t res = abs_bits > 0x3f000000u ? (sign | 0x3f800000u) : sign;
      float out;
      memcpy(&out, &res, sizeof(out));
      return out;
   }
   uint32_t frac_bits = 23u - (uint32_t)exp;
   uint32_t frac_mask = (1u << frac_bits) - 1u;
   uint32_t half = 1u << (frac_bits - 1u);
   uint32_t frac = bits & frac_mask;
   uint32_t trunc_bits = bits & ~frac_mask;
   if (frac > half || (frac == half && ((trunc_bits >> frac_bits) & 1u)))
      trunc_bits += (1u << frac_bits);
   float out;
   memcpy(&out, &trunc_bits, sizeof(out));
   return out;
}

struct tess_interval_ctx {
   struct tess_finterval *memo;
   uint8_t *state; /* 0 = unvisited, 1 = visiting, 2 = done */
   unsigned num_defs;
   const struct tess_finterval (*vs_outs)[4]; /* NULL when evaluating VS */
   const struct tess_finterval *tcs_levels;   /* Optional pass-1 TCS levels[6] */
};

static struct tess_finterval
eval_finterval(struct tess_interval_ctx *ctx, nir_def *def, unsigned comp,
               unsigned depth)
{
   nir_scalar s = nir_scalar_resolved(def, comp);
   if (nir_scalar_is_const(s))
      return s.def->bit_size == 32 ? finterval_const(nir_scalar_as_float(s))
                                   : finterval_top();
   if (s.def->bit_size != 32 || s.comp >= 4 || s.def->index >= ctx->num_defs ||
       depth > 64)
      return finterval_top();

   unsigned idx = s.def->index * 4u + s.comp;
   if (ctx->state[idx] == 2)
      return ctx->memo[idx];
   if (ctx->state[idx] == 1)
      return finterval_top(); /* Cycle / loop-carried value. */

   ctx->state[idx] = 1;
   struct tess_finterval res = finterval_top();
   nir_instr *instr = nir_def_instr(s.def);

   if (instr->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      bool in_loop = false;
      for (nir_cf_node *n = instr->block->cf_node.parent; n; n = n->parent) {
         if (n->type == nir_cf_node_loop) {
            in_loop = true;
            break;
         }
      }
      if (!in_loop && !exec_list_is_empty(&phi->srcs)) {
         bool first = true;
         nir_foreach_phi_src(psrc, phi) {
            struct tess_finterval iv =
               eval_finterval(ctx, psrc->src.ssa, s.comp, depth + 1);
            res = first ? iv : finterval_union(res, iv);
            first = false;
         }
      }
   } else if (instr->type == nir_instr_type_intrinsic) {
      struct hk_apple9_tess_level_src psrc = {0};
      if (resolve_tess_level_scalar(s.def, s.comp, &psrc) &&
          psrc.kind == HK_APPLE9_TESS_LEVEL_PUSH) {
         res = (struct tess_finterval){
            .lo = -INFINITY,
            .hi = INFINITY,
            .may_nan = true,
            .has_push_cap = true,
            .push_offset = psrc.push_offset,
         };
      } else {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_per_vertex_input &&
             ctx->vs_outs && nir_src_is_const(intr->src[1])) {
            uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned c = nir_intrinsic_component(intr) + s.comp;
            if (loc < VARYING_SLOT_MAX && c < 4)
               res = ctx->vs_outs[loc][c];
         } else if (intr->intrinsic == nir_intrinsic_load_output &&
                    ctx->tcs_levels && nir_src_is_const(intr->src[0])) {
            uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[0]);
            unsigned c = nir_intrinsic_component(intr) + s.comp;
            if (loc == VARYING_SLOT_TESS_LEVEL_OUTER && c < 4)
               res = ctx->tcs_levels[c];
            else if (loc == VARYING_SLOT_TESS_LEVEL_INNER && c < 2)
               res = ctx->tcs_levels[4u + c];
         }
      }
   } else if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_mov:
         res = eval_finterval(ctx, alu->src[0].src.ssa,
                              alu->src[0].swizzle[s.comp], depth + 1);
         break;
      case nir_op_fneg:
         res = finterval_neg(eval_finterval(ctx, alu->src[0].src.ssa,
                                            alu->src[0].swizzle[s.comp],
                                            depth + 1));
         break;
      case nir_op_fabs:
         res = finterval_abs(eval_finterval(ctx, alu->src[0].src.ssa,
                                            alu->src[0].swizzle[s.comp],
                                            depth + 1));
         break;
      case nir_op_fsat: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         res = (struct tess_finterval){
            .lo = a.may_nan ? 0.0f : finterval_ftz_lo(fminf(fmaxf(a.lo, 0.0f), 1.0f)),
            .hi = finterval_ftz_hi(fminf(fmaxf(a.hi, 0.0f), 1.0f)),
            .may_nan = a.may_nan,
         };
         break;
      }
      case nir_op_fmin: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         struct tess_finterval b =
            eval_finterval(ctx, alu->src[1].src.ssa,
                           alu->src[1].swizzle[s.comp], depth + 1);
         res = finterval_min(a, b);
         break;
      }
      case nir_op_fmax: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         struct tess_finterval b =
            eval_finterval(ctx, alu->src[1].src.ssa,
                           alu->src[1].swizzle[s.comp], depth + 1);
         res = finterval_max(a, b);
         break;
      }
      case nir_op_fadd: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         struct tess_finterval b =
            eval_finterval(ctx, alu->src[1].src.ssa,
                           alu->src[1].swizzle[s.comp], depth + 1);
         res = finterval_add(a, b);
         break;
      }
      case nir_op_fsub: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         struct tess_finterval b =
            eval_finterval(ctx, alu->src[1].src.ssa,
                           alu->src[1].swizzle[s.comp], depth + 1);
         res = finterval_add(a, finterval_neg(b));
         break;
      }
      case nir_op_fmul: {
         nir_scalar sa = nir_scalar_resolved(alu->src[0].src.ssa,
                                             alu->src[0].swizzle[s.comp]);
         nir_scalar sb = nir_scalar_resolved(alu->src[1].src.ssa,
                                             alu->src[1].swizzle[s.comp]);
         struct tess_finterval a = eval_finterval(ctx, sa.def, sa.comp, depth + 1);
         if (nir_scalar_equal(sa, sb))
            res = finterval_square(a);
         else
            res = finterval_mul(a, eval_finterval(ctx, sb.def, sb.comp, depth + 1));
         break;
      }
      case nir_op_ffma: {
         nir_scalar sa = nir_scalar_resolved(alu->src[0].src.ssa,
                                             alu->src[0].swizzle[s.comp]);
         nir_scalar sb = nir_scalar_resolved(alu->src[1].src.ssa,
                                             alu->src[1].swizzle[s.comp]);
         struct tess_finterval a = eval_finterval(ctx, sa.def, sa.comp, depth + 1);
         struct tess_finterval prod =
            nir_scalar_equal(sa, sb)
               ? finterval_square(a)
               : finterval_mul(a, eval_finterval(ctx, sb.def, sb.comp, depth + 1));
         struct tess_finterval c =
            eval_finterval(ctx, alu->src[2].src.ssa,
                           alu->src[2].swizzle[s.comp], depth + 1);
         res = finterval_add(prod, c);
         break;
      }
      case nir_op_fdot2:
      case nir_op_fdot3:
      case nir_op_fdot4: {
         unsigned n = alu->op == nir_op_fdot2 ? 2u
                      : (alu->op == nir_op_fdot3 ? 3u : 4u);
         struct tess_finterval acc = finterval_const(0.0f);
         for (unsigned i = 0; i < n; ++i) {
            nir_scalar sa =
               nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[i]);
            nir_scalar sb =
               nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[i]);
            struct tess_finterval a = eval_finterval(ctx, sa.def, sa.comp, depth + 1);
            struct tess_finterval prod =
               nir_scalar_equal(sa, sb)
                  ? finterval_square(a)
                  : finterval_mul(a, eval_finterval(ctx, sb.def, sb.comp, depth + 1));
            acc = i == 0 ? prod : finterval_add(acc, prod);
         }
         res = acc;
         break;
      }
      case nir_op_fsqrt:
         res = finterval_sqrt(eval_finterval(ctx, alu->src[0].src.ssa,
                                             alu->src[0].swizzle[s.comp],
                                             depth + 1));
         break;
      case nir_op_frsq:
         res = finterval_rsq(eval_finterval(ctx, alu->src[0].src.ssa,
                                            alu->src[0].swizzle[s.comp],
                                            depth + 1));
         break;
      case nir_op_fdiv: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         struct tess_finterval b =
            eval_finterval(ctx, alu->src[1].src.ssa,
                           alu->src[1].swizzle[s.comp], depth + 1);
         res = finterval_div(a, b);
         break;
      }
      case nir_op_frcp: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         res = finterval_rcp(a);
         break;
      }
      case nir_op_ftrunc:
      case nir_op_ffloor:
      case nir_op_fceil:
      case nir_op_fround_even: {
         struct tess_finterval a =
            eval_finterval(ctx, alu->src[0].src.ssa,
                           alu->src[0].swizzle[s.comp], depth + 1);
         float lo = a.lo, hi = a.hi;
         if (alu->op == nir_op_ftrunc) {
            lo = truncf(lo);
            hi = truncf(hi);
         } else if (alu->op == nir_op_ffloor) {
            lo = floorf(lo);
            hi = floorf(hi);
         } else if (alu->op == nir_op_fceil) {
            lo = ceilf(lo);
            hi = ceilf(hi);
         } else {
            lo = finterval_round_even(lo);
            hi = finterval_round_even(hi);
         }
         res = (struct tess_finterval){
            .lo = lo,
            .hi = hi,
            .may_nan = a.may_nan,
         };
         if (a.has_push_cap &&
             (alu->op == nir_op_ftrunc || alu->op == nir_op_ffloor)) {
            res.has_push_cap = true;
            res.push_offset = a.push_offset;
         }
         break;
      }
      case nir_op_bcsel: {
         nir_scalar cond = nir_scalar_resolved(alu->src[0].src.ssa,
                                               alu->src[0].swizzle[s.comp]);
         if (nir_scalar_is_const(cond)) {
            unsigned take = nir_scalar_as_bool(cond) ? 1u : 2u;
            res = eval_finterval(ctx, alu->src[take].src.ssa,
                                 alu->src[take].swizzle[s.comp], depth + 1);
         } else {
            struct tess_finterval t =
               eval_finterval(ctx, alu->src[1].src.ssa,
                              alu->src[1].swizzle[s.comp], depth + 1);
            struct tess_finterval f =
               eval_finterval(ctx, alu->src[2].src.ssa,
                              alu->src[2].swizzle[s.comp], depth + 1);
            res = finterval_union(t, f);
         }
         break;
      }
      case nir_op_b2f32:
         res = (struct tess_finterval){.lo = 0.0f, .hi = 1.0f, .may_nan = false};
         break;
      case nir_op_u2f32: {
         unsigned bits = alu->src[0].src.ssa->bit_size;
         float hi = bits == 1 ? 1.0f
                    : (bits == 8 ? 255.0f
                       : (bits == 16 ? 65535.0f
                          : (bits == 32 ? 4294967296.0f : INFINITY)));
         res = (struct tess_finterval){.lo = 0.0f, .hi = hi, .may_nan = false};
         break;
      }
      case nir_op_i2f32: {
         unsigned bits = alu->src[0].src.ssa->bit_size;
         float lo = bits == 1 ? -1.0f
                    : (bits == 8 ? -128.0f
                       : (bits == 16 ? -32768.0f
                          : (bits == 32 ? -2147483648.0f : -INFINITY)));
         float hi = bits == 1 ? 0.0f
                    : (bits == 8 ? 127.0f
                       : (bits == 16 ? 32767.0f
                          : (bits == 32 ? 2147483648.0f : INFINITY)));
         res = (struct tess_finterval){.lo = lo, .hi = hi, .may_nan = false};
         break;
      }
      default:
         break;
      }
   }

   ctx->memo[idx] = res;
   ctx->state[idx] = 2;
   return res;
}

static void
compute_vs_output_intervals(nir_shader *vs,
                            struct tess_finterval out[VARYING_SLOT_MAX][4])
{
   for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc)
      for (unsigned c = 0; c < 4; ++c)
         out[loc][c] = finterval_top();

   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   nir_index_ssa_defs(impl);
   unsigned n_defs = impl->ssa_alloc;
   if (!n_defs)
      return;

   struct tess_finterval *memo = calloc((size_t)n_defs * 4u, sizeof(*memo));
   uint8_t *state = calloc((size_t)n_defs * 4u, sizeof(*state));
   if (!memo || !state) {
      free(memo);
      free(state);
      return;
   }

   struct tess_interval_ctx ctx = {
      .memo = memo,
      .state = state,
      .num_defs = n_defs,
   };

   struct tess_finterval acc[VARYING_SLOT_MAX][4];
   uint8_t seen[VARYING_SLOT_MAX] = {0};
   uint8_t uncond[VARYING_SLOT_MAX] = {0};
   bool indirect_store = false;

   nir_foreach_block(block, impl) {
      bool is_top_level = block->cf_node.parent->type == nir_cf_node_function;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         if (!nir_src_is_const(intr->src[1])) {
            indirect_store = true;
            break;
         }
         uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc >= VARYING_SLOT_MAX)
            continue;
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c >= 4)
               continue;
            struct tess_finterval iv =
               eval_finterval(&ctx, intr->src[0].ssa, i, 0);
            if (!(seen[loc] & BITFIELD_BIT(c))) {
               acc[loc][c] = iv;
               seen[loc] |= BITFIELD_BIT(c);
            } else {
               acc[loc][c] = finterval_union(acc[loc][c], iv);
            }
            if (is_top_level)
               uncond[loc] |= BITFIELD_BIT(c);
         }
      }
      if (indirect_store)
         break;
   }

   if (!indirect_store) {
      for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
         for (unsigned c = 0; c < 4; ++c) {
            if (uncond[loc] & BITFIELD_BIT(c))
               out[loc][c] = acc[loc][c];
         }
      }
   }

   free(memo);
   free(state);
}

/* Analyze fused VS + TCS NIR to compute sound per-component upper bounds on
 * gl_TessLevelOuter[0..3] and gl_TessLevelInner[0..1], plus optional
 * draw-time push-constant caps. */
static void
analyze_tess_level_bounds(nir_shader *vs, nir_shader *tcs,
                          enum tess_primitive_mode domain,
                          enum gl_tess_spacing spacing,
                          struct hk_apple9_tess_level_src out_bounds[6],
                          bool *out_has_push)
{
   const float max_f = spacing == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f;
   const unsigned n_edges = domain == TESS_PRIMITIVE_QUADS ? 4u : 3u;
   const unsigned n_inner = domain == TESS_PRIMITIVE_QUADS ? 2u : 1u;
   *out_has_push = false;
   for (unsigned s = 0; s < 6; ++s) {
      out_bounds[s] = (struct hk_apple9_tess_level_src){
         .kind = HK_APPLE9_TESS_LEVEL_DYNAMIC,
         .const_value = max_f,
      };
   }

   struct tess_finterval (*vs_outs)[4] =
      calloc(VARYING_SLOT_MAX, sizeof(*vs_outs));
   if (!vs_outs)
      return;
   compute_vs_output_intervals(vs, vs_outs);

   nir_function_impl *impl = nir_shader_get_entrypoint(tcs);
   nir_index_ssa_defs(impl);
   unsigned n_defs = impl->ssa_alloc;
   if (!n_defs) {
      free(vs_outs);
      return;
   }

   struct tess_finterval *memo = calloc((size_t)n_defs * 4u, sizeof(*memo));
   uint8_t *state = calloc((size_t)n_defs * 4u, sizeof(*state));
   if (!memo || !state) {
      free(memo);
      free(state);
      free(vs_outs);
      return;
   }

   struct tess_finterval slot_iv[6];
   struct tess_finterval prev_iv[6];
   bool has_level_load = false;
   bool indirect_store = false;

   for (unsigned pass = 0; pass < 2; ++pass) {
      memset(state, 0, (size_t)n_defs * 4u * sizeof(*state));
      struct tess_interval_ctx ctx = {
         .memo = memo,
         .state = state,
         .num_defs = n_defs,
         .vs_outs = vs_outs,
         .tcs_levels = pass == 0 ? NULL : prev_iv,
      };
      /* build_object_shader zero-initializes shared tess_levels[0..5]. */
      for (unsigned s = 0; s < 6; ++s)
         slot_iv[s] = finterval_const(0.0f);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_output &&
                nir_src_is_const(intr->src[0])) {
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[0]);
               if (loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
                   loc == VARYING_SLOT_TESS_LEVEL_INNER)
                  has_level_load = true;
               continue;
            }
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            if (!nir_src_is_const(intr->src[1])) {
               indirect_store = true;
               break;
            }
            uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            if (loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
                loc != VARYING_SLOT_TESS_LEVEL_INNER)
               continue;
            bool is_outer = loc == VARYING_SLOT_TESS_LEVEL_OUTER;
            unsigned comp = nir_intrinsic_component(intr);
            unsigned wmask = nir_intrinsic_write_mask(intr);
            u_foreach_bit(i, wmask) {
               unsigned dst = comp + i;
               /* Track all 6 addressable TCS tess-level slots (Outer[0..3],
                * Inner[0..1]) across both passes so cross-invocation reads of
                * domain-inactive slots (e.g. Outer[3] on triangles) never
                * spuriously evaluate to 0.0f. Domain projection happens below
                * when deriving final generator bounds. */
               if ((is_outer && dst >= 4u) || (!is_outer && dst >= 2u))
                  continue;
               unsigned slot = is_outer ? dst : 4u + dst;
               struct tess_finterval iv =
                  eval_finterval(&ctx, intr->src[0].ssa, i, 0);
               slot_iv[slot] = finterval_union(slot_iv[slot], iv);
            }
         }
         if (indirect_store)
            break;
      }
      if (indirect_store || !has_level_load)
         break;
      memcpy(prev_iv, slot_iv, sizeof(prev_iv));
   }

   if (!indirect_store) {
      for (unsigned s = 0; s < 6; ++s) {
         bool active = s < 4 ? (s < n_edges) : (s - 4u < n_inner);
         if (!active)
            continue;
         float hi = slot_iv[s].hi;
         if (isnan(hi) || hi > max_f)
            hi = max_f;
         else if (hi < 0.0f)
            hi = 0.0f;
         if (slot_iv[s].has_push_cap) {
            out_bounds[s] = (struct hk_apple9_tess_level_src){
               .kind = HK_APPLE9_TESS_LEVEL_PUSH_BOUND,
               .push_offset = slot_iv[s].push_offset,
               .const_value = hi,
            };
            *out_has_push = true;
         } else {
            out_bounds[s] = (struct hk_apple9_tess_level_src){
               .kind = HK_APPLE9_TESS_LEVEL_DYNAMIC,
               .const_value = hi,
            };
         }
      }
   }

   free(memo);
   free(state);
   free(vs_outs);
}

void
hk_nir_tess_fused_eval_counts(const struct hk_shader_info *info,
                              const uint8_t *push_data,
                              uint32_t *out_max_child_wgs,
                              uint32_t *out_max_patch_prims)
{
   const struct hk_apple9_tess_push_levels *pl = &info->apple9_tess_push_levels;
   if (!pl->valid || !push_data ||
       (pl->domain != TESS_PRIMITIVE_QUADS &&
        pl->domain != TESS_PRIMITIVE_TRIANGLES) ||
       (pl->spacing != TESS_SPACING_EQUAL &&
        pl->spacing != TESS_SPACING_FRACTIONAL_ODD &&
        pl->spacing != TESS_SPACING_FRACTIONAL_EVEN) ||
       !pl->shape_p_strip || !pl->shape_tile_w || !pl->shape_tile_h) {
      *out_max_child_wgs = info->apple9_tess_max_child_wgs;
      *out_max_patch_prims = info->apple9_tess_max_patch_prims;
      return;
   }

   float raw_outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   float raw_inner[2] = {0.0f, 0.0f};
   for (unsigned i = 0; i < 6; ++i) {
      float v = 0.0f;
      if (pl->levels[i].kind == HK_APPLE9_TESS_LEVEL_CONST ||
          pl->levels[i].kind == HK_APPLE9_TESS_LEVEL_DYNAMIC) {
         v = pl->levels[i].const_value;
      } else if (pl->levels[i].kind == HK_APPLE9_TESS_LEVEL_PUSH ||
                 pl->levels[i].kind == HK_APPLE9_TESS_LEVEL_PUSH_BOUND) {
         if (pl->levels[i].push_offset > HK_MAX_PUSH_SIZE - sizeof(float)) {
            *out_max_child_wgs = info->apple9_tess_max_child_wgs;
            *out_max_patch_prims = info->apple9_tess_max_patch_prims;
            return;
         }
         memcpy(&v, push_data + pl->levels[i].push_offset, sizeof(float));
         if (pl->levels[i].kind == HK_APPLE9_TESS_LEVEL_PUSH_BOUND)
            v = isnan(v) ? pl->levels[i].const_value
                         : fminf(v, pl->levels[i].const_value);
      }
      if (i < 4)
         raw_outer[i] = v;
      else
         raw_inner[i - 4] = v;
   }

   const struct tess_meshlet_shape shape = {
      .max_v = pl->shape_max_v,
      .max_p = pl->shape_max_p,
      .p_strip = pl->shape_p_strip,
      .tile_w = pl->shape_tile_w,
      .tile_h = pl->shape_tile_h,
      .allow_whole_patch = pl->shape_allow_whole,
   };
   unsigned wgs = 0, prims = 0;
   cpu_patch_counts((enum tess_primitive_mode)pl->domain,
                    (enum gl_tess_spacing)pl->spacing, raw_outer, raw_inner,
                    &shape, &wgs, &prims);
   *out_max_child_wgs =
      info->apple9_tess_max_child_wgs ? MIN2(wgs, info->apple9_tess_max_child_wgs)
                                      : wgs;
   *out_max_patch_prims =
      info->apple9_tess_max_patch_prims ? MIN2(prims, info->apple9_tess_max_patch_prims)
                                        : prims;
}

/* =========================================================================
 * NIR Tessellation Math Helpers (Shared by OBJECT and MESH)
 * ========================================================================= */

/* 16.16 fixed-point constants matching poly_tessellator.h bit-for-bit so every
 * generated coordinate matches the fallback SW tessellator (satisfying Vulkan
 * tessellation invariance rules 2, 3, 4, and 8). */
#define HK_TESS_FXP_ONE        65536u
#define HK_TESS_FXP_ONE_HALF   32768u
#define HK_TESS_FXP_ONE_THIRD  0x00005555u
#define HK_TESS_FXP_TWO_THIRDS 0x0000aaaau

static nir_def *
nir_fxp_to_f32(nir_builder *b, nir_def *fxp)
{
   return nir_fmul_imm(b, nir_u2f32(b, fxp), 1.0f / 65536.0f);
}

static nir_def *
nir_float_to_fxp(nir_builder *b, nir_def *val)
{
   return nir_f2u32(
      b, nir_ffma(b, val, nir_imm_float(b, 65536.0f), nir_imm_float(b, 0.5f)));
}

static nir_def *
nir_remove_msb(nir_builder *b, nir_def *val)
{
   nir_def *msb = nir_ufind_msb(b, val);
   nir_def *bit = nir_ishl(b, nir_imm_int(b, 1), msb);
   return nir_bcsel(b, nir_ieq_imm(b, val, 0), nir_imm_int(b, 0),
                    nir_iand(b, val, nir_inot(b, bit)));
}

/* Exact 16.16 fixed-point 1D point placement matching ComputeTessFactorCtx +
 * PlacePointIn1D in src/poly/cl/tessellator.h bit-for-bit for all 3 spacings.
 * k in [0 .. n]. Returns integer q in [0 .. 65536]. */
static nir_def *
nir_eval_1d_fxp(nir_builder *b, enum gl_tess_spacing spacing, nir_def *n,
                nir_def *f, nir_def *k)
{
   nir_def *k2 = nir_ishl_imm(b, k, 1);
   nir_def *flip = nir_ult(b, n, k2);
   nir_def *j = nir_bcsel(b, flip, nir_isub(b, n, k), k);
   nir_def *j_is_0 = nir_ieq_imm(b, j, 0);
   nir_def *j_is_mid = nir_ieq(b, nir_ishl_imm(b, j, 1), n);

   nir_def *q_mid;
   if (spacing == TESS_SPACING_EQUAL) {
      nir_def *inv_n = nir_float_to_fxp(
         b, nir_fdiv(b, nir_imm_float(b, 1.0f), nir_u2f32(b, n)));
      q_mid = nir_imul(b, j, inv_n);
   } else {
      const bool odd = spacing == TESS_SPACING_FRACTIONAL_ODD;
      nir_def *fxp_tess = nir_float_to_fxp(b, f);
      nir_def *fxp_half = nir_ushr_imm(b, nir_iadd_imm(b, fxp_tess, 1), 1);
      if (odd) {
         fxp_half = nir_iadd_imm(b, fxp_half, HK_TESS_FXP_ONE_HALF);
      } else {
         fxp_half = nir_bcsel(
            b, nir_ieq_imm(b, fxp_half, HK_TESS_FXP_ONE_HALF),
            nir_imm_int(b, HK_TESS_FXP_ONE), fxp_half);
      }
      nir_def *ceil_int = nir_umax(
         b, nir_ushr_imm(b, nir_iadd_imm(b, n, odd ? 1 : 0), 1),
         nir_imm_int(b, 1));
      nir_def *fxp_ceil = nir_ishl_imm(b, ceil_int, 16);
      nir_def *below_ceil = nir_ult(b, fxp_half, fxp_ceil);
      nir_def *floor_int = nir_bcsel(
         b, below_ceil, nir_iadd_imm(b, ceil_int, -1), ceil_int);
      nir_def *fxp_floor = nir_ishl_imm(b, floor_int, 16);
      nir_def *frac = nir_bcsel(
         b, below_ceil, nir_isub(b, fxp_half, fxp_floor), nir_imm_int(b, 0));

      nir_def *split_raw;
      if (odd) {
         nir_def *rm = nir_remove_msb(b, nir_usub_sat(b, floor_int, nir_imm_int(b, 1)));
         split_raw = nir_bcsel(b, nir_ieq_imm(b, floor_int, 1),
                               nir_imm_int(b, 0),
                               nir_iadd_imm(b, nir_ishl_imm(b, rm, 1), 1));
      } else {
         nir_def *rm = nir_remove_msb(b, floor_int);
         split_raw = nir_iadd_imm(b, nir_ishl_imm(b, rm, 1), 1);
      }
      nir_def *split = nir_bcsel(b, nir_ieq(b, fxp_ceil, fxp_floor),
                                 nir_iadd_imm(b, ceil_int, 1), split_raw);

      nir_def *floor_segs = nir_umax(
         b, nir_usub_sat(b, nir_ishl_imm(b, floor_int, 1), nir_imm_int(b, odd ? 1 : 0)),
         nir_imm_int(b, 1));
      nir_def *ceil_segs = nir_umax(
         b, nir_usub_sat(b, nir_ishl_imm(b, ceil_int, 1), nir_imm_int(b, odd ? 1 : 0)),
         nir_imm_int(b, 1));
      nir_def *inv_floor = nir_float_to_fxp(
         b, nir_fdiv(b, nir_imm_float(b, 1.0f), nir_u2f32(b, floor_segs)));
      nir_def *inv_ceil = nir_float_to_fxp(
         b, nir_fdiv(b, nir_imm_float(b, 1.0f), nir_u2f32(b, ceil_segs)));

      nir_def *idx_floor = nir_bcsel(b, nir_ult(b, split, j),
                                     nir_iadd_imm(b, j, -1), j);
      nir_def *loc_floor = nir_imul(b, idx_floor, inv_floor);
      nir_def *loc_ceil = nir_imul(b, j, inv_ceil);
      nir_def *inv_frac = nir_isub(b, nir_imm_int(b, HK_TESS_FXP_ONE), frac);
      nir_def *lerp = nir_iadd(b, nir_imul(b, loc_floor, inv_frac),
                               nir_imul(b, loc_ceil, frac));
      q_mid = nir_ushr_imm(b, nir_iadd_imm(b, lerp, HK_TESS_FXP_ONE_HALF), 16);
   }

   nir_def *q_half = nir_bcsel(
      b, j_is_0, nir_imm_int(b, 0),
      nir_bcsel(b, j_is_mid, nir_imm_int(b, HK_TESS_FXP_ONE_HALF),
                nir_umin(b, q_mid, nir_imm_int(b, HK_TESS_FXP_ONE_HALF))));
   return nir_bcsel(b, flip,
                    nir_isub(b, nir_imm_int(b, HK_TESS_FXP_ONE), q_half),
                    q_half);
}

static nir_def *
nir_eval_1d(nir_builder *b, enum gl_tess_spacing spacing, nir_def *n,
            nir_def *f, nir_def *k)
{
   return nir_fxp_to_f32(b, nir_eval_1d_fxp(b, spacing, n, f, k));
}

/* Exact bitmask of positions in finalPointPositionTable(2..32) with bound < H
 * (H in [0..32]). Matches tessellator.h::StitchTransition bit-for-bit. */
static nir_def *
nir_stitch_mask(nir_builder *b, nir_def *h)
{
   nir_def *k = nir_usub_sat(b, h, nir_imm_int(b, 1)); /* 0..31 */
   nir_def *b_msb = nir_ufind_msb(b, k);               /* 0..4 for k >= 1 */
   nir_def *r = nir_isub(b, k, nir_ishl(b, nir_imm_int(b, 1), b_msb));
   nir_def *is_b0 = nir_ieq_imm(b, b_msb, 0);
   nir_def *is_b1 = nir_ieq_imm(b, b_msb, 1);
   nir_def *is_b2 = nir_ieq_imm(b, b_msb, 2);
   nir_def *is_b3 = nir_ieq_imm(b, b_msb, 3);
   nir_def *base = nir_bcsel(
      b, is_b0, nir_imm_int(b, 0x00000000u),
      nir_bcsel(b, is_b1, nir_imm_int(b, 0x00008000u),
                nir_bcsel(b, is_b2, nir_imm_int(b, 0x00808080u),
                          nir_bcsel(b, is_b3, nir_imm_int(b, 0x08888888u),
                                    nir_imm_int(b, 0x2aaaaaaau)))));
   nir_def *lvl = nir_bcsel(
      b, is_b0, nir_imm_int(b, 0x00008000u),
      nir_bcsel(b, is_b1, nir_imm_int(b, 0x00800080u),
                nir_bcsel(b, is_b2, nir_imm_int(b, 0x08080808u),
                          nir_bcsel(b, is_b3, nir_imm_int(b, 0x22222222u),
                                    nir_imm_int(b, 0x55555555u)))));
   nir_def *stride = nir_ushr(b, nir_imm_int(b, 32), b_msb);
   nir_def *bits = nir_imul(b, nir_iadd_imm(b, r, 1), stride);
   nir_def *cut = nir_bcsel(
      b, nir_uge_imm(b, bits, 32), nir_imm_int(b, 0xffffffffu),
      nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bits), -1));
   return nir_bcsel(b, nir_ieq_imm(b, k, 0), nir_imm_int(b, 0),
                    nir_ior(b, base, nir_iand(b, lvl, cut)));
}

static nir_def *
nir_stitch_i_half(nir_builder *b, nir_def *h_o, nir_def *h_i, nir_def *t)
{
   nir_def *m_o = nir_stitch_mask(b, h_o);
   nir_def *m_i = nir_stitch_mask(b, h_i);
   nir_def *o_init = nir_b2i32(b, nir_ine_imm(b, h_o, 0));
   nir_def *rem = nir_usub_sat(b, t, o_init);
   nir_def *pos = nir_imm_int(b, 0);
   const unsigned steps[5] = {16, 8, 4, 2, 1};
   for (unsigned s = 0; s < 5; ++s) {
      nir_def *cand = nir_iadd_imm(b, pos, steps[s]);
      nir_def *lo = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), cand), -1);
      nir_def *c = nir_iadd(b, nir_bit_count(b, nir_iand(b, m_i, lo)),
                            nir_bit_count(b, nir_iand(b, m_o, lo)));
      pos = nir_bcsel(b, nir_uge(b, rem, c), cand, pos);
   }
   nir_def *lo = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), pos), -1);
   nir_def *ci = nir_bit_count(b, nir_iand(b, m_i, lo));
   nir_def *co = nir_bit_count(b, nir_iand(b, m_o, lo));
   nir_def *extra = nir_bcsel(
      b, nir_ult(b, nir_iadd(b, ci, co), rem),
      nir_iand_imm(b, nir_ushr(b, m_i, pos), 1), nir_imm_int(b, 0));
   return nir_bcsel(b, nir_uge(b, o_init, t), nir_imm_int(b, 0),
                    nir_iadd(b, ci, extra));
}

/* Exact strip stitcher matching tessellator.h::StitchTransition (and
 * StitchRegular DIAGONALS_MIRRORED when n_out == n_in_seg + 2) bit-for-bit
 * across all (n_out, n_in_seg) in [1..64] x [0..62]: maps step t to inner
 * vertex index i(t) in [0..n_in_seg] with i(t+1)-i(t) in {0, 1}. */
static nir_def *
nir_stitch_i(nir_builder *b, nir_def *n_out, nir_def *n_in_seg, nir_def *t)
{
   nir_def *total = nir_iadd(b, n_out, n_in_seg);
   nir_def *h_o = nir_ushr_imm(b, n_out, 1);
   nir_def *k = nir_ushr_imm(b, n_in_seg, 1);
   nir_def *h_i = nir_iadd_imm(b, k, 1);
   nir_def *half_t = nir_iadd(b, h_o, k);
   nir_def *in_first = nir_uge(b, half_t, t);
   nir_def *rt = nir_usub_sat(b, total, t);
   nir_def *in_second = nir_uge(b, half_t, rt);
   nir_def *t_query = nir_bcsel(b, in_first, t, rt);
   nir_def *q = nir_stitch_i_half(b, h_o, h_i, t_query);
   return nir_bcsel(b, in_first, q,
                    nir_bcsel(b, in_second, nir_isub(b, n_in_seg, q),
                              nir_iadd_imm(b, k, 1)));
}

static nir_def *
nir_clamp_tess_factor(nir_builder *b, nir_def *raw, enum gl_tess_spacing spacing,
                      nir_def *lo_override)
{
   nir_def *lo = nir_imm_float(
      b, spacing == TESS_SPACING_FRACTIONAL_EVEN ? 2.0f : 1.0f);
   if (lo_override)
      lo = nir_fmax(b, lo, lo_override);
   nir_def *hi = nir_imm_float(
      b, spacing == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f);
   /* Map NaN deterministically to lo: !(raw >= lo) catches NaN and < lo. */
   nir_def *ge_lo = nir_fge(b, raw, lo);
   nir_def *clamped = nir_bcsel(b, ge_lo, nir_fmin(b, raw, hi), lo);
   if (spacing == TESS_SPACING_EQUAL)
      clamped = nir_fceil(b, clamped);
   return clamped;
}

static nir_def *
nir_seg_count_from_factor(nir_builder *b, nir_def *clamped,
                          enum gl_tess_spacing spacing)
{
   if (spacing == TESS_SPACING_EQUAL)
      return nir_f2u32(b, clamped);
   /* Parity-round on the clamped binary32 float before 16.16 quantization so
    * factors immediately above a parity boundary retain their required segment
    * count, matching tessellator.h::NumSegmentsForTessFactorFloat bit-for-bit. */
   if (spacing == TESS_SPACING_FRACTIONAL_ODD) {
      nir_def *half = nir_fmul_imm(b, nir_fadd_imm(b, clamped, -1.0f), 0.5f);
      return nir_iadd_imm(b, nir_ishl_imm(b, nir_f2u32(b, nir_fceil(b, half)), 1),
                          1);
   }
   nir_def *half = nir_fmul_imm(b, clamped, 0.5f);
   return nir_ishl_imm(b, nir_f2u32(b, nir_fceil(b, half)), 1);
}

static nir_def *
nir_div_round_up_imm(nir_builder *b, nir_def *num, unsigned div)
{
   if (div == 1)
      return num;
   return nir_udiv_imm(b, nir_iadd_imm(b, num, div - 1), div);
}

/* =========================================================================
 * Payload Layout Calculation (Tight Component Packing of Live TCS Outputs)
 * ========================================================================= */

struct tess_payload_layout {
   unsigned out_v;
   uint8_t pv_mask[VARYING_SLOT_MAX];
   uint16_t pv_off[VARYING_SLOT_MAX][4]; /* byte offset within vertex record */
   unsigned pv_stride_B;
   uint8_t pc_mask[32];
   uint16_t pc_off[32][4]; /* byte offset from start of payload */
   bool has_raw_levels;
   unsigned raw_levels_off;
   unsigned payload_B;
};

static void
compute_payload_layout(const nir_shader *tes, unsigned out_v,
                       struct tess_payload_layout *pl)
{
   memset(pl, 0, sizeof(*pl));
   pl->out_v = out_v;

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)tes);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_per_vertex_input) {
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned comp = nir_intrinsic_component(intr);
            pl->pv_mask[loc] |= BITFIELD_MASK(intr->def.num_components) << comp;
         } else if (intr->intrinsic == nir_intrinsic_load_input) {
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[0]);
            if (loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
                loc == VARYING_SLOT_TESS_LEVEL_INNER) {
               pl->has_raw_levels = true;
            } else if (loc >= VARYING_SLOT_PATCH0 &&
                       loc < VARYING_SLOT_PATCH0 + 32) {
               unsigned comp = nir_intrinsic_component(intr);
               pl->pc_mask[loc - VARYING_SLOT_PATCH0] |=
                  BITFIELD_MASK(intr->def.num_components) << comp;
            }
         } else if (intr->intrinsic == nir_intrinsic_load_tess_level_outer ||
                    intr->intrinsic == nir_intrinsic_load_tess_level_inner) {
            pl->has_raw_levels = true;
         }
      }
   }

   unsigned pv_off = 0;
   for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
      u_foreach_bit(c, pl->pv_mask[loc]) {
         pl->pv_off[loc][c] = pv_off;
         pv_off += 4;
      }
   }
   pl->pv_stride_B = pv_off;

   unsigned off = HK_TESS_PAYLOAD_HEADER_B + out_v * pl->pv_stride_B;
   for (unsigned p = 0; p < 32; ++p) {
      u_foreach_bit(c, pl->pc_mask[p]) {
         pl->pc_off[p][c] = off;
         off += 4;
      }
   }
   if (pl->has_raw_levels) {
      pl->raw_levels_off = off;
      off += 24;
   }
   pl->payload_B = ALIGN_POT(off, 16);
}

/* =========================================================================
 * Part 1: Synthesize OBJECT(VS + TCS) NIR Shader
 * ========================================================================= */

struct tcs_lower_state {
   const struct tess_payload_layout *pl;
   const uint8_t *vs_read_mask;
   const uint16_t (*vs_shm_off)[4];
   unsigned vs_stride_B;
   nir_def *lane;
   nir_def *patch_id;
   nir_def *instance_id;
   nir_def *base_instance;
   nir_def *input_patch_size;
};

static bool
lower_tcs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct tcs_lower_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def,
                      s->pl->out_v == 1 ? nir_imm_int(b, 0) : s->lane);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, s->patch_id);
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, s->instance_id);
      return true;
   case nir_intrinsic_load_base_instance:
      nir_def_replace(&intr->def, s->base_instance);
      return true;
   case nir_intrinsic_load_patch_vertices_in:
      nir_def_replace(&intr->def, s->input_patch_size);
      return true;
   case nir_intrinsic_load_per_vertex_input: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned num = intr->def.num_components;
      nir_def *vidx = nir_umin(b, intr->src[0].ssa, nir_imm_int(b, 31));
      nir_def *vbase = nir_iadd_imm(
         b, nir_imul_imm(b, vidx, s->vs_stride_B), HK_TESS_SHM_LEVELS_B);
      nir_def *comps[4];
      for (unsigned i = 0; i < num; ++i) {
         nir_def *off = nir_iadd_imm(b, vbase, s->vs_shm_off[loc][comp + i]);
         comps[i] = nir_load_shared(b, 1, 32, off, .align_mul = 4);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, num));
      return true;
   }
   case nir_intrinsic_store_per_vertex_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[2]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned wmask = nir_intrinsic_write_mask(intr);
      if (loc < VARYING_SLOT_MAX && s->pl->pv_mask[loc]) {
         nir_def *vidx =
            nir_umin(b, intr->src[1].ssa, nir_imm_int(b, s->pl->out_v - 1));
         nir_def *vbase = nir_iadd_imm(
            b, nir_imul_imm(b, vidx, s->pl->pv_stride_B),
            HK_TESS_PAYLOAD_HEADER_B);
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (!(s->pl->pv_mask[loc] & BITFIELD_BIT(c)))
               continue;
            nir_def *off = nir_iadd_imm(b, vbase, s->pl->pv_off[loc][c]);
            nir_store_task_payload(b, nir_channel(b, intr->src[0].ssa, i), off,
                                   .align_mul = 4, .write_mask = 1);
         }
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_store_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned wmask = nir_intrinsic_write_mask(intr);
      if (loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
          loc == VARYING_SLOT_TESS_LEVEL_INNER) {
         unsigned base_idx = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 4 : 0;
         unsigned max_c = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 2 : 4;
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c >= max_c)
               continue;
            nir_store_shared(b, nir_channel(b, intr->src[0].ssa, i),
                             nir_imm_int(b, (base_idx + c) * 4), .align_mul = 4,
                             .write_mask = 1);
         }
      } else if (loc >= VARYING_SLOT_PATCH0 && loc < VARYING_SLOT_PATCH0 + 32) {
         unsigned p = loc - VARYING_SLOT_PATCH0;
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (!(s->pl->pc_mask[p] & BITFIELD_BIT(c)))
               continue;
            nir_store_task_payload(b, nir_channel(b, intr->src[0].ssa, i),
                                   nir_imm_int(b, s->pl->pc_off[p][c]),
                                   .align_mul = 4, .write_mask = 1);
         }
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_load_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[0]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned num = intr->def.num_components;
      unsigned base_idx = loc == VARYING_SLOT_TESS_LEVEL_INNER ? 4 : 0;
      nir_def *comps[4];
      for (unsigned i = 0; i < num; ++i) {
         comps[i] = nir_load_shared(
            b, 1, 32, nir_imm_int(b, (base_idx + comp + i) * 4), .align_mul = 4);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, num));
      return true;
   }
   case nir_intrinsic_barrier: {
      /* Translate TCS output memory class to every backing storage class used
       * by lowered TCS outputs (shared tess_levels + task payload backed by
       * global memory). Preserve memory-only barriers and original semantics
       * and scope rather than dropping them. */
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (modes & nir_var_shader_out) {
         modes = (modes & ~nir_var_shader_out) | nir_var_mem_shared |
                 nir_var_mem_task_payload;
         nir_intrinsic_set_memory_modes(intr, modes);
      }
      if (nir_intrinsic_execution_scope(intr) != SCOPE_NONE) {
         nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
         if (modes && nir_intrinsic_memory_scope(intr) < SCOPE_WORKGROUP)
            nir_intrinsic_set_memory_scope(intr, SCOPE_WORKGROUP);
      }
      return true;
   }
   default:
      return false;
   }
}

struct tcs_split_barrier {
   mesa_scope memory_scope;
   nir_memory_semantics memory_semantics;
   nir_variable_mode memory_modes;
};

static void
emit_tcs_split_barrier(nir_builder *b, const struct tcs_split_barrier *sb)
{
   nir_variable_mode modes = sb->memory_modes;
   if (modes & nir_var_shader_out) {
      modes = (modes & ~nir_var_shader_out) | nir_var_mem_shared |
              nir_var_mem_task_payload;
   }
   mesa_scope mem_scope = sb->memory_scope;
   if (modes && mem_scope < SCOPE_WORKGROUP)
      mem_scope = SCOPE_WORKGROUP;
   nir_memory_semantics sem = sb->memory_semantics;
   if (modes && !sem)
      sem = NIR_MEMORY_ACQ_REL;
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = mem_scope,
               .memory_semantics = sem,
               .memory_modes = modes);
}

static nir_shader *
build_object_shader(nir_shader *vs, nir_shader *tcs,
                    enum tess_primitive_mode domain,
                    enum gl_tess_spacing spacing,
                    const struct tess_payload_layout *pl,
                    const struct tess_meshlet_shape *shape,
                    const char **reason)
{
   /* Scan VS outputs actually read by TCS. */
   uint8_t vs_read_mask[VARYING_SLOT_MAX] = {0};
   bool vs_any_read = false;
   nir_function_impl *tcs_impl = nir_shader_get_entrypoint(tcs);
   nir_foreach_block(block, tcs_impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_per_vertex_input) {
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned comp = nir_intrinsic_component(intr);
            vs_read_mask[loc] |= BITFIELD_MASK(intr->def.num_components) << comp;
            vs_any_read = true;
         }
      }
   }

   uint16_t vs_shm_off[VARYING_SLOT_MAX][4];
   memset(vs_shm_off, 0, sizeof(vs_shm_off));
   unsigned vs_stride_B = 0;
   for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
      u_foreach_bit(c, vs_read_mask[loc]) {
         vs_shm_off[loc][c] = vs_stride_B;
         vs_stride_B += 4;
      }
   }

   unsigned shared_bytes =
      ALIGN_POT(HK_TESS_SHM_LEVELS_B + 32u * vs_stride_B, 16);
   if (shared_bytes > AGX_APPLE9_MAX_SHARED_BYTES - HK_APPLE9_NATIVE_GRID_ROOT - 16) {
      if (reason)
         *reason = "fused tess VS->TCS shared staging exceeds 32 KiB";
      return NULL;
   }

   /* Prepare VS template for single-instance inlining into OBJECT. */
   nir_variable *vs_slot_vars[VARYING_SLOT_MAX] = {NULL};
   struct hk_nir_fused_vs_lower vs_lower = {
      .vs_read_mask = vs_read_mask,
      .vs_slot_vars = vs_slot_vars,
   };
   if (vs_any_read) {
      vs_lower.in_vertex_id =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "ft_vid");
      vs_lower.in_vertex_pos =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "ft_vpos");
      vs_lower.in_instance_id =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "ft_iid");
      vs_lower.in_first_vertex =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "ft_fv");
      vs_lower.in_base_instance =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "ft_bi");
      for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
         if (vs_read_mask[loc])
            vs_slot_vars[loc] = nir_variable_create(
               vs, nir_var_shader_temp, glsl_vec4_type(), "ft_vs_slot");
      }
      NIR_PASS(_, vs, nir_shader_intrinsics_pass, hk_nir_fused_lower_vs_intr,
               nir_metadata_none, &vs_lower);
      NIR_PASS(_, vs, nir_opt_dce);
      NIR_PASS(_, vs, nir_opt_dead_cf);
   }

   /* Build OBJECT shader in-place on `tcs` so TCS temps/resources stay intact. */
   nir_shader *obj = tcs;
   nir_function_impl *impl = nir_shader_get_entrypoint(obj);

   /* Collect top-level TCS control barriers before rewriting CF. */
   struct util_dynarray barriers = UTIL_DYNARRAY_INIT;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (is_control_barrier(instr))
            util_dynarray_append(&barriers, instr);
      }
   }

   /* Extract TCS segments split at top-level control barriers so barriers stay
    * at uniform top-level CF across all 32 physical lanes, preserving each
    * original barrier's memory modes, semantics, and scope. */
   unsigned num_bars = util_dynarray_num_elements(&barriers, nir_instr *);
   unsigned num_segs = num_bars + 1;
   nir_cf_list *tcs_segs = calloc(num_segs, sizeof(*tcs_segs));
   struct tcs_split_barrier *bar_info =
      num_bars ? calloc(num_bars, sizeof(*bar_info)) : NULL;
   if (!tcs_segs || (num_bars && !bar_info)) {
      free(tcs_segs);
      free(bar_info);
      util_dynarray_fini(&barriers);
      if (reason)
         *reason = "out of memory splitting TCS segments";
      return NULL;
   }
   nir_cursor cur = nir_before_impl(impl);
   for (unsigned s = 0; s < num_bars; ++s) {
      nir_instr *bar = *util_dynarray_element(&barriers, nir_instr *, s);
      nir_intrinsic_instr *bi = nir_instr_as_intrinsic(bar);
      bar_info[s] = (struct tcs_split_barrier){
         .memory_scope = nir_intrinsic_memory_scope(bi),
         .memory_semantics = nir_intrinsic_memory_semantics(bi),
         .memory_modes = nir_intrinsic_memory_modes(bi),
      };
      nir_cf_extract(&tcs_segs[s], cur, nir_before_instr(bar));
      cur = nir_after_instr(bar);
   }
   nir_cf_extract(&tcs_segs[num_segs - 1], cur, nir_after_impl(impl));
   /* Remove the original top-level barrier instructions (we re-emit them). */
   util_dynarray_foreach(&barriers, nir_instr *, bar)
      nir_instr_remove(*bar);
   util_dynarray_fini(&barriers);

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *wg = nir_load_workgroup_id(&b);
   nir_def *tp = nir_load_tess_param_buffer_poly(&b);

   nir_def *index_buffer = hk_nir_fused_param_u64(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, index_buffer));
   nir_def *index_buffer_size = hk_nir_fused_param_u64(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, index_buffer_size));
   nir_def *index_size_B = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, index_size_B));
   nir_def *input_patch_size = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, input_patch_size));
   nir_def *first_vertex = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, first_vertex));
   nir_def *base_instance = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, base_instance));
   nir_def *patches_per_instance = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, patches_per_instance));
   nir_def *first_patch = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, first_patch));
   nir_def *groups_x = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, groups_x));
   nir_def *first_instance_offset = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, first_instance_offset));
   nir_def *flip_winding = hk_nir_fused_param_u32(
      &b, tp, offsetof(struct hk_apple9_fused_tess_params, flip_winding));

   nir_def *patch_linear =
      nir_iadd(&b, nir_imul(&b, nir_channel(&b, wg, 1), groups_x),
               nir_channel(&b, wg, 0));
   nir_def *patch_in_bounds = nir_ult(&b, patch_linear, patches_per_instance);
   nir_def *patch_id = nir_iadd(&b, first_patch, patch_linear);
   nir_def *instance_id =
      nir_iadd(&b, nir_channel(&b, wg, 2), first_instance_offset);

   /* Zero-init shared tess_levels[0..5] in parallel across lanes 0..5. */
   nir_push_if(&b, nir_ult_imm(&b, lane, 6));
   nir_store_shared(&b, nir_imm_float(&b, 0.0f), nir_ishl_imm(&b, lane, 2),
                    .align_mul = 4, .write_mask = 1);
   nir_pop_if(&b, NULL);

   /* Zero-init non-header payload bytes so conditionally written or unwritten
    * TCS outputs never leak stale hardware payload memory to MESH(TES). */
   if (pl->payload_B > HK_TESS_PAYLOAD_HEADER_B) {
      unsigned body_B = pl->payload_B - HK_TESS_PAYLOAD_HEADER_B;
      unsigned iters = DIV_ROUND_UP(body_B, 32u * 4u);
      nir_push_if(&b, patch_in_bounds);
      for (unsigned it = 0; it < iters; ++it) {
         nir_def *off =
            nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 2),
                         HK_TESS_PAYLOAD_HEADER_B + it * 128u);
         nir_push_if(&b, nir_ult_imm(&b, off, pl->payload_B));
         nir_store_task_payload(&b, nir_imm_int(&b, 0), off, .align_mul = 4,
                                .write_mask = 1);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }

   /* Phase 1: Parallel VS execution across lanes 0 .. input_patch_size - 1. */
   if (vs_any_read) {
      nir_variable *obj_vid =
         nir_variable_create(obj, nir_var_shader_temp, glsl_uint_type(), "ft_o_vid");
      nir_variable *obj_vpos =
         nir_variable_create(obj, nir_var_shader_temp, glsl_uint_type(), "ft_o_vpos");
      nir_variable *obj_iid =
         nir_variable_create(obj, nir_var_shader_temp, glsl_uint_type(), "ft_o_iid");
      nir_variable *obj_fv =
         nir_variable_create(obj, nir_var_shader_temp, glsl_uint_type(), "ft_o_fv");
      nir_variable *obj_bi =
         nir_variable_create(obj, nir_var_shader_temp, glsl_uint_type(), "ft_o_bi");
      nir_variable *obj_vs_out[VARYING_SLOT_MAX] = {NULL};
      for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
         if (vs_read_mask[loc])
            obj_vs_out[loc] = nir_variable_create(
               obj, nir_var_shader_temp, glsl_vec4_type(), "ft_o_vs_out");
      }

      nir_if *vs_active = nir_push_if(
         &b, nir_iand(&b, patch_in_bounds, nir_ult(&b, lane, input_patch_size)));
      vs_active->control = nir_selection_control_dont_flatten;
      {
         for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
            if (!obj_vs_out[loc])
               continue;
            nir_def *def = loc == VARYING_SLOT_POS
                              ? nir_imm_vec4(&b, 0, 0, 0, 1)
                              : nir_imm_vec4(&b, 0, 0, 0, 0);
            nir_store_var(&b, obj_vs_out[loc], def, 15);
         }
         nir_def *vpos =
            nir_iadd(&b, nir_imul(&b, patch_id, input_patch_size), lane);
         nir_def *vid_zb = hk_nir_fused_fetch_vertex_id_zero_base(
            &b, vpos, index_buffer, index_buffer_size, index_size_B);
         nir_def *vid = nir_iadd(&b, vid_zb, first_vertex);
         nir_store_var(&b, obj_vpos, vid_zb, 1);
         nir_store_var(&b, obj_vid, vid, 1);
         nir_store_var(&b, obj_iid, instance_id, 1);
         nir_store_var(&b, obj_fv, first_vertex, 1);
         nir_store_var(&b, obj_bi, base_instance, 1);

         struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
         _mesa_hash_table_insert(remap, vs_lower.in_vertex_id, obj_vid);
         _mesa_hash_table_insert(remap, vs_lower.in_vertex_pos, obj_vpos);
         _mesa_hash_table_insert(remap, vs_lower.in_instance_id, obj_iid);
         _mesa_hash_table_insert(remap, vs_lower.in_first_vertex, obj_fv);
         _mesa_hash_table_insert(remap, vs_lower.in_base_instance, obj_bi);
         for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
            if (vs_slot_vars[loc])
               _mesa_hash_table_insert(remap, vs_slot_vars[loc], obj_vs_out[loc]);
         }
         nir_inline_function_impl(&b, nir_shader_get_entrypoint(vs), NULL, remap);
         _mesa_hash_table_destroy(remap, NULL);

         /* Store live VS outputs to threadgroup shared memory for TCS. */
         nir_def *vbase = nir_iadd_imm(
            &b, nir_imul_imm(&b, lane, vs_stride_B), HK_TESS_SHM_LEVELS_B);
         for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
            if (!obj_vs_out[loc])
               continue;
            nir_def *val = nir_load_var(&b, obj_vs_out[loc]);
            u_foreach_bit(c, vs_read_mask[loc]) {
               nir_store_shared(&b, nir_channel(&b, val, c),
                                nir_iadd_imm(&b, vbase, vs_shm_off[loc][c]),
                                .align_mul = 4, .write_mask = 1);
            }
         }
      }
      nir_pop_if(&b, vs_active);
   }

   /* Top-level barrier: VS outputs, tess_levels, and payload zero-init visible
    * to TCS before any TCS invocation writes to task payload. */
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared | nir_var_mem_task_payload);

   /* Phase 2: Parallel TCS execution across lanes 0 .. out_v - 1, keeping
    * top-level barriers outside the active-lane guard. */
   nir_def *tcs_cond =
      nir_iand(&b, patch_in_bounds, nir_ult_imm(&b, lane, pl->out_v));
   for (unsigned s = 0; s < num_segs; ++s) {
      nir_if *tcs_if = nir_push_if(&b, tcs_cond);
      tcs_if->control = nir_selection_control_dont_flatten;
      b.cursor = nir_cf_reinsert(&tcs_segs[s], b.cursor);
      nir_pop_if(&b, tcs_if);
      if (s + 1 < num_segs)
         emit_tcs_split_barrier(&b, &bar_info[s]);
   }
   free(tcs_segs);
   free(bar_info);
   /* Hoist load_const and undef out of guarded TCS segments into the entry
    * block so nir_repair_ssa_impl never wraps constant IO slot offsets in
    * phi(const, undef) across top-level barrier splits. */
   nir_block *start_block = nir_start_block(impl);
   nir_foreach_block(block, impl) {
      if (block == start_block)
         continue;
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_load_const ||
             instr->type == nir_instr_type_undef)
            nir_instr_move(nir_before_block(start_block), instr);
      }
   }
   /* Repair any SSA defs defined in an earlier TCS segment and used after a
    * top-level control barrier before running nir_shader_intrinsics_pass. */
   nir_progress(true, impl, nir_metadata_none);
   nir_repair_ssa_impl(impl);

   /* Lower TCS intrinsics across the reinserted TCS segments. */
   struct tcs_lower_state tcs_state = {
      .pl = pl,
      .vs_read_mask = vs_read_mask,
      .vs_shm_off = vs_shm_off,
      .vs_stride_B = vs_stride_B,
      .lane = lane,
      .patch_id = patch_id,
      .instance_id = instance_id,
      .base_instance = base_instance,
      .input_patch_size = input_patch_size,
   };
   NIR_PASS(_, obj, nir_shader_intrinsics_pass, lower_tcs_intr,
            nir_metadata_none, &tcs_state);

   /* Final barrier: TCS tess-level and payload stores visible before lane 0
    * computes counts, writes the 40 B header, and launches child mesh WGs. */
   b.cursor = nir_after_impl(impl);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared | nir_var_mem_task_payload);

   /* Phase 3: Lane 0 culls/clamps tess factors, writes 40 B header (+ optional
    * raw levels) to GET_SR 0xe0/0xe1 task payload, and launches exact child
    * MESH workgroup count. */
   nir_variable *launch_x =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_launch_x");
   nir_store_var(&b, launch_x, nir_imm_int(&b, 0), 1);

   nir_if *lane0_if = nir_push_if(
      &b, nir_iand(&b, patch_in_bounds, nir_ieq_imm(&b, lane, 0)));
   lane0_if->control = nir_selection_control_dont_flatten;
   {
      const unsigned n_edges = domain == TESS_PRIMITIVE_QUADS ? 4 : 3;
      const unsigned n_inner = domain == TESS_PRIMITIVE_QUADS ? 2 : 1;
      nir_def *raw_out[4], *raw_in[2];
      for (unsigned e = 0; e < 4; ++e) {
         raw_out[e] = nir_load_shared(&b, 1, 32, nir_imm_int(&b, e * 4),
                                      .align_mul = 4);
      }
      for (unsigned a = 0; a < 2; ++a) {
         raw_in[a] = nir_load_shared(&b, 1, 32, nir_imm_int(&b, (4 + a) * 4),
                                     .align_mul = 4);
      }

      /* Patch culling: any active outer <= 0 or NaN -> launch 0 child WGs. */
      nir_def *alive = nir_imm_true(&b);
      for (unsigned e = 0; e < n_edges; ++e)
         alive = nir_iand(&b, alive, nir_flt(&b, nir_imm_float(&b, 0.0f), raw_out[e]));

      nir_if *alive_if = nir_push_if(&b, alive);
      alive_if->control = nir_selection_control_dont_flatten;
      {
         nir_def *f_out[4], *f_in[2];
         nir_def *maxf = nir_imm_float(&b, 0.0f);
         for (unsigned e = 0; e < n_edges; ++e) {
            f_out[e] = nir_clamp_tess_factor(&b, raw_out[e], spacing, NULL);
            maxf = nir_fmax(&b, maxf, f_out[e]);
         }
         for (unsigned e = n_edges; e < 4; ++e)
            f_out[e] = nir_imm_float(&b, 1.0f);
         for (unsigned a = 0; a < n_inner; ++a) {
            /* Guard against NaN in raw_in[a] propagating into maxf. */
            nir_def *valid_in = nir_fge(&b, raw_in[a], nir_imm_float(&b, 0.0f));
            maxf = nir_bcsel(&b, valid_in, nir_fmax(&b, maxf, raw_in[a]), maxf);
         }
         nir_def *in_lo = NULL;
         if (spacing == TESS_SPACING_FRACTIONAL_ODD) {
            in_lo = nir_bcsel(
               &b, nir_flt(&b, nir_imm_float(&b, 1.0f), maxf),
               nir_imm_float(&b, HK_TESS_ODD_MIN_PLUS_EPS),
               nir_imm_float(&b, 1.0f));
         }
         for (unsigned a = 0; a < n_inner; ++a)
            f_in[a] = nir_clamp_tess_factor(&b, raw_in[a], spacing, in_lo);
         for (unsigned a = n_inner; a < 2; ++a)
            f_in[a] = nir_imm_float(&b, 1.0f);

         nir_def *n_out[4], *n_in[2];
         nir_def *all_ones =
            nir_imm_bool(&b, spacing != TESS_SPACING_FRACTIONAL_EVEN);
         for (unsigned e = 0; e < n_edges; ++e) {
            n_out[e] = nir_seg_count_from_factor(&b, f_out[e], spacing);
            all_ones = nir_iand(&b, all_ones, nir_ieq_imm(&b, n_out[e], 1));
         }
         for (unsigned e = n_edges; e < 4; ++e)
            n_out[e] = nir_imm_int(&b, 1);
         for (unsigned a = 0; a < n_inner; ++a) {
            n_in[a] = nir_seg_count_from_factor(&b, f_in[a], spacing);
            all_ones = nir_iand(&b, all_ones, nir_ieq_imm(&b, n_in[a], 1));
         }
         for (unsigned a = n_inner; a < 2; ++a)
            n_in[a] = nir_imm_int(&b, 1);

         /* Non-all-ones promotes inner segment count to >= 2 (Even/Equal) or
          * >= 3 (FractionalOdd). */
         unsigned min_in_seg = spacing == TESS_SPACING_FRACTIONAL_ODD ? 3 : 2;
         for (unsigned a = 0; a < n_inner; ++a) {
            nir_def *promoted =
               nir_umax(&b, n_in[a], nir_imm_int(&b, min_in_seg));
            n_in[a] = nir_bcsel(&b, all_ones, nir_imm_int(&b, 1), promoted);
            if (spacing == TESS_SPACING_EQUAL)
               f_in[a] = nir_u2f32(&b, n_in[a]);
         }

         nir_def *whole_patch = nir_imm_false(&b);
         nir_def *w_prefix = nir_imm_int(&b, 0);
         nir_def *child_wgs = nir_imm_int(&b, 1);

         if (domain == TESS_PRIMITIVE_QUADS) {
            nir_def *outer_sum = nir_iadd(
               &b, nir_iadd(&b, n_out[0], n_out[1]),
               nir_iadd(&b, n_out[2], n_out[3]));
            nir_def *qu = nir_usub_sat(&b, n_in[0], nir_imm_int(&b, 2));
            nir_def *qv = nir_usub_sat(&b, n_in[1], nir_imm_int(&b, 2));
            if (shape->allow_whole_patch) {
               nir_def *v_patch = nir_iadd(
                  &b, outer_sum,
                  nir_imul(&b, nir_usub_sat(&b, n_in[0], nir_imm_int(&b, 1)),
                           nir_usub_sat(&b, n_in[1], nir_imm_int(&b, 1))));
               nir_def *p_patch = nir_iadd(
                  &b,
                  nir_iadd(&b, outer_sum,
                           nir_ishl_imm(&b, nir_iadd(&b, qu, qv), 1)),
                  nir_ishl_imm(&b, nir_imul(&b, qu, qv), 1));
               whole_patch = nir_iand(
                  &b, nir_inot(&b, all_ones),
                  nir_iand(&b, nir_ule_imm(&b, v_patch, shape->max_v),
                           nir_ule_imm(&b, p_patch, shape->max_p)));
            }
            nir_def *w0 =
               nir_div_round_up_imm(&b, nir_iadd(&b, n_out[0], qv), shape->p_strip);
            nir_def *w1 =
               nir_div_round_up_imm(&b, nir_iadd(&b, n_out[1], qu), shape->p_strip);
            nir_def *w2 =
               nir_div_round_up_imm(&b, nir_iadd(&b, n_out[2], qv), shape->p_strip);
            nir_def *w3 =
               nir_div_round_up_imm(&b, nir_iadd(&b, n_out[3], qu), shape->p_strip);
            nir_def *wp0 = w0;
            nir_def *wp1 = nir_iadd(&b, wp0, w1);
            nir_def *wp2 = nir_iadd(&b, wp1, w2);
            nir_def *wp3 = nir_iadd(&b, wp2, w3);
            w_prefix = nir_ior(
               &b, nir_ior(&b, wp0, nir_ishl_imm(&b, wp1, 8)),
               nir_ior(&b, nir_ishl_imm(&b, wp2, 16),
                       nir_ishl_imm(&b, wp3, 24)));
            nir_def *w_int = nir_imul(
               &b, nir_div_round_up_imm(&b, qu, shape->tile_w),
               nir_div_round_up_imm(&b, qv, shape->tile_h));
            nir_def *w_multi = nir_iadd(&b, wp3, w_int);
            child_wgs = nir_bcsel(
               &b, nir_ior(&b, all_ones, whole_patch), nir_imm_int(&b, 1),
               w_multi);
         } else {
            /* Triangles domain. */
            nir_def *outer_sum =
               nir_iadd(&b, nir_iadd(&b, n_out[0], n_out[1]), n_out[2]);
            nir_def *ni = n_in[0];
            nir_def *nim2 = nir_usub_sat(&b, ni, nir_imm_int(&b, 2));
            if (shape->allow_whole_patch) {
               nir_def *m =
                  nir_ushr_imm(&b, nir_usub_sat(&b, ni, nir_imm_int(&b, 1)), 1);
               nir_def *even =
                  nir_ixor(&b, nir_iand_imm(&b, ni, 1), nir_imm_int(&b, 1));
               nir_def *v_int = nir_iadd(
                  &b, nir_imul_imm(&b, nir_imul(&b, m, nir_iadd(&b, m, even)), 3),
                  even);
               nir_def *v_patch = nir_iadd(&b, outer_sum, v_int);
               nir_def *p_int = nir_ushr_imm(
                  &b, nir_imul_imm(&b, nir_imul(&b, nim2, nim2), 3), 1);
               nir_def *p_patch = nir_iadd(
                  &b, nir_iadd(&b, outer_sum, nir_imul_imm(&b, nim2, 3)), p_int);
               whole_patch = nir_iand(
                  &b, nir_inot(&b, all_ones),
                  nir_iand(&b, nir_ule_imm(&b, v_patch, shape->max_v),
                           nir_ule_imm(&b, p_patch, shape->max_p)));
            }
            nir_def *w0 = nir_div_round_up_imm(&b, nir_iadd(&b, n_out[0], nim2),
                                               shape->p_strip);
            nir_def *w1 = nir_div_round_up_imm(&b, nir_iadd(&b, n_out[1], nim2),
                                               shape->p_strip);
            nir_def *w2 = nir_div_round_up_imm(&b, nir_iadd(&b, n_out[2], nim2),
                                               shape->p_strip);
            nir_def *wp0 = w0;
            nir_def *wp1 = nir_iadd(&b, wp0, w1);
            nir_def *wp2 = nir_iadd(&b, wp1, w2);
            w_prefix = nir_ior(
               &b, nir_ior(&b, wp0, nir_ishl_imm(&b, wp1, 8)),
               nir_ishl_imm(&b, wp2, 16));

            /* Sum interior ring WGs for r = 1 .. (ni >> 1) - 1 (<= 31 iters). */
            nir_variable *w_acc = nir_local_variable_create(
               impl, glsl_uint_type(), "ft_tri_w_acc");
            nir_variable *r_var = nir_local_variable_create(
               impl, glsl_uint_type(), "ft_tri_r");
            nir_store_var(&b, w_acc, nir_iadd(&b, wp2, nir_iand_imm(&b, ni, 1)), 1);
            nir_store_var(&b, r_var, nir_imm_int(&b, 1), 1);
            nir_def *rings = nir_ushr_imm(&b, ni, 1);
            nir_if *need_multi =
               nir_push_if(&b, nir_inot(&b, nir_ior(&b, all_ones, whole_patch)));
            {
               nir_loop *loop = nir_push_loop(&b);
               {
                  nir_def *r = nir_load_var(&b, r_var);
                  nir_push_if(&b, nir_uge(&b, r, rings));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, NULL);
                  /* T(r) = 2 * (ni - 2*r - 1). */
                  nir_def *t_r = nir_ishl_imm(
                     &b, nir_isub(&b, ni, nir_iadd_imm(&b, nir_ishl_imm(&b, r, 1), 1)), 1);
                  nir_def *wr = nir_div_round_up_imm(&b, t_r, shape->p_strip);
                  nir_store_var(&b, w_acc,
                                nir_iadd(&b, nir_load_var(&b, w_acc),
                                         nir_imul_imm(&b, wr, 3)),
                                1);
                  nir_store_var(&b, r_var, nir_iadd_imm(&b, r, 1), 1);
               }
               nir_pop_loop(&b, loop);
            }
            nir_pop_if(&b, need_multi);
            child_wgs = nir_bcsel(
               &b, nir_ior(&b, all_ones, whole_patch), nir_imm_int(&b, 1),
               nir_load_var(&b, w_acc));
         }

         /* Pack and write the 40 B payload header. */
         nir_def *packed_n_out = nir_ior(
            &b, nir_ior(&b, n_out[0], nir_ishl_imm(&b, n_out[1], 8)),
            nir_ior(&b, nir_ishl_imm(&b, n_out[2], 16),
                    nir_ishl_imm(&b, n_out[3], 24)));
         nir_def *flags = nir_bcsel(
            &b, nir_ine_imm(&b, flip_winding, 0),
            nir_imm_int(&b, HK_TESS_FLAG_FLIP_WINDING), nir_imm_int(&b, 0));
         flags = nir_ior(
            &b, flags,
            nir_bcsel(&b, all_ones, nir_imm_int(&b, HK_TESS_FLAG_ALL_ONES),
                      nir_imm_int(&b, 0)));
         flags = nir_ior(
            &b, flags,
            nir_bcsel(&b, whole_patch, nir_imm_int(&b, HK_TESS_FLAG_WHOLE_PATCH),
                      nir_imm_int(&b, 0)));
         nir_def *packed_n_in_flags = nir_ior(
            &b, nir_ior(&b, n_in[0], nir_ishl_imm(&b, n_in[1], 8)),
            nir_ishl_imm(&b, flags, 16));

         nir_store_task_payload(&b, nir_vec4(&b, f_out[0], f_out[1], f_out[2], f_out[3]),
                                nir_imm_int(&b, 0), .align_mul = 4, .write_mask = 15);
         nir_store_task_payload(&b, nir_vec2(&b, f_in[0], f_in[1]),
                                nir_imm_int(&b, 16), .align_mul = 4, .write_mask = 3);
         nir_store_task_payload(&b, patch_id, nir_imm_int(&b, 24),
                                .align_mul = 4, .write_mask = 1);
         nir_store_task_payload(&b, packed_n_out, nir_imm_int(&b, 28),
                                .align_mul = 4, .write_mask = 1);
         nir_store_task_payload(&b, packed_n_in_flags, nir_imm_int(&b, 32),
                                .align_mul = 4, .write_mask = 1);
         nir_store_task_payload(&b, w_prefix, nir_imm_int(&b, 36),
                                .align_mul = 4, .write_mask = 1);

         if (pl->has_raw_levels) {
            nir_store_task_payload(
               &b, nir_vec4(&b, raw_out[0], raw_out[1], raw_out[2], raw_out[3]),
               nir_imm_int(&b, pl->raw_levels_off), .align_mul = 4,
               .write_mask = 15);
            nir_store_task_payload(
               &b, nir_vec2(&b, raw_in[0], raw_in[1]),
               nir_imm_int(&b, pl->raw_levels_off + 16), .align_mul = 4,
               .write_mask = 3);
         }

         nir_store_var(&b, launch_x, child_wgs, 1);
      }
      nir_pop_if(&b, alive_if);
   }
   nir_pop_if(&b, lane0_if);

   nir_def *gx = nir_load_var(&b, launch_x);
   nir_def *nz = nir_ine_imm(&b, gx, 0);
   nir_def *gy = nir_bcsel(&b, nz, nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_launch_mesh_workgroups(&b, nir_vec3(&b, gx, gy, gy));

   /* Constant data for VS and TCS is already owned and bound at
    * root.constant_data[MESA_SHADER_VERTEX / TESS_CTRL] by their API shaders;
    * clear obj->constant_data so the native TES leaf's two variants don't
    * carry conflicting constant-data payloads in hk_api_shader_init_constant_data. */
   obj->constant_data = NULL;
   obj->constant_data_size = 0;

   NIR_PASS(_, obj, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out, NULL);
   obj->info.stage = MESA_SHADER_TASK;
   memset(&obj->info.tess, 0, sizeof(obj->info.tess));
   obj->info.workgroup_size[0] = 32;
   obj->info.workgroup_size[1] = obj->info.workgroup_size[2] = 1;
   obj->info.workgroup_size_variable = false;
   obj->info.shared_size = shared_bytes;
   obj->info.task_payload_size = pl->payload_B;
   obj->info.outputs_written = 0;
   obj->info.patch_outputs_written = 0;
   obj->info.inputs_read = 0;
   obj->info.patch_inputs_read = 0;

   nir_progress(true, impl, nir_metadata_none);
   nir_repair_ssa_impl(impl);
   NIR_PASS(_, obj, nir_lower_global_vars_to_local);
   NIR_PASS(_, obj, nir_lower_vars_to_ssa);
   NIR_PASS(_, obj, nir_opt_copy_prop);
   NIR_PASS(_, obj, nir_opt_constant_folding);
   NIR_PASS(_, obj, nir_opt_dce);
   NIR_PASS(_, obj, nir_opt_dead_cf);
   NIR_PASS(_, obj, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out, NULL);
   nir_shader_gather_info(obj, impl);
   /* Re-assert explicit shared/payload allocations after gather_info. */
   obj->info.shared_size = shared_bytes;
   obj->info.task_payload_size = pl->payload_B;
   nir_validate_shader(obj, "fused OBJECT(VS+TCS) tessellation lowering");
   return obj;
}

/* =========================================================================
 * Part 2: Synthesize Shared-Vertex MESH(TES) NIR Shader
 * ========================================================================= */

/* Compute (u, v) on Quad outer edge e (0..3) at point k in [0..n_out[e]]. */
static nir_def *
nir_quad_outer_uv(nir_builder *b, enum gl_tess_spacing spacing, nir_def *e,
                  nir_def *k, nir_def *const n_out[4], nir_def *const f_out[4])
{
   nir_def *n_e = nir_bcsel(
      b, nir_ieq_imm(b, e, 0), n_out[0],
      nir_bcsel(b, nir_ieq_imm(b, e, 1), n_out[1],
                nir_bcsel(b, nir_ieq_imm(b, e, 2), n_out[2], n_out[3])));
   nir_def *f_e = nir_bcsel(
      b, nir_ieq_imm(b, e, 0), f_out[0],
      nir_bcsel(b, nir_ieq_imm(b, e, 1), f_out[1],
                nir_bcsel(b, nir_ieq_imm(b, e, 2), f_out[2], f_out[3])));
   /* Edge 0 (U=0, V:1->0) and Edge 3 (V=1, U:1->0) traverse reversed. */
   nir_def *rev = nir_ior(b, nir_ieq_imm(b, e, 0), nir_ieq_imm(b, e, 3));
   nir_def *q = nir_bcsel(b, rev, nir_usub_sat(b, n_e, k), k);
   nir_def *t = nir_eval_1d(b, spacing, n_e, f_e, q);
   nir_def *is_odd = nir_ine_imm(b, nir_iand_imm(b, e, 1), 0);
   nir_def *u = nir_bcsel(
      b, is_odd, t,
      nir_bcsel(b, nir_ieq_imm(b, e, 2), nir_imm_float(b, 1.0f),
                nir_imm_float(b, 0.0f)));
   nir_def *v = nir_bcsel(
      b, is_odd,
      nir_bcsel(b, nir_ieq_imm(b, e, 3), nir_imm_float(b, 1.0f),
                nir_imm_float(b, 0.0f)),
      t);
   return nir_vec2(b, u, v);
}

/* Compute (u, v) on Quad inner border of edge e (0..3) at point k in
 * [0..n_in_axis - 2]. */
static nir_def *
nir_quad_inner_edge_uv(nir_builder *b, enum gl_tess_spacing spacing, nir_def *e,
                       nir_def *k, nir_def *n_in_u, nir_def *n_in_v,
                       nir_def *f_in_u, nir_def *f_in_v)
{
   nir_def *iu = nir_bcsel(
      b, nir_ieq_imm(b, e, 0), nir_imm_int(b, 1),
      nir_bcsel(b, nir_ieq_imm(b, e, 1), nir_iadd_imm(b, k, 1),
                nir_bcsel(b, nir_ieq_imm(b, e, 2),
                          nir_iadd_imm(b, n_in_u, -1),
                          nir_isub(b, nir_iadd_imm(b, n_in_u, -1), k))));
   nir_def *iv = nir_bcsel(
      b, nir_ieq_imm(b, e, 0), nir_isub(b, nir_iadd_imm(b, n_in_v, -1), k),
      nir_bcsel(b, nir_ieq_imm(b, e, 1), nir_imm_int(b, 1),
                nir_bcsel(b, nir_ieq_imm(b, e, 2), nir_iadd_imm(b, k, 1),
                          nir_iadd_imm(b, n_in_v, -1))));
   return nir_vec2(b, nir_eval_1d(b, spacing, n_in_u, f_in_u, iu),
                   nir_eval_1d(b, spacing, n_in_v, f_in_v, iv));
}

/* Compute (u, v) on Triangle ring r, edge e (0..2), point k in [0..s_r] in
 * exact 16.16 fixed-point (matching poly_tessellator.h) so w = 1 - u - v is
 * exact on all edges/rings and adjacent corners are bit-identical. */
static nir_def *
nir_tri_ring_uv(nir_builder *b, enum gl_tess_spacing spacing, nir_def *r,
                nir_def *e, nir_def *k, nir_def *const n_out[3],
                nir_def *const f_out[3], nir_def *n_in, nir_def *f_in)
{
   nir_def *uv_outer = NULL, *uv_inner = NULL;
   nir_push_if(b, nir_ieq_imm(b, r, 0));
   {
      nir_def *n_e = nir_bcsel(
         b, nir_ieq_imm(b, e, 0), n_out[0],
         nir_bcsel(b, nir_ieq_imm(b, e, 1), n_out[1], n_out[2]));
      nir_def *f_e = nir_bcsel(
         b, nir_ieq_imm(b, e, 0), f_out[0],
         nir_bcsel(b, nir_ieq_imm(b, e, 1), f_out[1], f_out[2]));
      nir_def *q = nir_bcsel(b, nir_ieq_imm(b, e, 1), k, nir_usub_sat(b, n_e, k));
      nir_def *t_fxp = nir_eval_1d_fxp(b, spacing, n_e, f_e, q);
      nir_def *u_fxp = nir_bcsel(b, nir_ieq_imm(b, e, 0), nir_imm_int(b, 0), t_fxp);
      nir_def *v_fxp = nir_bcsel(
         b, nir_ieq_imm(b, e, 0), t_fxp,
         nir_bcsel(b, nir_ieq_imm(b, e, 1), nir_imm_int(b, 0),
                   nir_isub(b, nir_imm_int(b, HK_TESS_FXP_ONE), t_fxp)));
      uv_outer = nir_vec2(b, nir_fxp_to_f32(b, u_fxp), nir_fxp_to_f32(b, v_fxp));
   }
   nir_push_else(b, NULL);
   {
      nir_def *r2 = nir_ishl_imm(b, r, 1);
      nir_def *is_center = nir_ieq(b, r2, n_in);
      nir_def *s_r = nir_usub_sat(b, n_in, r2);
      nir_def *perp_raw = nir_eval_1d_fxp(b, spacing, n_in, f_in, r);
      /* Exact floor(perp_raw / 3) via umul_high(x, 0xaaaaaaab) >> 1 so
       * perp_fxp = 2 * half_perp is even and 3 * half_perp <= perp_raw,
       * matching tessellator.h bit-for-bit with strictly non-negative winding
       * across all degenerate frac == 0 boundaries (e.g. 0x40400001). */
      nir_def *half_perp = nir_ushr_imm(
         b, nir_umul_high(b, perp_raw, nir_imm_int(b, 0xaaaaaaabu)), 1);
      nir_def *perp_fxp = nir_ishl_imm(b, half_perp, 1);
      /* Canonicalize ring endpoints (k == s_r) to the next edge's start
       * ((e + 1) % 3, k = 0) and evaluate the exact fallback tessellator.h
       * formula fxpParam - fxpHalfPerp so every interior vertex and corner
       * matches the SW fallback bit-for-bit. */
      nir_def *at_end = nir_ieq(b, k, s_r);
      nir_def *e_c = nir_bcsel(
         b, at_end,
         nir_bcsel(b, nir_ieq_imm(b, e, 2), nir_imm_int(b, 0),
                   nir_iadd_imm(b, e, 1)),
         e);
      nir_def *k_c = nir_bcsel(b, at_end, nir_imm_int(b, 0), k);
      nir_def *q = nir_bcsel(
         b, nir_ieq_imm(b, e_c, 1), nir_iadd(b, r, k_c),
         nir_usub_sat(b, nir_usub_sat(b, n_in, r), k_c));
      nir_def *param_fxp = nir_eval_1d_fxp(b, spacing, n_in, f_in, q);
      nir_def *tmp_fxp = nir_usub_sat(b, param_fxp, half_perp);
      nir_def *u_gen = nir_bcsel(b, nir_ieq_imm(b, e_c, 0), perp_fxp, tmp_fxp);
      nir_def *v_gen = nir_bcsel(
         b, nir_ieq_imm(b, e_c, 0), tmp_fxp,
         nir_bcsel(b, nir_ieq_imm(b, e_c, 1), perp_fxp,
                   nir_usub_sat(b, nir_imm_int(b, HK_TESS_FXP_ONE),
                                nir_iadd(b, tmp_fxp, perp_fxp))));
      nir_def *u_fxp =
         nir_bcsel(b, is_center, nir_imm_int(b, HK_TESS_FXP_ONE_THIRD), u_gen);
      nir_def *v_fxp =
         nir_bcsel(b, is_center, nir_imm_int(b, HK_TESS_FXP_ONE_THIRD), v_gen);
      uv_inner = nir_vec2(b, nir_fxp_to_f32(b, u_fxp), nir_fxp_to_f32(b, v_fxp));
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, uv_outer, uv_inner);
}

static void
emit_triangle_prim(nir_builder *b, nir_def *prim_idx, nir_def *i0, nir_def *i1,
                   nir_def *i2, nir_def *flip_winding)
{
   nir_def *a = i0;
   nir_def *c1 = nir_bcsel(b, flip_winding, i2, i1);
   nir_def *c2 = nir_bcsel(b, flip_winding, i1, i2);
   nir_store_per_primitive_output(
      b, nir_vec3(b, a, c1, c2), prim_idx, nir_imm_int(b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
}

/* Triangulate one interior quad cell (gx, gy) in [0..qu-1] x [0..qv-1] with
 * concentric-ring radial diagonals: v00--v11 in the bottom-left and top-right
 * quadrants, v10--v01 in top-left and bottom-right (matching Vulkan / poly
 * tessellator symmetry and keeping concentric ring corners sharp). */
static void
emit_quad_cell_tri(nir_builder *b, nir_def *prim_idx, nir_def *v00, nir_def *v10,
                   nir_def *v01, nir_def *v11, nir_def *tri, nir_def *gx,
                   nir_def *gy, nir_def *qu, nir_def *qv, nir_def *flip_winding)
{
   nir_def *is_t1 = nir_ine_imm(b, tri, 0);
   nir_def *gx2p1 = nir_iadd_imm(b, nir_ishl_imm(b, gx, 1), 1);
   nir_def *gy2p1 = nir_iadd_imm(b, nir_ishl_imm(b, gy, 1), 1);
   nir_def *diag_00_11 = nir_ior(
      b,
      nir_ior(b, nir_iand(b, nir_ult(b, gx2p1, qu), nir_ult(b, gy2p1, qv)),
              nir_iand(b, nir_ult(b, qu, gx2p1), nir_ult(b, qv, gy2p1))),
      nir_iand(b, nir_ieq(b, gx2p1, qu), nir_ine(b, gy2p1, qv)));
   /* Match tessellator.h cyclic vertex order (provoking vertex v01 for
    * !diag_00_11 center cells: (v01, v00, v10) and (v01, v10, v11); v00 for
    * diag_00_11 cells: (v00, v10, v11) and (v00, v11, v01)). */
   nir_def *a = nir_bcsel(b, diag_00_11, v00, v01);
   nir_def *b_v = nir_bcsel(
      b, is_t1, nir_bcsel(b, diag_00_11, v11, v10),
      nir_bcsel(b, diag_00_11, v10, v00));
   nir_def *c_v = nir_bcsel(
      b, is_t1, nir_bcsel(b, diag_00_11, v01, v11),
      nir_bcsel(b, diag_00_11, v11, v10));
   emit_triangle_prim(b, prim_idx, a, b_v, c_v, flip_winding);
}

struct tes_lower_state {
   const struct tess_payload_layout *pl;
   unsigned clip_size;
   unsigned cull_size;
   nir_def *lane;
   nir_def *uv;
   nir_def *patch_id;
};

static bool
lower_tes_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct tes_lower_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_tess_coord_xy:
      nir_def_replace(&intr->def, s->uv);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, s->patch_id);
      return true;
   case nir_intrinsic_load_patch_vertices_in:
      nir_def_replace(&intr->def, nir_imm_int(b, s->pl->out_v));
      return true;
   case nir_intrinsic_load_tess_level_outer:
   case nir_intrinsic_load_tess_level_inner: {
      unsigned off = s->pl->raw_levels_off +
                     (intr->intrinsic == nir_intrinsic_load_tess_level_inner ? 16 : 0);
      nir_def *v = nir_load_task_payload(
         b, intr->def.num_components, 32, nir_imm_int(b, off), .align_mul = 4);
      nir_def_replace(&intr->def, v);
      return true;
   }
   case nir_intrinsic_load_input: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[0]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned num = intr->def.num_components;
      if (loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
          loc == VARYING_SLOT_TESS_LEVEL_INNER) {
         unsigned base = s->pl->raw_levels_off +
                         (loc == VARYING_SLOT_TESS_LEVEL_INNER ? 16 : 0) +
                         comp * 4;
         nir_def *v = nir_load_task_payload(b, num, 32, nir_imm_int(b, base),
                                            .align_mul = 4);
         nir_def_replace(&intr->def, v);
         return true;
      }
      unsigned p = loc - VARYING_SLOT_PATCH0;
      nir_def *comps[4];
      for (unsigned i = 0; i < num; ++i) {
         comps[i] = nir_load_task_payload(
            b, 1, 32, nir_imm_int(b, s->pl->pc_off[p][comp + i]),
            .align_mul = 4);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, num));
      return true;
   }
   case nir_intrinsic_load_per_vertex_input: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned num = intr->def.num_components;
      nir_def *vidx =
         nir_umin(b, intr->src[0].ssa, nir_imm_int(b, s->pl->out_v - 1));
      nir_def *vbase = nir_iadd_imm(
         b, nir_imul_imm(b, vidx, s->pl->pv_stride_B), HK_TESS_PAYLOAD_HEADER_B);
      nir_def *comps[4];
      for (unsigned i = 0; i < num; ++i) {
         nir_def *off = nir_iadd_imm(b, vbase, s->pl->pv_off[loc][comp + i]);
         comps[i] = nir_load_task_payload(b, 1, 32, off, .align_mul = 4);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, num));
      return true;
   }
   case nir_intrinsic_store_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned wmask = nir_intrinsic_write_mask(intr);
      if (loc == VARYING_SLOT_PSIZ) {
         nir_instr_remove(&intr->instr);
         return true;
      }
      if (loc == VARYING_SLOT_CULL_DIST0 || loc == VARYING_SLOT_CULL_DIST1) {
         u_foreach_bit(i, wmask) {
            unsigned idx = s->clip_size +
                           4 * (loc - VARYING_SLOT_CULL_DIST0) + comp + i;
            unsigned tloc = VARYING_SLOT_CLIP_DIST0 + idx / 4;
            unsigned tcomp = idx % 4;
            nir_store_per_vertex_output(
               b, nir_channel(b, intr->src[0].ssa, i), s->lane,
               nir_imm_int(b, 0), .component = tcomp, .write_mask = 1,
               .src_type = nir_type_float32,
               .io_semantics = {.location = tloc, .num_slots = 1});
         }
         nir_instr_remove(&intr->instr);
         return true;
      }
      nir_store_per_vertex_output(
         b, intr->src[0].ssa, s->lane, nir_imm_int(b, 0), .component = comp,
         .write_mask = wmask, .src_type = nir_intrinsic_src_type(intr),
         .io_semantics = {.location = loc, .num_slots = 1});
      nir_instr_remove(&intr->instr);
      return true;
   }
   default:
      return false;
   }
}

static nir_shader *
build_mesh_shader(nir_shader *tes, enum tess_primitive_mode domain,
                  enum gl_tess_spacing spacing,
                  const struct tess_payload_layout *pl,
                  const struct tess_meshlet_shape *shape, unsigned clip_size,
                  unsigned cull_size, bool export_primitive_id)
{
   NIR_PASS(_, tes, nir_lower_returns);
   NIR_PASS(_, tes, nir_lower_tess_coord_z, domain == TESS_PRIMITIVE_TRIANGLES);

   nir_function_impl *impl = nir_shader_get_entrypoint(tes);
   nir_cf_list tes_body;
   nir_cf_extract(&tes_body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *wg_id = nir_channel(&b, nir_load_workgroup_id(&b), 0);

   /* Load the 40 B payload header from GET_SR 0xe0/0xe1 task payload. */
   nir_def *f_out_v =
      nir_load_task_payload(&b, 4, 32, nir_imm_int(&b, 0), .align_mul = 4);
   nir_def *f_in_v =
      nir_load_task_payload(&b, 2, 32, nir_imm_int(&b, 16), .align_mul = 4);
   nir_def *patch_id =
      nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 24), .align_mul = 4);
   nir_def *packed_n_out =
      nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 28), .align_mul = 4);
   nir_def *packed_n_in_flags =
      nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 32), .align_mul = 4);
   nir_def *w_prefix =
      nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 36), .align_mul = 4);

   nir_def *f_out[4] = {
      nir_channel(&b, f_out_v, 0), nir_channel(&b, f_out_v, 1),
      nir_channel(&b, f_out_v, 2), nir_channel(&b, f_out_v, 3),
   };
   nir_def *f_in[2] = {
      nir_channel(&b, f_in_v, 0), nir_channel(&b, f_in_v, 1),
   };
   nir_def *n_out[4];
   for (unsigned e = 0; e < 4; ++e) {
      n_out[e] = nir_iand_imm(&b, nir_ushr_imm(&b, packed_n_out, 8 * e), 0xff);
   }
   nir_def *n_in_u = nir_iand_imm(&b, packed_n_in_flags, 0xff);
   nir_def *n_in_v =
      nir_iand_imm(&b, nir_ushr_imm(&b, packed_n_in_flags, 8), 0xff);
   nir_def *flags = nir_ushr_imm(&b, packed_n_in_flags, 16);
   nir_def *flip_winding =
      nir_ine_imm(&b, nir_iand_imm(&b, flags, HK_TESS_FLAG_FLIP_WINDING), 0);
   nir_def *is_all_ones =
      nir_ine_imm(&b, nir_iand_imm(&b, flags, HK_TESS_FLAG_ALL_ONES), 0);
   nir_def *is_whole_patch =
      shape->allow_whole_patch
         ? nir_ine_imm(&b, nir_iand_imm(&b, flags, HK_TESS_FLAG_WHOLE_PATCH), 0)
         : nir_imm_false(&b);

   nir_def *wp0 = nir_iand_imm(&b, w_prefix, 0xff);
   nir_def *wp1 = nir_iand_imm(&b, nir_ushr_imm(&b, w_prefix, 8), 0xff);
   nir_def *wp2 = nir_iand_imm(&b, nir_ushr_imm(&b, w_prefix, 16), 0xff);
   nir_def *wp3 = nir_iand_imm(&b, nir_ushr_imm(&b, w_prefix, 24), 0xff);

   /* Shared state computed before the single TES body & reused for prims. */
   nir_variable *v_live_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_v_live");
   nir_variable *p_live_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_p_live");
   nir_variable *uv_var =
      nir_local_variable_create(impl, glsl_vec2_type(), "ft_uv");
   /* Strip / tile routing locals shared between vertex and primitive phases. */
   nir_variable *mode_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_mode");
   nir_variable *edge_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_edge");
   nir_variable *s_out_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_s_out");
   nir_variable *s_in_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_s_in");
   nir_variable *t0_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_t0");
   nir_variable *o0_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_o0");
   nir_variable *i0_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_i0");
   nir_variable *no_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_no");
   nir_variable *cw_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_cw");
   nir_variable *ox_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_ox");
   nir_variable *oy_var =
      nir_local_variable_create(impl, glsl_uint_type(), "ft_oy");

   nir_store_var(&b, v_live_var, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, p_live_var, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, uv_var, nir_imm_vec2(&b, 0.0f, 0.0f), 3);
   nir_store_var(&b, mode_var, nir_imm_int(&b, 0), 1);

   /* ---- Compute (V_live, P_live, uv) for this workgroup & lane ---- */
   if (domain == TESS_PRIMITIVE_QUADS) {
      /* mode: 0 = all_ones, 1 = whole_patch, 2 = outer_strip, 3 = interior_tile */
      nir_push_if(&b, is_all_ones);
      {
         nir_store_var(&b, mode_var, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, v_live_var, nir_imm_int(&b, 4), 1);
         nir_store_var(&b, p_live_var, nir_imm_int(&b, 2), 1);
         nir_def *u = nir_bcsel(
            &b, nir_ior(&b, nir_ieq_imm(&b, lane, 1), nir_ieq_imm(&b, lane, 2)),
            nir_imm_float(&b, 1.0f), nir_imm_float(&b, 0.0f));
         nir_def *v = nir_bcsel(&b, nir_uge_imm(&b, lane, 2),
                                nir_imm_float(&b, 1.0f), nir_imm_float(&b, 0.0f));
         nir_store_var(&b, uv_var, nir_vec2(&b, u, v), 3);
      }
      nir_push_else(&b, NULL);
      {
         if (shape->allow_whole_patch) {
            nir_push_if(&b, is_whole_patch);
            {
               nir_store_var(&b, mode_var, nir_imm_int(&b, 1), 1);
               nir_def *s0 = n_out[0];
               nir_def *s1 = nir_iadd(&b, s0, n_out[1]);
               nir_def *s2 = nir_iadd(&b, s1, n_out[2]);
               nir_def *s3 = nir_iadd(&b, s2, n_out[3]);
               nir_def *iw = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 1));
               nir_def *ih = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 1));
               nir_def *qu = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 2));
               nir_def *qv = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 2));
               nir_def *vl = nir_iadd(&b, s3, nir_imul(&b, iw, ih));
               nir_def *pl_c = nir_iadd(
                  &b,
                  nir_iadd(&b, s3, nir_ishl_imm(&b, nir_iadd(&b, qu, qv), 1)),
                  nir_ishl_imm(&b, nir_imul(&b, qu, qv), 1));
               nir_store_var(&b, v_live_var, vl, 1);
               nir_store_var(&b, p_live_var, pl_c, 1);
               nir_push_if(&b, nir_ult(&b, lane, vl));
               {
                  nir_push_if(&b, nir_ult(&b, lane, s3));
                  {
                     nir_def *e = nir_bcsel(
                        &b, nir_ult(&b, lane, s0), nir_imm_int(&b, 0),
                        nir_bcsel(&b, nir_ult(&b, lane, s1), nir_imm_int(&b, 1),
                                  nir_bcsel(&b, nir_ult(&b, lane, s2),
                                            nir_imm_int(&b, 2),
                                            nir_imm_int(&b, 3))));
                     nir_def *base = nir_bcsel(
                        &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
                        nir_bcsel(&b, nir_ieq_imm(&b, e, 1), s0,
                                  nir_bcsel(&b, nir_ieq_imm(&b, e, 2), s1, s2)));
                     nir_def *k = nir_isub(&b, lane, base);
                     nir_store_var(&b, uv_var,
                                   nir_quad_outer_uv(&b, spacing, e, k, n_out, f_out),
                                   3);
                  }
                  nir_push_else(&b, NULL);
                  {
                     nir_def *idx = nir_isub(&b, lane, s3);
                     nir_def *gx = nir_umod(&b, idx, iw);
                     nir_def *gy = nir_udiv(&b, idx, iw);
                     nir_def *u = nir_eval_1d(&b, spacing, n_in_u, f_in[0],
                                              nir_iadd_imm(&b, gx, 1));
                     nir_def *v = nir_eval_1d(&b, spacing, n_in_v, f_in[1],
                                              nir_iadd_imm(&b, gy, 1));
                     nir_store_var(&b, uv_var, nir_vec2(&b, u, v), 3);
                  }
                  nir_pop_if(&b, NULL);
               }
               nir_pop_if(&b, NULL);
            }
            nir_push_else(&b, NULL);
         }

         /* Multi-WG Quad: outer strip (wg_id < wp3) vs interior tile. */
         nir_push_if(&b, nir_ult(&b, wg_id, wp3));
         {
            nir_store_var(&b, mode_var, nir_imm_int(&b, 2), 1);
            nir_def *e = nir_bcsel(
               &b, nir_ult(&b, wg_id, wp0), nir_imm_int(&b, 0),
               nir_bcsel(&b, nir_ult(&b, wg_id, wp1), nir_imm_int(&b, 1),
                         nir_bcsel(&b, nir_ult(&b, wg_id, wp2),
                                   nir_imm_int(&b, 2), nir_imm_int(&b, 3))));
            nir_def *prev_wp = nir_bcsel(
               &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
               nir_bcsel(&b, nir_ieq_imm(&b, e, 1), wp0,
                         nir_bcsel(&b, nir_ieq_imm(&b, e, 2), wp1, wp2)));
            nir_def *chunk = nir_isub(&b, wg_id, prev_wp);
            nir_def *n_o = nir_bcsel(
               &b, nir_ieq_imm(&b, e, 0), n_out[0],
               nir_bcsel(&b, nir_ieq_imm(&b, e, 1), n_out[1],
                         nir_bcsel(&b, nir_ieq_imm(&b, e, 2), n_out[2], n_out[3])));
            nir_def *n_is = nir_usub_sat(
               &b,
               nir_bcsel(&b, nir_ine_imm(&b, nir_iand_imm(&b, e, 1), 0), n_in_u,
                         n_in_v),
               nir_imm_int(&b, 2));
            nir_def *t_total = nir_iadd(&b, n_o, n_is);
            nir_def *t0 = nir_imul_imm(&b, chunk, shape->p_strip);
            nir_def *t1 = nir_umin(&b, nir_iadd_imm(&b, t0, shape->p_strip), t_total);
            nir_def *cnt = nir_usub_sat(&b, t1, t0);
            nir_def *i0 = nir_stitch_i(&b, n_o, n_is, t0);
            nir_def *o0 = nir_isub(&b, t0, i0);
            nir_def *i1 = nir_stitch_i(&b, n_o, n_is, t1);
            nir_def *o1 = nir_isub(&b, t1, i1);
            nir_def *no = nir_iadd_imm(&b, nir_isub(&b, o1, o0), 1);
            nir_def *ni = nir_iadd_imm(&b, nir_isub(&b, i1, i0), 1);
            nir_def *vl = nir_iadd(&b, no, ni);

            nir_store_var(&b, edge_var, e, 1);
            nir_store_var(&b, s_out_var, n_o, 1);
            nir_store_var(&b, s_in_var, n_is, 1);
            nir_store_var(&b, t0_var, t0, 1);
            nir_store_var(&b, o0_var, o0, 1);
            nir_store_var(&b, i0_var, i0, 1);
            nir_store_var(&b, no_var, no, 1);
            nir_store_var(&b, v_live_var, vl, 1);
            nir_store_var(&b, p_live_var, cnt, 1);

            nir_push_if(&b, nir_ult(&b, lane, vl));
            {
               nir_push_if(&b, nir_ult(&b, lane, no));
               {
                  nir_def *k = nir_iadd(&b, o0, lane);
                  nir_store_var(&b, uv_var,
                                nir_quad_outer_uv(&b, spacing, e, k, n_out, f_out),
                                3);
               }
               nir_push_else(&b, NULL);
               {
                  nir_def *k = nir_iadd(&b, i0, nir_isub(&b, lane, no));
                  nir_store_var(
                     &b, uv_var,
                     nir_quad_inner_edge_uv(&b, spacing, e, k, n_in_u, n_in_v,
                                            f_in[0], f_in[1]),
                     3);
               }
               nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
         }
         nir_push_else(&b, NULL);
         {
            /* Quad interior tile. */
            nir_store_var(&b, mode_var, nir_imm_int(&b, 3), 1);
            nir_def *int_id = nir_isub(&b, wg_id, wp3);
            nir_def *qu = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 2));
            nir_def *qv = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 2));
            nir_def *tiles_u =
               nir_umax(&b, nir_div_round_up_imm(&b, qu, shape->tile_w),
                        nir_imm_int(&b, 1));
            nir_def *tx = nir_umod(&b, int_id, tiles_u);
            nir_def *ty = nir_udiv(&b, int_id, tiles_u);
            nir_def *ox = nir_imul_imm(&b, tx, shape->tile_w);
            nir_def *oy = nir_imul_imm(&b, ty, shape->tile_h);
            nir_def *cw =
               nir_umin(&b, nir_imm_int(&b, shape->tile_w), nir_usub_sat(&b, qu, ox));
            nir_def *ch =
               nir_umin(&b, nir_imm_int(&b, shape->tile_h), nir_usub_sat(&b, qv, oy));
            nir_def *vw = nir_iadd_imm(&b, cw, 1);
            nir_def *vh = nir_iadd_imm(&b, ch, 1);
            nir_def *vl = nir_imul(&b, vw, vh);
            nir_def *pl_c = nir_ishl_imm(&b, nir_imul(&b, cw, ch), 1);
            nir_store_var(&b, cw_var, cw, 1);
            nir_store_var(&b, ox_var, ox, 1);
            nir_store_var(&b, oy_var, oy, 1);
            nir_store_var(&b, v_live_var, vl, 1);
            nir_store_var(&b, p_live_var, pl_c, 1);
            nir_push_if(&b, nir_ult(&b, lane, vl));
            {
               nir_def *vx = nir_umod(&b, lane, vw);
               nir_def *vy = nir_udiv(&b, lane, vw);
               nir_def *u = nir_eval_1d(&b, spacing, n_in_u, f_in[0],
                                        nir_iadd(&b, nir_iadd_imm(&b, ox, 1), vx));
               nir_def *v = nir_eval_1d(&b, spacing, n_in_v, f_in[1],
                                        nir_iadd(&b, nir_iadd_imm(&b, oy, 1), vy));
               nir_store_var(&b, uv_var, nir_vec2(&b, u, v), 3);
            }
            nir_pop_if(&b, NULL);
         }
         nir_pop_if(&b, NULL);

         if (shape->allow_whole_patch)
            nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   } else {
      /* TRIANGLES domain.
       * mode: 0 = all_ones, 1 = whole_patch, 2 = ring_strip, 3 = center_tri */
      nir_def *ni = n_in_u;
      nir_def *fi = f_in[0];
      nir_push_if(&b, is_all_ones);
      {
         nir_store_var(&b, mode_var, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, v_live_var, nir_imm_int(&b, 3), 1);
         nir_store_var(&b, p_live_var, nir_imm_int(&b, 1), 1);
         nir_def *u = nir_bcsel(&b, nir_ieq_imm(&b, lane, 2),
                                nir_imm_float(&b, 1.0f), nir_imm_float(&b, 0.0f));
         nir_def *v = nir_bcsel(&b, nir_ieq_imm(&b, lane, 0),
                                nir_imm_float(&b, 1.0f), nir_imm_float(&b, 0.0f));
         nir_store_var(&b, uv_var, nir_vec2(&b, u, v), 3);
      }
      nir_push_else(&b, NULL);
      {
         if (shape->allow_whole_patch) {
            nir_push_if(&b, is_whole_patch);
            {
               nir_store_var(&b, mode_var, nir_imm_int(&b, 1), 1);
               nir_def *outer_sum =
                  nir_iadd(&b, nir_iadd(&b, n_out[0], n_out[1]), n_out[2]);
               nir_def *nim2 = nir_usub_sat(&b, ni, nir_imm_int(&b, 2));
               nir_def *m =
                  nir_ushr_imm(&b, nir_usub_sat(&b, ni, nir_imm_int(&b, 1)), 1);
               nir_def *even =
                  nir_ixor(&b, nir_iand_imm(&b, ni, 1), nir_imm_int(&b, 1));
               nir_def *v_int = nir_iadd(
                  &b, nir_imul_imm(&b, nir_imul(&b, m, nir_iadd(&b, m, even)), 3),
                  even);
               nir_def *vl = nir_iadd(&b, outer_sum, v_int);
               nir_def *p_int = nir_ushr_imm(
                  &b, nir_imul_imm(&b, nir_imul(&b, nim2, nim2), 3), 1);
               nir_def *pl_c = nir_iadd(
                  &b, nir_iadd(&b, outer_sum, nir_imul_imm(&b, nim2, 3)), p_int);
               nir_store_var(&b, v_live_var, vl, 1);
               nir_store_var(&b, p_live_var, pl_c, 1);

               nir_push_if(&b, nir_ult(&b, lane, vl));
               {
                  /* Locate ring r, edge e, offset k for vertex `lane` (R <= 2). */
                  nir_variable *rem = nir_local_variable_create(
                     impl, glsl_uint_type(), "ft_wp_vrem");
                  nir_variable *rv = nir_local_variable_create(
                     impl, glsl_uint_type(), "ft_wp_vr");
                  nir_variable *ev = nir_local_variable_create(
                     impl, glsl_uint_type(), "ft_wp_ve");
                  nir_variable *kv = nir_local_variable_create(
                     impl, glsl_uint_type(), "ft_wp_vk");
                  nir_store_var(&b, rem, lane, 1);
                  nir_store_var(&b, rv, nir_imm_int(&b, 0), 1);
                  nir_store_var(&b, ev, nir_imm_int(&b, 0), 1);
                  nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
                  nir_loop *loop = nir_push_loop(&b);
                  {
                     nir_def *r = nir_load_var(&b, rv);
                     nir_def *rem_v = nir_load_var(&b, rem);
                     nir_def *is_r0 = nir_ieq_imm(&b, r, 0);
                     nir_def *s_r = nir_usub_sat(&b, ni, nir_ishl_imm(&b, r, 1));
                     nir_def *is_ctr = nir_iand(&b, nir_inot(&b, is_r0),
                                                nir_ieq_imm(&b, s_r, 0));
                     nir_def *rs = nir_bcsel(
                        &b, is_r0, outer_sum,
                        nir_bcsel(&b, is_ctr, nir_imm_int(&b, 1),
                                  nir_imul_imm(&b, s_r, 3)));
                     nir_push_if(&b, nir_ult(&b, rem_v, rs));
                     {
                        nir_push_if(&b, is_r0);
                        {
                           nir_def *s01 = nir_iadd(&b, n_out[0], n_out[1]);
                           nir_def *e = nir_bcsel(
                              &b, nir_ult(&b, rem_v, n_out[0]), nir_imm_int(&b, 0),
                              nir_bcsel(&b, nir_ult(&b, rem_v, s01),
                                        nir_imm_int(&b, 1), nir_imm_int(&b, 2)));
                           nir_def *k = nir_bcsel(
                              &b, nir_ieq_imm(&b, e, 0), rem_v,
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 1),
                                        nir_isub(&b, rem_v, n_out[0]),
                                        nir_isub(&b, rem_v, s01)));
                           nir_store_var(&b, ev, e, 1);
                           nir_store_var(&b, kv, k, 1);
                        }
                        nir_push_else(&b, NULL);
                        {
                           nir_def *s_safe = nir_umax(&b, s_r, nir_imm_int(&b, 1));
                           nir_store_var(&b, ev, nir_udiv(&b, rem_v, s_safe), 1);
                           nir_store_var(&b, kv, nir_umod(&b, rem_v, s_safe), 1);
                        }
                        nir_pop_if(&b, NULL);
                        nir_jump(&b, nir_jump_break);
                     }
                     nir_pop_if(&b, NULL);
                     nir_store_var(&b, rem, nir_isub(&b, rem_v, rs), 1);
                     nir_store_var(&b, rv, nir_iadd_imm(&b, r, 1), 1);
                  }
                  nir_pop_loop(&b, loop);
                  nir_store_var(
                     &b, uv_var,
                     nir_tri_ring_uv(&b, spacing, nir_load_var(&b, rv),
                                     nir_load_var(&b, ev), nir_load_var(&b, kv),
                                     n_out, f_out, ni, fi),
                     3);
               }
               nir_pop_if(&b, NULL);
            }
            nir_push_else(&b, NULL);
         }

         /* Multi-WG Triangles: map wg_id -> (r, e, chunk) or odd center tri. */
         nir_variable *r_sel =
            nir_local_variable_create(impl, glsl_uint_type(), "ft_tr_r");
         nir_variable *c_sel =
            nir_local_variable_create(impl, glsl_uint_type(), "ft_tr_c");
         nir_variable *is_ctr_wg =
            nir_local_variable_create(impl, glsl_bool_type(), "ft_tr_ctr");
         nir_store_var(&b, r_sel, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, edge_var, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, c_sel, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, is_ctr_wg, nir_imm_false(&b), 1);

         nir_push_if(&b, nir_ult(&b, wg_id, wp2));
         {
            nir_def *e = nir_bcsel(
               &b, nir_ult(&b, wg_id, wp0), nir_imm_int(&b, 0),
               nir_bcsel(&b, nir_ult(&b, wg_id, wp1), nir_imm_int(&b, 1),
                         nir_imm_int(&b, 2)));
            nir_def *prev = nir_bcsel(
               &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
               nir_bcsel(&b, nir_ieq_imm(&b, e, 1), wp0, wp1));
            nir_store_var(&b, edge_var, e, 1);
            nir_store_var(&b, c_sel, nir_isub(&b, wg_id, prev), 1);
            nir_def *n_o = nir_bcsel(
               &b, nir_ieq_imm(&b, e, 0), n_out[0],
               nir_bcsel(&b, nir_ieq_imm(&b, e, 1), n_out[1], n_out[2]));
            nir_store_var(&b, s_out_var, n_o, 1);
            nir_store_var(&b, s_in_var, nir_usub_sat(&b, ni, nir_imm_int(&b, 2)), 1);
         }
         nir_push_else(&b, NULL);
         {
            nir_variable *rem =
               nir_local_variable_create(impl, glsl_uint_type(), "ft_tr_rem");
            nir_store_var(&b, rem, nir_isub(&b, wg_id, wp2), 1);
            nir_store_var(&b, r_sel, nir_imm_int(&b, 1), 1);
            nir_store_var(&b, is_ctr_wg, nir_imm_true(&b), 1);
            nir_def *rings = nir_ushr_imm(&b, ni, 1);
            nir_loop *loop = nir_push_loop(&b);
            {
               nir_def *r = nir_load_var(&b, r_sel);
               nir_push_if(&b, nir_uge(&b, r, rings));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);
               nir_def *s_o = nir_isub(&b, ni, nir_ishl_imm(&b, r, 1));
               nir_def *s_i = nir_usub_sat(&b, s_o, nir_imm_int(&b, 2));
               nir_def *t_r = nir_iadd(&b, s_o, s_i);
               nir_def *wr = nir_div_round_up_imm(&b, t_r, shape->p_strip);
               nir_def *w3 = nir_imul_imm(&b, wr, 3);
               nir_def *rem_v = nir_load_var(&b, rem);
               nir_push_if(&b, nir_ult(&b, rem_v, w3));
               {
                  nir_store_var(&b, edge_var, nir_udiv(&b, rem_v, wr), 1);
                  nir_store_var(&b, c_sel, nir_umod(&b, rem_v, wr), 1);
                  nir_store_var(&b, s_out_var, s_o, 1);
                  nir_store_var(&b, s_in_var, s_i, 1);
                  nir_store_var(&b, is_ctr_wg, nir_imm_false(&b), 1);
                  nir_jump(&b, nir_jump_break);
               }
               nir_pop_if(&b, NULL);
               nir_store_var(&b, rem, nir_isub(&b, rem_v, w3), 1);
               nir_store_var(&b, r_sel, nir_iadd_imm(&b, r, 1), 1);
            }
            nir_pop_loop(&b, loop);
         }
         nir_pop_if(&b, NULL);

         nir_push_if(&b, nir_load_var(&b, is_ctr_wg));
         {
            /* Center triangle of odd n_in (ring R = n_in >> 1). */
            nir_store_var(&b, mode_var, nir_imm_int(&b, 3), 1);
            nir_store_var(&b, v_live_var, nir_imm_int(&b, 3), 1);
            nir_store_var(&b, p_live_var, nir_imm_int(&b, 1), 1);
            nir_push_if(&b, nir_ult_imm(&b, lane, 3));
            {
               nir_store_var(
                  &b, uv_var,
                  nir_tri_ring_uv(&b, spacing, nir_ushr_imm(&b, ni, 1), lane,
                                  nir_imm_int(&b, 0), n_out, f_out, ni, fi),
                  3);
            }
            nir_pop_if(&b, NULL);
         }
         nir_push_else(&b, NULL);
         {
            nir_store_var(&b, mode_var, nir_imm_int(&b, 2), 1);
            nir_def *r = nir_load_var(&b, r_sel);
            nir_def *e = nir_load_var(&b, edge_var);
            nir_def *chunk = nir_load_var(&b, c_sel);
            nir_def *s_o = nir_load_var(&b, s_out_var);
            nir_def *s_i = nir_load_var(&b, s_in_var);
            nir_def *t_total = nir_iadd(&b, s_o, s_i);
            nir_def *t0 = nir_imul_imm(&b, chunk, shape->p_strip);
            nir_def *t1 = nir_umin(&b, nir_iadd_imm(&b, t0, shape->p_strip), t_total);
            nir_def *cnt = nir_usub_sat(&b, t1, t0);
            nir_def *i0 = nir_stitch_i(&b, s_o, s_i, t0);
            nir_def *o0 = nir_isub(&b, t0, i0);
            nir_def *i1 = nir_stitch_i(&b, s_o, s_i, t1);
            nir_def *o1 = nir_isub(&b, t1, i1);
            nir_def *no = nir_iadd_imm(&b, nir_isub(&b, o1, o0), 1);
            nir_def *ni_v = nir_iadd_imm(&b, nir_isub(&b, i1, i0), 1);
            nir_def *vl = nir_iadd(&b, no, ni_v);

            nir_store_var(&b, t0_var, t0, 1);
            nir_store_var(&b, o0_var, o0, 1);
            nir_store_var(&b, i0_var, i0, 1);
            nir_store_var(&b, no_var, no, 1);
            nir_store_var(&b, v_live_var, vl, 1);
            nir_store_var(&b, p_live_var, cnt, 1);

            nir_push_if(&b, nir_ult(&b, lane, vl));
            {
               nir_push_if(&b, nir_ult(&b, lane, no));
               {
                  nir_def *k = nir_iadd(&b, o0, lane);
                  nir_store_var(
                     &b, uv_var,
                     nir_tri_ring_uv(&b, spacing, r, e, k, n_out, f_out, ni, fi),
                     3);
               }
               nir_push_else(&b, NULL);
               {
                  nir_def *k = nir_iadd(&b, i0, nir_isub(&b, lane, no));
                  nir_store_var(
                     &b, uv_var,
                     nir_tri_ring_uv(&b, spacing, nir_iadd_imm(&b, r, 1), e, k,
                                     n_out, f_out, ni, fi),
                     3);
               }
               nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
         }
         nir_pop_if(&b, NULL);

         if (shape->allow_whole_patch)
            nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }

   /* ---- Emit Counts and Primitive Indices BEFORE the TES Body ----
    * Doing topology emission before TES lets every tess-factor and strip/tile
    * register die before the user TES body runs, cutting live register pressure
    * across heavy TES shaders (e.g. Elden Ring's 40-64-comp textured TES). */
   nir_def *v_live = nir_load_var(&b, v_live_var);
   nir_def *p_live = nir_load_var(&b, p_live_var);
   nir_def *uv_val = nir_load_var(&b, uv_var);

   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_set_vertex_and_primitive_count(&b, v_live, p_live, nir_imm_int(&b, 0));
   nir_pop_if(&b, NULL);

   nir_def *mode = nir_load_var(&b, mode_var);
   unsigned prim_iters = DIV_ROUND_UP(shape->max_p, 32u);
   for (unsigned it = 0; it < prim_iters; ++it) {
      nir_def *p = it ? nir_iadd_imm(&b, lane, it * 32) : lane;
      nir_if *p_if = nir_push_if(&b, nir_ult(&b, p, p_live));
      p_if->control = nir_selection_control_dont_flatten;
      {
         if (domain == TESS_PRIMITIVE_QUADS) {
            nir_push_if(&b, nir_ieq_imm(&b, mode, 0));
            {
               /* All-1s Quad: (0, 1, 3), (1, 2, 3). */
               nir_def *is_p1 = nir_ine_imm(&b, p, 0);
               emit_triangle_prim(
                  &b, p,
                  nir_bcsel(&b, is_p1, nir_imm_int(&b, 1), nir_imm_int(&b, 0)),
                  nir_bcsel(&b, is_p1, nir_imm_int(&b, 2), nir_imm_int(&b, 1)),
                  nir_imm_int(&b, 3), flip_winding);
            }
            nir_push_else(&b, NULL);
            {
               if (shape->allow_whole_patch) {
                  nir_push_if(&b, nir_ieq_imm(&b, mode, 1));
                  {
                     /* Whole-Patch Quad primitive p. */
                     nir_def *s0 = n_out[0];
                     nir_def *s1 = nir_iadd(&b, s0, n_out[1]);
                     nir_def *s2 = nir_iadd(&b, s1, n_out[2]);
                     nir_def *s3 = nir_iadd(&b, s2, n_out[3]);
                     nir_def *iw = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 1));
                     nir_def *ih = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 1));
                     nir_def *qu = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 2));
                     nir_def *qv = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 2));
                     nir_def *ep1 = nir_iadd(&b, n_out[0], qv);
                     nir_def *ep2 = nir_iadd(&b, ep1, nir_iadd(&b, n_out[1], qu));
                     nir_def *ep3 = nir_iadd(&b, ep2, nir_iadd(&b, n_out[2], qv));
                     nir_def *ep4 = nir_iadd(&b, ep3, nir_iadd(&b, n_out[3], qu));
                     nir_push_if(&b, nir_ult(&b, p, ep4));
                     {
                        nir_def *e = nir_bcsel(
                           &b, nir_ult(&b, p, ep1), nir_imm_int(&b, 0),
                           nir_bcsel(&b, nir_ult(&b, p, ep2), nir_imm_int(&b, 1),
                                     nir_bcsel(&b, nir_ult(&b, p, ep3),
                                               nir_imm_int(&b, 2),
                                               nir_imm_int(&b, 3))));
                        nir_def *ep_prev = nir_bcsel(
                           &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
                           nir_bcsel(&b, nir_ieq_imm(&b, e, 1), ep1,
                                     nir_bcsel(&b, nir_ieq_imm(&b, e, 2), ep2, ep3)));
                        nir_def *t = nir_isub(&b, p, ep_prev);
                        nir_def *n_o = nir_bcsel(
                           &b, nir_ieq_imm(&b, e, 0), n_out[0],
                           nir_bcsel(&b, nir_ieq_imm(&b, e, 1), n_out[1],
                                     nir_bcsel(&b, nir_ieq_imm(&b, e, 2), n_out[2],
                                               n_out[3])));
                        nir_def *n_is = nir_bcsel(
                           &b, nir_ine_imm(&b, nir_iand_imm(&b, e, 1), 0), qu, qv);
                        nir_def *ii = nir_stitch_i(&b, n_o, n_is, t);
                        nir_def *oi = nir_isub(&b, t, ii);
                        nir_def *ii_next =
                           nir_stitch_i(&b, n_o, n_is, nir_iadd_imm(&b, t, 1));
                        nir_def *step_i = nir_ine(&b, ii_next, ii);
                        nir_def *ebase = nir_bcsel(
                           &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
                           nir_bcsel(&b, nir_ieq_imm(&b, e, 1), s0,
                                     nir_bcsel(&b, nir_ieq_imm(&b, e, 2), s1, s2)));
                        nir_def *ov0 = nir_umod(&b, nir_iadd(&b, ebase, oi), s3);
                        nir_def *ov1 = nir_umod(
                           &b, nir_iadd_imm(&b, nir_iadd(&b, ebase, oi), 1), s3);
                        /* Inner vertex index on interior grid (s3 + gy * iw + gx). */
                        nir_def *iv[2];
                        for (unsigned d = 0; d < 2; ++d) {
                           nir_def *k = d ? nir_iadd_imm(&b, ii, 1) : ii;
                           nir_def *gx = nir_bcsel(
                              &b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 1), k,
                                        nir_bcsel(&b, nir_ieq_imm(&b, e, 2),
                                                  nir_iadd_imm(&b, iw, -1),
                                                  nir_isub(&b, nir_iadd_imm(&b, iw, -1), k))));
                           nir_def *gy = nir_bcsel(
                              &b, nir_ieq_imm(&b, e, 0),
                              nir_isub(&b, nir_iadd_imm(&b, ih, -1), k),
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 1), nir_imm_int(&b, 0),
                                        nir_bcsel(&b, nir_ieq_imm(&b, e, 2), k,
                                                  nir_iadd_imm(&b, ih, -1))));
                           iv[d] = nir_iadd(&b, s3, nir_iadd(&b, nir_imul(&b, gy, iw), gx));
                        }
                        nir_def *i0 = nir_bcsel(&b, step_i, iv[0], ov0);
                        nir_def *i1 = nir_bcsel(&b, step_i, ov0, ov1);
                        nir_def *i2 = nir_bcsel(&b, step_i, iv[1], iv[0]);
                        emit_triangle_prim(&b, p, i0, i1, i2, flip_winding);
                     }
                     nir_push_else(&b, NULL);
                     {
                        nir_def *ip = nir_isub(&b, p, ep4);
                        nir_def *q = nir_ushr_imm(&b, ip, 1);
                        nir_def *tri = nir_iand_imm(&b, ip, 1);
                        nir_def *qu_safe = nir_umax(&b, qu, nir_imm_int(&b, 1));
                        nir_def *qx = nir_umod(&b, q, qu_safe);
                        nir_def *qy = nir_udiv(&b, q, qu_safe);
                        nir_def *v00 =
                           nir_iadd(&b, s3, nir_iadd(&b, nir_imul(&b, qy, iw), qx));
                        nir_def *v10 = nir_iadd_imm(&b, v00, 1);
                        nir_def *v01 = nir_iadd(&b, v00, iw);
                        nir_def *v11 = nir_iadd_imm(&b, v01, 1);
                        emit_quad_cell_tri(&b, p, v00, v10, v01, v11, tri, qx,
                                           qy, qu, qv, flip_winding);
                     }
                     nir_pop_if(&b, NULL);
                  }
                  nir_push_else(&b, NULL);
               }

               nir_push_if(&b, nir_ieq_imm(&b, mode, 2));
               {
                  /* Multi-WG Quad outer strip primitive p. */
                  nir_def *n_o = nir_load_var(&b, s_out_var);
                  nir_def *n_is = nir_load_var(&b, s_in_var);
                  nir_def *t = nir_iadd(&b, nir_load_var(&b, t0_var), p);
                  nir_def *o0 = nir_load_var(&b, o0_var);
                  nir_def *i0_b = nir_load_var(&b, i0_var);
                  nir_def *no = nir_load_var(&b, no_var);
                  nir_def *ii_abs = nir_stitch_i(&b, n_o, n_is, t);
                  nir_def *oi = nir_isub(&b, nir_isub(&b, t, ii_abs), o0);
                  nir_def *ii = nir_isub(&b, ii_abs, i0_b);
                  nir_def *step_i = nir_ine(
                     &b, nir_stitch_i(&b, n_o, n_is, nir_iadd_imm(&b, t, 1)),
                     ii_abs);
                  nir_def *iv0 = nir_iadd(&b, no, ii);
                  nir_def *i0 = nir_bcsel(&b, step_i, iv0, oi);
                  nir_def *i1 = nir_bcsel(&b, step_i, oi, nir_iadd_imm(&b, oi, 1));
                  nir_def *i2 = nir_bcsel(&b, step_i, nir_iadd_imm(&b, iv0, 1), iv0);
                  emit_triangle_prim(&b, p, i0, i1, i2, flip_winding);
               }
               nir_push_else(&b, NULL);
               {
                  /* Multi-WG Quad interior tile primitive p. */
                  nir_def *cw = nir_umax(&b, nir_load_var(&b, cw_var), nir_imm_int(&b, 1));
                  nir_def *vw = nir_iadd_imm(&b, cw, 1);
                  nir_def *q = nir_ushr_imm(&b, p, 1);
                  nir_def *tri = nir_iand_imm(&b, p, 1);
                  nir_def *qx = nir_umod(&b, q, cw);
                  nir_def *qy = nir_udiv(&b, q, cw);
                  nir_def *v00 = nir_iadd(&b, nir_imul(&b, qy, vw), qx);
                  nir_def *v10 = nir_iadd_imm(&b, v00, 1);
                  nir_def *v01 = nir_iadd(&b, v00, vw);
                  nir_def *v11 = nir_iadd_imm(&b, v01, 1);
                  nir_def *qu = nir_usub_sat(&b, n_in_u, nir_imm_int(&b, 2));
                  nir_def *qv = nir_usub_sat(&b, n_in_v, nir_imm_int(&b, 2));
                  nir_def *gx = nir_iadd(&b, nir_load_var(&b, ox_var), qx);
                  nir_def *gy = nir_iadd(&b, nir_load_var(&b, oy_var), qy);
                  emit_quad_cell_tri(&b, p, v00, v10, v01, v11, tri, gx, gy, qu,
                                     qv, flip_winding);
               }
               nir_pop_if(&b, NULL);

               if (shape->allow_whole_patch)
                  nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
         } else {
            /* TRIANGLES primitive emission. */
            nir_push_if(&b, nir_ior(&b, nir_ieq_imm(&b, mode, 0),
                                    nir_ieq_imm(&b, mode, 3)));
            {
               /* All-1s or odd center triangle: (0, 1, 2). */
               emit_triangle_prim(&b, p, nir_imm_int(&b, 0), nir_imm_int(&b, 1),
                                  nir_imm_int(&b, 2), flip_winding);
            }
            nir_push_else(&b, NULL);
            {
               if (shape->allow_whole_patch) {
                  nir_push_if(&b, nir_ieq_imm(&b, mode, 1));
                  {
                     /* Whole-Patch Triangle primitive p. */
                     nir_def *ni = n_in_u;
                     nir_def *outer_sum =
                        nir_iadd(&b, nir_iadd(&b, n_out[0], n_out[1]), n_out[2]);
                     nir_def *rings = nir_ushr_imm(&b, ni, 1);
                     nir_variable *rem = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_prem");
                     nir_variable *rv = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_pr");
                     nir_variable *ob = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_ob");
                     nir_variable *v0_out = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_i0");
                     nir_variable *v1_out = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_i1");
                     nir_variable *v2_out = nir_local_variable_create(
                        impl, glsl_uint_type(), "ft_wp_i2");
                     nir_store_var(&b, rem, p, 1);
                     nir_store_var(&b, rv, nir_imm_int(&b, 0), 1);
                     nir_store_var(&b, ob, nir_imm_int(&b, 0), 1);
                     nir_store_var(&b, v0_out, nir_imm_int(&b, 0), 1);
                     nir_store_var(&b, v1_out, nir_imm_int(&b, 1), 1);
                     nir_store_var(&b, v2_out, nir_imm_int(&b, 2), 1);

                     nir_loop *loop = nir_push_loop(&b);
                     {
                        nir_def *r = nir_load_var(&b, rv);
                        nir_def *ob_v = nir_load_var(&b, ob);
                        nir_push_if(&b, nir_uge(&b, r, rings));
                        {
                           /* Odd center triangle at ob_v .. ob_v + 2. */
                           nir_store_var(&b, v0_out, ob_v, 1);
                           nir_store_var(&b, v1_out, nir_iadd_imm(&b, ob_v, 1), 1);
                           nir_store_var(&b, v2_out, nir_iadd_imm(&b, ob_v, 2), 1);
                           nir_jump(&b, nir_jump_break);
                        }
                        nir_pop_if(&b, NULL);

                        nir_def *is_r0 = nir_ieq_imm(&b, r, 0);
                        nir_def *s_o_ring = nir_isub(&b, ni, nir_ishl_imm(&b, r, 1));
                        nir_def *s_i = nir_usub_sat(&b, s_o_ring, nir_imm_int(&b, 2));
                        nir_def *outer_rs =
                           nir_bcsel(&b, is_r0, outer_sum, nir_imul_imm(&b, s_o_ring, 3));
                        nir_def *inner_rs = nir_bcsel(
                           &b, nir_ieq_imm(&b, s_i, 0), nir_imm_int(&b, 1),
                           nir_imul_imm(&b, s_i, 3));
                        nir_def *t0_e =
                           nir_iadd(&b, nir_bcsel(&b, is_r0, n_out[0], s_o_ring), s_i);
                        nir_def *t1_e =
                           nir_iadd(&b, nir_bcsel(&b, is_r0, n_out[1], s_o_ring), s_i);
                        nir_def *t2_e =
                           nir_iadd(&b, nir_bcsel(&b, is_r0, n_out[2], s_o_ring), s_i);
                        nir_def *tp1 = t0_e;
                        nir_def *tp2 = nir_iadd(&b, tp1, t1_e);
                        nir_def *band_tris = nir_iadd(&b, tp2, t2_e);
                        nir_def *rem_v = nir_load_var(&b, rem);

                        nir_push_if(&b, nir_ult(&b, rem_v, band_tris));
                        {
                           nir_def *e = nir_bcsel(
                              &b, nir_ult(&b, rem_v, tp1), nir_imm_int(&b, 0),
                              nir_bcsel(&b, nir_ult(&b, rem_v, tp2),
                                        nir_imm_int(&b, 1), nir_imm_int(&b, 2)));
                           nir_def *t = nir_bcsel(
                              &b, nir_ieq_imm(&b, e, 0), rem_v,
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 1),
                                        nir_isub(&b, rem_v, tp1),
                                        nir_isub(&b, rem_v, tp2)));
                           nir_def *s_o = nir_bcsel(
                              &b, is_r0,
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 0), n_out[0],
                                        nir_bcsel(&b, nir_ieq_imm(&b, e, 1),
                                                  n_out[1], n_out[2])),
                              s_o_ring);
                           nir_def *ii = nir_stitch_i(&b, s_o, s_i, t);
                           nir_def *oi = nir_isub(&b, t, ii);
                           nir_def *step_i = nir_ine(
                              &b, nir_stitch_i(&b, s_o, s_i, nir_iadd_imm(&b, t, 1)),
                              ii);
                           nir_def *oe_start = nir_bcsel(
                              &b, is_r0,
                              nir_bcsel(&b, nir_ieq_imm(&b, e, 0), nir_imm_int(&b, 0),
                                        nir_bcsel(&b, nir_ieq_imm(&b, e, 1),
                                                  n_out[0],
                                                  nir_iadd(&b, n_out[0], n_out[1]))),
                              nir_imul(&b, e, s_o_ring));
                           nir_def *ie_start = nir_imul(&b, e, s_i);
                           nir_def *ib = nir_iadd(&b, ob_v, outer_rs);
                           nir_def *ov0 = nir_iadd(
                              &b, ob_v, nir_umod(&b, nir_iadd(&b, oe_start, oi), outer_rs));
                           nir_def *ov1 = nir_iadd(
                              &b, ob_v,
                              nir_umod(&b, nir_iadd_imm(&b, nir_iadd(&b, oe_start, oi), 1),
                                       outer_rs));
                           nir_def *iv0 = nir_iadd(
                              &b, ib,
                              nir_bcsel(&b, nir_ieq_imm(&b, inner_rs, 1),
                                        nir_imm_int(&b, 0),
                                        nir_umod(&b, nir_iadd(&b, ie_start, ii),
                                                 inner_rs)));
                           nir_def *iv1 = nir_iadd(
                              &b, ib,
                              nir_bcsel(&b, nir_ieq_imm(&b, inner_rs, 1),
                                        nir_imm_int(&b, 0),
                                        nir_umod(&b,
                                                 nir_iadd_imm(&b, nir_iadd(&b, ie_start, ii), 1),
                                                 inner_rs)));
                           nir_store_var(&b, v0_out, nir_bcsel(&b, step_i, iv0, ov0), 1);
                           nir_store_var(&b, v1_out, nir_bcsel(&b, step_i, ov0, ov1), 1);
                           nir_store_var(&b, v2_out, nir_bcsel(&b, step_i, iv1, iv0), 1);
                           nir_jump(&b, nir_jump_break);
                        }
                        nir_pop_if(&b, NULL);
                        nir_store_var(&b, rem, nir_isub(&b, rem_v, band_tris), 1);
                        nir_store_var(&b, ob, nir_iadd(&b, ob_v, outer_rs), 1);
                        nir_store_var(&b, rv, nir_iadd_imm(&b, r, 1), 1);
                     }
                     nir_pop_loop(&b, loop);

                     emit_triangle_prim(&b, p, nir_load_var(&b, v0_out),
                                        nir_load_var(&b, v1_out),
                                        nir_load_var(&b, v2_out), flip_winding);
                  }
                  nir_push_else(&b, NULL);
               }

               /* Multi-WG Triangle ring-edge strip primitive p. */
               nir_def *s_o = nir_load_var(&b, s_out_var);
               nir_def *s_i = nir_load_var(&b, s_in_var);
               nir_def *t = nir_iadd(&b, nir_load_var(&b, t0_var), p);
               nir_def *o0 = nir_load_var(&b, o0_var);
               nir_def *i0_b = nir_load_var(&b, i0_var);
               nir_def *no = nir_load_var(&b, no_var);
               nir_def *ii_abs = nir_stitch_i(&b, s_o, s_i, t);
               nir_def *oi = nir_isub(&b, nir_isub(&b, t, ii_abs), o0);
               nir_def *ii = nir_isub(&b, ii_abs, i0_b);
               nir_def *step_i = nir_ine(
                  &b, nir_stitch_i(&b, s_o, s_i, nir_iadd_imm(&b, t, 1)),
                  ii_abs);
               nir_def *iv0 = nir_iadd(&b, no, ii);
               nir_def *i0 = nir_bcsel(&b, step_i, iv0, oi);
               nir_def *i1 = nir_bcsel(&b, step_i, oi, nir_iadd_imm(&b, oi, 1));
               nir_def *i2 = nir_bcsel(&b, step_i, nir_iadd_imm(&b, iv0, 1), iv0);
               emit_triangle_prim(&b, p, i0, i1, i2, flip_winding);

               if (shape->allow_whole_patch)
                  nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
         }
      }
      nir_pop_if(&b, p_if);
   }

   /* ---- Single Shared TES Body Execution (Parallel Across V_live Lanes) ---- */
   nir_if *tes_active = nir_push_if(&b, nir_ult(&b, lane, v_live));
   tes_active->control = nir_selection_control_dont_flatten;
   {
      /* Initialize declared clip and cull distances to finite 0.0f before TES runs. */
      for (unsigned d = 0; d < clip_size + cull_size; ++d) {
         nir_store_per_vertex_output(
            &b, nir_imm_float(&b, 0.0f), lane, nir_imm_int(&b, 0),
            .component = d % 4, .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0 + d / 4,
                             .num_slots = 1});
      }
      if (export_primitive_id) {
         /* Export the original patch ID (including first_patch chunk offset) as
          * a flat linked mesh varying so FS gl_PrimitiveID observes the input
          * patch ID rather than the rasterizer's generated triangle ID (0x60). */
         nir_store_per_vertex_output(
            &b, patch_id, lane, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID,
                             .num_slots = 1});
      }
      b.cursor = nir_cf_reinsert(&tes_body, b.cursor);
   }
   nir_pop_if(&b, tes_active);

   struct tes_lower_state tes_state = {
      .pl = pl,
      .clip_size = clip_size,
      .cull_size = cull_size,
      .lane = lane,
      .uv = uv_val,
      .patch_id = patch_id,
   };
   NIR_PASS(_, tes, nir_shader_intrinsics_pass, lower_tes_intr,
            nir_metadata_none, &tes_state);

   NIR_PASS(_, tes, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out, NULL);
   memset(&tes->info.mesh, 0, sizeof(tes->info.mesh));
   tes->info.stage = MESA_SHADER_MESH;
   tes->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   tes->info.mesh.max_vertices_out = shape->max_v;
   tes->info.mesh.max_primitives_out = shape->max_p;
   tes->info.workgroup_size[0] = 32;
   tes->info.workgroup_size[1] = tes->info.workgroup_size[2] = 1;
   tes->info.workgroup_size_variable = false;
   tes->info.task_payload_size = pl->payload_B;
   tes->info.inputs_read = 0;
   tes->info.patch_inputs_read = 0;

   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, tes, nir_lower_global_vars_to_local);
   NIR_PASS(_, tes, nir_lower_vars_to_ssa);
   NIR_PASS(_, tes, nir_opt_copy_prop);
   NIR_PASS(_, tes, nir_opt_constant_folding);
   NIR_PASS(_, tes, nir_opt_dce);
   NIR_PASS(_, tes, nir_opt_dead_cf);
   NIR_PASS(_, tes, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out, NULL);
   nir_shader_gather_info(tes, impl);
   tes->info.clip_distance_array_size = clip_size;
   tes->info.cull_distance_array_size = cull_size;
   tes->info.task_payload_size = pl->payload_B;
   nir_validate_shader(tes, "fused MESH(TES) tessellation lowering");
   return tes;
}

/* =========================================================================
 * Strip Unread TES Generic Outputs Against Linked FS Mask
 * ========================================================================= */

static bool
strip_unread_tes_var_outputs(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   const uint8_t *fs_mask = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned loc = nir_intrinsic_io_semantics(intr).location +
                  nir_src_as_uint(intr->src[1]);
   if (loc == VARYING_SLOT_PSIZ) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (loc < VARYING_SLOT_VAR0 || loc >= VARYING_SLOT_VAR0 + 32)
      return false;
   unsigned comp = nir_intrinsic_component(intr);
   unsigned wmask = nir_intrinsic_write_mask(intr);
   unsigned live_mask =
      wmask & ((unsigned)fs_mask[loc - VARYING_SLOT_VAR0] >> comp);
   if (live_mask == wmask)
      return false;
   if (!live_mask) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   nir_intrinsic_set_write_mask(intr, live_mask);
   return true;
}

/* =========================================================================
 * Narrow Synthetic TCS RMW Vector Stores (nir_lower_array_deref_of_vec)
 *
 * hk_shader_nir runs nir_lower_array_deref_of_vec on TCS outputs so scalar
 * writes to vec4 outputs become load_per_vertex_output + vec4 + store.
 * Stripping unmodified passthrough channels eliminates those synthetic RMW
 * loads cleanly while leaving genuine cross-invocation reads intact.
 * ========================================================================= */

static bool
same_tcs_vertex_index(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   nir_instr *ia = nir_def_instr(a);
   nir_instr *ib = nir_def_instr(b);
   if (ia->type == nir_instr_type_load_const &&
       ib->type == nir_instr_type_load_const) {
      nir_load_const_instr *ca = nir_instr_as_load_const(ia);
      nir_load_const_instr *cb = nir_instr_as_load_const(ib);
      return ca->def.bit_size == cb->def.bit_size &&
             ca->value[0].u64 == cb->value[0].u64;
   }
   if (ia->type == nir_instr_type_intrinsic &&
       ib->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(ia)->intrinsic == nir_intrinsic_load_invocation_id &&
       nir_instr_as_intrinsic(ib)->intrinsic == nir_intrinsic_load_invocation_id)
      return true;
   return false;
}

static void
resolve_tcs_store_channel(nir_def *val, unsigned ch, nir_def **out_def,
                          unsigned *out_comp)
{
   for (unsigned depth = 0; depth < 8; ++depth) {
      nir_instr *instr = nir_def_instr(val);
      if (instr->type != nir_instr_type_alu)
         break;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (nir_op_is_vec(alu->op) && ch < nir_op_infos[alu->op].num_inputs) {
         val = alu->src[ch].src.ssa;
         ch = alu->src[ch].swizzle[0];
      } else if (alu->op == nir_op_mov && ch < alu->def.num_components) {
         val = alu->src[0].src.ssa;
         ch = alu->src[0].swizzle[ch];
      } else {
         break;
      }
   }
   *out_def = val;
   *out_comp = ch;
}

static bool
no_intervening_tcs_write(nir_instr *from, nir_instr *to, unsigned loc,
                         unsigned channel, bool per_vertex)
{
   if (from->block != to->block)
      return false;
   unsigned ch_bit = BITFIELD_BIT(channel);
   for (nir_instr *cur = nir_instr_next(from); cur && cur != to;
        cur = nir_instr_next(cur)) {
      if (cur->type == nir_instr_type_call)
         return false;
      if (cur->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(cur);
      if (intr->intrinsic == nir_intrinsic_barrier)
         return false;
      if (intr->intrinsic == nir_intrinsic_store_deref &&
          nir_deref_mode_may_be(nir_src_as_deref(intr->src[0]),
                                nir_var_shader_out))
         return false;
      if (per_vertex && intr->intrinsic == nir_intrinsic_store_per_vertex_output) {
         if (!nir_src_is_const(intr->src[2]))
            return false;
         unsigned sloc = nir_intrinsic_io_semantics(intr).location +
                         nir_src_as_uint(intr->src[2]);
         if (sloc == loc) {
            unsigned smask =
               (nir_intrinsic_write_mask(intr) &
                nir_component_mask(intr->num_components))
               << nir_intrinsic_component(intr);
            if (smask & ch_bit)
               return false;
         }
      } else if (!per_vertex && intr->intrinsic == nir_intrinsic_store_output) {
         if (!nir_src_is_const(intr->src[1]))
            return false;
         unsigned sloc = nir_intrinsic_io_semantics(intr).location +
                         nir_src_as_uint(intr->src[1]);
         if (sloc == loc) {
            unsigned smask =
               (nir_intrinsic_write_mask(intr) &
                nir_component_mask(intr->num_components))
               << nir_intrinsic_component(intr);
            if (smask & ch_bit)
               return false;
         }
      }
   }
   return true;
}

static bool
narrow_tcs_rmw_stores(nir_shader *tcs)
{
   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(tcs);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
         bool pv;
         unsigned off_src;
         if (st->intrinsic == nir_intrinsic_store_per_vertex_output) {
            pv = true;
            off_src = 2;
         } else if (st->intrinsic == nir_intrinsic_store_output) {
            pv = false;
            off_src = 1;
         } else {
            continue;
         }
         if (!nir_src_is_const(st->src[off_src]))
            continue;
         unsigned loc = nir_intrinsic_io_semantics(st).location +
                        nir_src_as_uint(st->src[off_src]);
         unsigned comp = nir_intrinsic_component(st);
         unsigned ncomps = st->num_components;
         unsigned full_mask = nir_component_mask(ncomps);
         unsigned wmask = nir_intrinsic_write_mask(st) & full_mask;

         unsigned new_mask = wmask;
         u_foreach_bit(i, wmask) {
            nir_def *ch = NULL;
            unsigned swiz = 0;
            resolve_tcs_store_channel(st->src[0].ssa, i, &ch, &swiz);
            nir_instr *ch_instr = nir_def_instr(ch);
            if (ch_instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *ld = nir_instr_as_intrinsic(ch_instr);
            if (pv) {
               if (ld->intrinsic != nir_intrinsic_load_per_vertex_output ||
                   !nir_src_is_const(ld->src[1]))
                  continue;
               unsigned lloc = nir_intrinsic_io_semantics(ld).location +
                               nir_src_as_uint(ld->src[1]);
               if (lloc != loc ||
                   nir_intrinsic_component(ld) + swiz != comp + i ||
                   !same_tcs_vertex_index(ld->src[0].ssa, st->src[1].ssa))
                  continue;
            } else {
               if (ld->intrinsic != nir_intrinsic_load_output ||
                   !nir_src_is_const(ld->src[0]))
                  continue;
               unsigned lloc = nir_intrinsic_io_semantics(ld).location +
                               nir_src_as_uint(ld->src[0]);
               if (lloc != loc ||
                   nir_intrinsic_component(ld) + swiz != comp + i)
                  continue;
            }
            if (no_intervening_tcs_write(&ld->instr, &st->instr, loc, comp + i,
                                         pv))
               new_mask &= ~BITFIELD_BIT(i);
         }

         if (!new_mask) {
            nir_instr_remove(&st->instr);
            progress = true;
            continue;
         }

         unsigned first = ffs(new_mask) - 1;
         unsigned last = util_last_bit(new_mask) - 1;
         unsigned span = last - first + 1;
         bool need_rewrite =
            (new_mask != wmask) || (first > 0) || (span < ncomps);
         if (!need_rewrite && new_mask != full_mask) {
            /* Strip masked-off lanes that still keep dead RMW loads alive in
             * the store's source vector (e.g. wrmask=xz with load in .y). */
            unsigned dead_mask = full_mask & ~new_mask;
            u_foreach_bit(i, dead_mask) {
               nir_def *ch = NULL;
               unsigned swiz = 0;
               resolve_tcs_store_channel(st->src[0].ssa, i, &ch, &swiz);
               if (nir_def_instr(ch)->type != nir_instr_type_undef) {
                  need_rewrite = true;
                  break;
               }
            }
         }
         if (need_rewrite) {
            nir_builder b = nir_builder_at(nir_before_instr(&st->instr));
            if (util_bitcount(new_mask) == 1) {
               nir_def *ch = NULL;
               unsigned swiz = 0;
               resolve_tcs_store_channel(st->src[0].ssa, first, &ch, &swiz);
               nir_def *scalar = nir_channel(&b, ch, swiz);
               nir_src_rewrite(&st->src[0], scalar);
               st->num_components = 1;
               nir_intrinsic_set_component(st, comp + first);
               nir_intrinsic_set_write_mask(st, 1);
            } else {
               nir_def *elems[4];
               unsigned bit_size = st->src[0].ssa->bit_size;
               for (unsigned j = 0; j < span; ++j) {
                  if (new_mask & BITFIELD_BIT(first + j)) {
                     nir_def *ch = NULL;
                     unsigned swiz = 0;
                     resolve_tcs_store_channel(st->src[0].ssa, first + j, &ch,
                                               &swiz);
                     elems[j] = nir_channel(&b, ch, swiz);
                  } else {
                     elems[j] = nir_undef(&b, 1, bit_size);
                  }
               }
               nir_def *packed = nir_vec(&b, elems, span);
               nir_src_rewrite(&st->src[0], packed);
               st->num_components = span;
               nir_intrinsic_set_component(st, comp + first);
               nir_intrinsic_set_write_mask(st, new_mask >> first);
            }
            progress = true;
         }
      }
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/* =========================================================================
 * Public Entry Point
 * ========================================================================= */

bool
hk_nir_tess_fused_lower(const nir_shader *vs_source,
                        const nir_shader *tcs_source,
                        const nir_shader *tes_source,
                        const struct hk_tess_fused_options *options,
                        nir_shader **out_object, nir_shader **out_mesh,
                        struct hk_tess_fused_info *out_info,
                        const char **reason)
{
   if (reason)
      *reason = NULL;
   if (out_object)
      *out_object = NULL;
   if (out_mesh)
      *out_mesh = NULL;

   if (!vs_source || !tcs_source || !tes_source ||
       tcs_source->info.stage != MESA_SHADER_TESS_CTRL) {
      if (reason)
         *reason = "fused tess requires lowered VS, TCS, and TES NIR";
      return false;
   }

   nir_shader *tcs_clean = nir_shader_clone(NULL, tcs_source);
   if (!tcs_clean) {
      if (reason)
         *reason = "out of memory cloning TCS";
      return false;
   }
   if (narrow_tcs_rmw_stores(tcs_clean)) {
      NIR_PASS(_, tcs_clean, nir_opt_dce);
      NIR_PASS(_, tcs_clean, nir_opt_copy_prop);
      NIR_PASS(_, tcs_clean, nir_opt_constant_folding);
      NIR_PASS(_, tcs_clean, nir_opt_dce);
   }

   enum tess_primitive_mode domain;
   enum gl_tess_spacing spacing;
   bool ccw;
   unsigned out_v;
   if (!validate_stages(vs_source, tcs_clean, tes_source, options, &domain,
                        &spacing, &ccw, &out_v, reason)) {
      ralloc_free(tcs_clean);
      return false;
   }

   nir_shader *vs = nir_shader_clone(NULL, vs_source);
   nir_shader *tcs = tcs_clean;
   nir_shader *tes = nir_shader_clone(NULL, tes_source);
   if (!vs || !tcs || !tes) {
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      if (reason)
         *reason = "out of memory cloning fused tess stages";
      return false;
   }

   if (!hk_nir_fused_prepare_vs(vs, options ? &options->vertex : NULL, reason)) {
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      return false;
   }

   NIR_PASS(_, tcs, nir_lower_returns);
   NIR_PASS(_, tcs, nir_opt_constant_folding);
   NIR_PASS(_, tcs, nir_opt_dce);
   if (!validate_tcs_barriers(tcs, reason)) {
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      return false;
   }

   /* Strip TES generic outputs unused by the linked FS so both varying planes
    * and TCS->TES payload footprints shrink to the exact live set. */
   unsigned clip_size = 0, cull_size = 0;
   scan_tes_clip_cull_sizes(tes, &clip_size, &cull_size);
   if (options && options->has_fs_inputs_mask) {
      NIR_PASS(_, tes, nir_shader_intrinsics_pass, strip_unread_tes_var_outputs,
               nir_metadata_control_flow, (void *)options->fs_var_read_mask);
      NIR_PASS(_, tes, nir_opt_dce);
      NIR_PASS(_, tes, nir_opt_dead_cf);
   }

   /* Compute exact FS-linked TES output component count. */
   uint8_t out_masks[VARYING_SLOT_MAX] = {0};
   nir_foreach_block(block, nir_shader_get_entrypoint(tes)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         if (loc == VARYING_SLOT_PSIZ || loc == VARYING_SLOT_CULL_DIST0 ||
             loc == VARYING_SLOT_CULL_DIST1)
            continue;
         out_masks[loc] |= wmask << comp;
      }
   }
   const bool export_primitive_id =
      options && options->fs_reads_primitive_id;
   if (export_primitive_id)
      out_masks[VARYING_SLOT_PRIMITIVE_ID] |= 1;

   unsigned vertex_components = 4; /* VARYING_SLOT_POS */
   for (unsigned loc = VARYING_SLOT_VAR0; loc < VARYING_SLOT_VAR0 + 32; ++loc)
      vertex_components += util_bitcount(out_masks[loc]);
   if ((out_masks[VARYING_SLOT_LAYER] & 1) ||
       (out_masks[VARYING_SLOT_VIEWPORT] & 1))
      vertex_components += 1;
   if (out_masks[VARYING_SLOT_VIEWPORT] & 1)
      vertex_components += 1;
   if (out_masks[VARYING_SLOT_PRIMITIVE_ID] & 1)
      vertex_components += 1;
   vertex_components += clip_size;
   if (options && options->fs_reads_clip_distance)
      vertex_components += clip_size;

   if (vertex_components > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS) {
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      if (reason)
         *reason = "fused tess TES outputs exceed Apple9 UVS varying capacity";
      return false;
   }

   struct tess_payload_layout pl;
   compute_payload_layout(tes, out_v, &pl);
   unsigned max_payload =
      options && options->max_payload_bytes ? options->max_payload_bytes : 1024u;
   if (pl.payload_B > max_payload ||
       pl.payload_B > AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES) {
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      if (reason)
         *reason = "fused tess per-parent payload exceeds hardware slot stride";
      return false;
   }

   struct tess_meshlet_shape shape = choose_meshlet_shape(vertex_components);

   /* Compute exact static bounds, conservative VS+TCS interval upper bounds,
    * and push-constant level/cap metadata for draw-time TVB/ring bounds. */
   struct hk_apple9_tess_level_src level_srcs[6];
   bool const_levels = false;
   bool have_level_srcs =
      detect_tess_level_sources(tcs, domain, level_srcs, &const_levels);
   struct hk_apple9_tess_push_levels push_levels = {0};
   unsigned max_child_wgs = 0, max_patch_prims = 0;
   if (have_level_srcs && const_levels) {
      float const_outer[4] = {
         level_srcs[0].const_value, level_srcs[1].const_value,
         level_srcs[2].const_value, level_srcs[3].const_value,
      };
      float const_inner[2] = {
         level_srcs[4].const_value, level_srcs[5].const_value,
      };
      cpu_patch_counts(domain, spacing, const_outer, const_inner, &shape,
                       &max_child_wgs, &max_patch_prims);
   } else if (have_level_srcs) {
      const float max_f = spacing == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f;
      float hi_o[4], hi_i[2];
      for (unsigned e = 0; e < 4; ++e)
         hi_o[e] = level_srcs[e].kind == HK_APPLE9_TESS_LEVEL_CONST
                      ? level_srcs[e].const_value
                      : max_f;
      for (unsigned a = 0; a < 2; ++a)
         hi_i[a] = level_srcs[4u + a].kind == HK_APPLE9_TESS_LEVEL_CONST
                      ? level_srcs[4u + a].const_value
                      : max_f;
      cpu_patch_counts(domain, spacing, hi_o, hi_i, &shape, &max_child_wgs,
                       &max_patch_prims);
      push_levels = (struct hk_apple9_tess_push_levels){
         .valid = true,
         .domain = domain,
         .spacing = spacing,
         .shape_max_v = shape.max_v,
         .shape_max_p = shape.max_p,
         .shape_p_strip = shape.p_strip,
         .shape_tile_w = shape.tile_w,
         .shape_tile_h = shape.tile_h,
         .shape_allow_whole = shape.allow_whole_patch,
      };
      memcpy(push_levels.levels, level_srcs, sizeof(push_levels.levels));
   } else {
      struct hk_apple9_tess_level_src bound_srcs[6];
      bool has_push_bound = false;
      analyze_tess_level_bounds(vs, tcs, domain, spacing, bound_srcs,
                                &has_push_bound);
      float hi_o[4] = {
         bound_srcs[0].const_value, bound_srcs[1].const_value,
         bound_srcs[2].const_value, bound_srcs[3].const_value,
      };
      float hi_i[2] = {
         bound_srcs[4].const_value, bound_srcs[5].const_value,
      };
      cpu_patch_counts(domain, spacing, hi_o, hi_i, &shape, &max_child_wgs,
                       &max_patch_prims);
      if (has_push_bound) {
         push_levels = (struct hk_apple9_tess_push_levels){
            .valid = true,
            .domain = domain,
            .spacing = spacing,
            .shape_max_v = shape.max_v,
            .shape_max_p = shape.max_p,
            .shape_p_strip = shape.p_strip,
            .shape_tile_w = shape.tile_w,
            .shape_tile_h = shape.tile_h,
            .shape_allow_whole = shape.allow_whole_patch,
         };
         memcpy(push_levels.levels, bound_srcs, sizeof(push_levels.levels));
      }
   }

   nir_shader *obj =
      build_object_shader(vs, tcs, domain, spacing, &pl, &shape, reason);
   ralloc_free(vs);
   if (!obj) {
      ralloc_free(tcs);
      ralloc_free(tes);
      return false;
   }

   nir_shader *mesh =
      build_mesh_shader(tes, domain, spacing, &pl, &shape, clip_size, cull_size,
                        export_primitive_id);
   if (!mesh) {
      ralloc_free(obj);
      ralloc_free(tes);
      return false;
   }

   if (out_info) {
      *out_info = (struct hk_tess_fused_info){
         .domain = domain,
         .spacing = spacing,
         .ccw = ccw,
         .tcs_vertices_out = out_v,
         .payload_B = pl.payload_B,
         .max_child_wgs_per_patch = max_child_wgs,
         .max_prims_per_patch = max_patch_prims,
         .mesh_max_vertices = shape.max_v,
         .mesh_max_primitives = shape.max_p,
         .constant_tess_levels = const_levels,
         .push_levels = push_levels,
      };
   }
   if (out_object)
      *out_object = obj;
   else
      ralloc_free(obj);
   if (out_mesh)
      *out_mesh = mesh;
   else
      ralloc_free(mesh);
   return true;
}

/* =========================================================================
 * Bit-Exact 0-ULP Lossless Narrow-VBO Passthrough Varying Packing
 *
 * For monolithic fused tessellation pipelines whose vertex attribute formats
 * are statically fixed (!vertex->use_prolog), identify TES->FS varyings whose
 * live components are unmodified patch-constant passthroughs of narrow
 * normalized VBO attributes (UNORM8/SNORM8, or 2-component UNORM16/SNORM16).
 *
 * Each qualifying TES->FS varying slot packs its 2..4 live float32 channels
 * into a single flat uint32 varying via pack_{u,s}norm_{4x8,2x16} and unpacks
 * identically in the paired native FS via unpack_{u,s}norm_{4x8,2x16} over
 * existing 32-bit ITER_FLAT hardware (0-ULP roundtrip with hardware vertex
 * input decode, handling arbitrary format/component swizzles and partial masks).
 * When a packed slot's TCS->TES per-vertex slot is exclusively consumed by
 * that pack, packing is hoisted into TCS so the 0xe0/0xe1 task payload shrinks
 * by 12 B per packed vec4 as well.
 * ========================================================================= */

enum vbo_norm_kind {
   VBO_NORM_NONE = 0,
   VBO_NORM_UNORM8,
   VBO_NORM_SNORM8,
   VBO_NORM_UNORM16,
   VBO_NORM_SNORM16,
};

static enum vbo_norm_kind
classify_vbo_norm_format(enum pipe_format fmt)
{
   switch (fmt) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R8G8B8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8_UNORM:
      return VBO_NORM_UNORM8;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_B8G8R8A8_SNORM:
   case PIPE_FORMAT_R8G8B8_SNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8_SNORM:
      return VBO_NORM_SNORM8;
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16_UNORM:
   case PIPE_FORMAT_R16_UNORM:
      return VBO_NORM_UNORM16;
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16_SNORM:
   case PIPE_FORMAT_R16_SNORM:
      return VBO_NORM_SNORM16;
   default:
      return VBO_NORM_NONE;
   }
}

static bool
is_load_invocation_id(nir_def *def)
{
   nir_scalar s = nir_scalar_resolved(def, 0);
   return nir_def_instr(s.def)->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(nir_def_instr(s.def))->intrinsic ==
             nir_intrinsic_load_invocation_id;
}

static bool
is_same_tcs_vertex(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   if (is_load_invocation_id(a) && is_load_invocation_id(b))
      return true;
   nir_scalar sa = nir_scalar_resolved(a, 0);
   nir_scalar sb = nir_scalar_resolved(b, 0);
   return nir_scalar_is_const(sa) && nir_scalar_is_const(sb) &&
          nir_scalar_as_uint(sa) == nir_scalar_as_uint(sb);
}

static void
gather_info_preserving_clip_cull(nir_shader *nir)
{
   unsigned clip = nir->info.clip_distance_array_size;
   unsigned cull = nir->info.cull_distance_array_size;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = clip;
   nir->info.cull_distance_array_size = cull;
}

static nir_def *
emit_vbo_norm_pack_n(nir_builder *b, enum vbo_norm_kind kind, unsigned count,
                     nir_def *vals[4])
{
   if (kind == VBO_NORM_UNORM8 || kind == VBO_NORM_SNORM8) {
      assert(count >= 2 && count <= 4);
      nir_def *ch[4];
      for (unsigned c = 0; c < 4; ++c)
         ch[c] = c < count ? vals[c] : nir_imm_float(b, 0.0f);
      nir_def *v4 = nir_vec4(b, ch[0], ch[1], ch[2], ch[3]);
      return kind == VBO_NORM_SNORM8 ? nir_pack_snorm_4x8(b, v4)
                                     : nir_pack_unorm_4x8(b, v4);
   }
   assert(count == 2);
   nir_def *v2 = nir_vec2(b, vals[0], vals[1]);
   return kind == VBO_NORM_SNORM16 ? nir_pack_snorm_2x16(b, v2)
                                   : nir_pack_unorm_2x16(b, v2);
}

struct tes_comp_origin {
   bool valid;
   bool is_patch;
   enum vbo_norm_kind kind;
   unsigned tcs_loc;
   unsigned tcs_comp;
   unsigned cp_idx;
};

struct vbo_pack_bucket {
   enum vbo_norm_kind kind;
   unsigned count; /* 2..4 for 8-bit norms, 2 for 16-bit norms */
   unsigned src_var[4];
   unsigned src_comp[4];
   struct tes_comp_origin origin[4];
   unsigned dst_var;  /* VARYING_SLOT_VAR0 + dst_var */
   unsigned dst_comp; /* 0..3 */
   nir_intrinsic_instr *pack_store;
};

struct comp_pack_ref {
   bool packed;
   unsigned bucket_idx;
   unsigned lane; /* 0..3 within unpacked vec */
};

bool
hk_nir_tess_fused_pack_vbo_varyings(
   nir_shader *vs, nir_shader *tcs, nir_shader *tes, nir_shader *fs,
   const struct agx_apple9_vertex_layout *vertex)
{
   if (!vs || !tcs || !tes || !fs || !vertex || vertex->use_prolog)
      return false;
   if (!vs->info.io_lowered || !tcs->info.io_lowered || !tes->info.io_lowered ||
       !fs->info.io_lowered)
      return false;

   /* Narrow identity TCS self-RMW output stores (common in DXIL/SPIR-V)
    * before tracing so per-component RMW chains collapse to direct VS->TCS
    * passthroughs. */
   if (narrow_tcs_rmw_stores(tcs)) {
      NIR_PASS(_, tcs, nir_opt_dce);
      NIR_PASS(_, tcs, nir_opt_copy_prop);
      NIR_PASS(_, tcs, nir_opt_constant_folding);
      NIR_PASS(_, tcs, nir_opt_dce);
   }

   /* 1. Trace VS outputs back to statically-typed narrow normalized VBO loads. */
   enum vbo_norm_kind vs_kind[VARYING_SLOT_MAX][4] = {{VBO_NORM_NONE}};
   bool vs_bad[VARYING_SLOT_MAX][4] = {{false}};
   nir_foreach_block(block, nir_shader_get_entrypoint(vs)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         if (!nir_src_is_const(intr->src[1]))
            return false;
         uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc >= VARYING_SLOT_MAX)
            continue;
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c >= 4)
               continue;
            enum vbo_norm_kind k = VBO_NORM_NONE;
            if (intr->src[0].ssa->bit_size == 32 &&
                nir_intrinsic_src_type(intr) == nir_type_float32) {
               nir_scalar s = nir_scalar_resolved(intr->src[0].ssa, i);
               nir_instr *def_i = nir_def_instr(s.def);
               if (def_i->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *ld = nir_instr_as_intrinsic(def_i);
                  if (ld->intrinsic == nir_intrinsic_load_input &&
                      ld->def.bit_size == 32 &&
                      nir_intrinsic_dest_type(ld) == nir_type_float32 &&
                      nir_src_is_const(ld->src[0]) &&
                      nir_src_as_uint(ld->src[0]) == 0) {
                     unsigned in_loc = nir_intrinsic_io_semantics(ld).location;
                     if (in_loc >= VERT_ATTRIB_GENERIC0 &&
                         in_loc < VERT_ATTRIB_GENERIC0 + 16) {
                        unsigned attr = in_loc - VERT_ATTRIB_GENERIC0;
                        k = classify_vbo_norm_format(vertex->format[attr]);
                     }
                  }
               }
            }
            if (k == VBO_NORM_NONE) {
               vs_bad[loc][c] = true;
            } else if (vs_kind[loc][c] == VBO_NORM_NONE) {
               vs_kind[loc][c] = k;
            } else if (vs_kind[loc][c] != k) {
               vs_bad[loc][c] = true;
            }
         }
      }
   }

   /* 2. Propagate narrow normalized value domains across TCS passthroughs. */
   enum vbo_norm_kind tcs_pv_kind[VARYING_SLOT_MAX][4] = {{VBO_NORM_NONE}};
   bool tcs_pv_bad[VARYING_SLOT_MAX][4] = {{false}};
   enum vbo_norm_kind tcs_patch_kind[VARYING_SLOT_MAX][4] = {{VBO_NORM_NONE}};
   bool tcs_patch_bad[VARYING_SLOT_MAX][4] = {{false}};
   bool tcs_reads_pv_out[VARYING_SLOT_MAX] = {false};

   nir_foreach_block(block, nir_shader_get_entrypoint(tcs)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_per_vertex_output) {
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (!nir_src_is_const(intr->src[1])) {
               for (unsigned s = 0; s < sem.num_slots; ++s)
                  if (sem.location + s < VARYING_SLOT_MAX)
                     tcs_reads_pv_out[sem.location + s] = true;
            } else {
               unsigned loc = sem.location + nir_src_as_uint(intr->src[1]);
               if (loc < VARYING_SLOT_MAX)
                  tcs_reads_pv_out[loc] = true;
            }
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_store_per_vertex_output &&
             intr->intrinsic != nir_intrinsic_store_output)
            continue;
         bool is_pv = intr->intrinsic == nir_intrinsic_store_per_vertex_output;
         nir_src *off_src = is_pv ? &intr->src[2] : &intr->src[1];
         if (!nir_src_is_const(*off_src))
            return false;
         uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(*off_src);
         if (loc >= VARYING_SLOT_MAX)
            continue;
         bool vtx_ok = !is_pv || is_load_invocation_id(intr->src[1].ssa) ||
                       nir_src_is_const(intr->src[1]);
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         enum vbo_norm_kind (*dst_kind)[4] = is_pv ? tcs_pv_kind : tcs_patch_kind;
         bool (*dst_bad)[4] = is_pv ? tcs_pv_bad : tcs_patch_bad;
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c >= 4)
               continue;
            enum vbo_norm_kind k = VBO_NORM_NONE;
            if (vtx_ok && intr->src[0].ssa->bit_size == 32 &&
                nir_intrinsic_src_type(intr) == nir_type_float32) {
               nir_scalar s = nir_scalar_resolved(intr->src[0].ssa, i);
               nir_instr *def_i = nir_def_instr(s.def);
               if (def_i->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *ld = nir_instr_as_intrinsic(def_i);
                  if (ld->intrinsic == nir_intrinsic_load_per_vertex_input &&
                      ld->def.bit_size == 32 &&
                      nir_intrinsic_dest_type(ld) == nir_type_float32 &&
                      nir_src_is_const(ld->src[1])) {
                     uint64_t vs_loc =
                        (uint64_t)nir_intrinsic_io_semantics(ld).location +
                        nir_src_as_uint(ld->src[1]);
                     unsigned vs_comp = nir_intrinsic_component(ld) + s.comp;
                     if (vs_loc < VARYING_SLOT_MAX && vs_comp < 4 &&
                         !vs_bad[vs_loc][vs_comp])
                        k = vs_kind[vs_loc][vs_comp];
                  }
               }
            }
            if (k == VBO_NORM_NONE) {
               dst_bad[loc][c] = true;
            } else if (dst_kind[loc][c] == VBO_NORM_NONE) {
               dst_kind[loc][c] = k;
            } else if (dst_kind[loc][c] != k) {
               dst_bad[loc][c] = true;
            }
         }
      }
   }

   /* 3. Inspect FS inputs: gather exact per-component read masks and reject any
    *    slot with indirect offsets, per-vertex loads, or non-float32 reads. */
   uint8_t fs_read_mask[32] = {0};
   bool fs_slot_bad[32] = {false};
   nir_foreach_block(block, nir_shader_get_entrypoint(fs)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_input &&
             intr->intrinsic != nir_intrinsic_load_interpolated_input &&
             intr->intrinsic != nir_intrinsic_load_input_vertex &&
             intr->intrinsic != nir_intrinsic_load_per_vertex_input)
            continue;
         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         nir_src *off = nir_get_io_offset_src(intr);
         if (!off || !nir_src_is_const(*off)) {
            for (unsigned s = 0; s < sem.num_slots; ++s) {
               unsigned loc = sem.location + s;
               if (loc >= VARYING_SLOT_VAR0 && loc < VARYING_SLOT_VAR0 + 32) {
                  fs_slot_bad[loc - VARYING_SLOT_VAR0] = true;
                  fs_read_mask[loc - VARYING_SLOT_VAR0] = 0xf;
               }
            }
            continue;
         }
         unsigned loc = sem.location + nir_src_as_uint(*off);
         if (loc < VARYING_SLOT_VAR0 || loc >= VARYING_SLOT_VAR0 + 32)
            continue;
         unsigned var = loc - VARYING_SLOT_VAR0;
         if ((intr->intrinsic != nir_intrinsic_load_input &&
              intr->intrinsic != nir_intrinsic_load_interpolated_input) ||
             intr->def.bit_size != 32 ||
             nir_intrinsic_dest_type(intr) != nir_type_float32) {
            fs_slot_bad[var] = true;
            continue;
         }
         unsigned comp = nir_intrinsic_component(intr);
         nir_component_mask_t read = nir_def_components_read(&intr->def);
         fs_read_mask[var] |= (uint8_t)(read << comp);
      }
   }

   /* 4. Inspect TES outputs per-component: collect every live scalar component
    *    that is unconditionally written (top-level CF, no early returns) with a
    *    patch-constant TCS passthrough of a narrow normalized kind. */
   struct tes_comp_origin origins[32][4] = {{{0}}};
   uint8_t tes_written_any[32] = {0};
   uint8_t tes_comp_bad[32] = {0};
   bool tes_slot_readback[32] = {false};
   nir_intrinsic_instr *tes_anchor_store = NULL;

   nir_foreach_block(block, nir_shader_get_entrypoint(tes)) {
      bool top_level = block->cf_node.parent->type == nir_cf_node_function;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump &&
             nir_instr_as_jump(instr)->type == nir_jump_return)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_output) {
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            for (unsigned s = 0; s < sem.num_slots; ++s) {
               unsigned loc = sem.location + s;
               if (loc >= VARYING_SLOT_VAR0 && loc < VARYING_SLOT_VAR0 + 32)
                  tes_slot_readback[loc - VARYING_SLOT_VAR0] = true;
            }
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         if (!nir_src_is_const(intr->src[1]))
            return false;
         uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc < VARYING_SLOT_VAR0 || loc >= VARYING_SLOT_VAR0 + 32)
            continue;
         unsigned var = (unsigned)(loc - VARYING_SLOT_VAR0);
         if (top_level)
            tes_anchor_store = intr;
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c >= 4)
               continue;
            bool dup = !!(tes_written_any[var] & BITFIELD_BIT(c));
            tes_written_any[var] |= BITFIELD_BIT(c);
            if (!(fs_read_mask[var] & BITFIELD_BIT(c)))
               continue;
            if (!top_level || dup || intr->src[0].ssa->bit_size != 32 ||
                nir_intrinsic_src_type(intr) != nir_type_float32) {
               tes_comp_bad[var] |= BITFIELD_BIT(c);
               continue;
            }
            nir_scalar s = nir_scalar_resolved(intr->src[0].ssa, i);
            nir_instr *def_i = nir_def_instr(s.def);
            if (def_i->type != nir_instr_type_intrinsic) {
               tes_comp_bad[var] |= BITFIELD_BIT(c);
               continue;
            }
            nir_intrinsic_instr *ld = nir_instr_as_intrinsic(def_i);
            if (ld->intrinsic == nir_intrinsic_load_per_vertex_input &&
                ld->def.bit_size == 32 &&
                nir_intrinsic_dest_type(ld) == nir_type_float32 &&
                nir_src_is_const(ld->src[0]) && nir_src_is_const(ld->src[1])) {
               uint64_t tcs_loc =
                  (uint64_t)nir_intrinsic_io_semantics(ld).location +
                  nir_src_as_uint(ld->src[1]);
               unsigned tcs_comp = nir_intrinsic_component(ld) + s.comp;
               if (tcs_loc < VARYING_SLOT_MAX && tcs_comp < 4 &&
                   !tcs_pv_bad[tcs_loc][tcs_comp] &&
                   tcs_pv_kind[tcs_loc][tcs_comp] != VBO_NORM_NONE) {
                  origins[var][c] = (struct tes_comp_origin){
                     .valid = true,
                     .is_patch = false,
                     .kind = tcs_pv_kind[tcs_loc][tcs_comp],
                     .tcs_loc = (unsigned)tcs_loc,
                     .tcs_comp = tcs_comp,
                     .cp_idx = nir_src_as_uint(ld->src[0]),
                  };
               } else {
                  tes_comp_bad[var] |= BITFIELD_BIT(c);
               }
            } else if (ld->intrinsic == nir_intrinsic_load_input &&
                       ld->def.bit_size == 32 &&
                       nir_intrinsic_dest_type(ld) == nir_type_float32 &&
                       nir_src_is_const(ld->src[0])) {
               uint64_t tcs_loc =
                  (uint64_t)nir_intrinsic_io_semantics(ld).location +
                  nir_src_as_uint(ld->src[0]);
               unsigned tcs_comp = nir_intrinsic_component(ld) + s.comp;
               if (tcs_loc < VARYING_SLOT_MAX && tcs_comp < 4 &&
                   !tcs_patch_bad[tcs_loc][tcs_comp] &&
                   tcs_patch_kind[tcs_loc][tcs_comp] != VBO_NORM_NONE) {
                  origins[var][c] = (struct tes_comp_origin){
                     .valid = true,
                     .is_patch = true,
                     .kind = tcs_patch_kind[tcs_loc][tcs_comp],
                     .tcs_loc = (unsigned)tcs_loc,
                     .tcs_comp = tcs_comp,
                  };
               } else {
                  tes_comp_bad[var] |= BITFIELD_BIT(c);
               }
            } else {
               tes_comp_bad[var] |= BITFIELD_BIT(c);
            }
         }
      }
   }

   if (!tes_anchor_store)
      return false;

   /* 5. Group all qualifying scalar components by normalized kind (handling
    *    both whole-slot varyings and cross-slot components created by
    *    nir_opt_varyings_bulk compaction) into packed flat u32 buckets of size
    *    2..4 (8-bit) or 2 (16-bit). Single leftover scalars stay unpacked. */
   struct vbo_pack_bucket buckets[64];
   unsigned num_buckets = 0;
   struct comp_pack_ref comp_ref[32][4] = {{{0}}};
   uint8_t packed_mask[32] = {0};

   const enum vbo_norm_kind kinds[4] = {
      VBO_NORM_UNORM8, VBO_NORM_SNORM8, VBO_NORM_UNORM16, VBO_NORM_SNORM16,
   };
   for (unsigned ki = 0; ki < ARRAY_SIZE(kinds); ++ki) {
      enum vbo_norm_kind kind = kinds[ki];
      unsigned max_per =
         (kind == VBO_NORM_UNORM8 || kind == VBO_NORM_SNORM8) ? 4u : 2u;
      unsigned cand_var[128], cand_comp[128], ncand = 0;
      for (unsigned var = 0; var < 32; ++var) {
         if (fs_slot_bad[var] || tes_slot_readback[var])
            continue;
         uint8_t mask = fs_read_mask[var];
         u_foreach_bit(c, mask) {
            if (!(tes_comp_bad[var] & BITFIELD_BIT(c)) &&
                origins[var][c].valid && origins[var][c].kind == kind) {
               cand_var[ncand] = var;
               cand_comp[ncand] = c;
               ncand++;
            }
         }
      }
      unsigned idx = 0;
      while (idx + 2 <= ncand && num_buckets < ARRAY_SIZE(buckets)) {
         unsigned take = MIN2(max_per, ncand - idx);
         /* Avoid leaving a single 1-scalar remainder when splitting 5 -> 3+2. */
         if (max_per == 4 && ncand - idx == 5)
            take = 3;
         struct vbo_pack_bucket *bk = &buckets[num_buckets];
         memset(bk, 0, sizeof(*bk));
         bk->kind = kind;
         bk->count = take;
         for (unsigned j = 0; j < take; ++j) {
            unsigned v = cand_var[idx + j];
            unsigned c = cand_comp[idx + j];
            bk->src_var[j] = v;
            bk->src_comp[j] = c;
            bk->origin[j] = origins[v][c];
            comp_ref[v][c] = (struct comp_pack_ref){
               .packed = true,
               .bucket_idx = num_buckets,
               .lane = j,
            };
            packed_mask[v] |= BITFIELD_BIT(c);
         }
         num_buckets++;
         idx += take;
      }
   }

   if (!num_buckets)
      return false;

   /* Allocate dedicated flat VAR slot(s) for the packed u32 words: any VAR slot
    * that has 0 remaining unpacked live components (either newly vacated or
    * previously unused) is available as a pure flat u32 container (up to 4
    * packed buckets per VAR slot). */
   uint8_t dst_used[32] = {0};
   for (unsigned b = 0; b < num_buckets; ++b) {
      bool placed = false;
      for (unsigned var = 0; var < 32; ++var) {
         if (fs_slot_bad[var] || tes_slot_readback[var])
            continue;
         if ((fs_read_mask[var] & ~packed_mask[var]) != 0)
            continue;
         if (dst_used[var] == 0xf)
            continue;
         unsigned c = ffs(~dst_used[var]) - 1;
         dst_used[var] |= BITFIELD_BIT(c);
         buckets[b].dst_var = var;
         buckets[b].dst_comp = c;
         placed = true;
         break;
      }
      if (!placed)
         return false;
   }

   /* 6. Rewrite TES: strip packed components from original stores (and any
    *    dead components in fully vacated slots), then emit the packed flat
    *    u32 stores at tes_anchor_store. */
   nir_builder tb = nir_builder_at(nir_before_instr(&tes_anchor_store->instr));
   for (unsigned b = 0; b < num_buckets; ++b) {
      struct vbo_pack_bucket *bk = &buckets[b];
      nir_def *vals[4] = {NULL};
      for (unsigned j = 0; j < bk->count; ++j) {
         const struct tes_comp_origin *o = &bk->origin[j];
         if (o->is_patch) {
            vals[j] = nir_load_input(
               &tb, 1, 32, nir_imm_int(&tb, 0), .component = o->tcs_comp,
               .dest_type = nir_type_float32,
               .io_semantics = {.location = o->tcs_loc, .num_slots = 1});
         } else {
            vals[j] = nir_load_per_vertex_input(
               &tb, 1, 32, nir_imm_int(&tb, o->cp_idx), nir_imm_int(&tb, 0),
               .component = o->tcs_comp, .dest_type = nir_type_float32,
               .io_semantics = {.location = o->tcs_loc, .num_slots = 1});
         }
      }
      nir_def *packed = emit_vbo_norm_pack_n(&tb, bk->kind, bk->count, vals);
      bk->pack_store = nir_store_output(
         &tb, packed, nir_imm_int(&tb, 0), .component = bk->dst_comp,
         .write_mask = 1, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + bk->dst_var,
                          .num_slots = 1});
   }

   nir_foreach_block(block, nir_shader_get_entrypoint(tes)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         bool is_new = false;
         for (unsigned b = 0; b < num_buckets; ++b) {
            if (intr == buckets[b].pack_store) {
               is_new = true;
               break;
            }
         }
         if (is_new)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc < VARYING_SLOT_VAR0 || loc >= VARYING_SLOT_VAR0 + 32)
            continue;
         unsigned var = loc - VARYING_SLOT_VAR0;
         if (!packed_mask[var] && !dst_used[var])
            continue;
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = nir_intrinsic_write_mask(intr);
         unsigned keep_mask = 0;
         u_foreach_bit(i, wmask) {
            unsigned c = comp + i;
            if (c < 4 && (fs_read_mask[var] & BITFIELD_BIT(c)) &&
                !(packed_mask[var] & BITFIELD_BIT(c)))
               keep_mask |= BITFIELD_BIT(i);
         }
         if (!keep_mask)
            nir_instr_remove(instr);
         else if (keep_mask != wmask)
            nir_intrinsic_set_write_mask(intr, keep_mask);
      }
   }

   /* 7. Rewrite FS: splice bit-exact hardware unpack channels into every load
    *    touching packed components, preserving any remaining unpacked smooth
    *    components in partially packed slots. */
   nir_builder fb = nir_builder_create(nir_shader_get_entrypoint(fs));
   nir_foreach_block(block, nir_shader_get_entrypoint(fs)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_input &&
             intr->intrinsic != nir_intrinsic_load_interpolated_input)
            continue;
         if (nir_intrinsic_dest_type(intr) != nir_type_float32)
            continue;
         nir_src *off = nir_get_io_offset_src(intr);
         if (!off || !nir_src_is_const(*off))
            continue;
         unsigned loc =
            nir_intrinsic_io_semantics(intr).location + nir_src_as_uint(*off);
         if (loc < VARYING_SLOT_VAR0 || loc >= VARYING_SLOT_VAR0 + 32)
            continue;
         unsigned var = loc - VARYING_SLOT_VAR0;
         if (!packed_mask[var])
            continue;

         unsigned comp = nir_intrinsic_component(intr);
         bool touches_packed = false;
         bool keeps_orig = false;
         nir_component_mask_t read_bits = nir_def_components_read(&intr->def);
         for (unsigned i = 0; i < intr->def.num_components; ++i) {
            unsigned orig_c = comp + i;
            if (!(read_bits & BITFIELD_BIT(i)))
               continue;
            if (orig_c < 4 && comp_ref[var][orig_c].packed)
               touches_packed = true;
            else
               keeps_orig = true;
         }
         if (!touches_packed)
            continue;

         fb.cursor = keeps_orig ? nir_after_instr(instr) : nir_before_instr(instr);
         nir_def *unpacked_cache[64] = {NULL};
         nir_def *chs[4];
         for (unsigned i = 0; i < intr->def.num_components; ++i) {
            unsigned orig_c = comp + i;
            if (!(read_bits & BITFIELD_BIT(i))) {
               chs[i] = nir_undef(&fb, 1, 32);
               continue;
            }
            if (orig_c < 4 && comp_ref[var][orig_c].packed) {
               unsigned bi = comp_ref[var][orig_c].bucket_idx;
               unsigned lane = comp_ref[var][orig_c].lane;
               if (!unpacked_cache[bi]) {
                  const struct vbo_pack_bucket *bk = &buckets[bi];
                  nir_def *packed = nir_load_input(
                     &fb, 1, 32, nir_imm_int(&fb, 0), .component = bk->dst_comp,
                     .dest_type = nir_type_uint32,
                     .io_semantics = {.location = VARYING_SLOT_VAR0 + bk->dst_var,
                                      .num_slots = 1});
                  switch (bk->kind) {
                  case VBO_NORM_UNORM8:
                     unpacked_cache[bi] = nir_unpack_unorm_4x8(&fb, packed);
                     break;
                  case VBO_NORM_SNORM8:
                     unpacked_cache[bi] = nir_unpack_snorm_4x8(&fb, packed);
                     break;
                  case VBO_NORM_UNORM16:
                     unpacked_cache[bi] = nir_unpack_unorm_2x16(&fb, packed);
                     break;
                  case VBO_NORM_SNORM16:
                     unpacked_cache[bi] = nir_unpack_snorm_2x16(&fb, packed);
                     break;
                  default:
                     UNREACHABLE("invalid vbo norm kind");
                  }
               }
               chs[i] = nir_channel(&fb, unpacked_cache[bi], lane);
            } else {
               chs[i] = nir_channel(&fb, &intr->def, i);
            }
         }
         nir_def *repl = nir_vec(&fb, chs, intr->def.num_components);
         if (keeps_orig) {
            nir_def_rewrite_uses_after(&intr->def, repl);
         } else {
            nir_def_rewrite_uses(&intr->def, repl);
            nir_instr_remove(instr);
         }
      }
   }

   NIR_PASS(_, tes, nir_opt_dce);
   NIR_PASS(_, fs, nir_opt_dce);

   /* 8. Bonus TCS->TES task payload compaction: when all components of a
    *    packed bucket come from one exclusive per-vertex TCS VAR slot, hoist
    *    pack_* into TCS so the 0xe0/0xe1 task payload carries 1 u32 word
    *    instead of 2..4 float words and runs pack_* once per patch. */
   unsigned tes_pv_loads[VARYING_SLOT_MAX] = {0};
   bool tes_indirect_pv_load = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(tes)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_per_vertex_input)
            continue;
         if (!nir_src_is_const(intr->src[1])) {
            tes_indirect_pv_load = true;
            continue;
         }
         unsigned loc = nir_intrinsic_io_semantics(intr).location +
                        nir_src_as_uint(intr->src[1]);
         if (loc < VARYING_SLOT_MAX)
            tes_pv_loads[loc]++;
      }
   }

   bool tcs_changed = false;
   if (!tes_indirect_pv_load) {
      for (unsigned b = 0; b < num_buckets; ++b) {
         struct vbo_pack_bucket *bk = &buckets[b];
         unsigned tcs_loc = VARYING_SLOT_MAX;
         unsigned cp_idx = 0;
         bool single_tcs_slot = true;
         for (unsigned j = 0; j < bk->count; ++j) {
            const struct tes_comp_origin *o = &bk->origin[j];
            if (o->is_patch || o->tcs_loc < VARYING_SLOT_VAR0 ||
                o->tcs_loc >= VARYING_SLOT_VAR0 + 32) {
               single_tcs_slot = false;
               break;
            }
            if (tcs_loc == VARYING_SLOT_MAX) {
               tcs_loc = o->tcs_loc;
               cp_idx = o->cp_idx;
            } else if (o->tcs_loc != tcs_loc || o->cp_idx != cp_idx) {
               single_tcs_slot = false;
               break;
            }
         }
         if (!single_tcs_slot || tcs_reads_pv_out[tcs_loc] ||
             tes_pv_loads[tcs_loc] != bk->count)
            continue;

         nir_block *st_block = NULL;
         nir_def *st_vtx = NULL;
         nir_intrinsic_instr *last_tcs_st = NULL;
         nir_scalar tcs_comp_scalar[4] = {{0}};
         bool tcs_comp_seen[4] = {false};
         bool tcs_slot_ok = true;
         nir_foreach_block(block, nir_shader_get_entrypoint(tcs)) {
            bool in_loop = false;
            for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
               if (n->type == nir_cf_node_loop)
                  in_loop = true;
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
                  continue;
               unsigned loc = nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[2]);
               if (loc != tcs_loc)
                  continue;
               if (in_loop) {
                  tcs_slot_ok = false;
                  break;
               }
               if (!st_block) {
                  st_block = block;
                  st_vtx = intr->src[1].ssa;
               } else if (st_block != block ||
                          !is_same_tcs_vertex(st_vtx, intr->src[1].ssa)) {
                  tcs_slot_ok = false;
                  break;
               }
               last_tcs_st = intr;
               unsigned comp = nir_intrinsic_component(intr);
               unsigned wmask = nir_intrinsic_write_mask(intr);
               u_foreach_bit(i, wmask) {
                  unsigned c = comp + i;
                  if (c >= 4)
                     continue;
                  if (tcs_comp_seen[c]) {
                     tcs_slot_ok = false;
                     break;
                  }
                  tcs_comp_scalar[c] = nir_scalar_resolved(intr->src[0].ssa, i);
                  tcs_comp_seen[c] = true;
               }
            }
            if (!tcs_slot_ok)
               break;
         }
         if (!tcs_slot_ok || !last_tcs_st)
            continue;
         for (unsigned j = 0; j < bk->count; ++j) {
            if (!tcs_comp_seen[bk->origin[j].tcs_comp]) {
               tcs_slot_ok = false;
               break;
            }
         }
         if (!tcs_slot_ok)
            continue;

         nir_builder cb = nir_builder_at(nir_before_instr(&last_tcs_st->instr));
         nir_def *pack_vals[4] = {NULL};
         for (unsigned j = 0; j < bk->count; ++j) {
            nir_scalar s = tcs_comp_scalar[bk->origin[j].tcs_comp];
            pack_vals[j] =
               s.def->num_components == 1 ? s.def : nir_channel(&cb, s.def, s.comp);
         }
         nir_def *tcs_packed =
            emit_vbo_norm_pack_n(&cb, bk->kind, bk->count, pack_vals);
         nir_intrinsic_instr *new_tcs_st = nir_store_per_vertex_output(
            &cb, tcs_packed, st_vtx, nir_imm_int(&cb, 0), .component = 0,
            .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = tcs_loc, .num_slots = 1});
         nir_foreach_instr_safe(instr, st_block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_per_vertex_output ||
                intr == new_tcs_st)
               continue;
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[2]);
            if (loc == tcs_loc)
               nir_instr_remove(instr);
         }

         tb.cursor = nir_before_instr(&bk->pack_store->instr);
         nir_def *tes_packed = nir_load_per_vertex_input(
            &tb, 1, 32, nir_imm_int(&tb, cp_idx), nir_imm_int(&tb, 0),
            .component = 0, .dest_type = nir_type_uint32,
            .io_semantics = {.location = tcs_loc, .num_slots = 1});
         nir_src_rewrite(&bk->pack_store->src[0], tes_packed);
         tes_pv_loads[tcs_loc] = 0;
         tcs_changed = true;
      }
   }

   if (tcs_changed) {
      NIR_PASS(_, tcs, nir_opt_dce);
      NIR_PASS(_, tes, nir_opt_dce);
      gather_info_preserving_clip_cull(tcs);
   }
   gather_info_preserving_clip_cull(tes);
   gather_info_preserving_clip_cull(fs);
   return true;
}
