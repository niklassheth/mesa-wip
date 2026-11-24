/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile_apple9_priv.h"
#include "gravity/debug.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
apple9_dag_cost_less(const struct apple9_dag_cost *a,
                     const struct apple9_dag_cost *b)
{
   return a->weighted_spills < b->weighted_spills ||
          (a->weighted_spills == b->weighted_spills &&
           (a->spills < b->spills ||
            (a->spills == b->spills && a->instructions < b->instructions)));
}


struct apple9_preamble_eligibility {
   /* 0 unknown, 1 rejected/visiting, 2 accepted. Memoize shared DAGs. */
   uint8_t *state;
   /* Maps hoisted instructions in entry back to their pre-hoist block so
    * apple9_main_span measures their real live range rather than a synthetic
    * span from shader entry. */
   const struct hash_table *original_blocks;
};

static bool
apple9_preamble_source(nir_src *src, void *data);

static bool
apple9_preamble_instruction(const nir_instr *instr, const void *data)
{
   const struct apple9_preamble_eligibility *eligibility = data;
   nir_def *def = nir_instr_def((nir_instr *)instr);
   if (!def || (def->bit_size != 32 && def->bit_size != 16 && def->bit_size != 1))
      return false;
   /* The setup program is straight-line code, but nir_opt_preamble keeps
    * the control flow around work it hoists. Hoisting under control flow,
    * such as a counter address behind a uniform query test, would therefore
    * leave the whole shader without a preamble. */
   if (instr->block->cf_node.parent->type != nir_cf_node_function)
      return false;
   uint8_t *state = &eligibility->state[def->index];
   if (*state)
      return *state == 2;
   *state = 1;
   bool accepted = false;
   switch (instr->type) {
   case nir_instr_type_load_const:
      accepted = true;
      break;
   case nir_instr_type_alu:
      accepted = apple9_instruction_is_in_subset((nir_instr *)instr, false) &&
                 nir_foreach_src((nir_instr *)instr, apple9_preamble_source, (void *)data);
      break;
   case nir_instr_type_intrinsic: {
      /* Global reads of memory the shader cannot write, such as Honeykrisp
       * descriptors, keep buffer addresses in uniforms like Metal's arguments. */
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic((nir_instr *)instr);
      accepted = (intr->intrinsic == nir_intrinsic_load_ubo ||
                  (intr->intrinsic == nir_intrinsic_load_global_2x32 &&
                   (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))) &&
                 nir_foreach_src((nir_instr *)instr, apple9_preamble_source, (void *)data);
      break;
   }
   default:
      break;
   }
   *state = accepted ? 2 : 1;
   return accepted;
}

static bool
apple9_preamble_source(nir_src *src, void *data)
{
   return apple9_preamble_instruction(nir_def_instr(src->ssa), data);
}

static bool
apple9_preamble_avoid(const nir_instr *instr, const void *data)
{
   nir_def *def = nir_instr_def((nir_instr *)instr);
   /* Boolean and half intermediates can move with an expression, but only
    * 32-bit results have a validated argument-window transfer. */
   return !def || def->bit_size != 32 || !apple9_preamble_instruction(instr, data);
}

static void
apple9_preamble_def_size(nir_def *def, unsigned *size, unsigned *align,
                        nir_preamble_class *class_)
{
   /* Even-aligned tuples let a stored pointer serve as a memory instruction's
    * uniform base pair. */
   *size = def->num_components;
   *align = def->num_components > 1 ? 2 : 1;
   *class_ = nir_preamble_class_general;
}

/* Constants an ALU instruction can read from a uniform word. The backend
 * folds such preamble reads into uniform operands; a constant left in the
 * main program costs a mov_imm32 and a register, and above r63 a copy too.
 */
static bool
apple9_uniform_constant_use(nir_alu_instr *alu, unsigned s, uint32_t *value)
{
   bool is_float, comparison = false;
   switch (alu->op) {
   case nir_op_flt: case nir_op_fge: case nir_op_feq: case nir_op_fneu:
   case nir_op_ilt: case nir_op_ige: case nir_op_ieq: case nir_op_ine:
   case nir_op_ult: case nir_op_uge:
      comparison = true;
      /* Select/predicate sources have no folded float-immediate form yet. */
      is_float = false;
      break;
   case nir_op_ishl: case nir_op_ishr: case nir_op_ushr:
      /* Constant shift amounts already have an immediate encoding. */
      if (s != 0)
         return false;
      is_float = false;
      break;
   case nir_op_bcsel:
      /* The condition is a boolean; only the selected 32-bit values use
       * these validated uniform operands. */
      if (s == 0)
         return false;
      is_float = false;
      break;
   case nir_op_fadd: case nir_op_fmul: case nir_op_ffma:
   case nir_op_fmin: case nir_op_fmax:
      is_float = true;
      break;
   case nir_op_ffloor: case nir_op_fceil: case nir_op_ftrunc:
   case nir_op_fround_even: case nir_op_frsq: case nir_op_fexp2: case nir_op_flog2:
   case nir_op_bit_count: case nir_op_bitfield_reverse: case nir_op_ufind_msb:
   case nir_op_u2f32: case nir_op_i2f32: case nir_op_f2u32: case nir_op_f2i32:
   case nir_op_iadd: case nir_op_iand: case nir_op_ior: case nir_op_ixor:
   case nir_op_imin: case nir_op_imax: case nir_op_umin: case nir_op_umax:
      is_float = false;
      break;
   default:
      return false;
   }
   if (alu->def.num_components != 1 ||
       (alu->def.bit_size != 32 && !comparison) ||
       alu->src[s].src.ssa->bit_size != 32 ||
       !nir_src_is_const(alu->src[s].src))
      return false;
   *value = nir_src_comp_as_uint(alu->src[s].src, alu->src[s].swizzle[0]);
   /* Constant 0 is always cheap via 2B MOV_IMM_COMPACT / 4B ZERO / ALU imm. */
   if (*value == 0)
      return false;
   /* Low-bit masks select as bitfield extracts, and shifted masks on matching
    * shifts select as bfi_imm; both need no operand. */
   if (alu->op == nir_op_iand && *value && *value != UINT32_MAX) {
      if (!(*value & (*value + 1)))
         return false;
      unsigned shift = ffs(*value) - 1;
      uint32_t low = *value >> shift;
      if (shift > 0 && !(low & (low + 1)) && shift + util_bitcount(low) < 32) {
         nir_scalar other = apple9_chase_trivial(
            nir_get_scalar(alu->src[1 - s].src.ssa, alu->src[1 - s].swizzle[0]));
         uint32_t amt;
         if (nir_def_instr_type(other.def) == nir_instr_type_alu &&
             nir_scalar_alu_op(other) == nir_op_ishl &&
             apple9_const_u32(nir_scalar_chase_alu_src(other, 1), &amt) &&
             (amt & 31) == shift)
            return false;
      }
   }
   /* Packed float immediates are already free. */
   uint8_t encoded;
   if (is_float) {
      /* Floating min/max operands, FADD/FMUL operands and the FMA addend
       * also take a fourth mantissa bit (min/max commute to operand B). */
      bool low;
      bool wide = alu->op != nir_op_ffma || s == 2;
      return !(wide ? agx_apple9_encode_float_immediate_wide(*value, &encoded, &low)
                    : agx_apple9_encode_float_immediate(*value, &encoded));
   }
   /* So are 8-bit literals of integer add/subtract, logic and selected
    * values; a small negative addend is subtracted. */
   /* Compare operands take a literal as well (either side, by reversal). */
   if (comparison) {
      bool low;
      bool float_compare = alu->op == nir_op_flt || alu->op == nir_op_fge ||
                           alu->op == nir_op_feq || alu->op == nir_op_fneu;
      return float_compare
                ? !agx_apple9_encode_float_immediate_wide(*value, &encoded, &low)
                : *value > UINT8_MAX;
   }
   switch (alu->op) {
   case nir_op_iadd:
      return *value > UINT8_MAX && -*value > UINT8_MAX;
   case nir_op_iand: case nir_op_ior: case nir_op_ixor:
   case nir_op_imin: case nir_op_imax: case nir_op_umin: case nir_op_umax:
      return *value > UINT8_MAX;
   case nir_op_bcsel: {
      bool low;
      return *value > UINT8_MAX &&
             !agx_apple9_encode_float_immediate_wide(*value, &encoded, &low);
   }
   default:
      return true;
   }
}

struct apple9_constant_count {
   uint32_t value;
   unsigned uses;
};

static int
apple9_constant_count_cmp(const void *a, const void *b)
{
   const struct apple9_constant_count *x = a, *y = b;
   if (x->uses != y->uses)
      return x->uses < y->uses ? 1 : -1;
   return x->value < y->value ? -1 : x->value > y->value;
}

/* Store the most-used constants in free uniform words. Keep their literal
 * NIR values through instruction selection: opaque preamble loads would hide
 * address offsets, integer folds and source modifiers. Native optimization
 * uses this table to select uniforms for the remaining constant operands. */
static void
apple9_promote_constants(nir_shader *shader, nir_function_impl *preamble,
                         unsigned *used, unsigned budget,
                         struct agx_apple9_uniform_constants *constants)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct hash_table_u64 *counts = _mesa_hash_table_u64_create(NULL);
   struct util_dynarray list = UTIL_DYNARRAY_INIT;

   nir_foreach_block(block, impl) {
      unsigned depth = 0;
      for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent)
         depth += node->type == nir_cf_node_loop;
      /* Approximate repeated execution without assuming a known trip count.
       * Cap nesting and saturate the sum so large shaders cannot wrap it. */
      unsigned weight = 1u << (3 * MIN2(depth, 6));
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; ++s) {
            uint32_t value;
            if (!apple9_uniform_constant_use(alu, s, &value))
               continue;
            uintptr_t index = (uintptr_t)_mesa_hash_table_u64_search(counts, value);
            if (!index) {
               util_dynarray_append(&list, ((struct apple9_constant_count){value, 0}));
               index = util_dynarray_num_elements(&list, struct apple9_constant_count);
               _mesa_hash_table_u64_insert(counts, value, (void *)index);
            }
            struct apple9_constant_count *entry =
               util_dynarray_element(&list, struct apple9_constant_count, index - 1);
            entry->uses += MIN2(weight, UINT_MAX - entry->uses);
         }
      }
   }

   unsigned count = util_dynarray_num_elements(&list, struct apple9_constant_count);
   struct apple9_constant_count *sorted = list.data;
   if (count)
      qsort(sorted, count, sizeof(*sorted), apple9_constant_count_cmp);

   /* A single use outside loops gains nothing: it trades its mov_imm32 for
    * a preamble store and spends a uniform word. Loop uses carry their
    * execution weight, allowing a lone static use to be promoted. */
   constants->base = *used;
   constants->count = 0;
   nir_builder pb = nir_builder_at(nir_after_impl(preamble));
   /* nir_opt_preamble has already reserved its result words. Spend only the
    * remaining words; the completed setup is checked against the compiler's
    * preamble code-size bound before it is attached to the main program. */
   for (unsigned i = 0; i < count && *used < budget &&
                        sorted[i].uses >= 2; ++i) {
      nir_store_preamble(&pb, nir_imm_int(&pb, sorted[i].value), .base = *used,
                         .preamble_class = nir_preamble_class_general);
      constants->values[constants->count++] = sorted[i].value;
      ++*used;
   }

   _mesa_hash_table_u64_destroy(counts);
   util_dynarray_fini(&list);
}

/* Index of the last instruction of a loop (instructions are indexed in
 * program order, so it is the largest in the loop). */
static unsigned
apple9_loop_end_index(nir_loop *loop)
{
   unsigned last = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_instr *instr = nir_block_last_instr(block);
      if (instr)
         last = MAX2(last, instr->index);
   }
   return last;
}

/* Whether a CF node lies inside the loop. */
static bool
apple9_cf_inside(nir_cf_node *node, nir_loop *loop)
{
   for (; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

/* A value left in the main program holds a register from its definition to
 * its last use. When the uniform words run out, that span, not only the
 * instructions it takes to compute, is what a uniform word saves: long-lived
 * buffer bases and invariant values otherwise occupy GPRs across the whole
 * program and force spills.
 *
 * Instructions temporarily hoisted into the entry block for preamble
 * consideration will be restored to their original block (or dominating LCA)
 * if not selected, so measure their span from their actual uses and original
 * block rather than from entry; otherwise short-lived guarded addresses deep
 * in the shader appear to span the entire program and evict genuinely
 * long-lived uniforms from the preamble budget. */
static unsigned
apple9_main_span(nir_instr *instr, const struct hash_table *original_blocks)
{
   nir_def *def = nir_instr_def(instr);
   if (!def)
      return 0;
   struct hash_entry *hoisted =
      original_blocks ? _mesa_hash_table_search(original_blocks, instr) : NULL;
   nir_block *def_block = hoisted ? hoisted->data : instr->block;
   unsigned first = hoisted ? UINT_MAX : instr->index;
   unsigned last = hoisted ? 0 : instr->index;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         continue;
      nir_instr *use = nir_src_use_instr(src);
      struct hash_entry *use_hoisted =
         original_blocks ? _mesa_hash_table_search(original_blocks, use) : NULL;
      nir_block *use_block = use_hoisted ? use_hoisted->data : use->block;
      unsigned use_index = use->index;
      if (use_hoisted) {
         nir_instr *anchor = nir_block_first_instr(use_block);
         if (anchor)
            use_index = anchor->index;
      }
      first = MIN2(first, use_index);
      last = MAX2(last, use_index);
      /* A use inside a loop that does not contain the definition keeps the
       * value live to the end of that (outermost such) loop, except a phi in
       * the loop header, whose own value takes over at loop entry. */
      for (nir_cf_node *node = use_block->cf_node.parent; node;
           node = node->parent) {
         if (node->type != nir_cf_node_loop)
            continue;
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (use->type == nir_instr_type_phi &&
             use_block == nir_loop_first_block(loop))
            continue;
         if (!apple9_cf_inside(&def_block->cf_node, loop))
            last = MAX2(last, apple9_loop_end_index(loop));
      }
   }
   return last > first ? last - first : 0;
}

static float
apple9_preamble_cost(nir_instr *instr, const void *data)
{
   const struct apple9_preamble_eligibility *eligibility = data;
   if (!apple9_preamble_instruction(instr, data))
      return 0;
   float cost = instr->type == nir_instr_type_intrinsic ? 8 : 1;
   /* One cost unit per 25 main-program instructions of span; measured on
    * ER's draw-time fragments (spills 4,349 -> 1,396). */
   cost += (float)apple9_main_span(instr, eligibility->original_blocks) / 25;
   return cost;
}

static float
apple9_preamble_rewrite_cost(nir_def *def, const void *unused)
{
   return def->num_components;
}

/* Texture and sampler handles are descriptor offsets from the USC aperture
 * base, which the stage's texture and sampler table pointers name when
 * handles are used. Convert them to the native table indices read by the
 * texture unit, once all descriptor metadata has been loaded through them.
 * A stage with register-indexed samplers reaches the private texel-fetch
 * sampler the same way.
 */
/* Split an iadd tree into its non-constant terms (NULL when none) and the sum
 * of its constant terms. */
static nir_def *
apple9_split_addend(nir_builder *b, nir_scalar s, uint32_t *constant,
                    unsigned depth)
{
   if (nir_scalar_is_const(s)) {
      *constant += nir_scalar_as_uint(s);
      return NULL;
   }
   if (depth && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      nir_scalar x = nir_scalar_chase_alu_src(s, 0);
      nir_scalar y = nir_scalar_chase_alu_src(s, 1);
      uint32_t before = *constant;
      nir_def *xs = apple9_split_addend(b, x, constant, depth - 1);
      nir_def *ys = apple9_split_addend(b, y, constant, depth - 1);
      if (*constant == before && xs && ys)
         return nir_channel(b, s.def, s.comp);
      if (!xs || !ys)
         return xs ? xs : ys;
      return nir_iadd(b, xs, ys);
   }
   return nir_channel(b, s.def, s.comp);
}

/* Handles are byte offsets into the descriptor aperture, which never wraps,
 * so the multiple-of-32 part of a handle's constant terms moves only the
 * entry index. Fold it into the entry immediate: handles that differ by a
 * record-relative offset (the image and sampler parts of one combined
 * descriptor, planes, array elements at constant indices) then share one
 * shift. */
static nir_def *
apple9_descriptor_entry(nir_builder *b, nir_def *handle, unsigned entry)
{
   uint32_t constant = 0;
   nir_def *x = apple9_split_addend(b, nir_get_scalar(handle, 0), &constant, 4);
   int32_t entries = (int32_t)constant >> 5;
   uint32_t rest = constant & 31;
   if (!x)
      return nir_imm_int(b, entries + entry);
   if (rest)
      x = nir_iadd_imm(b, x, rest);
   return nir_iadd_imm(b, nir_ushr_imm(b, x, 5), entries + entry);
}

static bool
apple9_lower_texture_handles(nir_shader *nir)
{
   bool samplers = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex)
            samplers |= nir_tex_instr_src_index(nir_instr_as_tex(instr),
                                                nir_tex_src_sampler_handle) >= 0;
      }
   }

   bool progress = false;
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (samplers &&
                intr->intrinsic == nir_intrinsic_image_load_apple9_agx &&
                nir_intrinsic_base(intr) == 0) {
               b.cursor = nir_before_instr(instr);
               nir_src_rewrite(&intr->src[3],
                  nir_load_ubo(&b, 1, 32,
                     nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                     nir_imm_int(&b, AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET),
                     .align_mul = 4, .range = 4));
               nir_intrinsic_set_base(intr, 1);
               progress = true;
            }
            continue;
         }
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
         if (texture < 0 || (tex->backend_flags & AGX_APPLE9_TEXTURE_HANDLE_INDEX))
            continue;
         b.cursor = nir_before_instr(instr);
         nir_src_rewrite(&tex->src[texture].src,
            apple9_descriptor_entry(&b, tex->src[texture].src.ssa,
                                    AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY));
         int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
         if (sampler >= 0) {
            nir_src_rewrite(&tex->src[sampler].src,
               apple9_descriptor_entry(&b, tex->src[sampler].src.ssa,
                                       AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY));
         } else if (samplers && apple9_texture_uses_fetch_sampler(tex)) {
            nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
               nir_load_ubo(&b, 1, 32,
                  nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                  nir_imm_int(&b, AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET),
                  .align_mul = 4, .range = 4));
         }
         tex->backend_flags |= AGX_APPLE9_TEXTURE_HANDLE_INDEX;
         progress = true;
      }
   }
   return nir_progress(progress, nir_shader_get_entrypoint(nir),
                       nir_metadata_control_flow);
}

/* Read-only descriptor loads and uniform buffer-size limits can execute
 * before their original branch. Move only speculatable uniform dependency
 * chains, then let preamble
 * extraction choose which values are worth storing. Ordinary guarded buffer
 * reads lack CAN_SPECULATE and must remain guarded.
 */
struct apple9_uniform_hoist {
   nir_block *entry;
   uint8_t *state;
   nir_cursor cursor;
   struct hash_table *original_blocks;
   /* Whether apple9_preamble_safe_read reads may leave their branches. */
   bool safe_reads;
   /* Whether descriptor memory reads alone may (stages without safe_reads),
    * with the constant driver-UBO reads that form their handles. */
   bool descriptor_reads;
   bool descriptor_chain;
};

static bool
apple9_is_bounds_limit(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_iadd || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;

   nir_scalar root = nir_get_scalar(&alu->def, 0);
   for (unsigned s = 0; s < 2; ++s) {
      uint32_t addend, minimum;
      nir_scalar max = nir_scalar_chase_alu_src(root, 1 - s);
      if (!apple9_const_u32(nir_scalar_chase_alu_src(root, s), &addend) ||
          !addend || !nir_scalar_is_alu(max) ||
          nir_scalar_alu_op(max) != nir_op_umax)
         continue;
      for (unsigned m = 0; m < 2; ++m) {
         if (apple9_const_u32(nir_scalar_chase_alu_src(max, m), &minimum) &&
             minimum == -addend)
            return true;
      }
   }
   return false;
}

/* A spilling preamble setup retries without the preamble-only reads
 * (apple9_preamble_trial) by setting 0. */
static _Thread_local int apple9_safe_reads_override = -1;

/* Reads that are always safe once per dispatch at a uniform address, but
 * that must not be generally speculatable: if-conversion or code motion
 * would then run them for every thread on paths that skipped them.
 * Driver UBOs (app UBOs are global by now), descriptor memory in the USC
 * aperture and reads the driver marks ACCESS_UNIFORM_SAFE_AGX qualify. */
/* Reads of descriptor memory: constant, read-only records in the USC
 * aperture, at a uniform address. */
static bool
apple9_descriptor_read(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_global_2x32 ||
       !(nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))
      return false;
   nir_scalar hi = nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 1));
   return nir_scalar_is_const(hi) &&
          nir_scalar_as_uint(hi) == (AGX_APPLE9_USC_BASE >> 32);
}

static bool
apple9_preamble_safe_read(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_ubo)
      return nir_src_is_const(intr->src[0]) && nir_src_is_const(intr->src[1]);
   if (intr->intrinsic != nir_intrinsic_load_global_2x32 ||
       !(nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))
      return false;
   if (nir_intrinsic_access(intr) & ACCESS_UNIFORM_SAFE_AGX)
      return true;
   nir_scalar hi = nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 1));
   return nir_scalar_is_const(hi) &&
          nir_scalar_as_uint(hi) == (AGX_APPLE9_USC_BASE >> 32);
}

/* The first block of a loop at function level runs once whenever the shader
 * reaches the loop: no break or continue can precede it. A uniform value it
 * computes from read-only memory is the same on every iteration, so it can
 * move to the entry like a function-level block. DXIL's single-trip
 * "loop { ...; break; }" early-exit wrapper puts whole shaders there. */
static bool
apple9_block_runs_once_per_loop_entry(nir_block *block)
{
   nir_cf_node *parent = block->cf_node.parent;
   return parent->type == nir_cf_node_loop &&
          parent->parent->type == nir_cf_node_function &&
          block == nir_loop_first_block(nir_cf_node_as_loop(parent));
}

static bool
apple9_can_hoist_uniform(nir_src *src, void *data)
{
   struct apple9_uniform_hoist *s = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   uint8_t *state = &s->state[src->ssa->index];
   if (*state)
      return *state == 2;
   *state = 1;
   if (instr->block != s->entry && !nir_instr_can_speculate(instr) &&
       !apple9_block_runs_once_per_loop_entry(instr->block) &&
       !(s->safe_reads && apple9_preamble_safe_read(instr)) &&
       !(s->descriptor_reads && apple9_descriptor_read(instr)) &&
       !(s->descriptor_chain && apple9_preamble_safe_read(instr) &&
         nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo))
      return false;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_ubo &&
          !(intr->intrinsic == nir_intrinsic_load_global_2x32 &&
            (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER)))
         return false;
   } else if (instr->type != nir_instr_type_alu &&
              instr->type != nir_instr_type_load_const) {
      return false;
   }
   if (!nir_foreach_src(instr, apple9_can_hoist_uniform, s))
      return false;
   *state = 2;
   return true;
}

static bool
apple9_hoist_uniform_source(nir_src *src, void *data)
{
   struct apple9_uniform_hoist *s = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   if (instr->block == s->entry)
      return true;
   nir_foreach_src(instr, apple9_hoist_uniform_source, s);
   if (s->original_blocks)
      _mesa_hash_table_insert(s->original_blocks, instr, instr->block);
   nir_instr_move(s->cursor, instr);
   s->cursor = nir_after_instr(instr);
   return true;
}

/* Preamble-only reads leave their branches in every stage. A fragment or
 * vertex preamble runs once per draw, so moving a guarded uniform read there
 * costs the draw a few instructions and saves each thread its load, address
 * and the registers holding them. A setup that spills retries without them
 * (apple9_preamble_trial); hk chooses which reads qualify. */
static bool
apple9_safe_reads_enabled(void)
{
   return apple9_safe_reads_override != 0;
}

static bool
apple9_hoist_uniform_reads(nir_shader *nir, struct hash_table *original_blocks,
                            bool addresses, unsigned *address_hoists)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   struct apple9_uniform_hoist s = {
      .entry = nir_start_block(impl),
      .original_blocks = original_blocks,
      .safe_reads = apple9_safe_reads_enabled(),
      /* Descriptor records read through handles (texture info, runtime
       * sampler state) are hoisted in every stage: unlike buffer reads,
       * leaving them in their branches did not save registers. */
      .descriptor_reads = true,
      .state = calloc(MAX2(impl->ssa_alloc, 1), 1),
      .cursor = nir_after_block_before_jump(nir_start_block(impl)),
   };
   if (!s.state)
      return false;
   bool progress = false;
   /* Sampler state of conditionally executed samples (packed LOD/bias words,
    * descriptor handles) is uniform: offer its whole chain, including the
    * conversions and packing, to preamble extraction. Unselected
    * instructions are restored. Like address hoists, this placement is
    * compared against the original one when it spills or its setup does not
    * fit. */
   const bool alu_seeds = addresses && original_blocks;
   if (alu_seeds) {
      nir_foreach_block(block, impl) {
         if (block == s.entry)
            continue;
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            for (unsigned i = 0; i < tex->num_srcs; ++i) {
               switch (tex->src[i].src_type) {
               case nir_tex_src_coord: case nir_tex_src_ddx: case nir_tex_src_ddy:
               case nir_tex_src_comparator: case nir_tex_src_projector:
                  continue;
               default:
                  break;
               }
               nir_instr *source = nir_def_instr(tex->src[i].src.ssa);
               if (source->block == s.entry || source->type != nir_instr_type_alu)
                  continue;
               /* Constant driver-UBO reads (descriptor-set and root
                * pointers) may leave their branches with it. */
               s.descriptor_chain = s.descriptor_reads;
               bool hoist = apple9_can_hoist_uniform(&tex->src[i].src, &s);
               s.descriptor_chain = false;
               if (hoist) {
                  apple9_hoist_uniform_source(&tex->src[i].src, &s);
                  progress = true;
                  if (address_hoists)
                     ++*address_hoists;
               }
            }
         }
      }
   }
   nir_foreach_block(block, impl) {
      if (block == s.entry)
         continue;
      nir_foreach_instr_safe(instr, block) {
         /* In a run-once loop head, offer whole uniform expressions (the
          * bounds selects of the reads), not just the reads. */
         if (!apple9_is_bounds_limit(instr) &&
             !(instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_global_2x32) &&
             !(instr->type == nir_instr_type_alu &&
               apple9_block_runs_once_per_loop_entry(block)))
            continue;
         nir_src src = nir_src_for_ssa(nir_instr_def(instr));
         s.descriptor_chain = s.descriptor_reads && apple9_descriptor_read(instr);
         bool hoist = apple9_can_hoist_uniform(&src, &s);
         s.descriptor_chain = false;
         if (hoist) {
            apple9_hoist_uniform_source(&src, &s);
            progress = true;
         } else if (addresses && instr->type == nir_instr_type_intrinsic &&
                    nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_global_2x32) {
            /* The guarded read itself may not speculate, while its uniform
             * address can. Hoist that pure dependency chain so preamble
             * extraction can select bounds checks and redirected pointers
             * without moving the memory access across its condition. */
            nir_src *address = &nir_instr_as_intrinsic(instr)->src[0];
            if (nir_def_instr(address->ssa)->block != s.entry &&
                apple9_can_hoist_uniform(address, &s)) {
               apple9_hoist_uniform_source(address, &s);
               progress = true;
               if (address_hoists)
                  ++*address_hoists;
            }
         }
      }
   }
   free(s.state);
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
agx_nir_hoist_apple9_uniform_reads(nir_shader *nir)
{
   return apple9_hoist_uniform_reads(nir, NULL, true, NULL);
}

/* Exposing an address to preamble extraction must not leave every unselected
 * expression live from shader entry. Restore surviving hoisted instructions
 * toward their original blocks. CSE may have shared them with other branches,
 * so include all uses when choosing a dominating block. Reverse traversal
 * moves consumers before sources, preserving their order after the phis. */
static void
apple9_restore_unselected_hoists(nir_shader *nir, struct hash_table *original_blocks)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_dominance |
                              nir_metadata_dominance_lca);
   nir_block *entry = nir_start_block(impl);
   bool progress = false;
   nir_foreach_instr_reverse_safe(instr, entry) {
      /* Extraction creates load_preamble intrinsics and frees selected
       * definitions. Do not associate new allocations with old pointer keys. */
      if (instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_preamble)
         continue;
      struct hash_entry *found = _mesa_hash_table_search(original_blocks, instr);
      if (!found)
         continue;
      nir_def *def = nir_instr_def(instr);
      nir_block *target = found->data;
      nir_foreach_use_including_if(use, def)
         target = nir_dominance_lca(target, nir_src_get_block(use));
      if (target != entry) {
         nir_instr_move(nir_after_phis(target), instr);
         progress = true;
      }
   }
   nir_progress(progress, impl, nir_metadata_control_flow);
}

/* The main program every preamble attempt starts from: the attempts of both
 * placements share it, so it is prepared once per compile. */
static _Thread_local nir_shader *apple9_prepared_main;

static nir_shader *
apple9_prepare_preamble_main(nir_shader *nir)
{
   if (apple9_prepared_main)
      return nir_shader_clone(NULL, apple9_prepared_main);
   nir_shader *main = nir_shader_clone(NULL, nir);
   if (!main)
      return NULL;
   /* Avoid spending uniform words on unused components of frontend UBO
    * vectors. Removing leading components can also reduce the alignment:
    * vec4 at byte 16 may become vec3 at byte 20. Re-legalize that access
    * before selecting its memory format or extracting a preamble. */
   nir_opt_shrink_vectors(main, true);
   nir_opt_constant_folding(main);
   apple9_legalize_buffer_accesses(main);
   return main;
}

static bool
apple9_try_preamble(nir_shader *nir, struct agx_shader_part *out,
                   struct agx_apple9_compute_profile *profile,
                   const struct agx_apple9_varying_layout *varyings,
                   const struct apple9_buffer_map *resources,
                   unsigned uniform_budget, bool promote_constants,
                   bool hoist_addresses, struct apple9_dag_cost *cost,
                   unsigned *preamble_bytes, unsigned *preamble_words,
                   bool *setup_spilled, unsigned *address_hoists)
{
   nir_shader *main = apple9_prepare_preamble_main(nir);
   if (!main)
      return false;
   struct hash_table *original_blocks = _mesa_pointer_hash_table_create(NULL);
   if (!original_blocks) {
      ralloc_free(main);
      return false;
   }
   /* Remember where every hoisted read came from, in either trial, so a read
    * the preamble does not take returns to the branch that needed it rather
    * than running unconditionally for every thread. */
   if (apple9_hoist_uniform_reads(main, original_blocks,
                                  hoist_addresses, address_hoists))
      nir_opt_cse(main);
   nir_function_impl *main_impl = nir_shader_get_entrypoint(main);
   nir_index_ssa_defs(main_impl);
   struct apple9_preamble_eligibility eligibility = {
      .state = calloc(MAX2(main_impl->ssa_alloc, 1), sizeof(uint8_t)),
      .original_blocks = original_blocks,
   };
   if (!eligibility.state) {
      _mesa_hash_table_destroy(original_blocks, NULL);
      ralloc_free(main);
      return false;
   }
   const nir_opt_preamble_options options = {
      .cb_data = &eligibility,
      .def_size = apple9_preamble_def_size,
      .preamble_storage_size[nir_preamble_class_general] =
         uniform_budget,
      .instr_cost_cb = apple9_preamble_cost,
      .rewrite_cost_cb = apple9_preamble_rewrite_cost,
      .avoid_instr_cb = apple9_preamble_avoid,
   };
   unsigned sizes[nir_preamble_num_classes] = {0};
   struct agx_shader_part setup = {0}, candidate = {0};
   struct agx_apple9_uniform_constants constants = {0};
   struct agx_apple9_compute_profile candidate_profile = {0};
   const char *candidate_reason = NULL;
   nir_shader *preamble = NULL;
   if (main)
      nir_index_instrs(nir_shader_get_entrypoint(main));
   if (main && nir_opt_preamble(main, &options, sizes)) {
      apple9_restore_unselected_hoists(main, original_blocks);
      nir_function_impl *impl = nir_shader_get_preamble(main);
      /* Setup currently admits ordinary linear ALU/UBO work. Masked setup and
       * scratch have independent execution contracts and are not assumed. */
      if (impl && !apple9_cf_list_has_control_flow(&impl->body)) {
         if (promote_constants)
            apple9_promote_constants(main, impl,
                                     &sizes[nir_preamble_class_general],
                                     options.preamble_storage_size[nir_preamble_class_general],
                                     &constants);
         constants.base +=
            resources->descriptor_tables
               ? MAX2(AGX_APPLE9_GRAPHICS_ROOT_WORDS,
                      resources->native_root_words)
               : agx_apple9_compute_preamble_base(resources->count);
         preamble = nir_shader_create(NULL, nir->info.stage, nir->options);
         preamble->info.stage = nir->info.stage;
         preamble->info.float_controls_execution_mode = nir->info.float_controls_execution_mode;
         memcpy(preamble->info.workgroup_size, nir->info.workgroup_size,
                sizeof(preamble->info.workgroup_size));
         nir_function *fn = nir_function_create(preamble, "main");
         fn->is_entrypoint = true;
         fn->impl = nir_function_impl_clone(preamble, impl);
         fn->impl->function = fn;
         nir_opt_dce(main);
         nir_opt_dce(preamble);
         if (apple9_compile_dag_body(preamble, &setup, NULL, NULL, resources,
                                    NULL, true, NULL, &candidate_reason) &&
             !setup.info.scratch_size &&
             setup.info.main_size > 4 &&
             setup.info.main_size <=
                agx_apple9_max_preamble_bytes(nir->info.stage == MESA_SHADER_COMPUTE) &&
             apple9_compile_dag_body(main, &candidate,
                                    profile ? &candidate_profile : NULL,
                                    varyings, resources, &constants, false, cost, &candidate_reason)) {
            unsigned size = setup.info.main_size;
            uint8_t *binary = realloc(candidate.binary, candidate.info.binary_size + size);
            if (binary) {
               memcpy(binary + candidate.info.binary_size, setup.binary, size);
               candidate.info.apple9_preamble_offset = candidate.info.binary_size;
               candidate.info.apple9_preamble_size = size;
               candidate.info.binary_size += size;
               candidate.binary = binary;
               *out = candidate;
               if (profile) {
                  candidate_profile.preamble_offset = out->info.apple9_preamble_offset;
                  candidate_profile.preamble_size = size;
                  *profile = candidate_profile;
               }
               _mesa_hash_table_destroy(original_blocks, NULL);
               free(eligibility.state);
               free(setup.binary);
               ralloc_free(preamble);
               ralloc_free(main);
               return true;
            }
         }
      }
   }
   if (agx_apple9_trace_enabled() && candidate_reason)
      fprintf(stderr, "APPLE9_PREAMBLE_FALLBACK %s\n", candidate_reason);
   *setup_spilled = setup.info.scratch_size != 0;
   *preamble_bytes = setup.info.main_size;
   *preamble_words = sizes[nir_preamble_class_general];
   free(setup.binary);
   free(candidate.binary);
   _mesa_hash_table_destroy(original_blocks, NULL);
   free(eligibility.state);
   ralloc_free(preamble);
   ralloc_free(main);
   return false;
}

static bool
apple9_preamble_trial(nir_shader *nir, struct agx_shader_part *out,
                     struct agx_apple9_compute_profile *profile,
                     const struct agx_apple9_varying_layout *varyings,
                     const struct apple9_buffer_map *resources,
                     bool hoist_addresses, struct apple9_dag_cost *cost,
                     char **log, size_t *log_size, unsigned *address_hoists,
                     bool *setup_retried)
{
   const bool capture_log =
      (agx_apple9_debug_flags() &
       (AGX_A9_DBG_ASM | AGX_A9_DBG_BINARY | AGX_A9_DBG_STATS |
        AGX_A9_DBG_TRACE | AGX_A9_DBG_FINAL_NIR)) != 0;
   FILE *stream = capture_log ? open_memstream(log, log_size) : NULL;
   FILE *previous = stream ? apple9_swap_log_stream(stream) : NULL;
   bool ok = false;
   unsigned budget =
      AGX_APPLE9_UNIFORM_COUNT -
      (resources->descriptor_tables
          ? MAX2(AGX_APPLE9_GRAPHICS_ROOT_WORDS, resources->native_root_words)
          : agx_apple9_compute_preamble_base(resources->count));
   bool promote_constants = true;
   /* Constant stores must not discard an otherwise useful preamble when
    * they cross the code-size policy bound. Retry without them first.
    * If the extracted expressions alone exceed the bound, reduce their
    * uniform budget using the measured setup size. Setup code has no
    * scratch, so when it spills, first stop moving guarded reads out of their
    * branches (the reads and address chains they bring are what fill the
    * setup's registers), then halve the budget. Otherwise the shader would
    * lose its whole preamble and run all of that uniform work per thread. Every
    * attempt starts from the same prepared NIR and still checks the
    * assembled code size. Bound retries so pathological setup code cannot
    * multiply compile time.
    */
   const unsigned max_bytes =
      agx_apple9_max_preamble_bytes(nir->info.stage == MESA_SHADER_COMPUTE);
   const int safe_reads_override = apple9_safe_reads_override;
   for (unsigned attempt = 0; attempt < 4 && budget; ++attempt) {
      unsigned bytes = 0, words = 0;
      bool setup_spilled = false;
      if (apple9_try_preamble(nir, out, profile, varyings, resources, budget,
                             promote_constants, hoist_addresses, cost, &bytes, &words,
                             &setup_spilled, address_hoists)) {
         ok = true;
         break;
      }
      if (setup_spilled && setup_retried)
         *setup_retried = true;
      if (setup_spilled && apple9_safe_reads_enabled()) {
         if (agx_apple9_trace_enabled())
            fprintf(stderr, "APPLE9_PREAMBLE_RETRY setup spilled: safe reads dropped\n");
         apple9_safe_reads_override = 0;
      } else if (setup_spilled) {
         budget = MIN2(budget, words) / 2;
         if (agx_apple9_trace_enabled())
            fprintf(stderr, "APPLE9_PREAMBLE_RETRY setup spilled: budget %u\n", budget);
      } else if (bytes <= max_bytes) {
         break;
      } else if (promote_constants) {
         promote_constants = false;
      } else {
         budget = (uint64_t)MIN2(budget, words) *
                  max_bytes / bytes;
         /* Shared setup dependencies make code size nonlinear in result
          * words. Reserve the last retry for a substantial reduction if
          * the proportional estimate has not fitted the setup yet. */
         if (attempt == 2)
            budget = MIN2(budget, words / 2);
      }
   }
   apple9_safe_reads_override = safe_reads_override;
   if (stream)
      apple9_swap_log_stream(previous);
   if (stream)
      fclose(stream);
   return ok;
}

static bool
apple9_compile_dag_tail(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const struct agx_apple9_varying_layout *varyings,
                        const struct agx_apple9_mesh_key *native_key,
                        const struct apple9_buffer_map *reserved_resources,
                        bool allow_preamble, const char **reason);

bool
apple9_compile_dag(nir_shader *nir, struct agx_shader_part *out,
                   struct agx_apple9_compute_profile *profile,
                   const struct agx_apple9_varying_layout *varyings,
                   const struct agx_apple9_mesh_key *native_key,
                   const struct apple9_buffer_map *reserved_resources,
                   bool allow_preamble, const char **reason)
{
   bool has_tile_coords = false;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      bool tile_writes = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            has_tile_coords |= op == nir_intrinsic_load_tile_pixel_agx;
            tile_writes |= op == nir_intrinsic_store_local_pixel_agx;
         }
      }
      if (has_tile_coords && tile_writes) {
         *reason =
            "Apple9 mixed tile-coordinate reads with tile writes is unsupported";
         return false;
      }
      if (has_tile_coords)
         allow_preamble = false;
   }

   /* Preamble extraction runs divergence analysis, which requires generic
    * sink pointers to have been lowered to this driver's concrete addresses.
    * In particular, writes to the read-only zero page use the scratch page.
    */
   agx_nir_lower_sink_address(nir);
   apple9_lower_texture_handles(nir);
   /* As in agx_optimize_nir, reassociate before forming the preamble:
    * ordering chains by rank exposes uniform subexpressions to it and lets
    * CSE share common pairs. Buffer addresses are still 64-bit here, so the
    * later 2x32 split sees the carry chains it expects. The loop runs
    * algebraic cleanup, so give it this backend's options: reordered
    * shifted-byte chains otherwise become pack_32_4x8, which Apple9 lacks.
    */
   if (!has_tile_coords) {
      const nir_shader_compiler_options *saved = nir->options;
      nir_shader_compiler_options options = *saved;
      options.lower_pack_unorm_2x16 = false;
      options.lower_pack_unorm_4x8 = false;
      options.lower_pack_snorm_2x16 = false;
      options.lower_pack_snorm_4x8 = false;
      options.lower_unpack_unorm_2x16 = false;
      options.lower_unpack_unorm_4x8 = false;
      options.lower_unpack_snorm_2x16 = false;
      options.lower_unpack_snorm_4x8 = false;
      options.lower_bitfield_extract = true;
      options.has_bfe = false;
      options.has_pack_32_4x8 = false;
      nir->options = &options;
      nir_opt_reassociate_loop(
         nir, nir_reassociate_scalar_math | nir_reassociate_cse_heuristic);
      nir->options = saved;
   }
   apple9_prepare_dag(nir);
   return apple9_compile_dag_tail(nir, out, profile, varyings,
                                  native_key, reserved_resources,
                                  allow_preamble, reason);
}

static bool
apple9_compile_dag_tail(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const struct agx_apple9_varying_layout *varyings,
                        const struct agx_apple9_mesh_key *native_key,
                        const struct apple9_buffer_map *reserved_resources,
                        bool allow_preamble, const char **reason)
{
   struct apple9_buffer_map resources = {
      .native_root_words = native_key
                             ? agx_apple9_mesh_argument_words(native_key) : 0,
      .native_key = native_key,
   };
   if (reserved_resources) {
      assert(reserved_resources->count <= ARRAY_SIZE(resources.resource));
      resources.count = reserved_resources->count;
      resources.reserved_count = reserved_resources->count;
      resources.force_descriptor_tables =
         reserved_resources->force_descriptor_tables;
      resources.tile_mode = reserved_resources->tile_mode;
      resources.force_abi_return = reserved_resources->force_abi_return;
      resources.defer_zs_release = reserved_resources->defer_zs_release;
      resources.cf_base = reserved_resources->cf_base;
      resources.z_cf_index = reserved_resources->z_cf_index;
      memcpy(resources.resource, reserved_resources->resource,
             reserved_resources->count * sizeof(resources.resource[0]));
   }
   if (!apple9_collect_buffer_map(nir, &resources, reason))
      return false;

   if (!allow_preamble) {
      return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                     NULL, false, NULL, reason);
   }

   struct apple9_dag_cost address_cost = {0}, original_cost = {0};
   char *address_log = NULL, *original_log = NULL;
   size_t address_size = 0, original_size = 0;
   unsigned address_hoists = 0;
   apple9_prepared_main = NULL;
   nir_shader *prepared = apple9_prepare_preamble_main(nir);
   apple9_prepared_main = prepared;
   /* Moving uniform addresses competes with descriptor values and constants
    * for preamble storage. Run the address-hoisted preamble first (the common
    * non-spilling case finishes in a single compile). Only if it fails, retries
    * setup, or spills after hoisting addresses, compare against the unhoisted
    * placement and keep the lower-cost result. */
   struct agx_shader_part original = {0};
   struct agx_apple9_compute_profile original_profile = {0};
   bool address_retried = false;
   bool ok = apple9_preamble_trial(nir, out, profile, varyings, &resources,
                                   true, &address_cost, &address_log,
                                   &address_size, &address_hoists,
                                   &address_retried);
   bool use_original = false;
   if (!ok || (address_cost.spills && address_hoists) || address_retried) {
      bool original_ok = apple9_preamble_trial(
         nir, &original, profile ? &original_profile : NULL, varyings,
         &resources, false, &original_cost, &original_log, &original_size,
         NULL, NULL);
      use_original = original_ok &&
                     (!ok || address_cost.spills > original_cost.spills ||
                      apple9_dag_cost_less(&original_cost, &address_cost));
   }
   if (use_original) {
      if (ok)
         free(out->binary);
      *out = original;
      if (profile)
         *profile = original_profile;
      ok = true;
   } else {
      free(original.binary);
   }
   apple9_prepared_main = NULL;
   ralloc_free(prepared);
   const char *log = use_original ? original_log : address_log;
   size_t size = use_original ? original_size : address_size;
   if (log && size)
      fwrite(log, 1, size, apple9_log());
   free(address_log);
   free(original_log);
   if (ok)
      return true;

   return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                 NULL, false, NULL, reason);
}
