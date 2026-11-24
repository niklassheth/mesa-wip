/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_fused_vs.h"

bool
hk_nir_fused_validate_vs(const nir_shader *vs,
                         const struct agx_apple9_vertex_layout *vertex,
                         const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (!vs || vs->info.stage != MESA_SHADER_VERTEX || !vs->info.io_lowered)
      REJECT("fused VS requires lowered vertex NIR");
   if (vs->xfb_info || vs->info.writes_memory ||
       vs->info.outputs_written_16bit || vs->info.inputs_read_16bit)
      REJECT("fused VS excludes XFB, memory writes, and 16-bit IO");

   unsigned vs_impls = 0;
   nir_foreach_function_impl(impl, vs) {
      if (++vs_impls > 1)
         REJECT("fused VS requires a single entrypoint");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("fused VS requires inlined calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_input: {
               if (vertex && vertex->use_prolog)
                  REJECT("fused VS does not support dynamic vertex input prologs");
               unsigned loc = nir_intrinsic_io_semantics(intr).location;
               if (loc < VERT_ATTRIB_GENERIC0 ||
                   loc - VERT_ATTRIB_GENERIC0 >= 16 ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported VS vertex input");
               break;
            }
            case nir_intrinsic_store_output: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused VS requires constant output slot offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                         intr->src[0].ssa->num_components >
                      4)
                  REJECT("unsupported VS output store");
               break;
            }
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT("fused VS requires lowered external dereferences");
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_ro_sink_address_poly:
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_vertex_id_zero_base:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_first_vertex:
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_draw_id:
            case nir_intrinsic_load_constant:
            case nir_intrinsic_image_load_apple9_agx:
               break;
            default:
               REJECT("unsupported VS intrinsic in fused stage");
            }
         }
      }
   }
   return true;
#undef REJECT
}

bool
hk_nir_fused_prepare_vs(nir_shader *vs,
                        const struct agx_apple9_vertex_layout *vertex,
                        const char **reason)
{
   bool vs_has_inputs = false;
   nir_foreach_function_impl(vimpl, vs) {
      nir_foreach_block(block, vimpl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_input) {
               unsigned loc = nir_intrinsic_io_semantics(intr).location;
               nir_intrinsic_set_base(intr, loc - VERT_ATTRIB_GENERIC0);
               vs_has_inputs = true;
            }
         }
      }
   }
   if (vs_has_inputs) {
      if (!vertex || !agx_nir_lower_apple9_vertex_inputs(vs, vertex, true)) {
         if (reason)
            *reason = "unsupported VS vertex input format in fused stage";
         return false;
      }
   }
   NIR_PASS(_, vs, nir_lower_returns);
   return true;
}

nir_def *
hk_nir_fused_param_u32(nir_builder *b, nir_def *buf, unsigned offset)
{
   return nir_load_global_constant(
      b, 1, 32, nir_iadd_imm(b, buf, offset), .align_mul = 4);
}

nir_def *
hk_nir_fused_param_u64(nir_builder *b, nir_def *buf, unsigned offset)
{
   return nir_load_global_constant(
      b, 1, 64, nir_iadd_imm(b, buf, offset), .align_mul = 8);
}

nir_def *
hk_nir_fused_fetch_vertex_id_zero_base(nir_builder *b, nir_def *vertex_pos,
                                       nir_def *index_buffer,
                                       nir_def *index_buffer_size,
                                       nir_def *index_size_B)
{
   nir_def *indexed_id = NULL;
   nir_push_if(b, nir_ine_imm(b, index_size_B, 0));
   {
      nir_def *byte_off =
         nir_imul(b, nir_u2u64(b, vertex_pos), nir_u2u64(b, index_size_B));
      nir_def *size_m1 = nir_u2u64(b, nir_iadd_imm(b, index_size_B, -1));
      nir_def *limit = nir_bcsel(b, nir_uge(b, index_buffer_size, size_m1),
                                 nir_isub(b, index_buffer_size, size_m1),
                                 nir_imm_int64(b, 0));
      nir_def *in_bounds = nir_ult(b, byte_off, limit);
      nir_def *ro_sink = nir_load_ro_sink_address_poly(b);
      nir_def *addr =
         nir_bcsel(b, in_bounds, nir_iadd(b, index_buffer, byte_off), ro_sink);
      nir_def *idx32 = NULL, *idx16 = NULL, *idx8 = NULL, *idx_small = NULL;
      nir_push_if(b, nir_ieq_imm(b, index_size_B, 4));
      {
         idx32 = nir_load_global_constant(b, 1, 32, addr, .align_mul = 4);
      }
      nir_push_else(b, NULL);
      {
         nir_push_if(b, nir_ieq_imm(b, index_size_B, 2));
         {
            idx16 = nir_u2u32(
               b, nir_load_global_constant(b, 1, 16, addr, .align_mul = 2));
         }
         nir_push_else(b, NULL);
         {
            idx8 = nir_u2u32(
               b, nir_load_global_constant(b, 1, 8, addr, .align_mul = 1));
         }
         nir_pop_if(b, NULL);
         idx_small = nir_if_phi(b, idx16, idx8);
      }
      nir_pop_if(b, NULL);
      nir_def *loaded = nir_if_phi(b, idx32, idx_small);
      indexed_id = nir_bcsel(b, in_bounds, loaded, nir_imm_int(b, 0));
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, indexed_id, vertex_pos);
}

bool
hk_nir_fused_lower_vs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct hk_nir_fused_vs_lower *v = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      if (loc < VARYING_SLOT_MAX && v->vs_read_mask[loc] && v->vs_slot_vars[loc]) {
         nir_def *val = nir_undef(b, 4, 32);
         unsigned comp = nir_intrinsic_component(intr);
         for (unsigned i = 0; i < intr->src[0].ssa->num_components; ++i)
            val = nir_vector_insert_imm(
               b, val, nir_channel(b, intr->src[0].ssa, i), comp + i);
         nir_store_var(b, v->vs_slot_vars[loc], val,
                       nir_intrinsic_write_mask(intr) << comp);
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_load_vertex_id:
      nir_def_replace(&intr->def, nir_load_var(b, v->in_vertex_id));
      return true;
   case nir_intrinsic_load_vertex_id_zero_base:
      nir_def_replace(&intr->def, nir_load_var(b, v->in_vertex_pos));
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, nir_load_var(b, v->in_instance_id));
      return true;
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
      nir_def_replace(&intr->def, nir_load_var(b, v->in_first_vertex));
      return true;
   case nir_intrinsic_load_base_instance:
      nir_def_replace(&intr->def, nir_load_var(b, v->in_base_instance));
      return true;
   default:
      return false;
   }
}
