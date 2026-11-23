/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_shader.h"
#include "gravity/compiler/agx_apple9_ir.h"
#include "nir_builder.h"
#include "util/hash_table.h"

/* Apple9 (G16G) executes 8x8x8 SIMD-group MMA across all 32 lanes with a
 * single uniform layout across A, B, C, and D and across all element types.
 * Each lane (sr2 = 0..31) owns 2 horizontally adjacent elements in the same
 * row:
 *   row  = ((lane & 16) >> 2) | ((lane >> 1) & 3)   in [0..7]
 *   col0 = ((lane &  8) >> 1) | ((lane &  1) << 1)  in {0, 2, 4, 6}
 *   elem[0] = (row, col0), elem[1] = (row, col0 + 1).
 */
#define HK_APPLE9_CMAT_DIM 8u
#define HK_APPLE9_CMAT_LEN 2u

struct hk_lower_cmat_ctx {
   struct hash_table *type_mapping;
};

static const struct glsl_type *
remap_matrix_type(struct hk_lower_cmat_ctx *ctx, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(ctx->type_mapping, orig);
   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;
   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description desc = *glsl_get_cmat_description(orig);
      assert(desc.rows == HK_APPLE9_CMAT_DIM && desc.cols == HK_APPLE9_CMAT_DIM);
      new_type = glsl_vector_type(desc.element_type, HK_APPLE9_CMAT_LEN);
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *elem = glsl_get_array_element(orig);
      const struct glsl_type *new_elem = remap_matrix_type(ctx, elem);
      if (new_elem != elem) {
         new_type = glsl_array_type(new_elem, glsl_get_length(orig),
                                    glsl_get_explicit_stride(orig));
      }
   } else if (glsl_type_is_struct(orig)) {
      unsigned num_fields = glsl_get_length(orig);
      bool changed = false;
      for (unsigned i = 0; i < num_fields; ++i) {
         const struct glsl_type *ft = glsl_get_struct_field(orig, i);
         if (remap_matrix_type(ctx, ft) != ft) {
            changed = true;
            break;
         }
      }
      if (changed) {
         struct glsl_struct_field *fields =
            malloc(sizeof(*fields) * num_fields);
         if (!fields)
            return orig;
         for (unsigned i = 0; i < num_fields; ++i) {
            fields[i] = *glsl_get_struct_field_data(orig, i);
            fields[i].type = remap_matrix_type(ctx, fields[i].type);
         }
         new_type = glsl_struct_type(fields, num_fields,
                                     glsl_get_type_name(orig),
                                     glsl_struct_type_is_packed(orig));
         free(fields);
      }
   }

   _mesa_hash_table_insert(ctx->type_mapping, orig, (void *)new_type);
   return new_type;
}

static bool
remap_type_in_place(struct hk_lower_cmat_ctx *ctx, const struct glsl_type **type)
{
   const struct glsl_type *new_type = remap_matrix_type(ctx, *type);
   if (new_type == *type)
      return false;

   *type = new_type;
   return true;
}

static struct glsl_cmat_description
cmat_src_desc(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static nir_def *
load_cmat_src(nir_builder *b, nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(deref->type);

   return nir_build_load_deref(b, HK_APPLE9_CMAT_LEN,
                               glsl_base_type_bit_size(desc.element_type),
                               &deref->def, 0);
}

static void
store_cmat_src(nir_builder *b, nir_src dst, nir_def *val)
{
   nir_store_deref(b, nir_src_as_deref(dst), val,
                   nir_component_mask(val->num_components));
}

static enum agx_apple9_mma_type
glsl_to_apple9_mma_type(enum glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_FLOAT:
      return AGX_APPLE9_MMA_F32;
   case GLSL_TYPE_FLOAT16:
      return AGX_APPLE9_MMA_F16;
   case GLSL_TYPE_BFLOAT16:
      return AGX_APPLE9_MMA_BF16;
   default:
      UNREACHABLE("Unsupported Apple9 cooperative matrix element type");
   }
}

static bool
lower_cmat_load_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[!is_load]);
   const bool row_major =
      nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   const enum gl_access_qualifier access = nir_intrinsic_access(intr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);

   assert(glsl_type_is_vector_or_scalar(deref->type));
   const unsigned elem_size_B = glsl_base_type_bit_size(desc.element_type) / 8;
   const unsigned ptr_stride =
      (glsl_get_bit_size(deref->type) / 8) *
      glsl_get_vector_elements(deref->type);
   const unsigned idx_bits = deref->def.bit_size;
   const struct glsl_type *elem_type = glsl_scalar_type(desc.element_type);

   nir_deref_instr *base_deref =
      nir_build_deref_cast(b, &deref->def, deref->modes, deref->type,
                           ptr_stride);
   nir_def *stride = nir_u2uN(b, intr->src[2].ssa, idx_bits);

   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row = nir_ior(b, nir_ushr_imm(b, nir_iand_imm(b, lane, 16), 2),
                             nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3));
   nir_def *col0 = nir_ior(b, nir_ushr_imm(b, nir_iand_imm(b, lane, 8), 1),
                              nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 1));

   nir_def *elems[HK_APPLE9_CMAT_LEN];
   if (!is_load) {
      nir_def *src = load_cmat_src(b, intr->src[!is_load]);
      for (unsigned p = 0; p < HK_APPLE9_CMAT_LEN; ++p)
         elems[p] = nir_channel(b, src, p);
   }

   nir_deref_instr *row_deref = NULL;
   if (row_major) {
      nir_def *row_off = nir_imul(b, nir_u2uN(b, row, idx_bits), stride);
      row_deref = nir_build_deref_ptr_as_array(b, base_deref, row_off);
      row_deref = nir_build_deref_cast(b, &row_deref->def, deref->modes,
                                       elem_type, elem_size_B);
   }

   for (unsigned p = 0; p < HK_APPLE9_CMAT_LEN; ++p) {
      nir_def *col = p ? nir_iadd_imm(b, col0, 1) : col0;
      nir_deref_instr *e;
      if (row_major) {
         e = nir_build_deref_ptr_as_array(b, row_deref,
                                          nir_u2uN(b, col, idx_bits));
      } else {
         nir_def *col_off = nir_imul(b, nir_u2uN(b, col, idx_bits), stride);
         e = nir_build_deref_ptr_as_array(b, base_deref, col_off);
         e = nir_build_deref_cast(b, &e->def, deref->modes, elem_type,
                                  elem_size_B);
         e = nir_build_deref_ptr_as_array(b, e, nir_u2uN(b, row, idx_bits));
      }

      if (is_load)
         elems[p] = nir_load_deref_with_access(b, e, access);
      else
         nir_store_deref_with_access(b, e, elems[p], 1, access);
   }

   if (is_load)
      store_cmat_src(b, intr->src[0], nir_vec(b, elems, HK_APPLE9_CMAT_LEN));

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_instr(nir_builder *b, nir_instr *instr, struct hk_lower_cmat_ctx *ctx)
{
   /* Processed in reverse so cmat intrinsics still see original cmat derefs. */
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      return remap_type_in_place(ctx, &deref->type);
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct: {
      nir_def *r = nir_replicate(b, intr->src[1].ssa, HK_APPLE9_CMAT_LEN);
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_length: {
      nir_def_replace(&intr->def, nir_imm_int(b, HK_APPLE9_CMAT_LEN));
      return true;
   }

   case nir_intrinsic_cmat_extract: {
      nir_def *mat = load_cmat_src(b, intr->src[0]);
      nir_def *elem = nir_vector_extract(b, mat, intr->src[1].ssa);
      nir_def_replace(&intr->def, elem);
      return true;
   }

   case nir_intrinsic_cmat_insert: {
      nir_def *mat = load_cmat_src(b, intr->src[2]);
      nir_def *r =
         nir_vector_insert(b, mat, intr->src[1].ssa, intr->src[3].ssa);
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_copy: {
      nir_build_copy_deref(b, intr->src[0].ssa, intr->src[1].ssa);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
      return lower_cmat_load_store(b, intr);

   case nir_intrinsic_cmat_muladd: {
      struct glsl_cmat_description dst_desc = cmat_src_desc(intr->src[0]);
      struct glsl_cmat_description a_desc = cmat_src_desc(intr->src[1]);
      struct glsl_cmat_description b_desc = cmat_src_desc(intr->src[2]);
      struct glsl_cmat_description c_desc = cmat_src_desc(intr->src[3]);

      enum agx_apple9_mma_type ty_a = glsl_to_apple9_mma_type(a_desc.element_type);
      enum agx_apple9_mma_type ty_b = glsl_to_apple9_mma_type(b_desc.element_type);
      enum agx_apple9_mma_type ty_c = glsl_to_apple9_mma_type(c_desc.element_type);
      enum agx_apple9_mma_type ty_dst =
         glsl_to_apple9_mma_type(dst_desc.element_type);

      nir_def *a = load_cmat_src(b, intr->src[1]);
      nir_def *b_mat = load_cmat_src(b, intr->src[2]);
      nir_def *c = load_cmat_src(b, intr->src[3]);

      if (a->bit_size == 16)
         a = nir_pack_bits(b, a, 32);
      if (b_mat->bit_size == 16)
         b_mat = nir_pack_bits(b, b_mat, 32);
      if (c->bit_size == 16)
         c = nir_pack_bits(b, c, 32);

      unsigned dst_comps = agx_apple9_mma_type_gprs(ty_dst);
      uint32_t flags = agx_apple9_mma_flags(ty_a, ty_b, ty_c, ty_dst, true);
      nir_def *d =
         nir_cmat_muladd_agx(b, dst_comps, a, b_mat, c, .flags = flags);

      if (dst_desc.element_type == GLSL_TYPE_FLOAT16)
         d = nir_unpack_bits(b, d, 16);

      store_cmat_src(b, intr->src[0], d);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op: {
      nir_def *a = load_cmat_src(b, intr->src[1]);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *r;
      if (intr->intrinsic == nir_intrinsic_cmat_unary_op) {
         r = nir_build_alu1(b, nir_intrinsic_alu_op(intr), a);
      } else {
         nir_def *y = intr->intrinsic == nir_intrinsic_cmat_binary_op
                         ? load_cmat_src(b, intr->src[2])
                         : intr->src[2].ssa;
         r = nir_build_alu2(b, nir_intrinsic_alu_op(intr), a, y);
      }

      b->fp_math_ctrl = nir_fp_fast_math;
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_bitcast: {
      store_cmat_src(b, intr->src[0], load_cmat_src(b, intr->src[1]));
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_convert: {
      struct glsl_cmat_description dst = cmat_src_desc(intr->src[0]);
      struct glsl_cmat_description src = cmat_src_desc(intr->src[1]);
      nir_def *ret = load_cmat_src(b, intr->src[1]);

      if (dst.element_type != src.element_type) {
         b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
         nir_op op = nir_type_conversion_op(
            nir_get_nir_type_for_glsl_base_type(src.element_type),
            nir_get_nir_type_for_glsl_base_type(dst.element_type),
            nir_rounding_mode_undef);
         ret = nir_build_alu1(b, op, ret);
         b->fp_math_ctrl = nir_fp_fast_math;
      }

      store_cmat_src(b, intr->src[0], ret);
      nir_instr_remove(instr);
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_cmat_impl(nir_function_impl *impl, struct hk_lower_cmat_ctx *ctx)
{
   bool progress = false;

   nir_foreach_function_temp_variable(var, impl) {
      if (remap_type_in_place(ctx, &var->type))
         progress = true;
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         if (lower_cmat_instr(&b, instr, ctx))
            progress = true;
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
hk_nir_lower_cooperative_matrix(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       !nir->info.cs.has_cooperative_matrix)
      return false;

   /* Split struct-typed local/shader-temp variables before dimension splitting
    * so 16x16 cooperative matrix members inside structs are split cleanly. */
   bool progress = false;
   NIR_PASS(progress, nir, nir_split_var_copies);
   NIR_PASS(progress, nir, nir_split_struct_vars,
            nir_var_function_temp | nir_var_shader_temp);

   struct nir_lower_coopmat_args args = {
      .m_gran = HK_APPLE9_CMAT_DIM,
      .n_gran = HK_APPLE9_CMAT_DIM,
      .k_gran = HK_APPLE9_CMAT_DIM,
   };
   bool split_progress =
      nir_lower_cooperative_matrix_flexible_dimensions(nir, &args);
   progress |= split_progress;
   NIR_PASS(progress, nir, nir_opt_deref);
   NIR_PASS(progress, nir, nir_opt_dce);
   NIR_PASS(progress, nir, nir_remove_dead_variables,
            nir_var_function_temp | nir_var_shader_temp, NULL);

   struct hk_lower_cmat_ctx ctx = {
      .type_mapping = _mesa_pointer_hash_table_create(NULL),
   };

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      if (remap_type_in_place(&ctx, &var->type))
         progress = true;
   }

   nir_foreach_function_impl(impl, nir) {
      if (lower_cmat_impl(impl, &ctx))
         progress = true;
   }

   _mesa_hash_table_destroy(ctx.type_mapping, NULL);
   return progress;
}
