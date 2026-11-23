/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "gravity/compiler/agx_apple9_ir.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/vulkan/hk_physical_device.h"
#include "gravity/vulkan/hk_shader.h"
#include "hk_entrypoints.h"
#include "compiler/nir/nir_builder.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return false; \
} } while (0)

static bool
struct_wrapped_16x16_mixed_and_volatile(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                              &agx_nir_options, "coop_matrix_struct_16x16");
   b.shader->info.cs.has_cooperative_matrix = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   struct glsl_cmat_description desc_a = {
      .element_type = GLSL_TYPE_FLOAT16,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_A,
   };
   struct glsl_cmat_description desc_b = {
      .element_type = GLSL_TYPE_FLOAT16,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_B,
   };
   struct glsl_cmat_description desc_c = {
      .element_type = GLSL_TYPE_FLOAT16,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_ACCUMULATOR,
   };
   struct glsl_cmat_description desc_d = {
      .element_type = GLSL_TYPE_FLOAT,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_ACCUMULATOR,
   };

   struct glsl_struct_field fields[4] = {
      { .type = glsl_cmat_type(&desc_a), .name = "a" },
      { .type = glsl_cmat_type(&desc_b), .name = "b" },
      { .type = glsl_cmat_type(&desc_c), .name = "c" },
      { .type = glsl_cmat_type(&desc_d), .name = "d" },
   };
   const struct glsl_type *s_type = glsl_struct_type(fields, 4, "Mats", false);

   nir_variable *s_var = nir_local_variable_create(b.impl, s_type, "s");
   nir_variable *c_copy_var =
      nir_local_variable_create(b.impl, glsl_cmat_type(&desc_c), "c_copy");

   nir_deref_instr *s_deref = nir_build_deref_var(&b, s_var);
   nir_deref_instr *da = nir_build_deref_struct(&b, s_deref, 0);
   nir_deref_instr *db = nir_build_deref_struct(&b, s_deref, 1);
   nir_deref_instr *dc = nir_build_deref_struct(&b, s_deref, 2);
   nir_deref_instr *dd = nir_build_deref_struct(&b, s_deref, 3);
   nir_deref_instr *dc_copy = nir_build_deref_var(&b, c_copy_var);

   nir_deref_instr *ptr_a = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x10000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT16), 2);
   nir_deref_instr *ptr_b = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x20000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT16), 2);
   nir_deref_instr *ptr_c = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x30000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT16), 2);
   nir_deref_instr *ptr_d = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x40000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT), 4);
   nir_deref_instr *ptr_c_out = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x50000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT16), 2);

   nir_def *stride = nir_imm_int(&b, 16);
   nir_cmat_load(&b, &da->def, &ptr_a->def, stride,
                 .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR,
                 .access = ACCESS_VOLATILE);
   nir_cmat_load(&b, &db->def, &ptr_b->def, stride,
                 .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_load(&b, &dc->def, &ptr_c->def, stride,
                 .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);

   /* Copy struct-member 16x16 matrix to a standalone 16x16 variable. */
   nir_cmat_copy(&b, &dc_copy->def, &dc->def);
   /* Multiply-accumulate F16*F16 + F16 -> F32 at 16x16x16 (splits into 2x2x2). */
   nir_cmat_muladd(&b, &dd->def, &da->def, &db->def, &dc_copy->def);
   /* Store both D and original C copy (C must not be clobbered by k=0 split). */
   nir_cmat_store(&b, &ptr_d->def, &dd->def, stride,
                  .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_store(&b, &ptr_c_out->def, &dc_copy->def, stride,
                  .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR,
                  .access = ACCESS_VOLATILE);

   struct hk_physical_device pdev = {0};
   pdev.dev.chip = AGX_CHIP_G16G;
   hk_preprocess_nir_internal(&pdev.vk, b.shader);
   nir_validate_shader(b.shader, "coop_matrix_struct_16x16");

   unsigned mma_f16_c = 0, mma_f32_c = 0;
   unsigned volatile_loads = 0, volatile_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(intr->intrinsic != nir_intrinsic_cmat_load);
         CHECK(intr->intrinsic != nir_intrinsic_cmat_store);
         CHECK(intr->intrinsic != nir_intrinsic_cmat_muladd);
         CHECK(intr->intrinsic != nir_intrinsic_cmat_copy);
         if (intr->intrinsic == nir_intrinsic_cmat_muladd_agx) {
            uint32_t flags = nir_intrinsic_flags(intr);
            CHECK(agx_apple9_mma_src_type(flags, 0) == AGX_APPLE9_MMA_F16);
            CHECK(agx_apple9_mma_src_type(flags, 1) == AGX_APPLE9_MMA_F16);
            CHECK(agx_apple9_mma_dst_type(flags) == AGX_APPLE9_MMA_F32);
            CHECK(agx_apple9_mma_has_c(flags));
            if (agx_apple9_mma_src_type(flags, 2) == AGX_APPLE9_MMA_F16)
               ++mma_f16_c;
            else if (agx_apple9_mma_src_type(flags, 2) == AGX_APPLE9_MMA_F32)
               ++mma_f32_c;
         } else if (intr->intrinsic == nir_intrinsic_load_deref &&
                    (nir_intrinsic_access(intr) & ACCESS_VOLATILE)) {
            ++volatile_loads;
         } else if (intr->intrinsic == nir_intrinsic_store_deref &&
                    (nir_intrinsic_access(intr) & ACCESS_VOLATILE)) {
            ++volatile_stores;
         }
      }
   }

   /* 16x16x16 splits into 2x2 output tiles x 2 k-slices:
    * 4 k=0 MMAs consume F16 C -> F32 acc; 4 k=1 MMAs consume F32 acc -> F32 dst. */
   CHECK(mma_f16_c == 4);
   CHECK(mma_f32_c == 4);
   /* 4 split 8x8 tiles x 2 elements/lane = 8 volatile scalar loads & stores. */
   CHECK(volatile_loads == 8);
   CHECK(volatile_stores == 8);

   ralloc_free(b.shader);
   return true;
}

static bool
advertised_properties(void)
{
   struct hk_physical_device pdev = {0};
   pdev.vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   VkPhysicalDevice handle = hk_physical_device_to_handle(&pdev);
   uint32_t count = 0;
   CHECK(hk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR(handle, &count, NULL) ==
         VK_SUCCESS);
   CHECK(count == 8);

   VkCooperativeMatrixPropertiesKHR props[8] = {0};
   for (unsigned i = 0; i < 8; ++i)
      props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
   CHECK(hk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR(handle, &count, props) ==
         VK_SUCCESS);
   CHECK(count == 8);
   for (unsigned i = 0; i < 8; ++i) {
      CHECK(props[i].scope == VK_SCOPE_SUBGROUP_KHR);
      CHECK(props[i].saturatingAccumulation == VK_FALSE);
      unsigned dim = i < 4 ? 8 : 16;
      CHECK(props[i].MSize == dim && props[i].NSize == dim && props[i].KSize == dim);
   }
   return true;
}

static bool
unused_volatile_load_and_dead_16x16_local(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                              &agx_nir_options, "coop_matrix_volatile_unused");
   b.shader->info.cs.has_cooperative_matrix = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   struct glsl_cmat_description desc8 = {
      .element_type = GLSL_TYPE_FLOAT,
      .scope = SCOPE_SUBGROUP,
      .rows = 8, .cols = 8,
      .use = GLSL_CMAT_USE_A,
   };
   struct glsl_cmat_description desc16 = {
      .element_type = GLSL_TYPE_FLOAT,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_A,
   };

   /* Unused 16x16 local must be cleaned up cleanly without hitting 8x8 assert. */
   nir_local_variable_create(b.impl, glsl_cmat_type(&desc16), "dead_16x16");

   /* Unused volatile 8x8 load must NOT be eliminated by DCE. */
   nir_variable *v8 =
      nir_local_variable_create(b.impl, glsl_cmat_type(&desc8), "vol_8x8");
   nir_deref_instr *d8 = nir_build_deref_var(&b, v8);
   nir_deref_instr *ptr = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x10000), nir_var_mem_global,
      glsl_scalar_type(GLSL_TYPE_FLOAT), 4);
   nir_cmat_load(&b, &d8->def, &ptr->def, nir_imm_int(&b, 8),
                 .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR,
                 .access = ACCESS_VOLATILE);

   struct hk_physical_device pdev = {0};
   pdev.dev.chip = AGX_CHIP_G16G;
   hk_preprocess_nir_internal(&pdev.vk, b.shader);
   nir_validate_shader(b.shader, "coop_matrix_volatile_unused");

   unsigned volatile_loads = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref &&
             (nir_intrinsic_access(intr) & ACCESS_VOLATILE))
            ++volatile_loads;
      }
   }
   CHECK(volatile_loads == 2);

   /* Unused 16x16 local alone must report progress when cleaned up. */
   {
      nir_builder b2 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                 &agx_nir_options, "coop_matrix_dead_only");
      b2.shader->info.cs.has_cooperative_matrix = true;
      b2.shader->info.workgroup_size[0] = 32;
      b2.shader->info.workgroup_size[1] = 1;
      b2.shader->info.workgroup_size[2] = 1;
      nir_local_variable_create(b2.impl, glsl_cmat_type(&desc16), "dead_only");
      CHECK(hk_nir_lower_cooperative_matrix(b2.shader));
      nir_validate_shader(b2.shader, "coop_matrix_dead_only");
      CHECK(exec_list_is_empty(&b2.impl->locals));
      ralloc_free(b2.shader);
   }

   /* Recursive DCE (nir_instr_free_and_dce) must also preserve ACCESS_VOLATILE. */
   {
      nir_builder b3 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                 &agx_nir_options, "coop_matrix_free_and_dce");
      nir_deref_instr *p = nir_build_deref_cast(
         &b3, nir_imm_int64(&b3, 0x20000), nir_var_mem_global,
         glsl_scalar_type(GLSL_TYPE_UINT), 4);
      nir_def *vol = nir_load_deref_with_access(&b3, p, ACCESS_VOLATILE);
      nir_def *dead_consumer = nir_iadd_imm(&b3, vol, 1);
      nir_instr_free_and_dce(nir_def_instr(dead_consumer));
      nir_validate_shader(b3.shader, "coop_matrix_free_and_dce");

      bool vol_survived = false;
      nir_foreach_block(block, b3.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_deref &&
                (nir_intrinsic_access(nir_instr_as_intrinsic(instr)) & ACCESS_VOLATILE))
               vol_survived = true;
         }
      }
      CHECK(vol_survived);
      ralloc_free(b3.shader);
   }

   ralloc_free(b.shader);
   return true;
}

static bool
large_64bit_stride_no_wrap(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                              &agx_nir_options, "coop_matrix_large_stride");
   b.shader->info.cs.has_cooperative_matrix = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   struct glsl_cmat_description desc16 = {
      .element_type = GLSL_TYPE_FLOAT16,
      .scope = SCOPE_SUBGROUP,
      .rows = 16, .cols = 16,
      .use = GLSL_CMAT_USE_ACCUMULATOR,
   };
   nir_variable *v =
      nir_local_variable_create(b.impl, glsl_cmat_type(&desc16), "m16");
   nir_deref_instr *dv = nir_build_deref_var(&b, v);
   /* uvec4 (16B) pointer with 32-bit stride 0x20000000 -> row 8 is
    * 0x100000000 uvec4 elements (wraps to 0 if computed in 32-bit). */
   nir_deref_instr *ptr = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x100000), nir_var_mem_global,
      glsl_vector_type(GLSL_TYPE_UINT, 4), 16);
   nir_cmat_load(&b, &dv->def, &ptr->def, nir_imm_int(&b, 0x20000000u),
                 .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_store(&b, &ptr->def, &dv->def, nir_imm_int(&b, 0x20000000u),
                  .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);

   /* First run dimension splitting directly to verify split cmat_load/store
    * preserve the 32-bit stride bit-size (required by all NIR cmat backends). */
   struct nir_lower_coopmat_args args = {
      .m_gran = 8, .n_gran = 8, .k_gran = 8,
   };
   CHECK(nir_lower_cooperative_matrix_flexible_dimensions(b.shader, &args));
   nir_opt_constant_folding(b.shader);
   nir_validate_shader(b.shader, "coop_matrix_large_stride_split");

   unsigned split_loads = 0, split_stores = 0;
   uint64_t expected_offsets[4] = {
      0ull,
      8ull * 2ull,
      8ull * 0x20000000ull * 16ull,
      8ull * 0x20000000ull * 16ull + 8ull * 2ull,
   };
   unsigned load_mask = 0, store_mask = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_cmat_load &&
             intr->intrinsic != nir_intrinsic_cmat_store)
            continue;
         bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
         CHECK(intr->src[2].ssa->bit_size == 32);
         CHECK(nir_src_is_const(intr->src[2]) &&
               nir_src_as_uint(intr->src[2]) == 0x20000000u);
         nir_deref_instr *d = nir_src_as_deref(intr->src[is_load]);
         CHECK(d->type == glsl_vector_type(GLSL_TYPE_UINT, 4));

         /* Trace exact constant byte offset from ptr back through deref chain. */
         uint64_t byte_off = 0;
         for (nir_deref_instr *cur = d; cur != ptr;
              cur = nir_deref_instr_parent(cur)) {
            CHECK(cur != NULL);
            if (cur->deref_type == nir_deref_type_ptr_as_array) {
               CHECK(nir_src_is_const(cur->arr.index));
               byte_off += nir_src_as_uint(cur->arr.index) *
                           nir_deref_instr_array_stride(cur);
            }
         }
         unsigned match = 4;
         for (unsigned k = 0; k < 4; ++k) {
            if (byte_off == expected_offsets[k])
               match = k;
         }
         CHECK(match < 4);
         if (is_load) {
            load_mask |= 1u << match;
            ++split_loads;
         } else {
            store_mask |= 1u << match;
            ++split_stores;
         }
      }
   }
   CHECK(split_loads == 4 && load_mask == 0xf);
   CHECK(split_stores == 4 && store_mask == 0xf);

   struct hk_physical_device pdev = {0};
   pdev.dev.chip = AGX_CHIP_G16G;
   hk_preprocess_nir_internal(&pdev.vk, b.shader);
   nir_validate_shader(b.shader, "coop_matrix_large_stride");

   ralloc_free(b.shader);
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = struct_wrapped_16x16_mixed_and_volatile() &&
             advertised_properties() &&
             unused_volatile_load_and_dead_16x16_local() &&
             large_64bit_stride_no_wrap();
   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
