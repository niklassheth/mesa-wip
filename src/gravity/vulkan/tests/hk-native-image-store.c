/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <string.h>
#include "hk_descriptor_set.h"
#include "hk_descriptor_buffer.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_shader.h"
#include "compiler/nir/nir_builder.h"
#include "gravity/layout/layout.h"
#include "gravity/lib/agx_helpers.h"
#include "vk_format.h"
#include "util/u_math.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s\n", #c); return 1; } } while (0)

static int
test_image_storage(void)
{
   struct hk_image image = {0};
   CHECK(hk_apple9_native_store_image(&image, 0));

   /* Sparse images write natively too (unbound pages map scratch_bo RW in the
    * primary VA). Only lossless compression keeps the software fallback. */
   for (unsigned flags = 1; flags < 8; ++flags) {
      image.vk.create_flags =
         ((flags & 1) ? VK_IMAGE_CREATE_SPARSE_BINDING_BIT : 0) |
         ((flags & 2) ? VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT : 0) |
         ((flags & 4) ? VK_IMAGE_CREATE_SPARSE_ALIASED_BIT : 0);
      CHECK(hk_apple9_native_store_image(&image, 0));
   }

   struct agx_bo sparse_map = {0};
   image.planes[0].sparse_map = &sparse_map;
   image.vk.create_flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                           VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
   CHECK(hk_apple9_native_store_image(&image, 0));

   image.planes[0].sparse_map = NULL;
   image.vk.create_flags = VK_IMAGE_CREATE_ALIAS_BIT |
                           VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                           VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(hk_apple9_native_store_image(&image, 0));
   image.planes[0].layout.compressed = true;
   CHECK(!hk_apple9_native_store_image(&image, 0));

   /* Reinterpreting views of uncompressed images keep the image layout. */
   struct ail_layout rgba8 = {
      .width_px = 61, .height_px = 37, .depth_px = 1, .levels = 5,
      .sample_count_sa = 1, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .tiling = AIL_TILING_GPU, .writeable_image = true,
   };
   ail_make_miptree(&rgba8);
   struct ail_layout out;
   unsigned level;
   CHECK(hk_apple9_native_store_view_layout(&rgba8, PIPE_FORMAT_R32_FLOAT, 2, &out, &level));
   CHECK(level == 2 && out.size_B == rgba8.size_B && out.width_px == 61);
   CHECK(!hk_apple9_native_store_view_layout(&rgba8, PIPE_FORMAT_R16G16B16A16_FLOAT, 0,
                                             &out, &level));
   /* Block-texel views: a single-level layout over the level's blocks that
    * ail addresses exactly like the compressed level. */
   struct ail_layout bc1 = {
      .width_px = 1000, .height_px = 36, .depth_px = 1, .levels = 10,
      .sample_count_sa = 1, .format = PIPE_FORMAT_DXT1_RGB,
      .tiling = AIL_TILING_GPU, .writeable_image = true,
   };
   ail_make_miptree(&bc1);
   for (unsigned l = 0; l < bc1.levels; ++l) {
      CHECK(hk_apple9_native_store_view_layout(&bc1, PIPE_FORMAT_R32G32_UINT, l, &out, &level));
      CHECK(level == 0 && out.levels == 1 && out.format == PIPE_FORMAT_R32G32_UINT);
      CHECK(out.tilesize_el[0].width_el == bc1.tilesize_el[l].width_el &&
            out.tilesize_el[0].height_el == bc1.tilesize_el[l].height_el);
      CHECK(out.width_px >= util_format_get_nblocksx(bc1.format, u_minify(1000, l)));
   }
   CHECK(!hk_apple9_native_store_view_layout(&bc1, PIPE_FORMAT_R32G32B32A32_FLOAT, 0,
                                             &out, &level));
   puts("native storage: sparse, reinterpreting and block-texel views have entries");
   return 0;
}

int main(void)
{
   CHECK(test_image_storage() == 0);
   glsl_type_singleton_init_or_ref();
   /* Every Apple9 storage format has a native entry. */
   for (unsigned f = 0; f < PIPE_FORMAT_COUNT; ++f) {
      VkFormat vk = vk_format_from_pipe_format(f);
      if (vk != VK_FORMAT_UNDEFINED && hk_apple9_storage_format_supported(vk))
         CHECK(hk_apple9_native_store_format(f));
   }
   const enum pipe_format fallback[] = {
      PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_R9G9B9E5_FLOAT, PIPE_FORMAT_Z32_FLOAT,
      PIPE_FORMAT_DXT1_RGB,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fallback); ++i)
      CHECK(!hk_apple9_native_store_format(fallback[i]));
   CHECK(sizeof(struct hk_apple9_storage_descriptor) == 128);
   CHECK(HK_APPLE9_DB_STORAGE_SIZE == 128);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, pbe_sw) == 0);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, native_ok) == 24);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, extent) == 32);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, pbe) == 64);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, texture) == 96);
   CHECK(HK_APPLE9_DB_STORAGE_SIZE >= sizeof(struct hk_apple9_storage_descriptor));
   CHECK(util_is_power_of_two_nonzero(HK_APPLE9_DB_STORAGE_SIZE));
   CHECK(!(HK_APPLE9_STORAGE_READ_VALID & HK_APPLE9_STORAGE_WRITE_VALID));
   CHECK(!((HK_APPLE9_STORAGE_ENCODE_MASK << HK_APPLE9_STORAGE_ENCODE_SHIFT) &
           (HK_APPLE9_STORAGE_READ_VALID | HK_APPLE9_STORAGE_WRITE_VALID |
            HK_APPLE9_STORAGE_SPARSE)));
   CHECK(!((HK_APPLE9_STORAGE_TAIL_MASK << HK_APPLE9_STORAGE_TAIL_SHIFT) &
           (HK_APPLE9_STORAGE_READ_VALID | HK_APPLE9_STORAGE_WRITE_VALID |
            HK_APPLE9_STORAGE_SPARSE)));

   /* Layered storage views pack a single inline 2D-array native PBE anchored
    * at the view's base layer with its layer count in bits 44..59. */
   struct ail_layout layout = {
      .width_px = 8, .height_px = 8, .depth_px = 24, .levels = 1,
      .sample_count_sa = 1, .format = PIPE_FORMAT_R11G11B10_FLOAT,
      .tiling = AIL_TILING_TWIDDLED, .layer_stride_B = 0x4000,
   };
   struct agx_pbe_packed legacy = {0};
   ((uint8_t *)&legacy)[0] = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   struct agx_pbe_packed saved = legacy;
   uint8_t entry[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   uint64_t base = UINT64_C(0x2ffff00000) + ail_get_layer_offset_B(&layout, 3);
   CHECK(agx_apple9_translate_pbe(entry, &legacy, layout.format, &layout, base,
                                  0, 12, true, AGX_TEXTURE_DIMENSION_2D_ARRAY));
   CHECK((entry[0] & 0xf) == AGX_TEXTURE_DIMENSION_2D_ARRAY);
   uint64_t address;
   memcpy(&address, entry + 8, sizeof(address));
   CHECK((address & ((UINT64_C(1) << 40) - 1)) == base >> 4);
   CHECK(((address >> 44) & 0xffff) == 12 - 1);
   CHECK(!memcmp(&legacy, &saved, sizeof(legacy)));

   uint8_t null_pbe[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   agx_apple9_set_null_pbe(null_pbe);
   memcpy(&address, null_pbe + 8, sizeof(address));
   CHECK((address & ((UINT64_C(1) << 40) - 1)) == AGX_SCRATCH_PAGE_ADDRESS >> 4);
   /* Eligibility is frozen before any load is lowered/removed: a read after
    * a store must keep every store of the shader on the converter. */
   const nir_intrinsic_op reads[] = {
      nir_intrinsic_bindless_image_load, nir_intrinsic_bindless_image_atomic,
      nir_intrinsic_bindless_image_atomic_swap,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(reads); ++i) {
      const nir_shader_compiler_options options = {0};
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "read_write");
      CHECK(hk_apple9_native_store_shader(b.shader));
      const nir_intrinsic_info *store_info = &nir_intrinsic_infos[nir_intrinsic_bindless_image_store];
      nir_intrinsic_instr *store = nir_intrinsic_instr_create(b.shader, nir_intrinsic_bindless_image_store);
      store->num_components = 4;
      nir_intrinsic_set_image_dim(store, GLSL_SAMPLER_DIM_2D);
      nir_intrinsic_set_format(store, PIPE_FORMAT_R32_UINT);
      for (unsigned j = 0; j < store_info->num_srcs; ++j) {
         unsigned components = store_info->src_components[j] ? store_info->src_components[j] : 4;
         store->src[j] = nir_src_for_ssa(nir_imm_zero(&b, components, j == 0 ? 64 : 32));
      }
      nir_builder_instr_insert(&b, &store->instr);
      CHECK(hk_apple9_native_store_shader(b.shader));
      nir_intrinsic_instr *read = nir_intrinsic_instr_create(b.shader, reads[i]);
      read->num_components = reads[i] == nir_intrinsic_bindless_image_load ? 4 : 1;
      nir_intrinsic_set_image_dim(read, GLSL_SAMPLER_DIM_2D);
      nir_intrinsic_set_format(read, PIPE_FORMAT_R32_UINT);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[reads[i]];
      for (unsigned j = 0; j < info->num_srcs; ++j) {
         unsigned components = info->src_components[j] ? info->src_components[j] : read->num_components;
         read->src[j] = nir_src_for_ssa(nir_imm_zero(&b, components, j == 0 ? 64 : 32));
      }
      nir_def_init(&read->instr, &read->def, read->num_components, 32);
      nir_builder_instr_insert(&b, &read->instr);
      CHECK(!hk_apple9_native_store_shader(b.shader));
      ralloc_free(b.shader);
   }
   puts("native storage: eligibility, descriptor ABI and per-layer addresses pass");
   glsl_type_singleton_decref();
   return 0;
}
