/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <string.h>
#include "compiler/nir/nir_builder.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_buffer.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "gravity/compiler/agx_nir_texture.h"
#include "gravity/layout/layout.h"
#include "gravity/lib/agx_helpers.h"
#include "vk_alloc.h"
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

static int
test_storage_native_descriptors(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;

   uint8_t null_pbe[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   uint8_t null_tex[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   agx_apple9_set_null_pbe(null_pbe);
   agx_apple9_set_null_texture(null_tex);

   /* Normal 2D and 2D-MS storage views BOTH pack non-null inline native PBE
    * and native texture descriptors (2D-MS packs a 16384-wide linear 2D PBE). */
   for (unsigned samples = 1; samples <= 4; samples *= 4) {
      const VkImageCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
         .extent = {64, 64, 1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = (VkSampleCountFlagBits)samples,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      struct hk_image image = {0};
      vk_image_init(&dev->vk, &image.vk, &ci);
      image.plane_count = 1;
      image.planes[0].addr = UINT64_C(0x2000000000);
      image.planes[0].layout = (struct ail_layout){
         .width_px = 64, .height_px = 64, .depth_px = 1, .levels = 1,
         .sample_count_sa = samples, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .tiling = AIL_TILING_GPU, .writeable_image = true,
      };
      ail_make_miptree(&image.planes[0].layout);
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = hk_image_to_handle(&image), .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      VkImageView handle;
      CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &handle) == VK_SUCCESS);
      struct hk_image_view *v = hk_image_view_from_handle(handle);
      CHECK(v->planes[0].alias_width_el == 0);
      CHECK(memcmp(v->apple9_storage.texture, null_tex, sizeof(null_tex)) != 0);
      CHECK(memcmp(v->apple9_storage.pbe, null_pbe, sizeof(null_pbe)) != 0);
      CHECK((v->apple9_storage.pbe[0] & 0xf) == AGX_TEXTURE_DIMENSION_2D);
      uint64_t pbe_addr;
      memcpy(&pbe_addr, v->apple9_storage.pbe + 8, sizeof(pbe_addr));
      CHECK((pbe_addr & ((UINT64_C(1) << 40) - 1)) == image.planes[0].addr >> 4);
      CHECK(v->apple9_storage.extent[0] == 64 && v->apple9_storage.extent[1] == 64 &&
            v->apple9_storage.extent[3] == samples);
      hk_DestroyImageView(hk_device_to_handle(dev), handle, NULL);
      vk_image_finish(&image.vk);
   }

   /* Padded 2D block-texel aliases (large miptree AND POT mip tail) pack both
    * native texture and native PBE descriptors with physical block extents,
    * while extent[0..1] retains logical block counts for shader OOB clipping. */
   const VkImageCreateInfo bc_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
               VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
               VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
      .extent = {768, 512, 1}, .mipLevels = 10, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   struct hk_image bc_img = {0};
   vk_image_init(&dev->vk, &bc_img.vk, &bc_ci);
   bc_img.plane_count = 1;
   bc_img.planes[0].addr = UINT64_C(0x2000000000);
   bc_img.planes[0].layout = (struct ail_layout){
      .width_px = 768, .height_px = 512, .depth_px = 1, .levels = 10,
      .sample_count_sa = 1, .format = PIPE_FORMAT_ASTC_8x8,
      .tiling = AIL_TILING_GPU, .writeable_image = true,
   };
   ail_make_miptree(&bc_img.planes[0].layout);
   unsigned padded_levels = 0;
   for (unsigned l = 0; l < bc_img.planes[0].layout.levels; ++l) {
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = hk_image_to_handle(&bc_img), .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_R32G32B32A32_UINT,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l, 1, 0, 1},
      };
      VkImageView handle;
      CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &handle) == VK_SUCCESS);
      struct hk_image_view *v = hk_image_view_from_handle(handle);
      if (v->planes[0].alias_width_el != 0)
         ++padded_levels;
      CHECK(memcmp(v->apple9_storage.texture, null_tex, sizeof(null_tex)) != 0);
      CHECK(memcmp(v->apple9_storage.pbe, null_pbe, sizeof(null_pbe)) != 0);
      CHECK(v->apple9_storage.extent[0] ==
            util_format_get_nblocksx(PIPE_FORMAT_ASTC_8x8, u_minify(768, l)));
      CHECK(v->apple9_storage.extent[1] ==
            util_format_get_nblocksy(PIPE_FORMAT_ASTC_8x8, u_minify(512, l)));
      CHECK(!(v->apple9_storage.native_ok & HK_APPLE9_STORAGE_LINEAR_2D_INDEX));
      hk_DestroyImageView(hk_device_to_handle(dev), handle, NULL);
   }
   CHECK(padded_levels > 0);
   vk_image_finish(&bc_img.vk);

   /* Max-length R8_UINT texel buffer at byte offsets 0, 1, and 15 must fit
    * within 16384 rows and pack non-null native texture + PBE descriptors. */
   const uint32_t max_el = AGX_TEXTURE_BUFFER_MAX_SIZE - 16;
   const uint64_t tbuf_offsets[] = {0, 1, 15};
   for (unsigned i = 0; i < ARRAY_SIZE(tbuf_offsets); ++i) {
      struct agx_texture_packed tex;
      struct agx_pbe_packed pbe;
      struct hk_apple9_texel_metadata meta = {0};
      uint8_t a9_tex[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      uint8_t a9_pbe[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      uint32_t a9_store = 0;
      uint64_t addr = UINT64_C(0x2000000000) + tbuf_offsets[i];
      hk_pack_texel_buffer(dev, PIPE_FORMAT_R8_UINT, addr, addr, max_el, 0,
                           &tex, &pbe, &meta, a9_tex, a9_pbe, &a9_store);
      CHECK(memcmp(a9_tex, null_tex, sizeof(null_tex)) != 0);
      CHECK(memcmp(a9_pbe, null_pbe, sizeof(null_pbe)) != 0);
      CHECK(((a9_store >> HK_APPLE9_STORAGE_TAIL_SHIFT) &
             HK_APPLE9_STORAGE_TAIL_MASK) == tbuf_offsets[i]);
   }

   /* RGB32 uniform texel buffer at non-multiple-of-12 sub-16B offset (24) must
    * pack cleanly without tripping texel-tail assertions. */
   {
      struct agx_texture_packed tex;
      struct agx_pbe_packed pbe;
      struct hk_apple9_texel_metadata meta = {0};
      uint8_t a9_tex[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      uint8_t a9_pbe[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      uint32_t a9_store = 0;
      uint64_t addr = UINT64_C(0x2000000000) + 24;
      hk_pack_texel_buffer(dev, PIPE_FORMAT_R32G32B32_FLOAT, addr, addr, 16, 0,
                           &tex, &pbe, &meta, a9_tex, a9_pbe, &a9_store);
      CHECK(meta.base == addr && meta.elements == 16 && meta.size_B == 12);
      CHECK(memcmp(a9_tex, null_tex, sizeof(null_tex)) == 0);
      CHECK(memcmp(a9_pbe, null_pbe, sizeof(null_pbe)) == 0);
   }

   /* Sparse buffer residency lookup honors logical vk.size and range bounds. */
   {
      simple_mtx_init(&dev->sparse_buffers.lock, mtx_plain);
      list_inithead(&dev->sparse_buffers.list);
      struct agx_va buf_va = {.addr = UINT64_C(0x3000000000), .size_B = 65536};
      struct agx_va res_va = {.addr = UINT64_C(0x4000000000), .size_B = 16384};
      struct agx_bo res_bo = {.va = &res_va};
      struct hk_buffer sbuf = {
         .vk.size = 16384,
         .va = &buf_va,
         .apple9_residency = &res_bo,
      };
      list_addtail(&sbuf.sparse_link, &dev->sparse_buffers.list);
      p_atomic_set(&dev->sparse_buffers.count, 1);
      uint64_t expected = hk_buffer_apple9_residency(&sbuf);
      CHECK(hk_device_lookup_sparse_buffer_residency(dev, buf_va.addr, 16384) == expected);
      CHECK(hk_device_lookup_sparse_buffer_residency(dev, buf_va.addr + 1024, 15360) == expected);
      CHECK(hk_device_lookup_sparse_buffer_residency(dev, buf_va.addr + 16384, 4) == 0);
      CHECK(hk_device_lookup_sparse_buffer_residency(dev, buf_va.addr + 1024, 16384) == 0);
      list_del(&sbuf.sparse_link);
      p_atomic_set(&dev->sparse_buffers.count, 0);
      simple_mtx_destroy(&dev->sparse_buffers.lock);
   }

   free(dev);
   puts("native storage: 2D, 2D-MS, padded/POT-tail, max tbuf, RGB32, and sparse range checks pass");
   return 0;
}

int main(void)
{
   CHECK(test_image_storage() == 0);
   CHECK(test_storage_native_descriptors() == 0);
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
   CHECK(offsetof(struct hk_apple9_storage_descriptor, buf_residency) == 48);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, alias_3d) == 48);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, pbe) == 64);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, texture) == 96);
   CHECK(HK_APPLE9_DB_STORAGE_SIZE >= sizeof(struct hk_apple9_storage_descriptor));
   CHECK(util_is_power_of_two_nonzero(HK_APPLE9_DB_STORAGE_SIZE));
   CHECK(!((HK_APPLE9_STORAGE_TAIL_MASK << HK_APPLE9_STORAGE_TAIL_SHIFT) &
           (HK_APPLE9_STORAGE_LINEAR_2D_INDEX | HK_APPLE9_STORAGE_SPARSE)));

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
   puts("native storage: eligibility, descriptor ABI and per-layer addresses pass");
   glsl_type_singleton_decref();
   return 0;
}
