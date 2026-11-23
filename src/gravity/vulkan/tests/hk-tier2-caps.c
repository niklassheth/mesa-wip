/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include "compiler/nir/nir_builder.h"
#include "gravity/compiler/agx_compile.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static int
test_ycbcr_2plane_444_and_arrays(void)
{
   struct hk_physical_device pdev = {0};
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev.dev.chip = AGX_CHIP_G16G;

   const VkFormat ycbcr_444_formats[] = {
      VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
      VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
   };

   const VkFormatFeatureFlags2 required_features =
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT |
      VK_FORMAT_FEATURE_2_DISJOINT_BIT |
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT |
      VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT;

   const VkFormatFeatureFlags2 forbidden_features =
      VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT |
      VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
      VK_FORMAT_FEATURE_2_BLIT_SRC_BIT |
      VK_FORMAT_FEATURE_2_BLIT_DST_BIT;

   for (unsigned i = 0; i < ARRAY_SIZE(ycbcr_444_formats); ++i) {
      VkFormat fmt = ycbcr_444_formats[i];

      for (unsigned linear = 0; linear < 2; ++linear) {
         VkImageTiling tiling =
            linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
         VkFormatFeatureFlags2 f =
            hk_get_image_format_features(&pdev, fmt, tiling);
         CHECK((f & required_features) == required_features);
         CHECK((f & forbidden_features) == 0);

         VkPhysicalDeviceImageFormatInfo2 info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
            .format = fmt,
            .type = VK_IMAGE_TYPE_2D,
            .tiling = tiling,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         };
         VkSamplerYcbcrConversionImageFormatProperties ycbcr_props = {
            .sType =
               VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
         };
         VkImageFormatProperties2 props = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
            .pNext = &ycbcr_props,
         };
         CHECK(hk_GetPhysicalDeviceImageFormatProperties2(
                  hk_physical_device_to_handle(&pdev), &info, &props) ==
               VK_SUCCESS);
         CHECK(ycbcr_props.combinedImageSamplerDescriptorCount == 2);
         CHECK(props.imageFormatProperties.maxMipLevels == 1);
         CHECK(props.imageFormatProperties.sampleCounts ==
               VK_SAMPLE_COUNT_1_BIT);
         CHECK(props.imageFormatProperties.maxArrayLayers ==
               (linear ? 1 : 2048));
      }

      /* YCbCr formats are 2D-only; 1D and 3D must be rejected. */
      const VkImageType bad_types[] = {VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_3D};
      for (unsigned t = 0; t < ARRAY_SIZE(bad_types); ++t) {
         VkPhysicalDeviceImageFormatInfo2 info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
            .format = fmt,
            .type = bad_types[t],
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
         };
         VkImageFormatProperties2 props = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
         };
         CHECK(hk_GetPhysicalDeviceImageFormatProperties2(
                  hk_physical_device_to_handle(&pdev), &info, &props) ==
               VK_ERROR_FORMAT_NOT_SUPPORTED);
      }
   }

   /* Verify standard multi-plane YCbCr formats also expose array layers. */
   const VkFormat ycbcr_array_formats[] = {
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
      VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
      VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(ycbcr_array_formats); ++i) {
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .format = ycbcr_array_formats[i],
         .type = VK_IMAGE_TYPE_2D,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      };
      CHECK(hk_GetPhysicalDeviceImageFormatProperties2(
               hk_physical_device_to_handle(&pdev), &info, &props) ==
            VK_SUCCESS);
      CHECK(props.imageFormatProperties.maxArrayLayers == 2048);
   }

   /* Verify VK_EXT_4444_formats remain exposed for sampling and rendering. */
   const VkFormat fmt_4444[] = {
      VK_FORMAT_A4R4G4B4_UNORM_PACK16,
      VK_FORMAT_A4B4G4R4_UNORM_PACK16,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fmt_4444); ++i) {
      VkFormatFeatureFlags2 f = hk_get_image_format_features(
         &pdev, fmt_4444[i], VK_IMAGE_TILING_OPTIMAL);
      CHECK(f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
      CHECK(f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
      CHECK(f & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
   }

   vk_object_base_finish(&pdev.vk.base);
   return 0;
}

static int
test_subgroup_rotate_compile(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "subgroup_rotate_test");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *off = nir_imul_imm(&b, lane, 8);
   nir_def *val = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), off,
                                .access = ACCESS_NON_WRITEABLE, .align_mul = 4);

   nir_def *rot_full = nir_rotate(&b, val, nir_imm_int(&b, 3), .cluster_size = 0);
   nir_def *rot_clust =
      nir_rotate(&b, val, nir_imm_int(&b, 1), .cluster_size = 8);

   nir_store_ssbo(&b, nir_vec2(&b, rot_full, rot_clust), nir_imm_int(&b, 0),
                  off, .write_mask = 0x3, .align_mul = 8);

   struct agx_shader_part part = {0};
   struct agx_apple9_compute_profile profile = {0};
   const char *reason = NULL;
   bool ok = agx_compile_apple9_tiny(b.shader, &part, &profile, &reason);
   if (!ok) {
      fprintf(stderr, "subgroup_rotate compile failed: %s\n",
              reason ? reason : "(unknown)");
   }
   CHECK(ok);
   CHECK(part.binary != NULL && part.info.main_size > 0);
   free(part.binary);
   ralloc_free(b.shader);
   return 0;
}

int
main(void)
{
   if (test_ycbcr_2plane_444_and_arrays())
      return 1;
   if (test_subgroup_rotate_compile())
      return 1;
   puts("PASS Tier 2 caps: YCbCr 2-plane 444, YCbCr arrays, 4444, subgroup rotate");
   return 0;
}
