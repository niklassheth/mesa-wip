/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "gravity/layout/layout.h"
#include "util/list.h"
#include "vulkan/vulkan_core.h"

#include "hk_private.h"

#include "vk_format.h"
#include "vk_image.h"

/* Because small images can end up with an array_stride_B that is less than
 * the sparse block size (in bytes), we have to set SINGLE_MIPTAIL_BIT when
 * advertising sparse properties to the client.  This means that we get one
 * single memory range for the miptail of the image.  For large images with
 * mipTailStartLod > 0, we have to deal with the array stride ourselves.
 *
 * We do this by returning HK_MIP_TAIL_START_OFFSET as the image's
 * imageMipTailOffset.  We can then detect anything with that address as
 * being part of the miptail and re-map it accordingly.  The Vulkan spec
 * explicitly allows for this.
 *
 * From the Vulkan 1.3.279 spec:
 *
 *    "When VK_SPARSE_MEMORY_BIND_METADATA_BIT is present, the resourceOffset
 *    must have been derived explicitly from the imageMipTailOffset in the
 *    sparse resource properties returned for the metadata aspect. By
 *    manipulating the value returned for imageMipTailOffset, the
 *    resourceOffset does not have to correlate directly to a device virtual
 *    address offset, and may instead be whatever value makes it easiest for
 *    the implementation to derive the correct device virtual address."
 */
#define HK_MIP_TAIL_START_OFFSET 0x6d74000000000000UL

struct hk_device_memory;
struct hk_physical_device;

bool hk_apple9_storage_format_supported(VkFormat format);

static VkFormatFeatureFlags2
hk_get_image_plane_format_features(struct hk_physical_device *pdev,
                                   VkFormat vk_format, VkImageTiling tiling);

VkFormatFeatureFlags2
hk_get_image_format_features(struct hk_physical_device *pdevice,
                             VkFormat format, VkImageTiling tiling);

struct hk_image_plane {
   struct ail_layout layout;
   uint64_t addr;
   struct agx_va *va;

   /* Host image copy. On 32-bit, binding records the backing storage but
    * leaves CPU VA available until an individual copy needs a mapping.
    * The BO is owned by the bound VkDeviceMemory.
    */
   struct agx_bo *bo;
   uint64_t offset_B;
   void *map;

   /* With an explicit DRM format modifier layout, the plane sits at this
    * offset within its memory binding. Otherwise planes are consecutive.
    */
   bool has_explicit_offset;
   uint64_t explicit_offset_B;

   /* If the image has sparse residency, its residency is tracked in this
    * secondary page table. Otherwise, this map is NULL.
    */
   struct agx_bo *sparse_map;

   /* Tracking for hk_device_memory VA-alias detection. */
   uint64_t va_bind_bo_key;
   uint64_t va_bind_id;
};

struct hk_image {
   struct vk_image vk;

   /** True if the planes are bound separately
    *
    * This is set based on VK_IMAGE_CREATE_DISJOINT_BIT
    */
   bool disjoint;

   uint8_t plane_count;
   struct hk_image_plane planes[3];

   /* Sticky recording-time hint set atomically once repeated feedback
    * decompressions execute on this image, so later normal render passes also
    * store raw (with a submit-time-elidable decompression) rather than
    * re-compressing and thrashing. Correctness never relies on this alone.
    */
   uint32_t feedback_prefer_raw;

   /* True when cross-submit raw-state tracking is safe (internal, optimal,
    * non-external, non-swapchain, non-sparse compressed image).
    */
   bool feedback_track_cross_submit;

   /* Submit-time execution-order raw subresource tracking, protected by the
    * feedback tracker mutex in hk_feedback.c.
    */
   uint32_t feedback_gpu_decompressions;
   uint64_t feedback_submitted_raw_mask;
   struct list_head feedback_raw_link;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

/* Opaque capture data of an image: the addresses of each sparse plane and of
 * its residency table, which descriptors embed. Zero where not replayable. */
struct hk_image_plane_capture_data {
   uint64_t va, residency;
};

struct hk_image_capture_data {
   struct hk_image_plane_capture_data planes[3];
};

/* Image views have no capture data of their own. */
#define HK_IMAGE_VIEW_CAPTURE_DATA_SIZE 4

static inline uint64_t
hk_image_plane_base_address(const struct hk_image_plane *plane)
{
   return plane->addr;
}

static inline uint64_t
hk_image_base_address(const struct hk_image *image, uint8_t plane)
{
   return hk_image_plane_base_address(&image->planes[plane]);
}

static inline enum pipe_format
hk_format_to_pipe_format(VkFormat vkformat)
{
   switch (vkformat) {
   case VK_FORMAT_R10X6_UNORM_PACK16:
   case VK_FORMAT_R12X4_UNORM_PACK16:
      return PIPE_FORMAT_R16_UNORM;
   case VK_FORMAT_R10X6G10X6_UNORM_2PACK16:
   case VK_FORMAT_R12X4G12X4_UNORM_2PACK16:
      return PIPE_FORMAT_R16G16_UNORM;
   default:
      return vk_format_to_pipe_format(vkformat);
   }
}

static inline uint8_t
hk_image_aspects_to_plane(const struct hk_image *image,
                          VkImageAspectFlags aspectMask)
{
   /* Must only be one aspect unless it's depth/stencil */
   assert(aspectMask ==
             (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ||
          util_bitcount(aspectMask) == 1);

   switch (aspectMask) {
   default:
      assert(aspectMask != VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT);
      return 0;

   case VK_IMAGE_ASPECT_STENCIL_BIT:
      return image->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT;

   case VK_IMAGE_ASPECT_PLANE_1_BIT:
   case VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT:
      return 1;

   case VK_IMAGE_ASPECT_PLANE_2_BIT:
   case VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT:
      return 2;
   }
}

struct agx_device;
bool hk_can_compress_format(const struct agx_device *dev, VkFormat format);

struct hk_cmd_buffer;
void hk_clear_image(struct hk_cmd_buffer *cmd, struct hk_image *image,
                    enum pipe_format view_format, const uint32_t *clear_value,
                    const VkImageSubresourceRange *range, bool whole_3d);
