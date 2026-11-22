/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_descriptor_memory.h"
#include "hk_private.h"

#include "hk_device.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/lib/agx_apple9.h"
#include "vk_descriptor_update_template.h"
#include "vk_object.h"

#include "util/list.h"
#include "util/vma.h"

struct hk_descriptor_set_layout;

/* GPU-readable format metadata for Apple9 uniform texel-buffer loads. The
 * ordinary texture prefix remains available for descriptor queries.
 * A component word contains byte offset [7:0], bit shift [12:8], width minus
 * one [17:13], and conversion kind [21:18]. Swizzles are resolved per component.
 */
enum hk_apple9_texel_kind {
   HK_TEXEL_ZERO, HK_TEXEL_ONE, HK_TEXEL_UINT, HK_TEXEL_SINT,
   HK_TEXEL_UNORM, HK_TEXEL_SNORM, HK_TEXEL_FLOAT16, HK_TEXEL_FLOAT32,
   HK_TEXEL_FLOAT11, HK_TEXEL_RGB9E5, HK_TEXEL_SRGB,
};

/* Texel-buffer metadata: base address, element count, and format layout.
 * Storage images leave base, elements, and residency zero.
 */
struct hk_apple9_texel_metadata {
   uint64_t base;
   uint32_t elements;
   uint32_t size_B;
   uint32_t component[4];
   /* Sparse residency buffers: the residency byte of the page holding
    * address A is at residency + A / AIL_PAGESIZE, see
    * hk_buffer::apple9_residency. Zero for other resources. */
   uint64_t residency;
};

/* Uniform texel buffers read every format but RGB32 through the texture
 * unit. Like hk_apple9_image_part, the descriptor holds the native
 * texture entry for shaders that address it directly. */
struct hk_apple9_texel_descriptor {
   struct agx_texture_packed tex;
   struct hk_apple9_texel_metadata metadata;
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_texel_descriptor) == 96,
              "Apple9 texel descriptor has no holes");
static_assert(offsetof(struct hk_apple9_texel_descriptor, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");

enum hk_apple9_storage_native_flags {
   HK_APPLE9_STORAGE_READ_VALID = 1u << 0,
   HK_APPLE9_STORAGE_WRITE_VALID = 1u << 1,
   /* Bit 30 marks sparse-resident storage images (for unrouted loads). */
   HK_APPLE9_STORAGE_SPARSE = 1u << 30,
};

/* Storage images pack enum hk_apple9_store_encode in native_ok bits 4..6. */
#define HK_APPLE9_STORAGE_ENCODE_SHIFT 4
#define HK_APPLE9_STORAGE_ENCODE_MASK  0x7u

/* Storage texel buffers pack their sub-16-byte element offset in native_ok
 * bits 3..29. */
#define HK_APPLE9_STORAGE_TAIL_SHIFT 3
#define HK_APPLE9_STORAGE_TAIL_MASK  ((1u << 27) - 1)

bool hk_apple9_native_store_format(enum pipe_format format);

/* Apple9 storage images and storage texel buffers, 128 bytes (matching the
 * descriptor-buffer ABI). Both native PBE (offset 64) and native ro_storage
 * texture (offset 96) live inline in the USC-aperture descriptor, eliminating
 * per-view USC side-table allocations and enabling capture-replay directly. */
struct hk_apple9_storage_descriptor {
   /* [0..23] 24B: software PBE for atomics, MSAA, and rare software fallbacks.
    * Placed at offset 0 so storage handles address the descriptor base. */
   struct agx_pbe_packed pbe_sw;
   /* [24..27] 4B: enum hk_apple9_storage_native_flags + store_encode / tail. */
   uint32_t native_ok;
   /* [28..31] 4B: texel block size in bytes (1..16). */
   uint32_t size_B;
   /* [32..47] 16B:
    *   - Storage images: extent[0..3] = {width, height, depth/layers, samples}.
    *   - Storage texel buffers: elements at [32..35] (= extent[0]), RW base at [40..47]. */
   union {
      uint32_t extent[4];
      struct {
         uint32_t elements;
         uint32_t pad;
         uint64_t base;
      } buf;
   };
   /* [48..63] 16B: per-component formatless texel layout. */
   uint32_t component[4];
   /* [64..95] 32B (32-byte aligned): inline native Apple9 PBE descriptor
    * (2D or 2D-array for images; 16384-wide linear 2D for texel buffers). */
   uint8_t pbe[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   /* [96..127] 32B (32-byte aligned):
    *   - Storage images: inline native Apple9 ro_storage texture descriptor.
    *   - Storage texel buffers: sparse buffer residency table VA at [96..103]. */
   union {
      uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
      struct {
         uint64_t residency;
         uint8_t pad[24];
      } buf_sparse;
   };
};
static_assert(sizeof(struct hk_apple9_storage_descriptor) == 128,
              "Apple9 storage descriptor retains the 128-byte ABI");
static_assert(offsetof(struct hk_apple9_storage_descriptor, pbe_sw) == 0,
              "software PBE sits at the descriptor base");
static_assert(offsetof(struct hk_apple9_storage_descriptor, pbe) == 64,
              "inline native PBE is 32-byte aligned at slot 2");
static_assert(offsetof(struct hk_apple9_storage_descriptor, texture) ==
                 (AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY + 1) *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native storage entry is one table slot past the image one");
static_assert(offsetof(struct hk_apple9_storage_descriptor, extent[0]) ==
                 offsetof(struct hk_apple9_storage_descriptor, buf.elements),
              "buffer elements alias extent[0] for unified imageSize queries");

void hk_apple9_texel_layout(enum pipe_format format,
                            struct hk_apple9_texel_metadata *metadata);

/* Encodings for stores to storage images without a shader format. A
 * canonical format packs its N channels in order, each W bits wide from bit
 * 0, so stores can encode it with constant shifts instead of decoding the
 * per-component layout. Everything else uses the layout (GENERIC).
 */
enum hk_apple9_store_encode {
   HK_STORE_ENCODE_GENERIC = 0,
   /* 32-bit channels: the texel's bits as they are (float, uint, sint). */
   HK_STORE_ENCODE_RAW32,
   /* 8-bit unsigned normalized channels. */
   HK_STORE_ENCODE_UNORM8,
   /* 16-bit float channels. */
   HK_STORE_ENCODE_FLOAT16,
   /* R11G11B10 unsigned float (B10G11R11_UFLOAT_PACK32). */
   HK_STORE_ENCODE_FLOAT11,
};

enum hk_apple9_store_encode hk_apple9_store_encode(enum pipe_format format);

/* G16 descriptor records, 128 bytes each so translation layers can copy
 * them with their 128-byte paths: an image part and a sampler part. A
 * combined image sampler is the image part followed by the sampler part.
 * Shaders address each part through its own handle; the native entries keep
 * the table slots AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY and
 * AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY past their part's base, as the texel
 * and storage descriptors do.
 */
struct hk_apple9_image_part {
   /* Same layout as the per-draw texture info record. */
   uint32_t texture_info[AGX_APPLE9_TEXTURE_INFO_STRIDE / 4];
   uint32_t pad_res[3];
   uint16_t min_lod_fp16, min_lod_uint16;
   uint8_t pad0[16];
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   uint8_t pad1[32];
};
static_assert(sizeof(struct hk_apple9_image_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_image_part, texture_info) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO,
              "texture info where shaders read it");
static_assert(offsetof(struct hk_apple9_image_part, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");

struct hk_apple9_sampler_part {
   /* Same layout as the per-draw runtime sampler record. */
   uint32_t runtime_sampler[AGX_APPLE9_RUNTIME_SAMPLER_STRIDE / 4];
   uint16_t lod_bias_fp16, txl_nudge_fp16;
   uint8_t pad0[28];
   uint8_t sampler[AGX_APPLE9_SAMPLER_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_sampler_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_sampler_part, runtime_sampler) ==
                 AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER,
              "runtime sampler where shaders read it");
static_assert(offsetof(struct hk_apple9_sampler_part, sampler) ==
                 AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY *
                    AGX_APPLE9_SAMPLER_TABLE_STRIDE,
              "native sampler entry is where register indices expect it");

/* A combined image sampler is an image part followed by a sampler part; a
 * multi-plane descriptor repeats the whole record per plane. */
#define HK_APPLE9_COMBINED_DESCRIPTOR_SIZE                                    \
   (sizeof(struct hk_apple9_image_part) + sizeof(struct hk_apple9_sampler_part))
#define HK_MAX_SAMPLED_DESCRIPTOR_SIZE (3 * HK_APPLE9_COMBINED_DESCRIPTOR_SIZE)

void hk_apple9_texture_info(uint32_t *data, const struct agx_texture_packed *state);
bool hk_apple9_runtime_sampler(uint32_t *data, const struct agx_sampler_packed *state,
                               bool clamp_to_edge);
/* A custom border sampler takes its packed colour from border. */
bool hk_apple9_native_sampler(void *out, const struct agx_sampler_packed *state,
                              bool clamp_to_edge, const uint32_t *border);

/* This has to match nir_address_format_64bit_bounded_global */
struct hk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

struct hk_device;
struct hk_descriptor_set_layout;
void hk_descriptor_pack_embedded_samplers(
   struct hk_device *dev, const struct hk_descriptor_set_layout *layout,
   void *map);

struct hk_descriptor_pool {
   struct vk_object_base base;

   struct list_head sets;

   struct hk_descriptor_memory mem;

   /* VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT: never read by the GPU. */
   bool host_only;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

struct hk_descriptor_set {
   struct vk_object_base base;

   /* Link in hk_descriptor_pool::sets */
   struct list_head link;

   struct hk_descriptor_set_layout *layout;
   void *mapped_ptr;
   uint64_t addr;
   uint32_t size;
   struct hk_descriptor_allocation alloc;

   struct hk_buffer_address dynamic_buffers[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

static inline uint64_t
hk_descriptor_set_addr(const struct hk_descriptor_set *set)
{
   return set->addr;
}

struct hk_push_descriptor_set {
   uint8_t data[HK_PUSH_DESCRIPTOR_SET_SIZE];
   /* Byte extent of the current layout, including padding between bindings. */
   uint32_t size;
};

void hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                                   struct hk_descriptor_set_layout *layout,
                                   uint32_t write_count,
                                   const VkWriteDescriptorSet *writes);

void hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data);
