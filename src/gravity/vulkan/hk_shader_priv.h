/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_shader.h"
#include "hk_apple9_polygon_points.h"
#include "nir_builder.h"
#include "util/blob.h"
#include "util/mesa-blake3.h"
#include "util/simple_mtx.h"
#include "vk_graphics_state.h"
#include "vk_pipeline_cache.h"

struct hk_device;
struct hk_physical_device;
struct vk_descriptor_set_layout;
struct vk_features;
struct vk_pipeline_robustness_state;
struct vk_shader_compile_info;

enum hk_feature_key {
   HK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   /* Apple9 lowers the image view min LOD clamp only when it is enabled. */
   HK_FEAT_MIN_LOD = BITFIELD_BIT(1),
   /* Unrouted storage image loads check hardware sparse residency only when
    * sparse-resident images can exist on the device. */
   HK_FEAT_SPARSE_IMAGE = BITFIELD_BIT(2),
   /* Sparse residency buffers may back storage texel buffers. */
   HK_FEAT_STORE_BUFFER_RESIDENCY = BITFIELD_BIT(3),
   /* Pipeline statistics queries instrument fragment shaders. */
   HK_FEAT_PIPELINE_STATS = BITFIELD_BIT(4),
   /* Primitives-generated queries require SW VS/TES variants for passthrough GS. */
   HK_FEAT_PRIMS_GENERATED = BITFIELD_BIT(5),
};


#define HK_APPLE9_BODY_CACHE_SIZE 4096
/* Match the old count limit for minimum-size M4 BOs, but do not retain an
 * unbounded number of bytes when shader bodies span several pages. */
#define HK_APPLE9_BODY_CACHE_BYTES (64ull * 1024 * 1024)

#define HK_APPLE9_BODY_SLAB_SIZE (64 * 1024)
#define HK_APPLE9_BODY_ALIGNMENT 64
/* USC may prefetch beyond the last instruction. Keep the end of every slab
 * mapped but unallocated so its last body cannot prefetch into the VA guard. */
#define HK_APPLE9_BODY_TAIL 1024
#define HK_APPLE9_BODY_SLAB_USABLE \
   (HK_APPLE9_BODY_SLAB_SIZE - HK_APPLE9_BODY_TAIL)

struct hk_apple9_body_slab {
   struct agx_bo *bo;
   struct util_vma_heap heap;
   struct list_head link;
   unsigned live, cached;
};

struct hk_apple9_body {
   blake3_hash hash;
   uint64_t size, alloc_size, cache_charge;
   struct hk_apple9_body_slab *slab;
   struct agx_shader_body base;
   struct list_head link;
};

extern const struct vk_shader_ops hk_shader_ops;

/* hk_shader.c */
enum hk_feature_key hk_make_feature_key(const struct vk_features *feats);
void hk_hash_apple9_native_mesh_abi(struct mesa_blake3 *ctx);
void hk_shader_destroy(struct hk_device *dev, struct hk_shader *s);

/* hk_shader_nir.c */
bool hk_apple9_lower_clip_z_coeff(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data);
void hk_apple9_remap_fs_outputs(nir_shader *nir, const uint8_t map[8]);
bool hk_apple9_lower_bary_to_sample(nir_builder *b, nir_intrinsic_instr *intr,
                                    UNUSED void *data);
bool hk_apple9_lower_api_vertex_order(nir_shader *nir);
bool hk_is_barycentric_coord(nir_intrinsic_op op);
bool hk_apple9_lower_per_vertex_copies(nir_shader *fs, uint64_t producer_outputs,
                                       struct hk_per_vertex_copies *out);
bool hk_apple9_interpolates_off_center(nir_shader *nir);
void hk_apple9_write_vertex_weights(nir_shader *vs,
                                    const struct hk_per_vertex_copies *copies);
void hk_apple9_write_line_stipple_start(nir_shader *vs,
                                        struct hk_apple9_stipple_slots slots);
void hk_apple9_write_gs_line_stipple(nir_shader *gs,
                                     struct hk_apple9_stipple_slots slots);
void hk_apple9_lower_line_stipple_fs(nir_shader *fs,
                                     struct hk_apple9_stipple_slots slots);
bool hk_apple9_lower_polygon_slot(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data);
void hk_apple9_write_polygon_slot(nir_shader *s, unsigned slot);
void hk_apple9_write_primitive_id(nir_shader *s);
bool hk_apple9_polygon_point_budget(const nir_shader *last,
                                    const nir_shader *fs,
                                    uint64_t reserved, unsigned *face_slot);
void hk_lower_nir(struct hk_device *dev, nir_shader *nir,
                  const struct vk_pipeline_robustness_state *rs,
                  bool is_multiview, uint32_t set_layout_count,
                  struct vk_descriptor_set_layout *const *set_layouts,
                  enum hk_feature_key features);
#ifndef NDEBUG
void hk_assert_final_push_mask(const nir_shader *nir,
                               const struct hk_shader *shader);
#else
#define hk_assert_final_push_mask(nir, shader) ((void)0)
#endif
bool hk_apple9_texture_bindings(nir_shader *nir);
void hk_apple9_late_zs(nir_shader *nir);
bool hk_apple9_clamp_depth_export(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data);
bool hk_apple9_lower_roots(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data);
void hk_apple9_strip_unread_outputs(nir_shader *nir);
void hk_nir_opts(nir_shader *nir, void *data);
void hk_apple9_lower_fs_statistics(nir_shader *nir);
void hk_apple9_lower_prerast_sysvals(nir_shader *nir,
                                     mesa_shader_stage sw_stage);
void hk_apple9_lower_primitive_fs(nir_shader *nir, bool native_mesh);

/* hk_shader_compile.c */
bool hk_apple9_body_alloc(struct hk_device *dev, struct hk_apple9_body *body);
struct agx_shader_body *hk_apple9_body_upload(struct hk_device *dev,
                                              const struct hk_shader *shader);
VkResult hk_upload_shader(struct hk_device *dev, struct hk_shader *shader);
VkResult hk_apple9_native_gs_check(const nir_shader *source,
                                   const struct hk_apple9_graphics_key *key,
                                   bool *supported, const char **reason);
VkResult hk_apple9_native_tess_check(const nir_shader *source,
                                     const struct hk_apple9_graphics_key *key,
                                     bool *supported, const char **reason);
VkResult hk_apple9_compile_graphics_path(
   struct hk_device *dev, uint32_t count, struct vk_shader_compile_info *infos,
   const struct vk_graphics_pipeline_state *state,
   const struct hk_apple9_graphics_key *key_in, const uint64_t *fs_interp,
   const VkAllocationCallbacks *alloc, struct vk_shader **out,
   bool native_mesh, bool *native_rejected);
bool hk_apple9_vertex_unbounded(const struct vk_shader_compile_info *info);
VkResult hk_apple9_compile_graphics_mode(
   struct hk_device *dev, uint32_t count, struct vk_shader_compile_info *infos,
   const struct vk_graphics_pipeline_state *state,
   const struct hk_apple9_graphics_key *key_in, const uint64_t *fs_interp,
   const VkAllocationCallbacks *alloc, struct vk_shader **out,
   bool native_only);
VkResult hk_apple9_compile_graphics(
   struct hk_device *dev, uint32_t count, struct vk_shader_compile_info *infos,
   const struct vk_graphics_pipeline_state *state,
   const struct hk_apple9_graphics_key *key_in, const uint64_t *fs_interp,
   const VkAllocationCallbacks *alloc, struct vk_shader **out);
nir_shader *hk_apple9_null_fs_nir(bool sample_mask);
void hk_apple9_graphics_key(struct hk_apple9_graphics_key *key,
                            const struct vk_graphics_pipeline_state *state);
void hk_apple9_graphics_key_device(
   struct hk_apple9_graphics_key *key,
   const struct vk_graphics_pipeline_state *state,
   const struct hk_physical_device *pdev);
