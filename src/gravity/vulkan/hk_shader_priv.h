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
   /* Storage texel buffer descriptors may lack a native entry on devices with
    * both descriptorBuffer and sparseResidencyBuffer (vkGetDescriptorEXT
    * cannot attach a buffer residency table). */
   HK_FEAT_CHECKED_BUF_STORES = BITFIELD_BIT(4),
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



#define HK_APPLE9_CACHED_VARIANTS_PER_SOURCE 64

struct hk_apple9_cached_variant {
   struct hk_apple9_cached_variant *next;
   struct vk_pipeline_cache_object *object;
};

/* Serialized NIR after API lowering, before key-dependent compilation.
 * Precompile jobs hold references, so a shader can be destroyed while its
 * variants are still queued. */
struct hk_apple9_nir {
   uint32_t refs;
   uint32_t size;
   alignas(8) uint8_t data[];
};


struct hk_apple9_deferred {
   struct hk_apple9_nir *nir;
   uint32_t next_stage_mask;
   uint32_t set_layout_count;
   /* Static pipeline sample shading, which has no dynamic state. */
   bool sample_shading;
   /* Static depth/stencil feedback loop. The dynamic form reaches draw-time
    * variants through the dynamic feedback-loop state instead. */
   bool zs_self_dep;
   /* Vertex shader whose pipeline disables vertex input robustness. */
   bool vertex_unbounded;
   /* Color attachments the pipeline leaves undefined; writes are discarded
    * even when the render binds an image there. */
   uint8_t undefined_colors;
   /* A missing fragment stage needs a generated cull-distance consumer. */
   bool cull_distance;
   /* Hardware VS eligible for a linked vertex prolog (no XFB/cull/GS/TCS). */
   bool vs_can_prolog;
   /* Fragment shader uses per-vertex or barycentric inputs needing SW VS. */
   bool fs_uses_per_vertex;
   /* Fragment shader eligible for a linked fragment epilog. */
   bool fs_can_epilog;
   /* Fragment shader reads sample-rate intrinsics (depends on sample count). */
   bool fs_uses_sample_rate;
   /* Fragment shader reads gl_ViewportIndex or gl_PrimitiveID. */
   bool fs_reads_viewport;
   bool fs_reads_prim_id;
   /* Vertex shader compiled without static primitive topology (GPL/ESO/dynamic). */
   bool vs_dynamic_topology;
   /* VkPipelineCache serializes these API shaders. Retain the variants seen
    * with this source so application-managed caches include draw-time work.
    * The list owns only serialized bytes, never other API shader objects. */
   simple_mtx_t variant_lock;
   struct hk_apple9_cached_variant *variants;
   unsigned variant_count;
};

static inline struct hk_apple9_deferred *
hk_apple9_deferred_source(const struct hk_api_shader *api)
{
   if (!api)
      return NULL;
   return api->apple9_deferred ? api->apple9_deferred : api->apple9_fallback;
}


/* Why an Apple9 graphics compile waits for draw-time state. */
enum hk_apple9_deferral {
   HK_APPLE9_COMPILE_NOW,
   /* State that keys the compile is dynamic, belongs to another pipeline
    * library, or is absent. */
   HK_APPLE9_DEFER_STATE,
   /* The pipeline's state is complete, but the compile still waits for the
    * draw: the fragment stage is linked then (the null one for a depth-only
    * pipeline, or one without blend state), or a render with unused
    * attachments can lay its tiles out differently
    * (hk_apple9_color_layout_varies). Its own variant is resolved at
    * creation. */
   HK_APPLE9_DEFER_COMPLETE,
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

/* hk_shader_variant.c */
void hk_apple9_graphics_key(struct hk_apple9_graphics_key *key,
                            const struct vk_graphics_pipeline_state *state);
bool hk_apple9_partial_stages(VkShaderStageFlags stages);
bool hk_apple9_color_layout_varies(
   const struct vk_graphics_pipeline_state *state,
   const struct vk_features *features);
bool hk_apple9_fs_needs_epilog(const struct vk_graphics_pipeline_state *state,
                               const struct vk_features *features);
void hk_apple9_graphics_key_device(
   struct hk_apple9_graphics_key *key,
   const struct vk_graphics_pipeline_state *state,
   const struct hk_physical_device *pdev);
uint32_t hk_apple9_deferral_hash_bits(
   const struct vk_graphics_pipeline_state *state,
   const struct vk_features *features, VkShaderStageFlags stages);
struct hk_apple9_nir *hk_apple9_nir_create(const void *data, size_t size);
struct hk_apple9_nir *hk_apple9_nir_ref(struct hk_apple9_nir *nir);
void hk_apple9_nir_unref(struct hk_apple9_nir *nir);
void hk_apple9_remember_variant(struct hk_device *dev,
                                struct hk_api_shader *api,
                                struct vk_pipeline_cache_object *object);
bool hk_apple9_serialize_variants(struct hk_apple9_deferred *d,
                                  struct blob *blob);
bool hk_apple9_deserialize_variants(struct hk_device *dev,
                                    struct hk_api_shader *api,
                                    struct blob_reader *blob);
void hk_apple9_destroy_deferred(struct hk_device *dev,
                                struct hk_apple9_deferred *d);
bool hk_apple9_fs_can_epilog_nir(nir_shader *fs);
enum hk_apple9_deferral hk_apple9_needs_deferral(
   uint32_t count, const struct vk_shader_compile_info *infos,
   const struct vk_graphics_pipeline_state *state,
   const struct vk_features *features);
void hk_apple9_eager_variant(struct hk_device *dev, uint32_t count,
                             struct vk_shader *const *shaders,
                             const struct vk_graphics_pipeline_state *state,
                             bool precompile);
bool hk_apple9_eager_variants_disabled(void);
struct hk_apple9_deferred *hk_apple9_build_deferred(
   const struct vk_shader_compile_info *info,
   const struct vk_graphics_pipeline_state *state);
VkResult hk_apple9_defer_graphics(struct hk_device *dev, uint32_t count,
                                  struct vk_shader_compile_info *infos,
                                  const struct vk_graphics_pipeline_state *state,
                                  const VkAllocationCallbacks *alloc,
                                  struct vk_shader **out);
bool hk_apple9_variant_cache_key(struct hk_device *dev,
                                 struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
                                 const struct hk_apple9_graphics_key *key,
                                 bool prefer_native, blake3_hash out);
void hk_apple9_variant_cache_store(struct hk_device *dev,
                                   const unsigned *stages, unsigned count,
                                   struct vk_shader *const *compiled,
                                   struct blob *blob);
unsigned hk_apple9_variant_cache_load(struct hk_device *dev, const void *data,
                                      size_t size, uint32_t expected_stages,
                                      unsigned *stages,
                                      struct vk_shader **out);
void hk_apple9_forget_deferred(struct hk_device *dev,
                               struct hk_api_shader *obj);
