/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "gravity/compiler/agx_apple9_mesh.h"
#include "gravity/lib/agx_apple9.h"
#include "gravity/lib/agx_apple9_launch.h"
#include "gravity/lib/agx_apple9_mesh.h"
#include "util/u_math.h"

/* Explicit userspace/kernel development pairing. The stable UAPI has no
 * native-mesh capability bit; enabling this requires the matching kernel. */
bool hk_apple9_native_mesh_enabled(void);
struct hk_physical_device;
struct hk_device;
/* Default publication policy is separate from the broader research opt-in. */
bool hk_apple9_native_index_enabled(const struct hk_physical_device *pdev);
bool hk_apple9_native_mesh_available(const struct hk_physical_device *pdev);
bool hk_apple9_native_mesh_available_device(struct hk_device *dev);
uint64_t hk_apple9_native_index_shapes(void);
/* Separate default-off GS route; also requires native mesh and exact classes. */
bool hk_apple9_native_gs_enabled(void);
bool hk_apple9_native_tess_enabled(void);
bool hk_apple9_native_task_bridge_enabled(void);
bool hk_apple9_native_object_enabled(void);
bool hk_apple9_native_object_available(const struct hk_physical_device *pdev);
bool hk_apple9_native_object_available_device(struct hk_device *dev);

/* Generic 32-bit stage I/O shares the native mesh per-vertex layout. Builtins
 * other than Position need their own stage/rasterization semantics. */
#define HK_APPLE9_NATIVE_VERTEX_IO \
   (VARYING_BIT_POS | (UINT64_MAX << VARYING_SLOT_VAR0))
static inline bool
hk_apple9_native_vertex_slot(unsigned location)
{
   return location == VARYING_SLOT_POS ||
          (location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_VAR0 + 32);
}

/* Research (HK_APPLE9_NATIVE_GS_PACK, default off): native GS runs up to
 * HK_APPLE9_NATIVE_GS_PACK input triangles per T32 mesh group, one GS
 * invocation per lane, in fixed per-lane output slots (T32 V24/P17 for 3
 * output vertices, T32 V32/P16 for 4). A packed body is recognised by its
 * T32 class alone. */
#define HK_APPLE9_NATIVE_GS_PACK 8
bool hk_apple9_native_gs_pack_enabled(void);
/* Research (HK_APPLE9_NATIVE_TES_PACK, default off): native TES evaluates
 * HK_APPLE9_NATIVE_GS_PACK tessellated triangles per T32 mesh group (lane l
 * evaluates corner l % 3 of triangle l / 3), published as T32 V24/P17. */
bool hk_apple9_native_tes_pack_enabled(void);
/* Whether a GS mesh body's class is one of the two packed layouts. */
bool hk_apple9_native_gs_packed_class(unsigned threads, unsigned vertices,
                                      unsigned primitives);

/* Stage policy is narrower than native-index qualification. These bits name
 * the API producer, not the generated mesh/object execution stages. */
#define HK_APPLE9_NATIVE_GS_BIT (1u << 0)
#define HK_APPLE9_NATIVE_TES_BIT (1u << 1)
#define HK_APPLE9_NATIVE_STAGE_BITS \
   (HK_APPLE9_NATIVE_GS_BIT | HK_APPLE9_NATIVE_TES_BIT)
uint8_t hk_apple9_native_stage_policy(const struct hk_physical_device *pdev);
bool hk_apple9_native_stage_available_device(struct hk_device *dev,
                                             mesa_shader_stage stage);
bool hk_apple9_native_stage_class(mesa_shader_stage stage, unsigned threads,
                                  unsigned vertices, unsigned primitives);

bool hk_apple9_native_mesh_attributes_enabled(void);
bool hk_apple9_native_mesh_primitive_lowering_enabled(void);

/* Per-draw parameters for Zero-PREGFX fused VS + GS -> native MESH draws.
 * Uploaded to root.draw.geometry_params (loaded via geometry_param_buffer_poly). */
struct hk_apple9_fused_gs_params {
   /* Index buffer GPU address pre-offset by firstIndex * index_size_B (0 if direct). */
   uint64_t index_buffer;
   /* Remaining index buffer byte size from index_buffer (0 if direct or OOB). */
   uint64_t index_buffer_size;
   /* Input API topology (enum mesa_prim: POINTS, LINES, LINE_STRIP,
    * TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN). */
   uint32_t input_topology;
   /* Index element byte size: 0 for direct draws, 1/2/4 for indexed draws. */
   uint32_t index_size_B;
   /* firstVertex (direct) or vertexOffset (indexed). */
   uint32_t first_vertex;
   /* firstInstance. */
   uint32_t base_instance;
   /* Input primitives in this draw (or chunk), per instance. */
   uint32_t input_primitives;
   /* First input primitive index (for TVB-safe chunking of very large draws). */
   uint32_t first_primitive;
   /* X dimension of the 2D-tiled mesh grid (group_linear = wg.y * groups_x + wg.x). */
   uint32_t groups_x;
   /* Zero-based instance offset for chunked multi-instance draws (added to wg.z). */
   uint32_t first_instance_offset;
};

/* Per-draw parameters for Zero-PREGFX fused OBJECT(VS+TCS) -> MESH(TES) draws.
 * Uploaded to root.draw.tess_params (loaded via tess_param_buffer_poly). */
struct hk_apple9_fused_tess_params {
   /* Index buffer GPU address pre-offset by firstIndex * index_size_B (0 if direct). */
   uint64_t index_buffer;
   /* Remaining index buffer byte size from index_buffer (0 if direct or OOB). */
   uint64_t index_buffer_size;
   /* Index element byte size: 0 for direct draws, 1/2/4 for indexed draws. */
   uint32_t index_size_B;
   /* Dynamic patchControlPoints (1..32). */
   uint32_t input_patch_size;
   /* firstVertex (direct) or vertexOffset (indexed). */
   uint32_t first_vertex;
   /* firstInstance. */
   uint32_t base_instance;
   /* Patches in this draw (or chunk), per instance. */
   uint32_t patches_per_instance;
   /* First patch index in this chunk (for >max_groups patch chunking). */
   uint32_t first_patch;
   /* X dimension of the 2D-tiled parent object grid (patch_linear = wg.y * groups_x + wg.x). */
   uint32_t groups_x;
   /* Zero-based instance offset for chunked multi-instance draws (added to wg.z). */
   uint32_t first_instance_offset;
   /* Non-zero if emitted triangle winding must be flipped (ccw ^ lower_left). */
   uint32_t flip_winding;
   uint32_t _pad;
};

/* Explicit research selection, e.g. T16V3P2,T32V32P42, after qualification.
 * Empty, boolean/wildcard or malformed selections enable no new class.
 * The pure parser/predicate are shared with host tests. */
uint64_t hk_apple9_packed_mesh_shapes_parse(const char *selection);
uint64_t hk_apple9_native_mesh_packed_shapes(void);
bool hk_apple9_packed_mesh_class_selected(uint64_t selection,
                                         unsigned threads, unsigned vertices,
                                         unsigned primitives);
bool hk_apple9_packed_mesh_class_fit(uint64_t selection,
                                     unsigned min_threads,
                                     unsigned min_vertices,
                                     unsigned min_primitives,
                                     unsigned *out_threads,
                                     unsigned *out_vertices,
                                     unsigned *out_primitives);
bool hk_apple9_native_mesh_packed_shape_eligible(const nir_shader *nir);
bool hk_apple9_native_mesh_workgroup_eligible(const nir_shader *nir);

enum hk_apple9_native_stage {
   HK_APPLE9_NATIVE_NONE,
   HK_APPLE9_NATIVE_OBJECT,
   HK_APPLE9_NATIVE_MESH,
};

/* Stage-local byte offsets. These are not GPU addresses or resource aliases.
 * Object and mesh have separate local allocations. */
#define HK_APPLE9_NATIVE_LOCAL_BYTES 128
#define HK_APPLE9_NATIVE_OUTPUT_ROOT 0
#define HK_APPLE9_NATIVE_GRID_ROOT 64
/* Fits alongside the entire admitted V3/P1 output interface (2080 bytes)
 * in one 32-KiB allocation. Larger output classes need a new storage model. */
#define HK_APPLE9_NATIVE_SHARED_BYTES 28672

/* Actual hardware threadgroup memory allocation (power of two, >= 128 B)
 * requested by hk_apple9_native_launch_parameters() for a native OBJECT stage. */
static inline uint64_t
hk_apple9_native_object_tg_memory_bytes(
   const struct agx_apple9_mesh_body_info *info)
{
   uint64_t raw =
      (uint64_t)HK_APPLE9_NATIVE_GRID_ROOT + info->local_allocation_B;
   if (raw > 32768)
      return raw;
   return util_next_power_of_two(
      MAX2((unsigned)HK_APPLE9_NATIVE_LOCAL_BYTES, (unsigned)raw));
}

/* One taskless object publishes one mesh-child grid. Apple's Metal Feature
 * Set Tables give Apple9 a total of 2^20 - 1 mesh groups per grid. Retain the
 * existing, more conservative Vulkan per-axis bound. These are grid limits,
 * independent of the physical object workgroup size in the command stream.
 * https://developer.apple.com/metal/capabilities/ */
#define HK_APPLE9_NATIVE_MAX_GROUP_COUNT 65535u
#define HK_APPLE9_NATIVE_MAX_GROUP_TOTAL ((1u << 20) - 1)
/* Profitability cap on multi-chunk splitting for fused native tessellation
 * draws. Ring-bounded layouts allow at most one chunk; unbounded layouts
 * split only when the parent object grid exceeds AGX_NATIVE_OBJECT_MAX_GROUPS
 * (2^20 - 1 patches) and each chunk flushes a separate multi-parent dispatch.
 * Capping at 4 chunks preserves small/medium draws Zero-PREGFX while sending
 * giant multi-chunk draws cleanly to SW tess. */
#define HK_APPLE9_NATIVE_TESS_MAX_CHUNKS 4u

/* Validate a nonempty direct child grid without overflowing the product.
 * The draw entrypoint handles zero dimensions as a no-op before calling this. */
bool hk_apple9_native_mesh_grid_valid(const uint32_t groups[3]);

/* Zero rejects an invalid logical shape before any multiplication overflows. */
unsigned hk_apple9_native_mesh_threads(const uint16_t local_size[3]);
/* Research envelope beyond one publication round at physical T16/T32;
 * independent of the qualified class list. */
bool hk_apple9_native_mesh_round_counts_eligible(const uint16_t local_size[3],
                                                 unsigned vertices,
                                                 unsigned primitives);
/* Research envelope of the publication model at T32: every T16/T32
 * declaration up to V256/P256 except V3/P1. */
bool hk_apple9_native_mesh_model_counts_eligible(const uint16_t local_size[3],
                                                 unsigned vertices,
                                                 unsigned primitives);
bool hk_apple9_native_mesh_round_shape_eligible_ex(const nir_shader *nir,
                                                   bool allow_task_payload);
bool hk_apple9_native_mesh_round_shape_eligible(const nir_shader *nir);

bool hk_apple9_native_shader_valid(
   const struct agx_shader_info *code,
   const struct agx_apple9_mesh_body_info *body,
   enum hk_apple9_native_stage stage);
bool hk_apple9_native_launch_parameters(
   struct agx_apple9_launch_parameters *out,
   const struct agx_shader_info *code,
   const struct agx_apple9_mesh_body_info *body,
   enum hk_apple9_native_stage stage, uint64_t shader_base,
   uint64_t code_address);

#define HK_APPLE9_MESH_COMMAND_BYTES 160
bool hk_apple9_mesh_command_build_delta_ex(
   void *out, size_t capacity, const struct agx_apple9_render_pipeline *p,
   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS], const uint32_t groups[3],
   const uint32_t object_local_size[3], bool enable, bool multi_parent,
   const uint8_t *previous, const uint8_t *record, size_t *bytes);
bool hk_apple9_mesh_command_build_delta(
   void *out, size_t capacity, const struct agx_apple9_render_pipeline *p,
   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS], const uint32_t groups[3],
   const uint32_t object_local_size[3], bool enable, const uint8_t *previous,
   const uint8_t *record, size_t *bytes);
bool hk_apple9_mesh_command_build_ex(
   void *out, size_t capacity,
   const struct agx_apple9_render_pipeline *pipeline,
   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS],
   const uint32_t object_groups[3], const uint32_t object_local_size[3],
   bool enable, bool multi_parent);
bool hk_apple9_mesh_command_build(
   void *out, size_t capacity,
   const struct agx_apple9_render_pipeline *pipeline,
   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS],
   const uint32_t object_groups[3], const uint32_t object_local_size[3],
   bool enable);
