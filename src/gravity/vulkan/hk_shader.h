/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "gravity/compiler/agx_apple9_profile.h"
#include "gravity/compiler/agx_compile.h"
#include "gravity/lib/agx_apple9.h"
#include "poly/nir/poly_nir.h"
#include "util/macros.h"
#include "hk_descriptor_memory.h"

#include "hk_device.h"
#include "hk_apple9_mesh.h"
#include "hk_device_memory.h"
#include "hk_private.h"

#include "shader_enums.h"
#include "vk_pipeline_cache.h"

#include "nir_defines.h"

#include "vk_shader.h"

#define HK_APPLE9_DRAW_PARAMS_BINDING 65
#define HK_APPLE9_VERTEX_BUFFERS_BINDING 66

/* Compiler-only research: no native GS draw routing is enabled. */
struct hk_apple9_graphics_key;
nir_shader *hk_nir_gs_mesh(const nir_shader *source, const char **reason);
nir_shader *hk_nir_gs_mesh_take(nir_shader *nir, const char **reason);
bool hk_apple9_compile_gs_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason);

struct hk_gs_fused_options {
   struct agx_apple9_vertex_layout vertex;
   bool fs_reads_clip_distance;
};

struct hk_gs_fused_info {
   uint8_t pack;
   uint8_t input_primitive;
   uint8_t output_primitive;
   uint8_t invocations;
   uint16_t prims_per_run;
};

nir_shader *hk_nir_gs_fused_mesh(const nir_shader *vs_source,
                                 const nir_shader *gs_source,
                                 const struct hk_gs_fused_options *options,
                                 struct hk_gs_fused_info *out_info,
                                 const char **reason);
bool hk_apple9_compile_gs_fused_mesh_program(
   const nir_shader *vs_source, const nir_shader *gs_source,
   const struct hk_apple9_graphics_key *key, struct agx_shader_part *out,
   struct agx_apple9_mesh_body_info *info, struct hk_gs_fused_info *fused_info,
   const char **reason);

/* Position-only TES realization over the existing tessellator index stream. */
nir_shader *hk_nir_tes_mesh(const nir_shader *source, const char **reason);
nir_shader *hk_nir_tes_mesh_take(nir_shader *nir, const char **reason);
nir_shader *hk_nir_tes_mesh_object(void);
bool hk_apple9_compile_tes_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason);

struct hk_tess_fused_options {
   struct agx_apple9_vertex_layout vertex;
   uint32_t max_payload_bytes;
   bool fs_reads_clip_distance;
   bool fs_reads_primitive_id;
   bool has_fs_inputs_mask;
   /* Per-VAR0..31 4-bit component mask read by the linked FS (all 0 for depth-only). */
   uint8_t fs_var_read_mask[32];
};

enum hk_apple9_tess_level_kind {
   HK_APPLE9_TESS_LEVEL_DYNAMIC = 0,
   HK_APPLE9_TESS_LEVEL_CONST = 1,
   HK_APPLE9_TESS_LEVEL_PUSH = 2,
   /* Upper bound min(const_value, push_float) when push_float is non-NaN;
    * falls back to const_value if push_float is NaN. */
   HK_APPLE9_TESS_LEVEL_PUSH_BOUND = 3,
};

struct hk_apple9_tess_level_src {
   uint8_t kind;
   uint8_t _pad;
   uint16_t push_offset;
   float const_value;
};

struct hk_apple9_tess_push_levels {
   bool valid;
   uint8_t domain;
   uint8_t spacing;
   uint8_t shape_max_v;
   uint8_t shape_max_p;
   uint8_t shape_p_strip;
   uint8_t shape_tile_w;
   uint8_t shape_tile_h;
   bool shape_allow_whole;
   struct hk_apple9_tess_level_src levels[6]; /* 0..3 outer, 4..5 inner */
};

struct hk_tess_fused_info {
   uint8_t domain;
   uint8_t spacing;
   bool ccw;
   uint8_t tcs_vertices_out;
   uint16_t payload_B;
   uint16_t max_child_wgs_per_patch;
   uint16_t max_prims_per_patch;
   uint8_t mesh_max_vertices;
   uint8_t mesh_max_primitives;
   bool constant_tess_levels;
   struct hk_apple9_tess_push_levels push_levels;
};

struct hk_shader_info;

/* Evaluate exact per-patch child mesh WG and primitive counts for a fused
 * tessellation shader using its compile-time bounds or draw-time push
 * constants when all active tess levels come from constants / push constants. */
void hk_nir_tess_fused_eval_counts(const struct hk_shader_info *info,
                                   const uint8_t *push_data,
                                   uint32_t *out_max_child_wgs,
                                   uint32_t *out_max_patch_prims);

/* Bit-exact lossless packing for statically-typed UNORM8/SNORM8 (and
 * UNORM16/SNORM16) vertex-attribute passthroughs across monolithic
 * VS -> TCS -> TES -> FS fused tessellation pipelines. Packs qualifying
 * patch-constant TES->FS varyings into a single flat u32 varying (and compacts
 * exclusive TCS->TES task payload slots) without altering fallback NIR. */
bool hk_nir_tess_fused_pack_vbo_varyings(
   nir_shader *vs, nir_shader *tcs, nir_shader *tes, nir_shader *fs,
   const struct agx_apple9_vertex_layout *vertex);

/* Zero-PREGFX fused OBJECT(VS+TCS) -> MESH(TES) lowering over per-parent
 * GET_SR 0xe0/0xe1 task payload. Clones all inputs without mutating them. */
bool hk_nir_tess_fused_lower(const nir_shader *vs_source,
                             const nir_shader *tcs_source,
                             const nir_shader *tes_source,
                             const struct hk_tess_fused_options *options,
                             nir_shader **out_object,
                             nir_shader **out_mesh,
                             struct hk_tess_fused_info *out_info,
                             const char **reason);

struct hk_physical_device;
struct hk_pipeline_compilation_ctx;
struct vk_descriptor_set_layout;
struct vk_graphics_pipeline_state;
struct vk_pipeline_cache;
struct vk_pipeline_layout;
struct vk_pipeline_robustness_state;
struct vk_shader_module;

struct hk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct hk_tess_info) == 4, "packed");

static struct hk_tess_info
hk_tess_info_merge(struct hk_tess_info a, struct hk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct hk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

/* Texture operations whose descriptors may change after binding, or be
 * partially bound, must read them at execution time on G16. */
#define HK_APPLE9_TEX_DESCRIPTOR_HANDLE (1u << 30)

/* On a nir_texop_lod_bias query: load the sampler's explicit-LOD nudge
 * instead of its LOD bias, see hk_lower_d3d_txl_rounding. */
#define HK_TEX_TXL_NUDGE (1u << 29)

/* Mesh shaders run as compute kernels. Each workgroup writes a record of
 * group_stride_B bytes: its vertex and primitive counts, padded to 16 bytes,
 * then max_vertices vertices and max_primitives primitives, each the varying
 * slots in the respective mask in slot order. Slots have the words of a vec4,
 * except for scalar builtins and the primitive indices, which have their
 * primitive's vertex count. The primitive slots include the primitive indices
 * and cull flag.
 */
struct hk_mesh_info {
   uint64_t vertex_slots;
   uint64_t primitive_slots;
   uint32_t group_stride_B;
   uint16_t max_vertices, max_primitives;
   uint8_t prim;
   /* The rasterizing vertex shader draws HK_PP_MAX_VERTICES points per
    * triangle for polygon mode POINT. */
   uint8_t polygon_points;
   /* Native mesh consumes compute-task grids and per-parent payloads. */
   bool native_task_bridge;
   /* Native mesh runs behind a hardware OBJECT stage (no Task Bridge). */
   bool native_object;
   /* Native mesh publishes root.draw.view_index as its rasterizer Layer. */
   bool native_multiview;
   uint8_t _pad[3];
};

/* Fragment shaders read user per-primitive varyings from the output records,
 * at the byte offset of the primitive, passed in this flat varying.
 */
#define HK_MESH_PRIMITIVE_SLOT VARYING_SLOT_VAR31

static inline bool
hk_mesh_varying_in_memory(unsigned slot)
{
   return slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_VAR31;
}

static inline unsigned
hk_mesh_slot_words(const struct hk_mesh_info *info, unsigned slot)
{
   switch (slot) {
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_PRIMITIVE_SHADING_RATE:
   case VARYING_SLOT_CULL_PRIMITIVE:
      return 1;
   case VARYING_SLOT_PRIMITIVE_INDICES:
      return mesa_vertices_per_prim(info->prim);
   default:
      return 4;
   }
}

/* Offset of a slot within a vertex or primitive, or their size with
 * slot = 64.
 */
static inline unsigned
hk_mesh_slot_offset_B(const struct hk_mesh_info *info, uint64_t mask,
                      unsigned slot)
{
   unsigned offset_B = 0;
   u_foreach_bit64(s, mask & BITFIELD64_MASK(slot))
      offset_B += hk_mesh_slot_words(info, s) * 4;
   return offset_B;
}

static inline unsigned
hk_mesh_primitives_offset_B(const struct hk_mesh_info *info)
{
   return 16 + info->max_vertices *
                  hk_mesh_slot_offset_B(info, info->vertex_slots, 64);
}

void hk_mesh_gather_info(const nir_shader *nir, struct hk_mesh_info *info);
void hk_nir_lower_mesh_main(nir_shader *nir, const struct hk_mesh_info *info);
void hk_nir_lower_task_main(nir_shader *nir);
void hk_nir_lower_native_mesh_task(nir_shader *nir);
uint32_t hk_nir_task_mesh_bound(nir_shader *nir);
#define HK_MAX_PER_VERTEX_COPIES 4

/* Per-vertex fragment inputs read as flat copies: the VAR slot of each input,
 * and the first of the three consecutive VAR slots holding its values for API
 * vertices 0, 1 and 2. Barycentrics are read by interpolating the one-hot
 * weights of the API vertices from bary[0] (perspective) and bary[1]
 * (linear), as VAR index + 1 or 0 if unused.
 */
struct hk_per_vertex_copies {
   unsigned count;
   uint8_t location[HK_MAX_PER_VERTEX_COPIES];
   uint8_t copy_base[HK_MAX_PER_VERTEX_COPIES];
   uint8_t bary[2];
};

static inline bool
hk_per_vertex_copies_used(const struct hk_per_vertex_copies *map)
{
   return map->count || map->bary[0] || map->bary[1];
}

/* Map a per-vertex copy base slot and API vertex index (0..2) to its varying
 * slot. The third dedicated base VARYING_SLOT_BFC0 wraps vertex 2 onto
 * VARYING_SLOT_CLIP_VERTEX because BFC0..BFC1 only span two slots. */
static inline unsigned
hk_per_vertex_copy_slot(uint8_t base, unsigned vertex)
{
   if (base == VARYING_SLOT_BFC0 && vertex == 2)
      return VARYING_SLOT_CLIP_VERTEX;
   return base + vertex;
}

struct hk_pp_info;
nir_shader *hk_nir_mesh_rast(const nir_shader *mesh,
                             const struct hk_mesh_info *info,
                             const struct hk_per_vertex_copies *copies,
                             const struct hk_pp_info *polygon_points,
                             bool point_size);

/* Line stipple takes two generic varying slots of the last
 * pre-rasterization stage: a flat start and counter, and a noperspective
 * axis point. HK_APPLE9_OUTPUT_COMPONENTS matches the advertised output limit.
 */
#define HK_APPLE9_STIPPLE_COMPONENTS 5
#define HK_APPLE9_OUTPUT_COMPONENTS (AGX_APPLE9_MAX_VARYING_COMPONENTS - 2)

struct hk_apple9_stipple_slots {
   unsigned start, axis;
};

struct hk_apple9_stipple_slots
hk_apple9_line_stipple_slots(const nir_shader *last, const nir_shader *fs);

struct hk_shader_info {
   /* A complete compiler contract, qualified by the gated native uploader.
    * Kept with serialized shader metadata, never inferred from shader bytes. */
   uint8_t apple9_native_stage;
   /* Native GS mesh bodies: GS invocations per input primitive. Each
    * (primitive, invocation) is one logical GS run (hk_nir_gs_mesh). */
   uint8_t apple9_gs_invocations;
   /* Zero-PREGFX fused VS + GS -> native MESH metadata. */
   bool apple9_gs_fused;
   uint8_t apple9_gs_pack;
   uint8_t apple9_gs_input_primitive;
   uint16_t apple9_gs_prims_per_run;
   /* Zero-PREGFX fused OBJECT(VS+TCS) -> MESH(TES) metadata. */
   bool apple9_tess_fused;
   bool apple9_tess_const_levels;
   uint16_t apple9_tess_payload_B;
   uint16_t apple9_tess_max_child_wgs;
   uint16_t apple9_tess_max_patch_prims;
   struct hk_apple9_tess_push_levels apple9_tess_push_levels;
   struct agx_apple9_mesh_body_info apple9_native;

   uint8_t apple9_render_targets;
   /* Fragment shaders: stippled lines rasterize with the first vertex
    * provoking whatever the API mode, which no other input observes. */
   bool apple9_line_stipple_first;
   /* Vertex shaders using a prolog: pipeline disabled vertex input robustness. */
   bool apple9_vertex_unbounded;
   /* Fragment shaders using an epilog: pipeline left these attachments VK_FORMAT_UNDEFINED. */
   uint8_t apple9_undefined_colors;
   struct agx_apple9_compute_profile apple9;
   union {
      struct {
         struct agx_interp_info interp;

         bool early_fragment_tests;
         bool writes_memory;
         /* Fragment-invocation statistics counter was lowered inline
          * (early_fragment_tests, per-sample, or non-epilog FS). Otherwise the
          * draw-time linked stats prolog counts invocations while queried. */
         bool inline_statistics;

         /* Per-vertex inputs and barycentrics written by the passthrough
          * GS, as in struct hk_per_vertex_copies. */
         uint8_t per_vertex_copies;
         uint8_t per_vertex_location[HK_MAX_PER_VERTEX_COPIES];
         uint8_t per_vertex_copy_base[HK_MAX_PER_VERTEX_COPIES];
         uint8_t per_vertex_bary[2];
         /* Varying slot of the vertex-stage line stipple start, or 0. */
         uint8_t line_stipple_start;
         /* Polygon mode POINT is drawn by the software polygon stage, and
          * the varying slot of the polygon's facing, or 0 when unread. */
         bool polygon_points;
         uint8_t polygon_face_slot;
      } fs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct hk_tess_info info;
      } tess;

      struct poly_gs_info gs;

      struct hk_mesh_info mesh;

      struct {
         uint32_t payload_B;
         /* Per task workgroup; UINT32_MAX means not statically bounded. */
         uint32_t max_mesh_groups;
      } task;

      /* Used to initialize the union for other stages */
      uint8_t _pad[32];
   };

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   mesa_shader_stage stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t set_count;
   /* Apple9: cull distances are published as one flat mask per vertex, which
    * a geometry stage must combine over each primitive, rather than as
    * interpolated per-plane values. See hk_apple9_compile_vertex_culls. */
   bool cull_mask;

   /* XXX: is there a less goofy way to do this? I really don't want dynamic
    * allocation here.
    */
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[64];
};

struct hk_shader {
#ifndef NDEBUG
   /* Final-NIR validation uses the owning API shader's upload mask. */
   const struct hk_api_shader *debug_owner;
#endif
   struct agx_shader_part b;

   struct hk_shader_info info;

   const void *code_ptr;
   uint32_t code_size;

   const void *data_ptr;
   uint32_t data_size;

   /* Borrowed from apple9_body. */
   struct agx_bo *bo;
   struct agx_shader_body *apple9_body;

   /* G16 compute: launch program template, built once at upload from the
    * profile and preamble address and patched per dispatch. NULL falls back
    * to building the program every dispatch. */
   struct agx_apple9_launch_template *apple9_launch;
   /* G16 graphics: launch templates per tile allocation and sample count,
    * built on first use. NULL builds every launch. */
   struct agx_apple9_stage_launch_cache *apple9_stage_launch;
   /* G16 graphics: device-unique id of this compiled stage, so consecutive
    * draws with the same shaders reuse launches and coefficient tables (see
    * agx_apple9_prepare_draw_alloc). Zero disables the reuse. */
   uint64_t apple9_program_id;
   /* G16 graphics: the native stage description, built once at upload and
    * linked by pointer at draw time. */
   struct agx_apple9_render_stage apple9_stage;
   struct hk_descriptor_allocation apple9_coefficients;

   /* Address of the start of the shader data section */
   uint64_t data_addr;
};

enum hk_vs_variant {
   /* Hardware vertex shader, when next stage is fragment */
   HK_VS_VARIANT_HW,

   /* Hardware compute shader, when next is geometry/tessellation */
   HK_VS_VARIANT_SW,

   HK_VS_VARIANTS,
};

enum hk_gs_variant {
   /* Hardware vertex shader used for rasterization */
   HK_GS_VARIANT_RAST,

   /* Main compute shader */
   HK_GS_VARIANT_MAIN,

   /* Count compute shader */
   HK_GS_VARIANT_COUNT,

   /* Pre-GS compute shader */
   HK_GS_VARIANT_PRE,

   HK_GS_VARIANTS,

   /* A native GS leaf retains its API stage, but owns mesh and object code.
    * These names must never be used for the compute/raster realization. */
   HK_GS_VARIANT_NATIVE_MESH = HK_GS_VARIANT_RAST,
   HK_GS_VARIANT_NATIVE_OBJECT = HK_GS_VARIANT_MAIN,
};

enum hk_mesh_variant {
   /* Compute kernel writing the output records */
   HK_MESH_VARIANT_MAIN,

   /* Hardware vertex shader rasterizing the records */
   HK_MESH_VARIANT_RAST,

   HK_MESH_VARIANTS,

   /* Under the native gate the second variant is the generated object stage,
    * instead of an emulation rasterization shader. */
   HK_MESH_VARIANT_OBJECT = HK_MESH_VARIANT_RAST,
};

/* clang-format off */
static const char *hk_gs_variant_name[] = {
   [HK_GS_VARIANT_RAST] = "Rasterization",
   [HK_GS_VARIANT_MAIN] = "Main",
   [HK_GS_VARIANT_COUNT] = "Count",
   [HK_GS_VARIANT_PRE] = "Pre-GS",
};
/* clang-format on */

static inline unsigned
hk_num_variants(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      return HK_VS_VARIANTS;

   case MESA_SHADER_GEOMETRY:
      return HK_GS_VARIANTS;

   case MESA_SHADER_MESH:
      return HK_MESH_VARIANTS;

   default:
      return 1;
   }
}

/*
 * An hk_api shader maps 1:1 to a VkShader object. An hk_api_shader may contain
 * multiple hardware hk_shader's, built at shader compile time. This complexity
 * is required to efficiently implement the legacy geometry pipeline.
 */
/* Graphics stages of an Apple9 draw-time variant, indexed by stage. Compute
 * is never among them. */
#define HK_APPLE9_GFX_STAGES (MESA_SHADER_MESH + 1)

enum hk_pre_raster_reads {
   /* Sampled, storage or input-attachment images: may alias attachments. */
   HK_READS_IMAGES = BITFIELD_BIT(0),
   /* Buffers of any kind, vertex attributes, descriptors and indirect
    * arguments: only memory-writing renders can produce them. */
   HK_READS_BUFFERS = BITFIELD_BIT(1),
   /* Not classified: keep the tiling wait. */
   HK_READS_UNKNOWN = BITFIELD_BIT(2),
   /* hk-internal pre-rasterization work (meta, transform feedback,
    * tessellation, geometry): keep the tiling wait. */
   HK_READS_INTERNAL = BITFIELD_BIT(3),
   /* A meta draw (statistics only; its shaders carry their own class). */
   HK_READS_META = BITFIELD_BIT(4),
};

uint8_t hk_nir_pre_raster_reads(const nir_shader *nir);
bool hk_nir_reads_draw_id(const nir_shader *nir);
uint64_t hk_nir_push_constant_mask(const nir_shader *nir);
#ifndef NDEBUG
uint64_t hk_nir_final_push_constant_mask(const nir_shader *nir);
#endif

struct hk_api_shader {
   struct vk_shader vk;

   /* Is this an internal passthrough geometry shader? */
   bool is_passthrough;

   /* The program stores to memory or runs atomics (any variant). */
   bool writes_memory;

   /* Memory a pre-rasterization program reads (enum hk_pre_raster_reads):
    * images or buffers. Decides whether its render's tiling must wait for
    * earlier renders' writes (hk_queue). */
   uint8_t pre_raster_reads;

   /* The program reads the draw index (gl_DrawID / DrawIndex), which hk
    * passes in the root descriptor table. Draws whose bound stages do not
    * read it leave the root unchanged, so consecutive draws of a multi-draw
    * share one root upload. */
   bool reads_draw_id;

   /* API push-constant dwords read by this stage. Collected before Apple9
    * root lowering; an indirect or unrecognized root access keeps all words.
    * This is upload metadata and does not change the compiled program. */
   uint64_t push_constant_mask;

   /* Static depth/stencil attachment feedback loop bit from pipeline creation
    * (VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT).
    * Checked at draw time by hk_apple9_draw_late_zs to trigger epilog late_zs. */
   bool zs_self_dep;

   /* Pipelines own an emulated realization and this optional native leaf. */
   struct hk_api_shader *apple9_native;

   /* Optional secondary Phase 1 Task Bridge native leaf retained alongside a
    * native OBJECT -> MESH leaf so draw-time rejections cascade cleanly to
    * Task Bridge before falling back to Emulation. */
   struct hk_api_shader *apple9_task_bridge;

   /* Optional software polygon-mode POINT realization for an application
    * geometry shader compiled with dynamic polygon mode / ESO / GPL. */
   struct hk_api_shader *apple9_gs_polygon_points;

   /* Optional native per-vertex / barycentric realization for a standalone
    * fragment shader (GPL/ESO) whose primary realization lowered per-vertex
    * inputs to passthrough-GS flat copies. Used when linked at draw time
    * with an application geometry shader, which exports ordinary varyings. */
   struct hk_api_shader *apple9_fs_native_per_vertex;

   /* Native mesh/FS realizations have the same content-derived compile ID.
    * Store the ID in cache blobs so independently reloaded stages can still
    * prove that their producer/consumer interfaces were compiled together. */
   bool apple9_native_leaf;
   blake3_hash apple9_native_pair_id;

   /* Shader capability for native routing. Draw state can still require the
    * emulated realization. */
   bool apple9_native_candidate;
   /* Whole-primitive culling that requires degenerate triangles to be empty,
    * including CullPrimitiveEXT and per-vertex CullDistance. */
   bool apple9_native_cull_primitive;

   /* A native TES uses the program-info union for mesh metadata. Retain
    * the API tessellation mode independently for the existing producer. */
   struct hk_tess_info apple9_native_tess;
   /* Production admission of the API producer. Zero denotes ordinary fallback
    * or an explicitly selected research realization. */
   uint8_t apple9_native_stage_mask;

   /* True once compiled variants are populated (false on test-stub shells). */
   bool compiled;

   /* Immutable after compilation/cache reload; avoids walking variants at
    * every shader bind, including shaders with no constant data. */
   uint64_t constant_data_addr;

   struct hk_shader variants[];
};

static inline struct hk_api_shader *
hk_api_shader_native(const struct hk_api_shader *shader)
{
   if (!shader)
      return NULL;
   return shader->apple9_native_leaf ? (struct hk_api_shader *)shader
                                     : shader->apple9_native;
}

static inline struct hk_api_shader *
hk_api_shader_task_bridge(const struct hk_api_shader *shader)
{
   if (!shader || shader->apple9_native_leaf)
      return NULL;
   return shader->apple9_task_bridge;
}

static inline struct hk_shader *
hk_native_producer_main(struct hk_api_shader *shader)
{
   assert(shader->vk.stage == MESA_SHADER_MESH ||
          shader->vk.stage == MESA_SHADER_GEOMETRY ||
          shader->vk.stage == MESA_SHADER_TESS_EVAL);
   return &shader->variants[shader->vk.stage == MESA_SHADER_GEOMETRY
                               ? HK_GS_VARIANT_NATIVE_MESH
                            : shader->vk.stage == MESA_SHADER_TESS_EVAL
                               ? HK_VS_VARIANT_HW
                               : HK_MESH_VARIANT_MAIN];
}

static inline struct hk_shader *
hk_native_producer_object(struct hk_api_shader *shader)
{
   assert(shader->vk.stage == MESA_SHADER_MESH ||
          shader->vk.stage == MESA_SHADER_GEOMETRY ||
          shader->vk.stage == MESA_SHADER_TESS_EVAL);
   return &shader->variants[shader->vk.stage == MESA_SHADER_GEOMETRY
                               ? HK_GS_VARIANT_NATIVE_OBJECT
                            : shader->vk.stage == MESA_SHADER_TESS_EVAL
                               ? HK_VS_VARIANT_SW
                               : HK_MESH_VARIANT_OBJECT];
}

struct vk_graphics_pipeline_state;
struct vk_depth_stencil_state;

bool hk_apple9_zs_writes(const struct vk_depth_stencil_state *ds,
                         VkImageAspectFlags aspects);

void hk_apple9_bodies_finish(struct hk_device *dev);

bool hk_apple9_lower_sparse_residency(nir_shader *nir);
bool hk_lower_bounded_global_loads(nir_shader *nir, bool soft_fault, bool apple9);

#define hk_foreach_variant(api_shader, var)                                    \
   for (struct hk_shader *var = api_shader->variants;                          \
        var < api_shader->variants + hk_num_variants(api_shader->vk.stage);    \
        ++var)

static const char *
hk_variant_name(struct hk_api_shader *obj, struct hk_shader *variant)
{
   unsigned i = variant - obj->variants;
   assert(i < hk_num_variants(obj->vk.stage));

   if (hk_num_variants(obj->vk.stage) == 1) {
      return NULL;
   } else if (obj->vk.stage == MESA_SHADER_GEOMETRY) {
      assert(i < ARRAY_SIZE(hk_gs_variant_name));
      return hk_gs_variant_name[i];
   } else if (obj->vk.stage == MESA_SHADER_MESH) {
      return i == HK_MESH_VARIANT_RAST ? "Rasterization" : "Main";
   } else {
      assert(i < 2);
      return i == HK_VS_VARIANT_SW ? "Software" : "Hardware";
   }
}

static struct hk_shader *
hk_only_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   assert(hk_num_variants(obj->vk.stage) == 1);
   return &obj->variants[0];
}

/* Large constants are extracted before variants are cloned or generated.
 * Generated variants may not use them, but all original variants retain the
 * same data layout, including the optional native realization. Any uploaded
 * copy can serve the API stage's root slot without a draw-path rebind.
 */
static inline void
hk_api_shader_init_constant_data(struct hk_api_shader *shader)
{
   const struct hk_shader *data = NULL;
   const struct hk_api_shader *paths[] = {
      shader,
      shader->apple9_native,
      shader->apple9_task_bridge,
      shader->apple9_fs_native_per_vertex,
   };
   for (unsigned p = 0; p < ARRAY_SIZE(paths); ++p) {
      if (!paths[p])
         continue;
      for (unsigned v = 0; v < hk_num_variants(shader->vk.stage); ++v) {
         const struct hk_shader *s = &paths[p]->variants[v];
         if (!s->data_size)
            continue;
         assert(!data || (data->data_size == s->data_size &&
                         !memcmp(data->data_ptr, s->data_ptr, s->data_size)));
         if (!data)
            data = s;
      }
   }
   shader->compiled = true;
   shader->constant_data_addr = data ? data->data_addr : 0;
}

static inline uint64_t
hk_shader_constant_data(const struct hk_api_shader *shader)
{
   return shader ? shader->constant_data_addr : 0;
}

static struct hk_shader *
hk_any_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   return &obj->variants[0];
}

static struct hk_shader *
hk_main_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_MAIN];
}

static struct hk_shader *
hk_count_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_COUNT];
}

static struct hk_shader *
hk_pre_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_PRE];
}

extern const struct vk_device_shader_ops hk_device_shader_ops;

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev);

static inline nir_address_format
hk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage,
                        unsigned nr_vbos);

bool hk_nir_lower_cooperative_matrix(nir_shader *nir);

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts,
                         bool root_uniform);

VkResult hk_apple9_compile_compute(struct hk_device *dev,
                                   struct vk_shader_compile_info *info,
                                   const struct vk_features *vk_features,
                                   const VkAllocationCallbacks *pAllocator,
                                   struct hk_api_shader **shader_out);

void hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev,
                                nir_shader *nir);

void hk_api_shader_destroy(struct vk_device *vk_dev,
                           struct vk_shader *vk_shader,
                           const VkAllocationCallbacks *pAllocator);

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs);

/* Core pipelines specialize immutable vertex and attachment state. Descriptor
 * contents, draw parameters and blend constants remain GPU data. */
struct hk_apple9_graphics_key {
   struct agx_apple9_vertex_layout vertex;
   struct agx_apple9_blend blend[8];
   uint32_t targets;
   /* Fragment output location written to each color attachment. */
   uint8_t color_map[8];
   /* VK_EXT_depth_clip_control: clip-space depth ranges over [-w, w]. */
   bool clip_negative_one_to_one;
   /* Dynamic depth_clip_negative_one_to_one: read root.draw.clip_z_coeff. */
   bool clip_negative_one_to_one_dynamic;
   /* Decouple fragment color/blend/MSAA/color_map into a linked epilog. */
   bool fs_use_epilog;
   /* Color attachments left VK_FORMAT_UNDEFINED by the pipeline. */
   uint8_t undefined_colors;
   /* sampleShadingEnable: shade and interpolate per sample. */
   bool sample_shading;
   /* The depth/stencil attachment is also read as an input attachment. */
   bool zs_self_dep;
   /* True when a deferred/fallback VS is resolved alongside a separately
    * compiled passthrough FS; prevents stripping generic varyings and separates
    * the cache key from a true depth-only draw. */
   bool has_passthrough_fs;
   /* No fragment-invocation statistics query is active: omit the counter
    * (hk_apple9_lower_fs_statistics). Draw-time variants only; pipelines
    * compiled with their state keep it. */
   bool no_fs_statistics;
   /* stippledLineEnable: lines measure a stipple counter in the fragment
    * shader, see hk_apple9_lower_line_stipple_fs. */
   bool line_stipple;
   /* Line stipple may be enabled statically, dynamically, or via GPL/ESO. */
   bool may_line_stipple;
   /* Polygon mode POINT: triangles may be drawn by the software polygon
    * stage, see hk_apple9_polygon_point_gs. */
   bool polygon_points;
   /* Polygon mode POINT may be enabled statically, dynamically, or via GPL/ESO. */
   bool may_polygon_points;
   /* VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR: independently compiled GPL library. */
   bool is_library;
   /* A passthrough geometry shader rasterizing for a vertex stage that
    * publishes interpolated cull planes, see hk_apple9_compile_vertex_culls.
    * Otherwise geometry shaders publish one flat primitive cull mask. */
   bool gs_cull_planes;
   /* The linked fragment shader reads gl_ClipDistance: a native mesh also
    * links its clip distances as varyings (hk_apple9_native_mesh_options). */
   bool fs_reads_clip_distance;
   /* The linked fragment shader reads gl_PrimitiveID: fused tessellation must
    * export the patch ID via a flat linked mesh varying rather than letting FS
    * bind the rasterizer's generated triangle ID (0x60). */
   bool fs_reads_primitive_id;
   /* Per-device policy; actual use still requires an exact qualified mesh. */
   bool native_index;
   /* Immutable task records feed native mesh children. API task stays compute. */
   bool native_task_bridge;
   /* True native hardware OBJECT -> MESH with <= 16 KiB task payload. */
   bool native_object;
   /* Native OBJECT writes emitted child mesh grid for MESH gl_NumWorkGroups. */
   bool native_object_write_child_grid;
   /* Native OBJECT/MESH read per-parent hardware payload via GET_SR 0xe0/0xe1. */
   bool native_object_hw_payload;
   /* Maximum payload bytes across paired native OBJECT and MESH stages. */
   uint32_t native_object_payload_B;
   /* Generated mesh NIR from an API GS/TES stage (already guarded / own draw ABI). */
   bool native_stage_generated;
   /* Zero-PREGFX fused VS + GS -> native MESH realization. */
   bool native_gs_fused;
   /* Zero-PREGFX fused OBJECT(VS+TCS) -> MESH(TES) realization. */
   bool native_tess_fused;
   /* Pipeline compiles for a multiview render pass (viewMask != 0). */
   bool is_multiview;
   /* Independently keyed stage admission; never implied by a native mesh. */
   uint8_t native_stages;
};

bool hk_apple9_compile_native_mesh_program(
   nir_shader *nir, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason);

VkResult hk_apple9_native_mesh_check(nir_shader *nir,
   const struct hk_apple9_graphics_key *key, bool *supported,
   const char **reason);

VkResult hk_apple9_compile_internal_gs(struct hk_device *dev, nir_shader *nir,
                                       const uint64_t fs_interp[2],
                                       bool cull_mask,
                                       struct hk_api_shader **out);

struct hk_api_shader *hk_meta_shader(struct hk_device *dev,
                                     hk_internal_builder_t builder, void *data,
                                     size_t data_size);

VkResult hk_apple9_create_null_fs(struct hk_device *dev,
   const VkAllocationCallbacks *alloc, bool sample_mask,
   struct hk_api_shader **out);

nir_shader *hk_apple9_build_fs_stats_prolog_nir(void);
