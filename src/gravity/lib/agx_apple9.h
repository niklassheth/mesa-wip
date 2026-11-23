/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_H
#define AGX_APPLE9_H

/* Native 3D texture depth is an 11-bit minus-one field. Array textures have
 * a wider layer count; their range does not apply to 3D sampling.
 */
#define AGX_APPLE9_MAX_3D_TEXTURE_DIMENSION 2048

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
struct agx_ptr;
#include "util/format/u_formats.h"
#include "util/simple_mtx.h"

#include "gravity/lib/agx_apple9_layout.h"
#include "gravity/lib/agx_apple9_launch.h"
#include "gravity/compiler/agx_apple9_profile.h"
#include "gravity/compiler/agx_compile.h"
#include "gravity/compiler/agx_compile_apple9.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct resource table slot size; sampler payloads occupy only eight bytes. */
#define AGX_APPLE9_SAMPLER_TABLE_STRIDE 32

/* Linear sampled textures encode (row_bytes / 16) - 1 in 16 bits. */
#define AGX_APPLE9_MAX_LINEAR_STRIDE (1u << 20)

struct agx_device;
struct agx_bo;
struct agx_shader_body;
struct agx_apple9_graphics;

/* Graphics launch programs of one compiled stage, built on first use for each
 * fragment tile allocation and sample count (vertex stages use 0/0) and then
 * patched per launch. Entries are immutable once published; owned by the
 * compiled shader alongside its BO. */
#define AGX_APPLE9_STAGE_LAUNCH_TEMPLATES 4
struct agx_apple9_stage_launch_cache {
   simple_mtx_t lock;
   /* Published entries; written with release ordering under the lock. */
   unsigned count;
   struct {
      uint8_t tile_bytes, samples;
      struct agx_apple9_launch_template tmpl;
   } entry[AGX_APPLE9_STAGE_LAUNCH_TEMPLATES];
};

void agx_apple9_stage_launch_cache_init(struct agx_apple9_stage_launch_cache *c);
void agx_apple9_stage_launch_cache_finish(struct agx_apple9_stage_launch_cache *c);

struct agx_apple9_render_stage {
   /* Stable compiled-object identity; zero for externally supplied stages. */
   uint64_t program_id;
   /* Owned by the compiled shader; each using batch retains a BO reference. */
   struct agx_bo *bo;
   /* Optional body identity; whole-BO callers leave this NULL. */
   struct agx_shader_body *body;
   /* Optional preamble base VA when `body` is a linked stub whose `addr` is
    * the prolog rather than the main body that owns the preamble. */
   uint64_t preamble_base;
   const uint8_t *binary;
   size_t binary_size;
   uint32_t preamble_offset, preamble_size;
   uint32_t ubo_mask;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
   uint16_t publication_count;
   bool publication_count_valid;
   /* Native mesh uses the same raster interface/PPP serializer but supplies
    * object/mesh launches through the mesh command packet. */
   bool native_mesh;
   uint32_t resource_ssbo_mask, resource_write_mask;
   uint8_t resource_count;
   uint8_t resource_binding[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint32_t texture_mask, sampler_mask, image_mask;
   bool uses_texel_fetch;
   bool uses_discard;
   bool writes_depth;
   bool reads_tile;
   bool disable_tri_merging;
   /* Pixel interlock: order fragments per pixel (fragment control "Pixel
    * order") instead of per covered sample. */
   bool pixel_interlock;

   /* Scalar interface counts used by the bounded Apple9 stage linker. */
   uint8_t position_components;
   uint8_t varying_components;
   struct agx_apple9_varying_layout varyings;
   struct agx_apple9_interp_mask apple9_linear_mask, apple9_flat_mask;
   bool apple9_reads_z;
   bool reads_primitive_id;
   bool writes_point_size;
   bool writes_layer_viewport;
   uint8_t clip_distance_count;
   bool reads_point_coord;
   /* Coefficients after the point coordinates: barycentric weights (bit 0
    * perspective, bit 1 linear) and the varying scalars read per vertex. */
   uint8_t barycentric_mask;
   uint8_t per_vertex_count;
   uint8_t per_vertex[AGX_APPLE9_MAX_PER_VERTEX];
   uint8_t render_targets;
   /* Optional per-shader launch templates; NULL builds every launch. */
   struct agx_apple9_stage_launch_cache *launch_cache;
   /* Fragment-owned interpolation tables for each provoking vertex. */
   uint32_t coefficients[3];
   uint16_t coefficient_count;
   /* Optional linked FS prolog / epilog extra CF bindings:
    * - prolog_cull_count > 0: bind cull-distance varying(s) from vs_varyings
    *   at CF slot `prolog_cull_cf_base` (1 flat if prolog_cull_mask, else
    *   prolog_cull_count linear scalars).
    * - prolog_line_stipple: bind TEX6.xyz flat + TEX7.xy linear at
    *   `prolog_stipple_cf_base + 0..4`.
    * - epilog_z_cf_index > 0: bind FRAGCOORD_Z (0x12c) at `epilog_z_cf_index`
    *   when fs_main did not read Z itself (late_zs epilog). */
   uint8_t prolog_cull_count;
   bool prolog_cull_mask;
   uint8_t prolog_cull_cf_base;
   bool prolog_line_stipple;
   uint8_t prolog_stipple_cf_base;
   uint8_t epilog_z_cf_index;
};

/* Whether a linked (vertex, fragment) pair requires hardware-generated
 * primitive ID (PPP group+0x68 bit 12, VDM bit 6, CF 0x60). True either when
 * fragment set legacy reads_primitive_id or when an independent fragment reads
 * VARYING_SLOT_PRIMITIVE_ID and the producer does not export it. */
static inline bool
agx_apple9_pipeline_reads_hw_prim_id(
   const struct agx_apple9_render_stage *vertex,
   const struct agx_apple9_render_stage *fragment)
{
   return fragment->reads_primitive_id ||
          ((fragment->varyings.mask[VARYING_SLOT_PRIMITIVE_ID] & 1) &&
           !(vertex->varyings.mask[VARYING_SLOT_PRIMITIVE_ID] & 1));
}

struct agx_apple9_render_pipeline {
   /* Vertex fetch is lowered into the API vertex program. The stages belong
    * to their compiled shaders, which outlive every draw using them. */
   const struct agx_apple9_render_stage *vertex;
   const struct agx_apple9_render_stage *fragment;

   /* Color surfaces in draw-buffer order; absent attachments have address 0. */
   uint64_t color_targets[8];
   enum pipe_format color_formats[8];
   uint8_t samples;

   /* The first AGX_APPLE9_PPP_STATIC_SIZE bytes of the PPP record that depend
    * only on the linked vertex and fragment stages (agx_apple9_build_ppp_static),
    * built once per link; NULL builds them in every draw. */
   const uint8_t *ppp_static;

   /* Batch-owned PPP record, relative to the render-context aperture. */
   uint32_t ppp;
   /* Viewport block within the PPP record and its length in words. Zero
    * selects the single-viewport block at 0xc0.
    */
   uint16_t ppp_viewport_offset;
   uint16_t ppp_viewport_words;

   /* Apple9 VDM linkage words produced by the bounded pipeline linker. */
   uint32_t pipeline_word;
   uint32_t vertex_launch;
   uint32_t vertex_state_class;

   uint64_t index_buffer;
   uint32_t index_extent;
   uint8_t index_size;
   uint8_t primitive;
   bool primitive_restart;
   bool flatshade_first;
   uint32_t restart_index;

};

#define AGX_APPLE9_MAX_VIEWPORTS 16
/* Fixed state, then a viewport block of 8n + 2 words at 0x100. */
#define AGX_APPLE9_PPP_RECORD_MAX 0x340

/* CPU snapshot used while encoding a draw and for adjacent state reuse.
 * Encoded allocations belong to the batch's growing pools. */
struct agx_apple9_uniform_draw {
   uint64_t code[2];
   uint32_t launch[2], ppp, coefficients;
   uint64_t program_id[2];
   uint16_t tile_bytes, cf_count;
   uint8_t samples;
   /* CPU copy for bounded record reuse; never read back write-combined BOs.
    * Records with several viewports append the viewport block after the
    * fixed 256-byte state.
    */
   uint8_t ppp_record[AGX_APPLE9_PPP_RECORD_MAX];
   uint16_t ppp_size, ppp_viewport_offset, ppp_viewport_words;
   uint64_t buffers[2][AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint8_t buffer_count[2];
   uint64_t vertex_table;
   uint64_t fragment_table;
   /* Graphics stage indices: vertex = 0, fragment = 1. */
   uint64_t texture_table[2], sampler_table[2];
   uint32_t depth_control, depth_face[2], stencil[2];
   uint32_t raster_control;
   uint8_t object_type;
   float viewport_translate[3], viewport_scale[3];
   /* Viewports after the first, as {tx, sx, ty, sy, tz, sz}; zero or one
    * viewport_count selects the single viewport above. Scissors for the
    * additional viewports follow scissor_index consecutively.
    */
   uint8_t viewport_count;
   float viewports[AGX_APPLE9_MAX_VIEWPORTS - 1][6];
   uint32_t scissor_index;
   uint16_t depth_bias_index;
   uint16_t occlusion_index;
   uint8_t visibility_mode;
   uint16_t scissor_min[2], scissor_max[2];
   bool reads_tile;
   bool uses_discard;
   bool writes_depth;
   bool disable_tri_merging;
   bool pixel_interlock;
   bool flatshade_first;
   uint8_t provoking_vertex;
};

/* Clear a record for reuse without touching its large arrays:
 * agx_apple9_prepare_draw_alloc rebuilds ppp_record, and buffers and
 * viewports are read only up to the counts the caller fills in. */
static inline void
agx_apple9_uniform_draw_reset(struct agx_apple9_uniform_draw *draw)
{
   memset(draw, 0, offsetof(struct agx_apple9_uniform_draw, ppp_record));
   memset(&draw->ppp_size, 0,
          offsetof(struct agx_apple9_uniform_draw, buffers) -
             offsetof(struct agx_apple9_uniform_draw, ppp_size));
   memset(&draw->buffer_count, 0,
          offsetof(struct agx_apple9_uniform_draw, viewports) -
             offsetof(struct agx_apple9_uniform_draw, buffer_count));
   memset(&draw->scissor_index, 0,
          sizeof(*draw) -
             offsetof(struct agx_apple9_uniform_draw, scissor_index));
}

void agx_apple9_initialize_entries(struct agx_device *dev);

/* Per-sample tile allocation for a fragment-stage launch or render. As in
 * agx_build_tilebuffer_layout, multisampled renders keep a nonempty
 * allocation even without colour targets: a large G16 2x depth-only render
 * with zero-byte fragment launches can leave its fragment stage unretired.
 */
static inline unsigned
agx_apple9_tile_sample_bytes(unsigned color_bytes, unsigned samples)
{
   return color_bytes || samples <= 1 ? color_bytes : 8;
}

/* Prepare one native stage independently, including background/EOT helpers.
 * The caller retains the shader BO and immutable allocation until completion.
 */
bool agx_apple9_prepare_graphics_stage(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   const struct agx_apple9_render_stage *shader, bool fragment, uint64_t table,
   uint64_t textures, uint64_t samplers, unsigned tile_bytes, unsigned samples,
   uint32_t *launch);

/* Whether a draw can take the previous draw's launch for stage (0 vertex, 1
 * fragment) or its coefficient table. Both need nonzero, equal program ids:
 * a stage without an id is never reused. draw->program_id, code, tables,
 * samples, tile_bytes and provoking_vertex must already be filled in. */
static inline bool
agx_apple9_draw_reuses_launch(const struct agx_apple9_uniform_draw *previous,
                              const struct agx_apple9_uniform_draw *draw,
                              unsigned stage)
{
   return previous && draw->program_id[stage] &&
          previous->program_id[stage] == draw->program_id[stage] &&
          previous->code[stage] == draw->code[stage] &&
          previous->texture_table[stage] == draw->texture_table[stage] &&
          previous->sampler_table[stage] == draw->sampler_table[stage] &&
          (stage ? previous->fragment_table == draw->fragment_table
                 : previous->vertex_table == draw->vertex_table) &&
          (!stage || (previous->samples == draw->samples &&
                      previous->tile_bytes == draw->tile_bytes));
}

static inline bool
agx_apple9_draw_reuses_coefficients(const struct agx_apple9_uniform_draw *previous,
                                    const struct agx_apple9_uniform_draw *draw)
{
   return previous && draw->program_id[0] && draw->program_id[1] &&
          previous->program_id[0] == draw->program_id[0] &&
          previous->program_id[1] == draw->program_id[1] &&
          previous->provoking_vertex == draw->provoking_vertex;
}

/* The caller owns both immutable arenas. Context allocations must lie wholly
 * within the render-context aperture; USC allocations use the shader aperture.
 * A false return leaves the draw unpublished, including on allocation failure.
 */
#define AGX_APPLE9_CF_BINDINGS_SIZE   0x2e0
#define AGX_APPLE9_CF_BINDINGS_STRIDE 0x300
#define AGX_APPLE9_CF_ALLOCATION_SIZE (3 * AGX_APPLE9_CF_BINDINGS_STRIDE)
unsigned
agx_apple9_coefficients_size(const struct agx_apple9_render_stage *fragment);
void agx_apple9_init_coefficients(struct agx_device *dev,
                                  struct agx_apple9_render_stage *fragment,
                                  struct agx_ptr allocation);

/* Build a 2-stage (`third == 0`: call first, tail-jump second) or 3-stage
 * (`third != 0`: call first, call second, tail-jump third) linked entry stub
 * into a 64-byte entry code window. Returns false on invalid addresses or
 * out-of-range displacements. */
bool agx_apple9_build_linked_entry(uint8_t out[64], uint64_t entry,
                                   uint64_t first, uint64_t second,
                                   uint64_t third);

/* Byte capacity for a 4-stage linked body stub (call first, call second,
 * call third, tail-jump fourth = 66 bytes, rounded up for alignment). */
#define AGX_APPLE9_LINKED_STUB_4_MAX_BYTES 80
size_t agx_apple9_build_linked_stub_4(uint8_t *out, size_t capacity,
                                      uint64_t code_va, uint64_t first,
                                      uint64_t second, uint64_t third,
                                      uint64_t fourth);

/* Maximum byte length of a per-sample trampoline built by
 * agx_apple9_build_per_sample_trampoline[_prologs] (4 samples + up to 2
 * prologs + tile + ZS = 256 bytes). */
#define AGX_APPLE9_PER_SAMPLE_TRAMPOLINE_MAX_BYTES 256

/* Build an unrolled per-sample fragment trampoline (samples in {1, 2, 4}) at
 * `code_va`. For each sample s = 0..samples-1, loads (1 << s) into
 * AGX_APPLE9_SAMPLE_MASK_ABI_REG (r63), calls `main_addr`, reloads (1 << s)
 * into r63, and calls `epilog_addr`. Runs optional `prolog_addr` (and
 * `prolog2_addr` in the `_prologs` variant) once before the loop, opens
 * `tile_access_mode` (0 = none, 0x600, or 0x700) + 0x80c when
 * `emit_tile_fence` is set, and emits TILE_FENCE 1 (`emit_zs_release`) and
 * TILE_FENCE 0x020c (`emit_tile_fence`) once after the final sample.
 * Returns the exact emitted byte count, or 0 on error. */
size_t agx_apple9_build_per_sample_trampoline_prologs(
   uint8_t *out, size_t capacity, uint64_t code_va, uint8_t samples,
   uint64_t prolog1_addr, uint64_t prolog2_addr, uint64_t main_addr,
   uint64_t epilog_addr, uint16_t tile_access_mode, bool emit_zs_release,
   bool emit_tile_fence);
size_t agx_apple9_build_per_sample_trampoline(
   uint8_t *out, size_t capacity, uint64_t code_va, uint8_t samples,
   uint64_t prolog_addr, uint64_t main_addr, uint64_t epilog_addr,
   uint16_t tile_access_mode, bool emit_zs_release, bool emit_tile_fence);

#define AGX_APPLE9_PPP_STATIC_SIZE 0xc0
void agx_apple9_build_ppp_static(const struct agx_apple9_render_pipeline *pipeline,
                                 uint8_t out[AGX_APPLE9_PPP_STATIC_SIZE]);

bool agx_apple9_prepare_draw_alloc(
   struct agx_device *dev, void *usc_allocator,
   struct agx_ptr (*usc_alloc)(void *, size_t, size_t), void *context_allocator,
   struct agx_ptr (*context_alloc)(void *, size_t, size_t),
   const struct agx_apple9_render_pipeline *pipeline,
   struct agx_apple9_uniform_draw *draw,
   const struct agx_apple9_uniform_draw *previous);

/* Update only resource-table bindings of an already prepared draw. All shader,
 * texture/sampler, interpolation and raster state must remain unchanged.
 * GPU allocations remain immutable. Failure leaves the CPU draw unchanged.
 */
bool agx_apple9_update_draw_tables_alloc(
   struct agx_device *dev, void *usc_allocator,
   struct agx_ptr (*usc_alloc)(void *, size_t, size_t), void *context_allocator,
   struct agx_ptr (*context_alloc)(void *, size_t, size_t),
   const struct agx_apple9_render_pipeline *pipeline,
   struct agx_apple9_uniform_draw *draw, const uint64_t tables[2]);

#define AGX_APPLE9_COMPUTE_RESOURCE_STRIDE 0x20u
#define AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE 0x100u
#define AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET 0xc0u
#define AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE 0x2cu
#define AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE 0x28u
#define AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE 0x1cu
#define AGX_APPLE9_RENDER_CONTEXT_BASE UINT64_C(0x1000000000)

size_t agx_apple9_compute_launch_size(
   const struct agx_apple9_compute_profile *profile);

enum agx_apple9_compute_geometry_mode {
   AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
   AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT,
};

struct agx_apple9_compute_geometry {
   enum agx_apple9_compute_geometry_mode mode;
   union {
      uint32_t threads[3];
      uint64_t group_counts;
   };
   uint32_t local[3];
};

/* Populate the group-count pointer and inline counts in a resource record. */
bool agx_apple9_build_compute_geometry_fields(
   void *record, size_t record_size, uint64_t record_address,
   const struct agx_apple9_compute_geometry *geometry);

/* Build immutable resource and launch records in the batch's USC pool. */
bool agx_apple9_prepare_compute_dispatch_alloc(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   struct agx_shader_body *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address);

/* As agx_apple9_prepare_compute_dispatch_alloc, emitting the launch program
 * from tmpl (agx_apple9_compute_launch_template of the same profile and
 * preamble) when it is non-NULL and valid. HK_APPLE9_LAUNCH_CHECK=1 compares
 * every templated program with a full build. */
bool agx_apple9_prepare_compute_dispatch_alloc_tmpl(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   struct agx_shader_body *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, const struct agx_apple9_launch_template *tmpl,
   uint64_t *launch_address);

/* Build a compute pipeline's launch template, once at pipeline creation. */
bool agx_apple9_compute_launch_template(
   struct agx_apple9_launch_template *tmpl,
   const struct agx_apple9_compute_profile *profile, uint64_t usc_exec_base,
   uint64_t preamble_address);

/* Record-size constants include the optional trailing CDM barrier. Without
 * it, the emitted launch is one uint32_t shorter; the caller supplies the next
 * dependency barrier or a final stream barrier.
 */
/* Why the emitters below would reject a dispatch, or NULL if they accept it.
 * global is unused for indirect dispatches. */
const char *agx_apple9_dispatch_reject_reason(
   uint64_t launch, bool indirect, uint64_t indirect_address,
   const uint32_t global[3], const uint32_t local[3],
   const struct agx_apple9_compute_profile *profile);

bool agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile,
   bool barrier);

bool agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   bool indirect_local, const struct agx_apple9_compute_profile *profile,
   bool barrier);

/* Replace the trailing barrier of a dispatch record emitted with one by the
 * compute-only barrier (0x160 in the barrier bits): it orders the following
 * dispatches and makes this one's writes visible to them, but doesn't drain.
 * Consumers outside compute, and indirect grid fetches, still need the full
 * barrier before them.
 */
void agx_apple9_set_compute_barrier(void *record, unsigned record_size);

/* Translate common image/sampler state to native Apple9 table records. */
struct agx_texture_packed;
struct agx_sampler_packed;
struct AGX_SAMPLER;
struct AGX_TEXTURE;
bool agx_apple9_pack_texture(void *out, const struct AGX_TEXTURE *state);
void agx_apple9_set_null_texture(void *out);
void agx_apple9_set_null_pbe(void *out);
bool agx_apple9_pack_sampler_state(void *out, const struct AGX_SAMPLER *state);
bool agx_apple9_translate_texture(void *out,
                                  const struct agx_texture_packed *state);
struct agx_pbe_packed;
struct ail_layout;
/* layer_base is the selected first layer's level-zero address. The native
 * descriptor retains the allocation's mip tree and selects level separately.
 */
bool agx_apple9_translate_pbe(void *out, const struct agx_pbe_packed *state,
                            enum pipe_format format, const struct ail_layout *layout,
                            uint64_t layer_base, unsigned level,
                            unsigned layer_count, bool layered,
                            unsigned dimension);

void agx_apple9_pack_sampler(void *out, bool min_linear, bool mag_linear,
                              unsigned mip_filter, float min_lod, float max_lod,
                         unsigned wrap_s, unsigned wrap_t, unsigned max_anisotropy);
void agx_apple9_pack_fetch_sampler(void *out);

/*
 * Apple9 command and shader ABIs are intentionally kept outside the older
 * genxml path.  G16 and G17 share the Apple9 core shader ISA with each other;
 * neither is treated as the incompatible Apple8 ISA used by G13/G14.  Their
 * render packet and compiler-container ABIs also need Apple9-specific
 * handling.  This first encoder is deliberately narrow: it describes the
 * direct triangle lists, with optional 16- or 32-bit indices.
 */
bool
agx_apple9_link_render_pipeline(
   struct agx_apple9_render_pipeline *pipeline,
   const struct agx_apple9_render_stage *vertex,
   const struct agx_apple9_render_stage *fragment);

bool agx_apple9_link_render_pipeline_with_prolog(
   struct agx_apple9_render_pipeline *pipeline,
   const struct agx_apple9_render_stage *vertex_prolog,
   const struct agx_apple9_render_stage *vertex,
   const struct agx_apple9_render_stage *fragment);

size_t
agx_apple9_direct_draw_size(const struct agx_apple9_render_pipeline *pipeline);

uint8_t *agx_apple9_emit_direct_draw(
   uint8_t *out, const struct agx_apple9_render_pipeline *pipeline,
   unsigned vertex_count, unsigned instance_count, unsigned vertex_start);

uint8_t *agx_apple9_emit_indirect_draw(
   uint8_t *out, const struct agx_apple9_render_pipeline *pipeline,
   uint64_t indirect);

#ifdef __cplusplus
}
#endif

#endif
