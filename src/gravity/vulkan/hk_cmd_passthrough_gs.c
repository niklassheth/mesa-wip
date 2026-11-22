/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/compiler/agx_nir.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "gravity/lib/agx_apple9.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9_polygon_points.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_feedback.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"
#include "hk_shader_priv.h"

#include "gravity/genxml/agx_pack.h"
#include "gravity/libcl/compression.h"
#include "gravity/libcl/libagx.h"
#include "gravity/libcl/mesh.h"
#include "gravity/libcl/query.h"
#include "gravity/libcl/tessellator.h"
#include "poly/geometry.h"
#include "util/u_debug.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

#include "hk_cmd_draw_priv.h"

/* The Apple9 passthrough geometry shader key is the common key followed by
 * the fragment shader's {flat, linear} input masks, which shape the
 * rasterized varying layout, and the per-slot component masks of the
 * fragment shader's compacted input layout.
 */
struct hk_apple9_passthrough_suffix {
   uint64_t fs_interp[2];
   uint8_t components[64];
   /* Flat copies of per-vertex inputs, as in the fragment shader's info. */
   uint8_t per_vertex_copies;
   uint8_t per_vertex_location[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_copy_base[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_bary[2];
   /* Line stipple start slot to continue along strips, and whether the
    * input is a strip. */
   uint8_t line_stipple_start;
   uint8_t line_stipple_strip;
   /* Software polygon points (hk_apple9_polygon_point_gs), and the varying
    * slot of the polygon's facing when the fragment shader reads it. */
   uint8_t polygon_points;
   uint8_t polygon_face_slot;
   /* polygonModePointSize: forward the shader's PointSize, else size 1. */
   uint8_t polygon_point_size;
   /* The vertex stage publishes a flat cull mask, which the fragment shader
    * reads: combine it over each primitive. Otherwise cull planes stay
    * interpolated values. See hk_apple9_compile_vertex_culls. */
   uint8_t cull_mask;
};

/* Before every emitted vertex, write the per-vertex values of the input
 * primitive's API vertices 0, 1 and 2 to their flat copies, and the emitted
 * API vertex's one-hot barycentric weights. Points repeat their vertex and
 * lines their second vertex. The passthrough GS emits API vertex i as its
 * i-th vertex.
 */

static inline unsigned
hk_ptgs_per_vertex_copy_slot(uint8_t base, unsigned vertex)
{
   if (base == VARYING_SLOT_BFC0 && vertex == 2)
      return VARYING_SLOT_CLIP_VERTEX;
   return base + vertex;
}

static void
hk_apple9_write_per_vertex_copies(nir_shader *s,
                                  const struct hk_apple9_passthrough_suffix *suffix)
{
   enum mesa_prim prim = s->info.gs.input_primitive;
   unsigned start = prim == MESA_PRIM_LINES_ADJACENCY ? 1 : 0;
   unsigned step = prim == MESA_PRIM_TRIANGLES_ADJACENCY ? 2 : 1;
   unsigned vertices = mesa_vertices_per_prim(u_decomposed_prim(u_reduced_prim(prim)));
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);
   unsigned emitted = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_emit_vertex)
            continue;
         b.cursor = nir_before_instr(instr);
         for (unsigned m = 0; m < 2; ++m) {
            if (!suffix->per_vertex_bary[m])
               continue;
            unsigned slot = suffix->per_vertex_bary[m];
            for (unsigned c = 0; c < 3; ++c) {
               nir_store_output(&b, nir_imm_float(&b, c == emitted ? 1.0 : 0.0),
                                nir_imm_int(&b, 0),
                                .io_semantics = {.location = slot, .num_slots = 1},
                                .component = c, .src_type = nir_type_float32,
                                .write_mask = 1);
            }
            s->info.outputs_written |= BITFIELD64_BIT(slot);
         }
         ++emitted;
         for (unsigned j = 0; j < suffix->per_vertex_copies; ++j) {
            for (unsigned k = 0; k < 3; ++k) {
               unsigned api_vertex = MIN2(k, vertices - 1);
               nir_def *value = nir_load_per_vertex_input(
                  &b, 4, 32, nir_imm_int(&b, start + api_vertex * step),
                  nir_imm_int(&b, 0),
                  .io_semantics = {
                     .location = VARYING_SLOT_VAR0 + suffix->per_vertex_location[j],
                     .num_slots = 1,
                  },
                  .dest_type = nir_type_uint32);
               unsigned copy =
                  hk_ptgs_per_vertex_copy_slot(suffix->per_vertex_copy_base[j], k);
               for (unsigned c = 0; c < 4; ++c) {
                  nir_store_output(&b, nir_channel(&b, value, c), nir_imm_int(&b, 0),
                                   .io_semantics = {.location = copy, .num_slots = 1},
                                   .component = c, .src_type = nir_type_uint32,
                                   .write_mask = 1);
               }
               s->info.outputs_written |= BITFIELD64_BIT(copy);
            }
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

static nir_def *
hk_apple9_stipple_window_from_pos(nir_builder *b, nir_def *pos,
                                  nir_def *viewport)
{
   /* Vertices behind the eye clip the segment; keep the divide finite. */
   nir_def *w = nir_fmax(b, nir_channel(b, pos, 3), nir_imm_float(b, 0x1p-16));
   nir_def *ndc = nir_fdiv(b, nir_trim_vector(b, pos, 2), w);
   nir_def *index =
      viewport
         ? nir_umin(b, viewport,
                    nir_imm_int(b, AGX_APPLE9_VIEWPORT_TABLE_COUNT - 1))
         : nir_imm_int(b, 0);
   nir_def *transform = nir_load_ubo(
      b, 4, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_iadd_imm(b, nir_imul_imm(b, index, 16),
                   AGX_APPLE9_VIEWPORT_TABLE_OFFSET),
      .align_mul = 16, .range_base = AGX_APPLE9_VIEWPORT_TABLE_OFFSET,
      .range = AGX_APPLE9_VIEWPORT_TABLE_COUNT * 16);
   return agx_nir_line_stipple_snap(
      b, nir_ffma(b, ndc, nir_channels(b, transform, 0x3),
                  nir_channels(b, transform, 0xc)));
}

/*
 * Line stipple computed directly from VARYING_SLOT_POS (and VIEWPORT if
 * written) in the SW-VS/SW-TES output buffer, emitting flat start + counter
 * to VARYING_SLOT_TEX6 and noperspective vertex window position to
 * VARYING_SLOT_TEX7 with zero VS/TES modification. Every emitted vertex of
 * the segment gets the segment's first vertex, so either vertex may provoke.
 * On a strip the counter continues along earlier segments. With adjacency,
 * the line is input vertices 1 and 2, and strip vertex j + 1 begins segment j.
 */
static void
hk_apple9_line_stipple_gs(nir_shader *s,
                          const struct poly_passthrough_gs_key *common,
                          const struct hk_apple9_passthrough_suffix *suffix)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   const nir_io_semantics pos_sem = {.location = VARYING_SLOT_POS, .num_slots = 1};
   const nir_io_semantics vp_sem = {.location = VARYING_SLOT_VIEWPORT, .num_slots = 1};
   const nir_io_semantics start_sem = {.location = VARYING_SLOT_TEX6, .num_slots = 1};
   const nir_io_semantics axis_sem = {.location = VARYING_SLOT_TEX7, .num_slots = 1};
   bool has_vp = (common->outputs & VARYING_BIT_VIEWPORT) != 0;
   unsigned first = s->info.gs.input_primitive == MESA_PRIM_LINES_ADJACENCY;
   nir_def *win[2];
   for (unsigned k = 0; k < 2; ++k) {
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, nir_imm_int(&b, first + k), nir_imm_int(&b, 0),
         .io_semantics = pos_sem, .dest_type = nir_type_float32);
      nir_def *vp = has_vp
         ? nir_load_per_vertex_input(
              &b, 1, 32, nir_imm_int(&b, first + k), nir_imm_int(&b, 0),
              .io_semantics = vp_sem, .dest_type = nir_type_uint32)
         : NULL;
      win[k] = hk_apple9_stipple_window_from_pos(&b, pos, vp);
   }
   nir_def *start = win[0];
   nir_def *base = nir_imm_float(&b, 0.0);

   if (suffix->line_stipple_strip) {
      nir_variable *i = nir_local_variable_create(impl, glsl_uint_type(), NULL);
      nir_variable *sum = nir_local_variable_create(impl, glsl_float_type(), NULL);
      nir_store_var(&b, i, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum, nir_imm_float(&b, 0.0), 1);
      nir_def *prim = nir_load_primitive_id(&b);
      nir_def *limit =
         nir_bcsel(&b, nir_ult_imm(&b, prim, AGX_LINE_STIPPLE_MAX_STRIP_SEGMENTS),
                   prim, nir_imm_int(&b, 0));
      nir_push_loop(&b);
      {
         nir_def *j = nir_load_var(&b, i);
         nir_break_if(&b, nir_uge(&b, j, limit));
         nir_def *pos_a = nir_load_strip_output_poly(
            &b, 4, nir_iadd_imm(&b, j, first), .io_semantics = pos_sem);
         nir_def *vp_a = has_vp
            ? nir_load_strip_output_poly(
                 &b, 1, nir_iadd_imm(&b, j, first), .io_semantics = vp_sem)
            : NULL;
         nir_def *a = hk_apple9_stipple_window_from_pos(&b, pos_a, vp_a);
         nir_def *pos_c = nir_load_strip_output_poly(
            &b, 4, nir_iadd_imm(&b, j, first + 1), .io_semantics = pos_sem);
         nir_def *vp_c = has_vp
            ? nir_load_strip_output_poly(
                 &b, 1, nir_iadd_imm(&b, j, first + 1), .io_semantics = vp_sem)
            : NULL;
         nir_def *c = hk_apple9_stipple_window_from_pos(&b, pos_c, vp_c);
         nir_def *d = nir_fabs(&b, nir_fsub(&b, c, a));
         nir_store_var(&b, sum,
                       nir_fadd(&b, nir_load_var(&b, sum),
                                nir_fmax(&b, nir_channel(&b, d, 0),
                                         nir_channel(&b, d, 1))),
                       1);
         nir_store_var(&b, i, nir_iadd_imm(&b, j, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      base = nir_load_var(&b, sum);
   }

   unsigned emitted = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_emit_vertex)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_store_output(&b, nir_vec3(&b, nir_channel(&b, start, 0),
                                       nir_channel(&b, start, 1), base),
                          nir_imm_int(&b, 0), .io_semantics = start_sem,
                          .write_mask = 0x7, .src_type = nir_type_float32);
         nir_store_output(&b, win[MIN2(emitted, 1)], nir_imm_int(&b, 0),
                          .io_semantics = axis_sem, .write_mask = 0x3,
                          .src_type = nir_type_float32);
         ++emitted;
      }
   }
   s->info.outputs_written |= VARYING_BIT_TEX6 | VARYING_BIT_TEX7;
   nir_progress(true, impl, nir_metadata_none);
}

static bool
hk_apple9_trim_passthrough_output(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data)
{
   const uint8_t *components = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned slot = nir_intrinsic_io_semantics(intr).location;
   if (slot < VARYING_SLOT_VAR0 || slot >= 64 ||
       (components[slot] & BITFIELD_BIT(nir_intrinsic_component(intr))))
      return false;
   /* The fragment shader was linked against a producer that did not write
    * this component, so the rasterized layout must not contain it. */
   nir_instr_remove(&intr->instr);
   return true;
}

struct hk_pp_gs_load {
   unsigned step;
};

static nir_def *
hk_pp_gs_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
              unsigned offset, unsigned component, unsigned comps,
              nir_alu_type type)
{
   const struct hk_pp_gs_load *l = data;
   unsigned num_slots = 1;
   if (location == VARYING_SLOT_CLIP_DIST0)
      num_slots = b->shader->info.clip_distance_array_size +
                  b->shader->info.cull_distance_array_size;
   return nir_load_per_vertex_input(
      b, comps, 32, nir_imm_int(b, vertex * l->step), nir_imm_int(b, offset),
      .component = component, .dest_type = type,
      .io_semantics = {.location = location, .num_slots = num_slots});
}

static bool
hk_ptgs_split_pp_face_prim(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_FOGC ||
       intr->src[0].ssa->num_components < 2)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *val = intr->src[0].ssa;
   nir_store_output(b, nir_channel(b, val, 0), nir_imm_int(b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_FOGC,
                                     .num_slots = 1});
   nir_store_output(b, nir_channel(b, val, 1), nir_imm_int(b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID,
                                     .num_slots = 1});
   nir_instr_remove(&intr->instr);
   return true;
}

/*
 * Polygon mode POINT for vertex and evaluation shader pipelines: a generated
 * geometry shader draws each input triangle's clipped vertices as points,
 * writing facing to VARYING_SLOT_FOGC and the polygon's primitive ID to
 * VARYING_SLOT_PRIMITIVE_ID (see hk_apple9_pp_emit_triangle).
 */
static void
hk_apple9_polygon_point_gs(nir_builder *b,
                           const struct poly_passthrough_gs_key *key,
                           const struct hk_apple9_passthrough_suffix *suffix)
{
   nir_shader *s = b->shader;
   assert(u_reduced_prim(key->prim) == MESA_PRIM_TRIANGLES);
   unsigned clip = key->clip_distance_array_size;
   unsigned cull = key->cull_distance_array_size;
   uint64_t outputs = key->outputs &
      ~(VARYING_BIT_BOUNDING_BOX0 | VARYING_BIT_BOUNDING_BOX1 |
        VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
        VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 |
        VARYING_BIT_PRIMITIVE_ID);
   /* Without polygonModePointSize, polygon points have size 1: the point
    * path's default. */
   if (!suffix->polygon_point_size)
      outputs &= ~VARYING_BIT_PSIZ;

   s->info.stage = MESA_SHADER_GEOMETRY;
   s->info.inputs_read = key->outputs;
   s->info.outputs_written = (key->outputs &
      ~(VARYING_BIT_BOUNDING_BOX0 | VARYING_BIT_BOUNDING_BOX1)) |
      VARYING_BIT_FOGC | VARYING_BIT_PRIMITIVE_ID;
   if (!suffix->polygon_point_size)
      s->info.outputs_written &= ~VARYING_BIT_PSIZ;
   s->info.clip_distance_array_size = clip;
   s->info.cull_distance_array_size = cull;
   s->info.gs.input_primitive = key->prim;
   s->info.gs.output_primitive = MESA_PRIM_POINTS;
   s->info.gs.vertices_in = mesa_vertices_per_prim(key->prim);
   s->info.gs.vertices_out = HK_PP_MAX_VERTICES;
   s->info.gs.invocations = 1;
   s->info.gs.active_stream_mask = 1;

   /* Adjacency triangles use input vertices 0, 2 and 4. */
   struct hk_pp_gs_load l = {
      .step = key->prim == MESA_PRIM_TRIANGLES_ADJACENCY ? 2 : 1,
   };
   struct hk_pp_info info = {
      .clip = clip,
      .cull = cull,
      .outputs = outputs,
      .output_components = key->output_components,
      .flat = suffix->fs_interp[0],
      .face_slot = VARYING_SLOT_FOGC,
   };
   nir_def *last = nir_ine_imm(b, nir_load_provoking_last(b), 0);
   nir_def *provoking = nir_bcsel(b, last, nir_imm_int(b, 2), nir_imm_int(b, 0));
   hk_apple9_pp_emit_triangle(b, &info, hk_pp_gs_load, &l, provoking,
                              nir_load_primitive_id(b), NULL);
   nir_shader_intrinsics_pass(s, hk_ptgs_split_pp_face_prim,
                              nir_metadata_control_flow, NULL);
}

static void
hk_apple9_passthrough_gs(struct nir_builder *b, const void *key)
{
   {
      const struct poly_passthrough_gs_key *common = key;
      struct hk_apple9_passthrough_suffix suffix;
      memcpy(&suffix, (const uint8_t *)key +
                         poly_passthrough_gs_key_size(common->xfb_info.output_count),
             sizeof(suffix));
      if (suffix.polygon_points) {
         hk_apple9_polygon_point_gs(b, common, &suffix);
         nir_shader_intrinsics_pass(b->shader, hk_apple9_trim_passthrough_output,
                                    nir_metadata_control_flow, suffix.components);
         return;
      }
   }
   poly_nir_passthrough_gs(b, key);
   const struct poly_passthrough_gs_key *common = key;
   size_t key_size =
      poly_passthrough_gs_key_size(common->xfb_info.output_count);
   struct hk_apple9_passthrough_suffix suffix;
   memcpy(&suffix, (const uint8_t *)key + key_size, sizeof(suffix));
   if (suffix.per_vertex_copies || suffix.per_vertex_bary[0] ||
       suffix.per_vertex_bary[1])
      hk_apple9_write_per_vertex_copies(b->shader, &suffix);
   if (suffix.line_stipple_start)
      hk_apple9_line_stipple_gs(b->shader, common, &suffix);
   nir_shader_intrinsics_pass(b->shader, hk_apple9_trim_passthrough_output,
                              nir_metadata_control_flow, suffix.components);
}

struct hk_apple9_mesh_rast_key {
   struct hk_mesh_info info;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t line_stipple;
   uint8_t polygon_points;
   uint8_t point_size;
   uint8_t _pad[3];
   uint64_t fs_interp[2];
   uint8_t per_vertex_copies;
   uint8_t per_vertex_location[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_copy_base[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_bary[2];
   uint8_t _pad2;
};

static void
hk_apple9_mesh_rast_builder(UNUSED struct nir_builder *b, UNUSED const void *key)
{
   /* Handled directly in hk_build_meta_shader_locked because hk_nir_mesh_rast
    * constructs a vertex shader rather than mutating a compute builder. */
   UNREACHABLE("built via hk_build_meta_shader_locked");
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   struct hk_api_shader *s;
   if (builder == hk_apple9_mesh_rast_builder) {
      struct hk_apple9_mesh_rast_key mk;
      assert(key->key_size == sizeof(mk));
      memcpy(&mk, key->key, sizeof(mk));
      nir_shader stub = {
         .info = {
            .name = "mesh_meta",
            .clip_distance_array_size = mk.clip_distance_array_size,
            .cull_distance_array_size = mk.cull_distance_array_size,
         },
      };
      struct hk_per_vertex_copies copies = {
         .count = mk.per_vertex_copies,
      };
      memcpy(copies.location, mk.per_vertex_location, sizeof(copies.location));
      memcpy(copies.copy_base, mk.per_vertex_copy_base, sizeof(copies.copy_base));
      memcpy(copies.bary, mk.per_vertex_bary, sizeof(copies.bary));
      struct hk_pp_info pp = {
         .clip = mk.clip_distance_array_size,
         .cull = mk.cull_distance_array_size,
         .flat = mk.fs_interp[0],
         .face_slot = VARYING_SLOT_FOGC,
      };
      nir_shader *rast_nir = hk_nir_mesh_rast(
         &stub, &mk.info,
         !mk.polygon_points && hk_per_vertex_copies_used(&copies) ? &copies : NULL,
         mk.polygon_points ? &pp : NULL, mk.point_size);
      if (mk.line_stipple && mk.info.prim == MESA_PRIM_LINES) {
         hk_apple9_write_line_stipple_start(
            rast_nir, (struct hk_apple9_stipple_slots){
                         .start = VARYING_SLOT_TEX6,
                         .axis = VARYING_SLOT_TEX7,
                      });
      }
      if (hk_apple9_compile_internal_mesh_rast(dev, rast_nir, &mk.info,
                                               mk.fs_interp, mk.polygon_points,
                                               &s) != VK_SUCCESS)
         return NULL;
   } else {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, NULL);
      builder(&b, key->key);

      const struct vk_pipeline_robustness_state rs = {
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      };

      struct vk_shader_compile_info info = {
         .stage = b.shader->info.stage,
         .nir = b.shader,
         .robustness = &rs,
      };

      hk_preprocess_nir_internal(dev->vk.physical, b.shader);

      if (builder == hk_apple9_passthrough_gs) {
         struct hk_apple9_passthrough_suffix suffix;
         memcpy(&suffix, (uint8_t *)key->key + key->key_size - sizeof(suffix),
                sizeof(suffix));
         if (hk_apple9_compile_internal_gs(dev, b.shader, suffix.fs_interp,
                                           suffix.cull_mask, &s) != VK_SUCCESS)
            return NULL;
      } else if (hk_apple9_compile_compute(dev, &info, NULL, NULL, &s) !=
                 VK_SUCCESS) {
         return NULL;
      }
   }

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

struct hk_api_shader *
hk_apple9_mesh_rast_meta(struct hk_device *dev,
                         const struct hk_shader *mesh_main,
                         const struct hk_shader *fs,
                         bool line_stipple, bool polygon_points)
{
   struct hk_apple9_mesh_rast_key key = {0};
   key.info = mesh_main->info.mesh;
   bool pp = polygon_points && key.info.prim == MESA_PRIM_TRIANGLES;
   bool stipple = line_stipple && key.info.prim == MESA_PRIM_LINES;
   key.info.polygon_points = pp;
   key.clip_distance_array_size = mesh_main->info.clip_distance_array_size;
   key.cull_distance_array_size = mesh_main->info.cull_distance_array_size;
   key.line_stipple = stipple;
   key.polygon_points = pp;
   key.point_size =
      pp && hk_device_physical(dev)->vk.properties.polygonModePointSize;
   key.fs_interp[0] = (fs ? fs->info.fs.interp.flat : 0) |
                      (stipple ? VARYING_BIT_TEX6 : 0) |
                      (pp ? (VARYING_BIT_FOGC | VARYING_BIT_PRIMITIVE_ID) : 0);
   key.fs_interp[1] = (fs ? fs->info.fs.interp.linear : 0) |
                      (stipple ? VARYING_BIT_TEX7 : 0);
   if (fs && !pp) {
      key.per_vertex_copies = fs->info.fs.per_vertex_copies;
      memcpy(key.per_vertex_location, fs->info.fs.per_vertex_location,
             sizeof(key.per_vertex_location));
      memcpy(key.per_vertex_copy_base, fs->info.fs.per_vertex_copy_base,
             sizeof(key.per_vertex_copy_base));
      memcpy(key.per_vertex_bary, fs->info.fs.per_vertex_bary,
             sizeof(key.per_vertex_bary));
      for (unsigned j = 0; j < key.per_vertex_copies; ++j) {
         for (unsigned k = 0; k < 3; ++k) {
            key.fs_interp[0] |= BITFIELD64_BIT(
               hk_ptgs_per_vertex_copy_slot(key.per_vertex_copy_base[j], k));
         }
      }
      if (key.per_vertex_bary[1])
         key.fs_interp[1] |= BITFIELD64_BIT(key.per_vertex_bary[1]);
   }
   return hk_meta_shader(dev, hk_apple9_mesh_rast_builder, &key, sizeof(key));
}

struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size)
{
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s = hk_build_meta_shader_locked(dev, key, builder);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_shader *fs_stipple =
      gfx->shaders[MESA_SHADER_FRAGMENT]
         ? hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT])
         : NULL;
   struct hk_api_shader *vs_api = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *vs_hw =
      vs_api && !gfx->shaders[MESA_SHADER_TESS_EVAL]
         ? &vs_api->variants[HK_VS_VARIANT_HW]
         : NULL;
   bool vs_has_bary = vs_hw != NULL;
   if (vs_has_bary && fs_stipple) {
      for (unsigned m = 0; m < 2; ++m) {
         if (fs_stipple->info.fs.per_vertex_bary[m] &&
             !vs_hw->b.info.apple9_varyings.mask[fs_stipple->info.fs.per_vertex_bary[m]])
            vs_has_bary = false;
      }
   }
   bool weights_by_vs =
      vs_has_bary && !draw.indexed &&
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   /* Line stipple runs through the passthrough geometry shader (computing
    * snapped window positions directly from POS/VIEWPORT in the SW-VS/SW-TES
    * output buffer into TEX6/TEX7) with zero VS/TES modification. */
   enum mesa_prim in_prim = hk_gs_in_prim(cmd);
   bool stipple_strip =
      !gfx->shaders[MESA_SHADER_TESS_EVAL] &&
      (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
       dyn->ia.primitive_topology ==
          VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY);
   bool stipple_gs =
      dyn->rs.line.stipple.enable &&
      u_reduced_prim(in_prim) == MESA_PRIM_LINES;

   /* Polygon mode POINT on triangles draws real points from a software
    * polygon stage, see hk_apple9_polygon_point_gs. */
   struct hk_device *dev_check = hk_cmd_buffer_device(cmd);
   bool polygon_points =
      (!fs_stipple || fs_stipple == hk_only_variant(dev_check->null_fs) ||
       fs_stipple->info.fs.polygon_points) &&
      dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT &&
      u_reduced_prim(in_prim) == MESA_PRIM_TRIANGLES;

   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(TESS_EVAL) &&
       !IS_SHADER_DIRTY(GEOMETRY) && !IS_SHADER_DIRTY(FRAGMENT) &&
       !IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) &&
       weights_by_vs == gfx->apple9_weights_by_vs &&
       stipple_gs == gfx->apple9_stipple_gs &&
       polygon_points == gfx->apple9_polygon_points)
      return;
   gfx->apple9_weights_by_vs = weights_by_vs;
   gfx->apple9_stipple_gs = stipple_gs;
   gfx->apple9_polygon_points = polygon_points;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB or adjacency */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   /* Captured vertices are the polygon's, not points: transform feedback
    * keeps the hardware polygon mode. */
   polygon_points &= !xfb_outputs;
   /* A per-vertex cull mask is the primitive's only for points. */
   bool cull_mask = last_sw->info.cull_mask;
   /* Each reason for a passthrough GS, also counted under HK_CPU_TIMING. */
   unsigned reasons = (xfb_outputs ? HK_CPU_PTGS_XFB : 0) |
                      (stipple_gs ? HK_CPU_PTGS_STIPPLE : 0) |
                      (polygon_points ? HK_CPU_PTGS_POINTS : 0) |
                      (cull_mask &&
                             u_reduced_prim(hk_gs_in_prim(cmd)) != MESA_PRIM_POINTS
                          ? HK_CPU_PTGS_CULL_MASK : 0);
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
      reasons |= gfx->pg_query[i] ? HK_CPU_PTGS_PG_QUERY : 0;

   /* ...or to give the fragment shader flat copies of per-vertex inputs */
   struct hk_api_shader *fs_api_copies = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (fs_api_copies) {
      const struct hk_shader *fs_copies = hk_only_variant(fs_api_copies);
      /* The vertex shader writes the barycentric weights of non-indexed
       * triangle lists when there are no per-vertex inputs. */
      if (fs_copies->info.fs.per_vertex_copies ||
          ((fs_copies->info.fs.per_vertex_bary[0] ||
            fs_copies->info.fs.per_vertex_bary[1]) &&
           !weights_by_vs))
         reasons |= HK_CPU_PTGS_FS_COPIES;
   }

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!reasons) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   hk_cpu_timing_note_ptgs(reasons);
   size_t key_size = poly_passthrough_gs_key_size(xfb_outputs);
   struct poly_passthrough_gs_key *key = alloca(key_size);

   *key = (struct poly_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   for (uint32_t i = 0; i < NUM_TOTAL_VARYING_SLOTS; i++) {
      unsigned num_comps = 0;
      switch (i) {
         case VARYING_SLOT_TESS_LEVEL_OUTER:
         case VARYING_SLOT_TESS_LEVEL_INNER:
         case VARYING_SLOT_BOUNDING_BOX1:
            num_comps = 4;
            break;
         default:
            num_comps = nir_slot_num_components(i, MESA_SHADER_VERTEX);
            break;
      }
      key->output_components[i] = num_comps ?: 4;

      /* We don't care about VS varying types in AGX, just set everything to
       * uint32 to improve cache hits. */
      key->output_types[i] = nir_type_uint32;
   }

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   struct hk_api_shader *fs_api = gfx->shaders[MESA_SHADER_FRAGMENT];
   struct hk_shader *fs = fs_api ? hk_only_variant(fs_api) : NULL;
   struct hk_apple9_passthrough_suffix suffix = {
      .fs_interp = {
         (fs ? fs->info.fs.interp.flat : 0) |
            (stipple_gs ? VARYING_BIT_TEX6 : 0) |
            (polygon_points ? (VARYING_BIT_FOGC | VARYING_BIT_PRIMITIVE_ID) : 0),
         (fs ? fs->info.fs.interp.linear : 0) |
            (stipple_gs ? VARYING_BIT_TEX7 : 0),
      },
      .per_vertex_copies = fs ? fs->info.fs.per_vertex_copies : 0,
      .line_stipple_start = stipple_gs ? VARYING_SLOT_TEX6 : 0,
      .line_stipple_strip = stipple_gs && stipple_strip,
      .polygon_points = polygon_points,
      .polygon_face_slot = polygon_points ? VARYING_SLOT_FOGC : 0,
      .polygon_point_size = polygon_points &&
         hk_cmd_buffer_device(cmd)->vk.physical->properties.polygonModePointSize,
      .cull_mask = cull_mask,
   };
   if (fs) {
      memcpy(suffix.per_vertex_location, fs->info.fs.per_vertex_location,
             sizeof(suffix.per_vertex_location));
      memcpy(suffix.per_vertex_copy_base, fs->info.fs.per_vertex_copy_base,
             sizeof(suffix.per_vertex_copy_base));
      memcpy(suffix.per_vertex_bary, fs->info.fs.per_vertex_bary,
             sizeof(suffix.per_vertex_bary));
      for (unsigned j = 0; j < suffix.per_vertex_copies; ++j) {
         for (unsigned k = 0; k < 3; ++k) {
            suffix.fs_interp[0] |= BITFIELD64_BIT(
               hk_ptgs_per_vertex_copy_slot(suffix.per_vertex_copy_base[j], k));
         }
      }
      if (suffix.per_vertex_bary[1])
         suffix.fs_interp[1] |= BITFIELD64_BIT(suffix.per_vertex_bary[1]);
   }
   /* Without a fragment shader, the rasterized layout is unconstrained.
    * Always retain components captured by transform feedback even when an
    * independently-compiled fragment shader does not read them. */
   if (fs && fs != hk_only_variant(dev->null_fs)) {
      memcpy(suffix.components, fs->b.info.apple9_varyings.mask,
             sizeof(suffix.components));
      for (unsigned i = 0; i < xfb_outputs; ++i) {
         const nir_xfb_output_info *o = &key->xfb_info.outputs[i];
         if (o->location < ARRAY_SIZE(suffix.components))
            suffix.components[o->location] |= o->component_mask;
      }
   } else {
      memset(suffix.components, 0xff, sizeof(suffix.components));
   }
   uint8_t *full = alloca(key_size + sizeof(suffix));
   memcpy(full, key, key_size);
   memcpy(full + key_size, &suffix, sizeof(suffix));
   gs = hk_meta_shader(dev, hk_apple9_passthrough_gs, full,
                       key_size + sizeof(suffix));
   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

