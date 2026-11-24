/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gravity/debug.h"
#include <unistd.h>

#include "hk_shader.h"
#include "hk_gpu_busy.h"
#include "util/u_queue.h"

#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"
#include "pipe/p_defines.h"
#include "agx_debug.h"
#include "hk_apple9_polygon_points.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/compiler/agx_apple9_mesh.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_format_convert.h"

#include "agx_bo.h"
#include "gravity/libcl/mesh.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "agx_timeline.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_serialize.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_util.h"
#include "vk_ycbcr_conversion.h"

#include "gravity/compiler/agx_compile.h"
#include "gravity/compiler/agx_nir.h"
#include "gravity/compiler/agx_nir_texture.h"
#include "gravity/lib/agx_abi.h"
#include "gravity/lib/agx_linker.h"
#include "gravity/lib/agx_tilebuffer.h"
#include "compiler/spirv/nir_spirv.h"

#include "util/u_atomic.h"
#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

#include "hk_shader_priv.h"

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}



bool
hk_apple9_lower_clip_z_coeff(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_clip_z_coeff)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   bool dynamic = data && *(const bool *)data;
   if (dynamic) {
      unsigned offset = hk_root_descriptor_offset(draw.clip_z_coeff);
      nir_def_replace(&intr->def,
         nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, offset),
                      .align_mul = 4, .range_base = offset, .range = 4));
   } else {
      nir_def_replace(&intr->def, nir_imm_float(b, 0.5));
   }
   return true;
}

/* Route fragment outputs to the color attachments naming their locations. */
static bool
hk_apple9_remap_fs_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const uint8_t *map = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < FRAG_RESULT_DATA0 || sem.dual_source_blend_index)
      return false;
   unsigned location = sem.location - FRAG_RESULT_DATA0;
   unsigned rt = 0;
   while (rt < 8 && map[rt] != location)
      ++rt;
   if (rt == 8) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (rt == location)
      return false;
   sem.location = FRAG_RESULT_DATA0 + rt;
   nir_intrinsic_set_io_semantics(intr, sem);
   return true;
}

void
hk_apple9_remap_fs_outputs(nir_shader *nir, const uint8_t map[8])
{
   bool identity = true;
   for (unsigned rt = 0; rt < 8; ++rt)
      identity &= map[rt] == rt;
   if (identity)
      return;
   nir_shader_intrinsics_pass(nir, hk_apple9_remap_fs_output,
                              nir_metadata_control_flow, (void *)map);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Pipeline sample shading shades each sample, so interpolate every non-flat
 * input at its sample, as FragCoord already is through its sample position.
 * Apple9 interpolates explicitly and does not move pixel or centroid
 * barycentrics when shading per sample.
 */
bool
hk_apple9_lower_bary_to_sample(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   nir_def *sample;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
      sample = nir_load_barycentric_sample(
         b, intr->def.bit_size, .interp_mode = nir_intrinsic_interp_mode(intr));
      break;
   case nir_intrinsic_load_barycentric_coord_pixel:
   case nir_intrinsic_load_barycentric_coord_centroid:
      sample = nir_load_barycentric_coord_sample(
         b, intr->def.bit_size, .interp_mode = nir_intrinsic_interp_mode(intr));
      break;
   default:
      return false;
   }

   nir_def_replace(&intr->def, sample);
   return true;
}

/* Per-vertex inputs and barycentrics follow the API vertex order. Points read
 * their one vertex for every index and lines read their second vertex as the
 * third. The rasterizer orders a first-provoking triangle fan's vertices
 * (centre, then the rim), a rotation of the API order (rim, centre). The
 * primitive class (bits 0-1) and the rotation (bit 2) are draw state, so
 * select at run time from the root table.
 */
static nir_def *
hk_apple9_rasterized_prim(nir_builder *b)
{
   const unsigned offset = hk_root_descriptor_offset(draw.rasterized_prim);
   return nir_load_ubo(b, 1, 16, nir_imm_int(b, 0), nir_imm_int(b, offset),
                       .align_mul = 2, .range_base = offset, .range = 2);
}

static nir_def *
hk_apple9_load_vertex(nir_builder *b, nir_intrinsic_instr *intr,
                      unsigned vertex)
{
   nir_intrinsic_instr *load =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
   nir_def *index = nir_imm_int(b, vertex);
   nir_builder_instr_insert(b, &load->instr);
   nir_src_rewrite(&load->src[0], index);
   return &load->def;
}

bool
hk_apple9_lower_api_vertex_order(nir_shader *nir)
{
   struct util_dynarray loads;
   util_dynarray_init(&loads, NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_load_input_vertex &&
                 nir_src_is_const(intr->src[0])) ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_pixel ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_centroid ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_sample ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_sample ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_offset)
               util_dynarray_append(&loads, intr);
         }
      }
   }

   util_dynarray_foreach(&loads, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      nir_builder b = nir_builder_at(nir_after_instr(&intr->instr));
      nir_def *prim = hk_apple9_rasterized_prim(&b);
      nir_def *rotate = nir_ine_imm(&b, nir_iand_imm(&b, prim, 4), 0);
      nir_def *result;

      if (intr->intrinsic != nir_intrinsic_load_input_vertex) {
         /* API weights (w0, w1, w2) are the rasterizer's (w1, w2, w0). */
         nir_def *w = &intr->def;
         nir_def *rotated = nir_vec3(&b, nir_channel(&b, w, 1),
                                     nir_channel(&b, w, 2), nir_channel(&b, w, 0));
         result = nir_bcsel(&b, rotate, rotated, w);
      } else {
         unsigned k = nir_src_as_uint(intr->src[0]);
         nir_def *class = nir_iand_imm(&b, prim, 3);
         nir_def *v[3];
         for (unsigned i = 0; i < 3; ++i)
            v[i] = i == k ? &intr->def : hk_apple9_load_vertex(&b, intr, i);
         result = nir_bcsel(&b, rotate, v[(k + 1) % 3], v[k]);
         if (k == 2)
            result = nir_bcsel(&b, nir_ieq_imm(&b, class, 1), v[1], result);
         if (k != 0)
            result = nir_bcsel(&b, nir_ieq_imm(&b, class, 2), v[0], result);
      }
      nir_def_rewrite_uses_after(&intr->def, result);
   }

   bool progress = loads.size != 0;
   util_dynarray_fini(&loads);
   return nir_progress(progress, nir_shader_get_entrypoint(nir), nir_metadata_control_flow);
}

/* Per-vertex inputs are read as flat copies: the stage feeding the
 * rasterizer writes the values of the primitive's three API vertices to every
 * vertex. Flat varyings keep the provoking vertex's values when the
 * rasterizer clips at its guard band, so this is exact for clipped
 * primitives, where the raw-vertex coefficients hold the clipper's vertices.
 * Likewise the hardware barycentrics are relative to the clipped piece, so
 * barycentrics interpolate one-hot weights of the API vertices instead. The
 * copies take three consecutive VAR slots and the weights one VAR slot per
 * interpolation mode, which neither the producer writes nor the fragment
 * shader reads.
 */
bool
hk_is_barycentric_coord(nir_intrinsic_op op)
{
   return op == nir_intrinsic_load_barycentric_coord_pixel ||
          op == nir_intrinsic_load_barycentric_coord_centroid ||
          op == nir_intrinsic_load_barycentric_coord_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_offset;
}

static nir_def *
hk_barycentric_for_coord(nir_builder *b, nir_intrinsic_instr *intr)
{
   enum glsl_interp_mode mode = nir_intrinsic_interp_mode(intr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_coord_pixel:
      return nir_load_barycentric_pixel(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_centroid:
      return nir_load_barycentric_centroid(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_sample:
      return nir_load_barycentric_sample(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_at_sample:
      return nir_load_barycentric_at_sample(b, 32, intr->src[0].ssa,
                                            .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_at_offset:
      return nir_load_barycentric_at_offset(b, 32, intr->src[0].ssa,
                                            .interp_mode = mode);
   default:
      UNREACHABLE("not a barycentric coordinate");
   }
}

bool
hk_apple9_lower_per_vertex_copies(nir_shader *fs, uint64_t producer_outputs,
                                  struct hk_per_vertex_copies *map)
{
   memset(map, 0, sizeof(*map));
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   uint64_t used = fs->info.inputs_read | producer_outputs;
   bool bary[2] = {false, false};

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (hk_is_barycentric_coord(intr->intrinsic)) {
            bary[nir_intrinsic_interp_mode(intr) ==
                 INTERP_MODE_NOPERSPECTIVE] = true;
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_load_input_vertex)
            continue;
         if (!nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1]))
            return false;
         unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
         if (location < VARYING_SLOT_VAR0 || location >= VARYING_SLOT_VAR0 + 32)
            return false;
         uint8_t var = location - VARYING_SLOT_VAR0;
         unsigned j;
         for (j = 0; j < map->count && map->location[j] != var; ++j);
         if (j == map->count) {
            if (map->count == HK_MAX_PER_VERTEX_COPIES)
               return false;
            map->location[map->count++] = var;
         }
      }
   }
   if (!map->count && !bary[0] && !bary[1])
      return false;

   /* Use dedicated Vulkan-unused varying slots so FS can lower independently
    * of any producer: TEX0..2, TEX3..5, and {BFC0, BFC1, CLIP_VERTEX} for the
    * first 3 per-vertex inputs (falling back to unused VAR slots only if >3),
    * and COL0 / COL1 for perspective / noperspective barycentric weights. */
   static const uint8_t dedicated_copy_bases[3] = {
      VARYING_SLOT_TEX0,
      VARYING_SLOT_TEX3,
      VARYING_SLOT_BFC0,
   };
   int next = 31;
   for (unsigned j = 0; j < map->count; ++j) {
      if (j < ARRAY_SIZE(dedicated_copy_bases)) {
         map->copy_base[j] = dedicated_copy_bases[j];
         continue;
      }
      while (next >= 2 &&
             (used & BITFIELD64_RANGE(VARYING_SLOT_VAR0 + next - 2, 3)))
         --next;
      if (next < 2)
         return false;
      map->copy_base[j] = VARYING_SLOT_VAR0 + next - 2;
      used |= BITFIELD64_RANGE(VARYING_SLOT_VAR0 + next - 2, 3);
      next -= 3;
   }
   if (bary[0])
      map->bary[0] = VARYING_SLOT_COL0;
   if (bary[1])
      map->bary[1] = VARYING_SLOT_COL1;

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(&intr->instr);

         if (hk_is_barycentric_coord(intr->intrinsic)) {
            unsigned m = nir_intrinsic_interp_mode(intr) ==
                         INTERP_MODE_NOPERSPECTIVE;
            nir_def *weights = nir_load_interpolated_input(
               &b, 3, 32, hk_barycentric_for_coord(&b, intr),
               nir_imm_int(&b, 0), .dest_type = nir_type_float32,
               .io_semantics = {
                  .location = map->bary[m],
                  .num_slots = 1,
               });
            nir_def_replace(&intr->def, weights);
            continue;
         }

         if (intr->intrinsic != nir_intrinsic_load_input_vertex)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
         unsigned j;
         for (j = 0; map->location[j] != location - VARYING_SLOT_VAR0; ++j);
         unsigned vertex = nir_src_as_uint(intr->src[0]);

         nir_def *flat = nir_load_input(
            &b, intr->def.num_components, intr->def.bit_size, nir_imm_int(&b, 0),
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = {
               .location = hk_per_vertex_copy_slot(map->copy_base[j], vertex),
               .num_slots = 1,
            });
         nir_def_replace(&intr->def, flat);
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_shader_gather_info(fs, impl);
   return true;
}

/* Triangle merging shades a pixel's samples from adjacent strip or fan
 * triangles together, interpolated on one triangle's plane. That is exact at
 * the pixel centre but not at the other triangle's samples, so per-sample
 * shading and centroid, sample or offset interpolation keep the triangles
 * apart.
 */
bool
hk_apple9_interpolates_off_center(nir_shader *nir)
{
   if (nir->info.fs.uses_sample_shading)
      return true;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_barycentric_centroid:
            case nir_intrinsic_load_barycentric_sample:
            case nir_intrinsic_load_barycentric_at_sample:
            case nir_intrinsic_load_barycentric_at_offset:
            case nir_intrinsic_load_barycentric_coord_centroid:
            case nir_intrinsic_load_barycentric_coord_sample:
            case nir_intrinsic_load_barycentric_coord_at_sample:
            case nir_intrinsic_load_barycentric_coord_at_offset:
               return true;
            default:
               break;
            }
         }
      }
   }
   return false;
}

/* The one-hot barycentric weights of a vertex of a non-indexed triangle list:
 * its corner is its index within the draw, modulo 3.
 */
void
hk_apple9_write_vertex_weights(nir_shader *vs,
                               const struct hk_per_vertex_copies *map)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   /* The draw parameters hold the first vertex, as for load_first_vertex,
    * which the root lowering has already replaced. */
   nir_def *first = nir_load_ubo(
      &b, 1, 32, nir_imm_int(&b, HK_APPLE9_DRAW_PARAMS_BINDING),
      nir_imm_int(&b, 0), .align_mul = 4, .range_base = 0, .range = 4);
   nir_def *corner =
      nir_umod_imm(&b, nir_isub(&b, nir_load_vertex_id(&b), first), 3);
   nir_def *weights = nir_vec3(&b, nir_b2f32(&b, nir_ieq_imm(&b, corner, 0)),
                               nir_b2f32(&b, nir_ieq_imm(&b, corner, 1)),
                               nir_b2f32(&b, nir_ieq_imm(&b, corner, 2)));
   for (unsigned m = 0; m < 2; ++m) {
      if (!map->bary[m])
         continue;
      unsigned slot = map->bary[m];
      nir_store_output(&b, weights, nir_imm_int(&b, 0), .write_mask = 0x7,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
      vs->info.outputs_written |= BITFIELD64_BIT(slot);
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/*
 * Line stipple. The last pre-rasterization stage exports two slots. The flat
 * start slot holds a segment's first vertex in window space, snapped to the
 * vertex grid, and the stipple counter there: zero for independent segments,
 * the length of the strip so far where a geometry stage continues one. The
 * noperspective axis slot holds the vertex's own window position, which
 * interpolates to a point on the ideal line. The fragment's counter is the
 * start's counter plus its distance from the start along the major axis, and
 * the interpolated point decides that axis: a wide line's fragments lie off
 * the line, where comparing the fragment's own distances could pick the wrong
 * one. The window transform is the vertex's own viewport's, from the runtime
 * sysvals.
 */
/* Record the stage's latest position and viewport index outputs in
 * variables, for the stipple start written after them. */
struct hk_apple9_stipple_tap {
   nir_variable *pos, *viewport;
};

static struct hk_apple9_stipple_tap
hk_apple9_tap_position(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct hk_apple9_stipple_tap tap = {
      .pos = nir_local_variable_create(impl, glsl_vec4_type(), "stipple_pos"),
      .viewport =
         nir_local_variable_create(impl, glsl_uint_type(), "stipple_viewport"),
   };
   nir_store_var(&b, tap.pos, nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0), 0xf);
   nir_store_var(&b, tap.viewport, nir_imm_int(&b, 0), 1);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location;
         b.cursor = nir_after_instr(instr);
         nir_def *value = intr->src[0].ssa;
         if (location == VARYING_SLOT_POS) {
            nir_def *v = nir_load_var(&b, tap.pos);
            unsigned first = nir_intrinsic_component(intr);
            u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
               v = nir_vector_insert_imm(&b, v, nir_channel(&b, value, c),
                                         first + c);
            }
            nir_store_var(&b, tap.pos, v, 0xf);
         } else if (location == VARYING_SLOT_VIEWPORT) {
            nir_store_var(&b, tap.viewport, nir_channel(&b, value, 0), 1);
         }
      }
   }
   return tap;
}

/* The snapped window position of the tapped vertex, in its own viewport. */
static nir_def *
hk_apple9_stipple_window(nir_builder *b, struct hk_apple9_stipple_tap tap)
{
   nir_def *p = nir_load_var(b, tap.pos);
   /* Vertices behind the eye clip the segment, where the counter's start is
    * implementation-dependent; keep the divide finite. */
   nir_def *w = nir_fmax(b, nir_channel(b, p, 3), nir_imm_float(b, 0x1p-16));
   nir_def *ndc = nir_fdiv(b, nir_trim_vector(b, p, 2), w);
   nir_def *index = nir_umin(b, nir_load_var(b, tap.viewport),
                             nir_imm_int(b, AGX_APPLE9_VIEWPORT_TABLE_COUNT - 1));
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

/* The start slot holds the segment's first vertex and the stipple counter
 * there; the axis slot the vertex's own position. */
static void
hk_apple9_store_line_stipple(nir_builder *b, nir_shader *s,
                             struct hk_apple9_stipple_slots slots,
                             nir_def *start, nir_def *base, nir_def *window)
{
   nir_store_output(b, nir_vec3(b, nir_channel(b, start, 0),
                                nir_channel(b, start, 1), base),
                    nir_imm_int(b, 0), .write_mask = 0x7,
                    .src_type = nir_type_float32,
                    .io_semantics.location = slots.start,
                    .io_semantics.num_slots = 1);
   nir_store_output(b, window, nir_imm_int(b, 0), .write_mask = 0x3,
                    .src_type = nir_type_float32,
                    .io_semantics.location = slots.axis,
                    .io_semantics.num_slots = 1);
   s->info.outputs_written |=
      BITFIELD64_BIT(slots.start) | BITFIELD64_BIT(slots.axis);
}

/* Vertex and evaluation shaders: every vertex starts its segment, with the
 * counter at zero. Strips continue in the passthrough geometry shader. */
void
hk_apple9_write_line_stipple_start(nir_shader *vs,
                                   struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   struct hk_apple9_stipple_tap tap = hk_apple9_tap_position(vs);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_def *window = hk_apple9_stipple_window(&b, tap);
   hk_apple9_store_line_stipple(&b, vs, slots, window, nir_imm_float(&b, 0.0),
                                window);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, vs, nir_lower_vars_to_ssa);
}

/*
 * Geometry shaders emitting line strips: the counter continues along each
 * output strip and restarts at EndPrimitive. A segment's flat values come
 * from its provoking vertex, so under the first-vertex mode each vertex
 * carries its own position and counter, and under the last-vertex mode the
 * previous vertex's.
 */
void
hk_apple9_write_gs_line_stipple(nir_shader *gs,
                                struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   struct hk_apple9_stipple_tap tap = hk_apple9_tap_position(gs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *prev =
      nir_local_variable_create(impl, glsl_vec_type(2), "stipple_prev");
   nir_variable *prev_base =
      nir_local_variable_create(impl, glsl_float_type(), "stipple_prev_base");
   nir_variable *have_prev =
      nir_local_variable_create(impl, glsl_bool_type(), "stipple_have_prev");
   nir_store_var(&b, prev, nir_imm_vec2(&b, 0.0, 0.0), 0x3);
   nir_store_var(&b, prev_base, nir_imm_float(&b, 0.0), 1);
   nir_store_var(&b, have_prev, nir_imm_false(&b), 1);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         if (intr->intrinsic == nir_intrinsic_end_primitive ||
             intr->intrinsic == nir_intrinsic_end_primitive_with_counter) {
            nir_store_var(&b, have_prev, nir_imm_false(&b), 1);
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_emit_vertex &&
             intr->intrinsic != nir_intrinsic_emit_vertex_with_counter)
            continue;
         nir_def *window = hk_apple9_stipple_window(&b, tap);
         nir_def *had = nir_load_var(&b, have_prev);
         nir_def *p = nir_load_var(&b, prev);
         nir_def *pb = nir_load_var(&b, prev_base);
         nir_def *d = nir_fabs(&b, nir_fsub(&b, window, p));
         nir_def *base = nir_bcsel(
            &b, had,
            nir_fadd(&b, pb, nir_fmax(&b, nir_channel(&b, d, 0),
                                      nir_channel(&b, d, 1))),
            nir_imm_float(&b, 0.0));
         nir_def *last =
            nir_iand(&b, had, nir_ine_imm(&b, nir_load_provoking_last(&b), 0));
         hk_apple9_store_line_stipple(&b, gs, slots,
                                      nir_bcsel(&b, last, p, window),
                                      nir_bcsel(&b, last, pb, base), window);
         nir_store_var(&b, prev, window, 0x3);
         nir_store_var(&b, prev_base, base, 1);
         nir_store_var(&b, have_prev, nir_imm_true(&b), 1);
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
}

void
hk_apple9_lower_line_stipple_fs(nir_shader *fs,
                                struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *line = nir_load_input(
      &b, 3, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = slots.start, .num_slots = 1});
   nir_def *start = nir_trim_vector(&b, line, 2);
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *on_line = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = slots.axis, .num_slots = 1});
   nir_def *axis = nir_fabs(&b, nir_fsub(&b, on_line, start));
   nir_def *x_major =
      nir_fge(&b, nir_channel(&b, axis, 0), nir_channel(&b, axis, 1));
   nir_def *pixel = nir_fadd_imm(
      &b, nir_u2f32(&b, nir_u2u32(&b, nir_load_pixel_coord(&b))), 0.5);
   nir_def *d = nir_fabs(&b, nir_fsub(&b, pixel, start));
   nir_def *counter = nir_f2u32(
      &b, nir_fadd(&b, nir_channel(&b, line, 2),
                   nir_bcsel(&b, x_major, nir_channel(&b, d, 0),
                             nir_channel(&b, d, 1))));
   agx_nir_line_stipple_test(&b, counter);
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_shader_gather_info(fs, impl);
}

/* Generic varying components a shader stores (outputs) or loads (inputs),
 * per slot and component, 64-bit values counting twice. Indirectly addressed
 * arrays count their whole slots. Clip and cull distances are ordinary output
 * components here, as in the API limit.
 */
static unsigned
hk_apple9_generic_components(const nir_shader *s, bool outputs)
{
   uint8_t mask[32] = {0};
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned bits, comps;
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output:
            case nir_intrinsic_store_per_vertex_output:
            case nir_intrinsic_store_per_primitive_output:
               if (!outputs)
                  continue;
               bits = nir_src_bit_size(intr->src[0]);
               comps = nir_intrinsic_write_mask(intr);
               break;
            case nir_intrinsic_load_input:
            case nir_intrinsic_load_interpolated_input:
            case nir_intrinsic_load_per_vertex_input:
            case nir_intrinsic_load_per_primitive_input:
               if (outputs)
                  continue;
               bits = intr->def.bit_size;
               comps = BITFIELD_MASK(intr->def.num_components);
               break;
            default:
               continue;
            }
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (sem.location < VARYING_SLOT_VAR0 ||
                sem.location >= VARYING_SLOT_VAR0 + 32)
               continue;
            unsigned slot = sem.location - VARYING_SLOT_VAR0;
            uint32_t m = comps << nir_intrinsic_component(intr);
            if (bits == 64) {
               uint32_t wide = 0;
               u_foreach_bit(c, m)
                  wide |= 3u << (2 * c);
               m = wide;
            }
            if (sem.num_slots > 1) {
               for (unsigned i = 0; i < sem.num_slots && slot + i < 32; ++i)
                  mask[slot + i] = 0xf;
            } else {
               for (unsigned i = 0; m && slot + i < 32; ++i, m >>= 4)
                  mask[slot + i] |= m & 0xf;
            }
         }
      }
   }
   unsigned total = 0;
   for (unsigned i = 0; i < 32; ++i)
      total += util_bitcount(mask[i]);
   if (outputs) {
      total += s->info.clip_distance_array_size +
               s->info.cull_distance_array_size;
   }
   return total;
}

/* The stipple slots: the highest two generic slots neither stage uses,
 * provided the four stipple components fit beside both the last stage's
 * outputs and the fragment shader's inputs within the API output limit.
 * Otherwise zero: the line is drawn unstippled, a known limitation.
 */
struct hk_apple9_stipple_slots
hk_apple9_line_stipple_slots(const nir_shader *last, const nir_shader *fs)
{
   struct hk_apple9_stipple_slots slots = {0};
   unsigned used_components = MAX2(hk_apple9_generic_components(last, true),
                                   hk_apple9_generic_components(fs, false));
   if (used_components + HK_APPLE9_STIPPLE_COMPONENTS >
       HK_APPLE9_OUTPUT_COMPONENTS)
      return slots;
   uint64_t used = last->info.outputs_written | fs->info.inputs_read;
   for (int k = 31; k >= 0 && !slots.axis; --k) {
      unsigned slot = VARYING_SLOT_VAR0 + k;
      if (used & BITFIELD64_BIT(slot))
         continue;
      if (slots.start)
         slots.axis = slot;
      else
         slots.start = slot;
   }
   if (!slots.axis)
      slots.start = 0;
   return slots;
}

/*
 * Polygon points are rasterized as points: their hardware facing is always
 * front and their hardware primitive ID counts points. The software polygon
 * stage writes the polygon's back-facing bit to VARYING_SLOT_FOGC and its
 * primitive ID to VARYING_SLOT_PRIMITIVE_ID. Fragment shaders select FOGC == 0
 * when root.draw.polygon_point_facing is set, and hardware gl_FrontFacing
 * otherwise; gl_PrimitiveID travels directly via VARYING_SLOT_PRIMITIVE_ID.
 */
bool
hk_apple9_lower_polygon_slot(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   unsigned slot = data ? *(const unsigned *)data : VARYING_SLOT_FOGC;
   if (intr->intrinsic != nir_intrinsic_load_front_face)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   unsigned offset = hk_root_descriptor_offset(draw.polygon_point_facing);
   nir_def *use_pp = nir_ine_imm(
      b, nir_load_ubo(b, 1, 16, nir_imm_int(b, 0), nir_imm_int(b, offset),
                      .align_mul = 2, .range_base = offset, .range = 2),
      0);
   nir_def *back = nir_load_input(
      b, 1, 32, nir_imm_int(b, 0), .dest_type = nir_type_uint32,
      .io_semantics = {.location = slot, .num_slots = 1});
   nir_def *sel = nir_bcsel(b, use_pp, nir_ieq_imm(b, back, 0), &intr->def);
   nir_def_rewrite_uses_after(&intr->def, sel);
   return true;
}

void
hk_apple9_write_polygon_slot(nir_shader *s, unsigned slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = slot, .num_slots = 1});
   s->info.outputs_written |= BITFIELD64_BIT(slot);
   nir_progress(true, impl, nir_metadata_control_flow);
}

void
hk_apple9_write_primitive_id(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID,
                                     .num_slots = 1});
   s->info.outputs_written |= VARYING_BIT_PRIMITIVE_ID;
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* Decide whether a pipeline's polygon-mode points use the software stage:
 * its rasterized points publish the stage's outputs, clip and cull distances
 * included, VARYING_SLOT_FOGC (facing) and VARYING_SLOT_PRIMITIVE_ID, point
 * size and any layer or viewport word, within the varying bank. Returns
 * VARYING_SLOT_FOGC through face_slot when the fragment shader reads facing.
 */
bool
hk_apple9_polygon_point_budget(const nir_shader *last, const nir_shader *fs,
                               uint64_t reserved, unsigned *face_slot)
{
   *face_slot = 0;
   unsigned generic = MAX2(hk_apple9_generic_components(last, true),
                           hk_apple9_generic_components(fs, false));
   bool reads_face =
      BITSET_TEST(fs->info.system_values_read, SYSTEM_VALUE_FRONT_FACE);
   bool reads_prim = (fs->info.inputs_read & VARYING_BIT_PRIMITIVE_ID) != 0;
   /* The vertex shader rasterizing mesh records always writes the layer. */
   bool layer = (last->info.outputs_written &
                 (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)) ||
                last->info.stage == MESA_SHADER_MESH;
   if (generic + reads_face + reads_prim + 1 + layer >
       AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return false;
   if (reads_face)
      *face_slot = VARYING_SLOT_FOGC;
   return true;
}


/* Apple9 selects out-of-bounds zeroes with a generic comparison. */
static nir_def *
bounds_check(nir_builder *b, nir_def *data, nir_def *offs, nir_def *bound)
{
   /* TODO: Optimize */
   return nir_bcsel(b, nir_uge(b, bound, offs), data,
                    nir_imm_zero(b, data->num_components, data->bit_size));
}

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

/* Whether the first load_size bytes of a bounded access are in bounds. */
static nir_def *
check_in_bounds_size(nir_builder *b, nir_intrinsic_instr *intr,
                     unsigned load_size)
{
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;

   /* Try to bounds check in terms of elements */
   nir_scalar offset_s = nir_scalar_resolved(offset, 0);
   if (is_op(offset_s, nir_op_amul)) {
      nir_scalar srcs[] = {nir_scalar_chase_alu_src(offset_s, 0),
                           nir_scalar_chase_alu_src(offset_s, 1)};
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      if (nir_scalar_is_const(srcs[1 - i]) &&
          nir_scalar_as_uint(srcs[1 - i]) == load_size) {

         nir_def *index = nir_mov_scalar(b, srcs[i]);
         return nir_ult(b, index, nir_udiv_imm(b, bound, load_size));
      }
   }

   /* TODO: handle also the iadd(amul) pattern, this is important */

   /* Otherwise bounds check in bytes */
   nir_def *sat_offset = nir_umin_imm(b, offset, UINT32_MAX - (load_size - 1));
   return nir_ult(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);
}

static nir_def *
check_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   unsigned bit_size = intr->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   return check_in_bounds_size(b, intr, bit_size / 8 * intr->num_components);
}

/* An amul whose scale is not a shift of 0..4. */
static bool
hk_unencodable_scale(nir_scalar s)
{
   if (!is_op(s, nir_op_amul))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar k = nir_scalar_chase_alu_src(s, i);
      if (nir_scalar_is_const(k))
         return !util_is_power_of_two_nonzero(nir_scalar_as_uint(k)) ||
                nir_scalar_as_uint(k) > 16;
   }
   return true;
}

static nir_def *
bound_offset(nir_builder *b, nir_def *valid, nir_scalar offset,
             bool shift_only_scale)
{
   /* If we can, clamp the source of an amul instead of the result, to remain
    * compatible with the hardware address mode.
    */
   if (is_op(offset, nir_op_amul)) {
      nir_scalar srcs[] = {
         nir_scalar_chase_alu_src(offset, 0),
         nir_scalar_chase_alu_src(offset, 1),
      };
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      /* Apple9 scales an index only by a shift of 0..4. Any other scale
       * would multiply again after the select; selecting the product, which
       * the bounds check already computed, saves that multiply. */
      if (shift_only_scale &&
          !(nir_scalar_is_const(srcs[1 - i]) &&
            util_is_power_of_two_nonzero(nir_scalar_as_uint(srcs[1 - i])) &&
            nir_scalar_as_uint(srcs[1 - i]) <= 16))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset), nir_imm_int(b, 0));
      nir_def *x = nir_mov_scalar(b, srcs[i]);
      nir_def *y = nir_mov_scalar(b, srcs[1 - i]);

      return nir_amul(b, nir_bcsel(b, valid, x, nir_imm_int(b, 0)), y);
   }

   /* Similar case for when there's an addition (chain) in the way */
   if (is_op(offset, nir_op_iadd)) {
      nir_scalar x = nir_scalar_chase_alu_src(offset, 0);
      nir_scalar y = nir_scalar_chase_alu_src(offset, 1);

      /* With an Apple9 scale the index cannot encode, select the complete
       * offset the bounds check computed rather than each term. */
      if (shift_only_scale &&
          (hk_unencodable_scale(x) || hk_unencodable_scale(y)))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset), nir_imm_int(b, 0));

      if (is_op(x, nir_op_amul) || is_op(y, nir_op_amul)) {
         return nir_iadd(b, bound_offset(b, valid, x, shift_only_scale),
                         bound_offset(b, valid, y, shift_only_scale));
      }
   }

   nir_def *def = nir_mov_scalar(b, offset);

   /* If the offset fits within the zero page, clamping is pointless */
   if (nir_scalar_is_const(offset) &&
       (nir_scalar_as_uint(offset) + 16) < AGX_ZERO_PAGE_SIZE)
      return def;

   /* Otherwise, fallback on clamping the offset */
   return nir_bcsel(b, valid, def, nir_imm_int(b, 0));
}

static void
lower_load_global_bounded(nir_builder *b, nir_intrinsic_instr *intr,
                          bool per_component_bounds)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   /* A vector merged from separate accesses (bounds_checked_modes) keeps
    * each component's own check: robustness2 returns in-bounds components
    * even when a later one is out of bounds. Address the load by its first
    * component; the vectorizer only merges naturally aligned accesses, so
    * the read stays within one aligned block of the same page. */
   unsigned byte_size = intr->def.bit_size / 8;
   bool per_component = per_component_bounds && intr->def.num_components > 1;

   /* Bounds check compatibly with the hardware address mode */
   nir_def *valid = per_component ? check_in_bounds_size(b, intr, byte_size)
                                  : check_in_bounds(b, intr);
   base = nir_bcsel(b, valid, base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   offset = bound_offset(b, valid, nir_scalar_resolved(offset, 0),
                         per_component_bounds);

   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   }

   nir_def *val = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, base, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr), .access = access);

   if (per_component) {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      comps[0] = nir_channel(b, val, 0);
      for (unsigned c = 1; c < intr->def.num_components; ++c) {
         comps[c] = nir_bcsel(b, check_in_bounds_size(b, intr, (c + 1) * byte_size),
                              nir_channel(b, val, c),
                              nir_imm_zero(b, 1, intr->def.bit_size));
      }
      val = nir_vec(b, comps, intr->def.num_components);
   }

   nir_def_replace(&intr->def, val);
}

/* Merging loads issues the combined load at the first one, so every later
 * component stays live from there. Apple9 shaders that already reach the
 * register limit then need more copies and swaps, so only merge accesses that
 * are close together (HK_APPLE9_VECTORIZE_WINDOW).
 */
/* Index of the first use in the load's own block, or -1 if a use is in
 * another block or not an instruction. The vectorizer numbers each block's
 * instructions in order before calling back. */
static int64_t
hk_first_use_index(nir_intrinsic_instr *load)
{
   int64_t first = INT64_MAX;
   nir_foreach_use_including_if(use, &load->def) {
      if (nir_src_is_if(use))
         return -1;
      nir_instr *user = nir_src_use_instr(use);
      if (user->block != load->instr.block)
         return -1;
      first = MIN2(first, (int64_t)user->index);
   }
   return first == INT64_MAX ? -1 : first;
}

/* Windowed merging joins loads at most this many instructions apart. */
#define HK_APPLE9_VECTORIZE_WINDOW 4
/* Robust fields merge only when their first uses are this close. */
#define HK_APPLE9_VECTORIZE_USE_WINDOW 2
/* Largest merged robust load, in bytes. */
#define HK_APPLE9_ROBUST_MERGE_BYTES 16

/* Peak number of live 32-bit GPR words in a shader, from NIR liveness:
 * a backward walk of each block from its live-out set. 16-bit values count as
 * half a word; constants, undefs, and top-level non-divergent values that
 * nir_opt_preamble hoists into uniform registers are free.
 */
static bool
hk_src_is_preamble_eligible(nir_src *src, void *data)
{
   return ((const bool *)data)[src->ssa->index];
}

/* Compute preamble eligibility in a single forward SSA pass (zero recursion),
 * matching the backend's top-level straight-line preamble subset and
 * reorderability requirement. */
static bool
hk_instr_preamble_eligible(nir_instr *instr, const nir_def *def,
                           const bool *preamble_eligible)
{
   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef)
      return true;
   if (def->divergent || !instr->block ||
       instr->block->cf_node.parent->type != nir_cf_node_function)
      return false;

   switch (instr->type) {
   case nir_instr_type_alu:
      return nir_foreach_src(instr, hk_src_is_preamble_eligible,
                             (void *)preamble_eligible);
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_uniform:
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_ro_sink_address_poly:
         return nir_foreach_src(instr, hk_src_is_preamble_eligible,
                                (void *)preamble_eligible);
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_global_2x32:
      case nir_intrinsic_load_global_bounded:
      case nir_intrinsic_load_ssbo: {
         enum gl_access_qualifier access = nir_intrinsic_access(intr);
         return !(access & ACCESS_VOLATILE) &&
                ((access & ACCESS_CAN_REORDER) != 0) &&
                nir_foreach_src(instr, hk_src_is_preamble_eligible,
                                (void *)preamble_eligible);
      }
      default:
         return false;
      }
   }
   default:
      return false;
   }
}

static unsigned
hk_def_halves(const nir_def *def, const bool *preamble_eligible)
{
   if (preamble_eligible[def->index])
      return 0;
   return def->num_components * MAX2(def->bit_size, 16) / 16;
}

static bool
hk_index_def(nir_def *def, void *data)
{
   ((nir_def **)data)[def->index] = def;
   return true;
}

struct hk_live_walk {
   bool *live;
   const bool *preamble_eligible;
   unsigned *halves;
};

static bool
hk_add_live_src(nir_src *src, void *data)
{
   struct hk_live_walk *walk = data;
   if (!walk->live[src->ssa->index]) {
      walk->live[src->ssa->index] = true;
      *walk->halves += hk_def_halves(src->ssa, walk->preamble_eligible);
   }
   return true;
}

static unsigned
hk_apple9_peak_live_words(nir_shader *nir)
{
   nir_divergence_analysis(nir);
   unsigned peak = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_live_defs);
      nir_def **defs = calloc(impl->ssa_alloc, sizeof(*defs));
      bool *live = calloc(impl->ssa_alloc, sizeof(*live));
      bool *preamble_eligible = calloc(impl->ssa_alloc, sizeof(*preamble_eligible));
      if (!defs || !live || !preamble_eligible) {
         free(preamble_eligible);
         free(live);
         free(defs);
         continue;
      }
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            nir_foreach_def(instr, hk_index_def, defs);
            nir_def *def = nir_instr_def(instr);
            if (def) {
               preamble_eligible[def->index] =
                  hk_instr_preamble_eligible(instr, def, preamble_eligible);
            }
         }
      }
      nir_foreach_block(block, impl) {
         memset(live, 0, impl->ssa_alloc * sizeof(*live));
         unsigned halves = 0;
         U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
            live[i] = true;
            if (defs[i])
               halves += hk_def_halves(defs[i], preamble_eligible);
         }
         peak = MAX2(peak, halves);
         nir_foreach_instr_reverse(instr, block) {
            if (instr->type == nir_instr_type_phi)
               break;
            nir_def *def = nir_instr_def(instr);
            if (def && live[def->index]) {
               live[def->index] = false;
               halves -= hk_def_halves(def, preamble_eligible);
            }
            struct hk_live_walk walk = {live, preamble_eligible, &halves};
            nir_foreach_src(instr, hk_add_live_src, &walk);
            peak = MAX2(peak, halves);
         }
      }
      free(preamble_eligible);
      free(live);
      free(defs);
   }
   return DIV_ROUND_UP(peak, 2);
}

/* The issue and first-use windows in hk_apple9_vectorize_cb protect shaders
 * near the register limit, where a merged load keeps later fields live from
 * the first one's issue. Across DXVK they also leave most robust fields of
 * a vector or matrix as separate loads with their own bounds checks and
 * waits. So apply them only when the shader's estimated peak of live words
 * exceeds HK_APPLE9_VECTORIZE_PRESSURE_BUDGET.
 */
struct hk_vectorize_policy {
   bool windowed;
   bool soft_fault;
};

static bool hk_apple9_vectorize_cb(unsigned align_mul, unsigned align_offset,
                                   unsigned bit_size, unsigned num_components,
                                   int64_t hole_size, nir_intrinsic_instr *low,
                                   nir_intrinsic_instr *high, void *data);

/* G16G occupancy scales across real cliffs below 48 GPRs and stays flat at
 * 832 threads/core across 48..96 GPRs. Keep unwindowed merging active up to
 * 84 divergent live words (unless it inflates pressure substantially across
 * a real sub-48 occupancy cliff); above 84 words, allow unwindowed merging
 * only when it strictly reduces peak live words by at least 8 and stays
 * within the 96-GPR register file. */
#define HK_APPLE9_VECTORIZE_PRESSURE_BUDGET 84

static unsigned
hk_apple9_count_vectorizable_global_ops(const nir_shader *nir)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_offset:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_global_bounded:
            case nir_intrinsic_load_global_constant_bounded:
            case nir_intrinsic_store_global:
            case nir_intrinsic_store_global_offset:
               if (++count >= 2)
                  return count;
               break;
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (nir_src_as_deref(intr->src[0])->modes & nir_var_mem_global) {
                  if (++count >= 2)
                     return count;
               }
               break;
            default:
               break;
            }
         }
      }
   }
   return count;
}

/* Vectorize global/bounded loads and stores, gating pressure estimation and
 * trial cloning on the presence of candidate ops and reusing the winning
 * vectorized function implementation instead of re-running a third time. */
static void
hk_apple9_vectorize_global(nir_shader *nir, bool soft_fault)
{
   const unsigned global_ops = hk_apple9_count_vectorizable_global_ops(nir);
   if (global_ops == 0)
      return;

   struct hk_vectorize_policy policy = {
      .windowed = true,
      .soft_fault = soft_fault,
   };
   const nir_load_store_vectorize_options vec_opts = {
      .modes = nir_var_mem_global,
      .bounds_checked_modes = nir_var_mem_global,
      .callback = hk_apple9_vectorize_cb,
      .cb_data = &policy,
   };

   /* Bounded loads perform their own exact 32-bit non-wrap proof inside
    * hk_apple9_vectorize_cb (avoiding both false rejects on iand/umin offsets
    * and false accepts on non-power-of-two gcd strides in check_for_robustness). */
   if (global_ops == 1) {
      NIR_PASS(_, nir, nir_opt_load_store_vectorize, &vec_opts);
      return;
   }

   /* Below the pressure budget, merge without windows, unless that raises
    * the shader's peak of live words over windowed merging. */
   const bool dbg_pressure = agx_apple9_debug_enabled(AGX_A9_DBG_PRESSURE);
   const unsigned peak = hk_apple9_peak_live_words(nir);
   policy.windowed = peak > 2 * HK_APPLE9_VECTORIZE_PRESSURE_BUDGET;
   unsigned peak_windowed = 0, peak_free = 0;

   if (policy.windowed) {
      NIR_PASS(_, nir, nir_opt_load_store_vectorize, &vec_opts);
   } else if (exec_list_is_singular(&nir->functions)) {
      nir_function *func = nir_shader_get_entrypoint(nir)->function;
      nir_function_impl *orig_impl = func->impl;
      nir_function_impl *backup_impl =
         !orig_impl->preamble ? nir_function_impl_clone(nir, orig_impl) : NULL;

      if (!backup_impl) {
         policy.windowed = true;
         NIR_PASS(_, nir, nir_opt_load_store_vectorize, &vec_opts);
      } else {
         /* Call passes directly while backup_impl is detached from func->impl:
          * NIR_PASS under NIR_DEBUG=clone/serialize replaces nir's contents
          * and would free detached children owned by nir. */
         bool free_progress = nir_opt_load_store_vectorize(nir, &vec_opts);
         if (!free_progress) {
            /* Windowed mode is strictly more restrictive than unwindowed, so
             * it would also make no merges. */
            peak_free = peak_windowed = peak;
            ralloc_free(backup_impl);
         } else {
            nir_opt_dce(nir);
            peak_free = hk_apple9_peak_live_words(nir);
            /* Below 48 GPRs, real G16G occupancy cliffs sit at 18/24/28/32/36/
             * 40/44 GPRs; across 48..96 GPRs occupancy is flat at 832 thr/core.
             * Only check windowed fallback below 84 words when unwindowed
             * merging inflates pressure substantially across a real sub-48
             * occupancy cliff. */
            const unsigned occ_before =
               agx_occupancy_for_register_count(2 * peak).max_threads;
            const unsigned occ_free =
               agx_occupancy_for_register_count(2 * peak_free).max_threads;
            const bool sub48_cliff =
               peak_free <= HK_APPLE9_VECTORIZE_PRESSURE_BUDGET &&
               occ_free < occ_before && peak_free >= peak + 4u;
            if (peak_free <= HK_APPLE9_VECTORIZE_PRESSURE_BUDGET &&
                !sub48_cliff && !dbg_pressure) {
               ralloc_free(backup_impl);
            } else {
               nir_function_set_impl(func, backup_impl);
               policy.windowed = true;
               nir_opt_load_store_vectorize(nir, &vec_opts);
               if (sub48_cliff || peak_free <= 96u || dbg_pressure) {
                  nir_opt_dce(nir);
                  peak_windowed = hk_apple9_peak_live_words(nir);
               }
               if (sub48_cliff) {
                  const unsigned occ_win =
                     agx_occupancy_for_register_count(2 * peak_windowed).max_threads;
                  policy.windowed = occ_win > occ_free;
               } else {
                  /* Above the 84-word plateau budget, keep windowed merging
                   * unless unwindowed merging strictly reduces peak live words
                   * by at least 8 words and stays within the 96-GPR file. */
                  policy.windowed =
                     peak_free > HK_APPLE9_VECTORIZE_PRESSURE_BUDGET &&
                     !(peak_free + 8u <= peak_windowed && peak_free <= 96u);
               }
               if (policy.windowed) {
                  ralloc_free(orig_impl);
               } else {
                  nir_function_set_impl(func, orig_impl);
                  ralloc_free(backup_impl);
               }
            }
         }
         nir_validate_shader(nir, "after hk_apple9_vectorize_global");
      }
   } else {
      policy.windowed = true;
      NIR_PASS(_, nir, nir_opt_load_store_vectorize, &vec_opts);
   }

   if (dbg_pressure)
      fprintf(stderr,
              "HK_PRESSURE stage=%u peak=%u windowed=%u merged_peaks=%u/%u\n",
              nir->info.stage, peak,
              policy.windowed, peak_windowed, peak_free);
}

/* Compute the proven power-of-two stride modulo 2^32 and constant remainder of
 * a 32-bit offset expression (returning 0 when the offset is constant). Only
 * power-of-two factors survive 32-bit wrap-around in Z/2^32 Z. */
static uint32_t
hk_apple9_offset_pow2_stride(nir_scalar s, uint32_t *const_out, unsigned depth)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_scalar_is_const(s)) {
      *const_out = nir_scalar_as_uint(s);
      return 0;
   }
   if (!nir_scalar_is_alu(s) || depth >= 8) {
      *const_out = 0;
      return 1;
   }
   switch (nir_scalar_alu_op(s)) {
   case nir_op_iadd:
   case nir_op_isub: {
      uint32_t ca = 0, cb = 0;
      uint32_t sa = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 0),
                                                 &ca, depth + 1);
      uint32_t sb = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 1),
                                                 &cb, depth + 1);
      *const_out = nir_scalar_alu_op(s) == nir_op_iadd ? (ca + cb) : (ca - cb);
      if (sa == 0)
         return sb;
      if (sb == 0)
         return sa;
      return MIN2(sa, sb);
   }
   case nir_op_ior: {
      uint32_t ca = 0, cb = 0;
      uint32_t sa = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 0),
                                                 &ca, depth + 1);
      uint32_t sb = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 1),
                                                 &cb, depth + 1);
      if (sb == 0 && sa > 0 && (cb & (sa - 1)) == cb && (ca & cb) == 0) {
         *const_out = ca | cb;
         return sa;
      }
      if (sa == 0 && sb > 0 && (ca & (sb - 1)) == ca && (cb & ca) == 0) {
         *const_out = ca | cb;
         return sb;
      }
      *const_out = 0;
      return 1;
   }
   case nir_op_imul:
   case nir_op_amul: {
      uint32_t ca = 0, cb = 0;
      uint32_t sa = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 0),
                                                 &ca, depth + 1);
      uint32_t sb = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 1),
                                                 &cb, depth + 1);
      if (sb == 0) {
         if (cb == 0) {
            *const_out = 0;
            return 0;
         }
         *const_out = ca * cb;
         if (sa == 0)
            return 0;
         uint32_t p = cb & -cb;
         uint64_t r = (uint64_t)sa * p;
         return r >= (1ULL << 32) ? 0u : (uint32_t)r;
      }
      if (sa == 0) {
         if (ca == 0) {
            *const_out = 0;
            return 0;
         }
         *const_out = ca * cb;
         uint32_t p = ca & -ca;
         uint64_t r = (uint64_t)sb * p;
         return r >= (1ULL << 32) ? 0u : (uint32_t)r;
      }
      *const_out = 0;
      return 1;
   }
   case nir_op_ishl: {
      uint32_t ca = 0, cb = 0;
      uint32_t sa = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 0),
                                                 &ca, depth + 1);
      uint32_t sb = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 1),
                                                 &cb, depth + 1);
      if (sb == 0) {
         unsigned sh = cb & 31u;
         *const_out = ca << sh;
         if (sa == 0)
            return 0;
         uint64_t r = (uint64_t)sa << sh;
         return r >= (1ULL << 32) ? 0u : (uint32_t)r;
      }
      *const_out = 0;
      return 1;
   }
   case nir_op_iand: {
      uint32_t ca = 0, cb = 0;
      uint32_t sa = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 0),
                                                 &ca, depth + 1);
      uint32_t sb = hk_apple9_offset_pow2_stride(nir_scalar_chase_alu_src(s, 1),
                                                 &cb, depth + 1);
      if (sb == 0) {
         if (sa == 0) {
            *const_out = ca & cb;
            return 0;
         }
         uint32_t known_lo = (ca & (sa - 1)) & cb;
         uint32_t high_mask = cb & ~(sa - 1);
         *const_out = known_lo;
         return high_mask == 0 ? 0u : (high_mask & -high_mask);
      }
      if (sa == 0) {
         uint32_t known_lo = (cb & (sb - 1)) & ca;
         uint32_t high_mask = ca & ~(sb - 1);
         *const_out = known_lo;
         return high_mask == 0 ? 0u : (high_mask & -high_mask);
      }
      uint32_t min_s = MIN2(sa, sb);
      *const_out = (ca & cb) & (min_s - 1);
      return min_s;
   }
   default:
      *const_out = 0;
      return 1;
   }
}

/* Compute a sound unsigned 32-bit upper bound for a 32-bit offset expression. */
static uint32_t
hk_apple9_offset_max_u32(nir_scalar s, unsigned depth)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_scalar_is_const(s))
      return (uint32_t)nir_scalar_as_uint(s);
   if (!nir_scalar_is_alu(s) || depth >= 8)
      return UINT32_MAX;
   switch (nir_scalar_alu_op(s)) {
   case nir_op_umin:
   case nir_op_iand: {
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 0), depth + 1);
      uint32_t mb = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 1), depth + 1);
      return MIN2(ma, mb);
   }
   case nir_op_ushr: {
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 0), depth + 1);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      sh = nir_scalar_resolved(sh.def, sh.comp);
      if (nir_scalar_is_const(sh))
         return ma >> (nir_scalar_as_uint(sh) & 31u);
      return ma;
   }
   case nir_op_u2u32: {
      nir_scalar src = nir_scalar_chase_alu_src(s, 0);
      if (src.def->bit_size < 32)
         return (1u << src.def->bit_size) - 1u;
      if (src.def->bit_size == 32)
         return hk_apple9_offset_max_u32(src, depth + 1);
      return UINT32_MAX;
   }
   case nir_op_iadd:
   case nir_op_ior: {
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 0), depth + 1);
      uint32_t mb = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 1), depth + 1);
      uint64_t sum = (uint64_t)ma + mb;
      return sum > UINT32_MAX ? UINT32_MAX : (uint32_t)sum;
   }
   case nir_op_imul:
   case nir_op_amul: {
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 0), depth + 1);
      uint32_t mb = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 1), depth + 1);
      uint64_t prod = (uint64_t)ma * mb;
      return prod > UINT32_MAX ? UINT32_MAX : (uint32_t)prod;
   }
   case nir_op_ishl: {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      sh = nir_scalar_resolved(sh.def, sh.comp);
      if (!nir_scalar_is_const(sh))
         return UINT32_MAX;
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 0), depth + 1);
      uint64_t r = (uint64_t)ma << (nir_scalar_as_uint(sh) & 31u);
      return r > UINT32_MAX ? UINT32_MAX : (uint32_t)r;
   }
   case nir_op_bcsel: {
      uint32_t ma = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 1), depth + 1);
      uint32_t mb = hk_apple9_offset_max_u32(nir_scalar_chase_alu_src(s, 2), depth + 1);
      return MAX2(ma, mb);
   }
   default:
      return UINT32_MAX;
   }
}

static bool
hk_apple9_vectorize_cb(unsigned align_mul, unsigned align_offset,
                       unsigned bit_size, unsigned num_components,
                       int64_t hole_size, nir_intrinsic_instr *low,
                       nir_intrinsic_instr *high, void *data)
{
   if (!agx_mem_vectorize_cb(align_mul, align_offset, bit_size, num_components,
                             hole_size, low, high, NULL))
      return false;
   const struct hk_vectorize_policy *policy = data;
   int64_t window = policy->windowed ? HK_APPLE9_VECTORIZE_WINDOW : INT64_MAX;
   int64_t distance = (int64_t)high->instr.index - (int64_t)low->instr.index;
   if (llabs(distance) > window)
      return false;
   /* A merged load issues every field where the first one was. In shaders
    * near the register limit that keeps later fields live much longer, so
    * only merge robust fields whose first uses are close as well. */
   int64_t use_window = HK_APPLE9_VECTORIZE_USE_WINDOW;
   bool bounded = low->intrinsic == nir_intrinsic_load_global_bounded ||
                  low->intrinsic == nir_intrinsic_load_global_constant_bounded;
   /* Robust merges of fields whose first uses are adjacent shorten Lethal
    * Company's light loops without extra spills at the register limit. */
   int64_t robust_bytes = HK_APPLE9_ROBUST_MERGE_BYTES;
   if (bounded && num_components * bit_size / 8 > robust_bytes)
      return false;
   if (bounded && policy->windowed) {
      int64_t low_use = hk_first_use_index(low), high_use = hk_first_use_index(high);
      if (low_use < 0 || high_use < 0 || llabs(high_use - low_use) > use_window)
         return false;
   }

   /* A merged bounded load may read past the end of its buffer when only
    * its first components are in bounds. Without soft faults, natural
    * alignment keeps that read inside one aligned block so it cannot cross
    * into another page. With soft faults enabled, 4-byte-aligned 32-bit
    * vec2/vec3/vec4 loads are safe even across page boundaries. */
   if (bounded) {
      /* Never widen, narrow, or overlap bounded loads past their original
       * robustness check unit (e.g. u8vec3@0 + u8vec3@3, u32@0 + u16vec2@4,
       * or overlapping u16vec2@0 + u16@1). */
      if (hole_size != 0 || low->def.bit_size != bit_size ||
          high->def.bit_size != bit_size)
         return false;
      unsigned total_bytes = num_components * bit_size / 8;
      unsigned bytes = util_next_power_of_two(total_bytes);
      unsigned req_align = (policy->soft_fault && bit_size == 32) ? 4 : bytes;
      if (align_mul < req_align || align_offset % req_align)
         return false;
      /* Ensure the 32-bit dynamic offset cannot wrap modulo 2^32 across the
       * merged span (e.g. non-power-of-two stride 20*0x33333333 == 0xfffffffc
       * wraps at +4). Accept when either the proven power-of-two residue
       * satisfies rem + total_bytes <= stride or the proven unsigned upper
       * bound satisfies max_off + total_bytes <= 2^32. */
      if (low->src[1].ssa->bit_size != 32)
         return false;
      nir_scalar off_scalar = nir_get_scalar(low->src[1].ssa, 0);
      uint32_t off_const = 0;
      uint32_t off_stride =
         hk_apple9_offset_pow2_stride(off_scalar, &off_const, 0);
      bool non_wrapping;
      if (off_stride == 0) {
         non_wrapping = (uint64_t)off_const + total_bytes <= (1ULL << 32);
      } else {
         uint32_t rem = off_const & (off_stride - 1);
         non_wrapping = (uint64_t)rem + total_bytes <= off_stride;
      }
      if (!non_wrapping) {
         uint32_t max_off = hk_apple9_offset_max_u32(off_scalar, 0);
         non_wrapping = (uint64_t)max_off + total_bytes <= (1ULL << 32);
      }
      if (!non_wrapping)
         return false;
   }
   return true;
}

struct hk_bounded_lowering {
   bool soft_fault;
   /* Merged vector loads are checked per component (bounds_checked_modes). */
   bool per_component;
};

static nir_def *
hk_soft_fault_bounds_check_dyn(nir_builder *b, nir_def *data, nir_def *offset,
                               unsigned end_bytes, nir_def *bound)
{
   assert(end_bytes >= 1);
   /* Overflow-safe unsigned check: offset + end_bytes <= bound iff
    * offset < usub_sat(bound, end_bytes - 1). Because bound is typically
    * uniform, usub_sat(bound, end_bytes - 1) hoists into the preamble and
    * is shared across loads from the same buffer. */
   nir_def *limit =
      end_bytes == 1 ? bound : nir_usub_sat(b, bound, nir_imm_int(b, end_bytes - 1));
   return nir_bcsel(b, nir_ult(b, offset, limit), data,
                    nir_imm_zero(b, data->num_components, data->bit_size));
}

static bool
lower_load_global_constant_offset_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin, void *data)
{
   const struct hk_bounded_lowering *opts = data;
   const bool has_soft_fault = opts->soft_fault;
   const bool is_const_bounded =
      intrin->intrinsic == nir_intrinsic_load_global_constant_bounded;
   const bool is_ssbo_bounded =
      intrin->intrinsic == nir_intrinsic_load_global_bounded;
   const bool is_bounded = is_const_bounded || is_ssbo_bounded;

   if (intrin->intrinsic != nir_intrinsic_load_global_constant_offset &&
       !is_bounded)
      return false;

   const bool is_volatile =
      (nir_intrinsic_access(intrin) & ACCESS_VOLATILE) != 0;

   if (is_bounded && (!has_soft_fault || (is_ssbo_bounded && is_volatile))) {
      lower_load_global_bounded(b, intrin, opts->per_component);
      return true;
   }

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *base_addr = intrin->src[0].ssa;
   nir_def *offset = intrin->src[1].ssa;
   nir_def *bound = is_bounded ? intrin->src[2].ssa : NULL;

   unsigned bit_size = intrin->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intrin->num_components;

   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);

   /* Without robustness an access outside the buffer is undefined, so the
    * load itself is in bounds. The Apple9 backend relies on this to fold
    * constant offsets past an amul: the hardware adds the displacement in
    * 64 bits, which differs from the 32-bit offset only outside the buffer.
    */
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);
   if (intrin->intrinsic == nir_intrinsic_load_global_constant_offset)
      access |= ACCESS_IN_BOUNDS;

   nir_def *addr = nir_iadd(b, base_addr, nir_u2u64(b, offset));
   nir_def *val = is_ssbo_bounded
      ? nir_load_global(b, intrin->def.num_components, intrin->def.bit_size,
                        addr, .align_mul = align_mul,
                        .align_offset = align_offset, .access = access)
      : nir_load_global_constant(b, intrin->def.num_components,
                                 intrin->def.bit_size, addr,
                                 .align_mul = align_mul,
                                 .align_offset = align_offset, .access = access);

   if (is_bounded && opts->per_component && intrin->def.num_components > 1) {
      /* A vector merged from separate robust fields (bounds_checked_modes)
       * returns each in-bounds component, as in lower_load_global_bounded:
       * checking only its last byte would zero the in-bounds fields too. A
       * field split over two merged loads then reads (x, y, 0, 0) where
       * robustness2 allows only (x, y, z, 0) or zero (uniform_buffer
       * len_12/31/252).
       */
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      if (nir_scalar_is_const(offs)) {
         uint32_t offs_imm = nir_scalar_as_uint(offs);
         if (offs_imm <= UINT32_MAX - load_size &&
             offs_imm + load_size > UINT8_MAX) {
            /* Compute remaining bytes once (preamble-hoistable for uniform
             * bounds) so every component compares against an 8-bit literal. */
            nir_def *rem = nir_usub_sat(b, bound, nir_imm_int(b, offs_imm));
            for (unsigned c = 0; c < intrin->def.num_components; ++c) {
               comps[c] = bounds_check(b, nir_channel(b, val, c),
                                       nir_imm_int(b, (c + 1) * byte_size), rem);
            }
         } else {
            for (unsigned c = 0; c < intrin->def.num_components; ++c) {
               unsigned end_c = (c + 1) * byte_size;
               if (offs_imm > UINT32_MAX - end_c) {
                  comps[c] = nir_imm_zero(b, 1, intrin->def.bit_size);
               } else {
                  comps[c] = bounds_check(b, nir_channel(b, val, c),
                                          nir_imm_int(b, offs_imm + end_c),
                                          bound);
               }
            }
         }
      } else {
         for (unsigned c = 0; c < intrin->def.num_components; ++c) {
            comps[c] = hk_soft_fault_bounds_check_dyn(
               b, nir_channel(b, val, c), offset, (c + 1) * byte_size, bound);
         }
      }
      val = nir_vec(b, comps, intrin->def.num_components);
   } else if (is_bounded) {
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      if (nir_scalar_is_const(offs)) {
         uint32_t offs_imm = nir_scalar_as_uint(offs);
         if (offs_imm > UINT32_MAX - load_size) {
            val = nir_imm_zero(b, val->num_components, val->bit_size);
         } else {
            /* Compare the last byte loaded against the bound exactly. Bounds
             * are the descriptor's exact range, not rounded to 64 B, so
             * rounding the offset down would let loads past a short range
             * through (robustness2 uniform_buffer len_4..len_60). */
            val = bounds_check(b, val, nir_imm_int(b, offs_imm + load_size),
                               bound);
         }
      } else {
         val = hk_soft_fault_bounds_check_dyn(b, val, offset, load_size, bound);
      }
   }

   nir_def_replace(&intrin->def, val);
   return true;
}

/* Lower bounded and offset global loads. Apple9 checks each component of a
 * merged vector separately. */
bool
hk_lower_bounded_global_loads(nir_shader *nir, bool soft_fault, bool apple9)
{
   struct hk_bounded_lowering bounded = {
      .soft_fault = soft_fault,
      .per_component = apple9,
   };
   return nir_shader_intrinsics_pass(nir, lower_load_global_constant_offset_instr,
                                     nir_metadata_none, &bounded);
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct hk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct hk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion
             ? &sampler->vk.ycbcr_conversion->state
             : NULL;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * This is the world's worst multiview implementation. We simply duplicate each
 * draw on the CPU side, changing a uniform in between, and then plumb the view
 * index into the layer ID here. Whatever, it works.
 *
 * The "proper" implementation on AGX would use vertex amplification, but a
 * MacBook is not a VR headset.
 */
static void
hk_lower_multiview(nir_shader *nir)
{
   /* If there's an existing layer ID write, ignore it. This avoids validation
    * splat with vk_meta.
    */
   nir_variable *existing = nir_find_variable_with_location(
      nir, nir_var_shader_out, VARYING_SLOT_LAYER);

   if (existing) {
      existing->data.mode = nir_var_shader_temp;
      existing->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   /* Now write the view index as the layer */
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_variable *layer =
      nir_variable_create(nir, nir_var_shader_out, glsl_uint_type(), NULL);

   layer->data.location = VARYING_SLOT_LAYER;

   nir_store_var(&b, layer, nir_load_view_index(&b), nir_component_mask(1));
   b.shader->info.outputs_written |= VARYING_BIT_LAYER;
}

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1,
                                  nir_type_float16, false, false);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op == nir_texop_lod &&
       b->shader->info.stage == MESA_SHADER_COMPUTE &&
       b->shader->info.derivative_group == DERIVATIVE_GROUP_LINEAR) {
      /* The compute query consumes the view clamp after sampler/view level
       * clamping, before mip rounding. It is not a sampler minimum LOD. */
      b->cursor = nir_before_instr(&tex->instr);
      nir_tex_instr_add_src(tex, nir_tex_src_min_lod,
                            query_min_lod(b, tex));
      return true;
   }
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   /* Apple9 legalizes the minimum-LOD source as a float, so fetches take
    * the float minimum and compare against its floor. */
   nir_def *min_lod = query_min_lod(b, tex);
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (tex->op == nir_texop_tg4) {
      b->cursor = nir_after_instr(&tex->instr);

      /* The Vulkan spec section "Texel Gathering" says:
       *
       *    If levelbase < minLodIntegerimageView, then any values fetched are
       *    zero if the robustImageAccess2 feature is enabled.
       *
       * We currently always enable robustImageAccess2, so implement that
       * semantic here.
       *
       * We could probably optimize this with a special descriptor for this case
       * but tg4 is rare enough I'm not bothered.
       */
      nir_def *old = &tex->def;
      nir_def *oob = nir_fge_imm(b, min_lod, 1.0);
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

static bool
agx_nir_lower_image_view_min_lod(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none, NULL);
}

/* The Apple9 rasterizer delivers the layer to the fragment layer system
 * value. The viewport index is an ordinary flat varying.
 */
static bool
hk_apple9_lower_layer_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr,
                                  UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *layer = nir_iand_imm(b, nir_load_layer_id(b), 0xffff);
   nir_def_replace(&intr->def, nir_u2uN(b, layer, intr->def.bit_size));
   return true;
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static bool
should_lower_robust(const nir_intrinsic_instr *intr, const void *_)
{
   /* The hardware is robust, but our software image atomics are not. Unlike the
    * GL driver, we don't use the common buffer image lowering, using the
    * agx_nir_lower_texture lowering for robustImageAccess2 semantics.
    */
   return intr->intrinsic == nir_intrinsic_image_deref_atomic ||
          intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
}

/* Hardware PBE takes 16-bit unsigned spatial coordinates and clips >= extent
 * natively; saturating 32-bit coordinates prevents 16-bit wraparound while
 * mapping every out-of-bounds or negative coordinate above max extent (16384). */
#define HK_APPLE9_PBE_COORD_MAX 0xffffu

/* 2D linear texel-buffer sentinel ((0xffff << 14) | 0x3fff): decodes to
 * x = 16383, y = 65535 (> AGX_TEXTURE_BUFFER_MAX_HEIGHT and > 1 for null
 * descriptors) without 16-bit wraparound, triggering HW OOB clipping on both
 * TEXTURE_LOAD (format-aware zero/alpha_default) and IMAGE_WRITE (discard). */
#define HK_APPLE9_TBUF_OOB_SENTINEL ((HK_APPLE9_PBE_COORD_MAX << 14) | 0x3fffu)

/* Uniform texel formats are descriptor state, not shader specialization. The
 * texture unit converts every format but RGB32, which has no native format,
 * exactly as it does for images, so fetch through the view's linear 2D
 * descriptor using branchless OOB sentinel clipping; RGB32 bounds-checks its
 * three-word global load. Bounds and texel counts come from the descriptor's
 * metadata in GPU memory. */
static bool
hk_apple9_lower_texel_buffer(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF ||
       (tex->op != nir_texop_txf && tex->op != nir_texop_txs))
      return false;
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle_idx < 0)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *ptr = nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
   ptr = nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_descriptor, metadata));
   nir_def *count = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, elements)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   if (tex->op == nir_texop_txs) {
      nir_def_replace(&tex->def, count);
      return true;
   }

   int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = nir_u2u32(b, tex->src[coord_idx].src.ssa);
   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   nir_def *in_bounds = nir_ult(b, coord, count);
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *size = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_push_if(b, nir_ieq_imm(b, size, 12));
   nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
   nir_push_if(b, in_bounds);
   nir_def *xyz = nir_load_global(b, 3, 32,
      nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, coord, 12))), .align_mul = 4);
   nir_def *rgb32_in = nir_vec4(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1),
                                nir_channel(b, xyz, 2), one);
   nir_push_else(b, NULL);
   nir_def *rgb32_oob = nir_vec4(b, nir_imm_int(b, 0), nir_imm_int(b, 0),
                                 nir_imm_int(b, 0), one);
   nir_pop_if(b, NULL);
   nir_def *rgb32 = nir_if_phi(b, rgb32_in, rgb32_oob);
   nir_push_else(b, NULL);
   /* The descriptor starts at the 16-byte aligned address at or below the
    * first texel; other texel sizes are powers of two. Map OOB indices
    * branchlessly to HK_APPLE9_TBUF_OOB_SENTINEL so HW TEXTURE_LOAD returns
    * format-aware (0, 0, 0, alpha_default) without control flow. */
   nir_def *tail = nir_ushr(b, nir_iand_imm(b, nir_u2u32(b, base), 15),
                            nir_ufind_msb(b, size));
   nir_def *index = nir_bcsel(b, in_bounds, nir_iadd(b, coord, tail),
                              nir_imm_int(b, HK_APPLE9_TBUF_OOB_SENTINEL));
   nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 3);
   fetch->op = nir_texop_txf;
   fetch->sampler_dim = GLSL_SAMPLER_DIM_2D;
   fetch->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   fetch->coord_components = 2;
   fetch->texture_index = tex->texture_index;
   fetch->texture_non_uniform = tex->texture_non_uniform;
   /* Bounds are enforced via HK_APPLE9_TBUF_OOB_SENTINEL, and level is zero. */
   fetch->backend_flags = (tex->backend_flags & HK_APPLE9_TEX_DESCRIPTOR_HANDLE) |
                          AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   fetch->src[0] = (nir_tex_src){.src_type = nir_tex_src_texture_handle,
      .src = nir_src_for_ssa(tex->src[handle_idx].src.ssa)};
   fetch->src[1] = (nir_tex_src){.src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec2(b,
         nir_iand_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH - 1),
         nir_ushr_imm(b, index, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH))))};
   fetch->src[2] = (nir_tex_src){.src_type = nir_tex_src_lod,
      .src = nir_src_for_ssa(nir_imm_int(b, 0))};
   nir_def_init(&fetch->instr, &fetch->def, 4, 32);
   nir_builder_instr_insert(b, &fetch->instr);
   nir_pop_if(b, NULL);
   nir_def *result = nir_trim_vector(b, nir_if_phi(b, rgb32, &fetch->def),
                                     tex->def.num_components);
   nir_def_replace(&tex->def, result);
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Storage texel buffers carry their full RW base address inside
 * hk_apple9_storage_descriptor::buf.base. */
static nir_def *
hk_apple9_buffer_texel_address(nir_builder *b, nir_def *desc, nir_def *coord,
                               nir_def *size_B)
{
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, desc, offsetof(struct hk_apple9_storage_descriptor, buf.base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *offset = nir_imul(b, nir_channel(b, coord, 0), size_B);
   return nir_iadd(b, base, nir_u2u64(b, offset));
}

static bool
hk_apple9_lower_buffer_address(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   bool atomic = intr->intrinsic == nir_intrinsic_bindless_image_atomic;
   bool swap = intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   if (!atomic && !swap &&
       intr->intrinsic != nir_intrinsic_bindless_image_texel_address)
      return false;
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   unsigned size_B = atomic || swap ? intr->def.bit_size / 8 :
      util_format_get_blocksize(nir_intrinsic_format(intr));
   nir_def *addr = hk_apple9_buffer_texel_address(
      b, desc, intr->src[1].ssa, nir_imm_int(b, size_B));
   nir_def *result = addr;
   if (atomic || swap) {
      result = swap ? nir_global_atomic_swap(b, intr->def.bit_size, addr,
         intr->src[3].ssa, intr->src[4].ssa,
         .atomic_op = nir_intrinsic_atomic_op(intr)) :
         nir_global_atomic(b, intr->def.bit_size, addr, intr->src[3].ssa,
                           .atomic_op = nir_intrinsic_atomic_op(intr));
      /* Preserve the memory-to-texture fence used by common image lowering. */
      if (!(nir_intrinsic_access(intr) & ACCESS_NON_READABLE))
         nir_fence_mem_to_tex_agx(b);
   }
   nir_def_replace(&intr->def, result);
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

static nir_def *hk_apple9_storage_buffer_residency_code(nir_builder *b,
                                                        nir_def *desc,
                                                        nir_def *x);

/* Linear 2D-buffer coordinate for a multi-slice 3D block-texel alias whose
 * descriptor sets HK_APPLE9_STORAGE_LINEAR_2D_INDEX: twiddle (x, y) within
 * the compressed mip level's tile geometry, add z * layer_stride_el, and map
 * OOB lanes branchlessly to HK_APPLE9_TBUF_OOB_SENTINEL. */
static nir_def *
hk_apple9_alias_3d_xyz(nir_builder *b, nir_def *desc, nir_def *x, nir_def *y,
                       nir_def *z, nir_def *in_bounds)
{
   /* Omit ACCESS_UNIFORM_SAFE_AGX so this cold-arm metadata load stays inside
    * the uniform LINEAR_2D_INDEX branch instead of bloating normal 3D preambles. */
   nir_def *meta = nir_load_global_constant(b, 3, 32,
      nir_iadd_imm(b, desc,
                   offsetof(struct hk_apple9_storage_descriptor, alias_3d)),
      .align_mul = 16);
   nir_def *layer_stride_el = nir_channel(b, meta, 0);
   nir_def *aligned_width_el = nir_channel(b, meta, 1);
   nir_def *tile_wh = nir_channel(b, meta, 2);
   nir_def *tile_w = nir_u2u16(b, tile_wh);
   nir_def *tile_h = nir_u2u16(b, nir_ushr_imm(b, tile_wh, 16));
   nir_def *xy16 = nir_vec2(b, nir_u2u16(b, x), nir_u2u16(b, y));
   nir_def *twiddled =
      libagx_twiddle_coordinates(b, xy16, tile_w, tile_h, aligned_width_el);
   nir_def *idx = nir_iadd(b, twiddled, nir_imul(b, z, layer_stride_el));
   nir_def *e = nir_bcsel(b, in_bounds, idx,
                          nir_imm_int(b, HK_APPLE9_TBUF_OOB_SENTINEL));
   return nir_vec3(b, nir_iand_imm(b, e, AGX_TEXTURE_BUFFER_WIDTH - 1),
                   nir_ushr_imm(b, e, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH)),
                   nir_imm_int(b, 0));
}

static nir_def *
hk_apple9_pack_load_result(nir_builder *b, nir_def *color, nir_def *code,
                           unsigned num_components, bool sparse)
{
   if (!sparse)
      return nir_trim_vector(b, color, num_components);
   unsigned color_comps = num_components - 1;
   nir_scalar comps[5];
   for (unsigned c = 0; c < color_comps; ++c)
      comps[c] = nir_get_scalar(color, c);
   comps[color_comps] = nir_get_scalar(code, 0);
   return nir_vec_scalars(b, comps, num_components);
}

/* Fetch a storage texel through the inline native texture descriptor at
 * offset 96 of hk_apple9_storage_descriptor using the non-reorderable
 * image_load_apple9_agx intrinsic (propagating coherent/volatile directly
 * into hardware TEXTURE_LOAD / TEXTURE_LOAD_MS). Logical bounds are clipped
 * branchlessly against storage->extent so padded/POT-tail block-texel aliases,
 * sliced 3D views, MSAA views, storage texel buffers, and null descriptors
 * return format-aware (0, 0, 0, alpha_default) without software format decode. */
static nir_def *
hk_apple9_texture_load(nir_builder *b, nir_intrinsic_instr *intr,
                       nir_def *desc, nir_def *coord, bool sparse,
                       enum gl_access_qualifier access)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   enum pipe_format format = nir_intrinsic_format(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool ms = dim == GLSL_SAMPLER_DIM_MS;
   bool array = nir_intrinsic_image_array(intr) || cube;
   coord = nir_u2u32(b, coord);

   nir_alu_type base_type = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));
   if (base_type == nir_type_invalid)
      base_type = nir_type_uint;
   nir_alu_type dest_type = base_type | 32;
   nir_def *index = nir_ushr_imm(b, nir_iadd_imm(b, nir_u2u32(b, desc),
      offsetof(struct hk_apple9_storage_descriptor, texture)), 5);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *fetched;

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      nir_def *elements = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, desc,
                      offsetof(struct hk_apple9_storage_descriptor, buf.elements)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *packed = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, desc,
                      offsetof(struct hk_apple9_storage_descriptor, native_ok)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *tail = nir_iand_imm(b, nir_ushr_imm(b, packed, HK_APPLE9_STORAGE_TAIL_SHIFT),
                                   HK_APPLE9_STORAGE_TAIL_MASK);
      nir_def *x = nir_channel(b, coord, 0);
      nir_def *in_bounds = nir_ult(b, x, elements);
      nir_def *e = nir_bcsel(b, in_bounds, nir_iadd(b, x, tail),
                             nir_imm_int(b, HK_APPLE9_TBUF_OOB_SENTINEL));
      nir_def *xyz = nir_vec3(b, nir_iand_imm(b, e, AGX_TEXTURE_BUFFER_WIDTH - 1),
                              nir_ushr_imm(b, e, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH)),
                              zero);
      fetched = nir_image_load_apple9_agx(
         b, 4, 32, index, xyz, zero, zero,
         .base = 0, .access = access, .dest_type = dest_type,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false, .flags = 0);
   } else {
      unsigned ext_comps = ms ? 4 : (array || dim == GLSL_SAMPLER_DIM_3D) ? 3 :
                           dim == GLSL_SAMPLER_DIM_1D ? 1 : 2;
      nir_def *ext = nir_load_global_constant(b, ext_comps, 32,
         nir_iadd_imm(b, desc,
                      offsetof(struct hk_apple9_storage_descriptor, extent)),
         .align_mul = 16, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *raw_x = nir_channel(b, coord, 0);
      nir_def *raw_y = dim == GLSL_SAMPLER_DIM_1D ? zero : nir_channel(b, coord, 1);
      nir_def *in_bounds = nir_ult(b, raw_x, nir_channel(b, ext, 0));
      if (dim != GLSL_SAMPLER_DIM_1D)
         in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_y, nir_channel(b, ext, 1)));

      nir_def *raw_z = zero;
      if (array || dim == GLSL_SAMPLER_DIM_3D) {
         raw_z = nir_channel(b, coord, dim == GLSL_SAMPLER_DIM_1D ? 1 : 2);
         in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_z, nir_channel(b, ext, 2)));
      }
      nir_def *sample = zero;
      if (ms) {
         nir_def *raw_sample = nir_u2u32(b, intr->src[2].ssa);
         in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_sample, nir_channel(b, ext, 3)));
         sample = nir_bcsel(b, in_bounds, raw_sample, zero);
      }

      if (dim == GLSL_SAMPLER_DIM_3D &&
          (format == PIPE_FORMAT_NONE || util_format_get_blocksize(format) >= 8)) {
         nir_def *flags = nir_load_global_constant(b, 1, 32,
            nir_iadd_imm(b, desc,
                         offsetof(struct hk_apple9_storage_descriptor, native_ok)),
            .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
         nir_push_if(b, nir_test_mask(b, flags, HK_APPLE9_STORAGE_LINEAR_2D_INDEX));
         nir_def *lin_xyz = hk_apple9_alias_3d_xyz(b, desc, raw_x, raw_y, raw_z, in_bounds);
         nir_def *lin_res = nir_image_load_apple9_agx(
            b, 4 + sparse, 32, index, lin_xyz, zero, zero,
            .base = 0, .access = access, .dest_type = dest_type,
            .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
            .flags = sparse ? 1u : 0u);
         nir_push_else(b, NULL);
         nir_def *nat_xyz = nir_vec3(b,
            nir_bcsel(b, in_bounds, raw_x, nir_imm_int(b, -1)),
            nir_bcsel(b, in_bounds, raw_y, zero),
            nir_bcsel(b, in_bounds, raw_z, zero));
         nir_def *nat_res = nir_image_load_apple9_agx(
            b, 4 + sparse, 32, index, nat_xyz, zero, zero,
            .base = 0, .access = access, .dest_type = dest_type,
            .image_dim = GLSL_SAMPLER_DIM_3D, .image_array = false,
            .flags = sparse ? 1u : 0u);
         nir_pop_if(b, NULL);
         fetched = nir_if_phi(b, lin_res, nat_res);
      } else {
         nir_def *oob_x = nir_imm_int(b, ms ? HK_APPLE9_PBE_COORD_MAX : -1);
         nir_def *xyz = nir_vec3(b,
            nir_bcsel(b, in_bounds, raw_x, oob_x),
            nir_bcsel(b, in_bounds, raw_y, zero),
            nir_bcsel(b, in_bounds, raw_z, zero));
         enum glsl_sampler_dim hw_dim =
            (cube || dim == GLSL_SAMPLER_DIM_1D) ? GLSL_SAMPLER_DIM_2D : dim;
         fetched = nir_image_load_apple9_agx(
            b, 4 + sparse, 32, index, xyz, sample, zero,
            .base = 0, .access = access, .dest_type = dest_type,
            .image_dim = hw_dim, .image_array = array,
            .flags = sparse ? 1u : 0u);
      }
   }

   nir_def *color = sparse ? nir_trim_vector(b, fetched, 4) : fetched;
   nir_def *code = sparse ? nir_channel(b, fetched, 4) : NULL;
   if (intr->def.bit_size != 32) {
      color = base_type == nir_type_float
                 ? nir_f2fN(b, color, intr->def.bit_size)
                 : nir_u2uN(b, color, intr->def.bit_size);
      if (sparse)
         code = nir_u2uN(b, code, intr->def.bit_size);
   }
   return hk_apple9_pack_load_result(b, color, code, intr->def.num_components, sparse);
}

/* Store through the inline native PBE entry at offset 64 of the descriptor
 * with branchless logical bounds clipping against storage->extent: OOB lanes
 * map to x = 0xffff (or HK_APPLE9_TBUF_OOB_SENTINEL for MSAA / linear 3D
 * aliases) so HW PBE discards them without control flow (including padded
 * block-texel aliases, POT mip tails, and null descriptors where extent == 0).
 * Sparse-resident images map scratch_bo read-write in their primary VA, so
 * unbound writes discard cleanly without per-store residency checks. */
static void
hk_apple9_native_store(nir_builder *b, nir_intrinsic_instr *intr,
                       nir_def *desc, nir_def *coord,
                       enum gl_access_qualifier access)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   enum pipe_format format = nir_intrinsic_format(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool ms = dim == GLSL_SAMPLER_DIM_MS;
   bool array = nir_intrinsic_image_array(intr);
   bool native_array = array || cube || dim == GLSL_SAMPLER_DIM_3D;
   coord = nir_u2u32(b, coord);

   nir_def *zero = nir_imm_int(b, 0);
   nir_def *max_coord = nir_imm_int(b, HK_APPLE9_PBE_COORD_MAX);
   unsigned ext_comps = ms ? 4 : native_array ? 3 :
                        dim == GLSL_SAMPLER_DIM_1D ? 1 : 2;
   nir_def *ext = nir_load_global_constant(b, ext_comps, 32,
      nir_iadd_imm(b, desc,
                   offsetof(struct hk_apple9_storage_descriptor, extent)),
      .align_mul = 16, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *raw_x = nir_channel(b, coord, 0);
   nir_def *raw_y = dim == GLSL_SAMPLER_DIM_1D ? zero : nir_channel(b, coord, 1);
   nir_def *in_bounds = nir_ult(b, raw_x, nir_channel(b, ext, 0));
   if (dim != GLSL_SAMPLER_DIM_1D)
      in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_y, nir_channel(b, ext, 1)));
   nir_def *raw_z = zero;
   if (native_array) {
      raw_z = nir_channel(b, coord, dim == GLSL_SAMPLER_DIM_1D ? 1 : 2);
      in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_z, nir_channel(b, ext, 2)));
   }
   nir_def *raw_sample = zero;
   if (ms) {
      raw_sample = nir_u2u32(b, intr->src[2].ssa);
      in_bounds = nir_iand(b, in_bounds, nir_ult(b, raw_sample, nir_channel(b, ext, 3)));
   }

   nir_def *data = intr->src[3].ssa;
   nir_alu_type type = nir_intrinsic_src_type(intr);
   nir_alu_type base = nir_alu_type_get_base_type(type);
   if (base == nir_type_invalid)
      base = nir_type_uint;
   if (data->bit_size != 16 && data->bit_size != 32) {
      data = base == nir_type_float ? nir_f2f32(b, data) :
             base == nir_type_int   ? nir_i2i32(b, data) : nir_u2u32(b, data);
   }
   type = base | data->bit_size;
   nir_def *padded_data = nir_pad_vector_imm_int(b, data, 0, 4);

   /* The hardware drops helper invocations' image writes, but not under
    * sample shading (as in agx_nir_lower_frag_sidefx). */
   bool helpers = b->shader->info.stage == MESA_SHADER_FRAGMENT &&
                  b->shader->info.fs.uses_sample_shading;
   if (helpers)
      nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));

   nir_def *index = nir_ushr_imm(b, nir_iadd_imm(b, nir_u2u32(b, desc),
      offsetof(struct hk_apple9_storage_descriptor, pbe)), 5);

   if (ms) {
      nir_def *size_B;
      if (array && format == PIPE_FORMAT_NONE) {
         size_B = nir_load_global_constant(b, 1, 32,
            nir_iadd_imm(b, desc,
                         offsetof(struct hk_apple9_storage_descriptor, size_B)),
            .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
         size_B = nir_umax_imm(b, size_B, 1);
      } else {
         size_B = nir_imm_int(b, format == PIPE_FORMAT_NONE
                                    ? 4
                                    : util_format_get_blocksize(format));
      }
      nir_def *idx_sa = nir_u2u32(b, libagx_image_texel_address(
         b, desc, nir_pad_vector_imm_int(b, coord, 0, 4), raw_sample, size_B,
         nir_imm_false(b), nir_imm_true(b), nir_imm_bool(b, array),
         nir_imm_true(b)));
      nir_def *e = nir_bcsel(b, in_bounds, idx_sa,
                             nir_imm_int(b, HK_APPLE9_TBUF_OOB_SENTINEL));
      nir_def *xyz = nir_vec3(b, nir_iand_imm(b, e, AGX_TEXTURE_BUFFER_WIDTH - 1),
                              nir_ushr_imm(b, e, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH)),
                              zero);
      nir_image_store_apple9_agx(b, index, xyz, padded_data,
         .access = access, .src_type = type,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false);
   } else if (dim == GLSL_SAMPLER_DIM_3D &&
              (format == PIPE_FORMAT_NONE || util_format_get_blocksize(format) >= 8)) {
      nir_def *flags = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, desc,
                      offsetof(struct hk_apple9_storage_descriptor, native_ok)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_push_if(b, nir_test_mask(b, flags, HK_APPLE9_STORAGE_LINEAR_2D_INDEX));
      nir_def *lin_xyz = hk_apple9_alias_3d_xyz(b, desc, raw_x, raw_y, raw_z, in_bounds);
      nir_image_store_apple9_agx(b, index, lin_xyz, padded_data,
         .access = access, .src_type = type,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false);
      nir_push_else(b, NULL);
      nir_def *nat_xyz = nir_vec3(b,
         nir_bcsel(b, in_bounds, raw_x, max_coord),
         nir_bcsel(b, in_bounds, raw_y, zero),
         nir_bcsel(b, in_bounds, raw_z, zero));
      nir_image_store_apple9_agx(b, index, nat_xyz, padded_data,
         .access = access, .src_type = type,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true);
      nir_pop_if(b, NULL);
   } else {
      nir_def *xyz = nir_vec3(b,
         nir_bcsel(b, in_bounds, raw_x, max_coord),
         nir_bcsel(b, in_bounds, raw_y, zero),
         nir_bcsel(b, in_bounds, raw_z, zero));
      nir_image_store_apple9_agx(b, index, xyz, padded_data,
         .access = access, .src_type = type,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = native_array);
   }

   if (helpers)
      nir_pop_if(b, NULL);

   /* Coherent/volatile writes may be observed by other workgroups without a
    * shader barrier: drain the PBE write immediately. */
   if (access & (ACCESS_COHERENT | ACCESS_VOLATILE))
      nir_fence_image_write_apple9_agx(b);
}

/* Store element x of a storage texel buffer through the inline native PBE
 * entry at offset 64 of the descriptor (hk_pack_texel_buffer): a linear 2D
 * image 16384 texels wide, with the view's sub-16-byte offset (in native_ok)
 * added to the element. OOB indices map branchlessly to
 * HK_APPLE9_TBUF_OOB_SENTINEL (y = 0xffff > height) so HW PBE clips them
 * without control flow (including null descriptors where elements == 0). */
static void
hk_apple9_native_buffer_store(nir_builder *b, nir_intrinsic_instr *intr,
                              nir_def *desc, nir_def *x, bool residency,
                              enum gl_access_qualifier access)
{
   x = nir_u2u32(b, x);
   nir_def *elements = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, desc, offsetof(struct hk_apple9_storage_descriptor, buf.elements)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *packed = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, desc, offsetof(struct hk_apple9_storage_descriptor, native_ok)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *tail = nir_iand_imm(b, nir_ushr_imm(b, packed, HK_APPLE9_STORAGE_TAIL_SHIFT),
                                HK_APPLE9_STORAGE_TAIL_MASK);
   nir_def *in_bounds = nir_ult(b, x, elements);
   nir_def *e = nir_bcsel(b, in_bounds, nir_iadd(b, x, tail),
                          nir_imm_int(b, HK_APPLE9_TBUF_OOB_SENTINEL));
   nir_def *xyz = nir_vec3(b, nir_iand_imm(b, e, AGX_TEXTURE_BUFFER_WIDTH - 1),
                           nir_ushr_imm(b, e, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH)),
                           nir_imm_int(b, 0));
   nir_def *index = nir_ushr_imm(b, nir_iadd_imm(b, nir_u2u32(b, desc),
      offsetof(struct hk_apple9_storage_descriptor, pbe)), 5);
   nir_def *data = intr->src[3].ssa;
   nir_alu_type type = nir_intrinsic_src_type(intr);
   nir_alu_type base = nir_alu_type_get_base_type(type);
   if (base == nir_type_invalid)
      base = nir_type_uint;
   if (data->bit_size != 16 && data->bit_size != 32) {
      data = base == nir_type_float ? nir_f2f32(b, data) :
             base == nir_type_int   ? nir_i2i32(b, data) : nir_u2u32(b, data);
   }
   type = base | data->bit_size;
   bool helpers = b->shader->info.stage == MESA_SHADER_FRAGMENT &&
                  b->shader->info.fs.uses_sample_shading;
   if (helpers)
      nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));
   /* Sparse residency buffers map zero_bo read-only on unbound pages, which a
    * native PBE write cannot soft-fault; test the buffer residency table
    * (a zero table means non-sparse), skipping OOB lanes first. */
   if (residency) {
      nir_push_if(b, in_bounds);
      nir_push_if(b, nir_ieq_imm(b,
         hk_apple9_storage_buffer_residency_code(b, desc, x), 0));
   }
   nir_image_store_apple9_agx(b, index, xyz, nir_pad_vector_imm_int(b, data, 0, 4),
      .access = access, .src_type = type,
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false);
   if (residency) {
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   if (helpers)
      nir_pop_if(b, NULL);

   if (access & (ACCESS_COHERENT | ACCESS_VOLATILE))
      nir_fence_image_write_apple9_agx(b);
}

/* Residency code of texel x of a buffer view, from the table of its buffer's
 * pages (see hk_apple9_texel_metadata::residency). A zero table is resident.
 */
static nir_def *
hk_apple9_buffer_residency_lookup(nir_builder *b, nir_def *table,
                                  nir_def *base, nir_def *elements,
                                  nir_def *size_B, nir_def *x)
{
   nir_def *code;
   nir_push_if(b, nir_ine_imm(b, table, 0));
   {
      x = nir_umin(b, x, nir_iadd_imm(b, nir_umax_imm(b, elements, 1), -1));
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, x, size_B)));
      nir_def *entry = nir_iadd(b, table,
         nir_ushr_imm(b, addr, util_logbase2(AIL_PAGESIZE)));
      nir_def *word = nir_load_global_constant(b, 1, 32,
         nir_iand_imm(b, entry, ~3ull), .align_mul = 4);
      nir_def *byte = nir_iand_imm(b, nir_ushr(b, word,
         nir_imul_imm(b, nir_u2u32(b, nir_iand_imm(b, entry, 3)), 8)), 0xff);
      code = nir_b2i32(b, nir_ieq_imm(b, byte, 0));
   }
   nir_push_else(b, NULL);
   nir_def *resident = nir_imm_int(b, 0);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, code, resident);
}

static nir_def *
hk_apple9_buffer_residency_code(nir_builder *b, nir_def *meta, nir_def *x)
{
   nir_def *table = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, residency)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *base = nir_load_global_constant(b, 1, 64, meta,
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *fields = nir_load_global_constant(b, 2, 32,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, elements)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   return hk_apple9_buffer_residency_lookup(b, table, base,
      nir_channel(b, fields, 0), nir_channel(b, fields, 1), x);
}

static nir_def *
hk_apple9_storage_buffer_residency_code(nir_builder *b, nir_def *desc,
                                        nir_def *x)
{
   nir_def *table = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, desc,
         offsetof(struct hk_apple9_storage_descriptor, buf_residency)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, desc,
         offsetof(struct hk_apple9_storage_descriptor, buf.base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *elements = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, desc,
         offsetof(struct hk_apple9_storage_descriptor, buf.elements)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *size_B = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, desc,
         offsetof(struct hk_apple9_storage_descriptor, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   return hk_apple9_buffer_residency_lookup(b, table, base, elements, size_B, x);
}

static bool
hk_apple9_lower_sparse_tex(nir_builder *b, nir_tex_instr *tex,
                           UNUSED void *data)
{
   /* Images use hardware sparse texture ops; only sparse texel buffers consult
    * a software residency table. */
   if (!tex->is_sparse || tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *code = nir_imm_int(b, 0);
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle >= 0) {
      nir_def *meta = nir_iadd_imm(b,
         nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa),
         offsetof(struct hk_apple9_texel_descriptor, metadata));
      nir_def *coord =
         tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
      code = hk_apple9_buffer_residency_code(b, meta,
                                             nir_u2u32(b, nir_channel(b, coord, 0)));
   }

   tex->is_sparse = false;
   tex->def.num_components--;
   b->cursor = nir_after_instr(&tex->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < tex->def.num_components; ++c)
      comps[c] = nir_get_scalar(&tex->def, c);
   comps[tex->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, tex->def.num_components + 1);
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

static bool
hk_apple9_lower_sparse_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                                 UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_is_sparse_texels_resident:
      /* Hardware sparse ops return bit 0 == 0 for resident, matching buffer
       * residency codes (0 resident, 1 non-resident). */
      nir_def_replace(&intr->def,
                      nir_ieq_imm(b, nir_iand_imm(b, intr->src[0].ssa, 1), 0));
      return true;
   case nir_intrinsic_sparse_residency_code_and:
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;
   case nir_intrinsic_bindless_image_sparse_load:
      if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
         return false;
      break;
   default:
      return false;
   }

   nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *code = hk_apple9_storage_buffer_residency_code(b, desc,
      nir_u2u32(b, nir_channel(b, intr->src[1].ssa, 0)));

   intr->intrinsic = nir_intrinsic_bindless_image_load;
   intr->num_components--;
   intr->def.num_components--;
   b->cursor = nir_after_instr(&intr->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < intr->def.num_components; ++c)
      comps[c] = nir_get_scalar(&intr->def, c);
   comps[intr->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, intr->def.num_components + 1);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

bool
hk_apple9_lower_sparse_residency(nir_shader *nir)
{
   bool progress = nir_shader_tex_pass(nir, hk_apple9_lower_sparse_tex,
                                       nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(nir, hk_apple9_lower_sparse_intrinsic,
                                          nir_metadata_none, NULL);
   return progress;
}

struct hk_apple9_storage_lowering {
   /* Test residency before native stores to sparse residency buffers. */
   bool buffer_residency;
   /* Promote native image/buffer stores & loads to coherent(device) when
    * global-memory image atomics coexist on the same resource class. */
   bool has_image_atomics;
   bool has_buf_atomics;
};

static void
hk_apple9_scan_image_atomics(const nir_shader *nir,
                             struct hk_apple9_storage_lowering *lower)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_bindless_image_atomic &&
                intr->intrinsic != nir_intrinsic_bindless_image_atomic_swap)
               continue;
            if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF)
               lower->has_buf_atomics = true;
            else
               lower->has_image_atomics = true;
         }
      }
   }
}

/* Unified hardware lowering for all typed and formatless storage image and
 * storage texel buffer loads/stores on Apple9: every load goes through
 * non-reorderable TEXTURE_LOAD / TEXTURE_LOAD_MS (image_load_apple9_agx) and
 * every store goes through native PBE IMAGE_WRITE (image_store_apple9_agx),
 * with hardware memory_coherence_device bits and lazy coalesced PBE/texture
 * fences. */
static bool
hk_apple9_lower_storage_image(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   struct hk_apple9_storage_lowering *lower = data;
   bool sparse = intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
   bool load = sparse || intr->intrinsic == nir_intrinsic_bindless_image_load;
   if (!load && intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
   case GLSL_SAMPLER_DIM_MS:
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *coord = intr->src[1].ssa;
   enum gl_access_qualifier access = nir_intrinsic_access(intr);
   if (dim == GLSL_SAMPLER_DIM_BUF ? lower->has_buf_atomics
                                   : lower->has_image_atomics)
      access |= ACCESS_COHERENT;

   if (load) {
      /* Preserve program-order guarantees of image_load_apple9_agx even when
       * nir_opt_access annotated the original deref with ACCESS_CAN_REORDER. */
      nir_def *res = hk_apple9_texture_load(b, intr, desc, coord, sparse,
                                            access & ~ACCESS_CAN_REORDER);
      nir_def_replace(&intr->def, res);
   } else if (dim == GLSL_SAMPLER_DIM_BUF) {
      hk_apple9_native_buffer_store(b, intr, desc, nir_channel(b, coord, 0),
                                    lower->buffer_residency, access);
      nir_instr_remove(&intr->instr);
   } else {
      hk_apple9_native_store(b, intr, desc, coord, access);
      nir_instr_remove(&intr->instr);
   }

   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Lower storage image and storage texel buffer size/samples queries directly
 * from hk_apple9_storage_descriptor::extent (offset 32; buf.elements aliases
 * extent[0]). This replaces Apple8's libagx_txs unpacking on storage views. */
static bool
hk_apple9_lower_storage_queries(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_bindless_image_size &&
       intr->intrinsic != nir_intrinsic_bindless_image_samples)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *res;
   if (intr->intrinsic == nir_intrinsic_bindless_image_samples) {
      res = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, desc,
            offsetof(struct hk_apple9_storage_descriptor, extent[3])),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   } else {
      unsigned comps = intr->def.num_components;
      res = nir_load_global_constant(b, comps, 32,
         nir_iadd_imm(b, desc,
            offsetof(struct hk_apple9_storage_descriptor, extent)),
         .align_mul = 16, .access = ACCESS_UNIFORM_SAFE_AGX);
      if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE &&
          nir_intrinsic_image_array(intr) && comps == 3) {
         res = nir_vec3(b, nir_channel(b, res, 0), nir_channel(b, res, 1),
                        nir_udiv_imm(b, nir_channel(b, res, 2), 6));
      }
   }
   nir_def_replace(&intr->def, res);
   return true;
}

struct hk_lazy_fence_state {
   bool pbe;
   bool tex;
};

static bool
hk_apple9_addr_is_printf_buf(const nir_def *def, unsigned depth)
{
   if (!def || depth > 4)
      return false;
   const nir_instr *instr = nir_def_instr((nir_def *)def);
   if (instr->type == nir_instr_type_intrinsic) {
      return nir_instr_as_intrinsic((nir_instr *)instr)->intrinsic ==
             nir_intrinsic_load_printf_buffer_address;
   }
   if (instr->type == nir_instr_type_alu) {
      const nir_alu_instr *alu = nir_instr_as_alu((nir_instr *)instr);
      if (alu->op == nir_op_iadd) {
         return hk_apple9_addr_is_printf_buf(alu->src[0].src.ssa, depth + 1) ||
                hk_apple9_addr_is_printf_buf(alu->src[1].src.ssa, depth + 1);
      }
      if (alu->op == nir_op_mov)
         return hk_apple9_addr_is_printf_buf(alu->src[0].src.ssa, depth + 1);
   }
   return false;
}

static struct hk_lazy_fence_state
hk_apple9_instr_fence_need(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_tex) {
      if (!nir_tex_instr_is_query(nir_instr_as_tex((nir_instr *)instr)))
         return (struct hk_lazy_fence_state){.pbe = true, .tex = true};
      return (struct hk_lazy_fence_state){0};
   }
   if (instr->type != nir_instr_type_intrinsic)
      return (struct hk_lazy_fence_state){0};
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic((nir_instr *)instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_image_load_apple9_agx:
      return (struct hk_lazy_fence_state){.pbe = true, .tex = true};
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_2x32:
      return (struct hk_lazy_fence_state){.pbe = true};
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_global_2x32:
      /* Potentially aliasing global fallback stores must not race with an
       * earlier native PBE image store (WAW); ignore libcl printf stores. */
      if (hk_apple9_addr_is_printf_buf(intr->src[1].ssa, 0))
         return (struct hk_lazy_fence_state){0};
      return (struct hk_lazy_fence_state){.pbe = true};
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_global_atomic_2x32:
   case nir_intrinsic_global_atomic_swap_2x32:
      if (hk_apple9_addr_is_printf_buf(intr->src[0].ssa, 0))
         return (struct hk_lazy_fence_state){0};
      return (struct hk_lazy_fence_state){.pbe = true};
   case nir_intrinsic_barrier: {
      unsigned modes = nir_intrinsic_memory_modes(intr);
      unsigned scope = nir_intrinsic_memory_scope(intr);
      if (scope == SCOPE_NONE ||
          !(modes & (nir_var_image | nir_var_mem_global | nir_var_mem_ssbo)))
         return (struct hk_lazy_fence_state){0};
      return (struct hk_lazy_fence_state){.pbe = true};
   }
   default:
      return (struct hk_lazy_fence_state){0};
   }
}

static bool
hk_apple9_cf_list_has_native_image_store(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_image_store_apple9_agx)
               return true;
         }
      }
   }
   return false;
}

static struct hk_lazy_fence_state
hk_apple9_cf_list_fence_need(struct exec_list *cf_list)
{
   struct hk_lazy_fence_state need = {0};
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            struct hk_lazy_fence_state n = hk_apple9_instr_fence_need(instr);
            need.pbe |= n.pbe;
            need.tex |= n.tex;
            if (need.pbe && need.tex)
               return need;
         }
      }
   }
   return need;
}

static void
hk_apple9_emit_lazy_fence(nir_cursor cursor, bool flush_pbe, bool flush_tex)
{
   if (!flush_pbe && !flush_tex)
      return;
   nir_builder b = nir_builder_at(cursor);
   if (flush_pbe)
      nir_fence_image_write_apple9_agx(&b);
   if (flush_tex)
      nir_fence_tex_invalidate_apple9_agx(&b);
}

/* Lazily coalesce PBE drain (0x0650) + texture invalidate (0x06d0) fences:
 * back-to-back native image stores overlap without per-store barriers, and
 * PBE/tex dirty bits are tracked separately across straight-line code,
 * branches, barriers, global stores/loads/atomics, and loop break/continue
 * exits. Workgroup image barriers (0x0e51/0x0ed1 or 0x0e50/0x0ed0) already
 * drain PBE + invalidate tex cache. */
static struct hk_lazy_fence_state
hk_apple9_fence_cf_list(struct exec_list *cf_list,
                        struct hk_lazy_fence_state st,
                        struct hk_lazy_fence_state in_loop_need,
                        struct hk_lazy_fence_state *break_out)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_image_store_apple9_agx) {
                  st.pbe = true;
                  st.tex = true;
                  continue;
               }
               if (intr->intrinsic == nir_intrinsic_fence_image_write_apple9_agx) {
                  st.pbe = false;
                  continue;
               }
               if (intr->intrinsic == nir_intrinsic_fence_tex_invalidate_apple9_agx) {
                  st.tex = false;
                  continue;
               }
               if (intr->intrinsic == nir_intrinsic_barrier) {
                  unsigned modes = nir_intrinsic_memory_modes(intr);
                  unsigned scope = nir_intrinsic_memory_scope(intr);
                  if (scope == SCOPE_WORKGROUP && (modes & nir_var_image)) {
                     /* Workgroup texture barrier (0x0e51/0x0e50) drains PBE
                      * and invalidates texture cache directly. */
                     st.pbe = false;
                     st.tex = false;
                     continue;
                  }
                  if (st.pbe && scope != SCOPE_NONE &&
                      (modes & (nir_var_image | nir_var_mem_global |
                                nir_var_mem_ssbo))) {
                     /* Drain PBE writes before the barrier; leave st.tex true
                      * so non-image barriers (e.g. memoryBarrierBuffer) still
                      * invalidate the texture cache before a subsequent read.
                      * Image barriers already have an explicit
                      * fence_tex_invalidate right after them, which clears
                      * st.tex on the next iteration. */
                     hk_apple9_emit_lazy_fence(nir_before_instr(instr), true, false);
                     st.pbe = false;
                  }
                  continue;
               }
            }
            if (instr->type == nir_instr_type_jump) {
               nir_jump_instr *jump = nir_instr_as_jump(instr);
               if (jump->type == nir_jump_continue) {
                  bool fp = st.pbe && in_loop_need.pbe;
                  bool ft = st.tex && in_loop_need.tex;
                  if (fp || ft) {
                     hk_apple9_emit_lazy_fence(nir_before_instr(instr), fp, ft);
                     if (fp) st.pbe = false;
                     if (ft) st.tex = false;
                  }
               } else if (jump->type == nir_jump_break) {
                  if (break_out) {
                     break_out->pbe |= st.pbe;
                     break_out->tex |= st.tex;
                  }
                  st.pbe = false;
                  st.tex = false;
               }
               continue;
            }
            struct hk_lazy_fence_state need = hk_apple9_instr_fence_need(instr);
            bool fp = st.pbe && need.pbe;
            bool ft = st.tex && need.tex;
            if (fp || ft) {
               hk_apple9_emit_lazy_fence(nir_before_instr(instr), fp, ft);
               if (fp) st.pbe = false;
               if (ft) st.tex = false;
            }
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (((intr->intrinsic == nir_intrinsic_store_global ||
                     intr->intrinsic == nir_intrinsic_store_global_2x32) &&
                    !hk_apple9_addr_is_printf_buf(intr->src[1].ssa, 0)) ||
                   ((intr->intrinsic == nir_intrinsic_global_atomic ||
                     intr->intrinsic == nir_intrinsic_global_atomic_swap ||
                     intr->intrinsic == nir_intrinsic_global_atomic_2x32 ||
                     intr->intrinsic == nir_intrinsic_global_atomic_swap_2x32) &&
                    !hk_apple9_addr_is_printf_buf(intr->src[0].ssa, 0)))
                  st.tex = true;
            }
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         struct hk_lazy_fence_state then_need =
            hk_apple9_cf_list_fence_need(&nif->then_list);
         struct hk_lazy_fence_state else_need =
            hk_apple9_cf_list_fence_need(&nif->else_list);
         bool fp = st.pbe && then_need.pbe && else_need.pbe;
         bool ft = st.tex && then_need.tex && else_need.tex;
         if (fp || ft) {
            hk_apple9_emit_lazy_fence(nir_before_cf_node(node), fp, ft);
            if (fp) st.pbe = false;
            if (ft) st.tex = false;
         }
         struct hk_lazy_fence_state d_then =
            hk_apple9_fence_cf_list(&nif->then_list, st, in_loop_need, break_out);
         struct hk_lazy_fence_state d_else =
            hk_apple9_fence_cf_list(&nif->else_list, st, in_loop_need, break_out);
         st.pbe = d_then.pbe || d_else.pbe;
         st.tex = d_then.tex || d_else.tex;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         bool loop_stores = hk_apple9_cf_list_has_native_image_store(&loop->body);
         struct hk_lazy_fence_state loop_need =
            hk_apple9_cf_list_fence_need(&loop->body);
         bool fp = st.pbe && loop_need.pbe;
         bool ft = st.tex && loop_need.tex;
         if (fp || ft) {
            hk_apple9_emit_lazy_fence(nir_before_cf_node(node), fp, ft);
            if (fp) st.pbe = false;
            if (ft) st.tex = false;
         }
         struct hk_lazy_fence_state nested_need = {
            .pbe = loop_need.pbe || in_loop_need.pbe,
            .tex = loop_need.tex || in_loop_need.tex,
         };
         struct hk_lazy_fence_state loop_break = {0};
         struct hk_lazy_fence_state d_body =
            hk_apple9_fence_cf_list(&loop->body,
                                    (struct hk_lazy_fence_state){0},
                                    nested_need, &loop_break);
         bool bfp = d_body.pbe && loop_need.pbe;
         bool bft = d_body.tex && loop_need.tex;
         if (bfp || bft) {
            nir_block *last = nir_loop_last_block(loop);
            if (!nir_block_ends_in_jump(last))
               hk_apple9_emit_lazy_fence(nir_after_block(last), bfp, bft);
            if (bfp) d_body.pbe = false;
            if (bft) d_body.tex = false;
         }
         st.pbe = st.pbe || d_body.pbe || loop_break.pbe ||
                  (loop_stores && !loop_need.pbe);
         st.tex = st.tex || d_body.tex || loop_break.tex ||
                  (loop_stores && !loop_need.tex);
         break;
      }
      default:
         break;
      }
   }
   return st;
}

static void
hk_apple9_insert_lazy_image_fences(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!hk_apple9_cf_list_has_native_image_store(&impl->body))
      return;
   hk_apple9_fence_cf_list(&impl->body, (struct hk_lazy_fence_state){0},
                           (struct hk_lazy_fence_state){0}, NULL);
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* When every storage image access has become global memory, synchronize that
 * memory domain. Retain workgroup nir_var_image barriers when native PBE
 * stores are present so the compiler emits 0x0e51/0x0ed1 (or 0x0e50/0x0ed0).
 * Sampled images and uniform texel buffers need not observe writes from the
 * same dispatch, and descriptor queries only read metadata, so neither
 * requires texture fences.
 */
static void
hk_apple9_lower_image_barriers(nir_shader *nir, bool has_global_image_fallback)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool has_native_store = false;
   bool has_tex_read = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex) {
            if (!nir_tex_instr_is_query(nir_instr_as_tex(instr)))
               has_tex_read = true;
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_image_store_apple9_agx)
            has_native_store = true;
         else if (intr->intrinsic == nir_intrinsic_image_load_apple9_agx)
            has_tex_read = true;
         if (nir_intrinsic_has_image_dim(intr) &&
             intr->intrinsic != nir_intrinsic_image_store_apple9_agx &&
             intr->intrinsic != nir_intrinsic_image_load_apple9_agx &&
             intr->intrinsic != nir_intrinsic_image_size &&
             intr->intrinsic != nir_intrinsic_bindless_image_size &&
             intr->intrinsic != nir_intrinsic_image_samples &&
             intr->intrinsic != nir_intrinsic_bindless_image_samples &&
             intr->intrinsic != nir_intrinsic_image_levels &&
             intr->intrinsic != nir_intrinsic_bindless_image_levels)
            return;
      }
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_barrier) {
            unsigned modes = nir_intrinsic_memory_modes(intr);
            if (modes & nir_var_image) {
               unsigned mem_scope = nir_intrinsic_memory_scope(intr);
               if (has_native_store && mem_scope == SCOPE_WORKGROUP) {
                  /* Native PBE workgroup texture barrier lowers directly to
                   * 0x0e51/0x0ed1 (or 0x0e50/0x0ed0). Only add global sync
                   * when global-memory image fallbacks (e.g. image atomics)
                   * coexist in the shader. */
                  unsigned next_modes = modes;
                  if (has_global_image_fallback)
                     next_modes |= nir_var_mem_global;
                  nir_intrinsic_set_memory_modes(intr, next_modes);
               } else {
                  nir_intrinsic_set_memory_modes(intr,
                     (modes & ~nir_var_image) | nir_var_mem_global);
                  /* All storage loads use the texture unit: invalidate the
                   * texture cache after device/subgroup image barriers so
                   * readers observe drained PBE writes and image atomics. */
                  if (has_tex_read && mem_scope != SCOPE_NONE) {
                     nir_builder b = nir_builder_at(nir_after_instr(instr));
                     nir_fence_tex_invalidate_apple9_agx(&b);
                  }
               }
            }
         } else if (intr->intrinsic == nir_intrinsic_fence_pbe_to_tex_agx ||
                    intr->intrinsic == nir_intrinsic_fence_mem_to_tex_agx) {
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_barrier(&b, .execution_scope = SCOPE_NONE,
                        .memory_scope = SCOPE_DEVICE,
                        .memory_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,
                        .memory_modes = nir_var_mem_global);
            if (has_tex_read)
               nir_fence_tex_invalidate_apple9_agx(&b);
            nir_instr_remove(instr);
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* storageInputOutput16: 16-bit stage inputs and outputs are carried as 32-bit
 * values at every interface, converting exactly at the access (f16 <-> f32,
 * 16-bit integers truncated on load and extended on store), as both sides of an
 * interface are widened alike. Slots and components are unchanged.
 */
static bool
hk_lower_io_16bit_to_32bit(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_primitive_input:
   case nir_intrinsic_load_input_vertex:
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output: {
      if (intr->def.bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_dest_type(intr);
      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, nir_alu_type_get_base_type(type) | 32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *narrow = nir_alu_type_get_base_type(type) == nir_type_float
                           ? nir_f2f16(b, &intr->def)
                           : nir_u2u16(b, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, narrow);
      return true;
   }
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output: {
      nir_def *value = intr->src[0].ssa;
      if (value->bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_src_type(intr);
      nir_alu_type base = nir_alu_type_get_base_type(type);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *wide = base == nir_type_float ? nir_f2f32(b, value)
                      : base == nir_type_int  ? nir_i2i32(b, value)
                                              : nir_u2u32(b, value);
      nir_src_rewrite(&intr->src[0], wide);
      nir_intrinsic_set_src_type(intr, base | 32);
      return true;
   }
   default:
      return false;
   }
}

/* Mesh primitives rasterize with their per-primitive builtins as flat varyings
 * of each vertex. User per-primitive varyings are read from the records.
 */
static bool
hk_lower_per_primitive_input(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   bool native_broadcast = data && *(const bool *)data;
   if (!native_broadcast && hk_mesh_varying_in_memory(sem.location)) {
      nir_def *record = nir_load_input(
         b, 1, 32, nir_imm_int(b, 0), .range = 1, .dest_type = nir_type_uint32,
         .io_semantics.location = HK_MESH_PRIMITIVE_SLOT,
         .io_semantics.num_slots = 1);
      b->shader->info.inputs_read |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);

      unsigned params_offset = hk_root_descriptor_offset(draw.mesh_params);
      nir_def *params = nir_load_ubo(
         b, 1, 64, nir_imm_int(b, 0), nir_imm_int(b, params_offset),
         .align_mul = 8, .range_base = params_offset, .range = 8);
      nir_def *outputs = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, params, offsetof(struct libagx_mesh_params, outputs)),
         .align_mul = 8);
      nir_def *varying = nir_load_global(
         b, 1, 16,
         nir_iadd_imm(b, params,
                      offsetof(struct libagx_mesh_params, primitive_varying_B) +
                         (sem.location - VARYING_SLOT_VAR0) * 2),
         .align_mul = 2);
      nir_def *offset = nir_iadd(
         b, nir_iadd(b, record, nir_u2u32(b, varying)),
         nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, 16),
                      nir_intrinsic_component(intr) * 4));
      nir_def *value = nir_load_global(
         b, intr->def.num_components, 32,
         nir_iadd(b, outputs, nir_u2u64(b, offset)), .align_mul = 4);
      if (intr->def.bit_size != 32) {
         value = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
                       nir_type_float
                    ? nir_f2fN(b, value, intr->def.bit_size)
                    : nir_u2uN(b, value, intr->def.bit_size);
      }
      nir_def_replace(&intr->def, value);
      return true;
   }

   nir_def *value = nir_load_input(
      b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr), .range = sem.num_slots,
      .component = nir_intrinsic_component(intr),
      .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
   nir_def_replace(&intr->def, value);
   return true;
}

/* With hk_apple9_d3d_mip_rounding (driconf, set for vkd3d-proton), add the
 * sampler's nudge to explicit LODs, so a NEAREST mip selection exactly
 * halfway between levels rounds up as in D3D. The nudge is zero for other
 * mipmap modes, see hk_CreateSampler.
 */
static bool
hk_lower_d3d_txl_rounding(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_txl || !nir_tex_instr_need_sampler(tex))
      return false;

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (lod_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *nudge = nir_build_texture_query(b, tex, nir_texop_lod_bias, 1,
                                            nir_type_float16, false, false);
   nir_def_as_tex(nudge)->backend_flags |= HK_TEX_TXL_NUDGE;

   nir_def *lod = tex->src[lod_idx].src.ssa;
   nir_src_rewrite(&tex->src[lod_idx].src,
                   nir_fadd(b, lod, nir_f2fN(b, nudge, lod->bit_size)));
   return true;
}

/* With hk_apple9_arrayed_2d_views (driconf, set for vkd3d-proton), sampled
 * 1D and 2D views are one-layer 2D arrays on Apple9 (see pack_texture), since
 * its texture instructions must match the descriptor's arrayness. Sample
 * every non-array 2D texture, 1D having been lowered to 2D, as layer 0 of an
 * array. Layer 0 of a one-layer view is the view itself, and an array
 * declaration bound to a 2D view now honours its layer and level.
 */
static bool
hk_apple9_lower_tex_2d_as_array(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->is_array || (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                         tex->sampler_dim != GLSL_SAMPLER_DIM_MS))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   int c = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (c >= 0) {
      nir_def *coord = tex->src[c].src.ssa;
      nir_def *layer = nir_tex_instr_src_type(tex, c) == nir_type_float
                          ? nir_imm_floatN_t(b, 0.0, coord->bit_size)
                          : nir_imm_intN_t(b, 0, coord->bit_size);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < coord->num_components; ++i)
         comps[i] = nir_channel(b, coord, i);
      comps[coord->num_components] = layer;
      nir_src_rewrite(&tex->src[c].src, nir_vec(b, comps, coord->num_components + 1));
      tex->coord_components++;
   }
   tex->is_array = true;
   tex->backend_flags |= AGX_TEXTURE_FLAG_ARRAYED_2D;

   if (tex->op == nir_texop_txs) {
      /* Array sizes report the layer count too; return the 2D size. */
      unsigned old = tex->def.num_components;
      tex->def.num_components++;
      b->cursor = nir_after_instr(&tex->instr);
      nir_def *size = nir_trim_vector(b, &tex->def, old);
      nir_def_rewrite_uses_after(&tex->def, size);
   }
   return true;
}

static void hk_apple9_cap_loop_unroll_l1i(nir_shader *nir);

void
hk_lower_nir(struct hk_device *dev, nir_shader *nir,
             const struct vk_pipeline_robustness_state *rs, bool is_multiview,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             enum hk_feature_key features)
{
   if (HK_PERF(dev, NOROBUST)) {
      rs = &vk_robustness_disabled;
   }

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
               &(nir_input_attachment_options){
                  .use_view_id_for_layer = is_multiview,
               });

      NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
               NULL);
      NIR_PASS(_, nir, nir_lower_wpos_center);
   }

   /* XXX: should be last geometry stage, how do I get to that? */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (is_multiview)
         hk_lower_multiview(nir);
   }

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_patch_vertices,
               nir->info.tess.tcs_vertices_out, NULL);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   /* Turn cache flushes into image coherency bits while we still have derefs */
   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_robust_access, should_lower_robust, NULL);

   /* We must do early lowering before hk_nir_lower_descriptors, since this will
    * create lod_bias instructions.
    */
   /* Apple9 bounds-checks buffer images in software against their element
    * count (hk_apple9_lower_storage_image). */
   NIR_PASS(_, nir, agx_nir_lower_texture_early, true /* support_lod_bias */,
            true /* apple9 */);
   if (hk_device_physical(dev)->apple9_d3d_mip_rounding)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_lower_d3d_txl_rounding,
               nir_metadata_control_flow, NULL);
   if (hk_device_physical(dev)->apple9_layered_2d)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_apple9_lower_tex_2d_as_array,
               nir_metadata_control_flow, NULL);

   /* Apple9 pays for the clamp only when the application enables the
    * extension. */
   if (dev->vk.enabled_features.minLod) {
      NIR_PASS(_, nir, agx_nir_lower_image_view_min_lod);
   }

   NIR_PASS(_, nir, hk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts, true);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_buffer_addr_format(rs->uniform_buffers));

   /* Before inserting bounds checks, we want to do a fair bit of optimization.
    * lower_load_global_constant_offset_instr has special optimizations for
    * constant offsets, so we want as many offsets to be constant as possible.
    */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   /* Apple9 vectorizes before the buffer base and offset become one 64-bit
    * address. Afterwards the offset is hidden inside u2u64, where adjacent
    * fields at x + c are no longer provably adjacent, and every field of a
    * dynamically indexed struct becomes a separate scalar load.
    */
   /* Explicit I/O builds each access's base and offset afresh. The
    * vectorizer only merges accesses sharing those definitions. */
   NIR_PASS(_, nir, nir_opt_cse);
   if (agx_apple9_debug_enabled(AGX_A9_DBG_VECTORIZE) &&
       nir->info.stage == MESA_SHADER_FRAGMENT)
      nir_print_shader(nir, stderr);
   hk_apple9_vectorize_global(nir, agx_has_soft_fault(&dev->dev));

   nir_move_options opts = nir_move_load_ssbo | nir_move_load_ubo;
   NIR_PASS(progress, nir, nir_opt_sink, opts);
   NIR_PASS(progress, nir, nir_opt_move, opts);

   NIR_PASS(_, nir, hk_lower_bounded_global_loads,
            agx_has_soft_fault(&dev->dev), true);

   assert(nir->info.shared_size == 0);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.stage == MESA_SHADER_TASK ||
       nir->info.stage == MESA_SHADER_MESH) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_mem_task_payload, shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16B so we can write whole vec4s. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* TODO: we can do indirect VS output */
   nir_variable_mode lower_indirect_modes = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      /* Apple9 interpolation packets address a fixed coefficient. Expand
       * dynamic fragment input indexing before lowering away the array derefs. */
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;
   else if (nir->info.stage == MESA_SHADER_VERTEX)
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;

   /* Native mesh resolves indirect per-vertex/per-primitive output-array slots
    * strictly on its own native clone in agx_nir_lower_apple9_native_mesh,
    * keeping emulated fallback NIR untouched and dynamic vertex/primitive
    * indexing free of if-else trees. */
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            lower_indirect_modes, UINT32_MAX);

   /* nir_lower_io needs constant indices into compact arrays. Tess levels are
    * indexed dynamically (gl_TessLevelOuter[gl_InvocationID]), so turn them
    * into vectors and lower the component indexing, as RADV does.
    */
   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_tess_level_array_vars_to_vec);
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec,
               nir_var_shader_in | nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   }

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_lower_io_16bit_to_32bit,
            nir_metadata_control_flow, NULL);

   /* The qualified index profile publishes Position only. It cannot consume
    * per-primitive fragment inputs, so keep their original early lowering.
    * Only the broader research P1 interface defers these inputs to linking.
    * Delaying them for every enabled device would change excluded shaders. */
   bool native_primitive_inputs =
      hk_apple9_native_mesh_available(hk_device_physical(dev)) &&
      hk_apple9_native_mesh_enabled();
   if (nir->info.stage == MESA_SHADER_FRAGMENT && !native_primitive_inputs) {
      /* The emulated path reads records; research P1 uses flat channels. */
      bool native_broadcast = false;
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_lower_per_primitive_input, nir_metadata_control_flow,
               &native_broadcast);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_apple9_lower_layer_viewport_fs,
               nir_metadata_control_flow, NULL);
   }

   /* Buffer residency lookups add control flow. */
   NIR_PASS(_, nir, hk_apple9_lower_sparse_residency);
   /* Texel-buffer bounds and RGB32 checks add control flow. */
   bool texel_buf_global = false;
   NIR_PASS(texel_buf_global, nir, nir_shader_tex_pass,
            hk_apple9_lower_texel_buffer, nir_metadata_none, NULL);
   struct hk_apple9_storage_lowering lower = {
      .buffer_residency = (features & HK_FEAT_STORE_BUFFER_RESIDENCY) != 0,
   };
   hk_apple9_scan_image_atomics(nir, &lower);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_storage_image,
            nir_metadata_none, &lower);

   bool buf_addr_global = false;
   NIR_PASS(buf_addr_global, nir, nir_shader_intrinsics_pass,
            hk_apple9_lower_buffer_address, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_storage_queries,
            nir_metadata_control_flow, NULL);
   bool lower_images_global = false;
   NIR_PASS(lower_images_global, nir, agx_nir_lower_images);
   /* Sampled views' queries (AGX_TEXTURE_FLAG_METADATA_QUERY) are answered
    * from texture metadata by the Apple9 backend; uniform texel buffers use
    * agx_nir_lower_texture_queries on their prefix. */
   NIR_PASS(_, nir, agx_nir_lower_texture_queries);

   /* Keep radians until Apple9 range reduction. The common preprocessor
    * otherwise replaces sine/cosine with the older fsin_agx operation. */
   NIR_PASS(_, nir, agx_nir_lower_apple9_math);
   hk_apple9_cap_loop_unroll_l1i(nir);
   agx_preprocess_nir(nir);
   const bool has_global_image_fallback =
      texel_buf_global || buf_addr_global || lower_images_global;
   hk_apple9_lower_image_barriers(nir, has_global_image_fallback);
   hk_apple9_insert_lazy_image_fences(nir);

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);
}

static nir_intrinsic_instr *
hk_apple9_handle_intrinsic(nir_tex_instr *tex, unsigned sampler)
{
   int src = nir_tex_instr_src_index(
      tex, sampler ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle);
   if (src < 0)
      return NULL;
   nir_instr *instr = nir_def_instr(tex->src[src].src.ssa);
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == (sampler ? nir_intrinsic_bindless_sampler_agx
                                      : nir_intrinsic_bindless_image_agx)
             ? intr : NULL;
}

/* Address a descriptor relative to the USC aperture that holds G16
 * descriptor memory, see hk_apple9_image_part. */
static void
hk_apple9_use_descriptor_handle(nir_builder *b, nir_tex_instr *tex,
                                unsigned sampler)
{
   int src = nir_tex_instr_src_index(
      tex, sampler ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle);
   nir_intrinsic_instr *handle = hk_apple9_handle_intrinsic(tex, sampler);
   b->cursor = nir_before_instr(&tex->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   unsigned root = hk_root_descriptor_offset(sets) + set * sizeof(uint64_t);
   nir_def *base = nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, root),
                                .align_mul = 8, .range_base = root, .range = 4);
   /* Texture and sampler handles address their record part directly. */
   nir_def *offset = handle->src[0].ssa;
   nir_src_rewrite(&tex->src[src].src, nir_iadd(b, base, offset));
}


/* Which memory a shader reads, from its intrinsics: texture and image
 * operations read images; loads from any buffer, global or descriptor
 * memory and vertex attributes read buffers. Queries and stores do not
 * read. Unknown memory loads count as buffers. */
bool
hk_nir_reads_draw_id(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_draw_id)
               return true;
         }
      }
   }
   return false;
}

/* Recognize the address expressions produced by descriptor lowering. Stop at
 * memory loads: a descriptor pointer loaded from a root is not root memory.
 * Unknown arithmetic/phis conservatively keep the full push region. */
enum hk_root_address_kind {
   HK_ROOT_ADDRESS_OTHER,
   HK_ROOT_ADDRESS_CONSTANT,
   HK_ROOT_ADDRESS_ROOT,
   HK_ROOT_ADDRESS_UNKNOWN,
};

struct hk_root_address {
   enum hk_root_address_kind kind;
   uint64_t offset;
};

static struct hk_root_address
hk_root_address(nir_def *def, unsigned depth)
{
   if (depth > 32)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return (struct hk_root_address){
         HK_ROOT_ADDRESS_CONSTANT, nir_src_as_uint(nir_src_for_ssa(def))};

   if (instr->type == nir_instr_type_intrinsic) {
      return (struct hk_root_address){
         nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_root_agx
            ? HK_ROOT_ADDRESS_ROOT : HK_ROOT_ADDRESS_OTHER};
   }

   if (instr->type != nir_instr_type_alu)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   struct hk_root_address a = hk_root_address(alu->src[0].src.ssa, depth + 1);
   if (alu->op == nir_op_mov || alu->op == nir_op_u2u64)
      return a;

   unsigned n = nir_op_infos[alu->op].num_inputs;
   if (alu->op == nir_op_iadd) {
      struct hk_root_address b = hk_root_address(alu->src[1].src.ssa, depth + 1);
      if ((a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_ROOT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_ROOT)) {
         uint64_t offset = (a.offset + b.offset) & BITFIELD64_MASK(def->bit_size);
         return (struct hk_root_address){MAX2(a.kind, b.kind), offset};
      }
      return (struct hk_root_address){
         a.kind >= HK_ROOT_ADDRESS_ROOT || b.kind >= HK_ROOT_ADDRESS_ROOT
            ? HK_ROOT_ADDRESS_UNKNOWN : HK_ROOT_ADDRESS_OTHER};
   }

   bool root = a.kind >= HK_ROOT_ADDRESS_ROOT;
   for (unsigned i = 1; i < n; ++i)
      root |= hk_root_address(alu->src[i].src.ssa, depth + 1).kind >=
              HK_ROOT_ADDRESS_ROOT;
   return (struct hk_root_address){root ? HK_ROOT_ADDRESS_UNKNOWN
                                       : HK_ROOT_ADDRESS_OTHER};
}

static uint64_t
hk_push_range_mask(uint64_t start, uint64_t size)
{
   static_assert(HK_MAX_PUSH_SIZE == 64 * sizeof(uint32_t));
   if (start >= HK_MAX_PUSH_SIZE || size > HK_MAX_PUSH_SIZE - start)
      return UINT64_MAX;
   unsigned first = start / 4;
   unsigned end = DIV_ROUND_UP(start + size, 4);
   return BITFIELD64_RANGE(first, end - first);
}

uint64_t
hk_nir_push_constant_mask(const nir_shader *nir)
{
   uint64_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            unsigned bytes = in->num_components * in->def.bit_size / 8;
            if (in->intrinsic == nir_intrinsic_load_push_constant) {
               if (!nir_src_is_const(in->src[0]))
                  return UINT64_MAX;
               mask |= hk_push_range_mask((uint64_t)nir_intrinsic_base(in) +
                                             nir_src_as_uint(in->src[0]), bytes);
            } else if (in->intrinsic == nir_intrinsic_load_deref &&
                       nir_deref_mode_is(nir_src_as_deref(in->src[0]),
                                         nir_var_mem_push_const)) {
               return UINT64_MAX;
            } else if (in->intrinsic == nir_intrinsic_load_global_constant ||
                       in->intrinsic == nir_intrinsic_load_global ||
                       in->intrinsic == nir_intrinsic_load_global_constant_offset) {
               struct hk_root_address a = hk_root_address(in->src[0].ssa, 0);
               if (in->intrinsic == nir_intrinsic_load_global_constant_offset &&
                   a.kind >= HK_ROOT_ADDRESS_ROOT) {
                  if (!nir_src_is_const(in->src[1]))
                     return UINT64_MAX;
                  a.offset += nir_src_as_uint(in->src[1]);
               }
               if (a.kind == HK_ROOT_ADDRESS_UNKNOWN)
                  return UINT64_MAX;
               if (a.kind != HK_ROOT_ADDRESS_ROOT)
                  continue;
               const uint64_t start = hk_root_descriptor_offset(push);
               const uint64_t end = start + HK_MAX_PUSH_SIZE;
               if (a.offset < end && bytes > start - MIN2(a.offset, start)) {
                  uint64_t lo = MAX2(a.offset, start);
                  uint64_t hi = MIN2(a.offset + bytes, end);
                  mask |= hk_push_range_mask(lo - start, hi - lo);
               }
            }
         }
      }
   }
   return mask;
}

#ifndef NDEBUG
/* hk_apple9_lower_roots replaces load_root_agx with a load of root_desc_addr
 * from UBO 0. Recover that identity on a clone so the final-NIR check sees
 * both indirect root loads and direct UBO 0 reads. Other loaded pointers
 * still stop the address walk. The NIR passed to the backend is untouched.
 */
static bool
hk_debug_restore_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_ubo)
      return false;

   bool *unknown = data;
   if (!nir_src_is_const(intr->src[0])) {
      *unknown = true;
      return false;
   }
   if (nir_src_as_uint(intr->src[0]) != 0)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *root = nir_load_root_agx(b);
   nir_def *value;
   if (intr->num_components == 1 && intr->def.bit_size == 64 &&
       nir_src_is_const(intr->src[1]) &&
       nir_src_as_uint(intr->src[1]) ==
          hk_root_descriptor_offset(root_desc_addr)) {
      value = root;
   } else {
      nir_def *offset = nir_u2u64(b, intr->src[1].ssa);
      value = nir_load_global_constant(
         b, intr->num_components, intr->def.bit_size,
         nir_iadd(b, root, offset),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr));
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* Geometry lowering introduces application-memory address phis and vector
 * expressions which the early root walk conservatively calls unknown. They
 * cannot refer to our root unless they depend on load_root_agx. Follow phis
 * with a visited set, but stop at memory loads just like the early analysis.
 */
struct hk_debug_root_walk {
   struct set *visited;
   bool found;
};

static bool
hk_debug_root_source(nir_src *src, void *data)
{
   struct hk_debug_root_walk *walk = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   bool found;
   _mesa_set_search_or_add(walk->visited, instr, &found);
   if (found)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      walk->found |= nir_instr_as_intrinsic(instr)->intrinsic ==
                     nir_intrinsic_load_root_agx;
   } else {
      nir_foreach_src(instr, hk_debug_root_source, walk);
   }
   return !walk->found;
}

static bool
hk_debug_exclude_non_root_load(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global &&
       intr->intrinsic != nir_intrinsic_load_global_constant &&
       intr->intrinsic != nir_intrinsic_load_global_constant_offset)
      return false;
   struct hk_debug_root_walk walk = {
      .visited = _mesa_pointer_set_create(NULL),
   };
   assert(walk.visited);
   hk_debug_root_source(&intr->src[0], &walk);
   _mesa_set_destroy(walk.visited, NULL);
   if (walk.found)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0], nir_imm_int64(b, 0));
   return true;
}

uint64_t
hk_nir_final_push_constant_mask(const nir_shader *nir)
{
   nir_shader *copy = nir_shader_clone(NULL, nir);
   assert(copy);
   bool unknown = false;
   NIR_PASS(_, copy, nir_shader_intrinsics_pass, hk_debug_restore_root,
            nir_metadata_control_flow, &unknown);
   /* Fold only the newly introduced integer conversions/adds, so direct
    * constant UBO offsets remain recognizable to the conservative walk. */
   NIR_PASS(_, copy, nir_opt_constant_folding);
   NIR_PASS(_, copy, nir_shader_intrinsics_pass, hk_debug_exclude_non_root_load,
            nir_metadata_control_flow, NULL);
   uint64_t mask = unknown ? UINT64_MAX : hk_nir_push_constant_mask(copy);
   ralloc_free(copy);
   return mask;
}

void
hk_assert_final_push_mask(const nir_shader *nir, const struct hk_shader *shader)
{
   const struct hk_api_shader *obj = shader->debug_owner;
   assert(obj);
   uint64_t final = hk_nir_final_push_constant_mask(nir);
   assert((final & ~obj->push_constant_mask) == 0 ||
          obj->push_constant_mask == UINT64_MAX);
}
#endif

uint8_t
hk_nir_pre_raster_reads(const nir_shader *nir)
{
   uint8_t reads = 0;
   if (nir->info.stage == MESA_SHADER_VERTEX && nir->info.inputs_read)
      reads |= HK_READS_BUFFERS;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (!nir_tex_instr_is_query(nir_instr_as_tex(instr)))
                  reads |= HK_READS_IMAGES;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_load:
            case nir_intrinsic_image_sparse_load:
            case nir_intrinsic_image_deref_load:
            case nir_intrinsic_image_deref_sparse_load:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_image_load_apple9_agx:
            case nir_intrinsic_load_input_attachment_coord:
               reads |= HK_READS_IMAGES;
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_ubo_vec4:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_2x32:
            case nir_intrinsic_load_global_bounded:
            case nir_intrinsic_load_global_constant_bounded:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_input:
            case nir_intrinsic_load_input_vertex:
            case nir_intrinsic_load_per_vertex_input:
               reads |= HK_READS_BUFFERS;
               break;
            case nir_intrinsic_load_deref: {
               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               if (nir_deref_mode_may_be(deref, nir_var_mem_ubo | nir_var_mem_ssbo |
                                                   nir_var_mem_global |
                                                   nir_var_shader_in))
                  reads |= HK_READS_BUFFERS;
               if (nir_deref_mode_may_be(deref, nir_var_image | nir_var_uniform))
                  reads |= HK_READS_IMAGES;
               break;
            }
            default:
               break;
            }
         }
      }
   }
   return reads;
}

bool
hk_apple9_texture_bindings(nir_shader *nir)
{
   /* Every sampled descriptor is read through its memory (a descriptor
    * handle): native records live in set memory, written at update time, so
    * draws and dispatches build no texture/sampler tables and update-after-
    * bind needs nothing extra. The handle and the descriptor-record reads are
    * uniform and hoisted to the preamble. */
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         for (unsigned sampler = 0; sampler < 2; ++sampler) {
            tex->backend_flags &= ~HK_APPLE9_TEX_DESCRIPTOR_HANDLE;
            if (nir_tex_instr_src_index(tex, sampler
                   ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle) < 0)
               continue;
            if (!hk_apple9_handle_intrinsic(tex, sampler))
               return false;
            hk_apple9_use_descriptor_handle(&b, tex, sampler);
         }
      }
   }
   /* Handle loads were added and sources removed: liveness is stale. */
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_control_flow);
   nir_opt_dce(nir);
   return true;
}

/* An early depth/stencil test updates the attachment before the fragment
 * shader runs, and the shader's input-attachment reads then observe this
 * fragment's own write. With a depth/stencil feedback loop, export the
 * interpolated depth at the end instead, so the test follows those reads. */
void
hk_apple9_late_zs(nir_shader *nir)
{
   if (nir->info.fs.early_fragment_tests ||
       (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_frag_coord_z(&b), nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .src_type = nir_type_float32);
   nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* Hardware bounds shader-exported depth only to [0, 1]. Clamp depth exports,
 * including the side-effect pass's synthetic export, to the draw's clamp
 * range from the root table, as native Metal fragment shaders do with their
 * near/far uniforms.
 *
 * A NEVER depth test skips the fragment shader, which would drop side
 * effects, and a NaN export passes ALWAYS. Such draws therefore test LESS
 * (hk_cmd_apple9.c) against an exported depth above the storable range, so
 * every sample fails depth and still takes its stencil depth-fail op. */
bool
hk_apple9_clamp_depth_export(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_DEPTH)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned offset = hk_root_descriptor_offset(draw.depth_clamp);
   nir_def *range = nir_load_ubo(b, 2, 32, nir_imm_int(b, 0),
                                 nir_imm_int(b, offset), .align_mul = 4,
                                 .range_base = offset, .range = 8);
   nir_def *z = intr->src[0].ssa;
   nir_def *clamped = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                               nir_channel(b, range, 1));
   unsigned never = hk_root_descriptor_offset(draw.force_never_in_shader);
   nir_def *fail = nir_load_ubo(b, 1, 16, nir_imm_int(b, 0),
                                nir_imm_int(b, never), .align_mul = 2,
                                .range_base = never, .range = 2);
   nir_src_rewrite(&intr->src[0],
                   nir_bcsel(b, nir_ine_imm(b, fail, 0),
                             nir_imm_float(b, 2.0), clamped));
   return true;
}

/* The Apple9 argument table carries one UBO pointing at Honeykrisp's
 * root descriptor table. Descriptor sets remain GPU addresses loaded from
 * that table; descriptor contents are never baked into compiled shaders. */
bool
hk_apple9_lower_roots(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_depth_never_agx) {
      unsigned offset = hk_root_descriptor_offset(draw.force_never_in_shader);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_ubo(b, 1, intr->def.bit_size,
         nir_imm_int(b, 0), nir_imm_int(b, offset), .align_mul = 2,
         .range_base = offset, .range = 2));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_draw_id) {
      unsigned offset = hk_root_descriptor_offset(draw.draw_id);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_ubo(b, 1, 32,
         nir_imm_int(b, 0), nir_imm_int(b, offset), .align_mul = 4,
         .range_base = offset, .range = 4));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_base_vertex ||
       intr->intrinsic == nir_intrinsic_load_first_vertex ||
       intr->intrinsic == nir_intrinsic_load_base_instance) {
      unsigned offset = intr->intrinsic == nir_intrinsic_load_base_instance ? 4 : 0;
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *value = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, HK_APPLE9_DRAW_PARAMS_BINDING), nir_imm_int(b, offset),
         .align_mul = 4, .range_base = offset, .range = 4);
      nir_def_replace(&intr->def, value);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
      if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
         return false;
      b->cursor = nir_before_instr(&intr->instr);
      unsigned slot = offsetof(struct hk_root_descriptor_table, sets) +
                      nir_intrinsic_desc_set(handle) * sizeof(uint64_t);
      nir_def *set = nir_load_ubo(b, 1, 64, nir_imm_int(b, 0),
                                 nir_imm_int(b, slot), .align_mul = 8,
                                 .range_base = slot, .range = 8);
      nir_def_replace(&intr->def,
         nir_iadd(b, set, nir_u2u64(b, handle->src[0].ssa)));
      return true;
   }

   unsigned offset;
   if (intr->intrinsic == nir_intrinsic_load_root_agx)
      offset = offsetof(struct hk_root_descriptor_table, root_desc_addr);
   else if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx)
      offset = offsetof(struct hk_root_descriptor_table, sets) +
               nir_intrinsic_desc_set(intr) * sizeof(uint64_t);
   else
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *ptr = nir_load_ubo(b, 1, 64, nir_imm_int(b, 0),
                              nir_imm_int(b, offset), .align_mul = 8,
                              .range_base = offset, .range = 8);
   nir_def_replace(&intr->def, ptr);
   return true;
}

/* Without a fragment stage nothing reads the rasterized varyings: keep only
 * the outputs the fixed-function rasterizer consumes, as other drivers do when
 * linking a position-only pre-rasterization stage. */
static bool
hk_apple9_strip_unread_output(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   switch (nir_intrinsic_io_semantics(intr).location) {
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_SHADING_RATE:
      return false;
   default:
      nir_instr_remove(&intr->instr);
      return true;
   }
}

void
hk_apple9_strip_unread_outputs(nir_shader *nir)
{
   if (!nir_shader_intrinsics_pass(nir, hk_apple9_strip_unread_output,
                                   nir_metadata_control_flow, NULL))
      return;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_dead_cf);
}


/* G16G L1I cache is 16 KiB/core; crossing 16 KiB degrades real fragment
 * shader throughput by 8%-60%. Weight pre-lowering NIR ops by their post-
 * lowering Apple9 instruction count so ~1800 weighted instructions tracks
 * ~15 KiB of packed machine code (~8.5 B/insn). */
#define HK_APPLE9_L1I_UNROLL_INSTR_BUDGET 1800u

static unsigned
hk_nir_intrinsic_est_instrs(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return 24u;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_global_bounded:
   case nir_intrinsic_load_global_constant_bounded:
      return 2u;
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap: {
      nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
      if (deref && nir_deref_mode_may_be(deref, nir_var_mem_ssbo |
                                                nir_var_mem_ubo |
                                                nir_var_mem_global))
         return 2u;
      return 1u;
   }
   default:
      return 1u;
   }
}

static unsigned
hk_cf_list_est_instrs(struct exec_list *cf_list)
{
   unsigned count = 0;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, block) {
            switch (instr->type) {
            case nir_instr_type_alu: {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (!nir_op_is_vec_or_mov(alu->op))
                  count += alu->def.num_components;
               break;
            }
            case nir_instr_type_tex:
               count += 4u;
               break;
            case nir_instr_type_intrinsic:
               count += hk_nir_intrinsic_est_instrs(nir_instr_as_intrinsic(instr));
               break;
            default:
               break;
            }
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         count += hk_cf_list_est_instrs(&nif->then_list);
         count += hk_cf_list_est_instrs(&nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         count += hk_cf_list_est_instrs(&loop->body);
         break;
      }
      default:
         break;
      }
   }
   return count;
}

static bool
hk_loop_has_indirect_local_deref(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_array &&
                !nir_src_is_const(deref->arr.index) &&
                nir_deref_mode_may_be(deref, nir_var_function_temp |
                                             nir_var_shader_in |
                                             nir_var_shader_out))
               return true;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (hk_loop_has_indirect_local_deref(&nif->then_list) ||
             hk_loop_has_indirect_local_deref(&nif->else_list))
            return true;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (hk_loop_has_indirect_local_deref(&loop->body))
            return true;
         break;
      }
      default:
         break;
      }
   }
   return false;
}

struct hk_l1i_unroll_tags {
   nir_function_impl *impl;
   nir_variable *none_var;
   nir_variable *unroll_var;
};

static nir_variable *
hk_get_l1i_tag_var(struct hk_l1i_unroll_tags *tags, nir_loop_control control)
{
   nir_variable **slot =
      control == nir_loop_control_unroll ? &tags->unroll_var : &tags->none_var;
   if (!*slot) {
      *slot = nir_local_variable_create(tags->impl, glsl_uint_type(),
                                        "hk_l1i_unroll_tag");
   }
   return *slot;
}

static void
hk_collect_l1i_unroll_caps(const nir_shader *nir, struct exec_list *cf_list,
                           unsigned *est_instrs,
                           struct hk_l1i_unroll_tags *tags, bool persistent)
{
   const unsigned max_iter = nir->options->max_unroll_iterations;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         hk_collect_l1i_unroll_caps(nir, &nif->then_list, est_instrs, tags,
                                    persistent);
         hk_collect_l1i_unroll_caps(nir, &nif->else_list, est_instrs, tags,
                                    persistent);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         hk_collect_l1i_unroll_caps(nir, &loop->body, est_instrs, tags,
                                    persistent);
         const nir_loop_control orig_control = loop->control;
         if (orig_control == nir_loop_control_dont_unroll || !loop->info)
            break;
         const nir_loop_info *li = loop->info;
         unsigned trip_count =
            li->max_trip_count ? li->max_trip_count : li->guessed_trip_count;
         if (li->force_unroll || trip_count <= 1 || li->instr_cost == 0)
            break;
         if (orig_control != nir_loop_control_unroll &&
             max_iter && trip_count > max_iter)
            break;
         if (hk_loop_has_indirect_local_deref(&loop->body))
            break;
         /* Estimate scalarized post-lowering body cost so pre-scalarization
          * vector/image loops do not underestimate post-unroll code size. */
         unsigned body_est =
            MAX2(li->instr_cost, hk_cf_list_est_instrs(&loop->body));
         uint64_t added = (uint64_t)body_est * (trip_count - 1);
         if ((uint64_t)*est_instrs + added > HK_APPLE9_L1I_UNROLL_INSTR_BUDGET) {
            if (!persistent && tags) {
               /* Tag the loop's first block with a temporary load of a fresh
                * local variable so nir_cf_list_clone preserves exact cap
                * provenance even if an enclosing <=1-iteration loop clones
                * identical sibling loops. */
               nir_variable *tag_var = hk_get_l1i_tag_var(tags, orig_control);
               if (!tag_var)
                  break;
               nir_builder b =
                  nir_builder_at(nir_after_phis(nir_loop_first_block(loop)));
               nir_load_var(&b, tag_var);
            }
            loop->control = nir_loop_control_dont_unroll;
         } else {
            *est_instrs += (unsigned)added;
         }
         break;
      }
      default:
         break;
      }
   }
}

static void
hk_apple9_cap_loop_unroll_l1i(nir_shader *nir)
{
   const nir_variable_mode indirect_mask = nir->options->force_indirect_unrolling;
   const bool force_sampler = nir->options->force_indirect_unrolling_sampler;
   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_loop_analysis, indirect_mask,
                           (int)force_sampler);
      unsigned est = hk_cf_list_est_instrs(&impl->body);
      hk_collect_l1i_unroll_caps(nir, &impl->body, &est, NULL, true);
   }
}

static void
hk_restore_l1i_unroll_caps(struct exec_list *cf_list,
                           const struct hk_l1i_unroll_tags *tags)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         hk_restore_l1i_unroll_caps(&nif->then_list, tags);
         hk_restore_l1i_unroll_caps(&nif->else_list, tags);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         nir_block *first = nir_loop_first_block(loop);
         nir_foreach_instr_safe(instr, first) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_deref)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!deref || deref->deref_type != nir_deref_type_var)
               continue;
            if (deref->var == tags->none_var || deref->var == tags->unroll_var) {
               loop->control = deref->var == tags->unroll_var
                                  ? nir_loop_control_unroll
                                  : nir_loop_control_none;
               nir_instr_remove(instr);
               nir_deref_instr_remove_if_unused(deref);
            }
         }
         hk_restore_l1i_unroll_caps(&loop->body, tags);
         break;
      }
      default:
         break;
      }
   }
}

static void
hk_cleanup_l1i_tag_vars(struct hk_l1i_unroll_tags *tags)
{
   if (tags->none_var) {
      tags->none_var->data.mode = 0;
      exec_node_remove(&tags->none_var->node);
   }
   if (tags->unroll_var) {
      tags->unroll_var->data.mode = 0;
      exec_node_remove(&tags->unroll_var->node);
   }
}

bool
hk_apple9_opt_loop_unroll(nir_shader *nir)
{
   const nir_variable_mode indirect_mask = nir->options->force_indirect_unrolling;
   const bool force_sampler = nir->options->force_indirect_unrolling_sampler;
   struct util_dynarray impl_tags = UTIL_DYNARRAY_INIT;

   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_loop_analysis, indirect_mask,
                           (int)force_sampler);
      unsigned est = hk_cf_list_est_instrs(&impl->body);
      struct hk_l1i_unroll_tags tags = {.impl = impl};
      hk_collect_l1i_unroll_caps(nir, &impl->body, &est, &tags, false);
      if (tags.none_var || tags.unroll_var) {
         struct hk_l1i_unroll_tags *slot =
            util_dynarray_grow(&impl_tags, struct hk_l1i_unroll_tags, 1);
         if (slot) {
            *slot = tags;
         } else {
            /* OOM saving tag metadata: roll back immediately so no temporary
             * cap or tag variable can escape into the shader. */
            hk_restore_l1i_unroll_caps(&impl->body, &tags);
            hk_cleanup_l1i_tag_vars(&tags);
            nir_progress(true, impl, nir_metadata_control_flow |
                                     nir_metadata_loop_analysis);
         }
      }
   }

   bool progress = nir_opt_loop_unroll(nir);

   util_dynarray_foreach(&impl_tags, struct hk_l1i_unroll_tags, tags) {
      hk_restore_l1i_unroll_caps(&tags->impl->body, tags);
      hk_cleanup_l1i_tag_vars(tags);
      nir_progress(true, tags->impl, nir_metadata_control_flow |
                                     nir_metadata_loop_analysis);
   }
   util_dynarray_fini(&impl_tags);
   return progress;
}

void
hk_nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, hk_apple9_opt_loop_unroll);
   } while (progress);
}

/* Count fragment shader invocations for pipeline statistics queries. The
 * counter is behind a uniform test of the active query flags, so draws
 * without the query issue no atomics. With the query active, one lane adds
 * the wave's count: an atomic per fragment on the single counter serialized
 * every fragment of the draw, halving frame rates in games that keep
 * statistics queries open (Unigine Heaven through DXVK).
 */
void
hk_apple9_lower_fs_statistics(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   unsigned offset = hk_root_descriptor_offset(draw.pipeline_stats_flags);
   nir_def *flags = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                 nir_imm_int(&b, offset), .align_mul = 4,
                                 .range_base = offset, .range = 4);
   /* The branches are uniform and usually not taken: keep if-conversion
    * from hoisting the coverage read and ballot into every invocation. */
   nir_if *active = nir_push_if(&b, nir_test_mask(&b, flags,
      VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT));
   active->control = nir_selection_control_dont_flatten;
   {
      nir_def *helper = nir_load_helper_invocation(&b, 1);
      nir_def *lanes = nir_ballot(&b, 1, 32, nir_inot(&b, helper));
      nir_if *counted = nir_push_if(&b, nir_inot(&b, helper));
      counted->control = nir_selection_control_dont_flatten;
      nir_push_if(&b, nir_elect(&b, 1));
      nir_def *addr = nir_load_stat_query_address_poly(
         &b, .base = PIPE_STAT_QUERY_PS_INVOCATIONS);
      nir_global_atomic(&b, 32, addr, nir_bit_count(&b, lanes),
                        .atomic_op = nir_atomic_op_iadd);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
}

/* Standalone linked fragment prolog that counts active non-helper invocations
 * into the active FRAGMENT_SHADER_INVOCATIONS pipeline-statistics slot. Linked
 * at draw time only while the query is active, keeping the counter out of
 * normal fragment shaders.
 *
 * Reads draw.pipeline_stats and draw.pipeline_stats_flags directly from UBO 0
 * (gfx->root) inside non-uniform control flow without ACCESS_CAN_SPECULATE so
 * preamble extraction never hoists them into a preamble.
 */
nir_shader *
hk_apple9_build_fs_stats_prolog_nir(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_fs_stats_prolog");
   nir_shader *nir = b.shader;
   nir->info.internal = true;

   nir_def *helper = nir_load_helper_invocation(&b, 1);
   nir_def *lanes = nir_ballot(&b, 1, 32, nir_inot(&b, helper));
   nir_if *counted = nir_push_if(&b, nir_inot(&b, helper));
   counted->control = nir_selection_control_dont_flatten;
   nir_push_if(&b, nir_elect(&b, 1));
   unsigned stats_off = hk_root_descriptor_offset(draw.pipeline_stats);
   unsigned flags_off = hk_root_descriptor_offset(draw.pipeline_stats_flags);
   nir_def *base = nir_load_ubo(&b, 1, 64, nir_imm_int(&b, 0),
                                nir_imm_int(&b, stats_off), .align_mul = 8,
                                .range_base = stats_off, .range = 8);
   nir_def *flags = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                 nir_imm_int(&b, flags_off), .align_mul = 4,
                                 .range_base = flags_off, .range = 4);
   const VkQueryPipelineStatisticFlagBits bit =
      VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
   nir_def *idx = nir_bit_count(&b, nir_iand_imm(&b, flags, bit - 1));
   nir_def *addr = nir_iadd(
      &b, base, nir_amul_imm(&b, nir_u2u64(&b, idx), sizeof(uint64_t)));
   nir_global_atomic(&b, 32, addr, nir_bit_count(&b, lanes),
                     .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   return nir;
}

/* Draw parameters come from the Apple9 draw-parameter UBO; the geometry
 * pipeline's poly sysvals live in the root table. Lower the former first so
 * hk_lower_uvs_index only sees the table-backed values.
 */
void
hk_apple9_lower_prerast_sysvals(nir_shader *nir, mesa_shader_stage sw_stage)
{
   NIR_PASS(_, nir, poly_nir_lower_sysvals);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
}


void
hk_apple9_lower_primitive_fs(nir_shader *nir, bool native_mesh)
{
   bool broadcast =
      native_mesh && hk_apple9_native_mesh_primitive_lowering_enabled();
   if (!nir_shader_intrinsics_pass(nir, hk_lower_per_primitive_input,
                                   nir_metadata_control_flow, &broadcast))
      return;

   /* Layer reads introduced by primitive lowering use the same
    * fixed-function layer source as ordinary fragment inputs. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            hk_apple9_lower_layer_viewport_fs, nir_metadata_control_flow, NULL);

   /* This NIR has already been preprocessed. The new input/record loads only
    * need dead-code cleanup and refreshed IO metadata before linking; the
    * backend performs ordinary algebraic/scalar cleanup later. Re-running
    * agx_preprocess_nir would nest the non-idempotent helper-write guards. */
   NIR_PASS(_, nir, nir_opt_dce);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

