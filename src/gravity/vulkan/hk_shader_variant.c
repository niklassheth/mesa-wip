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

DEBUG_GET_ONCE_OPTION(cached_hk_apple9_eager_variants,
                      "HK_APPLE9_EAGER_VARIANTS", NULL)
DEBUG_GET_ONCE_BOOL_OPTION(apple9_variant_cache_verify,
                           "HK_APPLE9_VARIANT_CACHE_VERIFY", false)


/* Set while a pipeline-time eager variant compiles on this thread. */
static _Thread_local bool hk_apple9_eager_compile;
/* Resolve a variant into the pipeline cache only, see hk_apple9_eager_variant. */
static _Thread_local bool hk_apple9_precompile;


/* Whether a draw with this depth/stencil state can write the depth or
 * stencil aspects present. A depth/stencil feedback loop needs late tests
 * only then: an early test that writes nothing leaves the attachment the
 * fragment shader reads unchanged (turnip applies the same condition). */
bool
hk_apple9_zs_writes(const struct vk_depth_stencil_state *ds,
                    VkImageAspectFlags aspects)
{
   if (!ds)
      return true;
   struct vk_depth_stencil_state opt = *ds;
   opt.stencil.write_enable = true;
   vk_optimize_depth_stencil_state(&opt, aspects, true);
   return opt.depth.write_enable || opt.stencil.write_enable;
}

void
hk_apple9_graphics_key(struct hk_apple9_graphics_key *key,
                       const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));
   key->vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   key->vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   /* Strides are always read from the bound buffer table, which supports
    * VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE without recompilation. */
   key->vertex.dynamic_stride = true;
   /* Vulkan enables every clip distance the shader writes. */
   key->vertex.clip_distance_enable = 0xff;
   key->blend[0].samples = 1;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
      key->color_map[rt] = rt;
   if (!state) {
      /* Independent shader-object compile (ESO): decouple vertex input,
       * clip-space depth range, and fragment output/blending. */
      key->vertex.use_prolog = true;
      key->clip_negative_one_to_one_dynamic = true;
      key->fs_use_epilog = true;
      key->blend[0].dynamic_samples = true;
      key->blend[0].sample_mask_single_sample = true;
      key->blend[0].dynamic_sample_positions = true;
      key->blend[0].clamp_early_depth = true;
      key->blend[0].early_depth_clamp_offset_B =
         hk_root_descriptor_offset(draw.depth_clamp);
      return;
   }
   if (state->cal) {
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
         key->color_map[rt] = state->cal->color_map[rt];
   }
   if (BITSET_TEST(state->dynamic,
                   MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE)) {
      key->clip_negative_one_to_one_dynamic = true;
      key->clip_negative_one_to_one = false;
   } else {
      key->clip_negative_one_to_one =
         state->vp && state->vp->depth_clip_negative_one_to_one;
   }
   if (state->rp) {
      for (unsigned rt = 0; rt < MIN2(state->rp->color_attachment_count, 8); ++rt) {
         if (state->rp->color_attachment_formats[rt] == VK_FORMAT_UNDEFINED)
            key->undefined_colors |= BITFIELD_BIT(rt);
      }
   }
   key->sample_shading = state->ms && state->ms->sample_shading_enable;
   /* Draw-time resolution passes the flag only for loops the draw can
    * write; a pipeline compiled with static state applies the test here. */
   key->zs_self_dep = (state->pipeline_flags &
                       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) &&
      hk_apple9_zs_writes(state->ds,
         (state->rp && state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED
             ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
         (state->rp && state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED
             ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
   key->line_stipple =
      state->rs && state->rs->line.stipple.enable &&
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
   key->polygon_points =
      state->rs && state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);
   if (state->ia) {
      enum mesa_prim prim =
         u_reduced_prim(vk_topology_to_mesa(state->ia->primitive_topology));
      key->polygon_points &= prim != MESA_PRIM_POINTS && prim != MESA_PRIM_LINES;
   }

   if (state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {
      key->vertex.rasterize_points =
         state->ia->primitive_topology == MESA_PRIM_POINTS;
      key->vertex.ignore_point_size = !key->vertex.rasterize_points;
   }
   if (!state->vi || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI)) {
      key->vertex.use_prolog = true;
   } else {
      for (unsigned a = 0; a < 16; ++a) {
         if (!(state->vi->attributes_valid & BITFIELD_BIT(a)))
            continue;
         const struct vk_vertex_attribute_state *attr =
            &state->vi->attributes[a];
         const struct vk_vertex_binding_state *binding =
            &state->vi->bindings[attr->binding];
         key->vertex.format[a] = vk_format_to_pipe_format(attr->format);
         key->vertex.offset[a] = attr->offset;
         key->vertex.buffer[a] = attr->binding;
         /* A zero instance divisor repeats the first instance's element. */
         key->vertex.divisor[a] =
            binding->input_rate != VK_VERTEX_INPUT_RATE_INSTANCE ? 0
            : binding->divisor ? binding->divisor
                               : AGX_APPLE9_VERTEX_DIVISOR_ZERO;
      }
   }
   key->targets = state->rp ? state->rp->color_attachment_count : 0;
   for (unsigned rt = 0; rt < MAX2(MIN2(key->targets, 8), 1); ++rt) {
      struct agx_apple9_blend *blend = &key->blend[rt];
      blend->format = rt < key->targets
         ? vk_format_to_pipe_format(state->rp->color_attachment_formats[rt])
         : PIPE_FORMAT_NONE;
      blend->color_attachment_unused = blend->format == PIPE_FORMAT_NONE;
      blend->sample_mask_single_sample = true;
      blend->dynamic_sample_positions = true;
      /* A shader-triggered early test (memory writes, discard or depth/stencil
       * writes with early fragment tests) tests the rasterizer's depth, which
       * the hardware clamps to [0, 1]. Clipping keeps it within the viewport's
       * depth range unless depth clamp is enabled, so only then is a clamped
       * depth export needed. The export tests the shader's interpolated
       * frag_coord.z, which differs from the rasterizer's depth by a few ULPs:
       * an EQUAL pass after a depth prepass without such a shader would fail
       * (Zink EEVEE's G-buffer pass). Draw-time variants see the current
       * dynamic state; pipelines with dynamic depth clamp keep the export. */
      blend->clamp_early_depth =
         !state->rs || state->rs->depth_clamp_enable ||
         BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
      blend->early_depth_clamp_ubo = 0;
      blend->early_depth_clamp_offset_B =
         hk_root_descriptor_offset(draw.depth_clamp);
      blend->samples =
         (state->ms &&
          !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
          state->ms->rasterization_samples)
            ? state->ms->rasterization_samples : 1;
      if (rt == 0 &&
          (!state->ms ||
           BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES)))
         blend->dynamic_samples = true;
      blend->rgb_src = blend->alpha_src = PIPE_BLENDFACTOR_ONE;
      blend->rgb_dst = blend->alpha_dst = PIPE_BLENDFACTOR_ZERO;
      blend->rgb_func = blend->alpha_func = PIPE_BLEND_ADD;
      if (state->ms) {
         blend->disabled_samples = ~state->ms->sample_mask;
         blend->alpha_to_coverage = state->ms->alpha_to_coverage_enable;
         blend->alpha_to_one = state->ms->alpha_to_one_enable;
      }
      if (state->cb && rt < state->cb->attachment_count) {
         const struct vk_color_blend_attachment_state *cb =
            &state->cb->attachments[rt];
         blend->dynamic_enable = BITSET_TEST(
            state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
         blend->colormask = (blend->dynamic_enable ||
                             (state->cb->color_write_enables & BITFIELD_BIT(rt)))
                               ? cb->write_mask
                               : 0;
         blend->logicop_enable = state->cb->logic_op_enable;
         blend->logicop_func = vk_logic_op_to_pipe(state->cb->logic_op);
         if (cb->blend_enable && cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            /* Advanced equations are blended in the shader from the tile. */
            blend->advanced_mode =
               vk_advanced_blend_op_to_pipe(cb->color_blend_op);
            blend->advanced_overlap =
               vk_blend_overlap_to_pipe(cb->blend_overlap);
            blend->src_premultiplied = cb->src_premultiplied;
            blend->dst_premultiplied = cb->dst_premultiplied;
         } else if (cb->blend_enable) {
            blend->rgb_src =
               vk_blend_factor_to_pipe(cb->src_color_blend_factor);
            blend->rgb_dst =
               vk_blend_factor_to_pipe(cb->dst_color_blend_factor);
            blend->alpha_src =
               vk_blend_factor_to_pipe(cb->src_alpha_blend_factor);
            blend->alpha_dst =
               vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor);
            blend->rgb_func = vk_blend_op_to_pipe(cb->color_blend_op);
            blend->alpha_func = vk_blend_op_to_pipe(cb->alpha_blend_op);
         }
      }
   }
}

bool
hk_apple9_partial_stages(VkShaderStageFlags stages)
{
   return !(stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
          !(stages & (VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                      VK_SHADER_STAGE_GEOMETRY_BIT |
                      VK_SHADER_STAGE_MESH_BIT_EXT));
}

/* Dynamic states read by the Apple9 graphics key. */
static const enum mesa_vk_dynamic_graphics_state hk_apple9_keyed_dynamic[] = {
   MESA_VK_DYNAMIC_VI,
   MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES,
   MESA_VK_DYNAMIC_MS_SAMPLE_MASK,
   MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE,
   MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
   MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
   MESA_VK_DYNAMIC_CB_LOGIC_OP,
   MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT,
   MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
   MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
   MESA_VK_DYNAMIC_CB_WRITE_MASKS,
   MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE,
   MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
   MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE,
   MESA_VK_DYNAMIC_RS_POLYGON_MODE,
};

/* With unused attachments, a render's tile layout can differ from the one
 * pipeline formats imply: a null view occupies one word whatever the
 * pipeline format, and an attachment the pipeline leaves undefined occupies
 * its view's size. Such fragment shaders take their layout from the render.
 */
bool
hk_apple9_color_layout_varies(const struct vk_graphics_pipeline_state *state,
                              const struct vk_features *features)
{
   if (!state || !state->rp || !features ||
       !features->dynamicRenderingUnusedAttachments)
      return false;
   for (unsigned rt = 0; rt < state->rp->color_attachment_count; ++rt) {
      VkFormat format = state->rp->color_attachment_formats[rt];
      if (format == VK_FORMAT_UNDEFINED ||
          agx_apple9_color_words(vk_format_to_pipe_format(format)) != 1)
         return true;
   }
   return false;
}

static bool
hk_apple9_is_fs_epilog_dynamic(enum mesa_vk_dynamic_graphics_state state)
{
   switch (state) {
   case MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES:
   case MESA_VK_DYNAMIC_MS_SAMPLE_MASK:
   case MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE:
   case MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE:
   case MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE:
   case MESA_VK_DYNAMIC_CB_LOGIC_OP:
   case MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT:
   case MESA_VK_DYNAMIC_CB_BLEND_ENABLES:
   case MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS:
   case MESA_VK_DYNAMIC_CB_WRITE_MASKS:
   case MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP:
      return true;
   default:
      return false;
   }
}

bool
hk_apple9_fs_needs_epilog(const struct vk_graphics_pipeline_state *state,
                          const struct vk_features *features)
{
   if (!state || !state->rp || !state->ms || !state->cb)
      return true;
   if (hk_apple9_color_layout_varies(state, features))
      return true;
   for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_keyed_dynamic); ++i) {
      if (hk_apple9_is_fs_epilog_dynamic(hk_apple9_keyed_dynamic[i]) &&
          BITSET_TEST(state->dynamic, hk_apple9_keyed_dynamic[i]))
         return true;
   }
   return false;
}


void
hk_apple9_graphics_key_device(struct hk_apple9_graphics_key *key,
                               const struct vk_graphics_pipeline_state *state,
                               const struct hk_physical_device *pdev)
{
   hk_apple9_graphics_key(key, state);
   key->native_index = hk_apple9_native_index_enabled(pdev);
   key->native_stages = hk_apple9_native_stage_policy(pdev);
}


uint32_t
hk_apple9_deferral_hash_bits(const struct vk_graphics_pipeline_state *state,
                             const struct vk_features *features,
                             VkShaderStageFlags stages)
{
   uint32_t deferral = (!state->vi << 0) | (!state->rp << 1) |
                       (!state->ms << 2) | (!state->cb << 3) |
                       (!!(state->pipeline_flags &
                           VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) << 31) |
                       (hk_apple9_partial_stages(stages) << 30) |
                       (hk_apple9_color_layout_varies(state, features) << 29);
   for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_keyed_dynamic); ++i) {
      if (BITSET_TEST(state->dynamic, hk_apple9_keyed_dynamic[i]))
         deferral |= BITFIELD_BIT(4 + i);
   }
   return deferral;
}

bool
hk_apple9_eager_variants_disabled(void)
{
   return debug_get_option_cached_hk_apple9_eager_variants() != NULL;
}

struct hk_apple9_nir *
hk_apple9_nir_create(const void *data, size_t size)
{
   if (size > UINT32_MAX)
      return NULL;
   struct hk_apple9_nir *nir = malloc(sizeof(*nir) + MAX2(size, 1));
   if (!nir)
      return NULL;
   nir->refs = 1;
   nir->size = size;
   memcpy(nir->data, data, size);
   return nir;
}

struct hk_apple9_nir *
hk_apple9_nir_ref(struct hk_apple9_nir *nir)
{
   p_atomic_inc(&nir->refs);
   return nir;
}

void
hk_apple9_nir_unref(struct hk_apple9_nir *nir)
{
   if (nir && p_atomic_dec_zero(&nir->refs))
      free(nir);
}


void
hk_apple9_remember_variant(struct hk_device *dev, struct hk_api_shader *api,
                            struct vk_pipeline_cache_object *object)
{
   struct hk_apple9_deferred *d = hk_apple9_deferred_source(api);
   if (!d)
      return;
   struct hk_apple9_cached_variant *evicted = NULL;
   simple_mtx_lock(&d->variant_lock);
   for (struct hk_apple9_cached_variant **p = &d->variants; *p;
        p = &(*p)->next) {
      if ((*p)->object == object) {
         struct hk_apple9_cached_variant *v = *p;
         *p = v->next;
         v->next = d->variants;
         d->variants = v;
         simple_mtx_unlock(&d->variant_lock);
         return;
      }
   }
   struct hk_apple9_cached_variant *v = malloc(sizeof(*v));
   if (v) {
      v->object = vk_pipeline_cache_object_ref(object);
      v->next = d->variants;
      d->variants = v;
      if (d->variant_count == HK_APPLE9_CACHED_VARIANTS_PER_SOURCE) {
         struct hk_apple9_cached_variant **tail = &d->variants;
         while ((*tail)->next)
            tail = &(*tail)->next;
         evicted = *tail;
         *tail = NULL;
      } else {
         ++d->variant_count;
      }
   }
   simple_mtx_unlock(&d->variant_lock);
   /* Cache serialization takes the runtime-cache lock before variant_lock.
    * A last unref takes that cache lock, so eviction must happen outside. */
   if (evicted) {
      vk_pipeline_cache_object_unref(&dev->vk, evicted->object);
      free(evicted);
   }
}

bool
hk_apple9_serialize_variants(struct hk_apple9_deferred *d, struct blob *blob)
{
   intptr_t count_at = blob_reserve_uint32(blob);
   if (count_at < 0)
      return false;
   uint32_t count = 0;
   struct hk_apple9_cached_variant *variants[HK_APPLE9_CACHED_VARIANTS_PER_SOURCE];
   simple_mtx_lock(&d->variant_lock);
   for (struct hk_apple9_cached_variant *v = d->variants; v; v = v->next) {
      assert(count < ARRAY_SIZE(variants));
      variants[count++] = v;
   }
   /* Import remembers each entry at the head: write oldest first to preserve
    * recency when an application cache is loaded in another process. */
   for (unsigned i = count; i > 0; --i) {
      struct hk_apple9_cached_variant *v = variants[i - 1];
      struct vk_raw_data_cache_object *raw =
         container_of(v->object, struct vk_raw_data_cache_object, base);
      assert(v->object->key_size == BLAKE3_OUT_LEN);
      if (raw->data_size > UINT32_MAX) {
         blob->out_of_memory = true;
         break;
      }
      blob_write_bytes(blob, v->object->key_data, BLAKE3_OUT_LEN);
      blob_write_uint32(blob, raw->data_size);
      blob_write_bytes(blob, raw->data, raw->data_size);
   }
   simple_mtx_unlock(&d->variant_lock);
   blob_overwrite_uint32(blob, count_at, count);
   return !blob->out_of_memory;
}

bool
hk_apple9_deserialize_variants(struct hk_device *dev,
                                struct hk_api_shader *api,
                                struct blob_reader *blob)
{
   uint32_t count = blob_read_uint32(blob);
   if (blob->overrun || count > (blob->end - blob->current) / (BLAKE3_OUT_LEN + 4))
      return false;
   for (unsigned i = 0; i < count; ++i) {
      const void *key = blob_read_bytes(blob, BLAKE3_OUT_LEN);
      uint32_t size = blob_read_uint32(blob);
      const void *data = blob_read_bytes(blob, size);
      if (blob->overrun || !size)
         return false;
      if (!dev->vk.mem_cache)
         continue;
      struct vk_pipeline_cache_object *object =
         vk_pipeline_cache_create_and_insert_object(
            dev->vk.mem_cache, key, BLAKE3_OUT_LEN, data, size,
            &vk_raw_data_cache_object_ops);
      if (object) {
         hk_apple9_remember_variant(dev, api, object);
         vk_pipeline_cache_object_unref(&dev->vk, object);
      }
   }
   return true;
}


void
hk_apple9_destroy_deferred(struct hk_device *dev, struct hk_apple9_deferred *d)
{
   struct hk_apple9_cached_variant *v = d->variants;
   while (v) {
      struct hk_apple9_cached_variant *next = v->next;
      vk_pipeline_cache_object_unref(&dev->vk, v->object);
      free(v);
      v = next;
   }
   simple_mtx_destroy(&d->variant_lock);
   hk_apple9_nir_unref(d->nir);
   free(d);
}


static bool
hk_apple9_fs_uses_per_vertex(nir_shader *fs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (hk_is_barycentric_coord(op) ||
             op == nir_intrinsic_load_input_vertex)
            return true;
      }
   }
   return false;
}

static bool
hk_apple9_can_use_vs_prolog(uint32_t count,
                            const struct vk_shader_compile_info *infos,
                            const struct vk_graphics_pipeline_state *state)
{
   if (state && state->rs &&
       ((state->rs->line.stipple.enable &&
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE)) ||
        (state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE))))
      return false;

   bool has_vs = false;
   for (uint32_t i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX) {
         has_vs = true;
         if ((infos[i].next_stage_mask &
              (VK_SHADER_STAGE_GEOMETRY_BIT |
               VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) ||
             infos[i].nir->xfb_info ||
             infos[i].nir->info.cull_distance_array_size)
            return false;
      } else if (infos[i].stage == MESA_SHADER_FRAGMENT) {
         if (hk_apple9_fs_uses_per_vertex(infos[i].nir))
            return false;
      } else {
         return false;
      }
   }
   return has_vs;
}

static bool
hk_apple9_fs_uses_sample_rate(nir_shader *fs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_load_sample_id:
         case nir_intrinsic_load_sample_pos:
         case nir_intrinsic_load_sample_mask_in:
         case nir_intrinsic_load_sample_mask:
         case nir_intrinsic_load_barycentric_at_sample:
         case nir_intrinsic_load_barycentric_coord_centroid:
         case nir_intrinsic_load_barycentric_coord_at_sample:
         case nir_intrinsic_load_barycentric_coord_at_offset:
            return true;
         default:
            break;
         }
      }
   }
   return false;
}

bool
hk_apple9_fs_can_epilog_nir(nir_shader *fs)
{
   if (fs->info.internal || fs->info.fs.early_fragment_tests ||
       fs->info.writes_memory || fs->info.fs.uses_sample_shading ||
       fs->info.fs.pixel_interlock_ordered ||
       fs->info.fs.pixel_interlock_unordered)
      return false;
   if (fs->info.outputs_written &
       (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
        BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
        BITFIELD64_BIT(FRAG_RESULT_COLOR)))
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (op == nir_intrinsic_load_local_pixel_agx ||
             op == nir_intrinsic_load_tile_pixel_agx ||
             op == nir_intrinsic_begin_invocation_interlock)
            return false;
      }
   }
   return true;
}

static bool
hk_apple9_fs_can_epilog(uint32_t count,
                        const struct vk_shader_compile_info *infos,
                        const struct vk_graphics_pipeline_state *state,
                        bool *uses_sample_rate)
{
   *uses_sample_rate = false;
   if (state &&
       ((state->ms && state->ms->sample_shading_enable) ||
        (state->pipeline_flags &
         VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)))
      return false;
   for (uint32_t i = 0; i < count; ++i) {
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         continue;
      if (!hk_apple9_fs_can_epilog_nir(infos[i].nir))
         return false;
      *uses_sample_rate = hk_apple9_fs_uses_sample_rate(infos[i].nir);
      return true;
   }
   return true;
}

static bool
hk_apple9_can_compile_vs_independent(
   const struct vk_shader_compile_info *info,
   const struct vk_graphics_pipeline_state *state)
{
   if (info->stage != MESA_SHADER_VERTEX)
      return false;
   if ((info->next_stage_mask &
        (VK_SHADER_STAGE_GEOMETRY_BIT |
         VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) ||
       info->nir->xfb_info || info->nir->info.cull_distance_array_size)
      return false;
   if (state && state->rs &&
       ((state->rs->line.stipple.enable &&
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE)) ||
        (state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE))))
      return false;
   return true;
}

static bool
hk_apple9_can_compile_fs_independent(
   const struct vk_shader_compile_info *info,
   const struct vk_graphics_pipeline_state *state)
{
   if (info->stage != MESA_SHADER_FRAGMENT)
      return false;
   if (!hk_apple9_fs_can_epilog_nir(info->nir) ||
       hk_apple9_fs_uses_per_vertex(info->nir))
      return false;
   /* Viewport index must read 0 when the producer omits it, and primitive ID
    * switches between rasterizer-generated and flat varying when paired with
    * GS/TES/mesh; keep those rare fragment shaders on the paired path. */
   if (info->nir->info.inputs_read &
       (VARYING_BIT_VIEWPORT | VARYING_BIT_PRIMITIVE_ID))
      return false;
   if (state &&
       ((state->ms && state->ms->sample_shading_enable) ||
        (state->pipeline_flags &
         VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) ||
        (state->rs &&
         ((state->rs->line.stipple.enable &&
           !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE)) ||
          (state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
           !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE))))))
      return false;
   if (hk_apple9_fs_uses_sample_rate(info->nir) &&
       (!state || !state->ms ||
        BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES)))
      return false;
   return true;
}

/* Apple9 graphics shaders specialise on state that the common AGX path
 * handles with prologs and epilogs: vertex formats, blending, attachment
 * formats and multisampling. When that state is dynamic, belongs to another
 * pipeline library, or is absent for shader objects, compile independently
 * whenever prolog/epilog + crossbar CF linking can decouple the stage, and
 * defer otherwise.
 */
enum hk_apple9_deferral
hk_apple9_needs_deferral(uint32_t count,
                         const struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features)
{
   if (!state) {
      if (count == 1 &&
          (hk_apple9_can_compile_vs_independent(&infos[0], NULL) ||
           hk_apple9_can_compile_fs_independent(&infos[0], NULL)))
         return HK_APPLE9_COMPILE_NOW;
      return HK_APPLE9_DEFER_STATE;
   }

   /* Graphics pipeline libraries: compile independent VS and FS libraries on
    * the caller's worker thread so fast-linking and first draws do not stall. */
   if (state->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      if (count == 1 &&
          (hk_apple9_can_compile_vs_independent(&infos[0], state) ||
           hk_apple9_can_compile_fs_independent(&infos[0], state)))
         return HK_APPLE9_COMPILE_NOW;
      return HK_APPLE9_DEFER_STATE;
   }

   /* Stages compiled apart from the rest of their pipeline, as fast-linked
    * libraries are, cannot agree on the varying layout until they are
    * linked. Pipelines without a fragment shader link the null one then;
    * such a whole pipeline (a depth-only pass) has the rest of its state. */
   VkShaderStageFlags compiled = 0;
   for (uint32_t i = 0; i < count; ++i)
      compiled |= mesa_to_vk_shader_stage(infos[i].stage);
   bool no_fragment = false;
   if (hk_apple9_partial_stages(compiled)) {
      if (count == 1 &&
          (hk_apple9_can_compile_vs_independent(&infos[0], state) ||
           hk_apple9_can_compile_fs_independent(&infos[0], state)))
         return HK_APPLE9_COMPILE_NOW;
      if ((compiled & VK_SHADER_STAGE_FRAGMENT_BIT) || !state->rp || !state->ms)
         return HK_APPLE9_DEFER_STATE;
      no_fragment = true;
   }

   bool vs_prolog_ok = hk_apple9_can_use_vs_prolog(count, infos, state);
   bool fs_sample_rate = false;
   bool fs_epilog_ok =
      hk_apple9_fs_can_epilog(count, infos, state, &fs_sample_rate);
   bool clip_z_dyn_ok = !(compiled & VK_SHADER_STAGE_MESH_BIT_EXT);
   for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_keyed_dynamic); ++i) {
      enum mesa_vk_dynamic_graphics_state dyn = hk_apple9_keyed_dynamic[i];
      if (!BITSET_TEST(state->dynamic, dyn))
         continue;
      if (vs_prolog_ok && dyn == MESA_VK_DYNAMIC_VI)
         continue;
      if (clip_z_dyn_ok &&
          dyn == MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE)
         continue;
      if (fs_epilog_ok && hk_apple9_is_fs_epilog_dynamic(dyn)) {
         if (dyn == MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES && fs_sample_rate)
            return HK_APPLE9_DEFER_STATE;
         continue;
      }
      /* Compile pure VS/FS pipelines eagerly for the common non-stippled,
       * non-POINT polygon mode and retain fallback NIR in case a draw enables
       * stipple or polygon_mode=POINT dynamically. */
      if ((dyn == MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE ||
           dyn == MESA_VK_DYNAMIC_RS_POLYGON_MODE) &&
          !(compiled & ~(VK_SHADER_STAGE_VERTEX_BIT |
                         VK_SHADER_STAGE_FRAGMENT_BIT)))
         continue;
      return HK_APPLE9_DEFER_STATE;
   }

   /* A depth/stencil feedback loop keys on whether the draw can write depth
    * or stencil (hk_apple9_zs_writes): dynamic state decides at draw time. */
   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) {
      static const enum mesa_vk_dynamic_graphics_state ds_writes[] = {
         MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE,
         MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE,
         MESA_VK_DYNAMIC_DS_DEPTH_COMPARE_OP,
         MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE,
         MESA_VK_DYNAMIC_DS_STENCIL_OP,
         MESA_VK_DYNAMIC_DS_STENCIL_WRITE_MASK,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(ds_writes); ++i) {
         if (BITSET_TEST(state->dynamic, ds_writes[i]))
            return HK_APPLE9_DEFER_STATE;
      }
   }

   for (uint32_t i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX && !state->vi && !vs_prolog_ok)
         return HK_APPLE9_DEFER_STATE;
      /* Without color attachments there is no blend state to key on. */
      if (infos[i].stage == MESA_SHADER_FRAGMENT) {
         if (fs_epilog_ok) {
            if (!state->ms && fs_sample_rate)
               return HK_APPLE9_DEFER_STATE;
         } else if (!state->rp || !state->ms ||
                    (!state->cb && state->rp->color_attachment_count)) {
            return HK_APPLE9_DEFER_STATE;
         }
      }
   }
   if (no_fragment || (!state->cb && !fs_epilog_ok))
      return HK_APPLE9_DEFER_COMPLETE;
   if (!fs_epilog_ok && hk_apple9_color_layout_varies(state, features))
      return HK_APPLE9_DEFER_COMPLETE;
   return HK_APPLE9_COMPILE_NOW;
}

/* Debug only: a vertex shader writing zero to every generic varying a
 * fragment shader reads. Offline replay never links a fragment shader
 * compiled apart from its vertex shader; pairing it with this stub reaches
 * the backend with a matching varying layout. */
static nir_shader *
hk_apple9_stub_vs_nir(uint64_t inputs)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                 &agx_nir_options, "stub VS");
   nir_def *zero = nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0);
   uint64_t written = VARYING_BIT_POS;
   u_foreach_bit64(slot, inputs) {
      if (slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_VAR0 + 32)
         written |= BITFIELD64_BIT(slot);
   }
   u_foreach_bit64(slot, written) {
      nir_store_output(&b, zero, nir_imm_int(&b, 0), .base = 0,
                       .write_mask = 0xf, .component = 0,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
   }
   b.shader->info.outputs_written = written;
   return b.shader;
}

static __thread bool hk_apple9_eager_busy;

/* Compile a deferred pipeline's variant when it is created, as a draw would:
 * for pipelines whose state is complete (HK_APPLE9_DEFER_COMPLETE), and,
 * with HK_APPLE9_EAGER_VARIANTS for debugging, for every deferred pipeline, so
 * offline replay that never draws reaches the backend. State the pipeline
 * leaves dynamic takes simple defaults, so that variant can differ from the
 * one a real draw selects.
 *
 * With precompile, the variant is queued for a low-priority thread that
 * compiles it into the disk cache only: pipeline creation does not wait, a
 * pipeline an application creates but never draws costs no resident program
 * or serialized copy, and its first draw loads instead of compiling.
 */
void
hk_apple9_eager_variant(struct hk_device *dev, uint32_t count,
                        struct vk_shader *const *shaders,
                        const struct vk_graphics_pipeline_state *pstate,
                        bool precompile)
{
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {NULL};
   if (hk_apple9_eager_busy)
      return;
   for (uint32_t i = 0; i < count; ++i)
      api[shaders[i]->stage] = container_of(shaders[i], struct hk_api_shader, vk);

   /* A fragment shader compiled alone: pair it with a stub vertex shader. */
   struct hk_api_shader *stub = NULL;
   if (!api[MESA_SHADER_VERTEX] && !api[MESA_SHADER_MESH] &&
       api[MESA_SHADER_FRAGMENT] && api[MESA_SHADER_FRAGMENT]->apple9_deferred &&
       !(pstate && (pstate->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))) {
      const struct hk_apple9_deferred *d = api[MESA_SHADER_FRAGMENT]->apple9_deferred;
      struct blob_reader reader;
      blob_reader_init(&reader, d->nir->data, d->nir->size);
      nir_shader *fs = nir_deserialize(NULL, &agx_nir_options, &reader);
      if (fs) {
         struct vk_shader_compile_info info = {
            .stage = MESA_SHADER_VERTEX,
            .nir = hk_apple9_stub_vs_nir(fs->info.inputs_read),
            .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
            .robustness = &vk_robustness_disabled,
         };
         ralloc_free(fs);
         struct vk_shader *vs = NULL;
         hk_apple9_eager_busy = true;
         if (hk_apple9_defer_graphics(dev, 1, &info, NULL, NULL, &vs) == VK_SUCCESS)
            stub = container_of(vs, struct hk_api_shader, vk);
         hk_apple9_eager_busy = false;
         api[MESA_SHADER_VERTEX] = stub;
      }
   }
   if (agx_apple9_debug_enabled(AGX_A9_DBG_EAGER))
      fprintf(stderr, "HK_APPLE9_EAGER: %u stages vs=%d fs=%d flags=%#llx\n", count,
              !!api[MESA_SHADER_VERTEX], !!api[MESA_SHADER_FRAGMENT],
              pstate ? (unsigned long long)pstate->pipeline_flags : 0ull);
   if ((!api[MESA_SHADER_VERTEX] && !api[MESA_SHADER_MESH]) ||
       (pstate && (pstate->pipeline_flags &
                   VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)))
      return;

   struct vk_vertex_input_state vi = {.bindings_valid = 1};
   vi.bindings[0].stride = 16;
   vi.attributes_valid = BITFIELD_MASK(MESA_VK_MAX_VERTEX_ATTRIBUTES);
   for (unsigned a = 0; a < MESA_VK_MAX_VERTEX_ATTRIBUTES; ++a)
      vi.attributes[a].format = VK_FORMAT_R32G32B32A32_SFLOAT;
   struct vk_input_assembly_state ia = {
      .primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   struct vk_viewport_state vp = {.viewport_count = 1, .scissor_count = 1};
   struct vk_multisample_state ms = {
      .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
      .sample_mask = 0xffff,
   };
   struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   struct vk_color_blend_state cb = {0};
   struct vk_color_attachment_location_state cal;
   for (unsigned rt = 0; rt < MESA_VK_MAX_COLOR_ATTACHMENTS; ++rt) {
      cb.attachments[rt].write_mask = 0xf;
      cal.color_map[rt] = rt;
   }

   struct vk_graphics_pipeline_state state = {
      .vi = pstate && pstate->vi ? pstate->vi : &vi,
      .ia = pstate && pstate->ia ? pstate->ia : &ia,
      .vp = pstate && pstate->vp ? pstate->vp : &vp,
      .ms = pstate && pstate->ms ? pstate->ms : &ms,
      .rp = pstate && pstate->rp ? pstate->rp : &rp,
      .cal = pstate && pstate->cal ? pstate->cal : &cal,
      /* As draws resolve: rasterization state, but no depth/stencil state. */
      .rs = pstate ? pstate->rs : NULL,
      .pipeline_flags = pstate ? pstate->pipeline_flags : 0,
   };
   if (pstate && pstate->cb) {
      state.cb = pstate->cb;
   } else {
      cb.attachment_count = state.rp->color_attachment_count;
      state.cb = &cb;
   }
   BITSET_SET(state.dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);

   struct hk_api_shader *variant[HK_APPLE9_GFX_STAGES];
   hk_apple9_eager_compile = true;
   hk_apple9_precompile = precompile;
   /* Warm the variant draws use outside statistics queries. */
   if (hk_apple9_resolve_graphics(dev, api, &state, variant, true, false) !=
          VK_SUCCESS &&
       !precompile)
      fprintf(stderr, "HK_APPLE9_EAGER_VARIANTS: variant compile failed\n");
   hk_apple9_precompile = false;
   hk_apple9_eager_compile = false;
   if (stub)
      hk_api_shader_destroy(&dev->vk, &stub->vk, NULL);
}

struct hk_apple9_deferred *
hk_apple9_build_deferred(const struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state)
{
   struct hk_apple9_deferred *deferred = calloc(1, sizeof(*deferred));
   if (!deferred)
      return NULL;

   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, info->nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      free(deferred);
      return NULL;
   }
   deferred->nir = hk_apple9_nir_create(blob.data, blob.size);
   blob_finish(&blob);
   if (!deferred->nir) {
      free(deferred);
      return NULL;
   }
   deferred->next_stage_mask = info->next_stage_mask;
   deferred->set_layout_count = info->set_layout_count;
   deferred->cull_distance = info->nir->info.cull_distance_array_size != 0;
   deferred->vs_can_prolog =
      info->stage == MESA_SHADER_VERTEX &&
      !(info->next_stage_mask &
        (VK_SHADER_STAGE_GEOMETRY_BIT |
         VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) &&
      !info->nir->xfb_info &&
      !info->nir->info.cull_distance_array_size;
   deferred->fs_uses_per_vertex =
      info->stage == MESA_SHADER_FRAGMENT &&
      hk_apple9_fs_uses_per_vertex(info->nir);
   deferred->fs_can_epilog =
      info->stage == MESA_SHADER_FRAGMENT &&
      hk_apple9_fs_can_epilog_nir(info->nir);
   deferred->fs_uses_sample_rate =
      info->stage == MESA_SHADER_FRAGMENT &&
      hk_apple9_fs_uses_sample_rate(info->nir);
   deferred->fs_reads_viewport =
      info->stage == MESA_SHADER_FRAGMENT &&
      (info->nir->info.inputs_read & VARYING_BIT_VIEWPORT);
   deferred->fs_reads_prim_id =
      info->stage == MESA_SHADER_FRAGMENT &&
      (info->nir->info.inputs_read & VARYING_BIT_PRIMITIVE_ID);
   deferred->vs_dynamic_topology =
      info->stage == MESA_SHADER_VERTEX &&
      (!state || !state->ia ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY));
   deferred->sample_shading =
      state && state->ms && state->ms->sample_shading_enable;
   deferred->vertex_unbounded = info->stage == MESA_SHADER_VERTEX &&
                                hk_apple9_vertex_unbounded(info);
   deferred->zs_self_dep =
      state && (state->pipeline_flags &
                VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
   if (info->stage == MESA_SHADER_FRAGMENT && state && state->rp) {
      for (unsigned rt = 0; rt < MIN2(state->rp->color_attachment_count, 8);
           ++rt) {
         if (state->rp->color_attachment_formats[rt] == VK_FORMAT_UNDEFINED)
            deferred->undefined_colors |= BITFIELD_BIT(rt);
      }
   }
   simple_mtx_init(&deferred->variant_lock, mtx_plain);
   return deferred;
}

VkResult
hk_apple9_defer_graphics(struct hk_device *dev, uint32_t count,
                         struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const VkAllocationCallbacks *alloc,
                         struct vk_shader **out)
{
   VkResult result = VK_SUCCESS;
   memset(out, 0, count * sizeof(*out));

   for (uint32_t i = 0; i < count; ++i) {
      bool native_candidate = false;
      if (infos[i].stage == MESA_SHADER_MESH) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         key.native_task_bridge = hk_apple9_native_task_bridge_enabled() &&
                                  infos[i].nir->info.task_payload_size;
         result = hk_apple9_native_mesh_check(infos[i].nir, &key,
                                               &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      if (infos[i].stage == MESA_SHADER_GEOMETRY) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         result = hk_apple9_native_gs_check(infos[i].nir, &key,
                                           &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      if (infos[i].stage == MESA_SHADER_TESS_EVAL) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         result = hk_apple9_native_tess_check(infos[i].nir, &key,
                                              &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      size_t size = sizeof(struct hk_api_shader) +
                    hk_num_variants(infos[i].stage) * sizeof(struct hk_shader);
      struct hk_api_shader *obj =
         vk_shader_zalloc(&dev->vk, &hk_shader_ops, infos[i].stage, alloc, size);
      struct hk_apple9_deferred *deferred =
         obj ? hk_apple9_build_deferred(&infos[i], state) : NULL;
      if (!obj || !deferred) {
         if (obj)
            vk_shader_free(&dev->vk, alloc, &obj->vk);
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }
      obj->apple9_deferred = deferred;
      obj->apple9_native_candidate = native_candidate;
      if (native_candidate) {
         uint8_t policy = hk_apple9_native_stage_policy(hk_device_physical(dev));
         if (infos[i].stage == MESA_SHADER_GEOMETRY)
            obj->apple9_native_stage_mask = policy & HK_APPLE9_NATIVE_GS_BIT;
         if (infos[i].stage == MESA_SHADER_TESS_EVAL)
            obj->apple9_native_stage_mask = policy & HK_APPLE9_NATIVE_TES_BIT;
      }
      obj->apple9_native_cull_primitive =
         infos[i].stage == MESA_SHADER_MESH &&
         ((infos[i].nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) ||
          infos[i].nir->info.cull_distance_array_size);
      obj->writes_memory = infos[i].nir->info.writes_memory;
      obj->pre_raster_reads = hk_nir_pre_raster_reads(infos[i].nir);
      obj->reads_draw_id = hk_nir_reads_draw_id(infos[i].nir);
      obj->push_constant_mask = hk_nir_push_constant_mask(infos[i].nir);
      out[i] = &obj->vk;
   }

   for (uint32_t i = 0; i < count; ++i)
      ralloc_free(infos[i].nir);

   if (result != VK_SUCCESS) {
      for (uint32_t i = 0; i < count; ++i) {
         if (out[i])
            hk_api_shader_destroy(&dev->vk, out[i], alloc);
      }
      memset(out, 0, count * sizeof(*out));
   } else if (debug_get_option_cached_hk_apple9_eager_variants()) {
      hk_apple9_eager_variant(dev, count, out, state, false);
   }
   return result;
}

bool
hk_apple9_deferred_sample_shading(const struct hk_api_shader *fs)
{
   return fs && fs->apple9_deferred && fs->apple9_deferred->sample_shading;
}

/* Whether the fragment shader's pipeline declared a depth/stencil feedback
 * loop; draw-time resolution applies hk_apple9_zs_writes to it. */
bool
hk_apple9_deferred_zs_self_dep(const struct hk_api_shader *fs)
{
   return fs && fs->apple9_deferred && fs->apple9_deferred->zs_self_dep;
}

bool
hk_apple9_fallback_dynamic_topology(const struct hk_api_shader *vs)
{
   return vs && vs->apple9_fallback && vs->apple9_fallback->vs_dynamic_topology;
}

struct hk_apple9_variant {
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES];
   struct hk_apple9_graphics_key key;
   bool prefer_native;
   struct hk_api_shader *shaders[HK_APPLE9_GFX_STAGES];
   /* Keep the serialized object in the device's weak pipeline cache for the
    * lifetime of the live variant. The disk cache outlives both. */
   struct vk_pipeline_cache_object *cache_object;

   /* Signalled once the compile finished; result holds its outcome. The
    * variant is in the table from the start, so threads that need it wait
    * for this compile while others look up and compile variants freely. */
   struct util_queue_fence ready;
   VkResult result;
};

static uint32_t
hk_apple9_variant_hash(const void *key)
{
   return _mesa_hash_data(key, offsetof(struct hk_apple9_variant, shaders));
}

static bool
hk_apple9_variant_equal(const void *a, const void *b)
{
   return !memcmp(a, b, offsetof(struct hk_apple9_variant, shaders));
}

static void
hk_apple9_variant_destroy(struct hk_device *dev, struct hk_apple9_variant *v)
{
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (v->shaders[s])
         hk_api_shader_destroy(&dev->vk, &v->shaders[s]->vk, NULL);
   }
   util_queue_fence_destroy(&v->ready);
   if (v->cache_object)
      vk_pipeline_cache_object_unref(&dev->vk, v->cache_object);
   free(v);
}

static bool
hk_apple9_remove_generic_output(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location < VARYING_SLOT_VAR0)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* Draw-time variants persist through the device pipeline cache (and so the
 * disk cache), as API shaders do through the runtime. The key covers every
 * input of the variant compile: the deferred stages exactly as the compile
 * reads them, the resolved graphics key, the device features shaders lower
 * against, and the compiler flags (debug, drirc, soft faults, environment
 * knobs).
 */
bool
hk_apple9_variant_cache_key(struct hk_device *dev,
                            struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
                            const struct hk_apple9_graphics_key *key,
                            bool prefer_native, blake3_hash out)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   static const char tag[] = "hk-apple9-variant-v3";
   _mesa_blake3_update(&ctx, tag, sizeof(tag));
   /* The UUID covers the driver build ID, root layout and compiler flags,
    * including when these entries travel inside an application cache. */
   const struct vk_properties *props = &dev->vk.physical->properties;
   _mesa_blake3_update(&ctx, props->pipelineCacheUUID, VK_UUID_SIZE);
   _mesa_blake3_update(&ctx, &props->shaderBinaryVersion,
                       sizeof(props->shaderBinaryVersion));
   _mesa_blake3_update(&ctx, &dev->dev.chip, sizeof(dev->dev.chip));
   const uint64_t flags =
      hk_physical_device_compiler_flags(hk_device_physical(dev));
   _mesa_blake3_update(&ctx, &flags, sizeof(flags));
   const enum hk_feature_key features =
      hk_make_feature_key(&dev->vk.enabled_features);
   _mesa_blake3_update(&ctx, &features, sizeof(features));
   _mesa_blake3_update(&ctx, key, sizeof(*key));
   /* Use the effective request after candidate collapse. Native leaves cannot
    * satisfy an emulated draw, even when every graphics-state field agrees. */
   _mesa_blake3_update(&ctx, &prefer_native, sizeof(prefer_native));
   hk_hash_apple9_native_mesh_abi(&ctx);
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      const uint32_t present = api[s] != NULL;
      _mesa_blake3_update(&ctx, &present, sizeof(present));
      if (!present)
         continue;
      const struct hk_apple9_deferred *d = hk_apple9_deferred_source(api[s]);
      if (!d)
         return false;
      const uint64_t size = d->nir->size;
      const uint32_t fields[] = {d->next_stage_mask, d->set_layout_count,
                                 d->sample_shading, d->zs_self_dep,
                                 d->undefined_colors, d->cull_distance};
      _mesa_blake3_update(&ctx, &size, sizeof(size));
      _mesa_blake3_update(&ctx, d->nir->data, d->nir->size);
      _mesa_blake3_update(&ctx, fields, sizeof(fields));
   }
   _mesa_blake3_final(&ctx, out);
   return true;
}

void
hk_apple9_variant_cache_store(struct hk_device *dev,
                              const unsigned *stages, unsigned count,
                              struct vk_shader *const *compiled,
                              struct blob *blob)
{
   blob_write_uint32(blob, 0x484b5602);
   blob_write_bytes(blob, dev->vk.physical->properties.pipelineCacheUUID,
                     VK_UUID_SIZE);
   blob_write_uint32(blob, dev->vk.physical->properties.shaderBinaryVersion);
   blob_write_uint32(blob, count);
   for (unsigned i = 0; i < count; ++i) {
      blob_write_uint32(blob, stages[i]);
      intptr_t size_at = blob_reserve_uint32(blob);
      const size_t start = blob->size;
      if (size_at < 0 ||
          !compiled[i]->ops->serialize(&dev->vk, compiled[i], blob)) {
         blob->out_of_memory = true;
         return;
      }
      if (blob->size - start > UINT32_MAX) {
         blob->out_of_memory = true;
         return;
      }
      blob_overwrite_uint32(blob, size_at, blob->size - start);
   }
}

/* Returns the number of stages loaded into out, or 0 on a miss. */
unsigned
hk_apple9_variant_cache_load(struct hk_device *dev, const void *data,
                             size_t size, uint32_t expected_stages,
                             unsigned *stages,
                             struct vk_shader **out)
{
   for (unsigned i = 0; i < HK_APPLE9_GFX_STAGES; ++i)
      out[i] = NULL;
   struct blob_reader blob;
   blob_reader_init(&blob, data, size);
   const uint32_t magic = blob_read_uint32(&blob);
   uint8_t uuid[VK_UUID_SIZE];
   blob_copy_bytes(&blob, uuid, sizeof(uuid));
   const uint32_t version = blob_read_uint32(&blob);
   const unsigned count = blob_read_uint32(&blob);
   if (blob.overrun || magic != 0x484b5602 ||
       memcmp(uuid, dev->vk.physical->properties.pipelineCacheUUID, sizeof(uuid)) ||
       version != dev->vk.physical->properties.shaderBinaryVersion ||
       count == 0 || count != util_bitcount(expected_stages))
      return 0;
   uint32_t remaining = expected_stages;
   unsigned i;
   for (i = 0; i < count; ++i) {
      stages[i] = blob_read_uint32(&blob);
      const uint32_t len = blob_read_uint32(&blob);
      const uint8_t *bytes = blob_read_bytes(&blob, len);
      if (blob.overrun || stages[i] >= HK_APPLE9_GFX_STAGES ||
          stages[i] == MESA_SHADER_COMPUTE ||
          !(remaining & BITFIELD_BIT(stages[i])) || !len)
         break;
      remaining &= ~BITFIELD_BIT(stages[i]);
      struct blob_reader stage;
      blob_reader_init(&stage, bytes, len);
      if (dev->vk.shader_ops->deserialize(&dev->vk, &stage, version, NULL,
                                          &out[i]) != VK_SUCCESS ||
          !out[i] || out[i]->stage != stages[i] || stage.overrun ||
          stage.current != stage.end)
         break;
      /* Persistent draw variants contain compiled realizations only. Let the
       * one driver deserializer own the container format, then check the kind. */
      struct hk_api_shader *api = container_of(out[i], struct hk_api_shader, vk);
      if (api->apple9_deferred)
         break;
   }
   if (i == count && !remaining && blob.current == blob.end)
      return count;
   for (unsigned j = 0; j <= i && j < count; ++j) {
      if (out[j])
         hk_api_shader_destroy(&dev->vk, out[j], NULL);
      out[j] = NULL;
   }
   return 0;
}

/* What a draw-time variant compiles from, per stage: NULL nir for a stage the
 * draw does not bind. */
struct hk_apple9_variant_stage {
   struct hk_apple9_nir *nir;
   uint32_t next_stage_mask;
   uint32_t set_layout_count;
};

/* Compile the programs of a variant. On success, compiled[i] holds the
 * program of stages[i] for i < *count_out; on failure nothing is returned. */
static VkResult
hk_apple9_compile_variant(struct hk_device *dev,
                          const struct hk_apple9_variant_stage src[HK_APPLE9_GFX_STAGES],
                          const struct hk_apple9_graphics_key *key,
                          const uint64_t *fs_interp,
                          bool prefer_native,
                          struct vk_shader *compiled[HK_APPLE9_GFX_STAGES],
                          unsigned stages[HK_APPLE9_GFX_STAGES],
                          unsigned *count_out)
{
   struct vk_shader_compile_info infos[HK_APPLE9_GFX_STAGES];
   unsigned count = 0;
   VkResult result = VK_SUCCESS;
   *count_out = 0;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (!src[s].nir)
         continue;
      struct blob_reader reader;
      blob_reader_init(&reader, src[s].nir->data, src[s].nir->size);
      nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
      if (!nir) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }
      infos[count] = (struct vk_shader_compile_info){
         .stage = s,
         .nir = nir,
         .next_stage_mask = src[s].next_stage_mask,
         .robustness = &vk_robustness_disabled,
         .set_layout_count = src[s].set_layout_count,
      };
      stages[count++] = s;
   }
   const bool has_fs = src[MESA_SHADER_FRAGMENT].nir != NULL;

   /* Without a fragment shader nothing reads the generic varyings of a
    * vertex-only draw: drop them so their stores and the arithmetic feeding
    * them die. The variant is keyed on the bound stages, so the fragment
    * shader is known absent for every draw that uses it. Transform feedback
    * still captures them. This runs before the cull-distance null program
    * below is linked: it reads nothing, but as a consumer it would otherwise
    * keep every generic output.
    */
   if (result == VK_SUCCESS && !has_fs && !fs_interp && count == 1 &&
       infos[0].stage == MESA_SHADER_VERTEX && !infos[0].nir->xfb_info) {
      nir_shader *vs = infos[0].nir;
      if (nir_shader_intrinsics_pass(vs, hk_apple9_remove_generic_output,
                                     nir_metadata_control_flow, NULL)) {
         NIR_PASS(_, vs, nir_opt_dce);
         nir_shader_gather_info(vs, nir_shader_get_entrypoint(vs));
      }
   }

   /* Cull distances are applied by the fragment program, so a draw without
    * one links an empty program to the last pre-rasterization stage. */
   if (result == VK_SUCCESS && !has_fs && !fs_interp && count &&
       infos[count - 1].nir->info.cull_distance_array_size) {
      infos[count] = (struct vk_shader_compile_info){
         .stage = MESA_SHADER_FRAGMENT,
         .nir = hk_apple9_null_fs_nir(true),
         .robustness = &vk_robustness_disabled,
      };
      stages[count++] = MESA_SHADER_FRAGMENT;
   }

   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
      return result;
   }

   struct hk_cpu_compile_scope scope;
   hk_cpu_timing_compile_begin(&scope);
   /* Creation-time incompatibility is final. Dynamic state can narrow
    * a candidate further, but cannot make an incompatible shader native. */
   if (!prefer_native) {
      result = hk_apple9_compile_graphics_path(
         dev, count, infos, NULL, key, fs_interp, NULL, compiled, false, NULL);
   } else {
      result = hk_apple9_compile_graphics_mode(
         dev, count, infos, NULL, key, fs_interp, NULL, compiled, true);
   }
   hk_cpu_timing_compile_end(&scope, hk_apple9_eager_compile
                                        ? HK_CPU_COMPILE_EAGER
                                        : HK_CPU_COMPILE_DRAW);
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < count; ++i) {
         if (compiled[i])
            hk_api_shader_destroy(&dev->vk, compiled[i], NULL);
         compiled[i] = NULL;
      }
      return result;
   }
   *count_out = count;
   return VK_SUCCESS;
}

/* A queued precompile holds its own references to the stage NIR, so the
 * pipeline can be destroyed before the job runs. */
struct hk_apple9_precompile_job {
   struct hk_device *dev;
   blake3_hash cache_key;
   struct hk_apple9_graphics_key key;
   bool prefer_native;
   struct hk_apple9_variant_stage src[HK_APPLE9_GFX_STAGES];
};

static void
hk_apple9_precompile_execute(void *data, void *gdata, int thread_index)
{
   struct hk_apple9_precompile_job *job = data;
   struct hk_device *dev = job->dev;
   if (p_atomic_read(&dev->apple9_precompile.abort))
      return;

   /* A draw, or an earlier run, may have compiled it already. */
   struct vk_pipeline_cache_object *object = vk_pipeline_cache_lookup_object(
      dev->vk.mem_cache, job->cache_key, sizeof(job->cache_key),
      &vk_raw_data_cache_object_ops, NULL);
   if (object) {
      vk_pipeline_cache_object_unref(&dev->vk, object);
      return;
   }

   struct vk_shader *compiled[HK_APPLE9_GFX_STAGES] = {NULL};
   unsigned stages[HK_APPLE9_GFX_STAGES];
   unsigned count;
   hk_apple9_eager_compile = true;
   VkResult result = hk_apple9_compile_variant(
      dev, job->src, &job->key, NULL, job->prefer_native, compiled, stages, &count);
   hk_apple9_eager_compile = false;
   if (result != VK_SUCCESS)
      return;

   /* Only the disk cache keeps it: the first draw loads it from there. */
   struct blob blob;
   blob_init(&blob);
   hk_apple9_variant_cache_store(dev, stages, count, compiled, &blob);
   if (!blob.out_of_memory) {
      object = vk_pipeline_cache_create_and_insert_object(
         dev->vk.mem_cache, job->cache_key, sizeof(job->cache_key), blob.data,
         blob.size, &vk_raw_data_cache_object_ops);
      if (object)
         vk_pipeline_cache_object_unref(&dev->vk, object);
   }
   blob_finish(&blob);
   for (unsigned i = 0; i < count; ++i)
      hk_api_shader_destroy(&dev->vk, compiled[i], NULL);
}

static void
hk_apple9_precompile_cleanup(void *data, void *gdata, int thread_index)
{
   struct hk_apple9_precompile_job *job = data;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
      hk_apple9_nir_unref(job->src[s].nir);
   free(job);
}

/* Compile a variant into the pipeline cache on a low-priority thread, so
 * pipeline creation does not wait for it. A draw that needs the variant
 * before the job runs compiles it itself, and the job then finds it cached. */
static void
hk_apple9_queue_precompile(struct hk_device *dev,
                           const struct hk_apple9_variant *search)
{
   if (!dev->apple9_precompile.ready)
      return;
   struct hk_apple9_precompile_job *job = calloc(1, sizeof(*job));
   if (!job)
      return;
   if (!hk_apple9_variant_cache_key(dev, search->api, &search->key,
                                    search->prefer_native, job->cache_key)) {
      free(job);
      return;
   }
   job->dev = dev;
   job->key = search->key;
   job->prefer_native = search->prefer_native;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      const struct hk_api_shader *api = search->api[s];
      if (!api)
         continue;
      const struct hk_apple9_deferred *d = hk_apple9_deferred_source(api);
      if (!d) {
         for (unsigned j = 0; j < s; ++j)
            hk_apple9_nir_unref(job->src[j].nir);
         free(job);
         return;
      }
      job->src[s] = (struct hk_apple9_variant_stage){
         .nir = hk_apple9_nir_ref(d->nir),
         .next_stage_mask = d->next_stage_mask,
         .set_layout_count = d->set_layout_count,
      };
   }
   util_queue_add_job(&dev->apple9_precompile.queue, job, NULL,
                      hk_apple9_precompile_execute,
                      hk_apple9_precompile_cleanup, 0);
}

void
hk_apple9_precompile_init(struct hk_device *dev)
{
   /* Without a disk cache, a draw could not load a precompiled variant. */
   if (!dev->vk.mem_cache || !dev->vk.physical->disk_cache)
      return;
   /* One thread: each precompile thread compiles in its own glibc malloc
    * arena, which keeps its freed pages for the life of the process (two
    * threads left a cold ER replay 43 MiB larger on the mini), and trimming
    * them takes the application's arena lock too (14-16 ms against a game's
    * heap). One thread drains ER's 3,009 precompiles (32 s of CPU) well within
    * its two-minute load; a draw that outruns the queue compiles itself. */
   const unsigned threads = 1;
   dev->apple9_precompile.ready = util_queue_init(
      &dev->apple9_precompile.queue, "hk_precomp", 32, threads,
      UTIL_QUEUE_INIT_RESIZE_IF_FULL | UTIL_QUEUE_INIT_USE_MINIMUM_PRIORITY,
      NULL);
}

void
hk_apple9_precompile_finish(struct hk_device *dev)
{
   if (!dev->apple9_precompile.ready)
      return;
   /* Queued jobs are dropped, not compiled: their cleanup still runs. */
   p_atomic_set(&dev->apple9_precompile.abort, true);
   util_queue_finish(&dev->apple9_precompile.queue);
   util_queue_destroy(&dev->apple9_precompile.queue);
   dev->apple9_precompile.ready = false;
}

VkResult
hk_apple9_resolve_graphics(
   struct hk_device *dev, struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
   const struct vk_graphics_pipeline_state *state,
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES], bool prefer_native,
   bool fs_statistics)
{
   struct hk_apple9_variant search;
   memset(&search, 0, sizeof(search));
   hk_apple9_graphics_key_device(&search.key, state, hk_device_physical(dev));
   search.key.no_fs_statistics = !fs_statistics;

   /* Determine whether all bound stages must compile together (cross-stage
    * lowering), or whether already-compiled GPL/ESO stages can pass through
    * unchanged and link via hardware CF_BINDINGS + VS VARY_STORE patching. */
   const struct hk_apple9_deferred *fs_d =
      hk_apple9_deferred_source(api[MESA_SHADER_FRAGMENT]);
   bool any_cull_distance = false;
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
      const struct hk_apple9_deferred *d = hk_apple9_deferred_source(api[s]);
      if (d && d->cull_distance)
         any_cull_distance = true;
   }
   bool has_non_vs_prerast =
      api[MESA_SHADER_TESS_CTRL] || api[MESA_SHADER_TESS_EVAL] ||
      api[MESA_SHADER_GEOMETRY] || api[MESA_SHADER_MESH];
   bool need_cross_stage =
      search.key.line_stipple || search.key.polygon_points ||
      any_cull_distance ||
      (fs_d && fs_d->fs_uses_per_vertex) ||
      (fs_d && fs_d->fs_reads_viewport) ||
      (fs_d && fs_d->fs_reads_prim_id && has_non_vs_prerast);

   struct hk_api_shader *resolve_api[HK_APPLE9_GFX_STAGES] = {NULL};
   struct hk_api_shader *passthrough[HK_APPLE9_GFX_STAGES] = {NULL};
   bool any_resolve = false;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (!api[s])
         continue;
      if (need_cross_stage || api[s]->apple9_deferred || !api[s]->compiled ||
          (s == MESA_SHADER_VERTEX && has_non_vs_prerast &&
           api[s]->apple9_fallback) ||
          (s == MESA_SHADER_VERTEX && search.key.vertex.rasterize_points &&
           api[s]->apple9_fallback &&
           api[s]->apple9_fallback->vs_dynamic_topology) ||
          (s == MESA_SHADER_FRAGMENT && search.key.zs_self_dep &&
           api[s]->apple9_fallback)) {
         resolve_api[s] = api[s];
         any_resolve = true;
      } else {
         passthrough[s] = api[s];
      }
   }
   if (!any_resolve) {
      memcpy(out, passthrough, sizeof(passthrough));
      return VK_SUCCESS;
   }

   memcpy(search.api, resolve_api, sizeof(search.api));
   uint64_t fs_interp_vals[2] = {0, 0};
   const uint64_t *fs_interp = NULL;
   if (passthrough[MESA_SHADER_FRAGMENT]) {
      const struct hk_shader *pfs =
         hk_any_variant(passthrough[MESA_SHADER_FRAGMENT]);
      fs_interp_vals[0] = pfs->info.fs.interp.flat;
      fs_interp_vals[1] = pfs->info.fs.interp.linear;
      search.key.vertex.outputs_flat = fs_interp_vals[0];
      search.key.vertex.outputs_linear = fs_interp_vals[1];
      search.key.has_passthrough_fs = true;
      fs_interp = fs_interp_vals;
   }
   if (!resolve_api[MESA_SHADER_VERTEX] && !resolve_api[MESA_SHADER_TESS_CTRL] &&
       !resolve_api[MESA_SHADER_TESS_EVAL] && !resolve_api[MESA_SHADER_GEOMETRY] &&
       !resolve_api[MESA_SHADER_TASK] && !resolve_api[MESA_SHADER_MESH]) {
      /* Fragment-only resolution: ignore vertex-input and clip-Z fields so the
       * resolved FS variant is shared across all paired vertex shaders. */
      memset(&search.key.vertex, 0, sizeof(search.key.vertex));
      search.key.clip_negative_one_to_one = false;
      search.key.clip_negative_one_to_one_dynamic = false;
   }

   search.key.native_task_bridge = resolve_api[MESA_SHADER_MESH] &&
      resolve_api[MESA_SHADER_TASK] && hk_apple9_native_task_bridge_enabled();
   const struct hk_api_shader *vs = resolve_api[MESA_SHADER_VERTEX];
   const struct hk_apple9_deferred *vs_d = hk_apple9_deferred_source(vs);
   search.key.vertex.unbounded = vs_d && vs_d->vertex_unbounded;
   mesa_shader_stage producer =
      resolve_api[MESA_SHADER_MESH]        ? MESA_SHADER_MESH
      : resolve_api[MESA_SHADER_GEOMETRY]  ? MESA_SHADER_GEOMETRY
      : resolve_api[MESA_SHADER_TESS_EVAL] ? MESA_SHADER_TESS_EVAL
                                           : MESA_SHADER_MESH;
   /* Collapse ineligible requests onto the emulated cache entry. */
   search.prefer_native = prefer_native && resolve_api[producer] &&
                          p_atomic_read(&resolve_api[producer]->apple9_native_candidate);
   const struct hk_api_shader *fs = resolve_api[MESA_SHADER_FRAGMENT];
   const struct hk_apple9_deferred *rfs_d = hk_apple9_deferred_source(fs);
   if (rfs_d) {
      search.key.undefined_colors = rfs_d->undefined_colors;
      u_foreach_bit(rt, rfs_d->undefined_colors)
         search.key.blend[rt].color_attachment_unused = true;
   }
   /* Decouple dynamic vertex input from deferred pure-VS pipelines (shader
    * objects, GPL, dynamic blend/MSAA) so changing vertex formats/offsets
    * swaps only the tiny cached vertex prolog instead of recompiling vs_main. */
   if (vs_d && vs_d->vs_can_prolog &&
       !resolve_api[MESA_SHADER_TESS_CTRL] && !resolve_api[MESA_SHADER_TESS_EVAL] &&
       !resolve_api[MESA_SHADER_GEOMETRY] && !resolve_api[MESA_SHADER_MESH] &&
       !(rfs_d && rfs_d->fs_uses_per_vertex) &&
       !search.key.line_stipple && !search.key.polygon_points) {
      search.key.vertex.use_prolog = true;
      memset(search.key.vertex.format, 0, sizeof(search.key.vertex.format));
      memset(search.key.vertex.offset, 0, sizeof(search.key.vertex.offset));
      memset(search.key.vertex.stride, 0, sizeof(search.key.vertex.stride));
      memset(search.key.vertex.divisor, 0, sizeof(search.key.vertex.divisor));
      memset(search.key.vertex.buffer, 0, sizeof(search.key.vertex.buffer));
   }
   /* Decouple color/blend/MSAA/color_map from deferred fragment shaders so
    * draw-time blend/format/MSAA changes swap only the cached epilog. */
   if (rfs_d && rfs_d->fs_can_epilog &&
       !search.key.sample_shading && !search.key.zs_self_dep) {
      uint8_t samples = rfs_d->fs_uses_sample_rate
                           ? MAX2(search.key.blend[0].samples, 1)
                           : 1;
      search.key.fs_use_epilog = true;
      search.key.targets = 0;
      memset(search.key.color_map, 0, sizeof(search.key.color_map));
      memset(search.key.blend, 0, sizeof(search.key.blend));
      search.key.blend[0].samples = samples;
      search.key.blend[0].dynamic_samples = !rfs_d->fs_uses_sample_rate;
      search.key.blend[0].sample_mask_single_sample = true;
      search.key.blend[0].dynamic_sample_positions = true;
   }

   /* A precompile neither finds nor publishes a live variant: it queues a
    * compile into the pipeline cache, which the first draw loads from. */
   if (hk_apple9_precompile) {
      hk_apple9_queue_precompile(dev, &search);
      return VK_SUCCESS;
   }

   struct hk_apple9_variant *v;

retry:
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (!dev->apple9_variants.ht) {
      dev->apple9_variants.ht = _mesa_hash_table_create(
         NULL, hk_apple9_variant_hash, hk_apple9_variant_equal);
   }
   struct hash_entry *entry =
      dev->apple9_variants.ht
         ? _mesa_hash_table_search(dev->apple9_variants.ht, &search)
         : NULL;
   if (!entry && dev->apple9_variants.ht && !search.prefer_native) {
      /* A rejected native request already owns emulated programs under its
       * original key. Reuse that completed entry without moving/freeing it:
       * waiters may still hold its fence, and another thread may already be
       * compiling the emulated key. Do not wait on unrelated in-flight work. */
      struct hk_apple9_variant alternate = search;
      alternate.prefer_native = true;
      struct hash_entry *fallback =
         _mesa_hash_table_search(dev->apple9_variants.ht, &alternate);
      if (fallback) {
         struct hk_apple9_variant *f = fallback->data;
         if (util_queue_fence_is_signalled(&f->ready) &&
             f->result == VK_SUCCESS && f->shaders[producer] &&
             !f->shaders[producer]->apple9_native_leaf)
            entry = fallback;
      }
   }
   if (entry) {
      v = entry->data;
      if (!util_queue_fence_is_signalled(&v->ready)) {
         simple_mtx_unlock(&dev->apple9_variants.lock);
         util_queue_fence_wait(&v->ready);
         goto retry;
      }
      if (v->result != VK_ERROR_OUT_OF_HOST_MEMORY &&
          v->result != VK_ERROR_OUT_OF_DEVICE_MEMORY) {
         VkResult result = v->result;
         if (result == VK_SUCCESS) {
            for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
               out[s] = passthrough[s] ? passthrough[s] : v->shaders[s];
         }
         simple_mtx_unlock(&dev->apple9_variants.lock);
         return result;
      }

      /* Retry temporary allocation failures in place. Other recording threads
       * may still be waiting on this fence, so removing/freeing the entry is
       * unsafe. Claim the next compile while holding the table lock.
       */
      util_queue_fence_reset(&v->ready);
      simple_mtx_unlock(&dev->apple9_variants.lock);
      goto compile;
   }
   if (!dev->apple9_variants.ht) {
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   /* Publish the variant before compiling, then compile without the lock:
    * a draw-time compile takes up to a second, and other recording threads
    * must not block on it for unrelated variants. */
   v = calloc(1, sizeof(*v));
   if (!v) {
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   memcpy(v, &search, offsetof(struct hk_apple9_variant, shaders));
   util_queue_fence_init(&v->ready);
   util_queue_fence_reset(&v->ready);
   _mesa_hash_table_insert(dev->apple9_variants.ht, v, v);
   simple_mtx_unlock(&dev->apple9_variants.lock);

compile:;
   /* HK_APPLE9_VARIANT_CACHE_VERIFY=1 compiles on a cache hit as well and
    * reports any difference, to catch inputs missing from the key. */
   const bool verify = debug_get_option_apple9_variant_cache_verify();
   blake3_hash cache_key;
   const bool cacheable =
      dev->vk.mem_cache &&
      hk_apple9_variant_cache_key(dev, resolve_api, &search.key,
                                  search.prefer_native, cache_key);
   struct vk_shader *cached[HK_APPLE9_GFX_STAGES];
   unsigned cached_stages[HK_APPLE9_GFX_STAGES];
   unsigned cached_count = 0;
   struct vk_pipeline_cache_object *cache_object = NULL;
   if (cacheable) {
      uint32_t expected_stages = 0;
      const struct hk_apple9_deferred *last = NULL;
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
         if (resolve_api[s]) {
            expected_stages |= BITFIELD_BIT(s);
            if (s != MESA_SHADER_FRAGMENT)
               last = hk_apple9_deferred_source(resolve_api[s]);
         }
      }
      if (!resolve_api[MESA_SHADER_FRAGMENT] && last && last->cull_distance)
         expected_stages |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);
      cache_object = vk_pipeline_cache_lookup_object(
         dev->vk.mem_cache, cache_key, sizeof(cache_key),
         &vk_raw_data_cache_object_ops, NULL);
      if (cache_object) {
         struct vk_raw_data_cache_object *raw =
            container_of(cache_object, struct vk_raw_data_cache_object, base);
         cached_count = hk_apple9_variant_cache_load(
            dev, raw->data, raw->data_size, expected_stages, cached_stages, cached);
         if (!cached_count) {
            vk_pipeline_cache_object_unref(&dev->vk, cache_object);
            cache_object = NULL;
         }
      }
   }
   if (cached_count && !verify) {
      for (unsigned i = 0; i < cached_count; ++i) {
         v->shaders[cached_stages[i]] =
            container_of(cached[i], struct hk_api_shader, vk);
      }
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         out[s] = passthrough[s] ? passthrough[s] : v->shaders[s];
      v->cache_object = cache_object;
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         if (resolve_api[s])
            hk_apple9_remember_variant(dev, resolve_api[s], cache_object);
      simple_mtx_lock(&dev->apple9_variants.lock);
      if (search.prefer_native && resolve_api[producer] &&
          !resolve_api[MESA_SHADER_TASK] &&
          !v->shaders[producer]->apple9_native_leaf)
         p_atomic_set(&resolve_api[producer]->apple9_native_candidate, false);
      v->result = VK_SUCCESS;
      util_queue_fence_signal(&v->ready);
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return VK_SUCCESS;
   }

   struct hk_apple9_variant_stage src[HK_APPLE9_GFX_STAGES] = {0};
   VkResult result = VK_SUCCESS;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (!resolve_api[s])
         continue;
      const struct hk_apple9_deferred *d =
         hk_apple9_deferred_source(resolve_api[s]);
      if (!d) {
         /* Draw-time variants recompile every stage of the draw. */
         result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                            "Apple9: deferred and precompiled stages mixed");
         break;
      }
      uint32_t next_stage_mask = d->next_stage_mask;
      if (s == MESA_SHADER_VERTEX) {
         if (resolve_api[MESA_SHADER_TESS_CTRL])
            next_stage_mask |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
         if (resolve_api[MESA_SHADER_GEOMETRY])
            next_stage_mask |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }
      src[s] = (struct hk_apple9_variant_stage){
         .nir = d->nir,
         .next_stage_mask = next_stage_mask,
         .set_layout_count = d->set_layout_count,
      };
   }

   struct vk_shader *compiled[HK_APPLE9_GFX_STAGES] = {NULL};
   unsigned stages[HK_APPLE9_GFX_STAGES];
   unsigned count = 0;
   if (result == VK_SUCCESS) {
      result = hk_apple9_compile_variant(dev, src, &search.key, fs_interp,
                                         search.prefer_native, compiled,
                                         stages, &count);
   }

   /* Keep permanent failures, but allow a later caller to retry an allocation
    * failure once temporary memory pressure has passed. */
   if (result == VK_SUCCESS) {
      if (cacheable) {
         struct blob blob;
         blob_init(&blob);
         hk_apple9_variant_cache_store(dev, stages, count, compiled,
                                       &blob);
         if (cached_count) {
            struct blob check;
            blob_init(&check);
            hk_apple9_variant_cache_store(dev, cached_stages,
                                          cached_count, cached, &check);
            if (blob.out_of_memory || check.out_of_memory ||
                blob.size != check.size ||
                memcmp(blob.data, check.data, blob.size)) {
               mesa_loge("hk: draw-time variant cache mismatch pid=%u (%zu vs %zu "
                         "bytes): an input is missing from the key",
                         (unsigned)getpid(), blob.size, check.size);
            } else {
               mesa_logi("hk: draw-time variant cache verified pid=%u (%zu bytes)",
                         (unsigned)getpid(), blob.size);
            }
            blob_finish(&check);
         } else if (!blob.out_of_memory) {
            cache_object = vk_pipeline_cache_create_and_insert_object(
                  dev->vk.mem_cache, cache_key, sizeof(cache_key), blob.data,
                  blob.size, &vk_raw_data_cache_object_ops);
         }
         blob_finish(&blob);
      }
      for (unsigned i = 0; i < count; ++i) {
         v->shaders[stages[i]] =
            container_of(compiled[i], struct hk_api_shader, vk);
      }
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         out[s] = passthrough[s] ? passthrough[s] : v->shaders[s];
   }
   for (unsigned i = 0; i < cached_count; ++i)
      hk_api_shader_destroy(&dev->vk, cached[i], NULL);
   if (result != VK_SUCCESS && cache_object) {
      vk_pipeline_cache_object_unref(&dev->vk, cache_object);
      cache_object = NULL;
   }
   v->cache_object = cache_object;
   if (cache_object) {
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         if (resolve_api[s])
            hk_apple9_remember_variant(dev, resolve_api[s], cache_object);
   }
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (result == VK_SUCCESS && search.prefer_native && resolve_api[producer] &&
       !resolve_api[MESA_SHADER_TASK] && !v->shaders[producer]->apple9_native_leaf)
      p_atomic_set(&resolve_api[producer]->apple9_native_candidate, false);
   v->result = result;
   util_queue_fence_signal(&v->ready);
   simple_mtx_unlock(&dev->apple9_variants.lock);
   return result;
}

/* Variants live as long as all of their API shaders. */
void
hk_apple9_forget_deferred(struct hk_device *dev, struct hk_api_shader *obj)
{
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (dev->apple9_variants.ht) {
      hash_table_foreach(dev->apple9_variants.ht, entry) {
         struct hk_apple9_variant *v = entry->data;
         bool uses = false;
         for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
            uses |= v->api[s] == obj;
         if (uses) {
            _mesa_hash_table_remove(dev->apple9_variants.ht, entry);
            hk_apple9_variant_destroy(dev, v);
         }
      }
   }
   simple_mtx_unlock(&dev->apple9_variants.lock);
}

void
hk_apple9_variants_finish(struct hk_device *dev)
{
   if (dev->apple9_variants.ht) {
      hash_table_foreach(dev->apple9_variants.ht, entry)
         hk_apple9_variant_destroy(dev, entry->data);
      _mesa_hash_table_destroy(dev->apple9_variants.ht, NULL);
      dev->apple9_variants.ht = NULL;
   }
}

