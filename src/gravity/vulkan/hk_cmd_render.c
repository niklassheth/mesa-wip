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

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         const VkRenderingAttachmentLocationInfoKHR att_loc_info_default = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
            .colorAttachmentCount = inheritance_info->colorAttachmentCount,
         };
         const VkRenderingAttachmentLocationInfoKHR *att_loc_info =
            vk_get_command_buffer_rendering_attachment_location_info(
               cmd->vk.level, pBeginInfo);
         if (att_loc_info == NULL)
            att_loc_info = &att_loc_info_default;

         vk_cmd_set_rendering_attachment_locations(&cmd->vk, att_loc_info);

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
}

static void
hk_attachment_use_implicit_sync(struct hk_cmd_buffer *cmd,
                                const struct hk_attachment *att)
{
   if (att->iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
   if (att->resolve_iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->resolve_iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }

   const VkRenderingAttachmentFlagsInfoKHR *flags =
      vk_find_struct_const(info->pNext, RENDERING_ATTACHMENT_FLAGS_INFO_KHR);
   if (flags)
      att->flags = flags->flags;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area)
{
   return hk_apple9_build_bg_eot(cmd, info, store, partial_render,
                                 incomplete_render_area);
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

/* A continued control stream loads what the previous one stored. Only the
 * last control stream of a render pass may drop stores, and only Apple9
 * honours that: its firmware preserves depth/stencil across partial renders
 * independently of these stores.
 */
static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render, bool last,
                    const bool *chain_load)
{
   struct agx_zls zls = {0};

   if (z) {
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && z->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(z->imageLayout,
                                               VK_IMAGE_ASPECT_DEPTH_BIT))
         zls.z_store = false;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && s->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(s->imageLayout,
                                               VK_IMAGE_ASPECT_STENCIL_BIT))
         zls.s_store = false;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   /* A merged or suspend/resume chain loads as its first render pass did. */
   if (chain_load && !partial_render) {
      zls.z_load = z && chain_load[0];
      zls.s_load = s && chain_load[1];
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

/* Raw depth/stencil planes keep their compressed layout and metadata, which
 * marks every tile uncompressed, but ZLS must neither decode nor re-encode
 * them while the render may sample them.
 */
static void
hk_zls_control_make_raw(struct agx_zls_control_packed *packed)
{
   struct agx_zls_control_packed compress;
   agx_pack(&compress, ZLS_CONTROL, cfg) {
      cfg.z_load_compress = cfg.z_store_compress = true;
      cfg.s_load_compress = cfg.s_store_compress = true;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(packed->opaque); ++i)
      packed->opaque[i] &= ~compress.opaque[i];
}

/* Render-pass merging: continue the hardware render of a render pass in the
 * next render pass when nothing but attachment barriers separates them and
 * both use the same colour attachments. The render covers the whole
 * attachment whatever the render area, and tiles keep their contents between
 * the passes, which satisfies the second pass's load. Depth/stencil views and
 * render areas must also match. As with suspend/resume, keep the first pass's
 * ZLS loads and the final pass's stores.
 */
/* Whether the render pass's draws may read this attachment other than as an
 * attachment (render->feedback). A written attachment may be read so only
 * when feedback loop is enabled for it: the feedback loop layout, or
 * VkAttachmentFeedbackLoopInfoEXT under unified image layouts (as RADV
 * reads it). An attachment nothing has written yet, with LOAD_OP_NONE, may
 * also be sampled in GENERAL. GENERAL alone is not a feedback loop, so
 * renders in GENERAL keep barrier hoisting.
 */
static bool
hk_attachment_may_be_sampled(const VkRenderingAttachmentInfo *att)
{
   if (!att || !att->imageView)
      return false;
   if (att->imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
      return true;
   const VkAttachmentFeedbackLoopInfoEXT *loop =
      vk_find_struct_const(att->pNext, ATTACHMENT_FEEDBACK_LOOP_INFO_EXT);
   if (loop && loop->feedbackLoopEnable)
      return true;
   return att->imageLayout == VK_IMAGE_LAYOUT_GENERAL &&
          att->loadOp == VK_ATTACHMENT_LOAD_OP_NONE;
}

static bool
hk_render_merge_enabled(struct hk_cmd_buffer *cmd)
{
   return cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta;
}

/* Key a render pass that may merge, from its attachment views, or return false.
 * The caller checks load and store operations.
 */
bool
hk_render_merge_key_init(struct hk_render_merge_key *key, VkRenderingFlags flags,
                    uint32_t layer_count, uint32_t view_mask,
                    uint32_t color_count, struct hk_image_view *const *views,
                    struct hk_image_view *depth, struct hk_image_view *stencil,
                    uint8_t ds_read_only, VkRect2D area)
{
   if ((flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT |
                 VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) ||
       color_count > HK_MAX_RTS)
      return false;
   /* A depth/stencil-only pass has the same fixed framebuffer and ZLS state
    * as one with colour attachments. The depth/stencil key, loads, stores
    * and barriers below already constrain its continuation. Attachmentless
    * passes still have no sample count until draw time, so keep them out.
    */
   if (!color_count && !depth && !stencil)
      return false;
   /* Merged multiview renders fail multiview.readback_implicit_clear 8_1_1_8
    * (cause not understood yet); keep multiview renders separate. */
   if (view_mask)
      return false;

   memset(key, 0, sizeof(*key));
   key->color_count = color_count;
   key->layer_count = layer_count;
   key->view_mask = view_mask;
   key->ds_read_only = ds_read_only;
   if (depth || stencil)
      key->ds_area = area;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      const struct hk_image_view *view = i == HK_MAX_RTS ? depth :
         i == HK_MAX_RTS + 1 ? stencil : i < color_count ? views[i] : NULL;
      if (view) {
         key->image[i] = view->vk.image;
         key->level[i] = view->vk.base_mip_level;
         key->layer[i] = view->vk.base_array_layer;
         key->format[i] = view->vk.format;
      }
   }
   return true;
}

static bool
hk_render_merge_begin(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};
   struct hk_render_merge_key key;
   bool empty = cmd->empty_barriers.size != 0;
   if (empty && memcmp(&info->renderArea, &cmd->merge_empty_render->area,
                       sizeof(info->renderArea)))
      return false;
   const VkRenderingAttachmentInfo *ds[2] = {
      info->pDepthAttachment, info->pStencilAttachment};
   struct hk_image_view *ds_views[2] = {NULL};
   uint8_t ds_store_mask = 0, ds_read_only = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (!ds[i] || !ds[i]->imageView)
         continue;
      if (ds[i]->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
          ds[i]->resolveMode != VK_RESOLVE_MODE_NONE ||
          hk_attachment_may_be_sampled(ds[i]) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops &
             (i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT)))
         return false;
      if (empty && (ds[i]->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
                    (ds[i]->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout != (i ? VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL :
                                               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)) ||
                    hk_attachment_may_be_sampled(ds[i])))
         return false;
      ds_views[i] = hk_image_view_from_handle(ds[i]->imageView);
      if (vk_image_layout_is_read_only(ds[i]->imageLayout,
            i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
         ds_read_only |= BITFIELD_BIT(i);
      if (is_attachment_stored(ds[i]))
         ds_store_mask |= BITFIELD_BIT(i);
   }
   if (cmd->merge_ds_store_mask & ~ds_store_mask)
      return false;

   uint32_t store_mask = 0;
   for (unsigned i = 0; i < MIN2(info->colorAttachmentCount, HK_MAX_RTS); ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      views[i] = hk_image_view_from_handle(att->imageView);
      if (empty && att->imageView &&
          (att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
           (att->imageLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
            att->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) ||
           hk_attachment_may_be_sampled(att)))
         return false;
      /* Clears would need an in-pass clear over the render area. */
      if (att->imageView &&
          (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
           hk_attachment_may_be_sampled(att) ||
           (cmd->vk.dynamic_graphics_state.feedback_loops & VK_IMAGE_ASPECT_COLOR_BIT)))
         return false;
      if (att->imageView && is_attachment_stored(att))
         store_mask |= BITFIELD_BIT(i);
   }
   /* The merged render ends with this pass's store operations. An attachment
    * an earlier pass stored but this one does not would lose the earlier
    * pass's results outside this pass's render area. */
   if (cmd->merge_store_mask & ~store_mask) {
      hk_render_merge_trace("begin: store", cmd->merge_store_mask, store_mask);
      return false;
   }

   return !cmd->merge_feedback && hk_render_merge_enabled(cmd) &&
          hk_render_merge_key_init(&key,
                              info->flags & ~(empty ? VK_RENDERING_SUSPENDING_BIT : 0),
                              info->layerCount,
                              info->viewMask, info->colorAttachmentCount, views,
                              ds_views[0], ds_views[1], ds_read_only, info->renderArea) &&
          !memcmp(&key, &cmd->merge_key, sizeof(key));
}

/* A suspended render pass keeps its render open for the resuming one. Only
 * state commands may come between them, so the resuming pass continues the
 * same render pass instance in one hardware render: loads from the first
 * pass, stores from the last. */
static bool
hk_render_suspend_keeps_open(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   return cs && !render->feedback && !render->raw_color_mask &&
          !render->raw_zs &&
          (render->flags & VK_RENDERING_SUSPENDING_BIT) &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta &&
          !cmd->sync_fallback && !cmd->current_cs.pre_gfx &&
          !cmd->current_cs.post_gfx && !cmd->tess_batch.draws;
}

/* At the end of a render pass, whether its render can stay open. */
static bool
hk_render_merge_end(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};

   if (!cs || cs->timestamp.end.handle || render->feedback ||
       render->raw_color_mask || render->raw_zs ||
       !hk_render_merge_enabled(cmd) ||
       cmd->sync_fallback || cmd->current_cs.pre_gfx ||
       cmd->current_cs.post_gfx || cmd->tess_batch.draws ||
       cmd->state.gfx.occlusion.active_count)
      return false;

   for (unsigned i = 0; i < MIN2(render->color_att_count, HK_MAX_RTS); ++i) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         return false;
      views[i] = render->color_att[i].iview;
   }

   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE ||
       render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      return false;

   /* A render pass that resumed in its render ends it like any other. */
   return hk_render_merge_key_init(
      &cmd->merge_key,
      render->flags & ~(render->resumed ? VK_RENDERING_RESUMING_BIT : 0),
      render->layer_count, render->view_mask,
      render->color_att_count, views,
      render->depth_att.iview, render->stencil_att.iview,
      render->ds_read_only, render->area);
}

/* A merged empty chain can still end without any draws. Retain the first
 * pass's clear values so empty-render conversion does not lose them when a
 * later LOAD pass, including suspend/resume, supplies the final render state.
 */
static void
hk_render_inherit_empty_clears(struct hk_cmd_buffer *cmd)
{
   if (!cmd->merge_empty)
      return;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const struct hk_rendering_state *first = cmd->merge_empty_render;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      struct hk_attachment *att = i == HK_MAX_RTS ? &render->depth_att :
         i == HK_MAX_RTS + 1 ? &render->stencil_att : &render->color_att[i];
      const struct hk_attachment *old = i == HK_MAX_RTS ? &first->depth_att :
         i == HK_MAX_RTS + 1 ? &first->stencil_att : &first->color_att[i];
      att->clear = old->clear;
      memcpy(att->clear_colour, old->clear_colour, sizeof(att->clear_colour));
   }
}

/* A compressed attachment must use raw stores as soon as feedback is possible.
 * GENERAL alone does not imply sampling, but the layout/enable bit or explicit
 * dynamic feedback state does. Keep raw stores sticky within the render.
 */
static uint8_t
hk_feedback_color_mask(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint8_t mask = 0;
   for (unsigned i = 0; i < info->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *a = &info->pColorAttachments[i];
      if (!a->imageView)
         continue;
      const struct hk_image_view *view = hk_image_view_from_handle(a->imageView);
      const struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
      /* Decompression copies view depth/stencil planes as colour. */
      const struct ail_layout *layout =
         &image->planes[view->planes[0].image_plane].layout;
      if (!hk_feedback_can_compress(&dev->dev, image->vk.image_type,
                                    image->vk.format, image->vk.samples) ||
          !ail_is_level_logically_compressed(layout, view->vk.base_mip_level))
         continue;
      if (cmd->feedback_decompress || hk_attachment_may_be_sampled(a) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops & VK_IMAGE_ASPECT_COLOR_BIT))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/* The depth/stencil view whose compressed planes must be raw in this render,
 * if any. Sampling either aspect in a feedback loop makes the whole image raw:
 * its raw state is tracked per image, not per plane.
 */
static const struct hk_image_view *
hk_feedback_zs_view(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* Meta renders inherit the application's dynamic feedback state. */
   if (cmd->in_meta)
      return NULL;
   const VkRenderingAttachmentInfo *ds[2] = {info->pDepthAttachment,
                                             info->pStencilAttachment};
   const struct hk_image_view *view = NULL;
   bool sampled = cmd->vk.dynamic_graphics_state.feedback_loops &
                  (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   for (unsigned i = 0; i < ARRAY_SIZE(ds); ++i) {
      if (ds[i] && ds[i]->imageView) {
         view = hk_image_view_from_handle(ds[i]->imageView);
         sampled |= hk_attachment_may_be_sampled(ds[i]);
      }
   }
   if (!view || !sampled)
      return NULL;
   const struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   if (!hk_feedback_can_compress(&dev->dev, image->vk.image_type,
                                 image->vk.format, image->vk.samples))
      return NULL;
   for (unsigned p = 0; p < image->plane_count; ++p) {
      if (ail_is_level_logically_compressed(&image->planes[p].layout,
                                            view->vk.base_mip_level))
         return view;
   }
   return NULL;
}

static bool
hk_feedback_decompress_layers(struct hk_cmd_buffer *cmd,
                              const VkRenderingInfo *info,
                              const struct hk_image_view *view)
{
   if (info->viewMask) {
      u_foreach_bit(v, info->viewMask) {
         if (!hk_meta_decompress_view(cmd, view, view->vk.base_array_layer + v))
            return false;
      }
   } else {
      for (unsigned v = 0; v < info->layerCount; v++) {
         if (!hk_meta_decompress_view(cmd, view, view->vk.base_array_layer + v))
            return false;
      }
   }
   return true;
}

static bool
hk_feedback_decompress(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                         uint8_t mask, const struct hk_image_view *zs)
{
   if (cmd->feedback_decompress)
      return true;
   u_foreach_bit(i, mask) {
      const struct hk_image_view *view =
         hk_image_view_from_handle(info->pColorAttachments[i].imageView);
      if (!hk_feedback_decompress_layers(cmd, info, view))
         return false;
   }
   return !zs || hk_feedback_decompress_layers(cmd, info, zs);
}

static void
hk_feedback_make_zs_raw(struct hk_rendering_state *render)
{
   render->raw_zs = true;
   hk_zls_control_make_raw(&render->cr.zls_control);
   hk_zls_control_make_raw(&render->cr.zls_control_partial);
   hk_zls_control_make_raw(&render->zls_last);
   hk_zls_control_make_raw(&render->zls_partial_last);
}

/* Splitting a live render keeps the partial EOT's stores and the partial BG's
 * loads. Rebuild only its final/partial colour stores with the new raw mask.
 * This also handles feedback enabled by binding a static-state pipeline.
 */
bool
hk_feedback_prepare_draw(struct hk_cmd_buffer *cmd)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!cmd->rendering || cmd->in_meta ||
       !cmd->vk.dynamic_graphics_state.feedback_loops)
      return true;
   VkRenderingAttachmentInfo attachments[HK_MAX_RTS];
   for (unsigned i = 0; i < render->color_att_count; i++) {
      attachments[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = render->store_mask & BITFIELD_BIT(i)
                       ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
      };
   }
   VkRenderingAttachmentInfo zs[2];
   for (unsigned i = 0; i < 2; i++) {
      zs[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(
            i ? render->stencil_att.iview : render->depth_att.iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      };
   }
   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = attachments,
      .pDepthAttachment = &zs[0],
      .pStencilAttachment = &zs[1],
   };
   uint8_t mask = hk_feedback_color_mask(cmd, &info) & ~render->raw_color_mask;
   const struct hk_image_view *zs_view =
      render->raw_zs ? NULL : hk_feedback_zs_view(cmd, &info);
   if (!mask && !zs_view)
      return true;

   hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
   hk_cmd_buffer_end_graphics(cmd);
   struct hk_rendering_state saved = *render;
   if (!hk_feedback_decompress(cmd, &info, mask, zs_view))
      return false;
   *render = saved;
   cmd->rendering = true;
   render->feedback = true;
   render->raw_color_mask |= mask;
   if (zs_view)
      hk_feedback_make_zs_raw(render);
   render->cr.bg.main = render->cr.bg.partial;
   for (unsigned i = 0; i < render->color_att_count; i++)
      render->color_att[i].clear = false;
   render->cr.eot.main = hk_build_bg_eot(cmd, &info, true, false, true);
   render->cr.eot.partial = hk_build_bg_eot(cmd, &info, true, true, true);
   cmd->sync_render_continuation = false;
   cmd->state.gfx.render_gen++;
   hk_cmd_buffer_dirty_all(cmd);
   return !vk_command_buffer_has_error(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetAttachmentFeedbackLoopEnableEXT(VkCommandBuffer commandBuffer,
                                          VkImageAspectFlags aspectMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) ||
       dyn->feedback_loops != aspectMask) {
      dyn->feedback_loops = aspectMask;
      BITSET_SET(dyn->set, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
   }
   if (cmd->rendering && aspectMask)
      cmd->state.gfx.render.feedback = true;
   hk_feedback_prepare_draw(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   uint8_t raw_mask = hk_feedback_color_mask(cmd, pRenderingInfo);
   const struct hk_image_view *raw_zs = hk_feedback_zs_view(cmd, pRenderingInfo);
   if (raw_mask || raw_zs) {
      hk_cmd_render_merge_flush(cmd);
      if (!hk_feedback_decompress(cmd, pRenderingInfo, raw_mask, raw_zs))
         return;
   }
   if (!cmd->feedback_decompress && (raw_mask || raw_zs))
      cmd->sync_render_continuation = false;
   bool merge = false;
   /* The resuming render pass of a suspended one continues its render. */
   bool resume = false;
   if (cmd->merge_cs && cmd->merge_suspended) {
      resume = (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) &&
               !cmd->in_meta;
      if (!resume)
         hk_cmd_render_merge_flush(cmd);
   }
   if (cmd->merge_cs && !resume) {
      merge = hk_render_merge_begin(cmd, pRenderingInfo);
      if (merge)
         hk_cmd_empty_barriers_flush(cmd, true);
      if (!merge) {
         hk_render_merge_trace(cmd->in_meta ? "begin: meta" : "begin: mismatch",
                               pRenderingInfo->colorAttachmentCount,
                               pRenderingInfo->colorAttachmentCount ?
                                  pRenderingInfo->pColorAttachments[0].loadOp : ~0ull);
         hk_cmd_render_merge_flush_why(cmd, NULL);
      }
   }

   /* A continued render needs no dependency on itself. */
   cmd->sync_render_continuation =
      (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) != 0 && !resume && !raw_mask &&
      !raw_zs;
   cmd->sync_render_stages = cmd->sync_scope ?: HK_SYNC_GRAPHICS;
   cmd->state.gfx.render_gen++;
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));
   cmd->rendering = true;
   render->raw_color_mask = raw_mask;
   render->feedback = cmd->vk.dynamic_graphics_state.feedback_loops != 0;

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   for (uint32_t i = 0; i < MIN2(pRenderingInfo->colorAttachmentCount, 32); i++) {
      if (pRenderingInfo->pColorAttachments[i].imageView &&
          is_attachment_stored(&pRenderingInfo->pColorAttachments[i]))
         render->store_mask |= BITFIELD_BIT(i);
   }
   const VkRenderingAttachmentInfo *ds[2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (unsigned i = 0; i < 2; ++i) {
      if (ds[i] && ds[i]->imageView) {
         if (is_attachment_stored(ds[i]))
            render->ds_store_mask |= BITFIELD_BIT(i);
         if (vk_image_layout_is_read_only(ds[i]->imageLayout,
               i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
            render->ds_read_only |= BITFIELD_BIT(i);
      }
   }
   const VkRenderingAttachmentInfo *feedback_att[HK_MAX_RTS + 2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (uint32_t i = 0; i < MIN2(pRenderingInfo->colorAttachmentCount, HK_MAX_RTS); i++)
      feedback_att[2 + i] = &pRenderingInfo->pColorAttachments[i];
   for (unsigned i = 0; i < ARRAY_SIZE(feedback_att); ++i) {
      if (hk_attachment_may_be_sampled(feedback_att[i]))
         render->feedback = true;
   }
   for (unsigned i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      if (!(raw_mask & BITFIELD_BIT(i)))
         hk_feedback_invalidate_view(cmd,
            hk_image_view_from_handle(pRenderingInfo->pColorAttachments[i].imageView));
   }
   /* Compressed ZLS stores re-encode the planes' tiles. */
   for (unsigned i = 0; i < 2 && !raw_zs; i++) {
      if (ds[i] && ds[i]->imageView)
         hk_feedback_invalidate_view(cmd, hk_image_view_from_handle(ds[i]->imageView));
   }
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
      hk_attachment_use_implicit_sync(cmd, &render->color_att[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);
   hk_attachment_use_implicit_sync(cmd, &render->depth_att);
   hk_attachment_use_implicit_sync(cmd, &render->stencil_att);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   u_foreach_bit(i, render->store_mask) {
      struct hk_image_view *view = render->color_att[i].iview;
      struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
      if (image->planes[view->planes[0].image_plane].layout.tiling ==
          AIL_TILING_LINEAR)
         render->linear_store = true;
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   if (!render->cr.zls_width) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = hk_format_to_pipe_format(render->color_att[i].vk_format);
   }

   /* For now, we force layered=true since it makes compatibility problems way
    * easier.
    */
   render->tilebuffer = agx_build_tilebuffer_layout(
      formats, render->color_att_count, render->tilebuffer.nr_samples, true);

   unsigned bytes = 0;
   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      render->tilebuffer._offset_B[rt] = bytes;
      bytes += 4 * agx_apple9_color_words(formats[rt]);
      render->tilebuffer.spilled[rt] = false;
   }
   render->tilebuffer.sample_size_B = ALIGN_POT(
      agx_apple9_tile_sample_bytes(bytes, render->tilebuffer.nr_samples), 8);
   render->tilebuffer.tile_size = render->tilebuffer.nr_samples == 4
                                    ? 32 * 16 : 32 * 32;

   if (cmd->feedback_decompress) {
      /* A self-blit must retire each whole compression block before another
       * tile reads it. The metadata covers 16x16 texels for every colour bpp. */
      struct drm_asahi_cmd_render raw = {0};
      agx_tilebuffer_set_drm_cmd(&raw, &render->tilebuffer);
      assert(raw.utile_width_px % 16 == 0);
      assert(raw.utile_height_px % 16 == 0);
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area);
   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true, incomplete_render_area);

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false, incomplete_render_area);
   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true, incomplete_render_area);

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, false, NULL);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true, false,
                       NULL);

   const bool drop_stores = !(render->flags & VK_RENDERING_SUSPENDING_BIT);
   /* The first render pass of a chain decides its loads. */
   if (!resume && !merge) {
      render->chain_load[0] =
         attach_z && attach_z->imageView &&
         (attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || incomplete_render_area);
      render->chain_load[1] =
         attach_s && attach_s->imageView &&
         (attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || incomplete_render_area);
   } else {
      memcpy(render->chain_load, cmd->merge_chain_load, sizeof(render->chain_load));
   }
   hk_pack_zls_control(&render->zls_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);
   hk_pack_zls_control(&render->zls_partial_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, true, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);
   if (raw_zs)
      hk_feedback_make_zs_raw(render);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;

   if (resume) {
      /* Same render pass instance: keep the open render's background and
       * load configuration; this pass's store operations apply at its end. */
      perf_debug(cmd, "Resuming the suspended render pass in its render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      render->feedback |= cmd->merge_feedback;
      render->resumed = true;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      cmd->merge_suspended = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   if (merge) {
      /* Continue the open render. Its background program, made for the
       * first pass, loads or clears every tile it processes. */
      perf_debug(cmd, "Merging render pass into the previous render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      /* The render's earlier passes may have read their attachments too. */
      render->feedback |= cmd->merge_feedback;
      render->merge_reach = cmd->merge_reach;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;
   cs->linear_store = render->linear_store;
   render->first_cs = cs;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   /* hk_sync orders compute <-> graphics dependencies across streams via
    * explicit barriers. Only end an open compute stream here when fallback
    * sync is active or the stream carries a timestamp / HK_SYNC_ALL work
    * that this render must not delay or overtake.
    */
   if (cmd->sync_fallback ||
       (cmd->current_cs.cs &&
        (cmd->current_cs.cs->timestamp.end.handle ||
         cmd->current_cs.cs->sync_stages == HK_SYNC_ALL))) {
      hk_cs_end_hint(cmd, HK_CS_END_BEGIN_RENDER);
      hk_cmd_buffer_end_compute(cmd);
   }

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct libagx_decompress_images imgs = {
                  .compressed = view->planes[plane].emrt_texture,
                  .uncompressed = view->planes[plane].emrt_pbe,
               };

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                 layout, layer, level, base,
                                 hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear |= is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   if (clear_count > 0 && incomplete_render_area) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = resolved_clear;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                       const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs) {
      cs->cr.eot.main = (render->flags & VK_RENDERING_SUSPENDING_BIT)
                           ? render->cr.eot.partial : render->cr.eot.main;
      /* zls_last already preserves suspended stores. Its first-stream
       * loads must retain CLEAR; the partial control would force a load. */
      cs->cr.zls_control = cs == render->first_cs ? render->zls_last
                                                  : render->zls_partial_last;
   }

   perf_debug(cmd, "End rendering");
   const bool suspend = hk_render_suspend_keeps_open(cmd, cs);
   if (suspend || hk_render_merge_end(cmd, cs)) {
      cmd->merge_suspended = suspend;
      memcpy(cmd->merge_chain_load, render->chain_load,
             sizeof(cmd->merge_chain_load));
      /* Keep the render open for a following render pass to continue. */
      /* A continuing pass stores at least what the passes before it did, so
       * its mask covers the whole open render. */
      cmd->merge_store_mask = render->store_mask;
      cmd->merge_ds_store_mask = render->ds_store_mask;
      cmd->merge_feedback = render->feedback;
      cmd->merge_reach = hk_merge_graphics_stages() | render->merge_reach;
      cmd->merge_cs = cs;
      cmd->merge_first_cs = cs == render->first_cs;
      /* A render without draws yet: keep its state for the empty-render
       * optimization in case nothing continues it. */
      cmd->merge_empty = !cs->stats.cmds;
      if (cmd->merge_empty) {
         if (!cmd->merge_empty_render)
            cmd->merge_empty_render = malloc(sizeof(*render));
         if (cmd->merge_empty_render) {
            memcpy(cmd->merge_empty_render, render, sizeof(*render));
         } else {
            cmd->merge_cs = NULL;
            cmd->merge_empty = false;
            hk_cs_end_hint(cmd, HK_CS_END_RENDER_PASS);
            hk_cmd_buffer_end_graphics(cmd);
         }
      }
      if (cmd->merge_cs) {
         /* Barriers before the next pass go ahead of the stream's first event. */
         cmd->merge_event = *util_dynarray_element(&cmd->sync_first_event, uint32_t,
                                                   cs->sync_id);
         assert(cmd->merge_event != HK_SYNC_NONE &&
                "a stream with draws has a sync event");
      }
   } else {
      if (cs) {
         /* Why this render is not kept open (diagnostic). */
         bool resolve = false;
         for (unsigned i = 0; i < render->color_att_count; ++i)
            resolve |= render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE;
         hk_render_merge_trace(
            "end: ineligible",
            (render->depth_att.iview || render->stencil_att.iview) |
               resolve << 1 | cmd->sync_fallback << 2 |
               (cmd->current_cs.pre_gfx || cmd->current_cs.post_gfx) << 3 |
               !!cmd->tess_batch.draws << 4 |
               !!cmd->state.gfx.occlusion.active_count << 5 | cmd->in_meta << 6 |
               (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) << 7 |
               !cs->stats.cmds << 8 | !render->color_att_count << 9,
            render->flags);
      }
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_PASS);
      hk_cmd_buffer_end_graphics(cmd);
   }

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   VkRenderingAttachmentFlagsInfoKHR vk_color_flags[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         need_resolve = true;

      vk_color_flags[i] = (VkRenderingAttachmentFlagsInfoKHR){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
         .flags = render->color_att[i].flags,
      };
      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .pNext = &vk_color_flags[i],
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentFlagsInfoKHR vk_depth_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->depth_att.flags,
   };
   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_depth_flags,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentFlagsInfoKHR vk_stencil_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->stencil_att.flags,
   };
   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_stencil_flags,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));
   cmd->rendering = false;

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}
