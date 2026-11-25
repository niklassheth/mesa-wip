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

DEBUG_GET_ONCE_BOOL_OPTION(mesh_route_log, "HK_APPLE9_MESH_ROUTE_LOG", false)

/* The geometry heap's map in the read-only data. */
static struct poly_heap *
hk_heap_map(struct hk_device *dev)
{
   off_t off = dev->rodata.heap - dev->rodata.bo->va->addr;
   return agx_bo_map(dev->rodata.bo) + off;
}

/* Back the geometry heap's VA range with a new BO of size_B bytes from its
 * start, releasing the old backing. The heap must be idle.
 * Returns VK_SUCCESS on replacement, VK_ERROR_OUT_OF_DEVICE_MEMORY if the old
 * backing remains intact, or VK_ERROR_DEVICE_LOST if unbind succeeded but
 * neither new nor old backing could be bound. */
static VkResult
hk_heap_back(struct hk_device *dev, uint64_t size_B)
{
   struct agx_bo *bo =
      agx_bo_create(&dev->dev, size_B, 0, 0, "Geometry heap");
   if (!bo)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (dev->heap) {
      if (agx_bo_bind(&dev->dev, NULL, dev->heap_va->addr, dev->heap->size, 0,
                      DRM_ASAHI_BIND_UNBIND)) {
         agx_bo_unreference(&dev->dev, bo);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
   }

   if (agx_bo_bind(&dev->dev, bo, dev->heap_va->addr, size_B, 0,
                   DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)) {
      agx_bo_unreference(&dev->dev, bo);
      /* Keep the old backing mapped, if any. */
      if (dev->heap) {
         if (agx_bo_bind(&dev->dev, dev->heap, dev->heap_va->addr,
                         dev->heap->size, 0,
                         DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)) {
            agx_bo_unreference(&dev->dev, dev->heap);
            dev->heap = NULL;
            return vk_errorf(dev, VK_ERROR_DEVICE_LOST,
                             "Failed to restore geometry heap mapping");
         }
      }
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   agx_bo_unreference(&dev->dev, dev->heap);
   dev->heap = bo;
   hk_heap_map(dev)->size = size_B - POLY_HEAP_SINK_B;
   return VK_SUCCESS;
}

static void
hk_init_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   perf_debug(cmd, "Allocating heap");

   /* Unrolled and geometry-generated index buffers are allocated from this
    * heap. Apple9 draws address index buffers through the USC aperture.
    * Recorded draws address the heap by its VA range, so the range is fixed
    * and its backing grows instead, see hk_heap_prepare_submit. The USC
    * aperture is shared, so settle for a smaller range when it is crowded.
    */
   for (uint64_t va_B = HK_HEAP_VA_MAX_B;
        !dev->heap_va && va_B >= HK_HEAP_GEOMETRY_MIN_B; va_B /= 2) {
      dev->heap_va = agx_va_alloc(&dev->dev, va_B, 65536, AGX_VA_USC, 0);
   }
   if (!dev->heap_va)
      return;

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   *hk_heap_map(dev) = (struct poly_heap){
      .base = dev->heap_va->addr,
   };

   if (hk_heap_back(dev, MIN2(cmd->heap_min_B, dev->heap_va->size_B)) !=
       VK_SUCCESS) {
      agx_va_free(&dev->dev, dev->heap_va, false);
      dev->heap_va = NULL;
   }
}

/* Whether every queue has finished its submissions, without waiting. */
static bool
hk_device_idle(struct hk_device *dev)
{
   vk_foreach_queue(iter, &dev->vk) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      uint64_t completed = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &completed, 1) ||
          completed < queue->drm.timeline_value)
         return false;
   }
   return true;
}

/* Return a spike's memory. high_water only grows, so the heap's need is kept
 * per period of HK_HEAP_DECAY_NS instead: when one has passed and the device is
 * already idle (never wait for it), start a new period from a cleared
 * high_water, and shrink the backing if it is more than twice what this
 * submission and the period needed (at least HK_HEAP_MIN_B).
 */
static VkResult
hk_heap_decay(struct hk_device *dev, uint64_t min_B, uint64_t needed_B)
{
   dev->heap_period_need_B = MAX2(dev->heap_period_need_B, needed_B);

   uint64_t now = os_time_get_nano();
   if (now - dev->heap_period_start_ns < HK_HEAP_DECAY_NS ||
       !hk_device_idle(dev))
      return VK_SUCCESS;

   uint64_t target_B =
      MAX3(min_B, HK_HEAP_MIN_B,
           util_next_power_of_two64(dev->heap_period_need_B + POLY_HEAP_SINK_B));
   if (dev->heap->size > 2 * target_B) {
      perf_debug_dev(&dev->dev, "Shrinking the geometry heap to %" PRIu64
                     " MiB", target_B >> 20);
      VkResult res = hk_heap_back(dev, target_B);
      if (res == VK_ERROR_DEVICE_LOST)
         return res;
      if (res == VK_SUCCESS)
         dev->heap_reported_B = 0;
   }

   /* The heap is idle, so nothing races these writes. */
   struct poly_heap *heap = hk_heap_map(dev);
   heap->high_water = 0;
   heap->bottom = 0;
   dev->heap_period_need_B = 0;
   dev->heap_period_start_ns = now;
   return VK_SUCCESS;
}

/* Before a submission whose command buffers use the geometry heap, with the
 * heap lock held: back at least min_B bytes, and what earlier submissions
 * needed. Allocations that did not fit dropped their draws, and the heap grows
 * past them at once, to the next power of two. Waits for the device to idle.
 */
VkResult
hk_heap_prepare_submit(struct hk_device *dev, uint64_t min_B)
{
   if (!dev->heap)
      return VK_SUCCESS;

   const struct poly_heap *heap = hk_heap_map(dev);
   uint64_t needed_B = poly_heap_needed_B(heap);
   if (needed_B > heap->size && needed_B > dev->heap_reported_B) {
      dev->heap_reported_B = needed_B;
      dev->heap_overflows++;
      mesa_logw_once("hk: geometry heap exhausted (needed %" PRIu64
                     " MiB, had %u MiB): geometry or tessellation output was "
                     "dropped",
                     needed_B >> 20, heap->size >> 20);
   }

   uint64_t target_B =
      MAX2(min_B, needed_B > heap->size
                     ? util_next_power_of_two64(needed_B + POLY_HEAP_SINK_B)
                     : 0);
   target_B = MIN2(target_B, dev->heap_va->size_B);
   if (target_B <= dev->heap->size)
      return hk_heap_decay(dev, min_B, needed_B);

   /* Earlier submissions may still use the heap. */
   vk_foreach_queue(iter, &dev->vk) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      int err;
      do {
         err = drmSyncobjTimelineWait(dev->dev.fd, &queue->drm.syncobj,
                                      &queue->drm.timeline_value, 1, INT64_MAX,
                                      DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                      NULL);
      } while (err < 0 && errno == EINTR);
      if (err)
         return vk_errorf(dev, VK_ERROR_DEVICE_LOST,
                          "Geometry heap wait failed: %m");
   }

   perf_debug_dev(&dev->dev, "Growing the geometry heap to %" PRIu64 " MiB",
                  target_B >> 20);
   VkResult res = hk_heap_back(dev, target_B);
   if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY && dev->heap->size >= min_B)
      return VK_SUCCESS;
   return res;
}

uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Geometry and tessellation need the heap's historical size; the other
    * users (unrolled fans, adjacency, robust index copies) start small, see
    * hk_heap_prepare_submit. */
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint64_t min_B =
      gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TESS_EVAL]
         ? HK_HEAP_GEOMETRY_MIN_B
         : HK_HEAP_MIN_B;
   cmd->heap_min_B = MAX2(cmd->heap_min_B, min_B);

   util_call_once_data(&dev->heap_init_once, hk_init_heap, cmd);
   if (!dev->heap) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   /* Acquire the heap before this render's first allocation. The pre-graphics
    * stream waits for earlier work, including the previous heap consumer.
    * Resetting here also covers command-buffer resubmission without a separate
    * post-graphics Work just to release the allocation cursor.
    */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Resetting heap");
      uint64_t addr = dev->rodata.heap;

      cmd->heap_renders++;
      cmd->uses_heap = true;

      libagx_heap_reset(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, addr);
   }

   return dev->rodata.heap;
}

/* GPU-generated index/coordinate offsets use the single geometry heap, by its
 * fixed VA range: its backing may grow before a later submission. */
static struct agx_va *
hk_geometry_heap_va(struct hk_cmd_buffer *cmd)
{
   /* Recording can continue after OOM, but the command buffer cannot be
    * submitted. Keep address construction safe on those error paths. */
   if (!hk_heap(cmd))
      return NULL;
   return hk_cmd_buffer_device(cmd)->heap_va;
}

static struct agx_draw
hk_heap_indexed_indirect(struct hk_cmd_buffer *cmd, uint64_t ptr,
                         enum agx_index_size index_size, bool restart)
{
   struct agx_va *va = hk_geometry_heap_va(cmd);
   return agx_draw_indexed_indirect(ptr, va ? va->addr : 0,
                                   va ? va->size_B : 0, index_size, restart);
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params_for_vs(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                                const struct hk_shader *vs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_buffer_range_el = range_el;

      /* Apple9 software vertex shaders read the index size from here rather
       * than from a specialized prolog.
       */
      params.index_size_B = index_size_B;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   return hk_upload_vertex_params_for_vs(cmd, draw,
                                         hk_bound_sw_vs(&cmd->state.gfx));
}

enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

/* Apple9 draws address index buffers through the 32-bit USC aperture. */
static struct agx_ptr
hk_geom_index_alloc(struct hk_cmd_buffer *cmd, uint32_t size)
{
   return hk_pool_usc_alloc(cmd, size, 4);
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs =
      hk_apple9_select_gs(cmd, gfx->shaders[MESA_SHADER_GEOMETRY]);
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs ? fs->info.fs.interp.flat : 0;
   /* Tessellation precedes the geometry shader and has set up this draw's
    * parameters; the geometry shader recovers patch IDs from them. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL])
      params.tess_params = gfx->descriptors.root.draw.tess_params;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }
      params.pg_query_counter[i] = gfx->pg_query[i];

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      /* On failure the command buffer is in error and never submitted. */
      if (T.cpu)
         memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         struct agx_va *va = hk_geometry_heap_va(cmd);
         cmd->geom_index_buffer = va ? va->addr : 0;
         cmd->geom_index_count = va ? va->size_B : 0;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_geom_index_alloc(cmd, cmd->geom_index_count * 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      struct agx_ptr topology = hk_geom_index_alloc(cmd, gsi->max_indices * 4);
      if (topology.cpu)
         memcpy(topology.cpu, gsi->topology,
                MIN2(gsi->max_indices * 4, sizeof(gsi->topology)));
      cmd->geom_index_buffer = topology.gpu;
   }

   desc->root_dirty = true;
   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static void
hk_upload_tess_params_for_tcs(struct hk_cmd_buffer *cmd,
                              struct poly_tess_params *out,
                              struct agx_draw draw, struct hk_shader *tcs)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params args = {
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!args.points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      args.ccw = gfx->tess.info.ccw;
      args.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   /* heap is allocated by hk_heap */
   struct agx_va *va = hk_geometry_heap_va(cmd);
   args.patch_coord_buffer = va ? va->addr : 0;

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / args.input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * args.tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      args.tcs_buffer = blob.gpu + tcs_out_offs;
      args.patches_per_instance = in_patches;
      args.coord_allocs = blob.gpu + patch_coord_offs;
      args.nr_patches = unrolled_patches;
      args.out_draws = blob.gpu + draw_offs;
      args.counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      args.out_draws = hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = args.out_draws;
   memcpy(out, &args, sizeof(args));
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   hk_upload_tess_params_for_tcs(
      cmd, out, draw,
      hk_only_variant(cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL]));
}


static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = gfx->index.restart,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, draw.index_size,
                                   false /* restart */);
}

/* Whether a vertex-shader pipeline exposes its provoking vertex: through a
 * flat, per-vertex or barycentric fragment input, or a layer or viewport index
 * written by the vertex shader. Primitive ID is not taken from a vertex, and a
 * fragment shader's layer reads zero unless the vertex shader writes it. A
 * vertex-written primitive shading rate would observe it too, once supported. */
static bool
hk_apple9_provoking_observed(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   if (vs && hk_any_variant(vs)->b.info.writes_layer_viewport)
      return true;
   if (!fs || fs == dev->null_fs)
      return false;

   const struct agx_shader_info *info = &hk_any_variant(fs)->b.info;
   return info->apple9_flat_mask.lo || info->apple9_flat_mask.hi ||
          info->apple9_barycentric_mask || info->apple9_per_vertex_count;
}

/* Native Apple9 fans flat shade consistently only from their first vertex.
 * Draw fans with an observable last provoking vertex as an unrolled triangle
 * list. */
static struct agx_draw
hk_apple9_unroll_fan(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   perf_debug(cmd, "Unrolling a last-provoking triangle fan");

   uint64_t in_draw;
   if (draw.indexed) {
      draw = hk_draw_as_indexed_indirect(cmd, draw);
      in_draw = draw.b.ptr;
   } else if (agx_is_indirect(draw.b)) {
      in_draw = draw.b.ptr;
   } else {
      VkDrawIndirectCommand desc = {
         .vertexCount = draw.b.count[0],
         .instanceCount = draw.b.count[1],
         .firstVertex = draw.start,
         .firstInstance = draw.start_instance,
      };
      in_draw = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct libagx_unroll_geometry_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.indexed ? draw.index_buffer : 0,
      .in_draw = in_draw,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu,
      .restart_index = draw.restart ? gfx->index.restart : UINT32_MAX,
      .index_buffer_size_el = draw.indexed ? agx_draw_index_range_el(draw) : 0,
      .in_index_size_B = draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
      .flatshade_first = false,
   };

   libagx_unroll_geometry_struct(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                                 ia, poly_compact_prim(MESA_PRIM_TRIANGLE_FAN));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, AGX_INDEX_SIZE_U32,
                                   false /* restart */);
}

static void
hk_apple9_dispatch_prerast(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                           struct agx_grid grid,
                           const struct hk_apple9_prerast_bindings *bindings)
{
   if (agx_is_shader_empty(&s->b))
      return;
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_ALL);
   if (!cs)
      return;

   if (s->b.info.apple9_vs_prolog.uses_prolog) {
      hk_apple9_dispatch_sw_vs_prolog(cmd, cs, s, grid, bindings);
      return;
   }

   hk_apple9_dispatch(cmd, cs, s->apple9_body, &s->info.apple9, grid,
                      cmd->state.gfx.root, s, bindings);
}

/* Only geometry shaders observe adjacent vertices. Without one, draw the
 * primitive list of the remaining vertices through a generated index buffer.
 */
static struct agx_draw
hk_apple9_draw_without_adj(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      prim = u_decomposed_prim(prim);
   }

   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1]};
      if (draw.indexed) {
         desc[2] = draw.start;
         desc[3] = draw.index_bias;
         desc[4] = draw.start_instance;
      } else {
         desc[2] = draw.start;
         desc[3] = draw.start_instance;
      }
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }

   uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
   uint64_t heap = hk_heap(cmd);
   libagx_unroll_adjacency(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, heap, in_draw,
      draw.indexed ? draw.index_buffer : 0,
      draw.indexed ? draw.index_buffer_range_B >> draw.index_size : 0,
      draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim,
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);

   return hk_heap_indexed_indirect(cmd, out_draw, AGX_INDEX_SIZE_U32, false);
}

/* The Apple9 index fetch reads only from the USC aperture and is bounded in
 * whole words. Fetch from a copy for index buffers placed outside the
 * aperture, and when robustness requires zeroes past a binding that ends
 * inside a word.
 */
static bool
hk_apple9_needs_index_copy(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw.indexed || draw.index_buffer_range_B == 0)
      return false;

   /* Check the whole binding, not just its base: a direct draw's firstIndex
    * (or an indirect draw's GPU-provided firstIndex) may otherwise move the
    * hardware pointer past the end of the USC aperture. Subtraction avoids
    * wrapping the end of a high-VA binding.
    */
   if (draw.index_buffer < dev->dev.shader_base ||
       draw.index_buffer - dev->dev.shader_base >= (1ull << 32) ||
       draw.index_buffer_range_B >
          (1ull << 32) - (draw.index_buffer - dev->dev.shader_base))
      return true;

   if (!((draw.index_buffer + draw.index_buffer_range_B) & 3))
      return false;

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   return agx_is_indirect(draw.b) || agx_direct_draw_overreads_indices(draw);
}

static struct agx_draw
hk_apple9_robust_index(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1], draw.start,
                          draw.index_bias, draw.start_instance};
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }

   perf_debug(cmd, "copying robust index buffer");
   uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
   libagx_robust_index_apple9(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, hk_heap(cmd),
      in_draw, draw.index_buffer,
      draw.index_buffer_range_B >> draw.index_size,
      agx_index_size_to_B(draw.index_size));

   return hk_heap_indexed_indirect(cmd, out_draw, draw.index_size,
                                   draw.restart);
}

static void
hk_apple9_merge_tess_info(struct hk_cmd_buffer *cmd, bool native)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   struct hk_api_shader *tes_api = gfx->shaders[MESA_SHADER_TESS_EVAL];
   if (native) {
      tcs_api = hk_api_shader_native(tcs_api);
      tes_api = hk_api_shader_native(tes_api);
   }
   struct hk_shader *tcs = hk_only_variant(tcs_api);
   struct hk_tess_info tes_info = native
                                     ? tes_api->apple9_native_tess
                                     : hk_any_variant(tes_api)->info.tess.info;
   gfx->tess.info = hk_tess_info_merge(tes_info, tcs->info.tess.info);
   if (gfx->tess.info.points)
      gfx->tess.prim = MESA_PRIM_POINTS;
   else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES)
      gfx->tess.prim = MESA_PRIM_LINES;
   else
      gfx->tess.prim = MESA_PRIM_TRIANGLES;
}

/* Run the vertex and tessellation control shaders and the tessellator as
 * compute before the render pass, returning the draw of the tessellated
 * domain.
 */
static struct agx_draw
hk_apple9_launch_tess(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                      bool native)
{
   /* Largest direct draw, in patches x instances, that is batched. */
   const unsigned max_batch_patches = 2048;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   struct hk_api_shader *vs_api = gfx->shaders[MESA_SHADER_VERTEX];
   if (native) {
      tcs_api = hk_api_shader_native(tcs_api);
      vs_api = hk_api_shader_native(vs_api);
   }
   struct hk_shader *vs = &vs_api->variants[HK_VS_VARIANT_SW];
   struct hk_shader *tcs = hk_only_variant(tcs_api);

   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);

   desc->root.draw.vertex_params =
      native ? hk_upload_vertex_params_for_vs(cmd, draw, vs)
             : hk_upload_vertex_params(cmd, draw);
   struct agx_ptr tess_args =
      hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
   if (!tess_args.cpu)
      return draw;
   desc->root.draw.tess_params = tess_args.gpu;
   hk_upload_tess_params_for_tcs(cmd, tess_args.cpu, draw, tcs);
   desc->root.draw.clip_z_coeff =
      dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = desc->root.draw.tess_params;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   if (agx_is_indirect(draw.b)) {
      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;
      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      uint64_t tcs_stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);
      if (hk_stat_enabled(tcs_stat)) {
         libagx_increment_statistic(cmd, agx_1d(1),
                                    AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat,
                                    patches);
      }
   }

   /* Without a geometry shader, the tessellated primitives are the clipper
    * input.
    */
   uint64_t c_prims = 0, c_inv = 0;
   if (!gfx->shaders[MESA_SHADER_GEOMETRY]) {
      c_prims = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
      c_inv = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
   }

   /* A direct draw without statistics or transform feedback touches only
    * its own allocations here, so its passes can run alongside those of the
    * neighbouring tessellated draws: record them into the stage lanes (see
    * hk_tess_batch_flush). The stage order within each draw is kept by the
    * barrier after every lane.
    *
    * Only small draws are latency-bound enough to gain from overlapping;
    * a large one fills the GPU on its own and is kept inline, which also
    * keeps its passes out of the stream-end lanes.
    */
   unsigned batch_patches =
      agx_is_indirect(draw.b) ? 0
                              : (draw.b.count[0] / input_patch_size) *
                                   draw.b.count[1];
   /* The native object reads this draw's count. Keep all five producer passes
    * inline in pre_gfx, after flushing older lanes, so no later batch flush can
    * move the count write behind its consumer. Each pass has HK_SYNC_ALL.
    * Batched native renders record them into the lanes too:
    * hk_cmd_buffer_end_graphics flushes the lanes into the ending render's
    * own pre_gfx stream, so they still complete before its tiling. */
   bool batch =
      (!native || hk_apple9_native_lanes(hk_cmd_buffer_device(cmd))) &&
      !agx_is_indirect(draw.b) &&
      batch_patches <= max_batch_patches &&
      !hk_stat_enabled(hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT)) &&
      !hk_stat_enabled(c_prims) && !hk_stat_enabled(c_inv) && !gfx->xfb_enabled;
   /* A full fold table starts a new batch. */
   bool fold = batch && hk_tess_fold_enabled();
   if (!batch ||
       (fold && util_dynarray_num_elements(&cmd->tess_batch.fold,
                                           struct hk_tess_fold) >=
                   HK_TESS_FOLD_MAX_DRAWS))
      hk_tess_batch_flush(cmd);

#define HK_TESS_LANE(l) cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[l] : NULL

   HK_TESS_LANE(0);
   hk_apple9_dispatch_prerast(cmd, vs, grid_vs, &bindings);
   HK_TESS_LANE(1);
   hk_apple9_dispatch_prerast(cmd, tcs, grid_tcs, &bindings);

   if (fold) {
      /* The flush runs the count, prefix sum and emit passes of every folded
       * draw (hk_tess_fold_flush).
       */
      struct hk_tess_fold f = {
         .params = state,
         .nr_patches = ((struct poly_tess_params *)tess_args.cpu)->nr_patches,
         .prim = gfx->tess.info.mode,
      };
      util_dynarray_append(&cmd->tess_batch.fold, f);
      cmd->tess_batch.capture = NULL;
      cmd->tess_batch.draws++;
      return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws,
                                      AGX_INDEX_SIZE_U32, false);
   }

   /* First generate counts, then prefix sum them, and then tessellate. */
   HK_TESS_LANE(2);
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     gfx->tess.info.mode, POLY_TESS_MODE_COUNT, state);
   HK_TESS_LANE(3);
   /* One workgroup walking every patch takes milliseconds for large draws,
    * so direct draws past a few blocks scan blocks in parallel instead.
    */
   const unsigned block_patches = 8192;
   unsigned nr_patches =
      agx_is_indirect(draw.b) ? 0
                              : ((struct poly_tess_params *)tess_args.cpu)->nr_patches;
   /* The three passes depend on each other, so they cannot share a stage
    * lane with other draws' passes: batched draws keep the single scan.
    */
   if (!batch && nr_patches > 2 * block_patches) {
      uint64_t counts = ((struct poly_tess_params *)tess_args.cpu)->counts;
      unsigned nr_blocks = DIV_ROUND_UP(nr_patches, block_patches);
      uint64_t block_sums = hk_pool_alloc(cmd, nr_blocks * 4, 4).gpu;
      libagx_prefix_sum_tess_blocks(cmd, agx_1d(1024 * nr_blocks),
                                    AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                    block_sums, nr_patches, block_patches);
      libagx_prefix_sum_tess_offsets(cmd, agx_1d(1024),
                                     AGX_BARRIER_ALL | AGX_PREGFX, state,
                                     block_sums, nr_blocks, c_prims, c_inv,
                                     c_prims || c_inv);
      libagx_prefix_sum_tess_add(cmd, agx_1d(nr_patches - block_patches),
                                 AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                 block_sums, nr_patches, block_patches);
   } else {
      libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                             state, c_prims, c_inv, c_prims || c_inv);
   }
   HK_TESS_LANE(4);
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     gfx->tess.info.mode, POLY_TESS_MODE_WITH_COUNTS, state);
   cmd->tess_batch.capture = NULL;
#undef HK_TESS_LANE
   if (batch)
      cmd->tess_batch.draws++;

   return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws,
                                   AGX_INDEX_SIZE_U32, false);
}

/* Run the vertex and geometry shaders as compute before the render pass,
 * returning the draw that rasterizes the geometry shader output.
 */
static struct agx_draw
hk_apple9_launch_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs =
      hk_apple9_select_gs(cmd, gfx->shaders[MESA_SHADER_GEOMETRY]);
   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   /* The software vertex shader fetches attributes relative to the API draw,
    * so bind its parameters before any restart unrolling rewrites the draw.
    */
   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);

   desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
   desc->root.draw.geometry_params = hk_upload_geometry_params(cmd, draw);
   desc->root.draw.api_gs = !gs->is_passthrough;
   desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
   desc->root.draw.provoking =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2
                                                                           : 0;
   desc->root.draw.clip_z_coeff =
      dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   enum mesa_prim mode = hk_gs_in_prim(cmd);
   struct agx_grid grid_vs, grid_gs;

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = vertex_params,
         .p = geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));
      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   hk_cpu_timing_note_gs(count->info.gs.shape, agx_is_indirect(draw.b),
                         (uint64_t)grid_gs.count[0] * grid_gs.count[1],
                         count->info.gs.max_indices);

   /* Transform feedback and geometry statistics need the count and pre-GS
    * programs.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
   bool xfb_or_queries =
      main->info.gs.xfb || (desc->root.draw.pipeline_stats_flags & gs_queries);
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
      xfb_or_queries |= gfx->pg_query[i] || gfx->xfb_query[i];

   /* Without transform feedback or queries, a small direct draw is just the
    * VS and GS passes over its own allocations: record them into the first
    * two batch lanes (as for tessellation) so consecutive draws overlap.
    */
   const unsigned max_batch_prims = 2048;
   bool batch = !agx_is_indirect(draw.b) && !xfb_or_queries &&
                grid_gs.count[0] * grid_gs.count[1] <= max_batch_prims;
   if (!batch)
      hk_tess_batch_flush(cmd);

   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[0] : NULL;
   hk_apple9_dispatch_prerast(cmd, vs, grid_vs, &bindings);
   cmd->tess_batch.capture = NULL;

   if (xfb_or_queries) {
      if (count->info.gs.count_words)
         hk_apple9_dispatch_prerast(cmd, count, grid_gs, &bindings);

      if (count->info.gs.prefix_sum) {
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count->info.gs.count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      hk_apple9_dispatch_prerast(cmd, pre_gs, agx_1d(1), &bindings);
   }

   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[1] : NULL;
   hk_apple9_dispatch_prerast(cmd, main, grid_gs, &bindings);
   cmd->tess_batch.capture = NULL;
   if (batch)
      cmd->tess_batch.draws++;

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else if (agx_is_indirect(draw.b)) {
      return agx_draw_indirect(geometry_params +
                               offsetof(struct poly_geometry_params, draw));
   } else {
      return (struct agx_draw){
         .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
      };
   }
}

void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   uint64_t data = hk_shader_constant_data(shader);
   if (data || (desc->root_constant_stages & BITFIELD_BIT(stage))) {
      desc->root.constant_data[stage] = data;
      desc->root_constant_stages &= ~BITFIELD_BIT(stage);
      if (data)
         desc->root_constant_stages |= BITFIELD_BIT(stage);
      desc->root_dirty = true;
   }
   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT) {
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

/* Direct draws through software vertex shading size the vertex output buffer
 * on the CPU (hk_upload_vertex_params). Its 32-bit size wraps for very large
 * draws, e.g. a vertex count of UINT32_MAX, and the vertex shader would then
 * write past a small allocation. Such a buffer cannot be allocated anyway, so
 * record the out-of-memory error and skip the draw.
 */
static bool
hk_sw_vs_outputs_fit(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (agx_is_indirect(draw.b) || (!gfx->shaders[MESA_SHADER_GEOMETRY] &&
                                   !gfx->shaders[MESA_SHADER_TESS_EVAL]))
      return true;

   uint64_t vertex_B = util_bitcount64(hk_bound_sw_vs(gfx)->b.info.outputs) * 16;
   uint64_t vertices = (uint64_t)draw.b.count[0] * draw.b.count[1];

   if (!vertex_B || vertices <= UINT32_MAX / vertex_B)
      return true;

   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return false;
}


VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

/* Queue the input assembly counts of direct draws recorded since the last
 * flush. A dispatch per draw serializes the GPU on statistics queries that
 * span many draws, so they are summed when recorded instead, and queued when
 * the counters change and when the query ends.
 */
void
hk_flush_ia_stats(struct hk_cmd_buffer *cmd)
{
   struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
   const uint32_t counts[5] = {s->vertices, s->prims, s->vertices, s->clipped,
                               s->clipped};
   for (unsigned i = 0; i < ARRAY_SIZE(counts); ++i) {
      if (counts[i] && s->addr[i] && hk_stat_enabled(s->addr[i])) {
         struct libagx_imm_write add = {.address = s->addr[i],
                                        .value = counts[i]};
         util_dynarray_append(&cmd->query_adds, add);
      }
   }
   memset(s, 0, sizeof(*s));
}

static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw,
             uint64_t ia_vertices, uint64_t ia_prims, uint64_t vs_invocations,
             uint64_t c_prims, uint64_t c_inv)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
   uint32_t patch_size = dyn->ts.patch_control_points;

   if (!agx_is_indirect(draw.b) && !draw.restart) {
      struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
      const uint64_t addr[5] = {ia_vertices, ia_prims, vs_invocations,
                                c_prims, c_inv};
      if (memcmp(s->addr, addr, sizeof(addr))) {
         hk_flush_ia_stats(cmd);
         memcpy(s->addr, addr, sizeof(addr));
      }
      uint32_t prims = poly_decomposed_prims_for_vertices_with_tess(
                          prim, draw.b.count[0], patch_size) *
                       draw.b.count[1];
      s->vertices += draw.b.count[0] * draw.b.count[1];
      s->prims += prims;
      /* Clipper counters are written with the geometry or tessellation
       * output when those stages are used. */
      if (!geom && !tess)
         s->clipped += prims;
      return;
   }

   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      /* Direct draws pass their own indices (agx_draw_index_buffer) with a
       * range relative to the first index, so the descriptor's first index
       * (VkDrawIndexedIndirectCommand word 2) is 0. */
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used.
    */
   if (geom || tess) {
      c_prims = 0;
      c_inv = 0;
   }

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);

      libagx_increment_ia_restart(
         cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices, ia_prims,
         vs_invocations, c_prims, c_inv, draw_ptr, agx_draw_index_buffer(draw),
         agx_draw_index_range_el(draw), cmd->state.gfx.index.restart,
         index_size_B, prim, patch_size);
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

/* Late depth/stencil tests for a feedback loop, enabled dynamically or by
 * the fragment shader's pipeline, only when the draw can write depth or
 * stencil (hk_apple9_zs_writes). Zink reports a loop whenever a depth texture
 * stays bound while it is the attachment; late export costs the early
 * rejection of hidden fragments. */
bool
hk_apple9_draw_late_zs(const struct vk_dynamic_graphics_state *dyn,
                       const struct hk_rendering_state *render,
                       const struct hk_api_shader *fs)
{
   const bool dynamic = dyn->feedback_loops &
      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   const bool created = fs && fs->zs_self_dep;
   if (!dynamic && !created)
      return false;
   return hk_apple9_zs_writes(&dyn->ds,
      (render->depth_att.iview ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
      (render->stencil_att.iview ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
}

static struct agx_draw hk_apple9_predicate_draw(struct hk_cmd_buffer *cmd,
                                                struct agx_draw draw);

/* Input triangles per instance of a direct triangle list, strip or fan. */
static uint32_t
hk_apple9_native_gs_triangles(const struct vk_dynamic_graphics_state *dyn,
                              struct agx_draw draw)
{
   if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      return draw.b.count[0] / 3;
   return draw.b.count[0] >= 3 ? draw.b.count[0] - 2 : 0;
}

/* Direct triangle lists, strips and fans, indexed or not: one mesh group per
 * input triangle and instance, within one grid dimension each. Strips and
 * fans map corners as the fallback's software input assembly does
 * (hk_nir_gs_mesh), from the topology in the draw's geometry parameters. The software VS runs once per
 * index position (applying the index buffer, firstIndex, vertexOffset and
 * firstInstance as for the fallback GS), so a list's triangle x reads VS
 * outputs 3x..3x+2 either way. Groups publish in index order, so API
 * primitive order holds. Multiple GS invocations, primitive restart and
 * other topologies need their own input mapping.
 *
 * Native GS/TES draws are direct and exclude conditional rendering and
 * queries, so no GPU pass writes a native block's words, launch records or
 * argument words: they are CPU-written before SUBMIT. The kernel may rely on
 * this to decode native renders without waiting for their producers. Keep it
 * true (or tell the kernel) before admitting indirect, predicated or queried
 * native draws. */
static bool
hk_apple9_native_layers_samples_eligible(const struct hk_graphics_state *gfx,
                                         const struct vk_dynamic_graphics_state *dyn,
                                         bool allow_multilayer_msaa)
{
   unsigned samples =
      MAX2(gfx->render.tilebuffer.nr_samples ?: dyn->ms.rasterization_samples, 1);
   uint32_t layers = gfx->render.view_mask
                        ? util_last_bit(gfx->render.view_mask)
                        : gfx->render.layer_count;
   if (!allow_multilayer_msaa)
      return layers == 1 && samples == 1;
   return layers >= 1 && layers <= 2048 &&
          (samples == 1 || samples == 2 || samples == 4);
}

static uint32_t
hk_apple9_native_gs_fused_input_primitives(
   const struct vk_dynamic_graphics_state *dyn, struct agx_draw draw,
   enum mesa_prim expected_input)
{
   switch (expected_input) {
   case MESA_PRIM_POINTS:
      return dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                ? draw.b.count[0]
                : 0;
   case MESA_PRIM_LINES:
      if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
         return draw.b.count[0] / 2;
      if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
         return draw.b.count[0] >= 2 ? draw.b.count[0] - 1 : 0;
      return 0;
   case MESA_PRIM_TRIANGLES:
      if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
         return draw.b.count[0] / 3;
      if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
          dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)
         return draw.b.count[0] >= 3 ? draw.b.count[0] - 2 : 0;
      return 0;
   default:
      return 0;
   }
}

static bool
hk_apple9_native_gs_fused_grid(uint64_t total_groups, uint32_t instances,
                               uint32_t out_grid[3])
{
   if (!total_groups || !instances ||
       instances > HK_APPLE9_NATIVE_MAX_GROUP_COUNT ||
       total_groups > HK_APPLE9_NATIVE_MAX_GROUP_TOTAL)
      return false;
   if (total_groups <= HK_APPLE9_NATIVE_MAX_GROUP_COUNT) {
      out_grid[0] = (uint32_t)total_groups;
      out_grid[1] = 1;
      out_grid[2] = instances;
   } else {
      uint64_t gy = DIV_ROUND_UP(total_groups, HK_APPLE9_NATIVE_MAX_GROUP_COUNT);
      uint64_t gx = DIV_ROUND_UP(total_groups, gy);
      if (gx > HK_APPLE9_NATIVE_MAX_GROUP_COUNT ||
          gy > HK_APPLE9_NATIVE_MAX_GROUP_COUNT)
         return false;
      out_grid[0] = (uint32_t)gx;
      out_grid[1] = (uint32_t)gy;
      out_grid[2] = instances;
   }
   return hk_apple9_native_mesh_grid_valid(out_grid);
}

static bool
hk_apple9_native_gs_common_state_eligible(struct hk_cmd_buffer *cmd,
                                          struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_stage_available_device(hk_cmd_buffer_device(cmd),
                                                MESA_SHADER_GEOMETRY) ||
       cmd->in_meta || agx_is_indirect(draw.b) || draw.restart ||
       draw.b.count[2] != 1 || dyn->ia.primitive_restart_enable ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH] || gfx->render.view_mask ||
       !hk_apple9_native_layers_samples_eligible(
          gfx, dyn, hk_apple9_native_gs_enabled()) ||
       hk_cond_render_applies(cmd) ||
       gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.active_count || gfx->mesh_pg_query || gfx->xfb_enabled ||
       dyn->rs.rasterization_stream || dyn->rs.rasterizer_discard_enable ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->rs.conservative_mode !=
          VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i) {
      if (gfx->pg_query[i] || gfx->xfb_query[i])
         return false;
   }
   return true;
}

static bool
hk_apple9_native_gs_state_eligible(struct hk_cmd_buffer *cmd,
                                  struct agx_draw draw)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_gs_common_state_eligible(cmd, draw) ||
       draw.b.count[0] < 3 ||
       (dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
        dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
        dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) ||
       (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
        draw.b.count[0] % 3) ||
       !hk_apple9_native_mesh_grid_valid(
          (const uint32_t[]){hk_apple9_native_gs_triangles(dyn, draw), draw.b.count[1], 1}))
      return false;
   return true;
}

static bool
hk_apple9_native_tess_common_state_eligible(struct hk_cmd_buffer *cmd,
                                            struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_stage_available_device(hk_cmd_buffer_device(cmd),
                                                MESA_SHADER_TESS_EVAL) ||
       cmd->in_meta || agx_is_indirect(draw.b) || draw.restart ||
       !dyn->ts.patch_control_points || dyn->ts.patch_control_points > 32 ||
       draw.b.count[0] < dyn->ts.patch_control_points || !draw.b.count[1] ||
       draw.b.count[1] > HK_APPLE9_NATIVE_MAX_GROUP_COUNT ||
       draw.b.count[2] != 1 ||
       dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST ||
       dyn->ia.primitive_restart_enable ||
       !gfx->shaders[MESA_SHADER_VERTEX] ||
       !gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       !gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH] || gfx->render.view_mask ||
       !hk_apple9_native_layers_samples_eligible(
          gfx, dyn, hk_apple9_native_tess_enabled()) ||
       hk_cond_render_applies(cmd) ||
       gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.active_count || gfx->mesh_pg_query || gfx->xfb_enabled ||
       dyn->rs.rasterization_stream || dyn->rs.rasterizer_discard_enable ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->rs.conservative_mode !=
          VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i) {
      if (gfx->pg_query[i] || gfx->xfb_query[i])
         return false;
   }
   return true;
}

/* Stage 2 admits one direct patch and one instance only. As for GS, native
 * block words stay CPU-written (hk_apple9_native_gs_state_eligible); the TES
 * mesh grid itself is GPU-written, but only the object shader reads it. */
static bool
hk_apple9_native_tess_state_eligible(struct hk_cmd_buffer *cmd,
                                     struct agx_draw draw)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   return hk_apple9_native_tess_common_state_eligible(cmd, draw) &&
          !draw.indexed && !draw.start && !draw.start_instance &&
          draw.b.count[0] == dyn->ts.patch_control_points &&
          draw.b.count[1] == 1;
}

struct hk_apple9_fused_tess_chunk_plan {
   uint32_t inst_chunk_cap;
   uint32_t patch_chunk_cap;
};

static bool
hk_apple9_native_tess_fused_chunk_plan(
   uint32_t patches_per_inst, uint32_t instances, uint32_t max_groups,
   uint32_t max_child_wgs_per_patch, uint32_t max_prims_per_patch,
   uint64_t ring_max, uint64_t render_prim_cap,
   struct hk_apple9_fused_tess_chunk_plan *out)
{
   if (!patches_per_inst || !instances || !max_groups ||
       instances > HK_APPLE9_NATIVE_MAX_GROUP_COUNT)
      return false;
   const uint32_t group_cap =
      MIN2(max_groups, HK_APPLE9_NATIVE_MAX_GROUP_TOTAL);
   const uint64_t patch_prims = MAX2(max_prims_per_patch, 1u);
   if (render_prim_cap && patch_prims > render_prim_cap)
      return false;
   const uint64_t tvb_patch_cap =
      render_prim_cap ? render_prim_cap / patch_prims : UINT64_MAX;

   if (ring_max != UINT64_MAX) {
      uint32_t grid[3];
      if (!hk_apple9_native_gs_fused_grid(patches_per_inst, instances, grid))
         return false;
      uint64_t total_parents = (uint64_t)grid[0] * grid[1] * grid[2];
      if (total_parents > group_cap)
         return false;
      uint64_t total_patches = (uint64_t)patches_per_inst * instances;
      if (max_child_wgs_per_patch &&
          total_patches > ring_max / max_child_wgs_per_patch)
         return false;
      if (total_patches > tvb_patch_cap)
         return false;
      *out = (struct hk_apple9_fused_tess_chunk_plan){
         .inst_chunk_cap = instances,
         .patch_chunk_cap = patches_per_inst,
      };
      return true;
   }

   uint32_t single_grid[3];
   if (patches_per_inst <= group_cap && patches_per_inst <= tvb_patch_cap &&
       hk_apple9_native_gs_fused_grid(patches_per_inst, 1, single_grid)) {
      uint64_t grid_xy = (uint64_t)single_grid[0] * single_grid[1];
      if (grid_xy <= group_cap) {
         uint64_t by_prims = tvb_patch_cap / patches_per_inst;
         uint64_t by_grid = group_cap / grid_xy;
         uint32_t inst_cap = (uint32_t)MIN3(
            (uint64_t)instances, MIN2(by_prims, by_grid),
            (uint64_t)HK_APPLE9_NATIVE_MAX_GROUP_COUNT);
         uint32_t check_grid[3];
         if (inst_cap &&
             hk_apple9_native_gs_fused_grid(patches_per_inst, inst_cap,
                                            check_grid) &&
             (uint64_t)check_grid[0] * check_grid[1] * check_grid[2] <=
                group_cap &&
             DIV_ROUND_UP((uint64_t)instances, inst_cap) <=
                HK_APPLE9_NATIVE_TESS_MAX_CHUNKS) {
            *out = (struct hk_apple9_fused_tess_chunk_plan){
               .inst_chunk_cap = inst_cap,
               .patch_chunk_cap = patches_per_inst,
            };
            return true;
         }
      }
   }

   const uint64_t safe_group_cap =
      group_cap > 16 ? (uint64_t)group_cap - 16 : (uint64_t)group_cap;
   const uint32_t patch_cap = (uint32_t)MIN3(
      (uint64_t)patches_per_inst, safe_group_cap, tvb_patch_cap);
   if (!patch_cap)
      return false;
   uint32_t check_grid[3];
   if (!hk_apple9_native_gs_fused_grid(patch_cap, 1, check_grid) ||
       (uint64_t)check_grid[0] * check_grid[1] > group_cap)
      return false;
   uint64_t patch_chunks = DIV_ROUND_UP((uint64_t)patches_per_inst, patch_cap);
   if (patch_chunks > HK_APPLE9_NATIVE_TESS_MAX_CHUNKS ||
       (uint64_t)instances > HK_APPLE9_NATIVE_TESS_MAX_CHUNKS / patch_chunks)
      return false;
   *out = (struct hk_apple9_fused_tess_chunk_plan){
      .inst_chunk_cap = 1,
      .patch_chunk_cap = patch_cap,
   };
   return true;
}

bool
hk_apple9_use_native_tess(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_TESS_EVAL]);
   if (!tes || !tes->apple9_native_leaf ||
       (!hk_apple9_native_tess_enabled() &&
        tes->apple9_native_stage_mask != HK_APPLE9_NATIVE_TES_BIT))
      return false;
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_TESS_CTRL,
      MESA_SHADER_FRAGMENT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      struct hk_api_shader *api = gfx->shaders[stages[i]];
      if (stages[i] == MESA_SHADER_FRAGMENT &&
          (!api || api == dev->null_fs))
         continue;
      api = hk_api_shader_native(api);
      if (!api || !api->apple9_native_leaf ||
          (stages[i] != MESA_SHADER_FRAGMENT && api->writes_memory) ||
          api->apple9_native_stage_mask != tes->apple9_native_stage_mask ||
          memcmp(tes->apple9_native_pair_id, api->apple9_native_pair_id,
                 sizeof(tes->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = hk_native_producer_main(tes);
   const struct hk_shader *object = hk_native_producer_object(tes);
   const struct agx_apple9_mesh_body_info *body = &main->info.apple9_native;
   if (main->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
       object->info.apple9_native_stage != HK_APPLE9_NATIVE_OBJECT)
      return false;

   if (main->info.apple9_tess_fused) {
      if (!hk_apple9_native_tess_enabled() ||
          !hk_apple9_native_object_available_device(dev) ||
          !hk_apple9_native_tess_common_state_eligible(cmd, draw) ||
          main->info.apple9_tess_payload_B >
             AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES ||
          object->b.info.workgroup_size[0] >
             AGX_NATIVE_OBJECT_MAX_THREADS ||
          hk_apple9_native_object_tg_memory_bytes(&object->info.apple9_native) >
             AGX_NATIVE_OBJECT_MAX_SHARED_BYTES)
         return false;
      if (draw.indexed) {
         unsigned isz = agx_index_size_to_B(draw.index_size);
         if (isz != 1 && isz != 2 && isz != 4)
            return false;
      }
      uint64_t ring_max = 0;
      if (!agx_apple9_mesh_ring_max_groups(&body->layout, &ring_max))
         return false;
      const uint32_t pcp = cmd->vk.dynamic_graphics_state.ts.patch_control_points;
      const uint32_t patches = draw.b.count[0] / pcp;
      uint32_t max_child_wgs = 0, max_patch_prims = 0;
      hk_nir_tess_fused_eval_counts(&main->info,
                                    cmd->state.gfx.descriptors.root.push,
                                    &max_child_wgs, &max_patch_prims);
      struct hk_apple9_fused_tess_chunk_plan plan;
      return hk_apple9_native_tess_fused_chunk_plan(
         patches, draw.b.count[1], AGX_NATIVE_OBJECT_MAX_GROUPS,
         max_child_wgs, max_patch_prims, ring_max,
         hk_apple9_native_render_primitives(), &plan);
   }

   if (!hk_apple9_native_tess_state_eligible(cmd, draw))
      return false;
   /* One patch produces at most 2 * 64 * 64 triangles (packed TES launches
    * one group per HK_APPLE9_NATIVE_GS_PACK triangles). */
   const uint64_t max_tes_groups =
      body->layout.max_vertices == 3 * HK_APPLE9_NATIVE_GS_PACK
         ? DIV_ROUND_UP(2u * 64u * 64u, HK_APPLE9_NATIVE_GS_PACK)
         : 2u * 64u * 64u;
   return hk_apple9_native_stage_class(
             MESA_SHADER_TESS_EVAL, body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives) &&
          hk_apple9_packed_mesh_class_selected(
             hk_apple9_native_mesh_packed_shapes(),
             hk_apple9_native_tess_enabled() && body->publication_rounds
                ? hk_apple9_native_mesh_threads(body->logical_size)
                : body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives) &&
          agx_apple9_mesh_ring_fits(&body->layout, max_tes_groups);
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_route_log, "HK_APPLE9_TESS_ROUTE_LOG", false)

/* Zero-PREGFX fused OBJECT(VS+TCS) -> MESH(TES) launch: executes 100% inside
 * the active render pass with zero CDM dispatches, zero tessellation heap
 * round-trip, and zero HK_READS_INTERNAL tiling barrier. */
static void
hk_apple9_launch_native_tess_fused(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                                   struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *tes_api =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_TESS_EVAL]);
   const struct hk_shader *main = hk_native_producer_main(tes_api);
   const struct agx_apple9_mesh_layout *layout = &main->info.apple9_native.layout;

   if (cmd->tess_batch.draws)
      hk_tess_batch_flush(cmd);

   const uint32_t pcp = dyn->ts.patch_control_points;
   const uint32_t patches = draw.b.count[0] / pcp;
   const uint32_t instances = draw.b.count[1];
   uint64_t ring_max = 0;
   bool ring_ok = agx_apple9_mesh_ring_max_groups(layout, &ring_max);
   assert(ring_ok);
   (void)ring_ok;
   uint32_t max_child_wgs = 0, max_patch_prims = 0;
   hk_nir_tess_fused_eval_counts(&main->info, desc->root.push, &max_child_wgs,
                                 &max_patch_prims);
   struct hk_apple9_fused_tess_chunk_plan plan = {0};
   bool plan_ok = hk_apple9_native_tess_fused_chunk_plan(
      patches, instances, AGX_NATIVE_OBJECT_MAX_GROUPS, max_child_wgs,
      max_patch_prims, ring_max, hk_apple9_native_render_primitives(), &plan);
   assert(plan_ok);
   if (!plan_ok)
      return;

   const uint32_t index_size_B =
      draw.indexed ? agx_index_size_to_B(draw.index_size) : 0;
   uint64_t index_buffer = 0, index_buffer_size = 0;
   if (draw.indexed) {
      uint64_t byte_offset = (uint64_t)draw.start * index_size_B;
      if (byte_offset < draw.index_buffer_range_B) {
         index_buffer = draw.index_buffer + byte_offset;
         index_buffer_size = draw.index_buffer_range_B - byte_offset;
      }
   }
   const uint32_t first_vertex =
      draw.indexed ? (uint32_t)draw.index_bias : draw.start;
   const uint32_t flip_winding =
      (tes_api->apple9_native_tess.ccw ^
       (dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT))
         ? 1u
         : 0u;
   const uint64_t est_patch_prims = max_patch_prims;
   const uint64_t est_child_wgs = max_child_wgs;

   for (uint64_t first_inst = 0; first_inst < instances;) {
      const uint32_t chunk_insts =
         (uint32_t)MIN2((uint64_t)plan.inst_chunk_cap, instances - first_inst);
      for (uint64_t first_patch = 0; first_patch < patches;) {
         const uint32_t chunk_patches = (uint32_t)MIN2(
            (uint64_t)plan.patch_chunk_cap, patches - first_patch);
         uint32_t obj_groups[3];
         bool ok = hk_apple9_native_gs_fused_grid(chunk_patches, chunk_insts,
                                                  obj_groups);
         assert(ok);
         (void)ok;

         const uint64_t chunk_total_patches =
            (uint64_t)chunk_patches * chunk_insts;
         cmd->apple9_draw_primitives =
            est_patch_prims &&
                  chunk_total_patches > UINT64_MAX / est_patch_prims
               ? UINT64_MAX
               : chunk_total_patches * est_patch_prims;
         uint64_t chunk_child_wgs =
            est_child_wgs && chunk_total_patches > UINT64_MAX / est_child_wgs
               ? UINT64_MAX
               : chunk_total_patches * est_child_wgs;
         if (!agx_apple9_mesh_ring_usage(layout, chunk_child_wgs,
                                         &cmd->apple9_draw_ring_rows,
                                         &cmd->apple9_draw_ring_limited)) {
            cmd->apple9_draw_ring_rows = UINT64_MAX;
            cmd->apple9_draw_ring_limited = true;
         }
         if (!hk_apple9_prepare_draw_ex(cmd, draw.indexed, true, true))
            return;

         const struct hk_apple9_fused_tess_params tp = {
            .index_buffer = index_buffer,
            .index_buffer_size = index_buffer_size,
            .index_size_B = index_size_B,
            .input_patch_size = pcp,
            .first_vertex = first_vertex,
            .base_instance = draw.start_instance,
            .patches_per_instance = chunk_patches,
            .first_patch = (uint32_t)first_patch,
            .groups_x = obj_groups[0],
            .first_instance_offset = (uint32_t)first_inst,
            .flip_winding = flip_winding,
         };
         uint64_t tp_gpu = hk_pool_upload(cmd, &tp, sizeof(tp), 8);
         if (!tp_gpu)
            return;
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
         desc->root.draw.tess_params = tp_gpu;
         desc->root.draw.geometry_params = 0;
         desc->root.draw.api_gs = false;
         desc->root.draw.rasterization_stream = 0;
         desc->root.draw.draw_id = draw_id;
         desc->root_dirty = true;
         hk_apple9_draw_native_tess_object(cmd, draw_id, draw, obj_groups,
                                           tp_gpu);
         if (vk_command_buffer_has_error(&cmd->vk))
            return;
         first_patch += chunk_patches;
      }
      first_inst += chunk_insts;
   }
}

static void
hk_apple9_launch_native_tess(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                             struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   if (!hk_apple9_native_lanes(hk_cmd_buffer_device(cmd)))
      hk_tess_batch_flush(cmd);
   hk_apple9_merge_tess_info(cmd, true);
   desc->root.draw.geometry_params = 0;
   desc->root.draw.api_gs = false;
   desc->root.draw.rasterization_stream = 0;
   desc->root.draw.draw_id = draw_id;
   desc->root_dirty = true;
   hk_apple9_launch_tess(cmd, draw, true);
   /* The render consumes both the count written by prefix sum and the
    * indices/coordinates written by emit. HK_READS_INTERNAL in prepare_draw
    * gives it the pre_gfx tiling dependency, including merge continuations. */
   if (!vk_command_buffer_has_error(&cmd->vk))
      hk_apple9_draw_native_mesh(cmd, draw_id, gfx->tess.out_draws);
}

struct hk_apple9_fused_gs_chunk_plan {
   uint32_t inst_chunk_cap;
   uint32_t prim_chunk_cap;
};

/* Plan TVB- and grid-safe chunking for a fused GS draw while preserving exact
 * Vulkan (instance, primitive) submission order: either multiple whole
 * instances per dispatch (prim_chunk_cap == input_prims) or one complete
 * instance at a time chunked by first_primitive (inst_chunk_cap == 1). */
static bool
hk_apple9_native_gs_fused_chunk_plan(uint32_t input_prims, uint32_t instances,
                                     uint32_t inv, uint32_t pack,
                                     uint32_t prims_per_run, uint64_t ring_max,
                                     uint64_t render_prim_cap,
                                     struct hk_apple9_fused_gs_chunk_plan *out)
{
   if (!input_prims || !instances ||
       instances > HK_APPLE9_NATIVE_MAX_GROUP_COUNT)
      return false;

   const uint64_t per_prim_out = (uint64_t)inv * prims_per_run;
   if (!per_prim_out || per_prim_out > render_prim_cap)
      return false;

   const uint64_t per_inst_runs = (uint64_t)input_prims * inv;
   const uint64_t per_inst_groups = DIV_ROUND_UP(per_inst_runs, pack);

   if (ring_max != UINT64_MAX) {
      uint32_t grid[3];
      if (!hk_apple9_native_gs_fused_grid(per_inst_groups, instances, grid))
         return false;
      const uint64_t total_grid_groups =
         (uint64_t)grid[0] * grid[1] * grid[2];
      if (total_grid_groups > ring_max)
         return false;
      if (per_inst_runs > render_prim_cap / prims_per_run)
         return false;
      const uint64_t per_inst_out = per_inst_runs * prims_per_run;
      if (per_inst_out > render_prim_cap / instances)
         return false;
      *out = (struct hk_apple9_fused_gs_chunk_plan){
         .inst_chunk_cap = instances,
         .prim_chunk_cap = input_prims,
      };
      return true;
   }

   uint32_t single_grid[3];
   if (per_inst_runs <= render_prim_cap / prims_per_run &&
       hk_apple9_native_gs_fused_grid(per_inst_groups, 1, single_grid)) {
      const uint64_t per_inst_out = per_inst_runs * prims_per_run;
      const uint64_t by_prims = render_prim_cap / per_inst_out;
      const uint64_t grid_xy = (uint64_t)single_grid[0] * single_grid[1];
      const uint64_t by_grid = HK_APPLE9_NATIVE_MAX_GROUP_TOTAL / grid_xy;
      const uint32_t inst_cap = (uint32_t)MIN3(
         (uint64_t)instances, MIN2(by_prims, by_grid),
         (uint64_t)HK_APPLE9_NATIVE_MAX_GROUP_COUNT);
      uint32_t check_grid[3];
      if (!inst_cap ||
          !hk_apple9_native_gs_fused_grid(per_inst_groups, inst_cap, check_grid))
         return false;
      if (DIV_ROUND_UP((uint64_t)instances, inst_cap) > 65536)
         return false;
      *out = (struct hk_apple9_fused_gs_chunk_plan){
         .inst_chunk_cap = inst_cap,
         .prim_chunk_cap = input_prims,
      };
      return true;
   }

   const uint64_t tvb_prims_cap = render_prim_cap / per_prim_out;
   const uint64_t safe_grid_groups =
      HK_APPLE9_NATIVE_MAX_GROUP_TOTAL > 16
         ? HK_APPLE9_NATIVE_MAX_GROUP_TOTAL - 16
         : HK_APPLE9_NATIVE_MAX_GROUP_TOTAL;
   const uint64_t grid_prims_cap = (safe_grid_groups * pack) / inv;
   const uint32_t prim_cap =
      (uint32_t)MIN3((uint64_t)input_prims, tvb_prims_cap, grid_prims_cap);
   if (!prim_cap)
      return false;
   const uint64_t prim_chunks = DIV_ROUND_UP((uint64_t)input_prims, prim_cap);
   if (prim_chunks > 65536 || (uint64_t)instances > 65536 / prim_chunks)
      return false;
   *out = (struct hk_apple9_fused_gs_chunk_plan){
      .inst_chunk_cap = 1,
      .prim_chunk_cap = prim_cap,
   };
   return true;
}

static uint32_t hk_apple9_native_gs_groups(const struct hk_cmd_buffer *cmd,
                                           struct agx_draw draw);

bool
hk_apple9_use_native_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]);
   struct hk_api_shader *vs =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]);
   if (!gs || !vs ||
       gfx->shaders[MESA_SHADER_GEOMETRY]->is_passthrough ||
       gs->is_passthrough || !gs->apple9_native_leaf ||
       !vs->apple9_native_leaf || vs->writes_memory ||
       (!hk_apple9_native_gs_enabled() &&
        gs->apple9_native_stage_mask != HK_APPLE9_NATIVE_GS_BIT) ||
       vs->apple9_native_stage_mask != gs->apple9_native_stage_mask ||
       memcmp(gs->apple9_native_pair_id, vs->apple9_native_pair_id,
              sizeof(gs->apple9_native_pair_id)))
      return false;
   const struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (fs && fs != hk_cmd_buffer_device(cmd)->null_fs) {
      fs = hk_api_shader_native(fs);
      if (!fs || !fs->apple9_native_leaf ||
          fs->apple9_native_stage_mask != gs->apple9_native_stage_mask ||
          memcmp(gs->apple9_native_pair_id, fs->apple9_native_pair_id,
                 sizeof(gs->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = hk_native_producer_main(gs);
   const struct agx_apple9_mesh_body_info *body = &main->info.apple9_native;
   if (main->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
       hk_native_producer_object(gs)->info.apple9_native_stage !=
          HK_APPLE9_NATIVE_OBJECT)
      return false;

   if (main->info.apple9_gs_fused) {
      if (!hk_apple9_native_gs_enabled() ||
          !hk_apple9_native_gs_common_state_eligible(cmd, draw))
         return false;
      const uint32_t input_prims = hk_apple9_native_gs_fused_input_primitives(
         &cmd->vk.dynamic_graphics_state, draw,
         main->info.apple9_gs_input_primitive);
      if (draw.indexed) {
         unsigned isz = agx_index_size_to_B(draw.index_size);
         if (isz != 1 && isz != 2 && isz != 4)
            return false;
      }
      const uint32_t inv = MAX2(main->info.apple9_gs_invocations, 1u);
      const uint32_t pack = MAX2(main->info.apple9_gs_pack, 1u);
      const uint32_t prims_per_run = MAX2(main->info.apple9_gs_prims_per_run, 1u);
      uint64_t ring_max = 0;
      if (!agx_apple9_mesh_ring_max_groups(&body->layout, &ring_max))
         return false;
      struct hk_apple9_fused_gs_chunk_plan plan;
      return hk_apple9_native_gs_fused_chunk_plan(
         input_prims, draw.b.count[1], inv, pack, prims_per_run, ring_max,
         hk_apple9_native_render_primitives(), &plan);
   }

   if (!hk_apple9_native_gs_state_eligible(cmd, draw))
      return false;
   const uint32_t gs_groups = hk_apple9_native_gs_groups(cmd, draw);
   return (hk_apple9_native_gs_enabled() || hk_apple9_native_stage_class(
             MESA_SHADER_GEOMETRY, body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives)) &&
          hk_apple9_packed_mesh_class_selected(
             hk_apple9_native_mesh_packed_shapes(),
             hk_apple9_native_gs_enabled() && body->publication_rounds
                ? hk_apple9_native_mesh_threads(body->logical_size)
                : body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives) &&
          /* GS invocations multiply the groups beyond the draw-only bound. */
          hk_apple9_native_mesh_grid_valid(
             (const uint32_t[]){gs_groups, draw.b.count[1], 1}) &&
          agx_apple9_mesh_ring_fits(&body->layout,
                                    (uint64_t)gs_groups * draw.b.count[1]);
}

DEBUG_GET_ONCE_BOOL_OPTION(gs_route_log, "HK_APPLE9_GS_ROUTE_LOG", false)

/* Mesh groups per instance of a native GS draw: one per logical GS run
 * (input triangle and invocation), or one per HK_APPLE9_NATIVE_GS_PACK runs
 * for a packed body, recognised by its exact class (hk_nir_gs_mesh). */
static uint32_t
hk_apple9_native_gs_groups(const struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_apple9_mesh_body_info *body =
      &hk_native_producer_main(hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]))
          ->info.apple9_native;
   /* One logical GS run per input triangle and GS invocation. */
   const uint64_t runs =
      (uint64_t)hk_apple9_native_gs_triangles(&cmd->vk.dynamic_graphics_state, draw) *
      MAX2(hk_native_producer_main(hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]))
              ->info.apple9_gs_invocations, 1);
   bool packed = hk_apple9_native_gs_packed_class(
      body->key.physical_threads, body->layout.max_vertices, body->layout.max_primitives);
   /* Unpacked bodies publish V(n)/P(n-2) (T32 only for n > 16 on the
    * development route), never a packed class, so the class alone decides. */
   const uint64_t groups = packed ? DIV_ROUND_UP(runs, HK_APPLE9_NATIVE_GS_PACK) : runs;
   /* Saturate so an oversized grid fails hk_apple9_native_mesh_grid_valid. */
   return MIN2(groups, UINT32_MAX);
}

/* Zero-PREGFX fused VS + GS launch: executes 100% inside the render pass with
 * no CDM compute dispatch, no geometry heap round-trip, and no R->C->R barrier. */
static void
hk_apple9_launch_native_gs_fused(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                                 struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   const struct hk_shader *main = hk_native_producer_main(
      hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]));
   const struct agx_apple9_mesh_layout *layout = &main->info.apple9_native.layout;
   const uint32_t input_prims = hk_apple9_native_gs_fused_input_primitives(
      &cmd->vk.dynamic_graphics_state, draw,
      main->info.apple9_gs_input_primitive);
   const uint32_t instances = draw.b.count[1];
   const uint32_t inv = MAX2(main->info.apple9_gs_invocations, 1u);
   const uint32_t pack = MAX2(main->info.apple9_gs_pack, 1u);
   const uint32_t prims_per_run = MAX2(main->info.apple9_gs_prims_per_run, 1u);

   uint64_t ring_max = 0;
   bool ring_ok = agx_apple9_mesh_ring_max_groups(layout, &ring_max);
   assert(ring_ok);
   (void)ring_ok;
   struct hk_apple9_fused_gs_chunk_plan plan = {0};
   bool plan_ok = hk_apple9_native_gs_fused_chunk_plan(
      input_prims, instances, inv, pack, prims_per_run, ring_max,
      hk_apple9_native_render_primitives(), &plan);
   assert(plan_ok);
   if (!plan_ok)
      return;

   const uint32_t index_size_B =
      draw.indexed ? agx_index_size_to_B(draw.index_size) : 0;
   uint64_t index_buffer = 0, index_buffer_size = 0;
   if (draw.indexed) {
      uint64_t byte_offset = (uint64_t)draw.start * index_size_B;
      if (byte_offset < draw.index_buffer_range_B) {
         index_buffer = draw.index_buffer + byte_offset;
         index_buffer_size = draw.index_buffer_range_B - byte_offset;
      }
   }
   const uint32_t input_topology =
      vk_topology_to_mesa(cmd->vk.dynamic_graphics_state.ia.primitive_topology);
   const uint32_t first_vertex =
      draw.indexed ? (uint32_t)draw.index_bias : draw.start;

   for (uint64_t first_inst = 0; first_inst < instances;) {
      const uint32_t chunk_insts =
         (uint32_t)MIN2((uint64_t)plan.inst_chunk_cap, instances - first_inst);
      for (uint64_t first_prim = 0; first_prim < input_prims;) {
         const uint32_t chunk_prims = (uint32_t)MIN2(
            (uint64_t)plan.prim_chunk_cap, input_prims - first_prim);
         const uint64_t chunk_runs = (uint64_t)chunk_prims * inv;
         const uint64_t chunk_groups = DIV_ROUND_UP(chunk_runs, pack);
         uint32_t grid_dims[3];
         bool ok = hk_apple9_native_gs_fused_grid(chunk_groups, chunk_insts,
                                                  grid_dims);
         assert(ok);
         (void)ok;

         const uint64_t total_grid_groups =
            (uint64_t)grid_dims[0] * grid_dims[1] * grid_dims[2];
         cmd->apple9_draw_primitives = chunk_runs * prims_per_run * chunk_insts;
         if (!agx_apple9_mesh_ring_usage(layout, total_grid_groups,
                                         &cmd->apple9_draw_ring_rows,
                                         &cmd->apple9_draw_ring_limited)) {
            cmd->apple9_draw_ring_rows = UINT64_MAX;
            cmd->apple9_draw_ring_limited = true;
         }
         if (!hk_apple9_prepare_draw(cmd, draw.indexed, true))
            return;

         const struct hk_apple9_fused_gs_params gp = {
            .index_buffer = index_buffer,
            .index_buffer_size = index_buffer_size,
            .input_topology = input_topology,
            .index_size_B = index_size_B,
            .first_vertex = first_vertex,
            .base_instance = draw.start_instance,
            .input_primitives = chunk_prims,
            .first_primitive = (uint32_t)first_prim,
            .groups_x = grid_dims[0],
            .first_instance_offset = (uint32_t)first_inst,
         };
         desc->root.draw.geometry_params = hk_pool_upload(cmd, &gp, sizeof(gp), 8);
         if (!desc->root.draw.geometry_params)
            return;
         desc->root.draw.api_gs = true;
         desc->root.draw.rasterization_stream = 0;
         desc->root.draw.draw_id = draw_id;
         desc->root_dirty = true;
         uint64_t grid = hk_pool_upload(cmd, grid_dims, sizeof(grid_dims), 4);
         if (!grid)
            return;
         hk_apple9_draw_native_mesh_draw(cmd, draw_id, draw, grid);
         if (vk_command_buffer_has_error(&cmd->vk))
            return;
         first_prim += chunk_prims;
      }
      first_inst += chunk_insts;
   }
}

static void
hk_apple9_launch_native_gs(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                           struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_api_shader *vs_api =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]);
   struct hk_shader *vs = &vs_api->variants[HK_VS_VARIANT_SW];
   /* Complete any previously captured pre-raster lanes before this VS.
    * The existing pre_gfx stream is a dependency of the isolated render;
    * HK_READS_INTERNAL keeps the wait at tiling, never fragment-only.
    * In a batched native render the VS joins lane 0 instead (see
    * hk_apple9_native_lanes). */
   bool lanes = hk_apple9_native_lanes(hk_cmd_buffer_device(cmd)) &&
                !agx_is_indirect(draw.b) &&
                !desc->root.draw.pipeline_stats_flags;
   if (!lanes)
      hk_tess_batch_flush(cmd);
   struct hk_apple9_prerast_bindings bindings = hk_apple9_draw_bindings(cmd, draw);
   desc->root.draw.vertex_params = hk_upload_vertex_params_for_vs(cmd, draw, vs);
   /* The body reads only input_topology (list, strip or fan). */
   const struct poly_geometry_params gp = {
      .input_topology =
         vk_topology_to_mesa(cmd->vk.dynamic_graphics_state.ia.primitive_topology),
   };
   desc->root.draw.geometry_params = hk_pool_upload(cmd, &gp, sizeof(gp), 8);
   if (!desc->root.draw.geometry_params)
      return;
   desc->root.draw.api_gs = true;
   desc->root.draw.rasterization_stream = 0;
   desc->root.draw.draw_id = draw_id;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   cmd->tess_batch.capture = lanes ? &cmd->tess_batch.lane[0] : NULL;
   hk_apple9_dispatch_prerast(cmd, vs, draw.b, &bindings);
   cmd->tess_batch.capture = NULL;
   if (lanes)
      cmd->tess_batch.draws++;
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* One mesh group per input triangle (x) and instance (y), or per
    * HK_APPLE9_NATIVE_GS_PACK triangles for a packed (T32) body; the body
    * reads its corners from the VS buffer (hk_nir_gs_mesh). */
   const uint32_t groups[] = {hk_apple9_native_gs_groups(cmd, draw), draw.b.count[1], 1};
   uint64_t grid = hk_pool_upload(cmd, groups, sizeof(groups), 4);
   if (grid)
      hk_apple9_draw_native_mesh(cmd, draw_id, grid);
}

/* Index size and primitive restart come from command buffer state, not the
 * draw entry points (hk_draw_indexed passes neither). Everything that routes
 * or launches a draw reads both - an indexed native GS fetches with this index
 * size - so G16 completes the draw before any of it. */
static struct agx_draw
hk_draw_complete_from_state(const struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   draw.restart = cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable &&
                  draw.indexed;
   draw.index_size = cmd->state.gfx.index.size;
   return draw;
}

uint64_t
hk_apple9_ordinary_draw_primitives(const struct hk_cmd_buffer *cmd,
                                   struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   /* Emulated GS/TES and indirect draws have GPU-determined output counts. */
   if (agx_is_indirect(draw.b) || gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return UINT64_MAX;
   uint64_t prims =
      (uint64_t)u_decomposed_prims_for_vertices(
         vk_topology_to_mesa(dyn->ia.primitive_topology), draw.b.count[0]) *
      draw.b.count[1];
   /* Point/wireframe polygon modes can expand each triangle up to 3x. */
   if (dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL)
      prims = prims > UINT64_MAX / 3 ? UINT64_MAX : prims * 3;
   return prims;
}

static void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw_)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   draw_ = hk_draw_complete_from_state(cmd, draw_);
   if (!hk_feedback_prepare_draw(cmd))
      return;
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   /* Binned-primitive estimate for the native render bound; native GS/TES
    * below replace it with their output estimate. Reserve the full multiview
    * total during preparation so multi-view ordinary draws split cleanly
    * before view 0 instead of overflowing the backstop mid-loop. */
   const uint64_t per_view_prims =
      hk_apple9_ordinary_draw_primitives(cmd, draw_);
   const unsigned views = util_bitcount(gfx->render.view_mask ?: 1);
   cmd->apple9_draw_primitives =
      per_view_prims > UINT64_MAX / views ? UINT64_MAX : per_view_prims * views;
   /* Batched native renders defer the ordinary split until stage
    * selection: a native GS/TES draw may continue the render. Also defer when
    * a fused native GS draw is eligible so prepare_draw is never called with
    * native_mesh=false (which would tag the graphics CS with HK_READS_INTERNAL). */
   bool native_gs = hk_apple9_use_native_gs(cmd, draw_);
   bool native_tess = hk_apple9_use_native_tess(cmd, draw_);
   bool defer_prepare =
      (hk_apple9_native_blocks_enabled(hk_cmd_buffer_device(cmd)) &&
       cmd->current_cs.gfx && cmd->current_cs.gfx->apple9_native_mesh) ||
      native_gs || native_tess;
   /* Index fetch and indirect arguments are buffer reads of tiling. */
   if (!defer_prepare &&
       !hk_apple9_prepare_draw(cmd, agx_is_indirect(draw_.b) || draw_.indexed, false))
      return;
   mesa_shader_stage producer = gfx->shaders[MESA_SHADER_GEOMETRY]
                                   ? MESA_SHADER_GEOMETRY
                                   : MESA_SHADER_TESS_EVAL;
   const struct hk_api_shader *source = gfx->shaders[producer];
   bool prefer_native =
      source && !source->is_passthrough &&
      p_atomic_read(&source->apple9_native_candidate) &&
      (producer == MESA_SHADER_GEOMETRY
          ? hk_apple9_native_gs_common_state_eligible(cmd, draw_)
          : hk_apple9_native_tess_common_state_eligible(cmd, draw_));
   if (cmd->feedback_raw.size) {
      for (unsigned stage = 0; stage <= MESA_SHADER_MESH; stage++) {
         if (cmd->state.gfx.shaders[stage] &&
             cmd->state.gfx.shaders[stage]->writes_memory) {
            hk_feedback_invalidate(cmd);
            break;
         }
      }
   }
   if (producer == MESA_SHADER_GEOMETRY && source &&
       debug_get_option_gs_route_log()) {
      fprintf(stderr, "HK_GS_SELECT candidate=%u prefer=%u state=%u leaf=%u vs_leaf=%u vs_writes=%u\n",
              p_atomic_read(&source->apple9_native_candidate), prefer_native,
              hk_apple9_native_gs_common_state_eligible(cmd, draw_),
              !!hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]),
              !!hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]),
              gfx->shaders[MESA_SHADER_VERTEX]->writes_memory);
      struct hk_api_shader *gs = native_gs
         ? hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]) : NULL;
      const struct hk_shader *main = gs ? hk_native_producer_main(gs) : NULL;
      const struct agx_apple9_mesh_body_info *body =
         main ? &main->info.apple9_native : NULL;
      fprintf(stderr, "HK_GS_ROUTE native=%u fused=%u pack=%u T=%u V=%u P=%u\n",
              native_gs, main ? main->info.apple9_gs_fused : 0,
              main ? main->info.apple9_gs_pack : 0,
              body ? body->key.physical_threads : 0,
              body ? body->layout.max_vertices : 0,
              body ? body->layout.max_primitives : 0);
   }
   if (native_gs) {
      const struct hk_shader *gs_main = hk_native_producer_main(
         hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]));
      if (gs_main->info.apple9_gs_fused) {
         hk_apple9_launch_native_gs_fused(cmd, draw_id, draw_);
         return;
      }
      /* Each mesh group of each instance emits at most the body's
       * declared primitives (hk_apple9_native_gs_state_eligible). */
      const struct agx_apple9_mesh_layout *gs_layout =
         &gs_main->info.apple9_native.layout;
      uint64_t gs_groups =
         (uint64_t)hk_apple9_native_gs_groups(cmd, draw_) * draw_.b.count[1];
      cmd->apple9_draw_primitives = gs_layout->max_primitives * gs_groups;
      if (!agx_apple9_mesh_ring_usage(gs_layout, gs_groups,
                                      &cmd->apple9_draw_ring_rows,
                                      &cmd->apple9_draw_ring_limited)) {
         cmd->apple9_draw_ring_rows = UINT64_MAX;
         cmd->apple9_draw_ring_limited = true;
      }
      /* The software VS writes every vertex of every instance. The grid
       * bound keeps that buffer small today; check it as fallback does. */
      if (!hk_sw_vs_outputs_fit(cmd, draw_))
         return;
      if (hk_apple9_prepare_draw(cmd, true, true))
         hk_apple9_launch_native_gs(cmd, draw_id, draw_);
      return;
   }
   if (producer == MESA_SHADER_TESS_EVAL && source &&
       debug_get_option_tess_route_log())
      fprintf(stderr, "HK_TESS_ROUTE native=%u candidate=%u prefer=%u\n",
              native_tess, p_atomic_read(&source->apple9_native_candidate),
              prefer_native);
   if (native_tess) {
      const struct hk_shader *tes_main = hk_native_producer_main(
         hk_api_shader_native(gfx->shaders[MESA_SHADER_TESS_EVAL]));
      if (tes_main->info.apple9_tess_fused) {
         hk_apple9_launch_native_tess_fused(cmd, draw_id, draw_);
         return;
      }
      /* One patch (hk_apple9_native_tess_state_eligible), at most a
       * fully tessellated triangle domain: 2 * 64 * 64 triangles. */
      const struct agx_apple9_mesh_body_info *tes_body =
         &tes_main->info.apple9_native;
      uint64_t max_tes_groups =
         tes_body->key.physical_threads == 32
            ? DIV_ROUND_UP(2u * 64u * 64u, HK_APPLE9_NATIVE_GS_PACK)
            : 2u * 64u * 64u;
      cmd->apple9_draw_primitives = 2 * 64 * 64;
      if (!agx_apple9_mesh_ring_usage(&tes_body->layout, max_tes_groups,
                                      &cmd->apple9_draw_ring_rows,
                                      &cmd->apple9_draw_ring_limited)) {
         cmd->apple9_draw_ring_rows = UINT64_MAX;
         cmd->apple9_draw_ring_limited = true;
      }
      if (hk_apple9_prepare_draw(cmd, true, true))
         hk_apple9_launch_native_tess(cmd, draw_id, draw_);
      return;
   }
   if (defer_prepare &&
       !hk_apple9_prepare_draw(cmd, agx_is_indirect(draw_.b) || draw_.indexed, false))
      return;
   if (hk_cond_render_applies(cmd)) {
      draw_ = hk_apple9_predicate_draw(cmd, draw_);
      if (!draw_.b.ptr)
         return;
   }
   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;
   uint64_t ia_verts = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);
   uint64_t ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);
   uint64_t vs_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);
   uint64_t c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
   uint64_t c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
   if (hk_stat_enabled(ia_verts) || hk_stat_enabled(ia_prims) ||
       hk_stat_enabled(vs_inv) || hk_stat_enabled(c_inv) ||
       hk_stat_enabled(c_prims)) {
      hk_ia_update(cmd, draw_, ia_verts, ia_prims, vs_inv, c_prims, c_inv);
   }
   if (cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL])
      hk_apple9_merge_tess_info(cmd, false);
   hk_handle_passthrough_gs(cmd, draw_);
   if (!hk_sw_vs_outputs_fit(cmd, draw_))
      return;
   /* Geometry and tessellation emulation decompose fans themselves. When
    * nothing observes the provoking vertex, the fan is drawn natively from
    * its first vertex, see hk_apple9_draw_impl. */
   cmd->state.gfx.apple9_fan_unrolled =
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT &&
      !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
      !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
      hk_apple9_provoking_observed(cmd);
   if (cmd->state.gfx.apple9_fan_unrolled)
      draw_ = hk_apple9_unroll_fan(cmd, draw_);
   /* Software stages read the draw ID from the root table they upload,
    * when one of the bound stages they run reads it. */
   if (hk_apple9_reads_draw_id(&cmd->state.gfx) &&
       cmd->state.gfx.descriptors.root.draw.draw_id != draw_id) {
      cmd->state.gfx.descriptors.root.draw.draw_id = draw_id;
      cmd->state.gfx.descriptors.root_dirty = true;
   }
   bool adjacency =
      !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
      !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
      mesa_prim_has_adjacency(vk_topology_to_mesa(dyn->ia.primitive_topology));
   cmd->apple9_draw_primitives = per_view_prims;
   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      if (adjacency)
         draw = hk_apple9_draw_without_adj(cmd, draw);
      else if (!cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
               !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
               hk_apple9_needs_index_copy(cmd, draw)) {
         uint64_t t = hk_cpu_sub_begin();
         draw = hk_apple9_robust_index(cmd, draw);
         hk_cpu_sub_end(t, HK_CPU_SUB_ROBUST_INDEX);
      }
      if (cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL]) {
         uint64_t t = hk_cpu_sub_begin();
         draw = hk_apple9_launch_tess(cmd, draw, false);
         hk_cpu_sub_end(t, HK_CPU_SUB_TESS);
      }
      if (cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY]) {
         uint64_t t = hk_cpu_sub_begin();
         draw = hk_apple9_launch_gs(cmd, draw);
         hk_cpu_sub_end(t, HK_CPU_SUB_GS);
      }
      uint64_t t = hk_cpu_sub_begin();
      hk_apple9_draw(cmd, draw_id, draw);
      hk_cpu_sub_end(t, HK_CPU_SUB_A9DRAW);
   }
}

static void
hk_init_mesh_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   size_t size = 512 * 1024 * 1024;
   dev->mesh_heap = agx_bo_create(&dev->dev, size, 0, 0, "Mesh heap");

   off_t off = dev->rodata.mesh_heap - dev->rodata.bo->va->addr;
   struct poly_heap *map = agx_bo_map(dev->rodata.bo) + off;
   *map = (struct poly_heap){
      .base = dev->mesh_heap ? dev->mesh_heap->va->addr : 0,
      .size = dev->mesh_heap ? size : 0,
   };
}

/* Records of mesh shader draws sized on the GPU are allocated from a heap,
 * freed after each command buffer.
 */
static uint64_t
hk_mesh_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->mesh_heap_init_once, hk_init_mesh_heap, cmd);
   if (!dev->mesh_heap)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   if (!cmd->uses_mesh_heap) {
      /* Reset the mesh heap in pre_gfx, matching hk_heap(), instead of
       * spawning a post_gfx stream after the render.
       */
      perf_debug(cmd, "Resetting mesh heap");
      libagx_heap_reset(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                        dev->rodata.mesh_heap);
      cmd->uses_mesh_heap = true;
   }

   return dev->rodata.mesh_heap;
}

static uint64_t
hk_mesh_stat(struct hk_cmd_buffer *cmd, VkQueryPipelineStatisticFlagBits stat)
{
   uint64_t addr = hk_pipeline_stat_addr(cmd, stat);
   return hk_stat_enabled(addr) ? addr : 0;
}

/* Check state before resolving a deferred shader, so an ineligible draw only
 * compiles emulation. The final check also verifies the concrete program pair. */
static bool
hk_apple9_native_mesh_state_eligible(struct hk_cmd_buffer *cmd,
                                     const uint32_t *direct,
                                     uint64_t count_addr, bool cull_primitive)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   bool allow_native_mesh = hk_apple9_native_mesh_enabled();
   bool allow_task_native = hk_apple9_native_task_bridge_enabled() ||
                            hk_apple9_native_object_available_device(dev);
   bool allow_taskless_gpu_grid = allow_native_mesh &&
                                  !gfx->shaders[MESA_SHADER_TASK];
   if (!hk_apple9_native_mesh_available_device(dev) ||
       (direct ? (count_addr || !hk_apple9_native_mesh_grid_valid(direct))
               : !allow_taskless_gpu_grid) ||
       (gfx->shaders[MESA_SHADER_TASK] && !allow_task_native) ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       (!allow_native_mesh && gfx->render.view_mask) ||
       !hk_apple9_native_layers_samples_eligible(
          gfx, dyn, allow_native_mesh) ||
       (hk_cond_render_applies(cmd) && !allow_taskless_gpu_grid) ||
       gfx->mesh_pg_query || gfx->descriptors.root.draw.pipeline_stats_flags ||
       (!allow_native_mesh && gfx->occlusion.active_count) ||
       gfx->xfb_enabled ||
       dyn->rs.line.stipple.enable ||
       (dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL &&
        (!allow_native_mesh || dyn->rs.polygon_mode != VK_POLYGON_MODE_LINE)))
      return false;

   /* Primitive and cull-distance culling use degenerate indices, which
    * conservative rasterization or wireframe line mode may still cover. */
   return !cull_primitive ||
          (dyn->rs.polygon_mode == VK_POLYGON_MODE_FILL &&
           dyn->rs.conservative_mode ==
              VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
}

/* Maximum workgroup count safe to execute natively in a single mesh render:
 * bounded by Apple9's 2^20 - 1 child-grid total, the per-render TVB primitive
 * cap, and the 64 KiB hardware mesh output ring. Legal indirect or dynamic-task
 * grids above this bound overflow to the paired emulated fallback. */
static uint32_t
hk_apple9_native_mesh_max_groups(const struct agx_apple9_mesh_layout *layout)
{
   uint64_t ring_max = 0;
   if (!agx_apple9_mesh_ring_max_groups(layout, &ring_max))
      return 0;
   uint32_t max_prims = MAX2(layout->max_primitives, 1u);
   uint64_t limit = MIN2((uint64_t)HK_APPLE9_NATIVE_MAX_GROUP_TOTAL,
                         hk_apple9_native_render_primitives() / max_prims);
   limit = MIN2(limit, ring_max);
   return (uint32_t)limit;
}

static bool
hk_apple9_native_mesh_leaves_eligible(struct hk_cmd_buffer *cmd,
                                      const uint32_t *direct,
                                      uint64_t count_addr,
                                      const struct hk_api_shader *mesh,
                                      const struct hk_api_shader *task,
                                      const struct hk_api_shader *fs)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_api_shader *api_mesh = gfx->shaders[MESA_SHADER_MESH];
   const struct hk_api_shader *api_fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (!mesh)
      return false;
   if (direct && count_addr)
      return false;
   if (api_fs && api_fs != hk_cmd_buffer_device(cmd)->null_fs) {
      if (!fs || !fs->apple9_native_leaf || !mesh->apple9_native_leaf ||
          memcmp(fs->apple9_native_pair_id, mesh->apple9_native_pair_id,
                 sizeof(mesh->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = &mesh->variants[HK_MESH_VARIANT_MAIN];
   if (main->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH)
      return false;
   const bool has_task = !!gfx->shaders[MESA_SHADER_TASK];
   if (main->info.mesh.native_task_bridge && main->info.mesh.native_object)
      return false;
   if ((main->info.mesh.native_task_bridge || main->info.mesh.native_object) !=
       has_task)
      return false;
   if (!!gfx->render.view_mask != main->info.mesh.native_multiview ||
       (gfx->render.view_mask && !main->info.apple9_native.writes_layer_viewport))
      return false;
   /* Indirect, indirect-count, and predicated draws can evaluate to an empty
    * grid on the GPU and launch a safe (1, 1, 1) dummy group; require the
    * compiler-emitted empty-grid guard on the mesh shader binary. */
   if ((!direct || hk_cond_render_applies(cmd)) &&
       !main->info.apple9_native.guard_empty_grid)
      return false;
   if (direct) {
      if (!hk_apple9_native_mesh_grid_valid(direct))
         return false;
      uint64_t groups = (uint64_t)direct[0] * direct[1] * direct[2];
      if (gfx->shaders[MESA_SHADER_TASK]) {
         if (!task || !task->apple9_native_leaf || !mesh->apple9_native_leaf ||
             memcmp(task->apple9_native_pair_id, mesh->apple9_native_pair_id,
                    sizeof(mesh->apple9_native_pair_id)))
            return false;
         const struct hk_shader *t = &task->variants[0];
         const uint32_t max_groups =
            hk_apple9_native_mesh_max_groups(&main->info.apple9_native.layout);
         if (main->info.mesh.native_object) {
            struct hk_device *dev = hk_cmd_buffer_device(cmd);
            uint32_t payload_alloc_B =
               t->info.apple9_native.task_payload_stride_B +
               (t->info.apple9_native.writes_child_grid ? 16 : 0);
            uint32_t mesh_payload_alloc_B =
               main->info.apple9_native.task_payload_stride_B +
               (main->info.apple9_native.reads_child_grid ? 16 : 0);
            bool uses_payload = t->info.apple9_native.task_payload_size_B ||
                                t->info.apple9_native.writes_child_grid ||
                                main->info.apple9_native.reads_native_payload;
            if (!hk_apple9_native_object_available_device(dev) ||
                hk_cond_render_applies(cmd) ||
                t->info.apple9_native_stage != HK_APPLE9_NATIVE_OBJECT ||
                !max_groups ||
                groups > AGX_NATIVE_OBJECT_MAX_GROUPS ||
                t->info.apple9_native.key.physical_threads >
                   AGX_NATIVE_OBJECT_MAX_THREADS ||
                t->info.apple9_native.task_payload_size_B >
                   AGX_NATIVE_OBJECT_MAX_PAYLOAD_BYTES ||
                hk_apple9_native_object_tg_memory_bytes(
                   &t->info.apple9_native) >
                   AGX_NATIVE_OBJECT_MAX_SHARED_BYTES)
               return false;
            /* Per-parent hardware payload slots (GET_SR 0xe0/0xe1) require the
             * multi-parent payload aperture to cover the full allocation
             * (including the 12-byte child-grid tail). Both stages must agree
             * on the payload mechanism whenever payload is used. */
            if (uses_payload &&
                t->info.apple9_native.key.hw_payload !=
                   main->info.apple9_native.key.hw_payload)
               return false;
            if ((t->info.apple9_native.key.hw_payload ||
                 main->info.apple9_native.key.hw_payload) &&
                (payload_alloc_B > AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES ||
                 mesh_payload_alloc_B > AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES))
               return false;
            /* Multi-parent payload/child-grid traffic requires GET_SR 0xe0/0xe1;
             * single-parent payloads in (mp_payload_bytes, 16 KiB] continue to
             * use the draw-wide args.payload uniform isolated to groups == 1. */
            if (groups > 1 && uses_payload &&
                (!t->info.apple9_native.key.hw_payload ||
                 payload_alloc_B > AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES ||
                 mesh_payload_alloc_B > AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES))
               return false;
            if (t->info.task.max_mesh_groups == UINT32_MAX ||
                (t->info.task.max_mesh_groups &&
                 groups > max_groups / t->info.task.max_mesh_groups))
               return false;
         } else {
            if (!hk_apple9_native_task_bridge_enabled() || !max_groups ||
                groups * ALIGN_POT(t->info.task.payload_B, 16) > UINT32_MAX ||
                groups * 16 > UINT32_MAX)
               return false;
            if (t->info.task.max_mesh_groups == UINT32_MAX) {
               /* Dynamic EmitMeshTasksEXT counts cannot be bounded on the CPU:
                * execute the task stage once, route totals <= max_groups natively
                * (factored into 2D up to 2^20 - 1), and hand totals > max_groups
                * to the paired emulated mesh variant. Standalone native leaves
                * lack an emulated fallback, and task grids >= 1024 could legally
                * sum to >= 2^32 (1024 * maxMeshWorkGroupTotalCount) and wrap the
                * 32-bit prefix sum, so keep those off this path. */
               if (groups > UINT32_MAX / (1u << 22) ||
                   api_mesh->apple9_native_leaf ||
                   (api_fs && api_fs != hk_cmd_buffer_device(cmd)->null_fs &&
                    api_fs->apple9_native_leaf))
                  return false;
            } else {
               groups *= t->info.task.max_mesh_groups;
               if (groups > max_groups)
                  return false;
            }
         }
      } else if (groups * main->info.apple9_native.layout.max_primitives >
                    hk_apple9_native_render_primitives() ||
                 !agx_apple9_mesh_ring_fits(&main->info.apple9_native.layout,
                                            groups)) {
         /* A single native draw whose declared geometry could reach the TVB cap
          * would lose work in its mesh-mode render; emulate it instead
          * (ordinary renders recover through a partial render). Likewise fall
          * back when the draw's vertex planes can wrap the 64 KiB hardware
          * mesh output ring. */
         return false;
      }
   } else {
      /* Taskless indirect/indirect-count draws have unknown runtime group
       * counts on the CPU: run grids <= max_groups natively and hand oversized
       * legal grids (> max_groups, up to 4,194,304) to the paired emulated
       * fallback so no legal Vulkan draw or side effect is ever dropped.
       * Standalone native-only leaves lack emulated variants and stay direct. */
      if (gfx->shaders[MESA_SHADER_TASK] || api_mesh->apple9_native_leaf ||
          (api_fs && api_fs != hk_cmd_buffer_device(cmd)->null_fs &&
           api_fs->apple9_native_leaf) ||
          !hk_apple9_native_mesh_max_groups(&main->info.apple9_native.layout))
         return false;
   }
   return hk_apple9_native_mesh_state_eligible(
      cmd, direct, count_addr,
      (main->info.apple9_native.primitive_broadcast &
         VARYING_BIT_CULL_PRIMITIVE) ||
      main->info.apple9_native.cull_distances ||
      main->info.apple9_native.cull_primitive);
}

enum hk_apple9_native_mesh_route
hk_apple9_select_native_mesh_route(struct hk_cmd_buffer *cmd,
                                   const uint32_t *direct,
                                   uint64_t count_addr)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_api_shader *api_mesh = gfx->shaders[MESA_SHADER_MESH];
   const struct hk_api_shader *api_task = gfx->shaders[MESA_SHADER_TASK];
   const struct hk_api_shader *api_fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   const struct hk_api_shader *null_fs = hk_cmd_buffer_device(cmd)->null_fs;
   const struct hk_api_shader *fs_primary =
      (api_fs && api_fs != null_fs) ? hk_api_shader_native(api_fs) : api_fs;
   if (hk_apple9_native_mesh_leaves_eligible(
          cmd, direct, count_addr, hk_api_shader_native(api_mesh),
          hk_api_shader_native(api_task), fs_primary))
      return HK_APPLE9_MESH_ROUTE_NATIVE;

   const struct hk_api_shader *bridge_mesh = hk_api_shader_task_bridge(api_mesh);
   const struct hk_api_shader *bridge_task = hk_api_shader_task_bridge(api_task);
   if (api_task && bridge_mesh && bridge_task &&
       bridge_mesh->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge &&
       !bridge_mesh->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object) {
      const struct hk_api_shader *bridge_fs =
         (api_fs && api_fs != null_fs) ? hk_api_shader_task_bridge(api_fs)
                                       : api_fs;
      if (hk_apple9_native_mesh_leaves_eligible(cmd, direct, count_addr,
                                                bridge_mesh, bridge_task,
                                                bridge_fs))
         return HK_APPLE9_MESH_ROUTE_TASK_BRIDGE;
   }
   return HK_APPLE9_MESH_ROUTE_EMULATED;
}

bool
hk_apple9_use_native_mesh(struct hk_cmd_buffer *cmd, const uint32_t *direct,
                          uint64_t count_addr)
{
   return hk_apple9_select_native_mesh_route(cmd, direct, count_addr) !=
          HK_APPLE9_MESH_ROUTE_EMULATED;
}

/* Mesh shader draws run the task and mesh workgroups as compute kernels, and
 * then draw every primitive slot of the mesh workgroups' output records. The
 * workgroup grid is either given (direct) or read from grid_addr. For draws
 * from a count buffer, draws past the count at count_addr are empty.
 */
static void
hk_apple9_draw_mesh(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    const uint32_t *direct, uint64_t grid_addr,
                    uint64_t count_addr)
{
   /* Includes the hybrid native path: static pipeline feedback may change
    * after BeginRendering, and a failed decompression poisons recording. */
   if (!hk_feedback_prepare_draw(cmd))
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   if (direct && (!direct[0] || !direct[1] || !direct[2]))
      return;

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];
   struct hk_api_shader *task = gfx->shaders[MESA_SHADER_TASK];
   if (!mesh || (!direct && !grid_addr))
      return;

   enum hk_apple9_native_mesh_route route =
      hk_apple9_select_native_mesh_route(cmd, direct, count_addr);
   bool native = route != HK_APPLE9_MESH_ROUTE_EMULATED;
   bool use_bridge_leaf = route == HK_APPLE9_MESH_ROUTE_TASK_BRIDGE;
   bool predicated = hk_cond_render_applies(cmd);
   struct hk_api_shader *native_mesh_api =
      !native ? NULL
              : (use_bridge_leaf ? hk_api_shader_task_bridge(mesh)
                                 : hk_api_shader_native(mesh));
   struct hk_api_shader *native_task_api =
      (!native || !task) ? NULL
                         : (use_bridge_leaf ? hk_api_shader_task_bridge(task)
                                            : hk_api_shader_native(task));
   struct hk_shader *native_main =
      native_mesh_api ? &native_mesh_api->variants[HK_MESH_VARIANT_MAIN] : NULL;
   struct hk_shader *main = &mesh->variants[HK_MESH_VARIANT_MAIN];
   const struct hk_mesh_info *info = &main->info.mesh;
   const bool native_object =
      native && task && native_main->info.mesh.native_object;
   const bool task_dynamic =
      native && direct && task && !native_object &&
      hk_only_variant(native_task_api)->info.task.max_mesh_groups == UINT32_MAX;
   uint64_t draw_primitives = UINT64_MAX;
   uint64_t draw_ring_rows = UINT64_MAX;
   bool draw_ring_limited = true;
   if (native) {
      uint64_t native_groups = UINT64_MAX;
      if (direct && !task_dynamic) {
         native_groups =
            (uint64_t)direct[0] * direct[1] * direct[2] *
            (task ? hk_only_variant(native_task_api)->info.task.max_mesh_groups
                  : 1);
         draw_primitives =
            native_groups * native_main->info.apple9_native.layout.max_primitives;
      }
      if (!agx_apple9_mesh_ring_usage(&native_main->info.apple9_native.layout,
                                      native_groups, &draw_ring_rows,
                                      &draw_ring_limited)) {
         draw_ring_rows = UINT64_MAX;
         draw_ring_limited = true;
      }
   }
   if (debug_get_option_mesh_route_log()) {
      struct hk_shader *logged = native ? native_main : main;
      fprintf(stderr, "HK_MESH_ROUTE native=%u T=%u V=%u P=%u obj=%u bridge=%u\n",
              native,
              native ? logged->info.apple9_native.key.physical_threads : 0,
              logged->info.mesh.max_vertices, logged->info.mesh.max_primitives,
              native_object, native && task && !native_object);
   }
   struct hk_shader *mesh_rast =
      (!native || !direct || task_dynamic)
         ? hk_apple9_select_mesh_rast(cmd, mesh)
         : NULL;
   const bool rast_polygon_points =
      mesh_rast ? mesh_rast->info.mesh.polygon_points : info->polygon_points;
   unsigned vertices = MAX2(info->max_primitives, 1) *
                       (rast_polygon_points ? HK_PP_MAX_VERTICES
                                            : mesa_vertices_per_prim(info->prim));
   uint32_t overflow_min_groups = 0;
   uint64_t task_emu_params[32] = {0};
   if (native) {
      const uint32_t max_groups =
         hk_apple9_native_mesh_max_groups(&native_main->info.apple9_native.layout);
      hk_foreach_view(cmd) {
         cmd->apple9_draw_primitives = draw_primitives;
         cmd->apple9_draw_ring_rows = draw_ring_rows;
         cmd->apple9_draw_ring_limited = draw_ring_limited;
         if (!hk_apple9_prepare_draw_ex(cmd, true, true, native_object))
            return;
         if (task) {
            struct hk_shader *t = hk_only_variant(native_task_api);
            if (native_object) {
               uint64_t grid =
                  hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
               if (!grid)
                  return;
               uint64_t payload_gpu = 0;
               uint32_t payload_alloc_B =
                  t->info.apple9_native.task_payload_stride_B +
                  (t->info.apple9_native.writes_child_grid ? 16 : 0);
               if (payload_alloc_B && !t->info.apple9_native.key.hw_payload) {
                  payload_gpu = hk_pool_alloc(cmd, payload_alloc_B, 16).gpu;
                  if (!payload_gpu)
                     return;
               }
               hk_apple9_draw_native_object(cmd, draw_id, direct, grid,
                                            payload_gpu);
               continue;
            }
            uint32_t groups = (uint64_t)direct[0] * direct[1] * direct[2];
            struct agx_ptr params = hk_pool_alloc(cmd, sizeof(struct libagx_mesh_params), 16);
            if (!params.gpu)
               return;
            struct libagx_mesh_params *p = params.cpu;
            *p = (struct libagx_mesh_params){
               .task_count = groups,
               .payload_stride_B = ALIGN_POT(t->info.task.payload_B, 16),
            };
            /* Separate allocations for every draw/parent, owned by the command
             * buffer through native retirement. Native children never allocate
             * emulated vertex/primitive records or reuse another parent's payload. */
            if (p->payload_stride_B) {
               p->payloads = hk_pool_alloc(cmd, groups * p->payload_stride_B, 16).gpu;
               if (!p->payloads)
                  return;
            }
            p->task_grids = hk_pool_alloc(cmd, groups * 16, 16).gpu;
            if (!p->task_grids)
               return;
            uint64_t overflow_params_gpu = 0;
            if (task_dynamic) {
               struct agx_ptr emu_params =
                  hk_pool_alloc(cmd, sizeof(struct libagx_mesh_params), 16);
               if (!emu_params.gpu)
                  return;
               struct libagx_mesh_params *ep = emu_params.cpu;
               *ep = (struct libagx_mesh_params){
                  .payloads = p->payloads,
                  .task_grids = p->task_grids,
                  .task_count = groups,
                  .payload_stride_B = p->payload_stride_B,
               };
               for (unsigned v = 0; v < ARRAY_SIZE(ep->primitive_varying_B); ++v) {
                  ep->primitive_varying_B[v] = hk_mesh_slot_offset_B(
                     info, info->primitive_slots, VARYING_SLOT_VAR0 + v);
               }
               overflow_params_gpu = emu_params.gpu;
               task_emu_params[view_idx] = emu_params.gpu;
            }
            desc->root.draw.draw_id = draw_id;
            desc->root.draw.mesh_params = params.gpu;
            desc->root_dirty = true;
            hk_cmd_buffer_flush_push_descriptors(cmd, desc);
            gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
            if (!gfx->root)
               return;
            struct hk_apple9_prerast_bindings bindings = {0};
            hk_apple9_dispatch_prerast(cmd, t,
               agx_3d(direct[0] * t->info.apple9.local_size[0],
                      direct[1] * t->info.apple9.local_size[1],
                      direct[2] * t->info.apple9.local_size[2]), &bindings);
            libagx_mesh_prefix_native(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                                      params.gpu, max_groups, overflow_params_gpu);
            /* HK_READS_INTERNAL retains task/prefix -> tiling ordering. The
             * object consumes the GPU grid; the mesh consumes the same records. */
            hk_apple9_draw_native_task_bridge(
               cmd, draw_id,
               params.gpu + offsetof(struct libagx_mesh_params, mesh_dispatch),
               use_bridge_leaf);
            continue;
         }
         /* Taskless native object consumes a 3-word GPU grid directly. For
          * indirect/indirect-count or conditional rendering, pre_gfx validates
          * and predicates the grid (writing zeroes when skipped) and the
          * guarded object/mesh pair emits an empty mesh. */
         uint64_t grid;
         if (direct) {
            grid = hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
         } else {
            grid = hk_pool_alloc(cmd, 3 * sizeof(uint32_t), 4).gpu;
            if (grid) {
               libagx_mesh_indirect_native(cmd, agx_1d(1),
                                           AGX_BARRIER_ALL | AGX_PREGFX, grid,
                                           grid_addr, count_addr, draw_id,
                                           max_groups);
            }
         }
         if (!grid)
            return;
         if (predicated) {
            grid = hk_predicate_indirect(cmd, grid, 3, true);
            if (!grid)
               return;
         }
         if (!direct || predicated) {
            struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
            if (!cs)
               return;
            cs->pre_raster_reads |= HK_READS_INTERNAL;
         }
         hk_apple9_draw_native_mesh(cmd, draw_id, grid);
      }
      if (direct && !task_dynamic)
         return;
      /* Indirect/indirect-count grids exceeding max_groups (up to Vulkan's
       * 4,194,304 total workgroups) and dynamic task draws whose runtime sum
       * exceeded max_groups skipped native mesh work above; fall through to
       * the mutually exclusive emulated path so legal draws and side effects
       * are never dropped. */
      if (!task_dynamic)
         overflow_min_groups = max_groups + 1;
   }
   cmd->apple9_draw_primitives = UINT64_MAX;
   if (!hk_apple9_prepare_draw(cmd, !direct || count_addr || task_dynamic, false))
      return;
   if (task_dynamic) {
      struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return;
      cs->pre_raster_reads |= HK_READS_INTERNAL;
   }
   struct hk_shader *task_main = task ? hk_only_variant(task) : NULL;
   struct hk_apple9_prerast_bindings bindings = {0};

   /* Draws without a task shader and with a known grid are set up here.
    * Predicated draws read a grid of zeroes when skipped.
    */
   bool cpu_setup = direct && !task && !predicated;
   uint64_t direct_count =
      direct ? (uint64_t)direct[0] * direct[1] * direct[2] : 0;
   uint64_t records_B = direct_count * info->group_stride_B;
   if (cpu_setup && records_B > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   if (!task_dynamic) {
      if (direct && !cpu_setup)
         grid_addr = hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
      if (predicated)
         grid_addr = hk_predicate_indirect(cmd, grid_addr, 3, true);
      /* Upload or predication failed: the command buffer is in error. */
      if (!cpu_setup && !grid_addr)
         return;
   }

   desc->root.draw.draw_id = draw_id;

   /* Compute before graphics runs ahead of the whole render, so every
    * draw has its own records.
    */
   hk_foreach_view(cmd) {
      uint64_t params_gpu;
      if (task_dynamic) {
         params_gpu = task_emu_params[view_idx];
      } else {
         struct agx_ptr params =
            hk_pool_alloc(cmd, sizeof(struct libagx_mesh_params), 16);
         if (!params.gpu)
            return;

         struct libagx_mesh_params *p = params.cpu;
         *p = (struct libagx_mesh_params){
            .mesh_primitives = gfx->mesh_pg_query,
            .mesh_invocations = hk_mesh_stat(
               cmd, VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT),
            .task_invocations = hk_mesh_stat(
               cmd, VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT),
         };
         if (cpu_setup) {
            p->outputs = hk_pool_alloc(cmd, records_B, 16).gpu;
            p->mesh_count = direct_count;
         }
         for (unsigned v = 0; v < ARRAY_SIZE(p->primitive_varying_B); ++v) {
            p->primitive_varying_B[v] = hk_mesh_slot_offset_B(
               info, info->primitive_slots, VARYING_SLOT_VAR0 + v);
         }
         params_gpu = params.gpu;
      }

      desc->root.draw.mesh_params = params_gpu;
      desc->root_dirty = true;
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      if (!gfx->root)
         return;

      if (task_dynamic) {
         /* Allocate emulated output records inside this fallback render's own
          * mesh-heap epoch so earlier native render splits and subsequent
          * emulated draws cannot reset the heap underneath these records. */
         libagx_mesh_overflow_alloc(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                                    params_gpu, hk_mesh_heap(cmd),
                                    info->group_stride_B, vertices);
      } else {
         if (!cpu_setup) {
            libagx_mesh_setup(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                              params_gpu, grid_addr, count_addr, draw_id,
                              hk_mesh_heap(cmd), task != NULL,
                              task_main ? task_main->info.task.payload_B : 0,
                              info->group_stride_B, vertices,
                              overflow_min_groups);
         }

         if (task_main) {
            hk_apple9_dispatch_prerast(
               cmd, task_main,
               agx_grid_indirect(params_gpu +
                                 offsetof(struct libagx_mesh_params, task_dispatch)),
               &bindings);
            libagx_mesh_prefix(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                               params_gpu, hk_mesh_heap(cmd),
                               info->group_stride_B, vertices);
         }
      }

      struct agx_draw draw;
      if (cpu_setup) {
         hk_apple9_dispatch_prerast(
            cmd, main,
            agx_3d(direct[0] * main->info.apple9.local_size[0],
                   direct[1] * main->info.apple9.local_size[1],
                   direct[2] * main->info.apple9.local_size[2]),
            &bindings);
         draw = (struct agx_draw){.b = agx_3d(direct_count * vertices, 1, 1)};
      } else {
         hk_apple9_dispatch_prerast(
            cmd, main,
            agx_grid_indirect(params_gpu +
                              offsetof(struct libagx_mesh_params, mesh_dispatch)),
            &bindings);
         draw = agx_draw_indirect(params_gpu +
                                  offsetof(struct libagx_mesh_params, draw));
      }

      hk_apple9_draw(cmd, draw_id, draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                       uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};

   hk_apple9_draw_mesh(cmd, 0, groups, 0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, uint32_t drawCount,
                               uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   uint64_t base = drawCount ? hk_buffer_address_ro(buffer, offset) : 0;
   for (uint32_t i = 0; i < drawCount; ++i) {
      hk_apple9_draw_mesh(cmd, i, NULL, base + (uint64_t)i * stride, 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer,
                                    VkBuffer _buffer, VkDeviceSize offset,
                                    VkBuffer countBuffer,
                                    VkDeviceSize countBufferOffset,
                                    uint32_t maxDrawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   uint64_t count =
      maxDrawCount ? hk_buffer_address_ro(count_buffer, countBufferOffset) : 0;
   uint64_t base = maxDrawCount ? hk_buffer_address_ro(buffer, offset) : 0;
   for (uint32_t i = 0; i < maxDrawCount; ++i) {
      hk_apple9_draw_mesh(cmd, i, NULL, base + (uint64_t)i * stride, count);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw = {
      .b = agx_3d(vertexCount, instanceCount, 1),
      .start = firstVertex,
      .start_instance = firstInstance,
   };

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;
   struct agx_draw draw =
      agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                       firstInstance, buf.addr, buf.range, 0, 0);

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const uint32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw(cmd, draw_id, agx_draw_indirect(addr));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw(cmd, draw_id,
              agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");

   assert((stride % 4) == 0 && "aligned");

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, counter, counterBuffer);

   uint64_t indirect =
      hk_pool_alloc(cmd, sizeof(VkDrawIndirectCommand), 4).gpu;
   libagx_draw_xfb(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                   indirect, hk_buffer_address_ro(counter, counterBufferOffset),
                   counterOffset, vertexStride, instanceCount, firstInstance);
   hk_draw_indirect_inner(commandBuffer, indirect, 1, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   /* A null pCounterBuffers means no counter buffers. */
   for (unsigned i = 0; pCounterBuffers && i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, pConditionalRenderingBegin->buffer);
   hk_cmd_render_merge_flush(cmd);

   uint64_t predicate =
      hk_buffer_address(buffer, pConditionalRenderingBegin->offset, true);
   cmd->state.cond_render.predicate = predicate;
   cmd->state.cond_render.active = true;
   cmd->state.cond_render.inverted =
      pConditionalRenderingBegin->flags &
      VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_render_merge_flush(cmd);
   cmd->state.cond_render.active = false;
}

bool
hk_cond_render_applies(struct hk_cmd_buffer *cmd)
{
   return cmd->state.cond_render.active &&
          (!cmd->in_meta || cmd->state.cond_render.meta);
}

/* Returns indirect arguments of `words` 32-bit words that are a copy of
 * `in`, or zero when the conditional rendering predicate fails. */
uint64_t
hk_predicate_indirect(struct hk_cmd_buffer *cmd, uint64_t in, unsigned words,
                      bool graphics)
{
   /* A failed allocation has already put the command buffer in error, and it
    * will not be submitted: return 0 rather than predicating to address 0. */
   uint64_t out = in ? hk_pool_alloc(cmd, words * sizeof(uint32_t), 16).gpu : 0;
   if (!out)
      return 0;
   unsigned saved_scope = cmd->sync_scope;
   if (!graphics)
      cmd->sync_scope = HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT;
   libagx_predicate_indirect_apple9(
      cmd, agx_1d(1), AGX_BARRIER_ALL | (graphics ? AGX_PREGFX : 0), out, in,
      cmd->state.cond_render.predicate, cmd->state.cond_render.inverted, words);
   cmd->sync_scope = saved_scope;
   return out;
}

static struct agx_draw
hk_apple9_predicate_draw(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   unsigned words = draw.indexed ? 5 : 4;
   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1], draw.start,
                          draw.indexed ? (uint32_t)draw.index_bias
                                       : draw.start_instance,
                          draw.start_instance};
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }
   uint64_t out_draw = hk_predicate_indirect(cmd, in_draw, words, true);
   if (draw.indexed) {
      return agx_draw_indexed_indirect(out_draw, draw.index_buffer,
                                       draw.index_buffer_range_B,
                                       draw.index_size, draw.restart);
   } else {
      return agx_draw_indirect(out_draw);
   }
}
