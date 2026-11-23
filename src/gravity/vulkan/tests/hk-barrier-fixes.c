/* SPDX-License-Identifier: MIT */
#include "../hk_buffer.h"
#include "../hk_cmd_buffer.h"
#include "../hk_device.h"
#include "../hk_event.h"
#include "../hk_gpu_busy.h"
#include "../hk_image.h"
#include "../hk_image_view.h"
#include "../hk_query_pool.h"
#include "../hk_queue.h"
#include "hk_entrypoints.h"
#include "query.h"
#include "vk_drm_syncobj.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static unsigned precomp_calls;
static enum libagx_program precomp_last_idx;
static enum agx_barrier precomp_last_barrier;
static unsigned precomp_last_scope;
static VkPipelineStageFlags2 precomp_last_move_stage;

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                                enum agx_barrier barrier,
                                enum libagx_program idx, void *data,
                                size_t data_size);
void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                           enum agx_barrier barrier, enum libagx_program idx,
                           void *data, size_t data_size)
{
   precomp_calls++;
   precomp_last_idx = idx;
   precomp_last_barrier = barrier;
   precomp_last_scope = cmd->sync_scope ?: HK_SYNC_ALL;
   precomp_last_move_stage = cmd->merge_move_stage;

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, target, true, precomp_last_scope);
   if (cs)
      cs->stats.cmds++;
}

struct recorded_cmd {
   uint16_t cmd_type;
   uint16_t vdm_barrier;
   uint16_t cdm_barrier;
};

struct recorded_slice {
   uint32_t in_sync_count;
   uint32_t out_sync_count;
   uint32_t in_handle;
   uint32_t out_handle;
   unsigned cmd_count;
   struct recorded_cmd cmds[64];
};

static struct recorded_slice mock_slices[32];
static unsigned mock_slice_count;

static int
mock_submit(struct agx_device *dev, struct drm_asahi_submit *submit)
{
   assert(mock_slice_count < ARRAY_SIZE(mock_slices));
   struct recorded_slice *s = &mock_slices[mock_slice_count++];
   memset(s, 0, sizeof(*s));
   s->in_sync_count = submit->in_sync_count;
   s->out_sync_count = submit->out_sync_count;
   const struct drm_asahi_sync *syncs =
      (const struct drm_asahi_sync *)(uintptr_t)submit->syncs;
   if (s->in_sync_count)
      s->in_handle = syncs[0].handle;
   if (s->out_sync_count)
      s->out_handle = syncs[s->in_sync_count].handle;

   const uint8_t *buf = (const uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offs = 0;
   while (offs < submit->cmdbuf_size) {
      const struct drm_asahi_cmd_header *hdr = (const void *)(buf + offs);
      assert(s->cmd_count < ARRAY_SIZE(s->cmds));
      s->cmds[s->cmd_count++] = (struct recorded_cmd){
         .cmd_type = hdr->cmd_type,
         .vdm_barrier = hdr->vdm_barrier,
         .cdm_barrier = hdr->cdm_barrier,
      };
      offs += sizeof(*hdr) + hdr->size;
   }
   assert(offs == submit->cmdbuf_size);
   return 0;
}

static struct hk_cs *
add_test_cs(struct hk_cmd_buffer *cmd, enum hk_cs_type type, unsigned stages,
            uint8_t *buf, size_t size)
{
   struct hk_cs *cs = calloc(1, sizeof(*cs));
   assert(cs);
   cs->cmd = cmd;
   cs->type = type;
   cs->start = cs->current = buf;
   cs->end = buf + size;
   cs->addr = 0x1000000;
   cs->stats.cmds = 1;
   cs->cr.width = 64;
   cs->cr.height = 64;
   cs->cr.zls_width = 64;
   cs->cr.zls_height = 64;
   cs->cr.layers = 1;
   cs->tib.nr_samples = 1;
   cs->sync_id = util_dynarray_num_elements(&cmd->sync_remap, uint32_t);
   util_dynarray_append(&cmd->sync_remap, cs->sync_id);
   util_dynarray_append(&cmd->sync_first_event, (uint32_t)HK_SYNC_NONE);
   list_addtail(&cs->node, &cmd->control_streams);
   hk_cmd_sync_work(cmd, cs, stages);
   return cs;
}

/* Commit 1A: Hoisted barriers in hk_CmdPipelineBarrier2 must tag barrier_group
 * on the hoisted event(s) at events[merge_event - 1], not on the render event
 * at top_ptr. Two unrelated hoisted barriers in one VkDependencyInfo must not
 * transitively chain.
 */
static int
test_hoisted_barrier_group_tagging(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);

   uint8_t buf[256] = {0};
   struct hk_cs *comp = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   struct hk_cs *gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   cmd->current_cs.gfx = cmd->merge_cs = gfx;
   cmd->merge_event = 1;
   cmd->merge_reach = hk_merge_graphics_stages();

   /* Two barriers in ONE VkDependencyInfo, both unrelated to gfx (hoisted):
    *   B0: COMPUTE -> TRANSFER
    *   B1: TRANSFER -> COMPUTE
    * They must share barrier_group == 1 on events[1] and events[2], leaving
    * gfx's work event at events[3] with barrier_group == 0, and must NOT
    * transitively chain COMPUTE -> COMPUTE through TRANSFER.
    */
   VkMemoryBarrier2 mem_barriers[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      },
      {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      },
   };
   VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 2,
      .pMemoryBarriers = mem_barriers,
   };
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(cmd), &dep);

   CHECK(cmd->merge_cs == gfx);
   CHECK(cmd->merge_event == 3);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 4);

   const struct hk_sync_event *ev = cmd->sync_events.data;
   CHECK(ev[0].stream == comp->sync_id && ev[0].barrier_group == 0);
   CHECK(ev[1].stream == HK_SYNC_NONE && ev[1].barrier_group == 1);
   CHECK(ev[2].stream == HK_SYNC_NONE && ev[2].barrier_group == 1);
   CHECK(ev[3].stream == gfx->sync_id && ev[3].barrier_group == 0);

   /* Verify hk_sync_resolve does not chain B0 -> B1 onto a subsequent compute
    * stream (no transfer producer existed before B1).
    */
   struct hk_cs *comp2 = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   struct hk_sync_state state;
   hk_sync_init(&state);
   struct hk_sync_stream streams[3] = {
      {.engine = HK_SYNC_CDM, .index = 1, .before = {0, 0}},
      {.engine = HK_SYNC_RENDER, .index = 1, .before = {0, 1}},
      {.engine = HK_SYNC_CDM, .index = 2, .before = {1, 1}},
   };
   for (unsigned i = 0; i < 3; ++i)
      streams[i].wait = streams[i].wait_tiling = hk_sync_empty();

   CHECK(hk_sync_resolve(&state, cmd->sync_events.data,
                         util_dynarray_num_elements(&cmd->sync_events,
                                                    struct hk_sync_event),
                         cmd->sync_remap.data, streams, 3, false));
   CHECK(streams[comp2->sync_id].wait.engine[HK_SYNC_CDM] == HK_SYNC_NONE);

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   vk_object_base_finish(&cmd->vk.base);
   free(cmd);
   free(dev);
   return 0;
}

/* Commit 1C: Multi-slice mixed-engine submit must attach in_syncs on the first
 * slice touching VDM AND the first slice touching CDM so an independent
 * command on the second engine cannot start before wait semaphores signal.
 */
static int
test_mixed_engine_multislice_in_syncs(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_queue *queue = calloc(1, sizeof(*queue));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && queue && cmd);

   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.ops.submit = mock_submit;
   /* Force slice_limit = 1 so [VDM_0, VDM_1, CDM_0, CDM_1] becomes 4 slices. */
   dev->dev.params.max_commands_per_submission = 1;

   vk_object_base_init(&dev->vk, &queue->vk.base, VK_OBJECT_TYPE_QUEUE);
   hk_gpu_busy_init(queue);
   hk_sync_init(&queue->sync);
   queue->drm.id = 1;
   queue->drm.syncobj = 99;

   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);

   uint8_t buf[256] = {0};
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));

   struct vk_sync_type syncobj_type = {.finish = vk_drm_syncobj_finish};
   struct vk_drm_syncobj wait_syncobj = {
      .base = {.type = &syncobj_type},
      .syncobj = 42,
   };
   struct vk_drm_syncobj signal_syncobj = {
      .base = {.type = &syncobj_type},
      .syncobj = 77,
   };
   struct vk_sync_wait wait = {.sync = &wait_syncobj.base};
   struct vk_sync_signal signal = {.sync = &signal_syncobj.base};
   struct vk_command_buffer *cbs[] = {&cmd->vk};
   struct vk_queue_submit submit = {
      .wait_count = 1,
      .waits = &wait,
      .command_buffer_count = 1,
      .command_buffers = cbs,
      .signal_count = 1,
      .signals = &signal,
   };

   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 4);

   /* Slice 0: first VDM -> MUST have in_syncs, no out_syncs. */
   CHECK(mock_slices[0].cmd_count == 1);
   CHECK(mock_slices[0].cmds[0].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[0].in_sync_count == 1 && mock_slices[0].in_handle == 42);
   CHECK(mock_slices[0].out_sync_count == 0);

   /* Slice 1: second VDM -> VDM already waited in slice 0, no in_syncs needed. */
   CHECK(mock_slices[1].cmd_count == 1);
   CHECK(mock_slices[1].cmds[0].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[1].in_sync_count == 0);
   CHECK(mock_slices[1].out_sync_count == 0);

   /* Slice 2: first CDM (independent of VDM!) -> MUST have in_syncs! */
   CHECK(mock_slices[2].cmd_count == 1);
   CHECK(mock_slices[2].cmds[0].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[2].cmds[0].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[2].in_sync_count == 1 && mock_slices[2].in_handle == 42);
   CHECK(mock_slices[2].out_sync_count == 0);

   /* Slice 3: second CDM (last slice) -> no in_syncs, has out_syncs (signal + queue). */
   CHECK(mock_slices[3].cmd_count == 1);
   CHECK(mock_slices[3].cmds[0].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[3].in_sync_count == 0);
   CHECK(mock_slices[3].out_sync_count == 2 && mock_slices[3].out_handle == 77);

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   vk_object_base_finish(&cmd->vk.base);
   hk_gpu_busy_finish(queue);
   vk_object_base_finish(&queue->vk.base);
   free(cmd);
   free(queue);
   free(dev);
   return 0;
}

/* Commit 1B: Query flush paths must ensure CDM control stream space before
 * emitting hk_cdm_cache_flush(), chaining a new chunk when near the end.
 */
static int
test_query_flush_space_check(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->occlusion_accum = UTIL_DYNARRAY_INIT;
   cmd->query_adds = UTIL_DYNARRAY_INIT;
   cmd->query_writes = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   uint8_t small_chunk[64] = {0};
   struct hk_cs *cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE,
                                  small_chunk, sizeof(small_chunk));
   /* Position current so only the link tag fits: without hk_ensure_cs_has_space
    * hk_cdm_cache_flush would overflow or trip its space assertion.
    */
   cs->current = cs->end - AGX_CDM_STREAM_LINK_LENGTH;
   cmd->current_cs.cs = cs;

   struct libagx_imm_write write = {.address = 0x1000, .value = 1};
   util_dynarray_append(&cmd->query_writes, write);
   hk_flush_query_writes(cmd);

   CHECK(cs->stream_linked);
   CHECK(cs->stats.flushes == 1);
   CHECK(cs->current <= cs->end);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   util_dynarray_fini(&cmd->occlusion_accum);
   util_dynarray_fini(&cmd->query_adds);
   util_dynarray_fini(&cmd->query_writes);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

/* Commit 2: Multi-command submit batching enabled unconditionally up to the
 * kernel limit, preserving exact intra-submit barrier indices, avoiding
 * unnecessary cross-engine joins, and slicing cleanly with rebasing +
 * mixed-engine in_syncs.
 */
static int
test_batched_submit_and_intra_barriers(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_queue *queue = calloc(1, sizeof(*queue));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && queue && cmd);

   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.ops.submit = mock_submit;
   dev->dev.params.max_commands_per_submission = 64;

   vk_object_base_init(&dev->vk, &queue->vk.base, VK_OBJECT_TYPE_QUEUE);
   hk_gpu_busy_init(queue);
   hk_sync_init(&queue->sync);
   queue->drm.id = 1;
   queue->drm.syncobj = 99;

   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);

   uint8_t buf[256] = {0};
   /* Case A: Single-render mixed-engine batch
    * [CDM_1, VDM_1 (waits CDM_1), CDM_2 (waits VDM_1), CDM_3 (indep)]
    * executes in one ioctl with exact intra-submit barriers and no forced join.
    */
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_TRANSFER, buf, sizeof(buf));

   struct vk_sync_type syncobj_type = {.finish = vk_drm_syncobj_finish};
   struct vk_drm_syncobj wait_syncobj = {
      .base = {.type = &syncobj_type},
      .syncobj = 42,
   };
   struct vk_drm_syncobj signal_syncobj = {
      .base = {.type = &syncobj_type},
      .syncobj = 77,
   };
   struct vk_sync_wait wait = {.sync = &wait_syncobj.base};
   struct vk_sync_signal signal = {.sync = &signal_syncobj.base};
   struct vk_command_buffer *cbs[] = {&cmd->vk};
   struct vk_queue_submit submit = {
      .wait_count = 1,
      .waits = &wait,
      .command_buffer_count = 1,
      .command_buffers = cbs,
      .signal_count = 1,
      .signals = &signal,
   };

   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 1);
   CHECK(mock_slices[0].cmd_count == 4);
   CHECK(mock_slices[0].in_sync_count == 1 && mock_slices[0].out_sync_count == 2);
   /* CDM_1: no waits. */
   CHECK(mock_slices[0].cmds[0].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[0].cmds[0].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[0].cmds[0].cdm_barrier == DRM_ASAHI_BARRIER_NONE);
   /* VDM_1: waits on intra-submit CDM #1. */
   CHECK(mock_slices[0].cmds[1].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[0].cmds[1].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[0].cmds[1].cdm_barrier == 1);
   /* CDM_2: waits on intra-submit VDM #1 (and transitively CDM #1). */
   CHECK(mock_slices[0].cmds[2].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[0].cmds[2].vdm_barrier == 1);
   CHECK(mock_slices[0].cmds[2].cdm_barrier == 1);
   /* CDM_3: independent, no forced cross-engine join since both engines present. */
   CHECK(mock_slices[0].cmds[3].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[0].cmds[3].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[0].cmds[3].cdm_barrier == DRM_ASAHI_BARRIER_NONE);

   /* Case B: Multi-render mixed-engine batch
    * [VDM_1, CDM_1 (waits VDM_1), VDM_2 (waits CDM_1), CDM_2 (indep)]
    * caps each ioctl to 1 render (avoiding kernel shared_notifier=1 multi-render
    * flush-boundary bugs) while batching each render with its adjacent CDMs.
    */
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);
   hk_sync_init(&queue->sync);

   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_TRANSFER, buf, sizeof(buf));

   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 2);
   /* Slice 0: [VDM_1, CDM_1] covers both engines -> receives in_syncs. */
   CHECK(mock_slices[0].cmd_count == 2);
   CHECK(mock_slices[0].in_sync_count == 1 && mock_slices[0].out_sync_count == 0);
   CHECK(mock_slices[0].cmds[0].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[0].cmds[0].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[0].cmds[1].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[0].cmds[1].vdm_barrier == 1);
   /* Slice 1: [VDM_2, CDM_2] rebases cross-slice waits to 0; both engines
    * already saw in_syncs in Slice 0.
    */
   CHECK(mock_slices[1].cmd_count == 2);
   CHECK(mock_slices[1].in_sync_count == 0 && mock_slices[1].out_sync_count == 2);
   CHECK(mock_slices[1].cmds[0].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[1].cmds[0].vdm_barrier == 0);
   CHECK(mock_slices[1].cmds[0].cdm_barrier == 0);
   CHECK(mock_slices[1].cmds[1].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[1].cmds[1].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[1].cmds[1].cdm_barrier == DRM_ASAHI_BARRIER_NONE);

   /* Case C: Kernel limit 0 safely clamps to 1 command per ioctl slice. */
   hk_sync_init(&queue->sync);
   dev->dev.params.max_commands_per_submission = 0;
   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 4);
   CHECK(mock_slices[0].in_sync_count == 1);
   CHECK(mock_slices[1].in_sync_count == 1); /* first CDM slice */
   CHECK(mock_slices[2].in_sync_count == 0);
   CHECK(mock_slices[3].in_sync_count == 0);
   CHECK(mock_slices[3].out_sync_count == 2);
   dev->dev.params.max_commands_per_submission = 64;

   /* Case D: Multi-slice batch (limit=2) with
    * [VDM_1, VDM_2, CDM_1 (waits VDM_2), CDM_2 (waits CDM_1), CDM_3 (indep)].
    * VDM_1 runs alone (VDM_2 starts slice 1); [VDM_2, CDM_1] share slice 1
    * (receiving in_syncs as the first CDM slice); [CDM_2, CDM_3] share slice 2
    * with rebased cross-slice barriers and no redundant VDM join on CDM_3.
    */
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);
   hk_sync_init(&queue->sync);
   dev->dev.params.max_commands_per_submission = 2;

   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_TRANSFER, buf, sizeof(buf));

   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 3);
   /* Slice 0: [VDM_1] with in_syncs. */
   CHECK(mock_slices[0].cmd_count == 1);
   CHECK(mock_slices[0].in_sync_count == 1 && mock_slices[0].out_sync_count == 0);
   /* Slice 1: [VDM_2, CDM_1] receives in_syncs (first CDM slice); CDM_1 gets
    * intra-slice vdm_barrier=1 pointing to VDM_2.
    */
   CHECK(mock_slices[1].cmd_count == 2);
   CHECK(mock_slices[1].in_sync_count == 1 && mock_slices[1].out_sync_count == 0);
   CHECK(mock_slices[1].cmds[0].cmd_type == DRM_ASAHI_CMD_RENDER);
   CHECK(mock_slices[1].cmds[1].cmd_type == DRM_ASAHI_CMD_COMPUTE);
   CHECK(mock_slices[1].cmds[1].vdm_barrier == 1);
   /* Slice 2: [CDM_2, CDM_3] gets out_syncs only. CDM_2 rebases VDM and CDM
    * waits to 0; CDM_3 does NOT need a redundant vdm_barrier=0 since CDM_2
    * already waited on 0 in this slice.
    */
   CHECK(mock_slices[2].cmd_count == 2);
   CHECK(mock_slices[2].in_sync_count == 0 && mock_slices[2].out_sync_count == 2);
   CHECK(mock_slices[2].cmds[0].vdm_barrier == 0);
   CHECK(mock_slices[2].cmds[0].cdm_barrier == 0);
   CHECK(mock_slices[2].cmds[1].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[2].cmds[1].cdm_barrier == DRM_ASAHI_BARRIER_NONE);

   /* Case E: Single-engine multi-command final slice [CDM_1, CDM_2] with no
    * VDM barrier joins VDM only on the LAST command (CDM_2), not CDM_1.
    */
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);
   hk_sync_init(&queue->sync);
   dev->dev.params.max_commands_per_submission = 64;

   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, buf, sizeof(buf));
   add_test_cs(cmd, HK_CS_CDM, HK_SYNC_TRANSFER, buf, sizeof(buf));
   mock_slice_count = 0;
   CHECK(hk_queue_submit(&queue->vk, &submit) == VK_SUCCESS);
   CHECK(mock_slice_count == 1 && mock_slices[0].cmd_count == 2);
   CHECK(mock_slices[0].cmds[0].vdm_barrier == DRM_ASAHI_BARRIER_NONE);
   CHECK(mock_slices[0].cmds[1].vdm_barrier == 0);

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   vk_object_base_finish(&cmd->vk.base);
   hk_gpu_busy_finish(queue);
   vk_object_base_finish(&queue->vk.base);
   free(cmd);
   free(queue);
   free(dev);
   return 0;
}

/* Commit 3A: 2nd timestamp inside a render pass (post_gfx) must not poison
 * cmd->sync_fallback, and graphics-stage in-pass timestamps must not widen
 * the render to HK_SYNC_ALL.
 */
static int
test_sync_fallback_non_poisoning(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   struct agx_va va = {.addr = 0x1000000};
   struct agx_bo bo = {.va = &va};
   struct hk_query_pool qp = {
      .bo = &bo,
      .query_start = 0,
      .query_stride = sizeof(uint64_t),
      .handle = 5,
   };
   vk_object_base_init(&dev->vk, &qp.vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   qp.vk.query_type = VK_QUERY_TYPE_TIMESTAMP;
   qp.vk.query_count = 4;

   uint8_t gfx_buf[256] = {0};
   struct hk_cs *gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = gfx;

   /* 1st in-pass timestamp with graphics-only stage: must NOT tag HK_SYNC_ALL. */
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         hk_query_pool_to_handle(&qp), 0);
   CHECK(gfx->timestamp.end.handle == 5);
   CHECK(gfx->sync_stages == HK_SYNC_GRAPHICS);
   CHECK(!cmd->sync_fallback);

   /* 2nd in-pass timestamp spawns post_gfx via libagx_copy_timestamp: must NOT
    * poison cmd->sync_fallback, and later draws in gfx must not trigger fallback.
    */
   precomp_calls = 0;
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         hk_query_pool_to_handle(&qp), 1);
   CHECK(precomp_calls == 1 && precomp_last_idx == LIBAGX_COPY_TIMESTAMP);
   CHECK(cmd->current_cs.post_gfx != NULL);
   CHECK(cmd->current_cs.post_gfx->sync_after_gfx);
   CHECK(!cmd->sync_fallback);

   /* Simulate a subsequent draw in the same render pass. */
   hk_cmd_sync_work(cmd, gfx, HK_SYNC_GRAPHICS);
   struct hk_cs *post = cmd->current_cs.post_gfx;

   hk_cmd_buffer_end_graphics(cmd);
   CHECK(!cmd->sync_fallback);
   CHECK(cmd->current_cs.gfx == NULL && cmd->current_cs.post_gfx == NULL);
   CHECK(list_length(&cmd->control_streams) == 2);
   CHECK(list_first_entry(&cmd->control_streams, struct hk_cs, node) == gfx);
   CHECK(list_last_entry(&cmd->control_streams, struct hk_cs, node) == post);

   /* Resolve sync precisely (no fallback): post_gfx waits on gfx. */
   struct hk_sync_state state;
   hk_sync_init(&state);
   struct hk_sync_stream streams[2] = {
      {.engine = HK_SYNC_RENDER, .index = 1, .before = {0, 0}},
      {.engine = HK_SYNC_CDM, .index = 1, .before = {1, 0}},
   };
   streams[0].wait = streams[0].wait_tiling = hk_sync_empty();
   streams[1].wait = streams[1].wait_tiling = hk_sync_empty();
   CHECK(hk_sync_resolve(&state, cmd->sync_events.data,
                         util_dynarray_num_elements(&cmd->sync_events,
                                                    struct hk_sync_event),
                         cmd->sync_remap.data, streams, 2, cmd->sync_fallback));
   CHECK(streams[post->sync_id].wait.engine[HK_SYNC_RENDER] == 1);

   /* Subsequent pure compute-only barrier still stays intra-CS (not poisoned). */
   uint8_t cdm_buf[4096] = {0};
   struct hk_cs *cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, cdm_buf, sizeof(cdm_buf));
   cs->cdm_dirty = true;
   cmd->current_cs.cs = cs;
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   CHECK(cmd->current_cs.cs == cs);
   CHECK(*(uint32_t *)cdm_buf == 0x60000160);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   vk_object_base_finish(&qp.vk.base);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL
mock_get_buffer_device_address(VkDevice device,
                               const VkBufferDeviceAddressInfo *pInfo)
{
   VK_FROM_HANDLE(hk_buffer, buf, pInfo->buffer);
   return buf->vk.device_address;
}

/* Commit 3B: Narrowed HK_SYNC_ALL leaks (clears, copies, query copy, empty VDM,
 * HOST_BIT) and merge_move_stage preserving kept-open merge renders.
 */
static int
test_narrowed_scopes_and_merge_move(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->empty_barriers = UTIL_DYNARRAY_INIT;
   cmd->occlusion_accum = UTIL_DYNARRAY_INIT;
   cmd->query_adds = UTIL_DYNARRAY_INIT;
   cmd->query_writes = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   /* Set up a kept-open merge render to test merge_move_stage on buffer ops
    * and image clears. */
   uint8_t gfx_buf[256] = {0};
   struct hk_cs *gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = cmd->merge_cs = gfx;
   cmd->merge_event = 0;
   cmd->merge_reach = hk_merge_graphics_stages();

   struct agx_va buf_va = {.addr = 0x3000000, .size_B = 256};
   struct hk_buffer buf_obj = {.va = &buf_va};
   vk_object_base_init(&dev->vk, &buf_obj.vk.base, VK_OBJECT_TYPE_BUFFER);
   buf_obj.vk.size = 256;
   buf_obj.vk.device_address = 0x3000000;
   dev->vk.dispatch_table.GetBufferDeviceAddress =
      mock_get_buffer_device_address;

   /* vkCmdCopyBuffer2: HK_SYNC_TRANSFER + merge_move_stage keeps merge_cs open! */
   VkBufferCopy2 region = {.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .size = 64};
   VkCopyBufferInfo2 copy_info = {
      .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .srcBuffer = hk_buffer_to_handle(&buf_obj),
      .dstBuffer = hk_buffer_to_handle(&buf_obj),
      .regionCount = 1,
      .pRegions = &region,
   };
   precomp_calls = 0;
   hk_CmdCopyBuffer2(hk_cmd_buffer_to_handle(cmd), &copy_info);
   CHECK(precomp_calls == 1 && precomp_last_scope == HK_SYNC_TRANSFER);
   CHECK(precomp_last_move_stage & VK_PIPELINE_STAGE_2_COPY_BIT);
   CHECK(cmd->merge_cs == gfx && cmd->merge_pre_cs != NULL);

   /* vkCmdFillBuffer: HK_SYNC_TRANSFER + merge_move_stage keeps merge_cs open! */
   precomp_calls = 0;
   hk_CmdFillBuffer(hk_cmd_buffer_to_handle(cmd), hk_buffer_to_handle(&buf_obj),
                    0, 64, 0x12345678);
   CHECK(precomp_calls == 1 && precomp_last_scope == HK_SYNC_TRANSFER);
   CHECK(precomp_last_move_stage & VK_PIPELINE_STAGE_2_CLEAR_BIT);
   CHECK(cmd->merge_cs == gfx);

   /* vkCmdUpdateBuffer: HK_SYNC_TRANSFER + merge_move_stage keeps merge_cs open! */
   uint32_t data[4] = {1, 2, 3, 4};
   precomp_calls = 0;
   hk_CmdUpdateBuffer(hk_cmd_buffer_to_handle(cmd), hk_buffer_to_handle(&buf_obj),
                      0, sizeof(data), data);
   CHECK(precomp_calls == 1 && precomp_last_scope == HK_SYNC_TRANSFER);
   CHECK(precomp_last_move_stage & VK_PIPELINE_STAGE_2_COPY_BIT);
   CHECK(cmd->merge_cs == gfx);

   /* vkCmdClearColorImage: HK_SYNC_TRANSFER + merge_move_stage keeps merge_cs open! */
   struct hk_image img = {0};
   vk_object_base_init(&dev->vk, &img.vk.base, VK_OBJECT_TYPE_IMAGE);
   img.vk.image_type = VK_IMAGE_TYPE_2D;
   img.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   img.vk.extent = (VkExtent3D){16, 16, 1};
   img.vk.mip_levels = 1;
   img.vk.array_layers = 1;
   img.vk.samples = VK_SAMPLE_COUNT_1_BIT;
   img.plane_count = 1;
   img.planes[0].addr = 0x4000000;
   img.planes[0].layout.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   img.planes[0].layout.levels = 1;
   img.planes[0].layout.depth_px = 1;
   img.planes[0].layout.layer_stride_B = 1024;
   img.planes[0].layout.size_B = 1024;
   img.planes[0].layout.level_offsets_B[0] = 0;
   img.planes[0].layout.level_offsets_B[1] = 1024;

   VkClearColorValue clear_color = {.float32 = {1, 0, 0, 1}};
   VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
   };
   precomp_calls = 0;
   hk_CmdClearColorImage(hk_cmd_buffer_to_handle(cmd), hk_image_to_handle(&img),
                         VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &range);
   CHECK(precomp_calls == 1 && precomp_last_scope == HK_SYNC_TRANSFER);
   CHECK(precomp_last_move_stage & VK_PIPELINE_STAGE_2_CLEAR_BIT);
   CHECK(cmd->merge_cs == gfx);

   /* Flush merge render before testing vkCmdCopyQueryPoolResults. */
   hk_cmd_render_merge_flush(cmd);
   CHECK(cmd->merge_cs == NULL);

   /* vkCmdCopyQueryPoolResults: waits on timestamp/query producers (incoming
    * ALL -> TRANSFER barrier), while the copy itself is HK_SYNC_TRANSFER (not
    * HK_SYNC_ALL) so later independent graphics is not stalled.
    */
   struct agx_va qp_va = {.addr = 0x5000000};
   struct agx_bo qp_bo = {.va = &qp_va};
   struct hk_query_pool qp = {
      .bo = &qp_bo,
      .query_start = 0,
      .query_stride = sizeof(uint64_t),
      .handle = 11,
   };
   vk_object_base_init(&dev->vk, &qp.vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   qp.vk.query_type = VK_QUERY_TYPE_TIMESTAMP;
   qp.vk.query_count = 2;

   precomp_calls = 0;
   hk_CmdCopyQueryPoolResults(hk_cmd_buffer_to_handle(cmd),
                              hk_query_pool_to_handle(&qp), 0, 1,
                              hk_buffer_to_handle(&buf_obj), 0, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   CHECK(precomp_calls == 1 && precomp_last_idx == LIBAGX_COPY_QUERY);
   CHECK(precomp_last_scope == HK_SYNC_TRANSFER);
   /* Event before the copy dispatch is the incoming ALL -> TRANSFER barrier. */
   unsigned ev_count =
      util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
   CHECK(ev_count >= 2);
   const struct hk_sync_event *events = cmd->sync_events.data;
   CHECK(events[ev_count - 2].stream == HK_SYNC_NONE);
   CHECK(events[ev_count - 2].src == HK_SYNC_ALL);
   CHECK(events[ev_count - 2].dst == HK_SYNC_TRANSFER);
   CHECK(events[ev_count - 1].stream != HK_SYNC_NONE);
   CHECK(events[ev_count - 1].dst == HK_SYNC_TRANSFER);
   hk_cmd_buffer_end_compute(cmd);

   /* COMPUTE -> HOST_BIT write barrier emits full 0x600fffff flush (for host
    * visibility) without splitting CDM or stalling subsequent renders.
    */
   uint8_t cdm_buf[4096] = {0};
   struct hk_cs *cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, cdm_buf, sizeof(cdm_buf));
   cs->cdm_dirty = true;
   cmd->current_cs.cs = cs;
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_HOST_READ_BIT);
   CHECK(cmd->current_cs.cs == cs);
   CHECK(!cs->cdm_dirty && !cs->cdm_compute_barrier);
   CHECK(*(uint32_t *)cdm_buf != 0x60000160 && *(uint32_t *)cdm_buf != 0);
   const struct hk_sync_event *host_ev =
      util_dynarray_top_ptr(&cmd->sync_events, struct hk_sync_event);
   CHECK(host_ev->stream == HK_SYNC_NONE && host_ev->dst == 0);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   util_dynarray_fini(&cmd->empty_barriers);
   util_dynarray_fini(&cmd->occlusion_accum);
   util_dynarray_fini(&cmd->query_adds);
   util_dynarray_fini(&cmd->query_writes);
   vk_object_base_finish(&qp.vk.base);
   vk_object_base_finish(&img.vk.base);
   vk_object_base_finish(&buf_obj.vk.base);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

/* Commit 3C: In-pass barriers (attachment-only and execution-only/WAR
 * PRE_RASTER -> FRAGMENT keep gfx open; VS write, input attachment, or
 * feedback loop still split) + compute CS preserved across independent
 * renders while guarded against cross-engine dependencies.
 */
static int
test_in_pass_barriers_and_compute_preservation(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   uint8_t gfx_buf[256] = {0}, cdm_buf[4096] = {0};
   struct hk_cs *gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = gfx;

   /* 1. In-pass attachment-only barrier keeps gfx open. */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   CHECK(cmd->current_cs.gfx == gfx);

   /* 2. In-pass execution-only/WAR PRE_RASTER -> FRAGMENT keeps gfx open. */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
   CHECK(cmd->current_cs.gfx == gfx);

   /* 3. In-pass PRE_RASTER -> FRAGMENT with VS write access MUST split gfx. */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   CHECK(cmd->current_cs.gfx == NULL);

   /* 4. In-pass COLOR_ATTACHMENT -> FRAGMENT with feedback loop MUST split. */
   gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = gfx;
   cmd->state.gfx.render.feedback = true;
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 0);
   CHECK(cmd->current_cs.gfx == NULL);
   cmd->state.gfx.render.feedback = false;

   /* 5. Compute stream at tail vs preceding a render in control_streams:
    *    - Tail cs on pure COMPUTE -> COMPUTE barrier stays open with 0x60000160.
    *    - Behind-other cs closes on COMPUTE -> COMPUTE and GRAPHICS -> COMPUTE
    *      barriers so later compute work cannot be reordered ahead of newer
    *      streams.
    */
   struct hk_cs *cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, cdm_buf, sizeof(cdm_buf));
   cs->cdm_dirty = true;
   cmd->current_cs.cs = cs;
   CHECK(!hk_cs_is_behind_other(cmd, cs));

   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);
   CHECK(cmd->current_cs.cs == cs);
   CHECK(*(uint32_t *)cdm_buf == 0x60000160);

   /* Once a render follows cs in control_streams, any ordered barrier closes cs. */
   gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   CHECK(hk_cs_is_behind_other(cmd, cs));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);
   CHECK(cmd->current_cs.cs == NULL);

   /* Also verify GRAPHICS -> COMPUTE closes a behind-other TRANSFER stream. */
   cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_TRANSFER, cdm_buf, sizeof(cdm_buf));
   cmd->current_cs.cs = cs;
   gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   CHECK(hk_cs_is_behind_other(cmd, cs));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   CHECK(cmd->current_cs.cs == NULL);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

/* Commit 3D: Narrowed VkEvent handling (DEVICE_ONLY skips CDM write + merge
 * flush; host-visible SetEvent2 respects stage mask; WaitEvents2 replays
 * VkDependencyInfo without forcing ALL_COMMANDS).
 */
static int
test_narrowed_events(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->empty_barriers = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   uint8_t gfx_buf[256] = {0}, cdm_buf[4096] = {0};
   struct hk_cs *gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = cmd->merge_cs = gfx;
   cmd->merge_event = 0;
   cmd->merge_reach = hk_merge_graphics_stages();

   struct hk_event dev_ev = {.device_only = true};
   vk_object_base_init(&dev->vk, &dev_ev.base, VK_OBJECT_TYPE_EVENT);

   VkMemoryBarrier2 comp_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
   };
   VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &comp_barrier,
   };

   /* DEVICE_ONLY SetEvent2 / ResetEvent2: 0 dispatches, merge_cs stays open! */
   precomp_calls = 0;
   hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&dev_ev), &dep);
   hk_CmdResetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&dev_ev),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
   CHECK(precomp_calls == 0);
   CHECK(cmd->merge_cs == gfx);

   /* WaitEvents2 with COMPUTE -> COMPUTE: keeps CDM unsplit with 0x60000160
    * and does not flush unrelated open gfx!
    */
   cmd->merge_cs = NULL;
   struct hk_cs *cs = add_test_cs(cmd, HK_CS_CDM, HK_SYNC_COMPUTE, cdm_buf, sizeof(cdm_buf));
   cs->cdm_dirty = true;
   cmd->current_cs.cs = cs;
   VkEvent ev_handle = hk_event_to_handle(&dev_ev);
   hk_CmdWaitEvents2(hk_cmd_buffer_to_handle(cmd), 1, &ev_handle, &dep);
   CHECK(cmd->current_cs.cs == cs);
   CHECK(cmd->current_cs.gfx == gfx);
   CHECK(*(uint32_t *)cdm_buf == 0x60000160);

   /* Host-visible SetEvent2(COMPUTE): uses HK_SYNC_COMPUTE, not HK_SYNC_ALL. */
   VkResult host_status = VK_EVENT_RESET;
   struct hk_event host_ev = {
      .device_only = false,
      .addr = 0x6000000,
      .status = &host_status,
   };
   vk_object_base_init(&dev->vk, &host_ev.base, VK_OBJECT_TYPE_EVENT);
   precomp_calls = 0;
   hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&host_ev), &dep);
   CHECK(precomp_calls == 1 && precomp_last_idx == LIBAGX_WRITE_U32);
   CHECK(precomp_last_scope == HK_SYNC_COMPUTE);
   CHECK(cmd->current_cs.gfx == gfx);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   util_dynarray_fini(&cmd->empty_barriers);
   vk_object_base_finish(&host_ev.base);
   vk_object_base_finish(&dev_ev.base);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

/* Sync-ordering edge-case regression tests:
 *  - Open CS1 + merge_pre_cs (D1 -> R open -> D2(MPC) -> D3(indirect) ->
 *    COMPUTE->COMPUTE barrier -> D4) must close CS1 when MPC is created so
 *    D3/D4 run strictly after MPC in list and sync order; behind-other
 *    COMPUTE->COMPUTE barriers and graphics-scoped CDM clears also close CS1.
 *  - Consecutive compute timestamps D1 -> TS1(COMPUTE) -> D2 -> TS2(COMPUTE)
 *    must preserve the implicit execution dependency CS2 -> CS1.
 *  - Compute-only PIPELINE_STATISTICS EndQuery must order its availability
 *    write after an already-closed compute producer stream.
 *  - In-pass TOP_OF_PIPE / NONE timestamps (stage_mask == 0) must retain
 *    HK_SYNC_ALL instead of narrowing to HK_SYNC_GRAPHICS.
 */
static int
test_sync_ordering_edge_cases(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);

   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->empty_barriers = UTIL_DYNARRAY_INIT;
   cmd->occlusion_accum = UTIL_DYNARRAY_INIT;
   cmd->query_adds = UTIL_DYNARRAY_INIT;
   cmd->query_writes = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x2000000000ull;
   list_inithead(&cmd->control_streams);

   /* --- B1: D1 -> CS1; R open for merge; D2 -> MPC; D3 flushes merge;
    *         COMPUTE->COMPUTE barrier; D4 must be listed AFTER MPC!
    */
   struct hk_cs *cs1 = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   CHECK(cs1 != NULL);

   uint8_t gfx_buf[256] = {0};
   struct hk_cs *r = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = r;
   cmd->merge_cs = r;
   cmd->merge_event = 1;
   cmd->merge_reach = 0;

   /* D2 (movable compute) routes to merge_pre_cs and MUST close CS1. */
   cmd->merge_move_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
   struct hk_cs *mpc = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   cmd->merge_move_stage = 0;
   CHECK(mpc != NULL && mpc == cmd->merge_pre_cs && mpc != cs1);
   CHECK(cmd->current_cs.cs == NULL);

   /* D3 (non-movable, e.g. DispatchIndirect) flushes the merge (R + MPC close)
    * and opens a fresh CS2 at the tail after R.
    */
   struct hk_cs *cs2 = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT);
   CHECK(cs2 != NULL && cs2 != cs1 && cs2 != mpc);
   CHECK(cmd->merge_cs == NULL && cmd->merge_pre_cs == NULL);

   /* COMPUTE_WRITE -> COMPUTE_READ barrier + D4 stay in CS2, strictly AFTER MPC. */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   struct hk_cs *d4_cs = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   CHECK(d4_cs == cs2);

   /* Verify physical list order is [CS1, MPC, R, CS2] (D4 strictly after D2!). */
   struct hk_cs *order[4] = {0};
   unsigned n = 0;
   list_for_each_entry(struct hk_cs, iter, &cmd->control_streams, node) {
      CHECK(n < 4);
      order[n++] = iter;
   }
   CHECK(n == 4 && order[0] == cs1 && order[1] == mpc && order[2] == r &&
         order[3] == cs2);

   /* Verify hk_sync_resolve succeeds with no fallback and orders CS2 after MPC. */
   struct hk_sync_state st;
   hk_sync_init(&st);
   struct hk_sync_stream b1_streams[4] = {0};
   b1_streams[cs1->sync_id] = (struct hk_sync_stream){
      .engine = HK_SYNC_CDM, .index = 1, .before = {0, 0},
      .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty(),
   };
   b1_streams[mpc->sync_id] = (struct hk_sync_stream){
      .engine = HK_SYNC_CDM, .index = 2, .before = {0, 1},
      .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty(),
   };
   b1_streams[r->sync_id] = (struct hk_sync_stream){
      .engine = HK_SYNC_RENDER, .index = 1, .before = {0, 2},
      .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty(),
   };
   b1_streams[cs2->sync_id] = (struct hk_sync_stream){
      .engine = HK_SYNC_CDM, .index = 3, .before = {1, 2},
      .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty(),
   };
   /* Verify hk_sync_resolve succeeds with no fallback (no backward wait).
    * MPC (CDM #2) was closed and flushed before CS2 (CDM #3) in FIFO CDM
    * subqueue order, and the COMPUTE->COMPUTE barrier after D3 is satisfied
    * intra-stream inside CS2 (so CS2 does not wait on itself).
    */
   CHECK(hk_sync_resolve(&st, cmd->sync_events.data,
                         util_dynarray_num_elements(&cmd->sync_events,
                                                    struct hk_sync_event),
                         cmd->sync_remap.data, b1_streams, 4, false));
   CHECK(b1_streams[cs2->sync_id].wait.engine[HK_SYNC_CDM] == HK_SYNC_NONE);

   /* --- B1b + L3: CS1 behind R closes on COMPUTE->COMPUTE barrier AND on
    *               graphics-scoped CDM work (HK_SYNC_FRAGMENT).
    */
   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   cmd->current_cs.cs = cmd->current_cs.gfx = NULL;
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);

   struct hk_cs *behind = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   CHECK(hk_cs_is_behind_other(cmd, behind));
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT);
   CHECK(cmd->current_cs.cs == NULL);

   /* L3: graphics-scoped CDM clear closes CS1 sitting behind a render. */
   struct hk_cs *behind2 = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   CHECK(hk_cs_is_behind_other(cmd, behind2));
   struct hk_cs *frag_cdm = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_FRAGMENT);
   CHECK(frag_cdm != NULL && frag_cdm != behind2);
   CHECK(list_last_entry(&cmd->control_streams, struct hk_cs, node) == frag_cdm);

   /* --- H1: D1 -> TS1(COMPUTE) -> D2 -> TS2(COMPUTE) orders CS2 after CS1. */
   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   cmd->current_cs.cs = cmd->current_cs.gfx = NULL;
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);

   struct agx_va qp_va = {.addr = 0x5000000};
   struct agx_bo qp_bo = {.va = &qp_va};
   struct hk_query_pool ts_qp = {
      .bo = &qp_bo,
      .query_start = 0,
      .query_stride = sizeof(uint64_t),
      .handle = 11,
   };
   vk_object_base_init(&dev->vk, &ts_qp.vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   ts_qp.vk.query_type = VK_QUERY_TYPE_TIMESTAMP;
   ts_qp.vk.query_count = 4;

   struct hk_cs *ts_cs1 = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true, HK_SYNC_COMPUTE);
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         hk_query_pool_to_handle(&ts_qp), 0);
   CHECK(cmd->current_cs.cs == ts_cs1 && ts_cs1->timestamp.end.handle == 11);
   /* D2 into ts_cs1, then TS2 splits into ts_cs2. */
   hk_cmd_sync_work(cmd, ts_cs1, HK_SYNC_COMPUTE);
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         hk_query_pool_to_handle(&ts_qp), 1);
   struct hk_cs *ts_cs2 = cmd->current_cs.cs;
   CHECK(ts_cs2 != NULL && ts_cs2 != ts_cs1 && ts_cs2->timestamp.end.handle == 11);
   CHECK(ts_cs2->sync_stages == HK_SYNC_COMPUTE);

   hk_sync_init(&st);
   struct hk_sync_stream h1_streams[2] = {
      {.engine = HK_SYNC_CDM, .index = 1, .before = {0, 0},
       .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty()},
      {.engine = HK_SYNC_CDM, .index = 2, .before = {0, 1},
       .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty()},
   };
   CHECK(hk_sync_resolve(&st, cmd->sync_events.data,
                         util_dynarray_num_elements(&cmd->sync_events,
                                                    struct hk_sync_event),
                         cmd->sync_remap.data, h1_streams, 2, false));
   CHECK(h1_streams[ts_cs2->sync_id].wait.engine[HK_SYNC_CDM] == 1);

   /* --- M1: Compute-only PIPELINE_STATISTICS EndQuery orders availability
    *         write after an already-closed compute producer stream.
    */
   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   cmd->current_cs.cs = cmd->current_cs.gfx = NULL;
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);

   struct hk_query_pool stat_qp = {
      .bo = &qp_bo,
      .query_start = 64,
      .query_stride = 64,
      .handle = 12,
   };
   vk_object_base_init(&dev->vk, &stat_qp.vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   stat_qp.vk.query_type = VK_QUERY_TYPE_PIPELINE_STATISTICS;
   stat_qp.vk.pipeline_statistics =
      VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
   stat_qp.vk.query_count = 2;

   struct hk_cs *stat_prod = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.cs, true,
      HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT);
   /* Close producer stream before EndQuery. */
   hk_cmd_buffer_end_compute(cmd);
   CHECK(cmd->current_cs.cs == NULL);

   hk_CmdEndQueryIndexedEXT(hk_cmd_buffer_to_handle(cmd),
                            hk_query_pool_to_handle(&stat_qp), 0, 0);
   struct hk_cs *stat_avail = cmd->current_cs.cs;
   CHECK(stat_avail != NULL && stat_avail != stat_prod);
   CHECK(stat_avail->sync_stages == HK_SYNC_COMPUTE);

   hk_sync_init(&st);
   struct hk_sync_stream m1_streams[2] = {
      {.engine = HK_SYNC_CDM, .index = 1, .before = {0, 0},
       .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty()},
      {.engine = HK_SYNC_CDM, .index = 2, .before = {0, 1},
       .wait = hk_sync_empty(), .wait_tiling = hk_sync_empty()},
   };
   CHECK(hk_sync_resolve(&st, cmd->sync_events.data,
                         util_dynarray_num_elements(&cmd->sync_events,
                                                    struct hk_sync_event),
                         cmd->sync_remap.data, m1_streams, 2, false));
   CHECK(m1_streams[stat_avail->sync_id].wait.engine[HK_SYNC_CDM] == 1);

   /* --- L2: In-pass TOP_OF_PIPE / NONE timestamp keeps HK_SYNC_ALL. */
   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   cmd->current_cs.cs = cmd->current_cs.gfx = NULL;
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
   util_dynarray_clear(&cmd->sync_first_event);

   struct hk_cs *top_gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = top_gfx;
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         hk_query_pool_to_handle(&ts_qp), 0);
   CHECK(top_gfx->sync_stages == HK_SYNC_ALL);

   struct hk_cs *none_gfx = add_test_cs(cmd, HK_CS_VDM, HK_SYNC_GRAPHICS, gfx_buf, sizeof(gfx_buf));
   cmd->current_cs.gfx = none_gfx;
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(cmd),
                         VK_PIPELINE_STAGE_2_NONE,
                         hk_query_pool_to_handle(&ts_qp), 1);
   CHECK(none_gfx->sync_stages == HK_SYNC_ALL);

   list_for_each_entry_safe(struct hk_cs, iter, &cmd->control_streams, node) {
      list_del(&iter->node);
      hk_cs_destroy(iter);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   util_dynarray_fini(&cmd->empty_barriers);
   util_dynarray_fini(&cmd->occlusion_accum);
   util_dynarray_fini(&cmd->query_adds);
   util_dynarray_fini(&cmd->query_writes);
   vk_object_base_finish(&stat_qp.vk.base);
   vk_object_base_finish(&ts_qp.vk.base);
   vk_object_base_finish(&cmd->vk.base);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

int
main(void)
{
   if (test_hoisted_barrier_group_tagging())
      return 1;
   if (test_query_flush_space_check())
      return 1;
   if (test_mixed_engine_multislice_in_syncs())
      return 1;
   if (test_batched_submit_and_intra_barriers())
      return 1;
   if (test_sync_fallback_non_poisoning())
      return 1;
   if (test_narrowed_scopes_and_merge_move())
      return 1;
   if (test_in_pass_barriers_and_compute_preservation())
      return 1;
   if (test_narrowed_events())
      return 1;
   if (test_sync_ordering_edge_cases())
      return 1;
   puts("PASS barrier-vk Commits 1-3: all barrier, batching, sync_fallback, scope, in-pass, event, and sync-review checks");
   return 0;
}
