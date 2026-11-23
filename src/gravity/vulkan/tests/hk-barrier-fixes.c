/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "../hk_device.h"
#include "../hk_gpu_busy.h"
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

int
main(void)
{
   if (test_hoisted_barrier_group_tagging())
      return 1;
   if (test_query_flush_space_check())
      return 1;
   if (test_mixed_engine_multislice_in_syncs())
      return 1;
   puts("PASS barrier-vk Commit 1: hoisted barrier_group, query flush space, mixed-engine in_syncs");
   return 0;
}
