/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "../hk_query_pool.h"
#include "hk_entrypoints.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct test_ctx {
   struct hk_device *dev;
   struct hk_cmd_buffer *cmd;
   uint8_t *pool;
};

static void
ctx_init(struct test_ctx *t)
{
   t->dev = calloc(1, sizeof(*t->dev));
   t->cmd = calloc(1, sizeof(*t->cmd));
   t->pool = calloc(1, HK_CMD_BO_SIZE);
   assert(t->dev && t->cmd && t->pool);

   t->dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&t->dev->vk, &t->cmd->vk.base,
                       VK_OBJECT_TYPE_COMMAND_BUFFER);
   t->cmd->sync_events = UTIL_DYNARRAY_INIT;
   t->cmd->sync_remap = UTIL_DYNARRAY_INIT;
   t->cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   t->cmd->empty_barriers = UTIL_DYNARRAY_INIT;
   t->cmd->occlusion_accum = UTIL_DYNARRAY_INIT;
   t->cmd->query_adds = UTIL_DYNARRAY_INIT;
   t->cmd->query_writes = UTIL_DYNARRAY_INIT;
   t->cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = t->pool;
   list_inithead(&t->cmd->control_streams);
}

static void
ctx_finish(struct test_ctx *t)
{
   list_for_each_entry_safe(struct hk_cs, cs, &t->cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   free(t->cmd->merge_empty_render);
   util_dynarray_fini(&t->cmd->sync_events);
   util_dynarray_fini(&t->cmd->sync_remap);
   util_dynarray_fini(&t->cmd->sync_first_event);
   util_dynarray_fini(&t->cmd->empty_barriers);
   util_dynarray_fini(&t->cmd->occlusion_accum);
   util_dynarray_fini(&t->cmd->query_adds);
   util_dynarray_fini(&t->cmd->query_writes);
   vk_object_base_finish(&t->cmd->vk.base);
   free(t->pool);
   free(t->cmd);
   free(t->dev);
}

static struct hk_cs *
add_empty_gfx(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *gfx = calloc(1, sizeof(*gfx));
   assert(gfx);
   gfx->cmd = cmd;
   gfx->type = HK_CS_VDM;
   gfx->sync_id = util_dynarray_num_elements(&cmd->sync_remap, uint32_t);
   util_dynarray_append(&cmd->sync_remap, gfx->sync_id);
   util_dynarray_append(&cmd->sync_first_event, (uint32_t)HK_SYNC_NONE);
   list_addtail(&gfx->node, &cmd->control_streams);
   cmd->current_cs.gfx = gfx;
   hk_cmd_sync_work(cmd, gfx, HK_SYNC_GRAPHICS);
   return gfx;
}

static void
init_fake_query_pool(struct hk_device *dev, struct hk_query_pool *qp,
                     struct agx_bo *bo, struct agx_va *va, uint32_t handle,
                     uint64_t addr)
{
   memset(qp, 0, sizeof(*qp));
   memset(bo, 0, sizeof(*bo));
   memset(va, 0, sizeof(*va));
   vk_object_base_init(&dev->vk, &qp->vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   va->addr = addr;
   bo->va = va;
   qp->vk.query_type = VK_QUERY_TYPE_TIMESTAMP;
   qp->vk.query_count = 4;
   qp->bo = bo;
   qp->query_start = 0;
   qp->query_stride = sizeof(uint64_t);
   qp->handle = handle;
}

static int
test_single_timestamp_empty_rp(void)
{
   struct test_ctx t;
   ctx_init(&t);

   struct hk_query_pool qp;
   struct agx_bo bo;
   struct agx_va va;
   init_fake_query_pool(t.dev, &qp, &bo, &va, 7, 0x1000000);

   struct hk_cs *gfx = add_empty_gfx(t.cmd);
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(t.cmd),
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         hk_query_pool_to_handle(&qp), 2);
   CHECK(gfx->timestamp.end.handle == 7);
   CHECK(gfx->timestamp.end.offset_B == 16);
   CHECK(gfx->timestamp.end.addr == 0x1000010);

   hk_cmd_buffer_end_graphics(t.cmd);
   CHECK(t.cmd->current_cs.gfx == NULL);
   CHECK(t.cmd->current_cs.cs == NULL);

   /* The empty VDM was replaced by a flushed CDM carrying the timestamp. */
   CHECK(!list_is_empty(&t.cmd->control_streams));
   CHECK(list_length(&t.cmd->control_streams) == 1);
   struct hk_cs *ts_cs =
      list_first_entry(&t.cmd->control_streams, struct hk_cs, node);
   CHECK(ts_cs->type == HK_CS_CDM);
   CHECK(ts_cs->timestamp.end.handle == 7);
   CHECK(ts_cs->timestamp.end.offset_B == 16);
   CHECK(ts_cs->timestamp.end.addr == 0x1000010);
   CHECK(ts_cs->stats.flushes > 0);
   CHECK(*util_dynarray_element(&t.cmd->sync_remap, uint32_t, 0) ==
         HK_SYNC_NONE);

   /* A following CDM stream (e.g., query copy) must not merge into ts_cs. */
   struct hk_cs *later = hk_cmd_buffer_get_cs_general(
      t.cmd, &t.cmd->current_cs.cs, true, HK_SYNC_ALL);
   CHECK(later && later != ts_cs);
   later->stats.cmds = 1;
   CHECK(hk_EndCommandBuffer(hk_cmd_buffer_to_handle(t.cmd)) == VK_SUCCESS);
   CHECK(list_length(&t.cmd->control_streams) == 2);
   struct hk_cs *first =
      list_first_entry(&t.cmd->control_streams, struct hk_cs, node);
   struct hk_cs *second =
      list_last_entry(&t.cmd->control_streams, struct hk_cs, node);
   CHECK(first == ts_cs && first->timestamp.end.handle == 7);
   CHECK(second == later && second->timestamp.end.handle == 0);

   vk_object_base_finish(&qp.vk.base);
   ctx_finish(&t);
   return 0;
}

static int
test_timestamp_with_post_gfx_copy_order(void)
{
   struct test_ctx t;
   ctx_init(&t);

   struct hk_query_pool qp;
   struct agx_bo bo;
   struct agx_va va;
   init_fake_query_pool(t.dev, &qp, &bo, &va, 9, 0x2000000);

   add_empty_gfx(t.cmd);
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(t.cmd),
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         hk_query_pool_to_handle(&qp), 1);

   /* Simulate a second vkCmdWriteTimestamp2 inside the same drawless RP
    * emitting libagx_copy_timestamp into post_gfx. */
   struct hk_cs *post = hk_cmd_buffer_get_cs_general(
      t.cmd, &t.cmd->current_cs.post_gfx, true, HK_SYNC_ALL);
   CHECK(post && post->sync_after_gfx && !t.cmd->sync_fallback);
   post->stats.cmds = 1;

   /* Simulate an active compute stream created after post_gfx (e.g. by
    * clear_attachment_as_image during empty-render conversion). */
   struct hk_cs *clears = hk_cmd_buffer_get_cs_general(
      t.cmd, &t.cmd->current_cs.cs, true, HK_SYNC_ALL);
   CHECK(clears && clears != post);
   clears->stats.cmds = 1;

   hk_cmd_buffer_end_graphics(t.cmd);
   CHECK(t.cmd->current_cs.gfx == NULL);
   CHECK(t.cmd->current_cs.cs == NULL);
   CHECK(t.cmd->current_cs.post_gfx == NULL);

   /* ts_cs (reusing clears) must execute strictly BEFORE post_gfx so the
    * firmware timestamp write lands before post_gfx copies it. */
   CHECK(list_length(&t.cmd->control_streams) == 2);
   struct hk_cs *first =
      list_first_entry(&t.cmd->control_streams, struct hk_cs, node);
   struct hk_cs *second =
      list_last_entry(&t.cmd->control_streams, struct hk_cs, node);
   CHECK(first == clears);
   CHECK(second == post);
   CHECK(first->timestamp.end.handle == 9);
   CHECK(first->timestamp.end.offset_B == 8);
   CHECK(first->stats.flushes > 0);

   /* Precise sync (no fallback poisoning) must order post_gfx after ts_cs. */
   unsigned count = util_dynarray_num_elements(&t.cmd->sync_remap, uint32_t);
   struct hk_sync_stream *streams = calloc(count, sizeof(*streams));
   CHECK(streams);
   uint32_t next[HK_SYNC_ENGINE_COUNT] = {0, 0};
   list_for_each_entry(struct hk_cs, cs, &t.cmd->control_streams, node) {
      struct hk_sync_stream *s = &streams[cs->sync_id];
      s->wait = s->wait_tiling = hk_sync_empty();
      memcpy(s->before, next, sizeof(next));
      s->engine = cs->type == HK_CS_CDM ? HK_SYNC_CDM : HK_SYNC_RENDER;
      s->index = ++next[s->engine];
   }
   struct hk_sync_state state;
   hk_sync_init(&state);
   CHECK(!t.cmd->sync_fallback);
   CHECK(hk_sync_resolve(&state, t.cmd->sync_events.data,
                         util_dynarray_num_elements(&t.cmd->sync_events,
                                                    struct hk_sync_event),
                         t.cmd->sync_remap.data, streams, count,
                         t.cmd->sync_fallback));
   CHECK(streams[post->sync_id].wait.engine[HK_SYNC_CDM] ==
         streams[first->sync_id].index);
   free(streams);

   /* EndCommandBuffer must keep ts_cs and post_gfx separate. */
   CHECK(hk_EndCommandBuffer(hk_cmd_buffer_to_handle(t.cmd)) == VK_SUCCESS);
   CHECK(list_length(&t.cmd->control_streams) == 2);

   vk_object_base_finish(&qp.vk.base);
   ctx_finish(&t);
   return 0;
}

static int
test_suspended_empty_rp_restore_preserves_timestamp(void)
{
   struct test_ctx t;
   ctx_init(&t);

   struct hk_query_pool qp;
   struct agx_bo bo;
   struct agx_va va;
   init_fake_query_pool(t.dev, &qp, &bo, &va, 13, 0x3000000);

   struct hk_cs *gfx = add_empty_gfx(t.cmd);
   hk_CmdWriteTimestamp2(hk_cmd_buffer_to_handle(t.cmd),
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         hk_query_pool_to_handle(&qp), 3);

   /* Exercise the merge_empty / restore path in hk_cmd_buffer_end_graphics. */
   t.cmd->merge_cs = gfx;
   t.cmd->merge_empty = true;
   t.cmd->merge_empty_render = calloc(1, sizeof(*t.cmd->merge_empty_render));
   CHECK(t.cmd->merge_empty_render);

   hk_cmd_buffer_end_graphics(t.cmd);
   CHECK(list_length(&t.cmd->control_streams) == 1);
   struct hk_cs *ts_cs =
      list_first_entry(&t.cmd->control_streams, struct hk_cs, node);
   CHECK(ts_cs->type == HK_CS_CDM);
   CHECK(ts_cs->timestamp.end.handle == 13);
   CHECK(ts_cs->timestamp.end.offset_B == 24);
   CHECK(ts_cs->stats.flushes > 0);

   vk_object_base_finish(&qp.vk.base);
   ctx_finish(&t);
   return 0;
}

int
main(void)
{
   if (test_single_timestamp_empty_rp())
      return 1;
   if (test_timestamp_with_post_gfx_copy_order())
      return 1;
   if (test_suspended_empty_rp_restore_preserves_timestamp())
      return 1;
   puts("PASS empty render pass timestamps preserved and ordered before post_gfx");
   return 0;
}
