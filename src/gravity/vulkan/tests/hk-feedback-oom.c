/* SPDX-License-Identifier: MIT */
/* Exercise the real BeginRendering/draw/end paths after a decompression
 * allocation failure. No GPU or live Vulkan device is needed. */
#include "../hk_cmd_draw.c"
#include "../hk_feedback.h"
#include "gravity/lib/agx_apple9.h"
#include "gravity/lib/agx_nir_apple9_tilebuffer.h"

struct test_apple9_tile_helper {
   struct agx_apple9_tilebuffer_key key;
   struct agx_bo *bo;
   struct agx_shader_info info;
   struct agx_apple9_stage_launch_cache launch;
};

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
static unsigned attempts;
static bool fail_decompress = true;
static unsigned fail_after_calls;
static bool clear_rendering_on_preserve;
struct decomp_call {
   const struct hk_image_view *view;
   unsigned layer;
   bool discard_contents;
};
static struct decomp_call calls[8];
static unsigned call_count;

bool __wrap_hk_meta_decompress_view(struct hk_cmd_buffer *,
                                    const struct hk_image_view *, unsigned,
                                    bool);
bool
__wrap_hk_meta_decompress_view(struct hk_cmd_buffer *cmd,
                               const struct hk_image_view *view, unsigned layer,
                               bool discard_contents)
{
   attempts++;
   if (call_count < ARRAY_SIZE(calls))
      calls[call_count++] = (struct decomp_call){view, layer, discard_contents};
   if (clear_rendering_on_preserve && !discard_contents)
      cmd->rendering = false;
   if (fail_decompress || (fail_after_calls && attempts >= fail_after_calls)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   return true;
}

VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int, const char *, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

static struct hk_cmd_buffer *
new_cmd(struct hk_device *dev, uint8_t *pool, uint8_t *usc_pool)
{
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(cmd);
   cmd->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   if (pool) {
      cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
      cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
      cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE / 2;
      cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base = (uintptr_t)(pool + HK_CMD_BO_SIZE / 2);
   }
   if (usc_pool) {
      cmd->uploader[HK_CMD_POOL_USC].chunks[0].map = usc_pool;
      cmd->uploader[HK_CMD_POOL_USC].chunks[0].base = 0x1000;
   }
   list_inithead(&cmd->control_streams);
   return cmd;
}

static void
free_cmd(struct hk_cmd_buffer *cmd)
{
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->feedback_ops);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(cmd);
}

static uint32_t
tile_helper_key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct agx_apple9_tilebuffer_key));
}

static bool
tile_helper_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct agx_apple9_tilebuffer_key)) == 0;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   uint8_t *usc_pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && pool && usc_pool);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   list_inithead(&dev->dev.apple9_entry_lru);
   simple_mtx_init(&dev->apple9_tile_helpers.lock, mtx_plain);
   dev->apple9_tile_helpers.ht = _mesa_hash_table_create(
      NULL, tile_helper_key_hash, tile_helper_key_equal);
   CHECK(dev->apple9_tile_helpers.ht);

   /* Seed two dummy tile helpers (mask=0 no-store vs mask=1 store) so we can
    * verify hk_apple9_build_bg_eot honors no_store_color_mask even when
    * storeOp == VK_ATTACHMENT_STORE_OP_STORE, without needing a live GPU/BO.
    */
   struct agx_va dummy_va = {.addr = 0x80000000, .size_B = 4096};
   struct agx_bo dummy_bo = {.va = &dummy_va, .size = 4096};
   dummy_bo.apple9_body.entry_offset = 0x2000;
   list_addtail(&dummy_bo.apple9_body.entry_link, &dev->dev.apple9_entry_lru);
   struct test_apple9_tile_helper h_nostore = {0}, h_store = {0},
                                  h_empty_bg = {0}, h_empty_eot = {0};
   h_nostore.key.count = h_store.key.count = 1;
   h_nostore.key.samples = h_store.key.samples = 1;
   h_nostore.key.store = h_store.key.store = true;
   h_nostore.key.formats[0] = h_store.key.formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   h_nostore.key.mask = 0;
   h_store.key.mask = 1;
   h_empty_bg.key.samples = h_empty_eot.key.samples = 1;
   h_empty_bg.key.store = false;
   h_empty_eot.key.store = true;
   h_nostore.bo = h_store.bo = h_empty_bg.bo = h_empty_eot.bo = &dummy_bo;
   h_nostore.info.binary_size = h_store.info.binary_size = 16;
   h_empty_bg.info.binary_size = h_empty_eot.info.binary_size = 16;
   h_nostore.info.apple9_image_mask = 0;
   h_store.info.apple9_image_mask = 1;
   agx_apple9_stage_launch_cache_init(&h_nostore.launch);
   agx_apple9_stage_launch_cache_init(&h_store.launch);
   agx_apple9_stage_launch_cache_init(&h_empty_bg.launch);
   agx_apple9_stage_launch_cache_init(&h_empty_eot.launch);
   _mesa_hash_table_insert(dev->apple9_tile_helpers.ht, &h_nostore.key, &h_nostore);
   _mesa_hash_table_insert(dev->apple9_tile_helpers.ht, &h_store.key, &h_store);
   _mesa_hash_table_insert(dev->apple9_tile_helpers.ht, &h_empty_bg.key, &h_empty_bg);
   _mesa_hash_table_insert(dev->apple9_tile_helpers.ht, &h_empty_eot.key, &h_empty_eot);

   struct hk_image image = {.vk = {.base = {.type = VK_OBJECT_TYPE_IMAGE,
                                            .device = &dev->vk},
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .image_type = VK_IMAGE_TYPE_2D,
                                   .aspects = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .extent = {64, 64, 1},
                                   .mip_levels = 1,
                                   .array_layers = 1,
                                   .samples = 1}};
   list_inithead(&image.feedback_raw_link);
   image.plane_count = 1;
   image.planes[0].layout = (struct ail_layout){.compressed = true,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width_px = 64, .height_px = 64, .depth_px = 1, .levels = 1,
      .sample_count_sa = 1, .compression_layer_stride_B = 256};
   image.planes[0].layout.tilesize_el[0] = (struct ail_tile){16, 16};

   struct hk_image image2 = image;
   list_inithead(&image2.feedback_raw_link);

   struct hk_image zs_image = {.vk = {.base = {.type = VK_OBJECT_TYPE_IMAGE,
                                               .device = &dev->vk},
                                      .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
                                      .image_type = VK_IMAGE_TYPE_2D,
                                      .aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
                                                 VK_IMAGE_ASPECT_STENCIL_BIT,
                                      .extent = {64, 64, 1},
                                      .mip_levels = 1,
                                      .array_layers = 1,
                                      .samples = 1}};
   list_inithead(&zs_image.feedback_raw_link);
   zs_image.plane_count = 2;
   zs_image.planes[0].layout = (struct ail_layout){.compressed = true,
      .tiling = AIL_TILING_TWIDDLED,
      .format = PIPE_FORMAT_Z32_FLOAT,
      .width_px = 64, .height_px = 64, .depth_px = 1, .levels = 1,
      .sample_count_sa = 1, .layer_stride_B = AIL_PAGESIZE,
      .compression_layer_stride_B = 256};
   zs_image.planes[0].layout.tilesize_el[0] = (struct ail_tile){16, 16};
   zs_image.planes[1].layout = (struct ail_layout){.compressed = true,
      .tiling = AIL_TILING_TWIDDLED,
      .format = PIPE_FORMAT_S8_UINT,
      .width_px = 64, .height_px = 64, .depth_px = 1, .levels = 1,
      .sample_count_sa = 1, .layer_stride_B = AIL_PAGESIZE,
      .compression_layer_stride_B = 128};
   zs_image.planes[1].layout.tilesize_el[0] = (struct ail_tile){16, 16};

   struct hk_image_view view = {.vk = {.base = {.type = VK_OBJECT_TYPE_IMAGE_VIEW,
                                                .device = &dev->vk},
                                       .image = &image.vk,
                                       .format = VK_FORMAT_R8G8B8A8_UNORM,
                                       .layer_count = 1}};
   view.apple9_render[0].store_valid = true;
   view.apple9_render[0].texture_valid = true;
   struct hk_image_view view2 = {.vk = {.base = {.type = VK_OBJECT_TYPE_IMAGE_VIEW,
                                                 .device = &dev->vk},
                                        .image = &image2.vk,
                                        .format = VK_FORMAT_R8G8B8A8_UNORM,
                                        .layer_count = 1}};
   struct hk_image_view zs_view = {.vk = {.base = {.type = VK_OBJECT_TYPE_IMAGE_VIEW,
                                                   .device = &dev->vk},
                                          .image = &zs_image.vk,
                                          .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
                                          .layer_count = 1}};

   /* 1. OOM on preserving (discard_contents=false) path. */
   struct hk_cmd_buffer *cmd = new_cmd(dev, NULL, NULL);
   const VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&view),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo info = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &att};
   VkCommandBuffer handle = hk_cmd_buffer_to_handle(cmd);
   hk_CmdBeginRendering(handle, &info);
   CHECK(attempts == 1 && call_count == 1 && !calls[0].discard_contents);
   CHECK(vk_command_buffer_has_error(&cmd->vk));
   CHECK(!cmd->rendering && !cmd->current_cs.gfx);

   /* Draw entry points must not inspect absent shaders/render state or try
    * decompression again once recording has failed, including native mesh. */
   cmd->current_cs.gfx = (void *)(uintptr_t)1;
   hk_CmdDraw(handle, 3, 1, 0, 0);
   hk_CmdDrawIndexed(handle, 3, 1, 0, 0, 0);
   hk_CmdDrawMeshTasksEXT(handle, 1, 1, 1);
   hk_CmdEndRendering2KHR(handle, NULL);
   hk_CmdBeginRendering(handle, &info);
   CHECK(attempts == 1 && cmd->current_cs.gfx == (void *)(uintptr_t)1);
   CHECK(hk_EndCommandBuffer(handle) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   cmd->current_cs.gfx = NULL;
   free_cmd(cmd);

   /* 2. OOM on metadata-only (discard_contents=true) path (full-area CLEAR). */
   cmd = new_cmd(dev, NULL, NULL);
   handle = hk_cmd_buffer_to_handle(cmd);
   attempts = call_count = 0;
   const VkRenderingAttachmentInfo att_clear = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&view),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo info_clear = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &att_clear};
   hk_CmdBeginRendering(handle, &info_clear);
   CHECK(attempts == 1 && call_count == 1 && calls[0].discard_contents);
   CHECK(vk_command_buffer_has_error(&cmd->vk));
   CHECK(!cmd->rendering && !cmd->current_cs.gfx);
   CHECK(hk_EndCommandBuffer(handle) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   free_cmd(cmd);

   /* 3. Mid-render split across 2 MRTs + D32S8 (all full-area CLEAR) must pass
    * discard_contents=false for EVERY attachment even when the first preserving
    * meta self-copy clears cmd->rendering = false!
    */
   cmd = new_cmd(dev, NULL, NULL);
   fail_decompress = false;
   fail_after_calls = 3; /* fail on 3rd (ZS) after checking its discard flag */
   clear_rendering_on_preserve = true;
   attempts = call_count = 0;
   cmd->rendering = true;
   cmd->state.gfx.render.area = (VkRect2D){{0, 0}, {64, 64}};
   cmd->state.gfx.render.cr.width = 64;
   cmd->state.gfx.render.cr.height = 64;
   cmd->state.gfx.render.cr.layers = 1;
   cmd->state.gfx.render.layer_count = 1;
   cmd->state.gfx.render.color_att_count = 2;
   cmd->state.gfx.render.color_att[0] = (struct hk_attachment){
      .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
      .iview = &view,
      .image_layout = VK_IMAGE_LAYOUT_GENERAL,
      .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .store_op = VK_ATTACHMENT_STORE_OP_STORE,
   };
   cmd->state.gfx.render.color_att[1] = (struct hk_attachment){
      .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
      .iview = &view2,
      .image_layout = VK_IMAGE_LAYOUT_GENERAL,
      .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .store_op = VK_ATTACHMENT_STORE_OP_STORE,
   };
   cmd->state.gfx.render.depth_att = (struct hk_attachment){
      .vk_format = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .iview = &zs_view,
      .image_layout = VK_IMAGE_LAYOUT_GENERAL,
      .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .store_op = VK_ATTACHMENT_STORE_OP_STORE,
   };
   cmd->state.gfx.render.stencil_att = cmd->state.gfx.render.depth_att;
   cmd->vk.dynamic_graphics_state.feedback_loops =
      VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
      VK_IMAGE_ASPECT_STENCIL_BIT;
   CHECK(!hk_feedback_prepare_draw(cmd));
   CHECK(call_count == 3);
   CHECK(calls[0].view == &view && !calls[0].discard_contents);
   CHECK(calls[1].view == &view2 && !calls[1].discard_contents);
   CHECK(calls[2].view == &zs_view && !calls[2].discard_contents);
   clear_rendering_on_preserve = false;
   fail_after_calls = 0;
   fail_decompress = true;
   free_cmd(cmd);

   /* 4. D32S8 discard eligibility via hk_CmdBeginRendering:
    * - Single-aspect CLEAR (depth=CLEAR, stencil=LOAD) -> discard_contents=false
    * - Both-aspect CLEAR, full area -> discard_contents=true
    * - Both-aspect CLEAR, partial area -> discard_contents=false
    * - Both-aspect CLEAR, RESUMING -> discard_contents=false
    */
   const VkRenderingAttachmentInfo ds_clear = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingAttachmentInfo ds_load = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   VkRenderingInfo ds_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &ds_clear,
      .pStencilAttachment = &ds_load,
   };
   cmd = new_cmd(dev, NULL, NULL);
   attempts = call_count = 0;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &ds_info);
   CHECK(call_count == 1 && !calls[0].discard_contents);
   free_cmd(cmd);

   ds_info.pStencilAttachment = &ds_clear;
   cmd = new_cmd(dev, NULL, NULL);
   attempts = call_count = 0;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &ds_info);
   CHECK(call_count == 1 && calls[0].discard_contents);
   free_cmd(cmd);

   ds_info.renderArea.extent.width = 32;
   cmd = new_cmd(dev, NULL, NULL);
   attempts = call_count = 0;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &ds_info);
   CHECK(call_count == 1 && !calls[0].discard_contents);
   free_cmd(cmd);

   ds_info.renderArea.extent.width = 64;
   ds_info.flags = VK_RENDERING_RESUMING_BIT;
   cmd = new_cmd(dev, NULL, NULL);
   attempts = call_count = 0;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &ds_info);
   CHECK(call_count == 1 && !calls[0].discard_contents);
   free_cmd(cmd);

   /* 5. READ_ONLY_OPTIMAL + STORE_OP_STORE: hk_apple9_build_bg_eot MUST select
    * the mask=0 (no-store) helper for both main and partial EOT, while clearing
    * no_store_color_mask selects the mask=1 (store) helper.
    */
   cmd = new_cmd(dev, pool, usc_pool);
   const VkRenderingAttachmentInfo ro_color = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&view),
      .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingAttachmentInfo ro_ds = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo ro_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ro_color,
      .pDepthAttachment = &ro_ds,
      .pStencilAttachment = &ro_ds,
   };
   cmd->state.gfx.render.color_att_count = 1;
   cmd->state.gfx.render.color_att[0].iview = &view;
   cmd->state.gfx.render.tilebuffer.nr_samples = 1;
   cmd->state.gfx.render.cr.layers = 1;
   cmd->state.gfx.render.no_store_color_mask = 1;
   struct hk_bg_eot eot_main_ro =
      hk_apple9_build_bg_eot(cmd, &ro_info, true, false, false);
   struct hk_bg_eot eot_part_ro =
      hk_apple9_build_bg_eot(cmd, &ro_info, true, true, false);
   CHECK(!vk_command_buffer_has_error(&cmd->vk));
   CHECK(eot_main_ro.usc != 0 && eot_part_ro.usc != 0);
   CHECK(h_nostore.launch.count == 1 && h_store.launch.count == 0);
   cmd->state.gfx.render.no_store_color_mask = 0;
   struct hk_bg_eot eot_main_wr =
      hk_apple9_build_bg_eot(cmd, &ro_info, true, false, false);
   CHECK(!vk_command_buffer_has_error(&cmd->vk));
   CHECK(eot_main_wr.usc != 0 && h_store.launch.count == 1);
   free_cmd(cmd);

   /* 6. Secondary CBs and suspend/resume retain deferred no-store masks on
    * BeginRendering (skipping decompression and ZLS/EOT stores while unwritten).
    */
   const VkRenderingAttachmentInfo gen_none_ds = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_NONE,
   };
   VkRenderingInfo sec_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &gen_none_ds,
   };
   cmd = new_cmd(dev, pool, usc_pool);
   attempts = call_count = 0;
   cmd->vk.dynamic_graphics_state.feedback_loops = VK_IMAGE_ASPECT_DEPTH_BIT;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &sec_info);
   CHECK(call_count == 0);
   CHECK(cmd->state.gfx.render.no_store_ds_mask == 1);
   CHECK(cmd->state.gfx.render.store_none_ds_mask == 1);
   CHECK(cmd->state.gfx.render.ds_store_mask == 0);
   struct agx_zls_control_packed zls_no_store =
      cmd->state.gfx.render.cr.zls_control_partial;
   free_cmd(cmd);

   /* Suspending pass ignores storeOp: non-cleared depth in GENERAL with
    * STORE_OP_STORE is still deferred so unwritten suspending passes emit no
    * compressed store, while LOAD_OP_CLEAR is not deferred.
    */
   VkRenderingAttachmentInfo susp_ds_load = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   VkRenderingInfo susp_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_SUSPENDING_BIT,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &susp_ds_load,
   };
   cmd = new_cmd(dev, pool, usc_pool);
   attempts = call_count = 0;
   cmd->vk.dynamic_graphics_state.feedback_loops = VK_IMAGE_ASPECT_DEPTH_BIT;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &susp_info);
   CHECK(call_count == 0);
   CHECK(cmd->state.gfx.render.no_store_ds_mask == 1);
   CHECK(cmd->state.gfx.render.store_none_ds_mask == 1);
   free_cmd(cmd);

   /* Resuming pass ignores loadOp: LOAD_OP_CLEAR + STORE_OP_NONE is deferred. */
   VkRenderingAttachmentInfo res_clear_none = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_NONE,
   };
   VkRenderingInfo res_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_RESUMING_BIT,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &res_clear_none,
   };
   cmd = new_cmd(dev, pool, usc_pool);
   attempts = call_count = 0;
   cmd->vk.dynamic_graphics_state.feedback_loops = VK_IMAGE_ASPECT_DEPTH_BIT;
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &res_info);
   CHECK(call_count == 0);
   CHECK(cmd->state.gfx.render.no_store_ds_mask == 1);
   CHECK(cmd->state.gfx.render.store_none_ds_mask == 1);
   free_cmd(cmd);

   /* 7. Suspending pass with LOAD_OP_CLEAR + STORE_OP_NONE MUST set
    * process_empty_tiles=true so untouched tiles are cleared and stored before
    * stream closure for the resuming pass.
    */
   VkRenderingAttachmentInfo susp_ds_clear_none = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_NONE,
   };
   VkRenderingInfo susp_clear_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_SUSPENDING_BIT,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &susp_ds_clear_none,
   };
   cmd = new_cmd(dev, pool, usc_pool);
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &susp_clear_info);
   CHECK(cmd->current_cs.gfx && cmd->current_cs.gfx->cr.process_empty_tiles);
   CHECK(cmd->state.gfx.render.no_store_ds_mask == 0);
   free_cmd(cmd);

   /* 8. Positive draw promotion on a merged/chained render pass with a live
    * first_cs: pass 1 loaded depth (chain_load[0]=true), pass 2 merged with
    * DONT_CARE + STORE_OP_NONE and promoted depth on draw. Promotion MUST
    * preserve z_load=true on cs->cr.zls_control (matching chain_load[0]) while
    * enabling z_store=true.
    */
   cmd = new_cmd(dev, pool, usc_pool);
   VkRenderingAttachmentInfo ds_dontcare_none = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&zs_view),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_NONE,
   };
   VkRenderingInfo dc_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
      .pDepthAttachment = &ds_dontcare_none,
   };
   hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), &dc_info);
   CHECK(cmd->current_cs.gfx && cmd->current_cs.gfx == cmd->state.gfx.render.first_cs);
   /* Simulate a merged pass whose first pass loaded depth (chain_load[0]=true). */
   cmd->state.gfx.render.merge_reach = 1;
   cmd->state.gfx.render.chain_load[0] = true;
   cmd->vk.dynamic_graphics_state.ds.depth.test_enable = true;
   cmd->vk.dynamic_graphics_state.ds.depth.write_enable = true;
   CHECK(hk_feedback_prepare_draw(cmd));
   CHECK(cmd->state.gfx.render.no_store_ds_mask == 0);
   CHECK(cmd->state.gfx.render.store_none_ds_mask == 0);
   CHECK(util_dynarray_num_elements(&cmd->feedback_ops, struct hk_feedback_op) == 1);
   /* First stream's rebuilt zls_control must match a LOAD+store ZLS (identical
    * to partial ZLS here since partial also forces load+store), NOT DONT_CARE.
    */
   CHECK(memcmp(&cmd->current_cs.gfx->cr.zls_control,
                &cmd->state.gfx.render.cr.zls_control_partial,
                sizeof(zls_no_store)) == 0);
   CHECK(memcmp(&cmd->current_cs.gfx->cr.zls_control, &zls_no_store,
                sizeof(zls_no_store)) != 0);
   free_cmd(cmd);

   cmd = new_cmd(dev, NULL, NULL);
   fail_decompress = true;
   attempts = call_count = 0;
   cmd->rendering = true;
   cmd->state.gfx.render.area = (VkRect2D){{0, 0}, {64, 64}};
   cmd->state.gfx.render.cr.width = 64;
   cmd->state.gfx.render.cr.height = 64;
   cmd->state.gfx.render.cr.layers = 1;
   cmd->state.gfx.render.layer_count = 1;
   cmd->state.gfx.render.color_att_count = 1;
   cmd->state.gfx.render.color_att[0] = (struct hk_attachment){
      .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
      .iview = &view,
      .image_layout = VK_IMAGE_LAYOUT_GENERAL,
      .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .store_op = VK_ATTACHMENT_STORE_OP_NONE,
   };
   cmd->state.gfx.render.no_store_color_mask = 1;
   cmd->state.gfx.render.store_none_color_mask = 1;
   cmd->vk.dynamic_graphics_state.feedback_loops = VK_IMAGE_ASPECT_COLOR_BIT;
   const VkClearAttachment clear_c0 = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .colorAttachment = 0};
   CHECK(!hk_feedback_prepare_clear_attachments(cmd, 1, &clear_c0));
   CHECK(call_count == 1 && calls[0].view == &view && !calls[0].discard_contents);
   CHECK(vk_command_buffer_has_error(&cmd->vk));
   free_cmd(cmd);

   puts("failed feedback BeginRendering, discard safety, and deferred store-none promotion: PASS");
   agx_apple9_stage_launch_cache_finish(&h_nostore.launch);
   agx_apple9_stage_launch_cache_finish(&h_store.launch);
   agx_apple9_stage_launch_cache_finish(&h_empty_bg.launch);
   agx_apple9_stage_launch_cache_finish(&h_empty_eot.launch);
   _mesa_hash_table_destroy(dev->apple9_tile_helpers.ht, NULL);
   free(usc_pool);
   free(pool);
   free(dev);
   return 0;
}
