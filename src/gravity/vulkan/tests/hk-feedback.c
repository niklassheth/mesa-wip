/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "../hk_cmd_buffer.h"
#include "../hk_device.h"
#include "../hk_feedback.h"
#include "../hk_image.h"
#include "../hk_image_view.h"
#include "gravity/lib/agx_tilebuffer.h"
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #x);                            \
         return 1;                                                             \
      }                                                                        \
   } while (0)
int
main(void)
{
   const VkFormat formats[] = {VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT,
                               VK_FORMAT_R8G8_UINT, VK_FORMAT_R32_UINT,
                               VK_FORMAT_R16G16_UINT, VK_FORMAT_R32G32_UINT,
                               VK_FORMAT_R8G8B8A8_SRGB,
                               VK_FORMAT_R8G8B8A8_SNORM,
                               VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                               VK_FORMAT_R16_SFLOAT,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_FORMAT_R32G32B32A32_SFLOAT};
   struct agx_device dev = {.chip = AGX_CHIP_G16G};
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
      VkFormat raw = hk_feedback_raw_format(formats[i]);
      CHECK(vk_format_is_uint(raw));
      CHECK(vk_format_get_blocksize(raw) ==
            vk_format_get_blocksize(formats[i]));
      CHECK(ail_pixel_format[hk_format_to_pipe_format(raw)].channels ==
            ail_pixel_format[hk_format_to_pipe_format(formats[i])].channels);
      CHECK(hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, formats[i], 1));
      enum pipe_format pf = hk_format_to_pipe_format(raw);
      struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(&pf, 1, 1, true);
      struct drm_asahi_cmd_render render = {0};
      agx_tilebuffer_set_drm_cmd(&render, &tib);
      CHECK(render.utile_width_px % 16 == 0);
      CHECK(render.utile_height_px % 16 == 0);
      CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_3D, formats[i], 1));
      CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, formats[i], 4));
   }
   CHECK(hk_feedback_raw_format(VK_FORMAT_B10G11R11_UFLOAT_PACK32) ==
         VK_FORMAT_UNDEFINED);
   const VkFormat zs_formats[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_S8_UINT,
                                  VK_FORMAT_D32_SFLOAT_S8_UINT};
   for (unsigned i = 0; i < ARRAY_SIZE(zs_formats); ++i) {
      /* Depth/stencil planes decompress through copies, not raw blits. */
      CHECK(hk_feedback_raw_format(zs_formats[i]) == VK_FORMAT_UNDEFINED);
      CHECK(hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, zs_formats[i], 1));
      CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, zs_formats[i], 4));
   }
   CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 1));
   CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D,
                                   VK_FORMAT_D16_UNORM_S8_UINT, 1));

   struct hk_cmd_buffer cmd = {0}, other = {0};
   struct hk_image a = {0}, b = {0};
   a.planes[0].bo = b.planes[0].bo = (void *)0x1000;
   struct hk_image_view view = {.vk = {.image = &a.vk,
                                       .base_mip_level = 2,
                                       .base_array_layer = 3,
                                       .layer_count = 1}};
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &b, 2, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &b, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(!hk_feedback_is_raw(&other, &a, 2, 3));
   hk_feedback_invalidate_view(&cmd, &view);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_is_raw(&cmd, &a, 1, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &b, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   hk_feedback_invalidate_range(&cmd, &a, 2, 1, 3, 1);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_is_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   hk_feedback_invalidate(&cmd);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   util_dynarray_fini(&cmd.feedback_raw);
   util_dynarray_fini(&cmd.feedback_ops);

   /* Disjoint suballocations inside a shared BO pool must not alias, while
    * overlapping ranges (including multi-plane images) must alias.
    */
   struct hk_image pool_a = {.plane_count = 1};
   struct hk_image pool_b = {.plane_count = 1};
   struct hk_image pool_overlap = {.plane_count = 2};
   pool_a.planes[0].bo = pool_b.planes[0].bo = pool_overlap.planes[1].bo =
      (void *)0x2000;
   pool_a.planes[0].offset_B = 0;
   pool_a.planes[0].layout.size_B = 65536;
   pool_b.planes[0].offset_B = 65536;
   pool_b.planes[0].layout.size_B = 65536;
   pool_overlap.planes[0].bo = (void *)0x3000;
   pool_overlap.planes[0].offset_B = 0;
   pool_overlap.planes[0].layout.size_B = 4096;
   pool_overlap.planes[1].offset_B = 32768;
   pool_overlap.planes[1].layout.size_B = 4096;
   CHECK(!hk_feedback_images_alias(&pool_a, &pool_b));
   CHECK(hk_feedback_images_alias(&pool_a, &pool_overlap));
   CHECK(!hk_feedback_images_alias(&pool_b, &pool_overlap));

   /* Cross-CB submit-time elision, sticky prefer_raw latch after 2 executed
    * GPU feedback decompressions, reusable/out-of-order CB safety, multi-level
    * tracking, alias invalidation, and external opt-out.
    */
   struct hk_device dev_a = {.vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE}}};
   struct hk_device dev_b = {.vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE}}};
   struct hk_image img = {
      .vk = {.base = {.device = &dev_a.vk},
             .extent = {64, 64, 1},
             .mip_levels = 2,
             .array_layers = 2},
      .plane_count = 1,
      .feedback_track_cross_submit = true,
   };
   list_inithead(&img.feedback_raw_link);
   img.planes[0].bo = (void *)0x4000;
   img.planes[0].offset_B = 0;
   img.planes[0].layout.size_B = 65536;
   img.planes[0].layout.format = PIPE_FORMAT_R8G8B8A8_UNORM;

   struct hk_image img_disjoint = img;
   list_inithead(&img_disjoint.feedback_raw_link);
   img_disjoint.planes[0].offset_B = 65536;

   struct hk_image img_alias = img;
   list_inithead(&img_alias.feedback_raw_link);
   img_alias.planes[0].offset_B = 16384;

   struct hk_image img_ext = img;
   list_inithead(&img_ext.feedback_raw_link);
   img_ext.feedback_track_cross_submit = false;

   struct hk_image img_other_dev = img;
   list_inithead(&img_other_dev.feedback_raw_link);
   img_other_dev.vk.base.device = &dev_b.vk;
   img_other_dev.planes[0].bo = (void *)0x5000;

   struct hk_cmd_buffer cb_fb = {0}, cb_write = {0}, cb_normal_raw = {0},
                        cb_other = {0};
   list_inithead(&cb_fb.control_streams);
   list_inithead(&cb_write.control_streams);
   list_inithead(&cb_normal_raw.control_streams);
   list_inithead(&cb_other.control_streams);

   struct hk_cs cs_fb_vdm = {.type = HK_CS_VDM, .sync_id = 0};
   struct hk_cs cs_fb_cdm = {.type = HK_CS_CDM, .sync_id = 1};
   list_addtail(&cs_fb_vdm.node, &cb_fb.control_streams);
   list_addtail(&cs_fb_cdm.node, &cb_fb.control_streams);
   CHECK(hk_feedback_record_decompress(&cb_fb, &img, 0, 0, 2, 6, NULL, true));
   CHECK(cs_fb_vdm.feedback_decomp_op == 1 && cs_fb_cdm.feedback_decomp_op == 1);
   cb_fb.feedback_stats = (struct hk_feedback_stats){
      .calls = 1, .misses = 1, .bytes = 64 * 64 * 4,
   };

   /* Shift events helper preserves [event_start, event_end) ordering. */
   hk_feedback_shift_events(&cb_fb, 1, 3);
   struct hk_feedback_op *fb_op = cb_fb.feedback_ops.data;
   CHECK(fb_op->event_start == 5 && fb_op->event_end == 9);

   /* Reusable CB that writes compressed to img (recorded before prefer_raw). */
   hk_feedback_invalidate_range(&cb_write, &img, 0, 1, 0, 1);

   /* Normal pass recorded after prefer_raw (is_feedback = false). */
   struct hk_cs cs_nr_vdm = {.type = HK_CS_VDM, .sync_id = 0};
   list_addtail(&cs_nr_vdm.node, &cb_normal_raw.control_streams);
   CHECK(hk_feedback_record_decompress(&cb_normal_raw, &img, 0, 0, 0, 2, NULL,
                                       false));
   CHECK(cs_nr_vdm.feedback_decomp_op == 1);

   /* Submit 1: first GPU feedback decompression executes; prefer_raw stays 0. */
   struct vk_command_buffer *list1[] = {&cb_fb.vk};
   struct hk_feedback_submit_plan plan;
   CHECK(hk_feedback_plan_submit(&dev_a, list1, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   CHECK(!hk_feedback_plan_cs_skipped(&plan, 0, &cs_fb_vdm));
   CHECK(!hk_feedback_image_prefers_raw(&img));
   hk_feedback_plan_fini(&plan);

   /* Submit 2: repeated feedback CB across submit boundary is ELIDED and does
    * NOT increment feedback_gpu_decompressions or latch prefer_raw.
    */
   CHECK(hk_feedback_plan_submit(&dev_a, list1, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   CHECK(hk_feedback_plan_cs_skipped(&plan, 0, &cs_fb_vdm));
   CHECK(hk_feedback_plan_cs_skipped(&plan, 0, &cs_fb_cdm));
   struct hk_feedback_stats adj = cb_fb.feedback_stats;
   hk_feedback_plan_adjust_stats(&plan, 0, &adj);
   CHECK(adj.calls == 1 && adj.hits == 1 && adj.misses == 0 && adj.bytes == 0);
   CHECK(!hk_feedback_image_prefers_raw(&img));
   hk_feedback_plan_fini(&plan);

   /* Disjoint suballocation write must NOT invalidate img. */
   hk_feedback_invalidate_range(&cb_other, &img_disjoint, 0, 2, 0, 2);
   struct vk_command_buffer *list_disj[] = {&cb_other.vk, &cb_fb.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_disj, 2, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   CHECK(hk_feedback_plan_cs_skipped(&plan, 1, &cs_fb_vdm));
   hk_feedback_plan_fini(&plan);
   util_dynarray_clear(&cb_other.feedback_ops);

   /* Different mip level (1, 0) is NOT marked raw by (0, 0). */
   struct hk_cs cs_mip1 = {.type = HK_CS_VDM, .sync_id = 0};
   list_addtail(&cs_mip1.node, &cb_other.control_streams);
   CHECK(hk_feedback_record_decompress(&cb_other, &img, 1, 0, 0, 1, NULL, false));
   struct vk_command_buffer *list_mip1[] = {&cb_other.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_mip1, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);
   /* Invalidating level 1 leaves level 0 raw. */
   util_dynarray_clear(&cb_other.feedback_ops);
   list_delinit(&cs_mip1.node);
   hk_feedback_invalidate_range(&cb_other, &img, 1, 1, 0, 1);
   CHECK(hk_feedback_plan_submit(&dev_a, list_disj, 2, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   hk_feedback_plan_fini(&plan);
   util_dynarray_clear(&cb_other.feedback_ops);

   /* Submit 3: compressed write + feedback CB -> 2nd GPU feedback decomp
    * executes and latches sticky prefer_raw = 1.
    */
   struct vk_command_buffer *list_pingpong[] = {&cb_write.vk, &cb_fb.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_pingpong, 2, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   CHECK(hk_feedback_image_prefers_raw(&img));
   hk_feedback_plan_fini(&plan);

   /* Submit 4: normal pass recorded with prefer_raw=1 is ELIDED while raw. */
   struct vk_command_buffer *list_nr[] = {&cb_normal_raw.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   CHECK(hk_feedback_plan_cs_skipped(&plan, 0, &cs_nr_vdm));
   hk_feedback_plan_fini(&plan);

   /* Submit 5: out-of-order / reusable old compressed-write CB followed by
    * prefer_raw CB MUST execute the conditional decompression (no corruption!).
    */
   struct vk_command_buffer *list_reusable[] = {&cb_write.vk, &cb_normal_raw.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_reusable, 2, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   CHECK(!hk_feedback_plan_cs_skipped(&plan, 1, &cs_nr_vdm));
   hk_feedback_plan_fini(&plan);

   /* Overlapping BO alias write invalidates img. */
   hk_feedback_invalidate_range(&cb_other, &img_alias, 0, 1, 0, 1);
   struct vk_command_buffer *list_alias[] = {&cb_other.vk, &cb_normal_raw.vk};
   CHECK(hk_feedback_plan_submit(&dev_a, list_alias, 2, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);
   util_dynarray_clear(&cb_other.feedback_ops);

   /* Timeline wait-before-signal must NEVER elide, and must suppress cross-CB
    * elision on dev_a until that future-wait submission completes.
    */
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   hk_feedback_plan_fini(&plan);

   /* Future-wait submit on queue syncobj=77 invalidates and executes decomp. */
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, true, 77, 1, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   CHECK(!hk_feedback_plan_cs_skipped(&plan, 0, &cs_nr_vdm));
   hk_feedback_plan_fini(&plan);

   /* While syncobj=77 is still pending, even a normal submit does not elide or
    * populate raw state. */
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);

   /* Once syncobj=77 completes/finishes, tracking and elision resume cleanly. */
   hk_feedback_queue_finish(&dev_a, 77);
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   hk_feedback_plan_fini(&plan);

   /* External/swapchain image never elides across submits and never latches. */
   struct hk_cs cs_ext = {.type = HK_CS_VDM, .sync_id = 0};
   list_addtail(&cs_ext.node, &cb_other.control_streams);
   CHECK(hk_feedback_record_decompress(&cb_other, &img_ext, 0, 0, 0, 1, NULL,
                                       true));
   struct vk_command_buffer *list_ext[] = {&cb_other.vk};
   for (unsigned r = 0; r < 3; ++r) {
      CHECK(hk_feedback_plan_submit(&dev_a, list_ext, 1, false, 0, 0, &plan) == VK_SUCCESS);
      CHECK(plan.elided_total == 0);
      CHECK(!hk_feedback_image_prefers_raw(&img_ext));
      hk_feedback_plan_fini(&plan);
   }
   util_dynarray_clear(&cb_other.feedback_ops);
   list_delinit(&cs_ext.node);

   /* Device-scoped INVALIDATE_ALL clears dev_a without touching dev_b. */
   struct hk_cs cs_b = {.type = HK_CS_VDM, .sync_id = 0};
   list_addtail(&cs_b.node, &cb_other.control_streams);
   CHECK(hk_feedback_record_decompress(&cb_other, &img_other_dev, 0, 0, 0, 1,
                                       NULL, true));
   struct vk_command_buffer *list_b[] = {&cb_other.vk};
   CHECK(hk_feedback_plan_submit(&dev_b, list_b, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);

   hk_feedback_device_invalidate_all(&dev_a);
   CHECK(hk_feedback_plan_submit(&dev_a, list_nr, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 0);
   hk_feedback_plan_fini(&plan);
   CHECK(hk_feedback_plan_submit(&dev_b, list_b, 1, false, 0, 0, &plan) == VK_SUCCESS);
   CHECK(plan.elided_total == 1);
   hk_feedback_plan_fini(&plan);

   /* Clean image destruction unlinks from active raw list. */
   hk_feedback_image_destroy(&img);
   hk_feedback_image_destroy(&img_other_dev);
   CHECK(img.feedback_submitted_raw_mask == 0);
   CHECK(img_other_dev.feedback_submitted_raw_mask == 0);

   /* Drawless keep_raw_clear pass near end of a command pool chunk must
    * upload a valid mapped 8-byte zeroed scissor inside a BO (never a
    * past-end VA from a 0-byte upload) and leave uploaded_zbias == 0.
    */
   struct hk_cmd_buffer cb_raw_clear = {0};
   vk_object_base_init(&dev_a.vk, &cb_raw_clear.vk.base,
                       VK_OBJECT_TYPE_COMMAND_BUFFER);
   list_inithead(&cb_raw_clear.control_streams);
   uint8_t chunk0_mem[64] = {0};
   uint8_t chunk1_mem[128];
   memset(chunk1_mem, 0xaa, sizeof(chunk1_mem));
   cb_raw_clear.uploader[HK_CMD_POOL_MAIN].chunks[0] = (struct hk_upload_chunk){
      .map = chunk0_mem,
      .base = 0x10000000ull,
      .offset = HK_CMD_BO_SIZE - 32,
   };
   cb_raw_clear.uploader[HK_CMD_POOL_MAIN].chunks[1] = (struct hk_upload_chunk){
      .map = chunk1_mem,
      .base = 0x20000000ull,
      .offset = 0,
   };
   uint8_t vdm_mem[64] = {0};
   struct hk_cs cs_raw_clear = {
      .cmd = &cb_raw_clear,
      .type = HK_CS_VDM,
      .current = vdm_mem,
      .end = vdm_mem + sizeof(vdm_mem),
      .cr = {.process_empty_tiles = true},
   };
   cb_raw_clear.current_cs.gfx = &cs_raw_clear;
   cb_raw_clear.state.gfx.render.raw_color_mask = 1;
   CHECK(hk_pool_upload(&cb_raw_clear, NULL, 0, 64) == 0);
   hk_cmd_buffer_end_graphics(&cb_raw_clear);
   CHECK(cs_raw_clear.uploaded_scissor == 0x20000000ull);
   CHECK(cs_raw_clear.uploaded_scissor + AGX_SCISSOR_LENGTH <=
         0x20000000ull + HK_CMD_BO_SIZE);
   for (unsigned b = 0; b < AGX_SCISSOR_LENGTH; ++b)
      CHECK(chunk1_mem[b] == 0);
   CHECK(cs_raw_clear.uploaded_zbias == 0);

   util_dynarray_fini(&cb_fb.feedback_ops);
   util_dynarray_fini(&cb_write.feedback_ops);
   util_dynarray_fini(&cb_normal_raw.feedback_ops);
   util_dynarray_fini(&cb_other.feedback_ops);
   puts(
      "Integer compression classes and cross-CB submit-time feedback tracking: PASS");
   return 0;
}
