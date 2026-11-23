/* SPDX-License-Identifier: MIT */
#pragma once
#include "hk_private.h"
#include "util/u_dynarray.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;
struct hk_feedback_stats;
struct hk_image;
struct hk_image_view;
struct agx_device;
struct vk_command_buffer;

enum hk_feedback_op_kind {
   HK_FEEDBACK_OP_DECOMPRESS,
   HK_FEEDBACK_OP_INVALIDATE_RANGE,
   HK_FEEDBACK_OP_INVALIDATE_ALL,
};

struct hk_feedback_op {
   enum hk_feedback_op_kind kind;
   struct hk_image *image;
   uint16_t first_level, levels;
   uint16_t first_layer, layers;
   uint32_t event_start, event_end;
   uint64_t bytes;
   bool is_feedback;
};

struct hk_feedback_cb_plan {
   uint32_t op_offset;
   uint32_t op_count;
   uint32_t elided_count;
   uint64_t elided_bytes;
};

struct hk_feedback_submit_plan {
   unsigned cb_count;
   unsigned elided_total;
   struct hk_feedback_cb_plan *cbs;
   bool *skip_ops;
};

VkFormat hk_feedback_raw_format(VkFormat format);
bool hk_feedback_is_zs_format(VkFormat format);
bool hk_feedback_can_compress(const struct agx_device *dev, VkImageType type,
                              VkFormat format, unsigned samples);
bool hk_feedback_images_alias(const struct hk_image *a,
                              const struct hk_image *b);
bool hk_feedback_image_prefers_raw(const struct hk_image *image);
void hk_feedback_image_destroy(struct hk_image *image);
void hk_feedback_device_invalidate_all(const struct hk_device *dev);
void hk_feedback_queue_finish(const struct hk_device *dev, uint32_t syncobj);

bool hk_feedback_is_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                        unsigned level, unsigned layer);
bool hk_feedback_mark_raw(struct hk_cmd_buffer *cmd,
                          const struct hk_image *image, unsigned level,
                          unsigned layer);
bool hk_feedback_record_decompress(struct hk_cmd_buffer *cmd,
                                   struct hk_image *image, unsigned level,
                                   unsigned layer, uint32_t event_start,
                                   uint32_t event_end, struct hk_cs *prev_tail,
                                   bool is_feedback, bool discard_contents);
uint64_t hk_feedback_decompress_bytes(const struct hk_image *image,
                                      unsigned level, bool discard_contents);
void hk_feedback_record_invalidate_all(struct hk_cmd_buffer *cmd);
void hk_feedback_shift_events(struct hk_cmd_buffer *cmd, uint32_t pos,
                              uint32_t delta);
void hk_feedback_invalidate(struct hk_cmd_buffer *cmd);
void hk_feedback_invalidate_range(struct hk_cmd_buffer *cmd,
                                  const struct hk_image *image,
                                  unsigned first_level, unsigned levels,
                                  unsigned first_layer, unsigned layers);
void hk_feedback_invalidate_view(struct hk_cmd_buffer *cmd,
                                 const struct hk_image_view *view);
bool hk_meta_decompress_view(struct hk_cmd_buffer *cmd,
                             const struct hk_image_view *view, unsigned layer,
                             bool discard_contents);
bool hk_feedback_prepare_clear_attachments(struct hk_cmd_buffer *cmd,
                                           uint32_t attachment_count,
                                           const VkClearAttachment *attachments);

VkResult hk_feedback_plan_submit(struct hk_device *dev,
                                 struct vk_command_buffer *const *cmdbufs,
                                 unsigned cmdbuf_count, bool has_future_wait,
                                 uint32_t queue_syncobj,
                                 uint64_t queue_signal_point,
                                 struct hk_feedback_submit_plan *plan);
void hk_feedback_plan_fini(struct hk_feedback_submit_plan *plan);
bool hk_feedback_plan_cs_skipped(const struct hk_feedback_submit_plan *plan,
                                 unsigned cb_idx, const struct hk_cs *cs);
const bool *hk_feedback_plan_cb_skip_ops(
   const struct hk_feedback_submit_plan *plan, unsigned cb_idx,
   unsigned *num_ops_out);
void hk_feedback_plan_adjust_stats(const struct hk_feedback_submit_plan *plan,
                                   unsigned cb_idx,
                                   struct hk_feedback_stats *stats);
