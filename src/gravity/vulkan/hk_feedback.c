/* SPDX-License-Identifier: MIT */
#include "hk_feedback.h"
#include <xf86drm.h>
#include "gravity/lib/agx_apple9.h"
#include "util/format/u_format.h"
#include "util/list.h"
#include "util/simple_mtx.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_image_view.h"

/* Latch sticky prefer_raw on an image after this many executed GPU feedback
 * decompressions, stopping compressed <-> raw ping-pong across render passes
 * without demoting images that never re-compress after a single feedback pass.
 */
#define HK_FEEDBACK_PREFER_RAW_THRESHOLD 2u

static simple_mtx_t hk_feedback_lock = SIMPLE_MTX_INITIALIZER;
static struct list_head hk_feedback_raw_images = {
   &hk_feedback_raw_images, &hk_feedback_raw_images
};
static uint32_t hk_feedback_raw_image_count = 0;

struct hk_feedback_future_fence {
   struct list_head link;
   const struct hk_device *dev;
   uint32_t syncobj;
   uint64_t point;
};

static struct list_head hk_feedback_future_fences = {
   &hk_feedback_future_fences, &hk_feedback_future_fences
};
static uint32_t hk_feedback_future_fence_count = 0;

/* Repartitioning compressed channels is not a bitcast. Keep the compression
 * class as well as the texel size. Packed minifloats have no integer view and
 * retain the feedback compression ban rather than canonicalizing their NaNs.
 */
VkFormat
hk_feedback_raw_format(VkFormat format)
{
   enum pipe_format pf = hk_format_to_pipe_format(format);
   if (!ail_is_valid_pixel_format(pf) || util_format_is_depth_or_stencil(pf) ||
       util_format_is_compressed(pf))
      return VK_FORMAT_UNDEFINED;
   switch (ail_pixel_format[pf].channels) {
   case AGX_CHANNELS_R8:
      return VK_FORMAT_R8_UINT;
   case AGX_CHANNELS_R16:
      return VK_FORMAT_R16_UINT;
   case AGX_CHANNELS_R8G8:
      return VK_FORMAT_R8G8_UINT;
   case AGX_CHANNELS_R32:
      return VK_FORMAT_R32_UINT;
   case AGX_CHANNELS_R16G16:
      return VK_FORMAT_R16G16_UINT;
   case AGX_CHANNELS_R8G8B8A8:
      return VK_FORMAT_R8G8B8A8_UINT;
   case AGX_CHANNELS_R10G10B10A2:
      return VK_FORMAT_A2B10G10R10_UINT_PACK32;
   case AGX_CHANNELS_R32G32:
      return VK_FORMAT_R32G32_UINT;
   case AGX_CHANNELS_R16G16B16A16:
      return VK_FORMAT_R16G16B16A16_UINT;
   case AGX_CHANNELS_R32G32B32A32:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

/* Depth and stencil planes compress with the colour scheme of their integer
 * alias (Z32F as R32, S8 as R8). They decompress through raw colour stores of
 * those aliases like colour, and feedback renders keep their ZLS stores raw.
 *
 * D16 keeps the feedback allocation ban: with a feedback loop enabled only by
 * the pipeline (GENERAL layout), compressed D16 failed the
 * attachment_feedback_loop_layout depth_read_write_different_areas cases on
 * G16 while D32 and D32S8 passed.
 */
bool
hk_feedback_is_zs_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_D32_SFLOAT:
   case VK_FORMAT_S8_UINT:
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
   default:
      return false;
   }
}

bool
hk_feedback_can_compress(const struct agx_device *dev, VkImageType type,
                         VkFormat format, unsigned samples)
{
   (void)dev;
   return type == VK_IMAGE_TYPE_2D && samples == 1 &&
          (hk_feedback_is_zs_format(format) ||
           hk_feedback_raw_format(format) != VK_FORMAT_UNDEFINED);
}

/* True if two distinct images overlap in backing BO memory. Disjoint
 * suballocations inside a shared VkDeviceMemory pool do not alias. When a
 * plane's layout size is 0 (synthetic unit-test images), stay conservative.
 */
bool
hk_feedback_images_alias(const struct hk_image *a, const struct hk_image *b)
{
   if (a == b)
      return false;
   unsigned pa_count = MAX2(a->plane_count, 1u);
   unsigned pb_count = MAX2(b->plane_count, 1u);
   for (unsigned p = 0; p < pa_count; ++p) {
      const struct hk_image_plane *pa = &a->planes[p];
      if (!pa->bo)
         continue;
      for (unsigned q = 0; q < pb_count; ++q) {
         const struct hk_image_plane *pb = &b->planes[q];
         if (pa->bo != pb->bo)
            continue;
         if (pa->layout.size_B == 0 || pb->layout.size_B == 0)
            return true;
         uint64_t a0 = pa->offset_B;
         uint64_t a1 = a0 + pa->layout.size_B;
         uint64_t b0 = pb->offset_B;
         uint64_t b1 = b0 + pb->layout.size_B;
         if (a0 < b1 && b0 < a1)
            return true;
      }
   }
   return false;
}

bool
hk_feedback_image_prefers_raw(const struct hk_image *image)
{
   return image->feedback_track_cross_submit &&
          __atomic_load_n(&image->feedback_prefer_raw, __ATOMIC_ACQUIRE) != 0;
}

static void
hk_feedback_unlink_locked(struct hk_image *image)
{
   image->feedback_submitted_raw_mask = 0;
   if (image->feedback_raw_link.next &&
       !list_is_empty(&image->feedback_raw_link)) {
      list_delinit(&image->feedback_raw_link);
      assert(hk_feedback_raw_image_count > 0);
      __atomic_sub_fetch(&hk_feedback_raw_image_count, 1, __ATOMIC_RELEASE);
   }
}

void
hk_feedback_image_destroy(struct hk_image *image)
{
   /* Never read mutable list links unlocked: concurrent alias invalidation
    * from another thread's submit mutates feedback_raw_link under
    * hk_feedback_lock. The ACQUIRE load of count == 0 pairs with the RELEASE
    * decrement after list_delinit.
    */
   if (__atomic_load_n(&hk_feedback_raw_image_count, __ATOMIC_ACQUIRE) == 0)
      return;
   simple_mtx_lock(&hk_feedback_lock);
   hk_feedback_unlink_locked(image);
   simple_mtx_unlock(&hk_feedback_lock);
}

static void
hk_feedback_invalidate_device_locked(const struct hk_device *dev)
{
   list_for_each_entry_safe(struct hk_image, img, &hk_feedback_raw_images,
                            feedback_raw_link) {
      if (img->vk.base.device == &dev->vk)
         hk_feedback_unlink_locked(img);
   }
}

void
hk_feedback_device_invalidate_all(const struct hk_device *dev)
{
   if (__atomic_load_n(&hk_feedback_raw_image_count, __ATOMIC_ACQUIRE) == 0)
      return;
   simple_mtx_lock(&hk_feedback_lock);
   hk_feedback_invalidate_device_locked(dev);
   simple_mtx_unlock(&hk_feedback_lock);
}

void
hk_feedback_queue_finish(const struct hk_device *dev, uint32_t syncobj)
{
   if (__atomic_load_n(&hk_feedback_future_fence_count, __ATOMIC_ACQUIRE) == 0)
      return;
   simple_mtx_lock(&hk_feedback_lock);
   list_for_each_entry_safe(struct hk_feedback_future_fence, f,
                            &hk_feedback_future_fences, link) {
      if (f->dev == dev && f->syncobj == syncobj) {
         list_del(&f->link);
         free(f);
         assert(hk_feedback_future_fence_count > 0);
         __atomic_sub_fetch(&hk_feedback_future_fence_count, 1,
                            __ATOMIC_RELEASE);
      }
   }
   simple_mtx_unlock(&hk_feedback_lock);
}

struct hk_feedback_raw {
   const struct hk_image *image;
   unsigned level, layer;
   bool raw;
};

bool
hk_feedback_is_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                   unsigned level, unsigned layer)
{
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image && entry->level == level &&
          entry->layer == layer)
         return entry->raw;
   }
   return false;
}

bool
hk_feedback_mark_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                     unsigned level, unsigned layer)
{
   /* A raw store through another image alias can overwrite this image's
    * differently partitioned compression metadata. Do not reuse its state.
    */
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (hk_feedback_images_alias(entry->image, image))
         entry->raw = false;
   }
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image && entry->level == level &&
          entry->layer == layer) {
         entry->raw = true;
         return true;
      }
   }
   struct hk_feedback_raw *entry =
      util_dynarray_grow(&cmd->feedback_raw, struct hk_feedback_raw, 1);
   if (!entry) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }
   *entry = (struct hk_feedback_raw){image, level, layer, true};
   return true;
}

static uint64_t
hk_feedback_level_bytes(const struct hk_image *image, unsigned level)
{
   uint64_t bytes = 0;
   for (unsigned p = 0; p < image->plane_count; ++p) {
      bytes += (uint64_t)u_minify(image->vk.extent.width, level) *
               u_minify(image->vk.extent.height, level) *
               util_format_get_blocksize(image->planes[p].layout.format);
   }
   return bytes;
}

bool
hk_feedback_record_decompress(struct hk_cmd_buffer *cmd, struct hk_image *image,
                              unsigned level, unsigned layer,
                              uint32_t event_start, uint32_t event_end,
                              struct hk_cs *prev_tail, bool is_feedback)
{
   assert(event_start <= event_end);
   struct hk_feedback_op *op =
      util_dynarray_grow(&cmd->feedback_ops, struct hk_feedback_op, 1);
   if (!op) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }
   *op = (struct hk_feedback_op){
      .kind = HK_FEEDBACK_OP_DECOMPRESS,
      .image = image,
      .first_level = level,
      .levels = 1,
      .first_layer = layer,
      .layers = 1,
      .event_start = event_start,
      .event_end = event_end,
      .bytes = hk_feedback_level_bytes(image, level),
      .is_feedback = is_feedback,
   };
   uint32_t op_idx =
      util_dynarray_num_elements(&cmd->feedback_ops, struct hk_feedback_op);
   struct list_head *start = prev_tail ? prev_tail->node.next
                                       : cmd->control_streams.next;
   if (start) {
      for (struct list_head *n = start; n != &cmd->control_streams; n = n->next) {
         struct hk_cs *cs = list_entry(n, struct hk_cs, node);
         cs->feedback_decomp_op = op_idx;
      }
   }
   return true;
}

void
hk_feedback_record_invalidate_all(struct hk_cmd_buffer *cmd)
{
   struct hk_feedback_op *op =
      util_dynarray_grow(&cmd->feedback_ops, struct hk_feedback_op, 1);
   if (!op) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   *op = (struct hk_feedback_op){
      .kind = HK_FEEDBACK_OP_INVALIDATE_ALL,
   };
}

void
hk_feedback_shift_events(struct hk_cmd_buffer *cmd, uint32_t pos,
                         uint32_t delta)
{
   if (!delta || !cmd->feedback_ops.size)
      return;
   util_dynarray_foreach(&cmd->feedback_ops, struct hk_feedback_op, op) {
      if (op->kind != HK_FEEDBACK_OP_DECOMPRESS)
         continue;
      assert(pos <= op->event_start || pos >= op->event_end);
      if (op->event_start >= pos) {
         op->event_start += delta;
         op->event_end += delta;
      }
   }
}

/* Cache only commands before the current recording point, never global image
 * state. Each CB starts unknown; cross-CB reuse is resolved in execution order
 * at queue submission via cmd->feedback_ops.
 */
void
hk_feedback_invalidate(struct hk_cmd_buffer *cmd)
{
   util_dynarray_clear(&cmd->feedback_raw);
}

void
hk_feedback_invalidate_range(struct hk_cmd_buffer *cmd,
                             const struct hk_image *image, unsigned first_level,
                             unsigned levels, unsigned first_layer,
                             unsigned layers)
{
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image) {
         if (entry->level >= first_level &&
             entry->level - first_level < levels &&
             entry->layer >= first_layer && entry->layer - first_layer < layers)
            entry->raw = false;
      } else if (hk_feedback_images_alias(entry->image, image)) {
         /* Image aliases can have different mip/layer partitions. */
         entry->raw = false;
      }
   }

   if (!image)
      return;
   struct hk_feedback_op *op =
      util_dynarray_grow(&cmd->feedback_ops, struct hk_feedback_op, 1);
   if (!op) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   *op = (struct hk_feedback_op){
      .kind = HK_FEEDBACK_OP_INVALIDATE_RANGE,
      .image = (struct hk_image *)image,
      .first_level = MIN2(first_level, UINT16_MAX),
      .levels = MIN2(levels, UINT16_MAX),
      .first_layer = MIN2(first_layer, UINT16_MAX),
      .layers = MIN2(layers, UINT16_MAX),
   };
}

void
hk_feedback_invalidate_view(struct hk_cmd_buffer *cmd,
                            const struct hk_image_view *view)
{
   if (!view)
      return;
   const struct hk_image *image =
      container_of(view->vk.image, struct hk_image, vk);
   hk_feedback_invalidate_range(cmd, image, view->vk.base_mip_level, 1,
                                view->vk.base_array_layer,
                                view->vk.layer_count);
}

static bool
hk_feedback_subresource_bit(const struct hk_image *image, unsigned level,
                            unsigned layer, uint64_t *bit_out)
{
   unsigned mip_levels = MAX2(image->vk.mip_levels, 1u);
   if (level >= mip_levels)
      return false;
   uint64_t idx = (uint64_t)layer * mip_levels + level;
   if (idx >= 64)
      return false;
   *bit_out = 1ull << idx;
   return true;
}

static uint64_t
hk_feedback_range_mask(const struct hk_image *image, unsigned first_level,
                       unsigned level_count, unsigned first_layer,
                       unsigned layer_count)
{
   unsigned mip_levels = MAX2(image->vk.mip_levels, 1u);
   unsigned array_layers = MAX2(image->vk.array_layers, 1u);
   if (first_level >= mip_levels || first_layer >= array_layers)
      return 0;
   uint64_t end_level = MIN2((uint64_t)first_level + level_count, mip_levels);
   uint64_t end_layer = MIN2((uint64_t)first_layer + layer_count, array_layers);
   uint64_t mask = 0;
   for (uint64_t layer = first_layer; layer < end_layer; ++layer) {
      uint64_t base = layer * mip_levels;
      if (base >= 64)
         break;
      for (uint64_t level = first_level; level < end_level; ++level) {
         if (base + level >= 64)
            break;
         mask |= 1ull << (base + level);
      }
   }
   return mask;
}

static void
hk_feedback_invalidate_aliases_locked(const struct hk_device *dev,
                                      const struct hk_image *image)
{
   list_for_each_entry_safe(struct hk_image, other, &hk_feedback_raw_images,
                            feedback_raw_link) {
      if (other->vk.base.device == &dev->vk &&
          hk_feedback_images_alias(other, image))
         hk_feedback_unlink_locked(other);
   }
}

static bool
hk_feedback_prune_future_fences_locked(const struct hk_device *dev)
{
   bool pending = false;
   list_for_each_entry_safe(struct hk_feedback_future_fence, f,
                            &hk_feedback_future_fences, link) {
      if (f->dev != dev)
         continue;
      bool done = false;
      if (dev->dev.fd > 0) {
         uint32_t handle = f->syncobj;
         uint64_t point = f->point;
         done = drmSyncobjTimelineWait(dev->dev.fd, &handle, &point, 1, 0,
                                       DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
                                       NULL) == 0;
      }
      if (done) {
         list_del(&f->link);
         free(f);
         assert(hk_feedback_future_fence_count > 0);
         __atomic_sub_fetch(&hk_feedback_future_fence_count, 1,
                            __ATOMIC_RELEASE);
      } else {
         pending = true;
      }
   }
   return pending;
}

static bool
hk_feedback_record_future_fence_locked(const struct hk_device *dev,
                                       uint32_t syncobj, uint64_t point)
{
   if (!syncobj)
      return true;
   list_for_each_entry(struct hk_feedback_future_fence, f,
                       &hk_feedback_future_fences, link) {
      if (f->dev == dev && f->syncobj == syncobj) {
         f->point = MAX2(f->point, point);
         return true;
      }
   }
   struct hk_feedback_future_fence *f = malloc(sizeof(*f));
   if (!f)
      return false;
   *f = (struct hk_feedback_future_fence){
      .dev = dev,
      .syncobj = syncobj,
      .point = point,
   };
   list_addtail(&f->link, &hk_feedback_future_fences);
   __atomic_add_fetch(&hk_feedback_future_fence_count, 1, __ATOMIC_RELEASE);
   return true;
}

VkResult
hk_feedback_plan_submit(struct hk_device *dev,
                        struct vk_command_buffer *const *cmdbufs,
                        unsigned cmdbuf_count, bool has_future_wait,
                        uint32_t queue_syncobj, uint64_t queue_signal_point,
                        struct hk_feedback_submit_plan *plan)
{
   memset(plan, 0, sizeof(*plan));
   plan->cb_count = cmdbuf_count;

   unsigned total_ops = 0;
   bool has_decompress = false;
   bool has_streams = false;
   for (unsigned i = 0; i < cmdbuf_count; ++i) {
      struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)cmdbufs[i];
      if (!list_is_empty(&cmd->control_streams))
         has_streams = true;
      unsigned n =
         util_dynarray_num_elements(&cmd->feedback_ops, struct hk_feedback_op);
      total_ops += n;
      if (!has_decompress && n) {
         util_dynarray_foreach(&cmd->feedback_ops, struct hk_feedback_op, op) {
            if (op->kind == HK_FEEDBACK_OP_DECOMPRESS) {
               has_decompress = true;
               break;
            }
         }
      }
   }

   if (!total_ops && !has_future_wait)
      return VK_SUCCESS;

   /* Fast path: no future-wait reordering, no decompressions in this submit,
    * and no active raw images anywhere in the process.
    */
   if (!has_future_wait && !has_decompress &&
       __atomic_load_n(&hk_feedback_raw_image_count, __ATOMIC_ACQUIRE) == 0 &&
       __atomic_load_n(&hk_feedback_future_fence_count, __ATOMIC_ACQUIRE) == 0)
      return VK_SUCCESS;

   if (has_decompress && !has_future_wait) {
      plan->cbs = calloc(cmdbuf_count, sizeof(*plan->cbs));
      plan->skip_ops = calloc(total_ops, sizeof(*plan->skip_ops));
      if (!plan->cbs || !plan->skip_ops) {
         hk_feedback_plan_fini(plan);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   simple_mtx_lock(&hk_feedback_lock);
   bool unordered_window = false;
   if (__atomic_load_n(&hk_feedback_future_fence_count, __ATOMIC_ACQUIRE) > 0)
      unordered_window = hk_feedback_prune_future_fences_locked(dev);

   if (has_future_wait) {
      unordered_window = true;
      if (has_streams &&
          !hk_feedback_record_future_fence_locked(dev, queue_syncobj,
                                                  queue_signal_point)) {
         hk_feedback_invalidate_device_locked(dev);
         simple_mtx_unlock(&hk_feedback_lock);
         hk_feedback_plan_fini(plan);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   if (unordered_window) {
      /* A wait-before-signal timeline submission may execute out of CPU submit
       * order relative to other queues. Invalidate all cross-submit raw state
       * and disable elision until every such submission has completed on GPU.
       */
      hk_feedback_invalidate_device_locked(dev);
      simple_mtx_unlock(&hk_feedback_lock);
      hk_feedback_plan_fini(plan);
      return VK_SUCCESS;
   }

   unsigned offset = 0;
   for (unsigned i = 0; i < cmdbuf_count; ++i) {
      struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)cmdbufs[i];
      unsigned n =
         util_dynarray_num_elements(&cmd->feedback_ops, struct hk_feedback_op);
      const struct hk_feedback_op *ops = cmd->feedback_ops.data;
      if (plan->cbs) {
         plan->cbs[i].op_offset = offset;
         plan->cbs[i].op_count = n;
      }

      for (unsigned k = 0; k < n; ++k) {
         const struct hk_feedback_op *op = &ops[k];
         switch (op->kind) {
         case HK_FEEDBACK_OP_DECOMPRESS: {
            struct hk_image *img = op->image;
            uint64_t bit = 0;
            bool trackable =
               img && img->feedback_track_cross_submit &&
               hk_feedback_subresource_bit(img, op->first_level,
                                           op->first_layer, &bit);
            if (trackable && (img->feedback_submitted_raw_mask & bit)) {
               plan->skip_ops[offset + k] = true;
               plan->elided_total++;
               plan->cbs[i].elided_count++;
               plan->cbs[i].elided_bytes += op->bytes;
            } else {
               if (img)
                  hk_feedback_invalidate_aliases_locked(dev, img);
               if (trackable) {
                  img->feedback_submitted_raw_mask |= bit;
                  if (!img->feedback_raw_link.next)
                     list_inithead(&img->feedback_raw_link);
                  if (list_is_empty(&img->feedback_raw_link)) {
                     list_addtail(&img->feedback_raw_link,
                                  &hk_feedback_raw_images);
                     __atomic_add_fetch(&hk_feedback_raw_image_count, 1,
                                        __ATOMIC_RELEASE);
                  }
                  if (op->is_feedback) {
                     img->feedback_gpu_decompressions++;
                     if (img->feedback_gpu_decompressions >=
                         HK_FEEDBACK_PREFER_RAW_THRESHOLD) {
                        __atomic_store_n(&img->feedback_prefer_raw, 1,
                                         __ATOMIC_RELEASE);
                     }
                  }
               }
            }
            break;
         }
         case HK_FEEDBACK_OP_INVALIDATE_RANGE: {
            struct hk_image *img = op->image;
            if (!img)
               break;
            if (img->feedback_submitted_raw_mask) {
               uint64_t rmask = hk_feedback_range_mask(
                  img, op->first_level, op->levels, op->first_layer,
                  op->layers);
               img->feedback_submitted_raw_mask &= ~rmask;
               if (img->feedback_submitted_raw_mask == 0)
                  hk_feedback_unlink_locked(img);
            }
            hk_feedback_invalidate_aliases_locked(dev, img);
            break;
         }
         case HK_FEEDBACK_OP_INVALIDATE_ALL:
            hk_feedback_invalidate_device_locked(dev);
            break;
         }
      }
      offset += n;
   }
   simple_mtx_unlock(&hk_feedback_lock);
   return VK_SUCCESS;
}

void
hk_feedback_plan_fini(struct hk_feedback_submit_plan *plan)
{
   free(plan->cbs);
   free(plan->skip_ops);
   memset(plan, 0, sizeof(*plan));
}

bool
hk_feedback_plan_cs_skipped(const struct hk_feedback_submit_plan *plan,
                            unsigned cb_idx, const struct hk_cs *cs)
{
   if (!plan->elided_total || !cs->feedback_decomp_op)
      return false;
   assert(cb_idx < plan->cb_count);
   const struct hk_feedback_cb_plan *cb = &plan->cbs[cb_idx];
   uint32_t op_idx = cs->feedback_decomp_op - 1;
   assert(op_idx < cb->op_count);
   return plan->skip_ops[cb->op_offset + op_idx];
}

const bool *
hk_feedback_plan_cb_skip_ops(const struct hk_feedback_submit_plan *plan,
                             unsigned cb_idx, unsigned *num_ops_out)
{
   if (!plan->elided_total || !plan->cbs) {
      *num_ops_out = 0;
      return NULL;
   }
   assert(cb_idx < plan->cb_count);
   const struct hk_feedback_cb_plan *cb = &plan->cbs[cb_idx];
   if (!cb->elided_count) {
      *num_ops_out = 0;
      return NULL;
   }
   *num_ops_out = cb->op_count;
   return &plan->skip_ops[cb->op_offset];
}

void
hk_feedback_plan_adjust_stats(const struct hk_feedback_submit_plan *plan,
                              unsigned cb_idx, struct hk_feedback_stats *stats)
{
   if (!plan->elided_total || !plan->cbs)
      return;
   assert(cb_idx < plan->cb_count);
   const struct hk_feedback_cb_plan *cb = &plan->cbs[cb_idx];
   if (!cb->elided_count)
      return;
   uint64_t elided = MIN2((uint64_t)cb->elided_count, stats->misses);
   stats->misses -= elided;
   stats->hits += elided;
   stats->bytes = stats->bytes > cb->elided_bytes
                     ? stats->bytes - cb->elided_bytes
                     : 0;
}
