/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_feedback.h"
#include "hk_event.h"
#include "vulkan/vulkan_core.h"

#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "vk_alloc.h"
#include "util/bitset.h"

#define HK_EVENT_PAGE_SIZE 16384
#define HK_EVENTS_PER_PAGE (HK_EVENT_PAGE_SIZE / sizeof(VkResult))

struct hk_event_page {
   struct list_head link;
   struct agx_bo *bo;
   VkResult *map;
   unsigned used;
   BITSET_DECLARE(free, HK_EVENTS_PER_PAGE);
};

static VkResult
hk_event_alloc(struct hk_device *dev, struct hk_event *event)
{
   simple_mtx_lock(&dev->events.lock);
   struct hk_event_page *page;
   VkResult result = VK_SUCCESS;
   if (list_is_empty(&dev->events.available)) {
      page = vk_zalloc(&dev->vk.alloc, sizeof(*page), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!page) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      page->bo = agx_bo_create(&dev->dev, HK_EVENT_PAGE_SIZE, 0,
                               AGX_BO_WRITEBACK, "Events");
      if (!page->bo) {
         vk_free(&dev->vk.alloc, page);
         result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto out;
      }
      page->map = agx_bo_map(page->bo);
      if (!page->map) {
         agx_bo_unreference(&dev->dev, page->bo);
         vk_free(&dev->vk.alloc, page);
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      BITSET_ONES(page->free);
      list_addtail(&page->link, &dev->events.available);
   } else {
      page = list_first_entry(&dev->events.available, struct hk_event_page, link);
   }

   unsigned slot = BITSET_FFS(page->free) - 1;
   assert(slot < HK_EVENTS_PER_PAGE);
   BITSET_CLEAR(page->free, slot);
   if (++page->used == HK_EVENTS_PER_PAGE)
      list_del(&page->link);
   event->page = page;
   event->slot = slot;
   event->status = &page->map[slot];
   event->addr = page->bo->va->addr + slot * sizeof(VkResult);

out:
   simple_mtx_unlock(&dev->events.lock);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_event *event;

   event = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                            VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   event->device_only =
      (pCreateInfo->flags & VK_EVENT_CREATE_DEVICE_ONLY_BIT) != 0;

   if (!event->device_only) {
      VkResult result = hk_event_alloc(dev, event);
      if (result != VK_SUCCESS) {
         vk_object_free(&dev->vk, pAllocator, event);
         return vk_error(dev, result);
      }

      *event->status = VK_EVENT_RESET;
   }

   *pEvent = hk_event_to_handle(event);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyEvent(VkDevice device, VkEvent _event,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_event, event, _event);

   if (!event)
      return;

   if (event->page) {
      struct hk_event_page *page = event->page;
      simple_mtx_lock(&dev->events.lock);
      assert(!BITSET_TEST(page->free, event->slot));
      BITSET_SET(page->free, event->slot);
      if (page->used-- == HK_EVENTS_PER_PAGE)
         list_addtail(&page->link, &dev->events.available);
      bool empty = page->used == 0;
      if (empty)
         list_del(&page->link);
      simple_mtx_unlock(&dev->events.lock);

      /* Pages never move. Release only after every event using them is gone. */
      if (empty) {
         agx_bo_unreference(&dev->dev, page->bo);
         vk_free(&dev->vk.alloc, page);
      }
   }
   vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);
   assert(!event->device_only && event->status != NULL);

   return *event->status;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);
   assert(!event->device_only && event->status != NULL);

   *event->status = VK_EVENT_SET;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);
   assert(!event->device_only && event->status != NULL);

   *event->status = VK_EVENT_RESET;

   return VK_SUCCESS;
}

static void
hk_event_dep_src_scope(const VkDependencyInfo *info,
                       VkPipelineStageFlags2 *src_stages,
                       VkAccessFlags2 *src_access)
{
   VkPipelineStageFlags2 stages = 0;
   VkAccessFlags2 access = 0;
   bool any = false;

   for (unsigned i = 0; i < info->memoryBarrierCount; ++i) {
      stages |= info->pMemoryBarriers[i].srcStageMask;
      access |= info->pMemoryBarriers[i].srcAccessMask;
      any = true;
   }
   for (unsigned i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      stages |= info->pBufferMemoryBarriers[i].srcStageMask;
      access |= info->pBufferMemoryBarriers[i].srcAccessMask;
      any = true;
   }
   for (unsigned i = 0; i < info->imageMemoryBarrierCount; ++i) {
      stages |= info->pImageMemoryBarriers[i].srcStageMask;
      access |= info->pImageMemoryBarriers[i].srcAccessMask;
      any = true;
   }

   if (!any) {
      stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      access = VK_ACCESS_2_MEMORY_WRITE_BIT;
   }
   *src_stages = stages;
   *src_access = access;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_event, event, _event);

   /* Device-only events are never queried by the host, and hk_CmdWaitEvents2
    * enforces the matching VkDependencyInfo via pipeline barriers.
    */
   if (event->device_only)
      return;

   hk_apple9_end_native_batch(cmd);

   VkPipelineStageFlags2 src_stages = 0;
   VkAccessFlags2 src_access = 0;
   hk_event_dep_src_scope(pDependencyInfo, &src_stages, &src_access);

   perf_debug(cmd, "Set event");
   if (hk_sync_stage_mask(src_stages, true)) {
      hk_cs_end_hint(cmd, HK_CS_END_EVENT);
      hk_cmd_sync_barrier(cmd, src_stages,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          src_access | VK_ACCESS_2_MEMORY_WRITE_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
   }

   unsigned saved_scope = cmd->sync_scope;
   cmd->sync_scope = HK_SYNC_COMPUTE;
   hk_queue_write(cmd, event->addr, VK_EVENT_SET, false);
   cmd->sync_scope = saved_scope;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_event, event, _event);

   if (event->device_only)
      return;

   hk_apple9_end_native_batch(cmd);

   perf_debug(cmd, "Reset event");
   if (hk_sync_stage_mask(stageMask, true)) {
      hk_cs_end_hint(cmd, HK_CS_END_EVENT);
      hk_cmd_sync_barrier(cmd, stageMask,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);
   }

   unsigned saved_scope = cmd->sync_scope;
   cmd->sync_scope = HK_SYNC_COMPUTE;
   hk_queue_write(cmd, event->addr, VK_EVENT_RESET, false);
   cmd->sync_scope = saved_scope;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                  const VkEvent *pEvents,
                  const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!cmd->in_meta)
      hk_feedback_invalidate(cmd);
   hk_apple9_end_native_batch(cmd);

   if (HK_PERF(dev, NOBARRIER))
      return;

   perf_debug(cmd, "Wait events");
   hk_cs_end_hint(cmd, HK_CS_END_EVENT);

   uint32_t group = ++cmd->sync_next_group;
   assert(group != 0);
   for (uint32_t e = 0; e < eventCount; ++e) {
      const VkDependencyInfo *info = &pDependencyInfos[e];
      if (cmd->empty_barriers.size)
         hk_cmd_render_merge_flush(cmd);

      for (unsigned i = 0; i < info->memoryBarrierCount; ++i) {
         const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
         hk_cmd_sync_barrier_group(cmd, b->srcStageMask, b->dstStageMask,
                                   b->srcAccessMask, b->dstAccessMask, group);
      }
      for (unsigned i = 0; i < info->bufferMemoryBarrierCount; ++i) {
         const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
         hk_cmd_sync_barrier_group(cmd, b->srcStageMask, b->dstStageMask,
                                   b->srcAccessMask, b->dstAccessMask, group);
      }
      for (unsigned i = 0; i < info->imageMemoryBarrierCount; ++i) {
         const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
         cmd->barrier_image = hk_image_from_handle(b->image);
         cmd->barrier_qfot = b->srcQueueFamilyIndex != b->dstQueueFamilyIndex;
         hk_cmd_sync_barrier_group(cmd, b->srcStageMask, b->dstStageMask,
                                   b->srcAccessMask, b->dstAccessMask, group);
      }
      cmd->barrier_image = NULL;
   }
}
