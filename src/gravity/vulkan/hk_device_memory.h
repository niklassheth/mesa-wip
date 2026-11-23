/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_device_memory.h"

#include "util/list.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

struct hk_device;
struct hk_image_plane;

/* Leave most of USC available to shaders, descriptors and command uploads. */
#define HK_INDEX_ALIAS_ALLOCATION_MAX (256ull << 20)
#define HK_INDEX_ALIAS_BUDGET (1ull << 30)

struct hk_va_range {
   uint64_t start;
   uint64_t end;
   uint64_t va_base;
   uint64_t bind_id;
   bool va_base_mixed;
};

/* Per-BO GPU VA aliasing state, shared across all hk_device_memory handles
 * backed by the same GEM BO (including OPAQUE_FD / DMA-BUF re-imports).
 */
struct hk_va_bo {
   uint64_t key;
   uint64_t primary_base;
   simple_mtx_t lock;
   struct util_dynarray primary_ranges;
   struct util_dynarray other_ranges;
   uint32_t mem_refcnt;
   uint32_t obj_refcnt;
   bool aliased;
};

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   /* Budgeted USC mapping for ordinary Apple9 index buffers bound to this
    * allocation. Keep it until FreeMemory: destroying a short-lived buffer
    * must not unbind a mapping of an otherwise reusable allocation.
    */
   simple_mtx_t index_va_lock;
   struct agx_va *index_va;

   /* Per-BO GPU VA aliasing tracker (owned ref until hk_FreeMemory). */
   struct hk_va_bo *va_bo;

   /* The API mapping, which the memory object owns. */
   void *map;
   size_t map_size;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;

uint64_t hk_memory_index_address(struct hk_device *dev,
                                struct hk_device_memory *mem);
void hk_memory_free_index_va(struct hk_device *dev,
                             struct hk_device_memory *mem);
bool hk_memory_init_va_tracking(struct hk_device *dev,
                                struct hk_device_memory *mem);
void hk_memory_finish_va_tracking(struct hk_device *dev,
                                  struct hk_device_memory *mem);
void hk_memory_record_va_bind(struct hk_device *dev,
                              struct hk_device_memory *mem,
                              uint64_t mem_offset, uint64_t size,
                              uint64_t gpu_addr, uint64_t *out_bo_key,
                              uint64_t *out_bind_id);
void hk_memory_remove_va_bind(struct hk_device *dev, uint64_t bo_key,
                              uint64_t bind_id);
