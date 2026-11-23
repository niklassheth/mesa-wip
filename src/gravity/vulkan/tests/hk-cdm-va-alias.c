/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>

#include "../hk_buffer.h"
#include "../hk_cmd_buffer.h"
#include "../hk_device.h"
#include "../hk_device_memory.h"
#include "../hk_entrypoints.h"
#include "../hk_image.h"
#include "util/hash_table.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static struct hk_device *
make_device(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   if (!dev)
      return NULL;
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1000000000ull;
   dev->dev.sparse_ro_offset = 0x8000000000ull;
   dev->dev.params.features = DRM_ASAHI_FEATURE_SOFT_FAULTS;
   simple_mtx_init(&dev->va_alias_lock, mtx_plain);
   dev->va_alias_bos = _mesa_hash_table_u64_create(NULL);
   return dev;
}

static void
destroy_device(struct hk_device *dev)
{
   _mesa_hash_table_u64_destroy(dev->va_alias_bos);
   simple_mtx_destroy(&dev->va_alias_lock);
   free(dev);
}

static int
test_sparse_ro_address(void)
{
   struct hk_device *dev = make_device();
   CHECK(dev && dev->va_alias_bos);

   struct agx_va sparse_va = {.addr = 0x2000000000ull, .size_B = 65536};
   struct hk_buffer sparse_buf = {
      .vk = {
         .base = {.type = VK_OBJECT_TYPE_BUFFER, .device = &dev->vk},
         .create_flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
         .size = 65536,
         .device_address = sparse_va.addr,
      },
      .va = &sparse_va,
   };

   /* Soft-fault residency: unbound sparse buffer pages already map zero_bo RO
    * in the primary VA, so read-only accesses stay on the primary VA.
    * SPARSE_BINDING-only buffers need no full-flush guard; SPARSE_RESIDENCY
    * buffers do, so writes to unbound RO zero_bo pages drain to the MMU.
    */
   CHECK(hk_sparse_unbound_unmapped(&dev->dev));
   CHECK(hk_buffer_address(&sparse_buf, 64, false) == sparse_va.addr + 64);
   CHECK(hk_buffer_address(&sparse_buf, 64, true) == sparse_va.addr + 64);
   CHECK(!hk_buffer_needs_cdm_full_flush(dev, &sparse_buf));
   sparse_buf.vk.create_flags |= VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
   CHECK(hk_buffer_needs_cdm_full_flush(dev, &sparse_buf));
   sparse_buf.vk.create_flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;

   /* Non-sparse capture-replay buffer: never uses the sparse RO shadow, even
    * without soft-fault residency.
    */
   struct hk_buffer replay_buf = {
      .vk = {
         .base = {.type = VK_OBJECT_TYPE_BUFFER, .device = &dev->vk},
         .create_flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
         .size = 65536,
         .device_address = sparse_va.addr,
      },
      .va = &sparse_va,
   };
   dev->dev.params.features = 0;
   CHECK(!hk_sparse_unbound_unmapped(&dev->dev));
   CHECK(hk_buffer_address(&replay_buf, 64, true) == sparse_va.addr + 64);
   CHECK(!hk_buffer_needs_cdm_full_flush(dev, &replay_buf));

   /* Scratch-page fallback (!soft_faults): sparse buffer reads still use the
    * RO shadow mapping so unbound reads return zero, and require full flushes.
    */
   CHECK(hk_buffer_address(&sparse_buf, 64, false) == sparse_va.addr + 64);
   CHECK(hk_buffer_address(&sparse_buf, 64, true) ==
         sparse_va.addr + 64 + dev->dev.sparse_ro_offset);
   CHECK(hk_buffer_needs_cdm_full_flush(dev, &sparse_buf));

   /* USC-aperture VAs have no RO shadow even in scratch-page mode. */
   struct agx_va usc_va = {.addr = dev->dev.shader_base + 0x10000, .size_B = 65536};
   sparse_buf.va = &usc_va;
   sparse_buf.vk.device_address = usc_va.addr;
   CHECK(hk_buffer_address(&sparse_buf, 64, true) == usc_va.addr + 64);
   CHECK(!hk_buffer_needs_cdm_full_flush(dev, &sparse_buf));

   destroy_device(dev);
   return 0;
}

static int
test_va_alias_tracking(void)
{
   struct hk_device *dev = make_device();
   CHECK(dev && dev->va_alias_bos);

   const uint64_t primary_va = 0x4000000000ull;
   const uint64_t index_va = 0x1001000000ull;
   const uint64_t sparse_va_a = 0x5000000000ull;
   const uint64_t sparse_va_b = 0x6000000000ull;

   struct agx_va backing = {.addr = primary_va, .size_B = 1024 * 1024};
   struct agx_bo bo = {.va = &backing, .size = 1024 * 1024};
   struct hk_device_memory mem = {
      .vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE_MEMORY}, .size = bo.size},
      .bo = &bo,
   };
   hk_memory_init_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   /* 1. Two overlapping normal buffers on primary VA: same VA, no alias. */
   uint64_t m1 = 0, b1 = 0, m2 = 0, b2 = 0;
   hk_memory_record_va_bind(dev, &mem, 0, 16384, primary_va + 0, &m1, &b1);
   hk_memory_record_va_bind(dev, &mem, 0, 16384, primary_va + 0, &m2, &b2);
   CHECK(!hk_device_has_va_alias(dev));

   /* 2. Disjoint index buffer (index_va) on the same VMA slab: no alias. */
   uint64_t m3 = 0, b3 = 0;
   hk_memory_record_va_bind(dev, &mem, 65536, 16384, index_va + 65536, &m3, &b3);
   CHECK(!hk_device_has_va_alias(dev));

   /* 3. Destroy index buffer and reuse its offset for a primary-VA buffer
    * (VMA slab reuse): still no alias.
    */
   hk_memory_remove_va_bind(dev, m3, b3);
   uint64_t m4 = 0, b4 = 0;
   hk_memory_record_va_bind(dev, &mem, 65536, 16384, primary_va + 65536, &m4, &b4);
   CHECK(!hk_device_has_va_alias(dev));

   /* 4. Overlapping index buffer + primary-VA buffer simultaneously: alias! */
   uint64_t m5 = 0, b5 = 0;
   hk_memory_record_va_bind(dev, &mem, 65536, 16384, index_va + 65536, &m5, &b5);
   CHECK(hk_device_has_va_alias(dev));

   /* Freeing mem clears its alias contribution; stale buffer destroy after
    * FreeMemory must be a safe no-op.
    */
   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));
   hk_memory_remove_va_bind(dev, m1, b1);
   hk_memory_remove_va_bind(dev, m4, b4);
   CHECK(!hk_device_has_va_alias(dev));

   /* 5. Single sparse buffer bound across multiple pages of one mem: no alias. */
   hk_memory_init_va_tracking(dev, &mem);
   hk_memory_record_va_bind(dev, &mem, 0, 65536, sparse_va_a + 0, NULL, NULL);
   hk_memory_record_va_bind(dev, &mem, 65536, 65536, sparse_va_a + 65536, NULL, NULL);
   /* Rebinding the exact same range at the same VA is still not an alias. */
   hk_memory_record_va_bind(dev, &mem, 0, 65536, sparse_va_a + 0, NULL, NULL);
   CHECK(!hk_device_has_va_alias(dev));

   /* 6. Second sparse buffer bound to overlapping range of same mem: alias! */
   hk_memory_record_va_bind(dev, &mem, 0, 65536, sparse_va_b + 0, NULL, NULL);
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   /* 7. Single sparse buffer aliasing two resource offsets to one mem page. */
   CHECK(hk_memory_init_va_tracking(dev, &mem));
   hk_memory_record_va_bind(dev, &mem, 0, 65536, sparse_va_a + 0, NULL, NULL);
   CHECK(!hk_device_has_va_alias(dev));
   hk_memory_record_va_bind(dev, &mem, 0, 65536, sparse_va_a + 65536, NULL, NULL);
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   /* 8. Cross-VkDeviceMemory OPAQUE_FD / DMA-BUF re-import sharing the same
    * GEM BO handle: overlapping binds on mem1 and mem2 with distinct GPU VAs
    * must still detect aliasing, and freeing mem1 keeps va_alias_count > 0
    * until mem2 is also freed.
    */
   struct agx_bo shared_bo = {.va = &backing, .size = 1024 * 1024, .handle = 77};
   struct hk_device_memory mem1 = {
      .vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE_MEMORY}, .size = shared_bo.size},
      .bo = &shared_bo,
   };
   struct hk_device_memory mem2 = {
      .vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE_MEMORY}, .size = shared_bo.size},
      .bo = &shared_bo,
   };
   CHECK(hk_memory_init_va_tracking(dev, &mem1));
   CHECK(hk_memory_init_va_tracking(dev, &mem2));
   CHECK(mem1.va_bo == mem2.va_bo && mem1.va_bo->mem_refcnt == 2);
   uint64_t sb1_key = 0, sb1_id = 0, sb2_key = 0, sb2_id = 0;
   hk_memory_record_va_bind(dev, &mem1, 0, 16384, primary_va + 0, &sb1_key, &sb1_id);
   CHECK(!hk_device_has_va_alias(dev));
   hk_memory_record_va_bind(dev, &mem2, 0, 16384, index_va + 0, &sb2_key, &sb2_id);
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem1);
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem2);
   CHECK(!hk_device_has_va_alias(dev));

   /* 9. Twiddled 4096-tile sparse image bind (64x64 standard-tile rows) must
    * coalesce adjacent/bridged disjoint tiles down to 1 range instead of
    * growing quadratically, while still detecting an overlapping second bind.
    */
   CHECK(hk_memory_init_va_tracking(dev, &mem));
   const unsigned std_cols = 32, std_rows = 32; /* 32*32*4 = 4096 16KiB tiles */
   for (unsigned sy = 0; sy < std_rows; sy++) {
      for (unsigned sub_y = 0; sub_y < 2; sub_y++) {
         for (unsigned sx = 0; sx < std_cols; sx++) {
            for (unsigned sub_x = 0; sub_x < 2; sub_x++) {
               uint64_t mem_off =
                  ((uint64_t)(sy * std_cols + sx) * 4 + sub_y * 2 + sub_x) *
                  AIL_PAGESIZE;
               /* Distinct twiddled VA base per tile */
               uint64_t twiddled_va = sparse_va_a + ((mem_off ^ 0x55550000ull));
               hk_memory_record_va_bind(dev, &mem, mem_off, AIL_PAGESIZE,
                                        twiddled_va, NULL, NULL);
            }
         }
      }
   }
   CHECK(!hk_device_has_va_alias(dev));
   CHECK(util_dynarray_num_elements(&mem.va_bo->other_ranges, struct hk_va_range) == 1);
   /* Overlapping bind into the coalesced sparse image range detects alias! */
   hk_memory_record_va_bind(dev, &mem, 16 * AIL_PAGESIZE, AIL_PAGESIZE,
                            sparse_va_b, NULL, NULL);
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   /* 10. Non-adjacent sparse fragment cap falls back conservatively without
    * unbounded growth.
    */
   CHECK(hk_memory_init_va_tracking(dev, &mem));
   for (unsigned i = 0; i <= 260 && !hk_device_has_va_alias(dev); i++) {
      uint64_t mem_off = (uint64_t)i * 2 * AIL_PAGESIZE; /* leave 16K gaps */
      hk_memory_record_va_bind(dev, &mem, mem_off, AIL_PAGESIZE,
                               sparse_va_a + mem_off, NULL, NULL);
   }
   CHECK(hk_device_has_va_alias(dev));
   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   destroy_device(dev);
   return 0;
}

static int
test_cdm_barriers_under_va_alias(void)
{
   struct hk_device *dev = make_device();
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && dev->va_alias_bos && cmd);
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->cdm_narrow_barriers = UTIL_DYNARRAY_INIT;

   struct agx_va backing = {.addr = 0x4000000000ull, .size_B = 131072};
   struct agx_bo bo = {.va = &backing, .size = 131072};
   struct hk_device_memory mem = {
      .vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE_MEMORY}, .size = bo.size},
      .bo = &bo,
   };
   CHECK(hk_memory_init_va_tracking(dev, &mem));
   hk_memory_record_va_bind(dev, &mem, 0, 65536, 0x5000000000ull, NULL, NULL);
   CHECK(!hk_device_has_va_alias(dev));

   /* Phase 0: single sparse buffer (no alias) -> 0x60000160 for both merged
    * CDM junction and inline COMPUTE->COMPUTE barrier.
    * Phase 1: second sparse buffer aliases same mem -> full flush for both.
    * Phase 2: mem freed -> back to 0x60000160 for both.
    */
   for (unsigned phase = 0; phase < 3; ++phase) {
      if (phase == 1)
         hk_memory_record_va_bind(dev, &mem, 0, 65536, 0x6000000000ull, NULL, NULL);
      else if (phase == 2)
         hk_memory_finish_va_tracking(dev, &mem);

      bool expect_narrow = (phase != 1);
      CHECK(hk_device_has_va_alias(dev) == !expect_narrow);

      /* Merged CDM junction */
      uint8_t a_bytes[256] = {0}, b_bytes[256] = {0};
      struct hk_cs a = {.cmd = cmd, .type = HK_CS_CDM, .current = a_bytes,
                        .end = a_bytes + sizeof(a_bytes), .cdm_dirty = true};
      struct hk_cs b = {.cmd = cmd, .type = HK_CS_CDM,
                        .current = b_bytes + AGX_CDM_BARRIER_LENGTH,
                        .end = b_bytes + sizeof(b_bytes),
                        .end_flush = (uint32_t *)b_bytes, .addr = 0x1000};
      cmd->current_cs.cs = &a;
      hk_cmd_buffer_end_compute(cmd);
      uint32_t full = *(uint32_t *)a_bytes;
      CHECK(full != 0x60000160 && full != 0);
      hk_cs_merge_cdm(&a, &b, dev);
      CHECK(*(uint32_t *)a_bytes == (expect_narrow ? 0x60000160 : full));

      /* Inline COMPUTE -> COMPUTE barrier */
      uint8_t c_bytes[4096] = {0};
      struct hk_cs c = {.cmd = cmd, .type = HK_CS_CDM, .current = c_bytes,
                        .end = c_bytes + sizeof(c_bytes),
                        .sync_stages = HK_SYNC_COMPUTE, .cdm_dirty = true};
      cmd->current_cs.cs = &c;
      hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
      CHECK(*(uint32_t *)c_bytes == (expect_narrow ? 0x60000160 : full));
      cmd->current_cs.cs = NULL;
      util_dynarray_clear(&cmd->cdm_narrow_barriers);
      cmd->cdm_narrow_upgraded = 0;
   }

   /* Record-before-alias submit upgrade: record CB with both a merged CDM
    * junction and an inline COMPUTE->COMPUTE barrier while va_alias_count == 0
    * (both bake 0x60000160), THEN create an alias and upgrade at submit time.
    */
   CHECK(hk_memory_init_va_tracking(dev, &mem));
   hk_memory_record_va_bind(dev, &mem, 0, 65536, 0x5000000000ull, NULL, NULL);
   CHECK(!hk_device_has_va_alias(dev));

   uint8_t a_bytes[256] = {0}, b_bytes[256] = {0}, c_bytes[4096] = {0};
   struct hk_cs a = {.cmd = cmd, .type = HK_CS_CDM, .current = a_bytes,
                     .end = a_bytes + sizeof(a_bytes), .cdm_dirty = true};
   struct hk_cs b = {.cmd = cmd, .type = HK_CS_CDM,
                     .current = b_bytes + AGX_CDM_BARRIER_LENGTH,
                     .end = b_bytes + sizeof(b_bytes),
                     .end_flush = (uint32_t *)b_bytes, .addr = 0x1000};
   cmd->current_cs.cs = &a;
   hk_cmd_buffer_end_compute(cmd);
   uint32_t full = *(uint32_t *)a_bytes;
   hk_cs_merge_cdm(&a, &b, dev);
   CHECK(*(uint32_t *)a_bytes == 0x60000160);

   struct hk_cs c = {.cmd = cmd, .type = HK_CS_CDM, .current = c_bytes,
                     .end = c_bytes + sizeof(c_bytes),
                     .sync_stages = HK_SYNC_COMPUTE, .cdm_dirty = true};
   cmd->current_cs.cs = &c;
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
   CHECK(*(uint32_t *)c_bytes == 0x60000160);
   cmd->current_cs.cs = NULL;

   /* Alias arrives after CB recording (e.g. vkQueueBindSparse before submit). */
   hk_memory_record_va_bind(dev, &mem, 0, 65536, 0x6000000000ull, NULL, NULL);
   CHECK(hk_device_has_va_alias(dev));

   /* Submit-time upgrade rewrites every 0x60000160 word to full flush. */
   hk_cmd_upgrade_narrow_cdm_barriers(cmd, dev);
   CHECK(*(uint32_t *)a_bytes == full);
   CHECK(*(uint32_t *)c_bytes == full);
   CHECK(cmd->cdm_narrow_upgraded == 1);
   /* Re-submitting an already upgraded reusable CB is an idempotent no-op. */
   hk_cmd_upgrade_narrow_cdm_barriers(cmd, dev);
   CHECK(*(uint32_t *)a_bytes == full && *(uint32_t *)c_bytes == full);

   hk_memory_finish_va_tracking(dev, &mem);
   CHECK(!hk_device_has_va_alias(dev));

   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   util_dynarray_fini(&cmd->cdm_narrow_barriers);
   free(cmd);
   destroy_device(dev);
   return 0;
}

int
main(void)
{
   if (test_sparse_ro_address() || test_va_alias_tracking() ||
       test_cdm_barriers_under_va_alias())
      return 1;
   puts("PASS sparse RO suppression, VA-alias tracking, and CDM junction/inline barriers");
   return 0;
}
