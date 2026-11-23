/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/libcl/libcl.h"
#include "compiler/shader_enums.h"
#include "agx_pack.h"

#define agx_push(ptr, T, cfg)                                                  \
   for (unsigned _loop = 0; _loop < 1;                                         \
        ++_loop, ptr = (GLOBAL void *)(((uintptr_t)ptr) + AGX_##T##_LENGTH))   \
      agx_pack(ptr, T, cfg)

static inline enum agx_index_size
agx_translate_index_size(uint8_t size_B)
{
   /* Index sizes are encoded logarithmically */
   static_assert(__builtin_ctz(1) == AGX_INDEX_SIZE_U8);
   static_assert(__builtin_ctz(2) == AGX_INDEX_SIZE_U16);
   static_assert(__builtin_ctz(4) == AGX_INDEX_SIZE_U32);

   assert((size_B == 1) || (size_B == 2) || (size_B == 4));
   return __builtin_ctz(size_B);
}

static inline unsigned
agx_indices_to_B(unsigned x, enum agx_index_size size)
{
   return x << size;
}

static inline uint8_t
agx_index_size_to_B(enum agx_index_size size)
{
   return agx_indices_to_B(1, size);
}

struct agx_grid {
   enum agx_cdm_mode mode;
   union {
      uint32_t count[3];
      uint64_t ptr;
   };
};

static struct agx_grid
agx_3d(uint32_t x, uint32_t y, uint32_t z)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_DIRECT, .count = {x, y, z}};
}

static struct agx_grid
agx_1d(uint32_t x)
{
   return agx_3d(x, 1, 1);
}

static struct agx_grid
agx_2d(uint32_t x, uint32_t y)
{
   return agx_3d(x, y, 1);
}

static struct agx_grid
agx_grid_indirect(uint64_t ptr)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_INDIRECT_GLOBAL, .ptr = ptr};
}

static struct agx_grid
agx_grid_indirect_local(uint64_t ptr)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_INDIRECT_LOCAL, .ptr = ptr};
}

static inline bool
agx_is_indirect(struct agx_grid grid)
{
   return grid.mode != AGX_CDM_MODE_DIRECT;
}

enum agx_barrier {
   /* No barrier/cache operations needed */
   AGX_BARRIER_NONE = 0,

   /* Catch-all for all defined barriers. Because we have not yet
    * reverse-engineered the finer details here, this is the only barrier we
    * have....
    */
   AGX_BARRIER_ALL = (1 << 0),

   /* Specifies ordering requirements for the compute job to happen before/after
    * the concurrent enqueued graphics on a graphical command buffer.
    */
   AGX_PREGFX = (1 << 1),
   AGX_POSTGFX = (1 << 2),
};

struct agx_draw {
   struct agx_grid b;
   uint64_t index_buffer;
   uint32_t index_buffer_range_B;
   uint32_t start;
   uint32_t index_bias;
   uint32_t start_instance;

   /* Primitive restart enabled. If true, implies indexed */
   bool restart;
   enum agx_index_size index_size;

   /* TODO: Optimize this boolean. We can't just check if index_buffer != 0
    * because that breaks with null index buffers.
    */
   bool indexed;
};

static inline struct agx_draw
agx_draw_indirect(uint64_t ptr)
{
   return (struct agx_draw){.b = agx_grid_indirect(ptr)};
}

static inline struct agx_draw
agx_draw_indexed(uint32_t index_count, uint32_t instance_count,
                 uint32_t first_index, uint32_t index_bias,
                 uint32_t first_instance, uint64_t buf, uint32_t range_B,
                 enum agx_index_size index_size, bool restart)
{
   return (struct agx_draw){
      .b = agx_3d(index_count, instance_count, 1),
      .index_buffer = buf,
      .index_buffer_range_B = range_B,
      .start = first_index,
      .index_bias = index_bias,
      .start_instance = first_instance,
      .index_size = index_size,
      .restart = restart,
      .indexed = true,
   };
}

static inline struct agx_draw
agx_draw_indexed_indirect(uint64_t ptr, uint64_t buf, uint32_t range_B,
                          enum agx_index_size index_size, bool restart)
{
   return (struct agx_draw){
      .b = agx_grid_indirect(ptr),
      .index_buffer = buf,
      .index_buffer_range_B = range_B,
      .index_size = index_size,
      .restart = restart,
      .indexed = true,
   };
}

static inline unsigned
agx_draw_index_range_B(struct agx_draw d)
{
   uint range_B = d.index_buffer_range_B;
   if (!agx_is_indirect(d.b))
      range_B -= agx_indices_to_B(d.start, d.index_size);

   return range_B;
}

static inline unsigned
agx_draw_index_range_el(struct agx_draw d)
{
   assert(d.indexed);
   return agx_draw_index_range_B(d) >> d.index_size;
}

static inline uint64_t
agx_draw_index_buffer(struct agx_draw d)
{
   assert(d.indexed);

   uint64_t ib = d.index_buffer;
   if (!agx_is_indirect(d.b))
      ib += agx_indices_to_B(d.start, d.index_size);

   return ib;
}

static bool
agx_direct_draw_overreads_indices(struct agx_draw d)
{
   uint32_t range_B = agx_indices_to_B(d.start + d.b.count[0], d.index_size);
   return range_B > d.index_buffer_range_B;
}

enum agx_chip {
   AGX_CHIP_G13G,
   AGX_CHIP_G13X,
   AGX_CHIP_G14G,
   AGX_CHIP_G14X,
   AGX_CHIP_G16G,
   AGX_CHIP_G17P,
   /* Append to retain existing chip values in precompiled inputs. */
   AGX_CHIP_G16C,
};

static inline GLOBAL uint32_t *
agx_cdm_barrier(GLOBAL uint32_t *out, enum agx_chip chip)
{
   agx_push(out, CDM_BARRIER, cfg) {
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_8 = true;
      // cfg.unk_11 = true;
      // cfg.unk_20 = true;
      // cfg.unk_24 = true; if clustered?
      if (chip == AGX_CHIP_G13X) {
         cfg.unk_4 = true;
         // cfg.unk_26 = true;
      }

      /* With multiple launches in the same CDM stream, we can get cache
       * coherency (? or sync?) issues. We hit this with blits, which need - in
       * between dispatches - need the PBE cache to be flushed and the texture
       * cache to be invalidated. Until we know what bits mean what exactly,
       * let's just set these after every launch to be safe. We can revisit in
       * the future when we figure out what the bits mean.
       */
      cfg.unk_0 = true;
      cfg.unk_1 = true;
      cfg.unk_2 = true;
      cfg.usc_cache_inval = true;
      cfg.unk_4 = true;
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_7 = true;
      cfg.unk_8 = true;
      cfg.unk_9 = true;
      cfg.unk_10 = true;
      cfg.unk_11 = true;
      cfg.unk_12 = true;
      cfg.unk_13 = true;
      cfg.unk_14 = true;
      cfg.unk_15 = true;
      cfg.unk_16 = true;
      cfg.unk_17 = true;
      cfg.unk_18 = true;
      cfg.unk_19 = true;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_terminate(GLOBAL uint32_t *out)
{
   agx_push(out, CDM_STREAM_TERMINATE, _)
      ;

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_terminate(GLOBAL uint32_t *out)
{
   agx_push(out, VDM_STREAM_TERMINATE, _)
      ;

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_jump(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, CDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_jump(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, VDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_cs_jump(GLOBAL uint32_t *out, uint64_t target, bool vdm)
{
   return vdm ? agx_vdm_jump(out, target) : agx_cdm_jump(out, target);
}

static inline unsigned
libagx_remap_adj_count(unsigned count, enum mesa_prim prim)
{
   if (prim == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      /* Spec gives formula for # of primitives in a tri strip adj */
      unsigned c4 = count >= 4 ? count - 4 : 0;
      return 3 * (c4 / 2);
   } else if (prim == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      return 2 * (count >= 3 ? count - 3 : 0);
   } else {
      /* Adjacency lists just drop half the vertices. */
      return count / 2;
   }
}
