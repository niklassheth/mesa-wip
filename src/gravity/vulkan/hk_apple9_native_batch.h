/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Whether a draw may continue an open native render. A pure predicate over
 * what the stream holds, what the draw is and what the paired kernel admits,
 * so it can be tested against the kernel's decoder vectors. Callers map the
 * kernel profile (today: research environment plus boot parameters) onto
 * struct hk_apple9_native_profile outside it.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include "gravity/lib/agx_native_mesh.h"

/* Hardware 64 KiB mesh output ring capacity in 16-byte rows (static_assert'd
 * against AGX_APPLE9_MESH_RING_ROWS in hk_cmd_apple9.c). */
#define HK_APPLE9_NATIVE_RING_ROWS 4096u

/* Largest ordinary draw block in the stream, including its terminator: the
 * state header (8 words), 8 PPP header/pointer pairs and the trailing PPP
 * word (agx_apple9_emit_draw_state, 100 bytes), then the largest draw form,
 * indexed direct or indexed indirect (10 words). The PPP update count is
 * fixed per block; dirty state changes record contents, not stream size. */
#define HK_APPLE9_ORDINARY_BLOCK_MAX_B (100 + 40)
/* A complete native block, HK_APPLE9_MESH_COMMAND_BYTES (checked there). */
#define HK_APPLE9_NATIVE_BLOCK_MAX_B 160
#define HK_APPLE9_BATCH_BLOCK_MAX_B                                            \
   (HK_APPLE9_NATIVE_BLOCK_MAX_B > HK_APPLE9_ORDINARY_BLOCK_MAX_B              \
       ? HK_APPLE9_NATIVE_BLOCK_MAX_B                                          \
       : HK_APPLE9_ORDINARY_BLOCK_MAX_B)

/* Every draw first reserves this much (hk_ensure_cs_has_space), which links
 * to a new chunk when the root lacks it. The paired decoder admits no link. */
#define HK_APPLE9_BATCH_ENSURE_B 256
#define HK_APPLE9_BATCH_RESERVE_B                                              \
   (HK_APPLE9_BATCH_BLOCK_MAX_B > HK_APPLE9_BATCH_ENSURE_B                     \
       ? HK_APPLE9_BATCH_BLOCK_MAX_B                                           \
       : HK_APPLE9_BATCH_ENSURE_B)

/* Default bound on estimated binned primitives per native render. Calibrated
 * on the Air (2026-10-24, native-mesh-scale-air-v2): a mesh-mode render of
 * ~2.75M visible triangles reached the 2193-block TVB cap and lost work, so
 * stay below it with margin. Estimates count every declared primitive, an
 * upper bound on what is binned. */
#define HK_APPLE9_NATIVE_RENDER_PRIMITIVES 1000000ull

/* The paired decoder reads at most 1024 words through the terminator unless
 * the kernel profile says otherwise. */
#define HK_APPLE9_NATIVE_DECODE_BYTES 4096

struct hk_apple9_native_profile {
   /* Most native dispatches per render; 1 keeps one native draw per render. */
   unsigned batch;
   /* Independent complete native blocks (otherwise repeated-input only). */
   bool blocks;
   /* Ordinary blocks may follow the leading native block. */
   bool mixed;
   /* Words the kernel decodes through the terminator (the attribute's
    * words=); 0 means HK_APPLE9_NATIVE_DECODE_BYTES / 4. */
   unsigned words;
   /* Most estimated binned primitives in one native render; 0 is unbounded.
    * A mesh-mode render that reaches the TVB cap loses work (unlike an
    * ordinary render, which recovers through a partial render), so batching
    * must keep each native render's geometry below it. */
   uint64_t max_render_primitives;
};

static inline size_t
hk_apple9_native_decode_bytes(const struct hk_apple9_native_profile *p)
{
   return p->words ? 4 * (size_t)p->words : HK_APPLE9_NATIVE_DECODE_BYTES;
}

struct hk_apple9_stream_summary {
   /* The stream leads with a native block. */
   bool native;
   unsigned dispatches;
   /* Bytes before the terminator slot, and the bytes the root chunk can
    * hold without hk_ensure_cs_has_space linking (its padding and link
    * reservation already subtracted). */
   size_t bytes, capacity;
   /* Estimated binned primitives already in the render. */
   uint64_t primitives;
   /* Cumulative hardware 64 KiB mesh output ring rows of native blocks in the
    * render, and whether any block had wg_rows > 40 (ring-limited). */
   uint64_t ring_rows;
   bool ring_limited;
};

struct hk_apple9_draw_summary {
   bool native;
   bool meta, queries, task, xfb, writes_memory;
   /* A mesh, geometry or tessellation stage is bound. */
   bool mesh, gs, tess;
   /* Repeated-input mode only: the open stream's first programs. */
   bool same_shaders;
   /* Estimated binned primitives of the draw (UINT64_MAX: unknown). */
   uint64_t primitives;
   /* Hardware 64 KiB mesh output ring rows of this native draw (UINT64_MAX:
    * unknown), and whether its per-WG footprint exceeds the 40-row safe cap. */
   uint64_t ring_rows;
   bool ring_limited;
};

static inline bool
hk_apple9_native_stream_admits(const struct hk_apple9_stream_summary *s,
                               const struct hk_apple9_draw_summary *d,
                               const struct hk_apple9_native_profile *p)
{
   if (p->batch <= 1 || p->batch > 64 || !s->native || !s->dispatches ||
       d->meta || d->queries || d->task || d->writes_memory)
      return false;
   if (!p->blocks)
      return d->native && s->dispatches < p->batch && !d->mesh &&
             (d->gs || d->tess) && d->same_shaders;

   size_t decode = hk_apple9_native_decode_bytes(p);
   size_t limit = s->capacity < decode ? s->capacity : decode;
   if (d->xfb || s->bytes + HK_APPLE9_BATCH_RESERVE_B > limit)
      return false;
   if (p->max_render_primitives &&
       (d->primitives > p->max_render_primitives ||
        s->primitives > p->max_render_primitives - d->primitives))
      return false;
   if (!d->native)
      return p->mixed && !d->mesh && !d->gs && !d->tess;
   /* When either the open render or the incoming native block has wg_rows > 40,
    * enforce that combined ring occupancy stays within the 64 KiB (4096-row)
    * mesh output ring without assuming an implicit cross-block hardware drain. */
   if (s->ring_limited || d->ring_limited) {
      if (d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS ||
          s->ring_rows > HK_APPLE9_NATIVE_RING_ROWS - d->ring_rows)
         return false;
   }
   return s->dispatches < p->batch;
}

/* The production profile: the VM's latched asahi_native_blocks profile, only
 * together with the qualified native path. A kill switch (drirc
 * hk_disable_native_blocks or HK_APPLE9_NATIVE_BLOCKS=0) keeps one native
 * block per render, as do kernels without the attribute. */
static inline struct hk_apple9_native_profile
hk_apple9_native_profile_from_kernel(const struct agx_native_blocks_support *b,
                                     bool native_mesh_qualified, bool disabled)
{
   if (disabled || !native_mesh_qualified || !b->qualified || b->max_native < 2 ||
       b->max_native > 64 || b->words < 1024 || b->words > 4096)
      return (struct hk_apple9_native_profile){.batch = 1};
   return (struct hk_apple9_native_profile){
      .batch = b->max_native,
      .blocks = true,
      .mixed = b->mixed,
      .words = b->words,
      .max_render_primitives = HK_APPLE9_NATIVE_RENDER_PRIMITIVES,
   };
}
