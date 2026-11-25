/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Whether a draw may share an open graphics render pass that already contains
 * (or would acquire) a native Apple9 object/mesh dispatch. With opaque
 * DRM_ASAHI_RENDER_NATIVE_MESH (bit 28), the kernel does not parse VDM streams
 * on the CPU, so only genuine hardware invariants (cumulative 64 KiB mesh
 * output ring occupancy for ring-limited shaders, optional conservative TVB
 * primitive cap, and the per-draw isolation kill switch) can force a split.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Hardware 64 KiB mesh output ring capacity in 16-byte rows (static_assert'd
 * against AGX_APPLE9_MESH_RING_ROWS in hk_cmd_apple9.c). */
#define HK_APPLE9_NATIVE_RING_ROWS 4096u

/* Default bound on estimated binned primitives per native render. Estimates
 * count every declared primitive, an upper bound on what is binned. */
#define HK_APPLE9_NATIVE_RENDER_PRIMITIVES 1000000ull

struct hk_apple9_native_profile {
   /* Whether multiple draws (native and ordinary in any order) may share one
    * render pass under DRM_ASAHI_FEATURE_NATIVE_MESH. False when the feature
    * is absent or the kill switch (HK_APPLE9_NATIVE_BLOCKS=0 /
    * hk_disable_native_blocks) forces one isolated native draw per render. */
   bool blocks;
   /* Most estimated binned primitives in one native render; 0 is unbounded. */
   uint64_t max_render_primitives;
};

struct hk_apple9_stream_summary {
   /* The stream already contains at least one native block. */
   bool native;
   /* The stream already contains at least one draw (ordinary or native). */
   bool has_draws;
   /* True if any native block in the render is subject to the conservative
    * max_render_primitives TVB cap (false when all native draws so far are
    * unbounded multi-parent fused tessellation). */
   bool bounded_prims;
   /* Estimated binned primitives already in the render. */
   uint64_t primitives;
   /* Cumulative hardware 64 KiB mesh output ring rows of native blocks in the
    * render, and whether any block is ring-limited (lacks a wrap-safe U0_B). */
   uint64_t ring_rows;
   bool ring_limited;
};

struct hk_apple9_draw_summary {
   bool native;
   /* True when this native draw is exempt from the conservative 1M TVB
    * primitive cap (unbounded multi-parent fused tessellation). */
   bool unbounded_prims;
   /* Estimated binned primitives of the draw (UINT64_MAX: unknown). */
   uint64_t primitives;
   /* Hardware 64 KiB mesh output ring rows of this native draw (UINT64_MAX:
    * unknown), and whether its layout is ring-limited. */
   uint64_t ring_rows;
   bool ring_limited;
};

enum hk_apple9_native_admit_result {
   HK_APPLE9_NATIVE_ADMIT_APPEND = 0,
   HK_APPLE9_NATIVE_ADMIT_SPLIT,
   HK_APPLE9_NATIVE_ADMIT_REJECT,
};

/* Whether a single draw fits inside a fresh native render pass on its own.
 * Callers selecting native vs fallback must enforce this before recording
 * so splitting an open render never admits an individually oversized draw. */
static inline bool
hk_apple9_native_draw_fits_alone(const struct hk_apple9_draw_summary *d,
                                 const struct hk_apple9_native_profile *p)
{
   if (!d->native)
      return true;
   if (p->max_render_primitives && !d->unbounded_prims &&
       d->primitives > p->max_render_primitives)
      return false;
   if (d->ring_limited && d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS)
      return false;
   return true;
}

static inline enum hk_apple9_native_admit_result
hk_apple9_native_stream_classify(const struct hk_apple9_stream_summary *s,
                                 const struct hk_apple9_draw_summary *d,
                                 const struct hk_apple9_native_profile *p)
{
   if (!hk_apple9_native_draw_fits_alone(d, p))
      return HK_APPLE9_NATIVE_ADMIT_REJECT;
   /* An empty stream always starts cleanly once the draw fits alone; an
    * ordinary stream receiving an ordinary draw is never constrained by
    * native-mesh admission. */
   if (!s->native && !(s->has_draws && d->native))
      return HK_APPLE9_NATIVE_ADMIT_APPEND;
   if (!p->blocks)
      return HK_APPLE9_NATIVE_ADMIT_SPLIT;
   bool check_prims = s->bounded_prims || (d->native && !d->unbounded_prims);
   if (p->max_render_primitives && check_prims &&
       (d->primitives > p->max_render_primitives ||
        s->primitives > p->max_render_primitives - d->primitives))
      return HK_APPLE9_NATIVE_ADMIT_SPLIT;
   if (!d->native)
      return HK_APPLE9_NATIVE_ADMIT_APPEND;
   /* When either the open render or the incoming native block is ring-limited,
    * enforce that combined ring occupancy stays within the 64 KiB (4096-row)
    * mesh output ring without assuming an implicit cross-block hardware drain. */
   if (s->ring_limited || d->ring_limited) {
      if (d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS ||
          s->ring_rows > HK_APPLE9_NATIVE_RING_ROWS - d->ring_rows)
         return HK_APPLE9_NATIVE_ADMIT_SPLIT;
   }
   return HK_APPLE9_NATIVE_ADMIT_APPEND;
}

static inline bool
hk_apple9_native_stream_admits(const struct hk_apple9_stream_summary *s,
                               const struct hk_apple9_draw_summary *d,
                               const struct hk_apple9_native_profile *p)
{
   /* An empty stream always starts cleanly; an ordinary stream receiving an
    * ordinary draw is never constrained by native-mesh admission. */
   if (!s->native && !(s->has_draws && d->native))
      return true;
   if (!p->blocks)
      return false;
   bool check_prims = s->bounded_prims || (d->native && !d->unbounded_prims);
   if (p->max_render_primitives && check_prims &&
       (d->primitives > p->max_render_primitives ||
        s->primitives > p->max_render_primitives - d->primitives))
      return false;
   if (!d->native)
      return true;
   if (s->ring_limited || d->ring_limited) {
      if (d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS ||
          s->ring_rows > HK_APPLE9_NATIVE_RING_ROWS - d->ring_rows)
         return false;
   }
   return true;
}
