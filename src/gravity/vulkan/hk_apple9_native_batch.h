/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Whether a draw may share an open graphics render pass that already contains
 * (or would acquire) a native Apple9 object/mesh dispatch. With opaque
 * DRM_ASAHI_RENDER_NATIVE_MESH (bit 28), the kernel does not parse VDM streams
 * on the CPU, so only genuine hardware invariants (cumulative 64 KiB mesh
 * output ring occupancy for wg_rows > 40 shaders, optional conservative TVB
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
   /* Estimated binned primitives already in the render. */
   uint64_t primitives;
   /* Cumulative hardware 64 KiB mesh output ring rows of native blocks in the
    * render, and whether any block had wg_rows > 40 (ring-limited). */
   uint64_t ring_rows;
   bool ring_limited;
};

struct hk_apple9_draw_summary {
   bool native;
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
   /* An empty stream always starts cleanly; an ordinary stream receiving an
    * ordinary draw is never constrained by native-mesh admission. */
   if (!s->native && !(s->has_draws && d->native))
      return true;
   if (!p->blocks)
      return false;
   if (p->max_render_primitives &&
       (d->primitives > p->max_render_primitives ||
        s->primitives > p->max_render_primitives - d->primitives))
      return false;
   if (!d->native)
      return true;
   /* When either the open render or the incoming native block has wg_rows > 40,
    * enforce that combined ring occupancy stays within the 64 KiB (4096-row)
    * mesh output ring without assuming an implicit cross-block hardware drain. */
   if (s->ring_limited || d->ring_limited) {
      if (d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS ||
          s->ring_rows > HK_APPLE9_NATIVE_RING_ROWS - d->ring_rows)
         return false;
   }
   return true;
}
