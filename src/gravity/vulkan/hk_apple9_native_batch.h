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

static inline uint32_t
hk_apple9_native_gcd_u32(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t t = a % b;
      a = b;
      b = t;
   }
   return a;
}

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
    * render, whether any block is ring-limited (lacks a wrap-safe U0_B plan),
    * the running GCD of wrap-safe strides across all non-empty native blocks
    * (0: none yet / incompatible), the worst-case single-WG live rows across
    * those blocks, and whether all non-empty blocks share one exact constant
    * wrap-safe stride. */
   uint64_t ring_rows;
   uint32_t ring_stride_gcd;
   uint32_t ring_max_live_rows;
   bool ring_stride_constant;
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
    * unknown), the GCD of its reachable wrap-safe strides (0 when bounded or
    * unknown), its worst-case single-WG live rows, whether every WG in this
    * draw uses one constant stride, and whether its layout is ring-limited. */
   uint64_t ring_rows;
   uint32_t ring_stride_gcd;
   uint32_t ring_live_rows;
   bool ring_stride_constant;
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
   /* Check whether combined native ring occupancy exceeds 4,096 rows (wraps
    * the 64 KiB mesh output ring). Ring-limited layouts may never wrap.
    * Wrap-safe layouts may share a wrapping render pass whenever EITHER:
    * 1) All strides share a power-of-2 common divisor G | 4096 with
    *    G >= max(s->ring_max_live_rows, d->ring_live_rows), so the ring
    *    pointer always remains 0 mod G and leaves >= G >= live_rows before
    *    row 4,096 regardless of draw/bucket interleaving; OR
    * 2) Every draw in the pass shares the exact same constant wrap-safe stride
    *    (each individually proven 4096 % stride == 0 || >= live_rows). */
   const bool wraps_ring =
      d->ring_rows > HK_APPLE9_NATIVE_RING_ROWS ||
      s->ring_rows > HK_APPLE9_NATIVE_RING_ROWS - d->ring_rows;
   if (s->ring_limited || d->ring_limited) {
      if (wraps_ring)
         return HK_APPLE9_NATIVE_ADMIT_SPLIT;
   } else if (s->native && s->ring_rows > 0 && d->ring_rows > 0 && wraps_ring) {
      if (!s->ring_stride_gcd || !d->ring_stride_gcd)
         return HK_APPLE9_NATIVE_ADMIT_SPLIT;
      const uint32_t g =
         hk_apple9_native_gcd_u32(s->ring_stride_gcd, d->ring_stride_gcd);
      const uint32_t g_pow2 = g & (0u - g);
      const uint32_t g_ring =
         g_pow2 < HK_APPLE9_NATIVE_RING_ROWS ? g_pow2 : HK_APPLE9_NATIVE_RING_ROWS;
      const uint32_t max_live =
         s->ring_max_live_rows > d->ring_live_rows
            ? s->ring_max_live_rows
            : d->ring_live_rows;
      const bool pow2_safe = max_live > 0 && g_ring >= max_live;
      const bool same_const =
         s->ring_stride_constant && d->ring_stride_constant &&
         s->ring_stride_gcd == d->ring_stride_gcd;
      if (!pow2_safe && !same_const)
         return HK_APPLE9_NATIVE_ADMIT_SPLIT;
   }
   return HK_APPLE9_NATIVE_ADMIT_APPEND;
}

static inline bool
hk_apple9_native_stream_admits(const struct hk_apple9_stream_summary *s,
                               const struct hk_apple9_draw_summary *d,
                               const struct hk_apple9_native_profile *p)
{
   return hk_apple9_native_stream_classify(s, d, p) ==
          HK_APPLE9_NATIVE_ADMIT_APPEND;
}
