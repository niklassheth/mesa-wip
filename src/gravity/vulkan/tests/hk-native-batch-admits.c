/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * hk_apple9_native_stream_admits: hardware-invariant split policy for opaque
 * DRM_ASAHI_RENDER_NATIVE_MESH renders. Ordinary and native draws share one
 * render pass in any order (ordinary -> native -> ordinary -> native) unless
 * the isolation kill switch is active, cumulative ring occupancy for a
 * wg_rows > 40 shader exceeds 4096 rows (64 KiB), or the primitive bound is
 * exceeded. */
#include <stdio.h>
#include <stdlib.h>
#include "hk_apple9_native_batch.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static const struct hk_apple9_native_profile enabled_unbounded = {
   .blocks = true,
   .max_render_primitives = 0,
};
static const struct hk_apple9_native_profile enabled_capped = {
   .blocks = true,
   .max_render_primitives = HK_APPLE9_NATIVE_RENDER_PRIMITIVES,
};
static const struct hk_apple9_native_profile isolated = {
   .blocks = false,
   .max_render_primitives = HK_APPLE9_NATIVE_RENDER_PRIMITIVES,
};

static struct hk_apple9_draw_summary
native_draw(uint64_t prims, uint64_t ring_rows, bool ring_limited)
{
   return (struct hk_apple9_draw_summary){
      .native = true,
      .primitives = prims,
      .ring_rows = ring_rows,
      .ring_limited = ring_limited,
   };
}

static struct hk_apple9_draw_summary
ordinary_draw(uint64_t prims)
{
   return (struct hk_apple9_draw_summary){
      .native = false,
      .primitives = prims,
   };
}

int
main(void)
{
   /* Empty stream always admits first ordinary or native draw, even under the
    * isolation kill switch. */
   struct hk_apple9_stream_summary empty = {0};
   struct hk_apple9_draw_summary d_nat = native_draw(100, 40, false);
   struct hk_apple9_draw_summary d_ord = ordinary_draw(100);
   CHECK(hk_apple9_native_stream_admits(&empty, &d_ord, &isolated));
   CHECK(hk_apple9_native_stream_admits(&empty, &d_nat, &isolated));

   /* Ordinary-leading stream -> first native draw merges with 0 splits when
    * blocks=true, and splits when blocks=false (kill switch). */
   struct hk_apple9_stream_summary ord_stream = {
      .native = false,
      .has_draws = true,
      .primitives = 1000,
   };
   CHECK(hk_apple9_native_stream_admits(&ord_stream, &d_nat, &enabled_capped));
   CHECK(!hk_apple9_native_stream_admits(&ord_stream, &d_nat, &isolated));
   /* Ordinary -> ordinary is never constrained by native-mesh admission. */
   CHECK(hk_apple9_native_stream_admits(&ord_stream, &d_ord, &isolated));

   /* Ordinary -> native -> ordinary -> native all stay in one render pass
    * with no artificial 64-dispatch or 4096-word ceiling. */
   struct hk_apple9_stream_summary s = {
      .native = true,
      .has_draws = true,
      .primitives = 1000,
      .ring_rows = 40,
      .ring_limited = false,
   };
   for (unsigned i = 0; i < 200; ++i) {
      struct hk_apple9_draw_summary d =
         (i % 2 == 0) ? native_draw(100, 40, false) : ordinary_draw(100);
      CHECK(hk_apple9_native_stream_admits(&s, &d, &enabled_capped));
      s.primitives += d.primitives;
      if (d.native)
         s.ring_rows += d.ring_rows;
   }
   CHECK(!hk_apple9_native_stream_admits(&s, &d_nat, &isolated));
   CHECK(!hk_apple9_native_stream_admits(&s, &d_ord, &isolated));

   /* Genuine hardware invariant: 64 KiB (4096-row) ring budget for ring-limited
    * (wg_rows > 40) shaders. */
   s = (struct hk_apple9_stream_summary){
      .native = true,
      .has_draws = true,
      .ring_rows = 3000,
      .ring_limited = true,
   };
   CHECK(hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .ring_rows = 1096, .ring_limited = true}, &enabled_unbounded));
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .ring_rows = 1097, .ring_limited = true}, &enabled_unbounded));
   /* Even an unbounded incoming draw cannot push a ring-limited stream over 4096 rows. */
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .ring_rows = 1097, .ring_limited = false}, &enabled_unbounded));
   /* An ordinary draw uses 0 mesh ring rows and still shares the render pass. */
   CHECK(hk_apple9_native_stream_admits(&s, &d_ord, &enabled_unbounded));

   /* Two unbounded (wg_rows <= 40) draws batch freely past 4096 cumulative rows. */
   s = (struct hk_apple9_stream_summary){
      .native = true,
      .has_draws = true,
      .ring_rows = 100000,
      .ring_limited = false,
   };
   CHECK(hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .ring_rows = 100000, .ring_limited = false}, &enabled_unbounded));
   /* But appending a ring-limited draw onto that stream splits cleanly. */
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .ring_rows = 48, .ring_limited = true}, &enabled_unbounded));

   /* Conservative primitive bound (TVB cap) when max_render_primitives > 0
    * for bounded native renders (s.bounded_prims = true). */
   struct hk_apple9_native_profile b = {.blocks = true, .max_render_primitives = 1000};
   s = (struct hk_apple9_stream_summary){
      .native = true,
      .has_draws = true,
      .bounded_prims = true,
      .primitives = 900,
   };
   CHECK(hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .primitives = 100}, &b));
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .primitives = 101}, &b));
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = false, .primitives = 200}, &b));
   s.primitives = 0;
   CHECK(!hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .primitives = UINT64_MAX}, &b));
   b.max_render_primitives = 0;
   CHECK(hk_apple9_native_stream_admits(&s, &(struct hk_apple9_draw_summary){
      .native = true, .primitives = UINT64_MAX}, &b));

   /* Unbounded (wg_rows <= 40, != 32) multi-parent fused tessellation draws
    * (e.g. 20,000 patches * 8192 worst-case = 163,840,000 > 1,000,000) merge
    * into ordinary and unbounded native passes with 0 splits, and ordinary
    * draws (including indirect) following them also stay in the same pass. */
   const struct hk_apple9_draw_summary giant_tess = {
      .native = true,
      .unbounded_prims = true,
      .primitives = 20000ull * 8192ull,
      .ring_rows = 20000ull * 4096ull * 40ull,
      .ring_limited = false,
   };
   struct hk_apple9_stream_summary ord_indirect = {
      .native = false,
      .has_draws = true,
      .primitives = UINT64_MAX,
   };
   CHECK(hk_apple9_native_stream_admits(&ord_indirect, &giant_tess, &enabled_capped));
   struct hk_apple9_stream_summary unb_stream = {
      .native = true,
      .has_draws = true,
      .bounded_prims = false,
      .primitives = UINT64_MAX,
      .ring_rows = giant_tess.ring_rows,
      .ring_limited = false,
   };
   CHECK(hk_apple9_native_stream_admits(&unb_stream, &giant_tess, &enabled_capped));
   CHECK(hk_apple9_native_stream_admits(&unb_stream, &(struct hk_apple9_draw_summary){
      .native = false, .primitives = UINT64_MAX}, &enabled_capped));
   /* Appending a ring-limited draw onto that unbounded stream still enforces
    * the 64 KiB (4096-row) ring budget and splits cleanly. */
   CHECK(!hk_apple9_native_stream_admits(&unb_stream, &(struct hk_apple9_draw_summary){
      .native = true, .primitives = 100, .ring_rows = 48, .ring_limited = true},
      &enabled_capped));

   /* Both native ordering directions between capped and exempt draws MUST split
    * when combined primitives exceed max_render_primitives:
    * 1) Capped native (900k, bounded_prims=true) + exempt tess (163.84M) -> split!
    * 2) Exempt tess (163.84M accumulated in s.primitives) + capped native (100) -> split! */
   struct hk_apple9_stream_summary capped_stream = {
      .native = true,
      .has_draws = true,
      .bounded_prims = true,
      .primitives = 900000,
      .ring_rows = 40,
      .ring_limited = false,
   };
   CHECK(!hk_apple9_native_stream_admits(&capped_stream, &giant_tess, &enabled_capped));
   struct hk_apple9_stream_summary unb_accum = {
      .native = true,
      .has_draws = true,
      .bounded_prims = false,
      .primitives = giant_tess.primitives,
      .ring_rows = giant_tess.ring_rows,
      .ring_limited = false,
   };
   CHECK(!hk_apple9_native_stream_admits(&unb_accum, &(struct hk_apple9_draw_summary){
      .native = true, .unbounded_prims = false, .primitives = 100,
      .ring_rows = 40, .ring_limited = false}, &enabled_capped));

   /* M2: Standalone admission distinguishes REJECT (cannot fit alone even in
    * a fresh empty stream) from SPLIT (fits alone, must start a fresh stream). */
   struct hk_apple9_draw_summary oversized_capped =
      native_draw(HK_APPLE9_NATIVE_RENDER_PRIMITIVES + 1, 40, false);
   CHECK(!hk_apple9_native_draw_fits_alone(&oversized_capped, &enabled_capped));
   CHECK(hk_apple9_native_stream_classify(&empty, &oversized_capped,
                                          &enabled_capped) ==
         HK_APPLE9_NATIVE_ADMIT_REJECT);
   CHECK(hk_apple9_native_draw_fits_alone(&oversized_capped, &enabled_unbounded));
   struct hk_apple9_draw_summary oversized_ring =
      native_draw(10, HK_APPLE9_NATIVE_RING_ROWS + 1, true);
   CHECK(!hk_apple9_native_draw_fits_alone(&oversized_ring, &enabled_unbounded));
   CHECK(hk_apple9_native_stream_classify(&empty, &oversized_ring,
                                          &enabled_unbounded) ==
         HK_APPLE9_NATIVE_ADMIT_REJECT);
   struct hk_apple9_draw_summary fits_alone =
      native_draw(200000, 40, false);
   CHECK(hk_apple9_native_draw_fits_alone(&fits_alone, &enabled_capped));
   CHECK(hk_apple9_native_stream_classify(&capped_stream, &fits_alone,
                                          &enabled_capped) ==
         HK_APPLE9_NATIVE_ADMIT_SPLIT);
   CHECK(hk_apple9_native_stream_classify(&empty, &fits_alone,
                                          &enabled_capped) ==
         HK_APPLE9_NATIVE_ADMIT_APPEND);

   printf("PASS\n");
   return 0;
}
