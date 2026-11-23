/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include <assert.h>
#include "util/format/u_format.h"
#include "layout.h"

/* Maximum number of bytes per tile on G13G. This may change in future versions
 * of the architecture.
 */
#define MAX_BYTES_PER_TILE (32768 - 1)

/* Maximum bytes per sample in the tilebuffer. Greater allocations require
 * spilling render targets to memory.
 */
#define MAX_BYTES_PER_SAMPLE (64)

/* Minimum tile size in pixels, architectural. */
#define MIN_TILE_SIZE_PX (16 * 16)

/* Select the largest tile size that fits */
static uint16_t
agx_select_tile_size(unsigned px_size_B)
{
   assert(px_size_B <= (MAX_BYTES_PER_TILE / MIN_TILE_SIZE_PX));

   return ((px_size_B * 32 * 32) <= MAX_BYTES_PER_TILE)   ? (32 * 32)
          : ((px_size_B * 32 * 16) <= MAX_BYTES_PER_TILE) ? (32 * 16)
                                                          : MIN_TILE_SIZE_PX;
}

static inline unsigned
format_align_B(enum pipe_format format)
{
   /* For some reason util_format_get_blocksize(NONE) = 1 */
   enum pipe_format phys = ail_pixel_format[format].renderable;
   return (format != PIPE_FORMAT_NONE) ? util_format_get_blocksize(phys) : 0;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered)
{
   struct agx_tilebuffer_layout tib = {
      .nr_samples = nr_samples,
      .layered = layered,
   };

   uint32_t offset_B = 0;
   uint8_t order[] = {0, 1, 2, 3, 4, 5, 6, 7};

   /* Sort render targets in descending order of alignment, eliminating padding
    * and giving the optimal order of render targets. We use insertion sort
    * because it is simple, stable, fast for small n, and free for n=1.
    */
   for (int i = 1; i < nr_cbufs; ++i) {
      for (int j = i; j > 0 && format_align_B(formats[order[j - 1]]) <
                                  format_align_B(formats[order[j]]);
           --j) {
         SWAP(order[j], order[j - 1]);
      }
   }

   for (unsigned i = 0; i < nr_cbufs; ++i) {
      unsigned rt = order[i];
      enum pipe_format format = formats[rt];
      tib.logical_format[rt] = format;

      assert(util_is_aligned(offset_B, MAX2(format_align_B(formats[rt]), 1)) &&
             "loop invariant ensured by the sort");

      unsigned size_B = format_align_B(format);
      enum pipe_format phys = ail_pixel_format[format].renderable;
      if (util_format_get_nr_components(phys) == 1) {
         size_B *= util_format_get_nr_components(format);
      }

      /* If allocating this render target would exceed any tilebuffer limits, we
       * need to spill it to memory. Otherwise, allocate it to the tilebuffer.
       */
      unsigned new_offset_B = offset_B + size_B;
      bool fits = (new_offset_B <= MAX_BYTES_PER_SAMPLE) &&
                  (ALIGN_POT(new_offset_B, 8) * MIN_TILE_SIZE_PX *
                   nr_samples) <= MAX_BYTES_PER_TILE;
      if (fits) {
         tib._offset_B[rt] = offset_B;
         offset_B = new_offset_B;
      } else {
         tib.spilled[rt] = true;
      }
   }

   assert(offset_B <= MAX_BYTES_PER_SAMPLE && "loop invariant");

   /* Multisampling needs a nonempty allocation.
    * XXX: Check this against hw
    */
   if (nr_samples > 1)
      offset_B = MAX2(offset_B, 1);

   tib.sample_size_B = align(offset_B, 8);
   tib.tile_size = agx_select_tile_size(tib.sample_size_B * nr_samples);

   return tib;
}

uint32_t
agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib)
{
   return tib->sample_size_B * tib->nr_samples * tib->tile_size;
}
