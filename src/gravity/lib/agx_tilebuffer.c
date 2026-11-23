/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include "agx_apple9.h"
#include "util/u_math.h"

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered)
{
   struct agx_tilebuffer_layout tib = {
      .nr_samples = nr_samples,
      .layered = layered,
   };

   unsigned bytes = 0;
   for (unsigned rt = 0; rt < nr_cbufs; ++rt) {
      tib.logical_format[rt] = formats[rt];
      tib._offset_B[rt] = bytes;
      bytes += 4 * agx_apple9_color_words(formats[rt]);
   }

   tib.sample_size_B =
      ALIGN_POT(agx_apple9_tile_sample_bytes(bytes, nr_samples), 8);
   tib.tile_size = nr_samples == 4 ? 32 * 16 : 32 * 32;

   return tib;
}
