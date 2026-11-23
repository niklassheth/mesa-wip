/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir_lower_vbo.h"
#include "util/format/u_format.h"

static bool
is_rgb10_a2(const struct util_format_description *desc)
{
   return desc->channel[0].shift == 0 && desc->channel[0].size == 10 &&
          desc->channel[1].shift == 10 && desc->channel[1].size == 10 &&
          desc->channel[2].shift == 20 && desc->channel[2].size == 10 &&
          desc->channel[3].shift == 30 && desc->channel[3].size == 2;
}

static enum pipe_format
agx_vbo_internal_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);

   /* RGB10A2 formats are native for UNORM and unpacked otherwise */
   if (is_rgb10_a2(desc)) {
      if (desc->is_unorm)
         return PIPE_FORMAT_R10G10B10A2_UNORM;
      else
         return PIPE_FORMAT_R32_UINT;
   }

   /* R11G11B10F is native and special */
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return format;

   /* No other non-array formats handled */
   if (!desc->is_array)
      return PIPE_FORMAT_NONE;

   /* Otherwise look at one (any) channel */
   int idx = util_format_get_first_non_void_channel(format);
   if (idx < 0)
      return PIPE_FORMAT_NONE;

   /* We only handle RGB formats (we could do SRGB if we wanted though?) */
   if ((desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB) ||
       (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN))
      return PIPE_FORMAT_NONE;

   /* We have native 8-bit and 16-bit normalized formats */
   struct util_format_channel_description chan = desc->channel[idx];

   if (chan.normalized) {
      if (chan.size == 8)
         return desc->is_unorm ? PIPE_FORMAT_R8_UNORM : PIPE_FORMAT_R8_SNORM;
      else if (chan.size == 16)
         return desc->is_unorm ? PIPE_FORMAT_R16_UNORM : PIPE_FORMAT_R16_SNORM;
   }

   /* Otherwise map to the corresponding integer format */
   switch (chan.size) {
   case 32:
      return PIPE_FORMAT_R32_UINT;
   case 16:
      return PIPE_FORMAT_R16_UINT;
   case 8:
      return PIPE_FORMAT_R8_UINT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

bool
agx_vbo_supports_format(enum pipe_format format)
{
   return agx_vbo_internal_format(format) != PIPE_FORMAT_NONE;
}
