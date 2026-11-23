/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include "gravity/layout/layout.h"
#include "agx_abi.h"
#include "agx_border.h"
#include "agx_pack.h"
#include "libagx_shaders.h"

/* Channels agree for RGBA but are weird for force 0/1 */

static inline enum agx_channel
agx_channel_from_pipe(enum pipe_swizzle in)
{
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_X == AGX_CHANNEL_R);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_Y == AGX_CHANNEL_G);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_Z == AGX_CHANNEL_B);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_W == AGX_CHANNEL_A);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_0 & 0x4);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_1 & 0x4);
   STATIC_ASSERT((enum agx_channel)PIPE_SWIZZLE_NONE & 0x4);

   if ((in & 0x4) == 0)
      return (enum agx_channel)in;
   else if (in == PIPE_SWIZZLE_1)
      return AGX_CHANNEL_1;
   else
      return AGX_CHANNEL_0;
}

static inline enum agx_layout
agx_translate_layout(enum ail_tiling tiling)
{
   switch (tiling) {
   case AIL_TILING_GPU:
      return AGX_LAYOUT_GPU;
   case AIL_TILING_TWIDDLED:
      return AGX_LAYOUT_TWIDDLED;
   case AIL_TILING_LINEAR:
      return AGX_LAYOUT_LINEAR;
   }

   UNREACHABLE("Invalid tiling");
}

static inline enum agx_zls_tiling
agx_translate_zls_tiling(enum ail_tiling tiling)
{
   switch (tiling) {
   case AIL_TILING_GPU:
      return AGX_ZLS_TILING_GPU;
   case AIL_TILING_TWIDDLED:
      return AGX_ZLS_TILING_TWIDDLED;
   default:
      UNREACHABLE("Invalid ZLS tiling");
   }
}

struct agx_zls {
   bool z_load, z_store;
   bool s_load, s_store;
};

static inline void
agx_pack_zls_control(struct agx_zls_control_packed *packed,
                     const struct ail_layout *z, const struct ail_layout *s,
                     struct agx_zls *args)
{
   agx_pack(packed, ZLS_CONTROL, cfg) {
      if (z) {
         cfg.z_store = args->z_store;
         cfg.z_load = args->z_load;
         cfg.z_load_compress = cfg.z_store_compress = z->compressed;
         cfg.z_load_tiling = cfg.z_store_tiling =
            agx_translate_zls_tiling(z->tiling);

         if (z->format == PIPE_FORMAT_Z16_UNORM) {
            cfg.z_format = AGX_ZLS_FORMAT_16;
         } else {
            cfg.z_format = AGX_ZLS_FORMAT_32F;
         }
      }

      if (s) {
         cfg.s_load = args->s_load;
         cfg.s_store = args->s_store;
         cfg.s_load_compress = cfg.s_store_compress = s->compressed;
         cfg.s_load_tiling = cfg.s_store_tiling =
            agx_translate_zls_tiling(s->tiling);
      }
   }
}

static enum agx_sample_count
agx_translate_sample_count(unsigned samples)
{
   switch (samples) {
   case 2:
      return AGX_SAMPLE_COUNT_2;
   case 4:
      return AGX_SAMPLE_COUNT_4;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static inline uint32_t
agx_pack_line_width(float line_width)
{
   /* Line width is packed in a 4:4 fixed point format */
   unsigned line_width_fixed = ((unsigned)(line_width * 16.0f)) - 1;

   /* Clamp to maximum line width */
   return MIN2(line_width_fixed, 0xFF);
}

/*
 * Despite having both a layout *and* a flag that I only see Metal use with null
 * textures, AGX doesn't seem to have "real" null textures. Instead we need to
 * bind an arbitrary address and throw away the results to read all 0's.
 * Accordingly, the caller must pass some address that lives at least as long as
 * the texture descriptor itself.
 */
static void
agx_set_null_texture(struct agx_texture_packed *tex)
{
   agx_pack(tex, TEXTURE, cfg) {
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.channels = AGX_CHANNELS_R8;
      cfg.type = AGX_TEXTURE_TYPE_UNORM /* don't care */;
      cfg.swizzle_r = AGX_CHANNEL_0;
      cfg.swizzle_g = AGX_CHANNEL_0;
      cfg.swizzle_b = AGX_CHANNEL_0;
      cfg.swizzle_a = AGX_CHANNEL_0;
      cfg.address = AGX_ZERO_PAGE_ADDRESS;
   }
}

static void
agx_set_null_pbe(struct agx_pbe_packed *pbe)
{
   agx_pack(pbe, PBE, cfg) {
      cfg.width = 1;
      cfg.height = 1;
      cfg.levels = 1;
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.channels = AGX_CHANNELS_R8;
      cfg.type = AGX_TEXTURE_TYPE_UNORM /* don't care */;
      cfg.swizzle_r = AGX_CHANNEL_R;
      cfg.swizzle_g = AGX_CHANNEL_R;
      cfg.swizzle_b = AGX_CHANNEL_R;
      cfg.swizzle_a = AGX_CHANNEL_R;
      cfg.buffer = AGX_SCRATCH_PAGE_ADDRESS;
   }
}

static struct libagx_decompress_args
agx_fill_decompress_args(struct ail_layout *layout, unsigned layer,
                         unsigned level, uint64_t ptr, uint64_t images)
{
   return (struct libagx_decompress_args){
      .images = images,
      .tile_uncompressed = ail_tile_mode_uncompressed(layout->format),
      .metadata = ptr + layout->metadata_offset_B +
                  layout->level_offsets_compressed_B[level] +
                  (layer * layout->compression_layer_stride_B),
      .metadata_layer_stride_tl = layout->compression_layer_stride_B / 8,
      .metadata_width_tl = ail_metadata_width_tl(layout, level),
      .metadata_height_tl = ail_metadata_height_tl(layout, level),
   };
}

#undef libagx_decompress
#define libagx_decompress(context, grid, barrier, layout, layer, level, ptr,   \
                          images)                                              \
   libagx_decompress_struct(                                                   \
      context, grid, barrier,                                                  \
      agx_fill_decompress_args(layout, layer, level, ptr, images),             \
      util_logbase2(layout->sample_count_sa))
