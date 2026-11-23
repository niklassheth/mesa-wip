/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "util/format/u_formats.h"
#include "agx_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum render targets per framebuffer. This is NOT architectural, but it
 * is the ~universal API limit so there's no point in allowing more.
 */
#define AGX_MAX_RENDER_TARGETS (8)

/* Forward declarations to keep the header lean */
struct nir_shader;
struct nir_def;
struct nir_builder;

struct agx_tilebuffer_layout {
   /* Logical format of each render target. Use agx_tilebuffer_physical_format
    * to get the physical format.
    */
   enum pipe_format logical_format[AGX_MAX_RENDER_TARGETS];

   /* Which render targets are spilled. */
   bool spilled[AGX_MAX_RENDER_TARGETS];

   /* Offset into the sample of each render target. If a render target is
    * spilled, its offset is UNDEFINED. Use agx_tilebuffer_offset_B to access.
    */
   uint8_t _offset_B[AGX_MAX_RENDER_TARGETS];

   /* Total bytes per sample, rounded up as needed. Spilled render targets do
    * not count against this.
    */
   uint8_t sample_size_B;

   /* Number of samples per pixel */
   uint8_t nr_samples;

   /* If layered rendering is used */
   bool layered;

   /* Selected tile size in pixels. Either 32*32, 32*16, or 16*16. */
   uint16_t tile_size;
};

static inline void
agx_tilebuffer_set_drm_cmd(struct drm_asahi_cmd_render *c,
                           const struct agx_tilebuffer_layout *layout)
{
   c->utile_width_px = layout->tile_size >= 32 * 16 ? 32 : 16;
   c->utile_height_px = layout->tile_size <= 32 * 16 ? 16 : 32;
}

static inline bool
agx_tilebuffer_spills(struct agx_tilebuffer_layout *layout)
{
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (layout->spilled[rt])
         return true;
   }

   return false;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered);

bool agx_nir_lower_to_per_sample(struct nir_shader *shader);

bool agx_nir_lower_monolithic_msaa(struct nir_shader *shader,
                                   uint8_t nr_samples);

bool agx_nir_lower_sample_intrinsics(struct nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);

uint32_t agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib);

#ifdef __cplusplus
} /* extern C */
#endif
