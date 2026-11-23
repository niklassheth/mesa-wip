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
   /* Logical format of each render target. */
   enum pipe_format logical_format[AGX_MAX_RENDER_TARGETS];

   /* Offset into the sample of each render target. */
   uint8_t _offset_B[AGX_MAX_RENDER_TARGETS];

   /* Total bytes per sample, rounded up to 8-byte alignment. */
   uint8_t sample_size_B;

   /* Number of samples per pixel */
   uint8_t nr_samples;

   /* If layered rendering is used */
   bool layered;

   /* Selected tile size in pixels. Either 32*32 or 32*16. */
   uint16_t tile_size;
};

static inline void
agx_tilebuffer_set_drm_cmd(struct drm_asahi_cmd_render *c,
                           const struct agx_tilebuffer_layout *layout)
{
   c->utile_width_px = layout->tile_size >= 32 * 16 ? 32 : 16;
   c->utile_height_px = layout->tile_size <= 32 * 16 ? 16 : 32;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered);

bool agx_nir_lower_to_per_sample(struct nir_shader *shader);

bool agx_nir_lower_monolithic_msaa(struct nir_shader *shader,
                                   uint8_t nr_samples);

bool agx_nir_lower_sample_intrinsics(struct nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);

#ifdef __cplusplus
} /* extern C */
#endif
