/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/libcl/libcl.h"

uint32_t nir_interleave(uint16_t x, uint16_t y);

uint4 nir_bindless_image_load(uint handle, int4 coord, uint sample, uint lod,
                              uint image_dim, uint image_array, uint format,
                              uint access, uint dest_type);

void nir_bindless_image_store(uint handle, int4 coord, uint sample, uint4 datum,
                              uint lod, uint image_dim, uint image_array,
                              uint format, uint access, uint src_type);

uint nir_bindless_image_agx(uint offset_B, uint uniform);

uint32_t libagx_twiddle_coordinates(ushort2 coord, uint16_t tile_w_px,
                                    uint16_t tile_h_px,
                                    uint32_t aligned_width_px);
