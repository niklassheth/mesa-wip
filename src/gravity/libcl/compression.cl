/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "compiler/nir/nir_defines.h"
#include "compiler/shader_enums.h"
#include "agx_pack.h"
#include "libagx_intrinsics.h"

/*
 * The metadata buffer is fully twiddled, so interleave the X/Y coordinate bits.
 * While dimensions are padded to powers-of-two, they are not padded to a
 * square. If the width is more than 2x the height or vice versa, the additional
 * bits are linear. So we interleave as much as possible, and then add what's
 * remaining. Finally, layers are strided linear and added at the end.
 */
static uint
index_metadata(uint3 c, uint width, uint height, uint layer_stride)
{
   uint major_coord = width > height ? c.x : c.y;
   uint minor_dim = min(width, height);

   uint intl_bits = util_logbase2_ceil(minor_dim);
   uint intl_mask = (1 << intl_bits) - 1;
   uint2 intl_coords = c.xy & intl_mask;

   return nir_interleave(intl_coords.x, intl_coords.y) +
          ((major_coord & ~intl_mask) << intl_bits) + (layer_stride * c.z);
}

static ushort2
stretched_sa_to_px(ushort2 px, uint samples)
{
   /* clang-format off */
   switch (samples) {
   case  4: return (ushort2)(px.x / 2, px.y / 2);
   case  2: return (ushort2)(px.x, px.y / 2);
   case  1: return px;
   default: UNREACHABLE("invalid sample count");
   }
   /* clang-format on */
}

/*
 * Compute kernel to clear a compressed image to a solid colour by filling the
 * compression metadata to all solid and writing the colour to each tile.
 *
 * Invoked as <width in 16x16 tiles, height in 16x16 tiles, layers>. Each
 * invocation clears a single 16x16 compression tile.
 *
 * Despite the ALU complexity, this is twice as fast as clearing with the
 * 3D pipe for a few reasons.
 *
 * 1. At a kernel/firmware level, CDM submission is much faster than VDM.
 * 2. Far fewer threads are dispatched (1 per 16x16 tile, rather than 1 per
 *    pixel like the background program used to implement 3D clears).
 * 3. No useless tilebuffer load/stores.
 * 4. Many CDM dispatches can share a single control stream, whereas a VDM
 *    control stream covers only 1 render pass.
 *
 * AGX compression is not fully understood, but we know enough to clear solid
 * colours:
 *
 * - Metadata buffer has one byte per 8x4 subtile, grouped into 16x16 tiles with
 *   8 bytes of metadata.
 * - The 8x4 "solid mode" will read its colour from the beginning of the tile in
 *   the body buffer, addressed as-if twiddled. To clear the whole tile, we need
 *   (16x16) / (8x4) = 8 copies of the colour per 16x16 tile.
 *
 * Also, multisampling requires stretching the coordinates. This corresponds to
 * ail_effective_width_sa and ail_effective_height_sa. With multisampling, we
 * are invoked on tiles of 16x16 samples, so we need to squish back to tiles of
 * 16x8 or 8x8 pixels.
 */
KERNEL(32)
libagx_fast_clear(global uint2 *meta, global uint4 *body, uint width_tl,
                  uint height_tl, uint tile_w_el, uint tile_h_el,
                  uint meta_layer_stride_tl, uint body_layer_stride_uint4,
                  uint blocksize_B_log2__5, uint samples_log2__3, uint a,
                  uint b, uint c, uint d, uint solid_mode)
{
   uint samples = 1 << samples_log2__3;
   uint blocksize_B = 1 << blocksize_B_log2__5;

   /* Set the tile to be solid */
   meta[index_metadata(cl_global_id, width_tl, height_tl, meta_layer_stride_tl)] =
      solid_mode;

   /* SWAR vectorized multiply by 16, we know this won't overflow */
   ushort2 coord_el =
      as_ushort2(as_uint(convert_ushort2(cl_global_id.xy)) << 4);

   /* Calculate the offset of the tile in the body buffer */
   ushort2 dim_sa = (ushort2)(width_tl, height_tl) * (ushort)16;
   uint aligned_width_px =
      align(stretched_sa_to_px(dim_sa, samples).x, tile_w_el);

   ushort2 coord_px = stretched_sa_to_px(coord_el, samples);
   uint twiddled = libagx_twiddle_coordinates(coord_px, tile_w_el, tile_h_el,
                                              aligned_width_px);

   /* We scale blocksize_B by # of samples to make addressing work properly and
    * to splat the clear colour across the samples. See the corresponding scale
    * in libagx_image_texel_address.
    *
    * TODO: Check that this is actually right? Passes CTS at least.
    */
   blocksize_B *= samples;
   uint body_offs = (twiddled * blocksize_B) / sizeof(uint4);
   body_offs += cl_global_id.z * body_layer_stride_uint4;

   /* Write 8 copies of the clear colour. We vectorize with 128-bit writes,
    * unless 8 copies would be less than 128-bit which only occurs for 8-bit
    * formats where we write only 64-bit.
    */
   if (blocksize_B == 1) {
      *(global uint2 *)(body + body_offs) = (uint2)(a, b);
   } else {
      for (uint i = 0; i < (8 * blocksize_B) / 16; ++i) {
         body[body_offs + i] = (uint4)(a, b, c, d);
      }
   }
}
