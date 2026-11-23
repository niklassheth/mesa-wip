/*
 * Copyright 2023 Valve Corporation
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "glsl_types.h"
#include "libagx.h"
#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* Residency flags are inverted from NIR */
#define AGX_RESIDENT (0)

static bool
fence_image(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_after_instr(&intr->instr);

   /* If the image is write-only, there is no fencing needed */
   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
      return false;
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      nir_fence_pbe_to_tex_agx(b);
      return true;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic_swap:
      nir_fence_mem_to_tex_agx(b);
      return true;

   default:
      return false;
   }
}

static nir_def *
texture_descriptor_ptr(nir_builder *b, nir_tex_instr *tex)
{
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   assert(handle_idx >= 0 && "must be bindless");
   return nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
}

static bool
lower_tex_crawl(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if ((tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples &&
        tex->op != nir_texop_query_levels) ||
       (tex->backend_flags & AGX_TEXTURE_FLAG_METADATA_QUERY))
      return false;

   nir_def *ptr = texture_descriptor_ptr(b, tex);
   unsigned nr_comps = tex->def.num_components;
   assert(nr_comps <= 3);

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod = lod_idx >= 0 ? nir_u2u16(b, tex->src[lod_idx].src.ssa)
                               : nir_imm_intN_t(b, 0, 16);

   nir_def *res;
   if (tex->op == nir_texop_txs) {
      res =
         libagx_txs(b, ptr, lod, nir_imm_int(b, nr_comps),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_BUF),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
                    nir_imm_bool(b, tex->is_array));
   } else if (tex->op == nir_texop_query_levels) {
      res = libagx_texture_levels(b, ptr);
   } else {
      res = libagx_texture_samples(b, ptr);
   }

   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, res, nr_comps));
   nir_instr_remove(instr);
   return true;
}

/*
 * Given a 1D buffer texture coordinate, calculate the 2D coordinate vector that
 * will be used to access the linear 2D texture bound to the buffer.
 */
static nir_def *
coords_for_buffer_texture(nir_builder *b, nir_def *coord)
{
   return nir_vec2(b, nir_umod_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH),
                   nir_udiv_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH));
}

static bool
legalize_image_lod(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_src *src;

#define CASE(op, idx)                                                          \
   case nir_intrinsic_##op:                                                    \
   case nir_intrinsic_bindless_##op:                                           \
      src = &intr->src[idx];                                                   \
      break;

   switch (intr->intrinsic) {
      CASE(image_load, 3)
      CASE(image_sparse_load, 3)
      CASE(image_store, 4)
      CASE(image_size, 1)
   default:
      return false;
   }

#undef CASE

   if (src->ssa->bit_size == 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(src, nir_i2i16(b, src->ssa));
   return true;
}

static nir_def *
txs_for_image(nir_builder *b, nir_intrinsic_instr *intr,
              unsigned num_components, unsigned bit_size, bool query_samples)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *lod = query_samples ? NULL : intr->src[1].ssa;
   nir_texop op = query_samples ? nir_texop_texture_samples : nir_texop_txs;

   nir_def *res =
      nir_build_tex(b, op, .texture_handle = intr->src[0].ssa, .lod = lod,
                    .dim = dim, .is_array = nir_intrinsic_image_array(intr),
                    .can_speculate = nir_instr_can_speculate(&intr->instr));

   /* Cube images are implemented as 2D arrays, so we need to divide here. */
   if (dim == GLSL_SAMPLER_DIM_CUBE && res->num_components > 2 &&
       !query_samples) {
      nir_def *divided = nir_udiv_imm(b, nir_channel(b, res, 2), 6);
      res = nir_vector_insert_imm(b, res, divided, 2);
   }

   return res;
}

static nir_def *
image_texel_address(nir_builder *b, nir_intrinsic_instr *intr,
                    bool return_index)
{
   /* First, calculate the address of the PBE descriptor */
   nir_def *desc_address =
      nir_load_from_texture_handle_agx(b, intr->src[0].ssa);

   nir_def *coord = intr->src[1].ssa;

   /* For atomics, we always infer the format. We only go down this path with
    * formatless intrinsics when lowering multisampled image stores, but that
    * uses the return_index path that ignores the block size.
    */
   enum pipe_format format = nir_intrinsic_format(intr);
   assert(return_index || format != PIPE_FORMAT_NONE);

   nir_def *blocksize_B = nir_imm_int(b, util_format_get_blocksize(format));

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool layered = nir_intrinsic_image_array(intr) ||
                  (dim == GLSL_SAMPLER_DIM_CUBE) ||
                  (dim == GLSL_SAMPLER_DIM_3D);

   if (dim == GLSL_SAMPLER_DIM_BUF && return_index) {
      return nir_channel(b, coord, 0);
   } else if (dim == GLSL_SAMPLER_DIM_BUF) {
      return libagx_buffer_texel_address(b, desc_address, coord, blocksize_B);
   } else {
      return libagx_image_texel_address(
         b, desc_address, coord, nir_u2u32(b, intr->src[2].ssa), blocksize_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index));
   }
}

static void
lower_buffer_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord_vector = intr->src[1].ssa;
   nir_def *coord = nir_channel(b, coord_vector, 0);

   /* If we're not bindless, assume we don't need an offset (GL driver) */
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_texture_offset(b, desc, coord);
   } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_image_offset(b, desc, coord);
   }

   /* Lower the buffer load/store to a 2D image load/store, matching the 2D
    * texture/PBE descriptor the driver supplies for buffer images.
    */
   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

static void
lower_1d_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord = intr->src[1].ssa;
   bool is_array = nir_intrinsic_image_array(intr);
   nir_def *zero = nir_imm_intN_t(b, 0, coord->bit_size);

   if (is_array) {
      assert(coord->num_components >= 2);
      coord =
         nir_vec3(b, nir_channel(b, coord, 0), zero, nir_channel(b, coord, 1));
   } else {
      assert(coord->num_components >= 1);
      coord = nir_vec2(b, coord, zero);
   }

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

/*
 * Just like for txf, we need special handling around layers (and LODs, but we
 * don't support mipmapped images yet) for robust image_loads. See
 * libagx_lower_txf_robustness for more info.
 */
static bool
lower_image_load_robustness(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS)
      return false;

   /* We only need to worry about array-like loads */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if (!nir_intrinsic_image_array(intr) && dim != GLSL_SAMPLER_DIM_CUBE)
      return false;

   /* Determine the coordinate component of the layer. Cubes and cube arrays
    * keep their array in their last non-array coordinate component, other
    * arrays are immediately after.
    */
   unsigned lidx = glsl_get_sampler_dim_coordinate_components(dim);
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      lidx--;

   nir_def *coord = intr->src[1].ssa;
   nir_def *lod = nir_undef(b, 1, 16);
   nir_def *layer = nir_channel(b, coord, lidx);

   /* image_load is effectively the same as txf, reuse the txf lower */
   nir_def *replaced = libagx_lower_txf_robustness(
      b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
      nir_imm_bool(b, false /* lower LOD */), lod,
      nir_imm_bool(b, false /* lower min LOD */), lod,
      nir_imm_bool(b, true /* lower layer */), layer, nir_channel(b, coord, 0));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, replaced, 0));
   return true;
}

static bool
lower_images(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store: {
      /* Legalize MSAA index */
      nir_src_rewrite(&intr->src[2], nir_u2u16(b, intr->src[2].ssa));

      if (intr->intrinsic == nir_intrinsic_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {
         lower_image_load_robustness(b, intr);
      }

      switch (nir_intrinsic_image_dim(intr)) {
      case GLSL_SAMPLER_DIM_1D:
         lower_1d_image(b, intr);
         return true;

      case GLSL_SAMPLER_DIM_BUF:
         lower_buffer_image(b, intr);
         return true;

      default:
         return true;
      }
   }

   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
      nir_def_rewrite_uses(
         &intr->def,
         txs_for_image(
            b, intr, intr->def.num_components, intr->def.bit_size,
            intr->intrinsic == nir_intrinsic_bindless_image_samples));
      return true;

   case nir_intrinsic_bindless_image_texel_address:
      nir_def_rewrite_uses(&intr->def, image_texel_address(b, intr, false));
      return true;

   case nir_intrinsic_is_sparse_texels_resident:
      /* Residency information is in bit 0, so we need to mask. Unclear what's
       * in the upper bits. For now, let's match the blob.
       */
      nir_def_replace(
         &intr->def,
         nir_ieq_imm(b, nir_iand_imm(b, intr->src[0].ssa, 1), AGX_RESIDENT));
      return true;

   case nir_intrinsic_sparse_residency_code_and:
      /* ior because residency codes are inverted from NIR */
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;

   case nir_intrinsic_image_size:
   case nir_intrinsic_image_texel_address:
      UNREACHABLE("should've been lowered");

   default:
      return false;
   }
}

/*
 * Map out-of-bounds storage texel buffer accesses and multisampled image stores
 * to -1 indices, which will become an out-of-bounds hardware access. This gives
 * cheap robustness2.
 */
static bool
lower_robustness(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
      break;
   default:
      return false;
   }

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool array = nir_intrinsic_image_array(intr);
   unsigned size_components = nir_image_intrinsic_coord_components(intr);

   nir_def *deref = intr->src[0].ssa;
   nir_def *coord = intr->src[1].ssa;

   if (dim != GLSL_SAMPLER_DIM_BUF &&
       !(dim == GLSL_SAMPLER_DIM_MS &&
         intr->intrinsic == nir_intrinsic_image_deref_store))
      return false;

   /* Bounds check the coordinate */
   nir_def *size =
      nir_image_deref_size(b, size_components, 32, deref, nir_imm_int(b, 0),
                           .image_dim = dim, .image_array = array);
   nir_def *oob = nir_bany(b, nir_uge(b, coord, size));

   /* Bounds check the sample */
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, deref, .image_dim = dim,
                                                 .image_array = array);

      oob = nir_ior(b, oob, nir_uge(b, intr->src[2].ssa, samples));
   }

   /* Replace the last coordinate component with a large coordinate for
    * out-of-bounds. We pick 0xFFF0 as it fits in 16-bit, and it is not signed
    * as 32-bit so we won't get in-bounds coordinates for arrays due to two's
    * complement wraparound. Additionally it still meets this requirement after
    * adding 0xF, the maximum tail offset.
    *
    * This ensures the resulting hardware coordinate is definitely
    * out-of-bounds, giving hardware-level robustness2 behaviour.
    */
   /* A buffer coordinate indexes elements, and a view can have more than
    * 0xFFF0 of them. Where buffer images are bounds-checked in software
    * against the element count, as on Apple9, use the largest index instead.
    */
   const bool *software_buffer_bounds = data;
   unsigned c = size_components - 1;
   unsigned oob_coord =
      dim == GLSL_SAMPLER_DIM_BUF && *software_buffer_bounds ? 0xFFFFFFFF : 0xFFF0;
   nir_def *r =
      nir_bcsel(b, oob, nir_imm_int(b, oob_coord), nir_channel(b, coord, c));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, r, c));
   return true;
}

/*
 * Early texture lowering passes, called by the driver before lowering
 * descriptor bindings. That means these passes operate on texture derefs.
 */
bool
agx_nir_lower_texture_early(nir_shader *s, bool support_lod_bias,
                            bool software_buffer_bounds)
{
   bool progress = false;

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_robustness,
            nir_metadata_control_flow, &software_buffer_bounds);

   nir_lower_tex_options lower_tex_options = {
      .lower_txp = ~0,
      .lower_invalid_implicit_lod = true,
      .lower_tg4_offsets = true,
      .lower_index_to_offset = true,
      .lower_sampler_lod_bias = support_lod_bias,

      /* Unclear if/how mipmapped 1D textures work in the hardware. */
      .lower_1d = true,

      /* XXX: Metal seems to handle just like 3D txd, so why doesn't it work?
       * TODO: Stop using this lowering
       */
      .lower_txd_cube_map = true,
   };

   NIR_PASS(progress, s, nir_lower_tex, &lower_tex_options);
   return progress;
}

bool
agx_nir_lower_texture_queries(nir_shader *s)
{
   return nir_shader_instructions_pass(s, lower_tex_crawl,
                                       nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_images(nir_shader *s)
{
   bool progress = false;

   /* Insert fences before lowering image atomics, since image atomics need
    * different fencing than other image operations.
    */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, fence_image,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_image_atomics_to_global, NULL, NULL);

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, legalize_image_lod,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_images,
            nir_metadata_control_flow, NULL);
   return progress;
}

static bool
lower_multisampled_store(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
      return false;

   nir_def *index_px = nir_u2u32(b, image_texel_address(b, intr, true));
   nir_def *coord2d = coords_for_buffer_texture(b, index_px);

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_src_rewrite(&intr->src[2], nir_imm_int(b, 0));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(intr, false);
   return true;
}

bool
agx_nir_lower_multisampled_image_store(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_multisampled_store,
                                     nir_metadata_control_flow, NULL);
}
