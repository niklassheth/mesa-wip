/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile_apple9_priv.h"
#include "gravity/lib/agx_abi.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "compiler/nir/nir_lower_blend.h"
#include "compiler/nir/nir_xfb_info.h"
#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"

#include <stdlib.h>
#include <string.h>

/* Keep format conversion in NIR's native half operations until selection.
 * The machine conversion preserves finite subnormals and signed zero, rounds
 * to nearest even, and returns a canonical quiet NaN for either NaN sign. */
static nir_def *
apple9_pack_half(nir_builder *b, nir_def *src)
{
   return nir_pack_half_2x16_split(b, src, nir_imm_float(b, 0));
}

static nir_def *
apple9_unpack_half(nir_builder *b, nir_def *src)
{
   return nir_channel(b, nir_unpack_half_2x16(b, src), 0);
}

/* Specialize the common sample lowering to the raster state in this variant.
 * The position word is shared with PPP programming, preserving sample order. */
bool
apple9_lower_sample_state(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_blend *blend = data;
   const bool dynamic = blend && blend->dynamic_samples;
   unsigned samples = blend ? MAX2(blend->samples, 1) : 1;
   unsigned raster_samples = blend && blend->multisample_disabled ? 1 : samples;
   uint32_t value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_sample_positions_agx:
      if (dynamic || (blend && blend->dynamic_sample_positions && raster_samples > 1)) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_def_replace(&intr->def, nir_load_ubo(b, 1, 32,
            nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(b, AGX_APPLE9_SAMPLE_POSITIONS_OFFSET),
            .align_mul = 4, .range = 4));
         return true;
      }
      value = agx_default_sample_positions(raster_samples);
      break;
   case nir_intrinsic_load_samples_log2_agx:
      if (dynamic) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_def *val = nir_load_ubo(b, 1, 32,
            nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(b, AGX_APPLE9_SAMPLES_LOG2_OFFSET),
            .align_mul = 4, .range = 4);
         nir_def_replace(&intr->def, nir_u2uN(b, val, intr->def.bit_size));
         return true;
      }
      value = util_logbase2(raster_samples);
      break;
   case nir_intrinsic_load_api_sample_mask_agx:
      if (dynamic) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_def *val = nir_load_ubo(b, 1, 32,
            nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(b, AGX_APPLE9_API_SAMPLE_MASK_OFFSET),
            .align_mul = 4, .range = 4);
         nir_def_replace(&intr->def, nir_u2uN(b, val, intr->def.bit_size));
         return true;
      }
      value = BITFIELD_MASK(samples) & ~(blend ? blend->disabled_samples : 0);
      break;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

/* This backend emits complete fragment programs, so the common coverage pass
 * always finalizes depth/stencil tests in this shader part. */
/* No native interpolation position form takes an arbitrary offset. Evaluate
 * the plane at the pixel center and step along its fine screen-space
 * derivatives. This is exact for linear interpolation and a first-order
 * approximation for perspective-correct interpolation.
 */
bool
apple9_lower_interp_at_offset(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(intr->src[0]);
   if (!bary || bary->intrinsic != nir_intrinsic_load_barycentric_at_offset)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = bary->src[0].ssa;
   nir_def *center = nir_load_barycentric_pixel(
      b, 32, .interp_mode = nir_intrinsic_interp_mode(bary));
   nir_src_rewrite(&intr->src[0], center);

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *value = &intr->def;
   nir_def *result = nir_ffma(b, nir_ddy_fine(b, value),
                              nir_channel(b, offset, 1),
                              nir_ffma(b, nir_ddx_fine(b, value),
                                       nir_channel(b, offset, 0), value));
   nir_def_rewrite_uses_after(value, result);
   return true;
}

/* Barycentric coordinates use the same interpolation positions as varyings.
 * Single-sampled shaders interpolate at the pixel centre, sample
 * interpolation becomes interpolation at the sample ID, and offsets step
 * along the screen-space derivatives as for varyings. Only the first two
 * weights are interpolated; the third is the remainder.
 */
bool
apple9_lower_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (!apple9_is_barycentric_coord(intr->intrinsic))
      return false;

   const bool single_sampled = *(const bool *)data;
   const enum glsl_interp_mode mode = nir_intrinsic_interp_mode(intr);
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *coord;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_coord_at_offset: {
      nir_def *offset = intr->src[0].ssa;
      nir_def *center =
         nir_load_barycentric_coord_pixel(b, 32, .interp_mode = mode);
      coord = nir_ffma(b, nir_ddy_fine(b, center), nir_channel(b, offset, 1),
                       nir_ffma(b, nir_ddx_fine(b, center),
                                nir_channel(b, offset, 0), center));
      break;
   }
   case nir_intrinsic_load_barycentric_coord_centroid:
   case nir_intrinsic_load_barycentric_coord_sample:
   case nir_intrinsic_load_barycentric_coord_at_sample:
      if (single_sampled) {
         coord = nir_load_barycentric_coord_pixel(b, 32, .interp_mode = mode);
      } else if (intr->intrinsic == nir_intrinsic_load_barycentric_coord_sample) {
         coord = nir_load_barycentric_coord_at_sample(
            b, 32, nir_load_sample_id(b), .interp_mode = mode);
      } else {
         return false;
      }
      break;
   default:
      return false;
   }
   nir_def_replace(&intr->def, coord);
   return true;
}

bool
apple9_expand_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (!apple9_is_barycentric_coord(intr->intrinsic) ||
       intr->def.num_components != 3)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *x = nir_channel(b, &intr->def, 0);
   nir_def *y = nir_channel(b, &intr->def, 1);
   nir_def *z = nir_fsub_imm(b, 1.0, nir_fadd(b, x, y));
   nir_def *vec = nir_vec3(b, x, y, z);
   nir_def_rewrite_uses_after(&intr->def, vec);
   return true;
}

bool
apple9_lower_coverage_sysval(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Run after single-sample lowering, which can introduce helper queries
    * while simplifying sample-mask inputs.
    */
   if (intr->intrinsic == nir_intrinsic_load_helper_invocation) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_build_lowered_load_helper_invocation(b));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, ~0u, intr->def.bit_size));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Match the common AGX flat interpolation contract: a first-provoking
 * triangle fan consumes B, while ordinary flat modes consume C. Keep this
 * choice in draw sysvals so dynamic topology does not specialize the shader. */
bool
apple9_lower_flat_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input ||
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *first_fan = nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_FIRST_FAN_OFFSET),
      .align_mul = 4, .range_base = AGX_APPLE9_FIRST_FAN_OFFSET, .range = 4);
   first_fan = nir_ine_imm(b, first_fan, 0);
   nir_def *values[4];
   unsigned read = nir_def_components_read(&intr->def);
   for (unsigned c = 0; c < intr->def.num_components; ++c) {
      if (!(read & BITFIELD_BIT(c))) {
         values[c] = nir_undef(b, 1, 32);
         continue;
      }
      nir_def *cf = nir_load_coefficients_agx(b, intr->src[0].ssa,
         .component = nir_intrinsic_component(intr) + c,
         .interp_mode = INTERP_MODE_FLAT,
         .io_semantics = nir_intrinsic_io_semantics(intr));
      values[c] = nir_bcsel(b, first_fan, nir_channel(b, cf, 1),
                            nir_channel(b, cf, 2));
   }
   nir_def_replace(&intr->def, nir_vec(b, values, intr->def.num_components));
   return true;
}

bool
apple9_lower_fragment_uniform(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_line_stipple_agx ||
       intr->intrinsic == nir_intrinsic_load_viewport_scale ||
       intr->intrinsic == nir_intrinsic_load_viewport_offset) {
      b->cursor = nir_before_instr(&intr->instr);
      unsigned offset =
         intr->intrinsic == nir_intrinsic_load_line_stipple_agx
            ? AGX_APPLE9_LINE_STIPPLE_OFFSET
         : intr->intrinsic == nir_intrinsic_load_viewport_scale
            ? AGX_APPLE9_VIEWPORT_SCALE_OFFSET
            : AGX_APPLE9_VIEWPORT_OFFSET_OFFSET;
      unsigned n = intr->def.num_components;
      nir_def *value = nir_load_ubo(b, n, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, offset), .align_mul = 16, .range_base = offset,
         .range = n * 4);
      nir_def_replace(&intr->def, value);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_blend_const_color_rgba &&
       intr->intrinsic != nir_intrinsic_load_polygon_stipple_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   bool stipple = intr->intrinsic == nir_intrinsic_load_polygon_stipple_agx;
   nir_def *offset = stipple
      ? nir_iadd_imm(b, nir_ishl_imm(b, intr->src[0].ssa, 2),
                    AGX_APPLE9_POLYGON_STIPPLE_OFFSET)
      : nir_imm_int(b, 0);
   nir_def *value = nir_load_ubo(b, stipple ? 1 : 4, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING), offset,
      .align_mul = stipple ? 4 : 16,
      .range_base = stipple ? AGX_APPLE9_POLYGON_STIPPLE_OFFSET : 0,
      .range = stipple ? 128 : 16);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Rectangle textures share the ordinary 2D descriptor. The common texture
 * pass normalizes their coordinates; explicit gradients need the same scale.
 * Integer fetches use level zero without any coordinate normalization. */
static bool
apple9_lower_rectangle(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_RECT)
      return false;

   b->cursor = nir_before_instr(instr);
   if (tex->op == nir_texop_txf) {
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      if (nir_tex_instr_src_index(tex, nir_tex_src_lod) < 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_backend1) < 0)
         nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_int(b, 0));
      return true;
   }

   if (tex->op != nir_texop_txd)
      return false;

   nir_def *scale = nir_frcp(b, nir_i2f32(b, nir_get_texture_size(b, tex)));
   for (unsigned i = 0; i < tex->num_srcs; ++i) {
      if (tex->src[i].src_type == nir_tex_src_ddx ||
          tex->src[i].src_type == nir_tex_src_ddy)
         nir_src_rewrite(&tex->src[i].src,
                        nir_fmul(b, tex->src[i].src.ssa, scale));
   }
   return true;
}

/* Compute has no implicit texture-unit footprint. Derive it from the declared
 * derivative group, using the same spatial coordinates as explicit gradients.
 * Array layers and shadow comparisons never participate in the footprint. */
static bool
apple9_lower_compute_lod_query(nir_builder *b, nir_instr *instr, void *data)
{
   if (b->shader->info.stage != MESA_SHADER_COMPUTE ||
       b->shader->info.derivative_group != DERIVATIVE_GROUP_LINEAR ||
       instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;

   b->cursor = nir_before_instr(instr);
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1 :
                   tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                   tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : 2;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *p = nir_trim_vector(b, tex->src[coord].src.ssa, dims);
   nir_def *dx = nir_ddx(b, p), *dy = nir_ddy(b, p);
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   if (cube) {
      /* Project each derivative onto the selected face with the quotient
       * rule. Signs of the face axes do not affect the footprint length. */
      nir_def *a = nir_fabs(b, p);
      nir_def *x = nir_channel(b, a, 0), *y = nir_channel(b, a, 1);
      nir_def *z = nir_channel(b, a, 2);
      nir_def *use_z = nir_fge(b, z, nir_fmax(b, x, y));
      nir_def *use_y = nir_fge(b, y, nir_fmax(b, x, z));
      unsigned xzy[] = {0, 2, 1}, yzx[] = {1, 2, 0};
      nir_def *values[] = {p, dx, dy};
      for (unsigned i = 0; i < 3; ++i)
         values[i] = nir_bcsel(b, use_z, values[i],
            nir_bcsel(b, use_y, nir_swizzle(b, values[i], xzy, 3),
                                 nir_swizzle(b, values[i], yzx, 3)));
      nir_def *rcp = nir_frcp(b, nir_channel(b, values[0], 2));
      nir_def *uv = nir_fmul(b, nir_trim_vector(b, values[0], 2), rcp);
      for (unsigned i = 1; i < 3; ++i)
         values[i] = nir_fmul(b, nir_fmul_imm(b, rcp, 0.5),
            nir_fsub(b, nir_trim_vector(b, values[i], 2),
                        nir_fmul(b, uv, nir_channel(b, values[i], 2))));
      dx = values[1];
      dy = values[2];
      dims = 2;
   }
   nir_def *size[3];
   for (unsigned c = 0; c < dims; ++c)
      size[c] = nir_u2f32(b, agx_apple9_texture_info(b, tex, cube ? 0 : c));
   nir_def *extent = nir_vec(b, size, dims);
   dx = nir_fmul(b, dx, extent);
   dy = nir_fmul(b, dy, extent);
   nir_def *xx = nir_fdot(b, dx, dx), *yy = nir_fdot(b, dy, dy);
   nir_def *major = nir_fmax(b, xx, yy);
   /* Native 2D anisotropic sampling uses footprint area divided by its
    * longest derivative. This also handles nonorthogonal and rank-one
    * footprints; the shorter derivative's length would overestimate them.
    * Native 3D sampling is isotropic. Cube gradients are lowered to scalar
    * LODs by the common texture pass. */
   if (!cube && tex->sampler_dim != GLSL_SAMPLER_DIM_3D) {
      nir_def *area = dims == 1 ? nir_imm_float(b, 0) :
         nir_fsub(b, nir_fmul(b, nir_channel(b, dx, 0), nir_channel(b, dy, 1)),
                     nir_fmul(b, nir_channel(b, dx, 1), nir_channel(b, dy, 0)));
      nir_def *minor = nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), major),
         nir_fdiv(b, nir_fmul(b, area, area), major), nir_imm_float(b, 0));
      nir_def *aniso = agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MAX_ANISOTROPY);
      aniso = nir_fmax(b, aniso, nir_imm_float(b, 1.0));
      major = nir_fmax(b, minor, nir_fdiv(b, major, nir_fmul(b, aniso, aniso)));
   }
   nir_def *lod = nir_fmul_imm(b, nir_flog2(b, major), 0.5);
   int bias_src = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   nir_def *bias = bias_src >= 0 ? nir_f2f32(b, tex->src[bias_src].src.ssa) :
                                 nir_imm_float(b, 0);
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0) {
      nir_def *state_bias = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
         .align_mul = 4, .range = 4);
      bias = nir_fadd(b, bias, state_bias);
   }
   /* Honeykrisp advertises maxSamplerLodBias = 15. */
   bias = nir_fclamp(b, bias, nir_imm_float(b, -15), nir_imm_float(b, 15));
   lod = nir_fadd(b, lod, bias);
   nir_def *level = nir_fclamp(b, lod,
      agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MIN_LOD),
      agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MAX_LOD));
   nir_def *mip = agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MIP);
   nir_def *levels = agx_apple9_texture_info(b, tex, 3);
   level = nir_fclamp(b, level, nir_imm_float(b, 0),
                        nir_u2f32(b, nir_iadd_imm(b, levels, -1)));
   int view_min = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (view_min >= 0)
      level = nir_fmax(b, level, nir_f2f32(b, tex->src[view_min].src.ssa));
   level = nir_bcsel(b, nir_ieq_imm(b, mip, PIPE_TEX_MIPFILTER_NEAREST),
      nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, level, 0.5)), -1.0), level);
   level = nir_bcsel(b, nir_ieq_imm(b, mip, PIPE_TEX_MIPFILTER_NONE),
                       nir_imm_float(b, 0), level);
   nir_def *result = nir_bcsel(b, nir_ieq_imm(b, levels, 0),
      nir_imm_vec2(b, 0, 0), nir_vec2(b, level, lod));
   nir_def_rewrite_uses(&tex->def, result);
   nir_instr_remove(instr);
   return true;
}

/* LOD queries use the texture unit's derivative and clamp machinery. The
 * hardware returns continuous LODs; GL additionally applies mip selection to
 * the first component. Bias goes through the normal sample-bias operand. */
static bool
apple9_lower_texture_lod_query(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;
   /* LOD queries have spatial coordinates only, even for array/shadow
    * samplers. Supply the native array operand without reading a nonexistent
    * NIR component; comparison does not participate in a LOD query. */
   if (tex->is_array && (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                         tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      b->cursor = nir_before_instr(instr);
      nir_def *coords = tex->src[coord].src.ssa;
      unsigned spatial = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : 2;
      nir_src_rewrite(&tex->src[coord].src,
         nir_pad_vector_imm_int(b, nir_trim_vector(b, coords, spatial), 0,
                                spatial + 1));
      tex->coord_components = spatial + 1;
   }
   tex->is_shadow = false;
   tex->is_new_style_shadow = false;
   /* Always query in the biased form, with a zero bias if there is none:
    * table-addressed samplers get their state bias added this way, and a
    * descriptor-addressed query in the unbiased form returns a different
    * LOD (glsl.texture_functions textureclamp *_bias, cube-array bias). */
   if (nir_tex_instr_src_index(tex, nir_tex_src_bias) < 0) {
      b->cursor = nir_before_instr(instr);
      nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0));
   }
   tex->op = nir_texop_txb;
   tex->backend_flags |= AGX_APPLE9_TEXTURE_LOD_QUERY;
   /* As a sample, the instruction has a sample's destination size; the
    * query result is its first two components. Leaving the two-component
    * destination fails NIR validation after the next pass with progress. */
   b->cursor = nir_after_instr(instr);
   tex->def.num_components = nir_tex_instr_dest_size(tex);
   nir_def *level = nir_channel(b, &tex->def, 0);
   if (data && tex->sampler_index < 32) {
      const struct agx_apple9_sampler_key *key = data;
      if (key[tex->sampler_index].mip_filter == PIPE_TEX_MIPFILTER_NONE)
         level = nir_imm_float(b, 0);
      else if (key[tex->sampler_index].mip_filter == PIPE_TEX_MIPFILTER_NEAREST)
         level = nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, level, 0.5)), -1.0);
   }
   nir_def *result = nir_vec2(b, level, nir_channel(b, &tex->def, 1));
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

/* Sampler bias is per-draw state, independent of a shader's explicit bias.
 * Keep this in FP32, matching the rest of the Apple9 texture operands. */
bool
agx_nir_lower_apple9_sampler_state(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   const char **reason)
{
   nir_lower_global_vars_to_local(nir);
   nir_split_var_copies(nir);
   nir_lower_var_copies(nir);
   nir_lower_vars_to_ssa(nir);
   bool optimized;
   do {
      optimized = nir_opt_constant_folding(nir);
      optimized |= nir_opt_copy_prop(nir);
      optimized |= nir_opt_remove_phis(nir);
      optimized |= nir_opt_loop_unroll(nir);
      optimized |= nir_opt_dead_cf(nir);
      optimized |= nir_opt_dce(nir);
   } while (optimized);
   nir_lower_samplers(nir);
   nir_opt_constant_folding(nir);
   nir_shader_instructions_pass(nir, apple9_lower_rectangle,
                                nir_metadata_control_flow, NULL);
   const nir_lower_tex_options options = {
      .lower_txp = ~0u, .lower_1d = true,
      .lower_rect = true, .lower_rect_offset = true,
   };
   nir_lower_tex(nir, &options);
   nir_shader_instructions_pass(nir, apple9_lower_texture_lod_query,
                                nir_metadata_control_flow, (void *)key);
   if (!agx_nir_lower_apple9_texture_offsets(nir, key, reason))
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (tex->texture_index >= 32 || tex->sampler_index >= 32) {
            *reason = "Apple9 texture binding exceeds 32 entries";
            return false;
         }
      }
   }
   nir_builder b = nir_builder_create(impl);
   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
             tex->op != nir_texop_txl && tex->op != nir_texop_txd)
            continue;
         if (tex->backend_flags & AGX_APPLE9_TEXTURE_LOD_QUERY)
            continue;
         /* Normalized depth formats clamp the comparison reference to [0,1].
          * Z24 uses float storage through u_transfer_helper, so the texture
          * descriptor cannot infer this API-format requirement. */
         if (tex->is_shadow &&
             (key[tex->texture_index].flags & AGX_APPLE9_CLAMP_SHADOW_REFERENCE)) {
            int cmp = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
            if (cmp >= 0) {
               b.cursor = nir_before_instr(instr);
               nir_src_rewrite(&tex->src[cmp].src,
                              nir_fsat(&b, tex->src[cmp].src.ssa));
               progress = true;
            }
         }
         const struct agx_apple9_sampler_key *state = &key[tex->sampler_index];
         bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
         /* Comparison samples stay native: as an array layer they take the
          * array-comparison form, whose implicit-LOD encoding returns the
          * stored depth instead of the comparison result. 2D arrays reroute
          * that through gradients; a cube face here keeps the native cube
          * sample and its per-face clamp. */
         /* Implicit-LOD samples stay native too. As an array layer, each
          * pixel picks its own face, so a quad straddling a face edge gets a
          * discontinuous derivative and selects too high a level; the native
          * cube derives the LOD from the direction vector. */
         bool implicit_lod = tex->op == nir_texop_tex || tex->op == nir_texop_txb;
         if (cube && !state->seamless_cube_map && !tex->is_shadow && !implicit_lod) {
            /* Cube filtering always clamps the selected face on Apple9.
             * Sampling that face as an array layer restores ordinary 2D
             * wrapping when seamless cube filtering is disabled. */
            tex->backend_flags |= AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY;
            progress = true;
         }
      }
   }
   nir_progress(progress, impl, nir_metadata_control_flow);
   return true;
}


static nir_def *
apple9_fetch_buffer(nir_builder *b, const nir_tex_instr *orig, nir_def *index)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->texture_index = orig->texture_index;
   tex->sampler_index = APPLE9_FETCH_SAMPLER;
   tex->dest_type = orig->dest_type;
   tex->coord_components = 2;
   /* The caller bounds the logical buffer element before this physical 2D
    * access. Buffer views have no image levels/layers/sample metadata; do not
    * reinterpret that metadata when the ordinary image-fetch pass runs. */
   tex->backend_flags |= AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(
      nir_vec2(b, nir_umod_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH),
                  nir_udiv_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH)));
   tex->src[1].src_type = nir_tex_src_lod;
   tex->src[1].src = nir_src_for_ssa(nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* Linear 2D descriptors cover buffer views independently of the underlying
 * buffer allocation. RGB32 has no native format: fetch three scalar words and
 * apply the logical view's swizzle after assembling the texel. */
static bool
apple9_lower_buffer_texture(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *size = agx_apple9_texture_info(b, tex, 0);
   nir_if *valid = nir_push_if(b, nir_ult(b, index, size));
   nir_if *rgb32 = nir_push_if(b,
      nir_ine_imm(b, agx_apple9_texture_info(b, tex, 5), 0));
   nir_def *components[4];
   nir_def *scalar_index = nir_imul_imm(b, index, 3);
   for (unsigned c = 0; c < 3; ++c)
      components[c] = nir_channel(b,
         apple9_fetch_buffer(b, tex, nir_iadd_imm(b, scalar_index, c)), 0);
   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   components[3] = floating ? nir_imm_float(b, 1) : nir_imm_int(b, 1);
   nir_def *swizzle = agx_apple9_texture_info(b, tex, 6);
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *channel = nir_iand_imm(b, nir_ushr_imm(b, swizzle, c * 3), 7);
      rgba[c] = nir_imm_int(b, 0);
      for (unsigned j = 0; j < 4; ++j)
         rgba[c] = nir_bcsel(b, nir_ieq_imm(b, channel, j), components[j], rgba[c]);
      rgba[c] = nir_bcsel(b, nir_ieq_imm(b, channel, PIPE_SWIZZLE_1),
                         components[3], rgba[c]);
   }
   nir_def *assembled = nir_vec4(b, rgba[0], rgba[1], rgba[2], rgba[3]);
   nir_push_else(b, rgb32);
   nir_def *native = apple9_fetch_buffer(b, tex, index);
   nir_pop_if(b, rgb32);
   nir_def *result = nir_if_phi(b, assembled, native);
   nir_push_else(b, valid);
   nir_def *zero = nir_imm_ivec4(b, 0, 0, 0, 0);
   nir_pop_if(b, valid);
   result = nir_if_phi(b, result, zero);
   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, result, tex->def.num_components));
   nir_instr_remove(instr);
   return true;
}

/* Native hardware min_lod_clamp for gradient samples: 2D (0x0114 / 0x0249
 * with offset), 2D-array (0x051c), and 3D (0x219c). */
static bool
apple9_native_min_lod_grad(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_txd || tex->is_shadow ||
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0)
      return false;
   if (nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0)
      return tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
             agx_apple9_texture_native_offset(tex, NULL);
   return tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
          tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
}

/* Native hardware min_lod_clamp for fragment implicit and biased samples:
 * 2D (0x0804 / 0x02c1), 2D-array (0x040c / 0x03c1), 3D (0x2180 / 0x2190),
 * Cube implicit (0x0810), bindless Cube-array implicit (0x0c10), Depth2D
 * (0x0820 / 0x0824), and bindless Depth2D-array implicit (0x0c20 / 0x0361). */
static bool
apple9_native_min_lod_sample(const nir_tex_instr *tex, mesa_shader_stage stage)
{
   if (stage != MESA_SHADER_FRAGMENT ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0)
      return false;
   const bool bindless_sampler =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
   if (nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0) {
      if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
          !agx_apple9_texture_native_offset(tex, NULL))
         return false;
      if (!tex->is_shadow)
         return true;
      return tex->is_array && tex->op == nir_texop_tex && bindless_sampler;
   }
   const bool cube_as_array = !!(tex->backend_flags & AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY);
   if (tex->is_shadow) {
      if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D)
         return false;
      return !tex->is_array || (tex->op == nir_texop_tex && bindless_sampler);
   }
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_3D || cube_as_array)
      return true;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return tex->op == nir_texop_tex && (!tex->is_array || bindless_sampler);
   return false;
}

/* Whether a texture operation's final form, after the remaining backend
 * rewrites (implicit 3D and comparison samples become bias-0 samples), has a
 * measured offset encoding. Mirrors selection (see the texture emitters). */
static bool
apple9_offset_form_encodable(const nir_tex_instr *tex)
{
   const bool volume = tex->coord_components >= 3 || tex->is_shadow;
   const bool bindless_sampler =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
   unsigned dimension = 0;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return false;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_3D)
      dimension = 3;
   /* 2D arrays pack the layer with the LOD (dimension 2). */
   if (tex->is_array)
      dimension = 2;
   bool min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 ||
                  nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0;
   if (min_lod) {
      if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D)
         return false;
      if (tex->is_shadow)
         return dimension == 2 && tex->op == nir_texop_tex && bindless_sampler;
      /* 2D-array bias+min_lod+offset uses explicit mode 0x03c1. */
      if (dimension == 2)
         return tex->op == nir_texop_tex || tex->op == nir_texop_txb;
   }
   bool explicit_lod = false, bias = false, gradient = false;
   switch (tex->op) {
   case nir_texop_tg4:
      /* 2D-array gathers use implicit 0x0408/0x0420 (+offset 0x0301/0x0321).
       * Non-array comparison gathers select explicit-LOD; plain 2D implicit. */
      if (dimension == 3)
         return false;
      explicit_lod = tex->is_shadow && dimension != 2;
      break;
   case nir_texop_txd:
      gradient = true;
      break;
   case nir_texop_txl:
      explicit_lod = true;
      break;
   case nir_texop_txb:
      /* 2D-array shadow txb lowers to txd (no measured offset mode). */
      if (dimension == 2 && tex->is_shadow)
         return false;
      explicit_lod = bias = true;
      break;
   case nir_texop_tex:
      if (dimension == 2 && !min_lod) {
         /* Table-addressed 2D-array tex becomes txb (or txd for shadow) in
          * apple9_lower_sampler_bias / apple9_lower_implicit_gradients. */
         if (!bindless_sampler)
            return false;
         explicit_lod = bias = false;
      } else {
         explicit_lod = bias = volume || min_lod;
      }
      break;
   default:
      return false;
   }
   uint16_t mode = agx_apple9_texture_mode(explicit_lod, gradient, bias,
                                           dimension, tex->is_shadow, min_lod);
   return agx_apple9_texture_offset_mode(mode, NULL);
}

static bool
apple9_flag_unencodable_offsets(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 ||
       apple9_offset_form_encodable(tex))
      return false;
   tex->backend_flags |= AGX_APPLE9_TEXTURE_OFFSET_FALLBACK;
   *(bool *)data = true;
   return true;
}

static bool
apple9_lower_implicit_gradients(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   bool compute = b->shader->info.stage == MESA_SHADER_COMPUTE &&
                  b->shader->info.derivative_group == DERIVATIVE_GROUP_LINEAR;
   if (!compute && b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!compute) {
      bool bindless_sampler =
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
      if (!(tex->is_array && tex->is_shadow &&
            tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
            !(tex->op == nir_texop_tex && bindless_sampler) &&
            !apple9_native_min_lod_sample(tex, b->shader->info.stage)))
         return false;
   }
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;
   b->cursor = nir_before_instr(instr);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   /* Projection, including the shadow comparator, is already lowered. */
   assert(nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0);
   unsigned spatial = tex->coord_components - tex->is_array;
   nir_def *position = nir_trim_vector(b, tex->src[coord].src.ssa, spatial);
   nir_def *dx = nir_ddx(b, position), *dy = nir_ddy(b, position);
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0) {
      nir_def *scale = nir_fexp2(b, tex->src[bias].src.ssa);
      dx = nir_fmul(b, dx, scale);
      dy = nir_fmul(b, dy, scale);
      nir_tex_instr_remove_src(tex, bias);
   }
   tex->op = nir_texop_txd;
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
   return true;
}

static bool
apple9_lower_texel_fetch(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op == nir_texop_txf_ms && tex->sampler_dim == GLSL_SAMPLER_DIM_MS) {
      /* Input-attachment lowering can retain an explicit zero LOD.
       * Multisample images have only one level and the native fetch has no
       * LOD operand.
       */
      int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      if (lod >= 0) {
         if (!nir_src_is_const(tex->src[lod].src) ||
             nir_src_as_uint(tex->src[lod].src) != 0)
            return false;
         nir_tex_instr_remove_src(tex, lod);
      }
      b->cursor = nir_before_instr(instr);
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *sample = nir_steal_tex_src(tex, nir_tex_src_ms_index);
      nir_def *layer = tex->is_array ? nir_channel(b, xyz, 2) : nir_imm_int(b, 0);
      /* Full-width X/Y are followed by a 12-bit layer and sample index in
       * bits 12..13 of the third publication word. The fourth word is unused. */
      coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec4(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1),
                     nir_ior(b, layer, nir_ishl_imm(b, sample, 12)),
                     nir_imm_int(b, 0)));
      tex->coord_components = 4;
      tex->sampler_index = APPLE9_FETCH_SAMPLER;
      return true;
   }
   if (tex->op != nir_texop_txf || tex->is_shadow ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_3D))
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (coord < 0 || lod < 0)
      return false;
   /* The read instruction consumes integer spatial coordinates. It uses a
    * private sampler descriptor, just as the common AGX texel-fetch path does. */
   tex->sampler_index = APPLE9_FETCH_SAMPLER;
   return true;
}

/* Texel fetches outside the image read the border. The fetch sampler
 * clamps coordinates to it, but hardware clamps the level, layer and sample
 * instead; move such fetches out of bounds explicitly. This precedes the
 * native packing of layers and samples.
 */
static bool
apple9_bound_texel_fetch(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if ((tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms) ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_BUF || tex->texture_index >= 32 ||
       (tex->backend_flags & AGX_APPLE9_TEXTURE_FETCH_BOUNDED))
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *xyz = tex->src[coord].src.ssa;
   nir_def *valid = nir_imm_true(b);
   /* Multisample images have one level; their fetch keeps a literal zero
    * LOD, if any.
    */
   int lod = tex->op == nir_texop_txf ? nir_tex_instr_src_index(tex, nir_tex_src_lod)
                                      : -1;
   if (lod >= 0) {
      nir_def *level = tex->src[lod].src.ssa;
      valid = nir_iand(b, valid, nir_ult(b, level,
         agx_apple9_texture_info(b, tex, 3)));
      nir_src_rewrite(&tex->src[lod].src,
                      nir_bcsel(b, valid, level, nir_imm_int(b, 0)));
   }
   if (tex->is_array) {
      nir_def *layer = nir_channel(b, xyz, tex->coord_components - 1);
      valid = nir_iand(b, valid, nir_ult(b, layer,
         agx_apple9_texture_info(b, tex, 2)));
   }
   int ms = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   if (ms >= 0) {
      valid = nir_iand(b, valid, nir_ult(b, tex->src[ms].src.ssa,
         agx_apple9_texture_info(b, tex, 4)));
   }
   nir_def *x = nir_bcsel(b, valid, nir_channel(b, xyz, 0), nir_imm_int(b, -1));
   nir_src_rewrite(&tex->src[coord].src, nir_vector_insert_imm(b, xyz, x, 0));
   tex->backend_flags |= AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   return true;
}

static bool
apple9_lower_texture_queries_and_layers(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->texture_index >= 32)
      return false;
   b->cursor = nir_before_instr(instr);
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_texture_samples) {
      nir_def *result;
      if (tex->op == nir_texop_txs) {
         int src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
         nir_def *lod = src < 0 ? nir_imm_int(b, 0) : tex->src[src].src.ssa;
         lod = nir_umin(b, lod, nir_imm_int(b, 31));
         nir_def *size[3];
         for (unsigned c = 0; c < tex->def.num_components; ++c) {
            bool layer = tex->is_array && c + 1 == tex->def.num_components;
            size[c] = agx_apple9_texture_info(b, tex, layer ? 2 : c);
            if (!layer && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
               size[c] = nir_umax(b, nir_ushr(b, size[c], lod), nir_imm_int(b, 1));
         }
         result = nir_vec(b, size, tex->def.num_components);
      } else
         result = agx_apple9_texture_info(b, tex,
                                      tex->op == nir_texop_query_levels ? 3 : 4);
      /* A null view has no levels, and its size and sample queries return 0
       * (Vulkan's null descriptors). Buffer metadata has no level count. */
      if (tex->op != nir_texop_query_levels &&
          tex->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
         nir_def *null = nir_ieq_imm(b, agx_apple9_texture_info(b, tex, 3), 0);
         result = nir_bcsel(b, null, nir_imm_zero(b, result->num_components, 32),
                            result);
      }
      nir_def_rewrite_uses(&tex->def, result);
      nir_instr_remove(instr);
      return true;
   }
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
       tex->op != nir_texop_lod) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      nir_def *xyzw = tex->src[coord].src.ssa;
      nir_def *layer = nir_f2u32(b, nir_fmax(b,
         nir_fround_even(b, nir_channel(b, xyzw, 3)), nir_imm_float(b, 0)));
      nir_def *cubes = agx_apple9_texture_info(b, tex, 2);
      layer = nir_umin(b, layer, nir_iadd_imm(b, cubes, -1));
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec4(b, nir_channel(b, xyzw, 0), nir_channel(b, xyzw, 1),
                  nir_channel(b, xyzw, 2), layer));
      return true;
   }
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
       tex->op != nir_texop_txf && tex->op != nir_texop_lod) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *layer = nir_f2u32(b, nir_fmax(b,
         nir_fround_even(b, nir_channel(b, xyz, 2)), nir_imm_float(b, 0)));
      layer = nir_umin(b, layer,
         nir_iadd_imm(b, agx_apple9_texture_info(b, tex, 2), -1));
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec3(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1), layer));
      return true;
   }
   return false;
}

static bool
apple9_lower_sampler_bias(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txd && tex->op != nir_texop_txl)
      return false;
   if (tex->sampler_index >= AGX_APPLE9_SAMPLER_BIAS_COUNT)
      return false; /* Rejected by texture binding validation. */
   /* Descriptor-addressed samplers have their bias applied upstream, and
    * the offset lowering's levels already include it. */
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       (tex->backend_flags & AGX_APPLE9_TEXTURE_BIASED_LEVEL))
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *bias = nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
      .align_mul = 4, .range = 4);
   if (tex->op == nir_texop_txd) {
      nir_def *scale = nir_fexp2(b, bias);
      for (unsigned i = 0; i < tex->num_srcs; ++i) {
         if (tex->src[i].src_type == nir_tex_src_ddx ||
             tex->src[i].src_type == nir_tex_src_ddy)
            nir_src_rewrite(&tex->src[i].src,
               nir_fmul(b, tex->src[i].src.ssa, scale));
      }
   } else if (tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, bias);
   } else {
      int idx = nir_tex_instr_src_index(tex, tex->op == nir_texop_txl
                                              ? nir_tex_src_lod : nir_tex_src_bias);
      nir_src_rewrite(&tex->src[idx].src,
         nir_fadd(b, tex->src[idx].src.ssa, bias));
   }
   return true;
}

/* Selection encodes an implicit sample only in its two-coordinate form; the
 * volume, cube, array and comparison operands exist on the explicit forms.
 * Table-addressed samplers became txb in apple9_lower_sampler_bias; give
 * descriptor-addressed ones, and late implicit queries, the same zero-bias
 * form. */
static bool
apple9_lower_implicit_volume_tex(nir_builder *b, nir_instr *instr,
                                 UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tex ||
       (tex->coord_components < 3 && !tex->is_array && !tex->is_shadow) ||
       (tex->sampler_dim == GLSL_SAMPLER_DIM_2D && tex->is_array) ||
       nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0)
      return false;
   b->cursor = nir_before_instr(instr);
   tex->op = nir_texop_txb;
   nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0));
   return true;
}

static bool
apple9_blend_factor_supported(unsigned factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ZERO:
   case PIPE_BLENDFACTOR_ONE:
   case PIPE_BLENDFACTOR_SRC_COLOR:
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:
   case PIPE_BLENDFACTOR_DST_COLOR:
   case PIPE_BLENDFACTOR_INV_DST_COLOR:
   case PIPE_BLENDFACTOR_SRC_ALPHA:
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
   case PIPE_BLENDFACTOR_DST_ALPHA:
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA:
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
   case PIPE_BLENDFACTOR_SRC1_COLOR:
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

/* The validated color store consumes RGBA8. Express conversion as ordinary NIR
 * so register allocation and numerical behavior do not depend on a captured
 * native color-pack sequence. */
void
apple9_collect_color(nir_shader *nir, bool dual_source_by_location)
{
   /* Output assignments are ordinary mutable shader state. Collect them
    * before packing or blending so component writes and branch-local writes
    * produce one final color for each surviving invocation. */
   nir_lower_returns(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_variable *color[9] = {0};
   nir_variable *depth = NULL, *stencil = NULL, *sample_mask = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_SAMPLE_MASK) {
            if (!sample_mask)
               sample_mask = nir_local_variable_create(impl, glsl_int_type(), "sample mask");
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, sample_mask, intr->src[0].ssa, 1);
            nir_instr_remove(instr);
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH) {
            if (!depth)
               depth = nir_local_variable_create(impl, glsl_float_type(), "depth");
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, depth, intr->src[0].ssa, 1);
            nir_instr_remove(instr);
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_STENCIL) {
            if (!stencil)
               stencil = nir_local_variable_create(impl, glsl_uint_type(), "stencil");
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, stencil, nir_u2u32(&b, intr->src[0].ssa), 1);
            nir_instr_remove(instr);
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_intrinsic_io_semantics(intr).location < FRAG_RESULT_DATA0 ||
             nir_intrinsic_io_semantics(intr).location >= FRAG_RESULT_DATA0 + 8 ||
             intr->src[0].ssa->bit_size != 32 ||
             !nir_src_is_const(intr->src[1]) ||
             nir_intrinsic_component(intr) + intr->num_components > 4)
            continue;
         unsigned rt = nir_intrinsic_io_semantics(intr).location - FRAG_RESULT_DATA0 +
                       nir_src_as_uint(intr->src[1]);
         if (rt >= 8)
            continue;
         if (nir_intrinsic_io_semantics(intr).dual_source_blend_index) {
            if (rt != 0)
               continue;
            rt = 8;
         } else if (dual_source_by_location && rt == 1) {
            /* Unlike the epilog ABI, monolithic blending keeps MRT1 and
             * SRC1 separate. Remap before unused render targets are removed. */
            rt = 8;
         }
         if (!color[rt])
            color[rt] = nir_local_variable_create(impl, glsl_vec4_type(), "color");
         b.cursor = nir_before_instr(instr);
         unsigned component = nir_intrinsic_component(intr);
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c)
            channels[c] = c >= component && c < component + intr->num_components
               ? nir_channel(&b, intr->src[0].ssa, c - component)
               : nir_undef(&b, 1, 32);
         nir_store_var(&b, color[rt], nir_vec(&b, channels, 4),
                       nir_intrinsic_write_mask(intr) << component);
         nir_instr_remove(instr);
      }
   }
   if (sample_mask) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, sample_mask), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_int32,
         .io_semantics = {.location = FRAG_RESULT_SAMPLE_MASK, .num_slots = 1});
   }
   if (depth) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, depth), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});
   }
   if (stencil) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, stencil), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_uint32,
         .io_semantics = {.location = FRAG_RESULT_STENCIL, .num_slots = 1});
   }
   /* Collect the dual-source value first, then location 0 before the other
    * render targets so alpha-to-coverage precedes every tile store.
    */
   for (unsigned i = 0; i < 9; ++i) {
      unsigned rt = i == 0 ? 8 : i - 1;
      if (!color[rt])
         continue;
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, color[rt]), nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0 + (rt == 8 ? 0 : rt),
                                        .dual_source_blend_index = rt == 8,
                                        .num_slots = 1});
      nir_progress(true, impl, nir_metadata_control_flow);
   }
}

static bool
apple9_blend_channel_reads_destination(enum pipe_blend_func func,
                                       enum pipe_blendfactor src,
                                       enum pipe_blendfactor dst)
{
   if (func == PIPE_BLEND_MIN || func == PIPE_BLEND_MAX ||
       dst != PIPE_BLENDFACTOR_ZERO)
      return true;

   switch (util_blendfactor_without_invert(src)) {
   case PIPE_BLENDFACTOR_DST_COLOR:
   case PIPE_BLENDFACTOR_DST_ALPHA:
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return true;
   default:
      return false;
   }
}

static nir_def *apple9_pack_color(nir_builder *b, enum pipe_format format,
                                  nir_def *color);

nir_def *
apple9_a2c_sample_mask(nir_builder *b, nir_def *color, unsigned samples)
{
   nir_def *alpha = nir_channel(b, color, 3);
   alpha = nir_fmin(b, nir_fmax(b, alpha, nir_imm_float(b, 0)),
                    nir_imm_float(b, 1));
   nir_def *count =
      nir_f2u32(b, nir_fadd_imm(b, nir_fmul_imm(b, alpha, samples), 0.5));
   return nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), count), -1);
}

bool
apple9_lower_color(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   struct apple9_color_lower *lower = data;
   /* Depth and stencil exports are emitted by the ZS lowering. */
   if (nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH ||
       nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_STENCIL)
      return false;
   unsigned rt = nir_intrinsic_io_semantics(intr).location - FRAG_RESULT_DATA0;
   bool alpha_to_coverage = rt == 0 && lower->blend &&
                            lower->blend[0].alpha_to_coverage;
   if (rt < 8 && rt >= lower->nr_targets && !alpha_to_coverage) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   unsigned component = nir_intrinsic_component(intr);
   if (rt >= 8 || intr->src[0].ssa->bit_size != 32 || !intr->num_components ||
       component + intr->num_components > 4) {
      lower->valid = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   const struct agx_apple9_blend *blend =
      lower->blend ? &lower->blend[rt] : NULL;
   enum pipe_format format = blend && blend->format != PIPE_FORMAT_NONE
                               ? blend->format : PIPE_FORMAT_R8G8B8A8_UNORM;
   enum pipe_format tile_format = agx_apple9_color_tile_format(format);
   bool fp16 = agx_apple9_color_is_half(tile_format);
   bool integer = util_format_is_pure_integer(format);
   const struct util_format_channel_description *channel =
      &util_format_description(tile_format)->channel[0];
   unsigned bits = channel->size;
   bool raw = integer || bits == 32;
   bool packed_format = agx_apple9_color_is_packed(tile_format);
   bool normalized = !raw && !fp16 && !packed_format;
   bool snorm = normalized && channel->type == UTIL_FORMAT_TYPE_SIGNED;
   assert(!normalized || bits == 8);
   unsigned components = agx_apple9_color_components(format);
   unsigned words = agx_apple9_color_words(format), offset = 0;
   for (unsigned i = 0; i < rt; ++i)
      offset += agx_apple9_color_words(lower->blend ? lower->blend[i].format
                                                  : PIPE_FORMAT_NONE);
   nir_def *color = intr->src[0].ssa;
   /* Undef optimization can trim unwritten channels from the final collected
    * output. Materialize those undefined values as zero before RGBA packing. */
   if (component || intr->num_components != 4 ||
       nir_intrinsic_write_mask(intr) != 15) {
      nir_def *channels[4];
      for (unsigned c = 0; c < 4; ++c) {
         bool written =
            c >= component && c < component + intr->num_components &&
            (nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c - component));
         channels[c] = written ? nir_channel(b, color, c - component)
                               : nir_imm_float(b, 0);
      }
      color = nir_vec(b, channels, 4);
   }
   if (nir_intrinsic_io_semantics(intr).dual_source_blend_index) {
      lower->src1 = color;
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend && !blend->colormask && !blend->alpha_to_coverage) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend) {
      if (!blend->logicop_enable && !blend->advanced_mode &&
          (blend->rgb_func > PIPE_BLEND_MAX ||
          blend->alpha_func > PIPE_BLEND_MAX ||
          !apple9_blend_factor_supported(blend->rgb_src) ||
          !apple9_blend_factor_supported(blend->rgb_dst) ||
          !apple9_blend_factor_supported(blend->alpha_src) ||
          !apple9_blend_factor_supported(blend->alpha_dst))) {
         lower->valid = false;
         return false;
      }
   }
   unsigned samples = blend ? MAX2(blend->samples, 1) : 1;
   if (samples != 1 && samples != 2 && samples != 4) {
      lower->valid = false;
      return false;
   }
   /* A raw tile word must preserve masked channels, and blending may need a
    * different destination for each sample. Otherwise the same packed result
    * can be written to every covered sample in one masked store. */
   unsigned component_mask = BITFIELD_MASK(components);
   bool logicop_reads_destination = blend && blend->logicop_enable &&
      !util_format_is_float(format) && !util_format_is_srgb(format) &&
      blend->logicop_func != PIPE_LOGICOP_CLEAR &&
      blend->logicop_func != PIPE_LOGICOP_COPY &&
      blend->logicop_func != PIPE_LOGICOP_COPY_INVERTED &&
      blend->logicop_func != PIPE_LOGICOP_SET;
   bool reads_destination = blend &&
      (((blend->colormask & component_mask) != component_mask) ||
       logicop_reads_destination || (!blend->logicop_enable && !integer &&
        (blend->advanced_mode ||
         apple9_blend_channel_reads_destination(blend->rgb_func,
                                                blend->rgb_src, blend->rgb_dst) ||
         apple9_blend_channel_reads_destination(blend->alpha_func,
                                                blend->alpha_src, blend->alpha_dst))));
   bool sample_loop = samples > 1 && reads_destination && !lower->per_sample;
   enum pipe_format raw_format = agx_apple9_color_raw_format(words);
   nir_def *src_color = color;
   nir_def *coverage = samples > 1 ? nir_load_sample_mask_in(b) : nir_imm_int(b, 1);
   if (blend && blend->disabled_samples)
      coverage = nir_iand_imm(b, coverage, ~blend->disabled_samples);
   if (lower->a2c_mask)
      coverage = nir_iand(b, coverage, lower->a2c_mask);
   if (blend && blend->alpha_to_coverage && rt == 0) {
      nir_def *mask = apple9_a2c_sample_mask(b, color, samples);
      nir_demote_samples(b, nir_u2u16(b, nir_inot(b, mask)));
      coverage = nir_iand(b, coverage, mask);
   }
   if (blend && blend->dynamic_enable) {
      nir_def *enables = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET),
         .align_mul = 4, .range = 4);
      coverage = nir_bcsel(b, nir_i2b(b, nir_ubitfield_extract_imm(b, enables, rt, 1)),
                           coverage, nir_imm_int(b, 0));
   }
   /* Alpha-to-coverage applies even when location 0 has no attachment. */
   if (rt >= lower->nr_targets || (blend && blend->color_attachment_unused)) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend && blend->alpha_to_one)
      src_color = nir_vector_insert_imm(b, src_color, nir_imm_float(b, 1), 3);
   nir_variable *sample_index = NULL;
   nir_def *sample_mask = lower->per_sample ? nir_imm_int(b, 0xff) : coverage;
   if (sample_loop) {
      sample_index = nir_local_variable_create(b->impl, glsl_uint_type(), "color sample");
      nir_store_var(b, sample_index, nir_imm_int(b, 0), 1);
      nir_push_loop(b)->control = nir_loop_control_dont_unroll;
      nir_def *index = nir_load_var(b, sample_index);
      nir_push_if(b, nir_uge_imm(b, index, samples));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);
      sample_mask = nir_ishl(b, nir_imm_int(b, 1), index);
   }
   {
      color = src_color;
      if (blend) {
         nir_def *packed_dst = reads_destination
            ? nir_load_local_pixel_agx(b, words, 32, sample_mask,
                 .base = 4 * offset, .format = raw_format)
            : nir_undef(b, words, 32);
         nir_def *channels[4];
         for (unsigned c = 0; c < (packed_format ? 0 : 4); ++c) {
            if (c >= components) {
               channels[c] = integer ? nir_imm_int(b, c == 3) : nir_imm_float(b, c == 3);
               continue;
            }
            nir_def *value = nir_channel(b, packed_dst, (bits * c) / 32);
            if (normalized) {
               nir_def *unpacked = snorm ? nir_unpack_snorm_4x8(b, value)
                                          : nir_unpack_unorm_4x8(b, value);
               channels[c] = nir_channel(b, unpacked, c);
               continue;
            }
            if (bits < 32) {
               value = nir_ushr_imm(b, value, (bits * c) % 32);
               value = integer && channel->type == UTIL_FORMAT_TYPE_SIGNED
                  ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - bits), 32 - bits)
                  : nir_iand_imm(b, value, BITFIELD_MASK(bits));
            }
            channels[c] = raw ? value : apple9_unpack_half(b, value);
         }
         nir_def *dst;
         if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
            nir_def *r = apple9_unpack_half(b, nir_ishl_imm(b, nir_iand_imm(b, packed_dst, 0x7ff), 4));
            nir_def *g = apple9_unpack_half(b, nir_ishl_imm(b, nir_iand_imm(b, nir_ushr_imm(b, packed_dst, 11), 0x7ff), 4));
            nir_def *bl = apple9_unpack_half(b, nir_ishl_imm(b, nir_ushr_imm(b, packed_dst, 22), 5));
            dst = nir_vec4(b, r, g, bl, nir_imm_float(b, 1));
         } else
            dst = packed_format ? nir_format_unpack_rgba(b, packed_dst, format)
                                : nir_vec(b, channels, 4);
         if (util_format_is_float(tile_format)) {
            if (util_format_is_unorm(format))
               dst = nir_fsat(b, dst);
            else if (util_format_is_snorm(format))
               dst = nir_fclamp(b, dst, nir_imm_float(b, -1),
                                nir_imm_float(b, 1));
         }
         if (util_format_is_srgb(format))
            dst = nir_vector_insert_imm(b, nir_format_srgb_to_linear(b, dst),
                                       nir_channel(b, dst, 3), 3);
         const nir_lower_blend_rt rt = {
            .format = format,
            .rgb = {blend->rgb_func, blend->rgb_src, blend->rgb_dst},
            .alpha = {blend->alpha_func, blend->alpha_src, blend->alpha_dst},
            .colormask = blend->colormask,
            .advanced_blend = blend->advanced_mode != 0,
            .blend_mode = blend->advanced_mode,
            .overlap = blend->advanced_overlap,
            .src_premultiplied = blend->src_premultiplied,
            .dst_premultiplied = blend->dst_premultiplied,
         };
         if (blend->logicop_enable)
            color = nir_color_logicop(b, color, dst, blend->logicop_func, format);
         else if (!integer) {
#ifndef NDEBUG
            if (!lower->src1 && agx_apple9_blend_uses_src1(blend))
               mesa_logw_once("Apple9 dual-source blend has no SRC1 output");
#endif
            color = nir_color_blend(b, color, lower->src1 ? lower->src1 : nir_undef(b, 4, 32),
                                    dst, &rt, false);
         }
         color = nir_color_mask(b, color, dst, blend->colormask);
      }
      if (util_format_is_srgb(format))
         color = nir_vector_insert_imm(b, nir_format_linear_to_srgb(b, color),
                                      nir_channel(b, color, 3), 3);
      if (tile_format != format && util_format_is_unorm(format))
         color = nir_fsat(b, color);
      else if (tile_format != format && util_format_is_snorm(format))
         color = nir_fclamp(b, color, nir_imm_float(b, -1), nir_imm_float(b, 1));
      nir_def *packed[4] = {nir_imm_int(b, 0), nir_imm_int(b, 0),
                            nir_imm_int(b, 0), nir_imm_int(b, 0)};
      if (normalized) {
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c)
            channels[c] = c < components ? nir_channel(b, color, c)
                                          : nir_imm_float(b, 0);
         nir_def *rgba = nir_vec(b, channels, 4);
         packed[0] = snorm ? nir_pack_snorm_4x8(b, rgba)
                           : nir_pack_unorm_4x8(b, rgba);
      }
      for (unsigned c = 0; c < (packed_format || normalized ? 0 : components); ++c) {
         nir_def *v = nir_channel(b, color, c);
         if (raw) {
            if (bits < 32) {
               if (channel->type == UTIL_FORMAT_TYPE_SIGNED)
                  v = nir_imin(b, nir_imax(b, v, nir_imm_int(b, -(1 << (bits - 1)))),
                               nir_imm_int(b, (1 << (bits - 1)) - 1));
               else
                  v = nir_umin(b, v, nir_imm_int(b, BITFIELD_MASK(bits)));
               v = nir_iand_imm(b, v, BITFIELD_MASK(bits));
            }
            packed[(bits * c) / 32] = nir_ior(b, packed[(bits * c) / 32],
               nir_ishl_imm(b, v, (bits * c) % 32));
         } else if (fp16) {
            v = apple9_pack_half(b, v);
            packed[c / 2] = nir_ior(b, packed[c / 2], nir_ishl_imm(b, v, 16 * (c & 1)));
         }
      }
      if (packed_format)
         packed[0] = apple9_pack_color(b, format, color);
      nir_store_local_pixel_agx(b, nir_vec(b, packed, words),
         nir_iand(b, coverage, sample_mask), nir_undef(b, 2, 16),
         .base = 4 * offset, .format = raw_format,
         .write_mask = BITFIELD_MASK(words));
   }
   if (sample_loop) {
      nir_store_var(b, sample_index, nir_iadd_imm(b, nir_load_var(b, sample_index), 1), 1);
      nir_pop_loop(b, NULL);
   }
   nir_instr_remove(&intr->instr);
   return true;
}

/* Packed colour formats with a native single-instruction pack (G16: RG11B10
 * float and RGB10A2 unorm, component 0 in the low bits); others use NIR's
 * software pack. */

static nir_def *
apple9_pack_color(nir_builder *b, enum pipe_format format, nir_def *color)
{
   switch (format) {
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return nir_pack_format_agx(b, nir_trim_vector(b, color, 3),
                                 .format = PIPE_FORMAT_R11G11B10_FLOAT);
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return nir_pack_format_agx(b, color, .format = PIPE_FORMAT_R10G10B10A2_UNORM);
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return nir_pack_format_agx(b, nir_swizzle(b, color, (unsigned[]){2, 1, 0, 3}, 4),
                                 .format = PIPE_FORMAT_R10G10B10A2_UNORM);
   default:
      return nir_format_pack_rgba(b, format, color);
   }
}

/* Vertex channels preserve integer values; packed 32-bit formats are
 * extracted with ordinary NIR shifts and masks. */
bool
agx_apple9_vertex_format_supported(enum pipe_format format)
{
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return true;
   if (format == PIPE_FORMAT_NONE)
      return false;
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
       desc->nr_channels < 1 || desc->nr_channels > 4)
      return false;
   for (unsigned c = 0; c < desc->nr_channels; ++c) {
      const struct util_format_channel_description *ch = &desc->channel[c];
      if (ch->type == UTIL_FORMAT_TYPE_VOID)
         continue;
      if (!ch->size)
         return false;
      if (desc->block.bits == 32 && ch->size <= 32 &&
          ch->shift + ch->size <= 32 &&
          (ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED))
         continue;
      if (ch->shift % ch->size)
         return false;
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT && (ch->size == 16 || ch->size == 32))
         continue;
      if ((ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED) &&
          (ch->size == 8 || ch->size == 16 || ch->size == 32))
         continue;
      return false;
   }
   return true;
}

/* The capture job is an ordinary vertex program over the decomposed stream.
 * Poly's GPU input assembly preserves API vertex/instance identities, including
 * duplicate indexed and strip vertices, independently of the hardware cache. */
struct apple9_xfb_lower {
   nir_def *raw_id;
   nir_def *vertex, *instance, *base_instance, *draw_id;
};

static bool
apple9_lower_xfb_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   struct apple9_xfb_lower *lower = data;
   nir_def *replacement = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base:
      if (&intr->def != lower->raw_id)
         replacement = lower->vertex;
      break;
   case nir_intrinsic_load_instance_id:
      replacement = lower->instance;
      break;
   case nir_intrinsic_load_base_instance:
      replacement = lower->base_instance;
      break;
   case nir_intrinsic_load_draw_id:
      replacement = lower->draw_id;
      break;
   default:
      break;
   }
   if (replacement) {
      nir_def_replace(&intr->def, replacement);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_xfb xfb = nir_intrinsic_io_xfb(intr);
   for (unsigned c = 0; c < 4; ++c) {
      if (!xfb.out[c].num_components)
         continue;
      unsigned buffer = xfb.out[c].buffer;
      unsigned stride = b->shader->info.xfb_stride[buffer] * 4;
      nir_component_mask_t mask = nir_component_mask(xfb.out[c].num_components);
      mask = (mask << c) >> nir_intrinsic_component(intr);
      nir_def *value = nir_channels(b, intr->src[0].ssa, mask);
      nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, lower->raw_id, stride),
                                     xfb.out[c].offset * 4);
      nir_store_ssbo(b, value,
                     nir_imm_int(b, AGX_APPLE9_XFB_BUFFER_BASE + buffer),
                     offset, .align_mul = 4,
                     .write_mask = nir_component_mask(value->num_components));
   }
   /* Capture precedes clipping and rasterization. The capture-only pass has
    * no raster outputs, irrespective of the API program's varying interface. */
   nir_instr_remove(&intr->instr);
   return true;
}

void
apple9_lower_xfb(nir_shader *nir, const struct agx_apple9_vertex_layout *layout)
{
   nir_io_add_intrinsic_xfb_info(nir);
   nir_builder b =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
   struct apple9_xfb_lower lower = {0};
   lower.raw_id = nir_load_vertex_id_zero_base(&b);
   nir_def *params[8];
   for (unsigned i = 0; i < ARRAY_SIZE(params); ++i)
      params[i] =
         nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, AGX_APPLE9_XFB_PARAMS),
                       nir_imm_int(&b, i * 4), .align_mul = 4,
                       .access = ACCESS_NON_WRITEABLE);
   unsigned vertices = mesa_vertices_per_prim(layout->xfb_mode);
   nir_def *logical = nir_iadd(&b, lower.raw_id, params[4]);
   nir_def *instance = nir_udiv(&b, logical, params[5]);
   nir_def *within = nir_umod(&b, logical, params[5]);
   nir_def *prim = nir_udiv_imm(&b, within, vertices);
   nir_def *vert = nir_umod_imm(&b, within, vertices);
   nir_def *mode = nir_imm_int(&b, layout->xfb_mode);
   nir_def *id = within;
   if (vertices == 2)
      id = poly_vertex_id_for_line_class(&b, mode, prim, vert,
                                         nir_udiv_imm(&b, params[5], vertices));
   else if (vertices == 3)
      id = poly_vertex_id_for_tri_class(
         &b, mode, prim, vert, nir_imm_bool(&b, layout->xfb_flatshade_first));
   id = nir_iadd(&b, id, params[0]);
   if (layout->xfb_index_size) {
      id = nir_load_ssbo(&b, 1, layout->xfb_index_size * 8,
                         nir_imm_int(&b, AGX_APPLE9_XFB_INDICES),
                         nir_imul_imm(&b, id, layout->xfb_index_size),
                         .align_mul = layout->xfb_index_size,
                         .access = ACCESS_NON_WRITEABLE);
      id = nir_iadd(&b, nir_u2u32(&b, id), params[1]);
   }
   lower.vertex = id;
   lower.instance = nir_iadd(&b, instance, params[6]);
   lower.base_instance = params[2];
   lower.draw_id = params[3];
   nir_shader_intrinsics_pass(nir, apple9_lower_xfb_intrinsic,
                              nir_metadata_control_flow, &lower);
   b.cursor = nir_after_impl(nir_shader_get_entrypoint(nir));
   nir_store_output(
      &b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_POS);
   nir->info.clip_distance_array_size = 0;
   nir->info.cull_distance_array_size = 0;
   nir->xfb_info = NULL;
}

struct apple9_vertex_lower {
   const struct agx_apple9_vertex_layout *layout;
   nir_function_impl *entrypoint;
   const uint16_t *block_read_mask;
   unsigned num_blocks;
   bool software;
   bool valid;
};

static unsigned
apple9_vertex_format_component_align(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE || !agx_apple9_vertex_format_supported(format))
      return 0;
   const struct util_format_description *desc = util_format_description(format);
   unsigned align = 0;
   for (unsigned c = 0; c < 4; ++c) {
      unsigned swizzle = desc->swizzle[c];
      if (swizzle >= PIPE_SWIZZLE_0)
         continue;
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      bool packed = (ch->size % 8) || (ch->shift % 8);
      unsigned bytes = packed ? 4u : ch->size / 8u;
      align = MAX2(align, bytes);
   }
   return align;
}

/* Vulkan guarantees only component-size alignment for unpacked 8/16-bit vertex
 * formats (1B / 2B), even with dynamic stride or dynamic buffer offsets. A
 * widened 16-bit or 32-bit load is provably aligned to target_align iff:
 *  1. attr_off is a multiple of target_align (so ro_sink + attr_off is too),
 *  2. static stride (when not dynamic) is a multiple of target_align, and
 *  3. some attribute `a` read unconditionally in `block` or a dominator of
 *     `block` on the same binding & divisor has Vulkan component alignment >=
 *     target_align, has the same offset congruence mod target_align, and ends
 *     at or before this attribute's end (so whenever this attribute is
 *     in-bounds, `a` is guaranteed in-bounds and executed on that invocation,
 *     and therefore buffer_address + base_off + offset[a] is target_align-
 *     aligned on that invocation). */
static bool
apple9_vertex_attr_proven_align(const struct apple9_vertex_lower *lower,
                                const nir_block *block,
                                unsigned attribute, unsigned target_align)
{
   const struct agx_apple9_vertex_layout *layout = lower->layout;
   if (!layout || target_align <= 1u)
      return true;
   const uint64_t attr_off = layout->offset[attribute];
   if ((attr_off % target_align) != 0)
      return false;
   if (!layout->dynamic_stride && (layout->stride[attribute] % target_align) != 0)
      return false;

   uint16_t dom_read_mask = 0;
   if (lower->block_read_mask) {
      for (const nir_block *b = block; b; b = b->imm_dom) {
         if (b->index < lower->num_blocks)
            dom_read_mask |= lower->block_read_mask[b->index];
         if (b == b->imm_dom)
            break;
      }
   }
   if (!dom_read_mask)
      return false;

   const unsigned binding = layout->buffer[attribute];
   const unsigned divisor = layout->divisor[attribute];
   const uint64_t attr_end =
      attr_off + util_format_description(layout->format[attribute])->block.bits / 8u;

   for (unsigned a = 0; a < 16; ++a) {
      if (!(dom_read_mask & (1u << a)))
         continue;
      if (layout->buffer[a] != binding || layout->divisor[a] != divisor ||
          (!layout->dynamic_stride && layout->stride[a] != layout->stride[attribute]))
         continue;
      unsigned anchor_align = apple9_vertex_format_component_align(layout->format[a]);
      if (anchor_align < target_align)
         continue;
      const uint64_t anchor_off = layout->offset[a];
      if ((anchor_off % target_align) != 0)
         continue;
      const uint64_t anchor_end =
         anchor_off + util_format_description(layout->format[a])->block.bits / 8u;
      if (anchor_end <= attr_end)
         return true;
   }
   return false;
}

static bool
apple9_vertex_format_uniform_norm(enum pipe_format format,
                                  const struct util_format_description *desc,
                                  bool *is_signed)
{
   int first_ch = util_format_get_first_non_void_channel(format);
   if (first_ch < 0)
      return false;
   const struct util_format_channel_description *ch0 = &desc->channel[first_ch];
   if (!ch0->normalized || ch0->pure_integer ||
       (ch0->type != UTIL_FORMAT_TYPE_UNSIGNED &&
        ch0->type != UTIL_FORMAT_TYPE_SIGNED))
      return false;
   for (unsigned i = 0; i < desc->nr_channels; ++i) {
      if (desc->channel[i].type == UTIL_FORMAT_TYPE_VOID)
         continue;
      if (desc->channel[i].type != ch0->type ||
          desc->channel[i].normalized != ch0->normalized ||
          desc->channel[i].pure_integer != ch0->pure_integer)
         return false;
   }
   *is_signed = ch0->type == UTIL_FORMAT_TYPE_SIGNED;
   return true;
}

static nir_def *
apple9_convert_vertex_channel(nir_builder *b, nir_def *value,
                              const struct util_format_channel_description *ch)
{
   if (ch->type == UTIL_FORMAT_TYPE_FLOAT) {
      if (ch->size == 16)
         return apple9_unpack_half(b, nir_u2u32(b, value));
      if (ch->size == 10 || ch->size == 11) {
         /* Unsigned packed floats share the half-float exponent bias.
          * Insert the missing low mantissa bits and a zero sign bit.
          */
         return apple9_unpack_half(b, nir_ishl_imm(b, value, 15 - ch->size));
      }
      return value;
   }
   if (ch->pure_integer) {
      return ch->type == UTIL_FORMAT_TYPE_SIGNED ? nir_i2i32(b, value)
                                                 : nir_u2u32(b, value);
   }
   unsigned bits = ch->size;
   /* Use hardware UNPACK_NORM for 8/16-bit normalized channels on all load
    * modes (vectorized and scalar) so conversion is bit-exact IEEE-754 and
    * invariant across passes regardless of whether a load widens. */
   if (ch->normalized && (bits == 8 || bits == 16) &&
       (ch->type == UTIL_FORMAT_TYPE_SIGNED ||
        ch->type == UTIL_FORMAT_TYPE_UNSIGNED)) {
      nir_def *u32 = nir_u2u32(b, value);
      nir_def *unpacked;
      if (ch->type == UTIL_FORMAT_TYPE_SIGNED)
         unpacked = bits == 8 ? nir_unpack_snorm_4x8(b, u32)
                              : nir_unpack_snorm_2x16(b, u32);
      else
         unpacked = bits == 8 ? nir_unpack_unorm_4x8(b, u32)
                              : nir_unpack_unorm_2x16(b, u32);
      return nir_channel(b, unpacked, 0);
   }
   if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
      value = nir_i2f32(b, nir_i2i32(b, value));
      if (ch->normalized)
         value = nir_fmax(b, nir_fmul_imm(b, value,
            1.0 / ((UINT64_C(1) << (bits - 1)) - 1)), nir_imm_float(b, -1));
      return value;
   }
   value = nir_u2f32(b, nir_u2u32(b, value));
   if (ch->normalized)
      value = nir_fmul_imm(b, value, 1.0 / ((UINT64_C(1) << bits) - 1));
   return value;
}

static nir_def *
apple9_vertex_addr_with_disp(nir_builder *b, nir_def *base, uint64_t disp)
{
   return disp ? nir_iadd_imm(b, base, disp) : base;
}

static bool
apple9_lower_vertex_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct apple9_vertex_lower *lower = data;
   if (intr->intrinsic == nir_intrinsic_store_output && lower->layout &&
       lower->layout->clip_halfz &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS) {
      if (intr->num_components != 4 || nir_intrinsic_component(intr)) {
         lower->valid = false;
         return false;
      }
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *pos = intr->src[0].ssa;
      nir_def *z = nir_fmul_imm(b, nir_fadd(b, nir_channel(b, pos, 2),
                                           nir_channel(b, pos, 3)), .5f);
      nir_src_rewrite(&intr->src[0], nir_vector_insert_imm(b, pos, z, 2));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   /* Gallium compacts elements independently of API attribute locations. */
   unsigned attribute = nir_intrinsic_base(intr);
   if (lower->layout && attribute < 16 && intr->def.bit_size == 32 &&
       lower->layout->format[attribute] == PIPE_FORMAT_NONE) {
      /* An input without a vertex attribute reads (0, 0, 0, 1). */
      b->cursor = nir_before_instr(&intr->instr);
      bool integer = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) !=
                     nir_type_float;
      nir_def *one = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0);
      nir_def *values[4];
      for (unsigned i = 0; i < intr->num_components; ++i) {
         values[i] = nir_intrinsic_component(intr) + i == 3 ? one
                                                           : nir_imm_int(b, 0);
      }
      nir_def_replace(&intr->def, nir_vec(b, values, intr->num_components));
      return true;
   }
   if (!lower->layout || attribute >= 16 || intr->def.bit_size != 32 ||
       intr->num_components < 1 || intr->num_components > 4 ||
       nir_intrinsic_component(intr) + intr->num_components > 4 ||
       !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
       lower->layout->buffer[attribute] >= 32 ||
       !agx_apple9_vertex_format_supported(lower->layout->format[attribute])) {
      lower->valid = false;
      return false;
   }
   const enum pipe_format format = lower->layout->format[attribute];
   const struct util_format_description *desc = util_format_description(format);
   const unsigned block_bytes = desc->block.bits / 8;
   assert(block_bytes > 0);
   b->cursor = nir_before_instr(&intr->instr);

   /* Classify requested non-constant channels to select a single wide load
    * whenever the format layout allows it. */
   bool any_mem_channel = false;
   bool any_packed_channel = false;
   unsigned uniform_bits = 0;
   bool uniform_bits_ok = true;
   unsigned min_byte = UINT_MAX, max_byte = 0;
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned component = nir_intrinsic_component(intr) + c;
      unsigned swizzle = desc->swizzle[component];
      if (swizzle >= PIPE_SWIZZLE_0)
         continue;
      any_mem_channel = true;
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      bool packed = (ch->size % 8) || (ch->shift % 8);
      unsigned bytes = packed ? 4 : ch->size / 8;
      unsigned add = lower->layout->offset[attribute] + (packed ? 0 : ch->shift / 8);
      if (!bytes || (!lower->layout->buffer_descriptor_ubo &&
                     ((lower->layout->stride[attribute] % bytes) || (add % bytes)))) {
         lower->valid = false;
         return false;
      }
      if (packed) {
         any_packed_channel = true;
         uniform_bits_ok = false;
      } else {
         if (!uniform_bits)
            uniform_bits = ch->size;
         else if (uniform_bits != ch->size)
            uniform_bits_ok = false;
         unsigned byte_off = ch->shift / 8;
         min_byte = MIN2(min_byte, byte_off);
         max_byte = MAX2(max_byte, byte_off + bytes);
      }
   }

   if (!any_mem_channel) {
      nir_def *components[4];
      bool pure_int = util_format_is_pure_integer(format);
      for (unsigned c = 0; c < intr->num_components; ++c) {
         unsigned swizzle = desc->swizzle[nir_intrinsic_component(intr) + c];
         unsigned value = swizzle == PIPE_SWIZZLE_1;
         components[c] = pure_int ? nir_imm_int(b, value)
                                  : nir_imm_float(b, value);
      }
      nir_def_rewrite_uses(&intr->def, nir_vec(b, components, intr->num_components));
      nir_instr_remove(&intr->instr);
      return true;
   }

   /* Choose vectorized load mode on the descriptor path (and for 32-bit
    * aligned arrays on the legacy UBO path). */
   enum {
      VTX_LOAD_SCALAR,
      VTX_LOAD_PACKED32,
      VTX_LOAD_ARRAY32,
      VTX_LOAD_ARRAY16_U32,
      VTX_LOAD_ARRAY8_U32,
      VTX_LOAD_ARRAY8_U16,
   } load_mode = VTX_LOAD_SCALAR;
   unsigned load_align = 1;

   const nir_block *proven_block =
      b->impl == lower->entrypoint ? intr->instr.block : NULL;
   if (any_packed_channel && block_bytes == 4) {
      load_mode = VTX_LOAD_PACKED32;
      load_align = 4;
   } else if (uniform_bits_ok && uniform_bits == 32) {
      load_mode = VTX_LOAD_ARRAY32;
      load_align = 4;
   } else if (lower->layout->buffer_descriptor_ubo && uniform_bits_ok &&
              uniform_bits == 16 && (block_bytes == 4 || block_bytes == 8) &&
              apple9_vertex_attr_proven_align(lower, proven_block, attribute, 4)) {
      load_mode = VTX_LOAD_ARRAY16_U32;
      load_align = 4;
   } else if (lower->layout->buffer_descriptor_ubo && uniform_bits_ok &&
              uniform_bits == 8 && block_bytes == 4 &&
              apple9_vertex_attr_proven_align(lower, proven_block, attribute, 4)) {
      load_mode = VTX_LOAD_ARRAY8_U32;
      load_align = 4;
   } else if (lower->layout->buffer_descriptor_ubo && uniform_bits_ok &&
              uniform_bits == 8 && block_bytes == 2 &&
              apple9_vertex_attr_proven_align(lower, proven_block, attribute, 2)) {
      load_mode = VTX_LOAD_ARRAY8_U16;
      load_align = 2;
   } else if (uniform_bits_ok && uniform_bits > 0) {
      load_align = uniform_bits / 8;
   }

   unsigned divisor = lower->layout->divisor[attribute];
   nir_def *index = divisor == AGX_APPLE9_VERTEX_DIVISOR_ZERO ? nir_imm_int(b, 0)
                  : divisor ? nir_udiv_imm(b, nir_load_instance_id(b), divisor)
                            : nir_load_vertex_id(b);
   nir_def *index64 = nir_u2u64(b, index);
   if (divisor) {
      nir_def *base_instance = lower->layout->capture_xfb || lower->software
         ? nir_load_base_instance(b)
         : nir_load_ubo(b, 1, 32,
                       nir_imm_int(b, lower->layout->draw_params_ubo),
                       nir_imm_int(b, 4), .align_mul = 4,
                       .range_base = 4, .range = 4);
      index = nir_iadd(b, index, base_instance);
      index64 = nir_iadd(b, index64, nir_u2u64(b, base_instance));
   }
   nir_def *offset = nir_iadd_imm(b,
      nir_imul_imm(b, index, lower->layout->stride[attribute]),
      lower->layout->offset[attribute]);
   nir_def *safe_base = NULL;
   uint64_t safe_disp = 0;
   if (lower->layout->buffer_descriptor_ubo) {
      unsigned descriptor_size = lower->layout->dynamic_stride ? 32 : 16;
      unsigned descriptor_offset =
         lower->layout->buffer[attribute] * descriptor_size;
      nir_def *descriptor = nir_load_ubo(b, 4, 32,
         nir_imm_int(b, lower->layout->buffer_descriptor_ubo),
         nir_imm_int(b, descriptor_offset), .align_mul = 16,
         .range_base = descriptor_offset, .range = 16);
      nir_def *buffer_address =
         nir_pack_64_2x32_split(b, nir_channel(b, descriptor, 0),
                                nir_channel(b, descriptor, 1));
      nir_def *buffer_size =
         nir_pack_64_2x32_split(b, nir_channel(b, descriptor, 2),
                                nir_channel(b, descriptor, 3));
      nir_def *stride;
      if (lower->layout->dynamic_stride) {
         stride = nir_load_ubo(b, 1, 32,
            nir_imm_int(b, lower->layout->buffer_descriptor_ubo),
            nir_imm_int(b, descriptor_offset + 16), .align_mul = 16,
            .range_base = descriptor_offset + 16, .range = 4);
      } else {
         stride = nir_imm_int(b, lower->layout->stride[attribute]);
      }
      /* Share base_off and vertex_base across all attributes in the binding. */
      nir_def *base_off = nir_imul(b, index64, nir_u2u64(b, stride));
      nir_def *vertex_base = nir_iadd(b, buffer_address, base_off);
      const uint64_t attr_off = lower->layout->offset[attribute];
      const uint64_t attr_end = attr_off + block_bytes;

      /* Robust vertex input treats a whole attribute as out of bounds when
       * any byte of it is. Such an attribute reads the zero sink instead of
       * its buffer, so every component loads unconditionally.
       *
       * Unbounded fetches still must read zero from a null binding, which
       * the driver describes as the zero page with a stride of 0 in the
       * table. That holds only for table strides and while the attribute
       * lies inside the zero page; otherwise the check stays.
       */
      bool unbounded = lower->layout->unbounded && lower->layout->dynamic_stride &&
                       attr_end <= AGX_ZERO_PAGE_SIZE;
      if (unbounded) {
         safe_base = vertex_base;
         safe_disp = attr_off;
      } else {
         /* Overflow-safe uniform bound limit: base_off + attr_end <= buffer_size
          * iff base_off < usub_sat(buffer_size, attr_end - 1). Because
          * buffer_size is uniform and attr_end is constant, limit hoists into
          * the preamble, avoiding per-attribute 64-bit end additions in main. */
         nir_def *end_m1 = nir_imm_int64(b, attr_end - 1);
         nir_def *limit = nir_bcsel(b, nir_uge(b, buffer_size, end_m1),
                                    nir_isub(b, buffer_size, end_m1),
                                    nir_imm_int64(b, 0));
         nir_def *valid = nir_ult(b, base_off, limit);
         nir_def *ro_sink = nir_load_ro_sink_address_poly(b);
         /* When attr_off is aligned to load_align and attr_end lies inside the
          * 16 KB zero page, keep attr_off as a constant displacement past the
          * bcsel so DEVICE_LOAD folds it directly into its immediate offset. */
         if ((attr_off % load_align) == 0 && attr_end <= AGX_ZERO_PAGE_SIZE) {
            safe_base = nir_bcsel(b, valid, vertex_base, ro_sink);
            safe_disp = attr_off;
         } else {
            nir_def *attr_addr = apple9_vertex_addr_with_disp(b, vertex_base, attr_off);
            safe_base = nir_bcsel(b, valid, attr_addr, ro_sink);
            safe_disp = 0;
         }
      }
   }

   /* Emit the single vectorized load (when applicable) before unpacking. */
   nir_def *loaded_vec = NULL;
   unsigned vec_base_unit = 0;
   nir_def *unpacked_4x8 = NULL;
   nir_def *unpacked_2x16[2] = {NULL, NULL};
   bool packed_norm_4x8 = false;
   bool packed_norm_2x16 = false;
   bool packed_half_2x16 = false;

   if (load_mode == VTX_LOAD_PACKED32) {
      if (safe_base) {
         loaded_vec = nir_load_global_constant(
            b, 1, 32, apple9_vertex_addr_with_disp(b, safe_base, safe_disp),
            .align_mul = 4);
      } else {
         loaded_vec = nir_load_ubo(
            b, 1, 32, nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
            offset, .align_mul = 4, .range = ~0u);
      }
   } else if (load_mode == VTX_LOAD_ARRAY32) {
      unsigned min_ch = min_byte / 4;
      unsigned max_ch = (max_byte - 1) / 4;
      unsigned comps = max_ch - min_ch + 1;
      vec_base_unit = min_ch;
      if (safe_base) {
         loaded_vec = nir_load_global_constant(
            b, comps, 32,
            apple9_vertex_addr_with_disp(b, safe_base, safe_disp + min_ch * 4),
            .align_mul = 4);
      } else {
         loaded_vec = nir_load_ubo(
            b, comps, 32, nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
            nir_iadd_imm(b, offset, min_ch * 4), .align_mul = 4, .range = ~0u);
      }
   } else if (load_mode == VTX_LOAD_ARRAY16_U32) {
      unsigned min_w = min_byte / 4;
      unsigned max_w = (max_byte - 1) / 4;
      unsigned words = max_w - min_w + 1;
      vec_base_unit = min_w;
      loaded_vec = nir_load_global_constant(
         b, words, 32,
         apple9_vertex_addr_with_disp(b, safe_base, safe_disp + min_w * 4),
         .align_mul = 4);
      /* Use native 2x16 unpack instructions for FLOAT16 / UNORM16 / SNORM16. */
      int first_ch = util_format_get_first_non_void_channel(format);
      if (first_ch >= 0) {
         const struct util_format_channel_description *ch0 = &desc->channel[first_ch];
         bool same = true;
         for (unsigned i = 0; i < desc->nr_channels; ++i) {
            if (desc->channel[i].type == UTIL_FORMAT_TYPE_VOID)
               continue;
            if (desc->channel[i].type != ch0->type ||
                desc->channel[i].normalized != ch0->normalized ||
                desc->channel[i].pure_integer != ch0->pure_integer) {
               same = false;
               break;
            }
         }
         if (same && ch0->type == UTIL_FORMAT_TYPE_FLOAT) {
            packed_half_2x16 = true;
            for (unsigned w = 0; w < words; ++w)
               unpacked_2x16[min_w + w] = nir_unpack_half_2x16(b, nir_channel(b, loaded_vec, w));
         } else if (same && ch0->normalized && !ch0->pure_integer &&
                    (ch0->type == UTIL_FORMAT_TYPE_UNSIGNED ||
                     ch0->type == UTIL_FORMAT_TYPE_SIGNED)) {
            packed_norm_2x16 = true;
            for (unsigned w = 0; w < words; ++w) {
               nir_def *wd = nir_channel(b, loaded_vec, w);
               unpacked_2x16[min_w + w] = ch0->type == UTIL_FORMAT_TYPE_SIGNED
                                             ? nir_unpack_snorm_2x16(b, wd)
                                             : nir_unpack_unorm_2x16(b, wd);
            }
         }
      }
   } else if (load_mode == VTX_LOAD_ARRAY8_U32) {
      loaded_vec = nir_load_global_constant(
         b, 1, 32, apple9_vertex_addr_with_disp(b, safe_base, safe_disp),
         .align_mul = 4);
      bool is_signed = false;
      if (apple9_vertex_format_uniform_norm(format, desc, &is_signed)) {
         packed_norm_4x8 = true;
         unpacked_4x8 = is_signed ? nir_unpack_snorm_4x8(b, loaded_vec)
                                  : nir_unpack_unorm_4x8(b, loaded_vec);
      }
   } else if (load_mode == VTX_LOAD_ARRAY8_U16) {
      loaded_vec = nir_u2u32(
         b, nir_load_global_constant(
               b, 1, 16, apple9_vertex_addr_with_disp(b, safe_base, safe_disp),
               .align_mul = 2));
      bool is_signed = false;
      if (apple9_vertex_format_uniform_norm(format, desc, &is_signed)) {
         packed_norm_4x8 = true;
         unpacked_4x8 = is_signed ? nir_unpack_snorm_4x8(b, loaded_vec)
                                  : nir_unpack_unorm_4x8(b, loaded_vec);
      }
   }

   nir_def *components[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned component = nir_intrinsic_component(intr) + c;
      unsigned swizzle = desc->swizzle[component];
      if (swizzle >= PIPE_SWIZZLE_0) {
         unsigned value = swizzle == PIPE_SWIZZLE_1;
         components[c] = util_format_is_pure_integer(format)
                            ? nir_imm_int(b, value) : nir_imm_float(b, value);
         continue;
      }
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      nir_def *value = NULL;
      if (load_mode == VTX_LOAD_PACKED32) {
         value = nir_ushr_imm(b, loaded_vec, ch->shift);
         value = ch->type == UTIL_FORMAT_TYPE_SIGNED
                    ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - ch->size), 32 - ch->size)
                    : nir_iand_imm(b, value, BITFIELD_MASK(ch->size));
         value = apple9_convert_vertex_channel(b, value, ch);
      } else if (load_mode == VTX_LOAD_ARRAY32) {
         unsigned ch_idx = (ch->shift / 32) - vec_base_unit;
         value = nir_channel(b, loaded_vec, ch_idx);
         value = apple9_convert_vertex_channel(b, value, ch);
      } else if (load_mode == VTX_LOAD_ARRAY16_U32) {
         unsigned w_idx = ch->shift / 32;
         unsigned half = (ch->shift / 16) & 1;
         if (packed_half_2x16 || packed_norm_2x16) {
            value = nir_channel(b, unpacked_2x16[w_idx], half);
         } else {
            nir_def *word = nir_channel(b, loaded_vec, w_idx - vec_base_unit);
            if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
               value = half ? nir_ishr_imm(b, word, 16)
                            : nir_ishr_imm(b, nir_ishl_imm(b, word, 16), 16);
            } else {
               value = half ? nir_ushr_imm(b, word, 16)
                            : nir_iand_imm(b, word, 0xffff);
            }
            value = apple9_convert_vertex_channel(b, value, ch);
         }
      } else if (load_mode == VTX_LOAD_ARRAY8_U32) {
         unsigned byte_idx = ch->shift / 8;
         if (packed_norm_4x8) {
            value = nir_channel(b, unpacked_4x8, byte_idx);
         } else {
            if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
               value = byte_idx < 3 ? nir_ishl_imm(b, loaded_vec, (3 - byte_idx) * 8)
                                    : loaded_vec;
               value = nir_ishr_imm(b, value, 24);
            } else {
               value = byte_idx ? nir_ushr_imm(b, loaded_vec, byte_idx * 8)
                                : loaded_vec;
               if (byte_idx < 3)
                  value = nir_iand_imm(b, value, 0xff);
            }
            value = apple9_convert_vertex_channel(b, value, ch);
         }
      } else if (load_mode == VTX_LOAD_ARRAY8_U16) {
         unsigned byte_idx = ch->shift / 8;
         if (packed_norm_4x8) {
            value = nir_channel(b, unpacked_4x8, byte_idx);
         } else {
            if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
               value = nir_ishr_imm(b, nir_ishl_imm(b, loaded_vec, (3 - byte_idx) * 8), 24);
            } else {
               value = byte_idx ? nir_iand_imm(b, nir_ushr_imm(b, loaded_vec, 8), 0xff)
                                : nir_iand_imm(b, loaded_vec, 0xff);
            }
            value = apple9_convert_vertex_channel(b, value, ch);
         }
      } else {
         unsigned bytes = ch->size / 8;
         if (safe_base) {
            value = nir_load_global_constant(
               b, 1, ch->size,
               apple9_vertex_addr_with_disp(b, safe_base, safe_disp + ch->shift / 8),
               .align_mul = bytes);
         } else {
            value = nir_load_ubo(
               b, 1, ch->size,
               nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
               nir_iadd_imm(b, offset, ch->shift / 8),
               .align_mul = bytes, .range = ~0u);
         }
         value = apple9_convert_vertex_channel(b, value, ch);
      }
      components[c] = value;
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, components, intr->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout, bool software)
{
   struct apple9_vertex_lower lower = {
      .layout = layout, .software = software, .valid = true,
   };
   uint16_t stack_read_mask[64];
   uint16_t *heap_read_mask = NULL;
   if (layout) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_metadata_require(impl, nir_metadata_dominance | nir_metadata_block_index);
      uint16_t *masks = NULL;
      if (impl->num_blocks <= ARRAY_SIZE(stack_read_mask)) {
         memset(stack_read_mask, 0, impl->num_blocks * sizeof(*masks));
         masks = stack_read_mask;
      } else {
         heap_read_mask = calloc(impl->num_blocks, sizeof(*masks));
         masks = heap_read_mask;
      }
      if (masks) {
         nir_foreach_block(block, impl) {
            uint16_t read_mask = 0;
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_call)
                  break;
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_terminate ||
                   intr->intrinsic == nir_intrinsic_terminate_if ||
                   intr->intrinsic == nir_intrinsic_demote ||
                   intr->intrinsic == nir_intrinsic_demote_if)
                  break;
               if (intr->intrinsic != nir_intrinsic_load_input)
                  continue;
               unsigned attribute = nir_intrinsic_base(intr);
               if (attribute >= 16 || intr->def.bit_size != 32 ||
                   intr->num_components < 1 || intr->num_components > 4 ||
                   nir_intrinsic_component(intr) + intr->num_components > 4 ||
                   !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
                   layout->buffer[attribute] >= 32 ||
                   layout->format[attribute] == PIPE_FORMAT_NONE ||
                   !agx_apple9_vertex_format_supported(layout->format[attribute]))
                  continue;
               const struct util_format_description *desc =
                  util_format_description(layout->format[attribute]);
               bool reads_mem = false;
               for (unsigned c = 0; c < intr->num_components; ++c) {
                  if (desc->swizzle[nir_intrinsic_component(intr) + c] < PIPE_SWIZZLE_0) {
                     reads_mem = true;
                     break;
                  }
               }
               if (reads_mem)
                  read_mask |= (uint16_t)(1u << attribute);
            }
            if (block->index < impl->num_blocks)
               masks[block->index] = read_mask;
         }
         lower.entrypoint = impl;
         lower.block_read_mask = masks;
         lower.num_blocks = impl->num_blocks;
      }
   }
   nir_shader_intrinsics_pass(nir, apple9_lower_vertex_input,
                              nir_metadata_control_flow, &lower);
   free(heap_read_mask);
   return lower.valid;
}

/* EXP-0111 FS-01: fragment SR 0xa0/0xa1 are integer pixel coordinates.
 * Gallium advertises upper-left, integer centers; mesa/st performs the API
 * center/origin adjustment. Keep this separate from user interpolation slots.
 * Z/W need their own interpolation model and must not be fabricated. */
bool
apple9_lower_window_position(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned component = 0;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_frag_coord_xy:
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input: {
      if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS)
         return false;
      unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
      if (!nir_src_is_const(intr->src[offset]) ||
          nir_src_as_uint(intr->src[offset]) != 0) {
         *(bool *)data = false;
         return false;
      }
      component = nir_intrinsic_component(intr);
      break;
   }
   default:
      return false;
   }
   unsigned read = nir_def_components_read(&intr->def);
   if (intr->def.bit_size != 32 || component >= 4 ||
       ((read << component) & ~0xfu)) {
      *(bool *)data = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *xy = nir_u2f32(b, nir_load_pixel_coord(b));
   nir_def *values[4];
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      if (!(read & BITFIELD_BIT(i)))
         values[i] = nir_undef(b, 1, 32);
      else if (component + i == 2)
         values[i] = nir_load_frag_coord_z(b);
      else if (component + i == 3)
         values[i] = nir_load_frag_coord_w(b);
      else
         values[i] = nir_channel(b, xy, component + i);
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, values, intr->def.num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

bool
apple9_lower_point_coord(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if ((intr->intrinsic != nir_intrinsic_load_input &&
        intr->intrinsic != nir_intrinsic_load_interpolated_input) ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PNTC)
      return false;
   unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
   unsigned component = nir_intrinsic_component(intr);
   if (!nir_src_is_const(intr->src[offset]) || nir_src_as_uint(intr->src[offset]) ||
       intr->def.bit_size != 32 || component + intr->def.num_components > 4) {
      *(bool *)data = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *xy = nir_load_point_coord(b);
   nir_def *values[4];
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      unsigned c = component + i;
      values[i] = c < 2 ? nir_channel(b, xy, c) : nir_imm_float(b, c == 3);
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, values, intr->def.num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

bool
apple9_lower_unwritten_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_varying_layout *producer = data;
   if (!producer ||
       (intr->intrinsic != nir_intrinsic_load_input &&
        intr->intrinsic != nir_intrinsic_load_interpolated_input))
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*offset))
      return false;

   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(*offset);
   unsigned component = nir_intrinsic_component(intr);
   if (location >= ARRAY_SIZE(producer->mask) ||
       location == VARYING_SLOT_PRIMITIVE_ID ||
       component + intr->num_components > 4)
      return false;

   unsigned written = producer->mask[location] >> component;
   if (!(nir_def_components_read(&intr->def) & ~written))
      return false;

   /* A viewport index the vertex stage does not write is zero. */
   if (location == VARYING_SLOT_VIEWPORT) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def_rewrite_uses_after(&intr->def,
         nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
      return true;
   }

   /* Unwritten outputs have undefined values. Preserve every produced
    * component without requiring storage for the absent interface slots. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *channels[4];
   for (unsigned c = 0; c < intr->num_components; ++c)
      channels[c] = written & BITFIELD_BIT(c)
         ? nir_channel(b, &intr->def, c) : nir_undef(b, 1, intr->def.bit_size);
   nir_def *replacement = nir_vec(b, channels, intr->num_components);
   nir_def_rewrite_uses_after(&intr->def, replacement);
   return true;
}

bool
apple9_collect_varyings(nir_shader *nir,
                        const struct agx_apple9_varying_layout *producer,
                        bool keep_prim_id_varying,
                        struct agx_apple9_varying_layout *layout,
                        const char **reason)
{
   bool fragment = nir->info.stage == MESA_SHADER_FRAGMENT;
   const bool prim_id_varying =
      keep_prim_id_varying &&
      (!producer || producer->mask[VARYING_SLOT_PRIMITIVE_ID] ||
       producer->count < AGX_APPLE9_MAX_VARYING_COMPONENTS);
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (fragment
                ? (intr->intrinsic != nir_intrinsic_load_interpolated_input &&
                   intr->intrinsic != nir_intrinsic_load_input &&
                   intr->intrinsic != nir_intrinsic_load_input_vertex)
                : intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location;
         /* The viewport index also travels as a flat varying for fragment
          * shaders; the rasterizer's packed layer/viewport word does not
          * reach them.
          */
         if (!fragment && (location == VARYING_SLOT_POS ||
                           location == VARYING_SLOT_PSIZ ||
                           location == VARYING_SLOT_LAYER))
            continue;
         if (fragment && location == VARYING_SLOT_POS) {
            *reason = "Apple9 fragment window-position input is not implemented";
            return false;
         }
         unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
         if (!nir_src_is_const(intr->src[offset]) ||
             nir_src_as_uint(intr->src[offset]) >= 32)
            goto unsupported;
         location += nir_src_as_uint(intr->src[offset]);
         unsigned component = nir_intrinsic_component(intr);
         /* Without a geometry output, the rasterizer generates this scalar.
          * Epilog FS keeps PRIMITIVE_ID in the varying layout (when room
          * remains) so the CF crossbar can bind either a producer flat varying
          * or HW 0x60. */
         if (fragment && location == VARYING_SLOT_PRIMITIVE_ID &&
             !prim_id_varying &&
             (!producer || !producer->mask[VARYING_SLOT_PRIMITIVE_ID]) && component == 0 &&
             intr->intrinsic == nir_intrinsic_load_input && intr->num_components == 1)
            continue;
         nir_alu_type type = fragment ? nir_intrinsic_dest_type(intr)
                                      : nir_intrinsic_src_type(intr);
         if (type != nir_type_float32 &&
             !((!fragment || intr->intrinsic == nir_intrinsic_load_input ||
                intr->intrinsic == nir_intrinsic_load_input_vertex) &&
               (type == nir_type_int32 || type == nir_type_uint32)))
            goto unsupported;
         if (!agx_apple9_varying_supported(location) ||
             component + intr->num_components > 4)
            goto unsupported;
         unsigned mask = fragment ? nir_def_components_read(&intr->def)
                                  : nir_intrinsic_write_mask(intr);
         layout->mask[location] |= mask << component;
      }
   }
   const bool add_prim_id =
      fragment && producer && prim_id_varying &&
      (layout->mask[VARYING_SLOT_PRIMITIVE_ID] &
       ~producer->mask[VARYING_SLOT_PRIMITIVE_ID]);
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if (producer && (layout->mask[i] & ~producer->mask[i]) &&
          !(i == VARYING_SLOT_PRIMITIVE_ID && add_prim_id)) {
         *reason = "Apple9 fragment input is not written by the vertex stage";
         return false;
      }
      layout->count += util_bitcount(layout->mask[i]);
   }
   if (producer) {
      *layout = *producer;
      if (add_prim_id) {
         layout->mask[VARYING_SLOT_PRIMITIVE_ID] = 1;
         layout->group[VARYING_SLOT_PRIMITIVE_ID] = 1;
         layout->count += 1;
      }
   }
   unsigned count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if ((layout->mask[i] & ~0xf) ||
          (layout->mask[i] && !agx_apple9_varying_supported(i)))
         goto unsupported;
      count += util_bitcount(layout->mask[i]);
   }
   if (layout->count != count || count > AGX_APPLE9_MAX_VARYING_COMPONENTS) {
      *reason = "Apple9 graphics supports at most 124 user varying scalars";
      return false;
   }
   return true;
unsupported:
   *reason = "Apple9 graphics requires constant-indexed 32-bit user varyings";
   return false;
}

bool
apple9_lower_clip_distance(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != VARYING_SLOT_CLIP_DIST0 && location != VARYING_SLOT_CLIP_DIST1)
      return false;

   const struct apple9_clip_distance_state *state = data;
   unsigned base = 4 * (location - VARYING_SLOT_CLIP_DIST0 +
                        nir_src_as_uint(intr->src[1])) + nir_intrinsic_component(intr);
   b->cursor = nir_after_instr(&intr->instr);
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned plane = base + c;
      if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)) ||
          plane >= b->shader->info.clip_distance_array_size)
         continue;
      /* Retain the ordinary varying for fragment inputs and transform
       * feedback. Only the rasterizer copy is zeroed for disabled planes. */
      nir_def *distance = state->enable & BITFIELD_BIT(plane)
         ? nir_channel(b, intr->src[0].ssa, c) : nir_imm_float(b, 0);
      nir_store_uvs_agx(b, distance, nir_imm_int(b, state->base + plane));
   }
   return true;
}

/* Raster routing shares a packed layer/viewport word, after point size and
 * before clip distances: layer in the low half, viewport in the high half.
 * Either may be written anywhere, so collect both and publish at the end. */
static bool
apple9_collect_layer_viewport(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   nir_variable **vars = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != VARYING_SLOT_LAYER && location != VARYING_SLOT_VIEWPORT)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_store_var(b, vars[location == VARYING_SLOT_VIEWPORT], intr->src[0].ssa, 1);
   /* Keep the viewport store as the fragment-visible varying. */
   if (location == VARYING_SLOT_LAYER)
      nir_instr_remove(&intr->instr);
   return true;
}

void
apple9_lower_layer_viewport_output(nir_shader *nir, unsigned slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *vars[2] = {
      nir_local_variable_create(impl, glsl_uint_type(), "layer"),
      nir_local_variable_create(impl, glsl_uint_type(), "viewport"),
   };
   nir_store_var(&b, vars[0], nir_imm_int(&b, 0), 1);
   nir_store_var(&b, vars[1], nir_imm_int(&b, 0), 1);
   nir_shader_intrinsics_pass(nir, apple9_collect_layer_viewport,
                              nir_metadata_control_flow, vars);
   b.cursor = nir_after_impl(impl);
   nir_def *word = nir_ior(&b, nir_iand_imm(&b, nir_load_var(&b, vars[0]), 0xffff),
                           nir_ishl_imm(&b, nir_load_var(&b, vars[1]), 16));
   nir_store_uvs_agx(&b, word, nir_imm_int(&b, slot));
   nir_lower_vars_to_ssa(nir);
}

nir_def *
apple9_load_fixed_point_size(nir_builder *b)
{
   return nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_POINT_SIZE_OFFSET),
      .align_mul = 4, .range = 4);
}

bool
apple9_lower_point_size(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *fixed = apple9_load_fixed_point_size(b);
   /* Points of 512 pixels or more do not rasterize; keep sizes within the
    * advertised [1, 511.9375] range. */
   nir_def *size = nir_fmin(b, nir_fmax(b, intr->src[0].ssa, nir_imm_float(b, 1)),
                            nir_imm_float(b, 511.9375f));
   nir_src_rewrite(&intr->src[0],
      nir_bcsel(b, nir_fgt_imm(b, fixed, 0), fixed, size));
   return true;
}

/* The biased cube-array compare form is not yet decoded, and misreads
 * the comparison. Take derivatives of the direction and let the common
 * cube-gradient lowering select an explicit LOD instead. */
static bool
apple9_lower_shadow_cube_array_implicit_lod(nir_builder *b, nir_instr *instr,
                                            UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || !tex->is_array ||
       !tex->is_shadow ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *dir = nir_trim_vector(b, tex->src[coord].src.ssa, 3);
   nir_def *ddx = nir_ddx(b, dir), *ddy = nir_ddy(b, dir);
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0) {
      nir_def *scale = nir_fexp2(b, tex->src[bias].src.ssa);
      ddx = nir_fmul(b, ddx, scale);
      ddy = nir_fmul(b, ddy, scale);
      nir_tex_instr_remove_src(tex, bias);
   }
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   tex->op = nir_texop_txd;
   return true;
}

/* Texel fetches below the integer part of an image view's minimum LOD are
 * out of bounds. Such fetches use an out-of-range level, which the fetch
 * bounds check then rejects.
 */
static bool
apple9_lower_fetch_min_lod(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   if (!min_lod)
      return false;
   if (tex->op == nir_texop_txf_ms) {
      /* Multisample images have a single level. */
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *x = nir_bcsel(b, nir_fge_imm(b, nir_f2f32(b, min_lod), 1.0),
                             nir_imm_intN_t(b, -1, xyz->bit_size),
                             nir_channel(b, xyz, 0));
      nir_src_rewrite(&tex->src[coord].src, nir_vector_insert_imm(b, xyz, x, 0));
      return true;
   }
   int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *level = lod >= 0 ? nir_u2u32(b, tex->src[lod].src.ssa)
                             : nir_imm_int(b, 0);
   nir_def *min_level = nir_f2u32(b, nir_ffloor(b, nir_f2f32(b, min_lod)));
   level = nir_bcsel(b, nir_uge(b, level, min_level), level,
                     nir_imm_int(b, ~0));
   if (lod >= 0)
      nir_src_rewrite(&tex->src[lod].src, level);
   else
      nir_tex_instr_add_src(tex, nir_tex_src_lod, level);
   return true;
}

/* Hardware min_lod_clamp encodes signed Q6 in either the low 12 bits of word 3
 * (2D/3D/Cube/Depth2D biased or implicit, 3D grad) or bits 16..27 alongside
 * an array layer / cube-array index (2D/2D-array grad, 2D-array, Cube-array,
 * Depth2D-array). Unsupported combinations fall back to explicit LOD. */
static bool
apple9_lower_min_lod(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   const bool native_grad = apple9_native_min_lod_grad(tex);
   const bool native_sample = apple9_native_min_lod_sample(tex, b->shader->info.stage);
   if (native_grad || native_sample) {
      const bool cube_as_array =
         !!(tex->backend_flags & AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY);
      /* 2D (0x0804 / 0x02c1) and 2D-array (0x040c / 0x03c1) use the biased
       * form for implicit samples; 3D, Cube, Cube-array, Depth2D, and
       * Depth2D-array have dedicated implicit min_lod encodings. */
      if (native_sample && tex->op == nir_texop_tex && !tex->is_shadow &&
          (tex->sampler_dim == GLSL_SAMPLER_DIM_2D || cube_as_array)) {
         nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0.0f));
         tex->op = nir_texop_txb;
      }
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      uint32_t fp_math_ctrl = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *clamped = nir_fmin(b, nir_fmax(b, min_lod, nir_imm_float(b, 0.0f)),
                                  nir_imm_float(b, 2047.0f / 64.0f));
      nir_def *fixed = nir_f2u32(b, nir_ffloor(b, nir_fmul_imm(b, clamped, 64.0f)));
      b->fp_math_ctrl = fp_math_ctrl;
      /* Q6 min_lod occupies bits 16..27 (word 3, or word 2 for 2D-array txb)
       * in every native mode except non-array biased samples (2D 0x0804/0x02c1,
       * 3D 0x2190, Depth2D 0x0824), where bias takes bits 16..27 and min_lod
       * occupies bits 0..11. */
      bool shift_high =
         native_grad || tex->op != nir_texop_txb || tex->is_array ||
         cube_as_array;
      nir_tex_instr_add_src(tex, nir_tex_src_backend2,
                            shift_high ? nir_ishl_imm(b, fixed, 16) : fixed);
      return true;
   }
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;

   nir_def *lod = nir_get_texture_lod(b, tex);
   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias)
      lod = nir_fadd(b, lod, nir_f2fN(b, bias, lod->bit_size));
   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   lod = nir_fmax(b, lod, nir_f2fN(b, min_lod, lod->bit_size));
   nir_tex_instr_add_src(tex, nir_tex_src_lod, lod);
   tex->op = nir_texop_txl;
   return true;
}

/* External (EGLImage) textures are ordinary 2D views, as on Apple8. */
static bool
apple9_lower_external(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_EXTERNAL)
      return false;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   return true;
}

void
apple9_lower_textures(nir_shader *nir)
{
   nir_shader_instructions_pass(nir, apple9_lower_external,
                                nir_metadata_all, NULL);
   nir_lower_samplers(nir);
   nir_shader_instructions_pass(nir, apple9_lower_fetch_min_lod,
                                nir_metadata_control_flow, NULL);
   agx_nir_legalize_apple9_texture_sources(nir);
   /* Gradients must follow projection. Common NIR lowering also projects
    * shadow comparators and leaves array layers unchanged. */
   const nir_lower_tex_options projection = {.lower_txp = ~0u};
   nir_lower_tex(nir, &projection);
   nir_shader_instructions_pass(nir, apple9_lower_compute_lod_query,
                                nir_metadata_control_flow, NULL);
   const nir_lower_tex_options tex_options = {
      .lower_1d = true,
      .lower_rect = true, .lower_rect_offset = true,
      .lower_invalid_implicit_lod = true,
      .lower_txf_offset = true,
      .lower_txd_cube_map = true,
      .lower_txd_clamp = true,
   };
   nir_shader_instructions_pass(nir, apple9_lower_implicit_gradients,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_rectangle,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_shadow_cube_array_implicit_lod,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_min_lod,
                                nir_metadata_control_flow, NULL);
   nir_lower_tex(nir, &tex_options);
   /* Native texel offsets need a measured offset form for the sample's final
    * form; lower any other offset with runtime sampler state, never fail. */
   bool fallback = false;
   nir_shader_instructions_pass(nir, apple9_flag_unencodable_offsets,
                                nir_metadata_all, &fallback);
   if (fallback) {
      /* Format-qualified custom borders are handled directly by the hardware
       * sampler descriptor (AGX_BORDER_COLOUR_CUSTOM), so sample_border()
       * returns the swizzled custom border without any shader lowering. */
      const char *offset_reason = NULL;
      agx_nir_lower_apple9_texture_offsets(nir, NULL, &offset_reason);
   }
   nir_shader_instructions_pass(nir, apple9_lower_buffer_texture,
                                nir_metadata_none, NULL);
   nir_shader_instructions_pass(nir, apple9_bound_texel_fetch,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texel_fetch,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texture_queries_and_layers,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texture_lod_query,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_sampler_bias,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_implicit_volume_tex,
                                nir_metadata_control_flow, NULL);
   agx_nir_lower_apple9_texture_lod(nir);
}

/* Coverage-only tests clamp interpolated depth to [0, 1]. For a single
 * sample, replace the initial early test with a depth publication clamped
 * to the driver's range. Replacing it, rather than adding another test,
 * avoids testing twice or applying stencil operations twice.
 *
 * Multisample interpolation needs a depth value per sample and continues to
 * use the native coverage test. */
bool
apple9_clamp_early_depth(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_blend *blend = data;
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       !nir_src_is_const(intr->src[0]) ||
       !nir_src_is_const(intr->src[1]) ||
       nir_src_as_uint(intr->src[0]) != 0xff ||
       nir_src_as_uint(intr->src[1]) != 0xff)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (blend->use_epilog) {
      /* Guard single-sample early-depth clamping with uniform draw sysvals so
       * EQUAL depth passes without depth clamp keep exact rasterizer Z and
       * multisample draws stay on the native coverage test. */
      nir_def *enable = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_DEPTH_CLAMP_ENABLE_OFFSET),
         .align_mul = 4, .range = 4);
      nir_def *samples_log2 = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_SAMPLES_LOG2_OFFSET),
         .align_mul = 4, .range = 4);
      nir_def *do_clamp = nir_iand(b, nir_ine_imm(b, enable, 0),
                                   nir_ieq_imm(b, samples_log2, 0));
      nir_push_if(b, do_clamp);
      {
         unsigned ubo = blend->clamp_early_depth
            ? blend->early_depth_clamp_ubo
            : AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
         unsigned offset = blend->clamp_early_depth
            ? blend->early_depth_clamp_offset_B
            : AGX_APPLE9_DEPTH_CLAMP_RANGE_OFFSET;
         nir_def *range = nir_load_ubo(b, 2, 32,
            nir_imm_int(b, ubo), nir_imm_int(b, offset),
            .align_mul = 4, .range_base = offset, .range = 8);
         nir_def *z = nir_load_frag_coord_z(b);
         z = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                      nir_channel(b, range, 1));
         nir_store_zs_agx(b, intr->src[0].ssa, z, nir_undef(b, 1, 16), .base = 1);
      }
      nir_push_else(b, NULL);
      {
         nir_sample_mask_agx(b, intr->src[0].ssa, intr->src[1].ssa);
      }
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }

   unsigned offset = blend->early_depth_clamp_offset_B;
   nir_def *range = nir_load_ubo(b, 2, 32,
      nir_imm_int(b, blend->early_depth_clamp_ubo), nir_imm_int(b, offset),
      .align_mul = 4, .range_base = offset, .range = 8);
   nir_def *z = nir_load_frag_coord_z(b);
   z = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                   nir_channel(b, range, 1));
   nir_store_zs_agx(b, intr->src[0].ssa, z, nir_undef(b, 1, 16), .base = 1);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Native depth publication carries both affected and live sample masks, and
 * tile stores take an explicit sample mask built from entry coverage. Neither
 * observes earlier demotions, so keep demoted samples out of subsequent
 * publications and stores. */
void
apple9_preserve_demotions(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool depth = false, demotes = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         depth |= op == nir_intrinsic_store_zs_agx;
         demotes |= op == nir_intrinsic_sample_mask_agx;
      }
   }
   if (!demotes)
      return;
   /* A late depth/stencil publication must follow every coverage kill,
    * including alpha-to-coverage lowered with the colour output. Early tests
    * stay before shader side effects and discard. */
   nir_instr *last_kill = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_sample_mask_agx &&
             block->cf_node.parent == &impl->cf_node)
            last_kill = instr;
      }
   }
   if (last_kill && depth && !nir->info.fs.early_fragment_tests) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr == last_kill)
               goto moved;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_zs_agx &&
                block->cf_node.parent == &impl->cf_node)
               nir_instr_move(nir_after_instr(last_kill), instr);
         }
      }
   }
moved:;
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *killed =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "killed samples");
   nir_store_var(&b, killed, nir_imm_intN_t(&b, 0, 16), 1);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_sample_mask_agx &&
             intr->intrinsic != nir_intrinsic_store_zs_agx &&
             intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *dead = nir_load_var(&b, killed);
         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            /* A later coverage update must not revive a killed sample. Keep
             * coverage-only kills (live == 0) visibly constant 0 so selection
             * distinguishes them from test-triggering sample masks. */
            bool coverage_only =
               nir_src_is_const(intr->src[1]) && !nir_src_as_uint(intr->src[1]);
            if (!coverage_only) {
               nir_src_rewrite(&intr->src[1],
                               nir_iand(&b, intr->src[1].ssa, nir_inot(&b, dead)));
            }
            nir_def *new_dead =
               coverage_only
                  ? intr->src[0].ssa
                  : nir_iand(&b, intr->src[0].ssa, nir_inot(&b, intr->src[1].ssa));
            nir_store_var(&b, killed, nir_ior(&b, dead, new_dead), 1);
         } else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            nir_def *mask = intr->src[1].ssa;
            nir_src_rewrite(&intr->src[1],
               nir_iand(&b, mask,
                        nir_inot(&b, nir_u2uN(&b, dead, mask->bit_size))));
         } else {
            nir_src_rewrite(&intr->src[0],
               nir_iand(&b, intr->src[0].ssa, nir_inot(&b, dead)));
            /* The stencil reference is published as a zero-extended word. */
            if (nir_intrinsic_base(intr) & 2)
               nir_src_rewrite(&intr->src[2], nir_u2u32(&b, intr->src[2].ssa));
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* A per-vertex mask for the software geometry path. It never interpolates:
 * poly combines the original vertices' masks before rasterization, and the
 * fragment shader reads a flat primitive mask. Hardware-only points can use
 * the per-vertex mask directly. Keep the actual distance outputs unchanged.
 */
void
apple9_pack_cull_distance(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *mask = nir_local_variable_create(impl, glsl_uint_type(), NULL);
   nir_store_var(&b, mask, nir_imm_int(&b, 0), 1);
   unsigned clip = nir->info.clip_distance_array_size;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location;
         if (loc != VARYING_SLOT_CLIP_DIST0 && loc != VARYING_SLOT_CLIP_DIST1)
            continue;
         unsigned c = 4 * (loc - VARYING_SLOT_CLIP_DIST0 +
                            nir_src_as_uint(intr->src[1])) +
                      nir_intrinsic_component(intr);
         if (c < clip || c >= clip + nir->info.cull_distance_array_size)
            continue;
         unsigned bit = BITFIELD_BIT(c - clip);
         b.cursor = nir_before_instr(instr);
         nir_def *v = nir_bcsel(&b, nir_flt_imm(&b, intr->src[0].ssa, 0),
                                    nir_imm_int(&b, bit), nir_imm_int(&b, 0));
         nir_store_var(&b, mask,
            nir_ior(&b, nir_iand_imm(&b, nir_load_var(&b, mask), ~bit), v), 1);
      }
   }
   b.cursor = nir_after_impl(impl);
   nir_store_output(&b, nir_load_var(&b, mask), nir_imm_int(&b, 0),
      .src_type = nir_type_uint32, .write_mask = 1,
      .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE);
   nir->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_CULL_PRIMITIVE);
   nir_progress(true, impl, nir_metadata_none);
}

/* The vertex stage exports, per cull half-space, a linear varying that is
 * one where the cull distance is non-negative. A primitive is culled when
 * that plane is zero at every vertex: zero here with zero screen-space
 * derivatives. Interpolated zeros are exact, so partially culled primitives
 * keep a nonzero gradient. */
void
apple9_lower_cull_distance_fs(nir_shader *nir,
                              const struct agx_apple9_varying_layout *producer)
{
   if (!producer->mask[VARYING_SLOT_CULL_PRIMITIVE] &&
       !producer->mask[VARYING_SLOT_CULL_PRIMITIVE + 1])
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (producer->cull_distance_mask) {
      nir_def *mask = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
         .dest_type = nir_type_uint32,
         .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE);
      nir_demote_if(&b, nir_ine_imm(&b, mask, 0));
      nir->info.inputs_read |= BITFIELD64_BIT(VARYING_SLOT_CULL_PRIMITIVE);
      nir->info.fs.uses_discard = true;
      nir_progress(true, impl, nir_metadata_none);
      return;
   }
   nir_def *bary = nir_load_barycentric_pixel(&b, 32,
      .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *culled = nir_imm_false(&b);
   for (unsigned slot = 0; slot < 2; ++slot) {
      u_foreach_bit(c, producer->mask[VARYING_SLOT_CULL_PRIMITIVE + slot]) {
         nir_def *q = nir_load_interpolated_input(&b, 1, 32, bary,
            nir_imm_int(&b, 0), .component = c, .dest_type = nir_type_float32,
            .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE + slot,
            .io_semantics.num_slots = 1);
         nir_def *zero = nir_iand(&b, nir_feq_imm(&b, q, 0),
            nir_iand(&b, nir_feq_imm(&b, nir_ddx(&b, q), 0),
                         nir_feq_imm(&b, nir_ddy(&b, q), 0)));
         culled = nir_ior(&b, culled, zero);
      }
   }
   nir_demote_if(&b, culled);
   nir->info.inputs_read |= BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2);
   nir->info.fs.uses_discard = true;
   nir_progress(true, impl, nir_metadata_none);
}
