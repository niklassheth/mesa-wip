/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Apple9 images use hardware sparse texture ops directly, so
 * hk_apple9_lower_sparse_residency leaves non-buffer sparse textures intact
 * while lowering sparse texel buffers and residency code intrinsics.
 */
#include <stdio.h>

#include "gravity/compiler/agx_nir.h"
#include "gravity/compiler/agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "hk_shader.h"

static const nir_shader_compiler_options options = {0};

static bool
check_image(nir_texop op, bool array)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "sparse");
   bool fetch = op == nir_texop_txf;
   nir_def *handle = nir_imm_int(&b, 0x1000);
   nir_def *coord = fetch ? nir_imm_ivec3(&b, 64, 64, 0)
                          : nir_imm_vec3(&b, 0.125, 0.125, 0.0);
   coord = nir_trim_vector(&b, coord, array ? 3 : 2);
   nir_def *grad = nir_imm_vec2(&b, 0.01, 0.01);
   nir_def *value = nir_build_tex(
      &b, op, .coord = coord, .texture_handle = handle,
      .sampler_handle = nir_imm_int(&b, 0x2000),
      .lod = op == nir_texop_txl ? nir_imm_float(&b, 0.0)
             : fetch ? nir_imm_int(&b, 0) : NULL,
      .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32,
      .is_array = array, .is_sparse = true);
   nir_tex_instr *sample = nir_def_as_tex(value);
   if (op == nir_texop_txd) {
      nir_tex_instr_add_src(sample, nir_tex_src_ddx, grad);
      nir_tex_instr_add_src(sample, nir_tex_src_ddy, grad);
   }
   nir_def *res = nir_is_sparse_texels_resident(&b, 1, nir_channel(&b, value, 4));
   nir_store_global(&b, nir_vec4(&b, nir_channel(&b, value, 0),
                                 nir_channel(&b, value, 1),
                                 nir_channel(&b, value, 2),
                                 nir_b2i32(&b, res)),
                    nir_imm_int64(&b, 0), .align_mul = 4, .write_mask = 0xf);

   hk_apple9_lower_sparse_residency(b.shader);
   nir_validate_shader(b.shader, "after sparse residency lowering");

   bool ok = sample->is_sparse && sample->def.num_components == 5;
   if (!ok)
      fprintf(stderr, "sparse image op %u array %u was stripped instead of staying hw sparse\n",
              op, array);
   ralloc_free(b.shader);
   return ok;
}

static bool
check_buffer(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "sparse_buf");
   nir_def *handle = nir_imm_int(&b, 0x1000);
   nir_def *value = nir_build_tex(
      &b, nir_texop_txf, .coord = nir_imm_int(&b, 7),
      .texture_handle = handle, .dim = GLSL_SAMPLER_DIM_BUF,
      .dest_type = nir_type_float32, .is_sparse = true);
   nir_tex_instr *sample = nir_def_as_tex(value);
   nir_store_global(&b, value, nir_imm_int64(&b, 0), .align_mul = 4,
                    .write_mask = 0x1f);

   hk_apple9_lower_sparse_residency(b.shader);
   nir_validate_shader(b.shader, "after sparse buffer lowering");

   bool ok = !sample->is_sparse && sample->def.num_components == 4;
   if (!ok)
      fprintf(stderr, "sparse texel buffer was not lowered to software residency\n");
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   const nir_texop ops[] = {nir_texop_txl, nir_texop_tex, nir_texop_txd,
                            nir_texop_tg4, nir_texop_txf};
   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(ops); ++i) {
      ok &= check_image(ops[i], false);
      ok &= check_image(ops[i], true);
   }
   ok &= check_buffer();
   return ok ? 0 : 1;
}
