/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "agx_linker.h"
#include "nir.h"
#include "nir_builder.h"

/* Window-space positions recovered from NDC carry float error. Vertices are
 * snapped to a 1/256-pixel grid, so rounding to it makes a distance between
 * pixel centres an exact integer again instead of, say, 7.9999995. */
nir_def *
agx_nir_line_stipple_snap(nir_builder *b, nir_def *x)
{
   return nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, x, 256.0)),
                       1.0 / 256.0);
}

/* Demote the fragment when the stipple pattern bit of its counter is clear.
 * The pattern is the low half of the line stipple word, the factor the high.
 */
void
agx_nir_line_stipple_test(nir_builder *b, nir_def *counter)
{
   nir_def *word = nir_load_line_stipple_agx(b);
   nir_def *factor = nir_ushr_imm(b, word, 16);
   nir_def *index = nir_umod_imm(b, nir_udiv(b, counter, factor), 16);
   nir_def *bit = nir_ubitfield_extract(b, word, index, nir_imm_int(b, 1));

   nir_demote_if(b, nir_ieq_imm(b, bit, 0));
   b->shader->info.fs.uses_discard = true;
}

/* TEMP_STUB_PHASE1: The functions below are only referenced by the Apple7/8
 * path in hk_shader.c; remove in Gate 1 after vk-shader-draw merges.
 */
bool
agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr)
{
   (void)nir;
   (void)nr;
   UNREACHABLE("Apple7/8 non-monolithic shaders removed in Gravity");
}

bool
agx_nir_gather_vs_inputs(nir_shader *s, BITSET_WORD *attrib_components_read)
{
   (void)s;
   (void)attrib_components_read;
   UNREACHABLE("Apple7/8 VS prolog removed in Gravity");
}

bool
agx_nir_lower_vs_input_to_prolog(nir_shader *s)
{
   (void)s;
   UNREACHABLE("Apple7/8 VS prolog removed in Gravity");
}

bool
agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                  struct agx_fs_epilog_link_info *out)
{
   (void)s;
   (void)out;
   UNREACHABLE("Apple7/8 FS epilog removed in Gravity");
}

bool
agx_nir_lower_fs_active_samples_to_register(nir_shader *s)
{
   (void)s;
   UNREACHABLE("Apple7/8 FS epilog removed in Gravity");
}
