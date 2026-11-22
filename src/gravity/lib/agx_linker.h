/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/bitset.h"
#include "util/macros.h"
#include "agx_nir_lower_vbo.h"

struct nir_builder;
struct nir_def;
struct nir_shader;

/* These parts of the vertex element affect the generated code */
struct agx_velem_key {
   uint32_t divisor;
   uint16_t stride;
   uint8_t format;
   bool instanced;
};

/* Line stipple emulation: an internal passthrough GS forwards each segment's
 * start and direction in this flat varying, and the fragment shader discards
 * from the window-space distance along the segment's major axis. */
#define AGX_LINE_STIPPLE_SLOT VARYING_SLOT_VAR31
/* Strip-continuous counters loop over the strip's earlier segments, so the
 * work per draw is quadratic in its length. Beyond this many segments per
 * instance the counter restarts per segment instead. */
#define AGX_LINE_STIPPLE_MAX_STRIP_SEGMENTS 1024
/* Producers with this many user outputs leave no vec4 for the stipple
 * varying within the 124 non-position publication components. */
#define AGX_LINE_STIPPLE_MAX_USER_OUTPUTS 31

/* Round a window position to the 1/256-pixel vertex grid. */
struct nir_def *agx_nir_line_stipple_snap(struct nir_builder *b,
                                          struct nir_def *x);
/* Demote the fragment unless the stipple pattern keeps its counter. */
void agx_nir_line_stipple_test(struct nir_builder *b, struct nir_def *counter);

/* TEMP_STUB_PHASE1: The struct and declarations below are only referenced by
 * the Apple7/8 path in hk_shader.c / hk_shader.h; remove in Gate 1 after
 * vk-shader-draw merges.
 */
struct agx_fs_epilog_link_info {
   uint8_t rt_spill_base;
   uint8_t size_32;
   uint8_t loc_written;
   unsigned sample_shading : 1;
   unsigned broadcast_rt0 : 1;
   unsigned loc0_w_1 : 1;
   unsigned write_z : 1;
   unsigned write_s : 1;
   unsigned already_ran_zs : 1;
   bool sample_mask_after_force_early : 1;
   unsigned padding : 1;
};
static_assert(sizeof(struct agx_fs_epilog_link_info) == 4, "packed");

bool agx_nir_gather_vs_inputs(struct nir_shader *s,
                              BITSET_WORD *attrib_components_read);
bool agx_nir_lower_vs_input_to_prolog(struct nir_shader *s);
bool agx_nir_lower_fs_output_to_epilog(struct nir_shader *s,
                                       struct agx_fs_epilog_link_info *out);
bool agx_nir_lower_fs_active_samples_to_register(struct nir_shader *s);
bool agx_nir_lower_non_monolithic_uniforms(struct nir_shader *nir, unsigned nr);
