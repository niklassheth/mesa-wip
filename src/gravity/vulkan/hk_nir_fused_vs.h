/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "compiler/nir/nir_builder.h"
#include "gravity/compiler/agx_compile_apple9.h"

struct hk_nir_fused_vs_lower {
   const uint8_t *vs_read_mask;
   nir_variable **vs_slot_vars;
   nir_variable *in_vertex_id;
   nir_variable *in_vertex_pos;
   nir_variable *in_instance_id;
   nir_variable *in_first_vertex;
   nir_variable *in_base_instance;
};

/* Validate that a lowered VS NIR shader is eligible for direct VBO/IBO fusion
 * inside a native Apple9 MESH (VS+GS) or OBJECT (VS+TCS) shader. */
bool hk_nir_fused_validate_vs(const nir_shader *vs,
                              const struct agx_apple9_vertex_layout *vertex,
                              const char **reason);

/* Normalize VS load_input bases, lower vertex attribute pulls against
 * HK_APPLE9_VERTEX_BUFFERS_BINDING, and lower early returns on a cloned VS. */
bool hk_nir_fused_prepare_vs(nir_shader *vs,
                             const struct agx_apple9_vertex_layout *vertex,
                             const char **reason);

/* Load a 32-bit or 64-bit parameter word from a GPU parameter buffer. */
nir_def *hk_nir_fused_param_u32(nir_builder *b, nir_def *buf, unsigned offset);
nir_def *hk_nir_fused_param_u64(nir_builder *b, nir_def *buf, unsigned offset);

/* Convert a draw-relative vertex/index position into a zero-base vertex index:
 * fetched U8/U16/U32 index with overflow-safe bounds clamping to ro_sink when
 * index_size_B != 0, or vertex_pos directly for non-indexed draws. */
nir_def *hk_nir_fused_fetch_vertex_id_zero_base(nir_builder *b,
                                                nir_def *vertex_pos,
                                                nir_def *index_buffer,
                                                nir_def *index_buffer_size,
                                                nir_def *index_size_B);

/* Intrinsics pass callback to redirect VS store_output and vertex/instance
 * sysvals to shader-temp variables prior to nir_inline_function_impl. */
bool hk_nir_fused_lower_vs_intr(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data);
