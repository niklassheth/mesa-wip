/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_COMPILE_APPLE9_PRIV_H
#define AGX_COMPILE_APPLE9_PRIV_H

#include "agx_apple9_ir.h"
#include "agx_apple9_mesh.h"
#include "agx_compile_apple9.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* Reserved sampler index for texelFetch / txf_ms, backed by a default sampler
 * without LOD clamps, bias, or comparison state. */
#define APPLE9_FETCH_SAMPLER 32

static inline bool
apple9_texture_uses_fetch_sampler(const nir_tex_instr *tex)
{
   return tex->sampler_index == APPLE9_FETCH_SAMPLER;
}

static inline bool
apple9_is_barycentric_coord(nir_intrinsic_op op)
{
   return op == nir_intrinsic_load_barycentric_coord_pixel ||
          op == nir_intrinsic_load_barycentric_coord_centroid ||
          op == nir_intrinsic_load_barycentric_coord_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_offset;
}

struct apple9_buffer_resource {
   enum agx_apple9_compute_resource_kind kind;
   uint8_t binding;
   bool read;
   bool write;
};

enum apple9_tile_mode {
   APPLE9_TILE_NORMAL = 0,
   APPLE9_TILE_EPILOG_MAIN = 1,
   APPLE9_TILE_EPILOG_TAIL = 2,
   APPLE9_TILE_EPILOG_TAIL_LOCKED = 3,
   APPLE9_TILE_PROLOG = 4,
   APPLE9_TILE_EPILOG_PER_SAMPLE = 5,
   APPLE9_TILE_EPILOG_TAIL_PREAMBLE = 6,
};

struct apple9_buffer_map {
   unsigned count;
   unsigned reserved_count;
   uint32_t read_mask;
   uint32_t write_mask;
   bool writes_global;
   bool descriptor_tables;
   bool force_descriptor_tables;
   enum apple9_tile_mode tile_mode;
   bool force_abi_return;
   bool defer_zs_release;
   uint8_t cf_base;
   uint8_t z_cf_index;
   unsigned native_root_words;
   const struct agx_apple9_mesh_key *native_key;
   struct apple9_buffer_resource resource[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
};

struct apple9_dag_cost {
   uint64_t weighted_spills;
   unsigned spills, instructions, gprs;
};

struct apple9_color_lower {
   const struct agx_apple9_blend *blend;
   nir_def *src1;
   nir_def *a2c_mask;
   unsigned nr_targets;
   bool per_sample;
   bool valid;
};

struct apple9_clip_distance_state {
   unsigned base, enable;
};

/* Core DAG lowering, optimization & backend entrypoints (agx_compile_apple9.c) */
FILE *apple9_log(void);
FILE *apple9_swap_log_stream(FILE *stream);
nir_scalar apple9_chase_trivial(nir_scalar scalar);
bool apple9_const_u32(nir_scalar scalar, uint32_t *value);
bool apple9_cf_list_has_control_flow(struct exec_list *list);
void apple9_texture_table_masks(const nir_tex_instr *tex, uint32_t *textures,
                                uint32_t *samplers, bool *fetch);
bool apple9_texture_supported(const nir_tex_instr *tex);
bool apple9_instruction_is_in_subset(nir_instr *instr, bool graphics);
bool apple9_collect_buffer_map(nir_shader *nir, struct apple9_buffer_map *map,
                               const char **reason);
void apple9_legalize_buffer_accesses(nir_shader *nir);
void apple9_prepare_dag(nir_shader *nir);
bool apple9_compile_dag_body(
   nir_shader *nir, struct agx_shader_part *out,
   struct agx_apple9_compute_profile *profile,
   const struct agx_apple9_varying_layout *varyings,
   const struct apple9_buffer_map *resources,
   const struct agx_apple9_uniform_constants *constants, bool preamble,
   struct apple9_dag_cost *cost, const char **reason);
bool apple9_opt_algebraic(nir_shader *nir);
void apple9_lower_idiv(nir_shader *nir);
void apple9_lower_subgroups(nir_shader *nir);
bool apple9_compile_workgroup_body_ex(
   nir_shader *nir, struct agx_shader_part *out,
   struct agx_apple9_compute_profile *profile,
   const struct agx_apple9_mesh_key *native_key,
   const struct apple9_buffer_map *reserved_resources, bool allow_preamble,
   const char **reason_out);

/* Preamble extraction, uniform hoisting & DAG orchestration (agx_apple9_preamble.c) */
bool apple9_compile_dag(
   nir_shader *nir, struct agx_shader_part *out,
   struct agx_apple9_compute_profile *profile,
   const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_mesh_key *native_key,
   const struct apple9_buffer_map *reserved_resources, bool allow_preamble,
   const char **reason);

/* Graphics NIR lowering passes (agx_nir_lower_apple9_graphics.c) */
bool apple9_lower_sample_state(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data);
bool apple9_lower_interp_at_offset(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data);
bool apple9_lower_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                                    void *data);
bool apple9_expand_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                                     void *data);
bool apple9_lower_coverage_sysval(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data);
bool apple9_lower_flat_input(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data);
bool apple9_lower_fragment_uniform(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data);
void apple9_collect_color(nir_shader *nir, bool dual_source_by_location);
nir_def *apple9_a2c_sample_mask(nir_builder *b, nir_def *color,
                                unsigned samples);
bool apple9_lower_color(nir_builder *b, nir_intrinsic_instr *intr, void *data);
void apple9_lower_xfb(nir_shader *nir,
                      const struct agx_apple9_vertex_layout *layout);
bool apple9_lower_window_position(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data);
bool apple9_lower_point_coord(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data);
bool apple9_lower_unwritten_inputs(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data);
bool apple9_collect_varyings(nir_shader *nir,
                             const struct agx_apple9_varying_layout *producer,
                             bool keep_prim_id_varying,
                             struct agx_apple9_varying_layout *layout,
                             const char **reason);
bool apple9_lower_clip_distance(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data);
void apple9_lower_layer_viewport_output(nir_shader *nir, unsigned slot);
nir_def *apple9_load_fixed_point_size(nir_builder *b);
bool apple9_lower_point_size(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data);
void apple9_lower_textures(nir_shader *nir);
bool apple9_clamp_early_depth(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data);
void apple9_preserve_demotions(nir_shader *nir);
void apple9_pack_cull_distance(nir_shader *nir);
void apple9_lower_cull_distance_fs(
   nir_shader *nir, const struct agx_apple9_varying_layout *producer);

#endif /* AGX_COMPILE_APPLE9_PRIV_H */
