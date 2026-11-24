/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile_apple9_priv.h"
#include "gravity/debug.h"
#include "gravity/lib/agx_abi.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned
apple9_io_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

struct apple9_vs_prolog_lower {
   const struct agx_apple9_vertex_layout *layout;
   struct agx_apple9_vs_prolog_iface *iface;
   uint8_t reg_map[16][4];
   bool valid;
};

static bool
apple9_collect_vs_prolog_iface(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   struct apple9_vs_prolog_lower *state = data;
   if (intr->intrinsic == nir_intrinsic_store_output && state->layout &&
       state->layout->clip_halfz &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS) {
      if (intr->num_components != 4 || nir_intrinsic_component(intr)) {
         state->valid = false;
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

   unsigned attribute = nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);
   if (attribute >= 16 || intr->def.bit_size != 32 ||
       intr->num_components < 1 || intr->num_components > 4 ||
       comp + intr->num_components > 4 ||
       !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0])) {
      state->valid = false;
      return false;
   }
   unsigned read = nir_def_components_read(&intr->def);
   if (!read)
      return false;
   state->iface->attribs_read |= BITFIELD_BIT(attribute);
   state->iface->attrib_mask[attribute] |= (uint8_t)(read << comp);
   bool integer = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) !=
                  nir_type_float;
   if (integer)
      state->iface->integer_attribs |= BITFIELD_BIT(attribute);
   return false;
}

static bool
apple9_rewrite_vs_inputs_to_prolog(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data)
{
   const struct apple9_vs_prolog_lower *state = data;
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned attribute = nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);
   unsigned read = nir_def_components_read(&intr->def);
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *channels[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      if (read & BITFIELD_BIT(c)) {
         unsigned reg = state->reg_map[attribute][comp + c];
         channels[c] = nir_load_exported_agx(b, 1, 32, .base = reg);
      } else {
         channels[c] = nir_undef(b, 1, 32);
      }
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, channels, intr->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
apple9_lower_vertex_inputs_to_prolog(nir_shader *nir,
                                     const struct agx_apple9_vertex_layout *layout,
                                     struct agx_apple9_vs_prolog_iface *iface)
{
   memset(iface, 0, sizeof(*iface));
   nir_opt_dce(nir);

   struct apple9_vs_prolog_lower state = {
      .layout = layout,
      .iface = iface,
      .valid = true,
   };
   nir_shader_intrinsics_pass(nir, apple9_collect_vs_prolog_iface,
                              nir_metadata_control_flow, &state);
   if (!state.valid)
      return false;

   unsigned reg = AGX_APPLE9_ABI_REG_BASE;
   for (unsigned a = 0; a < 16; ++a) {
      uint8_t mask = iface->attrib_mask[a] & 0xf;
      if (!mask) {
         iface->attribs_read &= ~BITFIELD_BIT(a);
         continue;
      }
      u_foreach_bit(c, mask) {
         if (reg >= AGX_APPLE9_ABI_REG_BASE + AGX_APPLE9_MAX_ABI_REGS)
            return false;
         state.reg_map[a][c] = reg++;
      }
   }
   iface->uses_prolog = iface->attribs_read != 0;
   nir_shader_intrinsics_pass(nir, apple9_rewrite_vs_inputs_to_prolog,
                              nir_metadata_control_flow, &state);
   return true;
}

static void
apple9_seed_vs_prolog_resources(struct apple9_buffer_map *reserved,
                                uint16_t buffer_descriptor_ubo,
                                uint16_t draw_params_ubo)
{
   memset(reserved, 0, sizeof(*reserved));
   if (buffer_descriptor_ubo) {
      reserved->resource[reserved->count++] = (struct apple9_buffer_resource){
         .kind = AGX_APPLE9_COMPUTE_RESOURCE_UBO,
         .binding = buffer_descriptor_ubo,
         .read = true,
      };
   }
   if (draw_params_ubo && draw_params_ubo != buffer_descriptor_ubo) {
      reserved->resource[reserved->count++] = (struct apple9_buffer_resource){
         .kind = AGX_APPLE9_COMPUTE_RESOURCE_UBO,
         .binding = draw_params_ubo,
         .read = true,
      };
   }
   reserved->reserved_count = reserved->count;
}

static void
apple9_seed_fs_epilog_resources(struct apple9_buffer_map *reserved,
                                enum apple9_tile_mode tile_mode,
                                bool force_abi_return)
{
   memset(reserved, 0, sizeof(*reserved));
   reserved->resource[reserved->count++] = (struct apple9_buffer_resource){
      .kind = AGX_APPLE9_COMPUTE_RESOURCE_UBO,
      .binding = AGX_APPLE9_GRAPHICS_SYSVAL_BINDING,
      .read = true,
   };
   reserved->reserved_count = reserved->count;
   reserved->tile_mode = tile_mode;
   reserved->force_abi_return = force_abi_return;
}

struct apple9_fs_epilog_lower {
   struct agx_apple9_fs_epilog_iface *iface;
   uint8_t loc_base[9];
   uint8_t depth_reg;
   uint8_t stencil_reg;
   bool early_fragment_tests;
   bool valid;
};

static bool
apple9_collect_fs_epilog_outputs(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   struct apple9_fs_epilog_lower *state = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location == FRAG_RESULT_SAMPLE_MASK)
      return false;
   if (sem.location == FRAG_RESULT_DEPTH) {
      if (!state->early_fragment_tests)
         state->iface->writes_depth = true;
      return false;
   }
   if (sem.location == FRAG_RESULT_STENCIL) {
      if (!state->early_fragment_tests)
         state->iface->writes_stencil = true;
      return false;
   }
   if (sem.location < FRAG_RESULT_DATA0 ||
       sem.location >= FRAG_RESULT_DATA0 + 8 ||
       intr->src[0].ssa->bit_size != 32 ||
       !intr->num_components ||
       nir_intrinsic_component(intr) + intr->num_components > 4) {
      state->valid = false;
      return false;
   }
   if (sem.dual_source_blend_index) {
      if (sem.location != FRAG_RESULT_DATA0) {
         state->valid = false;
         return false;
      }
      state->iface->writes_src1 = true;
   } else {
      state->iface->outputs_written |= BITFIELD_BIT(sem.location - FRAG_RESULT_DATA0);
   }
   return false;
}

static bool
apple9_rewrite_fs_outputs_to_epilog(nir_builder *b, nir_intrinsic_instr *intr,
                                    void *data)
{
   const struct apple9_fs_epilog_lower *state = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   b->cursor = nir_before_instr(&intr->instr);
   if (sem.location == FRAG_RESULT_DEPTH) {
      if (!state->early_fragment_tests)
         nir_export_agx(b, nir_f2f32(b, intr->src[0].ssa),
                        .base = state->depth_reg);
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (sem.location == FRAG_RESULT_STENCIL) {
      if (!state->early_fragment_tests)
         nir_export_agx(b, nir_iand_imm(b, nir_u2u32(b, intr->src[0].ssa), 0xffff),
                        .base = state->stencil_reg);
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (sem.location < FRAG_RESULT_DATA0 ||
       sem.location >= FRAG_RESULT_DATA0 + 8)
      return false;
   unsigned slot = sem.dual_source_blend_index
                      ? 8
                      : (sem.location - FRAG_RESULT_DATA0);
   unsigned base_reg = state->loc_base[slot];
   unsigned comp = nir_intrinsic_component(intr);
   unsigned mask = nir_intrinsic_write_mask(intr);
   nir_def *color = intr->src[0].ssa;
   for (unsigned c = 0; c < 4; ++c) {
      bool written = c >= comp && c < comp + intr->num_components &&
                     (mask & BITFIELD_BIT(c - comp));
      nir_def *val = written ? nir_channel(b, color, c - comp)
                             : nir_imm_int(b, 0);
      nir_export_agx(b, val, .base = base_reg + c);
   }
   nir_instr_remove(&intr->instr);
   return true;
}

struct apple9_epilog_demote_state {
   nir_variable *killed;
   bool any;
};

static bool
apple9_lower_demote_samples_to_kill(nir_builder *b, nir_intrinsic_instr *intr,
                                    void *data)
{
   if (intr->intrinsic != nir_intrinsic_demote_samples)
      return false;
   struct apple9_epilog_demote_state *state = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *mask = nir_u2u16(b, intr->src[0].ssa);
   nir_sample_mask_agx(b, mask, nir_imm_intN_t(b, 0, 16));
   if (state->killed) {
      nir_def *prev = nir_load_var(b, state->killed);
      nir_store_var(b, state->killed, nir_ior(b, prev, mask), 1);
   }
   nir_instr_remove(&intr->instr);
   state->any = true;
   return true;
}

static bool
apple9_track_demote_samples_var(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data)
{
   if (intr->intrinsic != nir_intrinsic_demote_samples)
      return false;
   struct apple9_epilog_demote_state *state = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *mask = nir_u2u16(b, intr->src[0].ssa);
   nir_def *prev = nir_load_var(b, state->killed);
   nir_store_var(b, state->killed, nir_ior(b, prev, mask), 1);
   state->any = true;
   return true;
}

static bool
apple9_lower_active_samples_from_abi(nir_builder *b, nir_intrinsic_instr *intr,
                                     UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *raw = nir_load_exported_agx(b, 1, 32,
                                        .base = AGX_APPLE9_SAMPLE_MASK_ABI_REG);
   nir_def_replace(&intr->def, nir_u2uN(b, raw, intr->def.bit_size));
   return true;
}

static bool
apple9_mask_tile_stores_by_killed(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
      return false;
   nir_def *killed = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *mask = intr->src[1].ssa;
   nir_src_rewrite(&intr->src[1],
                   nir_iand(b, mask, nir_inot(b, nir_u2uN(b, killed, mask->bit_size))));
   return true;
}

static bool
apple9_lower_color_to_epilog(nir_shader *nir,
                             struct agx_apple9_fs_epilog_iface *iface,
                             unsigned *next_reg)
{
   bool early_z = nir->info.fs.early_fragment_tests;
   bool writes_frag_color = iface->writes_frag_color;
   bool per_sample = iface->per_sample;
   memset(iface, 0, sizeof(*iface));
   iface->writes_frag_color = writes_frag_color;
   iface->early_fragment_tests = early_z;
   iface->per_sample = per_sample;
   struct apple9_fs_epilog_lower state = {
      .iface = iface,
      .early_fragment_tests = early_z,
      .valid = true,
   };
   nir_shader_intrinsics_pass(nir, apple9_collect_fs_epilog_outputs,
                              nir_metadata_control_flow, &state);
   if (!state.valid)
      return false;
   unsigned reg = AGX_APPLE9_ABI_REG_BASE;
   u_foreach_bit(loc, iface->outputs_written) {
      state.loc_base[loc] = reg;
      reg += 4;
   }
   if (iface->writes_src1) {
      state.loc_base[8] = reg;
      reg += 4;
   }
   if (iface->writes_depth)
      state.depth_reg = reg++;
   if (iface->writes_stencil)
      state.stencil_reg = reg++;
   /* Reserve room for optional killed_reg (1 GPR) and stay below
    * AGX_APPLE9_SAMPLE_MASK_ABI_REG (r63). */
   if (reg + 1 > AGX_APPLE9_SAMPLE_MASK_ABI_REG ||
       reg + 1 > AGX_APPLE9_ABI_REG_BASE + AGX_APPLE9_MAX_ABI_REGS)
      return false;
   iface->uses_epilog = true;
   *next_reg = reg;
   nir_shader_intrinsics_pass(nir, apple9_rewrite_fs_outputs_to_epilog,
                              nir_metadata_control_flow, &state);
   nir->info.outputs_written &= ~(BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                  BITFIELD64_BIT(FRAG_RESULT_STENCIL));
   return true;
}

static bool
apple9_compile_graphics(nir_shader *nir, struct agx_shader_part *out,
                        const struct agx_apple9_vertex_layout *layout,
                        const struct agx_apple9_varying_layout *producer,
                        const struct agx_apple9_blend *blend,
                        unsigned nr_targets, const char **reason)
{
   bool preserves_color = false;
   struct agx_apple9_vs_prolog_iface vs_prolog = {0};
   struct agx_apple9_fs_epilog_iface fs_epilog = {0};
   struct apple9_buffer_map reserved_resources = {0};
   const bool fs_use_epilog =
      nir->info.stage == MESA_SHADER_FRAGMENT && blend && blend[0].use_epilog;
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   if (reason)
      *reason = NULL;
   /* Legacy gl_FragColor broadcasts to each enabled draw buffer. */
   agx_nir_lower_printf_buffer(nir);
   apple9_lower_subgroups(nir);
   bool writes_frag_color = false;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      writes_frag_color = !!(nir->info.outputs_written &
                             BITFIELD64_BIT(FRAG_RESULT_COLOR));
      nir_foreach_shader_out_variable(var, nir) {
         if (var->data.location == FRAG_RESULT_COLOR)
            writes_frag_color = true;
      }
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(nir_instr_as_intrinsic(instr))
                      .location == FRAG_RESULT_COLOR)
               writes_frag_color = true;
         }
      }
      nir_lower_fragcolor(nir, fs_use_epilog ? 1 : nr_targets);
   }
   /* Move varying arrays into local storage before flattening the interface.
    * Hardware slots remain static while shader-selected array elements are
    * handled by the same SSA lowering as local arrays. */
   nir_lower_io_vars_to_temporaries(nir, nir_shader_get_entrypoint(nir),
      nir->info.stage == MESA_SHADER_VERTEX ? nir_var_shader_out : nir_var_shader_in);
   nir_lower_global_vars_to_local(nir);
   nir_split_var_copies(nir);
   nir_lower_var_copies(nir);
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out, apple9_io_size,
                nir_lower_io_use_interpolated_input_intrinsics);
   if (nir->info.stage == MESA_SHADER_VERTEX &&
       nir->info.cull_distance_array_size) {
      nir_lower_io_to_scalar(nir, nir_var_shader_out, NULL, NULL);
      if (layout && layout->cull_distance_mask)
         apple9_pack_cull_distance(nir);
      else
         agx_nir_lower_cull_distance_vs(nir);
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT && producer)
      apple9_lower_cull_distance_fs(nir, producer);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* Position may be wholly or partially undefined, including shaders
       * used only for side effects with rasterization disabled. The native
       * export record still requires four words. Choose zero for components
       * which the shader never writes. */
      unsigned written = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS)
               written |= nir_intrinsic_write_mask(intr) <<
                          nir_intrinsic_component(intr);
         }
      }
      if (written != 0xf) {
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 0), nir_imm_int(&b, 0),
            .write_mask = 0xf & ~written, .src_type = nir_type_float32,
            .io_semantics.location = VARYING_SLOT_POS,
            .io_semantics.num_slots = 1);
      }
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      unsigned fc_targets = fs_use_epilog ? 1 : nr_targets;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
            if (semantics.location == FRAG_RESULT_COLOR) {
               semantics.location = FRAG_RESULT_DATA0;
               nir_intrinsic_set_io_semantics(intr, semantics);
               for (unsigned rt = 1; rt < fc_targets; ++rt) {
                  nir_instr *copy = nir_instr_clone(nir, &intr->instr);
                  semantics.location = FRAG_RESULT_DATA0 + rt;
                  nir_intrinsic_set_io_semantics(nir_instr_as_intrinsic(copy),
                                                 semantics);
                  nir_instr_insert(nir_before_instr(&intr->instr), copy);
               }
            }
         }
      }
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_COLOR)) {
         nir->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_COLOR);
         nir->info.outputs_written |= BITFIELD64_MASK(fc_targets)
                                      << FRAG_RESULT_DATA0;
      }
   }

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (layout && layout->ignore_point_size)
         nir_remove_outputs(nir, MESA_SHADER_FRAGMENT, 0,
                            BITFIELD64_BIT(VARYING_SLOT_PSIZ));
      if (layout && layout->use_prolog) {
         if (!apple9_lower_vertex_inputs_to_prolog(nir, layout, &vs_prolog)) {
            *reason = "Apple9 vertex prolog requires supported vertex inputs";
            return false;
         }
         if (vs_prolog.uses_prolog) {
            apple9_seed_vs_prolog_resources(&reserved_resources,
                                            layout->buffer_descriptor_ubo,
                                            layout->draw_params_ubo);
         }
      } else if (!agx_nir_lower_apple9_vertex_inputs(nir, layout, false)) {
         *reason = "Apple9 vertex inputs require supported naturally aligned formats";
         return false;
      }
      if (layout && layout->capture_xfb)
         apple9_lower_xfb(nir, layout);
      if (layout && layout->rasterize_points && !layout->capture_xfb) {
         /* mesa/st cannot insert point size when the API varying interface
          * is full. The hardware still needs the separate system output. */
         if (!nir_shader_intrinsics_pass(nir, apple9_lower_point_size,
                                           nir_metadata_control_flow, NULL)) {
            nir_builder b = nir_builder_at(
               nir_after_impl(nir_shader_get_entrypoint(nir)));
            /* APIs without a fixed size (zero) default to one pixel. */
            nir_def *fixed = apple9_load_fixed_point_size(&b);
            fixed = nir_bcsel(&b, nir_fgt_imm(&b, fixed, 0), fixed,
                              nir_imm_float(&b, 1.0f));
            nir_store_output(&b, fixed, nir_imm_int(&b, 0),
               .io_semantics.location = VARYING_SLOT_PSIZ,
               .io_semantics.num_slots = 1, .write_mask = 1,
               .src_type = nir_type_float32);
         }
      }
   }
   /* Use the same structured mask/loop model as compute. Continuations become
    * masked regions with one backedge before instruction selection. */
   nir_lower_continue_constructs(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      apple9_collect_color(nir, !fs_use_epilog && blend && blend[0].dual_source_by_location);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      /* Opaque visibility can discard earlier fragments for the whole pixel.
       * An omitted or masked target still needs their values, even when this
       * shader has no explicit tile load. Use ordered tile access, as for
       * blending, to preserve those attachments. */
      for (unsigned rt = 0; !fs_use_epilog && rt < nr_targets; ++rt) {
         /* With VK_EXT_dynamic_rendering_unused_attachments, a pipeline can
          * leave an attachment of the render undeclared. The attachment is
          * still bound and holds earlier draws' values, so preserve it.
          * Only a target with no attachment needs nothing. */
         if (blend && blend[rt].color_attachment_unused) {
            preserves_color |= blend[rt].format != PIPE_FORMAT_NONE;
            continue;
         }
         bool written = nir->info.outputs_written &
                        BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
         preserves_color |= !written ||
            (blend && (!blend[rt].colormask || blend[rt].dynamic_enable));
      }
   }
   /* As on Apple8, promote direct accesses to SSA, keep small arrays in
    * registers, and lower larger dynamically indexed arrays to scratch. */
   nir_lower_array_deref_of_vec(nir, nir_var_function_temp, NULL,
                               nir_lower_indirect_array_deref_of_vec_load |
                               nir_lower_indirect_array_deref_of_vec_store);
   nir_lower_vars_to_ssa(nir);
   nir_lower_vars_to_scratch(nir, 128, glsl_get_natural_size_align_bytes,
                              glsl_get_word_size_align_bytes);
   nir_lower_indirect_derefs_to_if_else_trees(nir, nir_var_function_temp, ~0u);
   /* Frontends may supply vector NIR or pre-simplified scalar NIR. Expose
    * scalar constants to standard algebraic cleanup before expanding math
    * and output packing, rather than relying on a caller-specific pipeline.
    */
   nir_lower_vars_to_ssa(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_lower_all_phis_to_scalar(nir);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_undef(nir);
      progress |= nir_opt_loop_unroll(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      unsigned samples = blend ? MAX2(blend[0].samples, 1) : 1;
      bool dynamic_samples = blend && blend[0].dynamic_samples;
      bool single_sampled =
         !dynamic_samples &&
         (samples == 1 || (blend && blend[0].multisample_disabled));
      if (single_sampled) {
         const nir_lower_single_sampled_options options = {0};
         nir_lower_single_sampled(nir, &options);
         nir->info.fs.uses_sample_shading = false;
      }
      nir_shader_intrinsics_pass(nir, apple9_lower_barycentric_coord,
                                 nir_metadata_control_flow, &single_sampled);
      nir_shader_intrinsics_pass(nir, apple9_expand_barycentric_coord,
                                 nir_metadata_control_flow, NULL);
      nir_shader_intrinsics_pass(nir, apple9_lower_interp_at_offset,
                                 nir_metadata_control_flow, NULL);
      agx_nir_lower_sample_intrinsics(nir,
         !(blend && blend[0].sample_mask_single_sample));
      nir_shader_intrinsics_pass(nir, apple9_lower_sample_state,
                                 nir_metadata_control_flow, (void *)blend);
      if (fs_use_epilog) {
         bool epilog_per_sample =
            nir->info.fs.uses_sample_shading || blend[0].per_sample;
         fs_epilog.writes_frag_color = writes_frag_color;
         fs_epilog.per_sample = epilog_per_sample;
         unsigned epilog_next_reg = 0;
         if (!apple9_lower_color_to_epilog(nir, &fs_epilog, &epilog_next_reg)) {
            if (reason)
               *reason = "Apple9 fragment epilog requires FP32/32-bit color outputs";
            return false;
         }
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         agx_nir_lower_discard_zs_emit(nir);
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_variable *killed_var = nir_local_variable_create(
            impl, glsl_uint16_t_type(), "epilog killed");
         if (nir->info.fs.early_fragment_tests) {
            struct apple9_epilog_demote_state dstate = {
               .killed = killed_var,
               .any = false,
            };
            nir_shader_intrinsics_pass(nir, apple9_track_demote_samples_var,
                                       nir_metadata_control_flow, &dstate);
            nir_shader_gather_info(nir, impl);
            agx_nir_lower_sample_mask(nir);
            /* Kill disabled API samples before early ZS tests and memory
             * side effects so disabled samples never pass early tests, and
             * initialize killed_var with them before any later demote. */
            {
               nir_builder b0 = nir_builder_at(nir_before_impl(impl));
               nir_def *api_mask = nir_load_ubo(&b0, 1, 32,
                  nir_imm_int(&b0, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                  nir_imm_int(&b0, AGX_APPLE9_API_SAMPLE_MASK_OFFSET),
                  .align_mul = 4, .range = 4);
               nir_def *disabled = nir_iand_imm(&b0,
                  nir_inot(&b0, nir_u2u16(&b0, api_mask)), 0xff);
               nir_sample_mask_agx(&b0, disabled, nir_imm_intN_t(&b0, 0, 16));
               nir_store_var(&b0, killed_var, disabled, 1);
            }
            if (epilog_per_sample)
               agx_nir_lower_to_per_sample(nir);
            nir_shader_intrinsics_pass(nir, apple9_clamp_early_depth,
                                       nir_metadata_none, (void *)blend);
            apple9_preserve_demotions(nir);
            if (dstate.any) {
               nir_builder bend = nir_builder_at(nir_after_impl(impl));
               nir_export_agx(&bend, nir_u2u32(&bend, nir_load_var(&bend, killed_var)),
                              .base = epilog_next_reg);
               fs_epilog.has_killed_mask = true;
            }
            fs_epilog.main_defers_zs_test = false;
         } else {
            nir_builder b0 = nir_builder_at(nir_before_impl(impl));
            nir_store_var(&b0, killed_var, nir_imm_intN_t(&b0, 0, 16), 1);
            if (nir->info.writes_memory &&
                !(blend && blend[0].writes_memory_stats_only)) {
               nir_def *api_mask = nir_load_ubo(&b0, 1, 32,
                  nir_imm_int(&b0, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                  nir_imm_int(&b0, AGX_APPLE9_API_SAMPLE_MASK_OFFSET),
                  .align_mul = 4, .range = 4);
               nir_def *disabled = nir_iand_imm(&b0,
                  nir_inot(&b0, nir_u2u16(&b0, api_mask)), 0xff);
               nir_demote_samples(&b0, disabled);
            }
            if (epilog_per_sample)
               agx_nir_lower_to_per_sample(nir);
            struct apple9_epilog_demote_state dstate = {
               .killed = killed_var,
               .any = false,
            };
            nir_shader_intrinsics_pass(nir, apple9_lower_demote_samples_to_kill,
                                       nir_metadata_control_flow, &dstate);
            if (dstate.any) {
               nir_builder bend = nir_builder_at(nir_after_impl(impl));
               nir_export_agx(&bend, nir_u2u32(&bend, nir_load_var(&bend, killed_var)),
                              .base = epilog_next_reg);
               fs_epilog.main_defers_zs_test = true;
               fs_epilog.has_killed_mask = true;
            }
         }
         if (epilog_per_sample) {
            nir_shader_intrinsics_pass(nir, apple9_lower_active_samples_from_abi,
                                       nir_metadata_control_flow, NULL);
         }
         if (samples == 1 && !dynamic_samples && !epilog_per_sample) {
            nir_lower_single_sampled_options lss_opts = {
               .lower_sample_mask_in = true,
            };
            nir_lower_single_sampled(nir, &lss_opts);
         }
         nir_lower_vars_to_ssa(nir);
         nir_lower_all_phis_to_scalar(nir);
         nir_shader_intrinsics_pass(nir, apple9_lower_coverage_sysval,
                                    nir_metadata_control_flow, NULL);
         apple9_seed_fs_epilog_resources(
            &reserved_resources,
            epilog_per_sample ? APPLE9_TILE_EPILOG_PER_SAMPLE
                              : APPLE9_TILE_EPILOG_MAIN,
            true);
         reserved_resources.defer_zs_release = epilog_per_sample;
      } else {
         struct apple9_color_lower color = {
            .blend = blend, .nr_targets = nr_targets, .valid = true};
         nir_shader_intrinsics_pass(nir, apple9_lower_color,
                                    nir_metadata_none, &color);
         nir_lower_vars_to_ssa(nir);
         nir_lower_all_phis_to_scalar(nir);
         if (blend && blend[0].disabled_samples) {
            nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
            nir_demote_samples(&b, nir_imm_intN_t(&b, blend[0].disabled_samples, 16));
         }
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         agx_nir_lower_discard_zs_emit(nir);
         agx_nir_lower_monolithic_msaa(nir, samples);
         if (samples == 1 && blend && blend[0].clamp_early_depth &&
             nir->info.fs.early_fragment_tests) {
            nir_shader_intrinsics_pass(nir, apple9_clamp_early_depth,
                                       nir_metadata_control_flow, (void *)blend);
         }
         apple9_preserve_demotions(nir);
         nir_lower_vars_to_ssa(nir);
         nir_lower_all_phis_to_scalar(nir);
         nir_shader_intrinsics_pass(nir, apple9_lower_coverage_sysval,
                                    nir_metadata_control_flow, NULL);
         if (!color.valid) {
            if (reason)
               *reason = "Apple9 render requires FP32 color outputs and standard single-source blending";
            return false;
         }
      }
   }
   nir_shader_intrinsics_pass(nir, apple9_lower_fragment_uniform,
                              nir_metadata_control_flow, NULL);
   nir_opt_dce(nir);
   apple9_lower_textures(nir);
   apple9_lower_idiv(nir);
   /* Format lowering can introduce powers and other high-level ALU ops. */
   apple9_opt_algebraic(nir);
   agx_nir_opt_apple9_bounds(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      bool valid = true;
      nir_shader_intrinsics_pass(nir, apple9_lower_window_position,
                                 nir_metadata_control_flow, &valid);
      nir_shader_intrinsics_pass(nir, apple9_lower_point_coord,
                                 nir_metadata_control_flow, &valid);
      nir_shader_intrinsics_pass(nir, apple9_lower_unwritten_inputs,
                                 nir_metadata_control_flow, (void *)producer);
      if (!valid) {
         *reason = "Apple9 fragment window position requires FP32 XYZW components";
         return false;
      }
      nir_lower_alu_to_scalar(nir, NULL, NULL);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   uint32_t texture_mask = 0, sampler_mask = 0;
   bool uses_texel_fetch = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_image_load_apple9_agx &&
                nir_intrinsic_base(intr) == 0)
               uses_texel_fetch = true;
            continue;
         }
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (!apple9_texture_supported(tex)) {
            if (agx_apple9_trace_enabled())
               nir_print_instr(instr, stderr);
            *reason = "Apple9 requires supported FP32 texture operations";
            return false;
         }
         apple9_texture_table_masks(tex, &texture_mask, &sampler_mask,
                                    &uses_texel_fetch);
      }
   }
   if (util_bitcount(texture_mask) > AGX_APPLE9_GRAPHICS_MAX_TEXTURES ||
       util_bitcount(sampler_mask) + uses_texel_fetch >
          AGX_APPLE9_GRAPHICS_MAX_SAMPLERS) {
      *reason = "Apple9 supports 32 textures and 17 physical samplers per stage";
      return false;
   }
   struct agx_apple9_varying_layout varyings = {0};
   if (!apple9_collect_varyings(nir, producer, fs_use_epilog, &varyings, reason))
      return false;
   if (layout && layout->cull_distance_mask &&
       varyings.mask[VARYING_SLOT_CULL_PRIMITIVE])
      varyings.cull_distance_mask = true;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_shader_intrinsics_pass(nir, apple9_lower_flat_input,
                                 nir_metadata_control_flow, &varyings);
      nir_opt_cse(nir);
      nir_opt_dce(nir);
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_sample_mask_agx)
            nir->info.fs.uses_discard = true;
      }
   }
   struct apple9_buffer_map buffers = reserved_resources;
   if (!apple9_collect_buffer_map(nir, &buffers, reason))
      return false;
   bool valid_buffers = buffers.count <= AGX_APPLE9_MAX_GRAPHICS_BUFFERS;
   for (unsigned i = 0; i < buffers.count; ++i)
      valid_buffers &=
         buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_SSBO
            ? buffers.resource[i].binding < 32
            : buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_UBO;
   if (!valid_buffers) {
      if (reason)
         *reason =
            "Apple9 graphics requires at most 32 supported buffer bindings";
      return false;
   }
   if (agx_apple9_trace_enabled())
      nir_print_shader(nir, stderr);
   if (layout) {
      for (unsigned i = 0; i < ARRAY_SIZE(varyings.mask); ++i) {
         if (varyings.mask[i])
            varyings.group[i] = agx_apple9_varying_group(
               layout->outputs_flat, layout->outputs_linear, i);
      }
   }
   if (varyings.cull_distance_mask) {
      varyings.group[VARYING_SLOT_CULL_PRIMITIVE] = 1;
   } else {
      if (varyings.mask[VARYING_SLOT_CULL_PRIMITIVE])
         varyings.group[VARYING_SLOT_CULL_PRIMITIVE] = 2;
      if (varyings.mask[VARYING_SLOT_CULL_PRIMITIVE + 1])
         varyings.group[VARYING_SLOT_CULL_PRIMITIVE + 1] = 2;
   }
   bool writes_layer = nir->info.outputs_written &
                       (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);
   if (nir->info.stage == MESA_SHADER_VERTEX && writes_layer) {
      apple9_lower_layer_viewport_output(nir, 4 + varyings.count +
         !!(nir->info.outputs_written & VARYING_BIT_PSIZ));
   }
   if (nir->info.stage == MESA_SHADER_VERTEX && nir->info.clip_distance_array_size) {
      if (nir->info.clip_distance_array_size > 8) {
         *reason = "Apple9 supports eight clip distances";
         return false;
      }
      const struct apple9_clip_distance_state state = {
         .base = 4 + varyings.count + !!(nir->info.outputs_written & VARYING_BIT_PSIZ) +
                 writes_layer,
         .enable = layout ? layout->clip_distance_enable : 0xff,
      };
      nir_shader_intrinsics_pass(nir, apple9_lower_clip_distance,
                                 nir_metadata_control_flow, (void *)&state);
   }
   const bool has_reserved =
      reserved_resources.count || reserved_resources.tile_mode ||
      reserved_resources.force_abi_return;
   if (!apple9_compile_dag(nir, out, NULL, &varyings, NULL,
                           has_reserved ? &reserved_resources : NULL,
                           true, reason))
      return false;
   fs_epilog.has_tile_lock = out->info.apple9_fs_epilog.has_tile_lock;
   fs_epilog.has_tile_preamble = out->info.apple9_fs_epilog.has_tile_preamble;
   out->info.apple9_vs_prolog = vs_prolog;
   out->info.apple9_fs_epilog = fs_epilog;
   out->info.apple9_reads_tile |= preserves_color;
   return true;
}

bool
agx_compile_apple9_fragment_epilog_key(
   const struct agx_apple9_fs_epilog_key *key,
   struct agx_shader_part *out, const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   *reason = NULL;

   if (!key || !key->iface.uses_epilog || key->nr_targets > 8) {
      *reason = "invalid Apple9 fragment epilog key";
      return false;
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_fs_epilog");
   nir_shader *nir = b.shader;

   nir_def *loaded[8] = {NULL};
   nir_def *loaded_src1 = NULL;
   nir_def *loaded_depth = NULL;
   nir_def *loaded_stencil = NULL;
   nir_def *loaded_killed = NULL;
   unsigned reg = AGX_APPLE9_ABI_REG_BASE;
   u_foreach_bit(loc, key->iface.outputs_written) {
      nir_def *ch[4];
      for (unsigned c = 0; c < 4; ++c)
         ch[c] = nir_load_exported_agx(&b, 1, 32, .base = reg++);
      loaded[loc] = nir_vec(&b, ch, 4);
   }
   if (key->iface.writes_src1) {
      nir_def *ch[4];
      for (unsigned c = 0; c < 4; ++c)
         ch[c] = nir_load_exported_agx(&b, 1, 32, .base = reg++);
      loaded_src1 = nir_vec(&b, ch, 4);
   }
   if (key->iface.writes_depth)
      loaded_depth = nir_load_exported_agx(&b, 1, 32, .base = reg++);
   if (key->iface.writes_stencil)
      loaded_stencil = nir_load_exported_agx(&b, 1, 32, .base = reg++);
   if (key->iface.has_killed_mask)
      loaded_killed = nir_u2u16(&b, nir_load_exported_agx(&b, 1, 32, .base = reg++));

   struct agx_apple9_blend blend[8];
   memcpy(blend, key->blend, sizeof(blend));
   const unsigned nr_targets = key->nr_targets;
   const unsigned samples = MAX2(blend[0].samples, 1);
   const bool early_z = key->iface.early_fragment_tests;
   bool emitted_demote = false;

   if (!early_z && loaded_killed) {
      nir_demote_samples(&b, loaded_killed);
      emitted_demote = true;
   }
   if (!early_z && blend[0].disabled_samples) {
      nir_demote_samples(&b, nir_imm_intN_t(&b, blend[0].disabled_samples, 16));
      emitted_demote = true;
   }

   /* Alpha-to-coverage always reads fragment output location 0's alpha before
    * attachment remapping (color_map) and masks every color target (RT0..7).
    * Under early_fragment_tests, Z/S already ran in fs_main so apply the A2C
    * mask directly to every tile store without emitting a late demote. */
   nir_def *a2c_mask = NULL;
   if (blend[0].alpha_to_coverage) {
      if (loaded[0]) {
         nir_def *mask = apple9_a2c_sample_mask(&b, loaded[0], samples);
         if (early_z) {
            a2c_mask = mask;
         } else {
            nir_demote_samples(&b, nir_u2u16(&b, nir_inot(&b, mask)));
            emitted_demote = true;
         }
      }
      blend[0].alpha_to_coverage = 0;
   }

   nir_def *zero = nir_imm_int(&b, 0);

   /* Depth/stencil exports (explicit from fs_main, or synthetic late_zs for
    * depth/stencil attachment feedback loops). Emit after coverage kills and
    * alpha-to-coverage but BEFORE color outputs so store_zs_agx performs the
    * depth/stencil test before store_local_pixel_agx writes color to the tile. */
   if (!early_z) {
      nir_def *depth_out = loaded_depth;
      if (!depth_out && key->late_zs) {
         nir_def *z = nir_load_frag_coord_z(&b);
         nir_def *clamp_en = nir_load_ubo(&b, 1, 32,
            nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(&b, AGX_APPLE9_DEPTH_CLAMP_ENABLE_OFFSET),
            .align_mul = 4, .range = 4);
         nir_def *range = nir_load_ubo(&b, 2, 32,
            nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(&b, AGX_APPLE9_DEPTH_CLAMP_RANGE_OFFSET),
            .align_mul = 4, .range = 8);
         nir_def *clamped = nir_fmin(&b,
            nir_fmax(&b, z, nir_channel(&b, range, 0)),
            nir_channel(&b, range, 1));
         z = nir_bcsel(&b, nir_ine_imm(&b, clamp_en, 0), clamped, z);
         nir_def *force_never = nir_load_ubo(&b, 1, 32,
            nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(&b, AGX_APPLE9_FORCE_NEVER_OFFSET),
            .align_mul = 4, .range = 4);
         depth_out = nir_bcsel(&b, nir_ine_imm(&b, force_never, 0),
                               nir_imm_float(&b, 2.0f), z);
      }
      if (depth_out) {
         nir_store_output(&b, depth_out, zero, .write_mask = 1,
            .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});
         nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
      }
      if (loaded_stencil) {
         nir_store_output(&b, loaded_stencil, zero, .write_mask = 1,
            .src_type = nir_type_uint32,
            .io_semantics = {.location = FRAG_RESULT_STENCIL, .num_slots = 1});
         nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
      }
   }

   if (loaded_src1) {
      nir_store_output(&b, loaded_src1, zero, .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0,
                          .dual_source_blend_index = 1,
                          .num_slots = 1});
   }
   for (unsigned rt = 0; rt < nr_targets; ++rt) {
      nir_def *val = NULL;
      if (key->iface.writes_frag_color) {
         val = loaded[0];
      } else {
         unsigned loc = key->color_map[rt];
         if (loc < 8)
            val = loaded[loc];
      }
      if (!val)
         continue;
      nir_store_output(&b, val, zero, .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0 + rt, .num_slots = 1});
   }

   bool preserves_color = false;
   for (unsigned rt = 0; rt < nr_targets; ++rt) {
      if (blend[rt].color_attachment_unused) {
         preserves_color |= blend[rt].format != PIPE_FORMAT_NONE;
         continue;
      }
      bool written;
      if (key->iface.writes_frag_color) {
         written = !!(key->iface.outputs_written & BITFIELD_BIT(0));
      } else {
         unsigned loc = key->color_map[rt];
         written = loc < 8 && (key->iface.outputs_written & BITFIELD_BIT(loc));
      }
      preserves_color |= !written || !blend[rt].colormask ||
                         blend[rt].dynamic_enable;
   }

   struct apple9_color_lower color = {
      .blend = blend,
      .a2c_mask = a2c_mask,
      .nr_targets = nr_targets,
      .per_sample = key->per_sample,
      .valid = true,
   };
   nir_shader_intrinsics_pass(nir, apple9_lower_color,
                              nir_metadata_none, &color);
   if (!color.valid) {
      ralloc_free(nir);
      *reason = "Apple9 fragment epilog requires supported blend and color format";
      return false;
   }
   if (early_z && loaded_killed) {
      nir_shader_intrinsics_pass(nir, apple9_mask_tile_stores_by_killed,
                                 nir_metadata_control_flow, loaded_killed);
   }
   nir_lower_vars_to_ssa(nir);
   nir_lower_all_phis_to_scalar(nir);

   /* Check if alpha-to-coverage inside apple9_lower_color emitted a demote. */
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_demote_samples)
            emitted_demote = true;
      }
   }
   bool writes_zs = !!(nir->info.outputs_written &
                       (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                        BITFIELD64_BIT(FRAG_RESULT_STENCIL)));
   if (!early_z && !writes_zs &&
       (key->iface.main_defers_zs_test || key->prolog_defers_zs_test) &&
       !emitted_demote) {
      nir_builder b0 = nir_builder_at(
         nir_before_impl(nir_shader_get_entrypoint(nir)));
      nir_demote_samples(&b0, nir_imm_intN_t(&b0, 0, 16));
      emitted_demote = true;
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   agx_nir_lower_discard_zs_emit(nir);
   if (emitted_demote) {
      nir->info.fs.uses_discard = true;
      agx_nir_lower_sample_mask(nir);
   }
   if (key->per_sample) {
      agx_nir_lower_to_per_sample(nir);
      nir_shader_intrinsics_pass(nir, apple9_lower_active_samples_from_abi,
                                 nir_metadata_control_flow, NULL);
   } else if (samples == 1) {
      nir_lower_single_sampled_options lss_opts = {
         .lower_sample_mask_in = true,
      };
      nir_lower_single_sampled(nir, &lss_opts);
   }
   apple9_preserve_demotions(nir);
   nir_lower_vars_to_ssa(nir);
   nir_lower_all_phis_to_scalar(nir);
   nir_shader_intrinsics_pass(nir, apple9_lower_coverage_sysval,
                              nir_metadata_control_flow, NULL);
   nir_shader_intrinsics_pass(nir, apple9_lower_fragment_uniform,
                              nir_metadata_control_flow, NULL);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_undef(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);
   apple9_lower_idiv(nir);
   apple9_opt_algebraic(nir);
   agx_nir_opt_apple9_bounds(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_sample_mask_agx)
            nir->info.fs.uses_discard = true;
      }
   }

   struct apple9_buffer_map reserved = {0};
   enum apple9_tile_mode tile_mode =
      key->per_sample ? APPLE9_TILE_EPILOG_PER_SAMPLE
      : key->iface.has_tile_lock ? APPLE9_TILE_EPILOG_TAIL_LOCKED
      : key->iface.has_tile_preamble ? APPLE9_TILE_EPILOG_TAIL_PREAMBLE
                                     : APPLE9_TILE_EPILOG_TAIL;
   apple9_seed_fs_epilog_resources(&reserved, tile_mode, key->per_sample);
   reserved.defer_zs_release = key->per_sample;
   if (key->late_zs && !key->iface.writes_depth && !early_z)
      reserved.z_cf_index = key->z_cf_index;

   bool ok = apple9_compile_dag(
      nir, out, NULL, NULL, NULL, &reserved, false /* allow_preamble */, reason);
   if (ok)
      out->info.apple9_reads_tile |= preserves_color;
   ralloc_free(nir);
   return ok;
}

bool
agx_compile_apple9_cull_prolog(
   const struct agx_apple9_cull_prolog_key *key,
   struct agx_shader_part *out, const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   *reason = NULL;

   if (!key || key->cf_base == 0 ||
       (key->mask ? key->count > 1 : (key->count == 0 || key->count > 8))) {
      *reason = "invalid Apple9 cull prolog key";
      return false;
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_cull_prolog");
   nir_shader *nir = b.shader;

   struct agx_apple9_varying_layout varyings = {0};
   nir_def *culled;
   if (key->mask) {
      varyings.mask[VARYING_SLOT_CULL_PRIMITIVE] = 1;
      varyings.group[VARYING_SLOT_CULL_PRIMITIVE] = 1;
      varyings.count = 1;
      varyings.cull_distance_mask = true;
      nir_def *mask = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
         .dest_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE,
                          .num_slots = 1});
      culled = nir_ine_imm(&b, mask, 0);
   } else {
      for (unsigned i = 0; i < key->count; ++i) {
         unsigned loc = VARYING_SLOT_CULL_PRIMITIVE + i / 4;
         varyings.mask[loc] |= BITFIELD_BIT(i % 4);
         varyings.group[loc] = 2;
      }
      varyings.count = key->count;
      nir_def *bary = nir_load_barycentric_pixel(&b, 32,
         .interp_mode = INTERP_MODE_NOPERSPECTIVE);
      culled = nir_imm_false(&b);
      for (unsigned i = 0; i < key->count; ++i) {
         nir_def *q = nir_load_interpolated_input(&b, 1, 32, bary,
            nir_imm_int(&b, 0), .component = i % 4,
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE + i / 4,
                             .num_slots = 1});
         nir_def *zero = nir_iand(&b, nir_feq_imm(&b, q, 0),
            nir_iand(&b, nir_feq_imm(&b, nir_ddx(&b, q), 0),
                         nir_feq_imm(&b, nir_ddy(&b, q), 0)));
         culled = nir_ior(&b, culled, zero);
      }
   }

   if (key->standalone) {
      nir_def *live = nir_bcsel(&b, culled, nir_imm_intN_t(&b, 0, 16),
                                nir_imm_intN_t(&b, 0xff, 16));
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16), live);
   } else {
      nir_push_if(&b, culled);
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                          nir_imm_intN_t(&b, 0, 16));
      nir_jump(&b, nir_jump_halt);
      nir_pop_if(&b, NULL);
   }
   nir->info.fs.uses_discard = true;

   nir_lower_alu_to_scalar(nir, NULL, NULL);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);

   struct apple9_buffer_map reserved = {
      .tile_mode = APPLE9_TILE_PROLOG,
      .force_abi_return = !key->standalone,
      .cf_base = key->cf_base,
   };
   bool ok = apple9_compile_dag(nir, out, NULL, &varyings, NULL, &reserved,
                                false /* allow_preamble */, reason);
   ralloc_free(nir);
   return ok;
}

bool
agx_compile_apple9_stipple_prolog(
   const struct agx_apple9_stipple_prolog_key *key,
   struct agx_shader_part *out, const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   *reason = NULL;

   if (!key || key->cf_base == 0) {
      *reason = "invalid Apple9 line-stipple prolog key";
      return false;
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_stipple_prolog");
   nir_shader *nir = b.shader;

   /* TEX6.xyz (flat, group 1) -> indices 0..2; TEX7.xy (linear, group 2) -> 3..4. */
   struct agx_apple9_varying_layout varyings = {0};
   varyings.mask[VARYING_SLOT_TEX6] = 0x7;
   varyings.group[VARYING_SLOT_TEX6] = 1;
   varyings.mask[VARYING_SLOT_TEX7] = 0x3;
   varyings.group[VARYING_SLOT_TEX7] = 2;
   varyings.count = 5;

   nir_def *line = nir_load_input(&b, 3, 32, nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TEX6, .num_slots = 1});
   nir_def *start = nir_trim_vector(&b, line, 2);
   nir_def *bary = nir_load_barycentric_pixel(&b, 32,
      .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *on_line = nir_load_interpolated_input(&b, 2, 32, bary,
      nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TEX7, .num_slots = 1});
   nir_def *axis = nir_fabs(&b, nir_fsub(&b, on_line, start));
   nir_def *x_major =
      nir_fge(&b, nir_channel(&b, axis, 0), nir_channel(&b, axis, 1));
   nir_def *pixel = nir_fadd_imm(
      &b, nir_u2f32(&b, nir_u2u32(&b, nir_load_pixel_coord(&b))), 0.5);
   nir_def *d = nir_fabs(&b, nir_fsub(&b, pixel, start));
   nir_def *counter = nir_f2u32(
      &b, nir_fadd(&b, nir_channel(&b, line, 2),
                   nir_bcsel(&b, x_major, nir_channel(&b, d, 0),
                             nir_channel(&b, d, 1))));
   nir_def *word = nir_load_line_stipple_agx(&b);
   nir_def *factor = nir_ushr_imm(&b, word, 16);
   nir_def *index = nir_umod_imm(&b, nir_udiv(&b, counter, factor), 16);
   nir_def *bit = nir_ubitfield_extract(&b, word, index, nir_imm_int(&b, 1));
   nir_def *culled = nir_ieq_imm(&b, bit, 0);

   if (key->standalone) {
      nir_def *live = nir_bcsel(&b, culled, nir_imm_intN_t(&b, 0, 16),
                                nir_imm_intN_t(&b, 0xff, 16));
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16), live);
   } else {
      nir_push_if(&b, culled);
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                          nir_imm_intN_t(&b, 0, 16));
      nir_jump(&b, nir_jump_halt);
      nir_pop_if(&b, NULL);
   }
   nir->info.fs.uses_discard = true;

   nir_shader_intrinsics_pass(nir, apple9_lower_fragment_uniform,
                              nir_metadata_control_flow, NULL);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);
   apple9_lower_idiv(nir);
   apple9_opt_algebraic(nir);
   agx_nir_opt_apple9_bounds(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);

   struct apple9_buffer_map reserved = {0};
   apple9_seed_fs_epilog_resources(&reserved, APPLE9_TILE_PROLOG,
                                   !key->standalone);
   reserved.cf_base = key->cf_base;
   bool ok = apple9_compile_dag(nir, out, NULL, &varyings, NULL, &reserved,
                                false /* allow_preamble */, reason);
   ralloc_free(nir);
   return ok;
}

static bool
apple9_emit_vertex_prolog_inputs(
   nir_builder *b, const struct agx_apple9_vertex_prolog_key *key,
   bool software, const char **reason)
{
   unsigned reg = AGX_APPLE9_ABI_REG_BASE;
   for (unsigned a = 0; a < 16; ++a) {
      if (!(key->iface.attribs_read & BITFIELD_BIT(a)))
         continue;
      uint8_t mask = key->iface.attrib_mask[a] & 0xf;
      if (!mask)
         continue;
      unsigned first = ffs(mask) - 1;
      unsigned last = util_last_bit(mask) - 1;
      unsigned comps = last - first + 1;
      nir_alu_type dest_type =
         (key->iface.integer_attribs & BITFIELD_BIT(a)) ? nir_type_uint32
                                                        : nir_type_float32;
      nir_def *v = nir_load_input(
         b, comps, 32, nir_imm_int(b, 0), .base = a, .component = first,
         .dest_type = dest_type,
         .io_semantics.location = VERT_ATTRIB_GENERIC0 + a,
         .io_semantics.num_slots = 1);
      u_foreach_bit(c, mask) {
         if (reg >= AGX_APPLE9_ABI_REG_BASE + AGX_APPLE9_MAX_ABI_REGS) {
            *reason = software
                         ? "Apple9 SW-VS prolog exceeds ABI register capacity"
                         : "Apple9 vertex prolog exceeds ABI register capacity";
            return false;
         }
         nir_export_agx(b, nir_channel(b, v, c - first), .base = reg++);
      }
   }

   struct agx_apple9_vertex_layout layout = {
      .draw_params_ubo = key->draw_params_ubo,
      .buffer_descriptor_ubo = key->buffer_descriptor_ubo,
      .dynamic_stride = key->dynamic_stride,
      .unbounded = key->unbounded,
   };
   memcpy(layout.format, key->format, sizeof(layout.format));
   memcpy(layout.offset, key->offset, sizeof(layout.offset));
   memcpy(layout.stride, key->stride, sizeof(layout.stride));
   memcpy(layout.divisor, key->divisor, sizeof(layout.divisor));
   memcpy(layout.buffer, key->buffer, sizeof(layout.buffer));

   if (!agx_nir_lower_apple9_vertex_inputs(b->shader, &layout, software)) {
      *reason = software
                   ? "Apple9 SW-VS prolog requires supported naturally aligned formats"
                   : "Apple9 vertex prolog requires supported naturally aligned formats";
      return false;
   }

   return true;
}

bool
agx_compile_apple9_vertex_prolog_key(
   const struct agx_apple9_vertex_prolog_key *key,
   struct agx_shader_part *out, const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   *reason = NULL;

   if (!key || !key->iface.uses_prolog || !key->iface.attribs_read) {
      *reason = "Apple9 vertex prolog key has no active attributes";
      return false;
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "apple9_vs_prolog");
   nir_shader *nir = b.shader;

   if (!apple9_emit_vertex_prolog_inputs(&b, key, false, reason)) {
      ralloc_free(nir);
      return false;
   }

   nir_lower_alu_to_scalar(nir, NULL, NULL);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_undef(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);

   apple9_lower_idiv(nir);
   apple9_opt_algebraic(nir);
   agx_nir_opt_apple9_bounds(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);

   struct apple9_buffer_map reserved = {0};
   apple9_seed_vs_prolog_resources(&reserved, key->buffer_descriptor_ubo,
                                   key->draw_params_ubo);

   bool ok = apple9_compile_dag(
      nir, out, NULL, NULL, NULL,
      reserved.count ? &reserved : NULL, false /* allow_preamble */, reason);
   ralloc_free(nir);
   return ok;
}

static void
apple9_seed_sw_vs_prolog_resources(struct apple9_buffer_map *reserved,
                                   uint16_t buffer_descriptor_ubo,
                                   uint16_t draw_params_ubo,
                                   uint16_t vertex_params_ubo,
                                   bool force_descriptor_tables,
                                   bool force_abi_return)
{
   memset(reserved, 0, sizeof(*reserved));
   uint16_t bindings[3] = {
      buffer_descriptor_ubo, draw_params_ubo, vertex_params_ubo,
   };
   bool have_ubo[3] = {
      buffer_descriptor_ubo != 0,
      draw_params_ubo != 0,
      true, /* vertex_params_ubo (commonly UBO 0, the root table) */
   };
   for (unsigned i = 0; i < 3; ++i) {
      if (!have_ubo[i])
         continue;
      bool dup = false;
      for (unsigned j = 0; j < reserved->count; ++j) {
         if (reserved->resource[j].binding == bindings[i]) {
            dup = true;
            break;
         }
      }
      if (dup)
         continue;
      reserved->resource[reserved->count++] = (struct apple9_buffer_resource){
         .kind = AGX_APPLE9_COMPUTE_RESOURCE_UBO,
         .binding = bindings[i],
         .read = true,
      };
   }
   reserved->reserved_count = reserved->count;
   reserved->force_descriptor_tables = force_descriptor_tables;
   reserved->force_abi_return = force_abi_return;
}

bool
agx_nir_lower_apple9_sw_vs_inputs_to_prolog(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_apple9_vs_prolog_iface *iface)
{
   return apple9_lower_vertex_inputs_to_prolog(nir, layout, iface);
}

bool
agx_compile_apple9_sw_vs_body(
   nir_shader *nir, uint16_t buffer_descriptor_ubo, uint16_t draw_params_ubo,
   uint16_t vertex_params_ubo, struct agx_shader_part *out,
   struct agx_apple9_compute_profile *profile, const char **reason)
{
   struct apple9_buffer_map reserved = {0};
   apple9_seed_sw_vs_prolog_resources(&reserved, buffer_descriptor_ubo,
                                      draw_params_ubo, vertex_params_ubo,
                                      false, false);
   return apple9_compile_workgroup_body_ex(nir, out, profile, NULL, &reserved,
                                           true, reason);
}

struct apple9_sw_vs_prolog_sysvals {
   nir_def *vertex_id;
   nir_def *instance_id;
   nir_def *base_instance;
};

static bool
apple9_lower_sw_vs_prolog_sysvals(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data)
{
   const struct apple9_sw_vs_prolog_sysvals *sv = data;
   if (intr->intrinsic == nir_intrinsic_load_vertex_id) {
      nir_def_replace(&intr->def, sv->vertex_id);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_instance_id) {
      nir_def_replace(&intr->def, sv->instance_id);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_base_instance) {
      nir_def_replace(&intr->def, sv->base_instance);
      return true;
   }
   return false;
}

bool
agx_compile_apple9_sw_vs_prolog_key(
   const struct agx_apple9_sw_vs_prolog_key *key,
   struct agx_shader_part *out, struct agx_apple9_compute_profile *profile,
   const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   if (profile)
      memset(profile, 0, sizeof(*profile));
   *reason = NULL;

   if (!key || !key->base.iface.uses_prolog || !key->base.iface.attribs_read) {
      *reason = "Apple9 SW-VS prolog key has no active attributes";
      return false;
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "apple9_sw_vs_prolog");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size_variable = false;
   nir->info.workgroup_size[0] = 64;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   if (!apple9_emit_vertex_prolog_inputs(&b, &key->base, true /* software */,
                                         reason)) {
      ralloc_free(nir);
      return false;
   }

   /* Lower load_vertex_id / load_instance_id / load_base_instance identically
    * to poly_nir_lower_sw_vs + hk_apple9_lower_prerast_sysvals. */
   bool reads_vid = false, reads_iid = false, reads_base_inst = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         reads_vid |= op == nir_intrinsic_load_vertex_id;
         reads_iid |= op == nir_intrinsic_load_instance_id;
         reads_base_inst |= op == nir_intrinsic_load_base_instance;
      }
   }
   if (reads_vid || reads_iid || reads_base_inst) {
      nir_builder b0 = nir_builder_at(
         nir_before_impl(nir_shader_get_entrypoint(nir)));
      struct apple9_sw_vs_prolog_sysvals sv = {0};
      nir_def *gid = (reads_vid || reads_iid)
                        ? nir_load_global_invocation_id(&b0, 32)
                        : NULL;
      if (reads_vid) {
         nir_def *id = nir_channel(&b0, gid, 0);
         nir_def *vp = nir_load_ubo(&b0, 1, 64,
            nir_imm_int(&b0, key->vertex_params_ubo),
            nir_imm_int(&b0, key->vertex_params_offset_B),
            .align_mul = 8, .range_base = key->vertex_params_offset_B,
            .range = 8);
         nir_def *index_size = nir_load_global_constant(&b0, 1, 32,
            nir_iadd_imm(&b0, vp, offsetof(struct poly_vertex_params, index_size_B)),
            .align_mul = 4);
         nir_def *index_buffer_id;
         nir_if *indexed = nir_push_if(&b0, nir_ine_imm(&b0, index_size, 0));
         {
            index_buffer_id = poly_load_index_buffer(&b0, vp, id, index_size);
         }
         nir_pop_if(&b0, indexed);
         nir_def *effective_id = nir_if_phi(&b0, index_buffer_id, id);
         nir_def *first_vertex = nir_load_ubo(&b0, 1, 32,
            nir_imm_int(&b0, key->base.draw_params_ubo), nir_imm_int(&b0, 0),
            .align_mul = 4, .range_base = 0, .range = 4);
         sv.vertex_id = nir_iadd(&b0, effective_id, first_vertex);
      }
      if (reads_iid)
         sv.instance_id = nir_channel(&b0, gid, 1);
      if (reads_base_inst) {
         sv.base_instance = nir_load_ubo(&b0, 1, 32,
            nir_imm_int(&b0, key->base.draw_params_ubo), nir_imm_int(&b0, 4),
            .align_mul = 4, .range_base = 4, .range = 4);
      }
      nir_shader_intrinsics_pass(nir, apple9_lower_sw_vs_prolog_sysvals,
                                 nir_metadata_none, &sv);
   }

   struct apple9_buffer_map reserved = {0};
   apple9_seed_sw_vs_prolog_resources(
      &reserved, key->base.buffer_descriptor_ubo, key->base.draw_params_ubo,
      key->vertex_params_ubo, key->descriptor_tables, true);

   bool ok = apple9_compile_workgroup_body_ex(
      nir, out, profile, NULL, &reserved, false /* allow_preamble */, reason);
   ralloc_free(nir);
   return ok;
}

bool
agx_compile_apple9_fragment(nir_shader *nir, struct agx_shader_part *out,
                            const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, 1, reason);
}

bool
agx_compile_apple9_fragment_blend(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, struct agx_shader_part *out,
   const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, blend, 1, reason);
}

bool
agx_compile_apple9_fragment_mrt(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, unsigned nr_targets,
   struct agx_shader_part *out, const char **reason)
{
   return nr_targets <= 8 &&
          nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, blend, nr_targets, reason);
}

bool
agx_compile_apple9_fragment_inputs(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, NULL, 1, reason);
}

bool
agx_compile_apple9_vertex(nir_shader *nir, struct agx_shader_part *out,
                          const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, 1, reason);
}

bool
agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, layout, NULL, NULL, 1, reason);
}

bool
agx_apple9_patch_vs_varyings(
   const uint8_t *src_bin, size_t size, const struct agx_shader_info *info,
   uint64_t flat_mask, uint64_t linear_mask, uint8_t *dst_bin,
   struct agx_apple9_varying_layout *new_layout)
{
   if (!src_bin || !dst_bin || !info || !new_layout ||
       !info->apple9_vary_relocs_valid || size < info->main_size)
      return false;

   struct agx_apple9_varying_layout target = info->apple9_varyings;
   for (unsigned s = 0; s < ARRAY_SIZE(target.mask); ++s) {
      if (target.mask[s])
         target.group[s] = agx_apple9_varying_group(flat_mask, linear_mask, s);
   }
   if (target.cull_distance_mask) {
      target.group[VARYING_SLOT_CULL_PRIMITIVE] = 1;
   } else {
      if (target.mask[VARYING_SLOT_CULL_PRIMITIVE])
         target.group[VARYING_SLOT_CULL_PRIMITIVE] = 2;
      if (target.mask[VARYING_SLOT_CULL_PRIMITIVE + 1])
         target.group[VARYING_SLOT_CULL_PRIMITIVE + 1] = 2;
   }

   uint8_t locs[AGX_APPLE9_MAX_VARYING_COMPONENTS];
   uint8_t comps[AGX_APPLE9_MAX_VARYING_COMPONENTS];
   unsigned count =
      agx_apple9_varying_slots(&info->apple9_varyings, locs, comps);
   if (count != info->apple9_varyings.count ||
       count > ARRAY_SIZE(locs) ||
       info->apple9_vary_reloc_count > ARRAY_SIZE(info->apple9_vary_relocs))
      return false;

   uint8_t remap[AGX_APPLE9_MAX_VARYING_COMPONENTS];
   for (unsigned i = 0; i < count; ++i) {
      int dst_idx = agx_apple9_varying_index(&target, locs[i], comps[i]);
      if (dst_idx < 0 || dst_idx >= (int)count)
         return false;
      remap[i] = (uint8_t)dst_idx;
   }

   if (dst_bin != src_bin)
      memcpy(dst_bin, src_bin, size);

   for (unsigned r = 0; r < info->apple9_vary_reloc_count; ++r) {
      for (unsigned j = 0; j < r; ++j) {
         if (info->apple9_vary_relocs[j] == info->apple9_vary_relocs[r])
            return false;
      }
      size_t off = (size_t)info->apple9_vary_relocs[r] * 2;
      if (off + 8 > info->main_size)
         return false;
      if (src_bin[off] != 0x57 || (src_bin[off + 1] & 0x0f) != 0x06 ||
          (src_bin[off + 2] & 0xfc) != 0x54)
         return false;
      unsigned old_slot =
         ((src_bin[off + 4] >> 5) & 7u) | ((src_bin[off + 5] & 0x3fu) << 3);
      if (old_slot < 4 || old_slot >= 4 + count)
         return false;
      unsigned new_slot = 4 + remap[old_slot - 4];
      dst_bin[off + 4] = (uint8_t)((new_slot & 7u) << 5);
      dst_bin[off + 5] = (uint8_t)(0x40u | (new_slot >> 3));
   }

   *new_layout = target;
   return true;
}
