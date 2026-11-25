/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gravity/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "hk_cmd_buffer.h"
#include "hk_shader.h"
#include "hk_shader_priv.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%d: %s\n", __LINE__, #x);                            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static nir_shader *
source(bool quad, bool inputs, bool partial)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_EVAL, &agx_nir_options, "TES mesh test");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode =
      quad ? TESS_PRIMITIVE_QUADS : TESS_PRIMITIVE_TRIANGLES;
   b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   nir_def *coord = nir_load_tess_coord(&b);
   nir_def *pos =
      nir_vec4(&b, nir_channel(&b, coord, 0), nir_channel(&b, coord, 1),
               nir_channel(&b, coord, 2), nir_imm_float(&b, 1));
   if (inputs)
      pos = nir_fadd(
         &b, pos,
         nir_load_per_vertex_input(
            &b, 4, 32, nir_imm_int(&b, 2), nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1}));
   for (unsigned c = 0; c < 4; ++c) {
      if (partial && c == 2)
         continue;
      nir_store_output(
         &b, nir_channel(&b, pos, c), nir_imm_int(&b, 0), .component = c,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
lower_immutable(nir_shader *src, bool accepted)
{
   struct blob before, after;
   blob_init(&before);
   blob_init(&after);
   nir_serialize(&before, src, false);
   const char *reason = NULL;
   nir_shader *mesh = hk_nir_tes_mesh(src, &reason);
   if (!mesh && accepted)
      fprintf(stderr, "lowering: %s\n", reason);
   CHECK((mesh != NULL) == accepted);
   nir_serialize(&after, src, false);
   CHECK(before.size == after.size &&
         !memcmp(before.data, after.data, before.size));
   blob_finish(&before);
   blob_finish(&after);
   return mesh;
}

/* Sparse test memory: resolve actual lowered load addresses, including the
 * producer's winding swap, instead of checking a second emission algorithm. */
struct constants {
   uint32_t count, group, lane, patch_size, y, z;
   bool ccw, quad, inputs, partial;
   unsigned phase, loads, index_reads, coord_reads, tcs_reads;
};

enum {
   PARAMS = 0x100000,
   DRAW = 0x200000,
   ALLOCS = 0x300000,
   INDICES = 0x400000,
   COORDS = 0x500000,
   TCS = 0x600000
};

static unsigned
corner(const struct constants *c)
{
   return c->ccw && c->lane ? 3 - c->lane : c->lane;
}

static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct constants *c = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   if (c->phase == 0) {
      switch (intr->intrinsic) {
      case nir_intrinsic_load_local_invocation_index:
         value = nir_imm_int(b, c->lane);
         break;
      case nir_intrinsic_load_workgroup_id:
         value = nir_imm_ivec3(b, c->group, c->y, c->z);
         break;
      case nir_intrinsic_load_tess_param_buffer_poly:
         value = nir_imm_int64(b, PARAMS);
         break;
      default:
         return false;
      }
   } else {
      if ((intr->intrinsic != nir_intrinsic_load_global_constant &&
           intr->intrinsic != nir_intrinsic_load_global) ||
          !nir_src_is_const(intr->src[0]))
         return false;
      uint64_t addr = nir_src_as_uint(intr->src[0]);
      uint64_t word = 0;
      if (addr >= PARAMS && addr < PARAMS + sizeof(struct poly_tess_params)) {
         switch (addr - PARAMS) {
         case offsetof(struct poly_tess_params, out_draws):
            word = DRAW;
            break;
         case offsetof(struct poly_tess_params, output_patch_size):
            word = c->patch_size;
            break;
         case offsetof(struct poly_tess_params, ccw):
            word = c->ccw;
            break;
         case offsetof(struct poly_tess_params, index_buffer):
            word = INDICES;
            break;
         case offsetof(struct poly_tess_params, patch_coord_buffer):
            word = COORDS;
            break;
         case offsetof(struct poly_tess_params, coord_allocs):
            word = ALLOCS;
            break;
         case offsetof(struct poly_tess_params, tcs_patch_constants):
            word = 0;
            break;
         case offsetof(struct poly_tess_params, tcs_per_vertex_outputs):
            word = VARYING_BIT_POS;
            break;
         case offsetof(struct poly_tess_params, tcs_buffer):
            word = TCS;
            break;
         default:
            CHECK(!"unexpected parameter load");
         }
      } else if (addr == DRAW) {
         word = c->count;
      } else if (c->phase < 2) {
         return false;
      } else if (intr->intrinsic == nir_intrinsic_load_global &&
                 intr->def.num_components == 1 && addr >= INDICES) {
         CHECK(addr == INDICES + ((uint64_t)c->group * 3 + corner(c)) * 4);
         word = 5 * POLY_TES_PATCH_ID_STRIDE + corner(c);
         c->index_reads++;
      } else if (addr == ALLOCS + 5 * 4) {
         word = 8;
      } else if (intr->intrinsic == nir_intrinsic_load_global &&
                 intr->def.num_components == 2) {
         CHECK(addr == COORDS + (8 + corner(c)) * 8);
         value = nir_imm_ivec2(b, corner(c) * 16384, 16384);
         c->coord_reads++;
      } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                 intr->def.num_components == 4) {
         CHECK(c->inputs);
         CHECK(addr == TCS + (5 * (6 + c->patch_size * 4) + 6 + 2 * 4) * 4);
         value = nir_imm_vec4(b, 10, 20, 30, 40);
         c->tcs_reads++;
      } else {
         fprintf(stderr, "unexpected memory load 0x%llx\n",
                 (unsigned long long)addr);
         CHECK(false);
      }
      if (!value)
         value = nir_imm_intN_t(b, word, intr->def.bit_size);
      c->loads++;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
optimize(nir_shader *s)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_if, 0);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
}

static void
levels(void)
{
   /* The Vulkan frontend represents tess levels as ordinary load_input. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_EVAL, &agx_nir_options, "TES levels test");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
   nir_def *outer = nir_load_input(
      &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER,
                       .num_slots = 1});
   nir_def *inner = nir_load_input(
      &b, 2, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER,
                       .num_slots = 1});
   nir_def *pos =
      nir_vec4(&b, nir_channel(&b, outer, 0), nir_channel(&b, inner, 0),
               nir_imm_float(&b, 0), nir_imm_float(&b, 1));
   nir_store_output(
      &b, pos, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_shader *mesh = lower_immutable(b.shader, true);
   /* A short patch blocks tess-level payload loads as well as coordinates. */
   struct constants c = {.count = 3, .patch_size = 0};
   for (c.phase = 0; c.phase < 3; ++c.phase) {
      bool progress;
      do {
         optimize(mesh);
         progress = false;
         NIR_PASS(progress, mesh, nir_shader_intrinsics_pass, specialize,
                  nir_metadata_none, &c);
      } while (progress);
   }
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic ==
                  nir_intrinsic_set_vertex_and_primitive_count);
            CHECK(nir_src_as_uint(intr->src[0]) == 0 &&
                  nir_src_as_uint(intr->src[1]) == 0);
         }
      }
   }
   ralloc_free(mesh);
   /* A vec2 spanning components 1..2 exceeds the two inner levels. */
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_input &&
                   nir_intrinsic_io_semantics(intr).location ==
                      VARYING_SLOT_TESS_LEVEL_INNER)
                  nir_intrinsic_set_component(intr, 1);
            }
         }
      }
   }
   CHECK(!lower_immutable(b.shader, false));
   ralloc_free(b.shader);
}

static unsigned bounds_cases;

static void
bounds(struct constants c)
{
   nir_shader *src = source(c.quad, c.inputs, c.partial);
   nir_shader *mesh = lower_immutable(src, true);
   CHECK(mesh->info.mesh.max_vertices_out == 3);
   CHECK(mesh->info.mesh.max_primitives_out == 1);
   CHECK(mesh->info.workgroup_size[0] == 16);
   bool valid = !c.y && !c.z && c.lane < 3 && c.group < c.count / 3 &&
                c.patch_size <= 32 && c.patch_size >= (c.inputs ? 3 : 1);
   for (c.phase = 0; c.phase < 3; ++c.phase) {
      bool progress;
      do {
         optimize(mesh);
         progress = false;
         NIR_PASS(progress, mesh, nir_shader_intrinsics_pass, specialize,
                  nir_metadata_none, &c);
      } while (progress);
   }
   optimize(mesh);
   nir_validate_shader(mesh, "specialized TES mesh bounds test");
   unsigned vertices = 0, primitives = 0, counts = 0, mask = 0;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_per_vertex_output: {
               CHECK(valid && nir_src_as_uint(intr->src[1]) == c.lane);
               CHECK(nir_intrinsic_write_mask(intr) == 1);
               unsigned component = nir_intrinsic_component(intr);
               float coord[4] = {corner(&c) * .25f, .25f,
                                 c.quad ? 0 : .75f - corner(&c) * .25f, 1};
               float expected =
                  coord[component] + (c.inputs ? (component + 1) * 10 : 0);
               CHECK(nir_src_as_float(intr->src[0]) == expected);
               mask |= BITFIELD_BIT(component);
               vertices++;
               break;
            }
            case nir_intrinsic_store_per_primitive_output:
               CHECK(valid && c.lane == 0 &&
                     nir_src_as_uint(intr->src[1]) == 0);
               for (unsigned k = 0; k < 3; ++k)
                  CHECK(nir_src_comp_as_uint(intr->src[0], k) == k);
               primitives++;
               break;
            case nir_intrinsic_set_vertex_and_primitive_count:
               CHECK(!c.lane);
               CHECK(nir_src_as_uint(intr->src[0]) == (valid ? 3 : 0));
               CHECK(nir_src_as_uint(intr->src[1]) == (valid ? 1 : 0));
               counts++;
               break;
            default:
               CHECK(!"unresolved intrinsic or unexpected memory access");
            }
         }
      }
   }
   CHECK(vertices == (valid ? (c.partial ? 3 : 4) : 0));
   CHECK(mask == (valid ? (c.partial ? 11 : 15) : 0));
   CHECK(primitives == (valid && !c.lane));
   CHECK(counts == !c.lane);
   CHECK(valid ? c.index_reads > 0 && c.coord_reads > 0
               : c.index_reads == 0 && c.coord_reads == 0 && c.tcs_reads == 0);
   CHECK(c.tcs_reads == (valid && c.inputs));
   if (c.y || c.z || c.lane >= 3)
      CHECK(c.loads == 0);
   ralloc_free(mesh);
   ralloc_free(src);
   bounds_cases++;
}

static void
rejections(void)
{
   for (unsigned k = 0; k < 14; ++k) {
      nir_shader *src = source(false, true, false);
      switch (k) {
      case 0:
         src->info.stage = MESA_SHADER_VERTEX;
         break;
      case 1:
         src->info.tess._primitive_mode = TESS_PRIMITIVE_ISOLINES;
         break;
      case 2:
         src->info.tess.point_mode = true;
         break;
      case 3:
         src->info.writes_memory = true;
         break;
      case 4:
         src->info.outputs_written |= VARYING_BIT_LAYER;
         break;
      case 5:
         src->info.inputs_read |= VARYING_BIT_PSIZ;
         break;
      case 6:
         src->info.outputs_written_16bit = 1;
         break;
      case 7:
         src->info.clip_distance_array_size = 1;
         break;
      case 8:
         src->info.cull_distance_array_size = 1;
         break;
      case 9:
         src->info.io_lowered = false;
         break;
      case 11:
      case 12:
      case 13: {
         nir_function_impl *impl = nir_shader_get_entrypoint(src);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         if (k == 12) {
            nir_load_tess_level_inner(&b);
         } else if (k == 13) {
            nir_load_tess_level_outer(&b);
         } else {
            nir_def *index = nir_imm_int(&b, 32);
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_intrinsic &&
                      nir_instr_as_intrinsic(instr)->intrinsic ==
                         nir_intrinsic_load_per_vertex_input)
                     nir_src_rewrite(&nir_instr_as_intrinsic(instr)->src[0],
                                     index);
               }
            }
         }
         break;
      }
      case 10: {
         nir_function_impl *impl = nir_shader_get_entrypoint(src);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_def *dynamic = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                         nir_imm_int(&b, 0), .align_mul = 4);
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_load_per_vertex_input)
                  nir_src_rewrite(&nir_instr_as_intrinsic(instr)->src[0],
                                  dynamic);
            }
         }
         break;
      }
      }
      CHECK(!lower_immutable(src, false));
      ralloc_free(src);
   }
}

static void
compile(bool quad, bool inputs, bool partial)
{
   nir_shader *src = source(quad, inputs, partial);
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   bool compiled =
      hk_apple9_compile_tes_mesh_program(src, &key, &part, &info, &reason);
   if (!compiled)
      fprintf(stderr, "compile quad=%u inputs=%u partial=%u: %s\n", quad,
              inputs, partial, reason);
   CHECK(compiled && part.binary && part.info.binary_size);
   CHECK(info.layout.max_vertices == 3 && info.layout.max_primitives == 1);
   CHECK(
      hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   free(part.binary);
   ralloc_free(src);
}

/* Specialize the actual object NIR, so the test checks its address, load width
 * and launch dimensions rather than repeating its count-to-grid expression. */
struct object_case {
   uint32_t count;
   unsigned loads;
};

static bool
specialize_object(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct object_case *c = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value;
   if (intr->intrinsic == nir_intrinsic_load_native_pointer_apple9_agx) {
      CHECK(nir_intrinsic_base(intr) == AGX_APPLE9_MESH_DRAW_UNIFORM);
      value = nir_imm_ivec2(b, DRAW, 0);
   } else if (intr->intrinsic == nir_intrinsic_load_global &&
              nir_src_is_const(intr->src[0])) {
      CHECK(nir_src_as_uint(intr->src[0]) == DRAW);
      CHECK(intr->def.bit_size == 32 && intr->def.num_components == 1);
      CHECK(nir_intrinsic_align_mul(intr) == 4);
      ++c->loads;
      value = nir_imm_int(b, c->count);
   } else {
      return false;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
object_grid(void)
{
   const uint32_t counts[] = {0, 1, 2, 3, 4, 5, 6, 24576, 49152, UINT32_MAX};
   const uint32_t expected[] = {1, 1, 1, 1, 1, 1, 2, 8192, 16384, 1431655765};
   for (unsigned c = 0; c < ARRAY_SIZE(counts); ++c) {
      nir_shader *s = hk_nir_tes_mesh_object();
      CHECK(s && s->info.stage == MESA_SHADER_TASK);
      struct object_case state = {.count = counts[c]};
      bool progress;
      do {
         optimize(s);
         progress = false;
         NIR_PASS(progress, s, nir_shader_intrinsics_pass, specialize_object,
                  nir_metadata_none, &state);
      } while (progress);
      optimize(s);
      unsigned launches = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic == nir_intrinsic_launch_mesh_workgroups);
            CHECK(nir_src_comp_as_uint(intr->src[0], 0) == expected[c]);
            CHECK(nir_src_comp_as_uint(intr->src[0], 1) == 1);
            CHECK(nir_src_comp_as_uint(intr->src[0], 2) == 1);
            ++launches;
         }
      }
      CHECK(state.loads == 1 && launches == 1);
      ralloc_free(s);
   }
   nir_shader *s = hk_nir_tes_mesh_object();
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   CHECK(
      agx_compile_apple9_native_object(s, true, &key, &part, &info, &reason));
   CHECK(part.binary && part.info.binary_size);
   CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                       HK_APPLE9_NATIVE_OBJECT));
   free(part.binary);
   ralloc_free(s);
}

/* =========================================================================
 * Zero-PREGFX Fused OBJECT(VS+TCS) -> MESH(TES) Comprehensive Test Suite
 * ========================================================================= */

static nir_shader *
make_fused_vs(unsigned extra_vars, bool with_vbo)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "fused_tess_vs");
   b.shader->info.io_lowered = true;
   nir_def *vid = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *iid = nir_u2f32(&b, nir_load_instance_id(&b));
   nir_def *pos = nir_vec4(&b, vid, iid, nir_imm_float(&b, 0.0f),
                           nir_imm_float(&b, 1.0f));
   if (with_vbo) {
      nir_def *attr = nir_load_input(
         &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
      pos = nir_fadd(&b, pos, attr);
   }
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned v = 0; v < extra_vars; ++v) {
      nir_store_output(
         &b, nir_vec4(&b, vid, iid, nir_imm_float(&b, (float)v), nir_imm_float(&b, 1.0f)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + v, .num_slots = 1});
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_fused_tcs(unsigned out_v, unsigned extra_vars, float outer_val,
               float inner_val, bool with_barrier, bool with_patch_const,
               bool split_vkd3d_modes, enum gl_tess_spacing spacing, bool ccw,
               bool dynamic_levels)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_CTRL, &agx_nir_options, "fused_tess_tcs");
   b.shader->info.io_lowered = true;
   b.shader->info.tess.tcs_vertices_out = out_v;
   if (split_vkd3d_modes) {
      /* vkd3d-proton puts Spacing/Winding on TCS and Domain on TES. */
      b.shader->info.tess.spacing = spacing;
      b.shader->info.tess.ccw = ccw;
   }
   nir_def *inv = nir_load_invocation_id(&b);
   nir_def *pos = nir_load_per_vertex_input(
      &b, 4, 32, inv, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_store_per_vertex_output(
      &b, pos, inv, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned v = 0; v < extra_vars; ++v) {
      nir_def *val = nir_load_per_vertex_input(
         &b, 4, 32, inv, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + v, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, val, inv, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + v, .num_slots = 1});
   }
   if (with_barrier) {
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
   }
   nir_push_if(&b, nir_ieq_imm(&b, inv, 0));
   nir_def *ov = dynamic_levels
                    ? nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                   nir_imm_int(&b, 0), .align_mul = 4, .range = 16)
                    : nir_imm_float(&b, outer_val);
   nir_def *iv = dynamic_levels
                    ? nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                   nir_imm_int(&b, 4), .align_mul = 4, .range = 16)
                    : nir_imm_float(&b, inner_val);
   nir_store_output(
      &b, nir_vec4(&b, ov, ov, ov, ov), nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
   nir_store_output(
      &b, nir_vec2(&b, iv, iv), nir_imm_int(&b, 0), .write_mask = 3,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
   if (with_patch_const) {
      nir_store_output(
         &b, nir_imm_vec4(&b, 1.0f, 2.0f, 3.0f, 4.0f), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PATCH0, .num_slots = 1});
   }
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_fused_tes(enum tess_primitive_mode domain, enum gl_tess_spacing spacing,
               bool ccw, bool split_vkd3d_modes, unsigned out_v,
               unsigned extra_out_vars, unsigned clip_size, unsigned cull_size,
               bool read_patch_const, bool read_raw_levels)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_EVAL, &agx_nir_options, "fused_tess_tes");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode = domain;
   if (!split_vkd3d_modes) {
      b.shader->info.tess.spacing = spacing;
      b.shader->info.tess.ccw = ccw;
   }
   nir_def *coord = nir_load_tess_coord(&b);
   nir_def *p0 = nir_load_per_vertex_input(
      &b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_def *pos = nir_fadd(&b, p0,
                           nir_vec4(&b, nir_channel(&b, coord, 0),
                                    nir_channel(&b, coord, 1),
                                    nir_channel(&b, coord, 2),
                                    nir_imm_float(&b, 0.0f)));
   if (out_v >= 2) {
      nir_def *p1 = nir_load_per_vertex_input(
         &b, 4, 32, nir_imm_int(&b, out_v - 1), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      pos = nir_fadd(&b, pos, nir_fmul_imm(&b, p1, 0.25f));
   }
   if (read_patch_const) {
      nir_def *pc = nir_load_input(
         &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PATCH0, .num_slots = 1});
      pos = nir_fadd(&b, pos, pc);
   }
   if (read_raw_levels) {
      nir_def *lo = nir_load_tess_level_outer(&b);
      nir_def *li = nir_load_tess_level_inner(&b);
      pos = nir_fadd(&b, pos,
                     nir_vec4(&b, nir_channel(&b, lo, 0), nir_channel(&b, li, 0),
                              nir_imm_float(&b, 0.0f), nir_imm_float(&b, 0.0f)));
   }
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned v = 0; v < extra_out_vars; ++v) {
      nir_store_output(
         &b, pos, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + v, .num_slots = 1});
   }
   for (unsigned d = 0; d < clip_size; ++d) {
      nir_store_output(
         &b, nir_channel(&b, coord, 0), nir_imm_int(&b, 0), .component = d % 4,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0 + d / 4,
                          .num_slots = 1});
   }
   for (unsigned d = 0; d < cull_size; ++d) {
      nir_store_output(
         &b, nir_channel(&b, coord, 1), nir_imm_int(&b, 0), .component = d % 4,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CULL_DIST0 + d / 4,
                          .num_slots = 1});
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   b.shader->info.clip_distance_array_size = clip_size;
   b.shader->info.cull_distance_array_size = cull_size;
   return b.shader;
}

static void
compile_fused_tess_pair(const nir_shader *vs, const nir_shader *tcs,
                        const nir_shader *tes,
                        const struct hk_tess_fused_options *opts,
                        struct hk_tess_fused_info *out_info)
{
   struct blob b_vs, b_tcs, b_tes, a_vs, a_tcs, a_tes;
   blob_init(&b_vs);
   blob_init(&b_tcs);
   blob_init(&b_tes);
   blob_init(&a_vs);
   blob_init(&a_tcs);
   blob_init(&a_tes);
   nir_serialize(&b_vs, vs, false);
   nir_serialize(&b_tcs, tcs, false);
   nir_serialize(&b_tes, tes, false);

   nir_shader *obj = NULL, *mesh = NULL;
   struct hk_tess_fused_info info = {0};
   const char *reason = NULL;
   bool ok = hk_nir_tess_fused_lower(vs, tcs, tes, opts, &obj, &mesh, &info,
                                     &reason);
   if (!ok)
      fprintf(stderr, "fused tess lower failed: %s\n", reason ?: "unknown");
   CHECK(ok && obj && mesh);

   /* Immutability: original VS, TCS, TES NIR are untouched. */
   nir_serialize(&a_vs, vs, false);
   nir_serialize(&a_tcs, tcs, false);
   nir_serialize(&a_tes, tes, false);
   CHECK(b_vs.size == a_vs.size && !memcmp(b_vs.data, a_vs.data, b_vs.size));
   CHECK(b_tcs.size == a_tcs.size && !memcmp(b_tcs.data, a_tcs.data, b_tcs.size));
   CHECK(b_tes.size == a_tes.size && !memcmp(b_tes.data, a_tes.data, b_tes.size));
   blob_finish(&b_vs);
   blob_finish(&b_tcs);
   blob_finish(&b_tes);
   blob_finish(&a_vs);
   blob_finish(&a_tcs);
   blob_finish(&a_tes);

   /* Compile OBJECT(VS+TCS) through Apple9 native object compiler with 0xe0/0xe1. */
   hk_apple9_lower_prerast_sysvals(obj, MESA_SHADER_COMPUTE);
   struct agx_apple9_mesh_key obj_key = AGX_APPLE9_MESH_KEY;
   obj_key.physical_threads = AGX_APPLE9_NATIVE_SERIAL_THREADS;
   struct agx_apple9_native_object_options obj_opts = {
      .payload_stride_B = info.payload_B,
      .write_child_grid = false,
      .hw_payload = true,
   };
   struct agx_shader_part obj_part = {0};
   struct agx_apple9_mesh_body_info obj_info = {0};
   ok = agx_compile_apple9_native_object_ex(obj, true, &obj_key, &obj_opts,
                                            &obj_part, &obj_info, &reason);
   if (!ok)
      fprintf(stderr, "fused tess OBJECT compile failed: %s\n", reason ?: "unknown");
   CHECK(ok && obj_part.binary && obj_part.info.binary_size);
   CHECK(obj_info.key.hw_payload);
   CHECK(hk_apple9_native_shader_valid(&obj_part.info, &obj_info,
                                       HK_APPLE9_NATIVE_OBJECT));
   free(obj_part.binary);
   ralloc_free(obj);

   /* Compile MESH(TES) through Apple9 native mesh compiler with 0xe0/0xe1. */
   hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
   struct hk_apple9_graphics_key mkey = {
      .native_stage_generated = true,
      .native_tess_fused = true,
      .native_object = true,
      .native_object_hw_payload = true,
      .native_object_payload_B = info.payload_B,
      .fs_reads_clip_distance = opts->fs_reads_clip_distance,
   };
   struct agx_shader_part mesh_part = {0};
   struct agx_apple9_mesh_body_info mesh_info = {0};
   ok = hk_apple9_compile_native_mesh_program(mesh, &mkey, &mesh_part,
                                              &mesh_info, &reason);
   if (!ok)
      fprintf(stderr, "fused tess MESH compile failed: %s\n", reason ?: "unknown");
   CHECK(ok && mesh_part.binary && mesh_part.info.binary_size);
   CHECK(mesh_info.key.hw_payload);
   CHECK(hk_apple9_native_shader_valid(&mesh_part.info, &mesh_info,
                                       HK_APPLE9_NATIVE_MESH));
   {
      uint32_t safe_u0 = 0, safe_stride = 0;
      CHECK(agx_apple9_mesh_safe_u0_b(&mesh_info.layout, &safe_u0, &safe_stride));
      uint64_t ring_max = 0;
      CHECK(agx_apple9_mesh_ring_max_groups(&mesh_info.layout, &ring_max));
      CHECK(ring_max == UINT64_MAX);
   }
   free(mesh_part.binary);
   ralloc_free(mesh);

   if (out_info)
      *out_info = info;
}

static void
fused_tess_suite(void)
{
   struct hk_tess_fused_options base_opts = {
      .max_payload_bytes = 1024,
   };
   base_opts.vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   base_opts.vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   base_opts.vertex.dynamic_stride = true;
   base_opts.vertex.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
   base_opts.vertex.buffer[0] = 0;
   base_opts.vertex.offset[0] = 0;
   base_opts.vertex.stride[0] = 16;

   /* 1. Full matrix: Quads & Triangles x Equal/FracOdd/FracEven x CW/CCW x
    * 1->1, 2->2, 3->3, 4->4 CPs, including vkd3d-proton split execution modes
    * and TCS OpControlBarrier. */
   const enum tess_primitive_mode domains[] = {
      TESS_PRIMITIVE_TRIANGLES,
      TESS_PRIMITIVE_QUADS,
   };
   const enum gl_tess_spacing spacings[] = {
      TESS_SPACING_EQUAL,
      TESS_SPACING_FRACTIONAL_ODD,
      TESS_SPACING_FRACTIONAL_EVEN,
   };
   const unsigned cps[] = {1, 2, 3, 4};
   unsigned compiled = 0;
   for (unsigned d = 0; d < ARRAY_SIZE(domains); ++d) {
      for (unsigned sp = 0; sp < ARRAY_SIZE(spacings); ++sp) {
         for (unsigned ccw = 0; ccw < 2; ++ccw) {
            unsigned out_v = cps[(d * 3 + sp + ccw) % ARRAY_SIZE(cps)];
            bool split = (d == 1 && sp == 1); /* vkd3d-proton split mode */
            bool barrier = out_v >= 2;
            nir_shader *vs = make_fused_vs(0, true);
            nir_shader *tcs = make_fused_tcs(out_v, 0, 4.0f, 4.0f, barrier,
                                             false, split, spacings[sp], ccw,
                                             false);
            nir_shader *tes = make_fused_tes(domains[d], spacings[sp], ccw,
                                             split, out_v, 0, 0, 0, false, false);
            struct hk_tess_fused_info info = {0};
            compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
            CHECK(info.domain == domains[d]);
            CHECK(info.spacing == spacings[sp]);
            CHECK(info.ccw == !!ccw);
            CHECK(info.tcs_vertices_out == out_v);
            CHECK(info.constant_tess_levels);
            /* Position-only: Equal/FracEven at 4.0 -> n=4 (Tri V19/P24, Quad V25/P32: 1 WG);
             * FracOdd at 4.0 -> n=5 (Tri V27/P37: 1 WG; Quad V36/P50: multi-WG). */
            CHECK(info.mesh_max_vertices == 32 && info.mesh_max_primitives == 42);
            if (spacings[sp] == TESS_SPACING_FRACTIONAL_ODD) {
               if (domains[d] == TESS_PRIMITIVE_TRIANGLES) {
                  CHECK(info.max_child_wgs_per_patch == 1);
                  CHECK(info.max_prims_per_patch == 37);
               } else {
                  CHECK(info.max_child_wgs_per_patch == 5);
                  CHECK(info.max_prims_per_patch == 50);
               }
            } else {
               CHECK(info.max_child_wgs_per_patch == 1);
               CHECK(info.max_prims_per_patch ==
                     (domains[d] == TESS_PRIMITIVE_QUADS ? 32 : 24));
            }
            ralloc_free(vs);
            ralloc_free(tcs);
            ralloc_free(tes);
            compiled++;
         }
      }
   }

   /* 2. Adaptive meshlet tiers (V32/P42, V24/P20, V16/P18, V8/P6) + FS-linked
    * varying DCE + ClipDistance[2] + CullDistance[1] (Elden Ring Triangle
    * shape) + raw tess levels + patch constants. */
   {
      /* 8 comps (POS + 1 vec4) -> golden packed V24/P20 (Tier 1 because raw
       * levels + patch constants are read by TES). */
      nir_shader *vs = make_fused_vs(1, false);
      nir_shader *tcs = make_fused_tcs(3, 1, 3.0f, 3.0f, true, true, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       1, 0, 0, true, true);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 2);
      CHECK(info.max_child_wgs_per_patch == 1);
      CHECK(info.max_prims_per_patch == 13);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }
   {
      /* 12 comps (POS + 2 vec4) and 16 comps (POS + 3 vec4) both select golden
       * packed V24/P20 (unbounded via safe-stride U0_B); with FS mask reading
       * 0 vars -> V32/P42! */
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(1, 0, 1.0f, 1.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 2, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(info.multipatch_tier == 2 && info.patches_per_group == 16);
      CHECK(info.max_child_wgs_per_patch == 1 && info.max_prims_per_patch == 2);

      nir_shader *tes16 = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                         false, false, 1, 3, 0, 0, false, false);
      compile_fused_tess_pair(vs, tcs, tes16, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(info.max_child_wgs_per_patch == 1 && info.max_prims_per_patch == 2);
      ralloc_free(tes16);

      struct hk_tess_fused_options depth_opts = base_opts;
      depth_opts.has_fs_inputs_mask = true; /* FS reads 0 generic vars */
      compile_fused_tess_pair(vs, tcs, tes, &depth_opts, &info);
      CHECK(info.mesh_max_vertices == 32 && info.mesh_max_primitives == 42);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled += 3;
   }
   {
      /* Elden Ring Triangle shape: 3->3 CPs, FracOdd, CW, Clip[2] + Cull[1] + 4 vec4
       * (4 + 16 + 2 = 22 comps -> planes = 24 -> golden V24/P20, Tier 2 K=16,
       * unbounded via safe-stride U0_B!). */
      nir_shader *vs = make_fused_vs(4, true);
      nir_shader *tcs = make_fused_tcs(3, 4, 4.0f, 4.0f, true, false, true,
                                       TESS_SPACING_FRACTIONAL_ODD, false, true);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_FRACTIONAL_ODD, false, true,
                                       3, 4, 2, 1, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(info.multipatch_tier == 2 && info.patches_per_group == 16);
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_child_wgs_per_patch > 0 && info.max_child_wgs_per_patch <= 4096);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }
   {
      /* Golden V=24/P=20 meshlets across planes == 24 (21..24 live vertex
       * components) and wider payloads (planes == 36..68 -> V16/P18,
       * planes == 72..128 -> V8/P6): safe-stride U0_B keeps all shapes
       * unbounded without dummy varying padding. */
      for (unsigned live_user = 17; live_user <= 20; ++live_user) {
         nir_shader *vs = make_fused_vs(5, false);
         nir_shader *tcs = make_fused_tcs(1, 5, 1.0f, 1.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                          false, false, 1, 5, 0, 0, false, false);
         struct hk_tess_fused_options mask_opts = base_opts;
         mask_opts.has_fs_inputs_mask = true;
         for (unsigned u = 0; u < live_user; ++u)
            mask_opts.fs_var_read_mask[u / 4] |= BITFIELD_BIT(u % 4);
         nir_shader *obj = NULL, *mesh = NULL;
         struct hk_tess_fused_info info = {0};
         const char *r = NULL;
         CHECK(hk_nir_tess_fused_lower(vs, tcs, tes, &mask_opts, &obj, &mesh,
                                       &info, &r));
         CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
         hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
         struct agx_apple9_native_mesh_options mopts = {
            .physical_threads = AGX_APPLE9_NATIVE_SERIAL_THREADS,
            .packed_index_publication = true,
            .publication_rounds = true,
            .attributes = true,
            .native_object_payload = true,
            .native_object_hw_payload = true,
            .native_object_payload_stride_B = info.payload_B,
         };
         struct agx_shader_part part = {0};
         struct agx_apple9_mesh_body_info body = {0};
         CHECK(agx_compile_apple9_native_mesh(mesh, true, &mopts, &part, &body,
                                              &r));
         CHECK(body.layout.vertex_components == 4u + live_user);
         CHECK(part.info.apple9_varyings.count == live_user);
         CHECK(hk_apple9_native_shader_valid(&part.info, &body,
                                             HK_APPLE9_NATIVE_MESH));
         uint32_t safe_u0 = 0, safe_stride = 0;
         CHECK(agx_apple9_mesh_safe_u0_b(&body.layout, &safe_u0, &safe_stride));
         CHECK(safe_u0 == 32u && safe_stride == 196u);
         uint64_t ring_max = 0;
         CHECK(agx_apple9_mesh_ring_max_groups(&body.layout, &ring_max));
         CHECK(ring_max == UINT64_MAX);
         free(part.binary);
         ralloc_free(obj);
         ralloc_free(mesh);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
         compiled++;
      }
      /* Wider varying payloads: 36 comps (planes=36) -> V16/P18, 72 comps -> V8/P6. */
      const struct {
         unsigned extra_vars, exp_v, exp_p;
      } wide_cases[] = {
         {8, 16, 18},  /* 4 + 32 = 36 comps -> planes 36 -> V16/P18 */
         {17, 8, 6},   /* 4 + 68 = 72 comps -> planes 72 -> V8/P6 */
      };
      for (unsigned wi = 0; wi < ARRAY_SIZE(wide_cases); ++wi) {
         nir_shader *vs = make_fused_vs(0, false);
         nir_shader *tcs = make_fused_tcs(1, 0, 1.0f, 1.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                          false, false, 1, wide_cases[wi].extra_vars,
                                          0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
         CHECK(info.mesh_max_vertices == wide_cases[wi].exp_v &&
               info.mesh_max_primitives == wide_cases[wi].exp_p);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
         compiled++;
      }
   }

   /* 3. Zero-child patch culling (outer == 0.0, outer == -2.0, outer == NaN). */
   const float cull_vals[] = {0.0f, -2.0f, NAN};
   for (unsigned i = 0; i < ARRAY_SIZE(cull_vals); ++i) {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, cull_vals[i], 4.0f, false, false,
                                       false, TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.constant_tess_levels);
      CHECK(info.max_child_wgs_per_patch == 0 && info.max_prims_per_patch == 0);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }

   /* 3b. Push-constant tess levels (exact draw-time TVB/ring count evaluation). */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "fused_tess_push_tcs");
      b.shader->info.io_lowered = true;
      b.shader->info.tess.tcs_vertices_out = 3;
      nir_def *inv = nir_load_invocation_id(&b);
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, inv, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, pos, inv, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&b, nir_ieq_imm(&b, inv, 0));
      unsigned root_off = offsetof(struct hk_root_descriptor_table, root_desc_addr);
      nir_def *root_ptr = nir_load_ubo(&b, 1, 64, nir_imm_int(&b, 0),
                                       nir_imm_int(&b, root_off),
                                       .align_mul = 8, .range_base = root_off,
                                       .range = 8);
      nir_def *paddr = nir_iadd(
         &b, root_ptr,
         nir_u2u64(&b, nir_imm_int(&b, hk_root_descriptor_offset(push) + 4)));
      nir_def *lvl = nir_load_global_constant(&b, 1, 32, paddr, .align_mul = 4);
      nir_store_output(
         &b, nir_vec4(&b, lvl, lvl, lvl, lvl), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER,
                          .num_slots = 1});
      nir_store_output(
         &b, nir_vec2(&b, lvl, lvl), nir_imm_int(&b, 0), .write_mask = 3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER,
                          .num_slots = 1});
      nir_pop_if(&b, NULL);
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      nir_shader *tcs = b.shader;
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, true, false, 3, 0, 0,
                                       0, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(!info.constant_tess_levels);
      CHECK(info.push_levels.valid);
      struct hk_shader_info sinfo = {
         .apple9_tess_fused = true,
         .apple9_tess_max_child_wgs = info.max_child_wgs_per_patch,
         .apple9_tess_max_patch_prims = info.max_prims_per_patch,
         .apple9_tess_push_levels = info.push_levels,
      };
      const struct {
         float lvl;
         uint32_t wgs;
         uint32_t prims;
      } cases[] = {
         {1.0f, 1, 1},
         {4.0f, 1, 24},
         {64.0f, info.max_child_wgs_per_patch, 6144},
         {0.0f, 0, 0},
         {NAN, 0, 0},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
         uint8_t push[HK_MAX_PUSH_SIZE] = {0};
         memcpy(push + 4, &cases[i].lvl, sizeof(float));
         uint32_t wgs = 999, prims = 999;
         hk_nir_tess_fused_eval_counts(&sinfo, push, &wgs, &prims);
         CHECK(wgs == cases[i].wgs && prims == cases[i].prims);
      }
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }

   /* 3c. Synthetic TCS RMW store narrowing (multi-component vec RMW, masked-off
    * dead-lane RMW loads, disjoint-channel multi-store scalar chains, direct
    * identity load->store passthroughs, and shared load_const across top-level
    * TCS barrier splits). */
   {
      nir_shader *vs = make_fused_vs(2, false);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "fused_tess_rmw_chain_tcs");
      b.shader->info.io_lowered = true;
      b.shader->info.tess.tcs_vertices_out = 3;
      b.shader->info.tess.spacing = TESS_SPACING_FRACTIONAL_ODD;
      nir_def *inv = nir_load_invocation_id(&b);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, inv, zero, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *v0 = nir_load_per_vertex_input(
         &b, 4, 32, inv, zero, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, pos, inv, zero, .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});

      /* (a) Multi-component RMW: 2 live lanes (.y, .w) + 2 passthrough lanes
       * (.x, .z) -> must strip passthrough lanes AND undef the hole lane so
       * load_per_vertex_output is dead. */
      nir_def *old0 = nir_load_per_vertex_output(
         &b, 4, 32, inv, zero, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *rmw_yw = nir_vec4(&b, nir_channel(&b, old0, 0),
                                 nir_channel(&b, v0, 1),
                                 nir_channel(&b, old0, 2),
                                 nir_channel(&b, v0, 3));
      nir_store_per_vertex_output(
         &b, rmw_yw, inv, zero, .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});

      /* (b) Pre-masked vec3 store (wrmask=yz) whose masked-off .x holds a
       * load_per_vertex_output, plus wrmask=xz with dead .y hole. */
      nir_def *old_x = nir_load_per_vertex_output(
         &b, 1, 32, inv, zero, .component = 0, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *yz_vec = nir_vec3(&b, old_x, nir_channel(&b, v0, 1),
                                 nir_channel(&b, v0, 2));
      nir_store_per_vertex_output(
         &b, yz_vec, inv, zero, .write_mask = 0x6, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});

      nir_def *old_y = nir_load_per_vertex_output(
         &b, 1, 32, inv, zero, .component = 1, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *xz_vec = nir_vec3(&b, nir_channel(&b, v0, 0), old_y,
                                 nir_channel(&b, v0, 2));
      nir_store_per_vertex_output(
         &b, xz_vec, inv, zero, .write_mask = 0x5, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});

      /* (c) Disjoint-channel multi-store scalar chain to the SAME loc + direct
       * identity load->store passthroughs (non-vec). */
      nir_def *old_c1 = nir_load_per_vertex_output(
         &b, 1, 32, inv, zero, .component = 1, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_def *old_c23 = nir_load_per_vertex_output(
         &b, 2, 32, inv, zero, .component = 2, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, nir_channel(&b, v0, 0), inv, zero, .component = 0, .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, old_c1, inv, zero, .component = 1, .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, old_c23, inv, zero, .component = 2, .write_mask = 3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});

      /* (d) Top-level barrier followed by load_per_vertex_input reusing `zero`
       * from before the barrier (regression for load_const hoisting before
       * nir_repair_ssa_impl). */
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
      nir_def *v1_post = nir_load_per_vertex_input(
         &b, 4, 32, inv, zero, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, nir_channel(&b, v1_post, 1), inv, zero, .component = 1,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_push_if(&b, nir_ieq_imm(&b, inv, 0));
      nir_store_output(
         &b, nir_imm_vec4(&b, 3.0f, 3.0f, 3.0f, 1.0f), zero, .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER,
                          .num_slots = 1});
      nir_store_output(
         &b, nir_imm_vec2(&b, 2.0f, 1.0f), zero, .write_mask = 3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER,
                          .num_slots = 1});
      nir_pop_if(&b, NULL);
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      nir_shader *tcs = b.shader;
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_FRACTIONAL_ODD, false, false,
                                       3, 2, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.constant_tess_levels);
      CHECK(info.max_child_wgs_per_patch > 0);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }

   /* 4. Clean rejection / fallback cases. */
   const char *reason = NULL;
   /* Isolines rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(2, 0, 4.0f, 4.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_ISOLINES,
                                       TESS_SPACING_EQUAL, false, false, 2,
                                       0, 0, 0, false, false);
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      CHECK(reason != NULL);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }
   /* PointMode rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, 4.0f, 4.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       0, 0, 0, false, false);
      tes->info.tess.point_mode = true;
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }
   /* writes_memory in TCS rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, 4.0f, 4.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       0, 0, 0, false, false);
      tcs->info.writes_memory = true;
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }
   /* Cross-invocation TCS output read (load_per_vertex_output) rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, 4.0f, 4.0f, true, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(tcs)));
      nir_def *other = nir_load_per_vertex_output(
         &b, 4, 32, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, other, nir_load_invocation_id(&b), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       0, 0, 0, false, false);
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }
   /* Payload exceeding mp_payload_bytes rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(4, 0, 1.0f, 1.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 4, 0, 0, 0, false, false);
      struct hk_tess_fused_options tiny_opts = base_opts;
      tiny_opts.max_payload_bytes = 64; /* 40 + 4*16 = 104 > 64 */
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &tiny_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }
   /* Nested TCS barrier inside if rejected */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, 4.0f, 4.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(tcs)));
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_invocation_id(&b), 0));
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
      nir_pop_if(&b, NULL);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       0, 0, 0, false, false);
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }

   /* Cross-invocation TCS output read inside a multi-component vec4 RMW store
    * must still be strictly rejected. */
   {
      nir_shader *vs = make_fused_vs(1, false);
      nir_shader *tcs = make_fused_tcs(3, 1, 4.0f, 4.0f, true, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(tcs)));
      nir_def *inv = nir_load_invocation_id(&b);
      nir_def *other = nir_load_per_vertex_output(
         &b, 4, 32, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *mix = nir_vec4(&b, nir_channel(&b, other, 0), nir_imm_float(&b, 1.0f),
                              nir_channel(&b, other, 2), nir_imm_float(&b, 2.0f));
      nir_store_per_vertex_output(
         &b, mix, inv, nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       1, 0, 0, false, false);
      CHECK(!hk_nir_tess_fused_lower(vs, tcs, tes, &base_opts, NULL, NULL, NULL,
                                     &reason));
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }

   printf("fused tessellation suite: %u OBJECT+MESH compiles + 7 fallback rejections passed\n",
          compiled);
}

static uint32_t
f32_bits(float v)
{
   uint32_t u;
   memcpy(&u, &v, sizeof(u));
   return u;
}

static void
fused_tess_vbo_pack_suite(void)
{
   /* 1. Exhaustive host-reference roundtrip verification for UNORM8, SNORM8,
    *    UNORM16, SNORM16 (verifies mathematical roundtrip including both SNORM
    *    encodings of -1; hardware instruction verification is separate). */
   for (unsigned u = 0; u < 256; ++u) {
      float f = (float)u / 255.0f;
      unsigned packed = (unsigned)lrintf(fminf(fmaxf(f, 0.0f), 1.0f) * 255.0f);
      CHECK(packed == u);
      CHECK(f32_bits((float)packed / 255.0f) == f32_bits(f));

      int8_t s = (int8_t)u;
      float sf = fmaxf((float)s / 127.0f, -1.0f);
      int spacked = (int)lrintf(fminf(fmaxf(sf, -1.0f), 1.0f) * 127.0f);
      float sunpacked = fmaxf((float)spacked / 127.0f, -1.0f);
      CHECK(f32_bits(sunpacked) == f32_bits(sf));
   }
   for (unsigned u = 0; u < 65536; ++u) {
      float f = (float)u / 65535.0f;
      unsigned packed = (unsigned)lrintf(fminf(fmaxf(f, 0.0f), 1.0f) * 65535.0f);
      CHECK(packed == u);
      CHECK(f32_bits((float)packed / 65535.0f) == f32_bits(f));

      int16_t s = (int16_t)u;
      float sf = fmaxf((float)s / 32767.0f, -1.0f);
      int spacked = (int)lrintf(fminf(fmaxf(sf, -1.0f), 1.0f) * 32767.0f);
      float sunpacked = fmaxf((float)spacked / 32767.0f, -1.0f);
      CHECK(f32_bits(sunpacked) == f32_bits(sf));
   }

   /* 2. Full vec4, partial .xyz mask, swizzled B8G8R8A8_UNORM/SNORM,
    *    cross-slot split components, TCS self-RMW chain, and R16G16_UNORM:
    *    verify plane reduction, flat u32 packing, and TCS payload compaction. */
   const enum pipe_format fmts[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_SNORM,
      PIPE_FORMAT_B8G8R8A8_SNORM,
   };
   for (unsigned fi = 0; fi < ARRAY_SIZE(fmts); ++fi) {
      /* Build VS: attr 0 (float4 pos), attr 1..3 (narrow norm vec4), attr 4 (R16G16_UNORM).
       * Also split attr 1.zw + attr 2.xy across VAR1/VAR2 like nir_opt_varyings_bulk. */
      nir_builder vb = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "vbo_pack_vs");
      vb.shader->info.io_lowered = true;
      nir_def *z = nir_imm_int(&vb, 0);
      nir_def *p = nir_load_input(
         &vb, 4, 32, z, .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
      nir_store_output(
         &vb, p, z, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *a1 = nir_load_input(
         &vb, 4, 32, z, .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC1, .num_slots = 1});
      nir_def *a2 = nir_load_input(
         &vb, 4, 32, z, .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC2, .num_slots = 1});
      nir_def *a3 = nir_load_input(
         &vb, 4, 32, z, .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC3, .num_slots = 1});
      nir_def *a4 = nir_load_input(
         &vb, 2, 32, z, .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC4, .num_slots = 1});
      nir_store_output(
         &vb, a1, z, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(
         &vb, a2, z, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_output(
         &vb, a3, z, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR2, .num_slots = 1});
      nir_store_output(
         &vb, a4, z, .write_mask = 0x3, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

      /* Build TCS (pcp=1) with an identity self-RMW chain on VAR0. */
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "vbo_pack_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *cz = nir_imm_int(&cb, 0);
      nir_def *cp = nir_load_per_vertex_input(
         &cb, 4, 32, inv, cz, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, cp, inv, cz, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *c0 = nir_load_per_vertex_input(
         &cb, 4, 32, inv, cz, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, nir_channel(&cb, c0, 0), inv, cz, .component = 0, .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *c0_old = nir_load_per_vertex_output(
         &cb, 1, 32, inv, cz, .component = 0, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *c0_rmw = nir_vec4(&cb, c0_old, nir_channel(&cb, c0, 1),
                                 nir_channel(&cb, c0, 2), nir_channel(&cb, c0, 3));
      nir_store_per_vertex_output(
         &cb, c0_rmw, inv, cz, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      for (unsigned s = 1; s <= 3; ++s) {
         unsigned nch = s == 3 ? 2 : 4;
         nir_def *v = nir_load_per_vertex_input(
            &cb, nch, 32, inv, cz, .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + s, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, v, inv, cz, .write_mask = BITFIELD_MASK(nch),
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + s, .num_slots = 1});
      }
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
      nir_store_output(
         &cb, nir_imm_vec4(&cb, 1.0f, 1.0f, 1.0f, 1.0f), cz, .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_imm_vec2(&cb, 1.0f, 1.0f), cz, .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      /* Build TES: VAR0 = (smooth_uv.xy, c0.xy) [mixed slot!], VAR1 = c0.zw+c1.xy,
       * VAR2 = c1.zw+c2.xy, VAR3 = (c2.z, c3.xy, smooth_w) [partial + R16G16!]. */
      nir_builder eb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_EVAL, &agx_nir_options, "vbo_pack_tes");
      eb.shader->info.io_lowered = true;
      eb.shader->info.tess._primitive_mode = TESS_PRIMITIVE_QUADS;
      eb.shader->info.tess.spacing = TESS_SPACING_EQUAL;
      nir_def *ez = nir_imm_int(&eb, 0);
      nir_def *tc = nir_load_tess_coord(&eb);
      nir_def *ep = nir_load_per_vertex_input(
         &eb, 4, 32, ez, ez, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_output(
         &eb, nir_fadd(&eb, ep, nir_vec4(&eb, nir_channel(&eb, tc, 0),
                                         nir_channel(&eb, tc, 1),
                                         nir_imm_float(&eb, 0), nir_imm_float(&eb, 0))),
         ez, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *e0 = nir_load_per_vertex_input(
         &eb, 4, 32, ez, ez, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *e1 = nir_load_per_vertex_input(
         &eb, 4, 32, ez, ez, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_def *e2 = nir_load_per_vertex_input(
         &eb, 4, 32, ez, ez, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR2, .num_slots = 1});
      nir_def *e3 = nir_load_per_vertex_input(
         &eb, 2, 32, ez, ez, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
      /* VAR0 = (tc.x, tc.y, e0.x, e0.y) */
      nir_store_output(
         &eb, nir_vec4(&eb, nir_channel(&eb, tc, 0), nir_channel(&eb, tc, 1),
                       nir_channel(&eb, e0, 0), nir_channel(&eb, e0, 1)),
         ez, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      /* VAR1 = (e0.z, e0.w, e1.x, e1.y) */
      nir_store_output(
         &eb, nir_vec4(&eb, nir_channel(&eb, e0, 2), nir_channel(&eb, e0, 3),
                       nir_channel(&eb, e1, 0), nir_channel(&eb, e1, 1)),
         ez, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      /* VAR2 = (e1.z, e1.w, e2.x, e2.y) */
      nir_store_output(
         &eb, nir_vec4(&eb, nir_channel(&eb, e1, 2), nir_channel(&eb, e1, 3),
                       nir_channel(&eb, e2, 0), nir_channel(&eb, e2, 1)),
         ez, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR2, .num_slots = 1});
      /* VAR3 = (e2.z, e3.x, e3.y, tc.x) -> 11 UNORM8/SNORM8 total + 2 UNORM16 + 3 smooth */
      nir_store_output(
         &eb, nir_vec4(&eb, nir_channel(&eb, e2, 2), nir_channel(&eb, e3, 0),
                       nir_channel(&eb, e3, 1), nir_channel(&eb, tc, 0)),
         ez, .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
      nir_shader_gather_info(eb.shader, nir_shader_get_entrypoint(eb.shader));

      /* Build FS reading all 16 components across VAR0..VAR3 as smooth float32. */
      nir_builder fb = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "vbo_pack_fs");
      fb.shader->info.io_lowered = true;
      nir_def *bary = nir_load_barycentric_pixel(&fb, 32, .interp_mode = INTERP_MODE_SMOOTH);
      nir_def *sum = nir_imm_vec4(&fb, 0, 0, 0, 0);
      for (unsigned s = 0; s < 4; ++s) {
         nir_def *v = nir_load_interpolated_input(
            &fb, 4, 32, bary, nir_imm_int(&fb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + s, .num_slots = 1});
         sum = nir_fadd(&fb, sum, v);
      }
      nir_store_output(
         &fb, sum, nir_imm_int(&fb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      nir_shader_gather_info(fb.shader, nir_shader_get_entrypoint(fb.shader));

      struct agx_apple9_vertex_layout vlayout = {
         .buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING,
         .draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING,
         .dynamic_stride = true,
      };
      vlayout.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
      vlayout.format[1] = fmts[fi];
      vlayout.format[2] = fmts[fi];
      vlayout.format[3] = fmts[fi];
      vlayout.format[4] = PIPE_FORMAT_R16G16_UNORM;

      /* Dynamic VI (use_prolog = true) MUST reject cleanly without mutating NIR. */
      struct agx_apple9_vertex_layout dyn_vlayout = vlayout;
      dyn_vlayout.use_prolog = true;
      CHECK(!hk_nir_tess_fused_pack_vbo_varyings(
         vb.shader, cb.shader, eb.shader, fb.shader, &dyn_vlayout));

      /* Static VI packs 11 8-bit norms -> 3 flat u32 + 2 16-bit norms -> 1 flat u32!
       * Total live varyings drop from 16 smooth -> 3 smooth + 4 flat = 7 (saving 9!). */
      CHECK(hk_nir_tess_fused_pack_vbo_varyings(
         vb.shader, cb.shader, eb.shader, fb.shader, &vlayout));
      nir_validate_shader(vb.shader, "vbo_pack_vs");
      nir_validate_shader(cb.shader, "vbo_pack_tcs");
      nir_validate_shader(eb.shader, "vbo_pack_tes");
      nir_validate_shader(fb.shader, "vbo_pack_fs");

      /* Verify TES + FS compile cleanly and live varying count dropped by 9. */
      struct hk_tess_fused_options opts = {
         .vertex = vlayout,
         .max_payload_bytes = 1024,
         .has_fs_inputs_mask = true,
      };
      nir_foreach_block(blk, nir_shader_get_entrypoint(fb.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic != nir_intrinsic_load_input &&
                intr->intrinsic != nir_intrinsic_load_interpolated_input)
               continue;
            unsigned loc = nir_intrinsic_io_semantics(intr).location;
            if (loc >= VARYING_SLOT_VAR0 && loc < VARYING_SLOT_VAR0 + 32)
               opts.fs_var_read_mask[loc - VARYING_SLOT_VAR0] |=
                  (uint8_t)(nir_def_components_read(&intr->def) <<
                            nir_intrinsic_component(intr));
         }
      }
      unsigned live_fs_scalars = 0;
      for (unsigned s = 0; s < 32; ++s)
         live_fs_scalars += util_bitcount(opts.fs_var_read_mask[s]);
      CHECK(live_fs_scalars == 7u);

      nir_shader *obj = NULL, *mesh = NULL;
      struct hk_tess_fused_info info = {0};
      const char *reason = NULL;
      CHECK(hk_nir_tess_fused_lower(vb.shader, cb.shader, eb.shader, &opts,
                                    &obj, &mesh, &info, &reason));
      /* 4 POS + 7 varyings = 11 comps -> planes=12 -> golden V24/P20 (unbounded)
       * and Tier 2 K=16 (recognizing hoisted narrow-VBO pack_* passthroughs!). */
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(info.multipatch_tier == 2 && info.patches_per_group == 16);
      /* Exclusive R16G16_UNORM slot VAR3 was also compacted in TCS (2 floats -> 1 u32). */
      ralloc_free(obj);
      ralloc_free(mesh);
      ralloc_free(vb.shader);
      ralloc_free(cb.shader);
      ralloc_free(eb.shader);
      ralloc_free(fb.shader);
   }

   /* 3. Verify TCS->TES payload compaction shrinks payload_B by 12 B per exclusive vec4,
    *    and verify float32 VBO passthrough is rejected cleanly. */
   for (unsigned mode = 0; mode < 2; ++mode) {
      bool is_unorm = (mode == 0);
      nir_builder vb = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "vbo_pack_payload_vs");
      vb.shader->info.io_lowered = true;
      nir_def *a0 = nir_load_input(
         &vb, 4, 32, nir_imm_int(&vb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
      nir_def *a1 = nir_load_input(
         &vb, 4, 32, nir_imm_int(&vb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC1, .num_slots = 1});
      nir_store_output(&vb, a0, nir_imm_int(&vb, 0), .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_output(&vb, a1, nir_imm_int(&vb, 0), .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

      nir_shader *tcs = make_fused_tcs(1, 1, 1.0f, 1.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);

      nir_builder eb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_EVAL, &agx_nir_options, "vbo_pack_payload_tes");
      eb.shader->info.io_lowered = true;
      eb.shader->info.tess._primitive_mode = TESS_PRIMITIVE_QUADS;
      eb.shader->info.tess.spacing = TESS_SPACING_EQUAL;
      nir_def *epos = nir_load_per_vertex_input(
         &eb, 4, 32, nir_imm_int(&eb, 0), nir_imm_int(&eb, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *ev0 = nir_load_per_vertex_input(
         &eb, 4, 32, nir_imm_int(&eb, 0), nir_imm_int(&eb, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(&eb, epos, nir_imm_int(&eb, 0), .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_output(&eb, ev0, nir_imm_int(&eb, 0), .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader_gather_info(eb.shader, nir_shader_get_entrypoint(eb.shader));

      nir_builder fb = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "vbo_pack_payload_fs");
      fb.shader->info.io_lowered = true;
      nir_def *bary = nir_load_barycentric_pixel(&fb, 32, .interp_mode = INTERP_MODE_SMOOTH);
      nir_def *v = nir_load_interpolated_input(
         &fb, 4, 32, bary, nir_imm_int(&fb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(
         &fb, v, nir_imm_int(&fb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      nir_shader_gather_info(fb.shader, nir_shader_get_entrypoint(fb.shader));

      struct agx_apple9_vertex_layout vl = {
         .buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING,
         .draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING,
         .dynamic_stride = true,
      };
      vl.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
      vl.format[1] = is_unorm ? PIPE_FORMAT_R8G8B8A8_UNORM
                              : PIPE_FORMAT_R32G32B32A32_FLOAT;

      struct hk_tess_fused_options opts = {
         .vertex = vl,
         .max_payload_bytes = 1024,
         .disable_multipatch = true,
      };
      struct hk_tess_fused_info before_info = {0}, after_info = {0};
      CHECK(hk_nir_tess_fused_lower(vb.shader, tcs, eb.shader, &opts, NULL,
                                    NULL, &before_info, NULL));
      if (is_unorm) {
         CHECK(hk_nir_tess_fused_pack_vbo_varyings(vb.shader, tcs, eb.shader,
                                                   fb.shader, &vl));
         CHECK(hk_nir_tess_fused_lower(vb.shader, tcs, eb.shader, &opts, NULL,
                                       NULL, &after_info, NULL));
         /* 40 B header + 16 B POS + 16 B VAR0 = 72 B (aligned 80 B) ->
          * 40 B header + 16 B POS + 4 B packed VAR0 = 60 B (aligned 64 B). */
         CHECK(after_info.payload_B < before_info.payload_B);
         CHECK(before_info.payload_B - after_info.payload_B == 16u);
         /* With multi-patch enabled, hoisted pack_unorm_4x8 stays Tier 2 K=16! */
         struct hk_tess_fused_options mp_opts = opts;
         mp_opts.disable_multipatch = false;
         struct hk_tess_fused_info mp_info = {0};
         CHECK(hk_nir_tess_fused_lower(vb.shader, tcs, eb.shader, &mp_opts, NULL,
                                       NULL, &mp_info, NULL));
         CHECK(mp_info.multipatch_tier == 2 && mp_info.patches_per_group == 16);
      } else {
         CHECK(!hk_nir_tess_fused_pack_vbo_varyings(vb.shader, tcs, eb.shader,
                                                    fb.shader, &vl));
      }
      ralloc_free(vb.shader);
      ralloc_free(tcs);
      ralloc_free(eb.shader);
      ralloc_free(fb.shader);
   }

   printf("fused tessellation VBO pack suite: host uint->unorm/snorm pack/unpack equivalence + 4 norm formats + payload compaction + rejections passed\n");
}

static void
fused_tess_interval_bounds_suite(void)
{
   struct hk_tess_fused_options opts = {
      .vertex = {
         .buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING,
         .draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING,
         .dynamic_stride = true,
      },
      .max_payload_bytes = 1024,
   };

   /* 1. Fused VS + TCS distance formula:
    *    VS:  v = cond ? (8.0 - (sqrt(dx^2 + dy^2 + dz^2) - 3.0) * 0.363636374) : 0.0
    *    TCS: e = cull ? 0.0 : fmin(fmax(0.5*(v[i]+v[j]), 1.0), 64.0)
    *         inner = (e0 + e1 + e2) * 0.333
    *    Provable hi <= 9.0909 -> FractionalOdd seg=11 -> 181 tris/patch (33x tighter). */
   for (unsigned safe_nan = 0; safe_nan < 2; ++safe_nan) {
      nir_builder vb = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "interval_dist_vs");
      vb.shader->info.io_lowered = true;
      nir_def *u = nir_load_ubo(&vb, 4, 32, nir_imm_int(&vb, 0),
                                nir_imm_int(&vb, 0), .align_mul = 4, .range = 16);
      nir_def *dx = nir_channel(&vb, u, 0);
      nir_def *dy = nir_channel(&vb, u, 1);
      nir_def *dz = nir_channel(&vb, u, 2);
      nir_def *dist2 =
         nir_fadd(&vb, nir_fadd(&vb, nir_fmul(&vb, dx, dx), nir_fmul(&vb, dy, dy)),
                  nir_fmul(&vb, dz, dz));
      nir_def *dist = nir_fsqrt(&vb, dist2);
      nir_def *lvl = nir_fadd(
         &vb, nir_imm_float(&vb, 8.0f),
         nir_fneg(&vb, nir_fmul_imm(&vb, nir_fadd_imm(&vb, dist, -3.0f),
                                    0.363636374f)));
      nir_def *en = nir_ine_imm(&vb, nir_channel(&vb, u, 3), 0);
      nir_def *v = nir_bcsel(&vb, en, lvl, nir_imm_float(&vb, 0.0f));
      nir_store_output(&vb, nir_vec4(&vb, dx, dy, dz, v), nir_imm_int(&vb, 0),
                       .write_mask = 0xf, .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_dist_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 3;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pin = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pin, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
      nir_def *v0 = nir_channel(
         &cb,
         nir_load_per_vertex_input(
            &cb, 4, 32, nir_imm_int(&cb, 0), nir_imm_int(&cb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1}),
         3);
      nir_def *v1 = nir_channel(
         &cb,
         nir_load_per_vertex_input(
            &cb, 4, 32, nir_imm_int(&cb, 1), nir_imm_int(&cb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1}),
         3);
      nir_def *v2 = nir_channel(
         &cb,
         nir_load_per_vertex_input(
            &cb, 4, 32, nir_imm_int(&cb, 2), nir_imm_int(&cb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1}),
         3);
      nir_def *cull = nir_ine_imm(
         &cb,
         nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0), nir_imm_int(&cb, 0),
                      .align_mul = 4, .range = 16),
         0);
      nir_def *edges[3] = {
         nir_fmul_imm(&cb, nir_fadd(&cb, v1, v2), 0.5f),
         nir_fmul_imm(&cb, nir_fadd(&cb, v2, v0), 0.5f),
         nir_fmul_imm(&cb, nir_fadd(&cb, v0, v1), 0.5f),
      };
      for (unsigned e = 0; e < 3; ++e) {
         nir_def *x = edges[e];
         /* With safe_nan=1, fmax(x, 1.0) eliminates NaN before fmin(..., 64.0).
          * With safe_nan=0, IEEE fmin(NaN, 64.0) == 64.0, so analysis MUST stay
          * conservative at worst-case 5953! */
         if (safe_nan)
            x = nir_fmax(&cb, x, nir_imm_float(&cb, 1.0f));
         x = nir_fmin(&cb, x, nir_imm_float(&cb, 64.0f));
         edges[e] = nir_bcsel(&cb, cull, x, nir_imm_float(&cb, 0.0f));
      }
      nir_store_output(
         &cb, nir_vec3(&cb, edges[0], edges[1], edges[2]), nir_imm_int(&cb, 0),
         .write_mask = 0x7, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_def *inner = nir_fmul_imm(
         &cb, nir_fadd(&cb, nir_fadd(&cb, edges[0], edges[1]), edges[2]), 0.333f);
      nir_store_output(
         &cb, inner, nir_imm_int(&cb, 0), .write_mask = 0x1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_FRACTIONAL_ODD, false, false,
                                       3, 4, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      const char *reason = NULL;
      CHECK(hk_nir_tess_fused_lower(vb.shader, cb.shader, tes, &opts, NULL,
                                    NULL, &info, &reason));
      CHECK(!info.constant_tess_levels);
      if (safe_nan) {
         CHECK(info.max_prims_per_patch == 181u);
         CHECK(info.max_child_wgs_per_patch == 16u);
      } else {
         CHECK(info.max_prims_per_patch == 5953u);
      }
      ralloc_free(vb.shader);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 2. Dynamic TCS clamped by both a static constant (8.0) and a push-constant
    *    cap (fmin(fmin(ubo, push[8]), 8.0)), plus load_output(OUTER) -> INNER. */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_push_cap_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *pcap = nir_load_push_constant(
         &cb, 1, 32, nir_imm_int(&cb, 8), .base = 0, .range = 16);
      nir_def *bounded = nir_fmin(&cb, nir_fmin(&cb, dyn, pcap),
                                  nir_imm_float(&cb, 8.0f));
      nir_store_output(
         &cb, nir_vec4(&cb, bounded, bounded, bounded, bounded),
         nir_imm_int(&cb, 0), .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_def *o0 = nir_load_output(
         &cb, 1, 32, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, o0, o0), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 4, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      /* Static clamp <= 8.0 on Quads Equal: 2 * 8 * 8 = 128 tris/patch (vs 8192). */
      CHECK(info.max_prims_per_patch == 128u);
      CHECK(info.push_levels.valid);
      CHECK(info.push_levels.levels[0].kind == HK_APPLE9_TESS_LEVEL_PUSH_BOUND);
      CHECK(info.push_levels.levels[4].kind == HK_APPLE9_TESS_LEVEL_PUSH_BOUND);

      struct hk_shader_info sinfo = {
         .apple9_tess_fused = true,
         .apple9_tess_max_child_wgs = info.max_child_wgs_per_patch,
         .apple9_tess_max_patch_prims = info.max_prims_per_patch,
         .apple9_tess_push_levels = info.push_levels,
      };
      const struct {
         float push_cap;
         uint32_t expected_prims;
      } cases[] = {
         {0.0f, 0u},      /* push cap <= 0 -> guaranteed culled */
         {-2.0f, 0u},
         {2.0f, 8u},      /* 2 * 2 * 2 */
         {4.0f, 32u},     /* 2 * 4 * 4 */
         {16.0f, 128u},   /* capped by static fmin(..., 8.0) */
         {NAN, 128u},     /* NaN push cap MUST fall back to static 8.0, never cull! */
      };
      for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
         uint8_t push[HK_MAX_PUSH_SIZE] = {0};
         memcpy(push + 8, &cases[i].push_cap, sizeof(float));
         uint32_t wgs = 999, prims = 999;
         hk_nir_tess_fused_eval_counts(&sinfo, push, &wgs, &prims);
         CHECK(prims == cases[i].expected_prims);
      }
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 3. Conservative fallback on conditional-only VS write and TCS loop-carried phi. */
   {
      nir_builder vb = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "interval_cond_vs");
      vb.shader->info.io_lowered = true;
      nir_def *u = nir_load_ubo(&vb, 1, 32, nir_imm_int(&vb, 0),
                                nir_imm_int(&vb, 0), .align_mul = 4, .range = 16);
      nir_store_output(&vb, nir_imm_vec4(&vb, 0, 0, 0, 1), nir_imm_int(&vb, 0),
                       .write_mask = 0x7, .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&vb, nir_ine_imm(&vb, u, 0));
      nir_store_output(&vb, nir_imm_float(&vb, 2.0f), nir_imm_int(&vb, 0),
                       .component = 3, .write_mask = 0x1, .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&vb, NULL);
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_cond_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pin = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pin, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *w = nir_channel(&cb, pin, 3);
      nir_store_output(
         &cb, nir_vec4(&cb, w, w, w, w), nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, w, w), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vb.shader, cb.shader, tes, &opts, NULL,
                                    NULL, &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 8192u);
      ralloc_free(vb.shader);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 4. fsat(ubo) * 6.0f + bcsel clamp -> provable hi <= 6.0f (72 tris/patch). */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_fsat_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *lvl = nir_fmul_imm(&cb, nir_fsat(&cb, dyn), 6.0f);
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 72u); /* 2 * 6 * 6 */
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 5. 64-bit u2f32(u64) * 2^-58 must NOT receive 32-bit bounds (x=2^63 -> 32.0). */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_u64_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *u64 = nir_load_ubo(&cb, 1, 64, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 8, .range = 16);
      nir_def *lvl = nir_fmul_imm(&cb, nir_u2f32(&cb, u64), ldexpf(1.0f, -58));
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 8192u);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 6. IEEE frsq(-0.0) = -Inf: fmin(-frsq(ftrunc(clamp(x, -0.5, 0.5))), 64.0)
    *    produces 64.0 at x=-0.25, so analyzer must NOT report 0 prims! */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_negzero_rsq_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *z = nir_ftrunc(
         &cb, nir_fmin(&cb, nir_fmax(&cb, dyn, nir_imm_float(&cb, -0.5f)),
                       nir_imm_float(&cb, 0.5f)));
      nir_def *lvl = nir_fmin(&cb, nir_fneg(&cb, nir_frsq(&cb, z)),
                              nir_imm_float(&cb, 64.0f));
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 8192u);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 7. fdiv(x in [0, 21], 7.0) lowered as x * rcp(7.0) yields 3.0000002384
    *    at x=21 -> Equal spacing rounds up to 4 segments (32 tris, NOT 18). */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_fdiv_rcp_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *x = nir_fmin(&cb, nir_fmax(&cb, dyn, nir_imm_float(&cb, 0.0f)),
                            nir_imm_float(&cb, 21.0f));
      nir_def *lvl = nir_fdiv(&cb, x, nir_imm_float(&cb, 7.0f));
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 32u); /* 2 * 4 * 4 */
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 8. Apple9 lowers fsqrt(x) as x * rsq(x): at x=49, 49 * rsq(49) = 7 +/- 2^-21,
    *    so neither sqrt(clamp(x, 0, 49)) - 7 nor 7 - sqrt(clamp(x, 49, 64)) may
    *    be falsely bounded to <= 0 (culled)! */
   for (unsigned dir = 0; dir < 2; ++dir) {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_sqrt_cancel_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *clamped = nir_fmin(
         &cb, nir_fmax(&cb, dyn, nir_imm_float(&cb, dir == 0 ? 0.0f : 49.0f)),
         nir_imm_float(&cb, dir == 0 ? 49.0f : 64.0f));
      nir_def *sq = nir_fsqrt(&cb, clamped);
      nir_def *lvl = dir == 0 ? nir_fsub(&cb, sq, nir_imm_float(&cb, 7.0f))
                              : nir_fsub(&cb, nir_imm_float(&cb, 7.0f), sq);
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 2u); /* Live positive epsilon -> 1x1 quad = 2 tris */
      CHECK(info.max_child_wgs_per_patch == 1u);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* 9. Exact boundary precision regressions:
    *    (a) fsqrt factor*x error propagation at x = 3.5145106315612793:
    *        1.8747024536132812 - sqrt(clamp(x, 3.5145106315612793, 4.0)) can be
    *        positive (> 0) for a 1.997-ULP factor -> must NOT report 0 prims.
    *    (b) frsq power-of-two binade crossing at x = 0.9999999403953552:
    *        0.9999998807907104 - frsq(clamp(x, 0.5, 0.9999999403953552)) can be
    *        positive for a 1.75-ULP Vulkan InverseSqrt -> must NOT report 0 prims. */
   for (unsigned k = 0; k < 2; ++k) {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "interval_ulp_boundary_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 1;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_def *lvl;
      if (k == 0) {
         nir_def *x = nir_fmin(&cb, nir_fmax(&cb, dyn, nir_imm_float(&cb, 3.5145106315612793f)),
                               nir_imm_float(&cb, 4.0f));
         lvl = nir_fsub(&cb, nir_imm_float(&cb, 1.8747024536132812f), nir_fsqrt(&cb, x));
      } else {
         nir_def *x = nir_fmin(&cb, nir_fmax(&cb, dyn, nir_imm_float(&cb, 0.5f)),
                               nir_imm_float(&cb, 0.9999999403953552f));
         lvl = nir_fsub(&cb, nir_imm_float(&cb, 0.9999998807907104f), nir_frsq(&cb, x));
      }
      nir_store_output(
         &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0), .write_mask = 0x3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_prims_per_patch == 2u);
      CHECK(info.max_child_wgs_per_patch == 1u);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   printf("fused tessellation interval bounds suite: VS+TCS distance + NaN/u64/-0/fdiv/sqrt/rsq soundness + push cap + fallbacks passed\n");
}

static uint32_t
ref_stitch_mask(uint32_t h)
{
   if (h <= 1u)
      return 0u;
   uint32_t k = h - 1u;
   uint32_t b = 31u - (uint32_t)__builtin_clz(k);
   uint32_t r = k - (1u << b);
   static const uint32_t base[5] = {
      0x00000000u, 0x00008000u, 0x00808080u, 0x08888888u, 0x2aaaaaaau,
   };
   static const uint32_t lvl[5] = {
      0x00008000u, 0x00800080u, 0x08080808u, 0x22222222u, 0x55555555u,
   };
   uint32_t bits = (r + 1u) * (32u >> b);
   uint32_t cut = bits >= 32u ? 0xffffffffu : ((1u << bits) - 1u);
   return base[b] | (lvl[b] & cut);
}

static uint32_t
ref_stitch_i_half(uint32_t h_o, uint32_t h_i, uint32_t t)
{
   uint32_t m_o = ref_stitch_mask(h_o);
   uint32_t m_i = ref_stitch_mask(h_i);
   uint32_t o_init = h_o ? 1u : 0u;
   if (t <= o_init)
      return 0u;
   uint32_t rem = t - o_init;
   uint32_t pos = 0u;
   static const uint32_t steps[5] = {16u, 8u, 4u, 2u, 1u};
   for (unsigned s = 0; s < 5; ++s) {
      uint32_t cand = pos + steps[s];
      uint32_t lo = (1u << cand) - 1u;
      uint32_t c = (uint32_t)__builtin_popcount(m_i & lo) +
                   (uint32_t)__builtin_popcount(m_o & lo);
      if (c <= rem)
         pos = cand;
   }
   uint32_t lo = (1u << pos) - 1u;
   uint32_t ci = (uint32_t)__builtin_popcount(m_i & lo);
   uint32_t co = (uint32_t)__builtin_popcount(m_o & lo);
   if (ci + co < rem)
      ci += (m_i >> pos) & 1u;
   return ci;
}

static uint32_t
ref_stitch_i(uint32_t n_out, uint32_t n_in_seg, uint32_t t)
{
   uint32_t total = n_out + n_in_seg;
   uint32_t h_o = n_out >> 1;
   uint32_t k = n_in_seg >> 1;
   uint32_t h_i = k + 1u;
   uint32_t half_t = h_o + k;
   if (t <= half_t)
      return ref_stitch_i_half(h_o, h_i, t);
   uint32_t rt = t <= total ? total - t : 0u;
   if (rt <= half_t)
      return n_in_seg - ref_stitch_i_half(h_o, h_i, rt);
   return k + 1u;
}

static uint32_t
ref_float_to_fxp(float val)
{
   return (uint32_t)(val * 65536.0f + 0.5f);
}

static uint32_t
ref_remove_msb(uint32_t val)
{
   if (!val)
      return 0;
   return val & ~(1u << (31u - __builtin_clz(val)));
}

static uint32_t
ref_eval_1d_fxp(enum gl_tess_spacing spacing, uint32_t n, float f, uint32_t k)
{
   if (spacing == TESS_SPACING_EQUAL) {
      bool odd = (n & 1u) != 0;
      uint32_t num_half = (n + 1u) >> 1;
      bool flip = k >= num_half;
      uint32_t j = flip ? n - k : k;
      if (!odd && j == num_half)
         return 32768u;
      uint32_t q = j * ref_float_to_fxp(1.0f / (float)n);
      return flip ? 65536u - q : q;
   }
   bool odd = spacing == TESS_SPACING_FRACTIONAL_ODD;
   uint32_t fxp_tess = ref_float_to_fxp(f);
   uint32_t fxp_half = (fxp_tess + 1u) >> 1;
   if (odd || fxp_half == 32768u)
      fxp_half += 32768u;
   uint32_t floor_int = fxp_half >> 16;
   uint32_t ceil_int = (fxp_half + 65535u) >> 16;
   uint32_t fxp_floor = floor_int << 16;
   uint32_t fxp_ceil = ceil_int << 16;
   uint32_t frac = fxp_half - fxp_floor;
   uint32_t split;
   if (fxp_ceil == fxp_floor)
      split = ceil_int + 1u;
   else if (odd)
      split = floor_int == 1u ? 0u : (ref_remove_msb(floor_int - 1u) << 1) + 1u;
   else
      split = (ref_remove_msb(floor_int) << 1) + 1u;
   uint32_t floor_segs = MAX2((floor_int << 1) - (odd ? 1u : 0u), 1u);
   uint32_t ceil_segs = MAX2((ceil_int << 1) - (odd ? 1u : 0u), 1u);
   uint32_t inv_floor = ref_float_to_fxp(1.0f / (float)floor_segs);
   uint32_t inv_ceil = ref_float_to_fxp(1.0f / (float)ceil_segs);
   bool flip = k >= ceil_int;
   uint32_t j = flip ? ((ceil_int << 1) - k - (odd ? 1u : 0u)) : k;
   if (j == ceil_int)
      return 32768u;
   uint32_t idx_floor = j > split ? j - 1u : j;
   uint32_t loc_floor = idx_floor * inv_floor;
   uint32_t loc_ceil = j * inv_ceil;
   uint32_t q = (loc_floor * (65536u - frac) + loc_ceil * frac + 32768u) >> 16;
   return flip ? 65536u - q : q;
}

static float
ref_eval_1d(enum gl_tess_spacing spacing, uint32_t n, float f, uint32_t k)
{
   float t = (float)ref_eval_1d_fxp(spacing, n, f, k) * (1.0f / 65536.0f);
   /* Verify exact-complement invariance (Vulkan tessellation invariance rules 3-4). */
   CHECK(1.0f - (1.0f - t) == t);
   return t;
}

static void
ref_quad_outer_uv(enum gl_tess_spacing spacing, uint32_t e, uint32_t k,
                  const uint32_t n_out[4], const float f_out[4], float uv[2])
{
   uint32_t n_e = n_out[e];
   float f_e = f_out[e];
   uint32_t q = (e == 0 || e == 3) ? (n_e > k ? n_e - k : 0) : k;
   float t = ref_eval_1d(spacing, n_e, f_e, q);
   bool odd = (e & 1) != 0;
   uv[0] = odd ? t : (e == 2 ? 1.0f : 0.0f);
   uv[1] = odd ? (e == 3 ? 1.0f : 0.0f) : t;
}

static void
ref_quad_inner_uv(enum gl_tess_spacing spacing, uint32_t e, uint32_t k,
                  uint32_t niu, uint32_t niv, float fiu, float fiv, float uv[2])
{
   uint32_t iu = e == 0 ? 1 : (e == 1 ? k + 1 : (e == 2 ? niu - 1 : niu - 1 - k));
   uint32_t iv = e == 0 ? niv - 1 - k : (e == 1 ? 1 : (e == 2 ? k + 1 : niv - 1));
   uv[0] = ref_eval_1d(spacing, niu, fiu, iu);
   uv[1] = ref_eval_1d(spacing, niv, fiv, iv);
}

static void
ref_tri_ring_uv(enum gl_tess_spacing spacing, uint32_t r, uint32_t e, uint32_t k,
                const uint32_t n_out[3], const float f_out[3], uint32_t ni,
                float fi, float uv[2])
{
   if (r == 0) {
      uint32_t n_e = n_out[e];
      float f_e = f_out[e];
      uint32_t q = e == 1 ? k : (n_e > k ? n_e - k : 0);
      uint32_t t_fxp = ref_eval_1d_fxp(spacing, n_e, f_e, q);
      uint32_t u_fxp = e == 0 ? 0 : t_fxp;
      uint32_t v_fxp = e == 0 ? t_fxp : (e == 1 ? 0 : 65536u - t_fxp);
      uv[0] = (float)u_fxp * (1.0f / 65536.0f);
      uv[1] = (float)v_fxp * (1.0f / 65536.0f);
      /* Verify exact w == 0.0f on edge 2 (u + v == 1.0f bit-for-bit). */
      if (e == 2)
         CHECK(1.0f - uv[0] - uv[1] == 0.0f);
      return;
   }
   uint32_t r2 = r << 1;
   if (r2 == ni) {
      uv[0] = 21845.0f * (1.0f / 65536.0f);
      uv[1] = 21845.0f * (1.0f / 65536.0f);
      return;
   }
   uint32_t s_r = ni - r2;
   uint32_t perp_raw = ref_eval_1d_fxp(spacing, ni, fi, r);
   uint32_t perp_fxp = (perp_raw * 0x0000aaaau + 32768u) >> 16;
   uint32_t half_perp = (perp_fxp + 1u) >> 1;
   uint32_t e_c = k == s_r ? (e == 2 ? 0 : e + 1) : e;
   uint32_t k_c = k == s_r ? 0 : k;
   uint32_t q = e_c == 1 ? r + k_c : (ni - r - k_c);
   uint32_t param_fxp = ref_eval_1d_fxp(spacing, ni, fi, q);
   uint32_t tmp_fxp = param_fxp - half_perp;
   uint32_t u_fxp = e_c == 0 ? perp_fxp : tmp_fxp;
   uint32_t v_fxp = e_c == 0 ? tmp_fxp : (e_c == 1 ? perp_fxp : 65536u - tmp_fxp - perp_fxp);
   uv[0] = (float)u_fxp * (1.0f / 65536.0f);
   uv[1] = (float)v_fxp * (1.0f / 65536.0f);
   CHECK(1.0f - (1.0f - uv[0]) == uv[0]);
   CHECK(1.0f - (1.0f - uv[1]) == uv[1]);
}

static uint32_t
ref_seg_count(enum gl_tess_spacing spacing, float clamped)
{
   if (spacing == TESS_SPACING_EQUAL)
      return (uint32_t)clamped;
   uint32_t fxp = ref_float_to_fxp(clamped);
   bool odd = spacing == TESS_SPACING_FRACTIONAL_ODD;
   uint32_t f = fxp + (odd ? 65537u : 1u);
   uint32_t half_ceil = ((f >> 1) + 65535u) >> 16;
   return odd ? (2u * half_ceil - 1u) : (2u * half_ceil);
}

static void
fused_tess_topology_suite(void)
{
   /* 0. Spot-check bit-exact match with upstream poly/cl/tessellator.h
    * PlacePointIn1D & NumPointsForTessFactor across sub-1/65536 boundaries. */
   CHECK(ref_eval_1d_fxp(TESS_SPACING_EQUAL, 7, 7.0f, 2) == 18724u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_EVEN, 4, 2.5f, 1) == 28672u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_EVEN, 4, 2.5f, 3) == 65536u - 28672u);
   /* Exact upstream PlacePointIn1D checks: promoted odd inner (n=3, f=1.0,
    * k=1 -> 65536) and unclamped fractional-even midpoint neighbor (6 + 2^-16,
    * n=8, k=3 -> 32769). */
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_ODD, 3, 1.0f, 1) == 65536u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_ODD, 3, 1.0f, 2) == 65536u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_EVEN, 8,
                         6.0f + 1.0f / 65536.0f, 3) == 32769u);
   /* Sub-1/65536 nextafter boundaries quantize to 16.16 before parity rounding,
    * matching upstream tessellator.h with zero collapsed frac==0 folds. */
   CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_ODD, nextafterf(3.0f, INFINITY)) == 3u);
   CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_ODD, 3.0f + 1.0f / 65536.0f) == 5u);
   CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_EVEN, nextafterf(4.0f, INFINITY)) == 4u);
   CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_EVEN, 4.0f + 1.0f / 65536.0f) == 6u);
   {
      uint32_t prev = 0;
      for (uint32_t k = 0; k <= 5; ++k) {
         uint32_t q = ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_ODD, 5,
                                      3.0f + 1.0f / 65536.0f, k);
         CHECK(q >= prev);
         prev = q;
      }
      CHECK(prev == 65536u);
   }

   /* 1. Exhaustively verify stitcher monotonicity, endpoints, and symmetry
    * over all 64 * 63 = 4,032 valid (n_out, n_in_seg) pairs ([1..64] x [0..62]). */
   for (uint32_t no = 1; no <= 64; ++no) {
      for (uint32_t ni = 0; ni <= 62; ++ni) {
         uint32_t total = no + ni;
         CHECK(ref_stitch_i(no, ni, 0) == 0);
         CHECK(ref_stitch_i(no, ni, total) == ni);
         for (uint32_t t = 0; t < total; ++t) {
            uint32_t a = ref_stitch_i(no, ni, t);
            uint32_t b = ref_stitch_i(no, ni, t + 1);
            CHECK(b == a || b == a + 1);
            uint32_t oa = t - a, ob = (t + 1) - b;
            CHECK(oa <= no && ob <= no && a <= ni && b <= ni);
         }
      }
   }

   /* 2. Verify exact area sum, strictly non-negative CW (+1 cross) winding,
    * and meshlet V/P bounds for Quads and Triangles across all 3 spacings and
    * all 4 meshlet shapes (including asymmetric levels and 63/64 max). */
   struct shape_cfg {
      uint32_t max_v, max_p, tw, th, ps;
      bool whole;
   };
   const struct shape_cfg shapes[] = {
      {32, 42, 5, 4, 30, true},
      {24, 20, 3, 3, 20, true},
      {16, 18, 3, 3, 14, true},
      {8, 6, 3, 1, 6, true},
      {4, 2, 1, 1, 2, false},
   };
   const enum gl_tess_spacing spacings[] = {
      TESS_SPACING_EQUAL,
      TESS_SPACING_FRACTIONAL_ODD,
      TESS_SPACING_FRACTIONAL_EVEN,
   };
   const float cases[][6] = {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f},
      {1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 1.0f},
      {2.25f, 3.75f, 5.1f, 6.9f, 3.3f, 5.8f},
      {1.0f, 17.5f, 9.25f, 32.0f, 2.5f, 15.0f},
      {3.000000476837158f, 3.000000476837158f, 3.000000476837158f,
       3.000000476837158f, 3.000000476837158f, 3.000000476837158f},
      /* Mixed outer/inner nextafter boundary cases (R1/R2 regression). */
      {2.0f, 2.0f, 2.0f, 2.0f, nextafterf(4.0f, INFINITY),
       nextafterf(4.0f, INFINITY)},
      {2.0f, 2.0f, 2.0f, 2.0f, 4.0f + 1.0f / 65536.0f,
       4.0f + 1.0f / 65536.0f},
      {1.0f, 1.0f, 1.0f, 1.0f, nextafterf(3.0f, INFINITY),
       nextafterf(3.0f, INFINITY)},
      {1.0f, 1.0f, 1.0f, 1.0f, 3.0f + 1.0f / 65536.0f,
       3.0f + 1.0f / 65536.0f},
      {nextafterf(1.0f, INFINITY), nextafterf(1.0f, INFINITY),
       nextafterf(1.0f, INFINITY), nextafterf(1.0f, INFINITY), 1.0f, 1.0f},
      {63.0f, 63.0f, 63.0f, 63.0f, 63.0f, 63.0f},
      {64.0f, 64.0f, 64.0f, 64.0f, 64.0f, 64.0f},
   };

   unsigned verified = 0;
   for (unsigned si = 0; si < ARRAY_SIZE(spacings); ++si) {
      enum gl_tess_spacing sp = spacings[si];
      for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
         for (unsigned shi = 0; shi < ARRAY_SIZE(shapes); ++shi) {
            const struct shape_cfg *sh = &shapes[shi];
            /* Quad verification */
            {
               float f_out[4], f_in[2], maxf = 0.0f;
               float lo = sp == TESS_SPACING_FRACTIONAL_EVEN ? 2.0f : 1.0f;
               float hi = sp == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f;
               for (unsigned e = 0; e < 4; ++e) {
                  float f = cases[ci][e];
                  f = !(f >= lo) ? lo : (f > hi ? hi : f);
                  if (sp == TESS_SPACING_EQUAL)
                     f = ceilf(f);
                  f_out[e] = f;
                  if (f > maxf)
                     maxf = f;
               }
               for (unsigned a = 0; a < 2; ++a)
                  if (cases[ci][4 + a] > maxf)
                     maxf = cases[ci][4 + a];
               float in_lo = (sp == TESS_SPACING_FRACTIONAL_ODD &&
                              maxf > 1.0f + 0.5f / 65536.0f)
                                ? 1.0f + 1.0f / 65536.0f
                                : lo;
               for (unsigned a = 0; a < 2; ++a) {
                  float f = cases[ci][4 + a];
                  f = !(f >= in_lo) ? in_lo : (f > hi ? hi : f);
                  if (sp == TESS_SPACING_EQUAL)
                     f = ceilf(f);
                  f_in[a] = f;
               }
               uint32_t n_out[4], n_in[2];
               bool all_ones = sp != TESS_SPACING_FRACTIONAL_EVEN;
               for (unsigned e = 0; e < 4; ++e) {
                  n_out[e] = ref_seg_count(sp, f_out[e]);
                  if (n_out[e] != 1)
                     all_ones = false;
               }
               for (unsigned a = 0; a < 2; ++a) {
                  n_in[a] = ref_seg_count(sp, f_in[a]);
                  if (n_in[a] != 1)
                     all_ones = false;
               }
               double area = 0.0;
               if (all_ones) {
                  area = 1.0;
               } else {
                  uint32_t min_in = sp == TESS_SPACING_FRACTIONAL_ODD ? 3 : 2;
                  for (unsigned a = 0; a < 2; ++a) {
                     if (n_in[a] < min_in) {
                        n_in[a] = min_in;
                        if (sp == TESS_SPACING_EQUAL)
                           f_in[a] = (float)n_in[a];
                     }
                  }
                  uint32_t qu = n_in[0] - 2, qv = n_in[1] - 2;
                  for (unsigned e = 0; e < 4; ++e) {
                     uint32_t n_is = (e & 1) ? qu : qv;
                     uint32_t t_tot = n_out[e] + n_is;
                     uint32_t chunks = DIV_ROUND_UP(t_tot, sh->ps);
                     for (uint32_t c = 0; c < chunks; ++c) {
                        uint32_t t0 = c * sh->ps;
                        uint32_t t1 = MIN2(t0 + sh->ps, t_tot);
                        uint32_t i0 = ref_stitch_i(n_out[e], n_is, t0);
                        uint32_t o0 = t0 - i0;
                        uint32_t i1 = ref_stitch_i(n_out[e], n_is, t1);
                        uint32_t o1 = t1 - i1;
                        uint32_t no = (o1 - o0) + 1;
                        uint32_t ni_v = (i1 - i0) + 1;
                        CHECK(no + ni_v <= sh->max_v);
                        CHECK(t1 - t0 <= sh->max_p);
                        float vbuf[64][2];
                        for (uint32_t l = 0; l < no; ++l)
                           ref_quad_outer_uv(sp, e, o0 + l, n_out, f_out, vbuf[l]);
                        for (uint32_t l = 0; l < ni_v; ++l)
                           ref_quad_inner_uv(sp, e, i0 + l, n_in[0], n_in[1],
                                             f_in[0], f_in[1], vbuf[no + l]);
                        for (uint32_t p = 0; p < t1 - t0; ++p) {
                           uint32_t t = t0 + p;
                           uint32_t ia = ref_stitch_i(n_out[e], n_is, t);
                           uint32_t oi = (t - ia) - o0;
                           uint32_t ii = ia - i0;
                           bool step_i =
                              ref_stitch_i(n_out[e], n_is, t + 1) != ia;
                           uint32_t a = step_i ? no + ii : oi;
                           uint32_t b = step_i ? oi : oi + 1;
                           uint32_t d = step_i ? no + ii + 1 : no + ii;
                           CHECK(a < no + ni_v && b < no + ni_v && d < no + ni_v);
                           double cr =
                              ((double)vbuf[b][0] - vbuf[a][0]) *
                                 ((double)vbuf[d][1] - vbuf[a][1]) -
                              ((double)vbuf[b][1] - vbuf[a][1]) *
                                 ((double)vbuf[d][0] - vbuf[a][0]);
                           CHECK(cr >= -1e-6);
                           area += 0.5 * cr;
                        }
                     }
                  }
                  if (qu && qv) {
                     uint32_t tu = DIV_ROUND_UP(qu, sh->tw);
                     uint32_t tv = DIV_ROUND_UP(qv, sh->th);
                     for (uint32_t ty = 0; ty < tv; ++ty) {
                        for (uint32_t tx = 0; tx < tu; ++tx) {
                           uint32_t ox = tx * sh->tw, oy = ty * sh->th;
                           uint32_t cw = MIN2(sh->tw, qu - ox);
                           uint32_t ch = MIN2(sh->th, qv - oy);
                           CHECK((cw + 1) * (ch + 1) <= sh->max_v);
                           CHECK(2 * cw * ch <= sh->max_p);
                           for (uint32_t qy = 0; qy < ch; ++qy) {
                              for (uint32_t qx = 0; qx < cw; ++qx) {
                                 float u0 = ref_eval_1d(sp, n_in[0], f_in[0], ox + 1 + qx);
                                 float u1 = ref_eval_1d(sp, n_in[0], f_in[0], ox + 2 + qx);
                                 float v0 = ref_eval_1d(sp, n_in[1], f_in[1], oy + 1 + qy);
                                 float v1 = ref_eval_1d(sp, n_in[1], f_in[1], oy + 2 + qy);
                                 double cr = (double)(u1 - u0) * (double)(v1 - v0);
                                 CHECK(cr >= 0.0);
                                 area += cr;
                              }
                           }
                        }
                     }
                  }
               }
               CHECK(fabs(area - 1.0) < 1e-4);
               verified++;
            }

            /* Triangle verification */
            {
               float f_out[3], f_in, maxf = 0.0f;
               float lo = sp == TESS_SPACING_FRACTIONAL_EVEN ? 2.0f : 1.0f;
               float hi = sp == TESS_SPACING_FRACTIONAL_ODD ? 63.0f : 64.0f;
               for (unsigned e = 0; e < 3; ++e) {
                  float f = cases[ci][e];
                  f = !(f >= lo) ? lo : (f > hi ? hi : f);
                  if (sp == TESS_SPACING_EQUAL)
                     f = ceilf(f);
                  f_out[e] = f;
                  if (f > maxf)
                     maxf = f;
               }
               if (cases[ci][4] > maxf)
                  maxf = cases[ci][4];
               float in_lo = (sp == TESS_SPACING_FRACTIONAL_ODD &&
                              maxf > 1.0f + 0.5f / 65536.0f)
                                ? 1.0f + 1.0f / 65536.0f
                                : lo;
               f_in = cases[ci][4];
               f_in = !(f_in >= in_lo) ? in_lo : (f_in > hi ? hi : f_in);
               if (sp == TESS_SPACING_EQUAL)
                  f_in = ceilf(f_in);
               uint32_t n_out[3], ni;
               bool all_ones = sp != TESS_SPACING_FRACTIONAL_EVEN;
               for (unsigned e = 0; e < 3; ++e) {
                  n_out[e] = ref_seg_count(sp, f_out[e]);
                  if (n_out[e] != 1)
                     all_ones = false;
               }
               ni = ref_seg_count(sp, f_in);
               if (ni != 1)
                  all_ones = false;
               double area = 0.0;
               if (all_ones) {
                  area = 0.5;
               } else {
                  uint32_t min_in = sp == TESS_SPACING_FRACTIONAL_ODD ? 3 : 2;
                  if (ni < min_in) {
                     ni = min_in;
                     if (sp == TESS_SPACING_EQUAL)
                        f_in = (float)ni;
                  }
                  uint32_t rings = ni >> 1;
                  for (uint32_t r = 0; r < rings; ++r) {
                     uint32_t s_o_ring = ni - 2u * r;
                     uint32_t s_i = s_o_ring > 2 ? s_o_ring - 2 : 0;
                     for (uint32_t e = 0; e < 3; ++e) {
                        uint32_t s_o = r == 0 ? n_out[e] : s_o_ring;
                        uint32_t t_tot = s_o + s_i;
                        uint32_t chunks = DIV_ROUND_UP(t_tot, sh->ps);
                        for (uint32_t c = 0; c < chunks; ++c) {
                           uint32_t t0 = c * sh->ps;
                           uint32_t t1 = MIN2(t0 + sh->ps, t_tot);
                           uint32_t i0 = ref_stitch_i(s_o, s_i, t0);
                           uint32_t o0 = t0 - i0;
                           uint32_t i1 = ref_stitch_i(s_o, s_i, t1);
                           uint32_t o1 = t1 - i1;
                           uint32_t no = (o1 - o0) + 1;
                           uint32_t ni_v = (i1 - i0) + 1;
                           CHECK(no + ni_v <= sh->max_v);
                           CHECK(t1 - t0 <= sh->max_p);
                           float vbuf[64][2];
                           for (uint32_t l = 0; l < no; ++l)
                              ref_tri_ring_uv(sp, r, e, o0 + l, n_out, f_out,
                                              ni, f_in, vbuf[l]);
                           for (uint32_t l = 0; l < ni_v; ++l)
                              ref_tri_ring_uv(sp, r + 1, e, i0 + l, n_out, f_out,
                                              ni, f_in, vbuf[no + l]);
                           for (uint32_t p = 0; p < t1 - t0; ++p) {
                              uint32_t t = t0 + p;
                              uint32_t ia = ref_stitch_i(s_o, s_i, t);
                              uint32_t oi = (t - ia) - o0;
                              uint32_t ii = ia - i0;
                              bool step_i = ref_stitch_i(s_o, s_i, t + 1) != ia;
                              uint32_t a = step_i ? no + ii : oi;
                              uint32_t b = step_i ? oi : oi + 1;
                              uint32_t d = step_i ? no + ii + 1 : no + ii;
                              CHECK(a < no + ni_v && b < no + ni_v && d < no + ni_v);
                              double cr =
                                 ((double)vbuf[b][0] - vbuf[a][0]) *
                                    ((double)vbuf[d][1] - vbuf[a][1]) -
                                 ((double)vbuf[b][1] - vbuf[a][1]) *
                                    ((double)vbuf[d][0] - vbuf[a][0]);
                              /* Upstream D3D11/poly 16.16 fixed-point ring
                               * rounding can produce a 1-LSB (1/65536) corner
                               * sliver (~-2.5e-6) at frac==1; forbid any macro
                               * fold (e.g. -0.08 from float parity rounding). */
                              CHECK(cr >= -1e-5);
                              area += 0.5 * cr;
                           }
                        }
                     }
                  }
                  if (ni & 1) {
                     float cbuf[3][2];
                     for (uint32_t e = 0; e < 3; ++e)
                        ref_tri_ring_uv(sp, rings, e, 0, n_out, f_out, ni, f_in,
                                        cbuf[e]);
                     double cr =
                        ((double)cbuf[1][0] - cbuf[0][0]) *
                           ((double)cbuf[2][1] - cbuf[0][1]) -
                        ((double)cbuf[1][1] - cbuf[0][1]) *
                           ((double)cbuf[2][0] - cbuf[0][0]);
                     CHECK(cr >= -1e-6);
                     area += 0.5 * cr;
                  }
               }
               CHECK(fabs(area - 0.5) < 1e-4);
               verified++;
            }
         }
      }
   }
   printf("fused tessellation topology suite: 4160 stitch pairs + %u domain/spacing/shape area & winding checks passed\n",
          verified);
}

static void
fused_tess_astra_regression_suite(void)
{
   struct hk_tess_fused_options opts = {
      .max_payload_bytes = 1024,
   };
   opts.vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   opts.vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   opts.vertex.dynamic_stride = true;
   opts.vertex.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
   opts.vertex.stride[0] = 16;

   /* F1: Triangle TCS cross-invocation read of domain-inactive Outer[3]
    * written with dynamic ubo.level, synced via barrier, copied to Inner[0].
    * Interval analysis MUST NOT assume Outer[3] == 0 and underbound to 1 prim. */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "f1_outer3_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 2;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
      nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                  nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
      nir_store_output(
         &cb, nir_vec4(&cb, nir_imm_float(&cb, 1.0f), nir_imm_float(&cb, 1.0f),
                       nir_imm_float(&cb, 1.0f), dyn),
         nir_imm_int(&cb, 0), .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_barrier(&cb, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 1));
      nir_def *o3 = nir_load_output(
         &cb, 1, 32, nir_imm_int(&cb, 0), .component = 3,
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, o3, nir_imm_int(&cb, 0), .component = 0, .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 2,
                                       0, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                    &info, NULL));
      CHECK(!info.constant_tess_levels);
      /* Inner[0] can be up to 64 -> must report full 5955 tris, NOT 1! */
      CHECK(info.max_prims_per_patch == 5955u);
      CHECK(info.max_child_wgs_per_patch > 1u);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* F2: Triangle equal-spacing (3,3,3 / 3) inner ring corner must match
    * upstream poly/cl/tessellator.h bit-for-bit:
    * perp = (21845 * 0xaaaa + 0x8000) >> 16 = 14563, half = (14563 + 1) >> 1
    * = 7282, param = 65536 - 21845 = 43691, tmp = 43691 - 7282 = 36409,
    * with exact corner equality across edge 2 -> edge 0. */
   {
      const uint32_t n_out[3] = {3, 3, 3};
      const float f_out[3] = {3.0f, 3.0f, 3.0f};
      float uv0[2], uv2_end[2];
      ref_tri_ring_uv(TESS_SPACING_EQUAL, 1, 0, 0, n_out, f_out, 3, 3.0f, uv0);
      ref_tri_ring_uv(TESS_SPACING_EQUAL, 1, 2, 1, n_out, f_out, 3, 3.0f, uv2_end);
      CHECK(uv0[0] == 14563.0f / 65536.0f && uv0[1] == 36409.0f / 65536.0f);
      CHECK(f32_bits(uv0[0]) == f32_bits(uv2_end[0]) &&
            f32_bits(uv0[1]) == f32_bits(uv2_end[1]));
   }

   /* F3: Triangle equal-spacing (4,4,4 / 4) StitchTransition second triangle
    * must advance inside before outside: t=0->(O0,O1,I0), t=1->(I0,O1,I1),
    * t=2->(O1,O2,I1). */
   {
      CHECK(ref_stitch_i(4, 2, 0) == 0);
      CHECK(ref_stitch_i(4, 2, 1) == 0);
      CHECK(ref_stitch_i(4, 2, 2) == 1);
      CHECK(ref_stitch_i(4, 2, 3) == 1);
      CHECK(ref_stitch_i(4, 2, 4) == 1);
      CHECK(ref_stitch_i(4, 2, 5) == 2);
      CHECK(ref_stitch_i(4, 2, 6) == 2);
   }

   /* F4: When paired FS reads PrimitiveId, fused TES exports patch_id via flat
    * linked VARYING_SLOT_PRIMITIVE_ID and falls back when varying slots are full. */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(3, 0, 1.0f, 1.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 3, 0, 0, 0, false, false);
      struct hk_tess_fused_options pid_opts = opts;
      pid_opts.fs_reads_primitive_id = true;
      nir_shader *obj = NULL, *mesh = NULL;
      struct hk_tess_fused_info info = {0};
      const char *r = NULL;
      CHECK(hk_nir_tess_fused_lower(vs, tcs, tes, &pid_opts, &obj, &mesh, &info, &r));
      /* 4 (pos) + 1 (PrimitiveId) = 5 components -> planes = 8 -> V24/P20. */
      CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
      CHECK(mesh->info.outputs_written & VARYING_BIT_PRIMITIVE_ID);
      hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
      struct hk_apple9_graphics_key mkey = {
         .native_stage_generated = true,
         .native_tess_fused = true,
         .native_object = true,
         .native_object_hw_payload = true,
         .native_object_payload_B = info.payload_B,
         .fs_reads_primitive_id = true,
      };
      struct agx_shader_part mesh_part = {0};
      struct agx_apple9_mesh_body_info mesh_info = {0};
      CHECK(hk_apple9_compile_native_mesh_program(mesh, &mkey, &mesh_part,
                                                  &mesh_info, &r));
      CHECK(mesh_part.info.apple9_varyings.mask[VARYING_SLOT_PRIMITIVE_ID] == 1);
      CHECK(mesh_part.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID] == 1);
      free(mesh_part.binary);
      ralloc_free(obj);
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
   }

   /* F5: Fractional even/odd 16.16 quantization before parity rounding matches
    * upstream poly/cl/tessellator.h bit-for-bit: sub-1/65536 nextafterf values
    * (0x40000001 -> 2, 0x3f800001 -> 1) quantize cleanly without collapsed
    * frac==0 folds, while +1/65536 advances to 4 (even) and 3 (odd) on both
    * host reference and hk_nir_tess_fused_lower. */
   {
      float even_sub, odd_sub;
      uint32_t eb = 0x40000001u, ob = 0x3f800001u;
      memcpy(&even_sub, &eb, sizeof(even_sub));
      memcpy(&odd_sub, &ob, sizeof(odd_sub));
      const float even_step = 2.0f + 1.0f / 65536.0f;
      const float odd_step = 1.0f + 1.0f / 65536.0f;
      CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_EVEN, even_sub) == 2u);
      CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_EVEN, even_step) == 4u);
      CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_ODD, odd_sub) == 1u);
      CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_ODD, odd_step) == 3u);

      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs_e = make_fused_tcs(3, 0, even_sub, 2.0f, false, false, false,
                                         TESS_SPACING_FRACTIONAL_EVEN, false, false);
      nir_shader *tes_e = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                         TESS_SPACING_FRACTIONAL_EVEN, false, false,
                                         3, 0, 0, 0, false, false);
      struct hk_tess_fused_info info_e = {0};
      CHECK(hk_nir_tess_fused_lower(vs, tcs_e, tes_e, &opts, NULL, NULL,
                                    &info_e, NULL));
      /* 3 outer edges * 2 segs = 6 tris (sub-1/65536 quantizes to 2.0). */
      CHECK(info_e.max_prims_per_patch == 6u);

      nir_shader *tcs_es = make_fused_tcs(3, 0, even_step, 2.0f, false, false, false,
                                          TESS_SPACING_FRACTIONAL_EVEN, false, false);
      nir_shader *tes_es = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_FRACTIONAL_EVEN, false, false,
                                          3, 0, 0, 0, false, false);
      struct hk_tess_fused_info info_es = {0};
      CHECK(hk_nir_tess_fused_lower(vs, tcs_es, tes_es, &opts, NULL, NULL,
                                    &info_es, NULL));
      /* 3 outer edges * 4 segs = 12 tris at 2.0 + 1/65536. */
      CHECK(info_es.max_prims_per_patch == 12u);

      nir_shader *tcs_o = make_fused_tcs(3, 0, odd_step, 1.0f, false, false, false,
                                         TESS_SPACING_FRACTIONAL_ODD, false, false);
      nir_shader *tes_o = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                         TESS_SPACING_FRACTIONAL_ODD, false, false,
                                         3, 0, 0, 0, false, false);
      struct hk_tess_fused_info info_o = {0};
      CHECK(hk_nir_tess_fused_lower(vs, tcs_o, tes_o, &opts, NULL, NULL,
                                    &info_o, NULL));
      /* 3 outer * 3 + 3*(3-2) + 1 = 13 tris at 1.0 + 1/65536. */
      CHECK(info_o.max_prims_per_patch == 13u);
      ralloc_free(vs);
      ralloc_free(tcs_e);
      ralloc_free(tes_e);
      ralloc_free(tcs_es);
      ralloc_free(tes_es);
      ralloc_free(tcs_o);
      ralloc_free(tes_o);
      /* Unfused native TES must decline fs_reads_primitive_id so FS never binds
       * hardware rasterizer ID 0x60 instead of the input patch ID. */
      {
         nir_shader *tes_pos = make_fused_tes(TESS_PRIMITIVE_QUADS,
                                              TESS_SPACING_EQUAL, false, false,
                                              1, 0, 0, 0, false, false);
         struct hk_apple9_graphics_key ukey = {
            .native_stages = HK_APPLE9_NATIVE_TES_BIT,
            .fs_reads_primitive_id = true,
         };
         bool supported = true;
         const char *reason = NULL;
         CHECK(hk_apple9_native_tess_check(tes_pos, &ukey, &supported, &reason) ==
               VK_SUCCESS);
         CHECK(!supported && reason && strstr(reason, "PrimitiveID"));
         ralloc_free(tes_pos);
      }
   }

   /* F6: TCS control barrier and memory-only barrier on nir_var_shader_out
    * MUST translate to include nir_var_mem_task_payload (and nir_var_mem_global
    * after object lowering) for task-payload WAW ordering. */
   {
      nir_shader *vs = make_fused_vs(0, false);
      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "f6_barrier_tcs");
      cb.shader->info.io_lowered = true;
      cb.shader->info.tess.tcs_vertices_out = 2;
      nir_def *inv = nir_load_invocation_id(&cb);
      nir_def *pos = nir_load_per_vertex_input(
         &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
      nir_store_output(
         &cb, nir_imm_float(&cb, 1.0f), nir_imm_int(&cb, 0), .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PATCH0, .num_slots = 1});
      nir_store_output(
         &cb, nir_imm_vec4(&cb, 1.0f, 1.0f, 1.0f, 1.0f), nir_imm_int(&cb, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
      nir_store_output(
         &cb, nir_imm_vec2(&cb, 1.0f, 1.0f), nir_imm_int(&cb, 0),
         .write_mask = 0x3, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      /* Control barrier with output memory ordering. */
      nir_barrier(&cb, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
      /* Memory-only barrier with output memory ordering. */
      nir_barrier(&cb, .execution_scope = SCOPE_NONE,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_shader_out);
      nir_push_if(&cb, nir_ieq_imm(&cb, inv, 1));
      nir_store_output(
         &cb, nir_imm_float(&cb, 2.0f), nir_imm_int(&cb, 0), .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PATCH0, .num_slots = 1});
      nir_pop_if(&cb, NULL);
      nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 2,
                                       0, 1, 0, false, false);
      nir_shader *obj = NULL, *mesh = NULL;
      struct hk_tess_fused_info info = {0};
      CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, &obj, &mesh,
                                    &info, NULL));
      unsigned ctrl_payload_bars = 0, mem_only_payload_bars = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(obj)) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_barrier)
               continue;
            if (nir_intrinsic_memory_modes(intr) & nir_var_mem_task_payload) {
               if (nir_intrinsic_execution_scope(intr) == SCOPE_NONE)
                  mem_only_payload_bars++;
               else
                  ctrl_payload_bars++;
            }
         }
      }
      /* At least 3 control barriers with payload (init, split, final) + 1 mem-only. */
      CHECK(ctrl_payload_bars >= 3u && mem_only_payload_bars >= 1u);

      /* Lower through native object: task_payload barriers become mem_global. */
      hk_apple9_lower_prerast_sysvals(obj, MESA_SHADER_COMPUTE);
      struct agx_apple9_mesh_key ok_key = AGX_APPLE9_MESH_KEY;
      ok_key.physical_threads = AGX_APPLE9_NATIVE_SERIAL_THREADS;
      struct agx_apple9_native_object_options oopts = {
         .payload_stride_B = info.payload_B,
         .hw_payload = true,
      };
      struct agx_apple9_mesh_body_info oinfo = {0};
      const char *r = NULL;
      CHECK(agx_nir_lower_apple9_native_object_ex(obj, true, &ok_key, &oopts,
                                                  &oinfo, &r));
      unsigned global_ctrl = 0, global_mem_only = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(obj)) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_barrier)
               continue;
            if (nir_intrinsic_memory_modes(intr) & nir_var_mem_global) {
               if (nir_intrinsic_execution_scope(intr) == SCOPE_NONE)
                  global_mem_only++;
               else
                  global_ctrl++;
            }
         }
      }
      CHECK(global_ctrl >= 3u && global_mem_only >= 1u);
      ralloc_free(obj);
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(cb.shader);
      ralloc_free(tes);
   }

   /* F7 / M3: Dynamic SetMeshOutputsEXT(ubo.v, 1) on T32 V=16 P=1 comps=8
    * (u0_b=24, stride=52) and V=32 P=32 comps=4 with constant P=1 (fixed
    * idx_rows=4 -> u0_b=48, stride=52) emits constant safe U0_B into
    * MESH_CONTROL for every non-empty WG, eliminating runtime stride holes. */
   {
      const struct {
         unsigned max_v, max_p, extra_vec4;
         uint32_t expected_u0_b;
      } f7_cases[] = {
         {16, 1, 1, 24},
         {32, 32, 0, 48},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(f7_cases); ++i) {
         nir_builder mb = nir_builder_init_simple_shader(
            MESA_SHADER_MESH, &agx_nir_options, "f7_dyn_mesh");
         mb.shader->info.io_lowered = true;
         mb.shader->info.workgroup_size[0] = 32;
         mb.shader->info.workgroup_size[1] = mb.shader->info.workgroup_size[2] = 1;
         mb.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
         mb.shader->info.mesh.max_vertices_out = f7_cases[i].max_v;
         mb.shader->info.mesh.max_primitives_out = f7_cases[i].max_p;
         nir_def *lane = nir_load_local_invocation_index(&mb);
         nir_def *dyn_v = nir_load_ubo(&mb, 1, 32, nir_imm_int(&mb, 0),
                                       nir_imm_int(&mb, 0), .align_mul = 4, .range = 16);
         nir_push_if(&mb, nir_ieq_imm(&mb, lane, 0));
         nir_set_vertex_and_primitive_count(&mb, dyn_v, nir_imm_int(&mb, 1),
                                            nir_imm_int(&mb, 0));
         nir_store_per_primitive_output(
            &mb, nir_imm_ivec3(&mb, 0, 1, 2), nir_imm_int(&mb, 0),
            nir_imm_int(&mb, 0), .write_mask = 7, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                             .num_slots = 1});
         nir_pop_if(&mb, NULL);
         nir_push_if(&mb, nir_ult_imm(&mb, lane, f7_cases[i].max_v));
         nir_store_per_vertex_output(
            &mb, nir_imm_vec4(&mb, 0, 0, 0, 1), lane, nir_imm_int(&mb, 0),
            .write_mask = 0xf, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         if (f7_cases[i].extra_vec4) {
            nir_store_per_vertex_output(
               &mb, nir_imm_vec4(&mb, 1, 2, 3, 4), lane, nir_imm_int(&mb, 0),
               .write_mask = 0xf, .src_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         }
         nir_pop_if(&mb, NULL);
         nir_shader_gather_info(mb.shader, nir_shader_get_entrypoint(mb.shader));

         struct agx_apple9_native_mesh_options mopts = {
            .physical_threads = 32,
            .packed_index_publication = true,
            .publication_rounds = true,
            .attributes = true,
         };
         struct agx_apple9_mesh_body_info body = {0};
         const char *r = NULL;
         CHECK(agx_nir_lower_apple9_native_mesh(mb.shader, true, &mopts, &body, &r));
         bool found_ctrl = false;
         nir_foreach_block(blk, nir_shader_get_entrypoint(mb.shader)) {
            nir_foreach_instr(instr, blk) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_mesh_control_apple9_agx)
                  continue;
               found_ctrl = true;
               /* Second operand (U0_B) must be u2u16(bcsel(v != 0, safe_u0_b, 0)). */
               nir_instr *u0_ins = nir_def_instr(intr->src[1].ssa);
               CHECK(u0_ins->type == nir_instr_type_alu);
               nir_alu_instr *u2u = nir_instr_as_alu(u0_ins);
               CHECK(u2u->op == nir_op_u2u16);
               nir_instr *sel_ins = nir_def_instr(u2u->src[0].src.ssa);
               CHECK(sel_ins->type == nir_instr_type_alu);
               nir_alu_instr *sel = nir_instr_as_alu(sel_ins);
               CHECK(sel->op == nir_op_bcsel);
               CHECK(nir_src_is_const(sel->src[1].src) &&
                     nir_src_as_uint(sel->src[1].src) == f7_cases[i].expected_u0_b);
               CHECK(nir_src_is_const(sel->src[2].src) &&
                     nir_src_as_uint(sel->src[2].src) == 0u);
            }
         }
         CHECK(found_ctrl);
         ralloc_free(mb.shader);
      }
   }

   /* F8: fround_even interval analysis must be independent of host fegetround(). */
   {
      const int modes[4] = {
         FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO,
      };
      for (unsigned mi = 0; mi < ARRAY_SIZE(modes); ++mi) {
         fesetround(modes[mi]);
         nir_shader *vs = make_fused_vs(0, false);
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "f8_roundeven_tcs");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = 1;
         nir_def *inv = nir_load_invocation_id(&cb);
         nir_def *pos = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                     nir_imm_int(&cb, 0), .align_mul = 4, .range = 16);
         nir_def *lvl = nir_fround_even(
            &cb, nir_fmin(&cb, dyn, nir_imm_float(&cb, 3.75f)));
         nir_store_output(
            &cb, nir_vec4(&cb, lvl, lvl, lvl, lvl), nir_imm_int(&cb, 0),
            .write_mask = 0xf, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(
            &cb, nir_vec2(&cb, lvl, lvl), nir_imm_int(&cb, 0),
            .write_mask = 0x3, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS,
                                          TESS_SPACING_EQUAL, false, false, 1,
                                          0, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         CHECK(hk_nir_tess_fused_lower(vs, cb.shader, tes, &opts, NULL, NULL,
                                       &info, NULL));
         /* roundEven(3.75) == 4 -> equal-spacing 4x4 quad = 32 tris in ALL modes! */
         CHECK(info.max_prims_per_patch == 32u);
         ralloc_free(vs);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }
      fesetround(FE_TONEAREST);
   }

   printf("fused tessellation Astra regression suite: F1..F8 + M3 safe-U0_B checks passed\n");
}

/* Simulate the unified bin_desc + large_wg_end multi-patch protocol emitted
 * by build_object_shader_mp() + build_mesh_shader_mp() for an arbitrary K-patch
 * OBJECT WG containing mixed culled (tess=0), small-packed (tess=1..2), and
 * large multi-WG fan-out (tess=8..64) patches, plus partial tail WGs and
 * non-zero first_patch. */
static void
sim_multipatch_object_wg(enum tess_primitive_mode domain,
                         enum gl_tess_spacing spacing,
                         uint32_t max_v, uint32_t max_p, uint32_t tile_w,
                         uint32_t tile_h, uint32_t p_strip, uint32_t K,
                         uint32_t first_patch, uint32_t patch_count,
                         uint32_t wg_idx, const float *levels_outer,
                         const float *levels_inner)
{
   uint32_t small_mask = 0, large_mask = 0;
   uint32_t pv[16] = {0}, pp[16] = {0}, pwgs[16] = {0}, pid[16] = {0};
   uint32_t expected_surviving = 0, expected_total_v = 0, expected_total_p = 0;

   for (uint32_t i = 0; i < K; ++i) {
      uint32_t local_p = wg_idx * K + i;
      if (local_p >= patch_count)
         continue;
      pid[i] = first_patch + local_p;
      float o0 = levels_outer[i * 4 + 0];
      float o1 = levels_outer[i * 4 + 1];
      float o2 = levels_outer[i * 4 + 2];
      float o3 = levels_outer[i * 4 + 3];
      float i0 = levels_inner[i * 2 + 0];

      bool culled = domain == TESS_PRIMITIVE_QUADS
                       ? (!(o0 > 0.0f) || !(o1 > 0.0f) || !(o2 > 0.0f) || !(o3 > 0.0f))
                       : (!(o0 > 0.0f) || !(o1 > 0.0f) || !(o2 > 0.0f));
      if (culled)
         continue;

      uint32_t n_v = 0, n_p = 0, wgs = 0;
      if (domain == TESS_PRIMITIVE_QUADS) {
         uint32_t no[4];
         for (unsigned e = 0; e < 4; ++e) {
            float c = fminf(fmaxf(levels_outer[i * 4 + e], 1.0f), 64.0f);
            no[e] = ref_seg_count(spacing, c);
         }
         uint32_t ni[2];
         for (unsigned e = 0; e < 2; ++e) {
            float c = fminf(fmaxf(levels_inner[i * 2 + e], 1.0f), 64.0f);
            ni[e] = ref_seg_count(spacing, c);
            if (spacing == TESS_SPACING_FRACTIONAL_ODD && ni[e] == 1 &&
                (no[0] > 1 || no[1] > 1 || no[2] > 1 || no[3] > 1))
               ni[e] = 3;
         }
         uint32_t outer_v = no[0] + no[1] + no[2] + no[3];
         uint32_t inner_v = ni[0] >= 2 && ni[1] >= 2 ? (ni[0] - 1) * (ni[1] - 1) : 0;
         n_v = outer_v + inner_v;
         uint32_t outer_p = (no[0] + (ni[1] >= 2 ? ni[1] - 2 : 0)) +
                            (no[1] + (ni[0] >= 2 ? ni[0] - 2 : 0)) +
                            (no[2] + (ni[1] >= 2 ? ni[1] - 2 : 0)) +
                            (no[3] + (ni[0] >= 2 ? ni[0] - 2 : 0));
         uint32_t inner_p =
            ni[0] >= 2 && ni[1] >= 2 ? 2u * (ni[0] - 2) * (ni[1] - 2) : 0;
         n_p = outer_p + inner_p;
         if (n_v <= max_v && n_p <= max_p) {
            wgs = 1;
         } else {
            uint32_t iw = ni[0] >= 2 ? ni[0] - 2 : 0;
            uint32_t ih = ni[1] >= 2 ? ni[1] - 2 : 0;
            uint32_t tiles = (iw && ih) ? ((iw + tile_w - 1) / tile_w) *
                                             ((ih + tile_h - 1) / tile_h)
                                        : 0;
            uint32_t s_wgs = 0;
            uint32_t in_s[4] = {ih, iw, ih, iw};
            for (unsigned e = 0; e < 4; ++e)
               s_wgs += (no[e] + in_s[e] + p_strip - 1) / p_strip;
            wgs = tiles + s_wgs;
         }
      } else {
         uint32_t no[3];
         for (unsigned e = 0; e < 3; ++e) {
            float c = fminf(fmaxf(levels_outer[i * 4 + e], 1.0f), 64.0f);
            no[e] = ref_seg_count(spacing, c);
         }
         float ci = fminf(fmaxf(i0, 1.0f), 64.0f);
         uint32_t ni = ref_seg_count(spacing, ci);
         if (spacing == TESS_SPACING_FRACTIONAL_ODD && ni == 1 &&
             (no[0] > 1 || no[1] > 1 || no[2] > 1))
            ni = 3;
         if (ni <= 1) {
            n_v = 3;
            n_p = 1;
         } else {
            uint32_t rings = ni / 2;
            uint32_t center_v = (ni & 1) == 0 ? 1u : 0u;
            uint32_t inner_v = 0, inner_p = 0;
            for (uint32_t r = 1; r < rings; ++r) {
               uint32_t sr = ni - 2 * r;
               inner_v += 3 * sr;
               inner_p += 3 * (sr + (sr - 2));
            }
            if (ni & 1) {
               uint32_t sr = ni - 2 * rings;
               inner_v += 3 * sr;
               inner_p += 1;
            } else {
               inner_v += center_v;
               inner_p += 3 * 2;
            }
            uint32_t outer_v = no[0] + no[1] + no[2];
            uint32_t outer_p = (no[0] + ni - 2) + (no[1] + ni - 2) + (no[2] + ni - 2);
            n_v = outer_v + inner_v;
            n_p = outer_p + inner_p;
         }
         if (n_v <= max_v && n_p <= max_p) {
            wgs = 1;
         } else {
            uint32_t s_wgs = 0;
            for (unsigned e = 0; e < 3; ++e)
               s_wgs += (no[e] + (ni >= 2 ? ni - 2 : 0) + p_strip - 1) / p_strip;
            uint32_t rings = ni / 2;
            for (uint32_t r = 1; r < rings; ++r) {
               uint32_t sr = ni - 2 * r;
               s_wgs += 3u * ((sr + (sr - 2) + p_strip - 1) / p_strip);
            }
            if ((ni & 1) || ni >= 2)
               s_wgs += 1;
            wgs = s_wgs;
         }
      }

      pv[i] = n_v;
      pp[i] = n_p;
      pwgs[i] = wgs;
      expected_surviving++;
      expected_total_v += n_v;
      expected_total_p += n_p;
      if (wgs == 1 && n_v <= max_v && n_p <= max_p)
         small_mask |= (1u << i);
      else
         large_mask |= (1u << i);
   }

   /* 2-way compaction: small in [0..n_small-1], large in [n_small..n_small+n_large-1]. */
   uint32_t n_small = util_bitcount(small_mask);
   uint32_t n_large = util_bitcount(large_mask);
   uint32_t slot_orig[16] = {0};
   uint32_t s_v[16] = {0}, s_p[16] = {0}, l_wgs[16] = {0};
   uint32_t si = 0, li = 0;
   for (uint32_t i = 0; i < K; ++i) {
      if (small_mask & (1u << i)) {
         slot_orig[si] = i;
         s_v[si] = pv[i];
         s_p[si] = pp[i];
         si++;
      } else if (large_mask & (1u << i)) {
         slot_orig[n_small + li] = i;
         l_wgs[li] = pwgs[i];
         li++;
      }
   }

   /* Greedy bin packing of small patches into V<=max_v, P<=max_p child WGs. */
   uint16_t bin_base[16] = {0}, bin_cnt[16] = {0};
   uint32_t small_wgs = 0;
   uint32_t cur_base = 0, cur_cnt = 0, cur_v = 0, cur_p = 0;
   for (uint32_t j = 0; j < n_small; ++j) {
      if (cur_cnt > 0 && (cur_v + s_v[j] > max_v || cur_p + s_p[j] > max_p)) {
         bin_base[small_wgs] = cur_base;
         bin_cnt[small_wgs] = cur_cnt;
         small_wgs++;
         cur_base = j;
         cur_cnt = 0;
         cur_v = 0;
         cur_p = 0;
      }
      cur_cnt++;
      cur_v += s_v[j];
      cur_p += s_p[j];
   }
   if (cur_cnt > 0) {
      bin_base[small_wgs] = cur_base;
      bin_cnt[small_wgs] = cur_cnt;
      small_wgs++;
   }

   /* Large multi-WG prefix sum. */
   uint32_t large_end[16] = {0};
   uint32_t large_wgs = 0;
   for (uint32_t j = 0; j < n_large; ++j) {
      large_wgs += l_wgs[j];
      large_end[j] = large_wgs;
   }
   uint32_t total_child_wgs = small_wgs + large_wgs;

   if (expected_surviving == 0) {
      CHECK(total_child_wgs == 0);
      return;
   }

   /* Simulate every child MESH WG [0..total_child_wgs-1] and verify exact
    * coverage of every surviving patch, PrimitiveID, and V/P budget. */
   uint32_t seen_patches[16] = {0};
   uint32_t obs_v = 0, obs_p = 0;
   for (uint32_t cwg = 0; cwg < total_child_wgs; ++cwg) {
      if (cwg < small_wgs) {
         uint32_t base = bin_base[cwg];
         uint32_t cnt = bin_cnt[cwg];
         CHECK(cnt >= 1 && base + cnt <= n_small);
         uint32_t v_sum = 0, p_sum = 0;
         for (uint32_t lp = 0; lp < cnt; ++lp) {
            uint32_t orig = slot_orig[base + lp];
            CHECK(pid[orig] == first_patch + wg_idx * K + orig);
            seen_patches[orig]++;
            v_sum += pv[orig];
            p_sum += pp[orig];
         }
         CHECK(v_sum <= max_v && p_sum <= max_p);
         obs_v += v_sum;
         obs_p += p_sum;
      } else {
         uint32_t l_idx = cwg - small_wgs;
         uint32_t li_found = 0;
         while (li_found < n_large && l_idx >= large_end[li_found])
            li_found++;
         CHECK(li_found < n_large);
         uint32_t prev_end = li_found ? large_end[li_found - 1] : 0;
         uint32_t local_wg = l_idx - prev_end;
         uint32_t orig = slot_orig[n_small + li_found];
         CHECK(local_wg < pwgs[orig]);
         CHECK(pid[orig] == first_patch + wg_idx * K + orig);
         if (local_wg == 0) {
            seen_patches[orig]++;
            obs_v += pv[orig];
            obs_p += pp[orig];
         }
      }
   }

   uint32_t obs_surviving = 0;
   for (uint32_t i = 0; i < K; ++i) {
      if ((small_mask | large_mask) & (1u << i)) {
         CHECK(seen_patches[i] == 1);
         obs_surviving++;
      } else {
         CHECK(seen_patches[i] == 0);
      }
   }
   CHECK(obs_surviving == expected_surviving);
   CHECK(obs_v == expected_total_v);
   CHECK(obs_p == expected_total_p);
}

static void
fused_tess_multipatch_suite(void)
{
   struct hk_tess_fused_options base_opts = {
      .max_payload_bytes = 1024,
   };
   base_opts.vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   base_opts.vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   base_opts.vertex.dynamic_stride = true;
   base_opts.vertex.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;

   /* 1. Tier 2 Passthrough-TCS K=16 across all 4 meshlet shapes:
    *    planes <= 4 -> V32/P42, 8..32 -> V24/P20, 36..68 -> V16/P18,
    *    72..128 -> V8/P6, plus full OBJECT + MESH AGX backend compilation. */
   {
      const struct {
         unsigned vec4_vars;
         unsigned exp_v, exp_p;
      } shape_cases[] = {
         {0, 32, 42},  /* 4 planes */
         {4, 24, 20},  /* 20 planes -> golden V24/P20 */
         {7, 24, 20},  /* 32 planes -> golden V24/P20 */
         {8, 16, 18},  /* 36 planes -> V16/P18 */
         {16, 16, 18}, /* 68 planes -> V16/P18 */
         {17, 8, 6},   /* 72 planes -> V8/P6 */
      };
      for (unsigned i = 0; i < ARRAY_SIZE(shape_cases); ++i) {
         unsigned nv = shape_cases[i].vec4_vars;
         nir_shader *vs = make_fused_vs(nv, false);
         nir_shader *tcs = make_fused_tcs(3, nv, 2.0f, 2.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          nv, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 2);
         CHECK(info.patches_per_group == 16);
         CHECK(info.payload_B == 720);
         CHECK(info.mesh_max_vertices == shape_cases[i].exp_v);
         CHECK(info.mesh_max_primitives == shape_cases[i].exp_p);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
      }
   }

   /* 2. Multi-invocation Tier 2 TCS (writes gl_TessLevelOuter[inv] without
    *    if (inv == 0)): still qualifies for Tier 2, capped cleanly at
    *    K = MIN2(16, 32 / out_v). */
   {
      for (unsigned out_v = 3; out_v <= 4; ++out_v) {
         nir_shader *vs = make_fused_vs(2, false);
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "mp_multi_inv_tcs");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = out_v;
         nir_def *inv = nir_load_invocation_id(&cb);
         for (unsigned s = 0; s < 3; ++s) {
            unsigned loc = s == 0 ? VARYING_SLOT_POS : VARYING_SLOT_VAR0 + s - 1;
            nir_def *v = nir_load_per_vertex_input(
               &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
               .io_semantics = {.location = loc, .num_slots = 1});
            nir_store_per_vertex_output(
               &cb, v, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
               .src_type = nir_type_float32,
               .io_semantics = {.location = loc, .num_slots = 1});
         }
         /* Dynamic tess level written per invocation (no if (inv == 0) guard). */
         nir_def *dyn = nir_load_ubo(&cb, 1, 32, nir_imm_int(&cb, 0),
                                     nir_imul_imm(&cb, inv, 4), .align_mul = 4, .range = 32);
         nir_store_output(
            &cb, nir_vec4(&cb, dyn, dyn, dyn, dyn), nir_imm_int(&cb, 0),
            .write_mask = 0xf, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(
            &cb, nir_vec2(&cb, dyn, dyn), nir_imm_int(&cb, 0),
            .write_mask = 0x3, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));
         nir_shader *tes = make_fused_tes(
            out_v == 4 ? TESS_PRIMITIVE_QUADS : TESS_PRIMITIVE_TRIANGLES,
            TESS_SPACING_EQUAL, false, false, out_v, 2, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, cb.shader, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 2);
         CHECK(info.patches_per_group == 32u / out_v);
         ralloc_free(vs);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }
   }

   /* 3. Negative Tier 2 rejection tests: EVERY non-passthrough / ineligible
    *    variant MUST select Tier 1 (K=2..8) and NEVER wrongly match Tier 2! */
   {
      /* 3a: Modified CP output (pos + 1.0) */
      {
         nir_shader *vs = make_fused_vs(1, false);
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "t1_mod_cp");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = 3;
         nir_def *inv = nir_load_invocation_id(&cb);
         nir_def *pos = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *v0 = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, nir_fadd(&cb, pos, nir_imm_vec4(&cb, 1, 0, 0, 0)), inv,
            nir_imm_int(&cb, 0), .write_mask = 0xf, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, v0, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
         nir_store_output(&cb, nir_imm_vec4(&cb, 2, 2, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(&cb, nir_imm_vec2(&cb, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0x3, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_pop_if(&cb, NULL);
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, cb.shader, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1);
         CHECK(info.patches_per_group >= 4 && info.patches_per_group <= 8);
         CHECK(info.payload_B <= 1024);
         ralloc_free(vs);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }

      /* 3b: Cross-CP read (out[inv].pos = in[0].pos) */
      {
         nir_shader *vs = make_fused_vs(1, false);
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "t1_cross_cp");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = 3;
         nir_def *inv = nir_load_invocation_id(&cb);
         nir_def *pos0 = nir_load_per_vertex_input(
            &cb, 4, 32, nir_imm_int(&cb, 0), nir_imm_int(&cb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *v0 = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, pos0, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, v0, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
         nir_store_output(&cb, nir_imm_vec4(&cb, 2, 2, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(&cb, nir_imm_vec2(&cb, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0x3, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_pop_if(&cb, NULL);
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, cb.shader, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vs);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }

      /* 3c: Conditional CP write (inside if) */
      {
         nir_shader *vs = make_fused_vs(1, false);
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "t1_cond_cp");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = 3;
         nir_def *inv = nir_load_invocation_id(&cb);
         nir_def *pos = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *v0 = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_push_if(&cb, nir_ult_imm(&cb, inv, 3));
         nir_store_per_vertex_output(
            &cb, pos, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, v0, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_pop_if(&cb, NULL);
         nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
         nir_store_output(&cb, nir_imm_vec4(&cb, 2, 2, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(&cb, nir_imm_vec2(&cb, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0x3, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_pop_if(&cb, NULL);
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, cb.shader, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vs);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }

      /* 3d: Patch-constant output + TES reading patch constant */
      {
         nir_shader *vs = make_fused_vs(1, false);
         nir_shader *tcs = make_fused_tcs(3, 1, 2.0f, 2.0f, true, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, true, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
      }

      /* 3e: TES reading raw TessLevelOuter */
      {
         nir_shader *vs = make_fused_vs(1, false);
         nir_shader *tcs = make_fused_tcs(3, 1, 2.0f, 2.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, false, true);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
      }

      /* 3f: float_controls_execution_mode mismatch between VS/TCS and TES */
      {
         nir_shader *vs = make_fused_vs(1, false);
         vs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
         nir_shader *tcs = make_fused_tcs(3, 1, 2.0f, 2.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         tcs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          1, 0, 0, false, false);
         tes->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
      }

      /* 3g: Heavy VS with loop -> rejected from Tier 2, runs Tier 1 */
      {
         nir_builder vb = nir_builder_init_simple_shader(
            MESA_SHADER_VERTEX, &agx_nir_options, "t1_loop_vs");
         vb.shader->info.io_lowered = true;
         nir_def *pos = nir_load_input(
            &vb, 4, 32, nir_imm_int(&vb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
         nir_variable *acc = nir_local_variable_create(
            nir_shader_get_entrypoint(vb.shader), glsl_vec4_type(), "acc");
         nir_variable *it = nir_local_variable_create(
            nir_shader_get_entrypoint(vb.shader), glsl_uint_type(), "it");
         nir_store_var(&vb, acc, pos, 0xf);
         nir_store_var(&vb, it, nir_imm_int(&vb, 0), 1);
         nir_push_loop(&vb);
         {
            nir_def *i = nir_load_var(&vb, it);
            nir_push_if(&vb, nir_uge_imm(&vb, i, 4));
            nir_jump(&vb, nir_jump_break);
            nir_pop_if(&vb, NULL);
            nir_store_var(&vb, acc,
                          nir_fadd(&vb, nir_load_var(&vb, acc), pos), 0xf);
            nir_store_var(&vb, it, nir_iadd_imm(&vb, i, 1), 1);
         }
         nir_pop_loop(&vb, NULL);
         nir_store_output(&vb, nir_load_var(&vb, acc), nir_imm_int(&vb, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));
         nir_shader *tcs = make_fused_tcs(3, 0, 2.0f, 2.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          0, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vb.shader, tcs, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vb.shader);
         ralloc_free(tcs);
         ralloc_free(tes);
      }

      /* 3h: Heavy VS with >1 texture fetch + TCS with top-level barrier -> Tier 1 */
      {
         nir_builder vb = nir_builder_init_simple_shader(
            MESA_SHADER_VERTEX, &agx_nir_options, "t1_2tex_vs");
         vb.shader->info.io_lowered = true;
         nir_def *pos = nir_load_input(
            &vb, 4, 32, nir_imm_int(&vb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
         nir_def *acc = pos;
         for (unsigned ti = 0; ti < 2; ++ti) {
            nir_tex_instr *tex = nir_tex_instr_create(vb.shader, 2);
            tex->op = nir_texop_txl;
            tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
            tex->coord_components = 2;
            tex->dest_type = nir_type_float32;
            tex->texture_index = ti;
            tex->sampler_index = ti;
            tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                              nir_channels(&vb, pos, 3));
            tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                              nir_imm_float(&vb, 0.0f));
            nir_def_init(&tex->instr, &tex->def, 4, 32);
            nir_builder_instr_insert(&vb, &tex->instr);
            acc = nir_fadd(&vb, acc, &tex->def);
         }
         nir_store_output(&vb, acc, nir_imm_int(&vb, 0), .write_mask = 0xf,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_TESS_CTRL, &agx_nir_options, "t1_barrier_tcs");
         cb.shader->info.io_lowered = true;
         cb.shader->info.tess.tcs_vertices_out = 3;
         nir_def *inv = nir_load_invocation_id(&cb);
         nir_def *p = nir_load_per_vertex_input(
            &cb, 4, 32, inv, nir_imm_int(&cb, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_per_vertex_output(
            &cb, p, inv, nir_imm_int(&cb, 0), .write_mask = 0xf,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_barrier(&cb, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_shader_out);
         nir_push_if(&cb, nir_ieq_imm(&cb, inv, 0));
         nir_store_output(&cb, nir_imm_vec4(&cb, 2, 2, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER, .num_slots = 1});
         nir_store_output(&cb, nir_imm_vec2(&cb, 2, 2), nir_imm_int(&cb, 0),
                          .write_mask = 0x3, .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER, .num_slots = 1});
         nir_pop_if(&cb, NULL);
         nir_shader_gather_info(cb.shader, nir_shader_get_entrypoint(cb.shader));

         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          0, 0, 0, false, false);
         struct hk_tess_fused_info info = {0};
         compile_fused_tess_pair(vb.shader, cb.shader, tes, &base_opts, &info);
         CHECK(info.multipatch_tier == 1 && info.patches_per_group >= 4);
         ralloc_free(vb.shader);
         ralloc_free(cb.shader);
         ralloc_free(tes);
      }
   }

   /* 4. Mixed-LOD (tess=0 culled + tess=1..2 packed + tess=8..64 multi-WG),
    *    all-culled, partial tail WGs (patch_count=1,3,17,31), first_patch!=0
    *    protocol verification across Triangles & Quads and all 3 spacings. */
   {
      const enum tess_primitive_mode doms[2] = {
         TESS_PRIMITIVE_TRIANGLES, TESS_PRIMITIVE_QUADS
      };
      const enum gl_tess_spacing sps[3] = {
         TESS_SPACING_EQUAL, TESS_SPACING_FRACTIONAL_ODD,
         TESS_SPACING_FRACTIONAL_EVEN
      };
      /* 16 patches mixing tess=0 (culled), 1.0, 1.5, 2.0 (small-packed),
       * 8.0, 17.5, 64.0 (large multi-WG fan-out) inside ONE OBJECT WG! */
      const float mixed_vals[16] = {
         0.0f, 1.0f, 8.0f, 0.0f, 2.0f, 1.0f, 64.0f, 0.0f,
         1.5f, 0.0f, 17.5f, 1.0f, 0.0f, 2.0f, 0.0f, 12.0f,
      };
      float lo[16 * 4], li[16 * 2];
      for (unsigned i = 0; i < 16; ++i) {
         for (unsigned c = 0; c < 4; ++c)
            lo[i * 4 + c] = mixed_vals[i];
         for (unsigned c = 0; c < 2; ++c)
            li[i * 2 + c] = mixed_vals[i];
      }
      float zero_o[16 * 4] = {0}, zero_i[16 * 2] = {0};
      const uint32_t tail_counts[] = {1, 3, 17, 31, 32};
      const uint32_t first_patches[] = {0, 7, 13};
      const uint32_t ks[] = {4, 8, 16};
      for (unsigned d = 0; d < 2; ++d) {
         for (unsigned s = 0; s < 3; ++s) {
            for (unsigned ki = 0; ki < ARRAY_SIZE(ks); ++ki) {
               /* Full mixed WG on V=24, P=20 and V=16, P=18 */
               sim_multipatch_object_wg(doms[d], sps[s], 24, 20, 3, 3, 20,
                                        ks[ki], 5, 64, 0, lo, li);
               sim_multipatch_object_wg(doms[d], sps[s], 16, 18, 3, 3, 14,
                                        ks[ki], 5, 64, 0, lo, li);
               /* All-culled WG */
               sim_multipatch_object_wg(doms[d], sps[s], 24, 20, 3, 3, 20,
                                        ks[ki], 0, 64, 0, zero_o, zero_i);
               /* Partial tail WGs + first_patch != 0 */
               for (unsigned ti = 0; ti < ARRAY_SIZE(tail_counts); ++ti) {
                  uint32_t pc = tail_counts[ti];
                  uint32_t nwg = (pc + ks[ki] - 1) / ks[ki];
                  for (unsigned fp = 0; fp < ARRAY_SIZE(first_patches); ++fp) {
                     sim_multipatch_object_wg(doms[d], sps[s], 24, 20, 3, 3, 20,
                                              ks[ki], first_patches[fp], pc,
                                              nwg - 1, lo, li);
                  }
               }
            }
         }
      }
   }

   /* 5. PrimitiveID export, CW/CCW, all 3 spacings, Quads/Triangles, indexed
    *    (1/2/4 B IBO) + disable_multipatch=true deferred payload verification. */
   {
      const enum gl_tess_spacing sps[3] = {
         TESS_SPACING_EQUAL, TESS_SPACING_FRACTIONAL_ODD,
         TESS_SPACING_FRACTIONAL_EVEN
      };
      for (unsigned d = 0; d < 2; ++d) {
         enum tess_primitive_mode dom =
            d == 0 ? TESS_PRIMITIVE_TRIANGLES : TESS_PRIMITIVE_QUADS;
         unsigned cp = d == 0 ? 3 : 4;
         for (unsigned s = 0; s < 3; ++s) {
            for (unsigned ccw = 0; ccw < 2; ++ccw) {
               nir_shader *vs = make_fused_vs(3, false);
               nir_shader *tcs = make_fused_tcs(cp, 3, 2.5f, 2.5f, false, false,
                                                false, sps[s], false, false);
               nir_shader *tes = make_fused_tes(dom, sps[s], ccw, false, cp, 3,
                                                0, 0, false, false);
               struct hk_tess_fused_options opts = base_opts;
               opts.fs_reads_primitive_id = true;
               struct hk_tess_fused_info info = {0};
               compile_fused_tess_pair(vs, tcs, tes, &opts, &info);
               CHECK(info.multipatch_tier == 2 && info.patches_per_group == 16);
               CHECK(info.mesh_max_vertices == 24 && info.mesh_max_primitives == 20);
               ralloc_free(vs);
               ralloc_free(tcs);
               ralloc_free(tes);
            }
         }
      }

      /* Verify OBJECT defers all 0xe0/0xe1 store_task_payload writes inside
       * the survival branch (ZERO payload writes in the top-level block before
       * culling!) for both Tier 2 K=16 AND disable_multipatch K=1. */
      for (unsigned mp = 0; mp < 2; ++mp) {
         nir_shader *vs = make_fused_vs(2, false);
         nir_shader *tcs = make_fused_tcs(3, 2, 2.0f, 2.0f, false, false, false,
                                          TESS_SPACING_EQUAL, false, false);
         nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                          TESS_SPACING_EQUAL, false, false, 3,
                                          2, 0, 0, false, false);
         struct hk_tess_fused_options opts = base_opts;
         opts.disable_multipatch = (mp == 0);
         nir_shader *obj = NULL, *mesh = NULL;
         struct hk_tess_fused_info info = {0};
         CHECK(hk_nir_tess_fused_lower(vs, tcs, tes, &opts, &obj, &mesh, &info, NULL));
         if (mp == 0) {
            CHECK(info.multipatch_tier == 0 && info.patches_per_group == 1);
         } else {
            CHECK(info.multipatch_tier == 2 && info.patches_per_group == 16);
         }
         /* Count store_task_payload at top-level (must be 0: all guarded by
          * survival / total_wgs > 0 check!) and inside guarded branches (> 0). */
         nir_function_impl *oimpl = nir_shader_get_entrypoint(obj);
         unsigned top_pay_stores = 0, guarded_pay_stores = 0;
         nir_foreach_block(blk, oimpl) {
            bool top = blk->cf_node.parent == &oimpl->cf_node;
            nir_foreach_instr(instr, blk) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               if (nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_store_task_payload) {
                  if (top)
                     top_pay_stores++;
                  else
                     guarded_pay_stores++;
               }
            }
         }
         CHECK(top_pay_stores == 0 && guarded_pay_stores > 0);
         ralloc_free(obj);
         ralloc_free(mesh);
         ralloc_free(vs);
         ralloc_free(tcs);
         ralloc_free(tes);
      }
   }

   printf("fused tessellation multi-patch suite: Tier 2 K=16, Tier 1 K=2..8, 8 negative guards, mixed-LOD/partial-tail/PrimitiveID/deferred-payload passed\n");
}

int
main(void)
{
   setenv("HK_APPLE9_NATIVE_MESH", "1", 1);
   setenv("HK_APPLE9_NATIVE_MESH_PACKED_SHAPES", "T16V3P1", 1);
   glsl_type_singleton_init_or_ref();
   nir_shader *s = source(false, true, false);
   nir_shader *m = lower_immutable(s, true);
   if (getenv("TES_TEST_DUMP"))
      nir_print_shader(m, stderr);
   ralloc_free(m);
   ralloc_free(s);
   for (unsigned q = 0; q < 2; ++q)
      for (unsigned i = 0; i < 2; ++i)
         for (unsigned p = 0; p < 2; ++p)
            compile(q, i, p);
   for (unsigned count = 0; count <= 10; ++count)
      for (unsigned group = 0; group <= 4; ++group)
         for (unsigned lane = 0; lane <= 4; ++lane)
            for (unsigned ccw = 0; ccw < 2; ++ccw)
               bounds((struct constants){.count = count,
                                         .group = group,
                                         .lane = lane,
                                         .patch_size = 3,
                                         .ccw = ccw,
                                         .inputs = true,
                                         .partial = (group & 1),
                                         .quad = (count & 1)});
   for (unsigned lane = 0; lane < 4; ++lane)
      for (unsigned ccw = 0; ccw < 2; ++ccw) {
         bounds((struct constants){.count = UINT32_MAX,
                                   .group = UINT32_MAX / 3 - 1,
                                   .lane = lane,
                                   .patch_size = 32,
                                   .inputs = true,
                                   .ccw = ccw});
         bounds((struct constants){.count = UINT32_MAX,
                                   .group = UINT32_MAX,
                                   .lane = lane,
                                   .patch_size = 3,
                                   .inputs = true,
                                   .ccw = ccw});
      }
   for (unsigned size = 0; size <= 33; ++size)
      for (unsigned input = 0; input < 2; ++input)
         bounds((
            struct constants){.count = 3, .patch_size = size, .inputs = input});
   bounds((struct constants){.count = 6, .group = 2, .patch_size = 3});
   bounds((struct constants){.count = 6, .patch_size = 3, .y = 1});
   bounds((struct constants){.count = 6, .patch_size = 3, .z = 1});
   object_grid();
   rejections();
   levels();
   fused_tess_suite();
   fused_tess_vbo_pack_suite();
   fused_tess_interval_bounds_suite();
   fused_tess_topology_suite();
   fused_tess_astra_regression_suite();
   fused_tess_multipatch_suite();
   /* Consuming entrypoint lowers in place without a second clone and frees on reject. */
   {
      nir_shader *owned = source(false, false, false);
      const char *reason = NULL;
      nir_shader *mesh = hk_nir_tes_mesh_take(owned, &reason);
      CHECK(mesh == owned);
      CHECK(mesh->info.stage == MESA_SHADER_MESH);
      ralloc_free(mesh);

      nir_shader *bad = source(false, false, false);
      bad->info.writes_memory = true;
      CHECK(hk_nir_tes_mesh_take(bad, &reason) == NULL);
      CHECK(reason != NULL);
   }
   printf(
      "%u bounds/winding/partial-output cases, 15 immutable rejections, tess-level bounds, 8 native compilations, 10 object grid cases + object compilation passed\n",
      bounds_cases);
   glsl_type_singleton_decref();
   return 0;
}
