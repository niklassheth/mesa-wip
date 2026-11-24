/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
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
      unsigned quads = DIV_ROUND_UP(mesh_info.layout.max_vertices, 4u);
      unsigned planes = ALIGN_POT(mesh_info.layout.vertex_components, 4u);
      unsigned wg_rows = quads * planes + AGX_APPLE9_MESH_RING_HEADER_ROWS;
      CHECK(wg_rows != AGX_APPLE9_MESH_RING_EXACT_WRAP_ROWS);
      if (quads == 1 && planes == 28) {
         uint64_t ring_max = 0;
         CHECK(agx_apple9_mesh_ring_max_groups(&mesh_info.layout, &ring_max));
         CHECK(ring_max == UINT64_MAX);
      }
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

   /* 2. Adaptive meshlet tiers (V16/P18, V8/P6, V4/P2) + FS-linked varying DCE
    * + ClipDistance[2] + CullDistance[1] (Elden Ring Triangle shape) + raw
    * tess levels + patch constants. */
   {
      /* 8 comps (POS + 1 vec4) -> V16/P18. */
      nir_shader *vs = make_fused_vs(1, false);
      nir_shader *tcs = make_fused_tcs(3, 1, 3.0f, 3.0f, true, true, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_EQUAL, false, false, 3,
                                       1, 0, 0, true, true);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 16 && info.mesh_max_primitives == 18);
      CHECK(info.max_child_wgs_per_patch == 1);
      CHECK(info.max_prims_per_patch == 13);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }
   {
      /* 12 comps (POS + 2 vec4) -> V4/P2 (avoids wg_rows==32 TVB-grow divisor,
       * wg_rows=20 unbounded); 16 comps (POS + 3 vec4) -> V8/P6 (wg_rows=40);
       * with FS mask reading 0 vars -> V32/P42! */
      nir_shader *vs = make_fused_vs(0, false);
      nir_shader *tcs = make_fused_tcs(1, 0, 1.0f, 1.0f, false, false, false,
                                       TESS_SPACING_EQUAL, false, false);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                       false, false, 1, 2, 0, 0, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 4 && info.mesh_max_primitives == 2);
      CHECK(info.max_child_wgs_per_patch == 1 && info.max_prims_per_patch == 2);

      nir_shader *tes16 = make_fused_tes(TESS_PRIMITIVE_QUADS, TESS_SPACING_EQUAL,
                                         false, false, 1, 3, 0, 0, false, false);
      compile_fused_tess_pair(vs, tcs, tes16, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 8 && info.mesh_max_primitives == 6);
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
       * (4 + 16 + 2 = 22 comps -> V4/P2, wg_rows <= 40 unbounded!). */
      nir_shader *vs = make_fused_vs(4, true);
      nir_shader *tcs = make_fused_tcs(3, 4, 4.0f, 4.0f, true, false, true,
                                       TESS_SPACING_FRACTIONAL_ODD, false, true);
      nir_shader *tes = make_fused_tes(TESS_PRIMITIVE_TRIANGLES,
                                       TESS_SPACING_FRACTIONAL_ODD, false, true,
                                       3, 4, 2, 1, false, false);
      struct hk_tess_fused_info info = {0};
      compile_fused_tess_pair(vs, tcs, tes, &base_opts, &info);
      CHECK(info.mesh_max_vertices == 4 && info.mesh_max_primitives == 2);
      CHECK(!info.constant_tess_levels);
      CHECK(info.max_child_wgs_per_patch > 0 && info.max_child_wgs_per_patch <= 4096);
      ralloc_free(vs);
      ralloc_free(tcs);
      ralloc_free(tes);
      compiled++;
   }
   {
      /* Planes == 24 trap (21..24 live vertex components at V=4): native mesh
       * lowering must pad to 25 components (planes = 28, wg_rows = 36,
       * ring_max == UINT64_MAX) so zero shaders emit wg_rows == 32. */
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
         CHECK(info.mesh_max_vertices == 4 && info.mesh_max_primitives == 2);
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
         CHECK(body.layout.vertex_components == 25u);
         CHECK(part.info.apple9_varyings.count == 21u);
         CHECK(hk_apple9_native_shader_valid(&part.info, &body,
                                             HK_APPLE9_NATIVE_MESH));
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
ref_stitch_i(uint32_t n_out, uint32_t n_in_seg, uint32_t t)
{
   uint32_t total = n_out + n_in_seg;
   uint32_t k = n_in_seg >> 1;
   uint32_t half_t = (n_out >> 1) + k;
   uint32_t denom = half_t ? half_t : 1;
   uint32_t left = (t * k) / denom;
   uint32_t rt = t <= total ? total - t : 0;
   uint32_t right = n_in_seg - (rt * k) / denom;
   return t <= half_t ? left : (rt <= half_t ? right : k + 1);
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
   bool flip = 2u * k > n;
   uint32_t j = flip ? n - k : k;
   uint32_t q_half;
   if (j == 0) {
      q_half = 0;
   } else if (2u * j == n) {
      q_half = 32768u;
   } else if (spacing == TESS_SPACING_EQUAL) {
      q_half = j * ref_float_to_fxp(1.0f / (float)n);
   } else {
      bool odd = spacing == TESS_SPACING_FRACTIONAL_ODD;
      uint32_t fxp_tess = ref_float_to_fxp(f);
      uint32_t fxp_half = (fxp_tess + 1u) >> 1;
      if (odd || fxp_half == 32768u)
         fxp_half += 32768u;
      uint32_t fxp_floor = fxp_half & ~0xffffu;
      uint32_t fxp_ceil = (fxp_half + 0xffffu) & ~0xffffu;
      uint32_t frac = fxp_half - fxp_floor;
      uint32_t floor_int = fxp_floor >> 16;
      uint32_t ceil_int = fxp_ceil >> 16;
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
      uint32_t idx_floor = j > split ? j - 1u : j;
      uint32_t loc_floor = idx_floor * inv_floor;
      uint32_t loc_ceil = j * inv_ceil;
      q_half = (loc_floor * (65536u - frac) + loc_ceil * frac + 32768u) >> 16;
   }
   return flip ? 65536u - q_half : q_half;
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
   uint32_t perp_fxp = (perp_raw * 0xaaaau + 32768u) >> 16;
   uint32_t one_m_2p_fxp = 65536u - 2u * perp_fxp;
   uint32_t u_fxp, v_fxp;
   if (k == 0 || k == s_r) {
      uint32_t c_id = k == 0 ? e : (e == 2 ? 0 : e + 1);
      u_fxp = c_id == 2 ? one_m_2p_fxp : perp_fxp;
      v_fxp = c_id == 0 ? one_m_2p_fxp : perp_fxp;
   } else {
      uint32_t q = e == 1 ? r + k : (ni - r - k);
      uint32_t param_fxp = ref_eval_1d_fxp(spacing, ni, fi, q);
      uint32_t tmp_fxp = param_fxp - ((perp_fxp + 1u) >> 1);
      u_fxp = e == 0 ? perp_fxp : tmp_fxp;
      v_fxp = e == 0 ? tmp_fxp : (e == 1 ? perp_fxp : 65536u - tmp_fxp - perp_fxp);
   }
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
   bool odd = spacing == TESS_SPACING_FRACTIONAL_ODD;
   uint32_t fxp = ref_float_to_fxp(clamped);
   uint32_t half = (fxp + (odd ? 65537u : 1u)) >> 1;
   uint32_t ceil_half = (half + 0xffffu) & ~0xffffu;
   uint32_t r = ceil_half >> 15;
   return odd ? r - 1u : r;
}

static void
fused_tess_topology_suite(void)
{
   /* 0. Spot-check bit-exact match with poly_tessellator.h PlacePointIn1D &
    * NumPointsForTessFactor across sub-1/65536 float boundaries. */
   CHECK(ref_eval_1d_fxp(TESS_SPACING_EQUAL, 7, 7.0f, 2) == 18724u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_EVEN, 4, 2.5f, 1) == 28672u);
   CHECK(ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_EVEN, 4, 2.5f, 3) == 65536u - 28672u);
   CHECK(ref_seg_count(TESS_SPACING_FRACTIONAL_ODD, 3.000000476837158f) == 3u);
   {
      uint32_t prev = 0;
      for (uint32_t k = 0; k <= 3; ++k) {
         uint32_t q = ref_eval_1d_fxp(TESS_SPACING_FRACTIONAL_ODD, 3,
                                      3.000000476837158f, k);
         CHECK(q >= prev);
         prev = q;
      }
      CHECK(prev == 65536u);
   }

   /* 1. Exhaustively verify stitcher monotonicity, endpoints, and symmetry
    * over all 64 * 65 = 4,160 (n_out, n_in_seg) pairs. */
   for (uint32_t no = 1; no <= 64; ++no) {
      for (uint32_t ni = 0; ni <= 64; ++ni) {
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
                                 CHECK(cr >= -1e-6);
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
                              CHECK(cr >= -1e-6);
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
   fused_tess_topology_suite();
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
