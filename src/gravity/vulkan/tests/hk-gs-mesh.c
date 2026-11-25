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
#include "hk_cmd_buffer.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "hk_shader_priv.h"
#include "libagx_shaders.h"
#include "poly/geometry.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

#include "hk-test-usc.h"

static unsigned allocations, live_bos, precomp_calls;

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
void __wrap_agx_bo_cache_evict_all(struct agx_device *);
VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int,
                            const char *, ...);
void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *, struct agx_grid,
                                enum agx_barrier, enum libagx_program, void *,
                                size_t);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   ++allocations;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = (flags & AGX_BO_CONTEXT)
                     ? AGX_APPLE9_RENDER_CONTEXT_BASE +
                          (uint64_t)allocations * 0x100000ull
                     : dev->shader_base + (uint64_t)allocations * 0x100000ull;
   ++live_bos;
   return bo;
}

void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   if (bo)
      ++bo->refcnt;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   CHECK(bo->refcnt > 0);
   if (--bo->refcnt)
      return;
   CHECK(live_bos > 0);
   --live_bos;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

void
__wrap_agx_bo_cache_evict_all(UNUSED struct agx_device *dev)
{
}

VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                           enum agx_barrier barrier, enum libagx_program idx,
                           void *data, size_t size)
{
   ++precomp_calls;
   if (barrier & AGX_PREGFX) {
      CHECK(hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true,
                                         HK_SYNC_ALL) != NULL);
   }
}

/* =========================================================================
 * Existing unfused hk_nir_gs_mesh tests
 * ========================================================================= */

static nir_shader *
source(unsigned vertices, unsigned ends, bool inputs)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS mesh test");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = vertices;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   nir_def *count =
      nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                   .align_mul = 4, .range = 4);
   for (unsigned i = 0; i < vertices; ++i) {
      nir_push_if(&b, nir_ult(&b, nir_imm_int(&b, i), count));
      nir_def *pos =
         inputs
            ? nir_load_per_vertex_input(
                 &b, 4, 32, nir_imm_int(&b, i % 3), nir_imm_int(&b, 0),
                 .dest_type = nir_type_float32,
                 .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1})
            : nir_imm_vec4(&b, i, i + 1, 0.5, 1);
      for (unsigned c = 0; c < 4; ++c)
         nir_store_output(
            &b, nir_channel(&b, pos, c), nir_imm_int(&b, 0), .component = c,
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_emit_vertex(&b, 0);
      nir_pop_if(&b, NULL);
      if (ends & BITFIELD_BIT(i))
         nir_end_primitive(&b, 0);
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
   nir_shader *mesh = hk_nir_gs_mesh(src, &reason);
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

struct constants {
   unsigned count, lane;
   bool first;
};

static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct constants *c = data;
   if (intr->intrinsic != nir_intrinsic_load_ubo &&
       intr->intrinsic != nir_intrinsic_load_local_invocation_index &&
       intr->intrinsic != nir_intrinsic_load_provoking_last)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(
      &intr->def,
      nir_imm_int(
         b, intr->intrinsic == nir_intrinsic_load_ubo ? c->count :
            intr->intrinsic == nir_intrinsic_load_provoking_last ? !c->first : c->lane));
   return true;
}

static void
emission_mode(unsigned vertices, unsigned count, unsigned ends, unsigned lane, bool first)
{
   nir_shader *src = source(MAX2(vertices, count), ends, false);
   src->info.gs.vertices_out = vertices;
   nir_shader *mesh = lower_immutable(src, true);
   CHECK(mesh->info.mesh.max_vertices_out == vertices);
   CHECK(mesh->info.mesh.max_primitives_out == vertices - 2);
   CHECK(mesh->info.workgroup_size[0] == (vertices <= 16 ? 16 : 32));
   struct constants c = {count, lane, first};
   NIR_PASS(_, mesh, nir_shader_intrinsics_pass, specialize, nir_metadata_none,
            &c);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, mesh, nir_opt_copy_prop);
      NIR_PASS(progress, mesh, nir_opt_constant_folding);
      NIR_PASS(progress, mesh, nir_opt_algebraic);
      NIR_PASS(progress, mesh, nir_opt_if, 0);
      NIR_PASS(progress, mesh, nir_opt_dead_cf);
      NIR_PASS(progress, mesh, nir_opt_dce);
   } while (progress);
   nir_validate_shader(mesh, "specialized GS mesh test");
   count = MIN2(count, vertices);
   unsigned expected[32][3], prims = 0, strip = 0;
   for (unsigned i = 0; i < count; ++i) {
      if (strip >= 2) {
         bool odd = strip & 1;
         expected[prims][0] = i - 2 + (odd && !first);
         expected[prims][1] = odd ? (first ? i : i - 2) : i - 1;
         expected[prims++][2] = odd && first ? i - 1 : i;
      }
      strip++;
      if (ends & BITFIELD_BIT(i))
         strip = 0;
   }
   unsigned seen_v = 0, seen_p = 0, seen_count = 0;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_per_vertex_output:
               CHECK(nir_src_as_uint(intr->src[1]) == seen_v);
               CHECK(nir_src_comp_as_float(intr->src[0], 0) == seen_v);
               CHECK(nir_src_comp_as_float(intr->src[0], 1) == seen_v + 1);
               CHECK(nir_src_comp_as_float(intr->src[0], 2) == .5f);
               CHECK(nir_src_comp_as_float(intr->src[0], 3) == 1.f);
               seen_v++;
               break;
            case nir_intrinsic_store_per_primitive_output:
               CHECK(seen_p < prims && nir_src_as_uint(intr->src[1]) == seen_p);
               for (unsigned k = 0; k < 3; ++k)
                  CHECK(nir_src_comp_as_uint(intr->src[0], k) ==
                        expected[seen_p][k]);
               seen_p++;
               break;
            case nir_intrinsic_set_vertex_and_primitive_count:
               CHECK(nir_src_as_uint(intr->src[0]) == (prims ? count : 0));
               CHECK(nir_src_as_uint(intr->src[1]) == prims);
               seen_count++;
               break;
            default:
               CHECK(!"unlowered intrinsic");
            }
         }
      }
   }
   CHECK(seen_v == (lane ? 0 : count));
   CHECK(seen_p == (lane ? 0 : prims));
   CHECK(seen_count == !lane);
   ralloc_free(mesh);
   ralloc_free(src);
}

static void
emission(unsigned vertices, unsigned count, unsigned ends, unsigned lane)
{
   emission_mode(vertices, count, ends, lane, false);
   emission_mode(vertices, count, ends, lane, true);
}

static void
compile_unfused(unsigned v, bool inputs)
{
   nir_shader *src = source(v, 0, inputs);
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   bool compiled =
      hk_apple9_compile_gs_mesh_program(src, &key, &part, &info, &reason);
   if (!compiled)
      fprintf(stderr, "compile V%u inputs=%u: %s\n", v, inputs, reason);
   CHECK(compiled && part.binary && part.info.binary_size);
   CHECK(info.layout.max_vertices == v && info.layout.max_primitives == v - 2);
   CHECK(
      hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   free(part.binary);
   ralloc_free(src);
}

/* =========================================================================
 * Zero-PREGFX Fused VS + GS -> Native MESH tests
 * ========================================================================= */

static nir_shader *
make_vs_vbo(bool with_var0, bool use_vbo, unsigned extra_alu)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "fused VS");
   b.shader->info.io_lowered = true;
   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *iid = nir_load_instance_id(&b);
   nir_def *bi = nir_load_base_instance(&b);
   nir_def *pos;
   if (use_vbo) {
      pos = nir_load_input(
         &b, 4, 32, nir_imm_int(&b, 0), .base = 0,
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
   } else {
      nir_def *fvid = nir_u2f32(&b, vid);
      nir_def *fiid = nir_u2f32(&b, nir_iadd(&b, iid, bi));
      pos = nir_vec4(&b, fvid, fiid, nir_imm_float(&b, 0.5f),
                     nir_imm_float(&b, 1.0f));
   }
   for (unsigned i = 0; i < extra_alu; ++i)
      pos = nir_fadd(&b, pos, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f));
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS,
                                     .num_slots = 1});
   if (with_var0) {
      nir_def *v0 = nir_vec4(&b, nir_u2f32(&b, vid),
                             nir_u2f32(&b, nir_iadd(&b, iid, bi)),
                             nir_channel(&b, pos, 0), nir_channel(&b, pos, 1));
      nir_store_output(&b, v0, nir_imm_int(&b, 0), .write_mask = 15,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_VAR0,
                                        .num_slots = 1});
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_fs_simple(bool reads_var0, bool reads_clip, bool reads_vp)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fused FS");
   b.shader->info.io_lowered = true;
   nir_def *color = nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f);
   if (reads_var0) {
      color = nir_load_interpolated_input(
         &b, 4, 32,
         nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH),
         nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   if (reads_clip) {
      nir_def *cd = nir_load_interpolated_input(
         &b, 1, 32,
         nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH),
         nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      color = nir_fadd(&b, color, nir_vec4(&b, cd, cd, cd, cd));
   }
   if (reads_vp) {
      nir_def *vp = nir_load_input(
         &b, 1, 32, nir_imm_int(&b, 0), .dest_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_VIEWPORT, .num_slots = 1});
      nir_def *fvp = nir_u2f32(&b, vp);
      color = nir_fadd(&b, color, nir_vec4(&b, fvp, fvp, fvp, fvp));
   }
   nir_store_output(&b, color, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0,
                                     .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_gs_passthrough(enum mesa_prim in_prim, enum mesa_prim out_prim,
                    unsigned v_out, unsigned invocations, bool with_var0,
                    bool with_psiz, bool with_layer, bool with_vp,
                    unsigned clip_size, unsigned cull_size,
                    bool standalone_cull_slot, bool dynamic_vtx_idx)
{
   unsigned k = in_prim == MESA_PRIM_POINTS ? 1
              : in_prim == MESA_PRIM_LINES  ? 2 : 3;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "fused GS");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = in_prim;
   b.shader->info.gs.output_primitive = out_prim;
   b.shader->info.gs.vertices_in = k;
   b.shader->info.gs.vertices_out = v_out;
   b.shader->info.gs.invocations = invocations;
   b.shader->info.gs.active_stream_mask = 1;

   nir_def *inv = invocations > 1 ? nir_load_invocation_id(&b)
                                  : nir_imm_int(&b, 0);
   nir_def *prim = nir_load_primitive_id(&b);

   if (dynamic_vtx_idx) {
      nir_variable *idx_var = nir_local_variable_create(
         nir_shader_get_entrypoint(b.shader), glsl_uint_type(), "idx");
      nir_store_var(&b, idx_var, nir_imm_int(&b, 0), 1);
      nir_loop *loop = nir_push_loop(&b);
      nir_def *idx = nir_load_var(&b, idx_var);
      nir_push_if(&b, nir_uge_imm(&b, idx, k));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, idx, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS,
                                        .num_slots = 1});
      nir_emit_vertex(&b, 0);
      nir_store_var(&b, idx_var, nir_iadd_imm(&b, idx, 1), 1);
      nir_pop_loop(&b, loop);
      nir_end_primitive(&b, 0);
   } else {
      for (unsigned i = 0; i < v_out; ++i) {
         nir_def *pos = nir_load_per_vertex_input(
            &b, 4, 32, nir_imm_int(&b, i % k), nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         if (invocations > 1) {
            pos = nir_fadd(&b, pos,
                           nir_vec4(&b, nir_u2f32(&b, inv),
                                    nir_u2f32(&b, prim), nir_imm_float(&b, 0),
                                    nir_imm_float(&b, 0)));
         }
         nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS,
                                           .num_slots = 1});
         if (with_var0) {
            nir_def *v0 = nir_load_per_vertex_input(
               &b, 4, 32, nir_imm_int(&b, i % k), nir_imm_int(&b, 0),
               .dest_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
            nir_store_output(&b, v0, nir_imm_int(&b, 0), .write_mask = 15,
                             .src_type = nir_type_float32,
                             .io_semantics = {.location = VARYING_SLOT_VAR0,
                                              .num_slots = 1});
         }
         if (with_psiz) {
            nir_store_output(&b, nir_imm_float(&b, 4.0f + i),
                             nir_imm_int(&b, 0), .write_mask = 1,
                             .src_type = nir_type_float32,
                             .io_semantics = {.location = VARYING_SLOT_PSIZ,
                                              .num_slots = 1});
         }
         if (with_layer) {
            nir_store_output(&b, nir_iadd_imm(&b, inv, 1), nir_imm_int(&b, 0),
                             .write_mask = 1, .src_type = nir_type_uint32,
                             .io_semantics = {.location = VARYING_SLOT_LAYER,
                                              .num_slots = 1});
         }
         if (with_vp) {
            nir_store_output(&b, nir_iadd_imm(&b, inv, 2), nir_imm_int(&b, 0),
                             .write_mask = 1, .src_type = nir_type_uint32,
                             .io_semantics = {.location = VARYING_SLOT_VIEWPORT,
                                              .num_slots = 1});
         }
         for (unsigned d = 0; d < clip_size; ++d) {
            nir_store_output(
               &b, nir_imm_float(&b, 0.5f + d), nir_imm_int(&b, d / 4),
               .component = d % 4, .write_mask = 1,
               .src_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0,
                                .num_slots = 2});
         }
         for (unsigned d = 0; d < cull_size; ++d) {
            if (standalone_cull_slot) {
               nir_store_output(
                  &b, nir_imm_float(&b, 0.25f + d), nir_imm_int(&b, d / 4),
                  .component = d % 4, .write_mask = 1,
                  .src_type = nir_type_float32,
                  .io_semantics = {.location = VARYING_SLOT_CULL_DIST0,
                                   .num_slots = 2});
            } else {
               unsigned idx = clip_size + d;
               nir_store_output(
                  &b, nir_imm_float(&b, 0.25f + d), nir_imm_int(&b, idx / 4),
                  .component = idx % 4, .write_mask = 1,
                  .src_type = nir_type_float32,
                  .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0,
                                   .num_slots = 2});
            }
         }
         nir_emit_vertex(&b, 0);
      }
      nir_end_primitive(&b, 0);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   b.shader->info.clip_distance_array_size = clip_size;
   b.shader->info.cull_distance_array_size = cull_size;
   return b.shader;
}

/* Textbook 1->4 midpoint triangle subdivision GS (vertices_out = 12). */
static nir_shader *
make_gs_split4(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "gs4 split");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 12;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;

   nir_def *p[3];
   for (unsigned i = 0; i < 3; ++i) {
      p[i] = nir_load_per_vertex_input(
         &b, 4, 32, nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *ab = nir_fmul(&b, nir_fadd(&b, p[0], p[1]), half);
   nir_def *bc = nir_fmul(&b, nir_fadd(&b, p[1], p[2]), half);
   nir_def *ca = nir_fmul(&b, nir_fadd(&b, p[2], p[0]), half);
   nir_def *tris[4][3] = {
      {p[0], ab, ca}, {ab, p[1], bc}, {ca, bc, p[2]}, {ab, bc, ca},
   };
   for (unsigned t = 0; t < 4; ++t) {
      for (unsigned c = 0; c < 3; ++c) {
         nir_store_output(&b, tris[t][c], nir_imm_int(&b, 0), .write_mask = 15,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS,
                                           .num_slots = 1});
         nir_emit_vertex(&b, 0);
      }
      nir_end_primitive(&b, 0);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static struct hk_gs_fused_options
make_vbo_opts(bool use_vbo)
{
   struct hk_gs_fused_options opts = {0};
   opts.vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   opts.vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   opts.vertex.dynamic_stride = true;
   if (use_vbo) {
      opts.vertex.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
      opts.vertex.buffer[0] = 0;
      opts.vertex.offset[0] = 0;
      opts.vertex.stride[0] = 16;
   }
   return opts;
}

/* Specialize fused mesh NIR against concrete CPU memory regions to verify
 * exact vertex pulling, index buffer fetch, OOB clamping, and strip/fan math. */
struct fused_eval_region {
   uint64_t base;
   const void *data;
   size_t size;
};

struct fused_eval_ctx {
   uint32_t lane;
   uint32_t wg[3];
   bool first_provoking;
   uint64_t gp_addr;
   uint64_t ro_sink_addr;
   uint64_t vb_table_addr;
   struct fused_eval_region regions[6];
   unsigned region_count;
};

static bool
fused_eval_read(const struct fused_eval_ctx *ctx, uint64_t addr, unsigned bytes,
                uint64_t *out)
{
   for (unsigned i = 0; i < ctx->region_count; ++i) {
      const struct fused_eval_region *r = &ctx->regions[i];
      if (addr >= r->base && addr - r->base <= r->size &&
          bytes <= r->size - (addr - r->base)) {
         uint64_t val = 0;
         memcpy(&val, (const uint8_t *)r->data + (addr - r->base), bytes);
         *out = val;
         return true;
      }
   }
   return false;
}

static bool
specialize_fused(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct fused_eval_ctx *ctx = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *v = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_index:
      v = nir_imm_int(b, ctx->lane);
      break;
   case nir_intrinsic_load_workgroup_id:
      v = nir_imm_ivec3(b, ctx->wg[0], ctx->wg[1], ctx->wg[2]);
      break;
   case nir_intrinsic_load_provoking_last:
      v = nir_imm_int(b, !ctx->first_provoking);
      break;
   case nir_intrinsic_load_geometry_param_buffer_poly:
      v = nir_imm_int64(b, ctx->gp_addr);
      break;
   case nir_intrinsic_load_ro_sink_address_poly:
      v = nir_imm_int64(b, ctx->ro_sink_addr);
      break;
   case nir_intrinsic_load_ubo: {
      if (!nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1]))
         return false;
      unsigned ubo = nir_src_as_uint(intr->src[0]);
      unsigned off = nir_src_as_uint(intr->src[1]);
      if (ubo != HK_APPLE9_VERTEX_BUFFERS_BINDING)
         return false;
      unsigned bytes = intr->def.bit_size / 8;
      nir_def *comps[4];
      for (unsigned c = 0; c < intr->def.num_components; ++c) {
         uint64_t val = 0;
         CHECK(fused_eval_read(ctx, ctx->vb_table_addr + off + c * bytes, bytes,
                               &val));
         comps[c] = nir_imm_intN_t(b, val, intr->def.bit_size);
      }
      v = nir_vec(b, comps, intr->def.num_components);
      break;
   }
   case nir_intrinsic_load_global_constant: {
      if (!nir_src_is_const(intr->src[0]))
         return false;
      uint64_t addr = nir_src_as_uint(intr->src[0]);
      unsigned bytes = intr->def.bit_size / 8;
      nir_def *comps[4];
      for (unsigned c = 0; c < intr->def.num_components; ++c) {
         uint64_t val = 0;
         CHECK(fused_eval_read(ctx, addr + c * bytes, bytes, &val));
         comps[c] = nir_imm_intN_t(b, val, intr->def.bit_size);
      }
      v = nir_vec(b, comps, intr->def.num_components);
      break;
   }
   default:
      return false;
   }
   nir_def_replace(&intr->def, v);
   return true;
}

static nir_shader *
fused_lower_immutable(const nir_shader *vs, const nir_shader *gs,
                      const struct hk_gs_fused_options *opts,
                      struct hk_gs_fused_info *info, bool expected)
{
   struct blob vs_before, vs_after, gs_before, gs_after;
   blob_init(&vs_before);
   blob_init(&vs_after);
   blob_init(&gs_before);
   blob_init(&gs_after);
   nir_serialize(&vs_before, vs, false);
   nir_serialize(&gs_before, gs, false);
   const char *reason = NULL;
   nir_shader *mesh = hk_nir_gs_fused_mesh(vs, gs, opts, info, &reason);
   if (!mesh && expected)
      fprintf(stderr, "fused lowering failed: %s\n", reason ?: "unknown");
   CHECK((mesh != NULL) == expected);
   if (!expected)
      CHECK(reason != NULL);
   nir_serialize(&vs_after, vs, false);
   nir_serialize(&gs_after, gs, false);
   CHECK(vs_before.size == vs_after.size &&
         !memcmp(vs_before.data, vs_after.data, vs_before.size));
   CHECK(gs_before.size == gs_after.size &&
         !memcmp(gs_before.data, gs_after.data, gs_before.size));
   blob_finish(&vs_before);
   blob_finish(&vs_after);
   blob_finish(&gs_before);
   blob_finish(&gs_after);
   return mesh;
}

static void
eval_fused_shader(nir_shader *mesh, struct fused_eval_ctx *ctx,
                  float out_pos[][4], unsigned *out_v_count,
                  uint32_t out_idx[][3], unsigned *out_p_count,
                  unsigned *out_set_v, unsigned *out_set_p)
{
   nir_shader *eval = nir_shader_clone(NULL, mesh);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, eval, nir_shader_intrinsics_pass, specialize_fused,
               nir_metadata_none, ctx);
      NIR_PASS(progress, eval, nir_opt_copy_prop);
      NIR_PASS(progress, eval, nir_opt_constant_folding);
      NIR_PASS(progress, eval, nir_opt_algebraic);
      NIR_PASS(progress, eval, nir_opt_if, 0);
      NIR_PASS(progress, eval, nir_opt_loop_unroll);
      NIR_PASS(progress, eval, nir_opt_dead_cf);
      NIR_PASS(progress, eval, nir_opt_dce);
   } while (progress);
   nir_validate_shader(eval, "specialized fused GS mesh");

   *out_v_count = 0;
   *out_p_count = 0;
   *out_set_v = 0;
   *out_set_p = 0;
   nir_foreach_function_impl(impl, eval) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_per_vertex_output) {
               unsigned loc = nir_intrinsic_io_semantics(intr).location;
               if (loc == VARYING_SLOT_POS) {
                  unsigned vidx = nir_src_as_uint(intr->src[1]);
                  for (unsigned c = 0; c < 4; ++c)
                     out_pos[vidx][c] = nir_src_comp_as_float(intr->src[0], c);
                  (*out_v_count)++;
               }
            } else if (intr->intrinsic == nir_intrinsic_store_per_primitive_output) {
               unsigned pidx = nir_src_as_uint(intr->src[1]);
               for (unsigned c = 0; c < intr->num_components; ++c)
                  out_idx[pidx][c] = nir_src_comp_as_uint(intr->src[0], c);
               (*out_p_count)++;
            } else if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
               *out_set_v = nir_src_as_uint(intr->src[0]);
               *out_set_p = nir_src_as_uint(intr->src[1]);
            }
         }
      }
   }
   ralloc_free(eval);
}

static void
test_fused_nir_topologies_and_indices(void)
{
   enum {
      GP_ADDR = 0x100000,
      RO_ADDR = 0x200000,
      VBT_ADDR = 0x300000,
      VBO_ADDR = 0x400000,
      IBO_ADDR = 0x500000,
   };
   uint8_t zero_page[256] = {0};
   float vbo[16][4];
   for (unsigned i = 0; i < 16; ++i) {
      vbo[i][0] = 10.0f + i;
      vbo[i][1] = 20.0f + i;
      vbo[i][2] = 0.5f;
      vbo[i][3] = 1.0f;
   }
   uint64_t vb_table[4] = {VBO_ADDR, sizeof(vbo), 16, 0};

   /* 1. Direct POINTS -> POINTS with VBO fetch + firstVertex + gl_PointSize. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(MESA_PRIM_POINTS, MESA_PRIM_POINTS,
                                           1, 1, false, true, false, false, 0,
                                           0, false, false);
      struct hk_gs_fused_options opts = make_vbo_opts(true);
      struct hk_gs_fused_info info = {0};
      nir_shader *mesh = fused_lower_immutable(vs, gs, &opts, &info, true);
      CHECK(info.pack == 1 && mesh->info.mesh.primitive_type == MESA_PRIM_POINTS);
      struct hk_apple9_fused_gs_params gp = {
         .input_topology = MESA_PRIM_POINTS,
         .index_size_B = 0,
         .first_vertex = 3,
         .base_instance = 0,
         .input_primitives = 5,
         .first_primitive = 0,
         .groups_x = 5,
      };
      struct fused_eval_ctx ctx = {
         .lane = 0, .wg = {2, 0, 0}, .first_provoking = false,
         .gp_addr = GP_ADDR, .ro_sink_addr = RO_ADDR, .vb_table_addr = VBT_ADDR,
         .regions = {
            {GP_ADDR, &gp, sizeof(gp)}, {RO_ADDR, zero_page, sizeof(zero_page)},
            {VBT_ADDR, vb_table, sizeof(vb_table)}, {VBO_ADDR, vbo, sizeof(vbo)},
         },
         .region_count = 4,
      };
      float pos[4][4] = {{0}};
      uint32_t idx[4][3] = {{0}};
      unsigned vc = 0, pc = 0, sv = 0, sp = 0;
      eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
      /* prim 2 + first_vertex 3 -> vertex 5 -> vbo[5] = (15, 25, 0.5, 1) */
      CHECK(vc == 1 && pc == 1 && sv == 1 && sp == 1);
      CHECK(pos[0][0] == 15.0f && pos[0][1] == 25.0f && idx[0][0] == 0);
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 2. Indexed (U16) LINE_STRIP -> LINE_STRIP with vertexOffset = -1 and VBO. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(MESA_PRIM_LINES, MESA_PRIM_LINE_STRIP,
                                           2, 1, false, false, false, false, 0,
                                           0, false, false);
      struct hk_gs_fused_options opts = make_vbo_opts(true);
      struct hk_gs_fused_info info = {0};
      nir_shader *mesh = fused_lower_immutable(vs, gs, &opts, &info, true);
      CHECK(info.pack == 1 && mesh->info.mesh.primitive_type == MESA_PRIM_LINES);
      uint16_t indices[4] = {2, 5, 8, 11};
      struct hk_apple9_fused_gs_params gp = {
         .index_buffer = IBO_ADDR,
         .index_buffer_size = sizeof(indices),
         .input_topology = MESA_PRIM_LINE_STRIP,
         .index_size_B = 2,
         .first_vertex = (uint32_t)-1, /* vertexOffset = -1 */
         .input_primitives = 3,
         .groups_x = 3,
      };
      struct fused_eval_ctx ctx = {
         .lane = 0, .wg = {1, 0, 0}, .first_provoking = false,
         .gp_addr = GP_ADDR, .ro_sink_addr = RO_ADDR, .vb_table_addr = VBT_ADDR,
         .regions = {
            {GP_ADDR, &gp, sizeof(gp)}, {RO_ADDR, zero_page, sizeof(zero_page)},
            {VBT_ADDR, vb_table, sizeof(vb_table)}, {VBO_ADDR, vbo, sizeof(vbo)},
            {IBO_ADDR, indices, sizeof(indices)},
         },
         .region_count = 5,
      };
      float pos[4][4] = {{0}};
      uint32_t idx[4][3] = {{0}};
      unsigned vc = 0, pc = 0, sv = 0, sp = 0;
      eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
      /* LINE_STRIP prim 1 -> index positions 1, 2 -> indices 5, 8 -> +(-1) -> vbo[4], vbo[7] */
      CHECK(vc == 2 && pc == 1 && sv == 2 && sp == 1);
      CHECK(pos[0][0] == 14.0f && pos[1][0] == 17.0f);
      CHECK(idx[0][0] == 0 && idx[0][1] == 1);
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 3. Indexed (U8, U32, + OOB clamp) TRIANGLE_STRIP -> TRIANGLE_STRIP
    *    with dynamic gl_in[i] indexing and both provoking modes! */
   for (unsigned pass = 0; pass < 3; ++pass) {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, pass == 2);
      struct hk_gs_fused_options opts = make_vbo_opts(true);
      struct hk_gs_fused_info info = {0};
      nir_shader *mesh = fused_lower_immutable(vs, gs, &opts, &info, true);
      CHECK(info.pack == 8 && mesh->info.mesh.max_vertices_out == 32 &&
            mesh->info.mesh.max_primitives_out == 16);

      uint32_t u32_idx[5] = {1, 3, 6, 9, 12};
      uint8_t u8_idx[3] = {2, 4, 7}; /* short buffer: pos 3 is OOB -> clamps to 0 */
      bool use_u8_oob = (pass == 1);
      bool first_pv = (pass == 2);
      struct hk_apple9_fused_gs_params gp = {
         .index_buffer = IBO_ADDR,
         .index_buffer_size = use_u8_oob ? sizeof(u8_idx) : sizeof(u32_idx),
         .input_topology = MESA_PRIM_TRIANGLE_STRIP,
         .index_size_B = use_u8_oob ? 1 : 4,
         .first_vertex = 0,
         .input_primitives = 3,
         .groups_x = 1,
      };
      /* Evaluate lane 1 (odd triangle prim = 1 of TRIANGLE_STRIP!). */
      struct fused_eval_ctx ctx = {
         .lane = 1, .wg = {0, 0, 0}, .first_provoking = first_pv,
         .gp_addr = GP_ADDR, .ro_sink_addr = RO_ADDR, .vb_table_addr = VBT_ADDR,
         .regions = {
            {GP_ADDR, &gp, sizeof(gp)}, {RO_ADDR, zero_page, sizeof(zero_page)},
            {VBT_ADDR, vb_table, sizeof(vb_table)}, {VBO_ADDR, vbo, sizeof(vbo)},
            {IBO_ADDR, use_u8_oob ? (const void *)u8_idx : (const void *)u32_idx,
             use_u8_oob ? sizeof(u8_idx) : sizeof(u32_idx)},
         },
         .region_count = 5,
      };
      float pos[32][4] = {{0}};
      uint32_t idx[32][3] = {{0}};
      unsigned vc = 0, pc = 0, sv = 0, sp = 0;
      eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
      CHECK(vc == 3 && pc == 1);
      if (use_u8_oob) {
         /* Odd triangle prim=1, last-provoking: corners (2, 1, 3) -> u8_idx[2]=7, u8_idx[1]=4, OOB=0 */
         CHECK(pos[3][0] == 17.0f && pos[4][0] == 14.0f && pos[5][0] == 10.0f);
      } else if (first_pv) {
         /* Odd triangle prim=1, first-provoking: corners (1, 3, 2) -> u32_idx[1]=3, [3]=9, [2]=6 */
         CHECK(pos[3][0] == 13.0f && pos[4][0] == 19.0f && pos[5][0] == 16.0f);
      } else {
         /* Odd triangle prim=1, last-provoking: corners (2, 1, 3) -> u32_idx[2]=6, [1]=3, [3]=9 */
         CHECK(pos[3][0] == 16.0f && pos[4][0] == 13.0f && pos[5][0] == 19.0f);
      }
      CHECK(idx[1][0] == 3 && idx[1][1] == 4 && idx[1][2] == 5);
      if (pass == 0) {
         /* Full pack=8 (8*3=24 vertices) must pad runtime SetMeshOutputsEXT
          * vertex count to 25 (<= 32) so runtime quad allocation avoids
          * wg_rows == 32 (TVB-grow exact-wrap divisor). */
         gp.input_primitives = 8;
         ctx.lane = 0;
         sv = sp = 0;
         eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
         CHECK(sv == 25 && sp == 8);
      }
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 4. Indexed load_vertex_id_zero_base + load_vertex_id (incl. negative vertexOffset)
    *    + nonzero first_primitive / first_instance_offset / base_instance
    *    + component-packed VS->GS varyings (VAR0.x, VAR0.yz, VAR0.w). */
   for (unsigned neg_bias = 0; neg_bias < 2; ++neg_bias) {
      nir_builder vb = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "fused_vs_zb");
      vb.shader->info.io_lowered = true;
      nir_def *vid = nir_load_vertex_id(&vb);
      nir_def *vzb = nir_load_vertex_id_zero_base(&vb);
      nir_def *iid = nir_load_instance_id(&vb);
      nir_def *bi = nir_load_base_instance(&vb);
      nir_store_output(
         &vb,
         nir_vec4(&vb, nir_u2f32(&vb, vzb), nir_i2f32(&vb, vid),
                  nir_u2f32(&vb, iid), nir_u2f32(&vb, bi)),
         nir_imm_int(&vb, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      /* Component-packed VAR0 writes: x, yz, w separately. */
      nir_store_output(
         &vb, nir_u2f32(&vb, vzb), nir_imm_int(&vb, 0), .component = 0,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(
         &vb, nir_vec2(&vb, nir_i2f32(&vb, vid), nir_u2f32(&vb, iid)),
         nir_imm_int(&vb, 0), .component = 1, .write_mask = 3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(
         &vb, nir_u2f32(&vb, bi), nir_imm_int(&vb, 0), .component = 3,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));

      nir_builder gb = nir_builder_init_simple_shader(
         MESA_SHADER_GEOMETRY, &agx_nir_options, "fused_gs_zb");
      gb.shader->info.io_lowered = true;
      gb.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      gb.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
      gb.shader->info.gs.vertices_in = 3;
      gb.shader->info.gs.vertices_out = 3;
      gb.shader->info.gs.invocations = 1;
      gb.shader->info.gs.active_stream_mask = 1;
      nir_def *pid = nir_load_primitive_id(&gb);
      for (unsigned i = 0; i < 3; ++i) {
         nir_def *p = nir_load_per_vertex_input(
            &gb, 4, 32, nir_imm_int(&gb, i), nir_imm_int(&gb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *var0 = nir_load_per_vertex_input(
            &gb, 4, 32, nir_imm_int(&gb, i), nir_imm_int(&gb, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         /* Combine VAR0 with gl_PrimitiveIDIn into POS to verify both! */
         nir_def *out = nir_vec4(
            &gb, nir_channel(&gb, var0, 0), nir_channel(&gb, var0, 1),
            nir_fadd(&gb, nir_channel(&gb, p, 2),
                     nir_fmul_imm(&gb, nir_channel(&gb, var0, 3), 100.0f)),
            nir_u2f32(&gb, pid));
         nir_store_output(
            &gb, out, nir_imm_int(&gb, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_emit_vertex(&gb, 0);
      }
      nir_end_primitive(&gb, 0);
      nir_shader_gather_info(gb.shader, nir_shader_get_entrypoint(gb.shader));

      struct hk_gs_fused_options opts = make_vbo_opts(true);
      struct hk_gs_fused_info info = {0};
      nir_shader *mesh =
         fused_lower_immutable(vb.shader, gb.shader, &opts, &info, true);
      const uint32_t idx_buf[9] = {0, 0, 0, 7, 2, 9, 4, 5, 6};
      int32_t fv = neg_bias ? -2 : 100;
      struct hk_apple9_fused_gs_params gp = {
         .index_buffer = IBO_ADDR,
         .index_buffer_size = sizeof(idx_buf),
         .input_topology = MESA_PRIM_TRIANGLES,
         .index_size_B = 4,
         .first_vertex = (uint32_t)fv,
         .base_instance = 7,
         .input_primitives = 2,
         .first_primitive = 1, /* prim_id = 1 -> corners 3,4,5 -> indices 7,2,9 */
         .groups_x = 1,
         .first_instance_offset = 3, /* instance_id = wg.z(2) + 3 = 5 */
      };
      struct fused_eval_ctx ctx = {
         .lane = 0, .wg = {0, 0, 2}, .first_provoking = true,
         .gp_addr = GP_ADDR, .ro_sink_addr = RO_ADDR, .vb_table_addr = VBT_ADDR,
         .regions = {
            {GP_ADDR, &gp, sizeof(gp)}, {RO_ADDR, zero_page, sizeof(zero_page)},
            {IBO_ADDR, idx_buf, sizeof(idx_buf)},
         },
         .region_count = 3,
      };
      float pos[32][4] = {{0}};
      uint32_t idx[32][3] = {{0}};
      unsigned vc = 0, pc = 0, sv = 0, sp = 0;
      eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
      CHECK(vc == 3 && pc == 1);
      const uint32_t expect_zb[3] = {7, 2, 9};
      for (unsigned i = 0; i < 3; ++i) {
         /* x = zero_base fetched index (NOT index stream position 3/4/5!),
          * y = fetched_index + vertexOffset, z = iid(5) + base_instance(7)*100,
          * w = first_primitive(1) + local_prim(0) = 1. */
         CHECK(pos[i][0] == (float)expect_zb[i]);
         CHECK(pos[i][1] == (float)((int32_t)expect_zb[i] + fv));
         CHECK(pos[i][2] == 705.0f);
         CHECK(pos[i][3] == 1.0f);
      }
      ralloc_free(mesh);
      ralloc_free(vb.shader);
      ralloc_free(gb.shader);
   }

   /* 5. TRIANGLE_FAN semantic evaluation (both provoking vertex modes) +
    *    looped VS path (>200 NIR instructions). */
   for (unsigned mode = 0; mode < 3; ++mode) {
      bool looped = mode == 2;
      bool first_pv = mode == 0;
      nir_shader *vs = make_vs_vbo(false, true, looped ? 220 : 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      struct hk_gs_fused_options opts = make_vbo_opts(true);
      struct hk_gs_fused_info info = {0};
      nir_shader *mesh = fused_lower_immutable(vs, gs, &opts, &info, true);
      struct hk_apple9_fused_gs_params gp = {
         .input_topology = looped ? MESA_PRIM_TRIANGLES : MESA_PRIM_TRIANGLE_FAN,
         .index_size_B = 0,
         .first_vertex = 0,
         .input_primitives = 2,
         .first_primitive = 0,
         .groups_x = 1,
      };
      struct fused_eval_ctx ctx = {
         .lane = 1, .wg = {0, 0, 0}, .first_provoking = first_pv,
         .gp_addr = GP_ADDR, .ro_sink_addr = RO_ADDR, .vb_table_addr = VBT_ADDR,
         .regions = {
            {GP_ADDR, &gp, sizeof(gp)}, {RO_ADDR, zero_page, sizeof(zero_page)},
            {VBT_ADDR, vb_table, sizeof(vb_table)}, {VBO_ADDR, vbo, sizeof(vbo)},
         },
         .region_count = 4,
      };
      float pos[32][4] = {{0}};
      uint32_t idx[32][3] = {{0}};
      unsigned vc = 0, pc = 0, sv = 0, sp = 0;
      eval_fused_shader(mesh, &ctx, pos, &vc, idx, &pc, &sv, &sp);
      CHECK(vc == 3 && pc == 1);
      if (looped) {
         /* Looped VS prim 1 (vertices 3, 4, 5): x = 13, 14, 15 (+220 * 0.0f). */
         CHECK(pos[3][0] == 13.0f && pos[4][0] == 14.0f && pos[5][0] == 15.0f);
      } else if (first_pv) {
         /* Fan prim 1, first-provoking: (prim+1, prim+2, 0) = (2, 3, 0). */
         CHECK(pos[3][0] == 12.0f && pos[4][0] == 13.0f && pos[5][0] == 10.0f);
      } else {
         /* Fan prim 1, last-provoking: (0, prim+1, prim+2) = (0, 2, 3). */
         CHECK(pos[3][0] == 10.0f && pos[4][0] == 12.0f && pos[5][0] == 13.0f);
      }
      ralloc_free(mesh);
      ralloc_free(vs);
      ralloc_free(gs);
   }
}

static void
test_fused_outputs_and_compile(void)
{
   struct hk_apple9_graphics_key key = {0};
   key.vertex = make_vbo_opts(true).vertex;

   /* 1. gs passthrough (V=3, pos-only) -> pack=8, T32 V32 P16 (avoids wg_rows=32), concurrent-safe ring. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && part.info.binary_size > 0);
      CHECK(finfo.pack == 8 && info.layout.max_vertices == 32 &&
            info.layout.max_primitives == 16);
      uint64_t ring_max = 0;
      CHECK(agx_apple9_mesh_ring_max_groups(&info.layout, &ring_max) &&
            ring_max == UINT64_MAX);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 2. gs4 (1->4 split, V=12, pos-only) -> pack=2, T32 V32 P32 (avoids wg_rows=32), concurrent-safe ring. */
   {
      nir_shader *vs = make_vs_vbo(false, false, 0);
      nir_shader *gs = make_gs_split4();
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && part.info.binary_size > 0);
      CHECK(finfo.pack == 2 && info.layout.max_vertices == 32 &&
            info.layout.max_primitives == 32);
      uint64_t ring_max = 0;
      CHECK(agx_apple9_mesh_ring_max_groups(&info.layout, &ring_max) &&
            ring_max == UINT64_MAX);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 3. V=64 multi-round TRIANGLES -> TRIANGLE_STRIP. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 64, 1, false, false,
         false, false, 0, 0, false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && info.publication_rounds && finfo.pack == 1 &&
            info.layout.max_vertices == 64 && info.layout.max_primitives == 62);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 4. POINTS -> POINTS with gl_PointSize + VAR0. */
   {
      nir_shader *vs = make_vs_vbo(true, true, 0);
      nir_shader *gs = make_gs_passthrough(MESA_PRIM_POINTS, MESA_PRIM_POINTS,
                                           4, 1, true, true, false, false, 0, 0,
                                           false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && info.writes_point_size &&
            info.layout.indices_per_primitive == 1 &&
            part.info.apple9_varyings.mask[VARYING_SLOT_VAR0] == 0xf);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 5. LINES -> LINE_STRIP with VAR0 + gl_Layer + gl_ViewportIndex + invocations=4. */
   {
      nir_shader *vs = make_vs_vbo(true, true, 0);
      nir_shader *gs = make_gs_passthrough(MESA_PRIM_LINES, MESA_PRIM_LINE_STRIP,
                                           4, 4, true, false, true, true, 0, 0,
                                           false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && info.writes_layer_viewport &&
            info.layout.indices_per_primitive == 2 && finfo.invocations == 4 &&
            (part.info.apple9_varyings.mask[VARYING_SLOT_VIEWPORT] & 1));
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 6. gl_ClipDistance[5] (CLIP_DIST0 + CLIP_DIST1) + linked FS read + gl_Layer. */
   {
      struct hk_apple9_graphics_key clip_key = key;
      clip_key.fs_reads_clip_distance = true;
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 4, 2, false, false,
         true, false, 5, 0, false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &clip_key, &part,
                                                    &info, &finfo, &reason));
      CHECK(part.binary && info.clip_distances == 5 && info.clip_linked &&
            info.writes_layer_viewport && finfo.invocations == 2);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 7. gl_CullDistance[2] (both packed and standalone CULL_DIST0, including V=3 bump). */
   for (unsigned standalone = 0; standalone < 2; ++standalone) {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 1, 2, standalone != 0, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && info.clip_distances == 1 && info.cull_distances == 2 &&
            info.publication_rounds && info.layout.max_vertices == 4);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* 8. Large VS (>200 NIR instructions) triggers looped corner evaluation. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 220);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      struct hk_gs_fused_info finfo = {0};
      const char *reason = NULL;
      CHECK(hk_apple9_compile_gs_fused_mesh_program(vs, gs, &key, &part, &info,
                                                    &finfo, &reason));
      CHECK(part.binary && hk_apple9_native_shader_valid(
                              &part.info, &info, HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(vs);
      ralloc_free(gs);
   }
}

static void
test_fused_rejections(void)
{
   struct hk_gs_fused_options opts = make_vbo_opts(true);

   /* XFB on GS or VS. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      nir_xfb_info xfb = {0};
      gs->xfb_info = &xfb;
      fused_lower_immutable(vs, gs, &opts, NULL, false);
      gs->xfb_info = NULL;
      vs->xfb_info = &xfb;
      fused_lower_immutable(vs, gs, &opts, NULL, false);
      vs->xfb_info = NULL;
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* Multi-stream (active_stream_mask != 1), adjacency, writes_memory, bad inv/verts. */
   for (unsigned kind = 0; kind < 8; ++kind) {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      switch (kind) {
      case 0:
         gs->info.gs.active_stream_mask = 3;
         break;
      case 1:
         gs->info.gs.input_primitive = MESA_PRIM_TRIANGLES_ADJACENCY;
         break;
      case 2:
         gs->info.gs.input_primitive = MESA_PRIM_LINES_ADJACENCY;
         break;
      case 3:
         gs->info.writes_memory = true;
         break;
      case 4:
         vs->info.writes_memory = true;
         break;
      case 5:
         gs->info.gs.invocations = 33;
         break;
      case 6:
         gs->info.gs.vertices_out = 257;
         break;
      case 7:
         /* Cull distance on points output is rejected cleanly. */
         gs->info.gs.output_primitive = MESA_PRIM_POINTS;
         gs->info.cull_distance_array_size = 1;
         break;
      }
      fused_lower_immutable(vs, gs, &opts, NULL, false);
      ralloc_free(vs);
      ralloc_free(gs);
   }

   /* Dynamic vertex input prolog: rejected with load_input, accepted without. */
   {
      struct hk_gs_fused_options dyn_opts = opts;
      dyn_opts.vertex.use_prolog = true;
      nir_shader *vs_vbo = make_vs_vbo(false, true, 0);
      nir_shader *vs_proc = make_vs_vbo(false, false, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      fused_lower_immutable(vs_vbo, gs, &dyn_opts, NULL, false);
      nir_shader *ok = fused_lower_immutable(vs_proc, gs, &dyn_opts, NULL, true);
      ralloc_free(ok);
      ralloc_free(vs_vbo);
      ralloc_free(vs_proc);
      ralloc_free(gs);
   }

   /* Mismatched VS/GS float controls are rejected; matching controls succeed
    * and preserve float_controls_execution_mode. */
   {
      nir_shader *vs = make_vs_vbo(false, true, 0);
      nir_shader *gs = make_gs_passthrough(
         MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP, 3, 1, false, false,
         false, false, 0, 0, false, false);
      vs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
      gs->info.float_controls_execution_mode = 0;
      fused_lower_immutable(vs, gs, &opts, NULL, false);
      vs->info.float_controls_execution_mode = 0;
      gs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
      fused_lower_immutable(vs, gs, &opts, NULL, false);
      vs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
      gs->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
      nir_shader *ok = fused_lower_immutable(vs, gs, &opts, NULL, true);
      CHECK(ok->info.float_controls_execution_mode ==
            FLOAT_CONTROLS_DENORM_PRESERVE_FP32);
      ralloc_free(ok);
      ralloc_free(vs);
      ralloc_free(gs);
   }
}

static void
setup_mock_device(struct hk_device *dev, struct hk_physical_device *pdev)
{
   dev->dev.chip = pdev->dev.chip = AGX_CHIP_G16G;
   dev->dev.vm_id = 1;
   dev->dev.params.features = pdev->dev.params.features =
      DRM_ASAHI_FEATURE_NATIVE_MESH;
   dev->dev.shader_base = (1ull << 40);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers =
      vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->apple9_fetch_sampler =
      agx_bo_create(&dev->dev, 64, 64, 0, "fetch_sampler");
   dev->rodata.bo = agx_bo_create(&dev->dev, 4096, 64, 0, "rodata");
   dev->rodata.heap = dev->rodata.bo->va->addr;
   dev->heap = agx_bo_create(&dev->dev, 4096, 64, 0, "heap");
   dev->heap->size = 256ull << 20;
   dev->heap_va = dev->heap->va;
   dev->heap_va->size_B = dev->heap->size;
   dev->heap_init_once.called = true;
   *(struct poly_heap *)agx_bo_map(dev->rodata.bo) = (struct poly_heap){
      .base = dev->heap_va->addr,
      .size = dev->heap->size - POLY_HEAP_SINK_B,
   };
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->dev.apple9_entry_back_lock, mtx_plain);
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   list_inithead(&dev->dev.apple9_entry_lru);
   util_vma_heap_init(&dev->dev.apple9_entry_heap,
                      AGX_APPLE9_ENTRY_HEADER_SIZE,
                      AGX_APPLE9_ENTRY_CHUNK_SIZE - AGX_APPLE9_ENTRY_HEADER_SIZE);
   dev->dev.apple9_entry_heap.alloc_high = false;
   dev->dev.apple9_entry_chunks[0] =
      agx_bo_create(&dev->dev, AGX_APPLE9_ENTRY_CHUNK_SIZE, 64, 0, "entry0");
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
}

static void
teardown_mock_device(struct hk_device *dev, struct hk_physical_device *pdev)
{
   hk_apple9_bodies_finish(dev);
   hk_cmd_bo_cache_trim(dev);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   agx_bo_unreference(&dev->dev, dev->dev.apple9_entry_chunks[0]);
   agx_bo_unreference(&dev->dev, dev->heap);
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
   agx_bo_unreference(&dev->dev, dev->apple9_fetch_sampler);
   util_vma_heap_finish(&dev->dev.apple9_entry_heap);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_back_lock);
   CHECK(live_bos == 0);
}

static struct hk_api_shader *
roundtrip_api_shader(struct hk_device *dev, struct hk_api_shader *s)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(s->vk.ops->serialize(&dev->vk, &s->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *copy = NULL;
   CHECK(hk_device_shader_ops.deserialize(&dev->vk, &reader, 0, NULL, &copy) ==
         VK_SUCCESS);
   CHECK(!reader.overrun && reader.current == reader.end);
   blob_finish(&blob);
   return container_of(copy, struct hk_api_shader, vk);
}

static void
record_and_verify_zero_pregfx(struct hk_device *dev,
                              struct hk_api_shader *api[HK_APPLE9_GFX_STAGES],
                              VkPrimitiveTopology topo, bool indexed,
                              uint32_t count, uint32_t instances,
                              bool expect_native)
{
   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   VkCommandPool pool;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &pool) == VK_SUCCESS);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .commandBufferCount = 1,
   };
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   memcpy(cmd->state.gfx.shaders, api, sizeof(cmd->state.gfx.shaders));
   cmd->state.gfx.render.layer_count = 4;
   cmd->state.gfx.render.area = (VkRect2D){{0, 0}, {64, 64}};
   cmd->state.gfx.render.tilebuffer.nr_samples = 1;
   cmd->state.gfx.vb[0].addr = dev->dev.shader_base + 0x10000;
   cmd->state.gfx.vb[0].range = 16384;
   cmd->state.gfx.index.buffer.addr = dev->dev.shader_base + 0x20000;
   cmd->state.gfx.index.buffer.range = 16384;
   cmd->state.gfx.index.size = AGX_INDEX_SIZE_U32;

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   dyn->ia.primitive_topology = topo;
   dyn->ia.primitive_restart_enable = false;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->rs.cull_mode = VK_CULL_MODE_NONE;
   dyn->rs.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   dyn->rs.line.width = 1.0f;
   dyn->vp.viewport_count = 1;
   dyn->vp.scissor_count = 1;
   dyn->vp.viewports[0] = (VkViewport){0, 0, 64, 64, 0, 1};
   dyn->vp.scissors[0] = (VkRect2D){{0, 0}, {64, 64}};
   dyn->ms.rasterization_samples = 1;
   dyn->ms.sample_mask = 1;
   dyn->vi_binding_strides[0] = 16;

   struct agx_draw probe = indexed
      ? agx_draw_indexed(count, instances, 0, 0, 0,
                         cmd->state.gfx.index.buffer.addr,
                         cmd->state.gfx.index.buffer.range,
                         AGX_INDEX_SIZE_U32, 0)
      : (struct agx_draw){.b = agx_3d(count, instances, 1)};
   CHECK(hk_apple9_use_native_gs(cmd, probe) == expect_native);

   precomp_calls = 0;
   if (indexed)
      hk_CmdDrawIndexed(cb, count, instances, 0, 0, 0);
   else
      hk_CmdDraw(cb, count, instances, 0, 0);
   CHECK(!vk_command_buffer_has_error(&cmd->vk));

   unsigned cdm_streams = 0, vdm_native_streams = 0, vdm_native_dispatches = 0;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM)
         cdm_streams++;
      if (cs->type == HK_CS_VDM && cs->apple9_native_mesh) {
         vdm_native_streams++;
         vdm_native_dispatches += cs->apple9_native_dispatches;
         /* Zero R->C->R internal geometry-heap wait! */
         CHECK(!(cs->pre_raster_reads & HK_READS_INTERNAL));
         CHECK(cs->apple9_native_dispatches >= 1);
      }
   }

   if (expect_native) {
      /* ZERO PREGFX compute stream created! */
      CHECK(precomp_calls == 0);
      CHECK(cmd->current_cs.pre_gfx == NULL);
      CHECK(cdm_streams == 0);
      CHECK(vdm_native_streams >= 1);
      CHECK(vdm_native_dispatches >= 1);
      CHECK(cmd->current_cs.gfx && cmd->current_cs.gfx->apple9_native_mesh);

      /* Verify ring-overflow fallback cleanly rejects oversized ring draws. */
      struct hk_api_shader *ng =
         hk_api_shader_native(api[MESA_SHADER_GEOMETRY]);
      struct agx_apple9_mesh_layout *l =
         &hk_native_producer_main(ng)->info.apple9_native.layout;
      unsigned saved_vc = l->vertex_components;
      unsigned saved_mv = l->max_vertices;
      l->max_vertices = 4;
      l->vertex_components = 128; /* 136 rows/WG -> ring_max = 30 groups */
      unsigned pack = MAX2(hk_native_producer_main(ng)->info.apple9_gs_pack, 1u);
      unsigned inv =
         MAX2(hk_native_producer_main(ng)->info.apple9_gs_invocations, 1u);
      unsigned k = hk_native_producer_main(ng)->info.apple9_gs_input_primitive ==
                      MESA_PRIM_POINTS ? 1
                 : hk_native_producer_main(ng)->info.apple9_gs_input_primitive ==
                      MESA_PRIM_LINES ? 2 : 3;
      if (topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
          topo == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
          topo == VK_PRIMITIVE_TOPOLOGY_POINT_LIST) {
         uint32_t prims_30 = (30 * pack) / inv;
         uint32_t prims_31 = (31 * pack) / inv + 1;
         if (prims_30 > 0) {
            struct agx_draw d30 = {.b = agx_3d(prims_30 * k, 1, 1)};
            struct agx_draw d31 = {.b = agx_3d(prims_31 * k, 1, 1)};
            CHECK(hk_apple9_use_native_gs(cmd, d30));
            CHECK(!hk_apple9_use_native_gs(cmd, d31));
         }
      }
      l->vertex_components = saved_vc;
      l->max_vertices = saved_mv;
   } else {
      /* Fallback emulated GS records cleanly via PREGFX CDM + non-native VDM. */
      CHECK(cdm_streams >= 1);
      CHECK(vdm_native_streams == 0);
      CHECK(cmd->current_cs.gfx && !cmd->current_cs.gfx->apple9_native_mesh);
   }

   hk_DestroyCommandPool(device, pool, NULL);
}

static void
test_fused_zero_pregfx_recording(bool expect_native)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   setup_mock_device(dev, pdev);

   struct hk_apple9_graphics_key base_key = {0};
   hk_apple9_graphics_key_device(&base_key, NULL, pdev);
   base_key.vertex.use_prolog = false;
   base_key.vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   base_key.vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   base_key.vertex.dynamic_stride = true;
   base_key.vertex.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
   base_key.vertex.buffer[0] = 0;
   base_key.vertex.stride[0] = 16;

   struct {
      const char *name;
      enum mesa_prim in_prim, out_prim;
      VkPrimitiveTopology draw_topo;
      unsigned v_out, invocations;
      bool split4, use_vbo, with_var0, with_psiz, with_layer, with_vp;
      unsigned clip_size;
      bool indexed;
      uint32_t draw_count;
   } cases[] = {
      {"gs-direct", MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 1, false, false, false, false,
       false, false, 0, false, 30},
      {"gs-indexed-big-524k", MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 1, false, false, false, false,
       false, false, 0, true, 3 * 524288},
      {"gs4-direct-big-524k", MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 12, 1, true, false, false, false,
       false, false, 0, false, 3 * 524288},
      {"points-psiz", MESA_PRIM_POINTS, MESA_PRIM_POINTS,
       VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 2, 1, false, true, true, true, false,
       false, 0, false, 16},
      {"lines-indexed-strip", MESA_PRIM_LINES, MESA_PRIM_LINE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 4, 1, false, true, true, false, false,
       false, 0, true, 12},
      {"tri-layer-vp-clip-inv2", MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 4, 2, false, true, true, false,
       true, true, 4, false, 10},
   };

   for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
      nir_shader *vs = make_vs_vbo(cases[c].with_var0, cases[c].use_vbo, 0);
      nir_shader *gs = cases[c].split4
         ? make_gs_split4()
         : make_gs_passthrough(cases[c].in_prim, cases[c].out_prim,
                               cases[c].v_out, cases[c].invocations,
                               cases[c].with_var0, cases[c].with_psiz,
                               cases[c].with_layer, cases[c].with_vp,
                               cases[c].clip_size, 0, false, false);
      nir_shader *fs = make_fs_simple(cases[c].with_var0,
                                      cases[c].clip_size > 0, cases[c].with_vp);

      struct vk_shader_compile_info infos[3] = {
         {.stage = MESA_SHADER_VERTEX, .nir = vs,
          .next_stage_mask = VK_SHADER_STAGE_GEOMETRY_BIT,
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_GEOMETRY, .nir = gs,
          .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = fs,
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out[3] = {0};
      struct hk_apple9_graphics_key key = base_key;
      CHECK(hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL, NULL,
                                       out) == VK_SUCCESS);

      struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {0};
      api[MESA_SHADER_VERTEX] = container_of(out[0], struct hk_api_shader, vk);
      api[MESA_SHADER_GEOMETRY] = container_of(out[1], struct hk_api_shader, vk);
      api[MESA_SHADER_FRAGMENT] = container_of(out[2], struct hk_api_shader, vk);

      struct hk_api_shader *ng = hk_api_shader_native(api[MESA_SHADER_GEOMETRY]);
      CHECK(!!ng == expect_native);
      if (expect_native) {
         struct hk_shader *main_s = hk_native_producer_main(ng);
         CHECK(main_s->info.apple9_gs_fused);
         CHECK(main_s->info.apple9_gs_input_primitive == cases[c].in_prim);
         CHECK(main_s->info.apple9_gs_invocations == cases[c].invocations);
         /* Native VS leaf skips SW-VS entirely in fused mode. */
         struct hk_api_shader *nv = hk_api_shader_native(api[MESA_SHADER_VERTEX]);
         CHECK(nv && !nv->variants[HK_VS_VARIANT_SW].bo);
      }

      record_and_verify_zero_pregfx(dev, api, cases[c].draw_topo,
                                    cases[c].indexed, cases[c].draw_count, 1,
                                    expect_native);

      for (unsigned i = 0; i < 3; ++i)
         hk_api_shader_destroy(&dev->vk, out[i], NULL);
   }

   if (expect_native) {
      /* Verify vertices_out = 255 and 256 for POINTS preserve 16-bit
       * apple9_gs_prims_per_run across compile and serialize/deserialize. */
      for (unsigned v_out = 255; v_out <= 256; ++v_out) {
         nir_shader *vs = make_vs_vbo(false, false, 0);
         nir_shader *gs = make_gs_passthrough(
            MESA_PRIM_POINTS, MESA_PRIM_POINTS, v_out, 1, false, false, false,
            false, 0, 0, false, false);
         nir_shader *fs = make_fs_simple(false, false, false);
         struct vk_shader_compile_info infos[3] = {
            {.stage = MESA_SHADER_VERTEX, .nir = vs,
             .next_stage_mask = VK_SHADER_STAGE_GEOMETRY_BIT,
             .robustness = &vk_robustness_disabled},
            {.stage = MESA_SHADER_GEOMETRY, .nir = gs,
             .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
             .robustness = &vk_robustness_disabled},
            {.stage = MESA_SHADER_FRAGMENT, .nir = fs,
             .robustness = &vk_robustness_disabled},
         };
         struct vk_shader *out[3] = {0};
         struct hk_apple9_graphics_key key = base_key;
         CHECK(hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL, NULL,
                                          out) == VK_SUCCESS);
         struct hk_api_shader *gs_api =
            container_of(out[1], struct hk_api_shader, vk);
         struct hk_api_shader *ng = hk_api_shader_native(gs_api);
         CHECK(ng && hk_native_producer_main(ng)->info.apple9_gs_fused);
         CHECK(hk_native_producer_main(ng)->info.apple9_gs_prims_per_run == v_out);

         struct hk_api_shader *copy = roundtrip_api_shader(dev, gs_api);
         struct hk_api_shader *ng_copy = hk_api_shader_native(copy);
         CHECK(ng_copy && hk_native_producer_main(ng_copy)->info.apple9_gs_fused);
         CHECK(hk_native_producer_main(ng_copy)->info.apple9_gs_prims_per_run ==
               v_out);

         hk_api_shader_destroy(&dev->vk, &copy->vk, NULL);
         for (unsigned i = 0; i < 3; ++i)
            hk_api_shader_destroy(&dev->vk, out[i], NULL);
      }

      /* Multi-instance chunking (gs 600k x 2 inst -> 2 chunks; gs4 524k x 2 inst
       * -> 6 chunks) and UINT32_MAX points cursor termination check. */
      struct {
         enum mesa_prim in_prim, out_prim;
         VkPrimitiveTopology topo;
         unsigned v_out;
         bool split4;
         uint32_t count, instances;
         unsigned expect_dispatches;
         uint32_t expect_last_first_inst;
         uint32_t expect_last_first_prim;
         uint32_t expect_last_input_prims;
      } chunk_cases[] = {
         {MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
          VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, false, 3 * 600000, 2, 2, 1,
          0, 600000},
         {MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
          VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 12, true, 3 * 524288, 2, 12, 1,
          500000, 24288},
         {MESA_PRIM_POINTS, MESA_PRIM_POINTS, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
          1, false, UINT32_MAX, 1, 4295, 0, 4294000000u, 967295u},
      };
      for (unsigned c = 0; c < ARRAY_SIZE(chunk_cases); ++c) {
         nir_shader *vs = make_vs_vbo(false, false, 0);
         nir_shader *gs = chunk_cases[c].split4
            ? make_gs_split4()
            : make_gs_passthrough(chunk_cases[c].in_prim,
                                  chunk_cases[c].out_prim, chunk_cases[c].v_out,
                                  1, false, false, false, false, 0, 0, false,
                                  false);
         nir_shader *fs = make_fs_simple(false, false, false);
         struct vk_shader_compile_info infos[3] = {
            {.stage = MESA_SHADER_VERTEX, .nir = vs,
             .next_stage_mask = VK_SHADER_STAGE_GEOMETRY_BIT,
             .robustness = &vk_robustness_disabled},
            {.stage = MESA_SHADER_GEOMETRY, .nir = gs,
             .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
             .robustness = &vk_robustness_disabled},
            {.stage = MESA_SHADER_FRAGMENT, .nir = fs,
             .robustness = &vk_robustness_disabled},
         };
         struct vk_shader *out[3] = {0};
         struct hk_apple9_graphics_key key = base_key;
         CHECK(hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL, NULL,
                                          out) == VK_SUCCESS);
         struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {0};
         api[MESA_SHADER_VERTEX] = container_of(out[0], struct hk_api_shader, vk);
         api[MESA_SHADER_GEOMETRY] = container_of(out[1], struct hk_api_shader, vk);
         api[MESA_SHADER_FRAGMENT] = container_of(out[2], struct hk_api_shader, vk);

         VkDevice device = hk_device_to_handle(dev);
         const VkCommandPoolCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         };
         VkCommandPool pool;
         CHECK(hk_CreateCommandPool(device, &ci, NULL, &pool) == VK_SUCCESS);
         VkCommandBuffer cb;
         const VkCommandBufferAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .commandBufferCount = 1,
         };
         CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
         struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
         cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
         memcpy(cmd->state.gfx.shaders, api, sizeof(cmd->state.gfx.shaders));
         cmd->state.gfx.render.layer_count = 4;
         cmd->state.gfx.render.area = (VkRect2D){{0, 0}, {64, 64}};
         cmd->state.gfx.render.tilebuffer.nr_samples = 1;
         struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
         dyn->ia.primitive_topology = chunk_cases[c].topo;
         dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
         dyn->rs.cull_mode = VK_CULL_MODE_NONE;
         dyn->rs.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
         dyn->rs.line.width = 1.0f;
         dyn->vp.viewport_count = 1;
         dyn->vp.scissor_count = 1;
         dyn->vp.viewports[0] = (VkViewport){0, 0, 64, 64, 0, 1};
         dyn->vp.scissors[0] = (VkRect2D){{0, 0}, {64, 64}};
         dyn->ms.rasterization_samples = 1;
         dyn->ms.sample_mask = 1;

         precomp_calls = 0;
         hk_CmdDraw(cb, chunk_cases[c].count, chunk_cases[c].instances, 0, 7);
         CHECK(!vk_command_buffer_has_error(&cmd->vk));
         CHECK(precomp_calls == 0);
         unsigned cdm_streams = 0, dispatches = 0;
         list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
            if (cs->type == HK_CS_CDM)
               cdm_streams++;
            if (cs->type == HK_CS_VDM && cs->apple9_native_mesh)
               dispatches += cs->apple9_native_dispatches;
         }
         CHECK(cdm_streams == 0);
         CHECK(dispatches == chunk_cases[c].expect_dispatches);

         /* Inspect the final uploaded hk_apple9_fused_gs_params. */
         uint64_t last_gp_gpu = cmd->state.gfx.descriptors.root.draw.geometry_params;
         const struct hk_apple9_fused_gs_params *last_gp = NULL;
         for (unsigned ch = 0; ch < HK_UPLOAD_CHUNKS; ++ch) {
            struct hk_upload_chunk *uc =
               &cmd->uploader[HK_CMD_POOL_MAIN].chunks[ch];
            if (!uc->map)
               continue;
            if (last_gp_gpu >= uc->base &&
                last_gp_gpu + sizeof(*last_gp) <= uc->base + HK_CMD_BO_SIZE) {
               last_gp = (const struct hk_apple9_fused_gs_params *)
                  ((const uint8_t *)uc->map + (last_gp_gpu - uc->base));
               break;
            }
         }
         CHECK(last_gp != NULL);
         CHECK(last_gp->base_instance == 7);
         CHECK(last_gp->first_instance_offset == chunk_cases[c].expect_last_first_inst);
         CHECK(last_gp->first_primitive == chunk_cases[c].expect_last_first_prim);
         CHECK(last_gp->input_primitives == chunk_cases[c].expect_last_input_prims);

         hk_DestroyCommandPool(device, pool, NULL);
         for (unsigned i = 0; i < 3; ++i)
            hk_api_shader_destroy(&dev->vk, out[i], NULL);
      }
   }

   teardown_mock_device(dev, pdev);
   free(dev);
   free(pdev);
}

int
main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();

   if (argc > 1 && (!strcmp(argv[1], "gs-disabled") ||
                    !strcmp(argv[1], "mesh-disabled"))) {
      CHECK(!hk_apple9_native_gs_enabled());
      if (!strcmp(argv[1], "mesh-disabled"))
         CHECK(!hk_apple9_native_mesh_enabled());
      test_fused_zero_pregfx_recording(false);
      glsl_type_singleton_decref();
      printf("fused GS disabled mode (%s) cleanly falls back\n", argv[1]);
      return 0;
   }

   CHECK(hk_apple9_native_mesh_enabled() && hk_apple9_native_gs_enabled());

   /* Existing unfused hk_nir_gs_mesh tests. */
   for (unsigned count = 0; count <= 6; ++count)
      for (unsigned ends = 0; ends < 64; ++ends)
         emission(6, count, ends, 0);
   emission(4, 4, 0, 1);
   emission(24, 24, 0, 0);
   for (unsigned ends = 0; ends < 64; ++ends) {
      emission(3, 6, ends, 0);
      emission(4, 6, ends, 0);
   }
   emission(4, 6, 0, 1);
   for (unsigned kind = 0; kind < 6; ++kind) {
      nir_shader *src = source(4, 0, false);
      switch (kind) {
      case 0:
         src->info.gs.invocations = 33;
         break;
      case 1:
         src->info.gs.active_stream_mask = 3;
         break;
      case 2:
         src->info.gs.input_primitive = MESA_PRIM_TRIANGLES_ADJACENCY;
         break;
      case 3:
         src->info.gs.output_primitive = MESA_PRIM_POINTS;
         break;
      case 4:
         src->info.outputs_written |= VARYING_BIT_LAYER;
         break;
      case 5:
         src->info.writes_memory = true;
         break;
      }
      lower_immutable(src, false);
      ralloc_free(src);
   }
   for (unsigned invocations = 2; invocations <= 32; invocations += 30) {
      nir_shader *src = source(4, 0, false);
      src->info.gs.invocations = invocations;
      ralloc_free(lower_immutable(src, true));
      ralloc_free(src);
   }
   {
      nir_shader *src = source(4, 0, false);
      src->info.gs.invocations = 0;
      lower_immutable(src, false);
      ralloc_free(src);
   }
   {
      nir_shader *owned = source(3, 0, false);
      const char *reason = NULL;
      nir_shader *mesh = hk_nir_gs_mesh_take(owned, &reason);
      CHECK(mesh == owned);
      CHECK(mesh->info.stage == MESA_SHADER_MESH);
      ralloc_free(mesh);

      nir_shader *bad = source(3, 0, false);
      bad->info.writes_memory = true;
      CHECK(hk_nir_gs_mesh_take(bad, &reason) == NULL);
      CHECK(reason != NULL);
   }
   compile_unfused(3, false);
   compile_unfused(4, false);
   compile_unfused(3, true);
   compile_unfused(4, true);

   /* New Zero-PREGFX Fused VS + GS -> Native MESH test suite. */
   test_fused_nir_topologies_and_indices();
   test_fused_outputs_and_compile();
   test_fused_rejections();
   test_fused_zero_pregfx_recording(true);

   glsl_type_singleton_decref();
   puts("unfused + Zero-PREGFX fused VS+GS native mesh tests passed");
   return 0;
}
