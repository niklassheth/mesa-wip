/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "gravity/compiler/agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "poly/nir/poly_nir.h"
#include "../hk_apple9_polygon_points.h"
#include "../hk_cmd_buffer.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

static unsigned
output_components(nir_shader *nir)
{
   unsigned components = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output)
               components += util_bitcount(nir_intrinsic_write_mask(intr));
         }
      }
   }
   return components;
}

static int
geometry_outputs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "gs_output_boundary");
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.io_lowered = true;
   nir_def *zero = nir_imm_int(&b, 0);
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), zero,
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned i = 0; i < 124; ++i) {
      nir_store_output(&b, nir_imm_float(&b, i), zero, .write_mask = 1,
         .component = i % 4, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + i / 4,
                          .num_slots = 1});
   }
   nir_emit_vertex(&b, 0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info = {0};
   CHECK(poly_nir_lower_gs(b.shader, &count, &rast, &pre, &info));
   CHECK(rast && rast->info.stage == MESA_SHADER_VERTEX);
   CHECK(output_components(rast) == 129);
   CHECK(rast->info.outputs_written & VARYING_BIT_PSIZ);
   puts("GS lowering: 128 API components become 129 with default point size");
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   ralloc_free(b.shader);
   return 0;
}

static nir_def *
pp_forbid_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
               unsigned offset, unsigned component, unsigned comps,
               nir_alu_type type)
{
   unsigned *calls = data;
   (*calls)++;
   return nir_imm_zero(b, comps, 32);
}

static nir_def *
pp_const_pos_load(nir_builder *b, void *data, unsigned vertex,
                  unsigned location, unsigned offset, unsigned component,
                  unsigned comps, nir_alu_type type)
{
   unsigned *calls = data;
   (*calls)++;
   if (location == VARYING_SLOT_POS)
      return nir_imm_vec4(b, vertex == 1 ? 1.0f : 0.0f,
                          vertex == 2 ? 1.0f : 0.0f, 0.5f, 1.0f);
   return nir_imm_zero(b, comps, 32);
}

static int
polygon_points_no_position(void)
{
   unsigned output_comps[VARYING_SLOT_MAX] = {0};
   output_comps[VARYING_SLOT_POS] = 4;
   output_comps[VARYING_SLOT_VAR0] = 4;

   /* Case 1: GS polygon-points lowering (only == NULL) without VARYING_BIT_POS.
    * Must not invoke load (even for POS or clip/cull distances), must leave
    * valid NIR, and must lower cleanly through poly_nir_lower_gs with 0 emitted
    * vertices. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_GEOMETRY, &agx_nir_options, "gs_pp_no_pos");
      b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
      b.shader->info.gs.vertices_in = 3;
      b.shader->info.gs.vertices_out = 1;
      b.shader->info.gs.invocations = 1;
      b.shader->info.gs.active_stream_mask = 1;
      b.shader->info.io_lowered = true;

      struct hk_pp_info pp_info = {
         .clip = 2,
         .cull = 1,
         .outputs = VARYING_BIT_VAR(0),
         .output_components = output_comps,
         .face_slot = VARYING_SLOT_FOGC,
      };
      unsigned loads = 0;
      hk_apple9_pp_emit_triangle(&b, &pp_info, pp_forbid_load, &loads,
                                 nir_imm_int(&b, 0), nir_imm_int(&b, 0), NULL);
      CHECK(loads == 0);
      nir_validate_shader(b.shader, "gs_pp_no_pos");
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      CHECK(b.shader->info.outputs_written == 0);

      nir_shader *count = NULL, *rast = NULL, *pre = NULL;
      struct poly_gs_info gs_info = {0};
      CHECK(poly_nir_lower_gs(b.shader, &count, &rast, &pre, &gs_info));
      CHECK(gs_info.max_indices == 0);
      CHECK(rast && rast->info.stage == MESA_SHADER_VERTEX);
      ralloc_free(count);
      ralloc_free(rast);
      ralloc_free(pre);
      ralloc_free(b.shader);
   }

   /* Case 2: Vertex/mesh single-point emission (only != NULL) without
    * VARYING_BIT_POS emits a behind-the-eye POS without calling load. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "vs_pp_no_pos");
      b.shader->info.io_lowered = true;

      struct hk_pp_info pp_info = {
         .outputs = VARYING_BIT_VAR(0),
         .output_components = output_comps,
         .face_slot = VARYING_SLOT_FOGC,
      };
      unsigned loads = 0;
      hk_apple9_pp_emit_triangle(&b, &pp_info, pp_forbid_load, &loads,
                                 nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                 nir_imm_int(&b, 0));
      CHECK(loads == 0);
      nir_validate_shader(b.shader, "vs_pp_no_pos");
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      CHECK(b.shader->info.outputs_written == VARYING_BIT_POS);
      CHECK(output_components(b.shader) == 4);
      ralloc_free(b.shader);
   }

   /* Case 3: With VARYING_BIT_POS set, normal triangle point emission loads
    * POS for all three vertices and emits output points. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_GEOMETRY, &agx_nir_options, "gs_pp_with_pos");
      b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
      b.shader->info.gs.vertices_in = 3;
      b.shader->info.gs.vertices_out = HK_PP_MAX_VERTICES;
      b.shader->info.gs.invocations = 1;
      b.shader->info.gs.active_stream_mask = 1;
      b.shader->info.io_lowered = true;

      struct hk_pp_info pp_info = {
         .outputs = VARYING_BIT_POS,
         .output_components = output_comps,
         .face_slot = VARYING_SLOT_FOGC,
      };
      unsigned loads = 0;
      hk_apple9_pp_emit_triangle(&b, &pp_info, pp_const_pos_load, &loads,
                                 nir_imm_int(&b, 0), nir_imm_int(&b, 0), NULL);
      CHECK(loads == 3);
      nir_validate_shader(b.shader, "gs_pp_with_pos");
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      CHECK(b.shader->info.outputs_written & VARYING_BIT_POS);
      CHECK(b.shader->info.outputs_written & VARYING_BIT_FOGC);
      ralloc_free(b.shader);
   }

   /* Case 4: hk_apple9_select_gs keeps the original application GS (preserving
    * geometry-primitives query accounting) when rasterizer discard is enabled
    * or when the polygon-points variant emits 0 indices (e.g. no_position GS).
    */
   {
      size_t sz = sizeof(struct hk_api_shader) +
                  HK_GS_VARIANTS * sizeof(struct hk_shader);
      struct hk_api_shader *gs = calloc(1, sz);
      struct hk_api_shader *pp_gs = calloc(1, sz);
      CHECK(gs && pp_gs);
      gs->variants[HK_GS_VARIANT_MAIN].info.gs.max_indices = 3;
      pp_gs->variants[HK_GS_VARIANT_MAIN].info.gs.max_indices = 3;
      gs->apple9_gs_polygon_points = pp_gs;

      struct hk_cmd_buffer cmd = {0};
      cmd.vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_POINT;
      cmd.vk.dynamic_graphics_state.rs.rasterizer_discard_enable = false;
      CHECK(hk_apple9_select_gs(&cmd, gs) == pp_gs);

      /* Rasterizer discard keeps original GS so primitive queries stay exact. */
      cmd.vk.dynamic_graphics_state.rs.rasterizer_discard_enable = true;
      CHECK(hk_apple9_select_gs(&cmd, gs) == gs);

      /* Zero-vertex polygon-points variant (no_position) keeps original GS. */
      cmd.vk.dynamic_graphics_state.rs.rasterizer_discard_enable = false;
      pp_gs->variants[HK_GS_VARIANT_MAIN].info.gs.max_indices = 0;
      CHECK(hk_apple9_select_gs(&cmd, gs) == gs);

      free(pp_gs);
      free(gs);
   }

   puts("Polygon points no_position lowering: PASS");
   return 0;
}

/* A full API interface plus the default point-size output inserted by TES
 * lowering needs 129 publications. Keep the native limit until there is an
 * overflow path; compute-based pre-rasterization alone cannot raise it.
 */
int main(void)
{
   glsl_type_singleton_init_or_ref();
   CHECK(geometry_outputs() == 0);
   CHECK(polygon_points_no_position() == 0);
   const unsigned counts[] = {116, 122, 123, 124};
   for (unsigned t = 0; t < ARRAY_SIZE(counts); ++t) {
      unsigned users = counts[t];
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_EVAL, &agx_nir_options, "tes_output_boundary");
      b.shader->info.tess.point_mode = true;
      b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
      b.shader->info.io_lowered = true;
      b.shader->info.num_ubos = 1;
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *position = nir_load_ubo(&b, 4, 32, zero, zero,
                                      .align_mul = 16, .range = 16);
      nir_store_output(&b, position, zero, .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned i = 0; i < users; ++i) {
         nir_def *value = nir_load_ubo(&b, 1, 32, zero,
            nir_imm_int(&b, 16 + i * 4), .align_mul = 4, .range = 4);
         nir_store_output(&b, value, zero, .write_mask = 1,
            .component = i % 4, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + i / 4,
                             .num_slots = 1});
      }
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      CHECK(poly_nir_lower_tes(b.shader, true));
      CHECK(b.shader->info.stage == MESA_SHADER_VERTEX);
      CHECK(b.shader->info.outputs_written & VARYING_BIT_PSIZ);
      struct agx_apple9_vertex_layout layout = {.rasterize_points = true};
      struct agx_shader_part out = {0};
      const char *reason = NULL;
      bool ok = agx_compile_apple9_vertex_inputs(b.shader, &layout, &out, &reason);
      printf("TES API components=%u native components=%u compile=%s reason=%s\n",
             4 + users, 4 + users + 1, ok ? "yes" : "no", reason ?: "none");
      CHECK(ok == (4 + users + 1 <= AGX_APPLE9_PUBLICATION_COUNT));
      if (ok) CHECK(out.info.apple9_publication_count <= 128);
      free(out.binary);
      ralloc_free(b.shader);
   }
   glsl_type_singleton_decref();
   return 0;
}
