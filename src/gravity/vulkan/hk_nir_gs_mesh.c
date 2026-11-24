/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "poly/geometry.h"
#include "util/hash_table.h"
#include "hk_apple9_mesh.h"
#include "hk_nir_fused_vs.h"
#include "hk_shader.h"

/* One triangle-class input primitive per GS invocation, with Position and
 * generic per-vertex outputs. This is a compiler contract;
 * draw routing must independently establish topology, ordering and state.
 * The public helper clones its input and never changes the fallback shader.
 */
static bool
supported(const nir_shader *s, const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (s->info.stage != MESA_SHADER_GEOMETRY || !s->info.io_lowered ||
       s->info.gs.input_primitive != MESA_PRIM_TRIANGLES ||
       s->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP ||
       !s->info.gs.invocations || s->info.gs.invocations > 32 ||
       s->info.gs.active_stream_mask != 1 ||
       s->info.gs.vertices_out < 3 || s->info.gs.vertices_out > 32)
      REJECT(
         "GS mesh probe needs one triangle-list invocation and 3..32 output vertices");

   if (s->xfb_info || s->info.writes_memory ||
       s->info.clip_distance_array_size || s->info.cull_distance_array_size ||
       !(s->info.outputs_written & VARYING_BIT_POS) ||
       (s->info.outputs_written & ~HK_APPLE9_NATIVE_VERTEX_IO) ||
       (s->info.inputs_read & ~HK_APPLE9_NATIVE_VERTEX_IO) ||
       s->info.outputs_written_16bit || s->info.inputs_read_16bit)
      REJECT(
         "GS mesh probe needs Position/generic I/O without XFB or side effects");

   unsigned implementations = 0;
   nir_foreach_function_impl(impl, s) {
      if (++implementations > 1)
         REJECT("GS mesh probe requires one entrypoint implementation");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("GS mesh probe requires inlined calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output:
               if (!hk_apple9_native_vertex_slot(
                      nir_intrinsic_io_semantics(intr).location) ||
                   !nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) ||
                   intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                         intr->src[0].ssa->num_components >
                      4)
                  REJECT("unsupported GS output store");
               break;
            case nir_intrinsic_load_per_vertex_input:
               if (!hk_apple9_native_vertex_slot(
                      nir_intrinsic_io_semantics(intr).location) ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) >= 3 ||
                   !nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported GS input load");
               break;
            case nir_intrinsic_emit_vertex:
            case nir_intrinsic_end_primitive:
               if (nir_intrinsic_stream_id(intr))
                  REJECT("GS mesh probe supports stream zero only");
               break;
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT(
                     "GS mesh probe requires lowered external dereferences");
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_invocation_id:
               break;
            default:
               REJECT("unsupported GS mesh probe intrinsic");
            }
         }
      }
   }
   return true;
#undef REJECT
}

struct gs_mesh_state {
   unsigned max_vertices;
   /* GS invocation of this run (0 with one invocation). */
   nir_def *invocation;
   /* Input triangle within the draw, and this invocation's first output
    * vertex and primitive slot: workgroup x and 0, 0 unpacked; with packing,
    * x * HK_APPLE9_NATIVE_GS_PACK + lane and the lane's fixed slots. */
   nir_def *primitive, *vertex_base, *primitive_base;
   /* The draw's triangle-class topology (list, strip or fan) and provoking
    * mode, as the fallback GS's software input assembly sees them. */
   nir_def *topology, *flatshade_first;
   nir_variable *outputs[64];
   uint8_t masks[64];
   nir_alu_type types[64][4];
   nir_variable *vertices;
   nir_variable *primitives;
   nir_variable *strip;
};

/* VS output of corner vert (0..2) of triangle prim in a triangle list, strip
 * or fan, matching poly_vertex_id_for_tri_class. */
static nir_def *
tri_class_vertex(nir_builder *b, const struct gs_mesh_state *s, nir_def *prim,
                 unsigned vert)
{
   nir_def *list = nir_iadd_imm(b, nir_imul_imm(b, prim, 3), vert);
   /* Strip: odd triangles swap the two non-provoking corners. */
   nir_def *pv = nir_bcsel(b, s->flatshade_first, nir_imm_int(b, 0), nir_imm_int(b, 2));
   nir_def *keep = nir_ior(b, nir_ieq_imm(b, pv, vert),
                           nir_ieq_imm(b, nir_iand_imm(b, prim, 1), 0));
   nir_def *strip = nir_iadd(
      b, prim,
      nir_bcsel(b, keep, nir_imm_int(b, vert), nir_isub_imm(b, 3 - vert, pv)));
   /* Fan: first-provoking rotates the corners; corner 0 is the hub. */
   nir_def *fan_vert = nir_bcsel(b, s->flatshade_first,
                                 nir_imm_int(b, (vert + 1) % 3), nir_imm_int(b, vert));
   nir_def *fan = nir_bcsel(b, nir_ieq_imm(b, fan_vert, 0), nir_imm_int(b, 0),
                            nir_iadd(b, prim, fan_vert));
   return nir_bcsel(b, nir_ieq_imm(b, s->topology, MESA_PRIM_TRIANGLE_STRIP), strip,
                    nir_bcsel(b, nir_ieq_imm(b, s->topology, MESA_PRIM_TRIANGLE_FAN),
                              fan, list));
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gs_mesh_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output: {
      nir_def *value = nir_undef(b, 4, 32);
      unsigned component = nir_intrinsic_component(intr);
      for (unsigned i = 0; i < intr->src[0].ssa->num_components; ++i)
         value = nir_vector_insert_imm(
            b, value, nir_channel(b, intr->src[0].ssa, i), component + i);
      unsigned location = nir_intrinsic_io_semantics(intr).location;
      nir_store_var(b, s->outputs[location], value,
                    nir_intrinsic_write_mask(intr) << component);
      break;
   }
   case nir_intrinsic_emit_vertex: {
      nir_def *v = nir_load_var(b, s->vertices);
      /* GS over-emission has no effect. Bound the stores themselves, not
       * only the counts published after the body. Each primitive requires
       * three accepted vertices, bounding primitive indices as well. */
      nir_push_if(b, nir_ult_imm(b, v, s->max_vertices));
      {
         nir_def *p = nir_load_var(b, s->primitives);
         nir_def *n = nir_load_var(b, s->strip);
         for (unsigned loc = 0; loc < ARRAY_SIZE(s->outputs); ++loc) {
            if (!s->outputs[loc])
               continue;
            nir_def *value = nir_load_var(b, s->outputs[loc]);
            if (loc == VARYING_SLOT_POS) {
               nir_store_per_vertex_output(
                  b, value, nir_iadd(b, s->vertex_base, v), nir_imm_int(b, 0),
                  .write_mask = 15, .src_type = nir_type_float32,
                  .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
               continue;
            }
            /* Keep each component's type: packed user locations may contain
             * both integer and floating-point scalars. */
            u_foreach_bit(c, s->masks[loc]) {
               nir_store_per_vertex_output(
                  b, nir_channel(b, value, c), nir_iadd(b, s->vertex_base, v),
                  nir_imm_int(b, 0), .component = c, .write_mask = 1,
                  .src_type = s->types[loc][c],
                  .io_semantics = {.location = loc, .num_slots = 1});
            }
         }
         nir_push_if(b, nir_uge_imm(b, n, 2));
         {
            /* Reset parity at EndPrimitive. Reverse odd triangles while
             * keeping the API's provoking corner in its list position. */
            nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, n, 1), 0);
            nir_def *a = nir_iadd_imm(b, v, -2), *c = nir_iadd_imm(b, v, -1);
            nir_def *first_odd = nir_iand(b, odd, s->flatshade_first);
            nir_def *last_odd = nir_iand(b, odd, nir_inot(b, s->flatshade_first));
            nir_def *indices = nir_iadd(
               b, nir_vec3(b, nir_bcsel(b, last_odd, c, a),
                            nir_bcsel(b, first_odd, v, nir_bcsel(b, last_odd, a, c)),
                            nir_bcsel(b, first_odd, c, v)),
               s->vertex_base);
            nir_store_per_primitive_output(
               b, indices, nir_iadd(b, s->primitive_base, p), nir_imm_int(b, 0),
               .write_mask = 7,
               .src_type = nir_type_uint32,
               .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                .num_slots = 1});
            nir_store_var(b, s->primitives, nir_iadd_imm(b, p, 1), 1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->vertices, nir_iadd_imm(b, v, 1), 1);
         nir_store_var(b, s->strip, nir_iadd_imm(b, n, 1), 1);
         /* GS outputs become undefined after EmitVertex. */
         for (unsigned loc = 0; loc < ARRAY_SIZE(s->outputs); ++loc) {
            if (s->outputs[loc])
               nir_store_var(b, s->outputs[loc], nir_undef(b, 4, 32), 15);
         }
      }
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_end_primitive:
      nir_store_var(b, s->strip, nir_imm_int(b, 0), 1);
      break;
   case nir_intrinsic_load_per_vertex_input: {
      /* Reuse poly's existing VS buffer layout. This narrow contract is a
       * triangle-list draw; dispatch x is primitive, y is instance. */
      nir_def *id = nir_load_workgroup_id(b);
      nir_def *vp = nir_load_vertex_param_buffer_poly(b);
      nir_def *verts = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, vp, offsetof(struct poly_vertex_params, grid[0])),
         .align_mul = 4);
      nir_def *vertex =
         nir_iadd(b, nir_imul(b, nir_channel(b, id, 1), verts),
                  tri_class_vertex(b, s, s->primitive, nir_src_as_uint(intr->src[0])));
      /* Stage is still GS here so poly reads the producer's output mask. */
      nir_def_replace(&intr->def, poly_load_per_vertex_input(b, intr, vertex));
      return true;
   }
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def, s->invocation);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, s->primitive);
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, nir_channel(b, nir_load_workgroup_id(b), 1));
      return true;
   default:
      return false;
   }
   nir_instr_remove(&intr->instr);
   return true;
}

static void
hk_nir_gs_mesh_lower(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   unsigned vertices = nir->info.gs.vertices_out;
   /* Packing covers the qualified per-triangle shapes V3/P1 and V4/P2. */
   const unsigned pack =
      hk_apple9_native_gs_pack_enabled() && (vertices == 3 || vertices == 4)
         ? HK_APPLE9_NATIVE_GS_PACK : 1;
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   /* The launch passes the draw's topology in geometry parameters. */
   nir_def *topology = nir_load_global_constant(
      &b, 1, 32,
      nir_iadd_imm(&b, nir_load_geometry_param_buffer_poly(&b),
                   offsetof(struct poly_geometry_params, input_topology)),
      .align_mul = 4);
   /* One logical GS run per (input triangle, invocation), invocations of a
    * triangle consecutive, so output order is API order. */
   const unsigned invocations = MAX2(nir->info.gs.invocations, 1);
   nir_def *run = pack > 1 ? nir_iadd(&b, nir_imul_imm(&b, group, pack), lane) : group;
   struct gs_mesh_state s = {
      .invocation = invocations > 1 ? nir_umod_imm(&b, run, invocations) : nir_imm_int(&b, 0),
      .topology = topology,
      .flatshade_first = nir_ieq_imm(&b, nir_load_provoking_last(&b), 0),
      .max_vertices = vertices,
      .primitive = invocations > 1 ? nir_udiv_imm(&b, run, invocations) : run,
      .vertex_base = pack > 1 ? nir_imul_imm(&b, lane, vertices) : nir_imm_int(&b, 0),
      .primitive_base =
         pack > 1 ? nir_imul_imm(&b, lane, vertices - 2) : nir_imm_int(&b, 0),
      .vertices =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_vertices"),
      .primitives =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_primitives"),
      .strip =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_strip_vertices"),
   };
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location;
         unsigned component = nir_intrinsic_component(intr);
         s.masks[loc] |= nir_intrinsic_write_mask(intr) << component;
         u_foreach_bit(c, nir_intrinsic_write_mask(intr))
            s.types[loc][component + c] = nir_intrinsic_src_type(intr);
      }
   }
   for (unsigned loc = 0; loc < ARRAY_SIZE(s.outputs); ++loc) {
      if (!s.masks[loc])
         continue;
      s.outputs[loc] = nir_local_variable_create(impl, glsl_vec4_type(), "gs_output");
      nir_store_var(&b, s.outputs[loc], nir_undef(&b, 4, 32), 15);
   }
   nir_store_var(&b, s.vertices, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.primitives, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.strip, nir_imm_int(&b, 0), 1);
   nir_cursor after_setup = b.cursor;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower, nir_metadata_none, &s);

   if (pack == 1) {
      b.cursor = nir_after_impl(impl);
      nir_def *p = nir_load_var(&b, s.primitives);
      nir_def *v = nir_bcsel(&b, nir_ine_imm(&b, p, 0),
                             nir_load_var(&b, s.vertices), nir_imm_int(&b, 0));
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));

      nir_cf_list body;
      nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
      b.cursor = nir_before_impl(impl);
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      b.cursor = nir_cf_reinsert(&body, b.cursor);
      nir_pop_if(&b, NULL);
   } else {
      /* Lanes 0..pack-1 run one GS invocation each, for consecutive input
       * triangles, so publication order is API order. Each owns fixed
       * slots; primitive slots its GS did not fill become degenerate
       * triangles on its first vertex, and the group publishes every slot
       * of its live lanes. */
      nir_cf_list body;
      nir_cf_extract(&body, after_setup, nir_after_impl(impl));
      b.cursor = after_setup;
      nir_def *vp = nir_load_vertex_param_buffer_poly(&b);
      nir_def *verts = nir_load_global_constant(
         &b, 1, 32,
         nir_iadd_imm(&b, vp, offsetof(struct poly_vertex_params, grid[0])),
         .align_mul = 4);
      nir_def *triangles = nir_bcsel(
         &b, nir_ieq_imm(&b, topology, MESA_PRIM_TRIANGLES), nir_udiv_imm(&b, verts, 3),
         nir_usub_sat(&b, verts, nir_imm_int(&b, 2)));
      nir_def *first = nir_imul_imm(&b, group, pack);
      nir_def *runs = nir_imul_imm(&b, triangles, invocations);
      nir_def *live = nir_umin(&b, nir_usub_sat(&b, runs, first),
                               nir_imm_int(&b, pack));
      nir_if *active = nir_push_if(&b, nir_ult(&b, lane, live));
      active->control = nir_selection_control_dont_flatten;
      b.cursor = nir_cf_reinsert(&body, b.cursor);
      /* A lane whose GS emitted nothing still publishes its slots: give the
       * degenerate fill a finite vertex. */
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_var(&b, s.vertices), 0));
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, 0, 0, 0, 1), s.vertex_base, nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&b, NULL);
      nir_def *p = nir_load_var(&b, s.primitives);
      for (unsigned i = 0; i < vertices - 2; ++i) {
         nir_push_if(&b, nir_ule_imm(&b, p, i));
         nir_store_per_primitive_output(
            &b, nir_vec3(&b, s.vertex_base, s.vertex_base, s.vertex_base),
            nir_iadd_imm(&b, s.primitive_base, i), nir_imm_int(&b, 0),
            .write_mask = 7, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                             .num_slots = 1});
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, active);
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_set_vertex_and_primitive_count(
         &b, nir_imul_imm(&b, live, vertices),
         nir_imul_imm(&b, live, vertices - 2), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
   }

   memset(&nir->info.mesh, 0, sizeof(nir->info.mesh));
   nir->info.stage = MESA_SHADER_MESH;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   if (pack > 1) {
      /* The qualified classes T32 V24/P17 and T32 V32/P16. */
      nir->info.mesh.max_vertices_out = pack * vertices;
      nir->info.mesh.max_primitives_out = vertices == 3 ? 17 : 16;
      nir->info.workgroup_size[0] = 32;
   } else {
      nir->info.mesh.max_vertices_out = vertices;
      nir->info.mesh.max_primitives_out = vertices - 2;
      nir->info.workgroup_size[0] = vertices <= 16 ? 16 : 32;
   }
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir_shader_gather_info(nir, impl);
   nir_validate_shader(nir, "GS mesh emission lowering");
}

nir_shader *
hk_nir_gs_mesh_take(nir_shader *nir, const char **reason)
{
   if (reason)
      *reason = NULL;
   if (!nir)
      return NULL;
   if (!supported(nir, reason)) {
      ralloc_free(nir);
      return NULL;
   }
   hk_nir_gs_mesh_lower(nir);
   return nir;
}

nir_shader *
hk_nir_gs_mesh(const nir_shader *source, const char **reason)
{
   if (reason)
      *reason = NULL;
   if (!supported(source, reason))
      return NULL;
   nir_shader *nir = nir_shader_clone(NULL, source);
   if (!nir) {
      if (reason)
         *reason = "out of memory cloning GS mesh probe";
      return NULL;
   }
   hk_nir_gs_mesh_lower(nir);
   return nir;
}

/* =========================================================================
 * Zero-PREGFX Fused VS + GS -> Native MESH Lowering
 *
 * Clones both the VS and GS NIR without mutating the originals, lowers VS
 * vertex attribute pulls directly against HK_APPLE9_VERTEX_BUFFERS_BINDING,
 * evaluates the input primitive's K (1..3) vertices directly in local
 * registers/SSA temps (zero DRAM round-trip, zero PREGFX compute pass), and
 * runs the GS body emitting POINTS, LINES, or TRIANGLES directly into Apple9
 * native mesh outputs.
 * ========================================================================= */

static unsigned
fused_input_verts(enum mesa_prim input_prim)
{
   switch (input_prim) {
   case MESA_PRIM_POINTS:
      return 1;
   case MESA_PRIM_LINES:
      return 2;
   case MESA_PRIM_TRIANGLES:
      return 3;
   default:
      return 0;
   }
}

static enum mesa_prim
fused_output_mesh_prim(enum mesa_prim gs_out, unsigned *min_verts,
                       unsigned *indices_per_prim)
{
   switch (gs_out) {
   case MESA_PRIM_POINTS:
      *min_verts = 1;
      *indices_per_prim = 1;
      return MESA_PRIM_POINTS;
   case MESA_PRIM_LINE_STRIP:
      *min_verts = 2;
      *indices_per_prim = 2;
      return MESA_PRIM_LINES;
   case MESA_PRIM_TRIANGLE_STRIP:
      *min_verts = 3;
      *indices_per_prim = 3;
      return MESA_PRIM_TRIANGLES;
   default:
      *min_verts = 0;
      *indices_per_prim = 0;
      return MESA_PRIM_COUNT;
   }
}

static bool
fused_gs_output_slot_ok(unsigned loc, unsigned comp, unsigned num_comps,
                        unsigned clip_size, unsigned cull_size)
{
   if (comp + num_comps > 4)
      return false;
   switch (loc) {
   case VARYING_SLOT_POS:
      return true;
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_ID:
      return comp == 0 && num_comps == 1;
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
      return 4 * (loc - VARYING_SLOT_CLIP_DIST0) + comp + num_comps <= 8;
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
      return clip_size + 4 * (loc - VARYING_SLOT_CULL_DIST0) + comp +
                num_comps <=
             8 &&
             cull_size > 0;
   default:
      return loc >= VARYING_SLOT_VAR0 && loc < VARYING_SLOT_VAR0 + 32;
   }
}

static bool
fused_supported(const nir_shader *vs, const nir_shader *gs,
                const struct hk_gs_fused_options *opts, const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (!vs || !gs || vs->info.stage != MESA_SHADER_VERTEX ||
       !vs->info.io_lowered || gs->info.stage != MESA_SHADER_GEOMETRY ||
       !gs->info.io_lowered)
      REJECT("fused GS requires lowered VS and GS NIR");

   unsigned k = fused_input_verts(gs->info.gs.input_primitive);
   if (!k)
      REJECT("fused GS does not support adjacency or unknown input topology");

   unsigned min_verts = 0, indices_per_prim = 0;
   enum mesa_prim mesh_prim = fused_output_mesh_prim(
      gs->info.gs.output_primitive, &min_verts, &indices_per_prim);
   if (mesh_prim == MESA_PRIM_COUNT)
      REJECT("fused GS supports points, line_strip, or triangle_strip outputs");

   unsigned v_out = gs->info.gs.vertices_out;
   unsigned p_per_run =
      mesh_prim == MESA_PRIM_POINTS ? v_out
      : mesh_prim == MESA_PRIM_LINES ? (v_out >= 1 ? v_out - 1 : 0)
                                     : (v_out >= 2 ? v_out - 2 : 0);
   if (!gs->info.gs.invocations || gs->info.gs.invocations > 32 ||
       gs->info.gs.active_stream_mask != 1 || v_out < min_verts ||
       v_out > 256 || !p_per_run || p_per_run > 256)
      REJECT("fused GS declaration exceeds native mesh invocation or vertex bounds");

   if (vs->xfb_info || gs->xfb_info || vs->info.writes_memory ||
       gs->info.writes_memory || vs->info.outputs_written_16bit ||
       vs->info.inputs_read_16bit || gs->info.outputs_written_16bit ||
       gs->info.inputs_read_16bit)
      REJECT("fused GS excludes XFB, memory writes, and 16-bit IO");

   if (vs->info.float_controls_execution_mode !=
       gs->info.float_controls_execution_mode)
      REJECT("fused GS requires matching VS and GS float execution controls");

   unsigned clip_size = gs->info.clip_distance_array_size;
   unsigned cull_size = gs->info.cull_distance_array_size;
   if (clip_size > 8 || cull_size > 8 || clip_size + cull_size > 8)
      REJECT("fused GS supports at most 8 clip plus cull distances");
   if (cull_size && mesh_prim != MESA_PRIM_TRIANGLES)
      REJECT("fused GS cull distances require triangle output");
   if (!(gs->info.outputs_written & VARYING_BIT_POS))
      REJECT("fused GS requires Position output");

   /* Validate VS instructions via shared fused-VS validator. */
   if (!hk_nir_fused_validate_vs(vs, opts ? &opts->vertex : NULL, reason))
      return false;

   /* Validate GS instructions. */
   unsigned gs_impls = 0;
   nir_foreach_function_impl(impl, gs) {
      if (++gs_impls > 1)
         REJECT("fused GS requires single-entrypoint GS");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("fused GS requires inlined GS calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused GS requires constant GS output slot offsets");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->src[0].ssa->bit_size != 32 ||
                   !fused_gs_output_slot_ok(loc, nir_intrinsic_component(intr),
                                            intr->src[0].ssa->num_components,
                                            clip_size, cull_size))
                  REJECT("unsupported GS output store in fused GS");
               break;
            }
            case nir_intrinsic_load_per_vertex_input: {
               if (!nir_src_is_const(intr->src[1]))
                  REJECT("fused GS requires constant GS input slot offsets");
               if (nir_src_is_const(intr->src[0]) &&
                   nir_src_as_uint(intr->src[0]) >= k)
                  REJECT("GS input vertex index out of range");
               uint64_t loc = (uint64_t)nir_intrinsic_io_semantics(intr).location +
                              nir_src_as_uint(intr->src[1]);
               if (loc >= VARYING_SLOT_MAX || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported GS input load in fused GS");
               break;
            }
            case nir_intrinsic_emit_vertex:
            case nir_intrinsic_end_primitive:
               if (nir_intrinsic_stream_id(intr))
                  REJECT("fused GS supports stream zero only");
               break;
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT("fused GS requires lowered external GS dereferences");
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_invocation_id:
            case nir_intrinsic_load_draw_id:
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_first_vertex:
            case nir_intrinsic_load_constant:
            case nir_intrinsic_image_load_apple9_agx:
               break;
            default:
               REJECT("unsupported GS intrinsic in fused GS");
            }
         }
      }
   }
   return true;
#undef REJECT
}

/* Assemble the draw-relative vertex/index position for corner `vert` (0..K-1)
 * of input primitive `prim` under the draw's runtime input_topology. */
static nir_def *
fused_vertex_pos(nir_builder *b, enum mesa_prim input_prim, nir_def *topology,
                 nir_def *flatshade_first, nir_def *prim, nir_def *vert)
{
   if (input_prim == MESA_PRIM_POINTS)
      return prim;
   if (input_prim == MESA_PRIM_LINES) {
      nir_def *list = nir_iadd(b, nir_imul_imm(b, prim, 2), vert);
      nir_def *strip = nir_iadd(b, prim, vert);
      return nir_bcsel(b, nir_ieq_imm(b, topology, MESA_PRIM_LINE_STRIP),
                       strip, list);
   }
   /* MESA_PRIM_TRIANGLES: list, strip, or fan (provoking-preserving). */
   nir_def *list = nir_iadd(b, nir_imul_imm(b, prim, 3), vert);
   nir_def *pv =
      nir_bcsel(b, flatshade_first, nir_imm_int(b, 0), nir_imm_int(b, 2));
   nir_def *keep = nir_ior(b, nir_ieq(b, pv, vert),
                           nir_ieq_imm(b, nir_iand_imm(b, prim, 1), 0));
   nir_def *swapped = nir_isub(b, nir_isub_imm(b, 3, vert), pv);
   nir_def *strip = nir_iadd(b, prim, nir_bcsel(b, keep, vert, swapped));
   nir_def *vert_p1 = nir_iadd_imm(b, vert, 1);
   nir_def *vert_rot =
      nir_bcsel(b, nir_ieq_imm(b, vert_p1, 3), nir_imm_int(b, 0), vert_p1);
   nir_def *fan_vert = nir_bcsel(b, flatshade_first, vert_rot, vert);
   nir_def *fan = nir_bcsel(b, nir_ieq_imm(b, fan_vert, 0), nir_imm_int(b, 0),
                            nir_iadd(b, prim, fan_vert));
   return nir_bcsel(b, nir_ieq_imm(b, topology, MESA_PRIM_TRIANGLE_STRIP),
                    strip,
                    nir_bcsel(b, nir_ieq_imm(b, topology, MESA_PRIM_TRIANGLE_FAN),
                              fan, list));
}

struct fused_gs_state {
   enum mesa_prim mesh_prim;
   unsigned k;
   unsigned max_vertices;
   unsigned clip_size;
   unsigned cull_size;
   nir_def *invocation;
   nir_def *primitive;
   nir_def *instance;
   nir_def *first_vertex;
   nir_def *base_instance;
   nir_def *vertex_base;
   nir_def *primitive_base;
   nir_def *flatshade_first;
   nir_variable *vs_out[3][VARYING_SLOT_MAX];
   nir_variable *outputs[VARYING_SLOT_MAX];
   uint8_t masks[VARYING_SLOT_MAX];
   nir_alu_type types[VARYING_SLOT_MAX][4];
   nir_variable *vertices;
   nir_variable *primitives;
   nir_variable *strip;
};

static bool
fused_lower_gs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct fused_gs_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned wmask = nir_intrinsic_write_mask(intr);
      /* Drop gl_PointSize on non-point outputs cleanly. */
      if (loc == VARYING_SLOT_PSIZ && s->mesh_prim != MESA_PRIM_POINTS) {
         nir_instr_remove(&intr->instr);
         return true;
      }
      if (loc == VARYING_SLOT_CULL_DIST0 || loc == VARYING_SLOT_CULL_DIST1) {
         /* Remap standalone CULL_DIST slots right after CLIP_DIST slots. */
         u_foreach_bit(i, wmask) {
            unsigned idx = s->clip_size + 4 * (loc - VARYING_SLOT_CULL_DIST0) +
                           comp + i;
            unsigned tloc = VARYING_SLOT_CLIP_DIST0 + idx / 4;
            unsigned tcomp = idx % 4;
            nir_def *val = nir_undef(b, 4, 32);
            val = nir_vector_insert_imm(b, val,
                                        nir_channel(b, intr->src[0].ssa, i),
                                        tcomp);
            nir_store_var(b, s->outputs[tloc], val, BITFIELD_BIT(tcomp));
         }
         nir_instr_remove(&intr->instr);
         return true;
      }
      nir_def *value = nir_undef(b, 4, 32);
      for (unsigned i = 0; i < intr->src[0].ssa->num_components; ++i)
         value = nir_vector_insert_imm(
            b, value, nir_channel(b, intr->src[0].ssa, i), comp + i);
      nir_store_var(b, s->outputs[loc], value, wmask << comp);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_emit_vertex: {
      nir_def *v = nir_load_var(b, s->vertices);
      nir_push_if(b, nir_ult_imm(b, v, s->max_vertices));
      {
         nir_def *p = nir_load_var(b, s->primitives);
         nir_def *n = nir_load_var(b, s->strip);
         nir_def *vidx = nir_iadd(b, s->vertex_base, v);
         for (unsigned loc = 0; loc < ARRAY_SIZE(s->outputs); ++loc) {
            if (!s->outputs[loc])
               continue;
            nir_def *value = nir_load_var(b, s->outputs[loc]);
            if (loc == VARYING_SLOT_POS) {
               nir_store_per_vertex_output(
                  b, value, vidx, nir_imm_int(b, 0), .write_mask = 15,
                  .src_type = nir_type_float32,
                  .io_semantics = {.location = VARYING_SLOT_POS,
                                   .num_slots = 1});
               continue;
            }
            if (loc == VARYING_SLOT_PSIZ || loc == VARYING_SLOT_LAYER ||
                loc == VARYING_SLOT_VIEWPORT ||
                loc == VARYING_SLOT_PRIMITIVE_ID) {
               nir_store_per_vertex_output(
                  b, nir_channel(b, value, 0), vidx, nir_imm_int(b, 0),
                  .component = 0, .write_mask = 1,
                  .src_type = loc == VARYING_SLOT_PSIZ ? nir_type_float32
                                                       : nir_type_uint32,
                  .io_semantics = {.location = loc, .num_slots = 1});
               continue;
            }
            u_foreach_bit(c, s->masks[loc]) {
               nir_store_per_vertex_output(
                  b, nir_channel(b, value, c), vidx, nir_imm_int(b, 0),
                  .component = c, .write_mask = 1, .src_type = s->types[loc][c],
                  .io_semantics = {.location = loc, .num_slots = 1});
            }
         }
         if (s->mesh_prim == MESA_PRIM_POINTS) {
            nir_store_per_primitive_output(
               b, vidx, nir_iadd(b, s->primitive_base, p), nir_imm_int(b, 0),
               .write_mask = 1, .src_type = nir_type_uint32,
               .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                .num_slots = 1});
            nir_store_var(b, s->primitives, nir_iadd_imm(b, p, 1), 1);
         } else if (s->mesh_prim == MESA_PRIM_LINES) {
            nir_push_if(b, nir_uge_imm(b, n, 1));
            {
               nir_def *indices =
                  nir_vec2(b, nir_iadd_imm(b, vidx, -1), vidx);
               nir_store_per_primitive_output(
                  b, indices, nir_iadd(b, s->primitive_base, p),
                  nir_imm_int(b, 0), .write_mask = 3,
                  .src_type = nir_type_uint32,
                  .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                   .num_slots = 1});
               nir_store_var(b, s->primitives, nir_iadd_imm(b, p, 1), 1);
            }
            nir_pop_if(b, NULL);
         } else {
            nir_push_if(b, nir_uge_imm(b, n, 2));
            {
               nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, n, 1), 0);
               nir_def *a = nir_iadd_imm(b, v, -2);
               nir_def *c = nir_iadd_imm(b, v, -1);
               nir_def *first_odd = nir_iand(b, odd, s->flatshade_first);
               nir_def *last_odd =
                  nir_iand(b, odd, nir_inot(b, s->flatshade_first));
               nir_def *indices = nir_iadd(
                  b,
                  nir_vec3(b, nir_bcsel(b, last_odd, c, a),
                           nir_bcsel(b, first_odd, v,
                                     nir_bcsel(b, last_odd, a, c)),
                           nir_bcsel(b, first_odd, c, v)),
                  s->vertex_base);
               nir_store_per_primitive_output(
                  b, indices, nir_iadd(b, s->primitive_base, p),
                  nir_imm_int(b, 0), .write_mask = 7,
                  .src_type = nir_type_uint32,
                  .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                   .num_slots = 1});
               nir_store_var(b, s->primitives, nir_iadd_imm(b, p, 1), 1);
            }
            nir_pop_if(b, NULL);
         }
         nir_store_var(b, s->vertices, nir_iadd_imm(b, v, 1), 1);
         nir_store_var(b, s->strip, nir_iadd_imm(b, n, 1), 1);
         /* Reset staging outputs after EmitVertex (keep declared-but-unwritten
          * clip distances finite at 0.0f so unused planes stay deterministic). */
         for (unsigned loc = 0; loc < ARRAY_SIZE(s->outputs); ++loc) {
            if (!s->outputs[loc])
               continue;
            nir_def *reset =
               (loc == VARYING_SLOT_CLIP_DIST0 || loc == VARYING_SLOT_CLIP_DIST1)
                  ? nir_imm_vec4(b, 0, 0, 0, 0)
                  : nir_undef(b, 4, 32);
            nir_store_var(b, s->outputs[loc], reset, 15);
         }
      }
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_end_primitive:
      if (s->mesh_prim != MESA_PRIM_POINTS)
         nir_store_var(b, s->strip, nir_imm_int(b, 0), 1);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_load_per_vertex_input: {
      unsigned loc = nir_intrinsic_io_semantics(intr).location +
                     nir_src_as_uint(intr->src[1]);
      unsigned comp = nir_intrinsic_component(intr);
      unsigned num = intr->def.num_components;
      nir_def *vtx_vec = NULL;
      if (nir_src_is_const(intr->src[0])) {
         unsigned idx = nir_src_as_uint(intr->src[0]);
         vtx_vec = nir_load_var(b, s->vs_out[idx][loc]);
      } else {
         nir_def *idx = intr->src[0].ssa;
         vtx_vec = nir_load_var(b, s->vs_out[0][loc]);
         for (unsigned c = 1; c < s->k; ++c)
            vtx_vec = nir_bcsel(b, nir_ieq_imm(b, idx, c),
                                nir_load_var(b, s->vs_out[c][loc]), vtx_vec);
      }
      nir_def *comps[4];
      for (unsigned i = 0; i < num; ++i)
         comps[i] = nir_channel(b, vtx_vec, comp + i);
      nir_def_replace(&intr->def, nir_vec(b, comps, num));
      return true;
   }
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def, s->invocation);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, s->primitive);
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, s->instance);
      return true;
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
      nir_def_replace(&intr->def, s->first_vertex);
      return true;
   case nir_intrinsic_load_base_instance:
      nir_def_replace(&intr->def, s->base_instance);
      return true;
   default:
      return false;
   }
}

nir_shader *
hk_nir_gs_fused_mesh(const nir_shader *vs_source, const nir_shader *gs_source,
                     const struct hk_gs_fused_options *options,
                     struct hk_gs_fused_info *out_info, const char **reason)
{
   if (reason)
      *reason = NULL;
   if (!fused_supported(vs_source, gs_source, options, reason))
      return NULL;

   const unsigned k = fused_input_verts(gs_source->info.gs.input_primitive);
   unsigned min_verts = 0, indices_per_prim = 0;
   const enum mesa_prim mesh_prim = fused_output_mesh_prim(
      gs_source->info.gs.output_primitive, &min_verts, &indices_per_prim);
   const unsigned v_out = gs_source->info.gs.vertices_out;
   const unsigned p_per_run =
      mesh_prim == MESA_PRIM_POINTS ? v_out
      : mesh_prim == MESA_PRIM_LINES ? v_out - 1
                                     : v_out - 2;
   const unsigned invocations = MAX2(gs_source->info.gs.invocations, 1u);
   const unsigned clip_size = gs_source->info.clip_distance_array_size;
   const unsigned cull_size = gs_source->info.cull_distance_array_size;

   /* Clone GS (becomes the output MESH shader) and VS (inlined into it). */
   nir_shader *nir = nir_shader_clone(NULL, gs_source);
   nir_shader *vs = nir_shader_clone(NULL, vs_source);
   if (!nir || !vs) {
      ralloc_free(nir);
      ralloc_free(vs);
      if (reason)
         *reason = "out of memory cloning fused VS + GS";
      return NULL;
   }

   /* Normalize VS load_input base and lower vertex attribute pulls. */
   if (!hk_nir_fused_prepare_vs(vs, options ? &options->vertex : NULL, reason)) {
      ralloc_free(nir);
      ralloc_free(vs);
      return NULL;
   }

   NIR_PASS(_, nir, nir_lower_returns);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   /* Scan GS inputs read from VS and GS output masks/types. */
   uint8_t vs_read_mask[VARYING_SLOT_MAX] = {0};
   bool vs_any_read = false;
   uint8_t out_masks[VARYING_SLOT_MAX] = {0};
   nir_alu_type out_types[VARYING_SLOT_MAX][4];
   memset(out_types, 0, sizeof(out_types));

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_per_vertex_input) {
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned comp = nir_intrinsic_component(intr);
            vs_read_mask[loc] |= BITFIELD_MASK(intr->def.num_components) << comp;
            vs_any_read = true;
         } else if (intr->intrinsic == nir_intrinsic_store_output) {
            unsigned loc = nir_intrinsic_io_semantics(intr).location +
                           nir_src_as_uint(intr->src[1]);
            unsigned comp = nir_intrinsic_component(intr);
            unsigned wmask = nir_intrinsic_write_mask(intr);
            if (loc == VARYING_SLOT_PSIZ && mesh_prim != MESA_PRIM_POINTS)
               continue;
            if (loc == VARYING_SLOT_CULL_DIST0 || loc == VARYING_SLOT_CULL_DIST1) {
               u_foreach_bit(i, wmask) {
                  unsigned idx = clip_size +
                                 4 * (loc - VARYING_SLOT_CULL_DIST0) + comp + i;
                  unsigned tloc = VARYING_SLOT_CLIP_DIST0 + idx / 4;
                  unsigned tcomp = idx % 4;
                  out_masks[tloc] |= BITFIELD_BIT(tcomp);
                  out_types[tloc][tcomp] = nir_type_float32;
               }
            } else {
               out_masks[loc] |= wmask << comp;
               u_foreach_bit(i, wmask)
                  out_types[loc][comp + i] = nir_intrinsic_src_type(intr);
            }
         }
      }
   }

   /* Ensure every declared clip distance plane is emitted (default 0.0f if
    * unwritten), matching agx_nir_lower_apple9_native_mesh's validation. */
   for (unsigned d = 0; d < clip_size; ++d) {
      unsigned loc = VARYING_SLOT_CLIP_DIST0 + d / 4;
      unsigned c = d % 4;
      out_masks[loc] |= BITFIELD_BIT(c);
      out_types[loc][c] = nir_type_float32;
   }

   /* Estimate vertex output plane count for ring-safe packing selection. */
   unsigned vertex_components = 4; /* VARYING_SLOT_POS */
   for (unsigned loc = VARYING_SLOT_VAR0; loc < VARYING_SLOT_VAR0 + 32; ++loc)
      vertex_components += util_bitcount(out_masks[loc]);
   if (mesh_prim == MESA_PRIM_POINTS)
      vertex_components += 1; /* VARYING_SLOT_PSIZ */
   if ((out_masks[VARYING_SLOT_LAYER] & 1) ||
       (out_masks[VARYING_SLOT_VIEWPORT] & 1))
      vertex_components += 1; /* packed Layer/Viewport plane */
   if (out_masks[VARYING_SLOT_VIEWPORT] & 1)
      vertex_components += 1; /* flat ViewportIndex varying */
   if (out_masks[VARYING_SLOT_PRIMITIVE_ID] & 1)
      vertex_components += 1;
   vertex_components += clip_size;
   if (options && options->fs_reads_clip_distance)
      vertex_components += clip_size;

   if (vertex_components > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS) {
      ralloc_free(nir);
      ralloc_free(vs);
      if (reason)
         *reason = "fused GS outputs exceed Apple9 UVS varying capacity";
      return NULL;
   }

   const unsigned planes = ALIGN_POT(vertex_components, 4);
   const bool position_only = mesh_prim == MESA_PRIM_TRIANGLES &&
                              vertex_components == 4 && !cull_size;
   unsigned pack = 1;
   unsigned mesh_threads = 32;
   unsigned mesh_max_v = v_out;
   unsigned mesh_max_p = p_per_run;

   if (mesh_prim == MESA_PRIM_TRIANGLES && !cull_size) {
      const unsigned safe_quads =
         (AGX_APPLE9_MESH_RING_CONCURRENT_SAFE_ROWS -
          AGX_APPLE9_MESH_RING_HEADER_ROWS) /
         planes;
      const unsigned max_safe_verts = safe_quads * 4;
      unsigned raw_pack = MIN2(32u, max_safe_verts / v_out);
      raw_pack = MIN2(raw_pack, 256u / p_per_run);
      pack = raw_pack >= 2 ? (1u << (util_last_bit(raw_pack) - 1)) : 1u;

      if (position_only) {
         uint64_t shapes = hk_apple9_native_mesh_packed_shapes();
         bool fitted = false;
         if (pack > 1) {
            fitted = hk_apple9_packed_mesh_class_fit(
               shapes, 32, pack * v_out, pack * p_per_run, &mesh_threads,
               &mesh_max_v, &mesh_max_p);
            if (!fitted)
               pack = 1;
         }
         if (!fitted) {
            if (v_out <= 32) {
               if (!hk_apple9_packed_mesh_class_fit(
                      shapes, 16, v_out, p_per_run, &mesh_threads, &mesh_max_v,
                      &mesh_max_p)) {
                  ralloc_free(nir);
                  ralloc_free(vs);
                  if (reason)
                     *reason = "fused GS shape is outside selected packed classes";
                  return NULL;
               }
            } else {
               mesh_threads = 32;
               mesh_max_v = v_out;
               mesh_max_p = p_per_run;
            }
         }
      } else {
         mesh_max_v = pack * v_out;
         mesh_max_p = pack * p_per_run;
         mesh_threads = (pack == 1 && v_out == 3 && p_per_run == 1) ? 16 : 32;
      }
   } else {
      pack = 1;
      mesh_threads = 32;
      mesh_max_v = v_out;
      mesh_max_p = p_per_run;
      if (cull_size && mesh_max_v == 3 && mesh_max_p == 1) {
         /* Cull distances require packed-index publication (publication_rounds). */
         mesh_max_v = 4;
         mesh_max_p = 2;
      }
   }

   /* Prepare VS template for inlining (only if GS reads at least one VS output). */
   nir_variable *vs_slot_vars[VARYING_SLOT_MAX] = {NULL};
   struct hk_nir_fused_vs_lower vs_lower = {
      .vs_read_mask = vs_read_mask,
      .vs_slot_vars = vs_slot_vars,
   };
   unsigned vs_instrs = 0;
   if (vs_any_read) {
      vs_lower.in_vertex_id =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "fgs_vid");
      vs_lower.in_vertex_pos =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "fgs_vpos");
      vs_lower.in_instance_id =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "fgs_iid");
      vs_lower.in_first_vertex =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "fgs_fv");
      vs_lower.in_base_instance =
         nir_variable_create(vs, nir_var_shader_temp, glsl_uint_type(), "fgs_bi");
      for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
         if (vs_read_mask[loc]) {
            vs_slot_vars[loc] = nir_variable_create(
               vs, nir_var_shader_temp, glsl_vec4_type(), "fgs_vs_slot");
         }
      }
      NIR_PASS(_, vs, nir_shader_intrinsics_pass, hk_nir_fused_lower_vs_intr,
               nir_metadata_none, &vs_lower);
      NIR_PASS(_, vs, nir_opt_dce);
      NIR_PASS(_, vs, nir_opt_dead_cf);
      nir_foreach_block(block, nir_shader_get_entrypoint(vs)) {
         nir_foreach_instr(instr, block)
            vs_instrs++;
      }
   }

   /* Extract the original GS body so we can place fused setup + VS evaluation
    * cleanly before it inside the active-lane guard. */
   nir_cf_list gs_body;
   nir_cf_extract(&gs_body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *wg = nir_load_workgroup_id(&b);
   nir_def *gp = nir_load_geometry_param_buffer_poly(&b);

   nir_def *index_buffer =
      hk_nir_fused_param_u64(&b, gp, offsetof(struct hk_apple9_fused_gs_params, index_buffer));
   nir_def *index_buffer_size =
      hk_nir_fused_param_u64(&b, gp, offsetof(struct hk_apple9_fused_gs_params, index_buffer_size));
   nir_def *topology =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, input_topology));
   nir_def *index_size_B =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, index_size_B));
   nir_def *first_vertex =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, first_vertex));
   nir_def *base_instance =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, base_instance));
   nir_def *input_prims =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, input_primitives));
   nir_def *first_prim =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, first_primitive));
   nir_def *groups_x =
      hk_nir_fused_param_u32(&b, gp, offsetof(struct hk_apple9_fused_gs_params, groups_x));
   nir_def *first_instance_offset = hk_nir_fused_param_u32(
      &b, gp, offsetof(struct hk_apple9_fused_gs_params, first_instance_offset));

   nir_def *group_linear =
      nir_iadd(&b, nir_imul(&b, nir_channel(&b, wg, 1), groups_x),
               nir_channel(&b, wg, 0));
   nir_def *instance_id =
      nir_iadd(&b, nir_channel(&b, wg, 2), first_instance_offset);
   nir_def *total_runs = nir_imul_imm(&b, input_prims, invocations);
   nir_def *first_run =
      pack > 1 ? nir_imul_imm(&b, group_linear, pack) : group_linear;
   nir_def *run = pack > 1 ? nir_iadd(&b, first_run, lane) : group_linear;
   nir_def *live =
      pack > 1 ? nir_umin(&b, nir_usub_sat(&b, total_runs, first_run),
                          nir_imm_int(&b, pack))
               : NULL;
   nir_def *local_prim =
      invocations > 1 ? nir_udiv_imm(&b, run, invocations) : run;
   nir_def *prim_id = nir_iadd(&b, first_prim, local_prim);
   nir_def *inv_id = invocations > 1 ? nir_umod_imm(&b, run, invocations)
                                     : nir_imm_int(&b, 0);
   nir_def *flatshade_first = nir_ieq_imm(&b, nir_load_provoking_last(&b), 0);

   struct fused_gs_state s = {
      .mesh_prim = mesh_prim,
      .k = k,
      .max_vertices = v_out,
      .clip_size = clip_size,
      .cull_size = cull_size,
      .invocation = inv_id,
      .primitive = prim_id,
      .instance = instance_id,
      .first_vertex = first_vertex,
      .base_instance = base_instance,
      .vertex_base =
         pack > 1 ? nir_imul_imm(&b, lane, v_out) : nir_imm_int(&b, 0),
      .primitive_base =
         pack > 1 ? nir_imul_imm(&b, lane, p_per_run) : nir_imm_int(&b, 0),
      .flatshade_first = flatshade_first,
      .vertices =
         nir_local_variable_create(impl, glsl_uint_type(), "fgs_vertices"),
      .primitives =
         nir_local_variable_create(impl, glsl_uint_type(), "fgs_primitives"),
      .strip = nir_local_variable_create(impl, glsl_uint_type(), "fgs_strip"),
   };
   memcpy(s.masks, out_masks, sizeof(s.masks));
   memcpy(s.types, out_types, sizeof(s.types));

   for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
      if (!s.masks[loc])
         continue;
      s.outputs[loc] =
         nir_local_variable_create(impl, glsl_vec4_type(), "fgs_output");
   }

   /* Allocate inter-stage VS -> GS variables in nir_var_shader_temp so
    * nir_inline_function_impl can remap VS outputs directly into them;
    * nir_lower_global_vars_to_local + nir_lower_vars_to_ssa promote all of
    * them into SSA registers. */
   nir_variable *mesh_in_vid = NULL, *mesh_in_vpos = NULL, *mesh_in_iid = NULL,
                *mesh_in_fv = NULL, *mesh_in_bi = NULL;
   if (vs_any_read) {
      mesh_in_vid =
         nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "fgs_m_vid");
      mesh_in_vpos =
         nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "fgs_m_vpos");
      mesh_in_iid =
         nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "fgs_m_iid");
      mesh_in_fv =
         nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "fgs_m_fv");
      mesh_in_bi =
         nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "fgs_m_bi");
      for (unsigned c = 0; c < k; ++c) {
         for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
            if (vs_read_mask[loc]) {
               s.vs_out[c][loc] = nir_variable_create(
                  nir, nir_var_shader_temp, glsl_vec4_type(), "fgs_vs_out");
            }
         }
      }
   }

   nir_store_var(&b, s.vertices, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.primitives, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.strip, nir_imm_int(&b, 0), 1);

   nir_def *active_cond =
      pack > 1 ? nir_ult(&b, lane, live)
               : nir_iand(&b, nir_ieq_imm(&b, lane, 0),
                          nir_ult(&b, run, total_runs));
   nir_if *active = nir_push_if(&b, active_cond);
   active->control = nir_selection_control_dont_flatten;
   {
      /* Initialize GS staging outputs inside active lane. */
      for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
         if (!s.outputs[loc])
            continue;
         nir_def *init =
            (loc == VARYING_SLOT_CLIP_DIST0 || loc == VARYING_SLOT_CLIP_DIST1)
               ? nir_imm_vec4(&b, 0, 0, 0, 0)
               : nir_undef(&b, 4, 32);
         nir_store_var(&b, s.outputs[loc], init, 15);
      }

      /* Run the cloned VS for the input primitive's K vertices. */
      if (vs_any_read) {
         nir_function_impl *vs_impl = nir_shader_get_entrypoint(vs);
         nir_store_var(&b, mesh_in_iid, instance_id, 1);
         nir_store_var(&b, mesh_in_fv, first_vertex, 1);
         nir_store_var(&b, mesh_in_bi, base_instance, 1);

         const bool loop_vs = k > 1 && vs_instrs > 200;
         if (!loop_vs) {
            for (unsigned c = 0; c < k; ++c) {
               for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
                  if (!s.vs_out[c][loc])
                     continue;
                  nir_def *def = loc == VARYING_SLOT_POS
                                    ? nir_imm_vec4(&b, 0, 0, 0, 1)
                                    : nir_imm_vec4(&b, 0, 0, 0, 0);
                  nir_store_var(&b, s.vs_out[c][loc], def, 15);
               }
               nir_def *vpos = fused_vertex_pos(
                  &b, gs_source->info.gs.input_primitive, topology,
                  flatshade_first, prim_id, nir_imm_int(&b, c));
               nir_def *vid_zb = hk_nir_fused_fetch_vertex_id_zero_base(
                  &b, vpos, index_buffer, index_buffer_size, index_size_B);
               nir_def *vid = nir_iadd(&b, vid_zb, first_vertex);
               nir_store_var(&b, mesh_in_vpos, vid_zb, 1);
               nir_store_var(&b, mesh_in_vid, vid, 1);

               struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
               _mesa_hash_table_insert(remap, vs_lower.in_vertex_id, mesh_in_vid);
               _mesa_hash_table_insert(remap, vs_lower.in_vertex_pos, mesh_in_vpos);
               _mesa_hash_table_insert(remap, vs_lower.in_instance_id, mesh_in_iid);
               _mesa_hash_table_insert(remap, vs_lower.in_first_vertex, mesh_in_fv);
               _mesa_hash_table_insert(remap, vs_lower.in_base_instance, mesh_in_bi);
               for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
                  if (vs_slot_vars[loc])
                     _mesa_hash_table_insert(remap, vs_slot_vars[loc],
                                             s.vs_out[c][loc]);
               }
               nir_inline_function_impl(&b, vs_impl, NULL, remap);
               _mesa_hash_table_destroy(remap, NULL);
            }
         } else {
            nir_variable *loop_out[VARYING_SLOT_MAX] = {NULL};
            for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
               if (vs_read_mask[loc]) {
                  loop_out[loc] = nir_variable_create(
                     nir, nir_var_shader_temp, glsl_vec4_type(), "fgs_loop_out");
               }
            }
            nir_variable *corner_var = nir_local_variable_create(
               impl, glsl_uint_type(), "fgs_corner");
            nir_store_var(&b, corner_var, nir_imm_int(&b, 0), 1);
            nir_loop *loop = nir_push_loop(&b);
            {
               nir_def *c_val = nir_load_var(&b, corner_var);
               nir_push_if(&b, nir_uge_imm(&b, c_val, k));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
                  if (!loop_out[loc])
                     continue;
                  nir_def *def = loc == VARYING_SLOT_POS
                                    ? nir_imm_vec4(&b, 0, 0, 0, 1)
                                    : nir_imm_vec4(&b, 0, 0, 0, 0);
                  nir_store_var(&b, loop_out[loc], def, 15);
               }
               nir_def *vpos = fused_vertex_pos(
                  &b, gs_source->info.gs.input_primitive, topology,
                  flatshade_first, prim_id, c_val);
               nir_def *vid_zb = hk_nir_fused_fetch_vertex_id_zero_base(
                  &b, vpos, index_buffer, index_buffer_size, index_size_B);
               nir_def *vid = nir_iadd(&b, vid_zb, first_vertex);
               nir_store_var(&b, mesh_in_vpos, vid_zb, 1);
               nir_store_var(&b, mesh_in_vid, vid, 1);

               struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
               _mesa_hash_table_insert(remap, vs_lower.in_vertex_id, mesh_in_vid);
               _mesa_hash_table_insert(remap, vs_lower.in_vertex_pos, mesh_in_vpos);
               _mesa_hash_table_insert(remap, vs_lower.in_instance_id, mesh_in_iid);
               _mesa_hash_table_insert(remap, vs_lower.in_first_vertex, mesh_in_fv);
               _mesa_hash_table_insert(remap, vs_lower.in_base_instance, mesh_in_bi);
               for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
                  if (vs_slot_vars[loc])
                     _mesa_hash_table_insert(remap, vs_slot_vars[loc],
                                             loop_out[loc]);
               }
               nir_inline_function_impl(&b, vs_impl, NULL, remap);
               _mesa_hash_table_destroy(remap, NULL);

               for (unsigned c = 0; c < k; ++c) {
                  nir_push_if(&b, nir_ieq_imm(&b, c_val, c));
                  for (unsigned loc = 0; loc < VARYING_SLOT_MAX; ++loc) {
                     if (loop_out[loc])
                        nir_store_var(&b, s.vs_out[c][loc],
                                      nir_load_var(&b, loop_out[loc]), 15);
                  }
                  nir_pop_if(&b, NULL);
               }
               nir_store_var(&b, corner_var, nir_iadd_imm(&b, c_val, 1), 1);
            }
            nir_pop_loop(&b, loop);
         }
      }

      /* Reinsert GS body inside the active-lane guard. */
      b.cursor = nir_cf_reinsert(&gs_body, b.cursor);

      if (pack > 1) {
         /* Packed triangle runs publish fixed per-lane slots; pad unused slots
          * with a finite vertex and degenerate triangles on vertex_base. */
         nir_push_if(&b, nir_ieq_imm(&b, nir_load_var(&b, s.vertices), 0));
         {
            nir_store_per_vertex_output(
               &b, nir_imm_vec4(&b, 0, 0, 0, 1), s.vertex_base,
               nir_imm_int(&b, 0), .write_mask = 15,
               .src_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
            for (unsigned d = 0; d < clip_size; ++d) {
               nir_store_per_vertex_output(
                  &b, nir_imm_float(&b, 0.0f), s.vertex_base, nir_imm_int(&b, 0),
                  .component = d % 4, .write_mask = 1,
                  .src_type = nir_type_float32,
                  .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0 + d / 4,
                                   .num_slots = 1});
            }
         }
         nir_pop_if(&b, NULL);

         nir_def *p = nir_load_var(&b, s.primitives);
         if (p_per_run <= 4) {
            for (unsigned i = 0; i < p_per_run; ++i) {
               nir_push_if(&b, nir_ule_imm(&b, p, i));
               nir_store_per_primitive_output(
                  &b, nir_vec3(&b, s.vertex_base, s.vertex_base, s.vertex_base),
                  nir_iadd_imm(&b, s.primitive_base, i), nir_imm_int(&b, 0),
                  .write_mask = 7, .src_type = nir_type_uint32,
                  .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                   .num_slots = 1});
               nir_pop_if(&b, NULL);
            }
         } else {
            nir_variable *pad_i =
               nir_local_variable_create(impl, glsl_uint_type(), "fgs_pad_i");
            nir_store_var(&b, pad_i, p, 1);
            nir_loop *pad_loop = nir_push_loop(&b);
            {
               nir_def *i = nir_load_var(&b, pad_i);
               nir_push_if(&b, nir_uge_imm(&b, i, p_per_run));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);
               nir_store_per_primitive_output(
                  &b, nir_vec3(&b, s.vertex_base, s.vertex_base, s.vertex_base),
                  nir_iadd(&b, s.primitive_base, i), nir_imm_int(&b, 0),
                  .write_mask = 7, .src_type = nir_type_uint32,
                  .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                   .num_slots = 1});
               nir_store_var(&b, pad_i, nir_iadd_imm(&b, i, 1), 1);
            }
            nir_pop_loop(&b, pad_loop);
         }
      }
   }
   nir_pop_if(&b, active);

   /* Lower GS intrinsics (store_output, emit_vertex, end_primitive,
    * load_per_vertex_input, sysvals) now that VS is already inlined and lowered. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, fused_lower_gs_intr,
            nir_metadata_none, &s);

   /* Publish vertex and primitive counts from lane 0. */
   b.cursor = nir_after_impl(impl);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   if (pack == 1) {
      nir_def *p = nir_load_var(&b, s.primitives);
      nir_def *v = nir_bcsel(&b, nir_ine_imm(&b, p, 0),
                             nir_load_var(&b, s.vertices), nir_imm_int(&b, 0));
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));
   } else {
      nir_set_vertex_and_primitive_count(
         &b, nir_imul_imm(&b, live, v_out), nir_imul_imm(&b, live, p_per_run),
         nir_imm_int(&b, 0));
   }
   nir_pop_if(&b, NULL);

   ralloc_free(vs);

   memset(&nir->info.mesh, 0, sizeof(nir->info.mesh));
   nir->info.stage = MESA_SHADER_MESH;
   nir->info.mesh.primitive_type = mesh_prim;
   nir->info.mesh.max_vertices_out = mesh_max_v;
   nir->info.mesh.max_primitives_out = mesh_max_p;
   nir->info.workgroup_size[0] = mesh_threads;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;

   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_dead_cf);
   nir_shader_gather_info(nir, impl);
   /* nir_shader_gather_info clears clip/cull array sizes on lowered IO. */
   nir->info.clip_distance_array_size = clip_size;
   nir->info.cull_distance_array_size = cull_size;
   nir_validate_shader(nir, "fused VS+GS native mesh lowering");

   if (out_info) {
      *out_info = (struct hk_gs_fused_info){
         .pack = pack,
         .input_primitive = gs_source->info.gs.input_primitive,
         .output_primitive = gs_source->info.gs.output_primitive,
         .invocations = invocations,
         .prims_per_run = p_per_run,
      };
   }
   return nir;
}
