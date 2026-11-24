/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"

static bool
fail(const char **reason, const char *message)
{
   if (reason)
      *reason = message;
   return false;
}

bool
agx_apple9_native_mesh_shape_eligible(const nir_shader *nir)
{
   return nir->info.stage == MESA_SHADER_MESH && !nir->info.task_payload_size &&
          nir->info.mesh.max_vertices_out == 3 &&
          nir->info.mesh.max_primitives_out == 1 &&
          nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES;
}

static bool
eligible(nir_shader *nir, unsigned physical_threads, bool packed_indices,
         bool allow_task_payload, const char **reason)
{
   if (allow_task_payload &&
       nir->info.task_payload_size > AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES)
      return fail(reason, "native mesh task payload exceeds 16 KiB hardware limit");
   if (packed_indices) {
      /* This compiler option does not widen the driver's native shape
       * predicate. Publication repeats in rounds of at most
       * AGX_APPLE9_PUBLICATION_ROUND entries, so the API maxima are covered,
       * subject to the local output allocation. */
      if (nir->info.stage != MESA_SHADER_MESH ||
          (!allow_task_payload && nir->info.task_payload_size) ||
          !agx_apple9_mesh_indices_per_primitive(nir->info.mesh.primitive_type) ||
          !nir->info.mesh.max_vertices_out ||
          nir->info.mesh.max_vertices_out > 256 ||
          !nir->info.mesh.max_primitives_out ||
          nir->info.mesh.max_primitives_out > 256)
         return fail(reason, "packed index candidate requires V1..256 P1..256 points, lines or triangles");
   } else if (nir->info.stage != MESA_SHADER_MESH ||
              (!allow_task_payload && nir->info.task_payload_size) ||
              nir->info.mesh.max_vertices_out != 3 ||
              nir->info.mesh.max_primitives_out != 1 ||
              nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES) {
      return fail(reason,
                  "native mesh development class requires V3 P1 triangles without payload");
   }
   if (physical_threads == AGX_APPLE9_NATIVE_THREADS) {
      if (!agx_apple9_native_workgroup_supported(nir, reason))
         return false;
   } else {
      if (physical_threads != AGX_APPLE9_NATIVE_SERIAL_THREADS)
         return fail(reason, "native mesh physical workgroup must be T16 or T32");
      if (!agx_apple9_native_workgroup_valid(
             nir, AGX_APPLE9_NATIVE_MAX_INVOCATIONS, reason))
         return false;
   }

   /* SetMeshOutputs operands need not be constants, and mutually exclusive
    * branches may each contain a call. The body lowerer checks constant
    * counts against the declaration and bounds runtime counts before
    * publishing lane zero's normalized values.
    * Vulkan requires at most one executed call under uniform control flow. */
   return true;
}

/* Point size does not participate in line or triangle rasterization. Strip
 * constant-offset stores before body IO validation so shaders writing
 * gl_PointSize in gl_PerVertex are not rejected; indirect offsets and loads
 * remain for the IO validator to decline. */
static bool
drop_unused_point_size(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;
   if (!nir_src_is_const(intr->src[2]))
      return false;
   uint64_t slot = nir_intrinsic_io_semantics(intr).location +
                   (uint64_t)nir_src_as_uint(intr->src[2]);
   if (slot != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* Apple9 hardware drops points with size >= 512.0 and requires >= 1.0.
 * Clamp authored gl_PointSize stores to [1.0, 511.9375], matching the
 * vertex shader path (apple9_lower_point_size). */
static bool
clamp_point_size(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;
   if (!nir_src_is_const(intr->src[2]))
      return false;
   uint64_t slot = nir_intrinsic_io_semantics(intr).location +
                   (uint64_t)nir_src_as_uint(intr->src[2]);
   if (slot != VARYING_SLOT_PSIZ || intr->num_components != 1 ||
       nir_intrinsic_component(intr) || nir_src_bit_size(intr->src[0]) != 32)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *size =
      nir_fmin(b, nir_fmax(b, intr->src[0].ssa, nir_imm_float(b, 1.0f)),
               nir_imm_float(b, 511.9375f));
   nir_src_rewrite(&intr->src[0], size);
   return true;
}

/* Keep the actual vertex count in compiler-owned local storage. The body
 * lowerer leaves the count marker so completion can consume both operands. */
static bool
consume_count(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_mesh_body_info *info = data;
   if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_local_apple9_agx(b, intr->src[0].ssa,
                                 nir_imm_int(b, info->layout.vertex_count_B),
                                 .base = info->key.output_root, .bit_size = 32);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

nir_def *
agx_nir_apple9_index_publication_lanes(nir_builder *b, nir_def *primitives,
                                       unsigned k)
{
   assert(k >= 1 && k <= 3);
   nir_def *bytes = nir_imul_imm(b, primitives, k);
   /* Counts are normalized before publication; one lane retains the empty
    * handoff used by the existing native path. */
   return nir_umax_imm(b, nir_ushr_imm(b, nir_iadd_imm(b, bytes, 3), 2), 1);
}

static void
device_fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

/* With one primitive, every referenced vertex has the same primitive value.
 * Keep body reads/writes in vertex zero's local plane, then broadcast only after
 * all body writes complete. This is an ordinary flat-varying lowering, separate
 * from the unresolved native primitive-record linking plane.
 */
struct primitive_lower {
   uint64_t mask;
   unsigned root, cull_B;
   unsigned vertex_base;
};

static bool
lower_primitive_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct primitive_lower *lower = data;
   uint64_t mask = lower->mask;
   bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output;
   if (!store && intr->intrinsic != nir_intrinsic_load_per_primitive_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned slot = sem.location + nir_src_as_uint(intr->src[store ? 2 : 1]);
   if (slot >= 64 || !(mask & BITFIELD64_BIT(slot)))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
      /* A compiler-owned word in the count header, separate from both native
       * primitive records and the fragment compiler's CullDistance transport. */
      nir_def *address = nir_imm_int(b, lower->cull_B);
      if (store) {
         nir_def *value = intr->src[0].ssa;
         if (value->bit_size == 1)
            value = nir_b2i32(b, value);
         nir_store_local_apple9_agx(b, value, address,
                                    .base = lower->root, .bit_size = 32);
         nir_instr_remove(&intr->instr);
      } else {
         nir_def_replace(&intr->def,
            nir_load_local_apple9_agx(b, address, .base = lower->root));
      }
      return true;
   }
   sem.location = slot;
   sem.num_slots = 1;
   nir_def *vertex = lower->vertex_base
      ? nir_iadd_imm(b, intr->src[store ? 1 : 0].ssa, lower->vertex_base)
      : nir_imm_int(b, 0);
   if (store) {
      nir_def *value = intr->src[0].ssa;
      nir_store_per_vertex_output(
         b, value, vertex, nir_imm_int(b, 0),
         .base = nir_intrinsic_base(intr), .range = 1,
         .component = nir_intrinsic_component(intr),
         .write_mask = nir_intrinsic_write_mask(intr),
         .src_type = nir_intrinsic_src_type(intr),
         .io_semantics = sem);
      nir_instr_remove(&intr->instr);
   } else {
      nir_def *value = nir_load_per_vertex_output(
         b, intr->def.num_components, 32, vertex, nir_imm_int(b, 0),
         .base = nir_intrinsic_base(intr), .range = 1,
         .component = nir_intrinsic_component(intr),
         .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
      nir_def_replace(&intr->def, value);
   }
   return true;
}

static bool
primitive_outputs(nir_shader *nir, const struct agx_apple9_mesh_key *key,
                  bool duplicate, bool allow_layer_viewport, uint64_t *mask,
                  const char **reason)
{
   if (!duplicate && (nir->info.mesh.max_primitives_out != 1 ||
                      nir->info.mesh.max_vertices_out > 3))
      return fail(reason, "primitive broadcast requires P1 and at most V3");

   uint64_t vertex = 0;
   *mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (duplicate &&
                intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count &&
                nir_src_is_const(intr->src[0]) &&
                nir_src_as_uint(intr->src[0]) > nir->info.mesh.max_vertices_out)
               return fail(reason, "primitive vertex counts exceed the original declaration");
            bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                         intr->intrinsic == nir_intrinsic_store_per_vertex_output;
            bool prim = intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                        intr->intrinsic == nir_intrinsic_load_per_primitive_output;
            if (!store && !prim && intr->intrinsic != nir_intrinsic_load_per_vertex_output)
               continue;
            unsigned offset = store ? 2 : 1;
            if (!nir_src_is_const(intr->src[offset]))
               return fail(reason, "primitive broadcast requires constant IO slots");
            uint64_t slot = nir_intrinsic_io_semantics(intr).location +
                            (uint64_t)nir_src_as_uint(intr->src[offset]);
            if (slot >= 64)
               return fail(reason, "primitive broadcast IO slot is unsupported");
            if (!prim) {
               vertex |= BITFIELD64_BIT(slot);
               continue;
            }
            if (slot == VARYING_SLOT_PRIMITIVE_INDICES)
               continue;
            bool user = slot >= VARYING_SLOT_VAR0 && slot <= VARYING_SLOT_VAR31;
            bool cull = slot == VARYING_SLOT_CULL_PRIMITIVE;
            bool layer_vp = allow_layer_viewport &&
                            (slot == VARYING_SLOT_LAYER ||
                             slot == VARYING_SLOT_VIEWPORT);
            if (duplicate && cull)
               return fail(reason, "primitive vertex copies require separately lowered culling");
            if (!user && !cull && !layer_vp && slot != VARYING_SLOT_PRIMITIVE_ID)
               return fail(reason, "primitive broadcast builtin is unsupported");
            unsigned bits = store ? nir_src_bit_size(intr->src[0]) : intr->def.bit_size;
            if (bits != 32 && !(cull && store && bits == 1))
               return fail(reason, "primitive broadcast requires 32-bit components");
            if (!user && (intr->num_components != 1 || nir_intrinsic_component(intr)))
               return fail(reason, "primitive broadcast builtin must be scalar");
            *mask |= BITFIELD64_BIT(slot);
         }
      }
   }
   if (!*mask)
      return true;
   if (duplicate && nir->info.mesh.max_vertices_out +
                    nir->info.mesh.max_primitives_out > 256)
      return fail(reason, "primitive vertex copies exceed 256 native vertices");
   if (*mask & vertex)
      return fail(reason, "primitive and vertex outputs cannot share a broadcast slot");
   struct primitive_lower lower = {
      .mask = *mask, .root = key->output_root,
      .cull_B = agx_apple9_mesh_primitive_count_offset(
                   nir->info.mesh.max_primitives_out,
                   agx_apple9_mesh_indices_per_primitive(
                      nir->info.mesh.primitive_type)) + 8,
      .vertex_base = duplicate ? nir->info.mesh.max_vertices_out : 0,
   };
   nir_shader_intrinsics_pass(nir, lower_primitive_output, nir_metadata_none, &lower);
   nir->info.per_primitive_outputs &= ~*mask;
   if (duplicate) {
      /* Keep runtime count normalization bounded by the application's
       * original declaration, before expanding the allocation. */
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
                  continue;
               nir_builder b = nir_builder_at(nir_before_instr(instr));
               nir_src_rewrite(&intr->src[0], nir_umin_imm(
                  &b, intr->src[0].ssa, nir->info.mesh.max_vertices_out));
            }
         }
      }
      nir->info.mesh.max_vertices_out += nir->info.mesh.max_primitives_out;
   }
   return true;
}

/* A flat per-primitive value can use the normal varying plane when the
 * provoking vertex belongs only to that primitive. Preserve every original
 * vertex component (including flat per-vertex attributes and clip mirrors),
 * and change only that corner's index. Smooth interpolation is unchanged.
 *
 * Primitive body stores start at the declared vertex maximum. The compact
 * destination starts at the actual vertex count, so it can overlap those
 * stores. Read each round's primitive values before any lane overwrites them;
 * ascending rounds cannot overwrite a later round's source. Original vertex
 * rows always precede all destinations and need no temporary copy.
 */
static nir_def *
duplicate_primitive_vertices(nir_builder *b,
                             const struct agx_apple9_mesh_body_info *info,
                             nir_def *lane, nir_def *vertices,
                             nir_def *primitives, unsigned threads)
{
   const struct agx_apple9_mesh_layout *l = &info->layout;
   unsigned root = info->key.output_root;
   nir_def *corner = nir_bcsel(b, nir_ine_imm(b, nir_load_provoking_last(b), 0),
                              nir_imm_int(b, l->indices_per_primitive - 1),
                              nir_imm_int(b, 0));
   for (unsigned first = 0; first < l->max_primitives; first += threads) {
      nir_def *p = nir_iadd_imm(b, lane, first);
      nir_def *active = nir_ult(b, p, primitives);
      nir_def *dst = nir_iadd(b, vertices, p);
      nir_def *byte = nir_iadd(b, nir_imul_imm(b, p, l->indices_per_primitive), corner);
      nir_def *value[VARYING_SLOT_MAX][4] = {0};
      nir_def *undefined = nir_undef(b, 1, 32);
      nir_push_if(b, active);
      nir_def *word = nir_load_local_apple9_agx(
         b, nir_iand_imm(b, byte, ~3), .base = root);
      nir_def *vertex = nir_iand_imm(b, nir_ushr(b, word,
         nir_imul_imm(b, nir_iand_imm(b, byte, 3), 8)), 0xff);
      for (unsigned slot = 0; slot < VARYING_SLOT_MAX; ++slot) {
         bool primitive = slot < 64 && (info->primitive_vertices & BITFIELD64_BIT(slot));
         /* A reserved Layer plane (when only ViewportIndex was authored) is
          * populated by pack_layer_viewport after duplication completes. */
         if (slot == VARYING_SLOT_LAYER && !primitive)
            continue;
         for (unsigned c = 0; c < 4; ++c) {
            unsigned component = l->vertex_component[slot][c];
            if (component == AGX_APPLE9_MESH_UNUSED)
               continue;
            unsigned plane = agx_apple9_mesh_plane_offset(l, slot, c);
            nir_def *src = primitive ? nir_iadd_imm(b, p, info->original_max_vertices)
                                     : vertex;
            nir_def *v = nir_load_local_apple9_agx(
               b, nir_iadd_imm(b, nir_imul_imm(b, src, 4), plane), .base = root);
            if (primitive)
               value[slot][c] = v;
            else
               nir_store_local_apple9_agx(
                  b, v, nir_iadd_imm(b, nir_imul_imm(b, dst, 4), plane),
                  .base = root, .bit_size = 32);
         }
      }
      nir_pop_if(b, NULL);
      u_foreach_bit64(slot, info->primitive_vertices) {
         for (unsigned c = 0; c < 4; ++c) {
            if (value[slot][c])
               value[slot][c] = nir_if_phi(b, value[slot][c], undefined);
         }
      }
      agx_nir_apple9_mesh_workgroup_barrier(b);
      nir_push_if(b, active);
      u_foreach_bit64(slot, info->primitive_vertices) {
         for (unsigned c = 0; c < 4; ++c) {
            if (!value[slot][c])
               continue;
            unsigned plane = agx_apple9_mesh_plane_offset(l, slot, c);
            nir_store_local_apple9_agx(
               b, value[slot][c], nir_iadd_imm(b, nir_imul_imm(b, dst, 4), plane),
               .base = root, .bit_size = 32);
         }
      }
      nir_store_local_apple9_agx(b, dst, byte, .base = root, .bit_size = 8);
      nir_pop_if(b, NULL);
   }
   agx_nir_apple9_mesh_workgroup_barrier(b);
   return nir_iadd(b, vertices, primitives);
}

static void
broadcast_primitive(nir_builder *b, const struct agx_apple9_mesh_body_info *info,
                    nir_def *lane, nir_def *vertices)
{
   const struct agx_apple9_mesh_layout *l = &info->layout;
   unsigned root = info->key.output_root;
   nir_push_if(b, nir_ult(b, lane, vertices));
   u_foreach_bit64(slot, info->primitive_broadcast) {
      for (unsigned c = 0; c < 4; ++c) {
         if (l->vertex_component[slot][c] == AGX_APPLE9_MESH_UNUSED)
            continue;
         unsigned plane = agx_apple9_mesh_plane_offset(l, slot, c);
         nir_def *value = nir_load_local_apple9_agx(b, nir_imm_int(b, plane), .base = root);
         nir_store_local_apple9_agx(
            b, value, nir_iadd_imm(b, nir_imul_imm(b, lane, 4), plane),
            .base = root, .bit_size = 32);
      }
   }
   nir_pop_if(b, NULL);
   if (info->primitive_broadcast & VARYING_BIT_CULL_PRIMITIVE) {
      nir_push_if(b, nir_ieq_imm(b, lane, 0));
      nir_def *address = nir_imm_int(b, l->primitive_count_B + 8);
      nir_def *cull = nir_load_local_apple9_agx(
         b, address, .base = root);
      nir_push_if(b, nir_ine_imm(b, cull, 0));
      for (unsigned c = 0; c < 3; ++c)
         nir_store_local_apple9_agx(b, nir_imm_int(b, 0), nir_imm_int(b, c),
                                    .base = root, .bit_size = 8);
      nir_pop_if(b, NULL);
      /* Restore the count header before publication: the hardware's use of
       * this word is unknown, so the compiler-owned flag must not escape. */
      nir_store_local_apple9_agx(b, nir_imm_int(b, 0), address,
                                 .base = root, .bit_size = 32);
      nir_pop_if(b, NULL);
   }
   agx_nir_apple9_mesh_workgroup_barrier(b);
}

/* Pack the rasterizer's shared Layer/Viewport word into VARYING_SLOT_LAYER:
 * layer in the low 16 bits, viewport index in the high 16 bits, matching
 * apple9_lower_layer_viewport_output() in the vertex path. */
static void
pack_layer_viewport(nir_builder *b, const struct agx_apple9_mesh_body_info *info,
                    nir_def *lane, nir_def *vertices, unsigned physical_threads)
{
   const struct agx_apple9_mesh_layout *l = &info->layout;
   unsigned root = info->key.output_root;
   unsigned layer_plane = agx_apple9_mesh_plane_offset(l, VARYING_SLOT_LAYER, 0);
   bool has_layer =
      (info->primitive_broadcast | info->primitive_vertices) & VARYING_BIT_LAYER;
   bool has_vp = l->vertex_component[VARYING_SLOT_VIEWPORT][0] != AGX_APPLE9_MESH_UNUSED;
   unsigned vp_plane =
      has_vp ? agx_apple9_mesh_plane_offset(l, VARYING_SLOT_VIEWPORT, 0) : 0;

   for (unsigned first = 0; first < l->max_vertices; first += physical_threads) {
      nir_def *v = nir_iadd_imm(b, lane, first);
      nir_push_if(b, nir_ult(b, v, vertices));
      nir_def *off = nir_imul_imm(b, v, 4);
      nir_def *word = nir_imm_int(b, 0);
      if (has_layer) {
         nir_def *layer = nir_load_local_apple9_agx(
            b, nir_iadd_imm(b, off, layer_plane), .base = root);
         word = nir_iand_imm(b, layer, 0xffff);
      }
      if (has_vp) {
         nir_def *vp = nir_load_local_apple9_agx(
            b, nir_iadd_imm(b, off, vp_plane), .base = root);
         nir_def *shifted = nir_ishl_imm(b, vp, 16);
         word = has_layer ? nir_ior(b, word, shifted) : shifted;
      }
      nir_store_local_apple9_agx(b, word, nir_iadd_imm(b, off, layer_plane),
                                 .base = root, .bit_size = 32);
      nir_pop_if(b, NULL);
   }
   agx_nir_apple9_mesh_workgroup_barrier(b);
}

/* Local planes stay in semantic/component order for shader output reads.
 * U channels use the fragment linker's interpolation groups instead. */
static bool
varying_layout(const struct agx_apple9_mesh_body_info *info,
               const struct agx_apple9_native_mesh_options *options,
               struct agx_apple9_varying_layout *varyings,
               const char **reason)
{
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   memset(varyings, 0, sizeof(*varyings));
   if (layout->primitive_components || layout->culling)
      return fail(reason,
                  "native mesh primitive attributes/culling are not implemented");
   unsigned count = 0;
   uint64_t flat = (options ? options->outputs_flat : 0) |
                   info->primitive_broadcast | info->primitive_vertices |
                   VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT;
   uint64_t linear = options ? options->outputs_linear : 0;
   for (unsigned slot = 0; slot < VARYING_SLOT_MAX; ++slot) {
      /* Link clip distances only when the fragment stage reads them. The
       * rasterizer then consumes distinct mirror planes; excluding those
       * from the varyings keeps the stride equal to the PPP output count.
       * PointSize and the packed Layer/Viewport word are rasterizer-only. */
      bool clip = slot == VARYING_SLOT_CLIP_DIST0 || slot == VARYING_SLOT_CLIP_DIST1;
      bool mirror = slot == AGX_APPLE9_CLIP_MIRROR_SLOT ||
                    slot == AGX_APPLE9_CLIP_MIRROR_SLOT + 1;
      if (slot == VARYING_SLOT_POS || slot == VARYING_SLOT_PSIZ ||
          slot == VARYING_SLOT_LAYER || (clip && !info->clip_linked) ||
          (mirror && info->clip_linked))
         continue;
      for (unsigned c = 0; c < 4; ++c) {
         if (layout->vertex_component[slot][c] == AGX_APPLE9_MESH_UNUSED)
            continue;
         if (slot >= ARRAY_SIZE(varyings->mask) ||
             !agx_apple9_varying_supported(slot))
            return fail(reason, "native mesh varying location is unsupported");
         varyings->mask[slot] |= BITFIELD_BIT(c);
         ++count;
      }
      if (slot < ARRAY_SIZE(varyings->mask) && varyings->mask[slot])
         varyings->group[slot] = agx_apple9_varying_group(flat, linear, slot);
   }
   if (count && (!options || !options->attributes))
      return fail(reason, "native mesh attributes development path is disabled");
   if (count > AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return fail(reason, "native mesh exceeds 124 user varying scalars");
   varyings->count = count;
   return true;
}

/* Per-vertex cull distances share CLIP_DIST0/1 with the clip distances,
 * after them. They are not rasterizer outputs on the native path: move them
 * into compiler-owned shared storage at state->base, one float per vertex
 * and plane, so the output planes keep matching the PPP output count. */
struct cull_storage {
   unsigned clip, cull, base;
   bool keep_outputs, failed;
};

static bool
move_cull_distance(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct cull_storage *st = data;
   bool store = intr->intrinsic == nir_intrinsic_store_per_vertex_output;
   if (!store && intr->intrinsic != nir_intrinsic_load_per_vertex_output &&
       intr->intrinsic != nir_intrinsic_load_output &&
       intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != VARYING_SLOT_CLIP_DIST0 && location != VARYING_SLOT_CLIP_DIST1)
      return false;
   nir_src *offset_src = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*offset_src)) {
      st->failed = true;
      return false;
   }
   unsigned first = 4 * (location - VARYING_SLOT_CLIP_DIST0 +
                         nir_src_as_uint(*offset_src)) +
                    nir_intrinsic_component(intr);
   unsigned mask = store ? nir_intrinsic_write_mask(intr)
                         : BITFIELD_MASK(intr->def.num_components);
   unsigned cull_mask = 0;
   u_foreach_bit(c, mask) {
      if (first + c >= st->clip && first + c < st->clip + st->cull)
         cull_mask |= BITFIELD_BIT(c);
   }
   if (!cull_mask)
      return false;
   /* Reading back a cull distance, or writing one from a non-vertex
    * output, keeps the emulated path. */
   if (!store) {
      st->failed = true;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *vertex = intr->src[1].ssa;
   u_foreach_bit(c, cull_mask) {
      unsigned plane = first + c - st->clip;
      nir_def *offset = nir_imul_imm(b, nir_iadd_imm(b, nir_imul_imm(b, vertex, st->cull),
                                                     plane), 4);
      nir_store_shared(b, nir_channel(b, intr->src[0].ssa, c), offset,
                       .base = st->base, .align_mul = 4);
   }
   if (!st->keep_outputs) {
      if (mask & ~cull_mask)
         nir_intrinsic_set_write_mask(intr, mask & ~cull_mask);
      else
         nir_instr_remove(&intr->instr);
   }
   return true;
}

/* Mirror each clip-distance store into the private slots, so the linked
 * varying and the rasterizer copy have separate planes and the mesh plane
 * count stays equal to the PPP output count. */
struct clip_mirror {
   unsigned distances;
   bool failed;
};

static bool
mirror_clip_distance(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct clip_mirror *clip = data;
   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_CLIP_DIST0 && sem.location != VARYING_SLOT_CLIP_DIST1)
      return false;
   /* This pass runs before the general output-IO validator. An unsupported
    * indirect slot must decline the native path before reading an immediate. */
   if (!nir_src_is_const(intr->src[2])) {
      clip->failed = true;
      return false;
   }
   uint64_t first = 4 * ((uint64_t)sem.location - VARYING_SLOT_CLIP_DIST0 +
                        nir_src_as_uint(intr->src[2])) + nir_intrinsic_component(intr);
   b->cursor = nir_after_instr(&intr->instr);
   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      uint64_t d = first + c;
      if (d >= clip->distances)
         continue;
      nir_io_semantics mirror = {
         .location = AGX_APPLE9_CLIP_MIRROR_SLOT + d / 4, .num_slots = 1,
      };
      nir_store_per_vertex_output(b, nir_channel(b, intr->src[0].ssa, c), intr->src[1].ssa,
                                  nir_imm_int(b, 0), .component = d % 4, .write_mask = 1,
                                  .src_type = nir_type_float32, .io_semantics = mirror);
   }
   return true;
}

struct primitive_cull_storage {
   unsigned base;
   bool written, failed;
};

static bool
move_primitive_cull(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct primitive_cull_storage *st = data;
   bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output;
   if ((!store && intr->intrinsic != nir_intrinsic_load_per_primitive_output) ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_CULL_PRIMITIVE)
      return false;

   /* EXT mesh output readback is not supported. Validate before reading an
    * immediate so a capability probe rejects unsupported IO without asserting. */
   if (!store || !nir_src_is_const(intr->src[2]) ||
       nir_src_as_uint(intr->src[2]) || intr->num_components != 1 ||
       nir_intrinsic_component(intr) || nir_intrinsic_write_mask(intr) != 1 ||
       (nir_src_bit_size(intr->src[0]) != 1 &&
        nir_src_bit_size(intr->src[0]) != 32)) {
      st->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = intr->src[0].ssa;
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_store_shared(b, value, nir_imul_imm(b, intr->src[1].ssa, 4),
                    .base = st->base, .align_mul = 4, .write_mask = 1);
   nir_instr_remove(&intr->instr);
   st->written = true;
   return true;
}

/* CullPrimitive and CullDistance are independent reasons to cull a triangle.
 * Cull distances cull when all three vertices are negative for some plane.
 * Culled primitives become degenerate (i0, i0, i0) before index publication.
 * Draw admission restricts this software culling to filled triangles. */
static void
cull_primitives(nir_builder *b, const struct agx_apple9_mesh_body_info *info,
                nir_def *lane, nir_def *primitives, unsigned physical_threads)
{
   const struct agx_apple9_mesh_layout *l = &info->layout;
   unsigned root = info->key.output_root;
   unsigned cull_B = info->shared_offset_B + info->cull_B;
   for (unsigned first = 0; first < l->max_primitives; first += physical_threads) {
      nir_def *p = nir_iadd_imm(b, lane, first);
      nir_push_if(b, nir_ult(b, p, primitives));
      nir_def *byte = nir_imul_imm(b, p, 3);
      nir_def *word = nir_iand_imm(b, byte, ~3);
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, byte, 3), 8);
      nir_def *lo = nir_load_local_apple9_agx(b, word, .base = root);
      nir_def *hi = nir_load_local_apple9_agx(b, nir_iadd_imm(b, word, 4), .base = root);
      nir_def *packed = nir_bcsel(b, nir_ieq_imm(b, shift, 0), lo,
         nir_ior(b, nir_ushr(b, lo, shift),
                    nir_ishl(b, hi, nir_isub(b, nir_imm_int(b, 32), shift))));
      nir_def *v[3];
      for (unsigned i = 0; i < 3; ++i)
         v[i] = nir_iand_imm(b, nir_ushr_imm(b, packed, 8 * i), 0xff);
      nir_def *culled = nir_imm_false(b);
      if (info->cull_primitive) {
         nir_def *offset = nir_iadd_imm(
            b, nir_imul_imm(b, p, 4),
            info->shared_offset_B + info->cull_primitive_B);
         culled = nir_ine_imm(
            b, nir_load_local_apple9_agx(b, offset, .base = root), 0);
      }
      for (unsigned d = 0; d < info->cull_distances; ++d) {
         nir_def *all = nir_imm_true(b);
         for (unsigned i = 0; i < 3; ++i) {
            nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, v[i], 4 * info->cull_distances),
                                           cull_B + 4 * d);
            nir_def *value = nir_load_local_apple9_agx(b, offset, .base = root);
            all = nir_iand(b, all, nir_flt_imm(b, value, 0));
         }
         culled = nir_ior(b, culled, all);
      }
      nir_push_if(b, culled);
      for (unsigned i = 1; i < 3; ++i)
         nir_store_local_apple9_agx(b, v[0], nir_iadd_imm(b, byte, i), .base = root,
                                    .bit_size = 8);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   agx_nir_apple9_mesh_workgroup_barrier(b);
}

/* nir_shader_gather_info resets clip/cull_distance_array_size when IO is
 * already lowered without nir_var_shader_out variables. Preserve the
 * remaining rasterizer clip-distance count across internal re-gathers. */
static void
gather_mesh_info_preserving_clips(nir_shader *nir, unsigned clip_distances)
{
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = clip_distances;
   nir->info.cull_distance_array_size = 0;
}

static nir_def *
emit_const_slot_io(nir_builder *b, nir_intrinsic_instr *orig, unsigned slot_off)
{
   nir_intrinsic_instr *clone =
      nir_intrinsic_instr_create(b->shader, orig->intrinsic);
   clone->num_components = orig->num_components;
   nir_intrinsic_copy_const_indices(clone, orig);
   nir_io_semantics sem = nir_intrinsic_io_semantics(orig);
   sem.location += slot_off;
   sem.num_slots = 1;
   nir_intrinsic_set_io_semantics(clone, sem);
   if (nir_intrinsic_has_base(clone))
      nir_intrinsic_set_base(clone, nir_intrinsic_base(orig) + slot_off);
   if (nir_intrinsic_has_range(clone))
      nir_intrinsic_set_range(clone, 1);

   bool store = orig->intrinsic == nir_intrinsic_store_per_vertex_output ||
                orig->intrinsic == nir_intrinsic_store_per_primitive_output;
   if (store) {
      clone->src[0] = nir_src_for_ssa(orig->src[0].ssa);
      clone->src[1] = nir_src_for_ssa(orig->src[1].ssa);
      clone->src[2] = nir_src_for_ssa(nir_imm_int(b, 0));
      nir_builder_instr_insert(b, &clone->instr);
      return NULL;
   }

   clone->src[0] = nir_src_for_ssa(orig->src[0].ssa);
   clone->src[1] = nir_src_for_ssa(nir_imm_int(b, 0));
   nir_def_init(&clone->instr, &clone->def, orig->def.num_components,
                orig->def.bit_size);
   nir_builder_instr_insert(b, &clone->instr);
   return &clone->def;
}

static nir_def *
emit_slot_tree(nir_builder *b, nir_intrinsic_instr *orig, nir_def *offset,
               unsigned start, unsigned end)
{
   assert(start < end);
   if (start == end - 1)
      return emit_const_slot_io(b, orig, start);

   unsigned mid = start + (end - start) / 2;
   nir_push_if(b, nir_ult_imm(b, offset, mid));
   nir_def *then_def = emit_slot_tree(b, orig, offset, start, mid);
   nir_push_else(b, NULL);
   nir_def *else_def = emit_slot_tree(b, orig, offset, mid, end);
   nir_pop_if(b, NULL);
   return then_def ? nir_if_phi(b, then_def, else_def) : NULL;
}

/* Native mesh links a fixed component map, so per-vertex/per-primitive
 * output-array slot offsets must be constant while dynamic vertex/primitive
 * indices stay untouched. Lower only non-constant user-varying slot offsets
 * strictly on the native clone so emulated fallback NIR never pays for
 * if-else trees on vertex/primitive indexing. */
static bool
lower_indirect_io_slots(nir_shader *nir)
{
   struct util_dynarray work = UTIL_DYNARRAY_INIT;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            bool store = intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
                         intr->intrinsic == nir_intrinsic_store_per_primitive_output;
            bool load = intr->intrinsic == nir_intrinsic_load_per_vertex_output ||
                        intr->intrinsic == nir_intrinsic_load_per_primitive_output;
            if (!store && !load)
               continue;
            unsigned offset_src = store ? 2 : 1;
            if (nir_src_is_const(intr->src[offset_src]))
               continue;
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (sem.location < VARYING_SLOT_VAR0 || sem.num_slots <= 1 ||
                sem.location + sem.num_slots > VARYING_SLOT_VAR0 + 32)
               continue;
            util_dynarray_append(&work, intr);
         }
      }
   }
   if (!util_dynarray_num_elements(&work, nir_intrinsic_instr *)) {
      util_dynarray_fini(&work);
      return false;
   }

   util_dynarray_foreach(&work, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      bool store = intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
                   intr->intrinsic == nir_intrinsic_store_per_primitive_output;
      unsigned offset_src = store ? 2 : 1;
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      nir_builder b = nir_builder_at(nir_before_instr(&intr->instr));
      nir_def *offset = intr->src[offset_src].ssa;

      if (store) {
         nir_push_if(&b, nir_ult_imm(&b, offset, sem.num_slots));
         emit_slot_tree(&b, intr, offset, 0, sem.num_slots);
         nir_pop_if(&b, NULL);
         nir_instr_remove(&intr->instr);
      } else {
         nir_push_if(&b, nir_ult_imm(&b, offset, sem.num_slots));
         nir_def *in_bounds = emit_slot_tree(&b, intr, offset, 0, sem.num_slots);
         nir_push_else(&b, NULL);
         nir_def *oob =
            nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size);
         nir_pop_if(&b, NULL);
         nir_def_replace(&intr->def, nir_if_phi(&b, in_bounds, oob));
      }
   }
   util_dynarray_fini(&work);

   unsigned saved_clip = nir->info.clip_distance_array_size;
   unsigned saved_cull = nir->info.cull_distance_array_size;
   nir_foreach_function_impl(impl, nir)
      nir_progress(true, impl, nir_metadata_none);
   nir_opt_constant_folding(nir);
   nir_opt_dce(nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = saved_clip;
   nir->info.cull_distance_array_size = saved_cull;
   return true;
}

struct native_child_grid_lower {
   unsigned payload_uniform;
   unsigned grid_offset_B;
   bool hw_payload;
   bool lowered;
};

static bool
lower_native_mesh_child_num_workgroups(nir_builder *b,
                                       nir_intrinsic_instr *intr, void *data)
{
   struct native_child_grid_lower *st = data;
   if (intr->intrinsic != nir_intrinsic_load_num_workgroups)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = nir_pack_64_2x32(
      b, st->hw_payload
            ? nir_load_mesh_payload_base_apple9_agx(b)
            : nir_load_native_pointer_apple9_agx(b, .base = st->payload_uniform));
   nir_def *addr = nir_iadd_imm(b, base, st->grid_offset_B);
   nir_def *nwg = nir_load_global(b, 3, 32, addr, .align_mul = 4,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def_replace(&intr->def, nir_u2uN(b, nwg, intr->def.bit_size));
   st->lowered = true;
   return true;
}

bool
agx_nir_lower_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_apple9_mesh_body_info *info, const char **reason)
{
   if (!enable_native_mesh)
      return fail(reason, "native mesh development path is disabled");
   unsigned physical_threads = options && options->physical_threads
                                  ? options->physical_threads
                                  : AGX_APPLE9_NATIVE_THREADS;
   bool packed_indices = options && options->packed_index_publication;
   bool native_object_payload = options && options->native_object_payload;
   bool native_object_hw_payload = options && options->native_object_hw_payload;
   if (native_object_hw_payload && !native_object_payload)
      return fail(reason,
                  "native mesh hardware payload requires native object payload mode");
   if (!eligible(nir, physical_threads, packed_indices, native_object_payload,
                 reason))
      return false;

   if (packed_indices && options->primitive_broadcast)
      return fail(reason, "packed index candidate has no primitive broadcast");
   bool rounds = options && options->publication_rounds;
   if (rounds && (!packed_indices || physical_threads != AGX_APPLE9_NATIVE_SERIAL_THREADS))
      return fail(reason, "multi-round publication requires packed indices at T32");
   if (nir->info.mesh.primitive_type == MESA_PRIM_LINES ||
       nir->info.mesh.primitive_type == MESA_PRIM_POINTS) {
      if (!rounds)
         return fail(reason, "native points/lines require publication rounds at T32");
      /* Degenerate-triangle index culling cannot represent culled points or
       * lines (a zero-length line or point still rasterizes). */
      if (nir->info.cull_distance_array_size ||
          (nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE))
         return fail(reason, "native point/line culling is unsupported");
   }

   lower_indirect_io_slots(nir);

   /* Strip unused gl_PointSize stores on lines and triangles before IO
    * validation runs. Point topology retains VARYING_SLOT_PSIZ and clamps
    * authored sizes to Apple9's [1.0, 511.9375] hardware range. */
   if (nir->info.mesh.primitive_type == MESA_PRIM_POINTS) {
      nir_shader_intrinsics_pass(nir, clamp_point_size,
                                 nir_metadata_control_flow, NULL);
   } else if (nir_shader_intrinsics_pass(nir, drop_unused_point_size,
                                         nir_metadata_control_flow, NULL)) {
      nir->info.outputs_written &= ~VARYING_BIT_PSIZ;
   }

   nir_remove_dead_variables(nir, nir_var_mem_shared, NULL);
   nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared,
                                    glsl_get_natural_size_align_bytes);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir_remove_dead_variables(nir, nir_var_mem_shared, NULL);

   unsigned payload_stride_B = 0;
   struct native_child_grid_lower child_grid = {0};
   if (native_object_payload) {
      nir_remove_dead_variables(nir, nir_var_mem_task_payload, NULL);
      nir_lower_vars_to_explicit_types(nir, nir_var_mem_task_payload,
                                       glsl_get_natural_size_align_bytes);
      nir_lower_explicit_io(nir, nir_var_mem_task_payload,
                            nir_address_format_32bit_offset);
      nir_remove_dead_variables(nir, nir_var_mem_task_payload, NULL);
      if (nir->info.task_payload_size > AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES)
         return fail(reason,
                     "native mesh task payload exceeds 16 KiB hardware limit");
      unsigned min_stride = ALIGN_POT(nir->info.task_payload_size, 16);
      payload_stride_B = options->native_object_payload_stride_B
                            ? options->native_object_payload_stride_B
                            : min_stride;
      if ((payload_stride_B & 15) || payload_stride_B < min_stride ||
          payload_stride_B > AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES)
         return fail(reason, "native mesh task payload stride is invalid");
      child_grid = (struct native_child_grid_lower){
         .payload_uniform = AGX_APPLE9_MESH_PAYLOAD_UNIFORM,
         .grid_offset_B = payload_stride_B,
         .hw_payload = native_object_hw_payload,
      };
      nir_shader_intrinsics_pass(nir, lower_native_mesh_child_num_workgroups,
                                 nir_metadata_control_flow, &child_grid);
      if (native_object_hw_payload && child_grid.lowered &&
          payload_stride_B > AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES - 16)
         return fail(reason,
                     "native mesh hardware child grid exceeds 16 KiB payload limit");
   }

   struct cull_storage cull = {
      .clip = nir->info.clip_distance_array_size,
      .cull = nir->info.cull_distance_array_size,
      .base = ALIGN_POT(nir->info.shared_size, 4),
      .keep_outputs = options && options->link_clip_distances,
   };
   if (cull.cull) {
      if (!packed_indices || cull.cull > 8 ||
          cull.base > AGX_APPLE9_MAX_SHARED_BYTES ||
          nir->info.mesh.max_vertices_out * cull.cull * 4 >
             AGX_APPLE9_MAX_SHARED_BYTES - cull.base)
         return fail(reason, "native mesh cull distances need packed indices and shared space");
      nir_shader_intrinsics_pass(nir, move_cull_distance, nir_metadata_control_flow, &cull);
      if (cull.failed)
         return fail(reason, "native mesh cull distances must be constant-indexed per-vertex stores");
      nir->info.shared_size = cull.base + nir->info.mesh.max_vertices_out * cull.cull * 4;
      nir->info.cull_distance_array_size = 0;
   }
   struct primitive_cull_storage primitive_cull = {
      .base = ALIGN_POT(nir->info.shared_size, 4),
   };
   if (rounds) {
      nir_shader_intrinsics_pass(nir, move_primitive_cull,
                                 nir_metadata_control_flow, &primitive_cull);
      if (primitive_cull.failed)
         return fail(reason, "native mesh CullPrimitive requires scalar stores with zero slot offset");
      if (primitive_cull.written) {
         if (primitive_cull.base > AGX_APPLE9_MAX_SHARED_BYTES ||
             nir->info.mesh.max_primitives_out * 4 >
                AGX_APPLE9_MAX_SHARED_BYTES - primitive_cull.base)
            return fail(reason, "native mesh CullPrimitive exceeds local shared capacity");
         nir->info.shared_size = primitive_cull.base +
                                nir->info.mesh.max_primitives_out * 4;
         gather_mesh_info_preserving_clips(nir, cull.clip);
      }
   }
   bool clip_linked = options && options->link_clip_distances &&
                      (nir->info.clip_distance_array_size || cull.cull);
   if (clip_linked && nir->info.clip_distance_array_size) {
      if (nir->info.outputs_written &
          BITFIELD64_RANGE(AGX_APPLE9_CLIP_MIRROR_SLOT, 2))
         return fail(reason, "native mesh clip mirror slots are already written");
      struct clip_mirror clip = {.distances = nir->info.clip_distance_array_size};
      nir_shader_intrinsics_pass(nir, mirror_clip_distance, nir_metadata_control_flow, &clip);
      if (clip.failed)
         return fail(reason, "native mesh linked clip distances require constant slot offsets");
      gather_mesh_info_preserving_clips(nir, clip.distances);
   }

   uint16_t logical_size[3];
   memcpy(logical_size, nir->info.workgroup_size, sizeof(logical_size));
   unsigned logical_threads = logical_size[0] * logical_size[1] * logical_size[2];
   if (logical_threads > physical_threads) {
      /* Preserve the application's subgroup width and execute its subgroups
       * in order between workgroup barriers. The common pass supplies masks
       * for the final partial subgroup and keeps logical IDs and values live
       * across barriers. Completion is appended only after this transform. */
      nir_lower_system_values(nir);
      agx_nir_lower_apple9_mesh_global_id(nir);
      nir_lower_halt_to_return(nir);
      nir_lower_returns(nir);
      nir_lower_workgroup_size(nir, physical_threads);
      bool progress;
      do {
         progress = nir_lower_vars_to_ssa(nir);
         progress |= nir_opt_copy_prop(nir);
         progress |= nir_opt_constant_folding(nir);
         progress |= nir_opt_dead_cf(nir);
         progress |= nir_opt_remove_phis(nir);
         progress |= nir_opt_dce(nir);
      } while (progress);
      gather_mesh_info_preserving_clips(nir, cull.clip);
   }

   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   uint64_t primitive_broadcast = 0;
   uint64_t primitive_vertices = 0;
   unsigned original_max_vertices = nir->info.mesh.max_vertices_out;
   bool allow_layer_viewport = options && options->layer_viewport;
   if (options && options->attributes && options->primitive_broadcast &&
       !primitive_outputs(nir, &key, false, allow_layer_viewport,
                          &primitive_broadcast, reason))
      return false;
   if (rounds && options->attributes && options->primitive_vertices &&
       !primitive_outputs(nir, &key, true, allow_layer_viewport,
                          &primitive_vertices, reason))
      return false;

   key.physical_threads = physical_threads;
   key.hw_payload = native_object_hw_payload;
   if (!agx_nir_lower_apple9_mesh(nir, &key, info, reason))
      return false;
   if (native_object_payload) {
      info->task_payload_stride_B = payload_stride_B;
      info->reads_child_grid = child_grid.lowered;
      info->reads_native_payload =
         info->task_payload_size_B > 0 || child_grid.lowered;
   }
   memcpy(info->logical_size, logical_size, sizeof(logical_size));
   info->primitive_broadcast = primitive_broadcast;
   info->primitive_vertices = primitive_vertices;
   info->original_max_vertices = primitive_vertices ? original_max_vertices : 0;
   info->publication_rounds = rounds;
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   struct agx_apple9_varying_layout varyings;
   info->clip_linked = clip_linked;
   if (!varying_layout(info, options, &varyings, reason))
      return false;
   /* Clip distances, point size, and packed layer/viewport need rasterizer
    * planes; cull distances stay in shared storage until completion. */
   info->writes_point_size =
      layout->vertex_component[VARYING_SLOT_PSIZ][0] != AGX_APPLE9_MESH_UNUSED;
   info->writes_layer_viewport =
      layout->vertex_component[VARYING_SLOT_LAYER][0] != AGX_APPLE9_MESH_UNUSED;
   info->clip_distances = nir->info.clip_distance_array_size;
   info->cull_distances = cull.cull;
   info->cull_B = cull.base;
   info->cull_primitive = primitive_cull.written;
   info->cull_primitive_B = primitive_cull.written ? primitive_cull.base : 0;
   if (info->clip_distances > 8 || info->clip_distances + info->cull_distances > 8)
      return fail(reason, "native mesh supports eight clip plus cull distances");
   for (unsigned d = 0; d < info->clip_distances; ++d) {
      if (layout->vertex_component[VARYING_SLOT_CLIP_DIST0 + d / 4][d % 4] ==
          AGX_APPLE9_MESH_UNUSED)
         return fail(reason, "native mesh clip distance is declared but not written");
   }
   if (4 + varyings.count + info->writes_point_size + info->writes_layer_viewport +
       info->clip_distances > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return fail(reason, "native mesh varyings and rasterizer outputs exceed the UVS");
   nir_shader_intrinsics_pass(nir, consume_count, nir_metadata_none, info);
   nir_lower_returns(nir);
   if (!agx_nir_mask_apple9_mesh_body(nir, &key, layout, reason))
      return false;
   bool guard_empty_grid = options && options->guard_empty_grid;
   info->guard_empty_grid = guard_empty_grid;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list user_body;
   if (guard_empty_grid)
      nir_cf_extract(&user_body, nir_before_impl(impl), nir_after_impl(impl));
   /* A uniform return can bypass SetMeshOutputs entirely. Such a workgroup
    * emits nothing. Initialize both count words before the application mask;
    * only lane zero writes them, and completion reads after its workgroup
    * barrier. No initialization of undefined output attributes is needed. */
   nir_builder entry = nir_builder_at(nir_before_impl(impl));
   nir_push_if(&entry, nir_ieq_imm(&entry,
                                 nir_load_local_invocation_index(&entry), 0));
   nir_store_local_apple9_agx(
      &entry, nir_imm_int(&entry, 0),
      nir_imm_int(&entry, layout->primitive_count_B),
      .base = key.output_root, .bit_size = key.primitive_count_bits);
   nir_store_local_apple9_agx(
      &entry, nir_imm_int(&entry, 0),
      nir_imm_int(&entry, layout->vertex_count_B),
      .base = key.output_root, .bit_size = 32);
   nir_pop_if(&entry, NULL);
   if (info->cull_primitive) {
      /* Unwritten flags are undefined by the API, but must not depend on a
       * previous workgroup's LDS contents. Initialize every flag before the
       * application mask, including flags owned by inactive logical lanes. */
      nir_def *lane = nir_load_local_invocation_index(&entry);
      for (unsigned first = 0; first < layout->max_primitives;
           first += physical_threads) {
         nir_def *primitive = nir_iadd_imm(&entry, lane, first);
         nir_push_if(&entry, nir_ult_imm(&entry, primitive,
                                       layout->max_primitives));
         nir_store_local_apple9_agx(
            &entry, nir_imm_int(&entry, 0),
            nir_iadd_imm(&entry, nir_imul_imm(&entry, primitive, 4),
                         info->shared_offset_B + info->cull_primitive_B),
            .base = key.output_root, .bit_size = 32);
         nir_pop_if(&entry, NULL);
      }
      agx_nir_apple9_mesh_workgroup_barrier(&entry);
   }
   if (primitive_broadcast & VARYING_BIT_CULL_PRIMITIVE) {
      nir_push_if(&entry, nir_ieq_imm(&entry,
                                    nir_load_local_invocation_index(&entry), 0));
      nir_store_local_apple9_agx(
         &entry, nir_imm_int(&entry, 0),
         nir_imm_int(&entry, layout->primitive_count_B + 8),
         .base = key.output_root, .bit_size = 32);
      nir_pop_if(&entry, NULL);
      /* All physical lanes rendezvous before any application cull store. */
      agx_nir_apple9_mesh_workgroup_barrier(&entry);
   }
   if (info->writes_point_size || info->writes_layer_viewport) {
      /* Initialize default PointSize (1.0f) and Layer/Viewport (0) planes
       * before the application body runs so omitted or conditionally unwritten
       * outputs never read stale LDS values from a prior workgroup. */
      nir_def *lane = nir_load_local_invocation_index(&entry);
      unsigned psiz_plane =
         info->writes_point_size
            ? agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_PSIZ, 0) : 0;
      unsigned layer_plane =
         info->writes_layer_viewport
            ? agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_LAYER, 0) : 0;
      bool has_vp =
         info->writes_layer_viewport &&
         layout->vertex_component[VARYING_SLOT_VIEWPORT][0] != AGX_APPLE9_MESH_UNUSED;
      unsigned vp_plane =
         has_vp ? agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_VIEWPORT, 0) : 0;
      for (unsigned first = 0; first < layout->max_vertices;
           first += physical_threads) {
         nir_def *v = nir_iadd_imm(&entry, lane, first);
         nir_push_if(&entry, nir_ult_imm(&entry, v, layout->max_vertices));
         nir_def *off = nir_imul_imm(&entry, v, 4);
         if (info->writes_point_size)
            nir_store_local_apple9_agx(&entry, nir_imm_float(&entry, 1.0f),
                                       nir_iadd_imm(&entry, off, psiz_plane),
                                       .base = key.output_root, .bit_size = 32);
         if (info->writes_layer_viewport)
            nir_store_local_apple9_agx(&entry, nir_imm_int(&entry, 0),
                                       nir_iadd_imm(&entry, off, layer_plane),
                                       .base = key.output_root, .bit_size = 32);
         if (has_vp)
            nir_store_local_apple9_agx(&entry, nir_imm_int(&entry, 0),
                                       nir_iadd_imm(&entry, off, vp_plane),
                                       .base = key.output_root, .bit_size = 32);
         nir_pop_if(&entry, NULL);
      }
      agx_nir_apple9_mesh_workgroup_barrier(&entry);
   }
   if (guard_empty_grid) {
      /* Guard GPU-zeroed grids (e.g. conditional rendering) that substitute a
       * (1, 1, 1) dummy launch when any grid axis is zero. Because
       * load_num_workgroups reads the draw's immutable 3-word grid, the branch
       * is workgroup-uniform: empty draws skip every application instruction
       * and side effect and publish zero vertices/primitives via the
       * pre-initialized header. */
      nir_def *nwg = nir_load_num_workgroups(&entry);
      nir_def *active = nir_iand(
         &entry,
         nir_iand(&entry, nir_ine_imm(&entry, nir_channel(&entry, nwg, 0), 0),
                  nir_ine_imm(&entry, nir_channel(&entry, nwg, 1), 0)),
         nir_ine_imm(&entry, nir_channel(&entry, nwg, 2), 0));
      nir_if *guard = nir_push_if(&entry, active);
      guard->control = nir_selection_control_dont_flatten;
      entry.cursor = nir_cf_reinsert(&user_body, entry.cursor);
      nir_pop_if(&entry, guard);
   }
   nir_builder b = nir_builder_at(nir_after_impl(impl));

   agx_nir_apple9_mesh_workgroup_barrier(&b);

   /* Counts are the
    * actual shader publication, not maxima or register-descriptor constants.
    * U0 consumes and clears its A halfword; B's wider meaning is unresolved.
    * The qualified one-triangle class supplies actual vertex count to B. */
   nir_def *primitives = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->primitive_count_B), .base = key.output_root);
   /* The output ABI writes only the low halfword. Ignore the unrelated upper
    * bytes before bounding cull loops and index publication by this count. */
   if (key.primitive_count_bits == 16)
      primitives = nir_iand_imm(&b, primitives, 0xffff);
   nir_def *vertices = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->vertex_count_B), .base = key.output_root);
   nir_def *lane = nir_load_local_invocation_index(&b);
   if (primitive_broadcast)
      broadcast_primitive(&b, info, lane, vertices);
   if (info->cull_distances || info->cull_primitive)
      cull_primitives(&b, info, lane, primitives, physical_threads);
   if (primitive_vertices)
      vertices = duplicate_primitive_vertices(&b, info, lane, vertices, primitives,
                                               physical_threads);
   if (info->writes_layer_viewport)
      pack_layer_viewport(&b, info, lane, vertices, physical_threads);
   if (options && (options->clip_halfz || options->dynamic_clip_z)) {
      /* Convert the completed homogeneous position, after all body writes and
       * output reads. Doing this at publication also handles scalar/partial
       * stores and different invocations writing Z and W. Keep W unchanged:
       * U consumes it for clipping and perspective division.
       * When dynamic_clip_z is set, load clip_z_coeff (0.5 for [-w,w], 0.0 for
       * [0,w]) from the caller's draw UBO and skip conversion when zero.
       * The body barrier above makes its final Z/W visible. */
      nir_def *coeff = options->dynamic_clip_z
         ? nir_load_ubo(&b, 1, 32, nir_imm_int(&b, options->clip_z_ubo),
                        nir_imm_int(&b, options->clip_z_offset_B),
                        .align_mul = 4, .range_base = options->clip_z_offset_B,
                        .range = 4)
         : nir_imm_float(&b, 0.5f);
      if (options->dynamic_clip_z)
         nir_push_if(&b, nir_fneu_imm(&b, coeff, 0.0f));
      unsigned z_plane = agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_POS, 2);
      unsigned w_plane = agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_POS, 3);
      for (unsigned first = 0; first < layout->max_vertices;
           first += physical_threads) {
         nir_def *vertex = nir_iadd_imm(&b, lane, first);
         nir_push_if(&b, nir_ult(&b, vertex, vertices));
         nir_def *offset = nir_imul_imm(&b, vertex, 4);
         nir_def *z_address = nir_iadd_imm(&b, offset, z_plane);
         nir_def *w_address = nir_iadd_imm(&b, offset, w_plane);
         nir_def *z = nir_load_local_apple9_agx(&b, z_address,
                                                .base = key.output_root);
         nir_def *w = nir_load_local_apple9_agx(&b, w_address,
                                                .base = key.output_root);
         nir_def *converted = nir_ffma_weak(
            &b, nir_fneg(&b, z), coeff, nir_ffma_weak(&b, w, coeff, z));
         nir_store_local_apple9_agx(&b, converted, z_address,
                                    .base = key.output_root, .bit_size = 32);
         nir_pop_if(&b, NULL);
      }
      if (options->dynamic_clip_z)
         nir_pop_if(&b, NULL);
      agx_nir_apple9_mesh_workgroup_barrier(&b);
   }
   /* Publish the completed output memory before issuing mesh allocation.
    * U0 consumes the counts but does not write these global output planes.
    * Keeping this fence before U0 lets the first channel submit consume U0's
    * separate completion dependency through its measured input-wait field. */
   device_fence(&b);
   nir_mesh_control_apple9_agx(&b, nir_u2u16(&b, primitives),
                               nir_u2u16(&b, vertices), .base = 1);

   /* Each channel/index collective publishes at most one round of entries.
    * Its lanes form a contiguous prefix supplying one common source byte
    * offset (second operand, S2's uniform-base invariant); the destination
    * entry is bits 16 and up of the first operand: the channel selector for
    * vertex channels, the vertex-count operand for packed index words. Later
    * rounds advance both. Air 2026-10-24: a repeat without the destination
    * aliases entry 0; with it V64/P126 at T32 is exact. Single-round classes
    * keep destination 0 and emit the same code. Rounds beyond the runtime
    * counts are skipped, never issued empty. */
   const unsigned round = MIN2(physical_threads, AGX_APPLE9_PUBLICATION_ROUND);
   const unsigned clip_distances = info->clip_distances;
   for (unsigned first = 0; first < layout->max_vertices; first += round) {
      nir_def *active = nir_ult(&b, nir_iadd_imm(&b, lane, first), vertices);
      if (round != physical_threads)
         active = nir_iand(&b, active, nir_ult_imm(&b, lane, round));
      nir_push_if(&b, active);
      const uint32_t destination = first << 16;
      for (unsigned c = 0; c < 4; ++c) {
         unsigned plane = agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_POS, c);
         nir_mesh_submit_channel_apple9_agx(&b, nir_imm_int(&b, c | destination),
                                            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
      }
      for (unsigned slot = 0; slot < ARRAY_SIZE(varyings.mask); ++slot) {
         u_foreach_bit(c, varyings.mask[slot]) {
            unsigned channel = 4 + agx_apple9_varying_index(&varyings, slot, c);
            unsigned plane = agx_apple9_mesh_plane_offset(layout, slot, c);
            nir_mesh_submit_channel_apple9_agx(
               &b, nir_imm_int(&b, channel | destination),
               nir_imm_int(&b, plane + 4 * first));
            device_fence(&b);
         }
      }
      /* Rasterizer outputs follow the user varyings in VS UVS order:
       * PointSize, packed Layer/Viewport word, then ClipDistances.
       * The total published channel count equals the PPP output count
       * 4 + varyings + point_size + layer_viewport + clip_distances. */
      unsigned raster_channel = 4 + varyings.count;
      if (info->writes_point_size) {
         unsigned plane = agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_PSIZ, 0);
         nir_mesh_submit_channel_apple9_agx(
            &b, nir_imm_int(&b, raster_channel | destination),
            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
         ++raster_channel;
      }
      if (info->writes_layer_viewport) {
         unsigned plane = agx_apple9_mesh_plane_offset(layout, VARYING_SLOT_LAYER, 0);
         nir_mesh_submit_channel_apple9_agx(
            &b, nir_imm_int(&b, raster_channel | destination),
            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
         ++raster_channel;
      }
      for (unsigned d = 0; d < clip_distances; ++d) {
         unsigned slot = (info->clip_linked ? AGX_APPLE9_CLIP_MIRROR_SLOT
                                            : VARYING_SLOT_CLIP_DIST0) + d / 4;
         unsigned plane = agx_apple9_mesh_plane_offset(layout, slot, d % 4);
         nir_mesh_submit_channel_apple9_agx(
            &b, nir_imm_int(&b, (raster_channel + d) | destination),
            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
      }
      nir_pop_if(&b, NULL);
   }
   /* Preserve the qualified first operand (actual vertex count) in the low
    * bits. Index words start at byte 0 of the output root. */
   if (packed_indices) {
      nir_def *words = agx_nir_apple9_index_publication_lanes(
         &b, primitives, layout->indices_per_primitive);
      unsigned max_words =
         MAX2(DIV_ROUND_UP(layout->max_primitives * layout->indices_per_primitive, 4), 1);
      for (unsigned first = 0; first < max_words; first += round) {
         nir_def *active = nir_ult(&b, nir_iadd_imm(&b, lane, first), words);
         if (round != physical_threads)
            active = nir_iand(&b, active, nir_ult_imm(&b, lane, round));
         nir_push_if(&b, active);
         nir_def *count = first ? nir_ior_imm(&b, vertices, first << 16) : vertices;
         nir_mesh_submit_scalar_apple9_agx(&b, count, nir_imm_int(&b, 4 * first));
         device_fence(&b);
         nir_pop_if(&b, NULL);
      }
   } else {
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_mesh_submit_scalar_apple9_agx(&b, vertices, nir_imm_int(&b, 0));
      device_fence(&b);
      nir_pop_if(&b, NULL);
   }

   info->requires_mesh_completion = false;
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_none);
   nir_validate_shader(nir, "native Apple9 mesh completion");
   return true;
}

bool
agx_compile_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (!agx_nir_lower_apple9_native_mesh(nir, enable_native_mesh, options,
                                         info, reason))
      return false;
   return agx_compile_apple9_native_mesh_lowered(nir, options, out, info, reason);
}

bool
agx_compile_apple9_native_mesh_lowered(
   nir_shader *nir, const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, const struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (!agx_compile_apple9_mesh_lowered(nir, &info->key, out, reason))
      return false;
   out->info.apple9_writes_point_size = info->writes_point_size;
   out->info.writes_layer_viewport = info->writes_layer_viewport;
   out->info.apple9_clip_distance_count = info->clip_distances;
   return varying_layout(info, options, &out->info.apple9_varyings,
                         reason);
}
