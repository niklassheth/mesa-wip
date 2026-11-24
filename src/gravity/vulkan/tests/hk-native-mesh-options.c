/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_shader.h"
#include "hk_shader_priv.h"
#include "hk_apple9_mesh.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "vk_graphics_state.h"
#include "gravity/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)
#include "gravity/compiler/test/apple9-mesh-executor.h"

static const unsigned slots[] = {VARYING_SLOT_VAR0, VARYING_SLOT_VAR3, VARYING_SLOT_VAR5};
static const float positions[3][4] = {
   {-.5, -.5, 0, 1}, {1, -1, 1, 2}, {0, .25, .125, .5},
};

static nir_shader *
producer(bool mesh)
{
   nir_builder b = nir_builder_init_simple_shader(
      mesh ? MESA_SHADER_MESH : MESA_SHADER_VERTEX, &agx_nir_options,
      "hk native key producer");
   b.shader->info.io_lowered = true;
   if (mesh) {
      b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
         b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                         nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   }
   for (unsigned v = 0; v < (mesh ? 3 : 1); ++v) {
      nir_def *position = nir_imm_vec4(&b, positions[v][0], positions[v][1],
                                          positions[v][2], positions[v][3]);
      if (mesh)
         nir_store_per_vertex_output(&b, position, nir_imm_int(&b, v),
            nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      else
         nir_store_output(&b, position, nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
         nir_def *value = nir_imm_float(&b, .25f * (i + v + 1));
         if (mesh)
            nir_store_per_vertex_output(&b, value, nir_imm_int(&b, v),
               nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
               .io_semantics = {.location = slots[i], .num_slots = 1});
         else
            nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 1,
               .src_type = nir_type_float32,
               .io_semantics = {.location = slots[i], .num_slots = 1});
      }
   }
   if (mesh)
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
consumer(unsigned flat, unsigned linear)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "hk native key consumer");
   b.shader->info.io_lowered = true;
   nir_def *values[4];
   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      if (i == flat)
         values[i] = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = slots[i], .num_slots = 1});
      else
         values[i] = nir_load_interpolated_input(&b, 1, 32,
            nir_load_barycentric_pixel(&b, 32,
               .interp_mode = i == linear ? INTERP_MODE_NOPERSPECTIVE : INTERP_MODE_SMOOTH),
            nir_imm_int(&b, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = slots[i], .num_slots = 1});
   }
   values[3] = nir_imm_float(&b, 1);
   nir_store_output(&b, nir_vec(&b, values, 4), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
check_key(bool halfz, unsigned flat, unsigned linear)
{
   const struct hk_apple9_graphics_key key = {
      .clip_negative_one_to_one = halfz,
      .vertex.outputs_flat = BITFIELD64_BIT(slots[flat]),
      .vertex.outputs_linear = BITFIELD64_BIT(slots[linear]),
   };
   nir_shader *mesh = producer(true), *vs = producer(false);
   struct agx_shader_part native = {0}, vertex = {0}, fragment = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &native, &info, &reason));
   CHECK(agx_compile_apple9_vertex_inputs(vs, &key.vertex, &vertex, &reason));
   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      unsigned expected = i == flat ? 1 : i == linear ? 2 : 0;
      CHECK(native.info.apple9_varyings.group[slots[i]] == expected);
      CHECK(vertex.info.apple9_varyings.group[slots[i]] == expected);
   }
   CHECK(!memcmp(&native.info.apple9_varyings, &vertex.info.apple9_varyings,
                  sizeof(native.info.apple9_varyings)));
   nir_shader *fs = consumer(flat, linear);
   CHECK(agx_compile_apple9_fragment_inputs(fs, &native.info.apple9_varyings,
                                           &fragment, &reason));
   CHECK(!memcmp(&fragment.info.apple9_varyings, &vertex.info.apple9_varyings,
                  sizeof(vertex.info.apple9_varyings)));

   /* Execute the lowered NIR returned by the real hk compilation call. This
    * checks the key's effect on publication, independently of its options
    * construction and without a device or shader-upload mock. */
   struct machine m = {.num_groups = {1, 1, 1}};
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   for (unsigned v = 0; v < 3; ++v) {
      float z, w;
      memcpy(&z, &m.submitted_position[v][2], 4);
      memcpy(&w, &m.submitted_position[v][3], 4);
      float expected = halfz ? fmaf(-positions[v][2], .5f,
                                    fmaf(positions[v][3], .5f, positions[v][2]))
                             : positions[v][2];
      CHECK(z == expected && w == positions[v][3]);
   }
   free(m.value);
   free(native.binary);
   free(vertex.binary);
   free(fragment.binary);
   ralloc_free(mesh);
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
check_primitive_key(void)
{
   CHECK(hk_apple9_native_mesh_primitive_lowering_enabled());
   nir_shader *mesh = producer(true);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(mesh)));
   nir_store_per_primitive_output(&b, nir_imm_int(&b, 37),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part out = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &out, &info, &reason));
   CHECK(info.primitive_broadcast == VARYING_BIT_PRIMITIVE_ID);
   CHECK(out.info.apple9_varyings.mask[VARYING_SLOT_PRIMITIVE_ID] == 1);
   CHECK(out.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID] == 1);
   free(out.binary);
   ralloc_free(mesh);
}

static void
check_workgroup_key(const uint16_t size[3], bool shared, unsigned expected)
{
   nir_shader *mesh = producer(true);
   memcpy(mesh->info.workgroup_size, size, sizeof(mesh->info.workgroup_size));
   if (shared) {
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(mesh)));
      mesh->info.shared_size = 28672;
      nir_store_shared(&b, nir_imm_int(&b, 123), nir_imm_int(&b, 28668),
                        .align_mul = 4, .write_mask = 1);
   }
   const struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_native_mesh_threads(size) == expected);
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &part, &info, &reason)
         == (expected != 0));
   if (expected) {
      CHECK(!memcmp(info.logical_size, size, sizeof(info.logical_size)));
      CHECK(info.key.physical_threads == expected);
      CHECK(part.info.workgroup_size[0] == expected);
      CHECK(part.info.workgroup_size[1] == 1 && part.info.workgroup_size[2] == 1);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
      /* A cached body cannot change its logical workgroup across the T16/T32
       * selection boundary while retaining the original compiled width. */
      struct agx_apple9_mesh_body_info stale = info;
      stale.logical_size[0] = expected == 16 ? 17 : 16;
      stale.logical_size[1] = stale.logical_size[2] = 1;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale,
                                          HK_APPLE9_NATIVE_MESH));
      struct agx_apple9_launch_parameters launch;
      CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &info,
         HK_APPLE9_NATIVE_MESH, 0, 0x1000));
      CHECK(launch.threadgroup_memory_bytes == (shared ? 32768 : 256));
      CHECK(info.shared_size_B == (shared ? 28672 : 0));
   } else {
      CHECK(reason && !part.binary);
   }
   free(part.binary);
   ralloc_free(mesh);
}

static unsigned
io_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
check_output_array(bool primitive, bool reverse)
{
   nir_shader *mesh = producer(true);
   mesh->info.io_lowered = false;
   const struct glsl_type *type = glsl_array_type(
      glsl_array_type(glsl_vec4_type(), 3, 0), primitive ? 1 : 3, 0);
   nir_variable *array = nir_variable_create(mesh, nir_var_shader_out, type,
                                             "output_array");
   array->data.location = VARYING_SLOT_VAR8;
   array->data.per_primitive = primitive;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(mesh)));
   nir_variable *readback = nir_variable_create(
      mesh, nir_var_shader_out,
      glsl_array_type(glsl_array_type(glsl_vec4_type(), 4, 0),
                      primitive ? 1 : 3, 0),
      "output_readback");
   readback->data.location = VARYING_SLOT_VAR12;
   readback->data.per_primitive = primitive;
   for (unsigned v = 0; v < (primitive ? 1 : 3); ++v) {
      for (unsigned j = 0; j < 3; ++j) {
         nir_def *index = nir_load_global(&b, 1, 32,
            nir_imm_int64(&b, INPUT_ADDR + j * 4), .align_mul = 4);
         nir_deref_instr *deref = nir_build_deref_array_imm(&b,
            nir_build_deref_var(&b, array), v);
         deref = nir_build_deref_array(&b, deref, index);
         nir_def *values[4];
         for (unsigned c = 0; c < 4; ++c)
            values[c] = nir_imm_int(&b, 0x80000 + 100 * v + 10 * j + c);
         nir_store_deref(&b, deref, nir_vec(&b, values, 4), 15);
      }
      /* Out-of-bounds dynamic store must be guarded and leave valid slots intact. */
      nir_def *oob_idx = nir_load_global(&b, 1, 32,
         nir_imm_int64(&b, INPUT_ADDR + 3 * 4), .align_mul = 4);
      nir_deref_instr *oob_deref = nir_build_deref_array(&b,
         nir_build_deref_array_imm(&b, nir_build_deref_var(&b, array), v),
         oob_idx);
      nir_store_deref(&b, oob_deref, nir_imm_ivec4(&b, -1, -1, -1, -1), 15);
      /* Exercise dynamic output-array loads (both in-bounds and OOB -> 0). */
      for (unsigned j = 0; j < 4; ++j) {
         nir_def *index = nir_load_global(&b, 1, 32,
            nir_imm_int64(&b, INPUT_ADDR + j * 4), .align_mul = 4);
         nir_deref_instr *deref = nir_build_deref_array(&b,
            nir_build_deref_array_imm(&b, nir_build_deref_var(&b, array), v),
            index);
         nir_def *loaded = nir_load_deref(&b, deref);
         nir_deref_instr *dst = nir_build_deref_array_imm(&b,
            nir_build_deref_array_imm(&b, nir_build_deref_var(&b, readback), v),
            j);
         nir_store_deref(&b, dst, loaded, 15);
      }
   }
   /* Exercise nir_lower_io WITHOUT pre-lowering nir_var_shader_out to if-else
    * trees (matching hk_lower_nir): native mesh lowering must resolve only the
    * inner dynamic slot offset on its own clone, keeping fallback NIR untouched. */
   nir_lower_io(mesh, nir_var_shader_out, io_size, 0);
   nir_opt_constant_folding(mesh);
   nir_opt_dce(mesh);
   nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
   const struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &part, &info, &reason));
   CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   uint32_t indices[4];
   for (unsigned j = 0; j < 3; ++j)
      indices[j] = reverse ? 2 - j : j;
   indices[3] = 99; /* OOB slot index */
   struct machine m = {.num_groups = {1, 1, 1}};
   m.region[5] = (struct region){INPUT_ADDR, sizeof(indices), (uint8_t *)indices, NULL};
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   for (unsigned v = 0; v < 3; ++v) {
      for (unsigned j = 0; j < 3; ++j) {
         for (unsigned c = 0; c < 4; ++c) {
            unsigned slot = VARYING_SLOT_VAR8 + indices[j];
            unsigned channel = 4 + agx_apple9_varying_index(
               &part.info.apple9_varyings, slot, c);
            uint32_t expected = 0x80000 + 100 * (primitive ? 0 : v) + 10 * j + c;
            CHECK(m.submitted_position[v][channel] == expected);
            CHECK(part.info.apple9_varyings.group[slot] == (primitive ? 1 : 0));
            unsigned rb_ch = 4 + agx_apple9_varying_index(
               &part.info.apple9_varyings, VARYING_SLOT_VAR12 + j, c);
            CHECK(m.submitted_position[v][rb_ch] == expected);
         }
      }
      for (unsigned c = 0; c < 4; ++c) {
         unsigned oob_ch = 4 + agx_apple9_varying_index(
            &part.info.apple9_varyings, VARYING_SLOT_VAR12 + 3, c);
         CHECK(m.submitted_position[v][oob_ch] == 0);
      }
   }
   free(m.value);
   free(part.binary);
   ralloc_free(mesh);
}

/* Classification must leave semantic NIR intact even when the native pass
 * rejects after rewriting outputs or shared-memory accesses. Compare the
 * serialized program, then let the normal compile tests consume the source. */
static void
check_capability(nir_shader *nir, bool expected)
{
   const struct hk_apple9_graphics_key key = {0};
   struct blob before, after;
   blob_init(&before);
   blob_init(&after);
   nir_serialize(&before, nir, false);
   bool supported = !expected;
   const char *reason = NULL;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &reason) == VK_SUCCESS);
   CHECK(supported == expected);
   CHECK(supported || reason);
   nir_serialize(&after, nir, false);
   CHECK(!before.out_of_memory && !after.out_of_memory);
   CHECK(before.size == after.size);
   CHECK(!memcmp(before.data, after.data, before.size));
   blob_finish(&before);
   blob_finish(&after);
   ralloc_free(nir);
}

static void
check_capabilities(void)
{
   check_capability(producer(true), true);
   const unsigned declarations[][2] = {
      {4, 2}, {96, 32}, {256, 256}, {6, 4}, {192, 64},
      {24, 8}, {256, 254}, {3, 4},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(declarations); ++i) {
      nir_shader *nir = producer(true);
      nir->info.mesh.max_vertices_out = declarations[i][0];
      nir->info.mesh.max_primitives_out = declarations[i][1];
      /* This producer writes attributes, so every research non-P1
       * declaration compiles through the publication model at T32. */
      check_capability(nir, hk_apple9_native_mesh_model_counts_eligible(
                               nir->info.workgroup_size, declarations[i][0],
                               declarations[i][1]));
   }
   for (unsigned prim = MESA_PRIM_POINTS; prim <= MESA_PRIM_LINES; ++prim) {
      nir_shader *nir = producer(true);
      nir->info.mesh.primitive_type = prim;
      check_capability(nir, false);
   }
   nir_shader *nir = producer(true);
   nir->info.task_payload_size = 16;
   check_capability(nir, false);
   nir = producer(true);
   nir->info.workgroup_size[0] = 129;
   check_capability(nir, false);
   nir = producer(true);
   nir->info.workgroup_size[0] = 128;
   check_capability(nir, true);
   /* Layer/ViewportIndex are admitted through P1 broadcast once qualified. */
   nir = producer(true);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_store_per_primitive_output(&b, nir_imm_int(&b, 1),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_LAYER, .num_slots = 1});
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   check_capability(nir, true);
   /* Unsupported per-primitive builtins remain rejected. */
   nir = producer(true);
   b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_store_per_primitive_output(&b, nir_imm_int(&b, 1),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_SHADING_RATE, .num_slots = 1});
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   check_capability(nir, false);
}

static void
check_multiview_key(void)
{
   /* hk_apple9_graphics_key propagates multiview state from state->mv. */
   struct vk_multiview_state mv_off = {.view_mask = 0};
   struct vk_multiview_state mv_on = {.view_mask = 0x5};
   struct vk_graphics_pipeline_state st_off = {.mv = &mv_off};
   struct vk_graphics_pipeline_state st_on = {.mv = &mv_on};
   struct hk_apple9_graphics_key k_null, k_off, k_on;
   hk_apple9_graphics_key(&k_null, NULL);
   hk_apple9_graphics_key(&k_off, &st_off);
   hk_apple9_graphics_key(&k_on, &st_on);
   CHECK(!k_null.is_multiview);
   CHECK(!k_off.is_multiview);
   CHECK(k_on.is_multiview);

   /* V3/P1 (primitive_broadcast) and V4/P2 (primitive_vertices multi-round)
    * both compile natively with multiview layer output into the packed
    * Layer/Viewport UVS plane. */
   const unsigned shapes[][2] = {{3, 1}, {4, 2}};
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i) {
      nir_shader *mesh = producer(true);
      mesh->info.mesh.max_vertices_out = shapes[i][0];
      mesh->info.mesh.max_primitives_out = shapes[i][1];
      struct hk_apple9_graphics_key key = {.is_multiview = true};
      bool supported = false;
      const char *reason = NULL;
      CHECK(hk_apple9_native_mesh_check(mesh, &key, &supported, &reason) == VK_SUCCESS);
      CHECK(supported);
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info;
      CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &part, &info, &reason));
      CHECK(info.writes_layer_viewport && part.info.writes_layer_viewport);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
      free(part.binary);
      ralloc_free(mesh);
   }

   /* V + P > 256 cannot allocate provoking-vertex layer duplicates and must
    * decline multiview native admission cleanly without mutating NIR. */
   nir_shader *oversized = producer(true);
   oversized->info.mesh.max_vertices_out = 200;
   oversized->info.mesh.max_primitives_out = 100;
   struct hk_apple9_graphics_key key = {.is_multiview = true};
   bool supported = true;
   const char *reason = NULL;
   CHECK(hk_apple9_native_mesh_check(oversized, &key, &supported, &reason) == VK_SUCCESS);
   CHECK(!supported && reason);
   ralloc_free(oversized);
}

static void
check_no_fallback_if_else_bloat(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "no fallback if-else bloat");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size[0] = 32;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = 128;
   nir->info.mesh.max_primitives_out = 64;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;

   nir_variable *pos = nir_variable_create(
      nir, nir_var_shader_out, glsl_array_type(glsl_vec4_type(), 128, 0), "pos");
   pos->data.location = VARYING_SLOT_POS;
   nir_variable *col = nir_variable_create(
      nir, nir_var_shader_out, glsl_array_type(glsl_vec4_type(), 128, 0), "col");
   col->data.location = VARYING_SLOT_VAR0;
   nir_variable *idx = nir_variable_create(
      nir, nir_var_shader_out, glsl_array_type(glsl_uvec_type(3), 64, 0), "idx");
   idx->data.location = VARYING_SLOT_PRIMITIVE_INDICES;
   idx->data.per_primitive = true;

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 128),
                                      nir_imm_int(&b, 64), nir_imm_int(&b, 0));
   nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, pos), lane),
                   nir_imm_vec4(&b, 0, 0, 0.5f, 1), 0xf);
   nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, col), lane),
                   nir_imm_vec4(&b, 1, 0, 0, 1), 0xf);
   nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, idx), lane),
                   nir_imm_ivec3(&b, 0, 1, 2), 0x7);

   const struct vk_pipeline_robustness_state rs = vk_robustness_disabled;
   hk_lower_nir(dev, nir, &rs, false, 0, NULL, 0);

   /* hk_lower_nir must NOT expand per-vertex/per-primitive dynamic indexing
    * into if-else trees even with HK_APPLE9_NATIVE_MESH=1. */
   unsigned if_count = 0;
   unsigned store_count = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if)
         ++if_count;
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (op == nir_intrinsic_store_per_vertex_output ||
             op == nir_intrinsic_store_per_primitive_output)
            ++store_count;
      }
   }
   CHECK(if_count == 0);
   CHECK(store_count == 3);
   ralloc_free(nir);
   free(dev);
}

int main(void)
{
   CHECK(hk_apple9_native_mesh_enabled());
   CHECK(hk_apple9_native_mesh_attributes_enabled());
   glsl_type_singleton_init_or_ref();
   for (unsigned halfz = 0; halfz < 2; ++halfz) {
      check_key(halfz, 0, 1);
      check_key(halfz, 2, 0);
   }
   check_primitive_key();
   check_capabilities();
   check_multiview_key();
   check_no_fallback_if_else_bloat();
   for (unsigned primitive = 0; primitive < 2; ++primitive)
      for (unsigned reverse = 0; reverse < 2; ++reverse)
         check_output_array(primitive, reverse);
   const uint16_t shapes[][3] = {
      {1, 1, 1}, {16, 1, 1}, {17, 1, 1}, {32, 1, 1},
      {33, 1, 1}, {64, 1, 1}, {127, 1, 1}, {128, 1, 1},
      {1, 128, 1}, {1, 1, 128}, {4, 4, 4}, {2, 4, 16},
      {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {129, 1, 1},
      {1, 129, 1}, {1, 1, 129}, {16, 16, 1}, {65535, 65535, 65535},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
      for (unsigned shared = 0; shared < 2; ++shared)
         check_workgroup_key(shapes[i], shared, i < 2 ? 16 : i < 12 ? 32 : 0);
   CHECK(!hk_apple9_native_mesh_threads(NULL));
   glsl_type_singleton_decref();
   puts("hk native key: halfz/W, interpolation, output arrays, T128 shapes, shared allocation and immutable capability checks pass");
   return 0;
}
