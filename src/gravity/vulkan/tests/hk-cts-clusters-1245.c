/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Focused regression tests for Vulkan CTS failure clusters:
 *  - Cluster 1: arrayed input attachments with texture_offset and index >= 32
 *  - Cluster 2: volatile bindless_image_texel_address removal (no stale bindless_image_agx)
 *  - Cluster 4: standalone GPL per-vertex/barycentric FS native companion + roundtrip + link
 *  - Cluster 5: fixed-topology GS strips Apple9 image stores/loads/fences from main
 *  - Bonus 7a: GPL vertex shaders preserve bounds checks when pipelineRobustness is enabled
 *  - Board #378: no-position GS rejected by hk_apple9_gs_polygon_points_ok
 */
#include "../hk_shader.c"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: FAIL: %s\n", __FILE__, __LINE__, #c);         \
         abort();                                                              \
      }                                                                        \
   } while (0)
#include "hk-test-usc.h"

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   static unsigned allocations;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, MAX2(size, 16u));
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = dev->shader_base + ++allocations * 0x100000;
   return bo;
}

void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   ++bo->refcnt;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo || --bo->refcnt)
      return;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

static void
init_test_device(struct hk_physical_device *pdev, struct hk_device *dev)
{
   memset(pdev, 0, sizeof(*pdev));
   memset(dev, 0, sizeof(*dev));
   pdev->dev.chip = AGX_CHIP_G16G;
   pdev->dev.params.gpu_generation = 16;
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.params.gpu_generation = 16;
   dev->dev.shader_base = 0x100000000ull;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
}

static void
finish_test_device(struct hk_physical_device *pdev, struct hk_device *dev)
{
   hk_apple9_bodies_finish(dev);
   hk_device_finish_usc_heap(dev);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&pdev->debug_compile_lock);
}

/* Cluster 1: arrayed input attachments carry nir_tex_src_texture_offset and
 * texture_index >= 32 after nir_lower_input_attachments. hk_nir_lower_descriptors
 * must strip texture_offset and reset texture_index/array_size/input_attachment_index
 * so Apple9 texture lowering and selection accept them. */
static void
test_input_attachment_offset_and_high_index(struct hk_device *dev)
{
   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(struct hk_descriptor_set_binding_layout));
   CHECK(layout);
   layout->binding_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      .array_size = 128,
      .stride = 64,
      .offset = 0,
   };
   struct vk_descriptor_set_layout *layouts[] = {&layout->vk};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "c1 input attachment");
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;

   const struct glsl_type *subpass_type =
      glsl_image_type(GLSL_SAMPLER_DIM_SUBPASS, false, GLSL_TYPE_FLOAT);
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_uniform, glsl_array_type(subpass_type, 128, 0), "ia");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   nir_deref_instr *deref =
      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, var), 63);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->is_array = false;
   tex->texture_index = 63;
   tex->texture_array_size = 128;
   tex->input_attachment_index = true;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_offset, nir_imm_int(&b, 63));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_ivec2(&b, 0, 0));
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});

   CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled, 1, layouts, false));

   bool saw_tex = false;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *t = nir_instr_as_tex(instr);
         saw_tex = true;
         CHECK(nir_tex_instr_src_index(t, nir_tex_src_texture_offset) == -1);
         CHECK(nir_tex_instr_src_index(t, nir_tex_src_texture_handle) >= 0);
         CHECK(t->texture_index == 0);
         CHECK(t->texture_array_size == 0);
         CHECK(!t->input_attachment_index);
      }
   }
   CHECK(saw_tex);

   /* Also verify the resulting FS compiles cleanly on Apple9. */
   nir_shader_gather_info(b.shader, b.impl);
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_FRAGMENT,
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
      .set_layout_count = 1,
      .set_layouts = layouts,
   };
   struct vk_shader *out = NULL;
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1,
      .color_write_enables = 1,
      .attachments = {{.write_mask = 0xf}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   CHECK(hk_apple9_compile_graphics(dev, 1, &info, &state, NULL, NULL, NULL,
                                    &out) == VK_SUCCESS);
   CHECK(out != NULL);
   out->ops->destroy(&dev->vk, out, NULL);
   free(layout);
}

/* Cluster 2: volatile bindless_image_texel_address must be removed in place by
 * agx_nir_lower_texture_early so DCE does not retain it (and its bindless_image_agx
 * handle) because of ACCESS_VOLATILE. */
static void
test_volatile_bindless_image_texel_address(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "c2 volatile texel address");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *handle = nir_bindless_image_agx(&b, nir_imm_int(&b, 0),
                                            .desc_set = 0);
   nir_def *addr = nir_bindless_image_texel_address(
      &b, 64, handle, nir_imm_ivec4(&b, 1, 2, 0, 0), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_VOLATILE);
   nir_def *old = nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_global(&b, old, nir_imm_int64(&b, 0x2000));

   CHECK(agx_nir_lower_images(b.shader));
   nir_opt_dce(b.shader);

   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         CHECK(op != nir_intrinsic_bindless_image_texel_address);
      }
   }
   ralloc_free(b.shader);
}

/* Cluster 4: standalone GPL fragment shader using load_input_vertex +
 * barycentrics builds a companion apple9_fs_native_per_vertex realization,
 * survives serialization, and links cleanly against a GS varying layout. */
static void
test_gpl_per_vertex_fs_native_companion(struct hk_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "c4 gpl per-vertex fs");
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;

   nir_def *bary = nir_load_barycentric_coord_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *pv_f = nir_load_input_vertex(
      &b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_def *pv_u = nir_load_input_vertex(
      &b, 1, 32, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
      .dest_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
   nir_def *w = nir_channel(&b, bary, 0);
   nir_def *u_f = nir_u2f32(&b, pv_u);
   nir_def *out_col = nir_vec4(&b, nir_fmul(&b, nir_channel(&b, pv_f, 0), w),
                               nir_channel(&b, pv_f, 1),
                               nir_channel(&b, pv_f, 2), u_f);
   nir_store_output(&b, out_col, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, b.impl);

   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_FRAGMENT,
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
   };
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1,
      .color_write_enables = 1,
      .attachments = {{.write_mask = 0xf}},
   };
   const struct vk_graphics_pipeline_state state = {
      .pipeline_flags = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
      .rp = &rp,
      .cb = &cb,
   };
   struct vk_shader *out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &info, &state, NULL, NULL, NULL,
                                    &out) == VK_SUCCESS);
   struct hk_api_shader *fs = container_of(out, struct hk_api_shader, vk);

   /* Primary FS uses flat-copy lowering for passthrough GS / mesh-rast-meta. */
   CHECK(fs->variants[0].info.fs.per_vertex_copies == 2);
   CHECK(fs->variants[0].info.fs.per_vertex_bary[0] != 0);

   /* Companion native FS keeps per-vertex inputs on VAR0/VAR1 + HW barycentrics. */
   struct hk_api_shader *pv = fs->apple9_fs_native_per_vertex;
   CHECK(pv != NULL);
   CHECK(pv->variants[0].info.fs.per_vertex_copies == 0);
   CHECK(pv->variants[0].info.fs.per_vertex_bary[0] == 0);
   CHECK(pv->variants[0].b.info.apple9_barycentric_mask == 1);
   CHECK(pv->variants[0].b.info.apple9_per_vertex_count == 4);
   CHECK(pv->variants[0].b.info.apple9_varyings.mask[VARYING_SLOT_VAR0] == 0x7);
   CHECK(pv->variants[0].b.info.apple9_varyings.mask[VARYING_SLOT_VAR1] == 0x1);

   /* Verify draw-time CF crossbar remapping against a GS RAST varying layout
    * whose slots differ from the standalone FS layout (extra VAR2 output). */
   {
      struct agx_apple9_varying_layout gs_varyings = {0};
      gs_varyings.mask[VARYING_SLOT_VAR0] = 0xf;
      gs_varyings.mask[VARYING_SLOT_VAR1] = 0x1;
      gs_varyings.mask[VARYING_SLOT_VAR2] = 0xf;
      gs_varyings.count = 9;
      uint8_t fs_locs[AGX_APPLE9_MAX_VARYING_COMPONENTS];
      uint8_t fs_comps[AGX_APPLE9_MAX_VARYING_COMPONENTS];
      const struct agx_apple9_render_stage *frag = &pv->variants[0].apple9_stage;
      unsigned fs_count = agx_apple9_varying_slots(&frag->varyings, fs_locs, fs_comps);
      CHECK(fs_count == 4);
      CHECK(frag->per_vertex_count == 4);
      for (unsigned i = 0; i < frag->per_vertex_count; ++i) {
         unsigned idx = frag->per_vertex[i];
         CHECK(idx < fs_count);
         int mapped = agx_apple9_varying_index(&gs_varyings, fs_locs[idx], fs_comps[idx]);
         CHECK(mapped >= 0);
         if (fs_locs[idx] == VARYING_SLOT_VAR1)
            CHECK(mapped == 4);
      }
   }

   /* Verify serialization/deserialization roundtrip preserves the companion. */
   struct blob blob;
   blob_init(&blob);
   CHECK(hk_api_shader_serialize(&dev->vk, &fs->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *copy_vk = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &copy_vk) ==
         VK_SUCCESS);
   blob_finish(&blob);
   struct hk_api_shader *copy = container_of(copy_vk, struct hk_api_shader, vk);
   CHECK(copy->apple9_fs_native_per_vertex != NULL);
   CHECK(copy->apple9_fs_native_per_vertex->variants[0].b.info.apple9_per_vertex_count == 4);
   CHECK(copy->apple9_fs_native_per_vertex->variants[0].b.info.apple9_barycentric_mask == 1);
   CHECK(copy->apple9_fs_native_per_vertex->variants[0].b.info.apple9_varyings.mask[VARYING_SLOT_VAR1] == 0x1);

   copy->vk.ops->destroy(&dev->vk, &copy->vk, NULL);
   fs->vk.ops->destroy(&dev->vk, &fs->vk, NULL);
}

/* Cluster 5: fixed-topology GS main compute shader must strip Apple9 image
 * loads/stores/fences and any dead guarding atomics so side effects run only
 * once in the rasterization shader. */
static void
test_gs_strip_apple9_image_side_effects(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "c5 mutable storage image gs");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;

   nir_def *flag_addr = nir_imm_int64(&b, 0x4000);
   nir_def *flag = nir_global_atomic_swap(
      &b, 32, flag_addr, nir_imm_int(&b, 0), nir_imm_int(&b, 1),
      .atomic_op = nir_atomic_op_cmpxchg);
   nir_push_if(&b, nir_ieq_imm(&b, flag, 0));
   {
      nir_def *v = nir_image_load_apple9_agx(
         &b, 4, 32, nir_imm_int(&b, 0), nir_imm_ivec3(&b, 0, 0, 0),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0),
         .base = 0, .image_dim = GLSL_SAMPLER_DIM_2D);
      nir_image_store_apple9_agx(
         &b, nir_imm_int(&b, 0), nir_imm_ivec3(&b, 0, 0, 0), v,
         .image_dim = GLSL_SAMPLER_DIM_2D);
      nir_fence_image_write_apple9_agx(&b);
      nir_fence_tex_invalidate_apple9_agx(&b);
      nir_global_atomic(&b, 32, flag_addr, nir_imm_int(&b, 1),
                        .atomic_op = nir_atomic_op_iadd);
   }
   nir_pop_if(&b, NULL);

   for (unsigned v = 0; v < 3; ++v) {
      nir_store_output(&b, nir_imm_vec4(&b, v == 1 ? 1.0f : -1.0f,
                                        v == 2 ? 1.0f : -1.0f, 0.0f, 1.0f),
                       nir_imm_int(&b, 0), .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_emit_vertex(&b, .stream_id = 0);
   }
   nir_end_primitive(&b, .stream_id = 0);
   nir_shader_gather_info(b.shader, b.impl);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info gs_info = {0};
   CHECK(poly_nir_lower_gs(b.shader, &count, &rast, &pre, &gs_info));
   CHECK(rast != NULL);

   /* Main compute shader must not retain any image or atomic side effects. */
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         CHECK(op != nir_intrinsic_image_load_apple9_agx);
         CHECK(op != nir_intrinsic_image_store_apple9_agx);
         CHECK(op != nir_intrinsic_fence_image_write_apple9_agx);
         CHECK(op != nir_intrinsic_fence_tex_invalidate_apple9_agx);
         CHECK(op != nir_intrinsic_global_atomic);
         CHECK(op != nir_intrinsic_global_atomic_swap);
      }
   }

   /* Rasterization shader must preserve the side effects. */
   bool rast_has_store = false, rast_has_swap = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(rast)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         rast_has_store |= op == nir_intrinsic_image_store_apple9_agx;
         rast_has_swap |= op == nir_intrinsic_global_atomic_swap;
      }
   }
   CHECK(rast_has_store);
   CHECK(rast_has_swap);

   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
}

static nir_shader *
simple_vs_with_attrib(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "gpl robustness vs");
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;
   nir_def *attr = nir_load_input(
      &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
   nir_store_output(&b, attr, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

/* Bonus 7a: GPL fast-linked and LTO vertex shaders must keep bounds checks when
 * pipelineRobustness is enabled, while monolithic VS without robustness stays
 * unbounded. Also verify cache key hashes stay distinct. */
static void
test_gpl_vertex_pipeline_robustness(struct hk_physical_device *pdev,
                                    struct hk_device *dev)
{
   dev->vk.enabled_features.pipelineRobustness = true;

   const struct vk_vertex_input_state vi = {
      .bindings_valid = 1,
      .bindings = {{.stride = 16, .input_rate = VK_VERTEX_INPUT_RATE_VERTEX}},
      .attributes_valid = 1,
      .attributes = {{.binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0}},
   };
   const struct vk_multiview_state mv = {0};

   /* Monolithic with static VI and GPL disabled: unbounded stays enabled. */
   dev->vk.enabled_features.graphicsPipelineLibrary = false;
   const struct vk_graphics_pipeline_state mono_state = {.vi = &vi, .mv = &mv};
   struct vk_shader_compile_info mono_info = {
      .stage = MESA_SHADER_VERTEX,
      .nir = simple_vs_with_attrib(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *mono_out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &mono_info, &mono_state, NULL, NULL,
                                    NULL, &mono_out) == VK_SUCCESS);
   struct hk_api_shader *mono_vs = container_of(mono_out, struct hk_api_shader, vk);
   CHECK(mono_vs->variants[HK_VS_VARIANT_HW].info.apple9_vertex_unbounded);

   /* Fast-linked GPL library (no VI): must keep bounds checks. */
   const struct vk_graphics_pipeline_state lib_state = {.mv = &mv};
   struct vk_shader_compile_info lib_info = {
      .stage = MESA_SHADER_VERTEX,
      .nir = simple_vs_with_attrib(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *lib_out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &lib_info, &lib_state, NULL, NULL,
                                    NULL, &lib_out) == VK_SUCCESS);
   struct hk_api_shader *lib_vs = container_of(lib_out, struct hk_api_shader, vk);
   CHECK(!lib_vs->variants[HK_VS_VARIANT_HW].info.apple9_vertex_unbounded);

   /* Link-time-optimized GPL (static VI with graphicsPipelineLibrary enabled):
    * must also keep bounds checks since vk_graphics_pipeline_state does not
    * retain LIBRARY/LTO flags or VERTEX_INPUT_INTERFACE robustness. */
   dev->vk.enabled_features.graphicsPipelineLibrary = true;
   struct vk_shader_compile_info lto_info = {
      .stage = MESA_SHADER_VERTEX,
      .nir = simple_vs_with_attrib(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *lto_out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &lto_info, &mono_state, NULL, NULL,
                                    NULL, &lto_out) == VK_SUCCESS);
   struct hk_api_shader *lto_vs = container_of(lto_out, struct hk_api_shader, vk);
   CHECK(!lto_vs->variants[HK_VS_VARIANT_HW].info.apple9_vertex_unbounded);

   /* Shader Object (state == NULL) must stay invariant across pipeline-only
    * features (pipelineRobustness / graphicsPipelineLibrary) so binary shaders
    * match across devices with differing pipeline features. */
   struct vk_shader_compile_info eso_pr_info = {
      .stage = MESA_SHADER_VERTEX,
      .nir = simple_vs_with_attrib(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *eso_pr_out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &eso_pr_info, NULL, NULL, NULL,
                                    NULL, &eso_pr_out) == VK_SUCCESS);
   struct hk_api_shader *eso_pr_vs = container_of(eso_pr_out, struct hk_api_shader, vk);
   CHECK(eso_pr_vs->variants[HK_VS_VARIANT_HW].info.apple9_vertex_unbounded);

   dev->vk.enabled_features.graphicsPipelineLibrary = false;
   dev->vk.enabled_features.pipelineRobustness = false;
   struct vk_shader_compile_info eso_min_info = {
      .stage = MESA_SHADER_VERTEX,
      .nir = simple_vs_with_attrib(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *eso_min_out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &eso_min_info, NULL, NULL, NULL,
                                    NULL, &eso_min_out) == VK_SUCCESS);
   struct hk_api_shader *eso_min_vs = container_of(eso_min_out, struct hk_api_shader, vk);
   CHECK(eso_min_vs->variants[HK_VS_VARIANT_HW].info.apple9_vertex_unbounded);
   CHECK(eso_pr_vs->variants[HK_VS_VARIANT_HW].code_size ==
         eso_min_vs->variants[HK_VS_VARIANT_HW].code_size);
   CHECK(memcmp(eso_pr_vs->variants[HK_VS_VARIANT_HW].code_ptr,
                eso_min_vs->variants[HK_VS_VARIANT_HW].code_ptr,
                eso_pr_vs->variants[HK_VS_VARIANT_HW].code_size) == 0);

   /* Verify graphics state hashes separate GPL/pipelineRobustness combinations. */
   struct vk_features feats_pr_only = {.pipelineRobustness = true};
   struct vk_features feats_pr_gpl = {
      .pipelineRobustness = true,
      .graphicsPipelineLibrary = true,
   };
   struct vk_features feats_gpl_only = {.graphicsPipelineLibrary = true};
   blake3_hash h_pr_only, h_pr_gpl, h_gpl_only;
   hk_hash_graphics_state(&pdev->vk, &mono_state, &feats_pr_only,
                          VK_SHADER_STAGE_VERTEX_BIT, h_pr_only);
   hk_hash_graphics_state(&pdev->vk, &mono_state, &feats_pr_gpl,
                          VK_SHADER_STAGE_VERTEX_BIT, h_pr_gpl);
   hk_hash_graphics_state(&pdev->vk, &mono_state, &feats_gpl_only,
                          VK_SHADER_STAGE_VERTEX_BIT, h_gpl_only);
   CHECK(memcmp(h_pr_only, h_pr_gpl, sizeof(h_pr_only)) != 0);
   CHECK(memcmp(h_pr_gpl, h_gpl_only, sizeof(h_pr_gpl)) != 0);

   mono_out->ops->destroy(&dev->vk, mono_out, NULL);
   lib_out->ops->destroy(&dev->vk, lib_out, NULL);
   lto_out->ops->destroy(&dev->vk, lto_out, NULL);
   eso_pr_out->ops->destroy(&dev->vk, eso_pr_out, NULL);
   eso_min_out->ops->destroy(&dev->vk, eso_min_out, NULL);
}

/* Board #378: geometry shader without VARYING_SLOT_POS must not synthesize
 * software polygon-mode POINT variant. */
static void
test_no_position_gs_polygon_points_rejected(struct hk_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "no-pos gs");
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   for (unsigned v = 0; v < 3; ++v) {
      nir_store_output(&b, nir_imm_vec4(&b, 1, 2, 3, 4), nir_imm_int(&b, 0),
                       .write_mask = 0xf, .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_emit_vertex(&b, .stream_id = 0);
   }
   nir_end_primitive(&b, .stream_id = 0);
   nir_shader_gather_info(b.shader, b.impl);

   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_GEOMETRY,
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
   };
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key_device(&key, NULL, hk_device_physical(dev));
   key.may_polygon_points = true;
   struct vk_shader *out = NULL;
   CHECK(hk_apple9_compile_graphics(dev, 1, &info, NULL, &key, NULL, NULL,
                                    &out) == VK_SUCCESS);
   struct hk_api_shader *gs = container_of(out, struct hk_api_shader, vk);
   CHECK(gs->apple9_gs_polygon_points == NULL);
   out->ops->destroy(&dev->vk, out, NULL);
}

/* Bonus 7b: under shared_memory_explicit_layout, Workgroup OpTypePointer casts
 * have ptr_stride == 0 (ignoring ArrayStride on OpTypePointer). Verify both:
 *  1) direct array-derived ptr_as_array inherits the array's explicit stride (4)
 *  2) bcsel-merged variable pointers recover the incoming arrays' explicit stride (16)
 *     without hitting assert(stride > 0) in nir_lower_explicit_io. */
static void
test_workgroup_explicit_layout_ptr_as_array(void)
{
   /* Case 1: direct array element -> ptr_stride=0 cast -> ptr_as_array(+1) */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "c7b direct explicit wg");
      b.shader->info.shared_memory_explicit_layout = true;
      const struct glsl_type *arr_t = glsl_array_type(glsl_uint_type(), 17, 4);
      struct glsl_struct_field field = {
         .type = arr_t,
         .name = "a",
         .offset = 0,
      };
      const struct glsl_type *blk_t =
         glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false, "wg");
      nir_variable *var = nir_variable_create(b.shader, nir_var_mem_shared, blk_t, "wg");
      nir_def *idx = nir_imm_int(&b, 3);
      nir_deref_instr *elem = nir_build_deref_array(
         &b, nir_build_deref_struct(&b, nir_build_deref_var(&b, var), 0), idx);
      nir_deref_instr *cast = nir_build_deref_cast(
         &b, &elem->def, nir_var_mem_shared, glsl_uint_type(), 0);
      nir_deref_instr *next = nir_build_deref_ptr_as_array(&b, cast, nir_imm_int(&b, 1));
      nir_def *val = nir_load_deref(&b, next);
      nir_store_global(&b, val, nir_imm_int64(&b, 0x3000));

      CHECK(nir_opt_deref(b.shader));
      nir_lower_vars_to_explicit_types(b.shader, nir_var_mem_shared,
                                       glsl_get_cl_type_size_align);
      CHECK(nir_lower_explicit_io(b.shader, nir_var_mem_shared,
                                  nir_address_format_32bit_offset));
      nir_opt_constant_folding(b.shader);
      nir_opt_algebraic(b.shader);
      nir_opt_dce(b.shader);

      bool saw_addr16 = false;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_shared) {
               CHECK(nir_src_is_const(intr->src[0]));
               /* Element 3 + 1 at ArrayStride 4 -> byte offset 16 (not 20). */
               CHECK(nir_src_as_uint(intr->src[0]) + nir_intrinsic_base(intr) == 16);
               saw_addr16 = true;
            }
         }
      }
      CHECK(saw_addr16);
      ralloc_free(b.shader);
   }

   /* Case 2: deeply nested (12-layer) bcsel of two explicit stride-16 shared
    * array pointers -> ptr_stride=0 cast -> ptr_as_array(+1) must recover
    * stride 16 without any depth-limit fallback to natural stride 4. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "c7b deep bcsel explicit wg");
      b.shader->info.shared_memory_explicit_layout = true;
      const struct glsl_type *arr_t = glsl_array_type(glsl_uint_type(), 8, 16);
      struct glsl_struct_field field = {
         .type = arr_t,
         .name = "a",
         .offset = 0,
      };
      const struct glsl_type *blk_t =
         glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false, "wg16");
      nir_variable *v0 = nir_variable_create(b.shader, nir_var_mem_shared, blk_t, "wg0");
      nir_variable *v1 = nir_variable_create(b.shader, nir_var_mem_shared, blk_t, "wg1");
      nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_deref_instr *e0 = nir_build_deref_array(
         &b, nir_build_deref_struct(&b, nir_build_deref_var(&b, v0), 0), nir_imm_int(&b, 0));
      nir_deref_instr *e1 = nir_build_deref_array(
         &b, nir_build_deref_struct(&b, nir_build_deref_var(&b, v1), 0), nir_imm_int(&b, 0));
      nir_def *p0 = &e0->def;
      nir_def *p1 = &e1->def;
      for (unsigned depth = 0; depth < 12; ++depth) {
         nir_def *cond = nir_ine_imm(&b, nir_iand_imm(&b, lid, 1u << depth), 0);
         nir_def *n0 = nir_bcsel(&b, cond, p0, p1);
         nir_def *n1 = nir_bcsel(&b, cond, p1, p0);
         p0 = n0;
         p1 = n1;
      }
      nir_deref_instr *cast = nir_build_deref_cast(
         &b, p0, nir_var_mem_shared, glsl_uint_type(), 0);
      nir_deref_instr *next = nir_build_deref_ptr_as_array(&b, cast, nir_imm_int(&b, 1));
      nir_def *val = nir_load_deref(&b, next);
      nir_store_global(&b, val, nir_imm_int64(&b, 0x3000));

      nir_opt_deref(b.shader);
      CHECK(nir_lower_vars_to_explicit_types(b.shader, nir_var_mem_shared,
                                             glsl_get_cl_type_size_align));
      CHECK(nir_lower_explicit_io(b.shader, nir_var_mem_shared,
                                  nir_address_format_32bit_offset));
      nir_opt_constant_folding(b.shader);
      nir_opt_algebraic(b.shader);
      nir_opt_dce(b.shader);

      bool saw_stride16 = false;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_shared) {
               nir_instr *src_i = nir_def_instr(intr->src[0].ssa);
               CHECK(src_i->type == nir_instr_type_alu);
               nir_alu_instr *alu = nir_instr_as_alu(src_i);
               CHECK(alu->op == nir_op_iadd);
               bool has_16 =
                  (nir_src_is_const(alu->src[0].src) &&
                   nir_src_as_uint(alu->src[0].src) == 16) ||
                  (nir_src_is_const(alu->src[1].src) &&
                   nir_src_as_uint(alu->src[1].src) == 16);
               CHECK(has_16);
               saw_stride16 = true;
            }
         }
      }
      CHECK(saw_stride16);
      ralloc_free(b.shader);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_physical_device pdev;
   struct hk_device dev;
   init_test_device(&pdev, &dev);

   test_input_attachment_offset_and_high_index(&dev);
   test_volatile_bindless_image_texel_address();
   test_gpl_per_vertex_fs_native_companion(&dev);
   test_gs_strip_apple9_image_side_effects();
   test_gpl_vertex_pipeline_robustness(&pdev, &dev);
   test_no_position_gs_polygon_points_rejected(&dev);
   test_workgroup_explicit_layout_ptr_as_array();

   finish_test_device(&pdev, &dev);
   glsl_type_singleton_decref();
   return 0;
}
