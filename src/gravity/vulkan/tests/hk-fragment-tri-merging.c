/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Triangle merging shades adjacent strip and fan triangles on one plane, which
 * is inexact away from the pixel centre. Compile real fragment shaders (BO
 * allocation replaced with host memory) and check that off-centre
 * interpolation disables merging whether or not the statistics counter is
 * compiled in, that pixel-centre interpolation keeps it, and that the choice
 * survives serialization.
 */
#include "../hk_shader.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
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
   bo->_map = calloc(1, size);
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

static nir_shader *
mesh_source(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                                  "tri merging mesh");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
      b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3), nir_imm_int(&b, 1),
                                      nir_imm_int(&b, 0));
   for (unsigned v = 0; v < 3; ++v) {
      nir_store_per_vertex_output(&b,
         nir_imm_vec4(&b, v == 0 ? -.5 : .5, v == 2 ? .5 : -.5, .5, 1),
         nir_imm_int(&b, v), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, v, 1, 2, 3),
         nir_imm_int(&b, v), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
fragment_source(bool centroid)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                                  "tri merging fragment");
   /* An application shader: hk does not gate internal ones. */
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;
   nir_def *bary = centroid
      ? nir_load_barycentric_centroid(&b, 32, .interp_mode = INTERP_MODE_SMOOTH)
      : nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *value = nir_load_interpolated_input(&b, 4, 32, bary, nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static bool
roundtrip_disables(struct hk_device *dev, struct hk_api_shader *fs)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(hk_api_shader_serialize(&dev->vk, &fs->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *result = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &result) == VK_SUCCESS);
   blob_finish(&blob);
   struct hk_api_shader *copy = container_of(result, struct hk_api_shader, vk);
   bool disable = copy->variants[0].b.info.disable_tri_merging;
   copy->vk.ops->destroy(&dev->vk, &copy->vk, NULL);
   return disable;
}

/* Returns whether the compiled fragment shader disables triangle merging. */
static bool
disables_merging(struct hk_device *dev, bool centroid, bool statistics)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH, .nir = mesh_source(),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT, .nir = fragment_source(centroid),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1, .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);
   key.no_fs_statistics = !statistics;
   CHECK(hk_apple9_compile_graphics(dev, 2, infos, NULL, &key, NULL, NULL, out) ==
         VK_SUCCESS);
   struct hk_api_shader *fs = container_of(out[1], struct hk_api_shader, vk);
   bool disable = fs->variants[0].b.info.disable_tri_merging;
   CHECK(roundtrip_disables(dev, fs) == disable);
   for (unsigned i = 0; i < 2; ++i)
      out[i]->ops->destroy(&dev->vk, out[i], NULL);
   return disable;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);

   /* Off-centre interpolation keeps triangles apart, counter or not. */
   CHECK(disables_merging(dev, true, false));
   CHECK(disables_merging(dev, true, true));
   /* Pixel-centre interpolation merges, also with the counter. */
   CHECK(!disables_merging(dev, false, false));
   CHECK(!disables_merging(dev, false, true));

   /* Cache key distinguishes static vs dynamic polygon mode / line stipple / GPL. */
   {
      const struct vk_rasterization_state rs_fill = {
         .polygon_mode = VK_POLYGON_MODE_FILL,
      };
      struct vk_graphics_pipeline_state s_static = {.rs = &rs_fill};
      struct vk_graphics_pipeline_state s_dyn_poly = s_static;
      BITSET_SET(s_dyn_poly.dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);
      struct vk_graphics_pipeline_state s_dyn_stipple = s_static;
      BITSET_SET(s_dyn_stipple.dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
      struct vk_graphics_pipeline_state s_gpl = s_static;
      s_gpl.pipeline_flags = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR;

      struct hk_apple9_graphics_key k_static, k_dyn_poly, k_dyn_stipple, k_gpl;
      hk_apple9_graphics_key(&k_static, &s_static);
      hk_apple9_graphics_key(&k_dyn_poly, &s_dyn_poly);
      hk_apple9_graphics_key(&k_dyn_stipple, &s_dyn_stipple);
      hk_apple9_graphics_key(&k_gpl, &s_gpl);

      CHECK(!k_static.may_polygon_points && !k_static.may_line_stipple &&
            !k_static.is_library);
      CHECK(k_dyn_poly.may_polygon_points && !k_dyn_poly.polygon_points);
      CHECK(k_dyn_stipple.may_line_stipple && !k_dyn_stipple.line_stipple);
      CHECK(k_gpl.is_library);
      CHECK(memcmp(&k_static, &k_dyn_poly, sizeof(k_static)) != 0);
      CHECK(memcmp(&k_static, &k_dyn_stipple, sizeof(k_static)) != 0);
      CHECK(memcmp(&k_static, &k_gpl, sizeof(k_static)) != 0);
   }

   /* Feature key distinguishes pipelineStatisticsQuery & primitivesGeneratedQuery. */
   {
      struct vk_features base = {0};
      struct vk_features f_stats = {.pipelineStatisticsQuery = true};
      struct vk_features f_pgq = {.primitivesGeneratedQuery = true};
      CHECK(hk_make_feature_key(&base) != hk_make_feature_key(&f_stats));
      CHECK(hk_make_feature_key(&base) != hk_make_feature_key(&f_pgq));
      CHECK(hk_make_feature_key(&f_stats) != hk_make_feature_key(&f_pgq));
   }

   /* Application line-strip GS with dynamic line stipple retains TEX6/TEX7
    * exports while leaving FS unstippled for the draw-time stipple prolog. */
   {
      nir_builder gb = nir_builder_init_simple_shader(
         MESA_SHADER_GEOMETRY, &agx_nir_options, "gs_dyn_stipple");
      gb.shader->info.internal = false;
      gb.shader->info.gs.input_primitive = MESA_PRIM_LINES;
      gb.shader->info.gs.output_primitive = MESA_PRIM_LINE_STRIP;
      gb.shader->info.gs.vertices_in = 2;
      gb.shader->info.gs.vertices_out = 2;
      gb.shader->info.gs.invocations = 1;
      gb.shader->info.gs.active_stream_mask = 1;
      for (unsigned v = 0; v < 2; ++v) {
         nir_store_output(&gb, nir_imm_vec4(&gb, v ? 1.0f : -1.0f, 0, 0, 1),
                          nir_imm_int(&gb, 0), .write_mask = 0xf,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS,
                                           .num_slots = 1});
         nir_emit_vertex(&gb, 0);
      }
      nir_end_primitive(&gb, 0);
      nir_shader_gather_info(gb.shader, nir_shader_get_entrypoint(gb.shader));

      struct vk_shader_compile_info infos[] = {
         {.stage = MESA_SHADER_GEOMETRY, .nir = gb.shader,
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = fragment_source(false),
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out[2] = {0};
      const struct vk_rasterization_state rs_fill = {
         .polygon_mode = VK_POLYGON_MODE_FILL,
      };
      const struct vk_render_pass_state rp = {
         .color_attachment_count = 1,
         .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
      };
      const struct vk_color_blend_state cb = {
         .attachment_count = 1, .color_write_enables = 1,
         .attachments = {{.write_mask = 15}},
      };
      struct vk_graphics_pipeline_state st = {.rs = &rs_fill, .rp = &rp, .cb = &cb};
      BITSET_SET(st.dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
      CHECK(hk_apple9_compile_graphics(dev, 2, infos, &st, NULL, NULL, NULL, out) ==
            VK_SUCCESS);
      struct hk_api_shader *gs = container_of(out[0], struct hk_api_shader, vk);
      struct hk_api_shader *fs = container_of(out[1], struct hk_api_shader, vk);
      struct hk_shader *gs_rast = &gs->variants[HK_GS_VARIANT_RAST];
      CHECK((gs_rast->b.info.apple9_varyings.mask[VARYING_SLOT_TEX6] & 0x7) == 0x7);
      CHECK((gs_rast->b.info.apple9_varyings.mask[VARYING_SLOT_TEX7] & 0x3) == 0x3);
      CHECK(fs->variants[0].info.fs.line_stipple_start == 0);
      for (unsigned i = 0; i < 2; ++i)
         out[i]->ops->destroy(&dev->vk, out[i], NULL);
   }

   /* Linked FS pipeline-statistics prolog: verify standalone prolog ABI
    * (no preamble, slot 0 = sysvals, slot 1 = root) and that normal epilog FS
    * omit the inline counter even with pipelineStatisticsQuery enabled, while
    * early_fragment_tests and sample_shading keep it inline (and persist the
    * inline_statistics flag across serialization). */
   {
      nir_shader *prolog_nir = hk_apple9_build_fs_stats_prolog_nir();
      struct agx_apple9_blend pblend = {
         .use_epilog = true,
         .dynamic_samples = true,
         .writes_memory_stats_only = true,
      };
      struct agx_shader_part part = {0};
      const char *reason = NULL;
      CHECK(agx_compile_apple9_fragment_mrt(prolog_nir, NULL, &pblend, 0,
                                            &part, &reason));
      CHECK(part.info.apple9_preamble_size == 0);
      CHECK(part.info.apple9_resource_count == 2);
      CHECK(part.info.apple9_resource_binding[0] ==
            AGX_APPLE9_GRAPHICS_SYSVAL_BINDING);
      CHECK(part.info.apple9_resource_binding[1] == 0);
      free(part.binary);
      ralloc_free(prolog_nir);

      dev->vk.enabled_features.pipelineStatisticsQuery = true;
      const struct vk_render_pass_state rp = {
         .color_attachment_count = 1,
         .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
      };
      const struct vk_color_blend_state cb = {
         .attachment_count = 1, .color_write_enables = 1,
         .attachments = {{.write_mask = 15}},
      };

      /* Normal epilog FS: no inline statistics counter. */
      struct vk_shader_compile_info infos_normal[] = {
         {.stage = MESA_SHADER_MESH, .nir = mesh_source(),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = fragment_source(false),
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out_normal[2] = {0};
      const struct vk_graphics_pipeline_state st_normal = {.rp = &rp, .cb = &cb};
      CHECK(hk_apple9_compile_graphics(dev, 2, infos_normal, &st_normal, NULL,
                                       NULL, NULL, out_normal) == VK_SUCCESS);
      struct hk_api_shader *fs_normal =
         container_of(out_normal[1], struct hk_api_shader, vk);
      CHECK(!fs_normal->variants[0].info.fs.inline_statistics);
      size_t normal_size = fs_normal->variants[0].code_size;

      /* early_fragment_tests FS: keeps inline statistics counter. */
      nir_shader *early_nir = fragment_source(false);
      early_nir->info.fs.early_fragment_tests = true;
      struct vk_shader_compile_info infos_early[] = {
         {.stage = MESA_SHADER_MESH, .nir = mesh_source(),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = early_nir,
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out_early[2] = {0};
      CHECK(hk_apple9_compile_graphics(dev, 2, infos_early, &st_normal, NULL,
                                       NULL, NULL, out_early) == VK_SUCCESS);
      struct hk_api_shader *fs_early =
         container_of(out_early[1], struct hk_api_shader, vk);
      CHECK(fs_early->variants[0].info.fs.inline_statistics);
      CHECK(fs_early->variants[0].code_size > normal_size + 200);

      /* sample_shading FS: keeps inline statistics counter. */
      struct vk_shader_compile_info infos_ss[] = {
         {.stage = MESA_SHADER_MESH, .nir = mesh_source(),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = fragment_source(false),
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out_ss[2] = {0};
      const struct vk_multisample_state ms_ss = {
         .rasterization_samples = 4,
         .sample_shading_enable = true,
         .sample_mask = 0xf,
      };
      const struct vk_graphics_pipeline_state st_ss = {
         .rp = &rp, .cb = &cb, .ms = &ms_ss,
      };
      CHECK(hk_apple9_compile_graphics(dev, 2, infos_ss, &st_ss, NULL,
                                       NULL, NULL, out_ss) == VK_SUCCESS);
      struct hk_api_shader *fs_ss =
         container_of(out_ss[1], struct hk_api_shader, vk);
      CHECK(fs_ss->variants[0].info.fs.inline_statistics);

      /* Serialization preserves inline_statistics (both false and true). */
      struct hk_api_shader *check_shaders[2] = {fs_normal, fs_early};
      for (unsigned k = 0; k < 2; ++k) {
         struct blob blob;
         blob_init(&blob);
         CHECK(hk_api_shader_serialize(&dev->vk, &check_shaders[k]->vk, &blob));
         struct blob_reader reader;
         blob_reader_init(&reader, blob.data, blob.size);
         struct vk_shader *deser = NULL;
         CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &deser) ==
               VK_SUCCESS);
         blob_finish(&blob);
         struct hk_api_shader *copy =
            container_of(deser, struct hk_api_shader, vk);
         CHECK(copy->variants[0].info.fs.inline_statistics == (k == 1));
         copy->vk.ops->destroy(&dev->vk, &copy->vk, NULL);
      }

      for (unsigned i = 0; i < 2; ++i) {
         out_normal[i]->ops->destroy(&dev->vk, out_normal[i], NULL);
         out_early[i]->ops->destroy(&dev->vk, out_early[i], NULL);
         out_ss[i]->ops->destroy(&dev->vk, out_ss[i], NULL);
      }
      dev->vk.enabled_features.pipelineStatisticsQuery = false;
   }

   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   puts("off-centre interpolation disables triangle merging with and without statistics");
   return 0;
}
