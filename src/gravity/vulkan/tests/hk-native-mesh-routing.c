/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "../hk_buffer.h"
#include "../hk_cmd_buffer.h"
#include "../hk_entrypoints.h"
#include "../hk_physical_device.h"
#include "../hk_shader.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct hk_api_shader *
shader(mesa_shader_stage stage)
{
   struct hk_api_shader *s =
      calloc(1, sizeof(*s) + hk_num_variants(stage) * sizeof(struct hk_shader));
   CHECK(s);
   s->vk.stage = stage;
   return s;
}

static void
routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *mesh = shader(MESA_SHADER_MESH);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *nm = shader(MESA_SHADER_MESH),
                        *nf = shader(MESA_SHADER_FRAGMENT);
   mesh->apple9_native = nm;
   fs->apple9_native = nf;
   nm->apple9_native_leaf = nf->apple9_native_leaf = true;
   memset(nm->apple9_native_pair_id, 0x57, sizeof(nm->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, nm->apple9_native_pair_id,
          sizeof(nf->apple9_native_pair_id));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage =
      HK_APPLE9_NATIVE_MESH;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.guard_empty_grid = true;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 3;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 4;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   const uint32_t direct[] = {2, 3, 4}, oversized[] = {1024, 1024, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Missing DRM_ASAHI_FEATURE_NATIVE_MESH refuses native submission even with
    * HK_APPLE9_NATIVE_MESH=1; advertising bit 28 admits it. */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint64_t saved_features = dev->dev.params.features;
   dev->dev.params.features = 0;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.params.features = saved_features;
   /* Guarded taskless indirect and indirect-count draws are admitted when the
    * output footprint is concurrent-safe (<= 64 rows/WG); invalid direct+count
    * and oversized direct grids still fall back. */
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0x1000));
   CHECK(!hk_apple9_use_native_mesh(cmd, oversized, 0));
   /* An unguarded binary (e.g. production 31-class cache) must fall back on
    * conditional rendering, indirect, and indirect-count draws. */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.guard_empty_grid = false;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   cmd->state.cond_render.active = true;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   cmd->state.cond_render.active = false;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.guard_empty_grid = true;
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));                       \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));                        \
   } while (0)
   FALLBACK(mesh->apple9_native, NULL);
   FALLBACK(fs->apple9_native, NULL);
   FALLBACK(nf->apple9_native_pair_id[0], 0x35);
   FALLBACK(nf->apple9_native_pair_id[31], 0x35);
   FALLBACK(nf->apple9_native_leaf, false);
   /* Deferred native variants are themselves leaves. Direct draws route
    * identically to eager alternatives, while indirect/indirect-count draws
    * fall back because standalone leaves lack paired emulated overflow shaders. */
   gfx->shaders[MESA_SHADER_MESH] = nm;
   gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   FALLBACK(nf->apple9_native_pair_id[7], 0x35);
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   FALLBACK(gfx->shaders[MESA_SHADER_TASK], mesh);
   FALLBACK(gfx->shaders[MESA_SHADER_GEOMETRY], mesh);
   FALLBACK(gfx->shaders[MESA_SHADER_TESS_EVAL], mesh);
   /* Multiview requires a matching native_multiview binary with packed
    * Layer/Viewport output, uses effective layer count from view_mask even
    * when VkRenderingInfo.layerCount == 0, and rejects mismatched binaries. */
   FALLBACK(gfx->render.view_mask, 3);
   nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_multiview = true;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->render.view_mask = 3;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0)); /* missing writes_layer_viewport */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.writes_layer_viewport = true;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   gfx->render.layer_count = 0; /* legal with nonzero viewMask */
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   gfx->render.layer_count = 1;
   gfx->render.view_mask = 0;
   nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_multiview = false;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.writes_layer_viewport = false;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Multi-layer (1..=2048) and MSAA (1x/2x/4x) are admitted under
    * HK_APPLE9_NATIVE_MESH=1 (paired with kernel 932adc520b3c); out-of-range
    * layer counts and sample counts still fall back cleanly. */
   FALLBACK(gfx->render.layer_count, 0);
   FALLBACK(gfx->render.layer_count, 2049);
   gfx->render.layer_count = 2;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->render.layer_count = 2048;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->render.layer_count = 1;
   FALLBACK(gfx->mesh_pg_query, 0x2000);
   FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags,
            VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT);
   /* Occlusion queries are handled in the 3D/ISP fragment stage and admitted
    * under HK_APPLE9_NATIVE_MESH=1. */
   gfx->occlusion.active_count = 1;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->occlusion.active_count = 0;
   /* Conditional rendering is admitted for guarded taskless direct, indirect,
    * and indirect-count draws. */
   cmd->state.cond_render.active = true;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   cmd->state.cond_render.active = false;
   /* VK_POLYGON_MODE_LINE is admitted natively without stipple or software
    * degenerate-index culling; VK_POLYGON_MODE_POINT stays emulated. */
   dyn->rs.polygon_mode = VK_POLYGON_MODE_LINE;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   dyn->rs.line.stipple.enable = true;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   dyn->rs.line.stipple.enable = false;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_POINT);
   dyn->ms.rasterization_samples = 2;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   dyn->ms.rasterization_samples = 4;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   FALLBACK(dyn->ms.rasterization_samples, 8);
   dyn->ms.rasterization_samples = 0;
   gfx->render.tilebuffer.nr_samples = 2;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->render.tilebuffer.nr_samples = 4;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   FALLBACK(gfx->render.tilebuffer.nr_samples, 8);
   gfx->render.tilebuffer.nr_samples = 0;
   FALLBACK(gfx->xfb_enabled, true);
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.primitive_broadcast =
      VARYING_BIT_CULL_PRIMITIVE;
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   /* CullDistance alone also removes a whole triangle with degenerate
    * indices. A cached native pair must fall back under conservative raster
    * or wireframe line mode. */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.primitive_broadcast = 0;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.cull_distances = 2;
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT);
   gfx->shaders[MESA_SHADER_MESH] = nm;
   gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.cull_distances = 0;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.cull_primitive = true;
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT);
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.cull_primitive = false;
   gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Hardware 64 KiB mesh output ring gate:
    * - V=128, n=8, P=32 (vtx_rows=256, idx_rows=8) has a wrap-safe U0_B stride
    *   so direct and indirect draws of any grid size stay native/unbounded.
    * - V=256, n=28, P=32 (vtx_rows=1792, idx_rows=8, already at max U0_B=256
    *   with 4096 % 1800 = 496 < 1792) has no safe U0_B and admits at most
    *   4096 / 1800 = 2 workgroups (indirect uses GPU max_groups=2 + emulated
    *   overflow fallback).
    * - A single-WG ring overflow (V=256, n=128) has max_groups=0 and falls
    *   back even for indirect draws. */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 128;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 8;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_primitives = 32;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.indices_per_primitive = 3;
   const uint32_t ring_16[] = {16, 1, 1}, ring_64[] = {64, 1, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, ring_16, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, ring_64, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));

   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 256;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 28;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_primitives = 32;
   const uint32_t ring_fit[] = {2, 1, 1}, ring_wrap[] = {3, 1, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, ring_fit, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_wrap, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_64, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 256;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 128;
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 256;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 28;
   /* Multi-axis product crossing the same threshold falls back too. */
   const uint32_t ring_2d_fit[] = {1, 2, 1}, ring_2d_wrap[] = {2, 2, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, ring_2d_fit, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_2d_wrap, 0));
   /* Direct predicated draws honor the same known-grid ring cap. */
   cmd->state.cond_render.active = true;
   CHECK(hk_apple9_use_native_mesh(cmd, ring_fit, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_wrap, 0));
   cmd->state.cond_render.active = false;
#undef FALLBACK
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(mesh);
   free(nm);
   free(fs);
   free(nf);
}

static void
gs_routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = shader(MESA_SHADER_GEOMETRY);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *ng = shader(MESA_SHADER_GEOMETRY);
   struct hk_api_shader *nv = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *nf = shader(MESA_SHADER_FRAGMENT);
   gs->apple9_native = ng; vs->apple9_native = nv; fs->apple9_native = nf;
   ng->apple9_native_leaf = nv->apple9_native_leaf = nf->apple9_native_leaf = true;
   memset(ng->apple9_native_pair_id, 0x71, sizeof(ng->apple9_native_pair_id));
   memcpy(nv->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   struct hk_shader *main = hk_native_producer_main(ng);
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_native_producer_object(ng)->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   main->info.apple9_native.key.physical_threads = 16;
   main->info.apple9_native.layout.max_vertices = 3;
   main->info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_GEOMETRY] = gs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   bool enabled = hk_apple9_native_gs_enabled();
   CHECK(hk_apple9_use_native_gs(cmd, draw) == enabled);
   if (enabled) {
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_gs(cmd, draw));                               \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_gs(cmd, draw));                                \
   } while (0)
      FALLBACK(cmd->in_meta, true);
      FALLBACK(gs->apple9_native, NULL);
      FALLBACK(vs->apple9_native, NULL);
      FALLBACK(fs->apple9_native, NULL);
      FALLBACK(ng->apple9_native_leaf, false);
      FALLBACK(nv->apple9_native_pair_id[0], 0x13);
      FALLBACK(nf->apple9_native_pair_id[31], 0x13);
      FALLBACK(gs->is_passthrough, true);
      FALLBACK(nv->writes_memory, true);
      FALLBACK(main->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(hk_native_producer_object(ng)->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(main->info.apple9_native.key.physical_threads, 32);
      FALLBACK(main->info.apple9_native.layout.max_vertices, 5);
      FALLBACK(draw.restart, true);
      /* Every triangle of a direct list and every instance is one group. */
#define ADMITTED(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(hk_apple9_use_native_gs(cmd, draw));                                \
      field = saved;                                                           \
   } while (0)
      ADMITTED(draw.b.count[0], 6);
      ADMITTED(draw.b.count[0], 3 * HK_APPLE9_NATIVE_MAX_GROUP_COUNT);
      ADMITTED(draw.b.count[1], 2);
      ADMITTED(draw.b.count[1], HK_APPLE9_NATIVE_MAX_GROUP_COUNT);
      /* The software VS applies indices, firstIndex/vertexOffset and
       * firstInstance before the GS reads its corners. */
      ADMITTED(draw.indexed, true);
      ADMITTED(draw.start, 1);
      ADMITTED(draw.start_instance, 1);
      FALLBACK(draw.b.count[0], 4);
      FALLBACK(draw.b.count[0], 2);
      FALLBACK(draw.b.count[0], 3 * (HK_APPLE9_NATIVE_MAX_GROUP_COUNT + 1));
      FALLBACK(draw.b.count[1], 0);
      FALLBACK(draw.b.count[1], HK_APPLE9_NATIVE_MAX_GROUP_COUNT + 1);
      FALLBACK(draw.b.count[2], 2);
      /* GS invocations multiply the groups: 32 invocations of 1,024
       * triangles fill 32,768 groups, of 2,048 triangles exceed the axis. */
      main->info.apple9_gs_invocations = 32;
      ADMITTED(draw.b.count[0], 3 * 1024);
      FALLBACK(draw.b.count[0], 3 * 2048);
      main->info.apple9_gs_invocations = 0;
      /* The grid total is bounded separately from each axis. */
      draw.b.count[0] = 3 * 1024;
      FALLBACK(draw.b.count[1], 1024);
      draw.b.count[0] = 3;
#undef ADMITTED
      CHECK(!hk_apple9_use_native_gs(cmd, agx_draw_indirect(0x1000)));
      FALLBACK(dyn->ia.primitive_restart_enable, true);
      /* Triangle strips and fans map corners in the body (hk_nir_gs_mesh). */
      dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      FALLBACK(dyn->ia.primitive_topology, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
      FALLBACK(dyn->ia.primitive_topology,
               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_CTRL], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_EVAL], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TASK], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_MESH], gs);
      FALLBACK(gfx->render.view_mask, 3);
      FALLBACK(gfx->render.layer_count, 0);
      FALLBACK(gfx->render.layer_count, 2049);
      gfx->render.layer_count = 4;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      gfx->render.layer_count = 1;
      FALLBACK(cmd->state.cond_render.active, true);
      FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags, 1);
      FALLBACK(gfx->occlusion.active_count, 1);
      FALLBACK(gfx->mesh_pg_query, 0x1000);
      FALLBACK(gfx->pg_query[0], 0x1000);
      FALLBACK(gfx->xfb_query[3], 0x1000);
      FALLBACK(gfx->xfb_enabled, true);
      FALLBACK(dyn->rs.rasterization_stream, 1);
      FALLBACK(dyn->rs.rasterizer_discard_enable, true);
      FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
      FALLBACK(dyn->rs.conservative_mode, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
      dyn->ms.rasterization_samples = 4;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      FALLBACK(dyn->ms.rasterization_samples, 8);
      dyn->ms.rasterization_samples = 0;
      gfx->render.tilebuffer.nr_samples = 4;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      FALLBACK(gfx->render.tilebuffer.nr_samples, 8);
      gfx->render.tilebuffer.nr_samples = 0;
#undef FALLBACK
      main->info.apple9_native.layout.max_vertices = 4;
      main->info.apple9_native.layout.max_primitives = 2;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      /* Hardware 64 KiB mesh output ring gate also protects native GS:
       * V=4, n=128 is now unbounded via safe-stride U0_B (stride=260, rem=196 >= 128),
       * while n=260 exceeds safe U0_B and stays bounded by 4096 / (260 + 8) = 15 groups. */
      main->info.apple9_native.layout.vertex_components = 128;
      draw.b.count[0] = 3 * 31;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 260;
      draw.b.count[0] = 3 * 15;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      draw.b.count[0] = 3 * 16;
      CHECK(!hk_apple9_use_native_gs(cmd, draw));
      draw.b.count[0] = 3;
      main->info.apple9_native.layout.vertex_components = 0;
      gfx->shaders[MESA_SHADER_GEOMETRY] = ng;
      gfx->shaders[MESA_SHADER_VERTEX] = nv;
      gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
   }
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(gs); free(vs); free(fs); free(ng); free(nv); free(nf);
}

static void
task_routes(struct hk_cmd_buffer *cmd)
{
   if (!hk_apple9_native_task_bridge_enabled())
      return;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *mesh = shader(MESA_SHADER_MESH);
   struct hk_api_shader *task = shader(MESA_SHADER_TASK);
   mesh->apple9_native_leaf = task->apple9_native_leaf = true;
   memset(mesh->apple9_native_pair_id, 0x92, sizeof(mesh->apple9_native_pair_id));
   memcpy(task->apple9_native_pair_id, mesh->apple9_native_pair_id,
          sizeof(mesh->apple9_native_pair_id));
   struct hk_shader *m = &mesh->variants[HK_MESH_VARIANT_MAIN];
   struct hk_shader *t = hk_only_variant(task);
   m->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   m->info.apple9_native.layout.max_primitives = 1;
   m->info.mesh.native_task_bridge = true;
   t->info.task.max_mesh_groups = 3;
   t->info.task.payload_B = 16384;
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_TASK] = task;
   gfx->render.layer_count = 1;
   cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
   uint32_t direct[] = {2,3,4};
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Task-bridged native mesh keeps indirect, indirect-count, and conditional
    * rendering on the emulated path so skipped tasks have no side effects. */
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   cmd->state.cond_render.active = true;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   cmd->state.cond_render.active = false;
   /* Multiview task-bridged draws require a matching native_multiview binary. */
   gfx->render.view_mask = 3;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   m->info.mesh.native_multiview = true;
   m->info.apple9_native.writes_layer_viewport = true;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   gfx->render.view_mask = 0;
   m->info.mesh.native_multiview = false;
   m->info.apple9_native.writes_layer_viewport = false;
   t->info.task.max_mesh_groups = 0;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Standalone native leaves lack paired emulated variants, so dynamic task
    * counts stay off; paired hybrid shaders admit dynamic task bridge. */
   t->info.task.max_mesh_groups = UINT32_MAX;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   {
      struct hk_api_shader *outer_mesh = shader(MESA_SHADER_MESH);
      struct hk_api_shader *outer_task = shader(MESA_SHADER_TASK);
      outer_mesh->apple9_native = mesh;
      outer_task->apple9_native = task;
      gfx->shaders[MESA_SHADER_MESH] = outer_mesh;
      gfx->shaders[MESA_SHADER_TASK] = outer_task;
      CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
      /* Legal aggregate task count must not be able to wrap 32 bits:
       * max 1023 task groups (1023 * 2^22 < 2^32 <= 1024 * 2^22). */
      uint32_t dyn_ok[] = {1023, 1, 1}, dyn_wrap[] = {1024, 1, 1};
      CHECK(hk_apple9_use_native_mesh(cmd, dyn_ok, 0));
      CHECK(!hk_apple9_use_native_mesh(cmd, dyn_wrap, 0));
      /* Still rejected when the native layout cannot fit even 1 WG in ring. */
      m->info.apple9_native.layout.max_vertices = 256;
      m->info.apple9_native.layout.vertex_components = 64;
      CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
      m->info.apple9_native.layout.max_vertices = 0;
      m->info.apple9_native.layout.vertex_components = 0;
      /* Indirect and conditional rendering still stay off task bridge. */
      CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
      cmd->state.cond_render.active = true;
      CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
      cmd->state.cond_render.active = false;
      gfx->shaders[MESA_SHADER_MESH] = mesh;
      gfx->shaders[MESA_SHADER_TASK] = task;
      free(outer_mesh); free(outer_task);
   }
   t->info.task.max_mesh_groups = 3;
   task->apple9_native_pair_id[5] ^= 1;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   task->apple9_native_pair_id[5] ^= 1;
   m->info.mesh.native_task_bridge = false;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   m->info.mesh.native_task_bridge = true;
   gfx->shaders[MESA_SHADER_TASK] = NULL;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->shaders[MESA_SHADER_TASK] = task;
   /* 2D dispatch factorization lifts the 65535 1D cap up to max_groups
    * (min(2^20 - 1, TVB primitive cap / max_primitives)). */
   direct[0] = 21845; direct[1] = direct[2] = 1;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0)); /* 65535 children */
   direct[0]++;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0)); /* 65538 children (2D) */
   t->info.task.max_mesh_groups = 10;
   direct[0] = 10000; direct[1] = 10; direct[2] = 1;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0)); /* 1000000 <= TVB cap */
   direct[0] = 10001; direct[1] = 10; direct[2] = 1;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0)); /* 1000100 > TVB cap */
   t->info.task.max_mesh_groups = 3;
   direct[0] = 2; direct[1] = direct[2] = 1;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Task bridge multiplies parent grid by max_mesh_groups before checking the
    * 64 KiB mesh output ring: V=128, n=8 is unbounded via safe-stride U0_B,
    * while V=256, n=28, P=32 has no safe U0_B (max_groups=2). */
   m->info.apple9_native.layout.max_vertices = 128;
   m->info.apple9_native.layout.vertex_components = 8;
   m->info.apple9_native.layout.max_primitives = 32;
   m->info.apple9_native.layout.indices_per_primitive = 3;
   direct[0] = 6; direct[1] = direct[2] = 1; /* 6 * 3 = 18 groups, safe U0_B */
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   m->info.apple9_native.layout.max_vertices = 256;
   m->info.apple9_native.layout.vertex_components = 28;
   t->info.task.max_mesh_groups = 2;
   direct[0] = 1; direct[1] = direct[2] = 1; /* 1 * 2 = 2 groups <= 2 */
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   direct[0] = 2; /* 2 * 2 = 4 groups > 2 */
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   gfx->shaders[MESA_SHADER_TASK] = gfx->shaders[MESA_SHADER_MESH] = NULL;
   free(mesh); free(task);
}

static void
tess_routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *tcs = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *nc = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *ng = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *nv = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *nf = shader(MESA_SHADER_FRAGMENT);
   gs->apple9_native = ng;
   vs->apple9_native = nv;
   fs->apple9_native = nf;
   tcs->apple9_native = nc;
   nc->apple9_native_leaf = true;
   ng->apple9_native_leaf = nv->apple9_native_leaf = nf->apple9_native_leaf =
      true;
   memset(ng->apple9_native_pair_id, 0x71, sizeof(ng->apple9_native_pair_id));
   memcpy(nv->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   memcpy(nc->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   struct hk_shader *main = hk_native_producer_main(ng);
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_native_producer_object(ng)->info.apple9_native_stage =
      HK_APPLE9_NATIVE_OBJECT;
   main->info.apple9_native.key.physical_threads = 16;
   main->info.apple9_native.layout.max_vertices = 3;
   main->info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_TESS_EVAL] = gs;
   gfx->shaders[MESA_SHADER_TESS_CTRL] = tcs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   dyn->ts.patch_control_points = 3;
   bool enabled = hk_apple9_native_tess_enabled();
   CHECK(hk_apple9_use_native_tess(cmd, draw) == enabled);
   if (enabled) {
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_tess(cmd, draw));                            \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_tess(cmd, draw));                             \
   } while (0)
      FALLBACK(cmd->in_meta, true);
      FALLBACK(gs->apple9_native, NULL);
      FALLBACK(vs->apple9_native, NULL);
      FALLBACK(fs->apple9_native, NULL);
      FALLBACK(ng->apple9_native_leaf, false);
      FALLBACK(nv->apple9_native_pair_id[0], 0x13);
      FALLBACK(nf->apple9_native_pair_id[31], 0x13);
      FALLBACK(tcs->apple9_native, NULL);
      FALLBACK(nc->apple9_native_pair_id[0], 0x13);
      FALLBACK(nc->writes_memory, true);
      FALLBACK(dyn->ts.patch_control_points, 0);
      FALLBACK(dyn->ts.patch_control_points, 33);
      FALLBACK(nv->writes_memory, true);
      FALLBACK(main->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(hk_native_producer_object(ng)->info.apple9_native_stage,
               HK_APPLE9_NATIVE_NONE);
      FALLBACK(main->info.apple9_native.key.physical_threads, 32);
      FALLBACK(main->info.apple9_native.layout.max_vertices, 5);
      FALLBACK(draw.indexed, true);
      FALLBACK(draw.restart, true);
      FALLBACK(draw.start, 1);
      FALLBACK(draw.start_instance, 1);
      FALLBACK(draw.b.count[0], 6);
      FALLBACK(draw.b.count[1], 2);
      FALLBACK(draw.b.count[2], 2);
      CHECK(!hk_apple9_use_native_tess(cmd, agx_draw_indirect(0x1000)));
      FALLBACK(dyn->ia.primitive_restart_enable, true);
      FALLBACK(dyn->ia.primitive_topology,
               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_CTRL], NULL);
      FALLBACK(gfx->shaders[MESA_SHADER_GEOMETRY], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TASK], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_MESH], gs);
      FALLBACK(gfx->render.view_mask, 3);
      FALLBACK(gfx->render.layer_count, 0);
      FALLBACK(gfx->render.layer_count, 2049);
      gfx->render.layer_count = 4;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      gfx->render.layer_count = 1;
      FALLBACK(cmd->state.cond_render.active, true);
      FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags, 1);
      FALLBACK(gfx->occlusion.active_count, 1);
      FALLBACK(gfx->mesh_pg_query, 0x1000);
      FALLBACK(gfx->pg_query[0], 0x1000);
      FALLBACK(gfx->xfb_query[3], 0x1000);
      FALLBACK(gfx->xfb_enabled, true);
      FALLBACK(dyn->rs.rasterization_stream, 1);
      FALLBACK(dyn->rs.rasterizer_discard_enable, true);
      FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
      FALLBACK(dyn->rs.conservative_mode,
               VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
      dyn->ms.rasterization_samples = 4;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      FALLBACK(dyn->ms.rasterization_samples, 8);
      dyn->ms.rasterization_samples = 0;
      gfx->render.tilebuffer.nr_samples = 4;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      FALLBACK(gfx->render.tilebuffer.nr_samples, 8);
      gfx->render.tilebuffer.nr_samples = 0;
#undef FALLBACK
      for (unsigned p = 1; p <= 32; ++p) {
         draw.b.count[0] = dyn->ts.patch_control_points = p;
         CHECK(hk_apple9_use_native_tess(cmd, draw));
      }
      /* Hardware 64 KiB mesh output ring gate also protects native TES:
       * V=3 with <= 128 components is now unbounded via safe-stride U0_B,
       * while 260 components exceeds safe U0_B and cannot fit 8192 max groups. */
      main->info.apple9_native.layout.vertex_components = 36;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 128;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 260;
      CHECK(!hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 0;
      gfx->shaders[MESA_SHADER_TESS_EVAL] = ng;
      gfx->shaders[MESA_SHADER_VERTEX] = nv;
      gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
   }
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(tcs);
   free(nc);
   free(gs);
   free(vs);
   free(fs);
   free(ng);
   free(nv);
   free(nf);
}

static void
tess_fused_routes(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint64_t saved_features = dev->dev.params.features;
   dev->dev.params.features |= DRM_ASAHI_FEATURE_NATIVE_MESH;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *tes = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *tcs = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *ntes = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *ntcs = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *nvs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *nfs = shader(MESA_SHADER_FRAGMENT);
   tes->apple9_native = ntes;
   tcs->apple9_native = ntcs;
   vs->apple9_native = nvs;
   fs->apple9_native = nfs;
   ntes->apple9_native_leaf = ntcs->apple9_native_leaf =
      nvs->apple9_native_leaf = nfs->apple9_native_leaf = true;
   memset(ntes->apple9_native_pair_id, 0x83, sizeof(ntes->apple9_native_pair_id));
   memcpy(ntcs->apple9_native_pair_id, ntes->apple9_native_pair_id,
          sizeof(ntes->apple9_native_pair_id));
   memcpy(nvs->apple9_native_pair_id, ntes->apple9_native_pair_id,
          sizeof(ntes->apple9_native_pair_id));
   memcpy(nfs->apple9_native_pair_id, ntes->apple9_native_pair_id,
          sizeof(ntes->apple9_native_pair_id));

   struct hk_shader *main = hk_native_producer_main(ntes);
   struct hk_shader *obj = hk_native_producer_object(ntes);
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   obj->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   main->info.apple9_tess_fused = true;
   main->info.apple9_tess_const_levels = true;
   main->info.apple9_tess_payload_B = 192;
   main->info.apple9_tess_max_child_wgs = 1;
   main->info.apple9_tess_max_patch_prims = 32;
   main->info.apple9_native.key.physical_threads = 32;
   main->info.apple9_native.layout.max_vertices = 32;
   main->info.apple9_native.layout.max_primitives = 42;
   main->info.apple9_native.layout.vertex_components = 4;
   obj->b.info.workgroup_size[0] = 32;
   obj->info.apple9_native.shared_size_B = 256;

   gfx->shaders[MESA_SHADER_TESS_EVAL] = tes;
   gfx->shaders[MESA_SHADER_TESS_CTRL] = tcs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   dyn->ts.patch_control_points = 4;
   struct agx_draw draw = {.b = agx_3d(4 * 1024, 1, 1)};

   bool enabled = hk_apple9_native_tess_enabled();
   CHECK(hk_apple9_use_native_tess(cmd, draw) == enabled);
   if (enabled) {
      /* Multi-patch, multi-instance, firstVertex, firstInstance, dynamic domain
       * origin, and U8/U16/U32 indexed draws are all admitted directly when
       * grid chunking stays within HK_APPLE9_NATIVE_TESS_MAX_CHUNKS. */
      draw.start = 12;
      draw.start_instance = 7;
      const uint64_t prim_cap = hk_apple9_native_render_primitives();
      main->info.apple9_tess_max_patch_prims = 1;
      draw.b.count[0] = 4 * 524288; /* > 65535 -> 2D grid, 2 chunks <= max */
      draw.b.count[1] = 2;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      dyn->ts.domain_origin = VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      dyn->ts.domain_origin = VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT;
      /* Unbounded (wg_rows <= 40, != 32) draws ignore the 1M primitive cap:
       * 524288 * 32 * 2 still plans in 2 grid chunks -> admitted! */
      main->info.apple9_tess_max_patch_prims = 32;
      CHECK(hk_apple9_use_native_tess(cmd, draw));

      /* Unbounded giant 20,000-patch dynamic tessellation draw (worst-case
       * 8192 prims/patch = 163.84M > 1M cap) plans 1 SINGLE chunk! */
      main->info.apple9_tess_max_patch_prims = 8192;
      draw.b.count[1] = 1;
      draw.b.count[0] = 4 * 20000;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      struct hk_apple9_fused_tess_chunk_plan plan = {0};
      CHECK(hk_apple9_native_tess_fused_chunk_plan(
         20000, 1, AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX,
         prim_cap, &plan));
      CHECK(plan.inst_chunk_cap == 1 && plan.patch_chunk_cap == 20000);

      /* Grid-overflow chunking boundary still enforces max_groups=1,048,575
       * and HK_APPLE9_NATIVE_TESS_MAX_CHUNKS=4u (patch & instance chunking). */
      const uint32_t safe_patch_cap = AGX_NATIVE_OBJECT_MAX_GROUPS - 16;
      CHECK(hk_apple9_native_tess_fused_chunk_plan(
         safe_patch_cap * HK_APPLE9_NATIVE_TESS_MAX_CHUNKS, 1,
         AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX, prim_cap,
         &plan));
      CHECK(plan.inst_chunk_cap == 1 && plan.patch_chunk_cap == safe_patch_cap);
      CHECK(!hk_apple9_native_tess_fused_chunk_plan(
         safe_patch_cap * HK_APPLE9_NATIVE_TESS_MAX_CHUNKS + 1, 1,
         AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX, prim_cap,
         &plan));
      /* Instance chunking on grid cap: 600,000 patches -> 1 inst/chunk -> 4 OK, 5 reject. */
      CHECK(hk_apple9_native_tess_fused_chunk_plan(
         600000, HK_APPLE9_NATIVE_TESS_MAX_CHUNKS,
         AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX, prim_cap,
         &plan));
      CHECK(plan.inst_chunk_cap == 1 && plan.patch_chunk_cap == 600000);
      CHECK(!hk_apple9_native_tess_fused_chunk_plan(
         600000, HK_APPLE9_NATIVE_TESS_MAX_CHUNKS + 1,
         AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX, prim_cap,
         &plan));
      /* Ring-bounded (ring_max != UINT64_MAX) still strictly enforces 1-chunk
       * ring AND render_prim_cap limits. */
      CHECK(!hk_apple9_native_tess_fused_chunk_plan(
         200, 1, AGX_NATIVE_OBJECT_MAX_GROUPS, 1, 8192, 4096, prim_cap, &plan));
      CHECK(hk_apple9_native_tess_fused_chunk_plan(
         100, 1, AGX_NATIVE_OBJECT_MAX_GROUPS, 1, 8192, 4096, prim_cap, &plan));
      CHECK(plan.inst_chunk_cap == 1 && plan.patch_chunk_cap == 100);

      /* Multi-patch K=16 chunk planning: 2,000,000 patches pack into
       * 125,000 OBJECT WGs <= AGX_NATIVE_OBJECT_MAX_GROUPS -> 1 SINGLE chunk! */
      CHECK(hk_apple9_native_tess_fused_chunk_plan_ex(
         2000000, 1, 16, AGX_NATIVE_OBJECT_MAX_GROUPS, 4096, 8192, UINT64_MAX,
         prim_cap, &plan));
      CHECK(plan.inst_chunk_cap == 1 && plan.patch_chunk_cap == 2000000);
      /* Partial tail patch counts (1, 3, 17, 31) with K=4 and K=16 plan in 1 chunk. */
      const uint32_t tail_pc[] = {1, 3, 17, 31};
      for (unsigned ti = 0; ti < ARRAY_SIZE(tail_pc); ++ti) {
         CHECK(hk_apple9_native_tess_fused_chunk_plan_ex(
            tail_pc[ti], 1, 4, AGX_NATIVE_OBJECT_MAX_GROUPS, 64, 64, UINT64_MAX,
            prim_cap, &plan));
         CHECK(plan.patch_chunk_cap == tail_pc[ti]);
         CHECK(hk_apple9_native_tess_fused_chunk_plan_ex(
            tail_pc[ti], 1, 16, AGX_NATIVE_OBJECT_MAX_GROUPS, 64, 64, UINT64_MAX,
            prim_cap, &plan));
         CHECK(plan.patch_chunk_cap == tail_pc[ti]);
      }

      /* Dynamic patchControlPoints guard: reject draws whose pcp > apple9_tess_max_input_pcp. */
      main->info.apple9_tess_patches_per_group = 16;
      main->info.apple9_tess_max_input_pcp = 4;
      dyn->ts.patch_control_points = 4;
      draw.b.count[0] = 4 * 31;
      draw.b.count[1] = 1;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      dyn->ts.patch_control_points = 5;
      draw.b.count[0] = 5 * 31;
      CHECK(!hk_apple9_use_native_tess(cmd, draw));
      dyn->ts.patch_control_points = 4;
      main->info.apple9_tess_max_input_pcp = 0;
      main->info.apple9_tess_patches_per_group = 1;

      /* Single-chunk unbounded draw stays admitted. */
      main->info.apple9_tess_max_patch_prims = 32;
      draw.b.count[0] = 4 * (uint32_t)(prim_cap / 32);
      draw.b.count[1] = 1;
      CHECK(hk_apple9_use_native_tess(cmd, draw));

      draw.b.count[0] = 4 * 64;
      draw.b.count[1] = 3;
      draw.indexed = true;
      draw.index_size = AGX_INDEX_SIZE_U8;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      draw.index_size = AGX_INDEX_SIZE_U16;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      draw.index_size = AGX_INDEX_SIZE_U32;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      draw.indexed = false;

      /* Zero-PREGFX prepare_draw omits HK_READS_INTERNAL and creates no PREGFX CS. */
      uint8_t *pool = calloc(2, HK_CMD_BO_SIZE);
      CHECK(pool);
      cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
      cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x1400000000ull;
      cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE;
      cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base = 0x1500000000ull;
      cmd->sync_events = UTIL_DYNARRAY_INIT;
      cmd->sync_remap = UTIL_DYNARRAY_INIT;
      cmd->sync_first_event = UTIL_DYNARRAY_INIT;
      list_inithead(&cmd->control_streams);
      CHECK(hk_apple9_prepare_draw_ex(cmd, false, true, true));
      CHECK(cmd->current_cs.gfx != NULL);
      CHECK(!(cmd->current_cs.gfx->pre_raster_reads & HK_READS_INTERNAL));
      CHECK(cmd->current_cs.pre_gfx == NULL);
      list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
         list_del(&cs->node);
         hk_cs_destroy(cs);
      }
      cmd->current_cs.gfx = NULL;
      cmd->current_cs.pre_gfx = NULL;
      util_dynarray_fini(&cmd->sync_events);
      util_dynarray_fini(&cmd->sync_remap);
      util_dynarray_fini(&cmd->sync_first_event);
      free(pool);

      /* Payload, feature bit-28, indirect, and ring-limit gates. */
      CHECK(!hk_apple9_use_native_tess(cmd, agx_draw_indirect(0x1000)));
      main->info.apple9_tess_payload_B = AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_tess_payload_B = AGX_NATIVE_OBJECT_MP_PAYLOAD_BYTES + 1;
      CHECK(!hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_tess_payload_B = 192;
      dev->dev.params.features &= ~DRM_ASAHI_FEATURE_NATIVE_MESH;
      CHECK(!hk_apple9_use_native_tess(cmd, draw));
      dev->dev.params.features |= DRM_ASAHI_FEATURE_NATIVE_MESH;

      /* V=4, n=40 and V=4, n=24 (wg_rows == 32) are now unbounded via
       * safe-stride U0_B; V=256, n=28, P=32 has no safe U0_B (max_groups=2)
       * and stays single-chunk only. */
      main->info.apple9_native.layout.max_vertices = 4;
      main->info.apple9_native.layout.max_primitives = 2;
      main->info.apple9_native.layout.indices_per_primitive = 3;
      main->info.apple9_native.layout.vertex_components = 40;
      draw.b.count[1] = 1;
      draw.b.count[0] = 4 * 86;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 24;
      draw.b.count[0] = 4 * 129;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.max_vertices = 256;
      main->info.apple9_native.layout.max_primitives = 32;
      main->info.apple9_native.layout.vertex_components = 28;
      draw.b.count[0] = 4 * 2;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      draw.b.count[0] = 4 * 3;
      CHECK(!hk_apple9_use_native_tess(cmd, draw));
   }

   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   dev->dev.params.features = saved_features;
   free(tes);
   free(tcs);
   free(vs);
   free(fs);
   free(ntes);
   free(ntcs);
   free(nvs);
   free(nfs);
}

static void
mark_draw(struct hk_cs *cs, bool native)
{
   /* The classification/transition test owns CPU storage, not GPU execution.
    * Reserve representative stream bytes before the real next preparation. */
   cs->current += 160;
   cs->stats.cmds++;
   if (native) {
      cs->apple9_native_mesh = true;
      cs->apple9_native_bounded_prims = true;
      cs->apple9_native_dispatches++;
   }
}

static void
transitions(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   uint8_t *pool = calloc(2, HK_CMD_BO_SIZE);
   CHECK(pool);
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x1400000000ull;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base = 0x1500000000ull;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);
   struct hk_rendering_state *r = &cmd->state.gfx.render;
   r->cr.bg.partial.usc = 0x1000;
   r->cr.eot.partial.usc = 0x2000;
   memset(&r->cr.zls_control_partial, 0x35, sizeof(r->cr.zls_control_partial));

   /* With DRM_ASAHI_FEATURE_NATIVE_MESH enabled (and native blocks not
    * disabled), ordinary -> native -> native -> ordinary -> native -> meta
    * all merge into ONE single render pass with 0 splits! */
   if (hk_apple9_native_mesh_available_device(dev)) {
      cmd->apple9_draw_primitives = 100;
      cmd->apple9_draw_ring_rows = 40;
      cmd->apple9_draw_ring_limited = false;
      CHECK(hk_apple9_prepare_draw(cmd, false, false));
      struct hk_cs *merged = cmd->current_cs.gfx;
      CHECK(merged && !merged->apple9_native_mesh);
      mark_draw(merged, false); /* ordinary draw leads the render pass */

      const bool seq[] = {true, true, false, true, false};
      for (unsigned i = 0; i < ARRAY_SIZE(seq); ++i) {
         cmd->in_meta = i == ARRAY_SIZE(seq) - 1;
         CHECK(hk_apple9_prepare_draw(cmd, seq[i], seq[i]));
         CHECK(cmd->current_cs.gfx == merged);
         mark_draw(merged, seq[i]);
      }
      cmd->in_meta = false;
      CHECK(merged->apple9_native_mesh && merged->apple9_native_dispatches == 3);

      /* Genuine hardware invariant: cumulative 64 KiB (4096-row) mesh ring
       * overflow on a ring-limited (wg_rows > 40) shader MUST split! */
      merged->apple9_native_ring_rows = 3000;
      merged->apple9_native_ring_limited = true;
      cmd->apple9_draw_ring_rows = 1200;
      cmd->apple9_draw_ring_limited = true;
      CHECK(hk_apple9_prepare_draw(cmd, true, true));
      CHECK(cmd->current_cs.gfx != merged);
      CHECK(merged->end_reason == HK_CS_END_RENDER_SPLIT);
      cmd->apple9_draw_ring_rows = 0;
      cmd->apple9_draw_ring_limited = false;
      hk_cmd_buffer_end_graphics(cmd);

      /* Emulated GS/TES output counts are GPU-determined (UINT64_MAX), so they
       * split both after and before a capped native draw instead of using their
       * API input primitive count. */
      struct hk_api_shader *emu_gs = shader(MESA_SHADER_GEOMETRY);
      struct hk_api_shader *emu_tes = shader(MESA_SHADER_TESS_EVAL);
      struct agx_draw big_input = {.b = agx_3d(3 * 20000, 1, 1)};
      cmd->vk.dynamic_graphics_state.ia.primitive_topology =
         VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
      CHECK(hk_apple9_ordinary_draw_primitives(cmd, big_input) == 20000);
      cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_POINT;
      CHECK(hk_apple9_ordinary_draw_primitives(cmd, big_input) == 60000);
      cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = emu_gs;
      CHECK(hk_apple9_ordinary_draw_primitives(cmd, big_input) == UINT64_MAX);
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = NULL;
      cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] = emu_tes;
      CHECK(hk_apple9_ordinary_draw_primitives(cmd, big_input) == UINT64_MAX);
      cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] = NULL;

      /* Emulated GS after a 900k capped native draw splits cleanly. */
      cmd->apple9_draw_primitives = 900000;
      CHECK(hk_apple9_prepare_draw(cmd, true, true));
      struct hk_cs *cs_cap = cmd->current_cs.gfx;
      mark_draw(cs_cap, true);
      cs_cap->apple9_native_primitives = 900000;
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = emu_gs;
      cmd->apple9_draw_primitives =
         hk_apple9_ordinary_draw_primitives(cmd, big_input);
      CHECK(cmd->apple9_draw_primitives == UINT64_MAX);
      CHECK(hk_apple9_prepare_draw(cmd, false, false));
      struct hk_cs *cs_emu = cmd->current_cs.gfx;
      CHECK(cs_emu != cs_cap && !cs_emu->apple9_native_mesh);
      /* Emulated GS before a 100-prim capped native draw also splits cleanly. */
      mark_draw(cs_emu, false);
      cs_emu->apple9_native_primitives = cmd->apple9_draw_primitives;
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = NULL;
      cmd->apple9_draw_primitives = 100;
      CHECK(hk_apple9_prepare_draw(cmd, true, true));
      CHECK(cmd->current_cs.gfx != cs_emu);
      hk_cmd_buffer_end_graphics(cmd);
      free(emu_gs);
      free(emu_tes);

      /* Multiview ordinary draw crossing 1M cap on view 2 (900k + 2 * 60k)
       * reserves the complete multiview total at preparation and splits cleanly
       * before view 0 so per-view emission never hits UNSUPPORTED(). */
      cmd->apple9_draw_primitives = 900000;
      CHECK(hk_apple9_prepare_draw(cmd, true, true));
      struct hk_cs *cs_mv_pre = cmd->current_cs.gfx;
      mark_draw(cs_mv_pre, true);
      cs_mv_pre->apple9_native_primitives = 900000;
      cmd->state.gfx.render.view_mask = 0x3;
      struct agx_draw mv_draw = {.b = agx_3d(3 * 60000, 1, 1)};
      uint64_t per_view = hk_apple9_ordinary_draw_primitives(cmd, mv_draw);
      unsigned views = util_bitcount(cmd->state.gfx.render.view_mask);
      CHECK(per_view == 60000 && views == 2);
      cmd->apple9_draw_primitives = per_view * views;
      CHECK(hk_apple9_prepare_draw(cmd, false, false));
      struct hk_cs *cs_mv = cmd->current_cs.gfx;
      CHECK(cs_mv != cs_mv_pre && !cs_mv->apple9_native_mesh);
      cmd->apple9_draw_primitives = per_view;
      for (unsigned v = 0; v < views; ++v) {
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         CHECK(cmd->current_cs.gfx == cs_mv);
         mark_draw(cs_mv, false);
         cs_mv->apple9_native_primitives += per_view;
      }
      CHECK(cs_mv->apple9_native_primitives == 120000);
      CHECK(cmd->vk.record_result == VK_SUCCESS);
      cmd->state.gfx.render.view_mask = 0;
      hk_cmd_buffer_end_graphics(cmd);

      /* M1: Large ordinary draws (>= 0x80000000 vertices) must never narrow to
       * negative int and undercount strips, fans, or adjacency topologies. */
      {
         struct agx_draw huge_v = {.b = agx_3d(0x80000000u, 1, 1)};
         cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
         CHECK(hk_apple9_ordinary_draw_primitives(cmd, huge_v) == 0x7ffffffeu);
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
         CHECK(hk_apple9_ordinary_draw_primitives(cmd, huge_v) == 0x7ffffffeu);
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
         CHECK(hk_apple9_ordinary_draw_primitives(cmd, huge_v) == 0x7fffffffu);
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
         CHECK(hk_apple9_ordinary_draw_primitives(cmd, huge_v) == 0x7ffffffdu);
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
         CHECK(hk_apple9_ordinary_draw_primitives(cmd, huge_v) ==
               1u + (0x80000000u - 6u) / 2u);
         /* Huge ordinary prefix followed by capped native draw MUST split. */
         cmd->vk.dynamic_graphics_state.ia.primitive_topology =
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
         cmd->apple9_draw_primitives =
            hk_apple9_ordinary_draw_primitives(cmd, huge_v);
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_big_ord = cmd->current_cs.gfx;
         mark_draw(cs_big_ord, false);
         cs_big_ord->apple9_native_primitives = cmd->apple9_draw_primitives;
         cmd->apple9_draw_primitives = 10;
         CHECK(hk_apple9_prepare_draw(cmd, true, true));
         CHECK(cmd->current_cs.gfx != cs_big_ord);
         /* Capped native followed by huge ordinary strip MUST split too. */
         struct hk_cs *cs_nat2 = cmd->current_cs.gfx;
         mark_draw(cs_nat2, true);
         cs_nat2->apple9_native_primitives = 10;
         cmd->apple9_draw_primitives =
            hk_apple9_ordinary_draw_primitives(cmd, huge_v);
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         CHECK(cmd->current_cs.gfx != cs_nat2);
         hk_cmd_buffer_end_graphics(cmd);
      }

      /* M4: Late splits (depth-bias representation change, scissor overflow,
       * and memory-writing draw after a hoisted barrier) must commit sample
       * locations onto the newly created stream so a subsequent draw with
       * different sample locations splits cleanly. */
      {
         struct vk_sample_locations_state sl_a = {
            .per_pixel = 1,
            .grid_size = {1, 1},
            .locations = {{0.25f, 0.25f}},
         };
         struct vk_sample_locations_state sl_b = {
            .per_pixel = 1,
            .grid_size = {1, 1},
            .locations = {{0.75f, 0.75f}},
         };
         cmd->vk.dynamic_graphics_state.ms.rasterization_samples = 1;
         cmd->vk.dynamic_graphics_state.ms.sample_locations_enable = true;
         cmd->vk.dynamic_graphics_state.ms.sample_locations = &sl_a;
         cmd->vk.dynamic_graphics_state.rs.depth_bias.enable = true;
         cmd->vk.dynamic_graphics_state.rs.depth_bias.representation =
            VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT;
         cmd->apple9_draw_primitives = 10;

         /* (a) Depth-bias representation change split. */
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_a = cmd->current_cs.gfx;
         CHECK(cs_a->has_sample_locations);
         uint32_t pos_a = cs_a->ppp_multisamplectl;
         mark_draw(cs_a, false);
         cmd->vk.dynamic_graphics_state.rs.depth_bias.representation =
            VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_b = cmd->current_cs.gfx;
         CHECK(cs_b != cs_a);
         CHECK(cs_b->has_sample_locations && cs_b->ppp_multisamplectl == pos_a);
         mark_draw(cs_b, false);
         /* Changing sample positions on the next draw MUST split cs_b. */
         cmd->vk.dynamic_graphics_state.ms.sample_locations = &sl_b;
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_c = cmd->current_cs.gfx;
         CHECK(cs_c != cs_b && cs_c->has_sample_locations &&
               cs_c->ppp_multisamplectl != pos_a);
         mark_draw(cs_c, false);

         /* (b) Memory-writing draw after a hoisted barrier split. */
         struct hk_api_shader *writer_mesh = shader(MESA_SHADER_MESH);
         writer_mesh->writes_memory = true;
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = writer_mesh;
         cs_c->merge_hoisted = true;
         cmd->vk.dynamic_graphics_state.ms.sample_locations = &sl_a;
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_d = cmd->current_cs.gfx;
         CHECK(cs_d != cs_c && cs_d->has_sample_locations &&
               cs_d->ppp_multisamplectl == pos_a);

         /* M5: Shader writes from MESA_SHADER_MESH / MESA_SHADER_TASK invalidate
          * cmd->feedback_raw on both native and emulated graphics preparation. */
         util_dynarray_append(&cmd->feedback_raw, UINT64_C(0xdeadbeef));
         CHECK(cmd->feedback_raw.size == sizeof(uint64_t));
         CHECK(hk_apple9_prepare_draw_ex(cmd, true, true, false));
         CHECK(cmd->feedback_raw.size == 0);
         util_dynarray_append(&cmd->feedback_raw, UINT64_C(0xdeadbeef));
         CHECK(hk_apple9_prepare_draw_ex(cmd, false, false, false));
         CHECK(cmd->feedback_raw.size == 0);

         cmd->state.gfx.shaders[MESA_SHADER_MESH] = NULL;
         free(writer_mesh);
         cmd->vk.dynamic_graphics_state.ms.sample_locations_enable = false;
         cmd->vk.dynamic_graphics_state.ms.sample_locations = NULL;
         cmd->vk.dynamic_graphics_state.rs.depth_bias.enable = false;
         hk_cmd_buffer_end_graphics(cmd);
      }

      /* Ordinary draw prefix followed by a native-eligible indirect mesh draw:
       * CPU admission must bound draw_primitives and ring usage by the
       * GPU-enforced max_groups ceiling (never UINT64_MAX -> REJECT abort). */
      {
         struct hk_api_shader *ind_mesh = shader(MESA_SHADER_MESH);
         struct hk_api_shader *ind_nm = shader(MESA_SHADER_MESH);
         struct hk_api_shader *ind_fs = shader(MESA_SHADER_FRAGMENT);
         struct hk_api_shader *ind_nf = shader(MESA_SHADER_FRAGMENT);
         ind_mesh->apple9_native = ind_nm;
         ind_fs->apple9_native = ind_nf;
         ind_nm->apple9_native_leaf = ind_nf->apple9_native_leaf = true;
         memset(ind_nm->apple9_native_pair_id, 0x33,
                sizeof(ind_nm->apple9_native_pair_id));
         memcpy(ind_nf->apple9_native_pair_id, ind_nm->apple9_native_pair_id,
                sizeof(ind_nf->apple9_native_pair_id));
         ind_mesh->variants[HK_MESH_VARIANT_MAIN].info.mesh.prim =
            MESA_PRIM_TRIANGLES;
         ind_mesh->variants[HK_MESH_VARIANT_MAIN].info.mesh.max_primitives = 1;
         ind_nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage =
            HK_APPLE9_NATIVE_MESH;
         ind_nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.guard_empty_grid =
            true;
         ind_nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout =
            (struct agx_apple9_mesh_layout){
               .max_vertices = 3,
               .max_primitives = 1,
               .indices_per_primitive = 3,
               .vertex_components = 4,
            };
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = ind_mesh;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = ind_fs;
         cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
         cmd->vk.dynamic_graphics_state.rs.conservative_mode =
            VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
         cmd->vk.dynamic_graphics_state.rs.rasterizer_discard_enable = false;

         /* Record an ordinary draw prefix on the active stream, then dispatch
          * a native-eligible indirect mesh draw (uses_pre_gfx=true splits the
          * active stream). CPU admission must bound draw_primitives and ring
          * rows by max_groups so stream classification splits cleanly instead
          * of aborting on REJECT for UINT64_MAX primitives. */
         cmd->apple9_draw_primitives = 100;
         cmd->apple9_draw_ring_rows = 0;
         cmd->apple9_draw_ring_limited = false;
         CHECK(hk_apple9_prepare_draw(cmd, false, false));
         struct hk_cs *cs_ord = cmd->current_cs.gfx;
         CHECK(cs_ord);
         mark_draw(cs_ord, false);
         cs_ord->apple9_native_primitives = 100;

         struct agx_va ind_va = {.addr = 0x100000};
         struct hk_buffer ind_buf = {
            .vk = {.base = {.type = VK_OBJECT_TYPE_BUFFER, .device = &dev->vk},
                   .size = 64,
                   .device_address = 0x100000},
            .va = &ind_va,
         };
         cmd->rendering = true;
         hk_CmdDrawMeshTasksIndirectEXT(hk_cmd_buffer_to_handle(cmd),
                                        hk_buffer_to_handle(&ind_buf), 0, 1, 12);
         CHECK(cmd->apple9_draw_primitives < UINT64_MAX);
         CHECK(cmd->apple9_draw_ring_rows < UINT64_MAX);
         CHECK(cmd->current_cs.gfx != cs_ord);
         CHECK(cs_ord->end_reason == HK_CS_END_RENDER_SPLIT);

         cmd->vk.record_result = VK_SUCCESS;
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = NULL;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = NULL;
         hk_cmd_buffer_end_graphics(cmd);
         free(ind_nm);
         free(ind_nf);
         free(ind_mesh);
         free(ind_fs);
      }
   }

   /* Under the isolation kill switch (disable_native_blocks = true), every
    * transition across native draws splits cleanly with partial-render state. */
   pdev->disable_native_blocks = true;
   cmd->apple9_draw_primitives = 100;
   CHECK(hk_apple9_prepare_draw(cmd, false, false));
   struct hk_cs *first = cmd->current_cs.gfx;
   CHECK(first && !first->stats.cmds);
   /* The first native draw can use an empty stream. */
   CHECK(hk_apple9_prepare_draw(cmd, true, true));
   CHECK(cmd->current_cs.gfx == first);
   mark_draw(first, true);
   const bool paths[] = {false, true, true, false, true, false};
   for (unsigned i = 0; i < ARRAY_SIZE(paths); ++i) {
      struct hk_cs *prev = cmd->current_cs.gfx;
      unsigned old_events =
         util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
      /* The final transition is an ordinary meta draw after native. */
      cmd->in_meta = i == ARRAY_SIZE(paths) - 1;
      CHECK(hk_apple9_prepare_draw(cmd, paths[i], paths[i]));
      struct hk_cs *next = cmd->current_cs.gfx;
      CHECK(next != prev && prev->end_reason == HK_CS_END_RENDER_SPLIT);
      CHECK(next->current == next->start && !next->apple9_native_mesh);
      CHECK(next->cr.bg.main.usc == r->cr.bg.partial.usc);
      CHECK(prev->cr.eot.main.usc == r->cr.eot.partial.usc);
      CHECK(!memcmp(&next->cr.zls_control, &r->cr.zls_control_partial,
                    sizeof(next->cr.zls_control)));
      const struct hk_sync_event *events = cmd->sync_events.data;
      CHECK(util_dynarray_num_elements(&cmd->sync_events,
                                       struct hk_sync_event) >= old_events + 2);
      CHECK(events[old_events].stream == HK_SYNC_NONE);
      CHECK(events[old_events].src == prev->sync_stages);
      CHECK(events[old_events].dst & HK_SYNC_FRAGMENT);
      CHECK(events[old_events + 1].stream == next->sync_id);
      mark_draw(next, paths[i]);
   }
   cmd->in_meta = false;
   pdev->disable_native_blocks = false;
   struct hk_cs *last = cmd->current_cs.gfx;
   CHECK(hk_apple9_prepare_draw(cmd, false, false));
   CHECK(cmd->current_cs.gfx == last); /* ordinary draws still coalesce */
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(pool);
}

static struct agx_bo *
mock_bo_alloc(struct agx_device *dev, size_t size, size_t align,
              enum agx_bo_flags flags, enum agx_va_flags va_flags,
              uint64_t fixed_va)
{
   (void)dev;
   (void)size;
   (void)align;
   (void)flags;
   (void)va_flags;
   (void)fixed_va;
   return NULL;
}

int
main(void)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(pdev && dev && cmd);
   pdev->dev.chip = dev->dev.chip = AGX_CHIP_G16G;
   pdev->dev.params.features = dev->dev.params.features =
      DRM_ASAHI_FEATURE_NATIVE_MESH;
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev->vk;
   dev->dev.fd = -1;
   dev->dev.bo_cache.forked = true;
   dev->dev.ops.bo_alloc = mock_bo_alloc;
   dev->bg_eot.dev = &dev->dev;
   simple_mtx_init(&dev->bg_eot.lock, mtx_plain);
   simple_mtx_init(&dev->dev.vma_lock, mtx_plain);
   glsl_type_singleton_init_or_ref();
   cmd->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   routes(cmd);
   task_routes(cmd);
   gs_routes(cmd);
   tess_routes(cmd);
   tess_fused_routes(cmd);
   transitions(cmd);
   simple_mtx_destroy(&dev->dev.vma_lock);
   simple_mtx_destroy(&dev->bg_eot.lock);
   glsl_type_singleton_decref();
   free(cmd);
   free(dev);
   free(pdev);
   puts(
      "hybrid draw routing, partial attachment state and continuation ordering pass");
   return 0;
}
