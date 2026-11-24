/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "../hk_cmd_buffer.h"
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
   /* Latched VM profile=none (including probe/read/parse failure on a real VM)
    * refuses native submission even with HK_APPLE9_NATIVE_MESH=1, while latched
    * research, qualified, and confirmed-absent (UNKNOWN) profiles admit it. */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t saved_vm_id = dev->dev.vm_id;
   dev->dev.vm_id = 9;
   dev->dev.native_mesh_profile = AGX_NATIVE_MESH_PROFILE_NONE;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.native_mesh_profile = AGX_NATIVE_MESH_PROFILE_RESEARCH;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.native_mesh_profile = AGX_NATIVE_MESH_PROFILE_QUALIFIED;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.native_mesh_profile = AGX_NATIVE_MESH_PROFILE_UNKNOWN;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   dev->dev.vm_id = saved_vm_id;
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
   /* Hardware 64 KiB mesh output ring gate: V=128, n=8 (264 rows/WG) admits up
    * to 15 direct workgroups (and indirect draws via GPU max_groups=15 +
    * emulated overflow fallback), while a single-WG ring overflow (V=256,
    * n=128) has max_groups=0 and falls back even for indirect draws. */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 128;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 8;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_primitives = 32;
   const uint32_t ring_fit[] = {15, 1, 1}, ring_wrap[] = {16, 1, 1},
                  ring_32[] = {32, 1, 1}, ring_64[] = {64, 1, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, ring_fit, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_wrap, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_32, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, ring_64, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 256;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 128;
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0x1000));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.max_vertices = 128;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.layout.vertex_components = 8;
   /* Multi-axis product crossing the same threshold falls back too. */
   const uint32_t ring_2d_fit[] = {3, 5, 1}, ring_2d_wrap[] = {4, 4, 1};
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
      /* Hardware 64 KiB mesh output ring gate also protects native GS. */
      main->info.apple9_native.layout.vertex_components = 128; /* 136 rows/WG -> 30 groups */
      draw.b.count[0] = 3 * 30;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      draw.b.count[0] = 3 * 31;
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
   t->info.task.max_mesh_groups = UINT32_MAX;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
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
   direct[0] = 21845; direct[1] = direct[2] = 1;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0)); /* 65535 children */
   direct[0]++;
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));
   direct[0] = 2;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   /* Task bridge multiplies parent grid by max_mesh_groups before checking the
    * 64 KiB mesh output ring (V=128, n=8 -> 15 safe groups). */
   m->info.apple9_native.layout.max_vertices = 128;
   m->info.apple9_native.layout.vertex_components = 8;
   direct[0] = 5; direct[1] = direct[2] = 1; /* 5 * 3 = 15 groups */
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   direct[0] = 6; /* 6 * 3 = 18 groups > 15 */
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
       * V=3 with <= 56 components stays <= 64 rows/WG; 60 components exceeds
       * the ring for a densely tessellated patch (8192 max groups). */
      main->info.apple9_native.layout.vertex_components = 56;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      main->info.apple9_native.layout.vertex_components = 60;
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
mark_draw(struct hk_cs *cs, bool native)
{
   /* The classification/transition test owns CPU storage, not GPU execution.
    * Reserve representative stream bytes before the real next preparation. */
   cs->current += 160;
   cs->stats.cmds++;
   cs->apple9_native_mesh = native;
}

static void
transitions(struct hk_cmd_buffer *cmd)
{
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

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   routes(cmd);
   task_routes(cmd);
   gs_routes(cmd);
   tess_routes(cmd);
   transitions(cmd);
   free(cmd);
   free(dev);
   puts(
      "hybrid draw routing, partial attachment state and continuation ordering pass");
   return 0;
}
