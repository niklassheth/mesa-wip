/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Host validation of Phase 2 native hardware OBJECT -> MESH with <= 16 KiB
 * task payload, multi-parent VDM encoding, pipeline compilation + cache
 * serialization roundtrip, draw-time routing/isolation, and clean fallback to
 * Phase 1 Task Bridge / Emulation.
 */
#include "../hk_cmd_apple9.c"
#include "../hk_entrypoints.h"
#include "../hk_shader_priv.h"
#include "compiler/nir/nir_builder.h"
#include "util/blob.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         abort();                                                              \
      }                                                                        \
   } while (0)
#include "hk-test-usc.h"

static unsigned allocations, live_bos, vk_allocs, fail_vk_alloc_at;

static void *
test_vk_alloc(void *user, size_t size, size_t align,
              VkSystemAllocationScope scope)
{
   (void)user;
   (void)scope;
   ++vk_allocs;
   if (fail_vk_alloc_at && vk_allocs == fail_vk_alloc_at)
      return NULL;
   void *p = NULL;
   if (align < sizeof(void *))
      align = sizeof(void *);
   return posix_memalign(&p, align, MAX2(size, 1)) == 0 ? p : NULL;
}

static void *
test_vk_realloc(void *user, void *orig, size_t size, size_t align,
                VkSystemAllocationScope scope)
{
   (void)user;
   (void)align;
   (void)scope;
   ++vk_allocs;
   if (fail_vk_alloc_at && vk_allocs == fail_vk_alloc_at)
      return NULL;
   return realloc(orig, MAX2(size, 1));
}

static void
test_vk_free(void *user, void *ptr)
{
   (void)user;
   free(ptr);
}

static const VkAllocationCallbacks test_alloc_cb = {
   .pfnAllocation = test_vk_alloc,
   .pfnReallocation = test_vk_realloc,
   .pfnFree = test_vk_free,
};

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int,
                            const char *, ...);

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
   bo->va->addr = dev->shader_base + allocations * 0x100000ull;
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

VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

static void
enable_native_mesh_kernel(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   dev->dev.vm_id = 1;
   dev->dev.params.features |= DRM_ASAHI_FEATURE_NATIVE_MESH;
   if (pdev)
      pdev->dev.params.features |= DRM_ASAHI_FEATURE_NATIVE_MESH;
}

static void
disable_native_mesh_kernel(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   dev->dev.vm_id = 1;
   dev->dev.params.features &= ~DRM_ASAHI_FEATURE_NATIVE_MESH;
   if (pdev)
      pdev->dev.params.features &= ~DRM_ASAHI_FEATURE_NATIVE_MESH;
}

static nir_shader *
make_task_nir(unsigned wx, unsigned payload_B, unsigned shared_B,
              unsigned child_x, bool dynamic_count)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "test_native_object_task");
   b.shader->info.workgroup_size[0] = wx;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = payload_B;
   b.shader->info.shared_size = shared_B;

   nir_def *lid = nir_load_local_invocation_index(&b);
   nir_def *val = nir_iadd(&b, lid, nir_channel(&b, nir_load_workgroup_id(&b), 0));
   if (shared_B >= 4) {
      nir_def *off = nir_imul_imm(&b, nir_iand_imm(&b, lid, (shared_B / 4) - 1), 4);
      nir_store_shared(&b, val, off, .write_mask = 1, .align_mul = 4);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
      val = nir_iadd(&b, val, nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0),
                                              .align_mul = 4));
   }
   if (payload_B >= 4) {
      nir_push_if(&b, nir_ieq_imm(&b, lid, 0));
      nir_store_task_payload(&b, val, nir_imm_int(&b, 0),
                             .write_mask = 1, .align_mul = 4);
      if (payload_B >= 8) {
         nir_store_task_payload(&b, val, nir_imm_int(&b, payload_B - 4),
                                .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(&b, NULL);
   }
   nir_def *gx = dynamic_count
                    ? nir_iadd_imm(&b, nir_channel(&b, nir_load_workgroup_id(&b), 0), 1)
                    : nir_imm_int(&b, child_x);
   nir_launch_mesh_workgroups(&b, nir_vec3(&b, gx, nir_imm_int(&b, 1),
                                           nir_imm_int(&b, 1)));
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_mesh_nir(unsigned payload_B, bool read_num_workgroups)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "test_native_object_mesh");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = payload_B;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;

   nir_def *v = nir_imm_int(&b, 1);
   if (payload_B >= 4) {
      v = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, payload_B - 4),
                                .align_mul = 4);
   }
   if (read_num_workgroups) {
      v = nir_iadd(&b, v, nir_channel(&b, nir_load_num_workgroups(&b), 0));
   }
   nir_def *xf = nir_u2f32(&b, v);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i) {
      nir_store_per_vertex_output(
         &b,
         nir_vec4(&b, xf, nir_imm_float(&b, i == 2 ? 0.5f : -0.5f),
                  nir_imm_float(&b, 0.5f), nir_imm_float(&b, 1.0f)),
         nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
make_fs_nir(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "test_native_object_fs");
   nir_store_output(&b, nir_imm_vec4(&b, 0.2f, 0.4f, 0.6f, 1.0f),
                    nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0,
                                     .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static struct hk_api_shader *
roundtrip_shader(struct hk_device *dev, struct hk_api_shader *src)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(src->vk.ops->serialize(&dev->vk, &src->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *copy_vk = NULL;
   CHECK(hk_device_shader_ops.deserialize(&dev->vk, &reader, 0, NULL,
                                          &copy_vk) == VK_SUCCESS);
   CHECK(!reader.overrun && reader.current == reader.end);
   blob_finish(&blob);
   return container_of(copy_vk, struct hk_api_shader, vk);
}

static void
check_shader_and_vdm_contract(void)
{
   /* Verify HK_APPLE9_NATIVE_OBJECT launch params and validation for T16, T32,
    * serialized T64/T128, shared memory, and 16 KiB task payload. */
   const struct {
      unsigned wx, phys, payload_B, shared_B;
   } cases[] = {
      {1, 16, 0, 0},
      {16, 16, 256, 256},
      {32, 32, 16384, 1024},
      {64, 32, 256, 4096},
      {128, 32, 0, 16384},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_shader *nir = make_task_nir(cases[i].wx, cases[i].payload_B,
                                      cases[i].shared_B, 2, false);
      struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      key.physical_threads = cases[i].phys;
      struct agx_apple9_native_object_options opts = {
         .payload_stride_B = ALIGN_POT(cases[i].payload_B, 16),
         .write_child_grid = cases[i].payload_B > 0,
      };
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info = {0};
      const char *reason = NULL;
      CHECK(agx_compile_apple9_native_object_ex(nir, true, &key, &opts, &part,
                                                &info, &reason));
      CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                          HK_APPLE9_NATIVE_OBJECT));
      struct agx_apple9_launch_parameters launch = {0};
      const uint64_t base = UINT64_C(0x1200000000), code = base + 0x20000;
      CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &info,
                                               HK_APPLE9_NATIVE_OBJECT,
                                               base, code));
      CHECK(launch.threadgroup_memory_bytes ==
            util_next_power_of_two(MAX2(128u, HK_APPLE9_NATIVE_GRID_ROOT +
                                                 info.local_allocation_B)));
      CHECK(launch.threadgroup_memory_bytes <= 32768);
      /* Reject spill, payload > 16 KiB, and shared overflow. */
      struct agx_shader_info bad_code = part.info;
      bad_code.scratch_size = 4;
      CHECK(!hk_apple9_native_shader_valid(&bad_code, &info,
                                           HK_APPLE9_NATIVE_OBJECT));
      struct agx_apple9_mesh_body_info bad_info = info;
      bad_info.task_payload_size_B = AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES + 4;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &bad_info,
                                           HK_APPLE9_NATIVE_OBJECT));
      bad_info = info;
      bad_info.local_allocation_B = 32768 - HK_APPLE9_NATIVE_GRID_ROOT + 16;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &bad_info,
                                           HK_APPLE9_NATIVE_OBJECT));
      free(part.binary);
      ralloc_free(nir);
   }

   /* Multi-parent VDM launch encoding vs strict Phase 1 single-parent gate. */
   uint32_t state[AGX_APPLE9_MESH_STATE_WORDS] = {0};
   state[0] = 0x40000d2c;
   struct agx_apple9_render_pipeline pipeline = {
      .ppp = 0x4000,
      .primitive = AGX_PRIMITIVE_TRIANGLES,
      .ppp_viewport_offset = 0x100,
      .ppp_viewport_words = 10,
   };
   const uint32_t g_multi[3] = {4, 3, 2};
   const uint32_t g_max[3] = {65535, 16, 1};
   const uint32_t g_over[3] = {65536, 1, 1};
   const uint32_t g_total_over[3] = {1024, 1024, 1};
   const uint32_t l16[3] = {16, 1, 1}, l32[3] = {32, 1, 1}, l64[3] = {64, 1, 1};
   uint32_t cmd_words[HK_APPLE9_MESH_COMMAND_BYTES / 4];
   uint32_t saved[HK_APPLE9_MESH_COMMAND_BYTES / 4];

   /* multi_parent=true accepts 3D multi-parent grids with T16 and T32. */
   CHECK(hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                         state, g_multi, l16, true, true));
   CHECK(cmd_words[0] == 0x40000d2c && cmd_words[39] == 0xc0000000);
   CHECK(cmd_words[33] == 64 && cmd_words[34] == 3 && cmd_words[35] == 2);
   CHECK(cmd_words[36] == 16 && cmd_words[37] == 1 && cmd_words[38] == 1);

   CHECK(hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                         state, g_multi, l32, true, true));
   CHECK(cmd_words[33] == 128 && cmd_words[34] == 3 && cmd_words[35] == 2);
   CHECK(cmd_words[36] == 32 && cmd_words[37] == 1 && cmd_words[38] == 1);

   CHECK(hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                         state, g_max, l32, true, true));
   CHECK(cmd_words[33] == 65535u * 32u && cmd_words[34] == 16 &&
         cmd_words[35] == 1);

   /* Delta build also accepts multi-parent + T32. */
   size_t delta_bytes = 0;
   uint8_t prev_ppp[256] = {0}, cur_ppp[256] = {0};
   CHECK(hk_apple9_mesh_command_build_delta_ex(
      cmd_words, sizeof(cmd_words), &pipeline, state, g_multi, l32, true, true,
      prev_ppp, cur_ppp, &delta_bytes));
   CHECK(delta_bytes >= 12 * 4 && delta_bytes <= HK_APPLE9_MESH_COMMAND_BYTES);

   /* Reject out-of-range grids / local sizes without mutating cmd_words. */
   memcpy(saved, cmd_words, sizeof(cmd_words));
   CHECK(!hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                          state, g_over, l32, true, true));
   CHECK(!memcmp(saved, cmd_words, sizeof(cmd_words)));
   CHECK(!hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                          state, g_total_over, l32, true, true));
   CHECK(!memcmp(saved, cmd_words, sizeof(cmd_words)));
   CHECK(!hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                          state, g_multi, l64, true, true));
   CHECK(!memcmp(saved, cmd_words, sizeof(cmd_words)));
   /* Phase 1 multi_parent=false strictly rejects groups > 1 and T32. */
   CHECK(!hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                          state, g_multi, l16, true, false));
   CHECK(!memcmp(saved, cmd_words, sizeof(cmd_words)));
   const uint32_t g1[3] = {1, 1, 1};
   CHECK(!hk_apple9_mesh_command_build_ex(cmd_words, sizeof(cmd_words), &pipeline,
                                          state, g1, l32, true, false));
   CHECK(!memcmp(saved, cmd_words, sizeof(cmd_words)));
}

static void
compile_pair(struct hk_device *dev, unsigned wx, unsigned payload_B,
             unsigned shared_B, unsigned child_x, bool dynamic_count,
             bool mesh_reads_nwg, struct hk_api_shader **out_task,
             struct hk_api_shader **out_mesh, struct hk_api_shader **out_fs)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_TASK,
       .nir = make_task_nir(wx, payload_B, shared_B, child_x, dynamic_count),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_MESH,
       .nir = make_mesh_nir(payload_B, mesh_reads_nwg),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT,
       .nir = make_fs_nir(),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[3] = {0};
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1,
      .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);
   VkResult res =
      hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL, NULL, out);
   CHECK(res == VK_SUCCESS);
   *out_task = container_of(out[0], struct hk_api_shader, vk);
   *out_mesh = container_of(out[1], struct hk_api_shader, vk);
   *out_fs = container_of(out[2], struct hk_api_shader, vk);
}

static void
destroy_pair(struct hk_device *dev, struct hk_api_shader *t,
             struct hk_api_shader *m, struct hk_api_shader *f)
{
   t->vk.ops->destroy(&dev->vk, &t->vk, NULL);
   m->vk.ops->destroy(&dev->vk, &m->vk, NULL);
   f->vk.ops->destroy(&dev->vk, &f->vk, NULL);
}

static void
check_vdm_recording(struct hk_device *dev, struct hk_api_shader *task,
                    struct hk_api_shader *mesh, struct hk_api_shader *fs,
                    const uint32_t groups[3], uint64_t payload_gpu,
                    unsigned expect_local, uint32_t expect_word9)
{
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(cmd);
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   uint8_t *pool = calloc(3, HK_CMD_BO_SIZE);
   CHECK(pool);
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base =
      dev->dev.shader_base + 0x10000000ull;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base =
      AGX_APPLE9_RENDER_CONTEXT_BASE + 0x1000000ull;
   cmd->uploader[HK_CMD_POOL_USC].chunks[0].map = pool + 2 * HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_USC].chunks[0].base =
      dev->dev.shader_base + 0x30000000ull;

   cmd->state.gfx.shaders[MESA_SHADER_TASK] = task;
   cmd->state.gfx.shaders[MESA_SHADER_MESH] = mesh;
   cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = fs;
   cmd->state.gfx.render.layer_count = 1;
   cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;

   struct hk_api_shader *native_mesh = hk_api_shader_native(mesh);
   struct hk_shader *main = &native_mesh->variants[HK_MESH_VARIANT_MAIN];
   uint32_t command[256] = {0};
   struct hk_cs cs = {
      .cmd = cmd,
      .type = HK_CS_VDM,
      .start = (uint8_t *)command,
      .current = (uint8_t *)command,
   };
   const struct agx_apple9_render_stage stage = {
      .binary = main->b.binary,
      .binary_size = main->b.info.binary_size,
   };
   const struct agx_apple9_render_pipeline pipeline = {
      .vertex = &stage,
      .fragment = &stage,
      .ppp = 0x4000,
      .primitive = AGX_PRIMITIVE_TRIANGLES,
      .ppp_viewport_offset = 0x100,
      .ppp_viewport_words = 10,
   };
   const struct agx_apple9_uniform_draw record = {0};
   const struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   CHECK(hk_apple9_record_draw(cmd, &cs, &pipeline, draw, 0x8000, groups,
                               payload_gpu, native_mesh, main, &record));
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   CHECK(cs.apple9_native_mesh);
   CHECK(!(cs.pre_raster_reads & HK_READS_INTERNAL));
   CHECK(command[0] == 0x40000d2c);
   CHECK(command[9] == expect_word9);
   CHECK(command[33] == groups[0] * expect_local);
   CHECK(command[34] == groups[1]);
   CHECK(command[35] == groups[2]);
   CHECK(command[36] == expect_local);
   CHECK(command[37] == 1 && command[38] == 1);
   CHECK(command[39] == 0xc0000000);

   /* Release pinned shader bodies owned by this synthetic cmd. */
   hk_apple9_unpin_entries(cmd);
   if (cmd->apple9_pinned)
      _mesa_set_destroy(cmd->apple9_pinned, NULL);
   free(pool);
   free(cmd);
}

static void
check_bridge_prepare_and_record(struct hk_device *dev,
                                struct hk_api_shader *task,
                                struct hk_api_shader *mesh,
                                struct hk_api_shader *fs)
{
   struct agx_va fetch_va = {.addr = dev->dev.shader_base + 0x10000};
   struct agx_bo fetch_bo = {.va = &fetch_va};
   dev->apple9_fetch_sampler = &fetch_bo;
   uint8_t *entry_mem = calloc(1, AGX_APPLE9_ENTRY_CHUNK_SIZE);
   CHECK(entry_mem);
   struct agx_va entry_va = {.addr = dev->dev.shader_base + 0x20000};
   struct agx_bo entry_bo = {.va = &entry_va, ._map = entry_mem,
                             .size = AGX_APPLE9_ENTRY_CHUNK_SIZE, .refcnt = 1};
   dev->dev.apple9_entry_va = &entry_va;
   dev->dev.apple9_entry_chunks[0] = &entry_bo;
   list_inithead(&dev->dev.apple9_entry_lru);
   util_vma_heap_init(&dev->dev.apple9_entry_heap, AGX_APPLE9_ENTRY_HEADER_SIZE,
                      AGX_APPLE9_ENTRY_CHUNK_SIZE - AGX_APPLE9_ENTRY_HEADER_SIZE);

   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(cmd);
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   uint8_t *pool = calloc(3, HK_CMD_BO_SIZE);
   CHECK(pool);
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base =
      dev->dev.shader_base + 0x10000000ull;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base =
      AGX_APPLE9_RENDER_CONTEXT_BASE + 0x1000000ull;
   cmd->uploader[HK_CMD_POOL_USC].chunks[0].map = pool + 2 * HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_USC].chunks[0].base =
      dev->dev.shader_base + 0x30000000ull;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);

   cmd->state.gfx.shaders[MESA_SHADER_TASK] = task;
   cmd->state.gfx.shaders[MESA_SHADER_MESH] = mesh;
   cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = fs;
   cmd->state.gfx.render.layer_count = 1;
   cmd->state.gfx.render.color_att_count = 1;
   cmd->state.gfx.render.area = (VkRect2D){{0, 0}, {64, 64}};
   cmd->state.gfx.render.tilebuffer.nr_samples = 1;
   cmd->vk.dynamic_graphics_state.ms.rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
   cmd->vk.dynamic_graphics_state.ms.sample_mask = 0x1;
   cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
   cmd->vk.dynamic_graphics_state.rs.rasterizer_discard_enable = false;
   cmd->vk.dynamic_graphics_state.vp.viewport_count = 1;
   cmd->vk.dynamic_graphics_state.vp.scissor_count = 1;
   cmd->vk.dynamic_graphics_state.vp.viewports[0] =
      (VkViewport){0, 0, 64, 64, 0, 1};
   cmd->vk.dynamic_graphics_state.vp.scissors[0] = (VkRect2D){{0, 0}, {64, 64}};

   /* Distinguish primary native_object FS vs secondary task_bridge FS so we
    * prove exact per-tier FS leaf provenance (not just linked wrapper match). */
   struct hk_api_shader *nf = hk_api_shader_native(fs);
   struct hk_api_shader *bf = hk_api_shader_task_bridge(fs);
   const bool saved_nf_discard = nf->variants[0].apple9_stage.uses_discard;
   const bool saved_bf_discard = bf->variants[0].apple9_stage.uses_discard;
   nf->variants[0].apple9_stage.uses_discard = false;
   bf->variants[0].apple9_stage.uses_discard = true;

   /* Tier 1 (Native OBJECT): no PREGFX compute -> must NOT set HK_READS_INTERNAL. */
   CHECK(hk_apple9_prepare_draw_ex(cmd, true, true, true));
   struct hk_cs *cs_obj = cmd->current_cs.gfx;
   CHECK(cs_obj && !(cs_obj->pre_raster_reads & HK_READS_INTERNAL));
   const uint32_t g1[3] = {1, 1, 1};
   uint32_t *vdm_obj = (uint32_t *)cs_obj->current;
   hk_apple9_draw_native_object(cmd, 0, g1, 0x1400000000ull, 0x1600000000ull);
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   CHECK(cs_obj->apple9_native_mesh && cs_obj->apple9_native_dispatches == 1);
   CHECK(!(cs_obj->pre_raster_reads & HK_READS_INTERNAL));
   CHECK(vdm_obj[0] == 0x40000d2c && vdm_obj[33] == 32 && vdm_obj[36] == 32);
   struct hk_api_shader *nm = hk_api_shader_native(mesh);
   CHECK(cmd->apple9_linked.vertex == &nm->variants[HK_MESH_VARIANT_MAIN].apple9_stage);
   CHECK(!cmd->apple9_linked.fragment->uses_discard);

   /* Tier 2 (Cascaded Task Bridge): merges into the same render pass under
    * opaque bit 28, promotes HK_READS_INTERNAL for PREGFX task+prefix, and
    * hk_apple9_draw_native_task_bridge(..., true) binds and records the
    * secondary bridge mesh/object/FS leaves (not the primary native_object leaf). */
   CHECK(hk_apple9_prepare_draw_ex(cmd, true, true, false));
   struct hk_cs *cs_br = cmd->current_cs.gfx;
   CHECK(cs_br && cs_br == cs_obj && (cs_br->pre_raster_reads & HK_READS_INTERNAL));
   hk_apple9_draw_native_task_bridge(cmd, 0, 0x1400000000ull, true);
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   CHECK(cs_br->apple9_native_mesh && cs_br->apple9_native_dispatches == 2);
   struct hk_api_shader *bm = hk_api_shader_task_bridge(mesh);
   CHECK(cmd->apple9_linked.vertex == &bm->variants[HK_MESH_VARIANT_MAIN].apple9_stage);
   /* Confirm secondary bridge FS leaf was linked (uses_discard == true). */
   CHECK(cmd->apple9_linked.fragment->uses_discard);

   /* Tier 3 (Emulated/ordinary transition after native): stays in same pass! */
   CHECK(hk_apple9_prepare_draw_ex(cmd, false, false, false));
   CHECK(cmd->current_cs.gfx == cs_br);

   /* Also verify standalone full-PPP VDM layout of secondary Task Bridge (T16 1,1,1). */
   hk_cmd_buffer_end_graphics(cmd);
   CHECK(hk_apple9_prepare_draw_ex(cmd, true, true, false));
   struct hk_cs *cs_br_fresh = cmd->current_cs.gfx;
   CHECK(cs_br_fresh && cs_br_fresh != cs_br &&
         (cs_br_fresh->pre_raster_reads & HK_READS_INTERNAL));
   uint32_t *vdm = (uint32_t *)cs_br_fresh->current;
   hk_apple9_draw_native_task_bridge(cmd, 0, 0x1400000000ull, true);
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   CHECK(cs_br_fresh->apple9_native_mesh && cs_br_fresh->apple9_native_dispatches == 1);
   CHECK(vdm[0] == 0x40000d2c && vdm[39] == 0xc0000000);
   CHECK(vdm[33] == 16 && vdm[34] == 1 && vdm[35] == 1);
   CHECK(vdm[36] == 16 && vdm[37] == 1 && vdm[38] == 1);
   nf->variants[0].apple9_stage.uses_discard = saved_nf_discard;
   bf->variants[0].apple9_stage.uses_discard = saved_bf_discard;

   /* End-to-end multiview ordinary hk_CmdDraw after a 900k capped native draw:
    * 1) 2 views x 40k prims (980k <= 1M): shares the native render pass.
    * 2) 2 views x 60k prims (900k + 120k > 1M): splits cleanly before view 0
    *    and records both views with VK_SUCCESS (never hitting UNSUPPORTED()). */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "test_mv_vs");
      nir_variable *pos =
         nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "pos");
      pos->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, pos, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
      struct vk_shader_compile_info mv_infos[2] = {
         {.stage = MESA_SHADER_VERTEX, .nir = b.shader, .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT, .nir = make_fs_nir(), .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *mv_out[2] = {0};
      const struct vk_render_pass_state rp = {
         .color_attachment_count = 1,
         .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
      };
      const struct vk_color_blend_state cb = {
         .attachment_count = 1,
         .color_write_enables = 1,
         .attachments = {{.write_mask = 15}},
      };
      const struct vk_graphics_pipeline_state st = {.rp = &rp, .cb = &cb};
      struct hk_apple9_graphics_key gkey;
      hk_apple9_graphics_key(&gkey, &st);
      gkey.is_multiview = true;
      CHECK(hk_apple9_compile_graphics(dev, 2, mv_infos, NULL, &gkey, NULL,
                                       NULL, mv_out) == VK_SUCCESS);
      struct hk_api_shader *mv_vs = container_of(mv_out[0], struct hk_api_shader, vk);
      struct hk_api_shader *mv_fs = container_of(mv_out[1], struct hk_api_shader, vk);

      cmd->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
      cmd->state.gfx.shaders[MESA_SHADER_TASK] = NULL;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = NULL;
      cmd->state.gfx.shaders[MESA_SHADER_VERTEX] = mv_vs;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = mv_fs;
      cmd->state.gfx.render.view_mask = 0x3;
      cmd->state.gfx.render.layer_count = 2;
      cmd->vk.dynamic_graphics_state.ia.primitive_topology =
         VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

      cs_br_fresh->apple9_native_primitives = 900000;
      hk_CmdDraw(hk_cmd_buffer_to_handle(cmd), 3 * 40000, 1, 0, 0);
      CHECK(cmd->vk.record_result == VK_SUCCESS);
      CHECK(cmd->current_cs.gfx == cs_br_fresh);
      CHECK(cs_br_fresh->apple9_native_primitives == 980000);

      cs_br_fresh->apple9_native_primitives = 900000;
      hk_CmdDraw(hk_cmd_buffer_to_handle(cmd), 3 * 60000, 1, 0, 0);
      CHECK(cmd->vk.record_result == VK_SUCCESS);
      CHECK(cmd->current_cs.gfx && cmd->current_cs.gfx != cs_br_fresh);
      CHECK(!cmd->current_cs.gfx->apple9_native_mesh);
      CHECK(cmd->current_cs.gfx->stats.cmds == 2);
      CHECK(cmd->current_cs.gfx->apple9_native_primitives == 120000);

      mv_vs->vk.ops->destroy(&dev->vk, &mv_vs->vk, NULL);
      mv_fs->vk.ops->destroy(&dev->vk, &mv_fs->vk, NULL);
   }

   hk_apple9_unpin_entries(cmd);
   if (cmd->apple9_pinned)
      _mesa_set_destroy(cmd->apple9_pinned, NULL);
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   hk_apple9_linked_entries_finish(dev);
   hk_apple9_fs_epilogs_finish(dev);
   list_for_each_entry_safe(struct agx_shader_body, body,
                            &dev->dev.apple9_entry_lru, entry_link) {
      body->entry_offset = 0;
      list_del(&body->entry_link);
   }
   util_vma_heap_finish(&dev->dev.apple9_entry_heap);
   dev->dev.apple9_entry_chunks[0] = NULL;
   dev->dev.apple9_entry_va = NULL;
   free(entry_mem);
   free(pool);
   free(cmd);
   dev->apple9_fetch_sampler = NULL;
}

static void
check_dual_leaf_failure_cleanup(struct hk_device *dev)
{
   /* OOM injection across every vk_shader allocation point (emulated 1..3,
    * primary native_object 4..6, and secondary apple9_task_bridge 7..9) must
    * return VK_ERROR_OUT_OF_HOST_MEMORY, leave out[] NULL, and leak 0 BOs. */
   hk_apple9_linked_entries_finish(dev);
   hk_apple9_fs_epilogs_finish(dev);
   hk_apple9_bodies_finish(dev);
   unsigned before_oom_bos = live_bos;

   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1,
      .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);

   unsigned before_vk = vk_allocs;
   {
      struct vk_shader_compile_info infos[] = {
         {.stage = MESA_SHADER_TASK,
          .nir = make_task_nir(32, 2048, 0, 4, false),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_MESH,
          .nir = make_mesh_nir(2048, false),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT,
          .nir = make_fs_nir(),
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out[3] = {0};
      CHECK(hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL,
                                       &test_alloc_cb, out) == VK_SUCCESS);
      for (unsigned i = 0; i < 3; ++i)
         out[i]->ops->destroy(&dev->vk, out[i], &test_alloc_cb);
   }
   unsigned needed_vk = vk_allocs - before_vk;
   /* 3 stages x 3 realizations (emulated + native_object + task_bridge) = >= 9. */
   CHECK(needed_vk >= 9);
   hk_apple9_bodies_finish(dev);
   CHECK(live_bos == before_oom_bos);

   for (unsigned k = 1; k <= needed_vk; ++k) {
      fail_vk_alloc_at = vk_allocs + k;
      struct vk_shader_compile_info infos[] = {
         {.stage = MESA_SHADER_TASK,
          .nir = make_task_nir(32, 2048, 0, 4, false),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_MESH,
          .nir = make_mesh_nir(2048, false),
          .robustness = &vk_robustness_disabled},
         {.stage = MESA_SHADER_FRAGMENT,
          .nir = make_fs_nir(),
          .robustness = &vk_robustness_disabled},
      };
      struct vk_shader *out[3] = {0};
      VkResult res = hk_apple9_compile_graphics(dev, 3, infos, NULL, &key, NULL,
                                                &test_alloc_cb, out);
      fail_vk_alloc_at = 0;
      CHECK(res == VK_ERROR_OUT_OF_HOST_MEMORY);
      CHECK(!out[0] && !out[1] && !out[2]);
      hk_apple9_bodies_finish(dev);
      CHECK(live_bos == before_oom_bos);
   }
}

int
main(int argc, char **argv)
{
   const char *mode = argc > 1 ? argv[1] : "phase2";
   glsl_type_singleton_init_or_ref();

   check_shader_and_vdm_contract();

   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = pdev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   simple_mtx_init(&dev->dev.apple9_entry_back_lock, mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&dev->apple9_fs_epilogs.lock, mtx_plain);
   simple_mtx_init(&dev->apple9_linked_entries.lock, mtx_plain);
   list_inithead(&dev->apple9_linked_entries.lru);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);

   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(cmd);
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   cmd->state.gfx.render.layer_count = 1;
   cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;

   if (!strcmp(mode, "mesh-disabled")) {
      /* HK_APPLE9_NATIVE_MESH=0: completely untouched even with bit 28 set. */
      enable_native_mesh_kernel(dev);
      CHECK(!hk_apple9_native_mesh_enabled());
      CHECK(!hk_apple9_native_object_available_device(dev));
      struct hk_api_shader *t, *m, *f;
      compile_pair(dev, 32, 256, 128, 4, false, true, &t, &m, &f);
      CHECK(!hk_api_shader_native(t) && !hk_api_shader_native(m));
      cmd->state.gfx.shaders[MESA_SHADER_TASK] = t;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = m;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f;
      const uint32_t g1[3] = {1, 1, 1};
      CHECK(!hk_apple9_use_native_mesh(cmd, g1, 0));
      destroy_pair(dev, t, m, f);
   } else if (!strcmp(mode, "object-disabled")) {
      /* HK_APPLE9_NATIVE_MESH=1 + HK_APPLE9_NATIVE_OBJECT=0: falls back to
       * Task Bridge even when DRM_ASAHI_FEATURE_NATIVE_MESH is advertised. */
      enable_native_mesh_kernel(dev);
      CHECK(!hk_apple9_native_object_enabled());
      CHECK(!hk_apple9_native_object_available_device(dev));
      struct hk_api_shader *t, *m, *f;
      compile_pair(dev, 32, 256, 128, 4, false, true, &t, &m, &f);
      struct hk_api_shader *nm = hk_api_shader_native(m);
      CHECK(nm && !nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object &&
            nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      destroy_pair(dev, t, m, f);
   } else {
      /* Without DRM_ASAHI_FEATURE_NATIVE_MESH: native mesh & object are off. */
      disable_native_mesh_kernel(dev);
      CHECK(!hk_apple9_native_mesh_available_device(dev));
      CHECK(!hk_apple9_native_object_available_device(dev));
      struct hk_api_shader *t, *m, *f;
      compile_pair(dev, 32, 256, 128, 4, false, true, &t, &m, &f);
      CHECK(!hk_api_shader_native(t) && !hk_api_shader_native(m));
      destroy_pair(dev, t, m, f);

      /* With DRM_ASAHI_FEATURE_NATIVE_MESH (mp_payload_bytes=1024): real
       * application OBJECT -> MESH with 256 B payload + child grid (272 B <=
       * 1024 B) uses GET_SR 0xe0/0xe1 AND retains dual-leaf Task Bridge. */
      enable_native_mesh_kernel(dev);
      CHECK(hk_apple9_native_object_available_device(dev));
      compile_pair(dev, 64, 256, 128, 4, false, true, &t, &m, &f);
      struct hk_api_shader *nt = hk_api_shader_native(t);
      struct hk_api_shader *nm = hk_api_shader_native(m);
      struct hk_api_shader *nf = hk_api_shader_native(f);
      struct hk_api_shader *bt = hk_api_shader_task_bridge(t);
      struct hk_api_shader *bm = hk_api_shader_task_bridge(m);
      struct hk_api_shader *bf = hk_api_shader_task_bridge(f);
      CHECK(nt && nm && nf && bt && bm && bf);
      CHECK(nt->apple9_native_leaf && nm->apple9_native_leaf);
      CHECK(bt->apple9_native_leaf && bm->apple9_native_leaf &&
            bf->apple9_native_leaf);
      CHECK(nt->variants[0].info.apple9_native_stage == HK_APPLE9_NATIVE_OBJECT);
      CHECK(nt->variants[0].info.apple9_native.key.hw_payload);
      CHECK(nt->variants[0].info.apple9_native.writes_child_grid);
      CHECK(nt->variants[0].info.apple9_native.task_payload_size_B == 256);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object);
      CHECK(!nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.key.hw_payload);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.reads_native_payload);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.reads_child_grid);
      CHECK(nm->variants[HK_MESH_VARIANT_OBJECT].code_size == 0);
      CHECK(bt->variants[0].info.apple9_native_stage == HK_APPLE9_NATIVE_NONE);
      CHECK(bm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      CHECK(!bm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object);
      CHECK(memcmp(bt->apple9_native_pair_id, bm->apple9_native_pair_id,
                   sizeof(bm->apple9_native_pair_id)) == 0);
      CHECK(memcmp(bf->apple9_native_pair_id, bm->apple9_native_pair_id,
                   sizeof(bm->apple9_native_pair_id)) == 0);
      CHECK(memcmp(nt->apple9_native_pair_id, bt->apple9_native_pair_id,
                   sizeof(bt->apple9_native_pair_id)) != 0);

      /* Cache identity must track DRM_ASAHI_FEATURE_NATIVE_MESH. */
      pdev->dev.chip = AGX_CHIP_G16G;
      pdev->dev.params.features = 0;
      uint64_t flags_off = hk_physical_device_compiler_flags(pdev);
      pdev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
      uint64_t flags_on = hk_physical_device_compiler_flags(pdev);
      CHECK(flags_off != flags_on);

      /* Cache roundtrip preserves BOTH native object and dual-leaf Task Bridge,
       * and rejects cached native leaves when bit 28 is absent. */
      struct hk_api_shader *t_rt = roundtrip_shader(dev, t);
      struct hk_api_shader *m_rt = roundtrip_shader(dev, m);
      struct hk_api_shader *f_rt = roundtrip_shader(dev, f);
      CHECK(hk_api_shader_native(t_rt) &&
            hk_api_shader_native(t_rt)->variants[0].info.apple9_native_stage ==
               HK_APPLE9_NATIVE_OBJECT &&
            hk_api_shader_native(t_rt)->variants[0].info.apple9_native.key.hw_payload);
      CHECK(hk_api_shader_native(m_rt)->variants[HK_MESH_VARIANT_MAIN]
               .info.mesh.native_object &&
            hk_api_shader_native(m_rt)->variants[HK_MESH_VARIANT_MAIN]
               .info.apple9_native.key.hw_payload);
      CHECK(hk_api_shader_task_bridge(t_rt) &&
            hk_api_shader_task_bridge(m_rt) &&
            hk_api_shader_task_bridge(f_rt));
      {
         struct blob tb, mb;
         blob_init(&tb);
         blob_init(&mb);
         CHECK(t->vk.ops->serialize(&dev->vk, &t->vk, &tb));
         CHECK(m->vk.ops->serialize(&dev->vk, &m->vk, &mb));

         /* Downgrade to kernel without bit 28: cached native leaves fail closed. */
         disable_native_mesh_kernel(dev);
         struct blob_reader tr, mr;
         struct vk_shader *out_vk = NULL;
         blob_reader_init(&tr, tb.data, tb.size);
         CHECK(hk_device_shader_ops.deserialize(&dev->vk, &tr, 0, NULL,
                                                &out_vk) ==
               VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
         CHECK(!out_vk);
         blob_reader_init(&mr, mb.data, mb.size);
         CHECK(hk_device_shader_ops.deserialize(&dev->vk, &mr, 0, NULL,
                                                &out_vk) ==
               VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
         CHECK(!out_vk);

         enable_native_mesh_kernel(dev);
         blob_finish(&tb);
         blob_finish(&mb);
      }

      /* Routing for hw_payload (256 B + 16 B child grid <= 1024 B):
       * BOTH single-parent and multi-parent route Native OBJECT (Zero-CDM)
       * with payload_gpu=0! Indirect/cond_render fall back to Emulated. */
      cmd->state.gfx.shaders[MESA_SHADER_TASK] = t_rt;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = m_rt;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f_rt;
      const uint32_t g1[3] = {1, 1, 1}, g64[3] = {64, 1, 1},
                     g1024[3] = {64, 4, 4};
      CHECK(hk_apple9_select_native_mesh_route(cmd, g1, 0) ==
            HK_APPLE9_MESH_ROUTE_NATIVE);
      CHECK(hk_apple9_select_native_mesh_route(cmd, g64, 0) ==
            HK_APPLE9_MESH_ROUTE_NATIVE);
      CHECK(hk_apple9_select_native_mesh_route(cmd, g1024, 0) ==
            HK_APPLE9_MESH_ROUTE_NATIVE);
      CHECK(hk_apple9_select_native_mesh_route(cmd, NULL, 0) ==
            HK_APPLE9_MESH_ROUTE_EMULATED);
      cmd->state.cond_render.active = true;
      CHECK(hk_apple9_select_native_mesh_route(cmd, g1, 0) ==
            HK_APPLE9_MESH_ROUTE_EMULATED);
      cmd->state.cond_render.active = false;
      check_vdm_recording(dev, t_rt, m_rt, f_rt, g1024, 0, 32, 1);

      destroy_pair(dev, t_rt, m_rt, f_rt);
      destroy_pair(dev, t, m, f);

      /* Hardware 16 KiB payload boundaries (words[9] = ceil(payload / 1024)):
       * 1008 B + 16 B child grid = 1024 B -> hw_payload=true (words[9]=1).
       * 1024 B + 16 B child grid = 1040 B -> hw_payload=true (words[9]=2).
       * 2048 B & 16384 B without child grid -> hw_payload=true (words[9]=2/16,
       *   multi-parent Native OBJECT!).
       * 16368 B + 16 B child grid = 16384 B -> hw_payload=true (words[9]=16).
       * 16384 B + 16 B child grid = 16400 B > 16384 B -> hw_payload=false
       *   (single-parent Native via args.payload, multi-parent cascades cleanly
       *   to Task Bridge!). */
      compile_pair(dev, 32, 1008, 0, 4, false, true, &t, &m, &f);
      CHECK(hk_api_shader_native(t)->variants[0].info.apple9_native.key.hw_payload);
      cmd->state.gfx.shaders[MESA_SHADER_TASK] = t;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = m;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f;
      CHECK(hk_apple9_select_native_mesh_route(cmd, g64, 0) ==
            HK_APPLE9_MESH_ROUTE_NATIVE);
      destroy_pair(dev, t, m, f);

      for (unsigned pbytes = 1024; pbytes <= 16368; pbytes = pbytes == 1024 ? 16368 : 16369) {
         compile_pair(dev, 32, pbytes, 0, 4, false, true, &t, &m, &f);
         CHECK(hk_api_shader_native(t)->variants[0].info.apple9_native.key.hw_payload);
         struct hk_api_shader *t2 = roundtrip_shader(dev, t);
         struct hk_api_shader *m2 = roundtrip_shader(dev, m);
         struct hk_api_shader *f2 = roundtrip_shader(dev, f);
         cmd->state.gfx.shaders[MESA_SHADER_TASK] = t2;
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = m2;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f2;
         CHECK(hk_apple9_select_native_mesh_route(cmd, g1024, 0) ==
               HK_APPLE9_MESH_ROUTE_NATIVE);
         check_vdm_recording(dev, t2, m2, f2, g1024, 0, 32,
                             pbytes == 1024 ? 2u : 16u);
         destroy_pair(dev, t2, m2, f2);
         destroy_pair(dev, t, m, f);
      }

      for (unsigned pbytes = 2048; pbytes <= 16384; pbytes *= 8) {
         compile_pair(dev, 32, pbytes, 0, 4, false, false, &t, &m, &f);
         CHECK(hk_api_shader_native(t)->variants[0].info.apple9_native.key.hw_payload);
         cmd->state.gfx.shaders[MESA_SHADER_TASK] = t;
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = m;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f;
         CHECK(hk_apple9_select_native_mesh_route(cmd, g1024, 0) ==
               HK_APPLE9_MESH_ROUTE_NATIVE);
         check_vdm_recording(dev, t, m, f, g1024, 0, 32, pbytes / 1024u);
         destroy_pair(dev, t, m, f);
      }

      /* Exact 16 KiB + 16 B child grid = 16400 B > 16384 B:
       * Single-parent runs Zero-CDM Native OBJECT via args.payload;
       * Multi-parent (groups=64) cascades to Task Bridge (NOT Emulated)! */
      {
         compile_pair(dev, 32, 16384, 0, 4, false, true, &t, &m, &f);
         CHECK(!hk_api_shader_native(t)->variants[0].info.apple9_native.key.hw_payload);
         CHECK(hk_api_shader_task_bridge(t) && hk_api_shader_task_bridge(m) &&
               hk_api_shader_task_bridge(f));
         struct hk_api_shader *t2 = roundtrip_shader(dev, t);
         struct hk_api_shader *m2 = roundtrip_shader(dev, m);
         struct hk_api_shader *f2 = roundtrip_shader(dev, f);
         cmd->state.gfx.shaders[MESA_SHADER_TASK] = t2;
         cmd->state.gfx.shaders[MESA_SHADER_MESH] = m2;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f2;
         CHECK(hk_apple9_select_native_mesh_route(cmd, g1, 0) ==
               HK_APPLE9_MESH_ROUTE_NATIVE);
         CHECK(hk_apple9_select_native_mesh_route(cmd, g64, 0) ==
               HK_APPLE9_MESH_ROUTE_TASK_BRIDGE);
         CHECK(hk_apple9_select_native_mesh_route(cmd, NULL, 0) ==
               HK_APPLE9_MESH_ROUTE_EMULATED);
         check_vdm_recording(dev, t2, m2, f2, g1, 0x1600000000ull, 32, 1u);
         check_bridge_prepare_and_record(dev, t2, m2, f2);
         struct blob m2_blob;
         blob_init(&m2_blob);
         CHECK(m2->vk.ops->serialize(&dev->vk, &m2->vk, &m2_blob));
         destroy_pair(dev, t2, m2, f2);
         destroy_pair(dev, t, m, f);
         /* Truncated bridge subtree & OOM cleanup with zero live pairs. */
         unsigned base_bos = live_bos;
         const size_t cuts[] = {m2_blob.size - 1, m2_blob.size - 8,
                                (m2_blob.size * 3) / 4, m2_blob.size / 2};
         for (unsigned ci = 0; ci < ARRAY_SIZE(cuts); ++ci) {
            struct blob_reader r;
            struct vk_shader *out_vk = NULL;
            blob_reader_init(&r, m2_blob.data, cuts[ci]);
            CHECK(hk_device_shader_ops.deserialize(&dev->vk, &r, 0, NULL,
                                                   &out_vk) ==
                  VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
            CHECK(!out_vk);
         }
         CHECK(live_bos == base_bos);
         check_dual_leaf_failure_cleanup(dev);
         blob_finish(&m2_blob);
      }

      /* Payload-free multi-parent native OBJECT -> MESH (T32, groups={4,3,2}). */
      compile_pair(dev, 32, 0, 128, 2, false, false, &t, &m, &f);
      nm = hk_api_shader_native(m);
      CHECK(nm && nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object);
      CHECK(!nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.reads_native_payload);
      CHECK(!nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.key.hw_payload);
      cmd->state.gfx.shaders[MESA_SHADER_TASK] = t;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = m;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f;
      const uint32_t g_multi[3] = {4, 3, 2};
      CHECK(hk_apple9_select_native_mesh_route(cmd, g_multi, 0) ==
            HK_APPLE9_MESH_ROUTE_NATIVE);
      check_vdm_recording(dev, t, m, f, g_multi, 0, 32, 1u);
      destroy_pair(dev, t, m, f);

      /* Fallback cases on Phase 2 kernel: payload > 16 KiB, invocations > 128,
       * and dynamic unbounded EmitMeshTasksEXT all fall back cleanly to Task Bridge. */
      compile_pair(dev, 16, 16388, 0, 2, false, false, &t, &m, &f);
      nm = hk_api_shader_native(m);
      CHECK(nm && !nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object &&
            nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      destroy_pair(dev, t, m, f);

      compile_pair(dev, 256, 64, 0, 2, false, false, &t, &m, &f);
      nm = hk_api_shader_native(m);
      CHECK(nm && !nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object &&
            nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      destroy_pair(dev, t, m, f);

      compile_pair(dev, 32, 64, 0, 2, true, false, &t, &m, &f);
      nm = hk_api_shader_native(m);
      CHECK(nm && !nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_object &&
            nm->variants[HK_MESH_VARIANT_MAIN].info.mesh.native_task_bridge);
      destroy_pair(dev, t, m, f);

      /* Public VK_EXT_shader_object binary compatibility across logical-off ->
       * logical-on opens sharing the same physical shaderBinaryUUID: serialize
       * an ordinary compute shader with physical-on / logical-off and import
       * it on logical-on. */
      {
         pdev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
         dev->dev.params.features = 0;
         dev->vk.robustness_state = vk_robustness_disabled;
         CHECK(hk_apple9_native_mesh_available(pdev) &&
               !hk_apple9_native_mesh_available_device(dev));
         nir_builder cb = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "cs_compat");
         cb.shader->info.workgroup_size[0] = 1;
         cb.shader->info.workgroup_size[1] = 1;
         cb.shader->info.workgroup_size[2] = 1;
         struct vk_shader_compile_info cinfo = {
            .stage = MESA_SHADER_COMPUTE,
            .nir = cb.shader,
            .robustness = &vk_robustness_disabled,
         };
         struct vk_shader *cs_vk = NULL;
         CHECK(hk_device_shader_ops.compile(&dev->vk, 1, &cinfo, NULL, NULL,
                                            NULL, &cs_vk) == VK_SUCCESS);
         struct blob cblob;
         blob_init(&cblob);
         CHECK(cs_vk->ops->serialize(&dev->vk, cs_vk, &cblob));
         hk_api_shader_destroy(&dev->vk, cs_vk, NULL);

          dev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
         CHECK(hk_apple9_native_mesh_available_device(dev));
         struct blob_reader cr;
         struct vk_shader *cs_imp = NULL;
         blob_reader_init(&cr, cblob.data, cblob.size);
         CHECK(hk_device_shader_ops.deserialize(&dev->vk, &cr, 0, NULL,
                                                &cs_imp) == VK_SUCCESS);
         CHECK(cs_imp && cs_imp->stage == MESA_SHADER_COMPUTE);
         hk_api_shader_destroy(&dev->vk, cs_imp, NULL);
         blob_finish(&cblob);

         /* Reverse direction (logical-on -> logical-off sharing the same
          * physical shaderBinaryUUID): export a Task + Mesh + FS set carrying
          * optional native + task-bridge leaves on logical-on, then import on
          * logical-off. Import MUST succeed with native leaves stripped and
          * zero BO leaks, while physical-off still fails closed. */
         unsigned base_bos = live_bos;
         compile_pair(dev, 32, 2048, 0, 4, false, false, &t, &m, &f);
         CHECK(hk_api_shader_native(t) && hk_api_shader_task_bridge(t));
         CHECK(hk_api_shader_native(m) && hk_api_shader_task_bridge(m));
         CHECK(hk_api_shader_native(f) && hk_api_shader_task_bridge(f));
         struct blob blobs[3];
         struct hk_api_shader *srcs[3] = {t, m, f};
         for (unsigned si = 0; si < 3; ++si) {
            blob_init(&blobs[si]);
            CHECK(srcs[si]->vk.ops->serialize(&dev->vk, &srcs[si]->vk,
                                              &blobs[si]));
         }
         destroy_pair(dev, t, m, f);

         dev->dev.params.features = 0;
         CHECK(hk_apple9_native_mesh_available(pdev) &&
               !hk_apple9_native_mesh_available_device(dev));
         for (unsigned si = 0; si < 3; ++si) {
            struct blob_reader r;
            struct vk_shader *imp = NULL;
            blob_reader_init(&r, blobs[si].data, blobs[si].size);
            CHECK(hk_device_shader_ops.deserialize(&dev->vk, &r, 0, NULL,
                                                   &imp) == VK_SUCCESS);
            CHECK(imp && r.current == r.end && !r.overrun);
            struct hk_api_shader *api = container_of(imp, struct hk_api_shader, vk);
            CHECK(!hk_api_shader_native(api) && !hk_api_shader_task_bridge(api));
            hk_api_shader_destroy(&dev->vk, imp, NULL);
         }
         CHECK(live_bos == base_bos);

         /* Physical-off (different shaderBinaryUUID) still rejects native leaves. */
         pdev->dev.params.features = 0;
         for (unsigned si = 0; si < 3; ++si) {
            struct blob_reader r;
            struct vk_shader *imp = NULL;
            blob_reader_init(&r, blobs[si].data, blobs[si].size);
            CHECK(hk_device_shader_ops.deserialize(&dev->vk, &r, 0, NULL,
                                                   &imp) ==
                  VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
            CHECK(!imp);
            blob_finish(&blobs[si]);
         }
         pdev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
         dev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;

         /* Retained non-native companion (apple9_fs_native_per_vertex) MUST
          * remain uploaded and executable after logical-on -> logical-off
          * binary import, even while optional native leaves are stripped. */
         {
            nir_builder fb = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "fs_pv_compat");
            fb.shader->info.internal = false;
            fb.shader->info.separate_shader = true;
            nir_def *bary = nir_load_barycentric_coord_pixel(
               &fb, 32, .interp_mode = INTERP_MODE_SMOOTH);
            nir_def *col = nir_pad_vec4(&fb, bary);
            nir_store_output(
               &fb, col, nir_imm_int(&fb, 0), .write_mask = 15,
               .src_type = nir_type_float32,
               .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
            nir_shader_gather_info(fb.shader, nir_shader_get_entrypoint(fb.shader));
            struct vk_shader_compile_info finfo = {
               .stage = MESA_SHADER_FRAGMENT,
               .nir = fb.shader,
               .robustness = &vk_robustness_disabled,
            };
            struct vk_shader *fs_vk = NULL;
            CHECK(hk_device_shader_ops.compile(&dev->vk, 1, &finfo, NULL, NULL,
                                               NULL, &fs_vk) == VK_SUCCESS);
            struct hk_api_shader *fs_src =
               container_of(fs_vk, struct hk_api_shader, vk);
            CHECK(fs_src->apple9_fs_native_per_vertex);
            CHECK(fs_src->apple9_fs_native_per_vertex->variants[0].code_size > 0);
            CHECK(fs_src->apple9_fs_native_per_vertex->variants[0].apple9_body);
            struct blob fblob;
            blob_init(&fblob);
            CHECK(fs_vk->ops->serialize(&dev->vk, fs_vk, &fblob));
            hk_api_shader_destroy(&dev->vk, fs_vk, NULL);

            dev->dev.params.features = 0;
            struct blob_reader fr;
            struct vk_shader *fs_imp_vk = NULL;
            blob_reader_init(&fr, fblob.data, fblob.size);
            CHECK(hk_device_shader_ops.deserialize(&dev->vk, &fr, 0, NULL,
                                                   &fs_imp_vk) == VK_SUCCESS);
            CHECK(fs_imp_vk && fr.current == fr.end && !fr.overrun);
            struct hk_api_shader *fs_imp =
               container_of(fs_imp_vk, struct hk_api_shader, vk);
            CHECK(fs_imp->variants[0].code_size > 0);
            CHECK(fs_imp->apple9_fs_native_per_vertex);
            CHECK(fs_imp->apple9_fs_native_per_vertex->variants[0].code_size > 0);
            CHECK(fs_imp->apple9_fs_native_per_vertex->variants[0].apple9_body);
            CHECK(!hk_api_shader_native(fs_imp));
            hk_api_shader_destroy(&dev->vk, fs_imp_vk, NULL);
            blob_finish(&fblob);
            dev->dev.params.features = DRM_ASAHI_FEATURE_NATIVE_MESH;
         }
         CHECK(live_bos == base_bos);
      }
   }

   free(cmd);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   hk_apple9_linked_entries_finish(dev);
   hk_apple9_fs_epilogs_finish(dev);
   simple_mtx_destroy(&dev->apple9_linked_entries.lock);
   simple_mtx_destroy(&dev->apple9_fs_epilogs.lock);
   hk_apple9_bodies_finish(dev);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_back_lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_lock);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   CHECK(live_bos == 0);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   printf("PASS (%s)\n", mode);
   return 0;
}
