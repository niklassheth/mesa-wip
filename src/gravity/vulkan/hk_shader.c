/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gravity/debug.h"
#include <unistd.h>

#include "hk_shader.h"
#include "hk_gpu_busy.h"
#include "util/u_queue.h"

#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"
#include "pipe/p_defines.h"
#include "agx_debug.h"
#include "hk_apple9_polygon_points.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/compiler/agx_apple9_mesh.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_format_convert.h"

#include "agx_bo.h"
#include "gravity/libcl/mesh.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "agx_timeline.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_serialize.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_util.h"
#include "vk_ycbcr_conversion.h"

#include "gravity/compiler/agx_compile.h"
#include "gravity/compiler/agx_nir.h"
#include "gravity/compiler/agx_nir_texture.h"
#include "gravity/lib/agx_abi.h"
#include "gravity/lib/agx_linker.h"
#include "gravity/lib/agx_tilebuffer.h"
#include "compiler/spirv/nir_spirv.h"

#include "util/u_atomic.h"
#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

#include "hk_shader_priv.h"

/* Code-generation environment controls key the shader caches: a warm cache
 * must not hand one setting's shaders to another. Diagnostics and known
 * runtime-only controls are excluded. Zero when none is set keeps existing
 * default caches valid.
 */
static uint64_t hk_compiler_env_hash_value;
static once_flag hk_compiler_env_hash_once = ONCE_FLAG_INIT;

/* Environment knobs whose value changes generated code: their values key the
 * shader caches. Keep this list next to the knobs; any other AGX_* or
 * HK_APPLE9_* variable (traces, dumps, statistics, runtime switches, or a
 * codegen knob missing here) keys the caches by name only, so output paths
 * never split them while an unlisted knob still does, conservatively.
 */
static const char *const hk_codegen_env[] = {
   "AGX_MESA_DEBUG",
   "AGX_SPILL_COALESCE",
   "AGX_APPLE9_NO_COMPARE_LITERAL",
   "AGX_APPLE9_NO_DESCRIPTOR_HOIST",
   "AGX_APPLE9_NO_POINTER_LOAD",
   "AGX_APPLE9_NO_PROMOTE",
   "AGX_APPLE9_NO_SAMPLER_HOIST",
   "AGX_APPLE9_NO_UNIFORM_MOVE",
   "HK_APPLE9_NATIVE_MESH",
   "HK_APPLE9_NATIVE_MESH_ATTRIBUTES",
   "HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING",
   "HK_APPLE9_NATIVE_MESH_PACKED_SHAPES",
   "HK_APPLE9_NATIVE_INDEX",
   "HK_APPLE9_NATIVE_STAGES",
   "HK_APPLE9_GS_MESH_PROBE",
   "HK_APPLE9_GS_NATIVE",
   "HK_APPLE9_TESS_MESH_PROBE",
   "HK_APPLE9_TESS_NATIVE",
};

/* These controls change diagnostics, memory backing, when a variant is
 * compiled, or command recording/submission. They do not change shader code.
 * Keep them available for runtime investigations without splitting caches. */
static const char *const hk_ignored_env[] = {
   "HK_APPLE9_DUMP_FRAGMENT_DIR",    "HK_APPLE9_SPARSE_SCRATCH",
   "HK_APPLE9_TESS_BATCH",           "HK_APPLE9_TESS_SCAN_BLOCK",
   "HK_APPLE9_FRAGMENT_BARRIER",     "HK_APPLE9_NO_TILING_ELISION",
   "HK_APPLE9_MESH_ROUTE_LOG",       "HK_APPLE9_GS_ROUTE_LOG",
   "HK_APPLE9_TESS_ROUTE_LOG",       "HK_APPLE9_TESS_FOLD",
   "HK_APPLE9_NATIVE_BLOCKS",        "HK_APPLE9_NATIVE_LANES",
   "HK_APPLE9_NATIVE_BATCH_LOG",     "HK_APPLE9_NATIVE_RENDER_PRIMITIVES",
   "AGX_VA_QUARANTINE_STATS",
};

static bool
hk_env_named(const char *var, const char *const *names, unsigned count)
{
   const size_t len = strcspn(var, "=");
   for (unsigned i = 0; i < count; ++i) {
      if (strlen(names[i]) == len && !strncmp(var, names[i], len))
         return true;
   }
   return false;
}

/* How an environment entry keys the caches: 0 not at all, 1 by name, 2 by
 * value. */
static unsigned
hk_env_key_kind(const char *var)
{
   if (strncmp(var, "AGX_", 4) && strncmp(var, "HK_APPLE9_", 10))
      return 0;
   if (agx_apple9_is_debug_env(var) ||
       hk_env_named(var, hk_ignored_env, ARRAY_SIZE(hk_ignored_env)))
      return 0;
   return hk_env_named(var, hk_codegen_env, ARRAY_SIZE(hk_codegen_env)) ? 2 : 1;
}

static int
hk_compare_env(const void *a, const void *b)
{
   return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Native ABI changes invalidate both device/compiler and graphics-state
 * caches. The source NIR already keys stage-specific output maps/capacities;
 * this keys the shared layout contract without struct padding. Bump the ABI
 * version when its layout algorithm or draw-slot interpretation changes. */
void
hk_hash_apple9_native_mesh_abi(struct mesa_blake3 *ctx)
{
   const struct agx_apple9_mesh_key k = AGX_APPLE9_MESH_KEY;
   uint64_t packed_shapes = hk_apple9_native_mesh_packed_shapes();
   const uint32_t abi[] = {
      AGX_APPLE9_MESH_ABI_VERSION,
      k.reserved_uniform_words,
      k.output_root,
      k.primitive_count_bits,
      k.payload_uniform,
      k.grid_root,
      k.grid_offset_B,
      k.physical_threads,
      AGX_APPLE9_MESH_DRAW_UNIFORM,
      AGX_APPLE9_MESH_ARGUMENT_WORDS,
      hk_apple9_native_mesh_attributes_enabled(),
      2, /* Explicit native pairing and on-demand deferred realizations. */
      packed_shapes,
      packed_shapes >> 32,
      hk_apple9_native_gs_enabled(),
      hk_apple9_native_tess_enabled(),
      hk_apple9_native_gs_pack_enabled() ? HK_APPLE9_NATIVE_GS_PACK : 1,
      hk_apple9_native_tes_pack_enabled() ? HK_APPLE9_NATIVE_GS_PACK : 1,
      1, /* Native GS runs every invocation of each input primitive. */
      1, /* P1 logical workgroups independent of physical output class. */
      2, /* Destination-addressed publication rounds beyond 32 entries. */
      hk_apple9_native_task_bridge_enabled(),
      1, /* Compute-task records and guarded native child grids. */
   };
   _mesa_blake3_update(ctx, abi, sizeof(abi));
}

static void
hk_compiler_env_hash_init(void)
{
   extern char **environ;
   unsigned count = 0;
   for (char **e = environ; e && *e; ++e)
      count += hk_env_key_kind(*e) != 0;
   if (!count)
      return;

   const char **vars = malloc(count * sizeof(*vars));
   if (!vars) {
      hk_compiler_env_hash_value = ~0ull;
      return;
   }
   unsigned n = 0;
   for (char **e = environ; e && *e && n < count; ++e) {
      if (hk_env_key_kind(*e))
         vars[n++] = *e;
   }
   qsort(vars, n, sizeof(*vars), hk_compare_env);

   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   for (unsigned i = 0; i < n; ++i) {
      /* Name-only knobs hash "NAME", value knobs "NAME=VALUE". */
      const size_t len = hk_env_key_kind(vars[i]) == 2 ? strlen(vars[i])
                                                       : strcspn(vars[i], "=");
      _mesa_blake3_update(&ctx, vars[i], len);
      _mesa_blake3_update(&ctx, "", 1);
   }
   if (debug_get_bool_option("HK_APPLE9_NATIVE_MESH", false))
      hk_hash_apple9_native_mesh_abi(&ctx);
   blake3_hash hash;
   _mesa_blake3_final(&ctx, hash);
   memcpy(&hk_compiler_env_hash_value, hash, sizeof(hk_compiler_env_hash_value));
   free(vars);
}

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev)
{
   call_once(&hk_compiler_env_hash_once, hk_compiler_env_hash_init);
   /* Soft faults follow a kernel parameter, so they can change between
    * boots with the same cache. */
   uint64_t flags = pdev->dev.debug |
                    ((uint64_t)pdev->apple9_layered_2d << 63) |
                    ((uint64_t)agx_has_soft_fault((struct agx_device *)&pdev->dev) << 62) |
                    ((uint64_t)pdev->apple9_d3d_mip_rounding << 61) |
                    ((uint64_t)hk_apple9_native_index_enabled(pdev) << 60) |
                    ((uint64_t)hk_apple9_native_stage_policy(pdev) << 58);
   return flags ^ hk_compiler_env_hash_value;
}

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &agx_nir_options;
}

static struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   const struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   return (struct spirv_to_nir_options){
      .ssbo_addr_format = hk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = hk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = hk_min_ssbo_alignment(&pdev->dev),
      .min_ubo_alignment = HK_MIN_UBO_ALIGNMENT,
   };
}

void
hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev, nir_shader *nir)
{
   /* Must lower before io to temps */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   } else if (nir->info.stage == MESA_SHADER_TASK) {
      /* Launching mesh workgroups ends the task workgroup. */
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }

   /* Unroll loops before lowering indirects via
    * nir_lower_io_vars_to_temporaries */
   UNUSED bool progress = false;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);

   /* Treat position, point size, clip/cull distances and tessellation levels
    * as invariant, as RADV does. Whether NIR fuses a multiply into an add
    * depends on the multiply's other users, so the same position math rounds
    * differently in shaders that also write other outputs from it. Blender
    * EEVEE's motion-vector prepass and its material pass then disagree on
    * depth by a few ULPs, and the material pass's EQUAL test drops most of
    * its pixels.
    */
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_propagate_invariant, true);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };

      nir_lower_sysvals_to_varyings(nir, &sysvals_opts);
   }

   NIR_PASS(_, nir, nir_lower_system_values);

   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Mesh and tessellation control outputs are shared by the workgroup or
    * patch: each invocation may write a different element, so copying whole
    * temporaries back at the end would overwrite the other invocations'
    * writes. Without temporaries, lower component accesses before I/O
    * lowering.
    */
   if (nir->info.stage == MESA_SHADER_MESH ||
       nir->info.stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   } else {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_shader_out);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);

   /* Extract immutable tables before agx_preprocess_nir lowers large local
    * arrays to scratch. The analysis requires explicit loads/stores rather
    * than copies. Keep the data attached to NIR through variant cloning.
    */
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   NIR_PASS(_, nir, agx_nir_lower_apple9_math);
   agx_preprocess_nir(nir);

   /* Constant-offset reads can fold away after extraction. Do not upload
    * or bind an unused blob: besides wasting storage, that would extend
    * every draw's root for a table the program never reads.
    */
   if (nir->constant_data_size) {
      bool used = false;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               used |= instr->type == nir_instr_type_intrinsic &&
                       nir_instr_as_intrinsic(instr)->intrinsic ==
                          nir_intrinsic_load_constant;
            }
         }
      }
      if (!used) {
         ralloc_free(nir->constant_data);
         nir->constant_data = NULL;
         nir->constant_data_size = 0;
      }
   }

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
}

static void
hk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   hk_preprocess_nir_internal(vk_pdev, nir);
   /* Only compute dispatches have a base workgroup. */
   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = nir->info.stage == MESA_SHADER_COMPUTE,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}


enum hk_feature_key
hk_make_feature_key(const struct vk_features *feats)
{
   return ((!feats || feats->customBorderColors) ? HK_FEAT_CUSTOM_BORDER : 0) |
          ((!feats || feats->minLod) ? HK_FEAT_MIN_LOD : 0) |
          ((!feats || feats->sparseResidencyImage2D) ? HK_FEAT_SPARSE_IMAGE : 0) |
          ((!feats || feats->sparseResidencyBuffer) ? HK_FEAT_STORE_BUFFER_RESIDENCY : 0) |
          ((!feats || (feats->descriptorBuffer && feats->sparseResidencyBuffer))
              ? HK_FEAT_CHECKED_BUF_STORES : 0) |
          ((!feats || feats->pipelineStatisticsQuery)
              ? HK_FEAT_PIPELINE_STATS : 0) |
          ((!feats || feats->primitivesGeneratedQuery)
              ? HK_FEAT_PRIMS_GENERATED : 0);
}

static void
hk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   struct hk_physical_device *pdev = container_of(device, struct hk_physical_device, vk);
   if (state) {
      struct hk_apple9_graphics_key key;
      hk_apple9_graphics_key_device(&key, state, pdev);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state &&
       (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT))) {
      const bool is_multiview = state->mv->view_mask != 0;
      _mesa_blake3_update(&blake3_ctx, &is_multiview, sizeof(is_multiview));
   }

   if ((stages & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) &&
       hk_apple9_native_mesh_available(pdev))
      hk_hash_apple9_native_mesh_abi(&blake3_ctx);

   enum hk_feature_key feature_key = hk_make_feature_key(features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}



void
hk_shader_destroy(struct hk_device *dev, struct hk_shader *s)
{
   free((void *)s->code_ptr);
   free((void *)s->data_ptr);
   if (s->apple9_body)
      agx_shader_body_unreference(&dev->dev, s->apple9_body);
   if (s->apple9_coefficients.size) {
      simple_mtx_lock(&dev->usc_heap_lock);
      hk_usc_heap_free(&dev->usc_heap, &s->apple9_coefficients);
      simple_mtx_unlock(&dev->usc_heap_lock);
   }
   free(s->apple9_launch);
   s->apple9_launch = NULL;
   if (s->apple9_stage_launch) {
      agx_apple9_stage_launch_cache_finish(s->apple9_stage_launch);
      free(s->apple9_stage_launch);
      s->apple9_stage_launch = NULL;
   }
}


void
hk_api_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                      const VkAllocationCallbacks *pAllocator)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   if (obj->apple9_native) {
      assert(!obj->apple9_native->apple9_native);
      hk_api_shader_destroy(vk_dev, &obj->apple9_native->vk, pAllocator);
   }

   if (obj->apple9_gs_polygon_points) {
      assert(!obj->apple9_gs_polygon_points->apple9_gs_polygon_points);
      hk_api_shader_destroy(vk_dev, &obj->apple9_gs_polygon_points->vk,
                            pAllocator);
   }

   hk_foreach_variant(obj, shader) {
      hk_shader_destroy(dev, shader);
   }

   vk_shader_free(&dev->vk, pAllocator, &obj->vk);
}


static VkResult
hk_compile_shaders_timed(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   agx_timeline_emit(100, 103, 0, shader_count, 0, 0, 0, 0, 0, 0);
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   nir_shader *shaders[shader_count];

   /* Lower shaders, notably lowering I/O. This is a prerequisite for
    * intershader optimization.
    */
   for (uint32_t i = 0; i < shader_count; i++) {
      /* Shader objects have no pipeline robustness, but still follow the
       * device's robustness features. The runtime passes them disabled.
       */
      if (infos[i].robustness == &vk_robustness_disabled)
         infos[i].robustness = &dev->vk.robustness_state;

      const struct vk_shader_compile_info *info = &infos[i];
      /* TODO: Multiview with ESO */
      const bool is_multiview = state && state->mv->view_mask != 0;
      enum hk_feature_key hk_features = hk_make_feature_key(features);
      nir_shader *nir = info->nir;

      hk_lower_nir(dev, nir, info->robustness, is_multiview,
                   info->set_layout_count, info->set_layouts, hk_features);

      if (nir->xfb_info) {
         /* Fold constant offset srcs for IO. */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         nir_io_add_intrinsic_xfb_info(nir);
      }

      shaders[i] = nir;
   }

   /* Mesh primitive outputs link to plain flat inputs. */
   bool mesh = false;
   for (uint32_t i = 0; i < shader_count; i++)
      mesh |= infos[i].stage == MESA_SHADER_MESH;
   if (!mesh) {
      nir_opt_varyings_bulk(shaders, shader_count, true, UINT32_MAX,
                            UINT32_MAX, hk_nir_opts, NULL);
   } else {
      /* Interpolation masks are otherwise gathered by the linker. */
      for (uint32_t i = 0; i < shader_count; i++)
         nir_shader_gather_info(shaders[i], nir_shader_get_entrypoint(shaders[i]));
   }

   if (shader_count && infos[0].stage != MESA_SHADER_COMPUTE) {
      return hk_apple9_compile_graphics(dev, shader_count, infos, state, NULL,
                                        NULL, pAllocator, shaders_out);
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      VkResult result =
         hk_apple9_compile_compute(dev, &infos[i], features, pAllocator,
                                   (struct hk_api_shader **)&shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            hk_api_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(infos[j].nir);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }
   }

   return VK_SUCCESS;
}

static VkResult
hk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   struct hk_cpu_compile_scope scope;
   hk_cpu_timing_compile_begin(&scope);
   VkResult result = hk_compile_shaders_timed(vk_dev, shader_count, infos,
                                              state, features, pAllocator,
                                              shaders_out);
   hk_cpu_timing_compile_end(&scope, HK_CPU_COMPILE_API);
   return result;
}

static VkResult
hk_deserialize_shader(struct hk_device *dev, struct blob_reader *blob,
                      struct hk_shader *shader)
{
   struct hk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));

   struct agx_shader_info b_info;
   blob_copy_bytes(blob, &b_info, sizeof(b_info));

   const uint32_t code_size = blob_read_uint32(blob);
   const uint32_t data_size = blob_read_uint32(blob);
   /* Validate payload lengths before any code/data allocation, including
    * nested native realizations and shader binaries outside the variant cache.
    * Subtract successively so even a 32-bit host cannot overflow the sum. */
   if (blob->overrun || (unsigned)info.stage >= MESA_SHADER_MESH_STAGES ||
       code_size > blob->end - blob->current ||
       data_size > (blob->end - blob->current) - code_size)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.info = b_info;
   shader->info = info;
   shader->code_size = code_size;
   shader->data_size = data_size;
   shader->b.info.binary_size = code_size;

   shader->code_ptr = malloc(MAX2(code_size, 1));
   if (shader->code_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->data_ptr = malloc(MAX2(data_size, 1));
   if (shader->data_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   blob_copy_bytes(blob, (void *)shader->code_ptr, shader->code_size);
   blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.binary = (void *)shader->code_ptr;
   return hk_upload_shader(dev, shader);
}

/* Version the container independently of the hardware programs it owns. */
#define HK_SHADER_CONTAINER_VERSION 0x484b000du

static VkResult
hk_deserialize_api_shader_inner(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out, bool allow_native)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   if (blob_read_uint32(blob) != HK_SHADER_CONTAINER_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   mesa_shader_stage stage = blob_read_uint8(blob);
   bool native_candidate = blob_read_uint8(blob);
   if (blob->overrun || stage >= MESA_SHADER_MESH_STAGES)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(stage);

   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, pAllocator, size);

   if (obj == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   obj->apple9_native_candidate = native_candidate;
   obj->apple9_native_cull_primitive = blob_read_uint8(blob);
   obj->apple9_native_leaf = blob_read_uint8(blob);
   obj->apple9_native_stage_mask = blob_read_uint8(blob);
   uint8_t stage_policy = hk_apple9_native_stage_policy(hk_device_physical(dev));
   uint8_t mask = obj->apple9_native_stage_mask;
   bool pair_stage =
      mask == 0 ||
      (mask == HK_APPLE9_NATIVE_GS_BIT &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY ||
        stage == MESA_SHADER_FRAGMENT)) ||
      (mask == HK_APPLE9_NATIVE_TES_BIT &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_CTRL ||
        stage == MESA_SHADER_TESS_EVAL || stage == MESA_SHADER_FRAGMENT));
   if (!pair_stage || (mask & ~stage_policy)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   blob_copy_bytes(blob, obj->apple9_native_pair_id,
                   sizeof(obj->apple9_native_pair_id));
   bool gs_leaf = (hk_apple9_native_gs_enabled() ||
                   (obj->apple9_native_stage_mask == HK_APPLE9_NATIVE_GS_BIT)) &&
                  (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY);
   bool tess_leaf =
      (hk_apple9_native_tess_enabled() ||
       (obj->apple9_native_stage_mask == HK_APPLE9_NATIVE_TES_BIT)) &&
      (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_CTRL ||
       stage == MESA_SHADER_TESS_EVAL);
   if (blob->overrun ||
       (obj->apple9_native_leaf && stage != MESA_SHADER_MESH &&
        stage != MESA_SHADER_FRAGMENT && !gs_leaf && !tess_leaf)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }

   hk_foreach_variant(obj, shader) {
      VkResult result = hk_deserialize_shader(dev, blob, shader);

      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }
   obj->writes_memory = blob_read_uint8(blob);
   if (obj->apple9_native_leaf && stage == MESA_SHADER_TESS_EVAL)
      blob_copy_bytes(blob, &obj->apple9_native_tess,
                      sizeof(obj->apple9_native_tess));
   obj->pre_raster_reads = blob_read_uint8(blob);
   obj->reads_draw_id = blob_read_uint8(blob);
   obj->zs_self_dep = blob_read_uint8(blob);
   blob_copy_bytes(blob, &obj->push_constant_mask,
                   sizeof(obj->push_constant_mask));
   bool has_native = blob_read_uint8(blob);
   if (blob->overrun ||
       (has_native && (!allow_native || obj->apple9_native_leaf)) ||
       (mask && !has_native && !obj->apple9_native_leaf)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   if (has_native) {
      struct vk_shader *native = NULL;
      VkResult result = hk_deserialize_api_shader_inner(
         vk_dev, blob, binary_version, pAllocator, &native, false);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return result;
      }
      obj->apple9_native = container_of(native, struct hk_api_shader, vk);
      if (native->stage != stage ||
          !obj->apple9_native->apple9_native_leaf ||
          obj->apple9_native->apple9_native_stage_mask != mask) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }

   bool has_gs_pp = blob_read_uint8(blob);
   if (blob->overrun ||
       (has_gs_pp && (!allow_native || stage != MESA_SHADER_GEOMETRY))) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   if (has_gs_pp) {
      struct vk_shader *gs_pp = NULL;
      VkResult result = hk_deserialize_api_shader_inner(
         vk_dev, blob, binary_version, pAllocator, &gs_pp, false);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return result;
      }
      obj->apple9_gs_polygon_points =
         container_of(gs_pp, struct hk_api_shader, vk);
      if (gs_pp->stage != MESA_SHADER_GEOMETRY) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

static VkResult
hk_deserialize_api_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out)
{
   return hk_deserialize_api_shader_inner(vk_dev, blob, binary_version,
                                           pAllocator, shader_out, true);
}

static void
hk_shader_serialize(struct vk_device *vk_dev, const struct hk_shader *shader,
                    struct blob *blob)
{
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   blob_write_bytes(blob, &shader->b.info, sizeof(shader->b.info));

   blob_write_uint32(blob, shader->code_size);
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->code_ptr, shader->code_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
}

static bool
hk_api_shader_serialize(struct vk_device *vk_dev,
                        const struct vk_shader *vk_shader, struct blob *blob)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   blob_write_uint32(blob, HK_SHADER_CONTAINER_VERSION);
   blob_write_uint8(blob, vk_shader->stage);
   blob_write_uint8(blob, p_atomic_read(&obj->apple9_native_candidate));
   blob_write_uint8(blob, obj->apple9_native_cull_primitive);
   blob_write_uint8(blob, obj->apple9_native_leaf);
   blob_write_uint8(blob, obj->apple9_native_stage_mask);
   blob_write_bytes(blob, obj->apple9_native_pair_id,
                    sizeof(obj->apple9_native_pair_id));

   hk_foreach_variant(obj, shader) {
      hk_shader_serialize(vk_dev, shader, blob);
   }
   /* Preserve the API shader's memory effects on a cache hit. In particular,
    * a read-only vertex shader must not turn its render into a memory writer
    * and prevent later renders from overlapping tiling with fragment work.
    */
   blob_write_uint8(blob, obj->writes_memory);
   if (obj->apple9_native_leaf && obj->vk.stage == MESA_SHADER_TESS_EVAL)
      blob_write_bytes(blob, &obj->apple9_native_tess,
                       sizeof(obj->apple9_native_tess));
   blob_write_uint8(blob, obj->pre_raster_reads);
   blob_write_uint8(blob, obj->reads_draw_id);
   blob_write_uint8(blob, obj->zs_self_dep);
   blob_write_bytes(blob, &obj->push_constant_mask,
                    sizeof(obj->push_constant_mask));

   blob_write_uint8(blob, obj->apple9_native != NULL);
   if (obj->apple9_native) {
      assert(!obj->apple9_native->apple9_native &&
             obj->apple9_native->apple9_native_leaf &&
             !obj->apple9_native_leaf);
      hk_api_shader_serialize(vk_dev, &obj->apple9_native->vk, blob);
   }

   blob_write_uint8(blob, obj->apple9_gs_polygon_points != NULL);
   if (obj->apple9_gs_polygon_points) {
      assert(obj->vk.stage == MESA_SHADER_GEOMETRY &&
             !obj->apple9_gs_polygon_points->apple9_gs_polygon_points);
      hk_api_shader_serialize(vk_dev, &obj->apple9_gs_polygon_points->vk, blob);
   }

   return !blob->out_of_memory;
}

static VkResult
hk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(obj->vk.stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(obj->vk.stage));
      VK_PRINT_STR(props->description, "%s shader",
                   _mesa_shader_stage_to_string(obj->vk.stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   assert(executable_index == 0);

   /* TODO: find a sane way to report multiple variants and have that play nice
    * with zink.
    */
   struct hk_shader *shader = hk_any_variant(obj);
   vk_add_agx2_stats(out, &shader->b.info.stats);
   return vk_outarray_status(&out);
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
hk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);
   bool incomplete_text = false;

   assert(executable_index == 0);

   /* TODO */
#if 0
   vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR, &out, ir) {
      VK_COPY_STR(ir->name, "AGX assembly");
      VK_COPY_STR(ir->description, "AGX assembly");
      if (!write_ir_text(ir, TODO))
         incomplete_text = true;
   }
#endif

   return incomplete_text ? VK_INCOMPLETE : vk_outarray_status(&out);
}

const struct vk_shader_ops hk_shader_ops = {
   .destroy = hk_api_shader_destroy,
   .serialize = hk_api_shader_serialize,
   .get_executable_properties = hk_shader_get_executable_properties,
   .get_executable_statistics = hk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      hk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops hk_device_shader_ops = {
   .get_nir_options = hk_get_nir_options,
   .get_spirv_options = hk_get_spirv_options,
   .preprocess_nir = hk_preprocess_nir,
   .hash_state = hk_hash_graphics_state,
   .compile = hk_compile_shaders,
   .deserialize = hk_deserialize_api_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = hk_cmd_bind_shaders,
};
