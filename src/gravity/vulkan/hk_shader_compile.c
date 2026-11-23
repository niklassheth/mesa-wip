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

DEBUG_GET_ONCE_OPTION(cached_hk_apple9_dump_fragment_dir,
                      "HK_APPLE9_DUMP_FRAGMENT_DIR", NULL)


/* Byte-identical Apple9 bodies share one code allocation and shader entry.
 * Pipelines and draw-time variants often compile to the same code: in the
 * CS2 capture, three in four bodies duplicate another. The table keeps the
 * most recently uploaded bodies referenced, so a body may outlive its last
 * shader until evicted.
 */
static uint32_t
hk_apple9_body_hash(const void *key)
{
   const struct hk_apple9_body *body = key;
   return _mesa_hash_data(body->hash, sizeof(body->hash));
}

static bool
hk_apple9_body_equal(const void *a, const void *b)
{
   const struct hk_apple9_body *x = a, *y = b;
   return x->size == y->size && !memcmp(x->hash, y->hash, sizeof(x->hash));
}

/* Heap and cache bookkeeping share apple9_bodies.lock. Body contents are
 * immutable before publication. Entry teardown releases its lock before this
 * callback, and BO release happens after dropping the body-cache lock. */
static void
hk_apple9_body_destroy(struct agx_device *adev, struct agx_shader_body *base)
{
   struct hk_device *dev = container_of(adev, struct hk_device, dev);
   struct hk_apple9_body *body =
      container_of(base, struct hk_apple9_body, base);
   struct hk_apple9_body_slab *slab = body->slab;
   bool release_bo = !slab;
   if (slab) {
      simple_mtx_lock(&dev->apple9_bodies.lock);
      util_vma_heap_free(
         &slab->heap,
         base->addr - slab->bo->va->addr + HK_APPLE9_BODY_ALIGNMENT,
         body->alloc_size);
      assert(slab->live);
      if (--slab->live == 0) {
         assert(!slab->cached);
         list_del(&slab->link);
         util_vma_heap_finish(&slab->heap);
         free(slab);
         release_bo = true;
      }
      simple_mtx_unlock(&dev->apple9_bodies.lock);
   }
   if (release_bo)
      agx_bo_unreference(adev, base->bo);
   free(body);
}

/* Offset zero is valid storage, but the VMA allocator uses it as failure.
 * Bias its offsets by one alignment unit and undo the bias for GPU addresses. */
bool
hk_apple9_body_alloc(struct hk_device *dev, struct hk_apple9_body *body)
{
   uint64_t offset = 0;
   struct agx_bo *bo = NULL;
   struct agx_bo *failed_bo = NULL;
   body->alloc_size = ALIGN_POT(body->size, HK_APPLE9_BODY_ALIGNMENT);
   if (body->alloc_size <= HK_APPLE9_BODY_SLAB_USABLE) {
      simple_mtx_lock(&dev->apple9_bodies.lock);
      list_for_each_entry(struct hk_apple9_body_slab, slab,
                          &dev->apple9_bodies.slabs, link) {
         offset = util_vma_heap_alloc(&slab->heap, body->alloc_size,
                                      HK_APPLE9_BODY_ALIGNMENT);
         if (offset) {
            body->slab = slab;
            break;
         }
      }
      if (!body->slab) {
         struct hk_apple9_body_slab *slab = calloc(1, sizeof(*slab));
         if (slab) {
            slab->bo =
               agx_bo_create(&dev->dev, HK_APPLE9_BODY_SLAB_SIZE, 0,
                             AGX_BO_EXEC | AGX_BO_LOW_VA, "Apple9 shader slab");
            /* Map once before another uploader can reserve a slice. */
            if (slab->bo && agx_bo_map(slab->bo)) {
               util_vma_heap_init(&slab->heap, HK_APPLE9_BODY_ALIGNMENT,
                                  slab->bo->size - HK_APPLE9_BODY_TAIL);
               slab->heap.alloc_high = false;
               offset = util_vma_heap_alloc(&slab->heap, body->alloc_size,
                                            HK_APPLE9_BODY_ALIGNMENT);
               assert(offset);
               list_addtail(&slab->link, &dev->apple9_bodies.slabs);
               body->slab = slab;
            } else {
               failed_bo = slab->bo;
               free(slab);
            }
         }
      }
      if (body->slab) {
         body->slab->live++;
         bo = body->slab->bo;
         offset -= HK_APPLE9_BODY_ALIGNMENT;
      }
      simple_mtx_unlock(&dev->apple9_bodies.lock);
      agx_bo_unreference(&dev->dev, failed_bo);
   } else {
      bo = agx_bo_create(&dev->dev, body->alloc_size, 0,
                         AGX_BO_EXEC | AGX_BO_LOW_VA, "Apple9 shader");
   }
   if (!bo)
      return false;
   body->base = (struct agx_shader_body){
      .bo = bo,
      .addr = bo->va->addr + offset,
      .refcnt = 1,
      .destroy = hk_apple9_body_destroy,
   };
   /* Keep the previous page-rounded admission policy as well as a separate
    * bound on actual backing retained by cached bodies. */
   body->cache_charge =
      body->slab ? ALIGN_POT(body->size, AIL_PAGESIZE) : bo->size;
   return true;
}

static void
hk_apple9_body_uncache(struct hk_device *dev, struct hk_apple9_body *body)
{
   dev->apple9_bodies.count--;
   dev->apple9_bodies.bytes -= body->cache_charge;
   if (!body->slab || --body->slab->cached == 0)
      dev->apple9_bodies.backing_bytes -= body->base.bo->size;
}

struct agx_shader_body *
hk_apple9_body_upload(struct hk_device *dev, const struct hk_shader *shader)
{
   uint64_t data_offset = ALIGN_POT((uint64_t)shader->code_size,
                                    HK_MIN_UBO_ALIGNMENT);
   uint64_t size = data_offset + shader->data_size;
   if (!size || size > SIZE_MAX - (HK_APPLE9_BODY_ALIGNMENT - 1))
      return NULL;
   struct hk_apple9_body key = {.size = size};
   blake3_hasher hash;
   blake3_hasher_init(&hash);
   blake3_hasher_update(&hash, &shader->code_size, sizeof(shader->code_size));
   blake3_hasher_update(&hash, shader->code_ptr, shader->code_size);
   blake3_hasher_update(&hash, shader->data_ptr, shader->data_size);
   blake3_hasher_finalize(&hash, key.hash, sizeof(key.hash));

   simple_mtx_lock(&dev->apple9_bodies.lock);
   if (!dev->apple9_bodies.ht) {
      dev->apple9_bodies.ht = _mesa_hash_table_create(
         NULL, hk_apple9_body_hash, hk_apple9_body_equal);
      list_inithead(&dev->apple9_bodies.lru);
   }
   struct hash_entry *entry =
      dev->apple9_bodies.ht ? _mesa_hash_table_search(dev->apple9_bodies.ht, &key)
                            : NULL;
   if (entry) {
      struct hk_apple9_body *body = entry->data;
      list_move_to(&body->link, &dev->apple9_bodies.lru);
      agx_shader_body_reference(&body->base);
      simple_mtx_unlock(&dev->apple9_bodies.lock);
      return &body->base;
   }
   simple_mtx_unlock(&dev->apple9_bodies.lock);

   struct hk_apple9_body *body = malloc(sizeof(*body));
   if (!body)
      return NULL;
   *body = key;
   if (!hk_apple9_body_alloc(dev, body)) {
      free(body);
      return NULL;
   }
   struct agx_bo *bo = body->base.bo;
   void *map = agx_bo_map(bo);
   if (!map) {
      agx_shader_body_unreference(&dev->dev, &body->base);
      return NULL;
   }
   uint64_t offset = body->base.addr - bo->va->addr;
   map = (char *)map + offset;
   memset(map, 0, body->alloc_size);
   memcpy(map, shader->code_ptr, shader->code_size);
   if (shader->data_size)
      memcpy((char *)map + data_offset,
             shader->data_ptr, shader->data_size);
   agx_bo_note_cpu_write(bo, offset, body->alloc_size);

   struct list_head evicted;
   list_inithead(&evicted);
   simple_mtx_lock(&dev->apple9_bodies.lock);
   if (dev->apple9_bodies.ht && bo->size <= HK_APPLE9_BODY_CACHE_BYTES &&
       !_mesa_hash_table_search(dev->apple9_bodies.ht, body) &&
       _mesa_hash_table_insert(dev->apple9_bodies.ht, body, body)) {
      agx_shader_body_reference(&body->base);
      list_add(&body->link, &dev->apple9_bodies.lru);
      dev->apple9_bodies.count++;
      dev->apple9_bodies.bytes += body->cache_charge;
      if (!body->slab || body->slab->cached++ == 0)
         dev->apple9_bodies.backing_bytes += bo->size;
      while (dev->apple9_bodies.count > HK_APPLE9_BODY_CACHE_SIZE ||
             dev->apple9_bodies.bytes > HK_APPLE9_BODY_CACHE_BYTES ||
             dev->apple9_bodies.backing_bytes > HK_APPLE9_BODY_CACHE_BYTES) {
         struct hk_apple9_body *old = list_last_entry(
            &dev->apple9_bodies.lru, struct hk_apple9_body, link);
         list_del(&old->link);
         _mesa_hash_table_remove_key(dev->apple9_bodies.ht, old);
         hk_apple9_body_uncache(dev, old);
         list_addtail(&old->link, &evicted);
      }
   }
   simple_mtx_unlock(&dev->apple9_bodies.lock);
   list_for_each_entry_safe(struct hk_apple9_body, old, &evicted, link) {
      list_del(&old->link);
      agx_shader_body_unreference(&dev->dev, &old->base);
   }
   /* A concurrent identical upload, oversized body or failed cache insertion
    * leaves the caller owning a valid body without a cache reference. */
   return &body->base;
}

struct agx_shader_body *
hk_apple9_body_alloc_raw(struct hk_device *dev, size_t size, void **map_out)
{
   if (!size || size > SIZE_MAX - (HK_APPLE9_BODY_ALIGNMENT - 1))
      return NULL;
   struct hk_apple9_body *body = calloc(1, sizeof(*body));
   if (!body)
      return NULL;
   body->size = size;
   if (!hk_apple9_body_alloc(dev, body)) {
      free(body);
      return NULL;
   }
   void *map = agx_bo_map(body->base.bo);
   if (!map) {
      agx_shader_body_unreference(&dev->dev, &body->base);
      return NULL;
   }
   uint64_t offset = body->base.addr - body->base.bo->va->addr;
   map = (char *)map + offset;
   memset(map, 0, body->alloc_size);
   *map_out = map;
   return &body->base;
}

void
hk_apple9_bodies_finish(struct hk_device *dev)
{
   if (!dev->apple9_bodies.ht)
      return;
   list_for_each_entry_safe(struct hk_apple9_body, body,
                            &dev->apple9_bodies.lru, link) {
      list_del(&body->link);
      hk_apple9_body_uncache(dev, body);
      agx_shader_body_unreference(&dev->dev, &body->base);
   }
   _mesa_hash_table_destroy(dev->apple9_bodies.ht, NULL);
   dev->apple9_bodies.ht = NULL;
   assert(!dev->apple9_bodies.count && !dev->apple9_bodies.bytes &&
          !dev->apple9_bodies.backing_bytes);
}

/* The native description of an uploaded G16 graphics stage. */
static void
hk_apple9_init_render_stage(struct hk_shader *s)
{
   bool fragment = s->info.stage == MESA_SHADER_FRAGMENT;
   const struct agx_shader_info *i = &s->b.info;
   struct agx_apple9_render_stage stage = {
      .program_id = s->apple9_program_id,
      .bo = s->bo,
      .body = s->apple9_body,
      .binary = s->code_ptr,
      .binary_size = i->binary_size,
      .preamble_offset = i->apple9_preamble_offset,
      .preamble_size = i->apple9_preamble_size,
      .publication_count = i->apple9_publication_count,
      .publication_count_valid = true,
      .native_mesh = s->info.apple9_native_stage == HK_APPLE9_NATIVE_MESH,
      .scratch_size = i->scratch_size,
      .atomic_frame_size = i->apple9_atomic_frame_size,
      .resource_count = i->apple9_resource_count,
      .resource_ssbo_mask = i->apple9_resource_ssbo_mask,
      .resource_write_mask = i->apple9_resource_write_mask,
      .position_components = fragment ? 0 : 4,
      .varying_components = i->apple9_varyings.count,
      .varyings = i->apple9_varyings,
      .apple9_linear_mask = i->apple9_linear_mask,
      .apple9_flat_mask = i->apple9_flat_mask,
      .apple9_reads_z = i->apple9_reads_z,
      .reads_primitive_id = i->apple9_reads_primitive_id,
      .reads_point_coord = i->apple9_reads_point_coord,
      .barycentric_mask = i->apple9_barycentric_mask,
      .per_vertex_count = i->apple9_per_vertex_count,
      .writes_point_size = i->apple9_writes_point_size,
      .writes_layer_viewport = i->writes_layer_viewport,
      .clip_distance_count = i->apple9_clip_distance_count,
      .render_targets = s->info.apple9_render_targets,
      .launch_cache = s->apple9_stage_launch,
      .texture_mask = i->apple9_texture_mask,
      .sampler_mask = i->apple9_sampler_mask,
      .image_mask = i->apple9_image_mask,
      .uses_texel_fetch = i->apple9_uses_texel_fetch,
      .reads_tile = i->apple9_reads_tile,
      .uses_discard = i->apple9_uses_discard,
      .writes_depth = i->depth_layout != FRAG_DEPTH_LAYOUT_UNCHANGED,
      .disable_tri_merging = i->disable_tri_merging,
      .pixel_interlock = i->apple9_pixel_interlock,
   };
   memcpy(stage.resource_binding, i->apple9_resource_binding,
          sizeof(stage.resource_binding));
   memcpy(stage.per_vertex, i->apple9_per_vertex, sizeof(stage.per_vertex));
   /* Every uploaded graphics stage has an id; without one, consecutive draws
    * silently stop reusing launches and coefficient tables. */
   assert(!s->bo || stage.program_id);
   s->apple9_stage = stage;
}

VkResult
hk_upload_shader(struct hk_device *dev, struct hk_shader *shader)
{
   bool native = shader->info.apple9_native_stage != HK_APPLE9_NATIVE_NONE;
   if ((shader->b.info.apple9_body_only || native) &&
       !(hk_apple9_native_mesh_available(hk_device_physical(dev)) &&
         shader->info.stage == (shader->info.apple9_native_stage ==
                                   HK_APPLE9_NATIVE_OBJECT
                                      ? MESA_SHADER_TASK : MESA_SHADER_MESH) &&
         hk_apple9_native_shader_valid(&shader->b.info,
                                       &shader->info.apple9_native,
                                       shader->info.apple9_native_stage)))
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 native stage requires its native launch contract");

   agx_apple9_dump_info(&shader->b.info);
   if (!shader->code_size ||
       (shader->info.stage == MESA_SHADER_COMPUTE &&
        agx_is_shader_empty(&shader->b)))
      return VK_SUCCESS;
   shader->apple9_body = hk_apple9_body_upload(dev, shader);
   if (!shader->apple9_body)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   shader->bo = shader->apple9_body->bo;
   if (shader->data_size)
      shader->data_addr = shader->apple9_body->addr +
                          ALIGN_POT((uint64_t)shader->code_size, HK_MIN_UBO_ALIGNMENT);
   hk_gpu_busy_shader(shader->apple9_body->addr, shader->code_ptr, shader->code_size,
                      shader->info.stage);
   if (shader->info.stage == MESA_SHADER_COMPUTE) {
      const struct agx_apple9_compute_profile *p = &shader->info.apple9;
      struct agx_apple9_launch_template *t = malloc(sizeof(*t));
      if (t && agx_apple9_compute_launch_template(
                  t, p, dev->dev.shader_base,
                  p->preamble_size ? shader->apple9_body->addr + p->preamble_offset
                                   : 0))
         shader->apple9_launch = t;
      else
         free(t);
   } else {
      shader->apple9_program_id =
         p_atomic_inc_return(&dev->apple9_last_program_id);
      shader->apple9_stage_launch =
         malloc(sizeof(*shader->apple9_stage_launch));
      if (shader->apple9_stage_launch)
         agx_apple9_stage_launch_cache_init(shader->apple9_stage_launch);
      hk_apple9_init_render_stage(shader);
      if (shader->info.stage == MESA_SHADER_FRAGMENT) {
         simple_mtx_lock(&dev->usc_heap_lock);
         bool allocated = hk_usc_heap_alloc(
            &dev->usc_heap,
            agx_apple9_coefficients_size(&shader->apple9_stage), 64,
            &shader->apple9_coefficients);
         simple_mtx_unlock(&dev->usc_heap_lock);
         if (!allocated)
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         agx_apple9_init_coefficients(
            &dev->dev, &shader->apple9_stage,
            (struct agx_ptr){.cpu = shader->apple9_coefficients.map,
                             .gpu = shader->apple9_coefficients.addr});
      }
   }
   return VK_SUCCESS;
}


/* Debug: HK_APPLE9_DUMP_FRAGMENT_DIR saves each draw-time fragment compile's
 * input (serialized NIR, varying layout, blend states) as
 * <binary hash>-<size>.a9frag, named like the GPU pass log's shader hashes.
 * agx_apple9_fragment_replay recompiles it on the host. */
#define HK_APPLE9_FRAGMENT_DUMP_MAGIC "A9FRAG01"

static bool
hk_apple9_fragment_dump_begin(const nir_shader *nir, struct blob *blob)
{
   if (!debug_get_option_cached_hk_apple9_dump_fragment_dir())
      return false;
   blob_init(blob);
   nir_serialize(blob, nir, false);
   if (blob->out_of_memory) {
      blob_finish(blob);
      return false;
   }
   return true;
}

static void
hk_apple9_fragment_dump_end(struct blob *nir_blob, bool compiled,
                            const struct agx_apple9_varying_layout *varyings,
                            const struct agx_apple9_blend *blend,
                            unsigned targets, const struct agx_shader_part *part)
{
   const char *dir = debug_get_option_cached_hk_apple9_dump_fragment_dir();
   if (compiled && dir) {
      const unsigned blends = MAX2(targets, 1);
      const uint32_t header[] = {
         sizeof(*varyings), sizeof(*blend), targets, blends, nir_blob->size,
      };
      char path[512];
      snprintf(path, sizeof(path), "%s/%08x-%u.a9frag", dir,
               _mesa_hash_data(part->binary, part->info.binary_size),
               part->info.binary_size);
      FILE *f = fopen(path, "wbx");
      if (f) {
         fwrite(HK_APPLE9_FRAGMENT_DUMP_MAGIC, 1, 8, f);
         fwrite(header, sizeof(header), 1, f);
         fwrite(varyings, sizeof(*varyings), 1, f);
         fwrite(blend, sizeof(*blend), blends, f);
         fwrite(nir_blob->data, 1, nir_blob->size, f);
         fclose(f);
      }
   }
   blob_finish(nir_blob);
}

static VkResult
hk_copy_shader_constant_data(struct hk_device *dev, struct hk_shader *shader,
                             const nir_shader *nir)
{
   if (nir->constant_data_size > 0) {
      uint64_t data_size = ALIGN_POT((uint64_t)nir->constant_data_size,
                                     HK_MIN_UBO_ALIGNMENT);
      /* The shader cache stores data lengths in 32 bits. */
      if (data_size > UINT32_MAX)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      void *data = malloc(data_size);
      if (data == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      memcpy(data, nir->constant_data, nir->constant_data_size);

      assert(nir->constant_data_size <= data_size);
      memset(data + nir->constant_data_size, 0,
             data_size - nir->constant_data_size);

      shader->data_ptr = data;
      shader->data_size = data_size;
   }

   return VK_SUCCESS;
}

VkResult
hk_apple9_compile_compute(struct hk_device *dev,
                          struct vk_shader_compile_info *info,
                          const struct vk_features *vk_features,
                          const VkAllocationCallbacks *pAllocator,
                          struct hk_api_shader **shader_out)
{
   enum hk_feature_key features = hk_make_feature_key(vk_features);

   /* We consume the NIR, regardless of success or failure */
   nir_shader *nir = info->nir;
   if (info->stage != MESA_SHADER_COMPUTE) {
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 compute compiler requires compute stage");
   }

   size_t size = sizeof(struct hk_api_shader) + sizeof(struct hk_shader);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, MESA_SHADER_COMPUTE,
                       pAllocator, size);
   if (obj == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   obj->pre_raster_reads = HK_READS_UNKNOWN;
   obj->reads_draw_id = hk_nir_reads_draw_id(nir);
   obj->push_constant_mask = hk_nir_push_constant_mask(nir);
   struct hk_shader *shader = hk_only_variant(obj);
#ifndef NDEBUG
   shader->debug_owner = obj;
#endif

   if (!nir->info.io_lowered) {
      hk_lower_nir(dev, nir, info->robustness, false, info->set_layout_count,
                   info->set_layouts, features);
   }

   if (!hk_apple9_texture_bindings(nir)) {
      ralloc_free(nir);
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 texture operation without a descriptor handle");
   }

   const char *reason = NULL;
   if (!agx_nir_lower_apple9_runtime_sampler_state(nir, &reason)) {
      ralloc_free(nir);
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT, "%s", reason);
   }

   unsigned root = 0;
   shader->info.set_count = info->set_layout_count;
   NIR_PASS(_, nir, hk_lower_uvs_index, MESA_SHADER_COMPUTE, 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, &root);

   simple_mtx_t *lock = NULL;
   if (agx_get_compiler_debug())
      lock = &hk_device_physical(dev)->debug_compile_lock;
   if (lock)
      simple_mtx_lock(lock);

   assert(nir->info.io_lowered);
   hk_assert_final_push_mask(nir, shader);
   /* A compute entrypoint without side effects has no GPU work, even
    * when loads and arithmetic remain, for example once optimization has
    * proved every store unreachable. Keep the common empty-shader
    * marker so cached and freshly compiled shaders agree.
    */
   bool empty = true;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_barrier &&
             !(nir_intrinsic_infos[intr->intrinsic].flags &
               NIR_INTRINSIC_CAN_ELIMINATE))
            empty = false;
      }
   }

   reason = NULL;
   if (empty) {
      shader->b.info.stats.instrs = 1;
   } else if (agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9,
                                      &reason)) {
      /* Compiled. */
   } else if (reason == agx_apple9_compute_without_effects) {
      /* Optimization removed every store: there is nothing to run. */
      memset(&shader->b, 0, sizeof(shader->b));
      memset(&shader->info.apple9, 0, sizeof(shader->info.apple9));
      shader->b.info.stats.instrs = 1;
   } else {
      if (agx_get_compiler_debug())
         fprintf(stderr, "Apple9 compute compilation failed: %s\n",
                 reason ?: "unsupported stage");
      if (lock)
         simple_mtx_unlock(lock);
      ralloc_free(nir);
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 shader: %s", reason ? reason : "unsupported stage");
   }

   if (lock)
      simple_mtx_unlock(lock);

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;
   hk_gpu_busy_shader_source(shader->code_ptr, shader->code_size, nir->info.name,
                             nir->info.source_blake3, MESA_SHADER_COMPUTE);

   shader->info.stage = MESA_SHADER_COMPUTE;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   shader->b.info.outputs = nir->info.outputs_written;

   VkResult result = hk_copy_shader_constant_data(dev, shader, nir);
   ralloc_free(nir);
   if (result == VK_SUCCESS)
      result = hk_upload_shader(dev, shader);
   if (result != VK_SUCCESS) {
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return result;
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = obj;
   return VK_SUCCESS;
}


static VkResult
hk_apple9_finish_shader(struct hk_device *dev, struct hk_shader *shader,
                        const nir_shader *nir)
{
   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   VkResult result = hk_copy_shader_constant_data(dev, shader, nir);
   return result == VK_SUCCESS ? hk_upload_shader(dev, shader) : result;
}

/* Compile a software pre-rasterization stage (vertex shader feeding a
 * geometry shader, or one of the geometry shader's compute variants) as an
 * Apple9 compute kernel. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_prerast(struct hk_device *dev, nir_shader *nir,
                          struct hk_shader *shader, unsigned local_size,
                          const char **reason)
{
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size_variable = false;
   nir->info.workgroup_size[0] = local_size;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->xfb_info = NULL;
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_dead_cf);
   NIR_PASS(_, nir, nir_opt_dce);

   hk_assert_final_push_mask(nir, shader);
   bool empty = true;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir))
      empty &= exec_list_is_empty(&block->instr_list);
   if (empty) {
      shader->b.info.stats.instrs = 1;
      return VK_SUCCESS;
   }

   if (!agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9, reason)) {
      /* Static GS topology can leave the main program with nothing to store:
       * like an empty program, there is nothing to run. */
      if (*reason == agx_apple9_compute_without_effects) {
         memset(&shader->b, 0, sizeof(shader->b));
         memset(&shader->info.apple9, 0, sizeof(shader->info.apple9));
         shader->b.info.stats.instrs = 1;
         *reason = NULL;
         return VK_SUCCESS;
      }
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   return hk_apple9_finish_shader(dev, shader, nir);
}

static bool
hk_apple9_lower_sw_vertex_zero_base(nir_builder *b, nir_intrinsic_instr *intr,
                                    UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_channel(b, nir_load_global_invocation_id(b, 32), 0));
   return true;
}

/* The vertex shader in front of a geometry shader pulls its attributes and
 * indices in software, writing outputs to the vertex output buffer.
 */
static VkResult
hk_apple9_compile_sw_vs(struct hk_device *dev, nir_shader *nir,
                        const struct agx_apple9_vertex_layout *layout,
                        struct hk_shader *shader, const char **reason)
{
   struct agx_apple9_vs_prolog_iface vs_prolog = {0};
   if (layout->use_prolog) {
      if (!agx_nir_lower_apple9_sw_vs_inputs_to_prolog(nir, layout,
                                                       &vs_prolog)) {
         *reason = "unsupported software vertex prolog input";
         return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      }
   } else if (!agx_nir_lower_apple9_vertex_inputs(nir, layout, true)) {
      *reason = "unsupported software vertex input";
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   uint64_t outputs = nir->info.outputs_written;
   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
   NIR_PASS(_, nir, poly_nir_lower_sw_vs);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            hk_apple9_lower_sw_vertex_zero_base, nir_metadata_control_flow,
            NULL);
   if (vs_prolog.uses_prolog) {
      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->info.workgroup_size_variable = false;
      nir->info.workgroup_size[0] = 64;
      nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
      nir->xfb_info = NULL;
      hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_opt_dead_cf);
      NIR_PASS(_, nir, nir_opt_dce);

      hk_assert_final_push_mask(nir, shader);
      bool empty = true;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir))
         empty &= exec_list_is_empty(&block->instr_list);
      if (empty) {
         shader->b.info.stats.instrs = 1;
         shader->b.info.outputs = outputs;
         return VK_SUCCESS;
      }

      if (!agx_compile_apple9_sw_vs_body(
             nir, layout->buffer_descriptor_ubo, layout->draw_params_ubo,
             0 /* vertex_params_ubo = root UBO 0 */,
             &shader->b, &shader->info.apple9, reason)) {
         if (*reason == agx_apple9_compute_without_effects) {
            memset(&shader->b, 0, sizeof(shader->b));
            memset(&shader->info.apple9, 0, sizeof(shader->info.apple9));
            shader->b.info.stats.instrs = 1;
            shader->b.info.outputs = outputs;
            *reason = NULL;
            return VK_SUCCESS;
         }
         return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      shader->b.info.apple9_vs_prolog = vs_prolog;
      shader->info.apple9_vertex_unbounded = layout->unbounded;
      shader->b.info.outputs = outputs;
      return hk_apple9_finish_shader(dev, shader, nir);
   }
   VkResult result = hk_apple9_compile_prerast(dev, nir, shader, 64, reason);
   shader->b.info.outputs = outputs;
   return result;
}

static void
hk_apple9_gather_tess_info(struct hk_shader *shader, const nir_shader *nir)
{
   shader->info.tess.info.ccw = nir->info.tess.ccw;
   shader->info.tess.info.points = nir->info.tess.point_mode;
   shader->info.tess.info.spacing = nir->info.tess.spacing;
   shader->info.tess.info.mode = nir->info.tess._primitive_mode;
}

/* Tessellation control shaders run as compute kernels with one workgroup per
 * patch and one thread per output vertex, writing the patch buffer read by
 * the tessellator. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_tcs(struct hk_device *dev, nir_shader *nir,
                      struct hk_shader *shader, const char **reason)
{
   hk_apple9_gather_tess_info(shader, nir);
   NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   shader->info.tess.tcs_output_patch_size = nir->info.tess.tcs_vertices_out;
   shader->info.tess.tcs_per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
   shader->info.tess.tcs_nr_patch_outputs =
      util_last_bit(nir->info.patch_outputs_written);
   shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
   return hk_apple9_compile_prerast(dev, nir, shader,
                                    nir->info.tess.tcs_vertices_out, reason);
}

/* Cull distances reach the fragment shader as one linear value per plane,
 * zero where the distance is negative, when the interface has room for them:
 * that needs no geometry stage. Interfaces at the publication limit instead
 * publish one flat word of negative-plane bits per vertex, and a geometry
 * stage combines it over each primitive's vertices (hk_handle_passthrough_gs);
 * points use it directly. Sets *cull_mask to the transport used.
 */
static bool
hk_apple9_compile_vertex_culls(nir_shader *nir,
                               const struct agx_apple9_vertex_layout *vertex,
                               struct agx_shader_part *out, bool *cull_mask,
                               const char **reason)
{
   struct agx_apple9_vertex_layout layout = *vertex;
   layout.cull_distance_mask = *cull_mask = false;
   if (!nir->info.cull_distance_array_size)
      return agx_compile_apple9_vertex_inputs(nir, &layout, out, reason);

   nir_shader *planes = nir_shader_clone(ralloc_parent(nir), nir);
   if (agx_compile_apple9_vertex_inputs(planes, &layout, out, NULL)) {
      nir_shader_replace(nir, planes);
      return true;
   }
   ralloc_free(planes);
   free(out->binary);
   memset(out, 0, sizeof(*out));
   layout.cull_distance_mask = *cull_mask = true;
   return agx_compile_apple9_vertex_inputs(nir, &layout, out, reason);
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_mesh_probe, "HK_APPLE9_TESS_MESH_PROBE", false)

bool
hk_apple9_compile_tes_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   nir_shader *mesh = hk_nir_tes_mesh(source, reason);
   if (!mesh)
      return false;
   hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
   bool compiled =
      hk_apple9_compile_native_mesh_program(mesh, key, out, info, reason);
   ralloc_free(mesh);
   return compiled;
}

static inline unsigned
hk_compile_per_vertex_copy_slot(uint8_t base, unsigned vertex)
{
   if (base == VARYING_SLOT_BFC0 && vertex == 2)
      return VARYING_SLOT_CLIP_VERTEX;
   return base + vertex;
}

static unsigned
hk_apple9_output_publications(const nir_shader *nir)
{
   uint8_t mask[64] = {0};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            unsigned loc = nir_intrinsic_io_semantics(intr).location;
            if (nir_src_is_const(intr->src[1]))
               loc += nir_src_as_uint(intr->src[1]);
            if (loc >= ARRAY_SIZE(mask) || !agx_apple9_varying_supported(loc))
               continue;
            mask[loc] |=
               nir_intrinsic_write_mask(intr) << nir_intrinsic_component(intr);
         }
      }
   }
   unsigned pubs = 4; /* VARYING_SLOT_POS */
   for (unsigned i = 0; i < ARRAY_SIZE(mask); ++i)
      pubs += util_bitcount(mask[i]);
   if (nir->info.outputs_written & VARYING_BIT_PSIZ)
      pubs += 1;
   if (nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT))
      pubs += 1;
   pubs += nir->info.clip_distance_array_size;
   if (nir->info.cull_distance_array_size)
      pubs += 1;
   return pubs;
}

static bool
hk_apple9_slot_fits(const nir_shader *nir, unsigned extra)
{
   return hk_apple9_output_publications(nir) + extra <=
          AGX_APPLE9_PUBLICATION_COUNT;
}

static bool
hk_apple9_psiz_fits(const nir_shader *nir)
{
   if (nir->info.outputs_written & VARYING_BIT_PSIZ)
      return true;
   return hk_apple9_slot_fits(nir, 1);
}

static bool
hk_compile_split_pp_face_prim(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_FOGC ||
       intr->src[0].ssa->num_components < 2)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *val = intr->src[0].ssa;
   nir_store_output(b, nir_channel(b, val, 0), nir_imm_int(b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_FOGC,
                                     .num_slots = 1});
   nir_store_output(b, nir_channel(b, val, 1), nir_imm_int(b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID,
                                     .num_slots = 1});
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
hk_apple9_lower_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_VIEWPORT)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   unsigned offset = hk_root_descriptor_offset(draw.producer_writes_viewport);
   nir_def *written = nir_ine_imm(
      b, nir_load_ubo(b, 1, 16, nir_imm_int(b, 0), nir_imm_int(b, offset),
                      .align_mul = 2, .range_base = offset, .range = 2),
      0);
   nir_def *zero = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
   nir_def *sel = nir_bcsel(b, written, &intr->def, zero);
   nir_def_rewrite_uses_after(&intr->def, sel);
   return true;
}

/* Evaluation shaders run as the hardware vertex program over the tessellated
 * domain, and as a software vertex program when a geometry shader or
 * transform feedback follows. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_tes(struct hk_device *dev, nir_shader *nir,
                      const struct agx_apple9_vertex_layout *vertex,
                      struct hk_api_shader *obj, VkShaderStageFlags next,
                      const char **reason)
{
   /* Compile an immutable diagnostic clone; discard all probe code. */
   if (debug_get_option_tess_mesh_probe()) {
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info body = {0};
      const char *why = NULL;
      const struct hk_apple9_graphics_key key = {.vertex = *vertex};
      bool compiled = hk_apple9_compile_tes_mesh_program(
         nir, &key, &part, &body, &why);
      fprintf(stderr, "HK_TESS_MESH_PROBE compiled=%u bytes=%u submitted=0 reason=%s\n",
              compiled, part.info.binary_size, why ? why : "supported");
      free(part.binary);
   }
   struct hk_shader *hw = &obj->variants[HK_VS_VARIANT_HW];
   struct hk_shader *sw = &obj->variants[HK_VS_VARIANT_SW];
   hk_apple9_gather_tess_info(hw, nir);
   hk_apple9_gather_tess_info(sw, nir);
   sw->info.set_count = hw->info.set_count;
   sw->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   sw->info.cull_distance_array_size = nir->info.cull_distance_array_size;

   /* The software variant feeds a geometry stage: the application's, or a
    * passthrough for transform feedback, cull distances, or dynamic raster
    * features. Only clone NIR when the software variant can be compiled. */
   nir_shader *clone =
      ((next & VK_SHADER_STAGE_GEOMETRY_BIT) || nir->xfb_info ||
       nir->info.cull_distance_array_size)
         ? nir_shader_clone(NULL, nir)
         : NULL;

   bool point_mode = nir->info.tess.point_mode;
   unsigned psiz_reserve =
      point_mode && !(nir->info.outputs_written & VARYING_BIT_PSIZ) ? 1 : 0;
   /* Always publish TES patch ID on the HW variant when it fits, so FS reads
    * VARYING_SLOT_PRIMITIVE_ID via flat varying without cross-stage recompile. */
   if (!(nir->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) &&
       hk_apple9_slot_fits(nir, 1 + psiz_reserve))
      hk_apple9_write_primitive_id(nir);

   struct agx_apple9_vertex_layout layout = *vertex;
   layout.outputs_flat |= VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT;
   NIR_PASS(_, nir, poly_nir_lower_tes, true);
   layout.rasterize_points = point_mode || hk_apple9_psiz_fits(nir);
   layout.ignore_point_size = !layout.rasterize_points;
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_TESS_EVAL);
   if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
      nir_print_shader(nir, stderr);
   bool cull_mask;
   hk_assert_final_push_mask(nir, hw);
   if (!hk_apple9_compile_vertex_culls(nir, &layout, &hw->b, &cull_mask,
                                       reason)) {
      ralloc_free(clone);
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   hw->info.cull_mask = sw->info.cull_mask = cull_mask;

   VkResult result = VK_SUCCESS;
   if ((next & VK_SHADER_STAGE_GEOMETRY_BIT) ||
       (clone && clone->xfb_info) || cull_mask) {
      assert(clone);
      if (clone->xfb_info) {
         memcpy(&sw->info.xfb_info, clone->xfb_info,
                nir_xfb_info_size(clone->xfb_info->output_count));
         typed_memcpy(sw->info.xfb_stride, clone->info.xfb_stride, 4);
      }
      NIR_PASS(_, clone, poly_nir_lower_tes, false);
      uint64_t outputs = clone->info.outputs_written;
      NIR_PASS(_, clone, poly_nir_lower_vs_before_gs);
      result = hk_apple9_compile_prerast(dev, clone, sw, 64, reason);
      sw->b.info.outputs = outputs;
   }
   ralloc_free(clone);
   if (result != VK_SUCCESS)
      return result;
   return hk_apple9_finish_shader(dev, hw, nir);
}

/* Application geometry shaders emitting triangle strips under polygon mode
 * POINT: every output store is tapped into the current vertex, the strip's two
 * previous vertices are kept, and each completed triangle is drawn as points
 * by hk_apple9_pp_emit_triangle instead of being emitted.
 */
struct hk_gs_pp {
   /* Per slot (clip/cull distances per element), the strip's two previous
    * vertices and the current one, as 32-bit words. */
   nir_variable *v[3][VARYING_SLOT_MAX + 8];
   nir_def *odd;
};

static unsigned
hk_gs_pp_key(unsigned location, unsigned offset)
{
   return location == VARYING_SLOT_CLIP_DIST0 ? VARYING_SLOT_MAX + offset
                                              : location;
}

static nir_def *
hk_gs_pp_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
              unsigned offset, unsigned component, unsigned comps,
              nir_alu_type type)
{
   struct hk_gs_pp *pp = data;
   unsigned k = hk_gs_pp_key(location, offset);
   /* Odd strip triangles are wound (i + 1, i, i + 2). */
   nir_def *value;
   if (vertex == 2) {
      value = nir_load_var(b, pp->v[2][k]);
   } else {
      nir_def *a = nir_load_var(b, pp->v[vertex][k]);
      nir_def *c = nir_load_var(b, pp->v[vertex ^ 1][k]);
      value = nir_bcsel(b, pp->odd, c, a);
   }
   return nir_channels(b, value, BITFIELD_RANGE(component, comps));
}

/* Whether a geometry shader's triangle strips can be drawn as polygon
 * points: constant-offset 32-bit stream-0 outputs and an output count within
 * the geometry shader's vertex limit. */
static bool
hk_apple9_gs_polygon_points_ok(const nir_shader *gs)
{
   if (gs->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP)
      return false;
   unsigned per_triangle = 3 + 3 + gs->info.clip_distance_array_size;
   unsigned triangles =
      gs->info.gs.vertices_out > 2 ? gs->info.gs.vertices_out - 2 : 0;
   if (triangles * per_triangle > 1024)
      return false;
   nir_foreach_function_impl(impl, gs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            if (!nir_src_is_const(intr->src[1]) ||
                nir_src_bit_size(intr->src[0]) != 32 ||
                nir_intrinsic_io_semantics(intr).gs_streams ||
                nir_intrinsic_io_semantics(intr).location >= VARYING_SLOT_MAX)
               return false;
         }
      }
   }
   return true;
}

static void
hk_apple9_lower_gs_polygon_points(nir_shader *gs, struct hk_pp_info info,
                                  bool point_size)
{
   assert(hk_apple9_gs_polygon_points_ok(gs));
   unsigned per_triangle = 3 + 3 + info.clip;
   unsigned triangles =
      gs->info.gs.vertices_out > 2 ? gs->info.gs.vertices_out - 2 : 0;

   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   unsigned components[VARYING_SLOT_MAX] = {0};
   uint64_t written = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location;
         written |= BITFIELD64_BIT(loc);
         components[loc] = MAX2(components[loc],
                                nir_intrinsic_component(intr) +
                                   util_last_bit(nir_intrinsic_write_mask(intr)));
      }
   }

   struct hk_gs_pp pp = {0};
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   for (unsigned k = 0; k < VARYING_SLOT_MAX + 8; ++k) {
      bool used = k < VARYING_SLOT_MAX
                     ? (written & BITFIELD64_BIT(k)) && k != VARYING_SLOT_CLIP_DIST0
                     : (written & VARYING_BIT_CLIP_DIST0) &&
                          k - VARYING_SLOT_MAX < DIV_ROUND_UP(info.clip + info.cull, 4);
      if (!used)
         continue;
      for (unsigned i = 0; i < 3; ++i) {
         pp.v[i][k] = nir_local_variable_create(impl, glsl_uvec4_type(), NULL);
         nir_store_var(&b, pp.v[i][k], nir_imm_ivec4(&b, 0, 0, 0, 0), 0xf);
      }
   }
   nir_variable *count =
      nir_local_variable_create(impl, glsl_uint_type(), "pp_strip");
   nir_store_var(&b, count, nir_imm_int(&b, 0), 1);

   info.outputs = written &
      ~(VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
        VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 |
        VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_BOUNDING_BOX0 |
        VARYING_BIT_BOUNDING_BOX1);
   /* Without polygonModePointSize, polygon points have size 1. */
   if (!point_size)
      info.outputs &= ~VARYING_BIT_PSIZ;
   info.output_components = components;

   /* Collect first: the triangle emission adds control flow and its own
    * emit_vertex intrinsics. */
   struct util_dynarray todo;
   util_dynarray_init(&todo, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (op == nir_intrinsic_store_output || op == nir_intrinsic_emit_vertex ||
             op == nir_intrinsic_end_primitive)
            util_dynarray_append(&todo, nir_instr_as_intrinsic(instr));
      }
   }
   util_dynarray_foreach(&todo, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      nir_instr *instr = &intr->instr;
      b.cursor = nir_before_instr(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_store_output: {
         unsigned k = hk_gs_pp_key(nir_intrinsic_io_semantics(intr).location,
                                   nir_src_as_uint(intr->src[1]));
         if (!pp.v[2][k]) {
            nir_instr_remove(instr);
            break;
         }
         unsigned first = nir_intrinsic_component(intr);
         nir_def *value = nir_load_var(&b, pp.v[2][k]);
         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            value = nir_vector_insert_imm(
               &b, value, nir_channel(&b, intr->src[0].ssa, c), first + c);
         }
         nir_store_var(&b, pp.v[2][k], value, 0xf);
         nir_instr_remove(instr);
         break;
      }
      case nir_intrinsic_end_primitive:
         nir_store_var(&b, count, nir_imm_int(&b, 0), 1);
         nir_instr_remove(instr);
         break;
      case nir_intrinsic_emit_vertex: {
         nir_def *n = nir_load_var(&b, count);
         nir_push_if(&b, nir_uge_imm(&b, n, 2));
         {
            pp.odd = nir_ine_imm(&b, nir_iand_imm(&b, n, 1), 0);
            /* Strip triangle i provokes from vertex i, or i + 2. */
            nir_def *last = nir_ine_imm(&b, nir_load_provoking_last(&b), 0);
            nir_def *provoking = nir_bcsel(
               &b, last, nir_imm_int(&b, 2),
               nir_bcsel(&b, pp.odd, nir_imm_int(&b, 1), nir_imm_int(&b, 0)));
            nir_def *prim_id;
            if (written & VARYING_BIT_PRIMITIVE_ID) {
               nir_def *first_id = nir_channel(
                  &b, nir_load_var(&b, pp.v[0][VARYING_SLOT_PRIMITIVE_ID]), 0);
               nir_def *last_id = nir_channel(
                  &b, nir_load_var(&b, pp.v[2][VARYING_SLOT_PRIMITIVE_ID]), 0);
               prim_id = nir_bcsel(&b, last, last_id, first_id);
            } else {
               prim_id = nir_load_primitive_id(&b);
            }
            hk_apple9_pp_emit_triangle(&b, &info, hk_gs_pp_load, &pp,
                                       provoking, prim_id, NULL);
         }
         nir_pop_if(&b, NULL);
         for (unsigned k = 0; k < VARYING_SLOT_MAX + 8; ++k) {
            if (!pp.v[2][k])
               continue;
            nir_store_var(&b, pp.v[0][k], nir_load_var(&b, pp.v[1][k]), 0xf);
            nir_store_var(&b, pp.v[1][k], nir_load_var(&b, pp.v[2][k]), 0xf);
         }
         nir_store_var(&b, count, nir_iadd_imm(&b, n, 1), 1);
         nir_instr_remove(instr);
         break;
      }
      default:
         break;
      }
   }
   util_dynarray_fini(&todo);

   if (info.face_slot == VARYING_SLOT_FOGC) {
      nir_shader_intrinsics_pass(gs, hk_compile_split_pp_face_prim,
                                 nir_metadata_control_flow, NULL);
   }
   gs->info.gs.output_primitive = MESA_PRIM_POINTS;
   gs->info.gs.vertices_out = MAX2(triangles * per_triangle, 1);
   gs->info.outputs_written &= ~VARYING_BIT_PRIMITIVE_ID;
   if (!point_size)
      gs->info.outputs_written &= ~VARYING_BIT_PSIZ;
   if (info.face_slot) {
      gs->info.outputs_written |= BITFIELD64_BIT(info.face_slot);
      if (info.face_slot == VARYING_SLOT_FOGC)
         gs->info.outputs_written |= VARYING_BIT_PRIMITIVE_ID;
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
}

/* Geometry shaders run as compute kernels writing an output vertex buffer,
 * followed by an ordinary hardware vertex program that rasterizes it.
 * Does not take ownership of nir.
 */
DEBUG_GET_ONCE_BOOL_OPTION(gs_mesh_probe, "HK_APPLE9_GS_MESH_PROBE", false)

bool
hk_apple9_compile_gs_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   nir_shader *mesh = hk_nir_gs_mesh(source, reason);
   if (!mesh)
      return false;
   hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
   bool compiled = hk_apple9_compile_native_mesh_program(mesh, key, out, info, reason);
   ralloc_free(mesh);
   return compiled;
}

static VkResult
hk_apple9_compile_gs(struct hk_device *dev, nir_shader *nir,
                     const struct agx_apple9_vertex_layout *vertex,
                     bool cull_mask, struct hk_api_shader *obj,
                     const char **reason)
{
   /* This diagnostic probe does not select a submission path. The separate
    * default-off GS option and draw predicate own native routing. */
   if (debug_get_option_gs_mesh_probe()) {
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info body = {0};
      const char *why = NULL;
      const struct hk_apple9_graphics_key key = {.vertex = *vertex};
      bool compiled = hk_apple9_compile_gs_mesh_program(
         nir, &key, &part, &body, &why);
      fprintf(stderr, "HK_GS_MESH_PROBE compiled=%u bytes=%u submitted=0 reason=%s\n",
              compiled, part.info.binary_size, why ? why : "supported");
      free(part.binary);
   }
   struct hk_shader *rast_shader = &obj->variants[HK_GS_VARIANT_RAST];
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info = {0};
   if (cull_mask)
      NIR_PASS(_, nir, poly_nir_lower_gs_cull_mask, &count, &rast, &pre, &info);
   else
      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre, &info);

   hk_foreach_variant(obj, variant) {
      variant->info.gs = info;
      variant->info.set_count = rast_shader->info.set_count;
   }

   VkResult result =
      hk_apple9_compile_prerast(dev, nir, hk_main_gs_variant(obj), 64, reason);
   if (result == VK_SUCCESS && count)
      result = hk_apple9_compile_prerast(dev, count, hk_count_gs_variant(obj),
                                         64, reason);
   if (result == VK_SUCCESS && pre)
      result = hk_apple9_compile_prerast(dev, pre, hk_pre_gs_variant(obj), 1,
                                         reason);
   if (result == VK_SUCCESS) {
      struct agx_apple9_vertex_layout layout = *vertex;
      layout.outputs_flat |=
         VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT |
         VARYING_BIT_FOGC | VARYING_BIT_TEX6;
      layout.outputs_linear |= VARYING_BIT_TEX7;
      layout.cull_distance_mask = cull_mask;
      layout.rasterize_points = info.mode == MESA_PRIM_POINTS;
      layout.ignore_point_size = !layout.rasterize_points;
      hk_apple9_lower_prerast_sysvals(rast, MESA_SHADER_GEOMETRY);
      if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
         nir_print_shader(rast, stderr);
      hk_assert_final_push_mask(rast, rast_shader);
      if (agx_compile_apple9_vertex_inputs(rast, &layout, &rast_shader->b,
                                           reason))
         result = hk_apple9_finish_shader(dev, rast_shader, rast);
      else
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return result;
}

/* Mesh shaders run as compute kernels writing records, which a generated
 * vertex shader rasterizes.
 */
static VkResult
hk_apple9_compile_mesh(struct hk_device *dev, nir_shader *nir,
                       const struct agx_apple9_vertex_layout *vertex,
                       const struct hk_per_vertex_copies *copies,
                       struct hk_apple9_stipple_slots stipple,
                       const struct hk_pp_info *polygon_points,
                       bool point_size, struct hk_api_shader *obj,
                       const char **reason)
{
   struct hk_shader *main = &obj->variants[HK_MESH_VARIANT_MAIN];
   struct hk_shader *rast = &obj->variants[HK_MESH_VARIANT_RAST];
   struct hk_mesh_info info;
   hk_mesh_gather_info(nir, &info);
   info.polygon_points = polygon_points != NULL;

   hk_foreach_variant(obj, variant) {
      variant->info.mesh = info;
      variant->info.set_count = main->info.set_count;
   }

   nir_shader *rast_nir = hk_nir_mesh_rast(nir, &info, copies, polygon_points,
                                           point_size);

   hk_nir_lower_mesh_main(nir, &info);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   VkResult result = VK_SUCCESS;
   hk_assert_final_push_mask(nir, main);
   if (!agx_compile_apple9_tiny(nir, &main->b, &main->info.apple9, reason))
      result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   else
      result = hk_apple9_finish_shader(dev, main, nir);

   if (result == VK_SUCCESS) {
      struct agx_apple9_vertex_layout layout = *vertex;
      layout.outputs_flat |=
         VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT |
         VARYING_BIT_FOGC | VARYING_BIT_TEX6;
      layout.outputs_linear |= VARYING_BIT_TEX7;
      layout.rasterize_points =
         info.prim == MESA_PRIM_POINTS || info.polygon_points;
      layout.ignore_point_size = !layout.rasterize_points;
      if (stipple.start)
         hk_apple9_write_line_stipple_start(rast_nir, stipple);
      hk_apple9_lower_prerast_sysvals(rast_nir, MESA_SHADER_VERTEX);
      /* Always map depth dynamically via root.draw.clip_z_coeff so emulated
       * mesh never keys on VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE. Polygon-point
       * mesh-rast applies clip_z_coeff inside hk_mesh_pp_load before clipping. */
      if (!info.polygon_points) {
         bool dynamic_clip = true;
         NIR_PASS(_, rast_nir, nir_lower_clip_halfz_dynamic);
         NIR_PASS(_, rast_nir, nir_shader_intrinsics_pass,
                  hk_apple9_lower_clip_z_coeff, nir_metadata_control_flow,
                  &dynamic_clip);
      }
      if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
         nir_print_shader(rast_nir, stderr);
      hk_assert_final_push_mask(rast_nir, rast);
      if (agx_compile_apple9_vertex_inputs(rast_nir, &layout, &rast->b, reason))
         result = hk_apple9_finish_shader(dev, rast, rast_nir);
      else
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   ralloc_free(rast_nir);
   return result;
}

/* Compile the native program from the same graphics key used by the linked
 * pipeline. Keep key translation with the compilation so hk tests exercise
 * the production call boundary before device upload. */
static struct agx_apple9_native_mesh_options
hk_apple9_native_mesh_options(nir_shader *nir,
                              const struct hk_apple9_graphics_key *key)
{
   bool rounds = hk_apple9_native_mesh_round_shape_eligible(nir);
   bool packed = (key->native_index || hk_apple9_native_mesh_enabled()) &&
                 hk_apple9_native_mesh_packed_shape_eligible(nir);
   return (struct agx_apple9_native_mesh_options){
      .physical_threads = rounds ? AGX_APPLE9_NATIVE_SERIAL_THREADS
                                 : hk_apple9_native_mesh_threads(nir->info.workgroup_size),
      .clip_halfz = !key->clip_negative_one_to_one_dynamic &&
                    key->clip_negative_one_to_one,
      .dynamic_clip_z = key->clip_negative_one_to_one_dynamic,
      .clip_z_ubo = 0,
      .clip_z_offset_B = hk_root_descriptor_offset(draw.clip_z_coeff),
      .attributes = (rounds || !packed) && hk_apple9_native_mesh_attributes_enabled(),
      .primitive_broadcast = !rounds && !packed && hk_apple9_native_mesh_primitive_lowering_enabled(),
      .primitive_vertices = rounds && hk_apple9_native_mesh_primitive_lowering_enabled(),
      .packed_index_publication = rounds || packed,
      .publication_rounds = rounds,
      .link_clip_distances = key->fs_reads_clip_distance,
      .layer_viewport = hk_apple9_native_mesh_primitive_lowering_enabled(),
      .outputs_flat = key->vertex.outputs_flat,
      .outputs_linear = key->vertex.outputs_linear,
   };
}

/* Native incompatibility is a capability result, not a Vulkan error. Run the
 * actual lowering on a clone: some rejection sites occur after NIR mutation.
 * This runs at pipeline/library/shader creation even when final compilation
 * is deferred, so a shader's shape cannot first fail at EndCommandBuffer.
 */
VkResult
hk_apple9_native_mesh_check(nir_shader *nir,
                           const struct hk_apple9_graphics_key *key,
                           bool *supported, const char **reason)
{
   *supported = false;
   if (reason)
      *reason = NULL;
   if ((!hk_apple9_native_mesh_enabled() && !key->native_index) || key->line_stipple)
      return VK_SUCCESS;

   nir_shader *probe = nir_shader_clone(NULL, nir);
   if (!probe)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (key->native_task_bridge)
      hk_nir_lower_native_mesh_task(probe);
   if (!hk_apple9_native_mesh_enabled() &&
       !hk_apple9_native_mesh_packed_shape_eligible(probe)) {
      ralloc_free(probe);
      return VK_SUCCESS;
   }

   struct agx_apple9_native_mesh_options options =
      hk_apple9_native_mesh_options(probe, key);
   if (!options.physical_threads) {
      if (reason)
         *reason = "native mesh supports 1..128 logical invocations";
      ralloc_free(probe);
      return VK_SUCCESS;
   }
   struct agx_apple9_mesh_body_info info;
   *supported = agx_nir_lower_apple9_native_mesh(probe, true, &options,
                                                &info, reason);
   ralloc_free(probe);
   return VK_SUCCESS;
}

bool
hk_apple9_compile_native_mesh_program(
   nir_shader *nir, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   if (key->native_task_bridge)
      hk_nir_lower_native_mesh_task(nir);
   const struct agx_apple9_native_mesh_options options =
      hk_apple9_native_mesh_options(nir, key);
   if (!options.physical_threads) {
      if (reason)
         *reason = "native mesh supports 1..128 logical invocations";
      return false;
   }
   memset(out, 0, sizeof(*out));
   if (!agx_nir_lower_apple9_native_mesh(
      nir, hk_apple9_native_mesh_enabled() ||
           (key->native_index && options.packed_index_publication),
      &options, info, reason))
      return false;
   if (info->primitive_vertices)
      hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   return agx_compile_apple9_native_mesh_lowered(nir, &options, out, info, reason);
}

/* Research varyings use the existing mesh publication model (T32 for shapes
 * beyond V3/P1). Keep the stage's explicit declaration-class selection; this
 * does not extend production Position-only qualification. */
static bool
hk_apple9_native_stage_varyings(const nir_shader *mesh, bool production)
{
   return !production && hk_apple9_native_mesh_enabled() &&
          hk_apple9_native_mesh_attributes_enabled() &&
          (mesh->info.outputs_written & (UINT64_MAX << VARYING_SLOT_VAR0)) &&
          hk_apple9_packed_mesh_class_selected(
             hk_apple9_native_mesh_packed_shapes(),
             hk_apple9_native_mesh_threads(mesh->info.workgroup_size),
             mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out);
}

/* GS capability is established on an immutable clone before normal
 * fallback lowering. Require an explicitly selected packed class even at P1;
 * enabling GS routing does not inherit old lane-0 qualification. */
DEBUG_GET_ONCE_BOOL_OPTION(gs_native_log, "HK_APPLE9_GS_ROUTE_LOG", false)

VkResult
hk_apple9_native_gs_check(const nir_shader *source,
                         const struct hk_apple9_graphics_key *key,
                         bool *supported, const char **reason)
{
   *supported = false;
   bool production = key->native_stages & HK_APPLE9_NATIVE_GS_BIT;
   if (!production && !hk_apple9_native_gs_enabled())
      return VK_SUCCESS;
   /* Reject common declaration/interface exclusions before cloning NIR. The
    * normal lowering still validates every instruction in admitted sources. */
   if (production &&
       (source->info.gs.input_primitive != MESA_PRIM_TRIANGLES ||
        source->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP ||
        !source->info.gs.invocations || source->info.gs.invocations > 32 ||
        source->info.gs.active_stream_mask != 1 ||
        source->info.gs.vertices_out < 3 || source->info.gs.vertices_out > 4 ||
        source->info.outputs_written != VARYING_BIT_POS ||
        (source->info.inputs_read & ~VARYING_BIT_POS) ||
        source->info.clip_distance_array_size || source->info.cull_distance_array_size ||
        source->xfb_info || source->info.writes_memory)) {
      if (reason)
         *reason = "GS declaration/interface is outside the admitted stage contract";
      return VK_SUCCESS;
   }
   const char *local_reason = NULL;
   if (!reason)
      reason = &local_reason;
   /* Early admission precedes graphics root lowering. Normalize an owned
    * clone just as the real graphics path does, so descriptor-backed counts
    * are checked as the loads that will reach the native backend. */
   nir_shader *normalized = nir_shader_clone(NULL, source);
   if (!normalized)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   NIR_PASS(_, normalized, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   nir_shader *mesh = hk_nir_gs_mesh(normalized, reason);
   ralloc_free(normalized);
   if (!mesh) {
      if (debug_get_option_gs_native_log())
         fprintf(stderr, "HK_GS_CAP accepted=0 reason=%s\n", *reason ?: "lowering");
      return VK_SUCCESS;
   }
   VkResult result = VK_SUCCESS;
   if ((hk_apple9_native_mesh_packed_shape_eligible(mesh) ||
        hk_apple9_native_stage_varyings(mesh, production)) &&
       (!production || hk_apple9_native_stage_class(
          source->info.stage, mesh->info.workgroup_size[0],
          mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out))) {
      hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
      result = hk_apple9_native_mesh_check(mesh, key, supported, reason);
   } else {
      *reason = "GS declaration is outside the admitted stage classes";
   }
   if (debug_get_option_gs_native_log())
      fprintf(stderr, "HK_GS_CAP accepted=%u reason=%s\n", *supported, *reason ?: "supported");
   ralloc_free(mesh);
   return result;
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_native_log, "HK_APPLE9_TESS_ROUTE_LOG", false)

VkResult
hk_apple9_native_tess_check(const nir_shader *source,
                            const struct hk_apple9_graphics_key *key,
                            bool *supported, const char **reason)
{
   *supported = false;
   bool production = key->native_stages & HK_APPLE9_NATIVE_TES_BIT;
   if (!production && !hk_apple9_native_tess_enabled())
      return VK_SUCCESS;
   if (production &&
       ((source->info.tess._primitive_mode != TESS_PRIMITIVE_TRIANGLES &&
         source->info.tess._primitive_mode != TESS_PRIMITIVE_QUADS) ||
        source->info.tess.point_mode || source->info.patch_inputs_read ||
        source->info.outputs_written != VARYING_BIT_POS ||
        (source->info.inputs_read & ~(VARYING_BIT_POS | VARYING_BIT_TESS_LEVEL_INNER |
                                     VARYING_BIT_TESS_LEVEL_OUTER)) ||
        source->info.clip_distance_array_size || source->info.cull_distance_array_size ||
        source->xfb_info || source->info.writes_memory)) {
      if (reason)
         *reason = "TES declaration/interface is outside the admitted stage contract";
      return VK_SUCCESS;
   }
   const char *local_reason = NULL;
   if (!reason)
      reason = &local_reason;
   /* Early admission precedes graphics root lowering. Normalize an owned
    * clone just as the real graphics path does, so descriptor-backed counts
    * are checked as the loads that will reach the native backend. */
   nir_shader *normalized = nir_shader_clone(NULL, source);
   if (!normalized)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   NIR_PASS(_, normalized, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   nir_shader *mesh = hk_nir_tes_mesh(normalized, reason);
   ralloc_free(normalized);
   if (!mesh) {
      if (debug_get_option_tess_native_log())
         fprintf(stderr, "HK_TESS_CAP accepted=0 reason=%s\n",
                 *reason ?: "lowering");
      return VK_SUCCESS;
   }
   VkResult result = VK_SUCCESS;
   if ((hk_apple9_native_mesh_packed_shape_eligible(mesh) ||
        hk_apple9_native_stage_varyings(mesh, production)) &&
       (!production || hk_apple9_native_stage_class(
          source->info.stage, mesh->info.workgroup_size[0],
          mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out))) {
      hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
      result = hk_apple9_native_mesh_check(mesh, key, supported, reason);
   } else {
      *reason = "TES declaration is outside the admitted stage classes";
   }
   if (debug_get_option_tess_native_log())
      fprintf(stderr, "HK_TESS_CAP accepted=%u reason=%s\n", *supported,
              *reason ?: "supported");
   ralloc_free(mesh);
   return result;
}

/* A native mesh API shader owns two ordinary compiled programs. The
 * generated object publishes the draw's mesh grid; it is not an API task shader.
 * Both BOs use the existing shader cache, upload and destruction paths. */
static VkResult
hk_apple9_compile_native_mesh(struct hk_device *dev, nir_shader *nir,
                               struct hk_api_shader *obj,
                               const struct hk_apple9_graphics_key *key,
                               const char **reason, bool *unsupported)
{
   struct hk_shader *mesh = hk_native_producer_main(obj);
   struct hk_shader *object = hk_native_producer_object(obj);
   hk_mesh_gather_info(nir, &mesh->info.mesh);
   mesh->info.mesh.native_task_bridge = key->native_task_bridge;
   /* The API object may be GS; this program executes at the mesh stage. */
   mesh->info.stage = MESA_SHADER_MESH;
   mesh->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_assert_final_push_mask(nir, mesh);
   bool compiled = hk_apple9_compile_native_mesh_program(
      nir, key, &mesh->b, &mesh->info.apple9_native, reason);
   /* The backend may have allocated a binary before a final capability
    * check fails. Give it to the ordinary destruction path in either case. */
   mesh->code_ptr = mesh->b.binary;
   mesh->code_size = mesh->b.info.binary_size;
   if (!compiled) {
      *unsupported = true;
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   /* Register allocation may require a frame that the native launch cannot
    * yet back. Decline this optional realization before upload. */
   if (!hk_apple9_native_shader_valid(&mesh->b.info,
                                      &mesh->info.apple9_native,
                                      HK_APPLE9_NATIVE_MESH)) {
      *unsupported = true;
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   VkResult result = hk_apple9_finish_shader(dev, mesh, nir);
   if (result != VK_SUCCESS)
      return result;

   struct agx_apple9_mesh_key k = AGX_APPLE9_MESH_KEY;
   nir_shader *generated = obj->vk.stage == MESA_SHADER_TESS_EVAL
                              ? hk_nir_tes_mesh_object()
                              : agx_nir_apple9_taskless_object(&k);
   if (!generated)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   object->info.stage = MESA_SHADER_TASK;
   object->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   hk_assert_final_push_mask(generated, object);
   compiled = agx_compile_apple9_native_object(
      generated, hk_apple9_native_mesh_available(hk_device_physical(dev)),
      &k, &object->b,
      &object->info.apple9_native, reason);
   object->code_ptr = object->b.binary;
   object->code_size = object->b.info.binary_size;
   if (!compiled) {
      *unsupported = true;
      result = VK_ERROR_FEATURE_NOT_PRESENT;
   } else {
      if (!hk_apple9_native_shader_valid(&object->b.info,
                                         &object->info.apple9_native,
                                         HK_APPLE9_NATIVE_OBJECT)) {
         *unsupported = true;
         result = VK_ERROR_FEATURE_NOT_PRESENT;
      } else {
         result = hk_apple9_finish_shader(dev, object, generated);
      }
   }
   ralloc_free(generated);
   return result;
}

/* Task shaders run as compute kernels writing their payloads and the grids of
 * mesh workgroups they launch.
 */
static VkResult
hk_apple9_compile_task(struct hk_device *dev, nir_shader *nir,
                       struct hk_shader *shader, const char **reason)
{
   shader->info.task.payload_B = nir->info.task_payload_size;
   shader->info.task.max_mesh_groups = hk_nir_task_mesh_bound(nir);
   hk_nir_lower_task_main(nir);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   hk_assert_final_push_mask(nir, shader);
   if (!agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9, reason))
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   return hk_apple9_finish_shader(dev, shader, nir);
}


/* fs_interp optionally gives the {flat, linear} input masks of the fragment
 * shader consuming a separately compiled last pre-rasterization stage. */
/* Whether a draw with this depth/stencil state can write the depth or
 * stencil aspects present. A depth/stencil feedback loop needs late tests
 * only then: an early test that writes nothing leaves the attachment the
 * fragment shader reads unchanged (turnip applies the same condition). */
bool
hk_apple9_zs_writes(const struct vk_depth_stencil_state *ds,
                    VkImageAspectFlags aspects)
{
   if (!ds)
      return true;
   struct vk_depth_stencil_state opt = *ds;
   opt.stencil.write_enable = true;
   vk_optimize_depth_stencil_state(&opt, aspects, true);
   return opt.depth.write_enable || opt.stencil.write_enable;
}

void
hk_apple9_graphics_key(struct hk_apple9_graphics_key *key,
                       const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));
   key->vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   key->vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   /* Strides are always read from the bound buffer table, which supports
    * VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE without recompilation. */
   key->vertex.dynamic_stride = true;
   /* Vulkan enables every clip distance the shader writes. */
   key->vertex.clip_distance_enable = 0xff;
   /* All non-internal fragment shaders link with a draw-time epilog. */
   key->fs_use_epilog = true;
   key->blend[0].samples = 1;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
      key->color_map[rt] = rt;
   if (!state) {
      /* Independent shader-object compile (ESO): decouple vertex input,
       * clip-space depth range, and fragment output/blending. */
      key->vertex.use_prolog = true;
      key->clip_negative_one_to_one_dynamic = true;
      key->may_line_stipple = true;
      key->may_polygon_points = true;
      key->blend[0].dynamic_samples = true;
      key->blend[0].sample_mask_single_sample = true;
      key->blend[0].dynamic_sample_positions = true;
      key->blend[0].clamp_early_depth = true;
      key->blend[0].early_depth_clamp_offset_B =
         hk_root_descriptor_offset(draw.depth_clamp);
      return;
   }
   key->is_library =
      !!(state->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR);
   if (state->cal) {
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
         key->color_map[rt] = state->cal->color_map[rt];
   }
   if (!state->vp ||
       key->is_library ||
       BITSET_TEST(state->dynamic,
                   MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE)) {
      key->clip_negative_one_to_one_dynamic = true;
      key->clip_negative_one_to_one = false;
   } else {
      key->clip_negative_one_to_one =
         state->vp->depth_clip_negative_one_to_one;
   }
   if (state->rp) {
      for (unsigned rt = 0; rt < MIN2(state->rp->color_attachment_count, 8); ++rt) {
         if (state->rp->color_attachment_formats[rt] == VK_FORMAT_UNDEFINED)
            key->undefined_colors |= BITFIELD_BIT(rt);
      }
   }
   key->sample_shading = state->ms && state->ms->sample_shading_enable;
   key->zs_self_dep = !!(state->pipeline_flags &
                         VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
   key->line_stipple =
      state->rs && state->rs->line.stipple.enable &&
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
   key->may_line_stipple =
      key->line_stipple || !state->rs ||
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
   key->polygon_points =
      state->rs && state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);
   if (state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {
      enum mesa_prim prim =
         u_reduced_prim(vk_topology_to_mesa(state->ia->primitive_topology));
      key->polygon_points &= prim != MESA_PRIM_POINTS && prim != MESA_PRIM_LINES;
   }
   key->may_polygon_points =
      key->polygon_points || !state->rs ||
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE) ||
      state->rs->polygon_mode == VK_POLYGON_MODE_POINT;

   if (state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {
      key->vertex.rasterize_points =
         state->ia->primitive_topology == MESA_PRIM_POINTS;
      key->vertex.ignore_point_size = !key->vertex.rasterize_points;
   }
   if (!state->vi || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI)) {
      key->vertex.use_prolog = true;
   } else {
      for (unsigned a = 0; a < 16; ++a) {
         if (!(state->vi->attributes_valid & BITFIELD_BIT(a)))
            continue;
         const struct vk_vertex_attribute_state *attr =
            &state->vi->attributes[a];
         const struct vk_vertex_binding_state *binding =
            &state->vi->bindings[attr->binding];
         key->vertex.format[a] = vk_format_to_pipe_format(attr->format);
         key->vertex.offset[a] = attr->offset;
         key->vertex.buffer[a] = attr->binding;
         /* A zero instance divisor repeats the first instance's element. */
         key->vertex.divisor[a] =
            binding->input_rate != VK_VERTEX_INPUT_RATE_INSTANCE ? 0
            : binding->divisor ? binding->divisor
                               : AGX_APPLE9_VERTEX_DIVISOR_ZERO;
      }
   }
   key->targets = state->rp ? state->rp->color_attachment_count : 0;
   for (unsigned rt = 0; rt < MAX2(MIN2(key->targets, 8), 1); ++rt) {
      struct agx_apple9_blend *blend = &key->blend[rt];
      blend->format = rt < key->targets
         ? vk_format_to_pipe_format(state->rp->color_attachment_formats[rt])
         : PIPE_FORMAT_NONE;
      blend->color_attachment_unused = blend->format == PIPE_FORMAT_NONE;
      blend->sample_mask_single_sample = true;
      blend->dynamic_sample_positions = true;
      blend->clamp_early_depth =
         !state->rs || state->rs->depth_clamp_enable ||
         BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
      blend->early_depth_clamp_ubo = 0;
      blend->early_depth_clamp_offset_B =
         hk_root_descriptor_offset(draw.depth_clamp);
      blend->samples =
         (state->ms &&
          !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
          state->ms->rasterization_samples)
            ? state->ms->rasterization_samples : 1;
      if (rt == 0 &&
          (!state->ms ||
           BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES)))
         blend->dynamic_samples = true;
      blend->rgb_src = blend->alpha_src = PIPE_BLENDFACTOR_ONE;
      blend->rgb_dst = blend->alpha_dst = PIPE_BLENDFACTOR_ZERO;
      blend->rgb_func = blend->alpha_func = PIPE_BLEND_ADD;
      if (state->ms) {
         blend->disabled_samples = ~state->ms->sample_mask;
         blend->alpha_to_coverage = state->ms->alpha_to_coverage_enable;
         blend->alpha_to_one = state->ms->alpha_to_one_enable;
      }
      if (state->cb && rt < state->cb->attachment_count) {
         const struct vk_color_blend_attachment_state *cb =
            &state->cb->attachments[rt];
         blend->dynamic_enable = BITSET_TEST(
            state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
         blend->colormask = (blend->dynamic_enable ||
                             (state->cb->color_write_enables & BITFIELD_BIT(rt)))
                               ? cb->write_mask
                               : 0;
         blend->logicop_enable = state->cb->logic_op_enable;
         blend->logicop_func = vk_logic_op_to_pipe(state->cb->logic_op);
         if (cb->blend_enable && cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            /* Advanced equations are blended in the shader from the tile. */
            blend->advanced_mode =
               vk_advanced_blend_op_to_pipe(cb->color_blend_op);
            blend->advanced_overlap =
               vk_blend_overlap_to_pipe(cb->blend_overlap);
            blend->src_premultiplied = cb->src_premultiplied;
            blend->dst_premultiplied = cb->dst_premultiplied;
         } else if (cb->blend_enable) {
            blend->rgb_src =
               vk_blend_factor_to_pipe(cb->src_color_blend_factor);
            blend->rgb_dst =
               vk_blend_factor_to_pipe(cb->dst_color_blend_factor);
            blend->alpha_src =
               vk_blend_factor_to_pipe(cb->src_alpha_blend_factor);
            blend->alpha_dst =
               vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor);
            blend->rgb_func = vk_blend_op_to_pipe(cb->color_blend_op);
            blend->alpha_func = vk_blend_op_to_pipe(cb->alpha_blend_op);
         }
      }
   }
}

void
hk_apple9_graphics_key_device(struct hk_apple9_graphics_key *key,
                              const struct vk_graphics_pipeline_state *state,
                              const struct hk_physical_device *pdev)
{
   hk_apple9_graphics_key(key, state);
   key->native_index = hk_apple9_native_index_enabled(pdev);
   key->native_stages = hk_apple9_native_stage_policy(pdev);
}

VkResult
hk_apple9_compile_graphics_path(struct hk_device *dev, uint32_t count,
                           struct vk_shader_compile_info *infos,
                           const struct vk_graphics_pipeline_state *state,
                           const struct hk_apple9_graphics_key *key_in,
                           const uint64_t *fs_interp,
                           const VkAllocationCallbacks *alloc,
                           struct vk_shader **out, bool native_mesh,
                           bool *native_unsupported)
{
   VkResult result = VK_SUCCESS;
   struct hk_apple9_graphics_key key;
   if (key_in)
      key = *key_in;
   else
      hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
   struct agx_apple9_varying_layout varyings = {0};
   const char *reason = NULL;
   simple_mtx_t *lock = agx_get_compiler_debug()
                           ? &hk_device_physical(dev)->debug_compile_lock
                           : NULL;
   if (lock)
      simple_mtx_lock(lock);

   memset(out, 0, count * sizeof(*out));
   if (native_mesh && !key.native_task_bridge) {
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage == MESA_SHADER_TASK) {
            reason = "native mesh does not support API task shaders";
            result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT, "%s", reason);
            goto done;
         }
      }
   }

   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         continue;
      hk_apple9_lower_primitive_fs(infos[i].nir, native_mesh);
   }

   /* Per-vertex inputs are read as flat copies, written by the passthrough
    * geometry shader of vertex/evaluation-shader pipelines, or by the vertex
    * shader rasterizing mesh records. Dedicated Vulkan-unused slots let FS
    * lower per-vertex and barycentric inputs independently of any producer.
    */
   struct hk_per_vertex_copies per_vertex = {0};
   bool copies_from_gs = false;
   struct hk_apple9_stipple_slots stipple = {0};
   bool fs_lowered_stipple = false;
   bool polygon_points = false;
   bool has_stipple_feature =
      dev->vk.enabled_features.stippledRectangularLines ||
      dev->vk.enabled_features.stippledBresenhamLines ||
      dev->vk.enabled_features.stippledSmoothLines;
   bool may_line_stipple =
      key.line_stipple ||
      (key.may_line_stipple && ((state && state->rs) || has_stipple_feature));
   bool may_polygon_points =
      key.may_polygon_points || key.is_library || !state;
   unsigned polygon_face_slot = 0;
   bool gs_dynamic_polygon_points = false;
   mesa_shader_stage polygon_stage = MESA_SHADER_NONE;
   mesa_shader_stage stipple_stage = MESA_SHADER_NONE;
   {
      nir_shader *pre_nir = NULL, *fs_nir = NULL;
      nir_shader *stage_nir[MESA_VULKAN_SHADER_STAGES] = {NULL};
      for (unsigned i = 0; i < count; ++i) {
         stage_nir[infos[i].stage] = infos[i].nir;
         if (infos[i].stage == MESA_SHADER_VERTEX ||
             infos[i].stage == MESA_SHADER_MESH)
            pre_nir = infos[i].nir;
         else if (infos[i].stage == MESA_SHADER_FRAGMENT)
            fs_nir = infos[i].nir;
      }
      if (fs_nir && !stage_nir[MESA_SHADER_GEOMETRY] &&
          hk_apple9_lower_per_vertex_copies(
             fs_nir, pre_nir ? pre_nir->info.outputs_written : 0, &per_vertex))
         copies_from_gs = !pre_nir || pre_nir->info.stage != MESA_SHADER_MESH;

      /* Stippled lines use dedicated Vulkan-unused slots TEX6 (flat start +
       * counter) and TEX7 (noperspective axis). For VS/TES, the passthrough
       * geometry shader computes them directly from POS/VIEWPORT with zero
       * VS/TES modification; app GS and emulated Mesh emit them directly. */
      nir_shader *last = NULL;
      if (stage_nir[MESA_SHADER_GEOMETRY]) {
         if (stage_nir[MESA_SHADER_GEOMETRY]->info.gs.output_primitive ==
             MESA_PRIM_LINE_STRIP)
            last = stage_nir[MESA_SHADER_GEOMETRY];
      } else if (stage_nir[MESA_SHADER_TESS_EVAL]) {
         nir_shader *tes = stage_nir[MESA_SHADER_TESS_EVAL];
         nir_shader *tcs = stage_nir[MESA_SHADER_TESS_CTRL];
         enum tess_primitive_mode mode = tes->info.tess._primitive_mode;
         if (mode == TESS_PRIMITIVE_UNSPECIFIED && tcs)
            mode = tcs->info.tess._primitive_mode;
         bool points = tes->info.tess.point_mode ||
                       (tcs && tcs->info.tess.point_mode);
         if (mode == TESS_PRIMITIVE_ISOLINES && !points)
            last = tes;
      } else if (pre_nir && pre_nir->info.stage == MESA_SHADER_VERTEX) {
         last = pre_nir;
      } else if (pre_nir && pre_nir->info.stage == MESA_SHADER_MESH &&
                 pre_nir->info.mesh.primitive_type == MESA_PRIM_LINES) {
         last = pre_nir;
      }
      if (key.line_stipple && last && fs_nir) {
         if (hk_apple9_slot_fits(last, HK_APPLE9_STIPPLE_COMPONENTS)) {
            stipple.start = VARYING_SLOT_TEX6;
            stipple.axis = VARYING_SLOT_TEX7;
            stipple_stage = last->info.stage;
            hk_apple9_lower_line_stipple_fs(fs_nir, stipple);
            fs_lowered_stipple = true;
         } else {
            static bool warned;
            if (!warned) {
               mesa_logw("Apple9: line stipple is ignored when the last "
                         "pre-rasterization stage uses more than %u of %u "
                         "output components",
                         HK_APPLE9_OUTPUT_COMPONENTS -
                            HK_APPLE9_STIPPLE_COMPONENTS,
                         HK_APPLE9_OUTPUT_COMPONENTS);
               warned = true;
            }
         }
      } else if (stage_nir[MESA_SHADER_GEOMETRY] &&
                 !stage_nir[MESA_SHADER_GEOMETRY]->info.internal &&
                 stage_nir[MESA_SHADER_GEOMETRY]->info.gs.output_primitive ==
                    MESA_PRIM_LINE_STRIP &&
                 may_line_stipple &&
                 hk_apple9_slot_fits(stage_nir[MESA_SHADER_GEOMETRY],
                                     HK_APPLE9_STIPPLE_COMPONENTS)) {
         /* Dynamic line stipple or GPL/ESO: retain stipple TEX6/TEX7 exports
          * on application line-strip GS so the draw-time stipple prolog can
          * consume them without recompiling the GS. */
         stipple.start = VARYING_SLOT_TEX6;
         stipple.axis = VARYING_SLOT_TEX7;
         stipple_stage = MESA_SHADER_GEOMETRY;
      }

      /* Lower FS gl_FrontFacing independently when polygon mode may be POINT
       * (static POINT, dynamic polygon mode, ESO, or GPL): selects flat(FOGC)==0
       * when root.draw.polygon_point_facing is nonzero, else load_front_face(). */
      if (fs_nir && !fs_nir->info.internal && may_polygon_points &&
          BITSET_TEST(fs_nir->info.system_values_read, SYSTEM_VALUE_FRONT_FACE)) {
         polygon_face_slot = VARYING_SLOT_FOGC;
         NIR_PASS(_, fs_nir, nir_shader_intrinsics_pass,
                  hk_apple9_lower_polygon_slot, nir_metadata_control_flow,
                  &polygon_face_slot);
         nir_shader_gather_info(fs_nir, nir_shader_get_entrypoint(fs_nir));
      }

      /* Polygon-mode points take the software polygon stage when the last
       * pre-rasterization stage's outputs fit: after a vertex or evaluation
       * shader, inside a geometry shader emitting triangle strips, or in the
       * vertex shader rasterizing mesh triangles. Otherwise they keep the
       * hardware polygon mode, a known limitation. */
      nir_shader *mesh_nir = stage_nir[MESA_SHADER_MESH];
      nir_shader *pp_last =
         mesh_nir
            ? (mesh_nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
                     !hk_per_vertex_copies_used(&per_vertex)
                  ? mesh_nir : NULL)
         : stage_nir[MESA_SHADER_GEOMETRY]
            ? (hk_apple9_gs_polygon_points_ok(stage_nir[MESA_SHADER_GEOMETRY])
                  ? stage_nir[MESA_SHADER_GEOMETRY] : NULL)
         : stage_nir[MESA_SHADER_TESS_EVAL] ? stage_nir[MESA_SHADER_TESS_EVAL]
         : stage_nir[MESA_SHADER_VERTEX];
      if (key.polygon_points && pp_last && fs_nir) {
         uint64_t reserved = 0;
         if (stipple.start) {
            reserved = BITFIELD64_BIT(stipple.start) | BITFIELD64_BIT(stipple.axis);
         }
         if (mesh_nir)
            reserved |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);
         unsigned budget_face_slot = 0;
         polygon_points = hk_apple9_polygon_point_budget(pp_last, fs_nir, reserved,
                                                         &budget_face_slot);
         if (polygon_points)
            polygon_stage = pp_last->info.stage;
         if (!polygon_points) {
            static bool warned;
            if (!warned) {
               mesa_logw("Apple9: polygon mode POINT keeps the hardware path "
                         "(size 1) when the stage's outputs leave no room for "
                         "point size and facing");
               warned = true;
            }
         }
      }
      if (!polygon_points && may_polygon_points &&
          stage_nir[MESA_SHADER_GEOMETRY] &&
          !stage_nir[MESA_SHADER_GEOMETRY]->info.internal &&
          !stage_nir[MESA_SHADER_GEOMETRY]->xfb_info &&
          hk_apple9_gs_polygon_points_ok(stage_nir[MESA_SHADER_GEOMETRY])) {
         nir_shader *gs_nir = stage_nir[MESA_SHADER_GEOMETRY];
         uint64_t reserved =
            stipple.start
               ? (BITFIELD64_BIT(stipple.start) | BITFIELD64_BIT(stipple.axis))
               : 0;
         unsigned budget_face_slot = 0;
         gs_dynamic_polygon_points =
            fs_nir
               ? hk_apple9_polygon_point_budget(gs_nir, fs_nir, reserved,
                                                &budget_face_slot)
               : hk_apple9_slot_fits(
                    gs_nir, 2 + !(gs_nir->info.outputs_written & VARYING_BIT_PSIZ));
      }
   }

   if (fs_interp) {
      key.vertex.outputs_flat = fs_interp[0];
      key.vertex.outputs_linear = fs_interp[1];
   }
   if (key.targets > 8) {
      reason = "more than eight color targets";
      result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      goto done;
   }
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_COMPUTE) {
         reason = "compute shaders are compiled separately";
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
         goto done;
      }
      if (infos[i].stage == MESA_SHADER_FRAGMENT) {
         key.vertex.outputs_flat = infos[i].nir->info.inputs_read &
                                   ~(infos[i].nir->info.perspective_varyings |
                                     infos[i].nir->info.linear_varyings);
         key.vertex.outputs_linear = infos[i].nir->info.linear_varyings;
         key.fs_reads_clip_distance = infos[i].nir->info.inputs_read &
            (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1);
      }
   }

   /* Tessellation modes may be declared by either tessellation stage, as
    * HLSL declares them on the control shader. Rasterization of the
    * evaluation shader's output depends on them.
    */
   nir_shader *tcs_nir = NULL, *tes_nir = NULL;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_TESS_CTRL)
         tcs_nir = infos[i].nir;
      else if (infos[i].stage == MESA_SHADER_TESS_EVAL)
         tes_nir = infos[i].nir;
   }
   if (tcs_nir && tes_nir) {
      tes_nir->info.tess.point_mode |= tcs_nir->info.tess.point_mode;
      tes_nir->info.tess.ccw |= tcs_nir->info.tess.ccw;
      if (tes_nir->info.tess.spacing == TESS_SPACING_UNSPECIFIED)
         tes_nir->info.tess.spacing = tcs_nir->info.tess.spacing;
      if (tes_nir->info.tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED)
         tes_nir->info.tess._primitive_mode = tcs_nir->info.tess._primitive_mode;
   }

   bool has_fragment = false;
   for (unsigned i = 0; i < count; ++i)
      has_fragment |= infos[i].stage == MESA_SHADER_FRAGMENT;

   /* The last pre-rasterization stage maps [-w, w] depth to [0, w]. */
   mesa_shader_stage last_prerast = MESA_SHADER_NONE;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX ||
          infos[i].stage == MESA_SHADER_TESS_EVAL ||
          infos[i].stage == MESA_SHADER_GEOMETRY)
         last_prerast = MAX2(last_prerast, (int)infos[i].stage);
      else if (infos[i].stage == MESA_SHADER_MESH)
         last_prerast = MESA_SHADER_MESH;
   }

   /* Compile the producer first, independent of the API stage-array order. */
   static const mesa_shader_stage order[] = {
      MESA_SHADER_VERTEX,    MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
      MESA_SHADER_GEOMETRY,  MESA_SHADER_TASK,      MESA_SHADER_MESH,
      MESA_SHADER_FRAGMENT,
   };
   for (unsigned pass = 0; pass < ARRAY_SIZE(order); ++pass) {
      mesa_shader_stage stage = order[pass];
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage != stage)
            continue;
         size_t size = sizeof(struct hk_api_shader) +
                       hk_num_variants(stage) * sizeof(struct hk_shader);
         struct hk_api_shader *obj =
            vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, alloc, size);
         if (!obj) {
            result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
            goto done;
         }
         out[i] = &obj->vk;
         hk_foreach_variant(obj, variant) {
            variant->info.stage = stage;
            variant->b.info.stats.instrs = 1;
         }
         struct hk_shader *shader = &obj->variants[0];
         nir_shader *nir = infos[i].nir;
         obj->writes_memory = nir->info.writes_memory;
         obj->pre_raster_reads = hk_nir_pre_raster_reads(nir);
         obj->reads_draw_id = hk_nir_reads_draw_id(nir);
         obj->push_constant_mask = hk_nir_push_constant_mask(nir);
#ifndef NDEBUG
         hk_foreach_variant(obj, variant)
            variant->debug_owner = obj;
#endif
         shader->info.set_count = infos[i].set_layout_count;
         shader->info.apple9_render_targets = key.targets;
         shader->info.apple9_line_stipple_first = false;
         if (!hk_apple9_texture_bindings(nir) ||
             !agx_nir_lower_apple9_runtime_sampler_state(nir, &reason)) {
            reason = reason ?: "texture operation without a descriptor handle";
            result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                               "Apple9 graphics textures: %s", reason);
            goto done;
         }
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
                  nir_metadata_control_flow, NULL);
         if ((key.clip_negative_one_to_one || key.clip_negative_one_to_one_dynamic) &&
             stage == last_prerast &&
             !(native_mesh && (stage == MESA_SHADER_GEOMETRY ||
                               stage == MESA_SHADER_TESS_EVAL))) {
            bool dynamic_clip = key.clip_negative_one_to_one_dynamic;
            NIR_PASS(_, nir, nir_lower_clip_halfz_dynamic);
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     hk_apple9_lower_clip_z_coeff, nir_metadata_control_flow,
                     &dynamic_clip);
         }
         /* Shadow and depth-prepass draws: a vertex shader rasterized without
          * a fragment stage computes only what the rasterizer reads. Never
          * strip outputs for GPL libraries or shader objects that may later
          * link with a fragment stage. */
         if (stage == MESA_SHADER_VERTEX && stage == last_prerast &&
             !has_fragment && !fs_interp && !nir->xfb_info &&
             !(infos[i].next_stage_mask &
               (VK_SHADER_STAGE_FRAGMENT_BIT |
                VK_SHADER_STAGE_GEOMETRY_BIT |
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) &&
             state &&
             !(state->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) &&
             !hk_per_vertex_copies_used(&per_vertex))
            hk_apple9_strip_unread_outputs(nir);

         if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
            nir_print_shader(nir, stderr);

         if (stage == MESA_SHADER_VERTEX) {
            /* Native vertex pulling indexes attributes by API location. */
            nir_foreach_function_impl(impl, nir) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic)
                        continue;
                     nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic == nir_intrinsic_load_input) {
                        unsigned loc =
                           nir_intrinsic_io_semantics(intr).location;
                        nir_intrinsic_set_base(intr,
                                               loc - VERT_ATTRIB_GENERIC0);
                     }
                  }
               }
            }
            /* Transform feedback, flat copies of per-vertex inputs, line
             * stipple, polygon points, and primitives-generated queries are
             * layered on a passthrough geometry shader, which consumes
             * HK_VS_VARIANT_SW. For GPL pre-rasterization libraries (where
             * !has_fragment), only compile HK_VS_VARIANT_SW when the library
             * or enabled device features can actually reach GS/TCS or a
             * passthrough GS — avoiding wasted SW-VS compiles on normal GPL
             * titles (CS2/DXVK). */
            bool static_point_mode =
               state && state->rs &&
               state->rs->polygon_mode == VK_POLYGON_MODE_POINT &&
               (!state->ia ||
                BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY) ||
                u_reduced_prim(vk_topology_to_mesa(state->ia->primitive_topology)) ==
                   MESA_PRIM_TRIANGLES);
            bool may_polygon_point_gs =
               key.polygon_points ||
               (key.may_polygon_points &&
                (state && state->rs
                    ? (BITSET_TEST(state->dynamic,
                                   MESA_VK_DYNAMIC_RS_POLYGON_MODE) ||
                       static_point_mode)
                    : dev->vk.enabled_features.fillModeNonSolid));
            bool may_link_barycentric =
               (!state || key.is_library) && !has_fragment &&
               dev->vk.enabled_features.fragmentShaderBarycentric;
            bool unknown_next_stage =
               !infos[i].next_stage_mask &&
               (dev->vk.enabled_features.geometryShader ||
                dev->vk.enabled_features.tessellationShader ||
                dev->vk.enabled_features.transformFeedback);
            bool sw_needed =
               (infos[i].next_stage_mask &
                (VK_SHADER_STAGE_GEOMETRY_BIT |
                 VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) ||
               unknown_next_stage ||
               nir->xfb_info || hk_per_vertex_copies_used(&per_vertex) ||
               stipple_stage == MESA_SHADER_VERTEX ||
               polygon_stage == MESA_SHADER_VERTEX ||
               !state || may_line_stipple || may_polygon_point_gs ||
               may_link_barycentric ||
               dev->vk.enabled_features.primitivesGeneratedQuery;
            struct hk_shader *sw = &obj->variants[HK_VS_VARIANT_SW];
            nir_shader *sw_nir = NULL;
            if (sw_needed || nir->info.cull_distance_array_size) {
               sw->info.set_count = shader->info.set_count;
               sw->info.clip_distance_array_size =
                  nir->info.clip_distance_array_size;
               sw->info.cull_distance_array_size =
                  nir->info.cull_distance_array_size;
               if (nir->xfb_info) {
                  assert(nir->xfb_info->output_count <
                         ARRAY_SIZE(sw->info.xfb_outputs));
                  memcpy(&sw->info.xfb_info, nir->xfb_info,
                         nir_xfb_info_size(nir->xfb_info->output_count));
                  typed_memcpy(sw->info.xfb_stride, nir->info.xfb_stride, 4);
               }
               sw_nir = nir_shader_clone(NULL, nir);
            }
            /* Without per-vertex inputs, the hardware vertex shader writes
             * the one-hot barycentric weights itself, so non-indexed
             * triangle lists need no passthrough geometry shader. */
            if (!per_vertex.count && hk_per_vertex_copies_used(&per_vertex))
               hk_apple9_write_vertex_weights(nir, &per_vertex);
            struct agx_apple9_vertex_layout vs_layout = key.vertex;
            /* Dynamic topology (or ESO/GPL without IA state): always run
             * point-size lowering and synthesize default 1.0 when unwritten
             * if output slots fit. */
            if (!vs_layout.rasterize_points && !vs_layout.ignore_point_size &&
                hk_apple9_psiz_fits(nir)) {
               vs_layout.rasterize_points = true;
               vs_layout.ignore_point_size = false;
            }
            if (nir->info.outputs_written & VARYING_BIT_VIEWPORT)
               vs_layout.outputs_flat |= VARYING_BIT_VIEWPORT;
            shader->info.apple9_vertex_unbounded = vs_layout.unbounded;
            bool cull_mask;
            hk_assert_final_push_mask(nir, shader);
            if (!hk_apple9_compile_vertex_culls(nir, &vs_layout, &shader->b,
                                                &cull_mask, &reason)) {
               ralloc_free(sw_nir);
               reason = reason ?: "unsupported vertex NIR";
               result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                                  "Apple9 vertex: %s", reason);
               goto done;
            }
            shader->info.cull_mask = sw->info.cull_mask = cull_mask;
            if (sw_needed || cull_mask) {
               result = hk_apple9_compile_sw_vs(dev, sw_nir, &vs_layout, sw,
                                                &reason);
            }
            ralloc_free(sw_nir);
            if (result != VK_SUCCESS)
               goto done;
            varyings = shader->b.info.apple9_varyings;
         } else if (stage == MESA_SHADER_TESS_CTRL) {
            result = hk_apple9_compile_tcs(dev, nir, shader, &reason);
            if (result != VK_SUCCESS)
               goto done;
            continue;
         } else if (stage == MESA_SHADER_TESS_EVAL) {
            if (native_mesh) {
               obj->apple9_native_tess = (struct hk_tess_info){
                  .mode = nir->info.tess._primitive_mode,
                  .spacing = nir->info.tess.spacing,
                  .ccw = nir->info.tess.ccw,
                  .points = nir->info.tess.point_mode,
               };
               nir_shader *mesh = hk_nir_tes_mesh(nir, &reason);
               if (!mesh) {
                  *native_unsupported = true;
                  result = VK_ERROR_FEATURE_NOT_PRESENT;
                  goto done;
               }
               hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
               result = hk_apple9_compile_native_mesh(
                  dev, mesh, obj, &key, &reason, native_unsupported);
               ralloc_free(mesh);
               if (result != VK_SUCCESS)
                  goto done;
               varyings = hk_native_producer_main(obj)->b.info.apple9_varyings;
               continue;
            }
            bool may_polygon_point_gs =
               key.polygon_points ||
               (key.may_polygon_points &&
                ((state && state->rs) ||
                 dev->vk.enabled_features.fillModeNonSolid));
            bool may_link_barycentric =
               (!state || key.is_library) && !has_fragment &&
               dev->vk.enabled_features.fragmentShaderBarycentric;
            bool unknown_next_stage =
               !infos[i].next_stage_mask &&
               (dev->vk.enabled_features.geometryShader ||
                dev->vk.enabled_features.transformFeedback);
            bool tes_sw_needed =
               polygon_stage == MESA_SHADER_TESS_EVAL ||
               stipple_stage == MESA_SHADER_TESS_EVAL ||
               copies_from_gs || !state || unknown_next_stage ||
               may_line_stipple || may_polygon_point_gs ||
               may_link_barycentric ||
               dev->vk.enabled_features.primitivesGeneratedQuery;
            result = hk_apple9_compile_tes(
               dev, nir, &key.vertex, obj,
               infos[i].next_stage_mask |
                  (tes_sw_needed ? VK_SHADER_STAGE_GEOMETRY_BIT : 0),
               &reason);
            if (result != VK_SUCCESS)
               goto done;
            varyings = shader->b.info.apple9_varyings;
            continue;
         } else if (stage == MESA_SHADER_GEOMETRY) {
            if (native_mesh) {
               nir_shader *mesh = hk_nir_gs_mesh(nir, &reason);
               if (!mesh) {
                  *native_unsupported = true;
                  result = VK_ERROR_FEATURE_NOT_PRESENT;
                  goto done;
               }
               hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
               result = hk_apple9_compile_native_mesh(
                  dev, mesh, obj, &key, &reason, native_unsupported);
               ralloc_free(mesh);
               if (result != VK_SUCCESS)
                  goto done;
               hk_native_producer_main(obj)->info.apple9_gs_invocations =
                  MAX2(nir->info.gs.invocations, 1);
               varyings = hk_native_producer_main(obj)->b.info.apple9_varyings;
               continue;
            }
            if (stipple_stage == MESA_SHADER_GEOMETRY)
               hk_apple9_write_gs_line_stipple(nir, stipple);
            nir_shader *pp_nir = NULL;
            if (gs_dynamic_polygon_points) {
               pp_nir = nir_shader_clone(NULL, nir);
               if (!pp_nir) {
                  result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
                  goto done;
               }
            }
            if (polygon_stage == MESA_SHADER_GEOMETRY) {
               struct hk_pp_info info = {
                  .clip = nir->info.clip_distance_array_size,
                  .cull = nir->info.cull_distance_array_size,
                  .flat = key.vertex.outputs_flat,
                  .face_slot = VARYING_SLOT_FOGC,
               };
               hk_apple9_lower_gs_polygon_points(
                  nir, info, hk_device_physical(dev)->vk.properties.polygonModePointSize);
            }
            result = hk_apple9_compile_gs(dev, nir, &key.vertex,
                                          !key.gs_cull_planes, obj, &reason);
            if (result != VK_SUCCESS) {
               ralloc_free(pp_nir);
               goto done;
            }
            if (pp_nir) {
               struct hk_api_shader *pp_obj =
                  vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, alloc, size);
               if (!pp_obj) {
                  ralloc_free(pp_nir);
                  result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
                  goto done;
               }
               hk_foreach_variant(pp_obj, variant) {
                  variant->info.stage = stage;
                  variant->b.info.stats.instrs = 1;
#ifndef NDEBUG
                  variant->debug_owner = pp_obj;
#endif
               }
               pp_obj->writes_memory = obj->writes_memory;
               pp_obj->pre_raster_reads = obj->pre_raster_reads;
               pp_obj->reads_draw_id = obj->reads_draw_id;
               pp_obj->push_constant_mask = obj->push_constant_mask;
               pp_obj->variants[0].info.set_count = shader->info.set_count;
               struct hk_pp_info info = {
                  .clip = pp_nir->info.clip_distance_array_size,
                  .cull = pp_nir->info.cull_distance_array_size,
                  .flat = key.vertex.outputs_flat,
                  .face_slot = VARYING_SLOT_FOGC,
               };
               hk_apple9_lower_gs_polygon_points(
                  pp_nir, info,
                  hk_device_physical(dev)->vk.properties.polygonModePointSize);
               const char *pp_reason = NULL;
               VkResult pp_res = hk_apple9_compile_gs(
                  dev, pp_nir, &key.vertex, !key.gs_cull_planes, pp_obj,
                  &pp_reason);
               ralloc_free(pp_nir);
               if (pp_res == VK_SUCCESS) {
                  hk_api_shader_init_constant_data(pp_obj);
                  obj->apple9_gs_polygon_points = pp_obj;
               } else {
                  hk_api_shader_destroy(&dev->vk, &pp_obj->vk, alloc);
                  if (pp_res == VK_ERROR_OUT_OF_HOST_MEMORY ||
                      pp_res == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
                     result = pp_res;
                     goto done;
                  }
               }
            }
            varyings = shader->b.info.apple9_varyings;
            continue;
         } else if (stage == MESA_SHADER_TASK) {
            result = hk_apple9_compile_task(dev, nir, shader, &reason);
            if (result != VK_SUCCESS)
               goto done;
            continue;
         } else if (stage == MESA_SHADER_MESH) {
            if (native_mesh) {
               result = hk_apple9_compile_native_mesh(
                  dev, nir, obj, &key, &reason, native_unsupported);
               if (result != VK_SUCCESS)
                  goto done;
               varyings = shader->b.info.apple9_varyings;
               continue;
            }

            struct hk_pp_info pp = {
               .clip = nir->info.clip_distance_array_size,
               .cull = nir->info.cull_distance_array_size,
               .flat = key.vertex.outputs_flat,
               .face_slot = VARYING_SLOT_FOGC,
            };
            result = hk_apple9_compile_mesh(
               dev, nir, &key.vertex, &per_vertex,
               stipple_stage == MESA_SHADER_MESH
                  ? stipple : (struct hk_apple9_stipple_slots){0},
               polygon_stage == MESA_SHADER_MESH ? &pp : NULL,
               hk_device_physical(dev)->vk.properties.polygonModePointSize,
               obj, &reason);
            if (result != VK_SUCCESS)
               goto done;
            varyings = obj->variants[HK_MESH_VARIANT_RAST].b.info.apple9_varyings;
            continue;
         } else {
            obj->zs_self_dep =
               state &&
               (state->pipeline_flags &
                VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
            shader->info.fs.interp.linear = key.vertex.outputs_linear;
            shader->info.fs.interp.flat = key.vertex.outputs_flat;
            shader->info.fs.line_stipple_start =
               fs_lowered_stipple ? stipple.start : 0;
            shader->info.fs.polygon_points = may_polygon_points || polygon_points;
            shader->info.fs.polygon_face_slot = polygon_face_slot;
            if (hk_per_vertex_copies_used(&per_vertex)) {
               shader->info.fs.per_vertex_copies = per_vertex.count;
               memcpy(shader->info.fs.per_vertex_bary, per_vertex.bary,
                      sizeof(per_vertex.bary));
            }
            memcpy(shader->info.fs.per_vertex_location, per_vertex.location,
                   sizeof(per_vertex.location));
            memcpy(shader->info.fs.per_vertex_copy_base, per_vertex.copy_base,
                   sizeof(per_vertex.copy_base));
            /* The statistics counter is the only memory write or wave-wide
             * operation of many shaders, but it runs only while a query is
             * active and counts correctly in merged waves. Keep triangle
             * merging unless the shader itself requires disabling it. */
            bool tri_merging = true;
            bool real_writes_memory = false;
            bool inline_statistics = false;
            if (!nir->info.internal) {
               if (key.zs_self_dep && !key.fs_use_epilog)
                  hk_apple9_late_zs(nir);
               nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
               real_writes_memory = nir->info.writes_memory;
               const bool per_sample =
                  key.sample_shading || nir->info.fs.uses_sample_shading;
               tri_merging = !nir->info.uses_wide_subgroup_intrinsics &&
                             !nir->info.fs.needs_coarse_quad_helper_invocations &&
                             !nir->info.writes_memory &&
                             !per_sample &&
                             !hk_apple9_interpolates_off_center(nir);
               /* Normal epilog fragment shaders count invocations via a
                * draw-time linked stats prolog only when queried. Keep the
                * counter inline for early_fragment_tests (must execute after
                * fs_main's early Z/S test), per-sample shaders (must count per
                * covered sample inside the trampoline loop), and non-epilog
                * shaders. */
               if (dev->vk.enabled_features.pipelineStatisticsQuery &&
                   !key.no_fs_statistics &&
                   (!key.fs_use_epilog || nir->info.fs.early_fragment_tests ||
                    per_sample)) {
                  hk_apple9_lower_fs_statistics(nir);
                  inline_statistics = true;
               }
               NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                        hk_apple9_clamp_depth_export,
                        nir_metadata_control_flow, NULL);
               NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                        hk_apple9_lower_viewport_fs,
                        nir_metadata_control_flow, NULL);
            }
            shader->info.fs.writes_memory = real_writes_memory;
            shader->info.fs.early_fragment_tests =
               nir->info.fs.early_fragment_tests;
            shader->info.fs.inline_statistics = inline_statistics;
            if (key.sample_shading) {
               nir->info.fs.uses_sample_shading = true;
               NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                        hk_apple9_lower_bary_to_sample,
                        nir_metadata_control_flow, NULL);
            }
            NIR_PASS(_, nir, hk_apple9_lower_api_vertex_order);
            if (key.fs_use_epilog && !nir->info.internal) {
               key.blend[0].use_epilog = true;
               key.blend[0].dynamic_samples = true;
               key.blend[0].per_sample =
                  key.sample_shading || nir->info.fs.uses_sample_shading;
               key.blend[0].writes_memory_stats_only = !real_writes_memory;
               shader->info.apple9_undefined_colors = key.undefined_colors;
            } else {
               hk_apple9_remap_fs_outputs(nir, key.color_map);
            }
            /* The passthrough geometry shader produces the flat copies,
             * line-stipple TEX6/TEX7, and polygon-point FOGC. Keep them in
             * the producer layout passed to the fragment compiler so
             * apple9_strip_unread_fragment_inputs does not strip them.
             */
            for (unsigned j = 0; copies_from_gs && j < per_vertex.count; ++j) {
               unsigned source = VARYING_SLOT_VAR0 + per_vertex.location[j];
               if (!(nir->info.inputs_read & BITFIELD64_BIT(source)) &&
                   varyings.mask[source]) {
                  varyings.count -= util_bitcount(varyings.mask[source]);
                  varyings.mask[source] = 0;
                  varyings.group[source] = 0;
               }
            }
            for (unsigned j = 0; copies_from_gs && j < per_vertex.count; ++j) {
               for (unsigned k = 0; k < 3; ++k) {
                  unsigned slot =
                     hk_compile_per_vertex_copy_slot(per_vertex.copy_base[j], k);
                  if (!varyings.mask[slot]) {
                     varyings.mask[slot] = 0xf;
                     varyings.group[slot] = 1;
                     varyings.count += 4;
                  }
               }
            }
            for (unsigned m = 0; copies_from_gs && m < 2; ++m) {
               unsigned slot = per_vertex.bary[m];
               if (slot && !varyings.mask[slot]) {
                  varyings.mask[slot] = 0x7;
                  varyings.group[slot] = m ? 2 : 0;
                  varyings.count += 3;
               }
            }
            if (stipple.start &&
                (stipple_stage == MESA_SHADER_VERTEX ||
                 stipple_stage == MESA_SHADER_TESS_EVAL)) {
               if (!varyings.mask[stipple.start]) {
                  varyings.mask[stipple.start] = 0x7;
                  varyings.group[stipple.start] = 1;
                  varyings.count += 3;
               }
               if (!varyings.mask[stipple.axis]) {
                  varyings.mask[stipple.axis] = 0x3;
                  varyings.group[stipple.axis] = 2;
                  varyings.count += 2;
               }
            }
            if (polygon_face_slot && !varyings.mask[polygon_face_slot]) {
               varyings.mask[polygon_face_slot] = 0x1;
               varyings.group[polygon_face_slot] = 1;
               varyings.count += 1;
            }
            struct blob dump_nir;
            const bool dump = hk_apple9_fragment_dump_begin(nir, &dump_nir);
            hk_assert_final_push_mask(nir, shader);
            const bool has_prerast = last_prerast != MESA_SHADER_NONE;
            const bool compiled = agx_compile_apple9_fragment_mrt(
               nir, has_prerast ? &varyings : NULL, key.blend, key.targets,
               &shader->b, &reason);
            if (dump)
               hk_apple9_fragment_dump_end(&dump_nir, compiled, &varyings,
                                           key.blend, key.targets, &shader->b);
            if (!compiled) {
               reason = reason ?: "unsupported fragment NIR";
               if (native_mesh)
                  *native_unsupported = true;
               result = native_mesh ? VK_ERROR_FEATURE_NOT_PRESENT :
                  vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                            "Apple9 fragment: %s", reason);
               goto done;
            }
            /* hk's decision covers the backend's reasons for the shader as
             * written (before the statistics counter, which counts correctly
             * in merged waves) and adds sample shading and off-centre
             * interpolation, which the backend can't see. */
            shader->b.info.disable_tri_merging = !tri_merging;
         }
         shader->code_ptr = shader->b.binary;
         shader->code_size = shader->b.info.binary_size;
         hk_gpu_busy_shader_source(shader->code_ptr, shader->code_size, nir->info.name,
                                   nir->info.source_blake3, nir->info.stage);
         shader->info.clip_distance_array_size =
            nir->info.clip_distance_array_size;
         shader->info.cull_distance_array_size =
            nir->info.cull_distance_array_size;
         result = hk_copy_shader_constant_data(dev, shader, nir);
         if (result == VK_SUCCESS)
            result = hk_upload_shader(dev, shader);
         if (result != VK_SUCCESS)
            goto done;
      }
   }

done:
   if (result != VK_SUCCESS && agx_get_compiler_debug())
      fprintf(stderr, "Apple9 graphics compilation failed: %s\n",
              reason ?: vk_Result_to_str(result));
   if (lock)
      simple_mtx_unlock(lock);
   for (unsigned i = 0; i < count; ++i) {
      ralloc_free(infos[i].nir);
      if (result != VK_SUCCESS && out[i]) {
         hk_api_shader_destroy(&dev->vk, out[i], alloc);
         out[i] = NULL;
      } else if (out[i]) {
         hk_api_shader_init_constant_data(
            container_of(out[i], struct hk_api_shader, vk));
      }
   }
   return result;
}

/* Match the whole native compile, including both interfaces, code and constant
 * tables. This identity survives independent per-stage cache round trips; a
 * pointer or a process-local counter would not. Nothing owns another stage. */
static void
hk_apple9_identify_native_pair(uint32_t count, struct vk_shader **shaders,
                               const struct hk_apple9_graphics_key *key)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, key, sizeof(*key));
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj =
         container_of(shaders[i], struct hk_api_shader, vk);
      uint32_t stage = obj->vk.stage;
      _mesa_blake3_update(&ctx, &stage, sizeof(stage));
      if (stage == MESA_SHADER_TESS_EVAL)
         _mesa_blake3_update(&ctx, &obj->apple9_native_tess,
                             sizeof(obj->apple9_native_tess));
      hk_foreach_variant(obj, shader) {
         _mesa_blake3_update(&ctx, &shader->b.info.apple9_varyings,
                             sizeof(shader->b.info.apple9_varyings));
         _mesa_blake3_update(&ctx, &shader->code_size,
                             sizeof(shader->code_size));
         _mesa_blake3_update(&ctx, shader->code_ptr, shader->code_size);
         _mesa_blake3_update(&ctx, &shader->data_size,
                             sizeof(shader->data_size));
         if (shader->data_size)
            _mesa_blake3_update(&ctx, shader->data_ptr, shader->data_size);
      }
   }
   uint8_t stage_mask = 0;
   for (unsigned i = 0; i < count; ++i) {
      if (shaders[i]->stage == MESA_SHADER_GEOMETRY)
         stage_mask = key->native_stages & HK_APPLE9_NATIVE_GS_BIT;
      if (shaders[i]->stage == MESA_SHADER_TESS_EVAL)
         stage_mask = key->native_stages & HK_APPLE9_NATIVE_TES_BIT;
   }
   _mesa_blake3_update(&ctx, &stage_mask, sizeof(stage_mask));
   blake3_hash id;
   _mesa_blake3_final(&ctx, id);
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj =
         container_of(shaders[i], struct hk_api_shader, vk);
      obj->apple9_native_leaf = true;
      obj->apple9_native_stage_mask = stage_mask;
      memcpy(obj->apple9_native_pair_id, id, sizeof(id));
      if (obj->vk.stage == MESA_SHADER_MESH ||
          obj->vk.stage == MESA_SHADER_GEOMETRY ||
          obj->vk.stage == MESA_SHADER_TESS_EVAL) {
         obj->apple9_native_candidate = true;
         obj->apple9_native_cull_primitive =
            (hk_native_producer_main(obj)->info.apple9_native.primitive_broadcast &
             VARYING_BIT_CULL_PRIMITIVE) ||
            hk_native_producer_main(obj)->info.apple9_native.cull_primitive ||
            hk_native_producer_main(obj)->info.apple9_native.cull_distances;
      }
   }
}

/* Eager pipelines own both realizations. Draw-time variants request one path:
 * native compiles only its own programs, falling back to emulation if the
 * native backend rejects the body. Independent NIR keeps either rejection
 * from leaving a half-lowered fallback behind. */
/* Whether a vertex shader's pipeline disables vertex input robustness, so
 * its attribute fetches may skip bounds checks. robustBufferAccess,
 * robustness2 and VK_EXT_pipeline_robustness all resolve into the stage's
 * vertex_inputs behaviour; shader objects take the device default. */
bool
hk_apple9_vertex_unbounded(const struct vk_shader_compile_info *info)
{
   return info->robustness &&
          info->robustness->vertex_inputs ==
             VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
}

VkResult
hk_apple9_compile_graphics_mode(struct hk_device *dev, uint32_t count,
                                struct vk_shader_compile_info *infos,
                                const struct vk_graphics_pipeline_state *state,
                                const struct hk_apple9_graphics_key *key_in,
                                const uint64_t *fs_interp,
                                const VkAllocationCallbacks *alloc,
                                struct vk_shader **out, bool native_only)
{
   assert(count <= HK_APPLE9_GFX_STAGES);
   struct hk_apple9_graphics_key key;
   if (key_in) {
      key = *key_in;
   } else {
      hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage == MESA_SHADER_VERTEX)
            key.vertex.unbounded = hk_apple9_vertex_unbounded(&infos[i]);
      }
   }

   nir_shader *mesh = NULL, *gs = NULL, *tes = NULL;
   bool task = false, tess = false, per_vertex = false, fs_inputs = false;
   bool producer_effects = false;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         producer_effects |= infos[i].nir->info.writes_memory || infos[i].nir->xfb_info;
      if (infos[i].stage == MESA_SHADER_MESH)
         mesh = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_GEOMETRY)
         gs = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_TESS_EVAL)
         tes = infos[i].nir;
      tess |= infos[i].stage == MESA_SHADER_TESS_CTRL ||
              infos[i].stage == MESA_SHADER_TESS_EVAL;
      task |= infos[i].stage == MESA_SHADER_TASK;
      if (infos[i].stage == MESA_SHADER_FRAGMENT)
         fs_inputs |= infos[i].nir->info.inputs_read & ~VARYING_BIT_POS;
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         continue;
      nir_foreach_function_impl(impl, infos[i].nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               per_vertex |= op == nir_intrinsic_load_input_vertex ||
                             op == nir_intrinsic_load_per_vertex_input ||
                             hk_is_barycentric_coord(op);
            }
         }
      }
   }

   bool native = false;
   VkResult result = VK_SUCCESS;
   const char *reason = NULL;
   key.native_task_bridge = mesh && task && hk_apple9_native_task_bridge_enabled();
   if (mesh && (!task || key.native_task_bridge) &&
       !per_vertex && !key.polygon_points)
      result = hk_apple9_native_mesh_check(mesh, &key, &native, &reason);
   else if (gs && !mesh && !task && !tess && !per_vertex &&
            (!fs_inputs || hk_apple9_native_gs_enabled()) &&
            !producer_effects &&
            !key.polygon_points && !key.line_stipple)
      result = hk_apple9_native_gs_check(gs, &key, &native, &reason);
   else if (tes && !gs && !mesh && !task && !per_vertex &&
            (!fs_inputs || hk_apple9_native_tess_enabled()) &&
            !producer_effects &&
            !key.polygon_points && !key.line_stipple)
      result = hk_apple9_native_tess_check(tes, &key, &native, &reason);

   struct vk_shader_compile_info native_infos[HK_APPLE9_GFX_STAGES] = {0};
   unsigned cloned = 0;
   if (result == VK_SUCCESS && native) {
      for (; cloned < count; ++cloned) {
         native_infos[cloned] = infos[cloned];
         native_infos[cloned].nir = nir_shader_clone(NULL, infos[cloned].nir);
         if (!native_infos[cloned].nir) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
         }
      }
   }
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < cloned; ++i)
         ralloc_free(native_infos[i].nir);
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
      memset(out, 0, count * sizeof(*out));
      return vk_error(dev, result);
   }

   memset(out, 0, count * sizeof(*out));
   if (!native_only || !native) {
      result = hk_apple9_compile_graphics_path(
         dev, count, infos, state, &key, fs_interp, alloc, out, false, NULL);
      if (result != VK_SUCCESS || !native) {
         for (unsigned i = 0; i < cloned; ++i)
            ralloc_free(native_infos[i].nir);
         return result;
      }
   }

   struct vk_shader *native_out[HK_APPLE9_GFX_STAGES] = {0};
   bool native_unsupported = false;
   result = hk_apple9_compile_graphics_path(dev, count, native_infos, state,
                                           &key, fs_interp, alloc,
                                           native_out, true,
                                           &native_unsupported);
   if (native_unsupported) {
      assert(result == VK_ERROR_FEATURE_NOT_PRESENT);
      /* Allocation failures remain real Vulkan errors. Only a native-only
       * capability failure compiles/uses the emulated realization. */
      return native_only ? hk_apple9_compile_graphics_path(
                              dev, count, infos, state, &key, fs_interp,
                              alloc, out, false, NULL)
                         : VK_SUCCESS;
   }
   if (native_only) {
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
   }
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < count; ++i) {
         if (out[i])
            hk_api_shader_destroy(&dev->vk, out[i], alloc);
         out[i] = NULL;
      }
      return result;
   }

   hk_apple9_identify_native_pair(count, native_out, &key);
   if (native_only) {
      memcpy(out, native_out, count * sizeof(*out));
      return VK_SUCCESS;
   }
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj = container_of(out[i], struct hk_api_shader, vk);
      obj->apple9_native = container_of(native_out[i], struct hk_api_shader, vk);
      obj->apple9_native_candidate = infos[i].stage == MESA_SHADER_MESH ||
                                     infos[i].stage == MESA_SHADER_GEOMETRY ||
                                     infos[i].stage == MESA_SHADER_TESS_EVAL;
      obj->apple9_native_cull_primitive =
         obj->apple9_native->apple9_native_cull_primitive;
      obj->apple9_native_stage_mask = obj->apple9_native->apple9_native_stage_mask;
      hk_api_shader_init_constant_data(obj);
   }
   return VK_SUCCESS;
}

VkResult
hk_apple9_compile_graphics(struct hk_device *dev, uint32_t count,
                           struct vk_shader_compile_info *infos,
                           const struct vk_graphics_pipeline_state *state,
                           const struct hk_apple9_graphics_key *key_in,
                           const uint64_t *fs_interp,
                           const VkAllocationCallbacks *alloc,
                           struct vk_shader **out)
{
   return hk_apple9_compile_graphics_mode(dev, count, infos, state, key_in,
                                          fs_interp, alloc, out, false);
}

/* Even without an API fragment shader, the pipeline sample mask controls
 * depth/stencil writes and occlusion. Keep the empty program independent of
 * pipeline state by reading the normal draw root table.
 */
nir_shader *
hk_apple9_null_fs_nir(bool sample_mask)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                 &agx_nir_options, "empty FS");
   /* With a full API sample mask the program has no work at all: no
    * coverage, no root reads, as Metal's empty fragment program. */
   if (!sample_mask)
      return b.shader;
   /* Read the per-draw API sample mask from AGX_APPLE9_GRAPHICS_SYSVAL_BINDING
    * (slot 0) so null_fs shares slot 0 cleanly with linked stipple prologs. */
   nir_def *mask = nir_u2u16(&b, nir_load_ubo(&b, 1, 32,
      nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(&b, AGX_APPLE9_API_SAMPLE_MASK_OFFSET),
      .align_mul = 4, .range_base = AGX_APPLE9_API_SAMPLE_MASK_OFFSET,
      .range = 4));
   nir_demote_samples(&b, nir_inot(&b, mask));
   return b.shader;
}

VkResult
hk_apple9_create_null_fs(struct hk_device *dev,
                         const VkAllocationCallbacks *alloc, bool sample_mask,
                         struct hk_api_shader **out)
{
   struct vk_shader_compile_info info = {
      .nir = hk_apple9_null_fs_nir(sample_mask),
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_FRAGMENT,
   };
   struct vk_shader *shader = NULL;
   VkResult result =
      hk_apple9_compile_graphics(dev, 1, &info, NULL, NULL, NULL, alloc, &shader);
   *out = shader ? container_of(shader, struct hk_api_shader, vk) : NULL;
   return result;
}

/* Internal geometry shaders, such as the transform feedback passthrough,
 * rasterize into the bound fragment shader's interpolation layout.
 */
VkResult
hk_apple9_compile_internal_gs(struct hk_device *dev, nir_shader *nir,
                              const uint64_t fs_interp[2], bool cull_mask,
                              struct hk_api_shader **out)
{
   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };
   hk_lower_nir(dev, nir, &rs, false, 0, NULL, 0);
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_GEOMETRY,
      .nir = nir,
      .robustness = &rs,
   };
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key_device(&key, NULL, hk_device_physical(dev));
   /* The SW-VS/SW-TES feeding a passthrough GS already applied clip-space
    * depth mapping; passthrough GS forwards scalarized POS untouched. */
   key.clip_negative_one_to_one = false;
   key.clip_negative_one_to_one_dynamic = false;
   key.gs_cull_planes = !cull_mask;
   struct vk_shader *shader = NULL;
   VkResult result =
      hk_apple9_compile_graphics(dev, 1, &info, NULL, &key, fs_interp, NULL, &shader);
   *out = shader ? container_of(shader, struct hk_api_shader, vk) : NULL;
   return result;
}

/* Compile an internal mesh-rasterization vertex shader (for hk_meta_shader
 * cache variants: per_vertex, line_stipple, polygon_points). Takes ownership
 * of nir.
 */
VkResult
hk_apple9_compile_internal_mesh_rast(struct hk_device *dev, nir_shader *nir,
                                     const struct hk_mesh_info *info,
                                     const uint64_t fs_interp[2],
                                     bool polygon_points,
                                     struct hk_api_shader **out)
{
   size_t size = sizeof(struct hk_api_shader) +
                 hk_num_variants(MESA_SHADER_VERTEX) * sizeof(struct hk_shader);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, MESA_SHADER_VERTEX, NULL, size);
   if (!obj) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   hk_foreach_variant(obj, variant) {
      variant->info.stage = MESA_SHADER_VERTEX;
      variant->info.mesh = *info;
      variant->info.mesh.polygon_points = polygon_points;
      variant->b.info.stats.instrs = 1;
#ifndef NDEBUG
      variant->debug_owner = obj;
#endif
   }
   obj->pre_raster_reads = HK_READS_INTERNAL;

   struct hk_shader *rast = &obj->variants[HK_VS_VARIANT_HW];
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_VERTEX);
   if (!polygon_points) {
      bool dynamic_clip = true;
      NIR_PASS(_, nir, nir_lower_clip_halfz_dynamic);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_clip_z_coeff,
               nir_metadata_control_flow, &dynamic_clip);
   }

   struct agx_apple9_vertex_layout layout = {
      .outputs_flat = (fs_interp ? fs_interp[0] : 0) |
                      VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT |
                      VARYING_BIT_FOGC | VARYING_BIT_TEX6,
      .outputs_linear = (fs_interp ? fs_interp[1] : 0) | VARYING_BIT_TEX7,
      .clip_distance_enable = 0xff,
      .rasterize_points = info->prim == MESA_PRIM_POINTS || polygon_points,
   };
   layout.ignore_point_size = !layout.rasterize_points;

   const char *reason = NULL;
   VkResult result = VK_SUCCESS;
   if (agx_compile_apple9_vertex_inputs(nir, &layout, &rast->b, &reason))
      result = hk_apple9_finish_shader(dev, rast, nir);
   else
      result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                         "Apple9 mesh rast meta: %s",
                         reason ? reason : "unsupported NIR");
   ralloc_free(nir);
   if (result != VK_SUCCESS) {
      hk_api_shader_destroy(&dev->vk, &obj->vk, NULL);
      *out = NULL;
      return result;
   }
   hk_api_shader_init_constant_data(obj);
   *out = obj;
   return VK_SUCCESS;
}

