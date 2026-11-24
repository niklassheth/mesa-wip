/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "gravity/compiler/agx_apple9_encoding.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "gravity/compiler/agx_nir.h"
#include "gravity/lib/agx_apple9.h"
#include "gravity/lib/agx_nir_apple9_tilebuffer.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"
#include "util/u_prim.h"
#include "util/u_debug.h"
#include "vk_blend.h"
#include "vk_util.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9_mesh.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_image.h"
#include "hk_image_view.h"

static struct agx_ptr
hk_apple9_usc_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_usc_alloc(data, size, alignment);
}

static void
hk_apple9_copy_vs_prolog_iface(struct agx_apple9_vs_prolog_iface *dst,
                               const struct agx_apple9_vs_prolog_iface *src)
{
   dst->attribs_read = src->attribs_read;
   dst->integer_attribs = src->integer_attribs;
   memcpy(dst->attrib_mask, src->attrib_mask, sizeof(dst->attrib_mask));
   dst->uses_prolog = src->uses_prolog;
}

static void
hk_apple9_copy_fs_epilog_iface(struct agx_apple9_fs_epilog_iface *dst,
                               const struct agx_apple9_fs_epilog_iface *src)
{
   dst->outputs_written = src->outputs_written;
   dst->writes_src1 = src->writes_src1;
   dst->writes_frag_color = src->writes_frag_color;
   dst->writes_depth = src->writes_depth;
   dst->writes_stencil = src->writes_stencil;
   dst->has_tile_lock = src->has_tile_lock;
   dst->has_tile_preamble = src->has_tile_preamble;
   dst->early_fragment_tests = src->early_fragment_tests;
   dst->has_killed_mask = src->has_killed_mask;
   dst->per_sample = src->per_sample;
   dst->main_defers_zs_test = src->main_defers_zs_test;
   dst->uses_epilog = src->uses_epilog;
}

/* A tile helper program depends only on its key. Each render binds its own
 * clear values, reload textures and store images through tables. */
struct hk_apple9_tile_helper {
   struct agx_apple9_tilebuffer_key key;
   struct agx_bo *bo;
   struct agx_shader_info info;
   struct agx_apple9_stage_launch_cache launch;
};

#define HK_DEFINE_KEY_HASH_EQ(prefix, key_type)                                \
   static uint32_t                                                             \
   prefix##_hash(const void *key)                                              \
   {                                                                           \
      return _mesa_hash_data(key, sizeof(key_type));                           \
   }                                                                           \
   static bool                                                                 \
   prefix##_equal(const void *a, const void *b)                                \
   {                                                                           \
      return !memcmp(a, b, sizeof(key_type));                                  \
   }

HK_DEFINE_KEY_HASH_EQ(hk_apple9_tile_helper, struct agx_apple9_tilebuffer_key)

static struct hk_apple9_tile_helper *
hk_apple9_tile_helper(struct hk_device *dev,
                      const struct agx_apple9_tilebuffer_key *key,
                      const struct agx_apple9_blend *blend)
{
   struct hk_apple9_tile_helper *helper = NULL;
   struct hk_apple9_tile_helper *created = NULL;
   struct agx_shader_part part = {0};
   nir_shader *nir = NULL;
   const char *reason = NULL;
   simple_mtx_lock(&dev->apple9_tile_helpers.lock);
   if (!dev->apple9_tile_helpers.ht) {
      dev->apple9_tile_helpers.ht = _mesa_hash_table_create(
         NULL, hk_apple9_tile_helper_hash, hk_apple9_tile_helper_equal);
   }
   struct hash_table *ht = dev->apple9_tile_helpers.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      helper = entry->data;
      goto done;
   }
   if (!ht)
      goto done;

   nir = agx_nir_apple9_tilebuffer(&agx_nir_options, key);
   if (!agx_compile_apple9_fragment_mrt(nir, NULL, blend, key->count, &part,
                                        &reason)) {
      fprintf(stderr, "Apple9 tile helper: %s\n",
              reason ?: "unsupported shader");
      goto done;
   }
   created = calloc(1, sizeof(*created));
   if (!created)
      goto done;
   memcpy(&created->key, key, sizeof(*key));
   created->info = part.info;
   created->bo = agx_bo_create(&dev->dev, part.info.binary_size, 0,
                               AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                               "Vulkan Apple9 tile helper");
   if (!created->bo) {
      free(created);
      goto done;
   }
   void *map = agx_bo_map(created->bo);
   if (!map) {
      agx_bo_unreference(&dev->dev, created->bo);
      free(created);
      goto done;
   }
   memcpy(map, part.binary, part.info.binary_size);
   agx_bo_note_cpu_write(created->bo, 0, part.info.binary_size);
   agx_apple9_stage_launch_cache_init(&created->launch);
   if (!_mesa_hash_table_insert(ht, &created->key, created)) {
      agx_apple9_stage_launch_cache_finish(&created->launch);
      agx_bo_unreference(&dev->dev, created->bo);
      free(created);
      goto done;
   }
   helper = created;

done:
   free(part.binary);
   ralloc_free(nir);
   simple_mtx_unlock(&dev->apple9_tile_helpers.lock);
   return helper;
}

void
hk_apple9_tile_helpers_finish(struct hk_device *dev)
{
   if (!dev->apple9_tile_helpers.ht)
      return;
   hash_table_foreach(dev->apple9_tile_helpers.ht, entry) {
      struct hk_apple9_tile_helper *helper = entry->data;
      agx_apple9_stage_launch_cache_finish(&helper->launch);
      agx_bo_unreference(&dev->dev, helper->bo);
      free(helper);
   }
   _mesa_hash_table_destroy(dev->apple9_tile_helpers.ht, NULL);
   dev->apple9_tile_helpers.ht = NULL;
}

struct hk_apple9_vs_prolog {
   struct agx_apple9_vertex_prolog_key key;
   struct agx_shader_body *body;
   uint16_t publication_count;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_vs_prolog, struct agx_apple9_vertex_prolog_key)

static struct hk_apple9_vs_prolog *
hk_apple9_get_vs_prolog(struct hk_device *dev,
                        const struct agx_apple9_vertex_prolog_key *key)
{
   simple_mtx_lock(&dev->apple9_vs_prologs.lock);
   if (!dev->apple9_vs_prologs.ht) {
      dev->apple9_vs_prologs.ht = _mesa_hash_table_create(
         NULL, hk_apple9_vs_prolog_hash, hk_apple9_vs_prolog_equal);
   }
   struct hash_table *ht = dev->apple9_vs_prologs.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      struct hk_apple9_vs_prolog *found = entry->data;
      simple_mtx_unlock(&dev->apple9_vs_prologs.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_vs_prologs.lock);
   if (!ht)
      return NULL;

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_vertex_prolog_key(key, &part, &reason)) {
      fprintf(stderr, "Apple9 vertex prolog: %s\n",
              reason ?: "unsupported vertex input");
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_vs_prolog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, key, sizeof(*key));
   created->body = body;
   created->publication_count = part.info.apple9_publication_count;
   created->scratch_size = part.info.scratch_size;
   created->atomic_frame_size = part.info.apple9_atomic_frame_size;

   simple_mtx_lock(&dev->apple9_vs_prologs.lock);
   entry = _mesa_hash_table_search(dev->apple9_vs_prologs.ht, key);
   if (entry) {
      struct hk_apple9_vs_prolog *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_vs_prologs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_vs_prologs.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_vs_prologs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_vs_prologs.lock);
   return created;
}

void
hk_apple9_vs_prologs_finish(struct hk_device *dev)
{
   if (!dev->apple9_vs_prologs.ht)
      return;
   hash_table_foreach(dev->apple9_vs_prologs.ht, entry) {
      struct hk_apple9_vs_prolog *prolog = entry->data;
      agx_shader_body_unreference(&dev->dev, prolog->body);
      free(prolog);
   }
   _mesa_hash_table_destroy(dev->apple9_vs_prologs.ht, NULL);
   dev->apple9_vs_prologs.ht = NULL;
}

struct hk_apple9_fs_epilog {
   struct agx_apple9_fs_epilog_key key;
   struct agx_shader_body *body;
   uint16_t publication_count;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
   bool reads_tile;
   bool uses_discard;
   bool writes_depth;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_fs_epilog, struct agx_apple9_fs_epilog_key)

static struct hk_apple9_fs_epilog *
hk_apple9_get_fs_epilog(struct hk_device *dev,
                        const struct agx_apple9_fs_epilog_key *key)
{
   simple_mtx_lock(&dev->apple9_fs_epilogs.lock);
   if (!dev->apple9_fs_epilogs.ht) {
      dev->apple9_fs_epilogs.ht = _mesa_hash_table_create(
         NULL, hk_apple9_fs_epilog_hash, hk_apple9_fs_epilog_equal);
   }
   struct hash_table *ht = dev->apple9_fs_epilogs.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      struct hk_apple9_fs_epilog *found = entry->data;
      simple_mtx_unlock(&dev->apple9_fs_epilogs.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_fs_epilogs.lock);
   if (!ht)
      return NULL;

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_fragment_epilog_key(key, &part, &reason)) {
      fprintf(stderr, "Apple9 fragment epilog: %s\n",
              reason ?: "unsupported fragment blend/format");
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_fs_epilog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, key, sizeof(*key));
   created->body = body;
   created->publication_count = part.info.apple9_publication_count;
   created->scratch_size = part.info.scratch_size;
   created->atomic_frame_size = part.info.apple9_atomic_frame_size;
   created->reads_tile = part.info.apple9_reads_tile;
   created->uses_discard = part.info.apple9_uses_discard;
   created->writes_depth =
      part.info.depth_layout != FRAG_DEPTH_LAYOUT_UNCHANGED;

   simple_mtx_lock(&dev->apple9_fs_epilogs.lock);
   entry = _mesa_hash_table_search(dev->apple9_fs_epilogs.ht, key);
   if (entry) {
      struct hk_apple9_fs_epilog *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_fs_epilogs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_fs_epilogs.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_fs_epilogs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_fs_epilogs.lock);
   return created;
}

void
hk_apple9_fs_epilogs_finish(struct hk_device *dev)
{
   if (!dev->apple9_fs_epilogs.ht)
      return;
   hash_table_foreach(dev->apple9_fs_epilogs.ht, entry) {
      struct hk_apple9_fs_epilog *epilog = entry->data;
      agx_shader_body_unreference(&dev->dev, epilog->body);
      free(epilog);
   }
   _mesa_hash_table_destroy(dev->apple9_fs_epilogs.ht, NULL);
   dev->apple9_fs_epilogs.ht = NULL;
}

struct hk_apple9_cull_prolog {
   struct agx_apple9_cull_prolog_key key;
   struct agx_shader_body *body;
   uint64_t program_id;
   uint16_t publication_count;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
   struct agx_apple9_stage_launch_cache launch_cache;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_cull_prolog, struct agx_apple9_cull_prolog_key)

static struct hk_apple9_cull_prolog *
hk_apple9_get_cull_prolog(struct hk_device *dev,
                          const struct agx_apple9_cull_prolog_key *key)
{
   simple_mtx_lock(&dev->apple9_cull_prologs.lock);
   if (!dev->apple9_cull_prologs.ht) {
      dev->apple9_cull_prologs.ht = _mesa_hash_table_create(
         NULL, hk_apple9_cull_prolog_hash, hk_apple9_cull_prolog_equal);
   }
   struct hash_table *ht = dev->apple9_cull_prologs.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      struct hk_apple9_cull_prolog *found = entry->data;
      simple_mtx_unlock(&dev->apple9_cull_prologs.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_cull_prologs.lock);
   if (!ht)
      return NULL;

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_cull_prolog(key, &part, &reason)) {
      fprintf(stderr, "Apple9 cull prolog: %s\n",
              reason ?: "unsupported cull distance prolog");
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_cull_prolog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, key, sizeof(*key));
   created->body = body;
   created->program_id = p_atomic_inc_return(&dev->apple9_last_program_id);
   created->publication_count = part.info.apple9_publication_count;
   created->scratch_size = part.info.scratch_size;
   created->atomic_frame_size = part.info.apple9_atomic_frame_size;
   agx_apple9_stage_launch_cache_init(&created->launch_cache);

   simple_mtx_lock(&dev->apple9_cull_prologs.lock);
   entry = _mesa_hash_table_search(dev->apple9_cull_prologs.ht, key);
   if (entry) {
      struct hk_apple9_cull_prolog *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_cull_prologs.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_cull_prologs.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_cull_prologs.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_cull_prologs.lock);
   return created;
}

void
hk_apple9_cull_prologs_finish(struct hk_device *dev)
{
   if (!dev->apple9_cull_prologs.ht)
      return;
   hash_table_foreach(dev->apple9_cull_prologs.ht, entry) {
      struct hk_apple9_cull_prolog *prolog = entry->data;
      agx_apple9_stage_launch_cache_finish(&prolog->launch_cache);
      agx_shader_body_unreference(&dev->dev, prolog->body);
      free(prolog);
   }
   _mesa_hash_table_destroy(dev->apple9_cull_prologs.ht, NULL);
   dev->apple9_cull_prologs.ht = NULL;
}

struct hk_apple9_stipple_prolog {
   struct agx_apple9_stipple_prolog_key key;
   struct agx_shader_body *body;
   uint64_t program_id;
   uint16_t publication_count;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
   struct agx_apple9_stage_launch_cache launch_cache;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_stipple_prolog, struct agx_apple9_stipple_prolog_key)

static struct hk_apple9_stipple_prolog *
hk_apple9_get_stipple_prolog(struct hk_device *dev,
                             const struct agx_apple9_stipple_prolog_key *key)
{
   simple_mtx_lock(&dev->apple9_stipple_prologs.lock);
   if (!dev->apple9_stipple_prologs.ht) {
      dev->apple9_stipple_prologs.ht = _mesa_hash_table_create(
         NULL, hk_apple9_stipple_prolog_hash, hk_apple9_stipple_prolog_equal);
   }
   struct hash_table *ht = dev->apple9_stipple_prologs.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      struct hk_apple9_stipple_prolog *found = entry->data;
      simple_mtx_unlock(&dev->apple9_stipple_prologs.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_stipple_prologs.lock);
   if (!ht)
      return NULL;

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_stipple_prolog(key, &part, &reason)) {
      fprintf(stderr, "Apple9 stipple prolog: %s\n",
              reason ?: "unsupported line stipple prolog");
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_stipple_prolog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, key, sizeof(*key));
   created->body = body;
   created->program_id = p_atomic_inc_return(&dev->apple9_last_program_id);
   created->publication_count = part.info.apple9_publication_count;
   created->scratch_size = part.info.scratch_size;
   created->atomic_frame_size = part.info.apple9_atomic_frame_size;
   agx_apple9_stage_launch_cache_init(&created->launch_cache);

   simple_mtx_lock(&dev->apple9_stipple_prologs.lock);
   entry = _mesa_hash_table_search(dev->apple9_stipple_prologs.ht, key);
   if (entry) {
      struct hk_apple9_stipple_prolog *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_stipple_prologs.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_stipple_prologs.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_stipple_prologs.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_stipple_prologs.lock);
   return created;
}

void
hk_apple9_stipple_prologs_finish(struct hk_device *dev)
{
   if (!dev->apple9_stipple_prologs.ht)
      return;
   hash_table_foreach(dev->apple9_stipple_prologs.ht, entry) {
      struct hk_apple9_stipple_prolog *prolog = entry->data;
      agx_apple9_stage_launch_cache_finish(&prolog->launch_cache);
      agx_shader_body_unreference(&dev->dev, prolog->body);
      free(prolog);
   }
   _mesa_hash_table_destroy(dev->apple9_stipple_prologs.ht, NULL);
   dev->apple9_stipple_prologs.ht = NULL;
}

struct hk_apple9_stats_prolog {
   struct agx_shader_body *body;
   uint16_t publication_count;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
};

static struct hk_apple9_stats_prolog *
hk_apple9_get_stats_prolog(struct hk_device *dev)
{
   simple_mtx_lock(&dev->apple9_stats_prolog.lock);
   if (dev->apple9_stats_prolog.prolog) {
      struct hk_apple9_stats_prolog *found = dev->apple9_stats_prolog.prolog;
      simple_mtx_unlock(&dev->apple9_stats_prolog.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_stats_prolog.lock);

   nir_shader *nir = hk_apple9_build_fs_stats_prolog_nir();
   struct agx_apple9_blend blend = {
      .use_epilog = true,
      .dynamic_samples = true,
      .writes_memory_stats_only = true,
   };
   struct agx_shader_part part = {0};
   const char *reason = NULL;
   bool ok = agx_compile_apple9_fragment_mrt(nir, NULL, &blend, 0, &part, &reason);
   ralloc_free(nir);
   if (!ok) {
      fprintf(stderr, "Apple9 FS stats prolog: %s\n",
              reason ?: "compilation failed");
      free(part.binary);
      return NULL;
   }
   if (part.info.apple9_preamble_size != 0 ||
       part.info.apple9_resource_count != 2 ||
       part.info.apple9_resource_binding[0] !=
          AGX_APPLE9_GRAPHICS_SYSVAL_BINDING ||
       part.info.apple9_resource_binding[1] != 0) {
      fprintf(stderr,
              "Apple9 FS stats prolog ABI mismatch: preamble=%u res=%u (%u, %u)\n",
              part.info.apple9_preamble_size, part.info.apple9_resource_count,
              part.info.apple9_resource_binding[0],
              part.info.apple9_resource_binding[1]);
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_stats_prolog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   created->body = body;
   created->publication_count = part.info.apple9_publication_count;
   created->scratch_size = part.info.scratch_size;
   created->atomic_frame_size = part.info.apple9_atomic_frame_size;

   simple_mtx_lock(&dev->apple9_stats_prolog.lock);
   if (dev->apple9_stats_prolog.prolog) {
      struct hk_apple9_stats_prolog *existing = dev->apple9_stats_prolog.prolog;
      simple_mtx_unlock(&dev->apple9_stats_prolog.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   dev->apple9_stats_prolog.prolog = created;
   simple_mtx_unlock(&dev->apple9_stats_prolog.lock);
   return created;
}

void
hk_apple9_stats_prolog_finish(struct hk_device *dev)
{
   struct hk_apple9_stats_prolog *prolog = dev->apple9_stats_prolog.prolog;
   if (!prolog)
      return;
   agx_shader_body_unreference(&dev->dev, prolog->body);
   free(prolog);
   dev->apple9_stats_prolog.prolog = NULL;
}

struct hk_apple9_sw_vs_prolog {
   struct agx_apple9_sw_vs_prolog_key key;
   struct agx_shader_body *body;
   struct agx_apple9_compute_profile profile;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_sw_vs_prolog, struct agx_apple9_sw_vs_prolog_key)

static struct hk_apple9_sw_vs_prolog *
hk_apple9_get_sw_vs_prolog(struct hk_device *dev,
                           const struct agx_apple9_sw_vs_prolog_key *key)
{
   simple_mtx_lock(&dev->apple9_sw_vs_prologs.lock);
   if (!dev->apple9_sw_vs_prologs.ht) {
      dev->apple9_sw_vs_prologs.ht = _mesa_hash_table_create(
         NULL, hk_apple9_sw_vs_prolog_hash, hk_apple9_sw_vs_prolog_equal);
   }
   struct hash_table *ht = dev->apple9_sw_vs_prologs.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      struct hk_apple9_sw_vs_prolog *found = entry->data;
      simple_mtx_unlock(&dev->apple9_sw_vs_prologs.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_sw_vs_prologs.lock);
   if (!ht)
      return NULL;

   struct agx_shader_part part = {0};
   struct agx_apple9_compute_profile profile = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_sw_vs_prolog_key(key, &part, &profile, &reason)) {
      fprintf(stderr, "Apple9 SW-VS prolog: %s\n",
              reason ?: "unsupported SW vertex input");
      free(part.binary);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = part.binary,
      .code_size = part.info.binary_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(part.binary);
   if (!body)
      return NULL;

   struct hk_apple9_sw_vs_prolog *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, key, sizeof(*key));
   created->body = body;
   created->profile = profile;

   simple_mtx_lock(&dev->apple9_sw_vs_prologs.lock);
   entry = _mesa_hash_table_search(dev->apple9_sw_vs_prologs.ht, key);
   if (entry) {
      struct hk_apple9_sw_vs_prolog *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_sw_vs_prologs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_sw_vs_prologs.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_sw_vs_prologs.lock);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_sw_vs_prologs.lock);
   return created;
}

void
hk_apple9_sw_vs_prologs_finish(struct hk_device *dev)
{
   if (!dev->apple9_sw_vs_prologs.ht)
      return;
   hash_table_foreach(dev->apple9_sw_vs_prologs.ht, entry) {
      struct hk_apple9_sw_vs_prolog *prolog = entry->data;
      agx_shader_body_unreference(&dev->dev, prolog->body);
      free(prolog);
   }
   _mesa_hash_table_destroy(dev->apple9_sw_vs_prologs.ht, NULL);
   dev->apple9_sw_vs_prologs.ht = NULL;
}

#define HK_APPLE9_LINKED_ENTRY_CACHE_SIZE 4096
#define HK_APPLE9_LINKED_STUB_5_MAX_BYTES 82

static size_t
hk_apple9_build_linked_stub_5(uint8_t *out, size_t capacity, uint64_t code_va,
                              uint64_t s1, uint64_t s2, uint64_t s3,
                              uint64_t s4, uint64_t s5)
{
   if (!out || !s1 || !s2 || !s3 || !s4 || !s5 ||
       ((code_va | s1 | s2 | s3 | s4 | s5) & 1) ||
       capacity < HK_APPLE9_LINKED_STUB_5_MAX_BYTES)
      return 0;

   size_t pos = 0;
   const uint8_t push[4] = {0x0f, 0x05, 0x54, 0x1a};
   memcpy(out + pos, push, 4);
   pos += 4;

   const uint64_t calls[4] = {s1, s2, s3, s4};
   for (unsigned i = 0; i < 4; ++i) {
      int64_t disp = (int64_t)calls[i] - (int64_t)(code_va + pos);
      if (disp < -(int64_t)UINT32_MAX || disp > UINT32_MAX ||
          !agx_apple9_encode_call(out + pos, 0, disp))
         return 0;
      pos += 10;
      if (!agx_apple9_encode_exec_pop(out + pos, 2))
         return 0;
      pos += 6;
   }

   int64_t jmp_disp = (int64_t)s5 - (int64_t)(code_va + pos);
   if (jmp_disp < -(int64_t)UINT32_MAX || jmp_disp > UINT32_MAX ||
       !agx_apple9_encode_branch(out + pos, true, jmp_disp))
      return 0;
   pos += 10;
   const uint32_t stop = 0x0e;
   memcpy(out + pos, &stop, 4);
   pos += 4;
   return pos;
}

struct hk_apple9_linked_entry_key {
   uint64_t first_addr;
   uint64_t second_addr;
   uint64_t third_addr;
   uint64_t fourth_addr;
   uint64_t fifth_addr;
   uint64_t preamble_addr;
   uint32_t frame_extent;
   uint16_t publication_count;
   uint8_t stage_kind; /* 0 = vertex, 1 = fragment, 2 = compute */
};

struct hk_apple9_linked_entry {
   struct hk_apple9_linked_entry_key key;
   struct agx_shader_body body;
   struct agx_shader_body *fourth_ref;
   struct agx_shader_body *fifth_ref;
   struct agx_shader_body *stub_ref;
   uint64_t program_id;
   struct agx_apple9_stage_launch_cache launch_cache;
   struct list_head lru_link;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_linked_entry, struct hk_apple9_linked_entry_key)

static void
hk_apple9_linked_entry_destroy(struct agx_device *dev,
                               struct agx_shader_body *body)
{
   struct hk_apple9_linked_entry *entry =
      container_of(body, struct hk_apple9_linked_entry, body);
   agx_shader_body_unreference(dev, entry->body.first_ref);
   agx_shader_body_unreference(dev, entry->body.second_ref);
   agx_shader_body_unreference(dev, entry->body.third_ref);
   agx_shader_body_unreference(dev, entry->fourth_ref);
   agx_shader_body_unreference(dev, entry->fifth_ref);
   agx_shader_body_unreference(dev, entry->stub_ref);
   agx_apple9_stage_launch_cache_finish(&entry->launch_cache);
   free(entry);
}

/* Returns a referenced linked entry (2-stage when third == NULL, 3-stage when
 * third != NULL and fourth == NULL, or a slab-allocated 4/5-stage stub when
 * fourth != NULL); caller must unreference &entry->body once pinned or unused. */
static struct hk_apple9_linked_entry *
hk_apple9_get_linked_entry(struct hk_device *dev,
                           struct agx_shader_body *first,
                           struct agx_shader_body *second,
                           struct agx_shader_body *third,
                           struct agx_shader_body *fourth,
                           struct agx_shader_body *fifth,
                           uint64_t preamble_addr, uint32_t frame_extent,
                           uint16_t publication_count, uint8_t stage_kind)
{
   struct hk_apple9_linked_entry_key key;
   memset(&key, 0, sizeof(key));
   key.first_addr = first->addr;
   key.second_addr = second->addr;
   key.third_addr = third ? third->addr : 0;
   key.fourth_addr = fourth ? fourth->addr : 0;
   key.fifth_addr = fifth ? fifth->addr : 0;
   key.preamble_addr = preamble_addr;
   key.frame_extent = frame_extent;
   key.publication_count = publication_count;
   key.stage_kind = stage_kind;

   struct hk_apple9_linked_entry *result = NULL;
   struct hk_apple9_linked_entry *evicted = NULL;

   simple_mtx_lock(&dev->apple9_linked_entries.lock);
   if (!dev->apple9_linked_entries.ht) {
      dev->apple9_linked_entries.ht = _mesa_hash_table_create(
         NULL, hk_apple9_linked_entry_hash, hk_apple9_linked_entry_equal);
      list_inithead(&dev->apple9_linked_entries.lru);
   }
   struct hash_table *ht = dev->apple9_linked_entries.ht;
   struct hash_entry *he = ht ? _mesa_hash_table_search(ht, &key) : NULL;
   if (he) {
      result = he->data;
      list_move_to(&result->lru_link, &dev->apple9_linked_entries.lru);
      agx_shader_body_reference(&result->body);
      simple_mtx_unlock(&dev->apple9_linked_entries.lock);
      return result;
   }
   if (!ht) {
      simple_mtx_unlock(&dev->apple9_linked_entries.lock);
      return NULL;
   }

   struct agx_shader_body *raw_stub = NULL;
   if (fourth) {
      simple_mtx_unlock(&dev->apple9_linked_entries.lock);
      void *map = NULL;
      size_t stub_bytes = fifth ? HK_APPLE9_LINKED_STUB_5_MAX_BYTES
                                : AGX_APPLE9_LINKED_STUB_4_MAX_BYTES;
      raw_stub = hk_apple9_body_alloc_raw(dev, stub_bytes, &map);
      if (!raw_stub)
         return NULL;
      size_t built = fifth
         ? hk_apple9_build_linked_stub_5(map, stub_bytes, raw_stub->addr,
                                         first->addr, second->addr, third->addr,
                                         fourth->addr, fifth->addr)
         : agx_apple9_build_linked_stub_4(map, stub_bytes, raw_stub->addr,
                                          first->addr, second->addr, third->addr,
                                          fourth->addr);
      if (!built) {
         agx_shader_body_unreference(&dev->dev, raw_stub);
         return NULL;
      }
      agx_bo_note_cpu_write(raw_stub->bo,
                            raw_stub->addr - raw_stub->bo->va->addr,
                            stub_bytes);
      simple_mtx_lock(&dev->apple9_linked_entries.lock);
      ht = dev->apple9_linked_entries.ht;
      he = _mesa_hash_table_search(ht, &key);
      if (he) {
         result = he->data;
         list_move_to(&result->lru_link, &dev->apple9_linked_entries.lru);
         agx_shader_body_reference(&result->body);
         simple_mtx_unlock(&dev->apple9_linked_entries.lock);
         agx_shader_body_unreference(&dev->dev, raw_stub);
         return result;
      }
   }

   result = calloc(1, sizeof(*result));
   if (!result) {
      simple_mtx_unlock(&dev->apple9_linked_entries.lock);
      if (raw_stub)
         agx_shader_body_unreference(&dev->dev, raw_stub);
      return NULL;
   }
   memcpy(&result->key, &key, sizeof(key));
   result->program_id = p_atomic_inc_return(&dev->apple9_last_program_id);
   agx_apple9_stage_launch_cache_init(&result->launch_cache);
   agx_shader_body_reference(first);
   agx_shader_body_reference(second);
   if (third)
      agx_shader_body_reference(third);
   if (fourth)
      agx_shader_body_reference(fourth);
   if (fifth)
      agx_shader_body_reference(fifth);
   result->fourth_ref = fourth;
   result->fifth_ref = fifth;
   result->stub_ref = raw_stub;
   result->body = (struct agx_shader_body){
      .bo = raw_stub ? raw_stub->bo : first->bo,
      .addr = raw_stub ? raw_stub->addr : first->addr,
      .mid_addr = raw_stub ? 0 : (third ? second->addr : 0),
      .tail_addr = raw_stub ? 0 : (third ? third->addr : second->addr),
      .first_ref = first,
      .second_ref = second,
      .third_ref = third,
      .refcnt = 2, /* 1 for cache + 1 for caller */
      .destroy = hk_apple9_linked_entry_destroy,
   };
   if (!_mesa_hash_table_insert(ht, &result->key, result)) {
      simple_mtx_unlock(&dev->apple9_linked_entries.lock);
      result->body.refcnt = 1;
      agx_shader_body_unreference(&dev->dev, &result->body);
      return NULL;
   }
   list_add(&result->lru_link, &dev->apple9_linked_entries.lru);
   dev->apple9_linked_entries.count++;
   if (dev->apple9_linked_entries.count > HK_APPLE9_LINKED_ENTRY_CACHE_SIZE) {
      evicted = list_last_entry(&dev->apple9_linked_entries.lru,
                                struct hk_apple9_linked_entry, lru_link);
      list_del(&evicted->lru_link);
      _mesa_hash_table_remove_key(ht, &evicted->key);
      dev->apple9_linked_entries.count--;
   }
   simple_mtx_unlock(&dev->apple9_linked_entries.lock);

   if (evicted)
      agx_shader_body_unreference(&dev->dev, &evicted->body);
   return result;
}

void
hk_apple9_linked_entries_finish(struct hk_device *dev)
{
   if (!dev->apple9_linked_entries.ht)
      return;
   list_for_each_entry_safe(struct hk_apple9_linked_entry, entry,
                            &dev->apple9_linked_entries.lru, lru_link) {
      list_del(&entry->lru_link);
      agx_shader_body_unreference(&dev->dev, &entry->body);
   }
   _mesa_hash_table_destroy(dev->apple9_linked_entries.ht, NULL);
   dev->apple9_linked_entries.ht = NULL;
   dev->apple9_linked_entries.count = 0;
}

#define HK_APPLE9_TRAMPOLINE_CACHE_SIZE 4096

struct hk_apple9_per_sample_trampoline_key {
   uint64_t prolog_addr;
   uint64_t prolog2_addr;
   uint64_t main_addr;
   uint64_t epilog_addr;
   uint64_t preamble_addr;
   uint32_t frame_extent;
   uint16_t publication_count;
   uint16_t tile_access_mode;
   uint8_t samples;
   bool emit_zs_release;
   bool emit_tile_fence;
};

struct hk_apple9_per_sample_trampoline {
   struct hk_apple9_per_sample_trampoline_key key;
   struct agx_shader_body body;
   struct agx_shader_body *prolog_ref;
   struct agx_shader_body *prolog2_ref;
   uint64_t program_id;
   struct agx_apple9_stage_launch_cache launch_cache;
   struct list_head lru_link;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_per_sample_trampoline,
                      struct hk_apple9_per_sample_trampoline_key)

static void
hk_apple9_per_sample_trampoline_destroy(struct agx_device *dev,
                                        struct agx_shader_body *body)
{
   struct hk_apple9_per_sample_trampoline *tramp =
      container_of(body, struct hk_apple9_per_sample_trampoline, body);
   agx_shader_body_unreference(dev, tramp->body.first_ref);
   agx_shader_body_unreference(dev, tramp->body.second_ref);
   agx_shader_body_unreference(dev, tramp->body.third_ref);
   agx_shader_body_unreference(dev, tramp->prolog_ref);
   agx_shader_body_unreference(dev, tramp->prolog2_ref);
   agx_apple9_stage_launch_cache_finish(&tramp->launch_cache);
   free(tramp);
}

static struct hk_apple9_per_sample_trampoline *
hk_apple9_get_per_sample_trampoline(
   struct hk_device *dev, struct agx_shader_body *prolog,
   struct agx_shader_body *prolog2, struct agx_shader_body *fs_main,
   struct agx_shader_body *epilog, uint8_t samples, uint16_t tile_access_mode,
   bool emit_zs_release, bool emit_tile_fence, uint64_t preamble_addr,
   uint32_t frame_extent, uint16_t publication_count)
{
   struct hk_apple9_per_sample_trampoline_key key;
   memset(&key, 0, sizeof(key));
   key.prolog_addr = prolog ? prolog->addr : 0;
   key.prolog2_addr = prolog2 ? prolog2->addr : 0;
   key.main_addr = fs_main->addr;
   key.epilog_addr = epilog->addr;
   key.preamble_addr = preamble_addr;
   key.frame_extent = frame_extent;
   key.publication_count = publication_count;
   key.tile_access_mode = tile_access_mode;
   key.samples = samples;
   key.emit_zs_release = emit_zs_release;
   key.emit_tile_fence = emit_tile_fence;

   simple_mtx_lock(&dev->apple9_per_sample_trampolines.lock);
   if (!dev->apple9_per_sample_trampolines.ht) {
      dev->apple9_per_sample_trampolines.ht = _mesa_hash_table_create(
         NULL, hk_apple9_per_sample_trampoline_hash,
         hk_apple9_per_sample_trampoline_equal);
      list_inithead(&dev->apple9_per_sample_trampolines.lru);
   }
   struct hash_table *ht = dev->apple9_per_sample_trampolines.ht;
   struct hash_entry *he = ht ? _mesa_hash_table_search(ht, &key) : NULL;
   if (he) {
      struct hk_apple9_per_sample_trampoline *found = he->data;
      list_move_to(&found->lru_link, &dev->apple9_per_sample_trampolines.lru);
      agx_shader_body_reference(&found->body);
      simple_mtx_unlock(&dev->apple9_per_sample_trampolines.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_per_sample_trampolines.lock);
   if (!ht)
      return NULL;

   void *map = NULL;
   struct agx_shader_body *raw = hk_apple9_body_alloc_raw(
      dev, AGX_APPLE9_PER_SAMPLE_TRAMPOLINE_MAX_BYTES, &map);
   if (!raw)
      return NULL;
   size_t len = agx_apple9_build_per_sample_trampoline_prologs(
      map, AGX_APPLE9_PER_SAMPLE_TRAMPOLINE_MAX_BYTES, raw->addr, samples,
      key.prolog_addr, key.prolog2_addr, key.main_addr, key.epilog_addr,
      tile_access_mode, emit_zs_release, emit_tile_fence);
   if (!len) {
      agx_shader_body_unreference(&dev->dev, raw);
      return NULL;
   }
   agx_bo_note_cpu_write(raw->bo, raw->addr - raw->bo->va->addr,
                         AGX_APPLE9_PER_SAMPLE_TRAMPOLINE_MAX_BYTES);

   struct hk_apple9_per_sample_trampoline *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, raw);
      return NULL;
   }
   memcpy(&created->key, &key, sizeof(key));
   created->program_id = p_atomic_inc_return(&dev->apple9_last_program_id);
   agx_apple9_stage_launch_cache_init(&created->launch_cache);
   agx_shader_body_reference(fs_main);
   agx_shader_body_reference(epilog);
   if (prolog)
      agx_shader_body_reference(prolog);
   if (prolog2)
      agx_shader_body_reference(prolog2);
   created->prolog_ref = prolog;
   created->prolog2_ref = prolog2;
   created->body = (struct agx_shader_body){
      .bo = raw->bo,
      .addr = raw->addr,
      .first_ref = raw,
      .second_ref = fs_main,
      .third_ref = epilog,
      .refcnt = 2, /* 1 for cache + 1 for caller */
      .destroy = hk_apple9_per_sample_trampoline_destroy,
   };

   struct hk_apple9_per_sample_trampoline *evicted = NULL;
   simple_mtx_lock(&dev->apple9_per_sample_trampolines.lock);
   he = _mesa_hash_table_search(dev->apple9_per_sample_trampolines.ht, &key);
   if (he) {
      struct hk_apple9_per_sample_trampoline *existing = he->data;
      list_move_to(&existing->lru_link,
                   &dev->apple9_per_sample_trampolines.lru);
      agx_shader_body_reference(&existing->body);
      simple_mtx_unlock(&dev->apple9_per_sample_trampolines.lock);
      created->body.refcnt = 1;
      agx_shader_body_unreference(&dev->dev, &created->body);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_per_sample_trampolines.ht,
                                &created->key, created)) {
      simple_mtx_unlock(&dev->apple9_per_sample_trampolines.lock);
      created->body.refcnt = 1;
      agx_shader_body_unreference(&dev->dev, &created->body);
      return NULL;
   }
   list_add(&created->lru_link, &dev->apple9_per_sample_trampolines.lru);
   dev->apple9_per_sample_trampolines.count++;
   if (dev->apple9_per_sample_trampolines.count >
       HK_APPLE9_TRAMPOLINE_CACHE_SIZE) {
      evicted = list_last_entry(&dev->apple9_per_sample_trampolines.lru,
                                struct hk_apple9_per_sample_trampoline,
                                lru_link);
      list_del(&evicted->lru_link);
      _mesa_hash_table_remove_key(dev->apple9_per_sample_trampolines.ht,
                                  &evicted->key);
      dev->apple9_per_sample_trampolines.count--;
   }
   simple_mtx_unlock(&dev->apple9_per_sample_trampolines.lock);

   if (evicted)
      agx_shader_body_unreference(&dev->dev, &evicted->body);
   return created;
}

void
hk_apple9_per_sample_trampolines_finish(struct hk_device *dev)
{
   if (!dev->apple9_per_sample_trampolines.ht)
      return;
   list_for_each_entry_safe(struct hk_apple9_per_sample_trampoline, tramp,
                            &dev->apple9_per_sample_trampolines.lru, lru_link) {
      list_del(&tramp->lru_link);
      agx_shader_body_unreference(&dev->dev, &tramp->body);
   }
   _mesa_hash_table_destroy(dev->apple9_per_sample_trampolines.ht, NULL);
   dev->apple9_per_sample_trampolines.ht = NULL;
   dev->apple9_per_sample_trampolines.count = 0;
}

void
hk_apple9_dispatch_sw_vs_prolog(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct agx_grid grid, const struct hk_apple9_prerast_bindings *bindings)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!s->apple9_body) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct agx_apple9_sw_vs_prolog_key pkey;
   memset(&pkey, 0, sizeof(pkey));
   hk_apple9_copy_vs_prolog_iface(&pkey.base.iface, &s->b.info.apple9_vs_prolog);
   pkey.base.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   pkey.base.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   pkey.base.dynamic_stride = true;
   pkey.base.unbounded = s->info.apple9_vertex_unbounded;
   pkey.vertex_params_ubo = 0;
   pkey.vertex_params_offset_B = hk_root_descriptor_offset(draw.vertex_params);
   pkey.descriptor_tables =
      s->info.apple9.abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES;
   if (dyn->vi) {
      u_foreach_bit(a, pkey.base.iface.attribs_read) {
         if (!(dyn->vi->attributes_valid & BITFIELD_BIT(a)))
            continue;
         const struct vk_vertex_attribute_state *attr = &dyn->vi->attributes[a];
         const struct vk_vertex_binding_state *binding =
            &dyn->vi->bindings[attr->binding];
         pkey.base.format[a] = vk_format_to_pipe_format(attr->format);
         pkey.base.offset[a] = attr->offset;
         pkey.base.buffer[a] = attr->binding;
         pkey.base.divisor[a] =
            binding->input_rate != VK_VERTEX_INPUT_RATE_INSTANCE ? 0
            : binding->divisor ? binding->divisor
                               : AGX_APPLE9_VERTEX_DIVISOR_ZERO;
      }
   }
   struct hk_apple9_sw_vs_prolog *prolog =
      hk_apple9_get_sw_vs_prolog(dev, &pkey);
   if (!prolog) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct agx_apple9_compute_profile merged = s->info.apple9;
   merged.publication_count =
      MAX2(merged.publication_count, prolog->profile.publication_count);
   merged.scratch_size = MAX2(merged.scratch_size, prolog->profile.scratch_size);
   merged.atomic_frame_size =
      MAX2(merged.atomic_frame_size, prolog->profile.atomic_frame_size);
   merged.resource_read_mask |= prolog->profile.resource_read_mask;

   uint64_t preamble_addr =
      merged.preamble_size ? s->apple9_body->addr + merged.preamble_offset : 0;
   struct hk_apple9_linked_entry *linked = hk_apple9_get_linked_entry(
      dev, prolog->body, s->apple9_body, NULL, NULL, NULL, preamble_addr,
      MAX2(merged.scratch_size, merged.atomic_frame_size),
      merged.publication_count, 2 /* compute */);
   if (!linked) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   hk_apple9_dispatch(cmd, cs, &linked->body, &merged, grid,
                      cmd->state.gfx.root, s, bindings);
   agx_shader_body_unreference(&dev->dev, &linked->body);
}

struct hk_shader *
hk_apple9_select_mesh_rast(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *mesh)
{
   struct hk_shader *mesh_main = &mesh->variants[HK_MESH_VARIANT_MAIN];
   struct hk_shader *mesh_rast = &mesh->variants[HK_MESH_VARIANT_RAST];
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_shader *fs =
      gfx->shaders[MESA_SHADER_FRAGMENT]
         ? hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT])
         : NULL;

   bool fs_has_pv =
      fs && (fs->info.fs.per_vertex_copies || fs->info.fs.per_vertex_bary[0] ||
             fs->info.fs.per_vertex_bary[1]);
   bool want_pp =
      dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT &&
      mesh_main->info.mesh.prim == MESA_PRIM_TRIANGLES && !fs_has_pv &&
      (!fs || fs == hk_only_variant(dev->null_fs) || fs->info.fs.polygon_points);
   bool want_stipple =
      dyn->rs.line.stipple.enable && mesh_main->info.mesh.prim == MESA_PRIM_LINES;

   bool need_meta = want_pp != mesh_rast->info.mesh.polygon_points;
   if (!need_meta && want_stipple &&
       !(mesh_rast->b.info.apple9_varyings.mask[VARYING_SLOT_TEX6]))
      need_meta = true;
   if (!need_meta && !want_pp && fs_has_pv) {
      for (unsigned m = 0; m < 2; ++m) {
         uint8_t bslot = fs->info.fs.per_vertex_bary[m];
         if (bslot && !mesh_rast->b.info.apple9_varyings.mask[bslot])
            need_meta = true;
      }
      for (unsigned j = 0; j < fs->info.fs.per_vertex_copies; ++j) {
         for (unsigned k = 0; k < 3; ++k) {
            unsigned slot = hk_per_vertex_copy_slot(
               fs->info.fs.per_vertex_copy_base[j], k);
            if (!mesh_rast->b.info.apple9_varyings.mask[slot])
               need_meta = true;
         }
      }
   }
   if (!need_meta)
      return mesh_rast;

   struct hk_api_shader *meta = hk_apple9_mesh_rast_meta(
      dev, mesh_main, fs, want_stipple, want_pp);
   return meta ? &meta->variants[HK_VS_VARIANT_HW] : mesh_rast;
}

struct hk_apple9_vs_vary_key {
   uint64_t vs_program_id;
   uint64_t flat_mask;
   uint64_t linear_mask;
};

struct hk_apple9_vs_vary_variant {
   struct hk_apple9_vs_vary_key key;
   struct agx_shader_body *body;
   uint64_t program_id;
   struct agx_apple9_varying_layout varyings;
   struct agx_apple9_stage_launch_cache launch_cache;
};

HK_DEFINE_KEY_HASH_EQ(hk_apple9_vs_vary, struct hk_apple9_vs_vary_key)

static struct hk_apple9_vs_vary_variant *
hk_apple9_get_vs_vary_variant(struct hk_device *dev,
                              const struct hk_shader *vs,
                              uint64_t fs_flat, uint64_t fs_linear)
{
   if (!vs || !vs->code_ptr || !vs->code_size || !vs->apple9_program_id ||
       !vs->b.info.apple9_vary_relocs_valid)
      return NULL;

   uint64_t written = 0;
   for (unsigned s = 0; s < ARRAY_SIZE(vs->b.info.apple9_varyings.mask); ++s) {
      if (vs->b.info.apple9_varyings.mask[s])
         written |= BITFIELD64_BIT(s);
   }
   struct hk_apple9_vs_vary_key key;
   memset(&key, 0, sizeof(key));
   key.vs_program_id = vs->apple9_program_id;
   key.flat_mask = fs_flat & written;
   key.linear_mask = fs_linear & written;

   simple_mtx_lock(&dev->apple9_vs_vary_variants.lock);
   if (!dev->apple9_vs_vary_variants.ht) {
      dev->apple9_vs_vary_variants.ht = _mesa_hash_table_create(
         NULL, hk_apple9_vs_vary_hash, hk_apple9_vs_vary_equal);
   }
   struct hash_table *ht = dev->apple9_vs_vary_variants.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, &key) : NULL;
   if (entry) {
      struct hk_apple9_vs_vary_variant *found = entry->data;
      simple_mtx_unlock(&dev->apple9_vs_vary_variants.lock);
      return found;
   }
   simple_mtx_unlock(&dev->apple9_vs_vary_variants.lock);
   if (!ht)
      return NULL;

   uint8_t *patched = malloc(vs->code_size);
   if (!patched)
      return NULL;
   struct agx_apple9_varying_layout new_layout = {0};
   if (!agx_apple9_patch_vs_varyings(vs->code_ptr, vs->code_size, &vs->b.info,
                                     key.flat_mask, key.linear_mask, patched,
                                     &new_layout)) {
      free(patched);
      return NULL;
   }

   struct hk_shader upload = {
      .code_ptr = patched,
      .code_size = vs->code_size,
      .data_ptr = vs->data_ptr,
      .data_size = vs->data_size,
   };
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &upload);
   free(patched);
   if (!body)
      return NULL;

   struct hk_apple9_vs_vary_variant *created = calloc(1, sizeof(*created));
   if (!created) {
      agx_shader_body_unreference(&dev->dev, body);
      return NULL;
   }
   memcpy(&created->key, &key, sizeof(key));
   created->body = body;
   created->program_id = p_atomic_inc_return(&dev->apple9_last_program_id);
   created->varyings = new_layout;
   agx_apple9_stage_launch_cache_init(&created->launch_cache);

   simple_mtx_lock(&dev->apple9_vs_vary_variants.lock);
   entry = _mesa_hash_table_search(dev->apple9_vs_vary_variants.ht, &key);
   if (entry) {
      struct hk_apple9_vs_vary_variant *existing = entry->data;
      simple_mtx_unlock(&dev->apple9_vs_vary_variants.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return existing;
   }
   if (!_mesa_hash_table_insert(dev->apple9_vs_vary_variants.ht, &created->key,
                                created)) {
      simple_mtx_unlock(&dev->apple9_vs_vary_variants.lock);
      agx_apple9_stage_launch_cache_finish(&created->launch_cache);
      agx_shader_body_unreference(&dev->dev, created->body);
      free(created);
      return NULL;
   }
   simple_mtx_unlock(&dev->apple9_vs_vary_variants.lock);
   return created;
}

void
hk_apple9_vs_vary_variants_finish(struct hk_device *dev)
{
   if (!dev->apple9_vs_vary_variants.ht)
      return;
   hash_table_foreach(dev->apple9_vs_vary_variants.ht, entry) {
      struct hk_apple9_vs_vary_variant *v = entry->data;
      agx_apple9_stage_launch_cache_finish(&v->launch_cache);
      agx_shader_body_unreference(&dev->dev, v->body);
      free(v);
   }
   _mesa_hash_table_destroy(dev->apple9_vs_vary_variants.ht, NULL);
   dev->apple9_vs_vary_variants.ht = NULL;
}

/* These inputs depend on attachment dimensions and clear values, not on which
 * helper loads/stores them. Immutable pool storage can also be shared by later
 * renders with identical inputs until the command buffer is reset. */
static bool
hk_apple9_render_inputs(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (render->apple9_arguments)
      return true;

   uint32_t params[48] = {0};
   uint32_t dimensions[HK_MAX_RTS][5] = {0};
   uint32_t samples = MAX2(render->tilebuffer.nr_samples, 1);
   bool layered = render->cr.layers > 1;
   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view)
         continue;
      memcpy(&params[16 + rt * 4],
             &info->pColorAttachments[rt].clearValue.color, 16);
      memcpy(dimensions[rt], view->apple9_render[layered].dimensions, 16);
      dimensions[rt][4] = samples;
   }
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint64_t sysvals = gfx->apple9_render_sysvals;
   if (!sysvals || memcmp(dimensions, gfx->apple9_render_dimensions,
                          sizeof(dimensions))) {
      uint8_t metadata[AGX_APPLE9_RUNTIME_SYSVAL_SIZE] = {0};
      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         memcpy(metadata + AGX_APPLE9_TEXTURE_INFO_OFFSET +
                   rt * AGX_APPLE9_TEXTURE_INFO_STRIDE,
                dimensions[rt], sizeof(dimensions[rt]));
      }
      sysvals = hk_pool_upload(cmd, metadata, sizeof(metadata), 64);
      if (!sysvals)
         return false;
      gfx->apple9_render_sysvals = sysvals;
      memcpy(gfx->apple9_render_dimensions, dimensions, sizeof(dimensions));
   }
   uint64_t arguments = hk_pool_upload_last(cmd, HK_UPLOAD_LAST_RENDER_PARAMS,
                                            params, sizeof(params), 16);
   if (!sysvals || !arguments)
      return false;
   render->apple9_sysvals = sysvals;
   render->apple9_arguments = arguments;
   return true;
}

struct hk_bg_eot
hk_apple9_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                       bool store, bool partial, bool incomplete)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   /* Keys are hashed as bytes, including padding. */
   struct agx_apple9_tilebuffer_key key;
   memset(&key, 0, sizeof(key));
   key.count = render->color_att_count;
   key.samples = MAX2(render->tilebuffer.nr_samples, 1);
   key.store = store;
   key.layered = render->cr.layers > 1;
   struct agx_apple9_blend blend[8] = {0};
   unsigned tile_bytes = 0;
   VkResult error = VK_ERROR_FEATURE_NOT_PRESENT;
   if (key.count > 8)
      goto fail;

   for (unsigned rt = 0; rt < key.count; ++rt) {
      struct hk_image_view *view = render->color_att[rt].iview;
      key.formats[rt] = view ? hk_format_to_pipe_format(view->vk.format)
                             : PIPE_FORMAT_R8G8B8A8_UNORM;
      blend[rt].format = key.formats[rt];
      blend[rt].samples = key.samples;
      tile_bytes += 4 * agx_apple9_color_words(key.formats[rt]);
      if (!view)
         continue;
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[rt];
      bool load =
         partial || incomplete || att->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
      bool clear = !load && att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      bool write =
         !(render->no_store_color_mask & BITFIELD_BIT(rt)) &&
         (partial || att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE);
      if (store ? write : (load || clear))
         key.mask |= BITFIELD_BIT(rt);
      if (clear)
         key.clear_mask |= BITFIELD_BIT(rt);
      if (view->apple9_render[key.layered].array)
         key.array_mask |= BITFIELD_BIT(rt);
   }
   blend[0].samples = key.samples;
   if (tile_bytes > 128) {
      fprintf(stderr, "Apple9 tile helper: %u tile bytes at %u samples\n",
              tile_bytes, key.samples);
      goto fail;
   }
   struct hk_apple9_tile_helper *helper =
      hk_apple9_tile_helper(dev, &key, blend);
   if (!helper)
      goto fail;
   const struct agx_shader_info *part_info = &helper->info;
   error = VK_ERROR_OUT_OF_DEVICE_MEMORY;
   if (!hk_apple9_render_inputs(cmd, info))
      goto fail;
   struct agx_ptr textures =
      hk_pool_alloc(cmd, 8 * AGX_APPLE9_TEXTURE_TABLE_STRIDE, 64);
   if (!textures.cpu)
      goto fail;
   unsigned slot = 0;
   error = VK_ERROR_FEATURE_NOT_PRESENT;
   u_foreach_bit(rt, part_info->apple9_texture_mask) {
      const struct hk_image_view *view =
         rt < key.count ? render->color_att[rt].iview : NULL;
      if (!view || !view->apple9_render[key.layered].texture_valid)
         goto fail;
      memcpy((uint8_t *)textures.cpu + slot++ * AGX_APPLE9_TEXTURE_TABLE_STRIDE,
             view->apple9_render[key.layered].texture,
             AGX_APPLE9_TEXTURE_TABLE_STRIDE);
   }
   u_foreach_bit(rt, part_info->apple9_image_mask) {
      const struct hk_image_view *view =
         rt < key.count ? render->color_att[rt].iview : NULL;
      uint8_t *store = (uint8_t *)textures.cpu + slot++ * AGX_APPLE9_TEXTURE_TABLE_STRIDE;
      /* Feedback-raw attachments store uncompressed; the prepacked store is
       * the compressed one. */
      if (view && (render->raw_color_mask & BITFIELD_BIT(rt))) {
         if (!hk_apple9_image_view_eot(view, key.layered, true, store)) {
            fprintf(stderr, "Apple9 tile helper: unsupported store image %u\n", rt);
            goto fail;
         }
         continue;
      }
      if (!view || !view->apple9_render[key.layered].store_valid)
         goto fail;
      memcpy(store, view->apple9_render[key.layered].store,
             AGX_APPLE9_TEXTURE_TABLE_STRIDE);
   }
   uint64_t resources[AGX_APPLE9_MAX_GRAPHICS_BUFFERS] = {0};
   for (unsigned i = 0; i < part_info->apple9_resource_count; ++i) {
      unsigned binding = part_info->apple9_resource_binding[i];
      if (binding == 0)
         resources[i] = render->apple9_arguments;
      else if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
         resources[i] = render->apple9_sysvals;
      else
         goto fail;
   }
   error = VK_ERROR_OUT_OF_HOST_MEMORY;
   uint64_t table = hk_pool_upload(
      cmd, resources, part_info->apple9_resource_count * sizeof(uint64_t), 64);
   if (part_info->apple9_resource_count && !table)
      goto fail;
   struct agx_apple9_render_stage shader = {
      .bo = helper->bo,
      .launch_cache = &helper->launch,
      .binary_size = part_info->binary_size,
      .preamble_offset = part_info->apple9_preamble_offset,
      .preamble_size = part_info->apple9_preamble_size,
      .publication_count = part_info->apple9_publication_count,
      .publication_count_valid = true,
      .scratch_size = part_info->scratch_size,
   };
   uint32_t launch;
   if (!agx_apple9_prepare_graphics_stage(
          &dev->dev, cmd, hk_apple9_usc_alloc, &shader, true, table,
          textures.gpu, dev->apple9_reload_sampler,
          agx_apple9_tile_sample_bytes(tile_bytes, key.samples), key.samples,
          &launch))
      goto fail;
   return (struct hk_bg_eot){.usc = launch};

fail:
   vk_command_buffer_set_error(&cmd->vk, error);
   return (struct hk_bg_eot){0};
}

static_assert(HK_MAX_VIEWPORTS == AGX_APPLE9_MAX_VIEWPORTS,
              "Vulkan viewport limit matches the PPP viewport block");

static struct agx_ptr
hk_apple9_context_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_context_alloc(data, size, alignment);
}

/* Draw parameters (first vertex or index bias, then base instance) and the
 * vertex buffer table, shared by the hardware vertex fetch and by software
 * vertex shaders in front of a geometry shader.
 */
struct hk_apple9_prerast_bindings
hk_apple9_draw_bindings(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t params[] = {draw.indexed ? draw.index_bias : draw.start,
                        draw.start_instance};
   uint64_t draw_params =
      agx_is_indirect(draw.b)
         ? draw.b.ptr +
              (draw.indexed
                  ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
                  : offsetof(VkDrawIndirectCommand, firstVertex))
         : hk_pool_upload_last(cmd, HK_UPLOAD_LAST_DRAW_PARAMS, params,
                               sizeof(params), 4);
   uint64_t vertex_buffers[ARRAY_SIZE(gfx->vb)][4] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->vb); ++i) {
      vertex_buffers[i][0] = gfx->vb[i].addr;
      vertex_buffers[i][1] = gfx->vb[i].range;
      /* A null or empty binding is the zero page with range 0. Unbounded
       * fetches skip the range check, so give it stride 0: every vertex
       * then reads the zero page at its attribute offset. */
      vertex_buffers[i][2] = gfx->vb[i].range ? dyn->vi_binding_strides[i] : 0;
   }
   return (struct hk_apple9_prerast_bindings){
      .draw_params = draw_params,
      .vertex_buffers =
         hk_pool_upload_last(cmd, HK_UPLOAD_LAST_VERTEX_BUFFERS, vertex_buffers,
                             sizeof(vertex_buffers), 16),
   };
}

static enum agx_primitive
hk_apple9_rast_primitive(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   default:
      return AGX_PRIMITIVE_TRIANGLES;
   }
}

/* Split before recording any of a draw's pre-rasterization work. Ending a
 * graphics stream also closes its pre/post compute streams and heap lifetime.
 * Reserve for every emulated multiview draw so these stay together.
 */
/* Sample positions for a draw: the application's when enabled for this
 * sample count, the standard ones otherwise. */
static uint32_t
hk_apple9_draw_sample_positions(const struct vk_dynamic_graphics_state *dyn,
                                unsigned samples)
{
   return dyn->ms.sample_locations_enable && dyn->ms.sample_locations &&
                dyn->ms.sample_locations->per_pixel == samples
             ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
             : hk_default_sample_positions(samples);
}

DEBUG_GET_ONCE_BOOL_OPTION(native_blocks, "HK_APPLE9_NATIVE_BLOCKS", true)
DEBUG_GET_ONCE_BOOL_OPTION(native_lanes, "HK_APPLE9_NATIVE_LANES", true)
DEBUG_GET_ONCE_BOOL_OPTION(native_batch_log, "HK_APPLE9_NATIVE_BATCH_LOG", false)
DEBUG_GET_ONCE_NUM_OPTION(native_render_primitives, "HK_APPLE9_NATIVE_RENDER_PRIMITIVES",
                          HK_APPLE9_NATIVE_RENDER_PRIMITIVES)

/* The bound on estimated binned primitives per native render, for batching
 * and for single native draws (hk_apple9_use_native_mesh). A mesh-mode render
 * that reaches the TVB cap loses work; HK_APPLE9_NATIVE_RENDER_PRIMITIVES
 * overrides the calibrated default (research, e.g. a lowered tvb_max_blocks). */
uint64_t
hk_apple9_native_render_primitives(void)
{
   int64_t v = debug_get_option_native_render_primitives();
   return v > 0 ? (uint64_t)v : HK_APPLE9_NATIVE_RENDER_PRIMITIVES;
}

/* The admission profile of this device's VM: its latched asahi_native_blocks
 * profile (sampled around VM_CREATE with asahi_native_mesh, at one
 * generation), only together with the qualified native path. drirc
 * hk_disable_native_blocks and HK_APPLE9_NATIVE_BLOCKS=0 keep one native
 * block per render. */
struct hk_apple9_native_profile
hk_apple9_native_profile(const struct hk_device *dev)
{
   /* Every graphics stream asks (hk_apple9_graphics_root_size): stay cheap and
    * device-independent unless the VM latched a qualified profile. */
   if (!dev->dev.native_blocks.qualified)
      return (struct hk_apple9_native_profile){.batch = 1};
   const struct hk_physical_device *pdev =
      hk_device_physical((struct hk_device *)dev);
   struct hk_apple9_native_profile p = hk_apple9_native_profile_from_kernel(
      &dev->dev.native_blocks, dev->dev.native_mesh_qualified,
      !debug_get_option_native_blocks() || !pdev || pdev->disable_native_blocks);
   if (p.max_render_primitives)
      p.max_render_primitives = hk_apple9_native_render_primitives();
   return p;
}

bool
hk_apple9_native_blocks_enabled(const struct hk_device *dev)
{
   struct hk_apple9_native_profile p = hk_apple9_native_profile(dev);
   return p.batch > 1 && p.blocks;
}

/* Whether native GS/TES producer passes of batched native renders use the
 * tessellation batch lanes (hk_tess_batch_flush): one compute barrier per
 * stage for all of a render's draws instead of one per pass. They follow the
 * native blocks profile; HK_APPLE9_NATIVE_LANES=0 opts out.
 * hk_cmd_buffer_end_graphics flushes the lanes into the ending render's own
 * pre_gfx stream, so every pass still completes before the tiling that
 * consumes it. */
bool
hk_apple9_native_lanes(const struct hk_device *dev)
{
   return hk_apple9_native_blocks_enabled(dev) && debug_get_option_native_lanes();
}

/* Batched native graphics streams must not link (the decoder admits none):
 * give them room for the profile's decode budget plus the ensure padding. */
size_t
hk_apple9_graphics_root_size(const struct hk_device *dev)
{
   if (!hk_apple9_native_blocks_enabled(dev))
      return 4096;
   struct hk_apple9_native_profile p = hk_apple9_native_profile(dev);
   return hk_apple9_native_decode_bytes(&p) > 4096 ? 16384 : 8192;
}

/* Whether a draw may continue the open native render, under the device's
 * profile (hk_apple9_native_stream_admits). */
static bool
hk_apple9_native_can_append(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            bool native)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!cs->apple9_native_mesh || !hk_apple9_native_mesh_available_device(dev))
      return false;
   uint8_t *start = cs->start, *current = cs->current;
   const struct hk_apple9_stream_summary stream = {
      .native = cs->apple9_native_mesh,
      .dispatches = cs->apple9_native_dispatches,
      /* A continued stream has left the root chunk: never admissible. */
      .bytes = current >= start ? current - start : SIZE_MAX / 2,
      .capacity = cs->stream_linked ||
                        cs->root_size < 0x800 + AGX_VDM_STREAM_LINK_LENGTH
                     ? 0
                     : cs->root_size - 0x800 - AGX_VDM_STREAM_LINK_LENGTH,
      .primitives = cs->apple9_native_primitives,
      .ring_rows = cs->apple9_native_ring_rows,
      .ring_limited = cs->apple9_native_ring_limited,
   };
   struct hk_apple9_draw_summary draw = {
      .native = native,
      .meta = cmd->in_meta,
      .queries = gfx->occlusion.active_count ||
                 gfx->descriptors.root.draw.pipeline_stats_flags ||
                 hk_cond_render_applies(cmd) ||
                 gfx->render.view_mask != 0,
      .task = gfx->shaders[MESA_SHADER_TASK],
      .xfb = gfx->xfb_enabled,
      .mesh = gfx->shaders[MESA_SHADER_MESH],
      .gs = gfx->shaders[MESA_SHADER_GEOMETRY],
      .tess = gfx->shaders[MESA_SHADER_TESS_CTRL] ||
              gfx->shaders[MESA_SHADER_TESS_EVAL],
      .primitives = cmd->apple9_draw_primitives,
      .ring_rows = native ? cmd->apple9_draw_ring_rows : 0,
      .ring_limited = native && cmd->apple9_draw_ring_limited,
   };
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
      draw.writes_memory |= gfx->shaders[s] && gfx->shaders[s]->writes_memory;
   const struct hk_apple9_native_profile profile = hk_apple9_native_profile(dev);
   return hk_apple9_native_stream_admits(&stream, &draw, &profile);
}

bool
hk_apple9_prepare_draw(struct hk_cmd_buffer *cmd, bool reads_buffers,
                         bool native_mesh)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      return false;

   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* A native render leads with a native block. End it before any draw the
    * VM's native blocks profile does not admit (always, without one), and
    * end an ordinary stream before its first native draw. Existing
    * partial-render stores/reloads and continuation dependencies preserve
    * attachments and order; no emitted work is rerouted retroactively. */
   bool append = hk_apple9_native_can_append(cmd, cs, native_mesh);
   if ((cs->apple9_native_mesh && !append) ||
       (native_mesh && !cs->apple9_native_mesh && cs->current != cs->start)) {
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
   }

   /* A draw that selects a new occlusion segment in a stream without free
    * segments ends that stream (hk_occlusion_prepare_draw). Do it here,
    * before this function commits render state (depth bias representation,
    * sample positions, write and read classes) to the stream, so that state
    * lands on the stream the draw records into. Later splits here start
    * streams with free segments.
    */
   const struct hk_graphics_state *ogfx = &cmd->state.gfx;
   if (ogfx->occlusion.segment_dirty && ogfx->occlusion.active_count &&
       cs->occlusion_segments == HK_OCCLUSION_SEGMENTS) {
      perf_debug(cmd, "Splitting control stream for occlusion segments");
      hk_cs_end_hint(cmd, HK_CS_END_QUERY);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
   }

   /* Sample positions are render state (the per-render multisample control),
    * so the first application draw of a render commits them and a draw with
    * other positions starts a new render. That lets them vary within a
    * subpass (variableSampleLocations, reported to Zink only). Meta draws
    * neither commit nor change them.
    */
   if (!cmd->in_meta) {
      unsigned samples = MAX2(render->tilebuffer.nr_samples ?:
                              dyn->ms.rasterization_samples, 1);
      uint32_t positions = hk_apple9_draw_sample_positions(dyn, samples);
      if (cs->has_sample_locations && cs->ppp_multisamplectl != positions) {
         perf_debug(cmd, "Splitting Apple9 render: sample positions changed");
         hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false);
         if (!cs)
            return false;
      }
      cs->has_sample_locations = true;
      cs->ppp_multisamplectl = positions;
   }

   unsigned views = util_bitcount(render->view_mask ?: 1);
   unsigned viewports = dyn->rs.rasterizer_discard_enable
                           ? 1 : MAX2(dyn->vp.viewport_count, 1);
   bool bias = dyn->rs.depth_bias.enable;
   bool dbias_is_int = render->depth_att.vk_format == VK_FORMAT_D16_UNORM ||
      dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

   bool overflow =
      cs->scissor.size / AGX_SCISSOR_LENGTH + viewports * views > UINT16_MAX ||
      (bias && cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH + views > UINT16_MAX);
   bool change_bias = bias && !u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
   if (unlikely(overflow || change_bias)) {
      perf_debug(cmd, "Splitting Apple9 render: %s",
                 overflow ? "scissor/depth bias overflow" : "depth bias representation");
      /* The existing partial-render background, EOT and ZLS state stores and
       * reloads attachments. Only the final stream uses the API store ops.
       */
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }

   /* A stream that continued across a hoisted image barrier must not write
    * memory: tiles of its earlier passes may still read what it writes. */
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   bool writes = gfx->xfb_enabled;
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT + 1 && !writes; ++s) {
      if (s != MESA_SHADER_COMPUTE && gfx->shaders[s])
         writes = gfx->shaders[s]->writes_memory;
   }
   writes |= gfx->shaders[MESA_SHADER_TASK] && gfx->shaders[MESA_SHADER_TASK]->writes_memory;
   writes |= gfx->shaders[MESA_SHADER_MESH] && gfx->shaders[MESA_SHADER_MESH]->writes_memory;
   if (unlikely(writes && cs->merge_hoisted)) {
      perf_debug(cmd, "Splitting Apple9 render: memory writes after a hoisted barrier");
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }
   cs->merge_writes |= writes;

   /* What the draw's pre-rasterization work reads. hk-internal pre-raster
    * work (tessellation, geometry, transform feedback, native task bridge)
    * keeps the tiling wait. Emulated task draws retain their existing read
    * classification. Meta draws compile their pipelines like applications', so their
    * shaders carry a read class (vk_meta vertex shaders read push constants
    * and its own vertex data). */
   uint8_t reads = reads_buffers ? HK_READS_BUFFERS : 0;
   if (cmd->in_meta)
      reads |= HK_READS_META;
   if (gfx->xfb_enabled || gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->shaders[MESA_SHADER_GEOMETRY] ||
       (native_mesh && gfx->shaders[MESA_SHADER_TASK]))
      reads |= HK_READS_INTERNAL;
   for (unsigned s = 0; s < ARRAY_SIZE(gfx->shaders); ++s) {
      if (s == MESA_SHADER_FRAGMENT || s == MESA_SHADER_COMPUTE || !gfx->shaders[s])
         continue;
      reads |= gfx->shaders[s]->pre_raster_reads;
   }
   cs->pre_raster_reads |= reads;
   cs->linear_store |= gfx->render.linear_store;

   return true;
}

/* The compiler's local roots and ordinary graphics roots share one table.
 * No scene allocation, cursor, resource ID or alias crosses into userspace.
 * Command-pool storage and pinned shader BOs survive command-buffer reuse. */
static bool
hk_apple9_prepare_native_mesh(struct hk_cmd_buffer *cmd,
                               const struct hk_shader *mesh,
                               const struct hk_shader *object,
                               const struct agx_apple9_uniform_draw *draw,
                               uint64_t grid,
                               uint32_t state[AGX_APPLE9_MESH_STATE_WORDS])
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_apple9_native_mesh_available_device(dev) ||
       !grid || (grid & 3) ||
       mesh->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
       object->info.apple9_native_stage != HK_APPLE9_NATIVE_OBJECT)
      return false;
   struct agx_apple9_launch_parameters p[2];
   const struct hk_shader *shaders[] = {mesh, object};
   for (unsigned i = 0; i < 2; ++i) {
      const struct hk_shader *shader = shaders[i];
      if (!shader->apple9_body ||
          !hk_apple9_native_launch_parameters(
             &p[i], &shader->b.info, &shader->info.apple9_native,
             shader->info.apple9_native_stage, dev->dev.shader_base,
             shader->apple9_body->addr))
         return false;
      simple_mtx_lock(&dev->dev.apple9_entry_lock);
      bool pinned = hk_apple9_pin_entry(cmd, shader->apple9_body);
      simple_mtx_unlock(&dev->dev.apple9_entry_lock);
      if (!pinned) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
   }
   struct agx_apple9_mesh_arguments args = {
      .graphics = {draw->texture_table[0], draw->sampler_table[0],
                   draw->vertex_table},
      .output_root = HK_APPLE9_NATIVE_OUTPUT_ROOT,
      .grid_root = HK_APPLE9_NATIVE_GRID_ROOT,
      .draw = grid,
   };
   uint64_t words[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2];
   if (!agx_apple9_mesh_arguments_build(words, &args))
      return false;
   uint64_t arguments = hk_pool_upload(cmd, words, sizeof(words), 64);
   struct agx_ptr launch = hk_pool_usc_alloc(
      cmd, 2 * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE, 128);
   if (!arguments || !launch.cpu || launch.gpu < dev->dev.shader_base ||
       launch.gpu - dev->dev.shader_base >
          UINT32_MAX - 2 * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      p[i].resource_table = arguments;
      p[i].launch_address = launch.gpu + i * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
      if (!agx_apple9_launch_build(
             (uint8_t *)launch.cpu + i * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
             AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
             i ? AGX_APPLE9_LAUNCH_TASK : AGX_APPLE9_LAUNCH_MESH, &p[i]))
         return false;
   }
   struct agx_apple9_mesh_state_parameters params = {
      .layout = &mesh->info.apple9_native.layout,
      .mesh_launch = agx_usc_addr(&dev->dev, launch.gpu) / 64,
      .object_launch = agx_usc_addr(
         &dev->dev, launch.gpu + AGX_APPLE9_GRAPHICS_LAUNCH_SIZE) / 64,
      .threads = mesh->b.info.workgroup_size[0],
      .has_object_stage = true,
   };
   return agx_apple9_mesh_state_build(state, &params);
}

/* Index fetch is per-command state. It does not feed the prepared raster,
 * shader bindings or launch records. Callers resolve null bindings and check
 * the aperture before using this for either the full or cached draw path. */
static void
hk_apple9_set_index_fetch(struct agx_apple9_render_pipeline *pipeline,
                          struct agx_draw draw, uint64_t buffer,
                          uint64_t range, uint64_t offset)
{
   pipeline->index_size = agx_index_size_to_B(draw.index_size);
   pipeline->index_buffer = buffer + offset;
   pipeline->index_extent =
      agx_is_indirect(draw.b)
         ? range - offset
         : MIN2(range - offset, (uint64_t)draw.b.count[0] * pipeline->index_size);
}

DEBUG_GET_ONCE_BOOL_OPTION(native_ppp_delta, "HK_APPLE9_NATIVE_PPP_DELTA", true)

static void
hk_apple9_emit_ordinary_draw(struct hk_cs *cs,
                             const struct agx_apple9_render_pipeline *pipeline,
                             struct agx_draw draw)
{
   /* Ordinary draws update the whole PPP state their own way. */
   cs->apple9_ppp_valid = false;
   if (agx_is_indirect(draw.b))
      cs->current =
         agx_apple9_emit_indirect_draw(cs->current, pipeline, draw.b.ptr) - 4;
   else
      cs->current = agx_apple9_emit_direct_draw(
                       cs->current, pipeline, draw.b.count[0], draw.b.count[1],
                       draw.indexed ? draw.index_bias : draw.start) - 4;
}

static_assert(HK_APPLE9_MESH_COMMAND_BYTES == HK_APPLE9_NATIVE_BLOCK_MAX_B,
              "batch reservation covers a complete native block");
static_assert(HK_APPLE9_NATIVE_RING_ROWS == AGX_APPLE9_MESH_RING_ROWS,
              "batch ring budget matches hardware 64 KiB mesh ring");

/* Backstop for hk_apple9_native_stream_admits: a block appended to a batched
 * native stream must stay in the root chunk and the decoder budget, and be
 * no larger than the reservation the predicate assumed. Otherwise the paired
 * kernel rejects the whole render at SUBMIT, so fail recording instead. */
static bool
hk_apple9_check_native_block(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             const uint8_t *before)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!cs->apple9_native_mesh || !hk_apple9_native_blocks_enabled(dev))
      return true;
   const struct hk_apple9_native_profile profile = hk_apple9_native_profile(dev);
   const uint8_t *start = cs->start, *current = cs->current;
   if (!cs->stream_linked && before >= start && current >= before &&
       current + 4 - before <= HK_APPLE9_BATCH_BLOCK_MAX_B &&
       current + 4 - start <=
          MIN2(cs->root_size, hk_apple9_native_decode_bytes(&profile)))
      return true;
   fprintf(stderr, "HK_NATIVE_BATCH_OVERFLOW stream=%" PRIx64 " linked=%u\n",
           cs->addr, cs->stream_linked);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
   return false;
}

/* Record the final draw packet after its resources and pipeline are prepared.
 * Keep both routes here so admission and actual packet emission use one policy. */
static inline void
hk_apple9_add_native_primitives(struct hk_cs *cs, uint64_t primitives)
{
   cs->apple9_native_primitives =
      primitives > UINT64_MAX - cs->apple9_native_primitives
         ? UINT64_MAX
         : cs->apple9_native_primitives + primitives;
}

static inline void
hk_apple9_add_native_ring_rows(struct hk_cs *cs, uint64_t rows, bool limited)
{
   cs->apple9_native_ring_rows =
      rows > UINT64_MAX - cs->apple9_native_ring_rows
         ? UINT64_MAX
         : cs->apple9_native_ring_rows + rows;
   cs->apple9_native_ring_limited |= limited;
}

static bool
hk_apple9_record_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct agx_apple9_render_pipeline *pipeline,
                       struct agx_draw draw, uint64_t native_grid,
                       struct hk_api_shader *mesh,
                       const struct hk_shader *vertex,
                       const struct agx_apple9_uniform_draw *record)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (native_grid) {
      uint32_t state[AGX_APPLE9_MESH_STATE_WORDS];
      const uint32_t object_groups[] = {1, 1, 1};
      const struct hk_shader *object = hk_native_producer_object(mesh);
      const uint32_t object_local_size[] = {
         object->b.info.workgroup_size[0], object->b.info.workgroup_size[1],
         object->b.info.workgroup_size[2],
      };
      /* Each native block is complete: its own state words, launch records
       * and arguments, whether it leads the render or continues it. */
      /* A block continuing a native render may update only the PPP blocks
       * that changed since the previous native block (HK_APPLE9_NATIVE_PPP_DELTA=0
       * restores full updates). */
      const bool delta = debug_get_option_native_ppp_delta() && cs->apple9_native_mesh &&
                         cs->apple9_ppp_valid && cs->apple9_ppp_size == record->ppp_size &&
                         cs->apple9_ppp_viewport_offset == pipeline->ppp_viewport_offset &&
                         cs->apple9_ppp_viewport_words == pipeline->ppp_viewport_words;
      size_t bytes;
      if ((cs->apple9_native_mesh && !hk_apple9_native_can_append(cmd, cs, true)) ||
          !hk_apple9_prepare_native_mesh(
             cmd, vertex, object, record, native_grid, state) ||
          !hk_apple9_mesh_command_build_delta(
             cs->current, HK_APPLE9_MESH_COMMAND_BYTES, pipeline, state,
             object_groups, object_local_size,
             hk_apple9_native_mesh_available_device(dev),
             delta ? cs->apple9_ppp_last : NULL, delta ? record->ppp_record : NULL,
             &bytes))
         return false;
      memcpy(cs->apple9_ppp_last, record->ppp_record, record->ppp_size);
      cs->apple9_ppp_size = record->ppp_size;
      cs->apple9_ppp_viewport_offset = pipeline->ppp_viewport_offset;
      cs->apple9_ppp_viewport_words = pipeline->ppp_viewport_words;
      cs->apple9_ppp_valid = true;
      const uint8_t *before = cs->current;
      cs->current += bytes - 4;
      cs->apple9_native_mesh = true;
      ++cs->apple9_native_dispatches;
      hk_apple9_add_native_primitives(cs, cmd->apple9_draw_primitives);
      hk_apple9_add_native_ring_rows(cs, cmd->apple9_draw_ring_rows,
                                     cmd->apple9_draw_ring_limited);
      if (!hk_apple9_check_native_block(cmd, cs, before))
         return false;
      if (debug_get_option_native_batch_log())
         fprintf(stderr, "HK_NATIVE_BATCH stream=%" PRIx64 " dispatch=%u cap=%u\n",
                 cs->addr, cs->apple9_native_dispatches,
                 hk_apple9_native_profile(dev).batch);
   } else {
      const uint8_t *before = cs->current;
      hk_apple9_emit_ordinary_draw(cs, pipeline, draw);
      if (!hk_apple9_check_native_block(cmd, cs, before))
         return false;
      if (cs->apple9_native_mesh)
         hk_apple9_add_native_primitives(cs, cmd->apple9_draw_primitives);
   }
   return true;
}

static void
hk_apple9_account_draw(struct hk_cs *cs,
                       const struct agx_apple9_render_pipeline *pipeline,
                       struct agx_draw draw, bool emulated,
                       const struct hk_shader *fragment)
{
   if (cs->primitive_count < AGX_PRIMITIVE_COUNT_UNKNOWN) {
      /* Emulated output counts are not the API draw's primitive count. */
      agx_add_primitive_count(
         &cs->primitive_count,
         agx_is_indirect(draw.b) || pipeline->primitive_restart || emulated
            ? AGX_PRIMITIVE_COUNT_UNKNOWN
            : agx_count_primitives(pipeline->primitive, draw.b.count[0],
                                   draw.b.count[1]));
   }
   cs->stats.cmds++;
   if (!cs->pass_count++)
      cs->pass_code = fragment->apple9_body ? fragment->apple9_body->addr :
                      (fragment->bo ? fragment->bo->va->addr : 0);
}

/* Whether a bound stage reads the draw index from the root table. Other
 * draws leave the root's draw_id unchanged: a multi-draw's identical draws
 * then reuse one root upload instead of uploading it per draw. */
bool
hk_apple9_reads_draw_id(const struct hk_graphics_state *gfx)
{
   for (unsigned s = 0; s < ARRAY_SIZE(gfx->shaders); ++s) {
      if (s != MESA_SHADER_COMPUTE && gfx->shaders[s] && gfx->shaders[s]->reads_draw_id)
         return true;
   }
   return false;
}

/* Reuse preparation, not emission: every draw still records its scope and
 * emits its ordinary VDM command. Root updates preserve the normal upload
 * order and rebuild only the tables/launches that bind the changed root.
 */
static bool
hk_apple9_try_cached_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                          struct agx_draw draw, uint64_t native_grid)
{
   if (!hk_apple9_draw_state_reusable(cmd, draw, native_grid))
      return false;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_apple9_render_pipeline *pipeline = &cmd->apple9_draw_state.pipeline;
   if (draw.indexed) {
      uint64_t offset = (uint64_t)draw.start * agx_index_size_to_B(draw.index_size);
      uint64_t address = draw.index_buffer + offset;
      if (address < dev->dev.shader_base ||
          address >= dev->dev.shader_base + (1ull << 32))
         return false;
      hk_apple9_set_index_fetch(pipeline, draw, draw.index_buffer,
                                draw.index_buffer_range_B, offset);
      pipeline->primitive_restart =
         cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable;
      pipeline->restart_index = gfx->index.restart;
   } else {
      pipeline->index_size = 0;
      pipeline->primitive_restart = false;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   assert(cs && cs == cmd->apple9_prev_cs);

   if (hk_apple9_reads_draw_id(gfx) && desc->root.draw.draw_id != draw_id) {
      desc->root.draw.draw_id = draw_id;
      desc->root_dirty = true;
   }
   if (desc->root_dirty || desc->push_dirty) {
      if (desc->push_dirty)
         hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      uint64_t root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      if (!root)
         return true; /* The pool recorded the allocation failure. */
      if (root != gfx->root) {
         gfx->root = root;
         struct agx_apple9_uniform_draw *record =
            &cmd->apple9_draw[!cmd->apple9_draw_current];
         uint64_t tables[2];
         for (unsigned stage = 0; stage < 2; ++stage) {
            const struct agx_apple9_render_stage *shader =
               stage ? pipeline->fragment : pipeline->vertex;
            for (unsigned slot = 0; slot < shader->resource_count; ++slot) {
               if (shader->resource_binding[slot] == 0)
                  record->buffers[stage][slot] = root;
            }
            tables[stage] = hk_pool_upload_last(
               cmd, stage ? HK_UPLOAD_LAST_FRAGMENT_TABLE : HK_UPLOAD_LAST_VERTEX_TABLE,
               record->buffers[stage], shader->resource_count * sizeof(uint64_t), 64);
         }
         if (!agx_apple9_update_draw_tables_alloc(
                &dev->dev, cmd, hk_apple9_usc_alloc, cmd, hk_apple9_context_alloc,
                pipeline, record, tables)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return true;
         }
         pipeline->ppp = record->ppp;
         pipeline->ppp_viewport_offset = record->ppp_viewport_offset;
         pipeline->ppp_viewport_words = record->ppp_viewport_words;
         pipeline->vertex_launch = record->launch[0] / 0x40;
      }
      desc->root_dirty = false;
   }

   if (!hk_ensure_cs_has_space(cmd, cs, 256))
      return true;
   const uint8_t *before = cs->current;
   hk_apple9_emit_ordinary_draw(cs, pipeline, draw);
   if (!hk_apple9_check_native_block(cmd, cs, before))
      return true;
   if (cs->apple9_native_mesh)
      hk_apple9_add_native_primitives(cs, cmd->apple9_draw_primitives);
   hk_apple9_account_draw(cs, pipeline, draw, false,
                          cmd->apple9_draw_state.fragment);
   return true;
}

static void
hk_apple9_draw_impl(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    struct agx_draw draw, uint64_t native_grid)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* Geometry shaders can produce empty rasterization draws. */
   if (!agx_is_indirect(draw.b) && (!draw.b.count[0] || !draw.b.count[1]))
      return;
   if (hk_apple9_try_cached_draw(cmd, draw_id, draw, native_grid))
      return;
   cmd->apple9_draw_state.valid = false;
   unsigned failure_line = 0;
   struct hk_shader *shaders[2] = {NULL, NULL};
   struct hk_apple9_linked_entry *linked_vs_entry = NULL;
   struct hk_apple9_linked_entry *linked_fs_entry = NULL;
   struct hk_apple9_per_sample_trampoline *tramp_fs_entry = NULL;
#define UNSUPPORTED()                                                          \
   do {                                                                        \
      failure_line = __LINE__;                                                 \
      goto unsupported;                                                        \
   } while (0)
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   bool rasterizer_discard = dyn->rs.rasterizer_discard_enable;
   /* A geometry shader has already run as compute; its rasterization
    * variant is the hardware vertex program for its output primitives.
    */
   struct hk_api_shader *gs =
      hk_apple9_select_gs(cmd, gfx->shaders[MESA_SHADER_GEOMETRY]);
   /* A non-zero rasterization stream discards unless a multistream geometry
    * shader selects that stream.
    */
   if (dyn->rs.rasterization_stream != 0) {
      rasterizer_discard |=
         !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   /* Mesh shaders have written their records as compute. */
   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];
   if (native_grid) {
      mesh = hk_api_shader_native(mesh ?: gs ?: tes);
      if (!mesh || (gs && !hk_apple9_native_stage_available_device(
                             dev, MESA_SHADER_GEOMETRY)) ||
          (tes && !hk_apple9_native_stage_available_device(
                             dev, MESA_SHADER_TESS_EVAL)))
         UNSUPPORTED();
   }
   /* Without an API fragment shader (or when rasterizer discard suppresses
    * rasterization), the null program only applies the API sample mask. A full
    * mask or rasterizer discard needs no fragment work at all. */
   const bool null_fs = rasterizer_discard ||
                        !gfx->shaders[MESA_SHADER_FRAGMENT] ||
                        gfx->shaders[MESA_SHADER_FRAGMENT] == dev->null_fs;
   const unsigned null_samples = MAX2(gfx->render.tilebuffer.nr_samples ?:
                                      dyn->ms.rasterization_samples, 1);
   const bool null_trivial =
      null_fs && dev->null_fs_trivial &&
      (rasterizer_discard ||
       (dyn->ms.sample_mask & BITFIELD_MASK(null_samples)) ==
          BITFIELD_MASK(null_samples));
   struct hk_api_shader *fragment_api = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (native_grid && !null_fs) {
      fragment_api = hk_api_shader_native(fragment_api);
      if (!fragment_api)
         UNSUPPORTED();
   } else if (!null_fs && gs && !gs->is_passthrough &&
              fragment_api && fragment_api->apple9_fs_native_per_vertex) {
      fragment_api = fragment_api->apple9_fs_native_per_vertex;
   }
   shaders[0] =
      native_grid ? hk_native_producer_main(mesh)
      : mesh ? hk_apple9_select_mesh_rast(cmd, mesh)
      : gs  ? &gs->variants[HK_GS_VARIANT_RAST]
      : tes ? &tes->variants[HK_VS_VARIANT_HW]
            : hk_any_variant(gfx->shaders[MESA_SHADER_VERTEX]);
   shaders[1] = hk_any_variant(null_trivial ? dev->null_fs_trivial
                                           : fragment_api ?: dev->null_fs);
   bool allow_line_multiview = hk_apple9_native_mesh_enabled() &&
                               mesh && mesh->vk.stage == MESA_SHADER_MESH;
   if (native_grid &&
       (!mesh ||
        (gfx->shaders[MESA_SHADER_TASK] &&
         (!hk_apple9_native_task_bridge_enabled() ||
          mesh->vk.stage != MESA_SHADER_MESH ||
          !shaders[0]->info.mesh.native_task_bridge)) || (tes && gs) ||
        shaders[0]->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
        (dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL &&
         (!allow_line_multiview || dyn->rs.polygon_mode != VK_POLYGON_MODE_LINE)) ||
        (gfx->render.view_mask &&
         (!allow_line_multiview || !shaders[0]->info.mesh.native_multiview ||
          !shaders[0]->info.apple9_native.writes_layer_viewport))))
      UNSUPPORTED();
   /* Geometry and tessellation have already run as compute, and draw their
    * own output topology. Adjacency primitives without a geometry shader
    * draw a generated list without the adjacent vertices.
    */
   enum mesa_prim api_prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool adjacency = !mesh && !gs && !tes && mesa_prim_has_adjacency(api_prim);
   bool emulated = mesh || gs || tes || adjacency;
   /* Unsupported state must fail recording before any job can be submitted. */
   if (!shaders[0] || !shaders[1] ||
       (!rasterizer_discard &&
        (!dyn->vp.viewport_count ||
         dyn->vp.viewport_count > AGX_APPLE9_MAX_VIEWPORTS ||
         dyn->vp.scissor_count != dyn->vp.viewport_count)) ||
       (!emulated && dyn->ia.primitive_topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
        dyn->ia.primitive_topology !=
           VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA))
      UNSUPPORTED();
   enum mesa_prim gs_prim = mesh  ? (shaders[0]->info.mesh.polygon_points
                                        ? MESA_PRIM_POINTS
                                        : shaders[0]->info.mesh.prim)
                            : gs  ? shaders[0]->info.gs.mode
                            : tes ? gfx->tess.prim
                            : adjacency ? u_reduced_prim(api_prim)
                                  : MESA_PRIM_COUNT;
   bool points = emulated ? gs_prim == MESA_PRIM_POINTS
                          : dyn->ia.primitive_topology ==
                               VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool lines = emulated ? (gs_prim == MESA_PRIM_LINES ||
                            gs_prim == MESA_PRIM_LINE_STRIP)
                   : (dyn->ia.primitive_topology >=
                         VK_PRIMITIVE_TOPOLOGY_LINE_LIST &&
                      dyn->ia.primitive_topology <=
                         VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
   const struct agx_apple9_render_stage *vertex = &shaders[0]->apple9_stage;
   const struct agx_apple9_render_stage *fragment = &shaders[1]->apple9_stage;
   struct agx_shader_body *vs_main_body = shaders[0]->apple9_body;
   uint64_t vs_vary_id = 0;
   if (!null_fs && !native_grid &&
       shaders[0]->b.info.stage == MESA_SHADER_VERTEX &&
       vertex->varying_components) {
      const uint64_t fs_flat =
         shaders[1]->info.fs.interp.flat | VARYING_BIT_PRIMITIVE_ID |
         VARYING_BIT_VIEWPORT | VARYING_BIT_FOGC | VARYING_BIT_TEX6;
      const uint64_t fs_linear =
         (shaders[1]->info.fs.interp.linear | VARYING_BIT_TEX7) & ~fs_flat;
      bool group_mismatch = false;
      for (unsigned s = 0; s < ARRAY_SIZE(vertex->varyings.mask); ++s) {
         if (!vertex->varyings.mask[s])
            continue;
         unsigned want = agx_apple9_varying_group(fs_flat, fs_linear, s);
         if (vertex->varyings.cull_distance_mask &&
             s == VARYING_SLOT_CULL_PRIMITIVE)
            want = 1;
         else if (!vertex->varyings.cull_distance_mask &&
                  (s == VARYING_SLOT_CULL_PRIMITIVE ||
                   s == VARYING_SLOT_CULL_PRIMITIVE + 1))
            want = 2;
         if (vertex->varyings.group[s] != want) {
            group_mismatch = true;
            break;
         }
      }
      if (group_mismatch) {
         struct hk_apple9_vs_vary_variant *vv =
            hk_apple9_get_vs_vary_variant(dev, shaders[0], fs_flat, fs_linear);
         if (!vv)
            UNSUPPORTED();
         cmd->apple9_linked_vs = *vertex;
         cmd->apple9_linked_vs.program_id = vv->program_id;
         cmd->apple9_linked_vs.body = vv->body;
         cmd->apple9_linked_vs.bo = vv->body->bo;
         cmd->apple9_linked_vs.preamble_base = vv->body->addr;
         cmd->apple9_linked_vs.varyings = vv->varyings;
         cmd->apple9_linked_vs.launch_cache = &vv->launch_cache;
         vertex = &cmd->apple9_linked_vs;
         vs_main_body = vv->body;
         vs_vary_id = vv->program_id;
      }
   }
   if (!native_grid && !mesh && !gs && !tes &&
       shaders[0]->b.info.apple9_vs_prolog.uses_prolog) {
      if (!vs_main_body)
         UNSUPPORTED();
      struct agx_apple9_vertex_prolog_key pkey;
      memset(&pkey, 0, sizeof(pkey));
      hk_apple9_copy_vs_prolog_iface(&pkey.iface,
                                     &shaders[0]->b.info.apple9_vs_prolog);
      pkey.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
      pkey.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
      pkey.dynamic_stride = true;
      pkey.unbounded = shaders[0]->info.apple9_vertex_unbounded;
      if (dyn->vi) {
         u_foreach_bit(a, pkey.iface.attribs_read) {
            if (!(dyn->vi->attributes_valid & BITFIELD_BIT(a)))
               continue;
            const struct vk_vertex_attribute_state *attr =
               &dyn->vi->attributes[a];
            const struct vk_vertex_binding_state *binding =
               &dyn->vi->bindings[attr->binding];
            pkey.format[a] = vk_format_to_pipe_format(attr->format);
            pkey.offset[a] = attr->offset;
            pkey.buffer[a] = attr->binding;
            pkey.divisor[a] =
               binding->input_rate != VK_VERTEX_INPUT_RATE_INSTANCE ? 0
               : binding->divisor ? binding->divisor
                                  : AGX_APPLE9_VERTEX_DIVISOR_ZERO;
         }
      }
      struct hk_apple9_vs_prolog *prolog = hk_apple9_get_vs_prolog(dev, &pkey);
      if (!prolog)
         UNSUPPORTED();
      uint16_t pub_count =
         MAX2(vertex->publication_count, prolog->publication_count);
      uint32_t scratch_size =
         MAX2(vertex->scratch_size, prolog->scratch_size);
      uint8_t atomic_frame =
         MAX2(vertex->atomic_frame_size, prolog->atomic_frame_size);
      uint64_t preamble_addr =
         vertex->preamble_size
            ? vs_main_body->addr + vertex->preamble_offset
            : 0;
      linked_vs_entry = hk_apple9_get_linked_entry(
         dev, prolog->body, vs_main_body, NULL, NULL, NULL, preamble_addr,
         MAX2(scratch_size, atomic_frame), pub_count, 0);
      if (!linked_vs_entry)
         UNSUPPORTED();
      if (vertex != &cmd->apple9_linked_vs)
         cmd->apple9_linked_vs = *vertex;
      cmd->apple9_linked_vs.program_id = linked_vs_entry->program_id;
      cmd->apple9_linked_vs.body = &linked_vs_entry->body;
      cmd->apple9_linked_vs.bo = linked_vs_entry->body.bo;
      cmd->apple9_linked_vs.preamble_base = vs_main_body->addr;
      cmd->apple9_linked_vs.publication_count = pub_count;
      cmd->apple9_linked_vs.scratch_size = scratch_size;
      cmd->apple9_linked_vs.atomic_frame_size = atomic_frame;
      cmd->apple9_linked_vs.launch_cache = &linked_vs_entry->launch_cache;
      vertex = &cmd->apple9_linked_vs;
   }
   /* Determine whether linked cull-distance and/or line-stipple fragment
    * prologs are required for this producer/fragment pair. */
   const bool cull_mask = vertex->varyings.cull_distance_mask;
   const unsigned cull_count = cull_mask
      ? ((vertex->varyings.mask[VARYING_SLOT_CULL_PRIMITIVE] & 1) ? 1 : 0)
      : (util_bitcount(vertex->varyings.mask[VARYING_SLOT_CULL_PRIMITIVE]) +
         util_bitcount(vertex->varyings.mask[VARYING_SLOT_CULL_PRIMITIVE + 1]));
   const bool need_cull_prolog =
      !rasterizer_discard &&
      cull_count > 0 &&
      !shaders[1]->b.info.apple9_varyings.mask[VARYING_SLOT_CULL_PRIMITIVE] &&
      !shaders[1]->b.info.apple9_varyings.mask[VARYING_SLOT_CULL_PRIMITIVE + 1];
   const bool need_stipple_prolog =
      !rasterizer_discard &&
      lines && dyn->rs.line.stipple.enable &&
      (vertex->varyings.mask[VARYING_SLOT_TEX6] & 0x7) == 0x7 &&
      (vertex->varyings.mask[VARYING_SLOT_TEX7] & 0x3) == 0x3 &&
      !shaders[1]->info.fs.line_stipple_start &&
      !shaders[1]->b.info.apple9_varyings.mask[VARYING_SLOT_TEX6];
   const bool need_stats_prolog =
      !null_fs &&
      !shaders[1]->info.fs.inline_statistics &&
      (gfx->descriptors.root.draw.pipeline_stats_flags &
       VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT);
   unsigned next_cf =
      (null_fs ? 0 : fragment->varying_components) + 1 +
      (null_fs ? 0 : fragment->reads_primitive_id +
                     fragment->apple9_reads_z +
                     2 * fragment->reads_point_coord +
                     2 * util_bitcount(fragment->barycentric_mask) +
                     fragment->per_vertex_count);
   uint8_t cull_cf_base = 0;
   uint8_t stipple_cf_base = 0;
   if (need_cull_prolog) {
      if (next_cf + (cull_mask ? 1 : cull_count) > 255)
         UNSUPPORTED();
      cull_cf_base = (uint8_t)next_cf;
      next_cf += cull_mask ? 1 : cull_count;
   }
   if (need_stipple_prolog) {
      if (next_cf + 5 > 255)
         UNSUPPORTED();
      stipple_cf_base = (uint8_t)next_cf;
      next_cf += 5;
   }
   /* Without a depth attachment, the depth test is disabled. Early tests keep
    * a hardware NEVER: the shader must not run at all. */
   const bool depth_test =
      dyn->ds.depth.test_enable && gfx->render.depth_att.iview;
   const bool force_never_in_shader =
      depth_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER &&
      shaders[1]->info.fs.writes_memory &&
      !shaders[1]->info.fs.early_fragment_tests;
   if (null_fs && !need_cull_prolog && !need_stipple_prolog) {
      /* The empty fragment program reads no varyings. Retain the producer's
       * layout for the native interface linker, independent of its contents. */
      if (!vertex->program_id ||
          cmd->apple9_null_fs_vertex_id != vertex->program_id ||
          cmd->apple9_null_fs_trivial != null_trivial) {
         cmd->apple9_null_fs_trivial = null_trivial;
         cmd->apple9_null_fs = *fragment;
         cmd->apple9_null_fs.varyings = vertex->varyings;
         cmd->apple9_null_fs.varying_components = vertex->varying_components;
         /* This derived stage has the producer's interface, so its tables
          * belong to this command buffer rather than the device's null FS. */
         struct agx_ptr cf = hk_pool_usc_alloc(
            cmd, agx_apple9_coefficients_size(&cmd->apple9_null_fs), 64);
         if (!cf.cpu)
            UNSUPPORTED();
         agx_apple9_init_coefficients(&dev->dev, &cmd->apple9_null_fs, cf);
         cmd->apple9_null_fs_vertex_id = vertex->program_id;
      }
      fragment = &cmd->apple9_null_fs;
   } else if (null_fs) {
      struct hk_apple9_cull_prolog *cull_p = NULL;
      struct hk_apple9_stipple_prolog *stipple_p = NULL;
      if (need_cull_prolog) {
         struct agx_apple9_cull_prolog_key ckey;
         memset(&ckey, 0, sizeof(ckey));
         ckey.mask = cull_mask;
         ckey.count = (uint8_t)cull_count;
         ckey.cf_base = cull_cf_base;
         ckey.standalone = null_trivial && !need_stipple_prolog;
         cull_p = hk_apple9_get_cull_prolog(dev, &ckey);
         if (!cull_p)
            UNSUPPORTED();
      }
      if (need_stipple_prolog) {
         struct agx_apple9_stipple_prolog_key skey;
         memset(&skey, 0, sizeof(skey));
         skey.cf_base = stipple_cf_base;
         skey.standalone = null_trivial;
         stipple_p = hk_apple9_get_stipple_prolog(dev, &skey);
         if (!stipple_p)
            UNSUPPORTED();
      }
      cmd->apple9_linked_fs = *fragment;
      cmd->apple9_linked_fs.uses_discard = true;
      cmd->apple9_linked_fs.prolog_cull_count =
         need_cull_prolog ? (uint8_t)cull_count : 0;
      cmd->apple9_linked_fs.prolog_cull_mask = need_cull_prolog && cull_mask;
      cmd->apple9_linked_fs.prolog_cull_cf_base = cull_cf_base;
      cmd->apple9_linked_fs.prolog_line_stipple = need_stipple_prolog;
      cmd->apple9_linked_fs.prolog_stipple_cf_base = stipple_cf_base;
      if (need_stipple_prolog) {
         if (cmd->apple9_linked_fs.resource_count == 0) {
            cmd->apple9_linked_fs.resource_count = 1;
            cmd->apple9_linked_fs.resource_binding[0] =
               AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
         } else if (cmd->apple9_linked_fs.resource_binding[0] !=
                    AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
            UNSUPPORTED();
         }
      }
      if (null_trivial && need_cull_prolog && !need_stipple_prolog) {
         cmd->apple9_linked_fs.program_id = cull_p->program_id;
         cmd->apple9_linked_fs.body = cull_p->body;
         cmd->apple9_linked_fs.bo = cull_p->body->bo;
         cmd->apple9_linked_fs.preamble_base = 0;
         cmd->apple9_linked_fs.preamble_size = 0;
         cmd->apple9_linked_fs.publication_count = cull_p->publication_count;
         cmd->apple9_linked_fs.scratch_size = cull_p->scratch_size;
         cmd->apple9_linked_fs.atomic_frame_size = cull_p->atomic_frame_size;
         cmd->apple9_linked_fs.launch_cache = &cull_p->launch_cache;
      } else if (null_trivial && !need_cull_prolog && need_stipple_prolog) {
         cmd->apple9_linked_fs.program_id = stipple_p->program_id;
         cmd->apple9_linked_fs.body = stipple_p->body;
         cmd->apple9_linked_fs.bo = stipple_p->body->bo;
         cmd->apple9_linked_fs.preamble_base = 0;
         cmd->apple9_linked_fs.preamble_size = 0;
         cmd->apple9_linked_fs.publication_count = stipple_p->publication_count;
         cmd->apple9_linked_fs.scratch_size = stipple_p->scratch_size;
         cmd->apple9_linked_fs.atomic_frame_size = stipple_p->atomic_frame_size;
         cmd->apple9_linked_fs.launch_cache = &stipple_p->launch_cache;
      } else {
         struct agx_shader_body *s1 = NULL, *s2 = NULL, *s3 = NULL;
         uint16_t pub_count = 0;
         uint32_t scratch_size = 0;
         uint8_t atomic_frame = 0;
         if (null_trivial) {
            s1 = cull_p->body;
            s2 = stipple_p->body;
            pub_count = MAX2(cull_p->publication_count,
                             stipple_p->publication_count);
            scratch_size = MAX2(cull_p->scratch_size, stipple_p->scratch_size);
            atomic_frame = MAX2(cull_p->atomic_frame_size,
                                stipple_p->atomic_frame_size);
         } else {
            if (!shaders[1]->apple9_body)
               UNSUPPORTED();
            pub_count = fragment->publication_count;
            scratch_size = fragment->scratch_size;
            atomic_frame = fragment->atomic_frame_size;
            if (cull_p && stipple_p) {
               s1 = cull_p->body;
               s2 = stipple_p->body;
               s3 = shaders[1]->apple9_body;
            } else {
               s1 = cull_p ? cull_p->body : stipple_p->body;
               s2 = shaders[1]->apple9_body;
            }
            if (cull_p) {
               pub_count = MAX2(pub_count, cull_p->publication_count);
               scratch_size = MAX2(scratch_size, cull_p->scratch_size);
               atomic_frame = MAX2(atomic_frame, cull_p->atomic_frame_size);
            }
            if (stipple_p) {
               pub_count = MAX2(pub_count, stipple_p->publication_count);
               scratch_size = MAX2(scratch_size, stipple_p->scratch_size);
               atomic_frame = MAX2(atomic_frame, stipple_p->atomic_frame_size);
            }
         }
         linked_fs_entry = hk_apple9_get_linked_entry(
            dev, s1, s2, s3, NULL, NULL, 0, MAX2(scratch_size, atomic_frame),
            pub_count, 1);
         if (!linked_fs_entry)
            UNSUPPORTED();
         cmd->apple9_linked_fs.program_id = linked_fs_entry->program_id;
         cmd->apple9_linked_fs.body = &linked_fs_entry->body;
         cmd->apple9_linked_fs.bo = linked_fs_entry->body.bo;
         cmd->apple9_linked_fs.preamble_base = 0;
         cmd->apple9_linked_fs.preamble_size = 0;
         cmd->apple9_linked_fs.publication_count = pub_count;
         cmd->apple9_linked_fs.scratch_size = scratch_size;
         cmd->apple9_linked_fs.atomic_frame_size = atomic_frame;
         cmd->apple9_linked_fs.launch_cache = &linked_fs_entry->launch_cache;
      }
      fragment = &cmd->apple9_linked_fs;
   } else if (shaders[1]->b.info.apple9_fs_epilog.uses_epilog) {
      if (!shaders[1]->apple9_body)
         UNSUPPORTED();
      struct hk_apple9_cull_prolog *cull_p = NULL;
      struct hk_apple9_stipple_prolog *stipple_p = NULL;
      struct hk_apple9_stats_prolog *stats_p = NULL;
      if (need_cull_prolog) {
         struct agx_apple9_cull_prolog_key ckey;
         memset(&ckey, 0, sizeof(ckey));
         ckey.mask = cull_mask;
         ckey.count = (uint8_t)cull_count;
         ckey.cf_base = cull_cf_base;
         ckey.standalone = false;
         cull_p = hk_apple9_get_cull_prolog(dev, &ckey);
         if (!cull_p)
            UNSUPPORTED();
      }
      if (need_stipple_prolog) {
         struct agx_apple9_stipple_prolog_key skey;
         memset(&skey, 0, sizeof(skey));
         skey.cf_base = stipple_cf_base;
         skey.standalone = false;
         stipple_p = hk_apple9_get_stipple_prolog(dev, &skey);
         if (!stipple_p)
            UNSUPPORTED();
      }
      if (need_stats_prolog) {
         if (shaders[1]->b.info.apple9_fs_epilog.per_sample)
            UNSUPPORTED();
         stats_p = hk_apple9_get_stats_prolog(dev);
         if (!stats_p)
            UNSUPPORTED();
      }
      struct agx_shader_body *prolog1 =
         cull_p ? cull_p->body : (stipple_p ? stipple_p->body : NULL);
      struct agx_shader_body *prolog2 =
         (cull_p && stipple_p) ? stipple_p->body : NULL;
      const unsigned nr_targets = MIN2(gfx->render.color_att_count, 8);
      const unsigned samples =
         dyn->ms.rasterization_samples
            ? dyn->ms.rasterization_samples
            : MAX2(gfx->render.tilebuffer.nr_samples, 1);
      const bool late_zs =
         hk_apple9_draw_late_zs(dyn, &gfx->render, fragment_api) ||
         force_never_in_shader;
      struct agx_apple9_fs_epilog_key ekey;
      memset(&ekey, 0, sizeof(ekey));
      hk_apple9_copy_fs_epilog_iface(&ekey.iface,
                                     &shaders[1]->b.info.apple9_fs_epilog);
      ekey.nr_targets = nr_targets;
      ekey.late_zs =
         late_zs && !ekey.iface.writes_depth && !ekey.iface.early_fragment_tests;
      uint8_t epilog_z_cf_index = 0;
      if (ekey.late_zs) {
         if (fragment->apple9_reads_z) {
            ekey.z_cf_index = (uint8_t)(fragment->varying_components + 1 +
                                        fragment->reads_primitive_id);
         } else {
            if (next_cf >= 255)
               UNSUPPORTED();
            epilog_z_cf_index = (uint8_t)next_cf++;
            ekey.z_cf_index = epilog_z_cf_index;
         }
      }
      ekey.per_sample = ekey.iface.per_sample;
      ekey.prolog_defers_zs_test = need_cull_prolog || need_stipple_prolog;
      for (unsigned rt = 0; rt < nr_targets; ++rt)
         ekey.color_map[rt] = dyn->cal.color_map[rt];
      for (unsigned rt = 0; rt < MAX2(nr_targets, 1); ++rt) {
         struct agx_apple9_blend *b = &ekey.blend[rt];
         b->format = rt < nr_targets && gfx->render.color_att[rt].iview
            ? hk_format_to_pipe_format(gfx->render.color_att[rt].vk_format)
            : PIPE_FORMAT_NONE;
         b->color_attachment_unused =
            b->format == PIPE_FORMAT_NONE ||
            !!(shaders[1]->info.apple9_undefined_colors & BITFIELD_BIT(rt));
         b->samples = samples;
         b->sample_mask_single_sample = true;
         b->disabled_samples =
            (uint8_t)(~dyn->ms.sample_mask & BITFIELD_MASK(samples));
         b->alpha_to_coverage = dyn->ms.alpha_to_coverage_enable;
         b->alpha_to_one = dyn->ms.alpha_to_one_enable;
         b->rgb_src = b->alpha_src = PIPE_BLENDFACTOR_ONE;
         b->rgb_dst = b->alpha_dst = PIPE_BLENDFACTOR_ZERO;
         b->rgb_func = b->alpha_func = PIPE_BLEND_ADD;
         if (rt < nr_targets) {
            const struct vk_color_blend_attachment_state *cb =
               &dyn->cb.attachments[rt];
            b->colormask =
               (dyn->cb.color_write_enables & BITFIELD_BIT(rt)) ? cb->write_mask : 0;
            b->logicop_enable = dyn->cb.logic_op_enable;
            b->logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op);
            if (cb->blend_enable && cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
               b->advanced_mode = vk_advanced_blend_op_to_pipe(cb->color_blend_op);
               b->advanced_overlap = vk_blend_overlap_to_pipe(cb->blend_overlap);
               b->src_premultiplied = cb->src_premultiplied;
               b->dst_premultiplied = cb->dst_premultiplied;
            } else if (cb->blend_enable) {
               b->rgb_src = vk_blend_factor_to_pipe(cb->src_color_blend_factor);
               b->rgb_dst = vk_blend_factor_to_pipe(cb->dst_color_blend_factor);
               b->alpha_src = vk_blend_factor_to_pipe(cb->src_alpha_blend_factor);
               b->alpha_dst = vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor);
               b->rgb_func = vk_blend_op_to_pipe(cb->color_blend_op);
               b->alpha_func = vk_blend_op_to_pipe(cb->alpha_blend_op);
            }
         }
      }
      struct hk_apple9_fs_epilog *epilog = hk_apple9_get_fs_epilog(dev, &ekey);
      if (!epilog)
         UNSUPPORTED();
      uint16_t pub_count =
         MAX2(fragment->publication_count, epilog->publication_count);
      uint32_t scratch_size =
         MAX2(fragment->scratch_size, epilog->scratch_size);
      uint8_t atomic_frame =
         MAX2(fragment->atomic_frame_size, epilog->atomic_frame_size);
      if (cull_p) {
         pub_count = MAX2(pub_count, cull_p->publication_count);
         scratch_size = MAX2(scratch_size, cull_p->scratch_size);
         atomic_frame = MAX2(atomic_frame, cull_p->atomic_frame_size);
      }
      if (stipple_p) {
         pub_count = MAX2(pub_count, stipple_p->publication_count);
         scratch_size = MAX2(scratch_size, stipple_p->scratch_size);
         atomic_frame = MAX2(atomic_frame, stipple_p->atomic_frame_size);
      }
      if (stats_p) {
         pub_count = MAX2(pub_count, stats_p->publication_count);
         scratch_size = MAX2(scratch_size, stats_p->scratch_size);
         atomic_frame = MAX2(atomic_frame, stats_p->atomic_frame_size);
      }
      uint64_t preamble_addr =
         fragment->preamble_size
            ? shaders[1]->apple9_body->addr + fragment->preamble_offset
            : 0;
      cmd->apple9_linked_fs = *fragment;
      if (need_stipple_prolog) {
         if (cmd->apple9_linked_fs.resource_count == 0) {
            cmd->apple9_linked_fs.resource_count = 1;
            cmd->apple9_linked_fs.resource_binding[0] =
               AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
         } else if (cmd->apple9_linked_fs.resource_binding[0] !=
                    AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
            UNSUPPORTED();
         }
      }
      if (need_stats_prolog) {
         if (cmd->apple9_linked_fs.resource_count == 0) {
            cmd->apple9_linked_fs.resource_binding[0] =
               AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
            cmd->apple9_linked_fs.resource_binding[1] = 0;
            cmd->apple9_linked_fs.resource_count = 2;
         } else if (cmd->apple9_linked_fs.resource_binding[0] !=
                    AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
            UNSUPPORTED();
         } else if (cmd->apple9_linked_fs.resource_count == 1) {
            cmd->apple9_linked_fs.resource_binding[1] = 0;
            cmd->apple9_linked_fs.resource_count = 2;
         } else if (cmd->apple9_linked_fs.resource_binding[1] != 0) {
            UNSUPPORTED();
         }
      }
      if (ekey.per_sample) {
         uint16_t tile_access_mode = fragment->image_mask ? 0x700 : 0x600;
         bool emit_tile_fence = nr_targets > 0 || fragment->reads_tile ||
                                epilog->reads_tile || fragment->pixel_interlock;
         bool emit_zs_release = fragment->uses_discard || epilog->uses_discard ||
                                fragment->writes_depth || epilog->writes_depth ||
                                ekey.iface.writes_stencil ||
                                ekey.iface.early_fragment_tests ||
                                ekey.late_zs || ekey.prolog_defers_zs_test;
         tramp_fs_entry = hk_apple9_get_per_sample_trampoline(
            dev, prolog1, prolog2, shaders[1]->apple9_body, epilog->body,
            (uint8_t)samples, tile_access_mode, emit_zs_release, emit_tile_fence,
            preamble_addr, MAX2(scratch_size, atomic_frame), pub_count);
         if (!tramp_fs_entry)
            UNSUPPORTED();
         cmd->apple9_linked_fs.program_id = tramp_fs_entry->program_id;
         cmd->apple9_linked_fs.body = &tramp_fs_entry->body;
         cmd->apple9_linked_fs.bo = tramp_fs_entry->body.bo;
         cmd->apple9_linked_fs.launch_cache = &tramp_fs_entry->launch_cache;
      } else {
         struct agx_shader_body *stages[5] = {NULL};
         unsigned nstages = 0;
         if (cull_p)
            stages[nstages++] = cull_p->body;
         if (stipple_p)
            stages[nstages++] = stipple_p->body;
         if (stats_p)
            stages[nstages++] = stats_p->body;
         stages[nstages++] = shaders[1]->apple9_body;
         stages[nstages++] = epilog->body;
         linked_fs_entry = hk_apple9_get_linked_entry(
            dev, stages[0], stages[1], stages[2], stages[3], stages[4],
            preamble_addr, MAX2(scratch_size, atomic_frame), pub_count, 1);
         if (!linked_fs_entry)
            UNSUPPORTED();
         cmd->apple9_linked_fs.program_id = linked_fs_entry->program_id;
         cmd->apple9_linked_fs.body = &linked_fs_entry->body;
         cmd->apple9_linked_fs.bo = linked_fs_entry->body.bo;
         cmd->apple9_linked_fs.launch_cache = &linked_fs_entry->launch_cache;
      }
      cmd->apple9_linked_fs.preamble_base = shaders[1]->apple9_body->addr;
      cmd->apple9_linked_fs.publication_count = pub_count;
      cmd->apple9_linked_fs.scratch_size = scratch_size;
      cmd->apple9_linked_fs.atomic_frame_size = atomic_frame;
      cmd->apple9_linked_fs.render_targets = nr_targets;
      cmd->apple9_linked_fs.reads_tile = fragment->reads_tile || epilog->reads_tile;
      cmd->apple9_linked_fs.uses_discard =
         fragment->uses_discard || epilog->uses_discard ||
         need_cull_prolog || need_stipple_prolog;
      cmd->apple9_linked_fs.writes_depth =
         fragment->writes_depth || epilog->writes_depth;
      cmd->apple9_linked_fs.prolog_cull_count =
         need_cull_prolog ? (uint8_t)cull_count : 0;
      cmd->apple9_linked_fs.prolog_cull_mask = need_cull_prolog && cull_mask;
      cmd->apple9_linked_fs.prolog_cull_cf_base = cull_cf_base;
      cmd->apple9_linked_fs.prolog_line_stipple = need_stipple_prolog;
      cmd->apple9_linked_fs.prolog_stipple_cf_base = stipple_cf_base;
      cmd->apple9_linked_fs.epilog_z_cf_index = epilog_z_cf_index;
      fragment = &cmd->apple9_linked_fs;
   } else if (need_cull_prolog || need_stipple_prolog) {
      if (!shaders[1]->apple9_body)
         UNSUPPORTED();
      struct hk_apple9_cull_prolog *cull_p = NULL;
      struct hk_apple9_stipple_prolog *stipple_p = NULL;
      if (need_cull_prolog) {
         struct agx_apple9_cull_prolog_key ckey;
         memset(&ckey, 0, sizeof(ckey));
         ckey.mask = cull_mask;
         ckey.count = (uint8_t)cull_count;
         ckey.cf_base = cull_cf_base;
         ckey.standalone = false;
         cull_p = hk_apple9_get_cull_prolog(dev, &ckey);
         if (!cull_p)
            UNSUPPORTED();
      }
      if (need_stipple_prolog) {
         struct agx_apple9_stipple_prolog_key skey;
         memset(&skey, 0, sizeof(skey));
         skey.cf_base = stipple_cf_base;
         skey.standalone = false;
         stipple_p = hk_apple9_get_stipple_prolog(dev, &skey);
         if (!stipple_p)
            UNSUPPORTED();
      }
      cmd->apple9_linked_fs = *fragment;
      if (need_stipple_prolog) {
         if (cmd->apple9_linked_fs.resource_count == 0) {
            cmd->apple9_linked_fs.resource_count = 1;
            cmd->apple9_linked_fs.resource_binding[0] =
               AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
         } else if (cmd->apple9_linked_fs.resource_binding[0] !=
                    AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
            UNSUPPORTED();
         }
      }
      uint16_t pub_count = fragment->publication_count;
      uint32_t scratch_size = fragment->scratch_size;
      uint8_t atomic_frame = fragment->atomic_frame_size;
      if (cull_p) {
         pub_count = MAX2(pub_count, cull_p->publication_count);
         scratch_size = MAX2(scratch_size, cull_p->scratch_size);
         atomic_frame = MAX2(atomic_frame, cull_p->atomic_frame_size);
      }
      if (stipple_p) {
         pub_count = MAX2(pub_count, stipple_p->publication_count);
         scratch_size = MAX2(scratch_size, stipple_p->scratch_size);
         atomic_frame = MAX2(atomic_frame, stipple_p->atomic_frame_size);
      }
      uint64_t preamble_addr =
         fragment->preamble_size
            ? shaders[1]->apple9_body->addr + fragment->preamble_offset
            : 0;
      linked_fs_entry = hk_apple9_get_linked_entry(
         dev, cull_p ? cull_p->body : stipple_p->body,
         (cull_p && stipple_p) ? stipple_p->body : shaders[1]->apple9_body,
         (cull_p && stipple_p) ? shaders[1]->apple9_body : NULL, NULL, NULL,
         preamble_addr, MAX2(scratch_size, atomic_frame), pub_count, 1);
      if (!linked_fs_entry)
         UNSUPPORTED();
      cmd->apple9_linked_fs.program_id = linked_fs_entry->program_id;
      cmd->apple9_linked_fs.body = &linked_fs_entry->body;
      cmd->apple9_linked_fs.bo = linked_fs_entry->body.bo;
      cmd->apple9_linked_fs.preamble_base = shaders[1]->apple9_body->addr;
      cmd->apple9_linked_fs.publication_count = pub_count;
      cmd->apple9_linked_fs.scratch_size = scratch_size;
      cmd->apple9_linked_fs.atomic_frame_size = atomic_frame;
      cmd->apple9_linked_fs.uses_discard = true;
      cmd->apple9_linked_fs.prolog_cull_count =
         need_cull_prolog ? (uint8_t)cull_count : 0;
      cmd->apple9_linked_fs.prolog_cull_mask = need_cull_prolog && cull_mask;
      cmd->apple9_linked_fs.prolog_cull_cf_base = cull_cf_base;
      cmd->apple9_linked_fs.prolog_line_stipple = need_stipple_prolog;
      cmd->apple9_linked_fs.prolog_stipple_cf_base = stipple_cf_base;
      cmd->apple9_linked_fs.launch_cache = &linked_fs_entry->launch_cache;
      fragment = &cmd->apple9_linked_fs;
   }
   const uint64_t link_vs_id =
      vs_vary_id ? (vertex->program_id ^ (vs_vary_id * 0x9e3779b97f4a7c15ull))
                 : vertex->program_id;
   if (!vertex->program_id || !fragment->program_id ||
       cmd->apple9_linked_id[0] != link_vs_id ||
       cmd->apple9_linked_id[1] != fragment->program_id ||
       cmd->apple9_linked.vertex != vertex ||
       cmd->apple9_linked.fragment != fragment) {
      if (!agx_apple9_link_render_pipeline(&cmd->apple9_linked, vertex,
                                           fragment)) {
         cmd->apple9_linked_id[0] = cmd->apple9_linked_id[1] = 0;
         UNSUPPORTED();
      }
      cmd->apple9_linked_id[0] = link_vs_id;
      cmd->apple9_linked_id[1] = fragment->program_id;
      agx_apple9_build_ppp_static(&cmd->apple9_linked, cmd->apple9_ppp_static);
      cmd->apple9_linked.ppp_static = cmd->apple9_ppp_static;
   }
   struct agx_apple9_render_pipeline pipeline = cmd->apple9_linked;
   /* Attachmentless render passes acquire their sample count from the first
    * drawing pipeline. Keep the native tile layout and submission state in
    * agreement with the fragment launch, including a pre-existing query CS.
    */
   if (!gfx->render.tilebuffer.nr_samples && dyn->ms.rasterization_samples) {
      gfx->render.tilebuffer.nr_samples = dyn->ms.rasterization_samples;
      gfx->render.tilebuffer.sample_size_B = ALIGN_POT(
         agx_apple9_tile_sample_bytes(0, dyn->ms.rasterization_samples), 8);
      gfx->render.tilebuffer.tile_size =
         dyn->ms.rasterization_samples == 4 ? 32 * 16 : 32 * 32;
   }
   pipeline.samples = MAX2(gfx->render.tilebuffer.nr_samples, 1);
   for (unsigned rt = 0; rt < gfx->render.color_att_count; ++rt) {
      struct hk_image_view *view = gfx->render.color_att[rt].iview;
      pipeline.color_formats[rt] =
         view ? hk_format_to_pipe_format(view->vk.format)
              : PIPE_FORMAT_R8G8B8A8_UNORM;
   }
   static const uint8_t primitives[] = {
      AGX_PRIMITIVE_POINTS,         AGX_PRIMITIVE_LINES,
      AGX_PRIMITIVE_LINE_STRIP,     AGX_PRIMITIVE_TRIANGLES,
      AGX_PRIMITIVE_TRIANGLE_STRIP, AGX_PRIMITIVE_TRIANGLE_FAN,
   };
   /* Mesa meta rectangles are already expanded to two ordinary triangles. */
   pipeline.primitive =
      emulated ? hk_apple9_rast_primitive(gs_prim)
      : dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA ||
           gfx->apple9_fan_unrolled
         ? AGX_PRIMITIVE_TRIANGLES
         : primitives[dyn->ia.primitive_topology];
   /* Emulated stages emit their primitives for the same provoking mode.
    * Stippled lines without them take the segment start from the first
    * vertex, see hk_apple9_lower_line_stipple_fs. Nothing else observes the
    * change: such fragment shaders read no flat varyings, layer or viewport
    * index (primitive ID does not depend on the provoking vertex), and the
    * root table keeps the API mode for transform feedback and adjacency.
    * Fans that are not unrolled have no observable provoking vertex either,
    * and native fans flat shade consistently only from their first vertex. */
   const bool native_fan =
      !mesh && !gs && !tes &&
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
      !gfx->apple9_fan_unrolled;
   pipeline.flatshade_first =
      dyn->rs.provoking_vertex != VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ||
      native_fan ||
      (!mesh && !gs && !adjacency && shaders[1]->info.apple9_line_stipple_first);
   if (draw.indexed) {
      pipeline.index_size = agx_index_size_to_B(draw.index_size);
      uint64_t offset = agx_is_indirect(draw.b)
                           ? 0
                           : (uint64_t)draw.start * pipeline.index_size;
      uint64_t index_buffer = draw.index_buffer;
      uint64_t index_range = draw.index_buffer_range_B;
      /* Null bindings and reads past the binding fetch index zero. The
       * hardware bounds the fetch by the extent, and the source must lie in
       * the USC aperture, so use a small zeroed pool allocation.
       */
      if (offset >= index_range) {
         struct agx_ptr zero = hk_pool_usc_alloc(cmd, 16, 16);
         if (!zero.cpu)
            goto cleanup;
         memset(zero.cpu, 0, 16);
         index_buffer = zero.gpu;
         index_range = 16;
         offset = 0;
      }
      pipeline.index_buffer = index_buffer + offset;
      if (pipeline.index_buffer < dev->dev.shader_base ||
          pipeline.index_buffer >= dev->dev.shader_base + (1ull << 32))
         UNSUPPORTED();
      hk_apple9_set_index_fetch(&pipeline, draw, index_buffer, index_range, offset);
      pipeline.primitive_restart =
         emulated ? draw.restart : dyn->ia.primitive_restart_enable;
      pipeline.restart_index =
         emulated ? BITFIELD_MASK(pipeline.index_size * 8) : gfx->index.restart;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      goto cleanup;
   cs = hk_occlusion_prepare_draw(cmd, cs);
   if (!cs)
      goto cleanup;
   /* Selection and preparation guarantee a leading native state and blocks
    * the profile admits. Keep this backstop before publishing a malformed
    * stream. */
   bool append = hk_apple9_native_can_append(cmd, cs, !!native_grid);
   bool allow_multilayer_msaa = hk_apple9_native_mesh_enabled();
   uint32_t layers = gfx->render.view_mask
                        ? util_last_bit(gfx->render.view_mask)
                        : gfx->render.layer_count;
   if ((cs->apple9_native_mesh && !append) ||
       (native_grid &&
        ((!append && cs->current != cs->start) ||
         (!allow_multilayer_msaa &&
          (pipeline.samples != 1 || layers != 1)) ||
         (pipeline.samples != 1 && pipeline.samples != 2 &&
          pipeline.samples != 4) ||
         !layers || layers > 2048)))
      UNSUPPORTED();
   cs->tib = gfx->render.tilebuffer;
   const struct vk_dynamic_graphics_state *ms_dyn = &cmd->vk.dynamic_graphics_state;
   gfx->apple9_sample_positions =
      hk_apple9_draw_sample_positions(ms_dyn, pipeline.samples);
   /* hk_apple9_prepare_draw committed the render's positions; meta draws
    * only fill them in for a render that has none yet. */
   if (!cs->has_sample_locations)
      cs->ppp_multisamplectl = gfx->apple9_sample_positions;
   struct agx_apple9_uniform_draw *record =
      &cmd->apple9_draw[cmd->apple9_draw_current];
   agx_apple9_uniform_draw_reset(record);
   /* Rasterizer discard pipelines may omit viewport/scissor state entirely.
    * Supply harmless hardware state without consulting those ignored API
    * values or dropping the draw through the empty-scissor optimization.
    */
   const VkViewport discard_viewport = {.width = 1, .height = 1, .maxDepth = 1};
   const VkRect2D *area = &gfx->render.area;
   unsigned viewports = rasterizer_discard ? 1 : dyn->vp.viewport_count;
   record->viewport_count = viewports;
   /* Viewport scissors are consecutive; the shader's viewport index selects
    * both the transform and the scissor. The tile-granular region clip is
    * their union.
    */
   struct agx_scissor_packed scissors[AGX_APPLE9_MAX_VIEWPORTS];
   int bbox[4] = {INT_MAX, INT_MAX, INT_MIN, INT_MIN};
   for (unsigned v = 0; v < viewports; ++v) {
      const VkViewport *vp =
         rasterizer_discard ? &discard_viewport : &dyn->vp.viewports[v];
      float transform[6] = {
         vp->x + vp->width * 0.5f,  vp->width * 0.5f,
         vp->y + vp->height * 0.5f, vp->height * 0.5f,
         vp->minDepth,              vp->maxDepth - vp->minDepth,
      };
      if (v == 0) {
         for (unsigned axis = 0; axis < 3; ++axis) {
            record->viewport_translate[axis] = transform[axis * 2];
            record->viewport_scale[axis] = transform[axis * 2 + 1];
         }
      } else {
         memcpy(record->viewports[v - 1], transform, sizeof(transform));
      }
      const VkRect2D *sc = rasterizer_discard ? area : &dyn->vp.scissors[v];
      int minx = MAX2(sc->offset.x, area->offset.x);
      int miny = MAX2(sc->offset.y, area->offset.y);
      int maxx = MIN2((int64_t)sc->offset.x + sc->extent.width,
                      (int64_t)area->offset.x + area->extent.width);
      int maxy = MIN2((int64_t)sc->offset.y + sc->extent.height,
                      (int64_t)area->offset.y + area->extent.height);
      /* Clipping has a guard band, so primitives are also scissored to
       * their viewport, which may be inverted.
       */
      if (!rasterizer_discard) {
         minx = MAX2(minx, (int)floorf(MIN2(vp->x, vp->x + vp->width)));
         miny = MAX2(miny, (int)floorf(MIN2(vp->y, vp->y + vp->height)));
         maxx = MIN2(maxx, (int)ceilf(MAX2(vp->x, vp->x + vp->width)));
         maxy = MIN2(maxy, (int)ceilf(MAX2(vp->y, vp->y + vp->height)));
      }
      if (minx >= maxx || miny >= maxy) {
         /* Empty scissor: a zero-area rectangle. */
         maxx = minx = MAX2(minx, 0);
         maxy = miny = MAX2(miny, 0);
      } else {
         bbox[0] = MIN2(bbox[0], minx);
         bbox[1] = MIN2(bbox[1], miny);
         bbox[2] = MAX2(bbox[2], maxx);
         bbox[3] = MAX2(bbox[3], maxy);
      }
      /* Depth clamping bounds depth to the viewport's range, unless a range
       * is requested instead. Unclamped depth is bounded to [0, 1], which
       * is what depthClampZeroOne specifies when clamping is disabled.
       */
      float clamp_min = MIN2(vp->minDepth, vp->maxDepth);
      float clamp_max = MAX2(vp->minDepth, vp->maxDepth);
      if (dyn->vp.depth_clamp_mode ==
          VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT) {
         clamp_min = dyn->vp.depth_clamp_range.minDepthClamp;
         clamp_max = dyn->vp.depth_clamp_range.maxDepthClamp;
      }
      agx_pack(&scissors[v], SCISSOR, cfg) {
         cfg.min_x = minx;
         cfg.min_y = miny;
         cfg.max_x = maxx;
         cfg.max_y = maxy;
         cfg.min_z = dyn->rs.depth_clamp_enable ? clamp_min : 0;
         cfg.max_z = dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
      /* The scissor range bounds fixed-function depth only; shaders clamp
       * their own depth exports (hk_apple9_clamp_depth_export). */
      if (v == 0) {
         gfx->descriptors.root.draw.depth_clamp[0] =
            dyn->rs.depth_clamp_enable ? clamp_min : 0;
         gfx->descriptors.root.draw.depth_clamp[1] =
            dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
   }
   if (bbox[0] >= bbox[2]) {
      /* Every viewport scissor is empty, so nothing rasterizes. The hardware
       * vertex program still owns pre-rasterization side effects: geometry
       * shader stores and atomics run in its rasterization variant (poly
       * emits a culled dummy primitive for them), and so do vertex shader
       * stores and transform feedback. Mesh records were written as compute.
       * Only drop draws without such effects; otherwise run against a
       * zero-area region.
       */
      bool effects = gfx->xfb_enabled;
      for (unsigned s = 0; s < MESA_SHADER_FRAGMENT && !effects; ++s)
         effects = gfx->shaders[s] && gfx->shaders[s]->writes_memory;
      effects |= native_grid && mesh && mesh->writes_memory;
      if (!effects)
         goto cleanup;
      bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0;
   }
   record->scissor_min[0] = bbox[0];
   record->scissor_min[1] = bbox[1];
   record->scissor_max[0] = bbox[2];
   record->scissor_max[1] = bbox[3];
   unsigned scissor_count = cs->scissor.size / AGX_SCISSOR_LENGTH;
   if (scissor_count >= viewports &&
       !memcmp((uint8_t *)cs->scissor.data +
                  (scissor_count - viewports) * AGX_SCISSOR_LENGTH,
               scissors, viewports * sizeof(scissors[0]))) {
      record->scissor_index = scissor_count - viewports;
   } else {
      assert(scissor_count + viewports <= UINT16_MAX);
      record->scissor_index = scissor_count;
      util_dynarray_append_array(&cs->scissor, struct agx_scissor_packed,
                                 scissors, viewports);
   }
   bool bias = dyn->rs.depth_bias.enable && !points && !lines;
   if (bias) {
      bool d16 = gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
      bool exact = dyn->rs.depth_bias.exact;
      bool unorm = d16 || force_unorm;
      /* UNORM representations use a fixed depth unit: 2^-16 for D16 and
       * 2^-24 when forced, or twice that unless exact. Floating depth uses
       * the exponent-dependent unit 2^(e - 23) directly. The submission
       * selects the matching mode.
       */
      float scale = d16         ? 1.0f / (exact ? (1 << 16) : (1 << 15))
                    : force_unorm ? 1.0f / (exact ? (1 << 24) : (1 << 23))
                                  : 1.0f;
      assert(cs->cr.dbias_is_int == (unorm ? U_TRISTATE_YES : U_TRISTATE_NO));
      struct agx_depth_bias_packed depth_bias;
      agx_pack(&depth_bias, DEPTH_BIAS, cfg) {
         cfg.depth_bias = dyn->rs.depth_bias.constant_factor * scale;
         cfg.slope_scale = dyn->rs.depth_bias.slope_factor;
         cfg.clamp = dyn->rs.depth_bias.clamp;
      }
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      if (count && !memcmp((uint8_t *)cs->depth_bias.data +
                              (count - 1) * AGX_DEPTH_BIAS_LENGTH,
                           &depth_bias, sizeof(depth_bias))) {
         record->depth_bias_index = count - 1;
      } else {
         assert(count < UINT16_MAX);
         record->depth_bias_index = count;
         util_dynarray_append(&cs->depth_bias, depth_bias);
      }
   }
   bool stencil = dyn->ds.stencil.test_enable && gfx->render.stencil_att.iview;
   record->depth_control = 0x200 | (1u << 16) | (bias ? 1u << 17 : 0) |
                          (stencil ? 3u << 18 : 0) |
                          (rasterizer_discard ? 1u << 21 : 0);
   record->occlusion_index = gfx->occlusion.index;
   record->visibility_mode = rasterizer_discard ? 0 : gfx->occlusion.mode;
   record->depth_control |= record->visibility_mode << 14;
   gfx->descriptors.root.draw.force_never_in_shader = force_never_in_shader;
   /* The shader fails a forced NEVER test itself (hk_apple9_clamp_depth_export). */
   VkCompareOp depth_compare = !depth_test ? VK_COMPARE_OP_ALWAYS
                               : force_never_in_shader ? VK_COMPARE_OP_LESS
                                                       : dyn->ds.depth.compare_op;
   record->depth_face[0] = record->depth_face[1] =
      (agx_pack_line_width(dyn->rs.line.width) << 8) |
      (depth_compare << 24) |
      ((depth_test && dyn->ds.depth.write_enable) ? 0 : (1 << 21)) |
      (dyn->rs.polygon_mode << 18);
   for (unsigned face = 0; face < 2; ++face) {
      const struct vk_stencil_test_face_state *st =
         face ? &dyn->ds.stencil.back : &dyn->ds.stencil.front;
      record->depth_face[face] |= st->reference & 0xff;
      agx_pack(&record->stencil[face], FRAGMENT_STENCIL, cfg) {
         cfg.compare = stencil ? st->op.compare : AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = stencil ? st->write_mask : 0xff;
         cfg.read_mask = stencil ? st->compare_mask : 0xff;
         cfg.depth_pass = stencil ? st->op.pass : AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = stencil ? st->op.depth_fail : AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = stencil ? st->op.fail : AGX_STENCIL_OP_KEEP;
      }
   }
   struct agx_cull_packed cull;
   agx_pack(&cull, CULL, cfg) {
      cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
      cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
      cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
      cfg.depth_clamp = !cfg.depth_clip;
      cfg.rasterizer_discard = rasterizer_discard;
      /* Bresenham lines cover every sample of a covered pixel. */
      cfg.primitive_msaa =
         lines && dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
   }
   memcpy(&record->raster_control, &cull, 4);
   bool front_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
   /* The hardware-VS TES consumes the tessellator's unswapped indices, so
    * encode its domain winding here. Native TES already applies the existing
    * software-TES index swap, including domain origin: do not flip twice. */
   if (tes && !gs && !native_grid) {
      front_ccw ^= gfx->tess.info.ccw;
      front_ccw ^= dyn->ts.domain_origin ==
                   VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }
   record->raster_control |= front_ccw ? 1u << 16 : 0;
   record->object_type = points ? AGX_OBJECT_TYPE_POINT_SPRITE_UV01
                        : lines ? AGX_OBJECT_TYPE_LINE
                                : AGX_OBJECT_TYPE_TRIANGLE;
   record->reads_tile = fragment->reads_tile;
   record->uses_discard = fragment->uses_discard;
   record->writes_depth = fragment->writes_depth;
   record->disable_tri_merging = fragment->disable_tri_merging;
   record->pixel_interlock = fragment->pixel_interlock;
   record->flatshade_first = pipeline.flatshade_first;
   gfx->descriptors.root.draw.provoking =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2 : 0;
   gfx->descriptors.root.draw.producer_writes_viewport =
      !!(vertex->varyings.mask[VARYING_SLOT_VIEWPORT] & 1);
   gfx->descriptors.root.draw.polygon_point_facing =
      dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT &&
      (vertex->varyings.mask[VARYING_SLOT_FOGC] & 1);
   gfx->descriptors.root.draw.depth_clamp_enable =
      dyn->rs.depth_clamp_enable ? 1 : 0;
   gfx->descriptors.root.draw.api_sample_mask =
      dyn->ms.sample_mask & BITFIELD_MASK(pipeline.samples);
   gfx->descriptors.root.draw.samples_log2 = util_logbase2(pipeline.samples);
   gfx->descriptors.root.draw.clip_z_coeff =
      dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
   gfx->descriptors.root.draw.rasterized_prim =
      (points ? 2 : lines ? 1 : 0) |
      (!emulated &&
       dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
       dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
          ? 4 : 0);
   if (hk_apple9_reads_draw_id(gfx))
      gfx->descriptors.root.draw.draw_id = draw_id;
   gfx->descriptors.root_dirty = true;
   uint64_t t_sub = hk_cpu_sub_begin();
   hk_cmd_buffer_flush_push_descriptors(cmd, &gfx->descriptors);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_PUSH_DESC);
   t_sub = hk_cpu_sub_begin();
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_ROOT);
   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);
   uint64_t draw_params = bindings.draw_params;
   uint64_t vertex_table = bindings.vertex_buffers;
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct agx_apple9_render_stage *s =
         stage ? fragment : vertex;
      uint64_t sysvals = 0;
      t_sub = hk_cpu_sub_begin();
      bool textures_ok = hk_apple9_upload_textures(
         cmd, shaders[stage], true,
         pipeline.primitive == AGX_PRIMITIVE_TRIANGLE_FAN, lines,
         &record->texture_table[stage], &record->sampler_table[stage], &sysvals);
      hk_cpu_sub_end(t_sub, HK_CPU_SUB_TEXTURES);
      if (!textures_ok)
         UNSUPPORTED();
      t_sub = hk_cpu_sub_begin();
      for (unsigned slot = 0; slot < s->resource_count; ++slot) {
         unsigned binding = s->resource_binding[slot];
         uint64_t address;
         if (binding == 0)
            address = gfx->root;
         else if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
            address = sysvals;
         else if (binding == HK_APPLE9_DRAW_PARAMS_BINDING)
            address = draw_params;
         else if (binding == HK_APPLE9_VERTEX_BUFFERS_BINDING)
            address = vertex_table;
         else if (binding >= 32 && binding < 32 + ARRAY_SIZE(gfx->vb))
            address = gfx->vb[binding - 32].addr;
         else
            UNSUPPORTED();
         record->buffers[stage][slot] = address;
      }
      uint64_t table = hk_pool_upload_last(
         cmd, stage ? HK_UPLOAD_LAST_FRAGMENT_TABLE : HK_UPLOAD_LAST_VERTEX_TABLE,
         record->buffers[stage], s->resource_count * sizeof(uint64_t), 64);
      if (stage)
         record->fragment_table = table;
      else
         record->vertex_table = table;
      hk_cpu_sub_end(t_sub, HK_CPU_SUB_TABLES);
   }
   const struct agx_apple9_uniform_draw *previous =
      cmd->apple9_prev_cs == cs ? &cmd->apple9_draw[!cmd->apple9_draw_current]
                                : NULL;
   t_sub = hk_cpu_sub_begin();
   bool prepared = agx_apple9_prepare_draw_alloc(
      &dev->dev, cmd, hk_apple9_usc_alloc, cmd, hk_apple9_context_alloc,
      &pipeline, record, previous);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_PREPARE_ALLOC);
   if (!prepared) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto cleanup;
   }
   cmd->apple9_draw_current = !cmd->apple9_draw_current;
   cmd->apple9_prev_cs = cs;
   pipeline.ppp = record->ppp;
   pipeline.ppp_viewport_offset = record->ppp_viewport_offset;
   pipeline.ppp_viewport_words = record->ppp_viewport_words;
   pipeline.vertex_launch = record->launch[0] / 0x40;
   pipeline.pipeline_word = 0x01000000;
   if (!hk_ensure_cs_has_space(cmd, cs, 256))
      goto cleanup;
   if (!hk_apple9_record_draw(cmd, cs, &pipeline, draw, native_grid,
                               mesh, shaders[0], record)) {
      if (vk_command_buffer_has_error(&cmd->vk))
         goto cleanup;
      UNSUPPORTED();
   }
   hk_apple9_account_draw(cs, &pipeline, draw, gs || tes || mesh, shaders[1]);
   /* Only ordinary single-view draws reuse prepared state. Other paths
    * still consume dynamic state in their software-stage lowering. */
   if (hk_apple9_draw_cache_supported(cmd, draw, native_grid)) {
      cmd->apple9_draw_state.valid = true;
      cmd->apple9_draw_state.render_gen = gfx->render_gen;
      cmd->apple9_draw_state.first_vertex = draw.indexed ? draw.index_bias : draw.start;
      cmd->apple9_draw_state.base_instance = draw.start_instance;
      cmd->apple9_draw_state.pipeline = pipeline;
      cmd->apple9_draw_state.fragment = shaders[1];
      /* Every dynamic, hk and shader dirty bit is consumed here, so the
       * next clean draw can reuse this state. G16 has no other dirty-bit
       * readers after this point (only shader resolution and the sysval
       * upload, both above); a new one must read its bits before this or
       * invalidate apple9_draw_state itself. */
      vk_dynamic_graphics_state_clear_dirty(dyn);
      gfx->shaders_dirty = 0;
      gfx->dirty = 0;
      gfx->descriptors.root_dirty = false;
   }
cleanup:
   if (linked_vs_entry)
      agx_shader_body_unreference(&dev->dev, &linked_vs_entry->body);
   if (linked_fs_entry)
      agx_shader_body_unreference(&dev->dev, &linked_fs_entry->body);
   if (tramp_fs_entry)
      agx_shader_body_unreference(&dev->dev, &tramp_fs_entry->body);
   return;
unsupported:
   fprintf(
      stderr,
      "Apple9 draw unsupported at %u: native=%d shaders=%d/%d indirect=%d vp=%u sc=%u stencil=%d bias=%d topology=%u\n",
      failure_line, !!native_grid, !!shaders[0], !!shaders[1], agx_is_indirect(draw.b),
      dyn->vp.viewport_count, dyn->vp.scissor_count,
      dyn->ds.stencil.test_enable, dyn->rs.depth_bias.enable,
      dyn->ia.primitive_topology);
#undef UNSUPPORTED
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   goto cleanup;
}

void
hk_apple9_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id, struct agx_draw draw)
{
   hk_apple9_draw_impl(cmd, draw_id, draw, 0);
}

void
hk_apple9_draw_native_mesh(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                            uint64_t grid)
{
   hk_apple9_draw_impl(cmd, draw_id,
                       (struct agx_draw){.b = agx_3d(3, 1, 1)}, grid);
}
