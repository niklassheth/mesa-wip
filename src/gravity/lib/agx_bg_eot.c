/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot.h"
#include "agx_apple9_libagx.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "agx_compile_apple9.h"
#include "agx_device.h"
#include "nir.h"

void
agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev)
{
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->mem_ctx = ralloc_context(NULL);
   cache->dev = dev;
}

void
agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache)
{
   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      if (cache->precomp[i])
         agx_bo_unreference(cache->dev, cache->precomp[i]->bo);
   }
   ralloc_free(cache->mem_ctx);
   simple_mtx_destroy(&cache->lock);
   cache->mem_ctx = NULL;
   cache->dev = NULL;
}

/* Returns false when the program's USC allocation fails (the aperture is
 * full); nothing is cached then, so a later call can retry. */
static bool
agx_compile_apple9_precompiled(struct agx_bg_eot_cache *cache,
                               struct agx_precompiled_shader *p,
                               unsigned program)
{
   nir_shader *nir = agx_apple9_libagx_nir(program, NULL);

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_tiny(nir, &part, &p->apple9_profile, &reason)) {
      fprintf(stderr, "Apple9 internal program %u (%s) compile failed: %s\n",
              program, nir->info.name ?: "unnamed", reason ?: "unknown");
      nir_print_shader(nir, stderr);
      abort();
   }
   ralloc_free(nir);
   p->apple9 = true;
   p->bo = agx_bo_create(cache->dev, part.info.binary_size, 0,
                         AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                         "Apple9 internal shader");
   if (!p->bo) {
      free(part.binary);
      return false;
   }
   memcpy(agx_bo_map(p->bo), part.binary, part.info.binary_size);
   agx_bo_note_cpu_write(p->bo, 0, part.info.binary_size);
   p->ptr = p->bo->va->addr;
   free(part.binary);
   return true;
}

static struct agx_precompiled_shader *
agx_get_precompiled_locked(struct agx_bg_eot_cache *cache, unsigned program)
{
   simple_mtx_assert_locked(&cache->lock);

   /* It is possible that, while waiting for the lock, another thread uploaded
    * the shader. Check for that so we don't double-upload.
    */
   if (cache->precomp[program])
      return cache->precomp[program];

   struct agx_precompiled_shader *p =
      rzalloc(cache->mem_ctx, struct agx_precompiled_shader);

   if (!agx_compile_apple9_precompiled(cache, p, program)) {
      ralloc_free(p);
      return NULL;
   }

   /* We must only write to the cache once we are done compiling, since other
    * threads may be reading the cache concurrently. Do this last.
    */
   p_atomic_set(&cache->precomp[program], p);
   return p;
}

struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program)
{
   /* Shaders are immutable once written, so if we atomically read a non-NULL
    * shader, then we have a valid cached shader and are done.
    */
   struct agx_precompiled_shader *ret = p_atomic_read(cache->precomp + program);

   if (ret != NULL)
      return ret;

   /* Otherwise, take the lock and upload. */
   simple_mtx_lock(&cache->lock);
   ret = agx_get_precompiled_locked(cache, program);
   simple_mtx_unlock(&cache->lock);

   return ret;
}
