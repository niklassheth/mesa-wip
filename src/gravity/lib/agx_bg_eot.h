/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "gravity/compiler/agx_apple9_profile.h"
#include "util/simple_mtx.h"
#include "agx_bo.h"
#include "libagx_shaders.h"

struct agx_device;

struct agx_precompiled_shader {
   struct agx_bo *bo;
   uint64_t ptr;
   bool apple9;
   struct agx_apple9_compute_profile apple9_profile;
};

struct agx_bg_eot_cache {
   struct agx_device *dev;
   simple_mtx_t lock;

   /* TEMP_STUB_PHASE1: ralloc context for cached entries; hk_device.c still
    * checks `!dev->bg_eot.ht`. Rename or drop check in Gate 1.
    */
   void *ht;

   struct agx_precompiled_shader *precomp[LIBAGX_NUM_PROGRAMS];
};

/*
 * Get a precompiled Apple9 internal compute shader, compiling and uploading on
 * first use. Thread-safe.
 */
struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program);

void agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev);
void agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache);
