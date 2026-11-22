/*
 * Copyright 2017-2019 Lyude Paul
 * Copyright 2017-2019 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

/* TEMP_STUB_PHASE1: Delete after vk-core-strip removes remaining decode.h
 * includes and agxdecode_* calls from hk_device.h, hk_queue.c, and
 * tests/hk-native-mesh-open.c.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "drm-uapi/asahi_drm.h"

struct agx_bo;
struct agxdecode_ctx;
struct util_dynarray;

static inline struct agxdecode_ctx *
agxdecode_new_context(uint64_t shader_base)
{
   (void)shader_base;
   return NULL;
}

static inline void
agxdecode_destroy_context(struct agxdecode_ctx *ctx)
{
   (void)ctx;
}

static inline void
agxdecode_next_frame(void)
{
}

static inline void
agxdecode_drm_cmdbuf(struct agxdecode_ctx *ctx,
                     struct drm_asahi_params_global *params,
                     struct util_dynarray *cmdbuf, bool verbose)
{
   (void)ctx;
   (void)params;
   (void)cmdbuf;
   (void)verbose;
}

static inline void
agxdecode_track_alloc(struct agxdecode_ctx *ctx, struct agx_bo *alloc)
{
   (void)ctx;
   (void)alloc;
}

static inline void
agxdecode_track_free(struct agxdecode_ctx *ctx, struct agx_bo *bo)
{
   (void)ctx;
   (void)bo;
}
