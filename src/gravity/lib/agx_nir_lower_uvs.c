/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

/* TEMP_STUB_PHASE1: Delete after vk-shader-draw removes the Apple7/8
 * agx_nir_lower_uvs caller in hk_shader.c.
 */

#include "util/macros.h"
#include "agx_uvs.h"

bool
agx_nir_lower_uvs(struct nir_shader *s, struct agx_unlinked_uvs_layout *layout)
{
   (void)s;
   (void)layout;
   UNREACHABLE("Apple7/8 UVS lowering removed in Gravity");
}
