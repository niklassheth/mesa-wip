/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* See compiler/README.md for the ABI */

#define AGX_ABI_VUNI_VBO_BASE(vbo) (4 * (vbo))
#define AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, vbo)                                   \
   (AGX_ABI_VUNI_VBO_BASE(nr_vbos) + 2 * (vbo))

#define AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos)                                     \
   AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, nr_vbos)

#define AGX_ABI_VUNI_BASE_INSTANCE(nr_vbos)                                    \
   (AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 2)

#define AGX_ABI_VUNI_DRAW_ID(nr_vbos) (AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 4)

#define AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos)                                    \
   (AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 8)

/* This address is in our reservation, and can be
 * addressed with only small integers in the low/high. That lets us do some
 * robustness optimization even without soft fault.
 */
#define AGX_ZERO_PAGE_ADDRESS (((uint64_t)1) << 32)
#define AGX_ZERO_PAGE_SIZE    (16384)

#define AGX_SCRATCH_PAGE_ADDRESS (AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE)
