/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_cmd_buffer.h"
#include "hk_shader.h"

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

/* hk_cmd_render.c */
bool hk_feedback_prepare_draw(struct hk_cmd_buffer *cmd);

/* hk_cmd_passthrough_gs.c */
void hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw);

/* hk_cmd_draw.c */
enum mesa_prim hk_gs_in_prim(struct hk_cmd_buffer *cmd);
