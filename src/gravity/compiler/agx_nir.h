/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

struct nir_shader;

bool agx_nir_lower_sink_address(struct nir_shader *shader);
bool agx_nir_lower_accurate_frcp(struct nir_shader *s);

bool agx_nir_lower_texture_early(struct nir_shader *s, bool support_lod_bias,
                                 bool software_buffer_bounds);
bool agx_nir_lower_images(struct nir_shader *s);
bool agx_nir_lower_texture_queries(struct nir_shader *s);
bool agx_nir_lower_multisampled_image_store(struct nir_shader *s);
