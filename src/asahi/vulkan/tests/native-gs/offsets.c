/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "hk_cmd_buffer.h"
#include "poly/geometry.h"
int main(void)
{
   printf("{\"vertex_params\":%zu,\"vertex_outputs\":%zu,\"geometry_params\":%zu,\"api_gs\":%zu,\"draw_id\":%zu,\"params_output\":%zu,\"params_mask\":%zu,\"params_vertices\":%zu,\"params_grid\":%zu,\"params_size\":%zu}\n",
      offsetof(struct hk_root_descriptor_table, draw.vertex_params),
      offsetof(struct hk_root_descriptor_table, draw.vertex_outputs),
      offsetof(struct hk_root_descriptor_table, draw.geometry_params),
      offsetof(struct hk_root_descriptor_table, draw.api_gs),
      offsetof(struct hk_root_descriptor_table, draw.draw_id),
      offsetof(struct poly_vertex_params, output_buffer),
      offsetof(struct poly_vertex_params, outputs),
      offsetof(struct poly_vertex_params, verts_per_instance),
      offsetof(struct poly_vertex_params, grid), sizeof(struct poly_vertex_params));
}
