/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "hk_cmd_buffer.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#define P(name, type, field) printf("\"" name "\":%zu,", offsetof(struct type, field))
int main(void)
{
   printf("{");
   P("vertex_params", hk_root_descriptor_table, draw.vertex_params);
   P("vertex_outputs", hk_root_descriptor_table, draw.vertex_outputs);
   P("geometry_params", hk_root_descriptor_table, draw.geometry_params);
   P("tess_params", hk_root_descriptor_table, draw.tess_params);
   P("params_output", poly_vertex_params, output_buffer);
   P("params_mask", poly_vertex_params, outputs);
   P("params_vertices", poly_vertex_params, verts_per_instance);
   P("params_grid", poly_vertex_params, grid);
   P("tess_heap", poly_tess_params, heap);
   P("heap_bottom", poly_heap, bottom);
   P("tess_out_draws", poly_tess_params, out_draws);
   P("tess_tcs_buffer", poly_tess_params, tcs_buffer);
   P("tess_counts", poly_tess_params, counts);
   P("tess_indices", poly_tess_params, index_buffer);
   P("tess_coord_allocs", poly_tess_params, coord_allocs);
   P("tess_nr_patches", poly_tess_params, nr_patches);
   P("tess_input_patch", poly_tess_params, input_patch_size);
   P("tess_output_patch", poly_tess_params, output_patch_size);
   P("tess_stride", poly_tess_params, tcs_stride_el);
   P("tess_partitioning", poly_tess_params, partitioning);
   P("tess_ccw", poly_tess_params, ccw);
   printf("\"draw_uniform_offset\":%u,\"tess_size\":%zu,\"schema\":1}\n",
           AGX_APPLE9_MESH_DRAW_UNIFORM * 4, sizeof(struct poly_tess_params));
}
