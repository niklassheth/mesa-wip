/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "gravity/lib/agx_abi.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "agx_pack.h"
#include "libagx_dgc.h"

/* Convert the XFB byte counter on the GPU so that the ordinary indirect
 * draw path also handles instancing, conditional rendering and GS/TES.
 * Subtract before dividing, saturating instead of wrapping below the offset.
 */
KERNEL(1)
libagx_draw_xfb(global VkDrawIndirectCommand *out,
                constant uint32_t *counter, uint32_t counter_offset,
                uint32_t vertex_stride, uint32_t instance_count,
                uint32_t first_instance)
{
   uint32_t bytes = *counter;
   *out = (VkDrawIndirectCommand){
      .vertexCount = (bytes > counter_offset ? bytes - counter_offset : 0) /
                     vertex_stride,
      .instanceCount = instance_count,
      .firstVertex = 0,
      .firstInstance = first_instance,
   };
}

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
KERNEL(32)
libagx_predicate_indirect(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *draw_count, uint32_t stride_el,
                          uint indexed__2)
{
   uint draw = cl_global_id.x;
   uint words = indexed__2 ? 5 : 4;
   bool enabled = draw < *draw_count;
   out += draw * words;
   in += draw * stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < words; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/*
 * Map the index of a vertex in a primitive list without adjacency to its
 * index in the adjacency topology, as in the VS prolog lowering.
 */
static uint
libagx_map_from_adj(uint id, enum mesa_prim prim, bool flatshade_first)
{
   if (prim == MESA_PRIM_LINES_ADJACENCY) {
      /* (1, 2), (5, 6), (9, 10), ... */
      return ((id & ~1) * 2) + (id & 1) + 1;
   } else if (prim == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      /* (1, 2), (2, 3), (3, 4), ... */
      return (id / 2) + (id & 1) + 1;
   } else if (prim == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      /* (0, 2, 4), (2, 6, 4), (4, 6, 8), (6, 10, 8), ... with the first
       * vertex provoking. With the last vertex provoking, odd triangles are
       * (2, 0, 4), (6, 4, 8), ... so that vertex 2 + 2 * tri stays last.
       */
      uint tri = id / 3;
      uint vtx = id % 3;
      if ((tri % 2) == 1) {
         if (flatshade_first)
            vtx = vtx == 0 ? 0 : 3 - vtx;
         else
            vtx = vtx == 2 ? 2 : 1 - vtx;
      }
      return 2 * (tri + vtx);
   } else {
      /* (0, 2, 4), (6, 8, 10), ... */
      return id * 2;
   }
}

/*
 * Draw adjacency primitives without a geometry shader: only a geometry
 * shader observes the adjacent vertices. Write an index buffer of the
 * remaining vertices into the heap and an indexed indirect draw of it,
 * preserving the vertex offset and instances.
 */
KERNEL(1)
libagx_unroll_adjacency(global uint32_t *out_draw, global struct poly_heap *heap,
                        constant uint32_t *in_draw, uint64_t index_buffer,
                        uint32_t index_buffer_range_el, uint32_t index_size_B,
                        uint32_t prim, uint32_t flatshade_first)
{
   bool indexed = index_size_B != 0;
   uint count = libagx_remap_adj_count(in_draw[0], prim);
   uint offs_B = 0;
   if (!poly_heap_try_alloc_offs(heap, count * 4, &offs_B)) {
      /* An exhausted heap draws nothing. */
      for (uint i = 0; i < 5; ++i)
         out_draw[i] = 0;
      return;
   }
   global uint32_t *out = (global uint32_t *)(heap->base + offs_B);

   uint64_t in_indices = 0;
   uint in_range_el = 0;
   if (indexed) {
      in_indices = poly_index_buffer(index_buffer, index_buffer_range_el,
                                     in_draw[2], index_size_B);
      in_range_el = poly_index_buffer_range_el(index_buffer_range_el,
                                               in_draw[2]);
   }

   for (uint i = 0; i < count; ++i) {
      uint raw = libagx_map_from_adj(i, prim, flatshade_first);
      out[i] = indexed ? poly_load_index(in_indices, in_range_el, raw,
                                         index_size_B)
                       : in_draw[2] + raw;
   }

   out_draw[0] = count;
   out_draw[1] = in_draw[1];
   out_draw[2] = offs_B / 4;
   out_draw[3] = indexed ? in_draw[3] : 0;
   out_draw[4] = indexed ? in_draw[4] : in_draw[3];
}

/* Apple9 bounds the hardware index fetch in whole words. Copy the indices of a
 * draw whose binding ends inside a word, reading zero past the binding.
 */
KERNEL(1)
libagx_robust_index_apple9(global uint32_t *out_draw,
                           global struct poly_heap *heap,
                           constant uint32_t *in_draw, uint64_t index_buffer,
                           uint32_t index_buffer_range_el,
                           uint32_t index_size_B)
{
   uint first = in_draw[2];
   uint range_el = poly_index_buffer_range_el(index_buffer_range_el, first);
   uint64_t in = poly_index_buffer(index_buffer, index_buffer_range_el, first,
                                   index_size_B);

   /* Zeroes past a few primitives only repeat the same degenerate one. */
   uint count = min(in_draw[0], range_el + 32);
   uint offs_B = 0;
   if (!poly_heap_try_alloc_offs(heap, (count * index_size_B + 3) & ~3,
                                 &offs_B)) {
      /* An exhausted heap draws nothing. */
      for (uint i = 0; i < 5; ++i)
         out_draw[i] = 0;
      return;
   }
   uintptr_t out = (uintptr_t)heap->base + offs_B;

   for (uint i = 0; i < count; ++i) {
      poly_store_index(out, index_size_B, i,
                       poly_load_index(in, range_el, i, index_size_B));
   }

   out_draw[0] = count;
   out_draw[1] = in_draw[1];
   out_draw[2] = offs_B / index_size_B;
   out_draw[3] = in_draw[3];
   out_draw[4] = in_draw[4];
}

/* Conditional rendering: copy indirect draw or dispatch arguments, or zero
 * them when the predicate fails, so the command does no work.
 */
KERNEL(1)
libagx_predicate_indirect_apple9(global uint32_t *out, constant uint32_t *in,
                                 constant uint32_t *predicate,
                                 uint32_t inverted, uint32_t words)
{
   bool pass = (*predicate != 0) != (inverted != 0);
   for (uint i = 0; i < words; ++i)
      out[i] = pass ? in[i] : 0;
}
