/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh.h"
#include <string.h>

bool
agx_apple9_mesh_state_build(uint32_t out[AGX_APPLE9_MESH_STATE_WORDS],
                            const struct agx_apple9_mesh_state_parameters *p)
{
   if (!out || !p || !p->layout)
      return false;
   const struct agx_apple9_mesh_layout *l = p->layout;
   /* Task payload and per-primitive outputs need separate completion and
    * PPP contracts; state field encodings alone do not qualify them. */
   if (p->task_payload_size_B || p->primitive_output.components ||
       p->primitive_output.stride_B || l->primitive_components ||
       l->primitive_stride_B)
      return false;
   unsigned n = l->vertex_components + (l->primitive_components != 0);
   if (!l->max_vertices || l->max_vertices > 256 || !l->max_primitives ||
       l->max_primitives > 512 ||
       !l->indices_per_primitive || l->indices_per_primitive > 3 ||
       l->vertex_components < 4 || l->vertex_components > 253 || n > 254 ||
       l->primitive_components > 128 || !p->threads || p->threads > 256 ||
       l->local_size_B > AGX_APPLE9_MAX_SHARED_BYTES ||
       l->local_size_B < l->cull_B || (l->local_size_B - l->cull_B) % 16 ||
       (p->object_launch & 1) || (p->mesh_launch & 1))
      return false;

   /* Measured output contract, shared with body lowering. Rows cover the
    * cull region and the vertex/link planes, excluding indices/count header.
    * Culling and broader output classes still need normal completion support. */
   uint32_t words[AGX_APPLE9_MESH_STATE_WORDS] = {
      0x40000d2c,
      ((n + 1) << 8) | n,
      l->primitive_components ? 0xd0 : 0x10,
      0,
      0,
      0x01000000,
      p->object_launch,
      0x01000000,
      p->mesh_launch,
      p->has_object_stage,
      (l->max_primitives - 1) |
         ((DIV_ROUND_UP(l->max_primitives, 16) - 1) << 16) |
         (l->culling ? BITFIELD_BIT(30) : 0),
      /* Direct count field: own-MSL DATA measures 0x400900 at 256 threads
       * (and 0x400a00 at 512). Bit 8 belongs to the count, not another field.
       * Large-output completion is independent of this encoding evidence. */
      0x400800 | p->threads,
      l->max_primitives * l->primitive_components,
      1 + (l->local_size_B - l->cull_B) / 16,
   };
   memcpy(out, words, sizeof(words));
   return true;
}

/* T8132's per-pipe mesh output ring holds 4,096 16-byte rows (64 KiB).
 * When agx_apple9_mesh_safe_u0_b() finds a wrap-safe U0_B stride
 * (4096 % stride == 0 || 4096 % stride >= actual_vtx_rows), common native
 * mesh completion passes that constant U0_B to MESH_CONTROL for every
 * non-empty workgroup, making all grid sizes concurrent-safe regardless of
 * runtime vertex counts (including wg_rows == 32 and wg_rows > 40).
 * Layouts too large for a wrap-safe stride <= 2048 fall back to bounded
 * admission where cumulative ring occupancy stays within 4,096 rows. */
static bool
agx_apple9_mesh_wg_rows(const struct agx_apple9_mesh_layout *layout,
                        uint64_t *wg_rows)
{
   if (!layout || !wg_rows)
      return false;
   uint64_t n =
      (uint64_t)layout->vertex_components + (layout->primitive_components != 0);
   uint64_t quads = ((uint64_t)layout->max_vertices + 3u) / 4u;
   uint64_t planes = (n + 3u) & ~UINT64_C(3);
   uint64_t header = MAX2(agx_apple9_mesh_idx_rows(layout),
                          (uint64_t)AGX_APPLE9_MESH_RING_HEADER_ROWS);
   if (header > AGX_APPLE9_MESH_RING_ROWS)
      return false;
   if (quads && planes > (AGX_APPLE9_MESH_RING_ROWS - header) / quads)
      return false;
   *wg_rows = quads * planes + header;
   return true;
}

bool
agx_apple9_mesh_ring_max_groups(const struct agx_apple9_mesh_layout *layout,
                                uint64_t *max_groups)
{
   uint64_t wg_rows;
   if (!max_groups || !agx_apple9_mesh_wg_rows(layout, &wg_rows))
      return false;
   uint32_t safe_u0 = 0, safe_stride = 0;
   if (agx_apple9_mesh_safe_u0_b(layout, &safe_u0, &safe_stride)) {
      *max_groups = UINT64_MAX;
      return true;
   }
   *max_groups = AGX_APPLE9_MESH_RING_ROWS / wg_rows;
   return true;
}

bool
agx_apple9_mesh_ring_usage(const struct agx_apple9_mesh_layout *layout,
                           uint64_t groups, uint64_t *rows, bool *limited)
{
   uint64_t wg_rows;
   if (!rows || !limited || !agx_apple9_mesh_wg_rows(layout, &wg_rows))
      return false;
   uint32_t safe_u0 = 0, safe_stride = 0;
   if (agx_apple9_mesh_safe_u0_b(layout, &safe_u0, &safe_stride)) {
      *limited = false;
      uint64_t stride = safe_stride ? safe_stride : wg_rows;
      *rows = groups > UINT64_MAX / stride ? UINT64_MAX : groups * stride;
      return true;
   }
   *limited = true;
   *rows = groups > UINT64_MAX / wg_rows ? UINT64_MAX : groups * wg_rows;
   return true;
}

bool
agx_apple9_mesh_ring_fits(const struct agx_apple9_mesh_layout *layout,
                          uint64_t groups)
{
   uint64_t max_groups;
   return agx_apple9_mesh_ring_max_groups(layout, &max_groups) &&
          groups <= max_groups;
}

bool
agx_apple9_mesh_arguments_build(uint64_t out[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2],
                                const struct agx_apple9_mesh_arguments *p)
{
   if (!out || !p)
      return false;
   uint64_t roots[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2] = {0};
   memcpy(roots, p->graphics, sizeof(p->graphics));
   roots[AGX_APPLE9_MESH_OUTPUT_ROOT] = p->output_root;
   roots[AGX_APPLE9_MESH_GRID_ROOT] = p->grid_root;
   roots[AGX_APPLE9_MESH_PAYLOAD_UNIFORM / 2] = p->payload;
   roots[AGX_APPLE9_MESH_DRAW_UNIFORM / 2] = p->draw;
   memcpy(out, roots, sizeof(roots));
   return true;
}
