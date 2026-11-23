/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/macros.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "glsl_types.h"
#include "nir_builder_opcodes.h"
#include "shader_enums.h"

/*
 * Lower cull distance to discard. From the spec:
 *
 *    If the cull distance for any enabled cull half-space is negative for all
 *    of the vertices of the primitive under consideration, the primitive is
 *    discarded.
 *
 * We don't have a direct way to read the cull distance at non-provoking
 * vertices in the fragment shader. Instead, we interpolate the quantity:
 *
 *    cull distance >= 0.0 ? 1.0 : 0.0
 *
 * Then, the discard condition is equivalent to:
 *
 *    "quantity is zero for all vertices of the primitive"
 *
 * which by linearity is equivalent to:
 *
 *    quantity is zero somewhere in the primitive and quantity has zero
 *    first-order screen space derivatives.
 *
 * which we can determine with ease in the fragment shader.
 */

static bool
lower_write(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_CLIP_DIST0 &&
       sem.location != VARYING_SLOT_CLIP_DIST1)
      return false;

   signed loc = sem.location + nir_src_as_uint(intr->src[1]);
   unsigned total_component =
      (loc - VARYING_SLOT_CLIP_DIST0) * 4 + nir_intrinsic_component(intr);

   unsigned base = b->shader->info.clip_distance_array_size;
   if (total_component < base)
      return false;

   unsigned component = total_component - base;
   if (component >= b->shader->info.cull_distance_array_size)
      return false;

   assert(nir_src_num_components(intr->src[0]) == 1 && "must be scalarized");

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offs = nir_imm_int(b, component / 4);
   nir_def *v = nir_b2f32(b, nir_fge_imm(b, intr->src[0].ssa, 0.0));

   nir_store_output(b, v, offs, .component = component % 4,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE,
                    .io_semantics.num_slots = 2);
   return true;
}

bool
agx_nir_lower_cull_distance_vs(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_VERTEX ||
          s->info.stage == MESA_SHADER_TESS_EVAL);

   nir_shader_intrinsics_pass(s, lower_write, nir_metadata_control_flow, NULL);

   s->info.outputs_written |=
      BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE,
                       DIV_ROUND_UP(s->info.cull_distance_array_size, 4));
   return true;
}
