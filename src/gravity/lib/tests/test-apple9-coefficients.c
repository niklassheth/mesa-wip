/* SPDX-License-Identifier: MIT
 * Compare object-owned interpolation state with draw-time packing using fixed
 * semantic allocation addresses. No shader compilation or GPU is involved. */
#include "agx_apple9.h"
#include "agx_device.h"
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

struct arena {
   uint8_t bytes[8192], coefficients[AGX_APPLE9_CF_BINDINGS_SIZE];
   uint64_t gpu, cf_gpu;
   unsigned used, cf_uploads;
};

static struct agx_ptr
allocate(void *data, size_t size, size_t alignment)
{
   struct arena *a = data;
   if (size == AGX_APPLE9_CF_BINDINGS_SIZE) {
      ++a->cf_uploads;
      return (struct agx_ptr){.cpu = a->coefficients, .gpu = a->cf_gpu};
   }
   unsigned off = ALIGN_POT(a->used, alignment);
   CHECK(off + size <= sizeof(a->bytes));
   a->used = off + size;
   return (struct agx_ptr){.cpu = a->bytes + off, .gpu = a->gpu + off};
}

int
main(void)
{
   struct agx_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->shader_base = 1ull << 40;
   simple_mtx_init(&dev->apple9_entry_lock, mtx_plain);
   list_inithead(&dev->apple9_entry_lru);
   struct agx_va va = {.addr = dev->shader_base + 0x10000};
   struct agx_bo bo = {.va = &va};
   agx_shader_body_init_bo(&bo);
   bo.apple9_body.entry_offset = 0x400;
   list_add(&bo.apple9_body.entry_link, &dev->apple9_entry_lru);
   uint8_t code[128] = {0};
   unsigned cases = 0;
   const unsigned components[] = {0, 1, 4, 65,
                                  AGX_APPLE9_MAX_VARYING_COMPONENTS};
   for (unsigned c = 0; c < ARRAY_SIZE(components); ++c) {
      for (unsigned mode = 0; mode < 4; ++mode) {
         for (unsigned extras = 0; extras < 16; ++extras) {
            struct agx_apple9_render_stage vs = {
               .bo = &bo,
               .binary = code,
               .binary_size = sizeof(code),
               .publication_count_valid = true,
               .publication_count = 32,
               .position_components = 4,
               .varying_components = components[c],
               .program_id = 1,
            };
            struct agx_apple9_render_stage fs = vs;
            fs.position_components = 0;
            fs.program_id = 2;
            fs.apple9_reads_z = extras & 1;
            fs.reads_point_coord = extras & 2;
            fs.reads_primitive_id = extras & 4;
            fs.barycentric_mask = (extras & 8) ? 3 : 0;
            fs.per_vertex_count = MIN2(components[c], 3);
            for (unsigned i = 0; i < fs.per_vertex_count; ++i)
               fs.per_vertex[i] = i;
            for (unsigned i = 0; i < components[c]; ++i) {
               if (mode == 1 || (mode == 3 && i % 3 == 0))
                  agx_apple9_interp_mask_set(&fs.apple9_flat_mask, i);
               if (mode == 2 || (mode == 3 && i % 3 == 1))
                  agx_apple9_interp_mask_set(&fs.apple9_linear_mask, i);
            }
            struct agx_apple9_render_stage owned = fs;
            uint8_t immutable[AGX_APPLE9_CF_ALLOCATION_SIZE];
            memset(immutable, 0xa5, sizeof(immutable));
            uint64_t cf_gpu = dev->shader_base + 0x30000;
            unsigned size = agx_apple9_coefficients_size(&owned);
            agx_apple9_init_coefficients(
               dev, &owned, (struct agx_ptr){.cpu = immutable, .gpu = cf_gpu});
            for (unsigned i = size; i < sizeof(immutable); ++i)
               CHECK(immutable[i] == 0xa5);
            for (unsigned pv = 0; pv < 3; ++pv) {
               unsigned offset =
                  owned.coefficients[pv] - (cf_gpu - dev->shader_base);
               struct arena usc[2] = {
                  {.gpu = dev->shader_base + 0x20000,
                   .cf_gpu = cf_gpu + offset},
                  {.gpu = dev->shader_base + 0x20000,
                   .cf_gpu = cf_gpu + offset},
               };
               struct arena context[2] = {
                  {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000},
                  {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000},
               };
               struct agx_apple9_uniform_draw draw[2] = {0};
               uint8_t commands[2][256] = {0};
               for (unsigned arm = 0; arm < 2; ++arm) {
                  struct agx_apple9_render_pipeline pipeline;
                  CHECK(agx_apple9_link_render_pipeline(&pipeline, &vs,
                                                        arm ? &owned : &fs));
                  pipeline.primitive = pv == 1 ? AGX_PRIMITIVE_TRIANGLE_FAN
                                               : AGX_PRIMITIVE_TRIANGLES;
                  pipeline.flatshade_first = pv != 2;
                  draw[arm].flatshade_first = pv != 2;
                  draw[arm].object_type = AGX_OBJECT_TYPE_TRIANGLE;
                  draw[arm].scissor_max[0] = draw[arm].scissor_max[1] = 32;
                  draw[arm].vertex_table = 0x40000;
                  draw[arm].fragment_table = 0x50000;
                  CHECK(agx_apple9_prepare_draw_alloc(
                     dev, &usc[arm], allocate, &context[arm], allocate,
                     &pipeline, &draw[arm], NULL));
                  CHECK(draw[arm].provoking_vertex == pv);
                  pipeline.ppp = draw[arm].ppp;
                  pipeline.vertex_launch = draw[arm].launch[0] / 64;
                  pipeline.pipeline_word = 0x01000000;
                  uint8_t *end = agx_apple9_emit_direct_draw(
                     commands[arm], &pipeline, 3, 1, 0);
                  CHECK(end - commands[arm] ==
                        agx_apple9_direct_draw_size(&pipeline));
               }
               CHECK(usc[0].cf_uploads == 1 && usc[1].cf_uploads == 0);
               CHECK(!memcmp(usc[0].coefficients, immutable + offset,
                             AGX_APPLE9_CF_BINDINGS_SIZE));
               CHECK(draw[0].cf_count == draw[1].cf_count);
               CHECK(!memcmp(draw[0].ppp_record, draw[1].ppp_record,
                             sizeof(draw[0].ppp_record)));
               CHECK(usc[0].used == usc[1].used &&
                     context[0].used == context[1].used);
               CHECK(!memcmp(usc[0].bytes, usc[1].bytes, usc[0].used));
               CHECK(
                  !memcmp(context[0].bytes, context[1].bytes, context[0].used));
               CHECK(!memcmp(commands[0], commands[1], sizeof(commands[0])));
               ++cases;
            }
         }
      }
   }
   /* Crossbar linking: VS writes a superset in (smooth, flat, linear) group
    * order; FS reads a reordered subset plus an unwritten varying and reads_z. */
   {
      struct agx_apple9_render_stage vs = {
         .bo = &bo,
         .binary = code,
         .binary_size = sizeof(code),
         .publication_count_valid = true,
         .publication_count = 32,
         .position_components = 4,
         .varying_components = 9,
         .program_id = 10,
      };
      /* VS UVS indices:
       * group 0 (smooth): VAR0.xyzw -> 0..3, VAR7.x -> 4
       * group 1 (flat):   VAR2.xy   -> 5..6
       * group 2 (linear): VAR5.zw   -> 7..8 */
      vs.varyings.mask[VARYING_SLOT_VAR0] = 0xf;
      vs.varyings.group[VARYING_SLOT_VAR0] = 0;
      vs.varyings.mask[VARYING_SLOT_VAR2] = 0x3;
      vs.varyings.group[VARYING_SLOT_VAR2] = 1;
      vs.varyings.mask[VARYING_SLOT_VAR5] = 0xc;
      vs.varyings.group[VARYING_SLOT_VAR5] = 2;
      vs.varyings.mask[VARYING_SLOT_VAR7] = 0x1;
      vs.varyings.group[VARYING_SLOT_VAR7] = 0;
      vs.varyings.count = 9;

      struct agx_apple9_render_stage fs = {
         .bo = &bo,
         .binary = code,
         .binary_size = sizeof(code),
         .publication_count_valid = true,
         .publication_count = 32,
         .position_components = 0,
         .varying_components = 6,
         .program_id = 11,
         .apple9_reads_z = true,
      };
      /* Independent FS (all group 0, semantic order):
       * cf1..2: VAR0.yz (smooth) -> VS idx 1..2 (src 3..4 with reads_z)
       * cf3:    VAR2.x  (flat)   -> VS idx 5    (src 7 with reads_z)
       * cf4..5: VAR5.zw (linear) -> VS idx 7..8 (src 9..10 with reads_z)
       * cf6:    VAR9.x  (unwritten) -> src 0 (1/W fallback) */
      fs.varyings.mask[VARYING_SLOT_VAR0] = 0x6;
      fs.varyings.mask[VARYING_SLOT_VAR2] = 0x1;
      fs.varyings.mask[VARYING_SLOT_VAR5] = 0xc;
      fs.varyings.mask[VARYING_SLOT_VAR9] = 0x1;
      fs.varyings.count = 6;
      agx_apple9_interp_mask_set(&fs.apple9_flat_mask, 2);
      agx_apple9_interp_mask_set(&fs.apple9_linear_mask, 3);
      agx_apple9_interp_mask_set(&fs.apple9_linear_mask, 4);

      struct agx_apple9_render_pipeline pipeline;
      CHECK(agx_apple9_link_render_pipeline(&pipeline, &vs, &fs));
      pipeline.primitive = AGX_PRIMITIVE_TRIANGLES;
      pipeline.flatshade_first = true;

      struct arena usc = {.gpu = dev->shader_base + 0x20000,
                          .cf_gpu = dev->shader_base + 0x30000};
      struct arena context = {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000};
      struct agx_apple9_uniform_draw draw = {
         .flatshade_first = true,
         .object_type = AGX_OBJECT_TYPE_TRIANGLE,
         .scissor_max = {32, 32},
         .vertex_table = 0x40000,
         .fragment_table = 0x50000,
      };
      CHECK(agx_apple9_prepare_draw_alloc(dev, &usc, allocate, &context,
                                          allocate, &pipeline, &draw, NULL));
      CHECK(usc.cf_uploads == 1);
      /* Expected bindings:
       * [0]: 1/W (0x0c)
       * [1]: count=2, smooth(7), src=3, dst=1 -> 1 | (7<<2) | (3<<8) | (1<<16)
       * [2]: count=1, flat0(0),  src=7, dst=3 -> 0 | (0<<2) | (7<<8) | (3<<16)
       * [3]: count=2, linear(3), src=9, dst=4 -> 1 | (3<<2) | (9<<8) | (4<<16)
       * [4]: unwritten VAR9.x    src=0, dst=6 -> 0x0c | (6<<16)
       * [5]: reads_z                    dst=7 -> 0x12c | (7<<16) */
      CHECK(draw.cf_count == 6);
      const uint32_t *w = (const uint32_t *)usc.coefficients;
      /* number_of_32_bit_slots = 1 (1/W) + 1 (reads_z) + 9 (vs varyings) = 11
       * number_of_coefficient_registers = 1 (1/W) + 6 (fs varyings) + 1 (reads_z) = 8 */
      CHECK(w[0] == (11u | (8u << 8)));
      CHECK(w[1] == 0x0cu);
      CHECK(w[2] == (1u | (7u << 2) | (3u << 8) | (1u << 16)));
      CHECK(w[3] == (0u | (0u << 2) | (7u << 8) | (3u << 16)));
      CHECK(w[4] == (1u | (3u << 2) | (9u << 8) | (4u << 16)));
      CHECK(w[5] == (0x0cu | (6u << 16)));
      CHECK(w[6] == (0x12cu | (7u << 16)));
   }
   simple_mtx_destroy(&dev->apple9_entry_lock);
   free(dev);
   printf(
      "PASS %u interpolation/primitive cases: coefficient payload, roots, PPP and VDM bytes identical at fixed addresses\n",
      cases);
}
