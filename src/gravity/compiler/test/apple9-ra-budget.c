/* SPDX-License-Identifier: MIT
 * Compare a bounded allocation with the same completed allocation. The loop
 * keeps independently evolving phi values live, forcing real spill/repair.
 */
#include "../agx_apple9_ra.c"
#include "compiler/nir/nir_builder.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
check_pressure(unsigned count, bool banks, bool latency, bool masks)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "spill budget");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = nir.shader->info.workgroup_size[2] = 1;
   nir_block logical[4] = {0};
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[2];
   logical[1].successors[1] = &logical[3];
   logical[2].successors[0] = &logical[1];
   struct agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   struct agx_apple9_block *blocks[4];
   for (unsigned i = 0; i < 4; ++i) {
      blocks[i] = agx_apple9_block_create(&p);
      blocks[i]->nir = &logical[i];
   }
   agx_apple9_block_begin(&p, blocks[0]);
   uint32_t phi[128];
   for (unsigned i = 0; i < count; ++i) {
      uint32_t initial =
         agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
                             NULL, 0, (i * 0x9e3779b9u) ^ 0xa511e9b3u);
      phi[i] = agx_apple9_vir_emit_phi(&p, blocks[1]);
      CHECK(agx_apple9_vir_emit_phi_source(&p, phi[i], initial));
   }
   if (masks) {
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      CHECK(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_LOOP_MASK_PUSH, AGX_APPLE9_ENC_LOOP_MASK_PUSH,
         NULL, 0, 0));
   }
   agx_apple9_block_begin(&p, blocks[1]);
   agx_apple9_block_begin(&p, blocks[2]);
   for (unsigned i = 0; i < count; ++i) {
      uint32_t src[] = {phi[i], phi[(i + 1) % count]};
      uint32_t next = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      CHECK(agx_apple9_vir_emit_phi_source(&p, phi[i], next));
   }
   if (masks) {
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      CHECK(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_LOOP_MASK_POP, AGX_APPLE9_ENC_LOOP_MASK_POP,
         NULL, 0, 0));
   }
   agx_apple9_block_begin(&p, blocks[3]);
   uint32_t hash = phi[0];
   for (unsigned i = 1; i < count; ++i) {
      uint32_t src[] = {hash, phi[i]};
      hash = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                 AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
   }
   uint32_t index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                        AGX_APPLE9_ENC_MOV_IMM32, NULL, 0, 0);
   CHECK(agx_apple9_vir_emit_device_store(&p, 0, index, &hash, 1, 32));
   agx_apple9_place_phis(&p);
   const char *reason = NULL;
   CHECK(agx_apple9_assign_vir_scoreboard_slots(&p, &reason));
   agx_apple9_vir_reindex(&p);
   CHECK(agx_apple9_allocate_publications(&p, &reason));
   struct agx_apple9_vir_program full, second;
   agx_apple9_vir_init(&full);
   agx_apple9_vir_init(&second);
   bool copy_reg = true;
   CHECK(apple9_run_native_candidate(&p, banks, latency, 0, true, &copy_reg,
                                     &full, &reason, NULL));
   uint64_t cost = apple9_allocation_cost(&full).spills;
   CHECK(count <= 96 || cost > 0);
   CHECK(agx_apple9_validate_vir_allocation(&full, &reason));
   copy_reg = true;
   CHECK(apple9_run_native_candidate(&p, banks, latency, 0, true, &copy_reg,
                                     &second, &reason, NULL));
   CHECK(full.instruction_count == second.instruction_count);
   for (unsigned i = 0; i < full.instruction_count; ++i) {
      struct agx_apple9_packed_instruction a = {0}, b = {0};
      CHECK(agx_apple9_pack_vir_instruction(full.instructions[i], full.phys, &a,
                                            &reason));
      CHECK(agx_apple9_pack_vir_instruction(second.instructions[i], second.phys,
                                            &b, &reason));
      CHECK(a.length == b.length && !memcmp(a.bytes, b.bytes, a.length));
   }
   agx_apple9_vir_finish(&full);
   agx_apple9_vir_finish(&second);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

static void
check_indexed_spill_hazard_and_cost(void)
{
   /* Verify words 1023, 1024, 2047, and 2048 with a killed compact SAVE source
    * (r8) followed by an indexed PRIVATE_LOAD (r1) wait on the store's slot
    * and count in apple9_allocation_cost().spills. */
   const unsigned words[] = {1023u, 1024u, 2047u, 2048u};
   for (unsigned wi = 0; wi < ARRAY_SIZE(words); ++wi) {
      unsigned word = words[wi];
      struct agx_apple9_vir_program p;
      CHECK(apple9_init_physical(&p));
      struct agx_apple9_block *blk = agx_apple9_block_create(&p);
      CHECK(blk);
      agx_apple9_block_begin(&p, blk);

      struct agx_apple9_vir_instr store;
      if (word >= AGX_APPLE9_SAVE_WORD_COUNT) {
         store = apple9_physical_instruction(AGX_APPLE9_VIR_PRIVATE_STORE,
                                             AGX_APPLE9_ENC_PRIVATE_STORE,
                                             AGX_APPLE9_VREG_INVALID);
         store.src[0] = 1;
         store.src[1] = 2;
         store.nr_srcs = 2;
         store.live_after_mask = 2;
         store.immediate = word;
         store.spill_access = true;
      } else {
         store = apple9_physical_instruction(AGX_APPLE9_VIR_SPILL_STORE,
                                             AGX_APPLE9_ENC_SPILL_STORE,
                                             AGX_APPLE9_VREG_INVALID);
         store.src[0] = 8; /* killed ordinary GPR, disjoint from r1 */
         store.nr_srcs = 1;
         store.immediate = word;
         store.producer_scoreboard_slot = 6;
         store.spill_access = true;
      }
      CHECK(apple9_emit_physical(&p, &store));

      struct agx_apple9_vir_instr load;
      if (word >= AGX_APPLE9_FILL_WORD_COUNT) {
         load = apple9_physical_instruction(AGX_APPLE9_VIR_PRIVATE_LOAD,
                                            AGX_APPLE9_ENC_PRIVATE_LOAD, 1);
         load.src[0] = 1;
         load.nr_srcs = 1;
         load.immediate = word;
         load.producer_scoreboard_slot = 6;
         load.spill_access = true;
      } else {
         load = apple9_physical_instruction(AGX_APPLE9_VIR_SPILL_LOAD,
                                            AGX_APPLE9_ENC_SPILL_LOAD, 9);
         load.immediate = word;
         load.producer_scoreboard_slot = 6;
         load.spill_access = true;
      }
      CHECK(apple9_emit_physical(&p, &load));
      p.physical = true;

      const char *reason = NULL;
      CHECK(agx_apple9_schedule_physical(&p, &reason));

      /* For compact SAVE (word < 2048) followed by reload (compact or indexed)
       * of the same word, the scheduler MUST emit or fold a wait on store_slot
       * before the load! */
      if (word < AGX_APPLE9_SAVE_WORD_COUNT) {
         unsigned store_slot = 0;
         bool waited = false, seen_load = false;
         for (unsigned i = 0; i < p.instruction_count; ++i) {
            const struct agx_apple9_vir_instr *I = p.instructions[i];
            if (I->op == AGX_APPLE9_VIR_SPILL_STORE && !store_slot)
               store_slot = I->producer_scoreboard_slot;
            if (store_slot && !seen_load) {
               if (I->op == AGX_APPLE9_VIR_WAIT &&
                   (I->immediate & BITFIELD_BIT(store_slot - 1)))
                  waited = true;
               if (I->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                   I->op == AGX_APPLE9_VIR_PRIVATE_LOAD) {
                  if (I->scoreboard_slot == store_slot)
                     waited = true;
                  seen_load = true;
               }
            }
         }
         CHECK(store_slot != 0 && seen_load && waited);
      }

      /* Both compact and indexed spill ops must contribute to spill cost. */
      struct agx_apple9_allocation_cost cost = apple9_allocation_cost(&p);
      CHECK(cost.spills >= 2u);
      agx_apple9_vir_finish(&p);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_indexed_spill_hazard_and_cost();
   const unsigned pressure[] = {8, 96, 128};
   for (unsigned i = 0; i < ARRAY_SIZE(pressure); ++i)
      for (unsigned banks = 0; banks < 2; ++banks)
         for (unsigned latency = 0; latency < 2; ++latency)
            for (unsigned masks = 0; masks < 2; ++masks)
               check_pressure(pressure[i], banks, latency, masks);
   glsl_type_singleton_decref();
   puts("native Apple9 RA preserves deterministic allocations across pressure, bank-cost, latency, and loop-mask policies");
   return 0;
}
