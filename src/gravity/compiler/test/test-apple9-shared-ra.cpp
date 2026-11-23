/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>
#include "agx_apple9_ir.h"
#include "agx_compile.h"
#include "agx_compile_apple9.h"
#include "apple9_ra_assign.h"

TEST(Apple9SharedRa, ShiftRetainsLiveInputWithoutCopies)
{
   for (bool retained : {false, true}) {
      SCOPED_TRACE(retained);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "shift_clobber");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      constexpr uint32_t input = 0xa511e9b3;
      auto base = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, input);
      auto shifted = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_ISHR,
         AGX_APPLE9_ENC_SHIFT_EXTENDED, &base, 1, 5);
      auto result = shifted;
      if (retained) {
         uint32_t src[] = {base, shifted};
         result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
            AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      }
      auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &result, 1, 32));
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
      EXPECT_EQ(p.scratch_size, 0u);
      /* Without spills no register is reserved. */
      for (unsigned r = 0; r < 5; ++r)
         EXPECT_FALSE(p.reserved_gprs[r]);
      uint32_t regs[96] = {}, output = 0;
      unsigned copies = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         auto I = p.instructions[i];
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[I->dest] = regs[I->src[0]] | regs[I->src[1]];
            ++copies;
            break;
         case AGX_APPLE9_VIR_ISHR: {
            auto shifted_value = uint32_t(int32_t(regs[I->src[0]]) >> I->immediate);
            if (!(I->live_after_mask & 1))
               regs[I->src[0]] = 0xdeadbeef;
            regs[I->dest] = shifted_value;
            break;
         }
         case AGX_APPLE9_VIR_IADD:
            regs[I->dest] = regs[I->src[0]] + regs[I->src[1]];
            break;
         case AGX_APPLE9_VIR_DEVICE_STORE:
            output = regs[I->src[0]];
            break;
         default:
            FAIL() << "unexpected opcode " << I->op;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << reason;
      }
      EXPECT_EQ(output, uint32_t(int32_t(input) >> 5) + (retained ? input : 0));
      EXPECT_EQ(copies, 0u);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, LoadWaitFoldsIntoConsumerAfterIndependentArithmetic)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "late_load_wait");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto loaded = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   auto constant = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x12345678);
   uint32_t add_src[] = {index, constant};
   auto independent = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, add_src, 2, 0);
   uint32_t xor_src[] = {independent, loaded};
   auto result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, xor_src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &result, 1, 32));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   unsigned load_position = ~0, arithmetic_position = ~0, wait_position = ~0;
   unsigned slot = 0, landing = ~0, copies = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto I = p.instructions[i];
      copies += I->op == AGX_APPLE9_VIR_IOR;
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         load_position = i;
         slot = I->producer_scoreboard_slot;
         landing = I->dest;
      } else if (I->op == AGX_APPLE9_VIR_IADD) {
         arithmetic_position = i;
         EXPECT_EQ(I->scoreboard_slot, 0u);
      } else if (I->op == AGX_APPLE9_VIR_IXOR) {
         wait_position = i;
         EXPECT_EQ(I->scoreboard_slot, slot);
         EXPECT_EQ(I->src[0], landing); /* Pending logic operand normalized to A. */
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason)) << reason;
   }
   EXPECT_LT(load_position, arithmetic_position);
   EXPECT_LT(arithmetic_position, wait_position);
   EXPECT_EQ(copies, 0u);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

template <unsigned count, bool force_spill = false>
static void
check_loop_register_pressure()
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "physical_spill_loop");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = nir.shader->info.workgroup_size[2] = 1;
   nir_block logical[4] = {};
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[2];
   logical[1].successors[1] = &logical[3];
   logical[2].successors[0] = &logical[1];
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   agx_apple9_block *blocks[4];
   for (unsigned i = 0; i < 4; ++i) {
      blocks[i] = agx_apple9_block_create(&p);
      blocks[i]->nir = &logical[i];
   }
   agx_apple9_block_begin(&p, blocks[0]);
   uint32_t initial[count], phi[count], expected[count];
   for (unsigned i = 0; i < count; ++i) {
      expected[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      initial[i] =
         agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
                             nullptr, 0, expected[i]);
      phi[i] = agx_apple9_vir_emit_phi(&p, blocks[1]);
      ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi[i], initial[i]));
   }
   agx_apple9_block_begin(&p, blocks[1]);
   agx_apple9_block_begin(&p, blocks[2]);
   for (unsigned i = 0; i < count; ++i) {
      uint32_t src[] = {phi[i], phi[(i + 1) % count]};
      uint32_t next = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi[i], next));
   }
   agx_apple9_block_begin(&p, blocks[3]);
   uint32_t hash = phi[0];
   uint32_t multiplier = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 33);
   for (unsigned i = 1; i < count; ++i) {
      uint32_t src[] = {hash, multiplier, phi[i]};
      hash = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMAD,
                                 AGX_APPLE9_ENC_INT_MAD_EXTENDED, src, 3, 0);
   }
   uint32_t index = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &hash, 1, 32));
   agx_apple9_place_phis(&p);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << reason;
   if (force_spill) {
      apple9_ra_assign_options opts = apple9_ra_assign_default_options();
      opts.force_spill = true;
      opts.validate = true;
      apple9_ra_assign_result res = {};
      ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
         << (res.error ?: "");
   } else {
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   }
   EXPECT_TRUE(p.physical);
   const bool expect_spill = force_spill || count > 96;
   if (expect_spill)
      ASSERT_GT(p.scratch_size, 0u);
   else
      ASSERT_EQ(p.scratch_size, 0u);
   EXPECT_LE(p.scratch_size, 4096u);
   uint32_t regs[96] = {}, memory[1024] = {}, result = 0;
   struct pending_write {
      bool memory;
      unsigned index, value, slot;
      unsigned source;
   };
   std::vector<pending_write> pending;
   unsigned saves = 0, fills = 0;
   auto execute = [&](agx_apple9_block *block) {
      unsigned end =
         block->next ? block->next->start_index : p.instruction_count;
      for (unsigned i = block->start_index; i < end; ++i) {
         auto I = p.instructions[i];
         ASSERT_EQ(agx_apple9_instr_block(I), block);
         for (auto it = pending.begin(); it != pending.end();) {
            if (it->slot == I->scoreboard_slot ||
                (I->op == AGX_APPLE9_VIR_WAIT &&
                 (I->immediate & BITFIELD_BIT(it->slot - 1)))) {
               if (it->memory) {
                  EXPECT_EQ(regs[it->source], it->value);
                  regs[it->source] = 0xdeadbeef;
               }
               (it->memory ? memory : regs)[it->index] = it->value;
               it = pending.erase(it);
            } else
               ++it;
         }
         if (I->producer_scoreboard_slot) {
            for (const auto &write : pending)
               ASSERT_NE(write.slot, I->producer_scoreboard_slot);
         }
         for (const auto &write : pending) {
            if (!write.memory)
               continue;
            for (unsigned c = 0; c < I->dest_components; ++c)
               EXPECT_NE(I->dest + c, write.source);
            if (I->op == AGX_APPLE9_VIR_SPILL_STORE) {
               EXPECT_NE(I->src[0], write.source);
            }
         }
         uint32_t a = I->nr_srcs ? regs[I->src[0]] : 0;
         uint32_t b = I->nr_srcs > 1 ? regs[I->src[1]] : 0;
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_IADD:
            regs[I->dest] = a + b;
            break;
         case AGX_APPLE9_VIR_IMAD:
            regs[I->dest] = a * b + regs[I->src[2]];
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[I->dest] = a | b;
            break;
         case AGX_APPLE9_VIR_IXOR:
            regs[I->dest] = a ^ b;
            break;
         case AGX_APPLE9_VIR_SPILL_LOAD:
            ASSERT_LT(I->immediate, 1024u);
            pending.push_back({false, I->dest, memory[I->immediate],
                               I->producer_scoreboard_slot, 0});
            ++fills;
            break;
         case AGX_APPLE9_VIR_SPILL_STORE:
            ASSERT_LT(I->immediate, 1024u);
            pending.push_back(
               {true, I->immediate, a, I->producer_scoreboard_slot, I->src[0]});
            ++saves;
            break;
         case AGX_APPLE9_VIR_DEVICE_STORE:
            result = a;
            break;
         case AGX_APPLE9_VIR_WAIT:
            break;
         default:
            FAIL() << "Unexpected physical opcode " << I->op;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << "opcode " << I->op << " " << (reason ?: "");
      }
   };
   auto entry = p.blocks, header = entry->next, body = header->next,
        exit = body->next;
   execute(entry);
   for (unsigned iteration = 0; iteration < 3; ++iteration) {
      execute(header);
      execute(body);
      uint32_t first = expected[0];
      for (unsigned i = 0; i + 1 < count; ++i)
         expected[i] += expected[i + 1];
      expected[count - 1] += first;
   }
   execute(header);
   execute(exit);
   uint32_t expected_hash = expected[0];
   for (unsigned i = 1; i < count; ++i)
      expected_hash = expected_hash * 33 + expected[i];
   EXPECT_EQ(result, expected_hash);
   EXPECT_EQ(saves > 0, expect_spill);
   EXPECT_EQ(fills > 0, expect_spill);
   EXPECT_TRUE(pending.empty());
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, PhysicalSpillLoweringPreservesLiveSourcesAndLoopPhis)
{
   check_loop_register_pressure<80>();
   check_loop_register_pressure<112>();
   check_loop_register_pressure<32, true>();
}

TEST(Apple9SharedRa, TexturePublicationPressurePreservesEarlyHandoffs)
{
   for (unsigned count : {2u, 3u, 6u, 8u, 9u, 12u}) {
      SCOPED_TRACE(count);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "publication_pressure");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      uint32_t coords[] = {zero, zero};
      std::vector<uint32_t> samples;
      for (unsigned t = 0; t < count; ++t)
         samples.push_back(agx_apple9_vir_emit_texture_lod(
            &p, coords, zero, t, 0, true));
      for (auto sample : samples) {
         uint32_t rgba[] = {sample, sample + 1, sample + 2, sample + 3};
         ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, zero, rgba, 4, 32));
      }
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
      unsigned handoffs = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i)
         handoffs += p.instructions[i]->publication_handoff;
      EXPECT_EQ(handoffs, count > 8 ? count - 8 : 0);
      EXPECT_LE(p.publication_count, 32u);
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
      unsigned retained = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i)
         retained += p.instructions[i]->publication_handoff;
      EXPECT_EQ(retained, handoffs);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, GradientPublicationFitsItsWholeEncodingWindow)
{
   for (unsigned dimension : {0u, 2u, 3u}) {
      for (unsigned pending : {0u, 4u, 6u, 7u}) {
         SCOPED_TRACE(dimension);
         SCOPED_TRACE(pending);
         nir_builder nir = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "gradient_publication_window");
         nir_block logical = {};
         agx_apple9_vir_program p;
         agx_apple9_vir_init(&p);
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical;
         agx_apple9_block_begin(&p, block);
         auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
         uint32_t coords[] = {zero, zero};
         std::vector<uint32_t> samples;
         for (unsigned i = 0; i < pending; ++i)
            samples.push_back(agx_apple9_vir_emit_texture_lod(
               &p, coords, zero, i, 0, false));
         uint32_t parameters[9];
         for (auto &value : parameters) value = zero;
         samples.push_back(agx_apple9_vir_emit_texture_grad(
            &p, parameters, pending, 0, dimension, false, false));
         for (auto sample : samples) {
            uint32_t rgba[] = {sample, sample + 1, sample + 2, sample + 3};
            ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, zero, rgba, 4, 32));
         }
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
         EXPECT_LE(p.publication_count, 32u);
         ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
         for (unsigned i = 0; i < p.instruction_count; ++i) {
            auto I = p.instructions[i];
            if (I->op != AGX_APPLE9_VIR_PUBLICATION_TUPLE) continue;
            EXPECT_LE(p.phys[I->dest] + I->dest_components, 32u);
            agx_apple9_packed_instruction packed;
            EXPECT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
               << (reason ?: "");
         }
         agx_apple9_vir_finish(&p);
         ralloc_free(nir.shader);
      }
   }
}

TEST(Apple9SharedRa, PublicationHandoffStaysInProducerBlock)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto entry = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, entry);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t coords[] = {zero, zero};
   auto first = agx_apple9_vir_emit_texture_lod(&p, coords, zero, 0, 0, true);
   auto successor = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, successor);
   for (unsigned t = 1; t <= 8; ++t)
      agx_apple9_vir_emit_texture_lod(&p, coords, zero, t, 0, true);
   p.output = first;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
   unsigned copies = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto ins = p.instructions[i];
      if (ins->publication_handoff) {
         EXPECT_EQ(agx_apple9_instr_block(ins), entry);
         EXPECT_EQ(ins->src[0], first + copies);
         ++copies;
      }
   }
   EXPECT_EQ(copies, 1u);
   EXPECT_NE(p.output, first);
   agx_apple9_vir_finish(&p);
}


TEST(Apple9SharedRa, SubgroupMaxWaitsForInputAndPreservesSource)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "subgroup_max_load");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto loaded = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   auto maximum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_SUBGROUP_REDUCE_UMAX,
      AGX_APPLE9_ENC_SUBGROUP_REDUCE_UMAX, &loaded, 1, 0);
   EXPECT_FALSE(agx_apple9_instr_is_pure_alu(p.instructions[p.instruction_count - 1]));
   uint32_t src[] = {maximum, loaded};
   auto result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &result, 1, 32));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   unsigned load_at = ~0u, wait_at = ~0u, max_at = ~0u;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) load_at = i;
      if (I->op == AGX_APPLE9_VIR_WAIT) wait_at = i;
      if (I->op == AGX_APPLE9_VIR_SUBGROUP_REDUCE_UMAX) {
         max_at = i;
         EXPECT_NE(p.phys[I->dest], p.phys[I->src[0]]);
         EXPECT_EQ(I->scoreboard_slot, 0u);
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason)) << reason;
   }
   EXPECT_LT(load_at, wait_at);
   EXPECT_LT(wait_at, max_at);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, MultiSlotSharedPrivateAndSharedAtomicOverlapWithoutWaits)
{
   for (unsigned kind = 0; kind < 3; ++kind) {
      SCOPED_TRACE(kind);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "multislot_mem");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);

      uint32_t idx[6], vals[6];
      uint32_t one = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 1);
      for (unsigned i = 0; i < 6; ++i)
         idx[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, i);

      for (unsigned i = 0; i < 6; ++i) {
         if (kind == 0) {
            vals[i] = agx_apple9_vir_emit_shared_load(&p, idx[i], 0, 1);
         } else if (kind == 1) {
            vals[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_PRIVATE_LOAD,
               AGX_APPLE9_ENC_PRIVATE_LOAD, &idx[i], 1, i);
            p.private_size = 256;
            p.instructions[p.instruction_count - 1]->producer_scoreboard_slot =
               AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         } else {
            uint32_t srcs[] = {one, idx[i]};
            vals[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_SHARED_ATOMIC,
               AGX_APPLE9_ENC_SHARED_ATOMIC, srcs, 2, 0);
            auto *ins = p.instructions[p.instruction_count - 1];
            ins->atomic_op = AGX_APPLE9_ATOMIC_ADD;
            ins->memory_components = 1;
            ins->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         }
         ASSERT_NE(vals[i], AGX_APPLE9_VREG_INVALID);
      }

      /* Consume all 6 results after issuing all 6 ops. */
      uint32_t acc = vals[0];
      for (unsigned i = 1; i < 6; ++i) {
         uint32_t srcs[] = {acc, vals[i]};
         acc = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
            AGX_APPLE9_ENC_INT_ADD_EXTENDED, srcs, 2, 0);
      }
      ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, idx[0], &acc, 1, 32));

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

      unsigned slot_mask = 0, ops_seen = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if ((kind == 0 && I->op == AGX_APPLE9_VIR_SHARED_LOAD) ||
             (kind == 1 && I->op == AGX_APPLE9_VIR_PRIVATE_LOAD) ||
             (kind == 2 && I->op == AGX_APPLE9_VIR_SHARED_ATOMIC)) {
            ASSERT_GE(I->producer_scoreboard_slot, 1u);
            ASSERT_LE(I->producer_scoreboard_slot, 6u);
            slot_mask |= BITFIELD_BIT(I->producer_scoreboard_slot - 1);
            ++ops_seen;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << (reason ?: "");
      }
      EXPECT_EQ(ops_seen, 6u);
      /* SHARED_LOAD and PRIVATE_LOAD have no memory hazards with each other and
       * spread across multiple scoreboard slots; SHARED_ATOMIC writes shared
       * memory so it exercises multi-slot packing cleanly too. */
      if (kind < 2) {
         EXPECT_GT(util_bitcount(slot_mask), 1u);
      }
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, AsyncImageStoresOverlapSixTuplesZeroJoinsAndDrainAtFence)
{
   for (unsigned count : {6u, 7u}) {
      SCOPED_TRACE(count);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "async_imgwrite_6x");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);

      uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      uint32_t one = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f800000);
      uint32_t rgba[] = {one, one, one, one};
      for (unsigned i = 0; i < count; ++i) {
         uint32_t x = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, i);
         uint32_t xy[] = {x, zero};
         ASSERT_TRUE(agx_apple9_vir_emit_image_store(&p, xy, rgba, zero,
                                                     AGX_APPLE9_VREG_INVALID, false));
      }
      /* Fence drains all pending image writes before texture invalidate. */
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_DEVICE_FENCE,
         AGX_APPLE9_ENC_DEVICE_FENCE, nullptr, 0, AGX_APPLE9_FENCE_IMAGE_WRITE));
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_DEVICE_FENCE,
         AGX_APPLE9_ENC_DEVICE_FENCE, nullptr, 0, AGX_APPLE9_FENCE_TEX_INVALIDATE));

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

      unsigned stores = 0, waits_before_fence = 0, waits_at_fence = 0;
      unsigned tuple_mask = 0, slot_mask = 0;
      uint8_t fence_mask = 0;
      bool saw_fence = false;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if (I->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
             I->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE) {
            saw_fence = true;
            fence_mask = I->fence_slot_mask;
         }
         if (I->op == AGX_APPLE9_VIR_WAIT) {
            if (!saw_fence && stores < count)
               ++waits_before_fence;
            else if (!saw_fence && stores == count)
               ++waits_at_fence;
         }
         if (I->op == AGX_APPLE9_VIR_IMAGE_STORE) {
            unsigned base = p.phys[I->src[0]];
            ASSERT_EQ(base % 8, 0u);
            ASSERT_LE(base, 40u);
            tuple_mask |= BITFIELD_BIT(base / 8);
            ASSERT_GE(I->producer_scoreboard_slot, 1u);
            ASSERT_LE(I->producer_scoreboard_slot, 6u);
            slot_mask |= BITFIELD_BIT(I->producer_scoreboard_slot - 1);
            ++stores;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << (reason ?: "");
         if (I->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
             I->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE) {
            const uint8_t exp_all6[] = {0x07, 0xf4, 0x57, 0x50, 0x06, 0x00};
            ASSERT_EQ(packed.length, 6u);
            EXPECT_EQ(memcmp(packed.bytes, exp_all6, sizeof(exp_all6)), 0);
         }
      }
      EXPECT_EQ(stores, count);
      EXPECT_EQ(tuple_mask, BITFIELD_MASK(6)); /* All 6 tuples p0..p40 used. */
      EXPECT_EQ(slot_mask, BITFIELD_MASK(6));  /* All 6 slots 1..6 used. */
      EXPECT_EQ(fence_mask, BITFIELD_MASK(6)); /* FENCE_IMAGE_WRITE drains all 6 slots. */
      /* 6 back-to-back stores issue with ZERO waits between them; the 7th
       * waits once when reusing p0..p7. 0x0650 drains all pending stores via
       * its 6-bit mask with ZERO standalone WAIT at the fence. */
      EXPECT_EQ(waits_before_fence, count == 6 ? 0u : 1u);
      EXPECT_EQ(waits_at_fence, 0u);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, VectorSharedLoadStoreAllocatesAndPacksUint2Uint4)
{
   for (unsigned comps : {2u, 4u}) {
      SCOPED_TRACE(comps);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "vec_shared_ra");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);

      uint32_t i0 = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      uint32_t i1 = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 1);
      uint32_t loaded = agx_apple9_vir_emit_shared_load(&p, i0, 2, comps);
      ASSERT_NE(loaded, AGX_APPLE9_VREG_INVALID);

      uint32_t out[4];
      for (unsigned c = 0; c < comps; ++c) {
         uint32_t k = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, c + 1);
         uint32_t add[] = {loaded + c, k};
         out[c] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
            AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
      }
      ASSERT_TRUE(agx_apple9_vir_emit_shared_store(&p, out, comps, i1, 2));

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

      unsigned loads = 0, stores = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if (I->op == AGX_APPLE9_VIR_SHARED_LOAD) {
            EXPECT_EQ(I->dest_components, comps);
            EXPECT_LE(p.phys[I->dest] + comps, 64u);
            ++loads;
         } else if (I->op == AGX_APPLE9_VIR_SHARED_STORE) {
            EXPECT_EQ(I->memory_components, comps);
            EXPECT_LE(p.phys[I->src[0]] + comps, 64u);
            for (unsigned c = 1; c < comps; ++c)
               EXPECT_EQ(p.phys[I->src[c]], p.phys[I->src[0]] + c);
            ++stores;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << (reason ?: "");
      }
      EXPECT_EQ(loads, 1u);
      EXPECT_EQ(stores, 1u);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, AllocationCheaperBucketsRegistersByOccupancyTier)
{
   /* 50 and 51 GPRs (100 and 102 half-regs) sit in the same top occupancy tier
    * (<= 104 half-regs, 1024 threads); 53 GPRs (106 half-regs) drops to 896. */
   const unsigned base_gprs = 50;
   const unsigned same_tier_gprs = 51;
   const unsigned worse_tier_gprs = 53;

   /* Latency-hiding schedule uses +1 GPR (same occupancy tier) and +2 bytes,
    * but eliminates 16 dependency-gap stalls: MUST NOT be vetoed. */
   agx_apple9_allocation_cost plain = {
      .spills = 0, .instructions = 10, .bytes = 80, .dependency_gaps = 32,
      .registers = base_gprs, .peak_registers = base_gprs,
   };
   agx_apple9_allocation_cost latency = {
      .spills = 0, .instructions = 10, .bytes = 82, .dependency_gaps = 16,
      .registers = same_tier_gprs, .peak_registers = same_tier_gprs,
   };
   EXPECT_TRUE(agx_apple9_allocation_cheaper(latency, plain));
   EXPECT_FALSE(agx_apple9_allocation_cheaper(plain, latency));

   /* Dropping to a lower occupancy tier IS vetoed even with fewer dep gaps. */
   agx_apple9_allocation_cost low_occ = latency;
   low_occ.registers = worse_tier_gprs;
   low_occ.dependency_gaps = 0;
   EXPECT_TRUE(agx_apple9_allocation_cheaper(plain, low_occ));
   EXPECT_FALSE(agx_apple9_allocation_cheaper(low_occ, plain));

   /* Spills remain strictly highest priority over occupancy and dep gaps. */
   agx_apple9_allocation_cost spilling = plain;
   spilling.spills = 1;
   spilling.dependency_gaps = 0;
   EXPECT_TRUE(agx_apple9_allocation_cheaper(low_occ, spilling));
   EXPECT_FALSE(agx_apple9_allocation_cheaper(spilling, low_occ));

   /* Equal spills, occupancy tier, and dep gaps tie-break by instrs -> bytes -> regs. */
   agx_apple9_allocation_cost fewer_bytes = plain;
   fewer_bytes.bytes = plain.bytes - 2;
   fewer_bytes.registers = same_tier_gprs;
   EXPECT_TRUE(agx_apple9_allocation_cheaper(fewer_bytes, plain));
   EXPECT_FALSE(agx_apple9_allocation_cheaper(plain, fewer_bytes));
}

TEST(Apple9SharedRa, NativeImageStoreExAndVolumeMinLodAllocateAndPack)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "native_img_and_minlod");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   p.fragment_shader = true;
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);

   uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t one = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f800000);
   uint32_t coords3[] = {zero, zero, zero};
   uint32_t rgba[] = {one, one, one, one};
   uint32_t packed16[] = {one, one};

   /* 3D f32 fragment, Cube u32 compute, 2D f16 fragment, plus FS fence pair. */
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, rgba, zero, 3, 32, false, true));
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, rgba, zero, 1, 32, true, false));
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, packed16, zero, 0, 16, false, true));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_DEVICE_FENCE,
      AGX_APPLE9_ENC_DEVICE_FENCE, nullptr, 0, AGX_APPLE9_FENCE_IMAGE_WRITE_FS));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_DEVICE_FENCE,
      AGX_APPLE9_ENC_DEVICE_FENCE, nullptr, 0, AGX_APPLE9_FENCE_TEX_INVALIDATE));

   /* 3D min_lod sample after the fence. */
   uint32_t samp = agx_apple9_vir_emit_texture_volume_min_lod(
      &p, coords3, zero, 0, 0, 3, false, false);
   ASSERT_NE(samp, AGX_APPLE9_VREG_INVALID);
   uint32_t out_rgba[] = {samp, samp + 1, samp + 2, samp + 3};
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, zero, out_rgba, 4, 32));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

   unsigned img_stores = 0, samples = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      img_stores += I->op == AGX_APPLE9_VIR_IMAGE_STORE;
      samples += I->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE;
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
         << (reason ?: "");
   }
   EXPECT_EQ(img_stores, 3u);
   EXPECT_EQ(samples, 1u);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, ImageFenceSlotMaskPropagatesAcrossReadFreeLoop)
{
   /* IMAGE_STORE #1 before a read-free loop and IMAGE_STORE #2 after the loop
    * must both appear in the post-loop FENCE_IMAGE_WRITE fence_slot_mask, even
    * though loop headers reset pending scoreboard state. */
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "img_fence_across_loop");
   nir_block b0 = {}, b1 = {}, b_latch = {}, b2 = {};
   b0.successors[0] = &b1;
   b1.successors[0] = &b_latch;
   b1.successors[1] = &b2; /* loop exit */
   b_latch.successors[0] = &b1; /* loop backedge (no critical edge) */

   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto *blk0 = agx_apple9_block_create(&p);
   auto *blk1 = agx_apple9_block_create(&p);
   auto *blk_latch = agx_apple9_block_create(&p);
   auto *blk2 = agx_apple9_block_create(&p);
   blk0->nir = &b0;
   blk1->nir = &b1;
   blk_latch->nir = &b_latch;
   blk2->nir = &b2;

   agx_apple9_block_begin(&p, blk0);
   uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t one = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f800000);
   uint32_t coords3[] = {zero, zero, zero};
   uint32_t rgba[] = {one, one, one, one};
   /* Emit 2 stores before the loop so store #1 occupies slot 2 and store #2
    * occupies slot 3; after the read-free loop, store #3 occupies slot 2 again
    * (a distinct subset from {2, 3}), proving slot 3 was NOT dropped across the
    * loop header! */
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, rgba, zero, 0, 32, false, false));
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, rgba, zero, 0, 32, false, false));

   agx_apple9_block_begin(&p, blk1);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_LOOP_MASK_PUSH, AGX_APPLE9_ENC_LOOP_MASK_PUSH,
      nullptr, 0, 0));

   agx_apple9_block_begin(&p, blk_latch);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_LOOP_MASK_POP, AGX_APPLE9_ENC_LOOP_MASK_POP,
      nullptr, 0, 0));

   agx_apple9_block_begin(&p, blk2);
   ASSERT_TRUE(agx_apple9_vir_emit_image_store_ex(
      &p, coords3, rgba, zero, 0, 32, false, false));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
      nullptr, 0, AGX_APPLE9_FENCE_IMAGE_WRITE));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << (reason ?: "");

   unsigned store_slots = 0;
   unsigned fence_mask = 0;
   unsigned fences = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_IMAGE_STORE)
         store_slots |= BITFIELD_BIT(I->producer_scoreboard_slot - 1);
      else if (I->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
               I->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE) {
         fences++;
         fence_mask = I->fence_slot_mask;
      }
   }
   EXPECT_EQ(fences, 1u);
   EXPECT_GE(util_bitcount(store_slots), 2u);
   EXPECT_EQ(fence_mask & store_slots, store_slots);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, TypedSharedPrivateDeviceU16AndAsyncGetSrAllocateAndPack)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "typed_mem_and_get_sr_async");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   p.private_size = 256;
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);

   /* 6 independent async GET_SR reads spread across slots 1..6 and shrink to
    * 4-byte GET_SR when allocated below r64. */
   uint32_t srs[6];
   for (unsigned i = 0; i < 6; ++i) {
      srs[i] = agx_apple9_vir_emit_get_sr_async(&p, 0xa0 + (i % 3), 0x10);
      ASSERT_NE(srs[i], AGX_APPLE9_VREG_INVALID);
   }
   uint32_t idx = srs[0];
   for (unsigned i = 1; i < 6; ++i) {
      uint32_t add[] = {idx, srs[i]};
      idx = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
   }

   /* Narrow & vector shared loads/stores: u8, u16, u16x4, u32x3. */
   uint32_t s8 = agx_apple9_vir_emit_shared_load_typed(&p, idx, 2, 1, 8);
   uint32_t s16 = agx_apple9_vir_emit_shared_load_typed(&p, idx, 2, 1, 16);
   uint32_t s16x4 = agx_apple9_vir_emit_shared_load_typed(&p, idx, 2, 4, 16);
   uint32_t s32x3 = agx_apple9_vir_emit_shared_load_typed(&p, idx, 2, 3, 32);
   ASSERT_NE(s8, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(s16, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(s16x4, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(s32x3, AGX_APPLE9_VREG_INVALID);
   uint32_t s16x4_data[] = {s16x4, s16x4 + 1};
   uint32_t s32x3_data[] = {s32x3, s32x3 + 1, s32x3 + 2};
   ASSERT_TRUE(agx_apple9_vir_emit_shared_store_typed(&p, &s8, 1, 8, idx, 2));
   ASSERT_TRUE(agx_apple9_vir_emit_shared_store_typed(&p, &s16, 1, 16, idx, 2));
   ASSERT_TRUE(agx_apple9_vir_emit_shared_store_typed(&p, s16x4_data, 4, 16, idx, 2));
   ASSERT_TRUE(agx_apple9_vir_emit_shared_store_typed(&p, s32x3_data, 3, 32, idx, 2));

   /* Narrow & wide private loads/stores: u8, u16, u32x2, u32x4. */
   uint32_t p8 = agx_apple9_vir_emit_private_load(&p, idx, 1, 1, 8);
   uint32_t p16 = agx_apple9_vir_emit_private_load(&p, idx, 17, 1, 16);
   uint32_t p32x2 = agx_apple9_vir_emit_private_load(&p, idx, 2, 2, 32);
   uint32_t p32x4 = agx_apple9_vir_emit_private_load(&p, idx, 4, 4, 32);
   ASSERT_NE(p8, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(p16, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(p32x2, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(p32x4, AGX_APPLE9_VREG_INVALID);
   uint32_t p32x2_data[] = {p32x2, p32x2 + 1};
   uint32_t p32x4_data[] = {p32x4, p32x4 + 1, p32x4 + 2, p32x4 + 3};
   ASSERT_TRUE(agx_apple9_vir_emit_private_store(&p, &p8, 1, 8, idx, 1));
   ASSERT_TRUE(agx_apple9_vir_emit_private_store(&p, &p16, 1, 16, idx, 17));
   ASSERT_TRUE(agx_apple9_vir_emit_private_store(&p, p32x2_data, 2, 32, idx, 2));
   ASSERT_TRUE(agx_apple9_vir_emit_private_store(&p, p32x4_data, 4, 32, idx, 4));

   /* Packed i16x2 and i16x4 device loads & stores. */
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };
   uint32_t d16x2 = agx_apple9_vir_emit_device_load_vector_u16(&p, 1, idx, 2, &contract);
   uint32_t d16x4 = agx_apple9_vir_emit_device_load_vector_u16(&p, 1, idx, 4, &contract);
   ASSERT_NE(d16x2, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(d16x4, AGX_APPLE9_VREG_INVALID);
   uint32_t d16x4_data[] = {d16x4, d16x4 + 1};
   ASSERT_TRUE(agx_apple9_vir_emit_device_store_vector_u16(&p, 0, idx, &d16x2, 2));
   ASSERT_TRUE(agx_apple9_vir_emit_device_store_vector_u16(&p, 0, idx, d16x4_data, 4));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

   unsigned get_sr_slots = 0, get_sr_count = 0, get_sr_compact = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_GET_SR) {
         ASSERT_GE(I->producer_scoreboard_slot, 1u);
         ASSERT_LE(I->producer_scoreboard_slot, 6u);
         get_sr_slots |= BITFIELD_BIT(I->producer_scoreboard_slot - 1);
         get_sr_compact += I->encoding == AGX_APPLE9_ENC_GET_SR;
         ++get_sr_count;
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
         << (reason ?: "");
   }
   EXPECT_EQ(get_sr_count, 6u);
   EXPECT_EQ(get_sr_compact, 6u);
   EXPECT_GT(util_bitcount(get_sr_slots), 1u);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, SubgroupAluUnlocksAndCompactImageStoreAllocateAndPack)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "hw_unlock_wave2_e2e");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   p.fragment_shader = true;
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);

   uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t one = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f800000);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };
   uint32_t a = agx_apple9_vir_emit_device_load(&p, 1, zero, &contract);
   uint32_t b = agx_apple9_vir_emit_device_load(&p, 2, zero, &contract);
   uint32_t delta = agx_apple9_vir_emit_device_load(&p, 3, zero, &contract);

   /* Subgroup reduce (folds wait on device load), inclusive scan, shuffle, rotate. */
   uint32_t r_sum = agx_apple9_vir_emit_subgroup_reduce(
      &p, AGX_APPLE9_SUBGROUP_REDUCE_FADD, a, 32, false, false);
   uint32_t r_scan = agx_apple9_vir_emit_subgroup_inclusive_scan(
      &p, AGX_APPLE9_SUBGROUP_REDUCE_IADD, a, 32, false);
   uint32_t r_qmin = agx_apple9_vir_emit_subgroup_reduce(
      &p, AGX_APPLE9_SUBGROUP_REDUCE_IMIN, b, 32, true, false);
   uint32_t r_shuf = agx_apple9_vir_emit_subgroup_shuffle(
      &p, AGX_APPLE9_SUBGROUP_SHUFFLE_XOR, r_sum, delta);
   uint32_t r_shuf_i = agx_apple9_vir_emit_subgroup_shuffle_imm(
      &p, AGX_APPLE9_QUAD_SHUFFLE_IDX, r_qmin, 2);
   uint32_t r_rot = agx_apple9_vir_emit_subgroup_rotate(&p, r_scan, b, delta);
   uint32_t r_rot_i = agx_apple9_vir_emit_subgroup_rotate_imm(&p, r_rot, a, 5);

   /* ALU unlocks: sat add/sub, andn, extr (reg & imm), bfi, pack_snorm, narrow convs. */
   uint32_t s_add = agx_apple9_vir_emit_iadd_sat(&p, r_rot_i, r_shuf_i, false, true);
   uint32_t u_sub = agx_apple9_vir_emit_iadd_sat(&p, s_add, r_shuf, true, false);
   uint32_t andn = agx_apple9_vir_emit_iandn(&p, u_sub, delta);
   uint32_t ext_r = agx_apple9_vir_emit_extr(&p, andn, b, delta);
   uint32_t ext_i = agx_apple9_vir_emit_extr_imm(&p, ext_r, a, 12);
   uint32_t bfi = agx_apple9_vir_emit_bfi_imm(&p, ext_i, 4, 8);
   uint32_t sn2 = agx_apple9_vir_emit_pack_snorm_2x16(&p, r_sum, one);
   uint32_t sn4_in[4] = {r_sum, one, a, b};
   uint32_t sn4 = agx_apple9_vir_emit_pack_snorm_4x8(&p, sn4_in);
   uint32_t h1 = agx_apple9_vir_emit_int_to_f16(&p, bfi, 32, false);
   uint32_t h2 = agx_apple9_vir_emit_int_to_f16(&p, sn2, 16, true);
   uint32_t n1 = agx_apple9_vir_emit_f32_to_narrow_int(&p, h1, 16, true);
   uint32_t n2 = agx_apple9_vir_emit_f32_to_narrow_int(&p, h2, 8, false);

   /* Tier 2 textures: 1D sample, 1D-array fetch, 2D-array implicit, Cube grad. */
   uint32_t t1d = agx_apple9_vir_emit_texture_1d(
      &p, a, AGX_APPLE9_VREG_INVALID, one, 0, 0, false);
   uint32_t t1da = agx_apple9_vir_emit_texture_1d(&p, a, zero, one, 1, 0, true);
   uint32_t c4[4] = {a, b, zero, one};
   uint32_t t2da = agx_apple9_vir_emit_texture_array_implicit(&p, c4, 2, 0, 2, true);
   uint32_t cg[9] = {a, b, one, a, b, one, a, b, one};
   uint32_t tcg = agx_apple9_vir_emit_texture_grad(&p, cg, 3, 0, 1, false, false);

   /* Compact 2D 32-bit image store consumes results -> IMAGE_WRITE_PARAMS_2D (60B). */
   uint32_t xy[2] = {n1, n2};
   uint32_t rgba[4] = {t1d, t1da, t2da, tcg};
   uint32_t mix[2] = {rgba[0], sn4};
   rgba[0] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, mix, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_image_store(&p, xy, rgba, zero,
                                               AGX_APPLE9_VREG_INVALID, false));

   ASSERT_TRUE(agx_apple9_optimize_vir(&p));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;

   unsigned compact_2d_params = 0, rotates = 0, extrs = 0, sat_adds = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->encoding == AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_2D)
         ++compact_2d_params;
      if (I->op == AGX_APPLE9_VIR_SUBGROUP_ROTATE)
         ++rotates;
      if (I->op == AGX_APPLE9_VIR_EXTR)
         ++extrs;
      if ((I->op == AGX_APPLE9_VIR_IADD || I->op == AGX_APPLE9_VIR_ISUB) &&
          I->saturate)
         ++sat_adds;
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
         << "instr " << i << " op=" << I->op << ": " << (reason ?: "");
      if (I->encoding == AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_2D) {
         EXPECT_EQ(packed.length, 24u);
      }
   }
   EXPECT_EQ(compact_2d_params, 1u);
   EXPECT_EQ(rotates, 2u);
   EXPECT_EQ(extrs, 2u);
   EXPECT_EQ(sat_adds, 2u);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, TexturePublicationRotationAndBindlessHandleWarExemption)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "tex_pub_rotate_and_handle_war");
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   /* Issue 6 back-to-back bindless texture samples whose handle GPRs die at
    * issue so RA can recycle the handle GPR immediately, while publication
    * tuples rotate across p0..p31 without colliding on p0..p3. */
   uint32_t u = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                    AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
   uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                    AGX_APPLE9_ENC_GET_SR, nullptr, 0, 1);
   uint32_t lod = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t coords[2] = {u, v};
   uint32_t results[6];
   for (unsigned i = 0; i < 6; ++i) {
      uint32_t tex_h = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                           AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, i + 1);
      uint32_t smp_h = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                           AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, i + 10);
      results[i] = agx_apple9_vir_emit_texture_lod(&p, coords, lod, 0, 0, false);
      ASSERT_NE(results[i], AGX_APPLE9_VREG_INVALID);
      auto *sample = p.instructions[p.instruction_count - 1];
      ASSERT_EQ(sample->op, AGX_APPLE9_VIR_TEXTURE_SAMPLE);
      sample->texture_bindless = true;
      sample->sampler_bindless = true;
      sample->src[sample->nr_srcs++] = tex_h;
      sample->src[sample->nr_srcs++] = smp_h;
   }
   uint32_t acc = results[0];
   for (unsigned i = 1; i < 6; ++i) {
      uint32_t add_srcs[] = {acc, results[i]};
      acc = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
                                AGX_APPLE9_ENC_FLOAT2_COMPACT, add_srcs, 2, 0);
   }
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, u, &acc, 1, 32));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << (reason ?: "");

   unsigned samples = 0, waits_during_issue = 0;
   unsigned pub_bases = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_WAIT && samples < 6)
         ++waits_during_issue;
      if (I->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         unsigned base = p.phys[I->src[0]];
         ASSERT_LT(base, 32u);
         pub_bases |= BITFIELD_BIT(base / 4);
         ++samples;
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
         << (reason ?: "");
   }
   EXPECT_EQ(samples, 6u);
   /* All 6 samples use distinct 4-word publication tuples and emit 0 WAITs while
    * issuing despite bindless handle GPR reuse. */
   EXPECT_EQ(util_bitcount(pub_bases), 6u);
   EXPECT_EQ(waits_during_issue, 0u);

   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, ImageFenceCfgFixpointAndReusedDeviceLoadWait)
{
   /* BLOCKER #1: IMAGE_STORE in block 0, conditionally fenced in block 1, then
    * block 2 reuses the IMAGE_STORE's scoreboard slot for a DEVICE_LOAD before
    * another FENCE_IMAGE_WRITE. Both fences must carry the reaching IMAGE_STORE
    * slot bit via CFG fixpoint, AND block 2's FENCE_IMAGE_WRITE must NOT
    * prematurely retire the reused DEVICE_LOAD slot. */
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "img_fence_cfg_reused_slot");
   nir_block logical[4] = {};
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   auto *b0 = agx_apple9_block_create(&p);
   auto *b1 = agx_apple9_block_create(&p);
   auto *b2 = agx_apple9_block_create(&p);
   auto *b3 = agx_apple9_block_create(&p);
   ASSERT_TRUE(b0 && b1 && b2 && b3);
   b0->nir = &logical[0];
   b1->nir = &logical[1];
   b2->nir = &logical[2];
   b3->nir = &logical[3];

   agx_apple9_block_begin(&p, b0);
   uint32_t gid = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                      AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
   uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                       AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t xy[2] = {gid, zero};
   uint32_t rgba[4] = {gid, zero, zero, zero};
   ASSERT_TRUE(agx_apple9_vir_emit_image_store(&p, xy, rgba, zero,
                                               AGX_APPLE9_VREG_INVALID, false));
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_NONE,
                                          AGX_APPLE9_ENC_JMP_EXEC_NONE, b2));

   agx_apple9_block_begin(&p, b1);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
      nullptr, 0, AGX_APPLE9_FENCE_IMAGE_WRITE));

   agx_apple9_block_begin(&p, b2);
   /* Else arm bypasses block 1's FENCE_IMAGE_WRITE and falls into b3. */

   agx_apple9_block_begin(&p, b3);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };
   /* Fill the 5 non-img_slot scoreboard slots with non-sharing device atomics,
    * then issue a DEVICE_LOAD which must evict IMAGE_STORE's slot (slot 2). */
   uint32_t vals[6];
   for (unsigned k = 0; k < 5; ++k) {
      ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
         &p, k + 1, gid, &zero, 1, AGX_APPLE9_ATOMIC_ADD, false, &vals[k]));
   }
   vals[5] = agx_apple9_vir_emit_device_load(&p, 6, gid, &contract);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
      nullptr, 0, AGX_APPLE9_FENCE_IMAGE_WRITE));
   uint32_t acc = vals[0];
   for (unsigned k = 1; k < 6; ++k) {
      uint32_t srcs[2] = {acc, vals[k]};
      acc = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                AGX_APPLE9_ENC_INT_ADD_EXTENDED, srcs, 2, 0);
   }
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, gid, &acc, 1, 32));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << (reason ?: "");

   unsigned img_slot = 0;
   unsigned fences = 0;
   unsigned fence2_idx = UINT_MAX;
   bool reused_load_seen = false;
   bool waited_after_fence2 = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_IMAGE_STORE) {
         img_slot = I->producer_scoreboard_slot;
         ASSERT_GE(img_slot, 1u);
         ASSERT_LE(img_slot, 6u);
      }
      if (I->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
          I->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE) {
         ++fences;
         ASSERT_NE(img_slot, 0u);
         EXPECT_NE(I->fence_slot_mask & BITFIELD_BIT(img_slot - 1), 0u)
            << "fence #" << fences << " missed reaching IMAGE_STORE slot " << img_slot;
         if (fences == 2)
            fence2_idx = i;
      }
      if (fences == 1 && I->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
          I->producer_scoreboard_slot == img_slot)
         reused_load_seen = true;
      if (i > fence2_idx) {
         if (I->op == AGX_APPLE9_VIR_WAIT &&
             (I->immediate & BITFIELD_BIT(img_slot - 1)))
            waited_after_fence2 = true;
         if (I->scoreboard_slot == img_slot)
            waited_after_fence2 = true;
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
         << (reason ?: "");
   }
   EXPECT_EQ(fences, 2u);
   EXPECT_TRUE(reused_load_seen);
   EXPECT_TRUE(waited_after_fence2);

   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, CrossDomainAcquireBarrierDeviceAndShared)
{
   /* HIGH #4: acquire ordering across FENCE_COMPILER_BARRIER must order
    * DEVICE_LOAD -> SHARED_LOAD and SHARED_LOAD -> DEVICE_LOAD. */
   for (unsigned dir = 0; dir < 2; ++dir) {
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         dir == 0 ? "acquire_dev_to_shared" : "acquire_shared_to_dev");
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);

      uint32_t gid = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                         AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
      agx_apple9_device_load_contract contract = {
         .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
         .flags = AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
         .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
      };
      uint32_t first, second;
      if (dir == 0) {
         first = agx_apple9_vir_emit_device_load(&p, 1, gid, &contract);
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
            &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_PSEUDO,
            nullptr, 0, AGX_APPLE9_FENCE_COMPILER_BARRIER));
         second = agx_apple9_vir_emit_shared_load(&p, gid, 0, 1);
      } else {
         first = agx_apple9_vir_emit_shared_load(&p, gid, 0, 1);
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
            &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_PSEUDO,
            nullptr, 0, AGX_APPLE9_FENCE_COMPILER_BARRIER));
         second = agx_apple9_vir_emit_device_load(&p, 1, gid, &contract);
      }
      uint32_t srcs[2] = {first, second};
      uint32_t sum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                         AGX_APPLE9_ENC_INT_ADD_EXTENDED,
                                         srcs, 2, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, gid, &sum, 1, 32));

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason))
         << (reason ?: "");

      enum agx_apple9_vir_opcode first_op =
         dir == 0 ? AGX_APPLE9_VIR_DEVICE_LOAD : AGX_APPLE9_VIR_SHARED_LOAD;
      enum agx_apple9_vir_opcode second_op =
         dir == 0 ? AGX_APPLE9_VIR_SHARED_LOAD : AGX_APPLE9_VIR_DEVICE_LOAD;
      unsigned first_slot = 0;
      bool waited_before_second = false, seen_second = false;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if (I->op == first_op && !first_slot)
            first_slot = I->producer_scoreboard_slot;
         if (first_slot && !seen_second) {
            if (I->op == AGX_APPLE9_VIR_WAIT &&
                (I->immediate & BITFIELD_BIT(first_slot - 1)))
               waited_before_second = true;
            if (I->op == second_op) {
               if (I->scoreboard_slot == first_slot)
                  waited_before_second = true;
               seen_second = true;
            }
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << (reason ?: "");
      }
      EXPECT_NE(first_slot, 0u);
      EXPECT_TRUE(seen_second);
      EXPECT_TRUE(waited_before_second)
         << "dir=" << dir << " missed acquire wait for slot " << first_slot;

      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, EmptyProgramAllocatesCleanly)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "empty_vir");
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason))
      << (reason ?: "");
   EXPECT_TRUE(p.physical);
   EXPECT_TRUE(p.dependencies_finalized);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}
