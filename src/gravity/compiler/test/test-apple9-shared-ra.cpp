/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>
#include "agx_test.h"

static void
test_shared_loop_initializers(bool computed, bool live_source = false)
{
   for (uint32_t initial : {0u, 7u, 0x3f800000u, 0xfeedfaceu}) {
      void *memctx = ralloc_context(nullptr);
      auto ctx = rzalloc(memctx, agx_context);
      list_inithead(&ctx->blocks);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "shared_constant_phis");
      agx_shader_key key = {};
      key.has_scratch = true;
      ctx->nir = nir.shader;
      ctx->stage = MESA_SHADER_COMPUTE;
      ctx->key = &key;
      ctx->ra_target.max_registers = 32;
      ctx->ra_target.cost_register_constraints = true;
      ctx->ra_target.disable_occupancy_rematerialization = true;
      auto entry = agx_test_block(ctx);
      auto header = agx_test_block(ctx);
      auto body = agx_test_block(ctx);
      auto exit = agx_test_block(ctx);
      header->loop_header = true;
      agx_block_add_successor(entry, header);
      agx_block_add_successor(header, body);
      agx_block_add_successor(header, exit);
      agx_block_add_successor(body, header);

      auto b = agx_init_builder(ctx, agx_after_block(entry));
      auto constant = agx_mov_imm(&b, 32, initial);
      if (computed)
         constant = agx_iadd(&b, constant, agx_immediate(0), 0);
      constexpr unsigned count = 8;
      agx_instr *phis[count];
      b.cursor = agx_after_block(header);
      for (unsigned i = 0; i < count; ++i) {
         phis[i] = agx_phi_to(&b, agx_temp(ctx, AGX_SIZE_32), 2);
         phis[i]->src[0] = constant;
      }
      b.cursor = agx_after_block(body);
      for (unsigned i = 0; i < count; ++i)
         phis[i]->src[1] = agx_iadd(&b, phis[i]->dest[0],
                                    agx_immediate(i + 1), 0);
      b.cursor = agx_after_block(exit);
      agx_index results[count];
      for (unsigned i = 0; i < count; ++i)
         results[i] = live_source
            ? agx_iadd(&b, phis[i]->dest[0], constant, 0)
            : phis[i]->dest[0];
      for (unsigned i = 0; i < count; ++i)
         agx_export(&b, results[i], 2 * i);

      ASSERT_TRUE(agx_ra(ctx));
      ASSERT_EQ(ctx->scratch_size_B, 0u);
      ASSERT_LT(ctx->max_reg, 32u);
      uint32_t regs[32] = {};
      auto execute = [&](agx_block *block) {
         agx_foreach_instr_in_block(block, I) {
            auto read = [&](agx_index x) {
               return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
            };
            switch (I->op) {
            case AGX_OPCODE_MOV_IMM:
               regs[I->dest[0].value] = I->imm;
               break;
            case AGX_OPCODE_MOV:
               regs[I->dest[0].value] = read(I->src[0]);
               break;
            case AGX_OPCODE_IADD:
               regs[I->dest[0].value] = read(I->src[0]) + read(I->src[1]);
               break;
            case AGX_OPCODE_SWAP:
               std::swap(regs[I->src[0].value], regs[I->src[1].value]);
               break;
            case AGX_OPCODE_EXPORT:
               regs[I->imm] = read(I->src[0]);
               break;
            default:
               FAIL() << "unexpected opcode " << I->op;
            }
         }
      };
      execute(entry);
      for (unsigned iteration = 0; iteration < 5; ++iteration) {
         execute(header);
         execute(body);
      }
      execute(header);
      execute(exit);
      for (unsigned i = 0; i < count; ++i)
         EXPECT_EQ(regs[2 * i], initial + 5 * (i + 1) +
                   (live_source ? initial : 0));
      ralloc_free(nir.shader);
      ralloc_free(memctx);
   }
}

TEST(Apple9SharedRa, SharedConstantInitializersKeepIndependentLoopValues)
{
   test_shared_loop_initializers(false);
}

TEST(Apple9SharedRa, SharedComputedInitializersKeepIndependentLoopValues)
{
   test_shared_loop_initializers(true);
}

TEST(Apple9SharedRa, PhiFanoutKeepsStillLiveSourcesSeparate)
{
   test_shared_loop_initializers(false, true);
   test_shared_loop_initializers(true, true);
}

TEST(Apple9SharedRa, PlacementCostsLeaveSpaceForRestrictedDefinitions)
{
   unsigned moves[2] = {};
   for (unsigned policy = 0; policy < 2; ++policy) {
      void *memctx = ralloc_context(nullptr);
      auto ctx = rzalloc(memctx, agx_context);
      list_inithead(&ctx->blocks);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "scarce_bank_cost");
      agx_shader_key key = {};
      key.has_scratch = true;
      ctx->nir = nir.shader; ctx->stage = MESA_SHADER_COMPUTE; ctx->key = &key;
      ctx->ra_target.max_registers = 32;
      ctx->ra_target.reserved_registers = 4;
      ctx->ra_target.late_kill_sources = true;
      ctx->ra_target.cost_register_constraints = policy;
      ctx->ra_target.disable_occupancy_rematerialization = true;
      auto block = agx_test_block(ctx);
      auto b = agx_init_builder(ctx, agx_after_block(block));
      agx_index v[12];
      for (unsigned i = 0; i < 12; ++i)
         v[i] = agx_mov_imm(&b, 32, i + 1);
      auto low = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), v[0], v[1], 0);
      low->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      low->reg_constraints[0] = {4,6,2};
      low->reg_constraints[1] = low->reg_constraints[2] = {0,30,2};
      auto sum = low->dest[0];
      for (unsigned i = 0; i < 12; ++i)
         sum = agx_iadd(&b, sum, v[i], 0);
      agx_export(&b, sum, 0);
      agx_ra(ctx);
      ASSERT_EQ(ctx->scratch_size_B, 0u);
      ASSERT_LT(ctx->max_reg, 32u);
      uint32_t regs[32] = {};
      agx_foreach_instr_in_block(block, I) {
         auto read = [&](agx_index x) {
            return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
         };
         switch (I->op) {
         case AGX_OPCODE_MOV_IMM: regs[I->dest[0].value] = I->imm; break;
         case AGX_OPCODE_MOV:
            regs[I->dest[0].value] = read(I->src[0]); ++moves[policy]; break;
         case AGX_OPCODE_IADD:
            regs[I->dest[0].value] = read(I->src[0]) + read(I->src[1]); break;
         case AGX_OPCODE_SWAP: {
            auto a = I->src[0].value, c = I->src[1].value;
            std::swap(regs[a], regs[c]); ++moves[policy]; break;
         }
         case AGX_OPCODE_EXPORT: regs[I->imm] = read(I->src[0]); break;
         default: FAIL() << "unexpected opcode " << I->op;
         }
      }
      EXPECT_EQ(regs[0], 81u);
      ralloc_free(nir.shader); ralloc_free(memctx);
   }
   EXPECT_LT(moves[1], moves[0]);
}

TEST(Apple9SharedRa, ConstantMaterializationRespectsOperandPreferences)
{
   for (bool high_use : {false, true}) {
      void *memctx = ralloc_context(nullptr);
      auto ctx = rzalloc(memctx, agx_context);
      list_inithead(&ctx->blocks);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "constant_bank_cost");
      agx_shader_key key = {};
      key.has_scratch = true;
      ctx->nir = nir.shader; ctx->stage = MESA_SHADER_COMPUTE; ctx->key = &key;
      ctx->ra_target.max_registers = 192;
      ctx->ra_target.cost_register_constraints = true;
      ctx->ra_target.disable_occupancy_rematerialization = true;
      ctx->ra_target.mov_imm_registers = 128;
      auto block = agx_test_block(ctx);
      auto b = agx_init_builder(ctx, agx_after_block(block));
      auto constant = agx_mov_imm_to(&b, agx_temp(ctx, AGX_SIZE_32), 0xfeedface);
      auto use = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32),
                              constant->dest[0], agx_immediate(1), 0);
      use->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      /* A later restricted destination makes the low bank scarce. The high
       * export keeps both banks within the already-required footprint. */
      use->reg_constraints[0] = {0, 126, 2};
      use->reg_constraints[1] = {uint16_t(high_use ? 128 : 0), 190, 2};
      use->reg_constraints[2] = {0, 190, 2};
      agx_export(&b, use->dest[0], 170);
      ASSERT_TRUE(agx_ra(ctx));
      EXPECT_EQ(ctx->scratch_size_B, 0u);
      EXPECT_EQ(constant->dest[0].value >= 128, high_use);
      EXPECT_EQ(use->src[0].value, constant->dest[0].value);
      ralloc_free(nir.shader); ralloc_free(memctx);
   }
}

TEST(Apple9SharedRa, ConstantMaterializationKeepsPhiEdgePreference)
{
   void *memctx = ralloc_context(nullptr);
   auto ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "constant_phi_bank");
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader; ctx->stage = MESA_SHADER_COMPUTE; ctx->key = &key;
   ctx->ra_target.max_registers = 192;
   ctx->ra_target.cost_register_constraints = true;
   ctx->ra_target.disable_occupancy_rematerialization = true;
   ctx->ra_target.mov_imm_registers = 128;
   auto entry = agx_test_block(ctx), header = agx_test_block(ctx);
   auto body = agx_test_block(ctx), exit = agx_test_block(ctx);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);
   auto b = agx_init_builder(ctx, agx_after_block(entry));
   auto initial = agx_preload(&b, agx_register(160, AGX_SIZE_32));
   b.cursor = agx_after_block(header);
   auto phi = agx_phi_to(&b, agx_temp(ctx, AGX_SIZE_32), 2);
   phi->src[0] = initial;
   b.cursor = agx_after_block(body);
   auto constant = agx_mov_imm_to(&b, agx_temp(ctx, AGX_SIZE_32), 7);
   phi->src[1] = constant->dest[0];
   b.cursor = agx_after_block(exit);
   agx_export(&b, phi->dest[0], 160);
   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   EXPECT_EQ(constant->dest[0].value, 160u);
   agx_foreach_instr_in_block(body, I) {
      EXPECT_NE(I->op, AGX_OPCODE_MOV);
      EXPECT_NE(I->op, AGX_OPCODE_SWAP);
   }
   ralloc_free(nir.shader); ralloc_free(memctx);
}

TEST(Apple9SharedRa, InstructionConstraintsRelocateLiveValuesOutOfFullBanks)
{
   void *memctx = ralloc_context(nullptr);
   auto ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "constrained_banks");
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 128;
   ctx->ra_target.reserved_registers = 8;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   auto block = agx_test_block(ctx);
   auto b = agx_init_builder(ctx, agx_after_block(block));
   constexpr unsigned count = 56;
   agx_index values[count];
   uint32_t expected[count];
   for (unsigned i = 0; i < count; ++i) {
      expected[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      values[i] = agx_mov_imm(&b, 32, expected[i]);
   }
   std::vector<agx_instr *> constrained;
   auto sum = agx_mov_imm(&b, 32, 0);
   uint32_t expected_sum = 0;
   for (unsigned i = 0; i < count; ++i) {
      /* All 56 values remain live. Both inputs and the output must fit in
       * r4-r7, so simply searching for an unused constrained register fails. */
      auto I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32),
                           values[i], values[count - 1 - i], 0);
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      for (unsigned c = 0; c < 3; ++c)
         I->reg_constraints[c] = {8, 14, 2};
      constrained.push_back(I);
      sum = agx_iadd(&b, sum, I->dest[0], 0);
      expected_sum += expected[i] + expected[count - 1 - i];
   }
   /* Check every relocated value, not just the constrained results. */
   for (unsigned i = 0; i < count; ++i) {
      sum = agx_iadd(&b, values[i], sum, 5);
      expected_sum = expected[i] + expected_sum * 32;
   }
   agx_export(&b, sum, 120);
   agx_ra(ctx);
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   for (auto I : constrained) {
      EXPECT_GE(I->dest[0].value, 8u);
      EXPECT_LE(I->dest[0].value, 14u);
      for (unsigned s = 0; s < 2; ++s) {
         EXPECT_GE(I->src[s].value, 8u);
         EXPECT_LE(I->src[s].value, 14u);
      }
   }
   uint32_t regs[128] = {};
   unsigned moves = 0;
   agx_foreach_instr_in_block(block, I) {
      auto read = [&](agx_index x) {
         return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
      };
      switch (I->op) {
      case AGX_OPCODE_MOV_IMM:
         regs[I->dest[0].value] = I->imm;
         break;
      case AGX_OPCODE_IADD:
         regs[I->dest[0].value] = read(I->src[0]) + (read(I->src[1]) << I->shift);
         break;
      case AGX_OPCODE_MOV:
         regs[I->dest[0].value] = read(I->src[0]);
         ++moves;
         break;
      case AGX_OPCODE_SWAP: {
         auto old = read(I->src[0]);
         regs[I->src[0].value] = read(I->src[1]);
         regs[I->src[1].value] = old;
         break;
      }
      case AGX_OPCODE_EXPORT:
         regs[I->imm] = read(I->src[0]);
         break;
      default:
         FAIL() << "unexpected opcode " << I->op;
      }
   }
   EXPECT_EQ(regs[120], expected_sum);
   EXPECT_GT(moves, 0u);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}

TEST(Apple9SharedRa, SpilledLoopPhisPreserveParallelAssignments)
{
   void *memctx = ralloc_context(NULL);
   agx_context *ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared_spill_loop");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = 1;
   nir.shader->info.workgroup_size[2] = 1;
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 128;
   ctx->ra_target.reserved_registers = 32;
   ctx->ra_target.spill_copy_register = 24;
   ctx->ra_target.spill_swap_registers[0] = 26;
   ctx->ra_target.spill_swap_registers[1] = 28;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   agx_block *entry = agx_test_block(ctx);
   agx_block *header = agx_test_block(ctx);
   agx_block *body = agx_test_block(ctx);
   agx_block *exit = agx_test_block(ctx);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);
   const unsigned count = 80;
   agx_index initial[count], value[count];
   agx_instr *phi[count];
   uint32_t expected_values[count];
   agx_builder b = agx_init_builder(ctx, agx_after_block(entry));
   for (unsigned i = 0; i < count; ++i) {
      expected_values[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      initial[i] = agx_mov_imm(&b, 32, expected_values[i]);
   }
   b.cursor = agx_after_block(header);
   for (unsigned i = 0; i < count; ++i) {
      value[i] = agx_temp(ctx, AGX_SIZE_32);
      phi[i] = agx_phi_to(&b, value[i], 2);
      phi[i]->src[0] = initial[i];
   }
   b.cursor = agx_after_block(body);
   for (unsigned i = 0; i < count; ++i)
      phi[i]->src[1] = agx_iadd(&b, value[i], value[(i + 1) % count], 0);
   b.cursor = agx_after_block(exit);
   agx_index hash = value[0];
   for (unsigned i = 1; i < count; ++i)
      hash = agx_iadd(&b, value[i], hash, 5);
   agx_export(&b, hash, 120);
   agx_ra(ctx);
   ASSERT_GT(ctx->scratch_size_B, 0u);
   EXPECT_LE(ctx->scratch_size_B, 4096u);
   EXPECT_LT(ctx->max_reg, 128u);
   uint32_t regs[128] = {}, memory[2048] = {};
   unsigned stores = 0, loads = 0;
   auto execute = [&](agx_block *block) {
      agx_foreach_instr_in_block(block, I) {
         auto read = [&](agx_index x) {
            return x.type == AGX_INDEX_IMMEDIATE
                      ? x.value
                      : (x.memory ? memory : regs)[x.value];
         };
         uint32_t result;
         switch (I->op) {
         case AGX_OPCODE_MOV_IMM:
            result = I->imm;
            break;
         case AGX_OPCODE_IADD:
            result = read(I->src[0]) + (read(I->src[1]) << I->shift);
            break;
         case AGX_OPCODE_MOV:
            result = read(I->src[0]);
            stores += I->dest[0].memory;
            loads += I->src[0].memory;
            break;
         case AGX_OPCODE_EXPORT:
            regs[I->imm] = read(I->src[0]);
            continue;
         case AGX_OPCODE_SWAP: {
            uint32_t old = read(I->src[0]);
            regs[I->src[0].value] = read(I->src[1]);
            regs[I->src[1].value] = old;
            continue;
         }
         default:
            ADD_FAILURE() << "unexpected opcode " << I->op;
            return;
         }
         auto dst = I->dest[0];
         ASSERT_EQ(dst.type, AGX_INDEX_REGISTER);
         ASSERT_LT(dst.value, dst.memory ? 2048u : 128u);
         (dst.memory ? memory : regs)[dst.value] = result;
      }
   };
   execute(entry);
   for (unsigned iteration = 0; iteration < 3; ++iteration) {
      execute(header);
      execute(body);
      uint32_t first = expected_values[0];
      for (unsigned i = 0; i + 1 < count; ++i)
         expected_values[i] += expected_values[i + 1];
      expected_values[count - 1] += first;
   }
   execute(header);
   execute(exit);
   uint32_t expected = expected_values[0];
   for (unsigned i = 1; i < count; ++i)
      expected = expected * 32 + expected_values[i];
   EXPECT_EQ(regs[120], expected);
   EXPECT_GT(stores, 0u);
   EXPECT_GT(loads, 0u);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}

#include "agx_apple9_ir.h"

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

template <unsigned count>
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
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   EXPECT_TRUE(p.physical);
   if (count > 96)
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
   EXPECT_EQ(saves > 0, count > 96);
   EXPECT_EQ(fills > 0, count > 96);
   EXPECT_TRUE(pending.empty());
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, PhysicalSpillLoweringPreservesLiveSourcesAndLoopPhis)
{
   check_loop_register_pressure<80>();
   check_loop_register_pressure<112>();
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

TEST(Apple9SharedRa, RematerializationPreservesHighOperandConstraints)
{
   void *memctx = ralloc_context(nullptr);
   auto ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "high_bank_rematerialization");
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 192;
   ctx->ra_target.reserved_registers = 8;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   auto block = agx_test_block(ctx);
   auto b = agx_init_builder(ctx, agx_after_block(block));
   agx_index values[72];
   uint32_t expected = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      uint32_t value = i * 0x9e3779b9u + 0xa511e9b3u;
      values[i] = agx_mov_imm(&b, 32, value);
      expected += value;
   }
   auto sum = agx_mov_imm(&b, 32, 0);
   std::vector<agx_instr *> constrained;
   for (auto value : values) {
      auto I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), value, sum, 0);
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      I->reg_constraints[0] = {0, 190, 2};
      I->reg_constraints[1] = {128, 190, 2};
      I->reg_constraints[2] = {0, 190, 2};
      constrained.push_back(I);
      sum = I->dest[0];
   }
   agx_export(&b, sum, 8);
   agx_ra(ctx);
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   for (auto I : constrained) {
      EXPECT_GE(I->src[0].value, 128u);
      EXPECT_LE(I->src[0].value, 190u);
   }
   uint32_t regs[192] = {};
   agx_foreach_instr_in_block(block, I) {
      auto read = [&](agx_index x) {
         return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
      };
      switch (I->op) {
      case AGX_OPCODE_MOV_IMM:
         regs[I->dest[0].value] = I->imm;
         break;
      case AGX_OPCODE_IADD:
         regs[I->dest[0].value] = read(I->src[0]) + read(I->src[1]);
         break;
      case AGX_OPCODE_MOV:
         regs[I->dest[0].value] = read(I->src[0]);
         break;
      case AGX_OPCODE_SWAP:
         std::swap(regs[I->src[0].value], regs[I->src[1].value]);
         break;
      case AGX_OPCODE_EXPORT:
         regs[I->imm] = read(I->src[0]);
         break;
      default:
         FAIL() << "unexpected opcode " << I->op;
      }
   }
   EXPECT_EQ(regs[8], expected);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}

TEST(Apple9SharedRa, RejectsImpossibleConstraintWindows)
{
   /* Includes the spill-prefix conflict formerly encountered in CS2, empty
    * aligned windows, inverted bounds, bad alignment and simultaneous pins. */
   const agx_reg_constraint cases[] = {
      {0, 6, 2}, {11, 11, 2}, {20, 18, 2}, {10, 30, 3}, {32, 40, 2}, {10, 10, 2},
   };
   for (const auto &constraint : cases) {
      void *memctx = ralloc_context(nullptr);
      auto ctx = rzalloc(memctx, agx_context);
      list_inithead(&ctx->blocks);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "invalid_constraint");
      agx_shader_key key = {};
      key.has_scratch = true;
      ctx->nir = nir.shader; ctx->stage = MESA_SHADER_COMPUTE; ctx->key = &key;
      ctx->ra_target.max_registers = 32;
      ctx->ra_target.reserved_registers = 10;
      ctx->ra_target.disable_occupancy_rematerialization = true;
      auto block = agx_test_block(ctx);
      auto b = agx_init_builder(ctx, agx_after_block(block));
      auto a = agx_mov_imm(&b, 32, 1);
      auto c = agx_mov_imm(&b, 32, 2);
      auto I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), a, c, 0);
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      I->reg_constraints[0] = {10, 30, 2};
      I->reg_constraints[1] = I->reg_constraints[2] = constraint;
      agx_export(&b, I->dest[0], 0);
      EXPECT_FALSE(agx_ra(ctx));
      EXPECT_NE(ctx->ra_error, nullptr);
      ralloc_free(nir.shader); ralloc_free(memctx);
   }
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
      bool saw_fence = false;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if (I->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
             I->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE)
            saw_fence = true;
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
      }
      EXPECT_EQ(stores, count);
      EXPECT_EQ(tuple_mask, BITFIELD_MASK(6)); /* All 6 tuples p0..p40 used. */
      EXPECT_EQ(slot_mask, BITFIELD_MASK(6));  /* All 6 slots 1..6 used. */
      /* 6 back-to-back stores issue with ZERO waits between them; the 7th
       * waits once when reusing p0..p7. All pending stores drain before 0x0650. */
      EXPECT_EQ(waits_before_fence, count == 6 ? 0u : 1u);
      EXPECT_EQ(waits_at_fence, 1u);
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
