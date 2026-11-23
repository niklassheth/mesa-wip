/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_assign.h"
#include "apple9_ra_pcopy.h"
#include "apple9_ra_sched.h"
#include "apple9_ra_spill.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

/* Helper for manually wiring logical SSA CFG edges in unit tests. */
static void
cfg_init(agx_apple9_vir_program *p,
         const std::vector<agx_apple9_block *> &blocks)
{
   for (unsigned i = 0; i < blocks.size(); ++i) {
      blocks[i]->cfg_active = true;
      blocks[i]->cfg_index = i;
      blocks[i]->cfg_succs[0] = blocks[i]->cfg_succs[1] = nullptr;
      free(blocks[i]->cfg_preds);
      blocks[i]->cfg_preds = nullptr;
      blocks[i]->num_cfg_preds = 0;
      blocks[i]->loop_header = false;
   }
   p->num_cfg_blocks = static_cast<unsigned>(blocks.size());
}

static void
cfg_edge(agx_apple9_block *from, agx_apple9_block *to)
{
   if (!from->cfg_succs[0])
      from->cfg_succs[0] = to;
   else {
      assert(!from->cfg_succs[1]);
      from->cfg_succs[1] = to;
   }
   to->cfg_preds = static_cast<agx_apple9_block **>(
      realloc(to->cfg_preds, (to->num_cfg_preds + 1) * sizeof(*to->cfg_preds)));
   to->cfg_preds[to->num_cfg_preds++] = from;
   if (to->cfg_index <= from->cfg_index)
      to->loop_header = true;
}

/* Lightweight physical Apple9 VIR interpreter for verifying semantic
 * preservation across ALU, copies, XOR swaps, spills, and loops. */
struct PhysVirSim {
   std::array<uint32_t, AGX_APPLE9_GPR_COUNT> gpr = {};
   std::vector<uint32_t> scratch;

   explicit PhysVirSim(unsigned scratch_words = 4096) : scratch(scratch_words, 0)
   {
   }

   void exec_block(const agx_apple9_block *b)
   {
      for (const agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         switch (ins->op) {
         case AGX_APPLE9_VIR_IMM:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = ins->immediate;
            break;
         case AGX_APPLE9_VIR_IADD:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = gpr[ins->src[0]] + gpr[ins->src[1]];
            break;
         case AGX_APPLE9_VIR_ISUB:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = gpr[ins->src[0]] - gpr[ins->src[1]];
            break;
         case AGX_APPLE9_VIR_IOR:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = gpr[ins->src[0]] | gpr[ins->src[1]];
            break;
         case AGX_APPLE9_VIR_IXOR:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = gpr[ins->src[0]] ^ gpr[ins->src[1]];
            break;
         case AGX_APPLE9_VIR_IAND:
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = gpr[ins->src[0]] & gpr[ins->src[1]];
            break;
         case AGX_APPLE9_VIR_SPILL_STORE:
            ASSERT_LT(ins->immediate, scratch.size());
            ASSERT_LT(ins->src[0], AGX_APPLE9_GPR_COUNT);
            scratch[ins->immediate] = gpr[ins->src[0]];
            /* Hardware SAVE destroys its input GPR: poison it to catch bugs! */
            gpr[ins->src[0]] = 0xDEADBEEFu;
            break;
         case AGX_APPLE9_VIR_SPILL_LOAD:
            ASSERT_LT(ins->immediate, scratch.size());
            ASSERT_LT(ins->dest, AGX_APPLE9_GPR_COUNT);
            gpr[ins->dest] = scratch[ins->immediate];
            break;
         case AGX_APPLE9_VIR_PRIVATE_STORE: {
            unsigned word = ins->immediate + gpr[ins->src[1]];
            ASSERT_LT(word, scratch.size());
            scratch[word] = gpr[ins->src[0]];
            break;
         }
         case AGX_APPLE9_VIR_PRIVATE_LOAD: {
            unsigned word = ins->immediate + gpr[ins->src[0]];
            ASSERT_LT(word, scratch.size());
            gpr[ins->dest] = scratch[word];
            break;
         }
         case AGX_APPLE9_VIR_WAIT:
         case AGX_APPLE9_VIR_HALT:
         case AGX_APPLE9_VIR_ABI_RETURN:
            break;
         default:
            FAIL() << "Unhandled VIR opcode in PhysVirSim: " << ins->op;
         }
      }
   }
};

/* Pack every instruction in a physical VIR program and return total byte size. */
static unsigned
verify_binary_emission(const agx_apple9_vir_program *p)
{
   const char *reason = nullptr;
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(p, &reason))
      << "agx_apple9_validate_vir_allocation failed: "
      << (reason ? reason : "(null)");

   unsigned total_bytes = 0;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const agx_apple9_vir_instr *ins = p->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         /* Device stores use the scalar/vector helper packers directly. */
         agx_apple9_packed_instruction packed = {};
         unsigned gprs = agx_apple9_mem_data_gprs(ins);
         bool ok = false;
         if (ins->memory_components == 1) {
            ok = agx_apple9_pack_device_store_scalar(
               p->phys[ins->src[0]], p->phys[ins->src[1]], ins->immediate,
               ins->memory_bits,
               static_cast<agx_apple9_scoreboard_slot>(ins->scoreboard_slot),
               !(ins->live_after_mask & BITFIELD_BIT(1)), &packed);
         } else {
            ok = agx_apple9_pack_device_store_vector_u32(
               p->phys[ins->src[0]], p->phys[ins->src[gprs]], ins->immediate,
               ins->memory_components,
               static_cast<agx_apple9_scoreboard_slot>(ins->scoreboard_slot),
               !(ins->live_after_mask & BITFIELD_BIT(gprs)), &packed);
         }
         EXPECT_TRUE(ok) << "pack device store failed at insn " << i;
         total_bytes += packed.length;
         continue;
      }
      agx_apple9_packed_instruction packed = {};
      reason = nullptr;
      EXPECT_TRUE(agx_apple9_pack_vir_instruction(ins, p->phys, &packed, &reason))
         << "agx_apple9_pack_vir_instruction failed at insn " << i
         << " (op=" << ins->op << " enc=" << ins->encoding << "): "
         << (reason ? reason : "(null)");
      EXPECT_GT(packed.length, 0u) << "zero-length packed insn at " << i;
      total_bytes += packed.length;
   }
   return total_bytes;
}

class Apple9RaAssignTest : public ::testing::Test {
 protected:
   agx_apple9_vir_program p;

   void SetUp() override { agx_apple9_vir_init(&p); }
   void TearDown() override { agx_apple9_vir_finish(&p); }

   uint32_t emit_imm(uint32_t val,
                     agx_apple9_encoding enc = AGX_APPLE9_ENC_MOV_IMM32)
   {
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM, enc, nullptr, 0, val);
   }

   uint32_t emit_iadd(uint32_t a, uint32_t b)
   {
      uint32_t s[2] = {a, b};
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                 AGX_APPLE9_ENC_INT_ADD_EXTENDED, s, 2, 0);
   }

   uint32_t emit_isub(uint32_t a, uint32_t b)
   {
      uint32_t s[2] = {a, b};
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_ISUB,
                                 AGX_APPLE9_ENC_INT_ADD_EXTENDED, s, 2, 0);
   }

   uint32_t emit_ixor(uint32_t a, uint32_t b)
   {
      uint32_t s[2] = {a, b};
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
                                 AGX_APPLE9_ENC_LOGIC_EXTENDED, s, 2, 0);
   }

   void emit_halt()
   {
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));
   }

   apple9_ra_assign_options validated_opts()
   {
      apple9_ra_assign_options opts = apple9_ra_assign_default_options();
      opts.validate = true;
      return opts;
   }
};

} /* anonymous namespace */

/* =========================================================================
 * 1. Straight-Line Arithmetic, Reuse-Killed & Binary Emission
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, StraightLineReuseKilledAndEmit)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t a = emit_imm(10);
   uint32_t b = emit_imm(25);
   uint32_t c = emit_iadd(a, b);
   uint32_t d = emit_imm(7);
   uint32_t e = emit_iadd(c, d);
   p.output = e;
   emit_halt();

   cfg_init(&p, {b0});

   auto opts = validated_opts();
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");

   EXPECT_TRUE(p.physical);
   EXPECT_TRUE(p.dependencies_finalized);
   EXPECT_FALSE(res.has_spills);
   EXPECT_EQ(res.reserved_gprs, 0u);
   /* Reuse-killed keeps straight-line chain in just 2 GPRs (r0, r1)! */
   EXPECT_LE(res.max_phys_gpr, 2u);

   PhysVirSim sim;
   sim.exec_block(p.blocks);
   EXPECT_EQ(sim.gpr[p.output], 42u);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 2. Diamond CFG with VIR_PHI + VIR_PHI_SRC Canonicalization
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, DiamondCfgPhiCanonicalizeAndSimulate)
{
   auto *entry = agx_apple9_block_create(&p);
   auto *left = agx_apple9_block_create(&p);
   auto *right = agx_apple9_block_create(&p);
   auto *merge = agx_apple9_block_create(&p);

   agx_apple9_block_begin(&p, entry);
   uint32_t base = emit_imm(100);

   agx_apple9_block_begin(&p, left);
   uint32_t c1 = emit_imm(11);
   uint32_t v_left = emit_iadd(base, c1);

   agx_apple9_block_begin(&p, right);
   uint32_t c2 = emit_imm(22);
   uint32_t v_right = emit_iadd(base, c2);

   agx_apple9_block_begin(&p, merge);
   uint32_t phi_v = agx_apple9_vir_emit_phi(&p, merge);
   uint32_t out_v = emit_iadd(phi_v, base);
   p.output = out_v;
   emit_halt();

   /* Add legacy VIR_PHI_SRC edges to verify canonicalization. */
   p.current_block = left;
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi_v, v_left));
   p.current_block = right;
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi_v, v_right));

   cfg_init(&p, {entry, left, right, merge});
   cfg_edge(entry, left);
   cfg_edge(entry, right);
   cfg_edge(left, merge);
   cfg_edge(right, merge);

   auto opts = validated_opts();
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);

   /* Execute both paths through the physical blocks. */
   auto *pb_entry = p.blocks;
   auto *pb_left = pb_entry->next;
   auto *pb_right = pb_left->next;
   auto *pb_merge = pb_right->next;

   {
      PhysVirSim sim;
      sim.exec_block(pb_entry);
      sim.exec_block(pb_left);
      sim.exec_block(pb_merge);
      EXPECT_EQ(sim.gpr[p.output], 211u);
   }
   {
      PhysVirSim sim;
      sim.exec_block(pb_entry);
      sim.exec_block(pb_right);
      sim.exec_block(pb_merge);
      EXPECT_EQ(sim.gpr[p.output], 222u);
   }

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 3. Loops & Nested Loops with Phi Web Coalescing
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, NestedLoopsPhiWebCoalescing)
{
   auto *entry = agx_apple9_block_create(&p);
   auto *outer_hdr = agx_apple9_block_create(&p);
   auto *inner_hdr = agx_apple9_block_create(&p);
   auto *inner_body = agx_apple9_block_create(&p);
   auto *outer_latch = agx_apple9_block_create(&p);
   auto *exit_blk = agx_apple9_block_create(&p);

   cfg_init(&p, {entry, outer_hdr, inner_hdr, inner_body, outer_latch, exit_blk});
   cfg_edge(entry, outer_hdr);
   cfg_edge(outer_hdr, inner_hdr);
   cfg_edge(outer_hdr, exit_blk);
   cfg_edge(inner_hdr, inner_body);
   cfg_edge(inner_hdr, outer_latch);
   cfg_edge(inner_body, inner_hdr);
   cfg_edge(outer_latch, outer_hdr);

   agx_apple9_block_begin(&p, entry);
   uint32_t zero = emit_imm(0);
   uint32_t one = emit_imm(1);
   uint32_t three = emit_imm(3);

   agx_apple9_block_begin(&p, outer_hdr);
   uint32_t acc_outer = agx_apple9_vir_emit_phi_canonical(&p, outer_hdr, 2);

   agx_apple9_block_begin(&p, inner_hdr);
   uint32_t acc_inner = agx_apple9_vir_emit_phi_canonical(&p, inner_hdr, 2);

   agx_apple9_block_begin(&p, inner_body);
   uint32_t next_inner = emit_iadd(acc_inner, one);

   agx_apple9_block_begin(&p, outer_latch);
   uint32_t next_outer = emit_iadd(acc_outer, three);
   uint32_t combined = emit_iadd(next_outer, acc_inner);

   agx_apple9_block_begin(&p, exit_blk);
   p.output = acc_outer;
   emit_halt();

   /* Wire canonical phi operands: pred 0 = entry/outer, pred 1 = latch/body. */
   agx_apple9_vir_reindex(&p);
   auto *phi_outer = agx_apple9_block_first_instr(outer_hdr);
   auto *phi_inner = agx_apple9_block_first_instr(inner_hdr);
   *agx_apple9_vir_src_ptr(phi_outer, 0) = zero;
   *agx_apple9_vir_src_ptr(phi_outer, 1) = combined;
   *agx_apple9_vir_src_ptr(phi_inner, 0) = acc_outer;
   *agx_apple9_vir_src_ptr(phi_inner, 1) = next_inner;

   auto opts = validated_opts();
   opts.cost_register_constraints = true;
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);

   /* Simulate 3 outer iterations x 4 inner iterations. */
   auto *pb_entry = p.blocks;
   auto *pb_ohdr = pb_entry->next;
   auto *pb_ihdr = pb_ohdr->next;
   auto *pb_ibody = pb_ihdr->next;
   auto *pb_olatch = pb_ibody->next;
   auto *pb_exit = pb_olatch->next;

   PhysVirSim sim;
   sim.exec_block(pb_entry);
   uint32_t ref_outer = 0;
   for (unsigned o = 0; o < 3; ++o) {
      sim.exec_block(pb_ohdr);
      uint32_t ref_inner = ref_outer;
      for (unsigned i = 0; i < 4; ++i) {
         sim.exec_block(pb_ihdr);
         sim.exec_block(pb_ibody);
         ref_inner += 1;
      }
      sim.exec_block(pb_ihdr);
      sim.exec_block(pb_olatch);
      ref_outer = (ref_outer + 3) + ref_inner;
   }
   sim.exec_block(pb_ohdr);
   sim.exec_block(pb_exit);
   EXPECT_EQ(sim.gpr[p.output], ref_outer);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 4. >10-Predecessor VIR_PHI (Sideband phi_srcs[])
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, WidePhiOverTenPredecessors)
{
   constexpr unsigned kNumArms = 14;
   std::vector<agx_apple9_block *> blocks;

   auto *entry = agx_apple9_block_create(&p);
   blocks.push_back(entry);

   /* Binary fanout tree so every block has <= 2 successors feeding kNumArms
    * leaf blocks that all merge into a single 14-pred merge block. */
   std::vector<agx_apple9_block *> dispatch;
   for (unsigned i = 0; i < kNumArms - 2; ++i) {
      auto *d = agx_apple9_block_create(&p);
      dispatch.push_back(d);
      blocks.push_back(d);
   }
   std::vector<agx_apple9_block *> arms;
   for (unsigned i = 0; i < kNumArms; ++i) {
      auto *a = agx_apple9_block_create(&p);
      arms.push_back(a);
      blocks.push_back(a);
   }
   auto *merge = agx_apple9_block_create(&p);
   blocks.push_back(merge);

   cfg_init(&p, blocks);
   /* Build a binary tree rooted at entry (node 0) with internal nodes
    * dispatch[0..11] and leaves arms[0..13]. */
   std::vector<agx_apple9_block *> tree;
   tree.push_back(entry);
   for (auto *d : dispatch)
      tree.push_back(d);
   for (auto *a : arms)
      tree.push_back(a);
   for (unsigned i = 0; i < kNumArms - 1; ++i) {
      cfg_edge(tree[i], tree[2 * i + 1]);
      cfg_edge(tree[i], tree[2 * i + 2]);
   }
   for (unsigned i = 0; i < kNumArms; ++i)
      cfg_edge(arms[i], merge);

   ASSERT_EQ(merge->num_cfg_preds, kNumArms);

   agx_apple9_block_begin(&p, entry);
   uint32_t seed = emit_imm(1000);

   for (auto *d : dispatch)
      agx_apple9_block_begin(&p, d);

   std::vector<uint32_t> arm_vals(kNumArms);
   for (unsigned i = 0; i < kNumArms; ++i) {
      agx_apple9_block_begin(&p, arms[i]);
      uint32_t c = emit_imm(i + 1);
      arm_vals[i] = emit_iadd(seed, c);
   }

   agx_apple9_block_begin(&p, merge);
   uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(&p, merge, kNumArms);
   ASSERT_NE(phi_v, AGX_APPLE9_VREG_INVALID);
   agx_apple9_vir_reindex(&p);
   auto *phi = agx_apple9_block_first_instr(merge);
   ASSERT_NE(phi->phi_srcs, nullptr); /* Confirm sideband array is used! */
   for (unsigned i = 0; i < kNumArms; ++i)
      *agx_apple9_vir_src_ptr(phi, i) = arm_vals[i];

   p.output = emit_iadd(phi_v, seed);
   emit_halt();

   auto opts = validated_opts();
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);

   /* Verify arm 0, arm 7, and arm 13 produce exact outputs. */
   std::vector<agx_apple9_block *> pblocks;
   for (auto *b = p.blocks; b; b = b->next)
      pblocks.push_back(b);
   ASSERT_EQ(pblocks.size(), blocks.size());

   unsigned first_arm_idx = 1 + (kNumArms - 2);
   for (unsigned arm : {0u, 7u, 13u}) {
      PhysVirSim sim;
      sim.exec_block(pblocks[0]);
      sim.exec_block(pblocks[first_arm_idx + arm]);
      sim.exec_block(pblocks.back());
      EXPECT_EQ(sim.gpr[p.output], 2000u + arm + 1u);
   }

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 5. Phi Swap Cycles (2-Cycle and 3-Cycle Rotation in a Loop)
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, PhiSwapCyclesRotationInLoop)
{
   auto *entry = agx_apple9_block_create(&p);
   auto *header = agx_apple9_block_create(&p);
   auto *body = agx_apple9_block_create(&p);
   auto *exit_blk = agx_apple9_block_create(&p);

   cfg_init(&p, {entry, header, body, exit_blk});
   cfg_edge(entry, header);
   cfg_edge(header, body);
   cfg_edge(header, exit_blk);
   cfg_edge(body, header);

   agx_apple9_block_begin(&p, entry);
   uint32_t init_a = emit_imm(10);
   uint32_t init_b = emit_imm(20);
   uint32_t init_c = emit_imm(30);
   uint32_t init_x = emit_imm(100);
   uint32_t init_y = emit_imm(200);

   agx_apple9_block_begin(&p, header);
   uint32_t pa = agx_apple9_vir_emit_phi_canonical(&p, header, 2);
   uint32_t pb = agx_apple9_vir_emit_phi_canonical(&p, header, 2);
   uint32_t pc = agx_apple9_vir_emit_phi_canonical(&p, header, 2);
   uint32_t px = agx_apple9_vir_emit_phi_canonical(&p, header, 2);
   uint32_t py = agx_apple9_vir_emit_phi_canonical(&p, header, 2);

   agx_apple9_vir_reindex(&p);
   auto *ins = agx_apple9_block_first_instr(header);
   /* 3-cycle: a <- b <- c <- a */
   *agx_apple9_vir_src_ptr(ins, 0) = init_a;
   *agx_apple9_vir_src_ptr(ins, 1) = pb;
   ins = agx_apple9_instr_next(ins);
   *agx_apple9_vir_src_ptr(ins, 0) = init_b;
   *agx_apple9_vir_src_ptr(ins, 1) = pc;
   ins = agx_apple9_instr_next(ins);
   *agx_apple9_vir_src_ptr(ins, 0) = init_c;
   *agx_apple9_vir_src_ptr(ins, 1) = pa;
   ins = agx_apple9_instr_next(ins);
   /* 2-cycle: x <-> y */
   *agx_apple9_vir_src_ptr(ins, 0) = init_x;
   *agx_apple9_vir_src_ptr(ins, 1) = py;
   ins = agx_apple9_instr_next(ins);
   *agx_apple9_vir_src_ptr(ins, 0) = init_y;
   *agx_apple9_vir_src_ptr(ins, 1) = px;

   agx_apple9_block_begin(&p, body);
   agx_apple9_block_begin(&p, exit_blk);
   uint32_t t1 = emit_iadd(pa, px);
   uint32_t t2 = emit_isub(pb, pc);
   p.output = emit_iadd(t1, t2);
   emit_halt();

   /* Constrain to exactly 5 GPRs so 0 free temp GPRs exist during the cycles,
    * forcing VIR_SWAP -> 3x IXOR lowering! */
   auto opts = validated_opts();
   opts.max_gprs = 5;
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);
   EXPECT_LT(res.max_phys_gpr, 5u);

   /* Verify XOR swaps were emitted in body. */
   auto *pb_entry = p.blocks;
   auto *pb_hdr = pb_entry->next;
   auto *pb_body = pb_hdr->next;
   auto *pb_exit = pb_body->next;

   unsigned ixor_count = 0;
   for (auto *I = agx_apple9_block_first_instr(pb_body); I;
        I = agx_apple9_instr_next(I)) {
      if (I->op == AGX_APPLE9_VIR_IXOR)
         ++ixor_count;
   }
   EXPECT_GE(ixor_count, 6u);

   /* After 1 iteration: (a,b,c)=(20,30,10), (x,y)=(200,100)
    * output = (20 + 200) + (30 - 10) = 240. */
   PhysVirSim sim;
   sim.exec_block(pb_entry);
   sim.exec_block(pb_hdr);
   sim.exec_block(pb_body);
   sim.exec_block(pb_hdr);
   sim.exec_block(pb_exit);
   EXPECT_EQ(sim.gpr[p.output], 240u);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 6. Hard-Low r0..r15 & r0..r63 Constrained Instructions + Scarcity Costing
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, HardLowR0ToR15AndR0ToR63ConstraintsUnderPressure)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   /* Create 24 live values to saturate r0..r15 and force relocation/scarcity. */
   std::vector<uint32_t> vals;
   for (unsigned i = 0; i < 24; ++i)
      vals.push_back(emit_imm(i + 1, AGX_APPLE9_ENC_MOV_IMM32));

   /* Emit a hard-low compact immediate (r0..r15) while all 24 values are live! */
   uint32_t low_imm = emit_imm(77, AGX_APPLE9_ENC_MOV_IMM_COMPACT);

   /* Also emit a FLOAT2_BASE (r0..r63 dest and srcs) and verify range. */
   uint32_t fsrcs[2] = {vals[0], vals[1]};
   uint32_t fadd_v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
                                         AGX_APPLE9_ENC_FLOAT2_BASE, fsrcs, 2, 0);

   uint32_t acc = emit_iadd(low_imm, fadd_v);
   for (uint32_t v : vals)
      acc = emit_iadd(acc, v);

   p.output = acc;
   emit_halt();

   cfg_init(&p, {b0});

   /* Run with keep_ssa_ir + validate first on a clone to inspect exact GPRs. */
   agx_apple9_vir_program clone;
   ASSERT_TRUE(agx_apple9_vir_clone(&clone, &p));

   auto opts = validated_opts();
   opts.cost_register_constraints = true;
   opts.keep_ssa_ir = true;
   opts.skip_schedule = true;
   apple9_ra_assign_result ssa_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&clone, &opts, &ssa_res))
      << (ssa_res.error ? ssa_res.error : "");
   EXPECT_LT(clone.phys[low_imm], 16u);
   EXPECT_LT(clone.phys[fadd_v], 64u);
   agx_apple9_vir_finish(&clone);

   /* Full physical allocation + binary packing check. */
   opts.keep_ssa_ir = false;
   apple9_ra_assign_result phys_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &phys_res))
      << (phys_res.error ? phys_res.error : "");
   EXPECT_FALSE(phys_res.has_spills);
   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 7. Aligned vec2/vec3/vec4 Tuples, VIR_SPLIT, Reuse Cache & retain_mask
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, AlignedTuplesCollectSplitReuseCacheAndRetainMask)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = emit_imm(0);

   /* Native vec4 device load (4 contiguous GPRs). */
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };
   uint32_t v4 = agx_apple9_vir_emit_device_load_vector(&p, 0, idx, 4, &contract);
   ASSERT_NE(v4, AGX_APPLE9_VREG_INVALID);

   /* Native vec2 device load (2 contiguous GPRs) + VIR_SPLIT (killed -> 0 copies). */
   uint32_t wide = agx_apple9_vir_emit_device_load_vector(&p, 1, idx, 2, &contract);
   ASSERT_NE(wide, AGX_APPLE9_VREG_INVALID);

   /* Explicit VIR_SPLIT of wide (killed source -> zero-copy coalescing). */
   uint32_t split_wide = agx_apple9_vir_emit_split(&p, wide, 2);
   ASSERT_NE(split_wide, AGX_APPLE9_VREG_INVALID);

   /* Pointer load reading leading 2 lanes (v4+0, v4+1) of the vec4 load ->
    * should reuse tuple directly AND set retain_mask on those lanes! */
   uint32_t ptr_load = agx_apple9_vir_emit_device_load(&p, 0, idx, &contract);
   ASSERT_NE(ptr_load, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_set_load_address(&p, ptr_load, v4));

   /* Two consecutive indirect loads sharing the same non-contiguous scalar
    * address pair (split_wide+0, ptr_load) -> exercises 16-entry intra-block
    * read-only tuple reuse cache (should emit only 1 VIR_COLLECT for both!). */
   uint32_t ind_srcs[3] = {idx, split_wide + 0, ptr_load};
   uint32_t ind_load1 = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_DEVICE_LOAD, AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT,
      ind_srcs, 3, 0);
   ASSERT_NE(ind_load1, AGX_APPLE9_VREG_INVALID);
   p.instructions[p.instruction_count - 1]->memory_bits = 32;
   p.instructions[p.instruction_count - 1]->memory_components = 1;
   p.instructions[p.instruction_count - 1]->memory_global = true;
   p.instructions[p.instruction_count - 1]->producer_scoreboard_slot =
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
   p.instructions[p.instruction_count - 1]->device_load_index_kind =
      AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR;

   uint32_t ind_load2 = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_DEVICE_LOAD, AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT,
      ind_srcs, 3, 0);
   ASSERT_NE(ind_load2, AGX_APPLE9_VREG_INVALID);
   p.instructions[p.instruction_count - 1]->memory_bits = 32;
   p.instructions[p.instruction_count - 1]->memory_components = 1;
   p.instructions[p.instruction_count - 1]->memory_global = true;
   p.instructions[p.instruction_count - 1]->producer_scoreboard_slot =
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
   p.instructions[p.instruction_count - 1]->device_load_index_kind =
      AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR;

   /* Consume split_wide+1, ind_load1, ind_load2, v4+2, v4+3 in a vec3 store. */
   uint32_t sum = emit_iadd(ind_load1, ind_load2);
   uint32_t sum2 = emit_iadd(sum, split_wide + 1);
   uint32_t vec3_data[3] = {sum2, v4 + 2, v4 + 3};
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 1, idx, vec3_data, 3, 32));

   emit_halt();
   cfg_init(&p, {b0});

   /* Verify SSA assignment + contiguous tuples + retain_mask + reuse cache. */
   agx_apple9_vir_program clone;
   ASSERT_TRUE(agx_apple9_vir_clone(&clone, &p));
   auto opts = validated_opts();
   opts.keep_ssa_ir = true;
   apple9_ra_assign_result ssa_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&clone, &opts, &ssa_res))
      << (ssa_res.error ? ssa_res.error : "");
   EXPECT_EQ(clone.phys[wide + 1], clone.phys[wide] + 1u);
   EXPECT_EQ(clone.phys[split_wide], clone.phys[wide]);
   EXPECT_EQ(clone.phys[split_wide + 1], clone.phys[wide + 1]);
   EXPECT_EQ(clone.phys[v4 + 1], clone.phys[v4] + 1u);
   EXPECT_EQ(clone.phys[v4 + 2], clone.phys[v4] + 2u);
   EXPECT_EQ(clone.phys[v4 + 3], clone.phys[v4] + 3u);

   /* Confirm 1 VIR_COLLECT for vec3 store + 1 shared VIR_COLLECT for BOTH
    * indirect loads (2 total instead of 3!), and retain_mask set on ptr_load! */
   unsigned collects = 0;
   bool saw_retain = false;
   for (unsigned i = 0; i < clone.instruction_count; ++i) {
      if (clone.instructions[i]->op == AGX_APPLE9_VIR_COLLECT)
         ++collects;
      if (clone.instructions[i]->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT &&
          (clone.instructions[i]->retain_mask & 0x6u) == 0x6u)
         saw_retain = true;
   }
   EXPECT_EQ(collects, 2u);
   EXPECT_TRUE(saw_retain);
   agx_apple9_vir_finish(&clone);

   /* Full physical allocation + validation + binary packing. */
   opts.keep_ssa_ir = false;
   apple9_ra_assign_result phys_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &phys_res))
      << (phys_res.error ? phys_res.error : "");
   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 8. AGX_APPLE9_OPERAND_CLOBBER Handling (Killed Elided vs Live Preserved)
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, ClobberOperandKilledElidedVsLivePreserved)
{
   for (bool keep_live : {false, true}) {
      agx_apple9_vir_program prog;
      agx_apple9_vir_init(&prog);
      auto *b0 = agx_apple9_block_create(&prog);
      agx_apple9_block_begin(&prog, b0);

      uint32_t prim = agx_apple9_vir_emit_entry_reg(&prog, 8);
      uint32_t aux = agx_apple9_vir_emit(&prog, AGX_APPLE9_VIR_IMM,
                                         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0,
                                         0);
      /* MESH_CONTROL clobbers src0 (prim). */
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&prog, prim, aux, false));

      if (keep_live) {
         uint32_t s[2] = {prim, aux};
         prog.output = agx_apple9_vir_emit(&prog, AGX_APPLE9_VIR_IADD,
                                           AGX_APPLE9_ENC_INT_ADD_EXTENDED, s,
                                           2, 0);
      }
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &prog, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));
      cfg_init(&prog, {b0});

      auto opts = validated_opts();
      apple9_ra_assign_result res = {};
      ASSERT_TRUE(apple9_ra_allocate_native(&prog, &opts, &res))
         << (res.error ? res.error : "");

      unsigned mesh_src0 = AGX_APPLE9_PHYS_INVALID;
      unsigned add_src0 = AGX_APPLE9_PHYS_INVALID;
      for (unsigned i = 0; i < prog.instruction_count; ++i) {
         const auto *ins = prog.instructions[i];
         if (ins->op == AGX_APPLE9_VIR_MESH_CONTROL)
            mesh_src0 = ins->src[0];
         if (ins->op == AGX_APPLE9_VIR_IADD)
            add_src0 = ins->src[0];
      }
      ASSERT_NE(mesh_src0, AGX_APPLE9_PHYS_INVALID);

      if (!keep_live) {
         /* Killed clobber source coalesces -> 0 copy instructions! */
         EXPECT_EQ(res.num_copies_emitted, 0u);
         EXPECT_EQ(mesh_src0, 8u);
      } else {
         /* Still-live clobber source preserves original via 1 IOR copy! */
         EXPECT_GE(res.num_copies_emitted, 1u);
         EXPECT_EQ(add_src0, 8u);
         EXPECT_NE(mesh_src0, add_src0);
      }

      EXPECT_GT(verify_binary_emission(&prog), 0u);
      agx_apple9_vir_finish(&prog);
   }
}

/* =========================================================================
 * 9. High-Pressure Spilling (>96 Live Values + Tight Budget)
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, HighPressureSpillingFullAndValidated)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   /* Build 108 simultaneously live non-rematerializable values (exceeding 96
    * Apple9 GPRs!). Use an ENTRY_REG input + IADD chain so uniform-remat
    * cannot eliminate them, forcing real RA_MEM spills. */
   uint32_t seed = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t one = emit_imm(1);

   constexpr unsigned kCount = 108;
   std::vector<uint32_t> live(kCount);
   live[0] = emit_iadd(seed, one);
   for (unsigned i = 1; i < kCount; ++i)
      live[i] = emit_iadd(live[i - 1], one);

   /* Consume all 108 values in reverse order so peak pressure >= 108 GPRs. */
   uint32_t acc = live[0];
   for (unsigned i = 1; i < kCount; ++i)
      acc = emit_iadd(acc, live[i]);

   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, acc, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));
   cfg_init(&p, {b0});

   auto opts = validated_opts();
   opts.cost_register_constraints = true;
   opts.disable_uniform_remat = true;
   opts.skip_schedule = true; /* Keep all 108 values simultaneously live! */
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");

   EXPECT_TRUE(res.has_spills);
   EXPECT_EQ(res.reserved_gprs, APPLE9_RA_SPILL_RESERVED_GPRS);
   EXPECT_GT(res.spill_slots, 0u);
   EXPECT_GT(res.num_spills_emitted, 0u);
   EXPECT_EQ(res.scratch_size % 16u, 0u);
   EXPECT_LE(res.scratch_size, AGX_APPLE9_MAX_SCRATCH_BYTES);
   EXPECT_LT(res.max_phys_gpr, AGX_APPLE9_GPR_COUNT);

   /* Verify exact runtime arithmetic with seed = 5! */
   PhysVirSim sim;
   sim.gpr[8] = 5;
   sim.exec_block(p.blocks);
   uint32_t expected = 0;
   for (unsigned i = 0; i < kCount; ++i)
      expected += 5u + (i + 1u);
   EXPECT_EQ(sim.gpr[8], expected);

   /* Verify 0 agx_apple9_validate_vir_allocation errors + full binary pack! */
   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 10. Returning Device Atomic 16B Frame Base Offset + Spilling
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, ReturningDeviceAtomic16BFrameBaseOffsetWithSpilling)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t seed = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t idx = emit_imm(0);
   uint32_t one = emit_imm(1);

   uint32_t atomic_old = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &p, 0, idx, &one, 1, AGX_APPLE9_ATOMIC_ADD, /*discard_result=*/false,
      &atomic_old));
   ASSERT_NE(atomic_old, AGX_APPLE9_VREG_INVALID);

   /* Force spilling with tight max_gprs = 20 while atomic_old is live. */
   constexpr unsigned kCount = 24;
   std::vector<uint32_t> vals(kCount);
   vals[0] = emit_iadd(seed, atomic_old);
   for (unsigned i = 1; i < kCount; ++i)
      vals[i] = emit_iadd(vals[i - 1], one);

   uint32_t acc = atomic_old;
   for (unsigned i = 0; i < kCount; ++i)
      acc = emit_iadd(acc, vals[i]);

   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, acc, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));
   cfg_init(&p, {b0});

   auto opts = validated_opts();
   opts.max_gprs = 20;
   opts.disable_uniform_remat = true;
   opts.skip_schedule = true; /* Keep all 24 values live across atomic_old! */
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");

   EXPECT_TRUE(res.has_spills);
   EXPECT_EQ(res.reserved_gprs, APPLE9_RA_SPILL_RESERVED_GPRS);
   EXPECT_GE(res.scratch_size, 32u); /* 16B atomic base + >=16B spill region */

   /* Confirm EVERY spill store/load starts at word >= 4 (16B atomic base)
    * AND DEVICE_ATOMIC is immediately followed by DEVICE_ATOMIC_RESULT! */
   bool saw_spill = false;
   bool saw_atomic_pair = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_SPILL_STORE ||
          ins->op == AGX_APPLE9_VIR_SPILL_LOAD) {
         saw_spill = true;
         EXPECT_GE(ins->immediate, 4u)
            << "spill overwrote 16B returning-device-atomic reservation!";
      }
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         ASSERT_LT(i + 1, p.instruction_count);
         const auto *next = p.instructions[i + 1];
         EXPECT_EQ(next->op, AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
         EXPECT_EQ(next->src[0], ins->dest);
         EXPECT_EQ(next->producer_scoreboard_slot,
                   ins->producer_scoreboard_slot);
         EXPECT_GE(ins->producer_scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         EXPECT_LE(ins->producer_scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_6);
         saw_atomic_pair = true;
      }
   }
   EXPECT_TRUE(saw_spill);
   EXPECT_TRUE(saw_atomic_pair);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 11. ABI_RETURN Export Cycle + ENTRY_REG Preloads
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, AbiEntryAndExportSwapCycle)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t in0 = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t in1 = agx_apple9_vir_emit_entry_reg(&p, 9);
   /* Cross-export in0 -> r9, in1 -> r8 (2-cycle at ABI_RETURN!). */
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, in0, 9));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, in1, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));
   cfg_init(&p, {b0});

   auto opts = validated_opts();
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);

   PhysVirSim sim;
   sim.gpr[8] = 0x11112222u;
   sim.gpr[9] = 0x33334444u;
   sim.exec_block(p.blocks);
   EXPECT_EQ(sim.gpr[8], 0x33334444u);
   EXPECT_EQ(sim.gpr[9], 0x11112222u);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 12. Killed Spill SAVE Skips COPY_REG Bridge
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, KilledSpillSaveElidesCopyReg)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t seed = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t one = emit_imm(1);

   constexpr unsigned kCount = 22;
   std::vector<uint32_t> vals(kCount);
   vals[0] = emit_iadd(seed, one);
   for (unsigned i = 1; i < kCount; ++i)
      vals[i] = emit_iadd(vals[i - 1], one);

   uint32_t acc = vals[0];
   for (unsigned i = 1; i < kCount; ++i)
      acc = emit_iadd(acc, vals[i]);

   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, acc, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));
   cfg_init(&p, {b0});

   auto opts = validated_opts();
   opts.max_gprs = 16;
   opts.disable_uniform_remat = true;
   opts.skip_schedule = true; /* Keep all 22 values simultaneously live! */
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_TRUE(res.has_spills);

   /* Verify at least one SPILL_STORE consumed its source GPR directly
    * (src[0] != COPY_REG=1) when the source died at SAVE! */
   bool saw_direct_save = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_SPILL_STORE && ins->src[0] != 1u)
         saw_direct_save = true;
   }
   EXPECT_TRUE(saw_direct_save);

   PhysVirSim sim;
   sim.gpr[8] = 10;
   sim.exec_block(p.blocks);
   uint32_t expected = 0;
   for (unsigned i = 0; i < kCount; ++i)
      expected += 10u + (i + 1u);
   EXPECT_EQ(sim.gpr[8], expected);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 13. Unified Pre-Color / Move / Phi Coalescing + General Immediate Lowering
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, PrecolorAndMoveCoalescingWithImmediateCopyLowering)
{
   auto *entry = agx_apple9_block_create(&p);
   auto *left = agx_apple9_block_create(&p);
   auto *right = agx_apple9_block_create(&p);
   auto *merge = agx_apple9_block_create(&p);

   agx_apple9_block_begin(&p, entry);
   uint32_t in0 = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t zero = emit_imm(0);
   uint32_t c10 = emit_imm(10);

   agx_apple9_block_begin(&p, left);
   uint32_t l_val = emit_iadd(in0, c10);

   agx_apple9_block_begin(&p, right);
   uint32_t c20 = emit_imm(20);
   uint32_t r_val = emit_iadd(in0, c20);

   agx_apple9_block_begin(&p, merge);
   uint32_t phi_v = agx_apple9_vir_emit_phi(&p, merge);
   /* Exercise general VIR_MOV coalescing alongside VIR_PHI coalescing. */
   uint32_t phi_mov = agx_apple9_vir_emit_mov(&p, phi_v);
   uint32_t col1 = emit_iadd(phi_mov, c10);
   uint32_t col1_mov = agx_apple9_vir_emit_mov(&p, col1);

   /* Export 6 fixed-register outputs: r8 (phi_mov), r9 (col1_mov), r10..r13
    * (shared immediate zero). */
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, phi_mov, 8));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, col1_mov, 9));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, zero, 10));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, zero, 11));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, zero, 12));
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, zero, 13));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));

   p.current_block = left;
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi_v, l_val));
   p.current_block = right;
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi_v, r_val));

   cfg_init(&p, {entry, left, right, merge});
   cfg_edge(entry, left);
   cfg_edge(entry, right);
   cfg_edge(left, merge);
   cfg_edge(right, merge);

   auto opts = validated_opts();
   opts.cost_register_constraints = true;
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_FALSE(res.has_spills);

   /* Unified pre-color + VIR_MOV + VIR_PHI coalescing places the phi/mov
    * webs directly into r8/r9 and lowers constant copies via 4B MOV_IMM_COMPACT
    * with ZERO IOR copies! */
   EXPECT_EQ(res.num_copies_emitted, 0u);

   unsigned compact_zeros = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_IMM &&
          ins->encoding == AGX_APPLE9_ENC_MOV_IMM_COMPACT &&
          ins->immediate == 0 && ins->dest >= 10 && ins->dest <= 13)
         ++compact_zeros;
   }
   EXPECT_EQ(compact_zeros, 4u);

   auto *pb_entry = p.blocks;
   auto *pb_left = pb_entry->next;
   auto *pb_right = pb_left->next;
   auto *pb_merge = pb_right->next;
   {
      PhysVirSim sim;
      sim.gpr[8] = 5;
      sim.exec_block(pb_entry);
      sim.exec_block(pb_left);
      sim.exec_block(pb_merge);
      EXPECT_EQ(sim.gpr[8], 15u);
      EXPECT_EQ(sim.gpr[9], 25u);
      for (unsigned r = 10; r <= 13; ++r)
         EXPECT_EQ(sim.gpr[r], 0u);
   }
   {
      PhysVirSim sim;
      sim.gpr[8] = 5;
      sim.exec_block(pb_entry);
      sim.exec_block(pb_right);
      sim.exec_block(pb_merge);
      EXPECT_EQ(sim.gpr[8], 25u);
      EXPECT_EQ(sim.gpr[9], 35u);
      for (unsigned r = 10; r <= 13; ++r)
         EXPECT_EQ(sim.gpr[r], 0u);
   }

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 14. Cross-Block Loop Spilling + Loop-Invariant Spill Hoisting
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, CrossBlockLoopSpillingWithSimulation)
{
   auto *entry = agx_apple9_block_create(&p);
   auto *header = agx_apple9_block_create(&p);
   auto *body = agx_apple9_block_create(&p);
   auto *exit_blk = agx_apple9_block_create(&p);

   cfg_init(&p, {entry, header, body, exit_blk});
   cfg_edge(entry, header);
   cfg_edge(header, body);
   cfg_edge(header, exit_blk);
   cfg_edge(body, header);

   agx_apple9_block_begin(&p, entry);
   uint32_t seed = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t one = emit_imm(1);

   /* 18 live-across-loop values defined in entry. */
   constexpr unsigned kLive = 18;
   std::vector<uint32_t> inv(kLive);
   inv[0] = emit_iadd(seed, one);
   for (unsigned i = 1; i < kLive; ++i)
      inv[i] = emit_iadd(inv[i - 1], one);

   agx_apple9_block_begin(&p, header);
   uint32_t acc_phi = agx_apple9_vir_emit_phi_canonical(&p, header, 2);

   agx_apple9_block_begin(&p, body);
   uint32_t next_acc = emit_iadd(acc_phi, inv[0]);

   agx_apple9_block_begin(&p, exit_blk);
   uint32_t final_sum = acc_phi;
   for (unsigned i = 0; i < kLive; ++i)
      final_sum = emit_iadd(final_sum, inv[i]);
   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, final_sum, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));

   agx_apple9_vir_reindex(&p);
   auto *phi = agx_apple9_block_first_instr(header);
   *agx_apple9_vir_src_ptr(phi, 0) = seed;
   *agx_apple9_vir_src_ptr(phi, 1) = next_acc;

   auto opts = validated_opts();
   opts.max_gprs = 16;
   opts.disable_uniform_remat = true;
   opts.skip_schedule = true;
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_TRUE(res.has_spills);
   EXPECT_EQ(res.reserved_gprs, APPLE9_RA_SPILL_RESERVED_GPRS);
   EXPECT_LT(res.max_phys_gpr, 16u);

   auto *pb_entry = p.blocks;
   auto *pb_hdr = pb_entry->next;
   auto *pb_body = pb_hdr->next;
   auto *pb_exit = pb_body->next;

   PhysVirSim sim;
   sim.gpr[8] = 3;
   sim.exec_block(pb_entry);
   for (unsigned it = 0; it < 4; ++it) {
      sim.exec_block(pb_hdr);
      sim.exec_block(pb_body);
   }
   sim.exec_block(pb_hdr);
   sim.exec_block(pb_exit);

   /* seed = 3, inv[i] = 4 + i; acc after 4 iters = 3 + 4 * 4 = 19;
    * final_sum = 19 + sum_{i=0..17} (4 + i) = 19 + 18*4 + 153 = 244. */
   uint32_t expected = 3u + 4u * 4u;
   for (unsigned i = 0; i < kLive; ++i)
      expected += 4u + i;
   EXPECT_EQ(sim.gpr[8], expected);

   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 15. Force-Spill + Latency Scheduling + Constraint Costing
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, ForceSpillWithLatencyScheduleAndCosting)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   uint32_t seed = agx_apple9_vir_emit_entry_reg(&p, 8);
   uint32_t idx = emit_imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };

   std::vector<uint32_t> loads;
   for (unsigned i = 0; i < 16; ++i) {
      uint32_t ld = agx_apple9_vir_emit_device_load(&p, i % 4, idx, &contract);
      ASSERT_NE(ld, AGX_APPLE9_VREG_INVALID);
      loads.push_back(ld);
   }

   uint32_t acc = seed;
   for (uint32_t ld : loads)
      acc = emit_iadd(acc, ld);

   ASSERT_TRUE(agx_apple9_vir_add_abi_output(&p, acc, 8));
   ASSERT_TRUE(agx_apple9_vir_emit_abi_return(&p));
   cfg_init(&p, {b0});

   auto opts = validated_opts();
   opts.force_spill = true;
   opts.cost_register_constraints = true;
   opts.latency_schedule = true;
   opts.pressure_slack = 4;
   apple9_ra_assign_result res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &res))
      << (res.error ? res.error : "");
   EXPECT_GT(verify_binary_emission(&p), 0u);
}

/* =========================================================================
 * 16. Even-GPR Alignment (alignment_halves = 4: UNPACK_NORM & IMUL_WIDE)
 * ========================================================================= */

TEST_F(Apple9RaAssignTest, EvenGprAlignedUnpackNormAndImulWide)
{
   auto *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);

   /* Occupy r0 so next free GPR would be odd r1 unless even alignment is enforced! */
   uint32_t a = emit_imm(0x12345678u);
   uint32_t b = emit_imm(0x9abcdef0u);
   uint32_t unp = agx_apple9_vir_emit_unpack_norm(&p, a, AGX_APPLE9_UNPACK_UNORM16);
   ASSERT_NE(unp, AGX_APPLE9_VREG_INVALID);

   uint32_t msrcs[2] = {a, b};
   uint32_t wide = agx_apple9_vir_emit_mul_wide(&p, msrcs, false);
   ASSERT_NE(wide, AGX_APPLE9_VREG_INVALID);

   uint32_t s1 = emit_iadd(unp + 0, unp + 1);
   uint32_t s2 = emit_iadd(wide + 0, wide + 1);
   p.output = emit_iadd(s1, s2);
   emit_halt();
   cfg_init(&p, {b0});

   agx_apple9_vir_program clone;
   ASSERT_TRUE(agx_apple9_vir_clone(&clone, &p));
   auto opts = validated_opts();
   opts.keep_ssa_ir = true;
   apple9_ra_assign_result ssa_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&clone, &opts, &ssa_res))
      << (ssa_res.error ? ssa_res.error : "");
   EXPECT_EQ(clone.phys[unp] % 2u, 0u);
   EXPECT_EQ(clone.phys[unp + 1], clone.phys[unp] + 1u);
   EXPECT_EQ(clone.phys[wide] % 2u, 0u);
   EXPECT_EQ(clone.phys[wide + 1], clone.phys[wide] + 1u);
   agx_apple9_vir_finish(&clone);

   opts.keep_ssa_ir = false;
   apple9_ra_assign_result phys_res = {};
   ASSERT_TRUE(apple9_ra_allocate_native(&p, &opts, &phys_res))
      << (phys_res.error ? phys_res.error : "");

   /* Pack every physical instruction cleanly via agx_apple9_pack_vir_instruction! */
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      agx_apple9_packed_instruction packed = {};
      const char *reason = nullptr;
      EXPECT_TRUE(agx_apple9_pack_vir_instruction(p.instructions[i], p.phys,
                                                  &packed, &reason))
         << "pack failed at insn " << i << ": " << (reason ? reason : "");
      EXPECT_GT(packed.length, 0u);
   }
}
