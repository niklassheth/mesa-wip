/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_pcopy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct PcopySim {
   std::array<uint32_t, AGX_APPLE9_GPR_COUNT> gpr;
   std::vector<uint32_t> mem;
   std::vector<apple9_ra_pcopy_op> ops;
   unsigned mov_imm_max_gpr = 64;

   explicit PcopySim(unsigned mem_slots = 64) : mem(mem_slots)
   {
      for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r)
         gpr[r] = 0xA9000000u | r;
      for (unsigned m = 0; m < mem_slots; ++m)
         mem[m] = 0xBEEF0000u | m;
   }

   static bool emit_cb(void *ctx, const apple9_ra_pcopy_op *op)
   {
      auto *self = static_cast<PcopySim *>(ctx);
      self->ops.push_back(*op);

      switch (op->kind) {
      case APPLE9_RA_PCOPY_MOV: {
         EXPECT_FALSE(op->dst.kind == APPLE9_RA_LOC_MEM &&
                      op->src.kind == APPLE9_RA_LOC_MEM);
         EXPECT_FALSE(op->dst.kind == op->src.kind &&
                      op->dst.value == op->src.value);
         uint32_t v = 0;
         if (op->src.kind == APPLE9_RA_LOC_GPR) {
            EXPECT_LT(op->src.value, AGX_APPLE9_GPR_COUNT);
            v = self->gpr[op->src.value];
         } else {
            EXPECT_EQ(op->src.kind, APPLE9_RA_LOC_MEM);
            EXPECT_LT(op->src.value, self->mem.size());
            v = self->mem[op->src.value];
         }
         if (op->dst.kind == APPLE9_RA_LOC_GPR) {
            EXPECT_LT(op->dst.value, AGX_APPLE9_GPR_COUNT);
            self->gpr[op->dst.value] = v;
         } else {
            EXPECT_EQ(op->dst.kind, APPLE9_RA_LOC_MEM);
            EXPECT_LT(op->dst.value, self->mem.size());
            self->mem[op->dst.value] = v;
         }
         return true;
      }

      case APPLE9_RA_PCOPY_MOV_IMM:
         EXPECT_EQ(op->dst.kind, APPLE9_RA_LOC_GPR);
         EXPECT_EQ(op->src.kind, APPLE9_RA_LOC_IMM);
         EXPECT_LT(op->dst.value, self->mov_imm_max_gpr);
         self->gpr[op->dst.value] = op->src.value;
         return true;

      case APPLE9_RA_PCOPY_SWAP:
         EXPECT_EQ(op->dst.kind, APPLE9_RA_LOC_GPR);
         EXPECT_EQ(op->src.kind, APPLE9_RA_LOC_GPR);
         EXPECT_LT(op->dst.value, AGX_APPLE9_GPR_COUNT);
         EXPECT_LT(op->src.value, AGX_APPLE9_GPR_COUNT);
         EXPECT_NE(op->dst.value, op->src.value);
         std::swap(self->gpr[op->dst.value], self->gpr[op->src.value]);
         return true;
      }
      return false;
   }
};

static apple9_ra_pcopy_opts
make_opts(uint64_t live_lo = 0, uint64_t live_hi = 0,
          uint8_t scratch0 = AGX_APPLE9_PHYS_INVALID,
          uint8_t scratch1 = AGX_APPLE9_PHYS_INVALID)
{
   apple9_ra_pcopy_opts opts = {};
   opts.live_gprs[0] = live_lo;
   opts.live_gprs[1] = live_hi;
   opts.scratch_gpr = scratch0;
   opts.scratch_gpr2 = scratch1;
   opts.max_temp_gpr = AGX_APPLE9_GPR_COUNT;
   opts.mov_imm_max_gpr = 64;
   return opts;
}

static apple9_ra_pcopy_opts
make_zero_free_opts(uint8_t scratch0 = AGX_APPLE9_PHYS_INVALID,
                    uint8_t scratch1 = AGX_APPLE9_PHYS_INVALID)
{
   return make_opts(~0ull, BITFIELD64_MASK(AGX_APPLE9_GPR_COUNT - 64), scratch0,
                    scratch1);
}

static void
verify_parallel_copy(const std::vector<apple9_ra_copy> &copies,
                     const apple9_ra_pcopy_opts &opts,
                     unsigned mem_slots = 64,
                     std::vector<apple9_ra_pcopy_op> *ops_out = nullptr)
{
   PcopySim sim(mem_slots);
   sim.mov_imm_max_gpr = opts.mov_imm_max_gpr ? opts.mov_imm_max_gpr : 64;

   const auto init_gpr = sim.gpr;
   const auto init_mem = sim.mem;

   ASSERT_TRUE(apple9_ra_lower_parallel_copies(
      copies.data(), static_cast<unsigned>(copies.size()), &opts,
      PcopySim::emit_cb, &sim));

   std::array<bool, AGX_APPLE9_GPR_COUNT> dst_gpr = {};
   std::vector<bool> dst_mem(mem_slots, false);

   auto expected_gpr = init_gpr;
   auto expected_mem = init_mem;

   for (const auto &c : copies) {
      unsigned count = c.count ? c.count : 1;
      for (unsigned w = 0; w < count; ++w) {
         if (c.dst.kind == APPLE9_RA_LOC_GPR)
            dst_gpr[c.dst.value + w] = true;
         else
            dst_mem[c.dst.value + w] = true;

         if (c.src.kind == APPLE9_RA_LOC_UNDEF)
            continue;

         uint32_t val = 0;
         if (c.src.kind == APPLE9_RA_LOC_GPR)
            val = init_gpr[c.src.value + w];
         else if (c.src.kind == APPLE9_RA_LOC_MEM)
            val = init_mem[c.src.value + w];
         else if (c.src.kind == APPLE9_RA_LOC_IMM)
            val = w == 0 ? c.src.value : 0u;

         if (c.dst.kind == APPLE9_RA_LOC_GPR)
            expected_gpr[c.dst.value + w] = val;
         else
            expected_mem[c.dst.value + w] = val;
      }
   }

   for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r) {
      bool is_live = ((opts.live_gprs[r / 64] >> (r % 64)) & 1ull) != 0;
      bool is_scratch = (r == opts.scratch_gpr || r == opts.scratch_gpr2);
      if (dst_gpr[r]) {
         EXPECT_EQ(sim.gpr[r], expected_gpr[r]) << "GPR r" << r << " mismatch";
      } else if (is_live && !is_scratch) {
         EXPECT_EQ(sim.gpr[r], init_gpr[r])
            << "Live non-dest GPR r" << r << " was clobbered";
      }
   }

   for (unsigned m = 0; m < mem_slots; ++m) {
      EXPECT_EQ(sim.mem[m], expected_mem[m])
         << "MEM m" << m << (dst_mem[m] ? " dst mismatch" : " clobbered");
   }

   if (ops_out)
      *ops_out = std::move(sim.ops);
}

/* Helper for wiring manual logical CFG blocks in unit tests. */
static void
cfg_set_active(agx_apple9_vir_program *p,
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
cfg_add_edge(agx_apple9_block *from, agx_apple9_block *to)
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

class Apple9RaPcopyTest : public ::testing::Test {
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
};

} /* anonymous namespace */

/* =========================================================================
 * Part 1: Parallel-Copy Lowerer Tests
 * ========================================================================= */

TEST(Apple9RaPcopy, EmptySelfAndUndefCopies)
{
   std::vector<apple9_ra_pcopy_op> ops;
   auto opts = make_zero_free_opts();

   verify_parallel_copy({}, opts, 16, &ops);
   EXPECT_TRUE(ops.empty());

   std::vector<apple9_ra_copy> trivial = {
      {apple9_ra_loc_gpr(3), apple9_ra_loc_gpr(3), 1},
      {apple9_ra_loc_mem(4), apple9_ra_loc_mem(4), 1},
      {apple9_ra_loc_gpr(7), apple9_ra_loc_undef(), 1},
      {apple9_ra_loc_mem(9), apple9_ra_loc_undef(), 1},
   };
   verify_parallel_copy(trivial, opts, 16, &ops);
   EXPECT_TRUE(ops.empty());
}

TEST(Apple9RaPcopy, LongChainsAndFanoutTrees)
{
   std::vector<apple9_ra_copy> copies;
   /* 16-element shift chain: r0 <- r1, r1 <- r2, ..., r15 <- r16 */
   for (unsigned i = 0; i < 16; ++i)
      copies.push_back({apple9_ra_loc_gpr(i), apple9_ra_loc_gpr(i + 1), 1});
   /* Fanout tree reading r20 into r21, r22, r23 while overwriting r20 <- r24 */
   copies.push_back({apple9_ra_loc_gpr(20), apple9_ra_loc_gpr(24), 1});
   copies.push_back({apple9_ra_loc_gpr(21), apple9_ra_loc_gpr(20), 1});
   copies.push_back({apple9_ra_loc_gpr(22), apple9_ra_loc_gpr(20), 1});
   copies.push_back({apple9_ra_loc_gpr(23), apple9_ra_loc_gpr(20), 1});

   std::vector<apple9_ra_pcopy_op> ops;
   verify_parallel_copy(copies, make_zero_free_opts(), 16, &ops);
   EXPECT_EQ(ops.size(), 20u);
   for (const auto &op : ops)
      EXPECT_EQ(op.kind, APPLE9_RA_PCOPY_MOV);
}

TEST(Apple9RaPcopy, PureGprCyclesZeroFreeTempUsesSwap)
{
   /* 2-cycle with zero free temp GPRs -> exactly 1 SWAP */
   {
      std::vector<apple9_ra_copy> c2 = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_gpr(9), 1},
         {apple9_ra_loc_gpr(9), apple9_ra_loc_gpr(5), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(c2, make_zero_free_opts(), 16, &ops);
      ASSERT_EQ(ops.size(), 1u);
      EXPECT_EQ(ops[0].kind, APPLE9_RA_PCOPY_SWAP);
   }

   /* 3-cycle with zero free temp GPRs -> exactly 2 SWAPs */
   {
      std::vector<apple9_ra_copy> c3 = {
         {apple9_ra_loc_gpr(10), apple9_ra_loc_gpr(11), 1},
         {apple9_ra_loc_gpr(11), apple9_ra_loc_gpr(12), 1},
         {apple9_ra_loc_gpr(12), apple9_ra_loc_gpr(10), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(c3, make_zero_free_opts(), 16, &ops);
      ASSERT_EQ(ops.size(), 2u);
      EXPECT_EQ(ops[0].kind, APPLE9_RA_PCOPY_SWAP);
      EXPECT_EQ(ops[1].kind, APPLE9_RA_PCOPY_SWAP);
   }

   /* 3-cycle WITH a free temp GPR -> 4 MOVs (cheaper than 2 SWAPs = 6 XORs) */
   {
      std::vector<apple9_ra_copy> c3 = {
         {apple9_ra_loc_gpr(10), apple9_ra_loc_gpr(11), 1},
         {apple9_ra_loc_gpr(11), apple9_ra_loc_gpr(12), 1},
         {apple9_ra_loc_gpr(12), apple9_ra_loc_gpr(10), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      auto opts = make_zero_free_opts(1); /* r1 scratch */
      verify_parallel_copy(c3, opts, 16, &ops);
      ASSERT_EQ(ops.size(), 4u);
      for (const auto &op : ops)
         EXPECT_EQ(op.kind, APPLE9_RA_PCOPY_MOV);
   }

   /* 2-cycle with prefer_swap_2cycle -> 1 SWAP even when free temp exists */
   {
      std::vector<apple9_ra_copy> c2 = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_gpr(9), 1},
         {apple9_ra_loc_gpr(9), apple9_ra_loc_gpr(5), 1},
      };
      auto opts = make_opts();
      opts.prefer_swap_2cycle = true;
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(c2, opts, 16, &ops);
      ASSERT_EQ(ops.size(), 1u);
      EXPECT_EQ(ops[0].kind, APPLE9_RA_PCOPY_SWAP);
   }
}

TEST(Apple9RaPcopy, CycleWithFanoutAndDisjointMultiCycles)
{
   /* 3-cycle (r0<-r1<-r2<-r0) + fanout (r3<-r0, r4<-r3) + disjoint 2-cycle (r7<->r8) */
   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_gpr(0), apple9_ra_loc_gpr(1), 1},
      {apple9_ra_loc_gpr(1), apple9_ra_loc_gpr(2), 1},
      {apple9_ra_loc_gpr(2), apple9_ra_loc_gpr(0), 1},
      {apple9_ra_loc_gpr(3), apple9_ra_loc_gpr(0), 1},
      {apple9_ra_loc_gpr(4), apple9_ra_loc_gpr(3), 1},
      {apple9_ra_loc_gpr(7), apple9_ra_loc_gpr(8), 1},
      {apple9_ra_loc_gpr(8), apple9_ra_loc_gpr(7), 1},
   };
   verify_parallel_copy(copies, make_zero_free_opts());
   verify_parallel_copy(copies, make_zero_free_opts(15));
}

TEST(Apple9RaPcopy, MixedGprMemCycles)
{
   /* 2-cycle r5 <-> m0 with ONLY 1 reserved temp (r1), all 96 GPRs live -> 3 ops */
   {
      std::vector<apple9_ra_copy> copies = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_mem(0), 1},
         {apple9_ra_loc_mem(0), apple9_ra_loc_gpr(5), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(copies, make_zero_free_opts(1), 16, &ops);
      EXPECT_EQ(ops.size(), 3u);
   }

   /* 3-cycle with 1 MEM<-MEM edge (r5<-m0, m0<-m1, m1<-r5) using ONLY 1 temp (r1) */
   {
      std::vector<apple9_ra_copy> copies = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_mem(0), 1},
         {apple9_ra_loc_mem(0), apple9_ra_loc_mem(1), 1},
         {apple9_ra_loc_mem(1), apple9_ra_loc_gpr(5), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(copies, make_zero_free_opts(1), 16, &ops);
      EXPECT_EQ(ops.size(), 4u);
   }

   /* Alternating 4-cycle (r5<-m0, m0<-r6, r6<-m1, m1<-r5) with ONLY 1 temp (r1) */
   {
      std::vector<apple9_ra_copy> copies = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_mem(0), 1},
         {apple9_ra_loc_mem(0), apple9_ra_loc_gpr(6), 1},
         {apple9_ra_loc_gpr(6), apple9_ra_loc_mem(1), 1},
         {apple9_ra_loc_mem(1), apple9_ra_loc_gpr(5), 1},
      };
      std::vector<apple9_ra_pcopy_op> ops;
      verify_parallel_copy(copies, make_zero_free_opts(1), 16, &ops);
      EXPECT_EQ(ops.size(), 5u);
   }

   /* Mixed cycle with 2 adjacent MEM<-MEM edges using scratch r1 & r3 */
   {
      std::vector<apple9_ra_copy> copies = {
         {apple9_ra_loc_gpr(5), apple9_ra_loc_mem(0), 1},
         {apple9_ra_loc_mem(0), apple9_ra_loc_mem(1), 1},
         {apple9_ra_loc_mem(1), apple9_ra_loc_mem(2), 1},
         {apple9_ra_loc_mem(2), apple9_ra_loc_gpr(5), 1},
      };
      verify_parallel_copy(copies, make_zero_free_opts(1, 3), 16);
   }
}

TEST(Apple9RaPcopy, PureMemCyclesAndLargeSlots)
{
   /* Pure MEM 2-cycle & 3-cycle + high slot indices exercising heap MEM table */
   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_mem(0), apple9_ra_loc_mem(1), 1},
      {apple9_ra_loc_mem(1), apple9_ra_loc_mem(0), 1},
      {apple9_ra_loc_mem(10), apple9_ra_loc_mem(11), 1},
      {apple9_ra_loc_mem(11), apple9_ra_loc_mem(12), 1},
      {apple9_ra_loc_mem(12), apple9_ra_loc_mem(10), 1},
      {apple9_ra_loc_mem(500), apple9_ra_loc_mem(2000), 1},
      {apple9_ra_loc_mem(2000), apple9_ra_loc_mem(500), 1},
   };
   verify_parallel_copy(copies, make_zero_free_opts(1, 3), 2048);
}

TEST(Apple9RaPcopy, ConstantsHighBankAndMemImmediates)
{
   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_gpr(5), apple9_ra_loc_imm(42u), 1},
      {apple9_ra_loc_gpr(70), apple9_ra_loc_imm(0xDEADBEEFu), 1},
      {apple9_ra_loc_gpr(71), apple9_ra_loc_gpr(70), 1},
      {apple9_ra_loc_mem(10), apple9_ra_loc_imm(0x12345678u), 1},
      {apple9_ra_loc_mem(11), apple9_ra_loc_mem(10), 1},
   };
   verify_parallel_copy(copies, make_zero_free_opts(1), 32);
}

TEST(Apple9RaPcopy, MultiWordTupleCopies)
{
   /* Overlapping 4-word GPR shift down + 4-word shift up + 2-word swap +
    * mixed 4-word GPR<->MEM swap + 2-word IMM */
   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_gpr(10), apple9_ra_loc_gpr(11), 4}, /* r10..r13 <- r11..r14 */
      {apple9_ra_loc_gpr(21), apple9_ra_loc_gpr(20), 4}, /* r21..r24 <- r20..r23 */
      {apple9_ra_loc_gpr(30), apple9_ra_loc_gpr(32), 2},
      {apple9_ra_loc_gpr(32), apple9_ra_loc_gpr(30), 2},
      {apple9_ra_loc_gpr(40), apple9_ra_loc_mem(4), 4},
      {apple9_ra_loc_mem(4), apple9_ra_loc_gpr(40), 4},
      {apple9_ra_loc_gpr(50), apple9_ra_loc_imm(0xCAFEBABEu), 2},
   };
   verify_parallel_copy(copies, make_zero_free_opts(1, 3), 32);
}

TEST(Apple9RaPcopy, RejectsInvalidInputs)
{
   auto noop_emit = [](void *, const apple9_ra_pcopy_op *) { return true; };
   auto opts = make_zero_free_opts(1, 3);

   /* Duplicate GPR destination */
   apple9_ra_copy dup_gpr[2] = {
      {apple9_ra_loc_gpr(5), apple9_ra_loc_gpr(6), 1},
      {apple9_ra_loc_gpr(5), apple9_ra_loc_gpr(7), 1},
   };
   EXPECT_FALSE(apple9_ra_lower_parallel_copies(dup_gpr, 2, &opts, noop_emit,
                                                nullptr));

   /* Overlapping tuple destination (r10..r11 and r11..r12 both write r11) */
   apple9_ra_copy dup_tuple[2] = {
      {apple9_ra_loc_gpr(10), apple9_ra_loc_gpr(20), 2},
      {apple9_ra_loc_gpr(11), apple9_ra_loc_gpr(22), 2},
   };
   EXPECT_FALSE(apple9_ra_lower_parallel_copies(dup_tuple, 2, &opts, noop_emit,
                                                nullptr));

   /* Out-of-range GPR (r96) and MEM (m8192) */
   apple9_ra_copy bad_gpr[1] = {{apple9_ra_loc_gpr(96), apple9_ra_loc_gpr(0), 1}};
   EXPECT_FALSE(apple9_ra_lower_parallel_copies(bad_gpr, 1, &opts, noop_emit,
                                                nullptr));
   apple9_ra_copy bad_mem[1] = {
      {apple9_ra_loc_mem(APPLE9_RA_MAX_SPILL_SLOTS), apple9_ra_loc_gpr(0), 1}};
   EXPECT_FALSE(apple9_ra_lower_parallel_copies(bad_mem, 1, &opts, noop_emit,
                                                nullptr));

   /* MEM cycle with zero free/scratch GPRs must fail cleanly */
   apple9_ra_copy mem_cycle[2] = {
      {apple9_ra_loc_mem(0), apple9_ra_loc_mem(1), 1},
      {apple9_ra_loc_mem(1), apple9_ra_loc_mem(0), 1},
   };
   auto no_scratch = make_zero_free_opts();
   EXPECT_FALSE(apple9_ra_lower_parallel_copies(mem_cycle, 2, &no_scratch,
                                                noop_emit, nullptr));
}

TEST(Apple9RaPcopy, ExhaustiveSmallPermutations)
{
   /* All 4! = 24 permutations of 4 GPRs (0 free temps AND 1 scratch temp) */
   std::array<unsigned, 4> perm = {0, 1, 2, 3};
   do {
      std::vector<apple9_ra_copy> copies;
      for (unsigned i = 0; i < 4; ++i)
         copies.push_back(
            {apple9_ra_loc_gpr(10 + i), apple9_ra_loc_gpr(10 + perm[i]), 1});
      verify_parallel_copy(copies, make_zero_free_opts());
      verify_parallel_copy(copies, make_zero_free_opts(1));
   } while (std::next_permutation(perm.begin(), perm.end()));

   /* All 4! = 24 permutations of 2 GPRs + 2 MEMs */
   std::array<apple9_ra_loc, 4> locs = {
      apple9_ra_loc_gpr(20),
      apple9_ra_loc_gpr(21),
      apple9_ra_loc_mem(2),
      apple9_ra_loc_mem(3),
   };
   std::array<unsigned, 4> mperm = {0, 1, 2, 3};
   do {
      std::vector<apple9_ra_copy> copies;
      for (unsigned i = 0; i < 4; ++i)
         copies.push_back({locs[i], locs[mperm[i]], 1});
      verify_parallel_copy(copies, make_zero_free_opts(1, 3), 16);
   } while (std::next_permutation(mperm.begin(), mperm.end()));
}

TEST(Apple9RaPcopy, RandomizedFuzzPermutations)
{
   std::mt19937 rng(0xA9162026u);

   for (unsigned trial = 0; trial < 200; ++trial) {
      SCOPED_TRACE(trial);
      unsigned regime = trial % 3;

      if (regime == 0) {
         /* Pure GPR copies across r0..r95 with ZERO free temps & ZERO scratch */
         std::vector<unsigned> regs(AGX_APPLE9_GPR_COUNT);
         std::iota(regs.begin(), regs.end(), 0);
         std::shuffle(regs.begin(), regs.end(), rng);
         unsigned n = 2 + (rng() % 30);

         std::vector<apple9_ra_copy> copies;
         for (unsigned i = 0; i < n; ++i) {
            unsigned src_r = (rng() % 5 == 0) ? regs[rng() % n]
                                              : regs[(i + 1 + (rng() % n)) % n];
            copies.push_back(
               {apple9_ra_loc_gpr(regs[i]), apple9_ra_loc_gpr(src_r), 1});
         }
         verify_parallel_copy(copies, make_zero_free_opts(), 16);
      } else {
         /* Mixed GPR + MEM + IMM + UNDEF copies; reserve r1 & r3 as scratch */
         const unsigned mem_slots = (trial % 10 == 0) ? 320u : 48u;
         std::vector<apple9_ra_loc> pool;
         for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r) {
            if (r == 1 || r == 3)
               continue;
            pool.push_back(apple9_ra_loc_gpr(r));
         }
         for (unsigned m = 0; m < std::min(mem_slots, 32u); ++m)
            pool.push_back(apple9_ra_loc_mem(m));
         if (mem_slots > 256)
            pool.push_back(apple9_ra_loc_mem(300));

         std::shuffle(pool.begin(), pool.end(), rng);
         unsigned n = 2 + (rng() % 28);
         n = std::min<unsigned>(n, static_cast<unsigned>(pool.size()));

         std::vector<apple9_ra_copy> copies;
         for (unsigned i = 0; i < n; ++i) {
            unsigned kind_roll = rng() % 10;
            apple9_ra_loc src;
            if (kind_roll == 0)
               src = apple9_ra_loc_imm(rng());
            else if (kind_roll == 1)
               src = apple9_ra_loc_undef();
            else
               src = pool[rng() % n];
            copies.push_back({pool[i], src, 1});
         }

         auto opts = (regime == 1) ? make_zero_free_opts(1, 3)
                                   : make_opts(0xFFFFull, 0, 1, 3);
         verify_parallel_copy(copies, opts, mem_slots);
      }
   }
}

/* =========================================================================
 * Part 2: VIR Phi-Edge Lowering & Integration with Post-RA Validator
 * ========================================================================= */

TEST_F(Apple9RaPcopyTest, LowerPhiEdgeDiamondBeforeLogicalEnd)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);
   agx_apple9_block *b3 = agx_apple9_block_create(&p);

   /* b0: define v0, v1, branch to b1/b2 */
   agx_apple9_block_begin(&p, b0);
   uint32_t v0 = emit_imm(10);
   uint32_t v1 = emit_imm(20);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_PUSH, AGX_APPLE9_ENC_EXEC_MASK_PUSH, nullptr,
      0, AGX_APPLE9_EXEC_MASK_TRUE));

   /* b1 (then): compute v2, v3, end with trailing JMP_EXEC_NONE */
   agx_apple9_block_begin(&p, b1);
   uint32_t v2 = emit_iadd(v0, v1);
   uint32_t v3 = emit_iadd(v1, v2);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(
      &p, AGX_APPLE9_VIR_JMP_EXEC_NONE, AGX_APPLE9_ENC_JMP_EXEC_NONE, b2));

   /* b2 (else): opens with EXEC_MASK_ELSE, defines v4, v5, ends with EXEC_MASK_POP */
   agx_apple9_block_begin(&p, b2);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_ELSE, AGX_APPLE9_ENC_EXEC_MASK_ELSE, nullptr,
      0, 0));
   uint32_t v4 = emit_imm(100);
   uint32_t v5 = emit_imm(200);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_POP, AGX_APPLE9_ENC_EXEC_MASK_POP, nullptr,
      0, 0));

   /* b3 (merge): 2 phis creating a 2-cycle on edge b2->b3 */
   agx_apple9_block_begin(&p, b3);
   uint32_t phi_a = agx_apple9_vir_emit_phi_canonical(&p, b3, 2);
   uint32_t phi_b = agx_apple9_vir_emit_phi_canonical(&p, b3, 2);
   uint32_t sum = emit_iadd(phi_a, phi_b);
   (void)sum;

   cfg_set_active(&p, {b0, b1, b2, b3});
   cfg_add_edge(b0, b1);
   cfg_add_edge(b0, b2);
   cfg_add_edge(b1, b3);
   cfg_add_edge(b2, b3);

   /* Wire canonical phi sources: pred 0 = b1, pred 1 = b2.
    * On edge b2 -> b3, phi_a (r4) receives v4 (r5) and phi_b (r5) receives
    * v5 (r4), creating a genuine GPR 2-cycle (r4 <-> r5)! */
   agx_apple9_vir_instr *phi_a_ins = agx_apple9_block_first_instr(b3);
   agx_apple9_vir_instr *phi_b_ins = agx_apple9_instr_next(phi_a_ins);
   *agx_apple9_vir_src_ptr(phi_a_ins, 0) = v2;
   *agx_apple9_vir_src_ptr(phi_a_ins, 1) = v4;
   *agx_apple9_vir_src_ptr(phi_b_ins, 0) = v3;
   *agx_apple9_vir_src_ptr(phi_b_ins, 1) = v5;
   agx_apple9_vir_reindex(&p);

   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_ssa(&p, &err)) << err.message;

   /* Assign GPRs so b2->b3 has a 2-cycle: phi_a=r4, phi_b=r5, v4=r5, v5=r4 */
   std::vector<uint8_t> gpr(p.value_count, 0);
   gpr[v0] = 2;
   gpr[v1] = 3;
   gpr[v2] = 4;
   gpr[v3] = 5;
   gpr[v4] = 5;
   gpr[v5] = 4;
   gpr[phi_a] = 4;
   gpr[phi_b] = 5;
   gpr[sum] = 6;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();

   /* Validate pre-phi-lowering post-RA first */
   ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

   /* Lower all phi edges (zero free temps -> VIR_SWAP inserted in b2!) */
   auto opts = make_zero_free_opts();
   ASSERT_TRUE(apple9_ra_lower_all_phi_edges(&p, &assign, &opts, false));

   /* Verify b2 has: EXEC_MASK_ELSE, IMM(v4), IMM(v5), SWAP(r4, r5), EXEC_MASK_POP */
   agx_apple9_vir_instr *last_b2 = agx_apple9_block_last_instr(b2);
   ASSERT_NE(last_b2, nullptr);
   EXPECT_EQ(last_b2->op, AGX_APPLE9_VIR_EXEC_MASK_POP);
   agx_apple9_vir_instr *before_pop = agx_apple9_instr_prev(last_b2);
   ASSERT_NE(before_pop, nullptr);
   EXPECT_EQ(before_pop->op, AGX_APPLE9_VIR_SWAP);

   /* Validate post-phi-lowering with phis_lowered = true! */
   assign.phis_lowered = true;
   EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
}

TEST_F(Apple9RaPcopyTest, LowerPhiEdgeMixedGprMemLoop)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);

   /* b0: init v0 (GPR r5), v1 (MEM m0 via SSA VIR_MOV from v1_init) */
   agx_apple9_block_begin(&p, b0);
   uint32_t v0 = emit_imm(1);
   uint32_t v1_init = emit_imm(2);
   uint32_t v1 = agx_apple9_vir_emit_mov(&p, v1_init);

   /* b1 (loop header): phi_gpr (r5), phi_mem (m0) */
   agx_apple9_block_begin(&p, b1);
   uint32_t phi_gpr = agx_apple9_vir_emit_phi_canonical(&p, b1, 2);
   uint32_t phi_mem = agx_apple9_vir_emit_phi_canonical(&p, b1, 2);
   uint32_t reload = agx_apple9_vir_emit_mov(&p, phi_mem);
   uint32_t sum = emit_iadd(phi_gpr, reload);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(
      &p, AGX_APPLE9_VIR_JMP_EXEC_ANY, AGX_APPLE9_ENC_JMP_EXEC_ANY, b1));

   /* b2 (exit): use sum */
   agx_apple9_block_begin(&p, b2);
   uint32_t out = emit_iadd(sum, sum);
   (void)out;

   cfg_set_active(&p, {b0, b1, b2});
   cfg_add_edge(b0, b1);
   cfg_add_edge(b1, b1); /* self-loop backedge! */
   cfg_add_edge(b1, b2);

   /* Backedge swaps phi_gpr (r5) <-> phi_mem (m0) via sum (r5) & phi_mem (m0)!
    * Specifically: phi_gpr gets phi_mem (m0) on backedge,
    *               phi_mem gets sum (r5) on backedge -> mixed r5 <-> m0 cycle! */
   agx_apple9_vir_instr *p_gpr_ins = agx_apple9_block_first_instr(b1);
   agx_apple9_vir_instr *p_mem_ins = agx_apple9_instr_next(p_gpr_ins);
   *agx_apple9_vir_src_ptr(p_gpr_ins, 0) = v0;
   *agx_apple9_vir_src_ptr(p_gpr_ins, 1) = phi_mem;
   *agx_apple9_vir_src_ptr(p_mem_ins, 0) = v1;
   *agx_apple9_vir_src_ptr(p_mem_ins, 1) = sum;
   agx_apple9_vir_reindex(&p);

   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_ssa(&p, &err)) << err.message;

   std::vector<uint8_t> gpr(p.value_count, 0);
   std::vector<BITSET_WORD> is_mem(BITSET_WORDS(p.value_count), 0);
   std::vector<uint16_t> mem_slot(p.value_count, 0);

   gpr[v0] = 5;
   gpr[v1_init] = 6;
   BITSET_SET(is_mem.data(), v1);
   mem_slot[v1] = 0;

   gpr[phi_gpr] = 5;
   BITSET_SET(is_mem.data(), phi_mem);
   mem_slot[phi_mem] = 0;

   gpr[reload] = 6;
   gpr[sum] = 5; /* reuses killed phi_gpr (r5) */
   gpr[out] = 7;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();
   assign.is_mem = is_mem.data();
   assign.mem_slot = mem_slot.data();
   assign.max_mem_slots = 4;

   ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

   /* Lower phi edge on b1 -> b1 (mixed GPR<->MEM 2-cycle using scratch r1) */
   auto opts = make_zero_free_opts(1);
   ASSERT_TRUE(apple9_ra_lower_phi_edge(&p, b1, b1, &assign, &opts));

   /* Verify copies were inserted BEFORE JMP_EXEC_ANY */
   EXPECT_EQ(agx_apple9_block_last_instr(b1)->op, AGX_APPLE9_VIR_JMP_EXEC_ANY);

   /* Note: b1->b1 overwrote r5/m0 before JMP_EXEC_ANY, wait — b2 uses sum (r5)
    * so in real SSA without critical edge splitting, b1->b2 would be split;
    * here verify the emitted ops on b1->b1 include SPILL_LOAD and SPILL_STORE! */
   bool saw_load = false, saw_store = false;
   for (auto *ins = agx_apple9_block_first_instr(b1); ins;
        ins = agx_apple9_instr_next(ins)) {
      saw_load |= (ins->op == AGX_APPLE9_VIR_SPILL_LOAD);
      saw_store |= (ins->op == AGX_APPLE9_VIR_SPILL_STORE);
   }
   EXPECT_TRUE(saw_load);
   EXPECT_TRUE(saw_store);
}

TEST_F(Apple9RaPcopyTest, WideSidebandPhiOver10Preds)
{
   /* Build a 12-predecessor merge block using sideband phi_srcs[] */
   constexpr unsigned N = 12;
   std::vector<agx_apple9_block *> blocks;
   /* Root chain splitting into 12 leaves, all merging into blocks.back() */
   agx_apple9_block *entry = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, entry);
   blocks.push_back(entry);

   std::vector<agx_apple9_block *> splits;
   for (unsigned i = 0; i < N - 1; ++i) {
      agx_apple9_block *s = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, s);
      splits.push_back(s);
      blocks.push_back(s);
   }

   std::vector<agx_apple9_block *> leaves;
   std::vector<uint32_t> leaf_vals;
   for (unsigned i = 0; i < N; ++i) {
      agx_apple9_block *leaf = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, leaf);
      leaf_vals.push_back(emit_imm(100 + i));
      leaves.push_back(leaf);
      blocks.push_back(leaf);
   }

   agx_apple9_block *merge = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, merge);
   uint32_t phi = agx_apple9_vir_emit_phi_canonical(&p, merge, N);
   uint32_t res = emit_iadd(phi, phi);
   (void)res;
   blocks.push_back(merge);

   cfg_set_active(&p, blocks);
   cfg_add_edge(entry, splits[0]);
   for (unsigned i = 0; i < N - 2; ++i) {
      cfg_add_edge(splits[i], leaves[i]);
      cfg_add_edge(splits[i], splits[i + 1]);
   }
   cfg_add_edge(splits[N - 2], leaves[N - 2]);
   cfg_add_edge(splits[N - 2], leaves[N - 1]);
   for (unsigned i = 0; i < N; ++i)
      cfg_add_edge(leaves[i], merge);

   agx_apple9_vir_instr *phi_ins = agx_apple9_block_first_instr(merge);
   ASSERT_NE(phi_ins->phi_srcs, nullptr);
   ASSERT_EQ(agx_apple9_vir_nr_srcs(phi_ins), N);
   for (unsigned i = 0; i < N; ++i)
      *agx_apple9_vir_src_ptr(phi_ins, i) = leaf_vals[i];
   agx_apple9_vir_reindex(&p);

   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_ssa(&p, &err)) << err.message;

   std::vector<uint8_t> gpr(p.value_count, 0);
   for (unsigned i = 0; i < N; ++i)
      gpr[leaf_vals[i]] = 10 + i;
   gpr[phi] = 4;
   gpr[res] = 5;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();
   ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

   auto opts = make_zero_free_opts();
   ASSERT_TRUE(apple9_ra_lower_all_phi_edges(&p, &assign, &opts, false));
   assign.phis_lowered = true;
   EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
}

/* =========================================================================
 * Part 3: Pre-RA SSA Validator Tests
 * ========================================================================= */

TEST_F(Apple9RaPcopyTest, ValidateSsaCatchesCfgViolations)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);
   emit_imm(1);
   agx_apple9_block_begin(&p, b1);
   emit_imm(2);

   /* 1. Asymmetric CFG edge (succ set without matching pred) */
   cfg_set_active(&p, {b0, b1});
   b0->cfg_succs[0] = b1;
   apple9_ra_validate_error err = {};
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_TRUE(err.failed);
   EXPECT_NE(std::string(err.message).find("asymmetry"), std::string::npos)
      << err.message;

   /* 2. Backedge missing loop_header flag */
   cfg_set_active(&p, {b0, b1});
   cfg_add_edge(b0, b1);
   cfg_add_edge(b1, b0);
   b0->loop_header = false;
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("loop_header"), std::string::npos)
      << err.message;

   /* 3. Unreachable active block */
   cfg_set_active(&p, {b0, b1});
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("unreachable"), std::string::npos)
      << err.message;
}

TEST_F(Apple9RaPcopyTest, ValidateSsaCatchesDominanceViolations)
{
   /* Case A: Use before def in same block */
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, b0);
      uint32_t a = emit_imm(1);
      uint32_t b = emit_imm(2);
      uint32_t c = emit_iadd(a, b);
      /* Move c BEFORE b */
      agx_apple9_vir_move_before(&p, p.instructions[2], p.instructions[1]);
      cfg_set_active(&p, {b0});

      apple9_ra_validate_error err = {};
      EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
      EXPECT_NE(std::string(err.message).find("dominance"), std::string::npos)
         << err.message;
      (void)c;
   }

   /* Case B: Merge block uses value defined only in one branch of diamond */
   agx_apple9_vir_finish(&p);
   agx_apple9_vir_init(&p);
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block *b1 = agx_apple9_block_create(&p);
      agx_apple9_block *b2 = agx_apple9_block_create(&p);
      agx_apple9_block *b3 = agx_apple9_block_create(&p);

      agx_apple9_block_begin(&p, b0);
      uint32_t v0 = emit_imm(1);
      agx_apple9_block_begin(&p, b1);
      uint32_t v_then = emit_imm(2);
      agx_apple9_block_begin(&p, b2);
      emit_imm(3);
      agx_apple9_block_begin(&p, b3);
      emit_iadd(v0, v_then); /* ILLEGAL: v_then does not dominate b3! */

      cfg_set_active(&p, {b0, b1, b2, b3});
      cfg_add_edge(b0, b1);
      cfg_add_edge(b0, b2);
      cfg_add_edge(b1, b3);
      cfg_add_edge(b2, b3);

      apple9_ra_validate_error err = {};
      EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
      EXPECT_NE(std::string(err.message).find("dominance"), std::string::npos)
         << err.message;
   }
}

TEST_F(Apple9RaPcopyTest, ValidateSsaCatchesBadPhiArityAndPlacement)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);
   agx_apple9_block *b3 = agx_apple9_block_create(&p);

   agx_apple9_block_begin(&p, b0);
   emit_imm(0);
   agx_apple9_block_begin(&p, b1);
   uint32_t v1 = emit_imm(1);
   agx_apple9_block_begin(&p, b2);
   uint32_t v2 = emit_imm(2);
   agx_apple9_block_begin(&p, b3);
   /* Wrong arity: 1 instead of 2 */
   agx_apple9_vir_emit_phi_canonical(&p, b3, 1);

   cfg_set_active(&p, {b0, b1, b2, b3});
   cfg_add_edge(b0, b1);
   cfg_add_edge(b0, b2);
   cfg_add_edge(b1, b3);
   cfg_add_edge(b2, b3);

   agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(b3);
   *agx_apple9_vir_src_ptr(phi, 0) = v1;

   apple9_ra_validate_error err = {};
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("arity"), std::string::npos)
      << err.message;

   /* Fix arity to 2, then place a body instruction BEFORE the PHI */
   ASSERT_TRUE(agx_apple9_vir_phi_alloc_srcs(phi, 2));
   *agx_apple9_vir_src_ptr(phi, 0) = v1;
   *agx_apple9_vir_src_ptr(phi, 1) = v2;
   ASSERT_TRUE(apple9_ra_validate_ssa(&p, &err)) << err.message;

   emit_imm(99);
   agx_apple9_vir_move_before(&p, agx_apple9_block_last_instr(b3), phi);
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("VIR_PHI"), std::string::npos)
      << err.message;
}

TEST_F(Apple9RaPcopyTest, ValidateSsaCatchesCollectSplitArity)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);
   uint32_t a = emit_imm(1);
   uint32_t b = emit_imm(2);
   uint32_t pair_src[2] = {a, b};
   uint32_t pair = agx_apple9_vir_emit_collect(&p, pair_src, 2);
   /* Valid SPLIT of 2-component tuple */
   uint32_t split = agx_apple9_vir_emit_split(&p, pair, 2);
   (void)split;
   cfg_set_active(&p, {b0});

   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_ssa(&p, &err)) << err.message;

   /* Corrupt SPLIT dest_components from 2 to 1 (mismatch vs 2-comp COLLECT) */
   agx_apple9_vir_instr *split_ins = agx_apple9_block_last_instr(b0);
   split_ins->dest_components = 1;
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("VIR_SPLIT"), std::string::npos)
      << err.message;

   /* Restore SPLIT, corrupt COLLECT nr_srcs */
   split_ins->dest_components = 2;
   agx_apple9_vir_instr *collect_ins = agx_apple9_instr_prev(split_ins);
   collect_ins->nr_srcs = 1;
   EXPECT_FALSE(apple9_ra_validate_ssa(&p, &err));
   EXPECT_NE(std::string(err.message).find("VIR_COLLECT"), std::string::npos)
      << err.message;
}

/* =========================================================================
 * Part 4: Post-RA Register-File Validator Tests
 * ========================================================================= */

TEST_F(Apple9RaPcopyTest, ValidatePostRaCatchesClobberedLiveGprAndLoopClobber)
{
   /* Case A: Clobbered live GPR inside a single block */
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, b0);
      uint32_t v0 = emit_imm(10);
      uint32_t v1 = emit_imm(20);
      uint32_t v2 = emit_iadd(v0, v1);
      cfg_set_active(&p, {b0});

      /* Assign v0 and v1 to the SAME GPR r2 -> v1 clobbers v0! */
      std::vector<uint8_t> gpr(p.value_count, 0);
      gpr[v0] = 2;
      gpr[v1] = 2;
      gpr[v2] = 3;
      apple9_ra_assignment assign = {};
      assign.gpr = gpr.data();

      apple9_ra_validate_error err = {};
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("clobbered"), std::string::npos)
         << err.message;

      /* Fix v1 to r4 -> passes! */
      gpr[v1] = 4;
      EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
   }

   /* Case B: Loop body clobbers a loop-invariant GPR live across backedge! */
   agx_apple9_vir_finish(&p);
   agx_apple9_vir_init(&p);
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block *b1 = agx_apple9_block_create(&p);
      agx_apple9_block *b2 = agx_apple9_block_create(&p);

      agx_apple9_block_begin(&p, b0);
      uint32_t v_inv = emit_imm(42);
      uint32_t v_init = emit_imm(0);

      agx_apple9_block_begin(&p, b1);
      uint32_t v_phi = agx_apple9_vir_emit_phi_canonical(&p, b1, 2);
      uint32_t v_next = emit_iadd(v_phi, v_inv);
      uint32_t v_clob = emit_imm(999); /* overwrites v_inv's r5 on backedge! */

      agx_apple9_block_begin(&p, b2);
      uint32_t v_out = emit_iadd(v_next, v_clob);
      (void)v_out;

      cfg_set_active(&p, {b0, b1, b2});
      cfg_add_edge(b0, b1);
      cfg_add_edge(b1, b1);
      cfg_add_edge(b1, b2);

      auto *phi = agx_apple9_block_first_instr(b1);
      *agx_apple9_vir_src_ptr(phi, 0) = v_init;
      *agx_apple9_vir_src_ptr(phi, 1) = v_next;

      std::vector<uint8_t> gpr(p.value_count, 0);
      gpr[v_inv] = 5;
      gpr[v_init] = 6;
      gpr[v_phi] = 6;
      gpr[v_next] = 6;
      gpr[v_clob] = 5; /* BUG: clobbers loop-invariant v_inv in r5! */
      gpr[v_out] = 7;

      apple9_ra_assignment assign = {};
      assign.gpr = gpr.data();

      apple9_ra_validate_error err = {};
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("clobbered"), std::string::npos)
         << err.message;

      /* Move v_clob to r8 -> valid! */
      gpr[v_clob] = 8;
      EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
   }
}

TEST_F(Apple9RaPcopyTest, ValidatePostRaCatchesEncodingRangeViolations)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);
   uint32_t v_low = emit_imm(5, AGX_APPLE9_ENC_MOV_IMM_COMPACT); /* r0..r15 */
   uint32_t v_mid = emit_imm(0x12345, AGX_APPLE9_ENC_MOV_IMM32); /* r0..r63 */
   /* PRIVATE_LOAD: BOTH dest and src0 restricted to r0..r63 */
   uint32_t v_pload = agx_apple9_vir_emit_private_load(&p, v_mid, 0, 1, 32);
   uint32_t v_hi = emit_iadd(v_low, v_pload); /* r0..r95 */
   cfg_set_active(&p, {b0});

   std::vector<uint8_t> gpr(p.value_count, 0);
   gpr[v_low] = 10;
   gpr[v_mid] = 50;
   gpr[v_pload] = 63;
   gpr[v_hi] = 95;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();
   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

   /* 1. Hard-low r0..r15 violation: r16 on MOV_IMM_COMPACT */
   gpr[v_low] = 16;
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("outside allowed"), std::string::npos)
      << err.message;
   gpr[v_low] = 10;

   /* 2. r0..r63 violation: r64 on MOV_IMM32 */
   gpr[v_mid] = 64;
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("outside allowed"), std::string::npos)
      << err.message;
   gpr[v_mid] = 50;

   /* 3. r0..r63 violation: r64 on PRIVATE_LOAD dest */
   gpr[v_pload] = 64;
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("outside allowed"), std::string::npos)
      << err.message;
   gpr[v_pload] = 63;

   /* 4. r0..r95 overflow: r96 on INT_ADD_EXTENDED */
   gpr[v_hi] = 96;
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   gpr[v_hi] = 95;

   /* 5. Direct MEM operand on ALU instruction (forbidden; only MOV/PHI allowed) */
   std::vector<BITSET_WORD> is_mem(BITSET_WORDS(p.value_count), 0);
   std::vector<uint16_t> mem_slot(p.value_count, 0);
   BITSET_SET(is_mem.data(), v_hi);
   assign.is_mem = is_mem.data();
   assign.mem_slot = mem_slot.data();
   assign.max_mem_slots = 4;
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("MEM"), std::string::npos)
      << err.message;
}

TEST_F(Apple9RaPcopyTest, ValidatePostRaCatchesBadKillMask)
{
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);
   uint32_t v0 = emit_imm(1);
   uint32_t v1 = emit_imm(2);
   uint32_t v2 = emit_iadd(v0, v1);
   uint32_t v3 = emit_iadd(v0, v2); /* v0 is still live after v2! */
   cfg_set_active(&p, {b0});

   std::vector<uint8_t> gpr(p.value_count, 0);
   gpr[v0] = 2;
   gpr[v1] = 3;
   gpr[v2] = 4;
   gpr[v3] = 5;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();

   /* Valid kill_mask: v1 dies at ins[2], v0 and v2 die at ins[3] */
   agx_apple9_vir_set_src_killed(p.instructions[2], 1, true);
   agx_apple9_vir_set_src_killed(p.instructions[3], 0, true);
   agx_apple9_vir_set_src_killed(p.instructions[3], 1, true);

   apple9_ra_validate_error err = {};
   ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

   /* False kill: mark v0 killed at ins[2] even though ins[3] still uses v0! */
   agx_apple9_vir_set_src_killed(p.instructions[2], 0, true);
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("kill_mask"), std::string::npos)
      << err.message;

   /* Clear false kill, set out-of-range bit 5 on 2-src ins[2] */
   agx_apple9_vir_set_src_killed(p.instructions[2], 0, false);
   agx_apple9_vir_set_src_killed(p.instructions[2], 5, true);
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_NE(std::string(err.message).find("kill_mask"), std::string::npos)
      << err.message;
}

TEST_F(Apple9RaPcopyTest, ValidatePostRaCatchesClobberAndReuseKilledAndTuple)
{
   /* 1. Legal reuse_killed on sync IADD vs illegal reuse_killed on retained src */
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, b0);
      uint32_t v0 = emit_imm(1);
      uint32_t v1 = emit_imm(2);
      uint32_t v2 = emit_iadd(v0, v1); /* v0 dies here; reusing r2 is legal! */
      cfg_set_active(&p, {b0});

      std::vector<uint8_t> gpr(p.value_count, 0);
      gpr[v0] = 2;
      gpr[v1] = 3;
      gpr[v2] = 2; /* reuse_killed of v0's r2 */

      apple9_ra_assignment assign = {};
      assign.gpr = gpr.data();
      apple9_ra_validate_error err = {};
      ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

      /* Marking retain_mask bit 0 forbids reusing r2 */
      p.instructions[2]->retain_mask = 1u;
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("retained"), std::string::npos)
         << err.message;
   }

   /* 2. AGX_APPLE9_OPERAND_CLOBBER on MESH_CONTROL with still-live source */
   agx_apple9_vir_finish(&p);
   agx_apple9_vir_init(&p);
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, b0);
      uint32_t prim = emit_imm(3);
      uint32_t aux = emit_imm(4);
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, prim, aux, false));
      /* Using prim after MESH_CONTROL (which clobbers src0) is illegal! */
      uint32_t bad = emit_iadd(prim, aux);
      (void)bad;
      cfg_set_active(&p, {b0});

      std::vector<uint8_t> gpr(p.value_count, 0);
      gpr[prim] = 2;
      gpr[aux] = 3;
      gpr[bad] = 4;

      apple9_ra_assignment assign = {};
      assign.gpr = gpr.data();
      apple9_ra_validate_error err = {};
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("clobbered"), std::string::npos)
         << err.message;
   }

   /* 3. Non-contiguous multi-word tuple on vector DEVICE_LOAD & illegal
    *    reuse_killed on multi-component load dest */
   agx_apple9_vir_finish(&p);
   agx_apple9_vir_init(&p);
   {
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      agx_apple9_block_begin(&p, b0);
      uint32_t idx = emit_imm(0);
      agx_apple9_device_load_contract contract = {
         .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
         .flags = 0,
         .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
      };
      uint32_t vec = agx_apple9_vir_emit_device_load_vector(&p, 0, idx, 2,
                                                            &contract);
      uint32_t sum = emit_iadd(vec, vec + 1);
      (void)sum;
      cfg_set_active(&p, {b0});

      std::vector<uint8_t> gpr(p.value_count, 0);
      gpr[idx] = 2;
      gpr[vec] = 4;
      gpr[vec + 1] = 5;
      gpr[sum] = 6;

      apple9_ra_assignment assign = {};
      assign.gpr = gpr.data();
      apple9_ra_validate_error err = {};
      ASSERT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;

      /* Non-contiguous tuple: vec=r4, vec+1=r7 */
      gpr[vec + 1] = 7;
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("contiguous"), std::string::npos)
         << err.message;

      /* Multi-component dest overlapping killed index (reuse_killed forbidden) */
      gpr[vec] = 2;
      gpr[vec + 1] = 3;
      EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
      EXPECT_NE(std::string(err.message).find("reuse_killed"),
                std::string::npos)
         << err.message;
   }
}

TEST_F(Apple9RaPcopyTest, ThenArmTrailingExecMaskElsePlacesPhiCopiesBeforeElse)
{
   /* Mirrors agx_compile_apple9.c when the then-arm has no skip target:
    * EXEC_MASK_ELSE (and optional JMP_EXEC_NONE over the else-arm) sit at the
    * tail of the then-arm block. Phi-edge copies MUST execute before
    * EXEC_MASK_ELSE under the then-arm execution mask. */
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b_then = agx_apple9_block_create(&p);
   agx_apple9_block *b_else = agx_apple9_block_create(&p);
   agx_apple9_block *b_pop = agx_apple9_block_create(&p);
   agx_apple9_block *b_merge = agx_apple9_block_create(&p);

   agx_apple9_block_begin(&p, b0);
   uint32_t v0 = emit_imm(7);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_PUSH, AGX_APPLE9_ENC_EXEC_MASK_PUSH, nullptr,
      0, AGX_APPLE9_EXEC_MASK_TRUE));

   /* Non-empty then-arm ending with EXEC_MASK_ELSE + JMP_EXEC_NONE b_pop */
   agx_apple9_block_begin(&p, b_then);
   uint32_t v_then = emit_imm(42);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_ELSE, AGX_APPLE9_ENC_EXEC_MASK_ELSE, nullptr,
      0, 0));
   ASSERT_TRUE(agx_apple9_vir_emit_branch(
      &p, AGX_APPLE9_VIR_JMP_EXEC_NONE, AGX_APPLE9_ENC_JMP_EXEC_NONE, b_pop));

   agx_apple9_block_begin(&p, b_else);
   uint32_t v_else = emit_iadd(v0, v0);

   agx_apple9_block_begin(&p, b_pop);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_POP, AGX_APPLE9_ENC_EXEC_MASK_POP, nullptr,
      0, 0));

   agx_apple9_block_begin(&p, b_merge);
   uint32_t phi = agx_apple9_vir_emit_phi_canonical(&p, b_merge, 2);
   uint32_t out = emit_iadd(phi, v0);
   (void)out;

   cfg_set_active(&p, {b0, b_then, b_else, b_merge});
   cfg_add_edge(b0, b_then);
   cfg_add_edge(b0, b_else);
   cfg_add_edge(b_then, b_merge);
   cfg_add_edge(b_else, b_merge);

   agx_apple9_vir_instr *phi_ins = agx_apple9_block_first_instr(b_merge);
   *agx_apple9_vir_src_ptr(phi_ins, 0) = v_then;
   *agx_apple9_vir_src_ptr(phi_ins, 1) = v_else;
   agx_apple9_vir_reindex(&p);

   /* Anchor for b_then must be EXEC_MASK_ELSE, not JMP_EXEC_NONE. */
   agx_apple9_vir_instr *anchor = apple9_ra_block_logical_end_anchor(b_then);
   ASSERT_NE(anchor, nullptr);
   EXPECT_EQ(anchor->op, AGX_APPLE9_VIR_EXEC_MASK_ELSE);

   std::vector<uint8_t> gpr(p.value_count, 0);
   gpr[v0] = 2;
   gpr[v_then] = 3;
   gpr[v_else] = 4;
   gpr[phi] = 4;
   gpr[out] = 5;

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();
   auto opts = make_zero_free_opts();
   ASSERT_TRUE(apple9_ra_lower_all_phi_edges(&p, &assign, &opts, false));

   /* Expect b_then order: IMM(v_then), MOV(r4<-r3), EXEC_MASK_ELSE, JMP_EXEC_NONE */
   agx_apple9_vir_instr *i0 = agx_apple9_block_first_instr(b_then);
   ASSERT_NE(i0, nullptr);
   EXPECT_EQ(i0->op, AGX_APPLE9_VIR_IMM);
   agx_apple9_vir_instr *i1 = agx_apple9_instr_next(i0);
   ASSERT_NE(i1, nullptr);
   EXPECT_EQ(i1->op, AGX_APPLE9_VIR_MOV);
   EXPECT_EQ(i1->dest, 4u);
   agx_apple9_vir_instr *i2 = agx_apple9_instr_next(i1);
   ASSERT_NE(i2, nullptr);
   EXPECT_EQ(i2->op, AGX_APPLE9_VIR_EXEC_MASK_ELSE);
   agx_apple9_vir_instr *i3 = agx_apple9_instr_next(i2);
   ASSERT_NE(i3, nullptr);
   EXPECT_EQ(i3->op, AGX_APPLE9_VIR_JMP_EXEC_NONE);
   EXPECT_EQ(agx_apple9_instr_next(i3), nullptr);

   assign.phis_lowered = true;
   apple9_ra_validate_error err = {};
   EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
}

TEST(Apple9RaPcopy, IdentityDestinationNotClobberedAsCycleScratch)
{
   /* P0-C9: (r0<-r0, r1<-r2, r2<-r1) with r0 not marked live must NOT borrow
    * identity destination r0 as cycle scratch. */
   apple9_ra_pcopy_opts opts = {};
   opts.scratch_gpr = AGX_APPLE9_PHYS_INVALID;
   opts.scratch_gpr2 = AGX_APPLE9_PHYS_INVALID;
   opts.max_temp_gpr = 64;
   /* Only mark r1..r2 live; r0 is NOT in live_gprs so without marking identity
    * destinations in gpr_written, find_temp_gpr() would clobber r0. */
   opts.live_gprs[0] = (1ull << 1) | (1ull << 2);

   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_gpr(0), apple9_ra_loc_gpr(0), 1},
      {apple9_ra_loc_gpr(1), apple9_ra_loc_gpr(2), 1},
      {apple9_ra_loc_gpr(2), apple9_ra_loc_gpr(1), 1},
   };
   verify_parallel_copy(copies, opts, 0);
}

TEST_F(Apple9RaPcopyTest, ValidatePostRaRejectsClobberedProgramOutput)
{
   /* P0-C10: Post-RA validator must reject an allocation where a later
    * instruction overwrites program.output's physical register before exit. */
   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, b0);
   uint32_t out_v = emit_imm(42);
   uint32_t clobber_v = emit_imm(99);
   p.output = out_v;

   cfg_set_active(&p, {b0});
   agx_apple9_vir_reindex(&p);

   std::vector<uint8_t> gpr(p.value_count, 0);
   gpr[out_v] = 4;
   gpr[clobber_v] = 4; /* Overwrites program.output in r4! */

   apple9_ra_assignment assign = {};
   assign.gpr = gpr.data();
   apple9_ra_validate_error err = {};
   EXPECT_FALSE(apple9_ra_validate_post_ra(&p, &assign, &err));
   EXPECT_TRUE(err.failed);

   gpr[clobber_v] = 5;
   EXPECT_TRUE(apple9_ra_validate_post_ra(&p, &assign, &err)) << err.message;
}

TEST(Apple9RaPcopy, MemFanOutReusesSingleFillAndRotatesStagingGprs)
{
   /* 1) 4-way MEM[1..4] <- MEM[0] fan-out must emit exactly 1 GPR<-MEM fill
    *    and 4 MEM<-GPR stores (instead of 4 redundant fills).
    * 2) 4-way MEM[10..13] <- IMM(0) fan-out must emit exactly 1 MOV_IMM and
    *    4 MEM<-GPR stores.
    * 3) Two distinct MEM <- IMM transfers rotate between scratch_gpr (r3) and
    *    scratch_gpr2 (r4) to avoid immediate WAR WAITs. */
   apple9_ra_pcopy_opts opts = make_opts(~0ull, ~0ull, 3, 4);

   std::vector<apple9_ra_copy> copies = {
      {apple9_ra_loc_mem(1), apple9_ra_loc_mem(0), 1},
      {apple9_ra_loc_mem(2), apple9_ra_loc_mem(0), 1},
      {apple9_ra_loc_mem(3), apple9_ra_loc_mem(0), 1},
      {apple9_ra_loc_mem(4), apple9_ra_loc_mem(0), 1},
      {apple9_ra_loc_mem(10), apple9_ra_loc_imm(0), 1},
      {apple9_ra_loc_mem(11), apple9_ra_loc_imm(0), 1},
      {apple9_ra_loc_mem(12), apple9_ra_loc_imm(0), 1},
      {apple9_ra_loc_mem(13), apple9_ra_loc_imm(0), 1},
      {apple9_ra_loc_mem(20), apple9_ra_loc_imm(0x3f800000u), 1},
   };

   std::vector<apple9_ra_pcopy_op> ops;
   verify_parallel_copy(copies, opts, 64, &ops);

   unsigned fills = 0, stores = 0, imms = 0;
   std::vector<unsigned> imm_regs;
   for (const auto &op : ops) {
      if (op.kind == APPLE9_RA_PCOPY_MOV &&
          op.dst.kind == APPLE9_RA_LOC_GPR && op.src.kind == APPLE9_RA_LOC_MEM)
         fills++;
      if (op.kind == APPLE9_RA_PCOPY_MOV &&
          op.dst.kind == APPLE9_RA_LOC_MEM && op.src.kind == APPLE9_RA_LOC_GPR)
         stores++;
      if (op.kind == APPLE9_RA_PCOPY_MOV_IMM) {
         imms++;
         imm_regs.push_back(op.dst.value);
      }
   }
   EXPECT_EQ(fills, 1u);
   EXPECT_EQ(stores, 9u);
   EXPECT_EQ(imms, 2u);
   ASSERT_EQ(imm_regs.size(), 2u);
   EXPECT_NE(imm_regs[0], imm_regs[1]);
}
