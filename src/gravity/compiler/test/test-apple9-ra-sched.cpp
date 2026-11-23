/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include "compiler/nir/nir.h"
#include "apple9_ra_pcopy.h"
#include "apple9_ra_sched.h"

namespace {

static uint32_t
emit_imm(agx_apple9_vir_program *p, uint32_t val,
         enum agx_apple9_encoding enc = AGX_APPLE9_ENC_MOV_IMM32)
{
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IMM, enc, nullptr, 0, val);
}

static uint32_t
emit_iadd(agx_apple9_vir_program *p, uint32_t a, uint32_t b)
{
   const uint32_t srcs[2] = {a, b};
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IADD,
                              AGX_APPLE9_ENC_INT_ADD_EXTENDED, srcs, 2, 0);
}

static uint32_t
emit_dev_load(agx_apple9_vir_program *p, unsigned binding, uint32_t idx,
              unsigned components = 1)
{
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100,
   };
   if (components == 1)
      return agx_apple9_vir_emit_device_load(p, binding, idx, &contract);
   return agx_apple9_vir_emit_device_load_vector(p, binding, idx, components,
                                                 &contract);
}

static bool
emit_dev_store(agx_apple9_vir_program *p, unsigned binding, uint32_t idx,
               uint32_t val)
{
   return agx_apple9_vir_emit_device_store(p, binding, idx, &val, 1, 32);
}

static unsigned
instr_pos_in_block(const agx_apple9_block *blk,
                   const agx_apple9_vir_instr *needle)
{
   unsigned pos = 0;
   for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
        I = agx_apple9_instr_next(I), ++pos) {
      if (I == needle)
         return pos;
   }
   return UINT_MAX;
}

static bool
block_contains(const agx_apple9_block *blk, const agx_apple9_vir_instr *needle)
{
   return instr_pos_in_block(blk, needle) != UINT_MAX;
}

static void
verify_ssa_def_before_use_in_block(const agx_apple9_vir_program *p,
                                   const agx_apple9_block *blk)
{
   std::vector<unsigned> def_pos(p->value_count, UINT_MAX);
   unsigned pos = 0;
   for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
        I = agx_apple9_instr_next(I), ++pos) {
      if (I->dest != AGX_APPLE9_VREG_INVALID) {
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->dest + c < p->value_count)
               def_pos[I->dest + c] = pos;
         }
      }
   }

   pos = 0;
   for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
        I = agx_apple9_instr_next(I), ++pos) {
      if (I->op == AGX_APPLE9_VIR_PHI)
         continue;
      const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t v = agx_apple9_vir_src(I, s);
         if (v < p->value_count && def_pos[v] != UINT_MAX) {
            EXPECT_LT(def_pos[v], pos);
         }
      }
   }
}

} /* namespace */

/* =========================================================================
 * 1. Native VIR Partial DCE Tests
 * ========================================================================= */

TEST(Apple9RaDce, EliminatesDeadPureAluChain)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t live_a = emit_imm(&p, 10);
   uint32_t live_b = emit_imm(&p, 20);
   uint32_t live_sum = emit_iadd(&p, live_a, live_b);

   /* Dead pure chain that feeds nothing. */
   uint32_t dead_a = emit_imm(&p, 100);
   uint32_t dead_b = emit_imm(&p, 200);
   uint32_t dead_sum = emit_iadd(&p, dead_a, dead_b);
   uint32_t dead_sum2 = emit_iadd(&p, dead_sum, live_a);
   (void)dead_sum2;

   ASSERT_TRUE(emit_dev_store(&p, 0, idx, live_sum));
   ASSERT_EQ(p.instruction_count, 8u);

   ASSERT_TRUE(apple9_ra_dce(&p));
   /* Only live_a, live_b, live_sum, and store remain. */
   EXPECT_EQ(p.instruction_count, 4u);
   EXPECT_EQ(p.instructions[0]->dest, live_a);
   EXPECT_EQ(p.instructions[1]->dest, live_b);
   EXPECT_EQ(p.instructions[2]->dest, live_sum);
   EXPECT_EQ(p.instructions[3]->op, AGX_APPLE9_VIR_DEVICE_STORE);

   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaDce, PreservesPublicationHandoffAndDest)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t src = emit_imm(&p, 42);
   uint32_t handoff_val = emit_iadd(&p, src, src);
   agx_apple9_vir_instr *handoff = p.instructions[p.instruction_count - 1];
   handoff->publication_handoff = true;

   /* Dead ordinary ALU beside it must still be removed. */
   uint32_t dead_val = emit_imm(&p, 99);
   (void)dead_val;

   ASSERT_EQ(p.instruction_count, 3u);
   ASSERT_TRUE(apple9_ra_dce(&p));

   EXPECT_EQ(p.instruction_count, 2u);
   EXPECT_TRUE(block_contains(b0, handoff));
   EXPECT_EQ(handoff->dead_dest_mask, 0u);
   EXPECT_EQ(p.instructions[0]->dest, src);
   EXPECT_EQ(p.instructions[1]->dest, handoff_val);

   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaDce, MarksPartialMultiDestDeadDestMask)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);

   /* Case A: COLLECT + SPLIT where only lanes 0 and 2 of SPLIT are live. */
   uint32_t c0 = emit_imm(&p, 1);
   uint32_t c1 = emit_imm(&p, 2);
   uint32_t c2 = emit_imm(&p, 3);
   uint32_t c3 = emit_imm(&p, 4);
   const uint32_t elems[4] = {c0, c1, c2, c3};
   uint32_t tup = agx_apple9_vir_emit_collect(&p, elems, 4);
   agx_apple9_vir_instr *collect_ins = p.instructions[p.instruction_count - 1];
   uint32_t split = agx_apple9_vir_emit_split(&p, tup, 4);
   agx_apple9_vir_instr *split_ins = p.instructions[p.instruction_count - 1];

   uint32_t sum_split = emit_iadd(&p, split + 0, split + 2);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, sum_split));

   /* Case B: Vector DEVICE_LOAD where only lane 1 is read directly. */
   uint32_t vload = emit_dev_load(&p, 1, idx, 4);
   agx_apple9_vir_instr *vload_ins = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(emit_dev_store(&p, 2, idx, vload + 1));

   /* Case C: Completely dead COLLECT + SPLIT (must both be eliminated). */
   uint32_t d0 = emit_imm(&p, 7);
   uint32_t d1 = emit_imm(&p, 8);
   const uint32_t delems[2] = {d0, d1};
   uint32_t dtup = agx_apple9_vir_emit_collect(&p, delems, 2);
   agx_apple9_vir_instr *dcollect_ins = p.instructions[p.instruction_count - 1];
   uint32_t dsplit = agx_apple9_vir_emit_split(&p, dtup, 2);
   agx_apple9_vir_instr *dsplit_ins = p.instructions[p.instruction_count - 1];
   (void)dsplit;

   ASSERT_TRUE(apple9_ra_dce(&p));

   EXPECT_TRUE(block_contains(b0, collect_ins));
   EXPECT_EQ(collect_ins->dead_dest_mask, 0u);
   EXPECT_TRUE(block_contains(b0, split_ins));
   EXPECT_EQ(split_ins->dead_dest_mask, (uint8_t)((1u << 1) | (1u << 3)));

   EXPECT_TRUE(block_contains(b0, vload_ins));
   EXPECT_EQ(vload_ins->dead_dest_mask,
             (uint8_t)((1u << 0) | (1u << 2) | (1u << 3)));

   EXPECT_FALSE(block_contains(b0, dcollect_ins));
   EXPECT_FALSE(block_contains(b0, dsplit_ins));

   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaDce, PreservesSideEffectsAtomicsVolatileAndControlFlow)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t data = emit_imm(&p, 5);

   /* Volatile load with unused result must stay live and mark dest dead. */
   uint32_t vload = emit_dev_load(&p, 0, idx, 1);
   agx_apple9_vir_instr *vload_ins = p.instructions[p.instruction_count - 1];
   vload_ins->memory_volatile = true;
   (void)vload;

   /* Returning DEVICE_ATOMIC whose result is never consumed by shader code. */
   uint32_t atom_res = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &p, 0, idx, &data, 1, AGX_APPLE9_ATOMIC_ADD, false, &atom_res));
   agx_apple9_vir_instr *atom_ins = p.instructions[p.instruction_count - 2];
   agx_apple9_vir_instr *atom_rec = p.instructions[p.instruction_count - 1];
   ASSERT_EQ(atom_ins->op, AGX_APPLE9_VIR_DEVICE_ATOMIC);
   ASSERT_EQ(atom_rec->op, AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);

   /* Barrier + HALT. */
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_WORKGROUP_BARRIER, AGX_APPLE9_ENC_WORKGROUP_BARRIER,
      nullptr, 0, 0));
   agx_apple9_vir_instr *barrier = p.instructions[p.instruction_count - 1];

   /* Live-out value must survive DCE. */
   uint32_t out_val = emit_imm(&p, 123);
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, out_val));

   /* Dead pure op must disappear. */
   uint32_t dead = emit_imm(&p, 999);
   (void)dead;

   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));
   agx_apple9_vir_instr *halt = p.instructions[p.instruction_count - 1];

   ASSERT_TRUE(apple9_ra_dce(&p));

   EXPECT_TRUE(block_contains(b0, vload_ins));
   EXPECT_EQ(vload_ins->dead_dest_mask, 1u);
   EXPECT_TRUE(block_contains(b0, atom_ins));
   EXPECT_EQ(atom_ins->dead_dest_mask, 1u);
   EXPECT_TRUE(block_contains(b0, atom_rec));
   EXPECT_TRUE(block_contains(b0, barrier));
   EXPECT_TRUE(block_contains(b0, halt));
   EXPECT_EQ(p.instruction_count, 7u); /* data, vload, atom, atom_rec, barrier, out_val, halt */

   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaDce, EliminatesDeadLoopPhiCycleAndKeepsLivePhi)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n0 = {}, n1 = {}, n2 = {};
   n0.successors[0] = &n1;
   n1.successors[0] = &n1;
   n1.successors[1] = &n2;

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);
   ASSERT_TRUE(b0 && b1 && b2);
   b0->nir = &n0;
   b1->nir = &n1;
   b2->nir = &n2;

   agx_apple9_block_begin(&p, b0);
   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t live_init = emit_imm(&p, 1);
   uint32_t dead_init = emit_imm(&p, 2);

   agx_apple9_block_begin(&p, b1);
   uint32_t live_phi = agx_apple9_vir_emit_phi_canonical(&p, b1, 2);
   agx_apple9_vir_instr *live_phi_ins = p.instructions[p.instruction_count - 1];
   uint32_t dead_phi = agx_apple9_vir_emit_phi_canonical(&p, b1, 2);
   agx_apple9_vir_instr *dead_phi_ins = p.instructions[p.instruction_count - 1];
   uint32_t live_next = emit_iadd(&p, live_phi, live_init);
   uint32_t dead_next = emit_iadd(&p, dead_phi, dead_init);
   agx_apple9_vir_instr *dead_next_ins = p.instructions[p.instruction_count - 1];

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   unsigned p0 = agx_apple9_block_pred_index(b1, b0);
   unsigned p1 = agx_apple9_block_pred_index(b1, b1);
   ASSERT_NE(p0, UINT_MAX);
   ASSERT_NE(p1, UINT_MAX);
   *agx_apple9_vir_src_ptr(live_phi_ins, p0) = live_init;
   *agx_apple9_vir_src_ptr(live_phi_ins, p1) = live_next;
   *agx_apple9_vir_src_ptr(dead_phi_ins, p0) = dead_init;
   *agx_apple9_vir_src_ptr(dead_phi_ins, p1) = dead_next;

   agx_apple9_block_begin(&p, b2);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, live_next));

   ASSERT_TRUE(apple9_ra_dce(&p));

   EXPECT_TRUE(block_contains(b1, live_phi_ins));
   EXPECT_FALSE(block_contains(b1, dead_phi_ins));
   EXPECT_FALSE(block_contains(b1, dead_next_ins));

   agx_apple9_vir_finish(&p);
}

/* =========================================================================
 * 2. Native VIR Sparse-Bitset CFG Liveness Tests
 * ========================================================================= */

TEST(Apple9RaLiveness, DiamondCfgCanonicalPhiKillMaskAnd32BitUnits)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n0 = {}, n1 = {}, n2 = {}, n3 = {};
   n0.successors[0] = &n1;
   n0.successors[1] = &n2;
   n1.successors[0] = &n3;
   n2.successors[0] = &n3;

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);
   agx_apple9_block *b3 = agx_apple9_block_create(&p);
   ASSERT_TRUE(b0 && b1 && b2 && b3);
   b0->nir = &n0;
   b1->nir = &n1;
   b2->nir = &n2;
   b3->nir = &n3;

   agx_apple9_block_begin(&p, b0);
   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t v0 = emit_imm(&p, 11);
   uint32_t v1 = emit_imm(&p, 22);

   agx_apple9_block_begin(&p, b1);
   /* Duplicate operand IADD(v0, v0): both src bits must be killed. */
   uint32_t left_val = emit_iadd(&p, v0, v0);
   agx_apple9_vir_instr *left_add = p.instructions[p.instruction_count - 1];

   agx_apple9_block_begin(&p, b2);
   uint32_t right_val = emit_iadd(&p, v1, v1);
   agx_apple9_vir_instr *right_add = p.instructions[p.instruction_count - 1];

   agx_apple9_block_begin(&p, b3);
   uint32_t merged = agx_apple9_vir_emit_phi_canonical(&p, b3, 2);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, merged));
   agx_apple9_vir_instr *store = p.instructions[p.instruction_count - 1];

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   unsigned p_left = agx_apple9_block_pred_index(b3, b1);
   unsigned p_right = agx_apple9_block_pred_index(b3, b2);
   ASSERT_NE(p_left, UINT_MAX);
   ASSERT_NE(p_right, UINT_MAX);
   *agx_apple9_vir_src_ptr(phi, p_left) = left_val;
   *agx_apple9_vir_src_ptr(phi, p_right) = right_val;

   apple9_ra_liveness_info live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &live));
   EXPECT_EQ(live.num_cfg_blocks, 4u);

   /* Entry block: idx, v0, v1 live out (3 32-bit GPRs, NOT 6 half-regs). */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b0->cfg_index], idx));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b0->cfg_index], v0));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b0->cfg_index], v1));
   EXPECT_EQ(u_sparse_bitset_count(&live.live_out[b0->cfg_index]), 3u);

   /* Left block: needs v0 + idx in, left_val + idx out; v1 NOT live here. */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_in[b1->cfg_index], v0));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_in[b1->cfg_index], v1));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b1->cfg_index], left_val));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_out[b1->cfg_index], right_val));

   /* Right block: needs v1 + idx in, right_val + idx out; v0 NOT live here. */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_in[b2->cfg_index], v1));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_in[b2->cfg_index], v0));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b2->cfg_index], right_val));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_out[b2->cfg_index], left_val));

   /* Merge block: merged + idx live in; empty live out. */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_in[b3->cfg_index], merged));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_in[b3->cfg_index], idx));
   EXPECT_EQ(u_sparse_bitset_count(&live.live_out[b3->cfg_index]), 0u);

   /* Kill masks: duplicate operand v0 kills both bits (0x3). */
   EXPECT_EQ(left_add->kill_mask, 0x3u);
   EXPECT_EQ(right_add->kill_mask, 0x3u);
   EXPECT_EQ(phi->kill_mask, 0u);
   EXPECT_EQ(store->kill_mask, 0x3u);

   /* Global peak GPRs = 3 (idx, v0, v1 at b0 exit). */
   EXPECT_EQ(live.block_peak_gprs[b0->cfg_index], 3u);
   EXPECT_EQ(live.peak_live_gprs, 3u);
   EXPECT_EQ(p.peak_live_gprs, 3u);

   apple9_ra_liveness_finish(&live);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaLiveness, LoopBackedgeKillMaskAndParallelPhiSwap)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n_entry = {}, n_hdr = {}, n_latch = {}, n_exit = {};
   n_entry.successors[0] = &n_hdr;
   n_hdr.successors[0] = &n_latch;
   n_hdr.successors[1] = &n_exit;
   n_latch.successors[0] = &n_hdr;

   agx_apple9_block *entry = agx_apple9_block_create(&p);
   agx_apple9_block *hdr = agx_apple9_block_create(&p);
   agx_apple9_block *latch = agx_apple9_block_create(&p);
   agx_apple9_block *exit_b = agx_apple9_block_create(&p);
   ASSERT_TRUE(entry && hdr && latch && exit_b);
   entry->nir = &n_entry;
   hdr->nir = &n_hdr;
   latch->nir = &n_latch;
   exit_b->nir = &n_exit;

   agx_apple9_block_begin(&p, entry);
   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t inv = emit_imm(&p, 7);
   uint32_t a0 = emit_imm(&p, 1);
   uint32_t b0 = emit_imm(&p, 2);

   agx_apple9_block_begin(&p, hdr);
   uint32_t phi_a = agx_apple9_vir_emit_phi_canonical(&p, hdr, 2);
   agx_apple9_vir_instr *phi_a_ins = p.instructions[p.instruction_count - 1];
   uint32_t phi_b = agx_apple9_vir_emit_phi_canonical(&p, hdr, 2);
   agx_apple9_vir_instr *phi_b_ins = p.instructions[p.instruction_count - 1];
   /* Uses phi_a (dies if not swapped) and inv (loop-invariant -> NOT killed!). */
   uint32_t step = emit_iadd(&p, phi_a, inv);
   agx_apple9_vir_instr *step_ins = p.instructions[p.instruction_count - 1];

   agx_apple9_block_begin(&p, latch);
   /* Parallel swap across backedge: phi_a <= phi_b, phi_b <= step. */

   agx_apple9_block_begin(&p, exit_b);
   uint32_t res = emit_iadd(&p, step, phi_b);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, res));

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   ASSERT_TRUE(hdr->loop_header);
   unsigned pe = agx_apple9_block_pred_index(hdr, entry);
   unsigned pl = agx_apple9_block_pred_index(hdr, latch);
   ASSERT_NE(pe, UINT_MAX);
   ASSERT_NE(pl, UINT_MAX);
   *agx_apple9_vir_src_ptr(phi_a_ins, pe) = a0;
   *agx_apple9_vir_src_ptr(phi_a_ins, pl) = phi_b;
   *agx_apple9_vir_src_ptr(phi_b_ins, pe) = b0;
   *agx_apple9_vir_src_ptr(phi_b_ins, pl) = step;

   apple9_ra_liveness_info live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &live));

   /* inv is loop-invariant: must stay live out of hdr & latch, NOT killed in step_ins. */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[hdr->cfg_index], inv));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[latch->cfg_index], inv));
   EXPECT_TRUE(agx_apple9_vir_src_killed(step_ins, 0));  /* phi_a is killed */
   EXPECT_FALSE(agx_apple9_vir_src_killed(step_ins, 1)); /* inv stays live */

   /* Both phi_b and step must be live out of latch for the parallel phi swap. */
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[latch->cfg_index], phi_b));
   EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[latch->cfg_index], step));

   apple9_ra_liveness_finish(&live);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaLiveness, SidebandCanonicalPhiMoreThanTenPreds)
{
   constexpr unsigned kPreds = 12; /* > AGX_APPLE9_MAX_VIR_SRCS (10) */
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   /* Build a binary dispatch tree of 11 internal nodes feeding 12 leaf arms
    * that all converge into one merge block with a 12-pred sideband VIR_PHI. */
   std::vector<nir_block> n_dispatch(kPreds - 1);
   std::vector<nir_block> n_arms(kPreds);
   nir_block n_merge = {};
   for (unsigned i = 0; i < kPreds - 1; ++i) {
      n_dispatch[i] = {};
      n_dispatch[i].successors[0] = &n_arms[i];
      n_dispatch[i].successors[1] =
         (i + 1 < kPreds - 1) ? &n_dispatch[i + 1] : &n_arms[kPreds - 1];
   }
   for (unsigned i = 0; i < kPreds; ++i) {
      n_arms[i] = {};
      n_arms[i].successors[0] = &n_merge;
   }

   std::vector<agx_apple9_block *> dispatch(kPreds - 1);
   std::vector<agx_apple9_block *> arms(kPreds);
   for (unsigned i = 0; i < kPreds - 1; ++i) {
      dispatch[i] = agx_apple9_block_create(&p);
      ASSERT_NE(dispatch[i], nullptr);
      dispatch[i]->nir = &n_dispatch[i];
      agx_apple9_block_begin(&p, dispatch[i]);
   }
   uint32_t idx = agx_apple9_vir_input(&p, 0);

   std::vector<uint32_t> arm_vals(kPreds);
   for (unsigned i = 0; i < kPreds; ++i) {
      arms[i] = agx_apple9_block_create(&p);
      ASSERT_NE(arms[i], nullptr);
      arms[i]->nir = &n_arms[i];
      agx_apple9_block_begin(&p, arms[i]);
      arm_vals[i] = emit_imm(&p, 100 + i);
   }

   agx_apple9_block *merge = agx_apple9_block_create(&p);
   ASSERT_NE(merge, nullptr);
   merge->nir = &n_merge;
   agx_apple9_block_begin(&p, merge);

   uint32_t merged = agx_apple9_vir_emit_phi_canonical(&p, merge, kPreds);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   ASSERT_NE(phi->phi_srcs, nullptr);
   ASSERT_EQ(agx_apple9_vir_nr_srcs(phi), kPreds);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, merged));

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   ASSERT_EQ(merge->num_cfg_preds, kPreds);
   for (unsigned i = 0; i < kPreds; ++i) {
      unsigned pidx = agx_apple9_block_pred_index(merge, arms[i]);
      ASSERT_NE(pidx, UINT_MAX);
      *agx_apple9_vir_src_ptr(phi, pidx) = arm_vals[i];
   }

   apple9_ra_liveness_info live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &live));

   EXPECT_TRUE(u_sparse_bitset_test(&live.live_in[merge->cfg_index], merged));
   for (unsigned i = 0; i < kPreds; ++i) {
      const unsigned b = arms[i]->cfg_index;
      EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b], arm_vals[i]));
      EXPECT_TRUE(u_sparse_bitset_test(&live.live_out[b], idx));
      EXPECT_EQ(u_sparse_bitset_count(&live.live_out[b]), 2u);
      for (unsigned j = 0; j < kPreds; ++j) {
         if (j != i) {
            EXPECT_FALSE(u_sparse_bitset_test(&live.live_out[b], arm_vals[j]));
         }
      }
   }

   apple9_ra_liveness_finish(&live);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaLiveness, SkipsPublicationsInlineUniformsAndUnreachableBlocks)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n_entry = {}, n_inf = {}, n_dead = {};
   n_entry.successors[0] = &n_inf;
   n_inf.successors[0] = &n_inf;
   /* Unreachable block after infinite loop, plus a synthetic !nir block. */
   n_dead.successors[0] = &n_inf;

   agx_apple9_block *entry = agx_apple9_block_create(&p);
   agx_apple9_block *inf = agx_apple9_block_create(&p);
   agx_apple9_block *synth = agx_apple9_block_create(&p);
   agx_apple9_block *dead = agx_apple9_block_create(&p);
   ASSERT_TRUE(entry && inf && synth && dead);
   entry->nir = &n_entry;
   inf->nir = &n_inf;
   synth->nir = nullptr;
   dead->nir = &n_dead;

   agx_apple9_block_begin(&p, entry);
   uint32_t gpr_x = emit_imm(&p, 1);
   /* Publication value marked via program.publication[]. */
   uint32_t pub_val = emit_imm(&p, 2);
   /* ALU instruction with inline uniform in slot 1 (only gpr_x in src[0]). */
   uint32_t add_u = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                        AGX_APPLE9_ENC_INT_ADD_EXTENDED,
                                        &gpr_x, 1, 0);
   agx_apple9_vir_instr *add_u_ins = p.instructions[p.instruction_count - 1];
   add_u_ins->alu_src_uniform_mask = 1u << 1;
   add_u_ins->alu_src_value[1] = 4;

   /* Side-effect consuming both add_u (GPR) and pub_val (publication). */
   const uint32_t pair_srcs[2] = {add_u, pub_val};
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_DEVICE_STORE, AGX_APPLE9_ENC_DEVICE_STORE,
      pair_srcs, 2, 0));
   agx_apple9_vir_instr *use_ins = p.instructions[p.instruction_count - 1];

   agx_apple9_block_begin(&p, inf);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_ANY,
                                          AGX_APPLE9_ENC_JMP_EXEC_ANY, inf));

   agx_apple9_block_begin(&p, synth);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_LOOP_MASK_POP, AGX_APPLE9_ENC_LOOP_MASK_POP,
      nullptr, 0, 0));

   agx_apple9_block_begin(&p, dead);
   /* Uses gpr_x in an unreachable block; must NOT keep gpr_x live past entry. */
   uint32_t dead_use = emit_iadd(&p, gpr_x, gpr_x);
   agx_apple9_vir_instr *dead_ins = p.instructions[p.instruction_count - 1];
   dead_ins->kill_mask = 0x3u; /* Stale kill_mask must be cleared. */
   (void)dead_use;

   p.publication = (bool *)calloc(p.value_count, sizeof(bool));
   ASSERT_NE(p.publication, nullptr);
   p.publication_capacity = p.value_count;
   p.publication[pub_val] = true;

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   EXPECT_TRUE(entry->cfg_active);
   EXPECT_TRUE(inf->cfg_active);
   EXPECT_FALSE(synth->cfg_active);
   EXPECT_FALSE(dead->cfg_active);

   apple9_ra_liveness_info live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &live));

   /* pub_val is skipped everywhere: kill bit 1 is 0, peak GPRs is 1 (not 2!). */
   EXPECT_TRUE(agx_apple9_vir_src_killed(add_u_ins, 0));
   EXPECT_TRUE(agx_apple9_vir_src_killed(use_ins, 0));
   EXPECT_FALSE(agx_apple9_vir_src_killed(use_ins, 1));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_out[entry->cfg_index], gpr_x));
   EXPECT_FALSE(u_sparse_bitset_test(&live.live_in[entry->cfg_index], pub_val));
   EXPECT_EQ(live.peak_live_gprs, 1u);
   EXPECT_EQ(dead_ins->kill_mask, 0u);

   apple9_ra_liveness_finish(&live);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaLiveness, SplitTupleAccountsFullWidthAndKillMask)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t a = emit_imm(&p, 1);
   uint32_t b = emit_imm(&p, 2);
   uint32_t c = emit_imm(&p, 3);
   const uint32_t elems[3] = {a, b, c};
   uint32_t tup = agx_apple9_vir_emit_collect(&p, elems, 3);
   uint32_t sp = agx_apple9_vir_emit_split(&p, tup, 3);
   agx_apple9_vir_instr *split_ins = p.instructions[p.instruction_count - 1];
   uint32_t sum = emit_iadd(&p, sp + 0, sp + 2);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, sum));

   apple9_ra_liveness_info live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &live));

   EXPECT_TRUE(agx_apple9_vir_src_killed(split_ins, 0));
   /* idx + 3 tuple elements (a,b,c / tup[0..2]) = 4 live 32-bit GPRs. */
   EXPECT_EQ(live.peak_live_gprs, 4u);

   apple9_ra_liveness_finish(&live);
   agx_apple9_vir_finish(&p);
}

/* =========================================================================
 * 3. Native VIR Pressure / Latency / Resource DAG Scheduler Tests
 * ========================================================================= */

TEST(Apple9RaPressureSchedule, ReducesPeakGprsOnHighPressureBinaryTree)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);

   /* Emit all 32 leaves first (breadth-first), creating peak pressure >= 32. */
   std::vector<uint32_t> level;
   for (unsigned i = 0; i < 32; ++i)
      level.push_back(emit_imm(&p, i + 1));

   while (level.size() > 1) {
      std::vector<uint32_t> next;
      for (size_t i = 0; i < level.size(); i += 2)
         next.push_back(emit_iadd(&p, level[i], level[i + 1]));
      level = std::move(next);
   }
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, level[0]));

   apple9_ra_liveness_info before;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before));
   EXPECT_GE(before.peak_live_gprs, 32u);
   apple9_ra_liveness_finish(&before);

   ASSERT_TRUE(apple9_ra_pressure_schedule(&p));

   apple9_ra_liveness_info after;
   ASSERT_TRUE(apple9_ra_liveness(&p, &after));
   /* Depth-first schedule needs at most log2(32) + 1 + idx = 7 GPRs. */
   EXPECT_LE(after.peak_live_gprs, 7u);
   apple9_ra_liveness_finish(&after);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, ReducesPeakGprsOnBatchedLoadsAndConsumers)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t k = emit_imm(&p, 3);
   std::vector<uint32_t> loads;
   for (unsigned i = 0; i < 16; ++i)
      loads.push_back(emit_dev_load(&p, i, idx, 1));

   uint32_t acc = emit_iadd(&p, loads[0], k);
   for (unsigned i = 1; i < 16; ++i) {
      uint32_t t = emit_iadd(&p, loads[i], k);
      acc = emit_iadd(&p, acc, t);
   }
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, acc));

   apple9_ra_liveness_info before;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before));
   EXPECT_GE(before.peak_live_gprs, 17u);
   apple9_ra_liveness_finish(&before);

   ASSERT_TRUE(apple9_ra_pressure_schedule(&p));

   apple9_ra_liveness_info after;
   ASSERT_TRUE(apple9_ra_liveness(&p, &after));
   EXPECT_LE(after.peak_live_gprs, 5u);
   apple9_ra_liveness_finish(&after);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, PreservesMemoryBarrierTileAtomicAndCfOrder)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n0 = {}, n1 = {};
   n0.successors[0] = &n1;

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   ASSERT_TRUE(b0 && b1);
   b0->nir = &n0;
   b1->nir = &n1;

   agx_apple9_block_begin(&p, b0);
   /* Pinned entry register must remain at position 0. */
   uint32_t entry_v = agx_apple9_vir_emit_entry_reg(&p, 8);
   agx_apple9_vir_instr *entry_ins = p.instructions[p.instruction_count - 1];

   uint32_t ld1 = emit_dev_load(&p, 0, entry_v, 1);
   agx_apple9_vir_instr *ld1_ins = p.instructions[p.instruction_count - 1];
   uint32_t u1 = emit_iadd(&p, ld1, entry_v);

   ASSERT_TRUE(emit_dev_store(&p, 0, entry_v, u1));
   agx_apple9_vir_instr *st1_ins = p.instructions[p.instruction_count - 1];

   uint32_t ld2 = emit_dev_load(&p, 0, entry_v, 1);
   agx_apple9_vir_instr *ld2_ins = p.instructions[p.instruction_count - 1];

   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_WORKGROUP_BARRIER, AGX_APPLE9_ENC_WORKGROUP_BARRIER,
      nullptr, 0, 0));
   agx_apple9_vir_instr *barrier = p.instructions[p.instruction_count - 1];

   uint32_t ld3 = emit_dev_load(&p, 0, entry_v, 1);
   agx_apple9_vir_instr *ld3_ins = p.instructions[p.instruction_count - 1];

   /* Tile access -> tile load -> tile store must remain strictly ordered. */
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, nullptr, 0,
      0));
   agx_apple9_vir_instr *tile_acc = p.instructions[p.instruction_count - 1];
   uint32_t tld = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_TILE_LOAD,
                                      AGX_APPLE9_ENC_TILE_LOAD, nullptr, 0, 0);
   agx_apple9_vir_instr *tile_ld = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_TILE_STORE, AGX_APPLE9_ENC_TILE_STORE, &tld, 1, 0));
   agx_apple9_vir_instr *tile_st = p.instructions[p.instruction_count - 1];

   /* Returning DEVICE_ATOMIC + DEVICE_ATOMIC_RESULT must remain adjacent. */
   uint32_t atom_res = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &p, 1, entry_v, &ld2, 1, AGX_APPLE9_ATOMIC_ADD, false, &atom_res));
   agx_apple9_vir_instr *atom_ins = p.instructions[p.instruction_count - 2];
   agx_apple9_vir_instr *atom_rec = p.instructions[p.instruction_count - 1];

   uint32_t final_sum = emit_iadd(&p, ld3, atom_res);
   ASSERT_TRUE(emit_dev_store(&p, 2, entry_v, final_sum));
   agx_apple9_vir_instr *final_st = p.instructions[p.instruction_count - 1];

   /* Pinned CF instruction at tail must stay last. */
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_NONE,
                                          AGX_APPLE9_ENC_JMP_EXEC_NONE, b1));
   agx_apple9_vir_instr *jmp = p.instructions[p.instruction_count - 1];

   agx_apple9_block_begin(&p, b1);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));

   /* Schedule with generous latency slack to aggressively test all hazards. */
   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 8,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   EXPECT_EQ(instr_pos_in_block(b0, entry_ins), 0u);
   EXPECT_LT(instr_pos_in_block(b0, ld1_ins), instr_pos_in_block(b0, st1_ins));
   EXPECT_LT(instr_pos_in_block(b0, st1_ins), instr_pos_in_block(b0, ld2_ins));
   EXPECT_LT(instr_pos_in_block(b0, ld2_ins), instr_pos_in_block(b0, barrier));
   EXPECT_LT(instr_pos_in_block(b0, barrier), instr_pos_in_block(b0, ld3_ins));
   EXPECT_LT(instr_pos_in_block(b0, tile_acc), instr_pos_in_block(b0, tile_ld));
   EXPECT_LT(instr_pos_in_block(b0, tile_ld), instr_pos_in_block(b0, tile_st));
   EXPECT_EQ(instr_pos_in_block(b0, atom_ins) + 1,
             instr_pos_in_block(b0, atom_rec));
   EXPECT_LT(instr_pos_in_block(b0, final_st), instr_pos_in_block(b0, jmp));
   EXPECT_EQ(agx_apple9_block_last_instr(b0), jmp);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, LatencyHidingRespectsPressureSlack)
{
   /* Part A: Separates a long-latency load from its consumer using independent
    * pure ALU work at zero pressure increase. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      ASSERT_NE(b0, nullptr);
      agx_apple9_block_begin(&p, b0);

      uint32_t idx = agx_apple9_vir_input(&p, 0);
      uint32_t ld = emit_dev_load(&p, 0, idx, 1);
      agx_apple9_vir_instr *ld_ins = p.instructions[p.instruction_count - 1];
      uint32_t use = emit_iadd(&p, ld, idx);
      agx_apple9_vir_instr *use_ins = p.instructions[p.instruction_count - 1];

      /* Independent pure chain that can fill the load latency window. */
      uint32_t x = idx;
      for (unsigned i = 0; i < 5; ++i)
         x = emit_iadd(&p, x, idx);

      uint32_t out = emit_iadd(&p, use, x);
      ASSERT_TRUE(emit_dev_store(&p, 1, idx, out));

      ASSERT_TRUE(apple9_ra_pressure_schedule(&p));
      unsigned p_ld = instr_pos_in_block(b0, ld_ins);
      unsigned p_use = instr_pos_in_block(b0, use_ins);
      EXPECT_LT(p_ld + 1, p_use);
      verify_ssa_def_before_use_in_block(&p, b0);
      agx_apple9_vir_finish(&p);
   }

   /* Part B: Two sequential load+tile_store pairs overlap iff pressure_slack > 0
    * (overlapping keeps both loads live simultaneously: +1 32-bit GPR). */
   for (unsigned slack : {0u, 2u}) {
      SCOPED_TRACE(slack);
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      agx_apple9_block *b0 = agx_apple9_block_create(&p);
      ASSERT_NE(b0, nullptr);
      agx_apple9_block_begin(&p, b0);

      uint32_t idx = agx_apple9_vir_input(&p, 0);
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, idx));
      uint32_t l0 = emit_dev_load(&p, 0, idx, 1);
      agx_apple9_vir_instr *l0_ins = p.instructions[p.instruction_count - 1];
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_TILE_STORE, AGX_APPLE9_ENC_TILE_STORE, &l0, 1, 0));
      agx_apple9_vir_instr *u0_ins = p.instructions[p.instruction_count - 1];

      uint32_t l1 = emit_dev_load(&p, 1, idx, 1);
      agx_apple9_vir_instr *l1_ins = p.instructions[p.instruction_count - 1];
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_TILE_STORE, AGX_APPLE9_ENC_TILE_STORE, &l1, 1, 0));
      agx_apple9_vir_instr *u1_ins = p.instructions[p.instruction_count - 1];

      apple9_ra_sched_options opts = {
         .latency = true,
         .pressure_slack = slack,
         .rejected_pressure_increase = 0,
      };
      ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

      unsigned p_l0 = instr_pos_in_block(b0, l0_ins);
      unsigned p_u0 = instr_pos_in_block(b0, u0_ins);
      unsigned p_l1 = instr_pos_in_block(b0, l1_ins);
      unsigned p_u1 = instr_pos_in_block(b0, u1_ins);
      EXPECT_LT(p_l0, p_u0);
      EXPECT_LT(p_l1, p_u1);

      bool overlapped = std::max(p_l0, p_l1) < std::min(p_u0, p_u1);
      EXPECT_EQ(overlapped, slack != 0);
      if (slack == 0) {
         EXPECT_GT(opts.rejected_pressure_increase, 0u);
      }

      verify_ssa_def_before_use_in_block(&p, b0);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9RaPressureSchedule, RespectsLowBankR0R15ConstrainedLifetimes)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t idx = agx_apple9_vir_input(&p, 0);

   /* Emit 18 r0..r15-constrained MOV_IMM_COMPACT values up front, exceeding the
    * 16-register low bank if all kept live simultaneously! */
   std::vector<uint32_t> low_vals;
   for (unsigned i = 0; i < 18; ++i) {
      uint32_t v = emit_imm(&p, i + 1, AGX_APPLE9_ENC_MOV_IMM_COMPACT);
      low_vals.push_back(v);
   }

   uint32_t acc = emit_iadd(&p, low_vals[0], idx);
   for (unsigned i = 1; i < 18; ++i)
      acc = emit_iadd(&p, acc, low_vals[i]);
   ASSERT_TRUE(emit_dev_store(&p, 0, idx, acc));

   apple9_ra_liveness_info before;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before));
   EXPECT_GE(before.peak_live_gprs, 18u);
   apple9_ra_liveness_finish(&before);

   ASSERT_TRUE(apple9_ra_pressure_schedule(&p));

   /* Recompute peak live r0..r15 values after scheduling: must fit easily
    * inside the 16-register low bank (<= 2 live at once!). */
   std::vector<bool> is_low(p.value_count, false);
   for (uint32_t v : low_vals)
      is_low[v] = true;

   std::vector<bool> live(p.value_count, false);
   unsigned cur_low = 0, peak_low = 0;
   for (agx_apple9_vir_instr *I = agx_apple9_block_last_instr(b0); I;
        I = agx_apple9_instr_prev(I)) {
      if (I->dest != AGX_APPLE9_VREG_INVALID) {
         for (unsigned c = 0; c < I->dest_components; ++c) {
            uint32_t d = I->dest + c;
            if (d < p.value_count && live[d]) {
               live[d] = false;
               if (is_low[d])
                  --cur_low;
            }
         }
      }
      const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t v = agx_apple9_vir_src(I, s);
         if (v < p.value_count && !live[v]) {
            live[v] = true;
            if (is_low[v])
               ++cur_low;
         }
      }
      peak_low = std::max(peak_low, cur_low);
   }

   EXPECT_LE(peak_low, 2u);
   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, LatencySlackDoesNotExceedR0R15BankCeiling)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   /* 15 live-out r0..r15 values (max_phys = 15) stay live across two
    * sequential r0..r15-constrained load+tile_store pairs. Sequential order
    * peaks at 16 r0..r15 values (legal); overlapping both loads would require
    * 17 r0..r15 values (overflowing r0..r15!). Even with pressure_slack = 4,
    * the scheduler must NOT overlap them past the 16-register r0..r15 ceiling. */
   uint32_t idx = agx_apple9_vir_input(&p, 0);
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, idx));
   p.max_phys[idx] = 15;
   for (unsigned i = 0; i < 14; ++i) {
      uint32_t v = agx_apple9_vir_input(&p, i + 1);
      p.max_phys[v] = 15;
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, v));
   }

   uint32_t l0 = emit_dev_load(&p, 0, idx, 1);
   p.max_phys[l0] = 15;
   agx_apple9_vir_instr *l0_ins = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_TILE_STORE, AGX_APPLE9_ENC_TILE_STORE, &l0, 1, 0));
   agx_apple9_vir_instr *u0_ins = p.instructions[p.instruction_count - 1];

   uint32_t l1 = emit_dev_load(&p, 1, idx, 1);
   p.max_phys[l1] = 15;
   agx_apple9_vir_instr *l1_ins = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_TILE_STORE, AGX_APPLE9_ENC_TILE_STORE, &l1, 1, 0));
   agx_apple9_vir_instr *u1_ins = p.instructions[p.instruction_count - 1];

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 4,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned p_l0 = instr_pos_in_block(b0, l0_ins);
   unsigned p_u0 = instr_pos_in_block(b0, u0_ins);
   unsigned p_l1 = instr_pos_in_block(b0, l1_ins);
   unsigned p_u1 = instr_pos_in_block(b0, u1_ins);
   bool overlapped = std::max(p_l0, p_l1) < std::min(p_u0, p_u1);
   EXPECT_FALSE(overlapped);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, AutoTierSlackOverlapsLoadsWithoutDroppingTier)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   /* With auto_tier_slack = true, two independent DEVICE_LOAD + consumer pairs
    * inside a low-pressure block (well below the 52-GPR 1024-thread tier) must
    * overlap their loads instead of snapping each consumer to dist=1 behind
    * its load for a 1-GPR intra-tier pressure savings. */
   uint32_t idx = agx_apple9_vir_input(&p, 0);
   uint32_t c0 = agx_apple9_vir_input(&p, 1);

   uint32_t l0 = emit_dev_load(&p, 0, idx, 1);
   agx_apple9_vir_instr *l0_ins = p.instructions[p.instruction_count - 1];
   uint32_t u0 = emit_iadd(&p, l0, c0);
   agx_apple9_vir_instr *u0_ins = p.instructions[p.instruction_count - 1];

   uint32_t l1 = emit_dev_load(&p, 1, idx, 1);
   agx_apple9_vir_instr *l1_ins = p.instructions[p.instruction_count - 1];
   uint32_t u1 = emit_iadd(&p, l1, c0);
   agx_apple9_vir_instr *u1_ins = p.instructions[p.instruction_count - 1];

   uint32_t sum = emit_iadd(&p, u0, u1);
   ASSERT_TRUE(emit_dev_store(&p, 2, idx, sum));

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned p_l0 = instr_pos_in_block(b0, l0_ins);
   unsigned p_u0 = instr_pos_in_block(b0, u0_ins);
   unsigned p_l1 = instr_pos_in_block(b0, l1_ins);
   unsigned p_u1 = instr_pos_in_block(b0, u1_ins);
   bool overlapped = std::max(p_l0, p_l1) < std::min(p_u0, p_u1);
   EXPECT_TRUE(overlapped);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, RematLongSpanDerivedAluDropsHighBlockPressure)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   /* Create 20 single-source integer ALU offsets derived from one base GPR,
    * use them once early, separate the second use by a 60-instruction chain of
    * unrelated live values (driving peak live GPRs well past 52), and add one
    * uniform-reading derivation that crosses a STORE_UNIFORM barrier (must NOT
    * be rematerialized past STORE_UNIFORM). */
   uint32_t base = agx_apple9_vir_input(&p, 0);
   uint32_t acc = emit_imm(&p, 1);

   std::vector<uint32_t> derived;
   for (unsigned k = 0; k < 20; ++k) {
      uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                       AGX_APPLE9_ENC_INT_ADD_EXTENDED, &base,
                                       1, k + 1);
      p.instructions[p.instruction_count - 1]->alu_src_immediate_mask = 1u << 1;
      p.instructions[p.instruction_count - 1]->alu_src_value[1] = k + 1;
      derived.push_back(v);
      uint32_t ld = emit_dev_load(&p, 0, v, 1);
      acc = emit_iadd(&p, acc, ld);
   }

   uint32_t unif_derived = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, &base, 1, 0);
   p.instructions[p.instruction_count - 1]->alu_src_uniform_mask = 1u << 1;
   p.instructions[p.instruction_count - 1]->alu_src_value[1] = 4;
   acc = emit_iadd(&p, acc, unif_derived);

   /* 40 simultaneously live values across a middle region plus a STORE_UNIFORM
    * barrier. */
   std::vector<uint32_t> mid;
   for (unsigned k = 0; k < 40; ++k) {
      uint32_t x = emit_dev_load(&p, 1, acc, 1);
      mid.push_back(x);
      acc = emit_iadd(&p, acc, x);
   }
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM, &acc, 1,
      4));
   for (uint32_t x : mid)
      acc = emit_iadd(&p, acc, x);

   /* Distant reuse of all 20 derived values + the uniform-reading derivation. */
   agx_apple9_vir_instr *unif_late_use = nullptr;
   for (uint32_t v : derived) {
      uint32_t ld = emit_dev_load(&p, 2, v, 1);
      acc = emit_iadd(&p, acc, ld);
   }
   acc = emit_iadd(&p, acc, unif_derived);
   unif_late_use = p.instructions[p.instruction_count - 1];
   ASSERT_TRUE(emit_dev_store(&p, 3, base, acc));

   apple9_ra_liveness_info before_live;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before_live));
   unsigned peak_before = p.peak_live_gprs;
   apple9_ra_liveness_finish(&before_live);
   ASSERT_GT(peak_before, 60u);

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   /* Peak pressure must drop substantially once the 20 long-span derivations
    * are rematerialized at the distant cluster, while the uniform-reading
    * derivation must remain the original SSA value across STORE_UNIFORM. */
   EXPECT_LE(p.peak_live_gprs + 15u, peak_before);
   EXPECT_EQ(agx_apple9_vir_src(unif_late_use, 1), unif_derived);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, AutoTierSlackCompressesSub45TierEvenWithCallerSlack)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   /* Build a block with 37 live values plus 12 independent early DEVICE_LOADs
    * whose original order peaks at >44 GPRs (832-thread floor tier), even
    * though sinking/serializing the loads keeps the entire block at <= 40 GPRs
    * (960-thread G16G tier). Even if the caller passes a nonzero pressure_slack,
    * auto_tier_slack must compress the block into the higher real G16G tier. */
   uint32_t base = agx_apple9_vir_input(&p, 0);
   std::vector<uint32_t> keep;
   for (unsigned i = 0; i < 36; ++i) {
      uint32_t v = agx_apple9_vir_input(&p, i + 1);
      keep.push_back(v);
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, v));
   }

   std::vector<uint32_t> loads;
   for (unsigned i = 0; i < 12; ++i)
      loads.push_back(emit_dev_load(&p, i % 4, base, 1));

   uint32_t acc = emit_iadd(&p, keep[0], loads[0]);
   for (unsigned i = 1; i < 12; ++i)
      acc = emit_iadd(&p, acc, loads[i]);
   ASSERT_TRUE(emit_dev_store(&p, 0, base, acc));

   apple9_ra_liveness_info before;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before));
   ASSERT_GT(p.peak_live_gprs, 44u);
   apple9_ra_liveness_finish(&before);

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 10,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));
   EXPECT_LE(p.peak_live_gprs, 40u);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, AutoTierSlackPreservesOverlapInFlat45To96Band)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   /* 50 live values + 4 independent DEVICE_LOADs: both 52 and 56 GPRs sit on
    * G16G's flat 45..96 GPR shelf (832 thr/core), so auto_tier_slack must NOT
    * sink loads to chase the obsolete G13 52-GPR cliff. */
   uint32_t base = agx_apple9_vir_input(&p, 0);
   std::vector<uint32_t> keep;
   for (unsigned i = 0; i < 49; ++i) {
      uint32_t v = agx_apple9_vir_input(&p, i + 1);
      keep.push_back(v);
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, v));
   }

   agx_apple9_vir_instr *ld_ins[4];
   agx_apple9_vir_instr *use_ins[4];
   uint32_t acc = keep[0];
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t ld = emit_dev_load(&p, i, base, 1);
      ld_ins[i] = p.instructions[p.instruction_count - 1];
      acc = emit_iadd(&p, acc, ld);
      use_ins[i] = p.instructions[p.instruction_count - 1];
   }
   ASSERT_TRUE(emit_dev_store(&p, 0, base, acc));

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned max_ld = 0, min_use = UINT_MAX;
   for (unsigned i = 0; i < 4; ++i) {
      max_ld = std::max(max_ld, instr_pos_in_block(b0, ld_ins[i]));
      min_use = std::min(min_use, instr_pos_in_block(b0, use_ins[i]));
   }
   EXPECT_LT(max_ld, min_use);
   EXPECT_LE(p.peak_live_gprs, 60u);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, FivePipeDualIssueInterleavesAluImadSfu)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t a = agx_apple9_vir_input(&p, 0);
   uint32_t b = agx_apple9_vir_input(&p, 1);
   uint32_t c = agx_apple9_vir_input(&p, 2);

   /* Emit clustered independent MAIN (FMA), IMUL (IMAD), and SFU (FRSQ) ops
    * that all feed independent live outputs. 5-pipe dual-issue scheduling must
    * interleave distinct pipes instead of keeping all ops of each pipe bunched. */
   std::vector<uint32_t> outs;
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t srcs[3] = {a, b, c};
      uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMA,
                                       AGX_APPLE9_ENC_FLOAT3_EXTENDED, srcs, 3, 0);
      outs.push_back(v);
   }
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t srcs[3] = {a, b, c};
      uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMAD,
                                       AGX_APPLE9_ENC_INT_MAD_EXTENDED, srcs, 3, 0);
      outs.push_back(v);
   }
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FRSQ,
                                       AGX_APPLE9_ENC_FLOAT_SPECIAL, &a, 1, 0);
      outs.push_back(v);
   }
   for (uint32_t v : outs)
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, v));

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned transitions = 0;
   enum apple9_ra_exec_pipe prev = APPLE9_RA_PIPE_NONE;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b0); I;
        I = agx_apple9_instr_next(I)) {
      enum apple9_ra_exec_pipe pipe = apple9_ra_exec_pipe(I);
      if (pipe == APPLE9_RA_PIPE_NONE)
         continue;
      if (prev != APPLE9_RA_PIPE_NONE && pipe != prev)
         transitions++;
      prev = pipe;
   }
   /* Original clustered order had only 2 pipe transitions (MAIN->IMUL->SFU);
    * interleaved order should switch pipes on nearly every instruction. */
   EXPECT_GE(transitions, 8u);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, CapsInFlightDeviceLoadsAtFourWave)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t base = agx_apple9_vir_input(&p, 0);
   uint32_t acc = agx_apple9_vir_input(&p, 1);

   /* 8 independent DEVICE_LOAD + consumer pairs in a low-pressure block:
    * scheduler should overlap loads into N=4 waves rather than bursting all 8
    * in flight simultaneously (avoiding L2 MSHR thrashing and slot sharing). */
   std::vector<uint32_t> load_vals;
   for (unsigned i = 0; i < 8; ++i) {
      uint32_t ld = emit_dev_load(&p, i % 4, base, 1);
      load_vals.push_back(ld);
      acc = emit_iadd(&p, acc, ld);
   }
   ASSERT_TRUE(emit_dev_store(&p, 0, base, acc));

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned inflight = 0, max_inflight = 0;
   std::vector<bool> pending(8, false);
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b0); I;
        I = agx_apple9_instr_next(I)) {
      for (unsigned i = 0; i < 8; ++i) {
         if (I->dest == load_vals[i]) {
            pending[i] = true;
            inflight++;
            max_inflight = std::max(max_inflight, inflight);
         }
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         for (unsigned i = 0; i < 8; ++i) {
            if (pending[i] && I->src[s] == load_vals[i]) {
               pending[i] = false;
               inflight--;
            }
         }
      }
   }
   EXPECT_GE(max_inflight, 2u);
   EXPECT_LE(max_inflight, 4u);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, CapsMultiLoadConsumerAtFourInFlightLoads)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   ASSERT_NE(b0, nullptr);
   agx_apple9_block_begin(&p, b0);

   uint32_t base = agx_apple9_vir_input(&p, 0);
   uint32_t seed = agx_apple9_vir_input(&p, 1);

   /* Two independent 3-load FMA groups (6 DEVICE_LOADs total) emitted
    * back-to-back with their consumers, followed by independent ALU work so
    * latency scheduling is accepted. When 3 loads are already in flight,
    * opening the second 3-load FMA would jump to 6 > 4; projected load count
    * must enforce max_inflight <= 4. */
   uint32_t v[6];
   v[0] = emit_dev_load(&p, 0, base, 1);
   v[1] = emit_dev_load(&p, 1, base, 1);
   v[2] = emit_dev_load(&p, 2, base, 1);
   uint32_t s0[3] = {v[0], v[1], v[2]};
   uint32_t f0 = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMA,
                                     AGX_APPLE9_ENC_FLOAT3_EXTENDED, s0, 3, 0);

   v[3] = emit_dev_load(&p, 0, base, 1);
   v[4] = emit_dev_load(&p, 1, base, 1);
   v[5] = emit_dev_load(&p, 2, base, 1);
   uint32_t s1[3] = {v[3], v[4], v[5]};
   uint32_t f1 = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMA,
                                     AGX_APPLE9_ENC_FLOAT3_EXTENDED, s1, 3, 0);

   uint32_t alu = seed;
   for (unsigned i = 0; i < 6; ++i)
      alu = emit_iadd(&p, alu, base);

   uint32_t acc = emit_iadd(&p, f0, f1);
   acc = emit_iadd(&p, acc, alu);
   ASSERT_TRUE(emit_dev_store(&p, 0, base, acc));

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   unsigned inflight = 0, max_inflight = 0;
   bool pending[6] = {};
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b0); I;
        I = agx_apple9_instr_next(I)) {
      for (unsigned i = 0; i < 6; ++i) {
         if (I->dest == v[i]) {
            pending[i] = true;
            inflight++;
            max_inflight = std::max(max_inflight, inflight);
         }
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         for (unsigned i = 0; i < 6; ++i) {
            if (pending[i] && I->src[s] == v[i]) {
               pending[i] = false;
               inflight--;
            }
         }
      }
   }
   EXPECT_GE(max_inflight, 3u);
   EXPECT_LE(max_inflight, 4u);

   verify_ssa_def_before_use_in_block(&p, b0);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaPressureSchedule, CrossBlockRematPreservesPhiOperands)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   /* Build diamond CFG: b0 -> {b1, b2} -> b3. */
   nir_block n0 = {}, n1 = {}, n2 = {}, n3 = {};
   n0.successors[0] = &n1;
   n0.successors[1] = &n2;
   n1.successors[0] = &n3;
   n2.successors[0] = &n3;

   agx_apple9_block *b0 = agx_apple9_block_create(&p);
   agx_apple9_block *b1 = agx_apple9_block_create(&p);
   agx_apple9_block *b2 = agx_apple9_block_create(&p);
   agx_apple9_block *b3 = agx_apple9_block_create(&p);
   ASSERT_TRUE(b0 && b1 && b2 && b3);
   b0->nir = &n0;
   b1->nir = &n1;
   b2->nir = &n2;
   b3->nir = &n3;

   agx_apple9_block_begin(&p, b0);
   uint32_t base = agx_apple9_vir_input(&p, 0);
   std::vector<uint32_t> carry;
   for (unsigned i = 0; i < 42; ++i)
      carry.push_back(agx_apple9_vir_input(&p, i + 1));

   /* Emit 10 derived 1-source integer offsets in b0 used only in b1, plus a
    * small compact immediate used as a VIR_PHI operand in b3 (which MUST NOT
    * be deleted or invalidated by cross-block remat). */
   std::vector<uint32_t> derived;
   for (unsigned k = 0; k < 10; ++k) {
      uint32_t v = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                       AGX_APPLE9_ENC_INT_ADD_EXTENDED, &base,
                                       1, k + 1);
      p.instructions[p.instruction_count - 1]->alu_src_immediate_mask = 1u << 1;
      p.instructions[p.instruction_count - 1]->alu_src_value[1] = k + 1;
      derived.push_back(v);
   }
   uint32_t phi_zero = emit_imm(&p, 0, AGX_APPLE9_ENC_MOV_IMM_COMPACT);

   agx_apple9_block_begin(&p, b1);
   uint32_t left = base;
   for (uint32_t v : derived)
      left = emit_iadd(&p, left, v);

   agx_apple9_block_begin(&p, b2);
   uint32_t right = emit_imm(&p, 7, AGX_APPLE9_ENC_MOV_IMM_COMPACT);

   agx_apple9_block_begin(&p, b3);
   uint32_t merged = agx_apple9_vir_emit_phi_canonical(&p, b3, 2);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   uint32_t sum = emit_iadd(&p, merged, right);
   for (uint32_t v : carry)
      sum = emit_iadd(&p, sum, v);
   ASSERT_TRUE(emit_dev_store(&p, 0, base, sum));

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   unsigned p1 = agx_apple9_block_pred_index(b3, b1);
   unsigned p2 = agx_apple9_block_pred_index(b3, b2);
   ASSERT_NE(p1, UINT_MAX);
   ASSERT_NE(p2, UINT_MAX);
   *agx_apple9_vir_src_ptr(phi, p1) = left;
   *agx_apple9_vir_src_ptr(phi, p2) = phi_zero;

   apple9_ra_liveness_info before;
   ASSERT_TRUE(apple9_ra_liveness(&p, &before));
   unsigned peak_before = p.peak_live_gprs;
   apple9_ra_liveness_finish(&before);
   ASSERT_GT(peak_before, 52u);

   apple9_ra_sched_options opts = {
      .latency = true,
      .pressure_slack = 0,
      .auto_tier_slack = true,
      .rejected_pressure_increase = 0,
   };
   ASSERT_TRUE(apple9_ra_pressure_schedule_opts(&p, &opts));

   /* Cross-block remat must compress peak live GPRs across b0 while keeping
    * phi_zero alive and valid for the VIR_PHI in b3. */
   EXPECT_LT(p.peak_live_gprs, peak_before);
   EXPECT_EQ(agx_apple9_vir_src(phi, p2), phi_zero);
   apple9_ra_validate_error verr = {};
   EXPECT_TRUE(apple9_ra_validate_ssa(&p, &verr)) << verr.message;

   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSched, UnifiedOperandGroupsAndLatencyTable)
{
   /* Exact G16G (T8132 @ 1470 MHz) calibrated latencies. */
   struct agx_apple9_vir_instr ins = {};
   ins.encoding = AGX_APPLE9_ENC_TEXTURE_SAMPLE;
   ins.op = AGX_APPLE9_VIR_TEXTURE_SAMPLE;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 48u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MEM);

   ins.encoding = AGX_APPLE9_ENC_DEVICE_LOAD;
   ins.op = AGX_APPLE9_VIR_DEVICE_LOAD;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 36u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MEM);

   ins.encoding = AGX_APPLE9_ENC_SHARED_ATOMIC;
   ins.op = AGX_APPLE9_VIR_SHARED_ATOMIC;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 24u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MEM);

   ins.encoding = AGX_APPLE9_ENC_TILE_LOAD;
   ins.op = AGX_APPLE9_VIR_TILE_LOAD;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 24u);

   ins.encoding = AGX_APPLE9_ENC_MMA_F32;
   ins.op = AGX_APPLE9_VIR_MMA;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 18u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MAIN);

   ins.encoding = AGX_APPLE9_ENC_INT_MUL_WIDE;
   ins.op = AGX_APPLE9_VIR_IMUL_WIDE;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 7u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_IMUL);

   ins.encoding = AGX_APPLE9_ENC_FLOAT_SPECIAL;
   ins.op = AGX_APPLE9_VIR_HRCP;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 6u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_SFU);

   ins.op = AGX_APPLE9_VIR_HLOG2;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 5u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_SFU);

   ins.encoding = AGX_APPLE9_ENC_INT_MAD_EXTENDED;
   ins.op = AGX_APPLE9_VIR_IMAD;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 5u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_IMUL);

   ins.encoding = AGX_APPLE9_ENC_SHIFT_EXTENDED;
   ins.op = AGX_APPLE9_VIR_ISHL;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 5u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_SHIFT);

   ins.encoding = AGX_APPLE9_ENC_GET_SR_EXTENDED;
   ins.op = AGX_APPLE9_VIR_GET_SR;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 5u);
   ins.encoding = AGX_APPLE9_ENC_GET_SR_ZEXT16;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 2u);

   ins.encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE;
   ins.op = AGX_APPLE9_VIR_SELECT;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 3u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MAIN);

   ins.encoding = AGX_APPLE9_ENC_FLOAT2_BASE;
   ins.op = AGX_APPLE9_VIR_FADD;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 2u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_MAIN);

   ins.encoding = AGX_APPLE9_ENC_PSEUDO;
   ins.op = AGX_APPLE9_VIR_SPLIT;
   EXPECT_EQ(apple9_ra_dependency_delay(&ins), 1u);
   EXPECT_EQ(apple9_ra_exec_pipe(&ins), APPLE9_RA_PIPE_NONE);

   /* Async & reads-sources-at-issue classification. */
   ins = {};
   ins.op = AGX_APPLE9_VIR_DEVICE_LOAD;
   ins.encoding = AGX_APPLE9_ENC_DEVICE_LOAD;
   ins.dest_components = 1;
   EXPECT_TRUE(apple9_ra_async(&ins));
   EXPECT_TRUE(apple9_ra_reads_sources_at_issue(&ins));
   EXPECT_TRUE(apple9_ra_dest_group(&ins).reuse_killed);

   ins = {};
   ins.op = AGX_APPLE9_VIR_TEXTURE_SAMPLE;
   ins.encoding = AGX_APPLE9_ENC_TEXTURE_SAMPLE;
   ins.dest_components = 4;
   ins.nr_srcs = 3;
   ins.texture_bindless = false;
   ins.sampler_bindless = false;
   struct apple9_ra_operand_group tex_dst = apple9_ra_dest_group(&ins);
   EXPECT_EQ(tex_dst.width, 4u);
   EXPECT_EQ(tex_dst.max_gpr, 60u);
   EXPECT_FALSE(tex_dst.reuse_killed);

   ins.texture_bindless = true;
   ins.sampler_bindless = true;
   ins.nr_srcs = 4;
   tex_dst = apple9_ra_dest_group(&ins);
   EXPECT_EQ(tex_dst.max_gpr, AGX_APPLE9_GPR_COUNT - 4u);
   struct apple9_ra_operand_group bindless_tex = apple9_ra_source_group(&ins, 2);
   struct apple9_ra_operand_group bindless_smp = apple9_ra_source_group(&ins, 3);
   ASSERT_NE(bindless_tex.constraint, nullptr);
   ASSERT_NE(bindless_smp.constraint, nullptr);
   EXPECT_EQ(bindless_tex.constraint->role, AGX_APPLE9_OPERAND_SRC4);
   EXPECT_EQ(bindless_smp.constraint->role, AGX_APPLE9_OPERAND_SRC5);
   EXPECT_EQ(bindless_tex.max_gpr, AGX_APPLE9_GPR_COUNT - 1u);

   /* Indirect device store: data tuple (w=2), index (w=1), pointer pair (w=2). */
   ins = {};
   ins.op = AGX_APPLE9_VIR_DEVICE_STORE;
   ins.encoding = AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
   ins.memory_components = 2;
   ins.memory_bits = 32;
   ins.nr_srcs = 5;
   struct apple9_ra_operand_group data_g = apple9_ra_source_group(&ins, 0);
   struct apple9_ra_operand_group idx_g = apple9_ra_source_group(&ins, 2);
   struct apple9_ra_operand_group ptr_g = apple9_ra_source_group(&ins, 3);
   EXPECT_EQ(data_g.width, 2u);
   EXPECT_EQ(data_g.max_gpr, AGX_APPLE9_GPR_COUNT - 2u);
   ASSERT_NE(data_g.constraint, nullptr);
   EXPECT_EQ(data_g.constraint->role, AGX_APPLE9_OPERAND_STORE_DATA);
   EXPECT_EQ(idx_g.width, 1u);
   ASSERT_NE(idx_g.constraint, nullptr);
   EXPECT_EQ(idx_g.constraint->role, AGX_APPLE9_OPERAND_INDEX);
   EXPECT_EQ(ptr_g.width, 2u);
   ASSERT_NE(ptr_g.constraint, nullptr);
   EXPECT_EQ(ptr_g.constraint->role, AGX_APPLE9_OPERAND_SRC1);
   EXPECT_EQ(ptr_g.max_gpr, 94u);
   /* Interior lane of indirect pointer pair resolves to SRC1 as well. */
   ASSERT_NE(apple9_ra_source_constraint(&ins, 4), nullptr);
   EXPECT_EQ(apple9_ra_source_constraint(&ins, 4)->role, AGX_APPLE9_OPERAND_SRC1);

   /* Private store multi-word data tuple respects 64-GPR bank ceiling. */
   ins = {};
   ins.op = AGX_APPLE9_VIR_PRIVATE_STORE;
   ins.encoding = AGX_APPLE9_ENC_PRIVATE_STORE;
   ins.memory_components = 4;
   ins.memory_bits = 32;
   ins.nr_srcs = 5;
   struct apple9_ra_operand_group priv_data = apple9_ra_source_group(&ins, 0);
   struct apple9_ra_operand_group priv_idx = apple9_ra_source_group(&ins, 4);
   EXPECT_EQ(priv_data.width, 4u);
   EXPECT_EQ(priv_data.max_gpr, 60u);
   EXPECT_TRUE(priv_data.clobber);
   EXPECT_EQ(priv_idx.width, 1u);
   EXPECT_EQ(priv_idx.max_gpr, 63u);
   EXPECT_FALSE(priv_idx.clobber);
}
