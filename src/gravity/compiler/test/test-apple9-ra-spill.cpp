/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "compiler/nir/nir.h"
#include "agx_apple9_ir.h"
#include "apple9_ra_spill.h"

namespace {

static agx_apple9_block *
make_cfg_block(agx_apple9_vir_program *p, nir_block *nb)
{
   memset(nb, 0, sizeof(*nb));
   agx_apple9_block *b = agx_apple9_block_create(p);
   assert(b != nullptr);
   b->nir = nb;
   return b;
}

static uint32_t
emit_imm32(agx_apple9_vir_program *p, uint32_t val)
{
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
                              nullptr, 0, val);
}

static uint32_t
emit_imm_compact(agx_apple9_vir_program *p, uint32_t val)
{
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IMM,
                              AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, val);
}

static uint32_t
emit_get_sr(agx_apple9_vir_program *p, uint32_t sel)
{
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_GET_SR, AGX_APPLE9_ENC_GET_SR,
                              nullptr, 0, sel);
}

static uint32_t
emit_ior_uniform(agx_apple9_vir_program *p, uint32_t uniform_word)
{
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IOR_UNIFORM,
                              AGX_APPLE9_ENC_LOGIC_UNIFORM, nullptr, 0,
                              uniform_word);
}

static uint32_t
emit_iadd(agx_apple9_vir_program *p, uint32_t a, uint32_t b)
{
   uint32_t srcs[2] = {a, b};
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IADD,
                              AGX_APPLE9_ENC_INT_ADD_EXTENDED, srcs, 2, 0);
}

static uint32_t
emit_ixor(agx_apple9_vir_program *p, uint32_t a, uint32_t b)
{
   uint32_t srcs[2] = {a, b};
   return agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IXOR,
                              AGX_APPLE9_ENC_LOGIC_EXTENDED, srcs, 2, 0);
}

static void
verify_vir_ssa_and_classes(agx_apple9_vir_program *p,
                           const apple9_ra_spill_state *state)
{
   ASSERT_TRUE(agx_apple9_analyze_uses(p));

   for (agx_apple9_block *b = p->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      bool seen_non_phi = false;
      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI) {
            EXPECT_FALSE(seen_non_phi) << "VIR_PHI found after non-PHI";
            EXPECT_EQ(agx_apple9_vir_nr_srcs(I), b->num_cfg_preds);
            if (state) {
               bool dst_mem = apple9_ra_value_is_mem(state, I->dest);
               for (unsigned s = 0; s < agx_apple9_vir_nr_srcs(I); ++s) {
                  uint32_t sv = agx_apple9_vir_src(I, s);
                  if (sv != AGX_APPLE9_VREG_INVALID) {
                     EXPECT_EQ(apple9_ra_value_is_mem(state, sv), dst_mem)
                        << "PHI class mismatch on pred " << s;
                  }
               }
            }
            continue;
         }
         seen_non_phi = true;

         if (state && I->op != AGX_APPLE9_VIR_MOV) {
            if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
               EXPECT_FALSE(apple9_ra_value_is_mem(state, I->dest))
                  << "Non-MOV/PHI op " << I->op << " writes RA_MEM";
            }
            for (unsigned s = 0; s < agx_apple9_vir_nr_srcs(I); ++s) {
               uint32_t sv = agx_apple9_vir_src(I, s);
               if (sv != AGX_APPLE9_VREG_INVALID) {
                  EXPECT_FALSE(apple9_ra_value_is_mem(state, sv))
                     << "Non-MOV/PHI op " << I->op << " reads RA_MEM";
               }
            }
         }
      }
   }
}

/* Straight-line evaluator for spilled + colored VIR programs using an explicit
 * invocation-frame scratch array mem[] to check bit-exactness of spilling,
 * SSA repair, rematerialization, and DSATUR spill-home slot reuse! */
static uint32_t
eval_straight_line(const agx_apple9_vir_program *p,
                   const apple9_ra_spill_state *state,
                   const std::vector<uint32_t> &mem_slot,
                   const std::unordered_map<uint32_t, uint32_t> &uniforms,
                   const std::unordered_map<uint32_t, uint32_t> &sr_vals)
{
   std::vector<uint32_t> gpr(p->value_count, 0);
   std::vector<uint32_t> frame(APPLE9_RA_MAX_SPILL_WORDS, 0);

   auto read_val = [&](uint32_t v) -> uint32_t {
      if (apple9_ra_value_is_mem(state, v))
         return frame[mem_slot[v]];
      return gpr[v];
   };
   auto write_val = [&](uint32_t v, uint32_t x) {
      if (apple9_ra_value_is_mem(state, v))
         frame[mem_slot[v]] = x;
      else
         gpr[v] = x;
   };

   for (agx_apple9_block *b = p->blocks; b; b = b->next) {
      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            write_val(I->dest, I->immediate);
            break;
         case AGX_APPLE9_VIR_GET_SR:
            write_val(I->dest, sr_vals.at(I->immediate));
            break;
         case AGX_APPLE9_VIR_IOR_UNIFORM:
            write_val(I->dest, uniforms.at(I->immediate));
            break;
         case AGX_APPLE9_VIR_IADD:
            write_val(I->dest, read_val(I->src[0]) + read_val(I->src[1]));
            break;
         case AGX_APPLE9_VIR_IXOR:
            write_val(I->dest, read_val(I->src[0]) ^ read_val(I->src[1]));
            break;
         case AGX_APPLE9_VIR_MOV:
            write_val(I->dest, read_val(I->src[0]));
            break;
         case AGX_APPLE9_VIR_HALT:
            break;
         default:
            ADD_FAILURE() << "Unhandled op in evaluator: " << I->op;
            return 0;
         }
      }
   }

   return p->output < p->value_count ? read_val(p->output) : 0;
}

} /* namespace */

TEST(Apple9RaSpill, TightGprSpillStraightLine)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   std::unordered_map<uint32_t, uint32_t> sr_vals;
   uint32_t vals[16];
   uint32_t expected = 0;
   for (unsigned i = 0; i < 16; ++i) {
      sr_vals[i] = 0x1000u + i * 37u;
      vals[i] = emit_get_sr(&p, i);
      ASSERT_NE(vals[i], AGX_APPLE9_VREG_INVALID);
      expected += sr_vals[i];
   }
   /* Use in reverse order so all 16 non-rematerializable values are
    * simultaneously live before the reduction begins. */
   uint32_t acc = vals[15];
   for (int i = 14; i >= 0; --i)
      acc = emit_iadd(&p, acc, vals[i]);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   EXPECT_GE(apple9_ra_spill_calc_gpr_demand(&p, &state), 16u);

   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* 10 - 5 reserved = 5 GPRs available! */
   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.has_spills);

   verify_vir_ssa_and_classes(&p, &state);
   unsigned post_demand = apple9_ra_spill_calc_gpr_demand(&p, &state);
   EXPECT_LE(post_demand, 5u);

   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   EXPECT_GT(frame_words, 0u);
   EXPECT_LE(frame_words, 16u);

   std::unordered_map<uint32_t, uint32_t> uniforms;
   EXPECT_EQ(eval_straight_line(&p, &state, mem_slot, uniforms, sr_vals),
             expected);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, MultiWordTupleSpillCollectSplit)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   /* Build two 4-word tuples and one 2-word tuple so total demand exceeds
    * k = 12 (max_gprs = 17, reserved = 5). */
   uint32_t s[10];
   for (unsigned i = 0; i < 10; ++i)
      s[i] = emit_get_sr(&p, i);

   uint32_t q0 = agx_apple9_vir_emit_collect(&p, &s[0], 4);
   uint32_t q1 = agx_apple9_vir_emit_collect(&p, &s[4], 4);
   uint32_t pair = agx_apple9_vir_emit_collect(&p, &s[8], 2);
   ASSERT_NE(q0, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(q1, AGX_APPLE9_VREG_INVALID);
   ASSERT_NE(pair, AGX_APPLE9_VREG_INVALID);

   /* Split all three tuples (leave lane 1 of q0 dead via dead_dest_mask). */
   uint32_t sp_pair = agx_apple9_vir_emit_split(&p, pair, 2);
   uint32_t sp_q1 = agx_apple9_vir_emit_split(&p, q1, 4);
   uint32_t sp_q0 = agx_apple9_vir_emit_split(&p, q0, 4);
   agx_apple9_vir_instr *sp_q0_ins = p.instructions[p.instruction_count - 1];
   sp_q0_ins->dead_dest_mask = 1u << 1;

   uint32_t acc = emit_iadd(&p, sp_pair + 0, sp_pair + 1);
   acc = emit_iadd(&p, acc, sp_q1 + 0);
   acc = emit_iadd(&p, acc, sp_q1 + 3);
   acc = emit_iadd(&p, acc, sp_q0 + 0);
   acc = emit_iadd(&p, acc, sp_q0 + 2);
   acc = emit_iadd(&p, acc, sp_q0 + 3);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);

   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 15; /* 15 - 5 reserved = 10 GPRs available. */
   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.has_spills);

   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 10u);

   /* Verify multi-word tuple spills retained their width and aligned slots. */
   bool saw_tuple_mem = false;
   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   for (unsigned v = 0; v < p.value_count; ++v) {
      if (!apple9_ra_value_is_mem(&state, v))
         continue;
      unsigned w = apple9_ra_value_width(&state, v);
      unsigned pot = util_next_power_of_two(w);
      ASSERT_NE(mem_slot[v], UINT32_MAX);
      EXPECT_EQ(mem_slot[v] % pot, 0u)
         << "Misaligned spill slot " << mem_slot[v] << " for width " << w;
      if (w > 1)
         saw_tuple_mem = true;
   }
   EXPECT_TRUE(saw_tuple_mem);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, HardLowAndR63ConstraintsPreserved)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   /* Entry input pinned to fixed_phys = 0 (hard-low), plus constrained values
    * with max_phys = 15 (hard-low) and compact r0..r63 MOV_IMM_COMPACT. */
   uint32_t in0 = agx_apple9_vir_input(&p, 0);
   ASSERT_NE(in0, AGX_APPLE9_VREG_INVALID);

   uint32_t low_vals[8];
   for (unsigned i = 0; i < 8; ++i) {
      low_vals[i] = emit_get_sr(&p, i);
      p.max_phys[low_vals[i]] = 15; /* Hard-low r0..r15 bound */
   }
   uint32_t c_compact = emit_imm_compact(&p, 42);
   p.max_phys[c_compact] = 63;

   uint32_t acc = emit_iadd(&p, in0, c_compact);
   for (int i = 7; i >= 0; --i)
      acc = emit_iadd(&p, acc, low_vals[i]);
   /* Reuse in0 at the end so it gets spilled and reloaded (testing entry
    * vir_input redefinition repair!). */
   acc = emit_iadd(&p, acc, in0);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */
   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.has_spills);

   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   /* Entry input must still have fixed_phys = 0, and reloaded GPR defs of
    * constrained values must preserve max_phys <= 15. */
   EXPECT_EQ(p.fixed_phys[0], 0u);
   unsigned preserved_hard_low = 0;
   for (unsigned v = 0; v < p.value_count; ++v) {
      if (!apple9_ra_value_is_mem(&state, v) && p.max_phys[v] == 15)
         ++preserved_hard_low;
   }
   EXPECT_GE(preserved_hard_low, 8u);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, ImmutableUniformRematZeroSpills)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   std::unordered_map<uint32_t, uint32_t> uniforms;
   std::unordered_map<uint32_t, uint32_t> sr_vals;

   /* Build 12 pure-ALU uniform expressions (each <= 4 ops: IOR_UNIFORM +
    * IMM + IADD = 3 ops) and sum them in reverse order under a tight budget
    * where spilling would otherwise be mandatory. */
   uint32_t exprs[12];
   uint32_t expected = 0;
   for (unsigned i = 0; i < 12; ++i) {
      uniforms[i] = 100u + i * 7u;
      uint32_t u = emit_ior_uniform(&p, i);
      uint32_t c = emit_imm32(&p, i + 1u);
      exprs[i] = emit_iadd(&p, u, c);
      expected += uniforms[i] + (i + 1u);
   }

   uint32_t acc = exprs[11];
   for (int i = 10; i >= 0; --i)
      acc = emit_iadd(&p, acc, exprs[i]);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 8;
   opts.remat_only = true; /* Must succeed with ZERO RA_MEM spills! */
   opts.reserved_gprs = 0;

   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.uniform_remat_used);
   EXPECT_FALSE(state.has_spills);

   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 8u);

   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 99;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   EXPECT_EQ(frame_words, 0u);
   EXPECT_EQ(apple9_ra_spill_scratch_bytes(&p, frame_words), 0u);
   EXPECT_EQ(eval_straight_line(&p, &state, mem_slot, uniforms, sr_vals),
             expected);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, MutableUniformDisablesRemat)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   /* Emit a STORE_UNIFORM (preamble shader -> mutable uniforms). */
   uint32_t w = emit_imm32(&p, 7);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
                                               AGX_APPLE9_ENC_STORE_UNIFORM,
                                               &w, 1, 0));

   uint32_t exprs[10];
   for (unsigned i = 0; i < 10; ++i)
      exprs[i] = emit_ior_uniform(&p, i);

   uint32_t acc = exprs[9];
   for (int i = 8; i >= 0; --i)
      acc = emit_iadd(&p, acc, exprs[i]);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */

   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_FALSE(state.uniform_remat_used);
   EXPECT_TRUE(state.has_spills);
   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, DiamondSpillAndCoupling)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block nA, nB, nC, nD;
   agx_apple9_block *A = make_cfg_block(&p, &nA);
   agx_apple9_block *B = make_cfg_block(&p, &nB);
   agx_apple9_block *C = make_cfg_block(&p, &nC);
   agx_apple9_block *D = make_cfg_block(&p, &nD);
   nA.successors[0] = &nB;
   nA.successors[1] = &nC;
   nB.successors[0] = &nD;
   nC.successors[0] = &nD;

   agx_apple9_block_begin(&p, A);
   uint32_t live[8];
   for (unsigned i = 0; i < 8; ++i)
      live[i] = emit_get_sr(&p, i);

   agx_apple9_block_begin(&p, B);
   uint32_t b_val = emit_iadd(&p, live[0], live[1]);
   /* Increase pressure inside B so some live[] values spill in B only. */
   uint32_t b_tmp[4];
   for (unsigned i = 0; i < 4; ++i)
      b_tmp[i] = emit_get_sr(&p, 20 + i);
   for (unsigned i = 0; i < 4; ++i)
      b_val = emit_iadd(&p, b_val, b_tmp[i]);

   agx_apple9_block_begin(&p, C);
   uint32_t c_val = emit_ixor(&p, live[2], live[3]);

   agx_apple9_block_begin(&p, D);
   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, B)) = b_val;
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, C)) = c_val;

   uint32_t acc = phi_v;
   for (unsigned i = 0; i < 8; ++i)
      acc = emit_iadd(&p, acc, live[i]);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */

   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.has_spills);
   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   EXPECT_GT(frame_words, 0u);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, LoopInvariantSpillHoisting)
{
   /* Pre -> Header -> Body -> Header, Header -> Exit.
    * 6 loop-invariant values defined in Pre are used inside Body and Exit,
    * while Body also creates local pressure under k = 5 (max_gprs = 10).
    * Verify that hoisting removes loop-invariant RA_MEM stores from inside
    * the loop and places them in Pre. */
   auto run_case = [](bool enable_hoist, unsigned *spills_in_loop_out,
                      unsigned *spills_in_pre_out) {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);

      nir_block nPre, nHdr, nBody, nExit;
      agx_apple9_block *Pre = make_cfg_block(&p, &nPre);
      agx_apple9_block *Hdr = make_cfg_block(&p, &nHdr);
      agx_apple9_block *Body = make_cfg_block(&p, &nBody);
      agx_apple9_block *Exit = make_cfg_block(&p, &nExit);
      nPre.successors[0] = &nHdr;
      nHdr.successors[0] = &nBody;
      nHdr.successors[1] = &nExit;
      nBody.successors[0] = &nHdr;

      agx_apple9_block_begin(&p, Pre);
      uint32_t inv[4];
      for (unsigned i = 0; i < 4; ++i)
         inv[i] = emit_get_sr(&p, i);
      uint32_t init_acc = emit_get_sr(&p, 10);

      agx_apple9_block_begin(&p, Hdr);
      agx_apple9_block_begin(&p, Body);
      agx_apple9_block_begin(&p, Exit);
      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      ASSERT_TRUE(Hdr->loop_header);

      /* Loop-carried accumulator phi in Hdr. */
      p.current_block = Hdr;
      uint32_t acc_phi = agx_apple9_vir_emit_phi_canonical(&p, Hdr, 2);
      agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(Hdr, Pre)) =
         init_acc;

      /* Body uses all 4 invariant values AND creates 5 local live values so
       * invariant values must be evicted inside Body (k = 5). */
      p.current_block = Body;
      uint32_t loc[5];
      for (unsigned i = 0; i < 5; ++i)
         loc[i] = emit_get_sr(&p, 20 + i);
      uint32_t next_acc = acc_phi;
      for (unsigned i = 0; i < 5; ++i)
         next_acc = emit_iadd(&p, next_acc, loc[i]);
      for (unsigned i = 0; i < 4; ++i)
         next_acc = emit_iadd(&p, next_acc, inv[i]);
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(Hdr, Body)) =
         next_acc;

      p.current_block = Exit;
      uint32_t out = acc_phi;
      for (unsigned i = 0; i < 4; ++i)
         out = emit_iadd(&p, out, inv[i]);
      p.output = out;

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      apple9_ra_spill_options opts = apple9_ra_spill_default_options();
      opts.max_gprs = 10; /* k = 5 */
      opts.hoist_loop_invariant = enable_hoist;

      ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
      verify_vir_ssa_and_classes(&p, &state);
      EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

      auto count_spills = [&](agx_apple9_block *blk) {
         unsigned cnt = 0;
         for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
              I = agx_apple9_instr_next(I)) {
            if (I->op == AGX_APPLE9_VIR_MOV &&
                apple9_ra_value_is_mem(&state, I->dest) &&
                !apple9_ra_value_is_mem(&state, I->src[0]))
               ++cnt;
         }
         return cnt;
      };

      *spills_in_pre_out = count_spills(Pre);
      *spills_in_loop_out = count_spills(Hdr) + count_spills(Body);

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   };

   unsigned loop_no_hoist = 0, pre_no_hoist = 0;
   unsigned loop_hoist = 0, pre_hoist = 0;
   run_case(false, &loop_no_hoist, &pre_no_hoist);
   run_case(true, &loop_hoist, &pre_hoist);

   /* Hoisting must strictly reduce loop-interior spill stores and move them
    * into the preheader. */
   EXPECT_LT(loop_hoist, loop_no_hoist);
   EXPECT_GT(pre_hoist, pre_no_hoist);
}

TEST(Apple9RaSpill, WidePhiOver10PredsSpillAndRepair)
{
   /* 16-leaf binary tree merging into b_merge (> AGX_APPLE9_MAX_VIR_SRCS = 10),
    * exercising sideband phi_srcs[] through both spilling and SSA repair. */
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);

   nir_block n_tree[31], n_merge;
   agx_apple9_block *b_tree[31];
   for (unsigned i = 0; i < 31; ++i)
      b_tree[i] = make_cfg_block(&p, &n_tree[i]);
   agx_apple9_block *b_merge = make_cfg_block(&p, &n_merge);

   for (unsigned i = 0; i < 15; ++i) {
      n_tree[i].successors[0] = &n_tree[2 * i + 1];
      n_tree[i].successors[1] = &n_tree[2 * i + 2];
   }
   for (unsigned i = 15; i < 31; ++i)
      n_tree[i].successors[0] = &n_merge;

   /* Entry block defines 8 live values so pressure exceeds k = 5. */
   agx_apple9_block_begin(&p, b_tree[0]);
   uint32_t live[8];
   for (unsigned i = 0; i < 8; ++i)
      live[i] = emit_get_sr(&p, i);

   for (unsigned i = 1; i < 15; ++i)
      agx_apple9_block_begin(&p, b_tree[i]);

   uint32_t leaf_vals[16];
   for (unsigned k = 0; k < 16; ++k) {
      agx_apple9_block_begin(&p, b_tree[15 + k]);
      /* Create pressure in every leaf and compute a distinct value for merge phi. */
      uint32_t t = emit_get_sr(&p, 30 + k);
      leaf_vals[k] = emit_iadd(&p, t, live[k % 8]);
   }

   agx_apple9_block_begin(&p, b_merge);
   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   ASSERT_EQ(b_merge->num_cfg_preds, 16u);

   uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(&p, b_merge, 16);
   ASSERT_NE(phi_v, AGX_APPLE9_VREG_INVALID);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   ASSERT_NE(phi->phi_srcs, nullptr);
   for (unsigned k = 0; k < 16; ++k) {
      unsigned pred_idx = agx_apple9_block_pred_index(b_merge, b_tree[15 + k]);
      ASSERT_NE(pred_idx, UINT_MAX);
      /* Exercise an undef incoming edge on leaf 15. */
      *agx_apple9_vir_src_ptr(phi, pred_idx) =
         (k == 15) ? AGX_APPLE9_VREG_INVALID : leaf_vals[k];
   }

   uint32_t acc = phi_v;
   for (unsigned i = 0; i < 8; ++i)
      acc = emit_iadd(&p, acc, live[i]);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */

   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   EXPECT_TRUE(state.has_spills);
   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   /* Verify sideband >10-pred phis exist at b_merge with 16 sources. */
   unsigned merge_phis = 0;
   for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b_merge);
        I && I->op == AGX_APPLE9_VIR_PHI; I = agx_apple9_instr_next(I)) {
      EXPECT_EQ(agx_apple9_vir_nr_srcs(I), 16u);
      EXPECT_NE(I->phi_srcs, nullptr);
      ++merge_phis;
   }
   EXPECT_GE(merge_phis, 1u);

   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   EXPECT_GT(frame_words, 0u);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaRepairSsa, LocalDiamondLoopAndSidebandPhi)
{
   /* 1. Diamond: x defined in A, redefined in B, untouched in C, used in D.
    *    Braun repair must insert 1 non-trivial phi in D; an untouched y defined
    *    in A and used in D must produce 0 phis! */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nB, nC, nD;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      agx_apple9_block *C = make_cfg_block(&p, &nC);
      agx_apple9_block *D = make_cfg_block(&p, &nD);
      nA.successors[0] = &nB;
      nA.successors[1] = &nC;
      nB.successors[0] = &nD;
      nC.successors[0] = &nD;

      agx_apple9_block_begin(&p, A);
      uint32_t x = emit_imm32(&p, 10);
      uint32_t y = emit_imm32(&p, 20);

      agx_apple9_block_begin(&p, B);
      /* Intentionally break SSA by redefining x in B. */
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 99));
      p.instructions[p.instruction_count - 1]->dest = x;
      p.instructions[p.instruction_count - 1]->dest_components = 1;

      agx_apple9_block_begin(&p, C);
      agx_apple9_block_begin(&p, D);
      p.output = emit_iadd(&p, x, y);

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      ASSERT_TRUE(apple9_ra_repair_ssa(&p, nullptr, false));
      verify_vir_ssa_and_classes(&p, nullptr);

      unsigned phis = 0;
      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(D);
           I && I->op == AGX_APPLE9_VIR_PHI; I = agx_apple9_instr_next(I))
         ++phis;
      EXPECT_EQ(phis, 1u);
      agx_apple9_vir_finish(&p);
   }

   /* 2. Loop: x defined in Pre, used in Hdr, redefined in Body (backedge).
    *    Braun repair must insert & seal 1 loop phi in Hdr; untouched y has 0 phis. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nPre, nHdr, nBody, nExit;
      agx_apple9_block *Pre = make_cfg_block(&p, &nPre);
      agx_apple9_block *Hdr = make_cfg_block(&p, &nHdr);
      agx_apple9_block *Body = make_cfg_block(&p, &nBody);
      agx_apple9_block *Exit = make_cfg_block(&p, &nExit);
      nPre.successors[0] = &nHdr;
      nHdr.successors[0] = &nBody;
      nHdr.successors[1] = &nExit;
      nBody.successors[0] = &nHdr;

      agx_apple9_block_begin(&p, Pre);
      uint32_t x = emit_imm32(&p, 1);
      uint32_t y = emit_imm32(&p, 2);

      agx_apple9_block_begin(&p, Hdr);
      uint32_t use_in_hdr = emit_iadd(&p, x, y);

      agx_apple9_block_begin(&p, Body);
      /* Redefine x on backedge. */
      uint32_t srcs[2] = {use_in_hdr, y};
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, srcs, 2, 0));
      p.instructions[p.instruction_count - 1]->dest = x;
      p.instructions[p.instruction_count - 1]->dest_components = 1;

      agx_apple9_block_begin(&p, Exit);
      p.output = emit_iadd(&p, x, y);

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      ASSERT_TRUE(Hdr->loop_header);
      ASSERT_TRUE(apple9_ra_repair_ssa(&p, nullptr, false));
      verify_vir_ssa_and_classes(&p, nullptr);

      unsigned hdr_phis = 0;
      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(Hdr);
           I && I->op == AGX_APPLE9_VIR_PHI; I = agx_apple9_instr_next(I)) {
         EXPECT_NE(agx_apple9_vir_src(I, 0), AGX_APPLE9_VREG_INVALID);
         EXPECT_NE(agx_apple9_vir_src(I, 1), AGX_APPLE9_VREG_INVALID);
         EXPECT_NE(agx_apple9_vir_src(I, 0), agx_apple9_vir_src(I, 1));
         ++hdr_phis;
      }
      EXPECT_EQ(hdr_phis, 1u);
      agx_apple9_vir_finish(&p);
   }

   /* 3. >10-pred sideband PHI inserted purely by Braun SSA repair (16 preds). */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block n_tree[31], n_merge;
      agx_apple9_block *b_tree[31];
      for (unsigned i = 0; i < 31; ++i)
         b_tree[i] = make_cfg_block(&p, &n_tree[i]);
      agx_apple9_block *b_merge = make_cfg_block(&p, &n_merge);

      for (unsigned i = 0; i < 15; ++i) {
         n_tree[i].successors[0] = &n_tree[2 * i + 1];
         n_tree[i].successors[1] = &n_tree[2 * i + 2];
      }
      for (unsigned i = 15; i < 31; ++i)
         n_tree[i].successors[0] = &n_merge;

      agx_apple9_block_begin(&p, b_tree[0]);
      uint32_t x = emit_imm32(&p, 0);
      for (unsigned i = 1; i < 15; ++i)
         agx_apple9_block_begin(&p, b_tree[i]);
      for (unsigned k = 0; k < 16; ++k) {
         agx_apple9_block_begin(&p, b_tree[15 + k]);
         /* Redefine x in leaf 0 and leaf 13; leave others inheriting root x. */
         if (k == 0 || k == 13) {
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0,
               100 + k));
            p.instructions[p.instruction_count - 1]->dest = x;
            p.instructions[p.instruction_count - 1]->dest_components = 1;
         }
      }
      agx_apple9_block_begin(&p, b_merge);
      p.output = emit_iadd(&p, x, x);

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      ASSERT_TRUE(apple9_ra_repair_ssa(&p, nullptr, false));
      verify_vir_ssa_and_classes(&p, nullptr);

      agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(b_merge);
      ASSERT_NE(phi, nullptr);
      ASSERT_EQ(phi->op, AGX_APPLE9_VIR_PHI);
      EXPECT_EQ(agx_apple9_vir_nr_srcs(phi), 16u);
      EXPECT_NE(phi->phi_srcs, nullptr);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9RaSpillCoalesce, DisjointLifetimesReuseSingleSlot)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nb;
   agx_apple9_block *b0 = make_cfg_block(&p, &nb);
   agx_apple9_block_begin(&p, b0);

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);

   uint32_t x = emit_imm32(&p, 7);
   uint32_t acc = x;
   /* 8 sequential disjoint RA_MEM lifetimes -> must coalesce to 1 word! */
   std::vector<uint32_t> mem_vals;
   for (unsigned i = 0; i < 8; ++i) {
      uint32_t m = agx_apple9_vir_emit_mov(&p, acc);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      BITSET_SET(state.is_mem, m);
      state.value_width[m] = 1;
      mem_vals.push_back(m);
      uint32_t r = agx_apple9_vir_emit_mov(&p, m);
      acc = emit_iadd(&p, r, x);
   }
   p.output = acc;

   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   EXPECT_EQ(frame_words, 1u);
   EXPECT_EQ(apple9_ra_spill_scratch_bytes(&p, frame_words), 16u);
   for (uint32_t m : mem_vals)
      EXPECT_EQ(mem_slot[m], 0u);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpillCoalesce, AtomicBaseOffsetReserves16Bytes)
{
   for (bool discard : {false, true}) {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nb;
      agx_apple9_block *b0 = make_cfg_block(&p, &nb);
      agx_apple9_block_begin(&p, b0);

      uint32_t idx = emit_imm32(&p, 0);
      uint32_t data = emit_imm32(&p, 1);
      uint32_t atomic_res = AGX_APPLE9_VREG_INVALID;
      ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
         &p, 0, idx, &data, 1, AGX_APPLE9_ATOMIC_ADD, discard, &atomic_res));

      EXPECT_EQ(apple9_ra_returns_device_atomic(&p), !discard);
      EXPECT_EQ(apple9_ra_spill_base_words(&p), discard ? 0u : 4u);

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);

      /* Spill two simultaneously live scalars. */
      uint32_t m0 = agx_apple9_vir_emit_mov(&p, idx);
      uint32_t m1 = agx_apple9_vir_emit_mov(&p, data);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      BITSET_SET(state.is_mem, m0);
      BITSET_SET(state.is_mem, m1);

      uint32_t r0 = agx_apple9_vir_emit_mov(&p, m0);
      uint32_t r1 = agx_apple9_vir_emit_mov(&p, m1);
      p.output = emit_iadd(&p, r0, r1);

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      unsigned frame_words = 0;
      ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                                 &frame_words));
      if (!discard) {
         /* Words 0..3 reserved for returning device atomic -> slots 4, 5! */
         EXPECT_GE(mem_slot[m0], 4u);
         EXPECT_GE(mem_slot[m1], 4u);
         EXPECT_NE(mem_slot[m0], mem_slot[m1]);
         EXPECT_EQ(frame_words, 6u);
         EXPECT_EQ(apple9_ra_spill_scratch_bytes(&p, frame_words), 32u);
      } else {
         EXPECT_EQ(frame_words, 2u);
         EXPECT_EQ(apple9_ra_spill_scratch_bytes(&p, frame_words), 16u);
      }

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9RaSpillCoalesce, BoundedApplyDiamondAndRollback)
{
   /* 1. Diamond frame phi coalesces 3 initial slots down to 1 word. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nB, nC, nD;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      agx_apple9_block *C = make_cfg_block(&p, &nC);
      agx_apple9_block *D = make_cfg_block(&p, &nD);
      nA.successors[0] = &nB;
      nA.successors[1] = &nC;
      nB.successors[0] = &nD;
      nC.successors[0] = &nD;

      agx_apple9_block_begin(&p, A);
      uint32_t x = emit_imm32(&p, 7);
      uint32_t y = emit_imm32(&p, 11);

      agx_apple9_block_begin(&p, B);
      uint32_t left = agx_apple9_vir_emit_mov(&p, x);

      agx_apple9_block_begin(&p, C);
      uint32_t right = agx_apple9_vir_emit_mov(&p, y);

      agx_apple9_block_begin(&p, D);
      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      uint32_t joined = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
      agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, B)) = left;
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, C)) = right;
      p.output = agx_apple9_vir_emit_mov(&p, joined);

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      state.spill_home = (uint32_t *)calloc(p.value_count, sizeof(uint32_t));
      state.spill_home_count = p.value_count;

      for (uint32_t m : {left, right, joined}) {
         BITSET_SET(state.is_mem, m);
         state.spill_home[m] = 1;
      }

      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      mem_slot[left] = 0;
      mem_slot[right] = 1;
      mem_slot[joined] = 2;
      unsigned frame_words = 3;

      auto *g = apple9_ra_spill_coalesce_prepare(&p, &state);
      ASSERT_NE(g, nullptr);
      EXPECT_TRUE(apple9_ra_spill_coalesce_apply(&p, &state, g, mem_slot.data(),
                                                 &frame_words));
      EXPECT_EQ(frame_words, 1u);
      EXPECT_EQ(mem_slot[left], mem_slot[joined]);
      EXPECT_EQ(mem_slot[right], mem_slot[joined]);

      apple9_ra_spill_coalesce_destroy(g);
      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }

   /* 2. Contraction that increases chromatic number rolls back cleanly:
    *    Path a-c-d-e fits in 2 words, but contracting same-home endpoints a/e
    *    forms a triangle requiring 3 words -> apply() must return false and
    *    keep the original 2-word assignment! */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nb;
      agx_apple9_block *b0 = make_cfg_block(&p, &nb);
      agx_apple9_block_begin(&p, b0);

      uint32_t x = emit_imm32(&p, 7);
      uint32_t a = agx_apple9_vir_emit_mov(&p, x);
      uint32_t c = agx_apple9_vir_emit_mov(&p, x);
      uint32_t ra = agx_apple9_vir_emit_mov(&p, a);
      uint32_t d = agx_apple9_vir_emit_mov(&p, x);
      uint32_t rc = agx_apple9_vir_emit_mov(&p, c);
      uint32_t e = agx_apple9_vir_emit_mov(&p, x);
      uint32_t rd = agx_apple9_vir_emit_mov(&p, d);
      uint32_t re = agx_apple9_vir_emit_mov(&p, e);
      p.output = emit_iadd(&p, emit_iadd(&p, ra, rc), emit_iadd(&p, rd, re));

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      state.spill_home = (uint32_t *)calloc(p.value_count, sizeof(uint32_t));
      state.spill_home_count = p.value_count;

      BITSET_SET(state.is_mem, a);
      BITSET_SET(state.is_mem, c);
      BITSET_SET(state.is_mem, d);
      BITSET_SET(state.is_mem, e);
      state.spill_home[a] = 1;
      state.spill_home[c] = 2;
      state.spill_home[d] = 3;
      state.spill_home[e] = 1;

      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      mem_slot[a] = 0;
      mem_slot[c] = 1;
      mem_slot[d] = 0;
      mem_slot[e] = 1;
      unsigned frame_words = 2;

      auto *g = apple9_ra_spill_coalesce_prepare(&p, &state);
      ASSERT_NE(g, nullptr);
      EXPECT_FALSE(apple9_ra_spill_coalesce_apply(&p, &state, g, mem_slot.data(),
                                                  &frame_words));
      EXPECT_EQ(frame_words, 2u);
      EXPECT_EQ(mem_slot[a], 0u);
      EXPECT_EQ(mem_slot[c], 1u);
      EXPECT_EQ(mem_slot[d], 0u);
      EXPECT_EQ(mem_slot[e], 1u);

      apple9_ra_spill_coalesce_destroy(g);
      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9RaSpill, DiamondAndLoopSemanticExecution)
{
   /* Execute spilled + repaired + DSATUR-colored VIR across both paths of a
    * diamond and across 5 iterations of a loop, verifying bit-exact results. */
   auto eval_block = [](const agx_apple9_block *blk,
                        const agx_apple9_block *prev_blk,
                        const apple9_ra_spill_state *state,
                        const std::vector<uint32_t> &mem_slot,
                        const std::unordered_map<uint32_t, uint32_t> &sr_vals,
                        std::vector<uint32_t> &gpr,
                        std::vector<uint32_t> &frame) {
      auto read_val = [&](uint32_t v) -> uint32_t {
         return apple9_ra_value_is_mem(state, v) ? frame[mem_slot[v]] : gpr[v];
      };
      auto write_val = [&](uint32_t v, uint32_t x) {
         if (apple9_ra_value_is_mem(state, v))
            frame[mem_slot[v]] = x;
         else
            gpr[v] = x;
      };

      /* Evaluate leading phis simultaneously before committing. */
      std::vector<std::pair<uint32_t, uint32_t>> phi_writes;
      unsigned pred_idx = prev_blk ? agx_apple9_block_pred_index(blk, prev_blk) : 0;
      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I && I->op == AGX_APPLE9_VIR_PHI; I = agx_apple9_instr_next(I)) {
         uint32_t sv = agx_apple9_vir_src(I, pred_idx);
         phi_writes.emplace_back(I->dest, read_val(sv));
      }
      for (auto [dst, val] : phi_writes)
         write_val(dst, val);

      for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
           I = agx_apple9_instr_next(I)) {
         switch (I->op) {
         case AGX_APPLE9_VIR_PHI:
            break;
         case AGX_APPLE9_VIR_IMM:
            write_val(I->dest, I->immediate);
            break;
         case AGX_APPLE9_VIR_GET_SR:
            write_val(I->dest, sr_vals.at(I->immediate));
            break;
         case AGX_APPLE9_VIR_IADD:
            write_val(I->dest, read_val(I->src[0]) + read_val(I->src[1]));
            break;
         case AGX_APPLE9_VIR_IXOR:
            write_val(I->dest, read_val(I->src[0]) ^ read_val(I->src[1]));
            break;
         case AGX_APPLE9_VIR_MOV:
            write_val(I->dest, read_val(I->src[0]));
            break;
         default:
            ADD_FAILURE() << "Unexpected op " << I->op;
         }
      }
   };

   /* Loop with 5 iterations, 4 loop-invariant values, and 5 local temps under
    * tight k = 5 (forces spilling + loop-invariant hoisting + repair + DSATUR). */
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nPre, nHdr, nBody, nExit;
   agx_apple9_block *Pre = make_cfg_block(&p, &nPre);
   agx_apple9_block *Hdr = make_cfg_block(&p, &nHdr);
   agx_apple9_block *Body = make_cfg_block(&p, &nBody);
   agx_apple9_block *Exit = make_cfg_block(&p, &nExit);
   nPre.successors[0] = &nHdr;
   nHdr.successors[0] = &nBody;
   nHdr.successors[1] = &nExit;
   nBody.successors[0] = &nHdr;

   std::unordered_map<uint32_t, uint32_t> sr_vals;
   agx_apple9_block_begin(&p, Pre);
   uint32_t inv[4];
   uint32_t inv_sum = 0;
   for (unsigned i = 0; i < 4; ++i) {
      sr_vals[i] = 10u + i * 3u;
      inv_sum += sr_vals[i];
      inv[i] = emit_get_sr(&p, i);
   }
   sr_vals[10] = 7u;
   uint32_t init_acc = emit_get_sr(&p, 10);

   agx_apple9_block_begin(&p, Hdr);
   agx_apple9_block_begin(&p, Body);
   agx_apple9_block_begin(&p, Exit);
   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));

   p.current_block = Hdr;
   uint32_t acc_phi = agx_apple9_vir_emit_phi_canonical(&p, Hdr, 2);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(Hdr, Pre)) = init_acc;

   p.current_block = Body;
   uint32_t loc[5];
   uint32_t loc_sum = 0;
   for (unsigned i = 0; i < 5; ++i) {
      sr_vals[20 + i] = 100u + i * 11u;
      loc_sum += sr_vals[20 + i];
      loc[i] = emit_get_sr(&p, 20 + i);
   }
   uint32_t next_acc = acc_phi;
   for (unsigned i = 0; i < 5; ++i)
      next_acc = emit_iadd(&p, next_acc, loc[i]);
   for (unsigned i = 0; i < 4; ++i)
      next_acc = emit_iadd(&p, next_acc, inv[i]);
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(Hdr, Body)) = next_acc;

   p.current_block = Exit;
   uint32_t out = acc_phi;
   for (unsigned i = 0; i < 4; ++i)
      out = emit_iadd(&p, out, inv[i]);
   p.output = out;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */
   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));

   std::vector<uint32_t> gpr(p.value_count, 0);
   std::vector<uint32_t> frame(APPLE9_RA_MAX_SPILL_WORDS, 0);
   eval_block(Pre, nullptr, &state, mem_slot, sr_vals, gpr, frame);
   agx_apple9_block *prev = Pre;
   for (unsigned iter = 0; iter < 5; ++iter) {
      eval_block(Hdr, prev, &state, mem_slot, sr_vals, gpr, frame);
      eval_block(Body, Hdr, &state, mem_slot, sr_vals, gpr, frame);
      prev = Body;
   }
   eval_block(Hdr, prev, &state, mem_slot, sr_vals, gpr, frame);
   eval_block(Exit, Hdr, &state, mem_slot, sr_vals, gpr, frame);

   uint32_t expected = 7u + 5u * (loc_sum + inv_sum) + inv_sum;
   uint32_t actual = apple9_ra_value_is_mem(&state, p.output)
                        ? frame[mem_slot[p.output]]
                        : gpr[p.output];
   EXPECT_EQ(actual, expected);

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpill, SpilledPhiWithRematSourcesAndReturningAtomic)
{
   /* Test two edge cases together:
    * 1. A returning DEVICE_ATOMIC + DEVICE_ATOMIC_RESULT pair stays adjacent
    *    when surrounding pressure forces spilling, and DSATUR reserves words
    *    0..3 (16 B).
    * 2. A diamond VIR_PHI whose sources are rematerializable VIR_IMM /
    *    IOR_UNIFORM values gets spilled to RA_MEM cleanly under tight k. */
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   nir_block nA, nB, nC, nD;
   agx_apple9_block *A = make_cfg_block(&p, &nA);
   agx_apple9_block *B = make_cfg_block(&p, &nB);
   agx_apple9_block *C = make_cfg_block(&p, &nC);
   agx_apple9_block *D = make_cfg_block(&p, &nD);
   nA.successors[0] = &nB;
   nA.successors[1] = &nC;
   nB.successors[0] = &nD;
   nC.successors[0] = &nD;

   agx_apple9_block_begin(&p, A);
   uint32_t idx = emit_get_sr(&p, 0);
   uint32_t data = emit_get_sr(&p, 1);
   uint32_t atomic_res = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &p, 0, idx, &data, 1, AGX_APPLE9_ATOMIC_ADD, false, &atomic_res));
   uint32_t live[6];
   for (unsigned i = 0; i < 6; ++i)
      live[i] = emit_get_sr(&p, 10 + i);

   agx_apple9_block_begin(&p, B);
   uint32_t b_imm = emit_imm32(&p, 0x1234);

   agx_apple9_block_begin(&p, C);
   uint32_t c_uni = emit_ior_uniform(&p, 3);

   agx_apple9_block_begin(&p, D);
   ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
   uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
   agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, B)) = b_imm;
   *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, C)) = c_uni;

   /* Use live[] BEFORE phi_v so phi_v has farthest distance and spills to RA_MEM! */
   uint32_t acc = atomic_res;
   for (unsigned i = 0; i < 6; ++i)
      acc = emit_iadd(&p, acc, live[i]);
   acc = emit_iadd(&p, acc, phi_v);
   p.output = acc;

   apple9_ra_spill_state state;
   apple9_ra_spill_state_init(&state);
   apple9_ra_spill_options opts = apple9_ra_spill_default_options();
   opts.max_gprs = 10; /* k = 5 */
   ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
   verify_vir_ssa_and_classes(&p, &state);
   EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 5u);

   /* Verify DEVICE_ATOMIC is STILL immediately followed by DEVICE_ATOMIC_RESULT
    * with matching SSA value! */
   bool found_atomic_pair = false;
   for (agx_apple9_vir_instr *I = agx_apple9_block_first_instr(A); I;
        I = agx_apple9_instr_next(I)) {
      if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         agx_apple9_vir_instr *next = agx_apple9_instr_next(I);
         ASSERT_NE(next, nullptr);
         EXPECT_EQ(next->op, AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
         EXPECT_EQ(next->src[0], I->dest);
         found_atomic_pair = true;
      }
   }
   EXPECT_TRUE(found_atomic_pair);

   /* Verify DSATUR reserves words 0..3 for the returning atomic. */
   std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
   unsigned frame_words = 0;
   ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                              &frame_words));
   for (unsigned v = 0; v < p.value_count; ++v) {
      if (apple9_ra_value_is_mem(&state, v)) {
         EXPECT_GE(mem_slot[v], 4u);
      }
   }

   apple9_ra_spill_state_finish(&state);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9RaSpillCoalesce, DeadPhiMixedWidthsLoopAndDenseBitset)
{
   /* 1. Dead RA_MEM phi cannot clobber live RA_MEM value at block entry. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nB, nC, nD;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      agx_apple9_block *C = make_cfg_block(&p, &nC);
      agx_apple9_block *D = make_cfg_block(&p, &nD);
      nA.successors[0] = &nB;
      nA.successors[1] = &nC;
      nB.successors[0] = &nD;
      nC.successors[0] = &nD;

      agx_apple9_block_begin(&p, A);
      uint32_t x = emit_imm32(&p, 7);
      uint32_t live_m = agx_apple9_vir_emit_mov(&p, x);

      agx_apple9_block_begin(&p, B);
      uint32_t left = agx_apple9_vir_emit_mov(&p, x);

      agx_apple9_block_begin(&p, C);
      uint32_t right = agx_apple9_vir_emit_mov(&p, x);

      agx_apple9_block_begin(&p, D);
      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      uint32_t dead_phi = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
      agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, B)) = left;
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(D, C)) = right;
      p.output = agx_apple9_vir_emit_mov(&p, live_m);

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      state.spill_home = (uint32_t *)calloc(p.value_count, sizeof(uint32_t));
      state.spill_home_count = p.value_count;
      BITSET_SET(state.is_mem, live_m);
      BITSET_SET(state.is_mem, left);
      BITSET_SET(state.is_mem, right);
      BITSET_SET(state.is_mem, dead_phi);
      state.spill_home[live_m] = 1;
      state.spill_home[left] = 2;
      state.spill_home[right] = 3;
      state.spill_home[dead_phi] = 1; /* Same home as live_m, MUST NOT merge! */

      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      unsigned frame_words = 0;
      ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                                 &frame_words));
      EXPECT_NE(mem_slot[dead_phi], mem_slot[live_m]);

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }

   /* 2. Loop-carried RA_MEM phi coalesces init/carried/next to 1 slot. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nH, nB, nE;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *H = make_cfg_block(&p, &nH);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      agx_apple9_block *E = make_cfg_block(&p, &nE);
      nA.successors[0] = &nH;
      nH.successors[0] = &nB;
      nH.successors[1] = &nE;
      nB.successors[0] = &nH;

      agx_apple9_block_begin(&p, A);
      uint32_t x = emit_imm32(&p, 7);
      uint32_t init = agx_apple9_vir_emit_mov(&p, x);

      agx_apple9_block_begin(&p, H);
      agx_apple9_block_begin(&p, B);
      agx_apple9_block_begin(&p, E);
      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));

      p.current_block = H;
      uint32_t carried = agx_apple9_vir_emit_phi_canonical(&p, H, 2);
      agx_apple9_vir_instr *phi = p.instructions[p.instruction_count - 1];
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(H, A)) = init;

      p.current_block = B;
      uint32_t v = agx_apple9_vir_emit_mov(&p, carried);
      uint32_t next = agx_apple9_vir_emit_mov(&p, v);
      *agx_apple9_vir_src_ptr(phi, agx_apple9_block_pred_index(H, B)) = next;

      p.current_block = E;
      p.output = agx_apple9_vir_emit_mov(&p, carried);

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
      state.spill_home = (uint32_t *)calloc(p.value_count, sizeof(uint32_t));
      state.spill_home_count = p.value_count;
      for (uint32_t m : {init, carried, next}) {
         BITSET_SET(state.is_mem, m);
         state.spill_home[m] = 1;
      }

      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      unsigned frame_words = 0;
      ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                                 &frame_words));
      EXPECT_EQ(frame_words, 1u);
      EXPECT_EQ(mem_slot[init], mem_slot[carried]);
      EXPECT_EQ(mem_slot[next], mem_slot[carried]);

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }

   /* 3. Dense live boundary spanning > 64 BITSET_WORD bits (97 simultaneous
    *    RA_MEM values across two blocks). */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nB;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      nA.successors[0] = &nB;

      agx_apple9_block_begin(&p, A);
      uint32_t src = emit_imm32(&p, 7);
      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);

      constexpr unsigned count = 97;
      std::vector<uint32_t> mems;
      for (unsigned i = 0; i < count; ++i) {
         uint32_t m = agx_apple9_vir_emit_mov(&p, src);
         ASSERT_TRUE(apple9_ra_spill_state_ensure(&state, p.value_count));
         BITSET_SET(state.is_mem, m);
         mems.push_back(m);
      }

      agx_apple9_block_begin(&p, B);
      uint32_t acc = src;
      for (uint32_t m : mems)
         acc = emit_iadd(&p, acc, agx_apple9_vir_emit_mov(&p, m));
      p.output = acc;

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      unsigned frame_words = 0;
      ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                                 &frame_words));
      EXPECT_EQ(frame_words, count);
      std::vector<bool> used(count, false);
      for (uint32_t m : mems) {
         ASSERT_LT(mem_slot[m], count);
         EXPECT_FALSE(used[mem_slot[m]]);
         used[mem_slot[m]] = true;
      }

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9RaSpill, SpilledPhiConstantStaysImmAndLiteralRematSurvivesStoreUniform)
{
   /* 1. Spilled phi whose predecessor supplies a scalar VIR_IMM must mark that
    *    value as is_imm rather than spilling it to a real scratch slot. */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nA, nB, nC, nD;
      agx_apple9_block *A = make_cfg_block(&p, &nA);
      agx_apple9_block *B = make_cfg_block(&p, &nB);
      agx_apple9_block *C = make_cfg_block(&p, &nC);
      agx_apple9_block *D = make_cfg_block(&p, &nD);
      nA.successors[0] = &nB;
      nA.successors[1] = &nC;
      nB.successors[0] = &nD;
      nC.successors[0] = &nD;

      agx_apple9_block_begin(&p, A);
      uint32_t live_a[8];
      for (unsigned i = 0; i < 8; ++i)
         live_a[i] = emit_get_sr(&p, i);

      agx_apple9_block_begin(&p, B);
      uint32_t b_val = emit_iadd(&p, live_a[0], live_a[1]);
      /* Force b_val to spill in B before joining at D. */
      uint32_t b_extra[6];
      for (unsigned i = 0; i < 6; ++i)
         b_extra[i] = emit_get_sr(&p, 10 + i);
      uint32_t b_keep = b_extra[0];
      for (unsigned i = 1; i < 6; ++i)
         b_keep = emit_iadd(&p, b_keep, b_extra[i]);

      agx_apple9_block_begin(&p, C);
      uint32_t c_const = emit_imm32(&p, 0x3f800000u);
      uint32_t c_keep = emit_get_sr(&p, 20);

      agx_apple9_block_begin(&p, D);
      uint32_t phi = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
      agx_apple9_vir_instr *phi_ins = p.instructions[p.instruction_count - 1];
      uint32_t phi_k = agx_apple9_vir_emit_phi_canonical(&p, D, 2);
      agx_apple9_vir_instr *phi_k_ins = p.instructions[p.instruction_count - 1];
      uint32_t acc = live_a[0];
      for (unsigned i = 1; i < 8; ++i)
         acc = emit_iadd(&p, acc, live_a[i]);
      acc = emit_iadd(&p, acc, phi_k);
      p.output = emit_iadd(&p, acc, phi);

      ASSERT_TRUE(agx_apple9_vir_build_logical_cfg(&p));
      *agx_apple9_vir_src_ptr(phi_ins, agx_apple9_block_pred_index(D, B)) = b_val;
      *agx_apple9_vir_src_ptr(phi_ins, agx_apple9_block_pred_index(D, C)) = c_const;
      *agx_apple9_vir_src_ptr(phi_k_ins, agx_apple9_block_pred_index(D, B)) = b_keep;
      *agx_apple9_vir_src_ptr(phi_k_ins, agx_apple9_block_pred_index(D, C)) = c_keep;

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      apple9_ra_spill_options opts = apple9_ra_spill_default_options();
      opts.max_gprs = 9;
      opts.spill_reserved_gprs = 5;
      ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
      verify_vir_ssa_and_classes(&p, &state);

      /* Any is_imm value must be skipped by spill coalescing (no scratch word). */
      std::vector<uint32_t> mem_slot(p.value_count, UINT32_MAX);
      unsigned frame_words = 0;
      ASSERT_TRUE(apple9_ra_spill_coalesce_color(&p, &state, mem_slot.data(),
                                                 &frame_words));
      unsigned imm_count = 0;
      for (uint32_t v = 0; v < p.value_count; ++v) {
         if (apple9_ra_value_is_imm(&state, v)) {
            ++imm_count;
            EXPECT_EQ(mem_slot[v], UINT32_MAX);
            EXPECT_EQ(state.imm_val[v], 0x3f800000u);
         }
      }
      EXPECT_GT(imm_count, 0u);

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }

   /* 2. STORE_UNIFORM must NOT block rematerialization of literal-only VIR_IMM
    *    chains (only uniform-reading recipes depend on STORE_UNIFORM). */
   {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      nir_block nb;
      agx_apple9_block *b0 = make_cfg_block(&p, &nb);
      agx_apple9_block_begin(&p, b0);

      /* Emit a STORE_UNIFORM (preamble shader -> mutable uniforms). */
      uint32_t w = emit_imm32(&p, 7);
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &w, 1, 0));

      uint32_t lits[12];
      for (unsigned i = 0; i < 12; ++i)
         lits[i] = emit_imm32(&p, 0x10000000u + i);

      uint32_t acc = lits[11];
      for (int i = 10; i >= 0; --i)
         acc = emit_iadd(&p, acc, lits[i]);
      p.output = acc;

      apple9_ra_spill_state state;
      apple9_ra_spill_state_init(&state);
      apple9_ra_spill_options opts = apple9_ra_spill_default_options();
      opts.max_gprs = 6;
      opts.remat_only = true;
      opts.reserved_gprs = 0;
      ASSERT_TRUE(apple9_ra_spill(&p, &state, &opts));
      EXPECT_FALSE(state.has_spills);
      EXPECT_FALSE(state.uniform_remat_used);
      EXPECT_LE(apple9_ra_spill_calc_gpr_demand(&p, &state), 6u);

      apple9_ra_spill_state_finish(&state);
      agx_apple9_vir_finish(&p);
   }
}

