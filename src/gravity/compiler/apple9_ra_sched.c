/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_sched.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/dag.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_worklist.h"

void
apple9_ra_liveness_init(struct apple9_ra_liveness_info *info)
{
   memset(info, 0, sizeof(*info));
}

void
apple9_ra_liveness_finish(struct apple9_ra_liveness_info *info)
{
   if (!info)
      return;
   if (info->mem_ctx)
      ralloc_free(info->mem_ctx);
   memset(info, 0, sizeof(*info));
}

bool
apple9_ra_after_logical_end(const struct agx_apple9_vir_instr *ins)
{
   if (!ins || ins->local_control)
      return false;
   switch (ins->op) {
   case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
   case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
   case AGX_APPLE9_VIR_EXEC_MASK_POP:
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_JMP_EXEC_NONE:
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
   case AGX_APPLE9_VIR_ABI_RETURN:
   case AGX_APPLE9_VIR_HALT:
      return true;
   default:
      return false;
   }
}

bool
apple9_ra_is_publication_instr(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
          ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE;
}

bool
apple9_ra_is_publication_val(const struct agx_apple9_vir_program *p,
                             uint32_t value)
{
   return p && p->publication && value < p->publication_capacity &&
          p->publication[value];
}

static inline bool
is_pub_val(const struct agx_apple9_vir_program *p,
           const BITSET_WORD *pub_defs, uint32_t v)
{
   if (apple9_ra_is_publication_val(p, v))
      return true;
   return pub_defs && v < p->value_count && BITSET_TEST(pub_defs, v);
}

static inline bool
is_valid_gpr(const struct agx_apple9_vir_program *p,
             const BITSET_WORD *pub_defs, uint32_t v)
{
   return v != AGX_APPLE9_VREG_INVALID && v < p->value_count &&
          !is_pub_val(p, pub_defs, v);
}

static inline unsigned
split_src_tuple_width(const struct agx_apple9_vir_program *p,
                      struct agx_apple9_vir_instr *const *defs,
                      const struct agx_apple9_vir_instr *ins)
{
   if (ins->op != AGX_APPLE9_VIR_SPLIT || agx_apple9_vir_nr_srcs(ins) == 0)
      return 1;
   uint32_t src0 = agx_apple9_vir_src(ins, 0);
   if (src0 == AGX_APPLE9_VREG_INVALID || src0 >= p->value_count)
      return 0;
   const struct agx_apple9_vir_instr *def =
      defs ? defs[src0] : agx_apple9_definition(p, src0);
   if (def && def->dest == src0 && def->dest_components > 1) {
      unsigned w = MIN2((unsigned)ins->dest_components,
                        (unsigned)def->dest_components);
      return MIN2(w, p->value_count - src0);
   }
   return 1;
}

static bool
can_eliminate_instr(const struct agx_apple9_vir_program *p,
                    const BITSET_WORD *pub_defs,
                    const struct agx_apple9_vir_instr *ins)
{
   if (ins->op == AGX_APPLE9_VIR_PHI_SRC)
      return true;
   if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components == 0)
      return false;
   if (ins->publication_handoff || apple9_ra_is_publication_instr(ins))
      return false;
   if (is_pub_val(p, pub_defs, ins->dest))
      return false;
   if (ins->memory_volatile)
      return false;
   if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
       ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC)
      return false;
   if (apple9_ra_after_logical_end(ins))
      return false;
   return true;
}

static void
mark_val_live_dce(const struct agx_apple9_vir_program *p, uint32_t v,
                  const uint32_t *def_idx, const uint32_t *phi_src_head,
                  const uint32_t *phi_src_next, BITSET_WORD *val_live,
                  bool *ins_live, uint32_t *work, unsigned *work_top)
{
   if (v == AGX_APPLE9_VREG_INVALID || v >= p->value_count)
      return;
   if (BITSET_TEST(val_live, v))
      return;
   BITSET_SET(val_live, v);

   uint32_t d = def_idx[v];
   if (d != UINT32_MAX && !ins_live[d]) {
      ins_live[d] = true;
      work[(*work_top)++] = d;
   }
   if (phi_src_head) {
      for (uint32_t ps = phi_src_head[v]; ps != UINT32_MAX;
           ps = phi_src_next[ps]) {
         if (!ins_live[ps]) {
            ins_live[ps] = true;
            work[(*work_top)++] = ps;
         }
      }
   }
}

bool
apple9_ra_dce(struct agx_apple9_vir_program *p)
{
   if (!p)
      return false;
   const unsigned n = p->instruction_count;
   if (n == 0)
      return true;

   const unsigned vcount = p->value_count;
   const unsigned valloc = MAX2(vcount, 1u);

   uint32_t *def_idx = malloc(valloc * sizeof(*def_idx));
   BITSET_WORD *pub_defs = BITSET_CALLOC(valloc);
   BITSET_WORD *val_live = BITSET_CALLOC(valloc);
   bool *ins_live = calloc(n, sizeof(*ins_live));
   uint32_t *work = malloc(n * sizeof(*work));
   uint32_t *phi_src_head = NULL;
   uint32_t *phi_src_next = NULL;
   struct agx_apple9_vir_instr **to_remove = NULL;
   bool ok = def_idx && pub_defs && val_live && ins_live && work;
   if (!ok)
      goto cleanup;

   for (unsigned v = 0; v < valloc; ++v)
      def_idx[v] = UINT32_MAX;

   bool has_phi_src = false;
   for (unsigned i = 0; i < n; ++i) {
      const struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC)
         has_phi_src = true;
      if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
         bool pub = apple9_ra_is_publication_instr(ins);
         for (unsigned c = 0; c < ins->dest_components; ++c) {
            uint32_t d = ins->dest + c;
            if (d < vcount) {
               def_idx[d] = i;
               if (pub)
                  BITSET_SET(pub_defs, d);
            }
         }
      }
   }

   if (has_phi_src) {
      phi_src_head = malloc(valloc * sizeof(*phi_src_head));
      phi_src_next = malloc(n * sizeof(*phi_src_next));
      if (!phi_src_head || !phi_src_next) {
         ok = false;
         goto cleanup;
      }
      for (unsigned v = 0; v < valloc; ++v)
         phi_src_head[v] = UINT32_MAX;
      for (unsigned i = 0; i < n; ++i) {
         const struct agx_apple9_vir_instr *ins = p->instructions[i];
         phi_src_next[i] = UINT32_MAX;
         if (ins->op == AGX_APPLE9_VIR_PHI_SRC && ins->target < vcount) {
            phi_src_next[i] = phi_src_head[ins->target];
            phi_src_head[ins->target] = i;
         }
      }
   }

   unsigned work_top = 0;

   /* Seed program outputs and ABI live-outs. */
   if (p->output != AGX_APPLE9_VREG_INVALID) {
      mark_val_live_dce(p, p->output, def_idx, phi_src_head, phi_src_next,
                        val_live, ins_live, work, &work_top);
   }
   for (unsigned k = 0; k < p->live_out_count; ++k) {
      mark_val_live_dce(p, p->live_out[k], def_idx, phi_src_head, phi_src_next,
                        val_live, ins_live, work, &work_top);
   }

   /* Seed all non-eliminable instructions. */
   for (unsigned i = 0; i < n; ++i) {
      const struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (!can_eliminate_instr(p, pub_defs, ins)) {
         if (!ins_live[i]) {
            ins_live[i] = true;
            work[work_top++] = i;
         }
         /* Handoffs and publication producers must retain their destinations. */
         if ((ins->publication_handoff || apple9_ra_is_publication_instr(ins)) &&
             ins->dest != AGX_APPLE9_VREG_INVALID) {
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               if (ins->dest + c < vcount)
                  BITSET_SET(val_live, ins->dest + c);
            }
         }
      }
   }

   /* Propagate liveness backwards through SSA def-use edges. */
   while (work_top > 0) {
      uint32_t idx = work[--work_top];
      const struct agx_apple9_vir_instr *ins = p->instructions[idx];

      /* DEVICE_ATOMIC_RESULT is a structural publication record paired with
       * its preceding DEVICE_ATOMIC; it is not an independent shader use of
       * the atomic's return value. */
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         continue;

      if (ins->op == AGX_APPLE9_VIR_SPLIT && agx_apple9_vir_nr_srcs(ins) >= 1) {
         uint32_t src0 = agx_apple9_vir_src(ins, 0);
         unsigned w = 1;
         if (src0 < vcount && def_idx[src0] != UINT32_MAX) {
            const struct agx_apple9_vir_instr *def =
               p->instructions[def_idx[src0]];
            if (def->dest == src0 && def->dest_components > 1)
               w = MIN2((unsigned)ins->dest_components,
                        (unsigned)def->dest_components);
         }
         for (unsigned c = 0; c < w; ++c) {
            mark_val_live_dce(p, src0 + c, def_idx, phi_src_head, phi_src_next,
                              val_live, ins_live, work, &work_top);
         }
         continue;
      }

      const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t v = agx_apple9_vir_src(ins, s);
         mark_val_live_dce(p, v, def_idx, phi_src_head, phi_src_next, val_live,
                           ins_live, work, &work_top);
      }
   }

   /* Sweep dead instructions and mark partially/fully dead destinations. */
   unsigned remove_count = 0;
   for (unsigned i = 0; i < n; ++i) {
      struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (!ins_live[i]) {
         if (!to_remove) {
            to_remove = malloc(n * sizeof(*to_remove));
            if (!to_remove) {
               ok = false;
               goto cleanup;
            }
         }
         to_remove[remove_count++] = ins;
         continue;
      }

      if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
         if (ins->publication_handoff || apple9_ra_is_publication_instr(ins) ||
             is_pub_val(p, pub_defs, ins->dest)) {
            ins->dead_dest_mask = 0;
         } else {
            uint8_t dead_mask = 0;
            for (unsigned c = 0; c < ins->dest_components && c < 8; ++c) {
               uint32_t d = ins->dest + c;
               if (d < vcount && !BITSET_TEST(val_live, d))
                  dead_mask |= (uint8_t)(1u << c);
            }
            ins->dead_dest_mask = dead_mask;
         }
      }
   }

   if (remove_count > 0)
      agx_apple9_vir_remove_many(p, to_remove, remove_count);

cleanup:
   free(to_remove);
   free(phi_src_next);
   free(phi_src_head);
   free(work);
   free(ins_live);
   free(val_live);
   free(pub_defs);
   free(def_idx);
   return ok;
}

static unsigned
liveness_ins_update_internal(const struct agx_apple9_vir_program *p,
                             struct agx_apple9_vir_instr *const *defs,
                             const BITSET_WORD *pub_defs,
                             struct u_sparse_bitset *live,
                             struct agx_apple9_vir_instr *ins,
                             bool update_kill_mask, unsigned *cur_live)
{
   if (ins->op == AGX_APPLE9_VIR_PHI ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
      if (update_kill_mask)
         ins->kill_mask = 0;
      return 0;
   }

   /* Count active (non-dead, non-publication) defs that are not already live
    * after this instruction (e.g. publication_handoff) for peak accounting. */
   unsigned unlive_defs = 0;
   if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
      for (unsigned c = 0; c < ins->dest_components; ++c) {
         uint32_t d = ins->dest + c;
         if (!is_valid_gpr(p, pub_defs, d))
            continue;
         if (u_sparse_bitset_test(live, d)) {
            u_sparse_bitset_clear(live, d);
            if (cur_live)
               (*cur_live)--;
         } else if (c >= 8 || !(ins->dead_dest_mask & (1u << c))) {
            unlive_defs++;
         }
      }
   }

   if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
      if (update_kill_mask)
         ins->kill_mask = 0;
      for (unsigned k = 0; k < p->live_out_count; ++k) {
         if (p->live_out_reg[k] == AGX_APPLE9_PHYS_INVALID)
            continue;
         uint32_t v = p->live_out[k];
         if (is_valid_gpr(p, pub_defs, v) && !u_sparse_bitset_test(live, v)) {
            u_sparse_bitset_set(live, v);
            if (cur_live)
               (*cur_live)++;
         }
      }
      return unlive_defs;
   }

   if (ins->op == AGX_APPLE9_VIR_SPLIT) {
      uint16_t kill_mask = 0;
      if (agx_apple9_vir_nr_srcs(ins) >= 1) {
         uint32_t src0 = agx_apple9_vir_src(ins, 0);
         unsigned w = split_src_tuple_width(p, defs, ins);
         bool any_valid = false;
         bool any_live = false;
         for (unsigned c = 0; c < w; ++c) {
            uint32_t v = src0 + c;
            if (!is_valid_gpr(p, pub_defs, v))
               continue;
            any_valid = true;
            if (u_sparse_bitset_test(live, v))
               any_live = true;
         }
         if (any_valid && !any_live)
            kill_mask = 1u;
         for (unsigned c = 0; c < w; ++c) {
            uint32_t v = src0 + c;
            if (is_valid_gpr(p, pub_defs, v) && !u_sparse_bitset_test(live, v)) {
               u_sparse_bitset_set(live, v);
               if (cur_live)
                  (*cur_live)++;
            }
         }
      }
      if (update_kill_mask)
         ins->kill_mask = kill_mask;
      return unlive_defs;
   }

   /* Evaluate kill_mask against live_after before inserting any source so that
    * duplicate operands (e.g. IADD(v, v) or IOR(v, v)) agree on kill status. */
   const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
   if (update_kill_mask) {
      uint16_t kill_mask = 0;
      for (unsigned s = 0; s < nr_srcs && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
         uint32_t v = agx_apple9_vir_src(ins, s);
         if (is_valid_gpr(p, pub_defs, v) && !u_sparse_bitset_test(live, v))
            kill_mask |= (uint16_t)(1u << s);
      }
      ins->kill_mask = kill_mask;
   }

   for (unsigned s = 0; s < nr_srcs; ++s) {
      uint32_t v = agx_apple9_vir_src(ins, s);
      if (is_valid_gpr(p, pub_defs, v) && !u_sparse_bitset_test(live, v)) {
         u_sparse_bitset_set(live, v);
         if (cur_live)
            (*cur_live)++;
      }
   }

   return unlive_defs;
}

void
apple9_ra_liveness_ins_update(const struct agx_apple9_vir_program *p,
                              struct u_sparse_bitset *live,
                              struct agx_apple9_vir_instr *ins)
{
   if (!p || !live || !ins)
      return;
   liveness_ins_update_internal(p, NULL, NULL, live, ins, true, NULL);
}

static bool
build_defs_and_pub_mask(const struct agx_apple9_vir_program *p, void *mem_ctx,
                        struct agx_apple9_vir_instr ***defs_out,
                        BITSET_WORD **pub_defs_out)
{
   const unsigned valloc = MAX2(p->value_count, 1u);
   struct agx_apple9_vir_instr **defs =
      rzalloc_array(mem_ctx, struct agx_apple9_vir_instr *, valloc);
   BITSET_WORD *pub_defs = BITSET_RZALLOC(mem_ctx, valloc);
   if (!defs || !pub_defs)
      return false;

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components == 0)
         continue;
      bool pub = apple9_ra_is_publication_instr(ins);
      for (unsigned c = 0; c < ins->dest_components; ++c) {
         uint32_t d = ins->dest + c;
         if (d < p->value_count) {
            defs[d] = ins;
            if (pub)
               BITSET_SET(pub_defs, d);
         }
      }
   }

   *defs_out = defs;
   *pub_defs_out = pub_defs;
   return true;
}

bool
apple9_ra_liveness(struct agx_apple9_vir_program *p,
                   struct apple9_ra_liveness_info *out)
{
   if (!p)
      return false;

   struct apple9_ra_liveness_info local_info;
   struct apple9_ra_liveness_info *info = out ? out : &local_info;
   apple9_ra_liveness_init(info);

   if (p->num_cfg_blocks == 0 && p->blocks != NULL) {
      if (!agx_apple9_vir_build_logical_cfg(p))
         return false;
   }

   if (p->num_cfg_blocks == 0) {
      p->peak_live_gprs = 0;
      return true;
   }

   const unsigned num_blocks = p->num_cfg_blocks;
   const unsigned vcount = p->value_count;
   void *mem_ctx = ralloc_context(NULL);
   void *scratch_ctx = ralloc_context(NULL);
   if (!mem_ctx || !scratch_ctx) {
      ralloc_free(mem_ctx);
      ralloc_free(scratch_ctx);
      return false;
   }

   info->mem_ctx = mem_ctx;
   info->num_cfg_blocks = num_blocks;
   info->value_count = vcount;
   info->live_in = rzalloc_array(mem_ctx, struct u_sparse_bitset, num_blocks);
   info->live_out = rzalloc_array(mem_ctx, struct u_sparse_bitset, num_blocks);
   info->block_peak_gprs = rzalloc_array(mem_ctx, unsigned, num_blocks);

   struct agx_apple9_vir_instr **defs = NULL;
   BITSET_WORD *pub_defs = NULL;
   struct agx_apple9_block **cfg_blocks =
      rzalloc_array(scratch_ctx, struct agx_apple9_block *, num_blocks);
   if (!info->live_in || !info->live_out || !info->block_peak_gprs ||
       !cfg_blocks ||
       !build_defs_and_pub_mask(p, scratch_ctx, &defs, &pub_defs)) {
      ralloc_free(scratch_ctx);
      apple9_ra_liveness_finish(info);
      return false;
   }

   u_worklist worklist;
   u_worklist_init(&worklist, num_blocks, scratch_ctx);

   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      if (!blk->cfg_active) {
         /* Clear stale kill bits in synthetic / unreachable blocks. */
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
              I; I = agx_apple9_instr_next(I)) {
            I->kill_mask = 0;
         }
         continue;
      }

      const unsigned b = blk->cfg_index;
      assert(b < num_blocks);
      cfg_blocks[b] = blk;

      u_sparse_bitset_init(&info->live_in[b], vcount, mem_ctx);
      u_sparse_bitset_init(&info->live_out[b], vcount, mem_ctx);

      /* Seed live-out values on logical CFG exit blocks. */
      if (!blk->cfg_succs[0] && !blk->cfg_succs[1]) {
         bool has_abi_ret = false;
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
              I; I = agx_apple9_instr_next(I)) {
            if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
               has_abi_ret = true;
               break;
            }
         }
         if (is_valid_gpr(p, pub_defs, p->output))
            u_sparse_bitset_set(&info->live_out[b], p->output);
         for (unsigned k = 0; k < p->live_out_count; ++k) {
            if (has_abi_ret && p->live_out_reg[k] != AGX_APPLE9_PHYS_INVALID)
               continue;
            uint32_t v = p->live_out[k];
            if (is_valid_gpr(p, pub_defs, v))
               u_sparse_bitset_set(&info->live_out[b], v);
         }
      }

      /* Push in forward order at head so pop_head visits blocks backwards. */
      u_worklist_push_head(&worklist, blk, cfg_index);
   }

   /* Backwards-may dataflow fixed point over the logical SSA CFG. */
   while (!u_worklist_is_empty(&worklist)) {
      struct agx_apple9_block *blk =
         u_worklist_pop_head(&worklist, struct agx_apple9_block, cfg_index);
      const unsigned b = blk->cfg_index;

      u_sparse_bitset_free(&info->live_in[b]);
      u_sparse_bitset_dup_with_ctx(&info->live_in[b], &info->live_out[b],
                                   mem_ctx);

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(blk);
           I; I = agx_apple9_instr_prev(I)) {
         if (I->op != AGX_APPLE9_VIR_PHI) {
            liveness_ins_update_internal(p, defs, pub_defs, &info->live_in[b],
                                         I, false, NULL);
         }
      }

      /* Propagate live_in across each incoming edge, applying canonical
       * VIR_PHI semantics in parallel: kill all phi dests, then gen each
       * phi's operand for that specific predecessor index. */
      for (unsigned p_idx = 0; p_idx < blk->num_cfg_preds; ++p_idx) {
         struct agx_apple9_block *pred = blk->cfg_preds[p_idx];
         assert(pred && pred->cfg_active);

         struct u_sparse_bitset edge_live;
         u_sparse_bitset_dup_with_ctx(&edge_live, &info->live_in[b],
                                      scratch_ctx);

         for (struct agx_apple9_vir_instr *phi =
                 agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (phi->dest != AGX_APPLE9_VREG_INVALID &&
                is_valid_gpr(p, pub_defs, phi->dest))
               u_sparse_bitset_clear(&edge_live, phi->dest);
         }

         for (struct agx_apple9_vir_instr *phi =
                 agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (p_idx < agx_apple9_vir_nr_srcs(phi)) {
               uint32_t v = agx_apple9_vir_src(phi, p_idx);
               if (is_valid_gpr(p, pub_defs, v))
                  u_sparse_bitset_set(&edge_live, v);
            }
         }

         if (u_sparse_bitset_merge(&info->live_out[pred->cfg_index],
                                   &edge_live)) {
            u_worklist_push_tail(&worklist, pred, cfg_index);
         }

         u_sparse_bitset_free(&edge_live);
      }
   }

   /* Final pass from converged live_out sets: compute exact kill_mask on every
    * instruction and exact per-block / global peak 32-bit GPR pressure. */
   unsigned global_peak = 0;
   for (unsigned b = 0; b < num_blocks; ++b) {
      struct agx_apple9_block *blk = cfg_blocks[b];
      if (!blk)
         continue;

      struct u_sparse_bitset live;
      u_sparse_bitset_dup_with_ctx(&live, &info->live_out[b], scratch_ctx);
      unsigned cur = u_sparse_bitset_count(&live);
      unsigned block_peak = cur;

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(blk);
           I; I = agx_apple9_instr_prev(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI) {
            I->kill_mask = 0;
            continue;
         }
         unsigned cur_after = cur;
         unsigned unlive_defs = liveness_ins_update_internal(
            p, defs, pub_defs, &live, I, true, &cur);
         block_peak = MAX2(block_peak, cur_after + unlive_defs);
         block_peak = MAX2(block_peak, cur);
      }

      info->block_peak_gprs[b] = block_peak;
      global_peak = MAX2(global_peak, block_peak);
      u_sparse_bitset_free(&live);
   }

   info->peak_live_gprs = global_peak;
   p->peak_live_gprs = global_peak;

   u_worklist_fini(&worklist);
   ralloc_free(scratch_ctx);

   if (!out)
      apple9_ra_liveness_finish(&local_info);
   return true;
}

void
apple9_ra_schedule_resources(const struct agx_apple9_vir_instr *ins,
                             uint32_t *reads, uint32_t *writes)
{
   /* Every lane operation observes the active execution/coverage state. */
   *reads = APPLE9_RA_SCHED_STATE;
   *writes = 0;

   switch (ins->op) {
   case AGX_APPLE9_VIR_MOV:
   case AGX_APPLE9_VIR_COLLECT:
   case AGX_APPLE9_VIR_SPLIT:
   case AGX_APPLE9_VIR_PHI:
   case AGX_APPLE9_VIR_ENTRY_REG:
      return;
   case AGX_APPLE9_VIR_SWAP:
      *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   default:
      break;
   }

   if (agx_apple9_instr_is_pure_alu(ins))
      return;

   if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
       agx_apple9_encoding_info(ins->encoding)->requires_prior_completion) {
      *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   }

   if (ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) {
      /* Scalar publication exports have 1-cycle latency and borrow their
       * source GPRs until the publication consumer retires their slot; keep
       * them ordered with surrounding ALU so RA does not clobber a borrowed
       * source and force a standalone WAIT. */
      *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   }

   if (ins->publication_handoff || apple9_ra_is_publication_instr(ins)) {
      *reads |= APPLE9_RA_SCHED_PUBLICATION;
      *writes |= APPLE9_RA_SCHED_PUBLICATION;
   }

   switch (ins->op) {
   case AGX_APPLE9_VIR_MESH_CONTROL:
   case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
   case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL:
      *reads |= APPLE9_RA_SCHED_MEMORY | APPLE9_RA_SCHED_MESH;
      *writes |= APPLE9_RA_SCHED_MEMORY | APPLE9_RA_SCHED_MESH;
      return;
   case AGX_APPLE9_VIR_GET_SR:
      if (ins->encoding == AGX_APPLE9_ENC_GET_DRAW_ID ||
          ins->encoding == AGX_APPLE9_ENC_GET_COVERAGE)
         *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   case AGX_APPLE9_VIR_GET_GLOBAL_ID:
   case AGX_APPLE9_VIR_DERIVATIVE:
   case AGX_APPLE9_VIR_MMA:
      return;
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SPILL_LOAD:
      *reads |= APPLE9_RA_SCHED_MEMORY;
      if (ins->memory_volatile)
         *writes |= APPLE9_RA_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_DEVICE_STORE:
   case AGX_APPLE9_VIR_PRIVATE_STORE:
   case AGX_APPLE9_VIR_SHARED_STORE:
   case AGX_APPLE9_VIR_SPILL_STORE:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
      *reads |= APPLE9_RA_SCHED_MEMORY;
      *writes |= APPLE9_RA_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_DEVICE_ATOMIC: {
      /* Keep returning DEVICE_ATOMIC strictly adjacent to its immediately
       * following DEVICE_ATOMIC_RESULT record. */
      struct agx_apple9_vir_instr *next = agx_apple9_instr_next(ins);
      if (next && next->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
         *reads = *writes = APPLE9_RA_SCHED_ALL;
      } else {
         *reads |= APPLE9_RA_SCHED_MEMORY;
         *writes |= APPLE9_RA_SCHED_MEMORY;
      }
      return;
   }
   case AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT:
      *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   case AGX_APPLE9_VIR_IMAGE_STORE:
   case AGX_APPLE9_VIR_BLOCK_IMAGE_STORE:
      *reads |= APPLE9_RA_SCHED_MEMORY | APPLE9_RA_SCHED_PUBLICATION;
      *writes |= APPLE9_RA_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      *reads |= APPLE9_RA_SCHED_MEMORY | APPLE9_RA_SCHED_PUBLICATION;
      *writes |= APPLE9_RA_SCHED_PUBLICATION;
      return;
   case AGX_APPLE9_VIR_INTERPOLATION_POSITION:
   case AGX_APPLE9_VIR_ITER:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_FMUL_PROJECT:
   case AGX_APPLE9_VIR_CUBE:
      *reads |= APPLE9_RA_SCHED_INTERPOLATION;
      *writes |= APPLE9_RA_SCHED_INTERPOLATION;
      return;
   case AGX_APPLE9_VIR_TILE_LOAD:
      *reads |= APPLE9_RA_SCHED_TILE;
      if (ins->memory_volatile)
         *writes |= APPLE9_RA_SCHED_TILE;
      return;
   case AGX_APPLE9_VIR_TILE_STORE:
      *reads |= APPLE9_RA_SCHED_TILE;
      *writes |= APPLE9_RA_SCHED_TILE;
      return;
   case AGX_APPLE9_VIR_PUBLICATION_TUPLE:
      return;
   default:
      if (*writes == APPLE9_RA_SCHED_PUBLICATION)
         return;
      *reads = *writes = APPLE9_RA_SCHED_ALL;
      return;
   }
}

unsigned
apple9_ra_dependency_delay(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_TILE_LOAD:
      return 16;
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_HRCP:
   case AGX_APPLE9_VIR_HRCP_F32:
   case AGX_APPLE9_VIR_HRSQ:
   case AGX_APPLE9_VIR_HSQRT_FACTOR:
   case AGX_APPLE9_VIR_HEXP2:
   case AGX_APPLE9_VIR_HLOG2:
   case AGX_APPLE9_VIR_SUBGROUP_REDUCE:
   case AGX_APPLE9_VIR_SUBGROUP_SHUFFLE:
   case AGX_APPLE9_VIR_SUBGROUP_ROTATE:
   case AGX_APPLE9_VIR_MMA:
      return 4;
   default:
      return 1;
   }
}

static const struct agx_apple9_operand_constraint *
vir_source_constraint(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return NULL;
   if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
       source + ins->texture_bindless + ins->sampler_bindless >= ins->nr_srcs)
      return NULL; /* Bindless table indices allow full r0..r95. */
   if (ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
       ins->op == AGX_APPLE9_VIR_PRIVATE_STORE) {
      unsigned data_gprs = agx_apple9_mem_data_gprs(ins);
      return agx_apple9_find_operand(ins->encoding,
                                     source >= data_gprs
                                        ? AGX_APPLE9_OPERAND_SRC1
                                        : AGX_APPLE9_OPERAND_SRC0);
   }
   if (ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
       ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) {
      unsigned data_gprs = ins->op == AGX_APPLE9_VIR_DEVICE_STORE
                              ? agx_apple9_mem_data_gprs(ins)
                              : MAX2((unsigned)ins->memory_components, 1u);
      if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT ||
           ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT) &&
          source > data_gprs) {
         return agx_apple9_find_operand(ins->encoding,
                                        source == data_gprs + 1
                                           ? AGX_APPLE9_OPERAND_SRC1
                                           : AGX_APPLE9_OPERAND_SRC2);
      }
      enum agx_apple9_operand_role role =
         source >= data_gprs ? AGX_APPLE9_OPERAND_INDEX
         : ins->op == AGX_APPLE9_VIR_DEVICE_STORE ? AGX_APPLE9_OPERAND_STORE_DATA
                                                  : AGX_APPLE9_OPERAND_ATOMIC_DATA;
      return agx_apple9_find_operand(ins->encoding, role);
   }

   if (ins->op == AGX_APPLE9_VIR_MMA)
      return agx_apple9_find_operand(ins->encoding,
                                     agx_apple9_mma_source_role(ins, source));

   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(ins->encoding);
   for (unsigned o = 0, s = 0; o < info->operand_count; ++o) {
      const struct agx_apple9_operand_constraint *c = &info->operands[o];
      if (!(c->files & AGX_APPLE9_FILE_GPR) ||
          agx_apple9_inline_source_file(ins, c->role) ||
          c->role == AGX_APPLE9_OPERAND_DEST)
         continue;
      if (s++ == source)
         return c;
   }
   return NULL;
}

static uint8_t *
compute_value_max_gprs(const struct agx_apple9_vir_program *p, void *mem_ctx)
{
   const unsigned vcount = p->value_count;
   uint8_t *max_gpr = ralloc_array(mem_ctx, uint8_t, MAX2(vcount, 1u));
   if (!max_gpr)
      return NULL;

   for (unsigned v = 0; v < vcount; ++v) {
      uint8_t bound = AGX_APPLE9_GPR_COUNT - 1;
      if (p->fixed_phys && p->fixed_phys[v] < AGX_APPLE9_GPR_COUNT)
         bound = MIN2(bound, p->fixed_phys[v]);
      if (p->max_phys && p->max_phys[v] < AGX_APPLE9_GPR_COUNT)
         bound = MIN2(bound, p->max_phys[v]);
      max_gpr[v] = bound;
   }

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (ins->encoding == AGX_APPLE9_ENC_PSEUDO)
         continue;

      if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
         const struct agx_apple9_operand_constraint *dc =
            ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC
               ? agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT,
                                         AGX_APPLE9_OPERAND_SRC0)
               : agx_apple9_find_operand(ins->encoding,
                                         AGX_APPLE9_OPERAND_DEST);
         if (dc && (dc->files & AGX_APPLE9_FILE_GPR) &&
             dc->max_index < AGX_APPLE9_GPR_COUNT) {
            uint8_t dmax = dc->max_index;
            if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
                (ins->texture_bindless || ins->sampler_bindless))
               dmax = AGX_APPLE9_GPR_COUNT - 1;
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               if (ins->dest + c < vcount)
                  max_gpr[ins->dest + c] = MIN2(max_gpr[ins->dest + c], dmax);
            }
         }
      }

      const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t v = agx_apple9_vir_src(ins, s);
         if (v >= vcount)
            continue;
         const struct agx_apple9_operand_constraint *sc =
            vir_source_constraint(ins, s);
         if (sc && (sc->files & AGX_APPLE9_FILE_GPR) &&
             sc->max_index < AGX_APPLE9_GPR_COUNT)
            max_gpr[v] = MIN2(max_gpr[v], sc->max_index);
      }
   }

   return max_gpr;
}

/* Native VIR intra-block pressure / latency / resource DAG scheduler. */

struct vir_sched_node {
   struct dag_node dag;
   struct agx_apple9_vir_instr *instr;
   unsigned ready;
   unsigned orig_idx;
};

struct vir_resource_reader {
   struct vir_sched_node *node;
   struct vir_resource_reader *next;
};

struct vir_pressure_delta {
   int32_t all;
   int32_t bank64;
   int32_t bank16;
};

struct vir_sched_state {
   const struct agx_apple9_vir_program *p;
   struct agx_apple9_vir_instr *const *defs;
   const BITSET_WORD *pub_defs;
   const uint8_t *val_max_gpr;
   struct dag *dag;
   struct u_sparse_bitset live;
   bool latency;
   unsigned position;
   int32_t cur_all;
   int32_t cur_64;
   int32_t cur_16;
};

static inline void
add_dep(struct vir_sched_node *a, struct vir_sched_node *b)
{
   assert(a != b && "no self-dependencies");
   if (a && b)
      dag_add_edge(&a->dag, &b->dag, 0);
}

static inline void
serialize_node(struct vir_sched_node *a, struct vir_sched_node **b)
{
   add_dep(a, *b);
   *b = a;
}

static inline bool
is_pinned_entry(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_PHI:
   case AGX_APPLE9_VIR_ENTRY_REG:
   case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
   case AGX_APPLE9_VIR_EXEC_MASK_POP:
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
      return !ins->local_control;
   default:
      return false;
   }
}

static inline bool
is_pinned_tail(const struct agx_apple9_vir_instr *ins)
{
   return apple9_ra_after_logical_end(ins) ||
          ins->op == AGX_APPLE9_VIR_PHI_SRC;
}

static struct vir_pressure_delta
calc_pressure_delta(const struct agx_apple9_vir_program *p,
                    struct agx_apple9_vir_instr *const *defs,
                    const BITSET_WORD *pub_defs, const uint8_t *val_max_gpr,
                    const struct agx_apple9_vir_instr *ins,
                    struct u_sparse_bitset *live)
{
   struct vir_pressure_delta d = {0, 0, 0};

   if (ins->op == AGX_APPLE9_VIR_PHI ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
      return d;

   if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
      for (unsigned c = 0; c < ins->dest_components; ++c) {
         uint32_t v = ins->dest + c;
         if (is_valid_gpr(p, pub_defs, v) && u_sparse_bitset_test(live, v)) {
            d.all -= 1;
            if (val_max_gpr[v] <= 63)
               d.bank64 -= 1;
            if (val_max_gpr[v] <= 15)
               d.bank16 -= 1;
         }
      }
   }

   if (ins->op == AGX_APPLE9_VIR_SPLIT) {
      if (agx_apple9_vir_nr_srcs(ins) >= 1) {
         uint32_t src0 = agx_apple9_vir_src(ins, 0);
         unsigned w = split_src_tuple_width(p, defs, ins);
         for (unsigned c = 0; c < w; ++c) {
            uint32_t v = src0 + c;
            if (is_valid_gpr(p, pub_defs, v) && !u_sparse_bitset_test(live, v)) {
               d.all += 1;
               if (val_max_gpr[v] <= 63)
                  d.bank64 += 1;
               if (val_max_gpr[v] <= 15)
                  d.bank16 += 1;
            }
         }
      }
      return d;
   }

   const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
   for (unsigned s = 0; s < nr_srcs; ++s) {
      uint32_t v = agx_apple9_vir_src(ins, s);
      if (!is_valid_gpr(p, pub_defs, v))
         continue;
      bool dupe = false;
      for (unsigned prev = 0; prev < s; ++prev) {
         if (agx_apple9_vir_src(ins, prev) == v) {
            dupe = true;
            break;
         }
      }
      if (!dupe && !u_sparse_bitset_test(live, v)) {
         d.all += 1;
         if (val_max_gpr[v] <= 63)
            d.bank64 += 1;
         if (val_max_gpr[v] <= 15)
            d.bank16 += 1;
      }
   }

   return d;
}

static struct dag *
create_vir_dag(const struct agx_apple9_vir_program *p,
               struct agx_apple9_vir_instr *const *defs,
               struct agx_apple9_vir_instr *first_body,
               struct agx_apple9_vir_instr *first_tail, void *memctx,
               struct vir_sched_node **last_write,
               struct util_dynarray *orig_order)
{
   struct dag *dag = dag_create(memctx);
   if (!dag)
      return NULL;

   struct vir_sched_node *preload = NULL;
   struct vir_sched_node *resource_writer[32] = {0};
   struct vir_resource_reader *resource_readers[32] = {0};

   for (struct agx_apple9_vir_instr *I = first_body; I && I != first_tail;
        I = agx_apple9_instr_next(I)) {
      struct vir_sched_node *node = rzalloc(memctx, struct vir_sched_node);
      if (!node)
         return NULL;
      node->instr = I;
      dag_init_node(dag, &node->dag);
      util_dynarray_append(orig_order, node);

      /* SSA read-after-write dependencies (including tuple components). */
      if (I->op == AGX_APPLE9_VIR_SPLIT && agx_apple9_vir_nr_srcs(I) >= 1) {
         uint32_t src0 = agx_apple9_vir_src(I, 0);
         unsigned w = split_src_tuple_width(p, defs, I);
         for (unsigned c = 0; c < w; ++c) {
            if (src0 + c < p->value_count)
               add_dep(node, last_write[src0 + c]);
         }
      } else {
         const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < p->value_count)
               add_dep(node, last_write[v]);
         }
      }

      if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->dest + c < p->value_count)
               last_write[I->dest + c] = node;
         }
      }

      if (is_pinned_entry(I))
         serialize_node(node, &preload);
      else
         add_dep(node, preload);

      /* Implicit hardware resource hazards (memory, barriers, publications,
       * interpolation, tile, mesh, execution state). */
      uint32_t reads = 0, writes = 0;
      apple9_ra_schedule_resources(I, &reads, &writes);
      u_foreach_bit(r, reads | writes) {
         add_dep(node, resource_writer[r]);
         if (writes & BITFIELD_BIT(r)) {
            for (struct vir_resource_reader *rr = resource_readers[r]; rr;
                 rr = rr->next) {
               add_dep(node, rr->node);
            }
            resource_readers[r] = NULL;
            resource_writer[r] = node;
         } else {
            struct vir_resource_reader *rr =
               rzalloc(memctx, struct vir_resource_reader);
            if (!rr)
               return NULL;
            rr->node = node;
            rr->next = resource_readers[r];
            resource_readers[r] = rr;
         }
      }
   }

   /* Clear last_write entries touched by this block in O(block_size). */
   for (struct agx_apple9_vir_instr *I = first_body; I && I != first_tail;
        I = agx_apple9_instr_next(I)) {
      if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->dest + c < p->value_count)
               last_write[I->dest + c] = NULL;
         }
      }
   }

   return dag;
}

static struct vir_sched_node *
choose_vir_instr(struct vir_sched_state *s)
{
   struct vir_sched_node *best = NULL;
   unsigned best_dependency = 0;
   int32_t best_score = INT32_MAX;
   struct vir_pressure_delta best_delta = {INT32_MAX, INT32_MAX, INT32_MAX};

   if (s->latency) {
      unsigned earliest = UINT_MAX;
      list_for_each_entry(struct vir_sched_node, n, &s->dag->heads, dag.link)
         earliest = MIN2(earliest, n->ready);
      s->position = MAX2(s->position, earliest);
   }

   list_for_each_entry(struct vir_sched_node, n, &s->dag->heads, dag.link) {
      if (s->latency && n->ready > s->position)
         continue;

      struct vir_pressure_delta delta = calc_pressure_delta(
         s->p, s->defs, s->pub_defs, s->val_max_gpr, n->instr, &s->live);

      /* Weight scarce r0..r15 and r0..r63 bank lifetimes when bank pressure
       * approaches hardware limits so we close constrained ranges early. */
      int32_t score = delta.all;
      if (s->cur_16 >= 12 || s->cur_16 + delta.bank16 > 16)
         score += 4 * delta.bank16;
      if (s->cur_64 >= 56 || s->cur_64 + delta.bank64 > 64)
         score += 2 * delta.bank64;

      unsigned dependency = 0;
      if (s->latency) {
         util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
            struct vir_sched_node *producer =
               (struct vir_sched_node *)edge->child;
            dependency = MAX2(dependency,
                              apple9_ra_dependency_delay(producer->instr) - 1);
         }
      }

      bool better = false;
      if (!best) {
         better = true;
      } else if (dependency != best_dependency) {
         better = dependency > best_dependency;
      } else if (score != best_score) {
         better = score < best_score;
      } else if (delta.all != best_delta.all) {
         better = delta.all < best_delta.all;
      } else if (delta.all > 0) {
         /* Depth-first expansion on positive-pressure ties avoids opening an
          * entire reduction tree breadth-first. */
         better = true;
      }

      if (better) {
         best = n;
         best_dependency = dependency;
         best_score = score;
         best_delta = delta;
      }
   }

   return best;
}

static unsigned
compute_schedule_stalls(struct vir_sched_node *const *nodes, unsigned count,
                        bool reverse, struct vir_sched_node *const *dep_pool,
                        const unsigned *dep_start)
{
   for (unsigned i = 0; i < count; ++i)
      nodes[i]->ready = 0;

   unsigned issue = 0, stalls = 0;
   for (unsigned k = 0; k < count; ++k) {
      struct vir_sched_node *node = nodes[reverse ? (count - 1 - k) : k];
      unsigned earliest = issue;
      for (unsigned e = dep_start[node->orig_idx];
           e < dep_start[node->orig_idx + 1]; ++e) {
         earliest = MAX2(earliest, dep_pool[e]->ready);
      }
      stalls += earliest - issue;
      node->ready = earliest + apple9_ra_dependency_delay(node->instr);
      issue = earliest + 1;
   }

   for (unsigned i = 0; i < count; ++i)
      nodes[i]->ready = 0;
   return stalls;
}

struct vir_sched_result {
   struct agx_apple9_vir_instr **order; /* bottom-up: [0] is last, [count-1] is first */
   unsigned count;
   int32_t max_all;
   int32_t max_64;
   int32_t max_16;
   unsigned stalls;
   bool valid;
};

static struct vir_sched_result
run_block_schedule_attempt(const struct agx_apple9_vir_program *p,
                           struct agx_apple9_vir_instr *const *defs,
                           const BITSET_WORD *pub_defs,
                           const uint8_t *val_max_gpr,
                           struct agx_apple9_vir_instr *first_body,
                           struct agx_apple9_vir_instr *first_tail,
                           struct u_sparse_bitset *live_after_body,
                           int32_t base_all, int32_t base_64, int32_t base_16,
                           bool latency, struct vir_sched_node **last_write,
                           void *attempt_ctx, unsigned *orig_stalls_out)
{
   struct vir_sched_result res = {0};
   struct util_dynarray orig_order = UTIL_DYNARRAY_INIT;
   util_dynarray_init(&orig_order, attempt_ctx);

   struct dag *dag = create_vir_dag(p, defs, first_body, first_tail,
                                    attempt_ctx, last_write, &orig_order);
   if (!dag)
      return res;

   struct vir_sched_node *const *orig =
      (struct vir_sched_node *const *)orig_order.data;
   const unsigned count =
      util_dynarray_num_elements(&orig_order, struct vir_sched_node *);
   if (count <= 1)
      return res;

   struct vir_sched_node **dep_pool = NULL;
   unsigned *dep_start = NULL;
   if (latency) {
      unsigned total_edges = 0;
      for (unsigned i = 0; i < count; ++i) {
         orig[i]->orig_idx = i;
         total_edges +=
            util_dynarray_num_elements(&orig[i]->dag.edges, struct dag_edge);
      }
      dep_start = rzalloc_array(attempt_ctx, unsigned, count + 1);
      dep_pool =
         rzalloc_array(attempt_ctx, struct vir_sched_node *, MAX2(total_edges, 1u));
      if (!dep_start || !dep_pool)
         return res;
      unsigned pos = 0;
      for (unsigned i = 0; i < count; ++i) {
         dep_start[i] = pos;
         util_dynarray_foreach(&orig[i]->dag.edges, struct dag_edge, edge) {
            if (edge->child)
               dep_pool[pos++] = (struct vir_sched_node *)edge->child;
         }
      }
      dep_start[count] = pos;
      if (orig_stalls_out) {
         *orig_stalls_out =
            compute_schedule_stalls(orig, count, false, dep_pool, dep_start);
      }
   }

   struct vir_sched_node **sched_nodes =
      ralloc_array(attempt_ctx, struct vir_sched_node *, count);
   res.order = ralloc_array(attempt_ctx, struct agx_apple9_vir_instr *, count);
   if (!sched_nodes || !res.order)
      return res;

   struct vir_sched_state s = {
      .p = p,
      .defs = defs,
      .pub_defs = pub_defs,
      .val_max_gpr = val_max_gpr,
      .dag = dag,
      .latency = latency,
      .position = 0,
      .cur_all = base_all,
      .cur_64 = base_64,
      .cur_16 = base_16,
   };
   u_sparse_bitset_dup_with_ctx(&s.live, live_after_body, attempt_ctx);

   res.max_all = base_all;
   res.max_64 = base_64;
   res.max_16 = base_16;
   unsigned nr = 0;

   while (!list_is_empty(&s.dag->heads)) {
      struct vir_sched_node *node = choose_vir_instr(&s);
      assert(node != NULL);

      struct vir_pressure_delta delta = calc_pressure_delta(
         p, defs, pub_defs, val_max_gpr, node->instr, &s.live);
      s.cur_all += delta.all;
      s.cur_64 += delta.bank64;
      s.cur_16 += delta.bank16;
      res.max_all = MAX2(res.max_all, s.cur_all);
      res.max_64 = MAX2(res.max_64, s.cur_64);
      res.max_16 = MAX2(res.max_16, s.cur_16);

      if (latency) {
         util_dynarray_foreach(&node->dag.edges, struct dag_edge, edge) {
            struct vir_sched_node *producer =
               (struct vir_sched_node *)edge->child;
            producer->ready = MAX2(producer->ready,
                                   s.position +
                                      apple9_ra_dependency_delay(producer->instr));
         }
         ++s.position;
      }

      dag_prune_head(s.dag, &node->dag);
      sched_nodes[nr] = node;
      res.order[nr] = node->instr;
      nr++;
      liveness_ins_update_internal(p, defs, pub_defs, &s.live, node->instr,
                                   false, NULL);
   }

   assert(nr == count);
   res.count = count;
   if (latency)
      res.stalls = compute_schedule_stalls(sched_nodes, count, true, dep_pool,
                                           dep_start);
   res.valid = true;
   return res;
}

static bool
pressure_strictly_better(int32_t new_all, int32_t new_64, int32_t new_16,
                         int32_t orig_all, int32_t orig_64, int32_t orig_16)
{
   if (new_all != orig_all)
      return new_all < orig_all;
   if (new_16 != orig_16)
      return new_16 < orig_16;
   return new_64 < orig_64;
}

static bool
bank_limits_respected(int32_t new_64, int32_t new_16,
                      int32_t orig_64, int32_t orig_16)
{
   if (orig_16 <= 16 && new_16 > 16)
      return false;
   if (orig_16 > 16 && new_16 > orig_16)
      return false;
   if (orig_64 <= 64 && new_64 > 64)
      return false;
   if (orig_64 > 64 && new_64 > orig_64)
      return false;
   return true;
}

static void
apply_vir_schedule(struct agx_apple9_vir_program *p,
                   struct agx_apple9_block *blk,
                   struct agx_apple9_vir_instr *last_pinned_entry,
                   struct agx_apple9_vir_instr *const *bottom_up_order,
                   unsigned count)
{
   struct agx_apple9_vir_instr *anchor = last_pinned_entry;
   for (unsigned k = count; k-- > 0;) {
      struct agx_apple9_vir_instr *ins = bottom_up_order[k];
      if (!anchor) {
         if (agx_apple9_block_first_instr(blk) != ins)
            agx_apple9_vir_move_to_block_start(p, ins, blk);
      } else {
         if (agx_apple9_instr_next(anchor) != ins)
            agx_apple9_vir_move_after(p, ins, anchor);
      }
      anchor = ins;
   }
}

bool
apple9_ra_pressure_schedule_opts(struct agx_apple9_vir_program *p,
                                 struct apple9_ra_sched_options *opts)
{
   if (!p)
      return false;

   struct apple9_ra_sched_options default_opts = {
      .latency = true,
      .pressure_slack = 0,
      .rejected_pressure_increase = 0,
   };
   struct apple9_ra_sched_options *cfg = opts ? opts : &default_opts;

   struct apple9_ra_liveness_info liveness;
   if (!apple9_ra_liveness(p, &liveness))
      return false;

   if (liveness.num_cfg_blocks == 0) {
      apple9_ra_liveness_finish(&liveness);
      return true;
   }

   void *mem_ctx = ralloc_context(NULL);
   if (!mem_ctx) {
      apple9_ra_liveness_finish(&liveness);
      return false;
   }

   struct agx_apple9_vir_instr **defs = NULL;
   BITSET_WORD *pub_defs = NULL;
   uint8_t *val_max_gpr = compute_value_max_gprs(p, mem_ctx);
   struct vir_sched_node **last_write =
      rzalloc_array(mem_ctx, struct vir_sched_node *, MAX2(p->value_count, 1u));
   if (!val_max_gpr || !last_write ||
       !build_defs_and_pub_mask(p, mem_ctx, &defs, &pub_defs)) {
      ralloc_free(mem_ctx);
      apple9_ra_liveness_finish(&liveness);
      return false;
   }

   bool any_reordered = false;

   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      if (!blk->cfg_active)
         continue;

      /* Partition block into:
       *   [entry: leading VIR_PHI / VIR_ENTRY_REG / opening mask-restore ops]
       *   [body:  schedulable instructions]
       *   [tail:  trailing after_logical_end / VIR_PHI_SRC ops]
       * Scan the tail backwards from block end so opening EXEC_MASK_ELSE /
       * EXEC_MASK_POP / LOOP_MASK_POP or mid-block control barriers never
       * truncate the schedulable body. */
      struct agx_apple9_vir_instr *last_entry = NULL;
      struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
      while (I && is_pinned_entry(I)) {
         last_entry = I;
         I = agx_apple9_instr_next(I);
      }
      struct agx_apple9_vir_instr *first_body = I;
      if (!first_body)
         continue;

      struct agx_apple9_vir_instr *first_tail = NULL;
      for (struct agx_apple9_vir_instr *T = agx_apple9_block_last_instr(blk);
           T; T = agx_apple9_instr_prev(T)) {
         if (is_pinned_tail(T)) {
            first_tail = T;
         } else {
            break;
         }
         if (T == first_body)
            break;
      }

      unsigned body_count = 0;
      for (I = first_body; I && I != first_tail; I = agx_apple9_instr_next(I))
         body_count++;
      if (body_count <= 1)
         continue;

      void *blk_ctx = ralloc_context(mem_ctx);
      if (!blk_ctx) {
         ralloc_free(mem_ctx);
         apple9_ra_liveness_finish(&liveness);
         return false;
      }

      /* Compute exact live set at the end of the schedulable body by walking
       * backwards across the pinned tail from block live_out. */
      struct u_sparse_bitset live_after_body;
      u_sparse_bitset_dup_with_ctx(
         &live_after_body, &liveness.live_out[blk->cfg_index], blk_ctx);
      if (first_tail) {
         for (struct agx_apple9_vir_instr *T = agx_apple9_block_last_instr(blk);
              T; T = agx_apple9_instr_prev(T)) {
            liveness_ins_update_internal(p, defs, pub_defs, &live_after_body, T,
                                         false, NULL);
            if (T == first_tail)
               break;
         }
      }

      int32_t base_all = 0, base_64 = 0, base_16 = 0;
      U_SPARSE_BITSET_FOREACH_SET(&live_after_body, v) {
         base_all++;
         if (val_max_gpr[v] <= 63)
            base_64++;
         if (val_max_gpr[v] <= 15)
            base_16++;
      }

      /* Measure original peak pressure across the schedulable body. */
      struct u_sparse_bitset orig_live;
      u_sparse_bitset_dup_with_ctx(&orig_live, &live_after_body, blk_ctx);
      int32_t cur_all = base_all, cur_64 = base_64, cur_16 = base_16;
      int32_t orig_all = base_all, orig_64 = base_64, orig_16 = base_16;
      struct agx_apple9_vir_instr *last_body =
         first_tail ? agx_apple9_instr_prev(first_tail)
                    : agx_apple9_block_last_instr(blk);
      for (struct agx_apple9_vir_instr *B = last_body; B;
           B = agx_apple9_instr_prev(B)) {
         struct vir_pressure_delta delta = calc_pressure_delta(
            p, defs, pub_defs, val_max_gpr, B, &orig_live);
         cur_all += delta.all;
         cur_64 += delta.bank64;
         cur_16 += delta.bank16;
         orig_all = MAX2(orig_all, cur_all);
         orig_64 = MAX2(orig_64, cur_64);
         orig_16 = MAX2(orig_16, cur_16);
         liveness_ins_update_internal(p, defs, pub_defs, &orig_live, B, false,
                                      NULL);
         if (B == first_body)
            break;
      }

      bool committed = false;

      if (cfg->latency) {
         void *lat_ctx = ralloc_context(blk_ctx);
         unsigned orig_stalls = 0;
         struct vir_sched_result lat = run_block_schedule_attempt(
            p, defs, pub_defs, val_max_gpr, first_body, first_tail,
            &live_after_body, base_all, base_64, base_16, true, last_write,
            lat_ctx, &orig_stalls);

         bool accept_lat = false;
         if (lat.valid) {
            int32_t limit = orig_all + (int32_t)cfg->pressure_slack;
            if (lat.max_all > limit) {
               cfg->rejected_pressure_increase = MAX2(
                  cfg->rejected_pressure_increase,
                  (unsigned)(lat.max_all - orig_all));
            } else if (bank_limits_respected(lat.max_64, lat.max_16, orig_64,
                                             orig_16)) {
               accept_lat =
                  (lat.stalls < orig_stalls) ||
                  (lat.stalls == orig_stalls &&
                   pressure_strictly_better(lat.max_all, lat.max_64, lat.max_16,
                                            orig_all, orig_64, orig_16));
            }
         }

         if (accept_lat) {
            bool overridden_by_pressure = false;
            if (cfg->pressure_slack == 0 && lat.max_all > base_all) {
               void *pres_ctx = ralloc_context(blk_ctx);
               struct vir_sched_result pres = run_block_schedule_attempt(
                  p, defs, pub_defs, val_max_gpr, first_body, first_tail,
                  &live_after_body, base_all, base_64, base_16, false,
                  last_write, pres_ctx, NULL);
               if (pres.valid &&
                   bank_limits_respected(pres.max_64, pres.max_16, orig_64,
                                         orig_16) &&
                   pressure_strictly_better(pres.max_all, pres.max_64,
                                            pres.max_16, lat.max_all,
                                            lat.max_64, lat.max_16)) {
                  apply_vir_schedule(p, blk, last_entry, pres.order, pres.count);
                  committed = true;
                  any_reordered = true;
                  overridden_by_pressure = true;
               }
               ralloc_free(pres_ctx);
            }
            if (!overridden_by_pressure) {
               apply_vir_schedule(p, blk, last_entry, lat.order, lat.count);
               committed = true;
               any_reordered = true;
            }
         }
         ralloc_free(lat_ctx);
      }

      /* Run pure pressure minimization when latency mode is disabled, when
       * zero slack is requested, or when pressure exceeds the register file
       * (avoiding undoing latency hiding in low-pressure blocks where the
       * caller explicitly granted occupancy-tier pressure slack). */
      if (!committed && (!cfg->latency || cfg->pressure_slack == 0 ||
                         orig_all > (int32_t)AGX_APPLE9_GPR_COUNT)) {
         void *pres_ctx = ralloc_context(blk_ctx);
         struct vir_sched_result pres = run_block_schedule_attempt(
            p, defs, pub_defs, val_max_gpr, first_body, first_tail,
            &live_after_body, base_all, base_64, base_16, false, last_write,
            pres_ctx, NULL);

         if (pres.valid &&
             bank_limits_respected(pres.max_64, pres.max_16, orig_64, orig_16) &&
             pressure_strictly_better(pres.max_all, pres.max_64, pres.max_16,
                                      orig_all, orig_64, orig_16)) {
            apply_vir_schedule(p, blk, last_entry, pres.order, pres.count);
            any_reordered = true;
         }
         ralloc_free(pres_ctx);
      }

      ralloc_free(blk_ctx);
   }

   ralloc_free(mem_ctx);
   apple9_ra_liveness_finish(&liveness);

   /* Refresh kill_mask and peak_live_gprs if any block was reordered. */
   if (any_reordered)
      return apple9_ra_liveness(p, NULL);
   return true;
}

bool
apple9_ra_pressure_schedule(struct agx_apple9_vir_program *p)
{
   return apple9_ra_pressure_schedule_opts(p, NULL);
}
