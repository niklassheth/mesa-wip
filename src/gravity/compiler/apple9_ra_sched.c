/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_sched.h"
#include "apple9_ra_spill.h"
#include "agx_compile.h"

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

   agx_apple9_vir_ensure_indexed(p);
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

   if (remove_count > 0) {
      agx_apple9_vir_remove_many(p, to_remove, remove_count);
      agx_apple9_vir_ensure_indexed(p);
   }

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

static bool
build_defs_and_pub_mask(const struct agx_apple9_vir_program *p, void *mem_ctx,
                        struct agx_apple9_vir_instr ***defs_out,
                        BITSET_WORD **pub_defs_out)
{
   assert(!p->index_dirty);
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

   agx_apple9_vir_ensure_indexed(p);
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

         agx_apple9_foreach_phi(blk, phi) {
            if (phi->dest != AGX_APPLE9_VREG_INVALID &&
                is_valid_gpr(p, pub_defs, phi->dest))
               u_sparse_bitset_clear(&edge_live, phi->dest);
         }

         agx_apple9_foreach_phi(blk, phi) {
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
   case AGX_APPLE9_VIR_PREDICATE_COMPARE:
      /* Writes only the hardware predicate register; orthogonal to pure ALU
       * and memory ops while still observing execution/coverage state. */
      *reads |= APPLE9_RA_SCHED_PREDICATE;
      *writes |= APPLE9_RA_SCHED_PREDICATE;
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

/* Single hardware-calibrated Apple9 (G16G / T8132 @ 1470 MHz) latency table
 * shared across VIR scheduling, allocation cost evaluation, and RA policy.
 *
 * Exact measured cycle latencies:
 *   - Main ALU (FMA/FADD/FMUL/HFMA/IADD/IAND/IOR/IXOR/MOV): 2
 *   - SELECT: 3
 *   - IMAD / ISHL / USHR / ISHR / EXTR / SFU / async GET_SR: 5
 *   - HRCP / HRCP_F32: 6
 *   - IMUL_WIDE (32x32->64): 7
 *   - 0xCF MMA (8x8x8 FP32/FP16): 18 (17..18 measured)
 *   - SHARED / PRIVATE / SPILL / TILE load: 24 (~75..88 raw cyc)
 *   - DEVICE_LOAD / DEVICE_ATOMIC: 36 (~72..152 raw cyc)
 *   - TEXTURE_SAMPLE: 48 (~166..253 raw cyc, highest priority) */
unsigned
apple9_ra_dependency_delay(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO && ins->op != AGX_APPLE9_VIR_MOV)
      return 1;

   switch (ins->op) {
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      return 48;
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
      return 36;
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_TILE_LOAD:
      return 24;
   case AGX_APPLE9_VIR_MMA:
      return 18;
   case AGX_APPLE9_VIR_IMUL_WIDE:
      return 7;
   case AGX_APPLE9_VIR_HRCP:
   case AGX_APPLE9_VIR_HRCP_F32:
      return 6;
   case AGX_APPLE9_VIR_IMAD:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_EXTR:
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_HRSQ:
   case AGX_APPLE9_VIR_HSQRT_FACTOR:
   case AGX_APPLE9_VIR_HEXP2:
   case AGX_APPLE9_VIR_HLOG2:
   case AGX_APPLE9_VIR_CUBE:
   case AGX_APPLE9_VIR_SUBGROUP_REDUCE:
   case AGX_APPLE9_VIR_SUBGROUP_REDUCE_UMAX:
   case AGX_APPLE9_VIR_SUBGROUP_SHUFFLE:
   case AGX_APPLE9_VIR_SUBGROUP_ROTATE:
   case AGX_APPLE9_VIR_SUBGROUP_BALLOT:
   case AGX_APPLE9_VIR_DERIVATIVE:
   case AGX_APPLE9_VIR_PREDICATE_COMPARE:
      return 5;
   case AGX_APPLE9_VIR_GET_SR:
      return ins->encoding == AGX_APPLE9_ENC_GET_SR_ZEXT16 ? 2 : 5;
   case AGX_APPLE9_VIR_SELECT:
      return 3;
   default:
      return 2;
   }
}

enum apple9_ra_exec_pipe
apple9_ra_exec_pipe(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO && ins->op != AGX_APPLE9_VIR_MOV)
      return APPLE9_RA_PIPE_NONE;

   switch (ins->op) {
   case AGX_APPLE9_VIR_IMAD:
   case AGX_APPLE9_VIR_IMUL_WIDE:
      return APPLE9_RA_PIPE_IMUL;
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_EXTR:
   case AGX_APPLE9_VIR_BIT_COUNT:
   case AGX_APPLE9_VIR_UFIND_MSB:
   case AGX_APPLE9_VIR_BIT_REVERSE:
      return APPLE9_RA_PIPE_SHIFT;
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
   case AGX_APPLE9_VIR_CUBE:
      return APPLE9_RA_PIPE_SFU;
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_STORE:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_STORE:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_STORE:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_SPILL_STORE:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_IMAGE_STORE:
   case AGX_APPLE9_VIR_BLOCK_IMAGE_STORE:
   case AGX_APPLE9_VIR_TILE_LOAD:
   case AGX_APPLE9_VIR_TILE_STORE:
   case AGX_APPLE9_VIR_ITER:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_VARY_STORE:
      return APPLE9_RA_PIPE_MEM;
   case AGX_APPLE9_VIR_WAIT:
   case AGX_APPLE9_VIR_WORKGROUP_BARRIER:
   case AGX_APPLE9_VIR_DEVICE_FENCE:
   case AGX_APPLE9_VIR_TILE_FENCE:
   case AGX_APPLE9_VIR_TILE_ACCESS:
   case AGX_APPLE9_VIR_HALT:
      return APPLE9_RA_PIPE_NONE;
   default:
      /* FP32, FP16, IADD/IAND/IOR/IXOR/SELECT/MOV, and 0xCF MMA all share
       * the 128-wide main ALU datapath on G16G. */
      return APPLE9_RA_PIPE_MAIN;
   }
}

bool
apple9_ra_async(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT)
      return true;
   if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
       ins->encoding < AGX_APPLE9_ENC_COUNT &&
       agx_apple9_encoding_info(ins->encoding)->fixed_producer_slot != 0)
      return true;
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_TILE_LOAD:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_SPILL_STORE:
   case AGX_APPLE9_VIR_IMAGE_STORE:
      return true;
   case AGX_APPLE9_VIR_GET_SR:
      return ins->encoding == AGX_APPLE9_ENC_GET_SR_EXTENDED ||
             ins->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
      return !ins->atomic_discard;
   default:
      return false;
   }
}

/* Memory operations read their address, index and data registers at issue.
 * Native Metal overwrites a pending device load's address or index 3,744
 * times in 4,000 public pipelines without waiting. Overwriting atomic,
 * private-load and shared-load inputs right after issue passed the CTS
 * atomic, indexing and shared-memory groups. */
bool
apple9_ra_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_MESH_CONTROL:
      return true;
   default:
      return false;
   }
}

unsigned
apple9_ra_source_width(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if ((ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
        ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) &&
       source == 0)
      return 2;
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
        ins->op == AGX_APPLE9_VIR_PRIVATE_STORE) &&
       source == 0)
      return agx_apple9_mem_data_gprs(ins);
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
        ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) &&
       source == 0)
      return MAX2((unsigned)ins->memory_components, 1u);
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT &&
       source == agx_apple9_mem_data_gprs(ins) + 1)
      return 2;
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT &&
       source == MAX2((unsigned)ins->memory_components, 1u) + 1)
      return 2;
   if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT && source == 1) ||
       (ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER && source == 0))
      return 2;
   if (ins->op == AGX_APPLE9_VIR_MMA)
      return agx_apple9_mma_source_width(ins, source);
   if (ins->op == AGX_APPLE9_VIR_IMAGE_STORE && source == 8)
      return 2;
   return 1;
}

static const struct agx_apple9_operand_constraint apple9_ra_table_index[2] = {
   {
      .role = AGX_APPLE9_OPERAND_SRC4,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
   {
      .role = AGX_APPLE9_OPERAND_SRC5,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
};

const struct agx_apple9_operand_constraint *
apple9_ra_source_constraint(const struct agx_apple9_vir_instr *ins,
                            unsigned source)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
       ins->encoding >= AGX_APPLE9_ENC_COUNT)
      return NULL;

   if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
       source + ins->texture_bindless + ins->sampler_bindless >= ins->nr_srcs)
      return &apple9_ra_table_index[ins->sampler_bindless &&
                                    source + 1 == ins->nr_srcs];

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
          source > data_gprs)
         return agx_apple9_find_operand(ins->encoding,
                                        source <= data_gprs + 2
                                           ? AGX_APPLE9_OPERAND_SRC1
                                           : AGX_APPLE9_OPERAND_SRC2);
      enum agx_apple9_operand_role role =
         source >= data_gprs                        ? AGX_APPLE9_OPERAND_INDEX
         : ins->op == AGX_APPLE9_VIR_DEVICE_STORE   ? AGX_APPLE9_OPERAND_STORE_DATA
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

struct apple9_ra_operand_group
apple9_ra_source_group(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   unsigned width = apple9_ra_source_width(ins, source);
   struct apple9_ra_operand_group g = {
      .width = (uint8_t)width,
      .min_gpr = 0,
      .max_gpr = (uint8_t)(AGX_APPLE9_GPR_COUNT - MIN2(width, AGX_APPLE9_GPR_COUNT)),
      .align_gprs = 1,
      .clobber = false,
      .reuse_killed = false,
      .constraint = NULL,
   };

   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
       ins->encoding >= AGX_APPLE9_ENC_COUNT)
      return g;

   const struct agx_apple9_operand_constraint *sc =
      apple9_ra_source_constraint(ins, source);
   g.constraint = sc;

   unsigned min_gpr = sc ? sc->min_index : 0u;
   unsigned max_gpr = sc ? sc->max_index : (AGX_APPLE9_GPR_COUNT - 1u);
   unsigned align_gprs = sc ? MAX2(sc->alignment_halves, 2u) / 2u : 1u;

   if (source == 0) {
      if ((ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
           ins->op == AGX_APPLE9_VIR_DEVICE_STORE) &&
          width > 1)
         max_gpr = AGX_APPLE9_GPR_COUNT - width;
      else if (ins->op == AGX_APPLE9_VIR_PRIVATE_STORE && width > 1)
         max_gpr = 64u - width;
      else if (ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC && width > 1)
         max_gpr = AGX_APPLE9_GPR_COUNT - width;
   }
   if (max_gpr + width > AGX_APPLE9_GPR_COUNT)
      max_gpr = AGX_APPLE9_GPR_COUNT - MIN2(width, AGX_APPLE9_GPR_COUNT);

   g.min_gpr = (uint8_t)min_gpr;
   g.max_gpr = (uint8_t)max_gpr;
   g.align_gprs = (uint8_t)MAX2(align_gprs, 1u);
   g.clobber = sc && (sc->flags & AGX_APPLE9_OPERAND_CLOBBER);
   return g;
}

struct apple9_ra_operand_group
apple9_ra_dest_group(const struct agx_apple9_vir_instr *ins)
{
   unsigned comps = ins->dest_components ? ins->dest_components : 1u;
   struct apple9_ra_operand_group g = {
      .width = (uint8_t)comps,
      .min_gpr = 0,
      .max_gpr = (uint8_t)(AGX_APPLE9_GPR_COUNT - MIN2(comps, AGX_APPLE9_GPR_COUNT)),
      .align_gprs = 1,
      .clobber = false,
      .reuse_killed = false,
      .constraint = NULL,
   };

   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
       ins->encoding >= AGX_APPLE9_ENC_COUNT) {
      g.reuse_killed = (comps == 1);
      return g;
   }

   const struct agx_apple9_operand_constraint *dc =
      ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC
         ? agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT,
                                   AGX_APPLE9_OPERAND_SRC0)
         : agx_apple9_find_operand(ins->encoding, AGX_APPLE9_OPERAND_DEST);
   g.constraint = dc;

   unsigned min_gpr = dc ? dc->min_index : 0u;
   unsigned max_gpr = dc ? dc->max_index : (AGX_APPLE9_GPR_COUNT - 1u);
   unsigned align_gprs = dc ? MAX2(dc->alignment_halves, 2u) / 2u : 1u;

   if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
      unsigned limit = (ins->texture_bindless || ins->sampler_bindless ||
                        ins->texture_uniform_bindless ||
                        ins->sampler_uniform_bindless)
                          ? AGX_APPLE9_GPR_COUNT
                          : 64u;
      max_gpr = limit - comps;
   }
   if ((ins->op == AGX_APPLE9_VIR_SHARED_LOAD ||
        ins->op == AGX_APPLE9_VIR_DEVICE_LOAD) &&
       comps > 1)
      max_gpr = AGX_APPLE9_GPR_COUNT - comps;
   if (ins->op == AGX_APPLE9_VIR_PRIVATE_LOAD && comps > 1)
      max_gpr = 64u - comps;
   if (max_gpr + comps > AGX_APPLE9_GPR_COUNT)
      max_gpr = AGX_APPLE9_GPR_COUNT - MIN2(comps, AGX_APPLE9_GPR_COUNT);

   g.min_gpr = (uint8_t)min_gpr;
   g.max_gpr = (uint8_t)max_gpr;
   g.align_gprs = (uint8_t)MAX2(align_gprs, 1u);
   g.reuse_killed =
      comps == 1 &&
      (!apple9_ra_async(ins) || (apple9_ra_reads_sources_at_issue(ins) &&
                                 ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC)) &&
      !apple9_ra_after_logical_end(ins) &&
      ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
      !agx_apple9_encoding_info(ins->encoding)->split_result;
   return g;
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
         struct apple9_ra_operand_group dg = apple9_ra_dest_group(ins);
         if (dg.constraint && (dg.constraint->files & AGX_APPLE9_FILE_GPR) &&
             dg.max_gpr < AGX_APPLE9_GPR_COUNT) {
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               uint8_t lane_max = MIN2((unsigned)dg.max_gpr + c,
                                       AGX_APPLE9_GPR_COUNT - 1u);
               if (ins->dest + c < vcount)
                  max_gpr[ins->dest + c] =
                     MIN2(max_gpr[ins->dest + c], lane_max);
            }
         }
      }

      const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
      for (unsigned s = 0; s < nr_srcs; ) {
         struct apple9_ra_operand_group sg = apple9_ra_source_group(ins, s);
         unsigned w = MAX2((unsigned)sg.width, 1u);
         if (sg.constraint && (sg.constraint->files & AGX_APPLE9_FILE_GPR) &&
             sg.max_gpr < AGX_APPLE9_GPR_COUNT) {
            for (unsigned c = 0; c < w && s + c < nr_srcs; ++c) {
               uint32_t v = agx_apple9_vir_src(ins, s + c);
               uint8_t lane_max = MIN2((unsigned)sg.max_gpr + c,
                                       AGX_APPLE9_GPR_COUNT - 1u);
               if (v < vcount)
                  max_gpr[v] = MIN2(max_gpr[v], lane_max);
            }
         }
         s += w;
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
   int32_t peak_all;
   int32_t bank64;
   int32_t bank16;
};

/* Optimal in-flight DEVICE_LOAD wave width measured on G16G (L1/L2/DRAM sweet
 * spot without triggering L2 MSHR thrashing at N >= 8..12). */
#define APPLE9_DEVICE_LOAD_MLP_CAP 4u

struct vir_sched_state {
   const struct agx_apple9_vir_program *p;
   struct agx_apple9_vir_instr *const *defs;
   const BITSET_WORD *pub_defs;
   const uint8_t *val_max_gpr;
   struct dag *dag;
   struct u_sparse_bitset live;
   bool latency;
   unsigned position;
   int32_t max_all_budget;
   int32_t cur_all;
   int32_t cur_64;
   int32_t cur_16;
   enum apple9_ra_exec_pipe last_pipe;
   unsigned inflight_dev_loads;
};

/* Edge data distinguishes true SSA data dependencies (1) from pure resource or
 * control-order edges (0) so producer latency is only scored on data edges. */
static inline void
add_data_dep(struct vir_sched_node *a, struct vir_sched_node *b)
{
   assert(a != b && "no self-dependencies");
   if (a && b)
      dag_add_edge_max_data(&a->dag, &b->dag, 1);
}

static inline void
add_order_dep(struct vir_sched_node *a, struct vir_sched_node *b)
{
   assert(a != b && "no self-dependencies");
   if (a && b)
      dag_add_edge_max_data(&a->dag, &b->dag, 0);
}

static inline void
serialize_node(struct vir_sched_node *a, struct vir_sched_node **b)
{
   add_order_dep(a, *b);
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
   struct vir_pressure_delta d = {0, 0, 0, 0};

   if (ins->op == AGX_APPLE9_VIR_PHI ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
      return d;

   int32_t dest_dead = 0;
   if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
      for (unsigned c = 0; c < ins->dest_components; ++c) {
         uint32_t v = ins->dest + c;
         if (!is_valid_gpr(p, pub_defs, v))
            continue;
         if (u_sparse_bitset_test(live, v)) {
            d.all -= 1;
            if (val_max_gpr[v] <= 63)
               d.bank64 -= 1;
            if (val_max_gpr[v] <= 15)
               d.bank16 -= 1;
         } else {
            dest_dead += 1;
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
      d.peak_all = MAX2(0, d.all);
      return d;
   }

   int32_t src_new = 0;
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
         src_new += 1;
         if (val_max_gpr[v] <= 63)
            d.bank64 += 1;
         if (val_max_gpr[v] <= 15)
            d.bank16 += 1;
      }
   }

   /* Simultaneous interference at this instruction: live_before + dest_total
    * == live_after + dest_dead + src_new. Matches a9_calc_gpr_demand(). */
   d.peak_all = dest_dead + src_new;
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

   bool ok = true;
   for (struct agx_apple9_vir_instr *I = first_body; I && I != first_tail;
        I = agx_apple9_instr_next(I)) {
      struct vir_sched_node *node = rzalloc(memctx, struct vir_sched_node);
      if (!node) {
         ok = false;
         break;
      }
      node->instr = I;
      dag_init_node(dag, &node->dag);
      util_dynarray_append(orig_order, node);

      /* SSA read-after-write dependencies (including tuple components). */
      if (I->op == AGX_APPLE9_VIR_SPLIT && agx_apple9_vir_nr_srcs(I) >= 1) {
         uint32_t src0 = agx_apple9_vir_src(I, 0);
         unsigned w = split_src_tuple_width(p, defs, I);
         for (unsigned c = 0; c < w; ++c) {
            if (src0 + c < p->value_count)
               add_data_dep(node, last_write[src0 + c]);
         }
      } else {
         const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < p->value_count)
               add_data_dep(node, last_write[v]);
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
         add_order_dep(node, preload);

      /* Implicit hardware resource hazards (memory, barriers, publications,
       * interpolation, tile, mesh, execution state). */
      uint32_t reads = 0, writes = 0;
      apple9_ra_schedule_resources(I, &reads, &writes);
      u_foreach_bit(r, reads | writes) {
         add_order_dep(node, resource_writer[r]);
         if (writes & BITFIELD_BIT(r)) {
            for (struct vir_resource_reader *rr = resource_readers[r]; rr;
                 rr = rr->next) {
               add_order_dep(node, rr->node);
            }
            resource_readers[r] = NULL;
            resource_writer[r] = node;
         } else {
            struct vir_resource_reader *rr =
               rzalloc(memctx, struct vir_resource_reader);
            if (!rr) {
               ok = false;
               break;
            }
            rr->node = node;
            rr->next = resource_readers[r];
            resource_readers[r] = rr;
         }
      }
      if (!ok)
         break;
   }

   /* Always clear last_write entries touched by this block (even on OOM). */
   for (struct agx_apple9_vir_instr *I = first_body; I && I != first_tail;
        I = agx_apple9_instr_next(I)) {
      if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->dest + c < p->value_count)
               last_write[I->dest + c] = NULL;
         }
      }
   }

   return ok ? dag : NULL;
}

static inline bool
dev_load_is_live(struct vir_sched_state *s,
                 const struct agx_apple9_vir_instr *ld)
{
   if (ld->op != AGX_APPLE9_VIR_DEVICE_LOAD ||
       ld->dest == AGX_APPLE9_VREG_INVALID)
      return false;
   for (unsigned c = 0; c < ld->dest_components; ++c) {
      if (u_sparse_bitset_test(&s->live, ld->dest + c))
         return true;
   }
   return false;
}

static unsigned
node_opened_dev_loads(struct vir_sched_state *s,
                      const struct vir_sched_node *n)
{
   unsigned opens = 0;
   util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
      if (!edge->data)
         continue;
      const struct vir_sched_node *prod =
         (const struct vir_sched_node *)edge->child;
      if (prod->instr->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
          !dev_load_is_live(s, prod->instr))
         opens++;
   }
   return opens;
}

static struct vir_sched_node *
choose_vir_instr(struct vir_sched_state *s)
{
   struct vir_sched_node *best = NULL;
   bool best_over_budget = true;
   bool best_over_dev_mlp = true;
   unsigned best_dependency = 0;
   int32_t best_score = INT32_MAX;
   bool best_dual_issue = false;
   struct vir_pressure_delta best_delta = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};

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

      bool over_budget = s->latency && s->max_all_budget < INT32_MAX &&
                         ((delta.all > 0 &&
                           s->cur_all + delta.all > s->max_all_budget) ||
                          (delta.peak_all > 0 &&
                           s->cur_all + delta.peak_all > s->max_all_budget));

      unsigned opened_loads = s->latency ? node_opened_dev_loads(s, n) : 0u;
      unsigned closed_loads =
         (s->latency && dev_load_is_live(s, n->instr)) ? 1u : 0u;
      unsigned projected_loads =
         s->inflight_dev_loads + opened_loads - closed_loads;
      bool over_dev_mlp = s->latency && opened_loads > 0 &&
                          projected_loads > APPLE9_DEVICE_LOAD_MLP_CAP;

      unsigned dependency = 0;
      if (s->latency && !over_budget && !over_dev_mlp) {
         util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
            if (!edge->data)
               continue;
            struct vir_sched_node *producer =
               (struct vir_sched_node *)edge->child;
            unsigned delay = apple9_ra_dependency_delay(producer->instr);
            /* Only multi-cycle latencies above the 2-cycle main-ALU baseline
             * should override register-pressure minimization. */
            if (delay > 2u)
               dependency = MAX2(dependency, delay - 1u);
         }
      }

      enum apple9_ra_exec_pipe pipe = apple9_ra_exec_pipe(n->instr);
      bool dual_issue = s->latency && !over_budget &&
                        s->last_pipe != APPLE9_RA_PIPE_NONE &&
                        pipe != APPLE9_RA_PIPE_NONE && pipe != s->last_pipe;

      bool better = false;
      if (!best) {
         better = true;
      } else if (over_budget != best_over_budget) {
         better = !over_budget;
      } else if (over_dev_mlp != best_over_dev_mlp) {
         better = !over_dev_mlp;
      } else if (!over_budget && !over_dev_mlp && dependency != best_dependency) {
         better = dependency > best_dependency;
      } else if (score != best_score) {
         better = score < best_score;
      } else if (delta.all != best_delta.all) {
         better = delta.all < best_delta.all;
      } else if (dual_issue != best_dual_issue) {
         better = dual_issue;
      } else if (delta.all > 0) {
         /* Depth-first expansion on positive-pressure ties avoids opening an
          * entire reduction tree breadth-first. */
         better = true;
      }

      if (better) {
         best = n;
         best_over_budget = over_budget;
         best_over_dev_mlp = over_dev_mlp;
         best_dependency = dependency;
         best_score = score;
         best_dual_issue = dual_issue;
         best_delta = delta;
      }
   }

   /* If every ready node would push pressure past the occupancy-tier budget,
    * scan non-ready heads for a node that either fits within budget or
    * strictly closes live registers (e.g. an in-flight vector load) rather
    * than crossing the tier cliff. Pick the earliest-ready such node to
    * minimize extra stall cycles. */
   if (s->latency && s->max_all_budget < INT32_MAX && best && best_over_budget) {
      struct vir_sched_node *fallback = NULL;
      bool fb_fits = false;
      unsigned fb_ready = UINT_MAX;
      int32_t fb_score = INT32_MAX;
      int32_t fb_delta_all = INT32_MAX;

      list_for_each_entry(struct vir_sched_node, n, &s->dag->heads, dag.link) {
         if (n->ready <= s->position)
            continue;
         struct vir_pressure_delta delta = calc_pressure_delta(
            s->p, s->defs, s->pub_defs, s->val_max_gpr, n->instr, &s->live);
         bool fits = (delta.all <= 0 ||
                      s->cur_all + delta.all <= s->max_all_budget) &&
                     (delta.peak_all <= 0 ||
                      s->cur_all + delta.peak_all <= s->max_all_budget);
         if (!fits && !(delta.all < 0 && delta.all < best_delta.all))
            continue;

         int32_t score = delta.all;
         if (s->cur_16 >= 12 || s->cur_16 + delta.bank16 > 16)
            score += 4 * delta.bank16;
         if (s->cur_64 >= 56 || s->cur_64 + delta.bank64 > 64)
            score += 2 * delta.bank64;

         if (!fallback ||
             (fits != fb_fits && fits) ||
             (fits == fb_fits &&
              (n->ready < fb_ready ||
               (n->ready == fb_ready && score < fb_score) ||
               (n->ready == fb_ready && score == fb_score &&
                delta.all < fb_delta_all)))) {
            fallback = n;
            fb_fits = fits;
            fb_ready = n->ready;
            fb_score = score;
            fb_delta_all = delta.all;
         }
      }
      if (fallback)
         return fallback;
   }

   /* If every ready node would exceed the N=4 in-flight DEVICE_LOAD wave cap,
    * retire the earliest-ready open DEVICE_LOAD head to form clean N=4 waves
    * on distinct scoreboard slots instead of bursting to 8..12 in-flight loads. */
   if (s->latency && best && !best_over_budget && best_over_dev_mlp) {
      struct vir_sched_node *retire = NULL;
      unsigned ret_ready = UINT_MAX;
      list_for_each_entry(struct vir_sched_node, n, &s->dag->heads, dag.link) {
         if (n->ready <= s->position)
            continue;
         if (n->instr->op != AGX_APPLE9_VIR_DEVICE_LOAD ||
             !dev_load_is_live(s, n->instr))
            continue;
         if (s->max_all_budget < INT32_MAX) {
            struct vir_pressure_delta delta = calc_pressure_delta(
               s->p, s->defs, s->pub_defs, s->val_max_gpr, n->instr, &s->live);
            if ((delta.all > 0 && s->cur_all + delta.all > s->max_all_budget) ||
                (delta.peak_all > 0 &&
                 s->cur_all + delta.peak_all > s->max_all_budget))
               continue;
         }
         if (!retire || n->ready < ret_ready) {
            retire = n;
            ret_ready = n->ready;
         }
      }
      if (retire)
         return retire;
   }

   return best;
}

static unsigned
compute_dual_issue_transitions(struct vir_sched_node *const *nodes,
                               unsigned count, bool reverse)
{
   enum apple9_ra_exec_pipe prev = APPLE9_RA_PIPE_NONE;
   unsigned transitions = 0;
   for (unsigned k = 0; k < count; ++k) {
      struct vir_sched_node *node = nodes[reverse ? (count - 1 - k) : k];
      enum apple9_ra_exec_pipe pipe = apple9_ra_exec_pipe(node->instr);
      if (pipe == APPLE9_RA_PIPE_NONE)
         continue;
      if (prev != APPLE9_RA_PIPE_NONE && pipe != prev)
         transitions++;
      prev = pipe;
   }
   return transitions;
}

static inline bool
tail_consumes_predicate(const struct agx_apple9_vir_instr *first_tail)
{
   return first_tail &&
          (first_tail->op == AGX_APPLE9_VIR_LOOP_MASK_UPDATE ||
           first_tail->op == AGX_APPLE9_VIR_EXEC_MASK_PUSH ||
           first_tail->op == AGX_APPLE9_VIR_BREAK_MASK_UNWIND);
}

static unsigned
compute_schedule_stalls(struct vir_sched_node *const *nodes, unsigned count,
                        bool reverse, struct vir_sched_node *const *dep_pool,
                        const unsigned *dep_start, bool tail_uses_pred)
{
   for (unsigned i = 0; i < count; ++i)
      nodes[i]->ready = 0;

   unsigned issue = 0, stalls = 0, last_pred_ready = 0;
   for (unsigned k = 0; k < count; ++k) {
      struct vir_sched_node *node = nodes[reverse ? (count - 1 - k) : k];
      unsigned earliest = issue;
      for (unsigned e = dep_start[node->orig_idx];
           e < dep_start[node->orig_idx + 1]; ++e) {
         earliest = MAX2(earliest, dep_pool[e]->ready);
      }
      stalls += earliest - issue;
      node->ready = earliest + apple9_ra_dependency_delay(node->instr);
      if (node->instr->op == AGX_APPLE9_VIR_PREDICATE_COMPARE)
         last_pred_ready = node->ready;
      issue = earliest + 1;
   }
   if (tail_uses_pred && last_pred_ready > issue)
      stalls += last_pred_ready - issue;

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
   unsigned dual_issue;
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
                           bool latency, int32_t max_all_budget,
                           struct vir_sched_node **last_write,
                           void *attempt_ctx, unsigned *orig_stalls_out,
                           unsigned *orig_dual_issue_out)
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

   const bool tail_uses_pred = tail_consumes_predicate(first_tail);
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
            if (edge->child && edge->data)
               dep_pool[pos++] = (struct vir_sched_node *)edge->child;
         }
      }
      dep_start[count] = pos;
      if (orig_stalls_out) {
         *orig_stalls_out =
            compute_schedule_stalls(orig, count, false, dep_pool, dep_start,
                                    tail_uses_pred);
      }
      if (orig_dual_issue_out) {
         *orig_dual_issue_out =
            compute_dual_issue_transitions(orig, count, false);
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
      .max_all_budget = max_all_budget,
      .cur_all = base_all,
      .cur_64 = base_64,
      .cur_16 = base_16,
      .last_pipe = APPLE9_RA_PIPE_NONE,
      .inflight_dev_loads = 0,
   };
   u_sparse_bitset_dup_with_ctx(&s.live, live_after_body, attempt_ctx);
   if (latency) {
      for (unsigned i = 0; i < count; ++i) {
         if (dev_load_is_live(&s, orig[i]->instr))
            s.inflight_dev_loads++;
      }
   }

   res.max_all = base_all;
   res.max_64 = base_64;
   res.max_16 = base_16;
   unsigned nr = 0;

   if (latency && tail_uses_pred) {
      list_for_each_entry(struct vir_sched_node, n, &s.dag->heads, dag.link) {
         if (n->instr->op == AGX_APPLE9_VIR_PREDICATE_COMPARE)
            n->ready = MAX2(n->ready, apple9_ra_dependency_delay(n->instr));
      }
   }

   while (!list_is_empty(&s.dag->heads)) {
      struct vir_sched_node *node = choose_vir_instr(&s);
      assert(node != NULL);

      struct vir_pressure_delta delta = calc_pressure_delta(
         p, defs, pub_defs, val_max_gpr, node->instr, &s.live);
      res.max_all = MAX2(res.max_all, s.cur_all + delta.peak_all);
      s.cur_all += delta.all;
      s.cur_64 += delta.bank64;
      s.cur_16 += delta.bank16;
      res.max_all = MAX2(res.max_all, s.cur_all);
      res.max_64 = MAX2(res.max_64, s.cur_64);
      res.max_16 = MAX2(res.max_16, s.cur_16);

      bool closed_dev_load = false;
      struct agx_apple9_vir_instr *opened_loads[AGX_APPLE9_MAX_VIR_SRCS];
      unsigned nr_opened_candidates = 0;

      if (latency) {
         closed_dev_load = dev_load_is_live(&s, node->instr);
         s.position = MAX2(s.position, node->ready);
         util_dynarray_foreach(&node->dag.edges, struct dag_edge, edge) {
            struct vir_sched_node *producer =
               (struct vir_sched_node *)edge->child;
            unsigned raw_delay =
               edge->data ? apple9_ra_dependency_delay(producer->instr) : 1u;
            /* Keep 2-cycle main-ALU chains immediately schedulable (hidden by
             * warp interleaving on G16G) so breadth-first bouncing doesn't
             * fragment live ranges or VIR_COLLECT tuples; enforce full latency
             * separation for all >2c ops (SELECT/IMAD/SHIFT/SFU/MMA/LOAD/TEX). */
            unsigned delay = raw_delay <= 2u ? 1u : raw_delay;
            producer->ready = MAX2(producer->ready, s.position + delay);
            if (edge->data &&
                producer->instr->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
                !dev_load_is_live(&s, producer->instr) &&
                nr_opened_candidates < ARRAY_SIZE(opened_loads)) {
               opened_loads[nr_opened_candidates++] = producer->instr;
            }
         }
         enum apple9_ra_exec_pipe pipe = apple9_ra_exec_pipe(node->instr);
         if (pipe != APPLE9_RA_PIPE_NONE)
            s.last_pipe = pipe;
         ++s.position;
      }

      dag_prune_head(s.dag, &node->dag);
      sched_nodes[nr] = node;
      res.order[nr] = node->instr;
      nr++;
      liveness_ins_update_internal(p, defs, pub_defs, &s.live, node->instr,
                                   false, NULL);

      if (latency) {
         if (closed_dev_load && s.inflight_dev_loads > 0)
            s.inflight_dev_loads--;
         for (unsigned k = 0; k < nr_opened_candidates; ++k) {
            if (dev_load_is_live(&s, opened_loads[k]))
               s.inflight_dev_loads++;
         }
      }
   }

   assert(nr == count);
   res.count = count;
   if (latency) {
      res.stalls = compute_schedule_stalls(sched_nodes, count, true, dep_pool,
                                           dep_start, tail_uses_pred);
      res.dual_issue = compute_dual_issue_transitions(sched_nodes, count, true);
   }
   res.valid = true;
   return res;
}

static bool
is_sink_load_instr(const struct agx_apple9_vir_instr *ins)
{
   if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components == 0 ||
       ins->memory_volatile)
      return false;
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_TILE_LOAD:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      return true;
   default:
      return false;
   }
}

static bool
is_sink_helper_alu(const struct vir_sched_node *node,
                   const BITSET_WORD *pub_defs)
{
   const struct agx_apple9_vir_instr *ins = node->instr;
   if (ins->op == AGX_APPLE9_VIR_SPLIT)
      return true;
   if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components != 1)
      return false;
   if (BITSET_TEST(pub_defs, ins->dest))
      return false;
   if (!agx_apple9_instr_is_pure_alu(ins))
      return false;
   if (agx_apple9_vir_nr_srcs(ins) <= 1)
      return true;
   /* Sink bounds-check SELECTs (and 2-src pure ALU) that consume a load so
    * the load itself can sink closer to its first real consumer. */
   if (ins->op == AGX_APPLE9_VIR_SELECT || agx_apple9_vir_nr_srcs(ins) == 2) {
      util_dynarray_foreach(&node->dag.edges, struct dag_edge, edge) {
         if (!edge->data)
            continue;
         const struct vir_sched_node *child =
            (const struct vir_sched_node *)edge->child;
         if (is_sink_load_instr(child->instr) ||
             child->instr->op == AGX_APPLE9_VIR_SPLIT)
            return true;
      }
   }
   return false;
}

/* Order-preserving intra-block load sinking: on high-pressure blocks where
 * greedy bottom-up pressure scheduling scrambles independent chains and fails
 * to beat original order, sink loads (and their bounds-check selects / cheap
 * 0-1 source helpers) down to right before their earliest dependent in the
 * exact DAG, preserving every data and resource hazard. */
static struct vir_sched_result
run_block_sink_loads(const struct agx_apple9_vir_program *p,
                     struct agx_apple9_vir_instr *const *defs,
                     const BITSET_WORD *pub_defs, const uint8_t *val_max_gpr,
                     struct agx_apple9_vir_instr *first_body,
                     struct agx_apple9_vir_instr *first_tail,
                     struct u_sparse_bitset *live_after_body, int32_t base_all,
                     int32_t base_64, int32_t base_16,
                     struct vir_sched_node **last_write, void *attempt_ctx)
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

   unsigned *parent_count = rzalloc_array(attempt_ctx, unsigned, count);
   unsigned *parent_start = rzalloc_array(attempt_ctx, unsigned, count + 1);
   struct vir_sched_node **work =
      ralloc_array(attempt_ctx, struct vir_sched_node *, count);
   unsigned *pos = ralloc_array(attempt_ctx, unsigned, count);
   res.order = ralloc_array(attempt_ctx, struct agx_apple9_vir_instr *, count);
   if (!parent_count || !parent_start || !work || !pos || !res.order)
      return res;

   unsigned total_edges = 0;
   for (unsigned i = 0; i < count; ++i) {
      orig[i]->orig_idx = i;
      work[i] = orig[i];
      pos[i] = i;
      util_dynarray_foreach(&orig[i]->dag.edges, struct dag_edge, edge) {
         if (edge->child) {
            struct vir_sched_node *c = (struct vir_sched_node *)edge->child;
            parent_count[c->orig_idx]++;
            total_edges++;
         }
      }
   }

   unsigned *parent_pool =
      ralloc_array(attempt_ctx, unsigned, MAX2(total_edges, 1u));
   unsigned *parent_cursor = rzalloc_array(attempt_ctx, unsigned, count);
   if (!parent_pool || !parent_cursor)
      return res;

   unsigned off = 0;
   for (unsigned i = 0; i < count; ++i) {
      parent_start[i] = off;
      parent_cursor[i] = off;
      off += parent_count[i];
   }
   parent_start[count] = off;

   for (unsigned i = 0; i < count; ++i) {
      util_dynarray_foreach(&orig[i]->dag.edges, struct dag_edge, edge) {
         if (edge->child) {
            struct vir_sched_node *c = (struct vir_sched_node *)edge->child;
            parent_pool[parent_cursor[c->orig_idx]++] = i;
         }
      }
   }

   bool moved = false;
   for (unsigned k = count; k-- > 0;) {
      struct vir_sched_node *u = orig[k];
      const bool load = is_sink_load_instr(u->instr);
      if (!load && !is_sink_helper_alu(u, pub_defs))
         continue;

      unsigned min_p = count;
      for (unsigned e = parent_start[k]; e < parent_start[k + 1]; ++e) {
         unsigned p_idx = parent_pool[e];
         if (pos[p_idx] < min_p)
            min_p = pos[p_idx];
      }
      unsigned cur = pos[k];
      if (min_p <= cur + 1)
         continue;

      unsigned target = min_p - 1;
      for (unsigned j = cur; j < target; ++j) {
         work[j] = work[j + 1];
         pos[work[j]->orig_idx] = j;
      }
      work[target] = u;
      pos[k] = target;
      moved = true;
   }

   if (!moved)
      return res;

   /* Measure exact peak pressure of the sunken schedule. */
   struct u_sparse_bitset live;
   u_sparse_bitset_dup_with_ctx(&live, live_after_body, attempt_ctx);
   int32_t cur_all = base_all, cur_64 = base_64, cur_16 = base_16;
   res.max_all = base_all;
   res.max_64 = base_64;
   res.max_16 = base_16;

   for (unsigned i = 0; i < count; ++i) {
      struct agx_apple9_vir_instr *ins = work[count - 1 - i]->instr;
      struct vir_pressure_delta delta = calc_pressure_delta(
         p, defs, pub_defs, val_max_gpr, ins, &live);
      res.max_all = MAX2(res.max_all, cur_all + delta.peak_all);
      cur_all += delta.all;
      cur_64 += delta.bank64;
      cur_16 += delta.bank16;
      res.max_all = MAX2(res.max_all, cur_all);
      res.max_64 = MAX2(res.max_64, cur_64);
      res.max_16 = MAX2(res.max_16, cur_16);
      liveness_ins_update_internal(p, defs, pub_defs, &live, ins, false, NULL);
      res.order[i] = ins;
   }

   res.count = count;
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
refresh_block_liveness_and_kills(const struct agx_apple9_vir_program *p,
                                 struct agx_apple9_vir_instr *const *defs,
                                 const BITSET_WORD *pub_defs,
                                 struct apple9_ra_liveness_info *liveness,
                                 struct agx_apple9_block *blk, void *scratch_ctx)
{
   unsigned b = blk->cfg_index;
   struct u_sparse_bitset live;
   u_sparse_bitset_dup_with_ctx(&live, &liveness->live_out[b], scratch_ctx);
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

   liveness->block_peak_gprs[b] = block_peak;
   u_sparse_bitset_free(&live);
}

static bool
apply_vir_schedule(struct agx_apple9_vir_program *p,
                   struct agx_apple9_block *blk,
                   struct agx_apple9_vir_instr *last_pinned_entry,
                   struct agx_apple9_vir_instr *const *bottom_up_order,
                   unsigned count, struct agx_apple9_vir_instr *const *defs,
                   const BITSET_WORD *pub_defs,
                   struct apple9_ra_liveness_info *liveness, void *blk_ctx)
{
   if (count <= 1)
      return false;

   struct agx_apple9_vir_instr *first_body =
      last_pinned_entry ? agx_apple9_instr_next(last_pinned_entry)
                        : agx_apple9_block_first_instr(blk);
   bool changed = false;
   struct agx_apple9_vir_instr *slot = first_body;
   for (unsigned i = 0; i < count; ++i, slot = agx_apple9_instr_next(slot)) {
      if (slot != bottom_up_order[count - 1 - i]) {
         changed = true;
         break;
      }
   }
   if (!changed)
      return false;

   /* Reorder the body's list_head nodes in blk->instructions in O(count);
    * move helpers defer reindexing until after all blocks are scheduled. */
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

   /* Intra-block reordering preserves live_in/live_out; refresh this block's
    * kill_mask and peak pressure directly from its unchanged live_out set. */
   refresh_block_liveness_and_kills(p, defs, pub_defs, liveness, blk, blk_ctx);
   return true;
}

static bool
is_remat_long_span_alu(const struct agx_apple9_vir_program *p,
                       const BITSET_WORD *pub_defs,
                       const struct agx_apple9_vir_instr *ins)
{
   if (!ins || ins->dest == AGX_APPLE9_VREG_INVALID ||
       ins->dest_components != 1 || ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
       ins->publication_handoff || apple9_ra_is_publication_instr(ins) ||
       ins->memory_volatile || ins->local_control ||
       ins->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
       ins->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE)
      return false;

   uint32_t v = ins->dest;
   if (!is_valid_gpr(p, pub_defs, v))
      return false;
   if (p->fixed_phys && p->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID)
      return false;

   switch (ins->op) {
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_ISUB:
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_IMAD:
      break;
   default:
      return false;
   }

   if (agx_apple9_vir_nr_srcs(ins) != 1)
      return false;
   uint32_t s0 = agx_apple9_vir_src(ins, 0);
   if (!is_valid_gpr(p, pub_defs, s0) || s0 == v)
      return false;

   return true;
}

/* On high-pressure blocks (> 52 live GPRs), undo overly aggressive CSE of cheap
 * 1-GPR-source pure integer address derivations (IMAD/USHR/IADD/...) across
 * long dead gaps (>= 48 instructions) when the base source is already live or
 * shared by multiple derived values. This prevents e.g. skinned-mesh shaders
 * from holding dozens of CSE'd bone-matrix offsets live across hundreds of
 * instructions and crossing into the >80-GPR occupancy tier or spilling. */
static bool
apple9_ra_remat_long_spans(struct agx_apple9_vir_program *p,
                           const struct apple9_ra_liveness_info *liveness)
{
   const unsigned vcount = p->value_count;
   if (vcount == 0 || liveness->num_cfg_blocks == 0)
      return false;

   bool any_high_pressure = false;
   for (unsigned b = 0; b < liveness->num_cfg_blocks; ++b) {
      if (liveness->block_peak_gprs[b] > 46) {
         any_high_pressure = true;
         break;
      }
   }
   if (!any_high_pressure)
      return false;

   void *mem_ctx = ralloc_context(NULL);
   if (!mem_ctx)
      return false;

   struct agx_apple9_vir_instr **defs = NULL;
   BITSET_WORD *pub_defs = NULL;
   if (!build_defs_and_pub_mask(p, mem_ctx, &defs, &pub_defs)) {
      ralloc_free(mem_ctx);
      return false;
   }

   BITSET_WORD *global_live_out = BITSET_RZALLOC(mem_ctx, vcount);
   if (!global_live_out) {
      ralloc_free(mem_ctx);
      return false;
   }
   if (p->output < vcount)
      BITSET_SET(global_live_out, p->output);
   for (unsigned k = 0; k < p->live_out_count; ++k) {
      if (p->live_out[k] < vcount)
         BITSET_SET(global_live_out, p->live_out[k]);
   }

   bool any_changed = false;

   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      if (!blk->cfg_active)
         continue;
      const unsigned b = blk->cfg_index;
      if (liveness->block_peak_gprs[b] <= 46)
         continue;

      struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
      while (I && is_pinned_entry(I))
         I = agx_apple9_instr_next(I);
      struct agx_apple9_vir_instr *first_body = I;
      if (!first_body)
         continue;

      struct agx_apple9_vir_instr *first_tail = NULL;
      for (struct agx_apple9_vir_instr *T = agx_apple9_block_last_instr(blk); T;
           T = agx_apple9_instr_prev(T)) {
         if (is_pinned_tail(T))
            first_tail = T;
         else
            break;
         if (T == first_body)
            break;
      }

      unsigned body_count = 0;
      for (I = first_body; I && I != first_tail; I = agx_apple9_instr_next(I))
         body_count++;
      if (body_count < 64)
         continue;

      unsigned blk_len = 0;
      unsigned total_src_uses = 0;
      for (I = agx_apple9_block_first_instr(blk); I; I = agx_apple9_instr_next(I)) {
         blk_len++;
         total_src_uses += agx_apple9_vir_nr_srcs(I);
      }
      if (total_src_uses == 0)
         continue;

      void *blk_ctx = ralloc_context(mem_ctx);
      if (!blk_ctx)
         continue;
      struct agx_apple9_vir_instr **ins_arr =
         ralloc_array(blk_ctx, struct agx_apple9_vir_instr *, blk_len);
      uint32_t *ctrl_epoch = ralloc_array(blk_ctx, uint32_t, blk_len);
      uint32_t *uniform_epoch = ralloc_array(blk_ctx, uint32_t, blk_len);
      struct agx_apple9_vir_instr **cand_ins =
         rzalloc_array(blk_ctx, struct agx_apple9_vir_instr *, vcount);
      uint32_t *def_pos = ralloc_array(blk_ctx, uint32_t, vcount);
      int32_t *last_use_pos = ralloc_array(blk_ctx, int32_t, vcount);
      uint32_t *use_head = ralloc_array(blk_ctx, uint32_t, vcount);
      uint32_t *use_tail = ralloc_array(blk_ctx, uint32_t, vcount);
      uint32_t *use_pos = ralloc_array(blk_ctx, uint32_t, total_src_uses);
      uint32_t *use_next = ralloc_array(blk_ctx, uint32_t, total_src_uses);
      uint32_t *first_gap_use = ralloc_array(blk_ctx, uint32_t, vcount);
      uint32_t *derived_cnt = rzalloc_array(blk_ctx, uint32_t, vcount);
      BITSET_WORD *disqualify = BITSET_RZALLOC(blk_ctx, vcount);
      BITSET_WORD *pinned_src = BITSET_RZALLOC(blk_ctx, vcount);
      uint32_t *cand_order = ralloc_array(blk_ctx, uint32_t, body_count);
      struct agx_apple9_vir_instr **to_remove =
         ralloc_array(blk_ctx, struct agx_apple9_vir_instr *, body_count);

      if (!blk_ctx || !ins_arr || !ctrl_epoch || !uniform_epoch || !cand_ins ||
          !def_pos || !last_use_pos || !use_head || !use_tail || !use_pos ||
          !use_next || !first_gap_use || !derived_cnt || !disqualify ||
          !pinned_src || !cand_order || !to_remove) {
         ralloc_free(blk_ctx);
         continue;
      }

      for (unsigned v = 0; v < vcount; ++v) {
         last_use_pos[v] = -1;
         use_head[v] = UINT32_MAX;
         use_tail[v] = UINT32_MAX;
         first_gap_use[v] = UINT32_MAX;
      }

      unsigned pos = 0;
      unsigned body_start_pos = UINT32_MAX;
      unsigned tail_start_pos = blk_len;
      uint32_t c_ep = 0;
      uint32_t u_ep = 0;
      unsigned use_cnt = 0;

      for (I = agx_apple9_block_first_instr(blk); I; I = agx_apple9_instr_next(I), ++pos) {
         if (I == first_body)
            body_start_pos = pos;
         if (I == first_tail)
            tail_start_pos = pos;
         if (I->op == AGX_APPLE9_VIR_STORE_UNIFORM)
            u_ep++;
         if (I->local_control || apple9_ra_after_logical_end(I))
            c_ep++;

         ins_arr[pos] = I;
         ctrl_epoch[pos] = c_ep;
         uniform_epoch[pos] = u_ep;

         if (pos >= body_start_pos && pos < tail_start_pos &&
             is_remat_long_span_alu(p, pub_defs, I) &&
             I->dest < vcount && !BITSET_TEST(global_live_out, I->dest) &&
             !u_sparse_bitset_test(&liveness->live_out[b], I->dest)) {
            cand_ins[I->dest] = I;
            def_pos[I->dest] = pos;
         }

         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT ||
             I->op == AGX_APPLE9_VIR_PHI)
            continue;

         const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         bool bad_user = (I->op == AGX_APPLE9_VIR_SPLIT ||
                          I->op == AGX_APPLE9_VIR_PHI_SRC ||
                          I->op == AGX_APPLE9_VIR_ABI_RETURN);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= vcount)
               continue;
            if (bad_user)
               BITSET_SET(disqualify, v);
            last_use_pos[v] = (int32_t)pos;
            if (use_tail[v] != UINT32_MAX && use_pos[use_tail[v]] == pos)
               continue;
            uint32_t uid = use_cnt++;
            use_pos[uid] = pos;
            use_next[uid] = UINT32_MAX;
            if (use_tail[v] != UINT32_MAX)
               use_next[use_tail[v]] = uid;
            else
               use_head[v] = uid;
            use_tail[v] = uid;
         }
      }

      unsigned cand_count = 0;
      for (unsigned p_i = body_start_pos; p_i < tail_start_pos; ++p_i) {
         struct agx_apple9_vir_instr *orig = ins_arr[p_i];
         if (orig->dest >= vcount || cand_ins[orig->dest] != orig)
            continue;
         uint32_t v = orig->dest;
         if (BITSET_TEST(disqualify, v) || use_head[v] == UINT32_MAX) {
            cand_ins[v] = NULL;
            continue;
         }

         /* Locate the first dead gap >= 48 instructions along v's uses. */
         uint32_t prev_p = def_pos[v];
         uint32_t gap_u = UINT32_MAX;
         for (uint32_t u = use_head[v]; u != UINT32_MAX; u = use_next[u]) {
            if (use_pos[u] >= prev_p + 48u) {
               gap_u = u;
               break;
            }
            prev_p = use_pos[u];
         }
         if (gap_u == UINT32_MAX) {
            cand_ins[v] = NULL;
            continue;
         }

         /* Every use from gap_u onward must be inside the schedulable body and
          * share orig's uniform epoch if orig reads a uniform operand. */
         bool ok = true;
         uint32_t def_u_ep = uniform_epoch[def_pos[v]];
         for (uint32_t u = gap_u; u != UINT32_MAX; u = use_next[u]) {
            uint32_t up = use_pos[u];
            if (up >= tail_start_pos) {
               ok = false;
               break;
            }
            if (orig->alu_src_uniform_mask && uniform_epoch[up] != def_u_ep) {
               ok = false;
               break;
            }
         }
         if (!ok) {
            cand_ins[v] = NULL;
            continue;
         }

         first_gap_use[v] = gap_u;
         uint32_t s0 = agx_apple9_vir_src(orig, 0);
         if (s0 < vcount)
            derived_cnt[s0]++;
         cand_order[cand_count++] = v;
      }

      /* Process leaves first (descending def_pos) and pin each remat source so
       * chained derivations stay SSA-consistent within a single pass. */
      unsigned remove_count = 0;
      for (unsigned ci = cand_count; ci-- > 0;) {
         uint32_t v = cand_order[ci];
         struct agx_apple9_vir_instr *orig = cand_ins[v];
         if (!orig || BITSET_TEST(pinned_src, v))
            continue;

         uint32_t s0 = agx_apple9_vir_src(orig, 0);
         int32_t max_up = last_use_pos[v];
         bool s0_already_live =
            u_sparse_bitset_test(&liveness->live_out[b], s0) ||
            last_use_pos[s0] >= max_up;
         if (!s0_already_live && derived_cnt[s0] < 2u)
            continue;

         uint32_t cur_clone_v = AGX_APPLE9_VREG_INVALID;
         uint32_t cluster_start_pos = 0;
         uint32_t cluster_c_ep = 0;
         bool failed = false;

         for (uint32_t u = first_gap_use[v]; u != UINT32_MAX; u = use_next[u]) {
            uint32_t up = use_pos[u];
            struct agx_apple9_vir_instr *u_ins = ins_arr[up];

            if (cur_clone_v == AGX_APPLE9_VREG_INVALID ||
                up >= cluster_start_pos + 32u ||
                ctrl_epoch[up] != cluster_c_ep) {
               uint32_t new_v = AGX_APPLE9_VREG_INVALID;
               if (!apple9_ra_vir_grow_values(p, NULL, 1, &new_v)) {
                  failed = true;
                  break;
               }
               if (p->max_phys && v < p->value_count &&
                   p->max_phys[v] != AGX_APPLE9_PHYS_INVALID)
                  p->max_phys[new_v] = p->max_phys[v];

               struct agx_apple9_block *saved_blk = p->current_block;
               p->current_block = blk;
               bool ok = agx_apple9_vir_emit_side_effect(
                  p, orig->op, orig->encoding, &s0, 1, orig->immediate);
               p->current_block = saved_blk;
               if (!ok) {
                  failed = true;
                  break;
               }

               struct agx_apple9_vir_instr *copy = agx_apple9_block_last_instr(blk);
               *copy = *orig;
               copy->dest = new_v;
               copy->dest_components = 1;
               copy->phi_block = NULL;
               copy->phi_edge = NULL;
               copy->phi_srcs = NULL;
               copy->phi_src_count = 0;
               copy->completion_consumer = NULL;
               copy->kill_mask = 0;
               copy->retain_mask = 0;
               copy->live_after_mask = 0;
               copy->dead_dest_mask = 0;
               copy->nr_srcs = 1;
               copy->src[0] = s0;
               agx_apple9_vir_move_before(p, copy, u_ins);

               cur_clone_v = new_v;
               cluster_start_pos = up;
               cluster_c_ep = ctrl_epoch[up];
            }

            const unsigned nr_srcs = agx_apple9_vir_nr_srcs(u_ins);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t *sp = agx_apple9_vir_src_ptr(u_ins, s);
               if (*sp == v)
                  *sp = cur_clone_v;
            }
         }

         if (failed) {
            if (cur_clone_v != AGX_APPLE9_VREG_INVALID)
               any_changed = true;
            break;
         }

         BITSET_SET(pinned_src, s0);
         if (max_up > last_use_pos[s0])
            last_use_pos[s0] = max_up;
         if (first_gap_use[v] == use_head[v])
            to_remove[remove_count++] = orig;
         any_changed = true;
      }

      if (remove_count > 0)
         agx_apple9_vir_remove_many(p, to_remove, remove_count);

      ralloc_free(blk_ctx);
   }

   ralloc_free(mem_ctx);
   return any_changed;
}

/* On high-pressure blocks (> 46 live GPRs), rematerialize cheap pure 0/1-GPR-src
 * integer derivations (e.g. CSE'd buffer-size bounds limits or offset IMADs)
 * into dominated sibling/child blocks at the same or lower loop depth when their
 * base GPR is already live-out (or is a cloneable IMM). This removes them from
 * live_out of the high-pressure block without cloning into deeper loops. */
static bool
apple9_ra_remat_cross_block(struct agx_apple9_vir_program *p,
                            const struct apple9_ra_liveness_info *liveness)
{
   const unsigned vcount = p->value_count;
   const unsigned nblks = liveness->num_cfg_blocks;
   if (vcount == 0 || nblks <= 1)
      return false;

   bool any_high_pressure = false;
   for (unsigned b = 0; b < nblks; ++b) {
      if (liveness->block_peak_gprs[b] > 46) {
         any_high_pressure = true;
         break;
      }
   }
   if (!any_high_pressure)
      return false;

   void *mem_ctx = ralloc_context(NULL);
   if (!mem_ctx)
      return false;

   struct agx_apple9_vir_instr **defs = NULL;
   BITSET_WORD *pub_defs = NULL;
   if (!build_defs_and_pub_mask(p, mem_ctx, &defs, &pub_defs)) {
      ralloc_free(mem_ctx);
      return false;
   }

   unsigned *blk_loop_depth = rzalloc_array(mem_ctx, unsigned, nblks);
   struct agx_apple9_block **def_blk =
      rzalloc_array(mem_ctx, struct agx_apple9_block *, vcount);
   unsigned *ext_use_blks = rzalloc_array(mem_ctx, unsigned, vcount);
   unsigned *in_blk_uses = rzalloc_array(mem_ctx, unsigned, vcount);
   BITSET_WORD *disqualify = BITSET_RZALLOC(mem_ctx, vcount);
   BITSET_WORD *pinned_src = BITSET_RZALLOC(mem_ctx, vcount);
   if (!blk_loop_depth || !def_blk || !ext_use_blks || !in_blk_uses ||
       !disqualify || !pinned_src) {
      ralloc_free(mem_ctx);
      return false;
   }

   if (p->output < vcount)
      BITSET_SET(disqualify, p->output);
   for (unsigned k = 0; k < p->live_out_count; ++k) {
      if (p->live_out[k] < vcount)
         BITSET_SET(disqualify, p->live_out[k]);
   }

   bool has_store_uniform = false;
   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      if (!blk->cfg_active)
         continue;
      const unsigned b = blk->cfg_index;
      for (unsigned si = 0; si < 2; ++si) {
         struct agx_apple9_block *succ = blk->cfg_succs[si];
         if (succ && succ->cfg_active && succ->cfg_index <= b) {
            for (unsigned lb = succ->cfg_index; lb <= b; ++lb)
               blk_loop_depth[lb]++;
         }
      }
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_STORE_UNIFORM)
            has_store_uniform = true;
         if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components == 1 &&
             I->dest < vcount)
            def_blk[I->dest] = blk;
      }
   }

   /* Scan all uses across the program to find disqualifiers and external-block
    * use counts. */
   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      const bool active = blk->cfg_active;
      const unsigned b = active ? blk->cfg_index : 0;
      BITSET_WORD *seen_in_blk = active ? BITSET_RZALLOC(mem_ctx, vcount) : NULL;
      if (active && !seen_in_blk) {
         ralloc_free(mem_ctx);
         return false;
      }
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
            continue;
         if (I->op == AGX_APPLE9_VIR_PHI) {
            const unsigned nr_phi = agx_apple9_vir_nr_srcs(I);
            for (unsigned k = 0; k < nr_phi; ++k) {
               uint32_t pv = agx_apple9_vir_src(I, k);
               if (pv < vcount)
                  BITSET_SET(disqualify, pv);
            }
            continue;
         }
         bool bad_user = (!active ||
                          I->op == AGX_APPLE9_VIR_SPLIT ||
                          I->op == AGX_APPLE9_VIR_PHI_SRC ||
                          I->op == AGX_APPLE9_VIR_ABI_RETURN ||
                          is_pinned_entry(I));
         const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= vcount)
               continue;
            if (bad_user) {
               BITSET_SET(disqualify, v);
               continue;
            }
            if (!def_blk[v]) {
               BITSET_SET(disqualify, v);
               continue;
            }
            if (def_blk[v] == blk) {
               in_blk_uses[v]++;
            } else {
               bool small_imm = defs[v] && defs[v]->op == AGX_APPLE9_VIR_IMM &&
                                defs[v]->dest_components == 1 &&
                                !defs[v]->local_control &&
                                agx_apple9_vir_nr_srcs(defs[v]) == 0 &&
                                defs[v]->immediate <= 0x7fu;
               if (!small_imm &&
                   blk_loop_depth[b] > blk_loop_depth[def_blk[v]->cfg_index])
                  BITSET_SET(disqualify, v);
               if (!BITSET_TEST(seen_in_blk, v)) {
                  BITSET_SET(seen_in_blk, v);
                  ext_use_blks[v]++;
               }
            }
         }
      }
      if (seen_in_blk)
         ralloc_free(seen_in_blk);
   }

   bool any_changed = false;
   for (struct agx_apple9_block *blk = p->blocks; blk; blk = blk->next) {
      if (!blk->cfg_active)
         continue;
      const unsigned b = blk->cfg_index;
      const bool high_blk = liveness->block_peak_gprs[b] > 46;

      /* Collect candidates in blk in reverse order (leaves first). */
      unsigned blk_len = 0;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I; I = agx_apple9_instr_next(I))
         blk_len++;
      if (blk_len == 0)
         continue;

      struct agx_apple9_vir_instr **cands =
         ralloc_array(mem_ctx, struct agx_apple9_vir_instr *, blk_len);
      struct agx_apple9_vir_instr **to_remove =
         ralloc_array(mem_ctx, struct agx_apple9_vir_instr *, blk_len);
      if (!cands || !to_remove)
         break;

      unsigned cand_count = 0;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I; I = agx_apple9_instr_next(I)) {
         if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components != 1 ||
             I->dest >= vcount || BITSET_TEST(pub_defs, I->dest))
            continue;
         uint32_t v = I->dest;
         if (BITSET_TEST(disqualify, v) || ext_use_blks[v] == 0 ||
             !u_sparse_bitset_test(&liveness->live_out[b], v))
            continue;
         if (I->alu_src_uniform_mask && has_store_uniform)
            continue;
         bool is_imm = (I->op == AGX_APPLE9_VIR_IMM && !I->local_control &&
                        agx_apple9_vir_nr_srcs(I) == 0);
         bool small_imm = is_imm && I->immediate <= 0x7fu;
         if (!high_blk && !small_imm)
            continue;
         if (!is_imm && !is_remat_long_span_alu(p, pub_defs, I))
            continue;
         cands[cand_count++] = I;
      }

      unsigned remove_count = 0;
      for (unsigned ci = cand_count; ci-- > 0;) {
         struct agx_apple9_vir_instr *orig = cands[ci];
         uint32_t v = orig->dest;
         if (BITSET_TEST(pinned_src, v))
            continue;

         const bool is_imm = (orig->op == AGX_APPLE9_VIR_IMM);
         uint32_t s0 = AGX_APPLE9_VREG_INVALID;
         struct agx_apple9_vir_instr *s0_imm = NULL;
         if (!is_imm) {
            s0 = agx_apple9_vir_src(orig, 0);
            if (s0 >= vcount)
               continue;
            bool s0_live_out = u_sparse_bitset_test(&liveness->live_out[b], s0);
            if (!s0_live_out) {
               if (defs[s0] && defs[s0]->op == AGX_APPLE9_VIR_IMM &&
                   defs[s0]->dest_components == 1 && !defs[s0]->local_control &&
                   agx_apple9_vir_nr_srcs(defs[s0]) == 0 &&
                   !BITSET_TEST(pub_defs, s0)) {
                  s0_imm = defs[s0];
               } else {
                  continue;
               }
            }
         }

         /* Clone orig (and s0_imm if needed) into every other block that uses v. */
         bool failed = false;
         bool cloned_any = false;
         for (struct agx_apple9_block *ub = p->blocks; ub; ub = ub->next) {
            if (!ub->cfg_active || ub == blk)
               continue;
            struct agx_apple9_vir_instr *first_u = NULL;
            for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(ub);
                 I; I = agx_apple9_instr_next(I)) {
               if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT ||
                   I->op == AGX_APPLE9_VIR_PHI)
                  continue;
               const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
               for (unsigned s = 0; s < nr_srcs; ++s) {
                  if (agx_apple9_vir_src(I, s) == v) {
                     first_u = I;
                     break;
                  }
               }
               if (first_u)
                  break;
            }
            if (!first_u)
               continue;

            uint32_t clone_src = s0;
            if (s0_imm) {
               uint32_t new_imm_v = AGX_APPLE9_VREG_INVALID;
               if (!apple9_ra_vir_grow_values(p, NULL, 1, &new_imm_v)) {
                  failed = true;
                  break;
               }
               if (p->max_phys && s0 < p->value_count &&
                   p->max_phys[s0] != AGX_APPLE9_PHYS_INVALID)
                  p->max_phys[new_imm_v] = p->max_phys[s0];
               struct agx_apple9_block *saved_blk = p->current_block;
               p->current_block = ub;
               bool ok = agx_apple9_vir_emit_side_effect(
                  p, s0_imm->op, s0_imm->encoding, NULL, 0, s0_imm->immediate);
               p->current_block = saved_blk;
               if (!ok) {
                  failed = true;
                  break;
               }
               struct agx_apple9_vir_instr *imm_copy = agx_apple9_block_last_instr(ub);
               *imm_copy = *s0_imm;
               imm_copy->dest = new_imm_v;
               imm_copy->dest_components = 1;
               imm_copy->phi_block = NULL;
               imm_copy->phi_edge = NULL;
               imm_copy->phi_srcs = NULL;
               imm_copy->phi_src_count = 0;
               imm_copy->completion_consumer = NULL;
               imm_copy->kill_mask = 0;
               imm_copy->retain_mask = 0;
               imm_copy->live_after_mask = 0;
               imm_copy->dead_dest_mask = 0;
               imm_copy->nr_srcs = 0;
               agx_apple9_vir_move_before(p, imm_copy, first_u);
               clone_src = new_imm_v;
            }

            uint32_t new_v = AGX_APPLE9_VREG_INVALID;
            if (!apple9_ra_vir_grow_values(p, NULL, 1, &new_v)) {
               failed = true;
               break;
            }
            if (p->max_phys && v < p->value_count &&
                p->max_phys[v] != AGX_APPLE9_PHYS_INVALID)
               p->max_phys[new_v] = p->max_phys[v];

            struct agx_apple9_block *saved_blk = p->current_block;
            p->current_block = ub;
            bool ok = agx_apple9_vir_emit_side_effect(
               p, orig->op, orig->encoding, is_imm ? NULL : &clone_src,
               is_imm ? 0 : 1, orig->immediate);
            p->current_block = saved_blk;
            if (!ok) {
               failed = true;
               break;
            }
            struct agx_apple9_vir_instr *copy = agx_apple9_block_last_instr(ub);
            *copy = *orig;
            copy->dest = new_v;
            copy->dest_components = 1;
            copy->phi_block = NULL;
            copy->phi_edge = NULL;
            copy->phi_srcs = NULL;
            copy->phi_src_count = 0;
            copy->completion_consumer = NULL;
            copy->kill_mask = 0;
            copy->retain_mask = 0;
            copy->live_after_mask = 0;
            copy->dead_dest_mask = 0;
            copy->nr_srcs = is_imm ? 0 : 1;
            if (!is_imm)
               copy->src[0] = clone_src;
            agx_apple9_vir_move_before(p, copy, first_u);

            for (struct agx_apple9_vir_instr *I = first_u; I;
                 I = agx_apple9_instr_next(I)) {
               if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT ||
                   I->op == AGX_APPLE9_VIR_PHI)
                  continue;
               const unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
               for (unsigned s = 0; s < nr_srcs; ++s) {
                  uint32_t *sp = agx_apple9_vir_src_ptr(I, s);
                  if (*sp == v)
                     *sp = new_v;
               }
            }
            cloned_any = true;
         }

         if (failed) {
            if (cloned_any)
               any_changed = true;
            break;
         }
         if (cloned_any) {
            if (!is_imm && !s0_imm)
               BITSET_SET(pinned_src, s0);
            if (in_blk_uses[v] == 0)
               to_remove[remove_count++] = orig;
            any_changed = true;
         }
      }

      if (remove_count > 0)
         agx_apple9_vir_remove_many(p, to_remove, remove_count);
      ralloc_free(cands);
      ralloc_free(to_remove);
   }

   ralloc_free(mem_ctx);
   return any_changed;
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
      if (cfg->liveness_out)
         *cfg->liveness_out = liveness;
      else
         apple9_ra_liveness_finish(&liveness);
      return true;
   }

   for (unsigned pass = 0; pass < 2; ++pass) {
      bool changed = apple9_ra_remat_cross_block(p, &liveness);
      if (changed) {
         apple9_ra_liveness_finish(&liveness);
         if (!apple9_ra_liveness(p, &liveness))
            return false;
      }
      bool long_changed = apple9_ra_remat_long_spans(p, &liveness);
      if (long_changed) {
         apple9_ra_liveness_finish(&liveness);
         if (!apple9_ra_liveness(p, &liveness))
            return false;
      }
      if (!changed && !long_changed)
         break;
   }

   void *mem_ctx = ralloc_context(NULL);
   if (!mem_ctx) {
      apple9_ra_liveness_finish(&liveness);
      return false;
   }

   agx_apple9_vir_ensure_indexed(p);

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
         orig_all = MAX2(orig_all, cur_all + delta.peak_all);
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

      if (cfg->latency && cfg->auto_tier_slack) {
         /* Find the best achievable G16G occupancy tier before granting
          * latency-hiding slack. Below 45 GPRs, real occupancy steps sharply
          * at 18/24/28/32/36/40/44 GPRs (2048 -> 896 thr/core); from 45..96
          * GPRs, occupancy is flat at 832 thr/core so we only compress there
          * when it unlocks a sub-45 tier, stays within compact r0..r63, or
          * avoids spilling past 96 GPRs. */
         void *pres_ctx = NULL;
         struct vir_sched_result pres = {0};
         bool have_pres = false;
         int32_t min_all = orig_all;
         unsigned orig_occ =
            agx_occupancy_for_register_count(
               2u * MIN2(MAX2((unsigned)MAX2(orig_all, 1), 1u),
                         (unsigned)AGX_APPLE9_GPR_COUNT))
               .max_threads;
         if (orig_all > 24) {
            pres_ctx = ralloc_context(blk_ctx);
            pres = run_block_schedule_attempt(
               p, defs, pub_defs, val_max_gpr, first_body, first_tail,
               &live_after_body, base_all, base_64, base_16, false, INT32_MAX,
               last_write, pres_ctx, NULL, NULL);
            if (pres.valid &&
                bank_limits_respected(pres.max_64, pres.max_16, orig_64,
                                      orig_16) &&
                pressure_strictly_better(pres.max_all, pres.max_64, pres.max_16,
                                         orig_all, orig_64, orig_16)) {
               unsigned pres_occ =
                  agx_occupancy_for_register_count(
                     2u * MIN2(MAX2((unsigned)MAX2(pres.max_all, 1), 1u),
                               (unsigned)AGX_APPLE9_GPR_COUNT))
                     .max_threads;
               if (pres_occ > orig_occ || orig_all > 84 ||
                   (orig_all > 60 && pres.max_all <= 60)) {
                  have_pres = true;
                  min_all = pres.max_all;
               }
            }
            /* Order-preserving load sinking: only pursue when it can cross a
             * real sub-45 occupancy cliff or avert spilling (> 84 GPRs). Never
             * sink loads just to move within the flat 45..84 GPR shelf. */
            if ((min_all > 24 && min_all <= 64) || min_all > 84) {
               unsigned cur_occ =
                  agx_occupancy_for_register_count(
                     2u * MIN2(MAX2((unsigned)MAX2(min_all, 1), 1u),
                               (unsigned)AGX_APPLE9_GPR_COUNT))
                     .max_threads;
               void *sink_ctx = ralloc_context(blk_ctx);
               struct vir_sched_result sink = run_block_sink_loads(
                  p, defs, pub_defs, val_max_gpr, first_body, first_tail,
                  &live_after_body, base_all, base_64, base_16, last_write,
                  sink_ctx);
               if (sink.valid &&
                   bank_limits_respected(sink.max_64, sink.max_16, orig_64,
                                         orig_16) &&
                   pressure_strictly_better(
                      sink.max_all, sink.max_64, sink.max_16,
                      have_pres ? pres.max_all : orig_all,
                      have_pres ? pres.max_64 : orig_64,
                      have_pres ? pres.max_16 : orig_16)) {
                  unsigned sink_occ =
                     agx_occupancy_for_register_count(
                        2u * MIN2(MAX2((unsigned)MAX2(sink.max_all, 1), 1u),
                                  (unsigned)AGX_APPLE9_GPR_COUNT))
                        .max_threads;
                  if (sink_occ > cur_occ || min_all > 84) {
                     ralloc_free(pres_ctx);
                     pres_ctx = sink_ctx;
                     pres = sink;
                     have_pres = true;
                     min_all = sink.max_all;
                  } else {
                     ralloc_free(sink_ctx);
                  }
               } else {
                  ralloc_free(sink_ctx);
               }
            }
         }

         unsigned clamped_min =
            MIN2(MAX2((unsigned)MAX2(min_all, 6), 6u),
                 (unsigned)AGX_APPLE9_GPR_COUNT);
         unsigned tier_gprs =
            MIN2(agx_occupancy_for_register_count(2u * clamped_min)
                    .max_registers / 2u,
                 (unsigned)AGX_APPLE9_GPR_COUNT);
         /* Leave alignment/RA headroom below each real G16G occupancy cliff
          * (18/24/28/32/36/40/44 GPRs). Across the flat 45..96 GPR shelf
          * (832 thr/core), protect the r0..r63 compact ALU boundary when
          * comfortably below 60, and grant generous latency slack up to 88
          * GPRs without risking spills past 96. */
         int32_t safe_tier;
         if (tier_gprs <= 28u)
            safe_tier = (int32_t)(tier_gprs - 2u);
         else if (tier_gprs <= 40u)
            safe_tier = (int32_t)(tier_gprs - 3u);
         else if (tier_gprs <= 44u)
            safe_tier = (int32_t)(tier_gprs - 4u);
         else
            safe_tier = (min_all <= 60) ? 60 : 88;
         int32_t lat_limit = min_all;
         if (min_all < safe_tier)
            lat_limit = MIN2(safe_tier, min_all + (tier_gprs > 44u ? 16 : 8));

         void *lat_ctx = ralloc_context(blk_ctx);
         unsigned orig_stalls = 0, orig_dual_issue = 0;
         struct vir_sched_result lat = run_block_schedule_attempt(
            p, defs, pub_defs, val_max_gpr, first_body, first_tail,
            &live_after_body, base_all, base_64, base_16, true, lat_limit,
            last_write, lat_ctx, &orig_stalls, &orig_dual_issue);

         bool accept_lat = false;
         if (lat.valid) {
            if (lat.max_all > lat_limit) {
               cfg->rejected_pressure_increase = MAX2(
                  cfg->rejected_pressure_increase,
                  (unsigned)MAX2(lat.max_all - min_all, 0));
            } else if (bank_limits_respected(lat.max_64, lat.max_16, orig_64,
                                             orig_16)) {
               if (have_pres) {
                  /* Accept latency schedule only if it stays within the
                   * compressed tier ceiling and hides stalls (or matches pres). */
                  accept_lat = (lat.stalls < orig_stalls) ||
                               (lat.max_all <= pres.max_all);
               } else {
                  accept_lat =
                     (lat.stalls < orig_stalls) ||
                     (lat.stalls == orig_stalls &&
                      (pressure_strictly_better(lat.max_all, lat.max_64,
                                                lat.max_16, orig_all, orig_64,
                                                orig_16) ||
                       (lat.max_all <= orig_all && lat.max_64 <= orig_64 &&
                        lat.max_16 <= orig_16 &&
                        lat.dual_issue > orig_dual_issue)));
               }
            }
         }

         if (accept_lat) {
            any_reordered |= apply_vir_schedule(p, blk, last_entry, lat.order,
                                                lat.count, defs, pub_defs,
                                                &liveness, blk_ctx);
            committed = true;
         } else if (have_pres) {
            any_reordered |= apply_vir_schedule(p, blk, last_entry, pres.order,
                                                pres.count, defs, pub_defs,
                                                &liveness, blk_ctx);
            committed = true;
         }

         ralloc_free(lat_ctx);
         if (pres_ctx)
            ralloc_free(pres_ctx);
         ralloc_free(blk_ctx);
         continue;
      }

      if (cfg->latency) {
         void *lat_ctx = ralloc_context(blk_ctx);
         unsigned orig_stalls = 0, orig_dual_issue = 0;
         struct vir_sched_result lat = run_block_schedule_attempt(
            p, defs, pub_defs, val_max_gpr, first_body, first_tail,
            &live_after_body, base_all, base_64, base_16, true, INT32_MAX,
            last_write, lat_ctx, &orig_stalls, &orig_dual_issue);

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
                   (pressure_strictly_better(lat.max_all, lat.max_64, lat.max_16,
                                             orig_all, orig_64, orig_16) ||
                    (lat.max_all <= orig_all && lat.max_64 <= orig_64 &&
                     lat.max_16 <= orig_16 &&
                     lat.dual_issue > orig_dual_issue)));
            }
         }

         if (accept_lat) {
            bool overridden_by_pressure = false;
            if (cfg->pressure_slack == 0 && lat.max_all > base_all) {
               void *pres_ctx = ralloc_context(blk_ctx);
               struct vir_sched_result pres = run_block_schedule_attempt(
                  p, defs, pub_defs, val_max_gpr, first_body, first_tail,
                  &live_after_body, base_all, base_64, base_16, false,
                  INT32_MAX, last_write, pres_ctx, NULL, NULL);
               if (pres.valid &&
                   bank_limits_respected(pres.max_64, pres.max_16, orig_64,
                                         orig_16) &&
                   pressure_strictly_better(pres.max_all, pres.max_64,
                                            pres.max_16, lat.max_all,
                                            lat.max_64, lat.max_16)) {
                  any_reordered |= apply_vir_schedule(p, blk, last_entry,
                                                      pres.order, pres.count,
                                                      defs, pub_defs, &liveness,
                                                      blk_ctx);
                  committed = true;
                  overridden_by_pressure = true;
               }
               ralloc_free(pres_ctx);
            }
            if (!overridden_by_pressure) {
               any_reordered |= apply_vir_schedule(p, blk, last_entry,
                                                   lat.order, lat.count, defs,
                                                   pub_defs, &liveness, blk_ctx);
               committed = true;
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
            &live_after_body, base_all, base_64, base_16, false, INT32_MAX,
            last_write, pres_ctx, NULL, NULL);

         if (pres.valid &&
             bank_limits_respected(pres.max_64, pres.max_16, orig_64, orig_16) &&
             pressure_strictly_better(pres.max_all, pres.max_64, pres.max_16,
                                      orig_all, orig_64, orig_16)) {
            any_reordered |= apply_vir_schedule(p, blk, last_entry, pres.order,
                                                pres.count, defs, pub_defs,
                                                &liveness, blk_ctx);
         }
         ralloc_free(pres_ctx);
      }

      ralloc_free(blk_ctx);
   }

   ralloc_free(mem_ctx);

   if (any_reordered) {
      agx_apple9_vir_ensure_indexed(p);
      unsigned global_peak = 0;
      for (unsigned b = 0; b < liveness.num_cfg_blocks; ++b)
         global_peak = MAX2(global_peak, liveness.block_peak_gprs[b]);
      liveness.peak_live_gprs = global_peak;
      p->peak_live_gprs = global_peak;
   }

   if (cfg->liveness_out)
      *cfg->liveness_out = liveness;
   else
      apple9_ra_liveness_finish(&liveness);
   return true;
}

bool
apple9_ra_pressure_schedule(struct agx_apple9_vir_program *p)
{
   return apple9_ra_pressure_schedule_opts(p, NULL);
}
