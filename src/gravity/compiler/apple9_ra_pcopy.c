/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_pcopy.h"

#include <stdlib.h>
#include <string.h>

#include "util/macros.h"

#define PCOPY_STACK_PENDING 128u
#define PCOPY_STACK_MEM     256u

struct pcopy_entry {
   struct apple9_ra_loc dst;
   struct apple9_ra_loc src;
};

struct pcopy_state {
   struct pcopy_entry *pending;
   unsigned nr_pending;

   /* Compact GPR[96] tracking on stack (no 384 KB copy_ctx malloc!). */
   uint16_t gpr_use[AGX_APPLE9_GPR_COUNT];
   uint64_t gpr_written[2];
   uint64_t gpr_pending_dest[2];
   uint64_t live_gprs[2];

   /* Tightly sized MEM[mem_count] tracking (stack for <=256 slots). */
   uint16_t *mem_use;
   unsigned mem_count;

   uint8_t scratch_gpr;
   uint8_t scratch_gpr2;
   uint8_t max_temp_gpr;
   uint8_t mov_imm_max_gpr;
   bool prefer_swap_2cycle;

   apple9_ra_pcopy_emit_fn emit;
   void *emit_ctx;
};

static inline bool
gpr_bit_test(const uint64_t mask[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   return (mask[r / 64] >> (r % 64)) & 1ull;
}

static inline void
gpr_bit_set(uint64_t mask[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   mask[r / 64] |= 1ull << (r % 64);
}

static inline void
gpr_bit_clear(uint64_t mask[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   mask[r / 64] &= ~(1ull << (r % 64));
}

static inline bool
loc_equal(struct apple9_ra_loc a, struct apple9_ra_loc b)
{
   return a.kind == b.kind && a.value == b.value;
}

static inline unsigned
loc_use_count(const struct pcopy_state *st, struct apple9_ra_loc loc)
{
   if (loc.kind == APPLE9_RA_LOC_GPR)
      return st->gpr_use[loc.value];
   if (loc.kind == APPLE9_RA_LOC_MEM)
      return st->mem_use[loc.value];
   return 0;
}

static inline void
loc_use_inc(struct pcopy_state *st, struct apple9_ra_loc loc)
{
   if (loc.kind == APPLE9_RA_LOC_GPR)
      st->gpr_use[loc.value]++;
   else if (loc.kind == APPLE9_RA_LOC_MEM)
      st->mem_use[loc.value]++;
}

static inline void
loc_use_dec(struct pcopy_state *st, struct apple9_ra_loc loc)
{
   if (loc.kind == APPLE9_RA_LOC_GPR) {
      assert(st->gpr_use[loc.value] > 0);
      st->gpr_use[loc.value]--;
   } else if (loc.kind == APPLE9_RA_LOC_MEM) {
      assert(st->mem_use[loc.value] > 0);
      st->mem_use[loc.value]--;
   }
}

static bool
gpr_usable_as_temp(const struct pcopy_state *st, unsigned r, unsigned max_gpr,
                   bool persistent)
{
   if (r >= max_gpr || r >= AGX_APPLE9_GPR_COUNT)
      return false;
   if (st->gpr_use[r] != 0)
      return false;
   if (gpr_bit_test(st->gpr_written, r))
      return false;
   if (persistent && gpr_bit_test(st->gpr_pending_dest, r))
      return false;
   return true;
}

/* Select a safe temporary GPR strictly below `max_gpr`.
 * When `persistent` is false, the temp is consumed immediately within a single
 * 2-op lowered transfer (e.g. MEM<-MEM, MEM<-IMM, GPR(>=64)<-IMM), so even an
 * unblocked pending destination GPR whose old value is dead may be borrowed. */
static unsigned
find_temp_gpr(const struct pcopy_state *st, unsigned max_gpr, bool persistent)
{
   if (st->scratch_gpr != AGX_APPLE9_PHYS_INVALID &&
       gpr_usable_as_temp(st, st->scratch_gpr, max_gpr, persistent))
      return st->scratch_gpr;

   if (st->scratch_gpr2 != AGX_APPLE9_PHYS_INVALID &&
       gpr_usable_as_temp(st, st->scratch_gpr2, max_gpr, persistent))
      return st->scratch_gpr2;

   unsigned free_limit = MIN2(max_gpr, st->max_temp_gpr);
   for (unsigned r = 0; r < free_limit; ++r) {
      if (!gpr_bit_test(st->live_gprs, r) &&
          !gpr_bit_test(st->gpr_pending_dest, r) &&
          gpr_usable_as_temp(st, r, free_limit, persistent))
         return r;
   }

   if (!persistent) {
      for (unsigned r = 0; r < max_gpr && r < AGX_APPLE9_GPR_COUNT; ++r) {
         if (gpr_bit_test(st->gpr_pending_dest, r) &&
             gpr_usable_as_temp(st, r, max_gpr, false))
            return r;
      }
   }

   return AGX_APPLE9_PHYS_INVALID;
}

static bool
emit_mov(struct pcopy_state *st, struct apple9_ra_loc dst,
         struct apple9_ra_loc src)
{
   assert(dst.kind == APPLE9_RA_LOC_GPR || dst.kind == APPLE9_RA_LOC_MEM);
   assert(src.kind == APPLE9_RA_LOC_GPR || src.kind == APPLE9_RA_LOC_MEM);
   assert(!(dst.kind == APPLE9_RA_LOC_MEM && src.kind == APPLE9_RA_LOC_MEM));

   if (loc_equal(dst, src))
      return true;

   struct apple9_ra_pcopy_op op = {
      .kind = APPLE9_RA_PCOPY_MOV,
      .dst = dst,
      .src = src,
   };
   return st->emit(st->emit_ctx, &op);
}

static bool
emit_mov_imm(struct pcopy_state *st, unsigned dst_gpr, uint32_t imm)
{
   assert(dst_gpr < st->mov_imm_max_gpr);
   struct apple9_ra_pcopy_op op = {
      .kind = APPLE9_RA_PCOPY_MOV_IMM,
      .dst = apple9_ra_loc_gpr(dst_gpr),
      .src = apple9_ra_loc_imm(imm),
   };
   return st->emit(st->emit_ctx, &op);
}

static bool
emit_swap(struct pcopy_state *st, unsigned a, unsigned b)
{
   assert(a < AGX_APPLE9_GPR_COUNT && b < AGX_APPLE9_GPR_COUNT && a != b);
   struct apple9_ra_pcopy_op op = {
      .kind = APPLE9_RA_PCOPY_SWAP,
      .dst = apple9_ra_loc_gpr(a),
      .src = apple9_ra_loc_gpr(b),
   };
   return st->emit(st->emit_ctx, &op);
}

static bool
emit_single_copy(struct pcopy_state *st, const struct pcopy_entry *e)
{
   if (e->src.kind == APPLE9_RA_LOC_IMM) {
      if (e->dst.kind == APPLE9_RA_LOC_GPR && e->dst.value < st->mov_imm_max_gpr)
         return emit_mov_imm(st, e->dst.value, e->src.value);

      unsigned tmp = find_temp_gpr(st, st->mov_imm_max_gpr, false);
      if (tmp == AGX_APPLE9_PHYS_INVALID)
         return false;
      return emit_mov_imm(st, tmp, e->src.value) &&
             emit_mov(st, e->dst, apple9_ra_loc_gpr(tmp));
   }

   if (e->dst.kind == APPLE9_RA_LOC_MEM && e->src.kind == APPLE9_RA_LOC_MEM) {
      unsigned tmp = find_temp_gpr(st, AGX_APPLE9_GPR_COUNT, false);
      if (tmp == AGX_APPLE9_PHYS_INVALID)
         return false;
      return emit_mov(st, apple9_ra_loc_gpr(tmp), e->src) &&
             emit_mov(st, e->dst, apple9_ra_loc_gpr(tmp));
   }

   return emit_mov(st, e->dst, e->src);
}

static bool
needs_bridge_temp(const struct pcopy_state *st, const struct pcopy_entry *e)
{
   if (e->dst.kind == APPLE9_RA_LOC_MEM && e->src.kind == APPLE9_RA_LOC_MEM)
      return true;
   if (e->src.kind == APPLE9_RA_LOC_IMM &&
       (e->dst.kind == APPLE9_RA_LOC_MEM || e->dst.value >= st->mov_imm_max_gpr))
      return true;
   return false;
}

static bool
drain_entry(struct pcopy_state *st, unsigned idx)
{
   struct pcopy_entry e = st->pending[idx];
   if (!emit_single_copy(st, &e))
      return false;

   loc_use_dec(st, e.src);
   if (e.dst.kind == APPLE9_RA_LOC_GPR) {
      gpr_bit_clear(st->gpr_pending_dest, e.dst.value);
      gpr_bit_set(st->gpr_written, e.dst.value);
   }

   st->pending[idx] = st->pending[--st->nr_pending];
   return true;
}

/* Drain all acyclic (unblocked) copies. Prioritize MEM<-GPR (frees source
 * GPRs) and bridge copies (may borrow unblocked pending-dest GPRs before they
 * are written) ahead of direct GPR<-* writes. */
static bool
drain_acyclic(struct pcopy_state *st, bool *ok)
{
   bool any = false;
   bool progress = true;
   while (progress) {
      progress = false;

      /* Pass 1: unblocked MEM <- GPR (frees GPRs). */
      for (unsigned i = 0; i < st->nr_pending; ) {
         struct pcopy_entry *e = &st->pending[i];
         if (e->dst.kind == APPLE9_RA_LOC_MEM &&
             e->src.kind == APPLE9_RA_LOC_GPR &&
             loc_use_count(st, e->dst) == 0) {
            if (!drain_entry(st, i)) {
               *ok = false;
               return false;
            }
            progress = any = true;
         } else {
            ++i;
         }
      }

      /* Pass 2: unblocked bridge copies (MEM<-MEM, MEM<-IMM, GPR>=64<-IMM). */
      for (unsigned i = 0; i < st->nr_pending; ) {
         struct pcopy_entry *e = &st->pending[i];
         if (needs_bridge_temp(st, e) && loc_use_count(st, e->dst) == 0) {
            if (!drain_entry(st, i)) {
               *ok = false;
               return false;
            }
            progress = any = true;
         } else {
            ++i;
         }
      }

      /* Pass 3: all other unblocked copies (GPR<-GPR, GPR<-MEM, GPR<64<-IMM). */
      for (unsigned i = 0; i < st->nr_pending; ) {
         struct pcopy_entry *e = &st->pending[i];
         if (loc_use_count(st, e->dst) == 0) {
            if (!drain_entry(st, i)) {
               *ok = false;
               return false;
            }
            progress = any = true;
         } else {
            ++i;
         }
      }
   }

   *ok = true;
   return any;
}

static unsigned
find_entry_by_dst(const struct pcopy_state *st, struct apple9_ra_loc dst)
{
   for (unsigned i = 0; i < st->nr_pending; ++i) {
      if (loc_equal(st->pending[i].dst, dst))
         return i;
   }
   return UINT_MAX;
}

static unsigned
find_entry_by_src(const struct pcopy_state *st, struct apple9_ra_loc src)
{
   for (unsigned i = 0; i < st->nr_pending; ++i) {
      if (loc_equal(st->pending[i].src, src))
         return i;
   }
   return UINT_MAX;
}

static bool
break_one_cycle(struct pcopy_state *st)
{
   assert(st->nr_pending >= 2);

   /* Walk the cycle containing pending[0] to classify it and pick the optimal
    * break edge. Every remaining node has in-degree == out-degree == 1. */
   unsigned len = 0;
   bool has_mem = false;
   unsigned mem_mem_edge = UINT_MAX;
   unsigned gpr_src_edge = UINT_MAX;

   unsigned cur = 0;
   do {
      const struct pcopy_entry *e = &st->pending[cur];
      assert(e->src.kind == APPLE9_RA_LOC_GPR ||
             e->src.kind == APPLE9_RA_LOC_MEM);
      if (e->dst.kind == APPLE9_RA_LOC_MEM || e->src.kind == APPLE9_RA_LOC_MEM)
         has_mem = true;
      if (mem_mem_edge == UINT_MAX &&
          e->dst.kind == APPLE9_RA_LOC_MEM && e->src.kind == APPLE9_RA_LOC_MEM)
         mem_mem_edge = cur;
      if (gpr_src_edge == UINT_MAX && e->src.kind == APPLE9_RA_LOC_GPR)
         gpr_src_edge = cur;

      ++len;
      cur = find_entry_by_dst(st, e->src);
      assert(cur != UINT_MAX && len <= st->nr_pending);
   } while (cur != 0);

   assert(len >= 2);

   if (!has_mem) {
      /* Pure GPR cycle: prefer temp-assisted MOVs when a free/scratch GPR
       * exists (k+1 MOVs vs 3*(k-1) XORs for SWAP), unless caller requests
       * SWAP for 2-cycles or zero temp GPRs are available. */
      unsigned tmp = AGX_APPLE9_PHYS_INVALID;
      if (!(len == 2 && st->prefer_swap_2cycle))
         tmp = find_temp_gpr(st, AGX_APPLE9_GPR_COUNT, true);

      if (tmp != AGX_APPLE9_PHYS_INVALID) {
         struct pcopy_entry *e = &st->pending[0];
         struct apple9_ra_loc saved = e->src;
         if (!emit_mov(st, apple9_ra_loc_gpr(tmp), saved))
            return false;
         loc_use_dec(st, saved);
         e->src = apple9_ra_loc_gpr(tmp);
         loc_use_inc(st, e->src);
         return true;
      }

      /* Zero free temp GPRs: resolve the entire GPR cycle with SWAPs. */
      unsigned anchor_idx = 0;
      for (;;) {
         unsigned d = st->pending[anchor_idx].dst.value;
         unsigned s = st->pending[anchor_idx].src.value;
         assert(d != s);

         if (!emit_swap(st, d, s))
            return false;

         gpr_bit_clear(st->gpr_pending_dest, d);
         gpr_bit_set(st->gpr_written, d);
         loc_use_dec(st, apple9_ra_loc_gpr(s));
         st->pending[anchor_idx] = st->pending[--st->nr_pending];

         unsigned reader = find_entry_by_src(st, apple9_ra_loc_gpr(d));
         assert(reader != UINT_MAX);
         loc_use_dec(st, apple9_ra_loc_gpr(d));

         if (st->pending[reader].dst.value == s) {
            /* 2-cycle partner is also satisfied by this swap. */
            gpr_bit_clear(st->gpr_pending_dest, s);
            gpr_bit_set(st->gpr_written, s);
            st->pending[reader] = st->pending[--st->nr_pending];
            break;
         }

         st->pending[reader].src = apple9_ra_loc_gpr(s);
         loc_use_inc(st, apple9_ra_loc_gpr(s));
         anchor_idx = reader;
      }
      return true;
   }

   /* Cycle involves MEM: break by saving one node into a persistent temp GPR.
    * Prefer breaking a MEM<-MEM edge (turns it into MEM<-GPR, saving a second
    * temp when only one MEM<-MEM edge exists), else break a GPR-sourced edge
    * (cheaper GPR<-GPR save). */
   unsigned break_idx = mem_mem_edge != UINT_MAX   ? mem_mem_edge
                        : gpr_src_edge != UINT_MAX ? gpr_src_edge
                                                   : 0;
   unsigned tmp = find_temp_gpr(st, AGX_APPLE9_GPR_COUNT, true);
   if (tmp == AGX_APPLE9_PHYS_INVALID)
      return false;

   struct pcopy_entry *e = &st->pending[break_idx];
   struct apple9_ra_loc saved = e->src;
   if (!emit_mov(st, apple9_ra_loc_gpr(tmp), saved))
      return false;

   loc_use_dec(st, saved);
   e->src = apple9_ra_loc_gpr(tmp);
   loc_use_inc(st, e->src);
   return true;
}

bool
apple9_ra_lower_parallel_copies(const struct apple9_ra_copy *copies,
                                unsigned num_copies,
                                const struct apple9_ra_pcopy_opts *opts,
                                apple9_ra_pcopy_emit_fn emit, void *ctx)
{
   if (!emit)
      return false;
   if (num_copies == 0)
      return true;
   if (!copies)
      return false;

   /* Pass 0: validate bounds, compute expanded word count and max MEM slot. */
   unsigned total_words = 0;
   bool has_mem = false;
   unsigned max_mem_slot = 0;

   for (unsigned i = 0; i < num_copies; ++i) {
      const struct apple9_ra_copy *c = &copies[i];
      unsigned count = c->count ? c->count : 1;
      if (count > 4)
         return false;

      if (c->dst.kind == APPLE9_RA_LOC_GPR) {
         if (c->dst.value >= AGX_APPLE9_GPR_COUNT ||
             c->dst.value + count > AGX_APPLE9_GPR_COUNT)
            return false;
      } else if (c->dst.kind == APPLE9_RA_LOC_MEM) {
         if (c->dst.value >= APPLE9_RA_MAX_SPILL_SLOTS ||
             c->dst.value + count > APPLE9_RA_MAX_SPILL_SLOTS)
            return false;
         has_mem = true;
         max_mem_slot = MAX2(max_mem_slot, c->dst.value + count - 1);
      } else {
         return false;
      }

      if (c->src.kind == APPLE9_RA_LOC_GPR) {
         if (c->src.value >= AGX_APPLE9_GPR_COUNT ||
             c->src.value + count > AGX_APPLE9_GPR_COUNT)
            return false;
      } else if (c->src.kind == APPLE9_RA_LOC_MEM) {
         if (c->src.value >= APPLE9_RA_MAX_SPILL_SLOTS ||
             c->src.value + count > APPLE9_RA_MAX_SPILL_SLOTS)
            return false;
         has_mem = true;
         max_mem_slot = MAX2(max_mem_slot, c->src.value + count - 1);
      } else if (c->src.kind != APPLE9_RA_LOC_IMM &&
                 c->src.kind != APPLE9_RA_LOC_UNDEF) {
         return false;
      }

      total_words += count;
   }

   struct pcopy_entry pending_stack[PCOPY_STACK_PENDING];
   uint16_t mem_use_stack[PCOPY_STACK_MEM];
   BITSET_WORD mem_seen_stack[BITSET_WORDS(PCOPY_STACK_MEM)];

   struct pcopy_entry *pending_heap = NULL;
   uint16_t *mem_use_heap = NULL;
   BITSET_WORD *mem_seen_heap = NULL;

   struct pcopy_entry *pending = pending_stack;
   if (total_words > PCOPY_STACK_PENDING) {
      pending_heap = malloc(total_words * sizeof(*pending_heap));
      if (!pending_heap)
         return false;
      pending = pending_heap;
   }

   unsigned mem_count = has_mem ? (max_mem_slot + 1) : 0;
   uint16_t *mem_use = NULL;
   BITSET_WORD *mem_seen = NULL;
   if (mem_count > 0) {
      if (mem_count <= PCOPY_STACK_MEM) {
         memset(mem_use_stack, 0, mem_count * sizeof(*mem_use_stack));
         memset(mem_seen_stack, 0,
                BITSET_WORDS(mem_count) * sizeof(*mem_seen_stack));
         mem_use = mem_use_stack;
         mem_seen = mem_seen_stack;
      } else {
         mem_use_heap = calloc(mem_count, sizeof(*mem_use_heap));
         mem_seen_heap = calloc(BITSET_WORDS(mem_count), sizeof(*mem_seen_heap));
         if (!mem_use_heap || !mem_seen_heap) {
            free(pending_heap);
            free(mem_use_heap);
            free(mem_seen_heap);
            return false;
         }
         mem_use = mem_use_heap;
         mem_seen = mem_seen_heap;
      }
   }

   struct pcopy_state st = {
      .pending = pending,
      .nr_pending = 0,
      .mem_use = mem_use,
      .mem_count = mem_count,
      .scratch_gpr = opts ? opts->scratch_gpr : AGX_APPLE9_PHYS_INVALID,
      .scratch_gpr2 = opts ? opts->scratch_gpr2 : AGX_APPLE9_PHYS_INVALID,
      .max_temp_gpr =
         (opts && opts->max_temp_gpr) ? MIN2(opts->max_temp_gpr, AGX_APPLE9_GPR_COUNT)
                                      : AGX_APPLE9_GPR_COUNT,
      .mov_imm_max_gpr =
         (opts && opts->mov_imm_max_gpr)
            ? MIN2(opts->mov_imm_max_gpr, AGX_APPLE9_GPR_COUNT)
            : 64u,
      .prefer_swap_2cycle = opts ? opts->prefer_swap_2cycle : false,
      .emit = emit,
      .emit_ctx = ctx,
   };
   if (opts) {
      st.live_gprs[0] = opts->live_gprs[0];
      st.live_gprs[1] = opts->live_gprs[1];
   }

   /* Expand copies into 32-bit scalar words, checking duplicate destinations
    * and dropping UNDEF / self-copies. */
   uint64_t gpr_seen[2] = {0, 0};
   bool ok = true;

   for (unsigned i = 0; i < num_copies && ok; ++i) {
      const struct apple9_ra_copy *c = &copies[i];
      unsigned count = c->count ? c->count : 1;

      for (unsigned w = 0; w < count; ++w) {
         struct apple9_ra_loc dst = {
            .kind = c->dst.kind,
            .value = c->dst.value + w,
         };

         if (dst.kind == APPLE9_RA_LOC_GPR) {
            if (gpr_bit_test(gpr_seen, dst.value)) {
               ok = false;
               break;
            }
            gpr_bit_set(gpr_seen, dst.value);
         } else {
            if (BITSET_TEST(mem_seen, dst.value)) {
               ok = false;
               break;
            }
            BITSET_SET(mem_seen, dst.value);
         }

         if (c->src.kind == APPLE9_RA_LOC_UNDEF)
            continue;

         struct apple9_ra_loc src = c->src;
         if (src.kind == APPLE9_RA_LOC_GPR || src.kind == APPLE9_RA_LOC_MEM)
            src.value += w;
         else if (src.kind == APPLE9_RA_LOC_IMM && w > 0)
            src.value = 0;

         if (loc_equal(dst, src))
            continue;

         st.pending[st.nr_pending++] = (struct pcopy_entry){
            .dst = dst,
            .src = src,
         };
         loc_use_inc(&st, src);
         if (dst.kind == APPLE9_RA_LOC_GPR)
            gpr_bit_set(st.gpr_pending_dest, dst.value);
      }
   }

   while (ok && st.nr_pending > 0) {
      drain_acyclic(&st, &ok);
      if (!ok || st.nr_pending == 0)
         break;
      ok = break_one_cycle(&st);
   }

   free(pending_heap);
   free(mem_use_heap);
   free(mem_seen_heap);
   return ok;
}

struct agx_apple9_vir_instr *
apple9_ra_block_logical_end_anchor(const struct agx_apple9_block *block)
{
   if (!block)
      return NULL;

   struct agx_apple9_vir_instr *anchor = NULL;
   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_last_instr(block);
        ins; ins = agx_apple9_instr_prev(ins)) {
      /* In NIR-lowered Apple9 VIR, when the then-arm has no skip target,
       * EXEC_MASK_ELSE (and any trailing JMP_EXEC_NONE) are emitted at the
       * tail of the then-arm block. Phi-edge copies for the then-arm must
       * execute under the then-arm mask, strictly BEFORE EXEC_MASK_ELSE.
       * The only case where EXEC_MASK_ELSE opens a block is a synthetic test
       * fixture that places both EXEC_MASK_ELSE and EXEC_MASK_POP in one
       * empty else-arm block. */
      if (ins->op == AGX_APPLE9_VIR_EXEC_MASK_ELSE && !ins->local_control &&
          anchor && anchor->op == AGX_APPLE9_VIR_EXEC_MASK_POP)
         break;
      if (apple9_ra_after_logical_end(ins) || ins->op == AGX_APPLE9_VIR_HALT) {
         anchor = ins;
         continue;
      }
      break;
   }
   return anchor;
}

struct vir_pcopy_emit_ctx {
   struct agx_apple9_vir_program *program;
   struct agx_apple9_block *block;
   struct agx_apple9_vir_instr *before;
};

static bool
emit_pcopy_op_to_vir(void *vctx, const struct apple9_ra_pcopy_op *op)
{
   struct vir_pcopy_emit_ctx *ctx = vctx;
   struct agx_apple9_vir_program *p = ctx->program;
   struct agx_apple9_block *saved_block = p->current_block;
   p->current_block = ctx->block;

   bool ok = false;
   struct agx_apple9_vir_instr *ins = NULL;

   switch (op->kind) {
   case APPLE9_RA_PCOPY_MOV:
      if (op->dst.kind == APPLE9_RA_LOC_GPR &&
          op->src.kind == APPLE9_RA_LOC_GPR) {
         const uint32_t srcs[1] = {op->src.value};
         ok = agx_apple9_vir_emit_side_effect(
            p, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, srcs, 1, 0);
         if (ok) {
            ins = p->instructions[p->instruction_count - 1];
            ins->dest = op->dst.value;
            ins->dest_components = 1;
            ins->target = APPLE9_RA_PHYS_OP_TAG;
         }
      } else if (op->dst.kind == APPLE9_RA_LOC_GPR &&
                 op->src.kind == APPLE9_RA_LOC_MEM) {
         ok = agx_apple9_vir_emit_side_effect(
            p, AGX_APPLE9_VIR_SPILL_LOAD, AGX_APPLE9_ENC_SPILL_LOAD, NULL, 0,
            op->src.value);
         if (ok) {
            ins = p->instructions[p->instruction_count - 1];
            ins->dest = op->dst.value;
            ins->dest_components = 1;
            ins->target = APPLE9_RA_PHYS_OP_TAG;
         }
      } else if (op->dst.kind == APPLE9_RA_LOC_MEM &&
                 op->src.kind == APPLE9_RA_LOC_GPR) {
         const uint32_t srcs[1] = {op->src.value};
         ok = agx_apple9_vir_emit_side_effect(
            p, AGX_APPLE9_VIR_SPILL_STORE, AGX_APPLE9_ENC_SPILL_STORE, srcs, 1,
            op->dst.value);
         if (ok) {
            ins = p->instructions[p->instruction_count - 1];
            ins->target = APPLE9_RA_PHYS_OP_TAG;
         }
      }
      break;

   case APPLE9_RA_PCOPY_MOV_IMM: {
      assert(op->dst.kind == APPLE9_RA_LOC_GPR &&
             op->src.kind == APPLE9_RA_LOC_IMM);
      enum agx_apple9_encoding enc =
         (op->dst.value < 16 && op->src.value < 0x80)
            ? AGX_APPLE9_ENC_MOV_IMM_COMPACT
            : AGX_APPLE9_ENC_MOV_IMM32;
      ok = agx_apple9_vir_emit_side_effect(p, AGX_APPLE9_VIR_IMM, enc, NULL, 0,
                                           op->src.value);
      if (ok) {
         ins = p->instructions[p->instruction_count - 1];
         ins->dest = op->dst.value;
         ins->dest_components = 1;
         ins->target = APPLE9_RA_PHYS_OP_TAG;
      }
      break;
   }

   case APPLE9_RA_PCOPY_SWAP: {
      assert(op->dst.kind == APPLE9_RA_LOC_GPR &&
             op->src.kind == APPLE9_RA_LOC_GPR);
      const uint32_t srcs[2] = {op->dst.value, op->src.value};
      ok = agx_apple9_vir_emit_side_effect(
         p, AGX_APPLE9_VIR_SWAP, AGX_APPLE9_ENC_PSEUDO, srcs, 2, 0);
      if (ok) {
         ins = p->instructions[p->instruction_count - 1];
         ins->target = APPLE9_RA_PHYS_OP_TAG;
      }
      break;
   }
   }

   p->current_block = saved_block;

   if (ok && ins && ctx->before)
      agx_apple9_vir_move_before(p, ins, ctx->before);

   return ok;
}

bool
apple9_ra_emit_parallel_copies_vir(struct agx_apple9_vir_program *program,
                                   struct agx_apple9_block *block,
                                   struct agx_apple9_vir_instr *before,
                                   const struct apple9_ra_copy *copies,
                                   unsigned num_copies,
                                   const struct apple9_ra_pcopy_opts *opts)
{
   if (!program || !block)
      return false;
   if (before && agx_apple9_instr_block(before) != block)
      return false;
   if (num_copies == 0)
      return true;

   struct vir_pcopy_emit_ctx ctx = {
      .program = program,
      .block = block,
      .before = before,
   };
   bool ok = apple9_ra_lower_parallel_copies(copies, num_copies, opts,
                                             emit_pcopy_op_to_vir, &ctx);
   if (ok)
      agx_apple9_vir_reindex(program);
   return ok;
}

static bool
value_to_loc(const struct agx_apple9_vir_program *program,
             const struct apple9_ra_assignment *assign, uint32_t val,
             struct apple9_ra_loc *out)
{
   if (val == AGX_APPLE9_VREG_INVALID) {
      *out = apple9_ra_loc_undef();
      return true;
   }
   if (val >= program->value_count)
      return false;

   if (assign && assign->is_mem && BITSET_TEST(assign->is_mem, val)) {
      if (!assign->mem_slot)
         return false;
      unsigned slot = assign->mem_slot[val];
      if (slot >= APPLE9_RA_MAX_SPILL_SLOTS)
         return false;
      if (assign->max_mem_slots && slot >= assign->max_mem_slots)
         return false;
      *out = apple9_ra_loc_mem(slot);
      return true;
   }

   const uint8_t *gpr_map = (assign && assign->gpr) ? assign->gpr : program->phys;
   if (!gpr_map)
      return false;
   unsigned gpr = gpr_map[val];
   if (gpr >= AGX_APPLE9_GPR_COUNT)
      return false;
   *out = apple9_ra_loc_gpr(gpr);
   return true;
}

static struct apple9_ra_pcopy_opts
default_phi_opts(const struct agx_apple9_vir_program *program,
                 const struct agx_apple9_block *pred,
                 const struct apple9_ra_assignment *assign)
{
   struct apple9_ra_pcopy_opts opts = {
      .live_gprs = {0, 0},
      .scratch_gpr = AGX_APPLE9_PHYS_INVALID,
      .scratch_gpr2 = AGX_APPLE9_PHYS_INVALID,
      .max_temp_gpr = AGX_APPLE9_GPR_COUNT,
      .mov_imm_max_gpr = 64u,
      .prefer_swap_2cycle = false,
   };

   /* Pick reserved scratch registers if available (prefer COPY_REG=1, 3, 4, 0). */
   static const uint8_t preferred_scratch[] = {
      APPLE9_RA_DEFAULT_COPY_REG,
      APPLE9_RA_DEFAULT_SWAP_REG0,
      APPLE9_RA_DEFAULT_SWAP_REG1,
      0u,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(preferred_scratch); ++i) {
      uint8_t r = preferred_scratch[i];
      if (!program->reserved_gprs[r])
         continue;
      if (opts.scratch_gpr == AGX_APPLE9_PHYS_INVALID)
         opts.scratch_gpr = r;
      else if (opts.scratch_gpr2 == AGX_APPLE9_PHYS_INVALID &&
               r != opts.scratch_gpr)
         opts.scratch_gpr2 = r;
   }
   for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r) {
      if (!program->reserved_gprs[r])
         continue;
      if (opts.scratch_gpr == AGX_APPLE9_PHYS_INVALID)
         opts.scratch_gpr = r;
      else if (opts.scratch_gpr2 == AGX_APPLE9_PHYS_INVALID &&
               r != opts.scratch_gpr)
         opts.scratch_gpr2 = r;
   }

   if (pred && pred->live_out && pred->live_words > 0) {
      unsigned max_vals = MIN2(program->value_count,
                               pred->live_words * BITSET_WORDBITS);
      for (unsigned v = 0; v < max_vals; ++v) {
         if (!BITSET_TEST(pred->live_out, v))
            continue;
         if (program->publication && v < program->publication_capacity &&
             program->publication[v])
            continue;
         struct apple9_ra_loc loc;
         if (value_to_loc(program, assign, v, &loc) &&
             loc.kind == APPLE9_RA_LOC_GPR)
            gpr_bit_set(opts.live_gprs, loc.value);
      }
   } else {
      /* Without explicit liveness, conservatively preserve all GPRs. */
      opts.live_gprs[0] = ~0ull;
      opts.live_gprs[1] = BITFIELD64_MASK(AGX_APPLE9_GPR_COUNT - 64);
   }

   return opts;
}

static bool
collect_phi_copies_for_succ(const struct agx_apple9_vir_program *program,
                            const struct agx_apple9_block *pred,
                            const struct agx_apple9_block *succ,
                            const struct apple9_ra_assignment *assign,
                            struct util_dynarray *copies)
{
   unsigned pred_idx = agx_apple9_block_pred_index(succ, pred);
   if (pred_idx == UINT_MAX)
      return false;

   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(succ);
        ins; ins = agx_apple9_instr_next(ins)) {
      if (ins->op == AGX_APPLE9_VIR_ENTRY_REG ||
          (ins->op == AGX_APPLE9_VIR_EXEC_MASK_ELSE && !ins->local_control))
         continue;
      if (ins->op != AGX_APPLE9_VIR_PHI)
         break;

      if (pred_idx >= agx_apple9_vir_nr_srcs(ins))
         return false;

      struct apple9_ra_copy cp = { .count = 1 };
      if (!value_to_loc(program, assign, ins->dest, &cp.dst) ||
          !value_to_loc(program, assign, agx_apple9_vir_src(ins, pred_idx),
                        &cp.src))
         return false;

      if (cp.dst.kind != APPLE9_RA_LOC_GPR && cp.dst.kind != APPLE9_RA_LOC_MEM)
         return false;

      util_dynarray_append(copies, cp);
   }

   return true;
}

bool
apple9_ra_lower_phi_edge(struct agx_apple9_vir_program *program,
                         struct agx_apple9_block *pred,
                         struct agx_apple9_block *succ,
                         const struct apple9_ra_assignment *assign,
                         const struct apple9_ra_pcopy_opts *opts)
{
   if (!program || !pred || !pred->cfg_active)
      return false;

   struct util_dynarray copies;
   util_dynarray_init(&copies, NULL);

   bool ok = true;
   if (succ) {
      ok = collect_phi_copies_for_succ(program, pred, succ, assign, &copies);
   } else {
      for (unsigned s = 0; s < 2 && ok; ++s) {
         if (pred->cfg_succs[s])
            ok = collect_phi_copies_for_succ(program, pred, pred->cfg_succs[s],
                                             assign, &copies);
      }
   }

   unsigned count = util_dynarray_num_elements(&copies, struct apple9_ra_copy);
   if (ok && count > 0) {
      struct apple9_ra_pcopy_opts fallback_opts;
      const struct apple9_ra_pcopy_opts *use_opts = opts;
      if (!use_opts) {
         fallback_opts = default_phi_opts(program, pred, assign);
         use_opts = &fallback_opts;
      }

      struct agx_apple9_vir_instr *anchor =
         apple9_ra_block_logical_end_anchor(pred);
      ok = apple9_ra_emit_parallel_copies_vir(
         program, pred, anchor,
         util_dynarray_begin(&copies), count, use_opts);
   }

   util_dynarray_fini(&copies);
   return ok;
}

bool
apple9_ra_lower_all_phi_edges(struct agx_apple9_vir_program *program,
                              const struct apple9_ra_assignment *assign,
                              const struct apple9_ra_pcopy_opts *opts,
                              bool remove_phis)
{
   if (!program)
      return false;

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      if (!apple9_ra_lower_phi_edge(program, b, NULL, assign, opts))
         return false;
   }

   if (remove_phis) {
      struct util_dynarray to_remove;
      util_dynarray_init(&to_remove, NULL);

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         struct agx_apple9_vir_instr *ins = program->instructions[i];
         if (ins->op == AGX_APPLE9_VIR_PHI)
            util_dynarray_append(&to_remove, ins);
      }

      unsigned nr = util_dynarray_num_elements(
         &to_remove, struct agx_apple9_vir_instr *);
      if (nr > 0) {
         agx_apple9_vir_remove_many(program, util_dynarray_begin(&to_remove),
                                    nr);
      }
      util_dynarray_fini(&to_remove);
   }

   return true;
}
