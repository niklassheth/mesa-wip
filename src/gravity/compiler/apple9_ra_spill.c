/*
 * Copyright 2026 The Asahi Linux Contributors
 * Copyright 2023-2024 Alyssa Rosenzweig
 * Copyright 2023-2024 Valve Corporation
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 *
 * Native Apple9 VIR implementation of "Register Spilling and Live-Range
 * Splitting for SSA-Form Programs" by Braun and Hack, operating directly on
 * agx_apple9_vir_program in 32-bit GPR units (0..95) + RA_MEM spill class
 * (0..8191 32-bit words = 32 KiB max).
 */

#include "apple9_ra_spill.h"
#include "gravity/debug.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_qsort.h"
#include "util/u_worklist.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIST_INFINITY UINT32_MAX
typedef uint32_t dist_t;

/* Braun and Hack weight next-use distances that are only reached by leaving a
 * loop with a large constant so values used only after a loop look far away
 * inside it. */
#define LOOP_EXIT_DIST 100000u

static inline dist_t
dist_sum(dist_t a, dist_t b)
{
   return (a + b < a) ? DIST_INFINITY : (a + b);
}

/* -------------------------------------------------------------------------
 * Shared state and VIR helpers
 * ------------------------------------------------------------------------- */

void
apple9_ra_spill_state_init(struct apple9_ra_spill_state *state)
{
   memset(state, 0, sizeof(*state));
}

void
apple9_ra_spill_state_finish(struct apple9_ra_spill_state *state)
{
   if (!state)
      return;
   free(state->is_mem);
   free(state->value_width);
   free(state->spill_home);
   memset(state, 0, sizeof(*state));
}

bool
apple9_ra_spill_state_ensure(struct apple9_ra_spill_state *state,
                             unsigned value_count)
{
   if (!state)
      return true;
   if (value_count <= state->is_mem_capacity)
      return true;

   unsigned new_cap = MAX2(state->is_mem_capacity ? state->is_mem_capacity * 2 : 64u,
                           value_count);
   unsigned old_words = state->is_mem ? BITSET_WORDS(state->is_mem_capacity) : 0;
   unsigned new_words = BITSET_WORDS(new_cap);

   BITSET_WORD *new_mem = realloc(state->is_mem, new_words * sizeof(BITSET_WORD));
   if (!new_mem)
      return false;
   if (new_words > old_words)
      memset(new_mem + old_words, 0, (new_words - old_words) * sizeof(BITSET_WORD));
   state->is_mem = new_mem;

   uint8_t *new_width = realloc(state->value_width, new_cap * sizeof(uint8_t));
   if (!new_width)
      return false;
   memset(new_width + state->is_mem_capacity, 1,
          (new_cap - state->is_mem_capacity) * sizeof(uint8_t));
   state->value_width = new_width;

   state->is_mem_capacity = new_cap;
   return true;
}

bool
apple9_ra_vir_is_publication_instr(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
          ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE;
}

bool
apple9_ra_vir_after_logical_end(const struct agx_apple9_vir_instr *ins)
{
   if (ins->local_control)
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
      return true;
   default:
      return false;
   }
}

bool
apple9_ra_vir_grow_values(struct agx_apple9_vir_program *program,
                          struct apple9_ra_spill_state *state,
                          unsigned extra_count, uint32_t *base_out)
{
   if (!program)
      return false;
   if (extra_count == 0) {
      if (base_out)
         *base_out = program->value_count;
      return true;
   }
   if (program->value_count > UINT32_MAX - extra_count)
      return false;

   unsigned old_count = program->value_count;
   unsigned new_count = old_count + extra_count;

   uint8_t *fixed = realloc(program->fixed_phys, new_count);
   if (!fixed)
      return false;
   program->fixed_phys = fixed;

   uint8_t *maximum = realloc(program->max_phys, new_count);
   if (!maximum)
      return false;
   program->max_phys = maximum;

   memset(&program->fixed_phys[old_count], AGX_APPLE9_PHYS_INVALID, extra_count);
   memset(&program->max_phys[old_count], AGX_APPLE9_PHYS_INVALID, extra_count);
   program->value_count = new_count;

   if (state && !apple9_ra_spill_state_ensure(state, new_count))
      return false;

   if (base_out)
      *base_out = old_count;
   return true;
}

/* -------------------------------------------------------------------------
 * VIR instruction traversal, placement, and operand classification helpers
 * ------------------------------------------------------------------------- */

static bool
vir_has_abi_return(const struct agx_apple9_vir_program *program)
{
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_ABI_RETURN)
            return true;
      }
   }
   return false;
}

static bool
vir_block_is_exit(const struct agx_apple9_block *b)
{
   return b->cfg_active && !b->cfg_succs[0] && !b->cfg_succs[1];
}

static unsigned
vir_num_succs(const struct agx_apple9_block *b)
{
   return (b->cfg_succs[0] != NULL) + (b->cfg_succs[1] != NULL);
}

static BITSET_WORD *
build_pub_bitset(const struct agx_apple9_vir_program *program, void *memctx,
                 unsigned n)
{
   BITSET_WORD *is_pub = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   if (!is_pub)
      return NULL;

   if (program->publication) {
      unsigned cap = MIN2(program->publication_capacity, n);
      for (unsigned v = 0; v < cap; ++v) {
         if (program->publication[v])
            BITSET_SET(is_pub, v);
      }
   }

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (!apple9_ra_vir_is_publication_instr(I) ||
             I->dest == AGX_APPLE9_VREG_INVALID)
            continue;
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->dest + c < n)
               BITSET_SET(is_pub, I->dest + c);
         }
      }
   }
   return is_pub;
}

/* Detect multi-component instructions (other than VIR_SPLIT) whose individual
 * sub-components dest + c (c > 0) are referenced directly without VIR_SPLIT. */
static BITSET_WORD *
build_direct_lanes_bitset(const struct agx_apple9_vir_program *program,
                          const BITSET_WORD *is_pub, void *memctx, unsigned n)
{
   BITSET_WORD *direct = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   uint32_t *base = malloc(MAX2(n, 1u) * sizeof(uint32_t));
   if (!direct || !base) {
      free(base);
      return NULL;
   }
   for (unsigned i = 0; i < MAX2(n, 1u); ++i)
      base[i] = AGX_APPLE9_VREG_INVALID;

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_SPLIT || I->dest == AGX_APPLE9_VREG_INVALID ||
             I->dest_components <= 1 || I->dest >= n || BITSET_TEST(is_pub, I->dest))
            continue;
         for (unsigned c = 0; c < I->dest_components && I->dest + c < n; ++c)
            base[I->dest + c] = I->dest;
      }
   }

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < n && base[v] != AGX_APPLE9_VREG_INVALID && v != base[v])
               BITSET_SET(direct, base[v]);
         }
      }
   }
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      uint32_t v = program->live_out[k];
      if (v < n && base[v] != AGX_APPLE9_VREG_INVALID && v != base[v])
         BITSET_SET(direct, base[v]);
   }
   if (program->output < n && base[program->output] != AGX_APPLE9_VREG_INVALID &&
       program->output != base[program->output])
      BITSET_SET(direct, base[program->output]);

   free(base);
   return direct;
}

/* Return the list of SSA destination values defined by I (excluding
 * publications and dead VIR_SPLIT lanes). Writes up to 8 values into dests[]. */
static unsigned
vir_instr_dests(const struct agx_apple9_vir_instr *I, const BITSET_WORD *is_pub,
                const BITSET_WORD *direct_lanes, unsigned n, uint32_t dests[8])
{
   if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components == 0 ||
       I->dest >= n || BITSET_TEST(is_pub, I->dest))
      return 0;

   if (I->op == AGX_APPLE9_VIR_SPLIT ||
       (I->dest_components > 1 && BITSET_TEST(direct_lanes, I->dest))) {
      unsigned count = 0;
      for (unsigned c = 0; c < I->dest_components && count < 8; ++c) {
         if (I->op == AGX_APPLE9_VIR_SPLIT && (I->dead_dest_mask & (1u << c)))
            continue;
         uint32_t v = I->dest + c;
         if (v < n && !BITSET_TEST(is_pub, v))
            dests[count++] = v;
      }
      return count;
   }

   dests[0] = I->dest;
   return 1;
}

static struct agx_apple9_vir_instr *
vir_block_logical_end_anchor(struct agx_apple9_block *block)
{
   struct agx_apple9_vir_instr *anchor = NULL;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(block); I;
        I = agx_apple9_instr_prev(I)) {
      if (apple9_ra_vir_after_logical_end(I) || I->op == AGX_APPLE9_VIR_HALT)
         anchor = I;
      else
         break;
   }
   return anchor;
}

static struct agx_apple9_vir_instr *
vir_block_after_phis_anchor(struct agx_apple9_block *block)
{
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(block); I;
        I = agx_apple9_instr_next(I)) {
      if (I->op != AGX_APPLE9_VIR_PHI)
         return I;
   }
   return NULL;
}

static struct agx_apple9_vir_instr *
vir_before_function_anchor(struct agx_apple9_vir_program *program)
{
   struct agx_apple9_block *entry = program->blocks;
   if (!entry)
      return NULL;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(entry); I;
        I = agx_apple9_instr_next(I)) {
      if (I->op != AGX_APPLE9_VIR_ENTRY_REG)
         return I;
   }
   return NULL;
}

static void
vir_insert_at(struct agx_apple9_vir_program *program,
              struct agx_apple9_block *block,
              struct agx_apple9_vir_instr *before,
              struct agx_apple9_vir_instr *ins)
{
   if (before)
      agx_apple9_vir_move_before(program, ins, before);
   else
      agx_apple9_vir_move_to_block_end(program, ins, block);
}

static struct agx_apple9_vir_instr *
vir_emit_mov_at(struct agx_apple9_vir_program *program,
                struct agx_apple9_block *block,
                struct agx_apple9_vir_instr *before,
                uint32_t dst, uint32_t src, unsigned width)
{
   struct agx_apple9_block *saved = program->current_block;
   program->current_block = block;
   bool ok = agx_apple9_vir_emit_side_effect(program, AGX_APPLE9_VIR_MOV,
                                             AGX_APPLE9_ENC_PSEUDO, &src, 1, 0);
   program->current_block = saved;
   if (!ok)
      return NULL;

   struct agx_apple9_vir_instr *mov =
      program->instructions[program->instruction_count - 1];
   mov->dest = dst;
   mov->dest_components = width;
   vir_insert_at(program, block, before, mov);
   return mov;
}

/* -------------------------------------------------------------------------
 * Logical-CFG liveness (dense BITSET_WORD per active block)
 * ------------------------------------------------------------------------- */

static bool
vir_compute_liveness(struct agx_apple9_vir_program *program,
                     struct agx_apple9_block **cfg_blocks,
                     const BITSET_WORD *is_pub,
                     const BITSET_WORD *direct_lanes,
                     const struct apple9_ra_spill_state *state,
                     bool skip_mem,
                     bool set_kills)
{
   unsigned n = program->value_count;
   unsigned words = BITSET_WORDS(MAX2(n, 1u));
   unsigned num_blocks = program->num_cfg_blocks;
   if (!num_blocks)
      return true;

   bool has_abi_ret = vir_has_abi_return(program);

   u_worklist worklist;
   u_worklist_init(&worklist, num_blocks, NULL);

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      if (b->live_words < words || !b->live_in || !b->live_out) {
         free(b->live_in);
         free(b->live_out);
         b->live_in = calloc(words, sizeof(BITSET_WORD));
         b->live_out = calloc(words, sizeof(BITSET_WORD));
         if (!b->live_in || !b->live_out) {
            u_worklist_fini(&worklist);
            return false;
         }
         b->live_words = words;
      } else {
         memset(b->live_in, 0, words * sizeof(BITSET_WORD));
         memset(b->live_out, 0, words * sizeof(BITSET_WORD));
      }

      /* Seed exit blocks with program->output (and live_out[] if there is no
       * explicit VIR_ABI_RETURN). */
      if (vir_block_is_exit(b)) {
         if (program->output < n && !BITSET_TEST(is_pub, program->output) &&
             (!skip_mem || !apple9_ra_value_is_mem(state, program->output)))
            BITSET_SET(b->live_out, program->output);
         if (!has_abi_ret) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v < n && !BITSET_TEST(is_pub, v) &&
                   (!skip_mem || !apple9_ra_value_is_mem(state, v)))
                  BITSET_SET(b->live_out, v);
            }
         }
      }

      u_worklist_push_head(&worklist, b, cfg_index);
   }

   BITSET_WORD *edge_live = malloc(words * sizeof(BITSET_WORD));
   if (!edge_live) {
      u_worklist_fini(&worklist);
      return false;
   }

   while (!u_worklist_is_empty(&worklist)) {
      struct agx_apple9_block *blk =
         u_worklist_pop_head(&worklist, struct agx_apple9_block, cfg_index);

      memcpy(blk->live_in, blk->live_out, words * sizeof(BITSET_WORD));

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(blk); I;
           I = agx_apple9_instr_prev(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI) {
            if (set_kills)
               I->kill_mask = 0;
            continue;
         }
         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
            continue;

         uint32_t dests[8];
         unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
         for (unsigned d = 0; d < nd; ++d)
            BITSET_CLEAR(blk->live_in, dests[d]);

         if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v < n && !BITSET_TEST(is_pub, v) &&
                   (!skip_mem || !apple9_ra_value_is_mem(state, v)))
                  BITSET_SET(blk->live_in, v);
            }
            continue;
         }

         if (set_kills)
            I->kill_mask = 0;
         unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= n || BITSET_TEST(is_pub, v))
               continue;
            if (skip_mem && apple9_ra_value_is_mem(state, v))
               continue;
            if (set_kills && s < AGX_APPLE9_MAX_VIR_SRCS)
               agx_apple9_vir_set_src_killed(I, s, !BITSET_TEST(blk->live_in, v));
            BITSET_SET(blk->live_in, v);
         }
      }

      for (unsigned p = 0; p < blk->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = blk->cfg_preds[p];
         if (!pred || !pred->cfg_active)
            continue;

         memcpy(edge_live, blk->live_in, words * sizeof(BITSET_WORD));

         /* Phis logically execute in parallel on the incoming edge. */
         for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (phi->dest < n)
               BITSET_CLEAR(edge_live, phi->dest);
         }
         for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (p >= agx_apple9_vir_nr_srcs(phi))
               continue;
            uint32_t sv = agx_apple9_vir_src(phi, p);
            if (sv < n && !BITSET_TEST(is_pub, sv) &&
                (!skip_mem || !apple9_ra_value_is_mem(state, sv)))
               BITSET_SET(edge_live, sv);
         }

         bool changed = false;
         for (unsigned w = 0; w < words; ++w) {
            BITSET_WORD merged = pred->live_out[w] | edge_live[w];
            if (merged != pred->live_out[w]) {
               pred->live_out[w] = merged;
               changed = true;
            }
         }
         if (changed)
            u_worklist_push_tail(&worklist, pred, cfg_index);
      }
   }

   free(edge_live);
   u_worklist_fini(&worklist);
   return true;
}

unsigned
apple9_ra_spill_calc_gpr_demand(const struct agx_apple9_vir_program *program,
                                const struct apple9_ra_spill_state *state)
{
   if (!program || !program->blocks || program->value_count == 0)
      return 0;

   void *memctx = ralloc_context(NULL);
   unsigned n = program->value_count;

   /* Collect active CFG blocks (fall back to layout blocks if CFG unbuilt). */
   unsigned num_blocks = program->num_cfg_blocks;
   struct agx_apple9_vir_program *mut = (struct agx_apple9_vir_program *)program;
   if (num_blocks == 0) {
      if (!agx_apple9_vir_build_logical_cfg(mut)) {
         ralloc_free(memctx);
         return 0;
      }
      num_blocks = mut->num_cfg_blocks;
   }
   if (!num_blocks) {
      ralloc_free(memctx);
      return 0;
   }

   struct agx_apple9_block **cfg_blocks =
      rzalloc_array(memctx, struct agx_apple9_block *, num_blocks);
   for (struct agx_apple9_block *b = mut->blocks; b; b = b->next) {
      if (b->cfg_active && b->cfg_index < num_blocks)
         cfg_blocks[b->cfg_index] = b;
   }

   BITSET_WORD *is_pub = build_pub_bitset(program, memctx, n);
   BITSET_WORD *direct_lanes = build_direct_lanes_bitset(program, is_pub, memctx, n);
   if (!is_pub || !direct_lanes ||
       !vir_compute_liveness(mut, cfg_blocks, is_pub, direct_lanes, state,
                             true, false)) {
      ralloc_free(memctx);
      return 0;
   }

   uint8_t *widths = rzalloc_array(memctx, uint8_t, n);
   for (unsigned v = 0; v < n; ++v)
      widths[v] = apple9_ra_value_width(state, v);
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->dest < n && I->dest_components > 1 &&
             I->op != AGX_APPLE9_VIR_SPLIT && !BITSET_TEST(direct_lanes, I->dest))
            widths[I->dest] = I->dest_components;
      }
   }

   unsigned words = BITSET_WORDS(MAX2(n, 1u));
   BITSET_WORD *live = rzalloc_array(memctx, BITSET_WORD, words);
   unsigned peak = 0;

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      memcpy(live, b->live_out, words * sizeof(BITSET_WORD));
      unsigned pressure = 0;
      unsigned v;
      BITSET_FOREACH_SET(v, live, n)
         pressure += util_next_power_of_two(widths[v]);
      peak = MAX2(peak, pressure);

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(b); I;
           I = agx_apple9_instr_prev(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI ||
             I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
            continue;

         unsigned defs = 0;
         uint32_t dests[8];
         unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
         for (unsigned d = 0; d < nd; ++d) {
            uint32_t dv = dests[d];
            if (apple9_ra_value_is_mem(state, dv))
               continue;
            unsigned sz = util_next_power_of_two(widths[dv]);
            defs += sz;
            if (BITSET_TEST(live, dv)) {
               BITSET_CLEAR(live, dv);
               pressure -= sz;
            }
         }

         if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t sv = program->live_out[k];
               if (sv < n && !BITSET_TEST(is_pub, sv) &&
                   !apple9_ra_value_is_mem(state, sv) && !BITSET_TEST(live, sv)) {
                  BITSET_SET(live, sv);
                  pressure += util_next_power_of_two(widths[sv]);
               }
            }
         } else {
            unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(I, s);
               if (sv < n && !BITSET_TEST(is_pub, sv) &&
                   !apple9_ra_value_is_mem(state, sv) && !BITSET_TEST(live, sv)) {
                  BITSET_SET(live, sv);
                  pressure += util_next_power_of_two(widths[sv]);
               }
            }
         }

         /* Late-kill: sources and destinations overlap simultaneously. */
         peak = MAX2(peak, pressure + defs);
      }
   }

   ralloc_free(memctx);
   return peak;
}

/* -------------------------------------------------------------------------
 * Next-use map (sparse hash_table_u64, O(live variables))
 * ------------------------------------------------------------------------- */

struct next_uses {
   struct hash_table_u64 *ht;
};

static void
init_next_uses(struct next_uses *nu, void *memctx)
{
   nu->ht = _mesa_hash_table_u64_create(memctx);
}

static void
destroy_next_uses(struct next_uses *nu)
{
   _mesa_hash_table_u64_destroy(nu->ht);
}

static void
clear_next_uses(struct next_uses *nu)
{
   _mesa_hash_table_u64_clear(nu->ht);
}

static void
copy_next_uses(struct next_uses *nu, const struct next_uses *from)
{
   clear_next_uses(nu);
   hash_table_u64_foreach(from->ht, use) {
      _mesa_hash_table_u64_insert(nu->ht, use.key, use.data);
   }
}

static void
set_next_use(struct next_uses *nu, unsigned node, dist_t dist)
{
   if (dist == DIST_INFINITY) {
      _mesa_hash_table_u64_remove(nu->ht, node);
   } else {
      uintptr_t as_ptr = (uintptr_t)(dist + 1u);
      assert(as_ptr != 0);
      _mesa_hash_table_u64_insert(nu->ht, node, (void *)as_ptr);
   }
}

static dist_t
search_next_uses(const struct next_uses *nu, unsigned node)
{
   void *ent = _mesa_hash_table_u64_search(nu->ht, node);
   if (!ent)
      return DIST_INFINITY;
   return (dist_t)((uintptr_t)ent - 1u);
}

#define foreach_next_use(nu, node, dist)                                       \
   hash_table_u64_foreach((nu)->ht, use_)                                      \
      for (uint32_t _terminator = 1, node = (uint32_t)use_.key,                \
                    UNUSED dist = (dist_t)((uintptr_t)use_.data - 1u);         \
           _terminator != 0; _terminator = 0)

static bool
minimum_next_uses(struct next_uses *nu, const struct next_uses *from)
{
   bool progress = false;
   foreach_next_use(from, node, from_dist) {
      dist_t nu_dist = search_next_uses(nu, node);
      if (from_dist < nu_dist) {
         set_next_use(nu, node, from_dist);
         progress = true;
      }
   }
   return progress;
}

/* -------------------------------------------------------------------------
 * Loop nesting analysis on logical CFG
 * ------------------------------------------------------------------------- */

static unsigned *
compute_loop_ends(struct agx_apple9_block **cfg_blocks, unsigned num_blocks,
                  void *memctx)
{
   unsigned *loop_end = ralloc_array(memctx, unsigned, MAX2(num_blocks, 1u));
   for (unsigned i = 0; i < MAX2(num_blocks, 1u); ++i)
      loop_end[i] = UINT_MAX;

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      if (!b->loop_header)
         continue;
      for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = b->cfg_preds[p];
         if (pred && pred->cfg_active && pred->cfg_index >= bi) {
            if (loop_end[bi] == UINT_MAX || pred->cfg_index > loop_end[bi])
               loop_end[bi] = pred->cfg_index;
         }
      }
   }
   return loop_end;
}

static inline bool
loop_contains(const unsigned *loop_end, unsigned header, unsigned blk)
{
   return loop_end[header] != UINT_MAX && header <= blk && blk <= loop_end[header];
}

static unsigned
loops_exited(unsigned num_blocks, const unsigned *loop_end,
             const struct agx_apple9_block *pred,
             const struct agx_apple9_block *succ)
{
   unsigned exits = 0;
   for (unsigned h = 0; h <= pred->cfg_index && h < num_blocks; ++h) {
      if (loop_contains(loop_end, h, pred->cfg_index) &&
          !loop_contains(loop_end, h, succ->cfg_index))
         ++exits;
   }
   return exits;
}

/* -------------------------------------------------------------------------
 * Braun-Hack spill context
 * ------------------------------------------------------------------------- */

struct spill_block {
   uint32_t W_exit[AGX_APPLE9_GPR_COUNT];
   unsigned nW_exit;

   uint32_t W_entry[AGX_APPLE9_GPR_COUNT];
   unsigned nW_entry;

   uint32_t *S_exit;
   unsigned nS_exit;

   uint32_t *S_entry;
   unsigned nS_entry;

   uint32_t cycles;

   struct next_uses next_use_in;
   struct next_uses next_use_out;
};

struct spill_ctx {
   void *memctx;
   struct agx_apple9_vir_program *program;
   struct apple9_ra_spill_state *state;
   const struct apple9_ra_spill_options *opts;
   struct agx_apple9_block **cfg_blocks;
   unsigned num_blocks;
   struct agx_apple9_block *block;

   const BITSET_WORD *is_pub;
   const BITSET_WORD *direct_lanes;
   bool has_abi_ret;

   BITSET_WORD *W;
   unsigned nW;

   dist_t *next_uses;
   uint32_t ip;

   BITSET_WORD *S;

   /* Width in 32-bit GPR units per original SSA value (0..n-1). */
   uint8_t *widths;

   /* Rematerializable defining VIR instruction per original SSA value, or NULL. */
   struct agx_apple9_vir_instr **remat;

   /* Tracks rematerializable values already materialized into RA_MEM at
    * function entry for spilled phi sources. */
   BITSET_WORD *remat_spilled;

   unsigned k;
   unsigned n;
   unsigned spill_base;

   struct spill_block *blocks;
   const unsigned *loop_end;

   unsigned *freq;
   uint32_t *members;

   bool failed;
};

static inline struct spill_block *
spill_block(struct spill_ctx *ctx, const struct agx_apple9_block *block)
{
   return &ctx->blocks[block->cfg_index];
}

static inline unsigned
node_size(const struct spill_ctx *ctx, unsigned node)
{
   return util_next_power_of_two(ctx->widths[node]);
}

/* -------------------------------------------------------------------------
 * Direct on-VIR immutable-uniform pure-ALU rematerialization (<= 4 ops)
 * ------------------------------------------------------------------------- */

static bool
can_remat_leaf(const struct agx_apple9_vir_instr *I, const BITSET_WORD *is_pub,
               unsigned n)
{
   return I->op == AGX_APPLE9_VIR_IMM && I->dest_components == 1 &&
          I->dest < n && !BITSET_TEST(is_pub, I->dest);
}

static bool
apple9_vir_remat_alu(const struct agx_apple9_vir_instr *I,
                     const BITSET_WORD *is_pub, unsigned n)
{
   if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components != 1 ||
       I->dest >= n || BITSET_TEST(is_pub, I->dest) ||
       !agx_apple9_instr_is_pure_alu(I))
      return false;

   switch (I->op) {
   case AGX_APPLE9_VIR_IOR_UNIFORM:
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_ISUB:
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_SELECT:
   case AGX_APPLE9_VIR_BIT_COUNT:
   case AGX_APPLE9_VIR_BIT_REVERSE:
   case AGX_APPLE9_VIR_UFIND_MSB:
   case AGX_APPLE9_VIR_U2F32:
   case AGX_APPLE9_VIR_I2F32:
   case AGX_APPLE9_VIR_F2I32:
   case AGX_APPLE9_VIR_F2U32:
      return true;
   default:
      return false;
   }
}

static unsigned
find_apple9_remat(struct agx_apple9_vir_program *program,
                  struct apple9_ra_spill_state *state,
                  const struct apple9_ra_spill_options *opts,
                  struct agx_apple9_block **cfg_blocks, unsigned num_blocks,
                  const BITSET_WORD *is_pub, const uint8_t *widths,
                  struct agx_apple9_vir_instr **remat, void *memctx,
                  unsigned n, unsigned max_extra)
{
   if (opts->disable_uniform_remat)
      return 0;

   /* Preamble writes make uniforms mutable; treat the whole shader part
    * conservatively. The main shader's uniform file is immutable. */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_STORE_UNIFORM)
            return 0;
      }
   }

   uint8_t *cost = rzalloc_array(memctx, uint8_t, MAX2(n, 1u));
   unsigned *peak = rzalloc_array(memctx, unsigned, MAX2(n, 1u));

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (can_remat_leaf(I, is_pub, n)) {
            cost[I->dest] = 1;
            peak[I->dest] = util_next_power_of_two(widths[I->dest]);
         }
      }
   }

   unsigned extra = 0;
   bool progress;
   do {
      progress = false;
      for (unsigned bi = 0; bi < num_blocks; ++bi) {
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
              I; I = agx_apple9_instr_next(I)) {
            if (!apple9_vir_remat_alu(I, is_pub, n) || cost[I->dest])
               continue;

            unsigned total = 1, held = 0, demand = 0;
            unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(I, s);
               if (sv >= n || BITSET_TEST(is_pub, sv) || !cost[sv]) {
                  total = 5;
                  break;
               }
               total += cost[sv];
               demand = MAX2(demand, held + peak[sv]);
               held += util_next_power_of_two(widths[sv]);
            }

            if (total <= 4) {
               unsigned width = util_next_power_of_two(widths[I->dest]);
               demand = MAX2(demand, held + width);
               if (demand - width > max_extra)
                  continue;
               peak[I->dest] = demand;
               extra = MAX2(extra, demand - width);
               cost[I->dest] = total;
               remat[I->dest] = I;
               state->uniform_remat_used = true;
               progress = true;
            }
         }
      }
   } while (progress);

   return extra;
}

static struct agx_apple9_vir_instr *
remat_to_at(struct spill_ctx *ctx, struct agx_apple9_block *block,
            struct agx_apple9_vir_instr *before, uint32_t dst, unsigned node)
{
   struct agx_apple9_vir_instr *orig = ctx->remat[node];
   assert(orig && (can_remat_leaf(orig, ctx->is_pub, ctx->n) ||
                   apple9_vir_remat_alu(orig, ctx->is_pub, ctx->n)));

   uint32_t new_srcs[AGX_APPLE9_MAX_VIR_SRCS];
   unsigned nr_srcs = agx_apple9_vir_nr_srcs(orig);
   assert(nr_srcs <= AGX_APPLE9_MAX_VIR_SRCS);

   for (unsigned s = 0; s < nr_srcs; ++s) {
      uint32_t tmp = AGX_APPLE9_VREG_INVALID;
      if (!apple9_ra_vir_grow_values(ctx->program, ctx->state, 1, &tmp)) {
         ctx->failed = true;
         return NULL;
      }
      ctx->state->value_width[tmp] = 1;
      if (!remat_to_at(ctx, block, before, tmp, orig->src[s]))
         return NULL;
      new_srcs[s] = tmp;
   }

   struct agx_apple9_block *saved = ctx->program->current_block;
   ctx->program->current_block = block;
   bool ok = agx_apple9_vir_emit_side_effect(ctx->program, orig->op,
                                             orig->encoding, new_srcs, nr_srcs,
                                             orig->immediate);
   ctx->program->current_block = saved;
   if (!ok) {
      ctx->failed = true;
      return NULL;
   }

   struct agx_apple9_vir_instr *copy =
      ctx->program->instructions[ctx->program->instruction_count - 1];
   *copy = *orig;
   copy->dest = dst;
   copy->dest_components = 1;
   copy->phi_block = NULL;
   copy->phi_edge = NULL;
   copy->phi_srcs = NULL;
   copy->phi_src_count = 0;
   copy->completion_consumer = NULL;
   copy->kill_mask = 0;
   copy->live_after_mask = 0;
   copy->nr_srcs = nr_srcs;
   for (unsigned s = 0; s < nr_srcs; ++s)
      copy->src[s] = new_srcs[s];

   vir_insert_at(ctx->program, block, before, copy);
   return copy;
}

static inline void
mark_mem_tuple(struct apple9_ra_spill_state *state, uint32_t base,
               unsigned width)
{
   unsigned w = MAX2(width, 1u);
   for (unsigned c = 0; c < w; ++c) {
      if (base + c < state->is_mem_capacity) {
         BITSET_SET(state->is_mem, base + c);
         state->value_width[base + c] = (c == 0) ? (uint8_t)w : 1u;
      }
   }
}

static void
insert_spill_at(struct spill_ctx *ctx, struct agx_apple9_block *block,
                struct agx_apple9_vir_instr *before, unsigned node)
{
   if (ctx->remat[node])
      return;

   if (ctx->opts->remat_only) {
      ctx->failed = true;
      return;
   }

   uint32_t mem_v = ctx->spill_base + node;
   unsigned width = ctx->widths[node];
   mark_mem_tuple(ctx->state, mem_v, width);
   ctx->state->has_spills = true;

   if (!vir_emit_mov_at(ctx->program, block, before, mem_v, node, width))
      ctx->failed = true;
}

static void
insert_reload_at(struct spill_ctx *ctx, struct agx_apple9_block *block,
                 struct agx_apple9_vir_instr *before, unsigned node)
{
   if (ctx->remat[node]) {
      remat_to_at(ctx, block, before, node, node);
   } else {
      uint32_t mem_v = ctx->spill_base + node;
      unsigned width = ctx->widths[node];
      mark_mem_tuple(ctx->state, mem_v, width);
      if (!vir_emit_mov_at(ctx->program, block, before, node, mem_v, width))
         ctx->failed = true;
   }
}

/* Map a non-critical CFG edge (pred -> succ) to its unique insertion point. */
static void
vir_edge_insertion_point(struct agx_apple9_block *pred,
                         struct agx_apple9_block *succ,
                         struct agx_apple9_block **block_out,
                         struct agx_apple9_vir_instr **before_out)
{
   if (vir_num_succs(pred) == 1) {
      *block_out = pred;
      *before_out = vir_block_logical_end_anchor(pred);
   } else {
      assert(succ->num_cfg_preds == 1 && "critical edge detected");
      *block_out = succ;
      *before_out = vir_block_after_phis_anchor(succ);
   }
}

/* -------------------------------------------------------------------------
 * Register-file set W and eviction (Belady MIN)
 * ------------------------------------------------------------------------- */

static inline void
insert_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(!BITSET_TEST(ctx->W, v));
   BITSET_SET(ctx->W, v);
   ctx->nW += node_size(ctx, v);
}

static inline void
remove_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(BITSET_TEST(ctx->W, v));
   BITSET_CLEAR(ctx->W, v);
   ctx->nW -= node_size(ctx, v);
}

static inline void
remove_W_if_present(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   if (BITSET_TEST(ctx->W, v))
      remove_W(ctx, v);
}

static uint32_t *
bitset_members(struct spill_ctx *ctx, const BITSET_WORD *set, unsigned *count)
{
   unsigned n = 0;
   unsigned i;
   BITSET_FOREACH_SET(i, set, ctx->n)
      ctx->members[n++] = i;

   uint32_t *out = ralloc_array(ctx->memctx, uint32_t, MAX2(n, 1u));
   if (n)
      memcpy(out, ctx->members, n * sizeof(*out));
   *count = n;
   return out;
}

struct candidate {
   unsigned node;
   dist_t dist;
};

static int
cmp_dist(const void *left_, const void *right_, void *ctx_)
{
   struct spill_ctx *ctx = ctx_;
   const struct candidate *left = left_;
   const struct candidate *right = right_;

   /* Rematerializing (with distance > 0) is strictly cheaper than spilling.
    * Within each class, sort ascending by next-use distance (nearest kept). */
   bool remat_left = ctx->remat[left->node] != NULL && left->dist > 0;
   bool remat_right = ctx->remat[right->node] != NULL && right->dist > 0;

   if (remat_left != remat_right)
      return remat_left ? 1 : -1;
   if (left->dist != right->dist)
      return (left->dist > right->dist) ? 1 : -1;
   return (left->node > right->node) - (left->node < right->node);
}

static void
limit_W(struct spill_ctx *ctx, struct agx_apple9_vir_instr *I, unsigned m)
{
   if (ctx->nW <= m)
      return;

   struct candidate candidates[AGX_APPLE9_GPR_COUNT * 2];
   unsigned j = 0;

   unsigned v;
   BITSET_FOREACH_SET(v, ctx->W, ctx->n) {
      assert(j < ARRAY_SIZE(candidates));
      dist_t raw = ctx->next_uses[v];
      dist_t rel = (raw == DIST_INFINITY) ? DIST_INFINITY
                   : (raw >= ctx->ip ? raw - ctx->ip : 0);
      candidates[j++] = (struct candidate){
         .node = v,
         .dist = rel,
      };
   }

   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   unsigned new_weight = 0;
   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);

      if (new_weight + comps <= m) {
         new_weight += comps;
      } else {
         if (!BITSET_TEST(ctx->S, node) && candidates[i].dist < DIST_INFINITY) {
            insert_spill_at(ctx, ctx->block, I, node);
            BITSET_SET(ctx->S, node);
         }
         remove_W(ctx, node);
      }
   }
}

/* -------------------------------------------------------------------------
 * Local next-use precomputation & Belady MIN per block
 * ------------------------------------------------------------------------- */

static void
calculate_local_next_use(struct spill_ctx *ctx, struct util_dynarray *out)
{
   struct spill_block *sb = spill_block(ctx, ctx->block);
   uint32_t ip = sb->cycles;

   *out = UTIL_DYNARRAY_INIT;

   struct next_uses nu;
   init_next_uses(&nu, NULL);

   foreach_next_use(&sb->next_use_out, i, dist) {
      set_next_use(&nu, i, dist_sum(ip, dist));
   }

   for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(ctx->block);
        I; I = agx_apple9_instr_prev(I)) {
      ip -= 1u;

      if (I->op == AGX_APPLE9_VIR_PHI ||
          I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         continue;

      if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
         for (unsigned k = 0; k < ctx->program->live_out_count; ++k) {
            uint32_t v = ctx->program->live_out[k];
            if (v < ctx->n && !BITSET_TEST(ctx->is_pub, v)) {
               util_dynarray_append(out, search_next_uses(&nu, v));
               set_next_use(&nu, v, ip);
            }
         }
         continue;
      }

      unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t v = agx_apple9_vir_src(I, s);
         if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
            continue;
         util_dynarray_append(out, search_next_uses(&nu, v));
         set_next_use(&nu, v, ip);
      }

      uint32_t dests[8];
      unsigned nd = vir_instr_dests(I, ctx->is_pub, ctx->direct_lanes, ctx->n, dests);
      for (int d = (int)nd - 1; d >= 0; --d) {
         util_dynarray_append(out, search_next_uses(&nu, dests[d]));
      }
   }

   assert(ip == 0);
   destroy_next_uses(&nu);
}

static void
min_algorithm(struct spill_ctx *ctx)
{
   struct spill_block *sblock = spill_block(ctx, ctx->block);
   struct util_dynarray local_next_ip;
   calculate_local_next_use(ctx, &local_next_ip);

   foreach_next_use(&sblock->next_use_in, key, dist) {
      assert(key < ctx->n);
      ctx->next_uses[key] = dist;
   }

   dist_t *next_ips = util_dynarray_element(&local_next_ip, dist_t, 0);
   unsigned next_use_cursor =
      util_dynarray_num_elements(&local_next_ip, dist_t);

   struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(ctx->block);
   while (I) {
      /* Save next pointer before inserting reloads/spills around I. */
      struct agx_apple9_vir_instr *next_I = agx_apple9_instr_next(I);
      assert(ctx->nW <= ctx->k);

      if (I->op == AGX_APPLE9_VIR_PHI) {
         if (I->dest < ctx->n && !BITSET_TEST(ctx->is_pub, I->dest) &&
             !BITSET_TEST(ctx->W, I->dest)) {
            uint32_t mem_v = ctx->spill_base + I->dest;
            unsigned width = ctx->widths[I->dest];
            I->dest = mem_v;
            I->dest_components = width;
            mark_mem_tuple(ctx->state, mem_v, width);
            ctx->state->has_spills = true;
         }
         ctx->ip += 1u;
         I = next_I;
         continue;
      }

      if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
         ctx->ip += 1u;
         I = next_I;
         continue;
      }

      if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
         uint32_t R[AGX_APPLE9_MAX_ABI_REGS];
         unsigned nR = 0;

         for (unsigned k = 0; k < ctx->program->live_out_count; ++k) {
            uint32_t v = ctx->program->live_out[k];
            if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
               continue;
            if (BITSET_TEST(ctx->W, v))
               continue;
            assert(BITSET_TEST(ctx->S, v));
            assert(nR < ARRAY_SIZE(R));
            R[nR++] = v;
            insert_W(ctx, v);
         }

         limit_W(ctx, I, ctx->k);

         for (int k = (int)ctx->program->live_out_count - 1; k >= 0; --k) {
            uint32_t v = ctx->program->live_out[k];
            if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
               continue;
            assert(next_use_cursor >= 1);
            dist_t next_ip = next_ips[--next_use_cursor];
            if (next_ip == DIST_INFINITY)
               remove_W_if_present(ctx, v);
            else
               ctx->next_uses[v] = next_ip;
         }

         for (unsigned r = 0; r < nR; ++r)
            insert_reload_at(ctx, ctx->block, I, R[r]);

         ctx->ip += 1u;
         I = next_I;
         continue;
      }

      uint32_t R[AGX_APPLE9_MAX_VIR_SRCS];
      unsigned nR = 0;
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);

      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t node = agx_apple9_vir_src(I, s);
         if (node >= ctx->n || BITSET_TEST(ctx->is_pub, node))
            continue;
         if (BITSET_TEST(ctx->W, node))
            continue;
         assert(BITSET_TEST(ctx->S, node));
         assert(nR < ARRAY_SIZE(R));
         R[nR++] = node;
         insert_W(ctx, node);
      }

      /* Make room for reloaded sources. */
      limit_W(ctx, I, ctx->k);

      uint32_t dests[8];
      unsigned nd = vir_instr_dests(I, ctx->is_pub, ctx->direct_lanes, ctx->n, dests);
      dist_t dest_next_ip[8];
      unsigned dest_size = 0;
      for (unsigned d = 0; d < nd; ++d) {
         assert(next_use_cursor >= 1);
         dest_next_ip[d] = next_ips[--next_use_cursor];
         if (dest_next_ip[d] != DIST_INFINITY)
            ctx->next_uses[dests[d]] = dest_next_ip[d];
         dest_size += node_size(ctx, dests[d]);
      }

      /* Make room for destinations while keeping current sources live. */
      assert(dest_size <= ctx->k);
      limit_W(ctx, I, ctx->k - dest_size);

      for (unsigned d = 0; d < nd; ++d) {
         if (dest_next_ip[d] != DIST_INFINITY)
            insert_W(ctx, dests[d]);
      }

      /* Late-kill source updates. */
      for (int s = (int)nr_srcs - 1; s >= 0; --s) {
         uint32_t v = agx_apple9_vir_src(I, (unsigned)s);
         if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
            continue;
         assert(next_use_cursor >= 1);
         dist_t next_ip = next_ips[--next_use_cursor];
         assert((next_ip == DIST_INFINITY) ==
                agx_apple9_vir_src_killed(I, (unsigned)s));
         if (next_ip == DIST_INFINITY)
            remove_W_if_present(ctx, v);
         else
            ctx->next_uses[v] = next_ip;
      }

      /* Emit reloads before I. */
      for (unsigned r = 0; r < nR; ++r)
         insert_reload_at(ctx, ctx->block, I, R[r]);

      ctx->ip += 1u;
      I = next_I;
   }

   /* If this is an exit block with program->output (or live_out[] without
    * ABI_RETURN), reload any exit value that is currently spilled so it is
    * materialized in a GPR at program exit. */
   if (vir_block_is_exit(ctx->block)) {
      struct agx_apple9_vir_instr *end_anchor =
         vir_block_logical_end_anchor(ctx->block);
      if (ctx->program->output < ctx->n &&
          !BITSET_TEST(ctx->is_pub, ctx->program->output) &&
          !BITSET_TEST(ctx->W, ctx->program->output)) {
         insert_W(ctx, ctx->program->output);
         limit_W(ctx, end_anchor, ctx->k);
         insert_reload_at(ctx, ctx->block, end_anchor, ctx->program->output);
      }
      if (!ctx->has_abi_ret) {
         for (unsigned k = 0; k < ctx->program->live_out_count; ++k) {
            uint32_t v = ctx->program->live_out[k];
            if (v < ctx->n && !BITSET_TEST(ctx->is_pub, v) &&
                !BITSET_TEST(ctx->W, v)) {
               insert_W(ctx, v);
               limit_W(ctx, end_anchor, ctx->k);
               insert_reload_at(ctx, ctx->block, end_anchor, v);
            }
         }
      }
   }

   assert(next_use_cursor == 0);

   unsigned v;
   BITSET_FOREACH_SET(v, ctx->W, ctx->n)
      sblock->W_exit[sblock->nW_exit++] = v;

   sblock->S_exit = bitset_members(ctx, ctx->S, &sblock->nS_exit);
   util_dynarray_fini(&local_next_ip);
}

/* -------------------------------------------------------------------------
 * Block entry sets W_entry / S_entry (Section 4.2 of Braun & Hack)
 * ------------------------------------------------------------------------- */

static unsigned
block_max_pressure(struct spill_ctx *ctx, struct agx_apple9_block *block,
                   BITSET_WORD *live, BITSET_WORD *used)
{
   memset(live, 0, BITSET_BYTES(ctx->n));
   unsigned pressure = 0;
   unsigned i;
   BITSET_FOREACH_SET(i, block->live_out, ctx->n) {
      if (!ctx->remat[i]) {
         BITSET_SET(live, i);
         pressure += node_size(ctx, i);
      }
   }

   unsigned max = pressure;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(block); I;
        I = agx_apple9_instr_prev(I)) {
      if (I->op == AGX_APPLE9_VIR_PHI)
         break;
      if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         continue;

      unsigned defs = 0;
      uint32_t dests[8];
      unsigned nd = vir_instr_dests(I, ctx->is_pub, ctx->direct_lanes, ctx->n, dests);
      for (unsigned d = 0; d < nd; ++d) {
         unsigned v = dests[d];
         defs += node_size(ctx, v);
         if (BITSET_TEST(live, v)) {
            BITSET_CLEAR(live, v);
            pressure -= node_size(ctx, v);
         }
      }

      unsigned remat_srcs = 0;
      if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
         for (unsigned k = 0; k < ctx->program->live_out_count; ++k) {
            uint32_t v = ctx->program->live_out[k];
            if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
               continue;
            BITSET_SET(used, v);
            if (ctx->remat[v]) {
               remat_srcs += node_size(ctx, v);
            } else if (!BITSET_TEST(live, v)) {
               BITSET_SET(live, v);
               pressure += node_size(ctx, v);
            }
         }
      } else {
         unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= ctx->n || BITSET_TEST(ctx->is_pub, v))
               continue;
            BITSET_SET(used, v);
            if (ctx->remat[v]) {
               remat_srcs += node_size(ctx, v);
            } else if (!BITSET_TEST(live, v)) {
               BITSET_SET(live, v);
               pressure += node_size(ctx, v);
            }
         }
      }

      max = MAX2(max, pressure + remat_srcs + defs);
   }
   return max;
}

static bool
in_preheader_w_exit(struct spill_ctx *ctx, struct agx_apple9_block *header,
                    unsigned node)
{
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(header);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      if (phi->dest == node)
         return true;
   }

   for (unsigned p = 0; p < header->num_cfg_preds; ++p) {
      struct agx_apple9_block *pred = header->cfg_preds[p];
      if (!pred || !pred->cfg_active || pred->cfg_index >= header->cfg_index)
         continue;
      struct spill_block *sp = spill_block(ctx, pred);
      bool found = false;
      for (unsigned i = 0; i < sp->nW_exit && !found; ++i)
         found = (sp->W_exit[i] == node);
      if (!found)
         return false;
   }
   return true;
}

static void
compute_w_entry_loop_header(struct spill_ctx *ctx)
{
   struct agx_apple9_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   unsigned nP = 0;
   foreach_next_use(&sb->next_use_in, i, dist) {
      (void)i;
      (void)dist;
      ++nP;
   }
   if (!nP)
      return;

   struct candidate *candidates = calloc(nP, sizeof(*candidates));
   unsigned j = 0;
   foreach_next_use(&sb->next_use_in, i, dist) {
      candidates[j++] = (struct candidate){.node = i, .dist = dist};
   }
   util_qsort_r(candidates, j, sizeof(*candidates), cmp_dist, ctx);

   unsigned through_budget = ctx->k;
   BITSET_WORD *used = NULL;
   unsigned end = ctx->loop_end[block->cfg_index];
   if (end != UINT_MAX) {
      BITSET_WORD *live = malloc(BITSET_BYTES(ctx->n));
      used = calloc(BITSET_WORDS(ctx->n), sizeof(BITSET_WORD));
      unsigned peak = 0;
      for (unsigned bi = block->cfg_index; bi <= end; ++bi) {
         struct agx_apple9_block *b = ctx->cfg_blocks[bi];
         peak = MAX2(peak, block_max_pressure(ctx, b, live, used));
         for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(b);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            unsigned nr_srcs = agx_apple9_vir_nr_srcs(phi);
            for (unsigned s = 0; s < nr_srcs && s < b->num_cfg_preds; ++s) {
               struct agx_apple9_block *pred = b->cfg_preds[s];
               uint32_t sv = agx_apple9_vir_src(phi, s);
               if (pred && pred->cfg_active &&
                   loop_contains(ctx->loop_end, block->cfg_index, pred->cfg_index) &&
                   sv < ctx->n && !BITSET_TEST(ctx->is_pub, sv))
                  BITSET_SET(used, sv);
            }
         }
      }
      free(live);

      unsigned through = 0;
      for (unsigned i = 0; i < j; ++i) {
         unsigned v = candidates[i].node;
         if (!BITSET_TEST(used, v) && !ctx->remat[v])
            through += node_size(ctx, v);
      }
      unsigned own = peak > through ? peak - through : 0;
      through_budget = own < ctx->k ? ctx->k - own : 0;
   }

   unsigned through_taken = 0;
   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);
      bool through = used && !BITSET_TEST(used, node);

      bool fits = (ctx->nW + comps) <= ctx->k;
      if (through) {
         fits &= (through_taken + comps <= through_budget) &&
                 in_preheader_w_exit(ctx, block, node);
      }

      if (fits) {
         insert_W(ctx, node);
         sb->W_entry[sb->nW_entry++] = node;
         if (through)
            through_taken += comps;
      }
   }

   assert(ctx->nW <= ctx->k);
   free(candidates);
   free(used);
}

static void
compute_w_entry(struct spill_ctx *ctx)
{
   struct agx_apple9_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   if (block->num_cfg_preds == 0)
      return;

   if (block->loop_header) {
      compute_w_entry_loop_header(ctx);
      return;
   }

   unsigned *freq = ctx->freq;
   for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
      struct spill_block *sp = spill_block(ctx, block->cfg_preds[p]);
      for (unsigned i = 0; i < sp->nW_exit; ++i)
         freq[sp->W_exit[i]]++;
   }

   struct util_dynarray candidate_array;
   util_dynarray_init(&candidate_array, NULL);

   foreach_next_use(&sb->next_use_in, i, dist) {
      if (freq[i] == block->num_cfg_preds) {
         insert_W(ctx, i);
      } else if (freq[i]) {
         util_dynarray_append(&candidate_array,
                              ((struct candidate){.node = i, .dist = dist}));
      }
   }

   for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
      struct spill_block *sp = spill_block(ctx, block->cfg_preds[p]);
      for (unsigned i = 0; i < sp->nW_exit; ++i)
         freq[sp->W_exit[i]] = 0;
   }

   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(block);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      if (phi->dest >= ctx->n || BITSET_TEST(ctx->is_pub, phi->dest))
         continue;

      bool all_found = true, any_found = false;
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(phi);
      for (unsigned p = 0; p < block->num_cfg_preds && p < nr_srcs; ++p) {
         uint32_t sv = agx_apple9_vir_src(phi, p);
         if (sv == AGX_APPLE9_VREG_INVALID || sv >= ctx->n ||
             BITSET_TEST(ctx->is_pub, sv))
            continue;

         struct spill_block *sp = spill_block(ctx, block->cfg_preds[p]);
         bool found = false;
         for (unsigned i = 0; i < sp->nW_exit; ++i) {
            if (sp->W_exit[i] == sv) {
               found = true;
               break;
            }
         }
         all_found &= found;
         any_found |= found;
      }

      if (!(ctx->opts->spill_partial_phi_sources ? any_found : all_found))
         continue;

      util_dynarray_append(&candidate_array,
                           ((struct candidate){
                              .node = phi->dest,
                              .dist = search_next_uses(&sb->next_use_in, phi->dest),
                           }));
   }

   struct candidate *candidates = candidate_array.data;
   unsigned j = util_dynarray_num_elements(&candidate_array, struct candidate);
   util_qsort_r(candidates, j, sizeof(*candidates), cmp_dist, ctx);

   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);
      if (ctx->nW + comps <= ctx->k) {
         insert_W(ctx, node);
         sb->W_entry[sb->nW_entry++] = node;
      }
   }

   assert(ctx->nW <= ctx->k);
   util_dynarray_fini(&candidate_array);
}

static void
compute_s_entry(struct spill_ctx *ctx)
{
   struct agx_apple9_block *block = ctx->block;
   unsigned v;

   for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
      struct spill_block *sp = spill_block(ctx, block->cfg_preds[p]);
      for (unsigned i = 0; i < sp->nS_exit; ++i) {
         v = sp->S_exit[i];
         if (BITSET_TEST(block->live_in, v))
            BITSET_SET(ctx->S, v);
      }
   }

   BITSET_FOREACH_SET(v, block->live_in, ctx->n) {
      if (!BITSET_TEST(ctx->W, v))
         BITSET_SET(ctx->S, v);
   }

   struct spill_block *sb = spill_block(ctx, block);
   sb->S_entry = bitset_members(ctx, ctx->S, &sb->nS_entry);
}

/* -------------------------------------------------------------------------
 * Block-boundary coupling code
 * ------------------------------------------------------------------------- */

static inline unsigned
chase_mem_index(uint32_t v, unsigned spill_base)
{
   return v >= spill_base ? (v - spill_base) : v;
}

static void
insert_coupling_code(struct spill_ctx *ctx, struct agx_apple9_block *pred,
                     struct agx_apple9_block *succ)
{
   struct spill_block *sp = spill_block(ctx, pred);
   struct spill_block *ss = spill_block(ctx, succ);
   unsigned pred_idx = agx_apple9_block_pred_index(succ, pred);
   assert(pred_idx != UINT_MAX);

   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_SET(ctx->S, sp->S_exit[i]);
   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_SET(ctx->W, sp->W_exit[i]);

   /* Pass 1: Spilled phis (phi->dest in RA_MEM). */
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      if (!apple9_ra_value_is_mem(ctx->state, phi->dest))
         continue;
      if (pred_idx >= agx_apple9_vir_nr_srcs(phi))
         continue;

      uint32_t sv = agx_apple9_vir_src(phi, pred_idx);
      if (sv == AGX_APPLE9_VREG_INVALID || sv >= ctx->n ||
          BITSET_TEST(ctx->is_pub, sv))
         continue;

      if (ctx->remat[sv]) {
         /* Rematerializable phi source entering a spilled phi: materialize
          * into a fresh GPR and spill once at function entry (pressure 0,
          * dominates all predecessors). */
         if (!BITSET_TEST(ctx->remat_spilled, sv)) {
            struct agx_apple9_block *entry = ctx->program->blocks;
            struct agx_apple9_vir_instr *anchor =
               vir_before_function_anchor(ctx->program);
            uint32_t tmp = AGX_APPLE9_VREG_INVALID;
            if (!apple9_ra_vir_grow_values(ctx->program, ctx->state, 1, &tmp)) {
               ctx->failed = true;
               goto cleanup;
            }
            ctx->state->value_width[tmp] = 1;
            remat_to_at(ctx, entry, anchor, tmp, sv);

            uint32_t mem_v = ctx->spill_base + sv;
            mark_mem_tuple(ctx->state, mem_v, ctx->widths[sv]);
            ctx->state->has_spills = true;
            vir_emit_mov_at(ctx->program, entry, anchor, mem_v, tmp,
                            ctx->widths[sv]);
            BITSET_SET(ctx->remat_spilled, sv);
         }
      } else if (!BITSET_TEST(ctx->S, sv)) {
         struct agx_apple9_vir_instr *end_anchor =
            vir_block_logical_end_anchor(pred);
         insert_spill_at(ctx, pred, end_anchor, sv);
      }

      uint32_t mem_sv = ctx->spill_base + sv;
      mark_mem_tuple(ctx->state, mem_sv, ctx->widths[sv]);
      *agx_apple9_vir_src_ptr(phi, pred_idx) = mem_sv;
   }

   /* Pass 2: Values in S_entry of succ must be spilled along the edge. */
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      unsigned orig = chase_mem_index(phi->dest, ctx->spill_base);
      if (orig < ctx->n)
         BITSET_SET(ctx->S, orig);
   }

   for (unsigned i = 0; i < ss->nS_entry; ++i) {
      unsigned v = ss->S_entry[i];
      if (BITSET_TEST(ctx->S, v))
         continue;
      struct agx_apple9_block *ins_blk;
      struct agx_apple9_vir_instr *ins_before;
      vir_edge_insertion_point(pred, succ, &ins_blk, &ins_before);
      insert_spill_at(ctx, ins_blk, ins_before, v);
   }

   /* Restore S to empty, then mark GPR phi dests in S to skip them in Pass 3. */
   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_CLEAR(ctx->S, sp->S_exit[i]);
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      unsigned orig = chase_mem_index(phi->dest, ctx->spill_base);
      if (orig < ctx->n)
         BITSET_CLEAR(ctx->S, orig);
   }
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      if (!apple9_ra_value_is_mem(ctx->state, phi->dest) && phi->dest < ctx->n)
         BITSET_SET(ctx->S, phi->dest);
   }

   /* Pass 3: Values in W_entry of succ must be in GPR along the edge. */
   for (unsigned i = 0; i < ss->nW_entry; ++i) {
      unsigned node = ss->W_entry[i];
      if (BITSET_TEST(ctx->W, node) || BITSET_TEST(ctx->S, node))
         continue;
      struct agx_apple9_block *ins_blk;
      struct agx_apple9_vir_instr *ins_before;
      vir_edge_insertion_point(pred, succ, &ins_blk, &ins_before);
      insert_reload_at(ctx, ins_blk, ins_before, node);
   }

   /* Pass 4: GPR phis must have their incoming source in GPR on the edge. */
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      if (apple9_ra_value_is_mem(ctx->state, phi->dest))
         continue;
      if (pred_idx >= agx_apple9_vir_nr_srcs(phi))
         continue;
      uint32_t sv = agx_apple9_vir_src(phi, pred_idx);
      if (sv == AGX_APPLE9_VREG_INVALID || sv >= ctx->n ||
          BITSET_TEST(ctx->is_pub, sv))
         continue;
      if (!BITSET_TEST(ctx->W, sv)) {
         struct agx_apple9_block *ins_blk;
         struct agx_apple9_vir_instr *ins_before;
         vir_edge_insertion_point(pred, succ, &ins_blk, &ins_before);
         insert_reload_at(ctx, ins_blk, ins_before, sv);
      }
   }

cleanup:
   for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
        phi && phi->op == AGX_APPLE9_VIR_PHI; phi = agx_apple9_instr_next(phi)) {
      unsigned orig = chase_mem_index(phi->dest, ctx->spill_base);
      if (orig < ctx->n)
         BITSET_CLEAR(ctx->S, orig);
   }
   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_CLEAR(ctx->S, sp->S_exit[i]);
   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_CLEAR(ctx->W, sp->W_exit[i]);
}

/* -------------------------------------------------------------------------
 * Global next-use distance dataflow
 * ------------------------------------------------------------------------- */

static void
global_next_use_distances(struct agx_apple9_vir_program *program,
                          struct agx_apple9_block **cfg_blocks,
                          unsigned num_blocks,
                          const BITSET_WORD *is_pub,
                          const BITSET_WORD *direct_lanes,
                          bool has_abi_ret,
                          void *memctx,
                          struct spill_block *blocks,
                          const unsigned *loop_end)
{
   unsigned n = program->value_count;
   u_worklist worklist;
   u_worklist_init(&worklist, num_blocks, NULL);

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      struct spill_block *sb = &blocks[bi];
      init_next_uses(&sb->next_use_in, memctx);
      init_next_uses(&sb->next_use_out, memctx);

      sb->cycles = 0;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I))
         sb->cycles += 1u;

      /* Seed exit blocks identically to vir_compute_liveness(). */
      if (vir_block_is_exit(b)) {
         if (program->output < n && !BITSET_TEST(is_pub, program->output))
            set_next_use(&sb->next_use_out, program->output, 0);
         if (!has_abi_ret) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v < n && !BITSET_TEST(is_pub, v))
                  set_next_use(&sb->next_use_out, v, 0);
            }
         }
      }

      u_worklist_push_head(&worklist, b, cfg_index);
   }

   struct block_facts {
      unsigned *gen_node;
      uint32_t *gen_dist;
      unsigned *kill;
      unsigned gen_count, kill_count;
   } *facts = rzalloc_array(memctx, struct block_facts, MAX2(num_blocks, 1u));

   BITSET_WORD *defined = calloc(BITSET_WORDS(MAX2(n, 1u)), sizeof(BITSET_WORD));
   BITSET_WORD *used = calloc(BITSET_WORDS(MAX2(n, 1u)), sizeof(BITSET_WORD));
   struct util_dynarray gen_node = UTIL_DYNARRAY_INIT;
   struct util_dynarray gen_dist = UTIL_DYNARRAY_INIT;
   struct util_dynarray kill = UTIL_DYNARRAY_INIT;

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *blk = cfg_blocks[bi];
      struct block_facts *f = &facts[bi];
      util_dynarray_clear(&gen_node);
      util_dynarray_clear(&gen_dist);
      util_dynarray_clear(&kill);
      uint32_t cycle = 0;

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk); I;
           I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI ||
             I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
            cycle += 1u;
            continue;
         }

         if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v >= n || BITSET_TEST(is_pub, v) ||
                   BITSET_TEST(defined, v) || BITSET_TEST(used, v))
                  continue;
               BITSET_SET(used, v);
               util_dynarray_append(&gen_node, v);
               util_dynarray_append(&gen_dist, cycle);
            }
            cycle += 1u;
            continue;
         }

         unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= n || BITSET_TEST(is_pub, v) ||
                BITSET_TEST(defined, v) || BITSET_TEST(used, v))
               continue;
            BITSET_SET(used, v);
            util_dynarray_append(&gen_node, v);
            util_dynarray_append(&gen_dist, cycle);
         }

         uint32_t dests[8];
         unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
         for (unsigned d = 0; d < nd; ++d) {
            uint32_t v = dests[d];
            if (!BITSET_TEST(defined, v)) {
               BITSET_SET(defined, v);
               util_dynarray_append(&kill, v);
            }
         }

         cycle += 1u;
      }

      f->gen_count = util_dynarray_num_elements(&gen_node, unsigned);
      f->kill_count = util_dynarray_num_elements(&kill, unsigned);
      f->gen_node = ralloc_array(memctx, unsigned, MAX2(f->gen_count, 1u));
      f->gen_dist = ralloc_array(memctx, uint32_t, MAX2(f->gen_count, 1u));
      f->kill = ralloc_array(memctx, unsigned, MAX2(f->kill_count, 1u));
      if (f->gen_count) {
         memcpy(f->gen_node, gen_node.data, f->gen_count * sizeof(unsigned));
         memcpy(f->gen_dist, gen_dist.data, f->gen_count * sizeof(uint32_t));
      }
      if (f->kill_count)
         memcpy(f->kill, kill.data, f->kill_count * sizeof(unsigned));

      for (unsigned i = 0; i < f->gen_count; ++i)
         BITSET_CLEAR(used, f->gen_node[i]);
      for (unsigned i = 0; i < f->kill_count; ++i)
         BITSET_CLEAR(defined, f->kill[i]);
   }

   free(defined);
   free(used);
   util_dynarray_fini(&gen_node);
   util_dynarray_fini(&gen_dist);
   util_dynarray_fini(&kill);

   struct next_uses dists, far;
   init_next_uses(&dists, NULL);
   init_next_uses(&far, NULL);

   while (!u_worklist_is_empty(&worklist)) {
      struct agx_apple9_block *blk =
         u_worklist_pop_head(&worklist, struct agx_apple9_block, cfg_index);
      struct spill_block *sb = &blocks[blk->cfg_index];
      const struct block_facts *f = &facts[blk->cfg_index];

      clear_next_uses(&dists);
      for (unsigned i = 0; i < f->gen_count; ++i)
         set_next_use(&dists, f->gen_node[i], f->gen_dist[i]);

      foreach_next_use(&sb->next_use_out, node, dist) {
         set_next_use(&sb->next_use_in, node, dist_sum(dist, sb->cycles));
      }
      foreach_next_use(&dists, node, dist) {
         set_next_use(&sb->next_use_in, node, dist);
      }
      for (unsigned i = 0; i < f->kill_count; ++i)
         set_next_use(&sb->next_use_in, f->kill[i], DIST_INFINITY);

      for (unsigned p = 0; p < blk->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = blk->cfg_preds[p];
         if (!pred || !pred->cfg_active)
            continue;
         struct spill_block *sp = &blocks[pred->cfg_index];

         copy_next_uses(&dists, &sb->next_use_in);

         for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (phi->dest < n)
               set_next_use(&dists, phi->dest, DIST_INFINITY);
         }

         unsigned exits = loops_exited(num_blocks, loop_end, pred, blk);
         if (exits) {
            clear_next_uses(&far);
            foreach_next_use(&dists, node, dist) {
               set_next_use(&far, node, dist_sum(dist, exits * LOOP_EXIT_DIST));
            }
            copy_next_uses(&dists, &far);
         }

         for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(blk);
              phi && phi->op == AGX_APPLE9_VIR_PHI;
              phi = agx_apple9_instr_next(phi)) {
            if (p >= agx_apple9_vir_nr_srcs(phi))
               continue;
            uint32_t sv = agx_apple9_vir_src(phi, p);
            if (sv < n && !BITSET_TEST(is_pub, sv))
               set_next_use(&dists, sv, 0);
         }

         if (minimum_next_uses(&sp->next_use_out, &dists))
            u_worklist_push_tail(&worklist, pred, cfg_index);
      }
   }

   u_worklist_fini(&worklist);
   destroy_next_uses(&dists);
   destroy_next_uses(&far);
}

static void
validate_next_use_info(UNUSED struct agx_apple9_block **cfg_blocks,
                       UNUSED unsigned num_blocks, UNUSED unsigned n,
                       UNUSED struct spill_block *blocks)
{
#ifndef NDEBUG
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *blk = cfg_blocks[bi];
      struct spill_block *sb = &blocks[bi];
      unsigned i;
      BITSET_FOREACH_SET(i, blk->live_in, n)
         assert(search_next_uses(&sb->next_use_in, i) < DIST_INFINITY);
      BITSET_FOREACH_SET(i, blk->live_out, n)
         assert(search_next_uses(&sb->next_use_out, i) < DIST_INFINITY);
      foreach_next_use(&sb->next_use_in, node, _)
         assert(BITSET_TEST(blk->live_in, node));
      foreach_next_use(&sb->next_use_out, node, _)
         assert(BITSET_TEST(blk->live_out, node));
   }
#endif
}

/* -------------------------------------------------------------------------
 * Loop-invariant spill hoisting over logical loop_header CFG
 * ------------------------------------------------------------------------- */

static void
hoist_loop_invariant_spills(struct agx_apple9_vir_program *program,
                            struct apple9_ra_spill_state *state,
                            struct agx_apple9_block **cfg_blocks,
                            unsigned num_blocks, unsigned n,
                            const struct spill_block *blocks,
                            const unsigned *loop_end)
{
   struct util_dynarray to_remove = UTIL_DYNARRAY_INIT;

   /* Process inner loops before outer loops so hoisted spills can cascade
    * out of nested loops when invariant across both. */
   for (int hi = (int)num_blocks - 1; hi >= 0; --hi) {
      unsigned h = (unsigned)hi;
      unsigned end = loop_end[h];
      if (end == UINT_MAX)
         continue;

      struct agx_apple9_block *header = cfg_blocks[h];
      struct agx_apple9_block *preheader = NULL;
      unsigned entries = 0;
      for (unsigned p = 0; p < header->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = header->cfg_preds[p];
         if (pred && pred->cfg_active &&
             !loop_contains(loop_end, h, pred->cfg_index)) {
            preheader = pred;
            ++entries;
         }
      }
      if (entries != 1 || !preheader || preheader->cfg_succs[1] != NULL ||
          preheader->cfg_succs[0] != header)
         continue;

      BITSET_WORD *available = calloc(BITSET_WORDS(MAX2(n, 1u)), sizeof(BITSET_WORD));
      BITSET_WORD *stored = calloc(BITSET_WORDS(MAX2(n, 1u)), sizeof(BITSET_WORD));
      BITSET_WORD *header_phis = calloc(BITSET_WORDS(MAX2(n, 1u)), sizeof(BITSET_WORD));
      if (!available || !stored || !header_phis) {
         free(available);
         free(stored);
         free(header_phis);
         break;
      }

      const struct spill_block *sb = &blocks[preheader->cfg_index];
      for (unsigned i = 0; i < sb->nW_exit; ++i)
         BITSET_SET(available, sb->W_exit[i]);
      for (unsigned i = 0; i < sb->nS_exit; ++i)
         BITSET_SET(stored, sb->S_exit[i]);
      for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(header);
           phi && phi->op == AGX_APPLE9_VIR_PHI;
           phi = agx_apple9_instr_next(phi)) {
         unsigned orig = chase_mem_index(phi->dest, n);
         if (orig < n)
            BITSET_SET(header_phis, orig);
      }

      util_dynarray_clear(&to_remove);
      for (unsigned bi = h; bi <= end; ++bi) {
         struct agx_apple9_block *block = cfg_blocks[bi];
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(block);
              I; I = agx_apple9_instr_next(I)) {
            if (I->op != AGX_APPLE9_VIR_MOV ||
                !apple9_ra_value_is_mem(state, I->dest) ||
                apple9_ra_value_is_mem(state, I->src[0]))
               continue;

            unsigned v = I->src[0];
            if (v >= n || I->dest != n + v ||
                BITSET_TEST(header_phis, v) ||
                !BITSET_TEST(header->live_in, v) ||
                (!BITSET_TEST(available, v) && !BITSET_TEST(stored, v)))
               continue;

            if (!BITSET_TEST(stored, v)) {
               struct agx_apple9_vir_instr *anchor =
                  vir_block_logical_end_anchor(preheader);
               vir_emit_mov_at(program, preheader, anchor, I->dest, I->src[0],
                               I->dest_components);
               BITSET_SET(stored, v);
            }
            util_dynarray_append(&to_remove, I);
         }
      }

      unsigned remove_count =
         util_dynarray_num_elements(&to_remove, struct agx_apple9_vir_instr *);
      if (remove_count) {
         agx_apple9_vir_remove_many(program, to_remove.data, remove_count);
      }

      free(available);
      free(stored);
      free(header_phis);
   }

   util_dynarray_fini(&to_remove);
}

/* -------------------------------------------------------------------------
 * Post-spill dead pure-ALU / MOV / PHI cleanup (removes dead remat origins)
 * ------------------------------------------------------------------------- */

static bool
vir_can_dce_post_spill(const struct agx_apple9_vir_instr *I,
                       const BITSET_WORD *is_pub, unsigned n)
{
   if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest >= n ||
       BITSET_TEST(is_pub, I->dest) || I->publication_handoff)
      return false;

   if (I->op == AGX_APPLE9_VIR_MOV || I->op == AGX_APPLE9_VIR_PHI ||
       I->op == AGX_APPLE9_VIR_COLLECT || I->op == AGX_APPLE9_VIR_SPLIT)
      return true;

   return agx_apple9_instr_is_pure_alu(I);
}

static void
vir_post_spill_dce(struct agx_apple9_vir_program *program,
                   const struct apple9_ra_spill_state *state)
{
   unsigned n = program->value_count;
   if (!n)
      return;

   void *memctx = ralloc_context(NULL);
   BITSET_WORD *is_pub = build_pub_bitset(program, memctx, n);
   BITSET_WORD *direct_lanes = build_direct_lanes_bitset(program, is_pub, memctx, n);
   BITSET_WORD *live = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   if (!is_pub || !direct_lanes || !live) {
      ralloc_free(memctx);
      return;
   }

   if (program->output < n)
      BITSET_SET(live, program->output);
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      if (program->live_out[k] < n)
         BITSET_SET(live, program->live_out[k]);
   }

   /* Fixed-point liveness mark over instructions (handles loop phis). */
   bool progress;
   do {
      progress = false;
      for (struct agx_apple9_block *b = program->last_block; b; b = b->prev) {
         if (!b->cfg_active)
            continue;
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(b); I;
              I = agx_apple9_instr_prev(I)) {
            bool needed = !vir_can_dce_post_spill(I, is_pub, n);
            if (!needed) {
               uint32_t dests[8];
               unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
               for (unsigned d = 0; d < nd; ++d) {
                  if (BITSET_TEST(live, dests[d])) {
                     needed = true;
                     break;
                  }
               }
            }
            if (!needed)
               continue;

            unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(I, s);
               if (sv < n && !BITSET_TEST(live, sv)) {
                  BITSET_SET(live, sv);
                  progress = true;
               }
            }
         }
      }
   } while (progress);

   struct util_dynarray to_remove = UTIL_DYNARRAY_INIT;
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (!vir_can_dce_post_spill(I, is_pub, n))
            continue;
         uint32_t dests[8];
         unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
         bool any_live = false;
         for (unsigned d = 0; d < nd; ++d) {
            if (BITSET_TEST(live, dests[d])) {
               any_live = true;
               break;
            }
         }
         if (!any_live)
            util_dynarray_append(&to_remove, I);
      }
   }

   unsigned remove_count =
      util_dynarray_num_elements(&to_remove, struct agx_apple9_vir_instr *);
   if (remove_count)
      agx_apple9_vir_remove_many(program, to_remove.data, remove_count);

   /* Refresh has_spills in case all RA_MEM defs were dead/removed. */
   if (state && state->is_mem) {
      bool any_mem = false;
      for (struct agx_apple9_block *b = program->blocks; b && !any_mem; b = b->next) {
         if (!b->cfg_active)
            continue;
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b);
              I && !any_mem; I = agx_apple9_instr_next(I)) {
            if (I->dest < program->value_count &&
                apple9_ra_value_is_mem(state, I->dest))
               any_mem = true;
         }
      }
      ((struct apple9_ra_spill_state *)state)->has_spills = any_mem;
   }

   util_dynarray_fini(&to_remove);
   ralloc_free(memctx);
}

/* -------------------------------------------------------------------------
 * Entry point: apple9_ra_spill()
 * ------------------------------------------------------------------------- */

bool
apple9_ra_spill(struct agx_apple9_vir_program *program,
                struct apple9_ra_spill_state *state,
                const struct apple9_ra_spill_options *opts_in)
{
   if (!program || !state)
      return false;

   struct apple9_ra_spill_options opts =
      opts_in ? *opts_in : apple9_ra_spill_default_options();
   if (opts.max_gprs == 0)
      opts.max_gprs = AGX_APPLE9_GPR_COUNT;
   if (opts.max_gprs > AGX_APPLE9_GPR_COUNT)
      opts.max_gprs = AGX_APPLE9_GPR_COUNT;

   if (!program->blocks || program->value_count == 0)
      return true;

   /* Ensure logical CFG exists and phis are canonicalized. */
   if (program->num_cfg_blocks == 0 && !agx_apple9_vir_build_logical_cfg(program))
      return false;

   bool has_phi_src = false;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      if (program->instructions[i]->op == AGX_APPLE9_VIR_PHI_SRC) {
         has_phi_src = true;
         break;
      }
   }
   if (has_phi_src && !agx_apple9_vir_canonicalize_phis(program))
      return false;

   unsigned num_blocks = program->num_cfg_blocks;
   if (num_blocks == 0)
      return true;

   unsigned reserved = MAX2(opts.reserved_gprs,
                            opts.remat_only ? 0u : opts.spill_reserved_gprs);
   if (reserved >= opts.max_gprs)
      return false;
   unsigned k = opts.max_gprs - reserved;

   unsigned n = program->value_count;
   if (!apple9_ra_spill_state_ensure(state, n))
      return false;

   void *memctx = ralloc_context(NULL);
   struct agx_apple9_block **cfg_blocks =
      rzalloc_array(memctx, struct agx_apple9_block *, num_blocks);
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->cfg_active && b->cfg_index < num_blocks)
         cfg_blocks[b->cfg_index] = b;
   }

   BITSET_WORD *is_pub = build_pub_bitset(program, memctx, n);
   BITSET_WORD *direct_lanes = build_direct_lanes_bitset(program, is_pub, memctx, n);
   uint8_t *widths = rzalloc_array(memctx, uint8_t, MAX2(n, 1u));
   dist_t *next_uses = rzalloc_array(memctx, dist_t, MAX2(n, 1u));
   struct agx_apple9_vir_instr **remat =
      rzalloc_array(memctx, struct agx_apple9_vir_instr *, MAX2(n, 1u));

   if (!is_pub || !direct_lanes || !widths || !next_uses || !remat) {
      ralloc_free(memctx);
      return false;
   }

   for (unsigned v = 0; v < n; ++v)
      widths[v] = apple9_ra_value_width(state, v);

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (can_remat_leaf(I, is_pub, n))
            remat[I->dest] = I;

         if (I->dest < n && I->dest_components > 1 &&
             !BITSET_TEST(is_pub, I->dest) &&
             I->op != AGX_APPLE9_VIR_SPLIT &&
             !BITSET_TEST(direct_lanes, I->dest)) {
            widths[I->dest] = I->dest_components;
         }
      }
   }

   for (unsigned v = 0; v < n; ++v)
      state->value_width[v] = widths[v];

   /* Compute minimum simultaneous GPRs required by any single instruction so
    * rematerialization scratch temporaries never shrink k below legality. */
   unsigned minimum = 0;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI ||
             I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
            continue;

         unsigned needed = 0;
         uint32_t dests[8];
         unsigned nd = vir_instr_dests(I, is_pub, direct_lanes, n, dests);
         for (unsigned d = 0; d < nd; ++d)
            needed += util_next_power_of_two(widths[dests[d]]);

         if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned j = 0; j < program->live_out_count; ++j) {
               uint32_t sv = program->live_out[j];
               if (sv >= n || BITSET_TEST(is_pub, sv))
                  continue;
               bool dup = false;
               for (unsigned prev = 0; prev < j; ++prev)
                  dup |= (program->live_out[prev] == sv);
               if (!dup)
                  needed += util_next_power_of_two(widths[sv]);
            }
         } else {
            unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(I, s);
               if (sv >= n || BITSET_TEST(is_pub, sv))
                  continue;
               bool dup = false;
               for (unsigned prev = 0; prev < s; ++prev)
                  dup |= (agx_apple9_vir_src(I, prev) == sv);
               if (!dup)
                  needed += util_next_power_of_two(widths[sv]);
            }
         }
         minimum = MAX2(minimum, needed);
      }
   }

   if (k < minimum) {
      ralloc_free(memctx);
      return false;
   }

   unsigned extra_remat = find_apple9_remat(program, state, &opts, cfg_blocks,
                                            num_blocks, is_pub, widths, remat,
                                            memctx, n, k - minimum);
   k -= extra_remat;

   /* Step 1: Fresh liveness + global next-use distances. */
   bool has_abi_ret = vir_has_abi_return(program);
   if (!vir_compute_liveness(program, cfg_blocks, is_pub, direct_lanes, state,
                             false, true)) {
      ralloc_free(memctx);
      return false;
   }

   struct spill_block *blocks =
      rzalloc_array(memctx, struct spill_block, num_blocks);
   unsigned *loop_end = compute_loop_ends(cfg_blocks, num_blocks, memctx);
   global_next_use_distances(program, cfg_blocks, num_blocks, is_pub,
                             direct_lanes, has_abi_ret, memctx, blocks, loop_end);
   validate_next_use_info(cfg_blocks, num_blocks, n, blocks);

   /* Reserve a shadow RA_MEM SSA value (n + v) for every original value v. */
   uint32_t spill_base = 0;
   if (!apple9_ra_vir_grow_values(program, state, n, &spill_base)) {
      ralloc_free(memctx);
      return false;
   }
   assert(spill_base == n);

   BITSET_WORD *W = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   BITSET_WORD *S = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   BITSET_WORD *remat_spilled =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   unsigned *freq = rzalloc_array(memctx, unsigned, MAX2(n, 1u));
   uint32_t *members = ralloc_array(memctx, uint32_t, MAX2(n, 1u));

   bool failed = false;

   /* Step 2: Per-block Braun-Hack MIN pass. */
   for (unsigned bi = 0; bi < num_blocks && !failed; ++bi) {
      struct agx_apple9_block *block = cfg_blocks[bi];
      struct spill_ctx sctx = {
         .memctx = memctx,
         .program = program,
         .state = state,
         .opts = &opts,
         .cfg_blocks = cfg_blocks,
         .num_blocks = num_blocks,
         .block = block,
         .is_pub = is_pub,
         .direct_lanes = direct_lanes,
         .has_abi_ret = has_abi_ret,
         .W = W,
         .S = S,
         .next_uses = next_uses,
         .widths = widths,
         .remat = remat,
         .remat_spilled = remat_spilled,
         .k = k,
         .n = n,
         .spill_base = spill_base,
         .blocks = blocks,
         .loop_end = loop_end,
         .freq = freq,
         .members = members,
      };

      compute_w_entry(&sctx);
      compute_s_entry(&sctx);
      min_algorithm(&sctx);
      failed |= sctx.failed;

      struct spill_block *sb = &blocks[bi];
      for (unsigned i = 0; i < sb->nW_exit; ++i)
         BITSET_CLEAR(W, sb->W_exit[i]);
      for (unsigned i = 0; i < sb->nS_exit; ++i)
         BITSET_CLEAR(S, sb->S_exit[i]);
   }

   /* Step 3: Block-boundary coupling code. */
   for (unsigned bi = 0; bi < num_blocks && !failed; ++bi) {
      struct agx_apple9_block *block = cfg_blocks[bi];
      struct spill_ctx sctx = {
         .memctx = memctx,
         .program = program,
         .state = state,
         .opts = &opts,
         .cfg_blocks = cfg_blocks,
         .num_blocks = num_blocks,
         .block = block,
         .is_pub = is_pub,
         .direct_lanes = direct_lanes,
         .has_abi_ret = has_abi_ret,
         .W = W,
         .S = S,
         .widths = widths,
         .remat = remat,
         .remat_spilled = remat_spilled,
         .k = k,
         .n = n,
         .spill_base = spill_base,
         .blocks = blocks,
         .loop_end = loop_end,
      };

      for (unsigned p = 0; p < block->num_cfg_preds && !sctx.failed; ++p) {
         struct agx_apple9_block *pred = block->cfg_preds[p];
         if (pred && pred->cfg_active)
            insert_coupling_code(&sctx, pred, block);
      }
      failed |= sctx.failed;
   }

   /* Step 4: Loop-invariant spill hoisting. */
   if (!failed && opts.hoist_loop_invariant && state->has_spills) {
      hoist_loop_invariant_spills(program, state, cfg_blocks, num_blocks, n,
                                  blocks, loop_end);
   }

   ralloc_free(memctx);
   if (failed)
      return false;

   /* Step 5: Reconstruct valid SSA via Braun SSA repair. */
   if (!apple9_ra_repair_ssa(program, state, opts.track_spill_homes))
      return false;

   /* Step 6: Prune dead pure ALU / constants left behind by remat. */
   vir_post_spill_dce(program, state);

   return true;
}
