/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Native Apple9 VIR DSATUR spill-home coalescing.
 *
 * Builds the RA_MEM interference graph directly over agx_apple9_vir_program
 * in 32-bit invocation-frame word units (0..8191 = 32 KiB max), contracts
 * non-interfering same-home spill versions, and runs DSATUR coloring while
 * honoring the 16 B (4-word) returning-device-atomic reservation at frame
 * word 0 when needed.
 */

#include "apple9_ra_spill.h"
#include "util/bitset.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_math.h"
#include "util/u_worklist.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct apple9_ra_spill_interference {
   unsigned n;
   unsigned stride;
   unsigned alloc;
   uint32_t *node;  /* SSA value -> [0..n-1] or UINT32_MAX */
   uint32_t *home;  /* node -> 1-based spill_home ID or 0 */
   uint8_t *width;  /* node -> power-of-two width in 32-bit spill words */
   BITSET_WORD *adj;
};

static inline BITSET_WORD *
neighbors(struct apple9_ra_spill_interference *g, unsigned x)
{
   return g->adj + (size_t)x * g->stride;
}

static inline void
interfere(struct apple9_ra_spill_interference *g, unsigned x, unsigned y)
{
   if (x != y) {
      BITSET_SET(neighbors(g, x), y);
      BITSET_SET(neighbors(g, y), x);
   }
}

static void
clique(struct apple9_ra_spill_interference *g, const BITSET_WORD *live)
{
   unsigned x;
   BITSET_FOREACH_SET(x, live, g->n) {
      BITSET_WORD *row = neighbors(g, x);
      for (unsigned word = 0; word < g->stride; ++word)
         row[word] |= live[word];
      BITSET_CLEAR(row, x);
   }
}

bool
apple9_ra_returns_device_atomic(const struct agx_apple9_vir_program *program)
{
   if (!program)
      return false;
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !I->atomic_discard)
            return true;
      }
   }
   return false;
}

unsigned
apple9_ra_spill_base_words(const struct agx_apple9_vir_program *program)
{
   /* A returning device atomic stores its old value in invocation-frame word
    * 0; spills and private arrays begin one 16-byte granule (4 words) above. */
   return apple9_ra_returns_device_atomic(program) ? 4u : 0u;
}

unsigned
apple9_ra_spill_scratch_bytes(const struct agx_apple9_vir_program *program,
                              unsigned frame_words)
{
   unsigned spill_bytes = frame_words ? ALIGN_POT(frame_words * 4u, 16u) : 0u;
   unsigned priv_size = program ? program->private_size : 0u;

   /* The returning-atomic reservation only protects spills or private arrays. */
   if (!spill_bytes && priv_size && apple9_ra_returns_device_atomic(program))
      spill_bytes = 16u;

   return spill_bytes + ALIGN_POT(priv_size, 16u);
}

void
apple9_ra_spill_coalesce_destroy(struct apple9_ra_spill_interference *g)
{
   ralloc_free(g);
}

struct apple9_ra_spill_interference *
apple9_ra_spill_coalesce_prepare(const struct agx_apple9_vir_program *program,
                                 const struct apple9_ra_spill_state *state)
{
   if (!program || !state || !state->is_mem || !program->blocks ||
       program->value_count == 0)
      return NULL;

   struct agx_apple9_vir_program *mut = (struct agx_apple9_vir_program *)program;
   if (mut->num_cfg_blocks == 0 && !agx_apple9_vir_build_logical_cfg(mut))
      return NULL;

   unsigned num_blocks = mut->num_cfg_blocks;
   if (!num_blocks)
      return NULL;

   unsigned alloc = program->value_count;
   struct agx_apple9_block **cfg_blocks =
      calloc(num_blocks, sizeof(*cfg_blocks));
   if (!cfg_blocks)
      return NULL;

   for (struct agx_apple9_block *b = mut->blocks; b; b = b->next) {
      if (b->cfg_active && b->cfg_index < num_blocks)
         cfg_blocks[b->cfg_index] = b;
   }

   /* Count RA_MEM SSA definitions across active blocks. */
   unsigned n = 0;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->dest < alloc && I->dest_components > 0 &&
             apple9_ra_value_is_mem(state, I->dest))
            ++n;
      }
   }

   /* Keep dense interference matrix bounded; large shaders fall back. */
   if (!n || n > 4096u) {
      free(cfg_blocks);
      return NULL;
   }

   struct apple9_ra_spill_interference *g =
      rzalloc(NULL, struct apple9_ra_spill_interference);
   if (!g) {
      free(cfg_blocks);
      return NULL;
   }

   g->n = n;
   g->alloc = alloc;
   g->stride = BITSET_WORDS(n);
   g->node = ralloc_array(g, uint32_t, alloc);
   g->home = ralloc_array(g, uint32_t, n);
   g->width = ralloc_array(g, uint8_t, n);
   g->adj = rzalloc_array(g, BITSET_WORD, (size_t)n * g->stride);
   if (!g->node || !g->home || !g->width || !g->adj) {
      free(cfg_blocks);
      ralloc_free(g);
      return NULL;
   }

   for (unsigned v = 0; v < alloc; ++v)
      g->node[v] = UINT32_MAX;

   unsigned idx = 0;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->dest < alloc && I->dest_components > 0 &&
             apple9_ra_value_is_mem(state, I->dest)) {
            uint32_t v = I->dest;
            assert(g->node[v] == UINT32_MAX && "broken SSA for RA_MEM value");
            g->node[v] = idx;
            g->home[idx] =
               (state->spill_home && v < state->spill_home_count)
                  ? state->spill_home[v]
                  : 0u;
            unsigned w = MAX2(apple9_ra_value_width(state, v),
                              (unsigned)I->dest_components);
            g->width[idx] = (uint8_t)util_next_power_of_two(w);
            ++idx;
         }
      }
   }
   assert(idx == n);

   /* Compute exact RA_MEM liveness over logical CFG (compact g->n bitsets). */
   BITSET_WORD *live_in = calloc((size_t)num_blocks * g->stride, sizeof(BITSET_WORD));
   BITSET_WORD *live_out = calloc((size_t)num_blocks * g->stride, sizeof(BITSET_WORD));
   BITSET_WORD *edge_live = calloc(g->stride, sizeof(BITSET_WORD));
   if (!live_in || !live_out || !edge_live) {
      free(live_in);
      free(live_out);
      free(edge_live);
      free(cfg_blocks);
      ralloc_free(g);
      return NULL;
   }

   u_worklist worklist;
   u_worklist_init(&worklist, num_blocks, NULL);
   for (unsigned bi = 0; bi < num_blocks; ++bi)
      u_worklist_push_head(&worklist, cfg_blocks[bi], cfg_index);

   while (!u_worklist_is_empty(&worklist)) {
      struct agx_apple9_block *blk =
         u_worklist_pop_head(&worklist, struct agx_apple9_block, cfg_index);
      BITSET_WORD *in = live_in + (size_t)blk->cfg_index * g->stride;
      BITSET_WORD *out = live_out + (size_t)blk->cfg_index * g->stride;
      memcpy(in, out, g->stride * sizeof(BITSET_WORD));

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(blk); I;
           I = agx_apple9_instr_prev(I)) {
         if (I->op == AGX_APPLE9_VIR_PHI)
            continue;
         if (I->dest < alloc && g->node[I->dest] != UINT32_MAX)
            BITSET_CLEAR(in, g->node[I->dest]);
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t sv = agx_apple9_vir_src(I, s);
            if (sv < alloc && g->node[sv] != UINT32_MAX)
               BITSET_SET(in, g->node[sv]);
         }
      }

      for (unsigned p = 0; p < blk->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = blk->cfg_preds[p];
         if (!pred || !pred->cfg_active)
            continue;
         memcpy(edge_live, in, g->stride * sizeof(BITSET_WORD));

         agx_apple9_foreach_phi(blk, phi) {
            if (phi->dest < alloc && g->node[phi->dest] != UINT32_MAX)
               BITSET_CLEAR(edge_live, g->node[phi->dest]);
         }
         agx_apple9_foreach_phi(blk, phi) {
            if (p >= agx_apple9_vir_nr_srcs(phi))
               continue;
            uint32_t sv = agx_apple9_vir_src(phi, p);
            if (sv < alloc && g->node[sv] != UINT32_MAX)
               BITSET_SET(edge_live, g->node[sv]);
         }

         BITSET_WORD *pred_out = live_out + (size_t)pred->cfg_index * g->stride;
         bool changed = false;
         for (unsigned w = 0; w < g->stride; ++w) {
            BITSET_WORD merged = pred_out[w] | edge_live[w];
            if (merged != pred_out[w]) {
               pred_out[w] = merged;
               changed = true;
            }
         }
         if (changed)
            u_worklist_push_tail(&worklist, pred, cfg_index);
      }
   }
   u_worklist_fini(&worklist);

   /* Build exact RA_MEM interference graph. */
   BITSET_WORD *live = edge_live;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *block = cfg_blocks[bi];
      memcpy(live, live_out + (size_t)bi * g->stride,
             g->stride * sizeof(BITSET_WORD));
      clique(g, live);

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_last_instr(block); I;
           I = agx_apple9_instr_prev(I)) {
         /* Phi definitions act in parallel at block entry; their sources
          * already participate in predecessors' live_out. */
         if (I->op == AGX_APPLE9_VIR_PHI)
            continue;
         if (I->dest < alloc && g->node[I->dest] != UINT32_MAX) {
            unsigned dst = g->node[I->dest];
            unsigned v;
            BITSET_FOREACH_SET(v, live, g->n)
               interfere(g, dst, v);
            BITSET_CLEAR(live, dst);
         }
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t sv = agx_apple9_vir_src(I, s);
            if (sv < alloc && g->node[sv] != UINT32_MAX)
               BITSET_SET(live, g->node[sv]);
         }
      }

      /* Even a dead phi writes its destination and must not clobber another
       * RA_MEM value live at block entry. All phi dests are simultaneous. */
      agx_apple9_foreach_phi(block, phi) {
         if (phi->dest < alloc && g->node[phi->dest] != UINT32_MAX)
            BITSET_SET(live, g->node[phi->dest]);
      }
      clique(g, live);
   }

   free(live_in);
   free(live_out);
   free(edge_live);
   free(cfg_blocks);
   return g;
}

static inline bool
slots_overlap(unsigned a, unsigned aw, unsigned b, unsigned bw)
{
   return a < b + bw && b < a + aw;
}

static bool
dsatur_color_groups(struct apple9_ra_spill_interference *g, void *memctx,
                    bool contract_same_home, unsigned base_words,
                    unsigned limit_words, unsigned *group_out,
                    unsigned *color_out, unsigned *high_out)
{
   if (limit_words < base_words)
      return false;

   unsigned *representative = ralloc_array(memctx, unsigned, g->n);
   unsigned count = 0;

   if (contract_same_home) {
      for (unsigned x = 0; x < g->n; ++x) {
         unsigned c;
         for (c = 0; c < count; ++c) {
            unsigned r = representative[c];
            if (!g->home[x] || g->home[x] != g->home[r] ||
                g->width[x] != g->width[r])
               continue;
            bool conflict = false;
            for (unsigned y = 0; y < x; ++y) {
               if (group_out[y] == c && BITSET_TEST(neighbors(g, x), y)) {
                  conflict = true;
                  break;
               }
            }
            if (!conflict)
               break;
         }
         group_out[x] = c;
         if (c == count)
            representative[count++] = x;
      }
   } else {
      for (unsigned x = 0; x < g->n; ++x) {
         group_out[x] = x;
         representative[x] = x;
      }
      count = g->n;
   }

   unsigned stride = BITSET_WORDS(count);
   BITSET_WORD *adj = rzalloc_array(memctx, BITSET_WORD, (size_t)count * stride);
   unsigned *degree = rzalloc_array(memctx, unsigned, count);
   for (unsigned x = 0; x < g->n; ++x) {
      unsigned y;
      BITSET_FOREACH_SET(y, neighbors(g, x), g->n) {
         assert(group_out[x] != group_out[y]);
         BITSET_SET(adj + (size_t)group_out[x] * stride, group_out[y]);
      }
   }
   for (unsigned c = 0; c < count; ++c) {
      unsigned y;
      BITSET_FOREACH_SET(y, adj + (size_t)c * stride, count)
         ++degree[c];
   }

   unsigned span = limit_words - base_words;
   if (span == 0)
      return false;

   unsigned color_stride = BITSET_WORDS(span);
   BITSET_WORD *blocked =
      rzalloc_array(memctx, BITSET_WORD, (size_t)count * color_stride);
   unsigned *saturation = rzalloc_array(memctx, unsigned, count);
   for (unsigned c = 0; c < count; ++c)
      color_out[c] = UINT32_MAX;

   unsigned high = base_words;
   for (unsigned assigned = 0; assigned < count; ++assigned) {
      unsigned best = UINT32_MAX, best_sat = 0;
      for (unsigned c = 0; c < count; ++c) {
         if (color_out[c] != UINT32_MAX)
            continue;
         unsigned sat = saturation[c];
         if (best == UINT32_MAX || sat > best_sat ||
             (sat == best_sat &&
              (degree[c] > degree[best] ||
               (degree[c] == degree[best] &&
                (g->home[representative[c]] < g->home[representative[best]] ||
                 (g->home[representative[c]] == g->home[representative[best]] &&
                  representative[c] < representative[best])))))) {
            best = c;
            best_sat = sat;
         }
      }

      BITSET_WORD *used = blocked + (size_t)best * color_stride;
      unsigned width = g->width[representative[best]];
      unsigned first_aligned = ALIGN_POT(base_words, width) - base_words;
      unsigned rel;
      for (rel = first_aligned; rel + width <= span; rel += width) {
         if (!BITSET_TEST_COUNT(used, rel, width))
            break;
      }
      if (rel + width > span)
         return false;

      unsigned slot = base_words + rel;
      color_out[best] = slot;
      high = MAX2(high, slot + width);

      unsigned y;
      BITSET_FOREACH_SET(y, adj + (size_t)best * stride, count) {
         BITSET_WORD *blocked_y = blocked + (size_t)y * color_stride;
         for (unsigned r = rel; r < rel + width; ++r) {
            if (!BITSET_TEST(blocked_y, r)) {
               BITSET_SET(blocked_y, r);
               ++saturation[y];
            }
         }
      }
   }

   *high_out = high;
   return true;
}

static void
write_mem_slots(const struct apple9_ra_spill_interference *g,
                const struct apple9_ra_spill_state *state,
                const unsigned *group, const unsigned *color,
                uint32_t *mem_slot)
{
   for (unsigned v = 0; v < g->alloc; ++v) {
      if (g->node[v] != UINT32_MAX) {
         unsigned base = color[group[g->node[v]]];
         unsigned w = apple9_ra_value_width(state, v);
         for (unsigned c = 0; c < w && v + c < g->alloc; ++c)
            mem_slot[v + c] = base + c;
      }
   }
}

bool
apple9_ra_spill_coalesce_apply(const struct agx_apple9_vir_program *program,
                               const struct apple9_ra_spill_state *state,
                               struct apple9_ra_spill_interference *g,
                               uint32_t *mem_slot, unsigned *frame_words)
{
   if (!program || !state || !g || !mem_slot || !frame_words)
      return false;

   unsigned base_words = apple9_ra_spill_base_words(program);
   if (*frame_words < base_words)
      return false;

   void *memctx = ralloc_context(NULL);
   unsigned *old = ralloc_array(memctx, unsigned, g->n);
   if (!old) {
      ralloc_free(memctx);
      return false;
   }
   for (unsigned i = 0; i < g->n; ++i)
      old[i] = UINT32_MAX;

   bool success = false;

   /* Validate existing assignment against the interference graph before
    * mutating anything. */
   for (unsigned v = 0; v < g->alloc; ++v) {
      unsigned node = g->node[v];
      if (node == UINT32_MAX)
         continue;
      unsigned slot = mem_slot[v];
      if (slot == UINT32_MAX || slot < base_words ||
          slot + g->width[node] > *frame_words)
         goto out;
      old[node] = slot;
   }

   for (unsigned x = 0; x < g->n; ++x) {
      if (old[x] == UINT32_MAX)
         goto out;
      unsigned y;
      BITSET_FOREACH_SET(y, neighbors(g, x), g->n) {
         if (slots_overlap(old[x], g->width[x], old[y], g->width[y]))
            goto out;
      }
   }

   unsigned *group = ralloc_array(memctx, unsigned, g->n);
   unsigned *color = ralloc_array(memctx, unsigned, g->n);
   if (!group || !color)
      goto out;
   unsigned high = base_words;

   if (!dsatur_color_groups(g, memctx, true, base_words, *frame_words,
                            group, color, &high))
      goto out;

   write_mem_slots(g, state, group, color, mem_slot);
   *frame_words = high;
   success = true;

out:
   ralloc_free(memctx);
   return success;
}

bool
apple9_ra_spill_coalesce_color(const struct agx_apple9_vir_program *program,
                               const struct apple9_ra_spill_state *state,
                               uint32_t *mem_slot, unsigned *frame_words)
{
   if (!program || !mem_slot || !frame_words)
      return false;

   for (unsigned v = 0; v < program->value_count; ++v)
      mem_slot[v] = UINT32_MAX;
   *frame_words = 0;

   unsigned priv_bytes = ALIGN_POT(program->private_size, 16u);
   if (priv_bytes > AGX_APPLE9_MAX_SCRATCH_BYTES)
      return false;
   unsigned max_words = (AGX_APPLE9_MAX_SCRATCH_BYTES - priv_bytes) / 4u;
   unsigned base_words = apple9_ra_spill_base_words(program);
   if (base_words > max_words)
      return false;

   if (!state || !state->is_mem)
      return true;

   struct apple9_ra_spill_interference *g =
      apple9_ra_spill_coalesce_prepare(program, state);
   if (!g) {
      /* Either 0 RA_MEM definitions (done) or >4096 fallback: walk only
       * surviving active-block definitions so DCE'd RA_MEM values stay
       * UINT32_MAX and never inflate the scratch frame. */
      unsigned cursor = base_words;
      bool any_mem = false;
      for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
         if (!b->cfg_active)
            continue;
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b);
              I; I = agx_apple9_instr_next(I)) {
            uint32_t v = I->dest;
            if (v == AGX_APPLE9_VREG_INVALID || I->dest_components == 0 ||
                v >= program->value_count ||
                !apple9_ra_value_is_mem(state, v) ||
                apple9_ra_value_is_imm(state, v) ||
                mem_slot[v] != UINT32_MAX)
               continue;
            any_mem = true;
            unsigned raw_w = MAX2((unsigned)I->dest_components,
                                  apple9_ra_value_width(state, v));
            unsigned w = util_next_power_of_two(raw_w);
            cursor = ALIGN_POT(cursor, w);
            if (cursor + w > max_words)
               return false;
            for (unsigned c = 0; c < raw_w && v + c < program->value_count; ++c)
               mem_slot[v + c] = cursor + c;
            cursor += w;
         }
      }
      *frame_words = any_mem ? cursor : 0u;
      return true;
   }

   void *memctx = ralloc_context(NULL);
   unsigned *group = ralloc_array(memctx, unsigned, g->n);
   unsigned *color = ralloc_array(memctx, unsigned, g->n);
   if (!group || !color) {
      ralloc_free(memctx);
      apple9_ra_spill_coalesce_destroy(g);
      return false;
   }

   /* Compute a tight upper bound on span to keep DSATUR bitsets compact. */
   unsigned worst_span = 0;
   for (unsigned x = 0; x < g->n; ++x)
      worst_span += 2u * g->width[x];
   unsigned limit = base_words + MIN2(max_words - base_words, MAX2(worst_span, 8u));

   unsigned high = base_words;
   bool ok = dsatur_color_groups(g, memctx, true, base_words, limit,
                                 group, color, &high);
   if (!ok) {
      /* Contraction can occasionally increase chromatic number on rare graphs;
       * retry without same-home contraction before failing. */
      ok = dsatur_color_groups(g, memctx, false, base_words, max_words,
                               group, color, &high);
   }

   if (ok) {
      write_mem_slots(g, state, group, color, mem_slot);
      *frame_words = high;
   }

   ralloc_free(memctx);
   apple9_ra_spill_coalesce_destroy(g);
   return ok;
}
