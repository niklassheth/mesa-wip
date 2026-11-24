/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Native Apple9 SSA Register Allocator (Phase 2):
 * Operates directly on struct agx_apple9_vir_program in 32-bit GPR units
 * (0..95, 2 uint64_t bitset words) + RA_MEM spill words (0..8191 = 32 KiB).
 *
 * Pipeline:
 *   1. Logical CFG + canonical variable-arity VIR_PHI
 *   2. Multi-word tuple canonicalization (16-entry intra-block reuse cache +
 *      retain_mask) + AGX_APPLE9_OPERAND_CLOBBER copy insertion
 *   3. Pre-RA DCE + optional SSA validation + pressure/latency DAG scheduling
 *   4. Demand check -> Braun-Hack spilling + SSA repair + DSATUR spill coloring
 *   5. Sparse-bitset CFG liveness (live_in / live_out / kill_mask)
 *   6. Dominance-order 32-bit GPR SSA assignment (phi_web, scarcity_floor,
 *      on-demand encoding constraints, reuse_killed, aligned tuples, acyclic
 *      SSA live-range splitting & constrained relocation)
 *   7. Parallel-copy phi-edge lowering + optional post-RA register-file check
 *   8. Physical VIR lowering (IOR copies, 3-IXOR swaps, SPILL_LOAD/SPILL_STORE
 *      & high-offset PRIVATE_LOAD/PRIVATE_STORE via COPY_REG=1 /
 *      SPILL_INDEX_REG=2, skipping COPY_REG on killed SAVE, ABI_RETURN exports,
 *      empty-else cleanup, and physical scoreboard finalization)
 */

#include "apple9_ra_assign.h"
#include "apple9_ra_pcopy.h"
#include "apple9_ra_sched.h"
#include "apple9_ra_spill.h"
#include "agx_apple9_profile.h"
#include "agx_compile.h"

#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Canonical physical VIR constants (matching agx_apple9_ra.c). */
#define A9_PUB_BASE        AGX_APPLE9_GPR_COUNT
#define A9_COPY_REG        1u
#define A9_SPILL_INDEX_REG 2u
#define A9_SPILL_TAG       6u

/* Window widths 1..4 supported by precomputed scarcity_floor[w][r]. */
#define A9_SCARCITY_MAX_WIDTH 4u

/* =========================================================================
 * 96-bit GPR bitset helpers (2 uint64_t words, zero heap allocations)
 * ========================================================================= */

static inline bool
gpr_test(const uint64_t m[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   return ((m[r >> 6] >> (r & 63u)) & 1ull) != 0;
}

static inline void
gpr_set(uint64_t m[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   m[r >> 6] |= 1ull << (r & 63u);
}

static inline void
gpr_clear(uint64_t m[2], unsigned r)
{
   assert(r < AGX_APPLE9_GPR_COUNT);
   m[r >> 6] &= ~(1ull << (r & 63u));
}

static inline void
gpr_set_range(uint64_t m[2], unsigned r, unsigned count)
{
   for (unsigned i = 0; i < count; ++i)
      gpr_set(m, r + i);
}

static inline bool
gpr_test_range(const uint64_t m[2], unsigned r, unsigned count)
{
   for (unsigned i = 0; i < count; ++i) {
      if (gpr_test(m, r + i))
         return true;
   }
   return false;
}

/* =========================================================================
 * Apple9 VIR classification & on-demand encoding constraints (32-bit GPRs)
 * ========================================================================= */

struct a9_gpr_constraint {
   uint8_t min;
   uint8_t max;
   uint8_t align;
   bool clobber;
   bool reuse_killed;
};

static inline bool
a9_is_block_entry_prefix(const struct agx_apple9_vir_instr *ins)
{
   return ins->op == AGX_APPLE9_VIR_ENTRY_REG ||
          (ins->op == AGX_APPLE9_VIR_EXEC_MASK_ELSE && !ins->local_control);
}

static inline bool
a9_is_pub_val(const struct agx_apple9_vir_program *p, uint32_t v)
{
   return p->publication && v < p->publication_capacity && p->publication[v];
}

static struct a9_gpr_constraint
a9_dest_constraint(const struct agx_apple9_vir_program *program,
                   const struct agx_apple9_vir_instr *ins)
{
   struct apple9_ra_operand_group g = apple9_ra_dest_group(ins);
   struct a9_gpr_constraint c = {
      .min = g.min_gpr,
      .max = g.max_gpr,
      .align = g.align_gprs,
      .clobber = false,
      .reuse_killed = g.reuse_killed,
   };

   if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
       ins->encoding < AGX_APPLE9_ENC_COUNT &&
       ins->dest < program->value_count && program->fixed_phys &&
       program->fixed_phys[ins->dest] != AGX_APPLE9_PHYS_INVALID &&
       !a9_is_pub_val(program, ins->dest)) {
      c.min = c.max = program->fixed_phys[ins->dest];
   }
   return c;
}

static struct a9_gpr_constraint
a9_src_constraint(const struct agx_apple9_vir_program *program,
                  const struct agx_apple9_vir_instr *ins, unsigned source)
{
   struct apple9_ra_operand_group g = apple9_ra_source_group(ins, source);
   struct a9_gpr_constraint c = {
      .min = g.min_gpr,
      .max = g.max_gpr,
      .align = g.align_gprs,
      .clobber = g.clobber,
      .reuse_killed = false,
   };

   if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
       ins->encoding < AGX_APPLE9_ENC_COUNT &&
       source < ins->nr_srcs && ins->src[source] < program->value_count &&
       program->fixed_phys &&
       program->fixed_phys[ins->src[source]] != AGX_APPLE9_PHYS_INVALID &&
       !a9_is_pub_val(program, ins->src[source])) {
      c.min = c.max = program->fixed_phys[ins->src[source]];
   }
   return c;
}

static bool
a9_constraint_window(struct a9_gpr_constraint c, unsigned width,
                     unsigned reserved, unsigned bound,
                     unsigned *first, unsigned *last)
{
   if (width == 0 || !util_is_power_of_two_nonzero(c.align) ||
       c.min > c.max || width > bound)
      return false;

   /* Fixed register inside reserved prefix is allowed only for pre-existing
    * fixed inputs; ordinary allocatable windows start at reserved. */
   unsigned lo = MAX2((unsigned)c.min, reserved);
   unsigned f = ALIGN_POT(lo, (unsigned)c.align);
   unsigned l = MIN2((unsigned)c.max, bound - width);
   *first = f;
   *last = l;
   return f <= l;
}

static bool
a9_validate_constraint_windows(const struct agx_apple9_vir_program *program,
                               const struct apple9_ra_spill_state *spill,
                               unsigned reserved, unsigned bound,
                               const char **error)
{
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
             ins->op == AGX_APPLE9_VIR_ENTRY_REG)
            continue;

         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0 &&
             !a9_is_pub_val(program, ins->dest) &&
             !apple9_ra_value_is_mem(spill, ins->dest)) {
            struct a9_gpr_constraint dc = a9_dest_constraint(program, ins);
            unsigned first, last;
            if (dc.min == dc.max && dc.max < reserved) {
               if (dc.max + ins->dest_components > bound ||
                   (dc.min % dc.align) != 0) {
                  *error = "Fixed destination register violates alignment/bounds";
                  return false;
               }
            } else if (!a9_constraint_window(dc, ins->dest_components, reserved,
                                             bound, &first, &last)) {
               *error = "Destination operand has no aligned allocatable GPR window";
               return false;
            }
         }

         for (unsigned s = 0; s < ins->nr_srcs; ) {
            unsigned w = apple9_ra_source_width(ins, s);
            if (s + w > ins->nr_srcs) {
               *error = "Source tuple width exceeds instruction nr_srcs";
               return false;
            }
            uint32_t sv = ins->src[s];
            if (!a9_is_pub_val(program, sv) &&
                !apple9_ra_value_is_mem(spill, sv)) {
               struct a9_gpr_constraint sc =
                  a9_src_constraint(program, ins, s);
               unsigned first, last;
               if (sc.min == sc.max && sc.max < reserved) {
                  if (sc.max + w > bound || (sc.min % sc.align) != 0) {
                     *error = "Fixed source register violates alignment/bounds";
                     return false;
                  }
               } else if (!a9_constraint_window(sc, w, reserved, bound, &first,
                                                &last)) {
                  *error = "Source operand has no aligned allocatable GPR window";
                  return false;
               }
            }
            s += w;
         }
      }
   }
   return true;
}

/* =========================================================================
 * Pre-RA preparation: tuple canonicalization + 16-entry cache + clobbers
 * ========================================================================= */

static bool
a9_ensure_program_pub_array(struct agx_apple9_vir_program *program,
                            const char **error)
{
   bool has_pub = false;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      if (apple9_ra_is_publication_instr(program->instructions[i])) {
         has_pub = true;
         break;
      }
   }

   if (!program->publication) {
      if (has_pub) {
         if (!agx_apple9_allocate_publications(program, error))
            return false;
      } else {
         unsigned cap = MAX2(program->value_count, 1u);
         program->publication = calloc(cap, sizeof(bool));
         if (!program->publication) {
            *error = "out of memory allocating publication table";
            return false;
         }
         program->publication_capacity = cap;
      }
   } else if (program->publication_capacity < program->value_count) {
      unsigned old_cap = program->publication_capacity;
      unsigned new_cap = MAX2(program->value_count, old_cap * 2u);
      bool *pub = realloc(program->publication, new_cap * sizeof(bool));
      if (!pub) {
         *error = "out of memory growing publication table";
         return false;
      }
      memset(pub + old_cap, 0, (new_cap - old_cap) * sizeof(bool));
      program->publication = pub;
      program->publication_capacity = new_cap;
   }
   return true;
}

static void
a9_strip_atomic_results(struct agx_apple9_vir_program *program)
{
   struct util_dynarray to_remove;
   util_dynarray_init(&to_remove, NULL);

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         util_dynarray_append(&to_remove, ins);
   }

   unsigned count = util_dynarray_num_elements(
      &to_remove, struct agx_apple9_vir_instr *);
   if (count > 0)
      agx_apple9_vir_remove_many(program, util_dynarray_begin(&to_remove), count);
   util_dynarray_fini(&to_remove);
}

static bool
a9_is_identity_ior(const struct agx_apple9_vir_program *program,
                   const struct agx_apple9_vir_instr *ins)
{
   if (ins->op != AGX_APPLE9_VIR_IOR ||
       ins->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED ||
       ins->nr_srcs != 2 || ins->src[0] != ins->src[1] ||
       ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components != 1 ||
       ins->publication_handoff || ins->alu_src_uniform_mask ||
       ins->alu_src_immediate_mask || ins->src_neg_mask || ins->src_abs_mask)
      return false;
   uint32_t d = ins->dest;
   uint32_t s = ins->src[0];
   if (d >= program->value_count || s >= program->value_count || d == s)
      return false;
   if (a9_is_pub_val(program, d) || a9_is_pub_val(program, s))
      return false;
   if (program->fixed_phys &&
       program->fixed_phys[d] != AGX_APPLE9_PHYS_INVALID)
      return false;
   return true;
}

/* Fold pre-RA non-publication-handoff identity copies (`IOR d, s, s`) away
 * before register allocation, matching agx_apple9_ra.c's value aliasing. */
static bool
a9_eliminate_identity_copies(struct agx_apple9_vir_program *program)
{
   unsigned vcount = program->value_count;
   if (!vcount || !program->instruction_count)
      return true;

   uint32_t *canon = NULL;
   struct util_dynarray to_remove;
   util_dynarray_init(&to_remove, NULL);

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (!a9_is_identity_ior(program, ins))
         continue;
      if (!canon) {
         canon = malloc(vcount * sizeof(*canon));
         if (!canon) {
            util_dynarray_fini(&to_remove);
            return false;
         }
         for (unsigned v = 0; v < vcount; ++v)
            canon[v] = v;
      }
      uint32_t root = ins->src[0];
      while (canon[root] != root)
         root = canon[root];
      if (root == ins->dest)
         continue;
      canon[ins->dest] = root;
      if (program->max_phys &&
          program->max_phys[ins->dest] != AGX_APPLE9_PHYS_INVALID) {
         if (program->max_phys[root] == AGX_APPLE9_PHYS_INVALID ||
             program->max_phys[ins->dest] < program->max_phys[root])
            program->max_phys[root] = program->max_phys[ins->dest];
      }
      util_dynarray_append(&to_remove, ins);
   }

   unsigned count = util_dynarray_num_elements(
      &to_remove, struct agx_apple9_vir_instr *);
   if (count > 0) {
      for (unsigned v = 0; v < vcount; ++v) {
         uint32_t r = v;
         while (canon[r] != r)
            r = canon[r];
         canon[v] = r;
      }
      for (unsigned i = 0; i < program->instruction_count; ++i) {
         struct agx_apple9_vir_instr *ins = program->instructions[i];
         unsigned ns = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t *sp = agx_apple9_vir_src_ptr(ins, s);
            if (*sp < vcount && canon[*sp] != *sp)
               *sp = canon[*sp];
         }
      }
      if (program->output < vcount)
         program->output = canon[program->output];
      for (unsigned k = 0; k < program->live_out_count; ++k) {
         if (program->live_out[k] < vcount)
            program->live_out[k] = canon[program->live_out[k]];
      }
      agx_apple9_vir_remove_many(program, util_dynarray_begin(&to_remove),
                                 count);
   }

   free(canon);
   util_dynarray_fini(&to_remove);
   return true;
}

/*
 * Canonicalize multi-word source operands into contiguous tuples:
 *  - Reuse existing contiguous multi-component definitions when non-clobber
 *    (setting retain_mask for indirect/pointer loads reading leading lanes).
 *  - Maintain a 16-entry intra-block read-only tuple reuse cache (window <=
 *    128 instructions) to avoid duplicate VIR_COLLECTs for repeated address
 *    pairs or vectors.
 *  - Emit private VIR_COLLECT pseudos for clobbered or uncached tuples.
 *  - Insert SSA VIR_MOV before scalar AGX_APPLE9_OPERAND_CLOBBER sources
 *    (coalesced away automatically when the source dies at the instruction).
 */
static bool
a9_prepare_tuples_and_clobbers(struct agx_apple9_vir_program *program,
                               struct apple9_ra_spill_state *spill,
                               bool do_clobbers, const char **error)
{
   agx_apple9_vir_reindex(program);
   unsigned n_orig = program->instruction_count;
   if (n_orig == 0)
      return true;

   /* Snapshot instruction order so newly inserted COLLECT/MOV instructions
    * do not shift our scan. */
   struct agx_apple9_vir_instr **snap =
      malloc(n_orig * sizeof(*snap));
   unsigned def_cap = MAX2(program->value_count, 64u);
   struct agx_apple9_vir_instr **defs = calloc(def_cap, sizeof(*defs));
   if (!snap || !defs) {
      free(snap);
      free(defs);
      *error = "out of memory preparing VIR tuples";
      return false;
   }
   memcpy(snap, program->instructions, n_orig * sizeof(*snap));

   for (unsigned i = 0; i < n_orig; ++i) {
      struct agx_apple9_vir_instr *ins = snap[i];
      if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components == 0)
         continue;
      for (unsigned c = 0; c < ins->dest_components; ++c) {
         if (ins->dest + c < def_cap)
            defs[ins->dest + c] = ins;
      }
   }

   struct {
      uint32_t src[4];
      unsigned width;
      unsigned first;
      unsigned last;
      uint32_t dest_base;
   } tuples[32];
   unsigned nr_tuples = 0;
   struct agx_apple9_block *tuple_block = NULL;

   bool ok = true;
   for (unsigned i = 0; i < n_orig && ok; ++i) {
      struct agx_apple9_vir_instr *ins = snap[i];
      struct agx_apple9_block *blk = agx_apple9_instr_block(ins);
      if (!blk || !blk->cfg_active)
         continue;
      if (ins->op == AGX_APPLE9_VIR_PHI ||
          ins->encoding == AGX_APPLE9_ENC_PSEUDO)
         continue;

      if (blk != tuple_block) {
         tuple_block = blk;
         nr_tuples = 0;
      }

      for (unsigned s = 0; s < ins->nr_srcs; ) {
         unsigned width = apple9_ra_source_width(ins, s);
         if (s + width > ins->nr_srcs) {
            *error = "source tuple width exceeds nr_srcs";
            ok = false;
            break;
         }

         uint32_t s0 = ins->src[s];
         if (a9_is_pub_val(program, s0)) {
            s += width;
            continue;
         }

         const struct agx_apple9_operand_constraint *op_c =
            apple9_ra_source_constraint(ins, s);
         const bool is_clobber =
            op_c && (op_c->flags & AGX_APPLE9_OPERAND_CLOBBER);

         if (width == 1) {
            if (do_clobbers && is_clobber) {
               uint32_t copy_v = 0;
               if (!apple9_ra_vir_grow_values(program, spill, 1, &copy_v)) {
                  *error = "out of memory allocating clobber copy value";
                  ok = false;
                  break;
               }
               struct agx_apple9_block *saved = program->current_block;
               program->current_block = blk;
               uint32_t mov_src[1] = {s0};
               bool emitted = agx_apple9_vir_emit_side_effect(
                  program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, mov_src,
                  1, 0);
               program->current_block = saved;
               if (!emitted) {
                  *error = "out of memory emitting clobber VIR_MOV";
                  ok = false;
                  break;
               }
               struct agx_apple9_vir_instr *mov =
                  program->instructions[program->instruction_count - 1];
               mov->dest = copy_v;
               mov->dest_components = 1;
               agx_apple9_vir_move_before(program, mov, ins);
               ins->src[s] = copy_v;
            }
            s += 1;
            continue;
         }

         /* Multi-word source tuple (width 2..4). */
         const bool is_load =
            ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT ||
            ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER;

         const struct agx_apple9_vir_instr *def =
            (s0 < def_cap) ? defs[s0] : NULL;
         bool is_existing_tuple =
            !is_clobber && def && def->op != AGX_APPLE9_VIR_SPLIT &&
            def->dest == s0 &&
            (is_load ? def->dest_components >= width
                     : def->dest_components == width);
         for (unsigned c = 1; is_existing_tuple && c < width; ++c)
            is_existing_tuple &= (ins->src[s + c] == s0 + c);

         if (is_existing_tuple) {
            const bool shared_existing = !is_clobber && width <= 4 &&
                                         def->op == AGX_APPLE9_VIR_COLLECT &&
                                         def->nr_srcs == width &&
                                         agx_apple9_instr_block(def) == blk;
            if (shared_existing) {
               bool reused_existing = false;
               for (unsigned t = 0; t < nr_tuples; ++t) {
                  if (tuples[t].width != width || i - tuples[t].last > 64 ||
                      i - tuples[t].first > 128)
                     continue;
                  bool same = true;
                  for (unsigned c = 0; c < width; ++c)
                     same &= (tuples[t].src[c] == def->src[c]);
                  if (same) {
                     for (unsigned c = 0; c < width; ++c)
                        ins->src[s + c] = tuples[t].dest_base + c;
                     tuples[t].last = i;
                     reused_existing = true;
                     break;
                  }
               }
               if (!reused_existing) {
                  unsigned slot;
                  if (nr_tuples < ARRAY_SIZE(tuples)) {
                     slot = nr_tuples++;
                  } else {
                     slot = 0;
                     for (unsigned t = 1; t < nr_tuples; ++t) {
                        if (tuples[t].last < tuples[slot].last)
                           slot = t;
                     }
                  }
                  tuples[slot].width = width;
                  for (unsigned c = 0; c < width; ++c)
                     tuples[slot].src[c] = def->src[c];
                  tuples[slot].first = i;
                  tuples[slot].last = i;
                  tuples[slot].dest_base = def->dest;
               }
            }
            if (is_load) {
               uint16_t retain = (uint16_t)BITFIELD_RANGE(s, width);
               ins->retain_mask |= retain;
            }
            s += width;
            continue;
         }

         const bool shared = !is_clobber && width <= 4;
         bool reused = false;
         if (shared) {
            for (unsigned t = 0; t < nr_tuples; ++t) {
               if (tuples[t].width != width || i - tuples[t].last > 64 ||
                   i - tuples[t].first > 128)
                  continue;
               bool same = true;
               for (unsigned c = 0; c < width; ++c)
                  same &= (tuples[t].src[c] == ins->src[s + c]);
               if (same) {
                  for (unsigned c = 0; c < width; ++c)
                     ins->src[s + c] = tuples[t].dest_base + c;
                  tuples[t].last = i;
                  reused = true;
                  break;
               }
            }
         }

         if (!reused) {
            uint32_t base = 0;
            if (!apple9_ra_vir_grow_values(program, spill, width, &base)) {
               *error = "out of memory allocating VIR_COLLECT tuple";
               ok = false;
               break;
            }
            uint32_t csrcs[4];
            for (unsigned c = 0; c < width; ++c)
               csrcs[c] = ins->src[s + c];

            struct agx_apple9_block *saved = program->current_block;
            program->current_block = blk;
            bool emitted = agx_apple9_vir_emit_side_effect(
               program, AGX_APPLE9_VIR_COLLECT, AGX_APPLE9_ENC_PSEUDO, csrcs,
               width, 0);
            program->current_block = saved;
            if (!emitted) {
               *error = "out of memory emitting VIR_COLLECT";
               ok = false;
               break;
            }
            struct agx_apple9_vir_instr *collect =
               program->instructions[program->instruction_count - 1];
            collect->dest = base;
            collect->dest_components = (uint8_t)width;
            agx_apple9_vir_move_before(program, collect, ins);

            for (unsigned c = 0; c < width; ++c)
               ins->src[s + c] = base + c;

            if (shared) {
               unsigned slot;
               if (nr_tuples < ARRAY_SIZE(tuples)) {
                  slot = nr_tuples++;
               } else {
                  /* Evict expired entries first, otherwise least-recently-used. */
                  slot = 0;
                  for (unsigned t = 1; t < nr_tuples; ++t) {
                     if (tuples[t].last < tuples[slot].last)
                        slot = t;
                  }
               }
               tuples[slot].width = width;
               for (unsigned c = 0; c < width; ++c)
                  tuples[slot].src[c] = csrcs[c];
               tuples[slot].first = i;
               tuples[slot].last = i;
               tuples[slot].dest_base = base;
            }
         }

         s += width;
      }
   }

   free(snap);
   free(defs);
   if (ok) {
      agx_apple9_vir_reindex(program);
      ok = a9_ensure_program_pub_array(program, error);
   }
   return ok;
}

/* =========================================================================
 * Exact 32-bit GPR demand calculation (for spilling & budget decisions)
 * ========================================================================= */

static unsigned
a9_calc_gpr_demand(const struct agx_apple9_vir_program *program,
                   const struct apple9_ra_liveness_info *liveness,
                   const struct apple9_ra_spill_state *spill,
                   unsigned reserved, unsigned *min_single_insn_out)
{
   unsigned max_demand = reserved;
   unsigned min_single = reserved;

   unsigned vcount = program->value_count;
   uint8_t *def_w = NULL;
   if (min_single_insn_out && vcount > 0) {
      def_w = malloc(vcount * sizeof(*def_w));
      if (def_w) {
         memset(def_w, 1, vcount * sizeof(*def_w));
         for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
            if (!b->cfg_active)
               continue;
            for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
                 ins; ins = agx_apple9_instr_next(ins)) {
               if (ins->dest != AGX_APPLE9_VREG_INVALID &&
                   ins->dest < vcount && ins->dest_components > 1 &&
                   !a9_is_pub_val(program, ins->dest))
                  def_w[ins->dest] = ins->dest_components;
            }
         }
      }
   }

   /* Ensure pre-defined fixed VIR inputs are never excluded by assign_bound. */
   if (program->fixed_phys) {
      for (unsigned v = 0; v < vcount; ++v) {
         if (!a9_is_pub_val(program, v) &&
             program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID &&
             program->fixed_phys[v] < AGX_APPLE9_GPR_COUNT)
            max_demand = MAX2(max_demand, (unsigned)program->fixed_phys[v] + 1u);
      }
   }

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;

      unsigned bi = b->cfg_index;
      unsigned demand = reserved;

      U_SPARSE_BITSET_FOREACH_SET(&liveness->live_in[bi], v) {
         if (!a9_is_pub_val(program, v) && !apple9_ra_value_is_mem(spill, v))
            demand += 1;
      }
      max_demand = MAX2(max_demand, demand);

      unsigned late_kill = 0;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI)
            continue;

         if (ins->op == AGX_APPLE9_VIR_ENTRY_REG) {
            max_demand = MAX2(max_demand, ins->immediate + 1u);
            min_single = MAX2(min_single, ins->immediate + 1u);
         } else if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               if (program->live_out_reg[k] != AGX_APPLE9_PHYS_INVALID) {
                  max_demand = MAX2(max_demand, program->live_out_reg[k] + 1u);
                  min_single = MAX2(min_single, program->live_out_reg[k] + 1u);
               }
            }
         }

         if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
             ins->op != AGX_APPLE9_VIR_ENTRY_REG) {
            if (ins->dest != AGX_APPLE9_VREG_INVALID &&
                ins->dest_components > 0 &&
                !a9_is_pub_val(program, ins->dest) &&
                !apple9_ra_value_is_mem(spill, ins->dest)) {
               struct a9_gpr_constraint dc = a9_dest_constraint(program, ins);
               unsigned first = ALIGN_POT(MAX2((unsigned)dc.min, reserved),
                                          (unsigned)dc.align);
               max_demand = MAX2(max_demand, first + ins->dest_components);
            }
            for (unsigned s = 0; s < ins->nr_srcs; ) {
               unsigned w = apple9_ra_source_width(ins, s);
               uint32_t sv = ins->src[s];
               if (!a9_is_pub_val(program, sv) &&
                   !apple9_ra_value_is_mem(spill, sv)) {
                  struct a9_gpr_constraint sc =
                     a9_src_constraint(program, ins, s);
                  unsigned first = ALIGN_POT(MAX2((unsigned)sc.min, reserved),
                                             (unsigned)sc.align);
                  max_demand = MAX2(max_demand, first + w);
               }
               s += w;
            }
         }

         demand -= late_kill;
         late_kill = 0;

         /* Count single-instruction footprint (for force_spill floor). */
         unsigned insn_need = reserved;
         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0 &&
             !a9_is_pub_val(program, ins->dest) &&
             !apple9_ra_value_is_mem(spill, ins->dest)) {
            insn_need += util_next_power_of_two(ins->dest_components);
         }

         if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            unsigned comps = ins->dest_components;
            const bool zero_copy = (ins->kill_mask & 1u) && ins->nr_srcs >= 1;
            if (zero_copy) {
               /* Zero-copy coalesced VIR_SPLIT reuses the killed source tuple
                * GPRs directly without needing simultaneous extra registers. */
               uint32_t s0 = ins->src[0];
               for (unsigned c = 0; c < comps; ++c) {
                  if (!a9_is_pub_val(program, s0 + c) &&
                      !apple9_ra_value_is_mem(spill, s0 + c) &&
                      demand > reserved)
                     demand -= 1;
               }
            }
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->dead_dest_mask & (1u << c))
                  continue;
               uint32_t dv = ins->dest + c;
               if (!a9_is_pub_val(program, dv) &&
                   !apple9_ra_value_is_mem(spill, dv))
                  demand += 1;
            }
            max_demand = MAX2(max_demand, demand);
            min_single = MAX2(min_single, zero_copy ? insn_need : insn_need + comps);
            continue;
         }

         unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t sv = agx_apple9_vir_src(ins, s);
            if (a9_is_pub_val(program, sv) || apple9_ra_value_is_mem(spill, sv))
               continue;
            bool dup = false;
            for (unsigned p = 0; p < s; ++p)
               dup |= (agx_apple9_vir_src(ins, p) == sv);
            if (dup)
               continue;
            unsigned sw = (def_w && sv < vcount) ? def_w[sv] : 1u;
            insn_need += util_next_power_of_two(sw);
            if (agx_apple9_vir_src_killed(ins, s))
               late_kill += 1;
         }
         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t sv = program->live_out[k];
               if (a9_is_pub_val(program, sv) ||
                   apple9_ra_value_is_mem(spill, sv))
                  continue;
               bool dup = false;
               for (unsigned p = 0; p < k; ++p)
                  dup |= (program->live_out[p] == sv);
               if (!dup) {
                  unsigned sw = (def_w && sv < vcount) ? def_w[sv] : 1u;
                  insn_need += util_next_power_of_two(sw);
               }
            }
         }

         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0 &&
             !a9_is_pub_val(program, ins->dest) &&
             !apple9_ra_value_is_mem(spill, ins->dest)) {
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               demand += 1;
               if (c < 8 && (ins->dead_dest_mask & (1u << c)))
                  late_kill += 1;
            }
         }

         max_demand = MAX2(max_demand, demand);
         min_single = MAX2(min_single, insn_need);
      }
      demand -= late_kill;
   }

   free(def_w);
   if (min_single_insn_out)
      *min_single_insn_out = min_single;
   return max_demand;
}

/* =========================================================================
 * RPO (dominance) block ordering
 * ========================================================================= */

static bool
a9_compute_rpo(struct agx_apple9_block *const *cfg_blocks, unsigned n,
               unsigned *rpo_out)
{
   unsigned *postorder = malloc(n * sizeof(unsigned));
   unsigned *dfs_blk = malloc(n * sizeof(unsigned));
   uint8_t *dfs_edge = malloc(n * sizeof(uint8_t));
   bool *seen = calloc(n, sizeof(bool));
   if (!postorder || !dfs_blk || !dfs_edge || !seen) {
      free(postorder);
      free(dfs_blk);
      free(dfs_edge);
      free(seen);
      return false;
   }

   unsigned sp = 0, po = 0;
   seen[0] = true;
   dfs_blk[sp] = 0;
   dfs_edge[sp] = 0;
   ++sp;

   while (sp > 0) {
      unsigned cur = dfs_blk[sp - 1];
      uint8_t edge = dfs_edge[sp - 1];
      if (edge < 2) {
         dfs_edge[sp - 1] = edge + 1;
         struct agx_apple9_block *succ = cfg_blocks[cur]->cfg_succs[edge];
         if (succ && succ->cfg_active && !seen[succ->cfg_index]) {
            seen[succ->cfg_index] = true;
            dfs_blk[sp] = succ->cfg_index;
            dfs_edge[sp] = 0;
            ++sp;
         }
      } else {
         postorder[po++] = cur;
         --sp;
      }
   }

   /* Append any disconnected active blocks defensively (should not happen). */
   for (unsigned i = 0; i < n; ++i) {
      if (!seen[i])
         postorder[po++] = i;
   }

   for (unsigned i = 0; i < n; ++i)
      rpo_out[i] = postorder[n - 1 - i];

   free(postorder);
   free(dfs_blk);
   free(dfs_edge);
   free(seen);
   return true;
}

/* =========================================================================
 * Phi Web Coalescing, Preferences, Scarcity & Placement Costing
 * ========================================================================= */

struct a9_phi_web_node {
   uint32_t parent;
   uint8_t rank;
};

struct a9_phi_preference {
   struct a9_phi_preference *next;
   uint32_t other_ssa;
   uint64_t weight;
};

struct a9_phi_web_members {
   uint32_t head, tail, next;
   unsigned count;
   struct a9_phi_preference *constant;
};

struct a9_phi_dead_dest {
   struct a9_phi_dead_dest *next;
   uint32_t dst_ssa;
};

struct a9_phi_dead_group {
   struct a9_phi_dead_group *next;
   struct agx_apple9_block *succ;
   struct agx_apple9_block *pred;
   struct a9_phi_dead_dest *dests;
   uint64_t weight;
};

struct a9_phi_reg_pref {
   unsigned reg;
   uint64_t dead, edge, affinity;
};

struct a9_placement_cost {
   uint64_t dead, edge, affinity;
   unsigned moves, collect, preference, materialization, async_waw,
      compact_r64, growth, scarcity, bank_conflict, reg;
};

#define A9_FMA_MAX_PEERS 8u

struct a9_fma_peers {
   uint32_t peer[A9_FMA_MAX_PEERS];
   uint8_t weight[A9_FMA_MAX_PEERS];
   uint8_t count;
};

static unsigned
phi_web_find(struct a9_phi_web_node *web, unsigned x)
{
   unsigned root = x;
   while (web[root].parent != root)
      root = web[root].parent;
   while (web[x].parent != x) {
      unsigned next = web[x].parent;
      web[x].parent = root;
      x = next;
   }
   return root;
}

static void
phi_web_union(struct a9_phi_web_node *web, unsigned x, unsigned y)
{
   x = phi_web_find(web, x);
   y = phi_web_find(web, y);
   if (x == y)
      return;
   if (web[x].rank < web[y].rank)
      SWAP(x, y);
   web[y].parent = x;
   if (web[x].rank == web[y].rank)
      web[x].rank++;
}

static void
phi_web_union_noninterfering(struct a9_phi_web_node *web,
                             struct a9_phi_web_members *members,
                             struct u_sparse_bitset *interference,
                             unsigned nr_ssa, unsigned x, unsigned y)
{
   x = phi_web_find(web, x);
   y = phi_web_find(web, y);
   if (x == y)
      return;

   if (members[x].count > members[y].count)
      SWAP(x, y);

   for (uint32_t v = members[x].head; v != UINT32_MAX; v = members[v].next) {
      if (v >= nr_ssa)
         continue;
      U_SPARSE_BITSET_FOREACH_SET(&interference[v], other) {
         if (phi_web_find(web, other) == y)
            return;
      }
   }

   phi_web_union(web, x, y);
   unsigned root = phi_web_find(web, x);
   unsigned other = (root == x) ? y : x;
   members[members[root].tail].next = members[other].head;
   members[root].tail = members[other].tail;
   members[root].count += members[other].count;
}

static struct u_sparse_bitset *
build_phi_web_interference(const struct agx_apple9_vir_program *program,
                           struct agx_apple9_block *const *cfg_blocks,
                           const struct apple9_ra_liveness_info *liveness,
                           const struct apple9_ra_spill_state *spill,
                           const BITSET_WORD *constants, void *memctx)
{
   unsigned n = program->value_count;
   unsigned alloc = MAX2(n, 1u);
   BITSET_WORD *participates = BITSET_RZALLOC(memctx, alloc);
   struct u_sparse_bitset *graph =
      rzalloc_array(memctx, struct u_sparse_bitset, alloc);
   if (!participates || !graph)
      return NULL;

   for (unsigned i = 0; i < alloc; ++i)
      u_sparse_bitset_init(&graph[i], alloc, memctx);

   for (unsigned bi = 0; bi < program->num_cfg_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI) {
            if (ins->dest >= n || a9_is_pub_val(program, ins->dest) ||
                apple9_ra_value_is_mem(spill, ins->dest))
               continue;
            BITSET_SET(participates, ins->dest);
            unsigned ns = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < ns; ++s) {
               uint32_t sv = agx_apple9_vir_src(ins, s);
               if (sv < n && !a9_is_pub_val(program, sv) &&
                   !apple9_ra_value_is_mem(spill, sv) &&
                   !BITSET_TEST(constants, sv))
                  BITSET_SET(participates, sv);
            }
         } else if (ins->op == AGX_APPLE9_VIR_MOV &&
                    ins->dest_components == 1 && ins->nr_srcs == 1) {
            uint32_t dv = ins->dest;
            uint32_t sv = ins->src[0];
            if (dv < n && sv < n && !a9_is_pub_val(program, dv) &&
                !a9_is_pub_val(program, sv) &&
                !apple9_ra_value_is_mem(spill, dv) &&
                !apple9_ra_value_is_mem(spill, sv)) {
               BITSET_SET(participates, dv);
               if (!BITSET_TEST(constants, sv))
                  BITSET_SET(participates, sv);
            }
         }
      }
   }

   struct u_sparse_bitset live;
   u_sparse_bitset_init(&live, alloc, memctx);

   for (unsigned bi = 0; bi < program->num_cfg_blocks; ++bi) {
      struct agx_apple9_block *b = cfg_blocks[bi];
      U_SPARSE_BITSET_FOREACH_SET(&liveness->live_out[bi], v) {
         if (v < n && BITSET_TEST(participates, v))
            u_sparse_bitset_set(&live, v);
      }

      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_last_instr(b);
           ins; ins = agx_apple9_instr_prev(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI)
            continue;

         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               uint32_t dv = ins->dest + c;
               if (dv < n && BITSET_TEST(participates, dv))
                  u_sparse_bitset_set(&live, dv);
            }
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               uint32_t dv = ins->dest + c;
               if (dv < n && BITSET_TEST(participates, dv)) {
                  U_SPARSE_BITSET_FOREACH_SET(&live, other) {
                     if (dv != other) {
                        u_sparse_bitset_set(&graph[dv], other);
                        u_sparse_bitset_set(&graph[other], dv);
                     }
                  }
               }
            }
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               uint32_t dv = ins->dest + c;
               if (dv < n)
                  u_sparse_bitset_clear(&live, dv);
            }
         }

         unsigned ns = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t sv = agx_apple9_vir_src(ins, s);
            if (sv < n && BITSET_TEST(participates, sv))
               u_sparse_bitset_set(&live, sv);
         }
      }

      /* Parallel phi definitions at block entry interfere with each other and
       * with all live-in phi participants. */
      for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(b);
           phi; phi = agx_apple9_instr_next(phi)) {
         if (a9_is_block_entry_prefix(phi))
            continue;
         if (phi->op != AGX_APPLE9_VIR_PHI)
            break;
         if (phi->dest < n && BITSET_TEST(participates, phi->dest))
            u_sparse_bitset_set(&live, phi->dest);
      }
      for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(b);
           phi; phi = agx_apple9_instr_next(phi)) {
         if (a9_is_block_entry_prefix(phi))
            continue;
         if (phi->op != AGX_APPLE9_VIR_PHI)
            break;
         if (phi->dest < n && BITSET_TEST(participates, phi->dest)) {
            uint32_t v = phi->dest;
            U_SPARSE_BITSET_FOREACH_SET(&live, other) {
               if (v != other) {
                  u_sparse_bitset_set(&graph[v], other);
                  u_sparse_bitset_set(&graph[other], v);
               }
            }
         }
      }
      u_sparse_bitset_clear_all(&live);
   }

   u_sparse_bitset_free(&live);
   return graph;
}

/* =========================================================================
 * Dominance-Order SSA Register Assignment Context
 * ========================================================================= */

struct a9_split_entry {
   uint32_t orig_v;
   uint32_t cur_v;
};

struct a9_synth_phi {
   uint32_t orig_v;
   struct agx_apple9_vir_instr *phi;
};

struct a9_ra_ctx {
   void *memctx;
   struct agx_apple9_vir_program *program;
   struct apple9_ra_spill_state *spill;
   const struct apple9_ra_liveness_info *liveness;
   struct agx_apple9_block **cfg_blocks;
   unsigned num_cfg_blocks;
   unsigned orig_vcount;

   unsigned bound;          /* Max 32-bit GPR count (<= 96) */
   unsigned reserved;       /* Reserved GPR prefix (0, 2, or 5) */
   unsigned demand;         /* Estimated 32-bit GPR demand */
   unsigned reg_count;      /* High-water mark: max assigned GPR + 1 */
   bool cost_constraints;
   bool any_split;          /* True if any SSA relocation/split occurred */
   bool needs_spill_headroom;
   const char *error;

   /* Per-value dynamic arrays (capacity = val_cap >= program->value_count). */
   unsigned val_cap;
   uint8_t *phys;           /* Physical GPR 0..95 or AGX_APPLE9_PHYS_INVALID */
   uint32_t *orig_ssa;      /* Maps split SSA value -> original SSA ID */
   uint32_t *last_reg;      /* Most recent GPR for phi_web history */
   BITSET_WORD *visited;    /* Bit v set once value v has been assigned */

   /* Current block register file (96-bit compact masks!). */
   uint64_t used_gprs[2];
   uint64_t pinned[2];
   uint32_t reg_to_ssa[AGX_APPLE9_GPR_COUNT];

   /* Soft tracking of dead destination lanes written by still-in-flight async
    * producers so RA avoids overwriting them and forcing a WAW WAIT stall. */
   struct {
      uint64_t live[2];
      uint64_t dead[2];
      bool active;
   } async_pending[6];
   unsigned async_next;
   uint64_t async_dead_gprs[2];

   /* Active SSA version mapping for the current block (orig_v -> cur_v). */
   uint32_t *cur_ssa;
   struct util_dynarray active_splits; /* orig_v entries currently != orig_v */

   /* Per-block exit split records and synthetic split-join phis. */
   struct util_dynarray *block_splits_out; /* array[num_cfg_blocks] of a9_split_entry */
   struct util_dynarray *block_synth_phis; /* array[num_cfg_blocks] of a9_synth_phi */
   bool *block_visited;

   /* Phi web & affinity metadata (indexed by orig_v < orig_vcount). */
   struct a9_phi_web_node *phi_web;
   struct a9_phi_web_members *web_members;
   struct a9_phi_preference **phi_prefs;
   struct a9_phi_dead_group **dead_groups;
   uint64_t *phi_weights;
   struct a9_collect_ref {
      uint32_t src[AGX_APPLE9_MAX_VIR_SRCS];
      uint8_t nr_srcs;
      uint8_t dest_components;
      unsigned weight;
      struct a9_collect_ref *next;
   } **collect_refs;
   uint8_t *precolor_hint;

   /* Cached per-value phi preferences for current placement query. */
   uint32_t affinity_ssa;
   unsigned affinity_reg;
   struct util_dynarray reg_prefs;
   int16_t reg_pref_index[AGX_APPLE9_GPR_COUNT];
   uint64_t dead_total, edge_total, affinity_total;
   unsigned bank_conflict[3];
   unsigned bank_conflict_min;

   /* Soft placement constraints, bank coloring, compact r0..r63 preference,
    * and bank scarcity. */
   struct a9_gpr_constraint *preferred;
   struct a9_fma_peers *fma_peers;
   unsigned fma_peers_count;
   BITSET_WORD *compact_alu;
   unsigned scarcity[AGX_APPLE9_GPR_COUNT];
   unsigned scarcity_floor[A9_SCARCITY_MAX_WIDTH + 1][AGX_APPLE9_GPR_COUNT + 1];

   /* Current block and instruction during local assignment. */
   struct agx_apple9_block *cur_block;
   struct agx_apple9_vir_instr *cur_instr;
};

static bool
a9_ra_ensure_val_cap(struct a9_ra_ctx *rctx, unsigned needed)
{
   if (needed <= rctx->val_cap)
      return true;

   unsigned old_cap = rctx->val_cap;
   unsigned new_cap = MAX2(old_cap ? old_cap * 2u : 64u, needed);
   unsigned old_words = old_cap ? BITSET_WORDS(old_cap) : 0u;
   unsigned new_words = BITSET_WORDS(new_cap);

   uint8_t *phys = realloc(rctx->phys, new_cap * sizeof(*phys));
   if (!phys) {
      rctx->error = "out of memory growing RA value tables";
      return false;
   }
   rctx->phys = phys;
   memset(phys + old_cap, AGX_APPLE9_PHYS_INVALID, new_cap - old_cap);

   uint32_t *orig = realloc(rctx->orig_ssa, new_cap * sizeof(*orig));
   if (!orig) {
      rctx->error = "out of memory growing RA value tables";
      return false;
   }
   rctx->orig_ssa = orig;
   for (unsigned v = old_cap; v < new_cap; ++v)
      orig[v] = v;

   uint32_t *last = realloc(rctx->last_reg, new_cap * sizeof(*last));
   if (!last) {
      rctx->error = "out of memory growing RA value tables";
      return false;
   }
   rctx->last_reg = last;
   for (unsigned v = old_cap; v < new_cap; ++v)
      last[v] = UINT32_MAX;

   BITSET_WORD *vis = realloc(rctx->visited, new_words * sizeof(*vis));
   if (!vis) {
      rctx->error = "out of memory growing RA value tables";
      return false;
   }
   rctx->visited = vis;
   if (new_words > old_words)
      memset(vis + old_words, 0, (new_words - old_words) * sizeof(*vis));

   rctx->val_cap = new_cap;
   return true;
}

static bool
a9_ra_grow_values(struct a9_ra_ctx *rctx, unsigned extra, uint32_t *base_out)
{
   if (!apple9_ra_vir_grow_values(rctx->program, rctx->spill, extra, base_out)) {
      rctx->error = "out of memory growing VIR values";
      return false;
   }
   if (!a9_ra_ensure_val_cap(rctx, rctx->program->value_count))
      return false;
   return a9_ensure_program_pub_array(rctx->program, &rctx->error);
}

static inline uint32_t
a9_resolve_cur(const struct a9_ra_ctx *rctx, uint32_t v)
{
   if (v >= rctx->orig_vcount)
      return v;
   return rctx->cur_ssa[v];
}

static void
a9_set_cur_ssa(struct a9_ra_ctx *rctx, uint32_t orig_v, uint32_t new_v)
{
   if (orig_v >= rctx->orig_vcount)
      return;
   if (rctx->cur_ssa[orig_v] == orig_v && new_v != orig_v)
      util_dynarray_append(&rctx->active_splits, orig_v);
   rctx->cur_ssa[orig_v] = new_v;
}

static uint32_t
a9_lookup_block_out(const struct a9_ra_ctx *rctx, unsigned pred_cfg_idx,
                    uint32_t orig_v)
{
   const struct util_dynarray *arr = &rctx->block_splits_out[pred_cfg_idx];
   util_dynarray_foreach(arr, struct a9_split_entry, e) {
      if (e->orig_v == orig_v)
         return e->cur_v;
   }
   return orig_v;
}

static inline void
a9_assign_gpr_val(struct a9_ra_ctx *rctx, uint32_t v, unsigned r)
{
   assert(r < rctx->bound && r < AGX_APPLE9_GPR_COUNT);
   rctx->phys[v] = (uint8_t)r;
   rctx->last_reg[v] = r;
   if (rctx->orig_ssa[v] < rctx->orig_vcount)
      rctx->last_reg[rctx->orig_ssa[v]] = r;
   BITSET_SET(rctx->visited, v);
   rctx->reg_count = MAX2(rctx->reg_count, r + 1u);
   rctx->affinity_ssa = UINT32_MAX;
}

static unsigned
a9_collect_affinity(struct a9_ra_ctx *rctx, uint32_t ssa)
{
   if (ssa >= rctx->orig_vcount)
      return UINT_MAX;

   if (rctx->precolor_hint &&
       rctx->precolor_hint[ssa] != AGX_APPLE9_PHYS_INVALID)
      return rctx->precolor_hint[ssa];

   struct a9_collect_ref *refs = rctx->collect_refs[ssa];
   if (!refs)
      return UINT_MAX;

   /* Pass 1: Highest-weight collect that already has a placed sibling so we
    * co-locate recurring address/operand pairs in consecutive GPRs. */
   unsigned best_reg = UINT_MAX;
   unsigned best_weight = 0;
   for (struct a9_collect_ref *ref = refs; ref; ref = ref->next) {
      if (ref->weight < best_weight)
         continue;

      unsigned ours = UINT_MAX;
      for (unsigned s = 0; s < ref->nr_srcs; ++s) {
         if (ref->src[s] == ssa) {
            ours = s;
            break;
         }
      }
      if (ours == UINT_MAX)
         continue;

      unsigned width = ref->dest_components;
      for (unsigned s = 0; s < ref->nr_srcs; ++s) {
         uint32_t other = ref->src[s];
         if (other == ssa || other >= rctx->val_cap ||
             !BITSET_TEST(rctx->visited, other) ||
             rctx->phys[other] == AGX_APPLE9_PHYS_INVALID)
            continue;
         if (rctx->phys[other] >= s) {
            unsigned base = rctx->phys[other] - s;
            if (base >= rctx->reserved && base + width <= rctx->bound &&
                base + ours < rctx->bound) {
               if (ref->weight > best_weight ||
                   (ref->weight == best_weight &&
                    !gpr_test(rctx->used_gprs, base + ours))) {
                  best_weight = ref->weight;
                  best_reg = base + ours;
               }
            }
         }
      }
   }
   if (best_reg != UINT_MAX)
      return best_reg;

   /* Pass 2: No sibling placed yet — reserve a free consecutive window for
    * the highest-weight collect referencing `ssa`. */
   struct a9_collect_ref *top = NULL;
   for (struct a9_collect_ref *ref = refs; ref; ref = ref->next) {
      if (!top || ref->weight > top->weight)
         top = ref;
   }
   if (top) {
      unsigned ours = UINT_MAX;
      for (unsigned s = 0; s < top->nr_srcs; ++s) {
         if (top->src[s] == ssa) {
            ours = s;
            break;
         }
      }
      unsigned width = top->dest_components;
      if (ours != UINT_MAX) {
         for (unsigned base = rctx->reserved; base + width <= rctx->bound; ++base) {
            if (!gpr_test_range(rctx->used_gprs, base, width))
               return base + ours;
         }
      }
   }
   return UINT_MAX;
}

static void
a9_add_reg_pref(struct a9_ra_ctx *rctx, unsigned reg, uint64_t edge,
                uint64_t affinity, uint64_t dead)
{
   if (reg >= AGX_APPLE9_GPR_COUNT)
      return;
   rctx->dead_total += dead;
   rctx->edge_total += edge;
   rctx->affinity_total += affinity;

   int16_t idx = rctx->reg_pref_index[reg];
   if (idx >= 0) {
      struct a9_phi_reg_pref *p = util_dynarray_element(
         &rctx->reg_prefs, struct a9_phi_reg_pref, idx);
      p->dead += dead;
      p->edge += edge;
      p->affinity += affinity;
      return;
   }
   rctx->reg_pref_index[reg] = (int16_t)util_dynarray_num_elements(
      &rctx->reg_prefs, struct a9_phi_reg_pref);
   struct a9_phi_reg_pref entry = {reg, dead, edge, affinity};
   util_dynarray_append(&rctx->reg_prefs, entry);
}

static void
a9_cache_phi_prefs(struct a9_ra_ctx *rctx, uint32_t ssa)
{
   if (rctx->affinity_ssa == ssa)
      return;

   util_dynarray_foreach(&rctx->reg_prefs, struct a9_phi_reg_pref, p) {
      if (p->reg < AGX_APPLE9_GPR_COUNT)
         rctx->reg_pref_index[p->reg] = -1;
   }
   util_dynarray_clear(&rctx->reg_prefs);
   rctx->dead_total = rctx->edge_total = rctx->affinity_total = 0;

   uint32_t orig = (ssa < rctx->val_cap) ? rctx->orig_ssa[ssa] : ssa;
   if (orig < rctx->orig_vcount) {
      for (struct a9_phi_preference *p = rctx->phi_prefs[orig]; p; p = p->next) {
         uint32_t other = p->other_ssa;
         if (other < rctx->val_cap && rctx->last_reg[other] != UINT32_MAX)
            a9_add_reg_pref(rctx, rctx->last_reg[other], p->weight, 0, 0);
      }

      if (rctx->precolor_hint &&
          rctx->precolor_hint[orig] != AGX_APPLE9_PHYS_INVALID)
         a9_add_reg_pref(rctx, rctx->precolor_hint[orig], 1ull << 20, 0, 0);

      unsigned root = phi_web_find(rctx->phi_web, orig);
      for (uint32_t v = rctx->web_members[root].head; v != UINT32_MAX;
           v = rctx->web_members[v].next) {
         if (v >= rctx->orig_vcount) {
            struct a9_phi_preference *p = rctx->web_members[v].constant;
            if (p && p->other_ssa < rctx->val_cap &&
                rctx->last_reg[p->other_ssa] != UINT32_MAX)
               a9_add_reg_pref(rctx, rctx->last_reg[p->other_ssa], 0,
                               p->weight, 0);
         } else {
            if (v != orig && rctx->last_reg[v] != UINT32_MAX)
               a9_add_reg_pref(rctx, rctx->last_reg[v], 0, rctx->phi_weights[v],
                               0);
            if (v != orig && rctx->precolor_hint &&
                rctx->precolor_hint[v] != AGX_APPLE9_PHYS_INVALID)
               a9_add_reg_pref(rctx, rctx->precolor_hint[v], 0, 1ull << 18, 0);
         }
      }

      for (struct a9_phi_dead_group *g = rctx->dead_groups[orig]; g;
           g = g->next) {
         uint64_t locs[2] = {0, 0};
         for (struct a9_phi_dead_dest *d = g->dests; d; d = d->next) {
            uint32_t dv = d->dst_ssa;
            if (dv < rctx->val_cap && rctx->last_reg[dv] != UINT32_MAX) {
               gpr_set(locs, rctx->last_reg[dv]);
               continue;
            }
            if (dv < rctx->orig_vcount) {
               unsigned droot = phi_web_find(rctx->phi_web, dv);
               for (uint32_t v = rctx->web_members[droot].head;
                    v != UINT32_MAX; v = rctx->web_members[v].next) {
                  if (v < rctx->orig_vcount && v != orig &&
                      rctx->last_reg[v] != UINT32_MAX)
                     gpr_set(locs, rctx->last_reg[v]);
               }
            }
         }
         for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r) {
            if (gpr_test(locs, r))
               a9_add_reg_pref(rctx, r, 0, 0, g->weight);
         }
      }
   }

   memset(rctx->bank_conflict, 0, sizeof(rctx->bank_conflict));
   rctx->bank_conflict_min = 0;
   if (rctx->fma_peers && orig < rctx->orig_vcount) {
      unsigned root = phi_web_find(rctx->phi_web, orig);
      if (root < rctx->fma_peers_count && rctx->fma_peers[root].count > 0) {
         const struct a9_fma_peers *fp = &rctx->fma_peers[root];
         for (unsigned k = 0; k < fp->count; ++k) {
            uint32_t peer_root = fp->peer[k];
            unsigned pr = UINT_MAX;
            unsigned fallback_pr = UINT_MAX;
            for (uint32_t v = rctx->web_members[peer_root].head;
                 v != UINT32_MAX; v = rctx->web_members[v].next) {
               if (v >= rctx->orig_vcount)
                  continue;
               uint32_t cur_peer = a9_resolve_cur(rctx, v);
               if (cur_peer < rctx->val_cap &&
                   rctx->phys[cur_peer] < AGX_APPLE9_GPR_COUNT) {
                  unsigned cand = rctx->phys[cur_peer];
                  if (gpr_test(rctx->used_gprs, cand) &&
                      rctx->reg_to_ssa[cand] == cur_peer) {
                     pr = cand;
                     break;
                  }
                  if (fallback_pr == UINT_MAX)
                     fallback_pr = cand;
               }
               if (fallback_pr == UINT_MAX && v < rctx->val_cap &&
                   rctx->last_reg[v] < AGX_APPLE9_GPR_COUNT) {
                  fallback_pr = rctx->last_reg[v];
               }
            }
            if (pr == UINT_MAX)
               pr = fallback_pr;
            if (pr < AGX_APPLE9_GPR_COUNT)
               rctx->bank_conflict[pr % 3u] += fp->weight[k];
         }
         unsigned mn = rctx->bank_conflict[0];
         for (unsigned b = 1; b < 3; ++b)
            mn = MIN2(mn, rctx->bank_conflict[b]);
         rctx->bank_conflict_min = mn;
      }
   }

   rctx->affinity_reg = a9_collect_affinity(rctx, orig);
   rctx->affinity_ssa = ssa;
}

static struct a9_placement_cost
a9_placement_cost(struct a9_ra_ctx *rctx, uint32_t ssa, unsigned reg,
                  unsigned count, unsigned extra_hint_reg)
{
   struct a9_placement_cost cost = { .reg = reg };
   a9_cache_phi_prefs(rctx, ssa);
   cost.dead = rctx->dead_total;
   cost.edge = rctx->edge_total;
   cost.affinity = rctx->affinity_total;

   if (reg < AGX_APPLE9_GPR_COUNT) {
      int16_t idx = rctx->reg_pref_index[reg];
      if (idx >= 0) {
         const struct a9_phi_reg_pref *pref = util_dynarray_element(
            &rctx->reg_prefs, struct a9_phi_reg_pref, idx);
         cost.dead -= pref->dead;
         cost.edge -= pref->edge;
         cost.affinity -= pref->affinity;
      }
   }

   unsigned hint = extra_hint_reg != UINT_MAX ? extra_hint_reg : rctx->affinity_reg;
   cost.collect = (hint != UINT_MAX && hint != reg) ? 1u : 0u;

   for (unsigned r = reg; r < reg + count; ++r)
      cost.async_waw += gpr_test(rctx->async_dead_gprs, r) ? 1u : 0u;

   uint32_t orig = (ssa < rctx->val_cap) ? rctx->orig_ssa[ssa] : ssa;
   if (rctx->bound > 64u && rctx->compact_alu && orig < rctx->orig_vcount &&
       BITSET_TEST(rctx->compact_alu, orig))
      cost.compact_r64 = (reg + count > 64u) ? 1u : 0u;

   if (count == 1)
      cost.bank_conflict = rctx->bank_conflict[reg % 3u];

   if (!rctx->cost_constraints)
      return cost;

   if (rctx->preferred && orig < rctx->orig_vcount) {
      struct a9_gpr_constraint c = rctx->preferred[orig];
      if (c.min <= c.max)
         cost.preference = (reg < c.min || reg > c.max || (reg % c.align)) ? 1u : 0u;
   }

   if (rctx->cur_instr && rctx->cur_instr->op == AGX_APPLE9_VIR_IMM &&
       rctx->cur_instr->dest == ssa)
      cost.materialization = (reg + count > 64u) ? 1u : 0u;

   unsigned limit = MAX2(rctx->demand, rctx->reg_count);
   cost.growth = (reg + count > limit) ? (reg + count - limit) : 0u;
   for (unsigned r = reg; r < reg + count; ++r)
      cost.scarcity += rctx->scarcity[r];
   return cost;
}

static struct a9_placement_cost
a9_placement_floor(struct a9_ra_ctx *rctx, uint32_t ssa, unsigned reg,
                   unsigned count, const struct a9_placement_cost *floors,
                   unsigned last_pref, unsigned extra_hint_reg)
{
   struct a9_placement_cost cost = *floors;
   cost.reg = reg;
   if (last_pref == UINT_MAX || reg > last_pref) {
      cost.dead = rctx->dead_total;
      cost.edge = rctx->edge_total;
      cost.affinity = rctx->affinity_total;
   }
   unsigned hint = extra_hint_reg != UINT_MAX ? extra_hint_reg : rctx->affinity_reg;
   cost.collect = (hint != UINT_MAX && hint < reg) ? 1u : 0u;

   uint32_t orig = (ssa < rctx->val_cap) ? rctx->orig_ssa[ssa] : ssa;
   if (rctx->bound > 64u && rctx->compact_alu && orig < rctx->orig_vcount &&
       BITSET_TEST(rctx->compact_alu, orig))
      cost.compact_r64 = (reg + count > 64u) ? 1u : 0u;

   if (count == 1)
      cost.bank_conflict = rctx->bank_conflict_min;

   if (!rctx->cost_constraints)
      return cost;

   if (rctx->preferred && orig < rctx->orig_vcount) {
      struct a9_gpr_constraint c = rctx->preferred[orig];
      cost.preference = (c.min <= c.max && reg > c.max) ? 1u : 0u;
   }

   if (rctx->cur_instr && rctx->cur_instr->op == AGX_APPLE9_VIR_IMM &&
       rctx->cur_instr->dest == ssa)
      cost.materialization = (reg + count > 64u) ? 1u : 0u;

   unsigned limit = MAX2(rctx->demand, rctx->reg_count);
   cost.growth = (reg + count > limit) ? (reg + count - limit) : 0u;
   if (count >= 1 && count <= A9_SCARCITY_MAX_WIDTH && reg <= AGX_APPLE9_GPR_COUNT)
      cost.scarcity = rctx->scarcity_floor[count][reg];
   return cost;
}

static bool
a9_placement_cheaper(struct a9_placement_cost a, struct a9_placement_cost b)
{
   if (a.moves != b.moves) return a.moves < b.moves;
   if (a.edge != b.edge) return a.edge < b.edge;
   if (a.dead != b.dead) return a.dead < b.dead;
   if (a.affinity != b.affinity) return a.affinity < b.affinity;
   if (a.collect != b.collect) return a.collect < b.collect;
   if (a.preference != b.preference) return a.preference < b.preference;
   if (a.materialization != b.materialization)
      return a.materialization < b.materialization;
   if (a.async_waw != b.async_waw) return a.async_waw < b.async_waw;
   if (a.compact_r64 != b.compact_r64) return a.compact_r64 < b.compact_r64;
   if (a.growth != b.growth) return a.growth < b.growth;
   if (a.bank_conflict != b.bank_conflict)
      return a.bank_conflict < b.bank_conflict;
   if (a.scarcity != b.scarcity) return a.scarcity < b.scarcity;
   return a.reg < b.reg;
}

/* =========================================================================
 * Acyclic SSA Live-Range Splitting & Victim Eviction
 * ========================================================================= */

/* Pick a free unpinned GPR outside `forbid` to receive an evicted scalar. */
static unsigned
a9_find_free_eviction_gpr(const struct a9_ra_ctx *rctx, const uint64_t forbid[2])
{
   unsigned best = UINT_MAX;
   unsigned best_scarcity = UINT_MAX;
   for (unsigned r = rctx->reserved; r < rctx->bound; ++r) {
      if (gpr_test(rctx->used_gprs, r) || gpr_test(rctx->pinned, r) ||
          gpr_test(forbid, r))
         continue;
      if (!rctx->cost_constraints)
         return r;
      if (rctx->scarcity[r] < best_scarcity) {
         best_scarcity = rctx->scarcity[r];
         best = r;
         if (best_scarcity == 0)
            return best;
      }
   }
   return best;
}

/* Evict the live scalar SSA value occupying `r_victim` to a free GPR outside
 * `forbid` by emitting an SSA VIR_MOV right before `before`. Rewrites any
 * source of `before` that referenced the evicted value to the new SSA ID. */
static bool
a9_evict_register(struct a9_ra_ctx *rctx, unsigned r_victim,
                  const uint64_t forbid[2], struct agx_apple9_vir_instr *before)
{
   assert(gpr_test(rctx->used_gprs, r_victim));
   assert(!gpr_test(rctx->pinned, r_victim));

   uint32_t u = rctx->reg_to_ssa[r_victim];
   assert(u != AGX_APPLE9_VREG_INVALID && u < rctx->val_cap);

   unsigned r_free = a9_find_free_eviction_gpr(rctx, forbid);
   if (r_free == UINT_MAX) {
      rctx->needs_spill_headroom = true;
      rctx->error = "no free GPR available for live-range eviction";
      return false;
   }

   uint32_t u_new = 0;
   if (!a9_ra_grow_values(rctx, 1, &u_new))
      return false;

   struct agx_apple9_block *saved = rctx->program->current_block;
   rctx->program->current_block = rctx->cur_block;
   uint32_t msrc[1] = {u};
   bool emitted = agx_apple9_vir_emit_side_effect(
      rctx->program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, msrc, 1, 0);
   rctx->program->current_block = saved;
   if (!emitted) {
      rctx->error = "out of memory emitting eviction VIR_MOV";
      return false;
   }

   struct agx_apple9_vir_instr *mov =
      rctx->program->instructions[rctx->program->instruction_count - 1];
   mov->dest = u_new;
   mov->dest_components = 1;
   if (before)
      agx_apple9_vir_move_before(rctx->program, mov, before);

   /* Update register file and SSA version mapping. */
   uint32_t orig_v = rctx->orig_ssa[u];
   rctx->orig_ssa[u_new] = orig_v;
   a9_set_cur_ssa(rctx, orig_v, u_new);
   rctx->any_split = true;

   gpr_clear(rctx->used_gprs, r_victim);
   rctx->reg_to_ssa[r_victim] = AGX_APPLE9_VREG_INVALID;

   gpr_set(rctx->used_gprs, r_free);
   rctx->reg_to_ssa[r_free] = u_new;
   a9_assign_gpr_val(rctx, u_new, r_free);

   /* If the instruction about to execute also reads `u`, point it at `u_new`. */
   if (before) {
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(before);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         if (agx_apple9_vir_src(before, s) == u)
            *agx_apple9_vir_src_ptr(before, s) = u_new;
      }
   }

   return true;
}

/* Search [first..last] (step `align`) for the lowest-cost unpinned `width`-GPR
 * window for `ssa` (using placement_cost + placement_floor). */
static unsigned
a9_choose_window(struct a9_ra_ctx *rctx, uint32_t ssa, unsigned width,
                 unsigned first, unsigned last, unsigned align,
                 unsigned extra_hint_reg, unsigned *moves_out)
{
   a9_cache_phi_prefs(rctx, ssa);
   uint64_t max_edge = 0, max_dead = 0, max_aff = 0;
   unsigned last_pref = UINT_MAX;
   util_dynarray_foreach(&rctx->reg_prefs, struct a9_phi_reg_pref, p) {
      max_edge = MAX2(max_edge, p->edge);
      max_dead = MAX2(max_dead, p->dead);
      max_aff = MAX2(max_aff, p->affinity);
      if (p->reg < AGX_APPLE9_GPR_COUNT &&
          (last_pref == UINT_MAX || p->reg > last_pref))
         last_pref = p->reg;
   }
   const struct a9_placement_cost floors = {
      .edge = rctx->edge_total - max_edge,
      .dead = rctx->dead_total - max_dead,
      .affinity = rctx->affinity_total - max_aff,
   };

   /* Moves rank first: check if any completely free unpinned window exists. */
   bool any_free = false;
   for (unsigned r = first; r <= last; r += align) {
      if (!gpr_test_range(rctx->pinned, r, width) &&
          !gpr_test_range(rctx->used_gprs, r, width)) {
         any_free = true;
         break;
      }
   }

   unsigned chosen = UINT_MAX;
   struct a9_placement_cost best = { .moves = UINT_MAX };

   for (unsigned r = first; r <= last; r += align) {
      if (chosen != UINT_MAX &&
          a9_placement_cheaper(
             best, a9_placement_floor(rctx, ssa, r, width, &floors, last_pref,
                                      extra_hint_reg)))
         break;

      if (gpr_test_range(rctx->pinned, r, width))
         continue;

      bool occ = gpr_test_range(rctx->used_gprs, r, width);
      if (occ && (any_free || (chosen != UINT_MAX && best.moves == 0)))
         continue;

      struct a9_placement_cost cost =
         a9_placement_cost(rctx, ssa, r, width, extra_hint_reg);
      if (occ) {
         for (unsigned c = 0; c < width; ++c)
            cost.moves += gpr_test(rctx->used_gprs, r + c) ? 1u : 0u;
      }

      if (chosen == UINT_MAX || a9_placement_cheaper(cost, best)) {
         best = cost;
         chosen = r;
      }
   }

   if (moves_out)
      *moves_out = (chosen != UINT_MAX) ? best.moves : UINT_MAX;
   return chosen;
}

/* Vacate all occupied registers in [base .. base + width - 1] via acyclic SSA
 * VIR_MOVs right before `before`. */
static bool
a9_vacate_window(struct a9_ra_ctx *rctx, unsigned base, unsigned width,
                 struct agx_apple9_vir_instr *before)
{
   uint64_t forbid[2] = {0, 0};
   gpr_set_range(forbid, base, width);
   for (unsigned c = 0; c < width; ++c) {
      if (gpr_test(rctx->used_gprs, base + c)) {
         if (!a9_evict_register(rctx, base + c, forbid, before))
            return false;
      }
   }
   return true;
}

/* =========================================================================
 * Instruction Operand Assignment (Constrained Sources + Destinations)
 * ========================================================================= */

static bool
a9_assign_constrained_sources(struct a9_ra_ctx *rctx,
                              struct agx_apple9_vir_instr *ins)
{
   const unsigned nr_srcs = MIN2((unsigned)ins->nr_srcs, AGX_APPLE9_MAX_VIR_SRCS);
   for (unsigned s = 0; s < nr_srcs; ) {
      unsigned w = apple9_ra_source_width(ins, s);
      if (w == 0 || w > 4 || s + w > nr_srcs) {
         rctx->error = "source tuple width exceeds instruction nr_srcs";
         return false;
      }
      uint32_t s0 = ins->src[s];
      if (a9_is_pub_val(rctx->program, s0) ||
          apple9_ra_value_is_mem(rctx->spill, s0)) {
         s += w;
         continue;
      }

      struct a9_gpr_constraint c = a9_src_constraint(rctx->program, ins, s);

      if (w == 1) {
         /* Merge constraints across duplicate occurrences of the same value. */
         bool seen_earlier = false;
         for (unsigned t = 0; t < ins->nr_srcs; ) {
            unsigned tw = apple9_ra_source_width(ins, t);
            if (tw == 1 && ins->src[t] == s0) {
               if (t < s)
                  seen_earlier = true;
               struct a9_gpr_constraint other =
                  a9_src_constraint(rctx->program, ins, t);
               c.min = MAX2(c.min, other.min);
               c.max = MIN2(c.max, other.max);
               c.align = MAX2(c.align, other.align);
            }
            t += tw;
         }
         if (seen_earlier) {
            s += 1;
            continue;
         }

         unsigned cur_r = rctx->phys[s0];
         assert(cur_r < AGX_APPLE9_GPR_COUNT);

         /* Check if already inside legal window (or fixed in reserved). */
         if (cur_r >= c.min && cur_r <= c.max && (cur_r % c.align) == 0) {
            /* Pin if this operand has a nontrivial constraint so later
             * operands cannot evict it out of its legal bank. */
            if (c.min > rctx->reserved || c.max + 1u < rctx->bound ||
                c.align > 1)
               gpr_set(rctx->pinned, cur_r);
            s += 1;
            continue;
         }

         /* Relocate `s0` into a legal register inside [first..last] via a
          * fresh SSA VIR_MOV right before `ins`. */
         unsigned first, last;
         if (!a9_constraint_window(c, 1, rctx->reserved, rctx->bound, &first,
                                   &last)) {
            rctx->error = "source constraint has no aligned allocatable GPR";
            return false;
         }

         /* Pin `s0`'s current register while choosing/evicting so we don't
          * accidentally evict `s0` itself before reading it. */
         bool was_pinned = gpr_test(rctx->pinned, cur_r);
         gpr_set(rctx->pinned, cur_r);

         unsigned moves = 0;
         unsigned r_dst =
            a9_choose_window(rctx, s0, 1, first, last, c.align, UINT_MAX, &moves);
         if (r_dst == UINT_MAX) {
            rctx->needs_spill_headroom = true;
            rctx->error = "instruction source constraints are unsatisfiable";
            return false;
         }

         if (moves > 0) {
            /* Check if a free GPR exists; if not and COPY_REG is reserved
             * and `s0` dies here, bridge via COPY_REG to swap cleanly. */
            uint64_t forbid[2] = {0, 0};
            gpr_set(forbid, r_dst);
            if (a9_find_free_eviction_gpr(rctx, forbid) != UINT_MAX) {
               if (!a9_vacate_window(rctx, r_dst, 1, ins))
                  return false;
            } else if (rctx->reserved >= 2 &&
                       agx_apple9_vir_src_killed(ins, s)) {
               /* 3-step SSA swap via COPY_REG (r1) when register file is 100%
                * full and source dies at `ins`:
                *   tmp (r1) <- victim (r_dst)
                *   v_reloc (r_dst) <- s0 (cur_r)
                *   victim_new (cur_r) <- tmp (r1)
                */
               uint32_t victim = rctx->reg_to_ssa[r_dst];
               uint32_t base_v = 0;
               if (!a9_ra_grow_values(rctx, 3, &base_v))
                  return false;
               uint32_t tmp_v = base_v + 0;
               uint32_t reloc_v = base_v + 1;
               uint32_t vic_new = base_v + 2;

               struct agx_apple9_block *saved = rctx->program->current_block;
               rctx->program->current_block = rctx->cur_block;
               uint32_t s1[1] = {victim}, s2[1] = {s0}, s3[1] = {tmp_v};
               bool e1 = agx_apple9_vir_emit_side_effect(
                  rctx->program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, s1,
                  1, 0);
               struct agx_apple9_vir_instr *m1 =
                  e1 ? rctx->program->instructions[rctx->program->instruction_count - 1]
                     : NULL;
               bool e2 = e1 && agx_apple9_vir_emit_side_effect(
                  rctx->program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, s2,
                  1, 0);
               struct agx_apple9_vir_instr *m2 =
                  e2 ? rctx->program->instructions[rctx->program->instruction_count - 1]
                     : NULL;
               bool e3 = e2 && agx_apple9_vir_emit_side_effect(
                  rctx->program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, s3,
                  1, 0);
               struct agx_apple9_vir_instr *m3 =
                  e3 ? rctx->program->instructions[rctx->program->instruction_count - 1]
                     : NULL;
               rctx->program->current_block = saved;
               if (!e3) {
                  rctx->error = "out of memory emitting COPY_REG swap bridge";
                  return false;
               }
               m1->dest = tmp_v;   m1->dest_components = 1;
               m2->dest = reloc_v; m2->dest_components = 1;
               m3->dest = vic_new; m3->dest_components = 1;
               agx_apple9_vir_move_before(rctx->program, m1, ins);
               agx_apple9_vir_move_before(rctx->program, m2, ins);
               agx_apple9_vir_move_before(rctx->program, m3, ins);

               a9_assign_gpr_val(rctx, tmp_v, A9_COPY_REG);
               a9_assign_gpr_val(rctx, reloc_v, r_dst);
               a9_assign_gpr_val(rctx, vic_new, cur_r);

               uint32_t vic_orig = rctx->orig_ssa[victim];
               rctx->orig_ssa[vic_new] = vic_orig;
               a9_set_cur_ssa(rctx, vic_orig, vic_new);
               rctx->any_split = true;

               rctx->reg_to_ssa[r_dst] = reloc_v;
               rctx->reg_to_ssa[cur_r] = vic_new;
               if (!was_pinned)
                  gpr_clear(rctx->pinned, cur_r);
               gpr_set(rctx->pinned, r_dst);

               for (unsigned t = 0; t < ins->nr_srcs; ++t) {
                  if (ins->src[t] == s0) {
                     ins->src[t] = reloc_v;
                     agx_apple9_vir_set_src_killed(ins, t, true);
                  } else if (ins->src[t] == victim) {
                     ins->src[t] = vic_new;
                  }
               }
               s += 1;
               continue;
            } else {
               rctx->needs_spill_headroom = true;
               rctx->error = "no free GPR available to relocate constrained source";
               return false;
            }
         }

         if (!was_pinned)
            gpr_clear(rctx->pinned, cur_r);

         uint32_t reloc_v = 0;
         if (!a9_ra_grow_values(rctx, 1, &reloc_v))
            return false;

         struct agx_apple9_block *saved = rctx->program->current_block;
         rctx->program->current_block = rctx->cur_block;
         uint32_t msrc[1] = {s0};
         bool emitted = agx_apple9_vir_emit_side_effect(
            rctx->program, AGX_APPLE9_VIR_MOV, AGX_APPLE9_ENC_PSEUDO, msrc, 1,
            0);
         rctx->program->current_block = saved;
         if (!emitted) {
            rctx->error = "out of memory emitting source relocation VIR_MOV";
            return false;
         }
         struct agx_apple9_vir_instr *mov =
            rctx->program->instructions[rctx->program->instruction_count - 1];
         mov->dest = reloc_v;
         mov->dest_components = 1;
         agx_apple9_vir_move_before(rctx->program, mov, ins);

         bool was_killed = agx_apple9_vir_src_killed(ins, s);
         if (was_killed) {
            mov->kill_mask = 1u;
            if (rctx->reg_to_ssa[cur_r] == s0) {
               gpr_clear(rctx->used_gprs, cur_r);
               rctx->reg_to_ssa[cur_r] = AGX_APPLE9_VREG_INVALID;
            }
         }

         gpr_set(rctx->used_gprs, r_dst);
         gpr_set(rctx->pinned, r_dst);
         rctx->reg_to_ssa[r_dst] = reloc_v;
         a9_assign_gpr_val(rctx, reloc_v, r_dst);
         rctx->any_split = true;

         for (unsigned t = 0; t < ins->nr_srcs; ++t) {
            if (ins->src[t] == s0) {
               ins->src[t] = reloc_v;
               agx_apple9_vir_set_src_killed(ins, t, true);
               ins->retain_mask &= (uint16_t)~BITFIELD_BIT(t);
            }
         }
         s += 1;
         continue;
      }

      /* Multi-word source tuple (width 2..4): verify contiguous & in-window. */
      unsigned base_r = rctx->phys[s0];
      bool ok_tuple =
         base_r < AGX_APPLE9_GPR_COUNT && base_r >= c.min && base_r <= c.max &&
         (base_r % c.align) == 0 && base_r + w <= rctx->bound;
      for (unsigned k = 1; ok_tuple && k < w; ++k)
         ok_tuple &= (rctx->phys[ins->src[s + k]] == base_r + k);

      if (ok_tuple) {
         gpr_set_range(rctx->pinned, base_r, w);
         s += w;
         continue;
      }

      /* Re-collect into a legal contiguous window via VIR_COLLECT. */
      unsigned first, last;
      if (!a9_constraint_window(c, w, rctx->reserved, rctx->bound, &first,
                                &last)) {
         rctx->error = "multi-word source tuple has no legal GPR window";
         return false;
      }

      /* Pin all current source lanes while picking/vacating so they stay intact
       * for the VIR_COLLECT read. */
      uint64_t saved_pin[2] = {rctx->pinned[0], rctx->pinned[1]};
      for (unsigned k = 0; k < w; ++k)
         gpr_set(rctx->pinned, rctx->phys[ins->src[s + k]]);

      unsigned moves = 0;
      unsigned r_dst =
         a9_choose_window(rctx, s0, w, first, last, c.align, UINT_MAX, &moves);
      if (r_dst == UINT_MAX) {
         /* Allow overlapping killed source lanes if no disjoint window fit. */
         rctx->pinned[0] = saved_pin[0];
         rctx->pinned[1] = saved_pin[1];
         r_dst = a9_choose_window(rctx, s0, w, first, last, c.align, UINT_MAX,
                                  &moves);
      }
      if (r_dst == UINT_MAX) {
         rctx->needs_spill_headroom = true;
         rctx->error = "multi-word source tuple constraints are unsatisfiable";
         return false;
      }

      if (moves > 0) {
         /* Restore non-tuple pins so vacate can move non-pinned occupants. */
         rctx->pinned[0] = saved_pin[0];
         rctx->pinned[1] = saved_pin[1];
         if (!a9_vacate_window(rctx, r_dst, w, ins))
            return false;
      } else {
         rctx->pinned[0] = saved_pin[0];
         rctx->pinned[1] = saved_pin[1];
      }

      uint32_t tbase = 0;
      if (!a9_ra_grow_values(rctx, w, &tbase))
         return false;

      uint32_t csrcs[4];
      for (unsigned k = 0; k < w; ++k)
         csrcs[k] = ins->src[s + k];

      struct agx_apple9_block *saved = rctx->program->current_block;
      rctx->program->current_block = rctx->cur_block;
      bool emitted = agx_apple9_vir_emit_side_effect(
         rctx->program, AGX_APPLE9_VIR_COLLECT, AGX_APPLE9_ENC_PSEUDO, csrcs, w,
         0);
      rctx->program->current_block = saved;
      if (!emitted) {
         rctx->error = "out of memory emitting tuple relocation VIR_COLLECT";
         return false;
      }
      struct agx_apple9_vir_instr *collect =
         rctx->program->instructions[rctx->program->instruction_count - 1];
      collect->dest = tbase;
      collect->dest_components = (uint8_t)w;
      agx_apple9_vir_move_before(rctx->program, collect, ins);

      for (unsigned k = 0; k < w; ++k) {
         unsigned dst_s = s + k;
         if (dst_s >= AGX_APPLE9_MAX_VIR_SRCS)
            break;
         if (agx_apple9_vir_src_killed(ins, dst_s)) {
            bool used_elsewhere = false;
            for (unsigned t = 0; t < nr_srcs; ++t) {
               if ((t < s || t >= s + w) && ins->src[t] == csrcs[k])
                  used_elsewhere = true;
            }
            if (!used_elsewhere) {
               agx_apple9_vir_set_src_killed(collect, k, true);
               unsigned sr = rctx->phys[csrcs[k]];
               if (rctx->reg_to_ssa[sr] == csrcs[k]) {
                  gpr_clear(rctx->used_gprs, sr);
                  rctx->reg_to_ssa[sr] = AGX_APPLE9_VREG_INVALID;
               }
            }
         }
         ins->src[dst_s] = tbase + k;
         agx_apple9_vir_set_src_killed(ins, dst_s, true);
         ins->retain_mask &= (uint16_t)~BITFIELD_BIT(dst_s);
         gpr_set(rctx->used_gprs, r_dst + k);
         gpr_set(rctx->pinned, r_dst + k);
         rctx->reg_to_ssa[r_dst + k] = tbase + k;
         a9_assign_gpr_val(rctx, tbase + k, r_dst + k);
      }
      rctx->any_split = true;
      s += w;
   }

   return true;
}

static bool
a9_assign_dest(struct a9_ra_ctx *rctx, struct agx_apple9_vir_instr *ins,
               uint16_t *reused_sources_out)
{
   *reused_sources_out = 0;
   if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components == 0)
      return true;
   if (a9_is_pub_val(rctx->program, ins->dest))
      return true;
   if (apple9_ra_value_is_mem(rctx->spill, ins->dest)) {
      for (unsigned c = 0; c < ins->dest_components; ++c)
         BITSET_SET(rctx->visited, ins->dest + c);
      return true;
   }

   unsigned w = ins->dest_components;
   struct a9_gpr_constraint dc = a9_dest_constraint(rctx->program, ins);

   /* Fixed physical destination (e.g. ENTRY_REG or pre-constrained dest). */
   if (dc.min == dc.max && dc.max < rctx->reserved) {
      for (unsigned c = 0; c < w; ++c) {
         gpr_set(rctx->used_gprs, dc.min + c);
         rctx->reg_to_ssa[dc.min + c] = ins->dest + c;
         a9_assign_gpr_val(rctx, ins->dest + c, dc.min + c);
      }
      return true;
   }

   unsigned first, last;
   if (!a9_constraint_window(dc, w, rctx->reserved, rctx->bound, &first, &last)) {
      rctx->error = "destination operand has no aligned allocatable GPR window";
      return false;
   }

   /* Build reusable mask for killed sources when allowed by the encoding. */
   uint64_t reusable[2] = {0, 0};
   unsigned extra_hint = UINT_MAX;

   if (ins->op == AGX_APPLE9_VIR_MOV && ins->nr_srcs >= 1 && w == 1 &&
       !a9_is_pub_val(rctx->program, ins->src[0]) &&
       !apple9_ra_value_is_mem(rctx->spill, ins->src[0])) {
      unsigned sr = rctx->phys[ins->src[0]];
      if (agx_apple9_vir_src_killed(ins, 0) && sr >= first && sr <= last &&
          (sr % dc.align) == 0) {
         gpr_set(reusable, sr);
         extra_hint = sr;
      }
   } else if (ins->op == AGX_APPLE9_VIR_COLLECT) {
      /* VIR_COLLECT lowers via parallel copy: killed sources can be overlapped
       * by the destination tuple, and we prefer a base matching existing lanes! */
      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         uint32_t sv = ins->src[s];
         if (a9_is_pub_val(rctx->program, sv) ||
             apple9_ra_value_is_mem(rctx->spill, sv))
            continue;
         if (agx_apple9_vir_src_killed(ins, s))
            gpr_set(reusable, rctx->phys[sv]);
      }
      /* Pick the base that maximizes already-in-place lanes. */
      unsigned best_matches = 0;
      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         unsigned sr = rctx->phys[ins->src[s]];
         if (sr < s)
            continue;
         unsigned cand_base = sr - s;
         if (cand_base < first || cand_base > last || (cand_base % dc.align) != 0)
            continue;
         unsigned matches = 0;
         for (unsigned k = 0; k < w; ++k)
            matches += (rctx->phys[ins->src[k]] == cand_base + k) ? 1u : 0u;
         if (matches > best_matches) {
            best_matches = matches;
            extra_hint = cand_base;
         }
      }
   } else if (dc.reuse_killed && w == 1) {
      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         uint32_t sv = ins->src[s];
         if (a9_is_pub_val(rctx->program, sv) ||
             apple9_ra_value_is_mem(rctx->spill, sv))
            continue;
         if (!agx_apple9_vir_src_killed(ins, s))
            continue;
         if ((ins->retain_mask >> s) & 1u)
            continue;
         const struct agx_apple9_operand_constraint *sc =
            apple9_ra_source_constraint(ins, s);
         if (sc && (sc->flags & AGX_APPLE9_OPERAND_CLOBBER))
            continue;
         unsigned sr = rctx->phys[sv];
         if (sr >= first && sr <= last && (sr % dc.align) == 0)
            gpr_set(reusable, sr);
      }
   }

   /* Temporarily clear reusable GPRs from used_gprs & pinned while choosing. */
   uint64_t saved_pin[2] = {rctx->pinned[0], rctx->pinned[1]};
   rctx->used_gprs[0] &= ~reusable[0];
   rctx->used_gprs[1] &= ~reusable[1];
   rctx->pinned[0] &= ~reusable[0];
   rctx->pinned[1] &= ~reusable[1];

   /* 4-byte compact FFMA (pack_fma encode_fma4) requires 2-address reuse
    * (dst == src2/src1/src0, all < 64, distinct from the other two srcs).
    * Query collect affinity AFTER exposing reusable registers so Pass 2
    * sees freed killed registers; when no collect affinity exists, softly
    * hint the lowest-bank-conflict killed source. */
   if (dc.reuse_killed && w == 1 && ins->op == AGX_APPLE9_VIR_FMA &&
       ins->nr_srcs == 3 &&
       !ins->alu_src_uniform_mask && !ins->alu_src_immediate_mask &&
       !ins->saturate && !ins->src_abs_mask && !ins->src_neg_mask &&
       !ins->src_half_mask && !ins->src_hi_mask && !ins->src_sext_mask) {
      rctx->affinity_ssa = UINT32_MAX;
      a9_cache_phi_prefs(rctx, ins->dest);
      if (rctx->affinity_reg == UINT_MAX) {
         unsigned r0 = rctx->phys[ins->src[0]];
         unsigned r1 = rctx->phys[ins->src[1]];
         unsigned r2 = rctx->phys[ins->src[2]];
         if (r0 < 64u && r1 < 64u && r2 < 64u) {
            const unsigned order[3] = {2u, 1u, 0u};
            unsigned best_sr = UINT_MAX;
            unsigned best_bc = UINT_MAX;
            for (unsigned oi = 0; oi < 3; ++oi) {
               unsigned s = order[oi];
               unsigned sr = (s == 0) ? r0 : (s == 1 ? r1 : r2);
               unsigned o1 = (s == 0) ? r1 : r0;
               unsigned o2 = (s == 2) ? r1 : r2;
               if (sr == o1 || sr == o2 || !gpr_test(reusable, sr))
                  continue;
               unsigned bc = rctx->bank_conflict[sr % 3u];
               if (best_sr == UINT_MAX || bc < best_bc) {
                  best_sr = sr;
                  best_bc = bc;
               }
            }
            if (best_sr != UINT_MAX)
               extra_hint = best_sr;
         }
      }
   }

   unsigned moves = 0;
   unsigned chosen = a9_choose_window(rctx, ins->dest, w, first, last, dc.align,
                                      extra_hint, &moves);

   /* Restore reusable bits before eviction so non-reused killed sources stay
    * tracked until the post-instruction kill step. */
   rctx->used_gprs[0] |= reusable[0];
   rctx->used_gprs[1] |= reusable[1];
   rctx->pinned[0] = saved_pin[0];
   rctx->pinned[1] = saved_pin[1];

   if (chosen == UINT_MAX) {
      rctx->needs_spill_headroom = true;
      rctx->error = "could not find legal GPR window for instruction dest";
      return false;
   }

   /* Evict any non-reusable occupants in [chosen .. chosen + w - 1]. */
   if (moves > 0) {
      uint64_t forbid[2] = {0, 0};
      gpr_set_range(forbid, chosen, w);
      for (unsigned c = 0; c < w; ++c) {
         unsigned r = chosen + c;
         if (gpr_test(rctx->used_gprs, r) && !gpr_test(reusable, r)) {
            if (!a9_evict_register(rctx, r, forbid, ins))
               return false;
         }
      }
   }

   /* Mark any killed source whose GPR was taken over by the destination. */
   uint16_t reused = 0;
   for (unsigned s = 0; s < ins->nr_srcs && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
      uint32_t sv = ins->src[s];
      if (a9_is_pub_val(rctx->program, sv) ||
          apple9_ra_value_is_mem(rctx->spill, sv))
         continue;
      unsigned sr = rctx->phys[sv];
      if (agx_apple9_vir_src_killed(ins, s) && gpr_test(reusable, sr) &&
          sr >= chosen && sr < chosen + w) {
         reused |= (uint16_t)BITFIELD_BIT(s);
      }
   }
   *reused_sources_out = reused;

   for (unsigned c = 0; c < w; ++c) {
      gpr_set(rctx->used_gprs, chosen + c);
      rctx->reg_to_ssa[chosen + c] = ins->dest + c;
      a9_assign_gpr_val(rctx, ins->dest + c, chosen + c);
   }
   return true;
}

/* =========================================================================
 * Loop-Header Backedge Split Repair (when a loop-invariant value is evicted)
 * ========================================================================= */

static bool
a9_splice_loop_header_phi(struct a9_ra_ctx *rctx,
                          struct agx_apple9_block *header, uint32_t orig_v)
{
   unsigned h_idx = header->cfg_index;
   /* Check if header already has a synthetic phi for orig_v. */
   util_dynarray_foreach(&rctx->block_synth_phis[h_idx], struct a9_synth_phi, sp) {
      if (sp->orig_v == orig_v)
         return true;
   }

   uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(
      rctx->program, header, header->num_cfg_preds);
   if (phi_v == AGX_APPLE9_VREG_INVALID) {
      rctx->error = "out of memory emitting loop-header split VIR_PHI";
      return false;
   }
   if (!apple9_ra_spill_state_ensure(rctx->spill, rctx->program->value_count) ||
       !a9_ra_ensure_val_cap(rctx, rctx->program->value_count) ||
       !a9_ensure_program_pub_array(rctx->program, &rctx->error))
      return false;

   struct agx_apple9_vir_instr *phi =
      rctx->program->instructions[rctx->program->instruction_count - 1];
   /* Place phi at logical block entry (right after any ENTRY_REG / ELSE). */
   struct agx_apple9_vir_instr *first_non_prefix = NULL;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(header);
        I; I = agx_apple9_instr_next(I)) {
      if (I == phi)
         continue;
      if (!a9_is_block_entry_prefix(I)) {
         first_non_prefix = I;
         break;
      }
   }
   if (first_non_prefix)
      agx_apple9_vir_move_before(rctx->program, phi, first_non_prefix);
   else
      agx_apple9_vir_move_to_block_start(rctx->program, phi, header);

   for (unsigned p = 0; p < header->num_cfg_preds; ++p) {
      struct agx_apple9_block *pred = header->cfg_preds[p];
      uint32_t pv = rctx->block_visited[pred->cfg_index]
                       ? a9_lookup_block_out(rctx, pred->cfg_index, orig_v)
                       : orig_v;
      *agx_apple9_vir_src_ptr(phi, p) = pv;
   }

   /* Assign the loop-header phi to the exact GPR `orig_v` already occupied. */
   a9_assign_gpr_val(rctx, phi_v, rctx->phys[orig_v]);
   rctx->orig_ssa[phi_v] = orig_v;

   struct a9_synth_phi entry = {orig_v, phi};
   util_dynarray_append(&rctx->block_synth_phis[h_idx], entry);

   /* Rewrite uses of `orig_v` inside visited blocks in the loop span
    * [h_idx .. max_backedge_idx] to `phi_v` (stopping after any split def). */
   unsigned loop_end = h_idx;
   for (unsigned p = 0; p < header->num_cfg_preds; ++p)
      loop_end = MAX2(loop_end, header->cfg_preds[p]->cfg_index);

   for (unsigned b = h_idx; b <= loop_end; ++b) {
      if (!rctx->block_visited[b])
         continue;
      struct agx_apple9_block *blk = rctx->cfg_blocks[b];
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(blk);
           I; I = agx_apple9_instr_next(I)) {
         if (I == phi)
            continue;
         if (I->op == AGX_APPLE9_VIR_PHI) {
            for (unsigned p = 0; p < agx_apple9_vir_nr_srcs(I); ++p) {
               unsigned p_idx = blk->cfg_preds[p]->cfg_index;
               if (p_idx >= h_idx && p_idx <= loop_end &&
                   agx_apple9_vir_src(I, p) == orig_v)
                  *agx_apple9_vir_src_ptr(I, p) = phi_v;
            }
            continue;
         }
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            if (agx_apple9_vir_src(I, s) == orig_v)
               *agx_apple9_vir_src_ptr(I, s) = phi_v;
         }
      }
      /* Record the spliced loop-header phi on visited loop blocks exiting with
       * `orig_v` live-out so downstream merge blocks outside the loop see
       * `phi_v` instead of stale `orig_v`. */
      if (u_sparse_bitset_test(&rctx->liveness->live_out[b], orig_v) &&
          a9_lookup_block_out(rctx, b, orig_v) == orig_v) {
         struct a9_split_entry se = {orig_v, phi_v};
         util_dynarray_append(&rctx->block_splits_out[b], se);
      }
   }
   return true;
}

static void
a9_recompute_async_dead(struct a9_ra_ctx *rctx)
{
   rctx->async_dead_gprs[0] = rctx->async_dead_gprs[1] = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(rctx->async_pending); ++i) {
      if (rctx->async_pending[i].active) {
         rctx->async_dead_gprs[0] |= rctx->async_pending[i].dead[0];
         rctx->async_dead_gprs[1] |= rctx->async_pending[i].dead[1];
      }
   }
}

static void
a9_retire_async_for_instr(struct a9_ra_ctx *rctx,
                          const struct agx_apple9_vir_instr *ins)
{
   bool changed = false;
   if (ins->op == AGX_APPLE9_VIR_WAIT ||
       ins->op == AGX_APPLE9_VIR_DEVICE_FENCE ||
       ins->op == AGX_APPLE9_VIR_WORKGROUP_BARRIER ||
       apple9_ra_after_logical_end(ins)) {
      for (unsigned i = 0; i < ARRAY_SIZE(rctx->async_pending); ++i) {
         if (rctx->async_pending[i].active) {
            rctx->async_pending[i].active = false;
            changed = true;
         }
      }
   } else {
      unsigned ns = agx_apple9_vir_nr_srcs(ins);
      for (unsigned s = 0; s < ns; ++s) {
         uint32_t sv = agx_apple9_vir_src(ins, s);
         if (sv >= rctx->val_cap || a9_is_pub_val(rctx->program, sv) ||
             apple9_ra_value_is_mem(rctx->spill, sv))
            continue;
         unsigned sr = rctx->phys[sv];
         if (sr >= AGX_APPLE9_GPR_COUNT)
            continue;
         for (unsigned i = 0; i < ARRAY_SIZE(rctx->async_pending); ++i) {
            if (rctx->async_pending[i].active &&
                gpr_test(rctx->async_pending[i].live, sr)) {
               rctx->async_pending[i].active = false;
               changed = true;
            }
         }
      }
   }
   if (changed)
      a9_recompute_async_dead(rctx);
}

/* =========================================================================
 * Per-Block Dominance-Order Assignment
 * ========================================================================= */

static bool
a9_assign_block(struct a9_ra_ctx *rctx, struct agx_apple9_block *block)
{
   unsigned bi = block->cfg_index;
   rctx->cur_block = block;
   rctx->used_gprs[0] = rctx->used_gprs[1] = 0;
   rctx->pinned[0] = rctx->pinned[1] = 0;
   memset(rctx->async_pending, 0, sizeof(rctx->async_pending));
   rctx->async_next = 0;
   rctx->async_dead_gprs[0] = rctx->async_dead_gprs[1] = 0;
   for (unsigned r = 0; r < AGX_APPLE9_GPR_COUNT; ++r)
      rctx->reg_to_ssa[r] = AGX_APPLE9_VREG_INVALID;
   gpr_set_range(rctx->used_gprs, 0, rctx->reserved);

   /* Reset sparse active_splits from previous block. */
   util_dynarray_foreach(&rctx->active_splits, uint32_t, orig_vp) {
      rctx->cur_ssa[*orig_vp] = *orig_vp;
   }
   util_dynarray_clear(&rctx->active_splits);

   /* Entry block: mark pre-defined fixed_phys inputs (those with no defining
    * instruction) live in their fixed GPRs. */
   if (bi == 0 && rctx->program->fixed_phys) {
      for (unsigned v = 0; v < rctx->orig_vcount; ++v) {
         if (BITSET_TEST(rctx->visited, v) &&
             rctx->program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID &&
             !a9_is_pub_val(rctx->program, v)) {
            unsigned r = rctx->phys[v];
            if (r < AGX_APPLE9_GPR_COUNT) {
               gpr_set(rctx->used_gprs, r);
               rctx->reg_to_ssa[r] = v;
            }
         }
      }
   }

   /* Reserve live-in GPRs from predecessors (reconciling any split live ranges
    * with a synthetic canonical VIR_PHI when predecessors disagree). */
   if (block->num_cfg_preds > 0) {
      bool any_pred_split = false;
      for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
         unsigned pi = block->cfg_preds[p]->cfg_index;
         if (rctx->block_visited[pi] &&
             util_dynarray_num_elements(&rctx->block_splits_out[pi],
                                        struct a9_split_entry) > 0) {
            any_pred_split = true;
            break;
         }
      }

      U_SPARSE_BITSET_FOREACH_SET(&rctx->liveness->live_in[bi], v) {
         if (v >= rctx->orig_vcount || !BITSET_TEST(rctx->visited, v) ||
             a9_is_pub_val(rctx->program, v) ||
             apple9_ra_value_is_mem(rctx->spill, v))
            continue;

         uint32_t use_v = v;
         if (any_pred_split) {
            if (block->num_cfg_preds == 1) {
               use_v = a9_lookup_block_out(rctx, block->cfg_preds[0]->cfg_index,
                                           v);
            } else {
               uint32_t first_v = UINT32_MAX;
               bool disagree = false;
               for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
                  unsigned pi = block->cfg_preds[p]->cfg_index;
                  if (!rctx->block_visited[pi]) {
                     disagree = true;
                     continue;
                  }
                  uint32_t pv = a9_lookup_block_out(rctx, pi, v);
                  if (first_v == UINT32_MAX)
                     first_v = pv;
                  else if (pv != first_v)
                     disagree = true;
               }
               if (!disagree && first_v != UINT32_MAX) {
                  use_v = first_v;
               } else if (first_v != UINT32_MAX &&
                          (disagree && (first_v != v || !block->loop_header))) {
                  /* Emit a synthetic join phi at block entry to unify split
                   * predecessors cleanly in SSA. */
                  uint32_t phi_v = agx_apple9_vir_emit_phi_canonical(
                     rctx->program, block, block->num_cfg_preds);
                  if (phi_v == AGX_APPLE9_VREG_INVALID ||
                      !apple9_ra_spill_state_ensure(rctx->spill,
                                                    rctx->program->value_count) ||
                      !a9_ra_ensure_val_cap(rctx, rctx->program->value_count) ||
                      !a9_ensure_program_pub_array(rctx->program, &rctx->error))
                     return false;

                  struct agx_apple9_vir_instr *phi =
                     rctx->program->instructions[rctx->program->instruction_count - 1];
                  struct agx_apple9_vir_instr *anchor = NULL;
                  for (struct agx_apple9_vir_instr *I =
                          agx_apple9_block_first_instr(block);
                       I; I = agx_apple9_instr_next(I)) {
                     if (I != phi && !a9_is_block_entry_prefix(I)) {
                        anchor = I;
                        break;
                     }
                  }
                  if (anchor)
                     agx_apple9_vir_move_before(rctx->program, phi, anchor);
                  else
                     agx_apple9_vir_move_to_block_start(rctx->program, phi,
                                                        block);

                  for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
                     unsigned pi = block->cfg_preds[p]->cfg_index;
                     uint32_t pv = rctx->block_visited[pi]
                                      ? a9_lookup_block_out(rctx, pi, v)
                                      : v;
                     *agx_apple9_vir_src_ptr(phi, p) = pv;
                  }
                  unsigned phi_r = rctx->phys[first_v];
                  if (gpr_test(rctx->used_gprs, phi_r)) {
                     unsigned pmoves = 0;
                     unsigned free_r = a9_choose_window(
                        rctx, phi_v, 1, rctx->reserved, rctx->bound - 1u, 1,
                        UINT_MAX, &pmoves);
                     if (free_r == UINT_MAX || pmoves > 0) {
                        rctx->error = "no free GPR for synthetic join phi";
                        return false;
                     }
                     phi_r = free_r;
                  }
                  a9_assign_gpr_val(rctx, phi_v, phi_r);
                  rctx->orig_ssa[phi_v] = v;
                  struct a9_synth_phi sp = {v, phi};
                  util_dynarray_append(&rctx->block_synth_phis[bi], sp);
                  use_v = phi_v;
               }
            }
            if (use_v != v)
               a9_set_cur_ssa(rctx, v, use_v);
         }

         unsigned r = rctx->phys[use_v];
         assert(r < AGX_APPLE9_GPR_COUNT);
         gpr_set(rctx->used_gprs, r);
         rctx->reg_to_ssa[r] = use_v;
      }
   }

   /* Step 1: Assign all leading VIR_PHI destinations simultaneously. */
   struct util_dynarray dead_phis;
   util_dynarray_init(&dead_phis, rctx->memctx);

   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(block);
        ins; ins = agx_apple9_instr_next(ins)) {
      if (a9_is_block_entry_prefix(ins))
         continue;
      if (ins->op != AGX_APPLE9_VIR_PHI)
         break;

      uint32_t dv = ins->dest;
      if (dv == AGX_APPLE9_VREG_INVALID || a9_is_pub_val(rctx->program, dv))
         continue;
      if (apple9_ra_value_is_mem(rctx->spill, dv)) {
         BITSET_SET(rctx->visited, dv);
         continue;
      }
      /* Skip synthetic phis already assigned in reserve_live_in above. */
      if (BITSET_TEST(rctx->visited, dv))
         continue;

      rctx->cur_instr = ins;
      unsigned moves = 0;
      unsigned chosen = a9_choose_window(rctx, dv, 1, rctx->reserved,
                                         rctx->bound - 1u, 1, UINT_MAX, &moves);
      if (chosen == UINT_MAX || moves > 0) {
         rctx->needs_spill_headroom = true;
         rctx->error = "no free GPR available for VIR_PHI destination";
         return false;
      }

      gpr_set(rctx->used_gprs, chosen);
      rctx->reg_to_ssa[chosen] = dv;
      a9_assign_gpr_val(rctx, dv, chosen);

      if (!u_sparse_bitset_test(&rctx->liveness->live_in[bi], dv))
         util_dynarray_append(&dead_phis, dv);
   }

   /* Release GPRs of dead phis immediately after the parallel phi bundle. */
   util_dynarray_foreach(&dead_phis, uint32_t, dvp) {
      unsigned r = rctx->phys[*dvp];
      if (r < AGX_APPLE9_GPR_COUNT && rctx->reg_to_ssa[r] == *dvp) {
         gpr_clear(rctx->used_gprs, r);
         rctx->reg_to_ssa[r] = AGX_APPLE9_VREG_INVALID;
      }
   }

   /* Step 2: Walk all non-PHI instructions in program order. */
   bool vacated_reserved = (bi != 0 || rctx->reserved == 0);
   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(block),
        *next_ins = ins ? agx_apple9_instr_next(ins) : NULL;
        ins;
        ins = next_ins, next_ins = ins ? agx_apple9_instr_next(ins) : NULL) {
      if (ins->op == AGX_APPLE9_VIR_PHI)
         continue;

      rctx->cur_instr = ins;
      rctx->pinned[0] = rctx->pinned[1] = 0;

      /* Handle VIR_ENTRY_REG directly (preloaded physical GPR). */
      if (ins->op == AGX_APPLE9_VIR_ENTRY_REG) {
         unsigned r = ins->immediate;
         if (r >= AGX_APPLE9_GPR_COUNT) {
            rctx->error = "ENTRY_REG physical GPR out of range";
            return false;
         }
         a9_assign_gpr_val(rctx, ins->dest, r);
         if (!(ins->dead_dest_mask & 1u)) {
            gpr_set(rctx->used_gprs, r);
            rctx->reg_to_ssa[r] = ins->dest;
         }
         continue;
      }

      /* Once all ENTRY_REG preloads in the entry block are registered, vacate
       * any live ABI/fixed inputs occupying the reserved scratch prefix
       * [0 .. reserved - 1] so later spill/fill/high-imm/pcopy lowering can
       * never clobber a live input. */
      if (!vacated_reserved) {
         vacated_reserved = true;
         uint64_t empty_forbid[2] = {0, 0};
         for (unsigned r = 0; r < rctx->reserved && r < AGX_APPLE9_GPR_COUNT;
              ++r) {
            if (gpr_test(rctx->used_gprs, r) &&
                rctx->reg_to_ssa[r] != AGX_APPLE9_VREG_INVALID) {
               if (!a9_evict_register(rctx, r, empty_forbid, ins))
                  return false;
               gpr_set(rctx->used_gprs, r);
            }
         }
      }

      /* Rewrite source references through active split mapping. */
      if (util_dynarray_num_elements(&rctx->active_splits, uint32_t) > 0) {
         unsigned ns = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t sv = agx_apple9_vir_src(ins, s);
            uint32_t mapped = a9_resolve_cur(rctx, sv);
            if (mapped != sv)
               *agx_apple9_vir_src_ptr(ins, s) = mapped;
         }
         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < rctx->program->live_out_count; ++k)
               rctx->program->live_out[k] =
                  a9_resolve_cur(rctx, rctx->program->live_out[k]);
         }
      }

      /* Handle VIR_SPLIT (zero-copy coalescing when source tuple dies). */
      if (ins->op == AGX_APPLE9_VIR_SPLIT) {
         unsigned comps = ins->dest_components;
         uint32_t s0 = ins->src[0];
         if (agx_apple9_vir_src_killed(ins, 0)) {
            for (unsigned c = 0; c < comps; ++c) {
               unsigned sr = rctx->phys[s0 + c];
               if (rctx->reg_to_ssa[sr] == s0 + c) {
                  gpr_clear(rctx->used_gprs, sr);
                  rctx->reg_to_ssa[sr] = AGX_APPLE9_VREG_INVALID;
               }
               a9_assign_gpr_val(rctx, ins->dest + c, sr);
               if (!(ins->dead_dest_mask & (1u << c))) {
                  gpr_set(rctx->used_gprs, sr);
                  rctx->reg_to_ssa[sr] = ins->dest + c;
               }
            }
         } else {
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->dead_dest_mask & (1u << c)) {
                  a9_assign_gpr_val(rctx, ins->dest + c, rctx->phys[s0 + c]);
                  continue;
               }
               unsigned moves = 0;
               unsigned chosen = a9_choose_window(
                  rctx, ins->dest + c, 1, rctx->reserved, rctx->bound - 1u, 1,
                  UINT_MAX, &moves);
               if (chosen == UINT_MAX) {
                  rctx->needs_spill_headroom = true;
                  rctx->error = "no free GPR for live VIR_SPLIT lane";
                  return false;
               }
               if (moves > 0 && !a9_vacate_window(rctx, chosen, 1, ins))
                  return false;
               gpr_set(rctx->used_gprs, chosen);
               rctx->reg_to_ssa[chosen] = ins->dest + c;
               a9_assign_gpr_val(rctx, ins->dest + c, chosen);
            }
         }
         continue;
      }

      /* Handle VIR_COLLECT zero-copy bypass when all source lanes already
       * occupy consecutive GPRs in a legal window and the tuple stays local
       * to this block. */
      if (ins->op == AGX_APPLE9_VIR_COLLECT &&
          ins->dest != AGX_APPLE9_VREG_INVALID &&
          ins->dest_components >= 2 && ins->dest_components <= 4 &&
          ins->nr_srcs == ins->dest_components &&
          !a9_is_pub_val(rctx->program, ins->dest)) {
         unsigned w = ins->dest_components;
         bool can_bypass = true;
         unsigned base_r = AGX_APPLE9_PHYS_INVALID;
         for (unsigned k = 0; k < w && can_bypass; ++k) {
            uint32_t sv = ins->src[k];
            uint32_t dv = ins->dest + k;
            if (a9_is_pub_val(rctx->program, sv) ||
                apple9_ra_value_is_mem(rctx->spill, sv) ||
                rctx->phys[sv] >= AGX_APPLE9_GPR_COUNT ||
                rctx->reg_to_ssa[rctx->phys[sv]] != sv ||
                rctx->program->output == dv ||
                u_sparse_bitset_test(&rctx->liveness->live_out[bi], dv)) {
               can_bypass = false;
               break;
            }
            if (k == 0)
               base_r = rctx->phys[sv];
            else if (rctx->phys[sv] != base_r + k)
               can_bypass = false;
         }
         for (unsigned lo = 0; lo < rctx->program->live_out_count && can_bypass;
              ++lo) {
            uint32_t lov = rctx->program->live_out[lo];
            if (lov >= ins->dest && lov < ins->dest + w)
               can_bypass = false;
         }
         if (can_bypass && base_r + w <= rctx->bound) {
            /* Ensure no consumer splits, returns, or clobbers the collected
             * tuple (clobbered operands require a private copy so live sources
             * are not overwritten by hardware). */
            for (struct agx_apple9_vir_instr *u = next_ins; u && can_bypass;
                 u = agx_apple9_instr_next(u)) {
               unsigned uns = agx_apple9_vir_nr_srcs(u);
               if (u->op == AGX_APPLE9_VIR_SPLIT ||
                   u->op == AGX_APPLE9_VIR_ABI_RETURN) {
                  for (unsigned s = 0; s < uns; ++s) {
                     uint32_t usv = agx_apple9_vir_src(u, s);
                     if (usv >= ins->dest && usv < ins->dest + w) {
                        can_bypass = false;
                        break;
                     }
                  }
                  continue;
               }
               for (unsigned s = 0; s < uns && can_bypass; ) {
                  unsigned sw = apple9_ra_source_width(u, s);
                  const struct agx_apple9_operand_constraint *op_c =
                     apple9_ra_source_constraint(u, s);
                  if (op_c && (op_c->flags & AGX_APPLE9_OPERAND_CLOBBER)) {
                     for (unsigned c = 0; c < sw && s + c < uns; ++c) {
                        uint32_t usv = agx_apple9_vir_src(u, s + c);
                        if (usv >= ins->dest && usv < ins->dest + w) {
                           can_bypass = false;
                           break;
                        }
                     }
                  }
                  s += sw;
               }
            }
         } else {
            can_bypass = false;
         }

         if (can_bypass) {
            for (unsigned k = 0; k < w; ++k) {
               uint32_t sv = ins->src[k];
               uint32_t dv = ins->dest + k;
               bool src_killed_here = agx_apple9_vir_src_killed(ins, k);
               struct agx_apple9_vir_instr *orig_kill_ins = NULL;
               struct agx_apple9_vir_instr *last_dest_use = NULL;
               bool seen_orig_kill_before_last_dest = false;

               for (struct agx_apple9_vir_instr *u = next_ins; u;
                    u = agx_apple9_instr_next(u)) {
                  unsigned uns = agx_apple9_vir_nr_srcs(u);
                  bool u_had_orig_kill = false;
                  bool u_used_dest = false;
                  for (unsigned s = 0; s < uns && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
                     uint32_t usv = agx_apple9_vir_src(u, s);
                     if (usv == sv && agx_apple9_vir_src_killed(u, s))
                        u_had_orig_kill = true;
                     if (usv == dv)
                        u_used_dest = true;
                  }
                  if (u_had_orig_kill)
                     orig_kill_ins = u;
                  if (u_used_dest) {
                     last_dest_use = u;
                     if (orig_kill_ins)
                        seen_orig_kill_before_last_dest = true;
                  }
               }

               bool transfer_kill_to_last_dest =
                  src_killed_here || seen_orig_kill_before_last_dest;
               if (!src_killed_here && seen_orig_kill_before_last_dest &&
                   orig_kill_ins) {
                  unsigned uns = agx_apple9_vir_nr_srcs(orig_kill_ins);
                  for (unsigned s = 0; s < uns && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
                     if (agx_apple9_vir_src(orig_kill_ins, s) == sv)
                        agx_apple9_vir_set_src_killed(orig_kill_ins, s, false);
                  }
               }

               for (struct agx_apple9_vir_instr *u = next_ins; u;
                    u = agx_apple9_instr_next(u)) {
                  unsigned uns = agx_apple9_vir_nr_srcs(u);
                  bool u_is_load =
                     u->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT ||
                     u->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER;
                  for (unsigned s = 0; s < uns && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
                     if (agx_apple9_vir_src(u, s) == dv) {
                        *agx_apple9_vir_src_ptr(u, s) = sv;
                        if (u_is_load)
                           u->retain_mask |= (uint16_t)BITFIELD_BIT(s);
                        if (!transfer_kill_to_last_dest || u != last_dest_use)
                           agx_apple9_vir_set_src_killed(u, s, false);
                     }
                  }
                  if (transfer_kill_to_last_dest && u == last_dest_use) {
                     for (unsigned s = 0; s < uns && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
                        if (agx_apple9_vir_src(u, s) == sv)
                           agx_apple9_vir_set_src_killed(u, s, true);
                     }
                  }
               }

               if (src_killed_here && !last_dest_use) {
                  unsigned sr = rctx->phys[sv];
                  if (rctx->reg_to_ssa[sr] == sv) {
                     gpr_clear(rctx->used_gprs, sr);
                     rctx->reg_to_ssa[sr] = AGX_APPLE9_VREG_INVALID;
                  }
               }
               a9_assign_gpr_val(rctx, dv, base_r + k);
            }
            agx_apple9_vir_remove(rctx->program, ins);
            continue;
         }
      }

      /* Retire any tracked async dead-lane groups whose live result is now
       * consumed (or retired by a barrier/wait/CF). */
      a9_retire_async_for_instr(rctx, ins);

      /* Place constrained machine sources first (relocating if needed). */
      if (ins->encoding != AGX_APPLE9_ENC_PSEUDO) {
         if (!a9_assign_constrained_sources(rctx, ins))
            return false;
      }

      /* Assign destination(s) (reusing killed source GPR when legal). */
      uint16_t reused_sources = 0;
      if (!a9_assign_dest(rctx, ins, &reused_sources))
         return false;

      /* Release killed sources not taken over by the destination. */
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
      for (unsigned s = 0; s < nr_srcs && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
         if (!agx_apple9_vir_src_killed(ins, s))
            continue;
         if ((reused_sources >> s) & 1u)
            continue;
         uint32_t sv = ins->src[s];
         if (a9_is_pub_val(rctx->program, sv) ||
             apple9_ra_value_is_mem(rctx->spill, sv))
            continue;
         unsigned sr = rctx->phys[sv];
         if (sr < AGX_APPLE9_GPR_COUNT && rctx->reg_to_ssa[sr] == sv) {
            gpr_clear(rctx->used_gprs, sr);
            rctx->reg_to_ssa[sr] = AGX_APPLE9_VREG_INVALID;
         }
      }

      /* Release dead destination lanes immediately (e.g. unused load channels). */
      if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0 &&
          !a9_is_pub_val(rctx->program, ins->dest) &&
          !apple9_ra_value_is_mem(rctx->spill, ins->dest)) {
         /* Clear any pending async dead-lane entries overwritten by this dest. */
         bool dead_cleared = false;
         for (unsigned c = 0; c < ins->dest_components && c < 8; ++c) {
            unsigned dr = rctx->phys[ins->dest + c];
            if (dr < AGX_APPLE9_GPR_COUNT &&
                gpr_test(rctx->async_dead_gprs, dr)) {
               for (unsigned i = 0; i < ARRAY_SIZE(rctx->async_pending); ++i) {
                  if (rctx->async_pending[i].active &&
                      gpr_test(rctx->async_pending[i].dead, dr)) {
                     rctx->async_pending[i].active = false;
                     dead_cleared = true;
                  }
               }
            }
         }
         if (dead_cleared)
            a9_recompute_async_dead(rctx);

         uint64_t live_mask[2] = {0, 0};
         uint64_t dead_mask[2] = {0, 0};
         bool any_dead = false;
         for (unsigned c = 0; c < ins->dest_components && c < 8; ++c) {
            unsigned dr = rctx->phys[ins->dest + c];
            if (!(ins->dead_dest_mask & (1u << c))) {
               if (dr < AGX_APPLE9_GPR_COUNT)
                  gpr_set(live_mask, dr);
               continue;
            }
            if (dr < AGX_APPLE9_GPR_COUNT &&
                rctx->reg_to_ssa[dr] == ins->dest + c) {
               gpr_clear(rctx->used_gprs, dr);
               rctx->reg_to_ssa[dr] = AGX_APPLE9_VREG_INVALID;
               gpr_set(dead_mask, dr);
               any_dead = true;
            }
         }
         if (any_dead && apple9_ra_async(ins)) {
            unsigned slot = rctx->async_next++ % ARRAY_SIZE(rctx->async_pending);
            rctx->async_pending[slot].live[0] = live_mask[0];
            rctx->async_pending[slot].live[1] = live_mask[1];
            rctx->async_pending[slot].dead[0] = dead_mask[0];
            rctx->async_pending[slot].dead[1] = dead_mask[1];
            rctx->async_pending[slot].active = true;
            a9_recompute_async_dead(rctx);
         }
      }
   }

   /* If block 0 had only ENTRY_REGs (or pre-defined fixed inputs), still vacate
    * any live values occupying [0 .. reserved - 1] before leaving block 0. */
   if (!vacated_reserved) {
      vacated_reserved = true;
      uint64_t empty_forbid[2] = {0, 0};
      for (unsigned r = 0; r < rctx->reserved && r < AGX_APPLE9_GPR_COUNT;
           ++r) {
         if (gpr_test(rctx->used_gprs, r) &&
             rctx->reg_to_ssa[r] != AGX_APPLE9_VREG_INVALID) {
            if (!a9_evict_register(rctx, r, empty_forbid, NULL))
               return false;
            gpr_set(rctx->used_gprs, r);
         }
      }
   }

   /* Rewrite program->output if this is an exit block and output was split. */
   if (!block->cfg_succs[0] && !block->cfg_succs[1] &&
       rctx->program->output != AGX_APPLE9_VREG_INVALID) {
      rctx->program->output = a9_resolve_cur(rctx, rctx->program->output);
   }

   /* Save live-out splits for successors. */
   util_dynarray_foreach(&rctx->active_splits, uint32_t, orig_vp) {
      uint32_t orig_v = *orig_vp;
      uint32_t cur_v = rctx->cur_ssa[orig_v];
      if (cur_v != orig_v &&
          u_sparse_bitset_test(&rctx->liveness->live_out[bi], orig_v)) {
         struct a9_split_entry se = {orig_v, cur_v};
         util_dynarray_append(&rctx->block_splits_out[bi], se);
      }
   }

   rctx->block_visited[bi] = true;

   /* Update successor VIR_PHIs (including loop backedges) with active splits. */
   for (unsigned s = 0; s < 2; ++s) {
      struct agx_apple9_block *succ = block->cfg_succs[s];
      if (!succ || !succ->cfg_active)
         continue;
      unsigned pred_idx = agx_apple9_block_pred_index(succ, block);
      assert(pred_idx != UINT_MAX);

      for (struct agx_apple9_vir_instr *phi = agx_apple9_block_first_instr(succ);
           phi; phi = agx_apple9_instr_next(phi)) {
         if (a9_is_block_entry_prefix(phi))
            continue;
         if (phi->op != AGX_APPLE9_VIR_PHI)
            break;
         uint32_t sv = agx_apple9_vir_src(phi, pred_idx);
         uint32_t mapped = a9_resolve_cur(rctx, sv);
         if (mapped != sv)
            *agx_apple9_vir_src_ptr(phi, pred_idx) = mapped;
      }

      /* If `succ` was already visited (backedge to loop header), reconcile any
       * loop-invariant live-in value that got evicted inside the loop body. */
      if (rctx->block_visited[succ->cfg_index]) {
         util_dynarray_foreach(&rctx->block_splits_out[bi],
                               struct a9_split_entry, se) {
            if (!u_sparse_bitset_test(&rctx->liveness->live_in[succ->cfg_index],
                                      se->orig_v))
               continue;
            /* Skip if orig_v is already defined by an explicit phi in succ. */
            bool is_phi_dest = false;
            for (struct agx_apple9_vir_instr *phi =
                    agx_apple9_block_first_instr(succ);
                 phi; phi = agx_apple9_instr_next(phi)) {
               if (a9_is_block_entry_prefix(phi))
                  continue;
               if (phi->op != AGX_APPLE9_VIR_PHI)
                  break;
               if (phi->dest == se->orig_v)
                  is_phi_dest = true;
            }
            if (!is_phi_dest) {
               if (!a9_splice_loop_header_phi(rctx, succ, se->orig_v))
                  return false;
            }
         }
      }
   }

   return true;
}

/* =========================================================================
 * Physical VIR Lowering & Scoreboard Finalization
 * ========================================================================= */

static bool
a9_init_physical_program(struct agx_apple9_vir_program *out)
{
   agx_apple9_vir_init(out);
   for (unsigned r = 0; r < A9_PUB_BASE + AGX_APPLE9_PUBLICATION_COUNT; ++r) {
      if (agx_apple9_vir_input(out, r < A9_PUB_BASE ? r : 0) != r)
         return false;
      if (r >= A9_PUB_BASE)
         out->fixed_phys[r] = AGX_APPLE9_PHYS_INVALID;
   }
   out->phys = malloc(out->value_count);
   out->publication = calloc(out->value_count, sizeof(bool));
   out->phys_capacity = out->publication_capacity = out->value_count;
   if (!out->phys || !out->publication)
      return false;
   for (unsigned r = 0; r < out->value_count; ++r) {
      out->phys[r] = (uint8_t)(r < A9_PUB_BASE ? r : r - A9_PUB_BASE);
      out->publication[r] = (r >= A9_PUB_BASE);
   }
   return true;
}

static bool
a9_emit_phys(struct agx_apple9_vir_program *out,
             const struct agx_apple9_vir_instr *ins)
{
   if (!agx_apple9_vir_emit_side_effect(out, ins->op, ins->encoding, NULL, 0, 0))
      return false;
   *out->instructions[out->instruction_count - 1] = *ins;
   for (unsigned c = 0; c < ins->dest_components; ++c) {
      if (ins->dest < A9_PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->dest + c);
   }
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (ins->src[s] < A9_PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->src[s]);
   }
   return true;
}

static struct agx_apple9_vir_instr
a9_make_phys_instr(enum agx_apple9_vir_opcode op,
                   enum agx_apple9_encoding encoding, uint32_t dest)
{
   return (struct agx_apple9_vir_instr){
      .op = op,
      .encoding = encoding,
      .dest = dest,
      .dest_components = (dest != AGX_APPLE9_VREG_INVALID) ? 1u : 0u,
      .producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
      .scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
   };
}

static bool
a9_emit_phys_copy(struct agx_apple9_vir_program *out, unsigned dst,
                  unsigned src, unsigned *copy_counter)
{
   if (dst == src)
      return true;
   struct agx_apple9_vir_instr ins = a9_make_phys_instr(
      AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, dst);
   ins.src[0] = ins.src[1] = src;
   ins.nr_srcs = 2;
   ins.live_after_mask = 3;
   if (copy_counter)
      (*copy_counter)++;
   return a9_emit_phys(out, &ins);
}

static bool
a9_emit_phys_xor(struct agx_apple9_vir_program *out, unsigned dst, unsigned a,
                 unsigned b)
{
   struct agx_apple9_vir_instr ins = a9_make_phys_instr(
      AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, dst);
   ins.src[0] = a;
   ins.src[1] = b;
   ins.nr_srcs = 2;
   ins.live_after_mask = 3;
   return a9_emit_phys(out, &ins);
}

static bool
a9_emit_phys_swap(struct agx_apple9_vir_program *out, unsigned a, unsigned b,
                  unsigned *copy_counter)
{
   if (copy_counter)
      (*copy_counter) += 3;
   return a9_emit_phys_xor(out, a, a, b) && a9_emit_phys_xor(out, b, b, a) &&
          a9_emit_phys_xor(out, a, a, b);
}

static bool
a9_emit_phys_imm(struct agx_apple9_vir_program *out, unsigned dst,
                 uint32_t value, unsigned *copy_counter,
                 bool *needs_copy_reg_out)
{
   if (dst >= 64u) {
      if (needs_copy_reg_out)
         *needs_copy_reg_out = true;
      if (!out->reserved_gprs[A9_COPY_REG])
         return false;
      return a9_emit_phys_imm(out, A9_COPY_REG, value, copy_counter,
                              needs_copy_reg_out) &&
             a9_emit_phys_copy(out, dst, A9_COPY_REG, copy_counter);
   }
   struct agx_apple9_vir_instr ins = a9_make_phys_instr(
      AGX_APPLE9_VIR_IMM,
      (dst < 16u && value < 0x80u) ? AGX_APPLE9_ENC_MOV_IMM_COMPACT
                                   : AGX_APPLE9_ENC_MOV_IMM32,
      dst);
   ins.immediate = value;
   return a9_emit_phys(out, &ins);
}

static bool
a9_emit_phys_spill_load(struct agx_apple9_vir_program *out, unsigned dst,
                        unsigned word, unsigned *copy_counter,
                        unsigned *spill_counter)
{
   if (spill_counter)
      (*spill_counter)++;
   if (word >= AGX_APPLE9_FILL_WORD_COUNT) {
      /* Compact FILL aliases above 4 KiB (1024 words). Use indexed
       * PRIVATE_LOAD via reserved COPY_REG=1, then copy to dst. */
      struct agx_apple9_vir_instr load = a9_make_phys_instr(
         AGX_APPLE9_VIR_PRIVATE_LOAD, AGX_APPLE9_ENC_PRIVATE_LOAD, A9_COPY_REG);
      load.src[0] = A9_COPY_REG;
      load.nr_srcs = 1;
      load.immediate = word;
      load.producer_scoreboard_slot = A9_SPILL_TAG;
      load.spill_access = true;
      return a9_emit_phys_imm(out, A9_COPY_REG, 0, copy_counter, NULL) &&
             a9_emit_phys(out, &load) &&
             a9_emit_phys_copy(out, dst, A9_COPY_REG, copy_counter);
   }
   struct agx_apple9_vir_instr load = a9_make_phys_instr(
      AGX_APPLE9_VIR_SPILL_LOAD, AGX_APPLE9_ENC_SPILL_LOAD, dst);
   load.immediate = word;
   load.producer_scoreboard_slot = A9_SPILL_TAG;
   load.spill_access = true;
   return a9_emit_phys(out, &load);
}

static bool
a9_emit_phys_spill_store(struct agx_apple9_vir_program *out, unsigned src,
                         unsigned word, bool killed, unsigned *copy_counter,
                         unsigned *spill_counter)
{
   if (spill_counter)
      (*spill_counter)++;
   if (word >= AGX_APPLE9_SAVE_WORD_COUNT) {
      /* Compact SAVE aliases above 8 KiB (2048 words). Use indexed
       * PRIVATE_STORE via reserved COPY_REG=1 & SPILL_INDEX_REG=2. */
      struct agx_apple9_vir_instr store = a9_make_phys_instr(
         AGX_APPLE9_VIR_PRIVATE_STORE, AGX_APPLE9_ENC_PRIVATE_STORE,
         AGX_APPLE9_VREG_INVALID);
      store.src[0] = A9_COPY_REG;
      store.src[1] = A9_SPILL_INDEX_REG;
      store.nr_srcs = 2;
      store.live_after_mask = 2;
      store.immediate = word;
      store.spill_access = true;
      return a9_emit_phys_copy(out, A9_COPY_REG, src, copy_counter) &&
             a9_emit_phys_imm(out, A9_SPILL_INDEX_REG, 0, copy_counter, NULL) &&
             a9_emit_phys(out, &store);
   }
   /* Compact SAVE destroys its input: skip COPY_REG on killed last use! */
   struct agx_apple9_vir_instr save = a9_make_phys_instr(
      AGX_APPLE9_VIR_SPILL_STORE, AGX_APPLE9_ENC_SPILL_STORE,
      AGX_APPLE9_VREG_INVALID);
   save.src[0] = killed ? src : A9_COPY_REG;
   save.nr_srcs = 1;
   save.immediate = word;
   save.producer_scoreboard_slot = A9_SPILL_TAG;
   save.spill_access = true;
   return (killed || a9_emit_phys_copy(out, A9_COPY_REG, src, copy_counter)) &&
          a9_emit_phys(out, &save);
}

struct a9_phys_pcopy_ctx {
   struct agx_apple9_vir_program *out;
   unsigned *copy_counter;
   unsigned *spill_counter;
   bool *needs_copy_reg;
};

static bool
a9_phys_pcopy_emit(void *vctx, const struct apple9_ra_pcopy_op *op)
{
   struct a9_phys_pcopy_ctx *ctx = vctx;
   switch (op->kind) {
   case APPLE9_RA_PCOPY_MOV:
      if (op->dst.kind == APPLE9_RA_LOC_GPR && op->src.kind == APPLE9_RA_LOC_GPR)
         return a9_emit_phys_copy(ctx->out, op->dst.value, op->src.value,
                                  ctx->copy_counter);
      if (op->dst.kind == APPLE9_RA_LOC_GPR && op->src.kind == APPLE9_RA_LOC_MEM)
         return a9_emit_phys_spill_load(ctx->out, op->dst.value, op->src.value,
                                        ctx->copy_counter, ctx->spill_counter);
      if (op->dst.kind == APPLE9_RA_LOC_MEM && op->src.kind == APPLE9_RA_LOC_GPR)
         return a9_emit_phys_spill_store(ctx->out, op->src.value, op->dst.value,
                                         false, ctx->copy_counter,
                                         ctx->spill_counter);
      return false;
   case APPLE9_RA_PCOPY_MOV_IMM:
      return a9_emit_phys_imm(ctx->out, op->dst.value, op->src.value,
                              ctx->copy_counter, ctx->needs_copy_reg);
   case APPLE9_RA_PCOPY_SWAP:
      return a9_emit_phys_swap(ctx->out, op->dst.value, op->src.value,
                               ctx->copy_counter);
   }
   return false;
}

static bool
a9_emit_phys_parallel_copies(struct agx_apple9_vir_program *out,
                             const struct apple9_ra_copy *copies,
                             unsigned num_copies, unsigned *copy_counter,
                             unsigned *spill_counter, bool *needs_copy_reg)
{
   if (num_copies == 0)
      return true;

   struct apple9_ra_pcopy_opts opts = {
      .live_gprs = {~0ull, BITFIELD64_MASK(AGX_APPLE9_GPR_COUNT - 64u)},
      .scratch_gpr = out->reserved_gprs[APPLE9_RA_DEFAULT_SWAP_REG0]
                        ? APPLE9_RA_DEFAULT_SWAP_REG0
                        : (out->reserved_gprs[A9_COPY_REG]
                              ? A9_COPY_REG
                              : AGX_APPLE9_PHYS_INVALID),
      .scratch_gpr2 = out->reserved_gprs[APPLE9_RA_DEFAULT_SWAP_REG1]
                         ? APPLE9_RA_DEFAULT_SWAP_REG1
                         : AGX_APPLE9_PHYS_INVALID,
      .max_temp_gpr = AGX_APPLE9_GPR_COUNT,
      .mov_imm_max_gpr = 64u,
      .prefer_swap_2cycle = false,
   };
   struct a9_phys_pcopy_ctx ctx = {
      .out = out,
      .copy_counter = copy_counter,
      .spill_counter = spill_counter,
      .needs_copy_reg = needs_copy_reg,
   };
   return apple9_ra_lower_parallel_copies(copies, num_copies, &opts,
                                          a9_phys_pcopy_emit, &ctx);
}

static bool
a9_remove_empty_else(struct agx_apple9_vir_program *p)
{
   struct agx_apple9_vir_instr **remove = NULL;
   unsigned count = 0;
   for (unsigned i = 0; i + 1 < p->instruction_count; ++i) {
      if (p->instructions[i]->op != AGX_APPLE9_VIR_EXEC_MASK_ELSE ||
          p->instructions[i + 1]->op != AGX_APPLE9_VIR_EXEC_MASK_POP)
         continue;
      if (!remove && !(remove = malloc(p->instruction_count * sizeof(*remove))))
         return false;
      remove[count++] = p->instructions[i];
   }
   if (count)
      agx_apple9_vir_remove_many(p, remove, count);
   free(remove);
   return true;
}

static bool
a9_lower_to_physical_vir(struct agx_apple9_vir_program *program,
                         const struct apple9_ra_spill_state *spill,
                         const uint16_t *mem_slot, unsigned frame_words,
                         unsigned reserved, unsigned peak_gprs,
                         bool skip_physical_schedule,
                         struct apple9_ra_assign_result *result,
                         const char **error)
{
   struct agx_apple9_vir_program out;
   if (!a9_init_physical_program(&out)) {
      agx_apple9_vir_finish(&out);
      *error = "out of memory initializing canonical physical VIR";
      return false;
   }

   for (unsigned r = 0; r < reserved && r < AGX_APPLE9_GPR_COUNT; ++r)
      out.reserved_gprs[r] = true;

   unsigned bcount = program->last_block ? (program->last_block->index + 1u) : 0u;
   struct agx_apple9_block **blocks = calloc(MAX2(bcount, 1u), sizeof(*blocks));
   if (!blocks) {
      agx_apple9_vir_finish(&out);
      *error = "out of memory allocating physical block table";
      return false;
   }

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      blocks[b->index] = agx_apple9_block_create(&out);
      if (!blocks[b->index]) {
         free(blocks);
         agx_apple9_vir_finish(&out);
         *error = "out of memory creating physical block";
         return false;
      }
      blocks[b->index]->nir = b->nir;
      blocks[b->index]->nir_exit = b->nir_exit;
   }

   const unsigned base_words = apple9_ra_spill_base_words(program);
   const unsigned spill_bytes = frame_words ? ALIGN_POT(frame_words * 4u, 16u) : 0u;
   unsigned private_base_words = spill_bytes / 4u;
   if (!spill_bytes && program->private_size && base_words)
      private_base_words = base_words;

   /* Track scalar VIR_IMM definitions so GPR-to-GPR copies of constants
    * (VIR_MOV, VIR_COLLECT, ABI_RETURN) can lower to compact/32-bit MOV_IMM
    * rather than 10-byte IOR copies whenever the destination is < 64. */
   const unsigned vcount = program->value_count;
   bool *is_imm = calloc(MAX2(vcount, 1u), sizeof(*is_imm));
   uint32_t *imm_val = calloc(MAX2(vcount, 1u), sizeof(*imm_val));
   if (!is_imm || !imm_val) {
      free(is_imm);
      free(imm_val);
      free(blocks);
      agx_apple9_vir_finish(&out);
      *error = "out of memory allocating physical immediate map";
      return false;
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->target != APPLE9_RA_PHYS_OP_TAG &&
             ins->op == AGX_APPLE9_VIR_IMM && ins->dest_components == 1 &&
             ins->dest < vcount) {
            is_imm[ins->dest] = true;
            imm_val[ins->dest] = ins->immediate;
         }
      }
   }

   unsigned num_copies = 0, num_spills = 0;
   bool needs_copy_reg = false;
   bool ok = true;

   for (struct agx_apple9_block *b = program->blocks; b && ok; b = b->next) {
      agx_apple9_block_begin(&out, blocks[b->index]);

      if (!b->cfg_active) {
         /* Inactive synthetic control blocks are emitted verbatim; pruned
          * unreachable NIR blocks skip any dead SSA instructions. */
         for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
              ins && ok; ins = agx_apple9_instr_next(ins)) {
            if (b->nir && (ins->dest != AGX_APPLE9_VREG_INVALID ||
                           agx_apple9_vir_nr_srcs(ins) > 0))
               continue;
            struct agx_apple9_vir_instr p_ins = *ins;
            if (p_ins.branch_target)
               p_ins.branch_target = blocks[p_ins.branch_target->index];
            ok = a9_emit_phys(&out, &p_ins);
         }
         continue;
      }

      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins && ok; ins = agx_apple9_instr_next(ins)) {
         /* Skip already-lowered phis and zero-byte entry preloads. */
         if (ins->op == AGX_APPLE9_VIR_PHI ||
             ins->op == AGX_APPLE9_VIR_ENTRY_REG)
            continue;

         /* Tagged physical ops emitted by apple9_ra_pcopy.c. */
         if (ins->target == APPLE9_RA_PHYS_OP_TAG) {
            if (ins->op == AGX_APPLE9_VIR_MOV) {
               ok = a9_emit_phys_copy(&out, ins->dest, ins->src[0], &num_copies);
            } else if (ins->op == AGX_APPLE9_VIR_SWAP) {
               ok = a9_emit_phys_swap(&out, ins->src[0], ins->src[1],
                                      &num_copies);
            } else if (ins->op == AGX_APPLE9_VIR_IMM) {
               ok = a9_emit_phys_imm(&out, ins->dest, ins->immediate,
                                     &num_copies, &needs_copy_reg);
            } else if (ins->op == AGX_APPLE9_VIR_SPILL_LOAD) {
               ok = a9_emit_phys_spill_load(&out, ins->dest, ins->immediate,
                                            &num_copies, &num_spills);
            } else if (ins->op == AGX_APPLE9_VIR_SPILL_STORE) {
               ok = a9_emit_phys_spill_store(&out, ins->src[0], ins->immediate,
                                             false, &num_copies, &num_spills);
            }
            continue;
         }

         /* SSA VIR_MOV (GPR<->GPR, GPR<->MEM, MEM<->MEM). */
         if (ins->op == AGX_APPLE9_VIR_MOV) {
            uint32_t dv = ins->dest;
            uint32_t sv = ins->src[0];
            unsigned w = MAX2(ins->dest_components, 1u);
            bool d_mem = apple9_ra_value_is_mem(spill, dv);
            bool s_mem = apple9_ra_value_is_mem(spill, sv);

            if (!d_mem && !s_mem) {
               for (unsigned c = 0; c < w && ok; ++c) {
                  unsigned dst_r = program->phys[dv + c];
                  unsigned src_r = program->phys[sv + c];
                  if (dst_r != src_r && sv + c < vcount && is_imm[sv + c] &&
                      dst_r < 64u) {
                     ok = a9_emit_phys_imm(&out, dst_r, imm_val[sv + c],
                                           &num_copies, &needs_copy_reg);
                  } else {
                     ok = a9_emit_phys_copy(&out, dst_r, src_r, &num_copies);
                  }
               }
            } else if (d_mem && !s_mem) {
               bool killed = agx_apple9_vir_src_killed(ins, 0);
               unsigned word = mem_slot[dv];
               for (unsigned c = 0; c < w && ok; ++c)
                  ok = a9_emit_phys_spill_store(&out, program->phys[sv + c],
                                                word + c, killed, &num_copies,
                                                &num_spills);
            } else if (!d_mem && s_mem) {
               unsigned word = mem_slot[sv];
               for (unsigned c = 0; c < w && ok; ++c)
                  ok = a9_emit_phys_spill_load(&out, program->phys[dv + c],
                                               word + c, &num_copies,
                                               &num_spills);
            } else {
               unsigned dw = mem_slot[dv], sw = mem_slot[sv];
               for (unsigned c = 0; c < w && ok; ++c) {
                  ok = a9_emit_phys_spill_load(&out, A9_COPY_REG, sw + c,
                                               &num_copies, &num_spills) &&
                       a9_emit_phys_spill_store(&out, A9_COPY_REG, dw + c, true,
                                                &num_copies, &num_spills);
               }
            }
            continue;
         }

         /* SSA VIR_COLLECT -> parallel copy into contiguous dest tuple. */
         if (ins->op == AGX_APPLE9_VIR_COLLECT) {
            unsigned w = ins->dest_components;
            unsigned base = program->phys[ins->dest];
            struct apple9_ra_copy copies[4];
            for (unsigned c = 0; c < w; ++c) {
               uint32_t sv = ins->src[c];
               unsigned dr = base + c;
               unsigned sr = program->phys[sv];
               struct apple9_ra_loc sloc =
                  (dr != sr && sv < vcount && is_imm[sv] && dr < 64u)
                     ? apple9_ra_loc_imm(imm_val[sv])
                     : apple9_ra_loc_gpr(sr);
               copies[c] = (struct apple9_ra_copy){
                  .dst = apple9_ra_loc_gpr(dr),
                  .src = sloc,
                  .count = 1,
               };
            }
            ok = a9_emit_phys_parallel_copies(&out, copies, w, &num_copies,
                                              &num_spills, &needs_copy_reg);
            continue;
         }

         /* SSA VIR_SPLIT -> parallel copy for any non-coalesced live lanes. */
         if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            unsigned w = ins->dest_components;
            unsigned sbase = program->phys[ins->src[0]];
            struct apple9_ra_copy copies[8];
            unsigned nc = 0;
            for (unsigned c = 0; c < w; ++c) {
               if (ins->dead_dest_mask & (1u << c))
                  continue;
               copies[nc++] = (struct apple9_ra_copy){
                  .dst = apple9_ra_loc_gpr(program->phys[ins->dest + c]),
                  .src = apple9_ra_loc_gpr(sbase + c),
                  .count = 1,
               };
            }
            ok = a9_emit_phys_parallel_copies(&out, copies, nc, &num_copies,
                                              &num_spills, &needs_copy_reg);
            continue;
         }

         /* SSA VIR_IMM -> compact / 32-bit immediate (or COPY_REG bridge). */
         if (ins->op == AGX_APPLE9_VIR_IMM) {
            ok = a9_emit_phys_imm(&out, program->phys[ins->dest],
                                  ins->immediate, &num_copies, &needs_copy_reg);
            continue;
         }

         /* SSA VIR_ABI_RETURN -> lower live_out exports before returning. */
         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN &&
             program->live_out_count > 0) {
            struct apple9_ra_copy *copies =
               calloc(program->live_out_count, sizeof(*copies));
            if (!copies) {
               ok = false;
               break;
            }
            unsigned nc = 0;
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               if (program->live_out_reg[k] == AGX_APPLE9_PHYS_INVALID)
                  continue;
               uint32_t sv = program->live_out[k];
               unsigned dr = program->live_out_reg[k];
               unsigned sr = program->phys[sv];
               struct apple9_ra_loc sloc =
                  (dr != sr && sv < vcount && is_imm[sv] && dr < 64u)
                     ? apple9_ra_loc_imm(imm_val[sv])
                     : apple9_ra_loc_gpr(sr);
               copies[nc++] = (struct apple9_ra_copy){
                  .dst = apple9_ra_loc_gpr(dr),
                  .src = sloc,
                  .count = 1,
               };
               out.max_phys_gpr = MAX2(out.max_phys_gpr, dr);
            }
            ok = a9_emit_phys_parallel_copies(&out, copies, nc, &num_copies,
                                              &num_spills, &needs_copy_reg);
            free(copies);
            if (!ok)
               break;
         }

         /* Ordinary physical machine instruction. */
         struct agx_apple9_vir_instr p_ins = *ins;
         bool pub_dest = ins->dest != AGX_APPLE9_VREG_INVALID &&
                         a9_is_pub_val(program, ins->dest);
         bool dead_scalar_dest =
            ins->dest_components == 1 && !pub_dest && !ins->publication_handoff &&
            (ins->dead_dest_mask & 1u) != 0;

         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
            if (dead_scalar_dest && ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
               p_ins.atomic_discard = true;
               p_ins.dest_components = 0;
               p_ins.dest = AGX_APPLE9_VREG_INVALID;
               p_ins.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
            } else {
               p_ins.dest = pub_dest ? (A9_PUB_BASE + program->phys[ins->dest])
                                     : program->phys[ins->dest];
            }
         } else {
            p_ins.dest = AGX_APPLE9_VREG_INVALID;
         }

         p_ins.live_after_mask = 0;
         for (unsigned s = 0; s < ins->nr_srcs; ++s) {
            uint32_t sv = ins->src[s];
            p_ins.src[s] = a9_is_pub_val(program, sv)
                              ? (A9_PUB_BASE + program->phys[sv])
                              : program->phys[sv];
            if (!agx_apple9_vir_src_killed(ins, s))
               p_ins.live_after_mask |= (uint16_t)BITFIELD_BIT(s);
         }
         p_ins.live_after_mask |= ins->retain_mask;
         p_ins.kill_mask = 0;
         p_ins.dead_dest_mask = 0;

         if (dead_scalar_dest && p_ins.op == AGX_APPLE9_VIR_SHARED_ATOMIC)
            p_ins.dest = p_ins.src[0];

         /* Shader-local private arrays follow the 16B-aligned spill region. */
         if (p_ins.op == AGX_APPLE9_VIR_PRIVATE_LOAD ||
             p_ins.op == AGX_APPLE9_VIR_PRIVATE_STORE)
            p_ins.immediate += private_base_words;

         if (p_ins.branch_target)
            p_ins.branch_target = blocks[p_ins.branch_target->index];
         p_ins.phi_block = NULL;
         p_ins.phi_edge = p_ins.completion_consumer = NULL;
         p_ins.phi_srcs = NULL;
         p_ins.phi_src_count = 0;
         p_ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         p_ins.device_load_index_kind =
            p_ins.encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER
               ? AGX_APPLE9_DEVICE_LOAD_INDEX_NONE
            : (p_ins.live_after_mask & 1u)
               ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
               : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;

         ok = a9_emit_phys(&out, &p_ins);
         if (ok && p_ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
             !p_ins.atomic_discard) {
            struct agx_apple9_vir_instr res = a9_make_phys_instr(
               AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT,
               AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT, AGX_APPLE9_VREG_INVALID);
            res.src[0] = p_ins.dest;
            res.nr_srcs = 1;
            res.live_after_mask = 1;
            res.producer_scoreboard_slot = p_ins.producer_scoreboard_slot;
            ok = a9_emit_phys(&out, &res);
         }
      }
   }

   free(is_imm);
   free(imm_val);
   free(blocks);
   if (!ok) {
      agx_apple9_vir_finish(&out);
      *error = needs_copy_reg
                  ? "high-bank immediate requires COPY_REG reservation"
                  : "failed lowering VIR instructions to physical form";
      if (result)
         result->needs_copy_register = needs_copy_reg;
      return false;
   }

   agx_apple9_vir_reindex(&out);
   if (!a9_remove_empty_else(&out)) {
      agx_apple9_vir_finish(&out);
      *error = "out of memory removing empty else arms";
      return false;
   }

   out.private_size = program->private_size;
   out.spill_slots = frame_words > base_words ? (spill_bytes / 4u - base_words) : 0u;
   out.scratch_size = apple9_ra_spill_scratch_bytes(program, frame_words);
   if (out.scratch_size > AGX_APPLE9_MAX_SCRATCH_BYTES) {
      agx_apple9_vir_finish(&out);
      *error = "Apple9 private arrays and spills exceed 32 KiB scratch frame";
      return false;
   }
   out.peak_live_gprs = MAX2(out.max_phys_gpr + 1u, peak_gprs);
   out.publication_count = program->publication_count;
   out.fragment_shader = program->fragment_shader;
   if (program->output != AGX_APPLE9_VREG_INVALID &&
       program->output < program->value_count) {
      out.output = a9_is_pub_val(program, program->output)
                      ? (A9_PUB_BASE + program->phys[program->output])
                      : program->phys[program->output];
   }
   out.physical = true;
   out.dependencies_finalized = false;

   if (!skip_physical_schedule) {
      if (!agx_apple9_schedule_physical(&out, error)) {
         agx_apple9_vir_finish(&out);
         return false;
      }
   }

   if (result) {
      result->max_phys_gpr = out.max_phys_gpr;
      result->peak_live_gprs = out.peak_live_gprs;
      result->reserved_gprs = reserved;
      result->spill_slots = out.spill_slots;
      result->scratch_size = out.scratch_size;
      result->num_copies_emitted = num_copies;
      result->num_spills_emitted = num_spills;
      result->needs_copy_register = needs_copy_reg;
   }

   agx_apple9_vir_finish(program);
   *program = out;
   for (struct agx_apple9_block *b = program->allocated_blocks; b;
        b = b->allocated_next)
      b->program = program;
   return true;
}

/* Estimate peak simultaneously in-flight dead destination lanes written by
 * multi-component async producers (e.g. vec4 TEXTURE_SAMPLE where only 1-2
 * channels are consumed). Hardware still writes all destination lanes
 * asynchronously, so assign_bound needs room for these dead lanes to avoid
 * forcing WAW WAIT stalls during back-to-back async issue. */
static unsigned
a9_calc_async_dead_gprs(const struct agx_apple9_vir_program *program,
                        const struct apple9_ra_spill_state *spill,
                        void *memctx)
{
   const unsigned vcount = program->value_count;
   if (!vcount)
      return 0;

   BITSET_WORD *used = BITSET_RZALLOC(memctx, vcount);
   if (!used)
      return 0;

   if (program->output < vcount)
      BITSET_SET(used, program->output);
   for (unsigned i = 0; i < program->live_out_count; ++i) {
      if (program->live_out[i] < vcount)
         BITSET_SET(used, program->live_out[i]);
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_SPLIT && ins->nr_srcs >= 1) {
            /* VIR_SPLIT reads the full contiguous dest_components-wide source
             * tuple starting at ins->src[0] (matching liveness). */
            unsigned w = ins->dest_components;
            for (unsigned c = 0; c < w; ++c) {
               uint32_t v = ins->src[0] + c;
               if (v < vcount)
                  BITSET_SET(used, v);
            }
            continue;
         }
         const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t sv = agx_apple9_vir_src(ins, s);
            if (sv < vcount)
               BITSET_SET(used, sv);
         }
      }
   }

   unsigned peak_dead = 0;
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (!b->cfg_active)
         continue;
      struct {
         uint32_t dest;
         uint8_t comps;
         uint8_t dead;
         bool active;
      } pending[6] = {0};
      unsigned next_slot = 0;
      unsigned cur_dead = 0;

      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_SPLIT && ins->nr_srcs >= 1) {
            unsigned w = ins->dest_components;
            for (unsigned c = 0; c < w; ++c) {
               uint32_t sv = ins->src[0] + c;
               for (unsigned k = 0; k < ARRAY_SIZE(pending); ++k) {
                  if (pending[k].active && sv >= pending[k].dest &&
                      sv < pending[k].dest + pending[k].comps) {
                     cur_dead -= pending[k].dead;
                     pending[k].active = false;
                  }
               }
            }
         } else {
            const unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(ins, s);
               for (unsigned k = 0; k < ARRAY_SIZE(pending); ++k) {
                  if (pending[k].active && sv >= pending[k].dest &&
                      sv < pending[k].dest + pending[k].comps) {
                     cur_dead -= pending[k].dead;
                     pending[k].active = false;
                  }
               }
            }
         }
         if (ins->dest == AGX_APPLE9_VREG_INVALID || ins->dest_components <= 1 ||
             a9_is_pub_val(program, ins->dest) ||
             apple9_ra_value_is_mem(spill, ins->dest) ||
             !apple9_ra_async(ins))
            continue;

         unsigned dead = 0;
         for (unsigned c = 0; c < ins->dest_components; ++c) {
            if (ins->dest + c < vcount && !BITSET_TEST(used, ins->dest + c))
               dead++;
         }
         if (!dead)
            continue;

         if (pending[next_slot].active)
            cur_dead -= pending[next_slot].dead;
         pending[next_slot].dest = ins->dest;
         pending[next_slot].comps = ins->dest_components;
         pending[next_slot].dead = (uint8_t)dead;
         pending[next_slot].active = true;
         next_slot = (next_slot + 1u) % ARRAY_SIZE(pending);
         cur_dead += dead;
         peak_dead = MAX2(peak_dead, cur_dead);
      }
   }

   ralloc_free(used);
   return MIN2(peak_dead, 18u);
}

static void
a9_add_fma_peer_dir(struct a9_fma_peers *peers, uint32_t a, uint32_t b,
                    unsigned weight)
{
   struct a9_fma_peers *fp = &peers[a];
   for (unsigned k = 0; k < fp->count; ++k) {
      if (fp->peer[k] == b) {
         unsigned nw = (unsigned)fp->weight[k] + weight;
         fp->weight[k] = (uint8_t)MIN2(nw, 255u);
         return;
      }
   }
   if (fp->count < A9_FMA_MAX_PEERS) {
      fp->peer[fp->count] = b;
      fp->weight[fp->count] = (uint8_t)MIN2(weight, 255u);
      fp->count++;
      return;
   }
   /* Replace lowest-weight peer if the new edge is strictly stronger. */
   unsigned min_idx = 0;
   for (unsigned k = 1; k < A9_FMA_MAX_PEERS; ++k) {
      if (fp->weight[k] < fp->weight[min_idx])
         min_idx = k;
   }
   if (weight > fp->weight[min_idx]) {
      fp->peer[min_idx] = b;
      fp->weight[min_idx] = (uint8_t)MIN2(weight, 255u);
   }
}

static bool
a9_instr_compact_alu_eligible(const struct agx_apple9_vir_instr *ins)
{
   if (ins->saturate || ins->src_abs_mask || ins->src_neg_mask)
      return false;
   unsigned inline_mask = ins->alu_src_uniform_mask | ins->alu_src_immediate_mask;
   switch (ins->op) {
   case AGX_APPLE9_VIR_FMA:
   case AGX_APPLE9_VIR_HFMA:
      return !(inline_mask & ~2u) && !ins->src_half_mask &&
             (!(ins->alu_src_uniform_mask & 2u) || ins->alu_src_value[1] < 64u) &&
             !((ins->alu_src_immediate_mask & 2u) && (ins->alu_src_value[1] >> 31));
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FMUL:
      return !inline_mask && !ins->src_hi_mask;
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_MOV:
      return !inline_mask;
   case AGX_APPLE9_VIR_IMM:
      return ins->encoding == AGX_APPLE9_ENC_MOV_IMM_COMPACT ||
             ins->encoding == AGX_APPLE9_ENC_MOV_IMM32;
   case AGX_APPLE9_VIR_GET_SR:
      return ins->encoding == AGX_APPLE9_ENC_GET_SR_EXTENDED;
   default:
      return false;
   }
}

/* =========================================================================
 * Main Entry Point: apple9_ra_allocate_native()
 * ========================================================================= */

bool
apple9_ra_allocate_native(struct agx_apple9_vir_program *program,
                          const struct apple9_ra_assign_options *opts_in,
                          struct apple9_ra_assign_result *result_out)
{
   struct apple9_ra_assign_result local_res;
   struct apple9_ra_assign_result *res = result_out ? result_out : &local_res;
   memset(res, 0, sizeof(*res));

   if (!program) {
      res->error = "NULL VIR program";
      return false;
   }

   struct apple9_ra_assign_options opts =
      opts_in ? *opts_in : apple9_ra_assign_default_options();
   unsigned max_gprs = opts.max_gprs ? MIN2(opts.max_gprs, AGX_APPLE9_GPR_COUNT)
                                     : AGX_APPLE9_GPR_COUNT;
   unsigned reserved = opts.reserved_gprs;

   agx_apple9_vir_reindex(program);
   if (!agx_apple9_validate_mesh_control(program, &res->error))
      return false;

   if (!a9_ensure_program_pub_array(program, &res->error))
      return false;

   if (!program->blocks || program->instruction_count == 0) {
      if (!opts.keep_ssa_ir) {
         struct apple9_ra_spill_state empty_spill;
         apple9_ra_spill_state_init(&empty_spill);
         bool ok = a9_lower_to_physical_vir(
            program, &empty_spill, NULL, 0, reserved, 0,
            opts.skip_physical_schedule, res, &res->error);
         apple9_ra_spill_state_finish(&empty_spill);
         return ok;
      }
      return true;
   }

   /* Step 1: Logical CFG + canonical variable-arity VIR_PHI. */
   if (program->num_cfg_blocks == 0 && !agx_apple9_vir_build_logical_cfg(program)) {
      res->error = "failed to build logical CFG";
      return false;
   }

   bool need_canon_phi = false;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC) {
         need_canon_phi = true;
         break;
      }
      if (ins->op == AGX_APPLE9_VIR_PHI) {
         struct agx_apple9_block *blk = agx_apple9_instr_block(ins);
         if (blk && blk->cfg_active &&
             agx_apple9_vir_nr_srcs(ins) != blk->num_cfg_preds) {
            need_canon_phi = true;
            break;
         }
      }
   }
   if (need_canon_phi && !agx_apple9_vir_canonicalize_phis(program)) {
      res->error = "failed to canonicalize VIR_PHI nodes";
      return false;
   }

   a9_strip_atomic_results(program);
   if (!a9_eliminate_identity_copies(program)) {
      res->error = "out of memory eliminating identity copies";
      return false;
   }

   struct apple9_ra_spill_state spill_state;
   apple9_ra_spill_state_init(&spill_state);
   struct apple9_ra_liveness_info liveness;
   apple9_ra_liveness_init(&liveness);

   uint32_t *mem_slot_u32 = NULL;
   uint16_t *mem_slot_u16 = NULL;
   unsigned frame_words = 0;
   void *memctx = ralloc_context(NULL);
   struct a9_ra_ctx rctx = {0};
   bool success = false;

   if (!memctx ||
       !apple9_ra_spill_state_ensure(&spill_state, program->value_count)) {
      res->error = "out of memory initializing RA state";
      goto cleanup;
   }

   /* Step 2: Pre-RA tuple canonicalization (with 16-entry reuse cache +
    * retain_mask) and AGX_APPLE9_OPERAND_CLOBBER copy insertion. */
   if (!a9_prepare_tuples_and_clobbers(program, &spill_state, true, &res->error))
      goto cleanup;

   /* Step 3: Pre-RA DCE, optional SSA validation, and pressure/latency DAG
    * scheduling. */
   if (!opts.skip_dce && !apple9_ra_dce(program)) {
      res->error = "pre-RA DCE failed";
      goto cleanup;
   }
   if (opts.validate && !apple9_ra_validate_ssa(program, &res->validate_err)) {
      res->error = res->validate_err.message;
      goto cleanup;
   }
   /* Step 3 & 4a: Pressure/latency DAG scheduling + CFG liveness. When the
    * scheduler runs, receive its post-schedule liveness directly instead of
    * freeing and immediately recomputing it. */
   if (!opts.skip_schedule) {
      struct apple9_ra_sched_options sched_opts = {
         .latency = opts.latency_schedule,
         .pressure_slack = opts.pressure_slack,
         .auto_tier_slack = opts.auto_tier_slack,
         .rejected_pressure_increase = 0,
         .liveness_out = &liveness,
      };
      if (!apple9_ra_pressure_schedule_opts(program, &sched_opts)) {
         res->error = "pressure/latency scheduler failed";
         goto cleanup;
      }
   } else if (!apple9_ra_liveness(program, &liveness)) {
      res->error = "liveness analysis failed";
      goto cleanup;
   }

   unsigned min_single = 0;
   unsigned effective_demand =
      a9_calc_gpr_demand(program, &liveness, &spill_state, reserved, &min_single);

   unsigned spill_target = max_gprs;
   if (opts.force_spill) {
      unsigned spill_res = MAX2(reserved, APPLE9_RA_SPILL_RESERVED_GPRS);
      unsigned floor_gprs = MAX2(min_single + spill_res, 12u);
      floor_gprs = ALIGN_POT(floor_gprs, 4u);
      unsigned forced_target = MIN2(floor_gprs + opts.spill_headroom, max_gprs);
      if (forced_target < spill_target)
         spill_target = forced_target;
      if (floor_gprs + opts.spill_headroom < max_gprs)
         max_gprs = floor_gprs + opts.spill_headroom;
   } else if (opts.spill_headroom > 0) {
      unsigned spill_res = MAX2(reserved, APPLE9_RA_SPILL_RESERVED_GPRS);
      unsigned floor = MAX2(min_single + spill_res, 12u);
      if (max_gprs > floor + opts.spill_headroom)
         spill_target = max_gprs - opts.spill_headroom;
      else if (max_gprs > floor)
         spill_target = floor;
   }

   if (!a9_validate_constraint_windows(program, &spill_state, reserved, max_gprs,
                                       &res->error))
      goto cleanup;

   bool need_spill = (effective_demand > spill_target) || opts.force_spill;
   if (need_spill) {
      unsigned spill_reserved = MAX2(reserved, APPLE9_RA_SPILL_RESERVED_GPRS);
      struct apple9_ra_spill_options sopts = apple9_ra_spill_default_options();
      sopts.max_gprs = spill_target;
      sopts.reserved_gprs = reserved;
      sopts.spill_reserved_gprs = spill_reserved;
      sopts.disable_uniform_remat = opts.disable_uniform_remat;
      sopts.spill_partial_phi_sources = true;
      sopts.hoist_loop_invariant = true;
      sopts.track_spill_homes = true;

      if (!apple9_ra_spill(program, &spill_state, &sopts)) {
         if (spill_target < max_gprs)
            res->needs_spill_headroom = true;
         res->error = "Braun-Hack spiller failed";
         goto cleanup;
      }
      res->has_spills = spill_state.has_spills;
      res->uniform_remat_used = spill_state.uniform_remat_used;
      if (spill_state.has_spills)
         reserved = spill_reserved;

      /* Re-canonicalize any multi-word source tuple split by reload. */
      if (!a9_prepare_tuples_and_clobbers(program, &spill_state, false,
                                          &res->error))
         goto cleanup;

      if (!a9_validate_constraint_windows(program, &spill_state, reserved,
                                          max_gprs, &res->error))
         goto cleanup;

      apple9_ra_liveness_finish(&liveness);
      if (!apple9_ra_liveness(program, &liveness)) {
         res->error = "post-spill liveness failed";
         goto cleanup;
      }
      effective_demand = a9_calc_gpr_demand(program, &liveness, &spill_state,
                                            reserved, NULL);
      if (effective_demand > max_gprs) {
         res->needs_spill_headroom = true;
         res->error = "post-spill tuple canonicalization exceeded GPR budget";
         goto cleanup;
      }

      if (spill_state.has_spills) {
         mem_slot_u32 =
            malloc(MAX2(program->value_count, 1u) * sizeof(*mem_slot_u32));
         if (!mem_slot_u32) {
            res->error = "out of memory allocating spill slot map";
            goto cleanup;
         }
         if (!apple9_ra_spill_coalesce_color(program, &spill_state, mem_slot_u32,
                                             &frame_words)) {
            res->error = "spill slot coloring exceeded 32 KiB scratch frame";
            goto cleanup;
         }
         if (apple9_ra_spill_scratch_bytes(program, frame_words) >
             AGX_APPLE9_MAX_SCRATCH_BYTES) {
            res->error = "Apple9 scratch frame exceeds 32 KiB limit";
            goto cleanup;
         }
      }
   }

   /* Step 5: Build dominance-order assignment structures (phi_web,
    * preferences, collect/export hints, scarcity & scarcity_floor). */
   const unsigned num_blocks = program->num_cfg_blocks;
   const unsigned orig_vcount = program->value_count;
   const unsigned valloc = MAX2(orig_vcount, 1u);

   /* Round up effective demand to the Apple9 (G16G) occupancy tier ceiling when
    * using the default full register file so coalescing and async-WAW avoidance
    * never cross a real sub-45 occupancy cliff (18/24/28/32/36/40/44 GPRs).
    * Include simultaneously in-flight dead async destination lanes so back-to-back
    * vec4 texture/load issue does not hit WAW WAIT stalls, and cap demand <= 64
    * on the flat 45..96 shelf at 64 GPRs to preserve 4B compact ALU (r0..r63). */
   unsigned assign_bound = max_gprs;
   if (opts.max_gprs == 0 && !opts.force_spill &&
       effective_demand <= max_gprs) {
      unsigned async_dead_pad = a9_calc_async_dead_gprs(program, &spill_state, memctx);
      unsigned target_demand =
         MIN2(MAX2(effective_demand + async_dead_pad + opts.spill_headroom, 6u),
              max_gprs);
      unsigned tier_gprs =
         agx_occupancy_for_register_count(2u * target_demand).max_registers / 2u;
      if (tier_gprs > 64u && target_demand <= 64u)
         tier_gprs = 64u;
      tier_gprs = MIN2(tier_gprs, max_gprs);
      assign_bound = MIN2(MAX2(target_demand, tier_gprs), max_gprs);
   }

   rctx.memctx = memctx;
   rctx.program = program;
   rctx.spill = &spill_state;
   rctx.liveness = &liveness;
   rctx.num_cfg_blocks = num_blocks;
   rctx.orig_vcount = orig_vcount;
   rctx.bound = assign_bound;
   rctx.reserved = reserved;
   rctx.demand = MIN2(effective_demand, assign_bound);
   rctx.cost_constraints = opts.cost_register_constraints;
   rctx.affinity_ssa = UINT32_MAX;
   rctx.affinity_reg = UINT_MAX;
   memset(rctx.reg_pref_index, 0xff, sizeof(rctx.reg_pref_index));
   util_dynarray_init(&rctx.reg_prefs, memctx);
   util_dynarray_init(&rctx.active_splits, memctx);

   rctx.cfg_blocks =
      rzalloc_array(memctx, struct agx_apple9_block *, num_blocks);
   rctx.cur_ssa = ralloc_array(memctx, uint32_t, valloc);
   rctx.block_splits_out =
      rzalloc_array(memctx, struct util_dynarray, num_blocks);
   rctx.block_synth_phis =
      rzalloc_array(memctx, struct util_dynarray, num_blocks);
   rctx.block_visited = rzalloc_array(memctx, bool, num_blocks);
   rctx.collect_refs =
      rzalloc_array(memctx, struct a9_collect_ref *, valloc);
   rctx.precolor_hint = ralloc_array(memctx, uint8_t, valloc);
   if (!rctx.cfg_blocks || !rctx.cur_ssa || !rctx.block_splits_out ||
       !rctx.block_synth_phis || !rctx.block_visited || !rctx.collect_refs ||
       !rctx.precolor_hint || !a9_ra_ensure_val_cap(&rctx, valloc)) {
      res->error = rctx.error ? rctx.error : "out of memory in RA setup";
      goto cleanup;
   }

   for (unsigned v = 0; v < valloc; ++v) {
      rctx.cur_ssa[v] = v;
      rctx.precolor_hint[v] = AGX_APPLE9_PHYS_INVALID;
   }
   for (unsigned b = 0; b < num_blocks; ++b) {
      util_dynarray_init(&rctx.block_splits_out[b], memctx);
      util_dynarray_init(&rctx.block_synth_phis[b], memctx);
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->cfg_active && b->cfg_index < num_blocks)
         rctx.cfg_blocks[b->cfg_index] = b;
   }

   /* Preserve publication physical indices and mark pre-defined fixed inputs
    * (values with fixed_phys and no defining instruction). */
   BITSET_WORD *has_def = BITSET_RZALLOC(memctx, valloc);
   BITSET_WORD *constants = BITSET_RZALLOC(memctx, valloc);
   if (!has_def || !constants) {
      res->error = "out of memory allocating RA bitsets";
      goto cleanup;
   }

   uint32_t *collect_uses = rzalloc_array(memctx, uint32_t, valloc);
   if (!collect_uses) {
      res->error = "out of memory allocating collect use counts";
      goto cleanup;
   }
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op != AGX_APPLE9_VIR_COLLECT && ins->op != AGX_APPLE9_VIR_PHI) {
            for (unsigned s = 0; s < ins->nr_srcs; ++s) {
               if (ins->src[s] < orig_vcount)
                  collect_uses[ins->src[s]]++;
            }
         }
         if (ins->op == AGX_APPLE9_VIR_IMM && ins->dest < orig_vcount &&
             ins->dest_components == 1)
            BITSET_SET(constants, ins->dest);
         if (ins->dest != AGX_APPLE9_VREG_INVALID && ins->dest_components > 0) {
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               if (ins->dest + c < orig_vcount)
                  BITSET_SET(has_def, ins->dest + c);
            }
         }
      }
   }

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op != AGX_APPLE9_VIR_COLLECT || ins->nr_srcs < 2 ||
             ins->nr_srcs > 4)
            continue;
         /* Skip self-duplicate tuples where a single scalar would need two
          * distinct GPR positions simultaneously. */
         bool has_dup = false;
         for (unsigned a = 0; a < ins->nr_srcs && !has_dup; ++a) {
            for (unsigned b = a + 1; b < ins->nr_srcs; ++b) {
               if (ins->src[a] == ins->src[b]) {
                  has_dup = true;
                  break;
               }
            }
         }
         if (has_dup)
            continue;

         unsigned w = (ins->dest < orig_vcount && collect_uses[ins->dest] > 0)
                         ? collect_uses[ins->dest]
                         : 1u;
         for (unsigned s = 0; s < ins->nr_srcs; ++s) {
            uint32_t sv = ins->src[s];
            if (sv >= orig_vcount)
               continue;
            struct a9_collect_ref *found = NULL;
            unsigned list_len = 0;
            for (struct a9_collect_ref *r = rctx.collect_refs[sv]; r; r = r->next) {
               list_len++;
               if (r->nr_srcs == ins->nr_srcs &&
                   r->dest_components == ins->dest_components &&
                   memcmp(r->src, ins->src,
                          ins->nr_srcs * sizeof(ins->src[0])) == 0) {
                  found = r;
                  break;
               }
            }
            if (found) {
               found->weight += w;
            } else if (list_len < 8) {
               struct a9_collect_ref *nr =
                  ralloc(memctx, struct a9_collect_ref);
               if (nr) {
                  nr->nr_srcs = ins->nr_srcs;
                  nr->dest_components = ins->dest_components;
                  memcpy(nr->src, ins->src,
                         ins->nr_srcs * sizeof(ins->src[0]));
                  nr->weight = w;
                  nr->next = rctx.collect_refs[sv];
                  rctx.collect_refs[sv] = nr;
               }
            }
         }
      }
   }

   /* Collect general pre-color affinity hints from ALL fixed-register operands
    * (ENTRY_REG preloads, fixed_phys values, single-GPR constrained operands,
    * and fixed-register live_out sinks). */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_ENTRY_REG && ins->dest < orig_vcount &&
             ins->immediate < max_gprs) {
            rctx.precolor_hint[ins->dest] = (uint8_t)ins->immediate;
         } else if (ins->encoding != AGX_APPLE9_ENC_PSEUDO) {
            for (unsigned s = 0; s < ins->nr_srcs; ) {
               unsigned w = apple9_ra_source_width(ins, s);
               uint32_t sv = ins->src[s];
               if (w == 1 && sv < orig_vcount && !a9_is_pub_val(program, sv) &&
                   !apple9_ra_value_is_mem(&spill_state, sv)) {
                  struct a9_gpr_constraint sc =
                     a9_src_constraint(program, ins, s);
                  if (sc.min == sc.max && sc.min < max_gprs &&
                      rctx.precolor_hint[sv] == AGX_APPLE9_PHYS_INVALID)
                     rctx.precolor_hint[sv] = sc.min;
               }
               s += w;
            }
         }
      }
   }
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      uint32_t v = program->live_out[k];
      if (v < orig_vcount &&
          program->live_out_reg[k] != AGX_APPLE9_PHYS_INVALID &&
          rctx.precolor_hint[v] == AGX_APPLE9_PHYS_INVALID)
         rctx.precolor_hint[v] = program->live_out_reg[k];
   }
   for (unsigned v = 0; v < orig_vcount; ++v) {
      if (program->fixed_phys &&
          program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID &&
          !a9_is_pub_val(program, v))
         rctx.precolor_hint[v] = program->fixed_phys[v];
   }

   for (unsigned v = 0; v < orig_vcount; ++v) {
      if (a9_is_pub_val(program, v)) {
         assert(program->phys && v < program->phys_capacity &&
                program->phys[v] < AGX_APPLE9_PUBLICATION_COUNT);
         rctx.phys[v] = program->phys[v];
         BITSET_SET(rctx.visited, v);
      } else if (!BITSET_TEST(has_def, v) && program->fixed_phys &&
                 program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID) {
         a9_assign_gpr_val(&rctx, v, program->fixed_phys[v]);
      }
   }

   /* Build unified coalescing web (non-interfering union-find + edge/dead-group
    * preferences over both VIR_PHI edges and same-class GPR VIR_MOVs). */
   unsigned nr_web_members = orig_vcount;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI) {
            unsigned ns = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < ns; ++s) {
               uint32_t sv = agx_apple9_vir_src(ins, s);
               if (sv < orig_vcount && BITSET_TEST(constants, sv))
                  nr_web_members++;
            }
         } else if (ins->op == AGX_APPLE9_VIR_MOV &&
                    ins->dest_components == 1 && ins->nr_srcs == 1) {
            uint32_t sv = ins->src[0];
            if (sv < orig_vcount && BITSET_TEST(constants, sv))
               nr_web_members++;
         }
      }
   }

   unsigned web_alloc = MAX2(nr_web_members, 1u);
   rctx.phi_web = rzalloc_array(memctx, struct a9_phi_web_node, web_alloc);
   rctx.web_members =
      rzalloc_array(memctx, struct a9_phi_web_members, web_alloc);
   rctx.phi_prefs =
      rzalloc_array(memctx, struct a9_phi_preference *, valloc);
   rctx.dead_groups =
      rzalloc_array(memctx, struct a9_phi_dead_group *, valloc);
   rctx.phi_weights = rzalloc_array(memctx, uint64_t, valloc);
   void *graph_ctx = ralloc_context(memctx);
   struct u_sparse_bitset *phi_interf =
      build_phi_web_interference(program, rctx.cfg_blocks, &liveness,
                                 &spill_state, constants, graph_ctx);
   if (!rctx.phi_web || !rctx.web_members || !rctx.phi_prefs ||
       !rctx.dead_groups || !rctx.phi_weights || !phi_interf) {
      res->error = "out of memory building phi web";
      goto cleanup;
   }

   for (unsigned i = 0; i < web_alloc; ++i) {
      rctx.phi_web[i].parent = i;
      rctx.web_members[i] = (struct a9_phi_web_members){i, i, UINT32_MAX, 1, NULL};
   }

   unsigned next_const_member = orig_vcount;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI) {
            uint32_t dv = ins->dest;
            bool d_mem = apple9_ra_value_is_mem(&spill_state, dv);
            unsigned ns = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < ns; ++s) {
               uint32_t sv = agx_apple9_vir_src(ins, s);
               if (sv >= orig_vcount)
                  continue;
               unsigned member =
                  BITSET_TEST(constants, sv) ? next_const_member++ : sv;
               if (d_mem || apple9_ra_value_is_mem(&spill_state, sv) ||
                   a9_is_pub_val(program, dv) || a9_is_pub_val(program, sv))
                  continue;
               phi_web_union_noninterfering(rctx.phi_web, rctx.web_members,
                                            phi_interf, orig_vcount, dv, member);
            }
         } else if (ins->op == AGX_APPLE9_VIR_MOV &&
                    ins->dest_components == 1 && ins->nr_srcs == 1) {
            uint32_t dv = ins->dest;
            uint32_t sv = ins->src[0];
            if (dv >= orig_vcount || sv >= orig_vcount)
               continue;
            unsigned member =
               BITSET_TEST(constants, sv) ? next_const_member++ : sv;
            if (apple9_ra_value_is_mem(&spill_state, dv) ||
                apple9_ra_value_is_mem(&spill_state, sv) ||
                a9_is_pub_val(program, dv) || a9_is_pub_val(program, sv))
               continue;
            phi_web_union_noninterfering(rctx.phi_web, rctx.web_members,
                                         phi_interf, orig_vcount, dv, member);
         }
      }
   }

   /* Compute loop nesting depth per CFG block for phi/move edge weighting. */
   unsigned *depth = rzalloc_array(memctx, unsigned, num_blocks);
   if (!depth) {
      res->error = "out of memory computing loop depths";
      goto cleanup;
   }
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *hdr = rctx.cfg_blocks[bi];
      if (!hdr->loop_header)
         continue;
      unsigned end = bi;
      for (unsigned p = 0; p < hdr->num_cfg_preds; ++p)
         end = MAX2(end, hdr->cfg_preds[p]->cfg_index);
      for (unsigned b = bi; b <= end; ++b)
         depth[b]++;
   }

   unsigned next_const_pref = orig_vcount;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *blk = rctx.cfg_blocks[bi];
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(blk);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_PHI) {
            uint32_t dv = ins->dest;
            bool d_mem = apple9_ra_value_is_mem(&spill_state, dv);
            for (unsigned p = 0; p < blk->num_cfg_preds; ++p) {
               struct agx_apple9_block *pred = blk->cfg_preds[p];
               uint32_t sv = agx_apple9_vir_src(ins, p);
               if (sv >= orig_vcount)
                  continue;
               unsigned cmember =
                  BITSET_TEST(constants, sv) ? next_const_pref++ : UINT_MAX;
               if (d_mem || apple9_ra_value_is_mem(&spill_state, sv) ||
                   a9_is_pub_val(program, dv) || a9_is_pub_val(program, sv))
                  continue;

               uint64_t weight = 1ull << (3u * MIN2(depth[pred->cfg_index], 6u));
               bool redefined = false;
               for (struct agx_apple9_vir_instr *o =
                       agx_apple9_block_first_instr(blk);
                    o; o = agx_apple9_instr_next(o)) {
                  if (a9_is_block_entry_prefix(o))
                     continue;
                  if (o->op != AGX_APPLE9_VIR_PHI)
                     break;
                  redefined |= (o->dest == sv);
               }
               bool dead = redefined ||
                           !u_sparse_bitset_test(&liveness.live_in[bi], sv);
               if (dead && !u_sparse_bitset_test(&phi_interf[sv], dv)) {
                  struct a9_phi_dead_group *g = rctx.dead_groups[sv];
                  while (g && (g->succ != blk || g->pred != pred))
                     g = g->next;
                  if (!g) {
                     g = rzalloc(memctx, struct a9_phi_dead_group);
                     if (!g) {
                        ralloc_free(graph_ctx);
                        res->error = "out of memory allocating phi dead group";
                        goto cleanup;
                     }
                     *g = (struct a9_phi_dead_group){
                        rctx.dead_groups[sv], blk, pred, NULL, weight};
                     rctx.dead_groups[sv] = g;
                  }
                  struct a9_phi_dead_dest *dd =
                     ralloc(memctx, struct a9_phi_dead_dest);
                  if (!dd) {
                     ralloc_free(graph_ctx);
                     res->error = "out of memory allocating phi dead dest";
                     goto cleanup;
                  }
                  *dd = (struct a9_phi_dead_dest){g->dests, dv};
                  g->dests = dd;
               }

               struct a9_phi_preference *fwd =
                  ralloc(memctx, struct a9_phi_preference);
               struct a9_phi_preference *rev =
                  ralloc(memctx, struct a9_phi_preference);
               if (!fwd || !rev) {
                  ralloc_free(graph_ctx);
                  res->error = "out of memory allocating phi preferences";
                  goto cleanup;
               }
               *fwd = (struct a9_phi_preference){rctx.phi_prefs[dv], sv, weight};
               *rev = (struct a9_phi_preference){rctx.phi_prefs[sv], dv, weight};
               rctx.phi_prefs[dv] = fwd;
               rctx.phi_prefs[sv] = rev;
               rctx.phi_weights[sv] += weight;
               rctx.phi_weights[dv] += weight;
               if (cmember != UINT_MAX)
                  rctx.web_members[cmember].constant = fwd;
            }
         } else if (ins->op == AGX_APPLE9_VIR_MOV &&
                    ins->dest_components == 1 && ins->nr_srcs == 1) {
            uint32_t dv = ins->dest;
            uint32_t sv = ins->src[0];
            if (dv >= orig_vcount || sv >= orig_vcount)
               continue;
            unsigned cmember =
               BITSET_TEST(constants, sv) ? next_const_pref++ : UINT_MAX;
            if (apple9_ra_value_is_mem(&spill_state, dv) ||
                apple9_ra_value_is_mem(&spill_state, sv) ||
                a9_is_pub_val(program, dv) || a9_is_pub_val(program, sv))
               continue;
            uint64_t weight = 1ull << (3u * MIN2(depth[bi], 6u));
            struct a9_phi_preference *fwd =
               ralloc(memctx, struct a9_phi_preference);
            struct a9_phi_preference *rev =
               ralloc(memctx, struct a9_phi_preference);
            if (!fwd || !rev) {
               ralloc_free(graph_ctx);
               res->error = "out of memory allocating mov preferences";
               goto cleanup;
            }
            *fwd = (struct a9_phi_preference){rctx.phi_prefs[dv], sv, weight};
            *rev = (struct a9_phi_preference){rctx.phi_prefs[sv], dv, weight};
            rctx.phi_prefs[dv] = fwd;
            rctx.phi_prefs[sv] = rev;
            rctx.phi_weights[sv] += weight;
            rctx.phi_weights[dv] += weight;
            if (cmember != UINT_MAX)
               rctx.web_members[cmember].constant = fwd;
         }
      }
   }
   ralloc_free(graph_ctx);

   /* Build soft 3-bank register-file bank-conflict peers (r % 3) for 3-source
    * FMA/HFMA operands and compact-ALU r0..r63 preferences (protecting 4B
    * compact encodings and the 16 KiB L1I cliff when bound > 64). */
   rctx.fma_peers = rzalloc_array(memctx, struct a9_fma_peers, web_alloc);
   rctx.fma_peers_count = rctx.fma_peers ? web_alloc : 0;
   if (rctx.bound > 64u)
      rctx.compact_alu = BITSET_RZALLOC(memctx, valloc);
   if (rctx.fma_peers || rctx.compact_alu) {
      for (unsigned bi = 0; bi < num_blocks; ++bi) {
         unsigned mult = depth[bi] > 0 ? 2u : 1u;
         for (struct agx_apple9_vir_instr *ins =
                 agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
              ins; ins = agx_apple9_instr_next(ins)) {
            if (rctx.fma_peers &&
                (ins->op == AGX_APPLE9_VIR_FMA ||
                 ins->op == AGX_APPLE9_VIR_HFMA) &&
                ins->nr_srcs >= 2) {
               unsigned inline_mask =
                  ins->alu_src_uniform_mask | ins->alu_src_immediate_mask;
               uint32_t g[3] = {
                  AGX_APPLE9_VREG_INVALID,
                  AGX_APPLE9_VREG_INVALID,
                  AGX_APPLE9_VREG_INVALID,
               };
               unsigned sidx = 0;
               for (unsigned slot = 0; slot < 3 && sidx < ins->nr_srcs; ++slot) {
                  if (inline_mask & BITFIELD_BIT(slot))
                     continue;
                  uint32_t sv = ins->src[sidx++];
                  if (sv < orig_vcount && !a9_is_pub_val(program, sv) &&
                      !apple9_ra_value_is_mem(&spill_state, sv)) {
                     unsigned root = phi_web_find(rctx.phi_web, sv);
                     if (root < rctx.fma_peers_count)
                        g[slot] = root;
                  }
               }
               /* Apple9 G16G has a 3-bank (r % 3) register file where
                * (src1, src2) is the critical read-port bank-conflict pair;
                * weight it highest and insert it first so it dominates peer
                * slots, while still avoiding (src0, src1) and (src0, src2)
                * collisions when a free bank is available. */
               if (g[1] != AGX_APPLE9_VREG_INVALID &&
                   g[2] != AGX_APPLE9_VREG_INVALID && g[1] != g[2]) {
                  a9_add_fma_peer_dir(rctx.fma_peers, g[1], g[2], 4u * mult);
                  a9_add_fma_peer_dir(rctx.fma_peers, g[2], g[1], 4u * mult);
               }
               if (g[0] != AGX_APPLE9_VREG_INVALID &&
                   g[1] != AGX_APPLE9_VREG_INVALID && g[0] != g[1]) {
                  a9_add_fma_peer_dir(rctx.fma_peers, g[0], g[1], 1u * mult);
                  a9_add_fma_peer_dir(rctx.fma_peers, g[1], g[0], 1u * mult);
               }
               if (g[0] != AGX_APPLE9_VREG_INVALID &&
                   g[2] != AGX_APPLE9_VREG_INVALID && g[0] != g[2]) {
                  a9_add_fma_peer_dir(rctx.fma_peers, g[0], g[2], 1u * mult);
                  a9_add_fma_peer_dir(rctx.fma_peers, g[2], g[0], 1u * mult);
               }
            }

            if (rctx.compact_alu && a9_instr_compact_alu_eligible(ins)) {
               if (ins->dest != AGX_APPLE9_VREG_INVALID &&
                   ins->dest_components == 1 && ins->dest < orig_vcount &&
                   !a9_is_pub_val(program, ins->dest) &&
                   !apple9_ra_value_is_mem(&spill_state, ins->dest))
                  BITSET_SET(rctx.compact_alu, ins->dest);
               for (unsigned s = 0; s < ins->nr_srcs; ++s) {
                  uint32_t sv = ins->src[s];
                  if (sv < orig_vcount && !a9_is_pub_val(program, sv) &&
                      !apple9_ra_value_is_mem(&spill_state, sv))
                     BITSET_SET(rctx.compact_alu, sv);
               }
            }
         }
      }
   }

   /* Build soft operand preferences, bank scarcity, and scarcity_floor. */
   if (rctx.cost_constraints) {
      rctx.preferred =
         ralloc_array(memctx, struct a9_gpr_constraint, valloc);
      if (!rctx.preferred) {
         res->error = "out of memory allocating constraint preferences";
         goto cleanup;
      }
      for (unsigned v = 0; v < valloc; ++v) {
         rctx.preferred[v] = (struct a9_gpr_constraint){
            .min = 0, .max = (uint8_t)(max_gprs - 1u), .align = 1};
      }

      for (unsigned bi = 0; bi < num_blocks; ++bi) {
         for (struct agx_apple9_vir_instr *ins =
                 agx_apple9_block_first_instr(rctx.cfg_blocks[bi]);
              ins; ins = agx_apple9_instr_next(ins)) {
            if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
                ins->op == AGX_APPLE9_VIR_ENTRY_REG)
               continue;

            if (ins->dest != AGX_APPLE9_VREG_INVALID &&
                ins->dest_components > 0 &&
                !a9_is_pub_val(program, ins->dest) &&
                !apple9_ra_value_is_mem(&spill_state, ins->dest)) {
               struct a9_gpr_constraint dc = a9_dest_constraint(program, ins);
               unsigned end = MIN2((unsigned)dc.max + ins->dest_components, max_gprs);
               /* Only protect genuinely tight windows (r0..r15 / r0..r31);
                * inflating scarcity across r0..r63 steers unconstrained defs
                * into r64..r95, breaking 4B MOV (dst < 64) and compact ALU. */
               if (end > dc.min && end - dc.min <= 32u) {
                  unsigned weight = DIV_ROUND_UP(max_gprs, end - dc.min);
                  for (unsigned r = dc.min; r < end; ++r)
                     rctx.scarcity[r] += weight;
               }
            }

            for (unsigned s = 0; s < ins->nr_srcs; ) {
               unsigned w = apple9_ra_source_width(ins, s);
               uint32_t sv = ins->src[s];
               if (sv < orig_vcount && !a9_is_pub_val(program, sv) &&
                   !apple9_ra_value_is_mem(&spill_state, sv)) {
                  struct a9_gpr_constraint sc =
                     a9_src_constraint(program, ins, s);
                  if (!(sc.min == sc.max && sc.max < reserved)) {
                     struct a9_gpr_constraint *p = &rctx.preferred[sv];
                     p->min = MAX2(p->min, sc.min);
                     p->max = MIN2(p->max, sc.max);
                     p->align = MAX2(p->align, sc.align);
                  }
                  unsigned end = MIN2((unsigned)sc.max + w, max_gprs);
                  if (end > sc.min && end - sc.min <= 32u) {
                     unsigned weight = DIV_ROUND_UP(max_gprs, end - sc.min);
                     for (unsigned r = sc.min; r < end; ++r)
                        rctx.scarcity[r] += weight;
                  }
               }
               s += w;
            }
         }
      }

      for (unsigned w = 1; w <= A9_SCARCITY_MAX_WIDTH; ++w) {
         rctx.scarcity_floor[w][AGX_APPLE9_GPR_COUNT] = UINT_MAX;
         for (int r = (int)AGX_APPLE9_GPR_COUNT - 1; r >= 0; --r) {
            unsigned sum = UINT_MAX;
            if ((unsigned)r + w <= AGX_APPLE9_GPR_COUNT) {
               sum = 0;
               for (unsigned k = 0; k < w; ++k)
                  sum += rctx.scarcity[(unsigned)r + k];
            }
            rctx.scarcity_floor[w][r] =
               MIN2(sum, rctx.scarcity_floor[w][r + 1]);
         }
      }
   }

   /* Step 6: Dominance-order (RPO) SSA register assignment! */
   unsigned *rpo = ralloc_array(memctx, unsigned, num_blocks);
   if (!rpo || !a9_compute_rpo(rctx.cfg_blocks, num_blocks, rpo)) {
      res->error = "out of memory computing RPO";
      goto cleanup;
   }

   for (unsigned i = 0; i < num_blocks; ++i) {
      if (!a9_assign_block(&rctx, rctx.cfg_blocks[rpo[i]])) {
         res->needs_spill_headroom |= rctx.needs_spill_headroom;
         res->error = rctx.error ? rctx.error : "register assignment failed";
         goto cleanup;
      }
   }

   /* Reindex & refresh liveness/kill_mask if any live-range split occurred. */
   agx_apple9_vir_reindex(program);
   if (rctx.any_split) {
      apple9_ra_liveness_finish(&liveness);
      if (!apple9_ra_liveness(program, &liveness)) {
         res->error = "post-assignment liveness refresh failed";
         goto cleanup;
      }
   }

   /* Commit program->phys[], program->reserved_gprs[], and block->live_out. */
   free(program->phys);
   program->phys = malloc(MAX2(program->value_count, 1u));
   if (!program->phys) {
      res->error = "out of memory committing program->phys";
      goto cleanup;
   }
   program->phys_capacity = MAX2(program->value_count, 1u);
   memcpy(program->phys, rctx.phys, program->value_count);

   memset(program->reserved_gprs, 0, sizeof(program->reserved_gprs));
   for (unsigned r = 0; r < reserved && r < AGX_APPLE9_GPR_COUNT; ++r)
      program->reserved_gprs[r] = true;

   unsigned live_words = BITSET_WORDS(MAX2(program->value_count, 1u));
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *blk = rctx.cfg_blocks[bi];
      BITSET_WORD *new_in = calloc(live_words, sizeof(BITSET_WORD));
      BITSET_WORD *new_out = calloc(live_words, sizeof(BITSET_WORD));
      if (!new_in || !new_out) {
         free(new_in);
         free(new_out);
         res->error = "out of memory populating block liveness";
         goto cleanup;
      }
      free(blk->live_in);
      free(blk->live_out);
      blk->live_in = new_in;
      blk->live_out = new_out;
      blk->live_words = live_words;
      U_SPARSE_BITSET_FOREACH_SET(&liveness.live_in[bi], v) {
         if (v < program->value_count)
            BITSET_SET(blk->live_in, v);
      }
      U_SPARSE_BITSET_FOREACH_SET(&liveness.live_out[bi], v) {
         if (v < program->value_count)
            BITSET_SET(blk->live_out, v);
      }
   }

   /* Narrow 32-bit spill slots to uint16_t for apple9_ra_assignment. */
   if (spill_state.has_spills) {
      mem_slot_u16 =
         calloc(MAX2(program->value_count, 1u), sizeof(*mem_slot_u16));
      if (!mem_slot_u16) {
         res->error = "out of memory allocating u16 spill slot map";
         goto cleanup;
      }
      for (unsigned v = 0; v < program->value_count; ++v) {
         if (apple9_ra_value_is_mem(&spill_state, v)) {
            if (apple9_ra_value_is_imm(&spill_state, v))
               continue;
            if (v < orig_vcount && mem_slot_u32[v] == UINT32_MAX) {
               /* Dead/DCE'd RA_MEM value with no active definition. */
               BITSET_CLEAR(spill_state.is_mem, v);
               continue;
            }
            assert(v < orig_vcount && mem_slot_u32[v] < APPLE9_RA_MAX_SPILL_SLOTS);
            mem_slot_u16[v] = (uint16_t)mem_slot_u32[v];
         }
      }
   }

   struct apple9_ra_assignment assign = {
      .gpr = program->phys,
      .is_mem = spill_state.has_spills ? spill_state.is_mem : NULL,
      .is_imm = spill_state.has_spills ? spill_state.is_imm : NULL,
      .imm_val = spill_state.has_spills ? spill_state.imm_val : NULL,
      .mem_slot = mem_slot_u16,
      .max_mem_slots = spill_state.has_spills ? frame_words : 0u,
      .phis_lowered = false,
   };

   /* Step 7: Lower all VIR_PHI edges via Phase 1C parallel-copy lowerer,
    * strictly honoring `max_gprs` as max_temp_gpr (and keeping VIR_PHI nodes
    * temporarily so post-RA validation can verify every lowered edge). */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *pred = rctx.cfg_blocks[bi];
      uint64_t actual_live[2] = {0, 0};
      U_SPARSE_BITSET_FOREACH_SET(&liveness.live_out[bi], v) {
         if (v < program->value_count && !a9_is_pub_val(program, v) &&
             !apple9_ra_value_is_mem(&spill_state, v) &&
             program->phys[v] < AGX_APPLE9_GPR_COUNT)
            gpr_set(actual_live, program->phys[v]);
      }
      uint8_t s0 = reserved >= 5
                      ? APPLE9_RA_DEFAULT_SWAP_REG0
                      : (reserved >= 2 ? A9_COPY_REG
                                       : AGX_APPLE9_PHYS_INVALID);
      uint8_t s1 = reserved >= 5 ? APPLE9_RA_DEFAULT_SWAP_REG1
                                 : AGX_APPLE9_PHYS_INVALID;
      if (s0 != AGX_APPLE9_PHYS_INVALID && gpr_test(actual_live, s0))
         s0 = AGX_APPLE9_PHYS_INVALID;
      if (s1 != AGX_APPLE9_PHYS_INVALID && gpr_test(actual_live, s1))
         s1 = AGX_APPLE9_PHYS_INVALID;
      struct apple9_ra_pcopy_opts edge_opts = {
         .live_gprs = {actual_live[0], actual_live[1]},
         .scratch_gpr = s0,
         .scratch_gpr2 = s1,
         .max_temp_gpr = (uint8_t)rctx.bound,
         .mov_imm_max_gpr = 64u,
         .prefer_swap_2cycle = false,
      };
      for (unsigned r = 0; r < reserved && r < AGX_APPLE9_GPR_COUNT; ++r)
         gpr_set(edge_opts.live_gprs, r);
      if (!apple9_ra_lower_phi_edge(program, pred, NULL, &assign, &edge_opts)) {
         res->error = "parallel-copy phi-edge lowering failed";
         goto cleanup;
      }
   }
   assign.phis_lowered = true;

   /* Optional Post-RA register-file validation (compact GPR[96]+MEM[n]). */
   if (opts.validate &&
       !apple9_ra_validate_post_ra(program, &assign, &res->validate_err)) {
      res->error = res->validate_err.message;
      goto cleanup;
   }

   unsigned peak_gprs = rctx.reg_count;

   if (opts.keep_ssa_ir) {
      unsigned base_words = apple9_ra_spill_base_words(program);
      unsigned spill_bytes = frame_words ? ALIGN_POT(frame_words * 4u, 16u) : 0u;
      program->max_phys_gpr = rctx.reg_count ? (rctx.reg_count - 1u) : 0u;
      program->peak_live_gprs = peak_gprs;
      program->spill_slots =
         frame_words > base_words ? (spill_bytes / 4u - base_words) : 0u;
      program->scratch_size =
         apple9_ra_spill_scratch_bytes(program, frame_words);

      res->max_phys_gpr = program->max_phys_gpr;
      res->peak_live_gprs = program->peak_live_gprs;
      res->reserved_gprs = reserved;
      res->spill_slots = program->spill_slots;
      res->scratch_size = program->scratch_size;
      success = true;
      goto cleanup;
   }

   /* Step 8: Lower SSA + pseudos + spills to canonical physical VIR. */
   if (!a9_lower_to_physical_vir(program, &spill_state, mem_slot_u16,
                                 frame_words, reserved, peak_gprs,
                                 opts.skip_physical_schedule, res, &res->error))
      goto cleanup;

   success = true;

cleanup:
   free(rctx.phys);
   free(rctx.orig_ssa);
   free(rctx.last_reg);
   free(rctx.visited);
   free(mem_slot_u32);
   free(mem_slot_u16);
   ralloc_free(memctx);
   apple9_ra_liveness_finish(&liveness);
   apple9_ra_spill_state_finish(&spill_state);
   return success;
}
