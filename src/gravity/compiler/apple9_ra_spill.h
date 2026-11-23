/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef APPLE9_RA_SPILL_H
#define APPLE9_RA_SPILL_H

#include <stdbool.h>
#include <stdint.h>
#include "agx_apple9_ir.h"
#include "agx_apple9_profile.h"
#include "util/bitset.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum 32-bit spill word slots in the 32 KiB Apple9 invocation frame. */
#define APPLE9_RA_MAX_SPILL_WORDS (AGX_APPLE9_MAX_SCRATCH_BYTES / 4u)

/* Default reserved 32-bit GPRs when spilling: r0..r4 (copy, index, swap). */
#define APPLE9_RA_SPILL_RESERVED_GPRS 5u

/*
 * Sideband 2-class SSA state (RA_GPR 0..95 vs RA_MEM 0..8191) for native
 * Apple9 VIR spilling, Braun SSA repair, and DSATUR spill-home coalescing.
 */
struct apple9_ra_spill_state {
   /* Bit v is set iff SSA value v resides in RA_MEM (32-bit spill frame words
    * 0..8191), clear iff it resides in RA_GPR (32-bit GPR units 0..95). */
   BITSET_WORD *is_mem;
   /* Bit v is set iff SSA value v is a rematerializable scalar immediate
    * carried directly into phi-edge lowering (never allocated a scratch word;
    * also marked in is_mem so GPR liveness and assignment skip it). */
   BITSET_WORD *is_imm;
   uint32_t *imm_val;
   unsigned is_mem_capacity;

   /* Tuple width in 32-bit GPR/word units for each SSA value (1 for scalar,
    * 2..8 for aligned multi-word tuples). Indexed by [0 .. is_mem_capacity-1]. */
   uint8_t *value_width;

   /* Optional 1-based spill-home identity per RA_MEM SSA value (original
    * variable ID + 1, or 0 if unassigned). Populated by apple9_ra_repair_ssa()
    * and consumed by apple9_ra_spill_coalesce_*(). */
   uint32_t *spill_home;
   unsigned spill_home_count;

   /* True if any real RA_MEM spill was emitted (as opposed to remat-only). */
   bool has_spills;

   /* True if immutable-uniform pure-ALU rematerialization was used. */
   bool uniform_remat_used;
};

struct apple9_ra_spill_options {
   /* Maximum 32-bit GPR budget (typically AGX_APPLE9_GPR_COUNT = 96). */
   unsigned max_gprs;

   /* Permanently reserved 32-bit GPRs (e.g., 0 or 2 for copy bridge). */
   unsigned reserved_gprs;

   /* Reserved 32-bit GPRs when spilling is allowed (default 5: r0..r4). */
   unsigned spill_reserved_gprs;

   /* If true, do not emit RA_MEM spills; only rematerialize. */
   bool remat_only;

   /* If true, disable immutable-uniform pure-ALU rematerialization (only
    * VIR_IMM constants remain rematerializable). */
   bool disable_uniform_remat;

   /* If true, keep a phi in GPR when at least one predecessor holds its source
    * in GPR (reloading on other edges); if false, spill the phi if any source
    * is spilled. Default for Apple9 is true. */
   bool spill_partial_phi_sources;

   /* If true, hoist loop-invariant spills to unique loop preheaders. */
   bool hoist_loop_invariant;

   /* If true, record spill_home[] during SSA repair for DSATUR coalescing. */
   bool track_spill_homes;
};

static inline struct apple9_ra_spill_options
apple9_ra_spill_default_options(void)
{
   struct apple9_ra_spill_options opts;
   opts.max_gprs = AGX_APPLE9_GPR_COUNT;
   opts.reserved_gprs = 0;
   opts.spill_reserved_gprs = APPLE9_RA_SPILL_RESERVED_GPRS;
   opts.remat_only = false;
   opts.disable_uniform_remat = false;
   opts.spill_partial_phi_sources = true;
   opts.hoist_loop_invariant = true;
   opts.track_spill_homes = true;
   return opts;
}

void apple9_ra_spill_state_init(struct apple9_ra_spill_state *state);
void apple9_ra_spill_state_finish(struct apple9_ra_spill_state *state);
bool apple9_ra_spill_state_ensure(struct apple9_ra_spill_state *state,
                                  unsigned value_count);

static inline bool
apple9_ra_value_is_mem(const struct apple9_ra_spill_state *state, uint32_t v)
{
   return state && state->is_mem && v < state->is_mem_capacity &&
          BITSET_TEST(state->is_mem, v);
}

static inline bool
apple9_ra_value_is_imm(const struct apple9_ra_spill_state *state, uint32_t v)
{
   return state && state->is_imm && v < state->is_mem_capacity &&
          BITSET_TEST(state->is_imm, v);
}

static inline unsigned
apple9_ra_value_width(const struct apple9_ra_spill_state *state, uint32_t v)
{
   if (!state || !state->value_width || v >= state->is_mem_capacity ||
       !state->value_width[v])
      return 1;
   return state->value_width[v];
}

/*
 * Native VIR Braun-Hack spiller operating directly on agx_apple9_vir_program
 * in 32-bit GPR units (0..95) + RA_MEM spill class (0..8191 words = 32 KiB).
 *
 * - Ensures logical CFG and canonical variable-arity VIR_PHI form.
 * - Finds immutable-uniform pure-ALU rematerialization trees (<= 4 ops).
 * - Runs Braun-Hack MIN spilling, block-boundary coupling, and loop-invariant
 *   spill hoisting over logical loop headers.
 * - Emits cross-class VIR_MOV (RA_GPR <-> RA_MEM) for Phase 2 physical spill
 *   lowering, repairs SSA via apple9_ra_repair_ssa(), and prunes dead pure ALU
 *   left behind by rematerialization.
 */
bool apple9_ra_spill(struct agx_apple9_vir_program *program,
                     struct apple9_ra_spill_state *state,
                     const struct apple9_ra_spill_options *opts);

/*
 * Compute peak live 32-bit GPR pressure across all active logical CFG blocks
 * (in power-of-two aligned 32-bit GPR units, excluding RA_MEM and publication
 * values). Useful for verifying the spiller post-condition (demand <= k).
 */
unsigned apple9_ra_spill_calc_gpr_demand(
   const struct agx_apple9_vir_program *program,
   const struct apple9_ra_spill_state *state);

/*
 * Native VIR Braun SSA repair over agx_apple9_block logical CFG.
 *
 * - Reconstructs valid SSA after spilling/reloading/rematerialization.
 * - Inserts canonical variable-arity VIR_PHI (inline <= 10 and sideband > 10
 *   preds via agx_apple9_vir_emit_phi_canonical()).
 * - Eliminates trivial phis, optionally populates state->spill_home[] for
 *   RA_MEM values, and compacts/reindexes SSA values.
 */
bool apple9_ra_repair_ssa(struct agx_apple9_vir_program *program,
                          struct apple9_ra_spill_state *state,
                          bool track_spill_homes);

/*
 * Returns true if program contains a returning device atomic, which reserves
 * invocation-frame word 0 (16 B granule = 4 words) below spills.
 */
bool apple9_ra_returns_device_atomic(
   const struct agx_apple9_vir_program *program);

/*
 * Base 32-bit word offset in the invocation frame where spill slots begin:
 * 4 words (16 B) when apple9_ra_returns_device_atomic(program), else 0.
 */
unsigned apple9_ra_spill_base_words(
   const struct agx_apple9_vir_program *program);

/*
 * Compute total per-invocation scratch bytes (rounded to 16 B granule),
 * accounting for the 16 B returning-device-atomic reservation, spill frame
 * word slots (including base offset), and program->private_size.
 */
unsigned apple9_ra_spill_scratch_bytes(
   const struct agx_apple9_vir_program *program, unsigned frame_words);

/* Opaque interference graph for RA_MEM SSA values. */
struct apple9_ra_spill_interference;

/*
 * Build interference graph over all RA_MEM SSA values while program is in SSA.
 * Returns NULL if there are no RA_MEM values (or > 4096 dense bound).
 */
struct apple9_ra_spill_interference *
apple9_ra_spill_coalesce_prepare(const struct agx_apple9_vir_program *program,
                                 const struct apple9_ra_spill_state *state);

void apple9_ra_spill_coalesce_destroy(struct apple9_ra_spill_interference *g);

/*
 * Bounded DSATUR spill-home recoloring against an existing RA_MEM assignment:
 * - mem_slot[v] holds each RA_MEM value's assigned 32-bit invocation-frame
 *   word slot (>= apple9_ra_spill_base_words(program)), length >= value_count.
 * - *frame_words holds the current spill-region high-water mark in 32-bit
 *   words (including atomic base offset); updated on success.
 * - If contraction cannot fit within *frame_words, leaves mem_slot[] and
 *   *frame_words unchanged and returns false.
 */
bool apple9_ra_spill_coalesce_apply(
   const struct agx_apple9_vir_program *program,
   const struct apple9_ra_spill_state *state,
   struct apple9_ra_spill_interference *g,
   uint32_t *mem_slot, unsigned *frame_words);

/*
 * Direct DSATUR spill-home coloring from scratch (no prior assignment needed):
 * - Assigns minimal 32-bit invocation-frame word slots (>= base_words) to all
 *   RA_MEM values in mem_slot[0 .. program->value_count - 1] (non-mem values
 *   are set to UINT32_MAX).
 * - Sets *frame_words to the high-water mark in 32-bit words (0 if no RA_MEM
 *   values exist), accounting for the 16 B atomic base offset when needed.
 * - Returns false if total scratch bytes exceed AGX_APPLE9_MAX_SCRATCH_BYTES.
 */
bool apple9_ra_spill_coalesce_color(
   const struct agx_apple9_vir_program *program,
   const struct apple9_ra_spill_state *state,
   uint32_t *mem_slot, unsigned *frame_words);

/* Internal shared helpers for the Phase 1B spill/repair/coalesce modules. */
bool apple9_ra_vir_is_publication_instr(const struct agx_apple9_vir_instr *ins);
bool apple9_ra_vir_after_logical_end(const struct agx_apple9_vir_instr *ins);
bool apple9_ra_vir_grow_values(struct agx_apple9_vir_program *program,
                               struct apple9_ra_spill_state *state,
                               unsigned extra_count, uint32_t *base_out);

#ifdef __cplusplus
}
#endif

#endif /* APPLE9_RA_SPILL_H */
