/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef APPLE9_RA_ASSIGN_H
#define APPLE9_RA_ASSIGN_H

#include <stdbool.h>
#include <stdint.h>

#include "agx_apple9_ir.h"
#include "apple9_ra_pcopy.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Configuration options for the native Apple9 SSA register allocator
 * operating directly on struct agx_apple9_vir_program in 32-bit GPR units
 * (0..95) + RA_MEM spill frame words (0..8191 = 32 KiB max).
 */
struct apple9_ra_assign_options {
   /* Maximum 32-bit GPR budget (0 defaults to AGX_APPLE9_GPR_COUNT = 96). */
   unsigned max_gprs;

   /* Permanently reserved 32-bit GPRs (0, or 2 for r0..r1 copy bridge).
    * When spilling occurs, at least APPLE9_RA_SPILL_RESERVED_GPRS (5: r0..r4)
    * are reserved automatically. */
   unsigned reserved_gprs;

   /* Enable bank-scarcity and operand-preference costing during assignment. */
   bool cost_register_constraints;

   /* Enable latency-hiding scheduling in apple9_ra_pressure_schedule_opts(). */
   bool latency_schedule;

   /* Pressure slack (in 32-bit GPR units) for latency scheduling when
    * latency_schedule is enabled; 0 enforces strict minimum pressure. */
   unsigned pressure_slack;

   /* Disable immutable-uniform pure-ALU rematerialization when spilling. */
   bool disable_uniform_remat;

   /* Run apple9_ra_validate_ssa() before scheduling/assignment and
    * apple9_ra_validate_post_ra() after phi-edge lowering. */
   bool validate;

   /* Stop after SSA assignment, phi-edge lowering, and optional post-RA
    * validation without rewriting to canonical physical VIR (program->physical
    * remains false and program->phys[] holds the per-SSA-value GPR map). */
   bool keep_ssa_ir;

   /* Skip pre-RA DCE or intra-block pressure/latency scheduling. */
   bool skip_dce;
   bool skip_schedule;

   /* Skip the built-in physical scoreboard finalizer (leaves
    * program->dependencies_finalized = false for callers that run an external
    * physical scoreboard scheduler such as apple9_schedule_physical()). */
   bool skip_physical_schedule;

   /* Force the spiller to run even when natural GPR demand fits in max_gprs
    * (clamps the GPR budget to the minimum legal ceiling). */
   bool force_spill;

   /* Extra free GPR headroom reserved in the spiller below max_gprs (0 by
    * default). Used when extreme 96-GPR spilling shaders need temporary
    * registers for post-spill tuple re-collection or constrained eviction. */
   unsigned spill_headroom;
};

struct apple9_ra_assign_result {
   /* Human-readable error message when apple9_ra_allocate_native() fails. */
   const char *error;

   /* Detailed validator diagnostic when opts->validate catches an error. */
   struct apple9_ra_validate_error validate_err;

   /* Allocation summary metrics. */
   unsigned max_phys_gpr;        /* Highest 32-bit GPR assigned (0..95) */
   unsigned peak_live_gprs;      /* Peak live 32-bit GPR count */
   unsigned reserved_gprs;       /* Reserved GPR count used (0, 2, or 5) */
   unsigned spill_slots;         /* 32-bit spill word count (excl. atomic base) */
   unsigned scratch_size;        /* Total per-invocation scratch bytes (16B aligned) */
   unsigned num_copies_emitted;  /* Physical GPR moves + swaps emitted */
   unsigned num_spills_emitted;  /* Physical spill stores + loads emitted */
   bool has_spills;              /* True if any RA_MEM spill occurred */
   bool uniform_remat_used;      /* True if immutable-uniform remat fired */
   bool needs_copy_register;     /* True if a high-bank immediate needed COPY_REG */
   bool needs_spill_headroom;    /* True if eviction ran out of free GPRs */
};

static inline struct apple9_ra_assign_options
apple9_ra_assign_default_options(void)
{
   struct apple9_ra_assign_options opts;
   opts.max_gprs = AGX_APPLE9_GPR_COUNT;
   opts.reserved_gprs = 0;
   opts.cost_register_constraints = false;
   opts.latency_schedule = false;
   opts.pressure_slack = 0;
   opts.disable_uniform_remat = false;
   opts.validate = false;
   opts.keep_ssa_ir = false;
   opts.skip_dce = false;
   opts.skip_schedule = false;
   opts.skip_physical_schedule = false;
   opts.force_spill = false;
   opts.spill_headroom = 0;
   return opts;
}

/*
 * End-to-end native Apple9 SSA register allocator driver:
 *   1. Logical CFG construction + canonical variable-arity VIR_PHI lowering
 *   2. Multi-word tuple canonicalization (16-entry intra-block read-only reuse
 *      cache + retain_mask) and AGX_APPLE9_OPERAND_CLOBBER copy insertion
 *   3. Pre-RA DCE + optional pre-RA SSA validation + pressure/latency schedule
 *   4. Demand check -> Braun-Hack spilling + SSA repair + DSATUR spill-home
 *      coloring (respecting 16 B returning-device-atomic frame reservation)
 *   5. Liveness (live_in / live_out / kill_mask)
 *   6. Dominance-order 32-bit GPR SSA register assignment (phi_web coalescing,
 *      on-demand encoding constraints, scarcity_floor, reuse_killed, aligned
 *      tuples, acyclic SSA live-range splitting / constrained relocation)
 *   7. Parallel-copy phi-edge lowering + optional post-RA register-file check
 *   8. Physical VIR lowering (IOR copies, 3-IXOR swaps, SPILL_LOAD/SPILL_STORE
 *      and high-offset PRIVATE_LOAD/PRIVATE_STORE using COPY_REG=1 and
 *      SPILL_INDEX_REG=2, skipping COPY_REG on killed SAVE, ABI_RETURN exports,
 *      empty-else cleanup, and physical scoreboard finalization)
 */
bool apple9_ra_allocate_native(struct agx_apple9_vir_program *program,
                               const struct apple9_ra_assign_options *opts,
                               struct apple9_ra_assign_result *result);

#ifdef __cplusplus
}
#endif

#endif /* APPLE9_RA_ASSIGN_H */
