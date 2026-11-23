/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef APPLE9_RA_SCHED_H
#define APPLE9_RA_SCHED_H

#include <stdbool.h>
#include <stdint.h>
#include "util/macros.h"
#include "util/sparse_bitset.h"
#include "agx_apple9_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Implicit hardware resource classes modeled by the native VIR DAG scheduler. */
enum apple9_ra_sched_resource {
   APPLE9_RA_SCHED_STATE         = BITFIELD_BIT(0),
   APPLE9_RA_SCHED_MEMORY        = BITFIELD_BIT(1),
   APPLE9_RA_SCHED_PUBLICATION   = BITFIELD_BIT(2),
   APPLE9_RA_SCHED_INTERPOLATION = BITFIELD_BIT(3),
   APPLE9_RA_SCHED_TILE          = BITFIELD_BIT(4),
   APPLE9_RA_SCHED_MESH          = BITFIELD_BIT(5),
   APPLE9_RA_SCHED_ALL           = BITFIELD_MASK(6),
};

/* Native VIR CFG liveness output (32-bit GPR SSA units, indexed by
 * block->cfg_index for active logical CFG blocks). */
struct apple9_ra_liveness_info {
   void *mem_ctx;
   unsigned num_cfg_blocks;
   unsigned value_count;
   struct u_sparse_bitset *live_in;
   struct u_sparse_bitset *live_out;
   unsigned *block_peak_gprs;
   unsigned peak_live_gprs;
};

/* Tuning options for the native VIR pressure / latency / resource scheduler. */
struct apple9_ra_sched_options {
   /* Enable latency-hiding scheduling using apple9_ra_dependency_delay().
    * When false, optimizes strictly for minimum 32-bit GPR register pressure. */
   bool latency;
   /* Maximum allowed peak 32-bit GPR pressure increase (in 32-bit GPR units)
    * when hiding latency stalls. Ignored when !latency. */
   unsigned pressure_slack;
   /* Automatically derive per-block occupancy-tier slack from exact post-DCE
    * CFG liveness and avoid overriding lower-stall schedules for intra-tier
    * 1-GPR savings. */
   bool auto_tier_slack;
   /* Output: largest peak 32-bit GPR pressure increase rejected across blocks. */
   unsigned rejected_pressure_increase;
   /* Optional output: when non-NULL, transfers ownership of the post-schedule
    * CFG liveness info to the caller (initialized on success; destroy with
    * apple9_ra_liveness_finish) instead of freeing and recomputing it. */
   struct apple9_ra_liveness_info *liveness_out;
};

void apple9_ra_liveness_init(struct apple9_ra_liveness_info *info);
void apple9_ra_liveness_finish(struct apple9_ra_liveness_info *info);

/* Resolved register-bank and contiguity contract for a VIR operand group
 * (one destination tuple or one contiguous source group starting at
 * `source`). Shared across RA scheduling, assignment, and validation. */
struct apple9_ra_operand_group {
   uint8_t width;      /* Contiguous 32-bit GPR count (1 for scalar). */
   uint8_t min_gpr;    /* Inclusive minimum base 32-bit GPR. */
   uint8_t max_gpr;    /* Inclusive maximum base 32-bit GPR (<= 96 - width). */
   uint8_t align_gprs; /* Power-of-two base GPR alignment (>= 1). */
   bool clobber;       /* Hardware overwrites this source group at issue. */
   bool reuse_killed;  /* Destination may reuse a killed non-clobber source GPR. */
   const struct agx_apple9_operand_constraint *constraint;
};

/* Five independently schedulable Apple9 (G16G) execution pipelines measured on
 * silicon (70%..97% concurrent dual-issue overlap across distinct pipes). */
enum apple9_ra_exec_pipe {
   APPLE9_RA_PIPE_NONE = 0,
   APPLE9_RA_PIPE_MAIN,  /* 128-wide FP32/FP16/INT-add/logic/select/0xCF MMA */
   APPLE9_RA_PIPE_IMUL,  /* 32-wide quarter-rate IMAD / IMUL_WIDE */
   APPLE9_RA_PIPE_SHIFT, /* 32-wide quarter-rate ISHL / USHR / ISHR / EXTR */
   APPLE9_RA_PIPE_SFU,   /* 32-wide quarter-rate SFU */
   APPLE9_RA_PIPE_MEM,   /* Load / store / texture / shared / tile / iter */
};

/* Shared VIR RA classification, operand-group, and latency helpers. */
bool apple9_ra_after_logical_end(const struct agx_apple9_vir_instr *ins);
bool apple9_ra_is_publication_instr(const struct agx_apple9_vir_instr *ins);
bool apple9_ra_is_publication_val(const struct agx_apple9_vir_program *p,
                                  uint32_t value);
bool apple9_ra_async(const struct agx_apple9_vir_instr *ins);
bool apple9_ra_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins);
unsigned apple9_ra_source_width(const struct agx_apple9_vir_instr *ins,
                                unsigned source);
const struct agx_apple9_operand_constraint *
apple9_ra_source_constraint(const struct agx_apple9_vir_instr *ins,
                            unsigned source);
struct apple9_ra_operand_group
apple9_ra_source_group(const struct agx_apple9_vir_instr *ins,
                       unsigned source);
struct apple9_ra_operand_group
apple9_ra_dest_group(const struct agx_apple9_vir_instr *ins);
void apple9_ra_schedule_resources(const struct agx_apple9_vir_instr *ins,
                                  uint32_t *reads, uint32_t *writes);
unsigned apple9_ra_dependency_delay(const struct agx_apple9_vir_instr *ins);
enum apple9_ra_exec_pipe
apple9_ra_exec_pipe(const struct agx_apple9_vir_instr *ins);

/* Native VIR partial dead code elimination. */
bool apple9_ra_dce(struct agx_apple9_vir_program *p);

/* Native VIR sparse-bitset CFG liveness, kill_mask, and peak 32-bit GPR
 * pressure analysis. `out` is initialized by this call (destroy with
 * apple9_ra_liveness_finish); `out` may be NULL if only kill_mask and
 * p->peak_live_gprs are needed. */
bool apple9_ra_liveness(struct agx_apple9_vir_program *p,
                        struct apple9_ra_liveness_info *out);

/* Native VIR intra-block pressure / latency / resource DAG scheduler. */
bool apple9_ra_pressure_schedule(struct agx_apple9_vir_program *p);
bool apple9_ra_pressure_schedule_opts(struct agx_apple9_vir_program *p,
                                      struct apple9_ra_sched_options *opts);

#ifdef __cplusplus
}
#endif

#endif /* APPLE9_RA_SCHED_H */
