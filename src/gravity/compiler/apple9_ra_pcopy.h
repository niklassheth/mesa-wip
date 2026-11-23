/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef APPLE9_RA_PCOPY_H
#define APPLE9_RA_PCOPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_apple9_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum 32-bit spill word slots per invocation (32 KiB). */
#define APPLE9_RA_MAX_SPILL_SLOTS 8192u

/* Tag placed in ins->target to mark physical-operand parallel-copy/spill
 * instructions (VIR_MOV, VIR_SWAP, VIR_IMM, VIR_SPILL_LOAD, VIR_SPILL_STORE)
 * before program->physical is set. */
#define APPLE9_RA_PHYS_OP_TAG 0x50485953u

/* Default Apple9 spill/copy reserved scratch GPRs (32-bit GPR indices). */
#define APPLE9_RA_DEFAULT_COPY_REG  1u
#define APPLE9_RA_DEFAULT_SWAP_REG0 3u
#define APPLE9_RA_DEFAULT_SWAP_REG1 4u

enum apple9_ra_loc_kind {
   APPLE9_RA_LOC_GPR = 0,
   APPLE9_RA_LOC_MEM = 1,
   APPLE9_RA_LOC_IMM = 2,
   APPLE9_RA_LOC_UNDEF = 3,
};

struct apple9_ra_loc {
   enum apple9_ra_loc_kind kind;
   uint32_t value; /* GPR 0..95, MEM slot 0..8191, or 32-bit immediate */
};

static inline struct apple9_ra_loc
apple9_ra_loc_gpr(unsigned gpr)
{
   return (struct apple9_ra_loc){ .kind = APPLE9_RA_LOC_GPR, .value = gpr };
}

static inline struct apple9_ra_loc
apple9_ra_loc_mem(unsigned slot)
{
   return (struct apple9_ra_loc){ .kind = APPLE9_RA_LOC_MEM, .value = slot };
}

static inline struct apple9_ra_loc
apple9_ra_loc_imm(uint32_t imm)
{
   return (struct apple9_ra_loc){ .kind = APPLE9_RA_LOC_IMM, .value = imm };
}

static inline struct apple9_ra_loc
apple9_ra_loc_undef(void)
{
   return (struct apple9_ra_loc){ .kind = APPLE9_RA_LOC_UNDEF, .value = 0 };
}

struct apple9_ra_copy {
   struct apple9_ra_loc dst; /* Must be GPR or MEM */
   struct apple9_ra_loc src; /* GPR, MEM, IMM, or UNDEF */
   uint8_t count;            /* 0 or 1 = scalar word; 2..4 = adjacent tuple */
};

enum apple9_ra_pcopy_op_kind {
   APPLE9_RA_PCOPY_MOV,     /* GPR<-GPR, GPR<-MEM, or MEM<-GPR (never MEM<-MEM) */
   APPLE9_RA_PCOPY_MOV_IMM, /* GPR (< mov_imm_max_gpr) <- 32-bit immediate */
   APPLE9_RA_PCOPY_SWAP,    /* GPR <-> GPR (zero-temp cycle resolution) */
};

struct apple9_ra_pcopy_op {
   enum apple9_ra_pcopy_op_kind kind;
   struct apple9_ra_loc dst; /* For SWAP: first GPR */
   struct apple9_ra_loc src; /* For SWAP: second GPR; for MOV_IMM: IMM */
};

struct apple9_ra_pcopy_opts {
   /* 96-bit mask of GPRs whose values are live across/after the parallel copy
    * and must not be clobbered unless they are destinations of the copy. */
   uint64_t live_gprs[2];
   /* Reserved scratch GPRs (e.g. COPY_REG = r0/r1, swap temp = r3), or
    * AGX_APPLE9_PHYS_INVALID if unavailable. */
   uint8_t scratch_gpr;
   uint8_t scratch_gpr2;
   /* Exclusive upper bound for hunting free temp GPRs (0 defaults to 96;
    * set live_gprs to all-ones to forbid free temp hunting). */
   uint8_t max_temp_gpr;
   /* Exclusive upper bound on GPRs directly encodable by MOV_IMM
    * (0 defaults to 64, matching Apple9 MOV_IMM32 r0..r63). */
   uint8_t mov_imm_max_gpr;
   /* Prefer VIR_SWAP over temp-assisted MOVs for 2-element GPR cycles even
    * when a free temp GPR is available. */
   bool prefer_swap_2cycle;
};

typedef bool (*apple9_ra_pcopy_emit_fn)(void *ctx,
                                        const struct apple9_ra_pcopy_op *op);

/* Per-SSA-value register/spill assignment view shared by phi-edge lowering
 * and post-RA validation. */
struct apple9_ra_assignment {
   /* Per-value physical GPR [0..95]; if NULL, falls back to program->phys. */
   const uint8_t *gpr;
   /* Optional bitset [BITSET_WORDS(program->value_count)] marking RA_MEM. */
   const BITSET_WORD *is_mem;
   /* Optional bitset + value table marking rematerializable immediate phi
    * sources carried directly into phi-edge lowering (never allocated a slot). */
   const BITSET_WORD *is_imm;
   const uint32_t *imm_val;
   /* Optional 32-bit spill word slot [0..max_mem_slots-1] when is_mem is set. */
   const uint16_t *mem_slot;
   /* Upper bound on spill slots (0 if no spills). */
   unsigned max_mem_slots;
   /* True if VIR_PHI edge copies have already been emitted into predecessor
    * blocks (so predecessor exit holds phi src in phi dest's location). */
   bool phis_lowered;
};

struct apple9_ra_validate_error {
   bool failed;
   unsigned block_index;
   unsigned instr_index;
   char message[256];
};

/* Logical block-end helpers (mirrors apple9_after_logical_end in agx_apple9_ra.c). */
bool apple9_ra_after_logical_end(const struct agx_apple9_vir_instr *ins);
struct agx_apple9_vir_instr *
apple9_ra_block_logical_end_anchor(const struct agx_apple9_block *block);

/* Core sized GPR[96] + MEM[max_slot] parallel-copy lowerer.
 * Emits sequential primitive operations via `emit(ctx, &op)`. */
bool apple9_ra_lower_parallel_copies(const struct apple9_ra_copy *copies,
                                     unsigned num_copies,
                                     const struct apple9_ra_pcopy_opts *opts,
                                     apple9_ra_pcopy_emit_fn emit,
                                     void *ctx);

/* Lower a parallel copy and emit native VIR instructions (VIR_MOV, VIR_SWAP,
 * VIR_IMM, VIR_SPILL_LOAD, VIR_SPILL_STORE, tagged with APPLE9_RA_PHYS_OP_TAG)
 * into `block` right before `before` (or at block end if `before == NULL`). */
bool apple9_ra_emit_parallel_copies_vir(struct agx_apple9_vir_program *program,
                                        struct agx_apple9_block *block,
                                        struct agx_apple9_vir_instr *before,
                                        const struct apple9_ra_copy *copies,
                                        unsigned num_copies,
                                        const struct apple9_ra_pcopy_opts *opts);

/* Lower canonical VIR_PHI edge copies for predecessor `pred` -> successor `succ`
 * right before `apple9_ra_block_logical_end_anchor(pred)`.
 * If `succ == NULL`, lowers all outgoing CFG edges of `pred`. */
bool apple9_ra_lower_phi_edge(struct agx_apple9_vir_program *program,
                              struct agx_apple9_block *pred,
                              struct agx_apple9_block *succ,
                              const struct apple9_ra_assignment *assign,
                              const struct apple9_ra_pcopy_opts *opts);

/* Lower all canonical VIR_PHI edges across the entire program into each
 * predecessor block right before `apple9_ra_after_logical_end(ins)`.
 * If `remove_phis` is true, removes the lowered VIR_PHI pseudos afterward. */
bool apple9_ra_lower_all_phi_edges(struct agx_apple9_vir_program *program,
                                   const struct apple9_ra_assignment *assign,
                                   const struct apple9_ra_pcopy_opts *opts,
                                   bool remove_phis);

/* Pre-RA SSA validator: checks logical CFG symmetry & reachability, strict
 * dominance of defs over uses, VIR_PHI placement at block entry + arity ==
 * num_cfg_preds, valid SSA value indices & single definition, and
 * VIR_COLLECT / VIR_SPLIT arity. */
bool apple9_ra_validate_ssa(const struct agx_apple9_vir_program *program,
                            struct apple9_ra_validate_error *err);

/* Post-RA register-file validator: compact GPR[96] + MEM[max_slot] tracking
 * across all instructions and CFG edges (including loops), kill_mask /
 * clobber / reuse_killed legality, and encoding register-class limits
 * (r0..r15, r0..r63, r0..r95, aligned contiguous tuples). */
bool apple9_ra_validate_post_ra(const struct agx_apple9_vir_program *program,
                                const struct apple9_ra_assignment *assign,
                                struct apple9_ra_validate_error *err);

#ifdef __cplusplus
}
#endif

#endif /* APPLE9_RA_PCOPY_H */
