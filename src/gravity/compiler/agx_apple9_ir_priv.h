/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_IR_PRIV_H
#define AGX_APPLE9_IR_PRIV_H

#include "agx_apple9_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal VIR helpers shared between agx_apple9_ir.c and
 * agx_apple9_ir_test_support.c (test-only support TU). */

bool agx_apple9_dependency_slot_valid(enum agx_apple9_dependency_layout layout,
                                      uint8_t slot);

bool agx_apple9_scalar_load_token_for_slot(enum agx_apple9_scoreboard_slot slot,
                                           uint16_t *token);

bool agx_apple9_mma_flags_valid(uint32_t flags);

bool agx_apple9_vir_values_form_tuple(
   const struct agx_apple9_vir_program *program, const uint32_t *values,
   unsigned count);

bool agx_apple9_atomic_op_valid(enum agx_apple9_atomic_op op);

unsigned agx_apple9_vir_dest_components(
   const struct agx_apple9_vir_instr *instruction);

bool agx_apple9_vir_is_control_side_effect(enum agx_apple9_vir_opcode op);

bool agx_apple9_vir_publishes(const struct agx_apple9_vir_instr *instruction);

unsigned agx_apple9_publication_bound(
   const struct agx_apple9_vir_instr *instruction);

unsigned agx_apple9_publication_end(
   const struct agx_apple9_vir_program *program, unsigned producer);

bool agx_apple9_validate_value_files(
   const struct agx_apple9_vir_program *program, const char **reason);

bool agx_apple9_vir_is_pending_load(const struct agx_apple9_vir_instr *ins);

bool agx_apple9_materialize_spill_completion(
   struct agx_apple9_vir_program *program, struct agx_apple9_vir_instr *store);

bool agx_apple9_materialize_pending_result_at(
   struct agx_apple9_vir_program *program, unsigned producer_index,
   unsigned insertion, const char **reason);

unsigned agx_apple9_vir_first_consumer(
   const struct agx_apple9_vir_program *program, unsigned producer_index);

#ifdef __cplusplus
}
#endif

#endif /* AGX_APPLE9_IR_PRIV_H */
