/*
 * Copyright 2026 The Asahi Linux Contributors
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 *
 * Native Apple9 VIR implementation of "Simple and Efficient Construction of
 * Static Single Assignment Form" by Braun et al., operating directly on
 * agx_apple9_block logical CFG with canonical variable-arity VIR_PHI
 * (both inline <= 10 and sideband > 10 predecessors).
 */

#include "apple9_ra_spill.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct incomplete_phi {
   struct agx_apple9_vir_instr *phi;
   uint32_t orig_var;
};

struct repair_block {
   bool sealed;

   /* Sparse map: original variable ID -> current SSA uint32_t value (+1). */
   struct hash_table_u64 *defs;

   /* Incomplete phis placed at an unsealed loop header. */
   struct util_dynarray incomplete_phis;
};

struct repair_ctx {
   struct agx_apple9_vir_program *program;
   struct apple9_ra_spill_state *state;
   struct agx_apple9_block **cfg_blocks;
   unsigned num_blocks;

   /* Original value_count before repair began. */
   unsigned n;

   /* Per-block state indexed by cfg_index. */
   struct repair_block *blocks;

   /* Optional map: RA_MEM SSA value -> (orig_var + 1). */
   struct hash_table_u64 *homes;

   /* Set of VIR_PHI instruction pointers created during this repair pass. */
   struct hash_table *created_phis;

   bool failed;
};

static inline struct repair_block *
repair_block(struct repair_ctx *ctx, const struct agx_apple9_block *block)
{
   return &ctx->blocks[block->cfg_index];
}

static void
record_write(struct repair_ctx *ctx, struct agx_apple9_block *block,
             uint32_t orig_var, uint32_t val)
{
   assert(orig_var < ctx->n);
   if (ctx->homes && apple9_ra_value_is_mem(ctx->state, val)) {
      _mesa_hash_table_u64_insert(ctx->homes, val,
                                  (void *)(uintptr_t)(orig_var + 1u));
   }
   struct hash_table_u64 *defs = repair_block(ctx, block)->defs;
   _mesa_hash_table_u64_insert(defs, orig_var, (void *)(uintptr_t)(val + 1u));
}

static uint32_t
alloc_ssa_like(struct repair_ctx *ctx, uint32_t orig_var, unsigned width)
{
   uint32_t val = AGX_APPLE9_VREG_INVALID;
   if (!apple9_ra_vir_grow_values(ctx->program, ctx->state, width, &val)) {
      ctx->failed = true;
      return AGX_APPLE9_VREG_INVALID;
   }

   bool is_mem = apple9_ra_value_is_mem(ctx->state, orig_var);
   if (ctx->state) {
      for (unsigned c = 0; c < MAX2(width, 1u); ++c) {
         if (val + c < ctx->state->is_mem_capacity) {
            if (is_mem)
               BITSET_SET(ctx->state->is_mem, val + c);
            ctx->state->value_width[val + c] = (c == 0) ? (uint8_t)width : 1u;
         }
      }
   }
   if (!is_mem && orig_var < ctx->n && ctx->program->max_phys)
      ctx->program->max_phys[val] = ctx->program->max_phys[orig_var];

   return val;
}

static struct agx_apple9_vir_instr *
emit_repair_phi(struct repair_ctx *ctx, struct agx_apple9_block *block,
                uint32_t orig_var, uint32_t *val_out)
{
   unsigned width = apple9_ra_value_width(ctx->state, orig_var);
   struct agx_apple9_block *saved = ctx->program->current_block;
   ctx->program->current_block = block;
   uint32_t val = agx_apple9_vir_emit_phi_canonical(ctx->program, block,
                                                    block->num_cfg_preds);
   ctx->program->current_block = saved;
   if (val == AGX_APPLE9_VREG_INVALID) {
      ctx->failed = true;
      return NULL;
   }

   struct agx_apple9_vir_instr *phi =
      ctx->program->instructions[ctx->program->instruction_count - 1];

   if (width > 1) {
      if (!apple9_ra_vir_grow_values(ctx->program, ctx->state, width - 1, NULL)) {
         ctx->failed = true;
         return NULL;
      }
      phi->dest_components = width;
   } else if (ctx->state &&
              !apple9_ra_spill_state_ensure(ctx->state, ctx->program->value_count)) {
      ctx->failed = true;
      return NULL;
   }

   bool is_mem = apple9_ra_value_is_mem(ctx->state, orig_var);
   if (ctx->state) {
      for (unsigned c = 0; c < MAX2(width, 1u); ++c) {
         if (val + c < ctx->state->is_mem_capacity) {
            if (is_mem)
               BITSET_SET(ctx->state->is_mem, val + c);
            ctx->state->value_width[val + c] = (c == 0) ? (uint8_t)width : 1u;
         }
      }
   }
   if (!is_mem && orig_var < ctx->n && ctx->program->max_phys)
      ctx->program->max_phys[val] = ctx->program->max_phys[orig_var];

   struct agx_apple9_vir_instr *first_non_prefix = NULL;
   for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(block);
        I; I = agx_apple9_instr_next(I)) {
      if (I == phi)
         continue;
      if (!agx_apple9_is_block_entry_prefix(I)) {
         first_non_prefix = I;
         break;
      }
   }
   if (first_non_prefix)
      agx_apple9_vir_move_before(ctx->program, phi, first_non_prefix);
   else
      agx_apple9_vir_move_to_block_start(ctx->program, phi, block);
   _mesa_hash_table_insert(ctx->created_phis, phi, phi);
   *val_out = val;
   return phi;
}

static void add_phi_operands(struct repair_ctx *ctx,
                             struct agx_apple9_block *block,
                             struct agx_apple9_vir_instr *phi,
                             uint32_t orig_var);

static uint32_t
resolve_read(struct repair_ctx *ctx, struct agx_apple9_block *block,
             uint32_t orig_var)
{
   if (ctx->failed)
      return orig_var;

   struct repair_block *rb = repair_block(ctx, block);
   void *local = _mesa_hash_table_u64_search(rb->defs, orig_var);
   if (local)
      return (uint32_t)((uintptr_t)local - 1u);

   unsigned nr_preds = block->num_cfg_preds;
   if (nr_preds == 0)
      return orig_var;

   uint32_t val = AGX_APPLE9_VREG_INVALID;

   if (block->loop_header && !rb->sealed) {
      struct agx_apple9_vir_instr *phi =
         emit_repair_phi(ctx, block, orig_var, &val);
      if (!phi)
         return orig_var;
      struct incomplete_phi inc = {.phi = phi, .orig_var = orig_var};
      util_dynarray_append(&rb->incomplete_phis, inc);
   } else if (nr_preds == 1) {
      val = resolve_read(ctx, block->cfg_preds[0], orig_var);
   } else {
      struct agx_apple9_vir_instr *phi =
         emit_repair_phi(ctx, block, orig_var, &val);
      if (!phi)
         return orig_var;
      record_write(ctx, block, orig_var, val);
      add_phi_operands(ctx, block, phi, orig_var);
   }

   record_write(ctx, block, orig_var, val);
   return val;
}

static void
add_phi_operands(struct repair_ctx *ctx, struct agx_apple9_block *block,
                 struct agx_apple9_vir_instr *phi, uint32_t orig_var)
{
   for (unsigned p = 0; p < block->num_cfg_preds; ++p) {
      uint32_t sv = resolve_read(ctx, block->cfg_preds[p], orig_var);
      *agx_apple9_vir_src_ptr(phi, p) = sv;
   }
}

static void
seal_block(struct repair_ctx *ctx, struct agx_apple9_block *block)
{
   struct repair_block *rb = repair_block(ctx, block);
   if (rb->sealed)
      return;
   rb->sealed = true;

   util_dynarray_foreach(&rb->incomplete_phis, struct incomplete_phi, inc) {
      add_phi_operands(ctx, block, inc->phi, inc->orig_var);
   }
   util_dynarray_clear(&rb->incomplete_phis);
}

static void
seal_loop_headers(struct repair_ctx *ctx, struct agx_apple9_block *exit)
{
   for (unsigned s = 0; s < 2; ++s) {
      struct agx_apple9_block *succ = exit->cfg_succs[s];
      if (!succ || !succ->cfg_active || !succ->loop_header)
         continue;

      bool any_unprocessed = false;
      for (unsigned p = 0; p < succ->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = succ->cfg_preds[p];
         if (pred && pred->cfg_active && pred->cfg_index > exit->cfg_index) {
            any_unprocessed = true;
            break;
         }
      }

      if (!any_unprocessed)
         seal_block(ctx, succ);
   }
}

/* -------------------------------------------------------------------------
 * Trivial PHI elimination (handles inline <= 10 and sideband > 10 preds)
 * ------------------------------------------------------------------------- */

struct trivial_phi_use {
   struct agx_apple9_vir_instr *I;
   unsigned s;
   struct trivial_phi_use *next;
};

static void
apple9_opt_trivial_phi(struct agx_apple9_vir_program *program,
                       struct agx_apple9_block **cfg_blocks,
                       unsigned num_blocks)
{
   unsigned alloc = program->value_count;
   if (!alloc)
      return;

   unsigned nr_uses = 0;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I))
         nr_uses += agx_apple9_vir_nr_srcs(I);
   }

   struct trivial_phi_use *pool = malloc(MAX2(nr_uses, 1u) * sizeof(*pool));
   struct trivial_phi_use **uses = calloc(alloc, sizeof(*uses));
   BITSET_WORD *candidate = BITSET_CALLOC(alloc);
   BITSET_WORD *removed = BITSET_CALLOC(alloc);
   struct util_dynarray worklist = UTIL_DYNARRAY_INIT;
   struct util_dynarray to_remove = UTIL_DYNARRAY_INIT;

   if (!pool || !uses || !candidate || !removed) {
      free(pool);
      free(uses);
      free(candidate);
      free(removed);
      return;
   }

   nr_uses = 0;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v >= alloc)
               continue;
            pool[nr_uses] = (struct trivial_phi_use){
               .I = I,
               .s = s,
               .next = uses[v],
            };
            uses[v] = &pool[nr_uses++];
         }
      }
   }

   /* Collect leading phis at the start of every active block. */
   for (int bi = (int)num_blocks - 1; bi >= 0; --bi) {
      agx_apple9_foreach_phi(cfg_blocks[bi], phi) {
         if (phi->dest < alloc)
            BITSET_SET(candidate, phi->dest);
      }
   }
   for (int bi = (int)num_blocks - 1; bi >= 0; --bi) {
      for (struct agx_apple9_vir_instr *phi = agx_apple9_block_last_instr(cfg_blocks[bi]);
           phi; phi = agx_apple9_instr_prev(phi)) {
         if (phi->op == AGX_APPLE9_VIR_PHI && phi->dest < alloc &&
             BITSET_TEST(candidate, phi->dest))
            util_dynarray_append(&worklist, phi);
      }
   }

   while (util_dynarray_num_elements(&worklist, struct agx_apple9_vir_instr *)) {
      struct agx_apple9_vir_instr *phi =
         util_dynarray_pop(&worklist, struct agx_apple9_vir_instr *);
      uint32_t dest = phi->dest;
      if (BITSET_TEST(removed, dest))
         continue;

      uint32_t same = AGX_APPLE9_VREG_INVALID;
      bool all_same = true;
      unsigned ns = agx_apple9_vir_nr_srcs(phi);
      if (ns == 0)
         continue;

      for (unsigned s = 0; s < ns; ++s) {
         uint32_t sv = agx_apple9_vir_src(phi, s);
         if (sv == AGX_APPLE9_VREG_INVALID || sv >= alloc) {
            all_same = false;
            break;
         }
         if (sv == same || sv == dest)
            continue;
         if (same != AGX_APPLE9_VREG_INVALID) {
            all_same = false;
            break;
         }
         same = sv;
      }

      if (!all_same)
         continue;

      BITSET_SET(removed, dest);
      util_dynarray_append(&to_remove, phi);

      if (same == AGX_APPLE9_VREG_INVALID)
         continue;

      /* Forward program->output and program->live_out[] if they name dest. */
      if (program->output == dest)
         program->output = same;
      for (unsigned k = 0; k < program->live_out_count; ++k) {
         if (program->live_out[k] == dest)
            program->live_out[k] = same;
      }

      struct trivial_phi_use *last = NULL;
      for (struct trivial_phi_use *u = uses[dest]; u; u = u->next) {
         if (u->I->op == AGX_APPLE9_VIR_PHI && u->I->dest < alloc &&
             BITSET_TEST(removed, u->I->dest)) {
            last = u;
            continue;
         }
         assert(agx_apple9_vir_src(u->I, u->s) == dest);
         *agx_apple9_vir_src_ptr(u->I, u->s) = same;
         last = u;

         if (u->I->op == AGX_APPLE9_VIR_PHI && u->I->dest < alloc &&
             BITSET_TEST(candidate, u->I->dest) &&
             !BITSET_TEST(removed, u->I->dest))
            util_dynarray_append(&worklist, u->I);
      }

      if (last) {
         last->next = uses[same];
         uses[same] = uses[dest];
         uses[dest] = NULL;
      }
   }

   unsigned remove_count =
      util_dynarray_num_elements(&to_remove, struct agx_apple9_vir_instr *);
   if (remove_count)
      agx_apple9_vir_remove_many(program, to_remove.data, remove_count);

   util_dynarray_fini(&to_remove);
   util_dynarray_fini(&worklist);
   free(removed);
   free(candidate);
   free(uses);
   free(pool);
}

/* -------------------------------------------------------------------------
 * Compact / reindex SSA values while preserving metadata and contiguous tuples
 * ------------------------------------------------------------------------- */

static bool
apple9_reindex_ssa(struct agx_apple9_vir_program *program,
                   struct apple9_ra_spill_state *state,
                   struct hash_table_u64 *homes)
{
   unsigned old_count = program->value_count;
   if (!old_count) {
      agx_apple9_vir_reindex(program);
      return true;
   }

   uint32_t *remap = malloc(old_count * sizeof(uint32_t));
   BITSET_WORD *has_def = BITSET_CALLOC(old_count);
   BITSET_WORD *referenced = BITSET_CALLOC(old_count);
   if (!remap || !has_def || !referenced) {
      free(remap);
      free(has_def);
      free(referenced);
      return false;
   }
   for (unsigned i = 0; i < old_count; ++i)
      remap[i] = AGX_APPLE9_VREG_INVALID;

   /* Mark all defined and referenced values across all blocks. */
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
            for (unsigned c = 0; c < I->dest_components && I->dest + c < old_count; ++c)
               BITSET_SET(has_def, I->dest + c);
         }
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < old_count)
               BITSET_SET(referenced, v);
         }
      }
   }
   if (program->output < old_count)
      BITSET_SET(referenced, program->output);
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      if (program->live_out[k] < old_count)
         BITSET_SET(referenced, program->live_out[k]);
   }

   unsigned new_count = 0;

   /* Preserve undefined entry inputs (vir_input / fixed_phys) and any undef
    * values still referenced by sources. */
   for (unsigned v = 0; v < old_count; ++v) {
      if (BITSET_TEST(has_def, v))
         continue;
      bool is_fixed = program->fixed_phys &&
                      program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID;
      if (is_fixed || BITSET_TEST(referenced, v))
         remap[v] = new_count++;
   }

   /* Assign dense contiguous ranges for all instruction destinations. */
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components == 0 ||
             I->dest >= old_count)
            continue;
         uint32_t base = new_count;
         for (unsigned c = 0; c < I->dest_components && I->dest + c < old_count; ++c)
            remap[I->dest + c] = base + c;
         new_count += I->dest_components;
      }
   }

   /* Allocate replacement metadata before mutating any IR IDs so OOM leaves
    * the program untouched. */
   unsigned alloc_count = MAX2(new_count, 1u);
   uint8_t *new_fixed = malloc(alloc_count);
   uint8_t *new_max = malloc(alloc_count);
   if (!new_fixed || !new_max) {
      free(new_fixed);
      free(new_max);
      free(remap);
      free(has_def);
      free(referenced);
      return false;
   }
   memset(new_fixed, AGX_APPLE9_PHYS_INVALID, alloc_count);
   memset(new_max, AGX_APPLE9_PHYS_INVALID, alloc_count);

   bool *new_pub = NULL;
   if (program->publication) {
      new_pub = calloc(alloc_count, sizeof(bool));
      if (!new_pub) {
         free(new_fixed);
         free(new_max);
         free(remap);
         free(has_def);
         free(referenced);
         return false;
      }
   }

   uint8_t *new_phys = NULL;
   if (program->phys) {
      new_phys = malloc(alloc_count);
      if (!new_phys) {
         free(new_pub);
         free(new_fixed);
         free(new_max);
         free(remap);
         free(has_def);
         free(referenced);
         return false;
      }
      memset(new_phys, AGX_APPLE9_PHYS_INVALID, alloc_count);
   }

   BITSET_WORD *new_mem = NULL;
   uint8_t *new_width = NULL;
   uint32_t *new_homes = NULL;
   if (state) {
      new_mem = calloc(BITSET_WORDS(alloc_count), sizeof(BITSET_WORD));
      new_width = malloc(alloc_count * sizeof(uint8_t));
      if (homes)
         new_homes = calloc(alloc_count, sizeof(uint32_t));
      if (!new_mem || !new_width || (homes && !new_homes)) {
         free(new_mem);
         free(new_width);
         free(new_homes);
         free(new_phys);
         free(new_pub);
         free(new_fixed);
         free(new_max);
         free(remap);
         free(has_def);
         free(referenced);
         return false;
      }
      memset(new_width, 1, alloc_count * sizeof(uint8_t));
   }

   /* Rewrite instruction operands and program outputs. */
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(b); I;
           I = agx_apple9_instr_next(I)) {
         if (I->dest < old_count && I->dest_components > 0)
            I->dest = remap[I->dest];
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t *sp = agx_apple9_vir_src_ptr(I, s);
            if (*sp < old_count) {
               assert(remap[*sp] != AGX_APPLE9_VREG_INVALID);
               *sp = remap[*sp];
            }
         }
      }
   }

   if (program->output < old_count && remap[program->output] != AGX_APPLE9_VREG_INVALID)
      program->output = remap[program->output];
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      if (program->live_out[k] < old_count &&
          remap[program->live_out[k]] != AGX_APPLE9_VREG_INVALID)
         program->live_out[k] = remap[program->live_out[k]];
   }

   for (unsigned v = 0; v < old_count; ++v) {
      uint32_t nv = remap[v];
      if (nv == AGX_APPLE9_VREG_INVALID)
         continue;
      if (program->fixed_phys)
         new_fixed[nv] = program->fixed_phys[v];
      if (program->max_phys)
         new_max[nv] = program->max_phys[v];
      if (new_pub && v < program->publication_capacity)
         new_pub[nv] = program->publication[v];
      if (new_phys && v < program->phys_capacity)
         new_phys[nv] = program->phys[v];
      if (state) {
         if (apple9_ra_value_is_mem(state, v)) {
            BITSET_SET(new_mem, nv);
            if (new_homes) {
               void *ent = _mesa_hash_table_u64_search(homes, v);
               new_homes[nv] = (uint32_t)(uintptr_t)ent;
            }
         }
         new_width[nv] = apple9_ra_value_width(state, v);
      }
   }

   free(program->fixed_phys);
   free(program->max_phys);
   program->fixed_phys = new_fixed;
   program->max_phys = new_max;
   program->value_capacity = alloc_count;
   if (new_pub) {
      free(program->publication);
      program->publication = new_pub;
      program->publication_capacity = alloc_count;
   }
   if (new_phys) {
      free(program->phys);
      program->phys = new_phys;
      program->phys_capacity = alloc_count;
   }
   program->value_count = new_count;

   if (state) {
      free(state->is_mem);
      free(state->value_width);
      free(state->spill_home);
      state->is_mem = new_mem;
      state->value_width = new_width;
      state->is_mem_capacity = alloc_count;
      state->spill_home = new_homes;
      state->spill_home_count = new_homes ? new_count : 0;
   }

   free(remap);
   free(has_def);
   free(referenced);
   agx_apple9_vir_reindex(program);
   return true;
}

/* -------------------------------------------------------------------------
 * Entry point: apple9_ra_repair_ssa()
 * ------------------------------------------------------------------------- */

bool
apple9_ra_repair_ssa(struct agx_apple9_vir_program *program,
                     struct apple9_ra_spill_state *state,
                     bool track_spill_homes)
{
   if (!program)
      return false;
   if (!program->blocks || program->value_count == 0) {
      agx_apple9_vir_reindex(program);
      return true;
   }

   if (program->num_cfg_blocks == 0 && !agx_apple9_vir_build_logical_cfg(program))
      return false;

   unsigned num_blocks = program->num_cfg_blocks;
   if (num_blocks == 0) {
      agx_apple9_vir_reindex(program);
      return true;
   }

   unsigned n = program->value_count;
   if (state && !apple9_ra_spill_state_ensure(state, n))
      return false;

   void *memctx = ralloc_context(NULL);
   struct agx_apple9_block **cfg_blocks =
      rzalloc_array(memctx, struct agx_apple9_block *, num_blocks);
   struct repair_block *blocks =
      rzalloc_array(memctx, struct repair_block, num_blocks);
   if (!cfg_blocks || !blocks) {
      ralloc_free(memctx);
      return false;
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->cfg_active && b->cfg_index < num_blocks)
         cfg_blocks[b->cfg_index] = b;
   }

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      blocks[bi].defs = _mesa_hash_table_u64_create(memctx);
      if (!blocks[bi].defs) {
         ralloc_free(memctx);
         return false;
      }
      util_dynarray_init(&blocks[bi].incomplete_phis, memctx);
   }

   struct hash_table_u64 *homes =
      (track_spill_homes && state) ? _mesa_hash_table_u64_create(memctx) : NULL;
   struct hash_table *created_phis = _mesa_pointer_hash_table_create(memctx);

   /* Detect direct sub-component uses of non-SPLIT multi-component defs. */
   BITSET_WORD *direct_lanes =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   uint32_t *def_base = malloc(MAX2(n, 1u) * sizeof(uint32_t));
   if (!created_phis || !direct_lanes || !def_base ||
       (track_spill_homes && state && !homes)) {
      free(def_base);
      ralloc_free(memctx);
      return false;
   }
   for (unsigned i = 0; i < MAX2(n, 1u); ++i)
      def_base[i] = AGX_APPLE9_VREG_INVALID;

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_SPLIT || I->dest == AGX_APPLE9_VREG_INVALID ||
             I->dest_components <= 1 || I->dest >= n)
            continue;
         for (unsigned c = 0; c < I->dest_components && I->dest + c < n; ++c)
            def_base[I->dest + c] = I->dest;
      }
   }
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         unsigned ns = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < ns; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < n && def_base[v] != AGX_APPLE9_VREG_INVALID && v != def_base[v])
               BITSET_SET(direct_lanes, def_base[v]);
         }
      }
   }
   for (unsigned k = 0; k < program->live_out_count; ++k) {
      uint32_t v = program->live_out[k];
      if (v < n && def_base[v] != AGX_APPLE9_VREG_INVALID && v != def_base[v])
         BITSET_SET(direct_lanes, def_base[v]);
   }
   if (program->output < n && def_base[program->output] != AGX_APPLE9_VREG_INVALID &&
       program->output != def_base[program->output])
      BITSET_SET(direct_lanes, def_base[program->output]);
   free(def_base);

   /* Identify variables defined more than once (and entry vir_inputs that get
    * reloaded later). */
   BITSET_WORD *defined = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   BITSET_WORD *redefined = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
   BITSET_WORD *predef_inputs =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));

   /* Entry vir_inputs have fixed_phys != INVALID and are not defined by
    * VIR_ENTRY_REG. Mark them pre-defined so any subsequent reload is detected
    * as a redefinition. */
   if (program->fixed_phys) {
      BITSET_WORD *entry_reg_defs =
         rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(n, 1u)));
      for (unsigned bi = 0; bi < num_blocks; ++bi) {
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
              I; I = agx_apple9_instr_next(I)) {
            if (I->op == AGX_APPLE9_VIR_ENTRY_REG && I->dest < n)
               BITSET_SET(entry_reg_defs, I->dest);
         }
      }
      for (unsigned v = 0; v < n; ++v) {
         if (program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID &&
             !BITSET_TEST(entry_reg_defs, v)) {
            BITSET_SET(predef_inputs, v);
            BITSET_SET(defined, v);
         }
      }
   }

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT ||
             I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components == 0 ||
             I->dest >= n)
            continue;

         bool multi_lane =
            I->op == AGX_APPLE9_VIR_SPLIT ||
            (I->dest_components > 1 && BITSET_TEST(direct_lanes, I->dest));
         if (multi_lane) {
            for (unsigned c = 0; c < I->dest_components; ++c) {
               if (I->op == AGX_APPLE9_VIR_SPLIT && (I->dead_dest_mask & (1u << c)))
                  continue;
               uint32_t v = I->dest + c;
               if (v >= n)
                  continue;
               if (BITSET_TEST(defined, v))
                  BITSET_SET(redefined, v);
               BITSET_SET(defined, v);
            }
         } else {
            uint32_t v = I->dest;
            if (BITSET_TEST(defined, v))
               BITSET_SET(redefined, v);
            BITSET_SET(defined, v);
         }
      }
   }

   /* If any live lane of a multi-lane instruction (e.g. VIR_SPLIT) is
    * redefined, mark all of its live lanes redefined so the instruction can
    * receive a single fresh contiguous dest base. */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
           I; I = agx_apple9_instr_next(I)) {
         if (I->dest == AGX_APPLE9_VREG_INVALID || I->dest_components <= 1 ||
             I->dest >= n)
            continue;
         bool multi_lane =
            I->op == AGX_APPLE9_VIR_SPLIT || BITSET_TEST(direct_lanes, I->dest);
         if (!multi_lane)
            continue;

         bool any_redef = false;
         for (unsigned c = 0; c < I->dest_components; ++c) {
            if (I->op == AGX_APPLE9_VIR_SPLIT && (I->dead_dest_mask & (1u << c)))
               continue;
            if (I->dest + c < n && BITSET_TEST(redefined, I->dest + c)) {
               any_redef = true;
               break;
            }
         }
         if (any_redef) {
            for (unsigned c = 0; c < I->dest_components; ++c) {
               if (I->op == AGX_APPLE9_VIR_SPLIT && (I->dead_dest_mask & (1u << c)))
                  continue;
               if (I->dest + c < n)
                  BITSET_SET(redefined, I->dest + c);
            }
         }
      }
   }

   struct repair_ctx rctx = {
      .program = program,
      .state = state,
      .cfg_blocks = cfg_blocks,
      .num_blocks = num_blocks,
      .n = n,
      .blocks = blocks,
      .homes = homes,
      .created_phis = created_phis,
      .failed = false,
   };

   /* Seed entry block with any redefined entry vir_inputs. */
   {
      unsigned v;
      BITSET_FOREACH_SET(v, predef_inputs, n) {
         if (BITSET_TEST(redefined, v))
            record_write(&rctx, cfg_blocks[0], v, v);
      }
   }

   /* Forward pass over logical CFG blocks (source/dominance order). */
   for (unsigned bi = 0; bi < num_blocks && !rctx.failed; ++bi) {
      struct agx_apple9_block *block = cfg_blocks[bi];
      struct agx_apple9_vir_instr *prev_ins = NULL;

      for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(block);
           I && !rctx.failed; I = agx_apple9_instr_next(I)) {
         if (I->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
            if (prev_ins && prev_ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                prev_ins->dest != AGX_APPLE9_VREG_INVALID)
               I->src[0] = prev_ins->dest;
            prev_ins = I;
            continue;
         }

         if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v < n && BITSET_TEST(redefined, v))
                  program->live_out[k] = resolve_read(&rctx, block, v);
            }
            prev_ins = I;
            continue;
         }

         if (I->op != AGX_APPLE9_VIR_PHI) {
            unsigned ns = agx_apple9_vir_nr_srcs(I);
            for (unsigned s = 0; s < ns; ++s) {
               uint32_t *sp = agx_apple9_vir_src_ptr(I, s);
               if (*sp < n && BITSET_TEST(redefined, *sp))
                  *sp = resolve_read(&rctx, block, *sp);
            }
         }

         /* Repair destinations (skip phis newly created by repair). */
         if (I->dest != AGX_APPLE9_VREG_INVALID && I->dest_components > 0) {
            if (I->dest >= n) {
               prev_ins = I;
               continue;
            }

            bool multi_lane =
               I->op == AGX_APPLE9_VIR_SPLIT ||
               (I->dest_components > 1 && BITSET_TEST(direct_lanes, I->dest));

            if (multi_lane) {
               bool any_redef = false;
               for (unsigned c = 0; c < I->dest_components; ++c) {
                  if (I->op == AGX_APPLE9_VIR_SPLIT &&
                      (I->dead_dest_mask & (1u << c)))
                     continue;
                  if (I->dest + c < n && BITSET_TEST(redefined, I->dest + c)) {
                     any_redef = true;
                     break;
                  }
               }
               if (any_redef) {
                  uint32_t old_base = I->dest;
                  uint32_t new_base =
                     alloc_ssa_like(&rctx, old_base, I->dest_components);
                  if (rctx.failed)
                     break;
                  I->dest = new_base;
                  for (unsigned c = 0; c < I->dest_components; ++c) {
                     if (I->op == AGX_APPLE9_VIR_SPLIT &&
                         (I->dead_dest_mask & (1u << c)))
                        continue;
                     if (state)
                        state->value_width[new_base + c] = 1;
                     record_write(&rctx, block, old_base + c, new_base + c);
                  }
               }
            } else {
               uint32_t handle = I->dest;
               if (!BITSET_TEST(redefined, handle)) {
                  if (homes && apple9_ra_value_is_mem(state, handle)) {
                     _mesa_hash_table_u64_insert(homes, handle,
                                                 (void *)(uintptr_t)(handle + 1u));
                  }
               } else {
                  unsigned width = I->dest_components;
                  uint32_t fresh = alloc_ssa_like(&rctx, handle, width);
                  if (rctx.failed)
                     break;
                  I->dest = fresh;
                  record_write(&rctx, block, handle, fresh);
               }
            }
         }

         prev_ins = I;
      }

      seal_loop_headers(&rctx, block);
   }

   /* Safety net: seal any remaining unsealed loop headers. */
   for (unsigned bi = 0; bi < num_blocks && !rctx.failed; ++bi) {
      if (cfg_blocks[bi]->loop_header && !blocks[bi].sealed)
         seal_block(&rctx, cfg_blocks[bi]);
   }

   /* Update pre-existing VIR_PHI sources (which read in their predecessors). */
   for (unsigned bi = 0; bi < num_blocks && !rctx.failed; ++bi) {
      struct agx_apple9_block *block = cfg_blocks[bi];
      agx_apple9_foreach_phi(block, phi) {
         phi->kill_mask = 0;
         if (_mesa_hash_table_search(created_phis, phi))
            continue;

         unsigned ns = agx_apple9_vir_nr_srcs(phi);
         for (unsigned s = 0; s < ns && s < block->num_cfg_preds; ++s) {
            uint32_t *sp = agx_apple9_vir_src_ptr(phi, s);
            if (*sp == AGX_APPLE9_VREG_INVALID || *sp >= n)
               continue;
            if (!BITSET_TEST(redefined, *sp))
               continue;
            *sp = resolve_read(&rctx, block->cfg_preds[s], *sp);
         }
      }
   }

   /* Update program->output (and live_out[] if no VIR_ABI_RETURN) at exit block. */
   if (!rctx.failed) {
      struct agx_apple9_block *exit_blk = NULL;
      for (int bi = (int)num_blocks - 1; bi >= 0; --bi) {
         if (!cfg_blocks[bi]->cfg_succs[0] && !cfg_blocks[bi]->cfg_succs[1]) {
            exit_blk = cfg_blocks[bi];
            break;
         }
      }
      if (!exit_blk)
         exit_blk = cfg_blocks[num_blocks - 1];

      if (program->output < n && BITSET_TEST(redefined, program->output))
         program->output = resolve_read(&rctx, exit_blk, program->output);

      bool has_abi_ret = false;
      for (unsigned bi = 0; bi < num_blocks && !has_abi_ret; ++bi) {
         for (struct agx_apple9_vir_instr *I = agx_apple9_block_first_instr(cfg_blocks[bi]);
              I; I = agx_apple9_instr_next(I)) {
            if (I->op == AGX_APPLE9_VIR_ABI_RETURN) {
               has_abi_ret = true;
               break;
            }
         }
      }
      if (!has_abi_ret) {
         for (unsigned k = 0; k < program->live_out_count; ++k) {
            uint32_t v = program->live_out[k];
            if (v < n && BITSET_TEST(redefined, v))
               program->live_out[k] = resolve_read(&rctx, exit_blk, v);
         }
      }
   }

   if (rctx.failed) {
      ralloc_free(memctx);
      return false;
   }

   apple9_opt_trivial_phi(program, cfg_blocks, num_blocks);
   bool ok = apple9_reindex_ssa(program, state, homes);
   ralloc_free(memctx);
   return ok;
}
