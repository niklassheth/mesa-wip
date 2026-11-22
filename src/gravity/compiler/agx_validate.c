/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/compiler.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_opcodes.h"

/* Validatation doesn't make sense in release builds */
#ifndef NDEBUG

#define agx_validate_assert(stmt)                                              \
   if (!(stmt)) {                                                              \
      return false;                                                            \
   }

/*
 * If a block contains phi nodes, they must come at the start of the block. If a
 * block contains control flow, it must come at the beginning/end as applicable.
 * Therefore the form of a valid block is:
 *
 *       Control flow instructions (else)
 *       Phi nodes
 *       General instructions
 *       Control flow instructions (except else)
 *
 * Validate that this form is satisfied.
 */
enum agx_block_state {
   AGX_BLOCK_STATE_CF_ELSE = 0,
   AGX_BLOCK_STATE_PHI = 1,
   AGX_BLOCK_STATE_BODY = 2,
   AGX_BLOCK_STATE_CF = 3
};

static bool
agx_validate_block_form(agx_block *block)
{
   enum agx_block_state state = AGX_BLOCK_STATE_CF_ELSE;

   agx_foreach_instr_in_block(block, I) {
      switch (I->op) {
      case AGX_OPCODE_PRELOAD:
         agx_validate_assert(state == AGX_BLOCK_STATE_CF_ELSE);
         break;

      case AGX_OPCODE_PHI:
         agx_validate_assert(state == AGX_BLOCK_STATE_CF_ELSE ||
                             state == AGX_BLOCK_STATE_PHI);

         state = AGX_BLOCK_STATE_PHI;
         break;

      case AGX_OPCODE_EXPORT:
         agx_validate_assert(agx_num_successors(block) == 0);
         state = AGX_BLOCK_STATE_CF;
         break;

      default:
         if (instr_after_logical_end(I)) {
            state = AGX_BLOCK_STATE_CF;
         } else {
            agx_validate_assert(state != AGX_BLOCK_STATE_CF);
            state = AGX_BLOCK_STATE_BODY;
         }
         break;
      }
   }

   return true;
}

/*
 * Only moves and phis use stack. Phis cannot use moves due to their
 * parallel nature, so we allow phis to take memory, later lowered to moves.
 */
static bool
is_stack_valid(agx_instr *I)
{
   return (I->op == AGX_OPCODE_MOV) || (I->op == AGX_OPCODE_PHI);
}

static bool
agx_validate_sources(agx_instr *I)
{
   agx_foreach_src(I, s) {
      agx_index src = I->src[s];

      if (src.type == AGX_INDEX_IMMEDIATE) {
         agx_validate_assert(!src.kill);
         agx_validate_assert(!src.cache);
         agx_validate_assert(!src.discard);

         bool ldst = agx_allows_16bit_immediate(I);

         /* Immediates are encoded as 8-bit (16-bit for memory load/store). For
          * integers, they extend to 16-bit. For floating point, they are 8-bit
          * minifloats. The 8-bit minifloats are a strict subset of 16-bit
          * standard floats, so we treat them as such in the IR, with an
          * implicit f16->f32 for 32-bit floating point operations.
          */
         agx_validate_assert(src.size == AGX_SIZE_16);
         agx_validate_assert(src.value < (1 << (ldst ? 16 : 8)));
      } else if (I->op == AGX_OPCODE_COLLECT && !agx_is_null(src)) {
         agx_validate_assert(src.size == I->src[0].size);
      } else if (I->op == AGX_OPCODE_PHI) {
         agx_validate_assert(src.size == I->dest[0].size);
         agx_validate_assert(!agx_is_null(src));
      }

      agx_validate_assert(!src.memory || is_stack_valid(I));
   }

   return true;
}

static bool
agx_validate_defs(agx_instr *I, BITSET_WORD *defs)
{
   agx_foreach_ssa_src(I, s) {
      /* Skip phis, they're special in loop headers */
      if (I->op == AGX_OPCODE_PHI)
         break;

      /* Sources must be defined before their use */
      if (!BITSET_TEST(defs, I->src[s].value))
         return false;
   }

   agx_foreach_ssa_dest(I, d) {
      /* Static single assignment */
      if (BITSET_TEST(defs, I->dest[d].value))
         return false;

      BITSET_SET(defs, I->dest[d].value);

      if (I->dest[d].memory && !is_stack_valid(I))
         return false;
   }

   return true;
}

/** Returns number of registers written by an instruction */
static unsigned
agx_write_registers(const agx_instr *I, unsigned d)
{
   unsigned size = agx_size_align_16(I->dest[d].size);

   switch (I->op) {
   case AGX_OPCODE_MOV:
   case AGX_OPCODE_PHI:
   case AGX_OPCODE_APPLE9:
   case AGX_OPCODE_APPLE9_PURE:
      /* Tautological */
      return agx_index_size_16(I->dest[d]);

   case AGX_OPCODE_STACK_LOAD:
      /* Can write 16-bit or 32-bit. Anything logically 64-bit is already
       * expanded to 32-bit in the mask.
       */
      return util_bitcount(I->mask) * MIN2(size, 2);

   case AGX_OPCODE_COLLECT:
      return I->nr_srcs * agx_size_align_16(I->src[0].size);
   default:
      return size;
   }
}

static unsigned
agx_read_registers(const agx_instr *I, unsigned s)
{
   unsigned size = agx_size_align_16(I->src[s].size);

   switch (I->op) {
   case AGX_OPCODE_APPLE9:
   case AGX_OPCODE_APPLE9_PURE:
   case AGX_OPCODE_APPLE9_CF:
      return agx_index_size_16(I->src[s]);
   case AGX_OPCODE_MOV:
   case AGX_OPCODE_EXPORT:
      /* Tautological */
      return agx_index_size_16(I->src[0]);

   case AGX_OPCODE_PHI:
      if (I->src[s].type == AGX_INDEX_IMMEDIATE)
         return size;
      else
         return agx_index_size_16(I->dest[0]);

   case AGX_OPCODE_SPLIT:
      return I->nr_dests * agx_size_align_16(agx_split_width(I));

   case AGX_OPCODE_STACK_STORE:
      /* See agx_write_registers */
      if (s == 0)
         return util_bitcount(I->mask) * MIN2(size, 2);
      else
         return size;

   default:
      return size;
   }
}

/* Type check the dimensionality of sources and destinations. */
static bool
agx_validate_width(agx_context *ctx)
{
   bool succ = true;
   enum agx_size *sizes = calloc(ctx->alloc, sizeof(*sizes));

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_dest(I, d) {
         unsigned exp = agx_write_registers(I, d);
         unsigned act =
            agx_channels(I->dest[d]) * agx_size_align_16(I->dest[d].size);

         if (exp != act) {
            succ = false;
            fprintf(stderr, "destination %u, expected width %u, got width %u\n",
                    d, exp, act);
            agx_print_instr(I, stderr);
            fprintf(stderr, "\n");
         }

         if (I->dest[d].type == AGX_INDEX_NORMAL)
            sizes[I->dest[d].value] = I->dest[d].size;
      }

      agx_foreach_src(I, s) {
         if (I->src[s].type == AGX_INDEX_NULL)
            continue;

         unsigned exp = agx_read_registers(I, s);
         unsigned act =
            agx_channels(I->src[s]) * agx_size_align_16(I->src[s].size);

         if (exp != act) {
            succ = false;
            fprintf(stderr, "source %u, expected width %u, got width %u\n", s,
                    exp, act);
            agx_print_instr(I, stderr);
            fprintf(stderr, "\n");
         }
      }
   }

   /* Check sources after all defs processed for proper backedge handling */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         if (sizes[I->src[s].value] != I->src[s].size) {
            succ = false;
            fprintf(stderr, "source %u, expected el size %u, got el size %u\n",
                    s, agx_size_align_16(sizes[I->src[s].value]),
                    agx_size_align_16(I->src[s].size));
            agx_print_instr(I, stderr);
            fprintf(stderr, "\n");
         }
      }
   }

   free(sizes);
   return succ;
}

static bool
agx_validate_predecessors(agx_block *block)
{
   /* Loop headers (only) have predecessors that are later in source form */
   bool has_later_preds = false;

   agx_foreach_predecessor(block, pred) {
      if ((*pred)->index >= block->index)
         has_later_preds = true;
   }

   if (has_later_preds && !block->loop_header)
      return false;

   /* Successors and predecessors are found together */
   agx_foreach_predecessor(block, pred) {
      bool found = false;

      agx_foreach_successor((*pred), succ) {
         if (succ == block)
            found = true;
      }

      if (!found)
         return false;
   }

   return true;
}

void
agx_validate(agx_context *ctx, const char *after)
{
   bool fail = false;

   if (agx_compiler_debug & AGX_DBG_NOVALIDATE)
      return;

   int last_index = -1;

   agx_foreach_block(ctx, block) {
      if ((int)block->index < last_index) {
         fprintf(stderr, "Out-of-order block index %d vs %d after %s\n",
                 block->index, last_index, after);
         agx_print_block(block, stderr);
         fail = true;
      }

      last_index = block->index;

      if (!agx_validate_block_form(block)) {
         fprintf(stderr, "Invalid block form after %s\n", after);
         agx_print_block(block, stderr);
         fail = true;
      }

      if (!agx_validate_predecessors(block)) {
         fprintf(stderr, "Invalid loop header flag after %s\n", after);
         agx_print_block(block, stderr);
         fail = true;
      }
   }

   {
      BITSET_WORD *defs = BITSET_CALLOC(ctx->alloc);

      agx_foreach_instr_global(ctx, I) {
         if (!agx_validate_defs(I, defs)) {
            fprintf(stderr, "Invalid defs after %s\n", after);
            agx_print_instr(I, stderr);
            fail = true;
         }
      }

      /* agx_validate_defs skips phi sources, so validate them now */
      agx_foreach_block(ctx, block) {
         agx_foreach_phi_in_block(block, phi) {
            agx_foreach_ssa_src(phi, s) {
               if (!BITSET_TEST(defs, phi->src[s].value)) {
                  fprintf(stderr, "Undefined phi source %u after %s\n",
                          phi->src[s].value, after);
                  agx_print_instr(phi, stderr);
                  fail = true;
               }
            }
         }
      }

      free(defs);
   }

   agx_foreach_instr_global(ctx, I) {
      if (!agx_validate_sources(I)) {
         fprintf(stderr, "Invalid sources form after %s\n", after);
         agx_print_instr(I, stderr);
         fail = true;
      }
   }

   if (!agx_validate_width(ctx)) {
      fprintf(stderr, "Invalid vectors after %s\n", after);
      fail = true;
   }

   if (fail) {
      agx_print_shader(ctx, stderr);
      exit(1);
   }
}

#endif /* NDEBUG */
