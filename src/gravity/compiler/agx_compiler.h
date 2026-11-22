/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_minifloat.h"
#include "compiler/nir/nir.h"
#include "util/bitset.h"
#include "util/half_float.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_worklist.h"
#include "agx_compile.h"
#include "agx_opcodes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* r0-r127 inclusive, as pairs of 16-bits, gives 256 registers */
#define AGX_NUM_REGS (256)

/* u0-u255 inclusive, as pairs of 16-bits */
#define AGX_NUM_UNIFORMS (512)

/* Model up to 32 KiB of spills, in 16-bit slots. This also covers the full
 * Apple9 private frame rather than imposing a smaller allocator-only limit. */
#define AGX_NUM_MODELED_REGS_LOG2 (14)
#define AGX_NUM_MODELED_REGS      (1 << AGX_NUM_MODELED_REGS_LOG2)

/* Limit on number of sources for non-phi instructions */
#define AGX_MAX_NORMAL_SOURCES (16)

enum agx_index_type {
   AGX_INDEX_NULL = 0,
   AGX_INDEX_NORMAL = 1,
   AGX_INDEX_IMMEDIATE = 2,
   AGX_INDEX_UNIFORM = 3,
   AGX_INDEX_REGISTER = 4,
   AGX_INDEX_UNDEF = 5,
};

enum agx_size { AGX_SIZE_16 = 0, AGX_SIZE_32 = 1, AGX_SIZE_64 = 2 };

static inline unsigned
agx_size_align_16(enum agx_size size)
{
   switch (size) {
   case AGX_SIZE_16:
      return 1;
   case AGX_SIZE_32:
      return 2;
   case AGX_SIZE_64:
      return 4;
   }

   UNREACHABLE("Invalid size");
}

/* Keep synced with hash_index */
typedef struct {
   /* Sufficient for as many SSA values, immediates, and uniforms as we need. */
   uint32_t value;

   /* Indicates that this source kills the referenced value (because it is the
    * last use in a block and the source is not live after the block). Set by
    * liveness analysis.
    */
   bool kill : 1;

   /* Cache hints */
   bool cache   : 1;
   bool discard : 1;

   /* src - float modifiers */
   bool abs : 1;
   bool neg : 1;

   /* Register class */
   bool memory : 1;

   unsigned channels_m1     : 3;
   enum agx_size size       : 2;
   enum agx_index_type type : 3;

   /* If has_reg is set (during register allocation), the register assigned to
    * this SSA value This is used with NORMAL. Contrast REGISTER which uses
    * value instead.
    *
    * TODO: Unify.
    */
   unsigned reg : AGX_NUM_MODELED_REGS_LOG2;
   bool has_reg     : 1;
   unsigned padding : 17 - AGX_NUM_MODELED_REGS_LOG2;
} agx_index;
static_assert(sizeof(agx_index) == 8, "packed");

static inline unsigned
agx_channels(agx_index idx)
{
   return idx.channels_m1 + 1;
}

static inline unsigned
agx_index_size_16(agx_index idx)
{
   return agx_size_align_16(idx.size) * agx_channels(idx);
}

static inline agx_index
agx_get_vec_index(unsigned value, enum agx_size size, unsigned channels)
{
   return (agx_index){
      .value = value,
      .channels_m1 = channels - 1,
      .size = size,
      .type = AGX_INDEX_NORMAL,
   };
}

static inline agx_index
agx_get_index(unsigned value, enum agx_size size)
{
   return agx_get_vec_index(value, size, 1);
}

static inline agx_index
agx_immediate(uint32_t imm)
{
   assert(imm < (1 << 16) && "overflowed immediate");

   return (agx_index){
      .value = imm,
      .size = AGX_SIZE_16,
      .type = AGX_INDEX_IMMEDIATE,
   };
}

/* in half-words, specify r0h as 1, r1 as 2... */
static inline agx_index
agx_register(uint32_t imm, enum agx_size size)
{
   assert(imm < AGX_NUM_REGS);

   return (agx_index){
      .value = imm,
      .size = size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_memory_register(uint32_t imm, enum agx_size size)
{
   return (agx_index){
      .value = imm,
      .memory = true,
      .size = size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_register_like(uint32_t imm, agx_index like)
{
   return (agx_index){
      .value = imm,
      .memory = like.memory,
      .channels_m1 = like.channels_m1,
      .size = like.size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_as_register(agx_index x)
{
   assert(x.has_reg);
   return agx_register_like(x.reg, x);
}

static inline agx_index
agx_undef(enum agx_size size)
{
   return (agx_index){
      .size = size,
      .type = AGX_INDEX_UNDEF,
   };
}

/* Also in half-words */
static inline agx_index
agx_uniform(uint32_t imm, enum agx_size size)
{
   assert(imm < AGX_NUM_UNIFORMS);

   return (agx_index){
      .value = imm,
      .size = size,
      .type = AGX_INDEX_UNIFORM,
   };
}

static inline agx_index
agx_null()
{
   return (agx_index){.type = AGX_INDEX_NULL};
}

/* Replaces an index, preserving any modifiers */

static inline agx_index
agx_replace_index(agx_index old, agx_index replacement)
{
   replacement.abs = old.abs;
   replacement.neg = old.neg;
   return replacement;
}

static inline bool
agx_is_null(agx_index idx)
{
   return idx.type == AGX_INDEX_NULL;
}

/* Compares equivalence as references */

static inline bool
agx_is_equiv(agx_index left, agx_index right)
{
   return (left.type == right.type) && (left.value == right.value);
}

enum ra_class {
   /* General purpose register */
   RA_GPR,

   /* Memory, used to assign stack slots */
   RA_MEM,

   /* Keep last */
   RA_CLASSES,
};

static inline enum ra_class
ra_class_for_index(agx_index idx)
{
   return idx.memory ? RA_MEM : RA_GPR;
}

/* Instruction-local register constraints, in 16-bit units. Bounds apply to
 * the first register of an operand; vectors must also fit in the register
 * file. A fixed source inside the reserved region is a short pre-instruction
 * copy, not a restriction on the source's whole live range. */
struct agx_reg_constraint {
   uint16_t min, max, align;
   bool clobber;

   /* Destination only: the instruction reads its sources before writing, so
    * the destination may occupy registers of sources that die here. */
   bool reuse_killed;
};

typedef struct {
   /* Must be first */
   struct list_head link;

   /* The sources list. */
   agx_index *src;

   /* Data flow */
   agx_index *dest;

   /* Optional constraints for destinations followed by sources. */
   struct agx_reg_constraint *reg_constraints;

   enum agx_opcode op;

   uint8_t nr_dests;
   uint8_t nr_srcs;

   union {
      uint64_t imm;

      /* Target-selected instruction carried through shared allocation. */
      struct agx_apple9_vir_instr *apple9;
   };

   /* Format for stack_load / stack_store */
   enum agx_format format;

   /* Temporary marker for newly-created phis during SSA repair */
   bool shadow : 1;

   /* Shift for iadd (unit tests) */
   unsigned shift : 4;

   /* Channel mask for stack_load / stack_store */
   unsigned mask : 4;
} agx_instr;

static inline void
agx_replace_src(agx_instr *I, unsigned src_index, agx_index replacement)
{
   I->src[src_index] = agx_replace_index(I->src[src_index], replacement);
}

struct agx_block;

typedef struct agx_block {
   /* Link to next block. Must be first */
   struct list_head link;

   /* List of instructions emitted for the current block */
   struct list_head instructions;

   /* Index of the block in source order */
   unsigned index;

   /* Control flow graph */
   struct agx_block *successors[2];
   struct util_dynarray predecessors;
   bool unconditional_jumps;

   /* Liveness analysis results */
   struct u_sparse_bitset live_in;
   struct u_sparse_bitset live_out;

   /* Live-out SSA names and their assigned registers, sorted by SSA name.
    * A non-NULL array also marks blocks already visited by register assignment.
    */
   struct {
      uint32_t ssa;
      uint32_t reg;
   } *ssa_to_reg_out;
   unsigned nr_ssa_to_reg_out;

   /* Is this block a loop header? If not, all of its predecessors precede it in
    * source order.
    */
   bool loop_header;
} agx_block;

/* Target allocation limits, in the IR's 16-bit register units. Zero selects
 * the established Apple7/8 defaults. Instruction selection and spill lowering
 * remain target-specific; SSA spilling and register assignment are shared. */
struct agx_context;

struct agx_ra_target {
   unsigned max_registers;
   unsigned reserved_registers;
   unsigned spill_reserved_registers;
   unsigned spill_copy_register;
   unsigned spill_swap_registers[2];
   bool late_kill_sources;
   bool preserve_source_kills;
   bool defer_spill_lowering;
   /* An occupancy table from another target is not a rematerialization cost
    * model. Targets without measured breakpoints can retain real spilling
    * while declining this optional code-size/occupancy tradeoff. */
   bool disable_occupancy_rematerialization;
   /* At a join, keep a phi in registers when any predecessor holds its
    * source in a register; the others reload it on their edge. Otherwise a
    * single evicted source spills the whole phi. */
   bool spill_partial_phi_sources;
   bool cost_register_constraints;
   bool disable_uniform_rematerialization;
   /* Number of half-registers directly addressable by MOV_IMM, or zero when
    * its destination range does not add a materialization cost. */
   uint16_t mov_imm_registers;
   /* Optional candidate-search bound, after spilling/SSA repair and before
    * register assignment. False rejects this candidate without allocating it.
    * A caller must retain a completed allocation when using this callback. */
   bool (*within_spill_budget)(const struct agx_context *ctx, void *data);
   void *spill_budget_data;
};

typedef struct agx_context {
   struct agx_ra_target ra_target;
   /* Static diagnostic set when constrained register allocation fails. */
   const char *ra_error;
   /* Opaque target instructions may refine the shared scheduling classes. */
   enum agx_schedule_class (*schedule_class)(const agx_instr *instruction);
   /* Optional target implicit-resource graph, replacing scheduling classes.
    * Readers may overlap; a writer orders all preceding readers and writers.
    * Resource bit assignments belong to the target. */
   void (*schedule_resources)(const agx_instr *instruction,
                              uint32_t *reads, uint32_t *writes);
   /* Set when uniform rematerialization marked at least one value. */
   bool uniform_remat_used;
   /* Relative dependency separation in issue positions, not hardware cycles. */
   unsigned (*schedule_delay)(const agx_instr *instruction);
   /* Half-registers a latency schedule may add to a block's peak pressure. */
   unsigned schedule_pressure_slack;
   /* Largest peak-pressure increase of a latency reorder that was rejected
    * for exceeding the slack, so callers can tell whether more slack changes
    * the schedule. */
   unsigned schedule_rejected_increase;
   nir_shader *nir;
   mesa_shader_stage stage;
   unsigned scratch_size_B;

   struct list_head blocks; /* list of agx_block */
   struct agx_shader_key *key;

   /* Maximum block index */
   unsigned num_blocks;

   /* For creating temporaries */
   unsigned alloc;

   /* Opt-in spill coalescing prototype: original repair variable + 1 for each
    * memory SSA definition, zero for other values. Reindexed with SSA. */
   unsigned *spill_home;
   unsigned spill_home_count;

   /* Beginning of our stack allocation used for spilling, below that is
    * NIR-level scratch.
    */
   unsigned spill_base_B;

   /* Whether spill parallel-copy temporaries have been reserved */
   bool has_spill_pcopy_reserved;

   unsigned max_reg;
} agx_context;

static inline void
agx_remove_instruction(agx_instr *ins)
{
   list_del(&ins->link);
}

static inline agx_index
agx_vec_temp(agx_context *ctx, enum agx_size size, unsigned channels)
{
   return agx_get_vec_index(ctx->alloc++, size, channels);
}

static inline agx_index
agx_temp(agx_context *ctx, enum agx_size size)
{
   return agx_get_index(ctx->alloc++, size);
}

static inline agx_index
agx_temp_like(agx_context *ctx, agx_index idx)
{
   idx.value = ctx->alloc++;
   return idx;
}

static enum agx_size
agx_size_for_bits(unsigned bits)
{
   switch (bits) {
   case 1:
   case 8:
   case 16:
      return AGX_SIZE_16;
   case 32:
      return AGX_SIZE_32;
   case 64:
      return AGX_SIZE_64;
   default:
      UNREACHABLE("Invalid bitsize");
   }
}

static inline unsigned
agx_num_predecessors(agx_block *block)
{
   return util_dynarray_num_elements(&block->predecessors, agx_block *);
}

static inline unsigned
agx_num_successors(agx_block *block)
{
   STATIC_ASSERT(ARRAY_SIZE(block->successors) == 2);
   return (block->successors[0] ? 1 : 0) + (block->successors[1] ? 1 : 0);
}

static inline agx_block *
agx_start_block(agx_context *ctx)
{
   agx_block *first = list_first_entry(&ctx->blocks, agx_block, link);
   assert(agx_num_predecessors(first) == 0);
   return first;
}

void agx_block_add_successor(agx_block *block, agx_block *successor);

/* Iterators for AGX IR */

#define agx_foreach_block(ctx, v)                                              \
   list_for_each_entry(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_safe(ctx, v)                                         \
   list_for_each_entry_safe(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_rev(ctx, v)                                          \
   list_for_each_entry_rev(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_from(ctx, from, v)                                   \
   list_for_each_entry_from(agx_block, v, from, &ctx->blocks, link)

#define agx_foreach_block_from_rev(ctx, from, v)                               \
   list_for_each_entry_from_rev(agx_block, v, from, &ctx->blocks, link)

#define agx_foreach_instr_in_block(block, v)                                   \
   list_for_each_entry(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_rev(block, v)                               \
   list_for_each_entry_rev(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_safe(block, v)                              \
   list_for_each_entry_safe(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_safe_rev(block, v)                          \
   list_for_each_entry_safe_rev(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_from(block, v, from)                        \
   list_for_each_entry_from(agx_instr, v, from, &(block)->instructions, link)

#define agx_foreach_instr_in_block_from_rev(block, v, from)                    \
   list_for_each_entry_from_rev(agx_instr, v, from, &(block)->instructions,    \
                                link)

#define agx_foreach_instr_global(ctx, v)                                       \
   agx_foreach_block(ctx, v_block)                                             \
      agx_foreach_instr_in_block(v_block, v)

#define agx_foreach_instr_global_rev(ctx, v)                                   \
   agx_foreach_block_rev(ctx, v_block)                                         \
      agx_foreach_instr_in_block_rev(v_block, v)

#define agx_foreach_instr_global_safe(ctx, v)                                  \
   agx_foreach_block(ctx, v_block)                                             \
      agx_foreach_instr_in_block_safe(v_block, v)

#define agx_foreach_instr_global_safe_rev(ctx, v)                              \
   agx_foreach_block_rev(ctx, v_block)                                         \
      agx_foreach_instr_in_block_safe_rev(v_block, v)

/* Based on set_foreach, expanded with automatic type casts */

#define agx_foreach_successor(blk, v)                                          \
   agx_block *v;                                                               \
   agx_block **_v;                                                             \
   for (_v = (agx_block **)&blk->successors[0], v = *_v;                       \
        v != NULL && _v < (agx_block **)&blk->successors[2]; _v++, v = *_v)

#define agx_foreach_predecessor(blk, v)                                        \
   util_dynarray_foreach(&blk->predecessors, agx_block *, v)

#define agx_foreach_src(ins, v) for (unsigned v = 0; v < ins->nr_srcs; ++v)

#define agx_foreach_src_rev(ins, v)                                            \
   for (signed v = ins->nr_srcs - 1; v >= 0; --v)

#define agx_foreach_dest(ins, v) for (unsigned v = 0; v < ins->nr_dests; ++v)

#define agx_foreach_dest_rev(ins, v)                                           \
   for (signed v = ins->nr_dests - 1; v >= 0; --v)

#define agx_foreach_ssa_src(ins, v)                                            \
   agx_foreach_src(ins, v)                                                     \
      if (ins->src[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_ssa_src_rev(ins, v)                                        \
   agx_foreach_src_rev(ins, v)                                                 \
      if (ins->src[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_reg_src(ins, v)                                            \
   agx_foreach_src(ins, v)                                                     \
      if (ins->src[v].type == AGX_INDEX_REGISTER)

#define agx_foreach_ssa_dest(ins, v)                                           \
   agx_foreach_dest(ins, v)                                                    \
      if (ins->dest[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_ssa_dest_rev(ins, v)                                       \
   agx_foreach_dest_rev(ins, v)                                                \
      if (ins->dest[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_reg_dest(ins, v)                                           \
   agx_foreach_dest(ins, v)                                                    \
      if (ins->dest[v].type == AGX_INDEX_REGISTER)

/* Phis only come at the start of a block so we stop as soon as we hit a non-phi */
#define agx_foreach_phi_in_block(block, v)                                     \
   agx_foreach_instr_in_block(block, v)                                        \
      if (v->op != AGX_OPCODE_PHI)                                             \
         break;                                                                \
      else

#define agx_foreach_phi_in_block_safe(block, v)                                \
   agx_foreach_instr_in_block_safe(block, v)                                   \
      if (v->op != AGX_OPCODE_PHI)                                             \
         break;                                                                \
      else

/*
 * Find the index of a predecessor, used as the implicit order of phi sources.
 */
static inline unsigned
agx_predecessor_index(agx_block *succ, agx_block *pred)
{
   unsigned index = 0;

   agx_foreach_predecessor(succ, x) {
      if (*x == pred)
         return index;

      index++;
   }

   UNREACHABLE("Invalid predecessor");
}

static inline agx_instr *
agx_prev_op(agx_instr *ins)
{
   return list_last_entry(&(ins->link), agx_instr, link);
}

static inline agx_instr *
agx_first_instr(agx_block *block)
{
   if (list_is_empty(&block->instructions))
      return NULL;
   else
      return list_first_entry(&block->instructions, agx_instr, link);
}

static inline agx_block *
agx_exit_block(agx_context *ctx)
{
   agx_block *last = list_last_entry(&ctx->blocks, agx_block, link);
   assert(!last->successors[0] && !last->successors[1]);
   return last;
}

#define agx_worklist_init(ctx, w)        u_worklist_init(w, ctx->num_blocks, ctx)
#define agx_worklist_push_head(w, block) u_worklist_push_head(w, block, index)
#define agx_worklist_push_tail(w, block) u_worklist_push_tail(w, block, index)
#define agx_worklist_peek_head(w)        u_worklist_peek_head(w, agx_block, index)
#define agx_worklist_pop_head(w)         u_worklist_pop_head(w, agx_block, index)
#define agx_worklist_peek_tail(w)        u_worklist_peek_tail(w, agx_block, index)
#define agx_worklist_pop_tail(w)         u_worklist_pop_tail(w, agx_block, index)

/* Like in NIR, for use with the builder */

enum agx_cursor_option {
   agx_cursor_after_block,
   agx_cursor_before_instr,
   agx_cursor_after_instr
};

typedef struct {
   enum agx_cursor_option option;

   union {
      agx_block *block;
      agx_instr *instr;
   };
} agx_cursor;

static inline agx_cursor
agx_after_block(agx_block *block)
{
   return (agx_cursor){
      .option = agx_cursor_after_block,
      .block = block,
   };
}

static inline agx_cursor
agx_before_instr(agx_instr *instr)
{
   return (agx_cursor){
      .option = agx_cursor_before_instr,
      .instr = instr,
   };
}

static inline agx_cursor
agx_after_instr(agx_instr *instr)
{
   return (agx_cursor){
      .option = agx_cursor_after_instr,
      .instr = instr,
   };
}

static inline agx_cursor
agx_before_nonempty_block(agx_block *block)
{
   agx_instr *I = list_first_entry(&block->instructions, agx_instr, link);
   assert(I != NULL);

   return agx_before_instr(I);
}

static inline agx_cursor
agx_before_block(agx_block *block)
{
   if (list_is_empty(&block->instructions))
      return agx_after_block(block);
   else
      return agx_before_nonempty_block(block);
}

static inline bool
instr_after_logical_end(const agx_instr *I)
{
   return I->op == AGX_OPCODE_APPLE9_CF || I->op == AGX_OPCODE_EXPORT;
}

/*
 * Get a cursor inserting at the logical end of the block. In particular, this
 * is before branches or control flow instructions, which occur after the
 * logical end but before the physical end.
 */
static inline agx_cursor
agx_after_block_logical(agx_block *block)
{
   /* Search for the first instruction that's not past the logical end */
   agx_foreach_instr_in_block_rev(block, I) {
      if (!instr_after_logical_end(I))
         return agx_after_instr(I);
   }

   /* If we got here, the block is either empty or entirely control flow */
   return agx_before_block(block);
}

/* Get a cursor at the start of a function, after any preloads */
static inline agx_cursor
agx_before_function(agx_context *ctx)
{
   agx_block *block = agx_start_block(ctx);

   agx_foreach_instr_in_block(block, I) {
      if (I->op != AGX_OPCODE_PRELOAD)
         return agx_before_instr(I);
   }

   /* The whole block is preloads, so insert at the end */
   return agx_after_block(block);
}

/* IR builder in terms of cursor infrastructure */

typedef struct {
   agx_context *shader;
   agx_cursor cursor;
} agx_builder;

static inline agx_builder
agx_init_builder(agx_context *ctx, agx_cursor cursor)
{
   return (agx_builder){
      .shader = ctx,
      .cursor = cursor,
   };
}

/* Insert an instruction at the cursor and move the cursor */

static inline void
agx_builder_insert(agx_cursor *cursor, agx_instr *I)
{
   switch (cursor->option) {
   case agx_cursor_after_instr:
      list_add(&I->link, &cursor->instr->link);
      cursor->instr = I;
      return;

   case agx_cursor_after_block:
      list_addtail(&I->link, &cursor->block->instructions);
      cursor->option = agx_cursor_after_instr;
      cursor->instr = I;
      return;

   case agx_cursor_before_instr:
      list_addtail(&I->link, &cursor->instr->link);
      cursor->option = agx_cursor_after_instr;
      cursor->instr = I;
      return;
   }

   UNREACHABLE("Invalid cursor option");
}

/* Routines defined for AIR */
void agx_print_index(agx_index index, bool is_float, FILE *fp);
void agx_print_instr(const agx_instr *I, FILE *fp);
void agx_print_block(const agx_block *block, FILE *fp);
void agx_print_shader(const agx_context *ctx, FILE *fp);
void agx_lower_spill(agx_context *ctx);
void agx_dce(agx_context *ctx, bool partial);
void agx_pressure_schedule(agx_context *ctx);
void agx_spill(agx_context *ctx, unsigned k, bool remat_only);
void agx_repair_ssa(agx_context *ctx);

struct agx_spill_interference;
struct agx_spill_interference *agx_spill_coalesce_prepare(agx_context *ctx);
bool agx_spill_coalesce_apply(agx_context *ctx,
                              struct agx_spill_interference *graph,
                              unsigned *half_slots);
void agx_reindex_ssa(agx_context *ctx);
bool agx_ra(agx_context *ctx);

#ifndef NDEBUG
void agx_validate(agx_context *ctx, const char *after_str);
void agx_validate_ra(agx_context *ctx);
#else
static inline void
agx_validate(UNUSED agx_context *ctx, UNUSED const char *after_str)
{
}

static inline void
agx_validate_ra(UNUSED agx_context *ctx)
{
}
#endif

enum agx_size agx_split_width(const agx_instr *I);
bool agx_allows_16bit_immediate(agx_instr *I);

struct agx_copy {
   /* Base register destination of the copy */
   unsigned dest;

   /* Destination is memory */
   bool dest_mem;

   /* Source of the copy */
   agx_index src;

   /* Whether the copy has been handled. Callers must leave to false. */
   bool done;
};

void agx_emit_parallel_copies(agx_builder *b, struct agx_copy *copies,
                              unsigned n);

void agx_compute_liveness(agx_context *ctx);
void agx_liveness_ins_update(struct u_sparse_bitset *live, agx_instr *I);

bool agx_nir_lower_frag_sidefx(nir_shader *s);

extern int agx_compiler_debug;

#ifdef __cplusplus
} /* extern C */
#endif
