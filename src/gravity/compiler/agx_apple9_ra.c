/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "gravity/debug.h"
#include "agx_compile.h"
#include "agx_compile_apple9.h"
#include "compiler/nir/nir.h"
#include "agx_apple9_ir.h"
#include "agx_apple9_profile.h"
#include "apple9_ra_assign.h"
#include "apple9_ra_sched.h"
#include "agx_debug.h"
#include "util/hash_table.h"
#include "util/macros.h"

static unsigned
apple9_dependency_delay(const struct agx_apple9_vir_instr *I)
{
   /* Scheduling priorities, not measured T8132 cycle counts. Memory receives
    * the largest separation; transcendental pipelines come next. Candidate
    * selection also accounts for the completed allocation's copies/spills. */
   switch (I->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_SPILL_LOAD:
      return 16;
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_SUBGROUP_REDUCE:
   case AGX_APPLE9_VIR_SUBGROUP_SHUFFLE:
   case AGX_APPLE9_VIR_SUBGROUP_ROTATE:
   case AGX_APPLE9_VIR_MMA:
      return 4;
   default:
      return 1;
   }
}


static bool
apple9_identity(const struct agx_apple9_vir_instr *ins)
{
   return ins->op == AGX_APPLE9_VIR_IOR &&
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED &&
          ins->nr_srcs == 2 && ins->src[0] == ins->src[1];
}


static bool
apple9_after_logical_end(const struct agx_apple9_vir_instr *ins)
{
   if (ins->local_control)
      return false;
   switch (ins->op) {
   case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
   case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
   case AGX_APPLE9_VIR_EXEC_MASK_POP:
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_JMP_EXEC_NONE:
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
   case AGX_APPLE9_VIR_ABI_RETURN:
      return true;
   default:
      return false;
   }
}

/* The first physical block selected from a NIR block. */
static struct agx_apple9_block *
apple9_nir_block(struct hash_table *nir_blocks, nir_block *nir)
{
   struct hash_entry *entry =
      nir ? _mesa_hash_table_search(nir_blocks, nir) : NULL;
   return entry ? entry->data : NULL;
}


#define PUB_BASE AGX_APPLE9_GPR_COUNT

static bool
apple9_emit_physical(struct agx_apple9_vir_program *out,
                     const struct agx_apple9_vir_instr *ins)
{
   if (!agx_apple9_vir_emit_side_effect(out, ins->op, ins->encoding, NULL, 0,
                                        0))
      return false;
   *out->instructions[out->instruction_count - 1] = *ins;
   for (unsigned c = 0; c < ins->dest_components; ++c) {
      if (ins->dest < PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->dest + c);
   }
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (ins->src[s] < PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->src[s]);
   }
   return true;
}

static struct agx_apple9_vir_instr
apple9_physical_instruction(enum agx_apple9_vir_opcode op,
                            enum agx_apple9_encoding encoding, unsigned dest)
{
   return (struct agx_apple9_vir_instr){
      .op = op,
      .encoding = encoding,
      .dest = dest,
      .dest_components = dest != AGX_APPLE9_VREG_INVALID,
      .producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
      .scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
   };
}


static bool
apple9_init_physical(struct agx_apple9_vir_program *out)
{
   agx_apple9_vir_init(out);
   for (unsigned r = 0; r < PUB_BASE + AGX_APPLE9_PUBLICATION_COUNT; ++r) {
      /* Reserve every canonical ID without asking the ordinary-input API to
       * validate a publication index as a GPR. The maps below supply its
       * namespace and physical index. */
      if (agx_apple9_vir_input(out, r < PUB_BASE ? r : 0) != r)
         return false;
      if (r >= PUB_BASE)
         out->fixed_phys[r] = AGX_APPLE9_PHYS_INVALID;
   }
   out->phys = malloc(out->value_count);
   out->publication = calloc(out->value_count, sizeof(bool));
   out->phys_capacity = out->publication_capacity = out->value_count;
   if (!out->phys || !out->publication)
      return false;
   for (unsigned r = 0; r < out->value_count; ++r) {
      out->phys[r] = r < PUB_BASE ? r : r - PUB_BASE;
      out->publication[r] = r >= PUB_BASE;
   }
   return true;
}

/* A slot may be pending on more than one producer when control-flow paths
 * merge. Each path contributes its own producer; waiting covers all of them.
 * A merge beyond the tracked capacity forgets the producers and waits on
 * entry to the block. */
#define APPLE9_PENDING_MERGE 8

/* A slot counts its outstanding operations: native Metal issues up to seven
 * loads and texture reads on one slot and waits for all of them with a
 * single consumer dependency (G13 allows eight). Never exceed seven. */
#define APPLE9_SLOT_SHARE_LIMIT 7

struct apple9_pending {
   unsigned count;
   unsigned latest_issue;
   /* Input position of the first instruction in this block that needs any
    * of the producers, or UINT_MAX when unknown or outside the block. */
   unsigned first_need;
   bool unknown;
   struct agx_apple9_vir_instr producer[APPLE9_PENDING_MERGE];
   /* Whether a device fence has issued since the producer did. */
   bool fenced[APPLE9_PENDING_MERGE];
   /* The if/else arm (apple9_arms node) that issued each producer. */
   unsigned arm[APPLE9_PENDING_MERGE];
};

/* Pending instructions include fixed producers with no GPR result. Each is
 * stored once, under the slot selected by its machine descriptor. */
struct apple9_completion {
   struct apple9_pending pending[7];
};

/* Scalar export producers complete asynchronously in the publication namespace.
 * Their completion tag is independent of the publication's storage index. */
static bool
apple9_scalar_publication(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT;
}

static unsigned
apple9_fixed_producer_slot(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return AGX_APPLE9_SCOREBOARD_SLOT_NONE;

   unsigned slot =
      agx_apple9_encoding_info(ins->encoding)->fixed_producer_slot;
   assert(slot <= AGX_APPLE9_SCOREBOARD_SLOT_6);
   return slot;
}

static bool
apple9_async(const struct agx_apple9_vir_instr *ins)
{
   if (apple9_scalar_publication(ins) || apple9_fixed_producer_slot(ins))
      return true;
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_TILE_LOAD:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_SPILL_STORE:
   case AGX_APPLE9_VIR_IMAGE_STORE:
      return true;
   case AGX_APPLE9_VIR_GET_SR:
      return ins->encoding == AGX_APPLE9_ENC_GET_SR_EXTENDED ||
             ins->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
      return !ins->atomic_discard;
   default:
      return false;
   }
}

static bool
apple9_pending_active(const struct apple9_pending *p)
{
   return p->count || p->unknown;
}

static unsigned
apple9_producer_slots(const struct agx_apple9_vir_instr *producer)
{
   assert(producer->producer_scoreboard_slot >= 1 &&
          producer->producer_scoreboard_slot <= 6);
   unsigned slots = BITFIELD_BIT(producer->producer_scoreboard_slot - 1);
   if (apple9_fixed_producer_slot(producer))
      slots |=
         agx_apple9_encoding_info(producer->encoding)->fixed_producer_alternates;
   return slots;
}

/* Return the canonical pending slots that conflict with a new reservation.
 * U0 reserves both selectors until either completes it; this never aliases
 * unrelated ordinary slot-1 and slot-2 producers. Unknown joins are drained
 * before issuing any instruction in the block. */
static unsigned
apple9_conflicting_slots(const struct apple9_completion *state, unsigned slots)
{
   unsigned conflicts = 0;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      const struct apple9_pending *p = &state->pending[slot];
      bool conflict = p->unknown && (slots & BITFIELD_BIT(slot - 1));
      for (unsigned i = 0; i < p->count; ++i)
         conflict |= (slots & apple9_producer_slots(&p->producer[i])) != 0;
      if (conflict)
         conflicts |= BITFIELD_BIT(slot - 1);
   }
   return conflicts;
}

/* Apply an actual input wait, folded or standalone, to each producer. At a
 * join, slot 1 may contain U0 on one path and an ordinary load on another:
 * waiting on slot 2 completes only the U0 alternative in that list. */
static void
apple9_complete(struct apple9_completion *state, unsigned mask)
{
   for (unsigned slot = 1; slot <= 6; ++slot) {
      struct apple9_pending *p = &state->pending[slot];
      if (mask & BITFIELD_BIT(slot - 1)) {
         *p = (struct apple9_pending){0};
         continue;
      }
      unsigned count = 0;
      for (unsigned i = 0; i < p->count; ++i) {
         if (!(apple9_producer_slots(&p->producer[i]) & mask)) {
            p->fenced[count] = p->fenced[i];
            p->arm[count] = p->arm[i];
            p->producer[count++] = p->producer[i];
         }
      }
      p->count = count;
   }
}

static bool
apple9_producer_result(const struct agx_apple9_vir_instr *producer, unsigned reg)
{
   return reg >= producer->dest && reg - producer->dest < producer->dest_components;
}

static bool
apple9_pending_result(const struct apple9_pending *p, unsigned reg)
{
   for (unsigned i = 0; i < p->count; ++i) {
      if (apple9_producer_result(&p->producer[i], reg))
         return true;
   }
   return false;
}

/* Wait for every slot in the mask with one standalone WAIT. Its operand is
 * the slot mask itself; it reads and writes no register. */
static bool
apple9_retire(struct agx_apple9_vir_program *out,
              struct apple9_completion *state, unsigned mask)
{
   struct apple9_pending *pending = state->pending;
   unsigned active = 0;
   u_foreach_bit(bit, mask) {
      if (apple9_pending_active(&pending[bit + 1]))
         active |= BITFIELD_BIT(bit);
   }
   if (!active)
      return true;
   struct agx_apple9_vir_instr wait = apple9_physical_instruction(
      AGX_APPLE9_VIR_WAIT, AGX_APPLE9_ENC_WAIT, AGX_APPLE9_VREG_INVALID);
   wait.immediate = active;
   if (!apple9_emit_physical(out, &wait))
      return false;
   apple9_complete(state, active);
   return true;
}

static bool
apple9_can_fold_wait(const struct agx_apple9_vir_instr *ins,
                     const struct agx_apple9_vir_instr *producer)
{
   if (ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC ||
       ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return false;
   if (agx_apple9_encoding_info(ins->encoding)->dependency_layout ==
       AGX_APPLE9_DEPENDENCY_NONE)
      return false;
   if (producer->op == AGX_APPLE9_VIR_SPILL_STORE)
      return true;
   bool reads = false;
   for (unsigned s = 0; s < ins->nr_srcs; ++s)
      reads |= apple9_producer_result(producer, ins->src[s]);
   /* An instruction that only orders against the producer (it overwrites
    * the result or a borrowed source, or accesses the same memory) waits in
    * its own dependency field before it executes, as a standalone WAIT
    * would. Atomics' dependency field gates their inputs (EXP-M4-50). */
   if (!reads)
      return ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC;
   /* Publication readers wait through their input mask (EXP-M4-61);
    * native coverage instructions wait for their publication the same way. */
   if (apple9_scalar_publication(producer))
      return (ins->op == AGX_APPLE9_VIR_VARY_STORE ||
              ins->op == AGX_APPLE9_VIR_OBJECT_GRID_EXPORT ||
              ins->op == AGX_APPLE9_VIR_COVERAGE) &&
             ins->src[0] == producer->dest;
   if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
       (ins->atomic_discard || ins->atomic_op == AGX_APPLE9_ATOMIC_CMPXCHG ||
        producer->op != AGX_APPLE9_VIR_DEVICE_LOAD ||
        producer->dest_components != 1 || ins->src[0] != producer->dest))
      return false;
   /* An identity copy explicitly requests materialization, independently of
    * which tuple components or later uses need the completed value. */
   if (apple9_identity(ins) && ins->src[0] == producer->dest)
      return true;
   /* Completion makes the whole producer tuple available. Keeping operands
    * live is independent of releasing its slot, but needs a source release
    * flag the consumer can clear: EXP-M4-34/35/63 for device loads, system
    * values and textures; CTS probes for atomic, private, shared and
    * interpolation results, including texture results in integer consumers.
    * Scratch fills keep their materialized form. */
   bool tuple = producer->op != AGX_APPLE9_VIR_SPILL_LOAD;
   bool can_retain =
      tuple && agx_apple9_encoding_info(ins->encoding)->dependency_retains_sources;
   unsigned consumed = 0;
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (!apple9_producer_result(producer, ins->src[s]))
         continue;
      if ((ins->live_after_mask & BITFIELD_BIT(s)) && !can_retain)
         return false;
      consumed |= BITFIELD_BIT(ins->src[s] - producer->dest);
   }
   return consumed == BITFIELD_MASK(producer->dest_components) ||
          (tuple && consumed != 0);
}

/* Producers merged from several paths share the slot, so one folded wait
 * retires whichever of them is outstanding. It must be valid for each. */
static bool
apple9_can_fold_pending(const struct agx_apple9_vir_instr *ins,
                        const struct apple9_pending *p)
{
   if (p->unknown || !p->count)
      return false;
   for (unsigned i = 0; i < p->count; ++i) {
      if (!apple9_can_fold_wait(ins, &p->producer[i]))
         return false;
   }
   return true;
}

/* Loop control repeats or leaves the body; results stay pending only across
 * forward execution-mask changes and jumps. */
static bool
apple9_loop_control(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
      return true;
   default:
      return false;
   }
}

static bool
apple9_completion_barrier(const struct agx_apple9_vir_instr *ins, bool drain)
{
   if (ins->op == AGX_APPLE9_VIR_ABI_RETURN)
      return true;
   if (apple9_after_logical_end(ins))
      return drain || apple9_loop_control(ins);
   return ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
          agx_apple9_encoding_info(ins->encoding)->requires_prior_completion;
}

/* Memory operations read their address, index and data registers at issue.
 * Native Metal overwrites a pending device load's address or index 3,744
 * times in 4,000 public pipelines without waiting. Overwriting atomic,
 * private-load and shared-load inputs right after issue passed the CTS
 * atomic, indexing and shared-memory groups. */
static bool
apple9_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_MESH_CONTROL:
      return true;
   default:
      return false;
   }
}

/* Hazards on shared execution resources rather than on a lane's registers or
 * memory: they hold however the producer's and consumer's lanes relate. */
static bool
apple9_resource_hazard(const struct agx_apple9_vir_instr *ins,
                       const struct agx_apple9_vir_instr *producer)
{
   /* T8132 r6 isolates the first channel submit as U0's publication consumer.
    * The post-U0 store is not a consumer. A/B are read at issue, so ordinary
    * register reuse must not manufacture an earlier publication deadline. */
   if (producer->op == AGX_APPLE9_VIR_MESH_CONTROL &&
       ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL)
      return true;
   /* Block-image stores and image-write parameter tuples publish on slot 1,
    * and image writes wait on slot 1 for their parameters. */
   return (ins->encoding == AGX_APPLE9_ENC_BLOCK_IMAGE_STORE ||
           ins->encoding == AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_EXTENDED ||
           ins->encoding == AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS ||
           ins->encoding == AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_16 ||
           ins->encoding == AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_2D ||
           ins->op == AGX_APPLE9_VIR_IMAGE_STORE) &&
          producer->producer_scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_1;
}

/* Hazards on a lane's own registers and memory. */
static bool
apple9_lane_hazard(const struct agx_apple9_vir_instr *ins,
                   const struct agx_apple9_vir_instr *producer)
{
   /* SAVE destroys its source just as a destination write would. In
    * particular, wait for earlier asynchronous borrowers before consuming it. */
   bool save = ins->op == AGX_APPLE9_VIR_SPILL_STORE;
   for (unsigned c = 0; c < ins->dest_components + save; ++c) {
      unsigned reg = save ? ins->src[0] : ins->dest + c;
      if (apple9_producer_result(producer, reg))
         return true;
      /* Includes texture and image-store parameter publications, which remain
       * borrowed until completion, and scratch SAVE's destructive input.
       * Bindless index GPRs (< PUB_BASE) for image stores and texture samples
       * are read at issue, like device-load address/index GPRs. */
      for (unsigned s = 0;
           !apple9_reads_sources_at_issue(producer) && s < producer->nr_srcs;
           ++s) {
         if ((producer->op == AGX_APPLE9_VIR_IMAGE_STORE ||
              producer->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) &&
             producer->src[s] < PUB_BASE)
            continue;
         if (producer->src[s] == reg)
            return true;
      }
   }
   bool is_spill_load =
      ins->op == AGX_APPLE9_VIR_SPILL_LOAD ||
      (ins->spill_access && ins->op == AGX_APPLE9_VIR_PRIVATE_LOAD);
   bool is_spill_store =
      ins->op == AGX_APPLE9_VIR_SPILL_STORE ||
      (ins->spill_access && ins->op == AGX_APPLE9_VIR_PRIVATE_STORE);
   bool prev_spill_load =
      producer->op == AGX_APPLE9_VIR_SPILL_LOAD ||
      (producer->spill_access && producer->op == AGX_APPLE9_VIR_PRIVATE_LOAD);
   bool prev_spill_store =
      producer->op == AGX_APPLE9_VIR_SPILL_STORE ||
      (producer->spill_access && producer->op == AGX_APPLE9_VIR_PRIVATE_STORE);
   if ((is_spill_load || is_spill_store) &&
       (prev_spill_load || prev_spill_store) &&
       ins->immediate == producer->immediate &&
       (is_spill_store || prev_spill_store))
      return true;
   /* Bindings name distinct buffers; global addresses may alias each other. */
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       (producer->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
        producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       ins->memory_global == producer->memory_global &&
       (ins->memory_global || ins->immediate == producer->immediate))
      return true;
   if (ins->op == AGX_APPLE9_VIR_PRIVATE_STORE &&
       producer->op == AGX_APPLE9_VIR_PRIVATE_LOAD)
      return true;
   if ((ins->op == AGX_APPLE9_VIR_SHARED_STORE ||
        ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) &&
       (producer->op == AGX_APPLE9_VIR_SHARED_LOAD ||
        producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC))
      return true;
   return ins->op == AGX_APPLE9_VIR_SHARED_LOAD &&
          producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC;
}

static bool
apple9_producer_hazard(const struct agx_apple9_vir_instr *ins,
                       const struct agx_apple9_vir_instr *producer)
{
   return apple9_resource_hazard(ins, producer) ||
          apple9_lane_hazard(ins, producer);
}

static enum agx_apple9_completion_class
apple9_completion_class(const struct agx_apple9_vir_instr *producer)
{
   if (apple9_scalar_publication(producer))
      return AGX_APPLE9_COMPLETION_PUBLICATION;
   /* Native T8132 Metal leaves device and threadgroup loads and returning
    * atomics pending across fences and barriers. Threadgroup and private
    * loads have fixed slots but are memory results too. Texture reads are
    * not: before a texture barrier Metal waits for a pending read_write
    * texture read, and our storage-image barriers have no such form. */
   switch (producer->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
      return AGX_APPLE9_COMPLETION_MEMORY;
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      return AGX_APPLE9_COMPLETION_TEXTURE;
   case AGX_APPLE9_VIR_IMAGE_STORE:
      return AGX_APPLE9_COMPLETION_IMAGE_STORE;
   default:
      return AGX_APPLE9_COMPLETION_OTHER;
   }
}

/* Producer classes a completion barrier leaves pending. Loop control and
 * drained block ends wait for everything, as do unspecialized fences. */
static unsigned
apple9_completion_exempt(const struct agx_apple9_vir_instr *ins)
{
   if (apple9_after_logical_end(ins) || ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return 0;
   if (ins->op == AGX_APPLE9_VIR_WORKGROUP_BARRIER &&
       ins->immediate == AGX_APPLE9_WORKGROUP_BARRIER_DEVICE) {
      /* Drain prior device reads/atomics before the workgroup device barrier
       * so release/acquire message passing never lets an in-flight read cross
       * the rendezvous. */
      return 0;
   }
   if (ins->op == AGX_APPLE9_VIR_DEVICE_FENCE) {
      switch (ins->immediate) {
      case AGX_APPLE9_FENCE_DEVICE:
         return AGX_APPLE9_COMPLETION_MEMORY;
      case AGX_APPLE9_FENCE_IMAGE_WRITE:
      case AGX_APPLE9_FENCE_IMAGE_WRITE_FS:
      case AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP:
      case AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM:
         /* FENCE_IMAGE_WRITE waits on IMAGE_STORE slots via its own 6-bit
          * fence_slot_mask without a standalone WAIT; unrelated memory and
          * publications stay pending while texture reads drain. */
         return AGX_APPLE9_COMPLETION_MEMORY | AGX_APPLE9_COMPLETION_PUBLICATION |
                AGX_APPLE9_COMPLETION_IMAGE_STORE;
      case AGX_APPLE9_FENCE_TEX_INVALIDATE:
      case AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP:
      case AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP_MEM:
         return AGX_APPLE9_COMPLETION_MEMORY | AGX_APPLE9_COMPLETION_PUBLICATION;
      default:
         return 0;
      }
   }
   return agx_apple9_encoding_info(ins->encoding)->prior_completion_exempt;
}

/* Acquire ordering: a device read (plain load or returning atomic) still
 * pending when a device fence issued must complete before a later device read
 * is performed, or that read may observe memory older than the earlier one
 * did (message passing through the fence). The device-fence drain this
 * replaces satisfied it; the dependency belongs on the later read, which
 * waits through its own input mask. Native Metal leaves such reads pending
 * across the fence, but places the later load on the same slot, so that is no
 * evidence they are unordered across slots. Threadgroup memory is not
 * covered: no workgroup-scope message-passing case failed without it. */
static bool
apple9_acquire_hazard(const struct agx_apple9_vir_instr *ins,
                      const struct agx_apple9_vir_instr *producer)
{
   /* Acquire ordering applies across both device and threadgroup memory
    * domains: a read (load or returning atomic) pending across a fence or
    * compiler barrier must complete before any subsequent memory operation. */
   const bool is_acquire_read =
      producer->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
      (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
       !producer->atomic_discard) ||
      producer->op == AGX_APPLE9_VIR_SHARED_LOAD ||
      producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC;
   if (!is_acquire_read)
      return false;
   return ins->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
          ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
          ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
          ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
          ins->op == AGX_APPLE9_VIR_IMAGE_STORE ||
          ins->op == AGX_APPLE9_VIR_SHARED_LOAD ||
          ins->op == AGX_APPLE9_VIR_SHARED_STORE ||
          ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC;
}

/* The if/else arms of the linear program. Node 0 is outside every region;
 * each other node is one arm (0 = then, 1 = else) of a push/else/pop region,
 * nested in its parent node. An asynchronous result writes only the lanes
 * active when it issued, so a producer in one arm of a region and an
 * instruction in the other arm act on disjoint lanes: native Metal issues a
 * sample into the same register in both arms with the first still pending. */
struct apple9_arm {
   unsigned parent, region;
   uint8_t arm;
};

struct apple9_arms {
   struct apple9_arm *node;
   unsigned *of; /* node of each input instruction */
};

static bool
apple9_build_arms(const struct agx_apple9_vir_program *program,
                  struct apple9_arms *arms)
{
   arms->node = calloc(program->instruction_count + 1, sizeof(*arms->node));
   arms->of = calloc(MAX2(program->instruction_count, 1), sizeof(*arms->of));
   if (!arms->node || !arms->of)
      return false;
   unsigned nodes = 1, regions = 0, current = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      arms->of[i] = current;
      switch (ins->op) {
      case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
         arms->node[nodes] = (struct apple9_arm){current, ++regions, 0};
         current = nodes++;
         break;
      case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
         /* Without a then arm to pair with, give up on disjointness. */
         if (!current || arms->node[current].arm)
            goto unstructured;
         arms->node[nodes] = (struct apple9_arm){
            arms->node[current].parent, arms->node[current].region, 1};
         current = nodes++;
         break;
      case AGX_APPLE9_VIR_EXEC_MASK_POP:
         if (!current)
            goto unstructured;
         current = arms->node[current].parent;
         break;
      default:
         break;
      }
   }
   return true;
unstructured:
   memset(arms->of, 0, program->instruction_count * sizeof(*arms->of));
   return true;
}

/* Results shown to be written only to the lanes active at issue: native Metal
 * issues a texture sample (k19) or device load (k20) into the same register
 * in both arms with the first still pending. Other asynchronous results have
 * no such evidence yet and keep their cross-arm hazards. */
static bool
apple9_masked_writeback(const struct agx_apple9_vir_instr *producer)
{
   return !apple9_fixed_producer_slot(producer) &&
          (producer->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
           producer->op == AGX_APPLE9_VIR_DEVICE_LOAD);
}

static bool
apple9_arms_disjoint(const struct apple9_arms *arms, unsigned a, unsigned b)
{
   for (unsigned x = a; x; x = arms->node[x].parent) {
      for (unsigned y = b; y; y = arms->node[y].parent) {
         if (arms->node[x].region == arms->node[y].region)
            return arms->node[x].arm != arms->node[y].arm;
      }
   }
   return false;
}

static bool
apple9_pending_hazard(const struct agx_apple9_vir_instr *ins,
                      const struct apple9_pending *p, bool barrier,
                      const struct apple9_arms *arms, unsigned arm)
{
   if (!apple9_pending_active(p))
      return false;
   if (p->unknown)
      return true;
   const unsigned exempt = barrier ? apple9_completion_exempt(ins) : 0;
   for (unsigned i = 0; i < p->count; ++i) {
      if (barrier && !(exempt & apple9_completion_class(&p->producer[i])))
         return true;
      /* TILE_ACCESS exempts plain async loads, never returning atomics. */
      if (barrier && ins->op == AGX_APPLE9_VIR_TILE_ACCESS &&
          (p->producer[i].op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
           p->producer[i].op == AGX_APPLE9_VIR_SHARED_ATOMIC))
         return true;
      if (apple9_resource_hazard(ins, &p->producer[i]))
         return true;
      /* Two masked writebacks from opposite arms of a region may target
       * the same register while the first is pending (k19, k20). Nothing
       * else crosses arms: an ALU write over another arm's pending load
       * has no such evidence (native Metal orders the arms so it never
       * happens) and failed memory_model.message_passing intermittently on
       * T8132 when allowed. */
      if (!(apple9_masked_writeback(&p->producer[i]) &&
            apple9_masked_writeback(ins) &&
            apple9_arms_disjoint(arms, p->arm[i], arm)) &&
          apple9_lane_hazard(ins, &p->producer[i]))
         return true;
      if (p->fenced[i] && apple9_acquire_hazard(ins, &p->producer[i]))
         return true;
   }
   return false;
}

static bool
apple9_same_producer(const struct agx_apple9_vir_instr *a,
                     const struct agx_apple9_vir_instr *b)
{
   if (a->op != b->op || a->encoding != b->encoding || a->dest != b->dest ||
       a->dest_components != b->dest_components || a->nr_srcs != b->nr_srcs ||
       a->immediate != b->immediate ||
       a->producer_scoreboard_slot != b->producer_scoreboard_slot)
      return false;
   for (unsigned s = 0; s < a->nr_srcs; ++s) {
      if (a->src[s] != b->src[s])
         return false;
   }
   return true;
}

/* Add one path's pending producers to a successor's entry state. */
static void
apple9_merge_pending(struct apple9_completion *to_state,
                     const struct apple9_completion *from_state)
{
   struct apple9_pending *to = to_state->pending;
   const struct apple9_pending *from = from_state->pending;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      struct apple9_pending *t = &to[slot];
      const struct apple9_pending *f = &from[slot];
      t->unknown |= f->unknown;
      t->latest_issue = MAX2(t->latest_issue, f->latest_issue);
      t->first_need = UINT_MAX;
      for (unsigned i = 0; i < f->count; ++i) {
         bool present = false;
         for (unsigned j = 0; j < t->count; ++j) {
            if (apple9_same_producer(&t->producer[j], &f->producer[i])) {
               t->fenced[j] |= f->fenced[i];
               present = true;
            }
         }
         if (present)
            continue;
         if (t->count == APPLE9_PENDING_MERGE) {
            t->unknown = true;
         } else {
            t->fenced[t->count] = f->fenced[i];
            t->arm[t->count] = f->arm[i];
            t->producer[t->count++] = f->producer[i];
         }
      }
      if (t->unknown)
         t->count = 0;
   }
}

/* Producers that may share a counting slot with each other. Fixed producers,
 * publications, atomics and scratch traffic keep a slot to themselves. */
static bool
apple9_shares_slot(const struct agx_apple9_vir_instr *ins)
{
   return !apple9_fixed_producer_slot(ins) &&
          (ins->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
           ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
           ins->op == AGX_APPLE9_VIR_SHARED_LOAD ||
           ins->op == AGX_APPLE9_VIR_PRIVATE_LOAD ||
           ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC);
}

/* The first instruction in [start, end) that needs the producer's completion,
 * by the same tests the scheduler applies, or UINT_MAX. */
static unsigned
apple9_first_need(const struct agx_apple9_vir_program *program, unsigned start,
                  unsigned end, const struct agx_apple9_vir_instr *producer,
                  bool drain)
{
   const unsigned limit = MIN2(end, start + 512);
   bool fenced = false;
   for (unsigned j = start; j < limit; ++j) {
      const struct agx_apple9_vir_instr *ins = program->instructions[j];
      if (ins->op == AGX_APPLE9_VIR_WAIT ||
          ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         continue;
      if (fenced && apple9_acquire_hazard(ins, producer))
         return j;
      fenced |= ins->op == AGX_APPLE9_VIR_DEVICE_FENCE;
      if (apple9_completion_barrier(ins, drain) &&
          !(apple9_completion_exempt(ins) & apple9_completion_class(producer)))
         return j;
      if (apple9_producer_hazard(ins, producer))
         return j;
      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         if (apple9_producer_result(producer, ins->src[s]))
            return j;
      }
   }
   return UINT_MAX;
}

/* Slots that fixed-slot producers issued in [start, end) will claim. A free
 * slot chosen for an ordinary producer should not be one of them, or the
 * fixed producer must retire it first. */
static unsigned
apple9_fixed_slots_ahead(const struct agx_apple9_vir_program *program,
                         unsigned start, unsigned end)
{
   unsigned slots = 0;
   for (unsigned j = start; j < MIN2(end, start + 512); ++j) {
      const struct agx_apple9_vir_instr *ins = program->instructions[j];
      unsigned slot = apple9_fixed_producer_slot(ins);
      if (slot) {
         slots |= BITFIELD_BIT(slot - 1) |
                  agx_apple9_encoding_info(ins->encoding)->fixed_producer_alternates;
      }
   }
   return slots;
}

/* Whether a sharing producer can be added to the slot without retiring it. */
static bool
apple9_slot_joinable(const struct apple9_completion *state, unsigned slot)
{
   const struct apple9_pending *p = &state->pending[slot];
   if (!p->count || p->unknown || p->count >= APPLE9_SLOT_SHARE_LIMIT)
      return false;
   for (unsigned i = 0; i < p->count; ++i) {
      if (!apple9_shares_slot(&p->producer[i]))
         return false;
   }
   return apple9_conflicting_slots(state, BITFIELD_BIT(slot - 1)) ==
          BITFIELD_BIT(slot - 1);
}

/* Choose a slot to join for a sharing producer first needed at need. With
 * free slots, only a slot whose first need is the same instruction, or whose
 * producers all issued in the other arm of an enclosing if/else: those are
 * alternatives on disjoint lanes that merge after it, so the consumer of the
 * merged value waits once for both (native Metal tags both arms' samples
 * alike).
 * When every slot is occupied, joining still beats retiring a slot now:
 * prefer the slot needed soonest at or after need (older producers are
 * waited for slightly early), else the one needed latest. */
static unsigned
apple9_join_slot(const struct apple9_completion *state, unsigned need,
                 bool exhausted, const struct apple9_arms *arms, unsigned arm)
{
   unsigned best = 0;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      if (!apple9_slot_joinable(state, slot))
         continue;
      const struct apple9_pending *p = &state->pending[slot];
      unsigned other = p->first_need;
      if (!exhausted) {
         if (need != UINT_MAX && other == need)
            return slot;
         bool alternatives = true;
         for (unsigned i = 0; i < p->count; ++i)
            alternatives &= apple9_arms_disjoint(arms, p->arm[i], arm);
         if (alternatives)
            return slot;
         continue;
      }
      if (!best) {
         best = slot;
         continue;
      }
      unsigned current = state->pending[best].first_need;
      bool after = other >= need, best_after = current >= need;
      if (after != best_after ? after
                              : after ? other < current : other > current)
         best = slot;
   }
   return best;
}

static void
apple9_select_compact_alu(struct agx_apple9_vir_instr *I)
{
   if (I->scoreboard_slot || I->saturate || I->src_abs_mask || I->src_neg_mask ||
       I->dest >= 64)
      return;
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (I->src[s] >= 64)
         return;
   }
   if (I->encoding == AGX_APPLE9_ENC_GET_SR_EXTENDED) {
      I->encoding = AGX_APPLE9_ENC_GET_SR;
      return;
   }
   unsigned inline_mask = I->alu_src_uniform_mask | I->alu_src_immediate_mask;
   if (I->encoding == AGX_APPLE9_ENC_FLOAT2_COMPACT && !inline_mask &&
       !I->src_hi_mask &&
       (I->op == AGX_APPLE9_VIR_FADD || I->op == AGX_APPLE9_VIR_FMUL)) {
      I->encoding = AGX_APPLE9_ENC_FLOAT2_BASE;
   } else if (I->encoding == AGX_APPLE9_ENC_FLOAT3_EXTENDED &&
              !(inline_mask & ~2) && !I->src_half_mask &&
              (!(I->alu_src_uniform_mask & 2) || I->alu_src_value[1] < 64) &&
              !((I->alu_src_immediate_mask & 2) && (I->alu_src_value[1] >> 31))) {
      I->encoding = AGX_APPLE9_ENC_FLOAT3_COMPACT;
   }
}

/* Schedule the final physical stream: allocation can introduce earlier uses,
 * register reuse and scratch traffic that do not exist in the original SSA.
 * Pending producers flow forward across execution-mask changes and forward
 * jumps, as native Metal code keeps them; each successor starts from the
 * union of its predecessors' pending producers. Loop control, back edges and
 * entry to a loop header wait for everything outstanding. */
static bool
apple9_schedule_physical(struct agx_apple9_vir_program *program,
                          const char **reason)
{
   if (!program->blocks || !program->last_block ||
       program->instruction_count == 0) {
      program->physical = true;
      program->dependencies_finalized = true;
      return true;
   }
   struct agx_apple9_vir_program out;
   if (!apple9_init_physical(&out)) {
      agx_apple9_vir_finish(&out);
      return false;
   }
   memcpy(out.reserved_gprs, program->reserved_gprs, sizeof(out.reserved_gprs));
   unsigned count = program->last_block->index + 1;
   struct agx_apple9_block **blocks = calloc(count, sizeof(*blocks));
   /* Machine successors (fall-through and jump target), then logical CFG
    * successors, which also name loop back edges. */
   struct agx_apple9_block **successors = calloc(4 * count, sizeof(*successors));
   unsigned *position = calloc(count, sizeof(*position));
   bool *header = calloc(count, sizeof(*header));
   struct apple9_completion *entry = calloc(count, sizeof(*entry));
   unsigned *out_start = calloc(count, sizeof(*out_start));
   unsigned *out_end = calloc(count, sizeof(*out_end));
   uint8_t *unfenced_in = calloc(count, sizeof(*unfenced_in));
   uint8_t *unfenced_out = calloc(count, sizeof(*unfenced_out));
   struct hash_table *nir_blocks = _mesa_pointer_hash_table_create(NULL);
   struct apple9_arms arms = {0};
   bool success = false;
   if (!blocks || !successors || !position || !header || !entry ||
       !out_start || !out_end || !unfenced_in || !unfenced_out ||
       !nir_blocks || !apple9_build_arms(program, &arms))
      goto cleanup;
   /* Several physical blocks may come from one NIR block. As with the old
    * layout scan, logical successors name its last physical block. Build the
    * mapping once instead of searching the whole shader for every CFG edge.
    */
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      if (block->nir)
         _mesa_hash_table_insert(nir_blocks, block->nir, block);
   }
   for (unsigned b = 0; b < count; ++b) {
      blocks[b] = agx_apple9_block_create(&out);
      if (!blocks[b])
         goto cleanup;
   }
   unsigned layout = 0;
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      blocks[block->index]->nir = block->nir;
      blocks[block->index]->nir_exit = block->nir_exit;
      position[block->index] = layout++;
      unsigned end = block->next ? block->next->start_index : program->instruction_count;
      struct agx_apple9_block **succ = &successors[4 * block->index];
      succ[0] = block->next;
      if (end > block->start_index) {
         const struct agx_apple9_vir_instr *last = program->instructions[end - 1];
         if (last->op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
             last->op == AGX_APPLE9_VIR_JMP_EXEC_NONE)
            succ[1] = last->branch_target;
      }
      nir_block *exit = block->nir_exit ?: block->nir;
      for (unsigned s = 0; exit && s < 2; ++s) {
         if (!exit->successors[s])
            continue;
         struct hash_entry *entry =
            _mesa_hash_table_search(nir_blocks, exit->successors[s]);
         if (entry)
            succ[2 + s] = entry->data;
      }
   }
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      for (unsigned i = 0; i < 4; ++i) {
         struct agx_apple9_block *succ = successors[4 * block->index + i];
         if (succ && position[succ->index] <= position[block->index])
            header[succ->index] = true;
      }
   }
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      agx_apple9_block_begin(&out, blocks[block->index]);
      out_start[block->index] = out.instruction_count;
      struct apple9_completion state = {0};
      if (!header[block->index])
         state = entry[block->index];
      struct apple9_pending *pending = state.pending;
      unsigned unknown = 0;
      for (unsigned slot = 1; slot <= 6; ++slot) {
         if (pending[slot].unknown)
            unknown |= BITFIELD_BIT(slot - 1);
      }
      if (!apple9_retire(&out, &state, unknown))
         goto cleanup;
      struct agx_apple9_block *const *succ = &successors[4 * block->index];
      bool drain = !succ[0] && !succ[1] && !succ[2] && !succ[3];
      for (unsigned i = 0; i < 4; ++i) {
         drain |= succ[i] && (header[succ[i]->index] ||
                              position[succ[i]->index] <= position[block->index]);
      }
      unsigned end = block->next ? block->next->start_index : program->instruction_count;
      for (unsigned i = block->start_index; i < end; ++i) {
         struct agx_apple9_vir_instr ins = *program->instructions[i];
         if (ins.branch_target)
            ins.branch_target = blocks[ins.branch_target->index];
         if (ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
            assert(out.instruction_count &&
                   out.instructions[out.instruction_count - 1]->op == AGX_APPLE9_VIR_DEVICE_ATOMIC);
            ins.producer_scoreboard_slot =
               out.instructions[out.instruction_count - 1]->producer_scoreboard_slot;
            if (!apple9_emit_physical(&out, &ins))
               goto cleanup;
            continue;
         }

         if (ins.op == AGX_APPLE9_VIR_WAIT) {
            apple9_complete(&state, ins.immediate);
            if (!apple9_emit_physical(&out, &ins))
               goto cleanup;
            continue;
         }

         ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         const bool is_img_fence =
            ins.op == AGX_APPLE9_VIR_DEVICE_FENCE &&
            (ins.immediate == AGX_APPLE9_FENCE_IMAGE_WRITE ||
             ins.immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_FS ||
             ins.immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP ||
             ins.immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM);
         const bool barrier = apple9_completion_barrier(&ins, drain);
         unsigned fold = 0, retire = 0;
         for (unsigned slot = 1; slot <= 6; ++slot) {
            const struct apple9_pending *p = &pending[slot];
            if (!apple9_pending_active(p))
               continue;
            bool needed =
               apple9_pending_hazard(&ins, p, barrier, &arms, arms.of[i]);
            for (unsigned s = 0; s < ins.nr_srcs; ++s)
               needed |= apple9_pending_result(p, ins.src[s]);
            if (!needed)
               continue;
            if (!fold && apple9_can_fold_pending(&ins, p))
               fold = slot;
            else
               retire |= BITFIELD_BIT(slot - 1);
         }
         if (!apple9_retire(&out, &state, retire))
            goto cleanup;
         /* FENCE_IMAGE_WRITE itself waits on fence_slot_mask in hardware,
          * retiring only scoreboard slots whose currently pending producer is
          * an IMAGE_STORE (never retiring reused non-IMAGE_STORE slots). */
         if (is_img_fence) {
            uint8_t active_img_slots = 0;
            for (unsigned slot = 1; slot <= 6; ++slot) {
               const struct apple9_pending *p = &pending[slot];
               if (p->unknown || !p->count)
                  continue;
               bool all_img_store = true;
               for (unsigned k = 0; k < p->count; ++k) {
                  if (p->producer[k].op != AGX_APPLE9_VIR_IMAGE_STORE) {
                     all_img_store = false;
                     break;
                  }
               }
               if (all_img_store)
                  active_img_slots |= (uint8_t)BITFIELD_BIT(slot - 1);
            }
            ins.fence_slot_mask = active_img_slots;
            if (active_img_slots)
               apple9_complete(&state, active_img_slots);
         }
         /* An alternate selector in the standalone wait may already have
          * completed the producer chosen for folding. */
         if (fold && !apple9_pending_active(&pending[fold]))
            fold = 0;
         if (fold) {
            /* Logic's pending-result source is A. Its operators commute. */
            if ((ins.encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED ||
                 ins.encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) && ins.nr_srcs == 2 &&
                !apple9_pending_result(&pending[fold], ins.src[0]) &&
                apple9_pending_result(&pending[fold], ins.src[1])) {
               unsigned src = ins.src[0];
               ins.src[0] = ins.src[1];
               ins.src[1] = src;
               ins.live_after_mask = ((ins.live_after_mask & 1) << 1) |
                                     ((ins.live_after_mask & 2) >> 1);
            }
            ins.scoreboard_slot = fold;
            apple9_complete(&state, BITFIELD_BIT(fold - 1));
         }

         if (apple9_async(&ins)) {
            unsigned slot = apple9_fixed_producer_slot(&ins);
            const bool share = apple9_shares_slot(&ins);
            const unsigned need =
               share ? apple9_first_need(program, i + 1, end, &ins, drain)
                     : UINT_MAX;
            bool joined = false;
            if (share) {
               slot = apple9_join_slot(&state, need, false, &arms, arms.of[i]);
               joined = slot != 0;
            }
            if (!slot) {
               /* Samples and memory traffic share six completion tags.
                * Keep independent operations pending until a real hazard,
                * consumer, or tag exhaustion requires their result handoff. */
               unsigned first = ins.op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ? 1
                              : ins.op == AGX_APPLE9_VIR_IMAGE_STORE ? 2 : 6;
               const unsigned avoid = apple9_fixed_slots_ahead(
                  program, i + 1, need == UINT_MAX ? end : need);
               for (unsigned pass = 0; !slot && pass < 2; ++pass) {
                  for (unsigned p = 0; p < 6; ++p) {
                     unsigned candidate = (first - 1 + p) % 6 + 1;
                     if ((!pass && (avoid & BITFIELD_BIT(candidate - 1))) ||
                         apple9_conflicting_slots(
                            &state, BITFIELD_BIT(candidate - 1)))
                        continue;
                     slot = candidate;
                     break;
                  }
               }
               if (!slot && share) {
                  slot = apple9_join_slot(&state, need, true, &arms, arms.of[i]);
                  joined = slot != 0;
               }
               if (!slot) {
                  /* All tags are occupied. Retire the oldest pending group,
                   * allowing recent loads to overlap instead of repeatedly
                   * draining the newest producer in the preferred slot.
                   * At a CFG merge use the latest issue on either path. */
                  slot = first;
                  for (unsigned candidate = 1; candidate <= 6; ++candidate) {
                     if (pending[candidate].latest_issue < pending[slot].latest_issue)
                        slot = candidate;
                  }
               }
            }
            unsigned reservations = BITFIELD_BIT(slot - 1);
            if (apple9_fixed_producer_slot(&ins))
               reservations |=
                  agx_apple9_encoding_info(ins.encoding)->fixed_producer_alternates;
            /* In particular, never issue another U0 while one is pending.
             * The fully waited double-U0 r6 control timed out, so this is a
             * compiler scheduling invariant, not a claim that replay is safe. */
            if (!joined &&
                !apple9_retire(&out, &state,
                               apple9_conflicting_slots(&state, reservations)))
               goto cleanup;
            ins.producer_scoreboard_slot = slot;
            if (ins.op == AGX_APPLE9_VIR_DEVICE_LOAD) {
               const uint16_t tokens[] = {
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_1100, AGX_APPLE9_DEVICE_LOAD_TOKEN_5100,
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_9100, AGX_APPLE9_DEVICE_LOAD_TOKEN_D100,
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_1101, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
               };
               ins.device_load_raw_token = tokens[slot - 1];
            }
            if (joined) {
               struct apple9_pending *p = &pending[slot];
               p->fenced[p->count] = false;
               p->arm[p->count] = arms.of[i];
               p->producer[p->count++] = ins;
               p->latest_issue = out.instruction_count;
               p->first_need = MIN2(p->first_need, need);
            } else {
               pending[slot] = (struct apple9_pending){
                  .count = 1, .latest_issue = out.instruction_count,
                  .first_need = need, .producer = {ins}, .arm = {arms.of[i]}};
            }
         }
         apple9_select_compact_alu(&ins);
         if (!apple9_emit_physical(&out, &ins))
            goto cleanup;
         if (ins.op == AGX_APPLE9_VIR_DEVICE_FENCE ||
             (ins.op == AGX_APPLE9_VIR_WORKGROUP_BARRIER &&
              ins.immediate == AGX_APPLE9_WORKGROUP_BARRIER_DEVICE)) {
            for (unsigned slot = 1; slot <= 6; ++slot) {
               for (unsigned k = 0; k < pending[slot].count; ++k)
                  pending[slot].fenced[k] = true;
            }
         }
      }
      if (drain) {
         if (!apple9_retire(&out, &state, BITFIELD_MASK(6)))
            goto cleanup;
      } else {
         for (unsigned i = 0; i < 4; ++i) {
            bool repeated = false;
            for (unsigned j = 0; j < i; ++j)
               repeated |= succ[j] == succ[i];
            if (succ[i] && !repeated)
               apple9_merge_pending(&entry[succ[i]->index], &state);
         }
      }
      out_end[block->index] = out.instruction_count;
   }
   /* Propagate unfenced IMAGE_STORE scoreboard slots across the complete CFG
    * (including loop headers, backedges, and drained blocks) using the
    * recorded per-block output ranges indexed by program->blocks indices
    * (matching successors[]). Never clear bits already credited to
    * active_img_slots above; only OR in additional reaching store slots. */
   bool unfenced_changed = true;
   while (unfenced_changed) {
      unfenced_changed = false;
      for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
         uint8_t m = unfenced_in[b->index];
         for (unsigned i = out_start[b->index]; i < out_end[b->index]; ++i) {
            struct agx_apple9_vir_instr *ins = out.instructions[i];
            if (ins->op == AGX_APPLE9_VIR_IMAGE_STORE &&
                ins->producer_scoreboard_slot >= 1 &&
                ins->producer_scoreboard_slot <= 6) {
               m |= (uint8_t)BITFIELD_BIT(ins->producer_scoreboard_slot - 1);
            } else if (ins->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
                       (ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE ||
                        ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_FS ||
                        ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP ||
                        ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM)) {
               ins->fence_slot_mask |= m;
               m = 0;
            }
         }
         if (m != unfenced_out[b->index]) {
            unfenced_out[b->index] = m;
            struct agx_apple9_block *const *bsucc = &successors[4 * b->index];
            for (unsigned s = 0; s < 4; ++s) {
               if (bsucc[s] && (unfenced_in[bsucc[s]->index] | m) !=
                                  unfenced_in[bsucc[s]->index]) {
                  unfenced_in[bsucc[s]->index] |= m;
                  unfenced_changed = true;
               }
            }
         }
      }
   }
   /* If a FENCE_IMAGE_WRITE still has no reaching IMAGE_STORE (e.g. a barrier
    * on a path where stores only occur in another branch), fall back to the
    * union of all IMAGE_STORE scoreboard slots in the program. */
   unsigned all_img_store_slots = 0;
   for (unsigned i = 0; i < out.instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = out.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_IMAGE_STORE &&
          ins->producer_scoreboard_slot >= 1 &&
          ins->producer_scoreboard_slot <= 6)
         all_img_store_slots |= BITFIELD_BIT(ins->producer_scoreboard_slot - 1);
   }
   if (all_img_store_slots) {
      for (unsigned i = 0; i < out.instruction_count; ++i) {
         struct agx_apple9_vir_instr *ins = out.instructions[i];
         if (ins->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
             !ins->fence_slot_mask &&
             (ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE ||
              ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_FS ||
              ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP ||
              ins->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM))
            ins->fence_slot_mask = (uint8_t)all_img_store_slots;
      }
   }
   agx_apple9_vir_reindex(&out);
   out.private_size = program->private_size;
   out.spill_slots = program->spill_slots;
   out.scratch_size = program->scratch_size;
   out.peak_live_gprs = program->peak_live_gprs;
   out.max_phys_gpr = MAX2(out.max_phys_gpr, program->max_phys_gpr);
   out.publication_count = program->publication_count;
   out.fragment_shader = program->fragment_shader;
   out.output = program->output;
   out.physical = program->physical;
   out.dependencies_finalized = true;
   agx_apple9_vir_finish(program);
   *program = out;
   for (struct agx_apple9_block *block = program->allocated_blocks; block;
        block = block->allocated_next)
      block->program = program;
   memset(&out, 0, sizeof(out));
   success = true;
cleanup:
   if (!success && reason)
      *reason = "Could not schedule Apple9 physical completions";
   free(blocks);
   free(successors);
   free(position);
   free(header);
   free(entry);
   free(out_start);
   free(out_end);
   free(unfenced_in);
   free(unfenced_out);
   free(arms.node);
   free(arms.of);
   _mesa_hash_table_destroy(nir_blocks, NULL);
   agx_apple9_vir_finish(&out);
   return success;
}


static struct agx_apple9_allocation_cost
apple9_allocation_cost(const struct agx_apple9_vir_program *p)
{
   struct agx_apple9_allocation_cost cost = {
      .registers = p->max_phys_gpr + 1,
      .peak_registers = MAX2(p->max_phys_gpr + 1, p->peak_live_gprs),
   };
   unsigned nesting = 0;
   unsigned ready[AGX_APPLE9_GPR_COUNT] = {0}, issue = 0;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *I = p->instructions[i];
      if (I->op == AGX_APPLE9_VIR_LOOP_MASK_POP && nesting)
         --nesting;
      /* Static loop weighting, not a cycle/occupancy estimate. It prevents
       * saving one setup move by adding a move on every loop iteration. */
      unsigned weight = 1u << (3 * MIN2(nesting, 3));
      unsigned earliest = issue;
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s] < ARRAY_SIZE(ready))
            earliest = MAX2(earliest, ready[I->src[s]]);
      }
      cost.dependency_gaps += weight * (earliest - issue);
      for (unsigned c = 0; c < I->dest_components; ++c) {
         if (I->dest + c < ARRAY_SIZE(ready))
            ready[I->dest + c] = earliest + apple9_dependency_delay(I);
      }
      issue = earliest + 1;
      cost.spills += weight * (I->op == AGX_APPLE9_VIR_SPILL_STORE ||
                               I->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                               I->spill_access);
      if (I->encoding == AGX_APPLE9_ENC_PSEUDO)
         continue;
      cost.instructions += weight;
      cost.bytes += weight * agx_apple9_encoding_info(I->encoding)->length;
      cost.has_copies |= apple9_identity(I) && I->dest != I->src[0];
      if (I->op == AGX_APPLE9_VIR_LOOP_MASK_PUSH)
         ++nesting;
   }
   return cost;
}

bool
agx_apple9_allocation_cheaper(struct agx_apple9_allocation_cost a,
                              struct agx_apple9_allocation_cost b)
{
   if (a.spills != b.spills)
      return a.spills < b.spills;
   unsigned a_threads =
      agx_occupancy_for_register_count(2 * MAX2(a.registers, 1)).max_threads;
   unsigned b_threads =
      agx_occupancy_for_register_count(2 * MAX2(b.registers, 1)).max_threads;
   if (a_threads != b_threads)
      return a_threads > b_threads;
   if (a.dependency_gaps != b.dependency_gaps)
      return a.dependency_gaps < b.dependency_gaps;
   if (a.instructions != b.instructions)
      return a.instructions < b.instructions;
   if (a.bytes != b.bytes)
      return a.bytes < b.bytes;
   return a.registers < b.registers;
}

/* Estimate peak live 32-bit GPRs across the VIR program in O(N + V) so
 * agx_apple9_allocate_shared can select bank-cost / latency-scheduling policy
 * directly without running a multi-candidate brute-force search. */
static unsigned
apple9_estimate_vir_pressure(const struct agx_apple9_vir_program *program)
{
   const unsigned n = program->instruction_count;
   const unsigned vcount = program->value_count;
   if (!n || !vcount)
      return 0;

   uint32_t *canon = malloc(vcount * sizeof(*canon));
   unsigned *def_pos = malloc(vcount * sizeof(*def_pos));
   unsigned *last_use = calloc(vcount, sizeof(*last_use));
   int32_t *delta = calloc(n + 2, sizeof(*delta));
   if (!canon || !def_pos || !last_use || !delta) {
      free(canon);
      free(def_pos);
      free(last_use);
      free(delta);
      return 0;
   }
   for (unsigned v = 0; v < vcount; ++v) {
      canon[v] = v;
      def_pos[v] = UINT_MAX;
   }

   struct {
      unsigned start, end;
   } loops[32];
   unsigned nr_loops = 0;
   unsigned loop_stack[32];
   unsigned depth = 0;

   for (unsigned i = 0; i < n; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_LOOP_MASK_PUSH && depth < ARRAY_SIZE(loop_stack))
         loop_stack[depth++] = i;
      else if (ins->op == AGX_APPLE9_VIR_LOOP_MASK_POP && depth) {
         unsigned start = loop_stack[--depth];
         if (nr_loops < ARRAY_SIZE(loops)) {
            loops[nr_loops].start = start;
            loops[nr_loops].end = i;
            nr_loops++;
         }
      }

      if (apple9_identity(ins) && !ins->publication_handoff &&
          ins->dest < vcount && ins->src[0] < vcount) {
         canon[ins->dest] = canon[ins->src[0]];
         continue;
      }

      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         uint32_t v = ins->src[s];
         if (v < vcount) {
            v = canon[v];
            if (!program->publication[v])
               last_use[v] = MAX2(last_use[v], i);
         }
      }
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC && ins->target < vcount) {
         uint32_t t = canon[ins->target];
         if (!program->publication[t]) {
            def_pos[t] = MIN2(def_pos[t], i);
            last_use[t] = MAX2(last_use[t], i);
         }
      }
      if (ins->dest != AGX_APPLE9_VREG_INVALID) {
         for (unsigned c = 0; c < ins->dest_components; ++c) {
            uint32_t d = ins->dest + c;
            if (d < vcount && !program->publication[d]) {
               def_pos[d] = MIN2(def_pos[d], i);
               last_use[d] = MAX2(last_use[d], i);
            }
         }
      }
   }

   if (program->output != AGX_APPLE9_VREG_INVALID && program->output < vcount) {
      uint32_t v = canon[program->output];
      if (!program->publication[v])
         last_use[v] = n;
   }
   for (unsigned i = 0; i < program->live_out_count; ++i) {
      uint32_t v = program->live_out[i];
      if (v < vcount) {
         v = canon[v];
         if (!program->publication[v])
            last_use[v] = n;
      }
   }

   for (unsigned v = 0; v < vcount; ++v) {
      if (canon[v] != v || program->publication[v] || def_pos[v] >= n)
         continue;
      unsigned d = def_pos[v];
      unsigned u = last_use[v];
      for (unsigned l = 0; l < nr_loops; ++l) {
         if (d < loops[l].start && u >= loops[l].start && u < loops[l].end)
            u = loops[l].end;
      }
      delta[d] += 1;
      delta[u + 1] -= 1;
   }

   int32_t cur = 0, peak = 0;
   for (unsigned i = 0; i <= n; ++i) {
      cur += delta[i];
      if (cur > peak)
         peak = cur;
   }

   free(canon);
   free(def_pos);
   free(last_use);
   free(delta);
   return (unsigned)peak;
}


static bool
apple9_run_native_candidate(struct agx_apple9_vir_program *program,
                            bool cost_constraints, bool latency, unsigned slack,
                            bool uniform_remat, bool *copy_register,
                            struct agx_apple9_vir_program *out,
                            const char **reason, bool *remat_used)
{
   unsigned headroom = 0;
   for (unsigned attempt = 0; attempt < 6; ++attempt) {
      agx_apple9_vir_finish(out);
      agx_apple9_vir_init(out);
      if (!agx_apple9_vir_clone(out, program)) {
         *reason = "out of memory cloning Apple9 VIR for native RA";
         return false;
      }

      struct apple9_ra_assign_options opts = apple9_ra_assign_default_options();
      opts.max_gprs = 0; /* Full Apple9 register file with occupancy-tier bounding */
      opts.reserved_gprs = *copy_register ? 2u : 0u;
      opts.cost_register_constraints = cost_constraints;
      opts.latency_schedule = latency;
      opts.pressure_slack = slack;
      opts.disable_uniform_remat = !uniform_remat;
      const uint64_t dbg = agx_get_compiler_debug();
      opts.skip_physical_schedule = true;
      opts.spill_headroom = headroom;
      opts.force_spill = !!(dbg & AGX_DBG_SPILL);
#ifndef NDEBUG
      opts.validate = !(dbg & AGX_DBG_NOVALIDATE);
#endif

      struct apple9_ra_assign_result res;
      if (apple9_ra_allocate_native(out, &opts, &res)) {
         if (remat_used)
            *remat_used = res.uniform_remat_used;
         if (!apple9_schedule_physical(out, reason))
            return false;
         return true;
      }

      if (!*copy_register && res.needs_copy_register) {
         *copy_register = true;
         continue;
      }
      if (res.needs_spill_headroom && headroom < 16u) {
         headroom += 4u;
         continue;
      }

      if (res.validate_err.failed) {
         static _Thread_local char validate_msg[256];
         snprintf(validate_msg, sizeof(validate_msg), "%s",
                  res.validate_err.message);
         *reason = validate_msg;
      } else {
         *reason = res.error ? res.error
                             : "native Apple9 register allocation failed";
      }
      return false;
   }
   return false;
}

static bool
apple9_allocate_native_driver(struct agx_apple9_vir_program *program,
                              const char **reason)
{
   const unsigned peak_gprs = apple9_estimate_vir_pressure(program);
   unsigned long_dependencies = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      long_dependencies += apple9_dependency_delay(program->instructions[i]) > 1;

   const bool high_pressure = peak_gprs > 104;
   const bool first_banks = high_pressure;
   const bool first_latency = !high_pressure && long_dependencies > 1;
   unsigned first_slack = 0;
   if (first_latency) {
      struct agx_occupancy tier =
         agx_occupancy_for_register_count(2 * MAX2(peak_gprs, 1));
      unsigned tier_gprs = tier.max_registers / 2;
      if (tier_gprs > peak_gprs + 1)
         first_slack = MIN2(8u, tier_gprs - peak_gprs - 1);
   }

   struct agx_apple9_vir_program best, alternative;
   agx_apple9_vir_init(&best);
   agx_apple9_vir_init(&alternative);
   bool copy_register = false;
   bool remat_used = false;
   bool used_banks = first_banks;

   bool ok = apple9_run_native_candidate(program, first_banks, first_latency,
                                         first_slack, true, &copy_register,
                                         &best, reason, &remat_used);
   if (!ok && (!first_banks || first_latency)) {
      agx_apple9_vir_finish(&best);
      agx_apple9_vir_init(&best);
      used_banks = true;
      ok = apple9_run_native_candidate(program, true, false, 0, true,
                                       &copy_register, &best, reason,
                                       &remat_used);
   }
   if (!ok) {
      agx_apple9_vir_finish(&best);
      agx_apple9_vir_finish(&alternative);
      return false;
   }

   struct agx_apple9_allocation_cost cost = apple9_allocation_cost(&best);
   if (cost.spills) {
      if (!used_banks) {
         const char *unused_reason = NULL;
         bool alt_remat = false;
         bool alt_copy = copy_register;
         if (apple9_run_native_candidate(program, true, false, 0, true,
                                         &alt_copy, &alternative,
                                         &unused_reason, &alt_remat)) {
            struct agx_apple9_allocation_cost alt_cost =
               apple9_allocation_cost(&alternative);
            if (agx_apple9_allocation_cheaper(alt_cost, cost)) {
               SWAP(best, alternative);
               cost = alt_cost;
               remat_used = alt_remat;
               copy_register = alt_copy;
            }
         }
         agx_apple9_vir_finish(&alternative);
         agx_apple9_vir_init(&alternative);
      }

      if (cost.spills && remat_used) {
         const char *unused_reason = NULL;
         bool alt_copy = copy_register;
         if (apple9_run_native_candidate(program, true, false, 0, false,
                                         &alt_copy, &alternative,
                                         &unused_reason, NULL)) {
            struct agx_apple9_allocation_cost alt_cost =
               apple9_allocation_cost(&alternative);
            if (agx_apple9_allocation_cheaper(alt_cost, cost)) {
               SWAP(best, alternative);
               cost = alt_cost;
            }
         }
         agx_apple9_vir_finish(&alternative);
      }
   }

   agx_apple9_vir_finish(program);
   *program = best;
   for (struct agx_apple9_block *block = program->allocated_blocks; block;
        block = block->allocated_next)
      block->program = program;
   return true;
}


bool
agx_apple9_allocate_shared(struct agx_apple9_vir_program *program,
                           nir_shader *nir, const char **reason)
{
   (void)nir;

   /* Reject unsafe U0 repetition before allocation or scheduling can insert
    * waits that would disguise it as an ordered, but still unqualified, form. */
   agx_apple9_vir_reindex(program);
   if (!agx_apple9_validate_mesh_control(program, reason))
      return false;

   if (!agx_apple9_allocate_publications(program, reason))
      return false;

   return apple9_allocate_native_driver(program, reason);
}
