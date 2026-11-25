/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * Test-only Apple9 VIR support routines (key serializer, allocation validator,
 * and legacy VIR scoreboard assigner). Linked only into compiler test targets,
 * never into libgravity_compiler.
 */

#include "agx_apple9_ir_priv.h"
#include "agx_compile.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APPLE9_LAST_ALLOCATABLE_GPR  (AGX_APPLE9_GPR_COUNT - 1)

static uint32_t
apple9_block_ref_map(const struct agx_apple9_block *block,
                     struct hash_table *block_ids)
{
   if (!block)
      return UINT32_MAX;
   if (!block_ids)
      return block->index;
   struct hash_entry *e = _mesa_hash_table_search(block_ids, block);
   return e ? (uint32_t)((uintptr_t)e->data - 1u) : UINT32_MAX;
}

static uint32_t
apple9_instr_ref_map(const struct agx_apple9_vir_instr *instr,
                     struct hash_table *instr_pos)
{
   if (!instr)
      return UINT32_MAX;
   if (!instr_pos)
      return agx_apple9_instr_position(instr);
   struct hash_entry *e = _mesa_hash_table_search(instr_pos, instr);
   return e ? (uint32_t)((uintptr_t)e->data - 1u) : UINT32_MAX;
}

struct apple9_key_nir {
   uint32_t first, last, first_exit;
};

/* The first and last placed block selected from a NIR block, and the first
 * block with it as NIR exit, as layout indices: the relations allocation
 * derives from the logical CFG. */
static void
apple9_key_nir_block(struct hash_table *nir_blocks, const struct nir_block *nir,
                     struct util_dynarray *key)
{
   struct hash_entry *entry = nir ? _mesa_hash_table_search(nir_blocks, nir) : NULL;
   const struct apple9_key_nir none = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
   const struct apple9_key_nir *k = entry ? entry->data : &none;
   util_dynarray_append(key, k->first);
   util_dynarray_append(key, k->last);
   util_dynarray_append(key, k->first_exit);
}

static struct apple9_key_nir *
apple9_key_nir_entry(struct hash_table *nir_blocks, const struct nir_block *nir)
{
   struct hash_entry *entry = _mesa_hash_table_search(nir_blocks, nir);
   if (entry)
      return entry->data;
   struct apple9_key_nir *k = ralloc(nir_blocks, struct apple9_key_nir);
   if (!k)
      return NULL;
   *k = (struct apple9_key_nir){UINT32_MAX, UINT32_MAX, UINT32_MAX};
   _mesa_hash_table_insert(nir_blocks, nir, k);
   return k;
}

bool
agx_apple9_vir_key(const struct agx_apple9_vir_program *program,
                   struct util_dynarray *key)
{
   const uint32_t header[] = {
      program->instruction_count, program->num_cfg_blocks, program->value_count,
      program->output, program->fragment_shader, program->physical,
      program->dependencies_finalized, program->live_out_count,
      program->publication_count, program->peak_live_gprs,
      program->max_phys_gpr, program->private_size, program->scratch_size,
      program->spill_slots, program->phys ? program->phys_capacity : 0,
      program->publication ? program->publication_capacity : 0,
   };
   util_dynarray_append_array(key, uint8_t, header, sizeof(header));
   util_dynarray_append_array(key, uint8_t, program->reserved_gprs,
                              sizeof(program->reserved_gprs));
   if (program->value_count) {
      util_dynarray_append_array(key, uint8_t, program->fixed_phys, program->value_count);
      util_dynarray_append_array(key, uint8_t, program->max_phys, program->value_count);
   }
   if (program->live_out_count) {
      util_dynarray_append_array(key, uint8_t, program->live_out,
                                 program->live_out_count * sizeof(*program->live_out));
      util_dynarray_append_array(key, uint8_t, program->live_out_reg,
                                 program->live_out_count);
   }
   if (program->phys)
      util_dynarray_append_array(key, uint8_t, program->phys, program->phys_capacity);
   if (program->publication)
      util_dynarray_append_array(key, uint8_t, program->publication,
                                 program->publication_capacity * sizeof(bool));

   struct hash_table *block_ids = NULL;
   struct hash_table *instr_pos = NULL;
   struct hash_table *block_starts = NULL;
   if (program->index_dirty) {
      block_ids = _mesa_pointer_hash_table_create(NULL);
      instr_pos = _mesa_pointer_hash_table_create(NULL);
      block_starts = _mesa_pointer_hash_table_create(NULL);
      if (!block_ids || !instr_pos || !block_starts) {
         _mesa_hash_table_destroy(block_ids, NULL);
         _mesa_hash_table_destroy(instr_pos, NULL);
         _mesa_hash_table_destroy(block_starts, NULL);
         return false;
      }
      uint32_t bi = 0;
      uint32_t pos = 0;
      for (const struct agx_apple9_block *b = program->blocks; b; b = b->next) {
         _mesa_hash_table_insert(block_ids, b, (void *)(uintptr_t)(bi + 1u));
         _mesa_hash_table_insert(block_starts, b, (void *)(uintptr_t)(pos + 1u));
         for (const struct agx_apple9_vir_instr *I =
                 agx_apple9_block_first_instr(b);
              I; I = agx_apple9_instr_next(I)) {
            _mesa_hash_table_insert(instr_pos, I, (void *)(uintptr_t)(pos + 1u));
            pos++;
         }
         bi++;
      }
   }

   struct hash_table *nir_blocks = _mesa_pointer_hash_table_create(NULL);
   if (!nir_blocks) {
      _mesa_hash_table_destroy(block_ids, NULL);
      _mesa_hash_table_destroy(instr_pos, NULL);
      _mesa_hash_table_destroy(block_starts, NULL);
      return false;
   }
   for (const struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      struct apple9_key_nir *k = b->nir ? apple9_key_nir_entry(nir_blocks, b->nir) : NULL;
      struct apple9_key_nir *e = b->nir_exit ? apple9_key_nir_entry(nir_blocks, b->nir_exit) : NULL;
      if ((b->nir && !k) || (b->nir_exit && !e)) {
         _mesa_hash_table_destroy(nir_blocks, NULL);
         _mesa_hash_table_destroy(block_ids, NULL);
         _mesa_hash_table_destroy(instr_pos, NULL);
         _mesa_hash_table_destroy(block_starts, NULL);
         return false;
      }
      uint32_t b_idx = apple9_block_ref_map(b, block_ids);
      if (k) {
         if (k->first == UINT32_MAX)
            k->first = b_idx;
         k->last = b_idx;
      }
      if (e && e->first_exit == UINT32_MAX)
         e->first_exit = b_idx;
   }

   for (const struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      uint32_t b_idx = apple9_block_ref_map(b, block_ids);
      uint32_t b_start = b->start_index;
      if (block_starts) {
         struct hash_entry *se = _mesa_hash_table_search(block_starts, b);
         b_start = se ? (uint32_t)((uintptr_t)se->data - 1u) : 0u;
      }
      const uint32_t fields[] = {
         b_idx, b->placed, b_start, b->cfg_active, b->loop_header,
         b->cfg_index, b->num_cfg_preds,
         apple9_block_ref_map(b->cfg_succs[0], block_ids),
         apple9_block_ref_map(b->cfg_succs[1], block_ids),
         b->nir != NULL, b->nir_exit != NULL,
      };
      util_dynarray_append_array(key, uint8_t, fields, sizeof(fields));
      for (unsigned p = 0; p < b->num_cfg_preds; ++p)
         util_dynarray_append(key, apple9_block_ref_map(b->cfg_preds[p], block_ids));
      /* Earlier liveness, which publication lifetimes still consult. */
      const uint32_t live_words = b->live_in && b->live_out ? b->live_words : UINT32_MAX;
      util_dynarray_append(key, live_words);
      if (live_words != UINT32_MAX) {
         util_dynarray_append_array(key, uint8_t, b->live_in, live_words * sizeof(BITSET_WORD));
         util_dynarray_append_array(key, uint8_t, b->live_out, live_words * sizeof(BITSET_WORD));
      }
      apple9_key_nir_block(nir_blocks, b->nir, key);
      apple9_key_nir_block(nir_blocks, b->nir_exit, key);
      const struct nir_block *exit = b->nir_exit ? b->nir_exit : b->nir;
      for (unsigned s = 0; exit && s < 2; ++s) {
         util_dynarray_append(key, (uint32_t)(exit->successors[s] != NULL));
         apple9_key_nir_block(nir_blocks, exit->successors[s], key);
      }
   }
   _mesa_hash_table_destroy(nir_blocks, NULL);

   for (const struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      for (const struct agx_apple9_vir_instr *I =
              agx_apple9_block_first_instr(b);
           I; I = agx_apple9_instr_next(I)) {
         struct agx_apple9_vir_instr copy = *I;
         copy.branch_target =
            (void *)(uintptr_t)apple9_block_ref_map(I->branch_target, block_ids);
         copy.phi_block =
            (void *)(uintptr_t)apple9_block_ref_map(I->phi_block, block_ids);
         copy.phi_edge =
            (void *)(uintptr_t)apple9_instr_ref_map(I->phi_edge, instr_pos);
         copy.phi_srcs = NULL;
         copy.completion_consumer =
            (void *)(uintptr_t)apple9_instr_ref_map(I->completion_consumer, instr_pos);
         util_dynarray_append(key, apple9_block_ref_map(b, block_ids));
         util_dynarray_append_array(key, uint8_t, &copy, sizeof(copy));
         if (I->phi_srcs) {
            util_dynarray_append_array(key, uint8_t, I->phi_srcs,
                                       I->phi_src_count * sizeof(*I->phi_srcs));
         }
      }
   }
   _mesa_hash_table_destroy(block_ids, NULL);
   _mesa_hash_table_destroy(instr_pos, NULL);
   _mesa_hash_table_destroy(block_starts, NULL);
   return true;
}

static bool
encoding_tuple(const struct agx_apple9_vir_instr *instruction,
               const uint8_t *phys, unsigned *gprs, unsigned *count)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(instruction->encoding);
   if (!info->allocator_safe)
      return false;

   if (instruction->op == AGX_APPLE9_VIR_SHARED_ATOMIC) {
      unsigned components = instruction->memory_components;
      if (components < 1 || components > 2 ||
          instruction->nr_srcs != components + 1)
         return false;
      gprs[0] = phys[instruction->dest];
      gprs[1] = phys[instruction->src[0]];
      gprs[2] = phys[instruction->src[components]];
      *count = 3;
      return true;
   }

   if (instruction->encoding == AGX_APPLE9_ENC_SHARED_STORE) {
      unsigned gprs_n = agx_apple9_mem_data_gprs(instruction);
      if (gprs_n < 1 || gprs_n > 4 ||
          instruction->nr_srcs != gprs_n + 1)
         return false;
      gprs[0] = phys[instruction->src[0]];
      gprs[1] = phys[instruction->src[gprs_n]];
      *count = 2;
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_PRIVATE_STORE) {
      unsigned gprs_n = agx_apple9_mem_data_gprs(instruction);
      if ((gprs_n != 1 && gprs_n != 2 && gprs_n != 4) ||
          instruction->nr_srcs != gprs_n + 1)
         return false;
      gprs[0] = phys[instruction->src[0]];
      gprs[1] = phys[instruction->src[gprs_n]];
      *count = 2;
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
      const unsigned gprs_n = agx_apple9_mem_data_gprs(instruction);
      const bool indirect =
         instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
      if (gprs_n < 1 || gprs_n > 4 ||
          instruction->nr_srcs != gprs_n + (indirect ? 3 : 1))
         return false;

      /* Machine-table order is address then data base.  COLLECT has already
       * made the remaining vector lanes adjacent before allocation. */
      gprs[0] = phys[instruction->src[gprs_n]];
      gprs[1] = phys[instruction->src[0]];
      *count = indirect ? 4 : 2;
      if (indirect) {
         gprs[2] = phys[instruction->src[gprs_n + 1]];
         gprs[3] = phys[instruction->src[gprs_n + 2]];
      }
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
      const unsigned components = instruction->memory_components;
      const bool indirect =
         instruction->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT;
      if (components < 1 || components > 2 ||
          instruction->nr_srcs != components + (indirect ? 3 : 1))
         return false;

      /* The packet names only the address index and RMW data tuple. A
       * returning atomic's adjacent publication record names its result;
       * a discarded atomic has no destination at all. */
      gprs[0] = phys[instruction->src[components]];
      gprs[1] = phys[instruction->src[0]];
      *count = indirect ? 4 : 2;
      if (indirect) {
         gprs[2] = phys[instruction->src[components + 1]];
         gprs[3] = phys[instruction->src[components + 2]];
      }
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_IMAGE_STORE &&
       instruction->texture_uniform_bindless) {
      if (instruction->nr_srcs != 8)
         return false;
      for (unsigned i = 0; i < 8; ++i)
         gprs[i] = phys[instruction->src[i]];
      gprs[8] = 0;
      gprs[9] = 1;
      *count = 10;
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_MMA) {
      if (!agx_apple9_mma_flags_valid(instruction->immediate) ||
          instruction->dest == AGX_APPLE9_VREG_INVALID)
         return false;
      unsigned off[3], w[3];
      unsigned expected = agx_apple9_mma_src_layout(instruction->immediate, off, w);
      if (instruction->nr_srcs != expected)
         return false;
      gprs[0] = phys[instruction->dest];
      gprs[1] = phys[instruction->src[off[0]]];
      gprs[2] = phys[instruction->src[off[1]]];
      gprs[3] = w[2] ? phys[instruction->src[off[2]]] : 0;
      *count = 4;
      return true;
   }

   unsigned gpr_operand_count = 0;
   for (unsigned i = 0; i < info->operand_count; ++i)
      gpr_operand_count += !!(info->operands[i].files & AGX_APPLE9_FILE_GPR);

   const bool has_destination = instruction->dest != AGX_APPLE9_VREG_INVALID;
   if (gpr_operand_count != instruction->nr_srcs + has_destination)
      return false;

   unsigned base = 0;
   if (has_destination)
      gprs[base++] = phys[instruction->dest];
   for (unsigned i = 0; i < instruction->nr_srcs; ++i)
      gprs[base + i] = phys[instruction->src[i]];
   *count = gpr_operand_count;
   return true;
}

static uint32_t
apple9_publication_write_mask(const struct agx_apple9_vir_instr *ins)
{
   if (ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE) {
      switch (ins->encoding) {
      case AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS:
         /* Writes offsets 0, 1, 3 (offset 2 is reserved). */
         return 0x0bu;
      case AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS:
         /* Writes offsets 0, 1, 4..7 (offsets 2, 3 are reserved). */
         return 0xf3u;
      case AGX_APPLE9_ENC_TEXTURE_GRAD_VOLUME_PARAMS:
         /* Writes offsets 0..2, 4..9 (offset 3 and 10..15 are reserved). */
         return 0x3f7u;
      case AGX_APPLE9_ENC_TEXTURE_GRAD_VOLUME_MIN_LOD_PARAMS:
         /* Writes offsets 0..9 (offsets 10..15 are reserved). */
         return 0x3ffu;
      case AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_2D:
         /* Writes offsets 0..3, 6, 7 (offsets 4, 5 are reserved). */
         return 0xcfu;
      case AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_16:
         /* Writes offsets 0..3 (offsets 4..7 are reserved). */
         return 0x0fu;
      case AGX_APPLE9_ENC_BLOCK_STORE_PARAMS:
         /* Writes offsets 0..2 (offset 3 is reserved). */
         return 0x07u;
      default:
         break;
      }
   }
   unsigned width = agx_apple9_vir_dest_components(ins);
   return (width >= 32) ? UINT32_MAX : (width ? ((1u << width) - 1u) : 0u);
}

static uint32_t
apple9_texture_sample_read_mask(const struct agx_apple9_vir_instr *sample)
{
   if (sample->encoding == AGX_APPLE9_ENC_TEXTURE_GRAD) {
      const bool volume_block = sample->texture_dimension == 3 ||
                                sample->texture_dimension == 1 ||
                                sample->texture_dimension == 4;
      const bool extended = sample->texture_dimension ||
                            sample->texture_shadow ||
                            sample->texture_min_lod;
      if (volume_block)
         return sample->texture_min_lod ? 0x3ffu : 0x3f7u;
      return extended ? 0x0ffu : 0x0f3u;
   }
   if (sample->encoding == AGX_APPLE9_ENC_TEXTURE_LOD) {
      if (sample->texture_multisampled || sample->texture_dimension ||
          sample->texture_shadow || sample->texture_min_lod)
         return 0x0fu;
      return 0x0bu;
   }
   return 0x03u;
}

bool
agx_apple9_validate_vir_allocation(const struct agx_apple9_vir_program *program,
                                   const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   if (!agx_apple9_validate_mesh_control(program, reason))
      return false;
   if (program->phys == NULL || program->publication == NULL) {
      if (reason != NULL)
         *reason = "Apple9 virtual IR has not been allocated";
      return false;
   }

   if (!program->dependencies_finalized) {
      if (reason)
         *reason = "Apple9 dependency metadata is stale after an IR edit";
      return false;
   }

   if (program->index_dirty) {
      if (reason)
         *reason = "Apple9 VIR instruction index is stale after an IR edit";
      return false;
   }

   if (!agx_apple9_validate_value_files(program, reason))
      return false;

   unsigned publication_end[AGX_APPLE9_PUBLICATION_COUNT] = {0};
   bool publication_used[AGX_APPLE9_PUBLICATION_COUNT] = {false};
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (!agx_apple9_vir_publishes(ins))
         continue;
      if (ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE &&
          ins->encoding != AGX_APPLE9_ENC_PUBLICATION_PAIR) {
         /* Apple9 TEXTURE_SAMPLE consumes its parameter publication tuple on
          * read; sharing one published tuple across multiple samples or across
          * loop iterations leaves zeroed coordinates for subsequent reads on
          * T8132 silicon. Enforce the strict single-block, single-consumer IR
          * invariant (tracking exact written/read physical slot footprints per
          * word so partial overwrites and reserved tuple holes cannot mask
          * stale reads of surviving words). */
         unsigned base = program->phys[ins->dest];
         unsigned width = agx_apple9_vir_dest_components(ins);
         uint32_t live_mask = apple9_publication_write_mask(ins);
         struct agx_apple9_block *pub_blk = agx_apple9_instr_block(ins);
         unsigned loop_depth = 0;
         unsigned tex_consumers = 0;
         for (unsigned pos = i + 1; pos < program->instruction_count && live_mask;
              ++pos) {
            const struct agx_apple9_vir_instr *u = program->instructions[pos];
            if (u->op == AGX_APPLE9_VIR_LOOP_MASK_PUSH)
               loop_depth++;
            else if (u->op == AGX_APPLE9_VIR_LOOP_MASK_POP && loop_depth > 0)
               loop_depth--;

            if (u->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
               bool reads = false;
               uint32_t rmask = apple9_texture_sample_read_mask(u);
               if (program->physical) {
                  if (u->nr_srcs > 0 && u->src[0] < program->value_count &&
                      program->publication[u->src[0]]) {
                     unsigned u_base = program->phys[u->src[0]];
                     uint32_t m = rmask;
                     while (m) {
                        unsigned bit = u_bit_scan(&m);
                        unsigned slot = u_base + bit;
                        if (slot >= base && slot < base + width &&
                            slot - base < 32 &&
                            (live_mask & (1u << (slot - base)))) {
                           reads = true;
                           break;
                        }
                     }
                  }
                  for (unsigned s = 1; !reads && s < u->nr_srcs; ++s) {
                     uint32_t v = u->src[s];
                     if (v >= program->value_count || !program->publication[v])
                        continue;
                     if (s < 32 && !(rmask & (1u << s)))
                        continue;
                     unsigned slot = program->phys[v];
                     if (slot >= base && slot < base + width &&
                         slot - base < 32 &&
                         (live_mask & (1u << (slot - base)))) {
                        reads = true;
                        break;
                     }
                  }
               } else {
                  for (unsigned s = 0; s < u->nr_srcs; ++s) {
                     uint32_t v = u->src[s];
                     if (v >= program->value_count || !program->publication[v])
                        continue;
                     if (s < 32 && !(rmask & (1u << s)))
                        continue;
                     if (v >= ins->dest && v < ins->dest + width &&
                         v - ins->dest < 32 &&
                         (live_mask & (1u << (v - ins->dest)))) {
                        reads = true;
                        break;
                     }
                  }
               }
               if (reads) {
                  if (agx_apple9_instr_block(u) != pub_blk || loop_depth > 0) {
                     if (reason)
                        *reason = "Apple9 texture parameter publication must stay in its consumer's block and loop";
                     return false;
                  }
                  if (++tex_consumers > 1) {
                     if (reason)
                        *reason = "Apple9 texture parameter publication cannot be shared across samples";
                     return false;
                  }
               }
            }
            if (program->physical && agx_apple9_vir_publishes(u)) {
               unsigned uw = agx_apple9_vir_dest_components(u);
               uint32_t wmask = apple9_publication_write_mask(u);
               while (wmask) {
                  unsigned uc = u_bit_scan(&wmask);
                  if (uc >= uw)
                     break;
                  unsigned slot = program->phys[u->dest + uc];
                  if (slot >= base && slot < base + width && slot - base < 32)
                     live_mask &= ~(1u << (slot - base));
               }
            }
         }
      }
      unsigned end = agx_apple9_publication_end(program, i);
      for (unsigned c = 0; c < agx_apple9_vir_dest_components(ins); ++c) {
         unsigned index = program->phys[ins->dest + c];
         if (index >= agx_apple9_publication_bound(ins) ||
             (publication_used[index] && publication_end[index] >= i)) {
            if (reason)
               *reason = "Apple9 publication slots overlap or exceed the encoding";
            return false;
         }
         publication_used[index] = true;
         publication_end[index] = end;
      }
   }

   unsigned execution_mask_depth = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         program->instructions[i];
      if (instruction->encoding != AGX_APPLE9_ENC_PSEUDO &&
          instruction->encoding >= AGX_APPLE9_ENC_COUNT) {
         if (reason != NULL)
            *reason = "Apple9 instruction has an out-of-range encoding";
         return false;
      }
      const enum agx_apple9_dependency_layout dependency_layout =
         instruction->encoding == AGX_APPLE9_ENC_PSEUDO
            ? AGX_APPLE9_DEPENDENCY_NONE
            : agx_apple9_encoding_info(instruction->encoding)->dependency_layout;
      if (!agx_apple9_dependency_slot_valid(dependency_layout,
                                        instruction->scoreboard_slot)) {
         if (reason != NULL)
            *reason = "Apple9 instruction has an invalid scoreboard dependency";
         return false;
      }

      if (instruction->op == AGX_APPLE9_VIR_SHARED_STORE ||
          instruction->op == AGX_APPLE9_VIR_SHARED_ATOMIC ||
          instruction->op == AGX_APPLE9_VIR_SHARED_LOAD ||
          instruction->op == AGX_APPLE9_VIR_PRIVATE_LOAD ||
          instruction->op == AGX_APPLE9_VIR_PRIVATE_STORE) {
         struct agx_apple9_packed_instruction packed;
         if (!agx_apple9_pack_vir_instruction(instruction, program->phys, &packed, reason))
            return false;
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_STORE_UNIFORM) {
         struct agx_apple9_packed_instruction packed;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys, &packed, reason))
            return false;
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_SPILL_STORE ||
          instruction->op == AGX_APPLE9_VIR_SPILL_LOAD) {
         struct agx_apple9_packed_instruction packed;
         if (instruction->immediate >= program->scratch_size / 4 ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys, &packed, reason)) {
            if (reason) *reason = "Apple9 spill exceeds its frame or has an invalid dependency";
            return false;
         }
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         const unsigned components = instruction->memory_components;
         const unsigned gprs = agx_apple9_mem_data_gprs(instruction);
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
         const unsigned bits = instruction->memory_bits;
         const bool valid_type =
            (bits == 8 && components == 1) ||
            (bits == 16 && (components == 1 || components == 2 || components == 4)) ||
            (bits == 32 && components >= 1 && components <= 4);
         if (instruction->dest != AGX_APPLE9_VREG_INVALID || !valid_type ||
             instruction->nr_srcs != gprs + (indirect ? 3 : 1) ||
             instruction->immediate > UINT8_MAX) {
            if (reason != NULL)
               *reason = "Apple9 device store has an invalid VIR contract";
            return false;
         }

         if ((instruction->encoding != AGX_APPLE9_ENC_DEVICE_STORE &&
              !indirect)) {
            if (reason != NULL)
               *reason =
                  "Apple9 device store form disagrees with its machine encoding";
            return false;
         }

         const unsigned allocation_bits =
            bits == 8 ? 16 : (bits == 16 && components > 1) ? 32 : bits;
         const unsigned data = program->phys[instruction->src[0]];
         const unsigned index = program->phys[instruction->src[gprs]];
         bool adjacent = data + gprs <= AGX_APPLE9_GPR_COUNT;
         for (unsigned c = 1; c < gprs; ++c)
            adjacent &= program->phys[instruction->src[c]] == data + c;
         if (!agx_apple9_encoding_accepts_gpr(
                instruction->encoding, AGX_APPLE9_OPERAND_INDEX, index, 32) ||
             !agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                              AGX_APPLE9_OPERAND_STORE_DATA,
                                              data, allocation_bits) ||
             !adjacent) {
            if (reason != NULL)
               *reason =
                  adjacent
                     ? "Apple9 device store violates an encoding constraint"
                     : "Apple9 vector store source is not an adjacent GPR tuple";
            return false;
         }

         if (indirect) {
            unsigned address = program->phys[instruction->src[gprs + 1]];
            unsigned high = program->phys[instruction->src[gprs + 2]];
            if (!agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                                 AGX_APPLE9_OPERAND_SRC1,
                                                 address, 32) ||
                !agx_apple9_encoding_accepts_gpr(
                   instruction->encoding, AGX_APPLE9_OPERAND_SRC2, high, 32) ||
                high != address + 1) {
               if (reason)
                  *reason =
                     "Apple9 indirect store requires an adjacent address pair";
               return false;
            }
         }

         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         const unsigned components = instruction->memory_components;
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT;
         const bool compare_exchange =
            instruction->atomic_op == AGX_APPLE9_ATOMIC_CMPXCHG;
         if ((instruction->encoding != AGX_APPLE9_ENC_DEVICE_ATOMIC && !indirect) ||
             components != (compare_exchange ? 2u : 1u) ||
             instruction->nr_srcs != components + (indirect ? 3u : 1u)) {
            if (reason != NULL)
               *reason = "Apple9 device atomic has an invalid VIR contract";
            return false;
         }
         const unsigned data = program->phys[instruction->src[0]];
         bool adjacent = true;
         for (unsigned c = 1; c < components; ++c)
            adjacent &= program->phys[instruction->src[c]] == data + c;
         const unsigned index = program->phys[instruction->src[components]];
         const bool scoreboard_valid =
            instruction->atomic_discard
               ? instruction->producer_scoreboard_slot ==
                    AGX_APPLE9_SCOREBOARD_SLOT_NONE
               : instruction->producer_scoreboard_slot >=
                       AGX_APPLE9_SCOREBOARD_SLOT_1 &&
                    instruction->producer_scoreboard_slot <=
                       AGX_APPLE9_SCOREBOARD_SLOT_6;
         const bool valid =
            instruction->memory_bits == 32 &&
            (instruction->atomic_discard
                ? instruction->dest == AGX_APPLE9_VREG_INVALID
                : instruction->dest < program->value_count) &&
            instruction->immediate <= UINT8_MAX &&
            agx_apple9_atomic_op_valid(instruction->atomic_op) && adjacent &&
            agx_apple9_encoding_accepts_gpr(
               instruction->encoding, AGX_APPLE9_OPERAND_INDEX, index, 32) &&
            agx_apple9_encoding_accepts_gpr(
               instruction->encoding, AGX_APPLE9_OPERAND_ATOMIC_DATA, data,
               32) && scoreboard_valid;
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 device atomic has an invalid VIR contract";
            return false;
         }
         if (indirect) {
            unsigned address = program->phys[instruction->src[components + 1]];
            unsigned high = program->phys[instruction->src[components + 2]];
            if (!agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                                AGX_APPLE9_OPERAND_SRC1,
                                                address, 32) ||
                !agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                                AGX_APPLE9_OPERAND_SRC2,
                                                high, 32) ||
                high != address + 1) {
               if (reason)
                  *reason = "Apple9 indirect atomic requires an adjacent address pair";
               return false;
            }
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT &&
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->nr_srcs == 1 && instruction->immediate == 0 &&
            instruction->src[0] < program->value_count && i > 0 &&
            program->instructions[i - 1]->op ==
               AGX_APPLE9_VIR_DEVICE_ATOMIC &&
            !program->instructions[i - 1]->atomic_discard &&
            program->instructions[i - 1]->producer_scoreboard_slot >=
               AGX_APPLE9_SCOREBOARD_SLOT_1 &&
            program->instructions[i - 1]->producer_scoreboard_slot <=
               AGX_APPLE9_SCOREBOARD_SLOT_6 &&
            instruction->producer_scoreboard_slot ==
               program->instructions[i - 1]->producer_scoreboard_slot &&
            program->instructions[i - 1]->dest == instruction->src[0] &&
            program->phys[instruction->src[0]] < 64;
         if (!valid) {
            if (reason != NULL)
               *reason =
                  "Apple9 atomic result materializer has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_PHI) {
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_PSEUDO &&
            instruction->dest != AGX_APPLE9_VREG_INVALID &&
            instruction->dest_components == 1 && instruction->nr_srcs == 0 &&
            instruction->target == AGX_APPLE9_VREG_INVALID &&
            instruction->immediate == 0 && instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            instruction->scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            instruction->dest < program->value_count &&
            program->phys[instruction->dest] <= APPLE9_LAST_ALLOCATABLE_GPR;
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 PHI has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_PHI_SRC) {
         const struct agx_apple9_vir_instr *target_producer =
            agx_apple9_definition(program, instruction->target);
         const unsigned target = instruction->target < program->value_count
                                    ? program->phys[instruction->target]
                                    : AGX_APPLE9_PHYS_INVALID;
         const unsigned source =
            instruction->nr_srcs == 1 &&
                  instruction->src[0] < program->value_count
               ? program->phys[instruction->src[0]]
               : AGX_APPLE9_PHYS_INVALID;
         const unsigned tuple[] = {target, source, source};
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED &&
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->nr_srcs == 1 &&
            instruction->target < program->value_count &&
            instruction->src[0] < program->value_count &&
            target_producer != NULL &&
            target_producer->op == AGX_APPLE9_VIR_PHI &&
            instruction->phi_edge != NULL &&
            instruction->phi_edge->op == AGX_APPLE9_VIR_PHI_SRC &&
            agx_apple9_instr_block(instruction) ==
               agx_apple9_instr_block(instruction->phi_edge) &&
            instruction->immediate == 0 && instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                  ARRAY_SIZE(tuple), 32);
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 phi incoming edge has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (agx_apple9_vir_is_control_side_effect(instruction->op)) {
         if (instruction->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
             instruction->immediate == AGX_APPLE9_FENCE_COMPILER_BARRIER) {
            if (instruction->encoding != AGX_APPLE9_ENC_PSEUDO ||
                instruction->nr_srcs != 0 ||
                instruction->dest != AGX_APPLE9_VREG_INVALID ||
                instruction->memory_bits != 0 ||
                instruction->memory_components != 0 ||
                instruction->fence_slot_mask != 0 ||
                instruction->producer_scoreboard_slot !=
                   AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
                instruction->scoreboard_slot !=
                   AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
               if (reason != NULL)
                  *reason = "Apple9 compiler barrier has an invalid VIR contract";
               return false;
            }
            continue;
         }
         bool mesh = instruction->op == AGX_APPLE9_VIR_MESH_CONTROL ||
                     instruction->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR ||
                     instruction->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL;
         bool valid =
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            (instruction->producer_scoreboard_slot ==
                AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
             instruction->op == AGX_APPLE9_VIR_MESH_CONTROL) &&
            /* Compares, EXEC_MASK_PUSH, and U instructions may consume a
             * pending source or folded wait; the layout check above validated
             * their dependency field. */
            (instruction->scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
             instruction->op == AGX_APPLE9_VIR_PREDICATE_COMPARE ||
             instruction->op == AGX_APPLE9_VIR_EXEC_MASK_PUSH || mesh);

         switch (instruction->op) {
         case AGX_APPLE9_VIR_MESH_CONTROL:
         case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
         case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL: {
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               valid &= instruction->src[s] < program->value_count;
            struct agx_apple9_packed_instruction packed;
            valid = valid && agx_apple9_pack_vir_instruction(
               instruction, program->phys, &packed, NULL);
            break;
         }
         case AGX_APPLE9_VIR_PREDICATE_COMPARE: {
            const unsigned condition = instruction->immediate & 0xff;
            const unsigned predicate_bank =
               (instruction->immediate & AGX_APPLE9_PREDICATE_BANK_MASK) >>
               AGX_APPLE9_PREDICATE_BANK_SHIFT;
            const bool short_form =
               instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
            const bool extended_form =
               instruction->encoding ==
               AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
            const bool loop_form =
               instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP;
            const bool valid_condition =
               short_form
                  ? (condition == AGX_APPLE9_PREDICATE_FGT ||
                     condition == AGX_APPLE9_PREDICATE_FLT ||
                     condition == AGX_APPLE9_PREDICATE_UGT ||
                     condition == AGX_APPLE9_PREDICATE_ULT ||
                     condition == AGX_APPLE9_PREDICATE_IGT ||
                     condition == AGX_APPLE9_PREDICATE_ILT)
                  : (extended_form || loop_form) &&
                       (condition == AGX_APPLE9_PREDICATE_EXT_FEQ ||
                        condition == AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE ||
                        condition == AGX_APPLE9_PREDICATE_EXT_FLE_SEQUENCE ||
                        condition == AGX_APPLE9_PREDICATE_EXT_IEQ);
            valid &= (short_form || extended_form || loop_form) &&
                     instruction->nr_srcs + util_bitcount(instruction->alu_src_uniform_mask) == 2 &&
                     (instruction->immediate &
                      ~(0xffu | AGX_APPLE9_PREDICATE_INVERT |
                        AGX_APPLE9_PREDICATE_BANK_MASK)) == 0 &&
                     predicate_bank < AGX_APPLE9_PREDICATE_BANK_COUNT &&
                     valid_condition;
            break;
         }
         case AGX_APPLE9_VIR_EXEC_MASK_PUSH: {
            const unsigned selector = instruction->immediate & 0xffu;
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_PUSH &&
                     instruction->nr_srcs == 0 &&
                     (instruction->immediate &
                      ~(0xffu | AGX_APPLE9_EXEC_MASK_INVERT)) == 0 &&
                     selector >= AGX_APPLE9_EXEC_MASK_SOURCE(0) &&
                     selector <= AGX_APPLE9_EXEC_MASK_SOURCE(7) &&
                     ((selector - AGX_APPLE9_EXEC_MASK_SOURCE(0)) % 4) == 0;
            if (valid)
               ++execution_mask_depth;
            break;
         }
         case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_ELSE &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     execution_mask_depth > 0;
            break;
         case AGX_APPLE9_VIR_EXEC_MASK_POP:
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_POP &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     execution_mask_depth > 0;
            if (valid)
               --execution_mask_depth;
            break;
         case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
            valid &= instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_PUSH &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_LOOP_MASK_UPDATE: {
            const unsigned selector = instruction->immediate;
            const unsigned bank_selector =
               selector & ~AGX_APPLE9_LOOP_MASK_INVERT;
            valid &=
               instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_UPDATE &&
               instruction->nr_srcs == 0 &&
               (selector & ~(0x1fu | AGX_APPLE9_LOOP_MASK_INVERT)) == 0 &&
               bank_selector >= AGX_APPLE9_LOOP_MASK_PREDICATE(0) &&
               bank_selector <= AGX_APPLE9_LOOP_MASK_PREDICATE(
                                   AGX_APPLE9_PREDICATE_BANK_COUNT - 1) &&
               ((bank_selector - AGX_APPLE9_LOOP_MASK_PREDICATE(0)) % 4) == 0;
            break;
         }
         case AGX_APPLE9_VIR_WORKGROUP_BARRIER:
            valid &= instruction->encoding == AGX_APPLE9_ENC_WORKGROUP_BARRIER &&
                     instruction->nr_srcs == 0 &&
                     instruction->immediate <= AGX_APPLE9_WORKGROUP_BARRIER_DEVICE;
            break;
         case AGX_APPLE9_VIR_DEVICE_FENCE: {
            const bool uses_slot_mask =
               instruction->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE ||
               instruction->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_FS ||
               instruction->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP ||
               instruction->immediate == AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM;
            valid &= instruction->encoding == AGX_APPLE9_ENC_DEVICE_FENCE &&
                     instruction->nr_srcs == 0 &&
                     instruction->immediate <=
                        AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP_MEM &&
                     (uses_slot_mask
                         ? (instruction->fence_slot_mask & ~BITFIELD_MASK(6)) == 0
                         : instruction->fence_slot_mask == 0);
            break;
         }
         case AGX_APPLE9_VIR_WAIT:
            valid &= instruction->encoding == AGX_APPLE9_ENC_WAIT &&
                     instruction->nr_srcs == 0 && instruction->immediate != 0 &&
                     instruction->immediate <= BITFIELD_MASK(6);
            break;
         case AGX_APPLE9_VIR_HALT:
            valid &= instruction->encoding == AGX_APPLE9_ENC_HALT &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_ABI_RETURN:
            valid &= instruction->encoding == AGX_APPLE9_ENC_ABI_RETURN &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_OBJECT_CALL:
         case AGX_APPLE9_VIR_OBJECT_RETURN:
         case AGX_APPLE9_VIR_OBJECT_RESUME:
            valid &= instruction->encoding ==
               (instruction->op == AGX_APPLE9_VIR_OBJECT_CALL
                  ? AGX_APPLE9_ENC_OBJECT_CALL
                : instruction->op == AGX_APPLE9_VIR_OBJECT_RETURN
                  ? AGX_APPLE9_ENC_OBJECT_RETURN : AGX_APPLE9_ENC_OBJECT_RESUME) &&
               instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_LOOP_MASK_POP:
            valid &= instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_POP &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_JMP_EXEC_ANY:
         case AGX_APPLE9_VIR_JMP_EXEC_NONE:
            valid &= instruction->encoding ==
                        (instruction->op == AGX_APPLE9_VIR_JMP_EXEC_ANY
                            ? AGX_APPLE9_ENC_JMP_EXEC_ANY
                            : AGX_APPLE9_ENC_JMP_EXEC_NONE) &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     instruction->branch_target && instruction->branch_target->placed;
            break;
         case AGX_APPLE9_VIR_BREAK_MASK_UNWIND: {
            const unsigned scope_tag =
               AGX_APPLE9_BREAK_SCOPE_TAG(instruction->immediate);
            const unsigned predicate_bank =
               AGX_APPLE9_BREAK_PREDICATE_BANK(instruction->immediate);
            valid &=
               instruction->encoding == AGX_APPLE9_ENC_BREAK_MASK_UNWIND &&
               instruction->nr_srcs == 0 && scope_tag >= 2 && predicate_bank >= 1 &&
               predicate_bank < AGX_APPLE9_PREDICATE_BANK_COUNT;
            break;
         }
         default:
            valid = false;
            break;
         }

         if (valid) {
            unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
            unsigned tuple_count = 0;
            const unsigned bits =
               instruction->op == AGX_APPLE9_VIR_MESH_CONTROL ? 16 : 32;
            valid &=
               encoding_tuple(instruction, program->phys, tuple, &tuple_count) &&
               agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                     tuple_count, bits);
         }
         if (!valid) {
            if (reason != NULL)
               *reason =
                  "Apple9 control-flow instruction has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_ENTRY_REG) {
         if (instruction->encoding != AGX_APPLE9_ENC_PSEUDO ||
             instruction->nr_srcs != 0 ||
             instruction->dest == AGX_APPLE9_VREG_INVALID ||
             instruction->dest_components != 1 ||
             instruction->immediate >= AGX_APPLE9_GPR_COUNT) {
            if (reason != NULL)
               *reason = "Apple9 ENTRY_REG has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_COVERAGE || instruction->op == AGX_APPLE9_VIR_DEPTH_STORE ||
          instruction->op == AGX_APPLE9_VIR_VARY_STORE ||
          instruction->op == AGX_APPLE9_VIR_OBJECT_GRID_EXPORT ||
          instruction->op == AGX_APPLE9_VIR_TILE_ACCESS ||
          instruction->op == AGX_APPLE9_VIR_TILE_STORE ||
          instruction->op == AGX_APPLE9_VIR_TILE_FENCE ||
          instruction->op == AGX_APPLE9_VIR_BLOCK_IMAGE_STORE ||
          instruction->op == AGX_APPLE9_VIR_IMAGE_STORE) {
         unsigned gprs[AGX_APPLE9_MAX_ENCODING_OPERANDS], count;
         struct agx_apple9_packed_instruction packed;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID ||
             !encoding_tuple(instruction, program->phys, gprs, &count) ||
             !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, gprs,
                                                    count, 32) ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys,
                                              &packed, reason)) {
            if (reason)
               *reason = "Apple9 graphics output has an invalid contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_MMA) {
         if (!agx_apple9_mma_flags_valid(instruction->immediate) ||
             instruction->dest == AGX_APPLE9_VREG_INVALID) {
            if (reason != NULL)
               *reason = "Apple9 MMA has invalid flags or destination";
            return false;
         }
         enum agx_apple9_mma_type ty_dst =
            agx_apple9_mma_dst_type(instruction->immediate);
         unsigned dst_comps = agx_apple9_mma_type_gprs(ty_dst);
         enum agx_apple9_encoding expected_enc =
            ty_dst == AGX_APPLE9_MMA_F32 ? AGX_APPLE9_ENC_MMA_F32
                                         : AGX_APPLE9_ENC_MMA_F16;
         unsigned off[3], w[3];
         unsigned expected_srcs =
            agx_apple9_mma_src_layout(instruction->immediate, off, w);
         if (instruction->encoding != expected_enc ||
             agx_apple9_vir_dest_components(instruction) != dst_comps ||
             instruction->nr_srcs != expected_srcs ||
             dst_comps > program->value_count ||
             instruction->dest > program->value_count - dst_comps) {
            if (reason != NULL)
               *reason = "Apple9 MMA violates its VIR contract";
            return false;
         }
         unsigned dst_reg = program->phys[instruction->dest];
         if (dst_reg + dst_comps > AGX_APPLE9_GPR_COUNT ||
             (dst_comps == 2 &&
              program->phys[instruction->dest + 1] != dst_reg + 1)) {
            if (reason != NULL)
               *reason = "Apple9 MMA destination is not a valid contiguous GPR tuple";
            return false;
         }
         for (unsigned c = 0; c < dst_comps; ++c) {
            unsigned maximum = program->max_phys[instruction->dest + c];
            if (maximum != AGX_APPLE9_PHYS_INVALID &&
                program->phys[instruction->dest + c] > maximum) {
               if (reason != NULL)
                  *reason = "Apple9 MMA destination violates register-class constraint";
               return false;
            }
         }
         for (unsigned k = 0; k < 3; ++k) {
            if (!w[k])
               continue;
            unsigned base = program->phys[instruction->src[off[k]]];
            if (base + w[k] > AGX_APPLE9_GPR_COUNT ||
                (w[k] == 2 &&
                 program->phys[instruction->src[off[k] + 1]] != base + 1)) {
               if (reason != NULL)
                  *reason = "Apple9 MMA source is not a valid contiguous GPR tuple";
               return false;
            }
         }
         unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
         unsigned tuple_count = 0;
         struct agx_apple9_packed_instruction packed;
         if (!encoding_tuple(instruction, program->phys, tuple, &tuple_count) ||
             !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                    tuple_count, 32) ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys,
                                              &packed, reason)) {
            if (reason != NULL && *reason == NULL)
               *reason = "Apple9 MMA violates an encoding constraint";
            return false;
         }
         continue;
      }

      const unsigned components = agx_apple9_vir_dest_components(instruction);
      const unsigned max_components =
         instruction->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE ? 16 :
         instruction->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ? 5 : 4;
      if (components > max_components ||
          components > program->value_count ||
          instruction->dest > program->value_count - components ||
          (components > 1 && instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD &&
           instruction->op != AGX_APPLE9_VIR_SHARED_LOAD &&
           instruction->op != AGX_APPLE9_VIR_PRIVATE_LOAD &&
           instruction->op != AGX_APPLE9_VIR_IMUL_WIDE &&
           instruction->op != AGX_APPLE9_VIR_CUBE &&
           instruction->op != AGX_APPLE9_VIR_UNPACK_NORM &&
           instruction->op != AGX_APPLE9_VIR_ITER_FLAT &&
           instruction->op != AGX_APPLE9_VIR_COLLECT &&
           instruction->op != AGX_APPLE9_VIR_PUBLICATION_TUPLE &&
           instruction->op != AGX_APPLE9_VIR_TEXTURE_SAMPLE)) {
         if (reason != NULL)
            *reason = "Apple9 instruction has an invalid destination tuple";
         return false;
      }
      for (unsigned c = 1; c < components; ++c) {
         if (program->phys[instruction->dest + c] !=
             program->phys[instruction->dest] + c) {
            if (reason != NULL)
               *reason =
                  "Apple9 vector destination is not an adjacent GPR tuple";
            return false;
         }
      }
      for (unsigned c = 0; c < components; ++c) {
         const unsigned maximum = program->max_phys[instruction->dest + c];
         if (maximum != AGX_APPLE9_PHYS_INVALID &&
             program->phys[instruction->dest + c] > maximum) {
            if (reason != NULL)
               *reason =
                  "Apple9 allocation violates a value register-class constraint";
            return false;
         }
      }

      if (instruction->op == AGX_APPLE9_VIR_COLLECT) {
         if (components < 2 || instruction->nr_srcs != components ||
             instruction->encoding != AGX_APPLE9_ENC_PSEUDO ||
             instruction->memory_bits != 0 ||
             instruction->memory_components != 0 ||
             instruction->producer_scoreboard_slot !=
                AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
             instruction->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
            if (reason != NULL)
               *reason = "Apple9 COLLECT has an invalid VIR contract";
            return false;
         }

         bool identity = true;
         bool disjoint = true;
         for (unsigned d = 0; d < components; ++d) {
            identity &= program->phys[instruction->dest + d] ==
                        program->phys[instruction->src[d]];
            for (unsigned s = 0; s < components; ++s) {
               disjoint &= program->phys[instruction->dest + d] !=
                           program->phys[instruction->src[s]];
            }
         }
         if (!identity && !disjoint) {
            if (reason != NULL)
               *reason =
                  "Apple9 COLLECT allocation requires an unsupported overlapping shuffle";
            return false;
         }
         continue;
      }

      /* Inline uniform/immediate operands and texture samples have their own
       * file/range rules; validate them directly through the packer. */
      if (instruction->alu_src_uniform_mask || instruction->alu_src_immediate_mask ||
          instruction->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         struct agx_apple9_packed_instruction packed;
         if (!agx_apple9_pack_vir_instruction(instruction, program->phys,
                                             &packed, reason))
            return false;
         continue;
      }
      unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
      unsigned tuple_count = 0;
      if (!encoding_tuple(instruction, program->phys, tuple, &tuple_count) ||
          !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                 tuple_count, 32)) {
         if (agx_apple9_trace_enabled()) {
            fprintf(stderr,
                    "APPLE9_VALIDATE_FAIL i=%u op=%u enc=%u dst=r%u src=", i,
                    instruction->op, instruction->encoding,
                    program->phys[instruction->dest]);
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               fprintf(stderr, "%sr%u", s ? "," : "",
                       program->phys[instruction->src[s]]);
            fputc('\n', stderr);
         }
         if (reason != NULL)
            *reason = "Apple9 allocation violates an encoding constraint";
         return false;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
          instruction->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER) {
         if (instruction->nr_srcs != 2 ||
             program->phys[instruction->src[1]] != program->phys[instruction->src[0]] + 1 ||
             instruction->device_load_index_kind != AGX_APPLE9_DEVICE_LOAD_INDEX_NONE) {
            if (reason)
               *reason = "Apple9 pointer load must name an adjacent pair";
            return false;
         }
      } else if (instruction->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT;
         if (instruction->nr_srcs != (indirect ? 3 : 1) ||
             (indirect && program->phys[instruction->src[2]] !=
                             program->phys[instruction->src[1]] + 1)) {
            if (reason)
               *reason = "Apple9 device-load address must be an adjacent pair";
            return false;
         }
         if ((instruction->nr_srcs != 1 && instruction->nr_srcs != 3) ||
             (instruction->device_load_index_kind !=
                 AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR &&
              instruction->device_load_index_kind !=
                 AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR)) {
            if (reason != NULL)
               *reason = "Apple9 device load has an invalid index contract";
            return false;
         }

         const bool last_use = !(instruction->live_after_mask & 1u);
         const bool expected_last_use =
            instruction->device_load_index_kind ==
            AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;
         if (last_use != expected_last_use) {
            if (reason != NULL)
               *reason =
                  "Apple9 device-load index lifetime disagrees with VIR liveness";
            return false;
         }
      }
   }

   if (execution_mask_depth != 0) {
      if (reason != NULL)
         *reason = "Apple9 execution-mask stack is unbalanced";
      return false;
   }

   return true;
}

static unsigned
apple9_vir_definition_index(const struct agx_apple9_vir_program *program,
                            uint32_t value)
{
   const struct agx_apple9_vir_instr *definition = agx_apple9_definition(program, value);
   return definition ? agx_apple9_instr_position(definition) : UINT_MAX;
}

static bool
apple9_vir_is_async(const struct agx_apple9_vir_instr *ins)
{
   return agx_apple9_vir_is_pending_load(ins) ||
          ins->op == AGX_APPLE9_VIR_SPILL_STORE ||
          (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins->atomic_discard);
}

static enum agx_apple9_dependency_layout
apple9_instruction_dependency_layout(
   const struct agx_apple9_vir_instr *instruction)
{
   if (instruction->encoding == AGX_APPLE9_ENC_PSEUDO)
      return AGX_APPLE9_DEPENDENCY_NONE;

   return agx_apple9_encoding_info(instruction->encoding)->dependency_layout;
}

static bool
apple9_instruction_accepts_scoreboard_slot(
   const struct agx_apple9_vir_instr *instruction)
{
   return instruction->op != AGX_APPLE9_VIR_PHI_SRC &&
          apple9_instruction_dependency_layout(instruction) !=
          AGX_APPLE9_DEPENDENCY_NONE;
}

static bool
apple9_instruction_accepts_specific_scoreboard_slot(
   const struct agx_apple9_vir_instr *instruction, uint8_t slot)
{
   return apple9_instruction_accepts_scoreboard_slot(instruction) &&
          slot >= AGX_APPLE9_SCOREBOARD_SLOT_1 &&
          slot <= AGX_APPLE9_SCOREBOARD_SLOT_6;
}

static bool
apple9_schedule_spill_completions(struct agx_apple9_vir_program *program)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = program->instructions[i];
      if (store->op != AGX_APPLE9_VIR_SPILL_STORE)
         continue;
      store->completion_consumer = NULL;
      for (unsigned j = i + 1; j < program->instruction_count; ++j) {
         struct agx_apple9_vir_instr *next = program->instructions[j];
         if (agx_apple9_instr_block(next) != agx_apple9_instr_block(store) ||
             agx_apple9_vir_is_control_side_effect(next->op) ||
             next->op == AGX_APPLE9_VIR_PHI_SRC)
            break;
         if (apple9_instruction_accepts_scoreboard_slot(next)) {
            store->completion_consumer = next;
            break;
         }
      }
      if (!store->completion_consumer && !agx_apple9_materialize_spill_completion(program, store))
         return false;
   }
   return true;
}

static bool
apple9_materialize_load(struct agx_apple9_vir_program *program,
                        unsigned producer_index, const char **reason)
{
   return agx_apple9_materialize_pending_result_at(program, producer_index,
                                               producer_index + 1, reason);
}

static bool
apple9_materialize_unsupported_loads(struct agx_apple9_vir_program *program,
                                     const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *producer = program->instructions[i];
      unsigned existing_handoff = agx_apple9_vir_first_consumer(program, i);
      if (existing_handoff != UINT_MAX &&
          program->instructions[existing_handoff]->scoreboard_materialize)
         continue;
      if (producer->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         /* Separate the sample's fixed first-result handoff from subsequent
          * arithmetic and memory traffic, using ordinary SSA bit copies. */
         const unsigned components = agx_apple9_vir_dest_components(producer);
         if (!apple9_materialize_load(program, i, reason))
            return false;
         i += components;
         continue;
      }
      if (!agx_apple9_vir_is_pending_load(producer) ||
          producer->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_AUTO)
         continue;

      const unsigned handoff = agx_apple9_vir_first_consumer(program, i);
      const unsigned components = agx_apple9_vir_dest_components(producer);
      bool live_out = program->output != AGX_APPLE9_VREG_INVALID &&
                      program->output >= producer->dest &&
                      program->output - producer->dest < components;
      for (unsigned l = 0; l < program->live_out_count; ++l)
         live_out |= program->live_out[l] >= producer->dest &&
                     program->live_out[l] - producer->dest < components;

      if (handoff == UINT_MAX && !live_out) {
         for (unsigned c = 0; c < components; ++c)
            program->fixed_phys[producer->dest + c] = AGX_APPLE9_PHYS_INVALID;
         agx_apple9_vir_remove(program, producer);
         agx_apple9_vir_ensure_indexed(program);
         --i;
         continue;
      }

      if (handoff != UINT_MAX && apple9_instruction_accepts_scoreboard_slot(
                                    program->instructions[handoff]))
         continue;

      if (!apple9_materialize_load(program, i, reason))
         return false;

      /* The identity bridge occupies the next position. */
      i += 1;
   }

   return true;
}

/* Scoreboard legalization may split a returned tuple into scalar copies.
 * Reassemble adjacent tuples for store data and indirect load addresses. */
static bool
apple9_collect_memory_source_tuples(struct agx_apple9_vir_program *program,
                                    const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = program->instructions[i];
      if (store->op == AGX_APPLE9_VIR_MMA) {
         unsigned off[3], w[3];
         agx_apple9_mma_src_layout(store->immediate, off, w);
         for (unsigned k = 0; k < 3; ++k) {
            if (w[k] < 2 ||
                agx_apple9_vir_values_form_tuple(program, store->src + off[k], w[k]))
               continue;
            uint32_t sources[2] = {store->src[off[k]], store->src[off[k] + 1]};
            const unsigned old_count = program->instruction_count;
            const uint32_t tuple =
               agx_apple9_vir_emit_collect(program, sources, 2);
            if (tuple == AGX_APPLE9_VREG_INVALID) {
               if (reason != NULL)
                  *reason = "out of memory collecting Apple9 MMA source tuples";
               return false;
            }
            struct agx_apple9_vir_instr *collect =
               program->instructions[old_count];
            agx_apple9_vir_move_before(program, collect, store);
            agx_apple9_vir_ensure_indexed(program);
            store = program->instructions[i + 1];
            store->src[off[k]] = tuple;
            store->src[off[k] + 1] = tuple + 1;
            ++i;
         }
         continue;
      }
      const bool load_pointer =
         store->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER;
      const bool load_indirect =
         store->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT || load_pointer;
      const bool store_indirect =
         store->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
      for (unsigned tuple_index = 0; tuple_index < (store_indirect ? 2 : 1);
           ++tuple_index) {
         const bool image = store->op == AGX_APPLE9_VIR_IMAGE_STORE &&
                            !store->texture_uniform_bindless;
         const bool indirect = load_indirect || tuple_index == 1 || image;
         const unsigned data_gprs = agx_apple9_mem_data_gprs(store);
         const unsigned first = image ? 8 : load_pointer ? 0
                                : load_indirect ? 1
                                : tuple_index == 1
                                   ? data_gprs + 1
                                   : 0;
         const unsigned components = indirect ? 2 : data_gprs;
         if ((!indirect && store->op != AGX_APPLE9_VIR_DEVICE_STORE &&
              store->encoding != AGX_APPLE9_ENC_SHARED_STORE &&
              store->op != AGX_APPLE9_VIR_PRIVATE_STORE) ||
             components < 2 ||
             agx_apple9_vir_values_form_tuple(program, store->src + first,
                                          components))
            continue;

         uint32_t sources[4];
         for (unsigned c = 0; c < components; ++c)
            sources[c] = store->src[first + c];

         const unsigned old_count = program->instruction_count;
         const uint32_t tuple =
            agx_apple9_vir_emit_collect(program, sources, components);
         if (tuple == AGX_APPLE9_VREG_INVALID) {
            if (reason != NULL)
               *reason = "out of memory collecting Apple9 memory source tuples";
            return false;
         }

         struct agx_apple9_vir_instr *collect =
            program->instructions[old_count];
         agx_apple9_vir_move_before(program, collect, store);
         agx_apple9_vir_ensure_indexed(program);
         store = program->instructions[i + 1];
         for (unsigned c = 0; c < components; ++c)
            store->src[first + c] = tuple + c;

         ++i;
      }
   }

   return true;
}

static const uint8_t scalar_load_preference[] = {6, 1, 2, 3, 4, 5};

static bool
apple9_materialize_scoreboard_pressure(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   for (;;) {
      bool changed = false;
      unsigned handoffs[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1];
      unsigned owners[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1];
      for (unsigned slot = 0; slot < ARRAY_SIZE(handoffs); ++slot)
         handoffs[slot] = owners[slot] = UINT_MAX;

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         for (unsigned slot = 1; slot < ARRAY_SIZE(handoffs); ++slot) {
            if (handoffs[slot] == i)
               handoffs[slot] = owners[slot] = UINT_MAX;
         }
         const struct agx_apple9_vir_instr *current = program->instructions[i];
         if (!apple9_vir_is_async(current))
            continue;
         unsigned handoff = agx_apple9_vir_first_consumer(program, i);
         if (handoff == UINT_MAX)
            continue;
         bool automatic = current->producer_scoreboard_slot ==
                             AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         unsigned slot = current->producer_scoreboard_slot;
         unsigned shared = 0;
         for (unsigned candidate = 1; candidate < ARRAY_SIZE(handoffs); ++candidate) {
            if (handoffs[candidate] == handoff)
               shared = candidate;
         }
         /* The assignment pass diagnoses explicitly incompatible slots in
          * a shared handoff; that is not a pressure problem. */
         if (shared)
            continue;
         if (automatic) {
            slot = 0;
            for (unsigned p = 0; p < ARRAY_SIZE(scalar_load_preference); ++p) {
               unsigned candidate = scalar_load_preference[p];
               if (handoffs[candidate] == UINT_MAX) {
                  slot = candidate;
                  break;
               }
            }
         }
         if (!shared && slot >= 1 && slot < ARRAY_SIZE(handoffs) &&
             handoffs[slot] == UINT_MAX) {
            handoffs[slot] = handoff;
            owners[slot] = i;
            continue;
         }

         /* Simulate the allocator, including fixed texture-result slots.
          * A fixed producer can conflict before all six slots are occupied.
          * Retire the conflicting group (or the oldest group under full
          * pressure) through ordinary SSA copies before this producer. */
         unsigned victim = shared;
         if (!victim && slot >= 1 && slot < ARRAY_SIZE(handoffs))
            victim = slot;
         if (!victim) {
            for (unsigned candidate = 1; candidate < ARRAY_SIZE(handoffs); ++candidate) {
               if (owners[candidate] != UINT_MAX &&
                   (!victim || owners[candidate] < owners[victim]))
                  victim = candidate;
            }
         }
         if (!victim || handoffs[victim] == UINT_MAX) {
            if (reason)
               *reason = "Apple9 pending producer has no legal scoreboard slot";
            return false;
         }
         const unsigned target_handoff = handoffs[victim];
         const struct agx_apple9_vir_instr *pressure_instruction = current;
         struct util_dynarray values = UTIL_DYNARRAY_INIT;
         for (unsigned p = 0; p < i; ++p) {
            const struct agx_apple9_vir_instr *producer = program->instructions[p];
            if (apple9_vir_is_async(producer) &&
                agx_apple9_vir_first_consumer(program, p) == target_handoff)
               util_dynarray_append(&values, producer);
         }

         util_dynarray_foreach(&values, const struct agx_apple9_vir_instr *, value) {
            unsigned producer = UINT_MAX;
            unsigned insertion = UINT_MAX;
            for (unsigned p = 0; p < program->instruction_count; ++p) {
               const struct agx_apple9_vir_instr *candidate =
                  program->instructions[p];
               if (candidate == *value) {
                  producer = p;
               }
               if (candidate == pressure_instruction)
                  insertion = p;
            }
            if (producer == UINT_MAX || insertion == UINT_MAX ||
                !agx_apple9_materialize_pending_result_at(
                   program, producer, insertion, reason)) {
               util_dynarray_fini(&values);
               if (reason != NULL && *reason == NULL)
                  *reason = "could not materialize Apple9 scoreboard pressure";
               return false;
            }
         }
         util_dynarray_fini(&values);
         changed = true;
         break;
      }

      if (!changed)
         return true;
   }
}

static bool
apple9_is_logic_handoff_source(const struct agx_apple9_vir_program *program,
                               unsigned consumer_index, uint32_t source)
{
   const struct agx_apple9_vir_instr *producer =
      agx_apple9_definition(program, source);
   return producer != NULL &&
          (agx_apple9_vir_is_pending_load(producer) ||
           (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
            !producer->atomic_discard)) &&
          agx_apple9_vir_first_consumer(program,
                                apple9_vir_definition_index(program, source)) ==
             consumer_index;
}

static void
apple9_normalize_logic_handoff_source(struct agx_apple9_vir_program *program,
                                      unsigned consumer_index)
{
   struct agx_apple9_vir_instr *consumer =
      program->instructions[consumer_index];
   if ((consumer->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED &&
        consumer->encoding != AGX_APPLE9_ENC_LOGIC_EXPORT) ||
       consumer->nr_srcs != 2 ||
       (consumer->op != AGX_APPLE9_VIR_IAND &&
        consumer->op != AGX_APPLE9_VIR_IOR &&
        consumer->op != AGX_APPLE9_VIR_IXOR))
      return;

   const bool source_a_pending =
      apple9_is_logic_handoff_source(program, consumer_index, consumer->src[0]);
   const bool source_b_pending =
      apple9_is_logic_handoff_source(program, consumer_index, consumer->src[1]);

   /* Native Apple9 ilogic uses its one-hot pending-result mask for source A.
    * Source B is an ordinary GPR unless both operands belong to the same
    * pending group.  AND/OR/XOR are commutative, so normalize a lone pending
    * source into the architectural role before allocation derives source
    * liveness.  This is required when a pending load reuses a register whose
    * old durable value is still present: leaving it in source B silently reads
    * that stale GPR value instead of the pending result. */
   if (!source_a_pending && source_b_pending) {
      const uint32_t temporary = consumer->src[0];
      consumer->src[0] = consumer->src[1];
      consumer->src[1] = temporary;
      agx_apple9_invalidate_uses(program);
   }
}


bool
agx_apple9_assign_vir_scoreboard_slots(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   if (program == NULL)
      return false;
   agx_apple9_vir_ensure_indexed(program);
   agx_apple9_invalidate_uses(program);

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->scoreboard_assigned) {
         ins->scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         ins->scoreboard_assigned = false;
      }
      if (ins->producer_scoreboard_assigned) {
         ins->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         ins->producer_scoreboard_assigned = false;
      }
   }
   if (!apple9_schedule_spill_completions(program)) {
      if (reason) *reason = "out of memory scheduling Apple9 spill completion";
      return false;
   }

   if (!apple9_materialize_unsupported_loads(program, reason))
      return false;

   if (!apple9_materialize_scoreboard_pressure(program, reason))
      return false;

   uint8_t *handoff_slot =
      calloc(program->instruction_count, sizeof(*handoff_slot));
   bool occupied[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1] = {false};
   if (handoff_slot == NULL && program->instruction_count != 0) {
      if (reason != NULL)
         *reason = "out of memory allocating Apple9 scoreboard slots";
      return false;
   }

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *producer = program->instructions[i];

      /* Consume an input handoff before allocating a result produced by the
       * same instruction. EXP-M4-51 proves that a returning atomic may reuse
       * its pending input's slot, including under full six-slot pressure. */
      if (handoff_slot[i] != AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
         struct agx_apple9_vir_instr *consumer = program->instructions[i];
         const uint8_t slot = handoff_slot[i];
         if (consumer->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
             consumer->scoreboard_slot != slot) {
            if (reason != NULL)
               *reason =
                  "Apple9 consumer conflicts with allocated scoreboard slot";
            goto fail;
         }
         consumer->scoreboard_slot = slot;
         consumer->scoreboard_assigned = true;
         apple9_normalize_logic_handoff_source(program, i);
         occupied[slot] = false;
      }

      const bool is_load = agx_apple9_vir_is_pending_load(producer);
      const bool is_returning_atomic =
         producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
         !producer->atomic_discard;
      if (is_load || is_returning_atomic || producer->op == AGX_APPLE9_VIR_SPILL_STORE) {
         const bool automatic =
            producer->producer_scoreboard_slot ==
            AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         const unsigned handoff = agx_apple9_vir_first_consumer(program, i);

         if (handoff == UINT_MAX) {
            if (automatic || is_returning_atomic) {
               if (reason != NULL)
                  *reason =
                     "Apple9 pending result has no scoreboard-capable handoff";
               goto fail;
            }
         } else if (!apple9_instruction_accepts_scoreboard_slot(
                       program->instructions[handoff])) {
            if (automatic || is_returning_atomic) {
               if (reason != NULL)
                  *reason =
                     "Apple9 pending result requires an unsupported consumer form";
               goto fail;
            }
         } else {
            uint8_t slot = handoff_slot[handoff];
            if (slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
               if (automatic) {
                  for (unsigned p = 0; p < sizeof(scalar_load_preference) /
                                              sizeof(scalar_load_preference[0]);
                       ++p) {
                     const uint8_t candidate = scalar_load_preference[p];
                     if (!occupied[candidate] &&
                         apple9_instruction_accepts_specific_scoreboard_slot(
                            program->instructions[handoff], candidate)) {
                        slot = candidate;
                        break;
                     }
                  }
               } else {
                  slot = producer->producer_scoreboard_slot;
               }

               if (!apple9_instruction_accepts_specific_scoreboard_slot(
                      program->instructions[handoff], slot) ||
                   occupied[slot]) {
                  if (reason != NULL)
                     *reason = "Apple9 scoreboard has no compatible free slot";
                  goto fail;
               }
               occupied[slot] = true;
               handoff_slot[handoff] = slot;
            } else if (!automatic &&
                       producer->producer_scoreboard_slot != slot) {
               if (reason != NULL)
                  *reason =
                     "Apple9 multi-source scoreboard group uses different slots";
               goto fail;
            }

            producer->producer_scoreboard_assigned = automatic;
            if (automatic && (producer->op == AGX_APPLE9_VIR_TILE_LOAD ||
                              producer->op == AGX_APPLE9_VIR_ITER_FLAT ||
                              producer->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                              producer->op == AGX_APPLE9_VIR_SPILL_STORE ||
                              producer->op == AGX_APPLE9_VIR_PRIVATE_LOAD ||
                              producer->op == AGX_APPLE9_VIR_SHARED_LOAD ||
                              producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC ||
                              producer->op == AGX_APPLE9_VIR_GET_SR)) {
               producer->producer_scoreboard_slot = slot;
            } else if (automatic && is_load) {
               uint16_t token;
               if (!agx_apple9_scalar_load_token_for_slot(slot, &token)) {
                  if (reason != NULL)
                     *reason = "Apple9 scalar-load slot has no producer tag";
                  goto fail;
               }
               producer->producer_scoreboard_slot = slot;
               producer->device_load_raw_token = token;
            } else if (automatic) {
               producer->producer_scoreboard_slot = slot;
               assert(i + 1 < program->instruction_count);
               struct agx_apple9_vir_instr *publication =
                  program->instructions[i + 1];
               assert(publication->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
               publication->producer_scoreboard_slot = slot;
            }
         }
      }

   }

   free(handoff_slot);
   return apple9_collect_memory_source_tuples(program, reason);

fail:
   free(handoff_slot);
   return false;
}
