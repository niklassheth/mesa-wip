/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "apple9_ra_pcopy.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/macros.h"

static void PRINTFLIKE(4, 5)
set_err(struct apple9_ra_validate_error *err, unsigned block_index,
        unsigned instr_index, const char *fmt, ...)
{
   if (!err || err->failed)
      return;
   err->failed = true;
   err->block_index = block_index;
   err->instr_index = instr_index;

   va_list ap;
   va_start(ap, fmt);
   vsnprintf(err->message, sizeof(err->message), fmt, ap);
   va_end(ap);
}

static void
clear_err(struct apple9_ra_validate_error *err)
{
   if (!err)
      return;
   err->failed = false;
   err->block_index = UINT_MAX;
   err->instr_index = UINT_MAX;
   err->message[0] = '\0';
}

static inline unsigned
vir_dest_components(const struct agx_apple9_vir_instr *ins)
{
   if (ins->dest == AGX_APPLE9_VREG_INVALID)
      return 0;
   return ins->dest_components ? ins->dest_components : 1;
}

static inline bool
is_publication_val(const struct agx_apple9_vir_program *program, uint32_t val)
{
   return program->publication && val < program->publication_capacity &&
          program->publication[val];
}

static inline bool
is_phys_op(const struct agx_apple9_vir_program *program,
           const struct agx_apple9_vir_instr *ins)
{
   if (ins->op == AGX_APPLE9_VIR_SWAP ||
       ins->op == AGX_APPLE9_VIR_SPILL_LOAD ||
       ins->op == AGX_APPLE9_VIR_SPILL_STORE)
      return true;
   if ((ins->op == AGX_APPLE9_VIR_MOV || ins->op == AGX_APPLE9_VIR_IMM) &&
       (ins->target == APPLE9_RA_PHYS_OP_TAG || program->physical))
      return true;
   return false;
}

static inline bool
is_block_entry_prefix_op(const struct agx_apple9_vir_instr *ins)
{
   if (ins->op == AGX_APPLE9_VIR_ENTRY_REG)
      return true;
   if (ins->op == AGX_APPLE9_VIR_EXEC_MASK_ELSE && !ins->local_control)
      return true;
   return false;
}

/* Validate logical CFG symmetry, numbering, inactive blocks, loop_header flags,
 * and reachability from entry. Outputs dense array cfg_blocks[0..num_out-1]. */
static bool
validate_logical_cfg(const struct agx_apple9_vir_program *program,
                     struct agx_apple9_block ***blocks_out, unsigned *num_out,
                     struct apple9_ra_validate_error *err)
{
   *blocks_out = NULL;
   *num_out = 0;

   if (!program) {
      set_err(err, UINT_MAX, UINT_MAX, "NULL VIR program");
      return false;
   }

   if (!program->blocks) {
      if (program->instruction_count != 0) {
         set_err(err, UINT_MAX, UINT_MAX,
                 "program has %u instructions but no blocks",
                 program->instruction_count);
         return false;
      }
      return true;
   }

   unsigned active_count = 0;
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->cfg_active) {
         if (b->cfg_index != active_count) {
            set_err(err, b->index, UINT_MAX,
                    "active block %u has cfg_index %u, expected %u", b->index,
                    b->cfg_index, active_count);
            return false;
         }
         ++active_count;
      } else {
         if (b->cfg_succs[0] || b->cfg_succs[1] || b->num_cfg_preds > 0) {
            set_err(err, b->index, UINT_MAX,
                    "inactive block %u carries logical CFG edges", b->index);
            return false;
         }
         if (!b->nir) {
            for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
                 ins; ins = agx_apple9_instr_next(ins)) {
               if (ins->dest != AGX_APPLE9_VREG_INVALID ||
                   agx_apple9_vir_nr_srcs(ins) > 0) {
                  set_err(err, b->index, agx_apple9_instr_position(ins),
                          "inactive synthetic block %u has SSA dataflow", b->index);
                  return false;
               }
            }
         }
      }
   }

   if (active_count == 0) {
      set_err(err, UINT_MAX, UINT_MAX,
              "logical CFG has 0 active blocks (build_logical_cfg not run?)");
      return false;
   }

   if (program->num_cfg_blocks != active_count) {
      set_err(err, UINT_MAX, UINT_MAX,
              "program->num_cfg_blocks (%u) != active block count (%u)",
              program->num_cfg_blocks, active_count);
      return false;
   }

   struct agx_apple9_block **blocks = calloc(active_count, sizeof(*blocks));
   if (!blocks) {
      set_err(err, UINT_MAX, UINT_MAX, "OOM allocating CFG block table");
      return false;
   }

   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->cfg_active)
         blocks[b->cfg_index] = b;
   }

   /* Verify cfg_succs <-> cfg_preds symmetry and loop_header flags. */
   for (unsigned i = 0; i < active_count; ++i) {
      struct agx_apple9_block *b = blocks[i];

      for (unsigned s = 0; s < 2; ++s) {
         struct agx_apple9_block *succ = b->cfg_succs[s];
         if (!succ)
            continue;
         if (!succ->cfg_active || succ->cfg_index >= active_count ||
             blocks[succ->cfg_index] != succ) {
            set_err(err, b->index, UINT_MAX,
                    "block %u cfg_succs[%u] points to invalid/inactive block",
                    b->index, s);
            free(blocks);
            return false;
         }

         unsigned succ_count_in_b =
            (b->cfg_succs[0] == succ) + (b->cfg_succs[1] == succ);
         if (succ->num_cfg_preds > 0 && !succ->cfg_preds) {
            set_err(err, succ->index, UINT_MAX,
                    "block %u has num_cfg_preds=%u but NULL cfg_preds",
                    succ->index, succ->num_cfg_preds);
            free(blocks);
            return false;
         }
         unsigned b_count_in_succ = 0;
         for (unsigned p = 0; p < succ->num_cfg_preds; ++p) {
            if (succ->cfg_preds[p] == b)
               ++b_count_in_succ;
         }
         if (b_count_in_succ != succ_count_in_b) {
            set_err(err, b->index, UINT_MAX,
                    "CFG edge asymmetry: block %u -> block %u (%u succ refs vs %u pred refs)",
                    b->index, succ->index, succ_count_in_b, b_count_in_succ);
            free(blocks);
            return false;
         }

         if (succ->cfg_index <= b->cfg_index && !succ->loop_header) {
            set_err(err, succ->index, UINT_MAX,
                    "backedge block %u -> block %u targets block without loop_header flag",
                    b->index, succ->index);
            free(blocks);
            return false;
         }
      }

      if (b->num_cfg_preds > 0 && !b->cfg_preds) {
         set_err(err, b->index, UINT_MAX,
                 "block %u has num_cfg_preds=%u but NULL cfg_preds", b->index,
                 b->num_cfg_preds);
         free(blocks);
         return false;
      }

      for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
         struct agx_apple9_block *pred = b->cfg_preds[p];
         if (!pred || !pred->cfg_active || pred->cfg_index >= active_count ||
             blocks[pred->cfg_index] != pred) {
            set_err(err, b->index, UINT_MAX,
                    "block %u cfg_preds[%u] points to invalid/inactive block",
                    b->index, p);
            free(blocks);
            return false;
         }

         unsigned b_count_in_pred =
            (pred->cfg_succs[0] == b) + (pred->cfg_succs[1] == b);
         unsigned pred_count_in_b = 0;
         for (unsigned q = 0; q < b->num_cfg_preds; ++q) {
            if (b->cfg_preds[q] == pred)
               ++pred_count_in_b;
         }
         if (b_count_in_pred != pred_count_in_b) {
            set_err(err, b->index, UINT_MAX,
                    "CFG pred asymmetry: block %u lists pred %u (%u times) vs %u succ edges",
                    b->index, pred->index, pred_count_in_b, b_count_in_pred);
            free(blocks);
            return false;
         }
      }
   }

   /* Verify every active block is reachable from entry (cfg_index 0). */
   bool *visited = calloc(active_count, sizeof(bool));
   unsigned *stack = malloc(active_count * sizeof(unsigned));
   if (!visited || !stack) {
      free(visited);
      free(stack);
      free(blocks);
      set_err(err, UINT_MAX, UINT_MAX, "OOM checking CFG reachability");
      return false;
   }

   unsigned sp = 0;
   visited[0] = true;
   stack[sp++] = 0;
   while (sp > 0) {
      struct agx_apple9_block *b = blocks[stack[--sp]];
      for (unsigned s = 0; s < 2; ++s) {
         struct agx_apple9_block *succ = b->cfg_succs[s];
         if (succ && !visited[succ->cfg_index]) {
            visited[succ->cfg_index] = true;
            stack[sp++] = succ->cfg_index;
         }
      }
   }

   bool all_reachable = true;
   for (unsigned i = 0; i < active_count; ++i) {
      if (!visited[i]) {
         set_err(err, blocks[i]->index, UINT_MAX,
                 "active CFG block %u (cfg_index %u) is unreachable from entry",
                 blocks[i]->index, i);
         all_reachable = false;
         break;
      }
   }

   free(visited);
   free(stack);
   if (!all_reachable) {
      free(blocks);
      return false;
   }

   *blocks_out = blocks;
   *num_out = active_count;
   return true;
}

/* Cooper-Harvey-Kennedy iterative dominator tree on active CFG blocks. */
static unsigned
dom_intersect(const unsigned *idom, const unsigned *rpo_num, unsigned b1,
              unsigned b2)
{
   unsigned f1 = b1, f2 = b2;
   while (f1 != f2) {
      while (rpo_num[f1] > rpo_num[f2])
         f1 = idom[f1];
      while (rpo_num[f2] > rpo_num[f1])
         f2 = idom[f2];
   }
   return f1;
}

static bool
compute_dominators(struct agx_apple9_block *const *blocks, unsigned n,
                   unsigned *idom)
{
   /* Iterative DFS postorder -> reverse postorder. */
   unsigned *postorder = malloc(n * sizeof(unsigned));
   unsigned *rpo = malloc(n * sizeof(unsigned));
   unsigned *rpo_num = malloc(n * sizeof(unsigned));
   unsigned *dfs_block = malloc(n * sizeof(unsigned));
   uint8_t *dfs_edge = malloc(n * sizeof(uint8_t));
   bool *seen = calloc(n, sizeof(bool));
   if (!postorder || !rpo || !rpo_num || !dfs_block || !dfs_edge || !seen) {
      free(postorder);
      free(rpo);
      free(rpo_num);
      free(dfs_block);
      free(dfs_edge);
      free(seen);
      return false;
   }

   unsigned sp = 0, po_count = 0;
   seen[0] = true;
   dfs_block[sp] = 0;
   dfs_edge[sp] = 0;
   ++sp;

   while (sp > 0) {
      unsigned cur = dfs_block[sp - 1];
      uint8_t edge = dfs_edge[sp - 1];
      if (edge < 2) {
         dfs_edge[sp - 1] = edge + 1;
         struct agx_apple9_block *succ = blocks[cur]->cfg_succs[edge];
         if (succ && !seen[succ->cfg_index]) {
            seen[succ->cfg_index] = true;
            dfs_block[sp] = succ->cfg_index;
            dfs_edge[sp] = 0;
            ++sp;
         }
      } else {
         postorder[po_count++] = cur;
         --sp;
      }
   }

   assert(po_count == n);
   for (unsigned i = 0; i < n; ++i) {
      rpo[i] = postorder[n - 1 - i];
      rpo_num[rpo[i]] = i;
      idom[i] = UINT_MAX;
   }
   idom[0] = 0;

   bool changed = true;
   while (changed) {
      changed = false;
      for (unsigned i = 1; i < n; ++i) {
         unsigned b_idx = rpo[i];
         const struct agx_apple9_block *b = blocks[b_idx];
         unsigned new_idom = UINT_MAX;

         for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
            unsigned p_idx = b->cfg_preds[p]->cfg_index;
            if (idom[p_idx] == UINT_MAX)
               continue;
            if (new_idom == UINT_MAX)
               new_idom = p_idx;
            else
               new_idom = dom_intersect(idom, rpo_num, p_idx, new_idom);
         }

         assert(new_idom != UINT_MAX);
         if (idom[b_idx] != new_idom) {
            idom[b_idx] = new_idom;
            changed = true;
         }
      }
   }

   free(postorder);
   free(rpo);
   free(rpo_num);
   free(dfs_block);
   free(dfs_edge);
   free(seen);
   return true;
}

static bool
dominates(const unsigned *idom, unsigned dom, unsigned node)
{
   unsigned cur = node;
   while (cur != dom) {
      if (cur == 0)
         return false;
      cur = idom[cur];
   }
   return true;
}

bool
apple9_ra_validate_ssa(const struct agx_apple9_vir_program *program,
                       struct apple9_ra_validate_error *err)
{
   clear_err(err);

   struct agx_apple9_block **blocks = NULL;
   unsigned num_blocks = 0;
   if (!validate_logical_cfg(program, &blocks, &num_blocks, err))
      return false;

   if (num_blocks == 0)
      return true;

   unsigned vcount = MAX2(program->value_count, 1u);
   unsigned *def_block = malloc(vcount * sizeof(unsigned));
   int *def_pos = malloc(vcount * sizeof(int));
   const struct agx_apple9_vir_instr **def_instr =
      calloc(vcount, sizeof(*def_instr));
   uint8_t *def_state = calloc(vcount, sizeof(uint8_t)); /* 0=undef,1=instr,2=input */
   unsigned *idom = malloc(num_blocks * sizeof(unsigned));
   if (!def_block || !def_pos || !def_instr || !def_state || !idom) {
      set_err(err, UINT_MAX, UINT_MAX, "OOM in SSA validator");
      goto fail;
   }

   /* Pass 1: block form, PHI placement/arity, COLLECT/SPLIT arity, and
    * single-definition SSA recording. */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = blocks[bi];
      bool seen_non_phi = false;
      unsigned pos = 0;

      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins), ++pos) {
         if (ins->op == AGX_APPLE9_VIR_PHI) {
            if (seen_non_phi) {
               set_err(err, b->index, pos,
                       "VIR_PHI (v%u) placed after non-phi instruction in block %u",
                       ins->dest, b->index);
               goto fail;
            }
            if (b->num_cfg_preds == 0) {
               set_err(err, b->index, pos,
                       "VIR_PHI (v%u) in block %u with 0 logical predecessors",
                       ins->dest, b->index);
               goto fail;
            }
            unsigned nr = agx_apple9_vir_nr_srcs(ins);
            if (nr != b->num_cfg_preds) {
               set_err(err, b->index, pos,
                       "VIR_PHI (v%u) arity %u != block %u num_cfg_preds %u",
                       ins->dest, nr, b->index, b->num_cfg_preds);
               goto fail;
            }
            if (ins->dest == AGX_APPLE9_VREG_INVALID ||
                vir_dest_components(ins) != 1) {
               set_err(err, b->index, pos,
                       "VIR_PHI in block %u must define exactly 1 SSA value",
                       b->index);
               goto fail;
            }
         } else if (!is_block_entry_prefix_op(ins)) {
            seen_non_phi = true;
         }

         if (ins->op != AGX_APPLE9_VIR_PHI) {
            if (ins->phi_srcs != NULL) {
               set_err(err, b->index, pos,
                       "non-PHI instruction has sideband phi_srcs array");
               goto fail;
            }
            if (ins->nr_srcs > AGX_APPLE9_MAX_VIR_SRCS) {
               set_err(err, b->index, pos,
                       "instruction has nr_srcs=%u > max %u", ins->nr_srcs,
                       AGX_APPLE9_MAX_VIR_SRCS);
               goto fail;
            }
         }

         if (ins->op == AGX_APPLE9_VIR_COLLECT) {
            unsigned comps = vir_dest_components(ins);
            if (ins->dest == AGX_APPLE9_VREG_INVALID || comps < 2 ||
                comps > 4 || ins->nr_srcs != comps) {
               set_err(err, b->index, pos,
                       "VIR_COLLECT arity mismatch: dest_components=%u nr_srcs=%u",
                       comps, ins->nr_srcs);
               goto fail;
            }
         } else if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            unsigned comps = vir_dest_components(ins);
            if (ins->dest == AGX_APPLE9_VREG_INVALID || comps < 1 ||
                comps > 5 || ins->nr_srcs != 1) {
               set_err(err, b->index, pos,
                       "VIR_SPLIT arity mismatch: dest_components=%u nr_srcs=%u",
                       comps, ins->nr_srcs);
               goto fail;
            }
         }

         if (is_phys_op(program, ins))
            continue;

         unsigned comps = vir_dest_components(ins);
         if (comps > 0) {
            if (ins->dest >= program->value_count ||
                comps > program->value_count - ins->dest) {
               set_err(err, b->index, pos,
                       "dest v%u..v%u out of range (value_count=%u)", ins->dest,
                       ins->dest + comps - 1, program->value_count);
               goto fail;
            }
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->op == AGX_APPLE9_VIR_SPLIT &&
                   (ins->dead_dest_mask & (1u << c)))
                  continue;
               uint32_t v = ins->dest + c;
               if (def_state[v] != 0) {
                  set_err(err, b->index, pos,
                          "SSA violation: value v%u defined more than once", v);
                  goto fail;
               }
               def_state[v] = 1;
               def_block[v] = bi;
               def_pos[v] = (int)pos;
               def_instr[v] = ins;
            }
         }
      }
   }

   /* Fixed program inputs created by agx_apple9_vir_input() have no defining
    * instruction and dominate the entire CFG from entry (pos = -1). */
   if (program->fixed_phys) {
      for (unsigned v = 0; v < program->value_count; ++v) {
         if (def_state[v] == 0 &&
             program->fixed_phys[v] != AGX_APPLE9_PHYS_INVALID) {
            def_state[v] = 2;
            def_block[v] = 0;
            def_pos[v] = -1;
         }
      }
   }

   if (!compute_dominators(blocks, num_blocks, idom)) {
      set_err(err, UINT_MAX, UINT_MAX, "OOM computing dominators");
      goto fail;
   }

   /* Pass 2: validate source value indices, SPLIT source tuple arity, and
    * strict dominance of defs over uses. */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = blocks[bi];
      unsigned pos = 0;

      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins), ++pos) {
         if (is_phys_op(program, ins))
            continue;

         if (ins->op == AGX_APPLE9_VIR_PHI) {
            for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
               uint32_t v = agx_apple9_vir_src(ins, p);
               if (v >= program->value_count) {
                  set_err(err, b->index, pos,
                          "VIR_PHI (v%u) src[%u] = v%u out of range (value_count=%u)",
                          ins->dest, p, v, program->value_count);
                  goto fail;
               }
               if (def_state[v] == 0) {
                  set_err(err, b->index, pos,
                          "VIR_PHI (v%u) src[%u] uses undefined value v%u",
                          ins->dest, p, v);
                  goto fail;
               }
               unsigned pred_cfg = b->cfg_preds[p]->cfg_index;
               if (!dominates(idom, def_block[v], pred_cfg)) {
                  set_err(err, b->index, pos,
                          "dominance violation: VIR_PHI (v%u) src[%u]=v%u defined in block %u does not dominate pred block %u",
                          ins->dest, p, v, blocks[def_block[v]]->index,
                          b->cfg_preds[p]->index);
                  goto fail;
               }
            }
            continue;
         }

         unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(ins, s);
            if (v >= program->value_count) {
               set_err(err, b->index, pos,
                       "src[%u] = v%u out of range (value_count=%u)", s, v,
                       program->value_count);
               goto fail;
            }
            if (def_state[v] == 0) {
               set_err(err, b->index, pos, "src[%u] uses undefined value v%u", s,
                       v);
               goto fail;
            }
            if (def_block[v] == bi) {
               if (def_pos[v] >= (int)pos) {
                  set_err(err, b->index, pos,
                          "dominance violation: src[%u]=v%u used at pos %u before/at def pos %d in block %u",
                          s, v, pos, def_pos[v], b->index);
                  goto fail;
               }
            } else if (!dominates(idom, def_block[v], bi)) {
               set_err(err, b->index, pos,
                       "dominance violation: src[%u]=v%u defined in block %u does not dominate use in block %u",
                       s, v, blocks[def_block[v]]->index, b->index);
               goto fail;
            }
         }

         if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            uint32_t s0 = ins->src[0];
            unsigned comps = vir_dest_components(ins);
            const struct agx_apple9_vir_instr *src_def = def_instr[s0];
            if (!src_def || src_def->dest != s0 ||
                vir_dest_components(src_def) != comps) {
               set_err(err, b->index, pos,
                       "VIR_SPLIT arity mismatch: dest_components=%u but src v%u defines %u components",
                       comps, s0, src_def ? vir_dest_components(src_def) : 0u);
               goto fail;
            }
         }
      }
   }

   free(blocks);
   free(def_block);
   free(def_pos);
   free(def_instr);
   free(def_state);
   free(idom);
   return true;

fail:
   free(blocks);
   free(def_block);
   free(def_pos);
   free(def_instr);
   free(def_state);
   free(idom);
   return false;
}

/* --- Encoding / Apple9 constraint helpers (mirrors agx_apple9_ra.c) --- */

static unsigned
val_source_width(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if ((ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
        ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) &&
       source == 0)
      return 2;
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
        ins->op == AGX_APPLE9_VIR_PRIVATE_STORE) &&
       source == 0)
      return agx_apple9_mem_data_gprs(ins);
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
        ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) &&
       source == 0)
      return MAX2(ins->memory_components, 1u);
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT &&
       source == agx_apple9_mem_data_gprs(ins) + 1)
      return 2;
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT &&
       source == ins->memory_components + 1)
      return 2;
   if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT && source == 1) ||
       (ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER && source == 0))
      return 2;
   if (ins->op == AGX_APPLE9_VIR_MMA)
      return agx_apple9_mma_source_width(ins, source);
   if (ins->op == AGX_APPLE9_VIR_IMAGE_STORE && source == 8)
      return 2;
   return 1;
}

static const struct agx_apple9_operand_constraint val_table_index[2] = {
   {
      .role = AGX_APPLE9_OPERAND_SRC4,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
   {
      .role = AGX_APPLE9_OPERAND_SRC5,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
};

static const struct agx_apple9_operand_constraint *
val_source_constraint(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO ||
       ins->encoding >= AGX_APPLE9_ENC_COUNT)
      return NULL;

   if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
       source + ins->texture_bindless + ins->sampler_bindless >= ins->nr_srcs)
      return &val_table_index[ins->sampler_bindless &&
                              source + 1 == ins->nr_srcs];

   if (ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
       ins->op == AGX_APPLE9_VIR_PRIVATE_STORE) {
      return agx_apple9_find_operand(ins->encoding,
                                     source ? AGX_APPLE9_OPERAND_SRC1
                                            : AGX_APPLE9_OPERAND_SRC0);
   }

   if (ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
       ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) {
      unsigned data_gprs = ins->op == AGX_APPLE9_VIR_DEVICE_STORE
                              ? agx_apple9_mem_data_gprs(ins)
                              : MAX2(ins->memory_components, 1u);
      if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT ||
           ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT) &&
          source > data_gprs)
         return agx_apple9_find_operand(ins->encoding,
                                        source == data_gprs + 1
                                           ? AGX_APPLE9_OPERAND_SRC1
                                           : AGX_APPLE9_OPERAND_SRC2);
      enum agx_apple9_operand_role role =
         source                                     ? AGX_APPLE9_OPERAND_INDEX
         : ins->op == AGX_APPLE9_VIR_DEVICE_STORE   ? AGX_APPLE9_OPERAND_STORE_DATA
                                                    : AGX_APPLE9_OPERAND_ATOMIC_DATA;
      return agx_apple9_find_operand(ins->encoding, role);
   }

   if (ins->op == AGX_APPLE9_VIR_MMA)
      return agx_apple9_find_operand(ins->encoding,
                                     agx_apple9_mma_source_role(ins, source));

   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(ins->encoding);
   for (unsigned o = 0, s = 0; o < info->operand_count; ++o) {
      const struct agx_apple9_operand_constraint *c = &info->operands[o];
      if (!(c->files & AGX_APPLE9_FILE_GPR) ||
          agx_apple9_inline_source_file(ins, c->role) ||
          c->role == AGX_APPLE9_OPERAND_DEST)
         continue;
      if (s++ == source)
         return c;
   }
   return NULL;
}

static bool
val_async(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT)
      return true;
   if (ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
       ins->encoding < AGX_APPLE9_ENC_COUNT &&
       agx_apple9_encoding_info(ins->encoding)->fixed_producer_slot != 0)
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
val_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins)
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

static bool
val_allows_reuse_killed(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return true;
   if (ins->encoding >= AGX_APPLE9_ENC_COUNT)
      return false;
   return vir_dest_components(ins) == 1 &&
          (!val_async(ins) ||
           (val_reads_sources_at_issue(ins) &&
            ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC)) &&
          !apple9_ra_after_logical_end(ins) &&
          ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
          !agx_apple9_encoding_info(ins->encoding)->split_result;
}

static bool
val_allows_mem_operand(const struct agx_apple9_vir_instr *ins)
{
   return ins->op == AGX_APPLE9_VIR_MOV || ins->op == AGX_APPLE9_VIR_PHI;
}

static bool
get_val_loc(const struct agx_apple9_vir_program *program,
            const struct apple9_ra_assignment *assign, unsigned mem_slots,
            uint32_t v, struct apple9_ra_loc *out)
{
   if (v >= program->value_count)
      return false;

   if (assign && assign->is_mem && BITSET_TEST(assign->is_mem, v)) {
      if (!assign->mem_slot)
         return false;
      unsigned slot = assign->mem_slot[v];
      if (slot >= mem_slots || slot >= APPLE9_RA_MAX_SPILL_SLOTS)
         return false;
      *out = apple9_ra_loc_mem(slot);
      return true;
   }

   const uint8_t *gpr_map = (assign && assign->gpr) ? assign->gpr : program->phys;
   if (!gpr_map)
      return false;
   unsigned gpr = gpr_map[v];
   if (gpr >= AGX_APPLE9_GPR_COUNT)
      return false;
   *out = apple9_ra_loc_gpr(gpr);
   return true;
}

/* Compact 8-byte register/memory slot state.
 * Per-block cost: 8 * (96 + mem_slots) bytes = 768 B when no spills
 * (replaces agx_validate_ra.c's 256 KB/block regfile[16384]). */
enum slot_state {
   SLOT_UNDEF = 0,
   SLOT_SSA = 1,
   SLOT_CONST = 2,
};

struct reg_slot {
   uint32_t value;
   uint8_t state;
   uint8_t pad[3];
};
static_assert(sizeof(struct reg_slot) == 8, "compact 8-byte reg_slot");

static inline struct reg_slot
slot_undef(void)
{
   return (struct reg_slot){ .value = 0, .state = SLOT_UNDEF };
}

static inline struct reg_slot
slot_ssa(uint32_t v)
{
   return (struct reg_slot){ .value = v, .state = SLOT_SSA };
}

static inline struct reg_slot
slot_const(uint32_t imm)
{
   return (struct reg_slot){ .value = imm, .state = SLOT_CONST };
}

static inline bool
slot_equal(struct reg_slot a, struct reg_slot b)
{
   return a.state == b.state && (a.state == SLOT_UNDEF || a.value == b.value);
}

static inline struct reg_slot *
file_slot(struct reg_slot *base, struct apple9_ra_loc loc)
{
   if (loc.kind == APPLE9_RA_LOC_GPR)
      return &base[loc.value];
   return &base[AGX_APPLE9_GPR_COUNT + loc.value];
}

static bool
validate_encoding_constraints(const struct agx_apple9_vir_program *program,
                              const struct apple9_ra_assignment *assign,
                              unsigned mem_slots,
                              const struct agx_apple9_vir_instr *ins,
                              unsigned block_idx, unsigned pos,
                              struct apple9_ra_validate_error *err)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO) {
      /* Verify contiguous GPR tuple for VIR_COLLECT and valid GPRs for SPLIT. */
      if (ins->op == AGX_APPLE9_VIR_COLLECT && !is_publication_val(program, ins->dest)) {
         unsigned comps = vir_dest_components(ins);
         struct apple9_ra_loc base_loc;
         if (!get_val_loc(program, assign, mem_slots, ins->dest, &base_loc) ||
             base_loc.kind != APPLE9_RA_LOC_GPR) {
            set_err(err, block_idx, pos,
                    "VIR_COLLECT dest v%u must be assigned a GPR", ins->dest);
            return false;
         }
         if (base_loc.value + comps > AGX_APPLE9_GPR_COUNT) {
            set_err(err, block_idx, pos,
                    "VIR_COLLECT tuple r%u..r%u exceeds GPR bank (%u)",
                    base_loc.value, base_loc.value + comps - 1,
                    AGX_APPLE9_GPR_COUNT);
            return false;
         }
         for (unsigned c = 1; c < comps; ++c) {
            struct apple9_ra_loc lane;
            if (!get_val_loc(program, assign, mem_slots, ins->dest + c, &lane) ||
                lane.kind != APPLE9_RA_LOC_GPR ||
                lane.value != base_loc.value + c) {
               set_err(err, block_idx, pos,
                       "VIR_COLLECT dest v%u+%u is not contiguous (expected r%u)",
                       ins->dest, c, base_loc.value + c);
               return false;
            }
         }
      }
      return true;
   }

   if (ins->encoding >= AGX_APPLE9_ENC_COUNT) {
      set_err(err, block_idx, pos, "invalid machine encoding %u", ins->encoding);
      return false;
   }

   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(ins->encoding);

   /* Physical VIR_IMM / SPILL_LOAD / SPILL_STORE encoding checks. */
   if (is_phys_op(program, ins)) {
      if (ins->op == AGX_APPLE9_VIR_IMM) {
         const struct agx_apple9_operand_constraint *dc =
            agx_apple9_find_operand(ins->encoding, AGX_APPLE9_OPERAND_DEST);
         if (dc && (ins->dest < dc->min_index || ins->dest > dc->max_index)) {
            set_err(err, block_idx, pos,
                    "encoding %s dest r%u outside allowed range r%u..r%u",
                    info->name, ins->dest, dc->min_index, dc->max_index);
            return false;
         }
      }
      return true;
   }

   /* Check destination encoding constraints. */
   unsigned comps = vir_dest_components(ins);
   if (comps > 0 && !is_publication_val(program, ins->dest)) {
      struct apple9_ra_loc dloc;
      if (!get_val_loc(program, assign, mem_slots, ins->dest, &dloc)) {
         set_err(err, block_idx, pos,
                 "dest v%u has invalid/out-of-range assignment", ins->dest);
         return false;
      }
      if (dloc.kind == APPLE9_RA_LOC_MEM) {
         if (!val_allows_mem_operand(ins)) {
            set_err(err, block_idx, pos,
                    "encoding %s cannot write directly to MEM slot m%u",
                    info->name, dloc.value);
            return false;
         }
      } else {
         const struct agx_apple9_operand_constraint *dc =
            ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC
               ? agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT,
                                         AGX_APPLE9_OPERAND_SRC0)
               : agx_apple9_find_operand(ins->encoding, AGX_APPLE9_OPERAND_DEST);

         unsigned min_gpr = dc ? dc->min_index : 0u;
         unsigned max_gpr = dc ? dc->max_index : (AGX_APPLE9_GPR_COUNT - 1u);
         unsigned align_gprs =
            dc ? MAX2(dc->alignment_halves, 2u) / 2u : 1u;

         if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
            unsigned limit = (ins->texture_bindless || ins->sampler_bindless ||
                              ins->texture_uniform_bindless ||
                              ins->sampler_uniform_bindless)
                                ? AGX_APPLE9_GPR_COUNT
                                : 64u;
            max_gpr = limit - comps;
         }
         if ((ins->op == AGX_APPLE9_VIR_SHARED_LOAD ||
              ins->op == AGX_APPLE9_VIR_DEVICE_LOAD) &&
             comps > 1)
            max_gpr = AGX_APPLE9_GPR_COUNT - comps;
         if (ins->op == AGX_APPLE9_VIR_PRIVATE_LOAD && comps > 1)
            max_gpr = 64u - comps;

         if (dloc.value < min_gpr || dloc.value > max_gpr) {
            set_err(err, block_idx, pos,
                    "encoding %s dest v%u assigned r%u outside allowed r%u..r%u",
                    info->name, ins->dest, dloc.value, min_gpr, max_gpr);
            return false;
         }
         if (align_gprs > 1 && (dloc.value % align_gprs) != 0) {
            set_err(err, block_idx, pos,
                    "encoding %s dest v%u assigned r%u violates %u-GPR alignment",
                    info->name, ins->dest, dloc.value, align_gprs);
            return false;
         }
         if (dloc.value + comps > AGX_APPLE9_GPR_COUNT) {
            set_err(err, block_idx, pos,
                    "encoding %s dest tuple v%u (width %u) at r%u overflows GPR bank",
                    info->name, ins->dest, comps, dloc.value);
            return false;
         }
         for (unsigned c = 1; c < comps; ++c) {
            struct apple9_ra_loc lane;
            if (!get_val_loc(program, assign, mem_slots, ins->dest + c, &lane) ||
                lane.kind != APPLE9_RA_LOC_GPR ||
                lane.value != dloc.value + c) {
               set_err(err, block_idx, pos,
                       "encoding %s dest tuple v%u+%u is not contiguous (expected r%u)",
                       info->name, ins->dest, c, dloc.value + c);
               return false;
            }
         }
      }
   }

   /* Check source encoding constraints & contiguous source tuples. */
   for (unsigned s = 0; s < ins->nr_srcs; ) {
      unsigned width = val_source_width(ins, s);
      if (s + width > ins->nr_srcs) {
         set_err(err, block_idx, pos,
                 "encoding %s source tuple at src[%u] width %u exceeds nr_srcs %u",
                 info->name, s, width, ins->nr_srcs);
         return false;
      }

      if (is_publication_val(program, ins->src[s])) {
         s += width;
         continue;
      }

      struct apple9_ra_loc sloc;
      if (!get_val_loc(program, assign, mem_slots, ins->src[s], &sloc)) {
         set_err(err, block_idx, pos,
                 "encoding %s src[%u] v%u has invalid/out-of-range assignment",
                 info->name, s, ins->src[s]);
         return false;
      }

      if (sloc.kind == APPLE9_RA_LOC_MEM) {
         if (!val_allows_mem_operand(ins)) {
            set_err(err, block_idx, pos,
                    "encoding %s src[%u] v%u cannot read directly from MEM slot m%u",
                    info->name, s, ins->src[s], sloc.value);
            return false;
         }
         s += width;
         continue;
      }

      const struct agx_apple9_operand_constraint *sc =
         val_source_constraint(ins, s);
      unsigned min_gpr = sc ? sc->min_index : 0u;
      unsigned max_gpr = sc ? sc->max_index : (AGX_APPLE9_GPR_COUNT - 1u);
      unsigned align_gprs = sc ? MAX2(sc->alignment_halves, 2u) / 2u : 1u;

      if (s == 0) {
         if ((ins->encoding == AGX_APPLE9_ENC_SHARED_STORE ||
              ins->op == AGX_APPLE9_VIR_DEVICE_STORE) &&
             width > 1)
            max_gpr = AGX_APPLE9_GPR_COUNT - width;
         else if (ins->op == AGX_APPLE9_VIR_PRIVATE_STORE && width > 1)
            max_gpr = 64u - width;
         else if (ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC && width > 1)
            max_gpr = AGX_APPLE9_GPR_COUNT - width;
      }

      if (program->fixed_phys && ins->src[s] < program->value_count &&
          program->fixed_phys[ins->src[s]] != AGX_APPLE9_PHYS_INVALID &&
          !is_publication_val(program, ins->src[s])) {
         unsigned fixed = program->fixed_phys[ins->src[s]];
         if (sloc.value != fixed) {
            set_err(err, block_idx, pos,
                    "encoding %s src[%u] v%u assigned r%u, violates fixed r%u",
                    info->name, s, ins->src[s], sloc.value, fixed);
            return false;
         }
      }

      if (sloc.value < min_gpr || sloc.value > max_gpr) {
         set_err(err, block_idx, pos,
                 "encoding %s src[%u] v%u assigned r%u outside allowed r%u..r%u",
                 info->name, s, ins->src[s], sloc.value, min_gpr, max_gpr);
         return false;
      }
      if (align_gprs > 1 && (sloc.value % align_gprs) != 0) {
         set_err(err, block_idx, pos,
                 "encoding %s src[%u] v%u assigned r%u violates %u-GPR alignment",
                 info->name, s, ins->src[s], sloc.value, align_gprs);
         return false;
      }
      if (sloc.value + width > AGX_APPLE9_GPR_COUNT) {
         set_err(err, block_idx, pos,
                 "encoding %s src[%u] tuple (width %u) at r%u overflows GPR bank",
                 info->name, s, width, sloc.value);
         return false;
      }
      for (unsigned c = 1; c < width; ++c) {
         struct apple9_ra_loc lane;
         if (!get_val_loc(program, assign, mem_slots, ins->src[s + c], &lane) ||
             lane.kind != APPLE9_RA_LOC_GPR ||
             lane.value != sloc.value + c) {
            set_err(err, block_idx, pos,
                    "encoding %s src[%u+%u] tuple is not contiguous (expected r%u)",
                    info->name, s, c, sloc.value + c);
            return false;
         }
      }

      s += width;
   }

   return true;
}

/* Compute exact backward CFG SSA liveness (live_in / live_out per active block)
 * so post-RA validation can verify kill_mask, clobber, and reuse_killed. */
static bool
compute_liveness(const struct agx_apple9_vir_program *program,
                 struct agx_apple9_block *const *blocks, unsigned num_blocks,
                 BITSET_WORD *live_in, BITSET_WORD *live_out,
                 unsigned live_words)
{
   BITSET_WORD *tmp = calloc(live_words, sizeof(BITSET_WORD));
   if (!tmp)
      return false;

   /* Seed live-out values on logical CFG exit blocks identically to
    * apple9_ra_liveness(). */
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = blocks[bi];
      if (b->cfg_succs[0] || b->cfg_succs[1])
         continue;
      BITSET_WORD *out = &live_out[bi * live_words];
      bool has_abi_ret = false;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            has_abi_ret = true;
            break;
         }
      }
      if (program->output < program->value_count &&
          !is_publication_val(program, program->output))
         BITSET_SET(out, program->output);
      for (unsigned k = 0; k < program->live_out_count; ++k) {
         if (has_abi_ret &&
             program->live_out_reg[k] != AGX_APPLE9_PHYS_INVALID)
            continue;
         uint32_t v = program->live_out[k];
         if (v < program->value_count && !is_publication_val(program, v))
            BITSET_SET(out, v);
      }
   }

   bool changed = true;
   while (changed) {
      changed = false;
      for (unsigned bi = num_blocks; bi-- > 0; ) {
         struct agx_apple9_block *b = blocks[bi];
         BITSET_WORD *in = &live_in[bi * live_words];
         BITSET_WORD *out = &live_out[bi * live_words];

         memcpy(tmp, out, live_words * sizeof(BITSET_WORD));

         for (struct agx_apple9_vir_instr *ins = agx_apple9_block_last_instr(b);
              ins; ins = agx_apple9_instr_prev(ins)) {
            if (ins->op == AGX_APPLE9_VIR_PHI || is_phys_op(program, ins))
               continue;

            unsigned comps = vir_dest_components(ins);
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->op == AGX_APPLE9_VIR_SPLIT &&
                   (ins->dead_dest_mask & (1u << c)))
                  continue;
               uint32_t v = ins->dest + c;
               if (v < program->value_count)
                  BITSET_CLEAR(tmp, v);
            }

            if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
               for (unsigned k = 0; k < program->live_out_count; ++k) {
                  uint32_t v = program->live_out[k];
                  if (v < program->value_count && !is_publication_val(program, v))
                     BITSET_SET(tmp, v);
               }
            }

            if (ins->op == AGX_APPLE9_VIR_SPLIT) {
               uint32_t s0 = ins->src[0];
               for (unsigned c = 0; c < comps; ++c) {
                  if (ins->dead_dest_mask & (1u << c))
                     continue;
                  if (s0 + c < program->value_count &&
                      !is_publication_val(program, s0 + c))
                     BITSET_SET(tmp, s0 + c);
               }
               continue;
            }

            unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t v = agx_apple9_vir_src(ins, s);
               if (v < program->value_count && !is_publication_val(program, v))
                  BITSET_SET(tmp, v);
            }
         }

         if (memcmp(in, tmp, live_words * sizeof(BITSET_WORD)) != 0) {
            memcpy(in, tmp, live_words * sizeof(BITSET_WORD));
            changed = true;
         }

         /* Propagate to each predecessor's live_out (killing PHI dests and
          * adding this predecessor's PHI sources). */
         for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
            unsigned p_idx = b->cfg_preds[p]->cfg_index;
            BITSET_WORD *p_out = &live_out[p_idx * live_words];

            memcpy(tmp, in, live_words * sizeof(BITSET_WORD));
            for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
                 ins; ins = agx_apple9_instr_next(ins)) {
               if (is_block_entry_prefix_op(ins))
                  continue;
               if (ins->op != AGX_APPLE9_VIR_PHI)
                  break;
               if (ins->dest < program->value_count)
                  BITSET_CLEAR(tmp, ins->dest);
            }
            for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
                 ins; ins = agx_apple9_instr_next(ins)) {
               if (is_block_entry_prefix_op(ins))
                  continue;
               if (ins->op != AGX_APPLE9_VIR_PHI)
                  break;
               if (p < agx_apple9_vir_nr_srcs(ins)) {
                  uint32_t sv = agx_apple9_vir_src(ins, p);
                  if (sv < program->value_count &&
                      !is_publication_val(program, sv))
                     BITSET_SET(tmp, sv);
               }
            }

            for (unsigned w = 0; w < live_words; ++w) {
               BITSET_WORD merged = p_out[w] | tmp[w];
               if (merged != p_out[w]) {
                  p_out[w] = merged;
                  changed = true;
               }
            }
         }
      }
   }

   free(tmp);
   return true;
}

static bool
step_block_regfile(const struct agx_apple9_vir_program *program,
                   const struct apple9_ra_assignment *assign,
                   unsigned mem_slots, struct agx_apple9_block *b,
                   struct reg_slot *cur, bool check_errors,
                   const BITSET_WORD *block_live_out, unsigned live_words,
                   struct apple9_ra_validate_error *err)
{
   /* When check_errors is true, precompute per-instruction live_after sets by
    * walking the block once in reverse, or record instruction list. */
   unsigned nr_insns = 0;
   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b); ins;
        ins = agx_apple9_instr_next(ins))
      ++nr_insns;

   BITSET_WORD *live_after_table = NULL;
   if (check_errors && nr_insns > 0) {
      live_after_table =
         calloc((size_t)nr_insns * live_words, sizeof(BITSET_WORD));
      if (!live_after_table) {
         set_err(err, b->index, UINT_MAX, "OOM allocating live_after table");
         return false;
      }

      BITSET_WORD *live = malloc(live_words * sizeof(BITSET_WORD));
      if (!live) {
         free(live_after_table);
         set_err(err, b->index, UINT_MAX, "OOM allocating live bitset");
         return false;
      }
      memcpy(live, block_live_out, live_words * sizeof(BITSET_WORD));

      unsigned idx = nr_insns;
      for (struct agx_apple9_vir_instr *ins = agx_apple9_block_last_instr(b);
           ins; ins = agx_apple9_instr_prev(ins)) {
         --idx;
         memcpy(&live_after_table[(size_t)idx * live_words], live,
                live_words * sizeof(BITSET_WORD));

         if (ins->op == AGX_APPLE9_VIR_PHI || is_phys_op(program, ins))
            continue;

         unsigned comps = vir_dest_components(ins);
         for (unsigned c = 0; c < comps; ++c) {
            if (ins->op == AGX_APPLE9_VIR_SPLIT &&
                (ins->dead_dest_mask & (1u << c)))
               continue;
            uint32_t v = ins->dest + c;
            if (v < program->value_count)
               BITSET_CLEAR(live, v);
         }

         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (v < program->value_count && !is_publication_val(program, v))
                  BITSET_SET(live, v);
            }
         }

         if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            uint32_t s0 = ins->src[0];
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->dead_dest_mask & (1u << c))
                  continue;
               if (s0 + c < program->value_count &&
                   !is_publication_val(program, s0 + c))
                  BITSET_SET(live, s0 + c);
            }
            continue;
         }

         unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(ins, s);
            if (v < program->value_count && !is_publication_val(program, v))
               BITSET_SET(live, v);
         }
      }
      free(live);
   }

   unsigned pos = 0;
   for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b); ins;
        ins = agx_apple9_instr_next(ins), ++pos) {
      if (is_phys_op(program, ins)) {
         if (ins->op == AGX_APPLE9_VIR_MOV) {
            if (ins->dest >= AGX_APPLE9_GPR_COUNT ||
                ins->nr_srcs < 1 || ins->src[0] >= AGX_APPLE9_GPR_COUNT) {
               if (check_errors)
                  set_err(err, b->index, pos,
                          "physical VIR_MOV operand out of GPR range (r%u <- r%u)",
                          ins->dest, ins->nr_srcs ? ins->src[0] : UINT_MAX);
               free(live_after_table);
               return false;
            }
            cur[ins->dest] = cur[ins->src[0]];
         } else if (ins->op == AGX_APPLE9_VIR_SWAP) {
            if (ins->nr_srcs < 2 || ins->src[0] >= AGX_APPLE9_GPR_COUNT ||
                ins->src[1] >= AGX_APPLE9_GPR_COUNT) {
               if (check_errors)
                  set_err(err, b->index, pos,
                          "VIR_SWAP operand out of GPR range");
               free(live_after_table);
               return false;
            }
            struct reg_slot tmp = cur[ins->src[0]];
            cur[ins->src[0]] = cur[ins->src[1]];
            cur[ins->src[1]] = tmp;
         } else if (ins->op == AGX_APPLE9_VIR_IMM) {
            if (ins->dest >= AGX_APPLE9_GPR_COUNT) {
               if (check_errors)
                  set_err(err, b->index, pos,
                          "physical VIR_IMM dest r%u out of GPR range",
                          ins->dest);
               free(live_after_table);
               return false;
            }
            if (check_errors &&
                !validate_encoding_constraints(program, assign, mem_slots, ins,
                                               b->index, pos, err)) {
               free(live_after_table);
               return false;
            }
            cur[ins->dest] = slot_const(ins->immediate);
         } else if (ins->op == AGX_APPLE9_VIR_SPILL_LOAD) {
            if (ins->dest >= AGX_APPLE9_GPR_COUNT ||
                ins->immediate >= mem_slots) {
               if (check_errors)
                  set_err(err, b->index, pos,
                          "VIR_SPILL_LOAD out of bounds (r%u <- m%u, max_mem=%u)",
                          ins->dest, ins->immediate, mem_slots);
               free(live_after_table);
               return false;
            }
            cur[ins->dest] = cur[AGX_APPLE9_GPR_COUNT + ins->immediate];
         } else if (ins->op == AGX_APPLE9_VIR_SPILL_STORE) {
            if (ins->nr_srcs < 1 || ins->src[0] >= AGX_APPLE9_GPR_COUNT ||
                ins->immediate >= mem_slots) {
               if (check_errors)
                  set_err(err, b->index, pos,
                          "VIR_SPILL_STORE out of bounds (m%u <- r%u, max_mem=%u)",
                          ins->immediate,
                          ins->nr_srcs ? ins->src[0] : UINT_MAX, mem_slots);
               free(live_after_table);
               return false;
            }
            cur[AGX_APPLE9_GPR_COUNT + ins->immediate] = cur[ins->src[0]];
         }
         continue;
      }

      if (ins->op == AGX_APPLE9_VIR_PHI) {
         if (ins->dest != AGX_APPLE9_VREG_INVALID &&
             !is_publication_val(program, ins->dest)) {
            unsigned comps = MAX2((unsigned)ins->dest_components, 1u);
            for (unsigned c = 0; c < comps; ++c) {
               uint32_t dv = ins->dest + c;
               struct apple9_ra_loc dloc;
               if (!get_val_loc(program, assign, mem_slots, dv, &dloc)) {
                  if (check_errors)
                     set_err(err, b->index, pos,
                             "VIR_PHI dest v%u has invalid assignment", dv);
                  free(live_after_table);
                  return false;
               }
               *file_slot(cur, dloc) = slot_ssa(dv);
            }
         }
         continue;
      }

      const BITSET_WORD *live_after =
         live_after_table ? &live_after_table[(size_t)pos * live_words] : NULL;

      /* Check encoding constraints on final pass. */
      if (check_errors &&
          !validate_encoding_constraints(program, assign, mem_slots, ins,
                                         b->index, pos, err)) {
         free(live_after_table);
         return false;
      }

      /* Validate kill_mask bounds and accuracy against live_after. */
      if (check_errors) {
         unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
         unsigned valid_mask =
            nr_srcs >= AGX_APPLE9_MAX_VIR_SRCS
               ? BITFIELD_MASK(AGX_APPLE9_MAX_VIR_SRCS)
               : BITFIELD_MASK(nr_srcs);
         if ((ins->kill_mask & ~valid_mask) != 0) {
            set_err(err, b->index, pos,
                    "kill_mask 0x%x sets bits beyond nr_srcs=%u",
                    ins->kill_mask, nr_srcs);
            free(live_after_table);
            return false;
         }

         for (unsigned s = 0; s < nr_srcs && s < AGX_APPLE9_MAX_VIR_SRCS; ++s) {
            if (!agx_apple9_vir_src_killed(ins, s))
               continue;
            uint32_t v = ins->src[s];
            if (is_publication_val(program, v))
               continue;
            if (v < program->value_count && BITSET_TEST(live_after, v)) {
               set_err(err, b->index, pos,
                       "bad kill_mask: src[%u] v%u marked killed but is still live after instruction",
                       s, v);
               free(live_after_table);
               return false;
            }
         }
      }

      /* Validate all SSA source operands live in their assigned GPR/MEM slot. */
      if (check_errors) {
         if (ins->op == AGX_APPLE9_VIR_SPLIT) {
            unsigned comps = vir_dest_components(ins);
            uint32_t s0 = ins->src[0];
            for (unsigned c = 0; c < comps; ++c) {
               if (ins->dead_dest_mask & (1u << c))
                  continue;
               uint32_t sv = s0 + c;
               if (is_publication_val(program, sv))
                  continue;
               struct apple9_ra_loc sloc;
               if (!get_val_loc(program, assign, mem_slots, sv, &sloc)) {
                  set_err(err, b->index, pos,
                          "VIR_SPLIT src v%u+%u has invalid assignment", s0, c);
                  free(live_after_table);
                  return false;
               }
               struct reg_slot actual = *file_slot(cur, sloc);
               if (actual.state != SLOT_SSA || actual.value != sv) {
                  set_err(err, b->index, pos,
                          "clobbered value: VIR_SPLIT src v%u expected in %c%u, found %s (v%u)",
                          sv, sloc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r',
                          sloc.value,
                          actual.state == SLOT_SSA     ? "other SSA"
                          : actual.state == SLOT_CONST ? "const"
                                                       : "undef",
                          actual.value);
                  free(live_after_table);
                  return false;
               }
            }
         } else if (ins->op == AGX_APPLE9_VIR_MOV && ins->dest_components > 1 &&
                    ins->nr_srcs == 1) {
            uint32_t s0 = ins->src[0];
            for (unsigned c = 0; c < ins->dest_components; ++c) {
               uint32_t sv = s0 + c;
               if (is_publication_val(program, sv))
                  continue;
               struct apple9_ra_loc sloc;
               if (!get_val_loc(program, assign, mem_slots, sv, &sloc)) {
                  set_err(err, b->index, pos,
                          "VIR_MOV src v%u+%u has invalid assignment", s0, c);
                  free(live_after_table);
                  return false;
               }
               struct reg_slot actual = *file_slot(cur, sloc);
               if (actual.state != SLOT_SSA || actual.value != sv) {
                  set_err(err, b->index, pos,
                          "clobbered value: VIR_MOV src v%u expected in %c%u, found %s (v%u)",
                          sv, sloc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r',
                          sloc.value,
                          actual.state == SLOT_SSA     ? "other SSA"
                          : actual.state == SLOT_CONST ? "const"
                                                       : "undef",
                          actual.value);
                  free(live_after_table);
                  return false;
               }
            }
         } else {
            unsigned nr_srcs = agx_apple9_vir_nr_srcs(ins);
            for (unsigned s = 0; s < nr_srcs; ++s) {
               uint32_t sv = agx_apple9_vir_src(ins, s);
               if (is_publication_val(program, sv))
                  continue;
               struct apple9_ra_loc sloc;
               if (!get_val_loc(program, assign, mem_slots, sv, &sloc)) {
                  set_err(err, b->index, pos,
                          "src[%u] v%u has invalid assignment", s, sv);
                  free(live_after_table);
                  return false;
               }
               struct reg_slot actual = *file_slot(cur, sloc);
               if (actual.state != SLOT_SSA || actual.value != sv) {
                  set_err(err, b->index, pos,
                          "clobbered value: src[%u] v%u expected in %c%u, found %s (v%u)",
                          s, sv, sloc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r',
                          sloc.value,
                          actual.state == SLOT_SSA     ? "other SSA"
                          : actual.state == SLOT_CONST ? "const"
                                                       : "undef",
                          actual.value);
                  free(live_after_table);
                  return false;
               }
            }
         }

         if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
            for (unsigned k = 0; k < program->live_out_count; ++k) {
               uint32_t v = program->live_out[k];
               if (is_publication_val(program, v))
                  continue;
               struct apple9_ra_loc loc;
               if (!get_val_loc(program, assign, mem_slots, v, &loc)) {
                  set_err(err, b->index, pos,
                          "ABI_RETURN live_out[%u] v%u has invalid assignment",
                          k, v);
                  free(live_after_table);
                  return false;
               }
               struct reg_slot actual = *file_slot(cur, loc);
               if (actual.state != SLOT_SSA || actual.value != v) {
                  set_err(err, b->index, pos,
                          "clobbered ABI_RETURN live_out[%u] v%u in %c%u", k, v,
                          loc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r', loc.value);
                  free(live_after_table);
                  return false;
               }
            }
         }
      }

      /* Check AGX_APPLE9_OPERAND_CLOBBER legality and invalidate clobbered
       * input registers. */
      for (unsigned s = 0; s < ins->nr_srcs; ) {
         unsigned width = val_source_width(ins, s);
         const struct agx_apple9_operand_constraint *sc =
            val_source_constraint(ins, s);
         if (sc && (sc->flags & AGX_APPLE9_OPERAND_CLOBBER)) {
            for (unsigned c = 0; c < width && s + c < ins->nr_srcs; ++c) {
               uint32_t sv = ins->src[s + c];
               if (is_publication_val(program, sv))
                  continue;
               if (check_errors && BITSET_TEST(live_after, sv)) {
                  set_err(err, b->index, pos,
                          "clobbered operand src[%u] v%u is still live after instruction",
                          s + c, sv);
                  free(live_after_table);
                  return false;
               }
               struct apple9_ra_loc sloc;
               if (get_val_loc(program, assign, mem_slots, sv, &sloc))
                  *file_slot(cur, sloc) = slot_undef();
            }
         }
         s += width;
      }

      /* Check reuse_killed legality when destination GPR overlaps a source GPR. */
      unsigned comps = vir_dest_components(ins);
      if (check_errors && comps > 0 &&
          ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
          !is_publication_val(program, ins->dest)) {
         struct apple9_ra_loc dloc;
         if (get_val_loc(program, assign, mem_slots, ins->dest, &dloc) &&
             dloc.kind == APPLE9_RA_LOC_GPR) {
            bool reuse_ok = val_allows_reuse_killed(ins);
            for (unsigned s = 0; s < ins->nr_srcs; ++s) {
               uint32_t sv = ins->src[s];
               if (is_publication_val(program, sv))
                  continue;
               struct apple9_ra_loc sloc;
               if (!get_val_loc(program, assign, mem_slots, sv, &sloc) ||
                   sloc.kind != APPLE9_RA_LOC_GPR)
                  continue;
               if (sloc.value >= dloc.value &&
                   sloc.value < dloc.value + comps) {
                  if (!reuse_ok) {
                     set_err(err, b->index, pos,
                             "illegal reuse_killed: dest r%u overlaps src[%u] r%u on instruction that forbids source reuse",
                             dloc.value, s, sloc.value);
                     free(live_after_table);
                     return false;
                  }
                  if (BITSET_TEST(live_after, sv)) {
                     set_err(err, b->index, pos,
                             "clobbered live GPR: dest v%u (r%u) overwrites still-live src[%u] v%u",
                             ins->dest, dloc.value, s, sv);
                     free(live_after_table);
                     return false;
                  }
                  if ((ins->retain_mask >> s) & 1u) {
                     set_err(err, b->index, pos,
                             "illegal reuse_killed: dest r%u overwrites retained src[%u] v%u",
                             dloc.value, s, sv);
                     free(live_after_table);
                     return false;
                  }
               }
            }
         }
      }

      /* Record destination lanes into the register/memory file. */
      for (unsigned c = 0; c < comps; ++c) {
         if (ins->op == AGX_APPLE9_VIR_SPLIT &&
             (ins->dead_dest_mask & (1u << c)))
            continue;
         uint32_t dv = ins->dest + c;
         if (is_publication_val(program, dv))
            continue;
         struct apple9_ra_loc dloc;
         if (!get_val_loc(program, assign, mem_slots, dv, &dloc)) {
            if (check_errors)
               set_err(err, b->index, pos, "dest v%u has invalid assignment",
                       dv);
            free(live_after_table);
            return false;
         }
         *file_slot(cur, dloc) = slot_ssa(dv);
      }
   }

   free(live_after_table);
   return true;
}

bool
apple9_ra_validate_post_ra(const struct agx_apple9_vir_program *program,
                           const struct apple9_ra_assignment *assign,
                           struct apple9_ra_validate_error *err)
{
   clear_err(err);

   struct agx_apple9_block **blocks = NULL;
   unsigned num_blocks = 0;
   if (!validate_logical_cfg(program, &blocks, &num_blocks, err))
      return false;

   if (num_blocks == 0)
      return true;

   const uint8_t *gpr_map =
      (assign && assign->gpr) ? assign->gpr : program->phys;
   if (!gpr_map && !(assign && assign->is_mem)) {
      set_err(err, UINT_MAX, UINT_MAX,
              "no physical GPR assignment table (program->phys and assign->gpr are NULL)");
      free(blocks);
      return false;
   }

   /* Determine tight MEM slot bound. */
   unsigned mem_slots = assign ? assign->max_mem_slots : 0u;
   if (assign && assign->is_mem) {
      if (!assign->mem_slot) {
         set_err(err, UINT_MAX, UINT_MAX,
                 "assign->is_mem set but assign->mem_slot is NULL");
         free(blocks);
         return false;
      }
      for (unsigned v = 0; v < program->value_count; ++v) {
         if (!BITSET_TEST(assign->is_mem, v))
            continue;
         unsigned slot = assign->mem_slot[v];
         if (slot >= APPLE9_RA_MAX_SPILL_SLOTS) {
            set_err(err, UINT_MAX, UINT_MAX,
                    "value v%u assigned MEM slot m%u >= max %u", v, slot,
                    APPLE9_RA_MAX_SPILL_SLOTS);
            free(blocks);
            return false;
         }
         if (assign->max_mem_slots && slot >= assign->max_mem_slots) {
            set_err(err, UINT_MAX, UINT_MAX,
                    "value v%u assigned MEM slot m%u >= max_mem_slots %u", v,
                    slot, assign->max_mem_slots);
            free(blocks);
            return false;
         }
         mem_slots = MAX2(mem_slots, slot + 1u);
      }
   }

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (ins->op == AGX_APPLE9_VIR_SPILL_LOAD ||
             ins->op == AGX_APPLE9_VIR_SPILL_STORE) {
            if (ins->immediate >= APPLE9_RA_MAX_SPILL_SLOTS) {
               set_err(err, blocks[bi]->index, agx_apple9_instr_position(ins),
                       "spill instruction slot m%u >= max %u", ins->immediate,
                       APPLE9_RA_MAX_SPILL_SLOTS);
               free(blocks);
               return false;
            }
            if (assign && assign->max_mem_slots &&
                ins->immediate >= assign->max_mem_slots) {
               set_err(err, blocks[bi]->index, agx_apple9_instr_position(ins),
                       "spill instruction slot m%u >= max_mem_slots %u",
                       ins->immediate, assign->max_mem_slots);
               free(blocks);
               return false;
            }
            mem_slots = MAX2(mem_slots, ins->immediate + 1u);
         }
      }
   }

   /* Identify pre-defined fixed inputs (values with fixed_phys and no defining
    * instruction) so entry block regfile starts populated with them. */
   unsigned vcount = MAX2(program->value_count, 1u);
   bool *has_def = calloc(vcount, sizeof(bool));
   unsigned live_words = BITSET_WORDS(vcount);
   BITSET_WORD *live_in =
      calloc((size_t)num_blocks * live_words, sizeof(BITSET_WORD));
   BITSET_WORD *live_out =
      calloc((size_t)num_blocks * live_words, sizeof(BITSET_WORD));

   unsigned total_slots = AGX_APPLE9_GPR_COUNT + mem_slots;
   struct reg_slot *entry_files =
      calloc((size_t)num_blocks * total_slots, sizeof(struct reg_slot));
   struct reg_slot *exit_files =
      calloc((size_t)num_blocks * total_slots, sizeof(struct reg_slot));
   struct reg_slot *cur = calloc(total_slots, sizeof(struct reg_slot));
   bool *visited = calloc(num_blocks, sizeof(bool));

   if (!has_def || !live_in || !live_out || !entry_files || !exit_files ||
       !cur || !visited) {
      set_err(err, UINT_MAX, UINT_MAX, "OOM in post-RA validator");
      goto fail;
   }

   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      for (struct agx_apple9_vir_instr *ins =
              agx_apple9_block_first_instr(blocks[bi]);
           ins; ins = agx_apple9_instr_next(ins)) {
         if (is_phys_op(program, ins))
            continue;
         unsigned comps = vir_dest_components(ins);
         for (unsigned c = 0; c < comps; ++c) {
            if (ins->op == AGX_APPLE9_VIR_SPLIT &&
                (ins->dead_dest_mask & (1u << c)))
               continue;
            if (ins->dest + c < program->value_count)
               has_def[ins->dest + c] = true;
         }
      }
   }

   if (!compute_liveness(program, blocks, num_blocks, live_in, live_out,
                         live_words)) {
      set_err(err, UINT_MAX, UINT_MAX, "OOM computing liveness");
      goto fail;
   }

   /* Populate entry block initial register file with pre-defined inputs. */
   if (program->fixed_phys) {
      for (unsigned v = 0; v < program->value_count; ++v) {
         if (has_def[v] || is_publication_val(program, v))
            continue;
         if (program->fixed_phys[v] == AGX_APPLE9_PHYS_INVALID)
            continue;
         struct apple9_ra_loc loc;
         if (!get_val_loc(program, assign, mem_slots, v, &loc)) {
            set_err(err, blocks[0]->index, UINT_MAX,
                    "input v%u has invalid assignment", v);
            goto fail;
         }
         if (loc.kind == APPLE9_RA_LOC_GPR &&
             loc.value != program->fixed_phys[v]) {
            set_err(err, blocks[0]->index, UINT_MAX,
                    "input v%u assigned r%u, violates fixed_phys r%u", v,
                    loc.value, program->fixed_phys[v]);
            goto fail;
         }
         *file_slot(&entry_files[0], loc) = slot_ssa(v);
      }
   }

   /* Phase 1: Iterative greatest-fixed-point meet of compact GPR[96]+MEM[n]
    * register files across all CFG edges (including loops). */
   bool changed = true;
   unsigned iter_guard = 0;
   const unsigned max_iters = num_blocks * 4u + 16u;
   while (changed) {
      changed = false;
      ++iter_guard;
      assert(iter_guard <= max_iters);

      for (unsigned bi = 0; bi < num_blocks; ++bi) {
         struct agx_apple9_block *b = blocks[bi];
         struct reg_slot *entry = &entry_files[(size_t)bi * total_slots];
         struct reg_slot *exit = &exit_files[(size_t)bi * total_slots];

         if (bi > 0) {
            bool have_pred = false;
            for (unsigned p = 0; p < b->num_cfg_preds; ++p) {
               unsigned p_idx = b->cfg_preds[p]->cfg_index;
               if (!visited[p_idx])
                  continue;
               const struct reg_slot *p_exit =
                  &exit_files[(size_t)p_idx * total_slots];
               if (!have_pred) {
                  memcpy(entry, p_exit, total_slots * sizeof(struct reg_slot));
                  have_pred = true;
               } else {
                  for (unsigned r = 0; r < total_slots; ++r) {
                     if (!slot_equal(entry[r], p_exit[r]))
                        entry[r] = slot_undef();
                  }
               }
            }
            if (!have_pred)
               continue;
         }

         memcpy(cur, entry, total_slots * sizeof(struct reg_slot));
         if (!step_block_regfile(program, assign, mem_slots, b, cur, false,
                                 NULL, 0, err)) {
            /* Re-run with check_errors=true to produce crisp diagnostic. */
            memcpy(cur, entry, total_slots * sizeof(struct reg_slot));
            step_block_regfile(program, assign, mem_slots, b, cur, true,
                               &live_out[(size_t)bi * live_words], live_words,
                               err);
            goto fail;
         }

         if (!visited[bi] ||
             memcmp(exit, cur, total_slots * sizeof(struct reg_slot)) != 0) {
            memcpy(exit, cur, total_slots * sizeof(struct reg_slot));
            visited[bi] = true;
            changed = true;
         }
      }
   }

   /* Phase 2: Full verification pass at the converged fixed point. */
   const bool phis_lowered = assign && assign->phis_lowered;
   for (unsigned bi = 0; bi < num_blocks; ++bi) {
      struct agx_apple9_block *b = blocks[bi];
      const struct reg_slot *entry = &entry_files[(size_t)bi * total_slots];
      memcpy(cur, entry, total_slots * sizeof(struct reg_slot));

      if (!step_block_regfile(program, assign, mem_slots, b, cur, true,
                              &live_out[(size_t)bi * live_words], live_words,
                              err))
         goto fail;

      /* Verify shader exit outputs (program->output and non-ABI-return
       * live_out[]) are intact in their assigned slots at logical exits. */
      if (!b->cfg_succs[0] && !b->cfg_succs[1]) {
         bool has_abi_ret = false;
         for (struct agx_apple9_vir_instr *ins = agx_apple9_block_first_instr(b);
              ins; ins = agx_apple9_instr_next(ins)) {
            if (ins->op == AGX_APPLE9_VIR_ABI_RETURN) {
               has_abi_ret = true;
               break;
            }
         }
         if (program->output < program->value_count &&
             !is_publication_val(program, program->output)) {
            struct apple9_ra_loc loc;
            if (!get_val_loc(program, assign, mem_slots, program->output,
                             &loc)) {
               set_err(err, b->index, UINT_MAX,
                       "program.output v%u has invalid assignment",
                       program->output);
               goto fail;
            }
            struct reg_slot actual = *file_slot(cur, loc);
            if (actual.state != SLOT_SSA || actual.value != program->output) {
               set_err(err, b->index, UINT_MAX,
                       "clobbered program.output v%u in %c%u (found %s v%u)",
                       program->output,
                       loc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r', loc.value,
                       actual.state == SLOT_SSA     ? "other SSA"
                       : actual.state == SLOT_CONST ? "const"
                                                    : "undef",
                       actual.value);
               goto fail;
            }
         }
         for (unsigned k = 0; k < program->live_out_count; ++k) {
            if (has_abi_ret &&
                program->live_out_reg[k] != AGX_APPLE9_PHYS_INVALID)
               continue;
            uint32_t v = program->live_out[k];
            if (v >= program->value_count || is_publication_val(program, v))
               continue;
            struct apple9_ra_loc loc;
            if (!get_val_loc(program, assign, mem_slots, v, &loc)) {
               set_err(err, b->index, UINT_MAX,
                       "exit live_out[%u] v%u has invalid assignment", k, v);
               goto fail;
            }
            struct reg_slot actual = *file_slot(cur, loc);
            if (actual.state != SLOT_SSA || actual.value != v) {
               set_err(err, b->index, UINT_MAX,
                       "clobbered exit live_out[%u] v%u in %c%u", k, v,
                       loc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r', loc.value);
               goto fail;
            }
         }
      }

      /* Validate outgoing CFG edges against successors' VIR_PHI sources. */
      for (unsigned s = 0; s < 2; ++s) {
         struct agx_apple9_block *succ = b->cfg_succs[s];
         if (!succ)
            continue;
         unsigned pred_idx = agx_apple9_block_pred_index(succ, b);
         assert(pred_idx != UINT_MAX);

         unsigned phi_pos = 0;
         for (struct agx_apple9_vir_instr *phi =
                 agx_apple9_block_first_instr(succ);
              phi; phi = agx_apple9_instr_next(phi), ++phi_pos) {
            if (is_block_entry_prefix_op(phi))
               continue;
            if (phi->op != AGX_APPLE9_VIR_PHI)
               break;
            if (pred_idx >= agx_apple9_vir_nr_srcs(phi)) {
               set_err(err, succ->index, phi_pos,
                       "VIR_PHI (v%u) arity %u < pred_idx %u for pred block %u",
                       phi->dest, agx_apple9_vir_nr_srcs(phi), pred_idx,
                       b->index);
               goto fail;
            }

            uint32_t sv = agx_apple9_vir_src(phi, pred_idx);
            if (sv == AGX_APPLE9_VREG_INVALID || is_publication_val(program, sv))
               continue;

            unsigned comps = MAX2((unsigned)phi->dest_components, 1u);
            for (unsigned c = 0; c < comps; ++c) {
               struct apple9_ra_loc check_loc;
               uint32_t loc_owner = (phis_lowered ? phi->dest : sv) + c;
               uint32_t expect_val = sv + c;
               if (!get_val_loc(program, assign, mem_slots, loc_owner,
                                &check_loc)) {
                  set_err(err, b->index, UINT_MAX,
                          "PHI edge block %u -> block %u value v%u has invalid assignment",
                          b->index, succ->index, loc_owner);
                  goto fail;
               }
               struct reg_slot actual = *file_slot(cur, check_loc);
               if (actual.state != SLOT_SSA || actual.value != expect_val) {
                  set_err(err, b->index, UINT_MAX,
                          "clobbered PHI edge value on block %u -> block %u: expected v%u in %c%u, found %s (v%u)",
                          b->index, succ->index, expect_val,
                          check_loc.kind == APPLE9_RA_LOC_MEM ? 'm' : 'r',
                          check_loc.value,
                          actual.state == SLOT_SSA     ? "other SSA"
                          : actual.state == SLOT_CONST ? "const"
                                                       : "undef",
                          actual.value);
                  goto fail;
               }
            }
         }
      }
   }

   free(blocks);
   free(has_def);
   free(live_in);
   free(live_out);
   free(entry_files);
   free(exit_files);
   free(cur);
   free(visited);
   return true;

fail:
   free(blocks);
   free(has_def);
   free(live_in);
   free(live_out);
   free(entry_files);
   free(exit_files);
   free(cur);
   free(visited);
   return false;
}
