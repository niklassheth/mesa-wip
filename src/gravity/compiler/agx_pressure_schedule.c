/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include "util/dag.h"
#include "util/sparse_bitset.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

struct sched_ctx {
   /* Dependency graph */
   struct dag *dag;

   /* Original forward instruction order of DAG nodes */
   struct util_dynarray orig_order;

   /* Live set */
   struct u_sparse_bitset live;
   agx_context *ctx;
   unsigned position;
};

struct sched_node {
   struct dag_node dag;

   /* Instruction this node represents */
   agx_instr *instr;
   unsigned ready;
   unsigned orig_idx;
};

struct resource_reader {
   struct sched_node *node;
   struct resource_reader *next;
};

static void
add_dep(struct sched_node *a, struct sched_node *b)
{
   assert(a != b && "no self-dependencies");

   if (a && b)
      dag_add_edge(&a->dag, &b->dag, 0);
}

static void
serialize(struct sched_node *a, struct sched_node **b)
{
   add_dep(a, *b);
   *b = a;
}

/* last_write is indexed by value and empty on entry and on return, so a block
 * costs its own size rather than the shader's value count. */
static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx,
           struct sched_node **last_write, struct util_dynarray *orig_order)
{
   struct dag *dag = dag_create(ctx);

   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load, to serialize stores against */
   struct sched_node *memory_load = NULL;

   /* Last memory store, to serialize loads and stores against */
   struct sched_node *memory_store = NULL;
   struct sched_node *resource_writer[32] = {0};
   struct resource_reader *resource_readers[32] = {0};

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);
      util_dynarray_append(orig_order, node);

      /* Reads depend on writes, no other hazards in SSA */
      agx_foreach_ssa_src(I, s) {
         add_dep(node, last_write[I->src[s].value]);
      }

      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         last_write[I->dest[d].value] = node;
      }

      if (ctx->schedule_resources) {
         /* PHIs/preloads define block-entry values. Their position is an IR
          * invariant even when a target supplies its own resource hazards. */
         if (agx_opcodes_info[I->op].schedule_class == AGX_SCHEDULE_CLASS_PRELOAD)
            serialize(node, &preload);
         else
            add_dep(node, preload);
         uint32_t reads = 0, writes = 0;
         ctx->schedule_resources(I, &reads, &writes);
         u_foreach_bit(r, reads | writes) {
            add_dep(node, resource_writer[r]);
            if (writes & BITFIELD_BIT(r)) {
               for (struct resource_reader *p = resource_readers[r]; p;
                    p = p->next)
                  add_dep(node, p->node);
               resource_readers[r] = NULL;
               resource_writer[r] = node;
            } else {
               struct resource_reader *p = rzalloc(memctx, struct resource_reader);
               p->node = node;
               p->next = resource_readers[r];
               resource_readers[r] = p;
            }
         }
         continue;
      }

      /* Classify the instruction and add dependencies according to the class */
      enum agx_schedule_class dep = ctx->schedule_class
                                      ? ctx->schedule_class(I)
                                      : agx_opcodes_info[I->op].schedule_class;
      assert(dep != AGX_SCHEDULE_CLASS_INVALID && "invalid instruction seen");

      bool barrier = dep == AGX_SCHEDULE_CLASS_BARRIER;

      if (dep == AGX_SCHEDULE_CLASS_STORE)
         add_dep(node, memory_load);
      else if (dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_load);

      if (dep == AGX_SCHEDULE_CLASS_LOAD || dep == AGX_SCHEDULE_CLASS_STORE ||
          dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_store);

      if (dep == AGX_SCHEDULE_CLASS_COVERAGE || barrier)
         serialize(node, &coverage);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload);
   }

   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I))
         break;

      agx_foreach_ssa_dest(I, d)
         last_write[I->dest[d].value] = NULL;
   }

   return dag;
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instruction. Equivalently, calculate the difference in the number of live
 * registers before and after the instruction, given the live set after the
 * instruction. This calculation follows immediately from the dataflow
 * definition of liveness:
 *
 *      live_in = (live_out - KILL) + GEN
 */
static signed
calculate_pressure_delta(agx_instr *I, struct u_sparse_bitset *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (u_sparse_bitset_test(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !u_sparse_bitset_test(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

/*
 * Choose the next instruction, bottom-up. For now we use a simple greedy
 * heuristic: choose the instruction that has the best effect on liveness, while
 * hoisting sample_mask.
 */
static struct sched_node *
choose_instr(struct sched_ctx *s)
{
   int32_t min_delta = INT32_MAX;
   unsigned best_dependency = 0;
   struct sched_node *best = NULL;
   if (s->ctx->schedule_delay) {
      unsigned earliest = UINT_MAX;
      list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link)
         earliest = MIN2(earliest, n->ready);
      s->position = MAX2(s->position, earliest);
   }

   list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link) {
      if (s->ctx->schedule_delay && n->ready > s->position)
         continue;

      int32_t delta = calculate_pressure_delta(n->instr, &s->live);
      unsigned dependency = 0;
      if (s->ctx->schedule_delay) {
         /* Choose a consumer soon enough to expose its long-latency producer.
          * Otherwise pressure alone exhausts all independent work before the
          * producer even reaches the ready list. */
         util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
            struct sched_node *producer = (struct sched_node *)edge->child;
            dependency = MAX2(dependency,
                                s->ctx->schedule_delay(producer->instr) - 1);
         }
      }

      /* Newly exposed producers are appended to the ready list. On a tie
       * which increases pressure, follow the most recently exposed producer
       * instead of opening another expression tree. Breadth-first expansion
       * can keep an entire reduction level live despite equal local costs. */
      if (dependency > best_dependency ||
          (dependency == best_dependency &&
           (delta < min_delta || (delta == min_delta && delta > 0)))) {
         best = n;
         min_delta = delta;
         best_dependency = dependency;
      }
   }

   return best;
}

static unsigned
schedule_stalls(agx_context *ctx, struct sched_node *const *nodes,
                unsigned count, bool reverse,
                struct sched_node *const *dep_pool, const unsigned *dep_start)
{
   for (unsigned i = 0; i < count; ++i)
      nodes[i]->ready = 0;

   unsigned issue = 0, stalls = 0;
   for (unsigned k = 0; k < count; ++k) {
      struct sched_node *node = nodes[reverse ? (count - 1 - k) : k];
      unsigned earliest = issue;
      for (unsigned e = dep_start[node->orig_idx];
           e < dep_start[node->orig_idx + 1]; ++e)
         earliest = MAX2(earliest, dep_pool[e]->ready);
      stalls += earliest - issue;
      node->ready = earliest + ctx->schedule_delay(node->instr);
      issue = earliest + 1;
   }
   for (unsigned i = 0; i < count; ++i)
      nodes[i]->ready = 0;
   return stalls;
}

static void
pressure_schedule_block(agx_context *ctx, agx_block *block, struct sched_ctx *s)
{
   /* off by a constant, that's ok */
   signed pressure = 0;
   signed orig_max_pressure = 0;
   unsigned nr_ins = 0;

   u_sparse_bitset_dup(&s->live, &block->live_out);

   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta(I, &s->live);
      orig_max_pressure = MAX2(pressure, orig_max_pressure);
      agx_liveness_ins_update(&s->live, I);
      nr_ins++;
   }

   u_sparse_bitset_dup(&s->live, &block->live_out);

   /* Snapshot DAG edges before dag_prune_head clears them so latency scheduling
    * only commits a reorder when it strictly improves stalls or pressure. */
   unsigned orig_stalls = 0;
   struct sched_node **dep_pool = NULL;
   unsigned *dep_start = NULL;
   if (ctx->schedule_delay) {
      struct sched_node *const *orig =
         (struct sched_node *const *)s->orig_order.data;
      unsigned orig_count =
         util_dynarray_num_elements(&s->orig_order, struct sched_node *);
      unsigned total_edges = 0;
      for (unsigned i = 0; i < orig_count; ++i) {
         orig[i]->orig_idx = i;
         total_edges +=
            util_dynarray_num_elements(&orig[i]->dag.edges, struct dag_edge);
      }
      dep_start = calloc(orig_count + 1, sizeof(*dep_start));
      dep_pool = calloc(MAX2(total_edges, 1), sizeof(*dep_pool));
      unsigned pos = 0;
      for (unsigned i = 0; i < orig_count; ++i) {
         dep_start[i] = pos;
         util_dynarray_foreach(&orig[i]->dag.edges, struct dag_edge, edge) {
            if (edge->child)
               dep_pool[pos++] = (struct sched_node *)edge->child;
         }
      }
      dep_start[orig_count] = pos;
      orig_stalls =
         schedule_stalls(ctx, orig, orig_count, false, dep_pool, dep_start);
   }

   /* off by a constant, that's ok */
   signed max_pressure = 0;
   pressure = 0;

   struct sched_node **schedule = calloc(nr_ins, sizeof(struct sched_node *));
   nr_ins = 0;

   while (!list_is_empty(&s->dag->heads)) {
      struct sched_node *node = choose_instr(s);
      pressure += calculate_pressure_delta(node->instr, &s->live);
      max_pressure = MAX2(pressure, max_pressure);
      if (ctx->schedule_delay) {
         util_dynarray_foreach(&node->dag.edges, struct dag_edge, edge) {
            struct sched_node *producer = (struct sched_node *)edge->child;
            producer->ready = MAX2(producer->ready, s->position +
                                     ctx->schedule_delay(producer->instr));
         }
         ++s->position;
      }
      dag_prune_head(s->dag, &node->dag);

      schedule[nr_ins++] = node;
      agx_liveness_ins_update(&s->live, node->instr);
   }

   /* Bail if it looks like it's worse */
   signed limit = orig_max_pressure +
                  (ctx->schedule_delay ? (signed)ctx->schedule_pressure_slack : 0);
   if (ctx->schedule_delay && max_pressure > limit)
      ctx->schedule_rejected_increase =
         MAX2(ctx->schedule_rejected_increase,
              (unsigned)(max_pressure - orig_max_pressure));
   if (max_pressure > limit) {
      free(dep_pool);
      free(dep_start);
      free(schedule);
      return;
   }
   if (ctx->schedule_delay) {
      unsigned new_stalls =
         schedule_stalls(ctx, schedule, nr_ins, true, dep_pool, dep_start);
      free(dep_pool);
      free(dep_start);
      if (new_stalls > orig_stalls ||
          (new_stalls == orig_stalls && max_pressure >= orig_max_pressure)) {
         free(schedule);
         return;
      }
   } else if (max_pressure == orig_max_pressure) {
      free(schedule);
      return;
   }

   /* Apply the schedule */
   for (unsigned i = 0; i < nr_ins; ++i) {
      agx_remove_instruction(schedule[i]->instr);
      list_add(&schedule[i]->instr->link, &block->instructions);
   }

   free(schedule);
}

void
agx_pressure_schedule(agx_context *ctx)
{
   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);
   struct sched_node **last_write =
      rzalloc_array(memctx, struct sched_node *, ctx->alloc);

   agx_foreach_block(ctx, block) {
      struct sched_ctx sctx = {
         .ctx = ctx,
      };
      util_dynarray_init(&sctx.orig_order, memctx);
      sctx.dag = create_dag(ctx, block, memctx, last_write, &sctx.orig_order);
      u_sparse_bitset_init(&sctx.live, ctx->alloc, memctx);

      pressure_schedule_block(ctx, block, &sctx);
   }

   /* Clean up after liveness analysis */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   ralloc_free(memctx);
}
