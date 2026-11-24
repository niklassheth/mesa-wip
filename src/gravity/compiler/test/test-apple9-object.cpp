/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_encoding.h"
#include "agx_apple9_ir.h"
#include "agx_apple9_mesh.h"
#include "agx_apple9_dag_cache.h"
extern "C" {
#include "agx_nir.h"
}
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>
#include <set>
#include <vector>

TEST(Apple9ObjectIR, ContextCacheResetClearsEveryByte)
{
   agx_apple9_dag_cache expected;
   memset(&expected, 0, sizeof(expected));
   expected.zero_vreg = AGX_APPLE9_VREG_INVALID;
   expected.entry_zero = AGX_APPLE9_VREG_INVALID;
   expected.perspective_reciprocal = AGX_APPLE9_VREG_INVALID;
   memset(expected.system_vreg, 0xff, sizeof(expected.system_vreg));
   memset(expected.binding_pointer, 0xff, sizeof(expected.binding_pointer));

   /* Poison the complete object, not a list of today's members: adding a cache
    * without clearing it must fail. Repeat to represent both region boundaries.
    * hk-native-object-mask separately checks real compiler boundary placement
    * and rejects a callee use of a caller-only value.
    */
   for (unsigned poison : {0x5au, 0xa5u}) {
      agx_apple9_dag_cache cache;
      memset(&cache, poison, sizeof(cache));
      agx_apple9_dag_cache_reset(&cache);
      EXPECT_EQ(memcmp(&cache, &expected, sizeof(cache)), 0);
   }
}

static int64_t
call_displacement(const uint8_t *bytes)
{
   uint64_t raw = 0;
   for (unsigned j = 0; j < 6; ++j)
      raw |= uint64_t(bytes[3 + j]) << (8 * j);
   return int64_t(raw << 16) >> 16;
}

TEST(Apple9ObjectEncoding, SignedCallFieldsAndAlignment)
{
   for (int64_t d : std::initializer_list<int64_t>{-(INT64_C(1) << 47), -65538LL, -252LL, -2LL, 0LL,
                     2LL, 26LL, 65538LL, (INT64_C(1) << 47) - 2}) {
      uint8_t bytes[10];
      ASSERT_TRUE(agx_apple9_encode_call(bytes, 1, d));
      EXPECT_EQ(bytes[0], 0x8f);
      EXPECT_EQ(bytes[1] & 15, 0);
      EXPECT_EQ(bytes[1] >> 4, 1);
      EXPECT_EQ(bytes[2], 0x54);
      EXPECT_EQ(bytes[9], 0);
      EXPECT_EQ(call_displacement(bytes), d);
   }
   uint8_t bytes[10];
   for (int64_t d : std::initializer_list<int64_t>{-3LL, 1LL, -(INT64_C(1) << 47) - 2,
                     INT64_C(1) << 47})
      EXPECT_FALSE(agx_apple9_encode_call(bytes, 1, d));
   for (unsigned convention : {2u, 15u, 16u}) {
      EXPECT_FALSE(agx_apple9_encode_call(bytes, convention, 2));
      EXPECT_FALSE(agx_apple9_encode_return(bytes, convention));
   }
   ASSERT_TRUE(agx_apple9_encode_call(bytes, 0, -24));
   EXPECT_EQ(bytes[0], 0x8f);
   EXPECT_EQ(bytes[1], 0x00);
   EXPECT_EQ(bytes[2], 0x54);
   EXPECT_EQ(call_displacement(bytes), -24);
   ASSERT_TRUE(agx_apple9_encode_return(bytes, 0));
   const uint8_t ret0[] = {0x8f, 0x02, 0x54, 0};
   EXPECT_EQ(memcmp(bytes, ret0, sizeof(ret0)), 0);
   ASSERT_TRUE(agx_apple9_encode_return(bytes, 1));
   const uint8_t ret[] = {0x8f, 0x12, 0x54, 0};
   EXPECT_EQ(memcmp(bytes, ret, sizeof(ret)), 0);
   for (unsigned depth : {1u, 2u}) {
      ASSERT_TRUE(agx_apple9_encode_exec_pop(bytes, depth));
      EXPECT_EQ(bytes[3], depth);
      EXPECT_EQ(bytes[0], 0x0f);
      EXPECT_EQ(bytes[1], 6);
      EXPECT_EQ(bytes[2], 4);
      EXPECT_EQ(bytes[4] | bytes[5], 0);
   }
   EXPECT_FALSE(agx_apple9_encode_exec_pop(bytes, 0));
   EXPECT_FALSE(agx_apple9_encode_exec_pop(bytes, 3));
}

TEST(Apple9ObjectEncoding, GridExportPublicationField)
{
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_OBJECT_GRID_EXPORT;
   I.encoding = AGX_APPLE9_ENC_OBJECT_GRID_EXPORT;
   I.dest = AGX_APPLE9_VREG_INVALID;
   I.nr_srcs = 1;
   for (unsigned publication : {0u, 1u, 31u, 63u, 64u, 127u}) {
      uint8_t phys[] = {uint8_t(publication)};
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
      EXPECT_EQ(packed.length, 8u);
      EXPECT_EQ(packed.bytes[0], 0x57);
      EXPECT_EQ(packed.bytes[1], 0x16);
      EXPECT_EQ(packed.bytes[2], 0x54);
      EXPECT_EQ(packed.bytes[3] >> 1, publication);
      EXPECT_EQ(packed.bytes[3] & 1, 0);
      EXPECT_EQ(packed.bytes[4] | packed.bytes[5] | packed.bytes[7], 0);
      EXPECT_EQ(packed.bytes[6], 0x48);
   }
   uint8_t phys[] = {128};
   agx_apple9_packed_instruction packed;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
}

/* Abstract clobbering leaf for the allocator/layout oracle. Grid semantics
 * are tested through real NIR selection by apple9-native-object. */
static bool
emit_test_leaf(agx_apple9_vir_program *p, unsigned tag)
{
   if (!agx_apple9_vir_emit_side_effect(p, AGX_APPLE9_VIR_OBJECT_CALL,
          AGX_APPLE9_ENC_OBJECT_CALL, nullptr, 0, 0))
      return false;
   uint32_t lane = agx_apple9_vir_emit(p, AGX_APPLE9_VIR_GET_SR,
      AGX_APPLE9_ENC_GET_SR_ZEXT16, nullptr, 0, 0xa7);
   uint32_t value = agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x9000 + tag);
   uint32_t sources[] = {lane, value};
   uint32_t sum = agx_apple9_vir_emit(p, AGX_APPLE9_VIR_IADD,
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   return sum != AGX_APPLE9_VREG_INVALID &&
      agx_apple9_vir_emit_side_effect(p, AGX_APPLE9_VIR_STORE_UNIFORM,
         AGX_APPLE9_ENC_STORE_UNIFORM, &sum, 1, 160 + tag) &&
      agx_apple9_vir_emit_side_effect(p, AGX_APPLE9_VIR_OBJECT_RETURN,
         AGX_APPLE9_ENC_OBJECT_RETURN, nullptr, 0, 0) &&
      agx_apple9_vir_emit_side_effect(p, AGX_APPLE9_VIR_OBJECT_RESUME,
         AGX_APPLE9_ENC_OBJECT_RESUME, nullptr, 0, 0);
}

/* Execute allocated instructions through the actual out-of-line call edges.
 * This is an oracle for our compiler's liveness/control model, not evidence
 * that the hardware preserves unmentioned GPRs or has the assumed mask ABI. */
static void
check_live_values(unsigned pressure, unsigned calls, std::set<unsigned> &lanes)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "object_call_live_values");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   agx_apple9_device_load_contract load = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0, .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   std::vector<uint32_t> values;
   for (unsigned i = 0; i < pressure; ++i)
      values.push_back(agx_apple9_vir_emit_device_load(&p, i, zero, &load));
   for (unsigned i = 0; i < calls; ++i)
      ASSERT_TRUE(emit_test_leaf(&p, i));
   for (unsigned i = 0; i < pressure; ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p,
         AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, 32 + i));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_HALT,
      AGX_APPLE9_ENC_HALT, nullptr, 0, 0));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   ASSERT_EQ(p.scratch_size, 0u);
   ASSERT_TRUE(agx_apple9_layout_object_calls(&p, &reason)) << reason;

   std::vector<unsigned> offsets(p.instruction_count + 1);
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(p.instructions[i], p.phys,
                                                &packed, &reason)) << reason;
      offsets[i + 1] = offsets[i] + packed.length;
   }
   unsigned call_count = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op != AGX_APPLE9_VIR_OBJECT_CALL)
         continue;
      ++call_count;
      ASSERT_TRUE(I->branch_target);
      EXPECT_GT(I->branch_target->start_index, i);
      ASSERT_EQ(p.instructions[i + 1]->op, AGX_APPLE9_VIR_OBJECT_RESUME);
      agx_apple9_vir_instr resolved = *I;
      resolved.immediate = offsets[I->branch_target->start_index] - offsets[i];
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&resolved, p.phys, &packed, &reason));
      EXPECT_EQ(offsets[i] + call_displacement(packed.bytes),
                offsets[I->branch_target->start_index]);
   }
   EXPECT_EQ(call_count, calls);

   /* Layout only produces forward calls today. Still resolve a real backward
    * target from these final offsets, from the second callee to the first,
    * through the same signed narrowing apple9_compile_dag_once applies. */
   if (calls >= 2) {
      std::vector<unsigned> callees;
      const agx_apple9_vir_instr *first_call = nullptr;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         if (p.instructions[i]->op != AGX_APPLE9_VIR_OBJECT_CALL)
            continue;
         first_call = first_call ?: p.instructions[i];
         callees.push_back(p.instructions[i]->branch_target->start_index);
      }
      ASSERT_GE(callees.size(), 2u);
      ASSERT_LT(offsets[callees[0]], offsets[callees[1]]);
      const int64_t displacement =
         (int64_t)offsets[callees[0]] - offsets[callees[1]];
      agx_apple9_vir_instr resolved = *first_call;
      resolved.immediate = (uint32_t)(int32_t)displacement;
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&resolved, p.phys, &packed, &reason));
      EXPECT_LT(call_displacement(packed.bytes), 0);
      EXPECT_EQ(offsets[callees[1]] + call_displacement(packed.bytes),
                offsets[callees[0]]);
   }

   for (unsigned lane : {0u, 1u, 2u, 15u}) {
      uint32_t regs[AGX_APPLE9_GPR_COUNT] = {}, uniform[256] = {};
      auto input = [lane](unsigned i) { return 0xcafe0000u + 256 * lane + i; };
      bool active = true;
      unsigned pc = 0, return_pc = 0, returns = 0, resumes = 0;
      unsigned pending = 0, steps = 0;
      while (pc < p.instruction_count && steps++ < 1000) {
         const auto *I = p.instructions[pc++];
         auto read = [&](unsigned s) {
            if (I->alu_src_immediate_mask & (1u << s))
               return I->alu_src_value[s];
            return regs[I->src[s]];
         };
         if (I->scoreboard_slot)
            pending &= ~(1u << (I->scoreboard_slot - 1));
         if (!active && I->op != AGX_APPLE9_VIR_EXEC_MASK_POP &&
             I->op != AGX_APPLE9_VIR_OBJECT_RESUME && I->op != AGX_APPLE9_VIR_WAIT)
            continue;
         if (I->producer_scoreboard_slot)
            pending |= 1u << (I->producer_scoreboard_slot - 1);
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM: regs[I->dest] = I->immediate; break;
         case AGX_APPLE9_VIR_DEVICE_LOAD: regs[I->dest] = input(I->immediate); break;
         case AGX_APPLE9_VIR_GET_SR:
            EXPECT_EQ(I->immediate, 0xa7u);
            EXPECT_EQ(I->encoding, AGX_APPLE9_ENC_GET_SR_ZEXT16);
            lanes.insert(I->dest);
            regs[I->dest] = I->encoding == AGX_APPLE9_ENC_GET_SR_ZEXT16
                              ? lane : (0xdead0000 | lane);
            break;
         case AGX_APPLE9_VIR_IADD: regs[I->dest] = read(0) + read(1); break;
         case AGX_APPLE9_VIR_IOR: regs[I->dest] = read(0) | read(1); break;
         case AGX_APPLE9_VIR_OBJECT_CALL:
            EXPECT_EQ(pending, 0u);
            EXPECT_EQ(return_pc, 0u);
            return_pc = pc;
            pc = I->branch_target->start_index;
            break;
         case AGX_APPLE9_VIR_OBJECT_RETURN:
            EXPECT_EQ(pending, 0u);
            ASSERT_NE(return_pc, 0u);
            pc = return_pc;
            return_pc = 0;
            active = false; /* Observed empty R before cleanup. */
            ++returns;
            break;
         case AGX_APPLE9_VIR_OBJECT_RESUME:
            EXPECT_FALSE(active);
            active = true;
            ++resumes;
            break;
         case AGX_APPLE9_VIR_STORE_UNIFORM:
            uniform[I->immediate] = read(0) +
               (I->uniform_op == AGX_APPLE9_VIR_IADD ? read(1) : 0);
            break;
         case AGX_APPLE9_VIR_DEVICE_FENCE:
            EXPECT_EQ(pending, 0u);
            break;
         case AGX_APPLE9_VIR_WAIT: pending &= ~I->immediate; break;
         case AGX_APPLE9_VIR_HALT: pc = p.instruction_count; break;
         default: FAIL() << "unhandled physical op " << I->op;
         }
      }
      EXPECT_LT(steps, 1000u);
      EXPECT_EQ(returns, calls);
      EXPECT_EQ(resumes, calls);
      for (unsigned i = 0; i < calls; ++i)
         EXPECT_EQ(uniform[160 + i], 0x9000 + i + lane);
      for (unsigned i = 0; i < pressure; ++i)
         EXPECT_EQ(uniform[32 + i], input(i)) << "live value " << i;
   }
   agx_apple9_vir_finish(&p);
   ralloc_free(b.shader);
}

TEST(Apple9ObjectIR, LiveValuesAcrossOneAndTwoForwardCalls)
{
   std::set<unsigned> lanes;
   for (unsigned pressure : {1u, 12u, 40u})
      for (unsigned calls : {1u, 2u})
         check_live_values(pressure, calls, lanes);
   EXPECT_GT(lanes.size(), 1u);
}

TEST(Apple9ObjectIR, RejectsUnclosedRegionsAndScratchFrames)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_OBJECT_RETURN,
      AGX_APPLE9_ENC_OBJECT_RETURN, nullptr, 0, 0));
   const char *reason = nullptr;
   EXPECT_FALSE(agx_apple9_layout_object_calls(&p, &reason));
   agx_apple9_vir_finish(&p);
   ASSERT_TRUE(emit_test_leaf(&p, 0));
   p.physical = p.dependencies_finalized = true;
   p.scratch_size = 16;
   EXPECT_FALSE(agx_apple9_layout_object_calls(&p, &reason));
   EXPECT_STREQ(reason, "object calls require scheduled allocation without scratch");
   agx_apple9_vir_finish(&p);
}

static bool
has_intrinsic(const nir_shader *nir, nir_intrinsic_op op)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return true;
         }
      }
   }
   return false;
}

static bool
has_native_pointer_base(const nir_shader *nir, unsigned base)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_native_pointer_apple9_agx &&
                (unsigned)nir_intrinsic_base(intr) == base)
               return true;
         }
      }
   }
   return false;
}

static nir_shader *
build_app_task_shader(unsigned wx, unsigned wy, unsigned wz,
                      unsigned payload_B, unsigned shared_B,
                      bool use_atomics, bool use_sysvals)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "app_native_object");
   b.shader->info.workgroup_size[0] = wx;
   b.shader->info.workgroup_size[1] = wy;
   b.shader->info.workgroup_size[2] = wz;
   b.shader->info.task_payload_size = payload_B;
   b.shader->info.shared_size = shared_B;

   nir_def *lid = nir_load_local_invocation_index(&b);
   nir_def *val = lid;
   if (use_sysvals) {
      nir_def *lid3 = nir_load_local_invocation_id(&b);
      nir_def *wgid = nir_load_workgroup_id(&b);
      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *nwg = nir_load_num_workgroups(&b);
      val = nir_iadd(&b, val, nir_channel(&b, lid3, 0));
      val = nir_iadd(&b, val, nir_channel(&b, lid3, 1));
      val = nir_iadd(&b, val, nir_channel(&b, lid3, 2));
      val = nir_iadd(&b, val, nir_channel(&b, wgid, 0));
      val = nir_iadd(&b, val, nir_channel(&b, gid, 1));
      val = nir_iadd(&b, val, nir_channel(&b, nwg, 2));
   }

   if (shared_B >= 4) {
      nir_def *soff = nir_imul_imm(&b, nir_iand_imm(&b, lid, (shared_B / 4) - 1), 4);
      nir_store_shared(&b, val, soff, .write_mask = 1, .align_mul = 4);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
      nir_def *loaded = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0),
                                        .align_mul = 4);
      val = nir_iadd(&b, val, loaded);
   }

   if (payload_B >= 16) {
      nir_push_if(&b, nir_ieq_imm(&b, lid, 0));
      nir_store_task_payload(&b, val, nir_imm_int(&b, 0),
                             .write_mask = 1, .align_mul = 4);
      if (payload_B >= 16) {
         nir_store_task_payload(&b, val, nir_imm_int(&b, payload_B - 4),
                                .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(&b, nullptr);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_task_payload);
      if (use_atomics) {
         nir_def *old = nir_task_payload_atomic(
            &b, 32, nir_imm_int(&b, 4), nir_imm_int(&b, 1),
            .atomic_op = nir_atomic_op_iadd);
         nir_def *prev = nir_task_payload_atomic_swap(
            &b, 32, nir_imm_int(&b, 8), old, val,
            .atomic_op = nir_atomic_op_cmpxchg);
         nir_push_if(&b, nir_ieq_imm(&b, lid, 0));
         nir_def *rd = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 0),
                                             .align_mul = 4);
         nir_store_task_payload(&b, nir_iadd(&b, rd, prev), nir_imm_int(&b, 12),
                                .write_mask = 1, .align_mul = 4);
         nir_pop_if(&b, nullptr);
      }
   }

   nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 4, 2, 1));
   return b.shader;
}

TEST(Apple9NativeObject, PayloadLoadStoreAndAtomics)
{
   for (unsigned payload_B : {16u, 256u, 16384u}) {
      nir_shader *nir = build_app_task_shader(16, 1, 1, payload_B, 0, true, false);
      nir_shader *lower_probe = nir_shader_clone(nullptr, nir);
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      agx_apple9_native_object_options opts = {
         .payload_stride_B = payload_B,
         .write_child_grid = true,
      };
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_native_object_ex(
         lower_probe, true, &key, &opts, &info, &reason))
         << (reason ? reason : "");
      EXPECT_EQ(info.task_payload_size_B, payload_B);
      EXPECT_EQ(info.task_payload_stride_B, payload_B);
      EXPECT_TRUE(info.writes_child_grid);
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_load_task_payload));
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_store_task_payload));
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_task_payload_atomic));
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_task_payload_atomic_swap));
      EXPECT_TRUE(has_native_pointer_base(lower_probe, AGX_APPLE9_MESH_PAYLOAD_UNIFORM));
      ralloc_free(lower_probe);

      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_apple9_native_object_ex(
         nir, true, &key, &opts, &part, &info, &reason))
         << (reason ? reason : "");
      EXPECT_TRUE(part.info.apple9_body_only);
      EXPECT_GT(part.info.binary_size, 0u);
      EXPECT_GT(part.info.nr_gprs, 0u);
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_EQ(part.info.workgroup_size[0], 16u);
      free(part.binary);
      ralloc_free(nir);
   }

   /* Payload > 16 KiB must be rejected cleanly. */
   nir_shader *too_big = build_app_task_shader(16, 1, 1, 16388, 0, false, false);
   agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   agx_apple9_native_object_options opts = {.payload_stride_B = 16400};
   agx_apple9_mesh_body_info info = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_nir_lower_apple9_native_object_ex(
      too_big, true, &key, &opts, &info, &reason));
   EXPECT_NE(reason, nullptr);
   ralloc_free(too_big);

   /* Out-of-bounds static payload offset > 16 KiB must be rejected. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "oob_payload");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = 16384;
   nir_store_task_payload(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 16384),
                          .write_mask = 1, .align_mul = 4);
   nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 1, 1, 1));
   opts.payload_stride_B = 16384;
   reason = nullptr;
   EXPECT_FALSE(agx_nir_lower_apple9_native_object_ex(
      b.shader, true, &key, &opts, &info, &reason));
   EXPECT_NE(reason, nullptr);
   ralloc_free(b.shader);
}

TEST(Apple9NativeObject, SharedMemoryAndBarriers)
{
   for (unsigned shared_B : {64u, 256u, 16384u}) {
      nir_shader *nir = build_app_task_shader(32, 1, 1, 64, shared_B, false, false);
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      key.physical_threads = 32;
      agx_apple9_native_object_options opts = {.payload_stride_B = 64};
      agx_shader_part part = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_native_object_ex(
         nir, true, &key, &opts, &part, &info, &reason))
         << (reason ? reason : "");
      EXPECT_GE(info.shared_size_B, shared_B);
      EXPECT_GE(info.local_allocation_B, 16u + shared_B);
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_EQ(part.info.workgroup_size[0], 32u);
      free(part.binary);
      ralloc_free(nir);
   }

   /* Shared memory overflowing 32 KiB after the grid header must be rejected. */
   nir_shader *overflow = build_app_task_shader(16, 1, 1, 0, 32768, false, false);
   agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   agx_apple9_native_object_options opts = {};
   agx_apple9_mesh_body_info info = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_nir_lower_apple9_native_object_ex(
      overflow, true, &key, &opts, &info, &reason));
   EXPECT_NE(reason, nullptr);
   ralloc_free(overflow);
}

TEST(Apple9NativeObject, WorkgroupSizes1To128AndSysvals)
{
   struct Shape { unsigned x, y, z, phys; };
   const Shape shapes[] = {
      {1, 1, 1, 16},
      {3, 1, 1, 16},
      {2, 2, 3, 16},
      {16, 1, 1, 16},
      {17, 1, 1, 32},
      {32, 1, 1, 32},
      {4, 8, 1, 32},
      {64, 1, 1, 32},
      {4, 4, 4, 32},
      {128, 1, 1, 32},
   };
   for (const Shape &s : shapes) {
      nir_shader *nir = build_app_task_shader(s.x, s.y, s.z, 64, 128, false, true);
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      key.physical_threads = s.phys;
      agx_apple9_native_object_options opts = {
         .payload_stride_B = 64,
         .write_child_grid = true,
      };
      agx_shader_part part = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_native_object_ex(
         nir, true, &key, &opts, &part, &info, &reason))
         << "shape " << s.x << "x" << s.y << "x" << s.z << ": "
         << (reason ? reason : "");
      EXPECT_EQ(part.info.workgroup_size[0], s.phys);
      EXPECT_EQ(part.info.workgroup_size[1], 1u);
      EXPECT_EQ(part.info.workgroup_size[2], 1u);
      EXPECT_EQ(info.logical_size[0], s.x);
      EXPECT_EQ(info.logical_size[1], s.y);
      EXPECT_EQ(info.logical_size[2], s.z);
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_GT(part.info.binary_size, 0u);
      free(part.binary);
      ralloc_free(nir);
   }

   /* > 128 invocations must be rejected cleanly. */
   for (const Shape &bad : std::initializer_list<Shape>{{129, 1, 1, 32}, {8, 8, 4, 32}}) {
      nir_shader *nir = build_app_task_shader(bad.x, bad.y, bad.z, 0, 0, false, false);
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      key.physical_threads = bad.phys;
      agx_apple9_native_object_options opts = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      EXPECT_FALSE(agx_nir_lower_apple9_native_object_ex(
         nir, true, &key, &opts, &info, &reason));
      EXPECT_NE(reason, nullptr);
      ralloc_free(nir);
   }
}

TEST(Apple9NativeObject, ZeroGridConditionalAndEarlyReturn)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TASK, &agx_nir_options, "zero_or_cond_task");
      b.shader->info.workgroup_size[0] = mode == 3 ? 64 : 16;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.task_payload_size = 32;
      nir_def *wgid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      if (mode == 0) {
         nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 0, 0, 0));
      } else if (mode == 1) {
         nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 0, 5, 1));
      } else if (mode == 2) {
         /* Early return without calling EmitMeshTasksEXT: exports (0,0,0). */
         nir_push_if(&b, nir_ieq_imm(&b, wgid, 0));
         nir_store_task_payload(&b, nir_imm_int(&b, 7), nir_imm_int(&b, 0),
                                .write_mask = 1, .align_mul = 4);
         nir_jump(&b, nir_jump_return);
         nir_pop_if(&b, nullptr);
         nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 2, 1, 1));
      } else {
         /* Serialized T64 with conditional launch_mesh_workgroups. */
         nir_push_if(&b, nir_ine_imm(&b, wgid, 0));
         nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 3, 2, 1));
         nir_pop_if(&b, nullptr);
      }
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      key.physical_threads = mode == 3 ? 32 : 16;
      agx_apple9_native_object_options opts = {
         .payload_stride_B = 32,
         .write_child_grid = true,
      };
      agx_shader_part part = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_native_object_ex(
         b.shader, true, &key, &opts, &part, &info, &reason))
         << "mode " << mode << ": " << (reason ? reason : "");
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_GT(part.info.binary_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

static nir_shader *
build_mesh_with_payload(unsigned payload_B, bool read_num_workgroups,
                        bool illegal_store)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh_with_payload");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = payload_B;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;

   nir_def *pv = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, payload_B - 4),
                                       .align_mul = 4);
   if (illegal_store) {
      nir_store_task_payload(&b, pv, nir_imm_int(&b, 0),
                             .write_mask = 1, .align_mul = 4);
   }
   nir_def *nwg_x = read_num_workgroups
                       ? nir_channel(&b, nir_load_num_workgroups(&b), 0)
                       : nir_imm_int(&b, 1);
   nir_def *xf = nir_u2f32(&b, nir_iadd(&b, pv, nwg_x));
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i) {
      nir_store_per_vertex_output(
         &b, nir_vec4(&b, xf, nir_imm_float(&b, float(i)),
                      nir_imm_float(&b, 0.5f), nir_imm_float(&b, 1.0f)),
         nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   return b.shader;
}

TEST(Apple9NativeObject, MeshPayloadReadAndNumWorkgroups)
{
   for (unsigned payload_B : {16u, 256u, 16384u}) {
      nir_shader *nir = build_mesh_with_payload(payload_B, true, false);
      nir_shader *lower_probe = nir_shader_clone(nullptr, nir);
      agx_apple9_native_mesh_options opts = {
         .attributes = true,
         .native_object_payload = true,
         .native_object_payload_stride_B = payload_B,
      };
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_native_mesh(
         lower_probe, true, &opts, &info, &reason))
         << (reason ? reason : "");
      EXPECT_TRUE(info.reads_native_payload);
      EXPECT_EQ(info.task_payload_size_B, payload_B);
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_load_task_payload));
      EXPECT_FALSE(has_intrinsic(lower_probe, nir_intrinsic_load_num_workgroups));
      EXPECT_TRUE(has_native_pointer_base(lower_probe, AGX_APPLE9_MESH_PAYLOAD_UNIFORM));
      ralloc_free(lower_probe);

      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_apple9_native_mesh(
         nir, true, &opts, &part, &info, &reason))
         << (reason ? reason : "");
      EXPECT_TRUE(info.reads_native_payload);
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_GT(part.info.binary_size, 0u);
      free(part.binary);
      ralloc_free(nir);
   }

   /* Mesh payload write is illegal and must be rejected. */
   nir_shader *bad_write = build_mesh_with_payload(64, false, true);
   agx_apple9_native_mesh_options opts = {
      .attributes = true,
      .native_object_payload = true,
      .native_object_payload_stride_B = 64,
   };
   agx_apple9_mesh_body_info info = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_nir_lower_apple9_native_mesh(
      bad_write, true, &opts, &info, &reason));
   ralloc_free(bad_write);

   /* Mesh payload read without native_object_payload option must be rejected. */
   nir_shader *disabled = build_mesh_with_payload(64, false, false);
   opts.native_object_payload = false;
   EXPECT_FALSE(agx_nir_lower_apple9_native_mesh(
      disabled, true, &opts, &info, &reason));
   ralloc_free(disabled);
}
