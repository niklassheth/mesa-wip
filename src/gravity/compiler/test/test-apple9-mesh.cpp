/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_ir.h"
#include "agx_apple9_mesh.h"
#include "agx_apple9_mesh_encoding.h"
#include "agx_compile_apple9.h"
extern "C" {
#include "agx_nir.h"
}
#include <array>
#include <set>
#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>

/* The 58 unique singleton encodings in UNITS-DECODE-20261018. Sorted by
 * bytes, not execution order. No ordinary code or helper span is included. */
static const uint8_t census[][6] = {
   {0x57, 0x03, 0x54, 0x01, 0x0c, 0x30}, {0x57, 0x03, 0x54, 0x01, 0x0c, 0x34},
   {0x57, 0x03, 0x54, 0x01, 0x1c, 0x30}, {0x57, 0x03, 0x54, 0x01, 0x1c, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x02, 0x34}, {0x57, 0x03, 0x54, 0x06, 0x0e, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x14, 0x30}, {0x57, 0x03, 0x54, 0x06, 0x14, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x1c, 0x30}, {0x57, 0x03, 0x54, 0x07, 0x0c, 0x30},
   {0x57, 0x03, 0x54, 0x07, 0x0c, 0x34}, {0x57, 0x03, 0x54, 0x0a, 0x0c, 0x30},
   {0x57, 0x03, 0x54, 0x0a, 0x0c, 0x34}, {0x57, 0x03, 0x54, 0x0c, 0x1c, 0x34},
   {0x57, 0x03, 0x54, 0x0d, 0x0c, 0x30}, {0x57, 0x03, 0x54, 0x0d, 0x0c, 0x34},
   {0x57, 0x03, 0x54, 0x0e, 0x02, 0x30}, {0x57, 0x03, 0x54, 0x0e, 0x0c, 0x34},
   {0xd7, 0x06, 0x54, 0x05, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x05, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x06, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x07, 0x06, 0x00},
   {0xd7, 0x06, 0x54, 0x85, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x85, 0x0e, 0x00},
   {0xd7, 0x06, 0x54, 0x85, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x86, 0x0e, 0x00},
   {0xd7, 0x06, 0x54, 0x87, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x87, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x88, 0x0e, 0x00}, {0xd7, 0x06, 0x54, 0x88, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x89, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8a, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8b, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8c, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8d, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8e, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8f, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x90, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x91, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x92, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x93, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x94, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x95, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x96, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x97, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x98, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x99, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x9a, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x9b, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x9c, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x9e, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa0, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xa2, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa4, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xa6, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa8, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xaa, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xac, 0x16, 0x00},
};

TEST(Apple9MeshEncoding, All58CensusSingletonsRoundTrip)
{
   static_assert(ARRAY_SIZE(census) == 58);
   unsigned control = 0, submit = 0;
   for (const auto &bytes : census) {
      agx_apple9_mesh_instruction ins = {};
      ASSERT_TRUE(agx_apple9_decode_mesh(bytes, &ins));
      uint8_t encoded[6] = {};
      ASSERT_TRUE(agx_apple9_encode_mesh(encoded, &ins));
      EXPECT_EQ(memcmp(bytes, encoded, 6), 0);
      /* Independently inspect the named field positions. */
      if (ins.kind == AGX_APPLE9_MESH_CONTROL) {
         ++control;
         EXPECT_EQ(2 * ins.count_gpr + ins.count_high, bytes[3]);
         EXPECT_EQ(2 * ins.auxiliary_gpr + ins.auxiliary_high,
                   (bytes[4] >> 1) | ((bytes[5] & 1) << 7));
         EXPECT_EQ(ins.control42, bool(bytes[5] & 4));
      } else {
         ++submit;
         EXPECT_EQ(ins.source_gpr, bytes[3] & 127);
         EXPECT_EQ(ins.release, bool(bytes[3] & 128));
         EXPECT_EQ(unsigned(ins.mode), (bytes[4] >> 3) & 3);
      }
   }
   EXPECT_EQ(control, 18u);
   EXPECT_EQ(submit, 40u);
}

TEST(Apple9MeshEncoding, RejectsUnknownControlsAndOutOfRangeRegisters)
{
   const uint8_t baseline[6] = {0x57, 3, 0x54, 6, 0x14, 0x34};
   for (unsigned bit : {18u, 23u, 32u, 41u, 43u, 44u, 45u, 46u, 47u}) {
      uint8_t changed[6];
      memcpy(changed, baseline, 6);
      changed[bit / 8] ^= 1u << (bit % 8);
      agx_apple9_mesh_instruction ins = {};
      EXPECT_FALSE(agx_apple9_decode_mesh(changed, &ins)) << bit;
   }
   agx_apple9_mesh_instruction ins = {};
   ins.kind = AGX_APPLE9_MESH_CONTROL;
   uint8_t bytes[6];
   ins.count_gpr = 96;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins.count_gpr = 0;
   ins.auxiliary_gpr = 96;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins = {};
   ins.kind = AGX_APPLE9_MESH_SUBMIT;
   for (auto mode : {AGX_APPLE9_MESH_SUBMIT_SCALAR,
                     AGX_APPLE9_MESH_SUBMIT_CHANNEL}) {
      ins.mode = mode;
      ins.source_gpr = 95;
      EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
      ins.source_gpr = 94;
      EXPECT_TRUE(agx_apple9_encode_mesh(bytes, &ins));
   }
   ins.mode = (agx_apple9_mesh_submit_mode)3;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins.mode = (agx_apple9_mesh_submit_mode)-1;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
}

TEST(Apple9MeshEncoding, UInputWaitMaskRoundTripsAllSlots)
{
   for (auto kind : {AGX_APPLE9_MESH_CONTROL, AGX_APPLE9_MESH_SUBMIT}) {
      for (auto mode : {AGX_APPLE9_MESH_SUBMIT_SCALAR,
                        AGX_APPLE9_MESH_SUBMIT_CHANNEL}) {
         agx_apple9_mesh_instruction ins = {};
         ins.kind = kind;
         ins.mode = mode;
         ins.count_gpr = 4;
         ins.auxiliary_gpr = 7;
         ins.source_gpr = 8;
         uint8_t baseline[6];
         ASSERT_TRUE(agx_apple9_encode_mesh(baseline, &ins));
         for (unsigned mask = 0; mask < 64; ++mask) {
            ins.input_wait_mask = mask;
            uint8_t bytes[6];
            ASSERT_TRUE(agx_apple9_encode_mesh(bytes, &ins));
            EXPECT_EQ((bytes[1] >> 4) | ((bytes[2] & 3) << 4), mask);
            agx_apple9_mesh_instruction decoded = {};
            ASSERT_TRUE(agx_apple9_decode_mesh(bytes, &decoded));
            EXPECT_EQ(decoded.input_wait_mask, mask);
            bytes[1] &= 0x0f;
            bytes[2] &= ~3;
            EXPECT_EQ(memcmp(bytes, baseline, 6), 0);
         }
         ins.input_wait_mask = 64;
         uint8_t bytes[6];
         EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
      }
   }
   /* CULL has no independently qualified input-wait field. */
   agx_apple9_mesh_instruction cull = {};
   cull.kind = AGX_APPLE9_MESH_SUBMIT;
   cull.mode = AGX_APPLE9_MESH_SUBMIT_CULL;
   uint8_t bytes[6];
   ASSERT_TRUE(agx_apple9_encode_mesh(bytes, &cull));
   bytes[1] |= 0x10;
   agx_apple9_mesh_instruction decoded = {};
   EXPECT_FALSE(agx_apple9_decode_mesh(bytes, &decoded));
   cull.input_wait_mask = 1;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &cull));
}

TEST(Apple9MeshEncoding, FixedProducerDescriptorAndInputPacking)
{
   const auto *control = agx_apple9_encoding_info(AGX_APPLE9_ENC_MESH_CONTROL);
   EXPECT_EQ(control->fixed_producer_slot, 1u);
   EXPECT_EQ(control->fixed_producer_alternates, 2u);
   EXPECT_FALSE(control->requires_prior_completion);
   EXPECT_EQ(agx_apple9_encoding_info(AGX_APPLE9_ENC_GET_DRAW_ID)->fixed_producer_slot, 1u);
   EXPECT_EQ(agx_apple9_encoding_info(AGX_APPLE9_ENC_SHARED_ATOMIC)->fixed_producer_slot, 0u);
   EXPECT_EQ(agx_apple9_encoding_info(AGX_APPLE9_ENC_SHARED_LOAD)->fixed_producer_slot, 0u);
   EXPECT_EQ(agx_apple9_encoding_info(AGX_APPLE9_ENC_PRIVATE_LOAD)->fixed_producer_slot, 0u);
   uint8_t phys[] = {4, 5};
   for (auto op : {AGX_APPLE9_VIR_MESH_CONTROL,
                  AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR,
                  AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL}) {
      bool u0 = op == AGX_APPLE9_VIR_MESH_CONTROL;
      agx_apple9_vir_instr I = {};
      I.op = op;
      I.encoding = u0 ? AGX_APPLE9_ENC_MESH_CONTROL :
         (op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ?
          AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL : AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR);
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.nr_srcs = 2;
      I.src[1] = 1;
      I.src_half_mask = u0 ? 3 : 0;
      I.producer_scoreboard_slot = u0 ? 1 : 0;
      for (unsigned slot = 0; slot <= 6; ++slot) {
         I.scoreboard_slot = slot;
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         agx_apple9_mesh_instruction decoded = {};
         ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
         EXPECT_EQ(decoded.input_wait_mask, slot ? BITFIELD_BIT(slot - 1) : 0);
      }
      I.producer_scoreboard_slot = 2;
      agx_apple9_packed_instruction packed = {};
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
   }
}

TEST(Apple9MeshEncoding, VirPacksAllocatedHalfwordsAndAdjacentPairs)
{
   for (unsigned a : {0u, 3u, 7u, 31u, 64u, 95u}) {
      for (unsigned halves = 0; halves < 4; ++halves) {
         agx_apple9_vir_instr I = {};
         I.op = AGX_APPLE9_VIR_MESH_CONTROL;
         I.encoding = AGX_APPLE9_ENC_MESH_CONTROL;
         I.dest = AGX_APPLE9_VREG_INVALID;
         I.src[0] = 0;
         I.src[1] = 1;
         I.nr_srcs = 2;
         I.src_half_mask = 3;
         I.src_hi_mask = halves;
         I.mesh_control42 = true;
         const uint8_t phys[] = {(uint8_t)a, (uint8_t)(95 - a)};
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         agx_apple9_mesh_instruction decoded = {};
         ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
         EXPECT_EQ(decoded.count_gpr, a);
         EXPECT_EQ(decoded.count_high, bool(halves & 1));
         EXPECT_EQ(decoded.auxiliary_gpr, 95 - a);
         EXPECT_EQ(decoded.auxiliary_high, bool(halves & 2));
         for (unsigned live : {1u, 2u, 3u}) {
            I.live_after_mask = live;
            EXPECT_EQ(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr),
                      live == 2u);
         }
      }
   }
   for (bool channel : {false, true}) {
      agx_apple9_vir_instr I = {};
      I.op = channel ? AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL
                     : AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR;
      I.encoding = channel ? AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL
                           : AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.src[0] = 0;
      I.src[1] = 1;
      I.nr_srcs = 2;
      I.mesh_release = channel;
      for (unsigned base : {0u, 5u, 11u, 44u, 63u, 64u, 94u}) {
         uint8_t phys[] = {(uint8_t)base, (uint8_t)(base + 1)};
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         agx_apple9_mesh_instruction decoded = {};
         ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
         EXPECT_EQ(decoded.source_gpr, base);
         EXPECT_EQ(decoded.mode, channel ? AGX_APPLE9_MESH_SUBMIT_CHANNEL
                                         : AGX_APPLE9_MESH_SUBMIT_SCALAR);
         EXPECT_EQ(decoded.release, channel);
         I.nr_srcs = 1;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         I.nr_srcs = 2;
         phys[1] = base;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
      }
   }
}

TEST(Apple9MeshIR, AllocationPreservesReadClearAndTupleInputsUsedLater)
{
   std::set<unsigned> count_locations, pair_locations;
   for (unsigned pressure : {0u, 12u, 112u}) {
      for (bool high : {false, true}) {
         for (unsigned primitive_count : {0u, 1u, 2u}) {
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_MESH, &agx_nir_options, "mesh_destructive_operands");
            nir_block logical = {};
            agx_apple9_vir_program p;
            agx_apple9_vir_init(&p);
            auto block = agx_apple9_block_create(&p);
            block->nir = &logical;
            agx_apple9_block_begin(&p, block);
            auto zero = agx_apple9_vir_emit(
               &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
            agx_apple9_device_load_contract load = {
               .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
               .flags = 0,
               .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
            };
            std::vector<uint32_t> values;
            for (unsigned i = 0; i < pressure + 4; ++i)
               values.push_back(
                  agx_apple9_vir_emit_device_load(&p, i, zero, &load));
            /* Keep every load before the mesh operations so the large case
             * exceeds the register file instead of scheduling loads later. */
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_WORKGROUP_BARRIER,
               AGX_APPLE9_ENC_WORKGROUP_BARRIER, nullptr, 0, 0));
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, values[0],
                                                         values[1], true));
            p.instructions[p.instruction_count - 1]->src_hi_mask = high;
            const auto *control = p.instructions[p.instruction_count - 1];
            EXPECT_FALSE(agx_apple9_instr_is_pure_alu(control));
            EXPECT_EQ(agx_apple9_find_operand(control->encoding,
                                              AGX_APPLE9_OPERAND_SRC0)
                         ->widths,
                      AGX_APPLE9_WIDTH_16);
            EXPECT_TRUE(agx_apple9_find_operand(control->encoding,
                                                AGX_APPLE9_OPERAND_SRC0)
                           ->flags &
                        AGX_APPLE9_OPERAND_CLOBBER);
            EXPECT_FALSE(agx_apple9_find_operand(control->encoding,
                                                 AGX_APPLE9_OPERAND_SRC1)
                            ->flags & AGX_APPLE9_OPERAND_CLOBBER);
            uint32_t pair[] = {values[2], values[3]};
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, true, true));
            uint32_t scalar[] = {values[1], zero};
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, scalar, false, false));
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
               &zero, 1, values.size()));
            /* Every original SSA value remains live across the destructive ops.
             */
            for (unsigned i = 0; i < values.size(); ++i)
               ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
                  &p, AGX_APPLE9_VIR_STORE_UNIFORM,
                  AGX_APPLE9_ENC_STORE_UNIFORM, &values[i], 1, i));
            const char *reason = nullptr;
            ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason))
               << reason;

            auto input = [&](unsigned i) -> uint32_t {
               if (i == 0)
                  return high ? (primitive_count << 16) | 0xcafe
                              : 0xcafe0000 | primitive_count;
               if (i == 1)
                  return 3;
               if (i == 2)
                  return 2;
               if (i == 3)
                  return 64;
               return 0x12340000 + i;
            };
            if (pressure == 112) {
               EXPECT_GT(p.scratch_size, 0u);
            }
            uint32_t regs[AGX_APPLE9_GPR_COUNT] = {}, uniform[256] = {};
            uint32_t scratch[AGX_APPLE9_SAVE_WORD_COUNT] = {};
            unsigned controls = 0, channels = 0, scalars = 0;
            for (unsigned i = 0; i < p.instruction_count; ++i) {
               const auto *I = p.instructions[i];
               agx_apple9_packed_instruction packed = {};
               ASSERT_TRUE(
                  agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
                  << "op=" << I->op << " " << (reason ?: "");
               switch (I->op) {
               case AGX_APPLE9_VIR_IMM:
                  regs[I->dest] = I->immediate;
                  break;
               case AGX_APPLE9_VIR_DEVICE_LOAD:
                  regs[I->dest] = input(I->immediate);
                  break;
               case AGX_APPLE9_VIR_IOR:
                  regs[I->dest] = regs[I->src[0]] | regs[I->src[1]];
                  break;
               case AGX_APPLE9_VIR_SPILL_STORE:
                  scratch[I->immediate] = regs[I->src[0]];
                  break;
               case AGX_APPLE9_VIR_SPILL_LOAD:
                  regs[I->dest] = scratch[I->immediate];
                  break;
               case AGX_APPLE9_VIR_WORKGROUP_BARRIER:
               case AGX_APPLE9_VIR_WAIT:
                  break;
               case AGX_APPLE9_VIR_MESH_CONTROL: {
                  agx_apple9_mesh_instruction decoded = {};
                  ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
                  count_locations.insert(decoded.count_gpr);
                  unsigned shift = decoded.count_high ? 16 : 0;
                  EXPECT_EQ((regs[decoded.count_gpr] >> shift) & 65535,
                            primitive_count);
                  EXPECT_EQ(regs[decoded.auxiliary_gpr] & 65535, 3u);
                  EXPECT_EQ(I->live_after_mask & ~2u, 0u);
                  regs[decoded.count_gpr] &= ~(65535u << shift);
                  /* B remains available to its later SSA uses. */
                  ++controls;
                  break;
               }
               case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
               case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL: {
                  agx_apple9_mesh_instruction decoded = {};
                  ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
                  pair_locations.insert(decoded.source_gpr);
                  EXPECT_EQ(I->src[1], I->src[0] + 1);
                  bool channel = I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL;
                  EXPECT_EQ(regs[decoded.source_gpr], channel ? 2u : 3u);
                  EXPECT_EQ(regs[decoded.source_gpr + 1], channel ? 64u : 0u);
                  EXPECT_EQ(I->live_after_mask, 0u);
                  /* Poison released storage: any use of the old SSA value
                   * must use the allocator's preserved copy. */
                  regs[decoded.source_gpr] = regs[decoded.source_gpr + 1] =
                     0xdeadbeef;
                  channel ? ++channels : ++scalars;
                  break;
               }
               case AGX_APPLE9_VIR_STORE_UNIFORM:
                  uniform[I->immediate] = regs[I->src[0]];
                  break;
               default:
                  FAIL() << "unhandled physical IR " << I->op;
               }
            }
            EXPECT_EQ(controls, 1u);
            EXPECT_EQ(channels, 1u);
            EXPECT_EQ(scalars, 1u);
            EXPECT_EQ(uniform[values.size()], 0u);
            for (unsigned i = 0; i < values.size(); ++i)
               EXPECT_EQ(uniform[i], input(i)) << "live value " << i;
            agx_apple9_vir_finish(&p);
            ralloc_free(b.shader);
         }
      }
   }
   EXPECT_GT(count_locations.size(), 1u);
   EXPECT_GT(pair_locations.size(), 1u);
}

TEST(Apple9MeshIR, CountOnlyProducerSurvivesAndDrainsInputs)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh_count_only");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                   AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   agx_apple9_device_load_contract load = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto count = agx_apple9_vir_emit_device_load(&p, 0, zero, &load);
   auto auxiliary = agx_apple9_vir_emit_device_load(&p, 1, zero, &load);
   ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, count, auxiliary, true));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned controls = 0, pending = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason));
      if (I->scoreboard_slot)
         pending &= ~(1u << (I->scoreboard_slot - 1));
      if (I->op == AGX_APPLE9_VIR_WAIT)
         pending &= ~I->immediate;
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         ASSERT_GT(I->producer_scoreboard_slot, 0u);
         pending |= 1u << (I->producer_scoreboard_slot - 1);
      }
      if (I->op == AGX_APPLE9_VIR_MESH_CONTROL) {
         EXPECT_EQ(pending, 0u);
         ++controls;
      }
   }
   EXPECT_EQ(controls, 1u);
   agx_apple9_vir_finish(&p);
   ralloc_free(b.shader);
}

/* The submit must order independent memory work, even when none of its
 * operands comes from that work. Exercise allocation and the physical
 * dependency pass, with zero, one and several occupied slots. */
TEST(Apple9MeshIR, SubmitsDrainIndependentProducers)
{
   for (bool channel : {false, true}) {
      for (unsigned count : {0u, 1u, 2u, 6u}) {
         SCOPED_TRACE(::testing::Message() << "channel=" << channel
                                          << " producers=" << count);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_MESH, &agx_nir_options, "mesh_completion_consumer");
         nir_block logical = {};
         agx_apple9_vir_program p;
         agx_apple9_vir_init(&p);
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical;
         agx_apple9_block_begin(&p, block);
         auto zero = agx_apple9_vir_emit(
            &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
         auto one = agx_apple9_vir_emit(
            &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 1);
         agx_apple9_device_load_contract contract = {
            .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
            .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
         };
         std::vector<uint32_t> values;
         for (unsigned i = 0; i < count; ++i)
            values.push_back(agx_apple9_vir_emit_device_load(
               &p, i + 1, zero, &contract));
         uint32_t pair[] = {one, zero};
         ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
         for (unsigned i = 0; i < count; ++i)
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
               &values[i], 1, i));
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
         unsigned pending = 0, issued = 0, waited = 0, submits = 0;
         for (unsigned i = 0; i < p.instruction_count; ++i) {
            const auto *I = p.instructions[i];
            agx_apple9_packed_instruction packed = {};
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                      &reason)) << reason;
            if (I->scoreboard_slot) {
               unsigned mask = BITFIELD_BIT(I->scoreboard_slot - 1);
               waited |= mask;
               pending &= ~mask;
            }
            if (I->op == AGX_APPLE9_VIR_WAIT) {
               EXPECT_NE(I->immediate, 0u);
               EXPECT_EQ(I->immediate & ~pending, 0u);
               waited |= I->immediate;
               pending &= ~I->immediate;
            }
            if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
               ASSERT_GE(I->producer_scoreboard_slot, 1u);
               ASSERT_LE(I->producer_scoreboard_slot, 6u);
               unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
               pending |= bit;
               issued |= bit;
            }
            /* Every load is first needed by the submit, so they share one
             * counting slot and the submit retires it in its own field. */
            if (I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
                I->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) {
               ++submits;
               EXPECT_EQ(pending, 0u);
               EXPECT_EQ(util_bitcount(issued), count ? 1u : 0u);
               EXPECT_EQ(waited, issued);
               EXPECT_EQ(I->scoreboard_slot != 0, count != 0);
               EXPECT_EQ(I->producer_scoreboard_slot, 0u);
               if (i) {
                  EXPECT_NE(p.instructions[i - 1]->op, AGX_APPLE9_VIR_WAIT);
               }
            }
         }
         EXPECT_EQ(submits, 1u);
         EXPECT_EQ(pending, 0u);
         agx_apple9_vir_finish(&p);
         ralloc_free(b.shader);
      }
   }
}

/* This documents the current handoff contract without treating a pending
 * mesh attribution experiment as proof that every input handoff is broken.
 * The consumed load remains live as an ordinary GPR across the submit. */
TEST(Apple9MeshIR, ConsumedLoadDoesNotCreateASubmitWait)
{
   for (bool channel : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "mesh_after_input_handoff");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      auto zero = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      agx_apple9_device_load_contract contract = {
         .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
         .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      };
      auto loaded = agx_apple9_vir_emit_device_load(&p, 1, zero, &contract);
      uint32_t src[] = {loaded, zero};
      auto value = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
      uint32_t pair[] = {value, zero};
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &loaded, 1, 0));
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
      unsigned slot = 0, handoffs = 0, submits = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                   &reason)) << reason;
         EXPECT_NE(I->op, AGX_APPLE9_VIR_WAIT);
         if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD)
            slot = I->producer_scoreboard_slot;
         if (I->op == AGX_APPLE9_VIR_IXOR) {
            ASSERT_NE(slot, 0u);
            EXPECT_EQ(I->scoreboard_slot, slot);
            ++handoffs;
         }
         if (I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
             I->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) {
            EXPECT_EQ(handoffs, 1u);
            EXPECT_EQ(I->scoreboard_slot, 0u);
            ++submits;
         }
      }
      EXPECT_EQ(handoffs, 1u);
      EXPECT_EQ(submits, 1u);
      agx_apple9_vir_finish(&p);
      ralloc_free(b.shader);
   }
}

/* Exhaust the encoding enum (including PSEUDO) with valid operand contracts
 * for the former opcode-based barrier list. An accepted encoding must carry
 * the scheduling flag, so a new alias cannot silently bypass completion. */
TEST(Apple9MeshEncoding, CompletionBarriersHaveOnlyFlaggedEncodings)
{
   const struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
   } barriers[] = {
      {AGX_APPLE9_VIR_OBJECT_CALL, AGX_APPLE9_ENC_OBJECT_CALL},
      {AGX_APPLE9_VIR_OBJECT_RETURN, AGX_APPLE9_ENC_OBJECT_RETURN},
      {AGX_APPLE9_VIR_OBJECT_RESUME, AGX_APPLE9_ENC_OBJECT_RESUME},
      {AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR, AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR},
      {AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL},
      {AGX_APPLE9_VIR_WORKGROUP_BARRIER, AGX_APPLE9_ENC_WORKGROUP_BARRIER},
      {AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE},
      {AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT},
      {AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS},
      {AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE},
      {AGX_APPLE9_VIR_COVERAGE, AGX_APPLE9_ENC_COVERAGE},
      {AGX_APPLE9_VIR_DEPTH_STORE, AGX_APPLE9_ENC_DEPTH_STORE},
   };
   const uint8_t phys[] = {0, 1};
   for (const auto &barrier : barriers) {
      SCOPED_TRACE(unsigned(barrier.op));
      agx_apple9_vir_instr I = {};
      I.op = barrier.op;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.src[0] = 0;
      I.src[1] = 1;
      switch (I.op) {
      case AGX_APPLE9_VIR_MESH_CONTROL:
         I.src_half_mask = 3;
         FALLTHROUGH;
      case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
      case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL:
         I.nr_srcs = 2;
         break;
      case AGX_APPLE9_VIR_COVERAGE:
         I.nr_srcs = 1;
         break;
      case AGX_APPLE9_VIR_DEPTH_STORE:
         I.nr_srcs = 2;
         I.immediate = 7;
         break;
      case AGX_APPLE9_VIR_TILE_ACCESS:
      case AGX_APPLE9_VIR_TILE_FENCE:
         I.immediate = 1;
         break;
      default:
         break;
      }
      unsigned accepted = 0;
      for (unsigned e = 0; e <= AGX_APPLE9_ENC_COUNT; ++e) {
         I.encoding = e == AGX_APPLE9_ENC_COUNT
            ? AGX_APPLE9_ENC_PSEUDO : agx_apple9_encoding(e);
         SCOPED_TRACE(unsigned(I.encoding));
         agx_apple9_packed_instruction packed = {};
         bool packs = agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr);
         EXPECT_EQ(packs, I.encoding == barrier.encoding);
         if (packs) {
            ASSERT_NE(I.encoding, AGX_APPLE9_ENC_PSEUDO);
            EXPECT_TRUE(agx_apple9_encoding_info(I.encoding)->requires_prior_completion);
            ++accepted;
         }
      }
      EXPECT_EQ(accepted, 1u);
   }
}

/* Exercise the physical dependency pass through normal shared allocation.
 * U0 produces no GPR and must never acquire a fabricated encoded output tag. */
class Apple9MeshCompletion : public ::testing::Test {
protected:
   nir_builder b;
   nir_block logical[8] = {};
   agx_apple9_vir_program p;
   std::vector<agx_apple9_block *> blocks;

   void SetUp() override
   {
      b = nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                         "mesh_fixed_producer");
      agx_apple9_vir_init(&p);
      begin(0);
   }

   void TearDown() override
   {
      agx_apple9_vir_finish(&p);
      ralloc_free(b.shader);
   }

   void begin(unsigned index)
   {
      while (blocks.size() <= index) {
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical[blocks.size()];
         blocks.push_back(block);
      }
      agx_apple9_block_begin(&p, blocks[index]);
   }

   uint32_t imm(unsigned x)
   {
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                               AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, x);
   }

   void control()
   {
      auto count = imm(1), auxiliary = imm(3);
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, count, auxiliary, true));
   }

   void fence()
   {
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
         nullptr, 0, 0));
   }

   void wait(unsigned mask)
   {
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_WAIT, AGX_APPLE9_ENC_WAIT, nullptr, 0, mask));
   }

   void submit(bool channel)
   {
      uint32_t pair[] = {imm(3), imm(0)};
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
   }

   void allocate(unsigned expected_waits)
   {
      allocate(std::vector<unsigned>(expected_waits, 0x01));
   }

   void allocate(std::initializer_list<unsigned> expected_waits)
   {
      allocate(std::vector<unsigned>(expected_waits));
   }

   void allocate(const std::vector<unsigned> &expected_waits)
   {
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
      unsigned waits = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                   &reason)) << reason;
         if (I->op == AGX_APPLE9_VIR_WAIT) {
            ASSERT_LT(waits, expected_waits.size());
            EXPECT_EQ(I->immediate, expected_waits[waits++]);
         }
         if (I->op == AGX_APPLE9_VIR_MESH_CONTROL) {
            EXPECT_EQ(I->dest, AGX_APPLE9_VREG_INVALID);
            EXPECT_EQ(I->dest_components, 0u);
            EXPECT_EQ(I->producer_scoreboard_slot, 1u);
         }
      }
      EXPECT_EQ(waits, expected_waits.size());
   }

   void folded_into(enum agx_apple9_vir_opcode op, unsigned occurrence = 0,
                    unsigned slot = 1)
   {
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         if (I->op != op || occurrence--)
            continue;
         EXPECT_EQ(I->scoreboard_slot, slot);
         return;
      }
      FAIL() << "missing consumer " << op;
   }

   void wait_before(enum agx_apple9_vir_opcode op, unsigned occurrence = 0)
   {
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         if (p.instructions[i]->op != op || occurrence--)
            continue;
         ASSERT_GT(i, 0u);
         EXPECT_EQ(p.instructions[i - 1]->op, AGX_APPLE9_VIR_WAIT);
         EXPECT_EQ(p.instructions[i - 1]->immediate, 0x01u);
         return;
      }
      FAIL() << "missing consumer " << op;
   }

   void rejected(const char *expected)
   {
      const char *reason = nullptr;
      EXPECT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
      EXPECT_STREQ(reason, expected);
      EXPECT_FALSE(agx_apple9_allocate_shared(&p, b.shader, &reason));
      EXPECT_STREQ(reason, expected);
      EXPECT_EQ(p.phys, nullptr);
   }
};

TEST_F(Apple9MeshCompletion, FenceRetiresResultlessControl)
{
   control();
   fence();
   allocate(1);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, SubmitsRetireControlOnlyOnce)
{
   control();
   submit(true);
   submit(false);
   allocate(0);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR, 0, 0);
}

TEST_F(Apple9MeshCompletion, ScalarSubmitRetiresControl)
{
   control();
   submit(false);
   allocate(0);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR);
}

TEST_F(Apple9MeshCompletion, AlreadyRetiredControlDoesNotWaitAgain)
{
   control();
   fence();
   begin(1);
   submit(true);
   fence();
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, RepeatedControlIsACompileError)
{
   control();
   control();
   submit(false);
   rejected("Apple9 multiple MESH_CONTROL instructions are unsafe");
}

TEST_F(Apple9MeshCompletion, FullyWaitedRepeatedControlIsACompileError)
{
   control();
   wait(0x3f);
   control();
   wait(0x3f);
   rejected("Apple9 multiple MESH_CONTROL instructions are unsafe");
}

TEST_F(Apple9MeshCompletion, ControlInBothBranchArmsIsACompileError)
{
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   control();
   begin(2);
   control();
   begin(3);
   submit(false);
   rejected("Apple9 multiple MESH_CONTROL instructions are unsafe");
}

TEST_F(Apple9MeshCompletion, RetiredControlInBothBranchArmsIsACompileError)
{
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   control();
   fence();
   begin(2);
   control();
   fence();
   begin(3);
   submit(false);
   rejected("Apple9 multiple MESH_CONTROL instructions are unsafe");
}

TEST_F(Apple9MeshCompletion, SkippedChannelScopeRetiresAtScalarJoin)
{
   control();
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[2];
   begin(1);
   submit(true);
   begin(2);
   submit(false);
   allocate(0);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR);
}

TEST_F(Apple9MeshCompletion, JoinRetiresThePathBypassingAnEarlierFence)
{
   control();
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   fence();
   begin(2);
   begin(3);
   fence();
   submit(false);
   allocate(2);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE, 1);
}

TEST_F(Apple9MeshCompletion, ControlInBackedgeIsACompileError)
{
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[1];
   logical[1].successors[1] = &logical[2];
   begin(1);
   control();
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_JMP_EXEC_ANY, AGX_APPLE9_ENC_JMP_EXEC_ANY,
      nullptr, 0, 0));
   p.instructions[p.instruction_count - 1]->branch_target = blocks[1];
   begin(2);
   submit(false);
   rejected("Apple9 MESH_CONTROL inside a loop is unsafe");
}

TEST_F(Apple9MeshCompletion, ControlInNestedNirLoopIsACompileError)
{
   nir_loop *loop = nir_push_loop(&b);
   nir_push_if(&b, nir_imm_true(&b));
   blocks[0]->nir = nir_cursor_current_block(b.cursor);
   control();
   nir_pop_if(&b, nullptr);
   nir_pop_loop(&b, loop);
   rejected("Apple9 MESH_CONTROL inside a loop is unsafe");
}

TEST_F(Apple9MeshCompletion, ControlBeforeLoopRemainsValid)
{
   control();
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[1];
   logical[1].successors[1] = &logical[2];
   begin(1);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_JMP_EXEC_ANY, AGX_APPLE9_ENC_JMP_EXEC_ANY,
      nullptr, 0, 0));
   p.instructions[p.instruction_count - 1]->branch_target = blocks[1];
   begin(2);
   submit(true);
   allocate(1);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, 0, 0);
}

TEST_F(Apple9MeshCompletion, FixedProducerSlotRetiresControl)
{
   control();
   auto value = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_GET_SR, AGX_APPLE9_ENC_GET_DRAW_ID, nullptr, 0, 0x10dd);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &value, 1, 0));
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_GET_SR);
}

TEST_F(Apple9MeshCompletion, PendingControlExcludesSlotsOneAndTwo)
{
   control();
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t values[4];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_apple9_vir_emit_device_load(&p, i + 1, index, &contract);
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, i));
   fence();
   allocate(1);
   unsigned slots = 0, loads = 0;
   bool retired = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      retired |= I->op == AGX_APPLE9_VIR_WAIT;
      if (I->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      EXPECT_FALSE(retired);
      ASSERT_GE(I->producer_scoreboard_slot, 3u);
      ASSERT_LE(I->producer_scoreboard_slot, 6u);
      unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
      EXPECT_EQ(slots & bit, 0u);
      slots |= bit;
      ++loads;
   }
   EXPECT_EQ(loads, 4u);
   EXPECT_EQ(slots, 0x3cu);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

/* U0 reserves slots 1 and 2; once 3..6 are taken, a further load joins one
 * of those instead of retiring U0 to reuse its selectors. */
TEST_F(Apple9MeshCompletion, ExhaustionJoinsLoadsWithoutReusingControlSlots)
{
   control();
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t values[5];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_apple9_vir_emit_device_load(&p, i + 1, index, &contract);
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, i));
   fence();
   allocate(1);
   unsigned slots = 0, loads = 0, waits = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      waits += I->op == AGX_APPLE9_VIR_WAIT;
      if (I->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      EXPECT_EQ(waits, 0u);
      ASSERT_GE(I->producer_scoreboard_slot, 3u);
      ASSERT_LE(I->producer_scoreboard_slot, 6u);
      unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
      EXPECT_EQ(slots & bit, loads < 4 ? 0u : bit);
      slots |= bit;
      ++loads;
   }
   EXPECT_EQ(slots, 0x3cu);
   EXPECT_EQ(loads, 5u);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, JoinedSlotOneResultAndControlFoldTogether)
{
   /* The load dominates both arms. One arm bypasses U0 and keeps its result
    * pending; the other retires it at U0 and issues the fixed producer.
    * The join must satisfy both possible obligations before reading it. */
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto first = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   auto joined = agx_apple9_vir_emit_device_load(&p, 2, index, &contract);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &first, 1, 0));
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   begin(2);
   control();
   begin(3);
   uint32_t src[] = {joined, imm(0x1234)};
   auto result = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &result, 1, 1));
   fence();
   allocate({0x01});
   unsigned loads = 0, consumers = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(I->producer_scoreboard_slot, loads++ ? 1u : 6u);
      }
      if (I->op == AGX_APPLE9_VIR_IXOR) {
         EXPECT_EQ(I->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         ++consumers;
      }
   }
   EXPECT_EQ(loads, 2u);
   EXPECT_EQ(consumers, 1u);
   folded_into(AGX_APPLE9_VIR_IXOR);
}

TEST_F(Apple9MeshCompletion, EndOfProgramDrainsControl)
{
   control();
   allocate(1);
   ASSERT_GT(p.instruction_count, 0u);
   EXPECT_EQ(p.instructions[p.instruction_count - 1]->op, AGX_APPLE9_VIR_WAIT);
}

TEST_F(Apple9MeshCompletion, StoreAndForwardExecScopeLeaveControlPending)
{
   control();
   auto index = imm(0), value = imm(123);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &value, 1, 32));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_PUSH, AGX_APPLE9_ENC_EXEC_MASK_PUSH,
      nullptr, 0, AGX_APPLE9_EXEC_MASK_PREDICATE(0)));
   begin(1);
   submit(true);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_EXEC_MASK_POP, AGX_APPLE9_ENC_EXEC_MASK_POP,
      nullptr, 0, 0));
   allocate(0);
   folded_into(AGX_APPLE9_VIR_DEVICE_STORE, 0, 0);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL);
}

TEST_F(Apple9MeshCompletion, HaltBeforeSubmitRetiresControl)
{
   control();
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, nullptr, 0, 0));
   allocate(1);
   wait_before(AGX_APPLE9_VIR_HALT);
}

TEST_F(Apple9MeshCompletion, ExplicitSlotOneWaitRetiresControl)
{
   control();
   wait(1);
   submit(true);
   allocate({1});
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, 0, 0);
}

TEST_F(Apple9MeshCompletion, ExplicitSlotTwoWaitRetiresControl)
{
   control();
   wait(2);
   submit(true);
   allocate({2});
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, 0, 0);
}

TEST_F(Apple9MeshCompletion, WrongSlotWaitLeavesControlPending)
{
   control();
   wait(4);
   submit(true);
   allocate({4});
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL);
}

TEST_F(Apple9MeshCompletion, JoinedSlotOneLoadSurvivesSlotTwoControlWait)
{
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto first = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   logical[0].successors[0] = &logical[1];
   begin(1);
   auto joined = agx_apple9_vir_emit_device_load(&p, 2, index, &contract);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &first, 1, 0));
   logical[1].successors[0] = &logical[2];
   logical[1].successors[1] = &logical[3];
   logical[2].successors[0] = &logical[4];
   logical[3].successors[0] = &logical[4];
   begin(2);
   begin(3);
   control();
   begin(4);
   wait(2);
   uint32_t src[] = {joined, imm(0x1234)};
   auto result = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &result, 1, 1));
   submit(true);
   allocate({1, 2});
   folded_into(AGX_APPLE9_VIR_IXOR);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, 0, 0);
}

TEST_F(Apple9MeshCompletion, JoinedSlotTwoLoadSurvivesSlotOneControlWait)
{
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t values[3];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_apple9_vir_emit_device_load(&p, i + 1, index, &contract);
   for (unsigned i = 0; i < 2; ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, i));
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   begin(2);
   control();
   begin(3);
   wait(1);
   uint32_t src[] = {values[2], imm(0x1234)};
   auto result = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &result, 1, 2));
   submit(true);
   allocate({2, 1});
   folded_into(AGX_APPLE9_VIR_IXOR, 0, 2);
   folded_into(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, 0, 0);
}

TEST(Apple9MeshIR, AuxiliaryClassRetainsLiveValueAndDetectsClobberMutation)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh_auxiliary_preservation");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                    AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   agx_apple9_device_load_contract load = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto a = agx_apple9_vir_emit_device_load(&p, 0, zero, &load);
   auto v = agx_apple9_vir_emit_device_load(&p, 1, zero, &load);
   ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, a, v, true));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM, &v, 1, 0));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason)) << reason;

   /* Evaluate the actual allocated instructions twice. Only the second run
    * mutates U0's B effect to a destructive one. This must be observable: a
    * hidden allocator clobber copy would incorrectly mask the mutation. */
   uint32_t regs[2][AGX_APPLE9_GPR_COUNT] = {};
   uint32_t outputs[2] = {};
   unsigned controls = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto *I = p.instructions[i];
      for (unsigned m = 0; m < 2; ++m) {
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[m][I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_DEVICE_LOAD:
            regs[m][I->dest] = I->immediate ? 3 : 1;
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[m][I->dest] = regs[m][I->src[0]] | regs[m][I->src[1]];
            break;
         case AGX_APPLE9_VIR_IXOR:
            regs[m][I->dest] = regs[m][I->src[0]] ^ regs[m][I->src[1]];
            break;
         case AGX_APPLE9_VIR_WAIT:
            break;
         case AGX_APPLE9_VIR_MESH_CONTROL:
            EXPECT_EQ(I->live_after_mask, 2u);
            EXPECT_EQ(regs[m][I->src[1]], 3u);
            EXPECT_NE(I->src[0], I->src[1]);
            regs[m][I->src[0]] &= 0xffff0000u;
            if (m)
               regs[m][I->src[1]] = 0xdeadbeef;
            else {
               ++controls;
               /* An out-of-class physical operand must still be rejected. */
               auto saved = I->src[1];
               I->src[1] = AGX_APPLE9_GPR_COUNT;
               EXPECT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
               I->src[1] = saved;
               EXPECT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason));
            }
            break;
         case AGX_APPLE9_VIR_STORE_UNIFORM:
            outputs[m] = regs[m][I->src[0]];
            break;
         default:
            FAIL() << "unhandled allocated opcode " << I->op;
         }
      }
   }
   EXPECT_EQ(controls, 1u);
   EXPECT_EQ(outputs[0], 3u);
   EXPECT_EQ(outputs[1], 0xdeadbeefu);
   agx_apple9_vir_finish(&p);
   ralloc_free(b.shader);
}

TEST(Apple9MemEncoding, MultiSlotSharedPrivateAndSharedAtomicSlots1To6)
{
   const uint8_t phys[] = {4, 5, 6, 7};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   /* SHARED_LOAD: byte 9 = 0x08 | (slot - 1) for slots 1..6. */
   for (unsigned slot = 1; slot <= 6; ++slot) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_SHARED_LOAD;
      I.encoding = AGX_APPLE9_ENC_SHARED_LOAD;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.immediate = 2;
      I.producer_scoreboard_slot = slot;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 14u);
      EXPECT_EQ(packed.bytes[0], 0x67u);
      EXPECT_EQ(packed.bytes[8], 0x44u);
      EXPECT_EQ(packed.bytes[9], 0x08u | (slot - 1));
      EXPECT_EQ(packed.bytes[11], 0xc0u);
      EXPECT_EQ(packed.bytes[12], 0x08u);
   }
   for (unsigned bad : {0u, 7u}) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_SHARED_LOAD;
      I.encoding = AGX_APPLE9_ENC_SHARED_LOAD;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.producer_scoreboard_slot = bad;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   }

   /* PRIVATE_LOAD: byte 6 = 0x09 | ((s & 3) << 6), byte 7 = s >> 2. */
   for (unsigned slot = 1; slot <= 6; ++slot) {
      unsigned s = slot - 1;
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_PRIVATE_LOAD;
      I.encoding = AGX_APPLE9_ENC_PRIVATE_LOAD;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.immediate = 12;
      I.producer_scoreboard_slot = slot;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(packed.bytes[0], 0x6fu);
      EXPECT_EQ(packed.bytes[6], (uint8_t)(0x09u | ((s & 3u) << 6)));
      EXPECT_EQ(packed.bytes[7], (uint8_t)(s >> 2));
   }
   for (unsigned bad : {0u, 7u}) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_PRIVATE_LOAD;
      I.encoding = AGX_APPLE9_ENC_PRIVATE_LOAD;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.producer_scoreboard_slot = bad;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   }

   /* SHARED_ATOMIC: byte 8 = 0x10 | (slot << 1) for slots 1..6. */
   for (unsigned slot = 1; slot <= 6; ++slot) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_SHARED_ATOMIC;
      I.encoding = AGX_APPLE9_ENC_SHARED_ATOMIC;
      I.dest = 0;
      I.dest_components = 1;
      I.memory_components = 1;
      I.atomic_op = AGX_APPLE9_ATOMIC_ADD;
      I.nr_srcs = 2;
      I.src[0] = 1;
      I.src[1] = 2;
      I.immediate = 2;
      I.producer_scoreboard_slot = slot;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(packed.bytes[0], 0x67u);
      EXPECT_EQ(packed.bytes[8], (uint8_t)(0x10u | (slot << 1)));
   }
   for (unsigned bad : {0u, 7u}) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_SHARED_ATOMIC;
      I.encoding = AGX_APPLE9_ENC_SHARED_ATOMIC;
      I.dest = 0;
      I.dest_components = 1;
      I.memory_components = 1;
      I.atomic_op = AGX_APPLE9_ATOMIC_ADD;
      I.nr_srcs = 2;
      I.src[0] = 1;
      I.src[1] = 2;
      I.producer_scoreboard_slot = bad;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   }
}

TEST(Apple9MemEncoding, VectorSharedLoadAndStoreUint2Uint4)
{
   /* phys: data tuple at r4..r7 (or r3..r4), index at r8. */
   const uint8_t phys[] = {4, 5, 6, 7, 8};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   for (unsigned comps : {2u, 4u}) {
      const uint8_t exp_b8 = comps == 2 ? 0x64u : 0x5cu;
      const uint8_t exp_ld11 = 0x00u;
      const uint8_t exp_ld12 = comps == 2 ? 0x09u : 0x08u;
      const uint8_t exp_st11 = comps == 2 ? 0x40u : 0x00u;
      const uint8_t exp_st12 = 0x02u;

      for (unsigned slot = 1; slot <= 6; ++slot) {
         agx_apple9_vir_instr ld = {};
         ld.op = AGX_APPLE9_VIR_SHARED_LOAD;
         ld.encoding = AGX_APPLE9_ENC_SHARED_LOAD;
         ld.dest = 0;
         ld.dest_components = comps;
         ld.memory_components = comps;
         ld.nr_srcs = 1;
         ld.src[0] = 4; /* r8 */
         ld.immediate = 2;
         ld.producer_scoreboard_slot = slot;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ld, phys, &packed, &reason))
            << (reason ?: "");
         ASSERT_EQ(packed.length, 14u);
         EXPECT_EQ(packed.bytes[0], 0x67u);
         EXPECT_EQ(packed.bytes[3], 4u << 1);
         EXPECT_EQ(packed.bytes[4], 2u << 2);
         EXPECT_EQ(packed.bytes[5], 8u << 2);
         EXPECT_EQ(packed.bytes[6], 0x80u); /* index last use */
         EXPECT_EQ(packed.bytes[8], exp_b8);
         EXPECT_EQ(packed.bytes[9], 0x08u | (slot - 1));
         EXPECT_EQ(packed.bytes[11], exp_ld11);
         EXPECT_EQ(packed.bytes[12], exp_ld12);
      }

      agx_apple9_vir_instr st = {};
      st.op = AGX_APPLE9_VIR_SHARED_STORE;
      st.encoding = AGX_APPLE9_ENC_SHARED_STORE;
      st.dest = AGX_APPLE9_VREG_INVALID;
      st.memory_components = comps;
      st.nr_srcs = comps + 1;
      for (unsigned c = 0; c < comps; ++c)
         st.src[c] = c;
      st.src[comps] = 4; /* r8 */
      st.immediate = 2;
      st.live_after_mask = BITFIELD_BIT(comps); /* index retained */
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&st, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 14u);
      EXPECT_EQ(packed.bytes[0], 0xe7u);
      EXPECT_EQ(packed.bytes[3], 4u << 1);
      EXPECT_EQ(packed.bytes[4], 2u << 2);
      EXPECT_EQ(packed.bytes[5], 8u << 2);
      EXPECT_EQ(packed.bytes[6], 0x00u); /* index retained */
      EXPECT_EQ(packed.bytes[8], exp_b8);
      EXPECT_EQ(packed.bytes[9], 0x02u);
      EXPECT_EQ(packed.bytes[11], exp_st11);
      EXPECT_EQ(packed.bytes[12], exp_st12);

      /* Reject non-adjacent data tuple on store. */
      uint8_t bad_phys[] = {4, 6, 7, 8, 9};
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&st, bad_phys, &packed, &reason));
      /* Accept high GPR tuple in r64..r95 with high index register. */
      uint8_t high_phys[] = {72, 73, 74, 75, 75};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&st, high_phys, &packed, &reason))
         << (reason ?: "");
      EXPECT_EQ(packed.bytes[3], 72u << 1);
      EXPECT_EQ(packed.bytes[5], (75u & 63u) << 2);
      EXPECT_EQ(packed.bytes[6], 0x01u); /* bit 0 = index>>6, retained */

      /* Reject overflowing r95 boundary. */
      uint8_t overflow_phys[] = {95, 96, 97, 98, 8};
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&st, overflow_phys, &packed, &reason));
   }
}

TEST(Apple9MemEncoding, NativeImageWrite3DCube16BitFragmentAndFences)
{
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   /* Test 2D, 3D, Cube, 16-bit, Fragment, and publication base p40. */
   struct {
      enum agx_apple9_encoding enc;
      unsigned dim;
      unsigned bits;
      unsigned integer;
      bool fragment;
      unsigned base;
      uint8_t exp_b5;
      uint8_t exp_b9;
      uint8_t exp_b13;
      uint8_t exp_b14;
   } cases[] = {
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 32, 0, false, 40, 0x40, 0x04, 0x3a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 16, 0, false, 32, 0x40, 0x04, 0x2a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 16, 1, false, 24, 0x40, 0x04, 0x0a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 3, 32, 0, false, 16, 0x40, 0x08, 0x3a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 1, 32, 0, false, 8,  0x40, 0x0c, 0x3a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 32, 0, true,  0,  0x40, 0x04, 0x3a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 3, 32, 0, true,  40, 0x40, 0x08, 0x3a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 16, 0, true,  0,  0x40, 0x04, 0x2a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE, 0, 16, 1, true,  0,  0x40, 0x04, 0x0a, 0x89},
      {AGX_APPLE9_ENC_IMAGE_WRITE_LAYERED, 0, 32, 1, true, 16, 0x00, 0x04, 0x1a, 0x89},
   };
   for (const auto &c : cases) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_IMAGE_STORE;
      I.encoding = c.enc;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.nr_srcs = 10;
      I.texture_dimension = c.dim;
      I.memory_bits = c.bits;
      I.immediate = c.integer;
      I.image_fragment = c.fragment;
      I.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_2;
      uint8_t phys[10];
      for (unsigned i = 0; i < 8; ++i) {
         I.src[i] = i;
         phys[i] = c.base + i;
      }
      I.src[8] = 8;
      I.src[9] = 9;
      phys[8] = 10;
      phys[9] = 11;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 16u);
      EXPECT_EQ(packed.bytes[0], 0xd7u);
      EXPECT_EQ(packed.bytes[3], (uint8_t)(c.base << 1));
      EXPECT_EQ(packed.bytes[5], c.exp_b5);
      EXPECT_EQ(packed.bytes[9], c.exp_b9);
      EXPECT_EQ(packed.bytes[12], 0x89u);
      EXPECT_EQ(packed.bytes[13], c.exp_b13);
      EXPECT_EQ(packed.bytes[14], c.exp_b14);
   }

   /* 16-bit 2D parameter tuple packs 4 moves (40 bytes) up to base p40. */
   {
      agx_apple9_vir_instr p16 = {};
      p16.op = AGX_APPLE9_VIR_PUBLICATION_TUPLE;
      p16.encoding = AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_16;
      p16.dest = 0;
      p16.dest_components = 8;
      p16.nr_srcs = 4;
      for (unsigned i = 0; i < 4; ++i)
         p16.src[i] = i + 1;
      uint8_t phys[] = {40, 2, 3, 4, 5};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&p16, phys, &packed, &reason))
         << (reason ?: "");
      EXPECT_EQ(packed.length, 40u);
      phys[0] = 48;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&p16, phys, &packed, &reason));
   }

   /* Fragment, workgroup, and multi-slot texture fences (bits 12..17). */
   const struct {
      enum agx_apple9_fence_mode mode;
      uint8_t slot_mask;
      uint8_t expected[6];
   } fences[] = {
      {AGX_APPLE9_FENCE_IMAGE_WRITE, 0x00, {0x07, 0x14, 0x54, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE, 0x03, {0x07, 0x34, 0x54, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE, 0x1e, {0x07, 0xe4, 0x55, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE, 0x3f, {0x07, 0xf4, 0x57, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE_FS, 0x00, {0x07, 0x14, 0x64, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE_FS, 0x02, {0x07, 0x24, 0x64, 0x50, 0x06, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP, 0x00, {0x07, 0x14, 0x54, 0x51, 0x0e, 0x00}},
      {AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP, 0x06, {0x07, 0x64, 0x54, 0x51, 0x0e, 0x00}},
      {AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP, 0x00, {0x07, 0x04, 0x54, 0xd1, 0x0e, 0x00}},
   };
   const uint8_t zero_phys[] = {0};
   for (const auto &f : fences) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_DEVICE_FENCE;
      I.encoding = AGX_APPLE9_ENC_DEVICE_FENCE;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.immediate = f.mode;
      I.fence_slot_mask = f.slot_mask;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, zero_phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 6u);
      EXPECT_EQ(memcmp(packed.bytes, f.expected, 6), 0);
   }
}

TEST(Apple9MemEncoding, FullDimensionMinLodClampModes)
{
   uint8_t phys[16];
   for (unsigned i = 0; i < 16; ++i)
      phys[i] = i;
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   /* Offset int2(1, -2): x=1, y=0xe (-2 in 4-bit signed) -> byte 9=0x08, byte 13=0x0e. */
   const uint16_t ofs_1_m2 = (1u & 0xfu) | ((0xeu & 0xfu) << 4);

   const struct {
      bool gradient;
      unsigned dim;
      bool bias;
      bool shadow;
      bool has_offset;
      uint16_t expected_mode;
   } cases[] = {
      {false, 3, false, false, false, 0x2180}, /* 3D min_lod */
      {false, 3, true,  false, false, 0x2190}, /* 3D bias+min_lod */
      {true,  3, false, false, false, 0x219c}, /* 3D grad+min_lod */
      {false, 1, false, false, false, 0x0810}, /* Cube min_lod */
      {false, 1, true,  false, false, 0x0814}, /* Cube bias+min_lod */
      {false, 4, false, false, false, 0x0c10}, /* Cube-array min_lod */
      {false, 0, false, true,  false, 0x0820}, /* Depth2D compare min_lod */
      {false, 0, true,  true,  false, 0x0824}, /* Depth2D compare bias+min_lod */
      {false, 2, false, true,  false, 0x0c20}, /* Depth2D-array compare min_lod */
      {false, 0, true,  false, true,  0x02c1}, /* 2D bias+min_lod+offset */
      {false, 2, true,  false, true,  0x03c1}, /* 2D-array bias+min_lod+offset */
   };

   for (const auto &c : cases) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_TEXTURE_SAMPLE;
      I.encoding = c.gradient ? AGX_APPLE9_ENC_TEXTURE_GRAD
                              : AGX_APPLE9_ENC_TEXTURE_LOD;
      I.dest = 4;
      I.dest_components = 4;
      I.nr_srcs = c.gradient ? 1 : 4;
      for (unsigned s = 0; s < I.nr_srcs; ++s)
         I.src[s] = s;
      I.texture_dimension = c.dim;
      I.immediate = c.bias ? 1 : 0;
      I.texture_shadow = c.shadow;
      I.texture_min_lod = true;
      I.texture_has_offset = c.has_offset;
      I.texture_offset = c.has_offset ? ofs_1_m2 : 0;
      I.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;

      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << "mode 0x" << std::hex << c.expected_mode << ": " << (reason ?: "");
      ASSERT_EQ(packed.length, 14u);
      uint16_t got_mode = packed.bytes[6] | (uint16_t(packed.bytes[7]) << 8);
      EXPECT_EQ(got_mode, c.expected_mode);
      EXPECT_EQ(packed.bytes[10], c.shadow ? 0x00u : 0x10u);
      if (c.has_offset) {
         EXPECT_EQ(packed.bytes[9], 0x08u);
         EXPECT_EQ(packed.bytes[13], 0x0eu);
      } else {
         EXPECT_EQ(agx_apple9_texture_mode(!c.gradient, c.gradient, c.bias,
                                           c.dim, c.shadow, true),
                   c.expected_mode);
      }
   }

   /* Verify agx_apple9_texture_offset_mode(0x0804) -> 0x02c1. */
   uint16_t ofs_mode = 0;
   EXPECT_TRUE(agx_apple9_texture_offset_mode(0x0804, &ofs_mode));
   EXPECT_EQ(ofs_mode, 0x02c1u);

   /* Verify 3D grad + min_lod parameter tuple packs 10 words (100 bytes). */
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                       AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t grad_srcs[10];
   for (unsigned i = 0; i < 10; ++i)
      grad_srcs[i] = zero;
   uint32_t res = agx_apple9_vir_emit_texture_grad(&p, grad_srcs, 0, 0, 3,
                                                   false, true);
   ASSERT_NE(res, AGX_APPLE9_VREG_INVALID);
   const agx_apple9_vir_instr *params = p.instructions[p.instruction_count - 2];
   EXPECT_EQ(params->encoding, AGX_APPLE9_ENC_TEXTURE_GRAD_VOLUME_MIN_LOD_PARAMS);
   EXPECT_EQ(params->nr_srcs, 10u);
   EXPECT_EQ(params->dest_components, 16u);
   uint8_t pphys[32] = {};
   pphys[zero] = 2;
   for (unsigned i = 0; i < 16; ++i)
      pphys[params->dest + i] = i;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(params, pphys, &packed, &reason))
      << (reason ?: "");
   EXPECT_EQ(packed.length, 100u);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9MemEncoding, NarrowAndVectorSharedMemoryAndHighGprs)
{
   struct {
      unsigned comps;
      unsigned bits;
      unsigned gprs;
      uint8_t b8;
      uint8_t ld11;
      uint8_t ld12;
      uint8_t st11;
      uint8_t st12;
   } cases[] = {
      {1, 8,  1, 0x84, 0x40, 0x08, 0x10, 0x02},
      {1, 16, 1, 0x04, 0x80, 0x08, 0x20, 0x02},
      {2, 16, 1, 0x24, 0xc0, 0x08, 0x30, 0x02},
      {4, 16, 2, 0x1c, 0x00, 0x09, 0x40, 0x02},
      {1, 32, 1, 0x44, 0xc0, 0x08, 0x30, 0x02},
      {2, 32, 2, 0x64, 0x00, 0x09, 0x40, 0x02},
      {3, 32, 3, 0x74, 0x00, 0x08, 0x00, 0x02},
      {4, 32, 4, 0x5c, 0x00, 0x08, 0x00, 0x02},
   };
   const uint8_t phys[] = {72, 73, 74, 75, 75};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   for (const auto &c : cases) {
      agx_apple9_vir_instr ld = {};
      ld.op = AGX_APPLE9_VIR_SHARED_LOAD;
      ld.encoding = AGX_APPLE9_ENC_SHARED_LOAD;
      ld.dest = 0;
      ld.dest_components = c.gprs;
      ld.memory_bits = c.bits == 32 ? 0 : c.bits;
      ld.memory_components = c.comps;
      ld.nr_srcs = 1;
      ld.src[0] = 4; /* r75 */
      ld.immediate = 2;
      ld.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ld, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 14u);
      EXPECT_EQ(packed.bytes[0], 0x67u);
      EXPECT_EQ(packed.bytes[1], 0x02u);
      EXPECT_EQ(packed.bytes[2], 0x54u);
      EXPECT_EQ(packed.bytes[3], 0x90u); /* r72 << 1 */
      EXPECT_EQ(packed.bytes[4], 0x08u); /* root 2 << 2 */
      EXPECT_EQ(packed.bytes[5], 0x2cu); /* (75 & 63) << 2 */
      EXPECT_EQ(packed.bytes[6], 0x81u); /* discard | (75 >> 6) */
      EXPECT_EQ(packed.bytes[8], c.b8);
      EXPECT_EQ(packed.bytes[9], 0x0du); /* slot 6 */
      EXPECT_EQ(packed.bytes[11], c.ld11);
      EXPECT_EQ(packed.bytes[12], c.ld12);

      agx_apple9_vir_instr st = {};
      st.op = AGX_APPLE9_VIR_SHARED_STORE;
      st.encoding = AGX_APPLE9_ENC_SHARED_STORE;
      st.dest = AGX_APPLE9_VREG_INVALID;
      st.memory_bits = c.bits == 32 ? 0 : c.bits;
      st.memory_components = c.comps;
      st.nr_srcs = c.gprs + 1;
      for (unsigned k = 0; k < c.gprs; ++k)
         st.src[k] = k;
      st.src[c.gprs] = 4; /* r75 */
      st.immediate = 2;
      st.live_after_mask = BITFIELD_BIT(c.gprs);
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&st, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 14u);
      EXPECT_EQ(packed.bytes[0], 0xe7u);
      EXPECT_EQ(packed.bytes[1], 0x02u);
      EXPECT_EQ(packed.bytes[2], 0x54u);
      EXPECT_EQ(packed.bytes[3], 0x90u);
      EXPECT_EQ(packed.bytes[4], 0x08u);
      EXPECT_EQ(packed.bytes[5], 0x2cu);
      EXPECT_EQ(packed.bytes[6], 0x01u); /* retained | (75 >> 6) */
      EXPECT_EQ(packed.bytes[8], c.b8);
      EXPECT_EQ(packed.bytes[9], 0x02u);
      EXPECT_EQ(packed.bytes[11], c.st11);
      EXPECT_EQ(packed.bytes[12], c.st12);
   }

   /* Shared atomic with high GPRs r76 = atomic_add(root=2, idx=r75, data=r73). */
   const uint8_t aphys[] = {76, 73, 75};
   agx_apple9_vir_instr at = {};
   at.op = AGX_APPLE9_VIR_SHARED_ATOMIC;
   at.encoding = AGX_APPLE9_ENC_SHARED_ATOMIC;
   at.dest = 0;
   at.dest_components = 1;
   at.memory_components = 1;
   at.atomic_op = AGX_APPLE9_ATOMIC_ADD;
   at.nr_srcs = 2;
   at.src[0] = 1;
   at.src[1] = 2;
   at.immediate = 2;
   at.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_5;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&at, aphys, &packed, &reason))
      << (reason ?: "");
   ASSERT_EQ(packed.length, 12u);
   EXPECT_EQ(packed.bytes[3], 76u);
   EXPECT_EQ(packed.bytes[4], 4u);
   EXPECT_EQ(packed.bytes[5], 73u);
   EXPECT_EQ(packed.bytes[6], 75u);
   EXPECT_EQ(packed.bytes[8], 0x1au);
}

TEST(Apple9MemEncoding, PrivateLoadStoreNarrowAndWideWithWait)
{
   struct {
      unsigned comps;
      unsigned bits;
      unsigned offset;
      uint8_t fmt;
      uint8_t ld_b9;
      uint8_t ld_b10;
      uint8_t st_b7;
      uint8_t st_b8;
      uint8_t st_b9;
   } cases[] = {
      {1, 8,  1,    0x21, 0x00, 0x02, 0x40, 0x00, 0x40},
      {1, 16, 0x11, 0x01, 0x80, 0x02, 0x40, 0x04, 0x50},
      {1, 32, 1,    0x09, 0x00, 0x03, 0x43, 0x00, 0x60},
      {2, 32, 2,    0x19, 0x80, 0x03, 0x80, 0x00, 0x70},
      {4, 32, 4,    0x17, 0x00, 0x02, 0x00, 0x01, 0x40},
   };
   const uint8_t phys[] = {4, 5, 6, 7, 1};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   for (const auto &c : cases) {
      agx_apple9_vir_instr ld = {};
      ld.op = AGX_APPLE9_VIR_PRIVATE_LOAD;
      ld.encoding = AGX_APPLE9_ENC_PRIVATE_LOAD;
      ld.dest = 0;
      ld.dest_components = c.comps;
      ld.memory_bits = c.bits == 32 ? 0 : c.bits;
      ld.memory_components = c.comps;
      ld.nr_srcs = 1;
      ld.src[0] = 4; /* r1 */
      ld.immediate = c.offset;
      ld.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      ld.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ld, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(packed.bytes[0], 0x6fu);
      EXPECT_EQ(packed.bytes[1], 0x10u); /* wait slot 1 */
      EXPECT_EQ(packed.bytes[2], 0x54u);
      EXPECT_EQ(packed.bytes[3], 8u);
      EXPECT_EQ(packed.bytes[4], 2u);
      EXPECT_EQ(packed.bytes[6], uint8_t(c.fmt | 0x40u)); /* slot 6: s=5 */
      EXPECT_EQ(packed.bytes[7], 0x01u);
      EXPECT_EQ(packed.bytes[8], c.offset & 0xffu);
      EXPECT_EQ(packed.bytes[9], c.ld_b9);
      EXPECT_EQ(packed.bytes[10], c.ld_b10);

      agx_apple9_vir_instr st = {};
      st.op = AGX_APPLE9_VIR_PRIVATE_STORE;
      st.encoding = AGX_APPLE9_ENC_PRIVATE_STORE;
      st.dest = AGX_APPLE9_VREG_INVALID;
      st.memory_bits = c.bits == 32 ? 0 : c.bits;
      st.memory_components = c.comps;
      st.nr_srcs = c.comps + 1;
      for (unsigned k = 0; k < c.comps; ++k)
         st.src[k] = k;
      st.src[c.comps] = 4; /* r1 */
      st.immediate = c.offset;
      st.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_2;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&st, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(packed.bytes[0], 0xefu);
      EXPECT_EQ(packed.bytes[1], 0x20u); /* wait slot 2 */
      EXPECT_EQ(packed.bytes[2], 0x54u);
      EXPECT_EQ(packed.bytes[6], c.fmt);
      EXPECT_EQ(packed.bytes[7], c.st_b7);
      EXPECT_EQ(packed.bytes[8], c.st_b8);
      EXPECT_EQ(packed.bytes[9], c.st_b9);
   }
}

TEST(Apple9MemEncoding, DeviceLoadStoreVectorU16)
{
   agx_apple9_packed_instruction packed = {};

   /* probe1_mem p_dev_u8x4 (i16x2 load & store) */
   ASSERT_TRUE(agx_apple9_pack_device_load_vector_u16_raw(
      1, 0, 1, 2, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_1, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      &packed));
   const uint8_t exp_ld2[14] = {0x67, 0x10, 0x44, 0x02, 0x01, 0x00, 0x20,
                                0x00, 0x49, 0x01, 0x00, 0x40, 0x46, 0x00};
   ASSERT_EQ(packed.length, 14u);
   EXPECT_EQ(memcmp(packed.bytes, exp_ld2, 14), 0);

   ASSERT_TRUE(agx_apple9_pack_device_store_vector_u16(
      1, 0, 0, 2, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   const uint8_t exp_st2[14] = {0xe7, 0x00, 0x54, 0x02, 0x00, 0x00, 0x21,
                                0x00, 0x09, 0x00, 0x00, 0x90, 0x11, 0x00};
   ASSERT_EQ(packed.length, 14u);
   EXPECT_EQ(memcmp(packed.bytes, exp_st2, 14), 0);

   /* probe1_mem p_dev_u16x4 (i16x4 load & store) */
   ASSERT_TRUE(agx_apple9_pack_device_load_vector_u16_raw(
      0, 2, 1, 4, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_1, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      &packed));
   const uint8_t exp_ld4[14] = {0x67, 0x10, 0x44, 0x00, 0x01, 0x02, 0x20,
                                0x00, 0x47, 0x01, 0x00, 0x40, 0x48, 0x00};
   ASSERT_EQ(packed.length, 14u);
   EXPECT_EQ(memcmp(packed.bytes, exp_ld4, 14), 0);

   ASSERT_TRUE(agx_apple9_pack_device_store_vector_u16(
      0, 2, 0, 4, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   const uint8_t exp_st4[14] = {0xe7, 0x00, 0x54, 0x00, 0x00, 0x02, 0x21,
                                0x00, 0x07, 0x00, 0x00, 0x10, 0x12, 0x00};
   ASSERT_EQ(packed.length, 14u);
   EXPECT_EQ(memcmp(packed.bytes, exp_st4, 14), 0);
}

TEST(Apple9MemEncoding, MultiSlotGetSrAndExtendedHighGpr)
{
   agx_apple9_packed_instruction packed = {};

   /* 4-byte form slots 1..6 */
   for (unsigned s = 1; s <= 6; ++s) {
      ASSERT_TRUE(agx_apple9_pack_get_sr_ex(
         5, 0xa0, 0x10, (enum agx_apple9_scoreboard_slot)s, false, &packed));
      ASSERT_EQ(packed.length, 4u);
      EXPECT_EQ(packed.bytes[0], 0x5cu);
      EXPECT_EQ(packed.bytes[1], 0xa0u);
      EXPECT_EQ(packed.bytes[2], 0x10u);
      EXPECT_EQ(packed.bytes[3], uint8_t(0x06u | ((s - 1) << 5)));
   }
   EXPECT_FALSE(agx_apple9_pack_get_sr_ex(
      64, 0xa0, 0x10, AGX_APPLE9_SCOREBOARD_SLOT_1, false, &packed));

   /* 8-byte extended form verified against probe r70 & r80 */
   ASSERT_TRUE(agx_apple9_pack_get_sr_ex(
      70, 0xa0, 0x10, AGX_APPLE9_SCOREBOARD_SLOT_1, true, &packed));
   const uint8_t exp_r70[8] = {0x6c, 0xa0, 0x11, 0x06, 0x00, 0x00, 0x00, 0x10};
   ASSERT_EQ(packed.length, 8u);
   EXPECT_EQ(memcmp(packed.bytes, exp_r70, 8), 0);

   ASSERT_TRUE(agx_apple9_pack_get_sr_ex(
      80, 0xa0, 0x10, AGX_APPLE9_SCOREBOARD_SLOT_6, true, &packed));
   const uint8_t exp_r80[8] = {0x0c, 0xa0, 0x51, 0xa6, 0x00, 0x00, 0x00, 0x10};
   ASSERT_EQ(packed.length, 8u);
   EXPECT_EQ(memcmp(packed.bytes, exp_r80, 8), 0);

   EXPECT_FALSE(agx_apple9_pack_get_sr_ex(
      96, 0xa0, 0x10, AGX_APPLE9_SCOREBOARD_SLOT_1, true, &packed));
}

TEST(Apple9SubgroupEncoding, ReduceScanShuffleAndRotateByteExact)
{
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   /* Verify 1-insn SIMD/Quad reductions & exclusive prefix scans against probes. */
   struct {
      enum agx_apple9_subgroup_reduce_op op;
      unsigned bits;
      bool quad;
      bool prefix;
      unsigned dst;
      unsigned src;
      bool retain;
      enum agx_apple9_scoreboard_slot wait;
      uint8_t exp[8];
   } red_cases[] = {
      /* sg_f32_ret: fmin r5, r0.cache, wait 5 */
      {AGX_APPLE9_SUBGROUP_REDUCE_FMIN, 32, false, false, 5, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_6, {0xbf, 0x05, 0x56, 0x0a, 0x02, 0x00, 0x0a, 0x02}},
      /* sg_f32_ret: fmax r1, r0.cache */
      {AGX_APPLE9_SUBGROUP_REDUCE_FMAX, 32, false, false, 1, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xbf, 0x07, 0x54, 0x02, 0x02, 0x00, 0x0a, 0x02}},
      /* sg_f32_ret: fadd r2, r0.cache */
      {AGX_APPLE9_SUBGROUP_REDUCE_FADD, 32, false, false, 2, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x3f, 0x06, 0x54, 0x04, 0x02, 0x00, 0x16, 0x02}},
      /* sg_f32_ret: fmul r3, r0.cache */
      {AGX_APPLE9_SUBGROUP_REDUCE_FMUL, 32, false, false, 3, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xbf, 0x06, 0x54, 0x06, 0x02, 0x00, 0x16, 0x02}},
      /* sg_f16_ret: hadd r3, r0.cache, wait 5 */
      {AGX_APPLE9_SUBGROUP_REDUCE_HADD, 16, false, false, 3, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_6, {0x3f, 0x04, 0x56, 0x06, 0x02, 0x00, 0x8a, 0x00}},
      /* sg_f16_ret: hadd prefix r1, r0.cache */
      {AGX_APPLE9_SUBGROUP_REDUCE_HADD, 16, false, true, 1, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x3f, 0x04, 0x54, 0x02, 0x02, 0x00, 0x8a, 0x10}},
      /* sg_f16_ret: hmin/hmax r1, r0.cache */
      {AGX_APPLE9_SUBGROUP_REDUCE_HMIN, 16, false, false, 1, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xbf, 0x05, 0x54, 0x02, 0x02, 0x00, 0x0a, 0x03}},
      {AGX_APPLE9_SUBGROUP_REDUCE_HMAX, 16, false, false, 1, 0, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xbf, 0x07, 0x54, 0x02, 0x02, 0x00, 0x0a, 0x03}},
      /* sg_quad_extra: quad or/xor/umax/umin/fadd_prefix */
      {AGX_APPLE9_SUBGROUP_REDUCE_OR, 32, true, false, 0, 3, true,
       AGX_APPLE9_SCOREBOARD_SLOT_6, {0xb7, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x16, 0x01}},
      {AGX_APPLE9_SUBGROUP_REDUCE_XOR, 32, true, false, 1, 3, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x37, 0x01, 0x54, 0x02, 0x02, 0x0c, 0x16, 0x01}},
      {AGX_APPLE9_SUBGROUP_REDUCE_UMAX, 32, true, false, 2, 3, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xb7, 0x03, 0x54, 0x04, 0x02, 0x0c, 0x16, 0x01}},
      {AGX_APPLE9_SUBGROUP_REDUCE_UMIN, 32, true, false, 3, 3, false,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x37, 0x03, 0x54, 0x06, 0x02, 0x0c, 0x14, 0x03}},
      {AGX_APPLE9_SUBGROUP_REDUCE_FADD, 32, true, true, 3, 3, false,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x37, 0x06, 0x54, 0x06, 0x02, 0x0c, 0x14, 0x32}},
      /* High GPR src r72 (src_hi bit in b6) and dst r80, 32-bit iadd prefix */
      {AGX_APPLE9_SUBGROUP_REDUCE_IADD, 32, false, true, 80, 72, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0xbf, 0x01, 0x54, 0xa0, 0x02, 0x20, 0x17, 0x09}},
      /* 16-bit signed imin sets bit 2 (0x04) in b7 */
      {AGX_APPLE9_SUBGROUP_REDUCE_IMIN, 16, false, false, 4, 5, true,
       AGX_APPLE9_SCOREBOARD_SLOT_NONE, {0x3f, 0x02, 0x54, 0x08, 0x02, 0x14, 0x12, 0x05}},
   };
   for (const auto &c : red_cases) {
      uint8_t phys[2] = {(uint8_t)c.dst, (uint8_t)c.src};
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_SUBGROUP_REDUCE;
      I.encoding = AGX_APPLE9_ENC_SUBGROUP_REDUCE;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.immediate = (uint32_t)c.op |
                    (c.quad ? AGX_APPLE9_SUBGROUP_REDUCE_QUAD : 0) |
                    (c.prefix ? AGX_APPLE9_SUBGROUP_REDUCE_PREFIX : 0) |
                    (c.bits == 16 ? AGX_APPLE9_SUBGROUP_REDUCE_16BIT : 0);
      I.live_after_mask = c.retain ? 1u : 0u;
      I.scoreboard_slot = c.wait;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 8u);
      EXPECT_EQ(memcmp(packed.bytes, c.exp, 8), 0);
   }

   /* 10-byte SIMD & Quad shuffles (reg & imm, high dst/src0). */
   {
      uint8_t phys[3] = {72, 75, 4};
      agx_apple9_vir_instr sh = {};
      sh.op = AGX_APPLE9_VIR_SUBGROUP_SHUFFLE;
      sh.encoding = AGX_APPLE9_ENC_SUBGROUP_SHUFFLE;
      sh.dest = 0;
      sh.dest_components = 1;
      sh.nr_srcs = 2;
      sh.src[0] = 1;
      sh.src[1] = 2;
      sh.immediate = AGX_APPLE9_SUBGROUP_SHUFFLE_XOR;
      sh.live_after_mask = 3;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&sh, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_reg[10] = {0xc7, 0x04, 0x54, 0x90, 0x02,
                                   0x2c, 0x11, 0x2c, 0x10, 0x00};
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(memcmp(packed.bytes, exp_reg, 10), 0);

      sh.nr_srcs = 1;
      sh.immediate = AGX_APPLE9_QUAD_SHUFFLE_XOR;
      sh.alu_src_immediate_mask = BITFIELD_BIT(1);
      sh.alu_src_value[1] = 3;
      sh.live_after_mask = 0;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&sh, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_imm[10] = {0xc7, 0x00, 0x54, 0x90, 0x02,
                                   0x2c, 0x07, 0x2c, 0x04, 0x00};
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(memcmp(packed.bytes, exp_imm, 10), 0);
   }

   /* 12-byte simd_shuffle_and_fill_down (C7 06): byte-exact vs probes. */
   {
      /* sg_rot_disc: r0 = rotate(r0.discard, r2.discard, r4.discard), wait 5 */
      uint8_t p1[4] = {0, 0, 2, 4};
      agx_apple9_vir_instr r = {};
      r.op = AGX_APPLE9_VIR_SUBGROUP_ROTATE;
      r.encoding = AGX_APPLE9_ENC_SUBGROUP_ROTATE;
      r.dest = 0;
      r.dest_components = 1;
      r.nr_srcs = 3;
      r.src[0] = 1;
      r.src[1] = 2;
      r.src[2] = 3;
      r.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&r, p1, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_disc[12] = {0xc7, 0x06, 0x56, 0x00, 0x02, 0x00,
                                    0x10, 0x10, 0x14, 0xa6, 0x06, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_disc, 12), 0);

      /* sg_rot_ret: r0 = rotate(r1, r2, r3), wait 5 */
      uint8_t p2[4] = {0, 1, 2, 3};
      r.live_after_mask = 7;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&r, p2, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_ret[12] = {0xc7, 0x06, 0x56, 0x00, 0x02, 0x04,
                                   0x0c, 0x10, 0x14, 0x80, 0x06, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_ret, 12), 0);

      /* sg_rot_imm: r0 = rotate(r0.discard, r2.discard, imm 5), wait 5 */
      r.nr_srcs = 2;
      r.live_after_mask = 0;
      r.alu_src_immediate_mask = BITFIELD_BIT(2);
      r.alu_src_value[2] = 5;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&r, p1, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_imm[12] = {0xc7, 0x06, 0x56, 0x00, 0x02, 0x00,
                                   0x0a, 0x10, 0x14, 0xa2, 0x02, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_imm, 12), 0);
   }
}

TEST(Apple9AluUnlockEncoding, SatAddAndnExtrBfiPackSnormAndNarrowConvs)
{
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   /* iadd.sat / isub.sat / uadd.sat / usub.sat vs probe3_alu.out */
   {
      uint8_t phys[3] = {1, 2, 3};
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_IADD;
      I.encoding = AGX_APPLE9_ENC_INT_ADD_EXTENDED;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 2;
      I.src[0] = 1;
      I.src[1] = 2;
      I.saturate = true;
      I.src_sext_mask = 3;
      I.live_after_mask = 3;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_sadd[10] = {0x9f, 0x01, 0x54, 0x02, 0x02,
                                    0x08, 0x1a, 0xec, 0x11, 0x0f};
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(memcmp(packed.bytes, exp_sadd, 10), 0);

      I.op = AGX_APPLE9_VIR_ISUB;
      I.live_after_mask = 0;
      phys[0] = 2;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_ssub[10] = {0x1f, 0x01, 0x54, 0x04, 0x02,
                                    0x08, 0x18, 0xe8, 0x17, 0x0f};
      EXPECT_EQ(memcmp(packed.bytes, exp_ssub, 10), 0);

      I.op = AGX_APPLE9_VIR_IADD;
      I.src_sext_mask = 0;
      I.live_after_mask = 3;
      phys[0] = 0;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_uadd[10] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                                    0x08, 0x1a, 0xec, 0x11, 0x05};
      EXPECT_EQ(memcmp(packed.bytes, exp_uadd, 10), 0);
   }

   /* IANDN (andn2) vs verify_bits.out:840 */
   {
      uint8_t phys[3] = {3, 3, 5};
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_IANDN;
      I.encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 2;
      I.src[0] = 1;
      I.src[1] = 2;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_andn[10] = {0x3b, 0x07, 0x1e, 0x0b, 0x02,
                                    0x00, 0x00, 0x00, 0x00, 0x00};
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(memcmp(packed.bytes, exp_andn, 10), 0);
   }

   /* EXTR (27 01) imm & reg vs verify_bits2.out:4,5,7 */
   {
      uint8_t phys[4] = {1, 4, 3, 2};
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_EXTR;
      I.encoding = AGX_APPLE9_ENC_EXTR;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 2;
      I.src[0] = 1; /* hi = r4 */
      I.src[1] = 2; /* lo = r3 */
      I.live_after_mask = 3;
      I.alu_src_immediate_mask = BITFIELD_BIT(2);
      I.alu_src_value[2] = 19;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_imm[12] = {0x27, 0x01, 0x54, 0x02, 0x02, 0x10,
                                   0x4e, 0x30, 0xf8, 0x10, 0x09, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_imm, 12), 0);

      phys[0] = 2;
      I.nr_srcs = 3;
      I.src[2] = 3; /* shift = r2.discard */
      I.alu_src_immediate_mask = 0;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_reg[12] = {0x27, 0x01, 0x54, 0x04, 0x02, 0x10,
                                   0x12, 0x30, 0xf8, 0x12, 0x0d, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_reg, 12), 0);
   }

   /* Masked ISHL (bfi base=0) vs verify_bits.out:834 */
   {
      uint8_t phys[2] = {7, 4};
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_ISHL;
      I.encoding = AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE;
      I.dest = 0;
      I.dest_components = 1;
      I.nr_srcs = 1;
      I.src[0] = 1;
      I.immediate = 5u | (7u << AGX_APPLE9_SHIFT_WIDTH_SHIFT);
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_bfi[12] = {0x27, 0x00, 0x54, 0x0e, 0x02, 0x10,
                                   0x14, 0x00, 0xf0, 0x11, 0x71, 0x00};
      ASSERT_EQ(packed.length, 12u);
      EXPECT_EQ(memcmp(packed.bytes, exp_bfi, 12), 0);
   }

   /* PACK_SNORM_2X16 (10B) & PACK_SNORM_4X8 (20B) vs probe3_alu.out */
   {
      uint8_t phys[5] = {4, 0, 1, 2, 3};
      agx_apple9_vir_instr p2 = {};
      p2.op = AGX_APPLE9_VIR_PACK_SNORM_2X16;
      p2.encoding = AGX_APPLE9_ENC_PACK_SNORM_2X16;
      p2.dest = 0;
      p2.dest_components = 1;
      p2.nr_srcs = 2;
      p2.src[0] = 1;
      p2.src[1] = 2;
      p2.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&p2, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(packed.bytes[2], 0x56u);
      EXPECT_EQ(packed.bytes[9], 0x42u);

      agx_apple9_vir_instr p4 = {};
      p4.op = AGX_APPLE9_VIR_PACK_SNORM_4X8;
      p4.encoding = AGX_APPLE9_ENC_PACK_SNORM_4X8;
      p4.dest = 0;
      p4.dest_components = 1;
      p4.nr_srcs = 4;
      for (unsigned i = 0; i < 4; ++i)
         p4.src[i] = i + 1;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&p4, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 20u);
      EXPECT_EQ(packed.bytes[3], 8u);
      EXPECT_EQ(packed.bytes[9], 0x62u);
      EXPECT_EQ(packed.bytes[13], 9u);
      EXPECT_EQ(packed.bytes[19], 0x62u);
   }

   /* Narrow integer/float conversions vs verify_bits.out:849-856 */
   {
      uint8_t phys[2] = {0, 0};
      agx_apple9_vir_instr u2h = {};
      u2h.op = AGX_APPLE9_VIR_U2F16;
      u2h.encoding = AGX_APPLE9_ENC_INT_TO_HALF;
      u2h.dest = 0;
      u2h.dest_components = 1;
      u2h.nr_srcs = 1;
      u2h.src[0] = 1;
      u2h.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&u2h, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_u2h[8] = {0xa7, 0x07, 0x56, 0x00, 0x03, 0x00, 0xa8, 0x20};
      ASSERT_EQ(packed.length, 8u);
      EXPECT_EQ(memcmp(packed.bytes, exp_u2h, 8), 0);

      u2h.op = AGX_APPLE9_VIR_I16_2F16;
      phys[1] = 2;
      u2h.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_2;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&u2h, phys, &packed, &reason))
         << (reason ?: "");
      const uint8_t exp_s16h[8] = {0xa7, 0x27, 0x54, 0x00, 0x03, 0x08, 0xa0, 0x60};
      EXPECT_EQ(memcmp(packed.bytes, exp_s16h, 8), 0);

      struct {
         enum agx_apple9_vir_opcode op;
         uint8_t b7;
         uint8_t b8;
      } f2n[] = {
         {AGX_APPLE9_VIR_F2I16, 0xc8, 0x02},
         {AGX_APPLE9_VIR_F2U16, 0x08, 0x03},
         {AGX_APPLE9_VIR_F2I8,  0x48, 0x02},
         {AGX_APPLE9_VIR_F2U8,  0x88, 0x02},
      };
      phys[0] = 1;
      phys[1] = 3;
      for (const auto &c : f2n) {
         agx_apple9_vir_instr cvt = {};
         cvt.op = c.op;
         cvt.encoding = AGX_APPLE9_ENC_FLOAT_TO_NARROW_INT;
         cvt.dest = 0;
         cvt.dest_components = 1;
         cvt.nr_srcs = 1;
         cvt.src[0] = 1;
         cvt.live_after_mask = 1;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&cvt, phys, &packed, &reason))
            << (reason ?: "");
         const uint8_t exp_f2n[10] = {0x27, 0x07, 0x54, 0x02, 0x03,
                                      0x0c, 0x92, c.b7, c.b8, 0x00};
         ASSERT_EQ(packed.length, 10u);
         EXPECT_EQ(memcmp(packed.bytes, exp_f2n, 10), 0);
      }
   }
}

TEST(Apple9MemEncoding, Tier2TextureModesAndCompact2DImageWrite)
{
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   const uint8_t phys[] = {0, 1, 2, 3, 4, 5, 6, 7};
   const uint16_t ofs_1_m2 = 1u | (0xeu << 4);

   struct {
      enum agx_apple9_encoding enc;
      unsigned dim;
      unsigned imm;
      bool fetch;
      bool shadow;
      bool min_lod;
      bool has_offset;
      uint16_t expected_mode;
   } cases[] = {
      {AGX_APPLE9_ENC_TEXTURE_SAMPLE, 5, 0, false, false, false, false, 0x0080},
      {AGX_APPLE9_ENC_TEXTURE_SAMPLE, 6, 0, false, false, false, false, 0x00a0},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    5, 0, true,  false, false, false, 0x0c80},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    6, 0, true,  false, false, false, 0x0ca0},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 2, false, false, false, false, 0x0408},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 2, false, false, false, true,  0x0301},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 2, false, true,  false, false, 0x0420},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 2, false, true,  false, true,  0x0321},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 1, false, true,  false, true,  0x03a1},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    2, 0, false, true,  true,  true,  0x0361},
      {AGX_APPLE9_ENC_TEXTURE_GRAD,   1, 0, false, false, false, false, 0x0304},
      {AGX_APPLE9_ENC_TEXTURE_GRAD,   4, 0, false, false, false, false, 0x030c},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    4, 2, false, true,  false, false, 0x0430},
      {AGX_APPLE9_ENC_TEXTURE_LOD,    4, 1, false, true,  false, false, 0x0038},
   };
   for (const auto &c : cases) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_TEXTURE_SAMPLE;
      I.encoding = c.enc;
      I.dest = 4;
      I.dest_components = 4;
      I.nr_srcs = c.enc == AGX_APPLE9_ENC_TEXTURE_GRAD ? 1
                : c.enc == AGX_APPLE9_ENC_TEXTURE_LOD  ? 4
                                                       : 2;
      for (unsigned s = 0; s < I.nr_srcs; ++s)
         I.src[s] = s;
      I.texture_dimension = c.dim;
      I.immediate = c.imm;
      I.texture_fetch = c.fetch;
      I.texture_shadow = c.shadow;
      I.texture_min_lod = c.min_lod;
      I.texture_has_offset = c.has_offset;
      I.texture_offset = c.has_offset ? ofs_1_m2 : 0;
      I.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason))
         << "expected 0x" << std::hex << c.expected_mode << ": " << (reason ?: "");
      uint16_t got = packed.bytes[6] | (uint16_t(packed.bytes[7]) << 8);
      EXPECT_EQ(got, c.expected_mode);
   }

   /* Compact 2D 32-bit IMAGE_WRITE_PARAMS_2D packs 6 moves (60B, -20B/store). */
   agx_apple9_vir_instr p2d = {};
   p2d.op = AGX_APPLE9_VIR_PUBLICATION_TUPLE;
   p2d.encoding = AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS_2D;
   p2d.dest = 0; /* p8 */
   p2d.dest_components = 8;
   p2d.nr_srcs = 6;
   for (unsigned i = 0; i < 6; ++i)
      p2d.src[i] = i + 1;
   const uint8_t p2d_phys[] = {8, 10, 11, 12, 13, 14, 15};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&p2d, p2d_phys, &packed, &reason))
      << (reason ?: "");
   ASSERT_EQ(packed.length, 60u);
   /* Verify destinations are cr8, cr9, cr10, cr11, cr14, cr15 (skipping cr12/13). */
   const unsigned exp_pub[6] = {8, 9, 10, 11, 14, 15};
   for (unsigned c = 0; c < 6; ++c) {
      unsigned dst_pub = ((packed.bytes[10 * c + 0] >> 4) & 0xf) |
                         (((packed.bytes[10 * c + 4] >> 4) & 0x3) << 4);
      EXPECT_EQ(dst_pub, exp_pub[c]);
   }
}

namespace {

struct NativeMeshSpec {
   enum mesa_prim prim = MESA_PRIM_TRIANGLES;
   unsigned max_v = 3;
   unsigned max_p = 1;
   uint16_t wg[3] = {16, 1, 1};
   bool dynamic_counts = false;
   bool write_var0 = false;
   bool write_psiz = false;
   bool read_psiz = false;
   bool write_layer = false;
   bool write_viewport = false;
   bool write_prim_id = false;
   bool write_cull_prim = false;
   unsigned clip_distances = 0;
   unsigned cull_distances = 0;
   bool gather_info = true;
};

static nir_shader *
build_native_mesh_shader(const NativeMeshSpec &spec)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "native_mesh_expand_test");
   nir_shader *nir = b.shader;
   nir->info.io_lowered = true;
   for (unsigned d = 0; d < 3; ++d)
      nir->info.workgroup_size[d] = spec.wg[d];
   nir->info.mesh.max_vertices_out = spec.max_v;
   nir->info.mesh.max_primitives_out = spec.max_p;
   nir->info.mesh.primitive_type = spec.prim;

   unsigned k = agx_apple9_mesh_indices_per_primitive(spec.prim);
   unsigned logical = spec.wg[0] * spec.wg[1] * spec.wg[2];

   nir_def *verts = spec.dynamic_counts
      ? nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x10000), .align_mul = 4)
      : nir_imm_int(&b, spec.max_v);
   nir_def *prims = spec.dynamic_counts
      ? nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x10004), .align_mul = 4)
      : nir_imm_int(&b, spec.max_p);
   nir_set_vertex_and_primitive_count(&b, verts, prims, nir_imm_int(&b, 0));

   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned first = 0; first < spec.max_v; first += logical) {
      nir_def *v = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, v, verts));
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, 0.25f, -0.25f, 0.5f, 1.0f), v, nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      if (spec.write_var0) {
         nir_store_per_vertex_output(
            &b, nir_u2f32(&b, v), v, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      }
      if (spec.write_psiz) {
         nir_store_per_vertex_output(
            &b, nir_imm_float(&b, 4.0f), v, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
      }
      if (spec.read_psiz) {
         nir_def *ps = nir_load_per_vertex_output(
            &b, 1, 32, v, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
         nir_store_global(&b, ps, nir_imm_int64(&b, 0x20000), .align_mul = 4);
      }
      unsigned total_cc = spec.clip_distances + spec.cull_distances;
      if (total_cc) {
         nir_store_per_vertex_output(
            &b, nir_imm_vec4(&b, 1.0f, 0.5f, -0.5f, 2.0f), v, nir_imm_int(&b, 0),
            .write_mask = BITFIELD_MASK(MIN2(total_cc, 4u)),
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
         if (total_cc > 4) {
            nir_store_per_vertex_output(
               &b, nir_imm_vec4(&b, 1.0f, 1.0f, 1.0f, 1.0f), v, nir_imm_int(&b, 0),
               .write_mask = BITFIELD_MASK(total_cc - 4),
               .src_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_CLIP_DIST1, .num_slots = 1});
         }
      }
      nir_pop_if(&b, NULL);
   }

   for (unsigned first = 0; first < spec.max_p; first += logical) {
      nir_def *p = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, p, prims));
      nir_def *idx[3];
      for (unsigned c = 0; c < k; ++c)
         idx[c] = nir_umod(&b, nir_iadd_imm(&b, nir_imul_imm(&b, p, k), c), verts);
      nir_store_per_primitive_output(
         &b, nir_vec(&b, idx, k), p, nir_imm_int(&b, 0),
         .write_mask = BITFIELD_MASK(k), .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      if (spec.write_layer) {
         nir_store_per_primitive_output(
            &b, nir_iadd_imm(&b, p, 1), p, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_LAYER, .num_slots = 1});
      }
      if (spec.write_viewport) {
         nir_store_per_primitive_output(
            &b, nir_iadd_imm(&b, p, 2), p, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_VIEWPORT, .num_slots = 1});
      }
      if (spec.write_prim_id) {
         nir_store_per_primitive_output(
            &b, nir_iadd_imm(&b, p, 100), p, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
      }
      if (spec.write_cull_prim) {
         nir_store_per_primitive_output(
            &b, nir_ieq_imm(&b, nir_iand_imm(&b, p, 1), 1), p, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_bool1,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
      }
      nir_pop_if(&b, NULL);
   }

   if (spec.gather_info)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = spec.clip_distances;
   nir->info.cull_distance_array_size = spec.cull_distances;
   return nir;
}

struct PublicationAudit {
   unsigned controls = 0;
   std::set<std::pair<unsigned, unsigned>> channels; /* (channel, offset) */
   std::set<unsigned> scalar_offsets;
};

static PublicationAudit
audit_publication(nir_shader *nir)
{
   PublicationAudit out;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_mesh_control_apple9_agx) {
            ++out.controls;
         } else if (intr->intrinsic == nir_intrinsic_mesh_submit_channel_apple9_agx) {
            out.channels.insert({nir_src_as_uint(intr->src[0]),
                                 nir_src_as_uint(intr->src[1])});
         } else if (intr->intrinsic == nir_intrinsic_mesh_submit_scalar_apple9_agx) {
            out.scalar_offsets.insert(nir_src_as_uint(intr->src[1]));
         }
      }
   }
   return out;
}

} /* namespace */

TEST(Apple9MeshNativeExpand, MultiVertexMultiPrimitiveTrianglesAndWorkgroups)
{
   struct {
      unsigned max_v, max_p;
      uint16_t wg[3];
      unsigned physical;
      bool rounds;
      bool dynamic_counts;
      bool write_var0;
   } cases[] = {
      {6, 2, {6, 1, 1}, 16, false, false, false},
      {6, 2, {16, 1, 1}, 32, true, true, true},
      {64, 64, {32, 1, 1}, 32, true, true, true},
      {64, 64, {4, 4, 4}, 32, true, true, true},
      {96, 32, {2, 4, 16}, 32, true, false, true},
      {256, 256, {32, 1, 1}, 32, true, true, false},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "V" << c.max_v << "P" << c.max_p << " T" << c.physical);
      NativeMeshSpec spec;
      spec.max_v = c.max_v;
      spec.max_p = c.max_p;
      memcpy(spec.wg, c.wg, sizeof(spec.wg));
      spec.dynamic_counts = c.dynamic_counts;
      spec.write_var0 = c.write_var0;

      agx_apple9_native_mesh_options opts = {};
      opts.physical_threads = c.physical;
      opts.packed_index_publication = true;
      opts.publication_rounds = c.rounds;
      opts.attributes = c.write_var0;

      nir_shader *lowered = build_native_mesh_shader(spec);
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_native_mesh(lowered, true, &opts, &info,
                                                   &reason))
         << (reason ?: "");
      EXPECT_FALSE(info.requires_mesh_completion);
      EXPECT_EQ(info.layout.max_vertices, c.max_v);
      EXPECT_EQ(info.layout.max_primitives, c.max_p);
      EXPECT_EQ(info.layout.indices_per_primitive, 3u);
      EXPECT_EQ(info.layout.primitive_count_B, ALIGN_POT(c.max_p * 3, 16));
      EXPECT_EQ(info.layout.vertex_count_B, info.layout.primitive_count_B + 4);
      EXPECT_EQ(info.layout.vertex_pitch_B, ALIGN_POT(c.max_v, 4) * 4);
      EXPECT_EQ(info.layout.vertex_components, c.write_var0 ? 5u : 4u);

      auto audit = audit_publication(lowered);
      EXPECT_EQ(audit.controls, 1u);
      unsigned round = MIN2(c.physical, AGX_APPLE9_PUBLICATION_ROUND);
      unsigned expected_v_rounds = DIV_ROUND_UP(c.max_v, round);
      unsigned expected_p_rounds =
         DIV_ROUND_UP(DIV_ROUND_UP(c.max_p * 3, 4), round);
      EXPECT_EQ(audit.channels.size(),
                expected_v_rounds * info.layout.vertex_components);
      EXPECT_EQ(audit.scalar_offsets.size(), expected_p_rounds);
      for (unsigned r = 0; r < expected_v_rounds; ++r) {
         unsigned first = r * round;
         for (unsigned comp = 0; comp < 4; ++comp) {
            unsigned plane =
               agx_apple9_mesh_plane_offset(&info.layout, VARYING_SLOT_POS, comp);
            EXPECT_TRUE(audit.channels.count({comp | (first << 16),
                                              plane + 4 * first}));
         }
         if (c.write_var0) {
            unsigned plane =
               agx_apple9_mesh_plane_offset(&info.layout, VARYING_SLOT_VAR0, 0);
            EXPECT_TRUE(audit.channels.count({4u | (first << 16),
                                              plane + 4 * first}));
         }
      }
      for (unsigned r = 0; r < expected_p_rounds; ++r)
         EXPECT_TRUE(audit.scalar_offsets.count(4 * r * round));
      ralloc_free(lowered);

      nir_shader *nir = build_native_mesh_shader(spec);
      agx_shader_part code = {};
      ASSERT_TRUE(agx_compile_apple9_native_mesh(nir, true, &opts, &code, &info,
                                                 &reason))
         << (reason ?: "");
      EXPECT_TRUE(code.binary && code.info.binary_size > 0);
      EXPECT_TRUE(code.info.apple9_body_only);
      EXPECT_EQ(code.info.workgroup_size[0], c.physical);
      EXPECT_EQ(code.info.workgroup_size[1], 1u);
      EXPECT_EQ(code.info.workgroup_size[2], 1u);
      EXPECT_EQ(code.info.apple9_varyings.count, c.write_var0 ? 1u : 0u);
      free(code.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9MeshNativeExpand, UnusedPointSizeDroppedOnTrianglesAndLines)
{
   for (enum mesa_prim prim : {MESA_PRIM_TRIANGLES, MESA_PRIM_LINES}) {
      NativeMeshSpec spec;
      spec.prim = prim;
      spec.max_v = prim == MESA_PRIM_TRIANGLES ? 6 : 4;
      spec.max_p = 2;
      spec.wg[0] = 32;
      spec.write_psiz = true;

      agx_apple9_native_mesh_options opts = {};
      opts.physical_threads = 32;
      opts.packed_index_publication = true;
      opts.publication_rounds = true;

      nir_shader *nir = build_native_mesh_shader(spec);
      agx_shader_part code = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_native_mesh(nir, true, &opts, &code, &info,
                                                 &reason))
         << (reason ?: "");
      EXPECT_FALSE(info.writes_point_size);
      EXPECT_FALSE(code.info.apple9_writes_point_size);
      EXPECT_EQ(info.layout.vertex_components, 4u);
      EXPECT_EQ(code.info.apple9_varyings.count, 0u);
      free(code.binary);
      ralloc_free(nir);
   }

   /* Reading gl_PointSize back on triangles must still be rejected cleanly. */
   NativeMeshSpec bad;
   bad.max_v = 3;
   bad.max_p = 1;
   bad.wg[0] = 16;
   bad.read_psiz = true;
   nir_shader *nir = build_native_mesh_shader(bad);
   agx_apple9_mesh_body_info info = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_nir_lower_apple9_native_mesh(nir, true, nullptr, &info, &reason));
   EXPECT_NE(reason, nullptr);
   ralloc_free(nir);
}

TEST(Apple9MeshNativeExpand, PointsTopologyAndPointSizeBuiltin)
{
   for (bool with_psiz : {false, true}) {
      for (unsigned max_v : {12u, 64u}) {
         SCOPED_TRACE(::testing::Message()
                      << "V" << max_v << " psiz=" << with_psiz);
         NativeMeshSpec spec;
         spec.prim = MESA_PRIM_POINTS;
         spec.max_v = max_v;
         spec.max_p = max_v;
         spec.wg[0] = 32;
         spec.dynamic_counts = true;
         spec.write_var0 = true;
         spec.write_psiz = with_psiz;
         spec.clip_distances = 2;

         agx_apple9_native_mesh_options opts = {};
         opts.physical_threads = 32;
         opts.packed_index_publication = true;
         opts.publication_rounds = true;
         opts.attributes = true;

         nir_shader *lowered = build_native_mesh_shader(spec);
         agx_apple9_mesh_body_info info = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_nir_lower_apple9_native_mesh(lowered, true, &opts,
                                                      &info, &reason))
            << (reason ?: "");
         EXPECT_EQ(info.layout.indices_per_primitive, 1u);
         EXPECT_EQ(info.layout.primitive_count_B, ALIGN_POT(max_v, 16));
         /* Apple9 requires PointSize in the UVS for points topology even when
          * the shader omits gl_PointSize (default 1.0f initialized at entry). */
         EXPECT_TRUE(info.writes_point_size);
         EXPECT_EQ(info.clip_distances, 2u);
         EXPECT_EQ(info.layout.vertex_components, 4u + 1u + 1u + 2u);

         bool saw_default_psiz = false, saw_fmax_1 = false, saw_fmin_max = false;
         nir_foreach_block(block, nir_shader_get_entrypoint(lowered)) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_store_local_apple9_agx &&
                      nir_src_is_const(intr->src[0]) &&
                      nir_src_as_float(intr->src[0]) == 1.0f)
                     saw_default_psiz = true;
               } else if (instr->type == nir_instr_type_alu) {
                  nir_alu_instr *alu = nir_instr_as_alu(instr);
                  if (alu->op == nir_op_fmax && nir_src_is_const(alu->src[1].src) &&
                      nir_src_as_float(alu->src[1].src) == 1.0f)
                     saw_fmax_1 = true;
                  if (alu->op == nir_op_fmin && nir_src_is_const(alu->src[1].src) &&
                      nir_src_as_float(alu->src[1].src) == 511.9375f)
                     saw_fmin_max = true;
               }
            }
         }
         EXPECT_TRUE(saw_default_psiz);
         EXPECT_EQ(saw_fmax_1, with_psiz);
         EXPECT_EQ(saw_fmin_max, with_psiz);

         auto audit = audit_publication(lowered);
         unsigned v_rounds = DIV_ROUND_UP(max_v, 32u);
         unsigned p_rounds = DIV_ROUND_UP(DIV_ROUND_UP(max_v, 4u), 32u);
         EXPECT_EQ(audit.controls, 1u);
         EXPECT_EQ(audit.scalar_offsets.size(), p_rounds);
         EXPECT_EQ(audit.channels.size(), v_rounds * info.layout.vertex_components);
         /* Verify exact UVS order: 0..3 POS, 4 VAR0, 5 PSIZ, 6..7 CLIP0..1. */
         unsigned psiz_ch = 5;
         unsigned clip_ch = 6;
         unsigned psiz_plane =
            agx_apple9_mesh_plane_offset(&info.layout, VARYING_SLOT_PSIZ, 0);
         EXPECT_TRUE(audit.channels.count({psiz_ch, psiz_plane}));
         for (unsigned d = 0; d < 2; ++d) {
            unsigned plane = agx_apple9_mesh_plane_offset(
               &info.layout, VARYING_SLOT_CLIP_DIST0, d);
            EXPECT_TRUE(audit.channels.count({clip_ch + d, plane}));
         }
         ralloc_free(lowered);

         nir_shader *nir = build_native_mesh_shader(spec);
         agx_shader_part code = {};
         ASSERT_TRUE(agx_compile_apple9_native_mesh(nir, true, &opts, &code,
                                                    &info, &reason))
            << (reason ?: "");
         EXPECT_TRUE(code.info.apple9_writes_point_size);
         EXPECT_EQ(code.info.apple9_clip_distance_count, 2u);
         EXPECT_EQ(code.info.apple9_varyings.count, 1u);
         free(code.binary);
         ralloc_free(nir);
      }
   }

   /* Points reject culling and require publication_rounds at T32. */
   for (unsigned bad_mode = 0; bad_mode < 3; ++bad_mode) {
      NativeMeshSpec spec;
      spec.prim = MESA_PRIM_POINTS;
      spec.max_v = 8;
      spec.max_p = 8;
      spec.wg[0] = 32;
      if (bad_mode == 0)
         spec.write_cull_prim = true;
      else if (bad_mode == 1)
         spec.cull_distances = 1;

      agx_apple9_native_mesh_options opts = {};
      opts.physical_threads = 32;
      opts.packed_index_publication = true;
      opts.publication_rounds = bad_mode != 2;

      nir_shader *nir = build_native_mesh_shader(spec);
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      EXPECT_FALSE(agx_nir_lower_apple9_native_mesh(nir, true, &opts, &info,
                                                    &reason));
      EXPECT_NE(reason, nullptr);
      ralloc_free(nir);
   }
}

TEST(Apple9MeshNativeExpand, LayerViewportAndPrimitiveIdBuiltins)
{
   struct {
      unsigned max_v, max_p;
      bool broadcast;
      bool layer, viewport, prim_id, var0;
      unsigned clips;
      bool gather_info;
   } cases[] = {
      /* P1 broadcast: Layer only, Viewport only, and both + PrimitiveID. */
      {3, 1, true, true, false, false, false, 0, true},
      {3, 1, true, false, true, false, false, 0, true},
      {3, 1, true, true, true, true, true, 2, true},
      /* Multi-primitive (V6/P2 and V64/P64) via provoking-vertex copies. */
      {6, 2, false, true, false, false, false, 0, true},
      {6, 2, false, false, true, false, false, 0, true},
      {64, 64, false, true, true, true, true, 2, true},
      /* Works even if caller did not pre-populate nir->info.per_primitive_outputs. */
      {6, 2, false, true, true, true, false, 0, false},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "V" << c.max_v << "P" << c.max_p << " L=" << c.layer
                   << " VP=" << c.viewport << " PID=" << c.prim_id);
      NativeMeshSpec spec;
      spec.max_v = c.max_v;
      spec.max_p = c.max_p;
      spec.wg[0] = c.broadcast ? 16 : 32;
      spec.write_var0 = c.var0;
      spec.write_layer = c.layer;
      spec.write_viewport = c.viewport;
      spec.write_prim_id = c.prim_id;
      spec.clip_distances = c.clips;
      spec.gather_info = c.gather_info;

      agx_apple9_native_mesh_options opts = {};
      opts.physical_threads = c.broadcast ? 16 : 32;
      opts.packed_index_publication = !c.broadcast;
      opts.publication_rounds = !c.broadcast;
      opts.attributes = true;
      opts.primitive_broadcast = c.broadcast;
      opts.primitive_vertices = !c.broadcast;
      opts.layer_viewport = true;

      nir_shader *lowered = build_native_mesh_shader(spec);
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_native_mesh(lowered, true, &opts, &info,
                                                   &reason))
         << (reason ?: "");
      EXPECT_TRUE(info.writes_layer_viewport);
      EXPECT_FALSE(info.writes_point_size);
      EXPECT_EQ(info.clip_distances, c.clips);
      if (!c.broadcast) {
         EXPECT_EQ(info.layout.max_vertices, c.max_v + c.max_p);
      }

      unsigned expected_varyings =
         unsigned(c.var0) + unsigned(c.viewport) + unsigned(c.prim_id);
      EXPECT_EQ(info.layout.vertex_components,
                4u + expected_varyings + 1u + c.clips);

      /* Verify packed Layer/Viewport bit packing (layer & 0xffff | vp << 16)
       * and that the channel sits right after varyings and before ClipDist. */
      bool saw_layer_mask = false, saw_vp_shift = false, saw_pack_or = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(lowered)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_iand && nir_src_is_const(alu->src[1].src) &&
                nir_src_as_uint(alu->src[1].src) == 0xffffu)
               saw_layer_mask = true;
            if (alu->op == nir_op_ishl && nir_src_is_const(alu->src[1].src) &&
                nir_src_as_uint(alu->src[1].src) == 16u)
               saw_vp_shift = true;
            if (alu->op == nir_op_ior)
               saw_pack_or = true;
         }
      }
      if (c.layer) {
         EXPECT_TRUE(saw_layer_mask);
      }
      if (c.viewport) {
         EXPECT_TRUE(saw_vp_shift);
      }
      if (c.layer && c.viewport) {
         EXPECT_TRUE(saw_pack_or);
      }

      /* Verify packed Layer/Viewport channel sits right after user varyings
       * and immediately before ClipDistances in every vertex round. */
      auto audit = audit_publication(lowered);
      unsigned layer_ch = 4 + expected_varyings;
      unsigned layer_plane =
         agx_apple9_mesh_plane_offset(&info.layout, VARYING_SLOT_LAYER, 0);
      EXPECT_TRUE(audit.channels.count({layer_ch, layer_plane}));
      for (unsigned d = 0; d < c.clips; ++d) {
         unsigned clip_plane = agx_apple9_mesh_plane_offset(
            &info.layout, VARYING_SLOT_CLIP_DIST0, d);
         EXPECT_TRUE(audit.channels.count({layer_ch + 1 + d, clip_plane}));
      }
      ralloc_free(lowered);

      /* Full compilation (for primitive_vertices, replace load_provoking_last
       * like hk_apple9_lower_prerast_sysvals does). */
      nir_shader *nir = build_native_mesh_shader(spec);
      ASSERT_TRUE(agx_nir_lower_apple9_native_mesh(nir, true, &opts, &info,
                                                   &reason))
         << (reason ?: "");
      if (info.primitive_vertices) {
         nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_provoking_last) {
                  nir_builder nb = nir_builder_at(nir_before_instr(instr));
                  nir_def_replace(&intr->def, nir_imm_int(&nb, 0));
               }
            }
         }
      }
      agx_shader_part code = {};
      ASSERT_TRUE(agx_compile_apple9_native_mesh_lowered(nir, &opts, &code,
                                                         &info, &reason))
         << (reason ?: "");
      EXPECT_TRUE(code.binary && code.info.binary_size > 0);
      EXPECT_TRUE(code.info.writes_layer_viewport);
      EXPECT_EQ(code.info.apple9_clip_distance_count, c.clips);
      EXPECT_EQ(code.info.apple9_varyings.count, expected_varyings);
      EXPECT_EQ(code.info.apple9_varyings.mask[VARYING_SLOT_LAYER], 0u);
      EXPECT_EQ(code.info.apple9_varyings.mask[VARYING_SLOT_VIEWPORT],
                c.viewport ? 1u : 0u);
      if (c.viewport) {
         EXPECT_EQ(code.info.apple9_varyings.group[VARYING_SLOT_VIEWPORT], 1u);
      }
      if (c.prim_id) {
         EXPECT_EQ(code.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID], 1u);
      }
      free(code.binary);
      ralloc_free(nir);

      /* Without opts.layer_viewport, Layer/ViewportIndex remain rejected. */
      opts.layer_viewport = false;
      nir = build_native_mesh_shader(spec);
      EXPECT_FALSE(agx_nir_lower_apple9_native_mesh(nir, true, &opts, &info,
                                                    &reason));
      EXPECT_NE(reason, nullptr);
      ralloc_free(nir);
   }
}

TEST(Apple9MeshNativeExpand, ClipDistancePreservedAcrossCullPrimitiveAndSerialization)
{
   struct {
      uint16_t wg[3];
      unsigned clips, culls;
      bool cull_prim, link_clips;
   } cases[] = {
      {{32, 1, 1}, 3, 0, true, false},
      {{64, 1, 1}, 4, 0, false, false},
      {{4, 4, 4}, 2, 2, true, false},
      {{128, 1, 1}, 2, 2, true, true},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "wg=" << c.wg[0] << "x" << c.wg[1] << "x" << c.wg[2]
                   << " clips=" << c.clips << " culls=" << c.culls
                   << " cull_prim=" << c.cull_prim << " link=" << c.link_clips);
      NativeMeshSpec spec;
      spec.max_v = 64;
      spec.max_p = 64;
      memcpy(spec.wg, c.wg, sizeof(spec.wg));
      spec.clip_distances = c.clips;
      spec.cull_distances = c.culls;
      spec.write_cull_prim = c.cull_prim;

      agx_apple9_native_mesh_options opts = {};
      opts.physical_threads = 32;
      opts.packed_index_publication = true;
      opts.publication_rounds = true;
      opts.attributes = true;
      opts.link_clip_distances = c.link_clips;

      nir_shader *nir = build_native_mesh_shader(spec);
      agx_shader_part code = {};
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_native_mesh(nir, true, &opts, &code, &info,
                                                 &reason))
         << (reason ?: "");
      EXPECT_EQ(info.clip_distances, c.clips);
      EXPECT_EQ(info.cull_distances, c.culls);
      EXPECT_EQ(info.cull_primitive, c.cull_prim);
      EXPECT_EQ(info.clip_linked, c.link_clips);
      EXPECT_EQ(code.info.apple9_clip_distance_count, c.clips);
      EXPECT_EQ(code.info.apple9_varyings.count, c.link_clips ? c.clips : 0u);
      EXPECT_EQ(info.layout.vertex_components,
                4u + c.clips * (c.link_clips ? 2u : 1u));
      free(code.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9MeshNativeExpand, CullingRegionAlignmentIs16ByteAligned)
{
   for (unsigned max_p : {1u, 2u, 3u, 4u, 5u, 8u, 17u}) {
      NativeMeshSpec spec;
      spec.max_v = 6;
      spec.max_p = max_p;
      spec.wg[0] = 16;
      spec.write_cull_prim = true;

      nir_shader *nir = build_native_mesh_shader(spec);
      agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      agx_apple9_mesh_body_info info = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_mesh(nir, &key, &info, &reason))
         << (reason ?: "");
      EXPECT_TRUE(info.layout.culling);
      EXPECT_EQ(info.layout.position_B,
                info.layout.cull_B + ALIGN_POT(max_p * 4, 16));
      EXPECT_EQ((info.layout.local_size_B - info.layout.cull_B) % 16, 0u);
      ralloc_free(nir);
   }
}
