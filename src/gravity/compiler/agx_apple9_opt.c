/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "util/bitscan.h"
#include "util/hash_table.h"
#include "util/u_memory.h"

/* Selection creates address arithmetic and packing which did not exist
 * during NIR optimization. Keep a semantic whitelist: publications, implicit
 * state, derivatives and memory operations are not ordinary scalar ALU. */
bool
agx_apple9_instr_is_pure_alu(const struct agx_apple9_vir_instr *I)
{
   if ((I->dest_components != 1 && I->op != AGX_APPLE9_VIR_IMUL_WIDE &&
        I->op != AGX_APPLE9_VIR_UNPACK_NORM) ||
       I->publication_handoff ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       I->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT)
      return false;

   switch (I->op) {
   case AGX_APPLE9_VIR_IMM:
   case AGX_APPLE9_VIR_BIT_COUNT:
   case AGX_APPLE9_VIR_UFIND_MSB:
   case AGX_APPLE9_VIR_BIT_REVERSE:
   case AGX_APPLE9_VIR_HADD:
   case AGX_APPLE9_VIR_HSUB:
   case AGX_APPLE9_VIR_HMUL:
   case AGX_APPLE9_VIR_HFMA:
   case AGX_APPLE9_VIR_HMIN:
   case AGX_APPLE9_VIR_HMAX:
   case AGX_APPLE9_VIR_HRCP_F32:
   case AGX_APPLE9_VIR_HRCP:
   case AGX_APPLE9_VIR_HSQRT_FACTOR:
   case AGX_APPLE9_VIR_HMUL_MIXED:
   case AGX_APPLE9_VIR_HRSQ:
   case AGX_APPLE9_VIR_HEXP2:
   case AGX_APPLE9_VIR_HLOG2:
   case AGX_APPLE9_VIR_HFLOOR:
   case AGX_APPLE9_VIR_HCEIL:
   case AGX_APPLE9_VIR_HTRUNC:
   case AGX_APPLE9_VIR_HROUND_EVEN:
   case AGX_APPLE9_VIR_U2F32:
   case AGX_APPLE9_VIR_I2F32:
   case AGX_APPLE9_VIR_F2I32:
   case AGX_APPLE9_VIR_F2U32:
   case AGX_APPLE9_VIR_U2F16:
   case AGX_APPLE9_VIR_I2F16:
   case AGX_APPLE9_VIR_U16_2F16:
   case AGX_APPLE9_VIR_I16_2F16:
   case AGX_APPLE9_VIR_F2I16:
   case AGX_APPLE9_VIR_F2U16:
   case AGX_APPLE9_VIR_F2I8:
   case AGX_APPLE9_VIR_F2U8:
   case AGX_APPLE9_VIR_F2F16:
   case AGX_APPLE9_VIR_PACK_HALF_2X16:
   case AGX_APPLE9_VIR_PACK_UNORM_2X16:
   case AGX_APPLE9_VIR_PACK_UNORM_4X8:
   case AGX_APPLE9_VIR_PACK_SNORM_2X16:
   case AGX_APPLE9_VIR_PACK_SNORM_4X8:
   case AGX_APPLE9_VIR_PACK_FORMAT:
   case AGX_APPLE9_VIR_UNPACK_HALF:
   case AGX_APPLE9_VIR_UNPACK_NORM:
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_ISUB:
   case AGX_APPLE9_VIR_IMAD:
   case AGX_APPLE9_VIR_IMUL_WIDE:
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IANDN:
   case AGX_APPLE9_VIR_IOR_UNIFORM:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_EXTR:
   case AGX_APPLE9_VIR_IMIN:
   case AGX_APPLE9_VIR_IMAX:
   case AGX_APPLE9_VIR_UMIN:
   case AGX_APPLE9_VIR_UMAX:
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_FSAT:
   case AGX_APPLE9_VIR_FMIN:
   case AGX_APPLE9_VIR_FMAX:
   case AGX_APPLE9_VIR_FMA:
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_FFLOOR:
   case AGX_APPLE9_VIR_FCEIL:
   case AGX_APPLE9_VIR_FTRUNC:
   case AGX_APPLE9_VIR_FROUND_EVEN:
   case AGX_APPLE9_VIR_SELECT:
      return true;
   default:
      return false;
   }
}

static bool
apple9_instr_is_cse_eligible(const struct agx_apple9_vir_instr *I)
{
   if (agx_apple9_instr_is_pure_alu(I))
      return true;
   if (I->publication_handoff)
      return false;
   return (I->op == AGX_APPLE9_VIR_ITER && I->dest_components == 1) ||
          (I->op == AGX_APPLE9_VIR_CUBE &&
           (I->dest_components == 1 || I->dest_components == 2));
}

struct expression {
   uint32_t op, encoding, immediate, nr_srcs, components;
   uint32_t src_abs, src_neg, saturate, select_float_immediates;
   uint32_t src_uniform, src_immediate, src_value[4];
   uint32_t src_half, src_hi, src_sext;
   uint32_t src[AGX_APPLE9_MAX_VIR_SRCS];
};

bool
agx_apple9_supports_saturate(const struct agx_apple9_vir_instr *I)
{
   if (I->publication_handoff ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT)
      return false;
   switch (I->op) {
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_FSAT:
   case AGX_APPLE9_VIR_FMA:
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_DERIVATIVE:
      return true;
   default:
      /* The ordinary SFU saturation bit is ineffective on direct rounding.
       * Min/max and selects also need an explicit clamping operation. */
      return false;
   }
}

static void
select_saturated_encoding(struct agx_apple9_vir_instr *I)
{
   if (!I->saturate)
      return;
   switch (I->encoding) {
   case AGX_APPLE9_ENC_FLOAT2_COMPACT:
      I->encoding = AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED;
      break;
   case AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT:
      I->encoding = AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED;
      break;
   case AGX_APPLE9_ENC_FLOAT3_EXTENDED:
      I->encoding = AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED;
      break;
   default:
      break;
   }
}

static uint32_t
expression_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct expression));
}

static bool
expression_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct expression)) == 0;
}

static uint32_t
resolve(uint32_t *replacement, uint32_t value)
{
   uint32_t root = value;
   while (replacement[root] != root)
      root = replacement[root];
   while (replacement[value] != root) {
      uint32_t next = replacement[value];
      replacement[value] = root;
      value = next;
   }
   return root;
}

static bool
fold_integer(struct agx_apple9_vir_instr *I, const bool *constant,
             const uint32_t *values)
{
   /* Register sources are only the operands still in the register file;
    * a uniform or immediate operand is not a known constant here. */
   if (!I->nr_srcs || I->nr_srcs > 3 || I->alu_src_uniform_mask ||
       I->alu_src_immediate_mask || I->src_half_mask)
      return false;
   uint32_t c[3] = {0};
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (!constant[I->src[s]])
         return false;
      c[s] = values[I->src[s]];
   }

   if (I->saturate)
      return false;

   uint32_t value;
   switch (I->op) {
   case AGX_APPLE9_VIR_IADD:
      if (I->immediate > 4)
         return false;
      value = c[0] + (c[1] << I->immediate);
      break;
   case AGX_APPLE9_VIR_ISUB:
      if (I->immediate > 4)
         return false;
      value = c[0] - (c[1] << I->immediate);
      break;
   case AGX_APPLE9_VIR_IMAD: value = c[0] * c[1] + c[2]; break;
   case AGX_APPLE9_VIR_IAND: value = c[0] & c[1]; break;
   case AGX_APPLE9_VIR_IANDN: value = c[0] & ~c[1]; break;
   case AGX_APPLE9_VIR_IOR: value = c[0] | c[1]; break;
   case AGX_APPLE9_VIR_IXOR: value = c[0] ^ c[1]; break;
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL: {
      unsigned shift = I->nr_srcs == 2 ? (c[1] & 0xffff) : I->immediate & 31;
      unsigned width = I->nr_srcs == 2 ? 0 : I->immediate >> AGX_APPLE9_SHIFT_WIDTH_SHIFT;
      if (shift >= 32) {
         value = I->op == AGX_APPLE9_VIR_ISHR && (c[0] & 0x80000000u)
            ? UINT32_MAX : 0;
         break;
      }
      uint32_t src0 = (I->op == AGX_APPLE9_VIR_ISHL && width)
                         ? (c[0] & BITFIELD_MASK(width))
                         : c[0];
      value = I->op == AGX_APPLE9_VIR_ISHL ? src0 << shift : c[0] >> shift;
      if (I->op == AGX_APPLE9_VIR_ISHR && shift && (c[0] & 0x80000000u))
         value |= UINT32_MAX << (32 - shift);
      if (I->op == AGX_APPLE9_VIR_USHR && width)
         value &= BITFIELD_MASK(width);
      break;
   }
   default:
      return false;
   }

   I->op = AGX_APPLE9_VIR_IMM;
   I->encoding = (value == 0 || value > 0x7f) ? AGX_APPLE9_ENC_MOV_IMM32
                                              : AGX_APPLE9_ENC_MOV_IMM_COMPACT;
   I->nr_srcs = 0;
   I->immediate = value;
   return true;
}

/* Fuse `a +/- (b << 1..4)` into a single 10B IADD/ISUB with inline lsl, and
 * lower standalone `x << 1..4` from 12B ISHL to 10B `iadd 0, x, lsl k`. */
static void
fold_iadd_shift(struct agx_apple9_vir_instr *I,
                struct agx_apple9_vir_instr **definitions)
{
   if (I->op == AGX_APPLE9_VIR_ISHL &&
       I->encoding == AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE &&
       I->nr_srcs == 1 && !I->alu_src_uniform_mask &&
       !I->alu_src_immediate_mask &&
       (I->immediate >> AGX_APPLE9_SHIFT_WIDTH_SHIFT) == 0 &&
       I->immediate >= 1 && I->immediate <= 4) {
      I->op = AGX_APPLE9_VIR_IADD;
      I->encoding = AGX_APPLE9_ENC_INT_ADD_EXTENDED;
      I->alu_src_immediate_mask = 1;
      I->alu_src_value[0] = 0;
      return;
   }

   if ((I->op != AGX_APPLE9_VIR_IADD && I->op != AGX_APPLE9_VIR_ISUB) ||
       I->encoding != AGX_APPLE9_ENC_INT_ADD_EXTENDED || I->immediate ||
       I->saturate || I->src_half_mask || I->alu_src_uniform_mask ||
       I->alu_src_immediate_mask || I->nr_srcs != 2)
      return;

   /* Prefer src1 (no swap needed); for commutative IADD, also check src0. */
   for (int s = 1; s >= (I->op == AGX_APPLE9_VIR_IADD ? 0 : 1); --s) {
      const struct agx_apple9_vir_instr *D = definitions[I->src[s]];
      if (!D || !agx_apple9_instr_is_pure_alu(D) ||
          agx_apple9_instr_block(D) != agx_apple9_instr_block(I) ||
          D->nr_srcs != 1 || D->alu_src_uniform_mask || D->src_half_mask)
         continue;
      bool is_shl = D->op == AGX_APPLE9_VIR_ISHL &&
                    D->encoding == AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE &&
                    !D->alu_src_immediate_mask &&
                    (D->immediate >> AGX_APPLE9_SHIFT_WIDTH_SHIFT) == 0 &&
                    D->immediate >= 1 && D->immediate <= 4;
      bool is_iadd_shl = D->op == AGX_APPLE9_VIR_IADD &&
                         D->encoding == AGX_APPLE9_ENC_INT_ADD_EXTENDED &&
                         !D->saturate && D->alu_src_immediate_mask == 1 &&
                         D->alu_src_value[0] == 0 &&
                         D->immediate >= 1 && D->immediate <= 4;
      if (!is_shl && !is_iadd_shl)
         continue;
      if (s == 0) {
         uint32_t other = I->src[1];
         I->src[0] = other;
         I->src[1] = D->src[0];
      } else {
         I->src[1] = D->src[0];
      }
      I->immediate = D->immediate;
      return;
   }
}

bool
agx_apple9_supports_special_uniform(enum agx_apple9_vir_opcode op)
{
   switch (op) {
   case AGX_APPLE9_VIR_FFLOOR:
   case AGX_APPLE9_VIR_FCEIL:
   case AGX_APPLE9_VIR_FTRUNC:
   case AGX_APPLE9_VIR_FROUND_EVEN:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
      return true;
   default:
      return false;
   }
}

static void
fold_select_xor(struct agx_apple9_vir_instr *I,
                struct agx_apple9_vir_instr **definitions,
                const bool *constant, const uint32_t *values,
                bool uniform_writes)
{
   if (I->op != AGX_APPLE9_VIR_IXOR || I->nr_srcs != 2 ||
       I->alu_src_uniform_mask || I->alu_src_immediate_mask)
      return;

   for (unsigned c = 0; c < 2; ++c) {
      if (!constant[I->src[c]])
         continue;
      const struct agx_apple9_vir_instr *S = definitions[I->src[1 - c]];
      if (!S || S->op != AGX_APPLE9_VIR_SELECT ||
          S->encoding == AGX_APPLE9_ENC_HALF_COMPARE_SELECT ||
          !agx_apple9_instr_is_pure_alu(S) ||
          agx_apple9_instr_block(S) != agx_apple9_instr_block(I) ||
          ((S->alu_src_uniform_mask | S->src_abs_mask | S->src_neg_mask) & 0xc) ||
          (uniform_writes && S->alu_src_uniform_mask))
         continue;

      /* The selected arms are the last two operand slots, each a literal
       * or one of the final compact GPR sources. Swapping them implements
       * XOR only when they are complements under this mask; an arbitrary
       * Boolean-valued select need not have opposite arms. */
      unsigned literals = S->alu_src_immediate_mask & 0xc;
      unsigned registers = 2 - util_bitcount(literals);
      if (S->nr_srcs < registers)
         continue;
      uint32_t arm[2];
      bool known = true;
      for (unsigned k = 0, r = S->nr_srcs - registers; k < 2; ++k) {
         if (literals & BITFIELD_BIT(2 + k)) {
            arm[k] = S->alu_src_value[2 + k];
         } else {
            known &= constant[S->src[r]];
            arm[k] = values[S->src[r++]];
         }
      }
      if (!known || (arm[0] ^ arm[1]) != values[I->src[c]])
         continue;

      /* Copy the expression into this definition, keeping the original for
       * any other users and preserving this instruction's destination. */
      I->op = S->op;
      I->encoding = S->encoding;
      I->nr_srcs = S->nr_srcs;
      I->immediate = S->immediate;
      I->src_abs_mask = S->src_abs_mask;
      I->src_neg_mask = S->src_neg_mask;
      I->select_float_immediates = S->select_float_immediates;
      I->alu_src_uniform_mask = S->alu_src_uniform_mask;
      I->alu_src_immediate_mask = S->alu_src_immediate_mask;
      memcpy(I->alu_src_value, S->alu_src_value, sizeof(I->alu_src_value));
      memcpy(I->src, S->src, S->nr_srcs * sizeof(I->src[0]));
      if (literals == 0xc) {
         I->alu_src_value[2] = S->alu_src_value[3];
         I->alu_src_value[3] = S->alu_src_value[2];
      } else if (literals) {
         /* One literal arm: its slot changes, the register stays last. */
         I->alu_src_immediate_mask ^= 0xc;
         I->alu_src_value[literals == 4 ? 3 : 2] =
            S->alu_src_value[literals == 4 ? 2 : 3];
      } else {
         unsigned a = S->nr_srcs - 2, b = S->nr_srcs - 1;
         I->src[a] = S->src[b];
         I->src[b] = S->src[a];
      }
      return;
   }
}

static void
select_float_sources(struct agx_apple9_vir_instr *I,
                     struct agx_apple9_vir_instr **definitions,
                     const bool *constant, const uint32_t *values)
{
   if (I->alu_src_uniform_mask || I->alu_src_immediate_mask)
      return;
   /* The half comparison form keeps narrow operands in word containers.
    * FP32 modifier folding must not change their interpretation. */
   if (I->encoding == AGX_APPLE9_ENC_HALF_COMPARE_SELECT)
      return;
   bool special = I->encoding == AGX_APPLE9_ENC_FLOAT_SPECIAL;
   bool select = I->op == AGX_APPLE9_VIR_SELECT;
   bool minmax = I->op == AGX_APPLE9_VIR_FMIN || I->op == AGX_APPLE9_VIR_FMAX;
   unsigned condition = I->immediate & 0xff;
   if (select && condition != AGX_APPLE9_SELECT_FEQ &&
       condition != AGX_APPLE9_SELECT_FGT && condition != AGX_APPLE9_SELECT_FLT)
      return;
   if (!special && !select && !minmax && I->op != AGX_APPLE9_VIR_FADD && I->op != AGX_APPLE9_VIR_FSUB &&
       I->op != AGX_APPLE9_VIR_FMUL && I->op != AGX_APPLE9_VIR_FMA)
      return;

   if (I->op == AGX_APPLE9_VIR_FSUB) {
      I->op = AGX_APPLE9_VIR_FADD;
      I->src_neg_mask ^= 2;
   }

   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (minmax && s == 0)
         continue;
      unsigned bit = 1u << s;
      /* Do not fold FP32 sign/abs bit-masks onto a packed FP16 half source:
       * bit 31 is not the sign bit of the low FP16 lane (bit 15). */
      if (I->src_half_mask & bit)
         continue;
      while (true) {
         const struct agx_apple9_vir_instr *producer = definitions[I->src[s]];
         if (!producer || !agx_apple9_instr_is_pure_alu(producer) ||
             producer->nr_srcs != 2 ||
             (producer->op != AGX_APPLE9_VIR_IXOR &&
              producer->op != AGX_APPLE9_VIR_IAND))
            break;
         uint32_t mask = producer->op == AGX_APPLE9_VIR_IXOR
            ? 0x80000000u : 0x7fffffffu;
         int source = -1;
         for (unsigned c = 0; c < 2; ++c) {
            if (constant[producer->src[c]] && values[producer->src[c]] == mask)
               source = 1 - c;
         }
         if (source < 0)
            break;
         if (minmax && producer->op == AGX_APPLE9_VIR_IAND)
            break;
         /* Compose modifiers from the outside in. Absolute value discards
          * any negation inside it, but keeps an outer negation. */
         if (producer->op == AGX_APPLE9_VIR_IAND)
            I->src_abs_mask |= bit;
         else if (!(I->src_abs_mask & bit))
            I->src_neg_mask ^= bit;
         I->src[s] = producer->src[source];
      }
   }

   if (select) {
      /* When both compare inputs are negated, reversing the relation removes
       * both negations while preserving equality and unordered NaNs. A lone
       * src0.neg stays in src0 (bit 65) so src1 remains free for a literal. */
      if ((I->src_neg_mask & 3) == 3) {
         if (condition == AGX_APPLE9_SELECT_FGT) condition = AGX_APPLE9_SELECT_FLT;
         else if (condition == AGX_APPLE9_SELECT_FLT) condition = AGX_APPLE9_SELECT_FGT;
         I->src_neg_mask &= ~3u;
         I->immediate = (I->immediate & ~0xffu) | condition;
      }
      I->encoding = (I->src_abs_mask & 3) ? AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED
                                          : AGX_APPLE9_ENC_SELECT_GPR_WIDE;
      return;
   }
   if (special || minmax)
      return;

   /* Inline constants use an exact minifloat. Preserve unrepresentable
    * constants, half-selected sources, and modifiers which this six-byte form
    * cannot express. */
   if (I->nr_srcs == 2 && !I->src_half_mask) {
      for (int c = 1; c >= 0; --c) {
         unsigned other = 1 - c;
         if (!constant[I->src[c]] || (I->src_abs_mask & (1u << other)))
            continue;
         uint32_t value = values[I->src[c]];
         if (I->src_abs_mask & (1u << c))
            value &= 0x7fffffffu;
         if (I->src_neg_mask & (1u << c))
            value ^= 0x80000000u;
         uint8_t encoded;
         bool low;
         if (!agx_apple9_encode_float_immediate_wide(value, &encoded, &low))
            continue;
         I->op = I->op == AGX_APPLE9_VIR_FMUL ? AGX_APPLE9_VIR_FMUL_IMM
                                             : AGX_APPLE9_VIR_FADD_IMM;
         I->encoding = I->saturate ? AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED
                                  : AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT;
         I->immediate = value;
         I->src[0] = I->src[other];
         I->src_neg_mask = (I->src_neg_mask >> other) & 1;
         I->src_abs_mask = 0;
         I->nr_srcs = 1;
         return;
      }
   }

   if (I->op == AGX_APPLE9_VIR_FMA) {
      I->encoding = (I->src_abs_mask & 3)
         ? AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED
         : AGX_APPLE9_ENC_FLOAT3_EXTENDED;
   } else {
      I->encoding = I->src_abs_mask
         ? (I->op == AGX_APPLE9_VIR_FMUL ? AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED
                                       : AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED)
         : (I->src_neg_mask & 1) ? AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED
                                : AGX_APPLE9_ENC_FLOAT2_COMPACT;
   }
   select_saturated_encoding(I);
}

/* Compare operand B of a select or predicate holds a literal: a minifloat
 * with sign for floating conditions, an 8-bit integer otherwise. A constant
 * operand A moves to B by reversing an ordered relation. Run after the
 * selected-value literals, which leave the compare operands first. */
static void
inline_compare_literal(struct agx_apple9_vir_instr *I, const bool *constant,
                       const uint32_t *values)
{
   if (!agx_apple9_compare_literal_slot(I, 1) || I->nr_srcs < 2 ||
       ((I->alu_src_uniform_mask | I->alu_src_immediate_mask) & 3))
      return;
   const bool is_float = agx_apple9_compare_is_float(I);
   const bool predicate = I->op == AGX_APPLE9_VIR_PREDICATE_COMPARE;
   const bool extended = I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
   const unsigned modifiers = I->src_abs_mask | I->src_neg_mask;

   uint32_t literal[2];
   bool can[2];
   for (unsigned s = 0; s < 2; ++s) {
      literal[s] = values[I->src[s]];
      if (is_float) {
         if (I->src_abs_mask & BITFIELD_BIT(s)) literal[s] &= 0x7fffffffu;
         if (I->src_neg_mask & BITFIELD_BIT(s)) literal[s] ^= 0x80000000u;
      }
      can[s] = constant[I->src[s]] &&
               (is_float || !(modifiers & BITFIELD_BIT(s))) &&
               agx_apple9_compare_literal_encodable(I, literal[s]);
   }

   if (!can[1]) {
      /* a OP k is k OP' a for the reversed ordered relation; equality is
       * symmetric. NaN operands compare false either way. The register
       * moving to A must not carry modifiers, which A cannot take here. */
      const unsigned condition = I->immediate & 0xff;
      /* A select's equality bit does not compose like "or equal": with ILT
       * it is integer equality, and with IGT it is not equality (see the
       * select lowering). Only its symmetric forms commute. */
      const bool equality = !predicate && (I->immediate & AGX_APPLE9_SELECT_EQUALITY);
      unsigned reversed;
      if (extended)
         reversed = condition == AGX_APPLE9_PREDICATE_EXT_FEQ ||
                    condition == AGX_APPLE9_PREDICATE_EXT_IEQ ? condition : ~0u;
      else if (equality)
         reversed = condition == AGX_APPLE9_SELECT_ILT ||
                    condition == AGX_APPLE9_SELECT_FEQ ? condition : ~0u;
      else if (condition == AGX_APPLE9_SELECT_FEQ && !predicate)
         reversed = condition;
      else if (condition >= AGX_APPLE9_SELECT_FGT &&
               condition <= AGX_APPLE9_SELECT_ILT)
         reversed = condition ^ 1;
      else
         reversed = ~0u;
      if (!can[0] || reversed == ~0u || (predicate && (modifiers & 2)))
         return;
      uint32_t src = I->src[0];
      I->src[0] = I->src[1]; I->src[1] = src;
      I->src_abs_mask = (I->src_abs_mask & ~3u) | ((I->src_abs_mask & 2) >> 1);
      I->src_neg_mask = (I->src_neg_mask & ~3u) | ((I->src_neg_mask & 2) >> 1);
      I->immediate = (I->immediate & ~0xffu) | reversed;
      literal[1] = literal[0];
   }

   I->alu_src_immediate_mask |= 2;
   I->alu_src_value[1] = literal[1];
   I->src_abs_mask &= ~2u;
   I->src_neg_mask &= ~2u;
   if (I->encoding == AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED &&
       !(I->src_abs_mask & 3))
      I->encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE;
   for (unsigned s = 1; s + 1 < I->nr_srcs; ++s)
      I->src[s] = I->src[s + 1];
   --I->nr_srcs;
}

/* Integer add/subtract, multiply-add, logic and select operand descriptors
 * hold an 8-bit zero-extended literal. Give a constant register operand that
 * form when its slot admits it, commuting it there or negating an add when
 * that is exact. Remaining constants may still select uniform words. */
static void
inline_integer_immediates(struct agx_apple9_vir_instr *I, const bool *constant,
                          const uint32_t *values)
{
   bool add = I->op == AGX_APPLE9_VIR_IADD || I->op == AGX_APPLE9_VIR_ISUB;
   bool mad = I->op == AGX_APPLE9_VIR_IMAD && !I->immediate;
   bool mul_wide = I->op == AGX_APPLE9_VIR_IMUL_WIDE && !I->immediate;
   bool iminmax = (I->op == AGX_APPLE9_VIR_IMIN || I->op == AGX_APPLE9_VIR_IMAX ||
                   I->op == AGX_APPLE9_VIR_UMIN || I->op == AGX_APPLE9_VIR_UMAX) &&
                  I->encoding == AGX_APPLE9_ENC_MINMAX_COMPACT;
   bool logic = I->op == AGX_APPLE9_VIR_IAND || I->op == AGX_APPLE9_VIR_IOR ||
                I->op == AGX_APPLE9_VIR_IXOR;
   bool select = I->op == AGX_APPLE9_VIR_SELECT &&
                 (I->encoding == AGX_APPLE9_ENC_SELECT_GPR_WIDE ||
                  I->encoding == AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED);
   unsigned arity = select ? 4 : mad ? 3 : 2;
   if (!(add || mad || mul_wide || iminmax || logic || select) ||
       (!iminmax && !agx_apple9_integer_immediate_encoding(I->encoding)) ||
       I->saturate || I->src_sext_mask ||
       I->alu_src_uniform_mask || I->alu_src_immediate_mask ||
       I->nr_srcs != arity || I->dest_components != (mul_wide ? 2 : 1))
      return;

   /* Operand values, with the literal each would take inline. Every source
    * of these encodings is one 32-bit register; there is no pair form, and
    * 64-bit integer arithmetic reaches VIR split into 32-bit halves, so a
    * constant low word is itself an exact 32-bit operand. */
   uint32_t literal[4];
   bool fits[4];
   for (unsigned s = 0; s < arity; ++s) {
      ASSERTED const struct agx_apple9_operand_constraint *constraint =
         agx_apple9_find_operand(I->encoding, AGX_APPLE9_OPERAND_SRC0 + s);
      assert(constraint && !(constraint->widths & AGX_APPLE9_WIDTH_64));
      literal[s] = values[I->src[s]];
      fits[s] = constant[I->src[s]] && literal[s] <= UINT8_MAX;
   }

   if (add) {
      if (I->immediate) {
         /* Shifted src1 cannot be commuted or turned into an immediate; src0
          * still takes an 8-bit immediate (including 0 for `0 + (x << k)`). */
         fits[1] = false;
      } else {
         /* An exact negation moves a small negative addend into the
          * subtrahend: x + (-c) is x - c, and x - (-c) is x + c. */
         bool sub = I->op == AGX_APPLE9_VIR_ISUB;
         if (!sub && !fits[1] && fits[0]) {
            uint32_t src = I->src[0];
            I->src[0] = I->src[1]; I->src[1] = src;
            literal[1] = literal[0]; fits[1] = true;
            fits[0] = constant[I->src[0]] && values[I->src[0]] <= UINT8_MAX;
            literal[0] = values[I->src[0]];
         }
         if (!fits[1] && constant[I->src[1]] && -literal[1] <= UINT8_MAX &&
             (sub || !fits[0])) {
            I->op = sub ? AGX_APPLE9_VIR_IADD : AGX_APPLE9_VIR_ISUB;
            literal[1] = -literal[1];
            fits[1] = true;
         } else if (!fits[1] && !sub && constant[I->src[0]] &&
                    -values[I->src[0]] <= UINT8_MAX) {
            uint32_t src = I->src[0];
            I->src[0] = I->src[1]; I->src[1] = src;
            I->op = AGX_APPLE9_VIR_ISUB;
            literal[1] = -values[src];
            fits[1] = true;
         }
      }
   } else if (mad || mul_wide || iminmax || logic) {
      /* Commutative binary/multiplicand slots: A has no literal form. */
      if (fits[0] && !fits[1]) {
         uint32_t src = I->src[0];
         I->src[0] = I->src[1]; I->src[1] = src;
         fits[1] = true; fits[0] = false;
         literal[1] = literal[0];
      }
      fits[0] = false;
   } else {
      /* Comparison operands are handled by inline_compare_literal. Selected
       * arms share bit 18: 1 = 8-bit integer immediates (no arm modifiers),
       * 0 = wide 9-bit signed minifloat immediates (compatible with arm
       * abs/neg modifiers). Pick the mode that inlines more arms. */
      fits[0] = fits[1] = false;
      bool has_arm_mods = ((I->src_abs_mask | I->src_neg_mask) & 0xc) != 0;
      bool int_ok[4] = {false};
      bool flt_ok[4] = {false};
      uint32_t flt_lit[4] = {0};
      for (unsigned s = 2; s < 4; ++s) {
         if (!constant[I->src[s]])
            continue;
         int_ok[s] = !has_arm_mods && literal[s] <= UINT8_MAX;
         uint32_t fval = literal[s];
         if (I->src_abs_mask & BITFIELD_BIT(s)) fval &= 0x7fffffffu;
         if (I->src_neg_mask & BITFIELD_BIT(s)) fval ^= 0x80000000u;
         uint8_t encoded;
         bool low;
         if (agx_apple9_encode_float_immediate_wide(fval, &encoded, &low)) {
            flt_ok[s] = true;
            flt_lit[s] = fval;
         }
      }
      unsigned int_cnt = (unsigned)int_ok[2] + (unsigned)int_ok[3];
      unsigned flt_cnt = (unsigned)flt_ok[2] + (unsigned)flt_ok[3];
      if (flt_cnt > int_cnt) {
         I->select_float_immediates = true;
         for (unsigned s = 2; s < 4; ++s) {
            fits[s] = flt_ok[s];
            if (fits[s]) {
               literal[s] = flt_lit[s];
               I->src_abs_mask &= ~BITFIELD_BIT(s);
               I->src_neg_mask &= ~BITFIELD_BIT(s);
            }
         }
      } else {
         I->select_float_immediates = false;
         fits[2] = int_ok[2];
         fits[3] = int_ok[3];
      }
   }
   /* Both add operands constant were folded already; keep one register. */
   if (add && fits[0] && fits[1])
      fits[0] = false;

   unsigned source = 0;
   for (unsigned s = 0; s < arity; ++s) {
      if (fits[s]) {
         I->alu_src_immediate_mask |= BITFIELD_BIT(s);
         I->alu_src_value[s] = literal[s];
      } else {
         I->src[source++] = I->src[s];
      }
   }
   I->nr_srcs = source;
}

/* A pointer-pair load whose element index is zero has an index-free form:
 * the pair and the byte displacement alone form the address. */
static void
drop_zero_load_index(struct agx_apple9_vir_instr *I, const bool *constant,
                     const uint32_t *values)
{
   if (I->op != AGX_APPLE9_VIR_DEVICE_LOAD ||
       I->encoding != AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT || I->nr_srcs != 3 ||
       !constant[I->src[0]] || values[I->src[0]] != 0)
      return;
   I->encoding = AGX_APPLE9_ENC_DEVICE_LOAD_POINTER;
   I->src[0] = I->src[1];
   I->src[1] = I->src[2];
   I->nr_srcs = 2;
   I->memory_index_shift = 0;
   I->device_load_index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_NONE;
}

/* Fold uniform-loaded bindless texture/sampler handles (u0..u127) directly
 * into 20B TEXTURE_SAMPLE (bit 71 / bit 72) and even zero-extended uniform
 * pairs (u0..u126 with u+1 == 0) into 16B IMAGE_WRITE (byte 14/15 = 09/01),
 * eliminating pinned IOR_UNIFORM GPRs across shaders. */
static void
inline_texture_image_uniforms(struct agx_apple9_vir_instr *I,
                              struct agx_apple9_vir_instr **definitions,
                              const uint16_t *uniform_read,
                              const bool *constant, const uint32_t *values,
                              const struct agx_apple9_uniform_constants *constants)
{
   if (I->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
      if (!I->texture_bindless && !I->sampler_bindless)
         return;
      if (I->sampler_bindless && I->nr_srcs >= 1) {
         uint32_t s_vreg = I->src[I->nr_srcs - 1];
         if (uniform_read[s_vreg] && (uniform_read[s_vreg] - 1u) < 128u) {
            I->sampler_index = (uint8_t)(uniform_read[s_vreg] - 1u);
            I->sampler_bindless = false;
            I->sampler_uniform_bindless = true;
            --I->nr_srcs;
         }
      }
      if (I->texture_bindless && I->nr_srcs >= 1u + I->sampler_bindless) {
         unsigned tex_pos = I->nr_srcs - 1u - I->sampler_bindless;
         uint32_t t_vreg = I->src[tex_pos];
         if (uniform_read[t_vreg] && (uniform_read[t_vreg] - 1u) < 128u) {
            I->texture_index = (uint8_t)(uniform_read[t_vreg] - 1u);
            I->texture_bindless = false;
            I->texture_uniform_bindless = true;
            if (I->sampler_bindless)
               I->src[tex_pos] = I->src[tex_pos + 1];
            --I->nr_srcs;
         }
      }
      return;
   }

   if (I->op == AGX_APPLE9_VIR_IMAGE_STORE && !I->texture_uniform_bindless &&
       I->nr_srcs == 10) {
      struct agx_apple9_vir_instr *cdef = definitions[I->src[8]];
      const bool from_collect = cdef && cdef->op == AGX_APPLE9_VIR_COLLECT &&
                                cdef->dest == I->src[8] &&
                                cdef->dest_components == 2 &&
                                cdef->nr_srcs == 2 &&
                                I->src[9] == I->src[8] + 1;
      uint32_t lo_v = from_collect ? cdef->src[0] : I->src[8];
      uint32_t hi_v = from_collect ? cdef->src[1] : I->src[9];
      if (!uniform_read[lo_v])
         return;
      unsigned u = uniform_read[lo_v] - 1u;
      if ((u & 1u) || u >= 128u)
         return;
      bool hi_ok = uniform_read[hi_v] == u + 2u;
      if (!hi_ok && constant[hi_v] && values[hi_v] == 0 && constants &&
          u + 1u >= constants->base &&
          u + 1u < constants->base + constants->count &&
          constants->values[(u + 1u) - constants->base] == 0)
         hi_ok = true;
      if (!hi_ok)
         return;
      I->texture_index = (uint8_t)u;
      I->texture_uniform_bindless = true;
      I->nr_srcs = 8;
   }
}

static void
inline_alu_sources(struct agx_apple9_vir_instr *I, const uint16_t *uniform_read,
                      const bool *constant, const uint32_t *values)
{
   bool special = I->encoding == AGX_APPLE9_ENC_FLOAT_SPECIAL &&
                  agx_apple9_supports_special_uniform(I->op);
   bool unary = special || I->op == AGX_APPLE9_VIR_BIT_COUNT ||
                I->op == AGX_APPLE9_VIR_UFIND_MSB ||
                I->op == AGX_APPLE9_VIR_BIT_REVERSE ||
                 I->op == AGX_APPLE9_VIR_U2F32 ||
                I->op == AGX_APPLE9_VIR_I2F32 ||
                I->op == AGX_APPLE9_VIR_F2U32 ||
                I->op == AGX_APPLE9_VIR_F2I32;
   bool shift = I->op == AGX_APPLE9_VIR_ISHR || I->op == AGX_APPLE9_VIR_USHR ||
                I->op == AGX_APPLE9_VIR_ISHL;
   bool predicate = I->op == AGX_APPLE9_VIR_PREDICATE_COMPARE &&
      (I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT ||
       I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED);
   bool select = I->op == AGX_APPLE9_VIR_SELECT &&
      I->encoding != AGX_APPLE9_ENC_HALF_COMPARE_SELECT;
   bool fma = I->op == AGX_APPLE9_VIR_FMA;
   bool minmax_float = (I->op == AGX_APPLE9_VIR_FMIN || I->op == AGX_APPLE9_VIR_FMAX) &&
                       I->encoding == AGX_APPLE9_ENC_MINMAX_COMPACT;
   bool integer = I->op == AGX_APPLE9_VIR_IADD || I->op == AGX_APPLE9_VIR_ISUB ||
                  I->op == AGX_APPLE9_VIR_IMAD || I->op == AGX_APPLE9_VIR_IMUL_WIDE;
   unsigned arity = unary ? 1 : shift ? I->nr_srcs : select ? 4 : fma || I->op == AGX_APPLE9_VIR_IMAD ? 3 : 2;
   bool float2 = I->op == AGX_APPLE9_VIR_FADD || I->op == AGX_APPLE9_VIR_FSUB ||
                 I->op == AGX_APPLE9_VIR_FMUL;
   bool binary = float2 || I->op == AGX_APPLE9_VIR_IAND ||
      I->op == AGX_APPLE9_VIR_IOR || I->op == AGX_APPLE9_VIR_IXOR ||
      I->op == AGX_APPLE9_VIR_IMIN || I->op == AGX_APPLE9_VIR_IMAX ||
      I->op == AGX_APPLE9_VIR_UMIN || I->op == AGX_APPLE9_VIR_UMAX ||
      I->op == AGX_APPLE9_VIR_FMIN || I->op == AGX_APPLE9_VIR_FMAX;
   /* A later pass may fill logical slots left as registers by an earlier
    * one, for example after literal folding selected some operands. */
   const unsigned filled = I->alu_src_uniform_mask | I->alu_src_immediate_mask;
   if ((!unary && !shift && !predicate && !select && !fma && !binary && !integer) ||
       I->nr_srcs + util_bitcount(filled) != arity)
      return;

   uint8_t file[4] = {0};
   uint32_t value[4] = {0};
   uint32_t reg[4] = {0};
   for (unsigned s = 0, k = 0; s < arity; ++s) {
      if (filled & BITFIELD_BIT(s))
         continue;
      reg[s] = I->src[k++];
      /* Packed half-register sources must remain in GPRs (uniform_mask &
       * half_mask is illegal in Apple9 encodings). */
      if (I->src_half_mask & BITFIELD_BIT(s))
         continue;
      /* Prefer encodable minifloat literals over constant-table uniforms on
       * FMA and floating min/max so a single pass matches literal selection. */
      if ((fma || minmax_float) && !filled && constant[reg[s]]) {
         /* The literal takes this slot's modifiers below. FMA operand B
          * and min/max operand A have no fourth mantissa bit. */
         uint32_t literal = values[reg[s]];
         if (I->src_abs_mask & BITFIELD_BIT(s)) literal &= 0x7fffffffu;
         if (I->src_neg_mask & BITFIELD_BIT(s)) literal ^= 0x80000000u;
         uint8_t encoded;
         bool low;
         bool wide = minmax_float || s == 2;
         if (wide ? agx_apple9_encode_float_immediate_wide(literal, &encoded, &low)
                  : agx_apple9_encode_float_immediate(literal, &encoded)) {
            file[s] = AGX_APPLE9_FILE_IMMEDIATE;
            value[s] = values[reg[s]];
            continue;
         }
      }
      if (uniform_read[reg[s]]) {
         file[s] = AGX_APPLE9_FILE_UNIFORM;
         value[s] = uniform_read[reg[s]] - 1;
      }
   }
   /* Floating min/max takes a literal only as operand B. They commute for
    * NaN operands (the result is the other value) but not for zeros of
    * opposite sign, where G16 returns operand B: max(+0, -0) is -0. A zero
    * literal therefore stays in operand A's register (or a uniform if none is
    * used in operand B). */
   if (minmax_float && file[0] == AGX_APPLE9_FILE_IMMEDIATE &&
       file[1] != AGX_APPLE9_FILE_IMMEDIATE && !(I->src_neg_mask & 2) &&
       (value[0] & 0x7fffffffu) != 0) {
      uint32_t src = I->src[0];
      I->src[0] = I->src[1]; I->src[1] = src;
      uint8_t f = file[0]; file[0] = file[1]; file[1] = f;
      uint32_t v = value[0]; value[0] = value[1]; value[1] = v;
      reg[0] = I->src[0]; reg[1] = I->src[1];
      I->src_neg_mask = ((I->src_neg_mask & 1) << 1) | ((I->src_neg_mask & 2) >> 1);
   }
   if (minmax_float && file[0] == AGX_APPLE9_FILE_IMMEDIATE) {
      if (!file[1] && uniform_read[reg[0]]) {
         file[0] = AGX_APPLE9_FILE_UNIFORM;
         value[0] = uniform_read[reg[0]] - 1;
      } else {
         file[0] = 0;
      }
   }
   if (filled) {
      /* Only one of the two leading operands of a plain binary operation may
       * leave the register file; keep whichever was already selected. */
      if (!shift && !predicate && !select && !fma && !float2 && !integer &&
          (filled & 3) && (file[0] || file[1]))
         file[0] = file[1] = 0;
      unsigned source = 0;
      for (unsigned s = 0; s < arity; ++s) {
         if (filled & BITFIELD_BIT(s))
            continue;
         if (file[s] == AGX_APPLE9_FILE_UNIFORM) {
            I->alu_src_uniform_mask |= BITFIELD_BIT(s);
            I->alu_src_value[s] = value[s];
         } else {
            I->src[source++] = reg[s];
         }
      }
      I->nr_srcs = source;
      if (special && I->alu_src_uniform_mask)
         I->encoding = AGX_APPLE9_ENC_FLOAT_SPECIAL_UNIFORM;
      if (fma) {
         I->encoding = (I->src_abs_mask & 3)
            ? AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED
            : AGX_APPLE9_ENC_FLOAT3_EXTENDED;
         select_saturated_encoding(I);
      }
      return;
   }
   /* FMA's second multiplicand has a native alternative file selector.
    * Commuting the product preserves its absolute/negate composition,
    * provided neither multiplicand carries a half-register selection.
    * Extended FMA also takes a uniform in A and an immediate in B, so
    * commuting (IMMEDIATE, UNIFORM) -> (UNIFORM, IMMEDIATE) avoids
    * materializing either operand. */
   if (fma && file[0] &&
       (!file[1] ||
        (file[0] == AGX_APPLE9_FILE_IMMEDIATE &&
         file[1] == AGX_APPLE9_FILE_UNIFORM)) &&
       !(I->src_half_mask & 3)) {
      uint32_t src = I->src[0];
      I->src[0] = I->src[1]; I->src[1] = src;
      uint8_t f = file[0]; file[0] = file[1]; file[1] = f;
      uint32_t v = value[0]; value[0] = value[1]; value[1] = v;
      reg[0] = I->src[0]; reg[1] = I->src[1];
      I->src_abs_mask = (I->src_abs_mask & 4) |
         ((I->src_abs_mask & 1) << 1) | ((I->src_abs_mask & 2) >> 1);
      I->src_neg_mask = (I->src_neg_mask & 4) |
         ((I->src_neg_mask & 1) << 1) | ((I->src_neg_mask & 2) >> 1);
   }
   /* Extended FMA has a uniform selector for A as well. Preserve the
    * compact form when commuting a lone uniform to B; A immediates still
    * fall back to a constant-table uniform if available, otherwise
    * materialization. Fixed integer operands are independent. */
   if (fma && file[0] == AGX_APPLE9_FILE_IMMEDIATE) {
      if (uniform_read[reg[0]]) {
         file[0] = AGX_APPLE9_FILE_UNIFORM;
         value[0] = uniform_read[reg[0]] - 1;
      } else {
         file[0] = 0;
      }
   } else if (!shift && !predicate && !select && !fma && !float2 && !integer &&
              file[0] && file[1]) {
      file[0] = 0;
   }

   unsigned source = 0;
   for (unsigned s = 0; s < arity; ++s) {
      if (file[s] == AGX_APPLE9_FILE_UNIFORM) {
         I->alu_src_uniform_mask |= BITFIELD_BIT(s);
         I->alu_src_value[s] = value[s];
      } else if (file[s] == AGX_APPLE9_FILE_IMMEDIATE) {
         if (I->src_abs_mask & BITFIELD_BIT(s)) value[s] &= 0x7fffffffu;
         if (I->src_neg_mask & BITFIELD_BIT(s)) value[s] ^= 0x80000000u;
         I->src_abs_mask &= ~BITFIELD_BIT(s);
         I->src_neg_mask &= ~BITFIELD_BIT(s);
         I->alu_src_immediate_mask |= BITFIELD_BIT(s);
         I->alu_src_value[s] = value[s];
      } else {
         I->src[source++] = I->src[s];
      }
   }
   I->nr_srcs = source;
   if (special && I->alu_src_uniform_mask)
      I->encoding = AGX_APPLE9_ENC_FLOAT_SPECIAL_UNIFORM;
   if (fma) {
      I->encoding = (I->src_abs_mask & 3)
         ? AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED
         : AGX_APPLE9_ENC_FLOAT3_EXTENDED;
      select_saturated_encoding(I);
   }
}

static bool
fold_saturate(struct agx_apple9_vir_program *p, const uint32_t *replacement)
{
   struct agx_apple9_vir_instr **definitions =
      CALLOC(p->value_count, sizeof(*definitions));
   unsigned *uses = CALLOC(p->value_count, sizeof(*uses));
   unsigned *scope = CALLOC(p->value_count, sizeof(*scope));
   if (!definitions || !uses || !scope) {
      free(definitions); free(uses); free(scope);
      return false;
   }
   unsigned epoch = 0;
   struct agx_apple9_block *block = NULL;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (block != agx_apple9_instr_block(I)) {
         block = agx_apple9_instr_block(I);
         ++epoch;
      }
      switch (I->op) {
      case AGX_APPLE9_VIR_STORE_UNIFORM:
      case AGX_APPLE9_VIR_OBJECT_CALL:
      case AGX_APPLE9_VIR_OBJECT_RETURN:
      case AGX_APPLE9_VIR_OBJECT_RESUME:
      case AGX_APPLE9_VIR_ABI_RETURN:
      case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
      case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
      case AGX_APPLE9_VIR_EXEC_MASK_POP:
      case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
      case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
      case AGX_APPLE9_VIR_LOOP_MASK_POP:
      case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
         ++epoch;
         break;
      default:
         break;
      }
      bool eliminated = I->dest != AGX_APPLE9_VREG_INVALID &&
                        replacement[I->dest] != I->dest;
      if (!eliminated) {
         unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
         for (unsigned s = 0; s < nr_srcs; ++s) {
            uint32_t v = agx_apple9_vir_src(I, s);
            if (v < p->value_count)
               ++uses[v];
         }
      }
      for (unsigned c = 0; c < I->dest_components; ++c) {
         definitions[I->dest + c] = I;
         scope[I->dest + c] = epoch;
      }
   }
   for (unsigned i = 0; i < p->live_out_count; ++i)
      ++uses[p->live_out[i]];
   if (p->output != AGX_APPLE9_VREG_INVALID)
      ++uses[p->output];

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (I->op != AGX_APPLE9_VIR_FSAT)
         continue;
      unsigned source = I->src[0];
      struct agx_apple9_vir_instr *producer = definitions[source];
      /* Keep the unclamped value when it has another use. Moving a pure
       * producer into its sole clamp use preserves SSA and source lifetime;
       * an execution-mask transition would change its active lanes. */
      if (producer && producer->op != AGX_APPLE9_VIR_FSAT &&
          producer->dest_components == 1 && uses[source] == 1 &&
          uses[I->dest] && scope[source] == scope[I->dest] &&
          agx_apple9_supports_saturate(producer)) {
         uint32_t dest = I->dest;
         *I = *producer;
         I->dest = dest;
      } else {
         /* Adding positive zero applies the ordinary FP32 canonicalization
          * before clamping. This also handles a load, select or rounding
          * result whose producer has no supported saturation modifier. */
         I->op = AGX_APPLE9_VIR_FADD_IMM;
         I->encoding = AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED;
         I->immediate = 0;
      }
      I->saturate = true;
      select_saturated_encoding(I);
   }
   free(definitions); free(uses); free(scope);
   return true;
}

/* Fuse a sole, adjacent integer producer into publication. Keeping the
 * operation at its store preserves the execution mask and uniform write order.
 * This is a preamble lowering optimization; uniform words remain outside SSA
 * GPR allocation, and publication is still an explicit side effect. */
static bool
fold_uniform_stores(struct agx_apple9_vir_program *p)
{
   agx_apple9_invalidate_uses(p);
   if (!agx_apple9_analyze_uses(p))
      return false;
   struct agx_apple9_vir_instr **remove = CALLOC(p->instruction_count, sizeof(*remove));
   if (!remove)
      return false;
   unsigned nr_remove = 0;
   for (unsigned i = 1; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = p->instructions[i];
      struct agx_apple9_vir_instr *alu = p->instructions[i - 1];
      if (store->op != AGX_APPLE9_VIR_STORE_UNIFORM ||
          store->encoding != AGX_APPLE9_ENC_STORE_UNIFORM || store->nr_srcs != 1 ||
          store->immediate >= AGX_APPLE9_UNIFORM_COUNT ||
          alu->dest != store->src[0] ||
          alu->saturate || alu->src_sext_mask ||
          alu->src_half_mask || alu->src_hi_mask ||
          agx_apple9_instr_block(alu) != agx_apple9_instr_block(store) ||
          alu->alu_src_uniform_mask || alu->alu_src_immediate_mask)
         continue;
      enum agx_apple9_encoding encoding;
      switch (alu->op) {
      case AGX_APPLE9_VIR_IADD:
      case AGX_APPLE9_VIR_ISUB:
         if (alu->immediate)
            continue;
         encoding = AGX_APPLE9_ENC_STORE_UNIFORM_ADD;
         break;
      case AGX_APPLE9_VIR_IMAD:
         encoding = AGX_APPLE9_ENC_STORE_UNIFORM_MAD;
         break;
      case AGX_APPLE9_VIR_IMUL_WIDE:
         encoding = AGX_APPLE9_ENC_STORE_UNIFORM_MUL_WIDE;
         break;
      default:
         continue;
      }
      bool wide = alu->op == AGX_APPLE9_VIR_IMUL_WIDE;
      struct agx_apple9_vir_instr *high = i + 1 < p->instruction_count ? p->instructions[i + 1] : NULL;
      if (wide && ((store->immediate & 1) || store->immediate + 1 >= AGX_APPLE9_UNIFORM_COUNT ||
                   !high || high->op != AGX_APPLE9_VIR_STORE_UNIFORM ||
                   high->encoding != AGX_APPLE9_ENC_STORE_UNIFORM || high->nr_srcs != 1 ||
                   high->src[0] != alu->dest + 1 || high->immediate != store->immediate + 1 ||
                   agx_apple9_instr_block(high) != agx_apple9_instr_block(store)))
         continue;
      bool sole = alu->dest_components == (wide ? 2 : 1);
      for (unsigned c = 0; c < alu->dest_components; ++c) {
         unsigned v = alu->dest + c;
         const struct agx_apple9_use *use = agx_apple9_uses(p, v);
         sole &= use && !use->next && use->instruction == (c ? high : store) &&
                 p->output != v && p->fixed_phys[v] == AGX_APPLE9_PHYS_INVALID &&
                 p->max_phys[v] == AGX_APPLE9_PHYS_INVALID;
         for (unsigned l = 0; l < p->live_out_count; ++l)
            sole &= p->live_out[l] != v;
      }
      if (!sole)
         continue;
      store->encoding = encoding;
      store->uniform_op = alu->op;
      store->uniform_signed = wide && alu->immediate;
      store->nr_srcs = alu->nr_srcs;
      memcpy(store->src, alu->src, alu->nr_srcs * sizeof(alu->src[0]));
      if (wide) {
         remove[nr_remove++] = high;
         ++i;
      }
   }
   if (nr_remove)
      agx_apple9_vir_remove_many(p, remove, nr_remove);
   free(remove);
   agx_apple9_invalidate_uses(p);
   return true;
}

/* The logical operand slot of register source `source`, skipping slots
 * filled by uniform words or literals. */
static int
register_slot(const struct agx_apple9_vir_instr *I, unsigned source)
{
   const unsigned filled = I->alu_src_uniform_mask | I->alu_src_immediate_mask;
   for (unsigned s = 0, k = 0; s < 4; ++s) {
      if (filled & BITFIELD_BIT(s))
         continue;
      if (k++ == source)
         return s;
   }
   return -1;
}

/* Whether register source `source` of I is the constant `value`: a literal
 * move, or a uniform word the constant table names. */
static bool
source_is_constant(const struct agx_apple9_vir_program *p,
                   const struct agx_apple9_uniform_constants *constants,
                   const struct agx_apple9_vir_instr *I, unsigned slot,
                   uint32_t value)
{
   if (I->alu_src_immediate_mask & BITFIELD_BIT(slot))
      return I->alu_src_value[slot] == value;
   if (I->alu_src_uniform_mask & BITFIELD_BIT(slot)) {
      unsigned word = I->alu_src_value[slot];
      return constants && word >= constants->base &&
             word < constants->base + constants->count &&
             constants->values[word - constants->base] == value;
   }
   int source = 0;
   for (unsigned s = 0; s < slot; ++s)
      source += !((I->alu_src_uniform_mask | I->alu_src_immediate_mask) & BITFIELD_BIT(s));
   const struct agx_apple9_vir_instr *D = agx_apple9_definition(p, I->src[source]);
   return D && D->op == AGX_APPLE9_VIR_IMM && D->immediate == value;
}

/* Recognize a 16-bit extract or extension of a 32-bit register x: x & 0xffff,
 * x >> 16 (logical or arithmetic), and (x << 16) >> 16 written as a shift or
 * a multiply by 0x10000. */
static bool
half_extract(const struct agx_apple9_vir_program *p,
             const struct agx_apple9_uniform_constants *constants,
             const struct agx_apple9_vir_instr *D, uint32_t *x, bool *hi,
             bool *sext)
{
   if (!D || D->dest_components != 1 || D->src_half_mask || D->saturate)
      return false;
   if (D->op == AGX_APPLE9_VIR_UNPACK_HALF && D->nr_srcs == 1 &&
       D->encoding == AGX_APPLE9_ENC_HALF_TO_FLOAT && D->immediate <= 1) {
      *x = D->src[0];
      *hi = D->immediate;
      *sext = false;
      return true;
   }
   if (D->op == AGX_APPLE9_VIR_IAND && D->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED) {
      for (unsigned s = 0; s < 2; ++s) {
         if (!source_is_constant(p, constants, D, s, 0xffff))
            continue;
         for (unsigned k = 0; k < D->nr_srcs; ++k) {
            if (register_slot(D, k) != (int)(1 - s))
               continue;
            *x = D->src[k];
            *hi = false;
            *sext = false;
            return true;
         }
      }
      return false;
   }
   const bool shift = D->op == AGX_APPLE9_VIR_USHR || D->op == AGX_APPLE9_VIR_ISHR;
   if (!shift || D->nr_srcs != 1 || D->alu_src_uniform_mask || D->immediate != 16)
      return false;
   const struct agx_apple9_vir_instr *Y = agx_apple9_definition(p, D->src[0]);
   if (D->op == AGX_APPLE9_VIR_ISHR && Y && Y->dest_components == 1 &&
       !Y->src_half_mask && !Y->alu_src_uniform_mask) {
      if (Y->op == AGX_APPLE9_VIR_ISHL && Y->nr_srcs == 1 && Y->immediate == 16) {
         *x = Y->src[0];
         *hi = false;
         *sext = true;
         return true;
      }
      if (Y->op == AGX_APPLE9_VIR_IMAD && !Y->immediate &&
          Y->encoding == AGX_APPLE9_ENC_INT_MAD_EXTENDED &&
          source_is_constant(p, constants, Y, 2, 0)) {
         for (unsigned s = 0; s < 2; ++s) {
            int reg = register_slot(Y, 0) == (int)(1 - s) ? 0
                    : register_slot(Y, 1) == (int)(1 - s) ? 1 : -1;
            if (reg < 0 || !source_is_constant(p, constants, Y, s, 0x10000))
               continue;
            *x = Y->src[reg];
            *hi = false;
            *sext = true;
            return true;
         }
      }
   }
   *x = D->src[0];
   *hi = true;
   *sext = D->op == AGX_APPLE9_VIR_ISHR;
   return true;
}

/* Integer add/subtract and multiply-add read 16-bit halves of a register
 * directly, zero- or sign-extended (Metal's encodings of our own kernels).
 * Fold a recognized extract into every reader when all of them can take
 * it, so the extract and its mask constant disappear without extending any
 * other register's lifetime. */
static bool
fold_half_sources(struct agx_apple9_vir_program *p,
                  const struct agx_apple9_uniform_constants *constants)
{
   if (!agx_apple9_analyze_uses(p))
      return false;

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *D = p->instructions[i];
      uint32_t x;
      bool hi, sext;
      if (D->dest == AGX_APPLE9_VREG_INVALID ||
          !half_extract(p, constants, D, &x, &hi, &sext))
         continue;
      bool live_out = p->output == D->dest;
      for (unsigned l = 0; l < p->live_out_count; ++l)
         live_out |= p->live_out[l] == D->dest;
      const struct agx_apple9_use *uses = agx_apple9_uses(p, D->dest);
      if (live_out || !uses)
         continue;
      const bool fp16 = D->op == AGX_APPLE9_VIR_UNPACK_HALF;
      bool all = true;
      for (const struct agx_apple9_use *u = uses; u && all; u = u->next) {
         const struct agx_apple9_vir_instr *R = u->instruction;
         const int slot = register_slot(R, u->source);
         const bool add = R->op == AGX_APPLE9_VIR_IADD || R->op == AGX_APPLE9_VIR_ISUB;
         const bool mad = R->op == AGX_APPLE9_VIR_IMAD && !R->immediate;
         const bool float2 = (R->op == AGX_APPLE9_VIR_FADD || R->op == AGX_APPLE9_VIR_FSUB ||
                              R->op == AGX_APPLE9_VIR_FMUL) &&
                             (R->encoding == AGX_APPLE9_ENC_FLOAT2_COMPACT ||
                              R->encoding == AGX_APPLE9_ENC_FLOAT2_BASE ||
                              R->encoding == AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED ||
                              R->encoding == AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED ||
                              R->encoding == AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED);
         const bool fma = R->op == AGX_APPLE9_VIR_FMA;
         bool ok;
         /* FMA's multiplicands commute: a half in A moves to B, which takes
          * a low half, when B is an ordinary register. */
         const bool fma_swap = fma && slot == 0 && !hi && R->nr_srcs >= 2 &&
            !((R->alu_src_uniform_mask | R->alu_src_immediate_mask | R->src_half_mask) & 3) &&
            R->src[1] != D->dest;
         if (fp16)
            ok = (float2 && slot < 2) ||
                 (fma && (slot == 2 || (slot == 1 && !hi) || fma_swap));
         else
            ok = (add && !R->saturate && !R->immediate &&
                  R->encoding == AGX_APPLE9_ENC_INT_ADD_EXTENDED && slot < 2) ||
                 (mad && R->encoding == AGX_APPLE9_ENC_INT_MAD_EXTENDED &&
                  (!sext || slot == 0));
         all = slot >= 0 && ok && !(R->src_half_mask & BITFIELD_BIT(slot)) &&
               agx_apple9_instr_block(R) == agx_apple9_instr_block(D);
      }
      if (!all)
         continue;
      /* Rewriting moves each use off this chain; walk a snapshot. */
      unsigned count = 0;
      for (const struct agx_apple9_use *u = uses; u; u = u->next)
         ++count;
      struct agx_apple9_use *readers = malloc(count * sizeof(*readers));
      if (!readers)
         return false;
      count = 0;
      for (const struct agx_apple9_use *u = uses; u; u = u->next)
         readers[count++] = *u;
      for (unsigned r = 0; r < count; ++r) {
         struct agx_apple9_vir_instr *R = readers[r].instruction;
         unsigned source = readers[r].source;
         if (fp16 && R->op == AGX_APPLE9_VIR_FMA && register_slot(R, source) == 0) {
            agx_apple9_vir_set_source(p, R, 0, R->src[1]);
            R->src_abs_mask = (R->src_abs_mask & ~3u) |
               ((R->src_abs_mask & 1) << 1) | ((R->src_abs_mask & 2) >> 1);
            R->src_neg_mask = (R->src_neg_mask & ~3u) |
               ((R->src_neg_mask & 1) << 1) | ((R->src_neg_mask & 2) >> 1);
            source = 1;
         }
         const unsigned bit = BITFIELD_BIT(register_slot(R, source));
         agx_apple9_vir_set_source(p, R, source, x);
         R->src_half_mask |= bit;
         if (hi) R->src_hi_mask |= bit;
         if (sext) R->src_sext_mask |= bit;
      }
      free(readers);
   }
   return true;
}

bool
agx_apple9_optimize_vir_with_constants(
   struct agx_apple9_vir_program *p,
   const struct agx_apple9_uniform_constants *constants)
{
   assert(!p->physical && !p->dependencies_finalized);
   if (!p->value_count)
      return true;

   uint32_t *replacement = MALLOC(p->value_count * sizeof(*replacement));
   uint32_t *values = CALLOC(p->value_count, sizeof(*values));
   bool *constant = CALLOC(p->value_count, sizeof(*constant));
   uint16_t *uniform_read = CALLOC(p->value_count, sizeof(*uniform_read));
   struct expression *keys = CALLOC(p->value_count, sizeof(*keys));
   struct agx_apple9_vir_instr **definitions =
      CALLOC(p->value_count, sizeof(*definitions));
   struct hash_table *expressions =
      _mesa_hash_table_create(NULL, expression_hash, expression_equal);
   struct hash_table_u64 *constant_words = NULL;
   bool success = replacement && values && constant && uniform_read && keys && definitions && expressions;
   if (!success)
      goto cleanup;
   for (unsigned v = 0; v < p->value_count; ++v)
      replacement[v] = v;

   /* Main-program uniforms cannot change. Their SSA imports keep naming
    * the same word across control-flow and active-lane boundaries. Setup
    * programs that write the window retain the local tracking rules. */
   bool uniform_writes = false;
   for (unsigned i = 0; i < p->instruction_count; ++i)
      uniform_writes |= p->instructions[i]->op == AGX_APPLE9_VIR_STORE_UNIFORM;

   constant_words = _mesa_hash_table_u64_create(NULL);
   if (!constant_words) {
      success = false;
      goto cleanup;
   }
   if (constants && !uniform_writes) {
      assert(constants->base + constants->count <= AGX_APPLE9_UNIFORM_COUNT);
      for (unsigned i = 0; i < constants->count; ++i)
         _mesa_hash_table_u64_insert(constant_words, constants->values[i],
            (void *)(uintptr_t)(constants->base + i + 1));
   }
   struct agx_apple9_block *block = NULL;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (block != agx_apple9_instr_block(I)) {
         block = agx_apple9_instr_block(I);
         _mesa_hash_table_clear(expressions, NULL);
         if (uniform_writes)
            memset(uniform_read, 0, p->value_count * sizeof(*uniform_read));
      }
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t *sp = agx_apple9_vir_src_ptr(I, s);
         if (*sp < p->value_count)
            *sp = resolve(replacement, *sp);
      }
      switch (I->op) {
      case AGX_APPLE9_VIR_STORE_UNIFORM:
         /* Uniform reads are reusable only while their argument window is
          * unchanged. Setup may publish another value to the same word. */
         _mesa_hash_table_clear(expressions, NULL);
         memset(uniform_read, 0, p->value_count * sizeof(*uniform_read));
         break;
      case AGX_APPLE9_VIR_OBJECT_CALL:
      case AGX_APPLE9_VIR_OBJECT_RETURN:
      case AGX_APPLE9_VIR_OBJECT_RESUME:
      case AGX_APPLE9_VIR_ABI_RETURN:
      case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
      case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
      case AGX_APPLE9_VIR_EXEC_MASK_POP:
      case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
      case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
      case AGX_APPLE9_VIR_LOOP_MASK_POP:
      case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
         /* A synthetic mask transition may share a logical block. Values
          * produced under its old active lanes need not cover the new ones. */
         _mesa_hash_table_clear(expressions, NULL);
         if (uniform_writes)
            memset(uniform_read, 0, p->value_count * sizeof(*uniform_read));
         break;
      default:
         break;
      }
      const bool pure_alu = agx_apple9_instr_is_pure_alu(I);
      if (!pure_alu && !apple9_instr_is_cse_eligible(I)) {
         if (I->op == AGX_APPLE9_VIR_COLLECT) {
            for (unsigned c = 0; c < I->dest_components; ++c)
               definitions[I->dest + c] = I;
         }
         /* A predicate has no SSA destination and cannot be CSE'd as an ALU
          * value, but its explicit operands may still read uniforms. */
         if (I->op == AGX_APPLE9_VIR_PREDICATE_COMPARE) {
            inline_compare_literal(I, constant, values);
            inline_alu_sources(I, uniform_read, constant, values);
         }
         inline_texture_image_uniforms(I, definitions, uniform_read, constant,
                                       values, uniform_writes ? NULL : constants);
         drop_zero_load_index(I, constant, values);
         continue;
      }
      /* Fixed interface values are not interchangeable with ordinary SSA. */
      bool constrained = false;
      for (unsigned c = 0; c < I->dest_components; ++c)
         constrained |= p->fixed_phys[I->dest + c] != AGX_APPLE9_PHYS_INVALID ||
                        p->max_phys[I->dest + c] != AGX_APPLE9_PHYS_INVALID;
      if (constrained)
         continue;

      if (pure_alu) {
         fold_integer(I, constant, values);
         fold_iadd_shift(I, definitions);
         fold_select_xor(I, definitions, constant, values, uniform_writes);
         select_float_sources(I, definitions, constant, values);
         inline_integer_immediates(I, constant, values);
         inline_compare_literal(I, constant, values);
         inline_alu_sources(I, uniform_read, constant, values);
      }
      for (unsigned c = 0; c < I->dest_components; ++c)
         definitions[I->dest + c] = I;
      if (I->op == AGX_APPLE9_VIR_IMM) {
         constant[I->dest] = true;
         values[I->dest] = I->immediate;
         uniform_read[I->dest] = (uintptr_t)_mesa_hash_table_u64_search(
            constant_words, I->immediate);
      }
      if (I->op == AGX_APPLE9_VIR_IOR_UNIFORM &&
          I->immediate < AGX_APPLE9_UNIFORM_COUNT &&
          (I->nr_srcs == 0 ||
           (I->nr_srcs == 1 && constant[I->src[0]] && !values[I->src[0]])))
         uniform_read[I->dest] = I->immediate + 1;
      struct expression *key = &keys[I->dest];
      *key = (struct expression){
         .op = I->op, .encoding = I->encoding,
         .immediate = I->immediate, .nr_srcs = I->nr_srcs,
         .components = I->dest_components,
         .src_abs = I->src_abs_mask, .src_neg = I->src_neg_mask,
         .saturate = I->saturate,
         .select_float_immediates = I->select_float_immediates,
         .src_uniform = I->alu_src_uniform_mask,
         .src_immediate = I->alu_src_immediate_mask,
         .src_half = I->src_half_mask, .src_hi = I->src_hi_mask,
         .src_sext = I->src_sext_mask,
      };
      for (unsigned s = 0; s < ARRAY_SIZE(key->src_value); ++s) {
         if ((key->src_uniform | key->src_immediate) & BITFIELD_BIT(s))
            key->src_value[s] = I->alu_src_value[s];
      }
      memcpy(key->src, I->src, I->nr_srcs * sizeof(I->src[0]));
      struct hash_entry *entry = _mesa_hash_table_search(expressions, key);
      if (entry) {
         struct agx_apple9_vir_instr *previous = entry->data;
         for (unsigned c = 0; c < I->dest_components; ++c)
            replacement[I->dest + c] = previous->dest + c;
      } else if (!_mesa_hash_table_insert(expressions, key, I)) {
         success = false;
         goto cleanup;
      }
   }

   /* Backedge phi uses can precede their definition in layout order.
    * Rewrite the complete use set once after collecting all replacements.
    * Shared DCE removes the now-unused definitions before allocation. */
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      unsigned nr_srcs = agx_apple9_vir_nr_srcs(I);
      for (unsigned s = 0; s < nr_srcs; ++s) {
         uint32_t *sp = agx_apple9_vir_src_ptr(I, s);
         if (*sp < p->value_count)
            *sp = resolve(replacement, *sp);
      }
   }
   for (unsigned i = 0; i < p->live_out_count; ++i)
      p->live_out[i] = resolve(replacement, p->live_out[i]);
   if (p->output != AGX_APPLE9_VREG_INVALID)
      p->output = resolve(replacement, p->output);
   success = fold_saturate(p, replacement) &&
             (!uniform_writes || fold_uniform_stores(p)) &&
             fold_half_sources(p, constants);

cleanup:
   if (constant_words)
      _mesa_hash_table_u64_destroy(constant_words);
   free(uniform_read);
   agx_apple9_invalidate_uses(p);
   if (expressions)
      _mesa_hash_table_destroy(expressions, NULL);
   free(keys);
   free(definitions);
   free(constant);
   free(values);
   free(replacement);
   return success;
}


bool
agx_apple9_optimize_vir(struct agx_apple9_vir_program *p)
{
   return agx_apple9_optimize_vir_with_constants(p, NULL);
}
