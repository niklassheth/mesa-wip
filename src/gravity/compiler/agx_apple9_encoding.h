/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_ENCODING_H
#define AGX_APPLE9_ENCODING_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* EXP-M4-37: native mode 2 materializes an arbitrary 32-bit value into
 * r0..r63. Shared by ordinary compiler lowering and stage ABI setup.
 * Destination bits 0..3 occupy byte 0's high nibble; bits 4..5 occupy
 * byte 2 bits 6..7. Byte 2 bit 5 is a separate native modifier, not a
 * destination extension; this established scalar form leaves it clear. */
static inline bool
agx_apple9_encode_literal32(uint8_t out[8], unsigned dst, uint32_t value)
{
   /* Six destination bits. Setting bit 60 as a seventh hung the GPU, and
    * native Metal never materializes a long immediate above r63. */
   if (dst >= 64)
      return false;
   const uint8_t bytes[8] = {
      ((dst & 0xf) << 4) | 0x0c, 0x80 | (value & 0x7f),
      ((dst >> 4) << 6) | 0x02,  (value >> 24) & 0xfe,
      (value >> 6) & 0x1e,       (value >> 9) & 0x0c,
      (value >> 13) & 0xff,      (value >> 21) & 0x0f,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* T8132 entry-table loads: base is a full address in a GPR pair, offset is
 * in eight-byte units, and results remain pending on dependency slot 2.
 * The established widths are one or two consecutive 64-bit pointers. */
static inline bool
agx_apple9_encode_pointer_load(uint8_t out[14], unsigned dst,
                              unsigned base, unsigned offset, unsigned count)
{
   if ((count != 1 && count != 2) || dst >= 64 || dst + 2 * count > 64 ||
       base >= 63 || offset > 255)
      return false;
   const uint8_t bytes[14] = {
      0x67, 0, 0x54, dst << 1, base, 0, 0, 0,
      count == 2 ? 0x57 : 0x59, 0, offset, 0x40, 0x26, 0,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Signed, instruction-start-relative branch used by shader control flow and
 * persistent shader entries. The low bit is reserved for instruction alignment. */
static inline bool
agx_apple9_encode_branch(uint8_t out[10], bool any, int64_t displacement)
{
   if (!out || (displacement & 1) || displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47))
      return false;
   uint8_t bytes[10] = {0x0f, any ? 0x00 : 0x01, 0x54};
   for (unsigned byte = 0; byte < 6; ++byte)
      bytes[3 + byte] = (uint64_t)displacement >> (8 * byte);
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Convention 0 is the standard function call/return form (used by linked
 * prologs and epilogs in graphics stages). Convention 1 is the object
 * collective-call form. Displacements are signed, byte-addressed and relative
 * to the instruction start, just as for branches. */
static inline bool
agx_apple9_encode_call(uint8_t out[10], unsigned convention,
                      int64_t displacement)
{
   if (!out || convention > 1 || (displacement & 1) ||
       displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47))
      return false;
   uint8_t bytes[10] = {0x8f, (uint8_t)(convention << 4), 0x54};
   for (unsigned byte = 0; byte < 6; ++byte)
      bytes[3 + byte] = (uint64_t)displacement >> (8 * byte);
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

static inline bool
agx_apple9_encode_return(uint8_t out[4], unsigned convention)
{
   if (!out || convention > 1)
      return false;
   const uint8_t bytes[4] = {0x8f, (uint8_t)((convention << 4) | 2), 0x54, 0};
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Only the independently used one- and two-level forms are admitted. The
 * object call ABI uses depth 2; this is sufficient, not a measured minimum. */
static inline bool
agx_apple9_encode_exec_pop(uint8_t out[6], unsigned depth)
{
   if (!out || (depth != 1 && depth != 2))
      return false;
   const uint8_t bytes[6] = {0x0f, 6, 4, (uint8_t)depth, 0, 0};
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Unified 4-byte Apple9 32-bit MOV / ZERO (opcode 0x0b).
 * - is_zero=true  (op=0): writes 0 to r0..r63 or cr0..cr63 (group 0).
 * - is_zero=false (op=1):
 *     src_is_uniform=false: copies r0..r127 to r0..r63, u0..u63, or cr0..cr63
 *                           (group 0), with inline 3-bit wait slot (0..6).
 *     src_is_uniform=true : copies u0..u255 to r0..r63. */
static inline bool
agx_apple9_encode_mov4(uint8_t out[4], unsigned dst, unsigned src,
                       bool is_zero, bool dst_is_uniform_or_cr,
                       bool src_is_uniform, bool dst_cache,
                       bool src_cache, bool src_discard, uint8_t wait_slot)
{
   if (!out || dst >= 64 || wait_slot > 6)
      return false;
   if (is_zero) {
      if (src != 0 || src_is_uniform || src_cache || src_discard ||
          wait_slot != 0)
         return false;
   } else if (src_is_uniform) {
      if (src >= 256 || dst_is_uniform_or_cr || src_cache || src_discard ||
          wait_slot != 0)
         return false;
   } else {
      if (src >= 128 || (src_cache && src_discard))
         return false;
   }
   uint32_t word = 0x0bu;
   word |= (dst & 0xfu) << 4;
   if (!is_zero) {
      word |= (src & 0x3fu) << 9;
      if (!src_is_uniform)
         word |= (uint32_t)src_cache << 15;
      word |= 1u << 16;
      word |= (src_is_uniform ? ((src >> 6) & 1u) : (uint32_t)src_discard)
              << 19;
      word |= (src_is_uniform ? ((src >> 7) & 1u) : ((src >> 6) & 1u)) << 28;
   }
   word |= (uint32_t)dst_cache << 21;
   word |= ((dst >> 4) & 3u) << 22;
   word |= (uint32_t)dst_is_uniform_or_cr << 26;
   word |= (uint32_t)src_is_uniform << 27;
   word |= (uint32_t)wait_slot << 29;
   out[0] = (uint8_t)(word & 0xffu);
   out[1] = (uint8_t)((word >> 8) & 0xffu);
   out[2] = (uint8_t)((word >> 16) & 0xffu);
   out[3] = (uint8_t)((word >> 24) & 0xffu);
   return true;
}

/* 4-byte Apple9 32-bit bitwise AND (op=2), XOR (op=3), OR (op=4) on r0..r63.
 * Cannot carry scoreboard waits, uniforms, immediates, or r64..r95. */
static inline bool
agx_apple9_encode_bitop4(uint8_t out[4], unsigned dst, unsigned src0,
                         unsigned src1, unsigned op, bool dst_cache,
                         bool src0_cache, bool src0_discard,
                         bool src1_cache, bool src1_discard)
{
   if (!out || dst >= 64 || src0 >= 64 || src1 >= 64 ||
       (op != 2 && op != 3 && op != 4) ||
       (src0_cache && src0_discard) || (src1_cache && src1_discard))
      return false;
   uint32_t word = 0x0bu;
   word |= (dst & 0xfu) << 4;
   word |= 1u << 8;
   word |= (src0 & 0x3fu) << 9;
   word |= (uint32_t)src0_cache << 15;
   word |= (op & 7u) << 16;
   word |= (uint32_t)src0_discard << 19;
   word |= (uint32_t)src1_discard << 20;
   word |= (uint32_t)dst_cache << 21;
   word |= ((dst >> 4) & 3u) << 22;
   word |= 1u << 24;
   word |= (src1 & 0x3fu) << 25;
   word |= (uint32_t)src1_cache << 31;
   out[0] = (uint8_t)(word & 0xffu);
   out[1] = (uint8_t)((word >> 8) & 0xffu);
   out[2] = (uint8_t)((word >> 16) & 0xffu);
   out[3] = (uint8_t)((word >> 24) & 0xffu);
   return true;
}

/* 4-byte Apple9 in-place FFMA / HFMA on r0..r63 (no waits or modifiers).
 * - op=2: dst = src_a * src_b + dst  (accumulator in dst)
 * - op=3: dst = src_a * dst   + src_b (multiplicand in dst; FP32 bit 24 = 0) */
static inline bool
agx_apple9_encode_fma4(uint8_t out[4], unsigned dst, unsigned src_a,
                       unsigned src_b, unsigned op, bool is_half,
                       bool dst_hi, bool src_a_hi, bool src_b_hi,
                       bool dst_cache, bool src_a_cache, bool src_a_discard,
                       bool src_b_cache, bool src_b_discard)
{
   if (!out || dst >= 64 || src_a >= 64 || src_b >= 64 ||
       (op != 2 && op != 3) ||
       (!is_half && (dst_hi || src_a_hi || src_b_hi)) ||
       (src_a_cache && src_a_discard) || (src_b_cache && src_b_discard))
      return false;
   uint32_t word = is_half ? (dst_hi ? 0x08u : 0x00u) : 0x09u;
   word |= (dst & 0xfu) << 4;
   if (is_half)
      word |= ((src_a << 1) | (uint32_t)src_a_hi) << 8;
   else
      word |= (1u << 8) | ((src_a & 0x3fu) << 9);
   word |= (uint32_t)src_a_cache << 15;
   word |= (op & 7u) << 16;
   word |= (uint32_t)src_a_discard << 19;
   word |= (uint32_t)src_b_discard << 20;
   word |= (uint32_t)dst_cache << 21;
   word |= ((dst >> 4) & 3u) << 22;
   if (is_half)
      word |= ((src_b << 1) | (uint32_t)src_b_hi) << 24;
   else
      word |= ((op == 2 ? 1u : 0u) << 24) | ((src_b & 0x3fu) << 25);
   word |= (uint32_t)src_b_cache << 31;
   out[0] = (uint8_t)(word & 0xffu);
   out[1] = (uint8_t)((word >> 8) & 0xffu);
   out[2] = (uint8_t)((word >> 16) & 0xffu);
   out[3] = (uint8_t)((word >> 24) & 0xffu);
   return true;
}

/* Transfer one pending word to the main's argument window. A first transfer
 * waits for the table-load group on slot 2; subsequent transfers reuse it.
 * Argument word 2*i is the low half of pointer i, word 2*i+1 its high half. */
static inline bool
agx_apple9_encode_argument_word(uint8_t out[4], unsigned dst,
                               unsigned src, bool wait)
{
   if (dst >= 64 || src >= 64)
      return false;
   return agx_apple9_encode_mov4(out, dst, src, false, true, false, false,
                                 false, true, wait ? 2 : 0);
}

#endif
