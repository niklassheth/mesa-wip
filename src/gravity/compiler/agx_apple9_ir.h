/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_IR_H
#define AGX_APPLE9_IR_H

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "util/bitset.h"
#include "util/list.h"
#include "util/u_dynarray.h"

#include "agx_apple9_machine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_APPLE9_VREG_INVALID UINT32_MAX
#define AGX_APPLE9_PHYS_INVALID UINT8_MAX
#define AGX_APPLE9_MAX_VIR_SRCS 10
/*
 * A deliberately small semantic IR for the first real Apple9 compiler.
 *
 * Values are scalar 32-bit SSA virtual registers.  Consecutive values may be
 * grouped into an adjacent physical-register tuple by an explicit COLLECT
 * pseudo.  Encoding selection is explicit before allocation, so physical-
 * register constraints come from the Apple9 machine table instead of being
 * implicit in byte templates.
 */
enum agx_apple9_vir_opcode {
   AGX_APPLE9_VIR_IMM,
   AGX_APPLE9_VIR_GET_GLOBAL_ID,
   AGX_APPLE9_VIR_GET_SR,
   AGX_APPLE9_VIR_DEVICE_LOAD,
   AGX_APPLE9_VIR_PUBLICATION_TUPLE,
   AGX_APPLE9_VIR_TEXTURE_SAMPLE,
   AGX_APPLE9_VIR_IMAGE_STORE,
   AGX_APPLE9_VIR_BLOCK_IMAGE_STORE,
   AGX_APPLE9_VIR_BIT_COUNT,
   AGX_APPLE9_VIR_UFIND_MSB,
   AGX_APPLE9_VIR_BIT_REVERSE,
   AGX_APPLE9_VIR_U2F32,
   AGX_APPLE9_VIR_I2F32,
   AGX_APPLE9_VIR_F2I32,
   AGX_APPLE9_VIR_F2U32,
   /* Direct integer-to-FP16 and FP32-to-narrow-integer conversions. Results
    * occupy the low 16-bit half of a 32-bit SSA register. */
   AGX_APPLE9_VIR_U2F16,
   AGX_APPLE9_VIR_I2F16,
   AGX_APPLE9_VIR_U16_2F16,
   AGX_APPLE9_VIR_I16_2F16,
   AGX_APPLE9_VIR_F2I16,
   AGX_APPLE9_VIR_F2U16,
   AGX_APPLE9_VIR_F2I8,
   AGX_APPLE9_VIR_F2U8,
   /* Native binary16 conversions. Values still occupy 32-bit SSA registers. */
   AGX_APPLE9_VIR_F2F16,
   AGX_APPLE9_VIR_PACK_HALF_2X16,
   AGX_APPLE9_VIR_PACK_UNORM_2X16,
   AGX_APPLE9_VIR_PACK_UNORM_4X8,
   AGX_APPLE9_VIR_PACK_SNORM_2X16,
   AGX_APPLE9_VIR_PACK_SNORM_4X8,
   /* Three or four FP32 channels to one packed word; immediate is an
    * enum agx_apple9_pack_format. */
   AGX_APPLE9_VIR_PACK_FORMAT,
   /* immediate selects the low (0) or high (1) half of the source. */
   AGX_APPLE9_VIR_UNPACK_HALF,
   /* Two normalized components to FP32. Mode plus input half in immediate. */
   AGX_APPLE9_VIR_UNPACK_NORM,
   AGX_APPLE9_VIR_IADD,
   AGX_APPLE9_VIR_IMUL,
   AGX_APPLE9_VIR_ISUB,
   AGX_APPLE9_VIR_IMAD,
   /* Full 32x32 product, low word followed by high word. immediate is signed. */
   AGX_APPLE9_VIR_IMUL_WIDE,
   AGX_APPLE9_VIR_IAND,
   AGX_APPLE9_VIR_IANDN,
   AGX_APPLE9_VIR_IOR,
   AGX_APPLE9_VIR_IXOR,
   /* immediate selects a 32-bit uniform word; IOR has one GPR source. */
   AGX_APPLE9_VIR_IOR_UNIFORM,
   AGX_APPLE9_VIR_STORE_UNIFORM,
   AGX_APPLE9_VIR_ISHR,
   AGX_APPLE9_VIR_USHR,
   AGX_APPLE9_VIR_ISHL,
   /* Funnel shift right: ((uint64_t)src0 << 32 | src1) >> src2. */
   AGX_APPLE9_VIR_EXTR,
   AGX_APPLE9_VIR_IMIN,
   AGX_APPLE9_VIR_IMAX,
   AGX_APPLE9_VIR_UMIN,
   AGX_APPLE9_VIR_UMAX,
   /* Binary16 arithmetic in the low half of a word-allocated SSA value.
    * The high half of a narrow result is undefined until explicitly extended. */
   AGX_APPLE9_VIR_HADD,
   AGX_APPLE9_VIR_HSUB,
   AGX_APPLE9_VIR_HMUL,
   AGX_APPLE9_VIR_HFMA,
   AGX_APPLE9_VIR_HMIN,
   AGX_APPLE9_VIR_HMAX,
   AGX_APPLE9_VIR_HRCP,
   AGX_APPLE9_VIR_HRCP_F32,
   AGX_APPLE9_VIR_HRSQ,
   /* FP16 input, FP32 factor; paired with a mixed multiply for sqrt. */
   AGX_APPLE9_VIR_HSQRT_FACTOR,
   /* FP32 source 0 times FP16 source 1, rounded to FP16. */
   AGX_APPLE9_VIR_HMUL_MIXED,
   AGX_APPLE9_VIR_HEXP2,
   AGX_APPLE9_VIR_HLOG2,
   AGX_APPLE9_VIR_HFLOOR,
   AGX_APPLE9_VIR_HCEIL,
   AGX_APPLE9_VIR_HTRUNC,
   AGX_APPLE9_VIR_HROUND_EVEN,
   AGX_APPLE9_VIR_FADD,
   AGX_APPLE9_VIR_FSUB,
   AGX_APPLE9_VIR_FMUL,
   AGX_APPLE9_VIR_FMUL_PROJECT,
   AGX_APPLE9_VIR_FADD_IMM,
   AGX_APPLE9_VIR_FMUL_IMM,
   AGX_APPLE9_VIR_FSAT,
   AGX_APPLE9_VIR_FMIN,
   AGX_APPLE9_VIR_FMAX,
   AGX_APPLE9_VIR_FMA,
   AGX_APPLE9_VIR_CUBE,
   AGX_APPLE9_VIR_FRCP,
   AGX_APPLE9_VIR_FRSQ,
   /* rsqrt-like factor with 1 at signed zero and +infinity. */
   AGX_APPLE9_VIR_FSQRT_FACTOR,
   /* sin(pi*x/2)/x on [-1,1], with limit pi/2 at zero. */
   AGX_APPLE9_VIR_FSIN_FACTOR,
   AGX_APPLE9_VIR_FEXP2,
   AGX_APPLE9_VIR_FLOG2,
   AGX_APPLE9_VIR_FFLOOR,
   AGX_APPLE9_VIR_FCEIL,
   AGX_APPLE9_VIR_FTRUNC,
   AGX_APPLE9_VIR_FROUND_EVEN,
   AGX_APPLE9_VIR_DERIVATIVE,
   AGX_APPLE9_VIR_SELECT,
   AGX_APPLE9_VIR_COLLECT,
   /* Pre-physical SSA RA pseudos: SPLIT extracts scalar SSA lanes from a
    * contiguous tuple, MOV copies an SSA/register/spill value, and SWAP
    * exchanges two registers during parallel-copy lowering. */
   AGX_APPLE9_VIR_SPLIT,
   AGX_APPLE9_VIR_MOV,
   AGX_APPLE9_VIR_SWAP,
   AGX_APPLE9_VIR_PHI,
   AGX_APPLE9_VIR_PHI_SRC,
   AGX_APPLE9_VIR_PREDICATE_COMPARE,
   AGX_APPLE9_VIR_EXEC_MASK_PUSH,
   AGX_APPLE9_VIR_EXEC_MASK_ELSE,
   AGX_APPLE9_VIR_EXEC_MASK_POP,
   AGX_APPLE9_VIR_LOOP_MASK_PUSH,
   AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
   AGX_APPLE9_VIR_LOOP_MASK_POP,
   AGX_APPLE9_VIR_JMP_EXEC_ANY,
   AGX_APPLE9_VIR_JMP_EXEC_NONE,
   AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
   AGX_APPLE9_VIR_INTERPOLATION_POSITION,
   AGX_APPLE9_VIR_ITER,
   AGX_APPLE9_VIR_ITER_FLAT,
   AGX_APPLE9_VIR_VARY_STORE,
   AGX_APPLE9_VIR_COVERAGE,
   AGX_APPLE9_VIR_DEPTH_STORE,
   AGX_APPLE9_VIR_TILE_ACCESS,
   AGX_APPLE9_VIR_TILE_LOAD,
   AGX_APPLE9_VIR_TILE_STORE,
   AGX_APPLE9_VIR_TILE_FENCE,
   AGX_APPLE9_VIR_DEVICE_STORE,
   AGX_APPLE9_VIR_DEVICE_ATOMIC,
   AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT,
   AGX_APPLE9_VIR_SUBGROUP_SCAN_IADD,
   AGX_APPLE9_VIR_SUBGROUP_REDUCE_UMAX,
   AGX_APPLE9_VIR_SUBGROUP_BROADCAST,
   AGX_APPLE9_VIR_SUBGROUP_BALLOT,
   /* Generalized 1-instruction SIMD & Quad reductions and exclusive scans.
    * immediate encodes enum agx_apple9_subgroup_reduce_op | flags. */
   AGX_APPLE9_VIR_SUBGROUP_REDUCE,
   /* Generalized 10-byte SIMD & Quad shuffles (immediate encodes
    * enum agx_apple9_subgroup_shuffle_mode; src1 is GPR or 5-bit immediate). */
   AGX_APPLE9_VIR_SUBGROUP_SHUFFLE,
   /* 12-byte simd_shuffle_and_fill_down (C7 06): shifts src0 down by src2
    * (GPR or 5-bit immediate), filling upper lanes from src1. */
   AGX_APPLE9_VIR_SUBGROUP_ROTATE,
   /* 12-byte 8x8x8 SIMD-group matrix multiply-accumulate (CF 02 FP32 dst,
    * CF 00 FP16 dst). immediate packs agx_apple9_mma_flags(); src[] packs
    * A (1-2 GPRs), B (1-2 GPRs), and optional C (1-2 GPRs). */
   AGX_APPLE9_VIR_MMA,
   AGX_APPLE9_VIR_WORKGROUP_BARRIER,
   AGX_APPLE9_VIR_DEVICE_FENCE,
   AGX_APPLE9_VIR_HALT,
   AGX_APPLE9_VIR_SHARED_LOAD,
   AGX_APPLE9_VIR_SHARED_ATOMIC,
   AGX_APPLE9_VIR_SHARED_STORE,
   AGX_APPLE9_VIR_PRIVATE_LOAD,
   AGX_APPLE9_VIR_PRIVATE_STORE,
   AGX_APPLE9_VIR_SPILL_STORE,
   AGX_APPLE9_VIR_SPILL_LOAD,
   /* Standalone scoreboard wait; immediate is the slot mask. */
   AGX_APPLE9_VIR_WAIT,
   /* Side-effecting mesh-engine operations. CONTROL reads/clears src0's
    * selected primitive-count halfword; src1 is the provisional B input.
    * CONTROL preserves src1's selected B halfword. The A source and scalar
    * submit source conservatively clobber their allocated words, independently
    * of the scalar release=0 encoding.
    * SUBMIT_CHANNEL consumes the adjacent (selector, local offset) tuple.
    * They are never ordinary ALU, even if their explicit inputs are constant. */
   AGX_APPLE9_VIR_MESH_CONTROL,
   AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR,
   AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL,
   /* A closed, call-site-specialized leaf region. Until allocation, logical
    * execution order is CALL, callee instructions, RETURN, RESUME, continuation.
    * All callee defs therefore interfere with caller values live across it.
    * Layout moves the callee through RETURN out of line and assigns CALL's
    * branch_target. RESUME restores the converged 16-lane caller mask.
    * No hidden GPR clobber is assumed; that remains a hardware promotion gate.
    * Nested calls, escaping callee SSA and divergent callers are unsupported. */
   AGX_APPLE9_VIR_OBJECT_CALL,
   AGX_APPLE9_VIR_OBJECT_RETURN,
   AGX_APPLE9_VIR_OBJECT_RESUME,
   /* Lane 0/1/2 publishes X/Y/Z from its allocated publication word. */
   AGX_APPLE9_VIR_OBJECT_GRID_EXPORT,
   /* Linked prolog/epilog stage ABI: ENTRY_REG is a zero-byte pseudo defining
    * a value preloaded in physical GPR `immediate` on shader entry; ABI_RETURN
    * exports `live_out[]` into `live_out_reg[]` and returns (convention 0). */
   AGX_APPLE9_VIR_ENTRY_REG,
   AGX_APPLE9_VIR_ABI_RETURN,
};

/* Reserve r0..r7 below the linked stage ABI window so spilling (r0..r4) never
 * collides with preloaded entry registers or exported return registers. */
#define AGX_APPLE9_ABI_REG_BASE 8
#define AGX_APPLE9_MAX_ABI_REGS 64

/* Workgroup execution barrier forms (VIR_WORKGROUP_BARRIER immediate). */
enum agx_apple9_workgroup_barrier_mode {
   /* Workgroup execution + threadgroup memory barrier (0x0961). */
   AGX_APPLE9_WORKGROUP_BARRIER_SHARED = 0,
   /* Workgroup execution-only barrier, omitting TG-mem bit 0x20 (0x0941). */
   AGX_APPLE9_WORKGROUP_BARRIER_NONE = 1,
   /* Core-local workgroup execution + device memory barrier (0x0885). */
   AGX_APPLE9_WORKGROUP_BARRIER_DEVICE = 2,
};

/* The object-grid form is not an API workgroup barrier. */
enum agx_apple9_fence_mode {
   AGX_APPLE9_FENCE_DEVICE = 0,
   AGX_APPLE9_FENCE_RASTER_ORDER = 1,
   AGX_APPLE9_FENCE_OBJECT_GRID = 2,
   /* The device fence with acquire semantics: same encoding, but device
    * reads issued before it must have completed when it executes. */
   AGX_APPLE9_FENCE_DEVICE_ACQUIRE = 3,
   /* Wait on image-write scoreboard slots and drain PBE writes to memory. */
   AGX_APPLE9_FENCE_IMAGE_WRITE = 4,
   /* Invalidate the texture cache so subsequent texture loads observe memory. */
   AGX_APPLE9_FENCE_TEX_INVALIDATE = 5,
   /* Fragment-stage PBE drain (07 <mask>4 64 50 06 00). */
   AGX_APPLE9_FENCE_IMAGE_WRITE_FS = 6,
   /* Workgroup-scope texture barrier drain (0x0e51) and invalidate (0x0ed1). */
   AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP = 7,
   AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP = 8,
   /* Workgroup-scope texture fence without rendezvous (0x0e50 / 0x0ed0). */
   AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM = 9,
   AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP_MEM = 10,
   /* Zero-byte compiler scheduling/ordering barrier (subgroup/WG mem-only). */
   AGX_APPLE9_FENCE_COMPILER_BARRIER = 11,
};

enum agx_apple9_unpack_norm_mode {
   AGX_APPLE9_UNPACK_SNORM16 = 2,
   AGX_APPLE9_UNPACK_SNORM8 = 3,
   AGX_APPLE9_UNPACK_UNORM16 = 4,
   AGX_APPLE9_UNPACK_UNORM8 = 6,
};
#define AGX_APPLE9_UNPACK_HIGH_HALF (1u << 8)

/* Shared native five-bit operation selector used by the device and
 * threadgroup atomic families.  The current compiler emits the independently
 * hardware-validated per-lane device form. */
enum agx_apple9_atomic_op {
   AGX_APPLE9_ATOMIC_ADD = 0x10,
   AGX_APPLE9_ATOMIC_AND = 0x11,
   AGX_APPLE9_ATOMIC_CMPXCHG = 0x12,
   AGX_APPLE9_ATOMIC_FADD = 0x13,
   AGX_APPLE9_ATOMIC_SMAX = 0x14,
   AGX_APPLE9_ATOMIC_SMIN = 0x15,
   AGX_APPLE9_ATOMIC_OR = 0x16,
   AGX_APPLE9_ATOMIC_SUB = 0x1b,
   AGX_APPLE9_ATOMIC_UMAX = 0x1c,
   AGX_APPLE9_ATOMIC_UMIN = 0x1d,
   AGX_APPLE9_ATOMIC_XCHG = 0x1e,
   AGX_APPLE9_ATOMIC_XOR = 0x1f,
};

enum agx_apple9_predicate_condition {
   AGX_APPLE9_PREDICATE_FGT = 0x02,
   AGX_APPLE9_PREDICATE_FLT = 0x03,
   AGX_APPLE9_PREDICATE_UGT = 0x04,
   AGX_APPLE9_PREDICATE_ULT = 0x05,
   AGX_APPLE9_PREDICATE_IGT = 0x06,
   AGX_APPLE9_PREDICATE_ILT = 0x07,
};

enum agx_apple9_predicate_extended_condition {
   AGX_APPLE9_PREDICATE_EXT_FEQ = 0x00,
   AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE = 0x02,
   AGX_APPLE9_PREDICATE_EXT_FLE_SEQUENCE = 0x03,
   AGX_APPLE9_PREDICATE_EXT_IEQ = 0x07,
};

/* EXP-M4-53 separates the six-entry predicate scratch bank from the implicit
 * execution-mask stack.  PREDICATE_INVERT complements a comparison result;
 * EXEC_MASK_INVERT complements the selected predicate when a push consumes
 * it.  Ordinary nested ifs may therefore reuse bank zero at every depth. */
#define AGX_APPLE9_PREDICATE_BANK_COUNT 6u
#define AGX_APPLE9_PREDICATE_INVERT     (1u << 8)
#define AGX_APPLE9_PREDICATE_BANK_SHIFT 16
#define AGX_APPLE9_PREDICATE_BANK_MASK                                        \
   (0x07u << AGX_APPLE9_PREDICATE_BANK_SHIFT)
#define AGX_APPLE9_PREDICATE_BANK(index)                                      \
   ((uint32_t)(index) << AGX_APPLE9_PREDICATE_BANK_SHIFT)

#define AGX_APPLE9_EXEC_MASK_INVERT (1u << 8)
#define AGX_APPLE9_EXEC_MASK_SOURCE(source) (0x01u + 4u * (source))
#define AGX_APPLE9_EXEC_MASK_PREDICATE(bank)                                  \
   AGX_APPLE9_EXEC_MASK_SOURCE(bank)
#define AGX_APPLE9_EXEC_MASK_TRUE  AGX_APPLE9_EXEC_MASK_SOURCE(6)
#define AGX_APPLE9_EXEC_MASK_FALSE AGX_APPLE9_EXEC_MASK_SOURCE(7)

/* The tested LOOP_MASK_UPDATE form selects the same predicate bank with
 * 0x02,0x06,0x0a,... .  Bit 0x20 is consumer polarity: EXP-M4-53 proves that
 * flipping it together with PREDICATE_INVERT preserves exact loop output. */
#define AGX_APPLE9_LOOP_MASK_INVERT 0x20u
#define AGX_APPLE9_LOOP_MASK_PREDICATE(bank) (0x02u + 4u * (bank))

/* BREAK_MASK_UNWIND packs the number of scopes crossed and the consumed
 * predicate bank into independent bytes. */
#define AGX_APPLE9_BREAK_SCOPE_TAG(immediate)  (((immediate) >> 8) & 0xffu)
#define AGX_APPLE9_BREAK_PREDICATE_BANK(immediate) ((immediate) & 0xffu)
#define AGX_APPLE9_BREAK_IMMEDIATE(scope_tag, predicate_bank)                      \
   (((uint32_t)(scope_tag) << 8) | (uint32_t)(predicate_bank))

enum agx_apple9_select_condition {
   AGX_APPLE9_SELECT_FEQ = 0x00,
   AGX_APPLE9_SELECT_FGT = 0x02,
   AGX_APPLE9_SELECT_FLT = 0x03,
   AGX_APPLE9_SELECT_UGT = 0x04,
   AGX_APPLE9_SELECT_ULT = 0x05,
   AGX_APPLE9_SELECT_IGT = 0x06,
   AGX_APPLE9_SELECT_ILT = 0x07,
};

#define AGX_APPLE9_SELECT_EQUALITY (1u << 8)
/* A logical shift-right immediate is a bitfield extract: immediate bits 0-4
 * hold the offset, bits 8-13 the width (0 keeps every bit above it). */
#define AGX_APPLE9_SHIFT_WIDTH_SHIFT 8

/*
 * Apple9 asynchronous producers publish through a six-entry scoreboard.
 * Slot 0 is the ordinary/materialized GPR path; slots 1--6 are transient
 * completion groups for outstanding asynchronous work.  AUTO is compiler IR only and must be resolved
 * after final instruction scheduling, before packing.
 */
/* Formats of AGX_APPLE9_VIR_PACK_FORMAT (Metal's encodings of our own render
 * pipelines: one instruction per packed colour). */
enum agx_apple9_pack_format {
   AGX_APPLE9_PACK_R11G11B10_FLOAT,
   AGX_APPLE9_PACK_R10G10B10A2_UNORM,
};

enum agx_apple9_scoreboard_slot {
   AGX_APPLE9_SCOREBOARD_SLOT_NONE = 0,
   AGX_APPLE9_SCOREBOARD_SLOT_1 = 1,
   AGX_APPLE9_SCOREBOARD_SLOT_2 = 2,
   AGX_APPLE9_SCOREBOARD_SLOT_3 = 3,
   AGX_APPLE9_SCOREBOARD_SLOT_4 = 4,
   AGX_APPLE9_SCOREBOARD_SLOT_5 = 5,
   AGX_APPLE9_SCOREBOARD_SLOT_6 = 6,
   AGX_APPLE9_SCOREBOARD_SLOT_AUTO = 0xff,
};

enum agx_apple9_device_load_flags {
   /* Byte 2 bit 4 marks that another load follows in linear issue order.
    * Native Metal leaves it set across execution-mask transitions. */
   AGX_APPLE9_DEVICE_LOAD_HAS_NEXT = 1u << 1,
};

enum agx_apple9_device_load_index_kind {
   /* Byte 5's low seven bits name a real architectural GPR.  Bit 7 is clear
    * when that SSA value has a later consumer and set when this load is its
    * final consumer.  VIR packing derives the bit from allocator liveness;
    * the raw packer accepts the explicit form for byte-level tests. */
   AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
   AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
   /* Pointer-pair loads without an index: byte 5 is zero and bit 53, set
    * by every indexed form, is clear (Metal's p[k] and struct-member loads). */
   AGX_APPLE9_DEVICE_LOAD_INDEX_NONE,
};

/* Human-order names for the two raw bytes at instruction offsets 8 and 9.
 * For example, 0x5101 is serialized as 51:01, independent of host endian. */
enum agx_apple9_device_load_raw_token {
   AGX_APPLE9_DEVICE_LOAD_TOKEN_1100 = 0x1100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_5100 = 0x5100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_9100 = 0x9100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_D100 = 0xd100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_1101 = 0x1101,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_5101 = 0x5101,
};

struct agx_apple9_device_load_contract {
   enum agx_apple9_device_load_index_kind index_kind;
   uint8_t flags;
   uint16_t raw_token;
};

struct agx_apple9_block;

enum agx_apple9_local_index_kind {
   AGX_APPLE9_LOCAL_INDEX_NONE,
   AGX_APPLE9_LOCAL_INDEX_GPR,
};

struct agx_apple9_vir_instr {
   enum agx_apple9_vir_opcode op;
   enum agx_apple9_encoding encoding;
   uint32_t dest;
   /* Number of adjacent SSA values and physical GPRs defined at dest.
    * Scalar instructions define one value. Native vector memory loads and
    * COLLECT define one two-, three-, or four-register tuple. A native load
    * uses one scoreboard allocation for the complete tuple. */
   uint8_t dest_components;
   /* Scalar device-memory element width.  Zero is the established u32
    * default; compiler-authored narrow operations use 8 or 16 explicitly. */
   uint8_t memory_bits;
   /* Local-output store index source, independent of data width. */
   enum agx_apple9_local_index_kind local_index_kind;
   /* Number of scalar lanes read or written by a memory instruction.  Loads
    * also define an adjacent dest_components tuple; stores have no SSA
    * destination and carry their ordered data values in src[0..N-1]. */
   uint8_t memory_components;
   /* Address = base64 + (zero_extend(index32) << shift) + signed byte offset.
    * Shift 0..4 is independent of element width and component count. */
   uint8_t memory_index_shift;
   int16_t memory_offset;
   uint8_t tile_sample_mask;
   enum agx_apple9_atomic_op atomic_op;
   bool atomic_discard;
   /* Preserve required memory reads even when their SSA results are dead. */
   bool memory_volatile;
   /* Device-coherent / L1-bypass access (memory_coherence_device). */
   bool memory_coherent;
   /* The access uses a global address rather than a binding, through a GPR
    * pair or a uniform pair, and may alias any other global access. */
   bool memory_global;
   uint8_t texture_index, sampler_index;
   uint32_t src[AGX_APPLE9_MAX_VIR_SRCS];
   uint8_t texture_dimension;
   bool texture_shadow;
   /* Gradient samples clamp the selected LOD from below by the signed Q6
    * value in bits 16..27 of coordinate word 3; biased 2D samples by the
    * one in its low 12 bits. Biased array samples carry theirs in word 2
    * without this flag. */
   bool texture_min_lod;
   bool texture_fetch;
   bool texture_lod_query;
   bool texture_multisampled;
   /* Gather one channel (texture_gather_component) of the 2x2 footprint at
    * the view's base level, returned as a dense RGBA tuple. */
   bool texture_gather;
   /* Hardware sparse residency return: appends the residency code word after
    * the packed color channels (byte 10 bit 1). */
   bool texture_sparse;
   /* Immediate texel offset: signed 4-bit x | y << 4 | z << 8. */
   bool texture_has_offset;
   uint16_t texture_offset;
   /* Register-supplied table indices, appended as the last sources in this
    * order, replace the immediate texture and sampler indices. */
   bool texture_bindless;
   bool sampler_bindless;
   /* Uniform-supplied bindless handles (u0..u127 for TEXTURE_SAMPLE; even
    * u0..u126 pair with zero upper word for IMAGE_STORE) stored in
    * texture_index / sampler_index without consuming GPR sources. */
   bool texture_uniform_bindless;
   bool sampler_uniform_bindless;
   /* Inline UQ4.2 immediate explicit LOD (mode 0x0009, 2D only, byte[11] =
    * texture_imm_lod_uq4_2 in 0..63; restricted to 0 in 20B bindless mode). */
   bool texture_imm_lod;
   uint8_t texture_imm_lod_uq4_2;
   /* Set when emitted from a fragment shader; GPR bindless PBE writes share
    * the same byte 13/14 encoding as compute/vertex stages. */
   bool image_fragment;
   uint8_t texture_gather_component;
   /* Logical RGBA channels returned in ascending channel order, packed at
    * dest. Zero retains the default RGBA selection for VIR callers. */
   uint8_t texture_result_mask;
   /* An incoming phi use names its successor's SSA definition. This is
    * edge metadata, not a mutable register assignment before allocation. */
   uint32_t target;
   uint32_t immediate;
   /* STORE_UNIFORM may compute its value directly into the uniform file.
    * immediate is the uniform word; these describe the fused integer ALU. */
   enum agx_apple9_vir_opcode uniform_op;
   bool uniform_signed;
   /* Stable destination identity; byte displacement is computed at packing. */
   struct agx_apple9_block *branch_target;
   /* A mask transition within a straight-line leaf region, rather than a
    * NIR CFG edge. It remains an ordered effect inside the allocation block. */
   bool local_control;
   /* Physical spill/fill traffic lowered from RA_MEM (covers both compact
    * SPILL_LOAD/SPILL_STORE and indexed PRIVATE_LOAD/PRIVATE_STORE >= 1024/2048
    * words, where `immediate` is the exact scratch word). */
   bool spill_access;
   /* Phi declaration and incoming edge uses remain SSA until packing. */
   struct agx_apple9_block *phi_block;
   struct agx_apple9_vir_instr *phi_edge;
   /* Sideband source array for canonical variable-arity VIR_PHI with more than
    * AGX_APPLE9_MAX_VIR_SRCS predecessors. When non-NULL, holds `phi_src_count`
    * SSA values indexed by logical predecessor index; otherwise inline
    * `src[0..nr_srcs-1]` is used. Access via agx_apple9_vir_nr_srcs() /
    * agx_apple9_vir_src() / agx_apple9_vir_src_ptr(). */
   uint32_t *phi_srcs;
   uint32_t phi_src_count;
   uint8_t nr_srcs;

   /* Floating source modifiers apply absolute value before negation. */
   uint8_t src_abs_mask, src_neg_mask;
   /* Validated ALU operand slots can name uniform words; floating operands
    * may also name exact constants.
    * src[] contains only their GPR inputs, in operand order. Liveness and SSA
    * use lists refer to that compact GPR list; these masks and modifiers refer
    * to the hardware operand slots. Uniform and immediate masks are disjoint. */
   uint8_t alu_src_uniform_mask, alu_src_immediate_mask;
   uint32_t alu_src_value[4];
   /* Register operand slots read as 16-bit halves of their 32-bit source
    * register: the high half when set in src_hi_mask, sign- rather than
    * zero-extended when set in src_sext_mask. */
   uint8_t src_half_mask, src_hi_mask, src_sext_mask;
   /* Explicit fields of the bounded mesh register forms. control42's
    * semantics and SUBMIT's release policy are not configuration formulas.
    * A releasing submission clobbers its source tuple; RA must preserve
    * still-live values, independently of this encoding flag. */
   bool mesh_control42, mesh_release;
   /* Clamp the floating result to [0, 1], mapping NaN to positive zero. */
   bool saturate;

   /* Scoreboard slot published by an asynchronous producer, including scalar
    * ALU exports to publication storage. Ordinary GPR ALU instructions leave
    * this at NONE. AUTO is resolved by the scheduled
    * scoreboard pass and is never serialized. */
   uint8_t producer_scoreboard_slot;
   /* 6-bit scoreboard slot drain mask for FENCE_IMAGE_WRITE (bits 12..17);
    * zero defaults to slot 1 (0x01). */
   uint8_t fence_slot_mask;
   bool producer_scoreboard_assigned, scoreboard_assigned;

   /* A scratch store has no register result. Its completion is consumed by
    * a scheduled instruction in the same block, independently of SSA uses. */
   struct agx_apple9_vir_instr *completion_consumer;

   /* Device-load sequence and index lifetime, independent of completion
    * tags. */
   uint8_t device_load_flags;
   enum agx_apple9_device_load_index_kind device_load_index_kind;

   /* Raw scalar-load instruction bytes 8:9.  These encode the producer's
    * scoreboard slot. */
   uint16_t device_load_raw_token;

   /* Bit s is set when src[s] has another consumer after this instruction. */
   uint16_t live_after_mask;
   /* Bit s is set when src[s] is killed (last use) at this instruction.
    * Populated by native VIR liveness for pre-physical SSA RA. */
   uint16_t kill_mask;
   /* Sources whose registers other live values share (a reused tuple's
    * components); allocation never releases them at this use. */
   uint16_t retain_mask;
   /* Bit c is set when component c of a multi-dest pseudo (VIR_SPLIT) is dead. */
   uint8_t dead_dest_mask;

   /*
    * The one pending-result slot consumed by this instruction, or NONE.
    * Metal has not been observed making one instruction consume multiple
    * slots. The selected encoding translates this logical slot into either
    * a binary index or a one-hot physical field. This is independent of
    * per-source release bits.
    */
   uint8_t scoreboard_slot;

   /* Conservative identity bridge inserted when an asynchronous load's first
    * user has no proven slot-bearing form.  It consumes the assigned slot
    * immediately and leaves an ordinary GPR value for the original users. */
   bool scoreboard_materialize;
   /* Preserve a result handoff that releases scarce texture publications. */
   bool publication_handoff;

   /*
    * Fragment iterator/perspective producers are addressed by compact ALU
    * through operand tokens, not by their architectural GPR numbers.  A value
    * of 0xff means that this source uses the ordinary GPR encoding.  These
    * per-source tokens are distinct from the instruction-wide scoreboard
    * slot and are kept on the use until the graphics compiler is generalized.
    */
};

static inline unsigned
agx_apple9_vir_nr_srcs(const struct agx_apple9_vir_instr *I)
{
   return I->phi_srcs ? I->phi_src_count : I->nr_srcs;
}

static inline uint32_t
agx_apple9_vir_src(const struct agx_apple9_vir_instr *I, unsigned s)
{
   assert(s < agx_apple9_vir_nr_srcs(I));
   return I->phi_srcs ? I->phi_srcs[s] : I->src[s];
}

static inline uint32_t *
agx_apple9_vir_src_ptr(struct agx_apple9_vir_instr *I, unsigned s)
{
   assert(s < agx_apple9_vir_nr_srcs(I));
   return I->phi_srcs ? &I->phi_srcs[s] : &I->src[s];
}

static inline bool
agx_apple9_vir_src_killed(const struct agx_apple9_vir_instr *I, unsigned s)
{
   return s < AGX_APPLE9_MAX_VIR_SRCS && ((I->kill_mask >> s) & 1u);
}

static inline void
agx_apple9_vir_set_src_killed(struct agx_apple9_vir_instr *I, unsigned s,
                              bool killed)
{
   assert(s < AGX_APPLE9_MAX_VIR_SRCS);
   if (killed)
      I->kill_mask |= (uint16_t)(1u << s);
   else
      I->kill_mask &= (uint16_t)~(1u << s);
}

static inline unsigned
agx_apple9_inline_source_file(const struct agx_apple9_vir_instr *I,
                              enum agx_apple9_operand_role role)
{
   unsigned s = role - AGX_APPLE9_OPERAND_SRC0;
   if (s >= 4)
      return 0;
   return ((I->alu_src_uniform_mask >> s) & 1 ? AGX_APPLE9_FILE_UNIFORM : 0) |
          ((I->alu_src_immediate_mask >> s) & 1 ? AGX_APPLE9_FILE_IMMEDIATE : 0);
}

/* Number of 32-bit GPRs occupied by a memory load/store's data tuple.
 * Packed 16-bit vectors pack two lanes per 32-bit GPR (i16x2 -> 1, i16x4 -> 2). */
static inline unsigned
agx_apple9_mem_data_gprs(const struct agx_apple9_vir_instr *I)
{
   unsigned components = I->memory_components ? I->memory_components : 1;
   if (I->memory_bits == 16 && (components == 2 || components == 4))
      return components / 2;
   return components;
}

/* Instruction storage and block identity survive every insertion/removal.
 * Layout links are independent of ownership, allowing forward branch targets. */
struct agx_apple9_block {
   struct agx_apple9_block *next, *prev, *allocated_next;
   struct agx_apple9_vir_program *program;
   struct nir_block *nir; /* Logical CFG, distinct from execution-mask layout. */
   /* A closed NIR call region may contain structured masks selected into this
    * same logical block. Its last NIR block owns the outgoing caller edges. */
   struct nir_block *nir_exit;
   /* Logical SSA CFG built by agx_apple9_vir_build_logical_cfg(). Synthetic
    * loop-mask restoration blocks (!nir && block != program->blocks) have
    * cfg_active == false and carry no logical SSA edges. */
   struct agx_apple9_block *cfg_succs[2];
   struct agx_apple9_block **cfg_preds;
   unsigned num_cfg_preds;
   unsigned cfg_index;
   bool cfg_active;
   bool loop_header;
   struct list_head instructions;
   bool placed;
   unsigned index;
   BITSET_WORD *live_in, *live_out;
   unsigned live_words; /* Allocated size of live_in and live_out. */
   unsigned start_index; /* Derived analysis position, never a branch identity. */
   unsigned offset;
};

static inline unsigned
agx_apple9_block_pred_index(const struct agx_apple9_block *block,
                            const struct agx_apple9_block *pred)
{
   for (unsigned i = 0; i < block->num_cfg_preds; ++i) {
      if (block->cfg_preds[i] == pred)
         return i;
   }
   return UINT_MAX;
}

static inline bool
agx_apple9_block_has_pred(const struct agx_apple9_block *block,
                          const struct agx_apple9_block *pred)
{
   return agx_apple9_block_pred_index(block, pred) != UINT_MAX;
}

/* Uses are ordered by machine layout. agx_apple9_vir_set_source keeps them
 * current; other operand edits must invalidate. Pointers remain valid only
 * until the next analysis rebuild. */
struct agx_apple9_use {
   struct agx_apple9_vir_instr *instruction;
   unsigned source;
   struct agx_apple9_use *next;
};

struct agx_apple9_vir_program {
   /* Indexed analysis view of stable, block-owned instructions. */
   struct agx_apple9_vir_instr **instructions;
   struct agx_apple9_block *blocks, *last_block, *current_block;
   struct agx_apple9_block *allocated_blocks;
   struct agx_apple9_use_analysis *use_analysis;
   bool dependencies_finalized;
   unsigned instruction_count;
   unsigned instruction_capacity;
   unsigned num_cfg_blocks;
   unsigned value_count;
   uint32_t output;
   /* Fragment execution can update depth/stencil without explicit stores. */
   bool fragment_shader;
   bool physical;

   /*
    * Values may enter or leave the bounded program in fixed physical GPRs.
    * This is used by fragment interpolation/color packing, whose surrounding
    * stage instructions have an independently validated register contract.
    */
   uint8_t *fixed_phys;
   /* Optional per-value upper bound used to propagate a constrained
    * consumer's compact-register requirement back to its producer. */
   uint8_t *max_phys;
   uint32_t *live_out;
   uint8_t *live_out_reg;
   unsigned live_out_count;
   unsigned live_out_capacity;
   bool reserved_gprs[AGX_APPLE9_GPR_COUNT];

   /* Physical GPR or publication index assigned to each SSA value. */
   uint8_t *phys;
   /* Export publications have their own index namespace. Slots remain distinct
    * through completion; ordinary GPRs with the same index are independent. */
   bool *publication;
   unsigned publication_count;
   /* Allocated entries of phys and publication. Values created after their
    * allocation have no entry. */
   unsigned phys_capacity, publication_capacity;
   unsigned peak_live_gprs;
   unsigned max_phys_gpr;
   unsigned private_size; /* NIR private arrays, placed after register spills. */
   unsigned scratch_size; /* Total bytes per invocation, rounded to 16. */
   unsigned spill_slots;
};

void agx_apple9_invalidate_uses(struct agx_apple9_vir_program *program);
bool agx_apple9_analyze_uses(struct agx_apple9_vir_program *program);
const struct agx_apple9_vir_instr *agx_apple9_definition(
   const struct agx_apple9_vir_program *program, uint32_t value);
const struct agx_apple9_use *agx_apple9_uses(
   const struct agx_apple9_vir_program *program, uint32_t value);
void agx_apple9_vir_set_source(struct agx_apple9_vir_program *program,
                               struct agx_apple9_vir_instr *instr,
                               unsigned source, uint32_t value);

void agx_apple9_vir_init(struct agx_apple9_vir_program *program);
void agx_apple9_vir_finish(struct agx_apple9_vir_program *program);
/* Deep copy. Logical NIR block pointers are copied as they are. */
bool agx_apple9_vir_clone(struct agx_apple9_vir_program *dst,
                          const struct agx_apple9_vir_program *src);
/* Append a serialization of everything an allocation reads from the program:
 * equal keys mean equal programs, with pointers expressed as layout indices
 * and NIR blocks as their relation to the program's blocks. Padding is
 * included, so equal programs may still differ in key. */
bool agx_apple9_vir_key(const struct agx_apple9_vir_program *program,
                        struct util_dynarray *key);

struct agx_apple9_block *agx_apple9_block_create(struct agx_apple9_vir_program *program);
void agx_apple9_block_begin(struct agx_apple9_vir_program *program,
                          struct agx_apple9_block *block);
/* Build logical SSA CFG (cfg_succs, cfg_preds, num_cfg_preds, cfg_index,
 * cfg_active, loop_header) from NIR block links, pruning unreachable
 * post-infinite-loop edges while keeping physical next/prev layout intact. */
bool agx_apple9_vir_build_logical_cfg(struct agx_apple9_vir_program *program);
/* Layout position, as of the last reindex. */
unsigned agx_apple9_instr_position(const struct agx_apple9_vir_instr *instr);
struct agx_apple9_block *agx_apple9_instr_block(const struct agx_apple9_vir_instr *instr);
struct agx_apple9_vir_instr *
agx_apple9_block_first_instr(const struct agx_apple9_block *block);
struct agx_apple9_vir_instr *
agx_apple9_block_last_instr(const struct agx_apple9_block *block);
struct agx_apple9_vir_instr *
agx_apple9_instr_next(const struct agx_apple9_vir_instr *instr);
struct agx_apple9_vir_instr *
agx_apple9_instr_prev(const struct agx_apple9_vir_instr *instr);
void agx_apple9_vir_move_before(struct agx_apple9_vir_program *program,
                              struct agx_apple9_vir_instr *instr,
                              struct agx_apple9_vir_instr *before);
void agx_apple9_vir_move_after(struct agx_apple9_vir_program *program,
                             struct agx_apple9_vir_instr *instr,
                             struct agx_apple9_vir_instr *after);
void agx_apple9_vir_move_to_block_start(struct agx_apple9_vir_program *program,
                                        struct agx_apple9_vir_instr *instr,
                                        struct agx_apple9_block *block);
void agx_apple9_vir_move_to_block_end(struct agx_apple9_vir_program *program,
                                      struct agx_apple9_vir_instr *instr,
                                      struct agx_apple9_block *block);
void agx_apple9_vir_remove(struct agx_apple9_vir_program *program,
                          struct agx_apple9_vir_instr *instruction);
void agx_apple9_vir_remove_many(struct agx_apple9_vir_program *program,
                                struct agx_apple9_vir_instr *const *instrs,
                                unsigned count);
void agx_apple9_vir_reindex(struct agx_apple9_vir_program *program);
bool agx_apple9_instr_is_pure_alu(const struct agx_apple9_vir_instr *instruction);
bool agx_apple9_supports_special_uniform(enum agx_apple9_vir_opcode op);

bool agx_apple9_supports_saturate(const struct agx_apple9_vir_instr *instruction);
/* Known literal values stored in a contiguous, immutable main-program
 * uniform window. Preserve their constant semantics until operand selection. */
struct agx_apple9_uniform_constants {
   unsigned base, count;
   uint32_t values[AGX_APPLE9_UNIFORM_COUNT];
};
bool agx_apple9_optimize_vir_with_constants(
   struct agx_apple9_vir_program *program,
   const struct agx_apple9_uniform_constants *constants);
bool agx_apple9_optimize_vir(struct agx_apple9_vir_program *program);
bool agx_apple9_encode_float_immediate(uint32_t value, uint8_t *encoded);
bool agx_apple9_encode_float_immediate_wide(uint32_t value, uint8_t *encoded,
                                            bool *low);


/* Integer add/multiply-add, logic and select source descriptors hold an
 * 8-bit zero-extended literal; the float forms hold a packed float. */
static inline bool
agx_apple9_integer_immediate_encoding(enum agx_apple9_encoding encoding)
{
   return encoding == AGX_APPLE9_ENC_INT_ADD_EXTENDED ||
          encoding == AGX_APPLE9_ENC_INT_MAD_EXTENDED ||
          encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED ||
          encoding == AGX_APPLE9_ENC_SELECT_GPR_WIDE;
}


/* Floating min/max operand B and the FMA addend have a fourth mantissa
 * bit beside the minifloat byte; FMA operand B does not. */
static inline bool
agx_apple9_wide_float_immediate_slot(enum agx_apple9_encoding encoding,
                                     unsigned slot)
{
   return (encoding == AGX_APPLE9_ENC_MINMAX_COMPACT && slot == 1) ||
          (slot == 2 && (encoding == AGX_APPLE9_ENC_FLOAT3_EXTENDED ||
                         encoding == AGX_APPLE9_ENC_FLOAT3_COMPACT ||
                         encoding == AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED ||
                         encoding == AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED));
}

/* AGX_APPLE9_NO_COMPARE_LITERAL=1 keeps compare operands in registers and
 * uniform words. */
bool agx_apple9_compare_literals_disabled(void);

/* Compare operand B of a select or predicate takes a literal in the byte
 * that otherwise names a uniform: a minifloat with its sign beside it for
 * floating conditions, an 8-bit integer otherwise. */
static inline bool
agx_apple9_compare_literal_slot(const struct agx_apple9_vir_instr *I,
                                unsigned slot)
{
   return slot == 1 &&
          ((I->op == AGX_APPLE9_VIR_SELECT &&
            I->encoding == AGX_APPLE9_ENC_SELECT_GPR_WIDE) ||
           (I->op == AGX_APPLE9_VIR_PREDICATE_COMPARE &&
            (I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT ||
             I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED)));
}

static inline bool
agx_apple9_compare_is_float(const struct agx_apple9_vir_instr *I)
{
   const unsigned condition = I->immediate & 0xff;
   if (I->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED)
      return condition != AGX_APPLE9_PREDICATE_EXT_IEQ;
   return condition == AGX_APPLE9_SELECT_FEQ ||
          condition == AGX_APPLE9_SELECT_FGT ||
          condition == AGX_APPLE9_SELECT_FLT;
}

static inline bool
agx_apple9_compare_literal_encodable(const struct agx_apple9_vir_instr *I,
                                     uint32_t value)
{
   uint8_t encoded;
   bool low;
   return agx_apple9_compare_is_float(I)
             ? agx_apple9_encode_float_immediate_wide(value, &encoded, &low)
             : value <= UINT8_MAX;
}

static inline bool
agx_apple9_source_immediate_encodable(enum agx_apple9_encoding encoding,
                                      unsigned slot, uint32_t value)
{
   uint8_t encoded;
   bool low;
   if (agx_apple9_integer_immediate_encoding(encoding))
      return value <= UINT8_MAX;
   if (agx_apple9_wide_float_immediate_slot(encoding, slot))
      return agx_apple9_encode_float_immediate_wide(value, &encoded, &low);
   return agx_apple9_encode_float_immediate(value, &encoded);
}

uint32_t agx_apple9_vir_emit(struct agx_apple9_vir_program *program,
                             enum agx_apple9_vir_opcode op,
                             enum agx_apple9_encoding encoding,
                             const uint32_t *src, unsigned nr_srcs,
                             uint32_t immediate);

bool agx_apple9_vir_emit_side_effect(struct agx_apple9_vir_program *program,
                                     enum agx_apple9_vir_opcode op,
                                     enum agx_apple9_encoding encoding,
                                     const uint32_t *src, unsigned nr_srcs,
                                     uint32_t immediate);

bool agx_apple9_vir_emit_mesh_control(struct agx_apple9_vir_program *program,
                                      uint32_t primitives, uint32_t auxiliary,
                                      bool control42);
bool agx_apple9_vir_emit_mesh_submit(struct agx_apple9_vir_program *program,
                                     const uint32_t src[2], bool channel,
                                     bool release);

/* After RA/scheduling, before packing. The main must end in HALT. No
 * instructions are introduced. Returns true also when there are no calls. */
bool agx_apple9_layout_object_calls(struct agx_apple9_vir_program *program,
                                   const char **reason);

bool agx_apple9_vir_emit_branch(struct agx_apple9_vir_program *program,
                                enum agx_apple9_vir_opcode op,
                                enum agx_apple9_encoding encoding,
                                struct agx_apple9_block *target);

uint32_t agx_apple9_vir_input(struct agx_apple9_vir_program *program,
                              unsigned phys);
bool agx_apple9_vir_set_load_address(struct agx_apple9_vir_program *program,
                                     uint32_t value, uint32_t address);
uint32_t agx_apple9_vir_emit_coefficients(struct agx_apple9_vir_program *program,
                                         unsigned coefficient);
uint32_t agx_apple9_vir_emit_iter_flat(struct agx_apple9_vir_program *program,
                                       unsigned coefficient);

uint32_t agx_apple9_vir_emit_device_load(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const struct agx_apple9_device_load_contract *contract);
/* mode 0 returns magnitude and a face in the low 16 bits of the next GPR.
 * Modes 1/2 return the signed half-coordinate for U/V, respectively. */
uint32_t agx_apple9_vir_emit_cube(struct agx_apple9_vir_program *program,
                                 const uint32_t src[3], unsigned mode);
uint32_t agx_apple9_vir_emit_unpack_norm(struct agx_apple9_vir_program *program,
                                        uint32_t src, unsigned mode);
uint32_t agx_apple9_vir_emit_mul_wide(struct agx_apple9_vir_program *program,
                                     const uint32_t src[2], bool is_signed);

uint32_t agx_apple9_vir_emit_device_load_vector(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   unsigned components, const struct agx_apple9_device_load_contract *contract);
/* Native 16-bit vector device load (i16x2 -> 1 packed GPR, i16x4 -> 2 packed
 * adjacent GPRs). Index scales by 4B (i16x2) or 8B (i16x4). */
uint32_t agx_apple9_vir_emit_device_load_vector_u16(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   unsigned components, const struct agx_apple9_device_load_contract *contract);
bool agx_apple9_vir_emit_device_store_vector_u16(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const uint32_t *data, unsigned components);
/* Publish two words to an even-aligned publication pair. Publications are
 * normally lane-major. A split pair instead publishes one word (src[0] ==
 * src[1]), with lanes 0-15 in the first publication and 16-31 in the second. */
uint32_t
agx_apple9_vir_emit_publication_pair(struct agx_apple9_vir_program *program,
                                     const uint32_t src[2], bool split);

/* Formatless 32-bit write through an ordinary bindless PBE table index, as
 * float or integer data. Pass layer != AGX_APPLE9_VREG_INVALID for a 2D-array
 * PBE descriptor. */
bool agx_apple9_vir_emit_image_store(struct agx_apple9_vir_program *program,
                                   const uint32_t xy[2], const uint32_t rgba[4],
                                   uint32_t index, uint32_t layer, bool integer);

/* Generalized bindless PBE image write:
 * - dimension: 0 = 2D, 1 = Cube, 2 = 2D-array (layered), 3 = 3D
 * - bits: 32 (4 RGBA data words) or 16 (2 packed RG/BA data words)
 * - coords: {x, y} for 2D; {x, y, z/layer/face} for 3D / 2D-array / Cube
 * - data: 4 words for 32-bit, 2 packed words for 16-bit
 * - integer: false = float (f32/f16), true = integer (i32/u32/i16/u16)
 * - fragment: true for fragment stage (fa/da/ea/ca + 0x08), false for CS/VS */
bool agx_apple9_vir_emit_image_store_ex(struct agx_apple9_vir_program *program,
                                      const uint32_t *coords,
                                      const uint32_t *data, uint32_t index,
                                      unsigned dimension, unsigned bits,
                                      bool integer, bool fragment);

/* Aligned threadgroup (shared) memory loads and stores. components may be
 * 1, 2, 3, or 4 for 32-bit elements (4B/8B/12B/16B-scaled index). Loads
 * allocate an adjacent SSA tuple with AUTO scoreboard slot; stores consume
 * an adjacent data tuple. Both address r0..r95 and accept input waits. */
uint32_t agx_apple9_vir_emit_shared_load(struct agx_apple9_vir_program *program,
                                         uint32_t index, unsigned root,
                                         unsigned components);
bool agx_apple9_vir_emit_shared_store(struct agx_apple9_vir_program *program,
                                      const uint32_t *data, unsigned components,
                                      uint32_t index, unsigned root);
/* Typed threadgroup (shared) memory loads and stores:
 * - bits=8:  components=1 (1 GPR low byte, 1B-scaled index)
 * - bits=16: components=1 (1 GPR low half, 2B-scaled), components=2 (1 packed
 *            GPR, 4B-scaled), components=4 (2 packed GPRs, 8B-scaled)
 * - bits=32: components=1..4 (1..4 GPRs, 4B/8B/12B/16B-scaled) */
uint32_t agx_apple9_vir_emit_shared_load_typed(
   struct agx_apple9_vir_program *program, uint32_t index, unsigned root,
   unsigned components, unsigned bits);
bool agx_apple9_vir_emit_shared_store_typed(
   struct agx_apple9_vir_program *program, const uint32_t *data,
   unsigned components, unsigned bits, uint32_t index, unsigned root);

/* Typed invocation-private (scratch) memory loads and stores with multi-slot
 * scoreboard allocation (1..6) and input wait mask:
 * - bits=8,  components=1: 1B-scaled index
 * - bits=16, components=1: 2B-scaled index
 * - bits=32, components=1: 4B-scaled index
 * - bits=32, components=2: 8B-scaled index (2 adjacent GPRs)
 * - bits=32, components=4: 1B-scaled (caller pre-shifts index by 4; 4 GPRs) */
uint32_t agx_apple9_vir_emit_private_load(
   struct agx_apple9_vir_program *program, uint32_t index, unsigned word_offset,
   unsigned components, unsigned bits);
bool agx_apple9_vir_emit_private_store(
   struct agx_apple9_vir_program *program, const uint32_t *data,
   unsigned components, unsigned bits, uint32_t index, unsigned word_offset);

/* Asynchronous multi-slot (1..6) GET_SR targeting r0..r95 (8-byte extended form,
 * automatically compacted to 4-byte GET_SR when allocated to r0..r63). */
uint32_t agx_apple9_vir_emit_get_sr_async(
   struct agx_apple9_vir_program *program, uint8_t selector, uint8_t datapath);

enum agx_apple9_subgroup_reduce_op {
   AGX_APPLE9_SUBGROUP_REDUCE_AND = 0,
   AGX_APPLE9_SUBGROUP_REDUCE_OR = 1,
   AGX_APPLE9_SUBGROUP_REDUCE_XOR = 2,
   AGX_APPLE9_SUBGROUP_REDUCE_IADD = 3,
   AGX_APPLE9_SUBGROUP_REDUCE_IMIN = 4,
   AGX_APPLE9_SUBGROUP_REDUCE_IMAX = 5,
   AGX_APPLE9_SUBGROUP_REDUCE_UMIN = 6,
   AGX_APPLE9_SUBGROUP_REDUCE_UMAX = 7,
   AGX_APPLE9_SUBGROUP_REDUCE_FADD = 8,
   AGX_APPLE9_SUBGROUP_REDUCE_FMUL = 9,
   AGX_APPLE9_SUBGROUP_REDUCE_FMIN = 10,
   AGX_APPLE9_SUBGROUP_REDUCE_FMAX = 11,
   AGX_APPLE9_SUBGROUP_REDUCE_HADD = 12,
   AGX_APPLE9_SUBGROUP_REDUCE_HMIN = 13,
   AGX_APPLE9_SUBGROUP_REDUCE_HMAX = 14,
};

enum agx_apple9_subgroup_reduce_flags {
   AGX_APPLE9_SUBGROUP_REDUCE_QUAD = 1u << 8,
   AGX_APPLE9_SUBGROUP_REDUCE_PREFIX = 1u << 9,
   AGX_APPLE9_SUBGROUP_REDUCE_16BIT = 1u << 10,
};

enum agx_apple9_subgroup_shuffle_mode {
   AGX_APPLE9_SUBGROUP_SHUFFLE_IDX = 0,
   AGX_APPLE9_SUBGROUP_SHUFFLE_XOR = 1,
   AGX_APPLE9_SUBGROUP_SHUFFLE_UP = 2,
   AGX_APPLE9_SUBGROUP_SHUFFLE_DOWN = 3,
   AGX_APPLE9_QUAD_SHUFFLE_IDX = 4,
   AGX_APPLE9_QUAD_SHUFFLE_XOR = 5,
};

/* Native 1-instruction SIMD & Quad reductions and exclusive prefix scans.
 * Integer ops support bits=16 or 32; FADD/FMUL/FMIN/FMAX require bits=32;
 * HADD/HMIN/HMAX require bits=16. Exclusive prefix scans are supported for
 * IADD (32/16-bit), FADD, FMUL (32-bit), and HADD (16-bit). */
uint32_t agx_apple9_vir_emit_subgroup_reduce(
   struct agx_apple9_vir_program *program,
   enum agx_apple9_subgroup_reduce_op op, uint32_t src, unsigned bits,
   bool quad, bool prefix_exclusive);

/* 2-instruction inclusive prefix scan (exclusive scan + IADD/FADD/FMUL/HADD). */
uint32_t agx_apple9_vir_emit_subgroup_inclusive_scan(
   struct agx_apple9_vir_program *program,
   enum agx_apple9_subgroup_reduce_op op, uint32_t src, unsigned bits,
   bool quad);

/* Native 10-byte SIMD & Quad shuffles (register or 5-bit immediate lane/delta). */
uint32_t agx_apple9_vir_emit_subgroup_shuffle(
   struct agx_apple9_vir_program *program,
   enum agx_apple9_subgroup_shuffle_mode mode, uint32_t src,
   uint32_t lane_or_delta);
uint32_t agx_apple9_vir_emit_subgroup_shuffle_imm(
   struct agx_apple9_vir_program *program,
   enum agx_apple9_subgroup_shuffle_mode mode, uint32_t src,
   uint8_t lane_or_delta);

/* Native 12-byte simd_shuffle_and_fill_down (C7 06) for subgroupRotate:
 * shifts data down by delta (register or 0..31 immediate), filling upper lanes
 * from fill (pass fill == data for full 32-lane rotate). */
uint32_t agx_apple9_vir_emit_subgroup_rotate(
   struct agx_apple9_vir_program *program, uint32_t data, uint32_t fill,
   uint32_t delta);
uint32_t agx_apple9_vir_emit_subgroup_rotate_imm(
   struct agx_apple9_vir_program *program, uint32_t data, uint32_t fill,
   uint8_t delta);

/* Apple9 (G16G) 8x8x8 SIMD-group MMA (opcode 0xCF: CF 02 FP32 dst, CF 00 FP16 dst). */
enum agx_apple9_mma_type {
   AGX_APPLE9_MMA_BF16 = 0,
   AGX_APPLE9_MMA_F16  = 1,
   AGX_APPLE9_MMA_F32  = 2,
};

#define AGX_APPLE9_MMA_HAS_C (1u << 8)

static inline uint32_t
agx_apple9_mma_flags(enum agx_apple9_mma_type ty_a,
                     enum agx_apple9_mma_type ty_b,
                     enum agx_apple9_mma_type ty_c,
                     enum agx_apple9_mma_type ty_dst,
                     bool has_c)
{
   return ((uint32_t)ty_a & 3u) |
          (((uint32_t)ty_b & 3u) << 2) |
          (((uint32_t)ty_c & 3u) << 4) |
          (((uint32_t)ty_dst & 3u) << 6) |
          (has_c ? AGX_APPLE9_MMA_HAS_C : 0u);
}

static inline enum agx_apple9_mma_type
agx_apple9_mma_src_type(uint32_t flags, unsigned op_idx)
{
   return (enum agx_apple9_mma_type)((flags >> (op_idx * 2u)) & 3u);
}

static inline enum agx_apple9_mma_type
agx_apple9_mma_dst_type(uint32_t flags)
{
   return (enum agx_apple9_mma_type)((flags >> 6u) & 3u);
}

static inline bool
agx_apple9_mma_has_c(uint32_t flags)
{
   return !!(flags & AGX_APPLE9_MMA_HAS_C);
}

static inline unsigned
agx_apple9_mma_type_gprs(enum agx_apple9_mma_type ty)
{
   return ty == AGX_APPLE9_MMA_F32 ? 2u : 1u;
}

static inline unsigned
agx_apple9_mma_src_layout(uint32_t flags,
                          unsigned offsets[3], unsigned widths[3])
{
   widths[0] = agx_apple9_mma_type_gprs(agx_apple9_mma_src_type(flags, 0));
   widths[1] = agx_apple9_mma_type_gprs(agx_apple9_mma_src_type(flags, 1));
   widths[2] = agx_apple9_mma_has_c(flags)
                  ? agx_apple9_mma_type_gprs(agx_apple9_mma_src_type(flags, 2))
                  : 0u;
   offsets[0] = 0;
   offsets[1] = widths[0];
   offsets[2] = widths[0] + widths[1];
   return widths[0] + widths[1] + widths[2];
}

static inline unsigned
agx_apple9_mma_source_width(const struct agx_apple9_vir_instr *ins,
                            unsigned source)
{
   unsigned off[3], w[3];
   agx_apple9_mma_src_layout(ins->immediate, off, w);
   for (unsigned k = 0; k < 3; ++k) {
      if (w[k] && source == off[k])
         return w[k];
   }
   return 1;
}

static inline enum agx_apple9_operand_role
agx_apple9_mma_source_role(const struct agx_apple9_vir_instr *ins,
                           unsigned source)
{
   unsigned off[3], w[3];
   agx_apple9_mma_src_layout(ins->immediate, off, w);
   if (source < off[1])
      return AGX_APPLE9_OPERAND_SRC0;
   if (source < off[2])
      return AGX_APPLE9_OPERAND_SRC1;
   return AGX_APPLE9_OPERAND_SRC2;
}

uint32_t agx_apple9_vir_emit_mma(struct agx_apple9_vir_program *program,
                                 const uint32_t *src_a,
                                 const uint32_t *src_b,
                                 const uint32_t *src_c,
                                 uint32_t flags);

/* Native single-instruction ALU unlocks. */
uint32_t agx_apple9_vir_emit_iadd_sat(struct agx_apple9_vir_program *program,
                                      uint32_t a, uint32_t b, bool is_sub,
                                      bool is_signed);
uint32_t agx_apple9_vir_emit_iandn(struct agx_apple9_vir_program *program,
                                   uint32_t a, uint32_t b);
uint32_t agx_apple9_vir_emit_extr(struct agx_apple9_vir_program *program,
                                  uint32_t hi, uint32_t lo, uint32_t shift);
uint32_t agx_apple9_vir_emit_extr_imm(struct agx_apple9_vir_program *program,
                                      uint32_t hi, uint32_t lo, uint8_t shift);
/* Masked left shift: (src & ((1u << width) - 1)) << shift (width 1..31). */
uint32_t agx_apple9_vir_emit_bfi_imm(struct agx_apple9_vir_program *program,
                                     uint32_t src, uint8_t shift,
                                     uint8_t width);
uint32_t agx_apple9_vir_emit_pack_snorm_2x16(
   struct agx_apple9_vir_program *program, uint32_t a, uint32_t b);
uint32_t agx_apple9_vir_emit_pack_snorm_4x8(
   struct agx_apple9_vir_program *program, const uint32_t src[4]);
uint32_t agx_apple9_vir_emit_int_to_f16(struct agx_apple9_vir_program *program,
                                        uint32_t src, unsigned src_bits,
                                        bool is_signed);
uint32_t agx_apple9_vir_emit_f32_to_narrow_int(
   struct agx_apple9_vir_program *program, uint32_t src, unsigned dst_bits,
   bool is_signed);

/* 1D and 1D-array texture sample (0x0080/0x00a0) or fetch (0x0c80/0x0ca0).
 * Pass layer_or_invalid == AGX_APPLE9_VREG_INVALID for non-array 1D. */
uint32_t agx_apple9_vir_emit_texture_1d(struct agx_apple9_vir_program *program,
                                        uint32_t coord,
                                        uint32_t layer_or_invalid, uint32_t one,
                                        unsigned texture, unsigned sampler,
                                        bool fetch);

/* Implicit-LOD array / shadow sample using a 4-word parameter tuple:
 * - dimension=2 (2D-array): !shadow (0x0408, +offset 0x0301) or shadow (0x0420,
 *   +offset 0x0321)
 * - dimension=4 (Cube-array): shadow (0x0430) */
uint32_t agx_apple9_vir_emit_texture_array_implicit(
   struct agx_apple9_vir_program *program, const uint32_t coords[4],
   unsigned texture, unsigned sampler, unsigned dimension, bool shadow);

bool agx_apple9_vir_emit_block_image_store(
   struct agx_apple9_vir_program *program, const uint32_t src[4],
   unsigned image, unsigned format, bool multisampled, bool array);

uint32_t agx_apple9_vir_emit_texture_sample(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint32_t one, unsigned texture, unsigned sampler);
uint32_t agx_apple9_vir_emit_texture_sample_mov(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   unsigned texture, unsigned sampler);
uint32_t agx_apple9_vir_emit_texture_lod_imm(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint8_t lod_uq4_2, unsigned texture, unsigned sampler);

uint32_t agx_apple9_vir_emit_texture_min_lod(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint32_t zero, uint32_t bias_min_lod, unsigned texture, unsigned sampler);
uint32_t agx_apple9_vir_emit_texture_ms(
   struct agx_apple9_vir_program *program, const uint32_t coords[4],
   unsigned texture, unsigned sampler, bool array);

/* Form an adjacent register tuple from independent scalar SSA values before
 * register allocation. The pseudo is coalesced when possible and otherwise
 * lowered to copies after allocation, following the Apple8 AGX IR model. */
/* FP32 coordinates and signed Q6 LOD packed in bits16..27. */
uint32_t agx_apple9_vir_emit_texture_lod(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint32_t packed_lod, unsigned texture, unsigned sampler, bool bias);

uint32_t agx_apple9_vir_emit_texture_volume(
   struct agx_apple9_vir_program *program, const uint32_t coords[3],
   uint32_t packed_lod, unsigned texture, unsigned sampler, unsigned dimension,
   bool bias, bool shadow);

/* Volume / cube / array / shadow sample with native min_lod_clamp in word 3
 * (packed Q6 min_lod << 16 for implicit forms, or bias<<16 | min_lod for
 * non-array biased forms):
 * - 2D (0): !shadow bias=true (0x0804, +offset 0x02c1);
 *           shadow bias=false (0x0820), bias=true (0x0824)
 * - Cube (1): !shadow bias=false (0x0810), bias=true (0x0814)
 * - 2D-array (2): !shadow bias=true (+offset 0x03c1);
 *                 shadow bias=false (0x0c20, +offset 0x0361)
 * - 3D (3): !shadow bias=false (0x2180), bias=true (0x2190)
 * - Cube-array (4): !shadow bias=false (0x0c10) */
uint32_t agx_apple9_vir_emit_texture_volume_min_lod(
   struct agx_apple9_vir_program *program, const uint32_t coords[3],
   uint32_t packed_min_lod, unsigned texture, unsigned sampler,
   unsigned dimension, bool bias, bool shadow);

/* Coordinates followed by derivatives interleaved by texture axis. Extended
 * forms publish four coordinate words, then four (array/shadow/min LOD) or
 * six (3D) derivatives. Array layers are unsigned integers in the low half
 * of coordinate word 3, and a minimum LOD occupies its high half. */
uint32_t agx_apple9_vir_emit_texture_grad(struct agx_apple9_vir_program *program,
                                          const uint32_t *src, unsigned texture,
                                          unsigned sampler, unsigned dimension,
                                          bool shadow, bool min_lod);
bool agx_apple9_vir_set_texture_sparse(struct agx_apple9_vir_program *program,
                                       struct agx_apple9_vir_instr *sample,
                                       unsigned mask);

uint32_t agx_apple9_vir_emit_collect(struct agx_apple9_vir_program *program,
                                     const uint32_t *src, unsigned components);
uint32_t agx_apple9_vir_emit_split(struct agx_apple9_vir_program *program,
                                   uint32_t src, unsigned components);
uint32_t agx_apple9_vir_emit_mov(struct agx_apple9_vir_program *program,
                                 uint32_t src);
bool agx_apple9_vir_emit_swap(struct agx_apple9_vir_program *program,
                              uint32_t a, uint32_t b);
void agx_apple9_place_phis(struct agx_apple9_vir_program *program);
/* Native 32-bit GPR parallel-copy scheduler (decoupled from agx_context). */
bool agx_apple9_resolve_phi_edge(const struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_instr *edge,
   bool (*emit)(void *data, bool swap, unsigned dest, unsigned source), void *data);

uint32_t agx_apple9_vir_emit_phi(struct agx_apple9_vir_program *program,
                                  struct agx_apple9_block *block);
/* Allocate or resize a canonical VIR_PHI's predecessor operand array. Uses
 * inline src[] for <= AGX_APPLE9_MAX_VIR_SRCS preds and sideband phi_srcs[]
 * above that; initializes all entries to AGX_APPLE9_VREG_INVALID. */
bool agx_apple9_vir_phi_alloc_srcs(struct agx_apple9_vir_instr *phi,
                                   unsigned num_preds);
uint32_t agx_apple9_vir_emit_phi_canonical(struct agx_apple9_vir_program *program,
                                           struct agx_apple9_block *block,
                                           unsigned num_preds);
/* Convert VIR_PHI + VIR_PHI_SRC records into canonical variable-arity VIR_PHI
 * nodes placed at logical block entry, indexed by block->cfg_preds[]. */
bool agx_apple9_vir_canonicalize_phis(struct agx_apple9_vir_program *program);
bool agx_apple9_vir_emit_phi_source(struct agx_apple9_vir_program *program,
                                     uint32_t target, uint32_t source);
struct agx_apple9_vir_copy {
   uint32_t target, source;
};
bool agx_apple9_vir_emit_phi_edge(
   struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_copy *copies, unsigned count);

bool
agx_apple9_vir_set_device_store_address(struct agx_apple9_vir_program *program,
                                        uint32_t address);
bool agx_apple9_vir_emit_device_store(struct agx_apple9_vir_program *program,
                                      unsigned binding, uint32_t index,
                                      const uint32_t *data, unsigned components,
                                      unsigned bits);
bool agx_apple9_vir_emit_device_atomic(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const uint32_t *data, unsigned data_components,
   enum agx_apple9_atomic_op op, bool discard_result, uint32_t *result_out);

bool agx_apple9_vir_set_device_atomic_address(
   struct agx_apple9_vir_program *program, uint32_t address);
bool agx_apple9_vir_set_device_load_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   enum agx_apple9_scoreboard_slot scoreboard_slot);
bool agx_apple9_vir_set_device_load_raw_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   uint16_t raw_token);
bool agx_apple9_vir_set_device_load_index_kind(
   struct agx_apple9_vir_program *program, uint32_t value,
   enum agx_apple9_device_load_index_kind index_kind);
bool agx_apple9_vir_add_live_out(struct agx_apple9_vir_program *program,
                                 uint32_t value);
bool agx_apple9_vir_add_abi_output(struct agx_apple9_vir_program *program,
                                   uint32_t value, unsigned phys_gpr);
uint32_t agx_apple9_vir_emit_entry_reg(struct agx_apple9_vir_program *program,
                                       unsigned phys_gpr);
bool agx_apple9_vir_emit_abi_return(struct agx_apple9_vir_program *program);

struct nir_shader;
struct agx_apple9_allocation_cost {
   uint64_t spills, instructions, bytes, dependency_gaps;
   unsigned registers, peak_registers;
   bool has_copies;
};
bool agx_apple9_allocation_cheaper(struct agx_apple9_allocation_cost a,
                                   struct agx_apple9_allocation_cost b);
bool agx_apple9_allocate_shared(struct agx_apple9_vir_program *program,
                                 struct nir_shader *nir, const char **reason);
bool agx_apple9_allocate_publications(struct agx_apple9_vir_program *program,
                                      const char **reason);

bool
agx_apple9_validate_mesh_control(const struct agx_apple9_vir_program *program,
                                const char **reason);

bool
agx_apple9_validate_vir_allocation(const struct agx_apple9_vir_program *program,
                                   const char **reason);

/* Allocate pending asynchronous producers to scoreboard slots from final VIR
 * order.  The first capable consumer performs the handoff; later reads use
 * slot 0. */
bool
agx_apple9_assign_vir_scoreboard_slots(struct agx_apple9_vir_program *program,
                                       const char **reason);

/* One physical Apple9 instruction, used by the compiler and packer tests. */
struct agx_apple9_packed_instruction {
   /* Includes bounded multi-instruction parameter-publication pseudos. */
   uint8_t bytes[128];
   uint8_t length;
};

/* Start-relative byte displacement, encoded in the branch's signed 48-bit
 * field. Unlike local CFG offsets, entry transfers may cross the signed-32-bit
 * range. */
bool agx_apple9_pack_branch(bool any, int64_t displacement,
                            struct agx_apple9_packed_instruction *packed);

bool agx_apple9_pack_vir_instruction(
   const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
   struct agx_apple9_packed_instruction *packed, const char **reason);

bool
agx_apple9_pack_get_global_id(unsigned dst, unsigned component,
                              struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_get_sr(unsigned dst, uint8_t selector, uint8_t datapath,
                            struct agx_apple9_packed_instruction *packed);
/* Multi-slot (1..6, or NONE for slot 1) GET_SR. Set extended=true for the
 * 8-byte r0..r95 form; extended=false emits the 4-byte r0..r63 form. */
bool agx_apple9_pack_get_sr_ex(unsigned dst, uint8_t selector, uint8_t datapath,
                               enum agx_apple9_scoreboard_slot slot,
                               bool extended,
                               struct agx_apple9_packed_instruction *packed);
/* Local-output stores use byte displacements and optional element indices.
 * Narrow data occupies the low half of a word-allocated SSA value. */
bool agx_apple9_pack_local_store(unsigned data, unsigned root, unsigned index,
                                 unsigned bits,
                                 enum agx_apple9_local_index_kind index_kind,
                                 unsigned offset_B, bool retain_data,
                                 bool release_index,
                                 struct agx_apple9_packed_instruction *packed);
bool
agx_apple9_pack_get_sr_zext16(unsigned dst, uint8_t selector,
                              struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov_imm(unsigned dst, unsigned value,
                             struct agx_apple9_packed_instruction *packed);
/* Full active-SIMD unsigned maximum. Operands are half-register indices;
 * 32-bit operands must be even. The normal GPR form preserves other halves. */
bool agx_apple9_pack_subgroup_reduce_umax(
   unsigned dst_half, unsigned src_half, unsigned bit_size,
   bool dst_cache, bool src_cache, bool src_discard,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov_imm32(unsigned dst, uint32_t value,
                               struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov(unsigned dst, unsigned src,
                         struct agx_apple9_packed_instruction *packed);

bool
agx_apple9_pack_device_load_u32(unsigned dst, unsigned index, unsigned binding,
                                uint8_t flags,
                                enum agx_apple9_scoreboard_slot scoreboard_slot,
                                struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_load_u32_raw(
   unsigned dst, unsigned index, unsigned binding,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);

/* Aligned native vector memory forms.  A load defines components consecutive
 * 32-bit GPRs beginning at dst and publishes the complete tuple through one
 * scoreboard slot.  A store consumes components consecutive GPRs beginning
 * at data and allocates no scoreboard slot of its own.  Device-store
 * access_desc bit 0 is the index last-use control: clear retains the index
 * GPR and set releases it after the address read. */
bool agx_apple9_pack_device_load_vector_u32_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_load_vector_u16_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_load_scalar_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_store_scalar(
   unsigned data, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_store_vector_u32(
   unsigned data, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_store_vector_u16(
   unsigned data, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_atomic(
   unsigned index, unsigned data, unsigned binding, enum agx_apple9_atomic_op op,
   bool discard_result,
   enum agx_apple9_scoreboard_slot input_dependency,
   struct agx_apple9_packed_instruction *packed);

#ifdef __cplusplus
}
#endif

#endif
