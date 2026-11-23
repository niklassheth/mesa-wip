#!/usr/bin/env python3
# Copyright 2026 The Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Declarative Apple9 machine encoding table and C table generator.

Generates `agx_apple9_machine_table.h` and `agx_apple9_machine_table.c` from a
compact, validated Python specification of every `enum agx_apple9_encoding`
form in `agx_apple9_machine.h`.

This table is deliberately conservative: `allocator_safe` means every
register-bearing field required by the form is understood well enough for a
packer, not merely that the compiler has emitted examples of the form.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


PUBLICATION_COUNT = 128
GPR_COUNT = 96
UNIFORM_COUNT = 256
MAX_ENCODING_OPERANDS = 11

# Evidence levels (enum agx_apple9_evidence).
UNRESOLVED = "AGX_APPLE9_EVIDENCE_UNRESOLVED"
BYTE_DIFF = "AGX_APPLE9_EVIDENCE_BYTE_DIFF"
HW = "AGX_APPLE9_EVIDENCE_HARDWARE"
VALID_EVIDENCE = {UNRESOLVED, BYTE_DIFF, HW}

# Register files (enum agx_apple9_register_file).
F_GPR = "AGX_APPLE9_FILE_GPR"
F_UNIFORM = "AGX_APPLE9_FILE_UNIFORM"
F_IMM = "AGX_APPLE9_FILE_IMMEDIATE"
F_IMPLICIT = "AGX_APPLE9_FILE_IMPLICIT"
VALID_FILES = {F_GPR, F_UNIFORM, F_IMM, F_IMPLICIT}
VALID_FILE_COMBOS = {
    frozenset({F_GPR}),
    frozenset({F_UNIFORM}),
    frozenset({F_IMPLICIT}),
    frozenset({F_GPR, F_UNIFORM}),
    frozenset({F_GPR, F_IMM}),
    frozenset({F_GPR, F_UNIFORM, F_IMM}),
}

# Operand roles (enum agx_apple9_operand_role).
DST = "AGX_APPLE9_OPERAND_DEST"
SRC0 = "AGX_APPLE9_OPERAND_SRC0"
SRC1 = "AGX_APPLE9_OPERAND_SRC1"
SRC2 = "AGX_APPLE9_OPERAND_SRC2"
SRC3 = "AGX_APPLE9_OPERAND_SRC3"
SRC4 = "AGX_APPLE9_OPERAND_SRC4"
SRC5 = "AGX_APPLE9_OPERAND_SRC5"
SRC6 = "AGX_APPLE9_OPERAND_SRC6"
SRC7 = "AGX_APPLE9_OPERAND_SRC7"
SRC8 = "AGX_APPLE9_OPERAND_SRC8"
SRC9 = "AGX_APPLE9_OPERAND_SRC9"
INDEX = "AGX_APPLE9_OPERAND_INDEX"
STORE_DATA = "AGX_APPLE9_OPERAND_STORE_DATA"
ATOMIC_DATA = "AGX_APPLE9_OPERAND_ATOMIC_DATA"
SRC_ROLES = (SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8, SRC9)
VALID_ROLES = {DST, *SRC_ROLES, INDEX, STORE_DATA, ATOMIC_DATA}

# Operand widths (enum agx_apple9_width).
W16 = "AGX_APPLE9_WIDTH_16"
W32 = "AGX_APPLE9_WIDTH_32"
W64 = "AGX_APPLE9_WIDTH_64"
W16_32 = (W16, W32)
W16_32_64 = (W16, W32, W64)
VALID_WIDTHS = {W16, W32, W64}

# Operand flags (enum agx_apple9_operand_flag).
ALLOC = "AGX_APPLE9_OPERAND_ALLOCATABLE"
HARD_LOW = "AGX_APPLE9_OPERAND_HARD_LOW"
COMPACT = "AGX_APPLE9_OPERAND_COMPACT_PREFERRED"
SCATTERED = "AGX_APPLE9_OPERAND_SCATTERED"
IMPLICIT_FLAG = "AGX_APPLE9_OPERAND_IMPLICIT"
UNIFORM_ALT = "AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE"
CLOBBER = "AGX_APPLE9_OPERAND_CLOBBER"
PUB = "AGX_APPLE9_OPERAND_PUBLICATION"
VALID_FLAGS = {
    ALLOC, HARD_LOW, COMPACT, SCATTERED, IMPLICIT_FLAG,
    UNIFORM_ALT, CLOBBER, PUB,
}

# Dependency layouts (enum agx_apple9_dependency_layout).
DEP_NONE = "AGX_APPLE9_DEPENDENCY_NONE"
DEP_IDX_29_31 = "AGX_APPLE9_DEPENDENCY_INDEX_29_31"
DEP_IDX_45_47 = "AGX_APPLE9_DEPENDENCY_INDEX_45_47"
DEP_IDX_61_63 = "AGX_APPLE9_DEPENDENCY_INDEX_61_63"
DEP_MASK_12_17 = "AGX_APPLE9_DEPENDENCY_MASK_12_17"
DEP_MASK_45_61 = "AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63"
DEP_MASK_61_77 = "AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79"
DEP_MASK_61_93 = "AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95"
VALID_DEPS = {
    DEP_NONE, DEP_IDX_29_31, DEP_IDX_45_47, DEP_IDX_61_63,
    DEP_MASK_12_17, DEP_MASK_45_61, DEP_MASK_61_77, DEP_MASK_61_93,
}

# Prior-completion exemption classes (enum agx_apple9_completion_class).
EXEMPT_MEM = "AGX_APPLE9_COMPLETION_MEMORY"
EXEMPT_PUB = "AGX_APPLE9_COMPLETION_PUBLICATION"
EXEMPT_OTHER = "AGX_APPLE9_COMPLETION_OTHER"
EXEMPT_TEX = "AGX_APPLE9_COMPLETION_TEXTURE"
EXEMPT_IMG_STORE = "AGX_APPLE9_COMPLETION_IMAGE_STORE"
VALID_EXEMPTIONS = {
    EXEMPT_MEM, EXEMPT_PUB, EXEMPT_OTHER, EXEMPT_TEX, EXEMPT_IMG_STORE,
}


def _as_tuple(val: str | tuple[str, ...]) -> tuple[str, ...]:
    if not val:
        return ()
    if isinstance(val, str):
        return (val,)
    return tuple(val)


@dataclass(frozen=True)
class Operand:
    role: str
    files: tuple[str, ...]
    widths: tuple[str, ...]
    min_index: int
    max_index: int
    uniform_max_index: int
    alignment_halves: int
    flags: tuple[str, ...]
    evidence: str | None = None


@dataclass(frozen=True)
class Encoding:
    name: str
    enum_name: str
    length: int
    operands: tuple[Operand, ...]
    allocator_safe: bool
    evidence: str
    dependency_layout: str
    dependency_retains_sources: bool
    fixed_producer_slot: int
    fixed_producer_alternates: int
    requires_prior_completion: bool
    prior_completion_exempt: tuple[str, ...]
    split_result: bool


def _operand(
    role: str,
    files: tuple[str, ...],
    max_idx: int,
    *,
    widths: str | tuple[str, ...] = W32,
    align: int = 2,
    flags: str | tuple[str, ...] = ALLOC,
    min_idx: int = 0,
    umax: int = 0,
    ev: str | None = None,
) -> Operand:
    return Operand(
        role=role, files=files, widths=_as_tuple(widths),
        min_index=min_idx, max_index=max_idx, uniform_max_index=umax,
        alignment_halves=align, flags=_as_tuple(flags), evidence=ev,
    )


def gpr(role: str, max_idx: int = GPR_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
        align: int = 2, flags: str | tuple[str, ...] = ALLOC, min_idx: int = 0,
        ev: str | None = None) -> Operand:
    """Ordinary GPR operand."""
    return _operand(role, (F_GPR,), max_idx, widths=widths, align=align,
                    flags=flags, min_idx=min_idx, ev=ev)


def gprs(*roles: str, max_idx: int = GPR_COUNT - 1, widths: str | tuple[str, ...] = W32,
         align: int = 2, flags: str | tuple[str, ...] = ALLOC, min_idx: int = 0,
         ev: str | None = None) -> tuple[Operand, ...]:
    """Multiple identical GPR operands."""
    return tuple(gpr(r, max_idx, widths=widths, align=align, flags=flags,
                     min_idx=min_idx, ev=ev) for r in roles)


def src_gprs(count: int, *, start: int = 0, max_idx: int = GPR_COUNT - 1,
             widths: str | tuple[str, ...] = W32, align: int = 2,
             flags: str | tuple[str, ...] = ALLOC, ev: str | None = None) -> tuple[Operand, ...]:
    """Consecutive GPR source operands SRC<start>..SRC<start+count-1>."""
    if start < 0 or count <= 0 or start + count > len(SRC_ROLES):
        raise ValueError(f"invalid src_gprs range: start={start}, count={count}")
    return gprs(*SRC_ROLES[start : start + count],
                max_idx=max_idx, widths=widths, align=align, flags=flags, ev=ev)


def reg_src(role: str, max_idx: int = GPR_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
            align: int = 2, flags: str | tuple[str, ...] = ALLOC,
            umax: int = UNIFORM_COUNT - 1, ev: str | None = None) -> Operand:
    """GPR | UNIFORM ALU source operand."""
    return _operand(role, (F_GPR, F_UNIFORM), max_idx, widths=widths, align=align,
                    flags=flags, umax=umax, ev=ev)


def imm_src(role: str, max_idx: int = GPR_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
            align: int = 2, flags: str | tuple[str, ...] = ALLOC,
            umax: int = UNIFORM_COUNT - 1, ev: str | None = None) -> Operand:
    """GPR | UNIFORM | IMMEDIATE ALU source operand."""
    return _operand(role, (F_GPR, F_UNIFORM, F_IMM), max_idx, widths=widths, align=align,
                    flags=flags, umax=umax, ev=ev)


def gpr_imm(role: str, max_idx: int = GPR_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
            align: int = 2, flags: str | tuple[str, ...] = ALLOC,
            ev: str | None = None) -> Operand:
    """GPR | IMMEDIATE operand (no uniform file)."""
    return _operand(role, (F_GPR, F_IMM), max_idx, widths=widths, align=align,
                    flags=flags, ev=ev)


def uniform(role: str, max_idx: int = UNIFORM_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
            align: int = 2, flags: str | tuple[str, ...] = (), min_idx: int = 0,
            ev: str | None = None) -> Operand:
    """Uniform-bank operand."""
    return _operand(role, (F_UNIFORM,), max_idx, widths=widths, align=align,
                    flags=flags, min_idx=min_idx, ev=ev)


def pub(role: str, max_idx: int = PUBLICATION_COUNT - 1, *, widths: str | tuple[str, ...] = W32,
        align: int = 2, flags: str | tuple[str, ...] = (ALLOC, PUB),
        ev: str | None = None) -> Operand:
    """Publication/export GPR operand."""
    return _operand(role, (F_GPR,), max_idx, widths=widths, align=align,
                    flags=flags, ev=ev)


def implicit(role: str, *, widths: str | tuple[str, ...] = W32,
             ev: str | None = None) -> Operand:
    """Implicit operand supplied by the immediately preceding instruction."""
    return _operand(role, (F_IMPLICIT,), 0xFF, widths=widths, align=1,
                    flags=(IMPLICIT_FLAG,), min_idx=0xFF, ev=ev)


def enc(
    name: str,
    length: int,
    operands: tuple[Operand, ...] | list[Operand] = (),
    *,
    enum_name: str | None = None,
    allocator_safe: bool = True,
    ev: str = HW,
    dep: str = DEP_NONE,
    retains: bool = False,
    fixed_slot: int = 0,
    fixed_alt: int = 0,
    prior: bool = False,
    exempt: str | tuple[str, ...] = (),
    split_result: bool = False,
) -> Encoding:
    """Construct an Apple9 encoding descriptor with sensible defaults."""
    resolved_ops = tuple(
        Operand(
            role=op.role,
            files=op.files,
            widths=op.widths,
            min_index=op.min_index,
            max_index=op.max_index,
            uniform_max_index=op.uniform_max_index,
            alignment_halves=op.alignment_halves,
            flags=op.flags,
            evidence=ev if op.evidence is None else op.evidence,
        )
        for op in operands
    )
    return Encoding(
        name=name,
        enum_name=enum_name or f"AGX_APPLE9_ENC_{name.upper()}",
        length=length,
        operands=resolved_ops,
        allocator_safe=allocator_safe,
        evidence=ev,
        dependency_layout=dep,
        dependency_retains_sources=retains,
        fixed_producer_slot=fixed_slot,
        fixed_producer_alternates=fixed_alt,
        requires_prior_completion=prior,
        prior_completion_exempt=_as_tuple(exempt),
        split_result=split_result,
    )


# Shared operand-flag shorthands.
_ALLOC_SCAT = (ALLOC, SCATTERED)
_ALLOC_COMP = (ALLOC, COMPACT)
_ALLOC_SCAT_COMP = (ALLOC, SCATTERED, COMPACT)
_ALLOC_CLOB = (ALLOC, CLOBBER)
_ALLOC_SCAT_CLOB = (ALLOC, SCATTERED, CLOBBER)

# Shared operand lists reused across sibling forms.
_FLOAT2_EXT_OPS = (
    gpr(DST, flags=_ALLOC_SCAT),
    reg_src(SRC0, flags=_ALLOC_SCAT),
    reg_src(SRC1, flags=_ALLOC_SCAT),
)
_FLOAT3_EXT_OPS = (
    gpr(DST, flags=_ALLOC_SCAT),
    reg_src(SRC0, flags=_ALLOC_SCAT),
    imm_src(SRC1, flags=_ALLOC_SCAT),
    imm_src(SRC2, flags=_ALLOC_SCAT),
)
_SELECT_WIDE_OPS = (
    gpr(DST, flags=_ALLOC_COMP),
    reg_src(SRC0, flags=_ALLOC_SCAT),
    imm_src(SRC1, flags=_ALLOC_SCAT),
    imm_src(SRC2, flags=_ALLOC_SCAT),
    imm_src(SRC3, flags=_ALLOC_SCAT),
)
_IMAGE_WRITE_OPS = (
    gpr(SRC0, 40, align=16),
    *(gpr(SRC_ROLES[i], 40 + i) for i in range(1, 8)),
    gpr(SRC8, 94, align=4),
    gpr(SRC9),
)


# Ordered identically to `enum agx_apple9_encoding` in agx_apple9_machine.h.
ENCODINGS: tuple[Encoding, ...] = (
    # Moves and system registers.
    enc("mov_imm_compact", 2, [gpr(DST, 15, flags=(ALLOC, HARD_LOW, COMPACT))]),
    # Split 6-bit destination map (mode 2 reaches r0..r63).
    enc("mov_imm32", 8, [gpr(DST, 63, flags=_ALLOC_SCAT)]),
    # Short SR forms split 6-bit dest across first nibble and bits 22..23.
    enc("get_sr", 4, [gpr(DST, 63, flags=_ALLOC_COMP)]),
    # Extended 8B SR form reaches r0..r95 and publishes to scoreboard slots 1..6.
    enc("get_sr_extended", 8, [gpr(DST, flags=_ALLOC_SCAT)]),
    enc("get_sr_zext16", 8, [gpr(DST, 63, flags=_ALLOC_COMP)]),

    # Bit unary, conversions, and format packing.
    enc("bit_unary", 8, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17),
    enc("uint_to_float", 8, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17),
    enc("sint_to_float", 8, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17),
    enc("float_to_sint", 10, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17),
    enc("float_to_uint", 10, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17),
    enc("float_to_half_zext", 10, [gpr(DST, 63), gpr(SRC0)], dep=DEP_IDX_45_47),
    enc("pack_half_2x16", 12, gprs(DST, SRC0, SRC1), split_result=True),
    enc("pack_unorm_2x16", 10, gprs(DST, SRC0, SRC1)),
    # Two native instructions, each with its own dependency fields.
    enc("pack_unorm_4x8", 20, [gpr(DST), *src_gprs(4)], split_result=True),
    enc("pack_format", 14, [gpr(DST), *src_gprs(4)], dep=DEP_MASK_12_17),
    enc("pack_format_rgb", 14, [gpr(DST), *src_gprs(3)], dep=DEP_MASK_12_17),
    enc("half_to_float", 6, gprs(DST, SRC0), dep=DEP_IDX_45_47),
    enc("unpack_norm", 8, [gpr(DST, 94, widths=W64, align=4), gpr(SRC0)], dep=DEP_MASK_12_17),

    # Half-precision float ALU and predicates.
    enc("half2", 6, gprs(DST, SRC0, SRC1), dep=DEP_IDX_45_47),
    enc("half3", 8, gprs(DST, SRC0, SRC1, SRC2), dep=DEP_IDX_61_63),
    enc("half_minmax", 6, gprs(DST, SRC0, SRC1), dep=DEP_IDX_45_47),
    enc("half_special", 10, gprs(DST, SRC0), dep=DEP_MASK_12_17),
    enc("half_compare_select", 10, [gpr(DST, flags=_ALLOC_COMP), *src_gprs(4, flags=_ALLOC_SCAT)],
        dep=DEP_IDX_61_63),
    enc("half_predicate_short", 6, gprs(SRC0, SRC1), dep=DEP_IDX_45_47),
    enc("half_predicate_extended", 10, gprs(SRC0, SRC1), dep=DEP_IDX_61_63),

    # Single-precision float ALU, cube, SFU, and derivatives.
    enc("float2_immediate_compact", 6, gprs(DST, SRC0, flags=_ALLOC_COMP),
        dep=DEP_IDX_45_47, retains=True),
    enc("float2_immediate_extended", 8, gprs(DST, SRC0, flags=_ALLOC_SCAT),
        dep=DEP_MASK_45_61, retains=True),
    enc("float2_compact", 6,
        [gpr(DST, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         reg_src(SRC0, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         reg_src(SRC1, widths=W16_32, align=1, flags=(*_ALLOC_SCAT_COMP, UNIFORM_ALT))],
        dep=DEP_IDX_45_47, retains=True),
    enc("float2_base", 4, gprs(DST, SRC0, SRC1, max_idx=63)),
    enc("float2_project", 8, gprs(DST, SRC0, SRC1, max_idx=63, align=1, flags=_ALLOC_SCAT_COMP)),
    enc("float2_modifier_extended", 8, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True),
    enc("float2_abs_extended", 10, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True),
    enc("float2_mul_abs_extended", 12, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True),
    enc("float3_extended", 8, _FLOAT3_EXT_OPS, dep=DEP_IDX_61_63, retains=True),
    enc("float3_compact", 6,
        [gpr(DST, 63), gpr(SRC0, 63), imm_src(SRC1, 63, umax=63), gpr(SRC2, 63)]),
    enc("float3_saturate_extended", 10, _FLOAT3_EXT_OPS, dep=DEP_MASK_61_77, retains=True),
    enc("float3_modifier_extended", 12, _FLOAT3_EXT_OPS, dep=DEP_MASK_61_77, retains=True),
    enc("cube", 12, gprs(DST, SRC0, SRC1, SRC2)),
    enc("float_special", 10, gprs(DST, SRC0), dep=DEP_MASK_12_17, retains=True),
    enc("derivative", 10, gprs(DST, SRC0), dep=DEP_MASK_12_17),

    # Integer ALU, compare-select, and bitwise logic.
    enc("int_add_extended", 10,
        [gpr(DST, widths=W16_32, align=1),
         imm_src(SRC0, widths=W16_32, align=1),
         imm_src(SRC1, widths=W16_32, align=1, flags=(ALLOC, UNIFORM_ALT))],
        dep=DEP_MASK_12_17, retains=True),
    enc("int_mad_extended", 12,
        [gpr(DST), reg_src(SRC0), imm_src(SRC1), imm_src(SRC2)],
        dep=DEP_MASK_12_17, retains=True),
    enc("int_mul_wide", 12,
        [gpr(DST, 94, widths=W64, align=4), reg_src(SRC0), imm_src(SRC1)],
        dep=DEP_MASK_12_17, retains=True),
    enc("minmax_compact", 6,
        [gpr(DST, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         reg_src(SRC0, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         imm_src(SRC1, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP)],
        dep=DEP_IDX_45_47, retains=True),
    enc("select_gpr_wide", 10, _SELECT_WIDE_OPS, dep=DEP_IDX_61_63, retains=True),
    enc("select_modifier_extended", 14, _SELECT_WIDE_OPS, dep=DEP_MASK_61_93),
    enc("logic_extended", 10,
        [gpr(DST, flags=_ALLOC_SCAT), reg_src(SRC0, flags=_ALLOC_SCAT), imm_src(SRC1, flags=_ALLOC_SCAT)],
        dep=DEP_MASK_45_61, retains=True),
    enc("logic_export", 10,
        [pub(DST, flags=(ALLOC, PUB, SCATTERED)), *gprs(SRC0, SRC1, flags=_ALLOC_SCAT)],
        dep=DEP_MASK_45_61, retains=True),
    enc("logic_uniform", 10, gprs(DST, SRC0, flags=_ALLOC_SCAT), dep=DEP_MASK_45_61),

    # Uniform-bank stores (independent of GPR bank).
    enc("store_uniform", 10, [uniform(DST), gpr(SRC0)], dep=DEP_MASK_12_17, retains=True),
    enc("store_uniform_add", 10, [uniform(DST), *gprs(SRC0, SRC1)], dep=DEP_MASK_12_17),
    enc("store_uniform_mad", 12, [uniform(DST), *src_gprs(3)], dep=DEP_MASK_12_17),
    enc("store_uniform_mul_wide", 12,
        [uniform(DST, UNIFORM_COUNT - 2, widths=W64, align=4), *gprs(SRC0, SRC1)],
        dep=DEP_MASK_12_17),

    # Bit shifts.
    enc("shift_extended", 10, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17, retains=True),
    enc("shift_arith_register", 10, [gpr(DST), reg_src(SRC0), reg_src(SRC1)],
        dep=DEP_MASK_12_17, retains=True),
    enc("shift_logical_immediate", 12, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17, retains=True),
    enc("shift_logical_register", 12, [gpr(DST), reg_src(SRC0), reg_src(SRC1)],
        dep=DEP_MASK_12_17, retains=True),

    # Predicate compares and execution-mask control flow.
    enc("predicate_compare_short", 6, [reg_src(SRC0), imm_src(SRC1)],
        dep=DEP_IDX_45_47, retains=True),
    enc("predicate_compare_extended", 10, [reg_src(SRC0), imm_src(SRC1)],
        dep=DEP_IDX_61_63, retains=True),
    enc("predicate_compare_loop", 10, gprs(SRC0, SRC1), dep=DEP_IDX_61_63),
    enc("exec_mask_push", 4, dep=DEP_MASK_12_17),
    enc("exec_mask_else", 4),
    enc("exec_mask_pop", 6),
    enc("loop_mask_push", 4),
    enc("loop_mask_update", 4),
    enc("loop_mask_pop", 6),
    enc("jmp_exec_any", 10),
    enc("jmp_exec_none", 10),
    enc("break_mask_unwind", 6),

    # Fragment/graphics system values and publication exports.
    enc("get_draw_id", 4, [gpr(DST, 63, flags=(ALLOC, HARD_LOW))], fixed_slot=1),
    enc("get_coverage", 4, [gpr(DST, 63, flags=(ALLOC, HARD_LOW))], fixed_slot=1),
    enc("float2_export", 8, [pub(DST), *gprs(SRC0, SRC1)], dep=DEP_MASK_45_61, retains=True),

    # Image writes and block image stores.
    enc("block_store_params", 12, [gpr(DST, 15, align=8), *src_gprs(3)]),
    enc("block_store_extended_params", 16, [gpr(DST, 15, align=8), *src_gprs(4)]),
    enc("image_write_params", 32, [gpr(DST, 47, align=16), *src_gprs(8)]),
    enc("image_write_params_16", 16, [gpr(DST, 47, align=16), *src_gprs(4)]),
    # Compact 2D 32-bit tuple: publishes {x, y, r, g, b, a} omitting unused cr4/cr5.
    enc("image_write_params_2d", 24, [gpr(DST, 47, align=16), *src_gprs(6)]),
    # Asynchronous PBE write: waits on slot 1 for params and produces completion on slot 1..6.
    enc("image_write", 16, _IMAGE_WRITE_OPS),
    enc("image_write_layered", 16, _IMAGE_WRITE_OPS),
    enc("block_image_store", 18, [gpr(SRC0, 15, align=8), *gprs(SRC1, SRC2, max_idx=15)]),
    enc("block_image_store_extended", 18, [gpr(SRC0, 15, align=8), *gprs(SRC1, SRC2, SRC3, max_idx=15)]),

    # Texture coordinate packing and sampling.
    enc("texture_coords", 16, [gpr(DST, 31, align=4), *src_gprs(3)]),
    enc("texture_coords_mov", 8, [gpr(DST, 31, align=4), *src_gprs(2)]),
    enc("texture_lod_params", 12, [gpr(DST, 31, align=8), *src_gprs(3)]),
    enc("texture_volume_params", 16, [gpr(DST, 31, align=8), *src_gprs(4)]),
    enc("texture_lod", 14, [gpr(DST, 63), gpr(SRC0, 31, align=8), *gprs(SRC1, SRC2, SRC3, max_idx=31)]),
    enc("texture_grad_params", 24, [gpr(DST, 31, align=8), *src_gprs(6)]),
    enc("texture_grad_array_params", 32, [gpr(DST, 31, align=8), *src_gprs(8)], ev=BYTE_DIFF),
    enc("texture_grad_volume_params", 36, [gpr(DST, 31, align=8), *src_gprs(9)], ev=BYTE_DIFF),
    enc("texture_grad_volume_min_lod_params", 40, [gpr(DST, 31, align=8), *src_gprs(10)], ev=BYTE_DIFF),
    enc("texture_grad", 14, [gpr(DST, 63), gpr(SRC0, 31, align=8)]),
    enc("texture_sample", 14, [gpr(DST, 63), gpr(SRC0, 31, align=4), gpr(SRC1, 31)]),

    # Interpolation, varying/depth/coverage exports, and tilebuffer ops.
    # Sample-index input is overwritten; live values require a disposable copy.
    enc("interpolation_position", 8,
        [gpr(DST), gpr(SRC0, align=4, flags=_ALLOC_CLOB, ev=HW)],
        ev=BYTE_DIFF, dep=DEP_MASK_12_17),
    enc("iter_coord", 10, [gpr(DST), gpr(SRC0, align=1)], ev=BYTE_DIFF),
    enc("iter", 10, [gpr(DST)]),
    enc("iter_flat", 6, [gpr(DST, 93)]),
    enc("vary_store", 8, [pub(SRC0)], dep=DEP_MASK_12_17),
    enc("object_grid_export", 8, [pub(SRC0)]),
    # Waits via DEP_MASK_12_17 for coverage publication rather than draining before tile access.
    enc("coverage", 6, [gpr(SRC0, 63)], dep=DEP_MASK_12_17, prior=True),
    enc("depth_store", 6, [pub(SRC0, 62, align=4), pub(SRC1, 63)], prior=True),
    enc("publication_pair", 20, [pub(DST, 62, align=4), *gprs(SRC0, SRC1)]),
    # Tile access opens with publications, memory loads, and texture loads in flight.
    enc("tile_access", 6, prior=True, exempt=(EXEMPT_PUB, EXEMPT_MEM, EXEMPT_TEX)),
    enc("tile_load", 12, [gpr(DST)]),
    enc("tile_store", 12, [gpr(SRC0)], dep=DEP_MASK_12_17),
    enc("tile_load_mask", 12, gprs(DST, SRC0), ev=BYTE_DIFF),
    enc("tile_store_mask", 12, gprs(SRC0, SRC1), ev=BYTE_DIFF, dep=DEP_MASK_12_17),
    enc("tile_fence", 6, prior=True),

    # Device memory and uniform-to-GPR moves.
    enc("device_load", 14, [gpr(DST, widths=W16_32_64, align=1), gpr(INDEX)],
        dep=DEP_MASK_12_17, retains=True),
    enc("device_load_indirect", 14,
        [gpr(DST, widths=W16_32_64, align=1), gpr(INDEX),
         gpr(SRC1, 94, ev=BYTE_DIFF), gpr(SRC2, ev=BYTE_DIFF)],
        dep=DEP_MASK_12_17, retains=True),
    # Pointer-pair load without an index operand (pair + byte displacement).
    enc("device_load_pointer", 14,
        [gpr(DST, widths=W16_32_64, align=1, ev=HW), gpr(SRC1, 94), gpr(SRC2)],
        ev=BYTE_DIFF, dep=DEP_MASK_12_17, retains=True),
    # Four-byte uniform-to-GPR move; destination field reaches r63.
    enc("uniform_move", 4, [gpr(DST, 63, flags=_ALLOC_SCAT)], ev=BYTE_DIFF),
    enc("device_store", 14, [gpr(INDEX), gpr(STORE_DATA, widths=W16_32_64, align=1)],
        dep=DEP_MASK_12_17),
    enc("device_atomic", 14, gprs(INDEX, ATOMIC_DATA, flags=_ALLOC_SCAT_CLOB),
        dep=DEP_MASK_12_17),
    enc("device_atomic_indirect", 14,
        [*gprs(INDEX, ATOMIC_DATA, flags=_ALLOC_SCAT_CLOB),
         gpr(SRC1, 94, flags=_ALLOC_CLOB), gpr(SRC2, flags=_ALLOC_CLOB)],
        dep=DEP_MASK_12_17),

    # Subgroup ops and cooperative matrix (MMA).
    enc("subgroup_scan_iadd", 8, gprs(DST, SRC0, max_idx=63)),
    enc("subgroup_reduce_umax", 8, gprs(DST, SRC0, max_idx=63)),
    enc("subgroup_broadcast", 10, gprs(DST, SRC0, SRC1, max_idx=63)),
    enc("subgroup_ballot", 10, [gpr(DST, 63), gpr(SRC0, 63, flags=_ALLOC_CLOB)], ev=BYTE_DIFF),
    enc("subgroup_reduce", 8, gprs(DST, SRC0), dep=DEP_MASK_12_17, retains=True),
    enc("subgroup_shuffle", 10, [gpr(DST), gpr(SRC0), gpr_imm(SRC1, 63)],
        dep=DEP_MASK_12_17, retains=True),
    enc("subgroup_rotate", 12, [*gprs(DST, SRC0, SRC1), gpr_imm(SRC2, 63)],
        dep=DEP_MASK_12_17, retains=True),
    enc("mma_f32", 12, [gpr(DST, 94), *src_gprs(3)], dep=DEP_MASK_12_17, retains=True),
    enc("mma_f16", 12, [gpr(DST), *src_gprs(3)], dep=DEP_MASK_12_17, retains=True),

    # Bit-extract, snorm packing, and narrow conversions.
    enc("extr", 12, [*gprs(DST, SRC0, SRC1), gpr_imm(SRC2)], dep=DEP_MASK_12_17, retains=True),
    enc("pack_snorm_2x16", 10, gprs(DST, SRC0, SRC1), dep=DEP_MASK_12_17, retains=True),
    enc("pack_snorm_4x8", 20, [gpr(DST), *src_gprs(4)], split_result=True),
    enc("int_to_half", 8, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17, retains=True),
    enc("float_to_narrow_int", 10, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17, retains=True),

    # Synchronization, threadgroup/local/private/spill memory, and waits.
    # Barriers and device fences may issue with memory loads/atomics outstanding.
    enc("workgroup_barrier", 6, prior=True, exempt=EXEMPT_MEM),
    enc("device_fence", 6, ev=BYTE_DIFF, prior=True, exempt=EXEMPT_MEM),
    enc("halt", 4, prior=True),
    enc("shared_load", 14, gprs(DST, SRC0), dep=DEP_MASK_12_17, retains=True),
    enc("shared_atomic", 12, [gpr(DST), *gprs(ATOMIC_DATA, INDEX, flags=_ALLOC_CLOB)]),
    enc("shared_store", 14, [gpr(SRC0, flags=_ALLOC_CLOB), gpr(SRC1)], dep=DEP_MASK_12_17),
    enc("local_store", 14, [gpr(SRC0, 63, flags=_ALLOC_CLOB)]),
    enc("local_store_indexed", 14, [gpr(SRC0, 63, flags=_ALLOC_CLOB), gpr(SRC1, 63)]),
    enc("private_load", 12, gprs(DST, SRC0, max_idx=63), dep=DEP_MASK_12_17, retains=True),
    enc("private_store", 10, [gpr(SRC0, 63, flags=_ALLOC_CLOB), gpr(SRC1, 63)], dep=DEP_MASK_12_17),
    enc("device_atomic_result", 8, [gpr(SRC0)]),
    enc("spill_store", 10, [gpr(SRC0, flags=_ALLOC_SCAT)], dep=DEP_MASK_45_61),
    enc("spill_load", 8, [gpr(DST, flags=_ALLOC_SCAT)], dep=DEP_MASK_45_61),
    enc("device_store_indirect", 14,
        [gpr(INDEX), gpr(STORE_DATA, widths=W16_32_64, align=1), gpr(SRC1, 94), gpr(SRC2)],
        dep=DEP_MASK_12_17),
    enc("tile_load_coords", 12, gprs(DST, SRC0)),
    # Standalone scoreboard wait (byte 0 = 0x06, bits 10..15 = 6-slot mask).
    enc("wait", 2),

    # Uniform-capable SFU, native mesh, object calls, and ABI return.
    enc("float_special_uniform", 10, [gpr(DST), reg_src(SRC0)], dep=DEP_MASK_12_17, retains=True),
    # T8132 r6: implicit producer completing on slot 1 (or alternate slot 2); clobbers src0 halfword.
    enc("mesh_control", 6,
        [gpr(SRC0, widths=W16, flags=_ALLOC_CLOB, ev=HW), gpr(SRC1, widths=W16, ev=HW)],
        ev=BYTE_DIFF, dep=DEP_MASK_12_17, fixed_slot=1, fixed_alt=0x02),
    enc("mesh_submit_scalar", 6,
        [gpr(SRC0, 94, flags=_ALLOC_CLOB), gpr(SRC1, flags=_ALLOC_CLOB, ev=HW)],
        ev=BYTE_DIFF, dep=DEP_MASK_12_17, prior=True),
    enc("mesh_submit_channel", 6,
        [gpr(SRC0, 94, flags=_ALLOC_CLOB), gpr(SRC1, flags=_ALLOC_CLOB)],
        ev=BYTE_DIFF, dep=DEP_MASK_12_17, prior=True),
    enc("object_call", 10, prior=True),
    enc("object_return", 4, prior=True),
    enc("object_resume", 6, prior=True),
    enc("abi_return", 4, prior=True),
)


def validate_encodings(encodings: tuple[Encoding, ...]) -> None:
    if not encodings:
        raise ValueError("ENCODINGS must not be empty")

    seen_names: set[str] = set()
    seen_enums: set[str] = set()

    for e in encodings:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", e.name):
            raise ValueError(f"invalid encoding name: {e.name!r}")
        if not re.fullmatch(r"AGX_APPLE9_ENC_[A-Z0-9_]+", e.enum_name):
            raise ValueError(f"invalid enum name: {e.enum_name!r}")
        if e.name in seen_names:
            raise ValueError(f"duplicate encoding name: {e.name!r}")
        if e.enum_name in seen_enums:
            raise ValueError(f"duplicate enum name: {e.enum_name!r}")
        seen_names.add(e.name)
        seen_enums.add(e.enum_name)

        if e.length <= 0 or e.length > 255 or (e.length % 2) != 0:
            raise ValueError(f"{e.name}: invalid length {e.length}")
        if len(e.operands) > MAX_ENCODING_OPERANDS:
            raise ValueError(f"{e.name}: {len(e.operands)} operands exceeds max {MAX_ENCODING_OPERANDS}")
        if e.evidence not in VALID_EVIDENCE:
            raise ValueError(f"{e.name}: invalid evidence {e.evidence!r}")
        if e.dependency_layout not in VALID_DEPS:
            raise ValueError(f"{e.name}: invalid dependency_layout {e.dependency_layout!r}")
        if e.dependency_retains_sources and e.dependency_layout == DEP_NONE:
            raise ValueError(f"{e.name}: dependency_retains_sources requires dependency_layout")
        if not (0 <= e.fixed_producer_slot <= 6):
            raise ValueError(f"{e.name}: invalid fixed_producer_slot {e.fixed_producer_slot}")
        if e.fixed_producer_alternates:
            if e.fixed_producer_slot == 0:
                raise ValueError(f"{e.name}: fixed_producer_alternates requires fixed_producer_slot")
            if e.fixed_producer_alternates < 0 or e.fixed_producer_alternates > 0x3F:
                raise ValueError(f"{e.name}: invalid fixed_producer_alternates {e.fixed_producer_alternates:#x}")
        if e.prior_completion_exempt:
            if not e.requires_prior_completion:
                raise ValueError(f"{e.name}: prior_completion_exempt requires requires_prior_completion")
            for ex in e.prior_completion_exempt:
                if ex not in VALID_EXEMPTIONS:
                    raise ValueError(f"{e.name}: invalid exemption {ex!r}")

        seen_roles: set[str] = set()
        for idx, op in enumerate(e.operands):
            if op.role not in VALID_ROLES:
                raise ValueError(f"{e.name}[{idx}]: invalid role {op.role!r}")
            if op.role in seen_roles:
                raise ValueError(f"{e.name}[{idx}]: duplicate operand role {op.role!r}")
            seen_roles.add(op.role)

            if not op.files or len(op.files) != len(set(op.files)) or any(f not in VALID_FILES for f in op.files):
                raise ValueError(f"{e.name}[{idx}]: invalid files {op.files!r}")
            file_set = frozenset(op.files)
            if file_set not in VALID_FILE_COMBOS:
                raise ValueError(f"{e.name}[{idx}]: unsupported file combination {op.files!r}")
            if not op.widths or len(op.widths) != len(set(op.widths)) or any(w not in VALID_WIDTHS for w in op.widths):
                raise ValueError(f"{e.name}[{idx}]: invalid widths {op.widths!r}")
            if len(op.flags) != len(set(op.flags)) or any(fl not in VALID_FLAGS for fl in op.flags):
                raise ValueError(f"{e.name}[{idx}]: invalid flags {op.flags!r}")
            if op.evidence not in VALID_EVIDENCE:
                raise ValueError(f"{e.name}[{idx}]: invalid evidence {op.evidence!r}")
            if op.alignment_halves not in (1, 2, 4, 8, 16):
                raise ValueError(f"{e.name}[{idx}]: invalid alignment_halves {op.alignment_halves}")
            if not (0 <= op.min_index <= op.max_index <= 0xFF):
                raise ValueError(f"{e.name}[{idx}]: invalid index range [{op.min_index}, {op.max_index}]")
            if not (0 <= op.uniform_max_index <= UNIFORM_COUNT - 1):
                raise ValueError(f"{e.name}[{idx}]: invalid uniform_max_index {op.uniform_max_index}")

            if file_set == frozenset({F_IMPLICIT}):
                if op.min_index != 0xFF or op.max_index != 0xFF:
                    raise ValueError(f"{e.name}[{idx}]: implicit operand must use 0xff bounds")
                if IMPLICIT_FLAG not in op.flags or PUB in op.flags:
                    raise ValueError(f"{e.name}[{idx}]: implicit operand has invalid flags {op.flags!r}")
                if op.uniform_max_index != 0:
                    raise ValueError(f"{e.name}[{idx}]: implicit operand must not set uniform_max_index")
            elif file_set == frozenset({F_UNIFORM}):
                if op.max_index >= UNIFORM_COUNT:
                    raise ValueError(f"{e.name}[{idx}]: uniform max_index {op.max_index} >= {UNIFORM_COUNT}")
                if IMPLICIT_FLAG in op.flags or PUB in op.flags:
                    raise ValueError(f"{e.name}[{idx}]: uniform operand has invalid flags {op.flags!r}")
                if op.uniform_max_index != 0:
                    raise ValueError(f"{e.name}[{idx}]: uniform-only operand should not set uniform_max_index")
            else:
                if IMPLICIT_FLAG in op.flags:
                    raise ValueError(f"{e.name}[{idx}]: non-implicit operand must not set IMPLICIT flag")
                if PUB in op.flags and file_set != frozenset({F_GPR}):
                    raise ValueError(f"{e.name}[{idx}]: publication operand must use GPR file only")
                limit = PUBLICATION_COUNT if PUB in op.flags else GPR_COUNT
                if op.max_index >= limit:
                    raise ValueError(f"{e.name}[{idx}]: GPR max_index {op.max_index} >= {limit}")
                if F_UNIFORM not in file_set and op.uniform_max_index != 0:
                    raise ValueError(f"{e.name}[{idx}]: non-uniform operand must not set uniform_max_index")


def _or_expr(items: tuple[str, ...], zero: str = "0") -> str:
    return " | ".join(items) if items else zero


def render_header(encodings: tuple[Encoding, ...]) -> str:
    return (
        "/*\n"
        " * Copyright 2026 The Asahi Linux Contributors\n"
        " * SPDX-License-Identifier: MIT\n"
        " *\n"
        " * Generated by agx_apple9_isa.py -- DO NOT EDIT.\n"
        " */\n"
        "\n"
        "#ifndef AGX_APPLE9_MACHINE_TABLE_H\n"
        "#define AGX_APPLE9_MACHINE_TABLE_H\n"
        "\n"
        "#include \"agx_apple9_machine.h\"\n"
        "\n"
        f"#define AGX_APPLE9_ISA_TABLE_COUNT {len(encodings)}u\n"
        "\n"
        "extern const struct agx_apple9_encoding_info\n"
        "   agx_apple9_encodings[AGX_APPLE9_ENC_COUNT];\n"
        "\n"
        "#endif\n"
    )


def _render_operand(op: Operand) -> str:
    fields = [
        f".role = {op.role}",
        f".files = {_or_expr(op.files, 'AGX_APPLE9_FILE_NONE')}",
        f".widths = {_or_expr(op.widths)}",
    ]
    if op.min_index:
        fields.append(f".min_index = {op.min_index}")
    fields.append(f".max_index = {op.max_index}")
    if op.uniform_max_index:
        fields.append(f".uniform_max_index = {op.uniform_max_index}")
    fields.append(f".alignment_halves = {op.alignment_halves}")
    if op.flags:
        fields.append(f".flags = {_or_expr(op.flags)}")
    fields.append(f".evidence = {op.evidence}")
    return "{ " + ", ".join(fields) + " }"


def render_source(encodings: tuple[Encoding, ...]) -> str:
    lines = [
        "/*",
        " * Copyright 2026 The Asahi Linux Contributors",
        " * SPDX-License-Identifier: MIT",
        " *",
        " * Generated by agx_apple9_isa.py -- DO NOT EDIT.",
        " */",
        "",
        '#include "agx_apple9_machine_table.h"',
        "",
        "#include <assert.h>",
        "",
        "static_assert(AGX_APPLE9_ISA_TABLE_COUNT == AGX_APPLE9_ENC_COUNT,",
        '              "agx_apple9_isa.py and enum agx_apple9_encoding count mismatch");',
        f"static_assert(AGX_APPLE9_GPR_COUNT == {GPR_COUNT},",
        '              "agx_apple9_isa.py GPR_COUNT mismatch");',
        f"static_assert(AGX_APPLE9_UNIFORM_COUNT == {UNIFORM_COUNT},",
        '              "agx_apple9_isa.py UNIFORM_COUNT mismatch");',
        f"static_assert(AGX_APPLE9_PUBLICATION_COUNT == {PUBLICATION_COUNT},",
        '              "agx_apple9_isa.py PUBLICATION_COUNT mismatch");',
        f"static_assert(AGX_APPLE9_MAX_ENCODING_OPERANDS == {MAX_ENCODING_OPERANDS},",
        '              "agx_apple9_isa.py MAX_ENCODING_OPERANDS mismatch");',
        "",
        *[
            f'static_assert({e.enum_name} == {idx}u, "{e.enum_name} enum index mismatch");'
            for idx, e in enumerate(encodings)
        ],
        "",
        "const struct agx_apple9_encoding_info agx_apple9_encodings[AGX_APPLE9_ENC_COUNT] = {",
    ]

    for e in encodings:
        lines.append(f"   [{e.enum_name}] = {{")
        lines.append(f'      .name = "{e.name}",')
        lines.append(f"      .length = {e.length},")
        if e.operands:
            lines.append(f"      .operand_count = {len(e.operands)},")
        if e.allocator_safe:
            lines.append("      .allocator_safe = true,")
        lines.append(f"      .evidence = {e.evidence},")
        if e.dependency_layout != DEP_NONE:
            lines.append(f"      .dependency_layout = {e.dependency_layout},")
        if e.dependency_retains_sources:
            lines.append("      .dependency_retains_sources = true,")
        if e.fixed_producer_slot:
            lines.append(f"      .fixed_producer_slot = {e.fixed_producer_slot},")
        if e.fixed_producer_alternates:
            lines.append(f"      .fixed_producer_alternates = {e.fixed_producer_alternates:#x},")
        if e.requires_prior_completion:
            lines.append("      .requires_prior_completion = true,")
        if e.prior_completion_exempt:
            lines.append(f"      .prior_completion_exempt = {_or_expr(e.prior_completion_exempt)},")
        if e.split_result:
            lines.append("      .split_result = true,")
        if e.operands:
            lines.append("      .operands = {")
            for op in e.operands:
                lines.append(f"         {_render_operand(op)},")
            lines.append("      },")
        lines.append("   },")

    lines.extend([
        "};",
        "",
        "static_assert((sizeof(agx_apple9_encodings) / sizeof(agx_apple9_encodings[0])) ==",
        "                 AGX_APPLE9_ENC_COUNT,",
        '              "every Apple9 encoding needs a machine-model entry");',
        "",
    ])
    return "\n".join(lines)


def check_against_header(encodings: tuple[Encoding, ...], header_path: Path) -> None:
    text = header_path.read_text(encoding="utf-8")
    m = re.search(r"enum\s+agx_apple9_encoding\s*\{(.*?)\};", text, flags=re.S)
    if not m:
        raise ValueError(f"could not find enum agx_apple9_encoding in {header_path}")
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    header_enums: list[str] = []
    for raw in body.split(","):
        tok = raw.strip()
        if not tok:
            continue
        has_assign = "=" in tok
        ident = tok.split("=", 1)[0].strip()
        if ident == "AGX_APPLE9_ENC_PSEUDO":
            continue
        if has_assign:
            raise ValueError(
                f"unexpected explicit enum value in {header_path}: {tok!r}"
            )
        if ident == "AGX_APPLE9_ENC_COUNT":
            continue
        header_enums.append(ident)

    table_enums = [e.enum_name for e in encodings]
    if header_enums != table_enums:
        missing = sorted(set(header_enums) - set(table_enums))
        extra = sorted(set(table_enums) - set(header_enums))
        raise ValueError(
            f"enum agx_apple9_encoding mismatch vs {header_path}: "
            f"header={len(header_enums)}, table={len(table_enums)}, "
            f"missing={missing}, extra={extra}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-h", type=Path, help="Output generated C header path")
    parser.add_argument("--out-c", type=Path, help="Output generated C source path")
    parser.add_argument(
        "--check",
        type=Path,
        metavar="MACHINE_H",
        help="Validate schema and check enum sync against agx_apple9_machine.h",
    )
    args = parser.parse_args(argv)

    validate_encodings(ENCODINGS)

    if args.check is not None:
        check_against_header(ENCODINGS, args.check)

    if args.out_h is not None:
        args.out_h.parent.mkdir(parents=True, exist_ok=True)
        args.out_h.write_text(render_header(ENCODINGS), encoding="utf-8", newline="\n")

    if args.out_c is not None:
        args.out_c.parent.mkdir(parents=True, exist_ok=True)
        args.out_c.write_text(render_source(ENCODINGS), encoding="utf-8", newline="\n")

    if args.check is None and args.out_h is None and args.out_c is None:
        parser.error("specify at least one of --out-h, --out-c, or --check")

    return 0


if __name__ == "__main__":
    sys.exit(main())
