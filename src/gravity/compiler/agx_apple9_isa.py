#!/usr/bin/env python3
# Copyright 2026 The Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Declarative Apple9 ISA tables and C table generator.

Generates:
  - `agx_apple9_machine_table.{h,c}`: machine encoding descriptors and per-slot
    operand legality for every `enum agx_apple9_encoding` form.
  - `agx_apple9_vir_table.{h,c}`: VIR opcode metadata (kind, family, arity,
    property bitmasks, per-slot inline/half legality), parameterized family
    tables (fixed bytes, conversions, SFU, minmax, bit-unary, subgroup,
    barriers/fences), and the unified checked texture-mode table.

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
    uniform_src_mask: int
    immediate_src_mask: int
    half_src_mask: int
    half_src_hi_mask: int
    half_src_sext_mask: int
    wide_float_imm_mask: int
    integer_immediates: bool
    compare_literal_src1: bool


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
    half_srcs: int = 0,
    half_hi: int = 0,
    half_sext: int = 0,
    wide_float_imm: int = 0,
    int_imms: bool = False,
    cmp_lit: bool = False,
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
    uniform_src_mask = 0
    immediate_src_mask = 0
    for op in resolved_ops:
        if op.role in SRC_ROLES[:4]:
            slot = SRC_ROLES.index(op.role)
            if F_UNIFORM in op.files:
                uniform_src_mask |= 1 << slot
            if F_IMM in op.files:
                immediate_src_mask |= 1 << slot
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
        uniform_src_mask=uniform_src_mask,
        immediate_src_mask=immediate_src_mask,
        half_src_mask=half_srcs,
        half_src_hi_mask=half_hi,
        half_src_sext_mask=half_sext,
        wide_float_imm_mask=wide_float_imm,
        integer_immediates=int_imms,
        compare_literal_src1=cmp_lit,
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
        dep=DEP_IDX_45_47, retains=True, half_srcs=0x3, half_hi=0x3),
    enc("float2_base", 4, gprs(DST, SRC0, SRC1, max_idx=63), half_srcs=0x3),
    enc("float2_project", 8, gprs(DST, SRC0, SRC1, max_idx=63, align=1, flags=_ALLOC_SCAT_COMP)),
    enc("float2_modifier_extended", 8, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True,
        half_srcs=0x3, half_hi=0x3),
    enc("float2_abs_extended", 10, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True,
        half_srcs=0x3, half_hi=0x3),
    enc("float2_mul_abs_extended", 12, _FLOAT2_EXT_OPS, dep=DEP_MASK_45_61, retains=True,
        half_srcs=0x3, half_hi=0x3),
    enc("float3_extended", 8, _FLOAT3_EXT_OPS, dep=DEP_IDX_61_63, retains=True,
        half_srcs=0x6, half_hi=0x4, wide_float_imm=0x4),
    enc("float3_compact", 6,
        [gpr(DST, 63), gpr(SRC0, 63), imm_src(SRC1, 63, umax=63), gpr(SRC2, 63)],
        wide_float_imm=0x4),
    enc("float3_saturate_extended", 10, _FLOAT3_EXT_OPS, dep=DEP_MASK_61_77, retains=True,
        half_srcs=0x6, half_hi=0x4, wide_float_imm=0x4),
    enc("float3_modifier_extended", 12, _FLOAT3_EXT_OPS, dep=DEP_MASK_61_77, retains=True,
        half_srcs=0x6, half_hi=0x4, wide_float_imm=0x4),
    enc("cube", 12, gprs(DST, SRC0, SRC1, SRC2)),
    enc("float_special", 10, gprs(DST, SRC0), dep=DEP_MASK_12_17, retains=True),
    enc("derivative", 10, gprs(DST, SRC0), dep=DEP_MASK_12_17),

    # Integer ALU, compare-select, and bitwise logic.
    enc("int_add_extended", 10,
        [gpr(DST, widths=W16_32, align=1),
         imm_src(SRC0, widths=W16_32, align=1),
         imm_src(SRC1, widths=W16_32, align=1, flags=(ALLOC, UNIFORM_ALT))],
        dep=DEP_MASK_12_17, retains=True, half_srcs=0x3, half_hi=0x3, half_sext=0x3,
        int_imms=True),
    enc("int_mad_extended", 12,
        [gpr(DST), reg_src(SRC0), imm_src(SRC1), imm_src(SRC2)],
        dep=DEP_MASK_12_17, retains=True, half_srcs=0x7, half_hi=0x7, half_sext=0x1,
        int_imms=True),
    enc("int_mul_wide", 12,
        [gpr(DST, 94, widths=W64, align=4), reg_src(SRC0), imm_src(SRC1)],
        dep=DEP_MASK_12_17, retains=True, int_imms=True),
    enc("minmax_compact", 6,
        [gpr(DST, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         reg_src(SRC0, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP),
         imm_src(SRC1, widths=W16_32, align=1, flags=_ALLOC_SCAT_COMP)],
        dep=DEP_IDX_45_47, retains=True, wide_float_imm=0x2),
    enc("select_gpr_wide", 10, _SELECT_WIDE_OPS, dep=DEP_IDX_61_63, retains=True,
        int_imms=True, cmp_lit=True),
    enc("select_modifier_extended", 14, _SELECT_WIDE_OPS, dep=DEP_MASK_61_93,
        int_imms=True, cmp_lit=True),
    enc("logic_extended", 10,
        [gpr(DST, flags=_ALLOC_SCAT), reg_src(SRC0, flags=_ALLOC_SCAT), imm_src(SRC1, flags=_ALLOC_SCAT)],
        dep=DEP_MASK_45_61, retains=True, int_imms=True),
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
        dep=DEP_IDX_45_47, retains=True, cmp_lit=True),
    enc("predicate_compare_extended", 10, [reg_src(SRC0), imm_src(SRC1)],
        dep=DEP_IDX_61_63, retains=True, cmp_lit=True),
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
        ev=BYTE_DIFF, dep=DEP_MASK_12_17, fixed_slot=1, fixed_alt=0x02,
        half_srcs=0x3, half_hi=0x3),
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
        for field_name, mask in (
            ("uniform_src_mask", e.uniform_src_mask),
            ("immediate_src_mask", e.immediate_src_mask),
            ("half_src_mask", e.half_src_mask),
            ("half_src_hi_mask", e.half_src_hi_mask),
            ("half_src_sext_mask", e.half_src_sext_mask),
            ("wide_float_imm_mask", e.wide_float_imm_mask),
        ):
            if mask & ~0xF:
                raise ValueError(f"{e.name}: {field_name} {mask:#x} exceeds 4-slot range")
        if e.half_src_hi_mask & ~e.half_src_mask:
            raise ValueError(f"{e.name}: half_src_hi_mask must be a subset of half_src_mask")
        if e.half_src_sext_mask & ~e.half_src_mask:
            raise ValueError(f"{e.name}: half_src_sext_mask must be a subset of half_src_mask")
        if e.compare_literal_src1 and not (e.immediate_src_mask & 0x2):
            raise ValueError(f"{e.name}: compare_literal_src1 requires SRC1 immediate support")

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
        if e.uniform_src_mask:
            lines.append(f"      .uniform_src_mask = {e.uniform_src_mask:#x},")
        if e.immediate_src_mask:
            lines.append(f"      .immediate_src_mask = {e.immediate_src_mask:#x},")
        if e.half_src_mask:
            lines.append(f"      .half_src_mask = {e.half_src_mask:#x},")
        if e.half_src_hi_mask:
            lines.append(f"      .half_src_hi_mask = {e.half_src_hi_mask:#x},")
        if e.half_src_sext_mask:
            lines.append(f"      .half_src_sext_mask = {e.half_src_sext_mask:#x},")
        if e.wide_float_imm_mask:
            lines.append(f"      .wide_float_imm_mask = {e.wide_float_imm_mask:#x},")
        if e.integer_immediates:
            lines.append("      .integer_immediates = true,")
        if e.compare_literal_src1:
            lines.append("      .compare_literal_src1 = true,")
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


# ---------------------------------------------------------------------------
# Phase 2 & 3: VIR opcode table, parameterized families & texture-mode map
# ---------------------------------------------------------------------------

K_PSEUDO = "AGX_APPLE9_VIR_KIND_PSEUDO"
K_FIXED = "AGX_APPLE9_VIR_KIND_FIXED_FORM"
K_FAMILY = "AGX_APPLE9_VIR_KIND_FAMILY"
K_CUSTOM = "AGX_APPLE9_VIR_KIND_CUSTOM"
VALID_KINDS = {K_PSEUDO, K_FIXED, K_FAMILY, K_CUSTOM}

FAM_NONE = "AGX_APPLE9_VIR_FAMILY_NONE"
FAM_FIXED_BYTES = "AGX_APPLE9_VIR_FAMILY_FIXED_BYTES"
FAM_I2F = "AGX_APPLE9_VIR_FAMILY_I2F"
FAM_F2I = "AGX_APPLE9_VIR_FAMILY_F2I"
FAM_FSPECIAL = "AGX_APPLE9_VIR_FAMILY_FSPECIAL"
FAM_MINMAX = "AGX_APPLE9_VIR_FAMILY_MINMAX"
FAM_BIT_UNARY = "AGX_APPLE9_VIR_FAMILY_BIT_UNARY"
FAM_HALF_ALU = "AGX_APPLE9_VIR_FAMILY_HALF_ALU"
FAM_FLOAT2 = "AGX_APPLE9_VIR_FAMILY_FLOAT2"
FAM_FLOAT2_IMM = "AGX_APPLE9_VIR_FAMILY_FLOAT2_IMM"
FAM_IADD = "AGX_APPLE9_VIR_FAMILY_IADD"
FAM_LOGIC = "AGX_APPLE9_VIR_FAMILY_LOGIC"
FAM_SHIFT = "AGX_APPLE9_VIR_FAMILY_SHIFT"
FAM_UNIFORM = "AGX_APPLE9_VIR_FAMILY_UNIFORM"
FAM_MESH = "AGX_APPLE9_VIR_FAMILY_MESH"
VALID_FAMILIES = {
    FAM_NONE, FAM_FIXED_BYTES, FAM_I2F, FAM_F2I, FAM_FSPECIAL, FAM_MINMAX,
    FAM_BIT_UNARY, FAM_HALF_ALU, FAM_FLOAT2, FAM_FLOAT2_IMM, FAM_IADD,
    FAM_LOGIC, FAM_SHIFT, FAM_UNIFORM, FAM_MESH,
}

P_PURE = "AGX_APPLE9_VIR_PROP_PURE_ALU"
P_PURE_MULTI = "AGX_APPLE9_VIR_PROP_PURE_ALU_MULTI_DEST"
P_SAT = "AGX_APPLE9_VIR_PROP_SATURATABLE"
P_SPEC_U = "AGX_APPLE9_VIR_PROP_SPECIAL_UNIFORM"
P_OPT_BAR = "AGX_APPLE9_VIR_PROP_OPT_BARRIER"
P_CTRL = "AGX_APPLE9_VIR_PROP_CONTROL_SIDE_EFFECT"
P_COMM = "AGX_APPLE9_VIR_PROP_COMMUTATIVE"
P_IADD_SAT = "AGX_APPLE9_VIR_PROP_INT_ADD_SAT"
P_OPT_U = "AGX_APPLE9_VIR_PROP_OPT_UNIFORM_INLINE"
P_DUAL_U = "AGX_APPLE9_VIR_PROP_INLINE_DUAL_UNIFORM"
P_FP16_HALF = "AGX_APPLE9_VIR_PROP_FP16_HALF_SRC"
P_INT_HALF = "AGX_APPLE9_VIR_PROP_INT_HALF_SRC"
P_NO_DEST = "AGX_APPLE9_VIR_PROP_NO_DEST"
VALID_PROPS = {
    P_PURE, P_PURE_MULTI, P_SAT, P_SPEC_U, P_OPT_BAR, P_CTRL, P_COMM,
    P_IADD_SAT, P_OPT_U, P_DUAL_U, P_FP16_HALF, P_INT_HALF, P_NO_DEST,
}

ARITY_VAR = 0xFF
TEXTURE_MODE_INVALID = 0xFFFF
TEX_DIM_COUNT = 7


@dataclass(frozen=True)
class FixedBytes:
    bytes: tuple[int, ...]


@dataclass(frozen=True)
class I2FParams:
    b6_type_bits: int
    b7: int


@dataclass(frozen=True)
class F2IParams:
    wide: bool
    b7: int
    b8: int


@dataclass(frozen=True)
class FSpecialParams:
    family: int
    function: int
    precision: int
    rounding: int
    reciprocal: bool
    is_half_op: bool
    wide_factor: bool


@dataclass(frozen=True)
class MinMaxParams:
    select: int
    floating: bool


@dataclass(frozen=True)
class BitUnaryParams:
    b0: int
    b1: int
    uniform_alt_fmt: bool


@dataclass(frozen=True)
class VirOpcode:
    name: str
    enum_name: str
    kind: str
    family: str
    fixed_encoding: str | None
    arity: int
    inline_uniform_mask: int
    inline_immediate_mask: int
    half_src_mask: int
    half_src_hi_mask: int
    half_src_sext_mask: int
    props: tuple[str, ...]
    payload: FixedBytes | I2FParams | F2IParams | FSpecialParams | MinMaxParams | BitUnaryParams | None = None


def vir(
    name: str,
    kind: str,
    *,
    fam: str = FAM_NONE,
    enc_name: str | None = None,
    arity: int = ARITY_VAR,
    u_mask: int = 0,
    i_mask: int = 0,
    half: int = 0,
    half_hi: int = 0,
    half_sext: int = 0,
    props: str | tuple[str, ...] = (),
    payload: FixedBytes | I2FParams | F2IParams | FSpecialParams | MinMaxParams | BitUnaryParams | None = None,
) -> VirOpcode:
    return VirOpcode(
        name=name,
        enum_name=f"AGX_APPLE9_VIR_{name}",
        kind=kind,
        family=fam,
        fixed_encoding=f"AGX_APPLE9_ENC_{enc_name.upper()}" if enc_name else None,
        arity=arity,
        inline_uniform_mask=u_mask,
        inline_immediate_mask=i_mask,
        half_src_mask=half,
        half_src_hi_mask=half_hi,
        half_src_sext_mask=half_sext,
        props=_as_tuple(props),
        payload=payload,
    )


def _fixed_bytes(name: str, enc_name: str, data: tuple[int, ...], *,
                 props: tuple[str, ...] = (P_CTRL, P_NO_DEST)) -> VirOpcode:
    return vir(name, K_FIXED, fam=FAM_FIXED_BYTES, enc_name=enc_name, arity=0,
               props=props, payload=FixedBytes(data))


def _i2f(name: str, enc_name: str, is_f32_dst: bool, is_32bit_src: bool,
         is_signed: bool, *, opt_u: bool = False) -> VirOpcode:
    props = (P_PURE, P_OPT_U) if opt_u else (P_PURE,)
    b6_type = (0x08 if is_32bit_src else 0x00) | (0x04 if is_f32_dst else 0x00)
    return vir(name, K_FAMILY, fam=FAM_I2F, enc_name=enc_name, arity=1,
               u_mask=0x1, props=props,
               payload=I2FParams(b6_type, 0x60 if is_signed else 0x20))


def _f2i(name: str, enc_name: str, wide: bool, b7: int, b8: int, *,
         opt_u: bool = False) -> VirOpcode:
    props = (P_PURE, P_OPT_U) if opt_u else (P_PURE,)
    return vir(name, K_FAMILY, fam=FAM_F2I, enc_name=enc_name, arity=1,
               u_mask=0x1, props=props, payload=F2IParams(wide, b7, b8))


def _fspec(name: str, *, half: bool, family: int = 0, function: int = 0,
           precision: int = 0x40, rounding: int = 0, reciprocal: bool = False,
           wide_factor: bool = False, sat: bool = False,
           spec_u: bool = False) -> VirOpcode:
    props_list = [P_PURE]
    if sat:
        props_list.append(P_SAT)
    if spec_u:
        props_list.extend((P_SPEC_U, P_OPT_U))
    enc_name = "half_special" if half else (None if spec_u else "float_special")
    return vir(name, K_FAMILY, fam=FAM_FSPECIAL, enc_name=enc_name, arity=1,
               u_mask=0x1 if spec_u else 0, props=tuple(props_list),
               payload=FSpecialParams(family, function, precision, rounding,
                                      reciprocal, half, wide_factor))


def _minmax(name: str, select: int, *, half: bool = False,
            floating: bool = False) -> VirOpcode:
    props_list = [P_PURE]
    if not half:
        props_list.append(P_OPT_U)
    if not floating:
        props_list.append(P_COMM)
    return vir(name, K_FAMILY, fam=FAM_MINMAX,
               enc_name="half_minmax" if half else "minmax_compact", arity=2,
               u_mask=0 if half else 0x3, i_mask=0 if half else 0x2,
               props=tuple(props_list),
               payload=MinMaxParams(select, floating))


# Ordered identically to `enum agx_apple9_vir_opcode` in agx_apple9_ir.h.
VIR_OPCODES: tuple[VirOpcode, ...] = (
    vir("IMM", K_CUSTOM, arity=0, props=P_PURE),
    vir("GET_GLOBAL_ID", K_FIXED, enc_name="get_sr", arity=0),
    vir("GET_SR", K_CUSTOM, arity=0),
    vir("DEVICE_LOAD", K_CUSTOM),
    vir("PUBLICATION_TUPLE", K_CUSTOM),
    vir("TEXTURE_SAMPLE", K_CUSTOM),
    vir("IMAGE_STORE", K_CUSTOM, props=P_NO_DEST),
    vir("BLOCK_IMAGE_STORE", K_CUSTOM, props=P_NO_DEST),

    # Bit-unary family.
    vir("BIT_COUNT", K_FAMILY, fam=FAM_BIT_UNARY, enc_name="bit_unary", arity=1,
        u_mask=0x1, props=(P_PURE, P_OPT_U), payload=BitUnaryParams(0x27, 0x05, True)),
    vir("UFIND_MSB", K_FAMILY, fam=FAM_BIT_UNARY, enc_name="bit_unary", arity=1,
        u_mask=0x1, props=(P_PURE, P_OPT_U), payload=BitUnaryParams(0xA7, 0x05, False)),
    vir("BIT_REVERSE", K_FAMILY, fam=FAM_BIT_UNARY, enc_name="bit_unary", arity=1,
        u_mask=0x1, props=(P_PURE, P_OPT_U), payload=BitUnaryParams(0xA7, 0x04, False)),

    # Integer/float conversions.
    _i2f("U2F32", "uint_to_float", True, True, False, opt_u=True),
    _i2f("I2F32", "sint_to_float", True, True, True, opt_u=True),
    _f2i("F2I32", "float_to_sint", True, 0x48, 0x03, opt_u=True),
    _f2i("F2U32", "float_to_uint", True, 0x08, 0x02, opt_u=True),
    _i2f("U2F16", "int_to_half", False, True, False),
    _i2f("I2F16", "int_to_half", False, True, True),
    _i2f("U16_2F16", "int_to_half", False, False, False),
    _i2f("I16_2F16", "int_to_half", False, False, True),
    _f2i("F2I16", "float_to_narrow_int", False, 0xC8, 0x02),
    _f2i("F2U16", "float_to_narrow_int", False, 0x08, 0x03),
    _f2i("F2I8", "float_to_narrow_int", False, 0x48, 0x02),
    _f2i("F2U8", "float_to_narrow_int", False, 0x88, 0x02),

    # Half/norm/format pack & unpack.
    vir("F2F16", K_CUSTOM, enc_name="float_to_half_zext", arity=1, props=P_PURE),
    vir("PACK_HALF_2X16", K_CUSTOM, enc_name="pack_half_2x16", arity=2, props=P_PURE),
    vir("PACK_UNORM_2X16", K_CUSTOM, enc_name="pack_unorm_2x16", arity=2, props=P_PURE),
    vir("PACK_UNORM_4X8", K_CUSTOM, enc_name="pack_unorm_4x8", arity=4, props=P_PURE),
    vir("PACK_SNORM_2X16", K_CUSTOM, enc_name="pack_snorm_2x16", arity=2, props=P_PURE),
    vir("PACK_SNORM_4X8", K_CUSTOM, enc_name="pack_snorm_4x8", arity=4, props=P_PURE),
    vir("PACK_FORMAT", K_CUSTOM, props=P_PURE),
    vir("UNPACK_HALF", K_CUSTOM, enc_name="half_to_float", arity=1, props=P_PURE),
    vir("UNPACK_NORM", K_CUSTOM, enc_name="unpack_norm", arity=1,
        props=(P_PURE, P_PURE_MULTI)),

    # Integer ALU, logic, uniforms, shifts, extr, minmax.
    vir("IADD", K_FAMILY, fam=FAM_IADD, enc_name="int_add_extended", arity=2,
        u_mask=0x3, i_mask=0x3, half=0x3, half_hi=0x3, half_sext=0x3,
        props=(P_PURE, P_COMM, P_IADD_SAT, P_OPT_U, P_DUAL_U, P_INT_HALF)),
    vir("ISUB", K_FAMILY, fam=FAM_IADD, enc_name="int_add_extended", arity=2,
        u_mask=0x3, i_mask=0x3, half=0x3, half_hi=0x3, half_sext=0x3,
        props=(P_PURE, P_IADD_SAT, P_OPT_U, P_DUAL_U, P_INT_HALF)),
    vir("IMAD", K_CUSTOM, enc_name="int_mad_extended", arity=3,
        u_mask=0x7, i_mask=0x6, half=0x7, half_hi=0x7, half_sext=0x1,
        props=(P_PURE, P_COMM, P_OPT_U, P_DUAL_U, P_INT_HALF)),
    vir("IMUL_WIDE", K_CUSTOM, enc_name="int_mul_wide", arity=2,
        u_mask=0x3, i_mask=0x2,
        props=(P_PURE, P_PURE_MULTI, P_COMM, P_OPT_U, P_DUAL_U)),
    vir("IAND", K_FAMILY, fam=FAM_LOGIC, arity=2, u_mask=0x3, i_mask=0x2,
        props=(P_PURE, P_COMM, P_OPT_U)),
    vir("IANDN", K_FAMILY, fam=FAM_LOGIC, arity=2, u_mask=0x3, i_mask=0x2,
        props=P_PURE),
    vir("IOR", K_FAMILY, fam=FAM_LOGIC, arity=2, u_mask=0x3, i_mask=0x2,
        props=(P_PURE, P_COMM, P_OPT_U)),
    vir("IXOR", K_FAMILY, fam=FAM_LOGIC, arity=2, u_mask=0x3, i_mask=0x2,
        props=(P_PURE, P_COMM, P_OPT_U)),
    vir("IOR_UNIFORM", K_FAMILY, fam=FAM_UNIFORM, props=P_PURE),
    vir("STORE_UNIFORM", K_FAMILY, fam=FAM_UNIFORM, props=P_NO_DEST),
    vir("ISHR", K_FAMILY, fam=FAM_SHIFT, u_mask=0x3,
        props=(P_PURE, P_OPT_U, P_DUAL_U)),
    vir("USHR", K_FAMILY, fam=FAM_SHIFT, u_mask=0x3,
        props=(P_PURE, P_OPT_U, P_DUAL_U)),
    vir("ISHL", K_FAMILY, fam=FAM_SHIFT, u_mask=0x3,
        props=(P_PURE, P_OPT_U, P_DUAL_U)),
    vir("EXTR", K_CUSTOM, enc_name="extr", arity=3, i_mask=0x4, props=P_PURE),
    _minmax("IMIN", 7),
    _minmax("IMAX", 6),
    _minmax("UMIN", 5),
    _minmax("UMAX", 4),

    # Half-precision float ALU and SFU.
    vir("HADD", K_FAMILY, fam=FAM_HALF_ALU, enc_name="half2", arity=2,
        props=(P_PURE, P_COMM)),
    vir("HSUB", K_FAMILY, fam=FAM_HALF_ALU, enc_name="half2", arity=2,
        props=P_PURE),
    vir("HMUL", K_FAMILY, fam=FAM_HALF_ALU, enc_name="half2", arity=2,
        props=(P_PURE, P_COMM)),
    vir("HFMA", K_FAMILY, fam=FAM_HALF_ALU, enc_name="half3", arity=3,
        props=(P_PURE, P_COMM)),
    _minmax("HMIN", 1, half=True, floating=True),
    _minmax("HMAX", 0, half=True, floating=True),
    _fspec("HRCP", half=True, family=1, precision=0x48, rounding=0x20, reciprocal=True),
    _fspec("HRCP_F32", half=True, family=1, precision=0x48, rounding=0x20,
           reciprocal=True, wide_factor=True),
    _fspec("HRSQ", half=True, family=1, function=1),
    _fspec("HSQRT_FACTOR", half=True, function=1, wide_factor=True),
    vir("HMUL_MIXED", K_FAMILY, fam=FAM_HALF_ALU, enc_name="half2", arity=2,
        props=P_PURE),
    _fspec("HEXP2", half=True, family=1, function=2),
    _fspec("HLOG2", half=True, function=2),
    _fspec("HFLOOR", half=True, rounding=2),
    _fspec("HCEIL", half=True, rounding=4),
    _fspec("HTRUNC", half=True, rounding=6),
    _fspec("HROUND_EVEN", half=True),

    # Single-precision float ALU, cube, SFU, derivatives, select.
    vir("FADD", K_FAMILY, fam=FAM_FLOAT2, arity=2, u_mask=0x3,
        half=0x3, half_hi=0x3,
        props=(P_PURE, P_SAT, P_COMM, P_OPT_U, P_DUAL_U, P_FP16_HALF)),
    vir("FSUB", K_FAMILY, fam=FAM_FLOAT2, arity=2, u_mask=0x3,
        half=0x3, half_hi=0x3,
        props=(P_PURE, P_SAT, P_OPT_U, P_DUAL_U, P_FP16_HALF)),
    vir("FMUL", K_FAMILY, fam=FAM_FLOAT2, arity=2, u_mask=0x3,
        half=0x3, half_hi=0x3,
        props=(P_PURE, P_SAT, P_COMM, P_OPT_U, P_DUAL_U, P_FP16_HALF)),
    vir("FMUL_PROJECT", K_FAMILY, fam=FAM_FLOAT2, enc_name="float2_project", arity=2),
    vir("FADD_IMM", K_FAMILY, fam=FAM_FLOAT2_IMM, arity=1, props=(P_PURE, P_SAT)),
    vir("FMUL_IMM", K_FAMILY, fam=FAM_FLOAT2_IMM, arity=1, props=(P_PURE, P_SAT)),
    vir("FSAT", K_CUSTOM, arity=1, props=(P_PURE, P_SAT)),
    _minmax("FMIN", 1, floating=True),
    _minmax("FMAX", 0, floating=True),
    vir("FMA", K_CUSTOM, arity=3, u_mask=0x7, i_mask=0x6, half=0x6, half_hi=0x4,
        props=(P_PURE, P_SAT, P_COMM, P_OPT_U, P_DUAL_U, P_FP16_HALF)),
    vir("CUBE", K_CUSTOM, enc_name="cube", arity=3),
    _fspec("FRCP", half=False, family=1, precision=0x48, rounding=0x20,
           reciprocal=True, sat=True),
    _fspec("FRSQ", half=False, family=1, function=1, sat=True, spec_u=True),
    _fspec("FSQRT_FACTOR", half=False, function=1, sat=True),
    _fspec("FSIN_FACTOR", half=False, function=3, sat=True),
    _fspec("FEXP2", half=False, family=1, function=2, sat=True, spec_u=True),
    _fspec("FLOG2", half=False, function=2, sat=True, spec_u=True),
    _fspec("FFLOOR", half=False, rounding=2, spec_u=True),
    _fspec("FCEIL", half=False, rounding=4, spec_u=True),
    _fspec("FTRUNC", half=False, rounding=6, spec_u=True),
    _fspec("FROUND_EVEN", half=False, spec_u=True),
    vir("DERIVATIVE", K_FIXED, enc_name="derivative", arity=1, props=P_SAT),
    vir("SELECT", K_CUSTOM, arity=4, u_mask=0xF, i_mask=0xE,
        props=(P_PURE, P_OPT_U, P_DUAL_U)),

    # Pre-physical SSA RA pseudos.
    vir("COLLECT", K_PSEUDO),
    vir("SPLIT", K_PSEUDO),
    vir("MOV", K_PSEUDO, arity=1),
    vir("SWAP", K_PSEUDO, arity=2, props=P_NO_DEST),
    vir("PHI", K_PSEUDO),
    vir("PHI_SRC", K_PSEUDO, props=P_NO_DEST),

    # Predicates and execution-mask control flow.
    vir("PREDICATE_COMPARE", K_CUSTOM, arity=2, u_mask=0x3, i_mask=0x2,
        props=(P_CTRL, P_OPT_U, P_DUAL_U, P_NO_DEST)),
    vir("EXEC_MASK_PUSH", K_FIXED, enc_name="exec_mask_push", arity=0,
        props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("EXEC_MASK_ELSE", "exec_mask_else", (0x0F, 0x04, 0x04, 0x19),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("EXEC_MASK_POP", "exec_mask_pop", (0x0F, 0x06, 0x04, 0x01, 0x00, 0x00),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("LOOP_MASK_PUSH", "loop_mask_push", (0x0F, 0x05, 0x54, 0x1A),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    vir("LOOP_MASK_UPDATE", K_FIXED, enc_name="loop_mask_update", arity=0,
        props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("LOOP_MASK_POP", "loop_mask_pop", (0x0F, 0x06, 0x04, 0x02, 0x00, 0x00),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    vir("JMP_EXEC_ANY", K_FIXED, enc_name="jmp_exec_any", arity=0,
        props=(P_CTRL, P_NO_DEST)),
    vir("JMP_EXEC_NONE", K_FIXED, enc_name="jmp_exec_none", arity=0,
        props=(P_CTRL, P_NO_DEST)),
    vir("BREAK_MASK_UNWIND", K_FIXED, enc_name="break_mask_unwind", arity=0,
        props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),

    # Fragment interpolation, exports, and tilebuffer ops.
    vir("INTERPOLATION_POSITION", K_FIXED, enc_name="interpolation_position", arity=1),
    vir("ITER", K_CUSTOM),
    vir("ITER_FLAT", K_FIXED, enc_name="iter_flat", arity=0),
    vir("VARY_STORE", K_FIXED, enc_name="vary_store", arity=1, props=P_NO_DEST),
    vir("COVERAGE", K_FIXED, enc_name="coverage", arity=1, props=P_NO_DEST),
    vir("DEPTH_STORE", K_FIXED, enc_name="depth_store", arity=2, props=P_NO_DEST),
    vir("TILE_ACCESS", K_FIXED, enc_name="tile_access", arity=0, props=P_NO_DEST),
    vir("TILE_LOAD", K_CUSTOM),
    vir("TILE_STORE", K_CUSTOM, props=P_NO_DEST),
    vir("TILE_FENCE", K_FIXED, enc_name="tile_fence", arity=0, props=P_NO_DEST),

    # Device stores & atomics.
    vir("DEVICE_STORE", K_CUSTOM, props=P_NO_DEST),
    vir("DEVICE_ATOMIC", K_CUSTOM),
    vir("DEVICE_ATOMIC_RESULT", K_FIXED, enc_name="device_atomic_result", arity=1,
        props=P_NO_DEST),

    # Subgroup ops & MMA.
    vir("SUBGROUP_REDUCE_UMAX", K_FIXED, enc_name="subgroup_reduce_umax", arity=1),
    vir("SUBGROUP_BALLOT", K_FIXED, enc_name="subgroup_ballot", arity=1),
    vir("SUBGROUP_REDUCE", K_CUSTOM, enc_name="subgroup_reduce", arity=1),
    vir("SUBGROUP_SHUFFLE", K_CUSTOM, enc_name="subgroup_shuffle", arity=2, i_mask=0x2),
    vir("SUBGROUP_ROTATE", K_CUSTOM, enc_name="subgroup_rotate", arity=3, i_mask=0x4),
    vir("MMA", K_CUSTOM),

    # Barriers, fences, halt, shared/private/spill memory, wait.
    vir("WORKGROUP_BARRIER", K_FIXED, enc_name="workgroup_barrier", arity=0,
        props=(P_CTRL, P_NO_DEST)),
    vir("DEVICE_FENCE", K_CUSTOM, arity=0, props=(P_CTRL, P_NO_DEST)),
    _fixed_bytes("HALT", "halt", (0x0E, 0x00, 0x00, 0x00),
                 props=(P_CTRL, P_NO_DEST)),
    vir("SHARED_LOAD", K_CUSTOM, enc_name="shared_load", arity=1),
    vir("SHARED_ATOMIC", K_CUSTOM, enc_name="shared_atomic"),
    vir("SHARED_STORE", K_CUSTOM, props=P_NO_DEST),
    vir("PRIVATE_LOAD", K_CUSTOM, enc_name="private_load", arity=1),
    vir("PRIVATE_STORE", K_CUSTOM, enc_name="private_store", props=P_NO_DEST),
    vir("SPILL_STORE", K_FIXED, enc_name="spill_store", arity=1, props=P_NO_DEST),
    vir("SPILL_LOAD", K_FIXED, enc_name="spill_load", arity=0),
    vir("WAIT", K_FIXED, enc_name="wait", arity=0, props=(P_CTRL, P_NO_DEST)),

    # Native mesh, object calls, linked stage ABI.
    vir("MESH_CONTROL", K_FAMILY, fam=FAM_MESH, enc_name="mesh_control", arity=2,
        half=0x3, half_hi=0x3, props=(P_CTRL, P_NO_DEST)),
    vir("MESH_SUBMIT_SCALAR", K_FAMILY, fam=FAM_MESH, enc_name="mesh_submit_scalar",
        arity=2, props=(P_CTRL, P_NO_DEST)),
    vir("MESH_SUBMIT_CHANNEL", K_FAMILY, fam=FAM_MESH, enc_name="mesh_submit_channel",
        arity=2, props=(P_CTRL, P_NO_DEST)),
    vir("OBJECT_CALL", K_FIXED, enc_name="object_call", arity=0,
        props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("OBJECT_RETURN", "object_return", (0x8F, 0x12, 0x54, 0x00),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    _fixed_bytes("OBJECT_RESUME", "object_resume", (0x0F, 0x06, 0x04, 0x02, 0x00, 0x00),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
    vir("OBJECT_GRID_EXPORT", K_FIXED, enc_name="object_grid_export", arity=1,
        props=P_NO_DEST),
    vir("ENTRY_REG", K_PSEUDO, arity=0),
    _fixed_bytes("ABI_RETURN", "abi_return", (0x8F, 0x02, 0x54, 0x00),
                 props=(P_OPT_BAR, P_CTRL, P_NO_DEST)),
)


# Subgroup reduce operation table (indexed by enum agx_apple9_subgroup_reduce_op).
SUBGROUP_REDUCE_OPS: tuple[tuple[str, int, int, tuple[str, ...], bool], ...] = (
    ("AGX_APPLE9_SUBGROUP_REDUCE_AND",  0x3F, 0x00, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_OR",   0xBF, 0x00, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_XOR",  0x3F, 0x01, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_IADD", 0xBF, 0x01, W16_32, True),
    ("AGX_APPLE9_SUBGROUP_REDUCE_IMIN", 0x3F, 0x02, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_IMAX", 0xBF, 0x02, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_UMIN", 0x3F, 0x03, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_UMAX", 0xBF, 0x03, W16_32, False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_FADD", 0x3F, 0x06, (W32,), True),
    ("AGX_APPLE9_SUBGROUP_REDUCE_FMUL", 0xBF, 0x06, (W32,), True),
    ("AGX_APPLE9_SUBGROUP_REDUCE_FMIN", 0xBF, 0x05, (W32,), False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_FMAX", 0xBF, 0x07, (W32,), False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_HADD", 0x3F, 0x04, (W16,), True),
    ("AGX_APPLE9_SUBGROUP_REDUCE_HMIN", 0xBF, 0x05, (W16,), False),
    ("AGX_APPLE9_SUBGROUP_REDUCE_HMAX", 0xBF, 0x07, (W16,), False),
)

# Subgroup shuffle mode table (indexed by enum agx_apple9_subgroup_shuffle_mode).
SUBGROUP_SHUFFLE_MODES: tuple[tuple[str, int, int], ...] = (
    ("AGX_APPLE9_SUBGROUP_SHUFFLE_IDX",  0x47, 0x04),
    ("AGX_APPLE9_SUBGROUP_SHUFFLE_XOR",  0xC7, 0x04),
    ("AGX_APPLE9_SUBGROUP_SHUFFLE_UP",   0x47, 0x05),
    ("AGX_APPLE9_SUBGROUP_SHUFFLE_DOWN", 0xC7, 0x05),
    ("AGX_APPLE9_QUAD_SHUFFLE_IDX",      0x47, 0x00),
    ("AGX_APPLE9_QUAD_SHUFFLE_XOR",      0xC7, 0x00),
)

# Workgroup barrier mode table (indexed by enum agx_apple9_workgroup_barrier_mode).
WORKGROUP_BARRIER_MODES: tuple[tuple[str, int, int], ...] = (
    ("AGX_APPLE9_WORKGROUP_BARRIER_SHARED", 0x61, 0x09),
    ("AGX_APPLE9_WORKGROUP_BARRIER_NONE",   0x41, 0x09),
    ("AGX_APPLE9_WORKGROUP_BARRIER_DEVICE", 0x85, 0x08),
)

# Hardware device fence mode table (indexed by enum agx_apple9_fence_mode 0..10).
DEVICE_FENCE_MODES: tuple[tuple[str, int, int, int, bool], ...] = (
    ("AGX_APPLE9_FENCE_DEVICE",                     0x54, 0x84, 0x0A, False),
    ("AGX_APPLE9_FENCE_RASTER_ORDER",               0x54, 0xC4, 0x08, False),
    ("AGX_APPLE9_FENCE_OBJECT_GRID",                0x54, 0x44, 0x0A, False),
    ("AGX_APPLE9_FENCE_DEVICE_ACQUIRE",             0x54, 0x84, 0x0A, False),
    ("AGX_APPLE9_FENCE_IMAGE_WRITE",                0x54, 0x50, 0x06, True),
    ("AGX_APPLE9_FENCE_TEX_INVALIDATE",             0x54, 0xD0, 0x06, False),
    ("AGX_APPLE9_FENCE_IMAGE_WRITE_FS",             0x64, 0x50, 0x06, True),
    ("AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP",      0x54, 0x51, 0x0E, True),
    ("AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP",   0x54, 0xD1, 0x0E, False),
    ("AGX_APPLE9_FENCE_IMAGE_WRITE_WORKGROUP_MEM",  0x54, 0x50, 0x0E, True),
    ("AGX_APPLE9_FENCE_TEX_INVALIDATE_WORKGROUP_MEM", 0x54, 0xD0, 0x0E, False),
)

# Unified checked texture-mode table (shared by lowering and packer).
# Kind indices match enum agx_apple9_texture_mode_kind:
#   0 = IMPLICIT, 1 = EXPLICIT_LOD, 2 = BIAS, 3 = GRAD
TEX_IMPLICIT = 0
TEX_EXPLICIT = 1
TEX_BIAS = 2
TEX_GRAD = 3
TEX_KIND_NAMES = (
    "AGX_APPLE9_TEX_MODE_IMPLICIT",
    "AGX_APPLE9_TEX_MODE_EXPLICIT_LOD",
    "AGX_APPLE9_TEX_MODE_BIAS",
    "AGX_APPLE9_TEX_MODE_GRAD",
)

# Measured (base_mode -> offset_mode) pairs (artifacts/msl-encodings-20261011
# and array-offset-probe). Shared by agx_apple9_texture_offset_mode() and the
# checked texture-mode table below.
TEXTURE_OFFSET_PAIRS: tuple[tuple[int, int], ...] = (
    (0x0000, 0x0201),  # 2D implicit LOD / gather
    (0x0004, 0x0281),  # 2D bias
    (0x0804, 0x02C1),  # 2D bias + min_lod
    (0x0100, 0x0211),  # 2D explicit LOD
    (0x0408, 0x0301),  # 2D array implicit LOD / gather
    (0x0508, 0x0651),  # 2D array explicit LOD
    (0x0104, 0x0209),  # 2D gradient
    (0x0114, 0x0249),  # 2D gradient + min_lod
    (0x0020, 0x0221),  # 2D compare implicit LOD
    (0x0024, 0x02A1),  # 2D compare bias
    (0x0120, 0x0231),  # 2D compare explicit LOD
    (0x0124, 0x0229),  # 2D compare gradient
    (0x0420, 0x0321),  # 2D array compare implicit LOD / gather
    (0x042C, 0x03A1),  # 2D array compare dynamic bias
    (0x0C20, 0x0361),  # 2D array compare + min_lod
    (0x0180, 0x0291),  # 3D implicit LOD (historical measured pair)
    (0x0190, 0x0299),  # 3D bias
    (0x2188, 0x0219),  # 3D explicit LOD
    (0x019C, 0x0319),  # 3D gradient
)

# Rows: (kind, dim, shadow, min_lod, base_mode, offset_mode)
# Dimensions: 0=2D, 1=Cube, 2=2D-array, 3=3D, 4=Cube-array, 5=1D, 6=1D-array.
TEXTURE_MODES: tuple[tuple[int, int, bool, bool, int | None, int | None], ...] = (
    # Implicit LOD (ENC_TEXTURE_SAMPLE for 2D/1D/1D-array; ENC_TEXTURE_LOD imm=2 for array).
    (TEX_IMPLICIT, 0, False, False, 0x0000, 0x0201),
    (TEX_IMPLICIT, 0, True,  False, 0x0020, 0x0221),
    (TEX_IMPLICIT, 2, False, False, 0x0408, 0x0301),
    (TEX_IMPLICIT, 2, True,  False, 0x0420, 0x0321),
    (TEX_IMPLICIT, 4, True,  False, 0x0430, None),
    (TEX_IMPLICIT, 5, False, False, 0x0080, None),
    (TEX_IMPLICIT, 5, True,  False, 0x00A0, None),
    (TEX_IMPLICIT, 6, False, False, 0x00A0, None),
    (TEX_IMPLICIT, 6, True,  False, 0x00A0, None),

    # Explicit LOD (ENC_TEXTURE_LOD imm=0; with min_lod=True this is implicit+min_lod).
    (TEX_EXPLICIT, 0, False, False, 0x0100, 0x0211),
    (TEX_EXPLICIT, 0, True,  False, 0x0120, 0x0231),
    (TEX_EXPLICIT, 0, True,  True,  0x0820, None),
    (TEX_EXPLICIT, 1, False, False, 0x0300, None),
    (TEX_EXPLICIT, 1, False, True,  0x0810, None),
    (TEX_EXPLICIT, 1, True,  False, 0x0320, None),
    (TEX_EXPLICIT, 2, False, False, 0x0508, 0x0651),
    (TEX_EXPLICIT, 2, True,  False, 0x0528, None),
    (TEX_EXPLICIT, 2, True,  True,  0x0C20, 0x0361),
    (TEX_EXPLICIT, 3, False, False, 0x2188, 0x0219),
    (TEX_EXPLICIT, 3, False, True,  0x2180, None),
    (TEX_EXPLICIT, 3, True,  False, 0x21A8, None),
    (TEX_EXPLICIT, 4, False, False, 0x0308, None),
    (TEX_EXPLICIT, 4, False, True,  0x0C10, None),
    (TEX_EXPLICIT, 4, True,  False, 0x0328, None),
    (TEX_EXPLICIT, 5, False, False, 0x0080, None),
    (TEX_EXPLICIT, 5, True,  False, 0x00A0, None),
    (TEX_EXPLICIT, 6, False, False, 0x00A0, None),
    (TEX_EXPLICIT, 6, True,  False, 0x00A0, None),

    # Biased LOD (ENC_TEXTURE_LOD imm=1).
    (TEX_BIAS, 0, False, False, 0x0004, 0x0281),
    (TEX_BIAS, 0, False, True,  0x0804, 0x02C1),
    (TEX_BIAS, 0, True,  False, 0x0024, 0x02A1),
    (TEX_BIAS, 0, True,  True,  0x0824, None),
    (TEX_BIAS, 1, False, False, 0x0014, None),
    (TEX_BIAS, 1, False, True,  0x0814, None),
    (TEX_BIAS, 1, True,  False, 0x0034, None),
    (TEX_BIAS, 2, False, False, 0x040C, None),
    (TEX_BIAS, 2, False, True,  None,   0x03C1),  # offset-only measured form
    (TEX_BIAS, 2, True,  False, 0x042C, 0x03A1),
    (TEX_BIAS, 3, False, False, 0x0190, 0x0299),
    (TEX_BIAS, 3, False, True,  0x2190, None),
    (TEX_BIAS, 3, True,  False, 0x01B0, None),
    (TEX_BIAS, 4, False, False, 0x001C, None),
    (TEX_BIAS, 4, True,  False, 0x0038, None),
    (TEX_BIAS, 5, False, False, 0x0080, None),
    (TEX_BIAS, 5, True,  False, 0x00A0, None),
    (TEX_BIAS, 6, False, False, 0x00A0, None),
    (TEX_BIAS, 6, True,  False, 0x00A0, None),

    # Explicit gradients (ENC_TEXTURE_GRAD imm=0).
    (TEX_GRAD, 0, False, False, 0x0104, 0x0209),
    (TEX_GRAD, 0, False, True,  0x0114, 0x0249),
    (TEX_GRAD, 0, True,  False, 0x0124, 0x0229),
    (TEX_GRAD, 1, False, False, 0x0304, None),
    (TEX_GRAD, 1, True,  False, 0x0324, None),
    (TEX_GRAD, 2, False, False, 0x050C, None),
    (TEX_GRAD, 2, False, True,  0x051C, None),
    (TEX_GRAD, 2, True,  False, 0x052C, None),
    (TEX_GRAD, 3, False, False, 0x019C, 0x0319),
    (TEX_GRAD, 3, False, True,  0x219C, None),
    (TEX_GRAD, 3, True,  False, 0x01BC, None),
    (TEX_GRAD, 4, False, False, 0x030C, None),
    (TEX_GRAD, 4, True,  False, 0x032C, None),
    (TEX_GRAD, 5, False, False, 0x0104, 0x0209),
    (TEX_GRAD, 5, True,  False, 0x0124, 0x0229),
    (TEX_GRAD, 6, False, False, 0x0104, 0x0209),
    (TEX_GRAD, 6, True,  False, 0x0124, 0x0229),
)

TEXTURE_FETCH_MODES: dict[int, int] = {
    0: 0x2480,
    2: 0x24A0,
    3: 0x0198,
    5: 0x0C80,
    6: 0x0CA0,
}

TEXTURE_MS_MODES: dict[int, int] = {
    0: 0x1480,
    2: 0x14A0,
}


def _build_family_tables(
    opcodes: tuple[VirOpcode, ...],
) -> tuple[
    dict[str, int],
    list[tuple[str, FixedBytes]],
    list[tuple[str, I2FParams]],
    list[tuple[str, F2IParams]],
    list[tuple[str, FSpecialParams]],
    list[tuple[str, MinMaxParams]],
    list[tuple[str, BitUnaryParams]],
]:
    family_indices: dict[str, int] = {}
    fixed_bytes: list[tuple[str, FixedBytes]] = []
    i2f: list[tuple[str, I2FParams]] = []
    f2i: list[tuple[str, F2IParams]] = []
    fspec: list[tuple[str, FSpecialParams]] = []
    minmax: list[tuple[str, MinMaxParams]] = []
    bit_unary: list[tuple[str, BitUnaryParams]] = []

    for op in opcodes:
        if op.payload is None:
            family_indices[op.enum_name] = 0
        elif isinstance(op.payload, FixedBytes):
            family_indices[op.enum_name] = len(fixed_bytes)
            fixed_bytes.append((op.enum_name, op.payload))
        elif isinstance(op.payload, I2FParams):
            family_indices[op.enum_name] = len(i2f)
            i2f.append((op.enum_name, op.payload))
        elif isinstance(op.payload, F2IParams):
            family_indices[op.enum_name] = len(f2i)
            f2i.append((op.enum_name, op.payload))
        elif isinstance(op.payload, FSpecialParams):
            family_indices[op.enum_name] = len(fspec)
            fspec.append((op.enum_name, op.payload))
        elif isinstance(op.payload, MinMaxParams):
            family_indices[op.enum_name] = len(minmax)
            minmax.append((op.enum_name, op.payload))
        elif isinstance(op.payload, BitUnaryParams):
            family_indices[op.enum_name] = len(bit_unary)
            bit_unary.append((op.enum_name, op.payload))
        else:
            raise ValueError(f"{op.name}: unsupported payload type")

    return family_indices, fixed_bytes, i2f, f2i, fspec, minmax, bit_unary


def validate_vir_tables(
    encodings: tuple[Encoding, ...],
    opcodes: tuple[VirOpcode, ...],
) -> None:
    enc_by_enum = {e.enum_name: e for e in encodings}
    seen_names: set[str] = set()

    payload_for_family = {
        FAM_FIXED_BYTES: FixedBytes,
        FAM_I2F: I2FParams,
        FAM_F2I: F2IParams,
        FAM_FSPECIAL: FSpecialParams,
        FAM_MINMAX: MinMaxParams,
        FAM_BIT_UNARY: BitUnaryParams,
    }

    for op in opcodes:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", op.name):
            raise ValueError(f"invalid VIR opcode name: {op.name!r}")
        if op.name in seen_names:
            raise ValueError(f"duplicate VIR opcode: {op.name!r}")
        seen_names.add(op.name)

        if op.kind not in VALID_KINDS:
            raise ValueError(f"{op.name}: invalid kind {op.kind!r}")
        if op.family not in VALID_FAMILIES:
            raise ValueError(f"{op.name}: invalid family {op.family!r}")
        if len(op.props) != len(set(op.props)) or any(p not in VALID_PROPS for p in op.props):
            raise ValueError(f"{op.name}: invalid props {op.props!r}")
        if op.arity != ARITY_VAR and not (0 <= op.arity <= 10):
            raise ValueError(f"{op.name}: invalid arity {op.arity}")

        for fname, mask in (
            ("inline_uniform_mask", op.inline_uniform_mask),
            ("inline_immediate_mask", op.inline_immediate_mask),
            ("half_src_mask", op.half_src_mask),
            ("half_src_hi_mask", op.half_src_hi_mask),
            ("half_src_sext_mask", op.half_src_sext_mask),
        ):
            if mask & ~0xF:
                raise ValueError(f"{op.name}: {fname} {mask:#x} exceeds 4-slot range")
        if op.half_src_hi_mask & ~op.half_src_mask:
            raise ValueError(f"{op.name}: half_src_hi_mask must be a subset of half_src_mask")
        if op.half_src_sext_mask & ~op.half_src_mask:
            raise ValueError(f"{op.name}: half_src_sext_mask must be a subset of half_src_mask")

        expected_payload = payload_for_family.get(op.family)
        if expected_payload is None:
            if op.payload is not None:
                raise ValueError(f"{op.name}: unexpected payload for family {op.family}")
        else:
            if not isinstance(op.payload, expected_payload):
                raise ValueError(f"{op.name}: family {op.family} requires {expected_payload.__name__}")

        if op.fixed_encoding is not None:
            if op.fixed_encoding not in enc_by_enum:
                raise ValueError(f"{op.name}: unknown fixed_encoding {op.fixed_encoding!r}")
            e = enc_by_enum[op.fixed_encoding]
            if isinstance(op.payload, FixedBytes):
                if len(op.payload.bytes) != e.length or len(op.payload.bytes) > 6:
                    raise ValueError(f"{op.name}: fixed bytes length {len(op.payload.bytes)} != encoding length {e.length}")
            if op.inline_uniform_mask & ~e.uniform_src_mask:
                raise ValueError(
                    f"{op.name}: inline_uniform_mask {op.inline_uniform_mask:#x} "
                    f"exceeds {e.name} uniform_src_mask {e.uniform_src_mask:#x}"
                )
            if op.inline_immediate_mask & ~e.immediate_src_mask:
                raise ValueError(
                    f"{op.name}: inline_immediate_mask {op.inline_immediate_mask:#x} "
                    f"exceeds {e.name} immediate_src_mask {e.immediate_src_mask:#x}"
                )
            if op.half_src_mask & ~e.half_src_mask:
                raise ValueError(f"{op.name}: half_src_mask exceeds {e.name} half_src_mask")
            if op.half_src_hi_mask & ~e.half_src_hi_mask:
                raise ValueError(f"{op.name}: half_src_hi_mask exceeds {e.name} half_src_hi_mask")
            if op.half_src_sext_mask & ~e.half_src_sext_mask:
                raise ValueError(f"{op.name}: half_src_sext_mask exceeds {e.name} half_src_sext_mask")

    # Validate texture mode table consistency against TEXTURE_OFFSET_PAIRS.
    offset_map = dict(TEXTURE_OFFSET_PAIRS)
    if len(offset_map) != len(TEXTURE_OFFSET_PAIRS):
        raise ValueError("duplicate base mode in TEXTURE_OFFSET_PAIRS")

    seen_tex_keys: set[tuple[int, int, bool, bool]] = set()
    for kind, dim, shadow, min_lod, base_mode, ofs_mode in TEXTURE_MODES:
        key = (kind, dim, shadow, min_lod)
        if not (0 <= kind < len(TEX_KIND_NAMES)):
            raise ValueError(f"invalid texture mode kind {kind}")
        if not (0 <= dim < TEX_DIM_COUNT):
            raise ValueError(f"invalid texture dimension {dim}")
        if key in seen_tex_keys:
            raise ValueError(f"duplicate TEXTURE_MODES key {key}")
        seen_tex_keys.add(key)
        if base_mode is None and ofs_mode is None:
            raise ValueError(f"empty TEXTURE_MODES row {key}")
        if base_mode is not None:
            if not (0 <= base_mode < TEXTURE_MODE_INVALID):
                raise ValueError(f"invalid base_mode {base_mode:#x} for {key}")
            expected_ofs = offset_map.get(base_mode)
            if ofs_mode != expected_ofs:
                raise ValueError(
                    f"TEXTURE_MODES {key}: base {base_mode:#06x} offset_mode "
                    f"{ofs_mode} disagrees with TEXTURE_OFFSET_PAIRS ({expected_ofs})"
                )
        if ofs_mode is not None and not (0 <= ofs_mode < TEXTURE_MODE_INVALID):
            raise ValueError(f"invalid offset_mode {ofs_mode:#x} for {key}")


def _extract_enum_idents(header_path: Path, enum_name: str) -> list[str]:
    text = header_path.read_text(encoding="utf-8")
    m = re.search(rf"enum\s+{enum_name}\s*\{{(.*?)\}};", text, flags=re.S)
    if not m:
        raise ValueError(f"could not find enum {enum_name} in {header_path}")
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    idents: list[str] = []
    for raw in body.split(","):
        tok = raw.strip()
        if not tok:
            continue
        ident = tok.split("=", 1)[0].strip()
        idents.append(ident)
    return idents


def check_against_ir_header(opcodes: tuple[VirOpcode, ...], header_path: Path) -> None:
    header_ops = _extract_enum_idents(header_path, "agx_apple9_vir_opcode")
    table_ops = [op.enum_name for op in opcodes]
    if header_ops != table_ops:
        missing = sorted(set(header_ops) - set(table_ops))
        extra = sorted(set(table_ops) - set(header_ops))
        raise ValueError(
            f"enum agx_apple9_vir_opcode mismatch vs {header_path}: "
            f"header={len(header_ops)}, table={len(table_ops)}, "
            f"missing={missing}, extra={extra}"
        )

    for enum_name, expected in (
        ("agx_apple9_subgroup_reduce_op", [r[0] for r in SUBGROUP_REDUCE_OPS]),
        ("agx_apple9_subgroup_shuffle_mode", [r[0] for r in SUBGROUP_SHUFFLE_MODES]),
        ("agx_apple9_workgroup_barrier_mode", [r[0] for r in WORKGROUP_BARRIER_MODES]),
    ):
        hdr = _extract_enum_idents(header_path, enum_name)
        if hdr != expected:
            raise ValueError(f"enum {enum_name} mismatch vs {header_path}: {hdr} != {expected}")

    hdr_fences = _extract_enum_idents(header_path, "agx_apple9_fence_mode")
    expected_fences = [r[0] for r in DEVICE_FENCE_MODES] + ["AGX_APPLE9_FENCE_COMPILER_BARRIER"]
    if hdr_fences != expected_fences:
        raise ValueError(
            f"enum agx_apple9_fence_mode mismatch vs {header_path}: {hdr_fences} != {expected_fences}"
        )


def render_vir_header(opcodes: tuple[VirOpcode, ...]) -> str:
    _, fixed_bytes, i2f, f2i, fspec, minmax, bit_unary = _build_family_tables(opcodes)
    return (
        "/*\n"
        " * Copyright 2026 The Asahi Linux Contributors\n"
        " * SPDX-License-Identifier: MIT\n"
        " *\n"
        " * Generated by agx_apple9_isa.py -- DO NOT EDIT.\n"
        " */\n"
        "\n"
        "#ifndef AGX_APPLE9_VIR_TABLE_H\n"
        "#define AGX_APPLE9_VIR_TABLE_H\n"
        "\n"
        '#include "agx_apple9_ir.h"\n'
        "\n"
        f"#define AGX_APPLE9_VIR_OPCODE_COUNT {len(opcodes)}u\n"
        f"#define AGX_APPLE9_FIXED_BYTES_COUNT {len(fixed_bytes)}u\n"
        f"#define AGX_APPLE9_I2F_COUNT {len(i2f)}u\n"
        f"#define AGX_APPLE9_F2I_COUNT {len(f2i)}u\n"
        f"#define AGX_APPLE9_FSPECIAL_COUNT {len(fspec)}u\n"
        f"#define AGX_APPLE9_MINMAX_COUNT {len(minmax)}u\n"
        f"#define AGX_APPLE9_BIT_UNARY_COUNT {len(bit_unary)}u\n"
        f"#define AGX_APPLE9_SUBGROUP_REDUCE_COUNT {len(SUBGROUP_REDUCE_OPS)}u\n"
        f"#define AGX_APPLE9_SUBGROUP_SHUFFLE_COUNT {len(SUBGROUP_SHUFFLE_MODES)}u\n"
        f"#define AGX_APPLE9_WORKGROUP_BARRIER_COUNT {len(WORKGROUP_BARRIER_MODES)}u\n"
        f"#define AGX_APPLE9_DEVICE_FENCE_HW_COUNT {len(DEVICE_FENCE_MODES)}u\n"
        f"#define AGX_APPLE9_TEXTURE_DIM_COUNT {TEX_DIM_COUNT}u\n"
        f"#define AGX_APPLE9_TEXTURE_MODE_INVALID {TEXTURE_MODE_INVALID:#06x}u\n"
        f"#define AGX_APPLE9_TEXTURE_OFFSET_PAIR_COUNT {len(TEXTURE_OFFSET_PAIRS)}u\n"
        "\n"
        "enum agx_apple9_texture_mode_kind {\n"
        "   AGX_APPLE9_TEX_MODE_IMPLICIT = 0,\n"
        "   AGX_APPLE9_TEX_MODE_EXPLICIT_LOD = 1,\n"
        "   AGX_APPLE9_TEX_MODE_BIAS = 2,\n"
        "   AGX_APPLE9_TEX_MODE_GRAD = 3,\n"
        "   AGX_APPLE9_TEX_MODE_KIND_COUNT = 4,\n"
        "};\n"
        "\n"
        "struct agx_apple9_fixed_bytes_desc {\n"
        "   uint8_t length;\n"
        "   uint8_t bytes[6];\n"
        "};\n"
        "\n"
        "struct agx_apple9_i2f_desc {\n"
        "   uint8_t b6_type_bits;\n"
        "   uint8_t b7;\n"
        "};\n"
        "\n"
        "struct agx_apple9_f2i_desc {\n"
        "   bool wide;\n"
        "   uint8_t b7;\n"
        "   uint8_t b8;\n"
        "};\n"
        "\n"
        "struct agx_apple9_fspecial_desc {\n"
        "   uint8_t family;\n"
        "   uint8_t function;\n"
        "   uint8_t precision;\n"
        "   uint8_t rounding;\n"
        "   bool reciprocal;\n"
        "   bool is_half_op;\n"
        "   bool wide_factor;\n"
        "};\n"
        "\n"
        "struct agx_apple9_minmax_desc {\n"
        "   uint8_t select;\n"
        "   bool floating;\n"
        "};\n"
        "\n"
        "struct agx_apple9_bit_unary_desc {\n"
        "   uint8_t b0;\n"
        "   uint8_t b1;\n"
        "   bool uniform_alt_fmt;\n"
        "};\n"
        "\n"
        "struct agx_apple9_subgroup_reduce_desc {\n"
        "   uint8_t b0;\n"
        "   uint8_t b1;\n"
        "   uint8_t width_mask;\n"
        "   bool prefix_supported;\n"
        "};\n"
        "\n"
        "struct agx_apple9_subgroup_shuffle_desc {\n"
        "   uint8_t b0;\n"
        "   uint8_t b1;\n"
        "};\n"
        "\n"
        "struct agx_apple9_workgroup_barrier_desc {\n"
        "   uint8_t b3;\n"
        "   uint8_t b4;\n"
        "};\n"
        "\n"
        "struct agx_apple9_device_fence_desc {\n"
        "   uint8_t b2_base;\n"
        "   uint8_t b3;\n"
        "   uint8_t b4;\n"
        "   bool uses_slot_mask;\n"
        "};\n"
        "\n"
        "struct agx_apple9_texture_offset_pair {\n"
        "   uint16_t mode;\n"
        "   uint16_t offset_mode;\n"
        "};\n"
        "\n"
        "extern const struct agx_apple9_vir_opcode_info\n"
        "   agx_apple9_vir_opcodes[AGX_APPLE9_VIR_OPCODE_COUNT];\n"
        "extern const struct agx_apple9_fixed_bytes_desc\n"
        "   agx_apple9_fixed_bytes_table[AGX_APPLE9_FIXED_BYTES_COUNT];\n"
        "extern const struct agx_apple9_i2f_desc\n"
        "   agx_apple9_i2f_table[AGX_APPLE9_I2F_COUNT];\n"
        "extern const struct agx_apple9_f2i_desc\n"
        "   agx_apple9_f2i_table[AGX_APPLE9_F2I_COUNT];\n"
        "extern const struct agx_apple9_fspecial_desc\n"
        "   agx_apple9_fspecial_table[AGX_APPLE9_FSPECIAL_COUNT];\n"
        "extern const struct agx_apple9_minmax_desc\n"
        "   agx_apple9_minmax_table[AGX_APPLE9_MINMAX_COUNT];\n"
        "extern const struct agx_apple9_bit_unary_desc\n"
        "   agx_apple9_bit_unary_table[AGX_APPLE9_BIT_UNARY_COUNT];\n"
        "extern const struct agx_apple9_subgroup_reduce_desc\n"
        "   agx_apple9_subgroup_reduce_table[AGX_APPLE9_SUBGROUP_REDUCE_COUNT];\n"
        "extern const struct agx_apple9_subgroup_shuffle_desc\n"
        "   agx_apple9_subgroup_shuffle_table[AGX_APPLE9_SUBGROUP_SHUFFLE_COUNT];\n"
        "extern const struct agx_apple9_workgroup_barrier_desc\n"
        "   agx_apple9_workgroup_barrier_table[AGX_APPLE9_WORKGROUP_BARRIER_COUNT];\n"
        "extern const struct agx_apple9_device_fence_desc\n"
        "   agx_apple9_device_fence_table[AGX_APPLE9_DEVICE_FENCE_HW_COUNT];\n"
        "extern const uint16_t agx_apple9_texture_modes\n"
        "   [AGX_APPLE9_TEX_MODE_KIND_COUNT][AGX_APPLE9_TEXTURE_DIM_COUNT][2][2][2];\n"
        "extern const uint16_t agx_apple9_texture_fetch_modes[AGX_APPLE9_TEXTURE_DIM_COUNT];\n"
        "extern const uint16_t agx_apple9_texture_ms_modes[AGX_APPLE9_TEXTURE_DIM_COUNT];\n"
        "extern const struct agx_apple9_texture_offset_pair\n"
        "   agx_apple9_texture_offset_pairs[AGX_APPLE9_TEXTURE_OFFSET_PAIR_COUNT];\n"
        "\n"
        "#endif\n"
    )


def render_vir_source(opcodes: tuple[VirOpcode, ...]) -> str:
    family_indices, fixed_bytes, i2f, f2i, fspec, minmax, bit_unary = _build_family_tables(opcodes)

    lines = [
        "/*",
        " * Copyright 2026 The Asahi Linux Contributors",
        " * SPDX-License-Identifier: MIT",
        " *",
        " * Generated by agx_apple9_isa.py -- DO NOT EDIT.",
        " */",
        "",
        '#include "agx_apple9_vir_table.h"',
        "",
        "#include <assert.h>",
        "",
        *[
            f'static_assert({op.enum_name} == {idx}u, "{op.enum_name} enum index mismatch");'
            for idx, op in enumerate(opcodes)
        ],
        "",
        *[
            f'static_assert({r[0]} == {idx}u, "{r[0]} enum index mismatch");'
            for idx, r in enumerate(SUBGROUP_REDUCE_OPS)
        ],
        *[
            f'static_assert({r[0]} == {idx}u, "{r[0]} enum index mismatch");'
            for idx, r in enumerate(SUBGROUP_SHUFFLE_MODES)
        ],
        *[
            f'static_assert({r[0]} == {idx}u, "{r[0]} enum index mismatch");'
            for idx, r in enumerate(WORKGROUP_BARRIER_MODES)
        ],
        *[
            f'static_assert({r[0]} == {idx}u, "{r[0]} enum index mismatch");'
            for idx, r in enumerate(DEVICE_FENCE_MODES)
        ],
        f'static_assert(AGX_APPLE9_FENCE_COMPILER_BARRIER == {len(DEVICE_FENCE_MODES)}u,',
        '              "AGX_APPLE9_FENCE_COMPILER_BARRIER enum index mismatch");',
        "",
        "const struct agx_apple9_vir_opcode_info agx_apple9_vir_opcodes[AGX_APPLE9_VIR_OPCODE_COUNT] = {",
    ]

    for op in opcodes:
        fields = [
            f'.name = "{op.name}"',
            f".kind = {op.kind}",
            f".family = {op.family}",
            f".fixed_encoding = {op.fixed_encoding or 'AGX_APPLE9_ENC_PSEUDO'}",
        ]
        fidx = family_indices[op.enum_name]
        if fidx:
            fields.append(f".family_index = {fidx}")
        if op.arity == ARITY_VAR:
            fields.append(".arity = AGX_APPLE9_VIR_ARITY_VARIABLE")
        else:
            fields.append(f".arity = {op.arity}")
        if op.inline_uniform_mask:
            fields.append(f".inline_uniform_mask = {op.inline_uniform_mask:#x}")
        if op.inline_immediate_mask:
            fields.append(f".inline_immediate_mask = {op.inline_immediate_mask:#x}")
        if op.half_src_mask:
            fields.append(f".half_src_mask = {op.half_src_mask:#x}")
        if op.half_src_hi_mask:
            fields.append(f".half_src_hi_mask = {op.half_src_hi_mask:#x}")
        if op.half_src_sext_mask:
            fields.append(f".half_src_sext_mask = {op.half_src_sext_mask:#x}")
        if op.props:
            fields.append(f".props = {_or_expr(op.props)}")
        lines.append(f"   [{op.enum_name}] = {{ " + ", ".join(fields) + " },")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_fixed_bytes_desc agx_apple9_fixed_bytes_table[AGX_APPLE9_FIXED_BYTES_COUNT] = {")
    for idx, (ename, fb) in enumerate(fixed_bytes):
        bstr = ", ".join(f"{b:#04x}" for b in fb.bytes)
        lines.append(f"   [{idx}] = {{ .length = {len(fb.bytes)}, .bytes = {{ {bstr} }} }}, /* {ename} */")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_i2f_desc agx_apple9_i2f_table[AGX_APPLE9_I2F_COUNT] = {")
    for idx, (ename, p) in enumerate(i2f):
        lines.append(f"   [{idx}] = {{ .b6_type_bits = {p.b6_type_bits:#04x}, .b7 = {p.b7:#04x} }}, /* {ename} */")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_f2i_desc agx_apple9_f2i_table[AGX_APPLE9_F2I_COUNT] = {")
    for idx, (ename, p) in enumerate(f2i):
        w = "true" if p.wide else "false"
        lines.append(f"   [{idx}] = {{ .wide = {w}, .b7 = {p.b7:#04x}, .b8 = {p.b8:#04x} }}, /* {ename} */")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_fspecial_desc agx_apple9_fspecial_table[AGX_APPLE9_FSPECIAL_COUNT] = {")
    for idx, (ename, p) in enumerate(fspec):
        lines.append(
            f"   [{idx}] = {{ .family = {p.family}, .function = {p.function}, "
            f".precision = {p.precision:#04x}, .rounding = {p.rounding:#04x}, "
            f".reciprocal = {'true' if p.reciprocal else 'false'}, "
            f".is_half_op = {'true' if p.is_half_op else 'false'}, "
            f".wide_factor = {'true' if p.wide_factor else 'false'} }}, /* {ename} */"
        )
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_minmax_desc agx_apple9_minmax_table[AGX_APPLE9_MINMAX_COUNT] = {")
    for idx, (ename, p) in enumerate(minmax):
        lines.append(
            f"   [{idx}] = {{ .select = {p.select}, .floating = {'true' if p.floating else 'false'} }}, /* {ename} */"
        )
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_bit_unary_desc agx_apple9_bit_unary_table[AGX_APPLE9_BIT_UNARY_COUNT] = {")
    for idx, (ename, p) in enumerate(bit_unary):
        lines.append(
            f"   [{idx}] = {{ .b0 = {p.b0:#04x}, .b1 = {p.b1:#04x}, "
            f".uniform_alt_fmt = {'true' if p.uniform_alt_fmt else 'false'} }}, /* {ename} */"
        )
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_subgroup_reduce_desc agx_apple9_subgroup_reduce_table[AGX_APPLE9_SUBGROUP_REDUCE_COUNT] = {")
    for ename, b0, b1, widths, prefix in SUBGROUP_REDUCE_OPS:
        lines.append(
            f"   [{ename}] = {{ .b0 = {b0:#04x}, .b1 = {b1:#04x}, "
            f".width_mask = {_or_expr(widths)}, .prefix_supported = {'true' if prefix else 'false'} }},"
        )
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_subgroup_shuffle_desc agx_apple9_subgroup_shuffle_table[AGX_APPLE9_SUBGROUP_SHUFFLE_COUNT] = {")
    for ename, b0, b1 in SUBGROUP_SHUFFLE_MODES:
        lines.append(f"   [{ename}] = {{ .b0 = {b0:#04x}, .b1 = {b1:#04x} }},")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_workgroup_barrier_desc agx_apple9_workgroup_barrier_table[AGX_APPLE9_WORKGROUP_BARRIER_COUNT] = {")
    for ename, b3, b4 in WORKGROUP_BARRIER_MODES:
        lines.append(f"   [{ename}] = {{ .b3 = {b3:#04x}, .b4 = {b4:#04x} }},")
    lines.append("};")
    lines.append("")

    lines.append("const struct agx_apple9_device_fence_desc agx_apple9_device_fence_table[AGX_APPLE9_DEVICE_FENCE_HW_COUNT] = {")
    for ename, b2, b3, b4, mask in DEVICE_FENCE_MODES:
        lines.append(
            f"   [{ename}] = {{ .b2_base = {b2:#04x}, .b3 = {b3:#04x}, .b4 = {b4:#04x}, "
            f".uses_slot_mask = {'true' if mask else 'false'} }},"
        )
    lines.append("};")
    lines.append("")

    # Build dense 5D texture mode table initialized to AGX_APPLE9_TEXTURE_MODE_INVALID.
    tex_grid: list[list[list[list[list[int]]]]] = [
        [[[[TEXTURE_MODE_INVALID, TEXTURE_MODE_INVALID] for _ in range(2)] for _ in range(2)]
         for _ in range(TEX_DIM_COUNT)]
        for _ in range(len(TEX_KIND_NAMES))
    ]
    for kind, dim, shadow, min_lod, base_mode, ofs_mode in TEXTURE_MODES:
        s = 1 if shadow else 0
        m = 1 if min_lod else 0
        if base_mode is not None:
            tex_grid[kind][dim][s][m][0] = base_mode
        if ofs_mode is not None:
            tex_grid[kind][dim][s][m][1] = ofs_mode

    lines.append(
        "const uint16_t agx_apple9_texture_modes"
        "[AGX_APPLE9_TEX_MODE_KIND_COUNT][AGX_APPLE9_TEXTURE_DIM_COUNT][2][2][2] = {"
    )
    for k_idx, k_name in enumerate(TEX_KIND_NAMES):
        lines.append(f"   [{k_name}] = {{")
        for dim in range(TEX_DIM_COUNT):
            cells = []
            for s in range(2):
                s_cells = []
                for m in range(2):
                    v0, v1 = tex_grid[k_idx][dim][s][m]
                    s_cells.append(f"{{ {v0:#06x}, {v1:#06x} }}")
                cells.append("{ " + ", ".join(s_cells) + " }")
            lines.append(f"      [{dim}] = {{ " + ", ".join(cells) + " },")
        lines.append("   },")
    lines.append("};")
    lines.append("")

    fetch_vals = ", ".join(
        f"{TEXTURE_FETCH_MODES.get(d, TEXTURE_MODE_INVALID):#06x}" for d in range(TEX_DIM_COUNT)
    )
    ms_vals = ", ".join(
        f"{TEXTURE_MS_MODES.get(d, TEXTURE_MODE_INVALID):#06x}" for d in range(TEX_DIM_COUNT)
    )
    lines.append(f"const uint16_t agx_apple9_texture_fetch_modes[AGX_APPLE9_TEXTURE_DIM_COUNT] = {{ {fetch_vals} }};")
    lines.append(f"const uint16_t agx_apple9_texture_ms_modes[AGX_APPLE9_TEXTURE_DIM_COUNT] = {{ {ms_vals} }};")
    lines.append("")

    lines.append(
        "const struct agx_apple9_texture_offset_pair "
        "agx_apple9_texture_offset_pairs[AGX_APPLE9_TEXTURE_OFFSET_PAIR_COUNT] = {"
    )
    for mode, ofs_mode in TEXTURE_OFFSET_PAIRS:
        lines.append(f"   {{ .mode = {mode:#06x}, .offset_mode = {ofs_mode:#06x} }},")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-h", type=Path, help="Output generated machine table header path")
    parser.add_argument("--out-c", type=Path, help="Output generated machine table source path")
    parser.add_argument("--out-vir-h", type=Path, help="Output generated VIR table header path")
    parser.add_argument("--out-vir-c", type=Path, help="Output generated VIR table source path")
    parser.add_argument(
        "--check",
        type=Path,
        metavar="MACHINE_H",
        help="Validate schema and check enum sync against agx_apple9_machine.h",
    )
    parser.add_argument(
        "--check-ir",
        type=Path,
        metavar="IR_H",
        help="Validate VIR opcode/sub-enum sync against agx_apple9_ir.h",
    )
    args = parser.parse_args(argv)

    validate_encodings(ENCODINGS)
    validate_vir_tables(ENCODINGS, VIR_OPCODES)

    if args.check is not None:
        check_against_header(ENCODINGS, args.check)

    if args.check_ir is not None:
        check_against_ir_header(VIR_OPCODES, args.check_ir)

    if args.out_h is not None:
        args.out_h.parent.mkdir(parents=True, exist_ok=True)
        args.out_h.write_text(render_header(ENCODINGS), encoding="utf-8", newline="\n")

    if args.out_c is not None:
        args.out_c.parent.mkdir(parents=True, exist_ok=True)
        args.out_c.write_text(render_source(ENCODINGS), encoding="utf-8", newline="\n")

    if args.out_vir_h is not None:
        args.out_vir_h.parent.mkdir(parents=True, exist_ok=True)
        args.out_vir_h.write_text(render_vir_header(VIR_OPCODES), encoding="utf-8", newline="\n")

    if args.out_vir_c is not None:
        args.out_vir_c.parent.mkdir(parents=True, exist_ok=True)
        args.out_vir_c.write_text(render_vir_source(VIR_OPCODES), encoding="utf-8", newline="\n")

    if (args.check is None and args.check_ir is None and
            args.out_h is None and args.out_c is None and
            args.out_vir_h is None and args.out_vir_c is None):
        parser.error("specify at least one output or check option")

    return 0


if __name__ == "__main__":
    sys.exit(main())
