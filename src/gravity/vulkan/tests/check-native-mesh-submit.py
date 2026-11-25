#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check mesh count ineligibility in real host-recorder submissions.

Run the same one-color CLEAR/STORE render pass with one native mesh draw and
one ordinary triangle draw through the no-op DRM recorder, then pass their
directories here. An optional five-triangle VS control checks accumulation.
The recorder executes no GPU work; this checks construction, not rendering.
The UAPI offsets below are from drm_asahi_cmd_render and drm_asahi_cmd_header.
"""
import argparse
import json
from pathlib import Path
import struct

FEW = 1 << 30
NATIVE_MESH = 1 << 28
PROCESS_EMPTY_TILES = 1 << 1
CONSERVATIVE_FLAGS = PROCESS_EMPTY_TILES
NATIVE_FLAGS = PROCESS_EMPTY_TILES | NATIVE_MESH


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_gpu(directory, index, address, size):
    bindings_path = directory / f'{index}.bindings.bin'
    if bindings_path.exists():
        # These bounded probes never unbind a range before submitting it. Reject
        # ambiguous mappings instead of picking an arbitrary BO from the log.
        bindings = set(struct.iter_unpack(
            '<IIQQQ', bindings_path.read_bytes()))
        hits = [b for b in bindings if b[4] <= address < b[4] + b[3]]
        require(len(hits) == 1, f'{directory}: ambiguous GPU address {address:#x}')
        flags, handle, offset, length, base = hits[0]
        require(flags & 2 and not flags & 9 and address + size <= base + length,
                f'{directory}: invalid mapping')
        with (directory / f'{index}.bo-{handle}.bin').open('rb') as f:
            f.seek(offset + address - base)
            data = f.read(size)
        require(len(data) == size, f'{directory}: short BO read')
        return data
    matches = sorted(directory.glob(f'{index}.bo.*.bin'))
    require(matches, f'{directory}: no BO dump for {index}')
    for path in matches:
        base = int(path.name.split('.bo.')[1].split('.')[0], 16)
        length = path.stat().st_size
        if base <= address and address + size <= base + length:
            with path.open('rb') as f:
                f.seek(address - base)
                data = f.read(size)
            require(len(data) == size, f'{directory}: short BO read')
            return data
    raise ValueError(f'{directory}: GPU address {address:#x} not found in BO dumps')


def walk_vdm_stream(directory, index, start_address):
    """Decode Apple9 VDM packet boundaries, following VDM Stream Links."""
    addr = start_address
    ret_stack = []
    seen = set()
    first_header = None
    has_native_pkt = False
    steps = 0
    while steps < 65536:
        steps += 1
        require((addr, len(ret_stack)) not in seen,
                f'{directory}: cycle in VDM stream at {addr:#x}')
        seen.add((addr, len(ret_stack)))
        hdr, = struct.unpack('<I', read_gpu(directory, index, addr, 4))
        if first_header is None:
            first_header = hdr
        block_type = hdr >> 29
        if block_type == 0:
            # PPP State Update: control word + 32-bit PPP address.
            addr += 8
        elif block_type == 1:
            # VDM Barrier: 1 word.
            addr += 4
        elif block_type == 2:
            if hdr == 0x40000d2c:
                # Native mesh state packet: AGX_APPLE9_MESH_STATE_WORDS (14 words).
                has_native_pkt = True
                addr += 14 * 4
            elif hdr == 0x4000002e:
                # Ordinary direct vertex state envelope: 7 words.
                addr += 7 * 4
            elif hdr == 0x40000001:
                # Primitive restart index state: 2 words.
                addr += 2 * 4
            else:
                raise ValueError(
                    f'{directory}: unexpected VDM state packet {hdr:#x} at {addr:#x}')
        elif block_type == 3:
            if (hdr & 0xf0000000) == 0x70000000:
                # Native mesh dispatch tail: header + 3 global + 3 local words.
                has_native_pkt = True
                addr += 7 * 4
            elif (hdr & 0xffff00ff) == 0x61c40000:
                # Ordinary direct non-indexed draw: 4 words.
                addr += 4 * 4
            elif (hdr & 0xfff800ff) == 0x61f00000:
                # Ordinary direct indexed draw: 7 words.
                addr += 7 * 4
            elif (hdr & 0xffff00ff) == 0x64040000:
                # Ordinary indirect non-indexed draw: 3 words.
                addr += 3 * 4
            elif (hdr & 0xfff800ff) == 0x64300000:
                # Ordinary indirect indexed draw: 6 words.
                addr += 6 * 4
            else:
                raise ValueError(
                    f'{directory}: unexpected VDM draw packet {hdr:#x} at {addr:#x}')
        elif block_type == 4:
            # VDM Stream Link: 2 words, optional With-return call.
            _, target_lo = struct.unpack('<2I', read_gpu(directory, index, addr, 8))
            target_hi = hdr & 0xff
            with_return = (hdr >> 28) & 1
            if with_return:
                require(len(ret_stack) < 16, f'{directory}: VDM link stack overflow')
                ret_stack.append(addr + 8)
            addr = ((target_hi << 32) | target_lo) if target_hi else (
                (addr & ~0xffffffff) | target_lo)
        elif block_type == 6 and hdr == 0xc0000000:
            if ret_stack:
                addr = ret_stack.pop()
            else:
                return first_header, has_native_pkt
        else:
            raise ValueError(
                f'{directory}: unexpected VDM block {hdr:#x} at {addr:#x}')
    raise ValueError(f'{directory}: VDM stream exceeded step limit')


def check(directory, native, expected):
    files = sorted(directory.glob('*.commands.bin'))
    require(files, f'{directory}: no submissions')
    renders = []
    for path in files:
        data, pos = path.read_bytes(), 0
        while pos < len(data):
            require(len(data) - pos >= 8, f'{path}: truncated command header')
            kind, size, _, _ = struct.unpack_from('<4H', data, pos)
            pos += 8
            require(size and pos + size <= len(data), f'{path}: invalid command size')
            if kind == 0:
                require(size == 240, f'{path}: unexpected render UAPI size {size}')
                flags, = struct.unpack_from('<I', data, pos)
                address, = struct.unpack_from('<Q', data, pos + 8)
                header, has_native_pkt = walk_vdm_stream(
                    directory, path.name.split('.')[0], address)
                # Ordinary-prefix renders merged with a native suffix may start
                # with an ordinary VDM packet (0x4000002e), but packet-boundary
                # decoding across linked segments MUST observe a native mesh
                # packet (0x40000d2c / 0x70000xxx) iff native=True.
                valid_headers = (0x40000d2c, 0x4000002e) if native else (0x4000002e,)
                require(header in valid_headers,
                        f'{path}: unexpected producer packet {header:#x}')
                require(has_native_pkt == native,
                        f'{path}: VDM stream native packet presence {has_native_pkt} != expected {native}')
                require(flags == expected,
                        f'{path}: render flags {flags:#010x}, expected {expected:#010x}')
                require(not ((flags & NATIVE_MESH) and (flags & FEW)),
                        f'{path}: NATIVE_MESH and FEW_PRIMITIVES are mutually exclusive')
                renders.append(flags)
            pos += size
    require(len(renders) == 1, f'{directory}: expected one render, found {len(renders)}')
    return {'directory': str(directory), 'native': native, 'flags': hex(renders[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--vertex', type=Path, required=True)
    parser.add_argument('--h1', type=Path,
                        help='Optional original H1 capture; verifies bit-28 flags')
    parser.add_argument('--vertex-five', type=Path)
    args = parser.parse_args()
    rows = []
    if args.h1:
        rows.append(check(args.h1, True, NATIVE_FLAGS))
    # Native mesh renders set DRM_ASAHI_RENDER_NATIVE_MESH (bit 28) and never
    # combine it with FEW_PRIMITIVES (bit 30).
    rows.append(check(args.native, True, NATIVE_FLAGS))
    # The positive control prevents a blanket disabling of the optimization.
    rows.append(check(args.vertex, False, CONSERVATIVE_FLAGS | FEW))
    if args.vertex_five:
        rows.append(check(args.vertex_five, False, CONSERVATIVE_FLAGS))
    print(json.dumps({'result': 'pass', 'gpu_execution': False, 'renders': rows}, indent=2))


if __name__ == '__main__':
    main()
