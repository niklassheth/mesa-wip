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
    # These bounded probes never unbind a range before submitting it. Reject
    # ambiguous mappings instead of picking an arbitrary BO from the log.
    bindings = set(struct.iter_unpack(
        '<IIQQQ', (directory / f'{index}.bindings.bin').read_bytes()))
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


def read_gpu_prefix(directory, prefix, address, max_size):
    matches = sorted(directory.glob(f'{prefix}.bo.*.bin'))
    require(matches, f'{directory}: no BO dump for {prefix}')
    for path in matches:
        base = int(path.name.split('.bo.')[1].split('.')[0], 16)
        length = path.stat().st_size
        if base <= address < base + length:
            with path.open('rb') as f:
                f.seek(address - base)
                return f.read(min(max_size, base + length - address))
    raise SystemExit(f'{directory}: GPU address {address:#x} not found in BO dumps')


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
                vdm = read_gpu_prefix(directory, path.name.split('.')[0], address, 16384)
                require(len(vdm) >= 4, f'{path}: short VDM read')
                words = struct.unpack_from(f'<{len(vdm) // 4}I', vdm)
                header = words[0]
                # Ordinary-prefix renders merged with a native suffix may start
                # with an ordinary VDM packet (0x4000002e), but the stream MUST
                # contain at least one native mesh packet (0x40000d2c) before
                # stream termination (0xc0000000) when native=True, and zero
                # native mesh packets when native=False.
                valid_headers = (0x40000d2c, 0x4000002e) if native else (0x4000002e,)
                require(header in valid_headers,
                        f'{path}: unexpected producer packet {header:#x}')
                has_native_pkt = False
                for w in words:
                    if w == 0x40000d2c:
                        has_native_pkt = True
                    elif w == 0xc0000000:
                        break
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
