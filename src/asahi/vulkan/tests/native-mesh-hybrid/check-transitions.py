#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check bounded hybrid transition probes' nonexecuting DRM captures.

This deliberately checks one fixed fixture, not arbitrary VDM streams. It
verifies submission construction, attachment programs and dependency fields;
only running transitions on hardware can validate pixels or GPU ordering.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


spec = importlib.util.spec_from_file_location(
    'capture', Path(__file__).resolve().parents[1] / 'check-native-mesh-submit.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
require = capture.require


def literal32(data, register):
    # Our launch constructor's two literal instructions, not a disassembler.
    require(len(data) == 8 and data[0] == (register << 4) | 0xc
            and data[1] & 0x80 and data[2] == 2
            and not data[3] & 1 and not data[4] & ~0x1e
            and not data[5] & ~0xc and not data[7] & ~0xf,
            'unexpected owned launch address encoding')
    return ((data[1] & 127) | (data[3] << 24) | (data[4] << 6)
            | (data[5] << 9) | (data[6] << 13) | (data[7] << 21))


def constant_table(directory, index, words):
    # G16's fixed USC base and the root layout are from our source. The mesh
    # constant pointer offset (1856) is checked with offsetof in host evidence.
    # Follow the actual launch -> arguments -> resources -> root -> data.
    launch = (1 << 40) + words[8] * 64
    header = capture.read_gpu(directory, index, launch, 16)
    arguments = literal32(header[:8], 2) | (literal32(header[8:], 3) << 32)
    resources, = struct.unpack('<Q', capture.read_gpu(directory, index, arguments + 16, 8))
    root, = struct.unpack('<Q', capture.read_gpu(directory, index, resources, 8))
    self_pointer, = struct.unpack('<Q', capture.read_gpu(directory, index, root, 8))
    require(self_pointer == root, 'mesh resource does not point to its descriptor root')
    address, = struct.unpack('<Q', capture.read_gpu(directory, index, root + 1856, 8))
    expected = b''.join(struct.pack('<4f', *(
        (0., 0., 1., 1.) if i == 37 else
        ((i + 1) / 128., (i * 29 % 64 + 1) / 128.,
         (i * 11 % 64 + 1) / 128., 1.))) for i in range(64))
    data = capture.read_gpu(directory, index, address, len(expected))
    require(data == expected, 'mesh constant-data root does not reach the complete immutable table')
    return dict(launch=hex(launch), arguments=hex(arguments), resources=hex(resources),
                root=hex(root), address=hex(address), bytes=len(data), selected_index=37,
                sha256=hashlib.sha256(data).hexdigest())


def check(directory, native, mode='plain', constants=False):
    run = json.loads((directory.parent / 'RUN.json').read_text())
    sequence = {'plain': 'NENVNMNN', 'timestamp': 'N',
                'merged': 'NN', 'deferred': 'NN'}[mode]
    require(run['native'] == str(int(native)) and run['sequence'] == sequence
            and run.get('mode', 'plain') == mode
            and run.get('constant_table', False) == constants
            and run['gpu_execution'] is False and run['queue_submission'] is True,
            f'{directory}: capture metadata does not match the requested fixture')
    # drm_asahi_cmd_render offsets, verified with offsetof against the tree's
    # shared UAPI. The command header is <type,size,vdm_barrier,cdm_barrier>.
    commands = []
    renders = []
    for path in sorted(directory.glob('*.commands.bin')):
        data, pos = path.read_bytes(), 0
        index = path.name.split('.')[0]
        while pos < len(data):
            require(pos + 8 <= len(data), f'{path}: truncated header')
            kind, size, vdm, cdm = struct.unpack_from('<4H', data, pos)
            pos += 8
            require(size and pos + size <= len(data), f'{path}: invalid size')
            body = data[pos:pos + size]
            pos += size
            require(kind in (0, 1), f'{path}: unexpected command type {kind}')
            # One hardware command per ioctl: zero waits for the preceding
            # submission on that subqueue; 0xffff omits the dependency.
            # Check the required producer/consumer edges below.
            row = dict(submission=index, kind=kind, size=size,
                       vdm_barrier=vdm, cdm_barrier=cdm)
            commands.append(row)
            if kind == 1:
                require(size == 64, f'{path}: unexpected compute UAPI size')
                continue
            require(size == 240, f'{path}: unexpected render UAPI size')
            address, = struct.unpack_from('<Q', body, 8)
            prefix, = struct.unpack('<I', capture.read_gpu(directory, index, address, 4))
            require(prefix in (0x40000d2c, 0x4000002e),
                    f'{path}: unexpected producer prefix {prefix:#x}')
            row.update(prefix=hex(prefix), native=prefix == 0x40000d2c)
            require(struct.unpack_from('<3H', body, 148) == (64, 64, 1)
                    and body[158] == 1, f'{path}: unexpected framebuffer')
            programs = dict(zip(('bg', 'eot', 'partial_bg', 'partial_eot'),
                                struct.unpack_from('<4Q', body, 168)))
            row.update({key: hex(value) for key, value in programs.items()})
            require(all(programs.values()), f'{path}: missing attachment program')
            timestamps = struct.unpack_from('<8I', body, 208)
            if mode == 'timestamp':
                require(not any(timestamps[:6]) and timestamps[6]
                        and timestamps[7] % 8 == 0,
                        f'{path}: expected only a fragment-end timestamp')
                row['fragment_end_timestamp'] = dict(handle=timestamps[6], offset=timestamps[7])
            else:
                require(not any(timestamps), f'{path}: unexpected timestamp')
            if row['native']:
                words = struct.unpack('<40I', capture.read_gpu(directory, index, address, 160))
                require(words[32] == 0x70000600 and words[39] == 0xc0000000,
                        f'{path}: expected one direct draw followed by stream stop')
                row.update(native_stream_bytes=160, native_draws=1)
                if constants:
                    row['constant_data'] = constant_table(directory, index, words)
            renders.append(row)
    if mode == 'plain':
        expected_kinds = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1] if native else [1, 0, 1]
        expected_native = [True, False, True, False, True, False, True, True] if native else [False]
    elif mode == 'timestamp':
        # Query reset before graphics and image/query copies after graphics.
        expected_kinds, expected_native = [1, 0, 1], [native]
    elif mode == 'merged':
        # The emulation producer prevents merging; native can merge the API
        # renders, then splits its hardware stream before the next draw.
        expected_kinds = [0, 0, 1] if native else [1, 0, 1, 0, 1]
        expected_native = [native, native]
        if native:
            log = (directory.parent / 'stdout.log').read_text()
            require(log.count('Merging render pass into the previous render') == 1,
                    f'{directory}: no evidence of the merged BeginRendering path')
    else:
        require(mode == 'deferred', f'unknown fixture mode {mode}')
        expected_kinds = [0, 0, 1] if native else [1, 0, 1]
        expected_native = [True, True] if native else [False]
    require([r['kind'] for r in commands] == expected_kinds,
            f'{directory}: unexpected compute/render sequence')
    require([r['native'] for r in renders] == expected_native,
            f'{directory}: unexpected native/emulated/vertex/meta sequence')
    if native and mode == 'timestamp':
        # The render waits for query reset; its post-graphics copy waits for
        # both the reset and the render's timestamp write.
        barriers = [(0xffff, 0xffff), (0xffff, 0), (0, 0)]
    elif native and mode in ('merged', 'deferred'):
        barriers = [(0xffff, 0xffff), (0, 0xffff), (0, 0xffff)]
    else:
        # Emulated mesh work retains the conservative two-subqueue joins.
        barriers = [(0, 0)] * len(commands)
    require([(r['vdm_barrier'], r['cdm_barrier']) for r in commands] == barriers,
            f'{directory}: unexpected render/compute dependency edges')
    require(len({r['submission'] for r in commands}) == len(commands),
            f'{directory}: checker requires one command per recorded ioctl')
    require(renders[0]['bg'] != renders[0]['partial_bg'],
            f'{directory}: first render must clear')
    for row in renders[1:]:
        if mode != 'merged' or native:
            require(row['bg'] == row['partial_bg'],
                    f'{directory}: continuation must load stored attachments')
        if mode != 'merged':
            require(row['bg'] == renders[0]['partial_bg'],
                    f'{directory}: continuation changed the attachment load program')
    if mode == 'merged':
        # EndRendering installs the API STORE program before the second
        # BeginRendering. Both API passes request STORE in this fixture.
        require(all(row['eot'] != row['partial_eot'] for row in renders),
                f'{directory}: both API renders must use their final-store programs')
    else:
        for row in renders[:-1]:
            require(row['eot'] == row['partial_eot'] == renders[0]['partial_eot'],
                    f'{directory}: intermediate render must store partial attachments')
    require(renders[-1]['eot'] != renders[-1]['partial_eot'],
            f'{directory}: final render must use final-store program')
    return dict(directory=str(directory), native=native, mode=mode,
                constant_table=constants, commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--emulated', type=Path, required=True)
    parser.add_argument('--mode', choices=('plain', 'timestamp', 'merged', 'deferred'), default='plain')
    parser.add_argument('--constant-table', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dict(result='pass', gpu_execution=False,
                          mode=args.mode, sequence={'plain': 'NENVNMNN', 'timestamp': 'N',
                                                   'merged': 'NN', 'deferred': 'NN'}[args.mode],
                          captures=[check(args.native, True, args.mode, args.constant_table),
                                    check(args.emulated, False, args.mode, args.constant_table)]), indent=2))


if __name__ == '__main__':
    main()
