#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the bounded NENVNMNN transition probe's nonexecuting DRM captures.

This deliberately checks one fixed fixture, not arbitrary VDM streams. It
verifies submission construction, attachment programs and dependency fields;
only running transitions on hardware can validate pixels or GPU ordering.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import struct


spec = importlib.util.spec_from_file_location(
    'capture', Path(__file__).resolve().parents[1] / 'check-native-mesh-submit.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
require = capture.require


def check(directory, native):
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
            # This recorder submits one hardware command per ioctl. Zero
            # waits for previous ioctls on the respective subqueue; 0xffff
            # would omit that dependency. Do not interpret these as flags.
            require(vdm == cdm == 0, f'{path}: missing previous-submit barrier')
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
            if row['native']:
                words = struct.unpack('<40I', capture.read_gpu(directory, index, address, 160))
                require(words[32] == 0x70000600 and words[39] == 0xc0000000,
                        f'{path}: expected one direct draw followed by stream stop')
                row.update(native_stream_bytes=160, native_draws=1)
            renders.append(row)
    expected_kinds = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1] if native else [1, 0, 1]
    expected_native = [True, False, True, False, True, False, True, True] if native else [False]
    require([r['kind'] for r in commands] == expected_kinds,
            f'{directory}: unexpected compute/render sequence')
    require([r['native'] for r in renders] == expected_native,
            f'{directory}: unexpected native/emulated/vertex/meta sequence')
    require(len({r['submission'] for r in commands}) == len(commands),
            f'{directory}: checker requires one command per recorded ioctl')
    require(renders[0]['bg'] != renders[0]['partial_bg'],
            f'{directory}: first render must clear')
    for row in renders[1:]:
        require(row['bg'] == row['partial_bg'] == renders[0]['partial_bg'],
                f'{directory}: continuation must load stored attachments')
    for row in renders[:-1]:
        require(row['eot'] == row['partial_eot'] == renders[0]['partial_eot'],
                f'{directory}: intermediate render must store partial attachments')
    require(renders[-1]['eot'] != renders[-1]['partial_eot'],
            f'{directory}: final render must use final-store program')
    return dict(directory=str(directory), native=native, commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--emulated', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(result='pass', gpu_execution=False,
                          sequence='NENVNMNN', captures=[
                              check(args.native, True),
                              check(args.emulated, False)]), indent=2))


if __name__ == '__main__':
    main()
