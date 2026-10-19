#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
# Reused read-only parser from implicit-cts-20261019/audit-host.py.
"""Audit real native packets and uploaded compiler programs in host-only submits.
Walk UAPI commands and the VDM stream structurally. Never scan for magic words.
"""
import hashlib, json, re, struct
from pathlib import Path

H = Path(__file__).resolve().parent
SOURCE = H.parents[4]
USC_BASE = 1 << 40  # agx_device.c, ordinary G16/G17 USC aperture
CONTEXT_BASE = 0x1000000000  # agx_apple9.h, ordinary render-context aperture

def sha(b):
    return hashlib.sha256(b).hexdigest()

def opcode_names():
    s = (SOURCE / 'src/asahi/compiler/agx_apple9_ir.h').read_text()
    s = s.split('enum agx_apple9_vir_opcode {')[1].split('};')[0]
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    return [x.strip().removeprefix('AGX_APPLE9_VIR_') for x in s.split(',') if x.strip()]

OPCODES = opcode_names()

def programs(log):
    result, current = [], None
    for line in log.splitlines():
        if line.startswith(('APPLE9_ASM_BEGIN', 'APPLE9_PREAMBLE_ASM_BEGIN')):
            assert current is None
            current = dict(stage=int(line.split('stage=')[1]),
                           preamble=line.startswith('APPLE9_PREAMBLE'), code=b'', ops=[], instructions=[])
        elif line == 'APPLE9_ASM_END':
            assert current is not None
            result.append(current)
            current = None
        elif line.startswith('APPLE9_ASM '):
            assert current is not None
            row = dict(x.split('=', 1) for x in line.split()[1:])
            code = bytes.fromhex(row['bytes'])
            assert int(row['offset']) == len(current['code'])
            current['code'] += code
            current['ops'].append(OPCODES[int(row['op'])])
            current['instructions'].append(row)
    assert current is None
    return result

class Memory:
    def __init__(self, directory, index):
        self.directory = directory
        self.bindings = list(struct.iter_unpack('<IIQQQ', (directory / f'{index}.bindings.bin').read_bytes()))
        self.bos = {b['handle']: b for b in json.loads((directory / f'{index}.bos.json').read_text())}
    def read(self, address, size):
        # These captures do not unbind/rebind intersecting live ranges. Reject
        # ambiguity instead of silently choosing a mapping from an event list.
        hits = [b for b in self.bindings if b[4] <= address < b[4] + b[3]]
        assert len(hits) == 1, (hex(address), size, hits)
        flags, handle, offset, length, base = hits[0]
        assert not (flags & 1) and flags & 2 and handle in self.bos
        assert address + size <= base + length and not (flags & 8)
        with (self.directory / self.bos[handle]['file']).open('rb') as f:
            f.seek(offset + address - base)
            b = f.read(size)
        assert len(b) == size
        return b

def literal(b, register):
    assert len(b) == 8 and b[0] == ((register & 15) << 4) | 12
    assert b[2] == ((register >> 4) << 6) | 2 and b[1] & 128
    return ((b[1] & 127) | ((b[4] & 30) << 6) | ((b[5] & 12) << 9) |
            (b[6] << 13) | ((b[7] & 15) << 21) | ((b[3] & 254) << 24))

def launch(memory, handle, object_stage, compiled, channels=4, local_bytes=128):
    address = USC_BASE + 64 * handle
    assert address % 128 == 0
    b = memory.read(address, 256)
    table = literal(b[:8], 2) | (literal(b[8:16], 3) << 32)
    args = struct.unpack('<10Q', memory.read(table, 80))
    assert args[3:9] == (0, 0, 0, 0, 64, 0), args
    assert args[9] and not args[9] & 3
    grid = struct.unpack('<3I', memory.read(args[9], 12))
    assert grid in ((1, 1, 1), (0, 0, 0)), grid
    target_at = 16 + 14 * 5 + 8
    assert b[target_at:target_at + 2] == bytes.fromhex('7701')
    assert b[target_at + 8:target_at + 10] == bytes([2 if object_stage else 6, 0])
    entry = (int.from_bytes(b[target_at + 2:target_at + 8], 'little') - 0x2a) // 2
    assert b[target_at - 8:target_at - 4] == bytes.fromhex('77002a41')
    assert b[target_at - 4:target_at] == struct.pack('<I', (local_bytes << 2) | 0x80)
    frame = b[target_at + 10:target_at + 20]
    assert frame[:3] == bytes.fromhex('f7002a')
    a, bb = struct.unpack_from('<HH', frame, 5)
    assert a == bb == (16 if object_stage else 64)
    publications = (struct.unpack_from('<H', frame, 3)[0] >> 7) * 2
    # This allocates compiler publication slots (object export p0), not
    # argument uniforms. The launch separately transfers all 20 root words.
    assert (2 if object_stage else 0) <= publications <= 1022
    stage = 6 if object_stage else 7
    matches = [p for p in compiled if not p['preamble'] and p['stage'] == stage and
               memory.read(entry, len(p['code'])) == p['code']]
    assert matches, ('uploaded shader does not match compiler listing', stage, hex(entry))
    matches = {sha(p['code']): p for p in matches}
    assert len(matches) == 1
    digest, program = next(iter(matches.items()))
    ops = program['ops']
    if object_stage:
        assert ops.count('OBJECT_CALL') == ops.count('OBJECT_RETURN') == ops.count('OBJECT_RESUME') == 1
        assert ops.count('OBJECT_GRID_EXPORT') == 1
        assert ops.index('HALT') < ops.index('OBJECT_GRID_EXPORT') < ops.index('OBJECT_RETURN')
        instructions = program['instructions']
        call = next(r for r in instructions if OPCODES[int(r['op'])] == 'OBJECT_CALL')
        call_bytes = bytes.fromhex(call['bytes'])
        target = int(call['offset']) + int.from_bytes(call_bytes[3:9], 'little', signed=True)
        local_reads = [r for r in instructions if int(r['offset']) >= target and
                       OPCODES[int(r['op'])] == 'GET_SR' and int(r['imm'], 0) & 255 == 0xa7]
        assert local_reads, 'callee must read its own local index'
        for r in local_reads:
            b = bytes.fromhex(r['bytes'])
            assert len(b) == 8 and b[0] & 15 == 4 and b[1] == 0xa7
            assert b[2] & 0x3f == 0x10 and b[3] == 6
            assert b[4:] == bytes([(b[0] & 0xf0) | 3, 0, b[2] & 0xc0, 1]), \
                'local index lacks its zero-extension companion'
    else:
        mesh_ops = [op for op in ops if op.startswith('MESH_')]
        assert mesh_ops == ['MESH_CONTROL', *['MESH_SUBMIT_CHANNEL'] * channels, 'MESH_SUBMIT_SCALAR']
    return dict(address=hex(address), entry=hex(entry), argument_table=hex(table),
                arguments=[hex(x) for x in args], grid=list(grid), frames=[a, bb],
                publications=publications, local_bytes=local_bytes, code_bytes=len(program['code']), code_sha256=digest,
                launch_sha256=sha(memory.read(address, 256)),
                callee_local_index_reads=local_reads if object_stage else [])

def stream(memory, address, compiled, user_components=0):
    components=4+user_components
    local_bytes=1 << (max(128,32+16*components)-1).bit_length()
    visited, draws = set(), []
    for _ in range(1024):
        assert address not in visited, 'VDM cycle'
        visited.add(address)
        header, = struct.unpack('<I', memory.read(address, 4))
        if header == 0xc0000000:
            return draws
        if header == 0x80000000:
            offset, = struct.unpack('<I', memory.read(address + 4, 4))
            address = CONTEXT_BASE + offset
            continue
        assert header == 0x40000d2c, ('unexpected VDM packet', hex(header), hex(address))
        words = struct.unpack('<39I', memory.read(address, 156))
        assert words[:6] == (0x40000d2c, ((components+1)<<8)|components, 0x10, 0, 0, 0x1000000)
        assert words[7] == 0x1000000
        assert words[9:14] == (1, 0, 0x400810, 0, components+1)
        assert words[14] == 0x500
        ppp = words[15]
        offsets = [0, 0x40, 0x5c, 0x70, 0x8c, 0xc0, 0xa0, 0xac]
        counts = [7, 5, 7, 5, 10, 3, 2, 2]
        for i, (off, count) in enumerate(zip(offsets, counts)):
            pointer, control = words[15 + 2*i:17 + 2*i]
            assert pointer == ppp + off and control == count << 8
            memory.read(CONTEXT_BASE + pointer, 4)
        assert words[31] == ppp + 0xf0
        assert words[32] == 0x70000600
        assert words[33:39] == (1, 1, 1, 1, 1, 1)
        mesh = launch(memory, words[8], False, compiled, components, local_bytes)
        obj = launch(memory, words[6], True, compiled)
        assert mesh['arguments'] == obj['arguments']
        ppp_header=struct.unpack('<2I',memory.read(CONTEXT_BASE+ppp,8))
        draws.append(dict(stream=hex(address), state=list(words[:14]), mesh=mesh, object=obj,
                          ppp_address=hex(CONTEXT_BASE+ppp), ppp_header=list(ppp_header)))
        address += 156
    raise AssertionError('VDM packet budget exceeded')
