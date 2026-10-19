#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Decode retained D4 data packets against the authored G16 PPP serializer.

Offsets refer to its canonical single-viewport record, not arbitrary PPP
packets. Shader code is neither opened nor decoded by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
rows = []
for name in ['D4_CONTROL', 'D4_SCENE_OWNED', 'D4_CURSOR_OWNED', 'D4_OUTPUT_OWNED']:
    log = a.results / name / 'log.txt'
    matches = re.findall(r'^NATIVE_MESH_PPP size=256 bytes=([0-9a-f]+)$',
                         log.read_text(), re.M)
    assert len(matches) == 1
    b = bytes.fromhex(matches[0])
    assert len(b) == 256
    u = lambda off: struct.unpack_from('<I', b, off)[0]
    f = lambda off: struct.unpack_from('<f', b, off)[0]
    expected = {
        'front_control': (0x74, 0x10200), 'fragment_control_2': (0x90, 0x200),
        'w_clamp_and_cull_2_header': (0xa0, 0x410000),
        'w_clamp_value': (0xa4, 0), 'cull_2': (0xa8, 0x80),
        'cull_header': (0xac, 0x200000), 'cull': (0xb0, 0x10480),
        'viewport_header': (0xc0, 0xc00), 'region_x': (0xc4, 0x80000001),
        'region_y': (0xc8, 1), 'viewport_control': (0xcc, 0),
        'scissor_header': (0xf0, 0x100), 'scissor_bias_indices': (0xf4, 0),
    }
    fields = {}
    for field, (off, value) in expected.items():
        assert u(off) == value, (name, field, hex(u(off)), hex(value))
        fields[field] = dict(offset=hex(off), actual=hex(u(off)),
                             source_model=hex(value), matches=True)
    transform = [f(off) for off in range(0xd0, 0xe8, 4)]
    assert transform == [32, 32, 32, -32, 0, 1]
    rows.append(dict(arm=name, log=str(log.resolve()),
                     log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                     ppp_sha256=hashlib.sha256(b).hexdigest(), fields=fields,
                     viewport_translate_scale=transform,
                     depth_clip=bool(u(0xb0) & (1 << 10)),
                     depth_clamp=bool(u(0xb0) & (1 << 11)),
                     tri_merging_disabled=bool(u(0x90) & (1 << 26)),
                     pixel_order=bool(u(0x90) & (1 << 23))))
a.out.write_text(json.dumps(dict(
    scope='Retained H1 native PPP vs ordinary G16 source serializer, not a fresh VS GPU capture',
    global_ppp_control_from_hk_queue_source='0x202',
    g16_cull_2='0x80 from shared serializer; do not replace with generic older-path clamp_w bit 5',
    arms=rows), indent=2) + '\n')
print(f'{len(rows)} retained native PPP packets match the ordinary G16 source model')
