#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Exact owned readback oracle. Work identity/retirement is audited separately.

Inputs in --readback: source.bin (actual independently recorded draw state),
geometry.bin, list.bin, root.bin, scene.bin, pixels.bin, object-grid.bin,
object-q.bin, mesh-q.bin, class-drops.bin. Record buffers start at zero; each
separate 2048-byte Q starts at 0xa5 with its first uint32 zero.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True, type=Path)
p.add_argument('--arm', required=True)
p.add_argument('--readback', required=True, type=Path)
p.add_argument('--decoder', required=True, type=Path)
p.add_argument('--fixtures', type=Path)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
arm = next(x for x in m['arms'] if x['name'] == a.arm)
f = (a.fixtures or Path(m['fixture_directory'])).resolve()
r = a.readback

def require(ok, text):
    if not ok:
        raise SystemExit(text)

for stage in ['object', 'mesh']:
    require((r / f'{stage}-q.bin').read_bytes() ==
            (f / arm[stage]['expected_q']).read_bytes(), f'{stage} Q differs (including inactive lanes/guards)')
require((r / 'object-grid.bin').read_bytes() == struct.pack('<3I', 1, 1, 1), 'grid differs')
require((r / 'class-drops.bin').read_bytes() == bytes(4), 'triangle outside the supported formation class')
source = (r / 'source.bin').read_bytes()
require(len(source) == 128, 'source.bin must hold 32 little-endian words')
reference = (f / (arm['mesh']['name'] + '.reference-source.bin')).read_bytes()
require(source[:60] == reference[:60], 'positions or primitive indices differ from authored source')
require(source[92:116] == reference[92:116], 'viewport differs from the 64x64 source fixture')
require((r / 'pixels.bin').read_bytes() ==
        (f / (arm['mesh']['name'] + '.expected-pixels.bin')).read_bytes(), 'pixels differ')
cmd = [str(a.decoder.resolve()), str(r / 'source.bin'),
       *[str(r / (name + '.bin')) for name in ['geometry', 'list', 'root', 'scene', 'pixels']]]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
require(result.returncode == 0, 'DATA decoder failed: ' + result.stderr)
print(json.dumps(dict(arm=a.arm, q_exact=True, grid_exact=True, class_drops=0,
                     data=json.loads(result.stdout), retirement_checked=False)))
