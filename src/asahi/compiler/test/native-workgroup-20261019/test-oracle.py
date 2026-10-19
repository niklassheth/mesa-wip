#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Exercise the device readback checker with source-derived positives/mutations."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--artifacts', type=Path, required=True)
a = p.parse_args()
h = a.artifacts.resolve()
m = json.loads((h / 'ARMS.json').read_text())
f = Path(m['fixture_directory'])
results = []
for arm in m['arms']:
    r = h / 'oracle-selftest' / arm['name']
    r.mkdir(parents=True, exist_ok=True)
    for stage in ['object', 'mesh']:
        (r / f'{stage}-q.bin').write_bytes((f / arm[stage]['expected_q']).read_bytes())
    for name, ext in [('source', 'reference-source'), ('pixels', 'expected-pixels'),
                      *[(name, f'expected-data{i}') for i, name in enumerate(['geometry', 'list', 'root', 'scene'])]]:
        (r / f'{name}.bin').write_bytes((f / (arm['mesh']['name'] + '.' + ext + '.bin')).read_bytes())
    (r / 'object-grid.bin').write_bytes(struct.pack('<3I', 1, 1, 1))
    (r / 'class-drops.bin').write_bytes(bytes(4))
    cmd = [sys.executable, str(Path(__file__).with_name('verify-device.py')),
           '--manifest', str(h / 'ARMS.json'), '--arm', arm['name'],
           '--readback', str(r), '--decoder', str(h / 'semantic-host')]
    test = subprocess.run(cmd, capture_output=True, text=True, check=True)
    results.append(json.loads(test.stdout))
    if arm['name'] == 'BOTH_T1_PROBE':
        mutations = [('object-q.bin', 320, 'inactive object lane'),
                     ('mesh-q.bin', 320, 'inactive mesh lane'),
                     ('mesh-q.bin', 284, 'subgroup size'),
                     ('mesh-q.bin', 0, 'atomic count'),
                     ('pixels.bin', 0, 'pixel'), ('geometry.bin', 0, 'DATA header'),
                     ('object-grid.bin', 4, 'grid Y'), ('class-drops.bin', 0, 'class drop')]
        for name, offset, description in mutations:
            path = r / name
            original = path.read_bytes()
            bad = bytearray(original); bad[offset] ^= 1
            try:
                path.write_bytes(bad)
                rejected = subprocess.run(cmd, capture_output=True, text=True)
                assert rejected.returncode != 0, description
                results.append(dict(mutation=description, rejected=True))
            finally:
                path.write_bytes(original)
(h / 'oracle-selftest.json').write_text(json.dumps(results, indent=2) + '\n')
print('14 source-derived positives; 8 mutations rejected')
