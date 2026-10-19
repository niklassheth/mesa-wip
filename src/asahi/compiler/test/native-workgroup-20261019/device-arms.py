#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Describe whole-program qualification arms; does not access a device."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--fixtures', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
fixtures = a.fixtures.resolve()

def program(stage, size, probe):
    name = f'{stage}_T{size}_{size}x1x1_{"PROBE" if probe else "PLAIN"}'
    meta = json.loads((fixtures / f'{name}.json').read_text())
    code = (fixtures / f'{name}.bin').read_bytes()
    assert len(code) == meta['binary_size'] and meta['scratch'] == 0
    assert meta['physical_size'] == [16, 1, 1] and meta['body_only']
    return dict(name=name, metadata=meta, binary=f'{name}.bin',
                sha256=hashlib.sha256(code).hexdigest(),
                expected_q=f'{name}.expected-q.bin')

rows = [('CONTROL_OPEN', 16, False, 16, False)]
for stage in ['MESH', 'OBJECT']:
    for size in [1, 3, 16]:
        for probe in ([False, True] if size != 16 else [True]):
            rows.append((f'{stage}_T{size}_{"PROBE" if probe else "PLAIN"}',
                         size if stage == 'OBJECT' else 16, probe if stage == 'OBJECT' else False,
                         size if stage == 'MESH' else 16, probe if stage == 'MESH' else False))
rows += [('BOTH_T1_PROBE', 1, True, 1, True),
         ('BOTH_T3_PROBE', 3, True, 3, True),
         ('CONTROL_CLOSE', 16, False, 16, False)]
arms = []
for name, osize, op, msize, mp in rows:
    arms.append(dict(name=name, object=program('OBJECT', osize, op),
                     mesh=program('MESH', msize, mp), grid=[1, 1, 1],
                     physical_object_threads=16, physical_mesh_threads=16,
                     expected_pixels='512 red RGBA(255,0,0,255), 3584 clear RGBA(0,0,0,0), exact source coverage',
                     expected_class_drops=0))
manifest = dict(scope='Qualification plan and authored binaries; no device result',
                fixture_directory=str(fixtures), fresh_process_and_vm_per_arm=True,
                whole_programs=True, timeout_seconds=20, memory_max='2G',
                memory_swap_max=0, stop_on_gpu_error=True,
                on_timeout_or_gpu_error='SIGSTOP the whole process group/unit; retain VM/BOs until controller reset',
                t32=dict(status='unsupported', action='host rejection only; no GPU submission'), arms=arms)
a.out.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{len(arms)} arms written to {a.out}')
