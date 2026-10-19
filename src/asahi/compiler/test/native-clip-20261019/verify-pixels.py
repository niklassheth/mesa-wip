#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Compare a single clip-arm readback with the authored exact pixel oracle."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifacts', required=True, type=Path)
p.add_argument('--arm', required=True)
p.add_argument('--pixels', required=True, type=Path)
a = p.parse_args()
m = json.loads((a.artifacts / 'ARMS.json').read_text())
row, = [r for r in m['arms'] if r['name'] == a.arm]
expected = (a.artifacts / row['image']).read_bytes()
assert hashlib.sha256(expected).hexdigest() == row['image_sha256']
actual = a.pixels.read_bytes()
if len(actual) != len(expected):
    raise SystemExit(f'wrong image size: {len(actual)}, expected {len(expected)}')
mismatch = [i // 4 for i in range(0, len(actual), 4)
            if actual[i:i+4] != expected[i:i+4]]
print(json.dumps(dict(arm=a.arm, pixels_exact=not mismatch,
    mismatch_pixels=len(mismatch), first_mismatch=mismatch[:8],
    expected_red_pixels=row['red_pixels'],
    data_checked=False, retirement_checked=False)))
raise SystemExit(bool(mismatch))
