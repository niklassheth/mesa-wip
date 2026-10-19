#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Pin complete compiler exports to the clip qualification matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--object-fixtures', type=Path, required=True)
a = p.parse_args()
h = a.artifacts
m = json.loads((h / 'ARMS.json').read_text())
object_name = 'OBJECT_T16_16x1x1_PLAIN'
for extension in ['bin', 'json']:
    shutil.copy2(a.object_fixtures / (object_name + '.' + extension),
                 h / 'fixtures-arm' / (object_name + '.' + extension))
def binary(name):
    filename = name + '.bin'
    data = (h / 'fixtures-arm' / filename).read_bytes()
    meta = json.loads((h / 'fixtures-arm' / (name + '.json')).read_text())
    assert len(data) == meta['binary_size'] and meta['scratch'] == 0
    return dict(file='fixtures-arm/' + filename,
                sha256=hashlib.sha256(data).hexdigest(), metadata=meta)
for row in m['arms']:
    row['mesh'] = binary(row['native_program'])
    row['object'] = binary(object_name)
    row['not_a_device_result'] = True
(h / 'ARMS.json').write_text(json.dumps(m, indent=2) + '\n')
print(f"Pinned whole object/mesh programs for {len(m['arms'])} arms")
