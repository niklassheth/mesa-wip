#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Synthetic oracle calibration only; never creates device evidence."""
import argparse,json,tempfile
from pathlib import Path
from analyze import compare
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
checks=[]
with tempfile.TemporaryDirectory(prefix='native-limits-oracle-') as tmp:
    d=Path(tmp);name='T64_LOGICAL'
    meta=dict(arm=name,stage='mesh',fence_success=True,no_gpu_execution=False,synthetic_fixture=True)
    (d/'RUN.json').write_text(json.dumps(meta))
    for f,e in [('color.bin','rgba'),('depth.bin','depth'),('q.bin','q')]:
        (d/f).write_bytes((a.root/'expected'/(name+'.'+e)).read_bytes())
    assert all(v['mismatched_bytes']==0 for v in compare(a.root,name,'mesh',d).values())
    checks.append('accept exact synthetic source oracle')
    for label,file,offset in [('missing invocation','q.bin',32*64),
            ('wrong subgroup ID','q.bin',32*64+9*4),
            ('bad barrier neighbor','q.bin',15*4),
            ('unused record corruption','q.bin',100*64+4),
            ('tail guard corruption','q.bin',16383),
            ('pixel mismatch','color.bin',0),('depth mismatch','depth.bin',0)]:
        path=d/file;original=path.read_bytes();bad=bytearray(original);bad[offset]^=1;path.write_bytes(bad)
        assert compare(a.root,name,'mesh',d)[file]['mismatched_bytes']>0
        path.write_bytes(original);checks.append('reject '+label)
    for key,value in [('no_gpu_execution',True),('fence_success',False),('arm','T32_IDS')]:
        invalid={**meta,key:value};(d/'RUN.json').write_text(json.dumps(invalid))
        try:compare(a.root,name,'mesh',d)
        except AssertionError:pass
        else:raise AssertionError('accepted invalid run metadata '+key)
        checks.append('reject metadata '+key)
result=dict(synthetic_only=True,no_gpu_execution=True,checks=checks,passed=len(checks))
print(json.dumps(result,indent=2))
