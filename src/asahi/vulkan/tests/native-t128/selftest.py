#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Synthetic T128/partial oracle calibration; never device evidence."""
import argparse,json,struct,tempfile
from pathlib import Path
from analyze import compare
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
checks=[]
for name,total in [('T128_LOGICAL',128),('T127_PARTIAL',127)]:
    q=(a.root/'expected'/(name+'.q')).read_bytes()
    words=struct.unpack('<4096I',q)
    assert words[96*16+11]==4
    assert words[96*16+12]==(1<<(total-96))-1
    assert words[96*16+13]==sum(range(96,total))
    assert words[96*16+14]==96
    assert words[(total-1)*16]==1
    if total==127:assert q[127*64:128*64]==(a.root/'expected/initial.q').read_bytes()[127*64:128*64]
    checks.append(name+': independent final subgroup/unused-record arithmetic')
    with tempfile.TemporaryDirectory(prefix='native-t128-oracle-') as tmp:
        d=Path(tmp);meta=dict(arm=name,stage='mesh',fence_success=True,no_gpu_execution=False,synthetic_fixture=True)
        (d/'RUN.json').write_text(json.dumps(meta))
        for f,e in [('color.bin','rgba'),('depth.bin','depth'),('q.bin','q')]:
            (d/f).write_bytes((a.root/'expected'/(name+'.'+e)).read_bytes())
        assert all(v['mismatched_bytes']==0 for v in compare(a.root,name,'mesh',d).values())
        checks.append(name+': accept synthetic exact oracle')
        for label,file,offset in [('last invocation','q.bin',(total-1)*64),
                ('wrong last ballot','q.bin',96*64+12*4),
                ('bad cross-subgroup read','q.bin',15*4),
                ('tail guard','q.bin',16383),('pixel','color.bin',0),('depth','depth.bin',0)]:
            path=d/file;original=path.read_bytes();bad=bytearray(original);bad[offset]^=1;path.write_bytes(bad)
            assert compare(a.root,name,'mesh',d)[file]['mismatched_bytes']>0
            path.write_bytes(original);checks.append(name+': reject '+label)
        for key,value in [('no_gpu_execution',True),('fence_success',False),('arm','T32_IDS')]:
            (d/'RUN.json').write_text(json.dumps({**meta,key:value}))
            try:compare(a.root,name,'mesh',d)
            except AssertionError:pass
            else:raise AssertionError('accepted invalid run metadata '+key)
            checks.append(name+': reject metadata '+key)
print(json.dumps(dict(synthetic_only=True,no_gpu_execution=True,checks=checks,passed=len(checks)),indent=2))
