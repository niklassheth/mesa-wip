#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Independent oracle invariants and readback/boot-gate mutation checks (no GPU)."""
import argparse, hashlib, importlib.util, json, os, struct, tempfile
from fractions import Fraction as F
from pathlib import Path
from arms import positions, oracle, bits
from analyze import verify

def main(root):
    source=Path(__file__).with_name('run-one.py');spec=importlib.util.spec_from_file_location('runner',source)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    n=0
    for line,accepted in [('',False),('asahi_m4.native_mesh=0',False),('asahi_m4.native_mesh=1',True),
        ('xasahi_m4.native_mesh=1',False),('asahi_m4.native_mesh=1 asahi_m4.native_mesh=0',False),
        ('asahi_m4.native_mesh=0 asahi_m4.native_mesh=1',True)]:
        try:runner.require_native(line);actual=True
        except ValueError:actual=False
        assert actual==accepted;n+=1
    before=dict(os.environ);runner.environment(Path('/dummy/icd.json'));assert dict(os.environ)==before;n+=1
    for kind in ['CONTROL','ZERO','NEAR','FAR','ONE','NEAR_OUT','FAR_OUT','CROSS','RECIP_W','SMOOTH']:
        a=oracle(positions(kind,False),False,kind=='SMOOTH',kind=='RECIP_W')
        b=oracle(positions(kind,True),True,kind=='SMOOTH',kind=='RECIP_W')
        assert a==b;n+=1
        if kind=='NEAR':
            covered=[i for i in range(4096) if a[0][4*i+3]]
            assert len(covered)==496 and all(struct.unpack_from('<I',a[1],4*i)[0]==bits(F(1,1024)) for i in covered);n+=1
    arm=next(x for x in json.loads((root/'ARMS.json').read_text())['arms'] if x['name']=='RECIP_W_ZERO_ONE')
    original=oracle(positions('RECIP_W',False),False,False,True)
    wrong=positions('RECIP_W',False)
    for v in wrong:v[3]=F(1)
    assert oracle(wrong,False,False,True)[:2]!=original[:2];n+=1
    # Synthetic positive fixture is only a unit-test input, never saved as GPU evidence.
    with tempfile.TemporaryDirectory() as tmp:
        dirs=[Path(tmp)/x for x in ['vs','mesh']]
        for stage,d in zip(['vert','mesh'],dirs):
            d.mkdir();meta=dict(arm=arm['name'],stage=stage,no_gpu_execution=False,returncode=0,fence_success=True,
                driver_sha256=hashlib.sha256((root/'lib-arm/libvulkan_asahi.so').read_bytes()).hexdigest())
            (d/'RUN.json').write_text(json.dumps(meta));(d/'color.bin').write_bytes(original[0]);(d/'depth.bin').write_bytes(original[1])
        assert verify(arm,root,*dirs)['hardware_pass'];n+=1
        for filename in ['color.bin','depth.bin','RUN.json']:
            saved=(dirs[1]/filename).read_bytes()
            if filename=='RUN.json':
                meta=json.loads(saved);meta['no_gpu_execution']=True;bad=json.dumps(meta).encode()
            else:bad=bytes([saved[0]^1])+saved[1:]
            (dirs[1]/filename).write_bytes(bad)
            try:verify(arm,root,*dirs)
            except ValueError:pass
            else:raise AssertionError(filename+' mutation accepted')
            (dirs[1]/filename).write_bytes(saved);n+=1
        # Equal bad images must fail the independent source oracle too.
        for d in dirs:(d/'depth.bin').write_bytes(b'\0'*16384)
        try:verify(arm,root,*dirs)
        except ValueError:pass
        else:raise AssertionError('equal incorrect depth accepted')
        n+=1
    print(n,'host-only checks passed; synthetic fixtures are not GPU results')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
