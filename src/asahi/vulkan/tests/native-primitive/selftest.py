#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Synthetic oracle checks; never device evidence."""
import argparse,hashlib,importlib.util,json,os,struct,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def module(name,file):
    s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
run=module('runner',root/'run-one.py');analyze=module('analyzer',root/'analyze.py')
before=dict(os.environ);assert 'HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING' not in run.environment(Path('/dummy'))
assert run.environment(Path('/dummy'),True,True)['HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING']=='1';assert dict(os.environ)==before
checks=2;arms=json.loads((root/'ARMS.json').read_text())['arms']
for arm in arms:
    color=(root/'expected'/(arm['name']+'.rgba')).read_bytes();depth=(root/'expected'/(arm['name']+'.depth')).read_bytes()
    assert len(color)==len(depth)==16384 and sum(bool(color[i]) for i in range(3,len(color),4))==arm['covered']
    if arm['name']=='CULL_TRUE':assert color==bytes(16384) and depth==struct.pack('<f',1)*4096
    try:analyze.verify(arm,root,a.dry/'host'/(arm['name']+'.vert'),a.dry/'host'/(arm['name']+'.mesh'))
    except ValueError:pass
    else:raise AssertionError('dry run counted as GPU evidence')
    checks+=2
assert (root/'expected/CULL_FALSE.rgba').read_bytes()==(root/'expected/CONTROL.rgba').read_bytes()
arm=arms[0]
with tempfile.TemporaryDirectory(prefix='primitive-oracle-synthetic-') as t:
    dirs=[]
    for stage in ['vert','mesh']:
        d=Path(t)/stage;d.mkdir();dirs.append(d)
        (d/'RUN.json').write_text(json.dumps(dict(arm=arm['name'],stage=stage,no_gpu_execution=False,returncode=0,fence_success=True,synthetic_calibration=True,driver_sha256=hashlib.sha256((root/'lib-arm/libvulkan_asahi.so').read_bytes()).hexdigest())))
        for name,ext in [('color.bin','.rgba'),('depth.bin','.depth')]:d.joinpath(name).write_bytes((root/'expected'/(arm['name']+ext)).read_bytes())
    analyze.verify(arm,root,*dirs);checks+=1
    for name in ['color.bin','depth.bin']:
        path=dirs[1]/name;original=path.read_bytes();bad=bytearray(original);bad[4096]^=1;path.write_bytes(bad)
        try:analyze.verify(arm,root,*dirs)
        except ValueError:pass
        else:raise AssertionError('readback corruption accepted')
        path.write_bytes(original);checks+=1
print(json.dumps(dict(host_only=True,synthetic_oracle_calibration=True,checks=checks,hardware_runs=0),indent=2))
