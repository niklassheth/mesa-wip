#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Host-only oracle/gate calibration; synthetic files never count as GPU results."""
import argparse, copy, hashlib, importlib.util, json, os, sys, tempfile
from fractions import Fraction as F
from pathlib import Path
from arms import attributes, positions, oracle

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(root):
    here=Path(__file__).resolve().parent
    shared=here if (here/'run-one.py').exists() else here.parent/'native-clip'
    run=module('runner',shared/'run-one.py');analyze=module('analyzer',shared/'analyze.py')
    before=dict(os.environ);assert 'HK_APPLE9_NATIVE_MESH_ATTRIBUTES' not in run.environment(Path('/example'))
    assert run.environment(Path('/example'),True)['HK_APPLE9_NATIVE_MESH_ATTRIBUTES']=='1';assert dict(os.environ)==before
    checks=2
    for cmdline,ok in [('asahi_m4.native_mesh=1',True),('',False),('asahi_m4.native_mesh=0',False),('asahi_m4.native_mesh=1 asahi_m4.native_mesh=0',False)]:
        try:run.require_native(cmdline);actual=True
        except ValueError:actual=False
        assert actual==ok;checks+=1
    arms=json.loads((root/'ARMS.json').read_text())['arms']
    for arm in arms:
        p=positions(arm['kind'],arm['halfz']);attrs=attributes(arm['kind']);color,depth,stats=oracle(p,arm['halfz'],attrs)
        assert color==(root/'expected'/(arm['name']+'.rgba')).read_bytes()
        assert depth==(root/'expected'/(arm['name']+'.depth')).read_bytes()
        twin=oracle(positions(arm['kind'],not arm['halfz']),not arm['halfz'],attrs)
        assert (color,depth)==twin[:2]
        if attrs:assert F(stats['min_threshold_margin'])>F(1,1<<20)
        checks+=1
    p=positions('SMOOTH',False);smooth=attributes('SMOOTH');color=oracle(p,False,smooth)[0]
    for mutated in [attributes('NOPERSPECTIVE'),attributes('FLAT')]:assert color!=oracle(p,False,mutated)[0];checks+=1
    wrong_w=[[x/w,y/w,z/w,F(1)] for x,y,z,w in p]
    assert color!=oracle(wrong_w,False,smooth)[0];checks+=1
    # Calibrate exact comparison using temporary, explicitly synthetic records.
    arm=next(x for x in arms if x['name']=='MULTIPLE_ZERO_ONE')
    driver=hashlib.sha256((root/'lib-arm/libvulkan_asahi.so').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='attribute-oracle-synthetic-') as tmp:
        ds=[]
        for stage in ['vert','mesh']:
            d=Path(tmp)/stage;d.mkdir();ds.append(d)
            (d/'RUN.json').write_text(json.dumps(dict(arm=arm['name'],stage=stage,no_gpu_execution=False,
                returncode=0,fence_success=True,driver_sha256=driver,synthetic_calibration=True)))
            for actual,expected in [('color.bin','.rgba'),('depth.bin','.depth')]:
                (d/actual).write_bytes((root/'expected'/(arm['name']+expected)).read_bytes())
        analyze.verify(arm,root,*ds);checks+=1
        for filename in ['color.bin','depth.bin','RUN.json']:
            path=ds[1]/filename;original=path.read_bytes()
            if filename=='RUN.json':
                data=json.loads(original);data['no_gpu_execution']=True;path.write_text(json.dumps(data))
            else:data=bytearray(original);data[4096]^=1;path.write_bytes(data)
            try:analyze.verify(arm,root,*ds)
            except ValueError:pass
            else:raise AssertionError('corruption accepted: '+filename)
            path.write_bytes(original);checks+=1
    print(json.dumps(dict(host_only=True,synthetic_oracle_calibration=True,checks=checks,hardware_results=0),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
