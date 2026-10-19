#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Require exact whole-image RGBA8 and D32 source oracles AND VS/native equality."""
import argparse, hashlib, json
from pathlib import Path

def require(condition,message):
    if not condition:raise ValueError(message)
def verify(arm, root, vs, mesh):
    images=[]
    for stage,d in [('vert',vs),('mesh',mesh)]:
        run=json.loads((d/'RUN.json').read_text())
        require(run['arm']==arm['name'] and run['stage']==stage,'wrong arm/stage')
        require(run.get('no_gpu_execution') is False,'host dry-run is not GPU evidence')
        require(run.get('returncode')==0 and run.get('fence_success') is True,'incomplete draw')
        require(run.get('driver_sha256')==hashlib.sha256((root/'lib-arm/libvulkan_asahi.so').read_bytes()).hexdigest(),'wrong driver bytes')
        pair=[]
        for filename,expected in [('color.bin','.rgba'),('depth.bin','.depth')]:
            actual=(d/filename).read_bytes();reference=(root/'expected'/(arm['name']+expected)).read_bytes()
            require(len(actual)==len(reference)==16384,'incomplete readback')
            require(actual==reference,stage+' '+filename+' differs from exact source oracle')
            pair.append(actual)
        images.append(pair)
    require(images[0]==images[1],'VS/native color or D32 differ')
    return dict(arm=arm['name'],exact_color=True,exact_d32=True,paired_images_identical=True,
                hardware_pass=True,covered=arm['covered'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--vs',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True)
    a=p.parse_args();arm=next(x for x in json.loads((a.root/'ARMS.json').read_text())['arms'] if x['name']==a.arm)
    result=verify(arm,a.root,a.vs,a.mesh);print(json.dumps(result,indent=2))
