#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Exact source oracle; untouched poison, missing/duplicated lanes and wrong IDs fail."""
import argparse,json
from pathlib import Path

def compare(root,arm,stage,directory):
    run=json.loads((directory/'RUN.json').read_text())
    assert run['arm']==arm and run['stage']==stage
    assert run.get('fence_success') and not run['no_gpu_execution'], 'retired device submission required'
    result={}
    for output,ext in [('color.bin','rgba'),('depth.bin','depth'),('q.bin','q')]:
        actual=(directory/output).read_bytes()
        expected=root/'expected'/(('initial.q') if stage=='vert' and ext=='q' else arm+'.'+ext)
        wanted=expected.read_bytes();assert len(actual)==len(wanted)
        mismatch=[i for i,(x,y) in enumerate(zip(actual,wanted)) if x!=y]
        result[output]=dict(bytes=len(actual),mismatched_bytes=len(mismatch),first_mismatch=mismatch[:1])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--arm',required=True);p.add_argument('--vert',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);a=p.parse_args()
    result={stage:compare(a.root,a.arm,stage,getattr(a,stage)) for stage in ['vert','mesh']}
    result['pass']=all(not x['mismatched_bytes'] for stage in ['vert','mesh'] for x in result[stage].values())
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
