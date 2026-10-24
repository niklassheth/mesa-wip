#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Logical invocation/subgroup IDs, cross-subgroup visibility and guard oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct

Q_BYTES=512*1024
W,H=128,64

def cases():
    interior=random.Random(0x180).randint(33,127)
    sizes=[(1,1,1),(17,1,1),(32,1,1),(64,1,1),(128,1,1),
           (interior,1,1),(8,8,1),(64,1,1)]
    return [dict(name='L%dx%dx%d%s'%(*size,'-shared' if i==7 else ''),
        logical=list(size),threads=size[0]*size[1]*size[2],shared=i==7,
        mode=3 if i==7 else 4,cookie=0x18000000+i,
        physical=16 if size[0]*size[1]*size[2]<=16 else 32)
        for i,size in enumerate(sizes)]

def raster():
    values=[0]*(W*H)
    for y in range(8,23):
        for x in range(8,24):
            # Triangle (8,8), (24,8), (8,23); centers never lie on its edges.
            values[y*W+x]=int(15*(2*x+1-16)+16*(2*y+1-16)<480)
    return struct.pack('<8192f',*values)

def witness(case,subgroup):
    assert subgroup>0 and subgroup&(subgroup-1)==0
    q=[0xcccccccc]*(Q_BYTES//4)
    q[4:7]=[3,1,case['cookie']]
    x,y,z=case['logical'];n=case['threads'];cookie=case['cookie']
    for lane in range(n):
        q[1024+lane]=cookie+lane
        start=2048+16*lane
        q[start:start+12]=[lane,lane%x,(lane//x)%y,lane//(x*y),
            lane//subgroup,lane%subgroup,subgroup,(n+subgroup-1)//subgroup,
            cookie+(lane+33)%n,cookie+lane,n,0x18000000+lane]
    return struct.pack('<%dI'%len(q),*q)

def compare(actual,expected):
    if len(actual)!=len(expected):return dict(ok=False,bytes=len(actual),expected_bytes=len(expected))
    mismatches=[i for i in range(0,len(actual),4) if actual[i:i+4]!=expected[i:i+4]]
    return dict(ok=not mismatches,checked_words=len(actual)//4,mismatches=len(mismatches),
        first=[dict(word=i//4,expected=expected[i:i+4].hex(),actual=actual[i:i+4].hex()) for i in mismatches[:16]])

def verify_bytes(pixels,q,case,subgroup=32):
    image=compare(pixels,raster());buffer=compare(q,witness(case,subgroup))
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',pixels=image,witness=buffer,
        logical_invocations=case['threads'],subgroup_size=subgroup,shared=case['shared'],
        sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))

def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),
        (directory/'Q.bin').read_bytes(),case,subgroup)

def mutations():
    detected=[]
    for case in cases():
        expected=witness(case,32);pixels=raster();n=case['threads']
        assert verify_bytes(pixels,expected,case)['verdict']=='PASS'
        changes=[('count',[ (5,0) ]),('cookie',[(6,0)]),
            ('leading-guard',[(0,0)]),('trailing-guard',[(Q_BYTES//4-1,0)]),
            ('write-past-logical-end',[(1024+n,0)]),
            ('pre-barrier-read',[(2048+16*lane+8,0xcccccccc) for lane in range(n)])]
        if n>32:
            changes += [('lost-upper-invocations',[(2048+16*lane,0xcccccccc) for lane in range(32,n)]),
                ('physical-subgroup-id',[(2048+16*lane+4,0) for lane in range(n)]),
                ('physical-num-subgroups',[(2048+16*lane+7,1) for lane in range(n)])]
        if case['logical'][1]>1:
            changes += [('flattened-xy',[(2048+16*lane+1,lane) for lane in range(n)])]
        for name,words in changes:
            changed=bytearray(expected)
            for word,value in words:struct.pack_into('<I',changed,4*word,value)
            assert verify_bytes(pixels,changed,case)['verdict']=='FAIL',(case,name)
            detected.append(case['name']+':'+name)
        for value in (0,2):
            changed=bytearray(pixels);struct.pack_into('<f',changed,4*(8*W+8),value)
            assert verify_bytes(changed,expected,case)['verdict']=='FAIL'
            detected.append(case['name']+(':lost-triangle' if value==0 else ':duplicate-triangle'))
        for name,p,q in [('unexecuted',bytes([0xa5])*len(pixels),bytes([0xcc])*len(expected)),
                         ('truncated',pixels[:-4],expected[:-4])]:
            assert verify_bytes(p,q,case)['verdict']=='FAIL'
            detected.append(case['name']+':'+name)
    return dict(verdict='PASS',mutations_detected=len(detected),cases=detected,native_evidence=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('directory',type=Path,nargs='?');p.add_argument('--case');a=p.parse_args()
    result=mutations() if a.self_test else verify(a.directory,next(c for c in cases() if c['name']==a.case))
    print(json.dumps(result,indent=2));raise SystemExit(result['verdict']!='PASS')
