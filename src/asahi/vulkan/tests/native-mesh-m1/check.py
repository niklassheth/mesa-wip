#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent exact raster multiplicity + full owned-buffer guard oracle."""
import argparse, hashlib, json, struct
from pathlib import Path
W,H=128,64
Q_BYTES=512*1024
# Screen-space source geometry, independent of GPU output or shader witnesses.
POINTS=[(8,8),(24,8),(8,23),(40,8),(56,8),(40,23),
        (72,8),(88,8),(72,23),(8,40),(24,40),(8,55),
        (40,40),(56,40),(40,55),(56,55)]
TRIANGLES=[(0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(13,15,14)]

def cases():
    return [dict(name=f'P{maximum}-{kind}',maximum=maximum,vertices=16 if kind=='full' else 15,
                 primitives=maximum if kind=='full' else maximum-1,cookie=0x17600000+maximum*2+(kind=='partial'))
            for maximum in (15,16,17) for kind in ('full','partial')]

def geometry(case):
    return [TRIANGLES[i%(6 if case['vertices']==16 else 5)] for i in range(case['primitives'])]

def raster(triangles):
    values=[0]*(W*H)
    for tri in triangles:
        points=[POINTS[i] for i in tri]
        a,b,c=points
        if (b[0]-a[0])*(c[1]-a[1])==(b[1]-a[1])*(c[0]-a[0]):continue
        for y in range(min(p[1] for p in points),max(p[1] for p in points)):
            for x in range(min(p[0] for p in points),max(p[0] for p in points)):
                edges=[(b[0]-a[0])*(2*y+1-2*a[1])-(b[1]-a[1])*(2*x+1-2*a[0])
                       for a,b in zip(points,points[1:]+points[:1])]
                assert all(edges), 'authored geometry must avoid edge/sample ties'
                if all(e>0 for e in edges) or all(e<0 for e in edges):values[y*W+x]+=1
    return struct.pack('<%df'%len(values),*values)

def witness(case):
    expected=[0xcccccccc]*(Q_BYTES//4)
    expected[4:7]=[case['vertices'],case['primitives'],case['cookie']]
    for i,(x,y) in enumerate(POINTS[:case['vertices']]):
        expected[16+4*i:20+4*i]=struct.unpack('<4I',struct.pack('<4f',x/64-1,y/32-1,.5,1))
    for i,indices in enumerate(geometry(case)):expected[128+3*i:131+3*i]=indices
    for i in range(16):expected[512+i]=case['cookie']+0x1000+i
    return struct.pack('<%dI'%len(expected),*expected)

def compare(data,expected):
    if len(data)!=len(expected):return dict(ok=False,bytes=len(data),expected_bytes=len(expected),mismatches=None)
    bad=[i for i in range(0,len(data),4) if data[i:i+4]!=expected[i:i+4]]
    return dict(ok=not bad,checked_words=len(data)//4,mismatches=len(bad),first=[dict(word=i//4,expected=expected[i:i+4].hex(),actual=data[i:i+4].hex()) for i in bad[:16]])

def verify(directory,case):
    pixels=(directory/'pixels.bin').read_bytes();q=(directory/'Q.bin').read_bytes()
    image=compare(pixels,raster(geometry(case)));buffer=compare(q,witness(case))
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',pixels=image,witness=buffer,
                sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()),
                max_index=15 if case['vertices']==16 else 14,primitive_multiplicity_checked=case['primitives'],
                distinct_spatial_classes=6 if case['vertices']==16 else 5)

def mutation_tests():
    checked=0
    for case in cases():
        triangles=geometry(case);good=raster(triangles);q=witness(case)
        for i,tri in enumerate(triangles):
            assert raster(triangles[:i]+triangles[i+1:])!=good
            assert raster(triangles+[tri])!=good
            changed=list(triangles);changed[i]=TRIANGLES[(i+1)%(6 if case['vertices']==16 else 5)]
            assert raster(changed)!=good;checked+=3
        if case['vertices']==16:
            assert raster([tuple(14 if v==15 else v for v in tri) for tri in triangles])!=good
            checked+=1
        for word in (0,4,5,6,16,128,512,Q_BYTES//4-1):
            changed=bytearray(q);changed[4*word]^=1;assert not compare(changed,q)['ok'];checked+=1
        assert not compare(bytes([0xa5])*len(good),good)['ok'];checked+=1
    return dict(verdict='PASS',mutations_detected=checked,includes='each missing/duplicate/rebound primitive, high index, counts/geometry/lane/edge guards, untouched readback')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?');p.add_argument('--case',choices=[c['name'] for c in cases()]);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    result=mutation_tests() if a.self_test else verify(a.directory,next(c for c in cases() if c['name']==a.case))
    print(json.dumps(result,indent=2));raise SystemExit(result['verdict']!='PASS')
