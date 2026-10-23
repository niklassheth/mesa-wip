#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit all frozen host submissions against the independent source oracle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
from oracles import expected


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def check_geometry(s, q, pixels):
    """Independently rasterize the authored snapshot's finite fixture geometry.

    All covered samples are strictly inside the triangle/point footprint;
    horizontal lines cross pixel centers. No GPU output enters this check.
    """
    local=q[0x4000:0x4000+s['local_B']]
    p,=struct.unpack_from('<H',local,s['count_B'])
    v,=struct.unpack_from('<I',local,s['count_B']+4)
    primitives=Counter()
    for i in range(p):
        vertices=[]
        for j in range(s['indices']):
            index=local[i*s['indices']+j];assert index<v
            xyzw=tuple(struct.unpack_from('<f',local,s['position_B']+c*s['pitch_B']+4*index)[0] for c in range(s['planes']))
            assert xyzw[2:4]==(0.,1.)
            if s['topology']==0:assert xyzw[4]==1.
            vertices.append(xyzw[:2])
        primitives[tuple(vertices)]+=1
    result=[0]*4096
    for vertices,multiplicity in primitives.items():
        if s['topology']==6:
            a,b,c=vertices
            area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            if not area:continue
        elif s['topology']==1:
            a,b=vertices;assert a[1]==b[1]
            left,right=sorted([32*a[0]+32,32*b[0]+32]);center=32-32*a[1]
            if left==right:
                assert left<0 or left>64  # zero-length controls stay outside
                continue
        else:
            a,=vertices;cx,cy=32*a[0]+32,32-32*a[1]
        for y in range(64):
            for x in range(64):
                if s['topology']==6:
                    point=((x+.5)/32-1,1-(y+.5)/32)
                    edges=[(v2[0]-v1[0])*(point[1]-v1[1])-(v2[1]-v1[1])*(point[0]-v1[0]) for v1,v2 in [(a,b),(b,c),(c,a)]]
                    inside=all(e>0 for e in edges) or all(e<0 for e in edges)
                elif s['topology']==1:
                    inside=left<=x+.5<right and abs(y+.5-center)<.5
                else:
                    inside=abs(x+.5-cx)<.5 and abs(y+.5-cy)<.5
                if inside:result[y*64+x]+=multiplicity
    assert struct.pack('<4096f',*result)==pixels

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--dry',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();dry=a.dry.resolve()
    cat=json.loads((root/'CATALOG.json').read_text());report=json.loads((dry/'DRY-RUN.json').read_text())
    assert report['gpu_execution'] is False and report['qualified_trials']==0
    assert len(report['rows'])==len(cat['arms'])==175
    fixtures=root/'fixtures';fixtures.mkdir(exist_ok=False);index=[];seen=set()
    for seq,(arm,row) in enumerate(zip(cat['arms'],report['rows']),1):
        assert arm['name']==row['arm'] and row['code']==2 and row['ordinal']==seq-1
        d=dry/f'evidence-{seq:03d}';s=arm['shape'];cookie=0x7a000000+seq
        initial,want,pixels=expected(s,*arm['runtime_counts'],cookie)
        check_geometry(s,want,pixels)
        assert (d/'q.bin').read_bytes()==initial
        assert (d/'expected-q.bin').read_bytes()==want
        assert (d/'expected-rgba.bin').read_bytes()==pixels
        assert (d/'rgba.bin').read_bytes()==bytes([0xcd])*len(pixels)
        for stage in ['mesh','object']:
            assert sha(d/f'cross-{stage}-code-after.bin')==arm['programs'][stage]['sha256']
        w=struct.unpack('<40I',(d/'cross-stream.bin').read_bytes())
        p=s['primitives'];n=s['planes'];v=s['vertices'];t=s['threads'];topo=s['topology']
        assert w[1]==n|((n+1)<<8) and w[2]==0x10 and w[3]==w[4]==w[12]==0
        assert w[10]==(p-1)|(((p+15)//16-1)<<16) and w[11]==0x400800|t
        assert w[13]==1+((v+3)//4)*n and w[32]==0x70000000|(topo<<8)
        launch=(d/'cross-mesh-launch-after.bin').read_bytes()
        assert int.from_bytes(launch[90:93],'little')==(s['allocation_B']<<2)|0x80
        select,=struct.unpack_from('<I',(d/'ppp-2.bin').read_bytes(),4)
        output,=struct.unpack_from('<I',(d/'ppp-2.bin').read_bytes(),16)
        assert bool(select&(1<<18))==(topo==0) and output==n
        ppp4=(d/'ppp-4.bin').read_bytes();face=struct.unpack_from('<2I',ppp4,8)
        assert all(x>>28=={6:0,1:1,0:4}[topo] for x in face)
        # Export the exact host-built stream for independent kernel admission
        # tests; these are data-only fixtures, not submitted device work.
        name=arm['programs']['mesh']['file'].removesuffix('.mesh.bin')
        if name not in seen:
            seen.add(name);dest=fixtures/(name+'.bin');shutil.copy2(d/'cross-stream.bin',dest)
            index.append(dict(name=name,file=dest.name,sha256=sha(dest),shape=s))
    (fixtures/'INDEX.json').write_text(json.dumps(dict(gpu_execution=False,cases=index),indent=2)+'\n')
    out=dict(arms=len(cat['arms']),program_shapes=len(index),gpu_execution=False,qualified_trials=0,
             exact_c_oracle_vs_python=True,independent_geometry_count_check=True,all_uploaded_bodies_exact=True,all_state_allocation_and_ppp_fields_exact=True,
             fixtures_catalog_sha256=sha(fixtures/'INDEX.json'),renderer=json.loads((dry/'IDENTITY.json').read_text()))
    (root/'HOST-CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
