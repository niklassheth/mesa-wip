#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact independent raster/G-buffer oracle for the original Niagara pair."""
import hashlib
import math
import struct

W,H,Q_BYTES=128,64,512*1024
MESH_SHA='7b25ad807d55ac3d5f201d6f72a8c2980593d461f91cbcceeb601f6207f386c3'

def cases():
    return [dict(name='Niagara-task0',logical=64,physical=32,max_vertices=64,
        max_primitives=96,vertices=64,primitives=96,cookie=0x188,mode=3,
        env=dict(HK_APPLE9_NATIVE_MESH='1'),mesh_sha256=MESH_SHA,
        fragment='Niagara-task0.frag.spv')]

def position(v):
    return 4+17*(v%8),3+8*(v//8)

def triangles():
    result=[]
    for cell in range(49):
        if cell==24:continue
        base=(cell//7)*8+cell%7
        result.extend([(base,base+1,base+8),(base+1,base+9,base+8)])
    assert len(result)==96 and set(v for tri in result for v in tri)==set(range(64))
    return result

def inputs():
    q=bytearray([0xcc])*Q_BYTES
    for offset,size in [(0,216),(4096,320),(8192,768),(12288,384),
                        (16384,1024),(20480,2048),(24576,64),(28672,1024)]:
        q[offset:offset+size]=bytes(size)
    for offset in (0,64):
        for c in range(4):struct.pack_into('<f',q,offset+20*c,1)
    struct.pack_into('<2f',q,208,W,H)
    # Cluster index selects command 3, then meshlet 4+2. Draw 7 selects material 3.
    struct.pack_into('<I',q,24576,3|(2<<24))
    struct.pack_into('<5I',q,4096+3*20,7,4,1,0,0)
    struct.pack_into('<8f4I',q,8192+7*48,0,0,0,1,0,0,0,1,0,0,0,3)
    struct.pack_into('<2I4B',q,12288+6*24+12,7,5,64,96,1,0)
    for v in range(64):
        index=(13*v+7)%64
        struct.pack_into('<H',q,16384+28+v*2,index)
        x,y=position(v)
        # Normal +Z stays away from the octahedral -Z seam. Packed
        # (511,511,0) is unsuitable: fused unpack makes X/Y slightly negative
        # where separate multiply/subtract gives zero, choosing another corner.
        struct.pack_into('<3eHI2e',q,20480+(5+index)*16,
                         x/64-1,y/32-1,.25,254|(127<<8),511|(511<<10)|(1022<<20),v/64,1-v/64)
    for i,tri in enumerate(triangles()):q[16384+156+3*i:16384+159+3*i]=bytes(tri)
    # No textures; original FS: endpoint sRGB values, zero emission and full gloss.
    struct.pack_into('<4I4f4f3f',q,28672+3*64,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0)
    # A second draw/material pair supports the unchanged-shader input control.
    struct.pack_into('<8f4I',q,8192+6*48,0,0,0,1,0,0,0,1,0,0,0,2)
    struct.pack_into('<4I4f4f3f',q,28672+2*64,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0)
    return bytes(q)

def coverage(tri):
    points=[(2*x,2*y) for x,y in map(position,tri)]
    covered=[]
    for y in range(H):
        for x in range(W):
            edges=[(b[0]-a[0])*(2*y+1-a[1])-(b[1]-a[1])*(2*x+1-a[0])
                   for a,b in zip(points,points[1:]+points[:1])]
            if all(e>=0 for e in edges) and 0 in edges:
                raise ValueError('pixel centre on edge')
            if all(e>0 for e in edges):covered.append(y*W+x)
    assert covered
    return covered

def noise(x,y):
    f=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
    dot=f(f(x*f(.06711056))+f(y*f(.00583715)))
    value=f(f(52.9829189)*f(dot-math.floor(dot)))
    return f(value-math.floor(value))

def normal_component(index):
    # +Z is the continuous centre of the octahedron. The original noise
    # chooses between the two nearest 10-bit values; quantization.py proves
    # that fused/separate arithmetic makes the same choice for this grid.
    return 512 if noise(index%W+.5,index//W+.5)>=.5 else 511

_expected=None
def expected(primitives=96):
    p=bytearray(W*H*8)
    for tri in triangles()[:primitives]:
        for index in coverage(tri):
            p[index*4:index*4+4]=bytes((255,0,255,0))
            component=normal_component(index)
            struct.pack_into('<I',p,W*H*4+index*4,
                             0x3ff00000|component|(component<<10))
    return bytes(p)

def verify_bytes(pixels,q,case=None):
    global _expected
    if _expected is None:_expected=expected()
    def compare(a,b):
        if len(a)!=len(b):return dict(ok=False,bytes=len(a),expected_bytes=len(b))
        bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        return dict(ok=not bad,checked_bytes=len(a),mismatches=len(bad),first=bad[:16])
    image=compare(pixels,_expected);guard=compare(q,inputs())
    return dict(verdict='PASS' if image['ok'] and guard['ok'] else 'FAIL',pixels=image,witness=guard,
        sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))

def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),(directory/'Q.bin').read_bytes(),case)

def mutations():
    p,q=expected(),inputs();assert verify_bytes(p,q)['verdict']=='PASS'
    edits=[('empty',bytes(len(p)),q),('unexecuted',bytes([0xa5])*len(p),q),
           ('last-primitive',expected(95),q),('last-index-round',expected(85),q),
           ('truncated',p[:-1],q),('missing-second-attachment',p[:W*H*4],q)]
    for name,offset in [('draw-command',4096+60),('draw',8192+7*48),('meshlet',12288+6*24),
                        ('packed-index',16384+156+287),('upper-vertex',20480+(5+(13*63+7)%64)*16),
                        ('cluster',24576),('material',28672+3*64),('guard',Q_BYTES-1)]:
        changed=bytearray(q);changed[offset]^=1;edits.append((name,p,changed))
    index=coverage(triangles()[0])[0]*4
    for target in range(2):
        for channel in range(4):
            changed=bytearray(p);changed[target*W*H*4+index+channel]^=1
            edits.append((f'attachment{target}-byte{channel}',changed,q))
    for name,image,guard in edits:assert verify_bytes(image,guard)['verdict']=='FAIL',name
    return dict(verdict='PASS',mutations_detected=len(edits),names=[x[0] for x in edits])
