#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent exact color/depth oracle for the captured CADSCENE basic pair."""
import hashlib
import math
import struct

W,H,Q_BYTES=128,64,512*1024
MESH_SHA='e6205c94bf8556dd9fc0a2802b88da53e8c2c7c383af5d383e7711fd2897d696'
FRAGMENT_SHA='718a68febd5ffe9461e386472df0fc7867aa60257e21b728a59a506a8296720a'

def cases():
    return [dict(name='CADSCENE-basic-task0',logical=32,physical=32,max_vertices=64,
        max_primitives=84,vertices=64,primitives=84,cookie=0x197,mode=3,
        env=dict(HK_APPLE9_NATIVE_MESH='1'),mesh_sha256=MESH_SHA,
        fragment='CADSCENE-basic-task0.frag.spv')]

def position(v):return 4+17*(v%8),3+8*(v//8)

def triangles():
    result=[]
    for cell in range(49):
        if cell%7==3:continue
        b=(cell//7)*8+cell%7
        result.extend([(b,b+8,b+1),(b+1,b+8,b+9)])
    # Deliberate backfaces test the captured raster state. Last triangle is visible.
    result=[tuple(reversed(t)) if i%11==3 else t for i,t in enumerate(result)]
    assert len(result)==84 and set(v for t in result for v in t)==set(range(64))
    return result

def inputs():
    q=bytearray([0xcc])*Q_BYTES
    for off,size in [(0,32),(4096,320),(8192,256),(12288,1024),
                     (16384,2048),(20480,2048),(24576,4096),(28672,1024)]:
        q[off:off+size]=bytes(size)
    struct.pack_into('<8I',q,0,7,64,0,11,5,5,0,0)
    for base in (4096,8192):
        for mat in (0,64,128):
            for c in range(4):struct.pack_into('<f',q,base+mat+20*c,1)
    struct.pack_into('<4f',q,4096+224,0,0,2**30,1)
    struct.pack_into('<2i4fi',q,4096+240,W,H,W,H,W,H,1)
    struct.pack_into('<f4f',q,8192+236,1,-.25,-.25,-.25,-.25)
    struct.pack_into('<4I',q,12288+12*16,63<<24,83<<24,2<<24,9)
    for v in range(64):
        index=(13*v+7)%64
        struct.pack_into('<H',q,16384+100+2*v,index)
        x,y=position(v)
        struct.pack_into('<4f',q,20480+(11+index)*16,x/64-1,y/32-1,.25,1)
        struct.pack_into('<8f',q,24576+(11+index)*32,0,0,1,0,.125,.25,.375,.125)
    for i,t in enumerate(triangles()):q[16384+232+3*i:16384+235+3*i]=bytes(t)
    return bytes(q)

def coverage(tri):
    pts=[(2*x,2*y) for x,y in map(position,tri)]
    cross=(pts[1][0]-pts[0][0])*(pts[2][1]-pts[0][1])-(pts[1][1]-pts[0][1])*(pts[2][0]-pts[0][0])
    if cross>=0:return []
    covered=[]
    for y in range(H):
        for x in range(W):
            e=[(b[0]-a[0])*(2*y+1-a[1])-(b[1]-a[1])*(2*x+1-a[0]) for a,b in zip(pts,pts[1:]+pts[:1])]
            if all(v<=0 for v in e) and 0 in e:raise ValueError('pixel center on edge')
            if all(v<0 for v in e):covered.append(y*W+x)
    assert covered
    return covered

def color():
    # Public sample's seeded Murmur mix, evaluated modulo 2^32.
    m=0x5bd1e995;k=(5*m)&0xffffffff;k^=k>>24
    h=((64684*m)&0xffffffff)^((k*m)&0xffffffff)
    values=[((h>>(8*i))&255)*.5+x*255 for i,x in enumerate((.125,.25,.375,.125))]
    assert min(abs(v-(math.floor(v)+.5)) for v in values)>=.125
    return bytes(math.floor(v+.5) for v in values)

def expected(primitives=84):
    p=bytearray(W*H*4)+bytearray(struct.pack('<f',1)*W*H)
    for t in triangles()[:primitives]:
        for i in coverage(t):
            p[i*4:i*4+4]=color()
            struct.pack_into('<f',p,W*H*4+i*4,.25)
    return bytes(p)

_expected=None

def verify_bytes(pixels,q,case=None):
    global _expected
    if _expected is None:_expected=expected()
    def cmp(a,b):
        if len(a)!=len(b):return dict(ok=False,bytes=len(a),expected_bytes=len(b))
        bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        return dict(ok=not bad,checked_bytes=len(a),mismatches=len(bad),first=bad[:16])
    image=cmp(pixels,_expected);guard=cmp(q,inputs())
    return dict(verdict='PASS' if image['ok'] and guard['ok'] else 'FAIL',pixels=image,witness=guard,
        sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))

def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),(directory/'Q.bin').read_bytes(),case)

def mutations():
    p,q=expected(),inputs();assert verify_bytes(p,q)['verdict']=='PASS'
    edits=[('empty',bytes(len(p)),q),('unexecuted',bytes([0xa5])*len(p),q),
           ('last-primitive',expected(83),q),('last-index-round',expected(64),q),
           ('truncated',p[:-1],q),('missing-depth',p[:W*H*4],q)]
    for name,offset in [('push',0),('scene',4096+264),('object',8192+240),
                        ('meshlet',12288+12*16),('indices',16384+232+251),
                        ('upper-vertex',20480+(11+(13*63+7)%64)*16),
                        ('attributes',24576),('stats',28672),('guard',Q_BYTES-1)]:
        changed=bytearray(q);changed[offset]^=1;edits.append((name,p,changed))
    index=coverage(triangles()[0])[0]*4
    for target in range(2):
        for channel in range(4):
            changed=bytearray(p);changed[target*W*H*4+index+channel]^=1
            edits.append((f'attachment{target}-byte{channel}',changed,q))
    for name,image,guard in edits:assert verify_bytes(image,guard)['verdict']=='FAIL',name
    return dict(verdict='PASS',mutations_detected=len(edits),names=[x[0] for x in edits])
