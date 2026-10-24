#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact raster/UNORM oracle for the unchanged NVIDIA basic mesh/fragment pair."""
from fractions import Fraction
import hashlib
import math
import struct

W, H, Q_BYTES = 128, 64, 512*1024
MESH_SHA = '7e177a3a5abec51344de53a657eaa33b7f00864e7fe230ca8c0652230f235a68'
NORMALS = ((0,0,1), (3,0,4), (0,4,3), (1,2,2))
DIFFUSE = (Fraction(1), Fraction(4,5), Fraction(3,5), Fraction(2,3))


def cases():
    return [dict(name='NVIDIA-basic-task0', physical=32, max_vertices=64,
        max_primitives=126, vertices=64, primitives=126, cookie=0x182,
        mode=3, env=dict(HK_APPLE9_NATIVE_MESH='1'),
        mesh_sha256=MESH_SHA, fragment='NVIDIA-basic-task0.frag.spv')]


def position(vertex):
    cell, corner = divmod(vertex,4)
    return (2+16*(cell%8)+12*(corner&1), 2+16*(cell//8)+11*(corner>>1))


def triangle(primitive):
    t = primitive%32
    base = 4*(t//2)
    return (base,base+1,base+2) if t%2 == 0 else (base+1,base+3,base+2)


def inputs():
    q = bytearray([0xcc])*Q_BYTES
    struct.pack_into('<8I',q,3072,2,32,0,5,7,7,0,0)
    for offset,size in ((4096,320),(8192,256),(12288,256),(16384,1024),
                        (20480,1280),(24576,1280),(28672,800)):
        q[offset:offset+size] = bytes(size)
    def matrix(offset, diagonal):
        for i,value in enumerate(diagonal):
            struct.pack_into('<f',q,offset+20*i,value)
    matrix(4096,(2**20,2**20,1,1))
    matrix(4160,(1,1,1,1)); matrix(4224,(1,1,1,1))
    struct.pack_into('<4f',q,4096+224,0,0,1.25,1)
    struct.pack_into('<I',q,4096+264,1)  # colorize: use the actual flat meshlet ID
    for offset in (8192,8256,8320): matrix(offset,(1,1,1,1))
    struct.pack_into('<4f',q,8192+240,*([-0.09375]*4))
    # drawRange.x=7, geometryOffsets.x=2 => descriptor 9. Offset.y=32 bytes.
    struct.pack_into('<4I',q,12288+9*16,63<<24,125<<24,2<<24,16)
    # Sixteen-bit indices are permuted, with nonzero base vertex 5. This
    # exercises both halves of each packed word and the shader's real TBOs.
    for v in range(64):
        index = (13*v+7)%64
        struct.pack_into('<H',q,16384+96+v*2,index)
        x,y = position(v)
        struct.pack_into('<4f',q,20480+(5+index)*16,
                         (x/64-1)*2**-20,(y/32-1)*2**-20,.25,1)
        struct.pack_into('<4f',q,24576+(5+index)*16,*NORMALS[(v//4)%4],0)
    for p in range(126):
        q[16384+224+p*3:16384+224+p*3+3] = bytes(triangle(p))
    return bytes(q)


def color_bytes(meshlet=7):
    # Public sample's murmurHash: independent integer arithmetic modulo 2^32.
    m = 0x5bd1e995
    k = (meshlet*m)&0xffffffff
    h = ((64684*m)&0xffffffff)^(((k^(k>>24))*m)&0xffffffff)
    return [(h>>(8*i))&255 for i in range(4)]


def source_color(cell):
    # object.color*.8+.2 = 1/8; colorize halves it and adds half the hash.
    # Quantize once per fragment. Additive UNORM storage then adds these
    # integer increments until saturation. No value is near a half tie.
    values = [(Fraction(255,16)+Fraction(c,2))*DIFFUSE[cell%4] for c in color_bytes()]
    return [int(v+Fraction(1,2)) for v in values]


def coverage(tri):
    points = [(2*x,2*y) for x,y in map(position,tri)]
    covered=[]
    for y in range(H):
        for x in range(W):
            edges = [(b[0]-a[0])*(2*y+1-a[1])-(b[1]-a[1])*(2*x+1-a[0])
                     for a,b in zip(points,points[1:]+points[:1])]
            if (all(e>=0 for e in edges) or all(e<=0 for e in edges)) and 0 in edges:
                raise ValueError('pixel center lies on an edge')
            if all(e>0 for e in edges) or all(e<0 for e in edges):
                covered.append(y*W+x)
    assert covered
    return covered


def expected(primitives=126):
    pixels = bytearray(W*H*4)
    cache={}
    for primitive in range(primitives):
        tri=triangle(primitive)
        if tri not in cache: cache[tri]=coverage(tri)
        color=source_color(tri[0]//4)
        for i in cache[tri]:
            for c in range(4): pixels[i*4+c]=min(255,pixels[i*4+c]+color[c])
    assert {v for tri in cache for v in tri} == set(range(64))
    return bytes(pixels)


def precision_margin():
    # Bound the change from actual interpolated position to the axial-light
    # rational oracle. |world XY| < sqrt(2)*2^-20, light delta Z exactly 1.
    # Reserve another 1e-5 of diffuse for floating point normalizing/shading.
    bound = (math.sqrt(2)*2**-20+1e-5)*255
    margins=[]
    for diffuse in DIFFUSE:
        for color in color_bytes():
            value=(Fraction(255,16)+Fraction(color,2))*diffuse
            margins.append(abs(value-(value.numerator//value.denominator+Fraction(1,2))))
    assert float(min(margins)) > bound
    return dict(minimum_half_step_margin=float(min(margins)),
                reserved_math_and_position_error_in_unorm_steps=bound,
                acceptance='exact RGBA8 bytes; no tolerance applied to readback')


_expected = None
def verify_bytes(pixels,q,case=None):
    global _expected
    if _expected is None: _expected=expected()
    def compare(a,b):
        if len(a)!=len(b): return dict(ok=False,bytes=len(a),expected_bytes=len(b))
        bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        return dict(ok=not bad,checked_bytes=len(a),mismatches=len(bad),first=bad[:16])
    image=compare(pixels,_expected);buffer=compare(q,inputs())
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',
        pixels=image,witness=buffer,sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))


def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),(directory/'Q.bin').read_bytes(),case)


def mutations():
    p,q=expected(),inputs();assert verify_bytes(p,q)['verdict']=='PASS';names=[]
    edits=[('empty',bytes(len(p)),q),('unexecuted',bytes([0xa5])*len(p),q),
           ('lost-last-primitive',expected(125),q),('extra-primitive',expected(127),q),
           ('lost-third-index-round',expected(85),q),('truncated',p[:-1],q)]
    upper=bytearray(p)
    for y in range(16,H): upper[y*W*4:(y+1)*W*4]=bytes(W*4)
    edits.append(('lost-upper-vertices',upper,q))
    for label,index in [('scene-ubo',4096),('object-ubo',8192),('meshlet-desc',12288+9*16),
          ('packed-vertex',16384+96),('last-index',16384+224+377),
          ('position-tbo',20480+5*16),('normal-tbo',24576+5*16),('guard',Q_BYTES-1)]:
        changed=bytearray(q);changed[index]^=1;edits.append((label,p,changed))
    sample=coverage(triangle(0))[0]
    for component in range(4):
        changed=bytearray(p);changed[sample*4+component]^=1
        edits.append(('channel-'+str(component),changed,q))
    for label,pixels,buffer in edits:
        assert verify_bytes(pixels,buffer)['verdict']=='FAIL',label
        names.append(label)
    return dict(verdict='PASS',mutations_detected=len(names),names=names,
                precision=precision_margin())
