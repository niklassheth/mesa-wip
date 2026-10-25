#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent raster/guard oracle for unchanged Khronos mesh+fragment modules."""
import hashlib
import struct
from pathlib import Path


def cases():
    return [dict(name='Khronos-'+language,language=language,threads=1,
        logical=[1,1,1],physical=16,shared=False,mode=4,cookie=0,
        max_vertices=3,max_primitives=1,vertices=3,primitives=1,pixel_bytes=16,
        env={'HK_APPLE9_NATIVE_INDEX':'1'},
        fragment='Khronos-'+language+'.frag.spv')
        for language in ('glsl','hlsl','slang')]


def expected():
    # Public source positions (0.5,-0.5),(0.5,0.5),(-0.5,0.5) in a
    # positive 128x64 viewport give (96,16),(96,48),(32,48).
    # At pixel centers, the diagonal inequality never ties (odd parity).
    pixels=[]
    for y in range(64):
        for x in range(128):
            covered=32<=x<96 and 16<=y<48 and (2*x+1)+2*(2*y+1)>256
            pixels.extend((1.,0.,0.,1.) if covered else (0.,0.,0.,0.))
    # Original shaders have no SSBO writes. The entire guarded allocation
    # must remain at initialization, not a witness inserted into the module.
    return struct.pack('<32768f',*pixels),bytes([0xcc])*(512*1024)


def compare(actual,reference):
    if len(actual)!=len(reference):
        return dict(ok=False,bytes=len(actual),expected_bytes=len(reference))
    bad=[i//4 for i in range(0,len(actual),4) if actual[i:i+4]!=reference[i:i+4]]
    return dict(ok=not bad,checked_words=len(actual)//4,mismatches=len(bad),first=bad[:16])


def verify_bytes(pixels,q):
    ep,eq=expected();image=compare(pixels,ep);buffer=compare(q,eq)
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',
        pixels=image,witness=buffer,sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))


def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),(directory/'Q.bin').read_bytes())


def mutations():
    p,q=expected();assert verify_bytes(p,q)['verdict']=='PASS';names=[]
    for name,pixels,buffer in [('unexecuted',bytes([0xa5])*len(p),q),
        ('empty-raster',bytes(len(p)),q),('truncated',p[:-4],q[:-4]),
        ('guard-store',p,bytes(4)+q[4:])]:
        assert verify_bytes(pixels,buffer)['verdict']=='FAIL';names.append(name)
    index=32*128+80
    for name,value in [('missing',0),('duplicate',2),('incorrect-fragment',0.5)]:
        changed=bytearray(p);struct.pack_into('<f',changed,16*index,value)
        assert verify_bytes(changed,q)['verdict']=='FAIL';names.append(name)
    for component in range(4):
        changed=bytearray(p);struct.pack_into('<f',changed,16*index+4*component,0.25)
        assert verify_bytes(changed,q)['verdict']=='FAIL';names.append('channel-'+str(component))
    return dict(verdict='PASS',mutations_detected=len(names),names=names)
