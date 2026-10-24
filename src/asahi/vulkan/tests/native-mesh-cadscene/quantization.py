#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the constant fragment color stays away from UNORM rounding ties."""
import math
import struct
from check import color

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def near(x,steps=1):
    word=struct.unpack('<I',struct.pack('<f',x))[0]
    return [struct.unpack('<f',struct.pack('<I',word+s))[0] for s in range(-steps,steps+1)] if x else [0.0]

def verify_margin():
    m=0x5bd1e995;k=(5*m)&0xffffffff;k^=k>>24
    h=((64684*m)&0xffffffff)^((k*m)&0xffffffff)
    # 32 relative float32 ULPs cover normalization, dot and interpolation
    # error here: normal +Z, positions bounded by one, light Z=2^30.
    # This is a deliberately conservative fixture error model, not a claim
    # about implementation precision for arbitrary fragment programs.
    diffuse=[1-32*2**-23,1,1+32*2**-23]
    separate=f32(f32(-.25*f32(.8))+f32(.2))
    fused=f32(-.25*f32(.8)+f32(.2))
    assert separate==fused==0
    variants=0;margin=1
    for channel,extra in enumerate((.125,.25,.375,.125)):
        byte=(h>>(8*channel))&255
        exact=byte*.5+255*extra
        margin=min(margin,abs(exact-(math.floor(exact)+.5)))
        for unpack in near(f32(byte/255)):
            for light in diffuse:
                for attr in near(extra,4):
                    base=f32(unpack*.5)
                    for value in (f32(f32(light*base)+attr),f32(light*base+attr)):
                        for scaled in (value*255,f32(value*255)):
                            assert math.floor(scaled+.5)==color()[channel],(channel,value)
                            variants+=1
    assert margin>=.125
    return dict(verdict='PASS',variants=variants,min_unorm_steps=margin,
        diffuse_relative_error=32*2**-23,unpack_error_ulps=1,extra_error_ulps=4,
        fused_and_separate=True,far_light_z=2**30,rgba=list(color()),
        scope='bounded constant fixture color; exact byte comparison, no tolerance')
