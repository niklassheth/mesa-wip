#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent RGBA interpolation, typed primitive and immutable-UBO oracle."""
import hashlib
import math
import struct

WIDTH, HEIGHT = 128, 64
TOLERANCE = 2e-5


def cases():
    result = [dict(name='Sascha-'+lang, language=lang, threads=1,
        logical=[1,1,1], physical=16, shared=False, mode=3, cookie=0,
        fragment='Sascha-'+lang+'.frag.spv', components=8, output_rows=9,
        pixel_bytes=16, initial_q='initial-Q.bin')
        for lang in ('glsl','hlsl','slang')]
    result.append(dict(name='P1-mixed', threads=32, logical=[32,1,1],
        physical=32, shared=False, mode=3, cookie=0, fragment='P1-mixed.frag.spv',
        components=11, output_rows=12, pixel_bytes=16, initial_q='initial-Q.bin'))
    return result


def initial_q():
    # Describe each affine transform by columns, independently of the probe.
    projection = [.5,0,0,0, 0,.5,0,0, 0,0,.5,0, 0,0,0,1]
    model = [.5,0,0,0, 0,.9375,0,0, 0,0,1,0, .25,-.25,.5,1]
    view = [1,0,0,0, 0,.5,0,0, 0,0,1,0, -.25,.15625,0,1]
    matrices = struct.pack('<48f', *(projection + model + view))
    return matrices + bytes([0xcc]) * (512*1024-len(matrices))


def expected(case):
    # Projection * view * model maps the source positions to screen vertices
    # A=(64,25), B=(48,40), C=(80,40). All samples avoid the edges: diagonal
    # edge determinants at half-integer centers have odd numerator.
    pixels = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            a = (40-(y+.5))/15
            c = ((x+.5)-48-16*a)/32
            b = 1-a-c
            if min(a,b,c) <= 0:
                pixels.extend((0,0,0,0))
            elif case['name'].startswith('Sascha-'):
                # Original colors: A green, B blue, C red; all alpha one.
                pixels.extend((c,a,b,1))
            else:
                # Screen barycentrics divided by clip W for smooth inputs;
                # noperspective deliberately uses the unmodified weights.
                weights = (a,b/2,c/4)
                denominator = sum(weights)
                sx = sum(t*v for t,v in zip(weights,(.125,.5,.875)))/denominator
                sy = sum(t*v for t,v in zip(weights,(.25,.75,1)))/denominator
                pixels.extend((sx,2*a+4*b+8*c,sy+.25,.75))
    return struct.pack('<%df' % len(pixels), *pixels), initial_q()


def verify_bytes(pixels, q, case):
    ep, eq = expected(case)
    image = dict(ok=False, bytes=len(pixels), expected_bytes=len(ep),
                 absolute_tolerance=TOLERANCE)
    if len(pixels) == len(ep):
        actual = struct.unpack('<32768f', pixels)
        reference = struct.unpack('<32768f', ep)
        # Background is exact. Reject NaNs explicitly; NaN comparisons alone
        # would fail open. The tolerance covers interpolator rounding only.
        bad = [i for i,(a,b) in enumerate(zip(actual,reference))
               if not math.isfinite(a) or abs(a-b) > (TOLERANCE if b else 0)]
        image.update(ok=not bad, checked_components=len(actual),
                     mismatches=len(bad), first=bad[:16],
                     max_abs_error=max(abs(a-b) for a,b in zip(actual,reference)))
    buffer = dict(ok=q == eq, bytes=len(q), expected_bytes=len(eq))
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',
        pixels=image, witness=buffer,
        sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(), Q=hashlib.sha256(q).hexdigest()))


def verify(directory, case, subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),
                        (directory/'Q.bin').read_bytes(), case)


def mutations():
    rows = []
    for case in cases():
        p,q = expected(case)
        assert verify_bytes(p,q,case)['verdict'] == 'PASS'
        edits = [('unexecuted',bytes([0xa5])*len(p),q),
                 ('empty',bytes(len(p)),q), ('truncated',p[:-4],q),
                 ('ubo-overwrite',p,bytes(4)+q[4:]),
                 ('guard-overwrite',p,q[:-4]+bytes(4))]
        index = (34*WIDTH+64)*4
        for label,component,value in [('missing',3,0),('duplicate',3,2),
                ('wrong-signed-or-unsigned',2,16),('nan',0,float('nan')),
                ('wrong-linear',1,7),('wrong-smooth',0,.9)]:
            changed = bytearray(p)
            struct.pack_into('<f', changed,4*(index+component),value)
            edits.append((label,changed,q))
        swapped = bytearray(p)
        for offset in range(0,len(p),16):
            swapped[offset:offset+4],swapped[offset+4:offset+8] = p[offset+4:offset+8],p[offset:offset+4]
        edits.append(('swapped-RG',swapped,q))
        for label,changed,buffer in edits:
            assert verify_bytes(changed,buffer,case)['verdict'] == 'FAIL', (case,label)
        rows.append(dict(case=case['name'], rejected=[e[0] for e in edits]))
    return dict(verdict='PASS', mutations_detected=sum(len(r['rejected']) for r in rows), cases=rows)
