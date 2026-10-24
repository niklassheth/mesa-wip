#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native shared and single-round publication cases and their independent CPU oracle.

The raster oracle evaluates exact integer edge functions at pixel centres for
the triangles the shader is specified to emit (rounds.mesh). It refuses any
geometry where a centre lies on an edge, so no fill-rule tie-break is assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

Q_BYTES = 512*1024
W, H = 128, 64
ROUND = 32


HERE = Path(__file__).resolve().parent


def caseset():
    return 'all'


def _rows(kind):
    if kind == 'shared':
        # API shared storage on the multi-round path (Wicked: 4 B, T32 L32 V64/P124).
        return [
            ('S0-V64P124-shared4-clip2', 32, 32, 32, 64, 124, 64, 124, dict(SHARED='1', CLIP='2')),
            ('S1-V64P126-shared1K', 32, 32, 32, 64, 126, 64, 126, dict(SHARED='256')),
            ('S2-V128P256-shared16', 32, 32, 32, 128, 256, 128, 256, dict(SHARED='4')),
            ('S3-L64-V64P96-shared4', 64, 32, 32, 64, 96, 64, 96, dict(SHARED='1')),
        ]
    if kind == 'single':
        # Single-round declarations whose extra outputs the class path cannot
        # carry (74128258089): always T32, masked for small workgroups.
        return [
            ('Z0-V33P42-control', 32, 32, 32, 33, 42, 33, 42, {}),
            ('Z1-V32P42-clip2', 32, 32, 32, 32, 42, 32, 42, dict(CLIP='2')),
            ('Z2-L16-V16P16-clip1', 16, 32, 32, 16, 16, 16, 16, dict(CLIP='1')),
            ('Z3-L1-V8P6-clip2', 1, 32, 32, 8, 6, 8, 6, dict(CLIP='2')),
            ('Z4-V20P10-shared4-clip1', 32, 32, 32, 20, 10, 20, 10, dict(CLIP='1', SHARED='1')),
        ]
    raise ValueError('unknown publication case set: ' + kind)


def cases(kind=None):
    if kind in (None, 'all'):
        return cases('shared') + cases('single')
    out = []
    for i, (name, logical, physical, lanes, mv, mp, v, p, env) in enumerate(_rows(kind or caseset())):
        env = dict(env)
        clip = int(env.pop('CLIP', 0))
        shared = int(env.pop('SHARED', 0))
        e = dict(HK_APPLE9_NATIVE_MESH='1', **env)
        words = (3*p + 3)//4
        out.append(dict(name=name, logical=logical, physical=physical, round=lanes,
                        max_vertices=mv, max_primitives=mp, vertices=v, primitives=p,
                        cookie=0x3b000000 + (i << 16), vertex_rounds=-(-v//lanes),
                        index_rounds=-(-words//lanes), index_words=words, mode=3, env=e,
                        **({'clip': clip} if clip else {}),
                        **({'shared_words': shared} if shared else {})))
    return out


def position(v):
    cell, corner = v//4, v % 4
    return (2 + 16*(cell % 8) + (12 if corner & 1 else 0),
            2 + 16*(cell//8) + (11 if corner & 2 else 0))


def triangle(p, vertices):
    full, remaining = vertices//4, vertices % 4
    unique = 2*full + (1 if remaining else 0)
    t = p % unique
    if t < 2*full:
        base = 4*(t//2)
        return (base, base+1, base+2) if t % 2 == 0 else (base+1, base+3, base+2)
    a = 4*full
    return {3: (a, a+1, a+2), 2: (a, a+1, a-1), 1: (a, a-1, a-2)}[remaining]


def coverage(tri):
    """Exact pixel-centre coverage, in doubled integer coordinates."""
    (x0, y0), (x1, y1), (x2, y2) = [(2*x, 2*y) for x, y in map(position, tri)]
    covered = set()
    for y in range(H):
        for x in range(W):
            px, py = 2*x + 1, 2*y + 1
            e = [(x1-x0)*(py-y0) - (y1-y0)*(px-x0),
                 (x2-x1)*(py-y1) - (y2-y1)*(px-x1),
                 (x0-x2)*(py-y2) - (y0-y2)*(px-x2)]
            inside = all(v > 0 for v in e) or all(v < 0 for v in e)
            on_edge = 0 in e and (all(v >= 0 for v in e) or all(v <= 0 for v in e))
            if not inside and on_edge:
                raise ValueError('pixel centre on an edge of %s at %d,%d' % (tri, x, y))
            if inside:
                covered.add(y*W + x)
    return covered


_cache = {}


def raster(case):
    key = (case['vertices'], case['primitives'], case.get('clip', 0))
    if key not in _cache:
        values = [0]*(W*H)
        tris = {}
        for p in range(case['primitives']):
            tri = triangle(p, case['vertices'])
            if tri not in tris:
                tris[tri] = coverage(tri)
                if not tris[tri]:
                    raise ValueError('triangle %s covers no pixel centre' % (tri,))
            for i in tris[tri]:
                # Pixel centres are 0.5 from both integer clip lines.
                x, y = i % W, i // W
                if case.get('clip', 0) >= 1 and not x + 0.5 > 40:
                    continue
                if case.get('clip', 0) >= 2 and not y + 0.5 < 24:
                    continue
                values[i] += 1
        referenced = {v for tri in tris for v in tri}
        assert referenced == set(range(case['vertices'])), 'unreferenced vertices'
        _cache[key] = struct.pack('<%df' % (W*H), *values)
    return _cache[key]


def witness(case):
    q = [0xcccccccc]*(Q_BYTES//4)
    v, p, cookie = case['vertices'], case['primitives'], case['cookie']
    q[4:7] = [v, p, cookie]
    n = case.get('shared_words', 0)
    for lane in range(case['logical']):
        if n:
            words = [cookie ^ (0x5a5a0000 + w) for w in range(n)]
            q[1024+lane] = (sum(words) + lane) & 0xffffffff
        else:
            q[1024+lane] = cookie + lane
    for i in range(v):
        x, y = position(i)
        for c, f in enumerate((x/64 - 1, y/32 - 1, 0.5, 1.0)):
            q[2048 + 4*i + c] = struct.unpack('<I', struct.pack('<f', f))[0]
    for i in range(p):
        q[4096 + 3*i:4096 + 3*i + 3] = list(triangle(i, v))
    return struct.pack('<%dI' % len(q), *q)


def compare(actual, expected, kind):
    if len(actual) != len(expected):
        return dict(ok=False, bytes=len(actual), expected_bytes=len(expected))
    fmt = '<%d' % (len(actual)//4) + ('f' if kind == 'f' else 'I')
    a, e = struct.unpack(fmt, actual), struct.unpack(fmt, expected)
    bad = [i for i in range(len(a)) if a[i] != e[i]]
    return dict(ok=not bad, checked_words=len(a), mismatches=len(bad),
                first=[dict(word=i, expected=e[i], actual=a[i]) for i in bad[:16]])


def classify(pixels, case):
    """Diagnostic only: which cells lost or gained coverage."""
    actual = struct.unpack('<%df' % (W*H), pixels)
    expected = struct.unpack('<%df' % (W*H), raster(case))
    cells = {}
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            x, y = i % W, i // W
            cell = (y - 2)//16*8 + (x - 2)//16 if x >= 2 and y >= 2 else -1
            cells.setdefault(cell, [0, 0])[0 if a < e else 1] += 1
    return dict(sum_expected=sum(expected), sum_actual=sum(actual),
                cells={str(k): dict(lost=v[0], gained=v[1]) for k, v in sorted(cells.items())})


def verify_bytes(pixels, q, case):
    image = compare(pixels, raster(case), 'f')
    buffer = compare(q, witness(case), 'I')
    result = dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',
                  pixels=image, witness=buffer,
                  rounds=dict(vertex=case['vertex_rounds'], index=case['index_rounds']),
                  sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),
                              Q=hashlib.sha256(q).hexdigest()))
    if not image['ok'] and len(pixels) == W*H*4:
        result['diagnosis'] = classify(pixels, case)
    return result


def verify(directory, case):
    return verify_bytes((directory/'pixels.bin').read_bytes(),
                        (directory/'Q.bin').read_bytes(), case)


def mutations():
    """Every mutation must be caught by the same verdict function."""
    detected = []

    def expect_fail(name, pixels, q, case):
        assert verify_bytes(pixels, q, case)['verdict'] == 'FAIL', (case['name'], name)
        detected.append(case['name'] + ':' + name)

    for case in cases():
        pixels, q = raster(case), witness(case)
        assert verify_bytes(pixels, q, case)['verdict'] == 'PASS'
        v, p = case['vertices'], case['primitives']
        # Lost/duplicated primitive at the first pixel of the last triangle.
        last = sorted(coverage(triangle(p - 1, v)))[0]
        for delta in (-1, 1):
            changed = bytearray(pixels)
            value = struct.unpack_from('<f', changed, 4*last)[0] + delta
            struct.pack_into('<f', changed, 4*last, value)
            expect_fail('last-primitive-%s' % ('lost' if delta < 0 else 'duplicated'),
                        bytes(changed), q, case)
        for name, word in [('leading-guard', 0), ('trailing-guard', Q_BYTES//4 - 1),
                           ('upper-vertex', 2048 + 4*(v - 1)), ('last-index', 4096 + 3*p - 1),
                           ('lane-31', 1024 + 31), ('count', 4)]:
            changed = bytearray(q)
            struct.pack_into('<I', changed, 4*word, 0x12345678)
            expect_fail(name, pixels, bytes(changed), case)
        expect_fail('unexecuted', bytes([0xa5])*len(pixels), bytes([0xcc])*len(q), case)
        expect_fail('truncated', pixels[:-4], q[:-4], case)
    return dict(verdict='PASS', mutations_detected=len(detected), cases=detected,
                native_evidence=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('directory', type=Path, nargs='?')
    p.add_argument('--case')
    a = p.parse_args()
    result = mutations() if a.self_test else verify(
        a.directory, next(c for c in cases() if c['name'] == a.case))
    print(json.dumps(result, indent=2))
    raise SystemExit(result['verdict'] != 'PASS')
