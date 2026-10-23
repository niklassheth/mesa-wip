# SPDX-License-Identifier: MIT
"""Exact source-derived local/output oracle; never imports GPU observations."""
import gzip
import struct

Q_BYTES = 0x80000
PIXEL_BYTES = 64 * 64 * 4
HEADER = struct.Struct('<8s6I')
PATCH = struct.Struct('<6I')


def shape(v, p, t, topology, control42=1, high_index=0):
    assert 1 <= v <= 256 and 1 <= p <= 512
    assert t >= v and 16 <= t <= 256 and not t & (t - 1)
    k = {6: 3, 1: 2, 0: 1}[topology]
    planes = 5 if topology == 0 else 4
    count = (p * k + 15) & ~15
    pitch = ((v + 3) & ~3) * 4
    local = count + 16 + pitch * planes
    allocation = max(128, 1 << (local - 1).bit_length())
    return dict(vertices=v, primitives=p, threads=t, topology=topology,
                planes=planes, local_B=local, allocation_B=allocation,
                control42=control42, high_index=high_index, indices=k, count_B=count,
                position_B=count+16, pitch_B=pitch)


def expected(s, vertices, primitives, cookie=0):
    initial = bytearray(b'\xa5' * Q_BYTES)
    struct.pack_into('<3I', initial, 0, vertices, primitives, cookie)
    q = bytearray(initial)
    v, p = min(vertices, s['vertices']), min(primitives, s['primitives'])
    k = s['indices']
    if v < k or not p:
        v = p = 0
    struct.pack_into('<8I', q, 0x40, v, p, s['vertices'], s['primitives'],
                     s['threads'], s['topology'], s['planes'], s['local_B'])
    for lane in range(s['threads']):
        struct.pack_into('<4I', q, 0x1000+lane*16, 0x43415032, lane, cookie, s['threads'])
    local = bytearray(s['local_B'])
    struct.pack_into('<H', local, s['count_B'], p)
    struct.pack_into('<I', local, s['count_B']+4, v)
    positions = {6: [(-1., -1.), (3., -1.), (-1., 3.)],
                 1: [(-1.+1/64, -1/64), (1.+1/64, -1/64)],
                 0: [(1/64, -1/64)]}[s['topology']]
    for i in range(v):
        role = k-1 if i == v-1 else i % k
        xy = positions[role]
        if s['high_index']:
            if s['topology'] == 6:
                xy = (xy[0], 3. if i == v-1 else -1.)
            elif s['topology'] == 1:
                xy = (3. if i == v-1 else -3., -1/64)
            elif i != v-1:
                xy = (4., 4.)
        for c, value in enumerate((*xy, 0., 1., 1.)[:s['planes']]):
            struct.pack_into('<f', local, s['position_B']+c*s['pitch_B']+4*i, value)
    for prim in range(p):
        base = 0 if s["high_index"] else prim % max(v//k, 1) * k
        for c in range(k):
            local[prim*k+c] = v-1 if prim == p-1 and c == k-1 else base+c
    q[0x4000:0x4000+len(local)] = local
    pixels = raster(s['topology'], min(p, 1) if s['high_index'] else p)
    return initial, q, pixels


def raster(topology, primitives):
    pixels = bytearray(PIXEL_BYTES)
    # Negative-height viewport: clip y=-1/64 maps to pixel center y=32.5.
    for y in range(64):
        for x in range(64):
            covered = topology == 6 or (y == 32 and (topology == 1 or x == 32))
            if covered:
                struct.pack_into('<f', pixels, (y*64+x)*4, primitives)
    return pixels


def write(path, s, v, p):
    initial, q, pixels = expected(s, v, p)
    patches = [(0, 8, 0, 0, 1, 0), (1, 8, 0, 0, 1, 0)]
    patches += [(1, 0x1000+lane*16+8, 0, 0, 1, 0) for lane in range(s['threads'])]
    raw = (HEADER.pack(b'HV2ORC01', Q_BYTES, PIXEL_BYTES, len(patches), 0, 0, 0) +
           initial + q + pixels + b'\xff' * Q_BYTES + b''.join(PATCH.pack(*x) for x in patches))
    path.write_bytes(gzip.compress(raw, mtime=0))
    # Check the dynamic protocol's materialized byte arrays for multiple cookies.
    for cookie in [1, 0x76543210, 0xffffffff]:
        targets = [bytearray(initial), bytearray(q)]
        for area, off, _, _, _, _ in patches:
            struct.pack_into('<I', targets[area], off, cookie)
        a, b, c = expected(s, v, p, cookie)
        assert targets == [a, b] and c == pixels
