#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Authored clip vectors and exact rational pixel oracle. No device access.

The guard-band polygon is a hypothesis to discriminate on the device, not an
oracle for an established native DATA layout. Coverage is separately derived
from API clip volume, viewport and scissor, using a convex-polygon edge test.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

def tri(xy, z=F(1, 2)):
    return [[F(x), F(y), z, F(1)] for x, y in xy]

H1 = tri([(-F(1, 2), -F(1, 2)), (F(1, 2), -F(1, 2)), (0, F(1, 2))])
VECTORS = {'CONTROL': H1,
           'BOUNDARY': tri([(-1, -1), (1, -1), (-1, 1)], F(0)),
           'FULLSCREEN': tri([(-1, -1), (3, -1), (-1, 3)], F(0))}
for name, extent in [('125', F(5, 4)), ('150', F(3, 2)), ('175', F(7, 4))]:
    VECTORS['X_' + name] = tri([(-F(1, 2), -F(1, 2)),
                                (extent, -F(1, 2)), (-F(1, 2), F(1, 2))])
VECTORS['NEGATIVE_X_175'] = [[-v[0], *v[1:]] for v in VECTORS['X_175']]
VECTORS['Y_175'] = [[v[1], v[0], *v[2:]] for v in VECTORS['X_175']]
for name, z in [('NEAR', -F(1, 2)), ('FAR', F(3, 2))]:
    VECTORS[name] = [v.copy() for v in H1]
    VECTORS[name][2][2] = z
    VECTORS[name + '_OUTSIDE'] = [[v[0], v[1], z, v[3]] for v in H1]
VECTORS['POSITIVE_W'] = [[c * w for c in v]
                         for v, w in zip(H1, [F(1), F(2), F(1, 2)])]

def polygon_clip(vertices, extent=F(1), depth=True):
    """Intersect an authored convex polygon with homogeneous half-spaces."""
    planes = [(1, 0, 0, extent), (-1, 0, 0, extent),
              (0, 1, 0, extent), (0, -1, 0, extent)]
    if depth:
        planes += [(0, 0, 1, 0), (0, 0, -1, 1)]
    polygon = [v.copy() for v in vertices]
    for plane in planes:
        out = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            da = sum(x * y for x, y in zip(a, plane))
            db = sum(x * y for x, y in zip(b, plane))
            if da >= 0:
                out.append(a)
            if (da < 0) != (db < 0):
                t = da / (da - db)
                out.append([x + t * (y - x) for x, y in zip(a, b)])
        polygon = []
        for v in out:
            if not polygon or v != polygon[-1]:
                polygon.append(v)
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
    return polygon

def snap(x):
    # The measured 1/256-pixel snapping contract, ties away from zero.
    return (-1 if x < 0 else 1) * int(abs(x) * 256 + F(1, 2))

def screen(poly, viewport):
    sx, sy, sz, tx, ty, tz = map(F, viewport)
    return [[snap(v[0] / v[3] * sx + tx), snap(v[1] / v[3] * sy + ty),
             v[2] / v[3] * sz + tz] for v in poly]

def pixels(poly, viewport, scissor):
    p = screen(poly, viewport)
    result = bytearray(64 * 64 * 4)
    if len(p) < 3:
        return bytes(result)
    area = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(p, p[1:] + p[:1]))
    if area < 0:
        p.reverse()
    if not area:
        return bytes(result)
    for y in range(scissor[1], scissor[3]):
        for x in range(scissor[0], scissor[2]):
            covered = True
            for a, b in zip(p, p[1:] + p[:1]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                edge = dx * (y * 256 + 128 - a[1]) - dy * (x * 256 + 128 - a[0])
                covered &= edge > 0 or (edge == 0 and (dy < 0 or (dy == 0 and dx > 0)))
            if covered:
                off = (y * 64 + x) * 4
                result[off:off + 4] = b'\xff\x00\x00\xff'
    return bytes(result)

def serialized(poly):
    return [[str(c) for c in v] for v in poly]

def bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def make(out, header):
    out.mkdir(parents=True, exist_ok=True)
    fields = []
    rows = []
    for name, vertices in VECTORS.items():
        words = [bits(x) for v in vertices for x in v]
        fields.append('   {"H1_CLIP_' + name + '", {' +
                      ', '.join(f'0x{x:08x}' for x in words) + '}},')
        position = ',\n    '.join('vec4(' + ', '.join(str(float(x)) for x in v) + ')'
                                  for v in vertices)
        declaration = 'const vec4 positions[3] = vec4[3](\n    ' + position + '\n);\n'
        (out / (name + '.vert')).write_text('#version 450\n' + declaration +
            'void main() { gl_Position = positions[gl_VertexIndex]; }\n')
        (out / (name + '.mesh')).write_text('#version 460\n'
            '#extension GL_EXT_mesh_shader : require\n'
            'layout(local_size_x=16) in;\nlayout(triangles,max_vertices=3,max_primitives=1) out;\n' +
            declaration + 'void main() {\n    SetMeshOutputsEXT(3,1);\n'
            '    uint i = gl_LocalInvocationIndex;\n'
            '    if (i < 3) gl_MeshVerticesEXT[i].gl_Position = positions[i];\n'
            '    if (i == 0) gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0,1,2);\n}\n')
        rows.append(dict(name=name, source=name, depth_clip=True,
                         viewport=[32, -32, 1, 32, 32, 0], scissor=[0, 0, 64, 64]))
    for name in ['NEAR', 'FAR']:
        rows.append(dict(name=name + '_CLAMP', source=name, depth_clip=False,
                         viewport=[32, -32, 1, 32, 32, 0], scissor=[0, 0, 64, 64]))
    rows += [dict(name='SCISSOR_HALF', source='FULLSCREEN', depth_clip=True,
                  viewport=[32, -32, 1, 32, 32, 0], scissor=[0, 0, 32, 64]),
             dict(name='VIEWPORT_INSET', source='FULLSCREEN', depth_clip=True,
                  viewport=[24, -24, 1, 32, 32, 0], scissor=[8, 8, 56, 56])]
    for row in rows:
        v = VECTORS[row['source']]
        poly = polygon_clip(v, depth=row['depth_clip'])
        guard = polygon_clip(v, F(3, 2), depth=row['depth_clip'])
        image = pixels(poly, row['viewport'], row['scissor'])
        # For these vectors the API and 1.5w guard polygon must cover the
        # same pixel centres inside the effective viewport/scissor.
        assert image == pixels(guard, row['viewport'], row['scissor'])
        filename = row['name'] + '.rgba8'
        (out / filename).write_bytes(image)
        row.update(positions=serialized(v), api_clip_polygon=serialized(poly),
                   guard_1_5_hypothesis_polygon=serialized(guard),
                   guard_hypothesis_fan_triangles=max(0, len(guard) - 2),
                   guard_hypothesis_screen=serialized(screen(guard, row['viewport'])),
                   native_program='H1_CLIP_' + row['source'],
                   explicit_formation_class_drops=int(any(
                       not (w > 0 and abs(x) < w and abs(y) < w and 0 <= z <= w)
                       for x, y, z, w in v)),
                   red_pixels=sum(image[i] == 255 for i in range(0, len(image), 4)),
                   image=filename, image_sha256=hashlib.sha256(image).hexdigest())
    (out / 'red.frag').write_text('#version 450\nlayout(location=0) out vec4 color;\n'
                                 'void main() { color = vec4(1,0,0,1); }\n')
    # API geometry oracle is independent of any expected native wire format.
    by_name = {r['name']: r for r in rows}
    assert by_name['CONTROL']['red_pixels'] == 512
    assert by_name['FULLSCREEN']['red_pixels'] == 4096
    assert by_name['SCISSOR_HALF']['red_pixels'] == 2048
    assert by_name['VIEWPORT_INSET']['red_pixels'] == 2304
    assert by_name['NEAR']['red_pixels'] == by_name['FAR']['red_pixels'] == 384
    assert by_name['NEAR_CLAMP']['red_pixels'] == by_name['FAR_CLAMP']['red_pixels'] == 512
    assert by_name['NEAR_OUTSIDE']['red_pixels'] == by_name['FAR_OUTSIDE']['red_pixels'] == 0
    assert by_name['POSITIVE_W']['image_sha256'] == by_name['CONTROL']['image_sha256']
    assert by_name['FULLSCREEN']['guard_hypothesis_fan_triangles'] == 3
    (out / 'ARMS.json').write_text(json.dumps(dict(
        scope='Host analysis and device qualification inputs; no GPU result',
        order='CONTROL, then each remaining row; paired VS/native draws; close with CONTROL',
        physical_threads=[16, 1, 1], grid=[1, 1, 1],
        image_format='64x64 row-major RGBA8, red on transparent black, one sample',
        data_status='Record layout/counts outside H1 are unknown. Guard polygon is a hypothesis, not a byte oracle.',
        stop_policy='20s deadline; MemoryMax=2G, MemorySwapMax=0; freeze unit and retain VM on GPU error/deadline',
        arms=rows), indent=2) + '\n')
    if header:
        header.write_text('/* Copyright 2026 Asahi Linux Contributors\n'
            ' * SPDX-License-Identifier: MIT\n * Generated by generate.py from authored source vertices.\n */\n'
            'static const struct {\n   const char *name;\n   uint32_t positions[12];\n'
            '} clip_fixtures[] = {\n' + '\n'.join(fields) + '\n};\n')
    print(f'{len(VECTORS)} source geometries, {len(rows)} arms; exact coverage invariants pass')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--header', type=Path)
    a = p.parse_args()
    make(a.out, a.header)
