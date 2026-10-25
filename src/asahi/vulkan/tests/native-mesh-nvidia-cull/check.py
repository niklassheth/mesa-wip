#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Input and raster oracle for the original NVIDIA culling pipelines."""
import hashlib
import importlib.util
import struct
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / 'raster.py'
if not path.exists():
    path = HERE.parent / 'native-mesh-nvidia/check.py'
spec = importlib.util.spec_from_file_location('raster', path)
raster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raster)
PROFILES = ('visible', 'mixed', 'culled')
CASES = [(task, profile) for task in (0, 1) for profile in PROFILES]


def retained(primitive, profile):
    return profile == 'visible' or (profile == 'mixed' and primitive % 3 != 1)


@lru_cache(None)
def inputs(profile):
    assert profile in PROFILES
    q = bytearray(raster.inputs())
    # Valid viewport dimensions for both triangle and task-box culling.
    struct.pack_into('<2i4f', q, 4096 + 240, 128, 64, 128, 64, 128, 64)
    struct.pack_into('<4f', q, 4096 + 192, 0, 0, 2, 1)
    # Descriptor 9 spans the whole in-frustum meshlet. A zero cone angle
    # disables the sample's early cone rejection; task lane 0 emits meshlet 7.
    struct.pack_into('<4f', q, 8192 + 192, -2**-20, -2**-20, .25, 1)
    struct.pack_into('<4f', q, 8192 + 208, 2**-20, 2**-20, .25, 1)
    struct.pack_into('<f', q, 8192 + 236, 1)
    struct.pack_into('<4I', q, 12288 + 9*16,
                     63 << 24, (125 << 24) | 0xffffff, 2 << 24, 16)
    for p in range(126):
        tri = raster.triangle(p)
        # The Vulkan sample keeps negative screen-space signed area. Reverse
        # only retained triangles; rejected triangles still cover real pixels
        # when culling is broken. Fixed-function face culling is disabled.
        if retained(p, profile):
            tri = (tri[0], tri[2], tri[1])
        q[16384 + 224 + p*3:16384 + 227 + p*3] = bytes(tri)
    return bytes(q)


@lru_cache(None)
def expected(profile, primitives=126):
    pixels = bytearray(raster.W * raster.H * 4)
    triangles = {raster.triangle(p) for p in range(126)}
    coverage = {tri: raster.coverage(tri) for tri in triangles}
    for p in range(primitives):
        if not retained(p, profile):
            continue
        tri = raster.triangle(p)
        color = raster.source_color(tri[0] // 4)
        for pixel in coverage[tri]:
            for c in range(4):
                pixels[4*pixel+c] = min(255, pixels[4*pixel+c] + color[c])
    return bytes(pixels)


def compare(actual, wanted):
    if len(actual) != len(wanted):
        return dict(ok=False, bytes=len(actual), expected_bytes=len(wanted))
    bad = [i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b]
    return dict(ok=not bad, checked_bytes=len(actual), mismatches=len(bad), first=bad[:16])


def verify(directory, profile):
    pixels = (directory / 'pixels.bin').read_bytes()
    q = (directory / 'Q.bin').read_bytes()
    image, buffer = compare(pixels, expected(profile)), compare(q, inputs(profile))
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',
                pixels=image, witness=buffer,
                retained_primitives=sum(retained(p, profile) for p in range(126)),
                sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),
                            Q=hashlib.sha256(q).hexdigest()))


def mutations():
    # Cull-ignore, cull-all and inverted decisions must all change readback.
    images = [expected(p) for p in PROFILES]
    assert all(images[i] != images[j] for i in range(3) for j in range(i))
    assert images[0] == raster.expected() and not any(images[2])
    assert expected('mixed') != expected('mixed', 125)
    return dict(verdict='PASS', retained={p: sum(retained(i,p) for i in range(126))
                for p in PROFILES}, cross_profile_mismatch_bytes={
                a+'-'+b: sum(x != y for x,y in zip(expected(a),expected(b)))
                for i,a in enumerate(PROFILES) for b in PROFILES[:i]},
                precision=raster.precision_margin())
