#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the fixed normal input against fused/separate shader arithmetic."""
import ctypes
import ctypes.util
import math
import struct
from check import coverage, triangles, normal_component, W

def verify_margin():
    f = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    near = lambda x, d: struct.unpack('<f', struct.pack('<I',
        struct.unpack('<I', struct.pack('<f', x))[0] + d))[0]
    lib = ctypes.CDLL(ctypes.util.find_library('m'))
    lib.fmaf.argtypes = [ctypes.c_float] * 3
    lib.fmaf.restype = ctypes.c_float
    ma = lib.fmaf
    frac = lambda x: f(x - math.floor(x))

    # The old fixture sits on a discontinuity: equivalent -Z directions
    # choose different octahedral corners. This check prevents reintroducing it.
    rcp = f(1 / 511)
    assert f(f(511 * rcp) - 1) == 0
    assert ma(511, rcp, -1) == -(2 ** -27)
    assert ma(0, rcp, -1) == -1
    assert ma(1022, rcp, -1) == 1

    # +Z remains in the continuous positive-Z hemisphere for either unpack
    # result. The tiny X/Y values survive normalization but cannot select a
    # different octahedral corner. Bracket the common rsqrt and reciprocal.
    oct_values = {0.0}
    for length in [near(1.0, d) for d in (-1, 0, 1)]:
        xy = f(-(2 ** -27) * length)
        denominator = f(f(abs(xy) + abs(xy)) + length)
        for reciprocal in [near(f(1 / denominator), d) for d in (-1, 0, 1)]:
            oct_values.add(f(xy * reciprocal))
    pixels = set(i for tri in triangles() for i in coverage(tri))
    margin = 1
    checked = 0
    for i in pixels:
        x, y = i % W + .5, i // W + .5
        a, b = f(.06711056), f(.00583715)
        for dot in {f(f(x * a) + f(y * b)), ma(x, a, f(y * b)), ma(y, b, f(x * a))}:
            noise = frac(f(f(52.9829189) * frac(dot)))
            deband = f(f(noise * 2) - 1)
            expected = normal_component(i)
            for oct_value in oct_values:
                value = f(f(oct_value * .5) + .5)
                rounded = {f(value + f(deband * f(.5 / 1023))),
                           ma(deband, f(.5 / 1023), value),
                           ma(oct_value, .5, ma(deband, f(.5 / 1023), .5))}
                for result in rounded:
                    scaled = result * 1023
                    assert math.floor(scaled + .5) == expected, (i, result, scaled)
                    # Both nearest-1023 and floor-1024 UNORM conversion models
                    # agree at this centre input, unlike the 1/3 candidate.
                    assert math.floor(result * 1024) == expected, (i, result)
                    margin = min(margin, .5 - abs(scaled - expected))
                    checked += 1
    assert margin > 0
    return dict(verdict='PASS', covered_pixels=len(pixels), checked_variants=checked,
                octahedral_coordinates=sorted(oct_values), minimum_half_step_margin=margin,
                old_input_separate=0, old_input_fused=-(2 ** -27),
                scope='f32 fused/separate unpack/dot/add; rsqrt/reciprocal +/-1 ULP; two UNORM conversion models; no tolerance')

if __name__ == '__main__':
    import json
    print(json.dumps(verify_margin(), indent=2))
