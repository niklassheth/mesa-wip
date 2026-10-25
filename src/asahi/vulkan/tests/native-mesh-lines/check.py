# SPDX-License-Identifier: MIT
"""Independent inputs and integer pixel oracle for the two original line shaders."""
import struct

WIDTH, HEIGHT, Q_BYTES = 128, 64, 512 * 1024
SHAPES = {'glsl': (64, 126, (0, 1, 31, 32)),
          'slang': (126, 64, (0, 1, 31, 32, 42))}


def inputs(primitives):
    assert 0 <= primitives <= 42
    data = bytearray([0xa5]) * Q_BYTES
    # Three identity matrices; bind exact ranges at offsets 4096/8192/12288.
    for matrix in range(3):
        struct.pack_into('<16f', data, 4096 + 64 * matrix,
                         *(float(i % 5 == 0) for i in range(16)))
    for i in range(64):
        struct.pack_into('<I', data, 8192 + 4 * i, 63 - i)
    for i in range(126):
        struct.pack_into('<I', data, 8192 + 256 + 4 * i, i)
    struct.pack_into('<2I', data, 8192 + 760, 3 * primitives, 3 * primitives)
    for v in range(126):
        p = v // 3
        # Three coincident input positions produce one horizontal line.
        # Pixel centers lie well away from the rasterization boundary.
        struct.pack_into('<8f', data, 12288 + 32 * v,
                         -.75, 2 * (4.5 + p) / HEIGHT - 1, .5, 1,
                         10, 0, 0, 0)
    return bytes(data)


def expected(primitives):
    pixels = bytearray(WIDTH * HEIGHT * 4)
    for p in range(primitives):
        for x in range(16, 80):
            # Smooth interpolation at x+.5 on [16,80], UNORM nearest.
            red = ((2 * x - 31) * 255 + 64) // 128
            blue = ((159 - 2 * x) * 255 + 64) // 128
            offset = 4 * ((4 + p) * WIDTH + x)
            pixels[offset:offset + 4] = bytes((red, 0, blue, 255))
    return bytes(pixels)


def verify(pixels, q, primitives):
    want = expected(primitives)
    mismatch = (sum(a != b for a, b in zip(want, pixels))
                if len(want) == len(pixels) else len(want))
    guard = q == inputs(primitives)
    passed = mismatch == 0 and guard
    return dict(verdict='PASS' if passed else 'FAIL',
                fault_class=None if passed else 'unknown',
                mismatched_bytes=mismatch, input_buffer_unchanged=guard,
                expected_covered_pixels=64 * primitives)


def mutations():
    results = []
    for primitives in (1, 31, 32, 42):
        pixels, q = expected(primitives), inputs(primitives)
        assert verify(pixels, q, primitives)['verdict'] == 'PASS'
        # Every primitive has its own observable row, including the last.
        for p in range(primitives):
            changed = bytearray(pixels)
            start = 4 * ((4 + p) * WIDTH + 16)
            changed[start:start + 64 * 4] = bytes(64 * 4)
            assert verify(changed, q, primitives)['verdict'] == 'FAIL'
        changed = bytearray(pixels)
        changed[4 * (4 * WIDTH + 16)] ^= 1
        assert verify(changed, q, primitives)['verdict'] == 'FAIL'
        changed_q = bytearray(q)
        changed_q[-1] ^= 1
        assert verify(pixels, changed_q, primitives)['verdict'] == 'FAIL'
        results.append(dict(primitives=primitives, omissions_detected=primitives,
                            gradient_mutation=True, input_guard_mutation=True))
    return dict(verdict='PASS', controls=results)
