#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and record the Niagara corpus draw with the real ARM ICD and no-op DRM.

No GPU executes; readback stays at its CPU initialization pattern. Each case
records exactly one native render whose mesh state words must match the
resource model, and keeps the packet for the paired kernel parser replay.
"""
import argparse, hashlib, importlib.util, json, os, re, struct, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check import cases
spec = importlib.util.spec_from_file_location('capture', HERE.parent/'check-native-mesh-submit.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def expected_state(case):
    v, p = case['max_vertices'], case['max_primitives']
    # Seventeen planes: position plus thirteen original user components.
    planes = 17
    return [0x40000d2c, ((planes + 1) << 8) | planes, 0x10, 0, 0, 0x1000000, None, 0x1000000, None, 1,
            (p - 1) | ((-(-p//16) - 1) << 16), 0x400800 | case['physical'], 0, 1 + planes*(-(-v//4))]


p = argparse.ArgumentParser(description=__doc__)
for name in ('driver', 'shim', 'probe', 'shaders', 'sysroot', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
out = a.out.resolve()
out.mkdir()
rows = []
for case in cases():
    d = out/case['name']
    d.mkdir()
    (d/'trace').mkdir()
    (d/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',
        ICD=dict(library_path=str(a.driver.resolve()), api_version='1.4.0'))))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'HK_', 'VK_', 'MESA_', 'AGX_', 'ASAHI_', 'M1_', 'M2_'))}
    settings = dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'), LD_PRELOAD=str(a.shim.resolve()),
        ASAHI_HOST_TRACE_DIR=str(d/'trace'), VK_DRIVER_FILES=str(d/'icd.json'), HK_EXPERIMENTAL_APPLE9='1',
        HK_APPLE9_MESH_ROUTE_LOG='1', MESA_SHADER_CACHE_DISABLE='true',
        MESA_VK_ENABLE_SUBMIT_THREAD='false', VK_LOADER_LAYERS_DISABLE='~implicit~', AGX_APPLE9_ASM='1',
        AGX_APPLE9_STATS='1', **case['env'])
    argv = ['qemu-aarch64-static', '-L', str(a.sysroot.resolve())]
    for k, v in settings.items():
        if ',' in v:
            env[k] = v
        else:
            argv += ['-E', k + '=' + v]
    argv += [str(a.probe.resolve()), str(a.shaders.resolve()/(case['name'] + '.mesh.spv')),
             str(a.shaders.resolve()/case['fragment']),str(a.shaders.resolve()/'inputs.bin'),str(d)]
    (d/'RUN.json').write_text(json.dumps(dict(argv=argv, environment=settings, gpu_execution=False), indent=2) + '\n')
    with (d/'stdout.log').open('w') as f:
        r = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
    log = (d/'stdout.log').read_text(errors='replace')
    assert r.returncode == 0 and log.count('SUBMIT_OK\n') == 1, (d, r.returncode, log[-2500:])
    routes = re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)', log)
    assert routes == [('1', str(case['physical']), str(case['max_vertices']), str(case['max_primitives']))], (d, routes)
    assert (d/'Q.bin').read_bytes() == (a.shaders/'inputs.bin').read_bytes() and (d/'pixels.bin').read_bytes() == bytes([0xa5])*65536
    renders = []
    for path in sorted((d/'trace').glob('*.commands.bin')):
        data = path.read_bytes()
        pos = 0
        index = path.name.split('.')[0]
        while pos < len(data):
            kind, size, vdm, cdm = struct.unpack_from('<4H', data, pos)
            pos += 8
            assert size and pos + size <= len(data)
            if kind == 0:
                assert size == 240
                address = struct.unpack_from('<Q', data, pos + 8)[0]
                packet = capture.read_gpu(d/'trace', index, address, 160)
                state = list(struct.unpack_from('<14I', packet))
                want = expected_state(case)
                assert all(w is None or s == w for s, w in zip(state, want)), (d, state, want)
                assert state[6] % 2 == state[8] % 2 == 0
                assert struct.unpack_from('<I', packet, 156)[0] == 0xc0000000
                (d/'stream.bin').write_bytes(packet)
                renders.append(dict(state=state, stream_sha256=hashlib.sha256(packet).hexdigest()))
            pos += size
    assert len(renders) == 1, (d, len(renders))
    rows.append(dict(case=case['name'], renders=renders))
    print(case['name'], 'native record PASS; state', ' '.join('%x' % w for w in renders[0]['state'][9:]), flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS', gpu_execution=False, cases=rows,
    driver_sha256=hashlib.sha256(a.driver.read_bytes()).hexdigest(),
    shim_sha256=hashlib.sha256(a.shim.read_bytes()).hexdigest()), indent=2) + '\n')
