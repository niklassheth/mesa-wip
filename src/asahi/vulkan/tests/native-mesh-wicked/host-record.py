#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record unchanged Wicked mesh/task shaders with diagnostic primitive consumers.

The descriptors are dummy host fixtures. No GPU executes and no pixels are
validated. This is compilation/link/upload/packet evidence, not engine coverage.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capture', HERE.parent / 'check-native-mesh-submit.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
p = argparse.ArgumentParser(description=__doc__)
for name in ('corpus', 'coverage', 'driver', 'probe', 'shim', 'sync', 'sysroot', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
out = a.out.resolve()
out.mkdir()
coverage = json.loads(a.coverage.read_text())['entries']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


task = a.corpus.resolve() / 'compiled/turanszkij-WickedEngine--objectAS.hlsl.spv'
manifest = json.loads((a.corpus / 'BUILD-MANIFEST.json').read_text())
assert sha(task) == next(e['sha256'] for e in manifest if e.get('path') == str(task.relative_to(a.corpus.resolve())))
rows = []
for name, locations in [('objectMS', [6]), ('objectMS_prepass', [0, 1]),
                        ('objectMS_prepass_alphatest', [1, 2]), ('objectMS_simple', [2])]:
    d = out / name
    d.mkdir()
    (d / 'trace').mkdir()
    entry = next(e for e in coverage if e['path'].endswith('--' + name + '.hlsl.spv'))
    mesh = a.corpus.resolve() / entry['path']
    assert sha(mesh) == entry['sha256']
    fragment = '#version 460\n#extension GL_EXT_mesh_shader : require\n'
    for i, location in enumerate(locations):
        fragment += f'layout(location={location}) perprimitiveEXT flat in uint p{i};\n'
    fragment += 'layout(location=0) out vec4 color;\nvoid main() { color=vec4('
    fragment += ','.join(f'float(p{i}&255u)/255.0' for i in range(len(locations)))
    fragment += ',0.25' * (4 - len(locations)) + '); }\n'
    (d / 'consumer.frag').write_text(fragment)
    subprocess.run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', 'frag',
                    '-o', str(d / 'frag.spv'), str(d / 'consumer.frag')], check=True)
    subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(d / 'frag.spv')], check=True)
    (d / 'inputs.bin').write_bytes(bytes(512 * 1024))
    write(d / 'icd.json', dict(file_format_version='1.0.0', ICD=dict(
        library_path=str(a.driver.resolve()), api_version='1.4.0')))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'HK_', 'VK_', 'MESA_', 'AGX_', 'ASAHI_'))}
    settings = dict(LD_LIBRARY_PATH=str(a.sysroot.resolve() / 'usr/lib64'),
        LD_PRELOAD=str(a.sync.resolve()) + ':' + str(a.shim.resolve()),
        ASAHI_HOST_TRACE_DIR=str(d / 'trace'), ASAHI_HOST_RECORD_SYNCHRONOUS='1',
        WICKED_HOST_RECORD_ONLY='1', VK_DRIVER_FILES=str(d / 'icd.json'),
        HK_EXPERIMENTAL_APPLE9='1', HK_APPLE9_NATIVE_INDEX='1', HK_APPLE9_NATIVE_MESH='1',
        HK_APPLE9_NATIVE_TASK_BRIDGE='1', HK_APPLE9_RESEARCH_NATIVE_BATCH='1',
        HK_APPLE9_MESH_ROUTE_LOG='1', MESA_SHADER_CACHE_DISABLE='true',
        MESA_VK_ENABLE_SUBMIT_THREAD='false', VK_LOADER_LAYERS_DISABLE='~implicit~')
    argv = ['qemu-aarch64-static', '-L', str(a.sysroot.resolve())]
    for k, v in settings.items():
        argv += ['-E', k + '=' + v]
    argv += [str(a.probe.resolve()), str(mesh), str(d / 'frag.spv'), str(d / 'inputs.bin'), str(d), str(task)]
    write(d / 'RUN.json', dict(argv=argv, environment=settings, gpu_execution=False,
        source=entry, task_sha256=sha(task), consumer_sha256=sha(d / 'frag.spv')))
    with (d / 'stdout.log').open('w') as f:
        result = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
    log = (d / 'stdout.log').read_text()
    routes = re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)', log)
    good = result.returncode == 0 and log.count('SUBMIT_OK\n') == 1 and routes == [('1', '32', '64', '124')]
    renders = []
    if good:
        assert (d / 'pixels.bin').read_bytes() == bytes([0xa5]) * 32768
        assert (d / 'Q.bin').read_bytes() == bytes(512 * 1024)
        for path in sorted((d / 'trace').glob('*.commands.bin')):
            data = path.read_bytes()
            pos = 0
            index = path.name.split('.')[0]
            while pos < len(data):
                kind, size, vdm, cdm = struct.unpack_from('<4H', data, pos)
                pos += 8
                assert size and pos + size <= len(data)
                if kind == 0:
                    assert size == 240
                    packet = capture.read_gpu(d / 'trace', index, struct.unpack_from('<Q', data, pos + 8)[0], 160)
                    state = list(struct.unpack_from('<14I', packet))
                    components = state[1] & 255
                    assert state[0] == 0x40000d2c and state[2:6] == [0x10, 0, 0, 0x1000000]
                    assert state[9:] == [1, 123 | (7 << 16), 0x400820, 0, 1 + 47 * components], state
                    assert cdm != 65535
                    assert struct.unpack_from('<I', packet, 156)[0] == 0xc0000000
                    (d / 'stream.bin').write_bytes(packet)
                    renders.append(dict(state=state, sha256=hashlib.sha256(packet).hexdigest(), vdm=vdm, cdm=cdm))
                pos += size
        assert len(renders) == 1
    rows.append(dict(name=name, returncode=result.returncode, routes=routes, verdict='PASS' if good else 'FAIL',
                     fault_class=None if good else 'unknown', mesh_sha256=sha(mesh), renders=renders))
    write(out / 'RESULT.json', dict(verdict='PASS' if all(r['verdict'] == 'PASS' for r in rows) else 'FAIL',
        gpu_execution=False, cases=rows, driver_sha256=sha(a.driver), probe_sha256=sha(a.probe)))
    print(name, rows[-1]['verdict'], routes, flush=True)
    if result.returncode:
        raise SystemExit(1)
