#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record original line pipelines or execute them with a software Vulkan driver.

Both SPIR-V stages remain byte-identical to the pinned public corpus. Recording
uses the non-executing ARM DRM shim; the software mode never selects an Apple GPU.
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
from check import SHAPES, inputs, mutations, verify

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capture', HERE.parent / 'check-native-mesh-submit.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
p = argparse.ArgumentParser(description=__doc__)
for name in ('corpus', 'driver', 'probe', 'sysroot', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
for name in ('shim', 'sync'):
    p.add_argument('--' + name, type=Path)
p.add_argument('--mode', choices=('record', 'software'), required=True)
a = p.parse_args()
record = a.mode == 'record'
assert not record or (a.shim and a.sync)
out = a.out.resolve()
out.mkdir()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')
manifest = json.loads((a.corpus / 'SOURCE-MANIFEST.json').read_text())
rows = []
write(out / 'MUTATIONS.json', mutations())

for variant, (vertices, primitives, counts) in SHAPES.items():
    shaders = []
    pins = []
    for suffix in ('gshader_to_mshader.mesh.spv', 'gshader_to_mshader_mesh.frag.spv'):
        relative = f'shaders/gshader_to_mshader/{variant}/{suffix}'
        entry = next(e for e in manifest if e['repository'] == 'KhronosGroup/Vulkan-Samples'
                     and e['path'] == relative)
        path = a.corpus.resolve() / 'sources/KhronosGroup-Vulkan-Samples' / relative
        assert sha(path) == entry['sha256']
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(path)], check=True)
        shaders.append(path)
        pins.append(entry)
    for count in (counts[-1],) if record else counts:
        d = out / f'{variant}-{count}'
        d.mkdir()
        if record: (d / 'trace').mkdir()
        (d / 'inputs.bin').write_bytes(inputs(count))
        write(d / 'icd.json', dict(file_format_version='1.0.0', ICD=dict(
            library_path=str(a.driver.resolve()), api_version='1.4.0')))
        settings = dict(LD_LIBRARY_PATH=str(a.sysroot.resolve() / 'usr/lib64'),
            VK_DRIVER_FILES=str(d / 'icd.json'), VK_LOADER_LAYERS_DISABLE='~implicit~',
            MESA_SHADER_CACHE_DISABLE='true', MESA_VK_ENABLE_SUBMIT_THREAD='false')
        if record:
            settings.update(LD_PRELOAD=str(a.sync.resolve()) + ':' + str(a.shim.resolve()),
                ASAHI_HOST_TRACE_DIR=str(d / 'trace'), ASAHI_HOST_RECORD_SYNCHRONOUS='1',
                LINES_HOST_RECORD_ONLY='1', HK_EXPERIMENTAL_APPLE9='1',
                HK_APPLE9_NATIVE_INDEX='1', HK_APPLE9_NATIVE_MESH='1',
                HK_APPLE9_MESH_ROUTE_LOG='1', HK_APPLE9_RESEARCH_NATIVE_BATCH='1')
        else:
            settings.update(LINES_SOFTWARE_REFERENCE='1', LIBGL_ALWAYS_SOFTWARE='1',
                            LP_NUM_THREADS='2')
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('LD_', 'HK_', 'VK_', 'MESA_', 'AGX_', 'ASAHI_', 'LINES_'))}
        argv = ['qemu-aarch64-static', '-L', str(a.sysroot.resolve())]
        for k, v in settings.items(): argv += ['-E', k + '=' + v]
        argv += [str(a.probe.resolve()), *(str(s) for s in shaders),
                 str(d / 'inputs.bin'), str(d)]
        write(d / 'RUN.json', dict(argv=argv, environment=settings, sources=pins,
                                  gpu_execution=False, mode=a.mode))
        with (d / 'stdout.log').open('w') as f:
            result = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        log = (d / 'stdout.log').read_text()
        good = result.returncode == 0 and log.count('SUBMIT_OK\n') == 1
        row = dict(variant=variant, count=count, returncode=result.returncode, sources=pins)
        if good and record:
            routes = re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)', log)
            assert routes == [('1', '32', str(vertices), str(primitives))], routes
            assert (d / 'Q.bin').read_bytes() == inputs(count)
            assert (d / 'pixels.bin').read_bytes() == bytes([0xa5]) * 32768
            renders = []
            for path in sorted((d / 'trace').glob('*.commands.bin')):
                data, pos = path.read_bytes(), 0
                index = path.name.split('.')[0]
                while pos < len(data):
                    kind, size, vdm, cdm = struct.unpack_from('<4H', data, pos)
                    pos += 8
                    assert size and pos + size <= len(data)
                    if kind == 0:
                        assert size == 240
                        address = struct.unpack_from('<Q', data, pos + 8)[0]
                        packet = capture.read_gpu(d / 'trace', index, address, 160)
                        state = list(struct.unpack_from('<14I', packet))
                        assert state[0:6] == [0x40000d2c, 0x908, 0x10, 0, 0, 0x1000000], state
                        assert state[9:] == [1, (primitives - 1) | (((primitives + 15) // 16 - 1) << 16),
                                             0x400820, 0, 1 + ((vertices + 3) // 4) * 8], state
                        assert cdm == 65535
                        assert struct.unpack_from('<I', packet, 128)[0] == 0x70000100
                        assert struct.unpack_from('<I', packet, 156)[0] == 0xc0000000
                        (d / 'stream.bin').write_bytes(packet)
                        renders.append(dict(state=state, sha256=hashlib.sha256(packet).hexdigest(), vdm=vdm, cdm=cdm))
                    pos += size
            assert len(renders) == 1
            row.update(routes=routes, renders=renders)
        elif good:
            row['pixels'] = verify((d / 'pixels.bin').read_bytes(), (d / 'Q.bin').read_bytes(), count)
            good = row['pixels']['verdict'] == 'PASS'
        row.update(verdict='PASS' if good else 'FAIL', fault_class=None if good else 'unknown')
        rows.append(row)
        write(out / 'RESULT.json', dict(verdict='PASS' if all(r['verdict'] == 'PASS' for r in rows) else 'FAIL',
            gpu_execution=False, mode=a.mode, cases=rows, driver_sha256=sha(a.driver), probe_sha256=sha(a.probe)))
        print(variant, count, a.mode, row['verdict'], flush=True)
        if not good:
            raise SystemExit(1)
