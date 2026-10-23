#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run transitions through a nonexecuting aarch64 DRM recorder, without a proxy.

For device use, run the transitions executable directly. This host runner
expects untouched readback memory, not a rendered image.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('driver', 'shim', 'sysroot', 'probes', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--native', choices=('0', '1'), required=True)
    p.add_argument('--sequence', default='NENVNMNN')
    p.add_argument('--mode', choices=('plain', 'timestamp', 'merged', 'deferred'), default='plain')
    p.add_argument('--constant-table', action='store_true',
                   help='Use the V3/P1 mesh with a dynamically indexed immutable table')
    a = p.parse_args()
    assert 1 <= len(a.sequence) <= 8 and set(a.sequence) <= set('NEVM')
    a.output.mkdir(parents=True, exist_ok=False)
    trace = a.output / 'trace'
    trace.mkdir()
    icd = a.output / 'icd.json'
    icd.write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {
        'library_path': str(a.driver.resolve()), 'api_version': '1.4.305'}}))
    values = dict(LD_LIBRARY_PATH=str(a.sysroot.resolve() / 'usr/lib64'),
                  LD_PRELOAD=str(a.shim.resolve()),
                  ASAHI_HOST_TRACE_DIR=str(trace.resolve()),
                  HK_EXPERIMENTAL_APPLE9='1', HK_APPLE9_NATIVE_MESH=a.native,
                  VK_DRIVER_FILES=str(icd.resolve()), MESA_SHADER_CACHE_DISABLE='true')
    if a.mode == 'merged':
        # Host diagnostic only: prove BeginRendering actually resumed the
        # open render before the next native draw forced a stream split.
        values['ASAHI_MESA_DEBUG'] = 'perf'
    files = [a.probes.resolve() / name for name in (
        'transitions', 'constant.mesh.spv' if a.constant_table else 'native.mesh.spv',
        'emulated.mesh.spv', 'vertex.spv',
        'mesh.frag.spv', 'vertex.frag.spv')]
    cmd = ['qemu-aarch64-static', '-L', str(a.sysroot.resolve()),
           *[x for k, v in values.items() for x in ('-E', k + '=' + v)],
           *map(str, files), str(a.output.resolve()), a.sequence, a.mode]
    inputs = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [a.driver, a.shim, *files]}
    (a.output / 'RUN.json').write_text(json.dumps(dict(
        gpu_execution=False, queue_submission=True, native=a.native,
        sequence=a.sequence, mode=a.mode, constant_table=a.constant_table,
        argv=cmd, input_sha256=inputs), indent=2) + '\n')
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(('LD_', 'HK_', 'VK_', 'MESA_', 'AGX_', 'ASAHI_'))}
    with (a.output / 'stdout.log').open('w') as out:
        result = subprocess.run(cmd, env=env, stdout=out,
                                stderr=subprocess.STDOUT, timeout=180)
    log = (a.output / 'stdout.log').read_text()
    assert result.returncode == 1 and 'SUBMIT_OK' in log and 'Vulkan:' not in log, log
    assert (a.output / 'pixels.bin').read_bytes() == bytes([0xa5]) * 64 * 64 * 4
    if a.mode == 'merged' and a.native == '1':
        assert log.count('Merging render pass into the previous render') == 1, log
    print('Recorded without GPU execution; untouched readback verified.')


if __name__ == '__main__':
    main()
