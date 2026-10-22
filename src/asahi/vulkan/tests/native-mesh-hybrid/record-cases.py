#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay a bounded CTS list through the real ICD's compiler/command recorder.

The proxy suppresses submission and the DRM shim executes no GPU work. Result
statuses describe this harness only; image mismatches are not CTS device results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('driver', 'proxy', 'shim', 'cts', 'sysroot', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native', choices=('0', '1'), default='1')
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    trace = a.output / 'trace'
    trace.mkdir()
    icd = a.output / 'icd.json'
    icd.write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {
        'library_path': str(a.proxy.resolve()), 'api_version': '1.4.305'}}))
    vals = dict(COVERAGE_REAL_ICD=str(a.driver.resolve()),
                LD_LIBRARY_PATH=str(a.sysroot.resolve() / 'usr/lib64'),
                LD_PRELOAD=str(a.shim.resolve()), ASAHI_HOST_TRACE_DIR=str(trace.resolve()),
                HK_EXPERIMENTAL_APPLE9='1', HK_APPLE9_NATIVE_MESH=a.native,
                VK_DRIVER_FILES=str(icd.resolve()), MESA_SHADER_CACHE_DISABLE='true')
    cmd = ['qemu-aarch64-static', '-L', str(a.sysroot.resolve()),
           *[x for k, v in vals.items() for x in ['-E', k + '=' + v]],
           str(a.cts.resolve()), '--deqp-caselist-file=' + str(a.cases.resolve()),
           '--deqp-log-filename=' + str((a.output / 'results.qpa').resolve()),
           '--deqp-log-images=disable', '--deqp-shadercache=disable',
           '--deqp-watchdog=disable', '--deqp-crashhandler=disable', '--deqp-log-flush=enable']
    manifest = {'gpu_execution': False, 'queue_submission': False, 'native': a.native,
                'argv': cmd, 'inputs': {k: {'path': str(getattr(a, k).resolve()),
                                           'sha256': sha(getattr(a, k))}
                                      for k in ('driver', 'proxy', 'shim', 'cts', 'cases')}}
    (a.output / 'RUN.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(('LD_', 'HK_', 'VK_', 'MESA_', 'AGX_', 'ASAHI_'))}
    timed_out = False
    with (a.output / 'stdout.log').open('w') as out:
        try:
            result = subprocess.run(cmd, cwd=a.output, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, timeout=a.timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
            timed_out = True
    log = (a.output / 'stdout.log').read_text(errors='replace')
    qpa = a.output / 'results.qpa'
    xml = re.findall(r'<TestCaseResult\b.*?</TestCaseResult>', qpa.read_text() if qpa.exists() else '', re.S)
    rows = []
    for block in xml:
        node = ET.fromstring(block)
        result = node.find('Result')
        rows.append({'case': node.attrib['CasePath'], 'status': result.attrib['StatusCode'],
                     'description': ''.join(result.itertext())})
    summary = {'gpu_execution': False, 'queue_submission': False, 'returncode': code,
               'timed_out': timed_out, 'completed': len(rows),
               'api_errors': [r for r in rows if 'VK_ERROR_' in r['description']],
               'debug_errors': re.findall(r'^COVERAGE_VK .*VK_ERROR_.*$', log, re.M),
               'suppressed_submissions': len(re.findall(r'^HOST_RECORDING_ONLY', log, re.M)),
               'cases': rows}
    (a.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=2), flush=True)
    expected = a.cases.read_text().splitlines()
    actual = [r['case'] for r in rows]
    if (timed_out or code not in (0, 1) or sorted(actual) != sorted(expected)
            or summary['api_errors'] or summary['debug_errors']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
