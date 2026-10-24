#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the fixtures and run software controls; never native evidence."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check import cases, verify, mutations


def shaders(out):
    d = out/'shaders'
    d.mkdir()
    commands = []
    for case in cases():
        defines = ['-DLOGICAL_THREADS=%d' % case['logical'], '-DMAX_VERTICES=%d' % case['max_vertices'],
                   '-DMAX_PRIMITIVES=%d' % case['max_primitives'],
                   '-DCLIP_DISTANCES=%d' % case.get('clip', 0),
                   '-DSHARED_WORDS=%d' % case.get('shared_words', 0)]
        commands.append(['glslangValidator', '--target-env', 'vulkan1.3', *defines,
                         str(HERE/'rounds.mesh'), '-o', str(d/(case['name'] + '.mesh.spv'))])
    commands.append(['glslangValidator', '--target-env', 'vulkan1.3', str(HERE/'count.frag'),
                     '-o', str(d/'count.frag.spv')])
    for cmd in commands:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    (out/'BUILD.json').write_text(json.dumps(commands, indent=2) + '\n')
    return d


def argv(probe, d, directory, case):
    return [str(probe), str(d/(case['name'] + '.mesh.spv')), str(d/'count.frag.spv'), str(directory),
            str(case['vertices']), str(case['primitives']), str(case['max_primitives']), str(case['cookie'])]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'lavapipe', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir()
    d = shaders(out)
    (out/'MUTATIONS.json').write_text(json.dumps(mutations(), indent=2) + '\n')
    (out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',
        ICD=dict(library_path=str(a.lavapipe.resolve()), api_version='1.4.0'))))
    rows = []
    for case in cases():
        directory = out/case['name']
        directory.mkdir()
        e = {k: v for k, v in os.environ.items() if not k.startswith(('HK_', 'VK_', 'LD_', 'MESA_', 'AGX_', 'ASAHI_', 'M1_', 'M2_'))}
        e.update(VK_DRIVER_FILES=str(out/'icd.json'), VK_LOADER_LAYERS_DISABLE='~implicit~',
                 LP_NUM_THREADS='2', M2_SOFTWARE_REFERENCE='1')
        cmd = argv(a.probe.resolve(), d, directory, case)
        with (directory/'stdout.log').open('w') as f:
            r = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT, timeout=60)
        log = (directory/'stdout.log').read_text()
        assert r.returncode == 0 and 'llvmpipe' in log and log.count('SUBMIT_OK\n') == 1, (case['name'], log[-2000:])
        result = verify(directory, case)
        (directory/'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        assert result['verdict'] == 'PASS', (case['name'], result)
        rows.append(dict(case=case['name'], argv=cmd, sha256=result['sha256']))
        print(case['name'], 'software oracle PASS', flush=True)
    (out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS', software_only=True, native_evidence=False,
        draws=len(rows), results=rows, lavapipe_sha256=hashlib.sha256(a.lavapipe.read_bytes()).hexdigest()),
        indent=2) + '\n')
