#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare one M1 kit without accessing hardware or changing a queue."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('/home/nsheth/Projects/asahi')
sys.path.insert(0, str(HERE))
from check import cases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'driver', 'probes-build', 'kernel-manifest', 'software', 'records', 'parser'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--driver-revision', required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir()
    kernel = json.loads(a.kernel_manifest.read_text())
    software = json.loads((a.software/'RESULT.json').read_text())
    records = json.loads((a.records/'RESULT.json').read_text())
    parser = json.loads((a.parser/'RESULT.json').read_text())
    assert all(x['verdict'] == 'PASS' for x in (software, records, parser))
    assert software['draws'] == len(records['cases']) == len(parser['cases']) == 6
    assert records['driver_sha256'] == digest(a.driver)
    for folder in ('bin', 'lib', 'source', 'shaders', 'drirc-empty', 'evidence'):
        (out/folder).mkdir()
    for name in ('run.py', 'check.py', 'MODEL.md'):
        shutil.copy2(HERE/name, out/name)
    common = HERE.parent/'native-stage-batching'
    for name in ('kmsg.py', 'native_log_filter.py'):
        shutil.copy2(common/name, out/name)
    setup = (common/'setup-probe.py').read_text().replace(
        "'bin/gs-transitions','bin/tes-transitions'", "'bin/mesh-m1',").replace(
        "'bin/gs-transitions', 'bin/tes-transitions',", "'bin/mesh-m1',")
    assert 'gs-transitions' not in setup and 'tes-transitions' not in setup
    (out/'setup-probe.py').write_text(setup)
    shutil.copytree(HERE, out/'source/native-mesh-m1', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'source/native-stage-batching').mkdir()
    for name in ('kmsg.py', 'native_log_filter.py', 'vm-audit.c', 'setup-probe.py'):
        shutil.copy2(common/name, out/'source/native-stage-batching'/name)
    shutil.copy2(a.probes_build/'mesh-m1', out/'bin/mesh-m1')
    shutil.copy2(a.probes_build/'vm-audit.so', out/'lib/vm-audit.so')
    shutil.copy2(a.driver, out/'lib/libvulkan_asahi.so')
    shutil.copy2(a.kernel_manifest, out/'KERNEL.json')
    shutil.copytree(a.software/'shaders', out/'shaders', dirs_exist_ok=True)
    (out/'drirc-empty/README').write_text('Empty drirc directory for isolated processes.\n')
    for source, name in ((a.software/'RESULT.json', 'SOFTWARE.json'),
                         (a.software/'MUTATIONS.json', 'MUTATIONS.json'),
                         (a.records/'RESULT.json', 'DRIVER-RECORDS.json'),
                         (a.parser/'RESULT.json', 'KERNEL-PARSER.json'),
                         (a.software/'BUILD.json', 'SHADER-BUILD.json')):
        shutil.copy2(source, out/'evidence'/name)
    source_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=HERE, text=True)
    assert not dirty, 'commit source before sealing the kit'
    plan = dict(cases=cases(), driver_revision=a.driver_revision, source_revision=source_revision,
        kernel_version=kernel['uts_version'], kernel_release=kernel['release'], mode=3,
        budget=dict(api_draws=6, submissions=6, native_render_works=6, native_object_dispatches=6,
                    draws_per_process=1, seconds=120, retries=0),
        first_failure='stop; retain uncertain in-flight backing for the operator',
        qualified=False, production_changes=False, hardware_trials=0,
        oracle=dict(pixels=8192, witness_bytes=524288, format='R32_SFLOAT', blend='ONE/ONE ADD'),
        driver_sha256=digest(a.driver))
    write(out/'PLAN.json', plan)
    inputs = {}
    for key in ('image', 'dtb'):
        inputs[kernel[key]] = digest(WORKSPACE/kernel[key])
        assert inputs[kernel[key]] == kernel[key+'_sha256']
    boot = {k: kernel[k] for k in ('name', 'image', 'dtb', 'uts_version')}
    boot['extra_args_append'] = 'asahi_m4.native_batch=16 asahi_m4.native_batch_payload=0'
    write(out/'KIT.json', dict(format='kit-v1', name=out.name, device='air',
        boot=dict(kernel=boot, fresh=True), headless=True, timing=False,
        on_fault='continue-with-reboot', fault_watch=True,
        requires=[dict(kernel_version=kernel['uts_version']), dict(kernel_release=kernel['release']),
            dict(value='/sys/kernel/debug/asahi_m4/native_mesh', equals='0'),
            dict(path_exists='/sys/kernel/debug/asahi_m4/dep_stats'),
            dict(path_exists='/sys/class/drm/renderD128/device/asahi_native_mesh'),
            dict(cmdline_has='asahi_m4.native_batch=16'),
            dict(fact='kernel.cmdline', matches=r'(?:^|\s)asahi_m4\.native_batch_payload=[01](?:\s|$)')],
        setup_probe=dict(argv=['python3','-I','setup-probe.py'],
            files=['setup-probe.py','bin/mesh-m1','lib/vm-audit.so','lib/libvulkan_asahi.so'], timeout=30),
        stage=dict(dir='/root/kit-'+out.name, files='*'),
        run=dict(where='device', argv=['python3','-I','run.py','{dev_out}'], env={}, memory_max='2G'),
        budget_seconds=120, verdict='VERDICT.json', inputs=inputs,
        expected_faults=[r'(?:^| )G16: native_mesh=(?:0|3) \(generation [0-9]+\)$']))
    write(out/'PROVENANCE.json', dict(source_revision=source_revision, driver_revision=a.driver_revision,
        kernel_revision=kernel['source_commit'], software_source=str(a.software.resolve()),
        records_source=str(a.records.resolve()), parser_source=str(a.parser.resolve()),
        software_draws=6, software_native_evidence=False, host_native_recordings=6,
        parser_streams=6, gpu_executions=0, rehearsal='one case; all native audits remain enabled',
        pins={str(p.resolve()):digest(p) for p in [a.driver, a.probes_build/'mesh-m1',
            a.probes_build/'vm-audit.so', a.software/'RESULT.json',a.records/'RESULT.json',a.parser/'RESULT.json']}))
    source_manifest = {str(p.relative_to(out)):digest(p) for p in sorted((out/'source').rglob('*')) if p.is_file()}
    write(out/'SOURCE-MANIFEST.json', source_manifest)
    shutil.copy2(HERE/'README.md', out/'README.md')
    print(out)


if __name__ == '__main__':
    main()
