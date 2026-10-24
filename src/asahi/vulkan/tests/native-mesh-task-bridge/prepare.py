#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal the task-to-native-mesh candidate with reproducible host receipts."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
from check import cases
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
for name in ('out', 'prior-kit', 'driver', 'probes', 'host', 'records', 'parser', 'corpus'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--driver-revision', required=True)
a = p.parse_args()
out = a.out.resolve()
out.mkdir()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, obj): (out/name).write_text(json.dumps(obj, indent=2)+'\n')
host = json.loads((a.host/'RESULT.json').read_text())
records = json.loads((a.records/'RESULT.json').read_text())
parser = json.loads((a.parser/'RESULT.json').read_text())
assert host['verdict'] == records['verdict'] == parser['verdict'] == 'PASS'
assert len(host['software']) == len(records['cases']) == len(parser['cases']) == len(cases())
assert records['driver_sha256'] == sha(a.driver) and parser['qualified_expected'] == 'reject'
prior = a.prior_kit.resolve()
for name in ('bin', 'lib', 'source', 'evidence', 'drirc-empty'): (out/name).mkdir()
for name in ('mesh-task', 'mesh-task-ubo'): shutil.copy2(a.probes/name, out/'bin'/name)
shutil.copy2(a.driver, out/'lib/libvulkan_asahi.so')
shutil.copy2(prior/'lib/vm-audit.so', out/'lib/vm-audit.so')
shutil.copytree(a.host/'shaders', out/'shaders')
for name in ('KERNEL.json', 'kmsg.py', 'native_log_filter.py'): shutil.copy2(prior/name, out/name)
setup = (prior/'setup-probe.py').read_text().replace("'bin/mesh-corpus',", "'bin/mesh-task', 'bin/mesh-task-ubo',")
(out/'setup-probe.py').write_text(setup)
for name in ('run.py', 'check.py', 'MODEL.md'): shutil.copy2(HERE/name, out/name)
shutil.copy2(HERE/'MODEL.md', out/'README.md')
(out/'drirc-empty/README').write_text('Isolated empty configuration.\n')
shutil.copytree(HERE, out/'source/native-mesh-task-bridge', ignore=shutil.ignore_patterns('__pycache__'))
(out/'source/native-mesh-m2').mkdir()
shutil.copy2(HERE.parent/'native-mesh-m2/probe.c', out/'source/native-mesh-m2/probe.c')
shutil.copy2(HERE.parent/'native-mesh-corpus/record-sync.c', out/'source/record-sync.c')
shutil.copy2(HERE.parent/'native-stage-batching/vm-audit.c', out/'source/vm-audit.c')
for name in ('RESULT.json', 'INPUTS.json', 'MUTATIONS.json', 'CASES.json'):
    shutil.copy2(a.host/name, out/'evidence'/name)
shutil.copy2(a.records/'RESULT.json', out/'evidence/NATIVE-RECORDS.json')
shutil.copy2(a.parser/'RESULT.json', out/'evidence/KERNEL-PARSER.json')
for case in cases():
    d = out/'evidence'/case['name']; d.mkdir()
    for name in ('RUN.json', 'stdout.log', 'stream.bin'):
        shutil.copy2(a.records/case['name']/name, d/name)
manifest = json.loads((a.corpus/'SOURCE-MANIFEST.json').read_text())
sources = []
for record in manifest:
    repo, path = record['repository'], record['path']
    if repo not in ('SaschaWillems/Vulkan', 'KhronosGroup/Vulkan-Samples'): continue
    if 'meshshader' not in path and 'mesh_shader_culling' not in path and 'license' not in path.lower(): continue
    source = a.corpus/'sources'/repo.replace('/', '-')/path
    assert sha(source) == record['sha256']
    dest = out/'source/upstream'/repo.replace('/', '-')/path
    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    sources.append(record)
write('UPSTREAM-SOURCES.json', sources)
rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=HERE, text=True)
plan = json.loads((prior/'PLAN.json').read_text())
plan.update(cases=cases(), driver_revision=a.driver_revision, driver_sha256=sha(a.driver), source_revision=rev,
    scope='compute API task with native mesh children: four original triples and complete 16 KiB payload ownership',
    budget=dict(api_draws=8, submissions=8, native_render_works=8, native_object_dispatches=8, draws_per_process=1, seconds=240, retries=0),
    modes=[3], oracle=dict(color_bytes=131072, witness_bytes=524288, color_absolute_tolerance=2e-5,
        witness_comparison='byte-exact', format='R32G32B32A32_SFLOAT', blend='additive'), queue_prerequisites=[])
write('PLAN.json', plan)
kit = json.loads((prior/'KIT.json').read_text())
kit.update(name=out.name, budget_seconds=240)
kit['stage']['dir'] = '/root/kit-' + out.name
kit['setup_probe']['files'] = ['setup-probe.py', 'bin/mesh-task', 'bin/mesh-task-ubo', 'lib/vm-audit.so', 'lib/libvulkan_asahi.so']
kit['output_growth'] = dict(planned_processes=8, per_process_bytes=8*1024*1024, per_process_files=32,
    measured=[dict(process=c['name'], bytes=sum(p.stat().st_size for p in (a.host/c['name']).iterdir() if p.is_file()),
        files=len(list((a.host/c['name']).iterdir()))) for c in cases()])
write('KIT.json', kit)
write('PROVENANCE.json', dict(driver_revision=a.driver_revision, driver_sha256=sha(a.driver), source_revision=rev,
    original_triples=4, gpu_executions=0, native_api_task_execution=False, complete_application_run=False,
    index_opt_in=True, research_mesh_override=True))
write('SOURCE-MANIFEST.json', {str(p.relative_to(out)):sha(p) for p in sorted((out/'source').rglob('*')) if p.is_file()})
print(out)
