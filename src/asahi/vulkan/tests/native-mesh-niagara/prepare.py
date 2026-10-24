#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sealable one-entry corpus kit; device/queue work belongs to the manager."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from check import cases, MESH_SHA

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','prior-kit','driver','probe','host','parser','coverage'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--driver-revision',required=True)
p.add_argument('--mini-kernel-kit',type=Path,help='Pinned research-mini-native-blocks kernel kit')
a=p.parse_args();out=a.out.resolve();out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
host=json.loads((a.host/'RESULT.json').read_text())
parser=json.loads((a.parser/'RESULT.json').read_text())
assert host['verdict']==parser['verdict']=='PASS'
assert host['native_pipeline_recordings']==len(parser['cases'])==1
assert host['driver_sha256']==sha(a.driver) and host['mesh_sha256']==MESH_SHA
assert parser['qualified_expected']=='reject'
prior=a.prior_kit.resolve()
kernel_source=a.mini_kernel_kit.resolve() if a.mini_kernel_kit else prior
kernel=json.loads((kernel_source/'KERNEL.json').read_text())
if a.mini_kernel_kit:assert kernel['name']=='research-mini-native-blocks'
assert parser['kernel_revision']==kernel['source_commit'][:len(parser['kernel_revision'])]
for folder in ('bin','lib','source','evidence','drirc-empty'):(out/folder).mkdir()
shutil.copy2(a.probe,out/'bin/mesh-niagara')
shutil.copy2(a.driver,out/'lib/libvulkan_asahi.so')
shutil.copy2(prior/'lib/vm-audit.so',out/'lib/vm-audit.so')
shutil.copytree(a.host/'shaders',out/'shaders')
for name in ('run.py','check.py','MODEL.md'):shutil.copy2(HERE/name,out/name)
shutil.copy2(HERE/'MODEL.md',out/'README.md')
for name in ('kmsg.py','native_log_filter.py'):shutil.copy2(prior/name,out/name)
shutil.copy2(kernel_source/'KERNEL.json',out/'KERNEL.json')
(out/'setup-probe.py').write_text((prior/'setup-probe.py').read_text().replace('mesh-m2','mesh-niagara').replace('mesh-nvidia','mesh-niagara'))
(out/'drirc-empty/README').write_text('Empty isolated drirc directory.\n')
shutil.copytree(HERE,out/'source/native-mesh-niagara',ignore=shutil.ignore_patterns('__pycache__'))
(out/'source/corpus').mkdir()
for filename,expected in json.loads((a.host/'INPUTS.json').read_text())['original_sources'].items():
    original=Path(filename)
    assert sha(original)==expected
    if original.suffix!='.spv':shutil.copy2(original,out/'source/corpus'/original.name)
shutil.copy2(HERE.parent/'native-stage-batching/vm-audit.c',out/'source/vm-audit.c')
shutil.copy2(HERE.parent/'native-mesh-m2/kernel-replay.py',out/'source/kernel-replay.py')
for name in ('RESULT.json','INPUTS.json','MUTATIONS.json','SOFTWARE.json','SOFTWARE-CONTROLS.json'):
    shutil.copy2(a.host/name,out/'evidence'/name)
shutil.copy2(a.host/'native-records/RESULT.json',out/'evidence/NATIVE-RECORDS.json')
shutil.copy2(a.parser/'RESULT.json',out/'evidence/KERNEL-PARSER.json')
shutil.copy2(a.coverage,out/'evidence/COVERAGE.json')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=HERE,text=True)
plan=json.loads((prior/'PLAN.json').read_text())
plan.update(cases=cases(),driver_revision=a.driver_revision,driver_sha256=sha(a.driver),
    source_revision=revision,scope='unchanged Niagara basic taskless L64 V64/P96 mesh and original fragment pipeline',
    modes=[3],budget=dict(api_draws=1,submissions=1,native_render_works=1,
        native_object_dispatches=1,draws_per_process=1,seconds=60,retries=0),
    oracle=dict(pixels_per_target=8192,components=65536,witness_bytes=524288,
        formats=['R8G8B8A8_UNORM','A2B10G10R10_UNORM_PACK32'],comparison='byte-exact',blend='disabled'),
    queue_prerequisites=['Air 39b7b logical64/V64/P96 publication shape PASS',
                         'Air 39b7c2 original NVIDIA 11-plane pipeline PASS'])
plan['kernel_profile']='mini-native-blocks' if a.mini_kernel_kit else 'air-native-batch'
plan['device']='mini' if a.mini_kernel_kit else 'air'
plan['kernel_version']=kernel['uts_version']
plan['kernel_release']=kernel['release']
if a.mini_kernel_kit:
    plan['queue_prerequisites']=['Air Niagara original shader pipeline PASS']
write('PLAN.json',plan)
kit=json.loads((prior/'KIT.json').read_text());kit.update(name=out.name,budget_seconds=60)
kit['stage']['dir']='/root/kit-'+out.name
if a.mini_kernel_kit:
    mini=json.loads((kernel_source/'KIT.json').read_text())
    for key in ('device','boot','requires','inputs'):kit[key]=mini[key]
    kit['boot']['fresh']=True
    kit['boot']['kernel']['extra_args_append']='asahi_m4.native_mesh=0'
    # The runner checks mode 0 after this fresh boot. An older snapshot may
    # have mode 4 latched, so require presence here rather than stale value.

kit['setup_probe']['files']=[f.replace('mesh-m2','mesh-niagara').replace('mesh-nvidia','mesh-niagara').replace('mesh-nvidia','mesh-niagara') for f in kit['setup_probe']['files']]
measurements=[]
for directory in [a.host/cases()[0]['name'],*sorted(a.host.glob('control-*'))]:
    files=[f for f in directory.iterdir() if f.is_file()]
    measurements.append(dict(process=directory.name,bytes=sum(f.stat().st_size for f in files),files=len(files)))
assert len(measurements)==5
kit['output_growth']=dict(planned_processes=1,per_process_bytes=8*1024*1024,
    per_process_files=32,measured=measurements,reason='Five software processes; enforced native audit/output reserve')
write('KIT.json',kit)
write('PROVENANCE.json',dict(driver_revision=a.driver_revision,source_revision=revision,
    driver_sha256=sha(a.driver),gpu_executions=0,original_mesh_sha256=MESH_SHA,
    original_fragment_sha256=host['fragment_sha256'],original_pipeline=True,
    complete_application_run=False,host=str(a.host.resolve()),parser=str(a.parser.resolve()),
    conditional_queue_prerequisites=plan['queue_prerequisites']))
write('SOURCE-MANIFEST.json',{str(f.relative_to(out)):sha(f) for f in sorted((out/'source').rglob('*')) if f.is_file()})
print(out)
