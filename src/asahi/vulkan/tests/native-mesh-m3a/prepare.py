#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare a four-interface kit using the passed M2a driver and shared audit runner."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from check import cases
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','m2-kit','host','parser','probe'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=a.m2_kit.resolve();host=json.loads((a.host/'RESULT.json').read_text());parser=json.loads((a.parser/'RESULT.json').read_text())
assert host['verdict']==parser['verdict']=='PASS' and host['native_pipeline_recordings']==len(parser['cases'])==4
assert host['driver_sha256']==sha(prior/'lib/libvulkan_asahi.so')
for folder in ('bin','lib','drirc-empty'):shutil.copytree(prior/folder,out/folder)
shutil.copy2(a.probe,out/'bin/mesh-m2')
shutil.copytree(a.host/'shaders',out/'shaders')
for name in ('KERNEL.json','setup-probe.py','kmsg.py','native_log_filter.py'):shutil.copy2(prior/name,out/name)
shutil.copy2(HERE.parent/'native-mesh-m2/run.py',out/'run.py')
for name in ('check.py','MODEL.md'):shutil.copy2(HERE/name,out/name)
shutil.copy2(HERE/'MODEL.md',out/'README.md')
(out/'source').mkdir();shutil.copytree(HERE,out/'source/native-mesh-m3a',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(prior/'source/native-stage-batching',out/'source/native-stage-batching')
shutil.copy2(HERE.parent/'native-mesh-m2/run.py',out/'source/common-run.py')
for name in ('host-record.py','probe.c','kernel-replay.py'):
    shutil.copy2(HERE.parent/'native-mesh-m2'/name,out/'source'/name)
(out/'evidence').mkdir()
for origin,name in ((a.host/'RESULT.json','HOST.json'),(a.host/'INPUTS.json','INPUTS.json'),(a.host/'MUTATIONS.json','MUTATIONS.json'),(a.host/'native-records/RESULT.json','NATIVE-RECORDS.json'),(a.parser/'RESULT.json','KERNEL-PARSER.json')):shutil.copy2(origin,out/'evidence'/name)
plan=json.loads((prior/'PLAN.json').read_text());plan.update(cases=cases(),scope='three original Sascha mesh/color components and one mixed P1 interface',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),production_candidate=False,budget=dict(api_draws=4,submissions=4,native_render_works=4,native_object_dispatches=4,draws_per_process=1,seconds=140,retries=0))
assert not subprocess.check_output(['git','status','--porcelain'],cwd=HERE,text=True)
plan.update(modes=[3],oracle=dict(pixels=8192,components=32768,witness_bytes=524288,format='R32G32B32A32_SFLOAT',absolute_tolerance=2e-5,blend='ONE/ONE ADD'))
write('PLAN.json',plan)
kit=json.loads((prior/'KIT.json').read_text());kit.update(name=out.name,budget_seconds=140);kit['stage']['dir']='/root/kit-'+out.name
measurements=[]
for case in cases():
    files=[p for p in (a.host/case['name']).rglob('*') if p.is_file()]
    measurements.append(dict(case=case['name'],bytes=sum(p.stat().st_size for p in files),files=len(files)))
kit['output_growth']=dict(planned_processes=4,per_process_bytes=8*1024*1024,per_process_files=32,measured=measurements,reason='Four software processes; native audit overhead reserved within enforced 8 MiB / 32-file per-process limits')
write('KIT.json',kit)
write('PROVENANCE.json',dict(driver_revision=plan['driver_revision'],source_revision=plan['source_revision'],gpu_executions=0,original_pairs=3,authored_pairs=1,host=str(a.host.resolve()),parser=str(a.parser.resolve()),driver_sha256=sha(out/'lib/libvulkan_asahi.so'),mesh_input_ledger='native-coverage-msg180-m2a-20261024-v5/COVERAGE.json'))
write('SOURCE-MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in sorted((out/'source').rglob('*')) if p.is_file()})
print(out)
