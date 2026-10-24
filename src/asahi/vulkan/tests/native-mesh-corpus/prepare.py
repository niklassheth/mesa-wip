#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal three original full-RGBA pairs with an explicit native index opt-in."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from check import cases
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','prior-kit','driver','probe','host','parser'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--driver-revision',required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
host=json.loads((a.host/'RESULT.json').read_text());parser=json.loads((a.parser/'RESULT.json').read_text());prior=a.prior_kit.resolve()
assert host['verdict']==parser['verdict']=='PASS' and host['native_pipeline_recordings']==len(parser['cases'])==3
assert host['driver_sha256']==sha(a.driver) and parser['qualified_expected']=='accept'
for n in ('bin','lib','source','evidence','drirc-empty'):(out/n).mkdir()
shutil.copy2(a.probe,out/'bin/mesh-corpus');shutil.copy2(a.driver,out/'lib/libvulkan_asahi.so');shutil.copy2(prior/'lib/vm-audit.so',out/'lib/vm-audit.so')
shutil.copytree(a.host/'shaders',out/'shaders')
for n in ('KERNEL.json','kmsg.py','native_log_filter.py'):shutil.copy2(prior/n,out/n)
(out/'setup-probe.py').write_text((prior/'setup-probe.py').read_text().replace('mesh-cadscene','mesh-corpus'))
for n in ('run.py','check.py','MODEL.md'):shutil.copy2(HERE/n,out/n)
shutil.copy2(HERE/'MODEL.md',out/'README.md');(out/'drirc-empty/README').write_text('Isolated empty configuration.\n')
shutil.copytree(HERE,out/'source/native-mesh-corpus',ignore=shutil.ignore_patterns('__pycache__'))
(out/'source/native-mesh-m2').mkdir();shutil.copy2(HERE.parent/'native-mesh-m2/probe.c',out/'source/native-mesh-m2/probe.c')
shutil.copy2(HERE.parent/'native-stage-batching/vm-audit.c',out/'source/vm-audit.c')
for n in ('RESULT.json','INPUTS.json','MUTATIONS.json'):shutil.copy2(a.host/n,out/'evidence'/n)
shutil.copy2(a.host/'native-records/RESULT.json',out/'evidence/NATIVE-RECORDS.json');shutil.copy2(a.parser/'RESULT.json',out/'evidence/KERNEL-PARSER.json')
rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=HERE,text=True)
plan=json.loads((prior/'PLAN.json').read_text());plan.update(cases=cases(),driver_revision=a.driver_revision,driver_sha256=sha(a.driver),source_revision=rev,scope='three original Khronos taskless mesh/fragment pairs, complete RGBA32F readback',budget=dict(api_draws=3,submissions=3,native_render_works=3,native_object_dispatches=3,draws_per_process=1,seconds=100,retries=0),modes=[4],oracle=dict(color_bytes=131072,witness_bytes=524288,comparison='byte-exact',format='R32G32B32A32_SFLOAT',blend='additive'),queue_prerequisites=[])
write('PLAN.json',plan)
kit=json.loads((prior/'KIT.json').read_text());kit.update(name=out.name,budget_seconds=100);kit['stage']['dir']='/root/kit-'+out.name
kit['setup_probe']['files']=[n.replace('mesh-cadscene','mesh-corpus') for n in kit['setup_probe']['files']]
kit['output_growth']=dict(planned_processes=3,per_process_bytes=8*1024*1024,per_process_files=32,measured=[dict(process=c['name'],bytes=sum(p.stat().st_size for p in (a.host/c['name']).iterdir() if p.is_file()),files=len(list((a.host/c['name']).iterdir()))) for c in cases()])
write('KIT.json',kit);write('PROVENANCE.json',dict(driver_revision=a.driver_revision,driver_sha256=sha(a.driver),source_revision=rev,original_pairs=True,gpu_executions=0,complete_application_run=False,prior_red_only_component_evidence=True))
write('SOURCE-MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in sorted((out/'source').rglob('*')) if p.is_file()})
print(out)
