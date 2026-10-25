#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy pinned original pairs, execute software controls, record native packets."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from check import cases,verify,mutations
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for key in ('out','corpus','coverage','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();(out/'shaders').mkdir()
coverage=json.loads(a.coverage.read_text());selected=[];rows=[];pins={}
for case in cases():
    folder='sources/KhronosGroup-Vulkan-Samples/shaders/mesh_shading/'+case['language']
    entry=next(e for e in coverage['entries'] if e['path']==folder+'/ms.mesh.spv')
    assert entry['stage']=='mesh'
    for stage,basename in [('mesh','ms.mesh.spv'),('frag','ps.frag.spv')]:
        source=a.corpus/folder/basename;data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
        if stage=='mesh':assert sha==entry['sha256']
        (out/'shaders'/(case['name']+'.'+stage+'.spv')).write_bytes(data)
        pins[str(source.resolve())]=sha
    case['source']=entry;selected.append(case)
(out/'CASES.json').write_text(json.dumps(selected,indent=2)+'\n')
(out/'INPUTS.json').write_text(json.dumps(pins,indent=2)+'\n')
(out/'MUTATIONS.json').write_text(json.dumps(mutations(),indent=2)+'\n')
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
for case in selected:
    d=out/case['name'];d.mkdir()
    e={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
    e.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
    cmd=[str(a.host_probe.resolve()),str(out/'shaders'/(case['name']+'.mesh.spv')),str(out/'shaders'/case['fragment']),str(d),'3','1','1','0']
    with (d/'stdout.log').open('w') as f:subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
    result=verify(d,case);assert result['verdict']=='PASS',(case,result)
    rows.append(dict(case=case,readback=result));print(case['name'],'software PASS',flush=True)
cmd=[sys.executable,str(HERE.parent/'native-mesh-m2/host-record.py'),'--cases',str(out/'CASES.json'),'--shaders',str(out/'shaders'),'--out',str(out/'native-records')]
for key in ('driver','shim','probe','sysroot'):cmd+=['--'+key,str(getattr(a,key).resolve())]
cmd+=['--completion-shim',str(a.probe.resolve().parent/'libhost-record-sync.so')]
subprocess.run(cmd,check=True)
records=json.loads((out/'native-records/RESULT.json').read_text());assert records['verdict']=='PASS' and len(records['cases'])==3
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,software=rows,native_pipeline_recordings=3,original_mesh_and_fragment=True,driver_sha256=records['driver_sha256']),indent=2)+'\n')
