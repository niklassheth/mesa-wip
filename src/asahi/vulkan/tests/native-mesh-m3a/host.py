#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy pinned original pairs, execute software controls, record native packets."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from check import cases,verify,mutations,initial_q
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for key in ('out','corpus','coverage','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();(out/'shaders').mkdir()
coverage=json.loads(a.coverage.read_text());selected=[];rows=[];pins={}
(out/'shaders/initial-Q.bin').write_bytes(initial_q())
for case in cases()[:3]:
    folder='sources/SaschaWillems-Vulkan/shaders/'+case['language']+'/meshshader'
    entry=next(e for e in coverage['entries'] if e['path']==folder+'/meshshader.mesh.spv')
    for stage,basename in [('mesh','meshshader.mesh.spv'),('frag','meshshader.frag.spv')]:
        source=a.corpus/folder/basename;data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
        if stage=='mesh':assert sha==entry['sha256']
        (out/'shaders'/(case['name']+'.'+stage+'.spv')).write_bytes(data)
        pins[str(source.resolve())]=sha
    case['source']=entry;selected.append(case)
selected.append(cases()[-1])
for stage in ('mesh','frag'):
    src=HERE/('interface.'+stage)
    pins[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
    subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(src),'-o',str(out/'shaders'/('P1-mixed.'+stage+'.spv'))],check=True)
    subprocess.run(['spirv-val','--target-env','vulkan1.3',str(out/'shaders'/('P1-mixed.'+stage+'.spv'))],check=True)
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
subprocess.run(cmd,check=True)
records=json.loads((out/'native-records/RESULT.json').read_text());assert records['verdict']=='PASS' and len(records['cases'])==4
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,software=rows,native_pipeline_recordings=4,original_mesh_and_fragment_pairs=3,authored_interface_pairs=1,driver_sha256=records['driver_sha256']),indent=2)+'\n')
