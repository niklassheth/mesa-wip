#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Original task triples, independent software oracles, then ARM packet capture."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from check import cases,initial_q,verify,mutations
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','corpus','coverage','host-probes','lavapipe'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir()
coverage=json.loads(a.coverage.read_text());manifest=json.loads((a.corpus/'SOURCE-MANIFEST.json').read_text())
pins={};builds=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selected=cases()
for case in selected:
    name=case['name']
    if case['kind'] in ('sascha','khronos'):
        if case['kind']=='sascha':
            repo='SaschaWillems/Vulkan';folder='shaders/'+case['language']+'/meshshader';base='meshshader'
        else:
            repo='KhronosGroup/Vulkan-Samples';folder='shaders/mesh_shader_culling';base='mesh_shader_culling'
        for stage in ('task','mesh','frag'):
            path=folder+'/'+base+'.'+stage+'.spv'
            record=next(r for r in manifest if r['repository']==repo and r['path']==path)
            source=a.corpus/'sources'/repo.replace('/','-')/path
            assert sha(source)==record['sha256']
            if stage=='mesh':
                entry=next(r for r in coverage['entries'] if r['path']=='sources/'+repo.replace('/','-')+'/'+path)
                assert sha(source)==entry['sha256']
                case['source']=entry
            (shaders/(name+'.'+stage+'.spv')).write_bytes(source.read_bytes());pins[str(source.resolve())]=sha(source)
    else:
        for stage in ('task','mesh','frag'):
            source=HERE/('payload.'+stage);pins[str(source)]=sha(source)
            command=['glslangValidator','-V','--target-env','vulkan1.3']
            if stage=='task':
                command += ['-DCHILD_'+axis+'='+str(value) for axis,value in zip('XYZ',case['children'])]
                command += ['-DSKIP_MIDDLE='+str(int(case['skip_middle']))]
            command += [str(source),'-o',str(shaders/(name+'.'+stage+'.spv'))]
            subprocess.run(command,check=True);builds.append(command)
    for stage in ('task','mesh','frag'):
        subprocess.run(['spirv-val','--target-env','vulkan1.3',str(shaders/(name+'.'+stage+'.spv'))],check=True)
    (shaders/(name+'.initial-Q.bin')).write_bytes(initial_q(case))
(out/'CASES.json').write_text(json.dumps(selected,indent=2)+'\n')
(out/'INPUTS.json').write_text(json.dumps(dict(sources=pins,build_commands=builds,shaders={p.name:sha(p) for p in shaders.iterdir()}),indent=2)+'\n')
(out/'MUTATIONS.json').write_text(json.dumps(mutations(),indent=2)+'\n')
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
rows=[]
for case in selected:
    d=out/case['name'];d.mkdir();name=case['name']
    env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_','TASK_PROBE_'))}
    env.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1',
        TASK_PROBE_TASK_SHADER=str(shaders/(name+'.task.spv')),TASK_PROBE_INPUT=str(shaders/(name+'.initial-Q.bin')),TASK_PROBE_GRID=','.join(map(str,case['parents'])))
    command=[str(a.host_probes.resolve()/case['probe']),str(shaders/(name+'.mesh.spv')),str(shaders/(name+'.frag.spv')),str(d),'3','1','1',str(case['cookie'])]
    (d/'RUN.json').write_text(json.dumps(dict(argv=command,environment=env),indent=2)+'\n')
    with (d/'stdout.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=40)
    result=verify(d,case);(d/'READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['verdict']=='PASS',(name,result)
    rows.append(dict(case=name,readback=result));print(name,'software PASS',flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,software=rows,original_triples=4,authored_payload_bytes=16384),indent=2)+'\n')
