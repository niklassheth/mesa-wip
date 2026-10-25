#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record normal ARM task/mesh pipelines. No GPU execution or native credit."""
import argparse,hashlib,importlib.util,json,os,re,struct,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capture',HERE.parent/'check-native-mesh-submit.py');capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','host','driver','shim','probes','sysroot'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();rows=[]
cases=json.loads((a.host/'CASES.json').read_text());shaders=a.host.resolve()/'shaders'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for case in cases:
    d=out/case['name'];d.mkdir();(d/'trace').mkdir();name=case['name']
    (d/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.driver.resolve()),api_version='1.4.0'))))
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','HK_','VK_','MESA_','AGX_','ASAHI_','M1_','M2_','TASK_PROBE_'))}
    settings=dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'),LD_PRELOAD=str(a.probes.resolve()/'libhost-record-sync.so')+':'+str(a.shim.resolve()),
        ASAHI_HOST_TRACE_DIR=str(d/'trace'),ASAHI_HOST_RECORD_SYNCHRONOUS='1',VK_DRIVER_FILES=str(d/'icd.json'),HK_EXPERIMENTAL_APPLE9='1',
        HK_APPLE9_MESH_ROUTE_LOG='1',HK_APPLE9_RESEARCH_NATIVE_BATCH='1',MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false',
        VK_LOADER_LAYERS_DISABLE='~implicit~',AGX_APPLE9_NIR='1',AGX_APPLE9_STATS='1',
        TASK_PROBE_TASK_SHADER=str(shaders/(name+'.task.spv')),TASK_PROBE_INPUT=str(shaders/(name+'.initial-Q.bin')),TASK_PROBE_GRID=','.join(map(str,case['parents'])))
    settings.update(case['env'])
    argv=['qemu-aarch64-static','-L',str(a.sysroot.resolve())]
    for k,v in settings.items():
        if ',' in v:env[k]=v
        else:argv+=['-E',k+'='+v]
    argv += [str(a.probes.resolve()/case['probe']),str(shaders/(name+'.mesh.spv')),str(shaders/(name+'.frag.spv')),str(d),'3','1','1',str(case['cookie'])]
    (d/'RUN.json').write_text(json.dumps(dict(argv=argv,environment=settings,gpu_execution=False),indent=2)+'\n')
    with (d/'stdout.log').open('w') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
    log=(d/'stdout.log').read_text();assert r.returncode==0 and log.count('SUBMIT_OK\n')==1,(d,r.returncode,log[-4000:])
    routes=re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)',log)
    assert routes==[('1',str(case['physical']),str(case['max_vertices']),str(case['max_primitives']))],(d,routes)
    assert (d/'Q.bin').read_bytes()==(shaders/(name+'.initial-Q.bin')).read_bytes()
    assert (d/'pixels.bin').read_bytes()==bytes([0xa5])*(8192*16)
    renders=[];commands=[]
    for path in sorted((d/'trace').glob('*.commands.bin')):
        data=path.read_bytes();pos=0;index=path.name.split('.')[0]
        while pos<len(data):
            kind,size,vdm,cdm=struct.unpack_from('<4H',data,pos);pos+=8;assert size and pos+size<=len(data)
            commands.append(dict(submit=index,kind=kind,size=size,vdm=vdm,cdm=cdm))
            if kind==0:
                assert size==240;address=struct.unpack_from('<Q',data,pos+8)[0]
                packet=capture.read_gpu(d/'trace',index,address,160);state=struct.unpack_from('<14I',packet)
                assert state[0]==0x40000d2c and state[1]==((case['components']+1)<<8)|case['components']
                assert state[11]==(case['physical']|0x400800) and state[12]==0 and state[13]==case['output_rows'],(name,state)
                assert struct.unpack_from('<I',packet,156)[0]==0xc0000000
                (d/'stream.bin').write_bytes(packet);renders.append(dict(state=list(state),sha256=hashlib.sha256(packet).hexdigest(),vdm=vdm,cdm=cdm))
            pos+=size
    assert len(renders)==1,(name,renders)
    # A native task bridge must retain a compute predecessor for tiling.
    assert renders[0]['cdm']!=65535,(name,commands)
    rows.append(dict(case=case,renders=renders,commands=commands))
    print(name,'native record PASS; GPU_execution=0',flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,cases=rows,driver_sha256=sha(a.driver),shim_sha256=sha(a.shim),selection='explicit research packet model; device routing is independently audited'),indent=2)+'\n')
