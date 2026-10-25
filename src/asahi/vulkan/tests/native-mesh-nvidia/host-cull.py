#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record both original NVIDIA culling mesh variants through the ARM driver.

No GPU executes. The corpus SPIR-V is hash checked and never changed; the matching
fragment/task modules use the same pinned public sources and build defines.
"""
import argparse,hashlib,importlib.util,json,os,re,struct,subprocess
from pathlib import Path
from check import inputs
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capture',HERE.parent/'check-native-mesh-submit.py')
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
p=argparse.ArgumentParser(description=__doc__)
for name in ('corpus','coverage','driver','probe','shim','sync','sysroot','out'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();rows=[];pins={}
manifest=json.loads((a.corpus/'BUILD-MANIFEST.json').read_text())
coverage=json.loads(a.coverage.read_text())['entries']
folder=a.corpus.resolve()/'sources/nvpro-samples-gl_vk_meshlet_cadscene'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
for task in (0,1):
    name='NVIDIA-cull-task'+str(task);d=out/name;d.mkdir();(d/'trace').mkdir()
    entry=next(e for e in coverage if e['path'].endswith('--cull-task'+str(task)+'.spv'))
    mesh=a.corpus.resolve()/entry['path'];assert sha(mesh)==entry['sha256']
    pins[str(mesh)]=sha(mesh)
    build=next(e for e in manifest if e.get('sha256')==entry['sha256'])
    defines=[x for x in build['argv'] if x.startswith('-D')]
    commands=[]
    for stage in ('frag','task') if task else ('frag',):
        source=folder/('drawmeshlet_ext.'+stage+'.glsl');pins[str(source)]=sha(source)
        cmd=['glslangValidator','--target-env','vulkan1.3','-S',stage,*defines,'-o',str(d/(stage+'.spv')),str(source)]
        with (d/(stage+'-build.log')).open('w') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['spirv-val','--target-env','vulkan1.3',str(d/(stage+'.spv'))],check=True)
        commands.append(cmd)
    (d/'inputs.bin').write_bytes(inputs())
    write(d/'icd.json',dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.driver.resolve()),api_version='1.4.0')))
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','HK_','VK_','MESA_','AGX_','ASAHI_'))}
    settings=dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'),LD_PRELOAD=str(a.sync.resolve())+':'+str(a.shim.resolve()),
        ASAHI_HOST_TRACE_DIR=str(d/'trace'),ASAHI_HOST_RECORD_SYNCHRONOUS='1',VK_DRIVER_FILES=str(d/'icd.json'),
        HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_INDEX='1',HK_APPLE9_NATIVE_MESH='1',
        HK_APPLE9_NATIVE_TASK_BRIDGE='1',HK_APPLE9_RESEARCH_NATIVE_BATCH='1',HK_APPLE9_MESH_ROUTE_LOG='1',
        MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false',VK_LOADER_LAYERS_DISABLE='~implicit~')
    argv=['qemu-aarch64-static','-L',str(a.sysroot.resolve())]
    for k,v in settings.items():argv+=['-E',k+'='+v]
    argv += [str(a.probe.resolve()),str(mesh),str(d/'frag.spv'),str(d/'inputs.bin'),str(d)]
    if task:argv.append(str(d/'task.spv'))
    write(d/'RUN.json',dict(argv=argv,environment=settings,build_commands=commands,gpu_execution=False,source=entry))
    with (d/'stdout.log').open('w') as f:
        result=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    log=(d/'stdout.log').read_text()
    routes=re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)',log)
    assert result.returncode==0 and log.count('SUBMIT_OK\n')==1,(name,result.returncode,log[-4000:])
    assert routes==[('1','32','64','126')],(name,routes)
    assert (d/'pixels.bin').read_bytes()==bytes([0xa5])*32768
    assert (d/'Q.bin').read_bytes()==inputs()
    renders=[];submitted=[]
    for path in sorted((d/'trace').glob('*.commands.bin')):
        data=path.read_bytes();pos=0;index=path.name.split('.')[0]
        while pos<len(data):
            kind,size,vdm,cdm=struct.unpack_from('<4H',data,pos);pos+=8
            assert size and pos+size<=len(data)
            submitted.append(dict(submit=index,kind=kind,size=size,vdm=vdm,cdm=cdm))
            if kind==0:
                assert size==240
                packet=capture.read_gpu(d/'trace',index,struct.unpack_from('<Q',data,pos+8)[0],160)
                state=list(struct.unpack_from('<14I',packet))
                assert state[:6]==[0x40000d2c,0xc0b,0x10,0,0,0x1000000],state
                assert state[9:]==[1,125|(7<<16),0x400820,0,177],state
                assert struct.unpack_from('<I',packet,156)[0]==0xc0000000
                (d/'stream.bin').write_bytes(packet)
                renders.append(dict(state=state,sha256=hashlib.sha256(packet).hexdigest(),vdm=vdm,cdm=cdm))
            pos+=size
    assert len(renders)==1,(name,renders)
    if task:assert renders[0]['cdm']!=65535,(name,submitted)
    rows.append(dict(name=name,mesh_sha256=sha(mesh),renders=renders,commands=submitted,task_bridge=bool(task)))
    print(name,'native record PASS; no GPU execution',flush=True)
for name in ('common.h','config.h','nvmeshlet_utils.glsl','draw_shading.glsl','drawmeshlet_ext_cull.mesh.glsl','LICENSE'):
    pins[str(folder/name)]=sha(folder/name)
write(out/'RESULT.json',dict(verdict='PASS',gpu_execution=False,cases=rows,driver_sha256=sha(a.driver),
    probe_sha256=sha(a.probe),source_pins=pins,scope='Original culling mesh variants plus matching public task/fragment modules; compilation and packet recording only'))
