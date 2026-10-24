#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Original Niagara shader pair: exact software pixels and native ARM recording."""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path
from check import cases,inputs,mutations,verify,MESH_SHA
from quantization import verify_margin
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','corpus','coverage','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
case=cases()[0];entry=next(e for e in json.loads(a.coverage.read_text())['entries'] if e['sha256']==MESH_SHA)
source=(a.corpus/entry['path']).resolve();assert sha(source)==MESH_SHA
(shaders/(case['name']+'.mesh.spv')).write_bytes(source.read_bytes())
folder=(a.corpus/'sources/zeux-niagara').resolve();src=folder/'src/shaders'
command=['glslangValidator','--target-env','vulkan1.3','-S','frag','-o',str(shaders/case['fragment']),str(src/'mesh.frag.glsl')]
with (out/'shader-build.log').open('w') as f:
    subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    for module in shaders.glob('*.spv'):
        subprocess.run(['spirv-val','--target-env','vulkan1.3',str(module)],stdout=f,stderr=subprocess.STDOUT,check=True)
pins={str(source):sha(source)}
for path in [src/'meshlet.mesh.glsl',src/'mesh.frag.glsl',src/'mesh.h',src/'math.h',folder/'src/config.h',folder/'LICENSE.md']:
    pins[str(path)]=sha(path)
(shaders/'inputs.bin').write_bytes(inputs())
write('INPUTS.json',dict(original_sources=pins,mesh_entry=entry,fragment_build=command,
    specialization='upstream defaults: TASK=false, POST=0; original modules unmodified',
    staged={f.name:sha(f) for f in shaders.iterdir()}))
write('MUTATIONS.json',mutations())
write('QUANTIZATION-MARGIN.json',verify_margin())
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
env.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
def run(directory,input_path):
    command=[str(a.host_probe.resolve()),str(shaders/(case['name']+'.mesh.spv')),str(shaders/case['fragment']),str(input_path),str(directory)]
    with (directory/'stdout.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
d=out/case['name'];d.mkdir();run(d,shaders/'inputs.bin');software=verify(d,case);write('SOFTWARE.json',software);assert software['verdict']=='PASS',software
controls=[]
for name in ('upper-vertex','last-primitive','normal','draw-material'):
    md=out/('control-'+name);md.mkdir();data=bytearray(inputs());index=(13*63+7)%64
    if name=='upper-vertex':struct.pack_into('<e',data,20480+(5+index)*16,0)
    if name=='last-primitive':data[16384+156+95*3:16384+156+96*3]=bytes(3)
    if name=='normal':struct.pack_into('<I',data,20480+(5+index)*16+8,1022<<20)
    if name=='draw-material':struct.pack_into('<I',data,4096+3*20,6)
    (md/'inputs.bin').write_bytes(data);run(md,md/'inputs.bin')
    assert (md/'Q.bin').read_bytes()==data
    changed=sum(a!=b for a,b in zip((md/'pixels.bin').read_bytes(),(d/'pixels.bin').read_bytes()));assert changed>0,name
    controls.append(dict(name=name,pixel_bytes_changed=changed,input_buffer_unchanged=True))
write('SOFTWARE-CONTROLS.json',dict(verdict='PASS',controls=controls))
command=[sys.executable,str(HERE/'host-record.py'),'--out',str(out/'native-records'),'--shaders',str(shaders)]
for name in ('driver','shim','probe','sysroot'):command+=['--'+name,str(getattr(a,name).resolve())]
subprocess.run(command,check=True);records=json.loads((out/'native-records/RESULT.json').read_text())
assert records['verdict']=='PASS' and len(records['cases'])==1
write('RESULT.json',dict(verdict='PASS',gpu_execution=False,software=software,native_pipeline_recordings=1,
    mesh_sha256=MESH_SHA,fragment_sha256=sha(shaders/case['fragment']),original_mesh_and_fragment=True,
    driver_sha256=records['driver_sha256'],scope='Niagara original taskless mesh and original fragment component pipeline, not a complete application'))
print('Niagara original mesh/fragment: exact software and native pipeline recording PASS')
