#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare unchanged corpus inputs, independent software execution and ARM packets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from check import cases, inputs, mutations, verify, MESH_SHA

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','corpus','coverage','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
case=cases()[0]
entry=next(e for e in json.loads(a.coverage.read_text())['entries'] if e['sha256']==MESH_SHA)
source=(a.corpus/entry['path']).resolve();assert sha(source)==MESH_SHA
(shaders/(case['name']+'.mesh.spv')).write_bytes(source.read_bytes())
build=next(e for e in json.loads((a.corpus/'BUILD-MANIFEST.json').read_text()) if e.get('sha256')==MESH_SHA)
folder=(a.corpus/'sources/nvpro-samples-gl_vk_meshlet_cadscene').resolve()
defines=[arg for arg in build['argv'] if arg.startswith('-D')]
command=['glslangValidator','--target-env','vulkan1.3','-S','frag',*defines,
         '-o',str(shaders/case['fragment']),str(folder/'drawmeshlet_ext.frag.glsl')]
with (out/'shader-build.log').open('w') as f:
    subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    for module in shaders.glob('*.spv'):
        subprocess.run(['spirv-val','--target-env','vulkan1.3',str(module)],stdout=f,stderr=subprocess.STDOUT,check=True)
pins={str(source):sha(source)}
for name in ('drawmeshlet_ext_basic.mesh.glsl','drawmeshlet_ext.frag.glsl',
             'config.h','common.h','nvmeshlet_utils.glsl','draw_shading.glsl','LICENSE'):
    pins[str(folder/name)]=sha(folder/name)
(shaders/'inputs.bin').write_bytes(inputs())
write('INPUTS.json',dict(original_sources=pins,mesh_entry=entry,fragment_build=command,
                       staged={f.name:sha(f) for f in shaders.iterdir()}))
write('MUTATIONS.json',mutations())
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',
    ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
d=out/case['name'];d.mkdir()
env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
env.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',
           LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
command=[str(a.host_probe.resolve()),str(shaders/(case['name']+'.mesh.spv')),
         str(shaders/case['fragment']),str(shaders/'inputs.bin'),str(d)]
with (d/'stdout.log').open('w') as f:
    subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
software=verify(d,case);write('SOFTWARE.json',software)
assert software['verdict']=='PASS',software
print('original NVIDIA mesh/fragment pair: software pixel-exact PASS',flush=True)
# Keep the original modules unchanged while perturbing application inputs.
# Require pixel differences independently of the inevitable buffer difference.
# These host-only controls prove that each selected interface affects pixels.
controls=[]
for name in ('upper-vertex','last-primitive','normal','flat-meshlet-id'):
    md=out/('control-'+name);md.mkdir();data=bytearray(inputs())
    index=5+(13*63+7)%64
    if name=='upper-vertex':struct.pack_into('<f',data,20480+index*16,0)
    if name=='last-primitive':data[16384+224+125*3:16384+224+126*3]=bytes(3)
    if name=='normal':struct.pack_into('<4f',data,24576+index*16,0,0,1,0)
    if name=='flat-meshlet-id':
        struct.pack_into('<I',data,3072+16,6)
        data[12288+8*16:12288+9*16]=data[12288+9*16:12288+10*16]
    (md/'inputs.bin').write_bytes(data)
    control_command=[*command[:3],str(md/'inputs.bin'),str(md)]
    with (md/'stdout.log').open('w') as f:
        subprocess.run(control_command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
    assert (md/'Q.bin').read_bytes()==data
    changed=sum(a!=b for a,b in zip((md/'pixels.bin').read_bytes(),(d/'pixels.bin').read_bytes()))
    assert changed>0,name
    controls.append(dict(name=name,pixel_bytes_changed=changed,input_buffer_unchanged=True))
write('SOFTWARE-CONTROLS.json',dict(verdict='PASS',controls=controls))
command=[sys.executable,str(HERE/'host-record.py'),'--out',str(out/'native-records'),'--shaders',str(shaders)]
for name in ('driver','shim','probe','sysroot'):command+=['--'+name,str(getattr(a,name).resolve())]
subprocess.run(command,check=True)
records=json.loads((out/'native-records/RESULT.json').read_text())
assert records['verdict']=='PASS' and len(records['cases'])==1
write('RESULT.json',dict(verdict='PASS',gpu_execution=False,software=software,
    native_pipeline_recordings=1,mesh_sha256=MESH_SHA,fragment_sha256=sha(shaders/case['fragment']),
    original_mesh_and_fragment=True,driver_sha256=records['driver_sha256']))
