#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Original CADSCENE shader pair: exact software pixels and native ARM recording."""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path
from check import cases,inputs,mutations,verify,MESH_SHA,FRAGMENT_SHA
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for name in ('out','modules','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
case=cases()[0]
original=[a.modules/'225636e04d3505d6.spv',a.modules/'4b476a849d94367f.spv']
assert [sha(f) for f in original]==[MESH_SHA,FRAGMENT_SHA]
for source,name in zip(original,[case['name']+'.mesh.spv',case['fragment']]):
    (shaders/name).write_bytes(source.read_bytes())
    subprocess.run(['spirv-val','--target-env','vulkan1.3',str(shaders/name)],check=True)
pins={str(f.resolve()):sha(f) for f in original}
(shaders/'inputs.bin').write_bytes(inputs())
write('INPUTS.json',dict(original_sources=pins,
    pipeline='f16d6602c9af8440',original_modules_unmodified=True,
    state_adaptations=['omit requiredSubgroupSize=32 (basic MS has no subgroup operations)',
        'explicit clear and transfer readback of isolated attachments',
        'fixed 128x64 viewport/scissor and deterministic buffers'],
    staged={f.name:sha(f) for f in shaders.iterdir()}))
write('MUTATIONS.json',mutations())
# Fixture values sit >=1/8 UNORM step from a rounding boundary. The
# light vector's transverse component is <2^-29 of Z; +Z normals and
# constant extras avoid interpolation seams. A 2^-12 intensity error
# budget is conservative relative to the resulting 1/(8*255) margin.
write('QUANTIZATION-MARGIN.json',dict(verdict='PASS',min_unorm_steps=.125,
    intensity_error_budget=2**-12,margin=.125/255,far_light_z=2**30,
    constant_normal=[0,0,1],constant_extra=[.125,.25,.375,.125],
    color_base='-.25 * .8 + .2 = 0; fused and unfused agree',
    exact_depth=[.25,1],edge_centers=0))
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
env.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
def run(directory,input_path):
    command=[str(a.host_probe.resolve()),str(shaders/(case['name']+'.mesh.spv')),str(shaders/case['fragment']),str(input_path),str(directory)]
    with (directory/'stdout.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
d=out/case['name'];d.mkdir();run(d,shaders/'inputs.bin');software=verify(d,case);write('SOFTWARE.json',software);assert software['verdict']=='PASS',software
controls=[]
for name in ('upper-vertex','last-primitive','normal','object-color'):
    md=out/('control-'+name);md.mkdir();data=bytearray(inputs());index=(13*63+7)%64
    if name=='upper-vertex':struct.pack_into('<f',data,20480+(11+index)*16,0)
    if name=='last-primitive':data[16384+232+83*3:16384+232+84*3]=bytes(3)
    if name=='normal':struct.pack_into('<3f',data,24576+(11+index)*32,1,0,0)
    if name=='object-color':struct.pack_into('<4f',data,8192+240,0,0,0,0)
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
    state_adaptations=json.loads((out/'INPUTS.json').read_text())['state_adaptations'],
    driver_sha256=records['driver_sha256'],scope='CADSCENE basic taskless original mesh/fragment, adapted subgroup state and isolated inputs; not a complete application'))
print('CADSCENE original mesh/fragment: exact software and native pipeline recording PASS')
