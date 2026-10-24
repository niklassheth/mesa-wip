#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and record ordinary Vulkan with the real ARM ICD and no-op DRM.

No GPU executes; expected readback stays at its CPU initialization pattern.
"""
import argparse,hashlib,importlib.util,json,os,re,struct,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from check import cases
spec=importlib.util.spec_from_file_location('capture',HERE.parent/'check-native-mesh-submit.py');capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
p=argparse.ArgumentParser(description=__doc__)
for name in ('driver','shim','probe','shaders','sysroot','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();rows=[]
for case in cases():
 d=out/case['name'];d.mkdir();(d/'trace').mkdir();(d/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.driver.resolve()),api_version='1.4.0'))))
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','HK_','VK_','MESA_','AGX_','ASAHI_','M1_'))}
 settings=dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'),LD_PRELOAD=str(a.shim.resolve()),ASAHI_HOST_TRACE_DIR=str(d/'trace'),VK_DRIVER_FILES=str(d/'icd.json'),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_MESH='1',HK_APPLE9_NATIVE_MESH_PACKED_SHAPES='T16V16P15,T16V16P16,T16V16P17',HK_APPLE9_MESH_ROUTE_LOG='1',HK_APPLE9_RESEARCH_NATIVE_BATCH='1',MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false',VK_LOADER_LAYERS_DISABLE='~implicit~',AGX_APPLE9_ASM='1',AGX_APPLE9_NIR='1')
 argv=['qemu-aarch64-static','-L',str(a.sysroot.resolve())]
 for k,v in settings.items():
  if ',' in v:env[k]=v
  else:argv+=['-E',k+'='+v]
 argv += [str(a.probe.resolve()),str(a.shaders.resolve()/f'P{case["maximum"]}.mesh.spv'),str(a.shaders.resolve()/'count.frag.spv'),str(d),str(case['vertices']),str(case['primitives']),str(case['maximum']),str(case['cookie'])]
 (d/'RUN.json').write_text(json.dumps(dict(argv=argv,environment=settings,gpu_execution=False),indent=2)+'\n')
 with (d/'stdout.log').open('w') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
 log=(d/'stdout.log').read_text();assert r.returncode==0 and log.count('SUBMIT_OK\n')==1,(d,r.returncode,log[-2500:])
 assert re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)',log)==[('1','16','16',str(case['maximum']))],(d,log[-2000:])
 assert (d/'Q.bin').read_bytes()==bytes([0xcc])*(512*1024) and (d/'pixels.bin').read_bytes()==bytes([0xa5])*32768
 renders=[]
 for path in sorted((d/'trace').glob('*.commands.bin')):
  data=path.read_bytes();pos=0;index=path.name.split('.')[0]
  while pos<len(data):
   kind,size,vdm,cdm=struct.unpack_from('<4H',data,pos);pos+=8;assert size and pos+size<=len(data)
   if kind==0:
    assert size==240;address=struct.unpack_from('<Q',data,pos+8)[0]
    # The fixture emits a single native draw; read just its bounded packet.
    packet=capture.read_gpu(d/'trace',index,address,160)
    state=struct.unpack_from('<14I',packet);maximum=case['maximum']
    assert state[0]==0x40000d2c and state[1:5]==(0x504,0x10,0,0)
    assert state[5]==state[7]==0x1000000 and state[9:]==(1,(maximum-1)|(((maximum+15)//16-1)<<16),0x400810,0,17)
    assert state[6]%2==state[8]%2==0
    # PPP blocks may differ with immutable bindings; retain exact words for
    # the paired kernel parser rather than implementing another packet parser.
    assert struct.unpack_from('<I',packet,156)[0]==0xc0000000
    (d/'stream.bin').write_bytes(packet)
    renders.append(dict(state=list(state),stream_sha256=hashlib.sha256(packet).hexdigest(),vdm_dependency=vdm,compute_dependency=cdm))
   pos+=size
 assert len(renders)==1,(d,len(renders));rows.append(dict(case=case,renders=renders))
 print(case['name'],'native record PASS; GPU_execution=0',flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,cases=rows,driver_sha256=hashlib.sha256(a.driver.read_bytes()).hexdigest(),shim_sha256=hashlib.sha256(a.shim.read_bytes()).hexdigest()),indent=2)+'\n')
