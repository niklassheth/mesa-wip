#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Software oracle checks only. Never substitutes for native device evidence."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check import cases,verify,mutation_tests
p=argparse.ArgumentParser(description=__doc__)
for name in ('probe','lavapipe','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir();src=Path(__file__).resolve().parent
commands=[]
for maximum in (15,16,17):commands.append(['glslangValidator','--target-env','vulkan1.3',f'-DMAX_PRIMITIVES={maximum}',str(src/'output.mesh'),'-o',str(shaders/f'P{maximum}.mesh.spv')])
commands.append(['glslangValidator','--target-env','vulkan1.3',str(src/'count.frag'),'-o',str(shaders/'count.frag.spv')])
for cmd in commands:subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
(out/'BUILD.json').write_text(json.dumps(commands,indent=2)+'\n');(out/'MUTATIONS.json').write_text(json.dumps(mutation_tests(),indent=2)+'\n')
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
rows=[]
for case in cases():
 d=out/case['name'];d.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_'))}
 env.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M1_SOFTWARE_REFERENCE='1')
 argv=[str(a.probe.resolve()),str(shaders/f'P{case["maximum"]}.mesh.spv'),str(shaders/'count.frag.spv'),str(d),str(case['vertices']),str(case['primitives']),str(case['maximum']),str(case['cookie'])]
 with (d/'stdout.log').open('w') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=30)
 log=(d/'stdout.log').read_text();assert r.returncode==0,(case,r.returncode,log);assert 'llvmpipe' in log and log.count('SUBMIT_OK\n')==1
 result=verify(d,case);(d/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');assert result['verdict']=='PASS',(case,result)
 rows.append(dict(case=case,argv=argv,**result));print(case['name'],'software oracle PASS',flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',software_only=True,native_evidence=False,draws=6,results=rows,lavapipe_sha256=hashlib.sha256(a.lavapipe.read_bytes()).hexdigest()),indent=2)+'\n')
