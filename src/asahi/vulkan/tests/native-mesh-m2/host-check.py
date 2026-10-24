#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Software-only controls; never counted as native device evidence."""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from check import cases,verify,mutations
p=argparse.ArgumentParser(description=__doc__)
for name in ('probe','lavapipe','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.out.resolve();out.mkdir();shaders=out/'shaders';shaders.mkdir();commands=[];rows=[]
for case in cases():
 defines=[f'-DSIZE_{axis}={value}' for axis,value in zip('XYZ',case['logical'])]+[f'-DUSE_SHARED={int(case["shared"])}']
 commands.append(['glslangValidator','--target-env','vulkan1.3',*defines,str(HERE/'workgroup.mesh'),'-o',str(shaders/(case['name']+'.mesh.spv'))])
commands.append(['glslangValidator','--target-env','vulkan1.3',str(HERE/'count.frag'),'-o',str(shaders/'count.frag.spv')])
for cmd in commands:subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
(out/'BUILD.json').write_text(json.dumps(commands,indent=2)+'\n')
(out/'MUTATIONS.json').write_text(json.dumps(mutations(),indent=2)+'\n')
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
for case in cases():
 d=out/case['name'];d.mkdir();e={k:v for k,v in os.environ.items() if not k.startswith(('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
 e.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
 argv=[str(a.probe.resolve()),str(shaders/(case['name']+'.mesh.spv')),str(shaders/'count.frag.spv'),str(d),'3','1','1',str(case['cookie'])]
 with (d/'stdout.log').open('w') as f:r=subprocess.run(argv,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=30)
 log=(d/'stdout.log').read_text();assert r.returncode==0 and 'llvmpipe' in log and log.count('SUBMIT_OK\n')==1,(case,r.returncode,log)
 # CPU physical-device property, not a shader-produced value, sets the
 # software subgroup oracle. Native device checks always require 32.
 subgroup=int(re.search(r'^subgroup_size=(\d+)$',log,re.M)[1])
 result=verify(d,case,subgroup);(d/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');assert result['verdict']=='PASS',(case,result)
 rows.append(dict(case=case,argv=argv,**result));print(case['name'],'software oracle PASS',flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',software_only=True,native_evidence=False,draws=len(rows),results=rows,lavapipe_sha256=hashlib.sha256(a.lavapipe.read_bytes()).hexdigest()),indent=2)+'\n')
