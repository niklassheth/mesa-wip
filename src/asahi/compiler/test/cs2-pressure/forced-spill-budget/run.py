#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only Vulkan compilation regression; requires the explicit host shim."""
from pathlib import Path
import argparse,os,subprocess,json
p=argparse.ArgumentParser();p.add_argument('--icd',type=Path,required=True);p.add_argument('--host-intercept',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.uname().machine=='x86_64', 'This runner is for the x86 host, not the mini'
assert a.icd.is_file() and a.host_intercept.is_file()
a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve();here=Path(__file__).resolve().parent
subprocess.run(['cc','-O2',str(here/'repro.c'),'-lvulkan','-o',str(out/'repro')],check=True)
subprocess.run(['glslangValidator','-V',str(here/'test.vert'),'-o',str(out/'test.vert.spv')],check=True)
env=dict(os.environ,LD_PRELOAD=str(a.host_intercept.resolve()),VK_DRIVER_FILES=str(a.icd.resolve()),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_EAGER_VARIANTS='1',AGX_MESA_DEBUG='spill',MESA_SHADER_CACHE_DISABLE='true',AGX_APPLE9_STATS='1')
for key in ['ASAHI_MESA_DEBUG','AGX_SPILL_COALESCE','AGX_SPILL_NO_LOOP_HANDLING','AGX_APPLE9_RA_WEBS']:env.pop(key,None)
results={}
for precision in ['highp','mediump']:
 for loop in ['for','while','do_while']:
  name=f'{loop}-{precision}';source=out/(name+'.frag')
  step='value=value.yzwx+vec4(1.0); i+=step;'
  code='for(;i<bound;) {'+step+'}' if loop=='for' else 'while(i<bound) {'+step+'}' if loop=='while' else 'do {'+step+'} while(i<bound);'
  source.write_text('''#version 310 es
layout(location=0) out mediump vec4 color;
layout(location=0) in mediump vec4 coords;
layout(std140,set=0,binding=0) uniform A {highp float step;};
layout(std140,set=0,binding=1) uniform B {highp float bound;};
void main(){mediump vec4 value=coords; PRECISION float i=0.05; LOOP color=vec4((value-vec4(3)).rgb,1);}
'''.replace('PRECISION',precision).replace('LOOP',code))
  spv=source.with_suffix('.spv');subprocess.run(['glslangValidator','-V',str(source),'-o',str(spv)],check=True)
  with (out/(name+'.log')).open('w') as f:r=subprocess.run([str(out/'repro'),str(out/'test.vert.spv'),str(spv)],env=env,stdout=f,stderr=subprocess.STDOUT)
  results[name]=r.returncode;print(name,r.returncode,flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(any(results.values()))
