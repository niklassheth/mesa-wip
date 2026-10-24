#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build authored Vulkan probes; generating this kit does not run a device."""
import argparse, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cc', default='cc')
p.add_argument('--shaders-only', action='store_true', help='Retain an already-built probe executable')
p.add_argument('--sysroot', type=Path)
p.add_argument('--host-build-command', type=Path, help='Compiler argv with Mesa includes for read-only merge observer')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
commands=[]
def run(cmd):
 commands.append([str(x) for x in cmd]);subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
def shader(name,source):
 path=a.output/name;path.write_text(source)
 run(['glslangValidator','-V','--target-env','vulkan1.2',str(path),'-o',str(path)+'.spv'])
for name in ['input.vert','blue.frag','white.frag']:
 shader(name,(here.parent/'native-gs'/name).read_text())
for size in range(1,33):
 shader(f'P{size}.tesc',f'''#version 450
layout(vertices={size}) out;
layout(set=0,binding=0,std140) uniform Params {{float inner; float outer; float z;}} params;
void main() {{
 gl_out[gl_InvocationID].gl_Position=gl_in[0].gl_Position;
 if (gl_InvocationID==0) {{
  gl_TessLevelInner[0]=params.inner; gl_TessLevelInner[1]=params.inner;
  gl_TessLevelOuter[0]=params.outer; gl_TessLevelOuter[1]=params.outer;
  gl_TessLevelOuter[2]=params.outer; gl_TessLevelOuter[3]=params.outer;
 }}
}}
''')
for domain in ['triangles','quads']:
 for order in ['cw','ccw']:
  for spacing in ['equal_spacing','fractional_odd_spacing','fractional_even_spacing']:
   for fallback in [False,True]:
    name=f'{domain}-{order}-{spacing}'+('-fallback' if fallback else '')+'.tese'
    index='uint(abs(params.z)) % gl_PatchVerticesIn' if fallback else '0'
    shader(name,f'''#version 450
layout({domain},{order},{spacing}) in;
layout(set=0,binding=0,std140) uniform Params {{float inner; float outer; float z;}} params;
void main() {{
 gl_Position=vec4(2*gl_TessCoord.xy-1,params.z,1);
 gl_Position.x+=gl_in[{index}].gl_Position.x+1;
}}
''')
for size in range(1,33):
 shader(f'P{size}-last.tese',f'''#version 450
layout(triangles,cw,equal_spacing) in;
layout(set=0,binding=0,std140) uniform Params {{float inner; float outer; float z;}} params;
void main() {{
 gl_Position=vec4(2*gl_TessCoord.xy-1,params.z,1);
 gl_Position.x+=gl_in[{size-1}].gl_Position.x+1;
}}
''')
if a.shaders_only:
 assert (a.output/'transitions').is_file(), 'Missing preserved executable'
elif a.host_build_command:
 cmd=json.loads(a.host_build_command.read_text())
 cmd=[str(here/'transitions.c') if x.endswith('/native-gs/transitions.c') else x for x in cmd]
 cmd[-1]=str(a.output/'transitions');run(cmd)
else:
 cmd=[a.cc,'-O2','-D_POSIX_C_SOURCE=200809L']
 if a.sysroot:
  lib=a.sysroot.resolve()/'usr/lib64'
  cmd += ['--sysroot='+str(a.sysroot.resolve()),'-L'+str(lib),'-Wl,-rpath-link,'+str(lib)]
 run(cmd+[str(here/'transitions.c'),'-lvulkan','-o',str(a.output/'transitions')])
(a.output/'BUILD.json').write_text(json.dumps(commands,indent=2)+'\n')
print(f'Built {len(commands)-(not a.shaders_only)} SPIR-V modules; Vulkan probe '+('retained' if a.shaders_only else 'built')+'; no device execution')
