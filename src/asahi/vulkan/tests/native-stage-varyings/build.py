#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build authored stage-varying probes; never executes a GPU workload."""
import argparse, json, subprocess
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cc', default='cc')
p.add_argument('--sysroot', type=Path)
a = p.parse_args(); a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
s = a.out/'shaders'; s.mkdir(exist_ok=True)
commands=[]
def run(argv):
    commands.append(list(map(str,argv))); subprocess.run(commands[-1],check=True)
def shader(name, text):
    (s/name).write_text(text)
    run(['glslangValidator','-V','--target-env','vulkan1.2',s/name,'-o',s/(name+'.spv')])
shader('gs.vert', '''#version 450
layout(location=7) out vec3 values;
layout(location=11) flat out uint token;
void main() {
 int p=gl_VertexIndex/3, c=gl_VertexIndex%3;
 vec2 xy=vec2(c==1 ? 1:0,c==2 ? 1:0);
 float w=c==0 ? 1.0 : c==1 ? 2.0 : .5;
 vec2 pixel=vec2(8+8*p,8)+xy*vec2(8,48);
 gl_Position=vec4((pixel/vec2(128,64)*2-1)*w,.5*w,w);
 values=vec3(.125+float(p)/32+.25*xy.x,.25+.5*xy.y,.125+float(p)/64+.125*xy.x+.25*xy.y);
 token=uint(p*8+c+1);
}
''')
output='''layout(location=0,component=1) out vec2 smoothValue;
layout(location=3,component=2) flat out uint flatValue;
layout(location=5) noperspective out float linearValue;
'''
for n in (3,4):
    shader(f'gs{n}.geom',f'''#version 450
layout(triangles) in;
layout(triangle_strip,max_vertices={n}) out;
layout(location=7) in vec3 values[];
layout(location=11) flat in uint token[];
{output}
void emitCorner(int c) {{
 vec3 value;
 uint id;
 if(c<3) {{gl_Position=gl_in[c].gl_Position;value=values[c];id=token[c];}}
 else {{
  vec2 ndc=gl_in[1].gl_Position.xy/gl_in[1].gl_Position.w+gl_in[2].gl_Position.xy/gl_in[2].gl_Position.w-gl_in[0].gl_Position.xy/gl_in[0].gl_Position.w;
  gl_Position=vec4(ndc*1.5,.75,1.5);
  value=values[1]+values[2]-values[0];id=token[2]+1u;
 }}
 smoothValue=vec2(value.x,-1);smoothValue.y=value.y;
 flatValue=id;linearValue=value.z;
 EmitVertex();
}}
void main() {{emitCorner(0);emitCorner(1);emitCorner(2);{('emitCorner(3);' if n==4 else '')}EndPrimitive();}}
''')
shader('tes.vert','''#version 450
layout(location=7) out vec3 values;
void main() {
 vec2 xy=vec2(gl_VertexIndex==1 ? 1:0,gl_VertexIndex==2 ? 1:0);
 gl_Position=vec4(xy,0,1);
 values=vec3(.125+.25*xy.x,.25+.5*xy.y,.125+.125*xy.x+.25*xy.y);
}
''')
shader('vary.tesc','''#version 450
layout(vertices=3) out;
layout(location=7) in vec3 values[];
layout(location=7) out vec3 controlValues[];
layout(location=2) patch out vec2 bias;
void main() {
 gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;
 controlValues[gl_InvocationID]=values[gl_InvocationID];
 if(gl_InvocationID==0) {
  bias=vec2(.0625,.03125);
  gl_TessLevelOuter[0]=3;gl_TessLevelOuter[1]=3;gl_TessLevelOuter[2]=3;gl_TessLevelOuter[3]=3;
  gl_TessLevelInner[0]=3;gl_TessLevelInner[1]=3;
 }
}
''')
for name,domain,uv in [('tri','triangles','gl_TessCoord.yz'),('quad','quads','gl_TessCoord.xy')]:
    shader(name+'.tese',f'''#version 450
layout({domain},equal_spacing,cw) in;
layout(location=7) in vec3 controlValues[];
layout(location=2) patch in vec2 bias;
{output}
void main() {{
 vec2 uv={uv};
 vec3 v=controlValues[0]+uv.x*(controlValues[1]-controlValues[0])+uv.y*(controlValues[2]-controlValues[0]);
 vec2 pixel=vec2(8,8)+uv*vec2(104,48);
 gl_Position=vec4(pixel/vec2(128,64)*2-1,.5,1);
 smoothValue=vec2(v.x+bias.x,v.y+bias.y);linearValue=v.z;flatValue=7u;
}}
''')
shader('vary.frag','''#version 450
layout(location=0,component=1) in vec2 smoothValue;
layout(location=3,component=2) flat in uint flatValue;
layout(location=5) noperspective in float linearValue;
layout(location=0) out vec4 color;
void main(){color=vec4(smoothValue,linearValue,float(flatValue)/128);}
''')
argv=[a.cc,'-O2','-std=c11','-Wall','-Wextra',
      '-I'+str(Path(__file__).resolve().parents[5]/'include')]
if a.sysroot:
    lib=a.sysroot.resolve()/'usr/lib64'
    argv += ['--sysroot='+str(a.sysroot.resolve()),'-L'+str(lib),'-Wl,-rpath-link,'+str(lib)]
run(argv+[Path(__file__).with_name('probe.c'),'-lvulkan','-o',a.out/'probe'])
(a.out/'BUILD.json').write_text(json.dumps(commands,indent=2)+'\n')
