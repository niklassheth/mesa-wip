#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regression controls for SPIR-V block, primitive and narrow IO metadata."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gate',type=Path,required=True);a=p.parse_args()
prefix='''#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=1) in;
layout(triangles,max_vertices=3,max_primitives=1) out;
'''
body='''void main() {
 SetMeshOutputsEXT(3,1);
 gl_MeshVerticesEXT[0].gl_Position=vec4(0,0,0,1);
 gl_MeshVerticesEXT[1].gl_Position=vec4(1,0,0,1);
 gl_MeshVerticesEXT[2].gl_Position=vec4(0,1,0,1);
 gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);
 %s
}
'''
cases=[('position','','',False,False,False),
 ('direct','layout(location=0) out vec4 color[];','color[0]=vec4(1);',True,False,False),
 ('block','layout(location=0) out Output {vec4 color;} data[];','data[0].color=vec4(1);',True,False,False),
 ('primitive','layout(location=0) perprimitiveEXT out vec4 color[];','color[0]=vec4(1);',True,True,False),
 ('narrow','layout(location=0) out f16vec2 color[];','color[0]=f16vec2(1);',True,False,True)]
env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','AGX_','ASAHI_'))}
rows=[]
with tempfile.TemporaryDirectory() as tmp:
 for name,declaration,store,extra,primitive,narrow in cases:
  source=Path(tmp)/(name+'.mesh');binary=source.with_suffix('.spv')
  source.write_text(prefix+declaration+'\n'+body%store)
  subprocess.run(['glslangValidator','--target-env','vulkan1.3',str(source),'-o',str(binary)],check=True,stdout=subprocess.DEVNULL)
  result=json.loads(subprocess.check_output([str(a.gate.resolve()),'mesh','main',str(binary)],env=env,text=True))
  assert result['analysis_version']==2 and result['outputs_extra']==extra and result['per_primitive_extra']==primitive,(name,result)
  assert result['outputs_16bit']==narrow,(name,result)
  assert result['gate']==(not extra),(name,result)
  rows.append(dict(case=name,metadata=result))
print(json.dumps(dict(verdict='PASS',cases=rows,native_evidence=False),indent=2))
