#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Paired flat-broadcast probes; no native primitive-record ABI assumptions."""
import argparse, importlib.util, json, struct, subprocess
from fractions import Fraction as F
from pathlib import Path

def generate(root):
    root.mkdir(parents=True,exist_ok=True)
    for d in ['shaders','expected']:(root/d).mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('attributes',Path(__file__).resolve().parents[1]/'native-attributes/arms.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=module.positions('CONTROL',False);rows=[]
    for name in ['CONTROL','USER','PARTIAL_T3','PID','USER_PID','MIXED','CULL_FALSE','CULL_TRUE']:
        user=name in ['USER','PARTIAL_T3','USER_PID','MIXED'];pid=name in ['PID','USER_PID']
        cull=name.startswith('CULL_');culled=name=='CULL_TRUE';threads=3 if name=='PARTIAL_T3' else 1
        linear=[dict(name='linearValue',slot=3,component=0,mode='noperspective',values=[[0],[1],[0]],type='float',n=1)] if name=='MIXED' else []
        rgba,depth,stats=module.oracle(p,False,linear)
        rgba=bytearray(rgba)
        for i in range(0,len(rgba),4):
            if rgba[i+3]:rgba[i:i+4]=bytes([rgba[i] if linear else 0,255,0,255])
        if culled:
            rgba=bytearray(len(rgba));depth=struct.pack('<f',1)*4096;stats['covered']=0
        (root/'expected'/(name+'.rgba')).write_bytes(rgba)
        (root/'expected'/(name+'.depth')).write_bytes(depth)
        for stage in ['vert','mesh']:
            mesh=stage=='mesh';prefix='#version 460\n'+('#extension GL_EXT_mesh_shader : require\n' if mesh else '')
            s=prefix
            if mesh:s+=f'layout(local_size_x={threads}) in;\nlayout(triangles,max_vertices=3,max_primitives=1) out;\n'
            if user:s+=f'layout(location=0,component=1) {"flat perprimitiveEXT" if mesh else "flat"} out uvec2 dataValue'+('[]' if mesh else '')+';\n'
            if linear:s+='layout(location=3) noperspective out float linearValue'+('[]' if mesh else '')+';\n'
            if pid and not mesh:s+='layout(location=8) flat out int authoredID;\n'
            if mesh and (pid or cull):
                s+='perprimitiveEXT out gl_MeshPerPrimitiveEXT { '+('int gl_PrimitiveID;' if pid else 'bool gl_CullPrimitiveEXT;')+' } gl_MeshPrimitivesEXT[];\n'
            s+='const vec4 positions[3]=vec4[3]('+','.join('vec4('+','.join(str(float(x)) for x in q)+')' for q in p)+');\nvoid main(){\n'
            if mesh:
                s+='SetMeshOutputsEXT(3,1);\nif(gl_LocalInvocationIndex==0u){\n'
                for v in range(3):
                    s+=f'gl_MeshVerticesEXT[{v}].gl_Position=positions[{v}];\n'
                    if linear:s+=f'linearValue[{v}]={float(v==1)};\n'
                s+='gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);\n}\n'
                if user:
                    s+=f'if(gl_LocalInvocationIndex=={threads-1}u) dataValue[0]=uvec2(0x7fc01234u,0x7fc00000u);\n'
                    if threads==3:s+='barrier();\nif(gl_LocalInvocationIndex==0u) dataValue[0].y=0x7fc01234u^0x1234u;\n'
                if pid:s+='gl_MeshPrimitivesEXT[0].gl_PrimitiveID=37;\n'
                if cull:s+='gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT='+('true' if culled else 'false')+';\n'
            else:
                s+='gl_Position=positions['+('0' if culled else 'gl_VertexIndex')+'];\n'
                if user:s+='dataValue=uvec2(0x7fc01234u,0x7fc00000u)+uint(gl_VertexIndex);\n'
                if pid:s+='authoredID=37+gl_VertexIndex;\n'
                if linear:s+='linearValue=gl_VertexIndex==1?1.0:0.0;\n'
            s+='}\n'
            path=root/'shaders'/(name+'.'+stage);path.write_text(s)
            frag=prefix
            if user:frag+=f'layout(location=0,component=1) {"flat perprimitiveEXT" if mesh else "flat"} in uvec2 dataValue;\n'
            if pid and not mesh:frag+='layout(location=8) flat in int authoredID;\n'
            if linear:frag+='layout(location=3) noperspective in float linearValue;\n'
            conditions=[]
            if user:conditions+=['all(equal(dataValue,uvec2(0x7fc01234u,0x7fc00000u)))']
            if pid:conditions += [('gl_PrimitiveID' if mesh else 'authoredID')+'==37']
            condition=' && '.join(conditions) or 'true'
            red='float(linearValue<0.1835784912109375)' if linear else '0.0'
            frag+='layout(location=0) out vec4 color;\nvoid main(){color=('+condition+')?vec4('+red+',1,0,1):vec4(1,0,1,1);}\n'
            fpath=root/'shaders'/(name+'.'+stage+'.frag');fpath.write_text(frag)
            for shader in [path,fpath]:
                subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(shader),'-o',str(shader)+'.spv'],check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['spirv-val','--target-env','vulkan1.3',str(shader)+'.spv'],check=True)
        flat=2*user+pid;components=flat+bool(linear)
        rows.append(dict(user_components=components,attributes=[dict(mode='flat',n=flat),dict(mode='noperspective',n=int(bool(linear)))],
                         channel_planes=[[c,32+16*c] for c in range(4+components)],name=name,kind=name,halfz=False,pending_attributes=False,fragment=name+'.mesh',
                         fragments={stage:name+'.'+stage for stage in ['vert','mesh']},
                         primitive_route='flat vertex broadcast',**stats))
    (root/'ARMS.json').write_text(json.dumps(dict(no_device_execution=True,native_attributes=True,primitive_lowering=True,arms=rows),indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();generate(a.root)
