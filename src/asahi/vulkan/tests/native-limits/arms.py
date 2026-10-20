#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Ordinary GLSL probes for physical T32 and serialized logical workgroups."""
import argparse, importlib.util, json, struct, subprocess
from pathlib import Path

def helper():
    path=Path(__file__).resolve().parent.parent/'native-attributes/arms.py'
    spec=importlib.util.spec_from_file_location('attributes',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);return a

def initial_q():
    q=bytearray(b'\xa5'*16384)
    for i in range(128):struct.pack_into('<I',q,i*64,0)
    return q

def expected_q(shape, enabled):
    q=initial_q();total=shape[0]*shape[1]*shape[2]
    if not enabled:return q
    for i in range(total):
        sg=i//32;lane=i%32
        neighbor=(i^(32 if total>32 else 1)) if total%32==0 else (i+32)%total
        active=min(32,total-32*sg)
        row=[1,100+i,i%shape[0],i//shape[0]%shape[1],i//(shape[0]*shape[1]),
             *shape,32,sg,lane,(total+31)//32,(1<<active)-1,
             active*(64*sg+active-1)//2,32*sg,100+neighbor]
        struct.pack_into('<16I',q,i*64,*row)
    return q

def generate(root, cases=None):
    a=helper();root.mkdir(parents=True,exist_ok=True)
    for sub in ['shaders','expected']:(root/sub).mkdir(exist_ok=True)
    p=a.positions('CONTROL',False);color,depth,stats=a.oracle(p,False,[])
    rows=[]
    if cases is None:cases=[
        ('T32_CONTROL',(32,1,1),False,False),
        ('T32_IDS',(32,1,1),True,False),
        ('T64_LOGICAL',(64,1,1),True,False),
        ('T64_LOGICAL_3D',(4,4,4),True,False),
        ('T128_LOGICAL_HOST',(128,1,1),True,True)]
    for name,shape,records,host_only in cases:
        s=a.shader(p,[],'mesh').replace('layout(local_size_x=1)',f'layout(local_size_x={shape[0]},local_size_y={shape[1]},local_size_z={shape[2]})')
        s=s.replace('void main(){','''layout(set=0,binding=0,std430) coherent buffer Results { uint q[]; };
void main(){''')
        s=s.replace('gl_MeshVerticesEXT[0]', 'if(gl_LocalInvocationIndex==0u){\ngl_MeshVerticesEXT[0]',1)
        s=s[:-2]+'}\n}\n'
        if records:
            s=s.replace('#extension GL_EXT_mesh_shader : require','#extension GL_EXT_mesh_shader : require\n#extension GL_KHR_shader_subgroup_basic : require\n#extension GL_KHR_shader_subgroup_ballot : require\n#extension GL_KHR_shader_subgroup_arithmetic : require')
            body='''
uint i=gl_LocalInvocationIndex, o=16u*i;
atomicAdd(q[o],1u);
q[o+1u]=100u+i;
q[o+2u]=gl_LocalInvocationID.x;q[o+3u]=gl_LocalInvocationID.y;q[o+4u]=gl_LocalInvocationID.z;
q[o+5u]=gl_WorkGroupSize.x;q[o+6u]=gl_WorkGroupSize.y;q[o+7u]=gl_WorkGroupSize.z;
q[o+8u]=gl_SubgroupSize;q[o+9u]=gl_SubgroupID;q[o+10u]=gl_SubgroupInvocationID;q[o+11u]=gl_NumSubgroups;
q[o+12u]=subgroupBallot(true).x;q[o+13u]=subgroupAdd(i);q[o+14u]=subgroupBroadcastFirst(i);
memoryBarrierBuffer();barrier();
q[o+15u]=q[16u*(i^NEIGHBORu)+1u];
'''.replace('NEIGHBOR',str(32 if shape[0]*shape[1]*shape[2]>32 else 1))
            s=s.replace('SetMeshOutputsEXT(3,1);','SetMeshOutputsEXT(3,1);\n'+body)
            total=shape[0]*shape[1]*shape[2]
            if total%32:s=s.replace('i^'+str(32 if total>32 else 1)+'u',f'(i+32u)%{total}u')
        for stage,source in [('mesh',s),('vert',a.shader(p,[],'vert')),('frag',a.shader(p,[],'frag'))]:
            path=root/'shaders'/(name+'.'+stage);path.write_text(source)
            subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(path),'-o',str(path)+'.spv'],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['spirv-val','--target-env','vulkan1.3',str(path)+'.spv'],check=True)
        for ext,data in [('rgba',color),('depth',depth),('q',expected_q(shape,records))]:
            (root/'expected'/(name+'.'+ext)).write_bytes(data)
        rows.append(dict(name=name,logical_size=shape,physical_threads=32,records=records,host_only=host_only,
                         fragment=name,halfz=False,pending_attributes=False,**stats))
    (root/'expected/initial.q').write_bytes(initial_q())
    (root/'ARMS.json').write_text(json.dumps(dict(no_device_execution=True,research_only=True,arms=rows),indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();generate(args.root)
