#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Author paired GLSL, exact source-space color/depth oracles and a manifest."""
import argparse, json, struct, subprocess
from fractions import Fraction as F
from pathlib import Path

def bits(x): return struct.unpack('<I', struct.pack('<f', float(x)))[0]
def number(x): return repr(float(x))
def positions(kind, halfz):
    w = [F(1), F(2), F(1,2)]
    if kind in ('CROSS', 'SMOOTH', 'RECIP_W'):
        xy = [(-F(1,4),-F(3,4)),(F(7,4),-F(3,4)),(-F(1,4),F(5,4))]
        z = [-F(1,2),F(1,2),F(3,2)]
    else:
        xy = [(-F(1,2),-F(1,2)),(F(1,2),-F(1,2)),(-F(1,2),F(1,2))]
        level = dict(CONTROL=F(1,2), NEAR=F(1,1024), FAR=F(1023,1024),
                     NEAR_OUT=-F(1,1024), FAR_OUT=F(1025,1024), ZERO=F(0), ONE=F(1))[kind]
        z = [level] * 3
    return [[x*ww,y*ww,(2*zz-1 if halfz else zz)*ww,ww] for (x,y),zz,ww in zip(xy,z,w)]

def oracle(p, halfz, smooth, reciprocal=False):
    # Project the original source triangle, then evaluate its screen-space
    # edges/depth at pixel centers. This does not read implementation output.
    v = [[(x/w+1)*32,(1-y/w)*32,(z/w+1)/2 if halfz else z/w,w]
         for x,y,z,w in p]
    def edge(a,b,x,y): return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
    attr = [F(0),F(1,2),F(1)]
    if edge(v[0],v[1],v[2][0],v[2][1]) < 0:
        v[1],v[2]=v[2],v[1];attr[1],attr[2]=attr[2],attr[1]
    area=edge(v[0],v[1],v[2][0],v[2][1]); assert area > 0
    color=bytearray(16384);depth=bytearray(struct.pack('<I',bits(1))*4096)
    weights=[];count=0;linear_differences=0;w_differences=0
    for yy in range(64):
        for xx in range(64):
            x,y=F(2*xx+1,2),F(2*yy+1,2)
            edges=[edge(v[i],v[(i+1)%3],x,y) for i in range(3)]
            hit=all(e>0 or (e==0 and (v[(i+1)%3][1]<v[i][1] or
                (v[(i+1)%3][1]==v[i][1] and v[(i+1)%3][0]>v[i][0]))) for i,e in enumerate(edges))
            bary=[edges[1]/area,edges[2]/area,edges[0]/area]
            z=sum(b*vert[2] for b,vert in zip(bary,v))
            if not hit or not 0<=z<=1: continue
            reciprocal_w=sum(b/vert[3] for b,vert in zip(bary,v))
            value=sum(b*a/vert[3] for b,a,vert in zip(bary,attr,v))/reciprocal_w
            linear=sum(b*a for b,a in zip(bary,attr))
            # Thresholded smooth values give an exact integer color oracle.
            # No selected pixel is on a threshold; record the minimum margin.
            thresholds=[F(3,16),F(7,16),F(11,16)]
            if reciprocal:
                thresholds=[F(5,8),F(7,8),F(9,8)]
                assert all(reciprocal_w != t for t in thresholds)
                rgb=[255*int(reciprocal_w<t) for t in thresholds]
            elif smooth:
                assert all(value != t for t in thresholds)
                rgb=[255*int(value<t) for t in thresholds]
                linear_differences += rgb != [255*int(linear<t) for t in thresholds]
            else: rgb=[255,0,0]
            off=4*(64*yy+xx);color[off:off+4]=bytes(rgb+[255]);depth[off:off+4]=struct.pack('<I',bits(z))
            count+=1;weights.append(str(value));w_differences+=reciprocal_w != 1
    if smooth: assert linear_differences > 100
    return bytes(color),bytes(depth),dict(covered=count,affine_color_mismatches=linear_differences,
        covered_nonunit_reciprocal_w=w_differences)

def shader(p, mesh, smooth):
    constants=', '.join('vec4('+', '.join(map(number,v))+')' for v in p)
    head='#version 460\n'
    if mesh: head+='#extension GL_EXT_mesh_shader : require\nlayout(local_size_x=1) in;\nlayout(triangles,max_vertices=3,max_primitives=1) out;\n'
    if smooth: head+='layout(location=0) out float value'+('[]' if mesh else '')+';\n'
    head+='const vec4 position[3]=vec4[3]('+constants+');\nvoid main() {\n'
    if mesh:
        head+='SetMeshOutputsEXT(3,1);\n'
        for i in range(3):
            # Separate component writes deliberately test the completed-position
            # conversion, including Z before W and preserved output readback.
            head+=f'gl_MeshVerticesEXT[{i}].gl_Position.xyz=position[{i}].xyz;\n'
            head+=f'gl_MeshVerticesEXT[{i}].gl_Position.w=position[{i}].w;\n'
            if smooth: head+=f'value[{i}]={i/2:.1f};\n'
        head+='gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);\n'
    else:
        head+='gl_Position=position[gl_VertexIndex];\n'
        if smooth: head+='value=float(gl_VertexIndex)*0.5;\n'
    return head+'}\n'

def generate(root):
    root.mkdir(parents=True,exist_ok=True);(root/'shaders').mkdir(exist_ok=True);(root/'expected').mkdir(exist_ok=True)
    for smooth in [False,True]:
        fragment='#version 460\nlayout(location=0) out vec4 color;\n'
        if smooth: fragment+='layout(location=0) in float value;\n'
        fragment+='void main(){color='+('vec4(value<0.1875?1:0,value<0.4375?1:0,value<0.6875?1:0,1)' if smooth else 'vec4(1,0,0,1)')+';}\n'
        path=root/'shaders'/('smooth.frag' if smooth else 'constant.frag');path.write_text(fragment)
        subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(path),'-o',str(path)+'.spv'],check=True,stdout=subprocess.DEVNULL)
    path=root/'shaders'/'reciprocal.frag'
    path.write_text('#version 460\nlayout(location=0) out vec4 color;\nvoid main(){float w=gl_FragCoord.w;color=vec4(w<0.625?1:0,w<0.875?1:0,w<1.125?1:0,1); }\n')
    subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(path),'-o',str(path)+'.spv'],check=True,stdout=subprocess.DEVNULL)
    rows=[]
    for kind in ['CONTROL','ZERO','NEAR','FAR','ONE','NEAR_OUT','FAR_OUT','CROSS','RECIP_W','SMOOTH']:
        for halfz in [False,True]:
            name=kind+('_HALFZ' if halfz else '_ZERO_ONE');p=positions(kind,halfz);smooth=kind=='SMOOTH'
            color,depth,stats=oracle(p,halfz,smooth,kind=='RECIP_W')
            (root/'expected'/(name+'.rgba')).write_bytes(color);(root/'expected'/(name+'.depth')).write_bytes(depth)
            for stage in ['vert','mesh']:
                path=root/'shaders'/(name+'.'+stage);path.write_text(shader(p,stage=='mesh',smooth))
                subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(path),'-o',str(path)+'.spv'],check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['spirv-val','--target-env','vulkan1.3',str(path)+'.spv'],check=True)
            rows.append(dict(name=name,halfz=halfz,pending_attributes=smooth,fragment='smooth' if smooth else 'reciprocal' if kind=='RECIP_W' else 'constant',
                positions=[[str(x) for x in v] for v in p],position_bits=[[bits(x) for x in v] for v in p],**stats))
    (root/'ARMS.json').write_text(json.dumps(dict(no_device_execution=True,arms=rows),indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();generate(a.root)
