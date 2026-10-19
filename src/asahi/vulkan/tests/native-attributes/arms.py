#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Paired ordinary GLSL and independent exact interpolation/color/depth oracles."""
import argparse, json, struct, subprocess
from fractions import Fraction as F
from pathlib import Path

def bits(x): return struct.unpack('<I', struct.pack('<f', float(x)))[0]
def scalar(x, typ): return str(int(x))+'u' if typ=='uint' else repr(float(x))
def ty(a): return ('uvec' if a['type']=='uint' else 'vec')+str(a['n']) if a['n']>1 else a['type']
def attributes(kind):
    def a(name,slot,comp,mode,values,typ='float'):
        return dict(name=name,slot=slot,component=comp,mode=mode,values=values,type=typ,n=len(values[0]))
    rgb=[[1,0,0],[0,1,0],[0,0,1]]
    if kind=='CONTROL':return []
    if kind.startswith('SMOOTH'):return [a('rgb',0,0,'smooth',rgb)]
    if kind=='FLAT':return [a('rgb',0,0,'flat',rgb)]
    if kind=='NOPERSPECTIVE':return [a('rgb',0,0,'noperspective',rgb)]
    assert kind.startswith('MULTIPLE')
    return [a('affine',0,2,'noperspective',[[0,F(1,2)],[1,0],[F(1,2),1]]),
            a('tag',1,1,'flat',[[3],[7],[11]],'uint'),
            a('color',3,0,'smooth',[[0,1,F(1,2),F(1,4)],[1,0,F(1,4),F(3,4)],[F(1,2),F(1,4),1,0]]),
            a('extra',5,1,'smooth',[[0],[1],[F(1,2)]])]

def positions(kind, halfz):
    w=[F(1),F(2),F(1,2)]
    xy=[(-F(1,2),-F(1,2)),(F(1,2),-F(1,2)),(-F(1,2),F(1,2))]
    z=[F(1,2)]*3
    if kind.endswith('_CROSS'):
        xy=[(-F(1,4),-F(3,4)),(F(7,4),-F(3,4)),(-F(1,4),F(5,4))]
        z=[-F(1,2),F(1,2),F(3,2)]
    return [[x*ww,y*ww,(2*zz-1 if halfz else zz)*ww,ww] for (x,y),zz,ww in zip(xy,z,w)]

def oracle(p, halfz, attrs):
    v=[[(x/w+1)*32,(1-y/w)*32,(z/w+1)/2 if halfz else z/w,w,i] for i,(x,y,z,w) in enumerate(p)]
    def edge(a,b,x,y):return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
    if edge(v[0],v[1],v[2][0],v[2][1])<0:v[1],v[2]=v[2],v[1]
    area=edge(v[0],v[1],v[2][0],v[2][1]);assert area>0
    color=bytearray(16384);depth=bytearray(struct.pack('<I',bits(1))*4096)
    count=0;different=0;min_margin=F(100);signatures=[]
    for yy in range(64):
        for xx in range(64):
            x,y=F(2*xx+1,2),F(2*yy+1,2)
            edges=[edge(v[i],v[(i+1)%3],x,y) for i in range(3)]
            hit=all(e>0 or (e==0 and (v[(i+1)%3][1]<v[i][1] or (v[(i+1)%3][1]==v[i][1] and v[(i+1)%3][0]>v[i][0]))) for i,e in enumerate(edges))
            bary=[edges[1]/area,edges[2]/area,edges[0]/area]
            z=sum(b*q[2] for b,q in zip(bary,v))
            if not hit or not 0<=z<=1:continue
            reciprocal=sum(b/q[3] for b,q in zip(bary,v));tests=[];affine_tests=[]
            for a in attrs:
                for c in range(a['n']):
                    values=[F(a['values'][q[4]][c]) for q in v]
                    linear=sum(b*q for b,q in zip(bary,values))
                    if a['mode']=='flat':value=F(a['values'][0][c])
                    elif a['mode']=='noperspective':value=linear
                    else:value=sum(b*q/vert[3] for b,q,vert in zip(bary,values,v))/reciprocal
                    if a['type']=='uint':
                        tests.append(value==3);affine_tests.append(value==3)
                    else:
                        threshold=F(12031,65536)
                        assert value!=threshold
                        min_margin=min(min_margin,abs(value-threshold))
                        tests.append(value<threshold)
                        affine_tests.append((linear if a['mode']=='smooth' else value)<threshold)
            # Every scalar affects at least one output color bit. RGB triangles
            # use one bit per component. Multiple-attribute arms XOR separate
            # component tests into those bits, including the integer flat tag.
            def rgb(xs):return [255*bool(sum(xs[j::3])%2) for j in range(3)] if xs else [255,0,0]
            result=rgb(tests);different+=result!=rgb(affine_tests)
            off=4*(64*yy+xx);color[off:off+4]=bytes(result+[255]);depth[off:off+4]=struct.pack('<I',bits(z));count+=1
    return bytes(color),bytes(depth),dict(covered=count,affine_color_mismatches=different,min_threshold_margin=str(min_margin))

def shader(p, attrs, stage):
    mesh=stage=='mesh';fs=stage=='frag'
    s='#version 460\n'
    if mesh:s+='#extension GL_EXT_mesh_shader : require\nlayout(local_size_x=1) in;\nlayout(triangles,max_vertices=3,max_primitives=1) out;\n'
    for a in attrs:
        s+=f"layout(location={a['slot']},component={a['component']}) {a['mode']} {'in' if fs else 'out'} {ty(a)} {a['name']}{'[]' if mesh else ''};\n"
    if fs:
        s+='layout(location=0) out vec4 outColor;\nvoid main(){\n'
        tests=[]
        for a in attrs:
            for c in range(a['n']):
                val=a['name']+(f'[{c}]' if a['n']>1 else '')
                tests.append(f'uint({val}==3u)' if a['type']=='uint' else f'uint({val}<0.1835784912109375)')
        channels=['(('+'+'.join(tests[i::3])+')&1u)' if tests[i::3] else ('1u' if not tests and i==0 else '0u') for i in range(3)]
        return s+'outColor=vec4('+','.join('float('+x+')' for x in channels)+',1);}\n'
    s+='const vec4 positions[3]=vec4[3]('+','.join('vec4('+','.join(scalar(x,'float') for x in q)+')' for q in p)+');\n'
    for a in attrs:
        constants=[ty(a)+'('+','.join(scalar(x,a['type']) for x in values)+')' for values in a['values']]
        s+=f"const {ty(a)} src_{a['name']}[3]={ty(a)}[3]("+','.join(constants)+');\n'
    s+='void main(){\n'
    if mesh:
        s+='SetMeshOutputsEXT(3,1);\n'
        for i in range(3):
            s+=f'gl_MeshVerticesEXT[{i}].gl_Position=positions[{i}];\n'
            for a in attrs:s+=f"{a['name']}[{i}]=src_{a['name']}[{i}];\n"
        s+='gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);\n'
    else:
        s+='gl_Position=positions[gl_VertexIndex];\n'
        for a in attrs:s+=f"{a['name']}=src_{a['name']}[gl_VertexIndex];\n"
    return s+'}\n'

def generate(root):
    root.mkdir(parents=True,exist_ok=True);(root/'shaders').mkdir(exist_ok=True);(root/'expected').mkdir(exist_ok=True)
    rows=[]
    for kind in ['CONTROL','SMOOTH','FLAT','NOPERSPECTIVE','MULTIPLE','SMOOTH_CROSS','MULTIPLE_CROSS']:
        for halfz in [False,True]:
            name=kind+('_HALFZ' if halfz else '_ZERO_ONE');p=positions(kind,halfz);attrs=attributes(kind)
            color,depth,stats=oracle(p,halfz,attrs)
            (root/'expected'/(name+'.rgba')).write_bytes(color);(root/'expected'/(name+'.depth')).write_bytes(depth)
            for stage in ['vert','mesh','frag']:
                path=root/'shaders'/(name+'.'+stage);path.write_text(shader(p,attrs,stage))
                subprocess.run(['glslangValidator','-V','--target-env','vulkan1.3',str(path),'-o',str(path)+'.spv'],check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['spirv-val','--target-env','vulkan1.3',str(path)+'.spv'],check=True)
            groups={'smooth':0,'flat':1,'noperspective':2}
            order=sorted([(groups[a['mode']],a['slot'],a['component']+c) for a in attrs for c in range(a['n'])])
            local=sorted(order,key=lambda x:(x[1],x[2]));channels=[[4+order.index(x),96+16*i] for i,x in enumerate(local)]
            rows.append(dict(name=name,kind=kind,halfz=halfz,pending_attributes=False,fragment=name,
                positions=[[str(x) for x in q] for q in p],position_bits=[[bits(x) for x in q] for q in p],
                attributes=[{**a,'values':[[str(x) for x in q] for q in a['values']]} for a in attrs],
                user_components=len(order),channel_planes=[[c,32+16*c] for c in range(4)]+channels,**stats))
    for halfz in [False,True]:
        a=next(x for x in rows if x['kind']=='SMOOTH' and x['halfz']==halfz)
        b=next(x for x in rows if x['kind']=='NOPERSPECTIVE' and x['halfz']==halfz)
        assert (root/'expected'/(a['name']+'.rgba')).read_bytes()!=(root/'expected'/(b['name']+'.rgba')).read_bytes()
    (root/'ARMS.json').write_text(json.dumps(dict(no_device_execution=True,native_attributes=True,arms=rows),indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();generate(a.root)
