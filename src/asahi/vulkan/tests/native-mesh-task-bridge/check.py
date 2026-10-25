#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent full-image and full-buffer oracles for original task triples."""
import hashlib, math, struct

Q_BYTES=512*1024
TOLERANCE=2e-5

def cases():
    common=dict(mode=3,env={'HK_APPLE9_NATIVE_INDEX':'1','HK_APPLE9_NATIVE_MESH':'1',
        'HK_APPLE9_NATIVE_TASK_BRIDGE':'1'},vertices=3,primitives=1,pixel_bytes=16)
    rows=[dict(name='Sascha-task-'+lang,kind='sascha',language=lang,probe='mesh-task-ubo',
        parents=[1,1,1],children=[3,1,1],cookie=0,physical=16,max_vertices=3,
        max_primitives=1,components=8,output_rows=9) for lang in ('glsl','hlsl','slang')]
    rows += [dict(name='Khronos-payload-'+name,kind='khronos',probe='mesh-task-ubo',
        parents=[2,1,1],children=[4,1,1] if radius else [0,1,1],radius=radius,cookie=0,
        physical=32,max_vertices=16,max_primitives=8,components=7,output_rows=29)
        for name,radius in [('two-parents',4),('zero-children',0)]]
    rows += [dict(name='payload16k-'+name,kind='authored',probe='mesh-task',parents=parents,
        children=children,skip_middle=skip,cookie=cookie,physical=32,max_vertices=3,
        max_primitives=1,components=8,output_rows=9)
        for name,parents,children,skip,cookie in [
            ('two-children',[1,1,1],[2,1,1],False,17),
            ('three-parents-3d',[1,3,1],[2,2,2],True,29),
            ('zero-children',[2,1,1],[0,1,1],False,41)]]
    return [{**common,**r} for r in rows]

def initial_q(case):
    q=bytearray([0xcc])*Q_BYTES
    if case['kind']=='sascha':
        identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
        model=[.25,0,0,0,0,.5,0,0,.5,0,.25,0,-.5,0,.125,1]
        struct.pack_into('<48f',q,0,*(identity+model+identity))
    elif case['kind']=='khronos':
        struct.pack_into('<4f',q,0,-1,-1,case['radius'],1)
    return bytes(q)

def expected(case):
    p=[0.0]*(128*64*4);q=bytearray(initial_q(case))
    if case['kind']=='sascha':
        for center in (32,64,96):
            for y in range(64):
                for x in range(128):
                    a=(48-(y+.5))/32
                    c=((x+.5)-(center-16)-16*a)/32
                    b=1-a-c
                    if min(a,b,c)>0:p[(y*128+x)*4:(y*128+x)*4+4]=[c,a,b,1]
    elif case['kind']=='khronos' and case['radius']:
        for y in range(64):
            for x in range(128):
                u=((x%16)+.5)/16;v=((y%16)+.5)/16
                p[(y*128+x)*4:(y*128+x)*4+4]=[1-u,abs(u-v),min(u,v),1]
    elif case['kind']=='authored':
        parents=math.prod(case['parents']);dx,dy,dz=case['children']
        for t in range(parents):
            struct.pack_into('<4I',q,t*16,case['cookie']^t,*case['parents'])
            if case['skip_middle'] and t==1:continue
            for z in range(dz):
                for y in range(dy):
                    for x in range(dx):
                        g=x+dx*(y+dy*z);slot=t*8+g
                        data=[case['cookie']^(t*0x10001)^(i*0x101) for i in range(4096)]
                        struct.pack_into('<4096I',q,(4096+slot*4096)*4,*data)
                        struct.pack_into('<12I',q,(256+slot*16)*4,t,x,y,z,dx,dy,dz,32*x,y,z,data[0],data[-1])
                        ox=slot%8*16;oy=slot//8*16
                        for py in range(oy+2,oy+13):
                            for px in range(ox+2,ox+14):
                                # 11*u+12*v<132 has no half-pixel ties.
                                u=px+.5-(ox+2);v=py+.5-(oy+2)
                                if 11*u+12*v<132:
                                    i=(py*128+px)*4;p[i:i+4]=[t+1,g+1,case['cookie'],1]
    return struct.pack('<32768f',*p),bytes(q)

def verify_bytes(pixels,q,case):
    ep,eq=expected(case)
    image=dict(ok=False,bytes=len(pixels),expected_bytes=len(ep),absolute_tolerance=TOLERANCE)
    if len(pixels)==len(ep):
        actual=struct.unpack('<32768f',pixels);reference=struct.unpack('<32768f',ep)
        bad=[i for i,(a,b) in enumerate(zip(actual,reference)) if not math.isfinite(a) or abs(a-b)>(TOLERANCE if b else 0)]
        image.update(ok=not bad,checked_components=len(actual),mismatches=len(bad),first=bad[:16])
    badq=[i for i in range(min(len(q),len(eq))//4) if q[4*i:4*i+4]!=eq[4*i:4*i+4]]
    buffer=dict(ok=q==eq,bytes=len(q),expected_bytes=len(eq),mismatches=len(badq),first=badq[:16])
    return dict(verdict='PASS' if image['ok'] and buffer['ok'] else 'FAIL',pixels=image,witness=buffer,
        sha256=dict(pixels=hashlib.sha256(pixels).hexdigest(),Q=hashlib.sha256(q).hexdigest()))

def verify(directory,case,subgroup=32):
    return verify_bytes((directory/'pixels.bin').read_bytes(),(directory/'Q.bin').read_bytes(),case)

def mutations():
    rows=[]
    for c in cases():
        p,q=expected(c);assert verify_bytes(p,q,c)['verdict']=='PASS'
        edits=[('unexecuted',bytes([0xa5])*len(p),initial_q(c)),('truncated',p[:-4],q),
            ('guard-write',p,q[:-4]+bytes(4))]
        if any(p):
            edits += [('missing-raster',bytes(len(p)),q)]
            words=struct.unpack('<32768f',p)
            pixel=next(i for i in range(0,len(words),4) if any(words[i:i+4]))
            for channel in range(4):
                changed=bytearray(p)
                struct.pack_into('<f',changed,(pixel+channel)*4,words[pixel+channel]+1)
                edits.append(('wrong-rgba-'+str(channel),changed,q))
            changed=bytearray(p);struct.pack_into('<f',changed,pixel*4,float('nan'))
            edits.append(('nonfinite',changed,q))
        if c['kind']=='authored':
            edits += [('task-side-effects-lost',p,initial_q(c))]
            if math.prod(c['children']):
                a=bytearray(q);a[(4096+4095)*4]^=1;edits += [('last-payload-word',p,a)]
                a=bytearray(q);a[256*4]^=1;edits += [('wrong-parent',p,a)]
        for name,pp,qq in edits:assert verify_bytes(pp,qq,c)['verdict']=='FAIL',(c['name'],name)
        rows.append(dict(case=c['name'],rejected=[e[0] for e in edits]))
    return dict(verdict='PASS',cases=rows,mutations_detected=sum(len(r['rejected']) for r in rows))
