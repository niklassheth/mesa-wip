#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check the actual U arguments, full uploaded programs and native state."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, re, struct, sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native-clip"))
from packet_audit import Memory, programs, stream, SOURCE, OPCODES

def independent_runs(program):
    # This comparison permits only scheduling of independent immediate moves
    # or full-register copies. A copy run must have disjoint sources/destinations
    # and no waits/producers. Never move across an execution-mask instruction.
    # It never changes, replaces or canonicalizes the uploaded program bytes.
    def kind(row):
        if int(row['wait'],0) or int(row['producer'],0):return None
        op=OPCODES[int(row['op'])]
        if op=='IMM' and not row['src']:return 'immediate'
        src=row['src'].split(',')
        if op=='IOR' and len(src)==2 and src[0]==src[1] and all(
                re.fullmatch(r'r[0-9]+',x) for x in [row['dst'],*src]):return 'copy'
        return None
    rows=program['instructions'];out=[];i=0
    while i<len(rows):
        j=i
        category=kind(rows[i])
        while category and j<len(rows) and kind(rows[j])==category:
            j+=1
        if j>i:
            run=rows[i:j];destinations={r['dst'] for r in run}
            independent=len(destinations)==len(run)
            if category=='copy':
                independent &= not destinations.intersection(r['src'].split(',')[0] for r in run)
            if independent:out.append(sorted(r['bytes'] for r in run))
            else:out.extend(r['bytes'] for r in run)
            i=j
        else:out.append(rows[i]['bytes']);i+=1
    return out

def audit(root, dry):
    path=SOURCE/'src/asahi/compiler/test/check-apple9-native-mesh.py'
    spec=importlib.util.spec_from_file_location('u_check',path);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    arms=json.loads((root/'ARMS.json').read_text())['arms'];results=[];by_arm={};host_programs={}
    for arch in ['host','arm']:
        for arm in arms:
            if arm['pending_attributes']:continue
            d=dry/arch/(arm['name']+'.mesh');run=json.loads((d/'RUN.json').read_text())
            assert run['host_check_pass'] and run['no_gpu_execution']
            compiled=programs((d/'log.txt').read_text());draws=[]
            for f in sorted(d.glob('*.commands.bin')):
                memory=Memory(d,f.name.split('.')[0]);b=f.read_bytes();at=0
                while at<len(b):
                    kind,size,vdm,cdm=struct.unpack_from('<4H',b,at);assert at+size+8<=len(b)
                    if kind==0:
                        flags,_,address=struct.unpack_from('<IIQ',b,at+8)
                        got=stream(memory,address,compiled,arm.get('user_components',0),physical_threads=32)
                        for draw in got:
                            for name,stage in [('mesh',7),('object',6)]:
                                launch=draw[name]
                                main=next(p for p in compiled if p['stage']==stage and not p['preamble'])
                                stop=bytes.fromhex('0e000000')
                                # Object code contains its own HALT followed by
                                # the out-of-line collective leaf and return.
                                if name=='mesh':
                                    assert memory.read(int(launch['entry'],0),len(main['code'])+4)==main['code']+stop
                                raw=memory.read(int(launch['address'],0),256);branch=194
                                preambles=[p for p in compiled if p['stage']==stage and p['preamble']]
                                assert len(preambles)<=1
                                pre=b''
                                if preambles:
                                    assert raw[branch:branch+3]==bytes.fromhex('0f0054')
                                    va=int(launch['address'],0)+branch+int.from_bytes(raw[branch+3:branch+9],'little',signed=True)
                                    pre=preambles[0]['code']+stop
                                    assert memory.read(va,len(pre))==pre
                                else:assert raw[branch:branch+4]==stop
                                if name=='mesh' and arm.get('require_preamble'):
                                    assert pre, 'arm must retain its ordinary preamble'
                                launch['preamble_bytes']=len(pre)
                                launch['preamble_sha256']=hashlib.sha256(pre).hexdigest() if pre else None
                        draws+=got
                    at+=8+size
                assert at==len(b)
            assert len(draws)==1
            draw=draws[0];mesh=next(p for p in compiled if p['stage']==7 and not p['preamble'])
            rows=[]
            for x in mesh['instructions']:
                r=dict(x);r.update(op=OPCODES[int(r['op'])],imm=int(r['imm'],0),offset=int(r['offset']),
                    code=bytes.fromhex(r['bytes']),src=r['src'].split(',') if r['src'] else []);rows.append(r)
            u.check_program(rows,arm.get('channel_planes'))
            scalar=next(r for r in rows if r['op']=='MESH_SUBMIT_SCALAR')
            base=(int.from_bytes(scalar['code'],'little')>>24)&127
            assert scalar['src']==[f'r{base}',f'r{base+1}']
            draw['mesh']['scalar_pair']=dict(base=base,neighbor=base+1,second_value=0,
                explicit_source=True,zero_def_use_checked=True,device_register_measurement=False)
            control=next(r for r in rows if r['op']=='MESH_CONTROL')
            word=int.from_bytes(control['code'],'little')
            draw['mesh']['U0_halves']=[(word>>24)&255,(word>>33)&255]
            # Counts and allocation follow the declared scalar output map.
            state=list(draw['state']);state[6]=state[8]=0
            components=4+arm.get('user_components',0)
            assert state==[0x40000d2c,((components+1)<<8)|components,0x10,0,0,0x1000000,0,0x1000000,0,1,0,0x400820,0,components+1]
            counts=[sum(a['n'] for a in arm.get('attributes',[]) if a['mode']==m)
                    for m in ['smooth','flat','noperspective']]
            assert draw['ppp_header']==[0x10040000,counts[0]|counts[1]<<8|counts[2]<<16]
            row=dict(arm=arm['name'],arch=arch,normalized_state=state,**draw)
            (d/'AUDIT.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
            if arch=='host':
                by_arm[arm['name']]=draw;host_programs[arm['name']]=compiled
            else:
                twin=by_arm[arm['name']]
                for stage in ['mesh','object']:
                    code_stage=7 if stage=='mesh' else 6
                    left=next(p for p in compiled if p['stage']==code_stage and not p['preamble'])
                    right=next(p for p in host_programs[arm['name']] if p['stage']==code_stage and not p['preamble'])
                    equal=left['code']==right['code']
                    assert equal or independent_runs(left)==independent_runs(right), (arm['name'],stage)
                    row[stage]['host_arm_code_relation']='byte_identical' if equal else 'independent_immediate_or_copy_order_only'
                    assert draw[stage]['launch_sha256']==twin[stage]['launch_sha256']
                    assert draw[stage]['preamble_sha256']==twin[stage]['preamble_sha256']
                (d/'AUDIT.json').write_text(json.dumps(row,indent=2)+'\n')
    (dry.parent/'SUBMISSION-AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
    print(len(results),'native submissions: full uploaded code, launches, roots, U0/channel planes/scalar and state verified; host/ARM launch identical; code differences limited to independent immediate/copy scheduling')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True);a=p.parse_args();audit(a.root,a.dry)
