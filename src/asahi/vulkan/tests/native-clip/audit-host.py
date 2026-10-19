#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check the actual U arguments, full uploaded programs and native state."""
import argparse, importlib.util, json, struct
from pathlib import Path
from packet_audit import Memory, programs, stream, SOURCE, OPCODES

def immediate_runs(program):
    # This comparison permits only scheduling of independent immediate moves.
    # It never changes, replaces or canonicalizes the uploaded program bytes.
    rows=program['instructions'];out=[];i=0
    while i<len(rows):
        j=i
        while j<len(rows) and OPCODES[int(rows[j]['op'])]=='IMM' and not rows[j]['src'] and int(rows[j]['wait'],0)==0 and int(rows[j]['producer'],0)==0:
            j+=1
        if j>i:
            run=rows[i:j];assert len({r['dst'] for r in run})==len(run)
            out.append(sorted(r['bytes'] for r in run));i=j
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
                        draws+=stream(memory,address,compiled,arm.get('user_components',0))
                    at+=8+size
                assert at==len(b)
            assert len(draws)==1
            draw=draws[0];mesh=next(p for p in compiled if p['stage']==7 and not p['preamble'])
            rows=[]
            for x in mesh['instructions']:
                r=dict(x);r.update(op=OPCODES[int(r['op'])],imm=int(r['imm'],0),offset=int(r['offset']),
                    code=bytes.fromhex(r['bytes']),src=r['src'].split(',') if r['src'] else []);rows.append(r)
            u.check_program(rows,arm.get('channel_planes'))
            # Counts and allocation follow the declared scalar output map.
            state=list(draw['state']);state[6]=state[8]=0
            components=4+arm.get('user_components',0)
            assert state==[0x40000d2c,((components+1)<<8)|components,0x10,0,0,0x1000000,0,0x1000000,0,1,0,0x400810,0,components+1]
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
                    assert equal or immediate_runs(left)==immediate_runs(right)
                    row[stage]['host_arm_code_relation']='byte_identical' if equal else 'independent_immediate_move_order_only'
                    assert draw[stage]['launch_sha256']==twin[stage]['launch_sha256']
                (d/'AUDIT.json').write_text(json.dumps(row,indent=2)+'\n')
    (dry.parent/'SUBMISSION-AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
    print(len(results),'native submissions: full uploaded code, launches, roots, U0/channel planes/scalar and state verified; host/ARM launch identical; code differences limited to independent immediate-move scheduling')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True);a=p.parse_args();audit(a.root,a.dry)
