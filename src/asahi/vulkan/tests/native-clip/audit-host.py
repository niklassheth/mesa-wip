#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check the actual U arguments, full uploaded programs and native state."""
import argparse, importlib.util, json, struct
from pathlib import Path
from packet_audit import Memory, programs, stream, SOURCE, OPCODES

def audit(root, dry):
    path=SOURCE/'src/asahi/compiler/test/check-apple9-native-mesh.py'
    spec=importlib.util.spec_from_file_location('u_check',path);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    arms=json.loads((root/'ARMS.json').read_text())['arms'];results=[];by_arm={}
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
                        draws+=stream(memory,address,compiled)
                    at+=8+size
                assert at==len(b)
            assert len(draws)==1
            draw=draws[0];mesh=next(p for p in compiled if p['stage']==7 and not p['preamble'])
            rows=[]
            for x in mesh['instructions']:
                r=dict(x);r.update(op=OPCODES[int(r['op'])],imm=int(r['imm'],0),offset=int(r['offset']),
                    code=bytes.fromhex(r['bytes']),src=r['src'].split(',') if r['src'] else []);rows.append(r)
            u.check_program(rows)
            # The native packet is the same position-only class in both modes.
            state=list(draw['state']);state[6]=state[8]=0
            assert state==[0x40000d2c,0x504,0x10,0,0,0x1000000,0,0x1000000,0,1,0,0x400810,0,5]
            row=dict(arm=arm['name'],arch=arch,normalized_state=state,**draw)
            (d/'AUDIT.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
            if arch=='host':by_arm[arm['name']]=draw
            else:
                twin=by_arm[arm['name']]
                for stage in ['mesh','object']:
                    assert draw[stage]['code_sha256']==twin[stage]['code_sha256']
                    assert draw[stage]['launch_sha256']==twin[stage]['launch_sha256']
    (dry.parent/'SUBMISSION-AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
    print(len(results),'native submissions: full uploaded code, launches, roots, U0/four channels/scalar and state verified; host/ARM code+launch identical')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True);a=p.parse_args();audit(a.root,a.dry)
