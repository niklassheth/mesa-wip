#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Audit U publication and compare each real VS/native PPP interpolation contract."""
import argparse, importlib.util, json, struct, sys
from pathlib import Path
shared=Path(__file__).resolve().parent.parent/'native-clip';sys.path.insert(0,str(shared))
from packet_audit import Memory, CONTEXT_BASE, USC_BASE
spec=importlib.util.spec_from_file_location('native_audit',shared/'audit-host.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

def interpolation(memory,address,mesh):
    visited=set()
    while True:
        assert address not in visited;visited.add(address)
        header=struct.unpack('<I',memory.read(address,4))[0]
        if header!=0x80000000:break
        address=CONTEXT_BASE+struct.unpack('<I',memory.read(address+4,4))[0]
    assert header==(0x40000d2c if mesh else 0x4000002e)
    ppp=CONTEXT_BASE+struct.unpack('<I',memory.read(address+(60 if mesh else 32),4))[0]
    data=bytearray(memory.read(ppp,256))
    count=struct.unpack_from('<I',data,0x44)[0]>>16
    cf=struct.unpack_from('<I',data,0x48)[0]
    coefficients=list(struct.unpack('<'+'I'*(count+1),memory.read(USC_BASE+cf,4*(count+1))))
    # Batch pointers may differ; every other PPP byte must match the VS draw.
    for offset in [0x48,0x54]:data[offset:offset+4]=bytes(4)
    return dict(ppp=data.hex(),coefficient_words=coefficients)

def read_stage(directory,mesh):
    results=[]
    for f in sorted(directory.glob('*.commands.bin')):
        memory=Memory(directory,f.name.split('.')[0]);data=f.read_bytes();at=0
        while at<len(data):
            kind,size,_,_=struct.unpack_from('<4H',data,at);assert at+8+size<=len(data)
            if kind==0:
                address=struct.unpack_from('<Q',data,at+16)[0]
                results.append(interpolation(memory,address,mesh))
            at+=8+size
        assert at==len(data)
    assert len(results)==1
    return results[0]

def audit(root,dry):
    native.audit(root,dry)
    rows=[]
    for arch in ['host','arm']:
        for arm in json.loads((root/'ARMS.json').read_text())['arms']:
            vs=read_stage(dry/arch/(arm['name']+'.vert'),False)
            mesh=read_stage(dry/arch/(arm['name']+'.mesh'),True)
            assert vs==mesh,(arch,arm['name'],'VS/native PPP or coefficients differ')
            n=arm['user_components'];words=vs['coefficient_words']
            assert words[0]==(n+1)|((n+1)<<8) and words[1]==0xc
            shades=[];sources=[]
            for word in words[2:]:
                count=(word&3)+1;shade=(word>>2)&7;source=(word>>8)&255;coefficient=word>>16
                assert source==coefficient==len(shades)+1
                shades += [shade]*count;sources += list(range(source,source+count))
            expected=[shade for mode,shade in [('smooth',7),('flat',0),('noperspective',3)]
                      for a in arm['attributes'] if a['mode']==mode for _ in range(a['n'])]
            assert shades==expected and sources==list(range(1,n+1))
            rows.append(dict(arch=arch,arm=arm['name'],paired_ppp_identical_except_batch_pointers=True,
                coefficient_words=words,shade_models=shades,user_components=n))
    (dry.parent/'INTERPOLATION-AUDIT.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(len(rows),'VS/native pairs: matching PPP, 1/W coefficient, contiguous varying sources, smooth/flat/linear modes')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.dry)
