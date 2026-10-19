#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
import argparse, importlib.util, json, re
from pathlib import Path

def programs(path):
    result=[];current=None
    for line in path.read_text().splitlines():
        if line.startswith('APPLE9_ASM_BEGIN'):
            current=[] if line=='APPLE9_ASM_BEGIN stage=4' else None
        elif line=='APPLE9_ASM_END':
            if current is not None:result.append(''.join(current))
            current=None
        elif current is not None and line.startswith('APPLE9_ASM offset='):
            current.append(re.search(r'bytes=(\S+)',line)[1])
    return result

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dry',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('link',Path(__file__).resolve().parents[1]/'native-attributes/audit-host.py')
link=importlib.util.module_from_spec(spec);spec.loader.exec_module(link)
link.native.audit(a.root,a.dry)
rows=[]
for arch in ['host','arm']:
    for arm in json.loads((a.root/'ARMS.json').read_text())['arms']:
        vs=a.dry/arch/(arm['name']+'.vert');mesh=a.dry/arch/(arm['name']+'.mesh')
        fvs=programs(vs/'log.txt');fmesh=programs(mesh/'log.txt')
        assert fvs and fmesh
        v=link.read_stage(vs,False);m=link.read_stage(mesh,True)
        words=m['coefficient_words'];n=arm['user_components']
        assert words[0]==(n+1)|((n+1)<<8) and words[1]==0xc
        shades=[]
        for word in words[2:]:
            count=(word&3)+1;shade=(word>>2)&7
            assert (word>>8)&255==word>>16==len(shades)+1
            shades += [shade]*count
        expected=[shade for mode,shade in [('flat',0),('noperspective',3)]
                  for attr in arm['attributes'] if attr['mode']==mode for _ in range(attr['n'])]
        assert shades==expected,(arch,arm['name'],shades,expected)
        if arm['name'] in ['CONTROL','PID','CULL_FALSE','CULL_TRUE']:
            assert fvs==fmesh and v==m
        rows.append(dict(arch=arch,arm=arm['name'],fragment_programs_byte_identical=fvs==fmesh,
            native_coefficient_words=words,expected_shades=expected,
            ppp_identical_except_batch_pointers=v==m,
            note='The ordinary VS linker can reduce correlated flat outputs; native mesh retains each declared component.'))
(a.dry.parent/'LINK-AUDIT.json').write_text(json.dumps(rows,indent=2)+'\n')
print(len(rows),'native output/fragment coefficient contracts verified; matching control/PID/cull fragment programs and PPP')
