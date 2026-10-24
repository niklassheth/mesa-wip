#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authored CTS-style M2a subset, software oracle plus real ARM ICD recording.

The software device executes the shader; the native ICD compiles and records
through the no-op DRM shim. Neither result is native GPU execution evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from check import compare, raster, witness

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
for name in ('out','host-probe','lavapipe','driver','shim','probe','sysroot'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
out = a.out.resolve()
out.mkdir()
shaders = out/'shaders'
shaders.mkdir()
cases = [dict(name=f'L{n}x1x1-subgroups', logical=[n,1,1], threads=n,
              physical=32, shared=False, mode=3, cookie=0x18100000+i)
         for i,n in enumerate((17,31,32,33,40,63,64,65,96,100,127,128))]
(out/'CASES.json').write_text(json.dumps(cases,indent=2)+'\n')
commands = []
for case in cases:
    commands.append(['glslangValidator','--target-env','vulkan1.3',
        f'-DSIZE_X={case["threads"]}','-DSIZE_Y=1','-DSIZE_Z=1',
        '-DUSE_SHARED=0','-DHOST_SUBGROUP_SUBSET=1',str(HERE/'workgroup.mesh'),
        '-o',str(shaders/(case['name']+'.mesh.spv'))])
commands.append(['glslangValidator','--target-env','vulkan1.3',str(HERE/'count.frag'),
                 '-o',str(shaders/'count.frag.spv')])
for cmd in commands:
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
(out/'BUILD.json').write_text(json.dumps(commands,indent=2)+'\n')
(out/'icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',
    ICD=dict(library_path=str(a.lavapipe.resolve()),api_version='1.4.0'))))
software = []
for case in cases:
    d=out/case['name']; d.mkdir()
    e={k:v for k,v in os.environ.items() if not k.startswith(
        ('HK_','VK_','LD_','MESA_','AGX_','ASAHI_','M1_','M2_'))}
    e.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_LOADER_LAYERS_DISABLE='~implicit~',
        LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',M2_SOFTWARE_REFERENCE='1')
    cmd=[str(a.host_probe.resolve()),str(shaders/(case['name']+'.mesh.spv')),
         str(shaders/'count.frag.spv'),str(d),'3','1','1',str(case['cookie'])]
    with (d/'stdout.log').open('w') as f:
        subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=30,check=True)
    log=(d/'stdout.log').read_text()
    assert 'llvmpipe' in log and log.count('SUBMIT_OK\n')==1
    sg=int(re.search(r'^subgroup_size=(\d+)$',log,re.M)[1])
    expected=bytearray(witness(case,sg)); n=case['threads']; cookie=case['cookie']
    def put(word,value): struct.pack_into('<I',expected,4*word,value)
    for lane in range(n):
        start=(lane//sg)*sg; stop=min(start+sg,n); row=2048+16*lane
        put(row+12,(1<<(stop-start))-1); put(row+13,int(lane==start))
        if lane&1:
            put(row+14,sum(1<<(k-start) for k in range(start,stop) if k&1))
            put(row+15,int(lane==start+1))
        put(8192+lane,cookie+lane+2000)
        put(8448+lane,3*(cookie+(lane+33)%n)+3000)
    image=compare((d/'pixels.bin').read_bytes(),raster())
    buffer=compare((d/'Q.bin').read_bytes(),expected)
    assert image['ok'] and buffer['ok'],(case,image,buffer)
    # Meaningful checker mutations for each requested semantic dimension.
    changed_words=[2048+12,2048+13,2048+16+14,2048+16+15,
                   2048+16*(n-1)+4,8192+n-1,8448+n-1,len(expected)//4-1]
    for word in changed_words:
        changed=bytearray(expected); changed[4*word]^=1
        assert not compare(changed,expected)['ok']
    software.append(dict(case=case,subgroup_size=sg,pixels=image,witness=buffer,
        semantic_mutations=len(changed_words)))
    print(case['name'],'software semantic oracle PASS',flush=True)
cmd=[sys.executable,str(HERE/'host-record.py'),'--cases',str(out/'CASES.json'),
     '--shaders',str(shaders),'--out',str(out/'native-records')]
for key in ('driver','shim','probe','sysroot'):
    cmd+=['--'+key,str(getattr(a,key).resolve())]
subprocess.run(cmd,check=True)
records=json.loads((out/'native-records/RESULT.json').read_text())
assert records['verdict']=='PASS' and len(records['cases'])==len(cases)
(out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,
    software_cases=software,native_compiled=len(records['cases']),native_rejected=0,
    native_selection=records['selection'],driver_sha256=records['driver_sha256'],
    source_sha256=hashlib.sha256((HERE/'workgroup.mesh').read_bytes()).hexdigest(),
    scope='shared-free subgroup ballot/elect/ID, nested masks, uniform loop/branch barriers, partial final subgroups'),indent=2)+'\n')
