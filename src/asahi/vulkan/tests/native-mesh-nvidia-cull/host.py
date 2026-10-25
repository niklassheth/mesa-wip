#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check unchanged NVIDIA culling shaders on a software Vulkan implementation."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from check import CASES, inputs, mutations, verify

p = argparse.ArgumentParser(description=__doc__)
for name in ('corpus', 'records', 'probe', 'lavapipe', 'out'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--sysroot', type=Path)
p.add_argument('--keep-going', action='store_true', help='Collect software reference mismatches; exit unsuccessfully')
a = p.parse_args()
out = a.out.resolve(); out.mkdir()
(out/'shaders').mkdir()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data): path.write_text(json.dumps(data, indent=2)+'\n')
write(out/'MUTATIONS.json', mutations())
pins = {}
for task in (0,1):
    for stage in ('mesh', 'frag', 'task') if task else ('mesh', 'frag'):
        source = (a.corpus/'compiled'/('nvpro-samples-gl_vk_meshlet_cadscene--cull-task'+str(task)+'.spv')
                  if stage=='mesh' else a.records/('NVIDIA-cull-task'+str(task))/(stage+'.spv'))
        dest = out/'shaders'/('task'+str(task)+'.'+stage+'.spv')
        shutil.copy2(source, dest)
        pins[str(source.resolve())] = sha(source)
        subprocess.run(['spirv-val','--target-env','vulkan1.3',str(dest)],check=True)
write(out/'SOURCE-PINS.json', pins)
write(out/'icd.json', dict(file_format_version='1.0.0', ICD=dict(
    library_path=str(a.lavapipe.resolve()), api_version='1.4.0')))
env = {k:v for k,v in os.environ.items() if not k.startswith(('HK_', 'VK_', 'LD_', 'MESA_', 'AGX_', 'ASAHI_'))}
env.update(VK_DRIVER_FILES=str(out/'icd.json'), VK_LOADER_LAYERS_DISABLE='~implicit~',
           LP_NUM_THREADS='2', LIBGL_ALWAYS_SOFTWARE='1', M2_SOFTWARE_REFERENCE='1')
rows = []
for task, profile in CASES:
    d = out/('task'+str(task)+'-'+profile); d.mkdir()
    (d/'inputs.bin').write_bytes(inputs(profile))
    stem = out/'shaders'/('task'+str(task))
    argv = [str(a.probe.resolve()), str(stem)+'.mesh.spv', str(stem)+'.frag.spv',
            str(d/'inputs.bin'), str(d)]
    if task: argv.append(str(stem)+'.task.spv')
    if a.sysroot:
        argv = ['qemu-aarch64-static','-L',str(a.sysroot.resolve()),'-E',
                'LD_LIBRARY_PATH='+str(a.sysroot.resolve()/'usr/lib64'), *argv]
    write(d/'RUN.json',dict(argv=argv,environment={k:v for k,v in env.items()
        if k.startswith(('VK_', 'M2_', 'LP_', 'GALLIVM_', 'LIBGL_'))}))
    with (d/'stdout.log').open('w') as f:
        subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
    result = verify(d,profile); write(d/'PIXELS.json',result)
    if not a.keep_going:
        assert result['verdict']=='PASS', (task,profile,result)
    sizes = [(f.name,f.stat().st_size) for f in d.iterdir() if f.is_file()]
    rows.append(dict(task=task,profile=profile,result=result,bytes=sum(n for _,n in sizes),files=len(sizes)))
    print('task'+str(task)+' '+profile+': software '+result['verdict'],flush=True)
passed = all(r['result']['verdict']=='PASS' for r in rows)
write(out/'RESULT.json',dict(verdict='PASS' if passed else 'FAIL',cases=rows,source_pins=pins,
    gpu_execution=False,scope='Host software reference; no Apple GPU execution'))
raise SystemExit(0 if passed else 1)
