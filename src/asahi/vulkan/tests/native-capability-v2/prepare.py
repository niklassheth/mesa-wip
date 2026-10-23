#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Freeze whole independently compiled native bodies, DATA shapes and oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from oracles import shape, write

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, obj): p.write_text(json.dumps(obj, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--compiler', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sysroot', type=Path, default=Path('/home/nsheth/Projects/asahi/tmp/mini-sysroot'))
    a = ap.parse_args()
    out, base = a.out.resolve(), a.base.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for d in ['exports', 'oracles', 'shaders', 'source', 'lib']:
        (out/d).mkdir()
    for line in (base/'SHA256SUMS').read_text().splitlines():
        h, f = line.split('  ', 1)
        assert sha(base/f) == h, f
    shutil.copytree(HERE, out/'source', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    compiler_source = REPO/'src/asahi/compiler/research/native-capability-20261022/compile_native.c'
    shutil.copy2(compiler_source, out/'source/compile_native.c')
    for name in ['transport.py', 'kmsg.py', 'run.py', 'dry_run.py', 'oracles.py', 'host_checks.py', 'check_parser.py', 'README.md']:
        if (HERE/name).exists(): shutil.copy2(HERE/name, out/name)
    for name in ['libvulkan_asahi.so', 'libasahi_host_trace_drm_shim.so']:
        shutil.copy2(base/'lib'/name, out/'lib'/name)
    shutil.copy2(base/'shaders/BASE.mesh.spv', out/'shaders')
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.2', str(HERE/'count.frag'), '-o', str(out/'shaders/depth-color.frag.spv')], check=True)
    for topo in [6, 1, 0]:
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.2', str(HERE/f'control-{topo}.vert'), '-o', str(out/f'shaders/control-{topo}.vert.spv')], check=True)
    arms = []
    specs = [(v,p,max(16,1 << (v-1).bit_length()),topo,1)
             for topo in [6,1,0]
             for v,p in [(3,1),(4,2),(16,16),(64,128),(128,256),(256,512)]]
    specs += [(v,1,max(16,1 << (v-1).bit_length()),6,1) for v in [4,16,64,128,256]]
    specs += [(3,p,16,6,1) for p in [2,128,256,512]]
    specs += [(3,1,t,6,1) for t in [32,64,128,256]]
    specs += [(64,1,64,6,0)]
    assert len(specs)==len(set(specs))
    specs = [(*x, 0) for x in specs]
    specs += [(v,p,max(16,1 << (v-1).bit_length()),topo,1,1)
              for topo in [6,1,0] for v,p in [(4,2),(16,16),(64,128),(128,256),(256,512)]]
    for v,p,t,topo,c42,high in specs:
        s=shape(v,p,t,topo,c42,high)
        name=f'T{topo}_V{v}_P{p}_W{t}_C{c42}_H{high}'
        prefix = out/'exports'/name
        subprocess.run([str(a.compiler.resolve()), str(prefix), str(v), str(p), str(t), str(topo), str(c42), str(prefix)+'.nir', str(high)], check=True)
        mesh, obj = [Path(str(prefix)+f'.{stage}.bin') for stage in ['mesh', 'object']]
        actual = {stage: json.loads(Path(str(prefix)+f'.{stage}.json').read_text()) for stage in ['mesh', 'object']}
        assert actual['mesh']['publications'] == 0 and actual['object']['publications'] == 1
        # Staging adapter labels are frozen. The bridge checks them, then
        # replaces owned DATA shape/allocation before any submit. Actual
        # compiler ABI/resource metadata remain separately recorded.
        Path(str(mesh)+'.meta').write_text(f'{mesh.stat().st_size} 96 0 0 0 96 20 7 12\n')
        Path(str(obj)+'.meta').write_text(f'{obj.stat().st_size} 96 1 0 0 0 16 7\n')
        fields = ['vertices', 'primitives', 'threads', 'topology', 'planes', 'local_B', 'allocation_B', 'control42']
        Path(str(mesh)+'.shape').write_text(' '.join(str(s[x]) for x in fields)+'\n')
        runtimes = [('full', v, p), ('partial', max(s['indices'],v-1), max(1,p//2)), ('zero_p', v, 0), ('zero_v', 0, p), ('tiny', s['indices'], 1)]
        if high: runtimes = [('high_index', v, p)]
        for runtime, rv, rp in runtimes:
            aid = len(arms)
            write(out/'oracles'/f'{aid:03d}.gz', s, rv, rp)
            arms.append(dict(id=aid, name=name+'_'+runtime, shape=s, runtime_counts=[rv,rp], actual_compiler_metadata=actual,
                api_grid=[1,1,1], cross_family='frozen', programs={stage:dict(file=b.name, sha256=sha(b), bytes=b.stat().st_size) for stage,b in [('mesh',mesh),('object',obj)]}))
    lib = a.sysroot.resolve()/'usr/lib64'
    cmd = ['aarch64-linux-gnu-gcc', '-std=gnu11', '-O2', '-g', '-Wall', '-Wextra', '-Wno-missing-field-initializers', '-Wno-misleading-indentation', '-Wno-sign-compare', '-rdynamic', '-I'+str(REPO/'include/drm-uapi'), '-idirafter', '/usr/include', str(HERE/'carrier/runner.c'), '-Wl,-rpath-link,'+str(lib), str(lib/'libvulkan.so.1'), str(lib/'libz.so.1'), '-ldl', '-pthread', '-o', str(out/'runner')]
    subprocess.run(cmd, check=True)
    cc = list(cmd)
    cc[cc.index(str(HERE/'carrier/runner.c'))] = str(HERE/'carrier/calibrate.c')
    cc[-1] = str(out/'calibrate')
    subprocess.run(cc, check=True)
    env = json.loads((base/'CATALOG.json').read_text())['environment']
    dump(out/'CATALOG.json', dict(campaign='native-capability-C2', arms=arms, environment=env, mode='kernel research admission value from frozen kernel requirements', lanes=1))
    dump(out/'KERNEL-REQUIREMENTS.template.json', dict(ready=False, reason='Kernel owner supplies this separate file after testing frozen fixtures; manager schedules device.', native_mesh_mode=3, fixtures_catalog_sha256='FILL_FROM_FROZEN_KIT', kernel_source_commit='FILL', kernel_image_sha256='FILL', kernel_dtb_sha256='FILL', kernel_release='FILL', kernel_version='FILL', all_fixtures_accepted=False, uapi_changed=False))
    dump(out/'BUILD.json', dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), command=cmd, base_manifest_sha256=sha(base/'SHA256SUMS'), compiler_sha256=sha(a.compiler), gpu_execution=False, arms=len(arms)))
    print(out, len(arms), 'arms; host validation and kernel pin pending')


if __name__ == '__main__': main()
