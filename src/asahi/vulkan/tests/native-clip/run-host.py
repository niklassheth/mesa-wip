#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Actual host/ARM Vulkan submissions through a recording, non-executing DRM shim."""
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean_environment():
    return {k:v for k,v in os.environ.items() if not k.upper().startswith(
        ('AGX_','ASAHI_','HK_','VK_','MESA_','LD_'))}
def run(root, out, workspace, arch):
    sysroot=workspace/'tmp/mini-sysroot'
    suite=json.loads((root/'ARMS.json').read_text());rows=suite['arms'];results=[]
    lib=root/('lib-'+arch);icd=lib/'icd.json'
    icd.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{
        'library_path':str(lib/'libvulkan_asahi.so'),'api_version':'1.4.305'}})+'\n')
    for arm in rows:
        for stage in ['vert','mesh']:
            d=out/arch/(arm['name']+'.'+stage);d.mkdir(parents=True,exist_ok=False)
            env=clean_environment()
            settings=dict(LD_PRELOAD=str(lib/'libasahi_host_trace_drm_shim.so'),
                VK_DRIVER_FILES=str(icd),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_MESH='1',
                MESA_SHADER_CACHE_DISABLE='true',ASAHI_HOST_TRACE_DIR=str(d),AGX_APPLE9_ASM='1',
                HK_APPLE9_TRACE_NIR='1')
            if suite.get('native_attributes'):settings['HK_APPLE9_NATIVE_MESH_ATTRIBUTES']='1'
            if suite.get('primitive_lowering'):settings['HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING']='1'
            cmd=[]
            if arch=='arm':
                settings['LD_LIBRARY_PATH']=str(sysroot/'usr/lib64')
                cmd=['qemu-aarch64-static','-L',str(sysroot)]
                for k,v in settings.items():cmd.extend(['-E',k+'='+v])
            else:env.update(settings)
            cmd += [str(lib/'probe'),str(root/'shaders'/(arm['name']+'.'+stage+'.spv')),
                str(root/'shaders'/(arm.get('fragments',{}).get(stage,arm['fragment'])+'.frag.spv')),
                str(int(arm['halfz'])),str(d/'color.bin'),str(d/'depth.bin'),'1']
            with (d/'log.txt').open('w') as log:
                rc=subprocess.run(cmd,env=env,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=45).returncode
            log=(d/'log.txt').read_text(errors='replace');commands=list(d.glob('*.commands.bin'))
            rejected=arm['pending_attributes'] and stage=='mesh'
            if rejected:
                ok=rc==2 and not commands and 'native mesh attributes development path is disabled' in log
            else:
                ok=rc==0 and commands and 'PIPELINE_OK' in log and 'submission=0 dry_run=1' in log
                # A no-op shim cannot produce a successful image result.
                ok=ok and all((d/f).read_bytes()==b'\xa5'*16384 for f in ['color.bin','depth.bin'])
            pruned=[]
            for f in d.glob('*.bin'):
                if f.stat().st_size >= 0x4000000:
                    pruned.append(dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)));f.unlink()
            row=dict(pruned_initializer_bos=pruned,arm=arm['name'],stage=stage,arch=arch,returncode=rc,host_check_pass=bool(ok),
                status='expected_attribute_rejection' if rejected else 'submission_prepared',
                no_gpu_execution=True,commands=[p.name for p in commands],
                driver_sha256=sha(lib/'libvulkan_asahi.so'),probe_sha256=sha(lib/'probe'),
                shim_sha256=sha(lib/'libasahi_host_trace_drm_shim.so'))
            (d/'RUN.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
            print(arch,arm['name'],stage,rc,len(commands),ok,flush=True)
    (out/(arch+'-results.json')).write_text(json.dumps(results,indent=2)+'\n')
    return all(x['host_check_pass'] for x in results)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--arch',choices=['host','arm'],required=True);a=p.parse_args()
    raise SystemExit(0 if run(a.root.resolve(),a.out.resolve(),a.workspace.resolve(),a.arch) else 1)
