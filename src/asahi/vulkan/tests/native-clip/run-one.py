#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Device-owner entry point. One process per draw; no installation or reset."""
import argparse, hashlib, json, os, platform, signal, subprocess, time
from pathlib import Path

def require_native(cmdline):
    values=[x.split('=',1)[1] for x in cmdline.split() if x.startswith('asahi_m4.native_mesh=')]
    if not values or values[-1]!='1':raise ValueError('boot must enable asahi_m4.native_mesh=1')
def manifest(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(root/name).resolve()
        if root.resolve() not in path.parents:raise ValueError('manifest path escaped kit')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('kit input hash mismatch: '+name)
def environment(icd):
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('AGX_','ASAHI_','HK_','VK_','MESA_','LD_'))}
    env.update(VK_DRIVER_FILES=str(icd),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_MESH='1',MESA_SHADER_CACHE_DISABLE='true')
    return env

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--arm',required=True);p.add_argument('--stage',choices=['vert','mesh'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--kernel-identity',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
    if platform.system()!='Linux' or platform.machine()!='aarch64':raise ValueError('requires Linux/aarch64 device')
    # Check boot and H1 qualification before constructing any native environment.
    cmdline=Path('/proc/cmdline').read_text().strip();require_native(cmdline)
    kernel=json.loads(a.kernel_identity.read_text());uname=os.uname()
    if kernel.get('release')!=uname.release or kernel.get('version')!=uname.version:raise ValueError('kernel identity mismatch')
    if not kernel.get('production_h1_qualified') or not kernel.get('source_commit') or not kernel.get('image_sha256') or not kernel.get('h1_report_sha256'):
        raise ValueError('production kernel H1 qualification and source/image/report hashes required')
    manifest(root)
    arm=next(x for x in json.loads((root/'ARMS.json').read_text())['arms'] if x['name']==a.arm)
    if arm['pending_attributes'] and a.stage=='mesh':
        raise ValueError('pending native attributes: this position-only compiler intentionally rejects the smooth mesh arm')
    out.mkdir(parents=True,exist_ok=False);icd=out/'icd.json'
    icd.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{
        'library_path':str(root/'lib-arm/libvulkan_asahi.so'),'api_version':'1.4.305'}})+'\n')
    cmd=[str(root/'lib-arm/probe'),str(root/'shaders'/(arm['name']+'.'+a.stage+'.spv')),
         str(root/'shaders'/(arm['fragment']+'.frag.spv')),str(int(arm['halfz'])),
         str(out/'color.bin'),str(out/'depth.bin'),'0']
    with (out/'log.txt').open('w') as log:
        process=subprocess.Popen(cmd,env=environment(icd),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        meta=dict(arm=arm['name'],stage=a.stage,pid=process.pid,no_gpu_execution=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline=cmdline,
            uname=list(uname),kernel=kernel,command=cmd,started=time.time(),
            driver_sha256=hashlib.sha256((root/'lib-arm/libvulkan_asahi.so').read_bytes()).hexdigest())
        (out/'RUN.json').write_text(json.dumps(meta,indent=2)+'\n')
        try:rc=process.wait(timeout=20)
        except subprocess.TimeoutExpired:os.kill(process.pid,signal.SIGSTOP);rc=None
    log=(out/'log.txt').read_text();meta.update(returncode=rc,ended=time.time(),
        fence_success=rc==0 and 'submission=0 dry_run=0' in log,backing_retained=rc is None)
    (out/'RUN.json').write_text(json.dumps(meta,indent=2)+'\n')
    if rc is None:
        print('INCOMPLETE: stopped PID',process.pid,'retains backing; collect evidence and use controller reset before terminating it.');return 2
    if rc!=0:return 2
    print('Readbacks saved. Run analyze.py on the VS/native pair; fence success alone is not a pixel/depth pass.')
    return 0
if __name__=='__main__':raise SystemExit(main())
