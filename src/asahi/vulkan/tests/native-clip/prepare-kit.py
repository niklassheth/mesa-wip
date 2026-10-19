#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Package this checkout's ordinary compiled drivers, probe, shaders and oracles."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
from arms import generate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--host-build',type=Path,required=True)
    p.add_argument('--arm-build',type=Path,required=True);p.add_argument('--trace-base',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();here=Path(__file__).resolve().parent;mesa=here.parents[4]
    sysroot=a.workspace/'tmp/mini-sysroot';generate(root)
    for arch,build in [('host',a.host_build),('arm',a.arm_build)]:
        dst=root/('lib-'+arch);dst.mkdir(exist_ok=True)
        shutil.copy2(build/'src/asahi/vulkan/libvulkan_asahi.so',dst/'libvulkan_asahi.so')
        shutil.copy2(a.trace_base/('lib-'+arch)/'libasahi_host_trace_drm_shim.so',dst/'libasahi_host_trace_drm_shim.so')
        cc=['cc'] if arch=='host' else ['aarch64-linux-gnu-gcc','--sysroot='+str(sysroot),
            '-L'+str(sysroot/'usr/lib64'),'-Wl,-rpath-link,'+str(sysroot/'usr/lib64')]
        subprocess.run(cc+['-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers',
            '-I'+str(mesa/'include'),str(here/'probe.c'),'-lvulkan','-o',str(dst/'probe')],check=True)
        # The host runner writes its absolute local ICD at run time. A device
        # invocation writes its own in the result directory.
        (dst/'icd.json').unlink(missing_ok=True)
    for name in ['arms.py','analyze.py','run-one.py','run-host.py','selftest.py','probe.c','README.md']:
        shutil.copy2(here/name,root/name)
    commit=subprocess.check_output(['git','-C',str(mesa),'rev-parse','HEAD'],text=True).strip()
    (root/'PROVENANCE.json').write_text(json.dumps(dict(source_commit=commit,base_commit='6388f1020cff2a5db2366cda573adf38df027266',
        no_gpu_execution=True,hardware_reference='CLIP-20261019: 36/36; position-only; generic attributes unqualified',
        compiler_gate='HK_APPLE9_NATIVE_MESH=1',kernel_prerequisite='production H1 qualified; asahi_m4.native_mesh=1'),indent=2)+'\n')
    (root/'kernel-identity.example.json').write_text(json.dumps(dict(release='FILL_FROM_QUALIFIED_KERNEL',version='FILL_FROM_QUALIFIED_KERNEL',
        source_commit='',image_sha256='',production_h1_qualified=False,h1_report_sha256=''),indent=2)+'\n')
    paths=sorted(x for x in root.rglob('*') if x.is_file() and x.name not in ['SHA256SUMS','icd.json'])
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x.relative_to(root))+'\n' for x in paths))
    print('prepared',len(paths),'hashed inputs from',commit)
if __name__=='__main__':main()
