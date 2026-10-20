#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from arms import generate
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--host-build',type=Path,required=True);p.add_argument('--arm-build',type=Path,required=True);p.add_argument('--trace-base',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();here=Path(__file__).resolve().parent;mesa=here.parents[4]
sources=['src/asahi/compiler/agx_nir_apple9_mesh_native.c','src/asahi/compiler/agx_nir_lower_apple9_mesh.c',
    'src/asahi/vulkan/hk_apple9_mesh.c','src/asahi/vulkan/hk_cmd_apple9.c','src/asahi/vulkan/hk_physical_device.c']
assert subprocess.check_output(['git','-C',str(mesa),'diff','--',*sources]) == (here/'research-overlay.patch').read_bytes(), 'apply exactly the research overlay before building/preparing'
generate(root);sysroot=a.workspace/'tmp/mini-sysroot'
for arch,build in [('host',a.host_build),('arm',a.arm_build)]:
    dst=root/('lib-'+arch);dst.mkdir(exist_ok=True)
    for name,source in [('libvulkan_asahi.so',build/'src/asahi/vulkan/libvulkan_asahi.so'),('libasahi_host_trace_drm_shim.so',a.trace_base/('lib-'+arch)/'libasahi_host_trace_drm_shim.so')]:shutil.copy2(source,dst/name)
    cc=['cc'] if arch=='host' else ['aarch64-linux-gnu-gcc','--sysroot='+str(sysroot),'-L'+str(sysroot/'usr/lib64'),'-Wl,-rpath-link,'+str(sysroot/'usr/lib64')]
    subprocess.run(cc+['-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-I'+str(mesa/'include'),str(here/'probe.c'),'-lvulkan','-o',str(dst/'probe')],check=True)
    (dst/'icd.json').unlink(missing_ok=True)
for name in ['README.md','selftest.py','analyze.py','run-one.py','run-host.py','probe.c','research-overlay.patch']:shutil.copy2(here/name,root/name)
for d in ['native-limits','native-attributes']:
    dst=root/'source'/d;dst.mkdir(parents=True,exist_ok=True);shutil.copy2(here.parent/d/'arms.py',dst/'arms.py')
(root/'PROVENANCE.json').write_text(json.dumps(dict(base_commit='98377944de480f9885e22eaefc6d7666e000e592',
    packaging_commit=subprocess.check_output(['git','-C',str(mesa),'rev-parse','HEAD'],text=True).strip(),
    research_overlay_sha256=hashlib.sha256((here/'research-overlay.patch').read_bytes()).hexdigest(),
    no_gpu_execution=True,physical_mesh_threads=32,physical_object_threads=16,
    mesh_launch_frame_bytes=64,object_launch_frame_bytes=16,
    warnings=['T32 launch and frame assumptions are unqualified','Shared GLSL memory remains rejected','All other native V3/P1 restrictions retained']),indent=2)+'\n')
(root/'kernel-identity.example.json').write_text(json.dumps(dict(release='FILL_FROM_QUALIFIED_KERNEL',version='FILL_FROM_QUALIFIED_KERNEL',source_commit='',image_sha256='',production_h1_qualified=False,h1_report_sha256='',native_mesh_physical_threads=32),indent=2)+'\n')
paths=sorted(x for x in root.rglob('*') if x.is_file() and x.name not in ['SHA256SUMS','icd.json'] and '__pycache__' not in x.parts)
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x.relative_to(root))+'\n' for x in paths))
print('prepared',len(paths),'research inputs')
