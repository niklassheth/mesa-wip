#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from arms import generate
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--driver-commit',default=None)
p.add_argument('--host-build',type=Path,required=True);p.add_argument('--arm-build',type=Path,required=True);p.add_argument('--trace-base',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();here=Path(__file__).resolve().parent;shared=here.parent/'native-clip';mesa=here.parents[4]
generate(root);sysroot=a.workspace/'tmp/mini-sysroot'
for arch,build in [('host',a.host_build),('arm',a.arm_build)]:
    dst=root/('lib-'+arch);dst.mkdir(exist_ok=True)
    for name,source in [('libvulkan_asahi.so',build/'src/asahi/vulkan/libvulkan_asahi.so'),('libasahi_host_trace_drm_shim.so',a.trace_base/('lib-'+arch)/'libasahi_host_trace_drm_shim.so')]:shutil.copy2(source,dst/name)
    cc=['cc'] if arch=='host' else ['aarch64-linux-gnu-gcc','--sysroot='+str(sysroot),'-L'+str(sysroot/'usr/lib64'),'-Wl,-rpath-link,'+str(sysroot/'usr/lib64')]
    subprocess.run(cc+['-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-I'+str(mesa/'include'),str(shared/'probe.c'),'-lvulkan','-o',str(dst/'probe')],check=True)
    (dst/'icd.json').unlink(missing_ok=True)
for name in ['analyze.py','run-one.py','run-host.py','probe.c']:shutil.copy2(shared/name,root/name)
shutil.copy2(here/'selftest.py',root/'selftest.py')
for d in ['native-primitive','native-attributes']:
    dst=root/'source'/d;dst.mkdir(parents=True,exist_ok=True);shutil.copy2(here.parent/d/'arms.py',dst/'arms.py')
commit=subprocess.check_output(['git','-C',str(mesa),'rev-parse','HEAD'],text=True).strip()
(root/'PROVENANCE.json').write_text(json.dumps(dict(source_commit=a.driver_commit or commit,packaging_commit=commit,no_gpu_execution=True,
    native_primitive_link='unresolved; kit exercises flat broadcast only',
    compiler_gates=['HK_APPLE9_NATIVE_MESH=1','HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1','HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING=1'],
    device_status='pending; separate paired VS/native pixel and depth comparison required'),indent=2)+'\n')
(root/'kernel-identity.example.json').write_text(json.dumps(dict(release='FILL_FROM_QUALIFIED_KERNEL',version='FILL_FROM_QUALIFIED_KERNEL',source_commit='',image_sha256='',production_h1_qualified=False,h1_report_sha256='',native_mesh_attributes_supported=False),indent=2)+'\n')
(root/'README.txt').write_text('Nine paired VS/native arms. Use run-one.py once per arm/stage with the matching kernel identity and H1 qualification, then analyze.py on the pair. No reset, install or GPU run has been performed by preparation. Cull tests use filled triangles; CULL_UNTAKEN writes true only when WorkGroupID.x is nonzero, which is false for this one-workgroup draw. Native primitive-record linking, layer, viewport and shading-rate output remain unsupported. The VS primitive-ID control uses an ordinary flat user input. Both halves have independent exact RGBA8/D32 oracles. Regenerate shaders with source/native-primitive/arms.py.\n')
paths=sorted(x for x in root.rglob('*') if x.is_file() and x.name not in ['SHA256SUMS','icd.json'] and '__pycache__' not in x.parts)
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x.relative_to(root))+'\n' for x in paths))
print('prepared',len(paths),'inputs from',commit)
