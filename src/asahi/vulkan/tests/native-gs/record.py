#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record a real GS draw with a nonexecuting AArch64 DRM shim. Never a GPU pass."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for k in ['driver','shim','sysroot','probes','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--native',choices=['unset','0','1'],required=True)
p.add_argument('--vertices',type=int,choices=[3,4],default=4)
p.add_argument('--sequence',default='N')
p.add_argument('--mode',default='plain',choices=['plain','indexed','instances','offset','indirect','timestamp','merged','deferred'])
p.add_argument('--emitted',type=int)
p.add_argument('--end-mask',type=int,default=0)
p.add_argument('--cache-dir',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);trace=a.output/'trace';trace.mkdir()
icd=a.output/'icd.json';icd.write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.driver.resolve()),api_version='1.4.305'))))
e=dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'),LD_PRELOAD=str(a.shim.resolve()),ASAHI_HOST_TRACE_DIR=str(trace.resolve()),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_MESH='1',HK_APPLE9_NATIVE_MESH_PACKED_SHAPES='T16V3P1,T16V4P2',HK_APPLE9_GS_ROUTE_LOG='1',VK_DRIVER_FILES=str(icd.resolve()),MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false',VK_LOADER_LAYERS_DISABLE='~implicit~')
if a.native!='unset':e['HK_APPLE9_GS_NATIVE']=a.native
if a.cache_dir:
 e['MESA_SHADER_CACHE_DISABLE']='false';e['MESA_SHADER_CACHE_DIR']=str(a.cache_dir.resolve())
files=[a.probes.resolve()/f for f in ['transitions',f'V{a.vertices}.geom.spv','V5.geom.spv','input.vert.spv','blue.frag.spv','white.frag.spv']]
cmd=['qemu-aarch64-static','-L',str(a.sysroot.resolve()),*[x for k,v in e.items() if ',' not in v for x in ['-E',k+'='+v]],*map(str,files),str(a.output.resolve()),a.sequence,a.mode,str(a.emitted if a.emitted is not None else a.vertices),str(a.end_mask)]
meta=dict(environment=e,gpu_execution=False,queue_submission=True,native=a.native,vertices=a.vertices,sequence=a.sequence,mode=a.mode,argv=cmd,input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.driver,a.shim,*files]})
(a.output/'RUN.json').write_text(json.dumps(meta,indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.upper().startswith(('LD_','HK_','VK_','MESA_','AGX_','ASAHI_'))}
clean.update({k:v for k,v in e.items() if ',' in v})
with (a.output/'stdout.log').open('w') as out:r=subprocess.run(cmd,env=clean,stdout=out,stderr=subprocess.STDOUT,timeout=180)
log=(a.output/'stdout.log').read_text()
assert r.returncode==1 and 'SUBMIT_OK' in log and 'Vulkan:' not in log,log
assert (a.output/'pixels.bin').read_bytes()==bytes([0xa5])*64*64*4
routes=[int(x) for x in re.findall(r'HK_GS_ROUTE native=(\d+)',log)]
expected=[int(k=='N' and a.native=='1' and a.mode in ['plain','timestamp','merged','deferred']) for k in a.sequence if k in 'NE']
assert routes==expected,(routes,expected,log)
print('Host record PASS: routes',routes,'; readback untouched, no GPU execution.')
