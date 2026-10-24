#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record a real TES draw with a nonexecuting AArch64 DRM shim. Never a GPU pass."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for k in ['driver','shim','sysroot','probes','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--native',choices=['unset','0','1'],required=True)
p.add_argument('--patch',type=int,choices=range(1,33),default=3)
p.add_argument('--domain',choices=['triangles','quads'],default='triangles')
p.add_argument('--order',choices=['cw','ccw'],default='cw')
p.add_argument('--spacing',choices=['equal_spacing','fractional_odd_spacing','fractional_even_spacing'],default='equal_spacing')
p.add_argument('--inner',type=float,default=1)
p.add_argument('--outer',type=float,default=1)
p.add_argument('--sequence',default='N')
p.add_argument('--mode',default='plain',choices=['plain','indexed','instances','offset','indirect','conditional','conservative','msaa','multiview','layers','occlusion','statistics','discard','nonfill','restart','firstinstance','multipatch','timestamp','merged','kept-merge','deferred'])
p.add_argument('--cache-dir',type=Path)
p.add_argument('--test-env',action='append',default=[])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);trace=a.output/'trace';trace.mkdir()
icd=a.output/'icd.json';icd.write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(a.driver.resolve()),api_version='1.4.305'))))
e=dict(LD_LIBRARY_PATH=str(a.sysroot.resolve()/'usr/lib64'),LD_PRELOAD=str(a.shim.resolve()),ASAHI_HOST_TRACE_DIR=str(trace.resolve()),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_NATIVE_MESH='1',HK_APPLE9_NATIVE_MESH_PACKED_SHAPES='T16V3P1',HK_APPLE9_TESS_ROUTE_LOG='1',VK_DRIVER_FILES=str(icd.resolve()),MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false',VK_LOADER_LAYERS_DISABLE='~implicit~')
for setting in a.test_env:
 k,v=setting.split('=',1);assert k.startswith('TESS_TEST_');e[k]=v
if a.native!='unset':e['HK_APPLE9_TESS_NATIVE']=a.native
if a.cache_dir:
 e['MESA_SHADER_CACHE_DISABLE']='false';e['MESA_SHADER_CACHE_DIR']=str(a.cache_dir.resolve())
name=f'{a.domain}-{a.order}-{a.spacing}'
e.update(TESS_TEST_PATCH_SIZE=str(a.patch),TESS_TEST_DOMAIN=a.domain,TESS_TEST_ORDER=a.order)
tes=f'P{a.patch}-last.tese.spv' if e.get('TESS_TEST_STATIC_LAST')=='1' else name+'.tese.spv'
if e.get('TESS_TEST_STATIC_LAST')=='1':assert (a.domain,a.order,a.spacing)==('triangles','cw','equal_spacing')
files=[a.probes.resolve()/f for f in ['transitions',tes,name+'-fallback.tese.spv','input.vert.spv','blue.frag.spv','white.frag.spv',f'P{a.patch}.tesc.spv']]
cmd=['qemu-aarch64-static','-L',str(a.sysroot.resolve()),*[x for k,v in e.items() if ',' not in v for x in ['-E',k+'='+v]],*map(str,files),str(a.output.resolve()),a.sequence,a.mode,str(a.inner),str(a.outer)]
meta=dict(environment=e,gpu_execution=False,queue_submission=True,native=a.native,patch=a.patch,domain=a.domain,order=a.order,spacing=a.spacing,inner=a.inner,outer=a.outer,sequence=a.sequence,mode=a.mode,argv=cmd,input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.driver,a.shim,*files]})
(a.output/'RUN.json').write_text(json.dumps(meta,indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.upper().startswith(('LD_','HK_','VK_','MESA_','AGX_','ASAHI_'))}
clean.update({k:v for k,v in e.items() if ',' in v})
with (a.output/'stdout.log').open('w') as out:r=subprocess.run(cmd,env=clean,stdout=out,stderr=subprocess.STDOUT,timeout=180)
log=(a.output/'stdout.log').read_text()
if r.returncode==77:
 assert 'SKIP required' in log and 'SUBMIT_OK' not in log,log
 (a.output/'UNSUPPORTED.json').write_text(json.dumps(dict(status='NotSupported',reason=log.split('SKIP required',1)[1].strip()))+'\n')
 print('Host preflight NotSupported: no draw recorded; not fallback qualification.');raise SystemExit(0)
assert r.returncode==(0 if a.mode in ['nonfill','conservative'] else 1) and 'SUBMIT_OK' in log and 'Vulkan:' not in log,log
assert (a.output/'pixels.bin').read_bytes()==bytes([0xa5])*64*64*4*(2 if a.mode in ['multiview','layers'] else 1)
routes=[int(x) for x in re.findall(r'HK_TESS_ROUTE native=(\d+)',log)]
expected=[int(k=='N' and a.native=='1' and a.mode in ['plain','timestamp','merged','kept-merge','deferred']) for k in a.sequence if k in 'NE']
assert routes==expected,(routes,expected,log)
if a.mode=='kept-merge':assert 'KEPT_MERGE_OK' in log,log
print('Host record PASS: routes',routes,'; readback untouched, no GPU execution.')
