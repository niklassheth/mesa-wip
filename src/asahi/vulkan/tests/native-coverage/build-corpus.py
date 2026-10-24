#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile pinned, unmodified public shaders; retain exact commands and logs."""
import argparse, hashlib, json, os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('corpus',type=Path);p.add_argument('--dxc',type=Path,required=True)
a=p.parse_args();base=a.corpus.resolve();source=base/'sources';out=base/'compiled';out.mkdir(exist_ok=True)
manifest=json.loads((base/'SOURCE-MANIFEST.json').read_text())
for f in manifest:
 assert hashlib.sha256((source/f['repository'].replace('/','-')/f['path']).read_bytes()).hexdigest()==f['sha256']
rows=[];env=os.environ.copy();env['LD_LIBRARY_PATH']=str(a.dxc.resolve().parent.parent/'lib')
for exe in ['glslangValidator',str(a.dxc.resolve())]:
 v=subprocess.run([exe,'--version'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/(Path(exe).name+'-version.txt')).write_text(v.stdout);assert v.returncode==0

def run(repo,src,stage,args,label=None,config='upstream defaults'):
 name=repo.replace('/','-')+'--'+(label or Path(src).name)
 dest=out/(name+'.spv');cmd=args+['-o' if args[0]=='glslangValidator' else '-Fo',str(dest),str(source/repo.replace('/','-')/src)]
 r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (out/(name+'.log')).write_text(r.stdout)
 row=dict(repository=repo,source=src,stage=stage,configuration=config,argv=cmd,returncode=r.returncode,log=str((out/(name+'.log')).relative_to(base)))
 if r.returncode==0:
  row.update(path=str(dest.relative_to(base)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
 rows.append(row)

for repo,stage in [('zeux/niagara','mesh'),('zeux/niagara','task')]:
 run(repo,'src/shaders/meshlet.'+stage+'.glsl',stage,['glslangValidator','--target-env','vulkan1.3','-S',stage])
# Public CAD demo, default meshlet limits, 32-wide device preference; both API
# taskless and task-backed app variants. This is not a capture of a game.
repo='nvpro-samples/gl_vk_meshlet_cadscene'
defines={'NVMESHLET_VERTEX_COUNT':64,'NVMESHLET_PRIMITIVE_COUNT':126,'NVMESHLET_ENCODING':1,'NVMESHLET_PER_TASK':32,'VERTEX_EXTRAS_COUNT':0,'USE_VERTEX_CULL':1,'USE_BARYCENTRIC_SHADING':0,'USE_BARYCENTRIC_SHADING_QUADSHUFFLE':0,'USE_BACKFACECULL':1,'USE_CLIPPING':0,'USE_STATS':0,'SHOW_PRIMIDS':0,'SHOW_BOX':0,'SHOW_NORMAL':0,'SHOW_CULLED':0,'EXT_LOCAL_INVOCATION_VERTEX_OUTPUT':0,'EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT':0,'EXT_COMPACT_PRIMITIVE_OUTPUT':0,'EXT_COMPACT_VERTEX_OUTPUT':0,'EXT_MAX_MESH_WORKGROUP_INVOCATIONS':32,'EXT_MAX_TASK_WORKGROUP_INVOCATIONS':32,'EXT_SUBGROUP_OPTIMIZATION':1,'EXT_MESH_SUBGROUP_SIZE':32,'EXT_TASK_SUBGROUP_SIZE':32,'EXT_MESH_SUBGROUP_COUNT':1,'EXT_TASK_SUBGROUP_COUNT':1}
for name in ['basic','cull']:
 for task in [0,1]:
  run(repo,'drawmeshlet_ext_'+name+'.mesh.glsl','mesh',['glslangValidator','--target-env','vulkan1.3','-S','mesh']+[f'-D{k}={v}' for k,v in defines.items()]+[f'-DUSE_TASK_STAGE={task}'],label=f'{name}-task{task}',config='app default V64/P126; subgroup and preferred workgroup 32; task='+str(task))
run(repo,'drawmeshlet_ext.task.glsl','task',['glslangValidator','--target-env','vulkan1.3','-S','task']+[f'-D{k}={v}' for k,v in defines.items()],config='app default V64/P126; subgroup and preferred workgroup 32')
repo='turanszkij/WickedEngine'
dxc=[str(a.dxc.resolve()),'-spirv','-fspv-target-env=vulkan1.3','-fvk-use-dx-layout','-fvk-use-dx-position-w','-Wno-conversion','-E','main']
for kind,shift in [('b',0),('t',1000),('u',2000),('s',3000)]:dxc+=['-fvk-'+kind+'-shift',str(shift),'0']
for i,name in enumerate(['SAMPLER','STORAGE_BUFFER','UNIFORM_TEXEL_BUFFER','SAMPLED_IMAGE','STORAGE_IMAGE','STORAGE_TEXEL_BUFFER','ACCELERATION_STRUCTURE'],1): dxc+=['-D',f'DESCRIPTOR_SET_BINDLESS_{name}={i}']
for name in ['objectMS','objectMS_prepass','objectMS_prepass_alphatest','objectMS_simple','objectAS']:
 stage='task' if name=='objectAS' else 'mesh'
 run(repo,'WickedEngine/shaders/'+name+'.hlsl',stage,dxc+['-T','as_6_5' if stage=='task' else 'ms_6_5'],config='engine Vulkan compiler flags and binding shifts; base permutation; SM6.5; validation enabled')
(base/'BUILD-MANIFEST.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{'source':x['source'],'returncode':x['returncode'],'log':x['log']} for x in rows],indent=2))
