#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Use Mesa's actual native declaration gate, with explicit corpus denominators."""
import argparse, collections, hashlib, json, os, struct, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('--gate',type=Path,required=True)
p.add_argument('--out',type=Path,help='Separate report directory; preserve the pinned corpus baseline')
p.add_argument('--baseline',type=Path,help='Previous COVERAGE.json with the identical entries')
p.add_argument('--qualified',action='store_true',help='Inspect explicitly enabled qualified policy rather than shipping defaults')
a=p.parse_args();base=a.corpus.resolve();gate=a.gate.resolve();out=a.out.resolve() if a.out else base
if a.out:out.mkdir()
rows=[]
for item in json.loads((base/'SOURCE-MANIFEST.json').read_text()):
 if item['path'].endswith(('.mesh.spv','.task.spv')):
  rows.append(dict(repository=item['repository'],path='sources/'+item['repository'].replace('/','-')+'/'+item['path'],sha256=item['sha256'],configuration='upstream precompiled; specialization defaults'))
for item in json.loads((base/'BUILD-MANIFEST.json').read_text()):
 if item['returncode']==0:rows.append({k:item[k] for k in ['repository','path','sha256','configuration']})

def entries(data):
 w=struct.unpack('<%dI'%(len(data)//4),data);assert w[0]==0x07230203
 i=5;out=[]
 while i<len(w):
  n=w[i]>>16;op=w[i]&65535;assert n and i+n<=len(w)
  if op==15 and w[i+1] in [5267,5268,5364,5365]:
   name=struct.pack('<%dI'%(n-3),*w[i+3:i+n]).split(b'\0')[0].decode();out.append(('mesh' if w[i+1] in [5268,5365] else 'task',name))
  i+=n
 return out

results=[];env={k:v for k,v in os.environ.items() if not k.startswith(('HK_','AGX_','ASAHI_'))};reports=out/'gate';reports.mkdir(exist_ok=True)
if a.qualified:env['HK_APPLE9_NATIVE_INDEX']='1'
for item in rows:
 data=(base/item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
 for stage,name in entries(data):
  row=dict(item,stage=stage,entry=name,category='engine' if item['repository']=='turanszkij/WickedEngine' else 'demo')
  cmd=[str(gate),stage,name,str(base/item['path'])];r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
  ordinal=len(results);(reports/f'{ordinal:03d}.stdout').write_text(r.stdout);(reports/f'{ordinal:03d}.stderr').write_text(r.stderr)
  row.update(gate_returncode=r.returncode,stdout=f'gate/{ordinal:03d}.stdout',stderr=f'gate/{ordinal:03d}.stderr')
  if r.returncode==0:
   m=json.loads(r.stdout);row['nir']=m;reasons=[]
   if stage=='task':reasons.append('api_task_stage')
   else:
    if not m.get('qualified_policy_enabled',True):reasons.append('native_index_opt_in_required')
    if not m['class_match']:reasons.append('unqualified_output_class')
    if not m.get('workgroup_match',m['workgroup']==[m['physical_threads'],1,1]):reasons.append('unqualified_workgroup')
    if not m['triangles']:reasons.append('non_triangle_topology')
    if m['payload_bytes']:reasons.append('task_payload')
    if m['shared_bytes']:reasons.append('application_shared_memory')
    if m['outputs_extra']:reasons.append('non_position_outputs')
    if m['outputs_16bit']:reasons.append('16bit_outputs')
    if m['per_primitive_extra']:reasons.append('per_primitive_outputs')
    if m['variable_workgroup']:reasons.append('variable_workgroup')
    if m['api_subgroup_size'] not in [0,32]:reasons.append('subgroup_size')
   assert bool(reasons)==(not m['gate']),row
   row.update(reasons=reasons,primary_reason=reasons[0] if reasons else None,status='REJECTED' if reasons else 'NEEDS_PIPELINE_COMPILE')
  else:row.update(status='UNANALYZED',reasons=[])
  results.append(row)
# Count binary+entry once per category; report variants/project families separately.
unique={}
for r in results:unique.setdefault((r['category'],r['sha256'],r['entry'],r['stage']),r)
summary=[]
for category in ['demo','engine']:
 for stage in ['mesh','task']:
  rs=[r for r in unique.values() if r['category']==category and r['stage']==stage]
  summary.append(dict(category=category,stage=stage,distinct_entries=len(rs),input_entries=sum(r['category']==category and r['stage']==stage for r in results),projects=len({r['repository'] for r in rs}),rejected=sum(r['status']=='REJECTED' for r in rs),potential=sum(r['status']=='NEEDS_PIPELINE_COMPILE' for r in rs),unanalyzed=sum(r['status']=='UNANALYZED' for r in rs),primary_reasons=dict(collections.Counter(r.get('primary_reason') for r in rs)),overlapping_reasons=dict(collections.Counter(reason for r in rs for reason in r['reasons']))))
result=dict(scope=('explicit qualified native opt-in' if a.qualified else 'shipping defaults') + ', assuming qualified G16G VM and eligible draw; specialization defaults; static rejection proofs, not GPU runs',mesa_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gate_binary_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),summary=summary,entries=results)
result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
if a.baseline:
 old=json.loads(a.baseline.read_text())
 def key(r):return (r['category'],r['sha256'],r['entry'],r['stage'])
 previous={key(r):r for r in old['entries']};current={key(r):r for r in results}
 assert previous.keys()==current.keys(), 'corpus denominator changed'
 mesh=[k for k in current if k[3]=='mesh']
 result['delta']=dict(baseline=str(a.baseline.resolve()),baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
  mesh_denominator=len(mesh),before=sum(previous[k]['nir']['gate'] for k in mesh),after=sum(current[k]['nir']['gate'] for k in mesh),
  changed=[dict(sha256=k[1],repository=current[k]['repository'],path=current[k]['path'],before=previous[k]['status'],after=current[k]['status'],remaining_reasons=current[k]['reasons'])
           for k in mesh if previous[k]['status']!=current[k]['status'] or previous[k]['reasons']!=current[k]['reasons']],
  meaning='declaration candidates only; normal compilation, upload, task pairing and device completion tracked separately')
(out/'COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,delta=result.get('delta')),indent=2))
