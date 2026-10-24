#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce the default-off TES device plan without opening a device."""
import argparse, itertools, json
from pathlib import Path
DOMAINS=('triangles','quads')
ORDERS=('cw','ccw')
SPACINGS=('equal_spacing','fractional_odd_spacing','fractional_even_spacing')
def cases():
 rows=[]
 def add(name,**kw):
  row=dict(case=name,domain='triangles',order='cw',spacing='equal_spacing',patch=3,inner=1,outer=1,mode='plain',sequence='N',environment={})
  row.update(kw);rows.append(row)
 for d,o,s,p,(inner,outer) in itertools.product(DOMAINS,ORDERS,SPACINGS,range(1,33),((1,1),(1,64),(64,1),(64,64),(64,0))):
  add(f'core-{d}-{o}-{s}-P{p}-I{inner}-O{outer}',domain=d,order=o,spacing=s,patch=p,inner=inner,outer=outer,family='core')
 for patch in range(1,33):
  add(f'static-last-P{patch}',patch=patch,environment={'TESS_TEST_STATIC_LAST':'1'},family='static-input')
 # Both origin encodings, with the upper-left cases already in core.
 for d,o,s in itertools.product(DOMAINS,ORDERS,SPACINGS):
  for inner,outer in ((1,1),(64,64),(64,0)):
   add(f'origin-{d}-{o}-{s}-I{inner}-O{outer}',domain=d,order=o,spacing=s,inner=inner,outer=outer,environment={'TESS_TEST_LOWER_LEFT':'1'},family='origin')
  for level in (1,64):
   add(f'timing-{d}-{o}-{s}-L{level}',domain=d,order=o,spacing=s,inner=level,outer=level,mode='timestamp',family='timing')
 for d,o,origin,front,cull,level in itertools.product(DOMAINS,ORDERS,(0,1),('cw','ccw'),('front','back'),(1,64)):
  add(f'cull-{d}-{o}-origin{origin}-front{front}-{cull}-L{level}',domain=d,order=o,inner=level,outer=level,environment={'TESS_TEST_LOWER_LEFT':str(origin),'TESS_TEST_FRONT':front,'TESS_TEST_CULL':cull},family='cull')
 for mode in ('indexed','instances','offset','indirect','conditional','conservative','msaa','multiview','layers','occlusion','statistics','discard','nonfill','firstinstance','multipatch'):
  add(f'fallback-{mode}',mode=mode,family='fallback')
 add('fallback-tes-body',sequence='E',family='fallback')
 for mode,seq in [('plain','NENVNMNN'),('plain','EN'),('plain','NN'),('merged','NN'),('kept-merge','VN'),('deferred','NN')]:
  add(f'transition-{mode}-{seq}',mode=mode,sequence=seq,family='transition')
 for mode in ('plain','merged'):
  add(f'producer-rewrite-{mode}',mode=mode,sequence='NN',environment={'TESS_TEST_VS_REWRITE':'1'},family='transition')
 assert len({r['case'] for r in rows})==len(rows)
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=cases();counts={family:sum(r['family']==family for r in rows) for family in sorted({r['family'] for r in rows})}
 assert counts['core']==1920
 result=dict(schema=1,status='DRAFT_NOT_ADMITTED',default_off=True,device_execution=False,source_review_required=True,packet_review_required=True,repetitions_per_arm=20,arms=[{'HK_APPLE9_TESS_NATIVE':'0'},{'HK_APPLE9_TESS_NATIVE':'1'}],fresh_cache_per_process=True,case_count=len(rows),process_count=len(rows)*40,families=counts,cases=rows)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
