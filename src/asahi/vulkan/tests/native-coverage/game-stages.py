#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory graphics-pipeline stage uses in local Fossilize game caches.

Counts records (including graphics pipeline libraries), never frames or draws.
Zero-padded database tails are identified; malformed/truncated entries fail.
No shaders are disassembled and no proprietary implementation is inspected.
"""
import argparse, collections, hashlib, json, struct, zlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('database',type=Path);p.add_argument('--game',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();tags=collections.Counter();stages=collections.Counter();combos=collections.Counter();bad=[];h=hashlib.sha256();padding=0;pipelines=0;unparsed=0;tail_nonzero=False
STAGE={1:'VS',2:'TCS',4:'TES',8:'GS',16:'FS',32:'CS',64:'TASK',128:'MESH'}
with a.database.open('rb') as f:
 def read(n):
  b=f.read(n);h.update(b);return b
 header=read(16);assert len(header)==16 and header[1:12]==b'FOSSILIZEDB'
 while True:
  offset=f.tell();key=read(40)
  if not key:break
  if not any(key):
   padding=len(key)
   while b:=read(1024*1024):
    tail_nonzero |= any(b);padding+=len(b)
   if tail_nonzero:unparsed=padding
   break
  assert len(key)==40,'truncated key'
  tag=int(key[:24],16);header=read(16);assert len(header)==16,'truncated header'
  stored,flags,crc,size=struct.unpack('<IIII',header);data=read(stored);assert len(data)==stored,'truncated payload';tags[tag]+=1
  if tag!=6:continue
  try:
   assert flags in (0,1,2),flags
   raw=zlib.decompress(data) if flags==2 else data
   assert len(raw)==size,(len(raw),size)
   j=json.loads(raw)
   for pipeline in (j.get('graphicsPipelines') or {}).values():
    seq=sorted(STAGE.get(s.get('stage'),str(s.get('stage'))) for s in pipeline.get('stages',[]));stages.update(seq);combos['+'.join(seq)]+=1;pipelines+=1
  except Exception as e:bad.append({'offset':offset,'error':str(e)})
result=dict(game=a.game,path=str(a.database.resolve()),bytes=a.database.stat().st_size,sha256=h.hexdigest(),entries_by_tag=dict(tags),graphics_pipeline_records=pipelines,stage_uses=dict(stages),stage_combinations=dict(combos),bad_records=bad,zero_tail_bytes=padding if not tail_nonzero else None,unparsed_tail_bytes=unparsed,complete=not bad and not unparsed,mesh_stage_uses=stages['MESH'],task_stage_uses=stages['TASK'])
a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.game,'pipeline records',pipelines,'mesh',stages['MESH'],'task',stages['TASK'],'bad',len(bad),'padding',padding)
raise SystemExit(bool(bad))
