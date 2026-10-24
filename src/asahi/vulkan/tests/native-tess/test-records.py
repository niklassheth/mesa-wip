#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Corrupt real host packets in memory; each dependency/ABI defect must fail."""
import argparse,importlib.util,json,struct
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
spec=importlib.util.spec_from_file_location('checker',Path(__file__).with_name('check-records.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
o=json.loads((a.directory/'OFFSETS.json').read_text())
case=dict(case='core-triangles-cw-equal_spacing-P1-I1-O1')
base=m.check(case,a.directory,o)
render=next(x for x in base['commands'] if x.get('native'))
producer=base['commands'][base['commands'].index(render)-1]
read=m.c.capture.read_gpu;path_read=Path.read_bytes
results=[]
def reject(name,context,reason):
 with context:
  try:m.check(case,a.directory,o)
  except ValueError as e:
   assert reason in str(e),(name,str(e),reason)
  else:raise AssertionError(name+' was accepted')
 results.append(dict(fault=name,rejected=True))
def gpu_fault(address,mask):
 def changed(d,index,addr,size):
  value=read(d,index,addr,size)
  if addr<=address and address+4<=addr+size:
   value=bytearray(value);off=address-addr
   struct.pack_into('<I',value,off,struct.unpack_from('<I',value,off)[0]^mask)
  return bytes(value)
 return patch.object(m.c.capture,'read_gpu',changed)
reject('object count pointer names another word',gpu_fault(render['object']['args']+o['draw_uniform_offset'],4),'object does not read')
# The first PPP address follows fourteen state words and the PPP header.
trace=a.directory/('host-'+case['case'])/'trace'
body=path_read(trace/(render['submission']+'.commands.bin'))[8:]
vdm=struct.unpack_from('<Q',body,8)[0]
words=struct.unpack('<40I',read(trace,render['submission'],vdm,160))
raster=0x1000000000+words[15]+0xb0
reject('native winding inverted twice',gpu_fault(raster,1<<16),'winding applied twice')
reject('producer and consumer patch metadata differ',gpu_fault(render['mesh']['tess']+o['tess_output_patch'],1),'wrong patch metadata')
def missing_join(path):
 value=path_read(path)
 if path.name==render['submission']+'.commands.bin':
  value=bytearray(value);struct.pack_into('<H',value,6,65535)
 return bytes(value)
reject('consumer drops populated compute join',patch.object(Path,'read_bytes',missing_join),'lost dependency on populated subqueue 1')
print(json.dumps(dict(result='pass',gpu_execution=False,negative_controls=results),indent=2))
