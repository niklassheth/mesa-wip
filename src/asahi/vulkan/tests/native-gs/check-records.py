#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check our bounded GS host recordings, never GPU execution or pixels."""
import argparse,importlib.util,json,struct
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
spec=importlib.util.spec_from_file_location('transition',Path(__file__).resolve().parents[1]/'native-mesh-hybrid/check-transitions.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
o=json.loads((a.directory/'OFFSETS.json').read_text())
require=c.require

def check(case):
 d=a.directory/('host-'+case['case'])/'trace';run=json.loads((d.parent/'RUN.json').read_text())
 require(run['gpu_execution'] is False and run['queue_submission'] is True,'wrong evidence scope')
 read=lambda index,addr,size:c.capture.read_gpu(d,index,addr,size)
 def u64(index,addr):return struct.unpack('<Q',read(index,addr,8))[0]
 def root(index,launch):
  header=read(index,launch,16);arg=c.literal32(header[:8],2)|(c.literal32(header[8:],3)<<32)
  resources=u64(index,arg+16);address=u64(index,resources)
  require(u64(index,address)==address,'missing descriptor self pointer')
  params=u64(index,address+o['vertex_params']);output=u64(index,params+o['params_output'])
  require(output and u64(index,params+o['params_mask'])==1,'software VS output allocation/mask missing')
  require(u64(index,address+o['vertex_outputs'])==1,'root output mask differs')
  require(not u64(index,address+o['geometry_params']),'emulated GS output root leaked into native draw')
  require(struct.unpack('<I',read(index,params+o['params_vertices'],4))[0]==3,'wrong input triangle size')
  require(struct.unpack('<6I',read(index,params+o['params_grid'],24))==(3,1,1,64,1,1),'wrong software VS grid')
  # Bounds and ownership only: the recorder never executes VS output stores.
  read(index,output,3*16)
  return dict(root=hex(address),params=hex(params),output=hex(output))
 commands=[];native=[];renders=[]
 for path in sorted(d.glob('*.commands.bin')):
  data=path.read_bytes();kind,size,vdm,cdm=struct.unpack_from('<4H',data);body=data[8:]
  require(len(body)==size,'one complete command per ioctl required')
  index=path.name.split('.')[0];addr=struct.unpack_from('<Q',body,8)[0]
  row=dict(submission=index,kind=kind,vdm_barrier=vdm,cdm_barrier=cdm)
  require((vdm,cdm)==(0,0),'GS/internal reads must retain full subqueue dependencies')
  if kind==1:
   require(size==64,'compute size');end=struct.unpack_from('<Q',body,16)[0]
   row['first_segment_bytes']=end-addr
   # Follow our CDM stream links: timestamp reset can exhaust a segment
   # before the software VS is appended. Only owned packet formats occur.
   cursor=addr;visited=set();dispatches=[];blob=bytearray();links=0
   for step in range(1024):
    require(cursor not in visited, 'cyclic CDM stream');visited.add(cursor)
    word=struct.unpack('<I',read(index,cursor,4))[0]
    if word==0x00880000:
     packet=read(index,cursor,40);w=struct.unpack('<10I',packet)
     require(w[1]==0x01000040,'unknown direct CDM format')
     dispatches.append(dict(global_size=list(w[4:7]),local_size=list(w[7:10]),launch=(1<<40)+w[2]*64))
    elif word in (0x08880000,0x10880000):
     packet=read(index,cursor,36 if word==0x08880000 else 24)
    elif word in (0x60000160,0x600fffff):
     packet=read(index,cursor,4)
    elif word==0x40000000:
     blob.extend(read(index,cursor,8));break
    elif word&0xff000000==0x20000000:
     require(links<8,'too many CDM links');links+=1
     low=struct.unpack('<I',read(index,cursor+4,4))[0]
     cursor=((word&0xffffff)<<32)|low;continue
    else:
     raise ValueError(f'{case["case"]}: unexpected CDM header {word:#x}')
    blob.extend(packet);cursor+=len(packet)
   else:raise ValueError('missing CDM stop')
   row['stream_bytes']=len(blob);row['links']=links
   row['dispatches']=dispatches;row['_blob']=blob
  else:
   require(kind==0 and size==240,'render kind/size')
   prefix=struct.unpack('<I',read(index,addr,4))[0];row['native']=prefix==0x40000d2c
   require(prefix in (0x40000d2c,0x4000002e),'unexpected producer packet')
   row['attachment_programs']=[hex(x) for x in struct.unpack_from('<4Q',body,168)]
   require(all(int(x,16) for x in row['attachment_programs']),'attachment program missing')
   if row['native']:
    require(commands and commands[-1]['kind']==1,'missing VS producer before native render')
    candidates=[r for r in commands if r['kind']==1 and r['dispatches'] and r['dispatches'][-1]['global_size']==[3,1,1] and r['dispatches'][-1]['local_size']==[64,1,1]]
    require(candidates, f'{case["case"]}: missing software VS dispatch')
    previous=candidates[-1];ds=previous['dispatches']
    intervening=commands[commands.index(previous)+1:]
    require(not intervening or (run['mode']=='timestamp' and all(r['kind']==1 for r in intervening)), 'unexpected work between software VS and render')
    # Timestamp reset may be submitted between pre_gfx and render. Every
    # recorded command above retains both joins, so the VS wait is transitive.
    require(previous['_blob'].endswith(struct.pack('<4I',0x60000160,0x600fffff,0x40000000,0)),'software VS stream lacks full cache flush/stop')
    if run['sequence']=='N' and run['mode']=='plain':require(len(ds)==1,'GS emulation compute work reached native route')
    words=struct.unpack('<40I',read(index,addr,160));require(words[32]==0x70000600 and words[39]==0xc0000000,'native render must contain one direct launch and stop')
    vs=root(previous['submission'],ds[-1]['launch']);mesh=root(index,(1<<40)+words[8]*64)
    require(vs['params']==mesh['params'] and vs['output']==mesh['output'],'native consumer and VS producer have different buffers')
    row['vs']=vs;row['mesh']=mesh;row['state_words']=list(words[:13]);native.append(row)
   renders.append(row)
  commands.append(row)
 expected=sum(x=='N' for x in run['sequence']) if run['native']=='1' and run['mode'] in ['plain','merged','deferred','timestamp'] else 0
 require(len(native)==expected,'native packet count differs from final route')
 require(renders and renders[0]['attachment_programs'][0]!=renders[0]['attachment_programs'][2],'first render must clear')
 # Separate API render passes build their own load programs; unlike native
 # mesh, GS still has a software VS dependency and need not hardware-merge.
 # Check byte-identical continuation programs only within one API pass.
 for render in renders[1:]:
  if run['mode']!='merged':require(render['attachment_programs'][0]==render['attachment_programs'][2],f'{case["case"]} continuation must load: {render}')
 for row in commands:row.pop('_blob',None)
 return dict(case=case['case'],native_renders=len(native),commands=commands)
rows=json.loads((a.directory/'HOST-RECORDS.json').read_text());require(all(r['returncode']==0 for r in rows),'failed host invocation')
result=dict(result='pass',gpu_execution=False,cases=[check(r) for r in rows]);print(json.dumps(result,indent=2))
