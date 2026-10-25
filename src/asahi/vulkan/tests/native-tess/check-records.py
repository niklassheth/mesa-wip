#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check our bounded TES host recordings, never GPU execution or pixels."""
import argparse,importlib.util,json,struct
from pathlib import Path
spec=importlib.util.spec_from_file_location('transition',Path(__file__).resolve().parents[1]/'native-mesh-hybrid/check-transitions.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
require=c.require

def check(case,directory,o):
 d=directory/('host-'+case['case'])/'trace';run=json.loads((d.parent/'RUN.json').read_text())
 require(run['gpu_execution'] is False and run['queue_submission'] is True,'wrong evidence scope')
 read=lambda index,addr,size:c.capture.read_gpu(d,index,addr,size)
 def u64(index,addr):return struct.unpack('<Q',read(index,addr,8))[0]
 def launch_root(index,launch,direct=False):
  header=read(index,launch,16);arg=c.literal32(header[:8],2)|(c.literal32(header[8:],3)<<32)
  address=u64(index,arg+8) if direct else u64(index,u64(index,arg+16))
  return address,arg
 def root(index,launch,direct=False):
  address,arg=launch_root(index,launch,direct)
  require(u64(index,address)==address,'missing descriptor self pointer')
  params=u64(index,address+o['vertex_params']);output=u64(index,params+o['params_output'])
  tess=u64(index,address+o['tess_params'])
  u32=lambda x:struct.unpack('<I',read(index,x,4))[0]
  patch=run['patch']
  require(output and u64(index,params+o['params_mask'])==1,'software VS output allocation/mask missing')
  require(u64(index,address+o['vertex_outputs'])==1,'root output mask differs')
  require(not u64(index,address+o['geometry_params']),'emulated GS output root leaked into native draw')
  require(u32(params+o['params_vertices'])==patch,'wrong input patch size')
  require(struct.unpack('<6I',read(index,params+o['params_grid'],24))==(patch,1,1,64,1,1),'wrong software VS grid')
  require(u32(tess+o['tess_input_patch'])==patch and u32(tess+o['tess_output_patch'])==patch,'wrong patch metadata')
  require(u32(tess+o['tess_nr_patches'])==1,'native draw is not one patch')
  spacing={'fractional_odd_spacing':0,'fractional_even_spacing':1,'equal_spacing':2}
  require(u32(tess+o['tess_partitioning'])==spacing[run['spacing']],'spacing lost')
  ccw=(run['order']=='ccw') ^ bool(int(run['environment'].get('TESS_TEST_LOWER_LEFT',0)))
  require(u32(tess+o['tess_ccw'])==ccw,'domain winding lost')
  draws=u64(index,tess+o['tess_out_draws'])
  require(draws and not draws%4,'missing tessellator draw allocation')
  read(index,draws,20);read(index,output,patch*16)
  read(index,u64(index,tess+o['tess_tcs_buffer']),4*u32(tess+o['tess_stride']))
  read(index,u64(index,tess+o['tess_counts']),4)
  read(index,u64(index,tess+o['tess_coord_allocs']),4)
  return dict(root=address,params=params,output=output,tess=tess,draw=draws,args=arg)
 commands=[];native=[];renders=[]
 for path in sorted(d.glob('*.commands.bin')):
  data=path.read_bytes();kind,size,vdm,cdm=struct.unpack_from('<4H',data);body=data[8:]
  require(len(body)==size,'one complete command per ioctl required')
  index=path.name.split('.')[0];addr=struct.unpack_from('<Q',body,8)[0]
  row=dict(submission=index,kind=kind,vdm_barrier=vdm,cdm_barrier=cdm)
  # Integration elides waits only before a subqueue has any prior work.
  # Every later software-VS/native-consumer edge must retain its full join.
  for queue,wait in [(0,vdm),(1,cdm)]:
   require(wait==0 if any(r['kind']==queue for r in commands) else wait in (0,65535),
           f'{case["case"]}: lost dependency on populated subqueue {queue}: {wait}')
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
     dispatches.append(dict(global_size=list(w[4:7]),local_size=list(w[7:10]),launch=(1<<40)+w[2]*64,offset=len(blob)))
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
    patch=run['patch']
    candidates=[r for r in commands if r['kind']==1 and len(r['dispatches'])>=5 and r['dispatches'][-5]['global_size']==[patch,1,1] and r['dispatches'][-5]['local_size']==[64,1,1]]
    require(candidates, f'{case["case"]}: missing tessellation producer passes')
    previous=candidates[-1];ds=previous['dispatches'][-5:];si=previous['submission']
    require([x['global_size'] for x in ds]==[[patch,1,1],[patch,1,1],[1,1,1],[1024,1,1],[1,1,1]],'wrong VS/TCS/count/prefix/emit order')
    require([x['local_size'] for x in ds[:2]]==[[64,1,1],[patch,1,1]],'wrong software VS/TCS local sizes')
    require(ds[2]['local_size']==ds[4]['local_size'] and ds[3]['local_size']==[1024,1,1],'wrong tessellation helper local sizes')
    intervening=commands[commands.index(previous)+1:]
    require(not intervening or (run['mode']=='timestamp' and all(r['kind']==1 for r in intervening)), 'unexpected work between tessellation and render')
    require(previous['_blob'].endswith(struct.pack('<4I',0x60000160,0x600fffff,0x40000000,0)),'tessellation stream lacks full cache flush/stop')
    for first,second in zip(ds,ds[1:]):
     gap=previous['_blob'][first['offset']+40:second['offset']]
     require(gap and len(gap)%4==0 and all(w[0] in (0x60000160,0x600fffff) for w in struct.iter_unpack('<I',gap)),'producer dependency barrier missing')
    if run['sequence']=='N' and run['mode']=='plain':require(len(previous['dispatches'])==6,f'unexpected work in native tess producer: {previous["dispatches"]}')
    words=struct.unpack('<40I',read(index,addr,160));require(words[32]==0x70000600 and words[39]==0xc0000000,'native render must contain one direct launch and stop')
    # Our PPP constructor puts raster control at group(0x40)+0x70.
    ppp=(0x1000000000)+words[15]
    raster=struct.unpack('<I',read(index,ppp+0xb0,4))[0]
    expected_front=run['environment'].get('TESS_TEST_FRONT','ccw')=='ccw'
    require(bool(raster&(1<<16))==expected_front,'native TES winding applied twice')
    expected_cull={'none':0,'front':1,'back':2}[run['environment'].get('TESS_TEST_CULL','none')]
    require(raster&3==expected_cull,'cull state lost')
    row['raster_control']=raster
    vs=root(si,ds[0]['launch']);tcs=root(si,ds[1]['launch'],True);mesh=root(index,(1<<40)+words[8]*64)
    obj=root(index,(1<<40)+words[6]*64)
    if run['sequence']=='N' and run['mode']=='plain':
     reset=previous['dispatches'][0]
     require(reset['global_size']==reset['local_size']==[1,1,1],'missing heap reset')
     reset_args,_=launch_root(si,reset['launch'],True)
     require(u64(si,reset_args)==u64(si,vs['tess']+o['tess_heap']),'wrong heap reset address')
    require(all(r['tess']==vs['tess'] and r['params']==vs['params'] for r in [tcs,mesh,obj]),'producer/consumer buffers differ')
    for helper in ds[2:]:
     helper_args,_=launch_root(si,helper['launch'],True)
     require(u64(si,helper_args)==vs['tess'],'tessellation helper uses different params')
    require(u64(index,obj['args']+o['draw_uniform_offset'])==vs['draw'],'object does not read the count writer draw word')
    require(obj['args']==mesh['args'],'native object/mesh arguments differ')
    row['vs']=vs;row['tcs']=tcs;row['mesh']=mesh;row['object']=obj;row['state_words']=list(words[:13]);native.append(row)
   renders.append(row)
  commands.append(row)
 expected=sum(x=='N' for x in run['sequence']) if run['native']=='1' and run['mode'] in ['plain','merged','kept-merge','deferred','timestamp'] else 0
 require(len(native)==expected,'native packet count differs from final route')
 require(renders and renders[0]['attachment_programs'][0]!=renders[0]['attachment_programs'][2],'first render must clear')
 # Separate API render passes build their own load programs; unlike native
 # mesh, GS still has a software VS dependency and need not hardware-merge.
 # Check byte-identical continuation programs only within one API pass.
 for render in renders[1:]:
  if run['mode'] not in ['merged','kept-merge']:require(render['attachment_programs'][0]==render['attachment_programs'][2],f'{case["case"]} continuation must load: {render}')
 for row in commands:row.pop('_blob',None)
 return dict(case=case['case'],native_renders=len(native),commands=commands)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
 o=json.loads((a.directory/'OFFSETS.json').read_text())
 rows=json.loads((a.directory/'HOST-RECORDS.json').read_text());require(all(r['returncode']==0 for r in rows),'failed host invocation')
 result=dict(result='pass',gpu_execution=False,cases=[check(r,a.directory,o) for r in rows]);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
