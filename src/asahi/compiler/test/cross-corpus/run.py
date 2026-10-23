#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream compiler measurements, isolating failures without losing later inputs."""
import argparse,collections,hashlib,json,os,re,resource,signal,sqlite3,subprocess,time,zlib
from pathlib import Path
from packed import Pack,results
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--population',required=True);p.add_argument('--icd',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--host-shim',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--exclude-results',type=Path,action='append',default=[]);a=p.parse_args()
assert os.uname().machine=='x86_64';a.output.mkdir(parents=True,exist_ok=a.resume);out=a.output.resolve();pack=Pack(a.pack);paths=pack.keys(a.population);excluded=set();prior_counts=collections.Counter()
if a.resume:
 for r in results(out):excluded.add(r['key']);prior_counts[r['status']]+=1
for prior in a.exclude_results:
 excluded.update(r['key'] for r in results(prior))
paths=[x for x in paths if x not in excluded]
paths=paths[:a.limit] if a.limit else paths
icd=a.icd.resolve();lib=Path(json.loads(icd.read_text())['ICD']['library_path']);lib=lib if lib.is_absolute() else icd.parent/lib
keys=['AGX_MESA_DEBUG','ASAHI_MESA_DEBUG','NIR_DEBUG','AGX_APPLE9_TRACE','AGX_APPLE9_ASM','AGX_APPLE9_TRACE_NIR','AGX_APPLE9_TRACE_FINAL_NIR','AGX_SPILL_NO_LOOP_HANDLING','AGX_SPILL_COALESCE','AGX_APPLE9_RA_WEBS','AGX_APPLE9_RA_AUDIT']
env=dict(os.environ)
for k in keys:env.pop(k,None)
env.update(LD_PRELOAD=str(a.host_shim.resolve()),VK_DRIVER_FILES=str(icd),HK_EXPERIMENTAL_APPLE9='1',HK_APPLE9_EAGER_VARIANTS='1',AGX_APPLE9_STATS='1',MESA_SHADER_CACHE_DISABLE='true',CORPUS_DIRECT_PRERAST='1')
(out/('resume-provenance.json' if a.resume else 'provenance.json')).write_text(json.dumps(dict(icd=str(icd),library=str(lib),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),shim_sha256=hashlib.sha256(a.host_shim.read_bytes()).hexdigest(),inputs=len(paths),pack=str(a.pack.resolve()),population=a.population,env={k:env[k] for k in env if k.startswith(('AGX_','HK_','MESA_','NIR_','CORPUS_'))},excluded_results=[str(x.resolve()) for x in a.exclude_results],batch_size=a.batch_size,load_before=os.getloadavg()),indent=2)+'\n')
index={v:i for i,v in enumerate(paths)};counts=prior_counts;start=time.perf_counter();begin=resource.getrusage(resource.RUSAGE_CHILDREN);offset=0;current=None;process=None
stat=re.compile(r'^(APPLE9(?:_PREAMBLE)?)_CODEGEN (.*)')
def fields(s):return {k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',s)}
def timed_out(*_):
 if process:process.kill()
 raise TimeoutError('shader compile exceeded 120 seconds')
signal.signal(signal.SIGALRM,timed_out)
with sqlite3.connect(out/'results.sqlite3') as output:
 output.execute('PRAGMA journal_mode=WAL')
 output.execute('PRAGMA synchronous=NORMAL')
 output.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,data BLOB NOT NULL) WITHOUT ROWID')
 def emit(row):
  output.execute('INSERT INTO results VALUES(?,?)',(row['key'],zlib.compress(json.dumps(row).encode(),1)));counts[row['status']]+=1
  if sum(counts.values())%64==0:output.commit()
 while offset<len(paths):
  pending=out/'pending.bin';batch={}
  with pending.open('wb') as f:
   for key in paths[offset:offset+a.batch_size]:
    frame,record=pack.frame(a.population,key);f.write(frame);batch[key]=record
  current=None;completed=offset
  with pending.open('rb') as stdin:
   process=subprocess.Popen([str(a.runner.resolve())],env=env,stdin=stdin,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,errors='replace')
   signal.alarm(120);failure=None
   try:
    for line in process.stderr:
     line=line.rstrip()
     if line.startswith('CORPUS_BEGIN '):
      path=line[13:];signal.alarm(120);stage=batch[path]['stage'];current=dict(key=path,population=a.population,index=index[path],stage={1:0,2:1,4:2,8:3,16:4,32:5}[stage],programs=[],attempts=[],finals=[],timings=[],diagnostics=[])
     elif current:
      if m:=stat.match(line):current['attempts'].append(dict(kind=m[1],**fields(m[2])))
      elif line.startswith('CORPUS_FINAL '):
       final=fields(line);current['finals'].append(final)
       for kind,size in [('APPLE9_PREAMBLE',final['preamble_bytes']),('APPLE9',final['main_bytes'])]:
        if not size:continue
        matches=[x for x in current['attempts'] if x['kind']==kind and x['stage']==final['stage'] and x['bytes']==size]
        if matches:current['programs'].append(matches[-1])
        else:current['diagnostics'].append('Missing final statistics: '+kind+' '+str(final))
       current['attempts']=[]
      elif line.startswith('CORPUS_EMPTY '):current['empty']=True
      elif line.startswith('CORPUS_TIME '):current['timings'].append(fields(line))
      elif line.startswith('CORPUS_CREATE '):current['create']=fields(line)
      elif line.startswith('CORPUS_END '):
       current['exit']=int(line.split()[-1]);mains=[x for x in current['programs'] if x['kind']=='APPLE9' and (current['stage'] in (1,2,3) or x['stage']==current['stage']) and not x['stub']];current['status']='compiler_failure' if current['exit'] or not all(x['ok'] for x in current['timings']) else 'missing_metrics' if any(x.startswith('Missing final statistics:') for x in current['diagnostics']) else 'pass' if mains else 'empty_shader' if current.get('empty') else 'missing_metrics';completed=current['index']+1;emit(current);current=None
      elif len(current['diagnostics'])<(256 if env.get('CORPUS_DIAGNOSTICS') else 15):current['diagnostics'].append(line[:4000] if env.get('CORPUS_DIAGNOSTICS') else line[:1000])
    rc=process.wait()
   except TimeoutError as e:failure=str(e);rc=process.wait()
   finally:signal.alarm(0)
  if current:
   current.update(status='timeout' if failure else 'crash',process_exit=rc,error=failure);completed=current['index']+1;emit(current)
  output.commit()
  if completed==offset:raise RuntimeError('worker failed before any shader: '+str(rc))
  offset=completed
  if offset%1024==0:print('completed',offset,dict(counts),flush=True)
 pending.unlink(missing_ok=True)
u=resource.getrusage(resource.RUSAGE_CHILDREN)
summary=dict(outcomes=dict(counts),inputs=sum(counts.values()),resumed=a.resume,new_inputs=len(paths),wall_s=time.perf_counter()-start,user_s=u.ru_utime-begin.ru_utime,system_s=u.ru_stime-begin.ru_stime,load_after=os.getloadavg())
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
