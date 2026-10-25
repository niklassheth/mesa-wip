#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded GS/TES varying comparison. The canonical operator owns device recovery."""
import hashlib, json, os, re, signal, struct, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from check import CASES,verify
KNOB=Path('/sys/kernel/debug/asahi_m4/native_mesh')
DEP=Path('/sys/kernel/debug/asahi_m4/dep_stats')
COUNTERS=('native_renders','native_dispatches')
ARMS=('unpacked','packed','emulated')

def write(path,value):
    with path.open('w') as f:
        f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def require(ok,reason):
    if not ok: raise RuntimeError(reason)
def counts(raw):
    result={}
    for name in COUNTERS:
        values=re.findall(r'^'+name+r'[ \t]+(\d+)[ \t]*$',raw,re.M)
        require(len(values)==1,'missing/ambiguous counter '+name)
        result[name]=int(values[0])
    return result

def audit(log,case,arm,revision,before,after,library):
    # Parse every evidence source before validation, including on a model that
    # cannot produce native routes. No rehearsal branch bypasses this checker.
    vm=re.findall(r'VM_AUDIT (before|after) fd=(\d+) dev=(\d+:\d+) result=(-?\d+) '
        r'asahi-native-mesh v1 generation=(\d+) profile=(none|research|qualified) classes=(\S+)',log)
    maps=[l for l in log.splitlines() if l.startswith('VM_MAP ')]
    gs=re.findall(r'HK_GS_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)',log)
    tes=re.findall(r'HK_TESS_ROUTE native=(\d+) candidate=(\d+) prefer=(\d+)',log)
    delta={k:after[k]-before[k] for k in COUNTERS}
    native=arm!='emulated';errors=[]
    def check(ok,why):
        if not ok:errors.append(why)
    check('Apple M4' in log and 'Honeykrisp' in log and 'git-'+revision[:10] in log,'wrong renderer/revision')
    check(len(vm)>=2 and len(vm)%2==0,'missing own-FD VM audit')
    for first,last in zip(vm[::2],vm[1::2]):
        check(first[0]=='before' and last[0]=='after' and first[1:]==last[1:],'VM changed during creation')
        check(first[3]=='0' and first[5:]==('research','-') and int(first[4])<1<<56,'wrong VM profile')
    check(bool(maps) and all(s.endswith(str(library)) for s in maps),'wrong mapped Mesa')
    if case.startswith('gs'):
        shape=(16,3,1) if case=='gs3' else (32,4,2)
        if arm=='packed':shape=(32,24,17) if case=='gs3' else (32,32,16)
        want=[('1',*map(str,shape))] if native else [('0','0','0','0')]
        check(gs==want and not tes,'GS native route mismatch (zero native routes is a harness error)')
    else:
        check(tes==[('1','1','1') if native else ('0','0','0')] and not gs,'TES native route mismatch')
    check(delta==dict.fromkeys(COUNTERS,int(native)),'native kernel counter mismatch')
    return dict(verdict='FAIL' if errors else 'PASS',errors=errors,vm=vm,routes=gs or tes,counters=delta,mapped_lines=len(maps))

def environment(out,d,arm):
    e={k:v for k,v in os.environ.items() if not k.upper().startswith(('LD_','VK_','HK_','MESA_','ASAHI_','AGX_','DRI')) and k not in ('DISPLAY','WAYLAND_DISPLAY')}
    e.update(VK_DRIVER_FILES=str(out/'icd.json'),VK_ICD_FILENAMES=str(out/'icd.json'),
        VK_LOADER_LAYERS_DISABLE='~implicit~',HK_EXPERIMENTAL_APPLE9='1',
        LD_PRELOAD=str(HERE/'lib/vm-audit.so'),MESA_SHADER_CACHE_DISABLE='true',
        MESA_VK_ENABLE_SUBMIT_THREAD='false',MESA_SHADER_CACHE_DIR=str(d/'cache'),
        DRIRC_CONFIGDIR=str(HERE/'drirc-empty'),XDG_CONFIG_HOME=str(out/'config'),
        HK_APPLE9_GS_ROUTE_LOG='1',HK_APPLE9_TESS_ROUTE_LOG='1',HK_APPLE9_NATIVE_INDEX='1',
        HK_APPLE9_NATIVE_MESH='0' if arm=='emulated' else '1',HK_APPLE9_GS_NATIVE='1',HK_APPLE9_TESS_NATIVE='1',
        HK_APPLE9_NATIVE_GS_PACK='1' if arm=='packed' else '0',HK_APPLE9_NATIVE_TES_PACK='1' if arm=='packed' else '0',
        HK_APPLE9_NATIVE_MESH_PACKED_SHAPES='T16V3P1,T16V4P2,T32V24P17,T32V32P16')
    return e

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];proc=None;fault_class='harness';attempted=retired=0
    plan=json.loads((HERE/'PLAN.json').read_text());library=HERE/'lib/libvulkan_asahi.so'
    def interrupted(sig,frame):raise RuntimeError('interrupted '+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    def verdict(value,**extra):
        return dict(verdict=value,results=rows,native_qualification=False,
            quality=dict(planned=12,attempted=attempted,retired=retired,
                readbacks_checked=sum('pixels' in r for r in rows),
                native_audits_passed=sum(r.get('audit',{}).get('verdict')=='PASS' and r['arm']!='emulated' for r in rows),
                retries=0),**extra)
    try:
        require(not (out/'START.json').exists(),'output already used')
        write(out/'START.json',plan)
        require(hashlib.sha256(library.read_bytes()).hexdigest()==plan['driver_sha256'],'wrong library hash')
        require(os.uname().machine=='aarch64' and os.sysconf('SC_PAGESIZE')==16384,'wrong architecture/page size')
        require(os.uname().version==plan['kernel_version'],'wrong kernel')
        require(KNOB.read_text().strip()=='0','mode must start at zero')
        require(not (Path.home()/'.drirc').exists(),'unexpected per-user drirc')
        write(out/'icd.json',dict(file_format_version='1.0.0',ICD=dict(library_path=str(library),api_version='1.4.305')))
        (out/'config').mkdir()
        # Driver blob dependencies are resolved after staging, before a GPU opens.
        dep=subprocess.run(['/lib/ld-linux-aarch64.so.1','--list',str(library)],capture_output=True,text=True,timeout=10)
        require(dep.returncode==0 and 'not found' not in dep.stdout+dep.stderr,'driver dependencies unavailable')
        KNOB.write_text('3\n');require(KNOB.read_text().strip()=='3','research mode did not latch')
        for case in CASES:
            reference=None
            for arm in ARMS:
                fault_class='harness';d=out/(case+'-'+arm);d.mkdir()
                before_raw=DEP.read_text();(d/'dep-before.txt').write_text(before_raw);before=counts(before_raw)
                e=environment(out,d,arm);argv=list(map(str,[HERE/'bin/probe',HERE/'shaders',d,case]))
                write(d/'RUN.json',dict(argv=argv,environment={k:v for k,v in e.items() if k.startswith(('HK_','MESA_','VK_','LD_'))}))
                attempted+=1;started=time.monotonic()
                with (d/'stdout.log').open('w') as f:
                    proc=subprocess.Popen(argv,env=e,cwd=d,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    while proc.poll() is None:
                        log=(d/'stdout.log').read_text(errors='replace')
                        if 'SUBMIT_BEGIN' in log:fault_class='product'
                        require(time.monotonic()-started<25,'process watchdog expired')
                        require('IN_FLIGHT_WAIT_FAILED' not in log,'uncertain GPU retirement')
                        time.sleep(.02)
                    code=proc.returncode;proc=None
                log=(d/'stdout.log').read_text(errors='replace')
                if 'SUBMIT_BEGIN' in log:fault_class='product'
                row=dict(case=case,arm=arm,exit_code=code);rows.append(row)
                require(code==0 and log.count('SUBMIT_OK\n')==1,'probe initialization or retirement failure')
                retired+=1
                row['pixels']=verify(d/'pixels.bin',case);write(d/'PIXELS.json',row['pixels'])
                require(row['pixels']['verdict']=='PASS','pixel mismatch')
                after_raw=DEP.read_text();(d/'dep-after.txt').write_text(after_raw)
                row['audit']=audit(log,case,arm,plan['driver_revision'],before,counts(after_raw),library)
                write(d/'AUDIT.json',row['audit']);fault_class='harness'
                require(row['audit']['verdict']=='PASS','; '.join(row['audit']['errors']))
                require(KNOB.read_text().strip()=='3','mode changed during draw')
                fault_class='product'
                data=struct.unpack('<32768f',(d/'pixels.bin').read_bytes())
                if reference is None:reference=data
                error=max(abs(a-b) for a,b in zip(data,reference));row['cross_arm_max_error']=error
                require(error<=2e-5,'native/emulated readbacks differ')
                row['verdict']='PASS';write(out/'RESULTS.json',rows)
                print(case+' '+arm+': PASS',flush=True)
        KNOB.write_text('0\n');require(KNOB.read_text().strip()=='0','mode restoration failed')
        write(out/'VERDICT.json',verdict('PASS',mode_restored=0));return 0
    except BaseException as error:
        retained=bool(proc and proc.poll() is None)
        if retained:os.killpg(proc.pid,signal.SIGSTOP)
        restored=None
        if not retained:
            try:KNOB.write_text('0\n');restored=0 if KNOB.read_text().strip()=='0' else None
            except OSError:pass
        write(out/'VERDICT.json',verdict('FAIL',fault_class=fault_class,reason=str(error),retained=retained,mode_restored=restored))
        print('STOP: '+str(error),flush=True)
        if retained:
            write(out/'RETAINED.json',dict(pid=proc.pid,recovery='canonical operator; no retry'))
            os.kill(os.getpid(),signal.SIGSTOP)
            while True:signal.pause()
        return 1
if __name__=='__main__':sys.exit(main(Path(sys.argv[1]).resolve()))
