#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Manager-scheduled native capability pilot; persistent v2 owner per mode."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from kmsg import Kmsg
from transport import Runner

MODE=Path('/sys/kernel/debug/asahi_m4/native_mesh')
STATUS=['PASS','COMPLETED_PIXEL_FAILURE','Q_FAILURE','Q_SAMPLE_FAILURE']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--kernel-identity',type=Path,required=True)
    ap.add_argument('--kernel-requirements',type=Path,required=True)
    ap.add_argument('--lease-deadline-epoch',type=float,required=True)
    ap.add_argument('--max-seconds',type=float,default=420)
    ap.add_argument('--trials',type=int,choices=range(1,5),default=2)
    a=ap.parse_args();root=Path(__file__).resolve().parent;out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();left=a.lease_deadline_epoch-time.time()
    assert 300<left<=900 and 0<a.max_seconds<=420
    deadline=min(start+a.max_seconds-20,start+left-260)
    pins=json.loads(a.kernel_requirements.read_text())
    assert pins['ready'] is True,'kernel owner must freeze an admitted kernel before scheduling'
    assert pins['all_fixtures_accepted'] is True and pins['uapi_changed'] is False
    assert pins['fixtures_catalog_sha256']==hashlib.sha256((root/'fixtures/INDEX.json').read_bytes()).hexdigest()
    research_mode=pins['native_mesh_mode'];assert type(research_mode) is int and 3<=research_mode<=255
    (out/'KERNEL-REQUIREMENTS.json').write_text(json.dumps(pins,indent=2)+'\n')
    identity=json.loads(a.kernel_identity.read_text())
    for key in ['kernel_source_commit','kernel_image_sha256','kernel_dtb_sha256','kernel_release','kernel_version']:
        assert identity[key]==pins[key],key
    assert os.uname().release==pins['kernel_release'] and os.uname().version==pins['kernel_version']
    assert os.sysconf('SC_PAGE_SIZE')==16384 and len(Path('/proc/swaps').read_text().splitlines())==1
    assert Path('/sys/module/zswap/parameters/enabled').read_text().strip()=='N'
    assert 'ro' in subprocess.check_output(['findmnt','-no','OPTIONS','/'],text=True).strip().split(',')
    assert int(MODE.read_text())==0 and not os.environ.get('ASAHI_HOST_TRACE_DIR')
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,f=line.split('  ',1);assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h,f
    (out/'KERNEL-IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
    catalog=json.loads((root/'CATALOG.json').read_text())
    monitor=Kmsg(out);monitor.check();runner=None;control=None;mode=0;seq=0;busy=False
    counts=Counter();rows=(out/'results.jsonl').open('w',buffering=1)
    interrupted=[]
    def stop(sig,frame):
        interrupted.append(sig)
        if monitor.pid:
            try:os.kill(monitor.pid,signal.SIGSTOP)
            except ProcessLookupError:pass
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,stop)
    def check():
        monitor.check()
        assert not interrupted,'interrupted: backing retained'
        assert int(MODE.read_text())==mode,'mode changed'
    try:
        # Ordinary vertex draws qualify the exact raster mask and additive
        # count before any broad native submission, with native mode off.
        env={k:v for k,v in os.environ.items() if not k.startswith(('AGX_','ASAHI_','HK_','MESA_','VK_','MULTI_','MESH_','CROSS_','LD_'))}
        icd=out/'calibration-icd.json'
        icd.write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path=str(root/'lib/libvulkan_asahi.so'),api_version='1.4.305'))))
        env.update(VK_DRIVER_FILES=str(icd),HK_EXPERIMENTAL_APPLE9='1',MESHPROBE_DRIVER_COMMIT='e08fa6cf73',MESA_SHADER_CACHE_DISABLE='true',MESA_VK_ENABLE_SUBMIT_THREAD='false')
        with (out/'calibration.log').open('wb') as log:
            control=subprocess.Popen([str(root/'calibrate'),str(root)],env=env,stdout=log,stderr=log)
            monitor.attach(control.pid)
            while control.poll() is None:
                check()
                if time.monotonic()>min(deadline,start+60):raise TimeoutError('calibration deadline')
                time.sleep(.05)
        assert control.returncode==0 and 'QUALIFIED 21' in (out/'calibration.log').read_text()
        control=None;monitor.attach(None)
        for next_mode in [research_mode]:
            check()
            if time.monotonic()>=deadline:break
            MODE.write_text(f'{next_mode}\n');mode=next_mode;check()
            runner=Runner(root,out/f'mode-{mode}',catalog,1,monitor=monitor,deadline=deadline)
            runner.load_all(check);runner.start()
            paths={l.split()[-1] for l in Path('/proc',str(runner.pid),'maps').read_text().splitlines() if '/libvulkan_asahi.so' in l}
            assert paths=={str(root/'lib/libvulkan_asahi.so')},paths
            # Low counts first. Independent V/P/T probes precede joint maxima;
            # repetitions stay adjacent so one failing arm stops immediately.
            ordered=sorted(catalog['arms'],key=lambda x:(x['shape']['control42']==0,x['shape']['topology']!=6,x['shape']['threads'],x['shape']['vertices']*x['shape']['primitives'],x['id']))
            for arm in ordered:
                for repetition in range(a.trials):
                    check()
                    if time.monotonic()>=deadline:break
                    seq+=1;cookie=(seq*0x9e3779b1)&0xffffffff
                    monitor.begin(seq,arm['name'],cookie);busy=True
                    runner.submit(arm,0,seq,cookie);done=None;until=time.monotonic()+10
                    while done is None:
                        check()
                        if time.monotonic()>until:raise TimeoutError('GPU completion deadline')
                        done=runner.poll(0,1000000)
                    check();assert done['sequence']==seq
                    monitor.complete(seq,done['complete_ns'])
                    status=STATUS[done['code']];counts[status]+=1
                    evidence=out/f'evidence-{seq:06d}';runner.dump(0,evidence)
                    row=dict(mode=mode,arm=arm['name'],repetition=repetition,cookie=cookie,status=status,evidence=evidence.name,**done)
                    rows.write(json.dumps(row)+'\n')
                    # An exact authored-input snapshot is separate from raster
                    # qualification. No failed shape authorizes bigger shapes.
                    if done['code']:raise RuntimeError(f'{status}: {arm["name"]}')
                    runner.release(0);busy=False
                if time.monotonic()>=deadline:break
            runner.protocol_deadline=None;runner.close();runner=None;monitor.attach(None)
            check()
        MODE.write_text('0\n');mode=0
        complete=seq==len(catalog['arms'])*a.trials
        (out/'COMPLETE.json').write_text(json.dumps(dict(complete=complete,submitted=seq,counts=counts,mode_restored=0,unattempted_are_not_passes=True),indent=2)+'\n')
    except BaseException as e:
        if monitor.pid:
            try:os.kill(monitor.pid,signal.SIGSTOP)
            except ProcessLookupError:pass
        (out/'STOP.json').write_text(json.dumps(dict(reason=repr(e),mode=mode,sequence=seq,counts=counts,owner_pid=monitor.pid,in_flight=busy,reset_required=monitor.pid is not None),indent=2)+'\n')
        print('STOP: collect evidence and reset before releasing owner:',repr(e),flush=True)
        if monitor.pid:
            while True:time.sleep(1)
        raise
    finally:rows.close();monitor.close()


if __name__=='__main__':main()
