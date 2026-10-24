#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Six ordinary Vulkan draws; operator owns staging, recovery and collection."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check import cases, verify
from kmsg import Kmsg

KNOB = Path('/sys/kernel/debug/asahi_m4/native_mesh')
DEP = Path('/sys/kernel/debug/asahi_m4/dep_stats')
ADVERTISEMENT = Path('/sys/class/drm/renderD128/device/asahi_native_mesh')
REHEARSAL = os.environ.get('KIT_REHEARSAL') == '1'


def write(path, value):
    with path.open('w') as f:
        f.write(json.dumps(value, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def counts(raw):
    result = {}
    for key in ('native_renders', 'native_dispatches'):
        matches = re.findall(r'^' + key + r'[ \t]+(\d+)[ \t]*$', raw, re.M)
        require(len(matches) == 1, 'missing/ambiguous counter ' + key)
        result[key] = int(matches[0])
    return result


def audit(log, case, revision, before, after):
    require('Apple M4' in log and 'Honeykrisp' in log and 'git-' + revision[:10] in log,
            'wrong renderer/revision')
    rows = re.findall(r'VM_AUDIT (before|after) fd=(\d+) dev=(\d+:\d+) result=(-?\d+) '
        r'asahi-native-mesh v1 generation=(\d+) profile=(none|research|qualified) classes=(\S+)', log)
    require(len(rows) >= 4 and len(rows) % 2 == 0, 'missing own-FD VM audit')
    for first, last in zip(rows[::2], rows[1::2]):
        require(first[0] == 'before' and last[0] == 'after' and first[1:] == last[1:],
                'VM generation/profile changed during creation')
        require(first[3] == '0' and first[5:] == ('research', '-'), 'wrong VM research latch')
    maps = [s for s in log.splitlines() if s.startswith('VM_MAP ')]
    require(maps and all(s.endswith(str(HERE/'lib/libvulkan_asahi.so')) for s in maps),
            'unexpected mapped driver')
    routes = re.findall(r'HK_MESH_ROUTE native=(\d+) T=(\d+) V=(\d+) P=(\d+)', log)
    require(routes == [('1', '16', '16', str(case['maximum']))], 'native route mismatch')
    delta = {k: after[k] - before[k] for k in before}
    # Admission counters are usable only after checking the retired fence and
    # complete readback. A failed submit never becomes completed native work.
    require(delta == dict(native_renders=1, native_dispatches=1), 'admission counter mismatch: ' + str(delta))
    return dict(vm_audit=rows, counters=delta, routes=routes)


def environment(out):
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ('VK_', 'HK_', 'MESA_', 'AGX_', 'ASAHI_', 'LD_', 'DRI', 'M1_'))
        and k not in ('DISPLAY', 'WAYLAND_DISPLAY')}
    e.update(VK_DRIVER_FILES=str(out/'icd.json'), VK_ICD_FILENAMES=str(out/'icd.json'),
        VK_LOADER_LAYERS_DISABLE='~implicit~', HK_EXPERIMENTAL_APPLE9='1',
        HK_APPLE9_NATIVE_MESH='1',
        HK_APPLE9_NATIVE_MESH_PACKED_SHAPES='T16V16P15,T16V16P16,T16V16P17',
        HK_APPLE9_MESH_ROUTE_LOG='1', HK_APPLE9_RESEARCH_NATIVE_BATCH='1',
        MESA_SHADER_CACHE_DISABLE='true', MESA_VK_ENABLE_SUBMIT_THREAD='false',
        DRIRC_CONFIGDIR=str(HERE/'drirc-empty'), XDG_CONFIG_HOME=str(out/'config'),
        LD_PRELOAD=str(HERE/'lib/vm-audit.so'))
    return e


def output_budget(directory):
    files = [p for p in directory.rglob('*') if p.is_file()]
    require(len(files) <= 32 and sum(p.stat().st_size for p in files) <= 8*1024*1024,
            'per-process output budget exhausted')


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'START.json').exists(), 'output already used')
    plan = json.loads((HERE/'PLAN.json').read_text())
    selected = cases()[:1] if REHEARSAL else cases()
    write(out/'START.json', dict(rehearsal=REHEARSAL, planned_draws=len(selected),
        driver_sha256=hashlib.sha256((HERE/'lib/libvulkan_asahi.so').read_bytes()).hexdigest()))
    rows = []
    proc = monitor = None
    attempted = retired = 0
    failure_class = 'harness'
    deadline = time.monotonic() + 110

    def interrupted(sig, frame):
        raise RuntimeError('interrupted by signal ' + str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def result(verdict, **extra):
        return dict(verdict=verdict, rehearsal=REHEARSAL, native_qualification=False,
            scope='initial T16V16P15..17 sample, full/partial counts', results=rows,
            quality=dict(planned_draws=len(selected), attempted=attempted, retired=retired,
                unattempted=len(selected)-attempted,
                checked_readbacks=sum('readback' in r for r in rows),
                pixel_buffer_passes=sum(r.get('readback', {}).get('verdict') == 'PASS' for r in rows),
                native_evidence_passes=sum(r.get('native_pass', False) for r in rows),
                native_evidence_checked=not REHEARSAL, retries=0), **extra)

    try:
        require(plan['cases'] == cases(), 'sealed plan does not match checker')
        require(os.uname().machine == 'aarch64' and os.sysconf('SC_PAGESIZE') == 16384,
                'wrong architecture/page size')
        require(os.uname().version == plan['kernel_version'] and
                os.uname().release == plan['kernel_release'], 'wrong kernel')
        args = Path('/proc/cmdline').read_text().split()
        require('asahi_m4.native_batch=16' in args and 'asahi_m4.native_batch_payload=0' in args,
                'wrong research boot parameters')
        require(not (Path.home()/'.drirc').exists(), 'unexpected per-user drirc')
        require(KNOB.read_text().strip() == '0', 'mode must start at 0')
        (out/'config').mkdir()
        write(out/'icd.json', dict(file_format_version='1.0.0',
            ICD=dict(library_path=str(HERE/'lib/libvulkan_asahi.so'), api_version='1.4.305')))
        monitor = Kmsg(out)
        monitor.mark()
        with (out/'results.jsonl').open('w', buffering=1) as ledger:
            for case in selected:
                failure_class = 'harness'
                require(time.monotonic() + 25 < deadline, 'insufficient case budget')
                monitor.mark()
                KNOB.write_text('3\n')
                require(KNOB.read_text().strip() == '3', 'research mode did not latch')
                directory = out/case['name']
                directory.mkdir()
                raw = DEP.read_text()
                (directory/'dep-before.txt').write_text(raw)
                before = counts(raw)
                e = environment(out)
                argv = list(map(str, [HERE/'bin/mesh-m1',
                    HERE/'shaders'/f'P{case["maximum"]}.mesh.spv', HERE/'shaders/count.frag.spv',
                    directory, case['vertices'], case['primitives'], case['maximum'], case['cookie']]))
                write(directory/'RUN.json', dict(argv=argv, case=case,
                    advertisement=ADVERTISEMENT.read_text(), environment={k: v for k, v in e.items()
                    if k.startswith(('HK_', 'MESA_', 'VK_', 'LD_', 'DRIRC_'))}))
                attempted += 1
                monitor.begin(attempted, 'native', case['cookie'])
                started = time.monotonic()
                with (directory/'stdout.log').open('w') as f:
                    proc = subprocess.Popen(argv, cwd=directory, env=e, stdout=f,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    monitor.attach(proc.pid)
                    while proc.poll() is None:
                        monitor.check()
                        require(time.monotonic()-started < 25, 'case watchdog expired; retain backing')
                        output_budget(directory)
                        log = (directory/'stdout.log').read_text(errors='replace')
                        if 'SUBMIT_BEGIN' in log:
                            failure_class = 'product'
                        require('IN_FLIGHT_WAIT_FAILED' not in log, 'uncertain GPU retirement')
                        time.sleep(.02)
                    monitor.mark()
                    monitor.complete(attempted, time.monotonic_ns())
                    monitor.attach(None)
                    code = proc.returncode
                    proc = None
                output_budget(directory)
                log = (directory/'stdout.log').read_text(errors='replace')
                if 'SUBMIT_BEGIN' in log:
                    failure_class = 'product'
                row = dict(case=case, directory=directory.name, exit_code=code, native_pass=False)
                rows.append(row)
                try:
                    require(code == 0 and log.count('SUBMIT_OK\n') == 1, 'probe/retirement failure')
                    retired += 1
                    row['readback'] = verify(directory, case)
                    write(directory/'READBACK.json', row['readback'])
                    require(row['readback']['verdict'] == 'PASS', 'pixel/buffer mismatch')
                    raw = DEP.read_text()
                    (directory/'dep-after.txt').write_text(raw)
                    row['audit'] = audit(log, case, plan['driver_revision'], before, counts(raw))
                    require(KNOB.read_text().strip() == '3', 'research mode changed during process')
                    row['native_pass'] = not REHEARSAL
                    row['verdict'] = 'PASS'
                except BaseException as error:
                    row.update(verdict='FAIL', reason=str(error))
                    raise
                finally:
                    ledger.write(json.dumps(row) + '\n')
                    ledger.flush()
                    os.fsync(ledger.fileno())
                print(case['name'] + ': PASS', flush=True)
        monitor.mark()
        KNOB.write_text('0\n')
        require(KNOB.read_text().strip() == '0', 'mode restoration failed')
        write(out/'VERDICT.json', result('PASS', mode_restored=0, faults=0))
        return 0
    except BaseException as error:
        retained = bool(proc and proc.poll() is None)
        fault = monitor.fault if monitor else None
        if retained:
            os.killpg(proc.pid, signal.SIGSTOP)
        if fault and fault['reason'].startswith('GPU/kernel fault'):
            failure_class = 'product'
        restored = None
        if not retained and not fault:
            try:
                KNOB.write_text('0\n')
                restored = 0 if KNOB.read_text().strip() == '0' else None
            except OSError:
                pass
        write(out/'VERDICT.json', result('FAIL', reason=str(error), fault_class=failure_class,
            retained=retained, kernel_fault=fault, mode_restored=restored))
        print('STOP: ' + str(error), flush=True)
        if retained:
            write(out/'RETAINED.json', dict(pid=proc.pid, reason=str(error), recovery='operator-owned; no retry'))
            os.kill(os.getpid(), signal.SIGSTOP)
            while True:
                signal.pause()
        return 1
    finally:
        if monitor:
            monitor.close()


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).resolve()))
