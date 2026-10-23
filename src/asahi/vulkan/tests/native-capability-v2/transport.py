# SPDX-License-Identifier: MIT
"""One persistent runner process. The host-shim path cannot qualify GPU work."""
import collections
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time


class Runner:
    def __init__(self, root, out, catalog, lanes=4, shim=None, monitor=None, deadline=None):
        self.root, self.out, self.shim = Path(root), Path(out), shim
        self.monitor=monitor
        self.protocol_deadline=deadline
        self.arms = catalog['arms']; self.lanes = lanes
        self.by_name = {a['name']: a for a in self.arms}
        self.out.mkdir(parents=True, exist_ok=True)
        icd = self.out/'icd.json'
        icd.write_text(json.dumps(dict(file_format_version='1.0.0', ICD=dict(
            library_path=str(self.root/'lib/libvulkan_asahi.so'), api_version='1.4.305'))))
        env = dict(catalog['environment'], VK_DRIVER_FILES=str(icd), MESA_VK_ENABLE_SUBMIT_THREAD='false')
        command = [str(self.root/'runner'), str(lanes), str(self.root/'shaders/BASE.mesh.spv'),
                   str(self.root/'shaders/depth-color.frag.spv')]
        clean = {k:v for k,v in os.environ.items() if not k.startswith(
            ('AGX_', 'ASAHI_', 'HK_', 'MESA_', 'VK_', 'MULTI_', 'MESH_', 'CROSS_', 'LD_'))}
        if shim:
            trace = self.out/'shim'; trace.mkdir()
            env.update(LD_PRELOAD=str(self.root/'lib/libasahi_host_trace_drm_shim.so'),
                       LD_LIBRARY_PATH=str(Path(shim)/'usr/lib64'), ASAHI_HOST_TRACE_DIR=str(trace))
            command = ['qemu-aarch64-static', '-L', str(shim), *[x for k,v in env.items() for x in ['-E', k+'='+v]], *command]
        else:
            clean.update(env)
        self.proc = subprocess.Popen(command, env=clean, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if monitor:monitor.attach(self.proc.pid)
        self.tail = collections.deque(maxlen=256)
        self.stderr_bytes = 0
        self.log_thread = threading.Thread(target=self._stderr, daemon=True); self.log_thread.start()
        self.buffer = bytearray()
        self.ready = self.read(timeout=60)
        assert self.ready[0] == 'READY', self.ready
        self.pid = int(self.ready[1])
        self.identity = self.decode_identity(self.ready)
        (self.out/'IDENTITY.json').write_text(json.dumps(self.identity,indent=2)+'\n')
        self.loaded = set()

    @staticmethod
    def decode_identity(ready):
        assert len(ready)==7 and ready[0]=='READY',ready
        identity=dict(zip(['renderer','driver','info'],[bytes.fromhex(x).decode() for x in ready[3:6]]))
        identity['version']=int(ready[6])
        assert identity['renderer']=='Apple M4 (G16G B1)' and 'Honeykrisp' in identity['driver'] and 'git-e08fa6cf73' in identity['info'],identity
        identity['source']='synchronized READY; checked Vulkan properties'
        return identity

    def _stderr(self):
        with (self.out/'runner.log').open('wb',buffering=0) as log:
            while block := self.proc.stderr.read(4096):
                self.stderr_bytes += len(block); self.tail.append(block)
                # Bounded startup log; retain the tail separately on exit.
                if log.tell() < 8*1024**2:
                    log.write(block[:8*1024**2-log.tell()])
            (self.out/'runner-tail.log').write_bytes(b''.join(self.tail))

    def read(self, timeout=30):
        deadline = time.monotonic()+timeout
        while b'\n' not in self.buffer:
            if self.monitor:self.monitor.check()
            if self.protocol_deadline is not None and time.monotonic()>=self.protocol_deadline:
                raise TimeoutError('campaign deadline during protocol; retain backing')
            left = deadline-time.monotonic()
            if left <= 0:raise TimeoutError('runner protocol timeout; backing may remain live')
            if not select.select([self.proc.stdout],[],[],min(left,.05))[0]:continue
            data = os.read(self.proc.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f'runner exited {self.proc.poll()}: '+b''.join(self.tail)[-4000:].decode(errors='replace'))
            self.buffer.extend(data)
            if len(self.buffer) > 1024*1024:
                raise RuntimeError('runner protocol overflow')
        line, _, rest = self.buffer.partition(b'\n'); self.buffer = bytearray(rest)
        words = line.decode().split()
        if words and words[0] == 'ERROR':
            raise RuntimeError(' '.join(words))
        return words

    def command(self, *parts, timeout=30):
        text = ' '.join(map(str, parts))+'\n'
        assert len(text) < 16384 and text.count('\n') == 1
        self.proc.stdin.write(text.encode())
        return self.read(timeout)

    def load(self, arm):
        i = len(self.loaded)
        if arm.get('compute'):
            row=self.command('COMPUTE',i,self.root/'shaders'/arm['spv'],self.root/'exports'/arm['original'],self.root/'exports'/arm['patched'],self.root/'oracles'/f"{arm['id']:03d}.gz",timeout=180)
            assert row==['LOADED',str(i)],row
            arm['runner_id']=i;self.loaded.add(arm['name']);return
        mesh = arm.get('upload_mesh', arm['programs']['mesh'])
        row = self.command('LOAD', i, self.root/'exports'/mesh['file'],
            self.root/'exports'/arm['programs']['object']['file'], self.root/'oracles'/f"{arm['id']:03d}.gz",
            *arm['api_grid'], int(arm['cross_family'] == 'direct'), int(arm.get('uniform_nonce_word') == 6), timeout=180)
        assert row == ['LOADED', str(i)], row
        arm['runner_id'] = i; self.loaded.add(arm['name'])

    def load_all(self, check=None):
        assert not self.loaded
        for i, arm in enumerate(self.arms):
            if check: check()
            self.load(arm)
            if i % 32 == 0 or i+1 == len(self.arms):
                print('uploaded', i+1, '/', len(self.arms), flush=True)

    def start(self):
        row = self.command('START'); assert row[0] == 'STARTED', row
        assert int(row[1]) == len(self.arms) == len(self.loaded)
        return row

    def submit(self, arm, slot, sequence, cookie):
        row = self.command('SUBMIT', arm['runner_id'], slot, sequence, cookie)
        assert row[:3] == ['SUBMITTED', str(slot), str(sequence)], row
        if row[3] != '0':
            raise RuntimeError('Vulkan submit failed: '+' '.join(row))
        return int(row[4])

    def poll(self, slot, timeout_ns=0):
        row = self.command('POLL', slot, timeout_ns, timeout=15)
        assert row[0] in ['PENDING', 'RESULT', 'FAULT','COMPUTE_RESULT'], row
        if row[0] == 'FAULT':
            raise RuntimeError('Vulkan completion failed: '+' '.join(row))
        if row[0] == 'PENDING':
            return None
        if row[0]=='COMPUTE_RESULT':
            assert len(row)==17,row
            return dict(slot=int(row[1]),sequence=int(row[2]),code=int(row[3]),q_differences=int(row[4]),first_q_difference=int(row[5]),q_crc32=row[6],submit_ns=int(row[7]),complete_ns=int(row[8]),fresh=int(row[9]),sample_differences=int(row[10]),timestamp_valid=bool(int(row[11])),ticks=int(row[12]),timestamp_period=float(row[13]),gpu_ns=int(row[12])*float(row[13]),timestamp_bracket=bool(int(row[14])),gpu_start_tick=int(row[15]),gpu_end_tick=int(row[16]))
        result = dict(slot=int(row[1]), sequence=int(row[2]), code=int(row[3]),
                    q_differences=int(row[4]), first_q_difference=int(row[5]),
                    q_crc32=row[6], pixels_crc32=row[7], submit_ns=int(row[8]),
                    complete_ns=int(row[9]), ordinal=int(row[10]))
        if len(row) == 21:
            result.update(pixels_exact=bool(int(row[11])), samples_crc32=row[12],
                          samples_min=[int(row[i]) for i in [13,15,17,19]],
                          samples_max=[int(row[i]) for i in [14,16,18,20]])
        else: assert len(row) == 11, row
        return result

    def dump(self, slot, path):
        path.mkdir(parents=True, exist_ok=False)
        assert self.command('DUMP', slot, path) == ['DUMPED', str(slot)]

    def release(self, slot):
        assert self.command('RELEASE', slot) == ['RELEASED', str(slot)]

    def close(self):
        assert self.command('EXIT') == ['BYE']
        assert self.proc.wait(timeout=10) == 0
        self.log_thread.join(timeout=5)

    def abandon(self):
        # Explicit host shim exception: there is no GPU-owned backing to retain.
        if self.shim:
            self.proc.kill(); self.proc.wait(timeout=10)
        else:
            try:os.kill(self.proc.pid,signal.SIGSTOP)
            except ProcessLookupError:pass
        self.log_thread.join(timeout=.1)
