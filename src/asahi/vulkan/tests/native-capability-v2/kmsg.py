# SPDX-License-Identifier: MIT
"""One cheap kernel-log follower for the whole run."""
from collections import deque
import errno
import json
import os
import re
import select
import signal
import threading
import time

FAULT = re.compile(r'ETIMEDOUT|\bEIO\b|hang flight|GPU.*(?:fault|error|halt|timed out|timeout|reset)|(?:asahi|AGX).*(?:fault|timed out|timeout|halted|unrecoverable|resetting)|IOMMU.*\bfault\b|\b(?:Oops|SError)\b|\bBUG:|G16: command \d+ preparation failed:', re.I)


class Kmsg:
    def __init__(self, out):
        self.out = out; self.fd = os.open('/dev/kmsg', os.O_RDONLY|os.O_NONBLOCK)
        self.lock = threading.Lock(); self.active = {}; self.recent = deque(maxlen=256)
        self.lines = deque(maxlen=4096); self.roots = set()
        self.log = (out/'kernel.log').open('w', buffering=1)
        self.fault = None; self.sequence = None; self.pid = None; self.done = False
        self.drain()  # Includes existing boot faults: refuse a contaminated boot.
        self.thread = threading.Thread(target=self.follow, daemon=True); self.thread.start()

    def fail(self, reason, record=None):
        with self.lock:
            if self.fault is not None:return
            self.fault = dict(reason=reason, kmsg=record, observed_ns=time.monotonic_ns(),
                in_flight=list(self.active.values()), recent_completed=list(self.recent))
            (self.out/'FAULT.json').write_text(json.dumps(self.fault, indent=2)+'\n')
            pid = self.pid
        if pid:
            # SIGSTOP also retains backing during the dynamic loader/startup,
            # before the runner has installed its cooperative signal handlers.
            try:os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:pass

    def drain(self):
        while not self.fault:
            try:raw = os.read(self.fd, 65536)
            except BlockingIOError:return
            except OSError as error:
                self.fail('kmsg records lost' if error.errno == errno.EPIPE else str(error));return
            if not raw:return
            try:
                self.log.write(raw.decode(errors='replace'))
                header, message = raw.decode(errors='replace').split(';', 1)
                fields = header.split(',');seq, micros = int(fields[1]), int(fields[2])
            except (ValueError, IndexError):
                self.fail('unparseable kmsg record');return
            if self.sequence is not None and seq != self.sequence+1:
                self.fail('kmsg sequence gap', dict(previous=self.sequence, sequence=seq));return
            self.sequence = seq
            record = dict(sequence=seq, timestamp_ns=micros*1000, message=message.rstrip())
            with self.lock:
                self.lines.append(record)
                if 'render root ' in message or 'attached firmware root ' in message:
                    self.roots.add(message.rstrip())
            if FAULT.search(message):self.fail('GPU/kernel fault: global stop', record)

    def follow(self):
        while not self.done and not self.fault:
            if select.select([self.fd], [], [], .05)[0]:self.drain()

    def attach(self, pid):
        with self.lock:self.pid = pid
        self.check()

    def check(self):
        if self.fault:raise RuntimeError(self.fault['reason'])

    def begin(self, sequence, arm, cookie):
        with self.lock:
            if self.fault:raise RuntimeError(self.fault['reason'])
            self.active[sequence] = dict(sequence=sequence, arm=arm, cookie=cookie,
                submit_ns=time.monotonic_ns(), kmsg_before=self.sequence)

    def complete(self, sequence, completed_ns):
        with self.lock:
            row = self.active.pop(sequence);row['complete_ns'] = completed_ns
            self.recent.append(row)

    def capture_log(self):
        with self.lock:
            return '\n'.join(sorted(self.roots))+'\n'+'\n'.join(
                r['message'] for r in self.lines if 'G16: dump render at=' in r['message'])+'\n'

    def close(self):
        self.done = True;self.thread.join(timeout=1);os.close(self.fd);self.log.close()
