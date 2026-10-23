#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Backend CPU time (sum of CORPUS_TIME) per corpus for an instrumented
compiler snapshot. Usage: ctime.py ICD.json corpus... (host-only, serial)."""
import sys, os, re, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import latmetrics as L
sys.path.insert(0, L.R + '/mesa-isa-probe/src/asahi/compiler/test/cross-corpus'); from packed import Pack
icd = sys.argv[1]; env = L.env_for(icd, asm=False)
def frames(name):
    if name in ('cts', 'timing-large-gs', 'robust'):
        pack = Pack(L.R + '/artifacts/compiler-cross-corpus-20261006/corpus.sqlite3')
        if name == 'robust':
            sel = json.loads(open(L.R + '/artifacts/dxvk-quality-20261006/robust-selection.json').read())['entries']
            ent = [(e['population'], e['key']) for e in sel]
        else:
            pop = 'timing-stratified-768' if name == 'cts' else name
            ent = [(pop, k) for k in pack.keys(pop)]
    else:
        pack = Pack(L.R + '/artifacts/compiler-cross-corpus-20261006/public.sqlite3'); ent = [(name, k) for k in pack.keys(name)]
    for off in range(0, len(ent), 32):
        yield b''.join(pack.frame(p, k)[0] for p, k in ent[off:off + 32])
for name in sys.argv[2:]:
    ns = create = 0
    for fr in frames(name):
        p = subprocess.run(['taskset', '-c', '12', '/tmp/asahi-cross-corpus/compile-shader-packed'], input=fr,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, timeout=3600)
        err = p.stderr.decode(errors='replace')
        ns += sum(int(x) for x in re.findall(r'CORPUS_TIME stage=\d+ cpu_ns=(\d+)', err))
        create += sum(int(x) for x in re.findall(r'CORPUS_CREATE result=-?\d+ cpu_ns=(\d+)', err))
    print(name, 'backend_s', round(ns / 1e9, 3), 'create_s', round(create / 1e9, 3),
          'frontend_share', round(1 - ns / create, 3) if create else None, flush=True)
