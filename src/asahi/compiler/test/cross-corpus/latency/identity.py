#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hash every compiled program's instruction bytes per corpus entry, so two
compilers can be checked for byte-identical output (host-only).
Usage: identity.py ICD.json OUT.json corpus...

OUT.state.json holds a separate digest of each entry's APPLE9_INFO lines (the
hardware state a driver derives outside the program bytes), or null when the
driver predates them. Compare two runs with identity-compare.py."""
import sys, os, re, json, hashlib, subprocess, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import latmetrics as L
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from packed import Pack

def digest(text):
    h = hashlib.sha256(); n = 0
    for l in text.splitlines():
        if l.startswith('APPLE9_ASM_BEGIN') or l.startswith('APPLE9_PREAMBLE_ASM_BEGIN') or l.startswith('APPLE9_ASM_END'):
            h.update(l.encode()); n += 1
        elif l.startswith('APPLE9_ASM '):
            m = re.search(r'bytes=(\S+)', l)
            h.update(m.group(1).encode() if m else b'?'); n += 1
    return h.hexdigest(), n

def state_digest(text):
    lines = [l for l in text.splitlines() if l.startswith('APPLE9_INFO ')]
    return [hashlib.sha256('\n'.join(lines).encode()).hexdigest(), len(lines)] if lines else None

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('icd')
parser.add_argument('out')
parser.add_argument('corpora', nargs='+')
parser.add_argument('--host-shim', help='Host-only DRM shim to preload')
parser.add_argument('--jobs', type=int, default=12)
parser.add_argument('--keep-going', action='store_true',
                    help='Collect all failures; still exit unsuccessfully if any fail')
args = parser.parse_args()
icd, out = args.icd, args.out
env = L.env_for(icd)
if args.host_shim:
    env['LD_PRELOAD'] = os.path.abspath(args.host_shim)
res = {}
states = {}
failures = {}
def checked_digest(process, key, output, packed=False):
    text = output.decode(errors='replace')
    if packed:
        # Device initialization also dumps programs. They cannot stand in for
        # a deferred shader or a failed corpus entry.
        marker = 'CORPUS_BEGIN ' + key + '\n'
        if marker not in text or 'CORPUS_END 0\n' not in text.split(marker, 1)[-1]:
            raise RuntimeError(f'{key}: missing successful corpus receipt\n{text[-4000:]}')
        text = text.split(marker, 1)[1].split('CORPUS_END ', 1)[0]
    else:
        marker = 'Fossilize INFO: Creating Vulkan device took:'
        if marker not in text:
            raise RuntimeError(f'{key}: missing replay device receipt\n{text[-4000:]}')
        text = text.split(marker, 1)[1]
        replayed = re.findall(r'Playing back (\d+) (?:graphics|compute) pipelines took', text)
        # Linked pipelines may replay their library dependencies too.
        if sum(map(int, replayed)) < 1:
            raise RuntimeError(f'{key}: missing replayed pipeline\n{text[-4000:]}')
        # State-only graphics pipeline libraries legitimately have no shader
        # modules. Keep these outcomes distinct from emitted-program hashes.
        modules = re.findall(r'Playing back (\d+) shader modules took', text)
        if (process.returncode == 0 and modules and sum(map(int, modules)) == 0
                and 'APPLE9_ASM_BEGIN' not in text
                and 'APPLE9_PREAMBLE_ASM_BEGIN' not in text):
            return key, (hashlib.sha256(b'FOSSILIZE_STATE_ONLY').hexdigest(), 0)
    begins = len(re.findall(r'^APPLE9(?:_PREAMBLE)?_ASM_BEGIN', text, re.M))
    ends = len(re.findall(r'^APPLE9_ASM_END', text, re.M))
    result = digest(text) + (state_digest(text),)
    if process.returncode or not begins or begins != ends or not result[1]:
        raise RuntimeError(f'{key}: exit={process.returncode}, programs={begins}/{ends}\n{text[-4000:]}')
    return key, result
def collect_digest(process, key, output, packed=False):
    try:
        return checked_digest(process, key, output, packed)
    except RuntimeError as err:
        if not args.keep_going:
            raise
        return key, {'failure': str(err), 'exit': process.returncode}

def pack_entries(name):
    if name in ('cts', 'robust', 'timing-large-gs'):
        pack = Pack(L.R + '/artifacts/compiler-cross-corpus-20261006/corpus.sqlite3')
        if name == 'cts': return pack, [('timing-stratified-768', k) for k in pack.keys('timing-stratified-768')]
        if name == 'timing-large-gs': return pack, [('timing-large-gs', k) for k in pack.keys('timing-large-gs')]
        sel = json.loads(open(L.R + '/artifacts/dxvk-quality-20261006/robust-selection.json').read())['entries']
        return pack, [(e['population'], e['key']) for e in sel]
    pack = Pack(L.R + '/artifacts/compiler-cross-corpus-20261006/public.sqlite3')
    return pack, [(name, k) for k in pack.keys(name)]
def one_pack(args):
    key, frame = args
    p = subprocess.run(['/tmp/asahi-cross-corpus/compile-shader-packed'], input=frame,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, timeout=900)
    return collect_digest(p, key, p.stderr, packed=True)
def one_foz(args):
    foz, h = args
    p = subprocess.run(['/tmp/asahi-cross-corpus/lethal-tools/fossilize-replay-audit', '--num-threads', '1',
                        '--disable-signal-handler', '--pipeline-hash', h, foz], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    return collect_digest(p, h, p.stdout)
for name in args.corpora:
    with ThreadPoolExecutor(args.jobs) as ex:
        if name in ('cs2', 'lethal'):
            foz = L.R + ('/tmp/cs2-fossilize-20261004/cs2-capture.foz' if name == 'cs2' else
                         '/artifacts/compiler-cross-corpus-20261006/lethal-company/lethal-capture.foz')
            items = ex.map(one_foz, [(foz, h) for h in L.foz_hashes(foz, name == 'cs2')])
        else:
            pack, entries = pack_entries(name)
            items = ex.map(one_pack, [(k, pack.frame(p, k)[0]) for p, k in entries])
        rows = dict(items)
        failures[name] = {k: v for k, v in rows.items() if isinstance(v, dict)}
        res[name] = {k: tuple(v[:2]) for k, v in rows.items() if not isinstance(v, dict)}
        states[name] = {k: (v[2] if len(v) > 2 else None) for k, v in rows.items() if not isinstance(v, dict)}
    print(name, len(res[name]), 'passed', len(failures[name]), 'failed',
          'state-only', sum(v[1] == 0 for v in res[name].values()),
          'programs-lines', sum(v[1] for v in res[name].values()), flush=True)
    Path(out).write_text(json.dumps(res) + '\n')
    Path(out).with_suffix('.state.json').write_text(json.dumps(states) + '\n')
    Path(out + '.failures.json').write_text(json.dumps(failures, indent=2) + '\n')
if any(failures.values()):
    sys.exit(1)
