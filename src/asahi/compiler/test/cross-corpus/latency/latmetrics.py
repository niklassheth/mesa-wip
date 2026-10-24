# SPDX-License-Identifier: MIT
"""Waited-load distance metrics over AGX_APPLE9_ASM listings (host-only)."""
import re, os, subprocess, collections
R = '/home/nsheth/Projects/asahi'
_src = open(R + '/mesa-isa-probe/src/asahi/compiler/agx_apple9_ir.h').read()
_body = _src[_src.index('enum agx_apple9_vir_opcode'):]
OPS = re.findall(r'AGX_APPLE9_VIR_([A-Z0-9_]+)', _body[:_body.index('};')])

def env_for(icd, asm=True):
    e = dict(os.environ, LD_PRELOAD=R + '/tmp/cs2-pressure-20261005/vertex-analysis/libhost-noop.so',
             VK_DRIVER_FILES=icd, HK_EXPERIMENTAL_APPLE9='1', HK_APPLE9_EAGER_VARIANTS='1',
             MESA_SHADER_CACHE_DISABLE='true', VK_LOADER_LAYERS_DISABLE='~implicit~',
             CORPUS_DIRECT_PRERAST='1', FOSSILIZE_IGNORE_LAYOUT_SUPPORT='1')
    if asm: e['AGX_APPLE9_ASM'] = '1'
    return e

def blocks(text):
    out, cur = [], None
    for l in text.splitlines():
        if l.startswith('APPLE9_ASM_BEGIN'): cur = []
        elif l.startswith('APPLE9_ASM_END') and cur is not None: out.append(cur); cur = None
        elif cur is not None and l.startswith('APPLE9_ASM '): cur.append(l)
    return out

def parse(block):
    I = []
    for l in block:
        m = dict(re.findall(r'(\w+)=(\S*)', l))
        I.append(dict(op=OPS[int(m['op'])], dst=m['dst'], src=[s for s in m['src'].split(',') if s],
                      wait=int(m['wait'], 0), prod=int(m['producer'], 0), imm=int(m['imm'], 0)))
    return I

def regs(I):
    return 1 + max([int(x[1:]) for i in I for x in [i['dst']] + i['src'] if re.fullmatch(r'r\d+', x)] or [-1])

def analyze(I, kinds=('DEVICE_LOAD',)):
    t = collections.Counter()
    for i, a in enumerate(I):
        if a['op'] not in kinds or not a['prod']: continue
        t['loads'] += 1; w = None
        for j in range(i + 1, min(i + 400, len(I))):
            b = I[j]; ws = [b['wait']] if b['wait'] else []
            if b['op'] == 'WAIT': ws = [k for k in range(1, 7) if b['imm'] >> (k - 1) & 1]
            if a['prod'] in ws: w = j; break
            if b['op'] in kinds: w = None; break
        if w is None: continue
        t['dist_sum'] += w - i
        if w - i > 2: continue
        t['short'] += 1
        for k in range(w, min(w + 40, len(I))):
            if I[k]['op'] in kinds:
                n0 = int(a['dst'][1:]) if a['dst'].startswith('r') else -9
                tn = {f'r{n0 + c}' for c in range(4)}
                for q in range(w, k):
                    if set(I[q]['src']) & tn and I[q]['dst'].startswith('r'):
                        d0 = int(I[q]['dst'][1:]); tn |= {f'r{d0 + c}' for c in range(4)}
                t['dep' if set(I[k]['src']) & tn else 'indep'] += 1; break
    return t

def foz_hashes(foz, sample):
    # A sampled capture may carry a fixed list of replayable hashes beside it
    # (the Steam CS2 cache replays only ~20% of its pipelines on the host).
    fixed = os.path.join(os.path.dirname(os.path.realpath(foz)), 'identity-hashes.txt')
    if sample and os.path.exists(fixed):
        return open(fixed).read().split()
    hs = []
    for tag, limit in [(6, 128), (7, 32)]:
        h = sorted(subprocess.check_output([R + '/tmp/fossilize-build-x86/cli/fossilize-list', foz, '--tag', str(tag)], text=True).split())
        hs += [h[i * len(h) // limit] for i in range(limit)] if sample and len(h) > limit else h
    return hs

def foz_programs(icd, foz, hashes, threads=12):
    from concurrent.futures import ThreadPoolExecutor
    env = env_for(icd)
    def one(h):
        p = subprocess.run([ '/tmp/asahi-cross-corpus/lethal-tools/fossilize-replay-audit', '--num-threads', '1',
                            '--disable-signal-handler', '--pipeline-hash', h, foz], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        return [(h, parse(b)) for b in blocks(p.stdout.decode(errors='replace'))]
    with ThreadPoolExecutor(threads) as ex:
        for res in ex.map(one, hashes):
            yield from res

def pack_programs(icd, name):
    import sys; sys.path.insert(0, R + '/mesa-isa-probe/src/asahi/compiler/test/cross-corpus')
    from packed import Pack
    pack = Pack(R + '/artifacts/compiler-cross-corpus-20261006/public.sqlite3'); keys = pack.keys(name)
    env = env_for(icd)
    for off in range(0, len(keys), 32):
        frames = b''.join(pack.frame(name, k)[0] for k in keys[off:off + 32])
        p = subprocess.run(['/tmp/asahi-cross-corpus/compile-shader-packed'], input=frames,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, timeout=600)
        for b in blocks(p.stderr.decode(errors='replace')):
            yield name, parse(b)
