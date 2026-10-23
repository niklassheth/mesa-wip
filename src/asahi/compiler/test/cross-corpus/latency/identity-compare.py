#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare two identity.py runs: program bytes and APPLE9_INFO hardware state.
Usage: identity-compare.py BASE.json CAND.json [--json OUT] [--list N]

Prints per corpus the entries whose programs changed, whose state changed, and
whose state changed while the programs stayed identical (these are invisible
to the program digest alone). State is "unavailable" for an entry when only one
run printed APPLE9_INFO (a driver that predates it); entries with no final
shader in either run (e.g. deferred to a draw-time variant) count as "no
state". Exit status: 0 when nothing changed and state was compared everywhere,
1 when anything changed, 2 when nothing changed but state was unavailable for
some entries."""
import sys, json, argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('base')
p.add_argument('cand')
p.add_argument('--json', help='write the summary here')
p.add_argument('--list', type=int, default=10, help='entries to list per category and corpus')
a = p.parse_args()

def load(path):
    progs = json.loads(Path(path).read_text())
    sp = Path(path).with_suffix('.state.json')
    states = json.loads(sp.read_text()) if sp.exists() else {}
    fp = Path(path + '.failures.json')
    fails = json.loads(fp.read_text()) if fp.exists() else {}
    return progs, states, fails

bp, bs, bf = load(a.base)
cp, cs, cf = load(a.cand)
summary = {}
for corpus in sorted(set(bp) | set(cp)):
    b, c = bp.get(corpus, {}), cp.get(corpus, {})
    common = sorted(set(b) & set(c))
    prog = [k for k in common if b[k] != c[k]]
    state, state_only, unavailable, no_state = [], [], [], []
    for k in common:
        x, y = bs.get(corpus, {}).get(k), cs.get(corpus, {}).get(k)
        if x is None and y is None:
            # State-only pipeline libraries, or shaders deferred to draw time.
            if b[k][1] or c[k][1]:
                no_state.append(k)
            continue
        if x is None or y is None:
            unavailable.append(k)
            continue
        if x != y:
            state.append(k)
            if b[k] == c[k]:
                state_only.append(k)
    s = dict(entries=len(common), programs_changed=len(prog), state_changed=len(state),
             state_only_changed=len(state_only), state_unavailable=len(unavailable), no_state=len(no_state),
             only_base=len(set(b) - set(c)), only_cand=len(set(c) - set(b)),
             failures_base=sorted(bf.get(corpus, {})), failures_cand=sorted(cf.get(corpus, {})),
             state_only_keys=state_only)
    summary[corpus] = s
    print(f"{corpus}: {len(common)} entries, programs changed {len(prog)}, state changed {len(state)} "
          f"(with identical programs {len(state_only)}), state unavailable {len(unavailable)}, "
          f"no state {len(no_state)}, "
          f"only in base {s['only_base']}, only in candidate {s['only_cand']}, "
          f"failures {len(s['failures_base'])} -> {len(s['failures_cand'])}")
    for k in state_only[:a.list]:
        print(f'  state only: {k}')

changed = any(s['programs_changed'] or s['state_changed'] or s['only_base'] or s['only_cand'] or
              s['failures_base'] != s['failures_cand'] for s in summary.values())
unavailable = any(s['state_unavailable'] for s in summary.values())
total = dict(changed=changed, state_compared=not unavailable, corpora=summary)
if a.json:
    Path(a.json).write_text(json.dumps(total, indent=1) + '\n')
print('CHANGED' if changed else 'IDENTICAL' + ('' if not unavailable else ' (programs only; state unavailable)'))
sys.exit(1 if changed else 2 if unavailable else 0)
