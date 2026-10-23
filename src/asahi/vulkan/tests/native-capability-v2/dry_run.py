#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""All shapes traverse the frozen carrier; host completion cannot qualify GPU."""
import argparse
import json
from pathlib import Path
from transport import Runner

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--sysroot',type=Path,default=Path('/home/nsheth/Projects/asahi/tmp/mini-sysroot'))
a=p.parse_args()
root=a.root.resolve();out=a.out.resolve()
c=json.loads((root/'CATALOG.json').read_text())
r=Runner(root,out,c,1,a.sysroot);rows=[]
try:
    r.load_all();r.start()
    for seq,arm in enumerate(c['arms'],1):
        r.submit(arm,0,seq,0x7a000000+seq)
        row=r.poll(0,1000000000)
        assert row is not None and row['code']==2,row
        row.update(arm=arm['name'],status='HOST_ONLY_NO_GPU_QUALIFICATION')
        rows.append(row)
        r.dump(0,out/f'evidence-{seq:03d}');r.release(0)
    r.close()
except BaseException:
    r.abandon();raise
(out/'DRY-RUN.json').write_text(json.dumps(dict(arms=len(rows),gpu_execution=False,qualified_trials=0,rows=rows),indent=2)+'\n')
print(len(rows),'arms: code/launch/state/PPP identity passed; 0 GPU trials')
