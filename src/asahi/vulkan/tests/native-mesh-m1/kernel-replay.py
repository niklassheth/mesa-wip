#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay captured native packets through the paired kernel's actual parser."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--kernel', type=Path, required=True)
p.add_argument('--revision', required=True)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir()
source = subprocess.check_output(['git', 'show', a.revision + ':drivers/gpu/drm/asahi/g16_mesh.rs'], cwd=a.kernel)
(a.out/'g16_mesh.rs').write_bytes(source)
fixture = '#[allow(dead_code)] mod g16_mesh;\nuse g16_mesh::{Mode,Profile,DecodeError};\n'
rows = []
for record in sorted(a.records.glob('P*/stream.bin')):
    data = record.read_bytes()
    name = record.parent.name.replace('-', '_')
    (a.out/(name+'.bin')).write_bytes(data)
    qualified = 'Ok(Some(Profile {payload:0}))' if name.startswith('P17_') else 'Err(DecodeError::Unsupported)'
    fixture += f'''
#[test] fn {name.lower()}() {{
    let bytes = include_bytes!("{name}.bin");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for base in [0x1004000000u64, 0x1004003ffc] {{
        let decode = |mode, count| g16_mesh::decode(base,mode,|va| {{
            let index = usize::try_from(va.checked_sub(base)? / 4).ok()?;
            words.get(..count)?.get(index).copied()
        }});
        assert_eq!(decode(Mode::Research,words.len()),Ok(Some(Profile {{payload:0}})));
        assert_eq!(decode(Mode::Qualified,words.len()),{qualified});
        for count in 0..words.len() {{ assert!(decode(Mode::Research,count).is_err()); }}
    }}
}}
'''
    rows.append(dict(case=record.parent.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
assert len(rows) == 6
(a.out/'replay.rs').write_text(fixture)
commands = [['rustc', '--edition=2021', '--test', '-Dwarnings', str(a.out/'replay.rs'), '-o', str(a.out/'replay')], [str(a.out/'replay'), '--nocapture']]
with (a.out/'stdout.log').open('w') as f:
    for command in commands:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
(a.out/'RESULT.json').write_text(json.dumps(dict(verdict='PASS',gpu_execution=False,
    kernel_revision=a.revision, parser_sha256=hashlib.sha256(source).hexdigest(),cases=rows,
    checks='research accepts all six; qualified rejects P15/P16, accepts P17; two relocations and every truncated prefix rejected', commands=commands),indent=2)+'\n')
print('Six captured streams: paired kernel parser PASS')
