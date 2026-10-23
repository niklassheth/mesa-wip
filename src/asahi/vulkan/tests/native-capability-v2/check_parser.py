#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-owner check: admit the kit's exact data streams without GPU access."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kernel',type=Path,required=True)
    ap.add_argument('--commit',required=True)
    ap.add_argument('--mode',required=True,help='Rust Mode variant for the new research admission value')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parent
    assert re.fullmatch('[0-9a-f]{40}',a.commit)
    assert re.fullmatch('[A-Za-z][A-Za-z0-9_]*',a.mode)
    a.out.mkdir(parents=True,exist_ok=False)
    source=subprocess.check_output(['git','-C',str(a.kernel),'show',a.commit+':drivers/gpu/drm/asahi/g16_mesh.rs'])
    (a.out/'g16_mesh.rs').write_bytes(source)
    (a.out/'check.rs').write_text('''#![allow(dead_code)]
#[path="g16_mesh.rs"] mod parser;
fn main() {
    for path in std::env::args().skip(1) {
        let bytes=std::fs::read(&path).unwrap();
        let read=|va: u64| {
            let at: usize=va.checked_sub(0x1000000000)?.try_into().ok()?;
            Some(u32::from_le_bytes(bytes.get(at..at+4)?.try_into().unwrap()))
        };
        let accepted=matches!(parser::decode(0x1000000000,parser::Mode::MODE_VARIANT,read),Ok(Some(_)));
        assert_eq!(parser::decode(0x1000000000,parser::Mode::Disabled,|_|panic!("disabled read")),Ok(None));
        println!("{} {}",std::path::Path::new(&path).file_stem().unwrap().to_str().unwrap(),accepted);
    }
}
'''.replace('MODE_VARIANT',a.mode))
    subprocess.run(['rustc','--edition=2021','-Dwarnings',str(a.out/'check.rs'),'-o',str(a.out/'check')],check=True)
    index=root/'fixtures/INDEX.json';cases=json.loads(index.read_text())['cases']
    for c in cases:assert hashlib.sha256((root/'fixtures'/c['file']).read_bytes()).hexdigest()==c['sha256']
    output=subprocess.check_output([str(a.out/'check'),*[str(root/'fixtures'/c['file']) for c in cases]],text=True)
    rows=[dict(name=n,accepted=x=='true') for n,x in (line.split() for line in output.splitlines())]
    result=dict(kernel_source_commit=a.commit,parser_sha256=hashlib.sha256(source).hexdigest(),mode=a.mode,
                fixtures_catalog_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),submitted=False,
                all_fixtures_accepted=len(rows)==len(cases) and all(x['accepted'] for x in rows),rows=rows)
    (a.out/'ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output,end='');assert result['all_fixtures_accepted'],'research parser must admit every frozen shape'


if __name__=='__main__':main()
