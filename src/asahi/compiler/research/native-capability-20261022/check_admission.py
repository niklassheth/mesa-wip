#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Demonstrate revision-2 parser restrictions with non-submitted DATA fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

PIN = "7b61c4f84c08e08c09ecfd97febc120a2597bd03"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "-C", str(a.kernel), "show",
                                      PIN + ":drivers/gpu/drm/asahi/g16_mesh.rs"])
    (a.out / "g16_mesh.rs").write_bytes(source)
    (a.out / "check.rs").write_text('''#![allow(dead_code)]
#[path="g16_mesh.rs"] mod parser;
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let read = |va: u64| {
            let at: usize = va.checked_sub(0x1000000000)?.try_into().ok()?;
            Some(u32::from_le_bytes(bytes.get(at..at+4)?.try_into().unwrap()))
        };
        let a = matches!(parser::decode(0x1000000000, parser::Mode::SingleParent, read), Ok(Some(_)));
        let b = matches!(parser::decode(0x1000000000, parser::Mode::MultipleParents, read), Ok(Some(_)));
        assert_eq!(a, b);
        assert_eq!(parser::decode(0x1000000000, parser::Mode::Disabled, |_| panic!("disabled read")), Ok(None));
        println!("{} {}", std::path::Path::new(&path).file_stem().unwrap().to_str().unwrap(), a);
    }
}
''')
    cases = [("baseline", 3, 1, 4, 0, 6), ("V4P2", 4, 2, 4, 0, 6),
             ("V16P16", 16, 16, 4, 0, 6), ("V64P128", 64, 128, 4, 0, 6),
             ("V128P256", 128, 256, 4, 0, 6), ("V256P512", 256, 512, 4, 0, 6),
             ("points", 4, 4, 5, 0, 0), ("lines", 16, 16, 4, 0, 1),
             ("primitive_data", 4, 2, 5, 4, 6)]
    rows = []
    for name, v, p, n, pc, topology in cases:
        state = [0x40000d2c, n | ((n + 1) << 8), 0xd0 if pc else 0x10,
                 0, 0, 0x1000000, 0x100, 0x1000000, 0x200, 1,
                 (p - 1) | (((p + 15) // 16 - 1) << 16), 0x400810, p * pc,
                 1 + ((v + 3) // 4) * n]
        draw = [0x70000000 | (topology << 8), 16, 1, 1, 16, 1, 1]
        words = state + draw + [0xc0000000]
        data = struct.pack("<" + "I" * len(words), *words)
        (a.out / (name + ".bin")).write_bytes(data)
        rows.append(dict(name=name, vertices=v, primitives=p, planes=n,
                         primitive_components=pc, topology=topology,
                         words=[hex(w) for w in words], sha256=hashlib.sha256(data).hexdigest()))
    subprocess.run(["rustc", "--edition=2021", "-Dwarnings", str(a.out / "check.rs"),
                    "-o", str(a.out / "check")], check=True)
    output = subprocess.check_output([str(a.out / "check"),
                                      *[str(a.out / (r["name"] + ".bin")) for r in rows]], text=True)
    status = {name: answer == "true" for name, answer in (line.split() for line in output.splitlines())}
    for row in rows:
        row["accepted_modes_1_and_2"] = status[row["name"]]
        assert status[row["name"]] == (row["name"] == "baseline")
    result = dict(scope=__doc__, submitted=False, kernel_commit=PIN,
                  parser_sha256=hashlib.sha256(source).hexdigest(), cases=rows)
    (a.out / "ADMISSION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(output, end="")


if __name__ == "__main__":
    main()
