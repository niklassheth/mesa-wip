#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect DATA and unordered mesh-instruction singletons from our own MSL.

No disassembly, surrounding instructions, addresses, ordering, or algorithm is
recorded. Register operands identify register fields, not their runtime values.
The two-byte-aligned singleton search is deliberately labeled as a candidate
census: it does not infer instruction boundaries by decoding helper sequences.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from container_data import Table, metadata, native


def sha(data):
    return hashlib.sha256(data).hexdigest()


def singletons(code):
    counts = Counter()
    for at in range(0, len(code) - 5, 2):
        word = int.from_bytes(code[at:at + 6], "little")
        key = word & 0xfff
        if key not in [0x357, 0x6d7] or (word >> 18) & 63 != 0x15:
            continue
        counts[word] += 1
    rows = []
    for word, count in sorted(counts.items()):
        row = dict(singleton_hex=word.to_bytes(6, "little").hex(),
                   occurrences=count, input_wait_mask=(word >> 12) & 63)
        if word & 0xfff == 0x357:
            row.update(family="control", A_halfword=(word >> 24) & 255,
                       B_halfword=(word >> 33) & 255,
                       flag42=(word >> 42) & 1, flag44=(word >> 44) & 1,
                       flag45=(word >> 45) & 1,
                       other_bits=hex(word & ~((63 << 12) | (255 << 24) |
                                               (255 << 33) | (1 << 42) |
                                               (3 << 44))))
        else:
            row.update(family="submit", source_gpr=(word >> 24) & 127,
                       release=(word >> 31) & 1, mode=(word >> 35) & 3,
                       other_bits=hex(word & ~((63 << 12) | (255 << 24) |
                                               (3 << 35))))
        rows.append(row)
    return rows


def inspect(path):
    raw = native(path.read_bytes())
    sections, _ = metadata(raw)
    result = {}
    for stage in ["object", "mesh"]:
        matches = [s for s in sections if s["name"] == "__" + stage]
        if not matches:
            continue
        assert len(matches) == 1
        section = matches[0]
        inner = raw[section["offset"]:section["offset"] + section["length"]]
        inner_sections, _ = metadata(inner)

        def data(segment, name):
            matches = [s for s in inner_sections if s["segment"] == segment and s["name"] == name]
            assert len(matches) == 1
            s = matches[0]
            return inner[s["offset"]:s["offset"] + s["length"]]

        md = data("__GPU_METADATA", "__compute")
        ld = data("__GPU_LD_MD", "__compute")
        main = Table(md).table(0)
        layout = Table(ld).table(8 if stage == "object" else 9)
        # FlatBuffer fields 3 (topology) and 14 (PointSize) are bytes.
        # Reading them as u32 accidentally includes an adjacent field. The
        # other reported spans are retained as data, not all scalar meanings:
        # e.g. field 0 is a relative vector pointer, not a hardware state word.
        fields = {str(i): (ld[layout.offset(i)] if stage == "mesh" and i in [3, 14]
                          else layout.value(i))
                  for i in range((layout.extent - 4) // 2)
                  if layout.offset(i) is not None}
        aux = main.vector(13) if main.offset(13) is not None else bytes()
        assert len(aux) % 4 == 0
        text = data("__TEXT", "__text")
        result[stage] = dict(metadata_sha256=sha(md), layout_sha256=sha(ld),
                             layout_fields=fields, metadata_gprs=main.value(0),
                             argument_words=main.value(1), frame=main.value(14),
                             auxiliary_words=list(struct.unpack("<" + "I" * (len(aux) // 4), aux)),
                             text_sha256=sha(text), text_bytes=len(text),
                             unordered_singleton_candidates=singletons(text))
        # Compiler DATA only; vector/table field interpretation stays provisional.
        (path.parent / (stage + "-layout.data")).write_bytes(ld)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    specs = {r["name"]: r for r in json.loads((args.root / "CASES.json").read_text())}
    compiles = json.loads((args.root / "COMPILE.json").read_text())
    assert len(compiles) == len(specs) and {c["name"] for c in compiles} == set(specs), "incomplete compilation inventory"
    rows = []
    for c in compiles:
        row = dict(case=specs[c["name"]], compiled=c["compiled"])
        folder = args.root / c["name"]
        if c["compiled"]:
            archive = folder / "probe-g16.metallib"
            assert sha(archive.read_bytes()) == c["archive_sha256"]
            row.update(archive_sha256=c["archive_sha256"], stages=inspect(archive))
        else:
            row["diagnostic"] = (folder / f"compile-{len(c['steps']) - 1}.log").read_text()
        rows.append(row)
    result = dict(scope=__doc__, gpu_execution=False,
                  compiler_version=(args.root / "COMPILER.txt").read_text(),
                  cases=rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Inspected {sum(c['compiled'] for c in rows)} compiled cases")


if __name__ == "__main__":
    main()
