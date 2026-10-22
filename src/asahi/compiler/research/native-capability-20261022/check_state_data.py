#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-extract classified T8132 Metal VDM DATA, then check its field model.

The existing reader never follows shader targets. These retained captures are
separate from the new offline MSL sweep; matching models do not join their runs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


def packet(data):
    """Bound the first state/draw at its terminator or next state record."""
    words = struct.unpack("<" + "I" * (len(data) // 4), data[:len(data) // 4 * 4])
    start = words.index(0x40000d2c)
    at, draws = start + 14, []
    while at < len(words):
        word = words[at]
        if word >> 29 == 0 and word & 0xff00:
            at += 2
        elif word >> 29 == 1:
            at += 1
        elif word & 0xff0000ff == 0x70000000:
            draws.append(dict(word_offset=at, words=[hex(w) for w in words[at:at + 7]]))
            at += 7
        elif word in [0xc0000000, 0x40000d2c]:
            assert len(draws) == 1
            return dict(word_offset=start, words=[hex(w) for w in words[start:start + 14]],
                        object_packets=draws, boundary_word_offset=at,
                        boundary="terminator" if word == 0xc0000000 else "next_state")
        else:
            raise ValueError(f"unsupported VDM packet {word:#x}")
    raise ValueError("missing VDM terminator")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("data_reader", a.reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reader.OUT = a.out / "re-extracted"
    reader.packet = packet
    rows = reader.xmat()
    checked, bad = [], []
    for row in rows:
        d = row["definition"]
        n = 4 + d["nva"] + (d["npa"] > 0) + (d["topo"] == "p")
        cull = "CU" in d["flags"]
        primitive = {"p": 0, "l": 1, "t": 6}[d["topo"]]
        expected = {0: 0x40000d2c, 1: n | ((n + 1) << 8),
                    2: 0xd0 if d["npa"] else 0x10, 3: 0, 4: 0,
                    5: 0x1000000, 7: 0x1000000,
                    10: (d["mp"] - 1) | (((d["mp"] + 15) // 16 - 1) << 16) | (int(cull) << 30),
                    11: 0x400800 | d["mt"], 12: d["mp"] * d["npa"],
                    13: 1 + (d["mv"] + 3) // 4 * n + ((d["mp"] + 3) // 4 if cull else 0)}
        actual = [int(w, 16) for w in row["vdm"]["words"]]
        errors = {i: dict(expected=e, actual=actual[i]) for i, e in expected.items() if actual[i] != e}
        for draw in row["vdm"]["object_packets"]:
            if int(draw["words"][0], 16) != 0x70000000 | (primitive << 8):
                errors["draw"] = draw["words"][0]
        if errors:
            bad.append(dict(name=d["name"], session=row["session"], errors=errors))
        checked.append(dict(name=d["name"], session=row["session"], repeat=row["repeat"],
                            trace=row["trace"], line=row["line"], definition=d,
                            words=row["vdm"]["words"], draws=row["vdm"]["object_packets"]))
    result = dict(scope=__doc__, observed_chip_id="0x8132", captures=len(rows),
                  cases=len({(r["session"], r["name"]) for r in rows}),
                  reader_sha256=hashlib.sha256(a.reader.read_bytes()).hexdigest(),
                  source_files=reader.SOURCES, checks=checked, mismatches=bad,
                  shader_targets_followed=False, new_device_access=False)
    (a.out / "STATE-DATA.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["captures", "cases", "mismatches"]}))
    assert not bad


if __name__ == "__main__":
    main()
