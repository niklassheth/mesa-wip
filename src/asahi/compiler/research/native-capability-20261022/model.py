#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check provisional compiler-DATA field models against independent MSL cases."""
import argparse
import json
from pathlib import Path


def aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inspection", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    data = json.loads(args.inspection.read_text())
    checks, rejected, mismatches = [], [], []
    for row in data["cases"]:
        c = row["case"]
        if not row["compiled"]:
            rejected.append(dict(name=c["name"], diagnostic=row["diagnostic"]))
            continue
        topology = {"point": 0, "line": 1, "triangle": 2}[c["topology"]]
        primitive_components = 4 * c["primitive_varyings"] + c["primitive_scalars"]
        point_size = topology == 0 and c["point_size"]
        slots = 4 + 4 * c["vertex_varyings"] + (primitive_components > 0) + point_size
        expected = {"1": c["vertices"], "2": c["primitives"], "3": topology,
                    "4": slots, "6": 16 + aligned(c["vertices"], 4) * 4 * slots,
                    "8": primitive_components * 4, "14": int(point_size),
                    "22": c["threads"], "28": aligned(c["primitives"] * (topology + 1), 16)}
        actual = row["stages"]["mesh"]["layout_fields"]
        bad = {k: dict(expected=v, actual=actual.get(k, 0))
               for k, v in expected.items() if actual.get(k, 0) != v}
        if c["payload"]:
            obj = row["stages"]["object"]["layout_fields"]
            for k, v in {"2": c["payload"], "4": c["threads"]}.items():
                if obj.get(k, 0) != v:
                    bad["object." + k] = dict(expected=v, actual=obj.get(k, 0))
        if bad:
            mismatches.append(dict(name=c["name"], fields=bad))
        census = row["stages"]["mesh"]["unordered_singleton_candidates"]
        controls = [u for u in census if u["family"] == "control"]
        checks.append(dict(name=c["name"], mesh_fields=expected,
                           inferred_local_bytes=expected["6"] + expected["28"],
                           application_output_bytes=aligned(c["vertices"], 4) *
                           (16 + 16 * c["vertex_varyings"] + (4 if point_size else 0)) +
                           c["primitives"] * primitive_components * 4,
                           control_flags=sorted({f"{u['flag45']}{u['flag44']}{u['flag42']}" for u in controls}),
                           submit_modes=sorted({u["mode"] for u in census if u["family"] == "submit"})))
    result = dict(scope="Compiler metadata correlations only; no GPU execution or live launch capture",
                  compiled=len(checks), rejected=len(rejected), checks=checks,
                  rejections=rejected, mismatches=mismatches,
                  hardware_capability_measured=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["compiled", "rejected", "mismatches"]}))
    assert not mismatches


if __name__ == "__main__":
    main()
