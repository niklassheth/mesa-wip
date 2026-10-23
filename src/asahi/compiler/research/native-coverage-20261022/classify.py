#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the pinned native CTS inventory against a non-executing host replay.

Reads existing evidence only. Does not run CTS, open a device, or change Mesa.
The disjoint first-rejection buckets are specific to Mesa bcb0837f948a.
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_sources(path):
    """Public CTS vkPrograms.cpp cache format; size includes the header."""
    data = path.read_bytes()
    offset = 0
    result = []
    while offset < len(data):
        size, _, fmt, binary_size = struct.unpack_from("<4I", data, offset)
        assert fmt == 0
        pos = offset + 16 + binary_size
        source_size = struct.unpack_from("<I", data, pos)[0]
        assert pos + 4 + source_size == offset + size
        result.append(data[pos + 4:offset + size].decode())
        offset += size
    assert offset == len(data)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True,
                    help="directory containing stdout.log and results.qpa")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="original device native/main directory")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    original = [r for r in json.loads(args.inventory.read_text())
                if r["phase"] == "native"]
    inventory = {r["case"]: r for r in original}
    assert len(inventory) == len(original) == 28012
    failures = {k: r for k, r in inventory.items() if r["category"] == "Fail"}
    feature_failures = {k for k, r in failures.items()
                        if "VK_ERROR_FEATURE_NOT_PRESENT" in r["main_description"]}
    assert len(failures) == 625 and len(feature_failures) == 545

    log = (args.replay / "stdout.log").read_text()
    sections = re.split(r"Test case '([^']+)'\.\.", log)
    diagnostics = {}
    for case, section in zip(sections[1::2], sections[2::2]):
        messages = re.findall(r"^COVERAGE_VK (.*)$", section, re.M)
        errors = [m for m in messages if "VK_ERROR_FEATURE_NOT_PRESENT" in m]
        assert case not in diagnostics and errors, case
        diagnostics[case] = errors
    assert set(diagnostics) == feature_failures

    caches = sorted(args.cache_dir.glob("worker-*/shadercache.bin"))
    assert len(caches) == 4
    original_sources = [s for path in caches for s in cache_sources(path)]
    qpa = (args.replay / "results.qpa").read_text()
    xml = [ET.fromstring(s) for s in
           re.findall(r"<TestCaseResult\b.*?</TestCaseResult>", qpa, re.S)]
    assert {x.attrib["CasePath"] for x in xml} == feature_failures
    assert len(xml) == len(feature_failures)
    all_glsl, all_asm = set(), set()
    metadata = {}
    for root in xml:
        case = root.attrib["CasePath"]
        assert root.find("Result").attrib["StatusCode"] == inventory[case]["main_status"]
        glsl = {n.text or "" for n in root.iter("ShaderSource")}
        assembly = [n.text or "" for n in root.iter("SpirVAssemblySource")]
        # The two assembly-input programs are logged verbatim; generated
        # disassembly is not a compiler-input identity check.
        input_asm = {s for s in assembly if any(k.endswith(s) for k in original_sources)}
        assert all(any(k.endswith(s) for k in original_sources) for s in glsl), case
        all_glsl.update(glsl)
        all_asm.update(input_asm)
        shapes = set()
        payload = False
        for source in assembly:
            if "OpEntryPoint MeshEXT" not in source:
                continue
            modes = dict(re.findall(r"OpExecutionMode %\w+ (\w+)([^\n]*)", source))
            topology = next(k for k in modes if k in
                            ("OutputTrianglesEXT", "OutputLinesEXT", "OutputPoints"))
            shapes.add((int(modes["OutputVertices"]),
                        int(modes.get("OutputPrimitivesEXT", modes.get("OutputPrimitivesNV"))),
                        topology))
            payload |= "TaskPayloadWorkgroupEXT" in source
        assert len(shapes) == 1, (case, shapes)
        metadata[case] = (next(iter(shapes)), payload)
    assert len(all_glsl) == 109 and len(all_asm) == 2

    audit = []
    for case, row in sorted(failures.items()):
        description = row["main_description"]
        api = ("createGraphicsPipelines" if description.startswith("vk.createGraphicsPipelines")
               else "endCommandBuffer" if description.startswith("vk.endCommandBuffer")
               else "pipeline construction" if description.startswith("retcode")
               else "property assertion")
        if case not in diagnostics:
            assert "maxMeshOutputComponents too low" in description
            bucket = "output_components_limit"
            code = "hk_physical_device.c:1877; vktMeshShaderInOutTestsEXT.cpp:628"
            first, shape, payload = description, "", False
        else:
            first = diagnostics[case][0]
            (vertices, primitives, topology), payload = metadata[case]
            shape = f"V{vertices}/P{primitives}/{topology}"
            if "requires V3 P1 triangles without payload" in first:
                code = "agx_nir_apple9_mesh_native.c:18-24; hk_shader.c:5993"
                if topology == "OutputPoints":
                    bucket = "points"
                elif topology == "OutputLinesEXT":
                    bucket = "lines"
                elif (vertices, primitives) != (3, 1):
                    bucket = "triangle_output_shape"
                else:
                    assert payload and case.endswith("payload_not_accessed")
                    bucket = "mesh_payload"
            elif "primitive attributes/culling are not implemented" in first:
                code = "agx_nir_apple9_mesh_native.c:260-262; hk_shader.c:5993"
                bucket = "primitive_attributes"
            elif "output builtin is not implemented" in first:
                code = "agx_nir_lower_apple9_mesh.c:229-233; hk_shader.c:5993"
                bucket = ("builtin_primitive_id" if ".primitive_id_" in case else
                          "builtin_viewport" if case.endswith("viewport_index") else
                          "builtin_layer")
            elif "attributes development path is disabled" in first:
                code = "agx_nir_apple9_mesh_native.c:263-264; hk_shader.c:5993"
                bucket = "vertex_attributes_gate"
            else:
                assert "hk_cmd_draw.c:5392" in first, first
                code = "hk_cmd_draw.c:5388-5392"
                if ".draw_indirect_count." in case:
                    bucket = "draw_indirect_count"
                elif ".draw_indirect." in case or case.endswith("draw_index_in_mesh"):
                    bucket = "draw_indirect"
                else:
                    assert ".conditional_rendering.draw." in case
                    bucket = "draw_conditional_direct"
        audit.append(dict(case=case, api=api, first_bucket=bucket, code_at_bcb0837f948a=code,
                          mesh_shape=shape, mesh_payload_declared=payload,
                          conditional_rendering=".conditional_rendering." in case,
                          first_diagnostic=first))

    buckets = collections.Counter(r["first_bucket"] for r in audit)
    expected = dict(triangle_output_shape=293, points=29, lines=6, mesh_payload=1,
                    primitive_attributes=27, builtin_primitive_id=2, builtin_viewport=1,
                    builtin_layer=4, vertex_attributes_gate=1, draw_indirect=61,
                    draw_indirect_count=60, draw_conditional_direct=60, output_components_limit=80)
    assert buckets == expected, buckets
    summary = dict(
        host_only=True, gpu_execution=False, mesa_source="bcb0837f948a",
        original_cts_source="f6a29701220f34dd1407513bfe80d74ca7b392ce",
        original_inventory_categories=dict(collections.Counter(r["category"] for r in original)),
        original_failure_api=dict(collections.Counter(r["api"] for r in audit)),
        disjoint_first_rejections=dict(sorted(buckets.items())),
        api_by_first_rejection={b: dict(collections.Counter(r["api"] for r in audit
                                                        if r["first_bucket"] == b))
                                for b in sorted(buckets)},
        original_skip_reasons=dict(sorted(collections.Counter(r["main_description"] for r in original
                                                               if r["category"] == "Skip").items())),
        source_identity=dict(original_cache_records=len(original_sources),
                             distinct_matching_glsl=len(all_glsl),
                             distinct_matching_assembly_inputs=len(all_asm),
                             caveat="Host CTS binary differs. Compiler inputs match original cache keys; no claim of bit-identical generated SPIR-V or GPU results."),
        replay_diagnostics=sum(map(len, diagnostics.values())),
        deferred_compiler_failures=sum(r["api"] == "endCommandBuffer" and
                                      "hk_shader.c" in r["first_diagnostic"] for r in audit),
        inputs=[dict(path=str(p), sha256=sha(p)) for p in
                [args.inventory, args.replay / "stdout.log", args.replay / "results.qpa", *caches]],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "CASE-AUDIT.tsv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(audit[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps(dict(cases=len(audit), buckets=buckets, source_identity=summary["source_identity"]), indent=2))


if __name__ == "__main__":
    main()
