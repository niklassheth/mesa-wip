#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Author independent MSL probes; no compiler output is an input."""
import argparse
import hashlib
import json
from pathlib import Path


def cases():
    rows = []

    def add(name, **kw):
        row = dict(name=name, vertices=3, primitives=1, topology="triangle",
                   vertex_varyings=0, primitive_varyings=0, primitive_scalars=0,
                   point_size=True, payload=0,
                   runtime=False, threads=16)
        row.update(kw)
        rows.append(row)

    add("base")
    for v in [4, 16, 64, 128, 256]:
        add(f"vertices_{v}", vertices=v)
    for p in [2, 16, 128, 256, 512]:
        add(f"primitives_{p}", primitives=p)
    for v, p in [(4, 2), (16, 16), (64, 128), (128, 256), (256, 512)]:
        add(f"shape_{v}_{p}", vertices=v, primitives=p)
    for topology in ["point", "line"]:
        for v, p in [(3, 1), (16, 16), (256, 512)]:
            add(f"{topology}_{v}_{p}", topology=topology, vertices=v, primitives=p)
    add("point_no_size", topology="point", point_size=False)
    for count in [1, 2, 4, 8, 16, 24, 28, 29, 30, 31]:
        add(f"vertex_varyings_{count}", vertex_varyings=count)
    for count in [1, 2, 4, 8, 16, 24, 28, 29, 30, 31]:
        add(f"primitive_varyings_{count}", primitive_varyings=count)
    for count in [1, 2, 3, 5, 7, 15]:
        add(f"primitive_scalars_{count}", primitive_scalars=count)
    for v, p, vv, pv in [(16, 16, 4, 4), (64, 64, 4, 4),
                          (128, 128, 4, 4), (256, 128, 4, 4),
                          (256, 256, 1, 1), (256, 512, 1, 1),
                          (256, 1, 6, 0), (256, 1, 7, 0),
                          (256, 1, 8, 0), (256, 1, 15, 0),
                          (3, 512, 0, 4)]:
        add(f"output_{v}_{p}_{vv}_{pv}", vertices=v, primitives=p,
            vertex_varyings=vv, primitive_varyings=pv)
    for size in [16, 32, 256, 1024, 4096, 8192, 16384]:
        add(f"payload_{size}", payload=size)
    for v, p in [(3, 1), (4, 2), (64, 128), (256, 512)]:
        add(f"runtime_{v}_{p}", vertices=v, primitives=p, runtime=True)
    return rows


def source(c):
    v, p = c["vertices"], c["primitives"]
    vv, pv = c["vertex_varyings"], c["primitive_varyings"]
    ps = c["primitive_scalars"]
    point_size = c["topology"] == "point" and c["point_size"]
    fields = lambda prefix, n: "\n".join(
        f"    float4 {prefix}{i} [[user({prefix}{i})]];" for i in range(n))
    vertex = "struct V { float4 position [[position]];\n"
    if point_size:
        vertex += "    float point_size [[point_size]];\n"
    vertex += fields("v", vv) + "\n};\n"
    scalar_fields = "\n".join(f"    float s{i} [[user(s{i})]];" for i in range(ps))
    primitive = "struct P {\n" + fields("p", pv) + ("\n" + scalar_fields if ps else "") + "\n};\n" if pv + ps else ""
    text = "// Authored capability probe. SPDX-License-Identifier: MIT\n"
    text += "#include <metal_stdlib>\nusing namespace metal;\n" + vertex + primitive
    text += f"using M = metal::mesh<V, {'P' if pv + ps else 'void'}, {v}, {p}, topology::{c['topology']}>;\n"
    if c["payload"]:
        words = c["payload"] // 4
        text += f"""struct Payload {{ uint words[{words}]; }};
[[object, max_total_threads_per_threadgroup(16)]]
void object_main(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                 device const uint *data [[buffer(0)]], uint t [[thread_index_in_threadgroup]]) {{
    for (uint i = t; i < {words}; i += 16) payload.words[i] = data[i];
    if (t == 0) grid.set_threadgroups_per_grid(uint3(1, 1, 1));
}}
"""
    payload_arg = "const object_data Payload &payload [[payload]], " if c["payload"] else ""
    text += f"""[[mesh, max_total_threads_per_threadgroup(16)]]
void mesh_main(M m, {payload_arg}device const uint *data [[buffer(0)]],
               uint t [[thread_index_in_threadgroup]]) {{
"""
    if c["runtime"]:
        text += f"    uint nv = clamp(data[0], 1u, {v}u);\n    uint np = min(data[1], {p}u);\n"
    else:
        text += f"    const uint nv = {v}, np = {p};\n"
    seed = f"payload.words[data[2] % {c['payload'] // 4}]" if c["payload"] else "data[2]"
    text += f"    uint seed = {seed};\n"
    text += "    for (uint i = t; i < nv; i += 16) {\n        V out;\n"
    text += "        out.position = float4(float(i & 15u) / 8.0 - 1.0, float(i >> 4) / 8.0 - 1.0, float(seed & 255u) / 256.0, 1.0);\n"
    if point_size:
        text += "        out.point_size = float((seed & 3u) + 1u);\n"
    for i in range(vv):
        text += f"        out.v{i} = float4(float(seed + i + {i}), float(i + {i + 1}), float(seed ^ {i + 2}u), float(seed + {i + 3}));\n"
    text += "        m.set_vertex(i, out);\n    }\n    for (uint i = t; i < np; i += 16) {\n"
    indices = {"point": 1, "line": 2, "triangle": 3}[c["topology"]]
    for i in range(indices):
        text += f"        m.set_index(i * {indices} + {i}, uchar((i * {indices} + {i}) % nv));\n"
    if pv + ps:
        text += "        P out;\n"
        for i in range(pv):
            text += f"        out.p{i} = float4(float(seed + i + {i}), float(i + {i + 1}), float(seed ^ {i + 2}u), float(seed + {i + 3}));\n"
        for i in range(ps):
            text += f"        out.s{i} = float(seed + i + {i});\n"
        text += "        m.set_primitive(i, out);\n"
    text += "    }\n    if (t == 0) m.set_primitive_count(np);\n}\n"
    # A merged stage-in type keeps each user varying live in pipeline compilation.
    text += "struct F {\n" + fields("v", vv) + "\n" + fields("p", pv) + ("\n" + scalar_fields if ps else "") + "\n};\n" if vv + pv + ps else ""
    text += "fragment float4 fragment_main(" + ("F in [[stage_in]]" if vv + pv + ps else "") + ") {\n"
    text += "    float4 value = float4(0.125, 0.25, 0.5, 1);\n"
    for prefix, count in [("v", vv), ("p", pv)]:
        for i in range(count):
            text += f"    value += in.{prefix}{i} * {1 / (i + 2):.9f};\n"
    for i in range(ps):
        text += f"    value += in.s{i} * {1 / (i + 2):.9f};\n"
    return text + "    return fract(value);\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = cases()
    for row in rows:
        folder = args.out / row["name"]
        folder.mkdir(exist_ok=True)
        data = source(row).encode()
        (folder / "probe.metal").write_bytes(data)
        row["source_sha256"] = hashlib.sha256(data).hexdigest()
        pipeline = dict(mesh_function="alias:lib#mesh_main",
                        fragment_function="alias:lib#fragment_main",
                        color_attachments=[dict(pixel_format="RGBA8Unorm")])
        if row["payload"]:
            pipeline.update(object_function="alias:lib#object_main",
                            payload_memory_length=row["payload"])
        spec = dict(libraries=dict(paths=[dict(label="lib", path="probe.metallib")]),
                    pipelines=dict(mesh_render_pipelines=[pipeline]))
        (folder / "pipeline.mtlp-json").write_text(json.dumps(spec, indent=2) + "\n")
    (args.out / "CASES.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Authored {len(rows)} cases")


if __name__ == "__main__":
    main()
