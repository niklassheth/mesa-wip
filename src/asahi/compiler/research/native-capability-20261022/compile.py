#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run on the controller in the generated scratch directory, offline only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
env = os.environ.copy()
env["PATH"] = "/opt/metal-toolchain/usr/bin:" + env["PATH"]
(root / "COMPILER.txt").write_bytes(subprocess.check_output(["metal", "--version"], env=env))
rows = []
previous = {r["name"]: r for r in json.loads((root / "COMPILE.json").read_text())} if (root / "COMPILE.json").exists() else {}
for case in json.loads((root / "CASES.json").read_text()):
    folder = root / case["name"]
    old = previous.get(case["name"])
    if old and old["source_sha256"] == case["source_sha256"]:
        assert hashlib.sha256((folder / "probe.metal").read_bytes()).hexdigest() == old["source_sha256"]
        if old["compiled"]:
            assert hashlib.sha256((folder / "probe-g16.metallib").read_bytes()).hexdigest() == old["archive_sha256"]
        rows.append(old)
        continue
    commands = [["metal", "-std=metal3.2", "-mmacosx-version-min=15.0",
                 "probe.metal", "-o", "probe.metallib"],
                ["metal-tt", "-macos-version-min", "15.0", "-arch", "applegpu_g16g",
                 "probe.metallib", "pipeline.mtlp-json", "-o", "probe-g16.metallib"]]
    row = dict(name=case["name"], source_sha256=case["source_sha256"], steps=[])
    assert hashlib.sha256((folder / "probe.metal").read_bytes()).hexdigest() == row["source_sha256"]
    for i, cmd in enumerate(commands):
        result = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, timeout=120)
        (folder / f"compile-{i}.log").write_bytes(result.stdout + result.stderr)
        row["steps"].append(dict(command=cmd, returncode=result.returncode))
        if result.returncode:
            break
    row["compiled"] = len(row["steps"]) == 2 and row["steps"][-1]["returncode"] == 0
    if row["compiled"]:
        row["archive_sha256"] = hashlib.sha256((folder / "probe-g16.metallib").read_bytes()).hexdigest()
    rows.append(row)
    (root / "COMPILE.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(row["name"], "OK" if row["compiled"] else "FAIL", flush=True)
(root / "COMPILE.json").write_text(json.dumps(rows, indent=2) + "\n")
