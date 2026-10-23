#!/usr/bin/env python3
"""Run the bounded sliced-3D kit through Mesa's runner; never install a driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time
import tomllib


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    kit = args.kit.resolve(strict=True)
    manifest = json.loads((kit / "manifest.json").read_text())
    arch = {"aarch64": "aarch64", "x86_64": "x86_64"}[platform.machine()]
    library = kit / args.arm / arch / "libvulkan_asahi.so"
    expected = manifest["libraries"][str(library.relative_to(kit))]["sha256"]
    if digest(library) != expected:
        raise RuntimeError("Library checksum does not match the kit manifest")
    cfg = json.loads(args.config.read_text())
    ci = Path(cfg["ci_dir"]).resolve(strict=True)
    template = (ci / "deqp-asahi-m4-vk.toml").read_text()
    for prefix, path in cfg["path_map"].items():
        replacement = json.dumps(str(Path(path).resolve(strict=True)))
        template = template.replace(json.dumps(prefix), replacement)
        template = template.replace(json.dumps(prefix)[:-1] + "/", replacement[:-1] + "/")
    deqp = Path(tomllib.loads(template)["deqp"][0]["deqp"]).resolve(strict=True)
    runner = Path(cfg["runner"]).resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    icd = output / "asahi.json"
    icd.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
        "library_path": str(library), "api_version": "1.4.0"}}, indent=2) + "\n")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MESA_", "ASAHI_", "AGX_", "GALLIUM_", "LIBGL_", "HK_", "VK_")) or key in ("LD_PRELOAD", "LD_LIBRARY_PATH"):
            del env[key]
    # Keep display/session configuration, but no inherited driver experiments.
    env.update({k: v for k, v in cfg.get("environment", {}).items()
                if k in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XAUTHORITY")})
    env.update(VK_DRIVER_FILES=str(icd), HK_EXPERIMENTAL_APPLE9="1",
               LD_LIBRARY_PATH=str(library.parent), MESA_SHADER_CACHE_DIR=str(output / "cache"))
    commands = []
    if args.suite in ("probe", "all"):
        commands.append(("probe", [str(kit / arch / "sliced-3d-probe"), str(kit / "shaders")]))
    groups = ["sliced", "storage"] if args.suite == "all" else [args.suite]
    inputs = {str(library): digest(library), str(deqp): digest(deqp), str(runner): digest(runner)}
    for group in groups:
        if group == "probe":
            continue
        cases = kit / "tests" / (group + ".txt")
        lines = cases.read_text().splitlines()
        if not lines or len(lines) != len(set(lines)):
            raise RuntimeError("Empty or duplicate caselist")
        inputs[str(cases)] = digest(cases)
        suite = output / (group + ".toml")
        suite.write_text("[[deqp]]\n" + "\n".join([
            "deqp = " + json.dumps(str(deqp)),
            "caselists = " + json.dumps([str(cases)]),
            "tests_per_group = 50", 'renderer_check = "Apple M4"']) + "\n")
        baseline = output / "baseline.csv"
        baseline.write_text("".join((ci / f"{prefix}-fails.txt").read_text() + "\n"
            for prefix in ("all", "asahi", "asahi-agx2") if (ci / f"{prefix}-fails.txt").is_file()))
        command = [str(runner), "suite", "--suite", str(suite), "--output", str(output / group),
                   "--jobs", "1", "--timeout", "120", "--max-fails", "20",
                   "--baseline", str(baseline), "--save-xfail-logs"]
        inputs[str(baseline)] = digest(baseline)
        for kind in ("skips", "flakes", "single-thread"):
            for prefix in ("all", "asahi", "asahi-agx2"):
                path = ci / f"{prefix}-{kind}.txt"
                if path.is_file():
                    inputs[str(path)] = digest(path)
                    command += ["--" + kind, str(path)]
        commands.append((group, command))
    record = {"arm": args.arm, "commit": manifest[args.arm], "created": time.time(),
              "prepared_only": args.prepare_only, "commands": commands,
              "environment": {k: env[k] for k in ("VK_DRIVER_FILES", "HK_EXPERIMENTAL_APPLE9", "LD_LIBRARY_PATH", "MESA_SHADER_CACHE_DIR")},
              "inputs": inputs, "results": []}
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return output, commands, env, record, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--config", type=Path, default=Path("/g16/tools/mesa-ci/config.json"))
    p.add_argument("--arm", choices=("candidate", "baseline"), default="candidate")
    p.add_argument("--suite", choices=("sliced", "storage", "probe", "all"), default="all")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    output, commands, env, record, cfg = prepare(args)
    for group, command in commands:
        print(group + ": " + shlex.join(command), flush=True)
    if args.prepare_only:
        return 0
    version = subprocess.check_output([cfg["runner"], "--version"], text=True).strip()
    if version != "deqp-runner " + cfg["runner_version"]:
        raise RuntimeError("Runner version differs from installed CI configuration")
    for group, command in commands:
        with (output / (group + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300 if group == "probe" else None)
        status = "skip" if group == "probe" and result.returncode == 77 and args.arm == "baseline" else "exit"
        record["results"].append({"group": group, "status": status, "returncode": result.returncode})
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{group}: {status} {result.returncode}; {output / (group + '.log')}", flush=True)
        if result.returncode and status != "skip":
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
