# Hybrid mesh host qualification

These probes use our Vulkan ICD and our nonexecuting DRM recorder. They cannot
establish GPU correctness or replace the manager-scheduled full-group CTS run.
`record-cases.py` additionally suppresses queue submission to avoid dumping the
large emulated mesh heap for every inventory case. Its image failures are
expected; its useful results are compilation/recording completion and errors.
`record-transitions.py` forwards real submissions to the recorder, so its
captures contain actual UAPI commands and control streams, but no GPU executes.

The C probes require Vulkan headers and a Vulkan loader. Example cross-build
from the Mesa source root (set `sysroot` and `out` to local scratch paths):

```sh
src=src/asahi/vulkan/tests/native-mesh-hybrid
mkdir -p "$out"
for probe in properties transitions; do
  aarch64-linux-gnu-gcc --sysroot="$sysroot" -Iinclude -O2 -Wall -Wextra -Werror \
    "$src/$probe.c" -L"$sysroot/usr/lib64" \
    -Wl,-rpath-link,"$sysroot/usr/lib64" -lvulkan -o "$out/$probe"
done
aarch64-linux-gnu-gcc --sysroot="$sysroot" -Iinclude -O2 -Wall -Wextra -Werror \
  -shared -fPIC "$src/recording-icd.c" -L"$sysroot/usr/lib64" \
  -Wl,-rpath-link,"$sysroot/usr/lib64" -ldl -o "$out/recording-icd.so"
glslangValidator -V --target-env vulkan1.3 "$src/triangle.mesh" -o "$out/native.mesh.spv"
glslangValidator -V --target-env vulkan1.3 -DVERTICES=4 -DPRIMITIVES=2 \
  '-DCOLOR=vec4(0,1,0,1)' "$src/triangle.mesh" -o "$out/emulated.mesh.spv"
glslangValidator -V --target-env vulkan1.3 "$src/triangle.vert" -o "$out/vertex.spv"
glslangValidator -V --target-env vulkan1.3 -DMESH=1 "$src/color.frag" -o "$out/mesh.frag.spv"
glslangValidator -V --target-env vulkan1.3 "$src/color.frag" -o "$out/vertex.frag.spv"
```

All Python tools expose their inputs through `--help`. They run aarch64 binaries
under `qemu-aarch64-static`, record input hashes, disable the disk shader cache,
and clear inherited driver/debug overrides. Keep the recorder and CTS binary
fixed across comparison arms. No host test changes the kernel or touches a
device. Use the real ICD directly for `properties`, without the submission
proxy, and compare its output files with native disabled and enabled.

For each native setting, run `record-cases.py --driver ICD --proxy PROXY --shim
RECORDER --cts CTS --sysroot SYSROOT --native 0_OR_1 --cases CASELIST --output
NEW_DIRECTORY`. Feed bounded batches of the cases in
`src/asahi/compiler/research/native-coverage-20261022/CASE-AUDIT.tsv` and place
outputs below `inventory-0/batch-NN` and `inventory-1/batch-NN`.
`compare-inventory.py --inventory CASE-AUDIT.tsv --native inventory-1 --emulated
inventory-0` checks exact inventory coverage, duplicate cases, errors, fixed
input hashes and paired harness outcomes. The 625-case inventory is a rejection
inventory from the previous candidate, not the full mesh_shader group.

For transition captures, run `record-transitions.py --driver ICD --shim RECORDER
--sysroot SYSROOT --probes PROBE_DIRECTORY --native 0_OR_1 --output NEW_DIRECTORY`.
Then `check-transitions.py --native ON_DIRECTORY/trace --emulated
OFF_DIRECTORY/trace` checks the default `NENVNMNN` sequence. N is an eligible
V3/P1 mesh, E declares V4/P2 and must use emulation, V is an ordinary vertex draw,
and M is a partial attachment clear. Each operation colors its own 8-pixel stripe
in a 64×64 image. Expected pixels are generated independently on the CPU.
The capture checker verifies five single-draw native streams, three ordinary
renders, the emulation compute position, previous-submit barriers, partial
attachment load/store programs and final store. It intentionally accepts only
this bounded fixture and the current UAPI layout. It does not cover depth/stencil
pixels, query results, secondary replay or the full native state space.

On the manager's scheduled device, run `transitions` directly with its five
SPIR-V paths, an existing output directory, and the sequence `NENVNMNN`.
It must return zero and report `pixels_mismatched=0/4096` in both modes. The host
runner instead requires the untouched `0xa5` readback sentinel. Device runs must
use the fixed runner's knob revision 2 and debugfs mode protocol. Leave
`ASAHI_MESA_DEBUG` unset and verify the actual renderer, library and kernel.

Recorded 2026-10-22 evidence and limitations are in
`src/asahi/compiler/README.apple9-native-mesh.md`, section `native-hybrid`.
