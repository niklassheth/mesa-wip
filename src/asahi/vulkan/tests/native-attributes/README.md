# Native mesh generic attribute qualification kit

Host preparation only. Native user attributes have not been executed or qualified
on the target. This kit uses normal Vulkan pipelines and compiler lowering, with
`HK_APPLE9_NATIVE_MESH=1` and `HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1` set only in each
child process. Compression remains at its normal default.

The shared `native-clip/probe.c` renders a 64x64 RGBA8 + D32 triangle with depth
writes and a negative-height viewport. Every arm has an ordinary VS equivalent
and a native mesh shader (V3/P1, physical T16, API size 1). All positions have
non-unit W. The seven families each run in zero-to-one and half-Z modes:

- CONTROL: position-only red triangle, repeated before/after qualification.
- SMOOTH: perspective RGB interpolation.
- FLAT: distinct vertex RGB values, first provoking vertex.
- NOPERSPECTIVE: the same RGB values interpolated in framebuffer space.
- MULTIPLE: eight scalars, sparse components, smooth/flat/noperspective groups,
  a flat integer, vec2/vec4 outputs, and gaps between locations.
- SMOOTH_CROSS and MULTIPLE_CROSS: near/far and side-plane clipping.

The fragment shader thresholds the interpolated components into exact 0/255
color bits. Every component contributes. `arms.py` computes source-space
rational barycentrics, clipping, perspective or affine attributes and D32 values;
it does not read compiler output. Smooth and noperspective oracles differ.
`analyze.py` requires every color byte and depth bit to match those independent
oracles, and requires VS/native equality. This strict diagnostic can expose
rounding differences that require investigation; a mismatch is not automatically
a Vulkan conformance violation. Fence completion alone is insufficient.

Per-primitive attributes remain rejected. The evidence shows a separate device
record and an extra local linking plane, but does not establish its publication
or fragment lookup contract. These tests do not emulate it with per-vertex data.

## Device-owner prerequisites and invocation

Wait for the production kernel's H1 qualification. The position-only kernel
validator also needs matching admission for the scalar state count, output rows,
and mesh local allocation used here. No kernel or UAPI changes are included.
Fill `kernel-identity.example.json` with the exact running kernel release/version,
source/image/H1 report hashes, `production_h1_qualified: true`, and
`native_mesh_attributes_supported: true` only after that support is present.
The last `asahi_m4.native_mesh` boot argument must be exactly `1`.

Run each VS and native arm in a fresh process. Start with both CONTROL pairs,
then SMOOTH, FLAT, NOPERSPECTIVE, MULTIPLE, and the CROSS arms. Repeat the controls
at the end in new result directories. Do not proceed after an incomplete draw.

```sh
python3 run-one.py --root . --arm SMOOTH_ZERO_ONE --stage vert \
  --kernel-identity /path/to/kernel-identity.json --out /path/to/results/SMOOTH_ZERO_ONE.vert
python3 run-one.py --root . --arm SMOOTH_ZERO_ONE --stage mesh \
  --kernel-identity /path/to/kernel-identity.json --out /path/to/results/SMOOTH_ZERO_ONE.mesh
python3 analyze.py --root . --arm SMOOTH_ZERO_ONE \
  --vs /path/to/results/SMOOTH_ZERO_ONE.vert --mesh /path/to/results/SMOOTH_ZERO_ONE.mesh
```

The runner verifies input hashes and kernel identity before creating a child
environment. It installs nothing and performs no reset. A timed-out process is
stopped with its backing retained; collect evidence and reset through the
controller before terminating it. Keep these arms separate from hardware results
for previously qualified position-only code.

## Reproduce host preparation from the source checkout

Build the assertion-enabled host driver and ARM driver. The ARM cross build must
use `mesa_clc`, `vtn_bindgen2`, and `asahi_clc` built from this same checkout;
serialized NIR from older external tools is incompatible. Put those tools first
on PATH and clear Meson's program lookup cache when reconfiguring an existing
cross build. Then run:

```sh
python3 src/asahi/vulkan/tests/native-attributes/prepare-kit.py \
  --root /path/to/device-kit --workspace /home/nsheth/Projects/asahi \
  --host-build /path/to/host-build --arm-build /path/to/arm-build \
  --trace-base /path/to/clip-followup-20261019/device-kit
python3 /path/to/device-kit/selftest.py --root /path/to/device-kit
python3 /path/to/device-kit/run-host.py --root /path/to/device-kit \
  --out /path/to/dry --workspace /home/nsheth/Projects/asahi --arch host
python3 /path/to/device-kit/run-host.py --root /path/to/device-kit \
  --out /path/to/dry --workspace /home/nsheth/Projects/asahi --arch arm
python3 src/asahi/vulkan/tests/native-attributes/audit-host.py \
  --root /path/to/device-kit --dry /path/to/dry
```

Host/QEMU uses a recording, non-executing DRM shim; readbacks must stay poisoned.
Its submissions establish compiler/driver preparation only. The audit follows the
actual PPP and coefficient tables for both stages, requiring equal interpolation
state and the expected smooth/flat/linear modes. `SHA256SUMS` covers
the immutable device inputs; per-run ICD files are generated at runtime.
