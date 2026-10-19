# Native mesh clip follow-up — host-prepared device arms

Based on `6388f1020cf`, gated by `HK_APPLE9_NATIVE_MESH=1`. This kit has no GPU
execution results. Use it only after the production kernel passes H1, with the
native kernel boot gate enabled. Nothing installs a driver, changes global
environment variables, resets the device, or enables native mesh persistently.

`ARMS.json` contains 20 VS/native pairs. `_ZERO_ONE` uses Vulkan's normal depth
range; `_HALFZ` uses `VK_EXT_depth_clip_control` with negativeOneToOne=true and
source Z adjusted to give the same final depth. All vertices have positive,
non-unit W (1, 2, 0.5). Each arm uses its actual GLSL through the normal compiler;
there is no shader replacement or private command carrier.

| Pair family | Expected result |
| --- | --- |
| CONTROL | 496 red samples, depth 0.5 |
| ZERO, ONE | 496 red samples, exact depth 0 or 1 |
| NEAR, FAR | 496 red samples, exact depth 1/1024 or 1023/1024 |
| NEAR_OUT, FAR_OUT | All clear; source depths -1/1024 or 1025/1024 |
| CROSS | 1,244 red samples; crosses near/far and side planes |
| RECIP_W | Same crossed triangle; thresholded `gl_FragCoord.w` colors |
| SMOOTH | Same triangle, true smooth user varying; **native pending** |

SMOOTH is intentionally rejected by the current position-only native class
before submission. Its VS shader, paired mesh shader, and exact oracle are
ready. The oracle differs from affine interpolation at 626 covered samples.
RECIP_W tests the reciprocal-W rasterizer input within the admitted class; it
does not qualify generic smooth attributes. The CLIP-20261019 36/36 reference
explicitly leaves attribute routing unqualified.

The probe uses a 64×64 negative-height viewport, no culling or blending,
single-sample RGBA8 and D32, depth ALWAYS/write enabled, black/transparent color
clear and depth clear 1. It copies both full attachments to a coherent CPU
buffer after attachment-to-transfer and transfer-to-host barriers. The fragment
shader never writes depth. `expected/*.depth` is the full little-endian D32
oracle, including clear pixels; color and depth comparisons are byte-exact.
The source-space rational oracle applies projected triangle coverage and the
0..1 depth interval, computes affine raster depth, and uses the perspective
formula for SMOOTH. This is a deliberately strict diagnostic: a hardware
rounding mismatch must be investigated; it is not automatically a Vulkan
conformance violation. No tolerance or excluded pixel list is hidden in it.

On the device, fill a separate kernel identity JSON from the H1-qualified
production kernel (see the example; do not replace hashes with guesses). Then,
for one pair, with a fresh output directory for each process:

```sh
python3 run-one.py --arm NEAR_HALFZ --stage vert --out /root/clip-near-vs --kernel-identity /root/qualified-kernel.json
python3 run-one.py --arm NEAR_HALFZ --stage mesh --out /root/clip-near-mesh --kernel-identity /root/qualified-kernel.json
python3 analyze.py --root . --arm NEAR_HALFZ --vs /root/clip-near-vs --mesh /root/clip-near-mesh
```

Start with CONTROL_ZERO_ONE and CONTROL_HALFZ. Run the remaining admitted
pairs only after both controls pass. Repeat those controls at the end. On a
submission timeout the probe retains backing and the runner stops that process;
collect evidence, then use the workspace's controller reset procedure before
terminating it. The runner does not reset hardware. The native SMOOTH arm stays
blocked until a separate attribute model and kernel class are qualified.

Host reproduction, from the Mesa checkout, after building the debug host and
ARM release driver (the workspace's existing ARM build needs mesa_clc and
vtn_bindgen2 on PATH):

```sh
python3 src/asahi/vulkan/tests/native-clip/prepare-kit.py --root ../clip-followup-20261019/device-kit --workspace /home/nsheth/Projects/asahi --host-build ../build-debug --arm-build ../build-implicit-arm --trace-base ../multi-group-20261019
python3 src/asahi/vulkan/tests/native-clip/run-host.py --root ../clip-followup-20261019/device-kit --out ../clip-followup-20261019/dry-new --workspace /home/nsheth/Projects/asahi --arch host
python3 src/asahi/vulkan/tests/native-clip/run-host.py --root ../clip-followup-20261019/device-kit --out ../clip-followup-20261019/dry-new --workspace /home/nsheth/Projects/asahi --arch arm
python3 src/asahi/vulkan/tests/native-clip/audit-host.py --root ../clip-followup-20261019/device-kit --dry ../clip-followup-20261019/dry-new
python3 src/asahi/vulkan/tests/native-clip/selftest.py ../clip-followup-20261019/device-kit
```

The host and QEMU ARM runs use a non-executing DRM recorder. Their buffers must
remain poisoned; they are preparation checks, never pixel or depth passes.
