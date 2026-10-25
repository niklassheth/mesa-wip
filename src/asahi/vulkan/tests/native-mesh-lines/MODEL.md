# Original Khronos line pipelines

This host harness uses both original mesh and fragment SPIR-V modules from
KhronosGroup/Vulkan-Samples `69622af06f7652c7fc9b9a9180e7142046315cbc`, under
`shaders/gshader_to_mshader/{glsl,slang}`. `host.py` checks every module against
the corpus SOURCE-MANIFEST and runs SPIR-V validation. Neither stage is rewritten.

The source emits a line from each input triangle's centroid along its first
vertex's normal, interpolating blue at the start to red at the end. The input
has identity matrices, three coincident positions per triangle, and a horizontal
normal of length ten. The resulting segment is [16,80] at pixel row 4+p in a
128x64 viewport. Every primitive occupies a distinct row. `check.py` derives
each RGBA8 value with integer arithmetic at pixel centers and compares the
entire 512-KiB input buffer, including guards. The checker rejects deletion
of every primitive, a one-byte gradient change, and a guard corruption.

GLSL declares V64/P126, so valid fixture counts are at most 32 lines. Slang
declares V126/P64; the input meshlet's 126 indices limit it to 42 lines. The
nine software cases use P0/1/31/32 and Slang P42. These cross vertex-publication
round boundaries; their index words fit one runtime round. The independent
`hk-native-mesh-lines` NIR test also covers up to V256/P256 and all four index
rounds, with logical widths 1/16/32/64 and partial/zero output. The primitive
transport test covers line endpoints under both provoking modes.

`probe.c` is guarded for host recording or explicit software reference. Build
it with the standalone Meson file or an ARM compiler using the matching Vulkan
sysroot. `host.py --mode software` takes `--corpus`, `--driver` (lavapipe),
`--probe`, `--sysroot`, and a new `--out` directory. `--mode record` takes the
same arguments with the candidate Honeykrisp ICD, plus `--shim` and `--sync`
for the non-executing DRM recorder. Each process records one draw, retains
original module hashes, and has a bounded timeout.

Native records require T32, the original declaration bounds, eight component
planes, a line draw header of `0x70000100`, no task predecessor, and unchanged
initialized readback. The state row counts are 129 and 257. This records the
normal driver command path; it does not patch packets or execute Apple GPU code.
The packet layout agrees with retained T8132 own-MSL DATA in
`compiler/research/native-capability-20261022/README.md`.

Native line admission is research-only. Index stride and count-header alignment
derive from two corners; publication uses the existing bounded T32 rounds.
Primitive attributes use the same private provoking-vertex model as triangles,
with last corner K-1. Line CullPrimitive/CullDistance stay emulated because a
zero-length line is not a valid cull representation. Line stipple also falls
back. The original line pipelines have neither feature. CPU execution,
software rasterization and host packet validation do not qualify the new
programs on Apple hardware.
