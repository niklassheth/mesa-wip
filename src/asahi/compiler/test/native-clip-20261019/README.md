# Native clipping qualification inputs

This is a host investigation and a handoff to the hardware owner. It contains
no device runner or production clipping change. Read `CLIP-20261019.md` in the
artifact root before selecting arms. Whole programs use the current compiler
and its strict explicit-formation predicate; no guard-band bypass is inserted.

Recreate source geometry, GLSL comparisons, exact RGBA images and the hypothesis
polygons, then export whole native mesh programs:

```sh
python generate.py --out /path/to/clip-artifacts --header fixtures.h
build/src/asahi/compiler/apple9-native-mesh --export-clip /path/to/clip-artifacts/fixtures-arm
python freeze.py --artifacts /path/to/clip-artifacts --object-fixtures /path/to/workgroup/fixtures-arm
```

For device inputs, the middle command must run the AArch64 compiler executable
(QEMU on the host is sufficient). The retained package was built this way.
The 13 vertex and 13 mesh GLSL sources plus the red fragment shader compile
with `glslangValidator -V --target-env vulkan1.2`. The native bins are generated
from equivalent ordinary NIR, not from those GLSL files. Every native mesh uses
API T16 and physical T16, V3/P1, Position only, one group. The object is the
complete T16 plain program from the workgroup qualification package.

Start with the hardware owner's qualified T16 carrier and pass the whole
authored CONTROL before proceeding. Preserve its actual entry/frame backing,
resource IDs and allocations; bind the ordinary draw descriptor and a zeroed
class-drop counter. No compiler spill is not evidence for zero launch frames.
One VM/process per arm. Pair each native arm with an ordinary VS draw using
the matching `.vert.spv` and `.frag.spv`, the same source winding, no culling,
no depth attachment/test, one sample, no blending, red RGBA and transparent
black clear. Render target is 64x64. Use the manifest viewport/scissor;
`viewport` order is scale XYZ then translate XYZ. The standard API viewport
is `(x=0, y=64, width=64, height=-64, minDepth=0, maxDepth=1)`.

The near/far CLAMP controls disable depth clipping and enable depth clamping
through the normal state serializer; record all resulting PPP/scissor words.
Other arms enable depth clipping. Do not patch unknown packet bits. The
generated red shader needs no discard, texture, derivatives or sample shading.
Do not force a production change to triangle-merging policy.

For every draw retain source vertices, actual PPP/VDM/launch/argument records,
program hashes, Work-selected scene/cursor/output IDs and complete before/after
record allocations. Pin pointers to this actual draw; do not infer them from
a previous scene. Read class drops, pixels and real TA/3D retirement stamps.
Record canaries and use the hardware owner's 20-second GPU/error monitor with
MemoryMax=2G and MemorySwapMax=0; on a live GPU error/deadline freeze the whole
unit and retain its VM/BOs until controller reset. Stop after any unexpected
failure, record error, missing retirement or pixel mismatch.

```sh
python verify-pixels.py --artifacts /path/to/clip-artifacts \
  --arm FULLSCREEN --pixels /path/to/readback.rgba8
```

This checks exact bytes, not just red counts, and does not certify DATA or
retirement. The Python polygon oracle is rational geometry, independently
clipped to the API volume and rasterized with the established 1/256 snap and
top-left rule. For these vectors its image also agrees with the 1.5w guard-band
hypothesis after viewport/scissor. No hardware clip fan order, helper-lane map,
record compression or wire format is inferred from that agreement.

DATA outside H1 is deliberately an investigation, not an invented pass oracle.
The manifest records the original vertices and the hypothetical 1.5w clipped
polygon (plus quantized screen coordinates and fan triangle count). Decode all
actual primitives and compare their geometric union, state/references and
bounds with source. A rotated fan or different compression can be valid. If
the existing single-triangle decoder cannot decode a new packet, retain it as
unknown and stop promotion; it is neither a failure nor a qualified pass.
Never compare a clipped draw with H1's fixed 68/12/32-byte record lengths.

The critical discrimination is whether actual output records already contain
the 1.5w/near-plane-generated vertices, remain the original triangle while a
later stage clips, or are absent despite retirement. Absence of an internal
helper seam alone cannot distinguish clipping from merging. Follow the report
for additional merge/primitive-ID diagnostics only after these bounded arms.
