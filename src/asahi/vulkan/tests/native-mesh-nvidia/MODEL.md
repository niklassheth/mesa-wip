# Original NVIDIA basic taskless meshlet

This one-draw candidate uses the unchanged pinned corpus mesh module for
nvpro-samples/gl_vk_meshlet_cadscene, basic taskless V64/P126, SHA-256
`7e177a3a5abec51344de53a657eaa33b7f00864e7fe230ca8c0652230f235a68`.
Its original drawmeshlet_ext.frag.glsl is compiled without source edits using
the same application macro configuration as that mesh module. Both modules
are validated and pinned. The Vulkan adapter supplies their real three-set
descriptor interface, push constants, packed meshlet data and texel buffers.
No shader is rewritten, no fragment is substituted, and no shader witness is
inserted. This exercises the taskless corpus shader pipeline end to end with
deterministic application data; it is not a run of the complete CAD viewer.

Research integration combines M3a interfaces and M3b's 32-lane publication
rounds. The application has 32 mesh invocations, 64 actual vertices and 126
actual primitives: two vertex rounds and three packed-index rounds (95 words).
Seven vertex scalars (world position, normal, flat meshlet ID) use the existing
varying linker. There are no per-primitive attributes, task stage or payload.
Physical width stays T32. Normal NIR, instruction selection and RA generate
the shader; the adapter adds no production compiler workaround.

Inputs use the public pack-basic format: 16-bit vertex indices, permutation
`(13*v+7)%64`, nonzero base vertex, nonzero primitive-buffer offset, descriptor
index 9, and meshlet ID 7. All 64 vertices are referenced. Sixteen disjoint
12-by-11 pixel quads have two triangles each; 126 primitives repeat these
32 triangles. Upper vertices and the final packed-index bytes affect pixels.
Positions and normals are read from separate RGBA32_FLOAT texel buffers.
Scene/object UBOs and all input ranges share one guarded 512-KiB allocation.
The entire allocation must remain byte-identical after the retired draw.

The original fragment shader uses smooth world position/normal and the flat
meshlet ID for diffuse lighting and hash coloring. Four cell-normal classes
have axial diffuse factors 1, 4/5, 3/5 and 2/3. World XY is scaled by 2^-20;
the view-projection matrix restores screen geometry, while light Z minus
world Z is exactly one. The independent oracle evaluates integer raster edges
at pixel centers and rational diffuse/hash colors. No pixel center lies on an
edge. It checks all 32,768 RGBA8_UNORM bytes exactly, including background and
additive multiplicity. Alpha never saturates, so each missing or extra covered
primitive changes the image even where RGB clamps.

The closest rational color is 0.05 UNORM steps from a rounding boundary.
Position-dependent lighting plus a reserved 1e-5 diffuse arithmetic error is
bounded by 0.0029 steps. This margin makes the expected byte independent of
small normalization/interpolation rounding differences. No tolerance is
applied to the device readback. The original-pair software run must match all
bytes, and four further software runs must show changed pixels when an upper
vertex, last primitive, normal or flat meshlet ID input changes. All modules
stay unchanged in those controls. Checker mutations also cover channel
corruption, empty/truncated output, dropped publication rounds and input stores.

The real ARM ICD must record one native T32/V64/P126 render, state word 1
0xc0b, 11 vertex planes, state word 13 177, zero primitive records. The paired
4b1e8012b687 parser must accept research mode 3 and reject qualified mode 4;
replay checks two relocations and every truncated prefix. GPU success requires
the exact mapped driver revision/hash, own-FD research VM/generation audit,
retired fence, full image/input checks, and render/object counter deltas 1/1.

Budget: one draw/process/submission, 60 seconds, zero retries, stop first.
This version uses the normal publication model: at most 32 entries per
collective, with the destination entry in bits 16 and above of the channel
selector or packed-index count operand. It uses only the explicit native
research opt-in; no wide/round-selector or repeated-batch controls are needed.
Air 39b3a F1 validated this destination mechanism at V64/P126. M3a and the
Khronos component pairs also passed. Execution of this unchanged NVIDIA pair
still requires its own collected device result. The manager owns queue,
device and recovery.
No native execution or production qualification is claimed while preparing.
