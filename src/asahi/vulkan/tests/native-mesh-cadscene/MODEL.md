# Captured CADSCENE basic taskless component fixture

One draw, one workgroup, one submission and fence; research mode 3 only.
Original Fossilize mesh `225636e04d3505d6` and fragment `4b476a849d94367f`
modules from pipeline `f16d6602c9af8440` are used byte for byte. Logical L32,
V64/P84, position plus eleven user components. This is component evidence;
it is not a complete application run and cannot qualify a default class.

The capture explicitly requests subgroup size 32. Current Honeykrisp does
not advertise required subgroup size for mesh/task stages. The fixture omits
that stage pNext: the basic mesh has no subgroup capabilities or operations,
so its semantics do not depend on subgroup width. This is an explicit pipeline
state adapter. The separate captured culling shader has subgroup assumptions
and is not substituted into this fixture. It needs a valid 32-lane rehearsal.

The original RGBA8 UNORM and D32_SFLOAT_S8_UINT formats, LESS depth test/write,
backface culling, CCW front face, depth clamp, two-pixel line width, sample count
and disabled blending are retained. The fixture supplies an isolated 128x64
viewport/scissor, clears color to zero and depth to one, and adds transfer
readback. The captured load operations and application resources are not replayed.

The CPU oracle uses integer edge functions on an 8x8 vertex grid with one
column of cells omitted: 84 triangles, including deliberate backfaces. No
sample center touches an edge. Packed 16-bit vertex indices permute all 64
vertices. Nonzero descriptor, packed-index, meshlet and vertex offsets exercise
the original addressing. Meshlet ID 5 feeds the original color hash; object
color, +Z normals and constant extra attributes give a color at least 1/8 UNORM
step from a rounding boundary. The light is far along +Z. Four real software
controls change the upper vertex, final primitive, normal and object color;
each must change pixels while leaving the full input allocation unchanged.

Every RGBA8 byte and depth float is compared exactly, along with the entire
512 KiB input/guard buffer. Synthetic mutations exercise missing rounds,
attachments, individual channels and each buffer region. No shader is rewritten
for software rehearsal. The native recording verifies a real ARM compilation,
T32 route and 15 output planes. The paired kernel parser must accept research
and reject qualified mode, including relocation and truncation checks.

The device run additionally requires a retired fence, correct mapped Mesa
hash/revision, own-file-descriptor VM profile and generation, exact native
counter deltas, and continuous kernel-log coverage. The canonical operator
owns the device, staging, reset, collection and recovery. No retries or Mesa
installation occur. Uncertain in-flight work retains its backing.
