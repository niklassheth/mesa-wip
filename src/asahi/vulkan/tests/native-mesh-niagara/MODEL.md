# Niagara original taskless pipeline

The fixed corpus mesh hash is `7b25ad807d55ac3d5f201d6f72a8c2980593d461f91cbcceeb601f6207f386c3`.
Both mesh and fragment sources remain unchanged, using their upstream defaults
(TASK=false and POST=0). The adapter implements the source's buffer layouts,
216-byte push constants, six mesh buffers, material buffer and descriptor sets.
Both original G-buffer formats are kept: RGBA8 UNORM and A2B10G10R10 UNORM.
It is an original shader-pipeline test with deterministic application inputs,
not a full Niagara application run.

64 vertices make an 8x8 grid with 96 nonoverlapping triangles (one interior
cell omitted). Every vertex and every primitive contributes to coverage,
including logical invocations 32..63 and all three index publication rounds.
The shader uses the actual packed 16-bit vertex references, 8-bit triangle
indices, half positions and encoded normal/tangent fields, with nonzero base
vertex and nontrivial descriptor/draw/material indices. Default TASK=false
removes task payload use through normal compilation; no shader rewriting or
payload injection occurs.

The oracle uses integer edge functions at pixel centres; no centre lies on
an edge. Endpoint material colors avoid uncertain transcendental rounding.
Normal -Z yields octahedral coordinates (1,1); the original fragment's noise
has less than half a packed-normal quantization step and leaves endpoint
values unchanged. The checker compares all 65,536 target bytes and all
524,288 input/witness bytes exactly, with zero tolerance. Software input
controls separately change an upper vertex, last primitive, normal and
flat draw/material selection. Twenty-two corrupt-output/guard mutations
must be rejected. Texture branches and complete engine rendering are outside
this one-input test's coverage; their descriptors are valid.

Real ARM ICD recording must select native physical T32, logical 64, V64/P96,
17 vertex component planes, no task payload and no primitive record. The
state model is checked and its stream replayed through the pinned kernel
parser (research accepts, qualified rejects). This is host evidence, not GPU
execution. Native mode 3 is explicit; shipping defaults remain off.

The device runner checks its own VM profile around VM_CREATE, exact mapped
library/revision, one native route, retired fence, kernel counters and exact
readback. A failed submit is never counted as completed work. It stops on
uncertain retirement, retains backing for the operator, and never retries.
One process, one draw, one submission, 60 seconds. Kernel/device operations
belong to the common kit operator.
