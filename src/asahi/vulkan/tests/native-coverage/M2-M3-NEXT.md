# Corpus-driven M2/M3 continuation — msg-180

The M2a candidate admits three of the fixed eighteen mesh declarations.
They also compile into native component pipelines on the host; none has new
native device evidence. The initial report of four candidates was an analyzer
error: unsplit GLSL output blocks hid Sascha's color output. The corrected
analyzer uses Vulkan's member splitting and explicit IO before gathering
metadata. Five controls cover direct, block, primitive and 16-bit interfaces.
Do not treat earlier overlapping-reason counts as a complete interface audit.

M2a is a useful independent gate: logical 1..128, subgroup identity, 2-D
flattening, cross-subgroup visibility and one shared-memory case, on the
existing T16/T32 completion. It excludes expanded derivative groups. A pass
would not qualify every control-flow/barrier pattern. Physical T64/T128 and
derivatives remain follow-up work, not benefits attributed to serialization.

The analyzer now runs the existing native **body** lowerer on a clone and
reports its layout separately from declaration admission. Body-layout success
is neither native completion support nor binary/GPU execution. The current
corpus gives the following useful resource boundaries (bytes):

| Family | V/P | Vertex / primitive components | Packed index words | Position offset / pitch | Output / total local bytes |
|---|---|---|---|---|---|
| Khronos mesh_shading, three variants | 3/1 | 4 / 0 | 1 | 32 / 16 | 96 / 96 |
| Sascha meshshader, three variants | 3/1 | 8 / 0 | 1 | 32 / 16 | 160 / 160 |
| Khronos mesh_shader_culling | 16/8 | 7 / 0 | 6 | 48 / 64 | 496 / 496 |
| Niagara | 64/96 | 17 / 0 | 72 | 304 / 256 | 4656 / 4656 |
| NVIDIA basic, task off/on | 64/126 | 11 / 0 | 95 | 400 / 256 | 3216 / 3216 |
| NVIDIA cull, task off/on | 64/126 | 11 / 0 + cull | 95 | 904 / 256 | 3720 / 4496 |
| Wicked simple | 64/124 | 12 / 1 | 93 | 400 / 256 | 3728 / 3732 |

NVIDIA cull's application shared storage starts at the aligned output end;
its 776-byte increase includes alignment plus 768 user bytes. Wicked simple
needs an independent primitive record/link plane in addition to its vertex
planes. The other three Wicked variants also write ClipDistance, which the
body lowerer currently rejects. The two geometry-to-mesh entries are lines
and also fail body topology admission. No recorded corpus entry uses 16-bit
output; the authored 16-bit control proves that the analyzer can see it, and
that the body lowerer still rejects it. It is a capability requirement with
no present corpus-count credit.

The next hardware model must separate **body execution**, **allocation/control
participation**, and **output publication participation**. Retained W64
evidence has two passing V3/P1 cases with all 64 body lanes but the entire
completion in subgroup zero. Repeated all-lane completion attempts time out.
That evidence does not tell whether control alone must be subgroup-scoped,
or whether channel/index publication may cross a subgroup boundary.

The next proposed discriminator keeps our ordinary NIR/RA and common array
bases, scopes allocation/control to subgroup zero, and tests wider output
prefixes independently. This is an unvalidated hypothesis, not an enabled
compiler rule. A normal-lowering prototype should represent body width and
publication capacity explicitly, with rejection before mutation if the
requested range exceeds the selected candidate. It must not emit captured
sequences, fix registers, or guess per-lane source offsets. S6/S7 already
failed those varying-base alternatives; do not repeat them as a new model.

Start with physical W64 body and a V32/P42 control, then V33/P43 to cross
both publication-prefix boundaries. Only after that result consider W128
for V64/P124 and V64/P126, plus one deterministic interior point and one
partial runtime-count case. A small stop-first kit should be at most six
draws for that phase, not a Cartesian sweep. Every actual vertex must be
referenced, upper-half vertices must affect pixels, packed tail bytes must
be witnessed, and the independent additive image must detect primitive
loss/duplication. The paired research parser currently limits W64 to the
small P1 resource shape and does not admit W128: a reviewed default-off
kernel grammar change is required before sealing this larger-output kit.
State-count encoding alone is not hardware qualification. No UAPI change
is proposed, and no such larger-output kit is ready or queued here.

In parallel, the smallest interface increment is the existing P1 attribute
lowering for Sascha's vec4 color, with real fragment interpolation and the
three-matrix UBO, then a per-primitive flat value. M5 owns task transport;
mesh-component progress must not be counted as a complete task/mesh app.
For multi-primitive outputs, a single value broadcast across vertices is
not a valid substitute for primitive-record ownership. For 16-bit outputs,
first choose and test a typed widening/packing and fragment-link contract;
accepting a declaration flag alone cannot make it correct. ClipDistance
also needs an explicit native raster/fragment contract for the other three
Wicked variants.

Before merging any of these steps, send the normal compiler/driver series
to claude-compiler. Re-run the same eighteen source hashes and record separate
declaration, body-layout, component-compile and native-device counts. M2a's
sealed Air kit is the only newly prepared device workload; hardware and queue
ownership stay with the manager. M4/M5 coordination and C1/C2 remain in
FOR-CODEX-AIR.md; this work makes no payload-placement assumption.
