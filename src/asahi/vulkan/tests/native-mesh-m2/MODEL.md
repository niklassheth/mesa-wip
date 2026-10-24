# M2a: logical workgroup admission through ordinary Vulkan

This candidate reuses the existing P1 compiler lowering. API workgroups of
1..16 invocations run on physical T16 with an application mask; 17..128 run
on physical T32, with NIR's workgroup-size lowering serializing logical
subgroups around workgroup barriers. It expands driver declaration/upload
admission for position-only V3/P1 triangles. It does not establish physical
T64 or T128 execution. Derivative-bearing expansion remains rejected.

The immutable 18-entry corpus moves from 0 to 3 mesh **declaration
candidates**, with unchanged inputs and denominator: three Khronos
mesh_shading variants. These are
component admission results, not three executed applications or complete
task/mesh pipelines. The precompiled corpus is used unchanged. Native execution coverage remains
0/18 until device evidence exists. All three small Sascha mesh entries
now fail on user varyings; larger declarations, payload, shared storage,
per-primitive output and line topology remain overlapping blockers.

The eight one-draw cases sample logical sizes 1,17,32,64,128, a deterministic
interior size 115 (seed 0x180), 8x8, and 64 with API shared storage. Seven use
mode 4 and no native-mesh override, testing the candidate default-selection
path. The shared arm uses explicit research mode 3: shared storage has not
been added to default admission. Each process has one render pass, one API
draw, one command-buffer submission, and one retirement fence. The driver
may emit separate kernel work for image copy; native render/object deltas
must each be exactly one after successful fence and readback checks.

Every logical invocation stores its index and xyz ID, subgroup ID and lane,
subgroup size/count, logical workgroup size, cookie and a cross-subgroup
consumer value. All invocations first write cookie+index, rendezvous at a
buffer-memory/workgroup barrier, then read `(index+33)%logical_size`. The
shared arm uses a shared array for the consumer. This exposes lost upper
invocations, incorrect subgroup numbering, incorrect flattening and missing
barrier visibility. It does not cover nested/divergent barriers, subgroup
collectives, derivatives, atomics or every shared-memory alias pattern.

One triangle supplies independent completion and raster evidence. Its
screen vertices are (8,8), (24,8), (8,23); the CPU checker uses integer edge
inequalities at pixel centers, not shader-generated coordinates. It checks
all 8,192 R32 float pixels with ONE/ONE ADD and every word of the 512-KiB
witness allocation, including untouched guards. Native subgroup size is
fixed at 32 in the oracle. Software controls use the CPU-reported subgroup
property and are never native evidence. Mutation controls exercise the same
verdict function with lost invocations, subgroup/ID corruption, stale reads,
guard damage, missing/duplicate pixels, unexecuted and truncated readback.

The first analyzer revision missed user members in unsplit GLSL output
blocks and reported four candidates. Interface normalization and explicit
IO lowering correct this to three. Five authored controls cover Position,
direct user output, an output block, per-primitive data, and 16-bit IO. The
original 0/18 baseline remains unchanged; its overlapping interface reasons
were incomplete. No device time was spent repairing this analyzer.

The output layout is 3 packed index bytes, counts at 16, Position at 32,
four component planes of pitch 16, and 96 output bytes. The launch backs
at least 128 bytes. API shared storage in the last case follows the output
region at 96 and adds 256 bytes, requiring 512 rounded bytes of backing.
It does not add vertex planes. The state header is T16/T32, V3/P1,
word 10 = 0, word 11 = 0x400810/0x400820, word 13 = 5.

Eight real ARM-driver host recordings compile through normal Vulkan/NIR
and emit these headers. The no-op DRM shim lacks qualified-VM discovery,
so these recordings use explicit research selection. They prove compilation
and emitted packets, not default VM selection or execution. Three additional recordings compile the unchanged candidate corpus mesh
modules with a descriptor-compatible harness and substitute constant fragment
shader; they are not full captured application pipelines. Separate driver
tests exercise default selection/upload for all 128 finite P1 sizes and
negative declaration/derivative cases. The paired 4b1e8012b687 kernel parser
accepts all eight physical resource headers in both modes at two relocations
and rejects every truncated prefix. The shared case's header alone cannot
describe or qualify its shader semantics.

The class hash still describes the same 31 physical output resource keys.
The research-mode cache ABI gains the logical-P1 policy version, hashed only
when `HK_APPLE9_NATIVE_MESH` is enabled. Production caches are partitioned by
the driver build UUID. Compiler review confirmed that no kernel advertisement
or class-hash change is needed: the physical classes are unchanged. The upload
validator independently recomputes the physical width from the cached logical
size. No UAPI structure or installed driver is changed.

Physical-width follow-up: the retained W64 discriminator has two successful
body64/completion-first-subgroup cases (with/without the final body barrier)
and repeated all-lane-completion timeouts. That supports studying separate
body and completion participation, not enabling arbitrary T64 output.
T128 and V64/P124..126 remain unqualified. At P124/P126 the existing packed
publication needs 93/95 active words; a W32 prefix cannot cover that range.
Neither raising a state count nor repeating publication with guessed offsets
establishes multi-round semantics. The M3 follow-up must distinguish a wider
collective from a proven multi-round contract before default admission.

M2a review follow-up (2026-10-24): the Air gate passed all eight draws, with
complete pixel/buffer checks, matching native VM routes, one retired native
render/dispatch each and zero faults. A separate authored CTS-style host
subset samples logical 17,31,32,33,40,63,64,65,96,100,127,128. Its ballot/elect
and subgroup-ID witnesses include nested masks and two buffer/workgroup
barriers inside a uniform branch in a push-constant-bounded loop. All twelve
compile and record through the real ARM ICD and no-op DRM shim, and all
software controls pass the independent full-buffer oracle. This establishes
host admission/compilation coverage, not native execution of that subset.
The existing native-lowered NIR interpreter also checks subgroup operations,
3D IDs and cross-subgroup barrier phases. `host-subgroups.py` reproduces the
additional subset without modifying the sealed Air kit.

The three unchanged Khronos candidate mesh modules each compile to 98 backend
instructions / 804 bytes / 10 GPRs / no scratch in the normal Vulkan recording
harness. Each declares one logical invocation, so none incurs subgroup body
duplication. Logical sizes through 128 can duplicate the body up to four
logical subgroups; these small candidates do not measure that worst case.
