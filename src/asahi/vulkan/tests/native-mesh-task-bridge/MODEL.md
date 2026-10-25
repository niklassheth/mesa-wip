# GPU task records feeding native mesh

The API task stage runs as ordinary GPU compute. Its per-parent grid and
payload remain owned by the command buffer until native rendering retires.
A GPU prefix pass flattens child grids; the native object reads that grid,
and native mesh groups find their parent and reconstruct Vulkan's 3D IDs.
Zero children use one guarded hardware group, which executes no application
mesh instructions. This is transitional native mesh coverage; it does not
claim native execution of the API task shader or hardware task payloads.

The opt-in is `HK_APPLE9_NATIVE_TASK_BRIDGE=1`, with
`HK_APPLE9_NATIVE_INDEX=1` and `HK_APPLE9_NATIVE_MESH=1`. All device arms use
research mode 3. Index opt-in is mandatory in the harness; the research
override is explicit, and this kit does not qualify a default class.

Admission proves a static per-parent child bound, checks allocation sizes,
caps the total at 65535 children, and applies the existing native primitive
budget before executing tasks. Unknown bounds, oversized draws, indirect
dispatch, unsupported state or mesh outputs use the existing emulation.
No launch is truncated. The existing native class and instruction encodings
are unchanged. Packed GS/TES, dense-render TVB recovery, and CPU wait policy
are outside this change.

Eight one-draw processes cover four original, hash-pinned public triples:
Sascha GLSL/HLSL/Slang and Khronos mesh culling. Khronos runs with two parents
and again with zero children. Three authored cases exercise complete 16 KiB
payloads, two children, three parents with an empty middle parent and 3D child
grids, and task side effects with zero children. Original SPIR-V is unchanged.
NVIDIA task1 and the per-primitive-output task users need separate admission
and publication work; they receive no coverage credit from these tests.

The independent checker examines all 32768 RGBA32F components (absolute
tolerance 2e-5 for nonzero values; clear pixels exact) and all 524288 witness
bytes exactly. Additive blending exposes duplicate rasterization. Authored
children copy every payload word to parent/child-specific output records,
and report WorkgroupID, NumWorkgroups and GlobalInvocationID. Task witnesses
and untouched guard regions expose missing side effects and stray writes.
Software execution validates fixtures; ARM no-op recording validates normal
pipeline compilation, one native route and its compute dependency. Neither
earns GPU coverage. The pinned kernel parser accepts all research streams
and rejects the unqualified interfaces and every truncated stream prefix.

The Air harness additionally requires the sealed mapped driver, own-FD VM
generation/profile, exactly one native route per draw, successful fence
retirement, exact native render/dispatch counter deltas and complete readback.
Zero native routes stop the kit. Faults retain live backing for operator-owned
recovery, with no retry. Kernel logging uses the shared canonical filter.
The full rehearsal runs all eight arms when KIT_REHEARSAL=1, retaining all
pixel and buffer checks while marking native evidence as unmodelled.

Budget: eight draws, eight submissions, 240 seconds, zero retries. The manager
owns the canonical operator and Air queue. Driver review by claude-compiler
is required before merge. Device coverage is pending until those receipts
are collected and independently verified.
