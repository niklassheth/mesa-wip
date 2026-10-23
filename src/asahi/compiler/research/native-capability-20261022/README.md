# Native capability study — 2026-10-22

The native V3/P1 triangle restriction describes our implementation, not a
demonstrated hardware limit. This study supplies evidence for broader native
programs. Production eligibility and advertised limits have not changed.

## C1: compiler metadata and retained hardware state

The M1 controller compiled **71 independently authored MSL cases** offline for
`applegpu_g16g`, using Metal 3.2 and macOS minimum 15.0. **66 compile**, including:

- V4/V16/V64/V128/V256 and P1/P2/P128/P256/P512, independently and combined;
- points and lines, through V256/P512, with a separate PointSize control;
- vertex and primitive varyings through 120 user scalars plus Position;
- primitive records with 1/2/3/5/7/15 scalars, establishing four-byte stride
  increments independently of the float4 cases;
- application output exactly 32 KiB;
- object payloads 16/32/256/1024/4096/8192/16384 bytes;
- runtime primitive counts and runtime-selected vertex/index writes.

Metal has no explicit vertex-count setter in these probes. Runtime vertex
coverage varies the initialized vertex prefix and bounds every referenced
index to that prefix; it is not evidence of a Vulkan-style vertex-count input.

Five cases are rejected by the offline compiler: two have 128 unique scalars
against its stated 124 limit; three exceed its stated 32768-byte mesh-output
limit. These are **compiler observations, not measured hardware maxima**.
[Apple's public description](https://developer.apple.com/videos/play/wwdc2022/10162/)
documents 256 vertices, 512 primitives, and 16 KiB payloads.

`CASES.json` records source hashes; `COMPILE.json` records commands, statuses,
and native archive hashes. Sources are reproduced by `generate.py`; full
archives and logs remain in the sibling `../native-capability-20261022/offline/`
scratch directory, and `/tmp/asahi-native-capability-20261022/` on the controller.
`INSPECTION.json` includes the exact compiler version and data hashes.

### Compiler DATA fields

In the mesh layout table (`__GPU_LD_MD/__compute`, child 9), the following
correlations match all 66 compiled cases. Absent scalar fields mean zero.
They are metadata field numbers, **not VDM word offsets**.

| Field | Observed model |
| --- | --- |
| 1 | Maximum vertices |
| 2 | Maximum primitives |
| 3, byte | Topology: point 0, line 1, triangle 2 |
| 4 | Local vertex-plane count: Position + vertex components + one primitive-link plane when primitive data exists + PointSize |
| 6 | 16 + align(vertices, 4) × 4 × plane count |
| 8 | Primitive record stride, four bytes per scalar in these probes |
| 14, byte | PointSize present |
| 22 | Declared maximum mesh threads, 16 in this sweep |
| 28 | align(primitives × indices per primitive, 16) |

Object layout child 8 field 2 follows payload bytes, and field 4 follows the
declared maximum object threads. `FIELD-MODEL.json` checks these independently
authored inputs against the extracted fields, with **zero mismatches**. Other
reported raw fields include relative pointers and are not treated as sizes.

Field 6 plus field 28 predicts the existing local output layout. For the
accepted V256, seven-float4-varying case, application output is 32768 bytes,
while that sum is **32800 bytes** including counts and indices. The production
compiler currently limits the combined local allocation to 32768. This is a
software coverage gap; the correct launch allocation needs device measurement.

### Instruction operand census

`inspect_fields.py` records unordered, unique six-byte control/submit candidates
and their individual operand fields. It does not decode or retain surrounding
instructions, sequence order, offsets, or helper algorithms. The aligned
singleton search is explicitly a candidate census, not a proof of boundaries.

The broad shapes retain the known control and submit register forms. Control
flags 44 and 45 remain set; flag 42 clears for the V64/P1 case but remains set
in the other cases. Register operands change. This does not establish the
runtime values of those registers or flag 42's semantics. Submit forms retain
modes 0 and 2. Larger output interfaces do not require a newly observed opcode.
The compiler must determine its own publication algorithm from behavioral
experiments; no Metal helper sequence is an input to ours.

### Retained T8132 VDM DATA

`check_state_data.py` re-extracts the classified encoder regions in the existing
own-MSL xmat traces 7 and 8. Both identify chip 0x8132. **138 captures across 69
session/case pairs** match the state model, including lines, points, V256,
P256, primitive stride changes, culling, thread widths and payload variants.
`STATE-DATA.json` retains source hashes and exact data coordinates. This is
existing hardware data, separate from the new offline compilations; it is not
a new native-program qualification run.

With VDM header at word 0, plane count n, maximum primitives P, maximum vertices
V, mesh threads T, and primitive scalar count C:

| Word | Confirmed state model |
| --- | --- |
| 1 | n + ((n + 1) << 8) |
| 2 | 0xd0 with primitive data, otherwise 0x10 |
| 10 | (P − 1) + ((ceil(P / 16) − 1) << 16), plus bit 30 for culling |
| 11 | 0x400800 OR T |
| 12 | P × C |
| 13 | 1 + ceil(V / 4) × n, plus ceil(P / 4) for the culling case |
| Draw header | 0x70000000 points; 0x70000100 lines; 0x70000600 triangles |

Words 3/4 are zero, words 5/7 are 0x01000000, and words 6/8 are relocatable
launch handles. Word 9 varies with empty-output cases; its general meaning
is unresolved and is not inferred solely from whether an object stage exists.
The reader bounds each first state/draw at its terminator or next state; it
does not attribute later buffer contents to the first draw's topology.

The existing DATA-only launch comparison remains at
`tmp/codex-warthog/metal-launch-compare-20261019/SECTION.md` in the workspace.
It contains classified object/mesh allocation, frame, argument and target
records for the narrow class and V64/P32 with a 32-byte payload. The new
offline archives do **not** provide live runtime launch records for all 71
shapes. Wider allocation/payload/primitive-link launch behavior remains a
device experiment, not something established by metadata alone.

## Device-kit dependency and next experiments

The pinned revision-2 parser (`7b61c4f84c08`) rejects broad state before it
reaches the GPU. `check_admission.py` compiles that exact parser and checks
non-submitted fixtures: the baseline passes both modes; V4/P2, V16/P16,
V64/P128, V128/P256, V256/P512, points, lines and primitive data all reject.
Mode zero performs no reads. See `ADMISSION.json`.

A research kernel admission extension is therefore required for the capability
kit. Keep the revision-2 debugfs lifecycle and existing UAPI. Full-count
experiments start with small triangle counts, then test the declared maxima,
points/lines, varyings, primitive records, payloads, and runtime/empty counts.
Our generated programs must use ordinary IR/encoding and exact source-derived
oracles. A reference triangle does not count repeated primitives; the kit
needs an additive floating-point raster oracle or another qualified observable
for primitive multiplicity, plus high-index vertex controls and full owned
buffer/guard checks. Author-supplied count snapshots alone are not hardware
primitive counters.

Harness v2 retains one persistent owner/device per mode, one command in flight,
an exact program/launch/state identity audit, and fault/timeout backing
retention. Kernel admission is not GPU qualification. Freeze packets and
oracles before the manager schedules codex-hw. Production generalization and
any advertised-limit changes follow measured native behavior and review.
Native remains default-off.

## Reproduction

From this directory, generate into a scratch directory, copy it and `compile.py`
to the controller, and run `python3 compile.py` there. Copy results back, then:

```sh
python3 inspect_fields.py SCRATCH/offline SCRATCH/INSPECTION.json
python3 model.py SCRATCH/INSPECTION.json SCRATCH/FIELD-MODEL.json
python3 check_state_data.py --reader WORKSPACE/tmp/codex-warthog/metal-launch-compare-20261019/audit.py --out SCRATCH/state-data
python3 check_admission.py --kernel WORKSPACE/linux-m4-integration --out SCRATCH/admission
```

The new work accesses only the controller for offline compilation. It has not
accessed, reset, installed on, or submitted work to the M4 target.

## C2: independently compiled count/topology pilot

`compile_native.c` now constructs whole native programs from our own NIR using
normal selection and register allocation. The device-kit source is
`../../../vulkan/tests/native-capability-v2/`. It contains separate count and
thread-width experiments, all three topologies, runtime/zero counts, a flag-42
comparison, exact full-buffer guards and an additive R32_SFLOAT raster oracle
qualified by ordinary draws. It intentionally starts with Position/PointSize;
varying/primitive/payload device tiers remain pending.

The frozen C2 pilot contains **47 programs and 175 runtime arms**: 32 normal
shapes with five runtime-count patterns, and 15 separate high-index controls.
The latter make only the last vertex visible, through index 255, and require
exactly one covered primitive. Their earlier point/line geometry lies outside
the viewport, so zero-length line behavior cannot imitate a passing control.

All 175 arms passed the host DRM-shim code/launch/state/argument/PPP identity
audit, and all C-materialized expected buffers matched the Python source
oracle. An independent geometric calculation checked every pixel count. Every
oracle has a full-byte mask over the entire 512-KiB owned buffer. The ordinary
render-control executable traversed all 21 calibration draws on the host shim.
These checks performed **zero GPU trials**. `C2-HOST.json` records program
hashes/resources and the exact scope of the evidence.

The kit freezes 47 actual carrier streams for the kernel owner. A negative
control with the old parser accepts only three narrow states and rejects 44,
including the wider threadgroups. The new research mode (expected value 3)
must admit every frozen fixture through the kit's `check_parser.py` before a
device run. Its actual mode and exact image/DTB/release/version pins are passed
in a separate requirements JSON; this keeps the program/oracle kit immutable
while the kernel artifact is finished. The manager schedules codex-hw.

Kit: `../native-capability-20261022/c2-r8/kit/` relative to the Mesa checkout.
Its README contains the parser and device commands, safe-boot contract, lease
budget and stop/collection procedure. Production eligibility, advertised
limits, varying/primitive/payload device tiers and hardware qualification
remain pending. The M4 target has not been accessed during this preparation.
