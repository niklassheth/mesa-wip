# Production U0 completion rule, 2026-10-21

## production-u0

The user's U0 hypothesis is supported by the decisive r4 experiment, reported
by codex-hw at `b54cc5d2271` in `tmp/codex-hw/mesh-u0-producer-r4/RESULTS.md`.
All 352 draws qualified and the references reproduced. For both masks 0x01
and 0x02, BEFORE_U0 failed 10/10 and AFTER_U0 passed 10/10; M00 and M04 failed
at both sites. This supports work issued by U0, without separating slot
ownership. H4 (one completion class spanning slots 1 and 2) and H5 (multiple
producers or domains) remain unresolved. The production rule therefore
retires the deliberately conservative mask 0x03, without claiming an owner.

### Compiler model

`MESH_CONTROL` declares a resultless `completion_effect_mask` in its encoding
descriptor. The physical scheduler tracks that effect independently of GPR
results and encoded producer tags. `pack_mesh` is unchanged: U0 still has no
output tag, result register, or invented input field.

Every unconditional completion barrier uses `requires_prior_completion`:
object call/return/resume, U0, workgroup barrier, device fence, halt, tile
access/fence, coverage, depth store, and both mesh submits. Conditional
logical-end and loop-control drains retain their existing CFG rules.
The pending effect flows through forward edges and unions at joins; incoming
loop edges drain before control changes. Issuing another U0 retires the old
one first. Retirement clears the effect. A later barrier with no reaching
pending work adds no wait. Slot allocation avoids the conservative occupied
set, and fixed-slot producers retire that whole set before reuse.

Existing result folding still requires a named validated dependency layout.
The current VIR represents a single input slot, not a multi-slot input mask;
the conservative resultless 0x03 obligation uses scheduler-placed WAIT. No
placement, register assignment, padding, or instruction sequence was taken
from Metal. There is no research override in the production compiler.

### Clean merge candidate

`apple9/native-mesh-production-u0-merge` starts at integration-next
`95d4caa95b0` and takes only the compiler changes from `b58c3ee907c` and
`e00e17fe7c0`, including the unit tests and this design/byte-comparison record.
The research history is retained separately on
`research/native-mesh-wait-localization` at `44e8a4aaf87`. No campaign runners,
r2-r4 kit sources or probe scaffolding are added by this candidate.

The merge candidate adds three allocation regressions: pending U0 excludes
slots 1/2 from free allocation; exhaustion of slots 3-6 retires the whole
0x03 effect before reusing 1/2; and a slot-1 load reaching a join alongside
U0's effect gets a standalone WAIT 0x03, with no folded slot-1 input field.
A fourth static packing test exhausts all encodings, including PSEUDO, for
each of the 13 former opcode-list barriers; every accepted encoding must
carry `requires_prior_completion`.

`HK_APPLE9_NATIVE_MESH` remains default-off. Passing the production device
run is required before merging. The ordinary hk N32 CTS subset and the gate
lift are separate later steps, and this candidate is not in the release.

### Initial production-rule host validation

All ten compiler Meson tests pass, including 513 unit tests and the native
shared, workgroup, object, primitive, mesh and limit fixtures. The focused new
cases cover U0 followed by fence, either submit, repeated U0, both branch
arms, all-retired and partially-retired joins, a loop backedge, program exit,
and collision with a fixed slot-1 producer. The native mesh Honeykrisp host
test also passes. No hardware or CTS was run.

The ordinary corpus remains **7025/7025 byte-identical**, compared with the
held consumer foundation `b58c3ee907c`: 5 large-GS, 69 Lethal Company, 268
Willems, 17 vkmark, 768 CTS, 160 CS2 and 5738 robust entries. No empty output.
Baseline and candidate identity JSON SHA-256:
`3f8611f78507763c652611bfd0a4a8fb727fe05e3cf9b4fd993bd85fa73c8c5c`.

All 49 native mesh compiler fixtures change by **exactly one two-byte WAIT
0x03 before DEVICE_FENCE**. All existing instruction bytes, registers,
metadata and branch displacements stay identical in these 49 fixtures.
`compare.py` verifies the complete stream and rejects other changes. Exact
changed fixtures (pressure and half-z variants are distinct programs):

| Fixture | Pressure | Half-z | Added WAIT offset |
|---|---:|---:|---:|
| ATTR_SMOOTH | 0 | 0 | 718 |
| ATTR_FLAT | 0 | 0 | 718 |
| ATTR_LINEAR | 0 | 0 | 718 |
| ATTR_MIXED_SPARSE | 0 | 0 | 1012 |
| ATTR_LIMIT | 0 | 0 | 8866 |
| PARTIAL_T1 | 0 | 0 | 610 |
| PARTIAL_T2 | 0 | 0 | 606 |
| PARTIAL_T3 | 0 | 0 | 614 |
| PARTIAL_T4 | 0 | 0 | 604 |
| PARTIAL_T5 | 0 | 0 | 604 |
| PARTIAL_T6 | 0 | 0 | 604 |
| PARTIAL_T7 | 0 | 0 | 604 |
| PARTIAL_T8 | 0 | 0 | 604 |
| PARTIAL_T9 | 0 | 0 | 604 |
| PARTIAL_T10 | 0 | 0 | 604 |
| PARTIAL_T11 | 0 | 0 | 604 |
| PARTIAL_T12 | 0 | 0 | 604 |
| PARTIAL_T13 | 0 | 0 | 604 |
| PARTIAL_T14 | 0 | 0 | 604 |
| PARTIAL_T15 | 0 | 0 | 604 |
| H1_BASE | 0 | 0 | 592 |
| H1_SHIFT_X | 0 | 0 | 586 |
| H1_BASE | 1 | 0 | 1558 |
| H1_CLIP_CONTROL | 0 | 0 | 592 |
| H1_CLIP_CONTROL | 0 | 1 | 738 |
| H1_CLIP_BOUNDARY | 0 | 0 | 596 |
| H1_CLIP_BOUNDARY | 0 | 1 | 742 |
| H1_CLIP_FULLSCREEN | 0 | 0 | 592 |
| H1_CLIP_FULLSCREEN | 0 | 1 | 738 |
| H1_CLIP_X_125 | 0 | 0 | 588 |
| H1_CLIP_X_125 | 0 | 1 | 734 |
| H1_CLIP_X_150 | 0 | 0 | 588 |
| H1_CLIP_X_150 | 0 | 1 | 734 |
| H1_CLIP_X_175 | 0 | 0 | 588 |
| H1_CLIP_X_175 | 0 | 1 | 734 |
| H1_CLIP_NEGATIVE_X_175 | 0 | 0 | 590 |
| H1_CLIP_NEGATIVE_X_175 | 0 | 1 | 736 |
| H1_CLIP_Y_175 | 0 | 0 | 588 |
| H1_CLIP_Y_175 | 0 | 1 | 734 |
| H1_CLIP_NEAR | 0 | 0 | 590 |
| H1_CLIP_NEAR | 0 | 1 | 736 |
| H1_CLIP_NEAR_OUTSIDE | 0 | 0 | 592 |
| H1_CLIP_NEAR_OUTSIDE | 0 | 1 | 738 |
| H1_CLIP_FAR | 0 | 0 | 590 |
| H1_CLIP_FAR | 0 | 1 | 736 |
| H1_CLIP_FAR_OUTSIDE | 0 | 0 | 588 |
| H1_CLIP_FAR_OUTSIDE | 0 | 1 | 734 |
| H1_CLIP_POSITIVE_W | 0 | 0 | 592 |
| H1_CLIP_POSITIVE_W | 0 | 1 | 738 |

### Production device fixture

CURRENT is recompiled from its unchanged source/telemetry fixture with the
production scheduler. The GPU binary is used whole, without probe detours,
byte patching, extra padding or fixed-register insertion. The research carrier
and known object shader remain identical to the old CURRENT controls.
The production change is in the mesh compiler, not the private loader.

Both recompiled fixture variants add two scheduler WAITs. N32 inserts them
at offsets 3688 and 3766; N1 at 3620 and 3698. The first retires the effect
before the lane-zero phase fence. The second retires the reaching obligation
from lanes/paths that bypass that fence, at the next unconditional fence.
One existing conditional jump displacement grows by two bytes. Every other
instruction byte remains identical, including U0 and all submit encodings;
GPR counts and the ABI remain unchanged. This is the CFG union rule, not an
unconditional repeat wait after a globally retired effect.

Kit preparation and the sealed handoff are recorded below once complete.
The next step is claude-compiler's review of code and kit, followed by
codex-hw's production-rule device check. An ordinary hk N32 CTS check and
the gate lift remain subsequent work. No push or hardware is authorized here.
