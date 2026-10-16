# Stage 4 compute review and ordinary-copy census

Compiler unchanged at `e495dbbfeea` on top of `d08757466f3`, `be242d6ce10`,
and `3f9e1013eb7`. No fifth compiler change is proposed for the small compute
regression. This report records the census before implementing copy changes.
The subsequent [dead-source SAVE experiment](SAVE-STAGING.md) removes the
predicted 8,300 staging copies. All census work here was performed on the host.

## Compute: real, small, and dispersed

Relative to stage 3, stage 4 adds 655 instructions and 195 swaps across 989
compute outputs: 0.0411% of stage-3 compute instructions. 317 kernels grow by
968 instructions gross, while 70 shrink by 313. Median growth is two
instructions; 232 regressors add 1–3, 78 add 4–9, and seven add ten or more.
The largest ten account for only 16.4% of gross growth. Of 155 kernels gaining
swaps, 103 gain one, 36 gain two, and 16 gain three; another 16 lose swaps.

[All changed compute outputs](../MEASUREMENTS.md#cs2-pressure) include stable Fossilize
pipeline hashes, indices, before/after metrics and the inspected mechanisms.
The capture provides no useful symbolic kernel names in these outputs.

| Compute index | Pipeline hash | Instructions | Swaps | Ordinary copies |
|---:|---|---:|---:|---:|
| 19 | `03e9c93a0bff575a` | +53 | +0 | +0 |
| 417 | `6d4b4181ae1390b7` | +16 | +0 | +7 |
| 320 | `57ecb3354857f055` | +15 | +3 | +0 |
| 763 | `cd986c1f4b7d40a5` | +14 | +3 | +0 |
| 901 | `ed59f627bcc7d0a3` | +14 | +0 | +2 |
| 482 | `7dbe2c460de263b3` | +10 | +2 | +0 |
| 770 | `ce688dad1f260833` | +10 | +3 | +1 |
| 56 | `103fdf4678899a3a` | +9 | +3 | +0 |
| 160 | `29c9c6d2b50c32e8` | +9 | +3 | +1 |
| 244 | `405a4d0ef106bc00` | +9 | +3 | +0 |

Three focused replays compare final stage-3 and stage-4 assemblies:

- `03e9c93a0bff575a`: 131,364 → 131,417 instructions. All 53 added
  instructions are WAITs. Swaps remain zero, copies 85, IMM 7,958, loads 1,083,
  IMAD 961, and GPRs 72. All other opcode counts are identical.
- `6d4b4181ae1390b7`: 5,885 → 5,901 instructions: seven extra copies and nine
  WAITs. Swaps remain zero, IMM 353, loads 104, IMAD 31, and GPRs 51.
- `57ecb3354857f055`: 1,221 → 1,236 instructions: three extra swaps (nine
  IXOR instructions) and six WAITs. Copies stay 40, IMM 143, loads 87, IMAD 25,
  and GPRs 52.

This is a placement and completion-folding tradeoff from the added private
constant hint, not extra constant materialization or scratch traffic. Across
all compute outputs, ordinary copies actually fall by three; 585 of the net
655 extra instructions are accounted for by the 195 added three-XOR swaps.
The remaining net 73 instructions are outside swaps/copies; the inspected
largest cases identify WAIT changes. This does not claim every remaining
instruction was individually classified.

The effect is deterministic, rather than measurement noise, but small enough
in the static aggregate and spread thin enough to leave alone. Compute still
beats the matched pre-stage-1 combined baseline by 1,298 instructions and 519
swaps. Avoid undoing the much larger vertex win to chase this tail. These are
static compilation counts, not an in-game timing claim.

## Counting and verification

The diagnostic build tags actual shared-IR moves by their insertion site and
phi origin, propagates those tags through physical lowering and scheduling,
and prints only the selected allocation and selected shader trial. Rejected
candidates are excluded. Each program's category sum is checked against its
final emitted IOR-copy count. The successful final replay produced 10,161 main
outputs plus 6,100 separately counted preambles.

The final diagnostic replay's **entire metrics JSONL is byte-identical** to the
frozen stage-4 host control, including instructions, swaps, copies, GPRs,
SAVE/FILL and scratch in every row. This validates the attribution against the
existing host population. It does not replace the deferred mini replay.

The historical copy metric counts IOR instructions with equal inputs. It
excludes the three-XOR sequences implementing swaps, but includes publication
writes and a handful of same-register materialization/wait instructions. The
strict GPR-movement column below removes those. Immediate materialization
instructions, uniform imports and uniform stores are separate operations.

## Copies by stage and register pressure

| Stage / cohort | Outputs | Copy-form instructions | Fraction of all instructions | Strict GPR moves |
|---|---:|---:|---:|---:|
| Fragment — all | 4,127 | 153,751 | 6.23% | 148,793 |
| Fragment — at 96 GPRs | 346 | 114,931 | 11.64% | 113,893 |
| Fragment — below 96 GPRs | 3,781 | 38,820 | 2.62% | 34,900 |
| Vertex — all | 5,045 | 20,225 | 2.12% | 20,225 |
| Vertex — at 96 GPRs | 22 | 8,726 | 5.91% | 8,726 |
| Vertex — below 96 GPRs | 5,023 | 11,499 | 1.42% | 11,499 |
| Compute — all | 989 | 63,263 | 3.97% | 63,261 |
| Compute — at 96 GPRs | 4 | 1,477 | 6.13% | 1,477 |
| Compute — below 96 GPRs | 985 | 61,786 | 3.94% | 61,784 |

Fragment copies are now 6.23% of instructions; the earlier mini loop-only
result was approximately 8.05% (216,662 copies). That comparison gives scale,
not an exact cross-architecture delta. Against the matched host original
combined baseline, stage 4 removes 57,248 fragment copies.

The 346 at-cap fragment outputs contain 74.75% of fragment copy-form
instructions. Compute differs: only four outputs reach the cap, and almost all
compute copies occur below it.

## Exact source attribution

| Source | Fragment | Vertex | Compute |
|---|---:|---:|---:|
| Original phi edges | 73,256 | 10,893 | 17,104 |
| Constants assigned above r63 | 41,906 | 7,594 | 13,536 |
| SAVE staging | 13,492 | 222 | 414 |
| Spill-repair phi edges | 6,443 | 10 | 48 |
| Live-in reconciliation phi edges | 3,280 | 33 | 22 |
| Constrained-operand placement | 6,570 | 341 | 572 |
| COLLECT lowering | 3,804 | 1,132 | 25,723 |
| SPLIT lowering | 42 | 0 | 388 |
| Private copies for destructive inputs | 0 | 0 | 5,401 |
| Publication writes | 4,922 | 0 | 0 |
| Other selected identity/materialization | 36 | 0 | 55 |
| **Total** | **153,751** | **20,225** | **63,263** |

[Category counts by pressure cohort](../MEASUREMENTS.md#cs2-pressure) and the
[machine-readable summary](../MEASUREMENTS.md#cs2-pressure) retain all subdivisions.
There are no unclassified copies. No direct generic eviction-copy category
survives; allocator reconciliation and constrained-operand moves are counted
at their actual insertion sites.

Original fragment phi copies split into 34,788 at ordinary joins, 28,006 on
loop-entry edges, and 10,462 on backedges. Including spill-repair and live-in
reconciliation gives 42,763 join, 28,006 loop-entry and 12,210 backedge copies.
These are non-swap MOVs actually emitted by parallel-copy lowering; moves in
an otherwise cyclic bundle still count, while its XOR swaps do not. This
classification is per edge, not a dynamic loop-frequency estimate.

The high-register constant copies come from the existing MOV-immediate
encoding limit: an assigned destination above r63 is materialized in the
reserved copy register, then moved to the destination. They are a real
instruction cost, distinct from duplicated IR constants. The fragment total
is spread across 677 outputs; its top ten contribute only 1,502 of 41,906.
Original fragment phi copies span 1,378 outputs, with just 2,140 of 73,256 in
the top ten. These are broad optimization targets.

SAVE staging contributes 13,492 fragment copies for 13,572 SAVEs; 80 SAVEs
already use the copy register and need no move. Ordinary FILL writes directly
to its allocated destination: there are no separate FILL staging copies in
this capture. The spill-repair and reconciliation rows account for GPR moves
at boundaries; the SAVE row accounts for physical lowering's staging move,
so those causes are not double-counted.

Constrained-operand placement is not synonymous with an exact fixed-register
operand: it also handles bank, tuple and destructive-input restrictions.
Fragment's 6,570 copies comprise 3,384 at device loads, 2,184 at texture LOD
operations, 996 at subgroup ballots and six at interpolation-position inputs.
This is much smaller than phi and constant placement.

Fragment's COLLECT copies comprise 3,452 for indirect device-load operands and
352 for indirect stores; all 42 SPLIT copies come from indirect device loads.
There is no surviving fragment COLLECT-copy category for texture-coordinate
or publication tuples. Texture-related constrained-operand copies are counted
above, and the 4,922 publication writes are explicit required handoffs rather
than ordinary GPR-to-GPR shuffles.

Compute's COLLECT cost is larger: 21,484 for indirect stores, 3,459 for indirect
loads, 610 for indirect atomics, and 170 other/mixed-consumer copies. It is
spread across 937 outputs; the top ten contribute only 781 of 25,723. These
counts identify the consuming instruction, not whether a particular tuple is
its address or payload. The current diagnostic does not claim that split.
Compute's 5,401 destructive-input copies comprise 4,283 for indirect atomics,
614 ballots, 404 shared stores and 100 shared atomics. Preserving a still-live
source for a destructive operand is not generally removable.

The selected-identity row includes 36 same-register fragment instructions and
two same-register compute instructions. These retain explicit materialization
or dependency semantics; they are excluded from strict GPR movement, not
assumed to be removable NOPs.

## Preamble and uniform work, separately counted

| Stage | Preambles | Preamble GPR copies | SPLIT / COLLECT | Main uniform imports | Preamble uniform writes |
|---|---:|---:|---:|---:|---:|
| Fragment | 4,126 | 12,847 | 11,523 / 1,324 | 49,953 | 91,768 |
| Vertex | 986 | 3,728 | 3,728 / 0 | 19,632 | 66,594 |
| Compute | 988 | 7,832 | 7,154 / 678 | 21,527 | 34,312 |

All preambles are below the register cap. Main uniform imports are
`IOR_UNIFORM` materializations of `load_preamble`. Uniform writes count
`STORE_UNIFORM` operations, including already-folded arithmetic store forms;
they are not all plain moves. None of these uniform transfers is included in
the historical ordinary-copy totals. Existing uniform-source and uniform-store
folding is already active. Static preamble and main counts also represent
different execution frequencies.

## Proposal and measured opportunity bounds

**First: elide staging for a SAVE whose source is already dead.** Preserve the
source's original last-use information across the allocator's early-kill
bookkeeping, then let SAVE consume that GPR directly instead of copying it to
the reserved staging register. The existing machine model accepts SAVE sources
through r95. Keep the physical scheduler's destructive-input and completion
hazards intact.

The diagnostic snapshots liveness before early-kill flags are cleared. Among
staging moves actually present in the selected output, it finds:

| Stage | Dead-source SAVE staging copies |
|---|---:|
| Fragment | 7,664 |
| Vertex | 222 |
| Compute | 414 |
| **Total** | **8,300** |

This is the clearest bounded next experiment: **8,300 candidate MOV removals**
under frozen assignments, including 4.98% of fragment copy-form instructions
and a 0.31% fragment instruction ceiling. It adds no SAVE/FILL operations or
scratch in that frozen model. It is not a measured compiler improvement:
waiting on the directly consumed register can change, and allocation-candidate
selection can change after its cost drops. Measure the resulting instruction,
WAIT, swap, GPR and scratch totals, then validate correctness on the mini.
Do not relax destructive-source semantics to obtain the saving.

**Second: price high-register constant materialization in placement.** There
are 41,906 fragment, 7,594 vertex and 13,536 compute high-bank constant moves.
At allocation of those same constants, a currently unused legal 32-bit register
below r64 existed for 36,853 / 7,592 / 11,477 respectively. A compact-register
slot below r16 existed for 27,545 / 7,448 / 11,009. These are candidate ceilings,
not predicted net savings: moving a constant to a low register can simply move
its cost onto a phi edge, or displace a future constrained operand.

Use the existing placement/candidate policy to include that extra instruction
cost, while retaining weighted real-edge affinity and checking completed
allocations. Avoid another whole compilation trial merely to test this policy,
given the main thread's compile-time work. The fragment candidate ceiling is
36.9k moves (1.49% of fragment instructions), but there is no honest net-gain
estimate until measuring displaced phi/operand copies. Keep stage 4's vertex
win and stage 3's zero coalescing-induced swap result as regression gates;
repeat the paired loop-only/coalesced measurement when testing a new policy.

**Then: joint phi placement and repeated tuples.** Fragment's 12,210 backedge
copies (10,462 original phis) are a more targeted hot-loop starting point than
trying to remove all 82,979 phi/reconciliation copies. The input layouts on
different predecessors conflict, so that total is not a removable budget.
Evaluate a bundle-level weighted assignment subject to interference instead
of independent scalar hints; compare total incoming-edge cost and downstream
constraint repairs before estimating a gain.

For compute, investigate reuse of identical dominating COLLECT tuples and all
uses' tuple affinities; the scalar-to-collect hint still remembers one consumer.
The 21,484 indirect-store COLLECT copies are the scope ceiling, not evidence
that all are redundant. Distinguish address and payload tuples and destructive
uses before changing the model. Neither large optimization is implemented.

## Artifacts and next hardware slot

Raw evidence and host-only diagnostic sources are in
`/home/nsheth/Projects/asahi/tmp/cs2-pressure-20261005/copy-analysis/`.
The final successful run is `opportunities-v3-whole`; its main JSONL is
byte-identical to `vertex-analysis/private-history-host.jsonl`.
`copy-summary.json` records the control checksum and exact category totals.
The diagnostic driver changes only logging/metadata in isolated copies of
three translation units; no repository compiler source was changed.

The four compiler commits remain intact. The stage-3 driver/CTS handoff and
separate stage-4 release driver from the previous report remain unchanged.
Mini replays and normal/forced CTS for **both stages 3 and 4**, with each run's
own spill verification, remain queued until the coordinator relays “mini free”.
No mini process, installation or reset was started for this analysis.
