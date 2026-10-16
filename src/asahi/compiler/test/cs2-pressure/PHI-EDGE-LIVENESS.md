# Original phi-edge copy liveness

Host-only attribution of the original phi-edge ordinary copies, before and
after [constant placement](CONSTANT-PLACEMENT.md). This adds no allocator
change. The original population is direct-SAVE `caf135c0d93`; the placement
population is `7e6bc8da2be`. The mini was not accessed.

## Fragment result

Of the original **73,256** phi-edge moves, **17,075** have a source still live
after the edge; **56,181** have a source that dies there. Death at the edge is
necessary for ordinary SSA coalescing but does not make every such move
avoidable. The following disjoint partition accounts for every original move.

| Classification | Original population | After constant placement |
|---|---:|---:|
| Source remains live after the edge | 17,075 | 17,075 |
| Source dies, but source/destination interfere elsewhere | 2,604 | 2,647 |
| Source dies, pair does not interfere, another phi already uses its register | 18,111 | 18,059 |
| Additional moves from a repeated source in the same phi bundle | 222 | 262 |
| Distinct candidate coalescing opportunities | **35,244** | **38,143** |
| Total original-phi ordinary copies | **73,256** | **76,186** |

The candidate count is an **upper bound under the existing SSA interference
and distinct-phi-destination model**, not a promised instruction reduction.
Register-bank legality, class interference, other incoming edges and parallel
copy cycles can further constrain simultaneous coalescing. Moving a source to
one phi destination when a sibling already shares its register merely moves
that sibling's copy elsewhere in the bundle. At most one candidate is counted
per remaining source group on an edge.

Sources that remain live conflict with the phi destination under the current
interference model. This is the relevant necessary-copy category for the
existing allocator; it is not proof against a future value-aware aliasing or
live-range transformation. Similarly, a constant-source opportunity can trade
an edge copy for a high-register materialization bridge rather than remove an
instruction. Keep those costs in the completed-allocation comparison.

Of the original 35,244 candidates, **32,127 have computed sources and 3,117
have constant sources**. After placement these become 35,205 and 2,938.
Thus placement's 2,930 additional original-phi copies increase this candidate
bound by 2,899, mostly in computed-value coalescing. It does not increase the
still-live-source count.

| Fragment candidate edge | Original population | After placement |
|---|---:|---:|
| Ordinary join | 25,670 | 27,978 |
| Loop entry | 6,378 | 6,770 |
| Loop backedge | 3,196 | 3,395 |

26,487 of the original candidates occur in outputs at 96 GPRs, versus 8,757
below the cap. After placement the split is 29,184 / 8,959. These are static
counts, without claiming equal execution frequency across edges or shaders.

## Other stages

| Stage / population | Original phi copies | Source live after edge | Source dies | Distinct candidate bound |
|---|---:|---:|---:|---:|
| Vertex, direct-SAVE | 10,893 | 7,388 | 3,505 | 2,879 |
| Vertex, placement | 10,987 | 7,388 | 3,599 | 2,971 |
| Compute, direct-SAVE | 17,104 | 7,784 | 9,320 | 8,903 |
| Compute, placement | 17,277 | 7,784 | 9,493 | 9,082 |

The complete partitions, constant/computed splits, edge kinds and register
pressure cohorts are in [phi-edge-liveness-summary.json](../MEASUREMENTS.md#cs2-pressure).
Per-program pipeline hashes and counts are in [phi-edge-liveness.csv](../MEASUREMENTS.md#cs2-pressure)
and [phi-edge-liveness-after-placement.csv](../MEASUREMENTS.md#cs2-pressure).

## Method and verification

The diagnostic overlay snapshots original phi sources after spilling and SSA
repair, before allocation erases SSA identities. Successor `live_in` describes
the state after the parallel phi bundle. A source is still live if it is in
that set and is not itself redefined by a successor phi; this distinction also
handles an old loop-iteration value whose SSA number is reused by a phi.

A separate diagnostic interference graph includes constants as well as computed
phi participants. It uses the same whole-CFG liveness rules as the allocator,
but does not change the allocator's private-constant affinity classes. Direct
source/destination interference is recorded separately from edge-local death.
Before SSA lowering, final assignments identify repeated-source fanout and any
phi already coalesced with the source. Metadata follows actual emitted parallel
copies through physical lowering and scheduling. Only the selected allocation
and selected shader trial are counted; swaps are separate from ordinary IOR
moves. One eligible emitted move represents each source group on each edge.

Both whole replays completed with **10,161 main outputs and 6,100 preambles**,
exit 0. Each entire main metrics JSONL is byte-identical to its frozen,
non-instrumented control. All origin tags are known; no source identity changed
between snapshots, and every still-live source has the expected interference.
No counted original-phi copy has a memory/vector or redefined-source exception.
The focused 100–150 graphics control also matches exactly.

The fresh direct-SAVE census independently confirms every source category used
in the constant-placement comparison: relative to the earlier stage-four census,
only the predicted 7,664 / 222 / 414 SAVE-staging copies disappear. This verifies
the baseline normalization in the placement report.

Evidence: `tmp/cs2-pressure-20261005/phi-edge-analysis/`, including all three
instrumented source files, diagnostic headers, build/parse scripts, raw selected
copy categories, tests of control identity and full per-program summaries.
`phi-edge-census-host.tar.gz` archives this evidence with source/hash provenance.
It depends on the existing workspace toolchain/support libraries and is not a
portable build environment. No diagnostic code was added to production.

## After dead-source placement and uniform promotion

The final `ff53faea9e7` census includes `abe5ef899fa`'s dead-source phi placement
and the complete uniform-source / loop-weighting series. It is not an isolated
measurement of the phi placement change. All 10,161 main metric rows are
byte-identical to the final uniform measurement, with 6,100 preambles; every
origin and source-identity check passes.

| Original-phi ordinary copy classification | Fragment | Vertex | Compute |
|---|---:|---:|---:|
| Source live after edge | 17,278 | 6,560 | 7,161 |
| Dead source, pair interferes elsewhere | 2,676 | 0 | 15 |
| Dead noninterfering source, peer already coalesced | 16,631 | 794 | 539 |
| Additional move from repeated-source fanout | 240 | 48 | 91 |
| Distinct candidate coalescing opportunities | 32,558 | 2,945 | 7,899 |
| Total remaining original-phi ordinary copies | 69,383 | 10,347 | 15,705 |
| Constrained fraction under the current model | **53.07%** | **71.54%** | **49.70%** |

“Constrained” is total minus the distinct candidate upper bound: live/interfering
pairs and moves required by distinct-destination fanout. Across stages it is
52,033 of 95,435 remaining copies, or 54.52%. This is a lower bound on required
moves under the current SSA/interference/distinct-destination model. It is not
a proof against value-aware aliasing or a different live-range transformation.
The other 43,402 copies are an opportunity upper bound, not guaranteed savings.
Of the fragment candidates, 30,440 are computed values and 2,118 are constants.

The fragment population falls from 76,186 copies after low-constant placement
to 69,383 here, while its candidate bound falls from 38,143 to 32,558. Keep the
complete instruction/copy/spill cost comparison in
[UNIFORM-SOURCES.md](UNIFORM-SOURCES.md), since coalescing and uniform promotion
can also move costs outside phi edges. Full stage, edge and pressure partitions
are in [phi-edge-liveness-after-uniforms.json](../MEASUREMENTS.md#cs2-pressure).
Per-program records and pipeline hashes remain in
`tmp/cs2-pressure-20261005/phi-edge-analysis/final-whole-summary/programs.csv`.
