# Spill traffic and latency work — 2026-10-06

Compared with uniform-family tip `c62490d01c2`, compiler `1f4c8abe909` reduces static SAVE+FILL from **24,164 to 16,724 (-30.79%)**. The loop-weighted estimate falls from **5,443,928 to 1,845,768 (-66.09%)**.

The fixed cohort is the 320 previously at-cap programs: 302 fragment, 14 vertex and 4 compute variants. All baseline spill traffic occurs in that cohort. Final at-cap count is 320; 0 new programs reach it. 271 programs have lower weighted spill traffic, 0 have higher traffic, and 9890 are unchanged.

These are host compiler results over all 10,161 captured main programs and 6,100 preambles, retaining capture multiplicity. They are not an in-game speed measurement. Loop weight is `8 ** min(depth, 6)`; the capture reaches depth three. Actual trip counts, branch probabilities and draw/dispatch frequencies are not measured.

## Changes and costs

- Rematerialize scalar, cheap expressions rooted in constants and immutable uniforms, with a four-instruction tree limit. Reconstruct every dependency and preserve the selected encoding and register constraints. Reject shaders containing uniform stores. Reserve the exact worst temporary demand under late-kill accounting, including edge reloads.
- Move stores of loop-invariant SSA values to a unique preheader when the value already has a register or scratch copy there. This does not extend its GPR live range. Loop-carried phi destinations stay inside the loop.
- Compare completed allocations with and without uniform rematerialization when scratch remains. This rejects cases where temporary reservation would add costly spills, including eight variants that otherwise acquired spills from a zero-spill baseline.
- Rank sample/reorderable-load hoisting by loop-weighted spill traffic, then static spills and instructions. The weighting change alone selected the same code in this capture.

| Stage | SAVE before → after | FILL before → after | Weighted SAVE+FILL before → after |
|---|---:|---:|---:|
| fragment | 12,077 → 6,389 | 10,640 → 8,936 | 5,442,481 → 1,844,369 |
| vertex | 108 → 84 | 108 → 84 | 216 → 168 |
| compute | 422 → 422 | 809 → 809 | 1,231 → 1,231 |
| all | 12,607 → 6,895 | 11,557 → 9,829 | 5,443,928 → 1,845,768 |

Main instruction count changes by -8,851; copies by +869; swaps by -1,176. The sum of per-program scratch-frame sizes changes by -2,624 bytes. 16 individual frames grow, by at most 16 bytes. All 6,100 preamble records, including code size, instructions and written uniform words, are unchanged.

## Where the baseline traffic comes from

Kinds follow the original defining instruction through MOV/SPLIT aliases. “Loop-header phi” is a direct loop-carried phi; computations using one remain arithmetic. Untracked coupling/lowering traffic is kept separate.

| Value kind | SAVE | FILL | Weighted SAVE+FILL |
|---|---:|---:|---:|
| arithmetic or other | 5,822 | 6,379 | 1,968,064 |
| phi | 3,525 | 2,277 | 1,815,407 |
| loop-header phi | 1,043 | 1,010 | 504,800 |
| memory result | 1,260 | 739 | 777,452 |
| texture result | 327 | 684 | 1,011 |
| uniform import | 448 | 320 | 272,704 |
| untracked coupling | 148 | 148 | 104,456 |
| constant | 34 | 0 | 34 |

Uniform-derived values overlap those primary categories: 818 SAVEs and 1,112 FILLs (515,618 weighted events). Within eight physical instructions of a sample in the same block: 403 SAVEs and 288 FILLs. Proximity does not prove a sample remains pending or caused the spill.

| Loop depth | SAVE | FILL | Weighted SAVE+FILL |
|---|---:|---:|---:|
| 0 | 1,139 | 1,493 | 2,632 |
| 1 | 1,509 | 1,197 | 21,648 |
| 2 | 4,023 | 5,395 | 602,752 |
| 3 | 5,936 | 3,472 | 4,816,896 |

## Remaining traffic after these changes

Proved uniform-derived traffic falls from 1,930 to 34 instructions: the remaining 34 are depth-zero constant stores, with no FILLs. Direct texture-result traffic is unchanged. FILLs now account for 87.3% of the weighted spill estimate.

| Value kind | SAVE | FILL | Weighted SAVE+FILL |
|---|---:|---:|---:|
| arithmetic or other | 3,770 | 5,154 | 640,317 |
| phi | 1,283 | 2,261 | 564,650 |
| loop-header phi | 954 | 1,019 | 369,424 |
| memory result | 427 | 611 | 215,028 |
| untracked coupling | 100 | 100 | 55,304 |
| texture result | 327 | 684 | 1,011 |
| constant | 34 | 0 | 34 |

## Hoisting audit

Both scheduling choices were measured for all 320 at-cap programs. Hoisting samples and reorderable loads raises static and weighted spill traffic in 193 programs and lowers static traffic in 0. Forced hoisting totals 28,874 static spill instructions and 7,254,302 weighted events, versus 24,164 and 5,443,928 without hoisting. The existing comparison already avoids that excess: it chooses the unhoisted schedule in 234 programs. Equal-spill cases still use instruction count to choose a schedule.

## Weight sensitivity

| Weight per nesting level | Before | After | Change |
|---|---:|---:|---:|
| 1 | 24,164 | 16,724 | -30.79% |
| 4 | 766,256 | 292,196 | -61.87% |
| 8 | 5,443,928 | 1,845,768 | -66.09% |
| 16 | 40,992,104 | 13,121,264 | -67.99% |

## Validation and artifacts

- 204 focused compiler, allocator, spilling and coalescing tests pass. Tests execute reconstructed values across uniform mutation and compare three loop iterations before/after spill placement, including loop-carried values.
- Full host replay passes. The diagnostic driver checks the post-spill register-demand bound even in release builds. Its main metrics match the uninstrumented driver exactly; the baseline also exactly reproduces the frozen uniform-family results.
- No mini access, resets, installation or GPU tests were performed for this work. Hardware correctness and game performance await an available slot.
- [Aggregate results](../MEASUREMENTS.md#cs2-pressure), [the fixed 320-program cohort](../MEASUREMENTS.md#cs2-pressure), and [the retained diagnostic kit](spill-census/README.md). Raw host runs are under `tmp/cs2-pressure-20261006/spill-analysis/` in the workspace.

## Frozen driver for the next uniform-family A/B

The requested pre-spill compiler is **`c62490d01c23d7ede0a152d4d1033176b442ce80`**. Use `artifacts/mesa-uniform-c62490d01c2/aarch64-game/libvulkan_asahi.so`, with its neighboring ICD and manifest. It retains the prior game-test runtime (`1a8869952423256baaa52097db13aaa250fa84ab` plus the exact source-B overlay), including GPU busy timing instrumentation. Every noncompiler source-B manifest hash was verified. Driver SHA-256: `d2396d2bfea94d962030b4eb1da5f527345f6b2c79c4597d41f1acd1dbdec1b2`.

This A/B driver intentionally predates the new spill work. The sibling `aarch64/` library uses the compiler checkout’s older runtime and should not be used for a controlled compiler-only comparison. The frozen compiler source archive and build instructions/provenance are in the same artifact directory. Everything remains local.

### Toolchain provenance correction

The runtime source match above does **not** make that earlier artifact a fully
controlled compiler-only build: it used cross GCC 16.1.0, whereas frozen game B
records GCC 16.2.1 (Red Hat 16.2.1-2), plus a 16.1.1 contribution in `.comment`.
The `1f4c8abe909` handoff must match frozen B's native toolchain and flags. Its
pinned source and CTS kit are prepared in `artifacts/mesa-spill-1f4c8abe909/`;
a matched native library and hardware validation are pending the mini slot.
See [the next latency levers and probe plan](NEXT-LATENCY-LEVERS.md).
