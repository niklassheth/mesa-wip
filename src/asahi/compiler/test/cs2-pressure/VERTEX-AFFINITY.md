# Private constant-edge affinity: vertex follow-up

The three phi-affinity commits remain unchanged: `3f9e1013eb7`,
`be242d6ce10`, `d08757466f3`. This fourth change restores a useful constant
register hint without restoring the shared-constant affinity bridges.
The analysis below was prepared on the host while the mini was reserved for
the main thread. Subsequent [mini replay and CTS validation](CTS-STAGE4-20261005.md)
passed; its separately labelled results supersede the pending status below.

## Distribution from the saved mini replay

[All 127 regressing vertex outputs](../MEASUREMENTS.md#cs2-pressure) are listed by
zero-based whole-replay row, vertex ordinal and Fossilize pipeline hash, with
metrics for the original combined baseline and each of the three stages.
Pipeline hashes identify work; a pipeline can emit multiple outputs or be
replayed as a dependency. The logged pipeline index can be relative to a
Fossilize dependency batch, so it is not always a standalone global range.
The appended host columns are separately labelled.

Of 5,045 vertex outputs, 127 grew and 168 shrank at stage 3. Regressions added
7,875 instructions, improvements removed 2,200: net +5,675. The median
regression was 42 instructions and the maximum 224. This is a distributed
family, not one dominant outlier: the top ten contribute 25.79% of gross growth,
the top twenty 45.52%, and the top fifty 80.15%.

| Added instructions | Outputs |
|---|---:|
| 1–9 | 39 |
| 10–49 | 27 |
| 50–99 | 29 |
| 100–199 | 25 |
| 200+ | 7 |

120 of the 127 already regressed at stage 1. None first regressed at stage 2;
the seven first regressing at stage 3 each grow by just one instruction.

| Whole row | Pipeline hash | Added instructions | Added swaps |
|---:|---|---:|---:|
| 309 | `075fc8894ab69e28` | 224 | 71 |
| 5671 | `a0ed6eb8bf6c3996` | 223 | 74 |
| 4280 | `7961fee4679e7e07` | 221 | 73 |
| 7403 | `d127351bb35ac524` | 213 | 69 |
| 389 | `0990a9e9543f902e` | 212 | 70 |
| 4371 | `7bf7eb6cb9663da3` | 207 | 68 |
| 8963 | `fb9caae4964c947f` | 205 | 68 |
| 8104 | `e3ea56bbb838b271` | 182 | 61 |
| 7365 | `cf8ec04c16a8f7b4` | 175 | 63 |
| 2164 | `3fffb0b53d9e264b` | 169 | 58 |

## Mechanism: a discarded constant hint creates diamond-edge swaps

For pipeline `075fc8894ab69e28` (global range 170–171), the selected allocation
candidate is 14. Its pre-allocation SSA is unchanged. The old shared web passed
the literal's assigned register to the computed value on the other branch.
Stage 1 correctly isolated the literal, but its private virtual member did not
retain a register preference. Stage 3 could see direct phi edges and real SSA
class members, but not that virtual member's already-known incoming register.

One repeated instance is the block 10 / block 11 diamond joining at block 12:

| Value / phi | Original combined | Stage 3 |
|---|---|---|
| Literal SSA 202, block 10 | h128 | h128 |
| Computed SSA 206, block 10 | h132 | h132 |
| Computed SSA 210, block 11 | h128 | h132 |
| Phi 198 = (202, 210), block 12 | h128 | h132 |
| Phi 197 = (206, 196), block 12 | h132 | h128 |

SSA 196 enters from h190. Stage 3's block 10 → 12 parallel copy exchanges
h128 and h132. The original assignment needs no copy for either phi on that
edge. SSA 202 has only one incoming phi use here: this local regression is
not a necessary consequence of preventing a constant from connecting unrelated
webs, nor an interference-check split. The missing private-member hint is the
cause. Weighted preference ordering has little opportunity to help when the
candidate preference itself is absent.

Final assembly and assignment traces agree: baseline 17 swaps, stage 3 88,
fix 10. All stage-3 swaps are original phi-edge cycles; there are no allocation
shuffle swaps. The program has two loops, but all these swaps are at loop depth
zero. There are exactly 955 IMM instructions, 253 loads and 20 IMADs in all
three assemblies. Baseline → stage 3 adds 213 IXOR instructions (71 swaps),
12 IOR copies and removes one EXEC_MASK_ELSE: exactly +224 instructions.
This is not extra constant materialization.

A second large case, pipeline `a0ed6eb8bf6c3996` (range 3179–3180), has
5 → 79 → 2 swaps and 6,846 → 7,069 → 6,830 instructions. Its 828 IMM
instructions and 178 loads also remain unchanged.

## Fix

Each private constant-use member now references its own phi operand preference.
While scoring a compatible class member, RA includes that operand's assigned
register with the edge's existing loop weight. This remains a soft class hint:
direct phi-edge preferences retain priority, register legality is unchanged,
and the interference check still prevents conflicting real SSA members from
joining. No shared constant is reintroduced as a bridge between classes.
The fix adds no IR, materialization, rematerialization or fixed placement.

## Matched host whole-capture comparison

Native x86 CPU replay uses the same captured pipelines and a host-only fake
G16 DRM device configured from the public kernel's T8132 parameters. No GPU
submissions occur. All variants use one Fossilize worker, release assertions
settings matching the prior replay, eager variants, disabled shader cache,
loop handling and explicit spill coalescing. The known capture NIR validation
exception (`NIR_DEBUG=novalidate`) remains limited to capture replay.

The stage-3 host control differs from the saved mini control in 36 of 10,161
rows (1 fragment, 27 vertex, 8 compute); the baseline differs in 37 rows.
Therefore these are **matched host comparisons**, not replacement mini results.
All three host runs completed with 10,161 outputs and successful exits.

| Stage | Original combined instr / swaps | Stage 3 instr / swaps | Fix instr / swaps | Fix vs stage 3 instructions | Fix vs original instructions |
|---|---:|---:|---:|---:|---:|
| Fragment (4,127) | 2,674,241 / 51,907 | 2,468,872 / 3,329 | 2,468,063 / 2,895 | −809 | −206,178 |
| Vertex (5,045) | 961,582 / 2,329 | 967,268 / 4,606 | 954,822 / 607 | −12,446 | −6,760 |
| Compute (989) | 1,593,950 / 1,285 | 1,591,997 / 571 | 1,592,652 / 766 | +655 | −1,298 |

The net change from stage 3 is −12,600 instructions and −4,238 swaps.
Fragment gains are preserved in aggregate, with another 809 instructions
removed; this is not a claim that every fragment improves. Relative to stage 3,
186 fragment outputs shrink and 138 grow; 98 vertex outputs shrink and 12 grow;
70 compute outputs shrink and 317 grow. The compute tradeoff is +655
instructions / +195 swaps relative to stage 3, while still beating the original
combined baseline.

90 of the original 127 vertex regressors now beat the matched original baseline.
The remaining 37 add only 1–3 instructions each (71 total); no new vertex
regressors appear relative to the original host baseline. Thus this fixes the
large swap regression, not every minor placement difference. Compared with
stage 3, none of the original 127 becomes worse: 90 shrink and 37 are unchanged.

Every output retains its SAVE count, FILL count and scratch allocation across
all three host versions. GPR assignments can change: fix vs stage 3 has 3/2
fragment outputs lower/higher, 0/1 vertex, and 10/16 compute. The number of
outputs at 96 GPRs remains 346 fragment, 22 vertex, 4 compute.
The original fragment pressure reproducer changes 4,019/17 instructions/swaps
to 4,017/15, with the same 191 SAVE, 157 FILL, 96 GPRs and 272-byte scratch.

Host tests with native assertions: 238/238 outside `Apple9Compiler.*`, including
the shared constant/computed initializer and spilling cases; separately 4/4
forced-spill `Apple9Spilling.*` tests pass. The excluded compiler suite has the
previously confirmed unrelated pre-RA `WritableSinkPointer` NIR assertion.
The AArch64 fix library cross-build succeeded. Mini replay and GPU correctness
remain pending the coordinator's “mini free” signal.

## Stage-3 CTS and in-game handoff

The unchanged stage-3 handoff is at
`/home/nsheth/Projects/asahi/tmp/phi-stage3-handoff/`:

- `stage3-matched-ab/`: preferred AArch64 in-game candidate. It keeps runtime
  base `514b023ef55` and all source files in the main A/B provenance manifest
  except `agx_register_allocate.c`, replaced with exact `d08757466f3` content.
  Shader-ring, runtime and GPU timestamp instrumentation remain intact.
  Library SHA256:
  `054633802fede3db6016e943ee3b4523b4d857429a11d097fc84724b4bc14675`.
- `stage3-d08757466f3/`: separate standalone compiler-branch release build.
- `run-stage3-cts.sh`: checks the immutable assertion-enabled stage-3 driver
  SHA256 before running 464 normal and 140 forced cases, three jobs on CPU
  cores 1–6 at nice 5. `verify-stage3-cts.py` checks outcomes/counts and requires
  passing forced cases with both SAVE and FILL in their own retained output.
  NIR validation stays enabled in CTS. Scripts were syntax-checked, not run.
- `private-constant-fix/`: separate standalone AArch64 fourth-change library
  for the forthcoming CPU replay. It does not replace either stage-3 artifact.

The in-game candidate is a recorded host cross-build, not byte-identical to the
old native build. Keep the main thread's existing native-thunk driver-switch,
cache warmup and measurement procedure; verify the loaded library mapping and
hash. No candidate has been installed, no target job started, and no reset
performed during this host-only work.

Raw host evidence is under
`tmp/cs2-pressure-20261005/vertex-analysis/`: the three `*-host.jsonl` files,
`fix-vs-stage3-host.json`, `fix-vs-baseline-host.json`, `host-vs-mini-stage3.json`,
full selected program assemblies, `*-170-audit/` traces, tests and build logs.
The CSV's first four versions are the saved mini results; its final three
versions are the matched host runs. Keep those measurement populations distinct.
