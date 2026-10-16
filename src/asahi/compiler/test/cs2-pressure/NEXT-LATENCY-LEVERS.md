# Next compiler work after `1f4c8abe909`

The next priority is reducing repeated FILLs in hot loops. Direct ALU frame
outputs are a useful bounded encoding experiment, but rank below reload reuse
for expected GPU-time payoff. These rankings are hypotheses informed by the
capture, not measured speedups. No proposed GPU or HV probe below has run.

The coordinator's matched-scene ABAB of `32a6b6ee740` versus `e096cec38df`
reported fragment GPU time 4.164 → 4.009 ms/frame (-3.7%), total GPU -2.2%,
and FPS +1.0%, despite much larger instruction/copy/swap reductions. That
motivates this ordering; it does not establish a universal cost for moves.

## Ranked work

| Priority | Change to investigate | Evidence and plausible payoff | First experiment / acceptance condition |
|---|---|---|---|
| 1 | Preserve/reuse reloads across hot loop uses and phi edges; rematerialize cheap varying expressions only when their inputs already remain available | FILLs account for 1,610,981 of 1,845,768 weighted events (87.3%). Ordinary and loop-header phis contribute 784,319 weighted FILLs; FMA-derived values another 466,667 | Trace original value/home identity, use blocks and reload lifetime in the highest-weight programs. Compare bounded local reload reuse and edge placement against baseline allocation; retain only candidates reducing weighted traffic without new spills elsewhere. Never extend all reloads to the loop header blindly |
| 2 | Reduce peak pressure with local scheduling, narrower live tuples and fewer spill temporaries where legal | 291 of the 320 capped programs have scratch. Spilling reserves r0–r4 for copy/index/swap work, in addition to rematerialization temporaries. Existing global sample/load hoisting increases spills in 193 capped programs | Record total and constrained-register pressure at the peak for each chosen schedule. Try local producer sinking / last-use scheduling around that peak, retaining load latency overlap; measure completed allocation and game GPU time |
| 3 | Fold ALU producers into frame destinations after T8132 validation | FMA/FMUL/FADD-derived stores total 2,853 static SAVEs, but only 46,421 weighted stores (2.5% of remaining weighted spill events). Other result uses may prevent a fold | Single GPR control + frame-destination case per encoding family, then mask/lifetime checks. Recount actual producer→SAVE eligibility before implementation. The write still goes to scratch: an instruction/GPR reduction is not proof of less memory traffic |
| 4 | Expand validated high-register forms using the prepared authored Metal/HV kit | May relax low-bank bottlenecks and indirectly prevent spills; current capture does not quantify that benefit | First obtain compiler-emitted high operands, then an isolated execution oracle. Prioritize texture-result destinations and compact FP forms if pressure census shows they bind; avoid speculative bit sweeps |

Priorities 1–2 overlap. Their estimates must not be added. Rematerializing a
varying expression is worthwhile only when it saves reload latency without
reloading its operands or raising the peak. Preserve exact FP semantics and
exclude memory operations, derivatives, subgroup operations and SFUs initially.
A candidate's weighted count is a screening metric; actual trip counts and
shader invocation counts still need game timing/frequency attribution.

## What is keeping the 320 programs at the cap?

The fixed cohort is 302 fragment, 14 vertex and four compute programs. It is
unchanged after the spill changes. **291 use scratch; 29 do not.** A reported
96-register extent alone does not prove excessive simultaneous liveness or
spilling. In particular, the 29 scratch-free cases need a peak-live/placement
census before claiming that a specific encoding or occupancy threshold binds.

The current evidence rules out simply enlarging the constant-promotion budget
as the general solution: 307 exhaust eligible constants while leaving uniform
words free; ten are word-budget limited, two exhaust both eligibility and the
window, and one has no extracted preamble. None reaches the 8 KiB setup-code
policy bound. The spill work leaves preambles unchanged.

Remaining weighted traffic follows varying arithmetic, phi values and memory
results. Proven uniform-derived values account for only 34 depth-zero stores
and no FILLs. Texture-coordinate publication tuples are in their own register
file; they must not be charged against GPR pressure. Texture **results** still
have GPR constraints. Low-bank limits, tuple alignment/contiguity, late source
kills and reserved spill temporaries are possible additional causes, not
measured per-program explanations yet.

The next host census should record, for every completed allocation candidate:

1. Peak simultaneous half-register demand before and after scheduling/spilling;
   separate ordinary liveness, tuple padding and rematerialization reservation.
2. At each peak, count values restricted to r0–r63, their exact constraints and
   defining/last-use opcodes. Count publication values separately.
3. Attribute inserted spills to the triggering peak and original value/home,
   separating loop-carried, invariant and branch-local lifetimes.
4. Tie the selected candidate back to pipeline hash and stage, then rank by
   observed game shader time/frequency when available. Preserve multiplicity in
   the capture and also report unique shader families.

The 20 highest-weight records account for 808,820 events (43.8%); 50 account for
1,320,068 (71.5%). Many are variants of the same shader shape, so these are
capture records, not 20 independent algorithms or 43.8% of measured frame time.
The highest group has 42,499 events per record; use the CSV pipeline identities
when selecting repros. The complete sorted
[320-program list](../MEASUREMENTS.md#cs2-pressure) and [producer/kind totals](../MEASUREMENTS.md#cs2-pressure)
provide the reproducible shortlist.

## Direct frame-destination probe proposal

The public M3 decoder at commit
[`fc8ad8dc5f85`](https://github.com/TellowKrinkle/applegpu/blob/fc8ad8dc5f858e9ab17381b8b0bd7b1ceedbce5a/applegpu.py)
models `VariableDstDesc.k` as a stack destination: bit 37 in variable-length
FADD/FMUL and ten-byte logic forms, and bit 53 in eight-/ten-byte FMA forms.
The smallest four-byte FP forms do not expose this field. `FixedDstDesc`
uses a different layout; do not apply this hypothesis to every ALU opcode.
Our authored Metal evidence has shown direct frame IADD, not FP frame outputs.
These are hypotheses for T8132, not confirmed encodings.

Start with FP32 FADD, FMUL and FMA, plus IAND if it fits the same existing probe
harness. Use one known working GPR encoding as the control and one documented
M3 `k` frame form as the candidate for each. Decode/re-encode the full frame
operand: changing `k` alone is not sufficient for arbitrary frame indices.

- Give each invocation an explicitly sized frame. Start at frame word 1, clear
  neighboring words to distinct canaries, and preserve frame word 0 for the
  existing atomic reservation. Produce lane-varying exact representable results,
  FILL the target word through the already validated path, and compare every
  output and canary against a CPU oracle and the GPR control.
- Begin with a conservative known completion sequence. If successful, test
  source reuse, immediate dependent FILL, divergent active masks/reconvergence,
  and repeated loop writes. Check inactive-lane preservation and wait semantics.
- Only after low-word success, test the specific index boundaries needed by
  actual frames, with explicit frame allocation and canaries. Add saturate,
  abs/negate, high GPR inputs and uniform inputs individually. Half destinations
  require separate preservation tests and are outside the first batch.
- Stop that hypothesis on a wrong result or hang; retain full bytes, inputs,
  expected/actual words and hardware identity. No broad bit sweep is proposed.

For a production implementation, model frame as a distinct destination file in
IR and machine constraints, keep memory ordering/liveness explicit, allocate a
frame home rather than a GPR result, and select the form from normal spilling.
Allow folding only when other GPR uses, edge copies and wait semantics remain
correct. Do not introduce capture-specific destinations or fixed registers.
The current producer totals include aliases and later SAVEs, so they are an
upper bound on eligibility. Folding all SAVEs would eliminate at most 12.7% of
the weighted *instruction* events, while still executing those scratch writes.

## Authored Metal/HV kit proposal

The [36-case kit](../apple9-high-operands/README.md) is prepared, uncompiled and
unrun: each of twelve families has an 8-value control and 90/96-value pressure
variants. In a later coordinator-owned macOS/HV slot, compile only these authored
sources first. Retain source hashes, archive bytes, register extent, scratch and
target instruction bytes. Verify the target actually uses a high operand;
pressure source code alone does not prove that allocation occurred.

Texture sample/LOD/gradient results, compact FP forms and shared/private access
operands are the first candidates if the peak-pressure census implicates them.
Subgroup operations, SR reads and half conversion remain coverage work unless
the game census gives them a pressure role. Compare each high form with its low
control, propose one high-bit hypothesis, and then separately execute an
isolated T8132 test with valid resources and a lane-by-lane oracle. No changes
to production constraints are justified by Metal output alone.

## Validation gate

First validate `1f4c8abe909` with the two focused CTS lists, require actual
SAVE+FILL output from passing forced-spill cases, and rerun the value-checking
loop tests. Match the game driver's noncompiler runtime, native toolchain and
flags to frozen B before ABAB. Keep failures, skips and flakes separate. The
host value-checking/pressure subset passed all eight tests again during this
handoff; mini testing and the matched native build await the coordinator slot.
