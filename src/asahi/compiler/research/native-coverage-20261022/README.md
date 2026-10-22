# Native coverage audit, 2026-10-22

Companion evidence for [native-coverage-plan](../../README.apple9-native-mesh.md#native-coverage-plan).
This is host-only analysis of the original native inventory, not a CTS device
run or an implementation of fallback.

`CASE-AUDIT.tsv` assigns every one of the original 625 failed cases to a
disjoint first-rejection bucket, retaining its API error class, exact code
location at Mesa `bcb0837f948a`, shader shape, payload declaration, conditional
rendering overlap and diagnostic. `SUMMARY.json` includes the full inventory
status counts, API/bucket cross-tabulation, exact first skip reasons and raw
input hashes. `classify.py` asserts complete/nonduplicated case coverage and
the source matches before regenerating them. `PROVENANCE.json` additionally
pins the diagnostic proxy, replay script, actual ICD, host CTS and shim,
invocation, and inspected source snapshots.

Raw host evidence is retained in
`tmp/codex-upstream/native-mesh/native-coverage-plan-20261022/` under the workspace.
The `all-rejections` directory contains `stdout.log`, `results.qpa` and
`RUN.json`. Its ARM/QEMU run loads the installed unmodified ICD through
`debug_icd.c`/`.so`, which only adds `VK_EXT_debug_utils` and a debug messenger
and forwards API calls. `replay.py` records its complete invocation. The
no-op DRM shim does not execute GPU work. Only the 545 existing FNP cases were
replayed; neither the 80 property failures nor the two OOM cases were executed.

The original device CTS binary has SHA256
`301f38012b63c7ae7700c76c5e391339f05518deb537add1f119ddc2b874464b`.
The locally available host-replay CTS has SHA256
`83acc28c211c515668ddaf108cbd5a38fdaaa1c46a3dd141ee1ed0e6683310aa` and reports
an unknown release. It must not be called the same binary. Nevertheless all
109 distinct logged GLSL sources and both assembly inputs occur exactly at the
end of an original device cache source key (the prefix contains compiler/build
options). The four original caches contain 272 records. This verifies input
text across these test versions, not identical generated SPIR-V, options or GPU
execution. All 545 host case statuses reproduce Fail. Diagnosis combines those
callbacks with pinned owned driver and public CTS source inspection.

Regenerate from the owned Mesa checkout (substitute the workspace location
in the absolute evidence paths if needed):

```sh
python src/asahi/compiler/research/native-coverage-20261022/classify.py \
  --inventory /home/nsheth/Projects/asahi/tmp/codex-hw/mesh-cts-full-native/CASE-RESULTS.json \
  --replay /home/nsheth/Projects/asahi/tmp/codex-upstream/native-mesh/native-coverage-plan-20261022/all-rejections \
  --cache-dir /home/nsheth/Projects/asahi/tmp/codex-hw/mesh-cts-full-native/device/results/native/main \
  --output src/asahi/compiler/research/native-coverage-20261022
```

The raw callbacks include 45 subsequent command-buffer errors after a compiler
failure. Only the first diagnostic is classified. In the combined draw gate,
indirect/count cases take precedence over the overlapping conditional predicate;
count-buffer cases still short-circuit first at `!direct`. In the combined
shader gate, point/line topology takes precedence over their size mismatches.
These are accounting conventions for overlapping conditions, not independent
root-cause experiments. No category is assigned to an unstarted case.
