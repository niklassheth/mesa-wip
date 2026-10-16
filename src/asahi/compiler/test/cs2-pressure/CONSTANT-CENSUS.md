# Main-program constant census

The requested starting point is frozen `7e6bc8da2be`, including its 64-promotion
cap. The final comparison is `ff53faea9e7`, after the cap removal, validated
uniform families, loop weighting and immutable-import folding. Both cover the
same 10,161 programs. Main metrics from the starting diagnostic run exactly
match the uninstrumented `7e6bc8da2be` replay. The final run is the uniform
measurement described in [UNIFORM-SOURCES.md](UNIFORM-SOURCES.md).

Counts of selected immediate definitions are filtered by actual common-IR
liveness after shared DCE. Consumer entries are filtered by retained common
instructions. Final packed IMM counts separately include literals introduced
by allocation/lowering. Thus unneeded pre-DCE definitions are not reported as
remaining main materialization.

| Stage | Live selected IMM definitions, before → after | Final packed IMMs, before → after | Single static loop-use definitions, before → after |
|---|---:|---:|---:|
| Fragment | 203,690 → 151,194 | 213,798 → 156,832 | 25,249 → 9,048 |
| Vertex | 78,991 → 64,286 | 79,043 → 64,322 | 1,730 → 686 |
| Compute | 159,864 → 123,766 | 160,374 → 124,083 | 28,132 → 17,633 |

The promotion reason is attached to the literal's bit pattern at NIR promotion
time, then associated with each remaining selected IMM definition. It is not a
proof that every later use of the same bit pattern was promotable. In particular,
“value promoted elsewhere” includes unsupported uses and constants recreated by
lowering. A phi source is an overlapping consumer category, not a mutually
exclusive reason.

| Reason | Fragment before → after | Vertex before → after | Compute before → after |
|---|---:|---:|---:|
| No eligible NIR promotion use | 154,974 → 39,028 | 51,901 → 10,801 | 98,821 → 14,625 |
| Single eligible use | 22,511 → 28,211 | 10,134 → 10,551 | 19,419 → 13,659 |
| Value promoted elsewhere; other materialization remains | 18,717 → 83,955 | 16,100 → 42,394 | 41,624 → 95,482 |
| Old 64 cap | 7,488 → 0 | 856 → 0 | 0 → 0 |
| Uniform word budget | 0 → 0 | 0 → 540 | 0 → 0 |

The final eligible-use score includes the new loop-depth weight, so a lone
eligible use inside a loop no longer falls into “single eligible use.” The
remaining one-use loop definitions can include unsupported consumers, phi
inputs, and constants introduced after promotion. Definitions feeding phis are
9,986 / 6,644 / 13,253 in the final fragment/vertex/compute populations; 6,960 /
6,092 / 9,303 feed only phis. An additional 470 fragment and 189 compute live
common values have no retained original VIR consumer, so this census does not
assign them a consumer slot.

[constant-consumer-census.csv](../MEASUREMENTS.md#cs2-pressure) records every
observed consumer opcode, encoding, physical GPR-source-list index, logical
machine operand role, uniform-file admission and maximum uniform index, both
before and after. Multiple slots can refer to one definition; their definition
counts must not be summed. `FMA_LOW_SOURCE` remains bounded at u63 where that
short encoding requires it. [constant-census-results.json](../MEASUREMENTS.md#cs2-pressure)
retains the aggregate counts.

After the new families, prominent GPR-only consumers are vertex `LOGIC_EXPORT`
source 1/source 0 (96,900 / 73,872 uses), fragment device-load indices (46,781
indirect and 21,900 ordinary), and vertex `FLOAT2_EXPORT` source 1/source 0
(20,180 / 16,625). `IOR_UNIFORM` also still consumes a zero GPR to import words
for unsupported uses. These figures identify future encoding/model work; they
do not authorize treating those slots as uniform-capable. Publication *sources*
can be GPRs while publication destinations remain in their separate file, as
verified in [PUBLICATION-FILE-20261005.md](PUBLICATION-FILE-20261005.md).
