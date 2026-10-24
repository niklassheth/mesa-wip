# M1: first narrow publication sample

This is a research discriminator for T16V16P15..17 through ordinary Vulkan.
It changes no qualified class or production admission policy. The six draws
are three declared primitive maxima times full/partial runtime counts:
`(V,P)=(16,Pmax)` and `(15,Pmax-1)`. All referenced vertices are written, and
all written vertices are referenced; the highest indices are 15 and 14.

The compiler's existing candidate has three separable responsibilities.
The application writes byte-packed triangle indices and component planes,
then a workgroup barrier makes them available to publication. All physical
lanes participate in allocation/control with the normalized runtime counts.
Position publication uses a prefix of `v` lanes and one common base per
component plane. Index publication uses the prefix
`max(1,ceil(3p/4))`, with the same base argument, zero, in every participating
lane. The compiler emits its own control, fence and publication sequence.

The implied array-collective contract is a hypothesis constrained by our
results, not a recovered implementation. S2's common-base arm was reported
as 48/48 passing, including four direct T16V16P16 trials. S6/S7 supplied
varying per-lane bases and failed. Those variants violate this candidate's
uniform-base invariant; the results are consistent with indexing being
applied twice. They do not determine the precise address arithmetic, whether
the implicit ordinal is physical lane ID or active-lane rank, or whether
arbitrary non-prefix masks work. All our proposed masks are contiguous
prefixes, so these tests cannot distinguish those ordinal models. We do not
repeat the known failing address variants or claim that S2 failed.

The historical v3 replay currently lacks its `NARROW-COMPLETE.json` receipt.
Individual retained trial records reproduce, but the full original seal
cannot be revalidated. M0 records this gap. The six new ordinary Vulkan
draws have their own pinned driver, packet evidence, independent oracle and
retirement evidence, rather than inheriting qualification from that receipt.

For position-only triangles, maxima determine the following byte layout;
runtime counts never change the allocation or packet resource state:

| Declared T/V/P | Index bytes | Count offset | Position offset | Plane pitch | Used local bytes | Rounded launch backing | State words 10 / 11 / 13 |
|---|---:|---:|---:|---:|---:|---:|---|
| 16/16/15 | 45 | 48 | 64 | 64 | 320 | 512 | `0x0000e / 0x400810 / 17` |
| 16/16/16 | 48 | 48 | 64 | 64 | 320 | 512 | `0x0000f / 0x400810 / 17` |
| 16/16/17 | 51 | 64 | 80 | 64 | 336 | 512 | `0x10010 / 0x400810 / 17` |

State word numbers include the leading packet header. The count region is
16 bytes at `align(3P,16)`; four Position planes occupy `4*4*align(V,4)` bytes.
Every application index lies below the count region, each four-byte vertex
component lies within its own plane, and all planes end within the 512-byte
launch backing. The last packed index word stays within the index padding.
For the tested runtime primitive counts 14,15,16,17, the publication prefix
is respectively 11,12,12,13, within T16. Runtime v15/v16 also fits T16.
The P16→17 change crosses both a count/header alignment and primitive state
boundary, while holding vertex pitch and launch allocation fixed. P15→16
changes the packed tail without changing that allocation. No task payload,
API shared storage, user varying, primitive attribute, culling or topology
change is included.

Each process uses one render pass, one draw, one command-buffer submission,
and one retirement fence. Image copy uses the driver's compute path. No
timestamps split the render. The real ARM driver has compiled all six
programs through normal lowering; non-executing DRM records contain one
native render and the stated resource header per process. The exact paired
kernel parser replays those streams at two relocated bases and rejects
every truncated prefix. Mode 3 admits all six; mode 4 rejects P15/P16 and
admits the already-listed P17. This proves packet admission, not execution.

The raster oracle computes triangle coverage from independently specified
integer screen coordinates. R32 float ONE/ONE addition counts primitive
multiplicity over five or six disjoint spatial regions, including a triangle
using vertex 15. Every one of 8,192 pixels is checked. The 512-KiB witness
buffer checks actual count/vertex/index stores, all 16 body lanes and every
untouched word/guard. Witness counts are application stores, not hardware
primitive counters; the image supplies independent multiplicity evidence.
Repeated triangles share geometry, so their ordering is not observable.
The checker detects 336 authored loss/duplicate/rebound/high-index and
buffer/readback corruptions. Six software-reference draws pass, but cannot
establish native publication semantics.

The device runner keeps own-FD VM generation/profile and mapped-library
checks, exact native route diagnostics and per-case kernel admission deltas.
It checks those deltas only after the fence and full readback. Rehearsal
reduces the workload to one case and retains all native audits. The stand-in
ICD may pass pixels while failing a native audit; that is not native PASS.
On uncertain retirement the process and backing are retained for operator
recovery; there are no automatic retries. After clean retirement it restores
mode 0, including after a completed mismatch.

Passing these six draws would remove the unexplained P16 exception for this
ordinary Vulkan fixture and justify subsequent boundary qualification. It
would not promote the interval, prove all programs sharing these maxima, or
prove all `V<=T, ceil(3P/4)<=T` combinations. Missing gates include zero output,
the other packed tails, vertex-pitch and launch-allocation transitions,
P21/P22, repeatability, and wider/multigroup/API semantics. The M0 sampled
demo/engine native coverage remains 0/18 even if this narrow sample passes.

Source model: `agx_nir_lower_apple9_mesh.c` (layout),
`agx_nir_apple9_mesh_native.c` (publication), `agx_apple9_mesh.c` (state),
`hk_apple9_mesh.c` (upload admission), and the pinned kernel `g16_mesh.rs`.
