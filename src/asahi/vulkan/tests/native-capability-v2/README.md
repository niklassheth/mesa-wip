# Native capability pilot on harness v2

This is a research device kit for **our independently authored native programs**.
It changes no production compiler eligibility, Vulkan properties, or installed
Mesa. Source generation uses NIR, ordinary instruction selection and register
allocation. No Metal shader body, helper sequence or fixed-register program is
an input. The offline Metal study is in
`../../../compiler/research/native-capability-20261022/`.

The pilot has 47 program shapes and 175 runtime arms. It covers points, lines
and triangles through V256/P512; separate V-only, P-only and thread-width
changes; widths 16/32/64/128/256; and a V64/P1 control-flag-42 comparison.
For the 32 ordinary shapes it tests full, partial, zero-primitive, zero-vertex
and minimal nonempty runtime counts. Fifteen additional high-index controls
make only the last active vertex visible: earlier triangles/lines are
degenerate outside coverage, and earlier points lie outside the viewport.
Only the final primitive uses the last vertex, so the expected image counts
exactly one primitive. These cover V4/16/64/128/256 in every topology. Independent widths help distinguish a count/layout
failure from a workgroup-width failure. Joint maxima are separate experiments.

This first device tier uses Position and PointSize. Large varying interfaces,
per-primitive records and task payloads remain later device tiers. Their
metadata was covered in the offline study, but this pilot does **not** establish
their native behavior or a hardware maximum.

## Exact oracles

Every primitive contributes exactly 1 to an R32_SFLOAT image using ONE/ONE
additive blending. Integer sums through 512 are represented exactly. Triangles
cover all 4096 pixels, lines cover row 32, and points cover pixel (32,32).
Every other pixel must be exactly zero. Before native work, 21 ordinary vertex
draws independently qualify these masks and multiplicities at P0/1/2/16/128/256/
512. A calibration failure stops the campaign; expectations are never fitted
to native GPU observations.

The native program also copies its fully initialized local output into an owned
buffer, including all padding/count/index bytes. The independent Python oracle
checks all 512 KiB with a full-byte mask: cookie, per-lane visit records, local
snapshot, and untouched poison guards. The last primitive references the last
active vertex, including index 255 at V256. Authored count snapshots are input
witnesses, **not hardware primitive counters**. Raster accumulation is the
independent count observable. Q or raster mismatches stop before any larger
shape. Completed failures stay separate from timeouts, kernel faults and
unattempted cases.

## Carrier and launch contract

The private adapter ICD and host-only DRM shim come from the sealed r8-v2 kit,
with driver identity `git-e08fa6cf73`. The staging adapter expects its old
V3/P1, local96, ABI7 sidecar labels. These are explicitly staging metadata:
actual compiler metadata (ABI8), declared shape and allocation are recorded
separately. The ABI8 uniforms used by these probes retain that adapter's roots:
output root 6, grid root 7, Q pointer at uniform words 16/17, draw-input pointer
at words 18/19. No task payload is claimed by using the Q uniform.

At registration, before any submission, the carrier verifies its original
launch bytes, complete uploaded mesh/object bodies and argument table. It then
updates only owned DATA fields: vertex planes, primitive capacity, workgroup
width, rows, topology, local allocation and the corresponding Apple9 PPP
PointSize/output-count/object-type fields. The allocation is rounded to a power
of two, at least 128 bytes; the largest pilot requests 8192 bytes. This is a
hypothesis to measure, not evidence that the broad allocation works.

It freezes and checks all resulting program, launch, state, argument and nine
bounded PPP records for every submit. Apple9 PPP pointers use the existing
context base 0x1000000000; this is not the older G13 packet layout. All complete
native code remains immutable. Repeated runtime variants share a pipeline but
own separate recorded launch/state storage. This startup-only research adapter
must never become a production lowering or replay path.

The v2 lifecycle uses one persistent native owner/device for the pinned research mode, one command
in flight, bounded kernel-log monitoring, fresh cookies, exact renderer/library
identity and retained backing on an uncertain completion. The ordinary-render
calibration owner exits before native mode is enabled. Mode 0 is restored only
after all owners have completed safely. No full CTS is part of this pilot.

## Reproduction and scheduling

Build `apple9-native-capability-probe` in a test-enabled host Mesa build, then:

```sh
python3 prepare.py --base PATH/TO/SEALED/R8-V2/kit \
  --compiler PATH/TO/BUILD/src/asahi/compiler/apple9-native-capability-probe \
  --out PATH/TO/NEW/kit
python3 dry_run.py --root PATH/TO/NEW/kit --out PATH/TO/NEW/dry
python3 host_checks.py --root PATH/TO/NEW/kit --dry PATH/TO/NEW/dry
```

`prepare.py` verifies the base manifest and compiles all program bodies and both
AArch64 owners. `dry_run.py` submits only to the host DRM shim; every result must
remain a Q failure because no GPU work executes. Host success proves upload and
identity plumbing only. See the frozen kit's host-check JSON and hashes.

The frozen host checks are in `HOST-CHECKS.json`, `HOST-CALIBRATION.json` and
`HOST-OLD-PARSER-CONTROL.json`. All 175 native arms passed the identity audit;
C oracle materialization matches Python, and an independent geometry check
confirms the expected pixel multiplicities. All masks cover every owned byte.
The host shim executes no GPU work: zero device trials are qualified. The
ordinary control's host pixel differences are expected poison, not GPU failures.

The old revision-2 parser pinned by C1 admits only the narrow class. The kernel
owner is preparing a separate research admission value (expected `native_mesh=3`)
with the same UAPI and ownership lifecycle. This frozen kit takes a separate
`--kernel-requirements` JSON so its program/oracle manifest need not change when
the kernel artifact arrives. Start from `KERNEL-REQUIREMENTS.template.json` and
supply exact commit/image/DTB/release/version pins, the actual research value,
`ready:true`, `all_fixtures_accepted:true` and the frozen fixture-index hash.
The runner requires a research value of at least 3; it does not try legacy
modes 1/2 for these broad packets.

The kernel owner must check all 47 exact frozen streams, including T64/T128/T256:

```sh
python3 check_parser.py --kernel KERNEL_TREE --commit FULL_COMMIT \
  --mode NEW_RUST_MODE_VARIANT --out PARSER_RESULTS
```

This checks mode 0 performs no reads, and the supplied research mode admits
every fixture. `PARSER_RESULTS/ADMISSION.json` records its source and fixture
hashes. The C1 `check_admission.py` deliberately tests the old pinned parser's
rejections; use this kit checker for the new acceptance result. Do not
transplant old image hashes or interpret parser admission as GPU success.
The manager schedules codex-hw only after kernel and packet review, on the same
revision-2 safe-boot contract used by harness v2 (read-only root, swap/zswap off,
16 KiB pages, no other native clients). No UAPI change is needed.

Before transfer, verify the frozen kit with `sha256sum -c SHA256SUMS` from its
root. Preserve all files and keep results outside that directory. The kernel
requirements and verified boot identity are separate manager-supplied files.
No edit or reseal of the kit is required when those arrive.

Once the kernel owner supplies the separate pinned requirements file, the scheduled command is:

```sh
python3 run.py --out RESULTS --kernel-identity KERNEL-IDENTITY.json \
  --kernel-requirements CAPABILITY-KERNEL.json \
  --lease-deadline-epoch MANAGER_DEADLINE --trials 2
```

The default cap is 420 seconds inside a 900-second lease, retaining collection/
reset time. Any stop names the owner PID and preserves GPU-reachable storage;
collect before the manager-controlled reset. Never infer a pass for unattempted
cases. Only measured native results can support production generalization.
