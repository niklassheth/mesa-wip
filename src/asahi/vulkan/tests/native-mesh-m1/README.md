# M1 initial T16V16P15..17 discriminator

One Air research kit: **six API draws, six submissions, six processes,
120-second run budget, zero retries**. Each process contains one mesh draw.
The operator owns boot/staging/hash checks/collection/recovery. Stop on the
first failure; keep all raw partial outputs. Uncertain in-flight work is
stopped with backing retained for operator recovery.

Three declared maxima P15/P16/P17 each run full V16/Pmax and partial
V15/(Pmax-1). All output geometry is exercised, including highest index 15.
An independent 8,192-pixel additive raster oracle checks multiplicity, and
every word of the 512-KiB output witness/guards is checked. See `MODEL.md` for
the common-base argument, resource boundaries and remaining qualification.

The pinned ARM driver runs the ordinary Vulkan lowering with explicit
research class selection. Kernel mode 3 must latch on each process's own DRM
FD/VM. Require the pinned mapped driver, one native route, one retired fence,
and native render/dispatch counter deltas 1/1. Mode starts at 0 and returns
to 0 after clean retirement. No new default class or UAPI change is made.

The pinned research kernel boots fresh with native_batch=16/payload=0;
Mesa uses batch=1. Setup checks dependencies and architecture before boot.
Both offline rehearsal and dry-run receipts must pass before the manager
queues it. Rehearsal reduces work to one case and retains every native audit:
a stand-in pixel PASS followed by missing native VM evidence is an expected
checker failure, never native qualification. Broken initialization must exit
promptly with a non-PASS verdict.

Run only through the operator (replace output with a fresh manager-chosen
path):

```
tools/kit/run-kit.py run <kit-directory> --device air --out <results-directory>
```

There are no timing queries or device fallback-control draws. Six host
software controls, checker mutation tests and six real-driver non-executing
packet captures precede this kit. The paired kernel parser accepts all six
in research mode and rejects P15/P16 in qualified mode; P17 retains its
existing qualified resource key. Host evidence is in `evidence/` and its
original locations/hashes are in `PROVENANCE.json`.

Report all attempted, retired, checked and native-passing cases separately.
One six-draw pass is initial boundary evidence, not range qualification or
mini evidence. Zero output, other packing/allocation transitions and repeats
remain for later bounded gates. This sample alone adds no native shaders to
the M0 demo/engine corpus.
