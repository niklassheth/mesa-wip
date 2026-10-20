# Native mesh limits research arms

This directory is a host-prepared experiment based on commit
`98377944de480f9885e22eaefc6d7666e000e592`. `research-overlay.patch` is intentionally
kept out of production sources. It changes mesh physical width to 32, uses
Mesa's common `nir_lower_workgroup_size(..., 32)` for logical T64/T128, admits
those invocation counts, and enables mesh-stage subgroup operations. Object
entry remains physical T16. Other native V3/P1 limits remain in place.

The overlay changes five files: native eligibility/body lowering, native
completion integration, Vulkan shader validation, Vulkan draw state, and
properties. Apply it only in a clean research checkout. Build x86 and ARM
`src/asahi/vulkan/libvulkan_asahi.so`, then run `prepare-kit.py` with those build
directories and a known recording-shim kit as `--trace-base`. Remove the overlay
and rebuild ordinary libraries after freezing the research kit. No source or
binary shader substitution is used.

`run-host.py` uses only the non-executing recording DRM shim (ARM through QEMU).
`audit-host.py` checks the actual recorded state, full uploaded programs,
launches, compiler-owned object path and U publication against compiler output.
Mesh state must be T32 while object dispatch remains T16. Failed preparation
or a successful host submission is never a device pass.

Device order: T32_CONTROL, T32_IDS, T64_LOGICAL, T64_LOGICAL_3D. Each gets its
own ordinary VS control and fresh process/VM. Stop on a failed/incomplete arm.
T128_LOGICAL_HOST is compiled and audited but refused by the device runner.
Previous physical-T64 M1_CONST timed out and remains a separately coordinated,
last-and-risky diagnostic; no automatic runner here submits physical T64.

A kernel must admit this exact T32 mesh/T16 object state with all ownership and
allocation checks intact. `kernel-identity.example.json` is a template, not
qualification. Native mesh boot enablement, exact kernel identity and existing
H1 qualification are checked by `run-one.py`. No kernel/UAPI change, reset,
installation or device execution is performed by these preparation tools.
Timeouts retain stopped-process backing for the device owner's recovery.

Each accepted mesh arm must match exact RGBA8/D32 output and all 16 KiB of its
SSBO oracle. Records cover atomic visits, logical IDs and size, subgroup IDs,
ballot, reduction, broadcast and cross-logical-subgroup reads after a buffer
memory barrier and workgroup barrier. This does not test GLSL `shared`, which
is still rejected. Runners explicitly enable the existing
`hk_enable_vertex_pipeline_stores_atomics=true` option for record writes.

The separate `apple9-limits-workgroup` host test executes the common NIR pass
on a bounded 32-lane semantic model for T32/T64/T128 and mixed XYZ shapes. It
checks subgroup results and cross-barrier records against arithmetic oracles;
it does not model GPU memory ordering, launch hardware, nested loops or all
possible workgroup control flow.
