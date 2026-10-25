# Three unchanged corpus shader pairs

These draws use the Khronos Vulkan-Samples mesh_shading GLSL, HLSL and Slang
precompiled mesh modules and their corresponding original fragment modules.
All six SPIR-V files are copied byte-for-byte from the pinned corpus checkout;
mesh hashes also match the unchanged 18-entry coverage ledger. They have no
API task stage, descriptors, user varyings or specialization overrides.
The Vulkan harness supplies a taskless native object stage through the normal
driver. No shader-generated witness, substitute fragment, patched entry point
or source translation is used. This is execution of corpus shader pairs, not
a run of the complete Vulkan-Samples application.

The fixture uses the current merged driver with HK_APPLE9_NATIVE_INDEX=1,
qualified mode 4 and no research/shape overrides. Each process must retire one
native T16/V3/P1 draw, report its exact renderer and mapped driver, pass its own
FD's VM profile/generation/class audit, and advance native render and object
counters by one. Missing or zero native routes stop the kit immediately.

All 32,768 floating-point color components in R32G32B32A32_SFLOAT are checked
byte for byte, including alpha and the background. Additive blending also
makes duplicated primitives visible. The original fragments output red with
alpha one; no channels are discarded. The full unreferenced 512 KiB buffer
must remain 0xcc. The independent edge oracle has no pixel-center ties.
Eleven mutations cover missing/duplicate fragments, each color channel,
truncation, unexecuted output and guard corruption.

The prior three-pair device fixture checked only R32 red; its component credit
stays separate. This kit closes that color-interface gap using unchanged
original SPIR-V. It is not a complete Vulkan-Samples application run.

Host validation runs all three original pairs on software and records all
three through the ARM native compiler. The paired kernel parser accepts
research and qualified profiles, checks relocations and rejects every prefix
truncation. Full rehearsal runs all three arms and injects broken GPU init.
Native VM/renderer/counter audits cannot be modelled by the software driver
and receive no rehearsal credit; actual device execution requires them.

Budget: three draws, 100 seconds, zero retries, stop on first failure. The
canonical operator owns device state, staging, collection and recovery.

Host packet recording uses the historical no-op DRM shim and an explicitly enabled synchronous fence model. It does not execute a GPU and cannot model the kernel qualification advertisement, so that recording uses the research override. The Air harness does not use either host helper or that override: every draw requires INDEX=1, a qualified own-FD VM, a native route, retired submission and exact readback.
