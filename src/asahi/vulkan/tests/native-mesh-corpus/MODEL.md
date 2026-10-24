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

The existing Air-passed M2a driver f987032b1d0 and kernel 4b1e8012b687 are used.
All three draws select qualified mode 4 without native/shape overrides, one
process/draw/submission each. Each has a retired fence, exact mapped-driver
revision, own-FD VM profile/generation/class hash, native T16/V3/P1 route and
native-render/object-dispatch deltas 1/1. The check covers the complete 128x64
R32_FLOAT target with additive blending and an untouched 512-KiB buffer.
The original fragment shaders output constant red. This target checks their
red component; it does not claim green/blue/alpha attachment coverage.

The independent pixel oracle derives the triangle from public source positions:
(96,16), (96,48), (32,48) in this viewport. It checks all 8192 pixels, including
background and multiplicity. The SSBO is not referenced by the original
modules and must remain 0xcc throughout. Seven checker mutations cover empty,
unexecuted, missing/duplicate, wrong fragment value, truncated readback and
guard corruption. All three software controls pass. Three real ARM-driver
host recordings compile/link the original pairs and emit native packets;
these host results are not GPU execution. Paired kernel-parser replay checks
both research and qualified profiles, relocations and every truncated prefix.

The budget is three draws, 100 seconds, zero retries, stop on first failure.
The manager owns the device run and recovery. A successful native draw earns
one exact corpus entry, potentially moving native execution from 0 to 3/18.
Failure or unattempted entries retain their separate counts. Wide M3 publication
and per-primitive interfaces are separate research; this kit does not exercise
the new subgroup-control prototype.
