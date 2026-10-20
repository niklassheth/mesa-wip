# T128 qualification with an explicit scalar source pair

Compiler base: `apple9/integration-next` at `81e87f46585`, scalar pair fix
`dc71a868ee0`, and the logical-T128 lowering rebased as `7115dedc3b6`.
The scalar submit now owns both adjacent 32-bit operands and receives an
explicit zero second value, including T128_CONTROL and T32_IDS. This is an
ordinary NIR/IR/allocation change. No register pin or post-allocation rewrite
is used. Both U0 operands retain this integration base's conservative clobber
contract; the earlier B-preservation candidate is not a dependency.

The research runtime overlay only selects physical mesh T32, admits its launch
and advertises the experimental logical-T128/subgroup class. Production defaults
remain T16. No preamble bypass, firmware/kernel/UAPI change or installed-Mesa
change is part of this kit. T128 and partial T127 remain unqualified on hardware.

Use SLOT.json: T32_CONTROL, T32_IDS, T64_LOGICAL, T128_CONTROL, T128_LOGICAL,
then T32_CONTROL; fresh paired vertex/native processes for each draw. Repeat
new T128 outcomes three times. After passing, test T64_LOGICAL_3D and
T127_PARTIAL. T32_IDS must now pass; its historical zero-geometry failures
were confounded by scalar base+1 and are not expected failures here.

Run `run-one.py --root KIT --arm NAME --stage vert|mesh --out NEW_DIRECTORY
--kernel-identity QUALIFIED.json`, then `analyze.py --root KIT --arm NAME
--vert VS_DIRECTORY --mesh MESH_DIRECTORY`. Require exact RGBA, depth and all
16 KiB of SSBO data, including guards. T128_LOGICAL checks all 128 records,
IDs, subgroup operations and cross-subgroup buffer reads after a workgroup
barrier. T127 exercises the final partial subgroup. T128_CONTROL alone does
not establish 128 logical invocations.

The runner requires the device owner's qualified physical-T32 kernel identity,
H1 qualification and native boot option. The identity example is a template.
Stop on any failure or incomplete submission and retain owned backing through
the usual capture/recovery protocol. Release/restructure work has first use
of the mini. No device commands are run during host preparation.

Reproduce by applying research-overlay.patch to the packaging commit recorded
in PROVENANCE.json, building host and ARM libraries, then running prepare-kit.py
with --root, --workspace, --host-build, --arm-build and --trace-base. The last
argument is a frozen recording-shim kit. run-host.py performs non-executing
Vulkan submission preparation; audit-host.py checks uploaded whole programs,
preambles, launch/state and both scalar inputs, including zero's def-use chain.
Original poisoned readbacks are required in these host-only runs. Revert only
the four runtime-overlay paths after freezing libraries and rebuild normal
libraries. Compiler sources remain committed.
