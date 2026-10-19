# Native workgroup qualification fixtures

Host-only generators and owned readback checks. There is no device runner,
loader or driver change in this directory. The hardware owner connects these
whole compiler programs to the existing qualified native research carrier.

Build `apple9-native-workgroup` with tests enabled. It checks all 1..16 API
sizes and six 3D shapes in both stages, then `--export DIR` emits complete
T1/T3/T16 mesh/object programs, compiler metadata and source-derived oracles.
`PLAIN` has no extra application effects. `PROBE` records API identity and
subgroup state, atomically counts invocations, and exercises a barrier.

```sh
build/src/asahi/compiler/apple9-native-workgroup --export artifacts/fixtures
python device-arms.py --fixtures artifacts/fixtures --out artifacts/ARMS.json
cc -O2 -I .. semantic-data.c -lm -o artifacts/semantic-host
```

Use the AArch64 compiler export for the device. Keep its recorded bytes and
allocated resources; do not splice its HALT/RET or substitute fixed registers.
Install the complete programs with owned entry/frame backing. All arms use a
physical **16x1x1 object and mesh launch**, even when the API shader is T1 or T3.
Audit both actual launch records, their entry addresses and the entire loaded
program hashes. The mesh state thread word is `0x400810`. Keep the known
object/mesh frame contracts and backing; zero compiler spills do not qualify
zero launch frames. Large mesh programs still need the established resource
contract. This package does not qualify a reservation provider.

`ARMS.json` specifies 14 ordered arms: opening T16 plain control; mesh-only T1
and T3 plain/probe and T16 probe; object-only T1 and T3 plain/probe and T16 probe;
both T1 and both T3 probes; closing T16 plain control. Draw exactly one object
and one mesh group, with grid `(1,1,1)`, in a fresh process/VM per arm. T32 is a
host rejection control, with no GPU arm. Repeat a failed arm only after the
hardware owner has reviewed its evidence and restored the target as needed.

Each stage has its own owned 2048-byte Q buffer. Bind that stage's Q pointer
in argument words 16/17 (otherwise unused by these fixtures). Initialize all
bytes to `0xa5`, then zero its first uint32. The probe atomically increments
that counter once per API invocation. At `Q + 256 + 64*invocation`, words 0..12
are: local index, local X/Y/Z, declared size X/Y/Z, subgroup size 32, subgroup
invocation, subgroup ID 0, NumSubgroups 1, active ballot, and index XOR
`0x12345678`. The last word is written after the API barrier. Words 13..15,
all inactive-lane records and all other bytes stay poisoned. Expected active
ballots are `0x1`, `0x7`, `0xffff`; expected counters are 1, 3, 16. Plain arms
leave Q unchanged. These Q buffers must not alias the carrier's observers.

Bind the ordinary 184-byte mesh draw descriptor in words 18/19, using the
actual draw's owned reservations, fragment state and allocation IDs. Bind an
owned zeroed class-drop counter in its slot 22. Bind the object grid root
separately; these constant-grid object fixtures do not need the draw pointer.
Only API lane 0 writes X/Y/Z; compiler completion reconverges to physical T16
and exports X/Y/Z using lanes 0/1/2. A T1 callee must **not** be restricted to
physical lane 0.

Raster setup is the authored H1 source: 64x64, one sample, no blending, red
fragment, transparent black clear, positions `(-.5,-.5,.5,1)`,
`(.5,-.5,.5,1)`, `(0,.5,.5,1)`, indices 0/1/2, viewport scale `(32,-32,1)`
and translate `(32,32,0)`. Every arm requires exactly the exported 16,384-byte
RGBA image: 512 red pixels and 3584 clear pixels with identical coordinates.

Record buffers start at zero. Read back the actual Work-selected geometry,
list, root and scene and independently derive `source.bin` from the input
positions/indices, actual PPP state, viewport and reservation IDs. Its 32
little-endian u32 words are positions[12], indices[3], state[8], viewport[6],
IDs[3]. The state fields are the existing `formation_fixture` contract; see
`../apple9-mesh-data.h`. The exported reference-source and expected-data files
use synthetic host addresses/IDs/state and are self-test inputs, not a
replacement for the actual device draw inputs. Exact DATA means all decoded
coordinates, state, references and zero tails match those independently
recorded inputs. It does not require one compressed packet encoding.

Save readback files with these names, then check them on the host or with the
AArch64 semantic decoder on the target:

```sh
python verify-device.py --manifest artifacts/ARMS.json --arm MESH_T1_PROBE \
  --fixtures artifacts/fixtures --readback results/MESH_T1_PROBE \
  --decoder artifacts/semantic-host
```

The readback directory must contain `source.bin`, `geometry.bin`, `list.bin`,
`root.bin`, `scene.bin`, `pixels.bin`, `object-grid.bin` (three u32 words),
`object-q.bin`, `mesh-q.bin`, and `class-drops.bin` (one u32). The decoder checks
exact source geometry, every reference/state field, complete zero tails and
pixel coverage; the Python wrapper checks all Q bytes and grid/class counts.
`test-oracle.py --artifacts DIR` runs 14 source-derived positives and eight
corruptions (inactive object/mesh lane, subgroup width, atomic count, pixel,
DATA header, grid dimension and class drop). It performs no device access.

Use the hardware owner's existing journal/retirement monitor with a 20-second
per-arm deadline and per-process `MemoryMax=2G`, `MemorySwapMax=0`. Any GPU
fault, timeout, EIO/ETIMEDOUT, command preparation failure or mismatched
readback stops the matrix. On a live GPU error/deadline, SIGSTOP the entire
process group/unit and keep its VM/BOs pinned until controller reset. A fence
success alone is insufficient: independently match each actual TA/3D Work,
scene, program and completion stamps. The readback checker deliberately
reports `retirement_checked=false`; passing it is only one part of the device
qualification. Keep unexpected failures, unsupported arms and unrun arms
separate.
