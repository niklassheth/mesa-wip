# M2a logical-workgroup Air kit

Eight API draws / eight submissions / eight processes, 150 seconds, zero
retries, stop at first failure. Corners 1/17/32/64/128, random interior 115,
2-D 8x8, and one shared-memory case. See MODEL.md for scope and limitations.

Seven processes test candidate default selection in qualified mode 4 without
HK_APPLE9_NATIVE_MESH or a shape override; the shared-memory process uses
research mode 3. All remain position-only V3/P1 on physical T16/T32. Require
the pinned driver mapping and own-FD VM profile/generation, expected native
route, retired fence, complete readback, and native counter deltas 1/1.
The kit restores mode 0 after clean retirement. Uncertain in-flight work is
stopped with backing retained for operator recovery.

The manager owns queueing, hardware and recovery through tools/kit/run-kit.py.
The fresh-boot kernel is research-air-native-batch at 4b1e8012b687 with
native_batch=16/payload=0. This candidate goes to claude-compiler for review
before merge; the sealed driver is not installed. Rehearsal retains all
native audits and cannot qualify native execution.

Host evidence comprises eight software controls, authored checker mutations,
eight normal-driver non-executing native packet recordings, eight paired
kernel-parser replays, and focused driver/runner tests. The fixed corpus
improves from 0/18 to 3/18 declaration candidates; measured native execution
coverage remains 0/18. This is an initial range sample, not complete M2/M3
qualification, physical T64/T128 support, or real application capture.
