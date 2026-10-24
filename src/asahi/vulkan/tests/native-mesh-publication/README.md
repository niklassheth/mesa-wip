# Native publication: shared storage and single-round extra outputs

These fixtures preserve the nine passing Air cases from claude-meshwide's
`a9b0b86fd50` (shared) and `6e356213d1a` (single). Only those case sets are
retained. The shader sources, selected case records and CPU image/witness
calculations are unchanged; earlier selector/offset/width experiments are
outside this fixture. The probe reuses `native-mesh-m2/probe.c` with compile-time
limits V256/P256; existing users retain their original V64/P126 limits.

Shared storage was already supported by the T32 publication model. Air
39b7c9g passed four exact cases using driver `fea439f9520`: V64/P124 with
4-byte shared and two clip planes; V64/P126 with 1 KiB shared; V128/P256 with
16-byte shared; and L64 serialized at T32 with a barrier and 4-byte shared.
The witness checks every shared word through every logical invocation and
checks the full 512 KiB output/guard buffer.

Single-round extra-output admission is `74128258089`. Air 39b7c9m passed
five exact cases with that driver: V33/P42 control; V32/P42 with two clip
planes; L16 V16/P16 with one clip plane; L1 V8/P6 with two clip planes; and
V20/P10 with 4-byte shared plus one clip plane. Each launches at T32.
Position-only single-round classes retain their existing policy, and V3/P1
retains its primitive-broadcast path. Shipping defaults remain off.

Both kits used the Air `research-air-native-batch` kernel, source
`4b1e8012b687`. Every case retired with exact pixels/witness, matching
native-route and own-VM audits, and one native render/dispatch counter delta;
no faults or retries. These are component proofs, not unmodified application
corpus executions. Their respective sealed kit SHA256SUMS hashes are recorded
in `HARDWARE.json` along with the original source/driver and collected outputs.

Build this directory as a standalone Meson project, then run:

```
python3 host-check.py --probe BUILD/mesh-publication \
    --lavapipe /path/to/libvulkan_lvp.so --out NEW_OUTPUT
```

This compiles all nine cases, verifies software readbacks, and checks 90
corrupt/truncated-output mutations. `check.py --self-test` runs only those
oracle mutations. Software results are not native hardware evidence.
