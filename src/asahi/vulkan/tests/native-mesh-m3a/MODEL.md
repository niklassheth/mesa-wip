# M3a: original color interfaces and typed P1 values

Three unchanged SaschaWillems Vulkan meshshader mesh/fragment SPIR-V pairs
consume their real three-matrix UBO and interpolate vec4 color. The fourth
draw is authored: T32, vertices written by lanes 0..2, primitive values and
indices written by lane 17, varying clip W, smooth vec2, noperspective float,
per-primitive vec2, signed -23 and unsigned 0x87654321. Ordinary compilation
uses the existing P1 flat-channel broadcast. It does not exercise or enable
multi-primitive record linking, typed 16-bit IO, wide publication or payload.

The original application dispatches through a task shader. Here each original
mesh component is invoked once at workgroup zero, paired with its original
fragment shader. This can earn three exact corpus mesh-component executions,
not three complete application executions or task/mesh pipeline passes. M5
owns task transport. No SPIR-V rewriting or substituted fragment is involved.

The probe binds 192 bytes of projection/model/view matrices at set 0 binding
0 as a uniform buffer. Every matrix affects the observed triangle. The
remaining 512-KiB allocation is a guard; all bytes including the UBO must be
unchanged after the retired draw. Both the probe and independent checker give
screen vertices (64,25), (48,40), (80,40). Pixel centers never lie on edges.
The full 128x64 RGBA32_FLOAT target uses ONE/ONE additive blending. The checker
compares 32,768 finite components per draw, exact background, and an absolute
2e-5 tolerance for nonzero interpolation results. Software controls measured
maximum absolute error <= 2.4e-7. The tolerance is a numerical interpolation
allowance, not a device-performance threshold. Alpha detects loss/duplication;
all color channels and the entire input allocation are checked. Forty-eight
negative checker mutations cover missing/duplicate work, wrong interpolation,
integer sentinel mismatch, NaN, channel swaps, truncation and UBO/guard stores.

The existing Air-passed f987032b1d0 driver is used without code changes.
All four normal pipelines compile through the real ARM ICD and record native
T16/T32 V3/P1 packets in a no-op DRM host fixture; this is not GPU execution.
The paired 4b1e8012b687 parser must accept all four only in research mode 3
and reject their attributes in qualified mode 4. Parser replay also tests
two relocations and every truncated prefix. No kernel change is required.

The device budget is four draws/submissions/processes, 140 seconds, no retries,
stop first. Audit the own-FD VM generation/research profile, exact mapped
driver, native route, retired fence, complete readbacks and per-draw native
render/object counters. The manager owns hardware, queue and recovery. This
kit has zero GPU executions at preparation time and changes no default policy.
