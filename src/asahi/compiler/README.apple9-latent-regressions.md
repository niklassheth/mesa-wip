# Latent regressions and the codegen gate

On 2026-10-23 the overnight Vulkan CTS showed two regressions. Neither was caused by a defect in the change that
exposed it: each was an older bug that had stayed hidden until a codegen change removed what was masking it.

## The two regressions

| | Triangle merging | Robust uniform loads |
|---|---|---|
| Failures | 30 `fragment_shading_barycentric.fast_linked_library.weights`, perspective, MSAA, strip/fan, dynamic topology | 54 `robustness2` `uniform_buffer{,_dynamic}` `dontunroll` `rgba32{i,ui}` `len_{12,31,252}` |
| Latent bug | hk computed when triangle merging is unsafe (off-centre interpolation, sample shading), but it only ever *cleared* `disable_tri_merging`, never set it | With soft faults, a bounded constant vector load was checked once against its last byte, which zeroes in-bounds components too |
| What masked it | The fragment-invocation counter (a ballot and a global atomic) in every fragment shader made the backend disable merging | The vectorizer happened to merge the whole vec4 field, and an all-zero vec4 is a valid robustness2 result |
| Unmasking change | d6e3b580280: draw-time fragment variants omit the counter | bcb0837f948: shader constant data moved out of scratch. The changed register pressure made the windowed vectorizer split the field into two vec2 loads, and the second check dropped an in-bounds `z` |
| What the merge ran | A 31,372-case Air kit covering statistics queries and feedback loops, with no barycentric cases. Identity changed as intended; the device step was batched for later | The host merge gate, corpus identity (2,693 programs changed as intended), ER and LC game pixels. No device CTS |
| What would have caught it | Barycentric `weights` under the library construction types. `bary-gate.txt` held only monolithic pipelines | The Air list `softfault-robust2.txt`, which contains all 54 |
| Fix | 4a10bd5f4fb: hk's decision is authoritative. Host test `hk-fragment-tri-merging` | a3c7c22bf67: per-component checks. Host test `hk-robust-bounded-vector` |

## Lessons

- **A codegen change reaches beyond its intent.**
  - Removing instructions, shifting register pressure or changing how loads merge alters every shader that is
    near some threshold.
  - A list chosen for the feature tests the feature, but latent bugs surface in unrelated families.
- **Both bugs were correctness decisions that rested on a side effect:**
  - a safety flag set only because unrelated instructions happened to be present;
  - a bounds check that was correct only for the vector width the vectorizer happened to choose.

  Make a decision where its reason is known, and give that reason a host test of its own.
- **Identity proves only that the intended programs changed.**
  - It hashes program text, so state outside the assembly, such as `agx_shader_info` flags, is invisible to it.
  - It says nothing about whether a changed program is correct.
- **Draw-time variants differ by pipeline construction type.** A list must include monolithic, `pipeline_library`
  and `fast_linked_library` pipelines, and shader objects where the group has them. The barycentric failures were
  fast-linked only.
- **The device step can't be deferred for a change that alters programs.** Both merges went in without the
  device run (one batched for later, the other with no device CTS), and the overnight run caught both the next
  morning.

## The gate

### Host: for every change under `src/asahi/compiler` or the hk shader lowering
- **Unchanged:** `tmp/claude-ct/merge-gate.sh` and strict corpus identity, using `--keep-going` and the pre-raster
  adapter for GS/tess changes.
- **Shader state:**
  - With `AGX_APPLE9_ASM`, hk prints one `APPLE9_INFO` line per uploaded shader (`agx_apple9_dump_info`). The
    line carries the final `agx_shader_info` fields that program hardware state: `disable_tri_merging`, tag
    writes, ZS and sample-mask behaviour, varying layout, resources and the like.
  - `identity.py` writes their digest to `OUT.state.json`.
  - `identity-compare.py BASE.json CAND.json --json SUMMARY` reports program changes, state changes and
    state-only changes (same program bytes, different state) separately.
  - A merge message must explain a state change the same way it explains a program change.
  - **Check:** removing hk's triangle-merging override changes no program bytes in lethal, willems, vkmark or
    CTS, but shows 14 state-only changes.

### Device: codegen sentinel on the Air
- **When:** required whenever `identity-compare.py` reports a program or state change, or can't compare state.
  It runs before the merge, not batched.
  - `merge-gate.py` takes either `--identity-summary SUMMARY` or `--no-codegen-change`, an attestation for docs and
    tooling.
  - With a change, it builds aarch64 libraries for the base and candidate (`gate-a64-lib.sh`) and runs
    `gate-sentinel.sh`.
- **Air access:**
  - `gate-sentinel.sh` drops `tmp/manager-queue/air/NEXT-codegen-sentinel-<rev>` and waits FIFO for the Air lease.
    The running Air campaign pauses at its next batch boundary.
  - With no Air within two hours, the step fails ("device step not run") instead of skipping.
- **How:** head against candidate, Air to Air, with `runner/run-list.sh` and `compare-results.py`.
- **Blocking:**
  - any case that passes on head but not on the candidate;
  - any new G16 line in dmesg.
- **List:** `tmp/claude-compiletime/gate/make-sentinel.sh VK_MAIN OUT` builds it from the Air runner's lists and the
  vk-main mustpass. On 2026-10-23 that gives 6,920 cases:

| part | cases | why |
|---|---:|---|
| `softfault-robust2.txt` | 5,017 | robustness2, image and buffer robustness under soft faults. These are sensitive to load merging and bounds lowering |
| `fragment_shading_barycentric.{,pipeline_library.,fast_linked_library.}weights` | 1,116 | interpolation and merging across the three construction types |
| `bary-regress.txt`, `e2-regress.txt` | 323 | earlier codegen regressions |
| `normal.txt` | 464 | broad smoke list |

- **Cost and coverage:**
  - Earlier Air results put it at about 1,350 s of serial case time, roughly 3 to 4 minutes at `--jobs 8`.
  - It contains all 54 robust and all 30 barycentric failures from 2026-10-23.
- **Maintenance:**
  - Each regression that escapes adds its family: the smallest pattern that contains the failing cases, plus
    their construction-type twins.
  - Keep the list under about 10 minutes. The overnight full CTS remains the backstop.
