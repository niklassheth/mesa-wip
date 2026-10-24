**M0 coverage ledger — msg-176/178, 24 October 2026**

**Today’s default native path admits 0/18 (0%) of the collected public mesh SPIR-V entry points: 0/14 demo shaders across four projects, and 0/4 Wicked Engine variants.** Their seven accompanying task entry points are also excluded. **Real-game mesh coverage is unknown (zero mesh entry points in the usable local captures), not 0% of games.** This is a small, named convenience corpus, not a market-weighted estimate. No new device draws were used.

| Corpus | Denominator | Default-native admission | Scope |
|---|---:|---:|---|
| Khronos Vulkan-Samples | 6 mesh / 1 task | 0 / 0 | Mesh shading, GS→mesh and mesh culling; upstream GLSL/HLSL/Slang binaries where present |
| Sascha Willems Vulkan | 3 mesh / 3 task | 0 / 0 | Mesh shader example, three upstream language binaries |
| Niagara | 1 mesh / 1 task | 0 / 0 | Unchanged renderer shaders, app’s Vulkan 1.3 compilation, specialization defaults; mesh T64/V64/P96 |
| NVIDIA meshlet CAD demo | 4 mesh / 1 task | 0 / 0 | Basic/culling, taskless/task variants; app V64/P126, 32-wide device preference |
| Wicked Engine | 4 mesh / 1 task | 0 / 0 | Object, prepass, alpha-test prepass, simple; engine Vulkan flags, base permutations; T32/V64/P124 |

Count distinct **binary SHA-256 + stage + entry point** once in each category. All 25 collected entries are distinct under that rule. Language and task variants still overweight related source: the 14 demo mesh entries cover seven source families in four projects; the four engine mesh entries are one object-shading family. All families/projects also have zero admitted entries. Engine code is an application proxy, **not a shipped-game capture**. The CAD `scull` file is an include, not a standalone entry point, and is not counted. No CTS or authored synthetic fixture enters these percentages.

The host utility imports the actual SPIR-V through Mesa, gathers NIR metadata and calls **`hk_apple9_native_mesh_packed_shape_eligible`**, with research overrides unset. All 18 fail at declaration/workgroup admission; none is a candidate requiring later binary/upload validation. For demos, primary mutually exclusive reasons are seven output-class exclusions and seven workgroup exclusions; all four engine shaders fail the output class. Overlapping, pre-link interface reasons include payloads (8 mesh entries), application shared storage (6), extra outputs (13), primitive outputs (6) and non-triangle topology (2). These secondary counts describe imported NIR; linking/specialization can eliminate unused interfaces. The recorded configurations and their primary structural exclusions establish the zero admission result; no shader was rewritten to fit the native path.

A passing declaration gate would **not** prove native execution: instruction lowering, register/frame allocation, linked fragment realization, VM qualification and draw state must also pass. Actual draw-weighted coverage remains unmeasured. Conversely a failed structural gate proves default fallback without a GPU trial. An authored T16/V3/P1 control passes the actual gate and the same T1 source fails; controls are excluded from the corpus. Assumed platform is a qualified G16G VM with native-index enabled. Explicit user opt-out or an unqualified VM can only reduce admission.

**Local game evidence.** The inventory reads pipeline stage records, including graphics pipeline libraries. These are neither unique linked pipelines nor draws.

| Local game cache | Graphics pipeline records parsed | Mesh/task stage uses | Interpretation |
|---|---:|---:|---|
| CS2 `steamapp_pipeline_cache.foz` | 7,323 | 0 / 0 | Usable cache; VS/FS library records |
| Dota 2 `dota-steam.foz` | 2,132,604 | 0 / 0 | Usable cache; includes 66 GS stage uses |
| Lethal Company capture | 54 | 0 / 0 | Usable capture; VS/FS records |
| Elden Ring whitelist | 0 | 0 / 0 | Only four compute shader/pipeline entries; inadequate graphics coverage |
| CS2 `steam_pipeline_cache.foz` | 0 before incomplete region | Not established for full file | Null key followed by nonzero trailing data; 1,414,878,043 bytes unparsed. Excluded from coverage conclusions |

Thus the three usable graphics caches contain **2,139,981 pipeline records and zero mesh/task stage uses**. They cannot supply a real-game mesh-shader denominator. GS/TES conversion is off by default and is not evidence of application mesh usage. Every cache’s whole-file hash, stage counts and completeness status are retained. A capture from a game actually using mesh shaders is still needed; no game result has been inferred from demos.

**Current contract and rejection ledger.** The public Vulkan feature/property contract remains backed by emulation: task/mesh 128 invocations, each local axis ≤128, grid axes ≤65,535 and total ≤2²²; payload 16 KiB; task/mesh shared and payload+shared 32 KiB; mesh output 32 KiB, payload+output 48,128 B; V/P256, components128, layers8, multiview and mesh queries. Preferred width32 is a hint. Independent and combined limits apply together. Native restrictions below do not reduce those advertised values.

| Restriction / provenance in current Mesa | Classification | Exit evidence / roadmap step |
|---|---|---|
| G16G, own-DRM-fd capability, immutable per-VM generation/profile; no qualified VM means fallback (`agx_device.c`, `hk_apple9_mesh.c`) | Required capability/lifetime contract | Preserve per-VM positive/negative cache identity for each new kernel/Mesa version |
| 31 exact `(T,V,P)` classes, T16/T32; V≤T, ceil(3P/4)≤T (`hk_apple9_native_index_shapes.def`) | Unqualified surrounding region | Common-base collective model and ordinary Vulkan P15/16/17 first; M1, then M2/M3 |
| T×1×1 only; variable size and non-32 required subgroup excluded (`hk_apple9_native_mesh_packed_shape_eligible`) | Implementation/qualification gap | Logical serialization and subgroup/barrier semantics; wider physical widths M2 |
| Position + triangle indices only; no API shared storage, payload or per-primitive values, no 16-bit outputs (`hk_apple9_mesh.c`) | Implementation/qualification gap | M5/M6 full interfaces; payload ownership before M4 rebinding |
| Points/lines excluded from packed path | Implementation/qualification gap | Native topology, primitive indices and raster oracle, M6 |
| Complete object is generated taskless T16; API task stage rejected (`hk_cmd_draw.c`, native object compiler) | Implementation gap | Native task workgroups, payload, child counts and side effects, M5 |
| Native direct grid: nonzero axes ≤65,535, product ≤2²⁰−1; indirect/count rejected (`hk_apple9_native_mesh_grid_valid`, draw eligibility) | Implementation gap; grid/API mismatch unresolved | Partition child grids preserving IDs and task effects; indirect/zero semantics M5/M7 |
| One native parent dispatch per production render; repeated-state/input lifetime unqualified | Qualification gap | Distinct actual payload values, roots and owned witnesses; M4 proposal |
| Output/shared combined backing ≤32 KiB; payload must not alias output root (`agx_nir_lower_apple9_mesh.c`) | Allocation model gap | Separate user-visible capacity/private overhead and lifetime; M3/M5/M6 |
| Native binary rejects scratch, returned-atomic frame, unexpected ABI/layout/preamble; object frames16/16, mesh64/64 (`hk_apple9_native_shader_valid`) | Implementation/qualification gap | Proper spill/frame backing and stress cases, M6 |
| Instruction capability check runs real native lowering on clone; failure retains fallback (`hk_apple9_native_mesh_check`) | Compiler implementation gap, reason-specific | Implement generalized lowering; preserve source semantics and test the rejected operation |
| Native linked fragment/program pair identity required (`hk_apple9_use_native_mesh`) | Linking/integration requirement | Linked native realizations for full interfaces, M6/M7 |
| One sample/layer, no multiview, conditional rendering, mesh-generated-primitives/pipeline/occlusion queries, XFB, non-fill polygon (`hk_apple9_native_mesh_state_eligible`) | Integration/qualification gaps | Separate bounded state/query/synchronization tests, M7 |
| Conservative raster restriction for culling emulation | Semantics/qualification gap | Correct native cull records and conservative raster tests, M6/M7 |
| GS/TES defaults off and much narrower draw/stage classes | Separate unqualified conversion feature | Correct rebinding, useful workload coverage and remaining floor; no inference from mesh admission |
| `HK_APPLE9_NATIVE_INDEX=0`, legacy override or driconf disable | Explicit user policy | Preserve opt-out; never classify as hardware rejection |

No row above has established a general hardware impossibility. A production native region needs kernel/Mesa versioning, resource/lifetime admission and cache separation together; the shared DRM UAPI stays unchanged.

**Pinned resource model.** For Position-only triangles, let maxima be V/P, runtime counts v/p and physical width T. Index bytes occupy `[0,3P)`; primitive count is at `align(3P,16)`, vertex count four bytes later, Position starts at that offset +16. Four component planes have pitch `4*align(V,4)`; total output backing is `align(3P,16)+16+16*align(V,4)`. Shared backing begins at output size aligned to16 and contributes its size aligned to4. Mesh launch local allocation is a power of two, at least128. State word10 (including leading header) is `(P−1)|((ceil(P/16)−1)<<16)`, word11 `0x400800|T`, word13 `1+4*ceil(V/4)`. Current packed completion uses a common base and a runtime prefix `max(1,ceil(3p/4))`; it cannot publish more index words or vertices than its established width. Local bytes, TVB capacity and payload ownership are separate resources. See the accepted roadmap for source links and the broader hardware questions.

**Retained evidence and replay quality.** The v3 checker re-audited all **1,058 completed trials**: 982 PASS, 76 completed pixel mismatches, one wide timeout and 195 unattempted; zero per-trial code/state/Q/pixel audit errors; 700 candidate-scope passes. Its historical `NARROW-COMPLETE.json` persistence receipt is now absent from the retained root, so the new replay explicitly returns **evidence GO=false** and cannot reverify that old 26,459-file seal. The older accepted report recorded a valid receipt; its report is preserved, not substituted for a fresh integrity check. This archival gap does not turn completed failures into passes or change today’s source allowlist. M1 reuses the S2 observations (48/48 for T16V16P16, four direct declaration trials), with the historical receipt limitation stated. The later Air default/index and B1 results remain separate evidence. The B1 raw audit rehashed 9,850 files and reproduced all 3,276 intervals, 36 counting images and 48 image groups; it establishes identical-input batching only.

**Reproduce / artifacts.** [Full corpus, hashes, commands, raw metadata and game inventories](/home/nsheth/Projects/asahi/tmp/codex-upstream/native-mesh/native-coverage-msg176-20261024), [per-entry coverage JSON](/home/nsheth/Projects/asahi/tmp/codex-upstream/native-mesh/native-coverage-msg176-20261024/COVERAGE.json), [v3 replay](/home/nsheth/Projects/asahi/tmp/codex-upstream/native-mesh/native-coverage-msg176-20261024/v3-replay/ANALYSIS.json), [accepted roadmap](../../README.apple9-native-mesh-roadmap.md).

```
python3 fetch-corpus.py CORPUS
python3 build-corpus.py CORPUS --dxc /path/to/dxc/bin/dxc
# Build hk-native-mesh-coverage through the persistent Mesa ccache build.
python3 analyze.py CORPUS --gate /path/to/build/src/asahi/vulkan/hk-native-mesh-coverage
python3 game-stages.py CACHE.foz --game NAME --out INVENTORY.json
```

The source manifest pins each project commit, URL, Git blob and SHA-256; licenses accompany sources. DXC v1.9.2609 and glslang versions/build commands are retained. DXC download SHA-256 is `96faadc7f5c282d2ffda49804beb4c3ee38127bc252b723234e3c5cdf7aa39a1`. Native hardware policy and imported behavioral documents are unchanged. M1’s P15/16/17 discriminator is expected to leave this corpus at 0/18: these shaders need logical-workgroup, larger-output and interface/task work beyond that first step. Its next device budget is six native draws.
