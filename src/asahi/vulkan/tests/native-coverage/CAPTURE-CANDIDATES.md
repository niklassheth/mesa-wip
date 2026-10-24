# Mesh capture candidates without a Steam account — msg-180

Use source-built Linux demos first. These are application captures to add
later, not evidence that we have a shipped-game capture. Native ARM builds
are the preferred route; FEX means native-host FEX on the approved kernel,
never muvm. No candidate was installed or run for this list.

| Priority | App / demo using VK_EXT_mesh_shader | Execution route | Useful capture |
|---|---|---|---|
| 1 | Khronos Vulkan-Samples: `mesh_shading` | Linux source build for aarch64; the upstream build supports Linux and Android | Cheapest complete taskless pipeline/draw capture for the three small M0 mesh language variants. Then `mesh_shader_culling` for task payload and a 2D workgroup. [Project/build](https://github.com/KhronosGroup/Vulkan-Samples), [sample](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/mesh_shading). |
| 2 | Sascha Willems `meshshader` | Linux CMake source build for aarch64; x86-64 Linux build under FEX is an alternative | Small task+mesh program. GLSL/HLSL/Slang variants give complete pipeline interfaces for the three M0 entries. Launch the `meshshader` executable; its shader-language and finite-frame benchmark options allow bounded capture. [Source](https://github.com/SaschaWillems/Vulkan/tree/master/examples/meshshader), [build/runtime options](https://github.com/SaschaWillems/Vulkan). |
| 3 | NVIDIA `vk_meshlet_cadscene` | Linux x86-64 build under FEX; native aarch64 source build is a portability candidate, not verified here | Select **mesh ext**, not mesh nv. The default bundled/downloaded scenes and task/culling switches cover V64/P126, payload and primitive culling. Vulkan-only CMake build avoids the OpenGL renderer. Start with one model copy and disable optional barycentric shading. [Upstream build, EXT renderer and scenes](https://github.com/nvpro-samples/gl_vk_meshlet_cadscene). |
| 4 | Niagara | Linux source build; try native aarch64 first, x86-64/FEX if a dependency needs it | Use the small `data/kitten.obj`, not Bistro, for the first bounded task/mesh and indirect-draw capture. The current project asks for Vulkan 1.4; check all enabled features before scheduling. Its mesh source explicitly uses EXT mesh shaders. [Build/run](https://github.com/zeux/niagara), [EXT mesh source](https://github.com/zeux/niagara/blob/master/src/shaders/meshlet.mesh.glsl). |

Architecture routes above are build/emulation plans inferred from the
upstream Linux sources and toolchains, not successful local ARM/FEX runs.
FEX supports x86/x86-64 Linux on ARM64 and graphics thunking, but this does
not prove any particular demo's extension path. [FEX upstream](https://github.com/FEX-Emu/FEX).
The first two are the cheapest candidates because they avoid a large scene
and broad engine feature dependencies. All use public source/assets and
require no Steam login.

For the later scheduled capture, retain the exact app commit and options,
SPIR-V entry points and specialization values, task/mesh/fragment pairing,
descriptor layouts and mesh draw calls, with renderer/driver identity.
Verify that the EXT path was actually selected. Capture both CPU-side
pipeline state and a bounded frame workload; shader binaries alone do not
measure native draw coverage. Keep these captures separate from the fixed
18-entry shader denominator until an explicit corpus revision is accepted.
