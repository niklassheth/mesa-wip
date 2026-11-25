/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_MESH_H
#define AGX_APPLE9_MESH_H

#include "compiler/nir/nir.h"
#include "agx_compile_apple9.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Native stage-body ABI. These are argument selectors, not physical GPRs or
 * values taken from a capture. The launch builder must populate the selected
 * local root and the two consecutive payload-pointer uniform words. */
/* The driver and compiler share AGX_APPLE9_MESH_KEY. */
struct agx_apple9_mesh_key {
   /* Entry-owned uniform words which stage pointers may not overlap. */
   unsigned reserved_uniform_words;
   unsigned output_root;
   /* Width consumed by the stage's primitive-count handoff. */
   unsigned primitive_count_bits;
   unsigned payload_uniform;
   /* The object root points at this workgroup's grid record. The collective
    * export helper must consume the same root/offset after body completion. */
   unsigned grid_root;
   unsigned grid_offset_B;
   /* Physical completion width, or zero for the API workgroup width. Extra
    * invocations skip the user body and reconverge before native completion.
    * This is independent of the output allocation family's capacity. */
   unsigned physical_threads;
   /* Load the per-parent hardware payload VA from GET_SR 0xe0/0xe1 instead of
    * the draw-wide AGX_APPLE9_MESH_PAYLOAD_UNIFORM pointer. */
   bool hw_payload;
};

#define AGX_APPLE9_MESH_KEY                                                    \
   {                                                                           \
      .reserved_uniform_words = AGX_APPLE9_GRAPHICS_ROOT_WORDS,                \
      .output_root = AGX_APPLE9_MESH_OUTPUT_ROOT,                              \
      .primitive_count_bits = 16,                                              \
      .payload_uniform = AGX_APPLE9_MESH_PAYLOAD_UNIFORM,                      \
      .grid_root = AGX_APPLE9_MESH_GRID_ROOT,                                  \
   }

#define AGX_APPLE9_MESH_UNUSED UINT16_MAX

/* Qualified native completion width, distinct from the 32-lane subgroup.
 * API invocations occupy its first 1..16 lanes. Wider launches need separate
 * hardware qualification of completion and object call participation. */
#define AGX_APPLE9_NATIVE_THREADS 16
/* Logical workgroups above T16 serialize at most four 32-lane subgroups.
 * This compiler model does not qualify physical multi-subgroup execution. */
#define AGX_APPLE9_NATIVE_SERIAL_THREADS 32
#define AGX_APPLE9_NATIVE_MAX_INVOCATIONS 128
/* Hardware task payload limit (16 KiB, matching Profile::PAYLOAD_BYTES). */
#define AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES 16384u

bool agx_apple9_native_workgroup_valid(const nir_shader *nir,
                                       unsigned max_invocations,
                                       const char **reason);
bool agx_apple9_native_workgroup_supported(const nir_shader *nir,
                                          const char **reason);

static inline unsigned
agx_apple9_mesh_indices_per_primitive(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return 1;
   case MESA_PRIM_LINES:
      return 2;
   case MESA_PRIM_TRIANGLES:
      return 3;
   default:
      return 0;
   }
}

static inline unsigned
agx_apple9_mesh_primitive_count_offset(unsigned max_primitives,
                                       unsigned indices_per_primitive)
{
   return ALIGN_POT(max_primitives * indices_per_primitive, 16);
}

/* Output declaration class shared by early driver IO lowering and the native
 * compiler. Other eligibility checks still depend on the body and shader key. */
bool agx_apple9_native_mesh_shape_eligible(const nir_shader *nir);

/* 32-bit component maps shared with the future launch and fragment linkers.
 * Local vertex outputs are component-major; primitive attributes are device
 * records. Indices and culling have distinct byte-addressed local regions. */
/* Most entries a single channel or packed-index collective publishes; more
 * vertices or index words repeat it with an explicit destination. */
#define AGX_APPLE9_PUBLICATION_ROUND 32

/* Compiler-private per-vertex slots mirroring gl_ClipDistance when a fragment
 * shader also reads it: GL-only legacy varyings that SPIR-V mesh shaders never
 * write, excluded from the linked varyings. */
#define AGX_APPLE9_CLIP_MIRROR_SLOT VARYING_SLOT_TEX6

struct agx_apple9_mesh_layout {
   uint16_t vertex_component[VARYING_SLOT_MAX][4];
   uint16_t primitive_component[VARYING_SLOT_MAX][4];
   unsigned max_vertices, max_primitives;
   unsigned vertex_components, primitive_components;
   unsigned primitive_count_B, vertex_count_B, cull_B, position_B;
   unsigned vertex_pitch_B, primitive_stride_B, local_size_B;
   unsigned indices_per_primitive;
   bool culling;
};

static inline unsigned
agx_apple9_mesh_plane_offset(const struct agx_apple9_mesh_layout *layout,
                             unsigned slot, unsigned component)
{
   unsigned idx = layout->vertex_component[slot][component];
   assert(idx != AGX_APPLE9_MESH_UNUSED);
   return layout->position_B + layout->vertex_pitch_B * idx;
}

/* Apple9 (T8132) per-pipe mesh vertex output ring capacity in 16-byte rows
 * (64 KiB). Each non-empty workgroup allocates a physical ring stride of:
 *   stride = ceil(U0_B / 4) * ALIGN_POT(n, 4) + idx_rows
 * where U0_B is the 16-bit auxiliary vertex count passed to MESH_CONTROL,
 * n = vertex_components + (primitive_components != 0), and
 * idx_rows = 4 * DIV_ROUND_UP(max_primitives * indices_per_primitive, 64).
 *
 * When a draw wraps the 4,096-row ring, the last workgroup before row 4,096
 * starts at row (4096 - rem) where rem = 4096 % stride. If rem == 0 or
 * rem >= ceil(max_vertices / 4) * ALIGN_POT(n, 4), its actual vertex writes
 * stay strictly below row 4,096 and never overwrite live WG #0 at row 0,
 * after which the ring pointer resets to 0 for the next workgroup. Passing
 * this constant safe U0_B to MESH_CONTROL for every non-empty workgroup keeps
 * the ring stride invariant across all dynamic SetMeshOutputsEXT(v, p) vertex
 * counts v in [1..max_vertices] and unlocks wg_rows == 32 and wg_rows > 40. */
#define AGX_APPLE9_MESH_RING_ROWS        4096u
#define AGX_APPLE9_MESH_RING_HEADER_ROWS 8u
#define AGX_APPLE9_MESH_SAFE_U0_B_MAX    256u
#define AGX_APPLE9_MESH_SAFE_STRIDE_MAX  2048u

static inline uint64_t
agx_apple9_mesh_idx_rows(const struct agx_apple9_mesh_layout *layout)
{
   if (!layout)
      return 4u;
   uint64_t idx_bytes =
      (uint64_t)layout->max_primitives * layout->indices_per_primitive;
   if (!idx_bytes)
      return 4u;
   return DIV_ROUND_UP(idx_bytes, 64u) * 4u;
}

static inline bool
agx_apple9_mesh_safe_u0_b(const struct agx_apple9_mesh_layout *layout,
                          uint32_t *out_u0_b, uint32_t *out_stride)
{
   if (!layout)
      return false;
   if (!layout->max_vertices) {
      if (out_u0_b)
         *out_u0_b = 0;
      if (out_stride)
         *out_stride = 0;
      return true;
   }
   uint64_t n =
      layout->vertex_components
         ? (uint64_t)layout->vertex_components + (layout->primitive_components != 0)
         : 4u;
   if (n > 256 || layout->max_vertices > AGX_APPLE9_MESH_SAFE_U0_B_MAX)
      return false;
   uint64_t quads = ((uint64_t)layout->max_vertices + 3u) / 4u;
   uint64_t planes = (n + 3u) & ~UINT64_C(3);
   uint64_t idx_rows = agx_apple9_mesh_idx_rows(layout);
   if (idx_rows >= AGX_APPLE9_MESH_RING_ROWS)
      return false;
   uint64_t vtx_rows = quads * planes;
   if (vtx_rows + idx_rows > AGX_APPLE9_MESH_SAFE_STRIDE_MAX)
      return false;

   for (uint64_t q = quads; q <= AGX_APPLE9_MESH_SAFE_U0_B_MAX / 4u; ++q) {
      uint64_t stride = q * planes + idx_rows;
      if (stride > AGX_APPLE9_MESH_SAFE_STRIDE_MAX)
         break;
      uint64_t rem = AGX_APPLE9_MESH_RING_ROWS % stride;
      if (rem == 0 || rem >= vtx_rows) {
         if (out_u0_b)
            *out_u0_b = q == quads ? layout->max_vertices : (uint32_t)(q * 4u);
         if (out_stride)
            *out_stride = (uint32_t)stride;
         return true;
      }
   }
   return false;
}

struct agx_apple9_mesh_body_info {
   struct agx_apple9_mesh_layout layout;
   struct agx_apple9_mesh_key key;
   /* User shared storage follows the output allocation on the same local
    * root. Keep output rows separate: mesh state must not count user data as
    * vertex planes. The allocation includes word padding for narrow accesses.
    * Launch integration must qualify this combined allocation independently. */
   unsigned shared_offset_B, shared_size_B, local_allocation_B;
   /* Declared API dimensions, before flattening to the physical launch. */
   uint16_t logical_size[3];
   /* Unresolved pieces of the native contract. Completion lowering clears
    * its own requirement; launch integration must honor the remaining ones. */
   bool requires_native_entry;
   bool requires_object_completion;
   bool requires_mesh_completion;
   /* Lowered for multi-round publication at T32 (bound at upload). */
   bool publication_rounds;
   /* Rasterizer point size (points topology) and packed layer/viewport word
    * published in VS UVS order after user varyings and before clip distances. */
   bool writes_point_size;
   bool writes_layer_viewport;
   /* Rasterizer clip distances published after the user varyings. */
   unsigned clip_distances;
   /* Clip distances are also linked varyings; their rasterizer copy reads
    * the mirror planes at AGX_APPLE9_CLIP_MIRROR_SLOT. */
   bool clip_linked;
   /* Cull distances evaluated at completion from compiler-owned shared
    * storage at cull_B within the shared region; not published. */
   unsigned cull_distances, cull_B;
   /* Multi-primitive CullPrimitive flags in compiler-owned shared storage,
    * consumed before publication. They are not hardware cull records. */
   bool cull_primitive;
   unsigned cull_primitive_B;
   /* V3/P1 primitive outputs lowered to flat planes or a local cull flag. */
   uint64_t primitive_broadcast;
   /* Multi-primitive flat attributes use one private provoking vertex per
    * primitive. Body stores occupy rows starting at original_max_vertices;
    * completion compacts them after the actual application vertices. */
   uint64_t primitive_vertices;
   unsigned original_max_vertices;
   /* User body is wrapped in a uniform non-empty grid check (gl_NumWorkGroups
    * xyz != 0) so empty GPU grids (e.g. conditional rendering) can launch a
    * safe (1,1,1) dummy workgroup without running application instructions or
    * side effects. */
   bool guard_empty_grid;
   /* Hardware task payload bytes (<= 16 KiB) and 16-byte aligned stride.
    * When writes_child_grid is set, the object epilogue stores the emitted
    * child mesh grid (3 x u32) at task_payload_stride_B so a paired native
    * mesh shader reading gl_NumWorkGroups reads the child grid instead of the
    * task draw grid. */
   unsigned task_payload_size_B;
   unsigned task_payload_stride_B;
   bool writes_child_grid;
   bool reads_child_grid;
   bool reads_native_payload;
   unsigned argument_uniform_words;
};

static inline unsigned
agx_apple9_mesh_argument_words(const struct agx_apple9_mesh_key *key)
{
   return ALIGN_POT(MAX3(AGX_APPLE9_GRAPHICS_ROOT_WORDS,
                         2 * (MAX2(key->output_root, key->grid_root) + 1),
                         key->payload_uniform + 2),
                    4);
}

/* Validate supported normal mesh/task NIR and lower only the understood IO.
 * Returns success, not progress. Rejects unsupported IO before changing NIR.
 * IO must already use per-vertex/per-primitive intrinsics and explicit payload
 * offsets; shared IO must also be explicit. Dynamic vertex/primitive indices
 * and component write masks work. Mesh shared IO is rebased after outputs;
 * task shared IO remains unsupported. */
bool agx_nir_lower_apple9_mesh(nir_shader *nir,
                               const struct agx_apple9_mesh_key *key,
                               struct agx_apple9_mesh_body_info *info,
                               const char **reason);

/* Lower load_global_invocation_id to workgroup_id * workgroup_size +
 * local_invocation_id before rewriting logical workgroup dimensions to a
 * different physical launch width or 1D flattened shape. */
bool agx_nir_lower_apple9_mesh_global_id(nir_shader *nir);

/* Preserve logical workgroup system values while masking extra completion
 * invocations between converged workgroup barriers. Call after IO/return
 * lowering and before ordinary compilation. Returns success, not progress.
 * Barriers nested in application branches/loops take the flag protocol in
 * the output count header, which needs the body layout. */
bool agx_nir_mask_apple9_mesh_body(nir_shader *nir,
                                   const struct agx_apple9_mesh_key *key,
                                   const struct agx_apple9_mesh_layout *layout,
                                   const char **reason);

/* Emit the understood dataflow of the generated object completion helper:
 * an output fence, then one local grid-word export per invocation 0/1/2.
 * The caller must provide a collective helper context. This is not a valid
 * inline task epilogue: call/frame/entry setup remains unresolved. */
struct nir_builder;
void agx_nir_apple9_mesh_workgroup_barrier(struct nir_builder *b);
/* Expand collective grid calls after NIR optimization and validate closed
 * regions before selection. Returns success, including when no calls exist. */
bool agx_nir_lower_apple9_object_calls(struct nir_shader *nir, const char **reason);

bool agx_nir_apple9_object_grid_export(struct nir_builder *b,
                                       const struct agx_apple9_mesh_key *key);

/* Driver-generated ordinary T1 task NIR. Its draw argument points to three
 * uint32_t mesh group counts passed to launch_mesh_workgroups. Compile through
 * native_object to mask the body and append the physical T16 collective. */
nir_shader *agx_nir_apple9_taskless_object(
   const struct agx_apple9_mesh_key *key);
/* Guarded variant for GPU-zeroable grids (e.g. conditional rendering):
 * substitutes (1, 1, 1) whenever any grid axis is zero so hardware never
 * launches an unqualified empty child grid; paired with guard_empty_grid on
 * the mesh shader. */
nir_shader *agx_nir_apple9_taskless_object_guarded(
   const struct agx_apple9_mesh_key *key);

/* Compile the taskless producer and a token-free collective grid call in one
 * allocation unit. The measured width-16/convention-1/POP2 contract remains
 * provisional pending portability and GPR-preservation results. body_only
 * stays set: native entry, frame backing and reservation/submission are still
 * driver promotion gates. The disabled gate neither mutates NIR nor output. */
bool agx_compile_apple9_taskless_object(
   bool enable_native_mesh, const struct agx_apple9_mesh_key *key,
   struct agx_shader_part *out, const char **reason);

/* Options for lowering real application MESA_SHADER_TASK shaders into native
 * Apple9 OBJECT programs with optional 16 KiB task payload and child-grid
 * handoff. NULL retains default payload stride with no child-grid write. */
struct agx_apple9_native_object_options {
   /* 16-byte aligned stride (bytes) for placing the optional 12-byte child-grid
    * handoff after user payload bytes. Zero uses ALIGN_POT(task_payload_size, 16). */
   unsigned payload_stride_B;
   /* Store the emitted (x, y, z) child mesh grid to payload + task_payload_stride_B
    * before OBJECT_GRID_EXPORT so a paired native mesh reading gl_NumWorkGroups
    * receives the emitted child grid. */
   bool write_child_grid;
   /* Load the per-parent hardware payload VA from GET_SR 0xe0/0xe1 instead of
    * AGX_APPLE9_MESH_PAYLOAD_UNIFORM. */
   bool hw_payload;
};

/* Application or taskless task NIR -> native Apple9 OBJECT program.
 * Supports 1..16 invocations at physical T16 or 1..128 invocations at physical
 * T32 (serialized), user shared memory (<= 32 KiB minus grid header), task
 * payload (<= 16 KiB) with loads/stores/atomics via AGX_APPLE9_MESH_PAYLOAD_UNIFORM,
 * and 0/1/conditional launch_mesh_workgroups. Both entrypoints retain body_only. */
bool agx_nir_lower_apple9_native_object_ex(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key,
   const struct agx_apple9_native_object_options *options,
   struct agx_apple9_mesh_body_info *info, const char **reason);
bool agx_nir_lower_apple9_native_object(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key,
   struct agx_apple9_mesh_body_info *info, const char **reason);
bool agx_compile_apple9_native_object_ex(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key,
   const struct agx_apple9_native_object_options *options,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason);
bool agx_compile_apple9_native_object(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key, struct agx_shader_part *out,
   struct agx_apple9_mesh_body_info *info, const char **reason);

/* Compile the lowered body through ordinary instruction selection and RA.
 * The returned binary is deliberately body-only (see info), not a production
 * native launch path. Native compilation is separate from the compute-emulated
 * Vulkan backend. */
bool agx_compile_apple9_mesh_body(nir_shader *nir,
                                  const struct agx_apple9_mesh_key *key,
                                  struct agx_shader_part *out,
                                  struct agx_apple9_mesh_body_info *info,
                                  const char **reason);

/* Compile already lowered native IO plus caller-owned generated operations.
 * This retains the body-only upload gate; an executable client must supply
 * and audit its own entry/completion/launch contract before clearing it. */
bool agx_compile_apple9_mesh_lowered(nir_shader *nir,
                                     const struct agx_apple9_mesh_key *key,
                                     struct agx_shader_part *out,
                                     const char **reason);

/* Fragment interpolation uses the same grouped scalar order as VS output.
 * NULL selects the position-only, zero-to-one-depth contract. */
struct agx_apple9_native_mesh_options {
   /* Zero uses the qualified T16 launch. T32 executes up to 128 logical
    * invocations as serialized subgroups; the driver must qualify that launch. */
   unsigned physical_threads;
   bool clip_halfz;
   /* Dynamic depth_clip_negative_one_to_one: load float clip_z_coeff
    * (0.5 for [-w,w], 0.0 for [0,w]) from UBO `clip_z_ubo` at `clip_z_offset_B`. */
   bool dynamic_clip_z;
   uint16_t clip_z_ubo;
   uint32_t clip_z_offset_B;
   bool attributes;
   /* Exact single-primitive lowering; no native primitive-record ABI. */
   bool primitive_broadcast;
   bool primitive_vertices;
   /* Provisional packed-index publication. Compiler-only opt-in until the
    * narrow declaration class and launch path receive device qualification. */
   bool packed_index_publication;
   /* The driver selected multi-round publication; requires T32. */
   bool publication_rounds;
   /* The fragment shader reads gl_ClipDistance: also link the distances as
    * varyings; the rasterizer copy then reads private mirror planes. */
   bool link_clip_distances;
   /* Opt-in lowering of per-primitive Layer/ViewportIndex via the packed
    * rasterizer word in VS UVS order (plus flat ViewportIndex varying). */
   bool layer_viewport;
   /* Guard the lowered mesh body against GPU-zeroed grids (e.g. conditional
    * rendering) launched as a (1, 1, 1) dummy workgroup by the guarded object. */
   bool guard_empty_grid;
   /* Mesh shader paired with a native hardware OBJECT stage: allow task payload
    * up to AGX_APPLE9_MAX_TASK_PAYLOAD_BYTES directly via
    * AGX_APPLE9_MESH_PAYLOAD_UNIFORM (or GET_SR 0xe0/0xe1 when
    * native_object_hw_payload is set), and lower load_num_workgroups from the
    * emitted child grid at native_object_payload_stride_B. */
   bool native_object_payload;
   bool native_object_hw_payload;
   unsigned native_object_payload_stride_B;
   uint64_t outputs_flat, outputs_linear;
};

/* Candidate number of publishing lanes for packed 8-bit topology indices.
 * Caller proves ceil(K * maximum_P / 4) fits the physical completion width.
 * The result is a prefix length, not a per-lane source offset. */
struct nir_builder;
nir_def *agx_nir_apple9_index_publication_lanes(struct nir_builder *b,
                                              nir_def *primitives, unsigned k);

/* Compiler-only native development gate, disabled unless explicitly enabled.
 * This accepts API sizes 1..16 at physical T16, or 1..128 at explicit T32,
 * and V3/P1. It emits the
 * complete mesh body and hardware output submissions in one compile.
 * clip_halfz converts final Z from [-W,W] to [0,W] before submission.
 * The result still requires a native launch/object contract; generic shader
 * upload rejects apple9_body_only. This does not enable a Vulkan feature. */
bool agx_nir_lower_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                      const struct agx_apple9_native_mesh_options *options,
                                      struct agx_apple9_mesh_body_info *info,
                                      const char **reason);
bool agx_compile_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                    const struct agx_apple9_native_mesh_options *options,
                                    struct agx_shader_part *out,
                                    struct agx_apple9_mesh_body_info *info,
                                    const char **reason);
/* Complete compilation after the driver lowers any sysvals introduced by
 * native completion (notably the dynamic provoking vertex mode). */
bool agx_compile_apple9_native_mesh_lowered(
   nir_shader *nir, const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, const struct agx_apple9_mesh_body_info *info,
   const char **reason);

#ifdef __cplusplus
}
#endif
#endif
