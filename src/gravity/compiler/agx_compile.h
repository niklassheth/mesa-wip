/*
 * Copyright 2018-2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/nir/nir.h"
#include "util/shader_stats.h"
#include "util/u_dynarray.h"
#include "util/u_tristate.h"
#include "shader_enums.h"

/* Shared by the standalone compiler and the driver's device initialization. */
bool agx_apple9_trace_enabled(void);

struct agx_varyings_fs {
   /* Number of coefficient registers used */
   unsigned nr_cf;
};

union agx_varyings {
   struct agx_varyings_fs fs;
};

struct agx_interp_info {
   /* Bit masks indexed by I/O location of flat and linear varyings */
   uint64_t flat;
   uint64_t linear;
};
static_assert(sizeof(struct agx_interp_info) == 16, "packed");

struct agx_rodata {
   /* Offset in the binary */
   uint32_t offset;

   /* Base uniform to map constants */
   uint16_t base_uniform;

   /* Number of 16-bit constants to map contiguously there */
   uint16_t size_16;
};

/* Produced user scalars: position occupies four of the 128 publication
 * entries. Other system outputs share the remaining bank. This is not a
 * limit on declared fragment inputs: semantic locations are compacted
 * independently of declaration order and never encoded directly as
 * hardware slot numbers. */
#define AGX_APPLE9_MAX_VARYING_COMPONENTS 124
/* UVS slots: position, user scalars, point size, the layer/viewport word and
 * eight clip distances. */
#define AGX_APPLE9_MAX_UVS_SLOTS (4 + AGX_APPLE9_MAX_VARYING_COMPONENTS + 10)
/* Varying scalars a fragment shader may read per vertex. */
#define AGX_APPLE9_MAX_PER_VERTEX 32
/* Interpolation modes are indexed by compacted scalar, independent of API
 * locations. Two words cover the complete publication-addressing namespace. */
struct agx_apple9_interp_mask {
   uint64_t lo, hi;
};

static inline void
agx_apple9_interp_mask_set(struct agx_apple9_interp_mask *mask, unsigned index)
{
   assert(index < 128);
   if (index < 64)
      mask->lo |= UINT64_C(1) << index;
   else
      mask->hi |= UINT64_C(1) << (index - 64);
}

static inline bool
agx_apple9_interp_mask_test(struct agx_apple9_interp_mask mask, unsigned index)
{
   assert(index < 128);
   return ((index < 64 ? mask.lo : mask.hi) >> (index & 63)) & 1;
}

static inline bool
agx_apple9_interp_mask_equal(struct agx_apple9_interp_mask a,
                            struct agx_apple9_interp_mask b)
{
   return a.lo == b.lo && a.hi == b.hi;
}

/* Interpolants from compatibility GL and programmable shaders. Clip distances
 * also have a separate copy in the rasterizer system-value group. */
static inline bool
agx_apple9_varying_supported(unsigned location)
{
   return (location >= VARYING_SLOT_COL0 && location <= VARYING_SLOT_TEX7) ||
          location == VARYING_SLOT_BFC0 || location == VARYING_SLOT_BFC1 ||
          location == VARYING_SLOT_PRIMITIVE_ID ||
          location == VARYING_SLOT_VIEWPORT ||
          location == VARYING_SLOT_CLIP_VERTEX ||
          location == VARYING_SLOT_CLIP_DIST0 || location == VARYING_SLOT_CLIP_DIST1 ||
          location == VARYING_SLOT_CULL_PRIMITIVE ||
          location == VARYING_SLOT_CULL_PRIMITIVE + 1 ||
          (location >= VARYING_SLOT_VAR0 && location <= VARYING_SLOT_VAR31);
}

struct agx_apple9_varying_layout {
   uint8_t mask[64]; /* Per-component masks indexed by gl_varying_slot. */
   /* Physical UVS order is smooth, flat, then linear, as required by the
    * clipper's varying counts. Values within each group use semantic order. */
   uint8_t group[64];
   uint8_t count;
   bool cull_distance_mask; /* Flat, primitive-wide negative-plane bits. */
   uint8_t reserved[2];
};

static inline unsigned
agx_apple9_varying_group(uint64_t flat, uint64_t linear, unsigned location)
{
   return (flat & BITFIELD64_BIT(location)) ? 1 :
          (linear & BITFIELD64_BIT(location)) ? 2 : 0;
}

/* Returns the packed user scalar index, excluding the four position words. */
static inline int
agx_apple9_varying_index(const struct agx_apple9_varying_layout *layout,
                     unsigned location, unsigned component)
{
   if (!layout || !agx_apple9_varying_supported(location) || component >= 4)
      return -1;
   unsigned semantic = location;
   if (!(layout->mask[semantic] & BITFIELD_BIT(component)))
      return -1;
   unsigned index = util_bitcount(layout->mask[semantic] & BITFIELD_MASK(component));
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if (layout->group[i] < layout->group[semantic] ||
          (layout->group[i] == layout->group[semantic] && i < semantic))
         index += util_bitcount(layout->mask[i]);
   }
   return index;
}

/* Unpack every active scalar in layout order into parallel (location, component)
 * arrays. Returns layout->count. */
static inline unsigned
agx_apple9_varying_slots(const struct agx_apple9_varying_layout *layout,
                         uint8_t locs[AGX_APPLE9_MAX_VARYING_COMPONENTS],
                         uint8_t comps[AGX_APPLE9_MAX_VARYING_COMPONENTS])
{
   unsigned idx = 0;
   if (!layout)
      return 0;
   for (unsigned g = 0; g < 3; ++g) {
      for (unsigned s = 0; s < ARRAY_SIZE(layout->mask); ++s) {
         if (!layout->mask[s] || layout->group[s] != g)
            continue;
         unsigned m = layout->mask[s];
         while (m) {
            unsigned c = u_bit_scan(&m);
            if (idx < AGX_APPLE9_MAX_VARYING_COMPONENTS) {
               if (locs)
                  locs[idx] = s;
               if (comps)
                  comps[idx] = c;
            }
            ++idx;
         }
      }
   }
   return idx;
}

/* Software capacity of the graphics address table, separate from launcher
 * slots. */
#define AGX_APPLE9_MAX_GRAPHICS_BUFFERS 32
#define AGX_APPLE9_MAX_VARY_RELOCS      256

/* Interface between a linked Apple9 vertex prolog and its vertex main.
 * Each read (attribute, component) scalar is passed in a fixed 32-bit GPR
 * starting at AGX_APPLE9_ABI_REG_BASE in ascending (attribute, component)
 * order. */
struct agx_apple9_vs_prolog_iface {
   uint16_t attribs_read;
   uint16_t integer_attribs;
   uint8_t attrib_mask[16];
   bool uses_prolog;
};

/* Interface between a linked Apple9 fragment main and its fragment epilog.
 * ABI GPRs start at AGX_APPLE9_ABI_REG_BASE (r8) in fixed deterministic order:
 *  1. Each written FRAG_RESULT_DATA0..7 vec4 (or FRAG_RESULT_COLOR at slot 0):
 *     4 consecutive 32-bit GPRs per set bit of `outputs_written`.
 *  2. Optional dual-source SRC1 vec4 (`writes_src1`): 4 GPRs.
 *  3. Optional FRAG_RESULT_DEPTH (`writes_depth`): 1 GPR (FP32).
 *  4. Optional FRAG_RESULT_STENCIL (`writes_stencil`): 1 GPR (low 16 bits).
 *  5. Optional accumulated killed-sample mask (`main_defers_zs_test` or
 *     `early_fragment_tests && has_killed_mask`): 1 GPR (low 16 bits).
 * Per-sample shading passes the active sample bitmask (1 << s) in fixed
 * physical GPR AGX_APPLE9_SAMPLE_MASK_ABI_REG (r63). */
#define AGX_APPLE9_SAMPLE_MASK_ABI_REG 63

struct agx_apple9_fs_epilog_iface {
   uint8_t outputs_written;
   bool writes_src1;
   bool writes_frag_color;
   bool writes_depth;
   bool writes_stencil;
   bool has_tile_lock;
   bool has_tile_preamble;
   bool early_fragment_tests;
   bool has_killed_mask;
   bool per_sample;
   bool main_defers_zs_test;
   bool uses_epilog;
};

struct agx_shader_info {
   mesa_shader_stage stage;
   uint32_t binary_size;

   /* Apple9 graphics API UBO binding mask (distinct from hardware slots). */
   uint32_t apple9_ubo_mask;
   struct agx_apple9_vs_prolog_iface apple9_vs_prolog;
   struct agx_apple9_fs_epilog_iface apple9_fs_epilog;
   struct agx_apple9_varying_layout apple9_varyings;
   struct agx_apple9_interp_mask apple9_linear_mask, apple9_flat_mask;
   /* 16-bit word offsets (byte_offset / 2) of user-varying VARY_STORE
    * instructions in a vertex shader binary, for fast link-time group
    * remapping without recompilation. */
   uint16_t apple9_vary_relocs[AGX_APPLE9_MAX_VARY_RELOCS];
   uint16_t apple9_vary_reloc_count;
   bool apple9_vary_relocs_valid;
   uint16_t apple9_publication_count;
   /* Per-thread frame bytes for Apple9 atomics that return a value. */
   uint8_t apple9_atomic_frame_size;
   /* Apple9 setup code is entered by the launcher, independently of USC
    * preshader configuration used by earlier generations. */
   uint32_t apple9_preamble_offset, apple9_preamble_size;
   /* Native task/mesh code requiring a native entry/launch contract.
    * Completion may already be lowered. Generic upload must reject it. */
   bool apple9_body_only;
   bool apple9_reads_z;
   bool apple9_reads_primitive_id;
   bool apple9_reads_tile;
   bool apple9_writes_point_size;
   uint8_t apple9_clip_distance_count;
   bool apple9_reads_point_coord;
   /* Barycentric weights (bit 0 perspective, bit 1 linear) and the varying
    * scalars read per vertex, in the order of their coefficients. Both follow
    * the point coordinates.
    */
   uint8_t apple9_barycentric_mask;
   uint8_t apple9_per_vertex_count;
   uint8_t apple9_per_vertex[AGX_APPLE9_MAX_PER_VERTEX];
   /* Ordered graphics table entries: API UBOs 0..31, vertex bindings 32..63. */
   uint32_t apple9_resource_ssbo_mask, apple9_resource_write_mask;
   uint8_t apple9_resource_count;
   uint8_t apple9_resource_binding[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   /* Live API bindings; descriptor tables compact each mask independently. */
   uint32_t apple9_texture_mask, apple9_sampler_mask, apple9_image_mask;
   bool apple9_uses_texel_fetch;
   bool apple9_uses_discard;
   /* Fragment shader with pixel_interlock_{ordered,unordered}. */
   bool apple9_pixel_interlock;

   union agx_varyings varyings;

   /* Number of uniforms */
   uint16_t push_count;

   /* Local memory allocation in bytes */
   uint16_t local_size;

   /* Local imageblock allocation in bytes per thread */
   uint16_t imageblock_stride;

   /* Scratch memory allocation in bytes for main/preamble respectively */
   unsigned scratch_size, preamble_scratch_size;

   /* Size in bytes of the main sahder */
   unsigned main_size;

   /* Does the shader have a preamble? If so, it is at offset preamble_offset.
    * The main shader is at offset main_offset. The preamble is executed first.
    */
   bool has_preamble;
   unsigned preamble_offset, main_offset;

   /* Does the shader read the tilebuffer? */
   bool reads_tib;

   /* Does the shader require early fragment tests? */
   bool early_fragment_tests;

   /* Does the shader potentially draw to a nonzero viewport? */
   bool nonzero_viewport;

   /* Does the shader write layer and/or viewport index? Written together */
   bool writes_layer_viewport;

   /* Does the shader control the sample mask? */
   bool writes_sample_mask;

   /* Depth layout, never equal to NONE */
   enum gl_frag_depth_layout depth_layout;

   /* Based only the compiled shader, should tag writes be disabled? This is set
    * based on what is outputted. Note if rasterizer discard is used, that needs
    * to disable tag writes regardless of this flag.
    */
   bool tag_write_disable;

   /* Shader is incompatible with triangle merging */
   bool disable_tri_merging;

   /* Reads draw ID system value */
   bool uses_draw_id;

   /* Reads base vertex/instance */
   bool uses_base_param;

   /* Uses txf and hence needs a txf sampler mapped */
   bool uses_txf;

   /* Potentially uses the sampler heap (conservative) */
   bool uses_sampler_heap;

   /* Number of texture/sampler state registers pushed by the preamble. */
   uint8_t texture_state_count, sampler_state_count;

   /* Number of 16-bit registers used by the main shader and preamble
    * respectively.
    */
   uint16_t nr_gprs, nr_preamble_gprs;

   /* Output mask set during driver lowering */
   uint64_t outputs;

   /* Workgroup size */
   uint16_t workgroup_size[3];

   /* There may be constants in the binary. The driver must map these to uniform
    * registers as specified hre.
    */
   struct agx_rodata rodata;

   struct agx2_stats stats;
};

struct agx_shader_part {
   struct agx_shader_info info;
   void *binary;
};

static inline bool
agx_is_shader_empty(struct agx_shader_part *s)
{
   /* Last instruction is a stop, so if there's one instruction, there is
    * nothing but a stop. The shader is thus empty.
    */
   return (s->info.stats.instrs == 1);
}

#define AGX_MAX_RTS (8)

void agx_preprocess_nir(nir_shader *nir);
bool agx_nir_lower_printf_buffer(nir_shader *nir);
bool agx_nir_lower_discard_zs_emit(nir_shader *s);
bool agx_nir_lower_sample_mask(nir_shader *s);

bool agx_nir_lower_cull_distance_vs(struct nir_shader *s);
bool agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                          unsigned bit_size, unsigned num_components,
                          int64_t hole_size, nir_intrinsic_instr *low,
                          nir_intrinsic_instr *high, void *data);

struct agx_occupancy {
   unsigned max_registers;
   unsigned max_threads;
};

struct agx_occupancy agx_occupancy_for_register_count(unsigned halfregs);

static const nir_shader_compiler_options agx_nir_options = {
   .lower_fdiv = true,
   .float_mul_add16 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .float_mul_add32 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fpow = true,
   .lower_fmod = true,
   .lower_bitfield_extract8 = true,
   .lower_bitfield_extract16 = true,
   .lower_bitfield_insert = true,
   .lower_ifind_msb = true,
   .lower_find_lsb = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_fisnormal = true,
   .lower_scmp = true,
   .lower_isign = true,
   .lower_fsign = true,
   .lower_iabs = true,
   .lower_fminmax_signed_zero = true,
   .lower_fdph = true,
   .lower_ffract = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_64_2x32 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_extract_byte = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .has_cs_global_id = true,
   .lower_device_index_to_zero = true,
   .lower_hadd = true,
   .has_amul = true,
   .has_isub = true,
   .txl_lod_from_zero_gradients_is_neg_inf = true,
   .ssbo_readonly_as_constant_offset = true,
   .has_load_global_bounded = true,
   .support_16bit_alu = true,
   .max_unroll_iterations = 32,
   .lower_uniforms_to_ubo = true,
   .late_lower_int64 = true,
   .lower_int64_options =
      (nir_lower_int64_options) ~(nir_lower_iadd64 | nir_lower_imul_2x32_64),
   .lower_doubles_options = (nir_lower_doubles_options)(~0),
   .support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                              BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                              BITFIELD_BIT(MESA_SHADER_FRAGMENT),
   .support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_STAGES),
   .max_samples = 4,
   .lower_fquantize2f16 = true,
   .compact_arrays = true,
   .discard_is_demote = true,
   .scalarize_ddx = true,
   .io_options = nir_io_always_interpolate_convergent_fs_inputs,
};

/*
 * Return the standard sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static inline uint32_t
agx_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

bool agx_nir_lower_monolithic_msaa(nir_shader *shader, uint8_t nr_samples);
bool agx_nir_lower_sample_intrinsics(nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);
bool agx_nir_lower_to_per_sample(nir_shader *shader);
