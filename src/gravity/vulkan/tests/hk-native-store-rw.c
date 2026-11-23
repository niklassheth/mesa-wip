/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/nir/nir_builder.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "hk_shader_priv.h"
#include "compiler/nir/nir_builder.h"
#include "gravity/compiler/agx_compile_apple9.h"
#include "vk_alloc.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"

/* Stores stay native only when no invocation can read its own native write:
 * every read before every store, no loop holding both, no calls.
 * Also verify branchless HW OOB clipping on native IMAGE_WRITE / TEXTURE_LOAD
 * / TEXTURE_LOAD_MS / texture_buffer, vertex-stage native + fenced RW, and
 * tightened GPL HK_VS_VARIANT_SW compilation in hk_shader_compile.c. */

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: failed: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static void
image_op(nir_builder *b, nir_intrinsic_op op)
{
   nir_intrinsic_instr *intr = nir_intrinsic_instr_create(b->shader, op);
   const nir_intrinsic_info *info = &nir_intrinsic_infos[op];
   intr->num_components = 4;
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_format(intr, PIPE_FORMAT_R32_FLOAT);
   for (unsigned j = 0; j < info->num_srcs; ++j) {
      unsigned components = info->src_components[j] ? info->src_components[j] : 4;
      intr->src[j] = nir_src_for_ssa(nir_imm_zero(b, components, j == 0 ? 64 : 32));
   }
   if (info->has_dest)
      nir_def_init(&intr->instr, &intr->def, 4, 32);
   nir_builder_instr_insert(b, &intr->instr);
}

#define LOAD nir_intrinsic_bindless_image_load
#define STORE nir_intrinsic_bindless_image_store

static nir_builder
shader(void)
{
   static const nir_shader_compiler_options options = {0};
   return nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "rw");
}

struct nir_stats {
   unsigned ifs;
   unsigned native_stores;
   unsigned native_store_array;
   unsigned native_store_16bit;
   unsigned fence_image_write;
   unsigned fence_tex_invalidate;
   unsigned txf_2d;
   unsigned txf_ms;
   unsigned txf_inside_nested_if;
   unsigned umin_ffff;
   unsigned global_stores;
   unsigned rw_valid_checks;
};

static void
collect_cf_stats(struct exec_list *cf_list, unsigned if_depth,
                 struct nir_stats *st)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               switch (intr->intrinsic) {
               case nir_intrinsic_image_store_apple9_agx:
                  st->native_stores++;
                  if (nir_intrinsic_image_array(intr))
                     st->native_store_array++;
                  if (intr->src[2].ssa->bit_size == 16)
                     st->native_store_16bit++;
                  break;
               case nir_intrinsic_fence_image_write_apple9_agx:
                  st->fence_image_write++;
                  break;
               case nir_intrinsic_fence_tex_invalidate_apple9_agx:
                  st->fence_tex_invalidate++;
                  break;
               case nir_intrinsic_store_global:
                  st->global_stores++;
                  break;
               default:
                  break;
               }
            } else if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (tex->op == nir_texop_txf &&
                   tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
                   (tex->backend_flags & AGX_APPLE9_TEXTURE_FETCH_BOUNDED)) {
                  st->txf_2d++;
                  if (if_depth > 1)
                     st->txf_inside_nested_if++;
               } else if (tex->op == nir_texop_txf_ms &&
                          tex->sampler_dim == GLSL_SAMPLER_DIM_MS &&
                          (tex->backend_flags & AGX_APPLE9_TEXTURE_FETCH_BOUNDED)) {
                  st->txf_ms++;
                  if (if_depth > 1)
                     st->txf_inside_nested_if++;
               }
            } else if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_umin) {
                  for (unsigned s = 0; s < 2; ++s) {
                     if (nir_src_is_const(alu->src[s].src) &&
                         nir_src_as_uint(alu->src[s].src) == 0xffffu)
                        st->umin_ffff++;
                  }
               } else if (alu->op == nir_op_iand) {
                  const uint32_t rw_mask = HK_APPLE9_STORAGE_WRITE_VALID |
                                           HK_APPLE9_STORAGE_READ_VALID;
                  for (unsigned s = 0; s < 2; ++s) {
                     if (nir_src_is_const(alu->src[s].src) &&
                         nir_src_as_uint(alu->src[s].src) == rw_mask)
                        st->rw_valid_checks++;
                  }
               }
            }
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         st->ifs++;
         collect_cf_stats(&nif->then_list, if_depth + 1, st);
         collect_cf_stats(&nif->else_list, if_depth + 1, st);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_cf_stats(&loop->body, if_depth, st);
         break;
      }
      default:
         break;
      }
   }
}

static struct nir_stats
gather_stats(nir_shader *nir)
{
   struct nir_stats st = {0};
   collect_cf_stats(&nir_shader_get_entrypoint(nir)->body, 0, &st);
   return st;
}

static struct hk_descriptor_set_layout *
make_set_layout(struct hk_device *dev, VkDescriptorType type)
{
   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   if (!layout)
      return NULL;
   vk_object_base_init(&dev->vk, &layout->vk.base,
                       VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
   layout->binding_count = 1;
   unsigned stride = type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
      ? sizeof(struct hk_apple9_texel_descriptor)
      : sizeof(struct hk_apple9_storage_descriptor);
   layout->non_variable_descriptor_buffer_size = stride;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = type,
      .array_size = 1,
      .stride = stride,
      .offset = 0,
   };
   return layout;
}

static void
destroy_set_layout(struct hk_descriptor_set_layout *layout)
{
   vk_object_base_finish(&layout->vk.base);
   free(layout);
}

static int
compile_lowered_compute(nir_shader *nir)
{
   unsigned root = 0;
   CHECK(hk_apple9_texture_bindings(nir));
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_runtime_sampler_state(nir, &reason));
   nir_shader_intrinsics_pass(nir, hk_apple9_lower_roots,
                              nir_metadata_control_flow, &root);
   struct agx_shader_part out = {0};
   struct agx_apple9_compute_profile profile = {0};
   bool ok = agx_compile_apple9_tiny(nir, &out, &profile, &reason);
   if (!ok)
      fprintf(stderr, "compute compile failed: %s\n", reason ?: "unknown");
   free(out.binary);
   CHECK(ok);
   return 0;
}

static int
compile_lowered_vertex(nir_shader *nir)
{
   CHECK(hk_apple9_texture_bindings(nir));
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_runtime_sampler_state(nir, &reason));
   nir_shader_intrinsics_pass(nir, hk_apple9_lower_roots,
                              nir_metadata_control_flow, NULL);
   struct agx_apple9_vertex_layout layout = {
      .draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING,
      .buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING,
      .dynamic_stride = true,
   };
   struct agx_shader_part out = {0};
   bool ok = agx_compile_apple9_vertex_inputs(nir, &layout, &out, &reason);
   if (!ok)
      fprintf(stderr, "vertex compile failed: %s\n", reason ?: "unknown");
   free(out.binary);
   CHECK(ok);
   return 0;
}

static int
compile_lowered_fragment(nir_shader *nir, bool use_epilog)
{
   CHECK(hk_apple9_texture_bindings(nir));
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_runtime_sampler_state(nir, &reason));
   nir_shader_intrinsics_pass(nir, hk_apple9_lower_roots,
                              nir_metadata_control_flow, NULL);
   struct agx_apple9_blend blend = {
      .use_epilog = use_epilog,
      .dynamic_samples = use_epilog,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .samples = 1,
      .colormask = 0xf,
      .rgb_src = PIPE_BLENDFACTOR_ONE,
      .rgb_dst = PIPE_BLENDFACTOR_ZERO,
      .alpha_src = PIPE_BLENDFACTOR_ONE,
      .alpha_dst = PIPE_BLENDFACTOR_ZERO,
      .rgb_func = PIPE_BLEND_ADD,
      .alpha_func = PIPE_BLEND_ADD,
   };
   struct agx_shader_part out = {0};
   bool ok = agx_compile_apple9_fragment_mrt(nir, NULL, &blend, 1, &out, &reason);
   if (!ok)
      fprintf(stderr, "fragment compile failed: %s\n", reason ?: "unknown");
   if (ok) {
      const uint8_t *bytes = out.binary;
      const uint8_t fs_fence[] = {0x07, 0x14, 0x64, 0x50, 0x06, 0x00};
      const uint8_t tex_inv[] = {0x07, 0x04, 0x54, 0xd0, 0x06, 0x00};
      const uint8_t tile_600[] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
      bool has_fs_fence = false, has_tex_inv = false, has_600 = false;
      for (size_t i = 0; i + 6 <= out.info.binary_size; i += 2) {
         if (!memcmp(bytes + i, fs_fence, 6))
            has_fs_fence = true;
         if (!memcmp(bytes + i, tex_inv, 6))
            has_tex_inv = true;
         if (!memcmp(bytes + i, tile_600, 6))
            has_600 = true;
      }
      CHECK(has_fs_fence && has_tex_inv && has_600);
      if (use_epilog)
         CHECK(out.info.apple9_fs_epilog.has_tile_preamble);
   }
   free(out.binary);
   CHECK(ok);
   return 0;
}

static nir_shader *
build_storage_shader_bits(mesa_shader_stage stage, enum glsl_sampler_dim dim,
                          bool array, enum pipe_format format,
                          nir_alu_type src_type, bool do_store, bool do_load)
{
   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                  "storage_test");
   b.shader->info.internal = false;
   if (stage == MESA_SHADER_COMPUTE) {
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   }
   nir_def *gid = stage == MESA_SHADER_COMPUTE
      ? nir_load_global_invocation_id(&b, 32)
      : stage == MESA_SHADER_VERTEX
         ? nir_vec3(&b, nir_load_vertex_id(&b), nir_load_instance_id(&b),
                    nir_imm_int(&b, 0))
         : nir_vec3(&b, nir_u2u32(&b, nir_channel(&b, nir_load_pixel_coord(&b), 0)),
                    nir_u2u32(&b, nir_channel(&b, nir_load_pixel_coord(&b), 1)),
                    nir_imm_int(&b, 0));
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1),
                             nir_channel(&b, gid, 2), nir_imm_int(&b, 0));
   nir_def *sample = nir_iand_imm(&b, nir_channel(&b, gid, 0), 3);
   nir_def *handle = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   unsigned bits = nir_alu_type_get_type_size(src_type);
   nir_alu_type base = nir_alu_type_get_base_type(src_type);
   nir_def *val;
   if (bits == 16) {
      val = base == nir_type_float
         ? nir_vec4(&b, nir_imm_float16(&b, 1.0f), nir_imm_float16(&b, 2.0f),
                    nir_imm_float16(&b, 3.0f), nir_imm_float16(&b, 4.0f))
         : nir_vec4(&b, nir_imm_intN_t(&b, 1, 16), nir_imm_intN_t(&b, 2, 16),
                    nir_imm_intN_t(&b, 3, 16), nir_imm_intN_t(&b, 4, 16));
   } else {
      val = nir_imm_vec4(&b, 1.0f, 2.0f, 3.0f, 4.0f);
   }

   if (do_store) {
      nir_bindless_image_store(&b, handle, coord, sample, val, nir_imm_int(&b, 0),
                               .image_dim = dim, .image_array = array,
                               .format = format, .src_type = src_type);
      b.shader->info.writes_memory = true;
   }
   if (do_load) {
      nir_def *loaded = nir_bindless_image_load(&b, 4, 32, handle, coord, sample,
                                                nir_imm_int(&b, 0),
                                                .image_dim = dim,
                                                .image_array = array,
                                                .format = format,
                                                .dest_type = nir_type_float32);
      if (stage == MESA_SHADER_COMPUTE) {
         nir_store_global(&b, loaded, nir_imm_int64(&b, 0x10000),
                          .align_mul = 16, .write_mask = 0xf);
      } else if (stage == MESA_SHADER_VERTEX) {
         nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out,
                                                 glsl_vec4_type(), "pos");
         pos->data.location = VARYING_SLOT_POS;
         nir_store_var(&b, pos, loaded, 0xf);
      } else if (stage == MESA_SHADER_FRAGMENT) {
         nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                                 glsl_vec4_type(), "color");
         out->data.location = FRAG_RESULT_DATA0;
         nir_store_var(&b, out, loaded, 0xf);
      }
   } else if (stage == MESA_SHADER_VERTEX) {
      nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out,
                                              glsl_vec4_type(), "pos");
      pos->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, pos, nir_imm_vec4(&b, 0, 0, 0, 1), 0xf);
   } else if (stage == MESA_SHADER_FRAGMENT) {
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                              glsl_vec4_type(), "color");
      out->data.location = FRAG_RESULT_DATA0;
      nir_store_var(&b, out, nir_imm_vec4(&b, 0, 1, 0, 1), 0xf);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
build_storage_shader(mesa_shader_stage stage, enum glsl_sampler_dim dim,
                     bool array, enum pipe_format format,
                     bool do_store, bool do_load)
{
   return build_storage_shader_bits(stage, dim, array, format, nir_type_float32,
                                    do_store, do_load);
}

static int
test_branchless_storage_and_tbuf(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;

   struct hk_descriptor_set_layout *img_layout =
      make_set_layout(dev, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
   struct hk_descriptor_set_layout *tbuf_layout =
      make_set_layout(dev, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER);
   CHECK(img_layout && tbuf_layout);
   struct vk_descriptor_set_layout *img_sets[] = {&img_layout->vk};
   struct vk_descriptor_set_layout *tbuf_sets[] = {&tbuf_layout->vk};
   const struct vk_pipeline_robustness_state rs = vk_robustness_disabled;

   /* 1. Compute 2D typed & formatless IMAGE_WRITE: 0 branches, umin(..., 0xffff). */
   const enum pipe_format fmts[] = {PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_NONE};
   for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             fmts[i], true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_array == 0);
      CHECK(st.ifs == 0);
      CHECK(st.umin_ffff >= 2);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 2. Compute 2D-array formatless IMAGE_WRITE: 0 branches, layer cmpsel. */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, true,
                                             PIPE_FORMAT_NONE, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_array == 1);
      CHECK(st.ifs == 0);
      CHECK(st.umin_ffff >= 2);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 3. Compute formatless storage texel buffer IMAGE_WRITE: 0 branches. */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_BUF, false,
                                             PIPE_FORMAT_NONE, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.ifs == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 4. Compute 2D & 2D-MS storage loads: branchless TEXTURE_LOAD / TEXTURE_LOAD_MS
    *    inside the single uniform if (READ_VALID), with 0 nested bounds branches. */
   for (unsigned ms = 0; ms < 2; ++ms) {
      enum glsl_sampler_dim dim = ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE, dim, false,
                                             PIPE_FORMAT_R32G32B32A32_FLOAT,
                                             false, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(ms ? st.txf_ms == 1 : st.txf_2d == 1);
      CHECK(st.txf_inside_nested_if == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 5. All image dimensions (1D, 1D-array, 2D, 2D-array, 3D, Cube, Cube-array,
    *    2D-MS, 2D-MS-array): branchless TEXTURE_LOAD / TEXTURE_LOAD_MS and
    *    branchless native IMAGE_WRITE compile cleanly end-to-end. */
   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
      bool can_store;
   } dims[] = {
      {GLSL_SAMPLER_DIM_1D, false, true},
      {GLSL_SAMPLER_DIM_1D, true, true},
      {GLSL_SAMPLER_DIM_2D, false, true},
      {GLSL_SAMPLER_DIM_2D, true, true},
      {GLSL_SAMPLER_DIM_3D, false, true},
      {GLSL_SAMPLER_DIM_CUBE, false, true},
      {GLSL_SAMPLER_DIM_CUBE, true, true},
      {GLSL_SAMPLER_DIM_MS, false, false},
      {GLSL_SAMPLER_DIM_MS, true, false},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(dims); ++i) {
      nir_shader *load_nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                                  dims[i].dim, dims[i].array,
                                                  PIPE_FORMAT_NONE, false, true);
      hk_lower_nir(dev, load_nir, &rs, false, 1, img_sets, 0);
      struct nir_stats lst = gather_stats(load_nir);
      CHECK(dims[i].dim == GLSL_SAMPLER_DIM_MS ? lst.txf_ms == 1 : lst.txf_2d == 1 || dims[i].dim == GLSL_SAMPLER_DIM_3D);
      CHECK(lst.txf_inside_nested_if == 0);
      CHECK(compile_lowered_compute(load_nir) == 0);
      ralloc_free(load_nir);

      if (dims[i].can_store) {
         nir_shader *store_nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                                      dims[i].dim, dims[i].array,
                                                      PIPE_FORMAT_NONE, true, false);
         hk_lower_nir(dev, store_nir, &rs, false, 1, img_sets, 0);
         struct nir_stats sst = gather_stats(store_nir);
         CHECK(sst.native_stores == 1);
         if (dims[i].dim != GLSL_SAMPLER_DIM_3D)
            CHECK(sst.ifs == 0);
         CHECK(compile_lowered_compute(store_nir) == 0);
         ralloc_free(store_nir);
      }
   }

   /* 6. Uniform texel buffer fetch: branchless sentinel 2D txf in the else of
    *    uniform if (size == 12), with 0 nested bounds branches around txf. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "tbuf");
      b.shader->info.internal = false;
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *handle = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_txf;
      tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 1;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, idx);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x10000),
                       .align_mul = 16, .write_mask = 0xf);
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      hk_lower_nir(dev, b.shader, &rs, false, 1, tbuf_sets, 0);
      struct nir_stats st = gather_stats(b.shader);
      CHECK(st.txf_2d == 1 && st.txf_inside_nested_if == 0);
      CHECK(compile_lowered_compute(b.shader) == 0);
      ralloc_free(b.shader);
   }

   /* 7. Vertex stage ordered native IMAGE_WRITE: 0 branches, compiles via HW VS. */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_VERTEX,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             PIPE_FORMAT_NONE, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.ifs == 0);
      CHECK(compile_lowered_vertex(nir) == 0);
      ralloc_free(nir);
   }

   /* 8. Vertex stage fenced RW (store -> load): native IMAGE_WRITE + 0x0650 +
    *    0x06d0 + TEXTURE_LOAD, compiles cleanly via HW VS. */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_VERTEX,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             PIPE_FORMAT_NONE, true, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1);
      CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
      CHECK(st.txf_2d == 1 && st.txf_inside_nested_if == 0);
      CHECK(compile_lowered_vertex(nir) == 0);
      ralloc_free(nir);
   }

   /* 9. Fragment stage fenced RW (store -> load): native FS IMAGE_WRITE +
    *    AGX_APPLE9_FENCE_IMAGE_WRITE_FS (0x64 50 06) + 0x06d0 + TEXTURE_LOAD +
    *    lazy TILE_ACCESS 0x600 (both monolithic and epilog). */
   for (unsigned ep = 0; ep < 2; ++ep) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_FRAGMENT,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             PIPE_FORMAT_NONE, true, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1);
      CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
      CHECK(st.txf_2d == 1 && st.txf_inside_nested_if == 0);
      CHECK(compile_lowered_fragment(nir, ep != 0) == 0);
      ralloc_free(nir);
   }

   /* 10. Native 16-bit (f16, u16, i16) IMAGE_WRITE stays 16-bit in NIR and
    *     compiles end-to-end across 2D, 2D-array, and texel buffers. */
   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
      nir_alu_type type;
   } cases_16[] = {
      {GLSL_SAMPLER_DIM_2D, false, nir_type_float16},
      {GLSL_SAMPLER_DIM_2D, false, nir_type_uint16},
      {GLSL_SAMPLER_DIM_2D, false, nir_type_int16},
      {GLSL_SAMPLER_DIM_2D, true, nir_type_float16},
      {GLSL_SAMPLER_DIM_BUF, false, nir_type_float16},
      {GLSL_SAMPLER_DIM_BUF, false, nir_type_uint16},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases_16); ++i) {
      nir_shader *nir = build_storage_shader_bits(
         MESA_SHADER_COMPUTE, cases_16[i].dim, cases_16[i].array,
         PIPE_FORMAT_NONE, cases_16[i].type, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_16bit == 1);
      CHECK(st.ifs == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 11. Fenced RW compute 2D stores (typed & formatless) must require BOTH
    *     WRITE_VALID and READ_VALID so views whose reads stay global (e.g.
    *     padded 2D block-texel aliases with READ_VALID == 0) fall back to
    *     coherent global stores rather than mixing PBE writes with L1 loads. */
   for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             fmts[i], true, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1);
      CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
      CHECK(st.rw_valid_checks >= 1);
      CHECK(st.global_stores >= 2); /* fallback image store + final result store */
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   destroy_set_layout(img_layout);
   destroy_set_layout(tbuf_layout);
   free(dev);
   puts("branchless HW OOB & VS native/fenced storage lowering: all cases pass");
   return 0;
}

static nir_shader *
build_simple_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  &agx_nir_options, "gpl_vs");
   b.shader->info.io_lowered = true;
   nir_store_output(&b, nir_imm_vec4(&b, 0.0f, 0.0f, 0.5f, 1.0f),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static int
check_vs_sw_variant(struct hk_device *dev,
                    const struct vk_graphics_pipeline_state *state,
                    VkShaderStageFlags next_stage_mask,
                    bool expect_sw)
{
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_VERTEX,
      .next_stage_mask = next_stage_mask,
      .nir = build_simple_vs(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *out = NULL;
   bool native_rejected = false;
   VkResult res = hk_apple9_compile_graphics_path(dev, 1, &info, state, NULL,
                                                  NULL, NULL, &out, false,
                                                  &native_rejected);
   CHECK(res == VK_SUCCESS && out != NULL);
   struct hk_api_shader *api = container_of(out, struct hk_api_shader, vk);
   bool has_hw = api->variants[HK_VS_VARIANT_HW].bo != NULL;
   bool has_sw = api->variants[HK_VS_VARIANT_SW].bo != NULL;
   hk_api_shader_destroy(&dev->vk, out, NULL);
   CHECK(has_hw);
   CHECK(has_sw == expect_sw);
   return 0;
}

static int
test_gpl_sw_needed(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = UINT64_C(0x1000000000);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);

   /* Pre-seed one host-backed shader body slab so hk_apple9_body_alloc serves
    * compiled VS binaries without calling agx_bo_create / DRM ioctls. */
   struct hk_apple9_body_slab *slab = calloc(1, sizeof(*slab));
   struct agx_bo *slab_bo = calloc(1, sizeof(*slab_bo));
   struct agx_va *slab_va = calloc(1, sizeof(*slab_va));
   void *slab_mem = calloc(1, HK_APPLE9_BODY_SLAB_SIZE);
   CHECK(slab && slab_bo && slab_va && slab_mem);
   slab_va->addr = dev->dev.shader_base + 0x100000;
   slab_bo->dev = &dev->dev;
   slab_bo->va = slab_va;
   slab_bo->_map = slab_mem;
   slab_bo->size = HK_APPLE9_BODY_SLAB_SIZE;
   slab_bo->refcnt = 1;
   slab->bo = slab_bo;
   slab->live = 1; /* Keep slab alive across hk_api_shader_destroy. */
   util_vma_heap_init(&slab->heap, HK_APPLE9_BODY_ALIGNMENT,
                      HK_APPLE9_BODY_SLAB_USABLE);
   list_addtail(&slab->link, &dev->apple9_bodies.slabs);

   const struct vk_rasterization_state rs_fill = {
      .polygon_mode = VK_POLYGON_MODE_FILL,
   };
   const struct vk_rasterization_state rs_point = {
      .polygon_mode = VK_POLYGON_MODE_POINT,
   };
   const struct vk_rasterization_state rs_stipple = {
      .polygon_mode = VK_POLYGON_MODE_FILL,
      .line.stipple.enable = true,
   };

   /* 1. Normal DXVK/CS2-style GPL VS (LIBRARY_BIT, FILL, next=FRAGMENT, even
    *    with transformFeedback enabled): must NOT waste a SW-VS compile. */
   dev->vk.enabled_features = (struct vk_features){.transformFeedback = true};
   struct vk_graphics_pipeline_state gpl_normal = {
      .pipeline_flags = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
      .rs = &rs_fill,
   };
   CHECK(check_vs_sw_variant(dev, &gpl_normal, VK_SHADER_STAGE_FRAGMENT_BIT, false) == 0);

   /* 2. Monolithic depth-only VS (!LIBRARY_BIT, !has_fragment, even with
    *    fragmentShaderBarycentric enabled): cannot link an FS, skip SW-VS. */
   dev->vk.enabled_features = (struct vk_features){.fragmentShaderBarycentric = true};
   struct vk_graphics_pipeline_state mono_depth = {.rs = &rs_fill};
   CHECK(check_vs_sw_variant(dev, &mono_depth, VK_SHADER_STAGE_FRAGMENT_BIT, false) == 0);

   /* 3. GPL VS with fragmentShaderBarycentric enabled (CTS fast_linked_library):
    *    MUST compile SW-VS for potential passthrough GS copies. */
   CHECK(check_vs_sw_variant(dev, &gpl_normal, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);

   /* 4. GPL VS with primitivesGeneratedQuery enabled: MUST compile SW-VS. */
   dev->vk.enabled_features = (struct vk_features){.primitivesGeneratedQuery = true};
   CHECK(check_vs_sw_variant(dev, &gpl_normal, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);

   /* 5. GPL VS with dynamic polygon mode or static POINT: MUST compile SW-VS. */
   dev->vk.enabled_features = (struct vk_features){0};
   struct vk_graphics_pipeline_state gpl_dyn_poly = gpl_normal;
   BITSET_SET(gpl_dyn_poly.dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);
   CHECK(check_vs_sw_variant(dev, &gpl_dyn_poly, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);
   struct vk_graphics_pipeline_state gpl_static_point = {
      .pipeline_flags = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
      .rs = &rs_point,
   };
   CHECK(check_vs_sw_variant(dev, &gpl_static_point, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);

   /* 6. GPL VS with dynamic or static line stipple: MUST compile SW-VS. */
   struct vk_graphics_pipeline_state gpl_dyn_stipple = gpl_normal;
   BITSET_SET(gpl_dyn_stipple.dynamic, MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE);
   CHECK(check_vs_sw_variant(dev, &gpl_dyn_stipple, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);
   struct vk_graphics_pipeline_state gpl_static_stipple = {
      .pipeline_flags = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
      .rs = &rs_stipple,
   };
   CHECK(check_vs_sw_variant(dev, &gpl_static_stipple, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);

   /* 7. GPL VS feeding GS or unknown next_stage_mask with geometryShader: SW-VS. */
   CHECK(check_vs_sw_variant(dev, &gpl_normal, VK_SHADER_STAGE_GEOMETRY_BIT, true) == 0);
   dev->vk.enabled_features = (struct vk_features){.geometryShader = true};
   CHECK(check_vs_sw_variant(dev, &gpl_normal, 0, true) == 0);

   /* 8. ESO VS (state == NULL, e.g. Zink SSO): MUST compile SW-VS. */
   dev->vk.enabled_features = (struct vk_features){0};
   CHECK(check_vs_sw_variant(dev, NULL, VK_SHADER_STAGE_FRAGMENT_BIT, true) == 0);

   hk_apple9_bodies_finish(dev);
   util_vma_heap_finish(&slab->heap);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   free(slab_mem);
   free(slab_va);
   free(slab_bo);
   free(slab);
   free(dev);
   puts("GPL sw_needed tightening: all 10 VS compile cases pass");
   return 0;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   /* Read, then store: native. */
   nir_builder b = shader();
   image_op(&b, LOAD);
   image_op(&b, STORE);
   CHECK(hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Store, then read: the read could observe the native write. */
   b = shader();
   image_op(&b, STORE);
   image_op(&b, LOAD);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* An atomic after a store reads it too. */
   b = shader();
   image_op(&b, STORE);
   image_op(&b, nir_intrinsic_bindless_image_atomic);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Read before store inside one loop: the next iteration reads the write. */
   b = shader();
   nir_push_loop(&b);
   image_op(&b, LOAD);
   image_op(&b, STORE);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Reads in a loop that ends before the stores: native. */
   b = shader();
   nir_push_loop(&b);
   image_op(&b, LOAD);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   nir_push_loop(&b);
   image_op(&b, STORE);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   CHECK(hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Store in the then-branch, read in the else-branch: rejected (block order). */
   b = shader();
   nir_push_if(&b, nir_imm_true(&b));
   image_op(&b, STORE);
   nir_push_else(&b, NULL);
   image_op(&b, LOAD);
   nir_pop_if(&b, NULL);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* A second function: block order is no longer program order. */
   b = shader();
   image_op(&b, LOAD);
   image_op(&b, STORE);
   nir_function *callee = nir_function_create(b.shader, "callee");
   nir_function_impl_create(callee);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   puts("native storage rw rule: order, loops, branches and calls pass");

   CHECK(test_branchless_storage_and_tbuf() == 0);
   CHECK(test_gpl_sw_needed() == 0);

   glsl_type_singleton_decref();
   return 0;
}
