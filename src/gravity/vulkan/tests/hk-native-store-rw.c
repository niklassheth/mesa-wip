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

struct nir_stats {
   unsigned ifs;
   unsigned native_stores;
   unsigned native_store_array;
   unsigned native_store_16bit;
   unsigned native_store_coherent;
   unsigned native_loads;
   unsigned native_load_coherent;
   unsigned fence_image_write;
   unsigned fence_tex_invalidate;
   unsigned txf_2d;
   unsigned txf_3d;
   unsigned txf_ms;
   unsigned txf_inside_nested_if;
   unsigned global_stores;
   unsigned global_loads;
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
                  if (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE))
                     st->native_store_coherent++;
                  break;
               case nir_intrinsic_image_load_apple9_agx: {
                  st->native_loads++;
                  if (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE))
                     st->native_load_coherent++;
                  enum glsl_sampler_dim d = nir_intrinsic_image_dim(intr);
                  if (d == GLSL_SAMPLER_DIM_MS)
                     st->txf_ms++;
                  else if (d == GLSL_SAMPLER_DIM_3D)
                     st->txf_3d++;
                  else
                     st->txf_2d++;
                  if (if_depth > 1)
                     st->txf_inside_nested_if++;
                  break;
               }
               case nir_intrinsic_fence_image_write_apple9_agx:
                  st->fence_image_write++;
                  break;
               case nir_intrinsic_fence_tex_invalidate_apple9_agx:
                  st->fence_tex_invalidate++;
                  break;
               case nir_intrinsic_store_global:
                  st->global_stores++;
                  break;
               case nir_intrinsic_load_global:
                  st->global_loads++;
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
      const uint8_t fs_fence[] = {0x07, 0x24, 0x64, 0x50, 0x06, 0x00};
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

   /* 1. Compute 2D typed & formatless IMAGE_WRITE: 0 branches, 0 global stores. */
   const enum pipe_format fmts[] = {PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_NONE};
   for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             fmts[i], true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_array == 0);
      CHECK(st.ifs == 0 && st.global_stores == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 2. Compute 2D-array formatless IMAGE_WRITE: 0 branches, 0 global stores. */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, true,
                                             PIPE_FORMAT_NONE, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_array == 1);
      CHECK(st.ifs == 0 && st.global_stores == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 3. Compute typed & formatless storage texel buffer load & store:
    *    100% native TEXTURE_LOAD + IMAGE_WRITE, 0 branches, 0 SW codec. */
   for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_BUF, false,
                                             fmts[i], true, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_loads == 1 && st.txf_2d == 1);
      CHECK(st.ifs == 0 && st.global_loads == 0 && st.global_stores == 1);
      CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 4. Compute 2D & 2D-MS storage loads: branchless image_load_apple9_agx
    *    with 0 branches and 0 global loads. */
   for (unsigned ms = 0; ms < 2; ++ms) {
      enum glsl_sampler_dim dim = ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE, dim, false,
                                             PIPE_FORMAT_R32G32B32A32_FLOAT,
                                             false, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(ms ? st.txf_ms == 1 : st.txf_2d == 1);
      CHECK(st.ifs == 0 && st.global_loads == 0 && st.txf_inside_nested_if == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 5. All image dimensions (1D, 1D-array, 2D, 2D-array, 3D, Cube, Cube-array,
    *    2D-MS, 2D-MS-array): branchless native TEXTURE_LOAD / TEXTURE_LOAD_MS
    *    and branchless native IMAGE_WRITE (including 2D-buffer MSAA PBE and
    *    3D linear-2D alias arm) compile cleanly end-to-end with 0 SW codec. */
   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
   } dims[] = {
      {GLSL_SAMPLER_DIM_1D, false},
      {GLSL_SAMPLER_DIM_1D, true},
      {GLSL_SAMPLER_DIM_2D, false},
      {GLSL_SAMPLER_DIM_2D, true},
      {GLSL_SAMPLER_DIM_3D, false},
      {GLSL_SAMPLER_DIM_CUBE, false},
      {GLSL_SAMPLER_DIM_CUBE, true},
      {GLSL_SAMPLER_DIM_MS, false},
      {GLSL_SAMPLER_DIM_MS, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(dims); ++i) {
      nir_shader *load_nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                                  dims[i].dim, dims[i].array,
                                                  PIPE_FORMAT_NONE, false, true);
      hk_lower_nir(dev, load_nir, &rs, false, 1, img_sets, 0);
      struct nir_stats lst = gather_stats(load_nir);
      if (dims[i].dim == GLSL_SAMPLER_DIM_3D) {
         CHECK(lst.txf_3d == 1 && lst.txf_2d == 1 && lst.ifs == 1);
      } else {
         CHECK(dims[i].dim == GLSL_SAMPLER_DIM_MS ? lst.txf_ms == 1 : lst.txf_2d == 1);
         CHECK(lst.ifs == 0);
      }
      CHECK(lst.global_loads == 0 && lst.txf_inside_nested_if == 0);
      CHECK(compile_lowered_compute(load_nir) == 0);
      ralloc_free(load_nir);

      nir_shader *store_nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                                   dims[i].dim, dims[i].array,
                                                   PIPE_FORMAT_NONE, true, false);
      hk_lower_nir(dev, store_nir, &rs, false, 1, img_sets, 0);
      struct nir_stats sst = gather_stats(store_nir);
      if (dims[i].dim == GLSL_SAMPLER_DIM_3D) {
         CHECK(sst.native_stores == 2 && sst.ifs == 1);
      } else {
         CHECK(sst.native_stores == 1 && sst.ifs == 0);
      }
      CHECK(sst.global_stores == 0);
      CHECK(compile_lowered_compute(store_nir) == 0);
      ralloc_free(store_nir);
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

   /* 11. Fenced RW compute 2D stores (typed & formatless): 100% branchless
    *     native PBE store + lazy fence pair + native TEXTURE_LOAD with 0
    *     software fallback branches or global fallback stores. */
   for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_2D, false,
                                             fmts[i], true, true);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_loads == 1);
      CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
      CHECK(st.ifs == 0 && st.global_loads == 0 && st.global_stores == 1);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 12. Typed 4-byte 3D store (e.g. R32_FLOAT) skips the linear-2D alias
    *     branch completely (0 ifs, 1 native IMAGE_WRITE_LAYERED). */
   {
      nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                             GLSL_SAMPLER_DIM_3D, false,
                                             PIPE_FORMAT_R32_FLOAT, true, false);
      hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
      struct nir_stats st = gather_stats(nir);
      CHECK(st.native_stores == 1 && st.native_store_array == 1);
      CHECK(st.ifs == 0 && st.global_stores == 0);
      CHECK(compile_lowered_compute(nir) == 0);
      ralloc_free(nir);
   }

   /* 13. MSAA 2D & 2D-array RW (typed & formatless): 100% native 2D-buffer
    *     PBE IMAGE_WRITE + lazy fence pair + native TEXTURE_LOAD_MS, 0 ifs. */
   for (unsigned arr = 0; arr < 2; ++arr) {
      for (unsigned i = 0; i < ARRAY_SIZE(fmts); ++i) {
         nir_shader *nir = build_storage_shader(MESA_SHADER_COMPUTE,
                                                GLSL_SAMPLER_DIM_MS, arr != 0,
                                                fmts[i], true, true);
         hk_lower_nir(dev, nir, &rs, false, 1, img_sets, 0);
         struct nir_stats st = gather_stats(nir);
         CHECK(st.native_stores == 1 && st.native_loads == 1 && st.txf_ms == 1);
         CHECK(st.fence_image_write == 1 && st.fence_tex_invalidate == 1);
         CHECK(st.ifs == 0 && st.global_loads == 0 && st.global_stores == 1);
         CHECK(compile_lowered_compute(nir) == 0);
         ralloc_free(nir);
      }
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

static int
test_soft_fault_ssbo_and_4b_vectorize(void)
{
   /* Verify 1) soft-fault load_global_bounded emits nir_load_global without
    * zero-page redirection (AGX_ZERO_PAGE_ADDRESS) and uses overflow-safe
    * usub_sat bounds checks, while volatile loads retain zero-page fallback;
    * 2) high constant UBO offsets (>255) share one usub_sat(bound, offs_imm)
    * across all vector components. */
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "soft_fault_ssbo");
   nir_def *base = nir_imm_int64(&b, 0x100000);
   nir_def *dyn_off = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *bound = nir_load_global_constant(&b, 1, 32, nir_imm_int64(&b, 0x20000),
                                             .align_mul = 4);
   nir_def *ssbo = nir_load_global_bounded(&b, 4, 32, base, dyn_off, bound,
                                           .align_mul = 4, .align_offset = 0);
   nir_def *ubo = nir_load_global_constant_bounded(&b, 4, 32, base,
                                                   nir_imm_int(&b, 512), bound,
                                                   .align_mul = 4, .align_offset = 0);
   nir_store_global(&b, nir_iadd(&b, ssbo, ubo), nir_imm_int64(&b, 0x80000),
                    .write_mask = 0xf, .align_mul = 4);

   CHECK(hk_lower_bounded_global_loads(b.shader, true, true));

   unsigned global_loads = 0, const_loads = 0, usub_sats = 0, bcsel_64 = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_global)
               global_loads++;
            else if (intr->intrinsic == nir_intrinsic_load_global_constant)
               const_loads++;
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_usub_sat)
               usub_sats++;
            else if (alu->op == nir_op_bcsel && alu->def.bit_size == 64)
               bcsel_64++;
         }
      }
   }
   CHECK(global_loads == 1);
   CHECK(const_loads == 2); /* bound load + UBO vec4 load */
   CHECK(bcsel_64 == 0);    /* no 64-bit zero-page base redirection */
   CHECK(usub_sats == 5);   /* 4 dynamic SSBO component limits + 1 shared UBO rem */
   ralloc_free(b.shader);

   /* Verify vectorizer robustness with soft faults enabled:
    * 1) Two scalar u32 load_global_bounded at x and iadd(x, 4) (potentially
    *    wrapping 32-bit x=0xfffffffc) MUST NOT merge because robust_modes
    *    includes nir_var_mem_global.
    * 2) Two adjacent 4B-aligned u32 load_global_bounded at constant offsets
    *    4 and 8 DO merge into a single vec2 load_global.
    * 3) Narrow <32-bit loads (u16 at byte 4 + u16vec3 at byte 6) MUST NOT be
    *    widened into 32-bit robustness units under the 4B relaxation. */
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;
   dev->dev.params.features = DRM_ASAHI_FEATURE_SOFT_FAULTS;
   const struct vk_pipeline_robustness_state rs = vk_robustness_disabled;

   /* Case A: potential 32-bit wrap at x and iadd(x, 4) stays unmerged. */
   nir_builder wb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "bounded_wrap_no_merge");
   nir_def *wbase = nir_imm_int64(&wb, 0x100000);
   nir_def *wx = nir_channel(&wb, nir_load_global_invocation_id(&wb, 32), 0);
   nir_def *wbound = nir_imm_int(&wb, 4);
   nir_def *w0 = nir_load_global_bounded(&wb, 1, 32, wbase, wx, wbound,
                                         .align_mul = 4, .align_offset = 0);
   nir_def *w1 = nir_load_global_bounded(&wb, 1, 32, wbase, nir_iadd_imm(&wb, wx, 4),
                                         wbound, .align_mul = 4, .align_offset = 0);
   nir_store_global(&wb, nir_vec2(&wb, w0, w1), nir_imm_int64(&wb, 0x80000),
                    .write_mask = 0x3, .align_mul = 4);
   nir_shader_gather_info(wb.shader, nir_shader_get_entrypoint(wb.shader));
   hk_lower_nir(dev, wb.shader, &rs, false, 0, NULL, 0);
   unsigned w_loads = 0, w_vec_loads = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(wb.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               w_loads++;
               if (intr->def.num_components > 1)
                  w_vec_loads++;
            }
         }
      }
   }
   CHECK(w_loads == 2 && w_vec_loads == 0);
   ralloc_free(wb.shader);

   /* Case B: safe 4B-aligned u32 pair at offsets 4 and 8 merges into vec2. */
   nir_builder mb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "bounded_4b_u32_merge");
   nir_def *mbase = nir_imm_int64(&mb, 0x100000);
   nir_def *mbound = nir_load_global_constant(&mb, 1, 32, nir_imm_int64(&mb, 0x20000),
                                              .align_mul = 4);
   nir_def *m0 = nir_load_global_bounded(&mb, 1, 32, mbase, nir_imm_int(&mb, 4),
                                         mbound, .align_mul = 4, .align_offset = 0);
   nir_def *m1 = nir_load_global_bounded(&mb, 1, 32, mbase, nir_imm_int(&mb, 8),
                                         mbound, .align_mul = 4, .align_offset = 0);
   nir_store_global(&mb, nir_vec2(&mb, m0, m1), nir_imm_int64(&mb, 0x80000),
                    .write_mask = 0x3, .align_mul = 4);
   nir_shader_gather_info(mb.shader, nir_shader_get_entrypoint(mb.shader));
   hk_lower_nir(dev, mb.shader, &rs, false, 0, NULL, 0);
   unsigned m_loads = 0, m_vec2 = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(mb.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               m_loads++;
               if (intr->def.num_components == 2 && intr->def.bit_size == 32)
                  m_vec2++;
            }
         }
      }
   }
   CHECK(m_loads == 1 && m_vec2 == 1);
   ralloc_free(mb.shader);

   /* Case C: narrow u16 at byte 4 + u16vec3 at byte 6 MUST NOT widen to 32-bit. */
   nir_builder nb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "bounded_u16_no_widen");
   nir_def *nbase = nir_imm_int64(&nb, 0x100000);
   nir_def *nbound = nir_imm_int(&nb, 6);
   nir_def *n0 = nir_load_global_bounded(&nb, 1, 16, nbase, nir_imm_int(&nb, 4),
                                         nbound, .align_mul = 4, .align_offset = 0);
   nir_def *n1 = nir_load_global_bounded(&nb, 3, 16, nbase, nir_imm_int(&nb, 6),
                                         nbound, .align_mul = 4, .align_offset = 2);
   nir_def *nvec = nir_vec4(&nb, n0, nir_channel(&nb, n1, 0),
                            nir_channel(&nb, n1, 1), nir_channel(&nb, n1, 2));
   nir_store_global(&nb, nvec, nir_imm_int64(&nb, 0x80000),
                    .write_mask = 0xf, .align_mul = 2);
   nir_shader_gather_info(nb.shader, nir_shader_get_entrypoint(nb.shader));
   hk_lower_nir(dev, nb.shader, &rs, false, 0, NULL, 0);
   unsigned n_loads32 = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(nb.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global &&
                intr->def.bit_size == 32)
               n_loads32++;
         }
      }
   }
   CHECK(n_loads32 == 0);
   ralloc_free(nb.shader);

   /* Case D (HIGH #2): naturally aligned narrow bounded loads MUST NOT widen:
    * u8vec3@0 + u8vec3@3 (bound=3) must stay 8-bit, and u32@0 + u16vec2@4
    * (bound=6) must not widen the u16vec2 into a 32-bit robustness unit. */
   nir_builder b8 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "bounded_u8vec3_no_widen");
   nir_def *b8_base = nir_imm_int64(&b8, 0x100000);
   nir_def *b8_bound = nir_imm_int(&b8, 3);
   nir_def *b8_0 = nir_load_global_bounded(&b8, 3, 8, b8_base, nir_imm_int(&b8, 0),
                                           b8_bound, .align_mul = 8, .align_offset = 0);
   nir_def *b8_1 = nir_load_global_bounded(&b8, 3, 8, b8_base, nir_imm_int(&b8, 3),
                                           b8_bound, .align_mul = 8, .align_offset = 3);
   nir_store_global(&b8, nir_iadd(&b8, b8_0, b8_1), nir_imm_int64(&b8, 0x80000),
                    .write_mask = 0x7, .align_mul = 1);
   nir_shader_gather_info(b8.shader, nir_shader_get_entrypoint(b8.shader));
   hk_lower_nir(dev, b8.shader, &rs, false, 0, NULL, 0);
   unsigned b8_widened = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(b8.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global &&
                intr->def.bit_size > 8)
               b8_widened++;
         }
      }
   }
   CHECK(b8_widened == 0);
   ralloc_free(b8.shader);

   nir_builder b32_16 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                       &agx_nir_options,
                                                       "bounded_u32_u16vec2_no_widen");
   nir_def *m32_base = nir_imm_int64(&b32_16, 0x100000);
   nir_def *m32_bound = nir_imm_int(&b32_16, 6);
   nir_def *m32_0 = nir_load_global_bounded(&b32_16, 1, 32, m32_base, nir_imm_int(&b32_16, 0),
                                            m32_bound, .align_mul = 8, .align_offset = 0);
   nir_def *m32_1 = nir_load_global_bounded(&b32_16, 2, 16, m32_base, nir_imm_int(&b32_16, 4),
                                            m32_bound, .align_mul = 8, .align_offset = 4);
   nir_def *m32_out = nir_vec3(&b32_16, m32_0,
                               nir_u2u32(&b32_16, nir_channel(&b32_16, m32_1, 0)),
                               nir_u2u32(&b32_16, nir_channel(&b32_16, m32_1, 1)));
   nir_store_global(&b32_16, m32_out, nir_imm_int64(&b32_16, 0x80000),
                    .write_mask = 0x7, .align_mul = 4);
   nir_shader_gather_info(b32_16.shader, nir_shader_get_entrypoint(b32_16.shader));
   hk_lower_nir(dev, b32_16.shader, &rs, false, 0, NULL, 0);
   unsigned m32_loads = 0, m32_u32_1 = 0, m32_u16_2 = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(b32_16.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               m32_loads++;
               if (intr->def.bit_size == 32 && intr->def.num_components == 1)
                  m32_u32_1++;
               if (intr->def.bit_size == 16 && intr->def.num_components == 2)
                  m32_u16_2++;
            }
         }
      }
   }
   /* Must stay separate (neither widened to u32vec2 nor narrowed to u16vec4). */
   CHECK(m32_loads == 2 && m32_u32_1 == 1 && m32_u16_2 == 1);
   ralloc_free(b32_16.shader);

   /* Case E (MEDIUM #6): non-power-of-two stride 20*i wraps at 20*0x33333333 ==
    * 0xfffffffc (+4 -> 0) and MUST NOT merge, while 16*i + 4 vec2/vec3 (4+12<=16)
    * is provably wrap-free and DOES merge into vec3. */
   nir_builder s20 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                    &agx_nir_options,
                                                    "bounded_stride20_no_merge");
   nir_def *s20_base = nir_imm_int64(&s20, 0x100000);
   nir_def *s20_i = nir_channel(&s20, nir_load_global_invocation_id(&s20, 32), 0);
   nir_def *s20_off = nir_imul_imm(&s20, s20_i, 20);
   nir_def *s20_bound = nir_imm_int(&s20, 4);
   nir_def *s20_0 = nir_load_global_bounded(&s20, 1, 32, s20_base, s20_off,
                                            s20_bound, .align_mul = 4, .align_offset = 0);
   nir_def *s20_1 = nir_load_global_bounded(&s20, 1, 32, s20_base,
                                            nir_iadd_imm(&s20, s20_off, 4),
                                            s20_bound, .align_mul = 4, .align_offset = 0);
   nir_store_global(&s20, nir_vec2(&s20, s20_0, s20_1), nir_imm_int64(&s20, 0x80000),
                    .write_mask = 0x3, .align_mul = 4);
   nir_shader_gather_info(s20.shader, nir_shader_get_entrypoint(s20.shader));
   hk_lower_nir(dev, s20.shader, &rs, false, 0, NULL, 0);
   unsigned s20_loads = 0, s20_vec = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(s20.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               s20_loads++;
               if (intr->def.num_components > 1)
                  s20_vec++;
            }
         }
      }
   }
   CHECK(s20_loads == 2 && s20_vec == 0);
   ralloc_free(s20.shader);

   nir_builder s16 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                    &agx_nir_options,
                                                    "bounded_stride16_off4_merge");
   nir_def *s16_base = nir_imm_int64(&s16, 0x100000);
   nir_def *s16_i = nir_channel(&s16, nir_load_global_invocation_id(&s16, 32), 0);
   nir_def *s16_off = nir_imul_imm(&s16, s16_i, 16);
   nir_def *s16_bound = nir_load_global_constant(&s16, 1, 32, nir_imm_int64(&s16, 0x20000),
                                                 .align_mul = 4);
   nir_def *s16_0 = nir_load_global_bounded(&s16, 1, 32, s16_base,
                                            nir_iadd_imm(&s16, s16_off, 4),
                                            s16_bound, .align_mul = 4, .align_offset = 0);
   nir_def *s16_1 = nir_load_global_bounded(&s16, 1, 32, s16_base,
                                            nir_iadd_imm(&s16, s16_off, 8),
                                            s16_bound, .align_mul = 4, .align_offset = 0);
   nir_store_global(&s16, nir_vec2(&s16, s16_0, s16_1),
                    nir_imm_int64(&s16, 0x80000), .write_mask = 0x3, .align_mul = 4);
   nir_shader_gather_info(s16.shader, nir_shader_get_entrypoint(s16.shader));
   hk_lower_nir(dev, s16.shader, &rs, false, 0, NULL, 0);
   unsigned s16_loads = 0, s16_vec2 = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(s16.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               s16_loads++;
               if (intr->def.num_components == 2 && intr->def.bit_size == 32)
                  s16_vec2++;
            }
         }
      }
   }
   CHECK(s16_loads == 1 && s16_vec2 == 1);
   ralloc_free(s16.shader);

   /* 32*i + 4 vec4 (4 + 16 <= 32) is provably wrap-free and merges into vec4. */
   nir_builder s32 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                    &agx_nir_options,
                                                    "bounded_stride32_off4_vec4");
   nir_def *s32_base = nir_imm_int64(&s32, 0x100000);
   nir_def *s32_i = nir_channel(&s32, nir_load_global_invocation_id(&s32, 32), 0);
   nir_def *s32_off = nir_imul_imm(&s32, s32_i, 32);
   nir_def *s32_bound = nir_load_global_constant(&s32, 1, 32, nir_imm_int64(&s32, 0x20000),
                                                 .align_mul = 4);
   nir_def *s32_offs[4], *s32_c[4];
   for (unsigned k = 0; k < 4; ++k)
      s32_offs[k] = nir_iadd_imm(&s32, s32_off, 4 + 4 * k);
   for (unsigned k = 0; k < 4; ++k) {
      s32_c[k] = nir_load_global_bounded(&s32, 1, 32, s32_base, s32_offs[k],
                                         s32_bound, .align_mul = 4, .align_offset = 0);
   }
   nir_store_global(&s32, nir_vec(&s32, s32_c, 4),
                    nir_imm_int64(&s32, 0x80000), .write_mask = 0xf, .align_mul = 4);
   nir_shader_gather_info(s32.shader, nir_shader_get_entrypoint(s32.shader));
   hk_lower_nir(dev, s32.shader, &rs, false, 0, NULL, 0);
   unsigned s32_loads = 0, s32_vec4 = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(s32.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               s32_loads++;
               if (intr->def.num_components == 4 && intr->def.bit_size == 32)
                  s32_vec4++;
            }
         }
      }
   }
   CHECK(s32_loads == 1 && s32_vec4 == 1);
   ralloc_free(s32.shader);

   /* Case F: (16*i + 8) & ~3 retains mod-16 residue rem=8: vec2 (8+8<=16)
    * merges, while vec3 (8+12>16) MUST NOT merge because +8 wraps at
    * 0xfffffff8. */
   {
      nir_builder ib = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                      &agx_nir_options,
                                                      "bounded_iand_residue_vec2");
      nir_def *ibase = nir_imm_int64(&ib, 0x100000);
      nir_def *ii = nir_channel(&ib, nir_load_global_invocation_id(&ib, 32), 0);
      nir_def *ioff = nir_iand_imm(&ib, nir_iadd_imm(&ib, nir_amul_imm(&ib, ii, 16), 8), ~3u);
      nir_def *ibound = nir_load_global_constant(&ib, 1, 32, nir_imm_int64(&ib, 0x20000),
                                                 .align_mul = 4);
      nir_def *i0 = nir_load_global_bounded(&ib, 1, 32, ibase, ioff, ibound,
                                            .align_mul = 4, .align_offset = 0);
      nir_def *i1 = nir_load_global_bounded(&ib, 1, 32, ibase, nir_iadd_imm(&ib, ioff, 4),
                                            ibound, .align_mul = 4, .align_offset = 0);
      nir_store_global(&ib, nir_vec2(&ib, i0, i1), nir_imm_int64(&ib, 0x80000),
                       .write_mask = 0x3, .align_mul = 4);
      nir_shader_gather_info(ib.shader, nir_shader_get_entrypoint(ib.shader));
      hk_lower_nir(dev, ib.shader, &rs, false, 0, NULL, 0);
      unsigned loads = 0, vec2 = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(ib.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               loads++;
               if (intr->def.num_components == 2 && intr->def.bit_size == 32)
                  vec2++;
            }
         }
      }
      CHECK(loads == 1 && vec2 == 1);
      ralloc_free(ib.shader);

      /* Same (16*i + 8) & ~3 for vec3 (8+12=20 > 16): MUST NOT merge to vec3. */
      nir_builder ib3 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                       &agx_nir_options,
                                                       "bounded_iand_residue_no_vec3");
      nir_def *i3base = nir_imm_int64(&ib3, 0x100000);
      nir_def *i3i = nir_channel(&ib3, nir_load_global_invocation_id(&ib3, 32), 0);
      nir_def *i3off = nir_iand_imm(&ib3, nir_iadd_imm(&ib3, nir_amul_imm(&ib3, i3i, 16), 8), ~3u);
      nir_def *i3bound = nir_load_global_constant(&ib3, 1, 32, nir_imm_int64(&ib3, 0x20000),
                                                  .align_mul = 4);
      nir_def *i3o[3], *i3c[3];
      for (unsigned k = 0; k < 3; ++k)
         i3o[k] = k ? nir_iadd_imm(&ib3, i3off, 4 * k) : i3off;
      for (unsigned k = 0; k < 3; ++k)
         i3c[k] = nir_load_global_bounded(&ib3, 1, 32, i3base, i3o[k], i3bound,
                                          .align_mul = 4, .align_offset = 0);
      nir_store_global(&ib3, nir_vec(&ib3, i3c, 3), nir_imm_int64(&ib3, 0x80000),
                       .write_mask = 0x7, .align_mul = 4);
      nir_shader_gather_info(ib3.shader, nir_shader_get_entrypoint(ib3.shader));
      hk_lower_nir(dev, ib3.shader, &rs, false, 0, NULL, 0);
      unsigned loads3 = 0, vec3 = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(ib3.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               loads3++;
               if (intr->def.num_components >= 3)
                  vec3++;
            }
         }
      }
      /* First 2 components (8+8<=16) merge into vec2; 3rd stays scalar. */
      CHECK(loads3 == 2 && vec3 == 0);
      ralloc_free(ib3.shader);
   }

   /* Case G: umin-clamped non-power-of-two stride (20 * umin(i, 1000)) is
    * provably wrap-free and merges into vec4, while amul-indexed 16*i also
    * merges into vec4. */
   {
      nir_builder ub = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                      &agx_nir_options,
                                                      "bounded_umin_stride20_vec4");
      nir_def *ubase = nir_imm_int64(&ub, 0x100000);
      nir_def *ui = nir_channel(&ub, nir_load_global_invocation_id(&ub, 32), 0);
      nir_def *uoff = nir_amul_imm(&ub, nir_umin(&ub, ui, nir_imm_int(&ub, 1000)), 20);
      nir_def *ubound = nir_load_global_constant(&ub, 1, 32, nir_imm_int64(&ub, 0x20000),
                                                 .align_mul = 4);
      nir_def *uo[4], *uc[4];
      for (unsigned k = 0; k < 4; ++k)
         uo[k] = k ? nir_iadd_imm(&ub, uoff, 4 * k) : uoff;
      for (unsigned k = 0; k < 4; ++k)
         uc[k] = nir_load_global_bounded(&ub, 1, 32, ubase, uo[k], ubound,
                                         .align_mul = 4, .align_offset = 0);
      nir_store_global(&ub, nir_vec(&ub, uc, 4), nir_imm_int64(&ub, 0x80000),
                       .write_mask = 0xf, .align_mul = 4);
      nir_shader_gather_info(ub.shader, nir_shader_get_entrypoint(ub.shader));
      hk_lower_nir(dev, ub.shader, &rs, false, 0, NULL, 0);
      unsigned loads = 0, vec4 = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(ub.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               loads++;
               if (intr->def.num_components == 4 && intr->def.bit_size == 32)
                  vec4++;
            }
         }
      }
      CHECK(loads == 1 && vec4 == 1);
      ralloc_free(ub.shader);
   }

   /* Case H: VK_DESCRIPTOR_TYPE_MUTABLE_EXT UBO descriptor zeroes .w so
    * constant-offset and amul-indexed robust UBO loads vectorize cleanly. */
   {
      struct hk_descriptor_set_layout *mut_layout =
         make_set_layout(dev, VK_DESCRIPTOR_TYPE_MUTABLE_EXT);
      CHECK(mut_layout);
      mut_layout->binding[0].stride = 32;
      mut_layout->non_variable_descriptor_buffer_size = 32;
      struct vk_descriptor_set_layout *mut_sets[] = {&mut_layout->vk};
      const struct vk_pipeline_robustness_state robust_rs = {
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      nir_builder mb2 = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                       &agx_nir_options,
                                                       "mutable_ubo_robust_vec4");
      nir_def *idx = nir_channel(&mb2, nir_load_global_invocation_id(&mb2, 32), 0);
      nir_def *res = nir_vulkan_resource_index(&mb2, 4, 32, nir_imm_int(&mb2, 0),
                                               .desc_set = 0, .binding = 0,
                                               .desc_type = nir_descriptor_type_uniform_buffer);
      nir_def *desc = nir_load_vulkan_descriptor(&mb2, 4, 32, res,
                                                 .desc_type = nir_descriptor_type_uniform_buffer);
      nir_def *dyn_off = nir_amul_imm(&mb2, idx, 16);
      nir_def *vals[4];
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *off = k ? nir_iadd_imm(&mb2, dyn_off, 4 * k) : dyn_off;
         vals[k] = nir_load_ubo(&mb2, 1, 32, desc, off, .align_mul = 16,
                                .align_offset = (4 * k) % 16, .range = ~0u);
      }
      nir_store_global(&mb2, nir_vec(&mb2, vals, 4), nir_imm_int64(&mb2, 0x80000),
                       .write_mask = 0xf, .align_mul = 4);
      nir_shader_gather_info(mb2.shader, nir_shader_get_entrypoint(mb2.shader));
      hk_lower_nir(dev, mb2.shader, &robust_rs, false, 1, mut_sets, 0);
      unsigned ubo_vec4 = 0, ubo_scalar = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(mb2.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global_constant) {
               if (intr->def.num_components == 4)
                  ubo_vec4++;
               else if (intr->def.num_components == 1)
                  ubo_scalar++;
            }
         }
      }
      /* 1 descriptor load (vec4/vec3) + 1 merged vec4 UBO load, 0 scalar UBO loads. */
      CHECK(ubo_vec4 >= 1 && ubo_scalar == 0);
      ralloc_free(mb2.shader);
      destroy_set_layout(mut_layout);
   }
   free(dev);

   puts("soft-fault SSBO & UBO bounded lowering + robust vectorize: pass");
   return 0;
}

static int
test_coalesced_lazy_image_fences(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;
   struct hk_descriptor_set_layout *img_layout =
      make_set_layout(dev, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
   struct vk_descriptor_set_layout *img_sets[] = {&img_layout->vk};
   const struct vk_pipeline_robustness_state rs = vk_robustness_disabled;

   /* 4 back-to-back imageStores followed by 1 imageLoad must coalesce into
    * ONE fence_image_write + ONE fence_tex_invalidate pair (instead of 4x). */
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "coalesced_4_stores");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *handle = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *val = nir_imm_vec4(&b, 1.0f, 2.0f, 3.0f, 4.0f);
   for (unsigned k = 0; k < 4; ++k) {
      nir_def *coord = nir_vec4(&b, nir_iadd_imm(&b, nir_channel(&b, gid, 0), k),
                                nir_channel(&b, gid, 1), nir_imm_int(&b, 0),
                                nir_imm_int(&b, 0));
      nir_bindless_image_store(&b, handle, coord, nir_imm_int(&b, 0), val,
                               nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                               .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                               .src_type = nir_type_float32);
   }
   nir_def *rcoord = nir_vec4(&b, nir_channel(&b, gid, 0), nir_channel(&b, gid, 1),
                              nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *loaded = nir_bindless_image_load(&b, 4, 32, handle, rcoord,
                                             nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                             .image_dim = GLSL_SAMPLER_DIM_2D,
                                             .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                             .dest_type = nir_type_float32);
   nir_store_global(&b, loaded, nir_imm_int64(&b, 0x80000),
                    .write_mask = 0xf, .align_mul = 16);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   hk_lower_nir(dev, b.shader, &rs, false, 1, img_sets, 0);
   struct nir_stats st = gather_stats(b.shader);
   CHECK(st.native_stores == 4);
   CHECK(st.fence_tex_invalidate == 1);
   CHECK(st.fence_image_write >= 1);
   /* Top-level hot path has 1 coalesced fence_image_write before imageLoad,
    * and 1 fence_tex_invalidate right before the texture-routed load. */
   unsigned top_pbe = 0;
   foreach_list_typed(nir_cf_node, n, node, &nir_shader_get_entrypoint(b.shader)->body) {
      if (n->type != nir_cf_node_block)
         continue;
      nir_foreach_instr(ins, nir_cf_node_as_block(n)) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         if (nir_instr_as_intrinsic(ins)->intrinsic ==
             nir_intrinsic_fence_image_write_apple9_agx)
            top_pbe++;
      }
   }
   CHECK(top_pbe == 1 && st.fence_tex_invalidate == 1);
   CHECK(compile_lowered_compute(b.shader) == 0);
   ralloc_free(b.shader);

   /* Workgroup image barrier between imageStore and imageLoad:
    * hk_apple9_lower_image_barriers preserves nir_var_image on SCOPE_WORKGROUP
    * so the backend emits workgroup-scoped 0x0e51 + 0x0ed1 without an extra
    * eager per-store fence pair. */
   nir_builder wb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "workgroup_image_barrier");
   wb.shader->info.internal = false;
   wb.shader->info.workgroup_size[0] = 64;
   wb.shader->info.workgroup_size[1] = wb.shader->info.workgroup_size[2] = 1;
   nir_def *wgid = nir_load_global_invocation_id(&wb, 32);
   nir_def *whandle = nir_bindless_image_agx(&wb, nir_imm_int(&wb, 0), .desc_set = 0);
   nir_def *wcoord = nir_vec4(&wb, nir_channel(&wb, wgid, 0), nir_channel(&wb, wgid, 1),
                              nir_imm_int(&wb, 0), nir_imm_int(&wb, 0));
   nir_bindless_image_store(&wb, whandle, wcoord, nir_imm_int(&wb, 0),
                            nir_imm_vec4(&wb, 1.0f, 2.0f, 3.0f, 4.0f),
                            nir_imm_int(&wb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32);
   nir_barrier(&wb,
               .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_image);
   nir_def *wloaded = nir_bindless_image_load(&wb, 4, 32, whandle, wcoord,
                                              nir_imm_int(&wb, 0), nir_imm_int(&wb, 0),
                                              .image_dim = GLSL_SAMPLER_DIM_2D,
                                              .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                              .dest_type = nir_type_float32);
   nir_store_global(&wb, wloaded, nir_imm_int64(&wb, 0x80000),
                    .write_mask = 0xf, .align_mul = 16);
   nir_shader_gather_info(wb.shader, nir_shader_get_entrypoint(wb.shader));

   hk_lower_nir(dev, wb.shader, &rs, false, 1, img_sets, 0);
   struct nir_stats wst = gather_stats(wb.shader);
   CHECK(wst.native_stores == 1);
   CHECK(wst.fence_image_write == 0 && wst.fence_tex_invalidate == 0);
   unsigned wg_img_barriers = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(wb.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (intr->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP &&
             (nir_intrinsic_memory_modes(intr) & nir_var_image))
            wg_img_barriers++;
      }
   }
   CHECK(wg_img_barriers == 1);
   CHECK(compile_lowered_compute(wb.shader) == 0);
   ralloc_free(wb.shader);

   /* Pure unchecked native image store + workgroup image barrier (MEDIUM #8):
    * no global image fallback exists, so nir_var_mem_global must NOT be added
    * (no extra 0x0885 barrier). */
   nir_builder pb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "pure_native_wg_img_barrier");
   pb.shader->info.internal = false;
   pb.shader->info.workgroup_size[0] = 64;
   pb.shader->info.workgroup_size[1] = pb.shader->info.workgroup_size[2] = 1;
   nir_def *pgid = nir_load_global_invocation_id(&pb, 32);
   nir_def *phandle = nir_bindless_image_agx(&pb, nir_imm_int(&pb, 0), .desc_set = 0);
   nir_def *pcoord = nir_vec4(&pb, nir_channel(&pb, pgid, 0), nir_channel(&pb, pgid, 1),
                              nir_imm_int(&pb, 0), nir_imm_int(&pb, 0));
   nir_bindless_image_store(&pb, phandle, pcoord, nir_imm_int(&pb, 0),
                            nir_imm_vec4(&pb, 1.0f, 2.0f, 3.0f, 4.0f),
                            nir_imm_int(&pb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32);
   nir_barrier(&pb,
               .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_image);
   nir_shader_gather_info(pb.shader, nir_shader_get_entrypoint(pb.shader));
   hk_lower_nir(dev, pb.shader, &rs, false, 1, img_sets, 0);
   unsigned pure_wg_img = 0, pure_wg_global = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(pb.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (intr->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
            if (nir_intrinsic_memory_modes(intr) & nir_var_image)
               pure_wg_img++;
            if (nir_intrinsic_memory_modes(intr) & nir_var_mem_global)
               pure_wg_global++;
         }
      }
   }
   CHECK(pure_wg_img == 1 && pure_wg_global == 0);
   CHECK(compile_lowered_compute(pb.shader) == 0);
   ralloc_free(pb.shader);

   /* HIGH #4: device-scope non-image barrier (memoryBarrierBuffer) between
    * imageStore and imageLoad drains PBE before the barrier and MUST STILL
    * emit fence_tex_invalidate before the subsequent imageLoad. */
   nir_builder db = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "device_ssbo_barrier_tex_inv");
   db.shader->info.internal = false;
   db.shader->info.workgroup_size[0] = 64;
   db.shader->info.workgroup_size[1] = db.shader->info.workgroup_size[2] = 1;
   nir_def *dgid = nir_load_global_invocation_id(&db, 32);
   nir_def *dhandle = nir_bindless_image_agx(&db, nir_imm_int(&db, 0), .desc_set = 0);
   nir_def *dcoord = nir_vec4(&db, nir_channel(&db, dgid, 0), nir_channel(&db, dgid, 1),
                              nir_imm_int(&db, 0), nir_imm_int(&db, 0));
   nir_bindless_image_store(&db, dhandle, dcoord, nir_imm_int(&db, 0),
                            nir_imm_vec4(&db, 1.0f, 2.0f, 3.0f, 4.0f),
                            nir_imm_int(&db, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32);
   nir_barrier(&db,
               .execution_scope = SCOPE_NONE,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_ssbo);
   nir_def *dloaded = nir_bindless_image_load(&db, 4, 32, dhandle, dcoord,
                                              nir_imm_int(&db, 0), nir_imm_int(&db, 0),
                                              .image_dim = GLSL_SAMPLER_DIM_2D,
                                              .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                              .dest_type = nir_type_float32);
   nir_store_global(&db, dloaded, nir_imm_int64(&db, 0x80000),
                    .write_mask = 0xf, .align_mul = 16);
   nir_shader_gather_info(db.shader, nir_shader_get_entrypoint(db.shader));
   hk_lower_nir(dev, db.shader, &rs, false, 1, img_sets, 0);
   struct nir_stats dst = gather_stats(db.shader);
   CHECK(dst.native_stores == 1);
   CHECK(dst.fence_image_write == 1 && dst.fence_tex_invalidate == 1);
   CHECK(compile_lowered_compute(db.shader) == 0);
   ralloc_free(db.shader);

   /* HIGH #3: dirty state across direct & nested loop break exits MUST emit
    * both fence_image_write and fence_tex_invalidate before post-loop load. */
   for (unsigned nested = 0; nested < 2; ++nested) {
      nir_builder lb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         nested ? "nested_loop_break_fence" : "direct_loop_break_fence");
      lb.shader->info.internal = false;
      lb.shader->info.workgroup_size[0] = 64;
      lb.shader->info.workgroup_size[1] = lb.shader->info.workgroup_size[2] = 1;
      nir_def *lgid = nir_load_global_invocation_id(&lb, 32);
      nir_def *lhandle = nir_bindless_image_agx(&lb, nir_imm_int(&lb, 0), .desc_set = 0);
      nir_def *lcoord = nir_vec4(&lb, nir_channel(&lb, lgid, 0), nir_channel(&lb, lgid, 1),
                                 nir_imm_int(&lb, 0), nir_imm_int(&lb, 0));
      nir_def *cond = nir_ine_imm(&lb, nir_channel(&lb, lgid, 0), 0);
      nir_def *cond2 = nir_ine_imm(&lb, nir_channel(&lb, lgid, 1), 0);
      nir_push_loop(&lb);
      if (nested)
         nir_push_loop(&lb);
      nir_push_if(&lb, cond);
      nir_bindless_image_store(&lb, lhandle, lcoord, nir_imm_int(&lb, 0),
                               nir_imm_vec4(&lb, 1.0f, 2.0f, 3.0f, 4.0f),
                               nir_imm_int(&lb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                               .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                               .src_type = nir_type_float32);
      nir_jump(&lb, nir_jump_break);
      nir_pop_if(&lb, NULL);
      nir_def *in_ld = nir_bindless_image_load(&lb, 4, 32, lhandle, lcoord,
                                               nir_imm_int(&lb, 0), nir_imm_int(&lb, 0),
                                               .image_dim = GLSL_SAMPLER_DIM_2D,
                                               .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                               .dest_type = nir_type_float32);
      nir_store_global(&lb, in_ld, nir_imm_int64(&lb, 0x80000),
                       .write_mask = 0xf, .align_mul = 16);
      nir_push_if(&lb, cond2);
      nir_jump(&lb, nir_jump_break);
      nir_pop_if(&lb, NULL);
      nir_pop_loop(&lb, NULL);
      if (nested) {
         nir_push_if(&lb, cond2);
         nir_jump(&lb, nir_jump_break);
         nir_pop_if(&lb, NULL);
         nir_pop_loop(&lb, NULL);
      }
      nir_def *post_ld = nir_bindless_image_load(&lb, 4, 32, lhandle, lcoord,
                                                 nir_imm_int(&lb, 0), nir_imm_int(&lb, 0),
                                                 .image_dim = GLSL_SAMPLER_DIM_2D,
                                                 .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                                 .dest_type = nir_type_float32);
      nir_store_global(&lb, post_ld, nir_imm_int64(&lb, 0x80010),
                       .write_mask = 0xf, .align_mul = 16);
      nir_shader_gather_info(lb.shader, nir_shader_get_entrypoint(lb.shader));
      hk_lower_nir(dev, lb.shader, &rs, false, 1, img_sets, 0);
      /* Verify post-loop block has BOTH fence_image_write and fence_tex_invalidate. */
      unsigned post_pbe = 0, post_tex = 0;
      bool after_loop = false;
      foreach_list_typed(nir_cf_node, n, node, &nir_shader_get_entrypoint(lb.shader)->body) {
         if (n->type == nir_cf_node_loop) {
            after_loop = true;
         } else if (after_loop) {
            nir_foreach_block_in_cf_node(blk, n) {
               nir_foreach_instr(ins, blk) {
                  if (ins->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_op op = nir_instr_as_intrinsic(ins)->intrinsic;
                  if (op == nir_intrinsic_fence_image_write_apple9_agx)
                     post_pbe++;
                  else if (op == nir_intrinsic_fence_tex_invalidate_apple9_agx)
                     post_tex++;
               }
            }
         }
      }
      CHECK(post_pbe >= 1 && post_tex >= 1);
      CHECK(compile_lowered_compute(lb.shader) == 0);
      ralloc_free(lb.shader);
   }

   /* Coherent/volatile storage images lower 100% natively via hardware
    * memory_coherence_device bits + immediate PBE drain (0x0650), with ZERO
    * global fallback stores and non-CSEable image_load_apple9_agx ordering. */
   nir_builder gb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "native_coherent_rw_and_nocse");
   gb.shader->info.internal = false;
   gb.shader->info.workgroup_size[0] = 64;
   gb.shader->info.workgroup_size[1] = gb.shader->info.workgroup_size[2] = 1;
   nir_def *ggid = nir_load_global_invocation_id(&gb, 32);
   nir_def *ghandle = nir_bindless_image_agx(&gb, nir_imm_int(&gb, 0), .desc_set = 0);
   nir_def *gcoord = nir_vec4(&gb, nir_channel(&gb, ggid, 0), nir_channel(&gb, ggid, 1),
                              nir_imm_int(&gb, 0), nir_imm_int(&gb, 0));
   nir_def *gload0 = nir_bindless_image_load(&gb, 4, 32, ghandle, gcoord,
                                             nir_imm_int(&gb, 0), nir_imm_int(&gb, 0),
                                             .image_dim = GLSL_SAMPLER_DIM_2D,
                                             .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                             .dest_type = nir_type_float32,
                                             .access = ACCESS_COHERENT);
   nir_bindless_image_store(&gb, ghandle, gcoord, nir_imm_int(&gb, 0),
                            nir_imm_vec4(&gb, 1.0f, 2.0f, 3.0f, 4.0f),
                            nir_imm_int(&gb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32);
   nir_bindless_image_store(&gb, ghandle, gcoord, nir_imm_int(&gb, 0),
                            nir_imm_vec4(&gb, 5.0f, 6.0f, 7.0f, 8.0f),
                            nir_imm_int(&gb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32,
                            .access = ACCESS_COHERENT);
   nir_def *gload1 = nir_bindless_image_load(&gb, 4, 32, ghandle, gcoord,
                                             nir_imm_int(&gb, 0), nir_imm_int(&gb, 0),
                                             .image_dim = GLSL_SAMPLER_DIM_2D,
                                             .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                             .dest_type = nir_type_float32,
                                             .access = ACCESS_COHERENT);
   nir_store_global(&gb, nir_fadd(&gb, gload0, gload1), nir_imm_int64(&gb, 0x80000),
                    .write_mask = 0xf, .align_mul = 16);
   nir_shader_gather_info(gb.shader, nir_shader_get_entrypoint(gb.shader));
   hk_lower_nir(dev, gb.shader, &rs, false, 1, img_sets, 0);
   /* Running CSE after lowering MUST keep both image_load_apple9_agx intact. */
   nir_opt_cse(gb.shader);
   struct nir_stats gst = gather_stats(gb.shader);
   CHECK(gst.native_stores == 2 && gst.native_store_coherent == 1);
   CHECK(gst.native_loads == 2 && gst.native_load_coherent == 2);
   CHECK(gst.fence_image_write == 1 && gst.fence_tex_invalidate == 1);
   CHECK(gst.global_loads == 0 && gst.global_stores == 1);
   CHECK(compile_lowered_compute(gb.shader) == 0);
   ralloc_free(gb.shader);

   /* Store-only shader with normal + coherent store: both native PBE, coherent
    * store emits immediate fence_image_write (0x0650), 0 global stores. */
   nir_builder sb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "store_only_native_then_coherent");
   sb.shader->info.internal = false;
   sb.shader->info.workgroup_size[0] = 64;
   sb.shader->info.workgroup_size[1] = sb.shader->info.workgroup_size[2] = 1;
   nir_def *sgid = nir_load_global_invocation_id(&sb, 32);
   nir_def *shandle = nir_bindless_image_agx(&sb, nir_imm_int(&sb, 0), .desc_set = 0);
   nir_def *scoord = nir_vec4(&sb, nir_channel(&sb, sgid, 0), nir_channel(&sb, sgid, 1),
                              nir_imm_int(&sb, 0), nir_imm_int(&sb, 0));
   nir_bindless_image_store(&sb, shandle, scoord, nir_imm_int(&sb, 0),
                            nir_imm_vec4(&sb, 1.0f, 2.0f, 3.0f, 4.0f),
                            nir_imm_int(&sb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32);
   nir_bindless_image_store(&sb, shandle, scoord, nir_imm_int(&sb, 0),
                            nir_imm_vec4(&sb, 5.0f, 6.0f, 7.0f, 8.0f),
                            nir_imm_int(&sb, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                            .src_type = nir_type_float32,
                            .access = ACCESS_COHERENT);
   nir_shader_gather_info(sb.shader, nir_shader_get_entrypoint(sb.shader));
   hk_lower_nir(dev, sb.shader, &rs, false, 1, img_sets, 0);
   struct nir_stats sst = gather_stats(sb.shader);
   CHECK(sst.native_stores == 2 && sst.native_store_coherent == 1);
   CHECK(sst.fence_image_write == 1 && sst.fence_tex_invalidate == 0);
   CHECK(sst.global_stores == 0);
   CHECK(compile_lowered_compute(sb.shader) == 0);
   ralloc_free(sb.shader);

   /* HIGH #5: 2D ACCESS_NON_READABLE bindless_image_atomic (lowered to global
    * by agx_nir_lower_images) + native 2D bindless_image_store + workgroup
    * nir_var_image barrier MUST keep nir_var_mem_global on the barrier, and
    * promote the coexisting image store to ACCESS_COHERENT. */
   nir_builder ab = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options,
                                                   "atomic_plus_native_wg_barrier");
   ab.shader->info.internal = false;
   ab.shader->info.workgroup_size[0] = 64;
   ab.shader->info.workgroup_size[1] = ab.shader->info.workgroup_size[2] = 1;
   nir_def *agid = nir_load_global_invocation_id(&ab, 32);
   nir_def *ahandle = nir_bindless_image_agx(&ab, nir_imm_int(&ab, 0), .desc_set = 0);
   nir_def *acoord = nir_vec4(&ab, nir_channel(&ab, agid, 0), nir_channel(&ab, agid, 1),
                              nir_imm_int(&ab, 0), nir_imm_int(&ab, 0));
   nir_def *ares = nir_bindless_image_atomic(&ab, 32, ahandle, acoord,
                                             nir_imm_int(&ab, 0),
                                             nir_imm_int(&ab, 1),
                                             .image_dim = GLSL_SAMPLER_DIM_2D,
                                             .format = PIPE_FORMAT_R32_UINT,
                                             .atomic_op = nir_atomic_op_iadd,
                                             .access = ACCESS_NON_READABLE);
   nir_bindless_image_store(&ab, ahandle, acoord, nir_imm_int(&ab, 0),
                            nir_vec4(&ab, ares, ares, ares, ares),
                            nir_imm_int(&ab, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32_UINT,
                            .src_type = nir_type_uint32);
   nir_barrier(&ab,
               .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_image);
   nir_shader_gather_info(ab.shader, nir_shader_get_entrypoint(ab.shader));
   hk_lower_nir(dev, ab.shader, &rs, false, 1, img_sets, 0);
   struct nir_stats abst = gather_stats(ab.shader);
   CHECK(abst.native_stores == 1 && abst.native_store_coherent == 1);
   unsigned ab_wg_img = 0, ab_wg_global = 0;
   nir_foreach_block(blk, nir_shader_get_entrypoint(ab.shader)) {
      nir_foreach_instr(ins, blk) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (intr->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
            if (nir_intrinsic_memory_modes(intr) & nir_var_image)
               ab_wg_img++;
            if (nir_intrinsic_memory_modes(intr) & nir_var_mem_global)
               ab_wg_global++;
         }
      }
   }
   CHECK(ab_wg_img == 1 && ab_wg_global == 1);
   CHECK(compile_lowered_compute(ab.shader) == 0);
   ralloc_free(ab.shader);

   /* Unused ACCESS_VOLATILE imageLoad must survive generic nir_opt_dce before
    * Apple9 lowering and compile to a hardware fetch; ACCESS_CAN_REORDER must
    * be cleared on image_load_apple9_agx. */
   {
      nir_builder vb = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                      &agx_nir_options,
                                                      "volatile_dead_and_noreorder");
      vb.shader->info.internal = false;
      vb.shader->info.workgroup_size[0] = 64;
      vb.shader->info.workgroup_size[1] = vb.shader->info.workgroup_size[2] = 1;
      nir_def *vgid = nir_load_global_invocation_id(&vb, 32);
      nir_def *vhandle = nir_bindless_image_agx(&vb, nir_imm_int(&vb, 0), .desc_set = 0);
      nir_def *vcoord = nir_vec4(&vb, nir_channel(&vb, vgid, 0), nir_channel(&vb, vgid, 1),
                                 nir_imm_int(&vb, 0), nir_imm_int(&vb, 0));
      /* Unused volatile load */
      (void)nir_bindless_image_load(&vb, 4, 32, vhandle, vcoord,
                                    nir_imm_int(&vb, 0), nir_imm_int(&vb, 0),
                                    .image_dim = GLSL_SAMPLER_DIM_2D,
                                    .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                    .dest_type = nir_type_float32,
                                    .access = ACCESS_VOLATILE);
      /* Live load annotated with ACCESS_CAN_REORDER */
      nir_def *vkept = nir_bindless_image_load(&vb, 4, 32, vhandle, vcoord,
                                               nir_imm_int(&vb, 0), nir_imm_int(&vb, 0),
                                               .image_dim = GLSL_SAMPLER_DIM_2D,
                                               .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
                                               .dest_type = nir_type_float32,
                                               .access = ACCESS_CAN_REORDER);
      nir_store_global(&vb, vkept, nir_imm_int64(&vb, 0x80000),
                       .write_mask = 0xf, .align_mul = 16);
      nir_shader_gather_info(vb.shader, nir_shader_get_entrypoint(vb.shader));
      nir_opt_dce(vb.shader);
      hk_lower_nir(dev, vb.shader, &rs, false, 1, img_sets, 0);
      nir_opt_dce(vb.shader);
      unsigned loads = 0, can_reorder = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(vb.shader)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_image_load_apple9_agx) {
               loads++;
               if (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER)
                  can_reorder++;
            }
         }
      }
      CHECK(loads == 2 && can_reorder == 0);
      CHECK(compile_lowered_compute(vb.shader) == 0);
      ralloc_free(vb.shader);
   }

   destroy_set_layout(img_layout);
   free(dev);
   puts("coalesced lazy image fences + loop break + barrier + coherent + atomics: pass");
   return 0;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   CHECK(test_branchless_storage_and_tbuf() == 0);
   CHECK(test_gpl_sw_needed() == 0);
   CHECK(test_soft_fault_ssbo_and_4b_vectorize() == 0);
   CHECK(test_coalesced_lazy_image_fences() == 0);

   glsl_type_singleton_decref();
   return 0;
}
