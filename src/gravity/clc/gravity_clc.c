/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2020 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gravity_clc.h"
#include "gravity/compiler/agx_compile.h"
#include "gravity/compiler/agx_nir.h"
#include "gravity/lib/agx_apple9_libagx.h"
#include "compiler/glsl_types.h"
#include "compiler/spirv/nir_spirv.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_precompiled.h"
#include "nir_serialize.h"
#include "shader_enums.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "util/macros.h"
#include <sys/mman.h>

static const struct spirv_to_nir_options spirv_options = {
   .environment = NIR_SPIRV_OPENCL,
   .shared_addr_format = nir_address_format_62bit_generic,
   .global_addr_format = nir_address_format_62bit_generic,
   .temp_addr_format = nir_address_format_62bit_generic,
   .constant_addr_format = nir_address_format_64bit_global,
   .create_library = true,
   .printf = true,
};

/* Standard optimization loop */
static void
optimize(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_split_var_copies);
      NIR_PASS(progress, nir, nir_split_struct_vars, nir_var_function_temp);
      NIR_PASS(progress, nir, nir_lower_var_copies);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_lower_all_phis_to_scalar);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_deref);
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_lower_undef_to_zero, NULL);

      NIR_PASS(progress, nir, nir_opt_shrink_vectors, true);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);

   } while (progress);
}

static nir_shader *
compile(void *memctx, const uint32_t *spirv, size_t spirv_size)
{
   const nir_shader_compiler_options *nir_options = &agx_nir_options;

   assert(spirv_size % 4 == 0);
   nir_shader *nir =
      spirv_to_nir(spirv, spirv_size / 4, NULL, MESA_SHADER_KERNEL, "library",
                   &spirv_options, nir_options);
   if (!nir) {
      fprintf(stderr, "Failed to translate SPIR-V to NIR\n");
      return NULL;
   }
   nir_validate_shader(nir, "after spirv_to_nir");
   ralloc_steal(memctx, nir);

   nir_fixup_is_exported(nir);

   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_calls_to_builtins);

   nir_lower_compute_system_values_options cs = {.global_id_is_32bit = true};
   NIR_PASS(_, nir, nir_lower_compute_system_values, &cs);

   NIR_PASS(_, nir, nir_lower_printf,
            &(const struct nir_lower_printf_options){
               .hash_format_strings = true,
            });

   /* We have to lower away local constant initializers right before we
    * inline functions.  That way they get properly initialized at the top
    * of the function and not at the top of its caller.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_exported(nir);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_deref);

   /* We can't deal with constant data, get rid of it */
   nir_lower_constant_to_temp(nir);

   /* We can go ahead and lower the rest of the constant initializers.  We do
    * this here so that nir_remove_dead_variables and split_per_member_structs
    * below see the corresponding stores.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);

   /* LLVM loves take advantage of the fact that vec3s in OpenCL are 16B
    * aligned and so it can just read/write them as vec4s.  This results in a
    * LOT of vec4->vec3 casts on loads and stores.  One solution to this
    * problem is to get rid of all vec3 variables.
    */
   NIR_PASS(_, nir, nir_lower_vec3_to_vec4,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant);

   /* We assign explicit types early so that the optimizer can take advantage
    * of that information and hopefully get rid of some of our memcpys.
    */
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_uniform | nir_var_shader_temp | nir_var_function_temp |
               nir_var_mem_shared | nir_var_mem_global,
            glsl_get_cl_type_size_align);

   optimize(nir);

   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);

   /* Lower again, this time after dead-variables to get more compact variable
    * layouts.
    */
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant,
            glsl_get_cl_type_size_align);
   assert(nir->constant_data_size == 0);

   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
            nir_address_format_64bit_global);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_uniform,
            nir_address_format_32bit_offset_as_64bit);

   /* Note: we cannot lower explicit I/O here, because we need derefs in tact
    * for function calls into the library to work.
    */

   NIR_PASS(_, nir, nir_lower_convert_alu_types, NULL);
   NIR_PASS(_, nir, nir_opt_if, 0);
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   NIR_PASS(_, nir, agx_nir_lower_images);
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   optimize(nir);

   return nir;
}

static nir_def *
load_kernel_input(nir_builder *b, unsigned num_components, unsigned bit_size,
                  unsigned offset_B)
{
   assert((offset_B & 1) == 0 && "half-aligned");
   return nir_load_preamble(b, num_components, bit_size, .base = offset_B / 2);
}

int
main(int argc, char **argv)
{
   if (argc != 4) {
      fprintf(stderr, "Usage: %s [input spir-v] [output header] [output C]\n",
              argv[0]);
      return 1;
   }

   const char *infile = argv[1];
   const char *outh_file = argv[2];
   const char *outc_file = argv[3];

   void *mem_ctx = ralloc_context(NULL);

   int fd = open(infile, O_RDONLY);
   if (fd < 0) {
      fprintf(stderr, "Failed to open %s\n", infile);
      ralloc_free(mem_ctx);
      return 1;
   }

   off_t spirv_len = lseek(fd, 0, SEEK_END);
   const void *spirv_map = mmap(NULL, spirv_len, PROT_READ, MAP_PRIVATE, fd, 0);
   close(fd);
   if (spirv_map == MAP_FAILED) {
      fprintf(stderr, "Failed to mmap the file: errno=%d, %s\n", errno,
              strerror(errno));
      ralloc_free(mem_ctx);
      return 1;
   }

   FILE *fp_h = fopen(outh_file, "w");
   FILE *fp_c = fopen(outc_file, "w");
   if (!fp_h || !fp_c) {
      fprintf(stderr, "Failed to open output file(s): %s / %s\n", outh_file,
              outc_file);
      if (fp_h)
         fclose(fp_h);
      if (fp_c)
         fclose(fp_c);
      ralloc_free(mem_ctx);
      return 1;
   }
   glsl_type_singleton_init_or_ref();

   nir_precomp_print_header(fp_c, fp_h, "The Asahi Linux Contributors",
                            "libagx_shaders.h");

   nir_shader *nir = compile(mem_ctx, spirv_map, spirv_len);
   if (!nir) {
      glsl_type_singleton_decref();
      fclose(fp_c);
      fclose(fp_h);
      ralloc_free(mem_ctx);
      return 1;
   }

   /* load_preamble works at 16-bit granularity */
   struct nir_precomp_opts opt = {.arg_align_B = 2};

   nir_foreach_entrypoint(libfunc, nir) {
      unsigned nr_vars = nir_precomp_nr_variants(libfunc);

      nir_precomp_print_layout_struct(fp_h, &opt, libfunc);

      for (unsigned v = 0; v < nr_vars; ++v) {
         nir_shader *s = nir_precompiled_build_variant(
            libfunc, MESA_SHADER_COMPUTE, v, &agx_nir_options, &opt,
            load_kernel_input);

         nir_link_shader_functions(s, nir);
         NIR_PASS(_, s, nir_inline_functions);
         nir_remove_non_entrypoints(s);
         NIR_PASS(_, s, nir_opt_deref);
         NIR_PASS(_, s, nir_lower_vars_to_ssa);
         NIR_PASS(_, s, nir_remove_dead_derefs);
         NIR_PASS(_, s, nir_remove_dead_variables,
                  nir_var_function_temp | nir_var_shader_temp, NULL);
         NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
                  nir_var_shader_temp | nir_var_function_temp,
                  glsl_get_cl_type_size_align);

         NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
                  glsl_get_cl_type_size_align);

         NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_shared,
                  nir_address_format_62bit_generic);

         NIR_PASS(_, s, nir_lower_array_deref_of_vec,
                  nir_var_function_temp, NULL,
                  nir_lower_indirect_array_deref_of_vec_load |
                     nir_lower_indirect_array_deref_of_vec_store);
         NIR_PASS(_, s, nir_lower_indirect_derefs_to_if_else_trees,
                  nir_var_function_temp, 32);
         optimize(s);
         NIR_PASS(_, s, nir_lower_explicit_io,
                  nir_var_shader_temp | nir_var_function_temp |
                     nir_var_mem_shared | nir_var_mem_global,
                  nir_address_format_62bit_generic);
         NIR_PASS(_, s, nir_remove_dead_variables, nir_var_all, NULL);

         /* Full-shader NIR serialization embeds shader_info verbatim, which
          * cannot be shared by a 64-bit build tool and a 32-bit driver. These
          * kernels have one inlined function and explicit memory accesses, so
          * serialize that function with fixed-width kernel metadata instead.
          * Reject additional module state rather than silently dropping it.
          */
         if (s->info.stage != MESA_SHADER_COMPUTE ||
             !exec_list_is_empty(&s->variables) ||
             exec_list_length(&s->functions) != 1 ||
             s->constant_data_size || s->info.uses_printf ||
             s->xfb_info) {
            fprintf(stderr, "Unsupported module state in libagx kernel %s\n",
                    libfunc->name);
            exit(EXIT_FAILURE);
         }
         const struct agx_apple9_libagx_metadata metadata = {
            .format = AGX_APPLE9_LIBAGX_FORMAT,
            .workgroup_size = {s->info.workgroup_size[0],
                               s->info.workgroup_size[1],
                               s->info.workgroup_size[2]},
            .shared_size = s->info.shared_size,
            .scratch_size = s->scratch_size,
            .float_controls = s->info.float_controls_execution_mode,
            .api_subgroup_size = s->info.api_subgroup_size,
            .min_subgroup_size = s->info.min_subgroup_size,
            .max_subgroup_size = s->info.max_subgroup_size,
            .variable_local_size = s->info.workgroup_size_variable,
         };
         struct blob serialized;
         blob_init(&serialized);
         blob_write_uint32(&serialized, 0);
         blob_write_bytes(&serialized, &metadata, sizeof(metadata));
         nir_serialize_function(&serialized,
                                nir_shader_get_entrypoint(s)->function);
         blob_overwrite_uint32(&serialized, 0, serialized.size - 4);
         nir_precomp_print_blob(fp_c, libfunc->name, "nir", v,
                               (uint32_t *)serialized.data, serialized.size, true);
         blob_finish(&serialized);
         ralloc_free(s);
      }
   }

   nir_precomp_print_program_enum(fp_h, nir, "libagx");
   nir_precomp_print_dispatch_macros(fp_h, &opt, nir);

   nir_precomp_print_extern_binary_map(fp_h, "libagx", "nir");
   nir_precomp_print_binary_map(fp_c, nir, "libagx", "nir", NULL);

   glsl_type_singleton_decref();
   bool io_err = ferror(fp_c) || ferror(fp_h);
   if (fclose(fp_c) != 0)
      io_err = true;
   if (fclose(fp_h) != 0)
      io_err = true;
   ralloc_free(mem_ctx);
   if (io_err) {
      fprintf(stderr, "Failed to write output file(s): %s / %s\n", outh_file,
              outc_file);
      return 1;
   }
   return 0;
}
