# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

libgravity_compiler_files = files(
  'agx_compile.c',
  'agx_apple9_machine.c',
  'agx_apple9_mesh_encoding.c',
  'agx_apple9_ir.c',
  'agx_apple9_opt.c',
  'agx_apple9_ra.c',
  'apple9_ra_sched.c',
  'apple9_ra_pcopy.c',
  'apple9_ra_validate.c',
  'apple9_ra_assign.c',
  'apple9_ra_spill.c',
  'apple9_ra_repair_ssa.c',
  'apple9_ra_spill_coalesce.c',
  'agx_compile_apple9.c',
  'agx_nir_lower_apple9_mesh.c',
  'agx_nir_apple9_mesh_native.c',
  'agx_nir_lower_apple9_texture.c',
  'agx_nir_lower_apple9_math.c',
  'agx_nir_lower_apple9_preserve.c',
  'agx_nir_lower_cull_distance.c',
  'agx_nir_lower_frag_sidefx.c',
  'agx_nir_lower_sample_mask.c',
  '../lib/agx_nir_lower_msaa.c',
  '../lib/agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_discard_zs_emit.c',
  'agx_nir_lower_texture.c',
)

agx_apple9_machine_table = custom_target(
  'agx_apple9_machine_table',
  input : ['agx_apple9_isa.py', 'agx_apple9_machine.h'],
  output : ['agx_apple9_machine_table.h', 'agx_apple9_machine_table.c'],
  command : [
    prog_python, '@INPUT0@',
    '--check', '@INPUT1@',
    '--out-h', '@OUTPUT0@',
    '--out-c', '@OUTPUT1@',
  ],
)

libgravity_compiler = static_library(
  'gravity_compiler',
  [libgravity_compiler_files, agx_apple9_machine_table],
  dependencies: [idep_gravity_debug, idep_nir, idep_mesautil, idep_gravity_cl, idep_libpoly],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests or with_tools.contains('drm-shim')
  executable(
    'gravity_apple9_tiny_tool',
    files('test/apple9-tiny-tool.c'),
    c_args : [no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  executable(
    'gravity_apple9_fragment_replay',
    files('test/apple9-fragment-replay.c'),
    c_args : [no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
endif

if with_tests
  libgravity_compiler_test_support = static_library(
    'gravity_compiler_test_support',
    files('agx_apple9_ir_test_support.c'),
    dependencies: [idep_gravity_debug, idep_nir, idep_mesautil],
    c_args : [no_override_init_args, '-Wno-c2x-extensions'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : false,
  )
  gravity_ra_budget_test = executable('gravity-apple9-ra-budget',
    'test/apple9-ra-budget.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler_test_support, libgravity_compiler],
  )
  test('gravity-apple9-ra-budget', gravity_ra_budget_test, suite : ['gravity'])
  gravity_mesh_limits_test = executable('gravity-apple9-mesh-limits',
    'test/apple9-mesh-limits.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-mesh-limits', gravity_mesh_limits_test, suite : ['gravity'])
  gravity_native_shared_test = executable('gravity-apple9-native-shared',
    'test/apple9-native-shared.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-native-shared', prog_python,
    args : [files('test/check-apple9-native-shared.py'), gravity_native_shared_test],
    suite : ['gravity'])
  gravity_native_workgroup_test = executable('gravity-apple9-native-workgroup',
    'test/apple9-native-workgroup.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-native-workgroup', gravity_native_workgroup_test, suite : ['gravity'])
  gravity_native_object_test = executable('gravity-apple9-native-object',
    'test/apple9-native-object.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-native-object', gravity_native_object_test, suite : ['gravity'])
  gravity_limits_workgroup_test = executable('gravity-apple9-limits-workgroup',
    'test/apple9-limits-workgroup.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-limits-workgroup', gravity_limits_workgroup_test, suite : ['gravity'])
  gravity_native_primitive_test = executable('gravity-apple9-native-primitive',
    'test/apple9-native-primitive.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-native-primitive', gravity_native_primitive_test, suite : ['gravity'])
  gravity_native_mesh_test = executable('gravity-apple9-native-mesh',
    'test/apple9-native-mesh.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  gravity_packed_indices_test = executable('gravity-apple9-packed-indices',
    'test/apple9-packed-indices.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libgravity_compiler],
  )
  test('gravity-apple9-packed-indices', gravity_packed_indices_test, suite : ['gravity'])
  test('gravity-apple9-native-mesh', prog_python,
    args : [files('test/check-apple9-native-mesh.py'), gravity_native_mesh_test],
    suite : ['gravity'],
  )
  test('gravity-apple9-native-mesh-counts', gravity_native_mesh_test,
    args : ['--counts'],
    suite : ['gravity'],
  )
  test('gravity-apple9-native-mesh-groups', prog_python,
    args : [files('test/check-apple9-mesh-groups.py'), gravity_native_mesh_test],
    suite : ['gravity'],
  )
  test(
    'gravity_compiler_tests',
    executable(
      'gravity_compiler_tests',
      files(
         'test/test-apple9.cpp',
         'test/test-apple9-mesh.cpp',
         'test/test-apple9-object.cpp',
         'test/test-apple9-opt.cpp',
         'test/test-apple9-ra-sched.cpp',
         'test/test-apple9-ra-pcopy.cpp',
         'test/test-apple9-shared-ra.cpp',
         'test/test-apple9-ra-spill.cpp',
         'test/test-apple9-ra-assign.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_nir, idep_mesautil],
      link_with : [libgravity_compiler_test_support, libgravity_compiler],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
endif
