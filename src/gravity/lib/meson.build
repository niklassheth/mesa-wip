# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libgravity_lib_files = files(
  'agx_apple9.c',
  'agx_apple9_geometry.c',
  'agx_apple9_launch.c',
  'agx_apple9_mesh.c',
  'agx_apple9_libagx.c',
  'agx_bo.c',
  'agx_shader_body.c',
  'agx_border.c',
  'agx_device.c',
  'agx_native_mesh.c',
  'agx_bg_eot.c',
  'agx_tilebuffer.c',
  'agx_nir_apple9_tilebuffer.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_va.c',
  'pool.c',
)

gravity_libcl_shaders = custom_target(
  'gravity_libcl_shaders',
  input : gravity_libcl_spv,
  output : ['libagx_shaders.h', 'libagx_shaders.c'],
  command : [prog_gravity_clc, gravity_libcl_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_gravity_cl_shaders_h = declare_dependency(
  sources : [gravity_libcl_shaders],
  include_directories : include_directories('.'),
)

libgravity_lib = static_library(
  'gravity_lib',
  [libgravity_lib_files, gravity_libcl_shaders, gravity_pack, sha1_h],
  include_directories : [inc_gallium, inc_gravity],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with: [libpoly_nir],
  dependencies: [idep_gravity_debug, dep_libdrm, dep_valgrind, idep_nir, idep_mesautil, idep_gravity_cl, idep_libpoly],
  build_by_default : false,
)

if with_tests
  test('gravity-native-mesh-advertisement', executable('gravity-native-mesh-advertisement',
    'tests/test-native-mesh-advertisement.c'), suite : ['gravity'])
  test('gravity-native-blocks-advertisement', executable('gravity-native-blocks-advertisement',
    'tests/test-native-blocks-advertisement.c', 'agx_native_mesh.c'),
    suite : ['gravity'])
  test('gravity-apple9-draw-tables', executable('gravity-apple9-draw-tables',
    'tests/test-apple9-draw-tables.c',
    include_directories : [inc_gravity],
    dependencies : [idep_mesautil, idep_nir, idep_gravity_cl],
    link_with : [libgravity_lib, libgravity_compiler]), suite : ['gravity'])
  test('gravity-bo-reaper', executable('gravity-bo-reaper', 'tests/test-bo-reaper.c',
    include_directories : [inc_gravity],
    dependencies : [idep_mesautil, idep_nir, idep_gravity_cl],
    link_with : [libgravity_lib, libgravity_compiler],
    link_args : ['-Wl,--wrap=drmIoctl', '-Wl,--wrap=drmPrimeFDToHandle',
                 '-Wl,--wrap=u_thread_create', '-Wl,--wrap=pthread_cond_signal',
                 '-Wl,--wrap=agx_va_free', '-Wl,--wrap=agx_va_retire']),
    suite : ['gravity'], timeout : 40)
  test('gravity-va-quarantine', executable('gravity-va-quarantine', 'tests/test-va-quarantine.c',
    include_directories : [inc_gravity],
    dependencies : [idep_mesautil, idep_nir, idep_gravity_cl],
    link_with : [libgravity_lib, libgravity_compiler]),
    suite : ['gravity'])
  test('gravity-apple9-coefficients', executable('gravity-apple9-coefficients',
    'tests/test-apple9-coefficients.c',
    include_directories : [inc_gravity],
    dependencies : [idep_mesautil, idep_nir, idep_gravity_cl],
    link_with : [libgravity_lib, libgravity_compiler]), suite : ['gravity'])
  test('gravity-apple9-graphics-roots', executable('gravity-apple9-graphics-roots',
    'tests/test-apple9-graphics-roots.c',
    include_directories : [inc_gravity],
    dependencies : [idep_mesautil, idep_nir, idep_gravity_cl],
    link_with : [libgravity_lib, libgravity_compiler]), suite : ['gravity'])
  test(
    'gravity_lib_tests',
    executable(
      'gravity_lib_tests',
      files(
        'tests/test-packing.cpp',
        'tests/test-border.cpp',
        'tests/test-apple9-cdm.cpp',
        'tests/test-apple9-launch-template.cpp',
        'tests/test-tilebuffer.cpp',
        'tests/test-primitive-count.cpp',
        'tests/test-poly-indirect-size.cpp',
        'tests/poly-indirect-size.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_gravity_pack, idep_mesautil, idep_nir],
      link_with : [libgravity_lib, libgravity_layout, libgravity_compiler],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )

  test(
    'gravity_libcl_apple9',
    executable(
      'gravity_libcl_apple9_test',
      files('tests/test-libagx-apple9.cpp',
            'tests/libagx-apple9-programs.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_gravity],
      dependencies: [idep_gtest, idep_nir, idep_mesautil, idep_gravity_cl_shaders_h],
      link_with : [libgravity_lib, libgravity_compiler],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
endif
