# Copyright 2024 Valve Corporation
# SPDX-License-Identifier: MIT

gravity_libcl_shader_files = files(
  'compression.cl',
  'copy.cl',
  'draws.cl',
  'geometry.cl',
  'mesh.cl',
  'query.cl',
  'tessellation.cl',
  'tessellator.cl',
  'texture.cl',
)

gravity_libcl_spv = custom_target(
  'gravity_libcl_spv',
  input : gravity_libcl_shader_files,
  output : 'libagx.spv',
  command : [
    prog_mesa_clc, '-o', '@OUTPUT@', '--depfile', '@DEPFILE@',
    gravity_libcl_shader_files, '--',
    '-I' + join_paths(meson.project_source_root(), 'include'),
    '-I' + join_paths(meson.project_source_root(), 'src/compiler/libcl'),
    '-I' + join_paths(meson.project_source_root(), 'src/poly/cl'),
    '-I' + join_paths(meson.current_source_dir(), '.'),
    '-I' + join_paths(meson.current_source_dir(), '../../'),
    '-I' + join_paths(meson.current_build_dir(), '../genxml'),
    cl_args,
    ],
  depends : [gravity_pack],
  depfile : 'gravity_libcl_shaders.h.d',
)

gravity_libcl = custom_target(
  'gravity_libcl',
  input : gravity_libcl_spv,
  output : ['libagx.cpp', 'libagx.h'],
  command : [prog_vtn_bindgen2, gravity_libcl_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_gravity_cl = declare_dependency(
  sources : [gravity_libcl],
  include_directories : include_directories('.'),
)
