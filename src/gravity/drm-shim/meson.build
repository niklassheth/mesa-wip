# Copyright 2019 Broadcom
# SPDX-License-Identifier: MIT

libgravity_null_drm_shim = shared_library(
  'gravity_null_drm_shim',
  'gravity_noop.c',
  include_directories: [inc_include, inc_src],
  dependencies: [dep_drm_shim, dep_valgrind],
  gnu_symbol_visibility : 'hidden',
  install : false,
)
