# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

inc_gravity = include_directories([
   '.', 'layout', 'lib', 'genxml', 'compiler'
])

libgravity_debug = static_library(
   'gravity_debug', 'debug.c',
   dependencies : [idep_mesautil],
   gnu_symbol_visibility : 'hidden',
   build_by_default : false,
)
idep_gravity_debug = declare_dependency(link_with : [libgravity_debug])

subdir('genxml')
subdir('libcl')
subdir('compiler')
subdir('clc')
subdir('layout')
subdir('lib')
subdir('vulkan')

if with_tools.contains('drm-shim')
   subdir('drm-shim')
endif
