# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

if get_option('precomp-compiler') == 'system'
  prog_gravity_clc = find_program('gravity_clc', native : true)
else
  prog_gravity_clc = executable(
    'gravity_clc',
    ['gravity_clc.c'],
    link_with : [libgravity_compiler],
    include_directories : [inc_include, inc_src],
    c_args : [pre_args, no_override_init_args],
    link_args : [ld_args_build_id],
    dependencies : [idep_vtn, idep_nir, idep_mesautil],
    # If we can run host binaries directly, just build gravity_clc for the host.
    native : not meson.can_run_host_binaries(),
    install : get_option('install-precomp-compiler'),
  )
endif
