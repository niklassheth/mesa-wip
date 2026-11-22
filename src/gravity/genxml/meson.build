# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

gravity_pack = custom_target(
  'gravity_pack',
  input : ['gen_pack.py', 'cmdbuf.xml', 'agx_pack_header.h'],
  output : 'agx_pack.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
)

idep_gravity_pack = declare_dependency(
  sources : [gravity_pack],
  dependencies : dep_valgrind,
  include_directories : include_directories('.'),
)
