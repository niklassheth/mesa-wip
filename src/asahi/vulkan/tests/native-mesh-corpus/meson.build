project('native-mesh-corpus-probe', 'c', default_options : ['buildtype=release', 'c_std=gnu11', 'warning_level=2'])
cc = meson.get_compiler('c')
vk = cc.find_library('vulkan')
headers = include_directories('../../../../../include')
executable('mesh-corpus', '../native-mesh-m2/probe.c',
  c_args : ['-DM2_RGBA'], include_directories : headers, dependencies : vk)
