project('native-mesh-publication-probe', 'c', default_options : ['buildtype=release', 'c_std=gnu11', 'warning_level=2'])
cc=meson.get_compiler('c')
vk=cc.find_library('vulkan')
headers=include_directories('../../../../../include')
executable('mesh-publication', '../native-mesh-m2/probe.c', include_directories : headers, dependencies : vk,
           c_args : ['-DM3_WIDE=1', '-DM3_MAX_VERTICES=256', '-DM3_MAX_PRIMITIVES=256'])
shared_library('vm-audit', '../native-stage-batching/vm-audit.c', name_prefix : '', include_directories : headers)
