project('native-mesh-task-bridge-probe', 'c',
  default_options : ['buildtype=release', 'c_std=gnu11', 'warning_level=2'])
cc=meson.get_compiler('c')
vk=cc.find_library('vulkan')
headers=include_directories('../../../../../include')
executable('mesh-task', '../native-mesh-m2/probe.c',
  c_args : ['-DM5_TASK', '-DM2_RGBA'], include_directories : headers, dependencies : vk)
executable('mesh-task-ubo', '../native-mesh-m2/probe.c',
  c_args : ['-DM5_TASK', '-DM3A_INTERFACE'], include_directories : headers, dependencies : vk)
shared_library('host-record-sync', '../native-mesh-corpus/record-sync.c',
  dependencies : [cc.find_library('dl'), dependency('threads')])
