# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT
gravity_vk_files = files(
  'agx_timeline.c',
  'hk_gpu_busy.c',
  'hk_gpu_busy.h',
  'hk_gpu_busy_intervals.h',
  'hk_buffer.c',
  'hk_buffer.h',
  'hk_buffer_view.c',
  'hk_buffer_view.h',
  'hk_cmd_buffer.c',
  'hk_cmd_buffer.h',
  'hk_cmd_clear.c',
  'hk_cmd_dispatch.c',
  'hk_cmd_draw.c',
  'hk_cmd_draw_priv.h',
  'hk_cmd_passthrough_gs.c',
  'hk_cmd_render.c',
  'hk_apple9_polygon_points.c',
  'hk_cmd_apple9.c',
  'hk_apple9_mesh.c',
  'hk_cmd_meta.c',
  'hk_cmd_pool.c',
  'hk_cmd_pool.h',
  'hk_descriptor_set.h',
  'hk_descriptor_memory.c',
  'hk_usc_heap.c',
  'hk_descriptor_set.c',
  'hk_descriptor_set_layout.c',
  'hk_descriptor_set_layout.h',
  'hk_descriptor_table.c',
  'hk_descriptor_table.h',
  'hk_device.c',
  'hk_cpu_timing.c',
  'hk_device.h',
  'hk_device_memory.c',
  'hk_device_memory.h',
  'hk_event.c',
  'hk_event.h',
  'hk_format.c',
  'hk_feedback.c',
  'hk_feedback.h',
  'hk_image.c',
  'hk_image.h',
  'hk_image_view.c',
  'hk_image_view.h',
  'hk_instance.c',
  'hk_instance.h',
  'hk_nir_lower_cooperative_matrix.c',
  'hk_nir_lower_descriptors.c',
  'hk_nir_lower_mesh.c',
  'hk_nir_gs_mesh.c',
  'hk_nir_tes_mesh.c',
  'hk_physical_device.c',
  'hk_physical_device.h',
  'hk_private.h',
  'hk_query_pool.c',
  'hk_query_pool.h',
  'hk_queue.c',
  'hk_queue.h',
  'hk_queue_completion.c',
  'hk_queue_completion.h',
  'hk_sampler.c',
  'hk_sampler.h',
  'hk_shader.c',
  'hk_shader.h',
  'hk_shader_compile.c',
  'hk_shader_nir.c',
  'hk_shader_priv.h',
  'hk_sync.c',
  'hk_sync.h',
  'hk_wsi.c',
  'hk_wsi.h'
)

gravity_drirc = custom_target(
  'gravity_drirc',
  input  : [files('hk_drirc_gen.py'), drirc_gen, '00-gravity-defaults.conf'],
  output : ['hk_drirc.c', 'hk_drirc.h'],
  command : [
    prog_python, '@INPUT0@',
    '--import-path', join_paths(dir_source_root, 'src/util'),
    '--drirc-src', '@OUTPUT0@',
    '--drirc-hdr', '@OUTPUT1@',
    '--validate', '@INPUT2@',
  ],
)

gravity_entrypoints = custom_target(
  'gravity_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['hk_entrypoints.h', 'hk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'hk',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

gravity_vk_deps = [
  dep_libdrm,
  idep_nir,
  idep_vulkan_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi,
  idep_vulkan_wsi_headers,
  idep_gravity_pack,
  idep_gravity_cl,
  idep_gravity_cl_shaders_h,
]

gravity_diagnostic_c_args = []
if get_option('hk-cpu-sub-timing')
  gravity_diagnostic_c_args += '-DHK_CPU_SUB_TIMING=1'
endif
if get_option('hk-gpu-busy-counters')
  if host_machine.system() != 'linux'
    error('HK GPU busy diagnostics currently require Linux')
  endif
  gravity_diagnostic_c_args += '-DHK_GPU_BUSY_COUNTERS=1'
endif

gravity_native_index_classes = custom_target(
  'gravity_apple9_native_index_classes.h',
  input : 'hk_apple9_native_index_shapes.def',
  output : 'hk_apple9_native_index_classes.h',
  command : [prog_python, files('hk_apple9_native_index_hash.py'), '@INPUT@', '@OUTPUT@'],
)

libgravity_vk = static_library(
  'gravity_vk',
  [
    gravity_entrypoints,
    gravity_drirc,
    gravity_native_index_classes,
    gravity_vk_files,
    sha1_h,
  ],
  include_directories : [
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
    inc_gravity,
  ],
  link_with : [libgravity_lib, libgravity_layout, libgravity_compiler],
  c_args : ['-Wno-c2x-extensions'] + gravity_diagnostic_c_args,
  dependencies : [gravity_vk_deps],
  gnu_symbol_visibility : 'hidden',
)

libvulkan_gravity = shared_library(
  'vulkan_gravity',
  link_whole : [libgravity_vk],
  link_args: [ld_args_build_id],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_tests
  test('gravity-queue-completion', executable('gravity-queue-completion',
    ['hk_queue_completion.c', 'tests/hk-queue-completion.c'],
    include_directories : [inc_src],
    dependencies : [idep_mesautil, dep_libdrm],
    link_args : ['-Wl,--wrap=drmSyncobjCreate',
                 '-Wl,--wrap=drmSyncobjDestroy',
                 '-Wl,--wrap=drmSyncobjExportSyncFile',
                 '-Wl,--wrap=drmSyncobjTransfer',
                 '-Wl,--wrap=drmSyncobjWait',
                 '-Wl,--wrap=poll', '-Wl,--wrap=ioctl', '-Wl,--wrap=close',
                 '-Wl,--wrap=realloc', '-Wl,--wrap=calloc']),
    suite : ['gravity'])
  foreach t : ['hk-index-aperture', 'hk-index-memory', 'hk-texture-upload-failure', 'hk-line-stipple-limit']
    gravity_shader_test = executable('gravity-' + t, 'tests/' + t + '.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
      link_args : {'hk-index-aperture': ['-Wl,--wrap=hk_dispatch_precomp'],
                   'hk-index-memory': ['-Wl,--wrap=agx_va_alloc', '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=agx_va_free'],
                   'hk-texture-upload-failure': ['-Wl,--wrap=hk_pool_upload'],
                   'hk-line-stipple-limit': []}[t])
    test('gravity-' + t, gravity_shader_test, suite : ['gravity'])
  endforeach
  test(
    'gravity-hk-heap-oom',
    executable(
      'gravity-hk-heap-oom', 'tests/hk-heap-oom.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
      link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_va_alloc',
                   '-Wl,--wrap=agx_va_free', '-Wl,--wrap=__vk_errorf'],
    ),
    suite : ['gravity'],
  )
  test('gravity-hk-heap-grow', executable('gravity-hk-heap-grow', 'tests/hk-heap-grow.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_bind']),
    suite : ['gravity'])
  test('gravity-hk-cmd-bo-cache', executable('gravity-hk-cmd-bo-cache', 'tests/hk-cmd-bo-cache.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_cache_evict_all']),
    suite : ['gravity'])
  test('gravity-hk-feedback-oom', executable('gravity-hk-feedback-oom', 'tests/hk-feedback-oom.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=hk_meta_decompress_view', '-Wl,--wrap=__vk_errorf']),
    suite : ['gravity'])
  test(
    'gravity-hk-tess-fold-oom',
    executable(
      'gravity-hk-tess-fold-oom', 'tests/hk-tess-fold-oom.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
      link_args : ['-Wl,--wrap=hk_pool_alloc_internal', '-Wl,--wrap=__vk_errorf'],
    ),
    suite : ['gravity'],
  )
  foreach t : ['hk-feedback', 'hk-descriptor-parts', 'hk-constant-data', 'hk-coop-matrix', 'hk-draw-state-cache', 'hk-render-merge', 'hk-empty-clear-barriers', 'hk-empty-render-timestamp', 'hk-tier2-caps', 'hk-image-limits', 'hk-stage-output-limits', 'hk-cull-output-limits', 'hk-pregfx-recording', 'hk-heap-resubmit', 'hk-heap-addresses', 'hk-native-image-store', 'hk-native-store-rw', 'hk-mesh-query', 'hk-native-mesh', 'hk-robust-bounded-vector', 'hk-gs-mesh', 'hk-tes-mesh']
    gravity_contract_test = executable(
      'gravity-' + t,
      'tests/' + t + '.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
      link_args : t in ['hk-heap-addresses', 'hk-heap-resubmit'] ? ['-Wl,--wrap=hk_dispatch_precomp'] : [],
    )
    test(
      'gravity-' + t,
      gravity_contract_test,
      env : t == 'hk-native-mesh' ? ['HK_APPLE9_NATIVE_MESH=1'] : [],
      suite : ['gravity'],
    )
    if t == 'hk-native-mesh'
      test('gravity-hk-native-object-mask', prog_python,
        args : [files('tests/check-native-object-mask.py'), gravity_contract_test],
        suite : ['gravity'])
    endif
  endforeach
  # Corpus inspection is an explicit host tool, not part of the driver.
  executable('gravity-hk-native-mesh-coverage', 'tests/hk-native-mesh-coverage.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk], build_by_default : false)
  test('gravity-hk-native-batch-admits',
       executable('gravity-hk-native-batch-admits', 'tests/hk-native-batch-admits.c',
                  include_directories : [inc_src]),
       suite : ['gravity'])
  test('gravity-hk-native-mesh-open', executable('gravity-hk-native-mesh-open',
    'tests/hk-native-mesh-open.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=drmGetVersion', '-Wl,--wrap=drmFreeVersion',
                 '-Wl,--wrap=drmIoctl', '-Wl,--wrap=agx_native_mesh_probe',
                 '-Wl,--wrap=agx_bo_create']), env : ['HK_APPLE9_NATIVE_INDEX=1'],
    suite : ['gravity'])
  gravity_routing_test = executable(
    'gravity-hk-native-mesh-routing', 'tests/hk-native-mesh-routing.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-native-task-bridge', executable(
    'gravity-hk-native-task-bridge', 'tests/hk-native-task-bridge.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk]),
    suite : ['gravity'])
  gravity_compiler_env_test = executable(
    'gravity-hk-compiler-env', 'tests/hk-compiler-env.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-compiler-env', prog_python,
    args : [files('tests/check-hk-compiler-env.py'), gravity_compiler_env_test],
    suite : ['gravity'],
  )
  test('gravity-hk-shader-memory', executable('gravity-hk-shader-memory', 'tests/hk-shader-memory.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_reference',
                 '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_note_cpu_write']), suite : ['gravity'])
  test('gravity-hk-sliced-3d', executable('gravity-hk-sliced-3d', 'tests/hk-sliced-3d.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk]),
    suite : ['gravity'])
  test('gravity-hk-structural-uploads', executable('gravity-hk-structural-uploads', 'tests/hk-structural-uploads.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_cache_evict_all', '-Wl,--wrap=__vk_errorf']), suite : ['gravity'])
  test('gravity-hk-push-uploads', executable('gravity-hk-push-uploads', 'tests/hk-push-uploads.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_cache_evict_all', '-Wl,--wrap=__vk_errorf']),
    suite : ['gravity'])
  test('gravity-hk-memory', executable('gravity-hk-memory', 'tests/hk-memory.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_cache_evict_all']), suite : ['gravity'])
  test(
    'gravity-hk-fragment-tri-merging',
    executable(
      'gravity-hk-fragment-tri-merging', 'tests/hk-fragment-tri-merging.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
      link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_reference',
                   '-Wl,--wrap=agx_bo_unreference'],
    ),
    suite : ['gravity'],
  )
  test('gravity-hk-native-mesh-routing', gravity_routing_test,
    env : ['HK_APPLE9_NATIVE_MESH=1'],
    suite : ['gravity'])
  test('gravity-hk-native-draw-state', executable('gravity-hk-native-draw-state', 'tests/hk-native-draw-state.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=__vk_errorf']),
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_GS_NATIVE=1',
           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2'],
    suite : ['gravity'])
  test('gravity-hk-native-task-routing', gravity_routing_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_TASK_BRIDGE=1'],
    suite : ['gravity'])
  test('gravity-hk-native-gs-routing', gravity_routing_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_GS_NATIVE=1',
           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2'],
    suite : ['gravity'])
  test('gravity-hk-native-tess-routing', gravity_routing_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_TESS_NATIVE=1',
           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2'],
    suite : ['gravity'])
  gravity_hybrid_test = executable(
    'gravity-hk-native-mesh-hybrid', 'tests/hk-native-mesh-hybrid.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_reference',
                 '-Wl,--wrap=agx_bo_unreference', '-Wl,--wrap=__vk_errorf',
                 '-Wl,--wrap=agx_compile_apple9_native_mesh_lowered',
                 '-Wl,--wrap=agx_compile_apple9_tiny',
                 '-Wl,--wrap=agx_compile_apple9_fragment_mrt'],
  )
  gravity_native_stage_policy = executable(
    'gravity-hk-native-stage-policy', 'tests/hk-native-stage-policy.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_reference',
                 '-Wl,--wrap=agx_bo_unreference', '-Wl,--wrap=agx_bo_cache_evict_all',
                 '-Wl,--wrap=__vk_errorf', '-Wl,--wrap=agx_compile_apple9_native_mesh_lowered',
                 '-Wl,--wrap=agx_compile_apple9_tiny',
                 '-Wl,--wrap=agx_compile_apple9_fragment_mrt'],
  )
  test('gravity-hk-native-stage-policy', prog_python,
    args : [files('tests/check-native-stage-policy.py'), gravity_native_stage_policy],
    suite : ['gravity'])
  test('gravity-hk-native-stage-policy-gs-pack', prog_python,
    args : [files('tests/check-native-stage-policy.py'), gravity_native_stage_policy],
    env : ['HK_APPLE9_NATIVE_GS_PACK=1'],
    suite : ['gravity'])
  test('gravity-hk-native-stage-policy-tes-pack', prog_python,
    args : [files('tests/check-native-stage-policy.py'), gravity_native_stage_policy],
    env : ['HK_APPLE9_NATIVE_TES_PACK=1'],
    suite : ['gravity'])
  gravity_native_index_identity = executable(
    'gravity-hk-native-index-identity', 'tests/hk-native-index-identity.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_reference',
                 '-Wl,--wrap=agx_bo_unreference', '-Wl,--wrap=__vk_errorf',
                 '-Wl,--wrap=agx_compile_apple9_native_mesh_lowered',
                 '-Wl,--wrap=agx_compile_apple9_tiny',
                 '-Wl,--wrap=agx_compile_apple9_fragment_mrt'],
  )
  gravity_native_index_recording = executable(
    'gravity-hk-native-index-recording', 'tests/hk-native-index-recording.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
    link_args : ['-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
                 '-Wl,--wrap=agx_bo_cache_evict_all'],
  )
  foreach mode, controls : {
    'default': [], 'enabled': ['HK_APPLE9_NATIVE_INDEX=1'],
    'env': ['HK_APPLE9_NATIVE_INDEX=0'],
    'drirc': ['HK_APPLE9_NATIVE_INDEX=1'],
    'kernel-absent': ['HK_APPLE9_NATIVE_INDEX=1'],
    'vm-absent': ['HK_APPLE9_NATIVE_INDEX=1'],
  }
    test('gravity-hk-native-index-recording-' + mode, gravity_native_index_recording,
      args : [mode], env : controls, suite : ['gravity'])
  endforeach
  test('gravity-hk-native-gs-hybrid', gravity_hybrid_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_GS_NATIVE=1',
           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2'],
    suite : ['gravity'])
  test('gravity-hk-native-tess-hybrid', gravity_hybrid_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_TESS_NATIVE=1',
           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2'],
    suite : ['gravity'])
  test('gravity-hk-native-mesh-hybrid', gravity_hybrid_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1',
           'HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING=1'],
    suite : ['gravity'],
  )
  test(
    'gravity-hk-native-mesh-options',
    executable(
      'gravity-hk-native-mesh-options', 'tests/hk-native-mesh-options.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
    ),
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1',
           'HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING=1'],
    suite : ['gravity'],
  )
  gravity_native_stage_varyings = executable(
    'gravity-hk-native-stage-varyings', 'tests/hk-native-stage-varyings.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  foreach packed : ['0', '1']
    test('gravity-hk-native-stage-varyings-pack-' + packed, gravity_native_stage_varyings,
         env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_GS_PACK=' + packed,
                'HK_APPLE9_NATIVE_TES_PACK=' + packed,
                'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1,T16V4P2,T32V24P17,T32V32P16'],
         suite : ['gravity'])
  endforeach
  gravity_native_rounds_test = executable(
    'gravity-hk-native-mesh-rounds', 'tests/hk-native-mesh-rounds.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-native-mesh-rounds-off', gravity_native_rounds_test,
    suite : ['gravity'])
  test('gravity-hk-native-mesh-rounds-research', gravity_native_rounds_test, args : ['research'],
    env : ['HK_APPLE9_NATIVE_MESH=1'],
    suite : ['gravity'])
  gravity_native_cull_test = executable(
    'gravity-hk-native-mesh-cull', 'tests/hk-native-mesh-cull.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-native-mesh-cull', gravity_native_cull_test,
    env : ['HK_APPLE9_NATIVE_MESH=1'], suite : ['gravity'])
  gravity_native_lines_test = executable(
    'gravity-hk-native-mesh-lines', 'tests/hk-native-mesh-lines.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-native-mesh-lines', gravity_native_lines_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1'],
    suite : ['gravity'])
  gravity_native_primitive_test = executable(
    'gravity-hk-native-mesh-primitive', 'tests/hk-native-mesh-primitive.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  test('gravity-hk-native-mesh-primitive', gravity_native_primitive_test,
    env : ['HK_APPLE9_NATIVE_MESH=1', 'HK_APPLE9_NATIVE_MESH_ATTRIBUTES=1',
           'HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING=1'],
    suite : ['gravity'])
  gravity_packed_indices_test = executable(
    'gravity-hk-native-packed-indices', 'tests/hk-native-packed-indices.c',
    include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                           inc_src, inc_gravity],
    dependencies : [gravity_vk_deps], link_with : [libgravity_vk],
  )
  foreach mode, selection : {
    'base': '',
    'subset': 'T16V3P2,T16V4P2',
    'capacity': 'T16V16P21,T32V32P42',
    'v16': 'T16V16P16',
    'boolean': '1',
    'unlisted': 'T16V3P2,T16V16P15',
    'off': 'T16V3P2',
  }
    test('gravity-hk-native-packed-indices-' + mode, gravity_packed_indices_test,
      args : [mode],
      env : ['HK_APPLE9_NATIVE_MESH=' + (mode == 'off' ? '0' : '1'),
             'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=' + selection],
      suite : ['gravity'])
  endforeach
  foreach mode, controls : {
    'default': [],
    'enabled': ['HK_APPLE9_NATIVE_INDEX=1'],
    'shape-cannot-widen': ['HK_APPLE9_NATIVE_INDEX=1',
                           'HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V16P16'],
    'killed': ['HK_APPLE9_NATIVE_INDEX=0'],
    'legacy-off': ['HK_APPLE9_NATIVE_INDEX=1', 'HK_APPLE9_NATIVE_MESH=0'],
    'drirc': ['HK_APPLE9_NATIVE_INDEX=1'],
  }
    argument = mode
    if mode == 'shape-cannot-widen'
      argument = 'enabled'
    elif mode == 'legacy-off'
      argument = 'killed'
    endif
    test('gravity-hk-native-index-' + mode, gravity_packed_indices_test,
      args : [argument],
      env : controls, suite : ['gravity'])
  endforeach
  test(
    'gravity_sparse_residency',
    executable(
      'gravity_sparse_residency',
      'tests/hk-sparse-residency.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
    ),
    suite : ['gravity'],
  )
  test(
    'gravity_barrier_recording',
    executable(
      'gravity_barrier_recording',
      'tests/hk_barrier_recording.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
    ),
    suite : ['gravity'],
  )
  test(
    'gravity_cdm_va_alias',
    executable(
      'gravity_cdm_va_alias',
      'tests/hk-cdm-va-alias.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
    ),
    suite : ['gravity'],
  )
  test(
    'gravity_barrier_fixes',
    executable(
      'gravity_barrier_fixes',
      'tests/hk-barrier-fixes.c',
      c_args : gravity_diagnostic_c_args,
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
      link_args : ['-Wl,--wrap=hk_dispatch_precomp'],
    ),
    suite : ['gravity'],
  )
  test(
    'gravity_texel_encode',
    executable(
      'gravity_texel_encode',
      'tests/hk-texel-encode.c',
      include_directories : [inc_gallium, inc_gallium_aux, inc_include,
                             inc_src, inc_gravity],
      dependencies : [gravity_vk_deps],
      link_with : [libgravity_vk],
    ),
    suite : ['gravity'],
  )
  test(
    'gravity_sync_tests',
    executable(
      'gravity_sync_tests',
      ['hk_sync.c', 'tests/hk_sync.cpp'],
      include_directories : [inc_include],
      dependencies : [idep_gtest],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
endif

gravity_icd_file_name = libname_prefix + 'vulkan_gravity.' + libname_suffix

gravity_icd = custom_target(
  'gravity_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'gravity_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', gravity_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

gravity_dev_icdname = 'gravity_devenv_icd.@0@.json'.format(host_machine.cpu())
gravity_dev_icd = custom_target(
  'gravity_dev_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : gravity_dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', gravity_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', gravity_dev_icd.full_path())

if with_tests
  test(
    'gravity_drirc',
    executable(
      'gravity_drirc_test',
      [gravity_drirc, 'tests/hk-drirc.cpp'],
      include_directories : [inc_src, include_directories('.')],
      dependencies : [idep_gtest, idep_mesautil, idep_xmlconfig],
    ),
    env : [
      'DRIRC_CONFIGDIR=' + meson.current_source_dir(),
      'HOME=' + meson.current_build_dir() / 'drirc-empty-home',
    ],
    suite : ['gravity'],
    protocol : 'gtest',
  )

  test(
    'gravity_descriptor_memory',
    executable(
      'gravity_descriptor_memory_test',
      ['hk_descriptor_memory.c', 'hk_usc_heap.c', 'tests/hk-descriptor-memory.cpp'],
      include_directories : [inc_src, inc_include, inc_gravity],
      dependencies : [idep_gtest, idep_mesautil],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
  test(
    'gravity_upload',
    executable(
      'gravity_upload_test',
      'tests/hk-upload.cpp',
      include_directories : [inc_src, include_directories('.')],
      dependencies : [idep_gtest, idep_mesautil],
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
  test(
    'gravity_gpu_busy',
    executable(
      'gravity_gpu_busy_test', 'tests/hk-gpu-busy.cpp',
      include_directories : include_directories('.'),
      dependencies : idep_gtest,
    ),
    suite : ['gravity'],
    protocol : 'gtest',
  )
  test('gravity_gpu_busy_decode', prog_python,
       args : files('tests/hk-gpu-busy-decode.py'),
       suite : ['gravity'])
endif
