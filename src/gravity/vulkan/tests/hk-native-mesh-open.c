/* SPDX-License-Identifier: MIT */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_physical_device.h"
#include "hk_device.h"
#include "hk_cmd_buffer.h"
#include "hk_queue.h"
#include "hk_shader.h"
#include "hk_apple9_mesh.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static jmp_buf stopped;
static unsigned generation, vm_creates;
static uint64_t mock_features;
static struct agx_device *opening_device;

/* Exercise agx_open_device itself, stopping before its first GPU allocation. */
drmVersionPtr __wrap_drmGetVersion(int fd);
drmVersionPtr __wrap_drmGetVersion(int fd)
{
   static drmVersion version = {.name = "asahi"};
   CHECK(fd == 101);
   return &version;
}
void __wrap_drmFreeVersion(drmVersionPtr version);
void __wrap_drmFreeVersion(drmVersionPtr version) { (void)version; }
int __wrap_drmIoctl(int fd, unsigned long request, void *arg);
int __wrap_drmIoctl(int fd, unsigned long request, void *arg)
{
   CHECK(fd == 101);
   if (request == DRM_IOCTL_ASAHI_GET_PARAMS) {
      struct drm_asahi_get_params *get = arg;
      struct drm_asahi_params_global *p = (void *)(uintptr_t)get->pointer;
      CHECK(get->size == sizeof(*p));
      CHECK(p == &opening_device->params && opening_device->chip == 0);
      *p = (struct drm_asahi_params_global){
         .features = mock_features,
         .gpu_generation = generation, .gpu_variant = generation == 17 ? 'P' : 'G',
         .vm_start = 0, .vm_end = 1ull << 42, .vm_kernel_min_size = 1ull << 26,
         .command_timestamp_frequency_hz = 24000000,
      };
      return 0;
   }
   CHECK(request == DRM_IOCTL_ASAHI_VM_CREATE);
   ((struct drm_asahi_vm_create *)arg)->vm_id = 9;
   vm_creates++;
   return 0;
}
struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                                   enum agx_bo_flags flags, const char *label);
struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                                   enum agx_bo_flags flags, const char *label)
{
   (void)size; (void)align; (void)flags; (void)label;
   CHECK(vm_creates == 1 && dev->vm_id == 9);
   longjmp(stopped, 1);
}

static uint64_t
check(unsigned gen, uint64_t features, bool expected_native, bool kill)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(pdev);
   pdev->dev.fd = 101;
   pdev->disable_native_index = kill;
   opening_device = &pdev->dev;
   generation = gen;
   mock_features = features;
   vm_creates = 0;
   if (!setjmp(stopped)) {
      agx_open_device(NULL, &pdev->dev);
      CHECK(!"device open returned before allocation boundary");
   }
   CHECK(agx_has_native_mesh(&pdev->dev) == expected_native);
   CHECK(hk_apple9_native_index_enabled(pdev) == (expected_native && !kill));
   CHECK(hk_apple9_native_mesh_available(pdev) == (expected_native && !kill));
   uint64_t flags = hk_physical_device_compiler_flags(pdev);
   CHECK(!pdev->dev.bo_cache.started);
   agx_bo_cache_finish(&pdev->dev);
   util_sparse_array_finish(&pdev->dev.bo_map);
   pthread_mutex_destroy(&pdev->dev.bo_map_lock);
   simple_mtx_destroy(&pdev->dev.vma_lock);
   simple_mtx_destroy(&pdev->dev.apple9_entry_lock);
   util_vma_heap_finish(&pdev->dev.main_heap);
   util_vma_heap_finish(&pdev->dev.replay_heap);
   util_vma_heap_finish(&pdev->dev.usc_heap);
   util_vma_heap_finish(&pdev->dev.replay_usc_heap);
   util_vma_heap_finish(&pdev->dev.context_heap);
   util_vma_heap_finish(&pdev->dev.apple9_entry_heap);
   glsl_type_singleton_decref();
   free(pdev);
   opening_device = NULL;
   return flags;
}

static void
check_vdm_render_flags(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cs *cs = calloc(1, sizeof(*cs));
   CHECK(dev && cs);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.params.features =
      DRM_ASAHI_FEATURE_FEW_PRIMITIVES | DRM_ASAHI_FEATURE_NATIVE_MESH;
   cs->addr = 0x1000000000ull;
   cs->cr.width = 64;
   cs->cr.height = 64;
   cs->cr.zls_width = 64;
   cs->cr.zls_height = 64;
   cs->cr.layers = 1;
   cs->primitive_count = 1;

   /* Ordinary few-primitive render sets FEW_PRIMITIVES and NOT NATIVE_MESH. */
   struct drm_asahi_cmd_render cmd;
   cs->apple9_native_mesh = false;
   hk_fill_vdm_command(dev, cs, &cmd);
   CHECK((cmd.flags & DRM_ASAHI_RENDER_FEW_PRIMITIVES) != 0);
   CHECK((cmd.flags & DRM_ASAHI_RENDER_NATIVE_MESH) == 0);

   /* Native-mesh render sets NATIVE_MESH (bit 28) and NEVER FEW_PRIMITIVES,
    * even when primitive_count == 1 and FEW_PRIMITIVES is advertised. */
   cs->apple9_native_mesh = true;
   hk_fill_vdm_command(dev, cs, &cmd);
   CHECK((cmd.flags & DRM_ASAHI_RENDER_NATIVE_MESH) != 0);
   CHECK((cmd.flags & DRM_ASAHI_RENDER_FEW_PRIMITIVES) == 0);

   free(cs);
   free(dev);
}

int main(void)
{
   uint64_t positive = check(16, DRM_ASAHI_FEATURE_NATIVE_MESH, true, false);
   uint64_t negative = check(16, 0, false, false);
   CHECK((positive ^ negative) == (1ull << 60));
   CHECK(check(16, DRM_ASAHI_FEATURE_NATIVE_MESH, true, true) == negative);
   CHECK(check(16, DRM_ASAHI_FEATURE_FEW_PRIMITIVES | DRM_ASAHI_FEATURE_FRAGMENT_BARRIER,
               false, false) == negative);
   /* Non-G16G chips do not enable native mesh even if bit 28 is set. */
   CHECK(check(17, DRM_ASAHI_FEATURE_NATIVE_MESH, false, false) == negative);
   check_vdm_render_flags();
   puts("PASS: bit-28 feature detection, compiler bit 60, VDM flag mutual exclusion; GPU execution=0");
   return 0;
}
