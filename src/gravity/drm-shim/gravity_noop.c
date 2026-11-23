/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

/* A DRM shim that accepts every submission without a GPU. It lets host CI
 * run the Gravity Vulkan driver through its CPU-side paths (shader
 * compilation, state emission, command stream construction). Rendered
 * results are meaningless.
 *
 * Reports an Apple M4 (G16G / T8132) GPU by default.
 */

#include <stdlib.h>
#include <string.h>

#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

static struct drm_asahi_params_global params = {
   .gpu_generation = 16,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .chip_id = 0x8132,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .max_frequency_khz = 1278000,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   /* 42-bit UAT with the fixed 1-TiB USC aperture, and a kernel-private
    * heap inside the compact render context. */
   .vm_end = (1ull << 42) - 0x4000,
   .vm_kernel_min_size = 0x10000000,
   .max_commands_per_submission = 64,
   .max_attachments = 64,
   .command_timestamp_frequency_hz = 1000000000,
};

struct asahi_bo {
   struct shim_bo base;
};

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_get_params(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   if (gp->param_group != 0)
      return -EINVAL;

   memcpy((void *)(uintptr_t)gp->pointer, &params,
          MIN2(gp->size, sizeof(params)));
   return 0;
}

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   static uint32_t next_vm = 1;
   struct drm_asahi_vm_create *create = arg;
   create->vm_id = __atomic_fetch_add(&next_vm, 1, __ATOMIC_RELAXED);
   return 0;
}

static int
asahi_ioctl_queue_create(int fd, unsigned long request, void *arg)
{
   static uint32_t next_queue = 1;
   struct drm_asahi_queue_create *create = arg;
   create->queue_id = __atomic_fetch_add(&next_queue, 1, __ATOMIC_RELAXED);
   return 0;
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return -ENOMEM;

   int ret = drm_shim_bo_init(&bo->base, create->size);
   if (ret) {
      free(bo);
      return ret;
   }

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);
   drm_shim_bo_put(&bo->base);
   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);
   if (!bo)
      return -ENOENT;

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);
   drm_shim_bo_put(bo);
   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_params,
   [DRM_ASAHI_GET_TIME] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_noop,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_noop,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_queue_create,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_noop,
};

void
drm_shim_driver_init(void)
{
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   drm_shim_platform_device_setup("asahi", "/soc/agx", "apple,gpu-g16g");
}
