#include <sys/ioctl.h>
#include <linux/sync_file.h>
/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * Copyright 2020 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_wait_watchdog.h"
#include "agx_device.h"
#include "agx_native_mesh.h"
#include "agx_apple9_layout.h"
#include <inttypes.h>
#include "clc/gravity_clc.h"
#include "drm-uapi/asahi_drm.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/timespec.h"
#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "glsl_types.h"
#include "layout.h"
#include "libagx_dgc.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include "drm-uapi/dma-buf.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "util/os_file.h"
#include "util/os_mman.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"
#include "util/u_printf.h"

static void agx_finish_device(struct agx_device *dev);

static inline int
asahi_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req)
{
   return U_WATCHDOG_IOCTL("asahi_simple_ioctl", dev->fd,
                           drmIoctl(dev->fd, cmd, req));
}

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"bodump",        AGX_DBG_BODUMP,        "Periodically dump live BOs"},
   {"perf",          AGX_DBG_PERF,          "Print performance warnings"},
   {"nocompress",    AGX_DBG_NOCOMPRESS,    "Disable lossless compression"},
   {"nocluster",     AGX_DBG_NOCLUSTER,     "Disable vertex clustering"},
   {"sync",          AGX_DBG_SYNC,          "Synchronously wait for all submissions"},
   {"1queue",        AGX_DBG_1QUEUE,        "Force usage of a single queue for multiple contexts"},
   {"nosoft",        AGX_DBG_NOSOFT,        "Disable soft fault optimizations"},
   {"nomerge",       AGX_DBG_NOMERGE,       "Disable control stream merging"},
   {"bodumpverbose", AGX_DBG_BODUMPVERBOSE, "Include extra info with dumps"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

static int
agx_drm_bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
                uint32_t count)
{
   struct drm_asahi_vm_bind vm_bind = {
      .num_binds = count,
      .vm_id = dev->vm_id,
      .userptr = (uintptr_t)ops,
      .stride = sizeof(*ops),
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_VM_BIND, &vm_bind);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_BIND failed: %m\n");
   }

   return ret;
}

/*
 * Convenience helper to bind a single BO regardless of kernel module.
 */
int
agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
            size_t size_B, uint64_t offset_B, uint32_t flags)
{
   assert((size_B % 16384) == 0 && "alignment required");
   assert((offset_B % 16384) == 0 && "alignment required");
   assert((addr % 16384) == 0 && "alignment required");

   struct drm_asahi_gem_bind_op op = {
      .flags = flags,
      .handle = bo ? bo->uapi_handle : 0,
      .offset = offset_B,
      .range = size_B,
      .addr = addr,
   };

   return dev->ops.bo_bind(dev, &op, 1);
}

int
agx_bind_timestamps(struct agx_device *dev, struct agx_bo *bo, uint32_t *handle)
{
   struct drm_asahi_gem_bind_object bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_BIND,
      .flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS,
      .handle = bo->uapi_handle,
      .range = bo->size,
   };

   int ret = dev->ops.bo_bind_object(dev, &bind);
   *handle = bind.object_handle;
   return ret;
}

static struct agx_bo *
agx_bo_alloc(struct agx_device *dev, size_t size, size_t align,
             enum agx_bo_flags flags,
             enum agx_va_flags extra_va_flags, uint64_t fixed_va)
{
   struct agx_bo *bo;
   unsigned handle = 0;

   /*
    * AGX_BO_EXEC describes contents, not addressability.  Compact Apple8/9
    * users request LOW_VA explicitly. Apple9 shader bodies and dispatch
    * state use independent allocations alongside the stable entry arena.
    */

   struct drm_asahi_gem_create gem_create = {.size = size};

   if (flags & AGX_BO_WRITEBACK)
      gem_create.flags |= DRM_ASAHI_GEM_WRITEBACK;

   if (!(flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))) {
      gem_create.flags |= DRM_ASAHI_GEM_VM_PRIVATE;
      gem_create.vm_id = dev->vm_id;
   }

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_CREATE, &gem_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GEM_CREATE failed: %m\n");
      return NULL;
   }

   handle = gem_create.handle;

   pthread_mutex_lock(&dev->bo_map_lock);
   bo = agx_lookup_bo(dev, handle);
   dev->max_handle = MAX2(dev->max_handle, handle);
   pthread_mutex_unlock(&dev->bo_map_lock);

   /* Fresh handle */
   assert(!memcmp(bo, &((struct agx_bo){}), sizeof(*bo)));

   bo->dev = dev;
   bo->size = gem_create.size;
   bo->align = align;
   bo->flags = flags;
   bo->handle = bo->uapi_handle = handle;
   bo->prime_fd = -1;

   assert(!(flags & AGX_BO_LOW_VA) || !(flags & AGX_BO_CONTEXT));
   enum agx_va_flags va_flags = flags & AGX_BO_CONTEXT  ? AGX_VA_CONTEXT
                                : flags & AGX_BO_LOW_VA ? AGX_VA_USC
                                                        : 0;
   bo->va = agx_va_alloc(dev, size, bo->align, va_flags | extra_va_flags,
                         fixed_va);
   if (!bo->va) {
      fprintf(stderr, "Failed to allocate BO VMA\n");
      agx_bo_free(dev, bo);
      return NULL;
   }

   uint32_t bind = DRM_ASAHI_BIND_READ;
   if (!(flags & AGX_BO_READONLY)) {
      bind |= DRM_ASAHI_BIND_WRITE;
   }

   ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0, bind);
   if (ret) {
      agx_bo_free(dev, bo);
      return NULL;
   }

   return bo;
}

/* BO mappings are MAP_SHARED views of GPU memory. A forked child inherits
 * them at the same addresses but none of the driver state that owns them,
 * and the parent may later reuse those addresses for other BOs: any write
 * the child makes there, through its copy of the parent's heap pointers or
 * an allocator reusing the range, lands in live GPU memory such as shader
 * code. A GPU mapping is never valid to use in a child, so leave them out
 * of fork(). A failure only loses this protection.
 */
void
agx_bo_mapping_dontfork(void *map, size_t size)
{
#ifdef MADV_DONTFORK
   if (madvise(map, size, MADV_DONTFORK))
      mesa_logw_once("madvise(MADV_DONTFORK) on a BO mapping failed: %s",
                     strerror(errno));
#endif
}

static void *
agx_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
            size_t size)
{
   assert(size <= bo->size);

   struct drm_asahi_gem_mmap_offset gem_mmap_offset = {.handle =
                                                          bo->uapi_handle};
   int ret, flags;

   ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &gem_mmap_offset);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_MMAP_BO failed: %m\n");
      return NULL;
   }

   flags = MAP_SHARED | (fixed_addr ? MAP_FIXED : 0);
   void *map = os_mmap(fixed_addr, size, PROT_READ | PROT_WRITE, flags,
                       dev->fd, gem_mmap_offset.offset);
   if (map == MAP_FAILED) {
      fprintf(stderr, "mmap failed: size=0x%llx fd=%i offset=0x%llx %m\n",
              (long long)size, dev->fd, (long long)gem_mmap_offset.offset);
      return NULL;
   }

   agx_bo_mapping_dontfork(map, size);
   return map;
}

struct agx_bo *
agx_bo_import(struct agx_device *dev, int fd)
{
   struct agx_bo *bo;
   ASSERTED int ret;
   unsigned gem_handle;

   /* Validate before importing a handle, so invalid sizes cannot leak a GEM
    * reference (or leave a half-initialized entry for the reaper/importer). */
   off_t size = lseek(fd, 0, SEEK_END);
   if (size <= 0 || (size & (AIL_PAGESIZE - 1))) {
      fprintf(stderr, "import failed: invalid BO size 0x%llx\n",
              (long long)size);
      return NULL;
   }

   pthread_mutex_lock(&dev->bo_map_lock);

retry:
   ret = drmPrimeFDToHandle(dev->fd, fd, &gem_handle);
   if (ret) {
      fprintf(stderr, "import failed: Could not map fd %d to handle\n", fd);
      pthread_mutex_unlock(&dev->bo_map_lock);
      return NULL;
   }

   bo = agx_lookup_bo(dev, gem_handle);
   dev->max_handle = MAX2(dev->max_handle, gem_handle);

   if (!bo->size && dev->bo_closing_handle == gem_handle) {
      /* The kernel may still return the old handle before the reaper's
       * close removes it. Wait without blocking allocation or other imports,
       * then import again. If allocation has already reused the handle, its
       * nonempty slot describes the new object and needs no such wait.
       */
      pthread_cond_wait(&dev->bo_map_cond, &dev->bo_map_lock);
      goto retry;
   }

   if (!bo->size) {
      int prime_fd = os_dupfd_cloexec(fd);
      if (prime_fd < 0) {
         fprintf(stderr, "import failed: Could not dup prime fd %d: %m\n", fd);
         struct drm_gem_close close_req = {.handle = gem_handle};
         drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &close_req);
         pthread_mutex_unlock(&dev->bo_map_lock);
         return NULL;
      }

      struct agx_va *va = agx_va_alloc(dev, size, AIL_PAGESIZE, 0, 0);
      if (!va) {
         fprintf(
            stderr,
            "import failed: Could not allocate from VMA heap (0x%llx bytes)\n",
            (long long)size);
         close(prime_fd);
         struct drm_gem_close close_req = {.handle = gem_handle};
         drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &close_req);
         pthread_mutex_unlock(&dev->bo_map_lock);
         return NULL;
      }

      bo->dev = dev;
      bo->size = size;
      bo->align = AIL_PAGESIZE;

      bo->flags = AGX_BO_SHARED | AGX_BO_SHAREABLE;
      bo->handle = gem_handle;
      bo->uapi_handle = gem_handle;
      bo->prime_fd = prime_fd;
      bo->label = "Imported BO";
      bo->va = va;
      p_atomic_set(&bo->refcnt, 1);

      ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0,
                        DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret) {
         fprintf(stderr, "import failed: Could not bind BO at 0x%llx\n",
                 (long long)bo->va->addr);
         agx_va_free(dev, va, false);
         close(prime_fd);
         memset(bo, 0, sizeof(*bo));
         struct drm_gem_close close_req = {.handle = gem_handle};
         drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &close_req);
         pthread_mutex_unlock(&dev->bo_map_lock);
         return NULL;
      }
   } else {
      /* Final unreference, resurrection and reaper removal all serialize
       * on bo_map_lock. A queued shared BO still owns its handle and VA. */
      if (p_atomic_read(&bo->refcnt) == 0)
         agx_bo_reclaim(dev, bo);
      else
         agx_bo_reference(bo);

      /* If this bo came back to us via import, it had better
       * been marked shared to begin with.
       */
      if (bo->shim_cpu_epoch)
         *bo->shim_cpu_epoch = 0;
      assert(bo->flags & AGX_BO_SHAREABLE);
      assert(bo->flags & AGX_BO_SHARED);
      assert(bo->prime_fd != -1);
   }
   pthread_mutex_unlock(&dev->bo_map_lock);

   assert(bo->dev != NULL && "post-condition");

   return bo;
}

bool
agx_bo_make_shared(struct agx_device *dev, struct agx_bo *bo)
{
   assert(bo->flags & AGX_BO_SHAREABLE);
   if (bo->flags & AGX_BO_SHARED) {
      assert(bo->prime_fd >= 0);
      return true;
   }

   assert(bo->prime_fd == -1);

   int prime_fd = -1;
   int ret = drmPrimeHandleToFD(dev->fd, bo->handle, DRM_CLOEXEC | DRM_RDWR,
                                &prime_fd);
   if (ret || prime_fd < 0)
      return false;

   bo->prime_fd = prime_fd;
   bo->flags |= AGX_BO_SHARED;

   /* If there is a pending writer to this BO, import it into the buffer
    * for implicit sync.
    */
   uint64_t writer = p_atomic_read_relaxed(&bo->writer);
   if (writer) {
      int out_sync_fd = -1;
      int ret = drmSyncobjExportSyncFile(dev->fd, agx_bo_writer_syncobj(writer),
                                         &out_sync_fd);
      assert(ret >= 0);
      assert(out_sync_fd >= 0);

      ret = agx_import_sync_file(dev, bo, out_sync_fd);
      assert(ret >= 0);
      close(out_sync_fd);
   }

   return true;
}

int
agx_bo_export(struct agx_device *dev, struct agx_bo *bo)
{
   if (!agx_bo_make_shared(dev, bo))
      return -1;

   assert(bo->prime_fd >= 0);
   return os_dupfd_cloexec(bo->prime_fd);
}

static int
agx_bo_bind_object(struct agx_device *dev,
                   struct drm_asahi_gem_bind_object *bind)
{
   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (handle=%d)\n",
              bind->handle);
   }

   return ret;
}

static int
agx_bo_unbind_object(struct agx_device *dev, uint32_t object_handle)
{
   struct drm_asahi_gem_bind_object gem_bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_UNBIND,
      .object_handle = object_handle,
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &gem_bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (object_handle=%d)\n",
              object_handle);
   }

   return ret;
}

static ssize_t
agx_get_params(struct agx_device *dev, void *buf, size_t size)
{
   struct drm_asahi_get_params get_param = {
      .param_group = 0,
      .pointer = (uint64_t)(uintptr_t)buf,
      .size = size,
   };

   memset(buf, 0, size);

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GET_PARAMS, &get_param);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_PARAMS failed: %m\n");
      return -EINVAL;
   }

   return get_param.size;
}

static int
agx_submit(struct agx_device *dev, struct drm_asahi_submit *submit)
{
   return U_WATCHDOG_IOCTL("DRM_IOCTL_ASAHI_SUBMIT", dev->fd,
                           drmIoctl(dev->fd, DRM_IOCTL_ASAHI_SUBMIT, submit));
}

const agx_device_ops_t agx_device_drm_ops = {
   .bo_alloc = agx_bo_alloc,
   .bo_bind = agx_drm_bo_bind,
   .bo_mmap = agx_bo_mmap,
   .get_params = agx_get_params,
   .submit = agx_submit,
   .bo_bind_object = agx_bo_bind_object,
   .bo_unbind_object = agx_bo_unbind_object,
};

static uint64_t
gcd(uint64_t n, uint64_t m)
{
   while (n != 0) {
      uint64_t remainder = m % n;
      m = n;
      n = remainder;
   }

   return m;
}

static void
agx_init_timestamps(struct agx_device *dev)
{
   uint64_t user_ts_gcd =
      gcd(dev->params.command_timestamp_frequency_hz, NSEC_PER_SEC);

   dev->user_timestamp_to_ns.num = NSEC_PER_SEC / user_ts_gcd;
   dev->user_timestamp_to_ns.den =
      dev->params.command_timestamp_frequency_hz / user_ts_gcd;
}

bool
agx_open_device(void *memctx, struct agx_device *dev)
{
   dev->debug =
      debug_get_flags_option("ASAHI_MESA_DEBUG", agx_debug_options, 0);

   dev->ops = agx_device_drm_ops;

   ssize_t params_size = -1;

   /* DRM version check */
   {
      drmVersionPtr version = drmGetVersion(dev->fd);
      if (!version) {
         fprintf(stderr, "cannot get version: %s", strerror(errno));
         return false;
      }

      bool is_asahi = !strcmp(version->name, "asahi");
      drmFreeVersion(version);
      if (!is_asahi)
         return false;

      dev->ops = agx_device_drm_ops;
   }

   params_size = dev->ops.get_params(dev, &dev->params, sizeof(dev->params));
   if (params_size <= 0) {
      assert(0);
      return false;
   }
   assert(params_size >= sizeof(dev->params));

   /* Gravity targets Apple9 (G16+) GPUs unconditionally. */
   if (dev->params.gpu_generation < 16)
      return false;

   /* Identify the chip before any helpers consume device state. */
   switch (dev->params.gpu_generation) {
   case 16:
      if (dev->params.gpu_variant != 'G' &&
          dev->params.gpu_variant != 'S' &&
          dev->params.gpu_variant != 'C')
         return false;
      /* Pro shares the multi-cluster Max graphics path and ISA. */
      dev->chip = dev->params.gpu_variant != 'G' ? AGX_CHIP_G16C
                                                 : AGX_CHIP_G16G;
      break;
   case 17:
      if (dev->params.gpu_variant != 'P')
         return false;
      dev->chip = AGX_CHIP_G17P;
      break;
   default:
      return false;
   }

   const char *variant = " Unknown";
   switch (dev->params.gpu_variant) {
   case 'G':
      variant = "";
      break;
   case 'S':
      variant = " Pro";
      break;
   case 'P':
      /* Apple9 firmware reports the Pro variant as P. */
      variant = " Pro";
      break;
   case 'C':
      variant = " Max";
      break;
   case 'D':
      variant = " Ultra";
      break;
   }
   snprintf(dev->name, sizeof(dev->name), "Apple M%d%s (G%d%c %02X)",
            dev->params.gpu_generation - 12, variant,
            dev->params.gpu_generation, dev->params.gpu_variant,
            dev->params.gpu_revision + 0xA0);

   /* We need a large chunk of VA space carved out for robustness. Hardware
    * loads can shift an i32 by up to 2, for a total shift of 4. If the base
    * address is zero, 36-bits is therefore enough to trap any zero-extended
    * 32-bit index. For more generality we would need a larger carveout, but
    * this is already optimal for VBOs.
    *
    * TODO: Maybe this should be on top instead? Might be ok.
    */
   uint64_t reservation = (1ull << 36);

   /* Also reserve VA space for the printf buffer at a stable address, avoiding
    * the need for relocs in precompiled shaders.
    */
   assert(reservation == LIBAGX_PRINTF_BUFFER_ADDRESS);
   reservation += LIBAGX_PRINTF_BUFFER_SIZE;

   dev->guard_size = AIL_PAGESIZE;
   uint64_t default_shader_base =
      ALIGN_POT(MAX2(dev->params.vm_start, reservation), 0x100000000ull);

   /* Apple9 compact shader references resolve through the fixed native 1-TiB
    * USC aperture on both G16 and G17.  The 42-bit UAT leaves room for the
    * complete 4-GiB heap there, so use the hardware address directly.
    */
   dev->shader_base = 1ull << 40;

   if (dev->shader_base < reservation) {
      /* Our robustness implementation requires the bottom unmapped */
      fprintf(stderr, "Unexpected address layout, can't cope\n");
      assert(0);
      return false;
   }

   uint64_t shader_size = 0x100000000ull;
   /* Apple9 leaves the ordinary resource/descriptor heap below the fixed USC
    * aperture. Executable/AGX_BO_LOW_VA allocations use usc_heap.
    */
   uint64_t user_start = default_shader_base;

   assert(dev->shader_base >= dev->params.vm_start);
   assert(dev->shader_base + shader_size < dev->params.vm_end);
   assert(user_start < dev->params.vm_end);

   agx_init_timestamps(dev);

   util_sparse_array_init(&dev->bo_map, sizeof(struct agx_bo), 512);
   pthread_mutex_init(&dev->bo_map_lock, NULL);

   agx_bo_cache_init(dev);

   /* The generic ABI puts a generous kernel-private heap at the top of the
    * VM.  G16 render work instead addresses that backend-owned state relative
    * to the fixed 0x1000000000 context and has a 4-GiB reach.  Keep its UAPI
    * carveout after the small fixed client graph and before the ordinary Mesa
    * heap, with exactly the advertised minimum capacity. */
   uint64_t kernel_size = MAX2(dev->params.vm_kernel_min_size, 32ull << 30);
   struct drm_asahi_vm_create vm_create = {0};
   if (dev->params.gpu_generation == 16 &&
       (dev->params.gpu_variant == 'G' || dev->params.gpu_variant == 'S' ||
        dev->params.gpu_variant == 'C')) {
      kernel_size = dev->params.vm_kernel_min_size;
      vm_create.kernel_end =
         AGX_APPLE9_RENDER_CONTEXT_BASE + (1ull << 32);
      vm_create.kernel_start = vm_create.kernel_end - kernel_size;
      /* PPP and VDM records share the same 4-GiB aperture, between the low
       * 64 MiB of fixed aliases and the kernel heap. Keep at least 1 GiB for
       * them rather than wrapping or starving that heap. */
      if (kernel_size < 0x4000000ull ||
          kernel_size > (1ull << 32) - 0x4000000ull - (1ull << 30)) {
         fprintf(stderr,
                 "G16 kernel-private heap (%" PRIu64 " MiB) does not fit its "
                 "compact render context with 1 GiB for render records\n",
                 kernel_size >> 20);
         return false;
      }
   } else {
      vm_create.kernel_start = dev->params.vm_end - kernel_size;
      vm_create.kernel_end = dev->params.vm_end;
   }

   /* Keep Apple9's ordinary heap and its sparse read-only mirror below the
    * 512-GiB ceiling. The separate USC heap remains at 1 TiB and is not part
    * of this partition.
    */
   uint64_t user_end = 1ull << 39;
   assert(user_start < user_end);
   uint64_t user_size = user_end - user_start;

   /* Only native G16G VMs can qualify for the current mesh profile. */
   const char *classes = dev->params.gpu_generation == 16 &&
                         dev->params.gpu_variant == 'G'
                            ? dev->native_mesh_classes : NULL;
   struct agx_native_mesh_support before =
      agx_native_mesh_probe(dev->fd, classes);
   struct agx_native_blocks_support blocks_before =
      classes ? agx_native_blocks_probe(dev->fd)
              : (struct agx_native_blocks_support){0};
   dev->native_mesh_qualified = false;
   dev->native_blocks = (struct agx_native_blocks_support){0};
   int ret = asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_VM_CREATE, &vm_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_CREATE failed: %m\n");
      assert(0);
      return false;
   }

   /* The kernel latches profile+generation at VM_CREATE and retains that
    * profile for the VM lifetime. A concurrent mode change fails closed. */
   struct agx_native_mesh_support after =
      agx_native_mesh_probe(dev->fd, classes);
   dev->native_mesh_qualified =
      agx_native_mesh_same_qualification(before, after);

   /* Batched native blocks are a property of the same latched profile: both
    * attributes, sampled around VM_CREATE, must agree on one generation. */
   struct agx_native_blocks_support blocks_after =
      classes ? agx_native_blocks_probe(dev->fd)
              : (struct agx_native_blocks_support){0};
   if (dev->native_mesh_qualified &&
       agx_native_blocks_same(blocks_before, blocks_after) &&
       blocks_before.generation == before.generation)
      dev->native_blocks = blocks_before;

   /* Round the user VA window to powers-of-two... */
   user_start = util_next_power_of_two64(user_start);
   user_size = util_next_power_of_two64(user_size + 1) >> 1;

   /* ...so when we cut user size in half to emulate sparse buffers... */
   user_size /= 2;

   /* ...or maybe in quarters if necessary to disambiguate */
   if (user_size == user_start) {
      user_size /= 2;
   }

   /* ...we can distinguish the top/bottom half by an address bit */
   dev->sparse_ro_offset = user_size;
   assert((user_start & dev->sparse_ro_offset) == 0);
   assert(((user_start + (user_size - 1)) & dev->sparse_ro_offset) == 0);

   simple_mtx_init(&dev->vma_lock, mtx_plain);
   /* Capture-replay allocations (VK_*_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) come
    * from the top of each heap, which nothing else uses: a captured address
    * is then free at replay, whatever the replaying driver allocated first.
    * Only devices created for capture or replay reserve them.
    */
   uint64_t replay_size =
      dev->capture_replay ? MIN2(user_size / 8, 16ull << 30) : 0;
   uint64_t replay_usc_size = dev->capture_replay ? shader_size / 8 : 0;
   util_vma_heap_init(&dev->main_heap, user_start, user_size - replay_size);
   util_vma_heap_init(&dev->replay_heap, user_start + user_size - replay_size,
                      replay_size);
   util_vma_heap_init(&dev->usc_heap, dev->shader_base,
                      shader_size - replay_usc_size);
   util_vma_heap_init(&dev->replay_usc_heap,
                      dev->shader_base + shader_size - replay_usc_size,
                      replay_usc_size);
   /* The low 64 MiB contain existing fixed context/firmware aliases. The
    * kernel owns the top of this aperture on native G16. PPP and VDM records
    * occupy the remaining interval and are retained by their submitting batch.
    * Keep this heap disjoint from main_heap and the fixed USC aperture. */
   /* G17P's source-owned initial tiler-growth tranche occupies offsets
    * [0x02000000, 0x05200000). Keep caller command/PPP state above it;
    * later growth tranches are selected around existing caller mappings. */
   uint64_t context_start = AGX_APPLE9_RENDER_CONTEXT_BASE +
      (dev->params.gpu_generation == 17 ? 0x8000000ull : 0x4000000ull);
   uint64_t context_end =
      MIN2(vm_create.kernel_start,
           AGX_APPLE9_RENDER_CONTEXT_BASE + (1ull << 32));
   util_vma_heap_init(&dev->context_heap, context_start,
                      context_end - context_start);
   dev->context_heap.alloc_high = false;
   agx_va_quarantine_init(dev, user_size - replay_size,
                          shader_size - replay_usc_size,
                          context_end - context_start);

   /* The compact Apple9 entry arena must start at the USC base. Allocate
    * it first from the bottom of the heap; shader bodies and batch state can
    * use the remaining 4-GiB aperture. Resource tables use full addresses.
    */
   dev->usc_heap.alloc_high = false;

   dev->vm_id = vm_create.vm_id;

   glsl_type_singleton_init_or_ref();

   simple_mtx_init(&dev->apple9_entry_back_lock, mtx_plain);
   simple_mtx_init(&dev->apple9_entry_lock, mtx_plain);
   list_inithead(&dev->apple9_entry_lru);
   util_vma_heap_init(&dev->apple9_entry_heap, AGX_APPLE9_ENTRY_HEADER_SIZE,
                     AGX_APPLE9_ENTRY_ARENA_SIZE - AGX_APPLE9_ENTRY_HEADER_SIZE);
   dev->apple9_entry_heap.alloc_high = false;

   /* Entry instructions are CPU-authored and GPU-read-only. The same mapping
    * remains installed for every graphics and compute job. Reserve the arena
    * and back its first chunk, which holds the header; later chunks are backed
    * as entries reach them. */
   if (!dev->probe_only) {
      dev->apple9_entry_va =
         agx_va_alloc(dev, AGX_APPLE9_ENTRY_ARENA_SIZE,
                      AGX_APPLE9_ENTRY_ARENA_SIZE, AGX_VA_USC | AGX_VA_FIXED,
                      dev->shader_base);
      if (!dev->apple9_entry_va) {
         fprintf(stderr, "Failed to reserve compact Apple9 shader entries\n");
         goto fail;
      }
      if (!agx_apple9_entry_back(dev, AGX_APPLE9_ENTRY_HEADER_SIZE)) {
         fprintf(stderr, "Failed to back compact Apple9 shader entries\n");
         goto fail;
      }
   }

   /* Bind read-only zero page at 2^32. This is in our reservation, and can be
    * addressed with only small integers in the low/high. That lets us do some
    * robustness optimization even without soft fault.
    */
   {
      struct agx_bo *bo = agx_bo_create(dev, 16384, 0, 0, "Zero page");
      if (!bo) {
         fprintf(stderr, "Failed to allocate zero page\n");
         goto fail;
      }
      int ret = agx_bo_bind(dev, bo, AGX_ZERO_PAGE_ADDRESS, 16384, 0,
                            DRM_ASAHI_BIND_READ);
      if (ret) {
         fprintf(stderr, "Failed to bind zero page\n");
         agx_bo_unreference(dev, bo);
         goto fail;
      }

      dev->zero_bo = bo;
   }

   {
      struct agx_bo *bo = agx_bo_create(dev, AIL_PAGESIZE, 0, 0, "Scratch page");
      if (!bo) {
         fprintf(stderr, "Failed to allocate scratch page\n");
         goto fail;
      }
      int ret = agx_bo_bind(dev, bo, AGX_SCRATCH_PAGE_ADDRESS, AIL_PAGESIZE, 0,
                            DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret) {
         fprintf(stderr, "Failed to bind scratch page\n");
         agx_bo_unreference(dev, bo);
         goto fail;
      }

      void *map = agx_bo_map(bo);
      if (!map) {
         fprintf(stderr, "Failed to map scratch page\n");
         agx_bo_unreference(dev, bo);
         goto fail;
      }

      dev->scratch_bo = bo;

      /* The contents of the scratch page are undefined, but making them nonzero
       * helps fuzz for bugs where we incorrectly read from the write section.
       */
      memset(map, 0xCA, AIL_PAGESIZE);
   }

   struct agx_bo *bo = agx_bo_create(dev, LIBAGX_PRINTF_BUFFER_SIZE, 0,
                                     AGX_BO_WRITEBACK, "Printf/abort");
   if (!bo) {
      fprintf(stderr, "Failed to allocate printf buffer\n");
      goto fail;
   }

   ret = agx_bo_bind(dev, bo, LIBAGX_PRINTF_BUFFER_ADDRESS,
                     LIBAGX_PRINTF_BUFFER_SIZE, 0,
                     DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
   if (ret) {
      fprintf(stderr, "Failed to bind printf buffer\n");
      agx_bo_unreference(dev, bo);
      goto fail;
   }

   void *printf_map = agx_bo_map(bo);
   if (!printf_map) {
      fprintf(stderr, "Failed to map printf buffer\n");
      agx_bo_unreference(dev, bo);
      goto fail;
   }

   u_printf_init(&dev->printf, bo, printf_map);
   /* Start only after initialization succeeds: no worker may outlive a
    * failed device open. All allocations made above are still idle. */
   if (!agx_bo_cache_start(dev))
      goto fail;
   return true;

fail:
   agx_finish_device(dev);
   return false;
}

bool
agx_apple9_entry_back(struct agx_device *dev, uint32_t end_B)
{
   assert(end_B <= AGX_APPLE9_ENTRY_ARENA_SIZE);
   simple_mtx_lock(&dev->apple9_entry_back_lock);
   for (unsigned i = 0; i * AGX_APPLE9_ENTRY_CHUNK_SIZE < end_B; ++i) {
      if (dev->apple9_entry_chunks[i])
         continue;

      struct agx_bo *bo =
         agx_bo_create(dev, AGX_APPLE9_ENTRY_CHUNK_SIZE, 0,
                       AGX_BO_EXEC | AGX_BO_WRITEBACK | AGX_BO_READONLY,
                       "Apple9 shader entries");
      if (!bo) {
         simple_mtx_unlock(&dev->apple9_entry_back_lock);
         return false;
      }

      /* The chunk's place in the arena, read-only for the GPU. */
      uint32_t offset = i * AGX_APPLE9_ENTRY_CHUNK_SIZE;
      if (agx_bo_bind(dev, bo, dev->shader_base + offset,
                      AGX_APPLE9_ENTRY_CHUNK_SIZE, 0, DRM_ASAHI_BIND_READ) ||
          !agx_bo_map(bo)) {
         agx_bo_unreference(dev, bo);
         simple_mtx_unlock(&dev->apple9_entry_back_lock);
         return false;
      }

      simple_mtx_lock(&dev->apple9_entry_lock);
      dev->apple9_entry_chunks[i] = bo;
      simple_mtx_unlock(&dev->apple9_entry_lock);
   }
   simple_mtx_unlock(&dev->apple9_entry_back_lock);
   return true;
}

void
agx_apple9_entry_write(struct agx_device *dev, uint32_t offset,
                       const void *data, uint32_t size)
{
   const uint8_t *src = data;
   while (size) {
      struct agx_bo *bo =
         dev->apple9_entry_chunks[offset / AGX_APPLE9_ENTRY_CHUNK_SIZE];
      uint32_t in_chunk = offset % AGX_APPLE9_ENTRY_CHUNK_SIZE;
      uint32_t n = MIN2(size, AGX_APPLE9_ENTRY_CHUNK_SIZE - in_chunk);
      memcpy((uint8_t *)agx_bo_map(bo) + in_chunk, src, n);
      agx_bo_note_cpu_write(bo, in_chunk, n);
      offset += n;
      src += n;
      size -= n;
   }
}

static void
agx_finish_device(struct agx_device *dev)
{
   for (unsigned i = 0; i < AGX_APPLE9_ENTRY_CHUNKS; ++i)
      agx_bo_unreference(dev, dev->apple9_entry_chunks[i]);
   if (dev->apple9_entry_va)
      agx_va_free(dev, dev->apple9_entry_va, true);
   agx_bo_unreference(dev, dev->printf.bo);
   agx_bo_unreference(dev, dev->zero_bo);
   agx_bo_unreference(dev, dev->scratch_bo);
   if (dev->printf.bo)
      u_printf_destroy(&dev->printf);
   agx_bo_cache_finish(dev);
   agx_va_quarantine_finish(dev);
   util_vma_heap_finish(&dev->apple9_entry_heap);
   simple_mtx_destroy(&dev->apple9_entry_lock);
   simple_mtx_destroy(&dev->apple9_entry_back_lock);
   util_sparse_array_finish(&dev->bo_map);
   pthread_mutex_destroy(&dev->bo_map_lock);

   util_vma_heap_finish(&dev->main_heap);
   util_vma_heap_finish(&dev->replay_heap);
   util_vma_heap_finish(&dev->usc_heap);
   util_vma_heap_finish(&dev->replay_usc_heap);
   util_vma_heap_finish(&dev->context_heap);
   glsl_type_singleton_decref();
}

void
agx_close_device(struct agx_device *dev)
{
   agx_finish_device(dev);
   close(dev->fd);
   dev->fd = -1;
}

int
agx_create_command_queue(struct agx_device *dev,
                         enum drm_asahi_priority priority,
                         uint32_t *queue_id_out)
{
   if (dev->debug & AGX_DBG_1QUEUE) {
      // Abuse this lock for this, it's debug only anyway
      simple_mtx_lock(&dev->vma_lock);
      if (dev->queue_id) {
         *queue_id_out = dev->queue_id;
         simple_mtx_unlock(&dev->vma_lock);
         return 0;
      }
   }

   struct drm_asahi_queue_create queue_create = {
      .vm_id = dev->vm_id,
      .priority = priority,
      .usc_exec_base = dev->shader_base,
   };

   int ret =
      asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_QUEUE_CREATE, &queue_create);
   if (ret) {
      int err = errno ? -errno : -EIO;
      fprintf(stderr, "DRM_IOCTL_ASAHI_QUEUE_CREATE failed: %m\n");
      if (dev->debug & AGX_DBG_1QUEUE)
         simple_mtx_unlock(&dev->vma_lock);
      return err;
   }

   if (dev->debug & AGX_DBG_1QUEUE) {
      dev->queue_id = queue_create.queue_id;
      simple_mtx_unlock(&dev->vma_lock);
   }

   *queue_id_out = queue_create.queue_id;
   return 0;
}

int
agx_destroy_command_queue(struct agx_device *dev, uint32_t queue_id)
{
   if (dev->debug & AGX_DBG_1QUEUE)
      return 0;

   struct drm_asahi_queue_destroy queue_destroy = {
      .queue_id = queue_id,
   };

   return asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_QUEUE_DESTROY,
                             &queue_destroy);
}

int
agx_import_sync_file(struct agx_device *dev, struct agx_bo *bo, int fd)
{
   struct dma_buf_import_sync_file import_sync_file_ioctl = {
      .flags = DMA_BUF_SYNC_WRITE,
      .fd = fd,
   };

   assert(fd >= 0);
   assert(bo->prime_fd != -1);

   int ret = drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
                      &import_sync_file_ioctl);
   assert(ret >= 0);

   return ret;
}

uint64_t
agx_get_gpu_timestamp(struct agx_device *dev)
{
   struct drm_asahi_get_time get_time = {.flags = 0};

   int ret = asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_GET_TIME, &get_time);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_TIME failed: %m\n");
   }

   return get_time.gpu_timestamp;
}

/* (Re)define UUID_SIZE to avoid including vulkan.h (or p_defines.h) here. */
#define UUID_SIZE 16

void
agx_get_device_uuid(const struct agx_device *dev, void *uuid)
{
   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   /* The device UUID uniquely identifies the given device within the machine.
    * Since we never have more than one device, this doesn't need to be a real
    * UUID, so we use BLAKE3("agx" + gpu_generation + gpu_variant + gpu_revision).
    */
   static const char *device_name = "agx";
   _mesa_blake3_update(&blake3_ctx, device_name, strlen(device_name));

   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_generation,
                     sizeof(dev->params.gpu_generation));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_variant,
                     sizeof(dev->params.gpu_variant));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_revision,
                     sizeof(dev->params.gpu_revision));

   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   assert(BLAKE3_KEY_LEN >= UUID_SIZE);
   memcpy(uuid, blake3, UUID_SIZE);
}
