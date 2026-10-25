/* SPDX-License-Identifier: MIT
 * Compatibility model for the historical host-only submission recorder.
 * It has no GPU execution or asynchronous completion. Never stage on a device.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <linux/sync_file.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int tracked[256]; /* fd + 1; zero is unused */

static bool enabled(void)
{
   const char *value = getenv("ASAHI_HOST_RECORD_SYNCHRONOUS");
   return value && !strcmp(value, "1") && getenv("ASAHI_HOST_TRACE_DIR");
}

int drmSyncobjExportSyncFile(int drm_fd, uint32_t handle, int *result)
{
   if (!enabled()) {
      int (*next)(int, uint32_t, int *) = dlsym(RTLD_NEXT, __func__);
      return next(drm_fd, handle, result);
   }
   int fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
   if (fd < 0) return -errno;
   pthread_mutex_lock(&lock);
   for (unsigned i = 0; i < 256; ++i) {
      if (!tracked[i]) {
         tracked[i] = fd + 1;
         pthread_mutex_unlock(&lock);
         *result = fd;
         return 0;
      }
   }
   pthread_mutex_unlock(&lock);
   close(fd);
   errno = EMFILE;
   return -errno;
}

int ioctl(int fd, unsigned long request, ...)
{
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);
   if (request == SYNC_IOC_FILE_INFO) {
      pthread_mutex_lock(&lock);
      bool found = false;
      for (unsigned i = 0; i < 256; ++i) found |= tracked[i] == fd + 1;
      if (found) {
         struct sync_file_info *info = arg;
         if (info->num_fences && info->sync_fence_info) {
            struct sync_fence_info *fence = (void *)(uintptr_t)info->sync_fence_info;
            *fence = (struct sync_fence_info){.status = 1};
            strcpy(fence->driver_name, "host-only-no-execution");
         }
         info->status = 1;
         info->num_fences = 1;
         pthread_mutex_unlock(&lock);
         return 0;
      }
      pthread_mutex_unlock(&lock);
   }
   int (*next)(int, unsigned long, ...) = dlsym(RTLD_NEXT, __func__);
   return next(fd, request, arg);
}

int close(int fd)
{
   pthread_mutex_lock(&lock);
   for (unsigned i = 0; i < 256; ++i) {
      if (tracked[i] == fd + 1) tracked[i] = 0;
   }
   pthread_mutex_unlock(&lock);
   int (*next)(int) = dlsym(RTLD_NEXT, __func__);
   return next(fd);
}
