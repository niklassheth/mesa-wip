/* SPDX-License-Identifier: MIT
 * Read-only qualification instrumentation. No ioctl arguments/results modified.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "drm-uapi/asahi_drm.h"

static void sample(int fd, const char *when, int result)
{
   int saved = errno;
   struct stat st;
   char path[256], line[160] = "UNREADABLE";
   if (fstat(fd, &st) == 0) {
      snprintf(path, sizeof(path), "/sys/dev/char/%u:%u/device/asahi_native_mesh",
               major(st.st_rdev), minor(st.st_rdev));
      int f = open(path, O_RDONLY | O_CLOEXEC);
      if (f >= 0) {
         ssize_t n = read(f, line, sizeof(line) - 1);
         if (n >= 0) line[n] = 0;
         close(f);
      }
      line[strcspn(line, "\r\n")] = 0;
      fprintf(stderr, "VM_AUDIT %s fd=%d dev=%u:%u result=%d %s\n",
              when, fd, major(st.st_rdev), minor(st.st_rdev), result, line);
   }
   if (!strcmp(when, "after")) {
      FILE *maps = fopen("/proc/self/maps", "r");
      char *s = NULL; size_t cap = 0;
      if (maps) {
         while (getline(&s, &cap, maps) > 0)
            if (strstr(s, "libvulkan_asahi.so"))
               fprintf(stderr, "VM_MAP %s", s);
         fclose(maps); free(s);
      }
   }
   errno = saved;
}

int ioctl(int fd, unsigned long request, ...)
{
   va_list ap; va_start(ap, request);
   void *arg = va_arg(ap, void *); va_end(ap);
   if (request == DRM_IOCTL_ASAHI_VM_CREATE) sample(fd, "before", 0);
   int ret = syscall(SYS_ioctl, fd, request, arg);
   if (request == DRM_IOCTL_ASAHI_VM_CREATE) sample(fd, "after", ret);
   return ret;
}
