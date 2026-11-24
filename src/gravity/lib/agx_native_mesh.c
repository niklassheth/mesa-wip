/* SPDX-License-Identifier: MIT */
#include "agx_native_mesh.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

static bool
valid_classes_hash(const char *classes)
{
   if (!classes || strlen(classes) != 64)
      return false;
   for (unsigned i = 0; i < 64; ++i) {
      if (!((classes[i] >= '0' && classes[i] <= '9') ||
            (classes[i] >= 'a' && classes[i] <= 'f')))
         return false;
   }
   return true;
}

struct agx_native_mesh_support
agx_native_mesh_parse(const char *text, size_t size, const char *classes)
{
   const struct agx_native_mesh_support no = {0};
   static const char prefix[] = "asahi-native-mesh v1 generation=";
   static const char prof_qualified[] = " profile=qualified classes=";
   static const char prof_research[] = " profile=research classes=-";
   static const char prof_none[] = " profile=none classes=-";
   if (!text || !size || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX ||
       size <= sizeof(prefix) - 1 ||
       memcmp(text, prefix, sizeof(prefix) - 1) ||
       text[size - 1] != '\n')
      return no;

   size_t pos = sizeof(prefix) - 1;
   size_t end = size - 1;
   if (pos >= end || text[pos] < '0' || text[pos] > '9')
      return no;
   bool leading_zero = text[pos] == '0';
   uint64_t generation = 0;
   size_t digits = 0;
   while (pos < end && text[pos] >= '0' && text[pos] <= '9') {
      unsigned digit = text[pos++] - '0';
      if (generation > (AGX_NATIVE_MESH_GENERATION_MAX - digit) / 10 ||
          (leading_zero && digits))
         return no;
      generation = generation * 10 + digit;
      ++digits;
   }

   if (end - pos == sizeof(prof_none) - 1 &&
       !memcmp(text + pos, prof_none, sizeof(prof_none) - 1))
      return (struct agx_native_mesh_support){
         .profile = AGX_NATIVE_MESH_PROFILE_NONE,
         .profile_generation = generation,
      };

   if (end - pos == sizeof(prof_research) - 1 &&
       !memcmp(text + pos, prof_research, sizeof(prof_research) - 1))
      return (struct agx_native_mesh_support){
         .profile = AGX_NATIVE_MESH_PROFILE_RESEARCH,
         .profile_generation = generation,
      };

   if (valid_classes_hash(classes) &&
       end - pos == sizeof(prof_qualified) - 1 + 64 &&
       !memcmp(text + pos, prof_qualified, sizeof(prof_qualified) - 1) &&
       !memcmp(text + pos + sizeof(prof_qualified) - 1, classes, 64))
      return (struct agx_native_mesh_support){
         .qualified = true,
         .generation = generation,
         .profile = AGX_NATIVE_MESH_PROFILE_QUALIFIED,
         .profile_generation = generation,
      };

   return no;
}

enum agx_attr_read_status {
   AGX_ATTR_READ_OK = 0,
   AGX_ATTR_READ_ABSENT,
   AGX_ATTR_READ_NOT_CHR,
   AGX_ATTR_READ_ERROR,
};

/* Read the driver-local attribute NAME of the DRM device behind drm_fd.
 * Distinguishes confirmed absence (ENOENT/ENOTDIR on pre-sysfs kernels) from
 * I/O, permission, FD-limit, or size errors so callers can fail closed. */
static enum agx_attr_read_status
read_attribute(int drm_fd, const char *name, char *text, size_t capacity,
               size_t *size_out)
{
   *size_out = 0;
   struct stat st;
   if (fstat(drm_fd, &st))
      return AGX_ATTR_READ_ERROR;
   if (!S_ISCHR(st.st_mode))
      return AGX_ATTR_READ_NOT_CHR;

   char path[128];
   int n = snprintf(path, sizeof(path), "/sys/dev/char/%u:%u/device/%s",
                    major(st.st_rdev), minor(st.st_rdev), name);
   if (n < 0 || (size_t)n >= sizeof(path))
      return AGX_ATTR_READ_ERROR;

   int fd = open(path, O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return (errno == ENOENT || errno == ENOTDIR) ? AGX_ATTR_READ_ABSENT
                                                   : AGX_ATTR_READ_ERROR;

   size_t size = 0;
   bool complete = false;
   while (size < capacity) {
      ssize_t count = read(fd, text + size, capacity - size);
      if (count < 0) {
         if (errno == EINTR)
            continue;
         break;
      }
      if (count == 0) {
         complete = true;
         break;
      }
      size += count;
   }
   close(fd);
   if (!complete)
      return AGX_ATTR_READ_ERROR;
   *size_out = size;
   return AGX_ATTR_READ_OK;
}

struct agx_native_mesh_support
agx_native_mesh_probe(int drm_fd, const char *classes)
{
   const struct agx_native_mesh_support absent = {
      .profile = AGX_NATIVE_MESH_PROFILE_UNKNOWN,
   };
   const struct agx_native_mesh_support fail = {
      .profile = AGX_NATIVE_MESH_PROFILE_NONE,
   };
   if (!classes)
      return fail;
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = 0;
   enum agx_attr_read_status st =
      read_attribute(drm_fd, "asahi_native_mesh", text, sizeof(text), &size);
   if (st == AGX_ATTR_READ_ABSENT)
      return absent;
   if (st != AGX_ATTR_READ_OK || !size)
      return fail;
   struct agx_native_mesh_support parsed =
      agx_native_mesh_parse(text, size, classes);
   /* The sysfs attribute exists, so an unparseable advertisement fails closed
    * instead of masquerading as a pre-sysfs kernel. */
   if (parsed.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN)
      return fail;
   return parsed;
}

/* A decimal field without leading zeros, at most max, advancing *pos. */
static bool
parse_decimal(const char *text, size_t size, size_t *pos, uint64_t max,
              uint64_t *value)
{
   size_t at = *pos;
   if (at >= size || text[at] < '0' || text[at] > '9' ||
       (text[at] == '0' && at + 1 < size && text[at + 1] >= '0' &&
        text[at + 1] <= '9'))
      return false;
   uint64_t v = 0;
   while (at < size && text[at] >= '0' && text[at] <= '9') {
      unsigned digit = text[at++] - '0';
      if (digit > max || v > (max - digit) / 10)
         return false;
      v = v * 10 + digit;
   }
   *pos = at;
   *value = v;
   return true;
}

static bool
parse_literal(const char *text, size_t size, size_t *pos, const char *literal)
{
   size_t n = strlen(literal);
   if (size - *pos < n || memcmp(text + *pos, literal, n))
      return false;
   *pos += n;
   return true;
}

/* asahi-native-blocks v1 generation=<G> profile=<none|qualified>
 * max_native=<N> mixed=<0|1> words=<W>\n, exactly, single spaces. Any other
 * line, including none with values other than 1/0/1024, is not qualified. */
struct agx_native_blocks_support
agx_native_blocks_parse(const char *text, size_t size)
{
   const struct agx_native_blocks_support no = {0};
   if (!text || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX || !size ||
       text[size - 1] != '\n')
      return no;
   size_t pos = 0, end = size - 1;
   uint64_t generation, max_native, mixed, words;
   bool qualified;
   if (!parse_literal(text, end, &pos, "asahi-native-blocks v1 generation=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_MESH_GENERATION_MAX, &generation))
      return no;
   if (parse_literal(text, end, &pos, " profile=qualified"))
      qualified = true;
   else if (parse_literal(text, end, &pos, " profile=none"))
      qualified = false;
   else
      return no;
   if (!parse_literal(text, end, &pos, " max_native=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_BLOCKS_MAX_NATIVE, &max_native) ||
       !parse_literal(text, end, &pos, " mixed=") ||
       !parse_decimal(text, end, &pos, 1, &mixed) ||
       !parse_literal(text, end, &pos, " words=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_BLOCKS_MAX_WORDS, &words) ||
       pos != end || !max_native || words < AGX_NATIVE_BLOCKS_MIN_WORDS)
      return no;
   if (!qualified)
      return no;
   return (struct agx_native_blocks_support){
      true, generation, max_native, mixed, words};
}

struct agx_native_blocks_support
agx_native_blocks_probe(int drm_fd)
{
   const struct agx_native_blocks_support no = {0};
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = 0;
   if (read_attribute(drm_fd, "asahi_native_blocks", text, sizeof(text),
                      &size) != AGX_ATTR_READ_OK || !size)
      return no;
   return agx_native_blocks_parse(text, size);
}

/* asahi-native-object v1 generation=<G> profile=<qualified|research|none>
 * multi_parent=<0|1> payload_bytes=<B> max_threads=<T> max_shared=<S>
 * max_groups=<M>\n, exact single spaces. Fails closed on any mismatch. */
struct agx_native_object_support
agx_native_object_parse(const char *text, size_t size)
{
   const struct agx_native_object_support no = {0};
   if (!text || !size || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX ||
       text[size - 1] != '\n')
      return no;
   size_t pos = 0, end = size - 1;
   uint64_t generation, multi_parent, payload_bytes, max_threads, max_shared,
      max_groups;
   enum agx_native_mesh_profile profile;
   if (!parse_literal(text, end, &pos, "asahi-native-object v1 generation=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_MESH_GENERATION_MAX,
                      &generation))
      return no;
   if (parse_literal(text, end, &pos, " profile=qualified"))
      profile = AGX_NATIVE_MESH_PROFILE_QUALIFIED;
   else if (parse_literal(text, end, &pos, " profile=research"))
      profile = AGX_NATIVE_MESH_PROFILE_RESEARCH;
   else if (parse_literal(text, end, &pos, " profile=none"))
      profile = AGX_NATIVE_MESH_PROFILE_NONE;
   else
      return no;
   if (!parse_literal(text, end, &pos, " multi_parent=") ||
       !parse_decimal(text, end, &pos, 1, &multi_parent) ||
       !parse_literal(text, end, &pos, " payload_bytes=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_OBJECT_MAX_PAYLOAD_BYTES,
                      &payload_bytes) ||
       !parse_literal(text, end, &pos, " max_threads=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_OBJECT_MAX_THREADS,
                      &max_threads) ||
       !parse_literal(text, end, &pos, " max_shared=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_OBJECT_MAX_SHARED_BYTES,
                      &max_shared) ||
       !parse_literal(text, end, &pos, " max_groups=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_OBJECT_MAX_GROUPS,
                      &max_groups) ||
       pos != end)
      return no;
   if (profile == AGX_NATIVE_MESH_PROFILE_NONE) {
      if (multi_parent || payload_bytes || max_threads != 16 || max_shared ||
          max_groups != 1)
         return no;
      return (struct agx_native_object_support){
         .profile = AGX_NATIVE_MESH_PROFILE_NONE,
         .generation = generation,
         .max_threads = 16,
         .max_groups = 1,
      };
   }
   if (!multi_parent || !payload_bytes ||
       (max_threads != 16 && max_threads != 32) || !max_shared ||
       max_groups < 2)
      return no;
   return (struct agx_native_object_support){
      .profile = profile,
      .generation = generation,
      .multi_parent = true,
      .payload_bytes = payload_bytes,
      .max_threads = max_threads,
      .max_shared = max_shared,
      .max_groups = max_groups,
   };
}

struct agx_native_object_support
agx_native_object_probe(int drm_fd)
{
   const struct agx_native_object_support no = {0};
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = 0;
   if (read_attribute(drm_fd, "asahi_native_object", text, sizeof(text),
                      &size) != AGX_ATTR_READ_OK || !size)
      return no;
   return agx_native_object_parse(text, size);
}

bool
agx_native_mesh_parse_disabled(const char *text, size_t size)
{
   if (!text || !size || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX ||
       text[size - 1] != '\n')
      return false;
   size_t pos = 0, end = size - 1;
   uint64_t generation;
   if (!parse_literal(text, end, &pos, "asahi-native-mesh v1 generation=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_MESH_GENERATION_MAX, &generation))
      return false;
   return parse_literal(text, end, &pos, " profile=none ");
}

static bool
agx_native_mesh_parse_enabled_strict(const char *text, size_t size)
{
   static const char prefix[] = "asahi-native-mesh v1 generation=";
   static const char prof_research[] = " profile=research classes=-";
   static const char prof_qualified[] = " profile=qualified classes=";
   if (!text || !size || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX ||
       size <= sizeof(prefix) - 1 ||
       memcmp(text, prefix, sizeof(prefix) - 1) ||
       text[size - 1] != '\n')
      return false;
   size_t pos = sizeof(prefix) - 1, end = size - 1;
   uint64_t generation;
   if (!parse_decimal(text, end, &pos, AGX_NATIVE_MESH_GENERATION_MAX,
                      &generation))
      return false;
   if (end - pos == sizeof(prof_research) - 1 &&
       !memcmp(text + pos, prof_research, sizeof(prof_research) - 1))
      return true;
   if (end - pos == sizeof(prof_qualified) - 1 + 64 &&
       !memcmp(text + pos, prof_qualified, sizeof(prof_qualified) - 1)) {
      const char *hex = text + pos + sizeof(prof_qualified) - 1;
      for (size_t i = 0; i < 64; ++i) {
         char c = hex[i];
         if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
      }
      return true;
   }
   return false;
}

bool
agx_native_mesh_probe_disabled(int drm_fd)
{
   if (drm_fd < 0)
      return false;
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = 0;
   enum agx_attr_read_status st =
      read_attribute(drm_fd, "asahi_native_mesh", text, sizeof(text), &size);
   if (st == AGX_ATTR_READ_ABSENT || st == AGX_ATTR_READ_NOT_CHR)
      return false;
   if (st != AGX_ATTR_READ_OK || !size)
      return true;
   /* Fail closed for profile=none and any malformed present advertisement. */
   return !agx_native_mesh_parse_enabled_strict(text, size);
}
