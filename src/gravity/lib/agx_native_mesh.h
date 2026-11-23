/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The driver-local v1 advertisement is not a shared DRM UAPI capability. */
#define AGX_NATIVE_MESH_ADVERTISEMENT_MAX 160
/* bits/219 stores generation << 8 | mode in its atomic u64. */
#define AGX_NATIVE_MESH_GENERATION_MAX (UINT64_MAX >> 8)

enum agx_native_mesh_profile {
   AGX_NATIVE_MESH_PROFILE_UNKNOWN = 0,
   AGX_NATIVE_MESH_PROFILE_NONE,
   AGX_NATIVE_MESH_PROFILE_RESEARCH,
   AGX_NATIVE_MESH_PROFILE_QUALIFIED,
};

struct agx_native_mesh_support {
   bool qualified;
   uint64_t generation;
   enum agx_native_mesh_profile profile;
   uint64_t profile_generation;
};

struct agx_native_mesh_support
agx_native_mesh_parse(const char *text, size_t size, const char *classes);

struct agx_native_mesh_support
agx_native_mesh_probe(int drm_fd, const char *classes);

/* True when the kernel's asahi_native_mesh sysfs attribute explicitly reports
 * profile=none (native mesh decoding disabled in kernel). */
bool agx_native_mesh_parse_disabled(const char *text, size_t size);
bool agx_native_mesh_probe_disabled(int drm_fd);

static inline bool
agx_native_mesh_same_qualification(struct agx_native_mesh_support before,
                                   struct agx_native_mesh_support after)
{
   return before.qualified && after.qualified &&
          before.generation == after.generation;
}

/* Determine the VM's latched native-mesh profile from samples taken before and
 * after VM_CREATE. Any concurrent change across VM_CREATE fails closed to
 * NONE; UNKNOWN is preserved only when both samples saw no recognized sysfs
 * attribute (older pre-sysfs kernels or host test devices). */
static inline enum agx_native_mesh_profile
agx_native_mesh_latched_profile(struct agx_native_mesh_support before,
                                struct agx_native_mesh_support after)
{
   if (before.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN &&
       after.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN)
      return AGX_NATIVE_MESH_PROFILE_UNKNOWN;
   if (before.profile != AGX_NATIVE_MESH_PROFILE_UNKNOWN &&
       before.profile == after.profile &&
       before.profile_generation == after.profile_generation)
      return before.profile;
   return AGX_NATIVE_MESH_PROFILE_NONE;
}

/* asahi_native_blocks (driver-local, beside asahi_native_mesh): whether the
 * VM's kernel admits several complete native blocks, and ordinary blocks after
 * the first, in one render. Same generation as asahi_native_mesh. */
struct agx_native_blocks_support {
   bool qualified;
   uint64_t generation;
   /* Native dispatches per render, whether ordinary blocks may follow the
    * leading native block, and the kernel's decode budget in words. */
   unsigned max_native;
   bool mixed;
   unsigned words;
};

#define AGX_NATIVE_BLOCKS_MAX_NATIVE 64
#define AGX_NATIVE_BLOCKS_MIN_WORDS 1024
#define AGX_NATIVE_BLOCKS_MAX_WORDS 4096

struct agx_native_blocks_support
agx_native_blocks_parse(const char *text, size_t size);

struct agx_native_blocks_support agx_native_blocks_probe(int drm_fd);

static inline bool
agx_native_blocks_same(struct agx_native_blocks_support before,
                       struct agx_native_blocks_support after)
{
   return before.qualified && after.qualified &&
          before.generation == after.generation &&
          before.max_native == after.max_native &&
          before.mixed == after.mixed && before.words == after.words;
}
