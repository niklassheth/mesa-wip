/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The driver-local v1 advertisement is not a shared DRM UAPI capability. */
#define AGX_NATIVE_MESH_ADVERTISEMENT_MAX 256
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

/* asahi_native_object (driver-local, beside asahi_native_mesh): whether the
 * VM's kernel admits Phase 2 multi-parent native OBJECT -> MESH dispatches
 * with 16 KiB task payload and up to 32 KiB object shared memory. Shares the
 * same atomic generation counter as asahi_native_mesh. */
#define AGX_NATIVE_OBJECT_MAX_PAYLOAD_BYTES 16384u
#define AGX_NATIVE_OBJECT_MAX_THREADS 32u
#define AGX_NATIVE_OBJECT_MAX_SHARED_BYTES 32768u
#define AGX_NATIVE_OBJECT_MAX_GROUPS ((1u << 20) - 1)

struct agx_native_object_support {
   enum agx_native_mesh_profile profile;
   uint64_t generation;
   bool multi_parent;
   uint32_t payload_bytes;
   uint32_t mp_payload_bytes;
   uint32_t max_threads;
   uint32_t max_shared;
   uint32_t max_groups;
};

struct agx_native_object_support
agx_native_object_parse(const char *text, size_t size);

struct agx_native_object_support agx_native_object_probe(int drm_fd);

static inline bool
agx_native_object_enabled(const struct agx_native_object_support *o)
{
   return o &&
          (o->profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED ||
           o->profile == AGX_NATIVE_MESH_PROFILE_RESEARCH) &&
          o->multi_parent && o->payload_bytes &&
          o->payload_bytes <= AGX_NATIVE_OBJECT_MAX_PAYLOAD_BYTES &&
          o->mp_payload_bytes <= o->payload_bytes &&
          !(o->mp_payload_bytes & 15) &&
          (o->max_threads == 16 || o->max_threads == 32) &&
          o->max_shared && o->max_shared <= AGX_NATIVE_OBJECT_MAX_SHARED_BYTES &&
          o->max_groups && o->max_groups <= AGX_NATIVE_OBJECT_MAX_GROUPS;
}

static inline bool
agx_native_object_same(struct agx_native_object_support before,
                       struct agx_native_object_support after)
{
   return agx_native_object_enabled(&before) &&
          before.profile == after.profile &&
          before.generation == after.generation &&
          before.multi_parent == after.multi_parent &&
          before.payload_bytes == after.payload_bytes &&
          before.mp_payload_bytes == after.mp_payload_bytes &&
          before.max_threads == after.max_threads &&
          before.max_shared == after.max_shared &&
          before.max_groups == after.max_groups;
}
