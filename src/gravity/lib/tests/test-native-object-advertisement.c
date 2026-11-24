/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * asahi_native_object: exact grammar, fail closed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../agx_native_mesh.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static struct agx_native_object_support
parse(const char *s)
{
   return agx_native_object_parse(s, strlen(s));
}

int
main(void)
{
   struct agx_native_object_support q = parse(
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n");
   CHECK(q.profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED);
   CHECK(agx_native_object_enabled(&q));
   CHECK(q.generation == 7 && q.multi_parent && q.payload_bytes == 16384 &&
         q.mp_payload_bytes == 0 && q.max_threads == 32 &&
         q.max_shared == 32768 && q.max_groups == 1048575);

   q = parse(
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1024\n");
   CHECK(q.profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED);
   CHECK(agx_native_object_enabled(&q));
   CHECK(q.generation == 7 && q.multi_parent && q.payload_bytes == 16384 &&
         q.mp_payload_bytes == 1024 && q.max_threads == 32 &&
         q.max_shared == 32768 && q.max_groups == 1048575);

   struct agx_native_object_support r = parse(
      "asahi-native-object v1 generation=42 profile=research multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1024\n");
   CHECK(r.profile == AGX_NATIVE_MESH_PROFILE_RESEARCH);
   CHECK(agx_native_object_enabled(&r));
   CHECK(r.generation == 42 && r.multi_parent && r.payload_bytes == 16384 &&
         r.mp_payload_bytes == 1024 && r.max_threads == 32 &&
         r.max_shared == 32768 && r.max_groups == 1048575);

   /* Minimal valid multi-parent qualified line (T16, 4 KiB payload, 2 groups). */
   q = parse(
      "asahi-native-object v1 generation=0 profile=qualified multi_parent=1 payload_bytes=4096 max_threads=16 max_shared=16384 max_groups=2 mp_payload_bytes=0\n");
   CHECK(agx_native_object_enabled(&q));
   CHECK(q.generation == 0 && q.multi_parent && q.payload_bytes == 4096 &&
         q.mp_payload_bytes == 0 && q.max_threads == 16 &&
         q.max_shared == 16384 && q.max_groups == 2);

   /* profile=none with mp_payload_bytes=0 parses cleanly as disabled (NONE). */
   struct agx_native_object_support none = parse(
      "asahi-native-object v1 generation=7 profile=none multi_parent=0 payload_bytes=0 max_threads=16 max_shared=0 max_groups=1 mp_payload_bytes=0\n");
   CHECK(none.profile == AGX_NATIVE_MESH_PROFILE_NONE &&
         none.generation == 7 && !agx_native_object_enabled(&none));
   struct agx_native_object_support none_bad = parse(
      "asahi-native-object v1 generation=7 profile=none multi_parent=0 payload_bytes=0 max_threads=16 max_shared=0 max_groups=1 mp_payload_bytes=1024\n");
   CHECK(none_bad.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN &&
         !agx_native_object_enabled(&none_bad));

   char max[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   snprintf(
      max, sizeof(max),
      "asahi-native-object v1 generation=%llu profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=16384\n",
      (unsigned long long)AGX_NATIVE_MESH_GENERATION_MAX);
   q = parse(max);
   CHECK(agx_native_object_enabled(&q) &&
         q.generation == AGX_NATIVE_MESH_GENERATION_MAX &&
         q.mp_payload_bytes == 16384);

   static const char *bad[] = {
      /* profile=none is never enabled */
      "asahi-native-object v1 generation=7 profile=none multi_parent=0 payload_bytes=0 max_threads=16 max_shared=0 max_groups=1\n",
      "asahi-native-object v1 generation=7 profile=none multi_parent=0 payload_bytes=0 max_threads=16 max_shared=0 max_groups=1 mp_payload_bytes=0\n",
      "asahi-native-object v1 generation=7 profile=none multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=experimental multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v2 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-blocks v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      /* framing */
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 \n",
      "asahi-native-object v1 generation=7  profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 x=1\n",
      /* invalid numbers / bounds */
      "asahi-native-object v1 generation=07 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=-1 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=72057594037927936 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=2 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=0 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=2\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16385 max_threads=32 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=15 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=24 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=33 max_shared=32768 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32769 max_groups=1048575\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=0\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048576\n",
      /* invalid mp_payload_bytes */
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1025\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=1024 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=2048\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=16385\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=01024\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575  mp_payload_bytes=1024\n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1024 \n",
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1024 x=1\n",
      "",
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      struct agx_native_object_support parsed = parse(bad[i]);
      if (agx_native_object_enabled(&parsed)) {
         fprintf(stderr, "accepted invalid line: %s\n", bad[i]);
         return 1;
      }
   }
   struct agx_native_object_support null_s = agx_native_object_parse(NULL, 10);
   CHECK(!agx_native_object_enabled(&null_s));
   char over[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 16];
   memset(over, 'x', sizeof(over));
   struct agx_native_object_support over_s =
      agx_native_object_parse(over, AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1);
   CHECK(!agx_native_object_enabled(&over_s));

   /* Latching across VM_CREATE: every field must match. */
   struct agx_native_object_support a = parse(
      "asahi-native-object v1 generation=7 profile=qualified multi_parent=1 payload_bytes=16384 max_threads=32 max_shared=32768 max_groups=1048575 mp_payload_bytes=1024\n");
   struct agx_native_object_support b = a;
   CHECK(agx_native_object_same(a, b));
   b.generation++;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.profile = AGX_NATIVE_MESH_PROFILE_RESEARCH;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.multi_parent = false;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.payload_bytes = 8192;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.mp_payload_bytes = 0;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.max_threads = 16;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.max_shared = 16384;
   CHECK(!agx_native_object_same(a, b));
   b = a; b.max_groups = 1024;
   CHECK(!agx_native_object_same(a, b));
   CHECK(!agx_native_object_same((struct agx_native_object_support){0},
                                 (struct agx_native_object_support){0}));
   printf("PASS\n");
   return 0;
}
