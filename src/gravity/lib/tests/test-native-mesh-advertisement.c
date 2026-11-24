/* SPDX-License-Identifier: MIT */
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)
#include <stdlib.h>

static const char classes[] =
   "5550ecf263192c596c3f9d8c3b84c0324ac3d0c8ee76da37c378205247f1a3bd";
static char input[512];
static size_t input_size, offset;
static bool fail_stat, regular_fd, fail_open, fail_read, interrupt_read;
static int fail_open_errno = EMFILE;
static unsigned opens, closes;

static int
test_fstat(int fd, struct stat *st)
{
   CHECK(fd == 42);
   if (fail_stat)
      return -1;
   memset(st, 0, sizeof(*st));
   st->st_mode = regular_fd ? S_IFREG : S_IFCHR;
   st->st_rdev = makedev(226, 137);
   return 0;
}

static int
test_open(const char *path, int flags, ...)
{
   CHECK(!strcmp(path, "/sys/dev/char/226:137/device/asahi_native_mesh"));
   CHECK(flags == (O_RDONLY | O_CLOEXEC));
   ++opens;
   offset = 0;
   if (fail_open) {
      errno = fail_open_errno;
      return -1;
   }
   return 43;
}

static ssize_t
test_read(int fd, void *data, size_t size)
{
   CHECK(fd == 43);
   if (fail_read || interrupt_read) {
      errno = interrupt_read ? EINTR : EIO;
      interrupt_read = false;
      return -1;
   }
   size_t count = input_size - offset;
   if (count > size)
      count = size;
   if (count > 17)
      count = 17; /* Exercise short reads, including the final newline. */
   memcpy(data, input + offset, count);
   offset += count;
   return count;
}

static int
test_close(int fd)
{
   CHECK(fd == 43);
   ++closes;
   return 0;
}

/* Exercise the exact fd-derived reader with deterministic filesystem errors. */
#define fstat test_fstat
#define open test_open
#define read test_read
#define close test_close
#include "../agx_native_mesh.c"
#undef fstat
#undef open
#undef read
#undef close

static struct agx_native_mesh_support
parse(const char *generation, const char *profile, const char *hash)
{
   input_size = snprintf(input, sizeof(input),
                         "asahi-native-mesh v1 generation=%s profile=%s classes=%s\n",
                         generation, profile, hash);
   return agx_native_mesh_parse(input, input_size, classes);
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--parse-stdin")) {
      char *line = NULL;
      size_t capacity = 0;
      ssize_t length;
      while ((length = getline(&line, &capacity, stdin)) >= 0) {
         struct agx_native_mesh_support s =
            agx_native_mesh_parse(line, length, classes);
         printf("%u %llu\n", s.qualified, (unsigned long long)s.generation);
      }
      free(line);
      return ferror(stdin) ? 1 : 0;
   }
   CHECK(argc == 1);
   struct agx_native_mesh_support zero = parse("0", "qualified", classes);
   CHECK(zero.qualified && zero.generation == 0 &&
         zero.profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED &&
         zero.profile_generation == 0);
   struct agx_native_mesh_support max =
      parse("72057594037927935", "qualified", classes);
   CHECK(max.qualified && max.generation == AGX_NATIVE_MESH_GENERATION_MAX &&
         max.profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED &&
         max.profile_generation == AGX_NATIVE_MESH_GENERATION_MAX);
   const char *bad_generations[] = {
      "", "00", "01", "-1", "+1", " 1", "1 ", "1x", "0x1",
      "72057594037927936", "9223372036854775807", "9223372036854775808",
      "18446744073709551615", "9999999999999999999999",
   };
   for (unsigned i = 0; i < sizeof(bad_generations) / sizeof(*bad_generations); ++i)
      CHECK(!parse(bad_generations[i], "qualified", classes).qualified);
   struct agx_native_mesh_support p_none = parse("1", "none", "-");
   CHECK(!p_none.qualified && p_none.generation == 0 &&
         p_none.profile == AGX_NATIVE_MESH_PROFILE_NONE &&
         p_none.profile_generation == 1);
   struct agx_native_mesh_support p_research = parse("1", "research", "-");
   CHECK(!p_research.qualified && p_research.generation == 0 &&
         p_research.profile == AGX_NATIVE_MESH_PROFILE_RESEARCH &&
         p_research.profile_generation == 1);
   CHECK(!parse("1", "research", classes).qualified &&
         parse("1", "research", classes).profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN);
   CHECK(!parse("1", "future", classes).qualified);
   CHECK(!parse("1", "qualified", "-").qualified);
   char wrong[65];
   memcpy(wrong, classes, sizeof(wrong));
   wrong[63] = '0';
   CHECK(!parse("1", "qualified", wrong).qualified);

   struct agx_native_mesh_support p_research2 = parse("2", "research", "-");
   CHECK(agx_native_mesh_latched_profile(zero, zero) == AGX_NATIVE_MESH_PROFILE_QUALIFIED);
   CHECK(agx_native_mesh_latched_profile(p_research, p_research) == AGX_NATIVE_MESH_PROFILE_RESEARCH);
   CHECK(agx_native_mesh_latched_profile(p_none, p_none) == AGX_NATIVE_MESH_PROFILE_NONE);
   /* Any mismatch or generation change across VM_CREATE fails closed to NONE. */
   CHECK(agx_native_mesh_latched_profile(p_research, p_research2) == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_latched_profile(p_none, zero) == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_latched_profile(p_research, p_none) == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_latched_profile(zero, max) == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_latched_profile((struct agx_native_mesh_support){0},
                                         (struct agx_native_mesh_support){0}) ==
         AGX_NATIVE_MESH_PROFILE_UNKNOWN);
   CHECK(agx_native_mesh_latched_profile(zero, (struct agx_native_mesh_support){0}) ==
         AGX_NATIVE_MESH_PROFILE_NONE);

   parse("1", "qualified", classes);
   char good[512];
   memcpy(good, input, input_size);
   size_t good_size = input_size;
   for (size_t i = 0; i < good_size; ++i) {
      CHECK(!agx_native_mesh_parse(good, i, classes).qualified);
      memcpy(input, good, good_size);
      input[i] = '\0';
      CHECK(!agx_native_mesh_parse(input, good_size, classes).qualified);
   }
   memcpy(input, good, good_size);
   input[19] = '2'; /* v2 */
   CHECK(!agx_native_mesh_parse(input, good_size, classes).qualified);
   memcpy(input, good, good_size);
   input[good_size] = '\n';
   CHECK(!agx_native_mesh_parse(input, good_size + 1, classes).qualified);
   memset(input + good_size, 'x', sizeof(input) - good_size);
   CHECK(!agx_native_mesh_parse(input, AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1, classes).qualified);
   CHECK(!agx_native_mesh_parse(good, good_size, NULL).qualified);
   CHECK(!agx_native_mesh_parse(NULL, good_size, classes).qualified);

   CHECK(agx_native_mesh_same_qualification(zero, zero));
   CHECK(agx_native_mesh_same_qualification(max, max));
   CHECK(!agx_native_mesh_same_qualification(zero, max));
   struct agx_native_mesh_support absent = {0};
   CHECK(!agx_native_mesh_same_qualification(absent, zero));
   CHECK(!agx_native_mesh_same_qualification(zero, absent));
   CHECK(!agx_native_mesh_same_qualification(absent, absent));

   parse("7", "qualified", classes);
   interrupt_read = true;
   struct agx_native_mesh_support readback = agx_native_mesh_probe(42, classes);
   CHECK(readback.qualified && readback.generation == 7 &&
         readback.profile == AGX_NATIVE_MESH_PROFILE_QUALIFIED &&
         readback.profile_generation == 7);
   CHECK(!agx_native_mesh_probe_disabled(42));
   parse("8", "research", "-");
   struct agx_native_mesh_support res_rb = agx_native_mesh_probe(42, classes);
   CHECK(!res_rb.qualified && res_rb.profile == AGX_NATIVE_MESH_PROFILE_RESEARCH &&
         res_rb.profile_generation == 8);
   CHECK(!agx_native_mesh_probe_disabled(42));
   parse("9", "none", "-");
   struct agx_native_mesh_support none_rb = agx_native_mesh_probe(42, classes);
   CHECK(!none_rb.qualified && none_rb.profile == AGX_NATIVE_MESH_PROFILE_NONE &&
         none_rb.profile_generation == 9);
   CHECK(agx_native_mesh_probe_disabled(42));
   CHECK(opens == 6 && closes == 6);
   /* I/O, stat, non-chr, non-ENOENT open, oversized, and parse failures all
    * fail closed to NONE (never UNKNOWN), so a VM cannot silently fall back to
    * legacy older-kernel permissiveness. */
   fail_stat = true;
   struct agx_native_mesh_support failed = agx_native_mesh_probe(42, classes);
   CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_probe_disabled(42));
   fail_stat = false;
   regular_fd = true;
   failed = agx_native_mesh_probe(42, classes);
   CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(!agx_native_mesh_probe_disabled(42));
   regular_fd = false;
   CHECK(opens == 6 && closes == 6);
   /* Confirmed attribute absence (ENOENT/ENOTDIR) preserves UNKNOWN for older
    * kernels. */
   fail_open = true;
   fail_open_errno = ENOENT;
   struct agx_native_mesh_support missing = agx_native_mesh_probe(42, classes);
   CHECK(!missing.qualified && missing.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN);
   CHECK(!agx_native_mesh_probe_disabled(42));
   fail_open_errno = ENOTDIR;
   missing = agx_native_mesh_probe(42, classes);
   CHECK(!missing.qualified && missing.profile == AGX_NATIVE_MESH_PROFILE_UNKNOWN);
   CHECK(!agx_native_mesh_probe_disabled(42));
   /* Transient/resource open failures (e.g. EMFILE) fail closed to NONE. */
   fail_open_errno = EMFILE;
   failed = agx_native_mesh_probe(42, classes);
   CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_probe_disabled(42));
   CHECK(agx_native_mesh_latched_profile(failed, failed) ==
         AGX_NATIVE_MESH_PROFILE_NONE);
   fail_open = false;
   CHECK(opens == 12 && closes == 6);
   fail_read = true;
   failed = agx_native_mesh_probe(42, classes);
   CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(agx_native_mesh_probe_disabled(42));
   fail_read = false;
   CHECK(opens == 14 && closes == 8);
   /* A present attribute with unparseable contents fails closed to NONE on both
    * probe entry points, including invalid generations, malformed qualified
    * hashes, malformed research payloads, truncation, and NUL corruption. */
   const char *malformed[] = {
      "asahi-native-mesh v2 future\n",
      "asahi-native-mesh v1 generation=01 profile=research classes=-\n",
      "asahi-native-mesh v1 generation=72057594037927936 profile=research classes=-\n",
      "asahi-native-mesh v1 generation=1 profile=research classes=x\n",
      "asahi-native-mesh v1 generation=1 profile=research classes=- \n",
      "asahi-native-mesh v1 generation=1 profile=qualified classes=-\n",
      "asahi-native-mesh v1 generation=1 profile=qualified classes=5550ecf263192c596c3f9d8c3b84c0324ac3d0c8ee76da37c378205247f1a3bZ\n",
      "asahi-native-mesh v1 generation=1 profile=qualified classes=5550ECF263192C596C3F9D8C3B84C0324AC3D0C8EE76DA37C378205247F1A3BD\n",
      "asahi-native-mesh v1 generation=1 profile=qualified classes=5550ecf263192c596c3f9d8c3b84c0324ac3d0c8ee76da37c378205247f1a3b\n",
      "asahi-native-mesh v1 generation=1 profile=unknown classes=-\n",
   };
   for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); ++i) {
      input_size = strlen(malformed[i]);
      memcpy(input, malformed[i], input_size);
      failed = agx_native_mesh_probe(42, classes);
      CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
      CHECK(agx_native_mesh_probe_disabled(42));
   }
   /* Truncation and single-byte NUL corruption of a valid qualified line. */
   for (size_t i = 0; i < good_size; ++i) {
      memcpy(input, good, i);
      input_size = i;
      CHECK(agx_native_mesh_probe_disabled(42));
      memcpy(input, good, good_size);
      input[i] = '\0';
      input_size = good_size;
      CHECK(agx_native_mesh_probe_disabled(42));
   }
   input_size = sizeof(input);
   memset(input, 'x', sizeof(input));
   failed = agx_native_mesh_probe(42, classes);
   CHECK(!failed.qualified && failed.profile == AGX_NATIVE_MESH_PROFILE_NONE);
   CHECK(offset == AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1);
   CHECK(agx_native_mesh_probe_disabled(42));
   CHECK(offset == AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1);
   puts("PASS: strict grammar, generation race, own-fd path, bounded reads and failures");
   return 0;
}
