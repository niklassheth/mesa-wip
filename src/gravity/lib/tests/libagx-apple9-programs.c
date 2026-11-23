/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

/* libagx_shaders.h is C-only, so program IDs are resolved here. */

#include <stdbool.h>
#include "libagx_shaders.h"

unsigned test_libagx_num_programs(void);
bool test_libagx_expected_to_fail(unsigned program);
unsigned test_libagx_fast_clear_4x(void);

unsigned
test_libagx_fast_clear_4x(void)
{
   return LIBAGX_FAST_CLEAR(2 /* four-byte pixels */, 2 /* four samples */);
}

unsigned
test_libagx_num_programs(void)
{
   return LIBAGX_NUM_PROGRAMS;
}

/* Programs that cannot compile for Apple9, each unreachable there. An entry
 * that starts compiling should be removed.
 */
bool
test_libagx_expected_to_fail(unsigned program)
{
   (void)program;
   return false;
}
