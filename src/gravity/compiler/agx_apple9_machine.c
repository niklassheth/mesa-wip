/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_machine.h"
#include "agx_apple9_machine_table.h"

#include <assert.h>
#include <stddef.h>

const struct agx_apple9_machine agx_apple9_machine = {
   .gpr_count = AGX_APPLE9_GPR_COUNT,
   .half_register_count = AGX_APPLE9_HALF_REGISTER_COUNT,
   .hardware_register_interlocks = true,
   .software_waits = false,
   .spilling_supported = true,
   .occupancy_model = AGX_APPLE9_OCCUPANCY_UNMEASURED,
};

const struct agx_apple9_encoding_info *
agx_apple9_encoding_info(enum agx_apple9_encoding encoding)
{
   assert(encoding < AGX_APPLE9_ENC_COUNT);
   return &agx_apple9_encodings[encoding];
}

const struct agx_apple9_operand_constraint *
agx_apple9_find_operand(enum agx_apple9_encoding encoding,
                        enum agx_apple9_operand_role role)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(encoding);

   for (unsigned i = 0; i < info->operand_count; ++i) {
      if (info->operands[i].role == role)
         return &info->operands[i];
   }

   return NULL;
}

bool
agx_apple9_encoding_accepts_gpr(enum agx_apple9_encoding encoding,
                                enum agx_apple9_operand_role role, unsigned gpr,
                                unsigned bits)
{
   const struct agx_apple9_operand_constraint *operand =
      agx_apple9_find_operand(encoding, role);

   if (operand == NULL || !(operand->files & AGX_APPLE9_FILE_GPR) ||
       !(operand->flags & AGX_APPLE9_OPERAND_ALLOCATABLE) ||
       gpr >= ((operand->flags & AGX_APPLE9_OPERAND_PUBLICATION)
                  ? AGX_APPLE9_PUBLICATION_COUNT : AGX_APPLE9_GPR_COUNT) ||
       gpr < operand->min_index ||
       gpr > operand->max_index)
      return false;

   unsigned width = bits == 16   ? AGX_APPLE9_WIDTH_16
                    : bits == 32 ? AGX_APPLE9_WIDTH_32
                    : bits == 64 ? AGX_APPLE9_WIDTH_64
                                 : 0;

   if (!(operand->widths & width))
      return false;

   unsigned first_half = 2 * gpr;
   unsigned natural_alignment = bits / 16;
   unsigned alignment = operand->alignment_halves > natural_alignment
                           ? operand->alignment_halves
                           : natural_alignment;
   return (first_half % alignment) == 0;
}

bool
agx_apple9_encoding_accepts_gpr_tuple(enum agx_apple9_encoding encoding,
                                      const unsigned *gprs,
                                      unsigned operand_count, unsigned bits)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(encoding);

   unsigned gpr_operand_count = 0;
   for (unsigned i = 0; i < info->operand_count; ++i)
      gpr_operand_count += !!(info->operands[i].files & AGX_APPLE9_FILE_GPR);

   if (operand_count != gpr_operand_count)
      return false;

   unsigned gpr_index = 0;
   for (unsigned i = 0; i < info->operand_count; ++i) {
      const struct agx_apple9_operand_constraint *operand = &info->operands[i];
      if (!(operand->files & AGX_APPLE9_FILE_GPR))
         continue;
      if (!agx_apple9_encoding_accepts_gpr(encoding, operand->role,
                                           gprs[gpr_index], bits))
         return false;
      ++gpr_index;
   }

   return true;
}
