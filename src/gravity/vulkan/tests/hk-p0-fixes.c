/* SPDX-License-Identifier: MIT */
/* Regression tests for Vulkan P0 correctness fixes:
 *  - Command buffer begin/end/trim/reset lifecycle (P0-V1/V2)
 *  - Mutable SSBO 256B descriptor stride lowering (P0-V4)
 *  - Consecutive inline UBO writes/copies, exact-capacity variable-count
 *    descriptor pools, and layout support overflow checks (P0-V5/V6/V7)
 *  - 64-bit timeline semaphore wait deduplication (P0-V9)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_bo.h"
#include "agx_device.h"
#include "drm-uapi/asahi_drm.h"
#include "hk_cmd_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "hk_shader.h"
#include "nir_builder.h"
#include "vk_drm_syncobj.h"
#include "vk_pipeline.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: FAIL: %s\n", __FILE__, __LINE__, #c);         \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static unsigned mock_live_bos;
static uint64_t mock_next_gpu = 0x10000000ULL;

struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size,
                                    unsigned align, enum agx_bo_flags flags,
                                    const char *label);
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo);
void __wrap_agx_bo_cache_evict_all(struct agx_device *dev);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   struct agx_va *va = calloc(1, sizeof(*va));
   void *mem = calloc(1, size ? size : 16);
   if (!bo || !va || !mem) {
      free(bo);
      free(va);
      free(mem);
      return NULL;
   }
   va->addr = mock_next_gpu;
   va->size_B = size;
   mock_next_gpu += align64(size + 0x4000, 0x4000);
   bo->dev = dev;
   bo->size = size;
   bo->va = va;
   bo->_map = mem;
   mock_live_bos++;
   return bo;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   assert(mock_live_bos > 0);
   mock_live_bos--;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

void
__wrap_agx_bo_cache_evict_all(struct agx_device *dev)
{
}

static void
init_mock_device(struct hk_physical_device *pdev, struct hk_device *dev)
{
   memset(pdev, 0, sizeof(*pdev));
   memset(dev, 0, sizeof(*dev));
   pdev->dev.chip = AGX_CHIP_G16G;
   pdev->dev.params.gpu_generation = 16;
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.params.gpu_generation = 16;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
}

/* Test 1: vkBeginCommandBuffer / vkEndCommandBuffer / vkTrimCommandPool /
 * vkResetCommandBuffer lifecycle (P0-V1/V2).
 */
static int
test_cmd_buffer_lifecycle_and_trim(void)
{
   struct hk_physical_device pdev;
   struct hk_device dev;
   init_mock_device(&pdev, &dev);

   VkCommandPoolCreateInfo pool_ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool pool_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateCommandPool(hk_device_to_handle(&dev), &pool_ci, NULL,
                              &pool_handle) == VK_SUCCESS);
   VK_FROM_HANDLE(hk_cmd_pool, pool, pool_handle);

   struct vk_command_buffer *vk_cb1 = NULL, *vk_cb2 = NULL;
   CHECK(hk_cmd_buffer_ops.create(&pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                  &vk_cb1) == VK_SUCCESS);
   CHECK(hk_cmd_buffer_ops.create(&pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                  &vk_cb2) == VK_SUCCESS);
   struct hk_cmd_buffer *cb1 = container_of(vk_cb1, struct hk_cmd_buffer, vk);
   struct hk_cmd_buffer *cb2 = container_of(vk_cb2, struct hk_cmd_buffer, vk);

   CHECK(cb1->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INITIAL);
   CHECK(cb2->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INITIAL);

   /* Begin CB1, populate recorded arrays + pinned large BO, then end CB1. */
   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(hk_BeginCommandBuffer(hk_cmd_buffer_to_handle(cb1), &begin_info) ==
         VK_SUCCESS);
   CHECK(cb1->vk.state == MESA_VK_COMMAND_BUFFER_STATE_RECORDING);

   uint32_t dummy_word = 0x60000160;
   hk_cmd_record_narrow_cdm_barrier(cb1, &dummy_word);
   uint64_t dummy_fb = 0xdeadbeefULL;
   util_dynarray_append(&cb1->feedback_raw, dummy_fb);
   struct agx_bo *pinned = agx_bo_create(&dev.dev, 4096, 0, 0, "pinned");
   CHECK(pinned != NULL);
   util_dynarray_append(&cb1->large_bos, pinned);

   CHECK(hk_EndCommandBuffer(hk_cmd_buffer_to_handle(cb1)) == VK_SUCCESS);
   CHECK(cb1->vk.state == MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE);

   /* Give INITIAL CB2 non-empty dynarray capacities (as after a non-releasing
    * reset) and verify vkTrimCommandPool frees CB2's arrays while preserving
    * EXECUTABLE CB1 intact.
    */
   hk_cmd_record_narrow_cdm_barrier(cb2, &dummy_word);
   util_dynarray_clear(&cb2->cdm_narrow_barriers);
   util_dynarray_append(&cb2->feedback_raw, dummy_fb);
   util_dynarray_clear(&cb2->feedback_raw);
   CHECK(cb2->cdm_narrow_barriers.capacity > 0);
   CHECK(cb2->feedback_raw.capacity > 0);

   hk_TrimCommandPool(hk_device_to_handle(&dev), pool_handle, 0);

   /* CB1 (EXECUTABLE) must NOT have been touched by TrimCommandPool! */
   CHECK(cb1->vk.state == MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE);
   CHECK(util_dynarray_num_elements(&cb1->cdm_narrow_barriers, uint32_t *) == 1);
   CHECK(*util_dynarray_element(&cb1->cdm_narrow_barriers, uint32_t *, 0) ==
         &dummy_word);
   CHECK(util_dynarray_num_elements(&cb1->feedback_raw, uint64_t) == 1);
   CHECK(util_dynarray_num_elements(&cb1->large_bos, struct agx_bo *) == 1);
   CHECK(mock_live_bos == 1);

   /* CB2 (INITIAL) MUST have had cdm_narrow_barriers and feedback_raw freed! */
   CHECK(cb2->cdm_narrow_barriers.data == NULL);
   CHECK(cb2->cdm_narrow_barriers.capacity == 0);
   CHECK(cb2->feedback_raw.data == NULL);
   CHECK(cb2->feedback_raw.capacity == 0);

   /* Releasing reset via public vkResetCommandBuffer entrypoint must transition
    * to INITIAL, unref large_bos, and free cdm_narrow_barriers + feedback_raw.
    */
   CHECK(hk_ResetCommandBuffer(hk_cmd_buffer_to_handle(cb1),
                               VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) ==
         VK_SUCCESS);
   CHECK(cb1->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INITIAL);
   CHECK(cb1->cdm_narrow_barriers.data == NULL);
   CHECK(cb1->feedback_raw.data == NULL);
   CHECK(mock_live_bos == 0);

   hk_DestroyCommandPool(hk_device_to_handle(&dev), pool_handle, NULL);
   CHECK(mock_live_bos == 0);
   return 0;
}

/* Test 2: Mutable SSBO 256B stride lowering (P0-V4). */
static int
test_mutable_ssbo_256b_stride(void)
{
   struct hk_physical_device pdev;
   struct hk_device dev;
   init_mock_device(&pdev, &dev);

   const VkDescriptorType mutable_types[] = {
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
   };
   const VkMutableDescriptorTypeListEXT mutable_list = {
      .descriptorTypeCount = ARRAY_SIZE(mutable_types),
      .pDescriptorTypes = mutable_types,
   };
   const VkMutableDescriptorTypeCreateInfoEXT mutable_ci = {
      .sType = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
      .mutableDescriptorTypeListCount = 1,
      .pMutableDescriptorTypeLists = &mutable_list,
   };
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_MUTABLE_EXT,
      .descriptorCount = 4,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo dsl_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &mutable_ci,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VkDescriptorSetLayout dsl_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateDescriptorSetLayout(hk_device_to_handle(&dev), &dsl_ci, NULL,
                                      &dsl_handle) == VK_SUCCESS);
   VK_FROM_HANDLE(hk_descriptor_set_layout, layout, dsl_handle);
   CHECK(layout->binding[0].stride == 256);

   /* Build minimal compute NIR using vulkan_resource_index + reindex + load. */
   const nir_shader_compiler_options nir_opts = {0};
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &nir_opts, "ssbo256");
   nir_def *idx = nir_load_local_invocation_index(&b);
   nir_def *res = nir_vulkan_resource_index(&b, 4, 32, idx, .desc_set = 0,
                                            .binding = 0,
                                            .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *reindexed = nir_vulkan_resource_reindex(
      &b, 32, res, nir_imm_int(&b, 1),
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      &b, 2, 32, reindexed, .desc_type = nir_descriptor_type_storage_buffer);
   nir_store_global(&b, desc, nir_imm_int64(&b, 0x1000));

   struct vk_pipeline_robustness_state rs = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
   };
   struct vk_descriptor_set_layout *layouts[] = {&layout->vk};
   CHECK(hk_nir_lower_descriptors(b.shader, &rs, 1, layouts, false));

   /* Verify:
    * 1) binding_addr gets tagged with (256 >> 4) << 56 == 16ULL << 56
    * 2) binding_size == 4 * 256 == 1024 (NOT 0!)
    * 3) reindex decodes (addr_high32 >> 24) << 4
    */
   bool found_tag = false;
   bool found_binding_size = false;
   bool found_shift_decode = false;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_ior && alu->def.bit_size == 64 &&
             nir_src_is_const(alu->src[1].src) &&
             nir_src_as_uint(alu->src[1].src) == ((uint64_t)(256 >> 4) << 56))
            found_tag = true;
         if (alu->op == nir_op_vec4 && alu->def.num_components == 4 &&
             nir_src_is_const(alu->src[2].src) &&
             nir_src_as_uint(alu->src[2].src) == 4 * 256)
            found_binding_size = true;
         if (alu->op == nir_op_ishl && nir_src_is_const(alu->src[1].src) &&
             nir_src_as_uint(alu->src[1].src) == 4) {
            nir_scalar s = nir_scalar_resolved(alu->src[0].src.ssa, 0);
            if (nir_scalar_is_alu(s) &&
                nir_scalar_alu_op(s) == nir_op_ushr) {
               nir_alu_instr *shr = nir_instr_as_alu(nir_def_instr(s.def));
               if (nir_src_is_const(shr->src[1].src) &&
                   nir_src_as_uint(shr->src[1].src) == 24)
                  found_shift_decode = true;
            }
         }
      }
   }
   CHECK(found_tag);
   CHECK(found_binding_size);
   CHECK(found_shift_decode);

   ralloc_free(b.shader);
   vk_descriptor_set_layout_unref(&dev.vk, &layout->vk);
   return 0;
}

/* Test 3: Consecutive inline UBO write/copy, exact-capacity variable-count
 * descriptor pool, and layout support overflow checks (P0-V5/V6/V7).
 */
static int
test_inline_ubo_and_variable_pool(void)
{
   struct hk_physical_device pdev;
   struct hk_device dev;
   init_mock_device(&pdev, &dev);

   /* 3a: Two consecutive 4B inline UBO bindings have 64B alignment gap between
    * them (offset 0 and offset 64). An 8-byte write/copy starting at binding 0
    * must place bytes 0..3 at offset 0 and bytes 4..7 at offset 64, leaving
    * padding bytes 4..63 untouched.
    */
   const VkDescriptorSetLayoutBinding ubo_bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
         .descriptorCount = 4,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
         .descriptorCount = 4,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo ubo_dsl_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = ubo_bindings,
   };
   VkDescriptorSetLayout ubo_dsl_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateDescriptorSetLayout(hk_device_to_handle(&dev), &ubo_dsl_ci,
                                      NULL, &ubo_dsl_handle) == VK_SUCCESS);
   VK_FROM_HANDLE(hk_descriptor_set_layout, ubo_layout, ubo_dsl_handle);
   CHECK(ubo_layout->binding[0].offset == 0);
   CHECK(ubo_layout->binding[1].offset == HK_MIN_UBO_ALIGNMENT);

   const VkDescriptorPoolSize ubo_pool_size = {
      .type = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
      .descriptorCount = 16,
   };
   const VkDescriptorPoolInlineUniformBlockCreateInfo ubo_pool_inline = {
      .sType =
         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO,
      .maxInlineUniformBlockBindings = 4,
   };
   const VkDescriptorPoolCreateInfo ubo_pool_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = &ubo_pool_inline,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &ubo_pool_size,
   };
   VkDescriptorPool ubo_pool_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateDescriptorPool(hk_device_to_handle(&dev), &ubo_pool_ci, NULL,
                                 &ubo_pool_handle) == VK_SUCCESS);

   VkDescriptorSetLayout layouts_two[2] = {ubo_dsl_handle, ubo_dsl_handle};
   VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   const VkDescriptorSetAllocateInfo alloc_two = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = ubo_pool_handle,
      .descriptorSetCount = 2,
      .pSetLayouts = layouts_two,
   };
   CHECK(hk_AllocateDescriptorSets(hk_device_to_handle(&dev), &alloc_two,
                                   sets) == VK_SUCCESS);

   const uint8_t payload[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
   const VkWriteDescriptorSetInlineUniformBlock write_inline = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
      .dataSize = sizeof(payload),
      .pData = payload,
   };
   const VkWriteDescriptorSet write_ds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = &write_inline,
      .dstSet = sets[0],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = sizeof(payload),
      .descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
   };
   hk_UpdateDescriptorSets(hk_device_to_handle(&dev), 1, &write_ds, 0, NULL);

   VK_FROM_HANDLE(hk_descriptor_set, src_set, sets[0]);
   VK_FROM_HANDLE(hk_descriptor_set, dst_set, sets[1]);
   const uint8_t *src_mem = (const uint8_t *)src_set->mapped_ptr;
   CHECK(memcmp(src_mem, payload, 4) == 0);
   for (unsigned i = 4; i < HK_MIN_UBO_ALIGNMENT; ++i)
      CHECK(src_mem[i] == 0);
   CHECK(memcmp(src_mem + HK_MIN_UBO_ALIGNMENT, payload + 4, 4) == 0);

   const VkCopyDescriptorSet copy_ds = {
      .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
      .srcSet = sets[0],
      .srcBinding = 0,
      .srcArrayElement = 0,
      .dstSet = sets[1],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = sizeof(payload),
   };
   hk_UpdateDescriptorSets(hk_device_to_handle(&dev), 0, NULL, 1, &copy_ds);

   const uint8_t *dst_mem = (const uint8_t *)dst_set->mapped_ptr;
   CHECK(memcmp(dst_mem, payload, 4) == 0);
   for (unsigned i = 4; i < HK_MIN_UBO_ALIGNMENT; ++i)
      CHECK(dst_mem[i] == 0);
   CHECK(memcmp(dst_mem + HK_MIN_UBO_ALIGNMENT, payload + 4, 4) == 0);

   hk_DestroyDescriptorPool(hk_device_to_handle(&dev), ubo_pool_handle, NULL);
   vk_descriptor_set_layout_unref(&dev.vk, &ubo_layout->vk);

   /* 3b: Variable-count pool sized to exact descriptor count (131072 sampled
    * images, maxSets=1) must succeed allocation even with trailing sentinel.
    */
   const uint32_t var_count = 131072;
   const VkDescriptorBindingFlags var_flags =
      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo var_flags_ci = {
      .sType =
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = 1,
      .pBindingFlags = &var_flags,
   };
   const VkDescriptorSetLayoutBinding var_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = var_count,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo var_dsl_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &var_flags_ci,
      .bindingCount = 1,
      .pBindings = &var_binding,
   };
   VkDescriptorSetLayout var_dsl_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateDescriptorSetLayout(hk_device_to_handle(&dev), &var_dsl_ci,
                                      NULL, &var_dsl_handle) == VK_SUCCESS);

   const VkDescriptorPoolSize var_pool_size = {
      .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = var_count,
   };
   const VkDescriptorPoolCreateInfo var_pool_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &var_pool_size,
   };
   VkDescriptorPool var_pool_handle = VK_NULL_HANDLE;
   CHECK(hk_CreateDescriptorPool(hk_device_to_handle(&dev), &var_pool_ci, NULL,
                                 &var_pool_handle) == VK_SUCCESS);

   const VkDescriptorSetVariableDescriptorCountAllocateInfo var_alloc_ci = {
      .sType =
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
      .descriptorSetCount = 1,
      .pDescriptorCounts = &var_count,
   };
   const VkDescriptorSetAllocateInfo var_alloc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = &var_alloc_ci,
      .descriptorPool = var_pool_handle,
      .descriptorSetCount = 1,
      .pSetLayouts = &var_dsl_handle,
   };
   VkDescriptorSet var_set = VK_NULL_HANDLE;
   CHECK(hk_AllocateDescriptorSets(hk_device_to_handle(&dev), &var_alloc,
                                   &var_set) == VK_SUCCESS);

   hk_DestroyDescriptorPool(hk_device_to_handle(&dev), var_pool_handle, NULL);
   VK_FROM_HANDLE(hk_descriptor_set_layout, var_layout, var_dsl_handle);
   vk_descriptor_set_layout_unref(&dev.vk, &var_layout->vk);
   CHECK(mock_live_bos == 0);

   /* 3c: Layout support must reject 32-bit stride*count wrap (2^24 * 256B =
    * 4 GiB) and 8-bit dynamic buffer count wrap (256 dynamic UBOs).
    */
   const VkDescriptorSetLayoutBinding wrap_size_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1u << 24,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo wrap_size_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &wrap_size_binding,
   };
   VkDescriptorSetLayoutSupport support = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
   };
   hk_GetDescriptorSetLayoutSupport(hk_device_to_handle(&dev), &wrap_size_ci,
                                    &support);
   CHECK(support.supported == VK_FALSE);

   const VkDescriptorSetLayoutBinding wrap_dyn_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      .descriptorCount = 256,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo wrap_dyn_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &wrap_dyn_binding,
   };
   hk_GetDescriptorSetLayoutSupport(hk_device_to_handle(&dev), &wrap_dyn_ci,
                                    &support);
   CHECK(support.supported == VK_FALSE);

   return 0;
}

/* Test 4: 64-bit timeline semaphore wait deduplication (P0-V9). */
static int
test_timeline_wait_dedup_64bit(void)
{
   const struct vk_sync_type syncobj_type = {
      .finish = vk_drm_syncobj_finish,
   };
   struct vk_drm_syncobj tl_a = {
      .base = {
         .type = &syncobj_type,
         .flags = VK_SYNC_IS_TIMELINE,
      },
      .syncobj = 11,
   };
   struct vk_drm_syncobj tl_b = {
      .base = {
         .type = &syncobj_type,
         .flags = VK_SYNC_IS_TIMELINE,
      },
      .syncobj = 22,
   };
   struct vk_drm_syncobj bin_c = {
      .base = {
         .type = &syncobj_type,
         .flags = 0,
      },
      .syncobj = 33,
   };

   /* 0xffffffff vs 0x100000000: uint32_t truncation saw 0xffffffff > 0 and
    * dropped the real maximum 0x100000000! Also test duplicate equal 64-bit
    * values, a second timeline syncobj, and a binary syncobj.
    */
   const struct vk_sync_wait waits[] = {
      {.sync = &tl_a.base, .wait_value = 0xffffffffULL},
      {.sync = &tl_a.base, .wait_value = 0x100000000ULL},
      {.sync = &tl_b.base, .wait_value = 0x200000005ULL},
      {.sync = &tl_b.base, .wait_value = 0x200000005ULL},
      {.sync = &bin_c.base, .wait_value = 0},
   };
   struct drm_asahi_sync out[ARRAY_SIZE(waits)];
   unsigned n = hk_fill_submit_wait_syncs(out, waits, ARRAY_SIZE(waits));
   CHECK(n == 3);
   CHECK(out[0].handle == 11 &&
         out[0].sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ &&
         out[0].timeline_value == 0x100000000ULL);
   CHECK(out[1].handle == 22 &&
         out[1].sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ &&
         out[1].timeline_value == 0x200000005ULL);
   CHECK(out[2].handle == 33 && out[2].sync_type == DRM_ASAHI_SYNC_SYNCOBJ);

   /* Reverse order: 0x100000000 before 0xffffffff still keeps 0x100000000. */
   const struct vk_sync_wait rev_waits[] = {
      {.sync = &tl_a.base, .wait_value = 0x100000000ULL},
      {.sync = &tl_a.base, .wait_value = 0xffffffffULL},
   };
   n = hk_fill_submit_wait_syncs(out, rev_waits, ARRAY_SIZE(rev_waits));
   CHECK(n == 1);
   CHECK(out[0].handle == 11 && out[0].timeline_value == 0x100000000ULL);

   return 0;
}

int
main(void)
{
   if (test_cmd_buffer_lifecycle_and_trim())
      return 1;
   if (test_mutable_ssbo_256b_stride())
      return 1;
   if (test_inline_ubo_and_variable_pool())
      return 1;
   if (test_timeline_wait_dedup_64bit())
      return 1;
   return 0;
}
