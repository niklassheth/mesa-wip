/* SPDX-License-Identifier: MIT
 * Sliced 3D read/store/atomic and robustness probe. No installed files change.
 * Pass the directory containing mode0.spv ... mode4.spv.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d at %u\n", #call, r_, __LINE__); exit(1); } } while (0)
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice physical;
static VkQueue queue;
static VkCommandPool pool;
static VkDescriptorSetLayout set_layout;
static VkPipelineLayout pipeline_layout;
static VkPipeline pipelines[5];
static PFN_vkCopyMemoryToImageEXT copy_to;
static PFN_vkCopyImageToMemoryEXT copy_from;
static PFN_vkTransitionImageLayoutEXT transition;

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties p;
   vkGetPhysicalDeviceMemoryProperties(physical, &p);
   for (uint32_t i = 0; i < p.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   REQUIRE(0);
   return 0;
}

static VkDeviceMemory
allocate(VkMemoryRequirements r, VkMemoryPropertyFlags flags)
{
   VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size, .memoryTypeIndex = memory_type(r.memoryTypeBits, flags),
   };
   VkDeviceMemory memory;
   VK(vkAllocateMemory(dev, &info, NULL, &memory));
   return memory;
}

static unsigned
minify(unsigned n, unsigned level)
{
   return (n >> level) ? (n >> level) : 1;
}

static void
run_case(VkExtent3D extent, unsigned levels, unsigned level, unsigned offset,
         unsigned slice_count)
{
   unsigned w = minify(extent.width, level), h = minify(extent.height, level);
   unsigned depth = minify(extent.depth, level);
   unsigned count = slice_count == VK_REMAINING_3D_SLICES_EXT ? depth - offset : slice_count;
   REQUIRE(count && count <= 8 && offset + count <= depth);
   printf("case %ux%ux%u levels=%u level=%u offset=%u count=%u%s\n",
          extent.width, extent.height, extent.depth, levels, level, offset, count,
          slice_count == VK_REMAINING_3D_SLICES_EXT ? " remaining" : "");
   VkImageCreateInfo ii = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_3D,
      .format = VK_FORMAT_R32_UINT, .extent = extent, .mipLevels = levels,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
   };
   VkImage image;
   VK(vkCreateImage(dev, &ii, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   VkDeviceMemory image_memory = allocate(req, 0);
   VK(vkBindImageMemory(dev, image, image_memory, 0));
   VkHostImageLayoutTransitionInfoEXT ti = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
      .image = image, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1},
   };
   VK(transition(dev, 1, &ti));
   VkImageViewSlicedCreateInfoEXT slicing = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_SLICED_CREATE_INFO_EXT,
      .sliceOffset = offset, .sliceCount = slice_count,
   };
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &slicing,
      .image = image, .viewType = VK_IMAGE_VIEW_TYPE_3D, .format = ii.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1},
   };
   VkImageView view;
   VK(vkCreateImageView(dev, &vi, NULL, &view));
   VkSamplerCreateInfo si = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   VK(vkCreateSampler(dev, &si, NULL, &sampler));
   VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64 * 4,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer buffer;
   VK(vkCreateBuffer(dev, &bi, NULL, &buffer));
   vkGetBufferMemoryRequirements(dev, buffer, &req);
   VkDeviceMemory memory = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VK(vkBindBufferMemory(dev, buffer, memory, 0));
   uint32_t *result;
   VK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&result));
   VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
   };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 3, .pPoolSizes = sizes,
   };
   VkDescriptorPool dp;
   VK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
   VkDescriptorSetAllocateInfo dai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout,
   };
   VkDescriptorSet ds;
   VK(vkAllocateDescriptorSets(dev, &dai, &ds));
   VkDescriptorImageInfo image_info = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .sampler = sampler};
   VkDescriptorBufferInfo buffer_info = {.buffer = buffer, .range = bi.size};
   VkWriteDescriptorSet writes[3] = {0};
   for (unsigned i = 0; i < 3; ++i) {
      writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
         .dstBinding = i, .descriptorCount = 1, .descriptorType = sizes[i].type,
         .pImageInfo = &image_info, .pBufferInfo = &buffer_info,
      };
   }
   vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);
   size_t pixels = (size_t)w * h * depth;
   uint32_t *seed = malloc(pixels * 4), *readback = malloc(pixels * 4);
   REQUIRE(seed && readback);
   for (unsigned z = 0; z < depth; ++z)
      for (size_t xy = 0; xy < (size_t)w * h; ++xy)
         seed[z * (size_t)w * h + xy] = z + 1;
   VkMemoryToImageCopyEXT upload = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT, .pHostPointer = seed,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
      .imageExtent = {w, h, depth},
   };
   VkCopyMemoryToImageInfoEXT upload_info = {
      .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
      .dstImage = image, .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1, .pRegions = &upload,
   };
   VkImageToMemoryCopyEXT download = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT, .pHostPointer = readback,
      .imageSubresource = upload.imageSubresource, .imageExtent = upload.imageExtent,
   };
   VkCopyImageToMemoryInfoEXT download_info = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
      .srcImage = image, .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1, .pRegions = &download,
   };
   for (unsigned mode = 0; mode < 5; ++mode) {
      VK(copy_to(dev, &upload_info));
      memset(result, 0xcd, bi.size);
      VkCommandBufferAllocateInfo cai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
      };
      VkCommandBuffer cmd;
      VK(vkAllocateCommandBuffers(dev, &cai, &cmd));
      VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      VK(vkBeginCommandBuffer(cmd, &begin));
      VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[mode]);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &ds, 0, NULL);
      uint32_t params[] = {count, offset};
      vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);
      vkCmdDispatch(cmd, count + 2, 1, 1);
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
      VK(vkEndCommandBuffer(cmd));
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .commandBufferCount = 1, .pCommandBuffers = &cmd};
      VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      VK(vkQueueWaitIdle(queue));
      REQUIRE(result[16] == w && result[17] == h && result[18] == count);
      if (mode < 2) {
         REQUIRE(result[19] == depth && result[20] == 1 && result[21] == offset + 1);
      }
      for (unsigned i = 0; i < count + 2; ++i) {
         uint32_t expected = mode == 2 || mode == 3 ? 0x12345678u :
            i == 0 || i == count + 1 ? 0 : offset + i;
         if (result[i] != expected) {
            fprintf(stderr, "mode=%u result[%u]=%08x expected=%08x\n", mode, i, result[i], expected);
            exit(1);
         }
      }
      VK(copy_from(dev, &download_info));
      for (unsigned z = 0; z < depth; ++z) {
         for (size_t xy = 0; xy < (size_t)w * h; ++xy) {
            uint32_t expected = z + 1;
            if (xy == 0 && z >= offset && z < offset + count) {
               if (mode == 2 || mode == 3)
                  expected = 0x80000000u | (z - offset + 1);
               else if (mode == 4)
                  expected += 0x10000;
            }
            if (readback[z * (size_t)w * h + xy] != expected) {
               fprintf(stderr, "mode=%u image z=%u xy=%zu mismatch\n", mode, z, xy);
               exit(1);
            }
         }
      }
      REQUIRE(result[63] == 0xcdcdcdcd);
      printf("PASS mode=%u (0 typed load, 1 formatless load, 2 typed store, 3 formatless store, 4 atomic)\n", mode);
      vkFreeCommandBuffers(dev, pool, 1, &cmd);
   }
   free(seed); free(readback);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkUnmapMemory(dev, memory);
   vkDestroyBuffer(dev, buffer, NULL); vkFreeMemory(dev, memory, NULL);
   vkDestroySampler(dev, sampler, NULL); vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, image, NULL); vkFreeMemory(dev, image_memory, NULL);
}

int
main(int argc, char **argv)
{
   REQUIRE(argc == 2);
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "sliced-3d-probe", .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance instance;
   VK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t n = 1;
   VK(vkEnumeratePhysicalDevices(instance, &n, &physical));
   REQUIRE(n == 1);
   VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
   vkGetPhysicalDeviceProperties2(physical, &props);
   printf("renderer=%s driver=%s info=%s\n", props.properties.deviceName, driver.driverName, driver.driverInfo);
   REQUIRE(strstr(props.properties.deviceName, "Apple M4"));
   VkPhysicalDeviceImageSlicedViewOf3DFeaturesEXT sliced = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT};
   VkPhysicalDeviceHostImageCopyFeaturesEXT host = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT, .pNext = &sliced};
   VkPhysicalDeviceRobustness2FeaturesEXT robust = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, .pNext = &host};
   VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &robust};
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!sliced.imageSlicedViewOf3D) { puts("SKIP imageSlicedViewOf3D is false"); return 77; }
   REQUIRE(host.hostImageCopy && robust.robustImageAccess2);
   REQUIRE(features.features.shaderStorageImageReadWithoutFormat && features.features.shaderStorageImageWriteWithoutFormat);
   VkPhysicalDeviceFeatures enable = {
      .shaderStorageImageReadWithoutFormat = VK_TRUE, .shaderStorageImageWriteWithoutFormat = VK_TRUE,
   };
   robust.robustBufferAccess2 = robust.nullDescriptor = VK_FALSE;
   uint32_t count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   VkQueueFamilyProperties *q = calloc(count, sizeof(*q));
   REQUIRE(q);
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, q);
   uint32_t family = 0;
   while (family < count && !(q[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
   free(q); REQUIRE(family < count);
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
   const char *extensions[] = {VK_EXT_IMAGE_SLICED_VIEW_OF_3D_EXTENSION_NAME,
      VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &robust,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .pEnabledFeatures = &enable,
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = extensions};
   VK(vkCreateDevice(physical, &di, NULL, &dev));
   vkGetDeviceQueue(dev, family, 0, &queue);
   copy_to = (PFN_vkCopyMemoryToImageEXT)vkGetDeviceProcAddr(dev, "vkCopyMemoryToImageEXT");
   copy_from = (PFN_vkCopyImageToMemoryEXT)vkGetDeviceProcAddr(dev, "vkCopyImageToMemoryEXT");
   transition = (PFN_vkTransitionImageLayoutEXT)vkGetDeviceProcAddr(dev, "vkTransitionImageLayoutEXT");
   REQUIRE(copy_to && copy_from && transition);
   VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
   VK(vkCreateCommandPool(dev, &pi, NULL, &pool));
   VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = bindings};
   VK(vkCreateDescriptorSetLayout(dev, &li, NULL, &set_layout));
   VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
   VK(vkCreatePipelineLayout(dev, &pli, NULL, &pipeline_layout));
   for (unsigned mode = 0; mode < 5; ++mode) {
      char path[4096];
      REQUIRE(snprintf(path, sizeof(path), "%s/mode%u.spv", argv[1], mode) < (int)sizeof(path));
      FILE *f = fopen(path, "rb"); REQUIRE(f);
      REQUIRE(fseek(f, 0, SEEK_END) == 0);
      long bytes = ftell(f); REQUIRE(bytes > 0 && bytes % 4 == 0);
      rewind(f);
      uint32_t *code = malloc(bytes); REQUIRE(code);
      REQUIRE(fread(code, 1, bytes, f) == (size_t)bytes); fclose(f);
      VkShaderModuleCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = bytes, .pCode = code};
      VkShaderModule module; VK(vkCreateShaderModule(dev, &sci, NULL, &module)); free(code);
      VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
         .layout = pipeline_layout};
      VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipelines[mode]));
      vkDestroyShaderModule(dev, module, NULL);
   }
   run_case((VkExtent3D){127, 65, 33}, 6, 0, 5, 3);
   run_case((VkExtent3D){127, 65, 33}, 6, 2, 3, VK_REMAINING_3D_SLICES_EXT);
   run_case((VkExtent3D){8, 4, 129}, 8, 0, 7, 3);
   run_case((VkExtent3D){8, 4, 129}, 8, 2, 7, 5);
   run_case((VkExtent3D){8, 4, 129}, 8, 7, 0, 1);
   for (unsigned mode = 0; mode < 5; ++mode) vkDestroyPipeline(dev, pipelines[mode], NULL);
   vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroyCommandPool(dev, pool, NULL); vkDestroyDevice(dev, NULL); vkDestroyInstance(instance, NULL);
   puts("PASS all 25 sliced-storage cases; sampled view and untouched texels verified");
   return 0;
}
