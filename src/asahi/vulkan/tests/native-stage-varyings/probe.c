/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * One ordinary GS or TES draw with full RGBA32F readback. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#define W 128
#define H 64
#define CHECK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { fprintf(stderr, "%s: %d\n", #call, r); exit(2); } } while (0)
static VkDevice dev;
static VkPhysicalDevice phys;
static uint32_t memory_type(uint32_t mask, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties p;
   vkGetPhysicalDeviceMemoryProperties(phys, &p);
   for (unsigned i = 0; i < p.memoryTypeCount; ++i)
      if ((mask & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   abort();
}
static VkShaderModule module(const char *dir, const char *name)
{
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s.spv", dir, name);
   FILE *f = fopen(path, "rb");
   if (!f || fseek(f, 0, SEEK_END)) exit(3);
   long n = ftell(f);
   rewind(f);
   if (n <= 0 || n % 4 || n > 1048576) exit(3);
   uint32_t *data = malloc(n);
   if (!data || fread(data, 1, n, f) != (size_t)n || fclose(f)) exit(3);
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = n, .pCode = data};
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   free(data);
   return m;
}
static void in_flight(VkResult r)
{
   if (r == VK_SUCCESS) return;
   fprintf(stderr, "IN_FLIGHT_WAIT_FAILED result=%d; retaining backing\n", r);
   fflush(NULL);
   raise(SIGSTOP);
   for (;;) pause();
}
int main(int argc, char **argv)
{
   if (argc != 4) return 3;
   bool gs = !strncmp(argv[3], "gs", 2);
   if (strcmp(argv[3], "gs3") && strcmp(argv[3], "gs4") && strcmp(argv[3], "tri") && strcmp(argv[3], "quad")) return 3;
   setvbuf(stdout, NULL, _IOLBF, 0);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "native-stage-varyings", .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t count = 1;
   VkResult found = vkEnumeratePhysicalDevices(inst, &count, &phys);
   if ((found != VK_SUCCESS && found != VK_INCOMPLETE) || !count) return 2;
   VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
   vkGetPhysicalDeviceProperties2(phys, &props);
   printf("DEVICE %s | %s | %s\n", props.properties.deviceName, driver.driverName, driver.driverInfo);
   VkPhysicalDeviceVulkan13Features v13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13,
      .features = {.geometryShader = VK_TRUE, .tessellationShader = VK_TRUE}};
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(vkCreateDevice(phys, &di, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
   char shader[64];
   snprintf(shader, sizeof(shader), "%s.%s", argv[3], gs ? "geom" : "tese");
   VkShaderModule mods[4] = {module(argv[1], gs ? "gs.vert" : "tes.vert"),
      module(argv[1], "vary.frag"), module(argv[1], shader), VK_NULL_HANDLE};
   VkPipelineShaderStageCreateInfo stages[4] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mods[0], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mods[1], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = gs ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, .module = mods[2], .pName = "main"}};
   if (!gs) {
      mods[3] = module(argv[1], "vary.tesc");
      stages[3] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, .module = mods[3], .pName = "main"};
   }
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = gs ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
   VkPipelineTessellationStateCreateInfo ts = {.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, .patchControlPoints = 3};
   VkViewport viewport = {0, 0, W, H, 0, 1};
   VkRect2D scissor = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState blend = {.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE, .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend};
   VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
   VkPipelineRenderingCreateInfo pr = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
   VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &pr,
      .stageCount = gs ? 3 : 4, .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pTessellationState = gs ? NULL : &ts, .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = layout};
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipe));
   puts("PIPELINE_OK");
   VkImageCreateInfo im = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = format, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image;
   CHECK(vkCreateImage(dev, &im, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory image_mem;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &image_mem));
   CHECK(vkBindImageMemory(dev, image, image_mem, 0));
   VkImageViewCreateInfo view_ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &view_ci, NULL, &view));
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = W * H * 16,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &bi, NULL, &buffer));
   vkGetBufferMemoryRequirements(dev, buffer, &req);
   alloc.allocationSize = req.size;
   alloc.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory buffer_mem;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &buffer_mem));
   CHECK(vkBindBufferMemory(dev, buffer, buffer_mem, 0));
   void *pixels;
   CHECK(vkMapMemory(dev, buffer_mem, 0, VK_WHOLE_SIZE, 0, &pixels));
   memset(pixels, 0xa5, W * H * 16);
   VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cp, NULL, &pool));
   VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &ca, &cmd));
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, 0, NULL, 0, NULL, 1, &barrier);
   VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = scissor,
      .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att};
   vkCmdBeginRendering(cmd, &render);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(cmd, gs ? 33 : 3, 1, 0, 0);
   vkCmdEndRendering(cmd);
   barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   barrier.oldLayout = barrier.newLayout;
   barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &barrier);
   VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   puts("COMMAND_OK");
   VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fi, NULL, &fence));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
   puts("SUBMIT_BEGIN");
   in_flight(vkQueueSubmit(queue, 1, &submit, fence));
   in_flight(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
   puts("SUBMIT_OK");
   char path[4096];
   snprintf(path, sizeof(path), "%s/pixels.bin", argv[2]);
   FILE *f = fopen(path, "wb");
   if (!f || fwrite(pixels, 1, W * H * 16, f) != W * H * 16 || fclose(f)) return 3;
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkUnmapMemory(dev, buffer_mem);
   vkDestroyBuffer(dev, buffer, NULL); vkFreeMemory(dev, buffer_mem, NULL);
   vkDestroyImageView(dev, view, NULL); vkDestroyImage(dev, image, NULL); vkFreeMemory(dev, image_mem, NULL);
   vkDestroyPipeline(dev, pipe, NULL); vkDestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 4; ++i) if (mods[i]) vkDestroyShaderModule(dev, mods[i], NULL);
   vkDestroyDevice(dev, NULL); vkDestroyInstance(inst, NULL);
   return 0;
}
