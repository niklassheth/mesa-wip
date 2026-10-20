/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * One ordinary VS or native mesh draw with independent RGBA8/D32 readback.
 * No private shader substitution or command-stream patching.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d\n", #x, r); exit(2); } } while (0)
static VKAPI_ATTR VkBool32 VKAPI_CALL
message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
   (void)severity; (void)type; (void)user;
   fprintf(stderr, "Vulkan: %s\n", data->pMessage); return VK_FALSE;
}
static VkDevice dev;
static VkPhysicalDevice phys;
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties m;
   vkGetPhysicalDeviceMemoryProperties(phys, &m);
   for (unsigned i = 0; i < m.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   exit(3);
}
static VkShaderModule module(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f || fseek(f, 0, SEEK_END)) exit(4);
   long n = ftell(f); rewind(f);
   if (n <= 0 || n > 1048576 || n % 4) exit(4);
   uint32_t *data = malloc(n);
   if (!data || fread(data, 1, n, f) != (size_t)n || fclose(f)) exit(4);
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = n, .pCode = data};
   VkShaderModule s; CHECK(vkCreateShaderModule(dev, &ci, NULL, &s));
   free(data); return s;
}
struct attachment { VkImage image; VkImageView view; VkDeviceMemory memory; };
static struct attachment attachment(bool depth)
{
   struct attachment a;
   VkFormat format = depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format, .extent = {64, 64, 1},
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)};
   CHECK(vkCreateImage(dev, &ci, NULL, &a.image));
   VkMemoryRequirements r; vkGetImageMemoryRequirements(dev, a.image, &r);
   VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size, .memoryTypeIndex = memory_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &a.memory));
   CHECK(vkBindImageMemory(dev, a.image, a.memory, 0));
   VkImageViewCreateInfo v = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = a.image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   CHECK(vkCreateImageView(dev, &v, NULL, &a.view)); return a;
}
int main(int argc, char **argv)
{
   if (argc != 8) { fprintf(stderr, "probe vertex-or-mesh.spv fragment.spv halfz color.bin depth.bin dry-run q.bin\n"); return 1; }
   bool mesh = strstr(argv[1], ".mesh.spv") != NULL;
   bool halfz = atoi(argv[3]), dry = atoi(argv[6]);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "native-limits", .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   const char *debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
   ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = &debug_extension;
   VkInstance instance; CHECK(vkCreateInstance(&ici, NULL, &instance));
   VkDebugUtilsMessengerCreateInfoEXT debug = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
      .pfnUserCallback = message};
   PFN_vkCreateDebugUtilsMessengerEXT create_debug = (void *)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
   PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug = (void *)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
   if (!create_debug || !destroy_debug) return 11;
   VkDebugUtilsMessengerEXT messenger; CHECK(create_debug(instance, &debug, NULL, &messenger));
   uint32_t count = 0; CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
   for (unsigned i = 0; i < count; ++i) {
      VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
      vkGetPhysicalDeviceProperties2(devices[i], &props);
      if (driver.driverID != VK_DRIVER_ID_MESA_HONEYKRISP) continue;
      phys = devices[i];
      printf("renderer=%s driver=%s info=%s\n", props.properties.deviceName, driver.driverName, driver.driverInfo);
      break;
   }
   free(devices); if (!phys) return 5;
   VkPhysicalDeviceDepthClipControlFeaturesEXT clip = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
   VkPhysicalDeviceMeshShaderFeaturesEXT mf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, .pNext = &clip};
   VkPhysicalDeviceVulkan13Features v13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &mf};
   VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13};
   vkGetPhysicalDeviceFeatures2(phys, &features);
   if (!v13.dynamicRendering || !clip.depthClipControl || (mesh && !mf.meshShader)) return 6;
   clip.depthClipControl = VK_TRUE;
   mf = (VkPhysicalDeviceMeshShaderFeaturesEXT){.sType = mf.sType, .pNext = &clip, .meshShader = mesh};
   v13 = (VkPhysicalDeviceVulkan13Features){.sType = v13.sType, .pNext = mesh ? (void *)&mf : (void *)&clip, .dynamicRendering = VK_TRUE};
   if (!features.features.vertexPipelineStoresAndAtomics) return 6;
   features = (VkPhysicalDeviceFeatures2){.sType = features.sType, .pNext = &v13,
      .features = {.vertexPipelineStoresAndAtomics = VK_TRUE}};
   uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
   VkQueueFamilyProperties *families = calloc(n, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families);
   uint32_t qfam = 0; while (qfam < n && !(families[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qfam;
   free(families); if (qfam == n) return 7;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &priority};
   const char *extensions[] = {VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = mesh ? 2 : 1, .ppEnabledExtensionNames = extensions};
   CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
   VkQueue queue; vkGetDeviceQueue(dev, qfam, 0, &queue);
   VkDescriptorSetLayoutBinding binding = {.binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT};
   VkDescriptorSetLayoutCreateInfo slci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding};
   VkDescriptorSetLayout set_layout; CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &set_layout));
   VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout};
   VkPipelineLayout layout; CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT,
       .module = module(argv[1]), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = module(argv[2]), .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 64, 64, -64, 0, 1}; VkRect2D scissor = {{0, 0}, {64, 64}};
   VkPipelineViewportDepthClipControlCreateInfoEXT dc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
      .negativeOneToOne = halfz};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = &dc, .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_ALWAYS};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ba};
   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo rendering = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format, .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT};
   VkGraphicsPipelineCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pDepthStencilState = &ds, .pColorBlendState = &cb, .layout = layout};
   VkPipeline pipeline; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline));
   puts("PIPELINE_OK"); fflush(stdout);
   struct attachment attachments[2] = {attachment(false), attachment(true)};
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 32768, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer; CHECK(vkCreateBuffer(dev, &bci, NULL, &buffer));
   VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, buffer, &r);
   VkMemoryAllocateInfo bai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size,
      .memoryTypeIndex = memory_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory memory; CHECK(vkAllocateMemory(dev, &bai, NULL, &memory)); CHECK(vkBindBufferMemory(dev, buffer, memory, 0));
   void *pixels; CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &pixels)); memset(pixels, 0xa5, 32768);
   VkBufferCreateInfo qbci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 16384, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   VkBuffer qbuf; CHECK(vkCreateBuffer(dev, &qbci, NULL, &qbuf));
   vkGetBufferMemoryRequirements(dev, qbuf, &r);
   bai.allocationSize = r.size;
   bai.memoryTypeIndex = memory_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory qmem; CHECK(vkAllocateMemory(dev, &bai, NULL, &qmem));
   CHECK(vkBindBufferMemory(dev, qbuf, qmem, 0));
   uint32_t *q; CHECK(vkMapMemory(dev, qmem, 0, VK_WHOLE_SIZE, 0, (void **)&q));
   memset(q, 0xa5, 16384);
   for (unsigned i = 0; i < 128; ++i) q[16 * i] = 0;
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
   VkDescriptorSet set; CHECK(vkAllocateDescriptorSets(dev, &sai, &set));
   VkDescriptorBufferInfo bi = {qbuf, 0, 16384};
   VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi};
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);
   VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam};
   VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cp, NULL, &pool));
   VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &ca, &cmd));
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier barriers[2]; VkRenderingAttachmentInfo att[2];
   for (unsigned i = 0; i < 2; ++i) {
      VkImageLayout l = i ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      barriers[i] = (VkImageMemoryBarrier){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = i ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = l,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = attachments[i].image, .subresourceRange = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      att[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = attachments[i].view, .imageLayout = l, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   }
   att[1].clearValue.depthStencil.depth = 1;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      0, 0, NULL, 0, NULL, 2, barriers);
   VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att[0], .pDepthAttachment = &att[1]};
   vkCmdBeginRendering(cmd, &render); vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
   if (mesh) {
      PFN_vkCmdDrawMeshTasksEXT draw = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
      if (!draw) return 8;
      draw(cmd, 1, 1, 1);
   } else vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRendering(cmd);
   for (unsigned i = 0; i < 2; ++i) {
      barriers[i].srcAccessMask = barriers[i].dstAccessMask;
      barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barriers[i].oldLayout = barriers[i].newLayout;
      barriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   }
   vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 2, barriers);
   for (unsigned i = 0; i < 2; ++i) {
      VkBufferImageCopy copy = {.bufferOffset = 16384 * i,
         .imageSubresource = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {64, 64, 1}};
      vkCmdCopyImageToBuffer(cmd, attachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
   }
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | (mesh ? VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT : VK_PIPELINE_STAGE_VERTEX_SHADER_BIT), VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
   VkResult result = vkQueueSubmit(queue, 1, &submit, fence);
   if (result == VK_SUCCESS) result = vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull);
   printf("submission=%d dry_run=%u\n", result, dry); fflush(stdout);
   if (result != VK_SUCCESS) { if (dry) return 9; for (;;) pause(); }
   for (unsigned i = 0; i < 2; ++i) {
      FILE *f = fopen(argv[4 + i], "wb");
      if (!f || fwrite((char *)pixels + 16384 * i, 1, 16384, f) != 16384 || fclose(f)) return 10;
   }
   FILE *qfile = fopen(argv[7], "wb");
   if (!qfile || fwrite(q, 1, 16384, qfile) != 16384 || fclose(qfile)) return 10;
   CHECK(vkDeviceWaitIdle(dev));
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkUnmapMemory(dev, qmem);
   vkDestroyBuffer(dev, qbuf, NULL); vkFreeMemory(dev, qmem, NULL);
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   for (unsigned i = 0; i < 2; ++i) {
      vkDestroyShaderModule(dev, stages[i].module, NULL);
      vkDestroyImageView(dev, attachments[i].view, NULL);
      vkDestroyImage(dev, attachments[i].image, NULL);
      vkFreeMemory(dev, attachments[i].memory, NULL);
   }
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkUnmapMemory(dev, memory);
   vkDestroyBuffer(dev, buffer, NULL); vkFreeMemory(dev, memory, NULL);
   vkDestroyDevice(dev, NULL); destroy_debug(instance, messenger, NULL);
   vkDestroyInstance(instance, NULL); return 0;
}
