/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Ordinary Vulkan native/emulated/vertex/meta draws in one render pass.
 * Derived from the authored mesh-num-workgroups probe. Disjoint colored
 * stripes make partial-render attachment loss observable on a real GPU.
 * The host DRM recorder checks only the emitted packets and dependencies.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: VkResult %d\n", #call, r);                       \
         exit(2);                                                              \
      }                                                                        \
   } while (0)
#define WIDTH  64
#define HEIGHT 64
static VkDevice dev;
static VkPhysicalDevice phys;

static VKAPI_ATTR VkBool32 VKAPI_CALL
message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
   (void)severity;
   (void)type;
   (void)user;
   fprintf(stderr, "Vulkan: %s\n", data->pMessage);
   return VK_FALSE;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties m;
   vkGetPhysicalDeviceMemoryProperties(phys, &m);
   for (unsigned i = 0; i < m.memoryTypeCount; ++i)
      if ((bits & (1u << i)) &&
          (m.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   fprintf(stderr, "no matching memory type\n");
   exit(3);
}

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
};

static struct buffer
buffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buffer b;
   VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = size,
                            .usage = usage};
   CHECK(vkCreateBuffer(dev, &ci, NULL, &b.buffer));
   VkMemoryRequirements r;
   vkGetBufferMemoryRequirements(dev, b.buffer, &r);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size,
      .memoryTypeIndex = memory_type(r.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CHECK(vkAllocateMemory(dev, &ai, NULL, &b.memory));
   CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
   CHECK(vkMapMemory(dev, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map));
   return b;
}

static VkShaderModule
module(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f || fseek(f, 0, SEEK_END))
      exit(3);
   long size = ftell(f);
   rewind(f);
   if (size <= 0 || size % 4 || size > 1048576)
      exit(3);
   uint32_t *data = malloc(size);
   if (!data || fread(data, 1, size, f) != (size_t)size || fclose(f))
      exit(3);
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = data};
   VkShaderModule s;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &s));
   free(data);
   return s;
}

static void
save(const char *dir, const char *name, const void *data, size_t size)
{
   char path[4096];
   if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path))
      exit(3);
   FILE *f = fopen(path, "wb");
   if (!f || fwrite(data, 1, size, f) != size || fclose(f))
      exit(3);
}

int
main(int argc, char **argv)
{
   if (argc != 8 || !*argv[7] || strlen(argv[7]) > 8 ||
       strspn(argv[7], "NEVM") != strlen(argv[7])) {
      fprintf(
         stderr,
         "usage: transitions native.mesh.spv emulated.mesh.spv vertex.spv mesh.frag.spv vertex.frag.spv output-directory NEVM-sequence\n");
      return 3;
   }
   const char *sequence = argv[7];
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "hybrid-mesh-transitions",
                            .apiVersion = VK_API_VERSION_1_3};
   const char *debug_ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app,
                               .enabledExtensionCount = 1,
                               .ppEnabledExtensionNames = &debug_ext};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   VkDebugUtilsMessengerCreateInfoEXT debug = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
      .pfnUserCallback = message};
   PFN_vkCreateDebugUtilsMessengerEXT create_debug =
      (void *)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
   PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug =
      (void *)vkGetInstanceProcAddr(instance,
                                    "vkDestroyDebugUtilsMessengerEXT");
   if (!create_debug || !destroy_debug)
      return 3;
   VkDebugUtilsMessengerEXT messenger;
   CHECK(create_debug(instance, &debug, NULL, &messenger));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
   VkPhysicalDeviceMeshShaderPropertiesEXT mesh_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
   for (unsigned i = 0; i < count; ++i) {
      VkPhysicalDeviceDriverProperties driver = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
         .pNext = &mesh_props};
      VkPhysicalDeviceProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &driver};
      vkGetPhysicalDeviceProperties2(devices[i], &props);
      if (driver.driverID != VK_DRIVER_ID_MESA_HONEYKRISP)
         continue;
      phys = devices[i];
      printf("renderer=%s driver=%s info=%s\n", props.properties.deviceName,
             driver.driverName, driver.driverInfo);
      printf("mesh_grid_limit=%u,%u,%u total=%u\n",
             mesh_props.maxMeshWorkGroupCount[0],
             mesh_props.maxMeshWorkGroupCount[1],
             mesh_props.maxMeshWorkGroupCount[2],
             mesh_props.maxMeshWorkGroupTotalCount);
      break;
   }
   free(devices);
   if (!phys)
      return 3;
   VkPhysicalDeviceMeshShaderFeaturesEXT mf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &mf};
   VkPhysicalDeviceVulkan13Features v13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &v12};
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &v13};
   vkGetPhysicalDeviceFeatures2(phys, &features);
   printf(
      "features dynamicRendering=%u meshShader=%u vertexPipelineStoresAndAtomics=%u drawIndirectCount=%u\n",
      v13.dynamicRendering, mf.meshShader,
      features.features.vertexPipelineStoresAndAtomics, v12.drawIndirectCount);
   if (!v13.dynamicRendering || !mf.meshShader)
      return 3;
   mf = (VkPhysicalDeviceMeshShaderFeaturesEXT){.sType = mf.sType,
                                                .meshShader = VK_TRUE};
   v12 = (VkPhysicalDeviceVulkan12Features){.sType = v12.sType,
                                            .pNext = &mf,
                                            .drawIndirectCount = VK_FALSE};
   v13 = (VkPhysicalDeviceVulkan13Features){.sType = v13.sType,
                                            .pNext = &v12,
                                            .dynamicRendering = VK_TRUE};
   features = (VkPhysicalDeviceFeatures2){.sType = features.sType,
                                          .pNext = &v13,
                                          .features = {0}};
   uint32_t n = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
   VkQueueFamilyProperties *families = calloc(n, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families);
   uint32_t qfam = 0;
   while (qfam < n && !(families[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++qfam;
   free(families);
   if (qfam == n)
      return 3;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qfam,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   const char *mesh_ext = VK_EXT_MESH_SHADER_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &features,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .enabledExtensionCount = 1,
                             .ppEnabledExtensionNames = &mesh_ext};
   CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkShaderModule modules[5];
   for (unsigned i = 0; i < 5; ++i)
      modules[i] = module(argv[1 + i]);
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
       .module = modules[0],
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = modules[3],
       .pName = "main"},
   };
   VkViewport viewport = {0, 0, WIDTH, HEIGHT, 0, 1};
   VkRect2D scissor = {{0, 0}, {WIDTH, HEIGHT}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ba};
   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &format};
   VkGraphicsPipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .layout = layout};
   VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
      .pDynamicStates = dynamic};
   pci.pDynamicState = &ds;
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipeline pipelines[3];
   for (unsigned i = 0; i < 3; ++i) {
      stages[0].module = modules[i];
      if (i == 2) {
         stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
         stages[1].module = modules[4];
         pci.pVertexInputState = &vi;
         pci.pInputAssemblyState = &ia;
      }
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL,
                                      &pipelines[i]));
   }
   puts("PIPELINE_OK");
   fflush(stdout);
   VkImageCreateInfo imci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                             .imageType = VK_IMAGE_TYPE_2D,
                             .format = format,
                             .extent = {WIDTH, HEIGHT, 1},
                             .mipLevels = 1,
                             .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT,
                             .tiling = VK_IMAGE_TILING_OPTIMAL,
                             .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image;
   CHECK(vkCreateImage(dev, &imci, NULL, &image));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, image, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex =
         memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory image_memory;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &image_memory));
   CHECK(vkBindImageMemory(dev, image, image_memory, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   struct buffer pixels =
      buffer(WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(pixels.map, 0xa5, WIDTH * HEIGHT * 4);
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qfam};
   VkCommandPool commands;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &commands));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commands,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                        NULL, 0, NULL, 1, &barrier);
   VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
                             .layerCount = 1,
                             .colorAttachmentCount = 1,
                             .pColorAttachments = &att};
   vkCmdBeginRendering(cmd, &render);
   PFN_vkCmdDrawMeshTasksEXT draw_mesh =
      (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   if (!draw_mesh)
      return 3;
   for (unsigned i = 0; sequence[i]; ++i) {
      VkRect2D stripe = {{i * 8, 0}, {8, HEIGHT}};
      vkCmdSetScissor(cmd, 0, 1, &stripe);
      if (sequence[i] == 'M') {
         VkClearAttachment clear = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .colorAttachment = 0,
                                    .clearValue.color.float32 = {1, 0, 0, 1}};
         VkClearRect rect = {.rect = stripe, .layerCount = 1};
         vkCmdClearAttachments(cmd, 1, &clear, 1, &rect);
      } else {
         unsigned pipeline = sequence[i] == 'N'   ? 0
                             : sequence[i] == 'E' ? 1
                                                  : 2;
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipelines[pipeline]);
         if (pipeline == 2)
            vkCmdDraw(cmd, 3, 1, 0, 0);
         else
            draw_mesh(cmd, 1, 1, 1);
      }
   }
   vkCmdEndRendering(cmd);
   barrier.srcAccessMask = barrier.dstAccessMask;
   barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   barrier.oldLayout = barrier.newLayout;
   barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &barrier);
   VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1}};
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          pixels.buffer, 1, &copy);
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   puts("COMMAND_OK");
   fflush(stdout);
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
   puts("SUBMIT_OK");
   fflush(stdout);
   save(argv[6], "pixels.bin", pixels.map, WIDTH * HEIGHT * 4);
   uint8_t expected_pixels[WIDTH * HEIGHT * 4] = {0};
   unsigned mismatches = 0;
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         unsigned stripe = x / 8;
         uint8_t *expected = expected_pixels + 4 * (y * WIDTH + x);
         if (stripe < strlen(sequence)) {
            char kind = sequence[stripe];
            expected[0] = (kind == 'V' || kind == 'M') ? 255 : 0;
            expected[1] = (kind == 'V' || kind == 'E') ? 255 : 0;
            expected[2] = (kind == 'V' || kind == 'N') ? 255 : 0;
            expected[3] = 255;
         }
         mismatches += memcmp((uint8_t *)pixels.map + 4 * (y * WIDTH + x),
                              expected, 4) != 0;
      }
   }
   save(argv[6], "expected-pixels.bin", expected_pixels,
        sizeof(expected_pixels));
   printf("sequence=%s pixels_mismatched=%u/4096\n", sequence, mismatches);
   CHECK(vkDeviceWaitIdle(dev));
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, commands, NULL);
   for (unsigned i = 0; i < 3; ++i)
      vkDestroyPipeline(dev, pipelines[i], NULL);
   for (unsigned i = 0; i < 5; ++i)
      vkDestroyShaderModule(dev, modules[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   struct buffer buffers[] = {pixels};
   for (unsigned i = 0; i < 1; ++i) {
      vkUnmapMemory(dev, buffers[i].memory);
      vkDestroyBuffer(dev, buffers[i].buffer, NULL);
      vkFreeMemory(dev, buffers[i].memory, NULL);
   }
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, image_memory, NULL);
   vkDestroyDevice(dev, NULL);
   destroy_debug(instance, messenger, NULL);
   vkDestroyInstance(instance, NULL);
   puts(mismatches ? "FAIL" : "PASS");
   return mismatches ? 1 : 0;
}
