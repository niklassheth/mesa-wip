/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Independent per-draw native blocks in one render: two different native mesh
 * pipelines (A: V3/P1, push constants, one group; B: V4/P2, uniform buffer,
 * two groups, drawn with two descriptor sets as B and C) and one ordinary
 * vertex draw V, recorded in one dynamic-rendering pass in the order given by
 * each sequence argument. Every draw adds its own power-of-two weight to a
 * cleared R32_SFLOAT target with ONE,ONE blending, so each pixel's value
 * names exactly which draws covered it. The probe checks all 4096 pixels
 * against its oracle; run.py checks them again independently.
 *
 * usage: blocks SHADER_DIR OUTPUT_DIR REPEATS SEQUENCE...
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         printf("VKFAIL %s: %d\n", #call, r_);                                 \
         exit(2);                                                              \
      }                                                                        \
   } while (0)
#define W 64
#define H 64

static VkDevice dev;
static VkPhysicalDevice phys;

/* Geometry in pixels; pixel centers never lie on an outer edge. */
static const float A_TRI[6] = {4, 4, 36.25f, 4, 4, 36.25f};
static const float A_WEIGHT = 1;
static const float V_RECT[4] = {2, 24, 62, 40};
static const float V_WEIGHT = 16;
static const float B_RECT[4] = {28, 4, 60, 36};
static const float B_WEIGHT = 256;
static const float C_RECT[4] = {8, 30, 40, 60};
static const float C_WEIGHT = 4096;
/* D: pipeline B over the whole target in 256 slices (512 triangles per draw),
 * to load the render's TVB lease; every pixel is covered exactly once. */
static const float D_RECT[4] = {0, 0, 64, 64};
static const float D_WEIGHT = 1;
#define D_SLICES 256

static float
oracle(int x, int y, const char *seq)
{
   float px = x + 0.5f, py = y + 0.5f, sum = 0;
   for (const char *c = seq; *c; ++c) {
      switch (*c) {
      case 'A':
         if (px > 4 && py > 4 && px + py < 40.25f)
            sum += A_WEIGHT;
         break;
      case 'V':
         if (px > V_RECT[0] && px < V_RECT[2] && py > V_RECT[1] && py < V_RECT[3])
            sum += V_WEIGHT;
         break;
      case 'B':
         if (px > B_RECT[0] && px < B_RECT[2] && py > B_RECT[1] && py < B_RECT[3])
            sum += B_WEIGHT;
         break;
      case 'C':
         if (px > C_RECT[0] && px < C_RECT[2] && py > C_RECT[1] && py < C_RECT[3])
            sum += C_WEIGHT;
         break;
      case 'D':
         sum += D_WEIGHT;
         break;
      }
   }
   return sum;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties m;
   vkGetPhysicalDeviceMemoryProperties(phys, &m);
   for (unsigned i = 0; i < m.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   printf("VKFAIL no memory type\n");
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
                            .size = size, .usage = usage};
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
module(const char *dir, const char *name)
{
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s", dir, name);
   FILE *f = fopen(path, "rb");
   if (!f || fseek(f, 0, SEEK_END)) {
      printf("VKFAIL missing %s\n", path);
      exit(3);
   }
   long size = ftell(f);
   rewind(f);
   if (size <= 0 || size % 4 || size > 1048576)
      exit(3);
   uint32_t *data = malloc(size);
   if (!data || fread(data, 1, size, f) != (size_t)size || fclose(f))
      exit(3);
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = data};
   VkShaderModule s;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &s));
   free(data);
   return s;
}

static VkPipeline
pipeline(VkPipelineLayout layout, VkShaderStageFlagBits first, VkShaderModule a,
         VkShaderModule fs)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = first, .module = a, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, W, H, 0, 1};
   VkRect2D scissor = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState blend = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend};
   VkFormat format = VK_FORMAT_R32_SFLOAT;
   VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
   bool mesh = first == VK_SHADER_STAGE_MESH_BIT_EXT;
   VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering, .stageCount = 2, .pStages = stages,
      .pVertexInputState = mesh ? NULL : &vi,
      .pInputAssemblyState = mesh ? NULL : &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = layout};
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
   return p;
}

static uint64_t
now_ns(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

int
main(int argc, char **argv)
{
   if (argc < 5) {
      fprintf(stderr, "usage: blocks SHADER_DIR OUTPUT_DIR REPEATS SEQUENCE...\n");
      return 3;
   }
   const char *shaders = argv[1], *out = argv[2];
   int repeats = atoi(argv[3]);
   if (repeats < 1 || repeats > 64)
      return 3;
   for (int i = 4; i < argc; ++i) {
      if (!*argv[i] || strlen(argv[i]) > 96 ||
          strspn(argv[i], "ABCDV") != strlen(argv[i]))
         return 3;
   }
   setvbuf(stdout, NULL, _IOLBF, 0);

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "native-mesh-blocks",
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkResult er = vkEnumeratePhysicalDevices(instance, &count, &phys);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !count) {
      printf("VKFAIL no physical device\n");
      return 2;
   }
   VkPhysicalDeviceDriverProperties driver = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
   vkGetPhysicalDeviceProperties2(phys, &props);
   printf("DEVICE %s | %s | %s\n", props.properties.deviceName, driver.driverName,
          driver.driverInfo);

   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .meshShader = VK_TRUE};
   VkPhysicalDeviceVulkan13Features v13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &mesh_features, .dynamicRendering = VK_TRUE,
      .synchronization2 = VK_TRUE};
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = 0, .queueCount = 1,
                                  .pQueuePriorities = &priority};
   const char *ext = VK_EXT_MESH_SHADER_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &v13, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .enabledExtensionCount = 1,
                             .ppEnabledExtensionNames = &ext};
   CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   PFN_vkCmdDrawMeshTasksEXT draw_mesh =
      (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   if (!draw_mesh)
      return 2;

   /* Target image and readback. */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32_SFLOAT, .extent = {W, H, 1}, .mipLevels = 1,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image;
   CHECK(vkCreateImage(dev, &imci, NULL, &image));
   VkMemoryRequirements ir;
   vkGetImageMemoryRequirements(dev, image, &ir);
   VkMemoryAllocateInfo iai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = ir.size,
      .memoryTypeIndex = memory_type(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory image_memory;
   CHECK(vkAllocateMemory(dev, &iai, NULL, &image_memory));
   CHECK(vkBindImageMemory(dev, image, image_memory, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_SFLOAT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   struct buffer readback = buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   /* Two immutable uniform buffers for pipeline B: draws B and C. */
   struct buffer ubo[3] = {buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
                           buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
                           buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)};
   float bdata[3][6] = {
      {B_RECT[0], B_RECT[1], B_RECT[2], B_RECT[3], B_WEIGHT, 2},
      {C_RECT[0], C_RECT[1], C_RECT[2], C_RECT[3], C_WEIGHT, 2},
      {D_RECT[0], D_RECT[1], D_RECT[2], D_RECT[3], D_WEIGHT, D_SLICES},
   };
   for (int i = 0; i < 3; ++i)
      memcpy(ubo[i].map, bdata[i], sizeof(bdata[i]));

   VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT};
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 3,
      .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
   VkDescriptorSetLayout dsls[3] = {dsl, dsl, dsl};
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 3, .pSetLayouts = dsls};
   VkDescriptorSet sets[3];
   CHECK(vkAllocateDescriptorSets(dev, &dsai, sets));
   for (int i = 0; i < 3; ++i) {
      VkDescriptorBufferInfo bi = {ubo[i].buffer, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = sets[i], .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .pBufferInfo = &bi};
      vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
   }

   VkPushConstantRange a_range = {
      VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
   VkPushConstantRange v_range = {
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
   VkPipelineLayoutCreateInfo plci[3] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
       .pushConstantRangeCount = 1, .pPushConstantRanges = &a_range},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
       .setLayoutCount = 1, .pSetLayouts = &dsl},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
       .pushConstantRangeCount = 1, .pPushConstantRanges = &v_range},
   };
   VkPipelineLayout layouts[3];
   for (int i = 0; i < 3; ++i)
      CHECK(vkCreatePipelineLayout(dev, &plci[i], NULL, &layouts[i]));
   VkPipeline pa = pipeline(layouts[0], VK_SHADER_STAGE_MESH_BIT_EXT,
                            module(shaders, "a.mesh.spv"), module(shaders, "a.frag.spv"));
   VkPipeline pb = pipeline(layouts[1], VK_SHADER_STAGE_MESH_BIT_EXT,
                            module(shaders, "b.mesh.spv"), module(shaders, "b.frag.spv"));
   VkPipeline pv = pipeline(layouts[2], VK_SHADER_STAGE_VERTEX_BIT,
                            module(shaders, "v.vert.spv"), module(shaders, "v.frag.spv"));

   VkQueryPoolCreateInfo qpci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                 .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
   VkQueryPool queries;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &queries));

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   float a_pc[8] = {A_TRI[0], A_TRI[1], A_TRI[2], A_TRI[3], A_TRI[4], A_TRI[5], A_WEIGHT, 0};
   float v_pc[8] = {V_RECT[0], V_RECT[1], V_RECT[2], V_RECT[3], V_WEIGHT, 0, 0, 0};
   unsigned failures = 0;

   for (int s = 4; s < argc; ++s) {
      const char *seq = argv[s];
      for (int rep = 0; rep < repeats; ++rep) {
         CHECK(vkResetCommandBuffer(cmd, 0));
         VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
         uint64_t t0 = now_ns();
         CHECK(vkBeginCommandBuffer(cmd, &bi));
         vkCmdResetQueryPool(cmd, queries, 0, 2);
         VkImageMemoryBarrier2 to_attachment = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
         VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                 .imageMemoryBarrierCount = 1,
                                 .pImageMemoryBarriers = &to_attachment};
         vkCmdPipelineBarrier2(cmd, &dep);
         vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, 0);
         VkRenderingAttachmentInfo color = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
         VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                               .renderArea = {{0, 0}, {W, H}}, .layerCount = 1,
                               .colorAttachmentCount = 1, .pColorAttachments = &color};
         vkCmdBeginRendering(cmd, &ri);
         for (const char *c = seq; *c; ++c) {
            switch (*c) {
            case 'A':
               vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pa);
               vkCmdPushConstants(cmd, layouts[0], a_range.stageFlags, 0, 32, a_pc);
               draw_mesh(cmd, 1, 1, 1);
               break;
            case 'B':
            case 'C':
            case 'D':
               vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pb);
               vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layouts[1],
                                       0, 1, &sets[*c - 'B'], 0, NULL);
               draw_mesh(cmd, *c == 'D' ? D_SLICES : 2, 1, 1);
               break;
            case 'V':
               vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pv);
               vkCmdPushConstants(cmd, layouts[2], v_range.stageFlags, 0, 32, v_pc);
               vkCmdDraw(cmd, 6, 1, 0, 0);
               break;
            }
         }
         vkCmdEndRendering(cmd);
         vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queries, 1);
         VkImageMemoryBarrier2 to_transfer = to_attachment;
         to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
         to_transfer.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
         to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
         to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
         to_transfer.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
         to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
         dep.pImageMemoryBarriers = &to_transfer;
         vkCmdPipelineBarrier2(cmd, &dep);
         VkBufferImageCopy region = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                     .imageExtent = {W, H, 1}};
         vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                readback.buffer, 1, &region);
         VkMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                  .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                  .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                                  .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
         VkDependencyInfo hdep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                  .memoryBarrierCount = 1, .pMemoryBarriers = &host};
         vkCmdPipelineBarrier2(cmd, &hdep);
         CHECK(vkEndCommandBuffer(cmd));
         uint64_t t1 = now_ns();
         memset(readback.map, 0xff, W * H * 4);
         VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1, .pCommandBuffers = &cmd};
         printf("SUBMIT_BEGIN %s %d\n", seq, rep);
         CHECK(vkQueueSubmit(queue, 1, &si, fence));
         VkResult wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull);
         if (wr != VK_SUCCESS) {
            printf("IN_FLIGHT_WAIT_FAILED %d\n", wr);
            return 2;
         }
         CHECK(vkResetFences(dev, 1, &fence));
         uint64_t t2 = now_ns();
         printf("SUBMIT_OK\n");
         uint64_t ts[4] = {0};
         VkResult qr = vkGetQueryPoolResults(dev, queries, 0, 2, sizeof(ts), ts,
                                             2 * sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT |
                                                VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
         const float *px = readback.map;
         unsigned bad = 0;
         for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
               float e = oracle(x, y, seq), o = px[y * W + x];
               if (memcmp(&e, &o, 4)) {
                  if (bad < 8)
                     printf("MISMATCH %s %d x=%d y=%d expected=%g observed=%g\n",
                            seq, rep, x, y, e, o);
                  ++bad;
               }
            }
         }
         failures += !!bad;
         char name[4096];
         if (snprintf(name, sizeof(name), "%s/%s-r%02d-pixels.bin", out, seq, rep) >=
             (int)sizeof(name))
            return 3;
         FILE *f = fopen(name, "wb");
         if (!f || fwrite(readback.map, 1, W * H * 4, f) != W * H * 4 || fclose(f))
            return 3;
         double gpu_ns = qr == VK_SUCCESS && ts[1] && ts[3] && ts[2] > ts[0]
                            ? (double)(ts[2] - ts[0]) * props.properties.limits.timestampPeriod
                            : -1;
         printf("SAMPLE {\"sequence\":\"%s\",\"rep\":%d,\"mismatches\":%u,"
                "\"gpu_ns\":%.1f,\"record_ns\":%" PRIu64 ",\"submit_wait_ns\":%" PRIu64 "}\n",
                seq, rep, bad, gpu_ns, t1 - t0, t2 - t1);
      }
   }
   printf(failures ? "FAIL\n" : "PASS\n");
   return failures ? 1 : 0;
}
