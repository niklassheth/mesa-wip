/* SPDX-License-Identifier: MIT. Owned-buffer diagnostic Vulkan harness.
 * Research carrier; candidate body is selected by the private driver adapter. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdbool.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define EFFECT_Q_BYTES 0x80000u

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); fatal("Vulkan call failed"); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pdev;
static VkQueue queue;
static uint32_t qfam;
static int mesh_mode = 1;
static unsigned reference_topology = 6;
static const char *shader_path;
#include "oracle.h"
static VkShaderModule module(const uint32_t *code, unsigned len)
{
   VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, len, (const uint32_t *)code};
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   exit(3);
}

static uint32_t find_q_memory(uint32_t bits)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags = mp.memoryTypes[i].propertyFlags;
      if ((bits & (1u << i)) && (flags & need) == need &&
          !(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
         fprintf(stderr, "Q_MEMORY type=%u properties=%u cached=0 coherent=1\n", i, flags);
         return i;
      }
   }
   fprintf(stderr, "No non-cached coherent Q memory type\n");
   exit(16);
}

static VkPipeline make_pipeline(VkPipelineLayout layout)
{
   VkPipelineShaderStageCreateInfo st[4];
   int ns = 0;
   FILE *sf = fopen(shader_path, "rb");
   if (!sf) exit(10);
   fseek(sf, 0, SEEK_END); long size = ftell(sf); rewind(sf);
   if (size <= 0 || size > 1048576 || size % 4) exit(10);
   uint32_t *words = malloc(size);
   if (!words || fread(words, 1, size, sf) != size) exit(10);
   fclose(sf);
   /* The private binary loader is outside the shader cache key. Give every
    * research pipeline a distinct SPIR-V module identity without changing
    * executable SPIR-V: the generator header word is informational. The
    * bridge still verifies the complete uploaded native program per arm. */
   static uint32_t module_identity;
   words[2] = 0x76520000u | ++module_identity;
   st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      NULL, 0, mesh_mode ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT,
      module(words, size), "main", NULL};
   free(words);
   sf = fopen(getenv("MULTI_FRAGMENT"), "rb");
   if (!sf) exit(10);
   fseek(sf, 0, SEEK_END); size = ftell(sf); rewind(sf);
   if (size <= 0 || size > 1048576 || size % 4) exit(10);
   words = malloc(size);
   if (!words || fread(words, 1, size, sf) != size || fclose(sf)) exit(10);
   st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, module(words, size), "main", NULL};
   free(words);
   VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
      reference_topology==6 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : reference_topology==1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_FALSE};
   /* Match Metal top-left pixel coordinates for the same clip positions. */
   VkViewport vp = {0, 64, 64, -64, 0, 1};
   VkRect2D sc = {{0, 0}, {64, 64}};
   VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vp, 1, &sc};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState ba = {0};
   ba.colorWriteMask = 0xf;
   if (getenv("MULTI_BLEND") && atoi(getenv("MULTI_BLEND"))) {
      ba.blendEnable = VK_TRUE;
      ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.colorBlendOp = VK_BLEND_OP_ADD;
      ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.alphaBlendOp = VK_BLEND_OP_ADD;
   }
   fprintf(stderr,"COUNT_BLEND enabled=%u src=ONE dst=ONE R32_SFLOAT clear=0 depth=off\n", ba.blendEnable);
   VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   cb.attachmentCount = 1; cb.pAttachments = &ba;
   VkFormat fmt = VK_FORMAT_R32_SFLOAT;
   VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &fmt};
   VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, ns, st,
      &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cb, NULL, layout};
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
   for (int i = 0; i < ns; ++i) vkDestroyShaderModule(dev, st[i].module, NULL);
   return p;
}


struct probe_features {
   VkPhysicalDeviceFeatures2 core;
   VkPhysicalDeviceVulkan13Features v13;
   VkPhysicalDeviceVulkan12Features v12;
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
};

static void feature_request(struct probe_features *p, int mesh)
{
   memset(p, 0, sizeof(*p));
   p->core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
   p->core.pNext = &p->v13;
   p->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
   p->v13.dynamicRendering = VK_TRUE;
   p->mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
   if (mesh) { p->v13.pNext = &p->mesh; p->mesh.meshShader = VK_TRUE; }
   if (true) {
      p->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
      p->v12.bufferDeviceAddress = VK_TRUE;
      p->v12.pNext = p->v13.pNext;
      p->v13.pNext = &p->v12;

   }
}


static char ready_device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
static char ready_driver_name[VK_MAX_DRIVER_NAME_SIZE];
static char ready_driver_info[VK_MAX_DRIVER_INFO_SIZE];
static uint32_t ready_driver_version;
static void device_init(void) {
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "native-mesh-v0", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   if (!n) fatal("device or queue missing");
   VkPhysicalDevice *devices = calloc(n, sizeof(*devices));
   if (!devices) fatal("device or queue missing");
   CHECK(vkEnumeratePhysicalDevices(inst, &n, devices));
   for (unsigned i = 0; i < n; ++i) {
      VkPhysicalDeviceDriverProperties driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
      vkGetPhysicalDeviceProperties2(devices[i], &props);
      if (driver.driverID != VK_DRIVER_ID_MESA_HONEYKRISP) continue;
      const char *expected = getenv("MESHPROBE_DRIVER_COMMIT");
      if (expected && !strstr(driver.driverInfo, expected)) {
         fprintf(stderr, "wrong driver build: %s expected %s\n", driver.driverInfo, expected);
         fatal("driver version");
      }
      pdev = devices[i];
      if(strcmp(props.properties.deviceName,"Apple M4 (G16G B1)"))fatal("renderer identity");
      memcpy(ready_device_name,props.properties.deviceName,sizeof ready_device_name);
      memcpy(ready_driver_name,driver.driverName,sizeof ready_driver_name);
      memcpy(ready_driver_info,driver.driverInfo,sizeof ready_driver_info);
      ready_driver_version=props.properties.driverVersion;
      fprintf(stderr,"renderer=%s driver=%s info=%s version=%u\n", props.properties.deviceName, driver.driverName, driver.driverInfo, props.properties.driverVersion);
      break;
   }
   free(devices);
   if (!pdev) fatal("device or queue missing");
   VkFormatProperties format;
   vkGetPhysicalDeviceFormatProperties(pdev,VK_FORMAT_R32_SFLOAT,&format);
   const VkFormatFeatureFlags required=VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if((format.optimalTilingFeatures&required)!=required)fatal("R32_SFLOAT additive blend unsupported");
   uint32_t families = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &families, NULL);
   VkQueueFamilyProperties *fp = calloc(families, sizeof(*fp));
   if (!fp) fatal("device or queue missing");
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &families, fp);
   for (qfam = 0; qfam < families; ++qfam)
      if (fp[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT) break;
   free(fp);
   if (qfam == families) fatal("device or queue missing");
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfam, 1, &prio};
   struct probe_features requested; feature_request(&requested, mesh_mode);
   struct probe_features available; feature_request(&available, mesh_mode);
   vkGetPhysicalDeviceFeatures2(pdev, &available.core);
   fprintf(stderr,"FEATURE_PREFLIGHT dynamicRendering=%u meshShader=%u vertexStoresAvailable=%u vertexStoresRequested=%u\n",
      available.v13.dynamicRendering, available.mesh.meshShader,
      available.core.features.vertexPipelineStoresAndAtomics,
      requested.core.features.vertexPipelineStoresAndAtomics);
   if (!available.v12.bufferDeviceAddress) fatal("buffer address");
   if (!available.v13.dynamicRendering || (mesh_mode && !available.mesh.meshShader) ||
       requested.core.features.vertexPipelineStoresAndAtomics) fatal("features");
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &requested.core, 0, 1, &qci};
   const char *extension = VK_EXT_MESH_SHADER_EXTENSION_NAME;
   if (mesh_mode) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &extension; }
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
}

static void lane_init(struct lane *l) {
   VkImageCreateInfo imci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT,
      {64, 64, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
      find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, img, mem, 0));
   VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D,
      VK_FORMAT_R32_SFLOAT, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));
   VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 64 * 64 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements br;
   vkGetBufferMemoryRequirements(dev, buf, &br);
   VkMemoryAllocateInfo bai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, br.size,
      find_memory(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
   VkBuffer sb;
   VkBufferCreateInfo sci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
   sci.size = EFFECT_Q_BYTES; sci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
   CHECK(vkCreateBuffer(dev, &sci, NULL, &sb));
   VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, sb, &sr);
   VkMemoryAllocateFlagsInfo saflags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   VkMemoryAllocateInfo sma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &saflags};
   sma.allocationSize = sr.size;
   sma.memoryTypeIndex = find_q_memory(sr.memoryTypeBits);
   VkDeviceMemory smem; CHECK(vkAllocateMemory(dev, &sma, NULL, &smem));
   CHECK(vkBindBufferMemory(dev, sb, smem, 0));
   uint32_t *snapshot; CHECK(vkMapMemory(dev, smem, 0, VK_WHOLE_SIZE, 0, (void **)&snapshot));
   VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo slci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
   slci.bindingCount = 1; slci.pBindings = &binding;
   VkDescriptorSetLayout sl; CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &sl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
   dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
   VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
   dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
   VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi = {sb, 0, EFFECT_Q_BYTES};
   VkWriteDescriptorSet dw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
   dw.dstSet = ds; dw.descriptorCount = 1; dw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dw.pBufferInfo = &dbi;
   vkUpdateDescriptorSets(dev, 1, &dw, 0, NULL);
   uint32_t *px; CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
   VkBufferDeviceAddressInfo qa = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, sb};
   l->qva = vkGetBufferDeviceAddress(dev, &qa);
   l->view=view; l->image=img; l->readback=buf; l->q=(void*)snapshot; l->pixels=(void*)px; l->ds=ds;
   l->want=malloc(Q_BYTES); if(!l->want||!l->qva||l->qva+Q_BYTES>=(1ull<<48))fatal("lane allocation");
   VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   CHECK(vkCreateFence(dev,&fci,NULL,&l->fence));
}

static VkPipelineLayout layout;
static void layout_init(void) {
   VkDescriptorSetLayoutBinding binding = {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo slci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
   slci.bindingCount=1;slci.pBindings=&binding;VkDescriptorSetLayout sl;
   CHECK(vkCreateDescriptorSetLayout(dev,&slci,NULL,&sl));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_ALL, 0, 8};
   VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
   plci.setLayoutCount = 1; plci.pSetLayouts = &sl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
}

static VkCommandBuffer record(struct arm *a, struct lane *l) {
   VkImage img=l->image;VkImageView view=l->view;VkBuffer buf=l->readback;
   VkDescriptorSet ds=l->ds;VkPipeline pipe=a->pipe;
   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qfam};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0};
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                        NULL, 0, NULL, 1, &b);
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, view,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkRenderingInfo rinfo = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {64, 64}}, 1, 0, 1, &att};
   vkCmdBeginRendering(cmd, &rinfo);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &ds, 0, NULL);
   struct { uint32_t n; float level; } pc = {1, 1.0f};
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
   if(mesh_mode) {
      PFN_vkCmdDrawMeshTasksEXT drawMesh = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
      if (!drawMesh) fatal("mesh entry point");
      drawMesh(cmd, a->grid[0], a->grid[1], a->grid[2]);
   } else {
      unsigned k=reference_topology==6?3:reference_topology==1?2:1;
      vkCmdDraw(cmd,a->shape.primitives*k,1,0,0);
   }
   vkCmdEndRendering(cmd);
   VkMemoryBarrier output_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &output_barrier, 0, NULL, 0, NULL);
   b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                        NULL, 0, NULL, 1, &b);
   VkBufferImageCopy reg = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {64, 64, 1}};
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
   VkMemoryBarrier host_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   return cmd;
}
