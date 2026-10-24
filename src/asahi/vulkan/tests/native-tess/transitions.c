/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Ordinary Vulkan TES native/emulated/vertex/meta draws in one render pass.
 * Derived from the authored mesh-num-workgroups probe. Disjoint colored
 * stripes make partial-render attachment loss observable on a real GPU.
 * The host DRM recorder checks only the emitted packets and dependencies.
 */
#include <errno.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
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
static uint64_t monotonic_ns(void)
{
   struct timespec t;
   if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
   return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static const char *option(const char *name, const char *fallback)
{
   const char *value = getenv(name);
   return value ? value : fallback;
}
#ifdef HOST_MERGE_OBSERVER
/* Only the nonexecuting host recorder links this read-only observer. */
void observe_merge(VkCommandBuffer cmd, unsigned phase);
#endif

/* A device test must retain buffers until owner recovery if submission or
 * retirement becomes uncertain. Host recording returns successful fences. */
static void
in_flight(VkResult result, const char *phase)
{
   if (result == VK_SUCCESS)
      return;
   fprintf(stderr, "IN_FLIGHT_WAIT_FAILED phase=%s result=%d; retaining backing\n",
           phase, result);
   fflush(NULL);
   raise(SIGSTOP);
   for (;;)
      pause();
}

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
   if ((argc < 9 || argc > 12) || !*argv[8] || strlen(argv[8]) > 8 ||
       strspn(argv[8], "NEVM") != strlen(argv[8])) {
      fprintf(
         stderr,
         "usage: transitions native.tese.spv fallback.tese.spv vertex.spv tess.frag.spv vertex.frag.spv control.tesc.spv output-directory NEVM-sequence [plain|timestamp|merged|kept-merge|deferred|indexed|instances|offset|indirect] [inner=1] [outer=1]\n");
      return 3;
   }
   const char *sequence = argv[8];
   const char *mode = argc >= 10 ? argv[9] : "plain";
   float inner = argc >= 11 ? strtof(argv[10], NULL) : 1;
   float outer = argc >= 12 ? strtof(argv[11], NULL) : 1;
   unsigned patch_size = atoi(option("TESS_TEST_PATCH_SIZE", "3"));
   bool quad = !strcmp(option("TESS_TEST_DOMAIN", "triangles"), "quads");
   bool lower_left = atoi(option("TESS_TEST_LOWER_LEFT", "0"));
   if (patch_size < 1 || patch_size > 32) return 3;
   const char *cull = option("TESS_TEST_CULL", "none");
   const char *front = option("TESS_TEST_FRONT", "ccw");
   bool depth = atoi(option("TESS_TEST_DEPTH", "0"));
   bool negative_one = atoi(option("TESS_TEST_NEGATIVE_ONE", "0"));
   bool depth_clamp = atoi(option("TESS_TEST_DEPTH_CLAMP", "0"));
   bool depth_clip = atoi(option("TESS_TEST_DEPTH_CLIP", "1"));
   bool rewrite = atoi(option("TESS_TEST_VS_REWRITE", "0"));
   float z = strtof(option("TESS_TEST_Z", "0.5"), NULL);
   VkCullModeFlags cull_mode = !strcmp(cull, "none") ? VK_CULL_MODE_NONE :
      !strcmp(cull, "front") ? VK_CULL_MODE_FRONT_BIT :
      !strcmp(cull, "back") ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_FRONT_AND_BACK;
   bool front_ccw = !strcmp(front, "ccw");
   if (cull_mode == VK_CULL_MODE_FRONT_AND_BACK ||
       (strcmp(front, "ccw") && strcmp(front, "cw"))) return 3;
   bool indexed = !strcmp(mode, "indexed");
   bool instances = !strcmp(mode, "instances");
   bool offset = !strcmp(mode, "offset");
   bool indirect = !strcmp(mode, "indirect");
   bool timestamp = !strcmp(mode, "timestamp");
   bool conditional = !strcmp(mode, "conditional");
   bool conservative = !strcmp(mode, "conservative");
   bool msaa = !strcmp(mode, "msaa");
   bool multiview = !strcmp(mode, "multiview");
   bool layered = !strcmp(mode, "layers");
   unsigned layers = multiview || layered ? 2 : 1;
   bool occlusion = !strcmp(mode, "occlusion");
   bool statistics = !strcmp(mode, "statistics");
   bool discard = !strcmp(mode, "discard");
   bool nonfill = !strcmp(mode, "nonfill");
   bool restart = !strcmp(mode, "restart");
   bool first_instance = !strcmp(mode, "firstinstance");
   bool multiple = !strcmp(mode, "multipatch");
   bool kept_merge = !strcmp(mode, "kept-merge");
   bool merged = !strcmp(mode, "merged") || kept_merge;
   bool deferred = !strcmp(mode, "deferred");
   if ((!conditional && !conservative && !msaa && !multiview && !layered && !occlusion && !statistics && !discard && !nonfill && !restart && !first_instance && !multiple && !timestamp && !merged && !deferred && !indexed && !instances && !offset && !indirect && strcmp(mode, "plain")) ||
       ((timestamp || occlusion || statistics) && strcmp(sequence, "N")) ||
       (merged && strcmp(sequence, kept_merge ? "VN" : "NN")))
      return 3;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "native-tess-transitions",
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
   float timestamp_period = 0;
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
      if (driver.driverID != VK_DRIVER_ID_MESA_HONEYKRISP &&
          !(atoi(option("TESS_TEST_SOFTWARE_REFERENCE", "0")) &&
            driver.driverID == VK_DRIVER_ID_MESA_LLVMPIPE))
         continue;
      phys = devices[i];
      timestamp_period = props.properties.limits.timestampPeriod;
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
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
   VkPhysicalDeviceDepthClipEnableFeaturesEXT clip_enable = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
      .pNext = deferred ? &dynamic3 : NULL};
   VkPhysicalDeviceDepthClipControlFeaturesEXT clip_control = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
      .pNext = &clip_enable};
   VkPhysicalDeviceConditionalRenderingFeaturesEXT cond_feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
      .pNext = depth ? &clip_control : deferred ? (void *)&dynamic3 : NULL};
   VkPhysicalDeviceMeshShaderFeaturesEXT mf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .pNext = conditional ? &cond_feature : depth ? (void *)&clip_control : deferred ? (void *)&dynamic3 : NULL};
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &mf};
   VkPhysicalDeviceVulkan11Features v11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &v12};
   VkPhysicalDeviceVulkan13Features v13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &v11};
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &v13};
   vkGetPhysicalDeviceFeatures2(phys, &features);
   printf(
      "features dynamicRendering=%u meshShader=%u vertexPipelineStoresAndAtomics=%u drawIndirectCount=%u\n",
      v13.dynamicRendering, mf.meshShader,
      features.features.vertexPipelineStoresAndAtomics, v12.drawIndirectCount);
   if (!v13.dynamicRendering || !features.features.tessellationShader ||
       (conditional && !cond_feature.conditionalRendering) ||
       (multiview && (!v11.multiview || !v11.multiviewTessellationShader)) ||
       (statistics && !features.features.pipelineStatisticsQuery) ||
       (nonfill && !features.features.fillModeNonSolid) ||
       (depth && (!clip_control.depthClipControl || !clip_enable.depthClipEnable ||
                  (depth_clamp && !features.features.depthClamp))) ||
       (timestamp && !v13.synchronization2) ||
       (deferred && !dynamic3.extendedDynamicState3ColorBlendEnable)) {
      puts("SKIP required device feature not advertised");
      return 77;
   }
   dynamic3 = (VkPhysicalDeviceExtendedDynamicState3FeaturesEXT){
      .sType = dynamic3.sType,
      .extendedDynamicState3ColorBlendEnable = deferred};
   cond_feature.conditionalRendering = conditional;
   cond_feature.inheritedConditionalRendering = VK_FALSE;
   clip_enable.depthClipEnable = depth;
   clip_control.depthClipControl = depth;
   mf = (VkPhysicalDeviceMeshShaderFeaturesEXT){
      .sType = mf.sType,
      .pNext = conditional ? &cond_feature : depth ? (void *)&clip_control : deferred ? (void *)&dynamic3 : NULL,
      .meshShader = VK_FALSE};
   v12 = (VkPhysicalDeviceVulkan12Features){.sType = v12.sType,
                                            .pNext = &mf,
                                            .drawIndirectCount = VK_FALSE};
   v11 = (VkPhysicalDeviceVulkan11Features){.sType = v11.sType, .pNext = &v12,
      .multiview = multiview, .multiviewTessellationShader = multiview};
   v13 = (VkPhysicalDeviceVulkan13Features){.sType = v13.sType,
                                            .pNext = &v11,
                                            .synchronization2 = timestamp,
                                            .dynamicRendering = VK_TRUE};
   features = (VkPhysicalDeviceFeatures2){.sType = features.sType,
                                          .pNext = &v13,
                                          .features = {.tessellationShader = VK_TRUE, .depthClamp = depth_clamp,
                                             .pipelineStatisticsQuery = statistics, .fillModeNonSolid = nonfill}};
   uint32_t n = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
   VkQueueFamilyProperties *families = calloc(n, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families);
   uint32_t qfam = 0;
   while (qfam < n && (!(families[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT) ||
                       (timestamp && !families[qfam].timestampValidBits)))
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
   const char *extensions[6];
   unsigned extension_count = 0;
   if (conditional) extensions[extension_count++] = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
   if (conservative) extensions[extension_count++] = VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME;
   if (deferred) extensions[extension_count++] = VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME;
   if (depth) {
      extensions[extension_count++] = VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME;
      extensions[extension_count++] = VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME;
   }
   uint32_t extension_total = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(phys, NULL, &extension_total, NULL));
   VkExtensionProperties *supported = calloc(extension_total, sizeof(*supported));
   CHECK(vkEnumerateDeviceExtensionProperties(phys, NULL, &extension_total, supported));
   for (unsigned i = 0; i < extension_count; ++i) {
      bool found = false;
      for (unsigned j = 0; j < extension_total; ++j)
         found |= !strcmp(extensions[i], supported[j].extensionName);
      if (!found) {
         printf("SKIP required extension %s not advertised\n", extensions[i]);
         free(supported); return 77;
      }
   }
   free(supported);
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &features,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .enabledExtensionCount = extension_count,
                             .ppEnabledExtensionNames = extensions};
   CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
   VkDescriptorSetLayoutCreateInfo slci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding};
   VkDescriptorSetLayout set_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &set_layout));
   VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
   VkDescriptorPool descriptor_pool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &descriptor_pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool, .descriptorSetCount = 1,
      .pSetLayouts = &set_layout};
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
   struct buffer counts = buffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
   ((float *)counts.map)[0] = inner;
   ((float *)counts.map)[1] = outer;
   ((float *)counts.map)[2] = z;
   VkDescriptorBufferInfo count_info = {counts.buffer, 0, 16};
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &count_info};
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);
   struct buffer indices = buffer(patch_size * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   for (unsigned i = 0; i < patch_size; ++i)
      ((uint32_t *)indices.map)[i] = i;
   struct buffer indirect_args = buffer(sizeof(VkDrawIndirectCommand),
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                        (conditional ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0));
   *(VkDrawIndirectCommand *)indirect_args.map = (VkDrawIndirectCommand){patch_size, 1, 0, 0};
   VkPushConstantRange push_range = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkShaderModule modules[6];
   for (unsigned i = 0; i < 6; ++i)
      modules[i] = module(argv[1 + i]);
   VkPipelineShaderStageCreateInfo stages[4] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = modules[2],
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = modules[3],
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
       .module = modules[0], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
       .module = modules[5], .pName = "main"},
   };
   VkViewport viewport = {0, 0, WIDTH, HEIGHT, 0, 1};
   VkRect2D scissor = {{0, 0}, {WIDTH, HEIGHT}};
   VkPipelineViewportDepthClipControlCreateInfoEXT clip = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
      .negativeOneToOne = negative_one};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = depth ? &clip : NULL,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationDepthClipStateCreateInfoEXT clip_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
      .depthClipEnable = depth_clip};
   VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
      .pNext = depth ? &clip_state : NULL,
      .conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = conservative ? &conservative_state : depth ? (void *)&clip_state : NULL,
      .depthClampEnable = depth_clamp,
      .frontFace = front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
      .polygonMode = nonfill ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL,
      .rasterizerDiscardEnable = discard,
      .cullMode = cull_mode,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = msaa ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ba};
   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &format, .viewMask = multiview ? 3 : 0};
   rendering.depthAttachmentFormat = depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED;
   VkPipelineDepthStencilStateCreateInfo depth_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth, .depthWriteEnable = depth,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS};
   VkGraphicsPipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &depth_state,
      .pColorBlendState = &cb,
      .layout = layout};
   VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_SCISSOR,
                               VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = deferred ? 2 : 1,
      .pDynamicStates = dynamic};
   pci.pDynamicState = &ds;
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = restart ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = restart};
   VkPipelineTessellationDomainOriginStateCreateInfo origin = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
      .domainOrigin = lower_left ? VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT
                                : VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT};
   VkPipelineTessellationStateCreateInfo tess = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      .pNext = &origin, .patchControlPoints = patch_size};
   uint64_t compile_start = monotonic_ns();
   VkPipeline pipelines[3];
   for (unsigned i = 0; i < 3; ++i) {
      stages[2].module = modules[i];
      pci.stageCount = i == 2 ? 2 : 4;
      pci.pTessellationState = i == 2 ? NULL : &tess;
      ia.topology = i == 2 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                          : VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
      pci.pVertexInputState = &vi;
      pci.pInputAssemblyState = &ia;
      if (i == 2) {
         stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
         stages[1].module = modules[4];
         pci.pVertexInputState = &vi;
         pci.pInputAssemblyState = &ia;
      }
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL,
                                      &pipelines[i]));
   }
   uint64_t compile_ns = monotonic_ns() - compile_start;
   puts("PIPELINE_OK");
   fflush(stdout);
   VkImageCreateInfo imci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                             .imageType = VK_IMAGE_TYPE_2D,
                             .format = format,
                             .extent = {WIDTH, HEIGHT, 1},
                             .mipLevels = 1,
                             .arrayLayers = layers,
                             .samples = msaa ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT,
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
      .viewType = layers == 2 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkImage resolve_image = VK_NULL_HANDLE;
   VkImageView resolve_view = VK_NULL_HANDLE;
   VkDeviceMemory resolve_memory = VK_NULL_HANDLE;
   if (msaa) {
      VkImageCreateInfo ri = imci; ri.samples = VK_SAMPLE_COUNT_1_BIT;
      CHECK(vkCreateImage(dev, &ri, NULL, &resolve_image));
      vkGetImageMemoryRequirements(dev, resolve_image, &mr);
      mai.allocationSize = mr.size;
      mai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      CHECK(vkAllocateMemory(dev, &mai, NULL, &resolve_memory));
      CHECK(vkBindImageMemory(dev, resolve_image, resolve_memory, 0));
      VkImageViewCreateInfo rv = ivci; rv.image = resolve_image;
      CHECK(vkCreateImageView(dev, &rv, NULL, &resolve_view));
   }
   VkImage depth_image = VK_NULL_HANDLE;
   VkImageView depth_view = VK_NULL_HANDLE;
   VkDeviceMemory depth_memory = VK_NULL_HANDLE;
   if (depth) {
      VkImageCreateInfo di = imci;
      di.format = VK_FORMAT_D32_SFLOAT;
      di.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      CHECK(vkCreateImage(dev, &di, NULL, &depth_image));
      vkGetImageMemoryRequirements(dev, depth_image, &mr);
      mai.allocationSize = mr.size;
      mai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      CHECK(vkAllocateMemory(dev, &mai, NULL, &depth_memory));
      CHECK(vkBindImageMemory(dev, depth_image, depth_memory, 0));
      VkImageViewCreateInfo dv = ivci;
      dv.image = depth_image; dv.format = VK_FORMAT_D32_SFLOAT;
      dv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      CHECK(vkCreateImageView(dev, &dv, NULL, &depth_view));
   }
   struct buffer depth_pixels = buffer(WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(depth_pixels.map, 0xa5, WIDTH * HEIGHT * 4);
   struct buffer pixels =
      buffer(WIDTH * HEIGHT * 4 * layers, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(pixels.map, 0xa5, WIDTH * HEIGHT * 4 * layers);
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
   VkQueryPool query = VK_NULL_HANDLE;
   if (timestamp || occlusion || statistics) {
      VkQueryPoolCreateInfo q = {
         .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType = timestamp ? VK_QUERY_TYPE_TIMESTAMP : occlusion ? VK_QUERY_TYPE_OCCLUSION : VK_QUERY_TYPE_PIPELINE_STATISTICS,
         .pipelineStatistics = statistics ? VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT : 0,
         .queryCount = timestamp ? 2 : 1};
      CHECK(vkCreateQueryPool(dev, &q, NULL, &query));
   }
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   if (query)
      vkCmdResetQueryPool(cmd, query, 0, timestamp ? 2 : 1);
   if (deferred) {
      PFN_vkCmdSetColorBlendEnableEXT set_blend =
         (void *)vkGetDeviceProcAddr(dev, "vkCmdSetColorBlendEnableEXT");
      if (!set_blend)
         return 3;
      const VkBool32 enabled = VK_FALSE;
      set_blend(cmd, 0, 1, &enabled);
   }
   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                        NULL, 0, NULL, 1, &barrier);
   if (msaa) {
      VkImageMemoryBarrier rb = barrier; rb.image = resolve_image;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &rb);
   }
   if (depth) {
      VkImageMemoryBarrier db = barrier;
      db.image = depth_image; db.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         0, 0, NULL, 0, NULL, 1, &db);
   }
   VkRenderingAttachmentInfo depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth_view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = {1.0f, 0}};
   VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .resolveMode = msaa ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
      .resolveImageView = resolve_view, .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
                             .layerCount = layered ? 2 : 1,
                             .viewMask = multiview ? 3 : 0,
                             .colorAttachmentCount = 1,
                             .pColorAttachments = &att,
                             .pDepthAttachment = depth ? &depth_att : NULL};
   if (timestamp) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query, 0);
   vkCmdBeginRendering(cmd, &render);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                           0, 1, &set, 0, NULL);
   vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
   if (conditional) {
      PFN_vkCmdBeginConditionalRenderingEXT start_cond =
         (void *)vkGetDeviceProcAddr(dev, "vkCmdBeginConditionalRenderingEXT");
      if (!start_cond) return 3;
      VkConditionalRenderingBeginInfoEXT ci = {
         .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
         .buffer = indirect_args.buffer};
      start_cond(cmd, &ci);
   }
   if (occlusion || statistics) vkCmdBeginQuery(cmd, query, 0, 0);
   for (unsigned i = 0; sequence[i]; ++i) {
      if (merged && i == 1) {
         vkCmdEndRendering(cmd);
#ifdef HOST_MERGE_OBSERVER
         if (kept_merge) observe_merge(cmd, 0);
#endif
         /* A valid attachment-only dependency can stay within a merged
          * hardware render. The continuation loads the first stripe. */
         VkMemoryBarrier continuation = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
         vkCmdPipelineBarrier(
            cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_DEPENDENCY_BY_REGION_BIT, 1, &continuation, 0, NULL, 0, NULL);
         att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
         depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
         vkCmdBeginRendering(cmd, &render);
#ifdef HOST_MERGE_OBSERVER
         if (kept_merge) observe_merge(cmd, 1);
#endif
      }
      float shift = rewrite && (i & 1) ? 0.25f : 0.0f;
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &shift);
      VkRect2D stripe = {{i * 8, 0}, {strlen(sequence) == 1 ? WIDTH : 8, HEIGHT}};
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
         if ((indexed || restart) && pipeline != 2)
            vkCmdDrawIndexed(cmd, patch_size, 1, 0, 0, 0);
         else if (indirect && pipeline != 2)
            vkCmdDrawIndirect(cmd, indirect_args.buffer, 0, 1,
                              sizeof(VkDrawIndirectCommand));
         else
            vkCmdDraw(cmd, pipeline == 2 ? 3 : patch_size * (multiple ? 2 : 1), instances ? 2 : 1, offset ? patch_size : 0, first_instance ? 1 : 0);
      }
      if (timestamp)
         vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, query,
                              1);
   }
   if (conditional) {
      PFN_vkCmdEndConditionalRenderingEXT end_cond =
         (void *)vkGetDeviceProcAddr(dev, "vkCmdEndConditionalRenderingEXT");
      if (!end_cond) return 3;
      end_cond(cmd);
   }
   if (occlusion || statistics) vkCmdEndQuery(cmd, query, 0);
   vkCmdEndRendering(cmd);
   if (msaa) barrier.image = resolve_image;
   barrier.srcAccessMask = barrier.dstAccessMask;
   barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   barrier.oldLayout = barrier.newLayout;
   barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &barrier);
   VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
      .imageExtent = {WIDTH, HEIGHT, 1}};
   vkCmdCopyImageToBuffer(cmd, msaa ? resolve_image : image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          pixels.buffer, 1, &copy);
   if (depth) {
      VkImageMemoryBarrier db = barrier;
      db.image = depth_image; db.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      db.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      db.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cmd,
         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &db);
      VkBufferImageCopy dc = copy; dc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vkCmdCopyImageToBuffer(cmd, depth_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            depth_pixels.buffer, 1, &dc);
   }
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
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
   uint64_t submit_start = monotonic_ns();
   in_flight(vkQueueSubmit(queue, 1, &submit, fence), "submit");
   in_flight(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull), "fence");
   uint64_t submit_wait_ns = monotonic_ns() - submit_start;
   printf("TIMING compile_ns=%" PRIu64 " submit_wait_ns=%" PRIu64 "\n", compile_ns, submit_wait_ns);
   puts("SUBMIT_OK");
   fflush(stdout);
   bool timestamp_ready = true;
   if (timestamp) {
      uint64_t data[4] = {0};
      VkResult result = vkGetQueryPoolResults(
         dev, query, 0, 2, sizeof(data), data, 2 * sizeof(uint64_t),
         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      if (result != VK_SUCCESS && result != VK_NOT_READY)
         return 2;
      timestamp_ready = result == VK_SUCCESS && data[1] && data[3] && data[2] >= data[0];
      printf("timestamp result=%d value=%" PRIu64 " available=%" PRIu64 "\n",
             result, data[0], data[1]);
      printf("TIMING gpu_ns=%.3f available=%u\n", (double)(data[2] - data[0]) * timestamp_period, timestamp_ready);
      save(argv[7], "timestamp.bin", data, sizeof(data));
   }
   if (occlusion || statistics) {
      uint64_t data[2] = {0};
      VkResult qr = vkGetQueryPoolResults(dev, query, 0, 1, sizeof(data), data, sizeof(data),
         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      timestamp_ready = qr == VK_SUCCESS && data[1] != 0 && data[0] > 0;
      save(argv[7], "query.bin", data, sizeof(data));
      printf("QUERY result=%d value=%" PRIu64 " available=%" PRIu64 "\n", qr, data[0], data[1]);
   }
   save(argv[7], "pixels.bin", pixels.map, WIDTH * HEIGHT * 4 * layers);
   if (depth) save(argv[7], "depth.bin", depth_pixels.map, WIDTH * HEIGHT * 4);
   /* Independent planar interior oracle. The future device runner also
    * compares complete native/emulated readbacks, including diagonal edges.
    * An empty patch has an outer factor <= 0. The TES maps domain coordinates
    * directly into clip XY, translating by control point zero's X offset. */
   unsigned mismatches = 0, checked = 0, depth_mismatches = 0;
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         unsigned stripe = strlen(sequence) == 1 ? 0 : x / 8;
         uint8_t expected[4] = {0};
         float expected_depth = 1.0f;
         if (stripe < strlen(sequence)) {
            char kind = sequence[stripe];
            int px = (int)x - (rewrite && (stripe & 1) ? 8 : 0);
            bool square = px >= 0 && px < WIDTH;
            if (!quad && px + (int)y == 63 && kind != 'V' && kind != 'M')
               continue;
            bool covered = outer > 0 && (quad || px + (int)y < 63);
            /* Face orientation is checked by native/emulated equality. */
            bool clockwise = !strcmp(option("TESS_TEST_ORDER", "cw"), "cw") ^ lower_left;
            bool culled = cull_mode == VK_CULL_MODE_FRONT_BIT
               ? (front_ccw != clockwise) : cull_mode == VK_CULL_MODE_BACK_BIT
               ? (front_ccw == clockwise) : false;
            bool clipped = depth && depth_clip && (z < (negative_one ? -1.0f : 0.0f) || z > 1.0f);
            if (kind == 'V' || kind == 'M' || (square && covered && !culled && !clipped && !discard)) {
               expected[0] = kind == 'V' || kind == 'M' ? 255 : 0;
               expected[1] = kind == 'V' ? 255 : 0;
               expected[2] = kind == 'M' ? 0 : 255;
               expected[3] = 255;
               expected_depth = negative_one ? (z + 1.0f) * .5f : z;
               if (depth_clamp) {
                  if (expected_depth < 0) expected_depth = 0;
                  if (expected_depth > 1) expected_depth = 1;
               }
            }
         }
         if (depth) {
            float actual = ((float *)depth_pixels.map)[y * WIDTH + x];
            float delta = actual - expected_depth;
            depth_mismatches += !(delta > -0.000001f && delta < 0.000001f);
         }
         ++checked;
         mismatches += memcmp((uint8_t *)pixels.map + 4 * (y * WIDTH + x), expected, 4) != 0;
      }
   }
   if (multiview) mismatches += memcmp(pixels.map, (char *)pixels.map + WIDTH * HEIGHT * 4, WIDTH * HEIGHT * 4) != 0;
   if (layered) {
      const uint8_t *second = (const uint8_t *)pixels.map + WIDTH * HEIGHT * 4;
      for (unsigned j = 0; j < WIDTH * HEIGHT * 4; ++j) mismatches += second[j] != 0;
   }
   printf("mode=%s sequence=%s patch=%u inner=%g outer=%g pixels_mismatched=%u/%u\n",
          mode, sequence, patch_size, inner, outer, mismatches, checked);
   printf("depth_mismatched=%u\n", depth_mismatches);
   CHECK(vkDeviceWaitIdle(dev));
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, commands, NULL);
   if (query)
      vkDestroyQueryPool(dev, query, NULL);
   for (unsigned i = 0; i < 3; ++i)
      vkDestroyPipeline(dev, pipelines[i], NULL);
   for (unsigned i = 0; i < 6; ++i)
      vkDestroyShaderModule(dev, modules[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorPool(dev, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   struct buffer buffers[] = {pixels, depth_pixels, counts, indices, indirect_args};
   for (unsigned i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
      vkUnmapMemory(dev, buffers[i].memory);
      vkDestroyBuffer(dev, buffers[i].buffer, NULL);
      vkFreeMemory(dev, buffers[i].memory, NULL);
   }
   if (msaa) {
      vkDestroyImageView(dev, resolve_view, NULL);
      vkDestroyImage(dev, resolve_image, NULL);
      vkFreeMemory(dev, resolve_memory, NULL);
   }
   if (depth) {
      vkDestroyImageView(dev, depth_view, NULL);
      vkDestroyImage(dev, depth_image, NULL);
      vkFreeMemory(dev, depth_memory, NULL);
   }
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, image_memory, NULL);
   vkDestroyDevice(dev, NULL);
   destroy_debug(instance, messenger, NULL);
   vkDestroyInstance(instance, NULL);
   bool passed = (nonfill || conservative || !mismatches) && !depth_mismatches && timestamp_ready;
   puts(passed ? "PASS" : "FAIL");
   return passed ? 0 : 1;
}
