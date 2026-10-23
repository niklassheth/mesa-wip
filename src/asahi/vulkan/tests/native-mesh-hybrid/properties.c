/* SPDX-License-Identifier: MIT */
/* Compare mesh API properties/features byte-for-byte with native off/on. */
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS)                                                     \
         return 2;                                                             \
   } while (0)
int
main(int argc, char **argv)
{
   if (argc != 2)
      return 2;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ci, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice device;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &device));
   if (count != 1)
      return 2;
   VkPhysicalDeviceMeshShaderPropertiesEXT mesh = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &mesh};
   vkGetPhysicalDeviceProperties2(device, &props);
   VkPhysicalDeviceMeshShaderFeaturesEXT mf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &mf};
   vkGetPhysicalDeviceFeatures2(device, &features);
   if (!strstr(props.properties.deviceName, "Apple M4"))
      return 2;
   printf("device=%s mesh=%u task=%u multiview=%u query=%u max_vertices=%u "
          "max_primitives=%u max_components=%u max_output=%u max_shared=%u "
          "max_payload=%u max_groups=%u max_layers=%u max_views=%u\n",
          props.properties.deviceName, mf.meshShader, mf.taskShader,
          mf.multiviewMeshShader, mf.meshShaderQueries,
          mesh.maxMeshOutputVertices, mesh.maxMeshOutputPrimitives,
          mesh.maxMeshOutputComponents, mesh.maxMeshOutputMemorySize,
          mesh.maxMeshSharedMemorySize, mesh.maxTaskPayloadSize,
          mesh.maxMeshWorkGroupTotalCount, mesh.maxMeshOutputLayers,
          mesh.maxMeshMultiviewViewCount);
   FILE *out = fopen(argv[1], "wb");
   if (!out || fwrite(&mesh, sizeof(mesh), 1, out) != 1 ||
       fwrite(&mf, sizeof(mf), 1, out) != 1 || fclose(out))
      return 2;
   vkDestroyInstance(instance, NULL);
   return 0;
}
