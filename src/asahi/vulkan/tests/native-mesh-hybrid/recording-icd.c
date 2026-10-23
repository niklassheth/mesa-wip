/* SPDX-License-Identifier: MIT
 * Host-only recording proxy. Compiles and records through the real Mesa ICD,
 * adds debug messages, and suppresses queue submission. The no-op DRM shim
 * supplies nonexecuting allocation/wait operations. Pixels are never validated
 * by this harness. Use the separate bounded recorder for real UAPI packets. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
static void *library;
static PFN_vkGetInstanceProcAddr real_gipa;
static PFN_vkGetDeviceProcAddr real_gdpa;
static struct {
   VkInstance instance;
   VkDebugUtilsMessengerEXT messenger;
} records[64];
static void
init(void)
{
   if (real_gipa)
      return;
   library = dlopen(getenv("COVERAGE_REAL_ICD"), RTLD_NOW | RTLD_LOCAL);
   if (!library) {
      fprintf(stderr, "COVERAGE_PROXY_FATAL %s\n", dlerror());
      abort();
   }
   real_gipa = (void *)dlsym(library, "vk_icdGetInstanceProcAddr");
   if (!real_gipa)
      abort();
}
static VKAPI_ATTR VkBool32 VKAPI_CALL
message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT *d, void *data)
{
   (void)severity;
   (void)type;
   (void)data;
   fprintf(stderr, "COVERAGE_VK %s: %s\n",
           d->pMessageIdName ? d->pMessageIdName : "?",
           d->pMessage ? d->pMessage : "");
   fflush(stderr);
   return VK_FALSE;
}
static VKAPI_ATTR VkResult VKAPI_CALL
create(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *alloc,
       VkInstance *out)
{
   init();
   VkInstanceCreateInfo copy = *ci;
   const char **extensions =
      calloc(ci->enabledExtensionCount + 1, sizeof(char *));
   if (!extensions)
      abort();
   unsigned n = 0;
   int found = 0;
   for (; n < ci->enabledExtensionCount; n++) {
      extensions[n] = ci->ppEnabledExtensionNames[n];
      found |= !strcmp(extensions[n], VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
   }
   if (!found)
      extensions[n++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
   copy.enabledExtensionCount = n;
   copy.ppEnabledExtensionNames = extensions;
   VkDebugUtilsMessengerCreateInfoEXT m = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .pNext = ci->pNext,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = message};
   copy.pNext = &m;
   PFN_vkCreateInstance f =
      (void *)real_gipa(VK_NULL_HANDLE, "vkCreateInstance");
   VkResult r = f(&copy, alloc, out);
   free(extensions);
   if (r == VK_SUCCESS) {
      m.pNext = NULL;
      unsigned i = 0;
      while (i < 64 && records[i].instance)
         i++;
      if (i == 64)
         abort();
      records[i].instance = *out;
      PFN_vkCreateDebugUtilsMessengerEXT mk =
         (void *)real_gipa(*out, "vkCreateDebugUtilsMessengerEXT");
      if (!mk || mk(*out, &m, NULL, &records[i].messenger) != VK_SUCCESS)
         abort();
   }
   return r;
}
static VKAPI_ATTR void VKAPI_CALL
destroy(VkInstance inst, const VkAllocationCallbacks *alloc)
{
   PFN_vkDestroyDebugUtilsMessengerEXT drop =
      (void *)real_gipa(inst, "vkDestroyDebugUtilsMessengerEXT");
   for (unsigned i = 0; i < 64; i++)
      if (records[i].instance == inst) {
         drop(inst, records[i].messenger, NULL);
         memset(&records[i], 0, sizeof(records[i]));
      }
   PFN_vkDestroyInstance f = (void *)real_gipa(inst, "vkDestroyInstance");
   f(inst, alloc);
}
static VKAPI_ATTR VkResult VKAPI_CALL
submit(VkQueue q, uint32_t count, const VkSubmitInfo *info, VkFence fence)
{
   (void)q;
   (void)info;
   (void)fence;
   fprintf(stderr, "HOST_RECORDING_ONLY suppressed QueueSubmit count=%u\n",
           count);
   return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL
submit2(VkQueue q, uint32_t count, const VkSubmitInfo2 *info, VkFence fence)
{
   (void)q;
   (void)info;
   (void)fence;
   fprintf(stderr, "HOST_RECORDING_ONLY suppressed QueueSubmit2 count=%u\n",
           count);
   return VK_SUCCESS;
}
static PFN_vkVoidFunction
overridden(const char *name)
{
   if (!strcmp(name, "vkQueueSubmit"))
      return (void *)submit;
   if (!strcmp(name, "vkQueueSubmit2") || !strcmp(name, "vkQueueSubmit2KHR"))
      return (void *)submit2;
   return NULL;
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
device_proc(VkDevice dev, const char *name)
{
   PFN_vkVoidFunction override = overridden(name);
   if (override)
      return override;
   if (!real_gdpa)
      abort();
   return real_gdpa(dev, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance inst, const char *name)
{
   init();
   PFN_vkVoidFunction override = overridden(name);
   if (override)
      return override;
   if (!strcmp(name, "vkGetDeviceProcAddr")) {
      real_gdpa = (void *)real_gipa(inst, name);
      return (void *)device_proc;
   }
   if (!strcmp(name, "vkCreateInstance"))
      return (void *)create;
   if (!strcmp(name, "vkDestroyInstance"))
      return (void *)destroy;
   return real_gipa(inst, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetPhysicalDeviceProcAddr(VkInstance inst, const char *name)
{
   init();
   PFN_vkGetInstanceProcAddr f =
      (void *)dlsym(library, "vk_icdGetPhysicalDeviceProcAddr");
   return f ? f(inst, name) : real_gipa(inst, name);
}
VKAPI_ATTR VkResult VKAPI_CALL
vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *v)
{
   init();
   VkResult (*f)(uint32_t *) =
      dlsym(library, "vk_icdNegotiateLoaderICDInterfaceVersion");
   return f ? f(v) : VK_SUCCESS;
}
