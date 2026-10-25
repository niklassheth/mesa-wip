/* SPDX-License-Identifier: MIT
 * Host-only pipeline/packet recording for unchanged Wicked mesh/task modules.
 * Dummy descriptors are initialized, but are not a runnable engine scene.
 * Only the host non-executing DRM shim may run this probe.
 */
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
#define Q_BYTES (512*1024)
#define CHECK(call) do { VkResult r=(call); if(r!=VK_SUCCESS) { \
   fprintf(stderr,"%s: VkResult %d\n",#call,r);exit(2); } } while(0)
static VkDevice dev;
static VkPhysicalDevice phys;
struct buffer { VkBuffer buffer; VkDeviceMemory memory; void *map; };

static uint32_t memory_type(uint32_t bits,VkMemoryPropertyFlags flags) {
   VkPhysicalDeviceMemoryProperties p;vkGetPhysicalDeviceMemoryProperties(phys,&p);
   for(unsigned i=0;i<p.memoryTypeCount;i++)
      if((bits&(1u<<i)) && (p.memoryTypes[i].propertyFlags&flags)==flags)return i;
   fprintf(stderr,"no matching memory type\n");exit(2);
}
static struct buffer buffer(size_t bytes,VkBufferUsageFlags usage) {
   struct buffer b={0};VkBufferCreateInfo ci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=usage};
   CHECK(vkCreateBuffer(dev,&ci,NULL,&b.buffer));VkMemoryRequirements m;vkGetBufferMemoryRequirements(dev,b.buffer,&m);
   VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=m.size,
      .memoryTypeIndex=memory_type(m.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CHECK(vkAllocateMemory(dev,&ai,NULL,&b.memory));CHECK(vkBindBufferMemory(dev,b.buffer,b.memory,0));
   CHECK(vkMapMemory(dev,b.memory,0,VK_WHOLE_SIZE,0,&b.map));return b;
}
static VkShaderModule module(const char *path) {
   FILE *f=fopen(path,"rb");if(!f||fseek(f,0,SEEK_END))exit(2);long bytes=ftell(f);rewind(f);
   if(bytes<20||bytes%4||bytes>1048576)exit(2);
   uint32_t *data=malloc(bytes);
   if(!data||fread(data,1,bytes,f)!=(size_t)bytes||fclose(f))exit(2);
   VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=bytes,.pCode=data};
   VkShaderModule m;CHECK(vkCreateShaderModule(dev,&ci,NULL,&m));free(data);return m;
}
static void save(const char *dir,const char *name,const void *data,size_t bytes) {
   char path[4096];if(snprintf(path,sizeof(path),"%s/%s",dir,name)>=(int)sizeof(path))exit(2);
   FILE *f=fopen(path,"wb");if(!f||fwrite(data,1,bytes,f)!=bytes||fclose(f))exit(2);
}
static void retain(VkResult result,const char *where) {
   fprintf(stderr,"IN_FLIGHT_WAIT_FAILED where=%s result=%d; retain backing\n",where,result);
   fflush(NULL);raise(SIGSTOP);for(;;)pause();
}
int main(int argc,char **argv) {
   if (!getenv("WICKED_HOST_RECORD_ONLY") || !getenv("ASAHI_HOST_TRACE_DIR")) return 2;
   setvbuf(stdout,NULL,_IOLBF,0);
   if(argc!=5&&argc!=6) {fprintf(stderr,"usage: probe mesh.spv frag.spv inputs.bin out [task.spv]\n");return 2;}
   const char *task_path=argc==6?argv[5]:NULL;
   const VkShaderStageFlags producer=VK_SHADER_STAGE_MESH_BIT_EXT|
      (task_path?VK_SHADER_STAGE_TASK_BIT_EXT:0);
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="native-mesh-wicked-record",.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance instance;CHECK(vkCreateInstance(&ici,NULL,&instance));uint32_t count=0;
   CHECK(vkEnumeratePhysicalDevices(instance,&count,NULL));VkPhysicalDevice *devices=calloc(count,sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance,&count,devices));
   for(unsigned i=0;i<count;i++) {
      VkPhysicalDeviceSubgroupProperties subgroup={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
      VkPhysicalDeviceDriverProperties driver={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,.pNext=&subgroup};
      VkPhysicalDeviceProperties2 props={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&driver};
      vkGetPhysicalDeviceProperties2(devices[i],&props);
      bool soft=getenv("M2_SOFTWARE_REFERENCE")&&strstr(props.properties.deviceName,"llvmpipe");
      if(driver.driverID!=VK_DRIVER_ID_MESA_HONEYKRISP&&!soft)continue;
      phys=devices[i];printf("renderer=%s driver=%s info=%s\n",props.properties.deviceName,driver.driverName,driver.driverInfo);
      printf("subgroup_size=%u\n",subgroup.subgroupSize);break;
   }
   free(devices);if(!phys){fprintf(stderr,"no expected renderer\n");return 2;}
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&mesh};
   VkPhysicalDeviceVulkan13Features f13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&f12};
   VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&f13};
   vkGetPhysicalDeviceFeatures2(phys,&features);
   if(!mesh.meshShader||(task_path&&!mesh.taskShader)||!f13.synchronization2||!f13.maintenance4||!f12.storageBuffer8BitAccess||!f12.shaderInt8){fprintf(stderr,"required Vulkan features missing\n");return 2;}
   VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(phys,VK_FORMAT_R8G8B8A8_UNORM,&fp);
   if(!(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)){fprintf(stderr,"R32 float blend unsupported\n");return 2;}
   uint32_t qcount=0;vkGetPhysicalDeviceQueueFamilyProperties(phys,&qcount,NULL);
   VkQueueFamilyProperties *families=calloc(qcount,sizeof(*families));vkGetPhysicalDeviceQueueFamilyProperties(phys,&qcount,families);
   uint32_t family=0;while(family<qcount&&!(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT))family++;
   free(families);if(family==qcount)return 2;float priority=1;
   VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
   VkPhysicalDeviceFeatures base=features.features;
   mesh=(VkPhysicalDeviceMeshShaderFeaturesEXT){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=VK_TRUE,.taskShader=task_path!=NULL};
   f12.pNext=&mesh;
   f13=(VkPhysicalDeviceVulkan13Features){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&f12,.synchronization2=VK_TRUE,.maintenance4=VK_TRUE};
   /* Mesh-stage stores do not require vertexPipelineStoresAndAtomics. */
   const char *extension=VK_EXT_MESH_SHADER_EXTENSION_NAME;
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f13,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi,
      .enabledExtensionCount=1,.ppEnabledExtensionNames=&extension,.pEnabledFeatures=&base};
   CHECK(vkCreateDevice(phys,&dci,NULL,&dev));VkQueue queue;vkGetDeviceQueue(dev,family,0,&queue);
   PFN_vkCmdDrawMeshTasksEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");if(!draw)return 2;
   struct buffer q=buffer(Q_BYTES,(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT));
   struct buffer pixels=buffer(W*H*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(q.map,0xcc,Q_BYTES);memset(pixels.map,0xa5,W*H*4);
   FILE *input=fopen(argv[3],"rb");
   if(!input||fread(q.map,1,Q_BYTES,input)!=Q_BYTES||fgetc(input)!=EOF||fclose(input))return 2;
   VkImageCreateInfo imci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,
      .extent={W,H,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
      .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image;CHECK(vkCreateImage(dev,&imci,NULL,&image));VkMemoryRequirements req;vkGetImageMemoryRequirements(dev,image,&req);
   VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=memory_type(req.memoryTypeBits,0)};
   VkDeviceMemory image_mem;CHECK(vkAllocateMemory(dev,&mai,NULL,&image_mem));CHECK(vkBindImageMemory(dev,image,image_mem,0));
   VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,.viewType=VK_IMAGE_VIEW_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkImageView view;CHECK(vkCreateImageView(dev,&ivci,NULL,&view));
   VkAttachmentDescription att={.format=VK_FORMAT_R8G8B8A8_UNORM,.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp=VK_ATTACHMENT_STORE_OP_STORE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ref};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&att,.subpassCount=1,.pSubpasses=&sub};
   VkRenderPass pass;CHECK(vkCreateRenderPass(dev,&rpci,NULL,&pass));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=pass,.attachmentCount=1,.pAttachments=&view,.width=W,.height=H,.layers=1};
   VkFramebuffer fb;CHECK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));
   const VkShaderStageFlags both=producer|VK_SHADER_STAGE_FRAGMENT_BIT;
   VkDescriptorSetLayoutBinding bindings[5][3]={
      {{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,both,NULL},
       {1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,both,NULL},
       {3102,VK_DESCRIPTOR_TYPE_SAMPLER,1,both,NULL}},
      {{0}},
      {{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,both,NULL}},
      {{0,VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,1,both,NULL}},
      {{0,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,both,NULL}},
   };
   unsigned binding_counts[5]={3,0,1,1,1};
   VkDescriptorSetLayout setlayouts[5];
   for(unsigned i=0;i<5;i++){
      VkDescriptorSetLayoutCreateInfo ci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=binding_counts[i],.pBindings=bindings[i]};
      CHECK(vkCreateDescriptorSetLayout(dev,&ci,NULL,&setlayouts[i]));
   }
   VkPushConstantRange push={producer,0,128};
   VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=5,.pSetLayouts=setlayouts,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
   VkPipelineLayout layout;CHECK(vkCreatePipelineLayout(dev,&plci,NULL,&layout));
   VkDescriptorPoolSize pools[]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1},{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,1},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1},{VK_DESCRIPTOR_TYPE_SAMPLER,1}};
   VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=5,.poolSizeCount=5,.pPoolSizes=pools};
   VkDescriptorPool dp;CHECK(vkCreateDescriptorPool(dev,&dpci,NULL,&dp));
   VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=5,.pSetLayouts=setlayouts};
   VkDescriptorSet sets[5];CHECK(vkAllocateDescriptorSets(dev,&dsai,sets));
   for(unsigned i=0;i<3;i++){
      VkDescriptorBufferInfo bi={q.buffer,0,65536};
      VkWriteDescriptorSet write={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[i<2?0:2],.dstBinding=i<2?i:0,.descriptorCount=1,.descriptorType=i<2?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi};
      vkUpdateDescriptorSets(dev,1,&write,0,NULL);
   }
   VkBufferView texel;
   VkBufferViewCreateInfo tvci={.sType=VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,.buffer=q.buffer,.format=VK_FORMAT_R32G32B32A32_SFLOAT,.range=65536};
   CHECK(vkCreateBufferView(dev,&tvci,NULL,&texel));
   VkWriteDescriptorSet twrite={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[3],.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,.pTexelBufferView=&texel};
   vkUpdateDescriptorSets(dev,1,&twrite,0,NULL);
   VkSamplerCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,.magFilter=VK_FILTER_LINEAR,.minFilter=VK_FILTER_LINEAR,.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,.addressModeU=VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,.addressModeV=VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,.addressModeW=VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,.maxLod=0};
   VkSampler sampler;CHECK(vkCreateSampler(dev,&sci,NULL,&sampler));
   VkDescriptorImageInfo sampler_info={.sampler=sampler};
   VkWriteDescriptorSet swrite={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[0],.dstBinding=3102,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER,.pImageInfo=&sampler_info};
   vkUpdateDescriptorSets(dev,1,&swrite,0,NULL);
   VkImageCreateInfo tci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_3D,.format=VK_FORMAT_R8G8B8A8_UNORM,.extent={1,1,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_SAMPLED_BIT};
   VkImage texture;CHECK(vkCreateImage(dev,&tci,NULL,&texture));
   vkGetImageMemoryRequirements(dev,texture,&req);
   mai.allocationSize=req.size;mai.memoryTypeIndex=memory_type(req.memoryTypeBits,0);
   VkDeviceMemory texture_mem;CHECK(vkAllocateMemory(dev,&mai,NULL,&texture_mem));CHECK(vkBindImageMemory(dev,texture,texture_mem,0));
   ivci.image=texture;ivci.viewType=VK_IMAGE_VIEW_TYPE_3D;
   VkImageView texture_view;CHECK(vkCreateImageView(dev,&ivci,NULL,&texture_view));
   VkDescriptorImageInfo texture_info={.imageView=texture_view,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   VkWriteDescriptorSet iwrite={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[4],.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,.pImageInfo=&texture_info};
   vkUpdateDescriptorSets(dev,1,&iwrite,0,NULL);
   VkShaderModule ms=module(argv[1]),fs=module(argv[2]),ts=task_path?module(task_path):VK_NULL_HANDLE;
   VkPipelineShaderStageCreateInfo stages[]={
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_MESH_BIT_EXT,.module=ms,.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_TASK_BIT_EXT,.module=ts,.pName="main"}};
   VkViewport viewport={0,0,W,H,0,1};VkRect2D scissor={{0,0},{W,H}};
   VkPipelineViewportStateCreateInfo vs={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&scissor};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
   VkPipelineMultisampleStateCreateInfo samples={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState blend={.blendEnable=VK_TRUE,.srcColorBlendFactor=VK_BLEND_FACTOR_ONE,.dstColorBlendFactor=VK_BLEND_FACTOR_ONE,.colorBlendOp=VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,.dstAlphaBlendFactor=VK_BLEND_FACTOR_ONE,.alphaBlendOp=VK_BLEND_OP_ADD,.colorWriteMask=(VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT)};
   VkPipelineColorBlendStateCreateInfo cb={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&blend};
   VkGraphicsPipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=task_path?3:2,.pStages=stages,.pViewportState=&vs,.pRasterizationState=&rs,
      .pMultisampleState=&samples,.pColorBlendState=&cb,.layout=layout,.renderPass=pass};
   VkPipeline pipeline;CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev,&cpci,NULL,&pool));VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
   VkCommandBuffer cmd;CHECK(vkAllocateCommandBuffers(dev,&cai,&cmd));VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};CHECK(vkBeginCommandBuffer(cmd,&begin));
   VkMemoryBarrier2 host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.srcAccessMask=VK_ACCESS_2_HOST_WRITE_BIT,
      .dstStageMask=VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT|VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,.dstAccessMask=VK_ACCESS_2_UNIFORM_READ_BIT|VK_ACCESS_2_SHADER_SAMPLED_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
   if(task_path)host.dstStageMask|=VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
   VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.memoryBarrierCount=1,.pMemoryBarriers=&host};vkCmdPipelineBarrier2(cmd,&dep);
   VkImageMemoryBarrier2 texture_barrier={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,.dstStageMask=VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,.dstAccessMask=VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=texture,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkDependencyInfo texture_dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.imageMemoryBarrierCount=1,.pImageMemoryBarriers=&texture_barrier};vkCmdPipelineBarrier2(cmd,&texture_dep);
   VkClearValue clear={.color={{0,0,0,0}}};VkRenderPassBeginInfo rpbegin={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=pass,.framebuffer=fb,.renderArea={{0,0},{W,H}},.clearValueCount=1,.pClearValues=&clear};
   vkCmdBeginRenderPass(cmd,&rpbegin,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
   vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,5,sets,0,NULL);
   vkCmdPushConstants(cmd,layout,producer,0,128,(uint8_t *)q.map+3072);
   draw(cmd,1,1,1);vkCmdEndRenderPass(cmd);
   VkMemoryBarrier2 before_copy={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      .srcAccessMask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT|VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_TRANSFER_READ_BIT|VK_ACCESS_2_HOST_READ_BIT};
   dep.pMemoryBarriers=&before_copy;vkCmdPipelineBarrier2(cmd,&dep);
   VkBufferImageCopy copy={.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={W,H,1}};
   vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,pixels.buffer,1,&copy);
   VkMemoryBarrier2 after_copy={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,.srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_HOST_READ_BIT};dep.pMemoryBarriers=&after_copy;vkCmdPipelineBarrier2(cmd,&dep);
   CHECK(vkEndCommandBuffer(cmd));VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;CHECK(vkCreateFence(dev,&fci,NULL,&fence));
   VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};puts("SUBMIT_BEGIN");
   VkResult result=vkQueueSubmit(queue,1,&submit,fence);if(result!=VK_SUCCESS)retain(result,"submit");
   result=vkWaitForFences(dev,1,&fence,VK_TRUE,15000000000ull);if(result!=VK_SUCCESS)retain(result,"wait");
   puts("SUBMIT_OK");save(argv[4],"pixels.bin",pixels.map,W*H*4);save(argv[4],"Q.bin",q.map,Q_BYTES);
   puts("HOST_RECORD_COMPLETE V=64 P=124 no_gpu_execution=1");
   vkDestroyFence(dev,fence,NULL);vkDestroyCommandPool(dev,pool,NULL);vkDestroyPipeline(dev,pipeline,NULL);
   vkDestroyShaderModule(dev,ms,NULL);vkDestroyShaderModule(dev,fs,NULL);vkDestroyDescriptorPool(dev,dp,NULL);
   if(ts)vkDestroyShaderModule(dev,ts,NULL);
   vkDestroyPipelineLayout(dev,layout,NULL);for(unsigned i=0;i<5;i++)vkDestroyDescriptorSetLayout(dev,setlayouts[i],NULL);
   vkDestroyBufferView(dev,texel,NULL);vkDestroySampler(dev,sampler,NULL);
   vkDestroyImageView(dev,texture_view,NULL);vkDestroyImage(dev,texture,NULL);vkFreeMemory(dev,texture_mem,NULL);
   vkDestroyFramebuffer(dev,fb,NULL);vkDestroyRenderPass(dev,pass,NULL);vkDestroyImageView(dev,view,NULL);
   vkDestroyImage(dev,image,NULL);vkFreeMemory(dev,image_mem,NULL);
   struct buffer *buffers[]={&q,&pixels};for(unsigned i=0;i<2;i++){vkUnmapMemory(dev,buffers[i]->memory);vkDestroyBuffer(dev,buffers[i]->buffer,NULL);vkFreeMemory(dev,buffers[i]->memory,NULL);}
   vkDestroyDevice(dev,NULL);vkDestroyInstance(instance,NULL);return 0;
}
