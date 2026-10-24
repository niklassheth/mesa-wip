/* SPDX-License-Identifier: MIT
 * Unchanged Niagara mesh and fragment modules, original two G-buffer formats. Each process submits one draw.
 * Python independently checks every output pixel and the entire guarded Q.
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
struct image { VkImage image; VkDeviceMemory memory; VkImageView view; };
static struct image image(VkFormat format, unsigned w, unsigned h, VkImageUsageFlags usage) {
   struct image im={0};
   VkImageCreateInfo ci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=format,
      .extent={w,h,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,.usage=usage};
   CHECK(vkCreateImage(dev,&ci,NULL,&im.image));VkMemoryRequirements req;vkGetImageMemoryRequirements(dev,im.image,&req);
   VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=memory_type(req.memoryTypeBits,0)};
   CHECK(vkAllocateMemory(dev,&ai,NULL,&im.memory));CHECK(vkBindImageMemory(dev,im.image,im.memory,0));
   VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=im.image,.viewType=VK_IMAGE_VIEW_TYPE_2D,
      .format=format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   CHECK(vkCreateImageView(dev,&vi,NULL,&im.view));return im;
}
int main(int argc,char **argv) {
   setvbuf(stdout,NULL,_IOLBF,0);
   if(argc!=5){fprintf(stderr,"usage: probe mesh.spv frag.spv inputs.bin out\n");return 2;}
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="native-mesh-niagara",.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance instance;CHECK(vkCreateInstance(&ici,NULL,&instance));uint32_t count=0;CHECK(vkEnumeratePhysicalDevices(instance,&count,NULL));
   VkPhysicalDevice *devices=calloc(count,sizeof(*devices));CHECK(vkEnumeratePhysicalDevices(instance,&count,devices));
   for(unsigned i=0;i<count;i++) {
      VkPhysicalDeviceSubgroupProperties subgroup={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
      VkPhysicalDeviceDriverProperties driver={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,.pNext=&subgroup};
      VkPhysicalDeviceProperties2 props={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&driver};vkGetPhysicalDeviceProperties2(devices[i],&props);
      bool soft=getenv("M2_SOFTWARE_REFERENCE")&&strstr(props.properties.deviceName,"llvmpipe");
      if(driver.driverID!=VK_DRIVER_ID_MESA_HONEYKRISP&&!soft)continue;
      phys=devices[i];printf("renderer=%s driver=%s info=%s\nsubgroup_size=%u\n",props.properties.deviceName,driver.driverName,driver.driverInfo,subgroup.subgroupSize);
      if(props.properties.limits.maxPushConstantsSize<216)return 2;
      break;
   }
   free(devices);if(!phys){fprintf(stderr,"no expected renderer\n");return 2;}
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan11Features f11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&mesh};
   VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&f11};
   VkPhysicalDeviceVulkan13Features f13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&f12};
   VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&f13};vkGetPhysicalDeviceFeatures2(phys,&features);
   if(!mesh.meshShader||!f11.storageBuffer16BitAccess||!f12.storageBuffer8BitAccess||!f12.runtimeDescriptorArray||
      !f12.shaderSampledImageArrayNonUniformIndexing||!f13.synchronization2||!f13.shaderDemoteToHelperInvocation){fprintf(stderr,"required features missing\n");return 2;}
   uint32_t qc=0;vkGetPhysicalDeviceQueueFamilyProperties(phys,&qc,NULL);VkQueueFamilyProperties *families=calloc(qc,sizeof(*families));vkGetPhysicalDeviceQueueFamilyProperties(phys,&qc,families);
   uint32_t family=0;while(family<qc&&!(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT))family++;free(families);if(family==qc)return 2;
   float priority=1;VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
   mesh=(VkPhysicalDeviceMeshShaderFeaturesEXT){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=VK_TRUE};
   f11=(VkPhysicalDeviceVulkan11Features){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&mesh,.storageBuffer16BitAccess=VK_TRUE};
   f12=(VkPhysicalDeviceVulkan12Features){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&f11,.storageBuffer8BitAccess=VK_TRUE,.runtimeDescriptorArray=VK_TRUE,.shaderSampledImageArrayNonUniformIndexing=VK_TRUE};
   f13=(VkPhysicalDeviceVulkan13Features){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&f12,.synchronization2=VK_TRUE,.shaderDemoteToHelperInvocation=VK_TRUE};
   const char *extension=VK_EXT_MESH_SHADER_EXTENSION_NAME;VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f13,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi,.enabledExtensionCount=1,.ppEnabledExtensionNames=&extension};
   CHECK(vkCreateDevice(phys,&dci,NULL,&dev));VkQueue queue;vkGetDeviceQueue(dev,family,0,&queue);PFN_vkCmdDrawMeshTasksEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");if(!draw)return 2;
   struct buffer q=buffer(Q_BYTES,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),pixels=buffer(W*H*8,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(pixels.map,0xa5,W*H*8);FILE *input=fopen(argv[3],"rb");if(!input||fread(q.map,1,Q_BYTES,input)!=Q_BYTES||fgetc(input)!=EOF||fclose(input))return 2;
   const VkFormat formats[2]={VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_A2B10G10R10_UNORM_PACK32};
   struct image images[3];VkAttachmentDescription atts[2];VkAttachmentReference refs[2];VkImageView views[2];
   for(unsigned i=0;i<2;i++) {
      VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(phys,formats[i],&fp);if(!(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))return 2;
      images[i]=image(formats[i],W,H,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);views[i]=images[i].view;
      atts[i]=(VkAttachmentDescription){.format=formats[i],.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
      refs[i]=(VkAttachmentReference){i,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   }
   images[2]=image(VK_FORMAT_R8G8B8A8_UNORM,1,1,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=2,.pColorAttachments=refs};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=2,.pAttachments=atts,.subpassCount=1,.pSubpasses=&sub};VkRenderPass pass;CHECK(vkCreateRenderPass(dev,&rpci,NULL,&pass));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=pass,.attachmentCount=2,.pAttachments=views,.width=W,.height=H,.layers=1};VkFramebuffer fb;CHECK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));
   const VkShaderStageFlags both=VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_FRAGMENT_BIT;
   VkDescriptorSetLayoutBinding bindings[8];for(unsigned i=0;i<7;i++)bindings[i]=(VkDescriptorSetLayoutBinding){.binding=i<6?i:7,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=both};
   bindings[7]=(VkDescriptorSetLayoutBinding){.binding=8,.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT};
   VkDescriptorSetLayoutBinding texture={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT};
   VkDescriptorSetLayout setlayouts[2];VkDescriptorSetLayoutCreateInfo slci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=8,.pBindings=bindings};CHECK(vkCreateDescriptorSetLayout(dev,&slci,NULL,&setlayouts[0]));slci.bindingCount=1;slci.pBindings=&texture;CHECK(vkCreateDescriptorSetLayout(dev,&slci,NULL,&setlayouts[1]));
   VkPushConstantRange push={both,0,216};VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=2,.pSetLayouts=setlayouts,.pushConstantRangeCount=1,.pPushConstantRanges=&push};VkPipelineLayout layout;CHECK(vkCreatePipelineLayout(dev,&plci,NULL,&layout));
   VkDescriptorPoolSize pool_sizes[]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,7},{VK_DESCRIPTOR_TYPE_SAMPLER,1},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1}};
   VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=2,.poolSizeCount=3,.pPoolSizes=pool_sizes};VkDescriptorPool dp;CHECK(vkCreateDescriptorPool(dev,&dpci,NULL,&dp));
   VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=2,.pSetLayouts=setlayouts};VkDescriptorSet sets[2];CHECK(vkAllocateDescriptorSets(dev,&dsai,sets));
   const unsigned offsets[]={4096,8192,12288,16384,20480,24576,28672};const unsigned sizes[]={320,768,384,1024,2048,64,1024};
   for(unsigned i=0;i<7;i++) {VkDescriptorBufferInfo bi={q.buffer,offsets[i],sizes[i]};VkWriteDescriptorSet wr={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[0],.dstBinding=i<6?i:7,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi};vkUpdateDescriptorSets(dev,1,&wr,0,NULL);}
   VkSamplerCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,.magFilter=VK_FILTER_NEAREST,.minFilter=VK_FILTER_NEAREST,.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,.addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};VkSampler sampler;CHECK(vkCreateSampler(dev,&sci,NULL,&sampler));
   VkDescriptorImageInfo di={.sampler=sampler};VkWriteDescriptorSet wr={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[0],.dstBinding=8,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER,.pImageInfo=&di};vkUpdateDescriptorSets(dev,1,&wr,0,NULL);
   di=(VkDescriptorImageInfo){.imageView=images[2].view,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};wr.dstSet=sets[1];wr.dstBinding=0;wr.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;vkUpdateDescriptorSets(dev,1,&wr,0,NULL);
   VkShaderModule ms=module(argv[1]),fs=module(argv[2]);
   VkPipelineShaderStageCreateInfo stages[]={{.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_MESH_BIT_EXT,.module=ms,.pName="main"},{.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"}};
   VkViewport viewport={0,0,W,H,0,1};VkRect2D scissor={{0,0},{W,H}};VkPipelineViewportStateCreateInfo vs={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&scissor};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
   VkPipelineMultisampleStateCreateInfo samples={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};VkPipelineColorBlendAttachmentState blend[2]={{.colorWriteMask=15},{.colorWriteMask=15}};
   VkPipelineColorBlendStateCreateInfo cb={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=2,.pAttachments=blend};
   VkGraphicsPipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=stages,.pViewportState=&vs,.pRasterizationState=&rs,.pMultisampleState=&samples,.pColorBlendState=&cb,.layout=layout,.renderPass=pass};VkPipeline pipeline;CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};VkCommandPool pool;CHECK(vkCreateCommandPool(dev,&cpci,NULL,&pool));
   VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};VkCommandBuffer cmd;CHECK(vkAllocateCommandBuffers(dev,&cai,&cmd));VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};CHECK(vkBeginCommandBuffer(cmd,&begin));
   VkImageMemoryBarrier2 ib={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_NONE,.dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,.dstAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.image=images[2].image,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,.imageMemoryBarrierCount=1,.pImageMemoryBarriers=&ib};vkCmdPipelineBarrier2(cmd,&dep);VkClearColorValue dummy={{1,1,1,1}};vkCmdClearColorImage(cmd,images[2].image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&dummy,1,&ib.subresourceRange);
   ib.srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT;ib.srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT;ib.dstStageMask=VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;ib.dstAccessMask=VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;ib.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;ib.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;vkCmdPipelineBarrier2(cmd,&dep);
   VkMemoryBarrier2 host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.srcAccessMask=VK_ACCESS_2_HOST_WRITE_BIT,.dstStageMask=VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT|VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,.dstAccessMask=VK_ACCESS_2_SHADER_STORAGE_READ_BIT};dep.imageMemoryBarrierCount=0;dep.memoryBarrierCount=1;dep.pMemoryBarriers=&host;vkCmdPipelineBarrier2(cmd,&dep);
   VkClearValue clear[2]={{{{0,0,0,0}}},{{{0,0,0,0}}}};VkRenderPassBeginInfo rpbegin={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=pass,.framebuffer=fb,.renderArea={{0,0},{W,H}},.clearValueCount=2,.pClearValues=clear};
   vkCmdBeginRenderPass(cmd,&rpbegin,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,2,sets,0,NULL);vkCmdPushConstants(cmd,layout,both,0,216,q.map);draw(cmd,1,1,1);vkCmdEndRenderPass(cmd);
   VkMemoryBarrier2 copybarrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,.srcAccessMask=VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,.dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,.dstAccessMask=VK_ACCESS_2_TRANSFER_READ_BIT};dep.pMemoryBarriers=&copybarrier;vkCmdPipelineBarrier2(cmd,&dep);
   for(unsigned i=0;i<2;i++){VkBufferImageCopy copy={.bufferOffset=i*W*H*4,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={W,H,1}};vkCmdCopyImageToBuffer(cmd,images[i].image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,pixels.buffer,1,&copy);}
   VkMemoryBarrier2 after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,.srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,.srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,.dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_HOST_READ_BIT};dep.pMemoryBarriers=&after;vkCmdPipelineBarrier2(cmd,&dep);
   CHECK(vkEndCommandBuffer(cmd));VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;CHECK(vkCreateFence(dev,&fci,NULL,&fence));VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};puts("SUBMIT_BEGIN");
   VkResult result=vkQueueSubmit(queue,1,&submit,fence);if(result!=VK_SUCCESS)retain(result,"submit");result=vkWaitForFences(dev,1,&fence,VK_TRUE,15000000000ull);if(result!=VK_SUCCESS)retain(result,"wait");puts("SUBMIT_OK");
   save(argv[4],"pixels.bin",pixels.map,W*H*8);save(argv[4],"Q.bin",q.map,Q_BYTES);puts("READBACK_OK Niagara L64 V64 P96 draws=1 attachments=2");
   vkDestroyFence(dev,fence,NULL);vkDestroyCommandPool(dev,pool,NULL);vkDestroyPipeline(dev,pipeline,NULL);vkDestroyShaderModule(dev,ms,NULL);vkDestroyShaderModule(dev,fs,NULL);vkDestroyDescriptorPool(dev,dp,NULL);vkDestroySampler(dev,sampler,NULL);vkDestroyPipelineLayout(dev,layout,NULL);for(unsigned i=0;i<2;i++)vkDestroyDescriptorSetLayout(dev,setlayouts[i],NULL);vkDestroyFramebuffer(dev,fb,NULL);vkDestroyRenderPass(dev,pass,NULL);
   for(unsigned i=0;i<3;i++){vkDestroyImageView(dev,images[i].view,NULL);vkDestroyImage(dev,images[i].image,NULL);vkFreeMemory(dev,images[i].memory,NULL);}struct buffer *buffers[]={&q,&pixels};for(unsigned i=0;i<2;i++){vkUnmapMemory(dev,buffers[i]->memory);vkDestroyBuffer(dev,buffers[i]->buffer,NULL);vkFreeMemory(dev,buffers[i]->memory,NULL);}vkDestroyDevice(dev,NULL);vkDestroyInstance(instance,NULL);return 0;
}
