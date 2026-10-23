/* SPDX-License-Identifier: MIT
 * Ordinary Vulkan draws qualify the exact R32 additive raster oracle.
 * One owner; any uncertain completion retains all backing until reset.
 */
#define _GNU_SOURCE
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <vulkan/vulkan.h>
static bool live_backing,host_shim;
static volatile sig_atomic_t fault_requested;
static void stop_signal(int sig){(void)sig;fault_requested=1;}
static void fatal(const char *why){fprintf(stderr,"CALIBRATION_FATAL %s\n",why);fflush(NULL);if(live_backing&&!host_shim)for(;;)pause();exit(90);}
#define ORACLE_CONSTANTS_ONLY
#include "oracle.h"
struct shape { unsigned primitives; };
struct arm {struct shape shape;VkPipeline pipe;unsigned grid[3];};
struct lane {VkImage image;VkImageView view;VkBuffer readback;VkDescriptorSet ds;VkFence fence;unsigned char *q,*pixels,*want;uint64_t qva;};
#include "renderer.inc"
int main(int argc,char **argv){
   if(argc!=2)fatal("usage: calibrate KIT");
   signal(SIGUSR1,stop_signal);host_shim=getenv("ASAHI_HOST_TRACE_DIR")!=NULL;
   prctl(PR_SET_NAME,"mesh-calibrate",0,0,0);setvbuf(stdout,NULL,_IOLBF,0);
   char path[4096];snprintf(path,sizeof path,"%s/shaders/depth-color.frag.spv",argv[1]);setenv("MULTI_FRAGMENT",path,1);setenv("MULTI_BLEND","1",1);
   mesh_mode=0;device_init();layout_init();struct lane l={0};lane_init(&l);
   unsigned counts[]={0,1,2,16,128,256,512};unsigned topologies[]={6,1,0};
   for(unsigned ti=0;ti<3;ti++){
      unsigned topo=topologies[ti];reference_topology=topo;
      snprintf(path,sizeof path,"%s/shaders/control-%u.vert.spv",argv[1],topo);shader_path=path;
      struct arm a={.pipe=make_pipeline(layout)};
      for(unsigned pi=0;pi<sizeof(counts)/sizeof(counts[0]);pi++){
         if(fault_requested)fatal("stopped before submission");
         a.shape.primitives=counts[pi];VkCommandBuffer cmd=record(&a,&l);
         memset(l.pixels,0xcd,PIXEL_BYTES);CHECK(vkResetFences(dev,1,&l.fence));
         VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,NULL,0,NULL,NULL,1,&cmd};
         live_backing=true;CHECK(vkQueueSubmit(queue,1,&si,l.fence));
         CHECK(vkWaitForFences(dev,1,&l.fence,VK_TRUE,10000000000ull));
         unsigned differences=0;float *pixels=(void*)l.pixels;
         for(unsigned y=0;y<64;y++)for(unsigned x=0;x<64;x++){
            bool covered=topo==6||(y==32&&(topo==1||x==32));float want=covered?counts[pi]:0;
            if(memcmp(&pixels[y*64+x],&want,4))differences++;
         }
         live_backing=false;printf("CONTROL %u %u %u\n",topo,counts[pi],differences);
         if(differences&&!host_shim)fatal("raster oracle failed");
      }
   }
   CHECK(vkDeviceWaitIdle(dev));puts(host_shim?"HOST_ONLY":"QUALIFIED 21");return 0;
}
