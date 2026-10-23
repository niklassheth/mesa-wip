/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Research-only persistent Vulkan carrier. Protocol is on stdout; diagnostics
 * are on stderr. No GPU resource is destroyed on an uncertain completion.
 */
#define _GNU_SOURCE
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <signal.h>
#include <vulkan/vulkan.h>
static bool live_backing,host_shim;
static volatile sig_atomic_t fault_requested;
static void stop_signal(int sig) { (void)sig;fault_requested=1; }
static void fatal(const char *why) {
   fprintf(stderr,"HARNESS_FATAL: %s\n",why);
   printf("ERROR %s\n",why);fflush(NULL);
   if((live_backing||fault_requested)&&!host_shim)for(;;)pause();
   exit(90);
}
#include "oracle.h"
#define MAX_ARMS 768
#define MAX_LANES 8
struct blob { unsigned char *data; size_t size; };
struct shape { unsigned vertices, primitives, threads, topology, planes, local_B, allocation_B, control42; };
struct arm {
   struct shape shape;
   unsigned id, grid[3], family, uniform_nonce;
   struct blob mesh, object;
   struct oracle oracle;
   VkPipeline pipe;
   VkCommandBuffer cmd[MAX_LANES];
};
struct lane {
   VkImage image; VkImageView view; VkBuffer readback;
   VkDescriptorSet ds; VkFence fence;
   unsigned char *q,*pixels,*want;
   uint64_t qva,sequence,submit_ns,complete_ns,ordinal;
   uint32_t cookie; unsigned index,code_checks;
   bool busy,verified; struct arm *arm;
   void *record;
   unsigned timed_commands;
   uint64_t timed_stream[2], timed_offset[2], timestamp_value[2];
   uint32_t timed_handle[2];
   bool timestamp_bracket;
};
static struct arm arms[MAX_ARMS];
static struct lane lanes[MAX_LANES];
static unsigned arm_count,lane_count;
static bool started,uncertain;
static uint64_t now_ns(void) {
   struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}
static struct blob read_blob(const char *path) {
   struct blob b={0};FILE *f=fopen(path,"rb");
   if(!f||fseek(f,0,SEEK_END))fatal("blob open");
   long n=ftell(f);if(n<=0||n>1048576||fseek(f,0,SEEK_SET))fatal("blob length");
   b.size=n;b.data=malloc(n);
   if(!b.data||fread(b.data,1,n,f)!=(size_t)n||fgetc(f)!=EOF||fclose(f))fatal("blob read");
   return b;
}
static void save(const char *directory,const char *leaf,const void *p,size_t n) {
   char path[4096];if(snprintf(path,sizeof path,"%s/%s",directory,leaf)>=sizeof path)fatal("dump path");
   FILE *f=fopen(path,"wb");if(!f||fwrite(p,1,n,f)!=n||fclose(f))fatal("dump write");
}
#include "renderer.inc"
#include "bridge.inc"
#include "shape.inc"

static void load_arm(unsigned id,const char *mesh,const char *object,const char *oracle,
                     unsigned x,unsigned y,unsigned z,unsigned family,unsigned nonce) {
   if(started||id!=arm_count||id>=MAX_ARMS||family||nonce||!x||!y||!z)fatal("LOAD contract");
   struct arm *a=&arms[id];a->id=id;a->grid[0]=x;a->grid[1]=y;a->grid[2]=z;a->family=family;a->uniform_nonce=nonce;
   a->mesh=read_blob(mesh);a->object=read_blob(object);a->oracle=oracle_load(oracle);
   a->shape=load_shape(mesh);
   setenv("HK_APPLE9_NATIVE_MESH_COMPILER_BINARY",mesh,1);
   setenv("HK_APPLE9_NATIVE_OBJECT_COMPILER_BINARY",object,1);
   /* Runtime-count variants share a program, but each recorded draw still
    * has its own frozen launch/state and exact oracle. */
   for(unsigned j=0;j<arm_count;j++) {
      struct arm *old=&arms[j];
      if(!memcmp(&a->shape,&old->shape,sizeof a->shape)&&a->mesh.size==old->mesh.size&&
         a->object.size==old->object.size&&!memcmp(a->mesh.data,old->mesh.data,a->mesh.size)&&
         !memcmp(a->object.data,old->object.data,a->object.size)){a->pipe=old->pipe;break;}
   }
   if(!a->pipe)a->pipe=make_pipeline(layout);
   for(unsigned i=0;i<lane_count;i++) {
      uint64_t generation=mapping_generation;
      a->cmd[i]=record(a,&lanes[i]);
      bridge_register(a,&lanes[i],generation);
   }
   arm_count++;
}
static void submit(unsigned id,unsigned slot,uint64_t sequence,uint32_t cookie) {
   if(!started||uncertain||fault_requested||id>=arm_count||slot>=lane_count||!cookie)fatal("SUBMIT stopped or invalid");
   struct lane *l=&lanes[slot];if(l->busy)fatal("lane reused before verification");
   l->arm=&arms[id];l->cookie=cookie;l->sequence=sequence;l->busy=true;l->verified=false;l->code_checks=0;
   l->timed_commands=0;l->timestamp_bracket=false;
   oracle_materialize(&l->arm->oracle,l->q,0,cookie,l->qva);
   /* Poison image readback too: a missing copy must not pass from old pixels. */
   memset(l->pixels,0xcd,PIXEL_BYTES);
   oracle_materialize(&l->arm->oracle,l->want,1,cookie,l->qva);
   CHECK(vkResetFences(dev,1,&l->fence));
   VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,NULL,0,NULL,NULL,1,&l->arm->cmd[slot]};
   live_backing=true;l->submit_ns=now_ns();VkResult r=vkQueueSubmit(queue,1,&si,l->fence);
   if(r!=VK_SUCCESS)uncertain=true;
   printf("SUBMITTED %u %"PRIu64" %d %"PRIu64"\n",slot,sequence,r,l->submit_ns);
}
static void poll_lane(unsigned slot,uint64_t timeout) {
   if(slot>=lane_count||!lanes[slot].busy)fatal("POLL contract");
   struct lane *l=&lanes[slot];VkResult r=vkWaitForFences(dev,1,&l->fence,VK_TRUE,timeout);
   if(r==VK_TIMEOUT){printf("PENDING %u %"PRIu64"\n",slot,l->sequence);return;}
   l->complete_ns=now_ns();
   if(r!=VK_SUCCESS){uncertain=true;printf("FAULT %u %"PRIu64" %d\n",slot,l->sequence,r);return;}
   if(l->code_checks!=1)fatal("missing or duplicate native code check");
   unsigned differences,first;
   unsigned status=oracle_check(&l->arm->oracle,l->q,l->pixels,l->want,&differences,&first);
   uint32_t qhash=crc32(0,l->q,Q_BYTES),phash=crc32(0,l->pixels,PIXEL_BYTES);
   l->verified=true;
   printf("RESULT %u %"PRIu64" %u %u %u %08x %08x %"PRIu64" %"PRIu64" %"PRIu64,
          slot,l->sequence,status,differences,first,qhash,phash,l->submit_ns,l->complete_ns,l->ordinal);
   uint32_t lo[4]={0},hi[4]={0},sample_hash=0;
   struct oracle *o=&l->arm->oracle;
   if(o->sample_only&&o->groups) {
      for(unsigned i=0;i<4;i++)lo[i]=UINT32_MAX;
      for(unsigned g=0;g<o->groups;g++)for(unsigned lane=0;lane<16;lane++) {
         uint32_t values[4];memcpy(values,l->q+0x10000+g*2048+lane*64+20,16);
         sample_hash=crc32(sample_hash,(void*)values,sizeof values);
         for(unsigned i=0;i<4;i++){if(values[i]<lo[i])lo[i]=values[i];if(values[i]>hi[i])hi[i]=values[i];}
      }
   }
   printf(" %u %08x",!memcmp(l->pixels,o->pixels,PIXEL_BYTES),sample_hash);
   for(unsigned i=0;i<4;i++)printf(" %u %u",lo[i],hi[i]);
   putchar('\n');
}
static void release_lane(unsigned slot) {
   if(slot>=lane_count||!lanes[slot].busy||!lanes[slot].verified)fatal("RELEASE contract");
   lanes[slot].busy=false;
   bool busy=false;for(unsigned i=0;i<lane_count;i++)busy|=lanes[i].busy;
   if(!busy&&!uncertain&&!fault_requested)live_backing=false;
   printf("RELEASED %u\n",slot);
}
int main(int argc,char **argv) {
   if(argc!=4)fatal("usage: runner LANES mesh.spv fragment.spv");
   host_shim=getenv("ASAHI_HOST_TRACE_DIR")!=NULL;
   signal(SIGUSR1,stop_signal);
   signal(SIGPIPE,SIG_IGN);
   prctl(PR_SET_NAME,"mesh-harness-v2",0,0,0);
   lane_count=strtoul(argv[1],NULL,0);if(!lane_count||lane_count>MAX_LANES)fatal("lane count");
   shader_path=argv[2];setenv("MULTI_FRAGMENT",argv[3],1);setenv("MULTI_BLEND","1",1);
   device_init();layout_init();
   for(unsigned i=0;i<lane_count;i++){lanes[i].index=i;lane_init(&lanes[i]);}
   setvbuf(stdout,NULL,_IOLBF,0);printf("READY %u %u",(unsigned)getpid(),lane_count);
   /* Identity belongs to the READY protocol, not a racing stderr reader. */
   const char *identity[]={ready_device_name,ready_driver_name,ready_driver_info};
   for(unsigned i=0;i<3;i++) {
      putchar(' ');
      for(const unsigned char *p=(const void *)identity[i];*p;p++)printf("%02x",*p);
   }
   printf(" %u\n",ready_driver_version);
   char line[16384],op[32],mesh[4096],object[4096],oracle[4096],path[4096];
   unsigned id,slot,x,y,z,family,nonce;uint64_t seq,timeout;uint32_t cookie;
   while(fgets(line,sizeof line,stdin)) {
      if(!strchr(line,'\n'))fatal("protocol line too long");
      if(sscanf(line,"%31s",op)!=1)fatal("empty command");
      if(!strcmp(op,"LOAD")) {
         if(sscanf(line,"LOAD %u %4095s %4095s %4095s %u %u %u %u %u",&id,mesh,object,oracle,&x,&y,&z,&family,&nonce)!=9)fatal("LOAD syntax");
         load_arm(id,mesh,object,oracle,x,y,z,family,nonce);printf("LOADED %u\n",id);
      } else if(!strcmp(op,"START")) {
         if(started||!arm_count)fatal("START contract");started=true;
         printf("STARTED %u %u %"PRIu64"\n",arm_count,lane_count,now_ns());
      } else if(!strcmp(op,"SUBMIT")) {
         if(sscanf(line,"SUBMIT %u %u %"SCNu64" %"SCNu32,&id,&slot,&seq,&cookie)!=4)fatal("SUBMIT syntax");
         submit(id,slot,seq,cookie);
      } else if(!strcmp(op,"POLL")) {
         if(sscanf(line,"POLL %u %"SCNu64,&slot,&timeout)!=2)fatal("POLL syntax");poll_lane(slot,timeout);
      } else if(!strcmp(op,"DUMP")) {
         if(sscanf(line,"DUMP %u %4095s",&slot,path)!=2||slot>=lane_count||!lanes[slot].busy)fatal("DUMP syntax");
         bridge_dump(&lanes[slot],path);printf("DUMPED %u\n",slot);
      } else if(!strcmp(op,"RELEASE")) {
         if(sscanf(line,"RELEASE %u",&slot)!=1)fatal("RELEASE syntax");release_lane(slot);
      } else if(!strcmp(op,"STOP")) { uncertain=true;printf("STOPPED\n");
      } else if(!strcmp(op,"EXIT")) {
         bool busy=false;for(unsigned i=0;i<lane_count;i++)busy|=lanes[i].busy;
         if(uncertain||busy)fatal("EXIT while backing required");
         CHECK(vkDeviceWaitIdle(dev));puts("BYE");return 0;
      } else fatal("unknown command");
   }
   /* Owner pipe loss cannot release memory still reachable by the GPU. */
   bool busy=false;for(unsigned i=0;i<lane_count;i++)busy|=lanes[i].busy;
   if(uncertain||busy||fault_requested){fprintf(stderr,"BACKING_RETAINED pid=%u\n",getpid());for(;;)pause();}
   return 0;
}
