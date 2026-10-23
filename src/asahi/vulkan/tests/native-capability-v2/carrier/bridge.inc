/* SPDX-License-Identifier: MIT
 * In-process owned-mapping tracker and code identity check. No routine files,
 * allocator capture, shader decoding, or kernel/private memory access.
 */
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "asahi_drm.h"
#include "assets.h"
static pthread_mutex_t bridge_lock=PTHREAD_MUTEX_INITIALIZER;
#define MAX_BO 16384
#define MAX_BIND 32768
static uint64_t mapping_generation;
static struct bo {int fd;uint32_t handle,vm;uint64_t size,offset,generation;void *cpu;size_t mapped;} bos[MAX_BO];
static unsigned nb;
static struct binding {int fd;uint32_t vm,handle;uint64_t addr,offset,size;bool live;} binds[MAX_BIND];
static unsigned nbinds;
static struct {int fd;uint32_t queue,vm;uint64_t ordinal;} queues[64];static unsigned nq;
struct draw_record {
   struct arm *arm;struct lane *lane;int fd;uint32_t vm;
   uint64_t stream,ma,oa,mc,oc,mt,ot;uint32_t nonce;
   unsigned char mesh_launch[256],object_launch[256];
   uint64_t mesh_args[10],object_args[10];
   uint32_t words[40];struct drm_asahi_cmd_render render;
   uint64_t ppp[9]; unsigned ppp_size[9]; unsigned char ppp_bytes[9][1024];
};
static struct draw_record records[MAX_ARMS*MAX_LANES];static unsigned nr;
static struct bo *bo(int fd,uint32_t h) {
   for(unsigned i=0;i<nb;i++)if(bos[i].fd==fd&&bos[i].handle==h)return &bos[i];return NULL;
}
#include "bridge-helpers.inc"
#include "patch-state.inc"
static bool stream_shape(const uint32_t *w) {
   return w[0]==0x40000d2c&&w[1]==0x504&&w[2]==0x10&&!w[3]&&!w[4]&&
      w[5]==0x1000000&&w[7]==0x1000000&&w[9]==1&&!w[10]&&
      w[11]==0x400810&&!w[12]&&w[13]==5&&w[32]==0x70000600&&
      w[33]==16&&w[34]==1&&w[35]==1&&w[36]==16&&w[37]==1&&w[38]==1&&w[39]==0xc0000000;
}
static bool registered(int fd,uint32_t vm,uint64_t stream) {
   for(unsigned i=0;i<nr;i++)if(records[i].fd==fd&&records[i].vm==vm&&records[i].stream==stream)return true;
   return false;
}
/* Text is immutable after pipeline creation; identical mesh/object bodies
 * may be shared by the driver. Launches and argument tables belong to one
 * recorded draw and must never alias another draw's storage. */
struct program_range {uint64_t address;size_t size;unsigned shared;};
static void program_ranges(const struct draw_record *r,struct program_range out[7]) {
   out[0]=(struct program_range){r->mc,r->arm->mesh.size,1+r->arm->family};
   out[1]=(struct program_range){r->oc,r->arm->object.size,3};
   out[2]=(struct program_range){r->ma,256,0};
   out[3]=(struct program_range){r->oa,256,0};
   out[4]=(struct program_range){r->mt,80,0};
   out[5]=(struct program_range){r->ot,80,0};
   out[6]=(struct program_range){r->stream,160,0};
}
static bool overlaps(struct program_range a,struct program_range b) {
   return a.address<b.address+b.size&&b.address<a.address+a.size;
}
static void check_program_overlap(const struct draw_record *r) {
   struct program_range current[7];program_ranges(r,current);
   for(unsigned i=0;i<7;i++)for(unsigned j=0;j<i;j++) {
      /* The carrier deliberately gives both stages the same argument table. */
      if(i==5&&j==4&&current[i].address==current[j].address)continue;
      if(overlaps(current[i],current[j]))fatal("program buffers overlap within draw");
   }
   for(unsigned k=0;k+1<nr;k++) {
      const struct draw_record *other=&records[k];
      if(other->fd!=r->fd||other->vm!=r->vm)continue;
      struct program_range previous[7];program_ranges(other,previous);
      for(unsigned i=0;i<7;i++)for(unsigned j=0;j<7;j++) {
         if(!overlaps(current[i],previous[j]))continue;
         if(current[i].shared&&current[i].shared==previous[j].shared&&
            current[i].address==previous[j].address&&current[i].size==previous[j].size)continue;
         fprintf(stderr,"PROGRAM_OVERLAP arm=%u lane=%u range=%u other_arm=%u other_lane=%u range=%u\n",
                 r->arm->id,r->lane->index,i,other->arm->id,other->lane->index,j);
         fatal("program buffers overlap across draws");
      }
   }
}
static void register_stream(struct arm *a,struct lane *l,int fd,uint32_t vm,uint64_t stream,const uint32_t *w) {
   if(nr==MAX_ARMS*MAX_LANES)fatal("record limit");
   struct draw_record *r=&records[nr++];
   *r=(struct draw_record){.arm=a,.lane=l,.fd=fd,.vm=vm,.stream=stream};
   memcpy(r->words,w,sizeof r->words);
   r->ma=(1ull<<40)+64ull*w[8];r->oa=(1ull<<40)+64ull*w[6];
   unsigned char *mesh=owned(fd,vm,r->ma,256),*object=owned(fd,vm,r->oa,256);
   check_launch(mesh,frozen_mesh_launch);check_launch(object,frozen_object_launch);
   r->mc=target(mesh,6);r->oc=target(object,2);
   if(memcmp(owned(fd,vm,r->mc,a->mesh.size),a->mesh.data,a->mesh.size)||
      memcmp(owned(fd,vm,r->oc,a->object.size),a->object.data,a->object.size))fatal("upfront uploaded program identity");
   r->mt=literal(mesh,2)|((uint64_t)literal(mesh+8,3)<<32);
   r->ot=literal(object,2)|((uint64_t)literal(object+8,3)<<32);
   check_program_overlap(r);
   memcpy(r->mesh_args,owned(fd,vm,r->mt,80),80);memcpy(r->object_args,owned(fd,vm,r->ot,80),80);
   for(unsigned stage=0;stage<2;stage++) {
      uint64_t *args=stage?r->object_args:r->mesh_args;
      if(args[3]||args[4]||args[5]||args[6]||args[7]!=64||args[8]!=l->qva||!args[9])fatal("argument ABI");
   }
   patch_state(r,mesh);
   memcpy(r->mesh_launch,mesh,256);memcpy(r->object_launch,object,256);
}
/* Scan only owned, mapped buffers at startup, immediately after recording.
 * Already registered streams are excluded. No scan is performed per draw.
 * Full program and launch checks reject a shader-cache alias before any GPU
 * submission. Only our independently compiled bodies are uploaded; startup
 * patches the owned state and allocation once, then freezes their identity.
 */
static void bridge_register(struct arm *a,struct lane *l,uint64_t after_generation) {
   pthread_mutex_lock(&bridge_lock);unsigned found=0;
   /* A command pool normally maps fresh BOs. Prefer just those; a reused
    * mapping gets one fallback scan, still only during startup. */
   for(unsigned pass=0;pass<2&&!found;pass++) {
   for(unsigned i=0;i<nbinds;i++) {
      struct binding *b=&binds[i];struct bo *o=bo(b->fd,b->handle);
      if(!b->live||!o||!o->cpu||b->offset>=o->mapped)continue;
      if((o->generation>after_generation)!=(pass==0))continue;
      size_t size=o->mapped-b->offset;if(size>b->size)size=b->size;
      const unsigned char *cpu=(unsigned char*)o->cpu+b->offset;
      /* VDM words are naturally 4-byte aligned. */
      for(size_t off=0;off+160<=size;off+=4) {
         uint32_t first;memcpy(&first,cpu+off,4);if(first!=0x40000d2c)continue;
         uint32_t w[40];memcpy(w,cpu+off,160);
         if(!stream_shape(w)||registered(b->fd,b->vm,b->addr+off))continue;
         uint64_t ma=(1ull<<40)+64ull*w[8];const unsigned char *ml=owned(b->fd,b->vm,ma,256);
         uint64_t table=literal(ml,2)|((uint64_t)literal(ml+8,3)<<32),args[10];
         memcpy(args,owned(b->fd,b->vm,table,80),80);
         if(args[8]!=l->qva)continue;
         register_stream(a,l,b->fd,b->vm,b->addr+off,w);found++;
      }
   }
   }
   if(found!=1)fatal("upfront stream uniqueness");
   pthread_mutex_unlock(&bridge_lock);
}
static void inspect_submit(struct drm_asahi_submit *s,int fd) {
   if(fault_requested)fatal("kernel fault stop before ioctl");
   unsigned queue_index=UINT32_MAX;
   for(unsigned i=0;i<nq;i++)if(queues[i].fd==fd&&queues[i].queue==s->queue_id)queue_index=i;
   if(queue_index==UINT32_MAX)fatal("submit queue VM");
   uint32_t vm=queues[queue_index].vm;
   for(size_t off=0;off<s->cmdbuf_size;) {
      if(s->cmdbuf_size-off<sizeof(struct drm_asahi_cmd_header))fatal("short header");
      struct drm_asahi_cmd_header h;memcpy(&h,(void*)((uintptr_t)s->cmdbuf+off),sizeof h);
      if(h.size>s->cmdbuf_size-off-sizeof h)fatal("short command");
      if(h.cmd_type==DRM_ASAHI_CMD_RENDER) {
         if(h.size!=sizeof(struct drm_asahi_cmd_render))fatal("render size");
         struct drm_asahi_cmd_render render;memcpy(&render,(void*)((uintptr_t)s->cmdbuf+off+sizeof h),sizeof render);
         struct draw_record *r=NULL;
         for(unsigned i=0;i<nr;i++)if(records[i].fd==fd&&records[i].vm==vm&&records[i].stream==render.vdm_ctrl_stream_base) {
            if(r)fatal("ambiguous render");r=&records[i];
         }
         if(!r)fatal("unregistered render");
         struct lane *l=r->lane;struct arm *a=r->arm;
         if(!l->busy||l->arm!=a||l->code_checks++)fatal("unowned in-flight command");
         if(render.width_px!=64||render.height_px!=64||render.isp_bgobjdepth||render.isp_zls_pixels!=0x1f803f||render.zls_ctrl)fatal("render ABI");
         if(memcmp(owned(fd,vm,r->stream,160),r->words,160)||
            memcmp(owned(fd,vm,r->ma,256),r->mesh_launch,256)||
            memcmp(owned(fd,vm,r->oa,256),r->object_launch,256)||
            memcmp(owned(fd,vm,r->mc,a->mesh.size),a->mesh.data,a->mesh.size)||
            memcmp(owned(fd,vm,r->oc,a->object.size),a->object.data,a->object.size))fatal("live program/launch identity");
         for(unsigned j=0;j<9;j++)if(memcmp(owned(fd,vm,r->ppp[j],r->ppp_size[j]),r->ppp_bytes[j],r->ppp_size[j]))fatal("live PPP identity");
         if(memcmp(owned(fd,vm,r->mt,80),r->mesh_args,80)||memcmp(owned(fd,vm,r->ot,80),r->object_args,80))fatal("live argument identity");
         if(a->uniform_nonce) {
            r->mesh_args[3]=l->cookie;memcpy(owned(fd,vm,r->mt,80),r->mesh_args,80);
            if(r->mt==r->ot)memcpy(r->object_args,r->mesh_args,80);
         }
         if(owned(fd,vm,l->qva,Q_BYTES)!=l->q)fatal("Q CPU mapping identity");
         r->render=render;l->record=r;l->ordinal=queues[queue_index].ordinal++;
      }
      off+=sizeof h+h.size;
   }
}
static void mapped(void *p,size_t n,int fd,uint64_t off) {
   if(p==MAP_FAILED)return;pthread_mutex_lock(&bridge_lock);
   for(unsigned i=0;i<nb;i++)if(bos[i].fd==fd&&bos[i].offset==off&&off){bos[i].cpu=p;bos[i].mapped=n;bos[i].generation=++mapping_generation;}
   pthread_mutex_unlock(&bridge_lock);
}
void *mmap(void *a,size_t n,int prot,int flags,int fd,off_t off) {
   void *(*next)(void*,size_t,int,int,int,off_t)=dlsym(RTLD_NEXT,"mmap");void *p=next(a,n,prot,flags,fd,off);mapped(p,n,fd,off);return p;
}
void *mmap64(void *a,size_t n,int prot,int flags,int fd,off64_t off) {
   void *(*next)(void*,size_t,int,int,int,off64_t)=dlsym(RTLD_NEXT,"mmap64");void *p=next(a,n,prot,flags,fd,off);mapped(p,n,fd,off);return p;
}
int munmap(void *p,size_t n) {
   int(*next)(void*,size_t)=dlsym(RTLD_NEXT,"munmap");pthread_mutex_lock(&bridge_lock);
   for(unsigned i=0;i<nb;i++)if(bos[i].cpu&&(uintptr_t)p<(uintptr_t)bos[i].cpu+bos[i].mapped&&(uintptr_t)bos[i].cpu<(uintptr_t)p+n){bos[i].cpu=NULL;bos[i].mapped=0;}
   pthread_mutex_unlock(&bridge_lock);return next(p,n);
}
int ioctl(int fd,unsigned long req,...) {
   va_list ap;va_start(ap,req);void *arg=va_arg(ap,void*);va_end(ap);
   int(*next)(int,unsigned long,...)=dlsym(RTLD_NEXT,"ioctl");
   if(req==DRM_IOCTL_ASAHI_SUBMIT){pthread_mutex_lock(&bridge_lock);inspect_submit(arg,fd);pthread_mutex_unlock(&bridge_lock);}
   int rc=next(fd,req,arg),saved=errno;pthread_mutex_lock(&bridge_lock);
   if(!rc) {
      if(req==DRM_IOCTL_ASAHI_GEM_CREATE) {
         struct drm_asahi_gem_create *s=arg;struct bo *b=bo(fd,s->handle);
         if(!b){if(nb==MAX_BO)fatal("BO limit");b=&bos[nb++];}
         *b=(struct bo){.fd=fd,.handle=s->handle,.vm=s->vm_id,.size=s->size};
      } else if(req==DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET) {
         struct drm_asahi_gem_mmap_offset *s=arg;struct bo *b=bo(fd,s->handle);if(b)b->offset=s->offset;
      } else if(req==DRM_IOCTL_ASAHI_QUEUE_CREATE) {
         struct drm_asahi_queue_create *s=arg;if(nq==64)fatal("queue limit");
         queues[nq].fd=fd;queues[nq].queue=s->queue_id;queues[nq].vm=s->vm_id;queues[nq++].ordinal=0;
      } else if(req==DRM_IOCTL_ASAHI_VM_BIND) {
         struct drm_asahi_vm_bind *s=arg;
         for(unsigned j=0;j<s->num_binds;j++) {
            struct drm_asahi_gem_bind_op *b=(void*)(uintptr_t)(s->userptr+j*s->stride);
            for(unsigned i=0;i<nbinds;i++)if(binds[i].fd==fd&&binds[i].vm==s->vm_id&&binds[i].addr==b->addr)binds[i].live=false;
            if(!(b->flags&DRM_ASAHI_BIND_UNBIND)) {
               unsigned at=0;while(at<nbinds&&binds[at].live)at++;
               if(at==MAX_BIND)fatal("binding limit");if(at==nbinds)nbinds++;
               binds[at]=(struct binding){fd,s->vm_id,b->handle,b->addr,b->offset,b->range,true};
            }
         }
      }
   }
   pthread_mutex_unlock(&bridge_lock);errno=saved;return rc;
}
static void bridge_dump(struct lane *l,const char *directory) {
   struct draw_record *r=l->record;if(!r)fatal("no submit record for dump");
   for(unsigned j=0;j<9;j++){char name[32];snprintf(name,sizeof name,"ppp-%u.bin",j);save(directory,name,r->ppp_bytes[j],r->ppp_size[j]);}
   save(directory,"shape.bin",&r->arm->shape,sizeof r->arm->shape);
   save(directory,"q.bin",l->q,Q_BYTES);save(directory,"rgba.bin",l->pixels,PIXEL_BYTES);
   save(directory,"expected-q.bin",l->want,Q_BYTES);save(directory,"expected-rgba.bin",l->arm->oracle.pixels,PIXEL_BYTES);
   save(directory,"cross-stream.bin",r->words,sizeof r->words);save(directory,"cross-render.bin",&r->render,sizeof r->render);
   save(directory,"cross-mesh-launch-after.bin",owned(r->fd,r->vm,r->ma,256),256);
   save(directory,"cross-object-launch-after.bin",owned(r->fd,r->vm,r->oa,256),256);
   save(directory,"cross-mesh-code-after.bin",owned(r->fd,r->vm,r->mc,l->arm->mesh.size),l->arm->mesh.size);
   save(directory,"cross-object-code-after.bin",owned(r->fd,r->vm,r->oc,l->arm->object.size),l->arm->object.size);
   char text[2048];
   snprintf(text,sizeof text,"{\"family\":\"%s\",\"pid\":%u,\"vm\":%u,\"q\":\"0x%llx\",\"mesh_code\":\"0x%llx\",\"object_code\":\"0x%llx\",\"mesh_launch\":\"0x%llx\",\"object_launch\":\"0x%llx\",\"stream\":\"0x%llx\",\"depth_clear\":%u,\"depth_dimensions\":%u}\n",
      l->arm->family?"direct":"frozen",getpid(),r->vm,(unsigned long long)l->qva,(unsigned long long)r->mc,(unsigned long long)r->oc,(unsigned long long)r->ma,(unsigned long long)r->oa,(unsigned long long)r->stream,r->render.isp_bgobjdepth,r->render.isp_zls_pixels);
   save(directory,"CROSS.json",text,strlen(text));
   snprintf(text,sizeof text,"{\"pid\":%u,\"sequence\":%llu,\"ordinal\":%llu,\"stream\":\"0x%llx\",\"gpu_error\":%s,\"runner_barrier\":true,\"renders\":1}\n",getpid(),(unsigned long long)l->sequence,(unsigned long long)l->ordinal,(unsigned long long)r->stream,l->verified?"false":"true");
   save(directory,"READY.json",text,strlen(text));
   snprintf(text,sizeof text,"Q_DEVICE address=%#llx bytes=%u\n",(unsigned long long)l->qva,Q_BYTES);save(directory,"log.txt",text,strlen(text));
   char sub[4096];if(snprintf(sub,sizeof sub,"%s/submit-000",directory)>=sizeof sub)fatal("dump path");
   if(mkdir(sub,0755))fatal("dump submit directory");
   save(sub,"stream.bin",r->words,sizeof r->words);
   snprintf(text,sizeof text,"{\"pid\":%u,\"vm\":%u,\"bindings\":[{\"address\":\"0x%llx\",\"captured\":160,\"file\":\"stream.bin\"}],\"renders\":[{\"stream\":\"0x%llx\"}]}\n",getpid(),r->vm,(unsigned long long)r->stream,(unsigned long long)r->stream);
   save(sub,"maps.json",text,strlen(text));
}
