/* SPDX-License-Identifier: MIT */
#pragma once
#include <zlib.h>
#define Q_BYTES 0x80000u
#define PIXEL_BYTES 16384u
#ifndef ORACLE_CONSTANTS_ONLY
struct patch { uint32_t area, offset, kind, x, a, b; };
struct oracle {
   unsigned char *initial, *expected, *pixels, *mask;
   struct patch *patches;
   uint32_t patch_count, sample_only, groups;
};
static void fatal(const char *why);
static void gzexact(gzFile f, void *p, size_t n) {
   if (gzread(f,p,n)!=(int)n) fatal("oracle truncated");
}
static struct oracle oracle_load(const char *path) {
   struct oracle o={0}; uint32_t h[8]; gzFile f=gzopen(path,"rb");
   if (!f) fatal("oracle open"); gzexact(f,h,sizeof h);
   if (memcmp(h,"HV2ORC01",8)||h[2]!=Q_BYTES||h[3]!=PIXEL_BYTES||h[4]>8192||h[5]>1||h[6]>32||h[7]) fatal("oracle header");
   o.patch_count=h[4];o.sample_only=h[5];o.groups=h[6];
   o.initial=malloc(Q_BYTES);o.expected=malloc(Q_BYTES);o.pixels=malloc(PIXEL_BYTES);o.mask=malloc(Q_BYTES);
   o.patches=calloc(o.patch_count,sizeof *o.patches);
   if(!o.initial||!o.expected||!o.pixels||!o.mask||!o.patches)fatal("oracle allocation");
   gzexact(f,o.initial,Q_BYTES);gzexact(f,o.expected,Q_BYTES);gzexact(f,o.pixels,PIXEL_BYTES);gzexact(f,o.mask,Q_BYTES);
   gzexact(f,o.patches,o.patch_count*sizeof *o.patches);
   if(gzgetc(f)!=-1||gzclose(f)!=Z_OK)fatal("oracle trailing data or CRC");
   for(unsigned i=0;i<o.patch_count;i++) {
      struct patch *p=&o.patches[i];
      if(p->area>1||p->kind>1||p->offset>Q_BYTES-(p->kind?8:4))fatal("oracle patch bounds");
   }
   return o;
}
static void oracle_materialize(struct oracle *o, void *dst, unsigned area, uint32_t cookie, uint64_t qva) {
   memcpy(dst,area?o->expected:o->initial,Q_BYTES);
   for(unsigned i=0;i<o->patch_count;i++) {
      struct patch *p=&o->patches[i]; if(p->area!=area)continue;
      if(p->kind){uint64_t v=qva+p->x;memcpy((char*)dst+p->offset,&v,8);}
      else {uint32_t v=p->a*(cookie^p->x)+p->b;memcpy((char*)dst+p->offset,&v,4);}
   }
}
/* 0 exact, 1 pixel mismatch with exact Q, 2 Q mismatch, 3 only declared
 * r5 sample words differ. Status 3 is still a Q failure, never qualification. */
static unsigned oracle_check(struct oracle *o,const unsigned char *q,const void *pixels,const unsigned char *want,
                             unsigned *differences,unsigned *first) {
   *differences=0;*first=UINT32_MAX;bool samples=true;
   for(unsigned i=0;i<Q_BYTES;i++) if(((q[i]^want[i])&o->mask[i])) {
      if(!*differences)*first=i;(*differences)++;
      unsigned r=i-0x10000, child=r/2048,lane=(r%2048)/64,byte=r%64;
      if(!o->sample_only||i<0x10000||child>=o->groups||lane>=16||byte<20||byte>=36)samples=false;
   }
   return *differences ? (samples?3:2) : (memcmp(pixels,o->pixels,PIXEL_BYTES)?1:0);
}

#endif
