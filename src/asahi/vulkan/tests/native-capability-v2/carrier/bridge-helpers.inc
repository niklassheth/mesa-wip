/* SPDX-License-Identifier: MIT
 * Owned address/launch helpers retained from the sealed r5 observer. */
static void *owned(int fd,uint32_t vm,uint64_t addr,size_t n) {
 void *answer=NULL;
 if(!n || addr > UINT64_MAX-n)fatal("range overflow");
 for(unsigned i=0;i<nbinds;i++) {
  struct binding *b=&binds[i];
  if(!b->live||b->fd!=fd||b->vm!=vm||addr<b->addr||
     addr-b->addr>b->size||n>b->size-(addr-b->addr))continue;
  struct bo *o=bo(fd,b->handle);
  if(b->offset>UINT64_MAX-(addr-b->addr))fatal("offset overflow");
  uint64_t off=b->offset+(addr-b->addr);
  if(!o||!o->cpu||off>o->mapped||n>o->mapped-off)fatal("unmapped owned range");
  if(answer)fatal("ambiguous owned range");
  answer=(char*)o->cpu+off;
 }
 if(!answer){fprintf(stderr,"MISSING_OWNED fd=%d vm=%u va=%#llx bytes=%zu\n",fd,vm,(unsigned long long)addr,n);fatal("missing owned range");}
 return answer;
}
static uint32_t literal(const unsigned char *p,unsigned reg) {
 if(p[0]!=((reg&15)<<4|12)||p[2]!=((reg>>4)<<6|2)||!(p[1]&128))
  fatal("not our literal");
 return (p[1]&127)|((p[4]&30)<<6)|((p[5]&12)<<9)|
        ((uint32_t)p[6]<<13)|((uint32_t)(p[7]&15)<<21)|((uint32_t)(p[3]&254)<<24);
}
static uint64_t target(const unsigned char *p,unsigned mode) {
 if(p[94]!=0x77||p[95]!=1||p[102]!=mode||p[103])fatal("not our target");
 uint64_t t=0;for(unsigned i=0;i<6;i++)t|=(uint64_t)p[96+i]<<(8*i);
 if(t<0x2a||(t-0x2a)%2)fatal("target alignment");
 return (t-0x2a)/2;
}
static void check_launch(const unsigned char *p,const unsigned char *known) {
 for(unsigned i=16;i<256;i++) {
  if(i>=96&&i<102)continue;
  if(p[i]!=known[i])fatal("unexpected frozen launch byte");
 }
}
