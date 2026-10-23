/* SPDX-License-Identifier: MIT
 * Research-only, owned DATA edits before submission. Field locations come
 * from retained state measurements and the existing open-source genxml.
 * No program instruction is patched or taken from Metal output.
 */
static void patch_ppp(struct draw_record *r) {
   const struct shape *s=&r->arm->shape;
   static const unsigned offsets[]={0,0x40,0x5c,0x70,0x8c,0xc0,0xa0,0xac,0xf0};
   static const unsigned sizes[]={20,28,20,28,20,40,12,8,8};
   /* Apple9's PPP layout and context-relative addresses differ from G13.
    * These are the same DATA fields constructed by agx_apple9.c, inside
    * this frozen carrier's 256-byte record. No instruction bytes are read. */
   uint64_t base=0x1000000000ull+r->words[15];
   unsigned char *record=owned(r->fd,r->vm,base,256);
   uint32_t select,count,control,face[2];
   memcpy(&select,record+0x60,4);memcpy(&count,record+0x6c,4);
   memcpy(&control,record+0x90,4);memcpy(face,record+0x94,8);
   if((select&(1u<<18))||count!=4||(face[0]>>28)||(face[1]>>28))fatal("baseline PPP outputs/topology");
   if(s->topology==0)select|=1u<<18;
   if(s->topology!=6)control|=1u<<26;
   unsigned object_type=s->topology==6?0:s->topology==1?1:4;
   for(unsigned i=0;i<2;i++)face[i]|=object_type<<28;
   memcpy(record+0x60,&select,4);memcpy(record+0x6c,&s->planes,4);
   memcpy(record+0x90,&control,4);memcpy(record+0x94,face,8);
   for(unsigned j=0;j<9;j++) {
      unsigned at=14+j*2;
      if(r->words[at]!=(sizes[j]/4)<<8||r->words[at+1]!=r->words[15]+offsets[j])fatal("PPP carrier layout");
      r->ppp[j]=base+offsets[j];r->ppp_size[j]=sizes[j];
      memcpy(r->ppp_bytes[j],record+offsets[j],sizes[j]);
   }
}
static void patch_state(struct draw_record *r,unsigned char *mesh) {
   const struct shape *s=&r->arm->shape;
   static const unsigned char allocation[]={0x77,0,0x2a,0x41,0x80,2,0,0};
   if(memcmp(mesh+86,allocation,sizeof allocation))fatal("baseline allocation");
   unsigned packed=(s->allocation_B<<2)|0x80;
   for(unsigned i=0;i<3;i++)mesh[90+i]=packed>>(i*8);
   r->words[1]=s->planes|((s->planes+1)<<8);
   r->words[10]=(s->primitives-1)|(((s->primitives+15)/16-1)<<16);
   r->words[11]=0x400800|s->threads;
   r->words[13]=1+((s->vertices+3)/4)*s->planes;
   r->words[32]=0x70000000|(s->topology<<8);
   patch_ppp(r);
   memcpy(owned(r->fd,r->vm,r->stream,sizeof r->words),r->words,sizeof r->words);
}
