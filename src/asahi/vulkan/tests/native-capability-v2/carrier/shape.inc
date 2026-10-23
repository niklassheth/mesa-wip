/* SPDX-License-Identifier: MIT */
static struct shape load_shape(const char *binary) {
   char path[4096];
   if(snprintf(path,sizeof path,"%s.shape",binary)>=sizeof path)fatal("shape path");
   FILE *f=fopen(path,"r");struct shape s={0};
   if(!f||fscanf(f,"%u %u %u %u %u %u %u %u",&s.vertices,&s.primitives,&s.threads,&s.topology,&s.planes,&s.local_B,&s.allocation_B,&s.control42)!=8)fatal("shape read");
   int c;while((c=fgetc(f))!=EOF)if(c!='\n'&&c!=' '&&c!='\t')fatal("shape tail");
   if(fclose(f))fatal("shape close");
   if(!s.vertices||s.vertices>256||!s.primitives||s.primitives>512||s.threads<16||s.threads>256||(s.threads&(s.threads-1))||s.threads<s.vertices||s.control42>1)fatal("shape bounds");
   if(s.topology!=0&&s.topology!=1&&s.topology!=6)fatal("shape topology");
   unsigned k=s.topology==6?3:s.topology==1?2:1;
   unsigned planes=s.topology==0?5:4;
   unsigned local=((s.primitives*k+15)&~15)+16+((s.vertices+3)&~3)*4*planes;
   unsigned alloc=128;while(alloc<local)alloc*=2;
   if(s.planes!=planes||s.local_B!=local||s.allocation_B!=alloc)fatal("shape layout");
   return s;
}
