/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Standalone adapter for the independent production DATA grammar/source oracle.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
struct formation_fixture {
 const char *name; unsigned vertices, primitives;
 uint32_t positions[12], indices[3], state[8], viewport[6], ids[3];
};
struct machine { struct { uint8_t *data; } region[4]; };
#include "apple9-mesh-data.h"
static uint8_t *load(const char *path, size_t *size) {
 FILE *f=fopen(path,"rb"); CHECK(f); CHECK(!fseek(f,0,SEEK_END));
 long n=ftell(f); CHECK(n>=0 && n<=0x4000000); *size=n;
 rewind(f); uint8_t *b=malloc(n?n:1); CHECK(b);
 CHECK(fread(b,1,n,f)==(size_t)n); CHECK(!fclose(f)); return b;
}
static bool zero(const uint8_t *b,size_t begin,size_t end) {
 for(size_t i=begin;i<end;i++) { if(b[i]) return false; }
 return true;
}
int main(int argc,char **argv) {
 CHECK(argc==7); size_t n; uint8_t *src=load(argv[1],&n); CHECK(n==128);
 struct formation_fixture f={.name="H1_device",.vertices=3,.primitives=1};
 unsigned at=0;
 #define WORDS(field) for(unsigned i=0;i<sizeof(f.field)/4;i++,at+=4) f.field[i]=read32(src+at)
 WORDS(positions); WORDS(indices); WORDS(state); WORDS(viewport); WORDS(ids);
 #undef WORDS
 uint32_t xyz[3][3]; bool drop; CHECK(source_geometry(&f,xyz,&drop) && !drop);
 struct machine m={0}; size_t sizes[4];
 for(unsigned i=0;i<4;i++) m.region[i].data=load(argv[2+i],&sizes[i]);
 CHECK(sizes[0]>=88 && sizes[1]>=12 && sizes[2]>=32 && sizes[3]>=48);
 struct decoded_geometry g; unsigned length=0,choices=0;
 for(unsigned s=44;s<=88;s+=4) {
  if(decode_geometry(m.region[0].data,s,&g)) {length=s;choices++;}
 }
 CHECK(choices==1); CHECK(decode_geometry(m.region[0].data,length,&g));
 CHECK(geometry_matches(&f,&g,xyz)); CHECK(zero(m.region[0].data,length,sizes[0]));
 CHECK(references_match(&f,&m));
 CHECK(zero(m.region[1].data,12,sizes[1])); CHECK(zero(m.region[2].data,32,sizes[2]));
 CHECK(zero(m.region[3].data,48,sizes[3]));
 check_geometry(&f,m.region[0].data,length,xyz);
 uint8_t coverage[4096]; raster(xyz,coverage); uint8_t *pixels=load(argv[6],&n); CHECK(n==16384);
 unsigned covered=0; for(unsigned i=0;i<4096;i++) {
  covered+=coverage[i]; CHECK(pixels[4*i]==(coverage[i]?255:0));
  CHECK(pixels[4*i+1]==0 && pixels[4*i+2]==0 && pixels[4*i+3]==(coverage[i]?255:0));
 }
 printf("{\"passed\":true,\"geometry_bytes\":%u,\"software_samples\":%u,\"source_pixels_exact\":true,\"references_exact\":true,\"tails_zero\":true,\"alternate_raw_and_mutations_checked\":true,\"positions\":[[%u,%u,%u],[%u,%u,%u],[%u,%u,%u]]}\n",length,covered,xyz[0][0],xyz[0][1],xyz[0][2],xyz[1][0],xyz[1][1],xyz[1][2],xyz[2][0],xyz[2][1],xyz[2][2]);
 return 0;
}
