#version 450
layout(triangles) in;
#ifndef MAX_VERTICES
#define MAX_VERTICES 4
#endif
layout(triangle_strip, max_vertices=MAX_VERTICES) out;
layout(set=0,binding=0) uniform Counts { uint emitted; uint end_mask; } counts;
void main() {
    if (counts.emitted > 0u) {
        gl_Position = gl_in[0].gl_Position * vec4(.75,.75,1,1);
        EmitVertex();
    }
    if ((counts.end_mask & 1u) != 0u) EndPrimitive();
    if (counts.emitted > 1u) {
        gl_Position = mix(gl_in[0].gl_Position,gl_in[1].gl_Position,.5) * vec4(.75,.75,1,1);
        EmitVertex();
    }
    if ((counts.end_mask & 2u) != 0u) EndPrimitive();
    if (counts.emitted > 2u) {
        gl_Position = mix(gl_in[0].gl_Position,gl_in[2].gl_Position,.5) * vec4(.75,.75,1,1);
        EmitVertex();
    }
    if ((counts.end_mask & 4u) != 0u) EndPrimitive();
#if MAX_VERTICES > 3
    if (counts.emitted > 3u) {
        gl_Position = mix(gl_in[1].gl_Position,gl_in[2].gl_Position,.5) * vec4(.75,.75,1,1);
        EmitVertex();
    }
    if ((counts.end_mask & 8u) != 0u) EndPrimitive();
#endif
}
