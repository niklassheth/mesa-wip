#version 460
// SPDX-License-Identifier: MIT
// Ordinary draw: a rectangle from push constants as two triangles.
layout(push_constant) uniform P { vec4 rect; float weight; } pc;
void main() {
    const vec2 c[6] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1), vec2(0,1), vec2(1,0), vec2(1,1));
    vec2 p = mix(pc.rect.xy, pc.rect.zw, c[gl_VertexIndex]);
    gl_Position = vec4(p / 32.0 - 1.0, 0, 1);
}
