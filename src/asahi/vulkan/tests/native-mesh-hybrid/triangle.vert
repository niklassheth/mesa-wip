#version 460
layout(location=0) flat out vec4 color;
void main() {
    const vec2 positions[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
    color = vec4(1);
}
