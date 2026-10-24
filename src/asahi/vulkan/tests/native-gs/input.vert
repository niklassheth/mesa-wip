#version 450
void main() {
    uint i = uint(gl_VertexIndex) % 3u;
    gl_Position = vec4(i == 1u ? 3.0 : -1.0, i == 2u ? 3.0 : -1.0, 0.5, 1.0);
}
