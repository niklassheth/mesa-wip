#version 460
#ifdef MESH
#extension GL_EXT_mesh_shader : require
layout(location=0) perprimitiveEXT in vec4 color;
#else
layout(location=0) flat in vec4 color;
#endif
layout(location=0) out vec4 result;
void main() { result = color; }
