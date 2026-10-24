#version 460
// SPDX-License-Identifier: MIT
layout(set=0, binding=0) uniform U { vec4 rect; float weight; float slices; } u;
layout(location=0) out float result;
void main() { result = u.weight; }
