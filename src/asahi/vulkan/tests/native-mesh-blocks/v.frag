#version 460
// SPDX-License-Identifier: MIT
layout(push_constant) uniform P { vec4 rect; float weight; } pc;
layout(location=0) out float result;
void main() { result = pc.weight; }
