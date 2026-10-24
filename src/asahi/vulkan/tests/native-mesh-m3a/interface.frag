/* SPDX-License-Identifier: MIT */
#version 460
#extension GL_EXT_mesh_shader : require
layout(location=0) in vec2 smooth_value;
layout(location=1) noperspective in float linear_value;
layout(location=2) perprimitiveEXT flat in vec2 primitive_value;
layout(location=3) perprimitiveEXT flat in int primitive_signed;
layout(location=4) perprimitiveEXT flat in uint primitive_unsigned;
layout(location=0) out vec4 result;
void main()
{
   bool valid=primitive_signed==-23 && primitive_unsigned==0x87654321u;
   result=vec4(smooth_value.x,linear_value,
               smooth_value.y+(valid ? primitive_value.x : 16.0),primitive_value.y);
}
