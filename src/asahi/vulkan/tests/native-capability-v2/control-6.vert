#version 450
// Independently authored ordinary-draw calibration, no mesh/native operations.
void main() {
    const vec2 positions[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
    gl_Position=vec4(positions[gl_VertexIndex%3],0,1);
    gl_PointSize=1.0;
}
