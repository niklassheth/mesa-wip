#version 450
// Independently authored ordinary-draw calibration, no mesh/native operations.
void main() {
    const vec2 positions[2]=vec2[](vec2(-1.0+1.0/64.0,-1.0/64.0),vec2(1.0+1.0/64.0,-1.0/64.0));
    gl_Position=vec4(positions[gl_VertexIndex%2],0,1);
    gl_PointSize=1.0;
}
