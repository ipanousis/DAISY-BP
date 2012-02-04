
__kernel void gradient8(__global float * srcArray,
                        __global float * dstArray,
                        const    int     arrWidth,
                        const    int     arrHeight,
                        const    int     arrHalo){

  // get x get y and store 'em
  const int id = (get_global_id(0) / arrWidth + arrHalo) * (arrHalo * 2 + arrWidth) + 
                                     arrHalo + get_global_id(0) % arrWidth;
  dstArray[0] = id;
  /*

  srcArray += id;
  dstArray += id;

  float4 n;
  n.x = srcArray[-1];
  n.y = srcArray[-arrHalo * 2 - arrWidth];
  n.z = srcArray[1];
  n.w = srcArray[arrHalo * 2 + arrWidth];

  float8 gradients;
  const float8 angles = (float8)(0.0f, M_PI / 8, M_PI / 4, 3 * (M_PI / 8), M_PI / 2,
                                  5 * (M_PI / 8), 6 * (M_PI / 8), 7 * (M_PI / 8));
  //gradients  = fmax(cos(angles) * (n.x - n.z) * 0.5,0.0);
  //gradients += fmax(sin(angles) * ( n.y - n.w) * 0.5,0.0);

  gradients.s0 = fmax(cos(angles.s0) * ( n.x - n.z) * 0.5 + 
                      sin(angles.s0) * ( n.y - n.w) * 0.5, 0.0);

  gradients.s1 = fmax(cos(angles.s1) * (n.x - n.z) * 0.5 + 
                 sin(angles.s1) * ( n.y - n.w) * 0.5, 0.0);
  gradients.s2 = fmax(cos(angles.s2) * (n.x - n.z) * 0.5 + 
                 sin(angles.s2) * ( n.y - n.w) * 0.5, 0.0);
  gradients.s3 = fmax(cos(angles.s3) * (n.x - n.z) * 0.5 + 
                 sin(angles.s3) * ( n.y - n.w) * 0.5, 0.0);
  gradients.s4 = fmax(cos(angles.s4) * (n.x - n.z) * 0.5 + 
                 sin(angles.s4) * ( n.y - n.w) * 0.5, 0.0);

  gradients.s5 = fmax(cos(angles.s5) * (n.x - n.z) * 0.5 + 
                 sin(angles.s5) * ( n.y - n.w) * 0.5, 0.0);
  gradients.s6 = fmax(cos(angles.s6) * (n.x - n.z) * 0.5 + 
                 sin(angles.s6) * ( n.y - n.w) * 0.5, 0.0);

  gradients.s7 = fmax(cos(angles.s7) * (n.x - n.z) * 0.5 + 
                 sin(angles.s7) * ( n.y - n.w) * 0.5, 0.0);

  const int offset = (arrWidth + arrHalo * 2) * (arrHeight + arrHalo * 2);  
  dstArray[0]        = gradients.s0;
  dstArray[offset]   = gradients.s1;
  dstArray[2*offset] = gradients.s2;
  dstArray[3*offset] = gradients.s3;
  dstArray[4*offset] = gradients.s4;
  dstArray[5*offset] = gradients.s5;
  dstArray[6*offset] = gradients.s6;
  dstArray[7*offset] = gradients.s7;
  */
}
