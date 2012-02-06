/*
Preprocessor Variables
fltWidth = 7
arrHeight = <height without halo>
arrWidth  = <width without halo>
HALO_WIDTH = 3
*/

__kernel void convolve_7x(__global   float * srcArray,
                          __global   float * dstArray,
                          __constant float * fltArray,
                          const      int     arrWidth,
                          const      int     arrHeight,
                          const      int     arrHalo)
{

    const int fullWidth = arrWidth + arrHalo * 2;
    const int dstI = (get_global_id(0) / arrWidth + arrHalo) * (arrHalo * 2 + arrWidth) + 
                                         arrHalo + get_global_id(0) % arrWidth;

    srcArray += dstI - 3;

    float4 v = vload4(0, srcArray);
    float4 u = vload4(0, fltArray);
    float4 w = v * u;

    v = vload4(0, srcArray+3);
    u = vload4(0, fltArray+3);
    w.x -= w.w;
    w += v * u;
    w.x = w.x + w.y + w.z + w.w;

    if(dstI / fullWidth == arrHalo){
      dstArray[dstI] = w.x;
      dstArray[dstI - fullWidth] = w.x;
      dstArray[dstI - fullWidth*2] = w.x;
      dstArray[dstI - fullWidth*3] = w.x;
    }
    else if(dstI / fullWidth == arrHalo + arrHeight - 1){
      dstArray[dstI] = w.x;
      dstArray[dstI + fullWidth] = w.x;
      dstArray[dstI + fullWidth*2] = w.x;
      dstArray[dstI + fullWidth*3] = w.x;
    }
    else dstArray[dstI] = w.x;

}

__kernel void convolve_7y(__global   float * srcArray,
                          __global   float * dstArray,
                          __constant float * fltArray,
                          const      int     arrWidth,
                          const      int     arrHeight,
                          const      int     arrHalo)
{

    const int fullWidth = arrWidth + arrHalo * 2;
    const int dstI = (get_global_id(0) / arrWidth + arrHalo) * (arrHalo * 2 + arrWidth) + 
                                         arrHalo + get_global_id(0) % arrWidth;

    srcArray += dstI - fullWidth * 3;

    float4 v = 0;
    v.x = fltArray[0] * srcArray[0];
    v.y = fltArray[1] * srcArray[fullWidth];
    v.z = fltArray[2] * srcArray[2*fullWidth];
    v.w = fltArray[3] * srcArray[3*fullWidth];

    v.x = v.x + v.y + v.z + v.w;
    v.y = fltArray[4] * srcArray[4*fullWidth];
    v.z = fltArray[5] * srcArray[5*fullWidth];
    v.w = fltArray[6] * srcArray[6*fullWidth];
  
    v.x = v.x + v.y + v.z + v.w;

    if(dstI % fullWidth == arrHalo){
      v = v.x;
      vstore4(v, 0, dstArray+dstI-3);
      //dstArray[dstI] = v.x;
      //dstArray[dstI - 1] = v.x;
      //dstArray[dstI - 2] = v.x;
      //dstArray[dstI - 3] = v.x;
    }
    else if(dstI % fullWidth == arrHalo + arrWidth - 1){
      v = v.x;
      vstore4(v, 0, dstArray+dstI);
      //dstArray[dstI] = v.x;
      //dstArray[dstI + 1] = v.x;
      //dstArray[dstI + 2] = v.x;
      //dstArray[dstI + 3] = v.x;
    }
    else dstArray[dstI] = v.x;
}

/*
Preprocessor Variables
fltWidth = 7
arrHeight = <height without halo>
arrWidth  = <width without halo>
HALO_WIDTH = 3
*/
/*
__kernel void convolve_13x(__global   float * srcArray,
                         __global   float * dstArray,
                         __constant float * fltArray,
                         const      int     fltWidth,
                         const      int     arrWidth,
                         const      int     arrHeight)
{

    const int fullWidth = arrWidth + fltWidth - 1;
    const int dstI = get_global_id(0) + (fullWidth + 1) * (fltWidth / 2) +
                      (get_global_id(0) / arrWidth) * (fltWidth - 1);

    srcArray += dstI - fltWidth / 2;

    float4 v = vload4(0, srcArray);
    float4 u = vload4(0, fltArray);
    float4 w = v * u;

    v = vload4(0, srcArray+3);
    u = vload4(0, fltArray+3);
    w.x -= w.w;
    w += v * u;
    w.x = w.x + w.y + w.z + w.w;

    if(dstI / fullWidth == fltWidth / 2){
      dstArray[dstI] = w.x;
      dstArray[dstI - fullWidth] = w.x;
      dstArray[dstI - fullWidth*2] = w.x;
      dstArray[dstI - fullWidth*3] = w.x;
    }
    else if(dstI / fullWidth == fltWidth / 2 + arrHeight - 1){
      dstArray[dstI] = w.x;
      dstArray[dstI + fullWidth] = w.x;
      dstArray[dstI + fullWidth*2] = w.x;
      dstArray[dstI + fullWidth*3] = w.x;
    }
    else dstArray[dstI] = w.x;

}

__kernel void convolve_13y(__global   float * srcArray,
                         __global   float * dstArray,
                         __constant float * fltArray,
                         const      int     fltWidth,
                         const      int     arrWidth,
                         const      int     arrHeight)
{

    const int fullWidth = arrWidth + fltWidth - 1;
    const int dstI = get_global_id(0) + (fullWidth + 1) * (fltWidth / 2) +
                      (get_global_id(0) / arrWidth) * (fltWidth - 1);

    srcArray += dstI - fullWidth * (fltWidth / 2);

    float4 v = 0;
    v.x = fltArray[0] * srcArray[0];
    v.y = fltArray[1] * srcArray[fullWidth];
    v.z = fltArray[2] * srcArray[2*fullWidth];
    v.w = fltArray[3] * srcArray[3*fullWidth];

    v.x = v.x + v.y + v.z + v.w;
    v.y = fltArray[4] * srcArray[4*fullWidth];
    v.z = fltArray[5] * srcArray[5*fullWidth];
    v.w = fltArray[6] * srcArray[6*fullWidth];
  
    v.x = v.x + v.y + v.z + v.w;

    if(dstI % fullWidth == fltWidth / 2){
      v = v.x;
      vstore4(v, 0, dstArray+dstI-3);
      //dstArray[dstI] = v.x;
      //dstArray[dstI - 1] = v.x;
      //dstArray[dstI - 2] = v.x;
      //dstArray[dstI - 3] = v.x;
    }
    else if(dstI % fullWidth == fltWidth / 2 + arrWidth - 1){
      v = v.x;
      vstore4(v, 0, dstArray+dstI);
      //dstArray[dstI] = v.x;
      //dstArray[dstI + 1] = v.x;
      //dstArray[dstI + 2] = v.x;
      //dstArray[dstI + 3] = v.x;
    }
    else dstArray[dstI] = v.x;
}*/


__kernel void gradient8(__global float * srcArray,
                        __global float * dstArray,
                        const    int     arrWidth,
                        const    int     arrHeight,
                        const    int     arrHalo){

  // get x get y and store 'em
  const int id = (get_global_id(0) / arrWidth + arrHalo) * (arrHalo * 2 + arrWidth) + 
                                     arrHalo + get_global_id(0) % arrWidth;
  dstArray += id;
  srcArray += id;

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
  
}
