/*
Preprocessor Variables
fltWidth = 7
arrHeight = <height without halo>
arrWidth  = <width without halo>
HALO_WIDTH = 3
*/

__kernel void convolve_x(__global   float * srcArray,
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

    float4 v = 0;
    v.x = fltArray[0] * srcArray[0];
    v.y = fltArray[1] * srcArray[1];
    v.z = fltArray[2] * srcArray[2];
    v.w = fltArray[3] * srcArray[3];

    v.x = v.x + v.y + v.z + v.w;
    v.y = fltArray[4] * srcArray[4];
    v.z = fltArray[5] * srcArray[5];
    v.w = fltArray[6] * srcArray[6];
  
    v.x = v.x + v.y + v.z + v.w;

    if(dstI / fullWidth == fltWidth / 2){
      dstArray[dstI] = v.x;
      dstArray[dstI - fullWidth] = v.x;
      dstArray[dstI - fullWidth*2] = v.x;
      dstArray[dstI - fullWidth*3] = v.x;
    }
    else if(dstI / fullWidth == fltWidth / 2 + arrHeight - 1){
      dstArray[dstI] = v.x;
      dstArray[dstI + fullWidth] = v.x;
      dstArray[dstI + fullWidth*2] = v.x;
      dstArray[dstI + fullWidth*3] = v.x;
    }
    else dstArray[dstI] = v.x;

}

__kernel void convolve_y(__global   float * srcArray,
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
      dstArray[dstI] = v.x;
      dstArray[dstI - 1] = v.x;
      dstArray[dstI - 2] = v.x;
      dstArray[dstI - 3] = v.x;
    }
    else if(dstI % fullWidth == fltWidth / 2 + arrWidth - 1){
      dstArray[dstI] = v.x;
      dstArray[dstI + 1] = v.x;
      dstArray[dstI + 2] = v.x;
      dstArray[dstI + 3] = v.x;
    }
    else dstArray[dstI] = v.x;
}
