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
}
