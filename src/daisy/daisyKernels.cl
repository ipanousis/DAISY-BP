__kernel void convolve_7x(__global   float * massArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{
    const int r = get_global_id(0) / pddWidth;
    const int c = get_global_id(0) % pddWidth;
    __global float * srcArray = massArray + r * pddWidth + c;
    __global float * dstArray = massArray + pddWidth * pddHeight + r * pddWidth + c;
    const int localSize = get_local_size(0);
    const int l = c % localSize;
    __local float l_srcArray[64 + 6];
    l_srcArray[l + 3] = srcArray[0]; // center value
    if(l == 0){
      l_srcArray[0] = (c > 0 ? srcArray[-3]:l_srcArray[3]);
      l_srcArray[1] = (c > 0 ? srcArray[-2]:l_srcArray[3]);
      l_srcArray[2] = (c > 0 ? srcArray[-1]:l_srcArray[3]);
    }
    else if(l == localSize-1){
      l_srcArray[localSize+3] = (c < pddWidth-1 ? srcArray[1]:l_srcArray[l+3]);
      l_srcArray[localSize+4] = (c < pddWidth-1 ? srcArray[2]:l_srcArray[l+3]);
      l_srcArray[localSize+5] = (c < pddWidth-1 ? srcArray[3]:l_srcArray[l+3]);
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 v = vload4(0, l_srcArray + l);
    float4 u = vload4(0, fltArray);
    float4 w = v * u;
    v = vload4(0, l_srcArray + l + 3);
    u = vload4(0, fltArray+3);
    w.x -= w.w;
    w += v * u;
    dstArray[0] = w.x + w.y + w.z + w.w;
}
__kernel void convolve_7y(__global   float * massArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{
    const int r = get_global_id(0) / pddHeight;
    const int c = get_global_id(0) % pddHeight;
    __global float * srcArray = massArray + pddWidth * pddHeight + c * pddWidth + r;
    __global float * dstArray = massArray + pddWidth * pddHeight * 8 + c * pddWidth + r;
    const int localSize = get_local_size(0);
    const int l = c % localSize;
    __local float l_srcArray[64 + 6];
    l_srcArray[l + 3] = srcArray[0]; // center value
    if(l == 0){
      l_srcArray[0] = (c > 0 ? srcArray[-pddWidth*3]:l_srcArray[3]);
      l_srcArray[1] = (c > 0 ? srcArray[-pddWidth*2]:l_srcArray[3]);
      l_srcArray[2] = (c > 0 ? srcArray[-pddWidth]:l_srcArray[3]);
    }
    else if(l == localSize-1){
      l_srcArray[localSize+3] = (c < pddHeight-1 ? srcArray[pddWidth]:l_srcArray[l+3]);
      l_srcArray[localSize+4] = (c < pddHeight-1 ? srcArray[pddWidth*2]:l_srcArray[l+3]);
      l_srcArray[localSize+5] = (c < pddHeight-1 ? srcArray[pddWidth*3]:l_srcArray[l+3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 v = vload4(0, l_srcArray + l);
    float4 u = vload4(0, fltArray);
    float4 w = v * u;
    v = vload4(0, l_srcArray + l + 3);
    u = vload4(0, fltArray+3);
    w.x -= w.w;
    w += v * u;
    dstArray[0] = w.x + w.y + w.z + w.w;
}

__kernel void gradient_8all(__global float * massArray,
                            const    int     pddWidth,
                            const    int     pddHeight){
  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;
  __global float * srcArray = massArray + pddWidth * pddHeight * 8 + r * pddWidth + c;
  __global float * dstArray = massArray + r * pddWidth + c;
  float4 n;
  n.x = (c > 0           ? srcArray[-1]:srcArray[0]);
  n.y = (r > 0           ? srcArray[-pddWidth]:srcArray[0]);
  n.z = (c < pddWidth-1  ? srcArray[1]:srcArray[0]);
  n.w = (r < pddHeight-1 ? srcArray[pddWidth]:srcArray[0]);
  float8 gradients;
  const float8 angles = (float8)(0.0f, M_PI / 8, M_PI / 4, 3 * (M_PI / 8), M_PI / 2,
                                  5 * (M_PI / 8), 6 * (M_PI / 8), 7 * (M_PI / 8));
  n.x = (n.x-n.z) * 0.5;
  n.y = (n.y-n.w) * 0.5;
  gradients.s0 = fmax(cos(angles.s0) * n.x + 
                      sin(angles.s0) * n.y, 0.0);
  gradients.s1 = fmax(cos(angles.s1) * n.x + 
                      sin(angles.s1) * n.y, 0.0);
  gradients.s2 = fmax(cos(angles.s2) * n.x + 
                      sin(angles.s2) * n.y, 0.0);
  gradients.s3 = fmax(cos(angles.s3) * n.x + 
                      sin(angles.s3) * n.y, 0.0);
  gradients.s4 = fmax(cos(angles.s4) * n.x + 
                      sin(angles.s4) * n.y, 0.0);
  gradients.s5 = fmax(cos(angles.s5) * n.x + 
                      sin(angles.s5) * n.y, 0.0);
  gradients.s6 = fmax(cos(angles.s6) * n.x + 
                      sin(angles.s6) * n.y, 0.0);
  gradients.s7 = fmax(cos(angles.s7) * n.x + 
                      sin(angles.s7) * n.y, 0.0);
  const int offset = pddWidth * pddHeight;
  dstArray[0]        = gradients.s0;
  dstArray[offset]   = gradients.s1;
  dstArray[2*offset] = gradients.s2;
  dstArray[3*offset] = gradients.s3;
  dstArray[4*offset] = gradients.s4;
  dstArray[5*offset] = gradients.s5;
  dstArray[6*offset] = gradients.s6;
  dstArray[7*offset] = gradients.s7;
}
__kernel void convolve_11x(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;
  __global float * srcArray = massArray + r * pddWidth + c; // section A
  __global float * dstArray = massArray + r * pddWidth + c + pddWidth * pddHeight * 8; // section B
  const int localSize = get_local_size(0);
  const int l = c % localSize;
  __local float l_srcArray[64 + 10];
  fltArray = fltArray + 7;
  for(int o = 0; o < 8; o++){
    l_srcArray[l + 5] = srcArray[0]; // center value
    if(l < 5){
      l_srcArray[l] = (c > 4 ? srcArray[l-5]:l_srcArray[5]);
    }
    else if(l > localSize-6){
      l_srcArray[l + 10] = (c < pddWidth-5 ? srcArray[l - localSize + 6]:l_srcArray[localSize+4]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 v, f;
    float4 s = 0;
    v = vload4(0, l_srcArray + l);
    f = vload4(0, fltArray);
    s += v * f;
    v = vload4(0, l_srcArray + l + 4);
    f = vload4(0, fltArray + 4);
    s += v * f;
    v.xyz = vload3(0, l_srcArray + l + 8);
    f.xyz = vload3(0, fltArray + 8);
    s.xyz += v.xyz * f.xyz;
    dstArray[0] = s.x + s.y + s.z + s.w;
    srcArray += pddWidth * pddHeight;
    dstArray += pddWidth * pddHeight;
  }
}
__kernel void convolve_11y(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{

    const int r = get_global_id(0) / pddHeight;
    const int c = get_global_id(0) % pddHeight;

    __global float * srcArray = massArray + c * pddWidth + r + pddWidth * pddHeight * 8; // section B
    __global float * dstArray = massArray + c * pddWidth + r; // section A

    const int localSize = get_local_size(0);
    const int l = c % localSize;

    __local float l_srcArray[64 + 10];

    fltArray += 7;

    for(int o = 0; o < 8; o++){

      l_srcArray[l + 5] = srcArray[0]; // center value
      if(l < 5)
        l_srcArray[l] = (c > 4 ? srcArray[(l-5) * pddWidth]:l_srcArray[5]);
      else if(l > localSize-6)
        l_srcArray[l+10] = (c < pddHeight-5 ? srcArray[(l-localSize+6) * pddWidth]:l_srcArray[localSize+4]);
      
      barrier(CLK_LOCAL_MEM_FENCE);

      float4 v, f;
      float4 s = 0;

      v = vload4(0, l_srcArray + l);
      f = vload4(0, fltArray);
      s += v * f;
      v = vload4(0, l_srcArray + l + 4);
      f = vload4(0, fltArray + 4);
      s += v * f;
      v.xyz = vload3(0, l_srcArray + l + 8);
      f.xyz = vload3(0, fltArray + 8);
      s.xyz += v.xyz * f.xyz;

      dstArray[0] = s.x + s.y + s.z + s.w;

      srcArray += pddWidth * pddHeight;
      dstArray += pddWidth * pddHeight;

  }
}

__kernel void convolve_23x(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{

  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;

  __global float * srcArray = massArray + r * pddWidth + c; // section A
  __global float * dstArray = massArray + r * pddWidth + c + pddWidth * pddHeight * 8 * 2; // section C

  const int localSize = get_local_size(0);
  const int l = c % localSize;

  __local float l_srcArray[64 + 22];

  fltArray += (7+11);

  for(int o = 0; o < 8; o++){

    l_srcArray[l + 11] = srcArray[0]; // center value
    if(l < 11){
      l_srcArray[l] = (c > 10 ? srcArray[l-11]:l_srcArray[11]);
    }
    else if(l > localSize-12){
      l_srcArray[l + 22] = (c < pddWidth-11 ? srcArray[l - localSize + 12]:l_srcArray[localSize+10]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 v, f;
    float4 s = 0;

    v = vload4(0, l_srcArray + l);
    f = vload4(0, fltArray);
    s += v * f;
    v = vload4(1, l_srcArray + l);
    f = vload4(1, fltArray);
    s += v * f;
    v = vload4(2, l_srcArray + l);
    f = vload4(2, fltArray);
    s += v * f;
    v = vload4(3, l_srcArray + l);
    f = vload4(3, fltArray);
    s += v * f;
    v = vload4(4, l_srcArray + l);
    f = vload4(4, fltArray);
    s += v * f;
    v.xyz = vload3(0, l_srcArray + l + 20);
    f.xyz = vload3(0, fltArray + 20);
    s.xyz += v.xyz * f.xyz;

    dstArray[0] = s.x + s.y + s.z + s.w;

    srcArray += pddWidth * pddHeight;
    dstArray += pddWidth * pddHeight;

  }
}

__kernel void convolve_23y(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{

    const int r = get_global_id(0) / pddHeight;
    const int c = get_global_id(0) % pddHeight;

    __global float * srcArray = massArray + c * pddWidth + r + pddWidth * pddHeight * 8 * 2; // section C
    __global float * dstArray = massArray + c * pddWidth + r + pddWidth * pddHeight * 8; // section B

    const int localSize = get_local_size(0);
    const int l = c % localSize;

    __local float l_srcArray[64 + 22];

    fltArray += (7+11);

    for(int o = 0; o < 8; o++){

      l_srcArray[l + 11] = srcArray[0]; // center value
      if(l < 11){
        l_srcArray[l] = (c > 10 ? srcArray[(l-11)*pddWidth]:l_srcArray[11]);
      }
      else if(l > localSize-12){
        l_srcArray[l + 22] = (c < pddHeight-11 ? srcArray[(l - localSize + 12)*pddWidth]:l_srcArray[localSize+10]);
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      float4 v, f;
      float4 s = 0;

      v = vload4(0, l_srcArray + l);
      f = vload4(0, fltArray);
      s += v * f;
      v = vload4(1, l_srcArray + l);
      f = vload4(1, fltArray);
      s += v * f;
      v = vload4(2, l_srcArray + l);
      f = vload4(2, fltArray);
      s += v * f;
      v = vload4(3, l_srcArray + l);
      f = vload4(3, fltArray);
      s += v * f;
      v = vload4(4, l_srcArray + l);
      f = vload4(4, fltArray);
      s += v * f;
      v.xyz = vload3(0, l_srcArray + l + 20);
      f.xyz = vload3(0, fltArray + 20);
      s.xyz += v.xyz * f.xyz;

      dstArray[0] = s.x + s.y + s.z + s.w;

      srcArray += pddWidth * pddHeight;
      dstArray += pddWidth * pddHeight;

  }
}

#define TRANS_GROUP_SIZE_X 32
#define TRANS_GROUP_SIZE_Y 8
#define GRADIENT_NUM 8
__kernel void transpose(__global float * srcArray,
                        __global float * dstArray,
                        __local  float * lclArray,
                        const    int     srcWidth,
                        const    int     srcHeight,
                        const    int     dstWidth,
                        const    int     dstHeight)
{
    const int xid = get_global_id(0);
    const int yid = get_global_id(1); 
    const int smoothSectionHeight = srcHeight * GRADIENT_NUM;

    const int smoothSection = yid / smoothSectionHeight;

    const int groupRow = (yid % smoothSectionHeight) / 8;

    const int groupRowGradientSection = get_local_id(1);

    const int srcIndex = (smoothSection * smoothSectionHeight + groupRowGradientSection * srcHeight + groupRow) * srcWidth + xid;

    lclArray[get_local_id(1) * (TRANS_GROUP_SIZE_X+1) + get_local_id(0)] = srcArray[srcIndex];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int localY = get_local_id(0) % TRANS_GROUP_SIZE_Y;
    const int localX = get_local_id(0) / TRANS_GROUP_SIZE_Y + get_local_id(1) * (TRANS_GROUP_SIZE_X / TRANS_GROUP_SIZE_Y);

    const int dstRow = smoothSection * dstHeight + groupRow;
    const int dstCol = get_group_id(0) * TRANS_GROUP_SIZE_X * GRADIENT_NUM + localX * GRADIENT_NUM + localY;

    const int dstIndex = dstRow * dstWidth + dstCol;

    dstArray[dstIndex] = lclArray[localY * (TRANS_GROUP_SIZE_X+1) + localX];
}







