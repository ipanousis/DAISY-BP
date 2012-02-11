
__kernel void convolve_7x(__global   float * massArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{
    const int r = get_global_id(0) / pddWidth;
    const int c = get_global_id(0) % pddWidth;

    const int srcOffset = r * pddWidth + c;

    const int localSize = get_local_size(0);
    const int l = c % localSize;

    __local float lclArray[64 + 6];

    lclArray[l + 3] = massArray[srcOffset]; // center value
    if(l < 3){
      lclArray[l] = (c > 2 ? massArray[srcOffset-3]:lclArray[3]);
    }
    else if(l > localSize-4){
      lclArray[l+6] = (c < pddWidth-3 ? massArray[srcOffset+3]:lclArray[localSize+2]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float s = 0;
    for(int i = l; i < l+7; i++)
      s += lclArray[i] * fltArray[i-l];

    const int dstOffset = pddWidth * pddHeight + r * pddWidth + c;
    massArray[dstOffset] = s;
}

__kernel void convolve_7y(__global   float * massArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{
    const int r = get_global_id(0) / pddHeight;
    const int c = get_global_id(0) % pddHeight;

    const int srcOffset = pddWidth * pddHeight + c * pddWidth + r;

    const int localSize = get_local_size(0);
    const int l = c % localSize;

    __local float l_srcArray[64 + 6];

    l_srcArray[l + 3] = massArray[srcOffset]; // center value
    if(l == 0){
      l_srcArray[0] = (c > 0 ? massArray[srcOffset-pddWidth*3]:l_srcArray[3]);
      l_srcArray[1] = (c > 0 ? massArray[srcOffset-pddWidth*2]:l_srcArray[3]);
      l_srcArray[2] = (c > 0 ? massArray[srcOffset-pddWidth]:l_srcArray[3]);
    }
    else if(l == localSize-1){
      l_srcArray[localSize+3] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth]:l_srcArray[l+3]);
      l_srcArray[localSize+4] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth*2]:l_srcArray[l+3]);
      l_srcArray[localSize+5] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth*3]:l_srcArray[l+3]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0;
    s += l_srcArray[l] * fltArray[0];
    s += l_srcArray[l+1] * fltArray[1];
    s += l_srcArray[l+2] * fltArray[2];
    s += l_srcArray[l+3] * fltArray[3];
    s += l_srcArray[l+4] * fltArray[4];
    s += l_srcArray[l+5] * fltArray[5];
    s += l_srcArray[l+6] * fltArray[6];

    const int dstOffset = pddWidth * pddHeight * 8 + c * pddWidth + r;
    massArray[dstOffset] = s;
}
__kernel void gradient_8all(__global float * massArray,
                            const    int     pddWidth,
                            const    int     pddHeight){

  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;

  const int srcOffset = pddWidth * pddHeight * 8 + r * pddWidth + c;

  float4 n;
  n.x = (c > 0           ? massArray[srcOffset-1]:massArray[srcOffset]);
  n.y = (r > 0           ? massArray[srcOffset-pddWidth]:massArray[srcOffset]);
  n.z = (c < pddWidth-1  ? massArray[srcOffset+1]:massArray[srcOffset]);
  n.w = (r < pddHeight-1 ? massArray[srcOffset+pddWidth]:massArray[srcOffset]);

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

  const int dstOffset = r * pddWidth + c;
  const int push = pddWidth * pddHeight;

  massArray[dstOffset]        = gradients.s0;
  massArray[dstOffset+push]   = gradients.s1;
  massArray[dstOffset+2*push] = gradients.s2;
  massArray[dstOffset+3*push] = gradients.s3;
  massArray[dstOffset+4*push] = gradients.s4;
  massArray[dstOffset+5*push] = gradients.s5;
  massArray[dstOffset+6*push] = gradients.s6;
  massArray[dstOffset+7*push] = gradients.s7;
}

#define CONV7_GROUP_SIZE_X 64
__kernel void convolve_11x(__global   float * massArray,
                           __constant float * fltArray,
                           __local    float * lclArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;

  const int srcOffset = r * pddWidth + c; // section A
  const int dstOffset = r * pddWidth + c + pddWidth * pddHeight * 8; // section B

  const int l = c % CONV7_GROUP_SIZE_X;

  fltArray += 7;

  lclArray[l + 5] = massArray[srcOffset]; // center value
  if(l < 5){
    lclArray[l] = (c > 4 ? massArray[srcOffset-5]:lclArray[5]);
  }
  else if(l > CONV7_GROUP_SIZE_X-6){
    lclArray[l + 10] = (c < pddWidth-5 ? massArray[srcOffset+5]:lclArray[CONV7_GROUP_SIZE_X+4]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float s = 0;
  for(int i = l; i < l+11; i++)
    s += lclArray[i] * fltArray[i-l];

  massArray[dstOffset] = s;
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

#define CONV23_GROUP_SIZE_X 64

__kernel void convolve_23x(__global   float * massArray,
                           __constant float * fltArray,
                           __local    float * lclArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{

  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;

  const int srcOffset = r * pddWidth + c; // section A

  const int l = get_local_id(0);

  fltArray += (7+11);

  lclArray[l + 11] = massArray[srcOffset]; // center value
  if(l < 11){
    lclArray[l] = (c > 10 ? massArray[srcOffset-11]:lclArray[11]);
  }
  else if(l > CONV23_GROUP_SIZE_X-12){
    lclArray[l + 22] = (c < pddWidth-11 ? massArray[srcOffset+11]:lclArray[CONV23_GROUP_SIZE_X+10]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float s = 0;
  for(int i = l; i < l+23; i++)
    s += lclArray[i] * fltArray[i-l];

  const int dstOffset = r * pddWidth + c + pddWidth * pddHeight * 8 * 2; // section C
  massArray[dstOffset] = s;

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

#define CONV29X_GROUP_SIZE_X 64

__kernel void convolve_29x(__global   float * massArray,
                           __constant float * fltArray,
                           __local    float * lclArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;

  const int srcOffset = r * pddWidth + c + pddWidth * pddHeight * 8; // section B

  const int l = get_local_id(0);

  fltArray += (7+11+23);

  lclArray[l + 14] = massArray[srcOffset]; // center value
  if(l < 14){
    lclArray[l] = (c > 13 ? massArray[srcOffset-14]:lclArray[14]);
  }
  else if(l > CONV29X_GROUP_SIZE_X-15){
    lclArray[l + 28] = (c < pddWidth-14 ? massArray[srcOffset+14]:lclArray[CONV29X_GROUP_SIZE_X+13]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float s = 0;
  for(int i = l; i < l+29; i++)
    s += lclArray[i] * fltArray[i-l];

  const int dstOffset = r * pddWidth + c + pddWidth * pddHeight * 8 * 3; // section D
  massArray[dstOffset] = s;
}

#define CONV29Y_GROUP_SIZE_X 16
#define CONV29Y_GROUP_SIZE_Y 16

__kernel void convolve_29y(__global   float * massArray,
                           __constant float * fltArray,
                           __local    float * lclArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{

    const int r = get_global_id(1);
    const int c = get_global_id(0) % pddWidth;

    const int srcOffset = r * pddWidth + c + pddWidth * pddHeight * 8 * 3; // section D

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    fltArray += (7+11+23);

    // Load main data first
    lclArray[(ly+14) * (CONV29Y_GROUP_SIZE_X+1) + lx] = massArray[srcOffset];

    // Load local upper halo second
    if(ly < 14){
      lclArray[ly * (CONV29Y_GROUP_SIZE_X+1) + lx] = ((r % pddHeight) > 13 ? massArray[srcOffset-14*pddWidth]:lclArray[14 * (CONV29Y_GROUP_SIZE_X+1) + lx]);
    }

    // Load local lower halo third
    if(ly > CONV29Y_GROUP_SIZE_Y-15){
      lclArray[(ly+28) * (CONV29Y_GROUP_SIZE_X+1) + lx] = ((r % pddHeight) < pddHeight-14 ? massArray[srcOffset+14*pddWidth]:lclArray[(14+CONV29Y_GROUP_SIZE_Y-1) * (CONV29Y_GROUP_SIZE_X+1) + lx]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float s = 0;
    for(int i = ly; i < ly+29; i++)
      s += lclArray[i * (CONV29Y_GROUP_SIZE_X+1) + lx] * fltArray[i-ly];

    const int dstOffset = srcOffset - pddWidth * pddHeight * 8; // section C
    massArray[dstOffset] = s;
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
