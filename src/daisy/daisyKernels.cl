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
    for(int i = l; i < l+7; i++)
      s += l_srcArray[i] * fltArray[i-l];

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
    float s = 0;
    for(int i = l; i < l+11; i++)
      s += l_srcArray[i] * fltArray[i-l]; 
    dstArray[0] = s;
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
                           __local    float * lclArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int yid = get_global_id(1);
  const int xid = get_global_id(0);
  const int srcOffset = yid * pddWidth + xid + pddWidth * pddHeight * 8 * 2;
  const int dstOffset = yid * pddWidth + xid + pddWidth * pddHeight * 8;
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  // Load main data first
  lclArray[(ly+11) * (16+1) + lx] = massArray[srcOffset];
  fltArray += (7+11);
  // Load local upper halo second
  if(ly < 11){
    lclArray[ly * (16+1) + lx] = ((yid % pddHeight) > 10 ? massArray[srcOffset-11*pddWidth]:lclArray[11 * (16+1) + lx]);
  }
  // Load local lower halo third
  if(ly > 16-12){
    lclArray[(ly+22) * (16+1) + lx] = ((yid % pddHeight) < pddHeight-11 ? massArray[srcOffset+11*pddWidth]:lclArray[(11+16-1) * (16+1) + lx]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float s = 0;
  for(int i = ly; i < ly+23; i++)
    s += lclArray[i * (16+1) + lx] * fltArray[i-ly];
  massArray[dstOffset] = s;
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
#define GRADIENT_NUM 8
#define TOTAL_PETALS_NO 25
#define REGION_PETALS_NO 8
#define TRANSD_DATA_HEIGHT 16
#define TRANSD_DATA_WIDTH 16
#define TRANSD_PAIRS_OFFSET_WIDTH 1000
#define TRANSD_PAIRS_SINGLE_ONLY -999
__kernel void transposeDaisy(__global   float * srcArray,
                             __global   float * dstArray,
                             __constant int   * transArray,
                             __local    float * lclArray,
                             const      int     srcWidth,
                             const      int     srcHeight,
                             const      int     srcGlobalOffset,
                             const      int     transArrayLength,
                             const      int     lclArrayPadding) // either 0 or 8
{
  const int xid = get_global_id(0); // 0 - srcWidth-1
  const int yid = get_global_id(1); // 0 - srcHeight-1

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  // coalesced read (srcGlobalOffset + xid,yid) + padded write to lclArray
  const int stepsPerWorker = (srcWidth * GRADIENT_NUM) / get_global_size(0); // => globalSizeX must divide 512 (16,32,64,128,256)
  for(int i = 0; i < stepsPerWorker; i++){
    lclArray[ly * (TRANSD_DATA_WIDTH * GRADIENT_NUM + lclArrayPadding)      // local Y
              + get_local_size(0) * i + lx] =                               // local X
        srcArray[srcGlobalOffset + yid * srcWidth * GRADIENT_NUM +          // global offset + global Y
          (get_group_id(0) * stepsPerWorker + i) * get_local_size(0) + lx]; // global X
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // non-bank-conflicting (at least attempted) read with transArray as well as coalesced write
  const int pairsPerHalfWarp = transArrayLength / ((get_local_size(0) * get_local_size(1)) / 16);
  const int halfWarps = (get_local_size(1) * get_local_size(0)) / 16;
  const int halfWarpId = (ly * get_local_size(0) + lx) / 16;

  const int topLeftY = get_group_id(1) * TRANSD_DATA_HEIGHT;
  const int topLeftX = get_group_id(0) * TRANSD_DATA_WIDTH;
  const int dstGroupOffset = (topLeftY * srcWidth + topLeftX) * GRADIENT_NUM * TOTAL_PETALS_NO;

  const int petalRegion = (srcGlobalOffset / (srcWidth * GRADIENT_NUM)) / srcHeight;

  const int offset = (halfWarpId < (transArrayLength % pairsPerHalfWarp) ? halfWarpId : (transArrayLength % pairsPerHalfWarp));
  for(int p = pairsPerHalfWarp * halfWarpId + offset; 
          p < (halfWarpId == halfWarps-1 ? transArrayLength : pairsPerHalfWarp * (halfWarpId+1) + offset + (halfWarpId < transArrayLength % pairsPerHalfWarp)); 
          p++){
    const int fromP1   = transArray[p * 4];
    const int fromP2   = transArray[p * 4 + 1];
    const int toOffset = transArray[p * 4 + 2];
    const int petalNo  = transArray[p * 4 + 3];
    
    const int toOffsetY = floor(toOffset / (float) TRANSD_PAIRS_OFFSET_WIDTH);
    const int toOffsetX = toOffset - toOffsetY * TRANSD_PAIRS_OFFSET_WIDTH - TRANSD_PAIRS_OFFSET_WIDTH/2;

    const int intraHalfWarpOffset = (lx >= 8) * (fromP2-fromP1);

    if(topLeftY+toOffsetY < 0 || topLeftY+toOffsetY >= srcHeight
    || topLeftX+toOffsetX < 0 || topLeftX+toOffsetX >= srcWidth)
    {     }
    else if(fromP2 != TRANSD_PAIRS_SINGLE_ONLY || (lx < 8)){
      dstArray[dstGroupOffset
               + (toOffsetY * srcWidth + toOffsetX) * GRADIENT_NUM * TOTAL_PETALS_NO
               + (petalRegion * REGION_PETALS_NO + 1 + petalNo) * GRADIENT_NUM + lx] =

        lclArray[((fromP1+intraHalfWarpOffset) / TRANSD_DATA_WIDTH) * (TRANSD_DATA_WIDTH * GRADIENT_NUM + lclArrayPadding) 
               + ((fromP1+intraHalfWarpOffset) % TRANSD_DATA_WIDTH) * GRADIENT_NUM + lx % 8];
    }
  }
}
#define TRANS_GROUP_SIZE_X 32
#define TRANS_GROUP_SIZE_Y 8
__kernel void transposeGradients(__global float * srcArray,
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

