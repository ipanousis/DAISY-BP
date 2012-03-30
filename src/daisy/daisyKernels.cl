#define GRADIENT_NUM 8

#define CONVX_GROUP_SIZE_X 16
#define CONVX_GROUP_SIZE_Y 8
#define CONVX_WORKER_STEPS 4

__kernel void convolve_x7(__global   float * pyramidArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  __local float lclArray[CONVX_GROUP_SIZE_Y][CONVX_GROUP_SIZE_X * (CONVX_WORKER_STEPS + 2)];

  const int srcOffsetX = (get_group_id(0) * CONVX_WORKER_STEPS-1) * CONVX_GROUP_SIZE_X + lx;
  const int srcOffset = get_global_id(1) * pddWidth + srcOffsetX;

  for(int i = 1; i < CONVX_WORKER_STEPS+1; i++)
    lclArray[ly][i * CONVX_GROUP_SIZE_X + lx] = pyramidArray[srcOffset + i * CONVX_GROUP_SIZE_X];

  lclArray[ly][lx] = (srcOffsetX >= 0 ? pyramidArray[srcOffset]:lclArray[ly][CONVX_GROUP_SIZE_X]);

  lclArray[ly][lx + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X] = (srcOffsetX + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X < pddWidth ? pyramidArray[srcOffset + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X]:lclArray[ly][(CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int w = 1; w < CONVX_WORKER_STEPS+1; w++){
    const int dstOffset = pddWidth * pddHeight + srcOffset;
    float s = 0;

    for(int i = lx-1; i < lx+2; i++)
      s += lclArray[ly][w * CONVX_GROUP_SIZE_X + i] * fltArray[i-lx+1];

    pyramidArray[dstOffset + w * CONVX_GROUP_SIZE_X] = s;
  }
}

#define CONVY_GROUP_SIZE_X 16
#define CONVY_GROUP_SIZE_Y 8
#define CONVY_WORKER_STEPS 4

__kernel void convolve_y7(__global   float * pyramidArray,
                          __constant float * fltArray,
                          const      int     pddWidth,
                          const      int     pddHeight)
{
  const int ly = get_local_id(1);
  const int lx = get_local_id(0);  
  __local float lclArray[CONVY_GROUP_SIZE_X][CONVY_GROUP_SIZE_Y * (CONVY_WORKER_STEPS+2) + 1];

  const int srcOffsetY = ((get_group_id(1) * CONVY_WORKER_STEPS-1) * CONVY_GROUP_SIZE_Y + ly);
  const int srcOffset =  srcOffsetY * pddWidth + get_global_id(0) + pddWidth * pddHeight;

  for(int i = 1; i < CONVY_WORKER_STEPS+1; i++)
    lclArray[lx][i * CONVY_GROUP_SIZE_Y + ly] = pyramidArray[srcOffset + i * CONVY_GROUP_SIZE_Y * pddWidth];

  lclArray[lx][ly] = (srcOffsetY >= 0 ? pyramidArray[srcOffset]:lclArray[lx][CONVY_GROUP_SIZE_Y]);

  lclArray[lx][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y + ly] = (srcOffsetY + (CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y < pddHeight ? pyramidArray[srcOffset + (CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y * pddWidth]:lclArray[lx][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int w = 1; w < CONVY_WORKER_STEPS+1; w++){
    const int dstOffset = srcOffset - pddWidth * pddHeight;
    float s = 0;

    for(int i = ly-1; i < ly+2; i++)
      s += lclArray[lx][w * CONVY_GROUP_SIZE_Y + i] * fltArray[i-ly+1];

    pyramidArray[dstOffset + w * CONVY_GROUP_SIZE_Y * pddWidth] = s;
  }
}

__kernel void gradient_all(__global float * pyramidArray,
                           const    int     pddWidth,
                           const    int     pddHeight,
                           const    int     dstGlobalOffset)
{

  const int r = get_global_id(0) / pddWidth;
  const int c = get_global_id(0) % pddWidth;
  const int srcOffset = r * pddWidth + c;

  float4 n;
  n.x = (c > 0           ? pyramidArray[srcOffset-1]:pyramidArray[srcOffset]);
  n.y = (r > 0           ? pyramidArray[srcOffset-pddWidth]:pyramidArray[srcOffset]);
  n.z = (c < pddWidth-1  ? pyramidArray[srcOffset+1]:pyramidArray[srcOffset]);
  n.w = (r < pddHeight-1 ? pyramidArray[srcOffset+pddWidth]:pyramidArray[srcOffset]);

  float8 gradients;
  const float8 angles = (float8)(0.0f, M_PI / 4, M_PI / 2, 3 * (M_PI / 4), M_PI,
                                  5 * (M_PI / 4), 3 * (M_PI / 2), 7 * (M_PI / 4));
  n.x = (n.z-n.x) * 0.5;
  n.y = (n.w-n.y) * 0.5;

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

  const int dstOffset = dstGlobalOffset + r * pddWidth + c;
  const int push = pddWidth * pddHeight;

  pyramidArray[dstOffset]        = gradients.s0;
  pyramidArray[dstOffset+push]   = gradients.s1;
  pyramidArray[dstOffset+2*push] = gradients.s2;
  pyramidArray[dstOffset+3*push] = gradients.s3;
  pyramidArray[dstOffset+4*push] = gradients.s4;
  pyramidArray[dstOffset+5*push] = gradients.s5;
  pyramidArray[dstOffset+6*push] = gradients.s6;
  pyramidArray[dstOffset+7*push] = gradients.s7;
}

#define CONVX_GROUP_SIZE_X 16
#define CONVX_GROUP_SIZE_Y 4
#define CONVX_WORKER_STEPS 4
//#define DOWNSAMPLE_RATE 4
//#define FILTER_RADIUS 6

__kernel void convolveDs_x(__global   float * pyramidArray,
                           const      int     pddWidth,
                           const      int     pddHeight,
                           const      int     srcGlobalOffset,
                           __constant float * fltArray,
                           const      int     fltRadius,
                           const      int     downsampleRate) // downsample in x dimension, should be 1 if no downsample, must be a power of 2
{

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  __local float lclArray[CONVX_GROUP_SIZE_Y][CONVX_GROUP_SIZE_X * (CONVX_WORKER_STEPS + 2)];

  const int srcOffsetX = (get_group_id(0) * CONVX_WORKER_STEPS-1) * CONVX_GROUP_SIZE_X + lx;
  const int srcOffset = srcGlobalOffset + get_global_id(1) * pddWidth + srcOffsetX;

  for(int i = 1; i < CONVX_WORKER_STEPS+1; i++)
    lclArray[ly][i * CONVX_GROUP_SIZE_X + lx] = pyramidArray[srcOffset + i * CONVX_GROUP_SIZE_X];

  lclArray[ly][lx] = (srcOffsetX >= 0 ? pyramidArray[srcOffset]:lclArray[ly][CONVX_GROUP_SIZE_X]);

  lclArray[ly][lx + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X] = (srcOffsetX + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X < pddWidth ? pyramidArray[srcOffset + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X]:lclArray[ly][(CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  // if worker local id is greater than the number of elements this group must output
  if(lx >= (CONVX_GROUP_SIZE_X * CONVX_WORKER_STEPS) / downsampleRate) return; 

  const int dstOffset = get_global_id(1) * (pddWidth / downsampleRate) + get_group_id(0) * ((CONVX_GROUP_SIZE_X * CONVX_WORKER_STEPS) / downsampleRate) + lx;
  for(int w = 1; w < CONVX_WORKER_STEPS + 1; w += downsampleRate){
    float s = 0;
    int f = lx * downsampleRate;
    for(int i = f - fltRadius; i < f + fltRadius + 1; i++)
      s += lclArray[ly][w * CONVX_GROUP_SIZE_X + i] * fltArray[i-f+fltRadius];

    pyramidArray[dstOffset + ((w-1) / downsampleRate) * CONVX_GROUP_SIZE_X] = s;
  }
}

#define CONVY_GROUP_SIZE_X 16
#define CONVY_GROUP_SIZE_Y 8
#define CONVY_WORKER_STEPS 4

__kernel void convolveDs_y(__global   float * pyramidArray,
                           const      int     pddWidth, // should be original width / downsampleRate
                           const      int     pddHeight,
                           const      int     dstGlobalOffset,
                           __constant float * fltArray,
                           const      int     fltRadius,
                           const      int     downsampleRate)
{
  const int ly = get_local_id(1);
  const int lx = get_local_id(0);  
  __local float lclArray[CONVY_GROUP_SIZE_X][CONVY_GROUP_SIZE_Y * (CONVY_WORKER_STEPS+2) + 1];

  const int srcOffsetY = ((get_group_id(1) * CONVY_WORKER_STEPS) * CONVY_GROUP_SIZE_Y + ly);
  const int srcOffset =  srcOffsetY * pddWidth + get_global_id(0);

  for(int i = 0; i < CONVY_WORKER_STEPS; i++)
    lclArray[lx][(i+1) * CONVY_GROUP_SIZE_Y + ly] = pyramidArray[srcOffset + i * CONVY_GROUP_SIZE_Y * pddWidth];

  lclArray[lx][ly] = (get_group_id(1) % ((pddHeight / CONVY_WORKER_STEPS) / get_local_size(1)) ? pyramidArray[srcOffset-CONVY_GROUP_SIZE_Y*pddWidth]:lclArray[lx][CONVY_GROUP_SIZE_Y]);

  lclArray[lx][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y + ly] = ((srcOffsetY % pddHeight) + (CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y < pddHeight ? pyramidArray[srcOffset + CONVY_WORKER_STEPS * CONVY_GROUP_SIZE_Y * pddWidth]:lclArray[lx][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  const int dstOffset = dstGlobalOffset + (get_group_id(1) * ((CONVY_GROUP_SIZE_Y * CONVY_WORKER_STEPS) / downsampleRate) + ly) * pddWidth + get_global_id(0);
  for(int w = 1; w < CONVY_WORKER_STEPS + 1; w += downsampleRate){
    float s = 0;
    int f = ly * downsampleRate;
    for(int i = f - fltRadius; i < f + fltRadius + 1; i++)
      s += lclArray[lx][w * CONVY_GROUP_SIZE_Y + i] * fltArray[i-f+fltRadius];

    pyramidArray[dstOffset + ((w-1) / downsampleRate) * CONVY_GROUP_SIZE_Y * pddWidth] = s;
  }
}

#define TOTAL_PETALS_NO 25
#define REGION_PETALS_NO 8
#define TRANS_GROUP_SIZE_X 32
#define TRANS_GROUP_SIZE_Y 8

__kernel void transposeGradients(__global float * srcArray,
                                 __global float * dstArray,
                                 const    int     srcWidth,
                                 const    int     srcHeight,
                                 const    int     srcOffset,
                                 const    int     dstOffset)
{

    const int groupRow = get_global_id(1) / GRADIENT_NUM;
    const int groupRowGradientSection = get_local_id(1);

    const int srcIndex = srcOffset + (groupRowGradientSection * srcHeight + groupRow) * srcWidth + get_global_id(0);

    __local float lclArray[(TRANS_GROUP_SIZE_X+2) * TRANS_GROUP_SIZE_Y];

    lclArray[get_local_id(1) * (TRANS_GROUP_SIZE_X+2) + get_local_id(0)] = srcArray[srcIndex];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int localY = get_local_id(0) % TRANS_GROUP_SIZE_Y;
    const int localX = get_local_id(0) / TRANS_GROUP_SIZE_Y + get_local_id(1) * (TRANS_GROUP_SIZE_X / TRANS_GROUP_SIZE_Y);

    //
    // Normalisation piggy-backing along with the transposition
    //
    float l2normSum = .0f;
    for(int i = 0; i < GRADIENT_NUM; i++){
      const float g = lclArray[((localY+i) % GRADIENT_NUM) * (TRANS_GROUP_SIZE_X+2) + localX];
      l2normSum += g*g;
    }
    l2normSum = (l2normSum == 0.0 ? 1 : 1 / sqrt(l2normSum));
    //
    //
    //

    const int dstRow = groupRow;
    const int dstCol = get_group_id(0) * TRANS_GROUP_SIZE_X * GRADIENT_NUM + localX * GRADIENT_NUM + localY;

    dstArray[dstOffset + dstRow * srcWidth * GRADIENT_NUM + dstCol] = lclArray[localY * (TRANS_GROUP_SIZE_X+2) + localX] * l2normSum; // this division... the division ALONE... seems to take 10 ms !!!
}

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
                             const      int     downsampleFactor,
                             const      int     petalStart) // power of 2
{

  const int gx = get_global_id(0) - TRANSD_DATA_WIDTH; 
                                   // range across all blocks: [0, srcWidth+2*TRANSD_DATA_WIDTH-1] (pushed back to start from -TRANSD_DATA_WIDTH)
                                   // range for a block:
                                   // (same as for all blocks given that the blocks will now be rectangular --> whole rows)

  const int gy = get_global_id(1) - TRANSD_DATA_WIDTH; 
                                   // range across all blocks: [0, srcHeight+2*TRANSD_DATA_WIDTH-1] (pushed back to start from -TRANSD_DATA_WIDTH)
                                   // range for a block:
                                   // [k * TRANSD_BLOCK_WIDTH,
                                   //  min((k+1) * TRANSD_BLOCK_WIDTH + 2*TRANSD_DATA_WIDTH-1, srcHeight + 2*TRANSD_DATA_WIDTH-1)]

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  const int layerWidth  = srcWidth / downsampleFactor;

  //__local float lclArray[TRANSD_DATA_WIDTH * (TRANSD_DATA_WIDTH * GRADIENT_NUM)];

  // coalesced read (srcGlobalOffset + xid,yid) + padded write to lclArray
  //const int stepsPerWorker = (srcWidth * GRADIENT_NUM) / get_global_size(0); // => globalSizeX must divide 512 (16,32,64,128,256)

  // should be no divergence, whole workgroups take the same path because; 
  // srcWidth and srcHeight must be multiples of TRANSD_DATA_WIDTH = GROUP_SIZE_X = GROUP_SIZE_Y = 16
  if(gx < 0 || gx >= srcWidth || gy < 0 || gy >= srcHeight){
    const int stepsPerWorker = 8;

    for(int i = 0; i < stepsPerWorker; i++){
      lclArray[ly * (TRANSD_DATA_WIDTH * GRADIENT_NUM)      // local Y
                + get_local_size(0) * i + lx] =                               // local X
                                                0;                            // outside border
    }
  }
  else{
    const int stepsPerWorker = 8;

    for(int i = 0; i < stepsPerWorker; i++){

      const int y = gy / downsampleFactor;
      const int x = (gx / get_local_size(0)) * (stepsPerWorker / downsampleFactor);
      const int p = i / (downsampleFactor / 2);

      lclArray[ly * (TRANSD_DATA_WIDTH * GRADIENT_NUM)        // local Y
                + get_local_size(0) * i + lx] =                                 // local X
          srcArray[srcGlobalOffset + y * layerWidth * GRADIENT_NUM +            // global offset + global Y
                                     x * get_local_size(0) + p * 8 +            // global X
                                  + lx % 8];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // non-bank-conflicting (at least attempted) read with transArray as well as coalesced write
  const int pairsPerHalfWarp = transArrayLength / ((get_local_size(0) * get_local_size(1)) / 16);
  const int halfWarps = (get_local_size(1) * get_local_size(0)) / 16;
  const int halfWarpId = (ly * get_local_size(0) + lx) / 16;

  const int blockHeight = get_global_size(1) - 2 * TRANSD_DATA_WIDTH;
  const int topLeftY = (get_group_id(1)-1) * TRANSD_DATA_WIDTH;
  const int topLeftX = (get_group_id(0)-1) * TRANSD_DATA_WIDTH;

  const int dstGroupOffset = (topLeftY * srcWidth + topLeftX) * GRADIENT_NUM * TOTAL_PETALS_NO;

  //const int petalStart = ((srcGlobalOffset / (srcWidth * GRADIENT_NUM)) / srcHeight) * REGION_PETALS_NO + (srcGlobalOffset > 0);

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

    if(topLeftY+toOffsetY < 0 || topLeftY+toOffsetY >= blockHeight
    || topLeftX+toOffsetX < 0 || topLeftX+toOffsetX >= srcWidth)
    {     }
    else if(fromP2 != TRANSD_PAIRS_SINGLE_ONLY || (lx < 8)){
      dstArray[dstGroupOffset
               + (toOffsetY * srcWidth + toOffsetX) * GRADIENT_NUM * TOTAL_PETALS_NO
               + (petalStart + petalNo) * GRADIENT_NUM + lx] =

        lclArray[((fromP1+intraHalfWarpOffset) / TRANSD_DATA_WIDTH) * (TRANSD_DATA_WIDTH * GRADIENT_NUM)
               + ((fromP1+intraHalfWarpOffset) % TRANSD_DATA_WIDTH) * GRADIENT_NUM + lx % 8];
    }
  }
}

__kernel void searchDaisy(__global    float * refArray,
                          __global    float * tarArray,
                          __global    float * dspArray,
                          __local     float * lclRefArray,
                          __local     float * lclTarArray,
                          __constant  int   * buildArray,
                          const       int     pyramidOffset,
                          const       int     buildHalo){

  const int radius = get_local_size(0);
  const int blockWidth = radius+2*buildHalo;

  //pyrArrayOffset = pyramidOffset * GRADIENT_NUM
  //dspArrayOffset = pyramidOffset * size(SEARCH_RANGE) // SEARCH_RANGE = (1+2*radius)*(1+2*radius)
  
  //width = get_global_size(0)
  //height = get_global_size(1)

  const int width = get_global_size(0);
  const int gy = get_global_id(1);
  const int gx = get_global_id(0);

  // no matches for border pixels
  if(gx < radius || gx >= width-radius || gy < radius || gy >= get_global_size(1)-radius) return;

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  // import reference block of size (radius+2*buildHalo)*(radius+2*buildHalo)
  if(1){
    const int importSteps = (blockWidth-1) / radius + 1;
    for(int i = 0; i < importSteps; i++){
      if(i < importSteps-1 || ly < blockWidth%radius){

        for(int j = 0; j < blockWidth / (radius / GRADIENT_NUM); j++){

          lclRefArray[(i * radius + ly) * blockWidth * GRADIENT_NUM +     // local Y
                                            j * 2 * GRADIENT_NUM + lx] =  // local X

            refArray[pyramidOffset * GRADIENT_NUM + (get_group_id(1) * radius - buildHalo + i * radius + ly) * width * GRADIENT_NUM + // global Y
                                           (get_group_id(0) * radius + j * 2) * GRADIENT_NUM + lx]; // global X

        }
      }
    }
  }

  for(int by = -1; by < 1; by++){

    for(int bx = -1; bx < 1; bx++){

      // import target block of size 'same'
      if(1){
        const int importSteps = (blockWidth-1) / radius + 1;
        for(int i = 0; i < importSteps; i++){
          if(i < importSteps-1 || ly < blockWidth%radius){

            for(int j = 0; j < blockWidth / (radius / GRADIENT_NUM); j++){

                lclTarArray[(i * radius + ly) * blockWidth * GRADIENT_NUM +     // local Y
                                                  j * 2 * GRADIENT_NUM + lx] =  // local X

                  tarArray[pyramidOffset * GRADIENT_NUM + (get_group_id(1) * radius - buildHalo + i * radius + ly) * width * GRADIENT_NUM + // global Y
                                                 (get_group_id(0) * radius + j * 2) * GRADIENT_NUM + lx]; // global X

            }
          }
        }
      }

      // per ref point
        // per tar point
          // store diff
      for(int tary = max(ly,(by+1)*radius); tary < min(ly+2*radius,(by+2)*radius); tary++){
        for(int tarx = max(lx,(bx+1)*radius); tarx < min(lx+2*radius,(bx+2)*radius); tarx++){

  
          float sum = .0f;
          for(int r = 0; r < REGION_PETALS_NO; r++){
            const int offsetY = buildHalo + buildArray[r * 2]; // if 0 is row
            const int offsetX = buildHalo + buildArray[r * 2 + 1]; // if 1 is column

            // 8 times (gradients no)
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+1]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+1]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+2]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+2]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+3]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+3]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+4]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+4]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+5]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+5]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+6]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+6]);
            sum += fabs( lclRefArray[(offsetY+ly) * blockWidth * GRADIENT_NUM + (offsetX+lx) * GRADIENT_NUM+7]
                       -lclTarArray[(offsetY+tary%radius) * blockWidth * GRADIENT_NUM + (offsetX+tarx%radius) * GRADIENT_NUM+7]);
          }
          dspArray[(gy * width + gx) * (radius*2+1) * (radius*2+1) + (tary-ly-radius) * (radius*2+1) + (tarx-lx-radius)] = sum;
        }
      }

    }

  }
}

