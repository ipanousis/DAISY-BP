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
#define GRADIENT_NUM 8
#define TOTAL_PETALS_NO 25
#define REGION_PETALS_NO 8
#define TRANS_GROUP_SIZE_X 32
#define TRANS_GROUP_SIZE_Y 8
__kernel void transposeGradients(__global float * srcArray,
                                 __global float * dstArray,
                                 const    int     srcWidth,
                                 const    int     srcHeight,
                                 const    int     dstWidth,
                                 const    int     dstHeight)
{
    const int smoothSectionHeight = srcHeight * GRADIENT_NUM;

    const int smoothSection = get_global_id(1) / smoothSectionHeight;

    const int groupRow = (get_global_id(1) % smoothSectionHeight) / 8;

    const int groupRowGradientSection = get_local_id(1);

    const int srcIndex = (smoothSection * smoothSectionHeight + groupRowGradientSection * srcHeight + groupRow) * srcWidth + get_global_id(0);

    __local float lclArray[(TRANS_GROUP_SIZE_X+1) * TRANS_GROUP_SIZE_Y];

    lclArray[get_local_id(1) * (TRANS_GROUP_SIZE_X+1) + get_local_id(0)] = srcArray[srcIndex];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int localY = get_local_id(0) % TRANS_GROUP_SIZE_Y;
    const int localX = get_local_id(0) / TRANS_GROUP_SIZE_Y + get_local_id(1) * (TRANS_GROUP_SIZE_X / TRANS_GROUP_SIZE_Y);

    const int dstRow = smoothSection * dstHeight + groupRow;
    const int dstCol = get_group_id(0) * TRANS_GROUP_SIZE_X * GRADIENT_NUM + localX * GRADIENT_NUM + localY;

    dstArray[dstRow * dstWidth + dstCol] = lclArray[localY * (TRANS_GROUP_SIZE_X+1) + localX];
}
