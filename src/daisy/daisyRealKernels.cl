
#define TRANS_GROUP_SIZE_X 32
#define TRANS_GROUP_SIZE_Y 8
#define GRADIENT_NUM 8

// dstWidth = srcWidth * GRADIENT_NUM
// smooth section rows = srcHeight * GRADIENT_NUM
// dstHeight = srcHeight
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

  //
  // Transpose from SxGxHxW --> SxHxWxG
  //

  // coalesced reads from global to local

  const int smoothSectionHeight = srcHeight * GRADIENT_NUM; // smoothing section rows
  const int smoothSection = yid / smoothSectionHeight;      // smoothing section index

  const int groupRow = (yid % smoothSectionHeight) / 8;     // image row within 0-(srcHeight-1)

  const int groupRowGradientSection = get_local_id(1);      // gradient section index 0-7

  const int srcIndex = (smoothSection * smoothSectionHeight + groupRowGradientSection * srcHeight + groupRow) * srcWidth + xid;

  lclArray[get_local_id(1) * (TRANS_GROUP_SIZE_X+1) + get_local_id(0)] = srcArray[srcIndex];

  barrier(CLK_LOCAL_MEM_FENCE);

  // non-conflicting reads of local mem
  const int localY = get_local_id(0) % TRANS_GROUP_SIZE_Y;
  const int localX = get_local_id(0) / TRANS_GROUP_SIZE_Y + get_local_id(1) * (TRANS_GROUP_SIZE_X / TRANS_GROUP_SIZE_Y);

  // coalesced writes to global
  const int dstRow = smoothSection * dstHeight + groupRow;
  //const int dstCol = xid * GRADIENT_NUM + (yid % smoothSectionHeight) / srcHeight;
  const int dstCol = xid * GRADIENT_NUM + localY;

  const int dstIndex = dstRow * dstWidth + dstCol;

  dstArray[dstIndex] = lclArray[localY * (TRANS_GROUP_SIZE_X+1) + localX];

}


