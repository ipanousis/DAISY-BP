// "TEMPLATE" CONVOLUTION FUNCTIONS
#define CONVX_GROUP_SIZE_X 16
#define CONVX_GROUP_SIZE_Y 4
#define CONVX_WORKER_STEPS 4
__kernel void convolve_x(__global   float * massArray,
                         __constant float * fltArray,
                         const      int     pddWidth,
                         const      int     pddHeight)
{
  const int lx = get_local_id(0);
  const int srcOffset = get_global_id(1) * pddWidth + (get_group_id(0) * CONVX_WORKER_STEPS-1) * CONVX_GROUP_SIZE_X + lx;

  __local float lclArray[CONVX_GROUP_SIZE_Y][CONVX_GROUP_SIZE_X * (CONVX_WORKER_STEPS + 2)];

  for(int i = 1; i < CONVX_WORKER_STEPS+1; i++)
    lclArray[get_local_id(1)][i * CONVX_GROUP_SIZE_X + lx] = massArray[srcOffset + i * CONVX_GROUP_SIZE_X];

  lclArray[get_local_id(1)][lx] = (get_global_id(0) >= CONVX_GROUP_SIZE_X ? massArray[srcOffset]:lclArray[get_local_id(1)][CONVX_GROUP_SIZE_X]);

  lclArray[get_local_id(1)][lx + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X] = (get_global_id(0) < pddWidth / CONVX_WORKER_STEPS - CONVX_GROUP_SIZE_X ? massArray[srcOffset + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X]:lclArray[get_local_id(1)][(CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X-1]);

  barrier(CLK_LOCAL_MEM_FENCE);
  fltArray += 7;

  for(int w = 1; w < CONVX_WORKER_STEPS+1; w++){
    const int dstOffset = pddWidth * pddHeight * 8 + srcOffset;
    float s = 0;
    for(int i = lx-5; i < lx+6; i++)
      s += lclArray[get_local_id(1)][w * CONVX_GROUP_SIZE_X + i] * fltArray[i-lx+5];

    massArray[dstOffset + w * CONVX_GROUP_SIZE_X] = s;
  }
}
#define CONVY_GROUP_SIZE_Y 16
#define CONVY_WORKER_STEPS 8
__kernel void convolve_y(__global   float * massArray,
                         __constant float * fltArray,
                         const      int     pddWidth,
                         const      int     pddHeight)
{
  __local float lclArray[CONVX_GROUP_SIZE_X][CONVY_GROUP_SIZE_Y * (CONVY_WORKER_STEPS+2) + 1];

  const int srcOffset = ((get_group_id(1) * CONVY_WORKER_STEPS-1) * CONVY_GROUP_SIZE_Y + get_local_id(1)) * pddWidth + get_global_id(0) + pddWidth * pddHeight * 8 * 2;
  for(int i = 1; i < CONVY_WORKER_STEPS+1; i++)
    lclArray[get_local_id(0)][i * CONVY_GROUP_SIZE_Y + get_local_id(1)] = massArray[srcOffset + i * CONVY_GROUP_SIZE_Y * pddWidth];

  lclArray[get_local_id(0)][get_local_id(1)] = ((get_global_id(1) * CONVY_WORKER_STEPS) % pddHeight >= CONVY_GROUP_SIZE_Y ? massArray[srcOffset]:lclArray[get_local_id(0)][CONVY_GROUP_SIZE_Y]);

  lclArray[get_local_id(0)][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y + get_local_id(1)] = ((get_global_id(1) * CONVY_WORKER_STEPS) % pddHeight < pddHeight-CONVY_GROUP_SIZE_Y ? massArray[srcOffset + (CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y * pddWidth]:lclArray[get_local_id(0)][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  fltArray += (7+11);

  for(int w = 1; w < CONV23Y_WORKER_STEPS+1; w++){
    const int dstOffset = srcOffset - pddWidth * pddHeight * 8;
    const int ly = get_local_id(1);
    float s = 0;
    for(int i = ly-11; i < ly+12; i++)
      s += lclArray[get_local_id(0)][w * CONVY_GROUP_SIZE_Y + i] * fltArray[i-ly+11];
    massArray[dstOffset + w * CONVY_GROUP_SIZE_Y * pddWidth] = s;
  }
}
/*
#define CONV23Y_GROUP_SIZE_Y 16
#define CONV23Y_WORKER_STEPS 8
__kernel void convolve_23y(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  __local float lclArray[CONVX_GROUP_SIZE_X][CONV23Y_GROUP_SIZE_Y * (CONV23Y_WORKER_STEPS+2) + 1]; // 1 halo steps either side and 1 padding
  const int srcOffset = ((get_group_id(1) * CONV23Y_WORKER_STEPS-1) * CONV23Y_GROUP_SIZE_Y + get_local_id(1)) * pddWidth + get_global_id(0) + pddWidth * pddHeight * 8 * 2;
  for(int i = 1; i < CONV23Y_WORKER_STEPS+1; i++)
    lclArray[get_local_id(0)][i * CONV23Y_GROUP_SIZE_Y + get_local_id(1)] = massArray[srcOffset + i * CONV23Y_GROUP_SIZE_Y * pddWidth];

  lclArray[get_local_id(0)][get_local_id(1)] = ((get_global_id(1) * CONV23Y_WORKER_STEPS) % pddHeight >= CONV23Y_GROUP_SIZE_Y ? massArray[srcOffset]:lclArray[get_local_id(0)][CONV23Y_GROUP_SIZE_Y]);

  lclArray[get_local_id(0)][(CONV23Y_WORKER_STEPS+1) * CONV23Y_GROUP_SIZE_Y + get_local_id(1)] = ((get_global_id(1) * CONV23Y_WORKER_STEPS) % pddHeight < pddHeight-CONV23Y_GROUP_SIZE_Y ? massArray[srcOffset + (CONV23Y_WORKER_STEPS+1) * CONV23Y_GROUP_SIZE_Y * pddWidth]:lclArray[get_local_id(0)][(CONV23Y_WORKER_STEPS+1) * CONV23Y_GROUP_SIZE_Y-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  fltArray += (7+11);
  const int dstOffset = srcOffset - pddWidth * pddHeight * 8;
  const int ly = get_local_id(1);
  for(int w = 1; w < CONV23Y_WORKER_STEPS+1; w++){
    float s = 0;
    for(int i = ly-11; i < ly+12; i++)
      s += lclArray[get_local_id(0)][w * CONV23Y_GROUP_SIZE_Y + i] * fltArray[i-ly+11];
    massArray[dstOffset + w * CONV23Y_GROUP_SIZE_Y * pddWidth] = s;
  }
}
*/
/*
#define CONV_GROUP_SIZE_X 16
#define CONV_GROUP_SIZE_Y 4
#define CONV_WORKER_STEPS 4
__kernel void convolve_11x(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int lx = get_local_id(0);
  const int srcOffset = get_global_id(1) * pddWidth + (get_group_id(0) * CONV_WORKER_STEPS-1) * CONV_GROUP_SIZE_X + lx; // section A
  __local float lclArray[CONV_GROUP_SIZE_Y][CONV_GROUP_SIZE_X * (CONV_WORKER_STEPS + 2)];

  for(int i = 1; i < CONV_WORKER_STEPS+1; i++)
    lclArray[get_local_id(1)][i * CONV_GROUP_SIZE_X + lx] = massArray[srcOffset + i * CONV_GROUP_SIZE_X];

  lclArray[get_local_id(1)][lx] = (get_global_id(0) >= CONV_GROUP_SIZE_X ? massArray[srcOffset]:lclArray[get_local_id(1)][CONV_GROUP_SIZE_X]);

  lclArray[get_local_id(1)][lx + (CONV_WORKER_STEPS+1) * CONV_GROUP_SIZE_X] = (get_global_id(0) < pddWidth / CONV_WORKER_STEPS - CONV_GROUP_SIZE_X ? massArray[srcOffset + (CONV_WORKER_STEPS+1) * CONV_GROUP_SIZE_X]:lclArray[get_local_id(1)][(CONV_WORKER_STEPS+1) * CONV_GROUP_SIZE_X-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  fltArray += 7;
  const int dstOffset = srcOffset + pddWidth * pddHeight * 8; // section B

  for(int w = 1; w < CONV_WORKER_STEPS+1; w++){
    float s = 0;
    for(int i = lx-5; i < lx+6; i++)
      s += lclArray[get_local_id(1)][w * CONV_GROUP_SIZE_X + i] * fltArray[i-lx+5];

    massArray[dstOffset + w * CONV_GROUP_SIZE_X] = s;
  }
}
*/
