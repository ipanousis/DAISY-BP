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
if(l == 0){ l_srcArray[0] = (c > 0 ? massArray[srcOffset-pddWidth*3]:l_srcArray[3]);
  l_srcArray[1] = (c > 0 ? massArray[srcOffset-pddWidth*2]:l_srcArray[3]);
  l_srcArray[2] = (c > 0 ? massArray[srcOffset-pddWidth]:l_srcArray[3]); }
else if(l == localSize-1){ l_srcArray[localSize+3] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth]:l_srcArray[l+3]);
  l_srcArray[localSize+4] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth*2]:l_srcArray[l+3]);
  l_srcArray[localSize+5] = (c < pddHeight-1 ? massArray[srcOffset+pddWidth*3]:l_srcArray[l+3]); }
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
#define CONVX_WORKER_STEPS 8
__kernel void convolve_x23(__global   float * massArray,
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
  fltArray += (7+11);

  for(int w = 1; w < CONVX_WORKER_STEPS+1; w++){
    const int dstOffset = pddWidth * pddHeight * 8 * 2 + srcOffset;
    float s = 0;
    for(int i = lx-11; i < lx+12; i++)
      s += lclArray[get_local_id(1)][w * CONVX_GROUP_SIZE_X + i] * fltArray[i-lx+11];

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

  for(int w = 1; w < CONVY_WORKER_STEPS+1; w++){
    const int dstOffset = srcOffset - pddWidth * pddHeight * 8;
    const int ly = get_local_id(1);
    float s = 0;
    for(int i = ly-11; i < ly+12; i++)
      s += lclArray[get_local_id(0)][w * CONVY_GROUP_SIZE_Y + i] * fltArray[i-ly+11];
    massArray[dstOffset + w * CONVY_GROUP_SIZE_Y * pddWidth] = s;
  }
}
#define CONVX_WORKER_STEPS 4
__kernel void convolve_x29(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  const int lx = get_local_id(0);
  const int srcOffset = pddWidth * pddHeight * 8 + get_global_id(1) * pddWidth + (get_group_id(0) * CONVX_WORKER_STEPS-1) * CONVX_GROUP_SIZE_X + lx;

  __local float lclArray[CONVX_GROUP_SIZE_Y][CONVX_GROUP_SIZE_X * (CONVX_WORKER_STEPS + 2)];

  for(int i = 1; i < CONVX_WORKER_STEPS+1; i++)
    lclArray[get_local_id(1)][i * CONVX_GROUP_SIZE_X + lx] = massArray[srcOffset + i * CONVX_GROUP_SIZE_X];

  lclArray[get_local_id(1)][lx] = (get_global_id(0) >= CONVX_GROUP_SIZE_X ? massArray[srcOffset]:lclArray[get_local_id(1)][CONVX_GROUP_SIZE_X]);

  lclArray[get_local_id(1)][lx + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X] = (get_global_id(0) < pddWidth / CONVX_WORKER_STEPS - CONVX_GROUP_SIZE_X ? massArray[srcOffset + (CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X]:lclArray[get_local_id(1)][(CONVX_WORKER_STEPS+1) * CONVX_GROUP_SIZE_X-1]);

  barrier(CLK_LOCAL_MEM_FENCE);
  fltArray += (7+11+23);

  for(int w = 1; w < CONVX_WORKER_STEPS+1; w++){
    const int dstOffset = pddWidth * pddHeight * 8 * 2 + srcOffset;
    float s = 0;
    for(int i = lx-14; i < lx+15; i++)
      s += lclArray[get_local_id(1)][w * CONVX_GROUP_SIZE_X + i] * fltArray[i-lx+14];

    massArray[dstOffset + w * CONVX_GROUP_SIZE_X] = s;
  }
}
__kernel void convolve_y29(__global   float * massArray,
                           __constant float * fltArray,
                           const      int     pddWidth,
                           const      int     pddHeight)
{
  __local float lclArray[CONVX_GROUP_SIZE_X][CONVY_GROUP_SIZE_Y * (CONVY_WORKER_STEPS+2) + 1];

  const int srcOffset = ((get_group_id(1) * CONVY_WORKER_STEPS-1) * CONVY_GROUP_SIZE_Y + get_local_id(1)) * pddWidth + get_global_id(0) + pddWidth * pddHeight * 8 * 3;

  for(int i = 1; i < CONVY_WORKER_STEPS+1; i++)
    lclArray[get_local_id(0)][i * CONVY_GROUP_SIZE_Y + get_local_id(1)] = massArray[srcOffset + i * CONVY_GROUP_SIZE_Y * pddWidth];

  lclArray[get_local_id(0)][get_local_id(1)] = ((get_global_id(1) * CONVY_WORKER_STEPS) % pddHeight >= CONVY_GROUP_SIZE_Y ? massArray[srcOffset]:lclArray[get_local_id(0)][CONVY_GROUP_SIZE_Y]);

  lclArray[get_local_id(0)][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y + get_local_id(1)] = ((get_global_id(1) * CONVY_WORKER_STEPS) % pddHeight < pddHeight-CONVY_GROUP_SIZE_Y ? massArray[srcOffset + (CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y * pddWidth]:lclArray[get_local_id(0)][(CONVY_WORKER_STEPS+1) * CONVY_GROUP_SIZE_Y-1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  fltArray += (7+11+23);

  for(int w = 1; w < CONVY_WORKER_STEPS+1; w++){
    const int dstOffset = srcOffset - pddWidth * pddHeight * 8;
    const int ly = get_local_id(1);
    float s = 0;
    for(int i = ly-14; i < ly+15; i++)
      s += lclArray[get_local_id(0)][w * CONVY_GROUP_SIZE_Y + i] * fltArray[i-ly+14];
    massArray[dstOffset + w * CONVY_GROUP_SIZE_Y * pddWidth] = s;
  }
}
