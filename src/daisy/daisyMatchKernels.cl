
/*

  Match layer 3 of a small set of DAISY descriptors (the template)
  to a subsampled set of DAISY descriptors (the target frame)

  Compare for one rotation - use either a different parameterisation or a
  different kernel for each rotation

*/

#define GRADIENTS_NO 8
#define REGION_PETALS_NO 8
#define TOTAL_PETALS_NO 25
#define DESCRIPTOR_LENGTH (TOTAL_PETALS_NO * GRADIENTS_NO)

#define ROTATIONS_NO 8

#define TMP_PETALS_NO 2
#define TRG_PIXELS_NO 2
#define WGX_MATCH_COARSE 64

__kernel void diffCoarse(__constant float * tmp,
                         __global   float * trg,
                         __global   float * out,
                         const      int     height,
                         const      int     width,
                         const      int     petalNo)
{

  __local float lclTrg[TRG_PIXELS_NO+1][REGION_PETALS_NO * GRADIENTS_NO];

  const int lid = get_local_id(0);
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);

  // fetch target pixels to local memory; GRADIENTS_NO x REGION_PETALS_NO x TRG_PIXELS_NO (128 = 2 steps)
  int i;
  for(i = 0; i < (TRG_PIXELS_NO * REGION_PETALS_NO * GRADIENTS_NO) / WGX_MATCH_COARSE; i++){

    lclTrg[i][lid] =    

      trg[(gy * width + (gx / WGX_MATCH_COARSE) * WGX_MATCH_COARSE + i) * DESCRIPTOR_LENGTH + 
          (TOTAL_PETALS_NO-REGION_PETALS_NO) * GRADIENTS_NO + lid]; 

  }

  // do 4 diffs and sum them
  float diffs = 0.0;
  
  // first 32 threads do diffs for the first pixel, others for the second pixel
  // first 4 threads do diffs for rotation 0, next 4 threads rotation 1....
  const int pixelNo = lid / (WGX_MATCH_COARSE / TRG_PIXELS_NO);
  const int rotationNo = (lid / (WGX_MATCH_COARSE / (TRG_PIXELS_NO * ROTATIONS_NO))) % ROTATIONS_NO;

  const int DIFFS = (TRG_PIXELS_NO * TMP_PETALS_NO * GRADIENTS_NO * ROTATIONS_NO) / WGX_MATCH_COARSE;

  // get these by rotationNo and lid
  const int trgPetal = (petalNo + rotationNo) % REGION_PETALS_NO;
  const int trgFirstGradient = (lid % (GRADIENTS_NO / DIFFS)) * DIFFS + rotationNo;
  for(i = 0; i < DIFFS; i++){

    // pick pixel, pick rotation => pick petal, pick gradient
    diffs += fabs(tmp[lid] - lclTrg[pixelNo][trgPetal * GRADIENTS_NO + 
                                   (trgFirstGradient + i) % GRADIENTS_NO]);

  }

  // put them in local memory
  lclTrg[TRG_PIXELS_NO][lid] = diffs;

  barrier(CLK_LOCAL_MEM_FENCE);

  // the first 32 threads sum half of the 64 values
  if(lid < WGX_MATCH_COARSE / TRG_PIXELS_NO)
    lclTrg[TRG_PIXELS_NO][lid * 2] = lclTrg[TRG_PIXELS_NO][lid * 2] + 
                                     lclTrg[TRG_PIXELS_NO][lid * 2 + 1];

  barrier(CLK_LOCAL_MEM_FENCE);

  // the first 16 threads sum the half of half
  if(lid < (WGX_MATCH_COARSE / (TRG_PIXELS_NO * 2))){

    diffs = lclTrg[TRG_PIXELS_NO][lid * 4] + lclTrg[TRG_PIXELS_NO][lid * 4 + 2];

    // first 16 fetch and write to global
    out[(gy * width + gx / (WGX_MATCH_COARSE / TRG_PIXELS_NO)) * ROTATIONS_NO + 
        lid] += diffs;

  }

}

// a kernel to normalise per rotation per template point


