#include <stdio.h>
#include <math.h>

#include "ocl/cachedProgram.h"
#include "ocl/cachedConstructs.h"

#include "kutility/math.h"

#define min(a,b) (a > b ? b : a)
#define max(a,b) (a > b ? a : b)

// downsample for each sigma of PHI_SIGMA_DOWNSAMPLE
#define PHI_SIGMA_DOWNSAMPLE 2
#define DOWNSAMPLE_RATE 4

#define DAISY_PROFILING
#define DAISY_HOST_TRANSFER

#ifndef OCL_DAISY_PROGRAMS
#define OCL_DAISY_PROGRAMS
typedef struct ocl_daisy_programs_tag{
  cl_program program_f7;
  cl_program program_gX;
  cl_program program_gY;
  cl_program program_gAll;
  cl_program program_f13;
  cl_program program_f23;
  cl_program program_f29;
  cl_program program_fAny;
  cl_program program_trans;
  cl_kernel kernel_f7x;
  cl_kernel kernel_f7y;
  cl_kernel kernel_gX;
  cl_kernel kernel_gY;
  cl_kernel kernel_gAll;
  cl_kernel kernel_fxds;
  cl_kernel kernel_fyds;
  cl_kernel kernel_f23x;
  cl_kernel kernel_f23y;
  cl_kernel kernel_f29x;
  cl_kernel kernel_f29y;
  cl_kernel kernel_fAny;
  cl_kernel kernel_gAny;
  cl_kernel kernel_trans;
  cl_kernel kernel_transd;
} ocl_daisy_programs;
#endif

#ifndef PYRAMID_LAYER_SET
#define PYRAMID_LAYER_SET
typedef struct pyramid_layer_set_tag{
  float phi;
  float sigma; // sigma to get from previous layer
  int downsample; // downsample due to smoothing on previous layer
  float t_sigma; // total sigma
  int t_downsample; // total downsample
} pyramid_layer_set;
#endif

#ifndef DAISY_PARAMS
#define DAISY_PARAMS
typedef struct daisy_params_tag{
  unsigned char * array;
  float * descriptors;
  int width;
  int height;
  int petalsNo;
  int gradientsNo;
  int smoothingsNo;
  int totalPetalsNo;
  int descriptorLength;
  float * sigmas;
  pyramid_layer_set ** pyramidLayerSettings;
  int * pyramidLayerSizes;
  int * pyramidLayerOffsets;
  int paddedWidth;
  int paddedHeight;
  ocl_daisy_programs oclPrograms;
} daisy_params;
#endif

#ifdef DAISY_PROFILING
typedef struct time_params_tag{

  // Time structures - measure down to microseconds

  struct timeval startFull, endFull; // whole daisy computation - conv,grad,transA(,transB(,transfers))

  struct timeval startConvGrad, endConvGrad; // measure all convolutions and gradient

  struct timeval startTransGrad, endTransGrad; // measure transposition of gradients

  struct timeval startTransDaisy, endTransDaisy; // measure transposition to daisy descriptors, with or without transfers

  double startt, endt, difft;

  short int measureDeviceHostTransfers;

} time_params;
#endif

pyramid_layer_set * newPyramidLayerSetting(float, float, int);

daisy_params * newDaisyParams(unsigned char*, int, int, int, int, int, float*);

int initOcl(ocl_daisy_programs*, ocl_constructs *);

int oclDaisy(daisy_params *, ocl_constructs *, time_params *);

