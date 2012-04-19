/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : February/2012

  File: oclDaisy.h

*/

#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

#include "ocl/cachedProgram.h"
#include "ocl/cachedConstructs.h"

#include "kutility/math.h"

#define min(a,b) (a > b ? b : a)
#define max(a,b) (a > b ? a : b)

#define DAISY_PROFILING
#define DAISY_HOST_TRANSFER
//#define DEBUG_ALL

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
  cl_kernel kernel_f11x;
  cl_kernel kernel_f11y;
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

#ifndef DAISY_PARAMS

#define SMOOTHINGS_NO 3
#define SIGMA_A 2.5f
#define SIGMA_B 5.0f
#define SIGMA_C 7.5f
#define NO_GRADIENTS 8
#define REGION_PETALS_NO 8

#define DAISY_PARAMS
typedef struct daisy_params_tag{
  unsigned char * array;
  float * descriptors;
  int width;
  int height;
  int petalsNo;
  int totalPetalsNo;
  int gradientsNo;
  int smoothingsNo;
  int paddedWidth;
  int paddedHeight;
  int descriptorLength;
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

daisy_params * newDaisyParams(unsigned char*, int, int, int, int, int);

int initOcl(ocl_daisy_programs*, ocl_constructs *);

int oclDaisy(daisy_params *, ocl_constructs *, time_params *);

void unpadDescriptorArray(daisy_params *);

