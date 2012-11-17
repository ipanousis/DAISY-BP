#include <CL/cl.h>
#include <math.h>

#include "ocl/cachedConstructs.h"
#include "ocl/cachedProgram.h"

#define GRADIENTS_NO 8
#define REGION_PETALS_NO 8
#define SMOOTHINGS_NO 3

// downsample for each sigma of PHI_SIGMA_DOWNSAMPLE
#define PHI_SIGMA_DOWNSAMPLE 2.0f
#define DOWNSAMPLE_RATE 2

#define ARRAY_PADDING 256

#define DAISY_SEARCH

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
  cl_kernel kernel_search;
  cl_kernel kernel_match;
} ocl_daisy_programs;
#endif

#ifndef OCL_DAISY_BUFFERS
#define OCL_DAISY_BUFFERS
typedef struct ocl_daisy_buffers_tag{

  cl_mem transBuffer; // pyramid of SxHxWxG data
  //cl_mem daisyBuffer;

} ocl_daisy_buffers;
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
  ocl_daisy_programs * oclPrograms;
  ocl_daisy_buffers * oclBuffers;
} daisy_params;
#endif

pyramid_layer_set * newPyramidLayerSetting(float, float, int);

daisy_params * newDaisyParams(unsigned char*, int, int, int, int, int, float*);

void generatePyramidSettings(daisy_params * params);

float * generatePetalOffsets(float, int, short int);

int * generateTranspositionOffsets(int, int, float*, int, int*, int*);
