#include <stdio.h>

#include "ocl/cachedProgram.h"
#include "ocl/cachedConstructs.h"

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
  cl_program program_gAny;
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
} ocl_daisy_programs;
#endif

#ifndef DAISY_PARAMS
#define DAISY_PARAMS
typedef struct daisy_params_tag{
  unsigned char * array;
  int width;
  int height;
  int orientationsNo;
  int smoothingsNo;
  int paddedWidth;
  int paddedHeight;
  ocl_daisy_programs oclPrograms;
} daisy_params;
#endif

daisy_params * newDaisyParams(unsigned char*, int, int, int, int);

int initOcl(daisy_params *, ocl_constructs *);

int oclDaisy(daisy_params *, ocl_constructs *);
