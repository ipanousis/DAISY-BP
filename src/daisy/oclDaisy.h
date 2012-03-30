#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "general.h"

#include "ocl/cachedProgram.h"
#include "ocl/cachedConstructs.h"

#include "kutility/math.h"

#define min(a,b) (a > b ? b : a)
#define max(a,b) (a > b ? a : b)

#define DAISY_PROFILING
#define DAISY_HOST_TRANSFER

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


int initOcl(ocl_daisy_programs*, ocl_constructs *);

int oclDaisy(daisy_params *, ocl_constructs *, time_params *);

