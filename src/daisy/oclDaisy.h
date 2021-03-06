/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : February/2012

  File: oclDaisy.h

*/

#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/time.h>

#include "ocl/cachedProgram.h"
#include "ocl/cachedConstructs.h"

#include "kutility/general.h"
#include "kutility/math.h"
#include "kutility/image.h"
#include "kutility/progress_bar.h"
#include "kutility/fileio.h"
#include "kutility/corecv.h"

using kutility::allocate;
using kutility::deallocate;
using kutility::type_cast;
using kutility::divide;
using kutility::is_outside;
using kutility::save;
using kutility::l2norm;
using kutility::scale;
using kutility::point_transform_via_homography;
using kutility::load_gray_image;
using kutility::save_binary;

//#define TEST_FETCHDAISY
//#define CPU_VERIFICATION

#ifndef OCL_DAISY_KERNELS
#define OCL_DAISY_KERNELS
typedef struct ocl_daisy_kernels_tag{
  cl_kernel denx;
  cl_kernel deny;
  cl_kernel grad;
  cl_kernel G0x;
  cl_kernel G0y;
  cl_kernel G1x;
  cl_kernel G1y;
  cl_kernel G2x;
  cl_kernel G2y;
  cl_kernel trans;
  cl_kernel transd;
  cl_kernel transdp;
  cl_kernel transds;
  cl_kernel fetchd;
  cl_kernel diffCoarse;
  cl_kernel transposeRotations;
  cl_kernel reduceMin;
  cl_kernel reduceMinAll;
  cl_kernel normaliseRotation;
  cl_kernel diffMiddle;
  unsigned int kernelsNo;
} ocl_daisy_kernels;
#endif

#define SUBSAMPLE_RATE 2
#define SMOOTHINGS_NO 3
#define SIGMA_DEN 0.5f
#define SIGMA_A 2.5f
#define SIGMA_B 5.0f
#define SIGMA_C 7.5f
#define GRADIENTS_NO 8
#define REGION_PETALS_NO 8
#define TOTAL_PETALS_NO (SMOOTHINGS_NO * REGION_PETALS_NO + 1)

// Padding for aligned memory mapping in fast kernel
// not needed for NVIDIA CC 3.0 which uses cache to serve misaligned accesses
// but if such is not the case then it should be used to give petal writes a 64-byte alignment
#define TRANSD_FAST_PETAL_PADDING 0

#define DESCRIPTOR_LENGTH (TOTAL_PETALS_NO + TRANSD_FAST_PETAL_PADDING) * GRADIENTS_NO

#ifndef DAISY_PARAMS
#define DAISY_PARAMS
typedef struct daisy_params_tag{
  char * filename;
  unsigned char * array;
  float * descriptors;
  int width;
  int height;
  int regionPetalsNo;
  int totalPetalsNo;
  int gradientsNo;
  int smoothingsNo;
  int paddedWidth;
  int paddedHeight;
  int descriptorLength;
  ocl_daisy_kernels * oclKernels;
  cl_mem * buffers;
  unsigned int buffersSize;
  short int cpuTransfer;
} daisy_params;
#endif

typedef struct time_params_tag{

  // Time structures - measure down to microseconds

  struct timeval startFull, endFull; // whole daisy computation - conv,grad,transA(,transB(,transfers))

  struct timeval startConv, endConv; // measure the Gaussian convolutions on the gradients

  struct timeval startConvX, endConvX; // measure the middle convolution in X (for different steps)

  struct timeval startGrad, endGrad; // measure gradients

  struct timeval startConvGrad, endConvGrad; // measure all convolutions and gradient

  struct timeval startTransGrad, endTransGrad; // measure transposition of gradients

  struct timeval startTransDaisy, endTransDaisy; // measure transposition to daisy descriptors, with or without transfers

  struct timeval startTransPinned, endTransPinned; // time to compute a block + transfer it to pinned memory

  struct timeval startTransRam, endTransRam; // time to transfer a block from pinned memory to ram

  struct timeval startFetchDaisy, endFetchDaisy; // time to fetch a daisy descriptor from global to local memory on the gpu

  struct timeval startMatchDaisy, endMatchDaisy; // time for coarse and middle layer matching

  struct timeval startDiffCoarse, endDiffCoarse; // time to do 1 to all diff at coarse spacing, for all 3 petal regions

  struct timeval startDiffTranspose, endDiffTranspose; // time to transpose diff rotations

  struct timeval startReduceCoarse1, endReduceCoarse1; // time to do the first reduce kernel (once per rotation)

  struct timeval startReduceCoarse2, endReduceCoarse2; // time to run the final reduce kernel (once for all rotations)

  struct timeval startDiffMiddle, endDiffMiddle; // time to do middle layer diffs of many points in local regions with middle spacing, for all 3 petal regions

  struct timeval startMatchCpu, endMatchCpu;

  double diffCoarse, transRot, reduceMin, reduceMinAll;

  double transPinned, transRam;

  double startt, endt, difft;

  short int measureDeviceHostTransfers;

  short int displayRuntimes;

  short int enabled;

} time_params;

daisy_params * newDaisyParams(const char *, unsigned char *, int, int, short int);

daisy_params * initDaisy(const char *, short int);

int initOcl(daisy_params *, ocl_constructs *);

int oclDaisy(daisy_params *, ocl_constructs *, time_params *);

void unpadDescriptorArray(daisy_params *);

int oclCleanUp(ocl_daisy_kernels *, ocl_constructs *, int);

int daisyCleanUp(daisy_params *, ocl_constructs *);

void displayTimes(daisy_params *, time_params *);

void saveToBinary(daisy_params *);

double timeDiff(struct timeval start, struct timeval end);
