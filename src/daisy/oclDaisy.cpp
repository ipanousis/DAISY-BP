/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : February/2012

  File: oclDaisy.cpp

*/

#include "oclDaisy.h"
#include <omp.h>
#include <sys/time.h>

double timeDiffK(struct timeval start, struct timeval end);

float * generatePetalOffsets(float, int, short int);
int * generateTranspositionOffsets(int, int, float*, int, int*, int*);
long int verifyConvolutionY(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample);
long int verifyConvolutionX(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample);
long int verifyTransposeGradientsPartialNorm(float * inputData, float * outputData, int width, int height, int gradientsNo);
long int verifyTransposeDaisyPairs(daisy_params * daisy, float * transArray, float * daisyArray, float ** allPetalOffsets);

// Maximum width that this TR_BLOCK_SIZE is effective on (assuming at least 
// TR_DATA_WIDTH rows should be allocated per block) is currently 16384

// Input 2D array
#define ARRAY_PADDING 64

// Configuration & special constants for slow kernel
#define TR_BLOCK_SIZE 512*512
#define TR_DATA_WIDTH 16
#define TR_PAIRS_SINGLE_ONLY -999
#define TR_PAIRS_OFFSET_WIDTH 1000

// DAISY configuration
#define SMOOTHINGS_NO 3
#define REGION_PETALS_NO 8
#define TOTAL_PETALS_NO 25

// Padding for aligned memory mapping in fast kernel
#define TRANSD_FAST_PETAL_PADDING 1

daisy_params * newDaisyParams(unsigned char* array, int height, int width,
                              int gradientsNo, int petalsNo, int smoothingsNo){

  daisy_params * params = (daisy_params*) malloc(sizeof(daisy_params));
  params->array = array;
  params->height = height;
  params->width = width;
  params->gradientsNo = gradientsNo;
  params->petalsNo = petalsNo;
  params->smoothingsNo = smoothingsNo;
  params->totalPetalsNo = petalsNo * smoothingsNo + 1;
  params->descriptors = NULL;
  params->descriptorLength = params->totalPetalsNo * gradientsNo;

  return params;
}

void unpadDescriptorArray(daisy_params * daisy){

  float * array = daisy->descriptors;
  int paddingAdded = (ARRAY_PADDING - daisy->height % ARRAY_PADDING) % ARRAY_PADDING;

  if(paddingAdded == 0) return;

  int descriptorLength = (REGION_PETALS_NO * SMOOTHINGS_NO + 1) * NO_GRADIENTS;

  for(int row = 1; row < daisy->height; row++){
    
    int rewind = row * paddingAdded * descriptorLength;

    for(int i = row * daisy->paddedWidth * descriptorLength; i < (row+1) * daisy->paddedWidth * descriptorLength; i++){

      array[i - rewind] = array[i];

    }

  }

}

int oclError(const char * function, const char * functionCall, int error){

  if(error){
    fprintf(stderr, "oclDaisy.cpp::%s %s failed: %d\n",function,functionCall,error);
    return error;
  }
  
  return 0;
}

int oclCleanUp(ocl_daisy_programs * daisy, ocl_constructs * daisyCl, int error){

  // do clean up

  return error;
}

int initOcl(ocl_daisy_programs * daisy, ocl_constructs * daisyCl){

  cl_int error;

  // Prepare/Reuse platform, device, context, command queue
  cl_bool recreateBuffers = 0;

  error = buildCachedConstructs(daisyCl, &recreateBuffers);
  if(oclError("initOcl","buildCachedConstructs",error)) return oclCleanUp(daisy,daisyCl,error);

  // Pass preprocessor build options
  const char options[128] = "-cl-mad-enable -cl-fast-relaxed-math -DFSC=14";

  // Build denoising filter
  error = buildCachedProgram(daisyCl, "daisyKernels.cl", options);
  if(oclError("initOcl","buildCachedProgram",error)) return oclCleanUp(daisy,daisyCl,error);
  
  if(daisyCl->program == NULL){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
    return oclCleanUp(daisy,daisyCl,1);
  }

  // Prepare the kernel
  daisy->kernel_denx = clCreateKernel(daisyCl->program, "convolve_denx", &error);
  daisy->kernel_deny = clCreateKernel(daisyCl->program, "convolve_deny", &error);
  if(oclError("initOcl","clCreateKernel (den)",error)) return oclCleanUp(daisy,daisyCl,error);

  // Build gradient kernel
  daisy->kernel_grad = clCreateKernel(daisyCl->program, "gradients", &error);
  if(oclError("initOcl","clCreateKernel (grad)",error)) return oclCleanUp(daisy,daisyCl,error);
  
  daisy->kernel_G0x = clCreateKernel(daisyCl->program, "convolve_G0x", &error);
  daisy->kernel_G0y = clCreateKernel(daisyCl->program, "convolve_G0y", &error);
  if(oclError("initOcl","clCreateKernel (G0)",error)) return oclCleanUp(daisy,daisyCl,error);
  
  daisy->kernel_G1x = clCreateKernel(daisyCl->program, "convolve_G1x", &error);
  daisy->kernel_G1y = clCreateKernel(daisyCl->program, "convolve_G1y", &error);
  if(oclError("initOcl","clCreateKernel (G1)",error)) return oclCleanUp(daisy,daisyCl,error);
  
  daisy->kernel_G2x = clCreateKernel(daisyCl->program, "convolve_G2x", &error);
  daisy->kernel_G2y = clCreateKernel(daisyCl->program, "convolve_G2y", &error);
  if(oclError("initOcl","clCreateKernel(G2)",error)) return oclCleanUp(daisy,daisyCl,error);

  daisy->kernel_trans = clCreateKernel(daisyCl->program, "transposeGradients", &error);
  if(oclError("initOcl","clCreateKernel (trans)",error)) return oclCleanUp(daisy,daisyCl,error);

  //daisy->kernel_transd = clCreateKernel(daisyCl->program, "transposeDaisy", &error);
  daisy->kernel_transdp = clCreateKernel(daisyCl->program, "transposeDaisyPairs", &error);
  if(oclError("initOcl","clCreateKernel (transdp)",error)) return oclCleanUp(daisy,daisyCl,error);

  return error;
}

int oclDaisy(daisy_params * daisy, ocl_constructs * daisyCl, time_params * times){

  cl_int error;

  daisy->paddedWidth = daisy->width + (ARRAY_PADDING - daisy->width % ARRAY_PADDING) % ARRAY_PADDING;
  daisy->paddedHeight = daisy->height + (ARRAY_PADDING - daisy->height % ARRAY_PADDING) % ARRAY_PADDING;

  float * inputArray = (float*)malloc(sizeof(float) * daisy->paddedWidth * daisy->paddedHeight * 8);

  int windowHeight = TR_DATA_WIDTH;
  int windowWidth  = TR_DATA_WIDTH;

  float sigmas[3] = {SIGMA_A,SIGMA_B,SIGMA_C};

  float * allPetalOffsets[SMOOTHINGS_NO];
  int * allPairOffsets[3];
  int allPairOffsetsLengths[3];
  cl_mem allPairOffsetBuffers[3];

  for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

    int petalsNo = daisy->petalsNo + (smoothingNo==0);

    float * petalOffsets = generatePetalOffsets(sigmas[smoothingNo], daisy->petalsNo, (smoothingNo==0));

    allPetalOffsets[smoothingNo] = petalOffsets;

    int pairOffsetsLength, actualPairs;
    int * pairOffsets = generateTranspositionOffsets(windowHeight, windowWidth,
                                                     petalOffsets, petalsNo,
                                                     &pairOffsetsLength, &actualPairs);


    cl_mem pairOffsetBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             pairOffsetsLength * 4 * sizeof(int), (void*)pairOffsets, &error);

    oclError("oclDaisy","clCreateBuffer (pairOffset)",error);

    allPairOffsets[smoothingNo] = pairOffsets;
    allPairOffsetsLengths[smoothingNo] = pairOffsetsLength;
    allPairOffsetBuffers[smoothingNo] = pairOffsetBuffer;

  }

  //
  // Preparation for daisy transposition parameters and enqueue the time-consuming memory mapping
  //

  int daisyBlockWidth = daisy->paddedWidth;
  int daisyBlockHeight = min(TR_BLOCK_SIZE, daisy->paddedWidth * daisy->paddedHeight) / daisyBlockWidth;

  if(daisyBlockHeight % TR_DATA_WIDTH > 0){

    daisyBlockHeight = max(TR_DATA_WIDTH, (daisyBlockHeight / TR_DATA_WIDTH) * TR_DATA_WIDTH);

  }

  int totalSections = daisy->paddedHeight / daisyBlockHeight;

  // the height of the final block is taken care of just before the computation later on
  if(totalSections * daisyBlockHeight < daisy->paddedHeight) totalSections++;

  unsigned long int daisySectionSize = daisyBlockWidth * daisyBlockHeight * 
                   (daisy->totalPetalsNo + TRANSD_FAST_PETAL_PADDING) * daisy->gradientsNo * sizeof(float);

  //
  // End of parameter preparation
  // 

#ifdef DAISY_HOST_TRANSFER

  cl_mem hostPinnedDaisyDescriptors = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
                                                     daisySectionSize, NULL, &error);


  void * daisyDescriptorsSection = (void*)clEnqueueMapBuffer(daisyCl->queue, hostPinnedDaisyDescriptors, 0,
                                                             CL_MAP_WRITE, 0, daisySectionSize,
                                                             0, NULL, NULL, &error);

  if(totalSections == 1){

    // transfer only to pinned without doing the memcpy to non-pinned
    daisy->descriptors = (float*)daisyDescriptorsSection;

  }
  else{
  
    unsigned long int daisyDescriptorSize = daisy->paddedWidth * daisy->paddedHeight * 
                      (daisy->totalPetalsNo + TRANSD_FAST_PETAL_PADDING) * daisy->gradientsNo * sizeof(float);

    if(daisy->descriptors == NULL){
      daisy->descriptors = (float*)malloc(daisyDescriptorSize);
    }
    else{
      daisy->descriptors = (float*)realloc(daisy->descriptors, daisyDescriptorSize);
    }
    

  }
  //printf("\nBlock Size calculated (HxW): %dx%d\n",daisyBlockHeight,daisyBlockWidth);
  
#endif

  clFinish(daisyCl->queue);

  gettimeofday(&times->startFull,NULL);

  int paddedWidth  = daisy->paddedWidth;
  int paddedHeight = daisy->paddedHeight;

  long int memorySize = daisy->gradientsNo * (daisy->smoothingsNo+1) * 
                        paddedWidth * paddedHeight * sizeof(cl_float);

  cl_mem massBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                     memorySize, (void*)NULL, &error);

  if(oclError("oclDaisy","clCreateBuffer (mass)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  //printf("massBuffer size = %ld (%ldMB)\n", memorySize, memorySize / (1024 * 1024));
  //printf("paddedWidth = %d, paddedHeight = %d\n", paddedWidth, paddedHeight);

  int filterDenSize = 5;
  float * filterDen = (float*)malloc(sizeof(float)*filterDenSize);
  kutility::gaussian_1d(filterDen,filterDenSize,SIGMA_DEN,0);

  int filterG0Size = 11;
  float * filterG0 = (float*)malloc(sizeof(float)*filterG0Size);
  kutility::gaussian_1d(filterG0,filterG0Size,sqrt(SIGMA_A * SIGMA_A - SIGMA_DEN * SIGMA_DEN),0);

  int filterG1Size = 23;
  float * filterG1 = (float*)malloc(sizeof(float)*filterG1Size);
  kutility::gaussian_1d(filterG1,filterG1Size,sqrt(SIGMA_B * SIGMA_B - SIGMA_A * SIGMA_A),0);

  int filterG2Size = 29;
  float * filterG2 = (float*)malloc(sizeof(float)*filterG2Size);
  kutility::gaussian_1d(filterG2,filterG2Size,sqrt(SIGMA_C * SIGMA_C - SIGMA_B * SIGMA_B),0);

  const int filterOffsets[4] = {0, filterDenSize, filterDenSize + filterG0Size,
                                   filterDenSize + filterG0Size + filterG1Size};

  cl_mem filterBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                       (filterDenSize + filterG0Size + filterG1Size + filterG2Size) * sizeof(float),
                                       (void*)NULL, &error);

  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[0] * sizeof(float), filterDenSize * sizeof(float), (void*)filterDen,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[1] * sizeof(float), filterG0Size * sizeof(float), (void*)filterG0,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[2] * sizeof(float), filterG1Size * sizeof(float), (void*)filterG1,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[3] * sizeof(float), filterG2Size * sizeof(float), (void*)filterG2,
                       0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueWriteBuffer (filters)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  // Pad edges of input array for i) to fit the workgroup size ii) convolution halo - resample nearest pixel
  int i;
  for(i = 0; i < daisy->height; i++){
    int j;
    for(j = 0; j < daisy->width; j++)
      inputArray[i * paddedWidth + j] = daisy->array[i * daisy->width + j];
    for(j = daisy->width; j < paddedWidth; j++)
      inputArray[i * paddedWidth + j] = daisy->array[i * daisy->width + daisy->width-1];
  }
  for(i = daisy->height; i < paddedHeight; i++){
    int j;
    for(j = 0; j < paddedWidth; j++)
      inputArray[i * paddedWidth + j] = daisy->array[(daisy->height-1) * daisy->width + j];
  }

  error = clEnqueueWriteBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                               0, paddedWidth * paddedHeight * sizeof(float),
                               (void*)inputArray,
                               0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueWriteBuffer (inputArray)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  cl_int k;

  gettimeofday(&times->startConvGrad,NULL);

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)
  size_t convWorkerSizeDenx[2] = {daisy->paddedWidth / 4, daisy->paddedHeight};
  size_t convGroupSizeDenx[2] = {16,8};

  // convolve X - A.0 to A.1
  clSetKernelArg(daisy->oclPrograms.kernel_denx, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_denx, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_denx, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_denx, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_denx, 2, NULL, 
                                 convWorkerSizeDenx, convGroupSizeDenx, 0, 
                                 NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (denx)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  error = clFinish(daisyCl->queue);

  // convolve Y - A.1 to B.0
  size_t convWorkerSizeDeny[2] = {daisy->paddedWidth,daisy->paddedHeight / 4};
  size_t convGroupSizeDeny[2] = {16,8};

  clSetKernelArg(daisy->oclPrograms.kernel_deny, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_deny, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_deny, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_deny, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_deny, 2, 
                                 NULL, convWorkerSizeDeny, convGroupSizeDeny, 
                                 0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (deny)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  error = clFinish(daisyCl->queue);

  // Gradients
  size_t gradWorkerSize = daisy->paddedWidth * daisy->paddedHeight;
  size_t gradGroupSize = 64;

  gettimeofday(&times->startGrad,NULL);

  // gradient X,Y,all - B.0 to A.0-7
  clSetKernelArg(daisy->oclPrograms.kernel_grad, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_grad, 1, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_grad, 2, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_grad, 1, NULL, 
                                 &gradWorkerSize, &gradGroupSize, 0, 
                                 NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (grad)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  gettimeofday(&times->endGrad,NULL);
    
  // Smooth all to 2.5 - keep at massBuffer section A
  
  gettimeofday(&times->startConv,NULL);

  // convolve X - massBuffer sections: A to B
  size_t convWorkerSizeG0x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight * daisy->gradientsNo};
  size_t convGroupSizeG0x[2] = {16,4};

  clSetKernelArg(daisy->oclPrograms.kernel_G0x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G0x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G0x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G0x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G0x, 2, NULL, 
                                 convWorkerSizeG0x, convGroupSizeG0x, 0, 
                                 NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G0x)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  // convolve Y - massBuffer sections: B to A
  size_t convWorkerSizeG0y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 8};
  size_t convGroupSizeG0y[2] = {16,8};

  clSetKernelArg(daisy->oclPrograms.kernel_G0y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G0y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G0y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G0y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G0y, 2, NULL, 
                                 convWorkerSizeG0y, convGroupSizeG0y, 0, 
                                 NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G0y)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  // smooth all with size 23 - keep

  gettimeofday(&times->startConvX,NULL);

  // convolve X - massBuffer sections: A to C
  size_t convWorkerSizeG1x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight * daisy->gradientsNo};
  size_t convGroupSizeG1x[2] = {16,4};

  clSetKernelArg(daisy->oclPrograms.kernel_G1x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G1x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G1x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G1x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G1x, 2, NULL, 
                                 convWorkerSizeG1x, convGroupSizeG1x, 0, 
                                 NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G1x)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  gettimeofday(&times->endConvX,NULL);

  // convolve Y - massBuffer sections: C to B
  size_t convWorkerSizeG1y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 4};
  size_t convGroupSizeG1y[2]  = {16, 16};

  clSetKernelArg(daisy->oclPrograms.kernel_G1y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G1y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G1y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G1y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G1y, 2, 
                                 NULL, convWorkerSizeG1y, convGroupSizeG1y,
                                 0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G1y)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  // smooth all with size 29 - keep
  
  // convolve X - massBuffer sections: B to D
  size_t convWorkerSizeG2x[2] = {daisy->paddedWidth / 4, (daisy->paddedHeight * daisy->gradientsNo)};
  size_t convGroupSizeG2x[2]  = {16, 4};

  clSetKernelArg(daisy->oclPrograms.kernel_G2x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G2x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G2x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G2x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G2x, 2, 
                                 NULL, convWorkerSizeG2x, convGroupSizeG2x, 
                                 0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G2x)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);
  
  // convolve Y - massBuffer sections: D to C
  size_t convWorkerSizeG2y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 4};
  size_t convGroupSizeG2y[2]  = {16, 16};

  clSetKernelArg(daisy->oclPrograms.kernel_G2y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G2y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_G2y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_G2y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_G2y, 2, 
                                 NULL, convWorkerSizeG2y, convGroupSizeG2y, 
                                 0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (G2y)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  gettimeofday(&times->endConvGrad,NULL);

  gettimeofday(&times->endConv,NULL);

  // A) transpose SxGxHxW to SxHxWxG first

  gettimeofday(&times->startTransGrad,NULL);

  memorySize = daisy->gradientsNo * daisy->smoothingsNo * 
               paddedWidth * paddedHeight * sizeof(cl_float);

  cl_mem transBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                      memorySize, (void*)NULL, &error);

  if(oclError("oclDaisy","clCreateBuffer (trans)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);
  
  size_t transWorkerSize[2] = {daisy->paddedWidth,daisy->paddedHeight * daisy->smoothingsNo * daisy->gradientsNo};
  size_t transGroupSize[2] = {32,8};

  clSetKernelArg(daisy->oclPrograms.kernel_trans, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 1, sizeof(transBuffer), (void*)&transBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_trans, 2, 
                                 NULL, transWorkerSize, transGroupSize,
                                 0, NULL, NULL);

  if(oclError("oclDaisy","clEnqueueNDRangeKernel (trans)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  clFinish(daisyCl->queue);

  clReleaseMemObject(massBuffer);

  gettimeofday(&times->endTransGrad,NULL);

  // B) final transposition

  gettimeofday(&times->startTransDaisy,NULL);
  
  cl_mem daisyBufferA, daisyBufferB;

  daisyBufferA = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY,
                                daisySectionSize,(void*)NULL, &error);

  if(oclError("oclDaisy","clCreateBuffer (daisyBufferA)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  if(totalSections > 1)
    daisyBufferB = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY,
                                  daisySectionSize,(void*)NULL, &error);

  if(oclError("oclDaisy","clCreateBuffer (daisyBufferB)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  size_t daisyGroupSize[2] = {64,1};

  cl_event * memoryEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections);
  cl_event * kernelEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections * daisy->smoothingsNo * REGION_PETALS_NO);

  error = clFinish(daisyCl->queue);
  if(oclError("oclDaisy","clFinish (pre transdp)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

  // For each 512x512 section
    // For each sigma
  for(int sectionNo = 0; sectionNo < totalSections; sectionNo++){

    int sectionY = sectionNo;
    int sectionX = 0;

    int sectionWidth = daisyBlockWidth;
    int sectionHeight = (sectionNo < totalSections-1 ? daisyBlockHeight : 
                        (daisy->paddedHeight % daisyBlockHeight ? daisy->paddedHeight % daisyBlockHeight : daisyBlockHeight));

    // undefined behaviour when sectionHeight is not a multiple of 16!!
    int TRANSD_FAST_STEPS = 4;
    size_t daisyWorkerSize[2] = {(sectionWidth * daisy->gradientsNo * 2) / TRANSD_FAST_STEPS, sectionHeight};
      
    short int resourceContext = sectionNo%2;

    cl_event * prevMemoryEvents = NULL;
    cl_event * currMemoryEvents = &memoryEvents[sectionNo];
    cl_event * prevKernelEvents = NULL;
    cl_event * currKernelEvents = &kernelEvents[sectionNo * (REGION_PETALS_NO * SMOOTHINGS_NO) / 2];

    if(resourceContext != sectionNo){
      prevMemoryEvents = &memoryEvents[sectionNo-2];
      prevKernelEvents = &kernelEvents[(sectionNo-2) * daisy->smoothingsNo];
    }

    gettimeofday(&times->startTransPinned,NULL);

    cl_mem daisyBuffer = (!resourceContext ? daisyBufferA : daisyBufferB);

    printf("Worker size %d,%d\n",daisyWorkerSize[0],daisyWorkerSize[1]);

    for(int petalNo = 0; petalNo < REGION_PETALS_NO * SMOOTHINGS_NO; petalNo+=2){

      //int petalTwoOffset = offset from region pair petal 2 from petal 1;
      //int petalOutOffset = offset from petal 1 to target descriptor in number of petals;

      int petalRegion = petalNo / REGION_PETALS_NO;
      int petalOne = petalNo % REGION_PETALS_NO + (petalRegion == 0);
      int petalTwo = petalOne+1;

      int petalOneY = round(allPetalOffsets[petalRegion][petalOne*2]);
      int petalOneX = round(allPetalOffsets[petalRegion][petalOne*2+1]);

      int petalTwoY = round(allPetalOffsets[petalRegion][petalTwo*2]);
      int petalTwoX = round(allPetalOffsets[petalRegion][petalTwo*2+1]);

      //int petalTwoOffset = (petalTwoY - petalOneY) * daisy->paddedWidth + (petalTwoX - petalOneX);
      int petalTwoOffset = (petalTwoY - petalOneY) * TR_PAIRS_OFFSET_WIDTH +\
                           TR_PAIRS_OFFSET_WIDTH / 2 + (petalTwoX - petalOneX);
      int petalTwoOffY = (petalTwoY - petalOneY);
      int petalTwoOffX = (petalTwoX - petalOneX);

      int petalOutOffset = (-petalOneY * daisy->paddedWidth - petalOneX) * daisy->totalPetalsNo;
      petalOutOffset += (petalOutOffset < 0 ? -(petalNo + 1) : (petalNo + 1));

      size_t daisyWorkerOffsets[2] = {sectionX * daisyBlockWidth, petalRegion * daisy->paddedHeight + sectionY * daisyBlockHeight};

//      printf("\nWorkerSize X,Y: %d,%d || WorkerOffset X,Y: %d,%d || PetalOutOffset: %d",
//             daisyWorkerSize[0],daisyWorkerSize[1],daisyWorkerOffsets[0],daisyWorkerOffsets[1],petalOutOffset);

      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 0, sizeof(transBuffer), (void*)&transBuffer);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 1, sizeof(daisyBufferA), (void*)&daisyBufferA);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 2, sizeof(int), (void*)&paddedWidth);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 3, sizeof(int), (void*)&paddedHeight);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 4, sizeof(int), (void*)&petalTwoOffY);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 5, sizeof(int), (void*)&petalTwoOffX);
      clSetKernelArg(daisy->oclPrograms.kernel_transdp, 6, sizeof(int), (void*)&petalOutOffset);

      error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_transdp, 2,
                                     daisyWorkerOffsets, daisyWorkerSize, daisyGroupSize,
                                     (resourceContext!=sectionNo),
                                     prevMemoryEvents,
                                     &currKernelEvents[petalNo / 2]);

      if(oclError("oclDaisy","clEnqueueNDRangeKernel (daisy block)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

      error = clFinish(daisyCl->queue);
      char * finishstr = (char*)malloc(sizeof(char) * 200);
      sprintf(finishstr,"clFinish (end block:%d)",petalNo);
      if(oclError("oclDaisy",finishstr,error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

    }

#ifdef DAISY_HOST_TRANSFER
    int y;

    if(sectionNo > 0){


      int descriptorsOffset = ((sectionY-1) * daisyBlockHeight * daisyBlockWidth + 
                               sectionX * daisyBlockWidth) * daisy->totalPetalsNo * daisy->gradientsNo;

      clWaitForEvents(1, currMemoryEvents-1);

      gettimeofday(&times->startTransRam,NULL);

      #pragma omp parallel for private(y)
      for(y = 0; y < daisyBlockHeight; y++)
        memcpy(daisy->descriptors + descriptorsOffset + y * daisyBlockWidth * 200, 
               (float*)daisyDescriptorsSection + y * daisyBlockWidth * 200, 
               daisyBlockWidth * 200 * sizeof(float));
        
      gettimeofday(&times->endTransRam,NULL);

      times->transRam += timeDiffK(times->startTransRam,times->endTransRam);
    }

    error = clEnqueueReadBuffer(daisyCl->queue, daisyBuffer, 0,
                                0, daisySectionSize, daisyDescriptorsSection,
                                3, currKernelEvents, &currMemoryEvents[0]);


    if(oclError("oclDaisy","clEnqueueReadBuffer (daisyBuffer)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);

    if(sectionNo == totalSections-1 && sectionNo > 0){

      int descriptorsOffset = (sectionY * daisyBlockHeight * daisyBlockWidth + 
                               sectionX * daisyBlockWidth) * daisy->totalPetalsNo * daisy->gradientsNo;

      clWaitForEvents(1, currMemoryEvents);

      gettimeofday(&times->startTransRam,NULL);

      #pragma omp parallel for private(y)
      for(y = 0; y < sectionHeight; y++)
        memcpy(daisy->descriptors + descriptorsOffset + y * daisyBlockWidth * 200, 
               (float*)daisyDescriptorsSection + y * daisyBlockWidth * 200, 
               daisyBlockWidth * 200 * sizeof(float));

      gettimeofday(&times->endTransRam,NULL);

      times->transRam += timeDiffK(times->startTransRam,times->endTransRam);
    }

#else
    
    currMemoryEvents[0] = currKernelEvents[2];

#endif

  /*
    //
    // VERIFICATION CODE
    //

      clFinish(daisyCl->queue);

      long int issues = verifyTransposeDaisyPairs(daisy,transArray,daisyArray,allPetalOffsets);

      if(issues > 0)
        fprintf(stderr,"Got %ld issues with transposeDaisyPairs\n",issues);
  */

  }

  gettimeofday(&times->endTransDaisy,NULL);

  times->transPinned += timeDiffK(times->startTransDaisy,times->endTransDaisy) - times->transRam;

  gettimeofday(&times->endFull,NULL);

  times->difft = timeDiffK(times->startFull,times->endFull);

  // Release and unmap buffers, free allocated space
#ifdef DAISY_HOST_TRANSFER

  // don't unmap the pinned memory if there was only one block
  //if(totalSections > 1){

  // Uncomment when calling oclDaisy multiple times
  error = clEnqueueUnmapMemObject(daisyCl->queue, hostPinnedDaisyDescriptors, daisyDescriptorsSection, 0, NULL, NULL);	

  if(oclError("oclDaisy","clEnqueueUnmapMemObject (pinned daisy)",error)) return oclCleanUp(&daisy->oclPrograms,daisyCl,error);
  //}

  // Uncomment when calling oclDaisy multiple times
  clReleaseMemObject(hostPinnedDaisyDescriptors);

  //free(daisyDescriptorsSection);

#endif

  clReleaseMemObject(daisyBufferA);
  if(totalSections > 1) clReleaseMemObject(daisyBufferB);
  clReleaseMemObject(transBuffer);
  clReleaseMemObject(filterBuffer);

  for(int s = 0; s < daisy->smoothingsNo; s++){
    clReleaseMemObject(allPairOffsetBuffers[s]);
    free(allPairOffsets[s]);
  }

  free(inputArray);

  free(filterDen);
  free(filterG0);
  free(filterG1);
  free(filterG2);
  free(memoryEvents);
  free(kernelEvents);

  return error;
}

// Generates the offsets to points in the circular petal region
// of sigma * 2 in petalsNo directions
float* generatePetalOffsets(float sigma, int petalsNo, short int firstRegion){

  if(firstRegion != 0 && firstRegion != 1) return NULL;

  float regionRadius = sigma * 2;
  float * petalOffsets = (float*)malloc(sizeof(float) * (petalsNo + firstRegion) * 2);

  if(firstRegion){
    petalOffsets[0] = 0;
    petalOffsets[1] = 0;
  }

  int i;
  for(i = firstRegion; i < petalsNo+firstRegion; i++){
    petalOffsets[i*2]   = regionRadius * sin((i-firstRegion) * (M_PI / 4));
    petalOffsets[i*2+1] = regionRadius * cos((i-firstRegion) * (M_PI / 4));
  }

  return petalOffsets;
}

// Generates pairs of neighbouring destination petal points given;
// window dimensions of local data
// offsets of a petal region
// the number of those offsets (here 8)
//
// O offset = Y * WIDTH + WIDTH/2 + X
// with; Y = [-maxOffsetY,windowHeight+maxOffsetY]  
//       X = [-maxOffsetX,windowWidth+maxOffsetX]
//       WIDTH = TR_PAIRS_OFFSET_WIDTH (special, artificially large, width to be 
//                                      able to encode negative x values in the offset)
//    decode X,Y again from O;
//    Y = floor(O / WIDTH)
//    X = O - Y * WIDTH - WIDTH/2
//
// recovery;
//    k = floor(pairedOffsets[currentPair * pairIngredients+2] / (float)TR_PAIRS_OFFSET_WIDTH);
//    l = pairedOffsets[currentPair * pairIngredients+2] - k * TR_PAIRS_OFFSET_WIDTH - TR_PAIRS_OFFSET_WIDTH/2;
//
// Current pair ratios;
// S=2.5: 812/1492 = 0.54
// S=5  : 633/1415 = 0.44
// S=7.5: 417/1631 = 0.25
//
int* generateTranspositionOffsets(int windowHeight, int windowWidth,
                                  float*  petalOffsets,
                                  int     petalsNo,
                                  int*    pairedOffsetsLength,
                                  int*    numberOfActualPairs){

  const int DEBUG = 0;

  // all offsets will be 2d array of;
  // (windowHeight+2 * maxY(petalregionoffsets)) X 
  // (windowWidth +2 * maxX(petalRegionOffsets))

  float maxOffsetY = 0;
  float maxOffsetX = 0;
  int i;
  for(i = 0; i < petalsNo; i++){
    maxOffsetY = (fabs(petalOffsets[i*2]   > maxOffsetY) ? fabs(petalOffsets[i*2])  :maxOffsetY);
    maxOffsetX = (fabs(petalOffsets[i*2+1] > maxOffsetX) ? fabs(petalOffsets[i*2+1]):maxOffsetX);
  }
  maxOffsetY = ceil(fabs(maxOffsetY));
  maxOffsetX = ceil(fabs(maxOffsetX));

  int offsetsHeight = windowHeight + 2 * maxOffsetY;
  int offsetsWidth  = windowWidth  + 2 * maxOffsetX;

  int * allSources = (int*)malloc(sizeof(int) * offsetsHeight * offsetsWidth
                                              * petalsNo * 2);

  const int noSource = -999;

  // generate em
  int x,y,j;
  j = 0;
  for(y = -maxOffsetY; y < windowHeight+maxOffsetY; y++){
    for(x = -maxOffsetX; x < windowWidth+maxOffsetX; x++){
      int fromY,fromX;
      for(i = 0; i < petalsNo; i++,j++){
        fromY = round(y+petalOffsets[i*2]);
        fromX = round(x+petalOffsets[i*2+1]);
        if(fromY >= 0 && fromY < windowHeight && fromX >= 0 && fromX < windowWidth){
          allSources[j*2] = fromY;
          allSources[j*2+1] = fromX;
        }
        else{
          allSources[j*2] = noSource;
          allSources[j*2+1] = noSource;
        }
        if(DEBUG){
          if(y == 0 && x == 0)
            printf("Offsets at %d,%d,%d: (%d,%d)\n",y,x,i,allSources[j*2],allSources[j*2+1]);
        }
      }
    }
  }

  int pairIngredients = 4;
  int * pairedOffsets = (int*)malloc(sizeof(int) * offsetsHeight * offsetsWidth
                                                 * petalsNo * pairIngredients);

  // pair em up
  const int isLeftOver = 1;

  char * singlesLeftOver = (char*)malloc(sizeof(char) * offsetsHeight * offsetsWidth * petalsNo);

  for(i = 0; i < offsetsHeight * offsetsWidth * petalsNo; i++)
    singlesLeftOver[i] = !isLeftOver; // initialise value

  int currentPair = 0;
  int thisSourceY,thisSourceX;
  int nextSourceY,nextSourceX;
  j = 0;
  for(y = 0; y < offsetsHeight; y++){
    for(i = 0; i < offsetsWidth*petalsNo-1; i++){
      j = y*offsetsWidth*petalsNo+i;
      x = i / petalsNo;


      thisSourceY = allSources[j*2];
      thisSourceX = allSources[j*2+1];
      nextSourceY = allSources[j*2+2];
      nextSourceX = allSources[j*2+3];

      if(i % petalsNo == petalsNo-1){ // can't pair up last with first, the sets of values in the destination array are not continuous
        if(thisSourceY != noSource)
          singlesLeftOver[j] = isLeftOver;
        continue;
      }
      if(DEBUG){
        if(y == 15)
          printf("(Y %d,X %d): (sourceY,sourceX),(nextY,nextX) = (%d,%d),(%d,%d)\n",y,x,thisSourceY,thisSourceX,nextSourceY,nextSourceX);
      }
      if(thisSourceY != noSource && nextSourceY != noSource){
        pairedOffsets[currentPair * pairIngredients]   = thisSourceY * windowWidth + thisSourceX; // p1
        pairedOffsets[currentPair * pairIngredients+1] = nextSourceY * windowWidth + nextSourceX; // p2
        pairedOffsets[currentPair * pairIngredients+2] = (y-maxOffsetY) * TR_PAIRS_OFFSET_WIDTH +\
                                                         TR_PAIRS_OFFSET_WIDTH/2 + (x-maxOffsetX); // o, special 1D offset in image coordinates
        pairedOffsets[currentPair * pairIngredients+3] = i % petalsNo; // petal
        if(DEBUG){
          if(y == 15)
            printf("Pair %d: (p1,p2,o,petal) = (%d,%d,%d,%d)\n",currentPair,pairedOffsets[currentPair * pairIngredients],
                                                                            pairedOffsets[currentPair * pairIngredients+1],
                                                                            pairedOffsets[currentPair * pairIngredients+2],
                                                                            pairedOffsets[currentPair * pairIngredients+3]);
        }
        currentPair++;
        i++;
      }
      else if(thisSourceY != noSource && nextSourceY == noSource){
        singlesLeftOver[j] = isLeftOver;
      }
    }
    if(i == offsetsWidth*petalsNo-1){ // ie if the last data was not paired up
      j = y*offsetsWidth*petalsNo+i-1;
      if(allSources[j*2] != noSource)
        singlesLeftOver[j] = isLeftOver;
    }
  }
  
  *numberOfActualPairs = currentPair;

  for(j = 0; j < offsetsHeight*offsetsWidth*petalsNo; j++){

    if(singlesLeftOver[j] == isLeftOver){
      y = j / (offsetsWidth * petalsNo);
      x = (j % (offsetsWidth * petalsNo)) / petalsNo;
      thisSourceY = allSources[j*2];
      thisSourceX = allSources[j*2+1];
      pairedOffsets[currentPair * pairIngredients] = thisSourceY * windowWidth + thisSourceX; // p1
      pairedOffsets[currentPair * pairIngredients+1] = TR_PAIRS_SINGLE_ONLY;
      pairedOffsets[currentPair * pairIngredients+2] = (y-maxOffsetY) * TR_PAIRS_OFFSET_WIDTH +\
                                                       TR_PAIRS_OFFSET_WIDTH/2 + (x-maxOffsetX); // o, the special 1d offset
      pairedOffsets[currentPair * pairIngredients+3] = j % petalsNo;
      if(DEBUG){
        if(x == 15){
          printf("Single %d (%.0f,%.0f): (p1,p2,o,petal) = (%d,%d,%d,%d)\n",currentPair,y-maxOffsetY,x-maxOffsetX,
                                                                        pairedOffsets[currentPair * pairIngredients],
                                                                        pairedOffsets[currentPair * pairIngredients+1],
                                                                        pairedOffsets[currentPair * pairIngredients+2],
                                                                        pairedOffsets[currentPair * pairIngredients+3]);
          int k,l;
          k = floor(pairedOffsets[currentPair * pairIngredients+2] / (float)TR_PAIRS_OFFSET_WIDTH);
          l = pairedOffsets[currentPair * pairIngredients+2] - k * TR_PAIRS_OFFSET_WIDTH - TR_PAIRS_OFFSET_WIDTH/2;
          printf("Recovered Y,X = %d,%d\n",k,l);
        }
      }
      currentPair++;
    }
  }

  free(allSources);
  free(singlesLeftOver);

  *pairedOffsetsLength = currentPair;

  return pairedOffsets;
}

long int verifyTransposeDaisyPairs(daisy_params * daisy, float * transArray, float * daisyArray, float ** allPetalOffsets){

  long int issues = 0;
  printf("test seg 01\n");
  for(int petalNo = 0; petalNo < REGION_PETALS_NO * SMOOTHINGS_NO; petalNo+=2){

//    int petalTwoOffset = offset from region pair petal 2 from petal 1;
//    int petalOutOffset = offset from petal 1 to target descriptor in number of petals;

    int petalRegion = petalNo / REGION_PETALS_NO;
    int petalOne = petalNo % REGION_PETALS_NO + (petalRegion == 0);
    int petalTwo = petalOne+1;

    int petalOneY = round(allPetalOffsets[petalRegion][petalOne*2]);
    int petalOneX = round(allPetalOffsets[petalRegion][petalOne*2+1]);

    int petalTwoY = round(allPetalOffsets[petalRegion][petalTwo*2]);
    int petalTwoX = round(allPetalOffsets[petalRegion][petalTwo*2+1]);

    printf("Petals: 1 = (%d,%d), 2 = (%d,%d)\n",petalOneY,petalOneX,petalTwoY,petalTwoX);

//    int petalTwoOffset = (petalTwoY - petalOneY) * daisy->paddedWidth + (petalTwoX - petalOneX);
//        pairedOffsets[currentPair * pairIngredients+2] = (y-maxOffsetY) * TR_PAIRS_OFFSET_WIDTH +\
//                                                         TR_PAIRS_OFFSET_WIDTH/2 + (x-maxOffsetX); // o, special 1D offset in image coordinates
    int petalTwoOffset = (petalTwoY - petalOneY) * TR_PAIRS_OFFSET_WIDTH +\
                         TR_PAIRS_OFFSET_WIDTH / 2 + (petalTwoX - petalOneX);

    int petalOutOffset = (-petalOneY * daisy->paddedWidth - petalOneX) * daisy->totalPetalsNo;
    petalOutOffset += (petalOutOffset < 0 ? -(petalNo + 1) : (petalNo + 1));

    for(int y = petalRegion * daisy->paddedHeight; y < (petalRegion+1) * daisy->paddedHeight; y++){
      
      for(int x = 0; x < daisy->paddedWidth; x++){

//    k = floor(pairedOffsets[currentPair * pairIngredients+2] / (float)TR_PAIRS_OFFSET_WIDTH);
//    l = pairedOffsets[currentPair * pairIngredients+2] - k * TR_PAIRS_OFFSET_WIDTH - TR_PAIRS_OFFSET_WIDTH/2;
        int petalTwoOffsetY = floor(petalTwoOffset / (float) TR_PAIRS_OFFSET_WIDTH);
        int petalTwoOffsetX = petalTwoOffset - petalTwoOffsetY * TR_PAIRS_OFFSET_WIDTH - TR_PAIRS_OFFSET_WIDTH / 2;

        if(petalTwoOffsetY != (petalTwoY-petalOneY) || petalTwoOffsetX != (petalTwoX-petalOneX)){
          fprintf(stderr,"petalTwoOffset alarm in YX!!!\n");
          fprintf(stderr,"Input YX (%d,%d) gone to %d and output YX (%d,%d)\n",(petalTwoY-petalOneY),(petalTwoX-petalOneX),petalTwoOffset,petalTwoOffsetY,petalTwoOffsetX);
        }
        if((petalOutOffset / TOTAL_PETALS_NO) / daisy->paddedWidth != -petalOneY)
          fprintf(stderr,"petalOutOffset alarm in Y!!!\n");
        if((petalOutOffset / TOTAL_PETALS_NO) % daisy->paddedWidth != -petalOneX)
          fprintf(stderr,"petalOutOffset alarm in X!!!\n");
        if(abs(petalOutOffset % TOTAL_PETALS_NO) != (petalNo+1))
          fprintf(stderr,"petalOutOffset alarm in PETAL!!!\n");

        int targetY = y % daisy->paddedHeight + (petalOutOffset / TOTAL_PETALS_NO) / daisy->paddedWidth;
        int targetX = x + (petalOutOffset / TOTAL_PETALS_NO) % daisy->paddedWidth;

        if(targetY < 0 || targetY >= daisy->paddedHeight || targetX < 0 || targetX >= daisy->paddedWidth) continue;

        float cpuVal = .0f;
        float gpuVal = .0f;
        for(int k = 0; k < daisy->gradientsNo; k++){
          cpuVal += transArray[(y * daisy->paddedWidth + x) * daisy->gradientsNo + k];

          gpuVal += daisyArray[((targetY * daisy->paddedWidth + targetX) * (TOTAL_PETALS_NO + TRANSD_FAST_PETAL_PADDING) + abs(petalOutOffset % TOTAL_PETALS_NO) + TRANSD_FAST_PETAL_PADDING) * daisy->gradientsNo + k];

        }

        if(fabs(gpuVal - cpuVal) > 0.01){
          if(issues++ < 100){
            fprintf(stderr,"daisyTransPairs issue (1): from (%d,%d) to (%d,%d,%d) - %f\n",y,x,targetY,targetX,abs(petalOutOffset % TOTAL_PETALS_NO),fabs(gpuVal - cpuVal));
          }
        }

        cpuVal = .0f;
        gpuVal = .0f;
        int y2 = y % daisy->paddedHeight + petalTwoOffsetY;
        int x2 = x + petalTwoOffsetX;
        if(y2 < 0 || y2 >= daisy->paddedHeight || x2 < 0 || x2 >= daisy->paddedWidth){}
        else{ 
          y2 = y + petalTwoOffsetY;
          for(int k = 0; k < daisy->gradientsNo; k++){
            cpuVal += transArray[(y2 * daisy->paddedWidth + x2) * daisy->gradientsNo + k];
          }
        }

        for(int k = 0; k < daisy->gradientsNo; k++)
          gpuVal += daisyArray[((targetY * daisy->paddedWidth + targetX) * (TOTAL_PETALS_NO + TRANSD_FAST_PETAL_PADDING) + abs(petalOutOffset % TOTAL_PETALS_NO) + 1 + TRANSD_FAST_PETAL_PADDING) * daisy->gradientsNo + k];

        if(fabs(gpuVal - cpuVal) > 0.01){
          if(issues++ < 100){
            fprintf(stderr,"daisyTransPairs issue (2): from (%d,%d) to (%d,%d,%d) - %f\n",y,x,targetY,targetX,abs(petalOutOffset % TOTAL_PETALS_NO)+1,fabs(gpuVal - cpuVal));
          }
        }

      }

    }

  }

  return issues;

}

double timeDiffK(struct timeval start, struct timeval end){

  return (end.tv_sec+(end.tv_usec/1000000.0)) - (start.tv_sec+(start.tv_usec/1000000.0));

}

/*

void unused_old_kernel_verification_code(){

  //
  // VERIFICATION CODE
  //

  int issues = 0;

  // Verify transposition b)
  int TESTING_TRANSD = times->measureDeviceHostTransfers;
  if(DEBUG_ALL && TESTING_TRANSD){


    for(int block = 0; block < totalSections; block++){

      int sectionY = block;
      int sectionX = 0;

      int sectionWidth = daisyBlockWidth;
      int sectionHeight = (block < totalSections-1 ? daisyBlockHeight : 
                          (daisy->paddedHeight % daisyBlockHeight ? daisy->paddedHeight % daisyBlockHeight : daisyBlockHeight));

      for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

        int pairOffsetsLength = allPairOffsetsLengths[smoothingNo];
        int * pairOffsets = allPairOffsets[smoothingNo];

        int petalsNo = daisy->petalsNo + (smoothingNo==0);

        int petalStart = (smoothingNo > 0 ? smoothingNo * daisy->petalsNo + 1 : 0);

        //printf("\nPetalsNo = %d\n",petalsNo);
        float * daisyArray = daisy->descriptors + block * daisyBlockHeight * daisyBlockWidth * 200;

        clFinish(daisyCl->queue);

        error = clEnqueueReadBuffer(daisyCl->queue, transBuffer, CL_TRUE,
                                    paddedWidth * paddedHeight * 8 * smoothingNo * sizeof(float), 
                                    paddedWidth * paddedHeight * 8 * sizeof(float), inputArray,
                                    0, NULL, NULL);

        clFinish(daisyCl->queue);

        short int issued = 0;
  
        int topLeftY = 16;
        int topLeftX = 16;
        int yStep = 16;
        int xStep = 16;
        int y,x;
        int yBlockOffset = sectionY * daisyBlockHeight;
        int xBlockOffset = sectionX * daisyBlockWidth;
        for(y = topLeftY; y < sectionHeight-topLeftY; y+=yStep){
          for(x = topLeftX; x < sectionWidth-topLeftX; x+=xStep){
            //printf("Testing yx %d,%d\n",y,x);
            int p;
            for(p = 0; p < pairOffsetsLength; p++){
              int src1 = (yBlockOffset + pairOffsets[p * 4] / 16 + y) * paddedWidth * 8 + (xBlockOffset + pairOffsets[p * 4] % 16 + x) * 8;
              int src2 = (yBlockOffset + pairOffsets[p * 4+1] / 16 + y) * paddedWidth * 8 + (xBlockOffset + pairOffsets[p * 4+1] % 16 + x) * 8;
              int dst  = floor(pairOffsets[p * 4+2] / 1000.0f);
              int petal = pairOffsets[p * 4+3];
              dst = (dst + y) * daisyBlockWidth * 200 + (pairOffsets[p * 4+2] - dst * 1000 - 500 + x) * 200 + (petalStart+petal) * 8;
              int j;
              for(j = 0; j < 8; j++){
                if(fabs(daisyArray[dst+j] - inputArray[src1+j]) > 0.00001){
                  if(!issued)
                    printf("Issue at section %d,%d S=%d(1)\n",sectionY,sectionX,smoothingNo);
                  //printf("P%d - Issue at (1)%d,%d\n",p,y,x);
                  issued=1;
                  issues++;
                }
              }
              if(pairOffsets[p * 4+1] != TR_PAIRS_SINGLE_ONLY){
                for(j = 8; j < 16; j++){
                  if(fabs(daisyArray[dst+j] - inputArray[src2+j%8]) > 0.00001){
                    if(!issued)
                      printf("Issues at section %d,%d S=%d(2)\n",sectionY,sectionX,smoothingNo);
                    //printf("P%d - Issue at (2)%d,%d\n",p,y,x);
                    issued=1;
                  }
                }
              }
            }
          }
        }
      }
    }
    printf("%d issues\n",issues);
  }

  //
  // END OF VERIFICATION CODE
  //

}

*/

