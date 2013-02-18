#include "oclMatchDaisy.h"

int oclErrorM(const char * function, const char * functionCall, int error){

  if(error){
    fprintf(stderr, "oclMatchDaisy.cpp::%s %s failed: %d\n",function,functionCall,error);
    return error;
  }
  
  return 0;

}

int initOclMatch(daisy_params * daisy, ocl_constructs * daisyCl){

  cl_int error;

  if(daisyCl->platformId == NULL){

    // Prepare/Reuse platform, device, context, command queue
    cl_bool recreateBuffers = 0;

    error = buildCachedConstructs(daisyCl, &recreateBuffers);
    if(oclErrorM("initOclMatch","buildCachedConstructs",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

    // Pass preprocessor build options
    const char options[128] = "-cl-mad-enable -cl-fast-relaxed-math -DFSC=14";

    // Build denoising filter
    error = buildCachedProgram(daisyCl, "daisyMatchKernels.cl", options);
    if(oclErrorM("initOclMatch","buildCachedProgram",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

    if(daisyCl->program == NULL){

      fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
      return oclCleanUp(daisy->oclKernels,daisyCl,1);

    }

  }
  
  // Prepare the kernel  
  daisy->oclKernels->diffCoarse = clCreateKernel(daisyCl->program, "diffCoarse", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (diffCoarse)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  return error;

}

int oclMatchDaisy(daisy_params * daisyTemplate, daisy_params * daisyTarget, ocl_constructs * daisyCl, time_params * times){

  cl_int error = 0;

  cl_mem templateBuffer = daisyTemplate->buffers[0];
  cl_mem targetBuffer = daisyTarget->buffers[0];

  int coarseWidth  = daisyTarget->paddedWidth  / pow(SUBSAMPLE_RATE,2);
  int coarseHeight = daisyTarget->paddedHeight / pow(SUBSAMPLE_RATE,2);
  int rotationsNo = ROTATIONS_NO;
  int templatePetalsPerRun = 4;

  printf("Matching coarse layer [%dx%d] (subsampled by %d) for %d rotations\n",
         coarseHeight,coarseWidth,(int)pow(SUBSAMPLE_RATE,2),rotationsNo);

  cl_mem diffBuffer = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY,
                                       (coarseWidth * coarseHeight * rotationsNo) * sizeof(float),
                                       (void*)NULL, &error);

  // The petal pair(s) that will be coarsely matched in the diffCoarse kernel
  cl_mem petalBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                      templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
                                      (void*) NULL, &error);

  unsigned int offsetToCoarse = (TOTAL_PETALS_NO - REGION_PETALS_NO) * GRADIENTS_NO;

  // from template; take a list of pixels to match;
  // start with centre pixel
  unsigned int pixelY, pixelX;

  pixelY = daisyTemplate->paddedHeight / 2;
  pixelX = daisyTemplate->paddedWidth / 2;

  unsigned int offsetToDescriptor = (pixelY * daisyTemplate->paddedWidth + pixelX) 
                                    * DESCRIPTOR_LENGTH;

  error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBuffer,
  	                          offsetToDescriptor + offsetToCoarse, 0,
                              templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
  	                          0, NULL, NULL);

  int petalNo = 0;
  int workersPerPixel = 32;
  const size_t groupSizeDiffCoarse[2] = {64, 1};
  const size_t workerSizeDiffCoarse[2] = {coarseWidth * workersPerPixel, coarseHeight};

  clFinish(daisyCl->ioqueue);
  
  gettimeofday(&times->startConv,NULL);

  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 0, sizeof(petalBuffer), (void*)&petalBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 1, sizeof(targetBuffer), (void*)&targetBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 2, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 3, sizeof(int), (void*)&(daisyTarget->paddedWidth));
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 4, sizeof(int), (void*)&petalNo);

  error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffCoarse, 2, NULL,
                                 workerSizeDiffCoarse, groupSizeDiffCoarse, 0,
                                 NULL, NULL);

  if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (diffCoarse)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clFinish(daisyCl->ioqueue);

  if(oclErrorM("oclDaisy","clFinish (diffCoarse)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  gettimeofday(&times->endConv,NULL);
  
  times->difft = timeDiff(times->startConv,times->endConv);

  printf("Match: %.2f ms\n",times->difft);

  clReleaseMemObject(diffBuffer);
  clReleaseMemObject(petalBuffer);

  return error;

}


