#include "oclMatchDaisy.h"

long int verifyDiffCoarse(daisy_params * daisyTarget, float * petalArray, 
                          float * targetArray, float * diffArray);

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

  daisy->oclKernels->transposeRotations = clCreateKernel(daisyCl->program, "transposeRotations", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (transposeRotations)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->reduceMin = clCreateKernel(daisyCl->program, "reduceMin", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (reduceMin)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->normaliseRotation = clCreateKernel(daisyCl->program, "normaliseRotation", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (normaliseRotation)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  return error;

}

int oclMatchDaisy(daisy_params * daisyTemplate, daisy_params * daisyTarget, ocl_constructs * daisyCl, time_params * times){

  cl_int error = 0;

  cl_mem templateBuffer = daisyTemplate->buffers[0];
  cl_mem targetBuffer = daisyTarget->buffers[0];

  int coarseWidth  = daisyTarget->paddedWidth  / pow(SUBSAMPLE_RATE,2);
  int coarseHeight = daisyTarget->paddedHeight / pow(SUBSAMPLE_RATE,2);
  int rotationsNo = ROTATIONS_NO;
  int templatePetalsPerRun = 8;

  printf("Matching coarse layer [%dx%d] (subsampled by %d) for %d rotations\n",
         coarseHeight,coarseWidth,(int)pow(SUBSAMPLE_RATE,2),rotationsNo);

  cl_mem diffBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       (coarseWidth * coarseHeight * rotationsNo) * sizeof(float),
                                       (void*)NULL, &error);

  cl_mem diffBufferTrans = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                          (coarseWidth * coarseHeight * rotationsNo) * sizeof(float),
                                          (void*)NULL, &error);

  // The petal pair(s) that will be coarsely matched in the diffCoarse kernel
  cl_mem petalBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                      templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
                                      (void*) NULL, &error);
  cl_mem petalBufferB = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
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

  // Setup diffCoarse kernel
  int workersPerPixel = 32;
  const size_t wgsDiffCoarse[2] = {64, 1};
  const size_t wsDiffCoarse[2] = {coarseWidth * workersPerPixel, coarseHeight};

  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 0, sizeof(petalBuffer), (void*)&petalBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 1, sizeof(targetBuffer), (void*)&targetBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 2, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 3, sizeof(int), (void*)&(daisyTarget->paddedWidth));

  // Setup transposeRotations kernel
  const size_t wgsTransposeRotations[2] = {128, 1};
  const size_t wsTransposeRotations[2] = {coarseWidth * rotationsNo, coarseHeight};

  clSetKernelArg(daisyTemplate->oclKernels->transposeRotations, 0, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->transposeRotations, 1, sizeof(diffBufferTrans), (void*)&diffBufferTrans);
  clSetKernelArg(daisyTemplate->oclKernels->transposeRotations, 2, sizeof(int), (void*)&coarseHeight);
  clSetKernelArg(daisyTemplate->oclKernels->transposeRotations, 3, sizeof(int), (void*)&coarseWidth);

  // Setup reduceMin kernel
  int diffBufferSize = coarseHeight * coarseWidth; // reduce per rotation

  const size_t wgsreduceMin = 256;
  const size_t wsreduceMin = diffBufferSize;

  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 0, sizeof(diffBufferTrans), (void*)&diffBufferTrans);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 1, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 2, sizeof(int), (void*)&diffBufferSize);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 3, sizeof(float) * wgsreduceMin, (void*)NULL);

/*  const size_t wgsNormaliseRotation = 256;
  const size_t wsNormaliseRotation = wsreduceMin;

  clSetKernelArg(daisyTemplate->oclKernels->normaliseRotation, 0, sizeof(diffBufferTrans), (void*)&diffBufferTrans);
  clSetKernelArg(daisyTemplate->oclKernels->normaliseRotation, 1, sizeof(diffBuffer), (void*)&diffBuffer);*/

  // Copy first descriptor petals
  error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBuffer,
  	                          offsetToDescriptor + offsetToCoarse, 0,
                              templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
  	                          0, NULL, NULL);

  error = clFinish(daisyCl->ioqueue);

  gettimeofday(&times->startConv,NULL);

  // Compute diffCoarse
  error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffCoarse, 2, 
                                 NULL, wsDiffCoarse, wgsDiffCoarse, 
                                 0, NULL, NULL);

  if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (diffCoarse)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

/*  
    *** can work async from the kernel run when looping over multiple template points ***

error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBufferB,
  	                          offsetToDescriptor + offsetToCoarse, 0,
                              templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
  	                          0, NULL, NULL);*/

  // Transpose rotations from HxWxR to RxHxW
  error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->transposeRotations, 2, 
                                 NULL, wsTransposeRotations, wgsTransposeRotations,
                                 0, NULL, NULL);

  if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (transposeRotations)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  for(int rotation = 0; rotation < 8; rotation++){

    const size_t wsoreduceMin = rotation * wsreduceMin;

    // Find maximum for HxW of each rotation
    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->reduceMin, 1, 
                                   &wsoreduceMin, &wsreduceMin, &wgsreduceMin,
                                   0, NULL, NULL);

    // Normalise per rotation
//    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->normaliseRotation, 1, 
//                                   &wsoreduceMin, &wsNormaliseRotation, &wgsNormaliseRotation,
//                                   0, NULL, NULL);


  }

  error = clFinish(daisyCl->ioqueue);

  if(oclErrorM("oclMatchDaisy","clFinish",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  gettimeofday(&times->endConv,NULL);
  
  times->difft = timeDiff(times->startConv,times->endConv);

  printf("Match: %.2f ms\n",times->difft);

#ifdef CPU_VERIFICATION
//
// VERIFICATION CODE
//

  float * diffArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);
  float * petalArray = (float*)malloc(sizeof(float) * REGION_PETALS_NO * GRADIENTS_NO);
  float * targetArray = (float*)malloc(sizeof(float) * (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH));

  error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBuffer, CL_TRUE,
                              0, coarseWidth * coarseHeight * rotationsNo * sizeof(float), 
                              diffArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (diffBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, targetBuffer, CL_TRUE,
                              0, (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH) * sizeof(float), 
                              targetArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (targetBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, petalBuffer, CL_TRUE,
                              0, REGION_PETALS_NO * GRADIENTS_NO * sizeof(float), 
                              petalArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (petalBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  long int issues = verifyDiffCoarse(daisyTarget, petalArray, targetArray, diffArray);
  printf("diffCoarse verification: %ld issues\n",issues);

  free(diffArray);
  free(petalArray);
  free(targetArray);

#endif

  clReleaseMemObject(diffBuffer);
  clReleaseMemObject(petalBuffer);
  clReleaseMemObject(diffBufferTrans);

  return error;

}

long int verifyDiffCoarse(daisy_params * daisyTarget, float * petalArray, 
                          float * targetArray, float * diffArray){

  long int shown = 0;
  long int issues = 0;
  int rotationsNo = 8;
  int subsample = 4;
  int coarseWidth = daisyTarget->paddedWidth / subsample;

  for(int targetY = 0; targetY < daisyTarget->paddedHeight; targetY+=subsample){

    for(int targetX = 0; targetX < daisyTarget->paddedWidth; targetX+=subsample){
      
      long int offsetToDescriptor = (targetY * daisyTarget->paddedWidth + targetX) * DESCRIPTOR_LENGTH;
      long int offsetToCoarseRegion = (TOTAL_PETALS_NO - REGION_PETALS_NO) * GRADIENTS_NO;
      float * targetDescriptor = targetArray + offsetToDescriptor;
      float * targetPetal = targetDescriptor + offsetToCoarseRegion;

      for(int rotation = 0; rotation < rotationsNo; rotation++){

        float diff = 0.0;

        int targetPetalNo = rotation;
        int targetGradientNo = rotation;

        for(int p = 0; p < REGION_PETALS_NO; p++){

          for(int g = 0; g < GRADIENTS_NO; g++){

            diff += fabs(petalArray[p * GRADIENTS_NO + g] - 

                         targetPetal[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                      (targetGradientNo + g) % GRADIENTS_NO]);

          }

        }
        float gpudiff = diffArray[((targetY / subsample) * coarseWidth + targetX / subsample) * rotationsNo + rotation];
        if(fabs(gpudiff - diff) > 0.0001){
          issues++;
          if(targetY == 8 && shown++ < 100){
            printf("X,Y,R = %d,%d,%d | CPU = %.3f and GPU = %.3f\n",targetX / subsample,targetY/subsample,rotation,diff,gpudiff);
          }
        }
        else if(0 && shown++ < 200)
          printf("X,Y,R = %d,%d,%d | CPU = %.3f and GPU = %.3f\n",targetX / subsample,targetY/subsample,rotation,diff,gpudiff);

      }

    }

  }

  return issues;

}

