#include "oclMatchDaisy.h"

#define VERBOSE 0

long int verifyDiffCoarse(daisy_params * daisyTarget, float * petalArray, 
                          float * targetArray, float * diffArray);
long int verifyReduceMinAll(daisy_params * daisyTarget, float * diffTransArray, float * diffArray, int templatePointsNo);
long int verifyTransposeRotations(daisy_params * daisyTarget, float * diffArray, float * diffTransArray);
long int verifyDiffMiddle(daisy_params * daisyTarget, daisy_params * daisyTemplate, float * targetArray, 
                          float * templateArray, float * diffArray, int searchWidth, float * corrs, int votedRotation,
                          int templatesNo, int rotationsNoMiddle);
void checkpoint(ocl_constructs * daisyCl, struct timeval * s, short int finishQueue);

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
    char * options = (char*) malloc(sizeof(char) * 200);
    sprintf(options, "-cl-mad-enable -cl-fast-relaxed-math -DWGX_MATCH_MIDDLE=%d -DWG_TARGETS_NO=%d -DTARGETS_PER_LOOP=%d", 
                     WGX_MATCH_MIDDLE, WG_TARGETS_NO, TARGETS_PER_LOOP);

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

  daisy->oclKernels->reduceMinAll = clCreateKernel(daisyCl->program, "reduceMinAll", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (reduceMinAll)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->normaliseRotation = clCreateKernel(daisyCl->program, "normaliseRotation", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (normaliseRotation)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->diffMiddle = clCreateKernel(daisyCl->program, "diffMiddle", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (diffMiddle)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  return error;

}

point * generateTemplatePoints(daisy_params * daisy, int templatePointsNo, int xOffset, int yOffset){

  // give back a an equally spaced grid of points on the template
  // do not have points on the edges of the template, use boundary offset of 15
  point * points = (point*) malloc(sizeof(point) * templatePointsNo);

  int boundaryOffset = 15;
  int xRange = daisy->width - boundaryOffset;
  int yRange = daisy->height - boundaryOffset * 2;

  float step = sqrtf((yRange * xRange) / templatePointsNo);

  printf("Seed Grid Spacing = %.2f\n", step);

  float x = boundaryOffset;
  float y = boundaryOffset;
  for(int i = 0; i < templatePointsNo; i++){

    points[i].x = round(x); points[i].y = round(y);

    x += step;

    if(x > xRange){
      x = boundaryOffset;
      y = y + step;
      //y = (y > yRange ? y % step + boundaryOffset : y);
    }
  }

  return points;

}

void saveBinary(float * array, int size, string filename){

  FILE * fp;
  fp = fopen((char*)filename.c_str(), "wb");
  fwrite(array, sizeof(float), size, fp);
  fclose(fp);


}

void checkpoint(ocl_constructs * daisyCl, struct timeval * s, short int finishQueue){

  if(finishQueue)
    clFinish(daisyCl->ioqueue);

  gettimeofday(s, NULL);

}

int oclMatchDaisy(daisy_params * daisyTemplate, daisy_params * daisyTarget,
                  ocl_constructs * daisyCl, time_params * times){

  cl_int error = 0;

  int templatePointsNo = COARSE_TEMPLATES_NO; // default is 16
  point * templatePoints = generateTemplatePoints(daisyTemplate, templatePointsNo, 0, 0);

  cl_mem templateBuffer = daisyTemplate->buffers[0];
  cl_mem targetBuffer = daisyTarget->buffers[0];

  int gridSpacing = pow(SUBSAMPLE_RATE,2);
  int coarseWidth  = daisyTarget->paddedWidth  / gridSpacing;
  int coarseHeight = daisyTarget->paddedHeight / gridSpacing;

  point coarseTargetSize = { (float)coarseWidth, (float)coarseHeight };

  int rotationsNo = ROTATIONS_NO;
  int templatePetalsPerRun = 8;

  int rotationsNoMiddle = MIDDLE_ROTATIONS_NO; // default is 4
  int rotationsNoFine = 1;
  int searchWidthMiddle = MIDDLE_SEARCH_WIDTH; // default is 32
  int searchWidthFine = 8;
  int seedTemplatePointsNo = MIDDLE_TEMPLATES_NO; // default is 512

  int diffBufferSize1 = (coarseWidth * coarseHeight * rotationsNo);
  int diffBufferSize2 = (seedTemplatePointsNo * rotationsNoMiddle * (searchWidthMiddle * searchWidthMiddle));

  printf("\nA) CoarseLayer [%dx%d] - Search Resolution %d - Seeds %d\n",
         coarseHeight,coarseWidth,(int)pow(SUBSAMPLE_RATE,2),templatePointsNo);

  int argminBufferLength = templatePointsNo * rotationsNo * 2;

  cl_mem diffBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       max(diffBufferSize1, diffBufferSize2) * sizeof(float),
                                       (void*)NULL, &error);

  cl_mem diffBufferTrans = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                          (coarseWidth * coarseHeight * rotationsNo) * sizeof(float),
                                          (void*)NULL, &error);

  cl_mem argminBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       argminBufferLength * sizeof(float),
                                       (void*)NULL, &error);

  cl_mem corrsBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       (seedTemplatePointsNo * 2) * sizeof(float),
                                       (void*)NULL, &error);

  // The petal pair(s) that will be coarsely matched in the diffCoarse kernel
  cl_mem petalBufferA = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                       3 * templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
                                       (void*) NULL, &error);
  cl_mem petalBufferB = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                       3 * templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
                                       (void*) NULL, &error);
  cl_mem petalBuffer;

  cl_mem pinnedArgminBuffer = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
                                             argminBufferLength * sizeof(float), NULL, &error);

  if(oclErrorM("oclDaisy","clCreateBuffer (pinnedArgmin)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  float * argmin = (float *) clEnqueueMapBuffer(daisyCl->ioqueue, pinnedArgminBuffer, 1,
                                                CL_MAP_READ, 0, argminBufferLength * sizeof(float),
                                                0, NULL, NULL, &error);

  if(oclErrorM("oclDaisy","clEnqueueMapBuffer (pinnedArgmin)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  // Setup diffCoarse kernel
  int workersPerPixel = 16;
  const size_t wgsDiffCoarse[2] = {64, 1};
  const size_t wsDiffCoarse[2] = {coarseWidth * workersPerPixel, coarseHeight};

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

  const size_t wgsReduceMin = 256;
  const size_t wsReduceMin = diffBufferSize;

  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 0, sizeof(diffBufferTrans), (void*)&diffBufferTrans);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 1, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 2, sizeof(int), (void*)&diffBufferSize);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMin, 3, sizeof(float) * wgsReduceMin, (void*)NULL);

  const size_t wgsReduceMinAll = (coarseHeight * coarseWidth) / wgsReduceMin;
  const size_t wsReduceMinAll = wgsReduceMinAll * rotationsNo;

  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 0, sizeof(diffBufferTrans), (void*)&diffBufferTrans);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 1, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 2, sizeof(argminBuffer), (void*)&argminBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 3, sizeof(int), (void*)&wgsReduceMinAll);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 4, sizeof(float) * wgsReduceMinAll, (void*)NULL);
  clSetKernelArg(daisyTemplate->oclKernels->reduceMinAll, 5, sizeof(int), (void*)&templatePointsNo);

  unsigned int offsetToFine = 1 * GRADIENTS_NO;
  unsigned int offsetToDescriptor = (templatePoints[0].y  * daisyTemplate->paddedWidth + 
                                     templatePoints[0].x) * DESCRIPTOR_LENGTH;

  const size_t wgsDiffMiddle[2] = { WGX_MATCH_MIDDLE, 1}; // OPTIMAL = 128
  int targetPixelsPerWorkgroup = WG_TARGETS_NO;           // OPTIMAL = 32
  int workersPerTargetPixel = wgsDiffMiddle[0] / targetPixelsPerWorkgroup;

  int seedTemplatesPerRun = seedTemplatePointsNo; // multi-block processing causes errors - keep this until fixed
  const size_t wsDiffMiddle[2] = { (searchWidthMiddle * searchWidthMiddle * wgsDiffMiddle[0]) / targetPixelsPerWorkgroup, seedTemplatesPerRun };
//  const size_t wsDiffMiddle[2] = { searchWidthFine * searchWidthFine * workersPerTargetPixel, seedTemplatesPerRun };

  clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 0, sizeof(templateBuffer), (void*)&templateBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 1, sizeof(targetBuffer), (void*)&targetBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 2, sizeof(diffBuffer), (void*)&diffBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 3, sizeof(corrsBuffer), (void*)&corrsBuffer);
  clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 4, sizeof(int), (void*)&daisyTarget->paddedWidth);

  // Copy first descriptor petals
  error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBufferA,
  	                          (offsetToDescriptor + offsetToFine) * sizeof(float), 0,
                              3 * templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
  	                          0, NULL, NULL);

  cl_event lastReduction;

  checkpoint(daisyCl, &times->startMatchDaisy, 1);

  for(int templatePointNo = 0; templatePointNo < templatePointsNo; templatePointNo++){

    petalBuffer = (templatePointNo % 2 ? petalBufferB : petalBufferA);

    clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 0, sizeof(petalBuffer), (void*)&petalBuffer);

    checkpoint(daisyCl, &times->startDiffCoarse, times->enabled);

    for(int regionNo = 2; regionNo > -1; regionNo--){

      clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 4, sizeof(int), (void*)&regionNo);

      // Compute diffCoarse
      error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffCoarse, 2, 
                                     NULL, wsDiffCoarse, wgsDiffCoarse, 
                                     0, NULL, NULL);

      if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (diffCoarse)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

    }

    checkpoint(daisyCl, &times->endDiffCoarse, times->enabled);

    // Do an async transfer of the template petal to the petal buffer for the next iteration
    if(templatePointNo < templatePointsNo-1){

      petalBuffer = (templatePointNo % 2 ? petalBufferA : petalBufferB);

      // pixel to use from template
      point p = templatePoints[templatePointNo + 1];

      offsetToDescriptor = (p.y * daisyTemplate->paddedWidth + p.x) * DESCRIPTOR_LENGTH;

      error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBuffer,
      	                          (offsetToDescriptor + offsetToFine) * sizeof(float), 0,
                                  3 * templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
      	                          0, NULL, NULL);

    }

    checkpoint(daisyCl, &times->startDiffTranspose, times->enabled);

    // Transpose rotations from HxWxR to RxHxW
    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->transposeRotations, 2,
                                   NULL, wsTransposeRotations, wgsTransposeRotations,
                                   0, NULL, NULL);

    if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (transposeRotations)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

    checkpoint(daisyCl, &times->endDiffTranspose, times->enabled);

#ifdef STOREOUTPUT

    clFinish(daisyCl->ioqueue);

    float * diffArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);

    error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBufferTrans, CL_TRUE,
                                0, coarseWidth * coarseHeight * rotationsNo * sizeof(float),
                                diffArray, 0, NULL, NULL);

    if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (diffBufferTrans)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

    clFinish(daisyCl->ioqueue);

    char * fn = (char*) malloc(sizeof(char) * 200);
    sprintf(fn, "%s-diff%02d.bin", daisyTarget->filename, templatePointNo);

    saveBinary(diffArray,coarseWidth*coarseHeight*rotationsNo,fn);

    free(diffArray);

#endif


    checkpoint(daisyCl, &times->startReduceCoarse1, times->enabled);

    for(int rotation = 0; rotation < rotationsNo; rotation++){

      const size_t wsoReduceMin = rotation * wsReduceMin;

      // Find maximum for HxW of each rotation
      error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->reduceMin, 1, 
                                     &wsoReduceMin, &wsReduceMin, &wgsReduceMin,
                                     0, NULL, NULL);

    }

    checkpoint(daisyCl, &times->endReduceCoarse1, times->enabled);

    const size_t wsoReduceMinAll = templatePointNo;

    // Find minima for each rotation

    checkpoint(daisyCl, &times->startReduceCoarse2, times->enabled);

    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->reduceMinAll, 1, 
                                   &wsoReduceMinAll, &wsReduceMinAll, &wgsReduceMinAll,
                                   0, NULL, (templatePointNo == templatePointsNo-1 ? &lastReduction : NULL));

    checkpoint(daisyCl, &times->endReduceCoarse2, times->enabled);

  }

  error = clEnqueueReadBuffer(daisyCl->ioqueue, argminBuffer, CL_TRUE,
                              0, argminBufferLength * sizeof(float), argmin,
                              1, &lastReduction, NULL);


  //
  // Process correspondences
  //

  // 1. Get mode of rot of min(minima) of all template points
  int votedRotation = -1;
  int rotationVotes[ROTATIONS_NO+2];
  int rotationVotesArg[ROTATIONS_NO * templatePointsNo];
  for(int i = 0; i < rotationsNo; i++)
    rotationVotes[i] = 0;

  for(int i = 0; i < templatePointsNo; i++){

    int argrot = -1;
    float mind = 9999;
    for(int r = 0; r < rotationsNo; r++){
      if(argmin[(r+rotationsNo) * templatePointsNo + i] < mind){
        mind = argmin[(r+rotationsNo) * templatePointsNo + i];
        argrot = r;
      }
    }
    rotationVotesArg[argrot * templatePointsNo + rotationVotes[argrot]] = i;
    rotationVotes[argrot] += 1;

  }
  rotationVotes[rotationsNo] = 0;
  for(int r = 0; r < rotationsNo; r++){
    if(rotationVotes[r] > rotationVotes[rotationsNo]){
      rotationVotes[rotationsNo] = rotationVotes[r];
      votedRotation = r;
    }
  }

  printf("| VotedRotation %d || Votes %d |\n",votedRotation,rotationVotes[votedRotation]);

  point targetSize = { (float)daisyTarget->paddedWidth, (float)daisyTarget->paddedHeight };
  point templateSize = { (float)daisyTemplate->paddedWidth, (float)daisyTemplate->paddedHeight };

  // 2. Get the correspondences that result
  int * templateMatches = rotationVotesArg + votedRotation * templatePointsNo;
  int corrsNo = rotationVotes[votedRotation];
  int * targetMatches = (int*)malloc(sizeof(int) * corrsNo);
  for(int i = 0; i < corrsNo; i++){

    if(VERBOSE)
      printf("VotedRotation %d, templateMatch %d, PinnedArgmin %f\n",votedRotation,templateMatches[i],argmin[votedRotation * templatePointsNo + templateMatches[i]]);

    targetMatches[i] = (int)argmin[votedRotation * templatePointsNo + templateMatches[i]];

  }

  // 3. Estimate object projection
  for(int i = 0; i < corrsNo; i++){
    // upsample target points
    targetMatches[i] = (floor(targetMatches[i] / coarseTargetSize.x) * gridSpacing * targetSize.x)
                     + (targetMatches[i] % (int)coarseTargetSize.x) * gridSpacing;

    if(VERBOSE){
        printf("TargetMatch [%d,%d,%d]\n",targetMatches[i],
                            (int)(targetMatches[i] / targetSize.x), 
                            (int)(targetMatches[i] % (int)targetSize.x));
    }

  }

  float * projectionErrors = (float*) malloc(sizeof(float) * corrsNo);
  transform * t = minimise2dProjection(templatePoints, templateMatches, targetMatches, corrsNo,
                                       targetSize, templateSize, projectionErrors);

  // Filter out some of the correspondences with the projection errors
  float PROJERROR_THRESH = 0.15;
  int * filteredTemplateMatches = (int*)malloc(sizeof(int) * corrsNo);
  int * filteredMatches = (int*)malloc(sizeof(int) * corrsNo);
  int matchesNo = 0;
  for(int i = 0; i < corrsNo; i++){
    if(projectionErrors[i] < PROJERROR_THRESH){
      filteredTemplateMatches[matchesNo] = templateMatches[i];
      filteredMatches[matchesNo++] = targetMatches[i];

      if(VERBOSE){
        printf("Filtered Match [%d,%d] TO [%d,%d]\n", (int)templatePoints[templateMatches[i]].y,(int)templatePoints[templateMatches[i]].x,
                                                      (int)(targetMatches[i]/targetSize.x),targetMatches[i]%(int)targetSize.x);
      }
    }
  }

  printf("| FilteredMatches = %d |\n", matchesNo);

  //  transform t;
  //  point centre = estimateObjectCentre(templatePoints, templateMatches, targetMatches, corrsNo,
  //                                      coarseTargetSize, coarseTemplateSize, &t);

  // 4. Get region of interest
  /*point p1 = projectPoint({ 0, 0 }, *t);
  point p2 = projectPoint({ templateSize.x, 0 }, *t);
  point p3 = projectPoint({ templateSize.x, templateSize.y }, *t);
  point p4 = projectPoint({ 0, templateSize.y }, *t);

  point topLeft = { max(min(min(min(p1.x, p2.x), p3.x), p4.x), 0),
                    max(min(min(min(p1.y, p2.y), p3.y), p4.y), 0)};
  point bottomRight = { min(max(max(max(p1.x, p2.x), p3.x), p4.x), targetSize.x-1),
                        min(max(max(max(p1.y, p2.y), p3.y), p4.y), targetSize.y-1)};*/

  // (pre) 5. Generate seed descriptors (512 of them)
  point * seedTemplatePoints = generateTemplatePoints(daisyTemplate, seedTemplatePointsNo, 0, 0);  

  // 5. Project them with the overall projection, then find 2 closest target descriptors
  point * seedTargetPoints = (point*)malloc(sizeof(point) * seedTemplatePointsNo);

  // Get local projection of those 2 target descriptors and re-project seed descriptor
  projectTargetSeeds(seedTemplatePoints, seedTargetPoints, seedTemplatePointsNo,
                     templatePoints, filteredTemplateMatches, 
                     filteredMatches, matchesNo, targetSize, t);

  // 6. Fill up the corrs buffer with template to target correspondences
  float * corrs = (float*)malloc(sizeof(float) * seedTemplatePointsNo * 2);

  for(int i = 0; i < seedTemplatePointsNo; i++){

    if(VERBOSE){
      printf("Seed From [%d,%d] To [%d,%d]    ", (int)seedTemplatePoints[i].y, 
                                                 (int)seedTemplatePoints[i].x, 
                                                 (int)seedTargetPoints[i].y, 
                                                 (int)seedTargetPoints[i].x);
    }

    /*topLeft = { min(seedTargetPoints[i].x, topLeft.x),
                  min(seedTargetPoints[i].y, topLeft.y)};
      bottomRight = { max(seedTargetPoints[i].x, bottomRight.x),
                      max(seedTargetPoints[i].y, bottomRight.y)};*/

    corrs[i * 2] = floor(seedTemplatePoints[i].y) * daisyTemplate->paddedWidth + floor(seedTemplatePoints[i].x);
    corrs[i * 2 + 1] = floor(seedTargetPoints[i].y) * daisyTarget->paddedWidth + floor(seedTargetPoints[i].x);

  }

#ifdef STOREOUTPUT

  string fn0 = daisyTarget->filename;
  string sfx0 = "-corrPoints.bin";
  string sfx1 = "-matches.bin";

  saveBinary(corrs, seedTemplatePointsNo * 2, fn0+sfx0);

  float * arr = (float*)malloc(sizeof(float)*matchesNo);

  for(int i = 0; i < matchesNo; i++)
    arr[i] = filteredMatches[i];

  saveBinary(arr, matchesNo, fn0+sfx1);

#endif

  // 7. Search the middle layer
  int spacingMiddle = 2;

  printf("\nB) MiddleLayer [%dx%d] - Search Resolution = %d - Seeds = %d - Rotations = %d\n", 
         searchWidthMiddle, searchWidthMiddle,
         spacingMiddle,
         seedTemplatePointsNo, rotationsNoMiddle);

  int templateRuns = ceil(seedTemplatePointsNo / seedTemplatesPerRun);

  printf("\nWorkerSize = [%d,%d], WorkersPerPixel = %d, WorkgroupSize = [%d,%d]\n",
                      (int)wsDiffMiddle[0], (int)wsDiffMiddle[1],
                      (int)workersPerTargetPixel, (int)wgsDiffMiddle[0], (int)wgsDiffMiddle[1]);

  for(int run = 0; run < templateRuns; run++){

    cl_event corrTransfer;

    clEnqueueWriteBuffer(daisyCl->ioqueue, corrsBuffer, CL_TRUE,
                         0, seedTemplatesPerRun * 2 * sizeof(float), (void*) (corrs + seedTemplatesPerRun * 2 * run),
                         0, NULL, &corrTransfer);

    checkpoint(daisyCl, &times->startDiffMiddle, times->enabled);

    for(int petalRegionNo = 2; petalRegionNo > -1; petalRegionNo--){

      int rotationNo = ((votedRotation-2) + ROTATIONS_NO) % ROTATIONS_NO;
      int regionNo = petalRegionNo;
      int templateNoOffset = seedTemplatesPerRun * run;

      clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 5, sizeof(int), (void*)&regionNo);
      clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 6, sizeof(int), (void*)&rotationNo);
      clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 7, sizeof(int), (void*)&templateNoOffset);

      //const size_t wsoDiffMiddle[2] = { 0, seedTemplatesPerRun * run };

      // Compute diffMiddle
      error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffMiddle, 2, 
                                     NULL, wsDiffMiddle, wgsDiffMiddle, 
                                     1, &corrTransfer, NULL);

    }

    checkpoint(daisyCl, &times->endDiffMiddle, times->enabled);

  }

  checkpoint(daisyCl, &times->endMatchDaisy, 1);

  times->difft = timeDiff(times->startMatchDaisy,times->endMatchDaisy);

  printf("Match: %.2f ms\n",times->difft);

#ifdef STOREOUTPUT

  int diffMiddleSize = seedTemplatePointsNo * searchWidthMiddle * searchWidthMiddle * rotationsNoMiddle;

  float * diffMiddle = (float*) malloc(sizeof(float) * diffMiddleSize);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBuffer, CL_TRUE,
                              0, diffMiddleSize * sizeof(float), 
                              diffMiddle, 0, NULL, NULL);

  string fn = daisyTarget->filename;
  string sfx = "-coarseArgmin.bin";

  saveBinary(argmin, argminBufferLength, fn+sfx);

  sfx = "-templatePoints.bin";

  float * tp = (float*) malloc(sizeof(float) * templatePointsNo);

  for(int i = 0; i < templatePointsNo; i++)
    tp[i] = templatePoints[i].y * daisyTemplate->width + templatePoints[i].x;

  saveBinary(tp, templatePointsNo, fn+sfx);

  sfx = "-diffMiddle.bin";

  saveBinary(diffMiddle, diffMiddleSize, fn+sfx);

#endif

#ifdef CPU_VERIFICATION
//
// VERIFICATION CODE
//

  float * argminArray = (float*)malloc(sizeof(float) * templatePointsNo * rotationsNo);
  float * diffArray = (float*)malloc(sizeof(float) * seedTemplatePointsNo * searchWidthMiddle * searchWidthMiddle * rotationsNoMiddle);
  float * diffTransArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);
  float * petalArray = (float*)malloc(sizeof(float) * 3 * REGION_PETALS_NO * GRADIENTS_NO);
  float * targetArray = (float*)malloc(sizeof(float) * (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH));
  float * templateArray = (float*)malloc(sizeof(float) * (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH));

  error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBuffer, CL_TRUE,
                              0, seedTemplatePointsNo * searchWidthMiddle * searchWidthMiddle * rotationsNoMiddle * sizeof(float), 
                              diffArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (diffBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBufferTrans, CL_TRUE,
                              0, coarseWidth * coarseHeight * rotationsNo * sizeof(float), 
                              diffTransArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (diffBufferTrans)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, targetBuffer, CL_TRUE,
                              0, (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH) * sizeof(float), 
                              targetArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (targetBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);
 
  error = clEnqueueReadBuffer(daisyCl->ioqueue, templateBuffer, CL_TRUE,
                              0, (daisyTemplate->paddedWidth * daisyTemplate->paddedHeight * DESCRIPTOR_LENGTH) * sizeof(float), 
                              templateArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (templateBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, petalBuffer, CL_TRUE,
                              0, 3 * REGION_PETALS_NO * GRADIENTS_NO * sizeof(float), 
                              petalArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (petalBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, argminBuffer, CL_TRUE,
                              0, templatePointsNo * rotationsNo * sizeof(float), 
                              argminArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (argminBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  gettimeofday(&times->startMatchCpu, NULL);

  long int issues;
  for(int k = 0; k < templatePointsNo; k++){
    issues = verifyDiffCoarse(daisyTarget, petalArray, targetArray, diffArray);
    printf("diffCoarse verification: %ld issues (ignore if =512 and running reduceMin)\n",issues);
  }

  issues = verifyTransposeRotations(daisyTarget, diffArray, diffTransArray);
  printf("transposeRotations verification: %ld issues (ignore if =512 and running reduceMin)\n",issues);

  issues = verifyReduceMinAll(daisyTarget, diffTransArray, argminArray, templatePointsNo);
  printf("reduceMin verification: %ld issues\n",issues);

  issues = verifyDiffMiddle(daisyTarget, daisyTemplate, targetArray, templateArray, diffArray, 
                            searchWidthMiddle, corrs, votedRotation, seedTemplatePointsNo, rotationsNoMiddle);
  printf("diffMiddle verification: %ld issues\n",issues);

  gettimeofday(&times->endMatchCpu, NULL);

  free(diffArray);
  free(petalArray);
  free(targetArray);

#endif

  clReleaseMemObject(argminBuffer);
  clReleaseMemObject(diffBuffer);
  clReleaseMemObject(corrsBuffer);
  clReleaseMemObject(petalBufferA);
  clReleaseMemObject(petalBufferB);
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

  // thread vars
  int targetY, targetX, rotation, p, g;
  long int offsetToDescriptor, offsetToCoarseRegion;
  int targetPetalNo, targetGradientNo;
  float diff;
  float gpudiff = -1;
  float * targetDescriptor;
  float * targetPetal;
  float * targetPetalMiddle;
  float * targetPetalFine;

  //#pragma omp parallel for private(targetY,targetX,rotation,p,g,offsetToDescriptor,offsetToCoarseRegion,targetPetalNo,targetGradientNo,diff,gpudiff,targetDescriptor,targetPetal,targetPetalMiddle,targetPetalFine)
  for(targetY = subsample / 2 -1; targetY < daisyTarget->paddedHeight; targetY+=subsample){

    for(targetX = subsample / 2 -1; targetX < daisyTarget->paddedWidth; targetX+=subsample){
      
      offsetToDescriptor = (targetY * daisyTarget->paddedWidth + targetX) * DESCRIPTOR_LENGTH;
      offsetToCoarseRegion = (TOTAL_PETALS_NO - REGION_PETALS_NO) * GRADIENTS_NO;
      targetDescriptor = targetArray + offsetToDescriptor;
      targetPetal = targetDescriptor + offsetToCoarseRegion;
      targetPetalMiddle = targetDescriptor + offsetToCoarseRegion - REGION_PETALS_NO * GRADIENTS_NO;
      targetPetalFine = targetDescriptor + 1 * GRADIENTS_NO;

      for(rotation = 0; rotation < rotationsNo; rotation++){

        diff = 0.0;

        targetPetalNo = rotation;
        targetGradientNo = rotation;

        for(p = 0; p < REGION_PETALS_NO; p++){

          for(g = 0; g < GRADIENTS_NO; g++){

            diff += fabs(petalArray[(p + 16) * GRADIENTS_NO + g] - 

                         targetPetal[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                      (targetGradientNo + g) % GRADIENTS_NO]);

            diff += fabs(petalArray[(p + 8) * GRADIENTS_NO + g] - 

                         targetPetalMiddle[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                           (targetGradientNo + g) % GRADIENTS_NO]);
   
            diff += fabs(petalArray[p * GRADIENTS_NO + g] - 

                           targetPetalFine[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                           (targetGradientNo + g) % GRADIENTS_NO]);

          }

        }
        diffArray[((targetY / subsample) * coarseWidth + targetX / subsample) * rotationsNo + rotation] += diff;
        if(fabs(gpudiff - diff) > 0.0001){
          issues++;
//          if(shown++ < 100){
//            printf("X,Y,R = %d,%d,%d | CPU = %.3f and GPU = %.3f\n",targetX / subsample,targetY/subsample,rotation,diff,gpudiff);
//          }
        }
        else if(0 && shown++ < 200)
          printf("X,Y,R = %d,%d,%d | CPU = %.3f and GPU = %.3f\n",targetX / subsample,targetY/subsample,rotation,diff,gpudiff);

      }

    }

  }

  return issues;

}

long int verifyTransposeRotations(daisy_params * daisyTarget, float * diffArray, float * diffTransArray){

  long int issues = 0;

  int rotationsNo = ROTATIONS_NO;
  int subsample = 4;
  int coarseHeight = daisyTarget->paddedHeight / subsample;
  int coarseWidth = daisyTarget->paddedWidth / subsample;

  int rotation, targetY, targetX;
  float cpuTrans, gpuTrans;

  //#pragma omp parallel for private(rotation,targetY,targetX,cpuTrans,gpuTrans)
  for(rotation = 0; rotation < rotationsNo; rotation++){

    for(targetY = 0; targetY < coarseHeight; targetY++){

      for(targetX = 0; targetX < coarseWidth; targetX++){

          cpuTrans = diffArray[(targetY * coarseWidth + targetX) * rotationsNo + rotation];
          gpuTrans = diffTransArray[rotation * coarseHeight * coarseWidth + targetY * coarseWidth + targetX];

          // imitate a memory store
          diffTransArray[rotation * coarseHeight * coarseWidth + targetY * coarseWidth + targetX] = cpuTrans;

          if(fabs(cpuTrans - gpuTrans) > 0.001){
            issues++;
            //if(issues < 100)
            //  printf("R,Y,X = %d,%d,%d | CPU TRANS = %f | GPU TRANS = %f\n", rotation,targetY,targetX,cpuTrans,gpuTrans);
          }

      }

    }

  }

  return issues;

}

long int verifyReduceMinAll(daisy_params * daisyTarget, float * diffArray, float * minArray, int templatePointsNo){

  long int issues = 0;
  int rotationsNo = ROTATIONS_NO;
  int subsample = 4;
  int diffArrayLengthPerRotation = (daisyTarget->paddedWidth / subsample) * (daisyTarget->paddedHeight / subsample);

  int rotation, i;
  float min, gpumin;
  float argmin = -1;
  //#pragma omp parallel for private(rotation,min,argmin,gpumin,i)
  for(rotation = 0; rotation < rotationsNo; rotation++){

    min = 999;

    for(i = 0; i < diffArrayLengthPerRotation; i++){
      if(diffArray[rotation * diffArrayLengthPerRotation + i] < min){
        min = diffArray[rotation * diffArrayLengthPerRotation + i];
        argmin = i;
      }
    }

    // verify min
    gpumin = minArray[rotation*templatePointsNo+templatePointsNo-1];
      printf("(%s) RotationNo = %d | CPU argmin = %f (%f)| GPU argmin = %f (%f)\n",(argmin==gpumin?"PASS":"FAIL"),
                      rotation,argmin,min,gpumin,diffArray[rotation*diffArrayLengthPerRotation+(int)gpumin]);
    if(argmin!=gpumin) issues++;

  }

  return issues; 

}

long int verifyDiffMiddle(daisy_params * daisyTarget, daisy_params * daisyTemplate, float * targetArray, 
                          float * templateArray, float * diffArray, int searchWidth, float * corrs, int votedRotation,
                          int templatesNo, int rotationsNoMiddle){



  long int issues = 0;

  int pointNo = 0;
  int y,x,r,p,g;
  int rotation;

  long int offsetToDescriptor, offsetToCoarseRegion;
  int targetPetalNo, targetGradientNo;
  float diff, gpudiff;
  float * targetDescriptor;
  float * targetPetal;
  float * targetPetalMiddle;
  float * targetPetalFine;
  float * petalArray;
  int templateWidth = daisyTemplate->paddedWidth;
  int targetWidth = daisyTarget->paddedWidth;

  //#pragma omp parallel for private(y,x,r,rotation,p,g,offsetToDescriptor,offsetToCoarseRegion,targetPetalNo,targetGradientNo,diff,gpudiff,targetDescriptor,targetPetal,targetPetalMiddle,targetPetalFine,petalArray)
  for(pointNo = 0; pointNo < templatesNo; pointNo++){

    for(y = -searchWidth/2; y < searchWidth/2; y++){
      for(x = -searchWidth/2; x < searchWidth/2; x++){

        offsetToDescriptor = ((floor(corrs[pointNo * 2 + 1] / targetWidth)+y) * daisyTarget->paddedWidth + ((int)corrs[pointNo * 2 + 1] % targetWidth) + x) * DESCRIPTOR_LENGTH;
        offsetToCoarseRegion = (TOTAL_PETALS_NO - REGION_PETALS_NO) * GRADIENTS_NO;
        targetDescriptor = targetArray + offsetToDescriptor;
        targetPetal = targetDescriptor + offsetToCoarseRegion;
        targetPetalMiddle = targetDescriptor + offsetToCoarseRegion - REGION_PETALS_NO * GRADIENTS_NO;
        targetPetalFine = targetDescriptor + 1 * GRADIENTS_NO;
        petalArray = templateArray + (int)(floor(corrs[pointNo * 2] / templateWidth) + (int)corrs[pointNo * 2] % templateWidth) * DESCRIPTOR_LENGTH + 1 * GRADIENTS_NO;

        for(r = 0; r < rotationsNoMiddle; r++){
          
          diff = 0.0;

          rotation = (votedRotation + r) % ROTATIONS_NO;
          targetPetalNo = rotation;
          targetGradientNo = rotation;

          for(p = 0; p < REGION_PETALS_NO; p++){

            for(g = 0; g < GRADIENTS_NO; g++){

              diff += fabs(petalArray[(p + 16) * GRADIENTS_NO + g] - 

                           targetPetal[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                        (targetGradientNo + g) % GRADIENTS_NO]);

              diff += fabs(petalArray[(p + 8) * GRADIENTS_NO + g] - 

                           targetPetalMiddle[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                             (targetGradientNo + g) % GRADIENTS_NO]);
     
              diff += fabs(petalArray[p * GRADIENTS_NO + g] - 

                             targetPetalFine[((targetPetalNo + p) % REGION_PETALS_NO) * GRADIENTS_NO + 
                                             (targetGradientNo + g) % GRADIENTS_NO]);

            }

          }
          diffArray[pointNo * (searchWidth * searchWidth * rotationsNoMiddle) + r] += diff;
          if(fabs(gpudiff - diff) > 0.0001){
  //          if(shown++ < 100){
  //            printf("X,Y,R = %d,%d,%d | CPU = %.3f and GPU = %.3f\n",targetX / subsample,targetY/subsample,rotation,diff,gpudiff);
  //          }
          }

        }

      }
    }


  }

  return issues;


}














