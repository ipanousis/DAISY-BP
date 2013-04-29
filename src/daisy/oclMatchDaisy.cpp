#include "oclMatchDaisy.h"
#include <string.h>

#define min(a,b) ( a < b ? a : b)
#define max(a,b) ( a < b ? b : a)

long int verifyDiffCoarse(daisy_params * daisyTarget, float * petalArray, 
                          float * targetArray, float * diffArray);
long int verifyReduceMinAll(daisy_params * daisyTarget, float * diffTransArray, float * diffArray, int templatePointsNo);
long int verifyTransposeRotations(daisy_params * daisyTarget, float * diffArray, float * diffTransArray);

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

  daisy->oclKernels->reduceMinAll = clCreateKernel(daisyCl->program, "reduceMinAll", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (reduceMinAll)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->normaliseRotation = clCreateKernel(daisyCl->program, "normaliseRotation", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (normaliseRotation)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  daisy->oclKernels->diffMiddle = clCreateKernel(daisyCl->program, "diffMiddle", &error);
  if(oclErrorM("initOclMatch","clCreateKernel (diffMiddle)",error)) return oclCleanUp(daisy->oclKernels,daisyCl,error);

  return error;

}

#define max(a,b) (a > b ? a : b)

point * generateTemplatePoints(daisy_params * daisy, int templatePointsNo, int xOffset, int yOffset){

  // give back a an equally spaced grid of points on the template
  // do not have points on the edges of the template, use boundary offset of 15
  point * points = (point*) malloc(sizeof(point) * templatePointsNo);

  int boundaryOffset = 15;
  int xRange = daisy->width - boundaryOffset * 2;
  int yRange = daisy->height - boundaryOffset * 2;

//  int yStep = max(floor(yRange / sqrt(templatePointsNo)), 1);
//  int xStep = max(floor(xRange / sqrt(templatePointsNo)), 1);
//  printf("xrange,yrange,step | %d,%d,%d\n",xRange,yRange,step);
  int step = max(floor(sqrt((yRange * xRange) / templatePointsNo)), 1);

  int x = boundaryOffset;
  int y = boundaryOffset;
  for(int i = 0; i < templatePointsNo; i++){

//    printf("(%d,%d), ",x,y);
    points[i].x = x; points[i].y = y;

    x += step;

    if(x > xRange){
      x = x % step + boundaryOffset;
      y = y + step;
//      y = (y > yRange ? y % step + boundaryOffset : y);
    }

//    if(x > xRange){
//      x = x % xRange + boundaryOffset;
//      y = (y + yStep);
//      y = (y > yRange ? y % yRange + boundaryOffset : y);
//    }

  }

  return points;

}

void saveBinary(float * array, int size, string filename){

  FILE * fp;
  fp = fopen((char*)filename.c_str(), "wb");
  fwrite(array, sizeof(float), size, fp);
  fclose(fp);

}

int oclMatchDaisy(daisy_params * daisyTemplate, daisy_params * daisyTarget,
                  ocl_constructs * daisyCl, time_params * times){

  cl_int error = 0;

  int templatePointsNo = 16;
  point * templatePoints = generateTemplatePoints(daisyTemplate, templatePointsNo, 0, 0);

  cl_mem templateBuffer = daisyTemplate->buffers[0];
  cl_mem targetBuffer = daisyTarget->buffers[0];

  int gridSpacing = pow(SUBSAMPLE_RATE,2);
  int coarseWidth  = daisyTarget->paddedWidth  / gridSpacing;
  int coarseHeight = daisyTarget->paddedHeight / gridSpacing;
  point coarseTemplateSize = { daisyTemplate->width  / gridSpacing, daisyTemplate->height  / gridSpacing };
  point coarseTargetSize = { coarseWidth, coarseHeight };
  int rotationsNo = ROTATIONS_NO;
  int templatePetalsPerRun = 8;

  int rotationsNoRefined = 4;
  int searchWidthRefined = 32;
  int seedTemplatePointsNo = 512;

  int diffBufferSize1 = (coarseWidth * coarseHeight * rotationsNo);
  int diffBufferSize2 = (seedTemplatePointsNo * rotationsNoRefined * (searchWidthRefined * searchWidthRefined));

  printf("Matching coarse layer [%dx%d] (subsampled by %d) for %d rotations\n",
         coarseHeight,coarseWidth,(int)pow(SUBSAMPLE_RATE,2),rotationsNo);

  cl_mem diffBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       max(diffBufferSize1, diffBufferSize2) * sizeof(float),
                                       (void*)NULL, &error);

  cl_mem diffBufferTrans = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                          (coarseWidth * coarseHeight * rotationsNo) * sizeof(float),
                                          (void*)NULL, &error);

  cl_mem argminBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                       (templatePointsNo * rotationsNo) * sizeof(float),
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

  int argminBufferLength = templatePointsNo * rotationsNo * 2;

  cl_mem pinnedArgminBuffer = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
                                             argminBufferLength * sizeof(float), NULL, &error);

  if(oclErrorM("oclDaisy","clCreateBuffer (pinnedArgmin)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  float * pinnedArgminArray = (float*) clEnqueueMapBuffer(daisyCl->ioqueue, pinnedArgminBuffer, 0,
                                                          CL_MAP_WRITE, 0, argminBufferLength * sizeof(float),
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

  // Copy first descriptor petals
  error = clEnqueueCopyBuffer(daisyCl->ioqueue, templateBuffer, petalBufferA,
  	                          (offsetToDescriptor + offsetToFine) * sizeof(float), 0,
                              3 * templatePetalsPerRun * GRADIENTS_NO * sizeof(float),
  	                          0, NULL, NULL);

  error = clFinish(daisyCl->ioqueue);

  cl_event lastReduction;

  gettimeofday(&times->startConv,NULL);

  for(int templatePointNo = 0; templatePointNo < templatePointsNo; templatePointNo++){

    petalBuffer = (templatePointNo % 2 ? petalBufferB : petalBufferA);

    clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 0, sizeof(petalBuffer), (void*)&petalBuffer);

    for(int regionNo = 2; regionNo > -1; regionNo--){

      clSetKernelArg(daisyTemplate->oclKernels->diffCoarse, 4, sizeof(int), (void*)&regionNo);

      // Compute diffCoarse
      error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffCoarse, 2, 
                                     NULL, wsDiffCoarse, wgsDiffCoarse, 
                                     0, NULL, NULL);

      if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (diffCoarse)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

    }

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

    // Transpose rotations from HxWxR to RxHxW
    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->transposeRotations, 2,
                                   NULL, wsTransposeRotations, wgsTransposeRotations,
                                   0, NULL, NULL);

    if(oclErrorM("oclDaisy","clEnqueueNDRangeKernel (transposeRotations)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

/*    clFinish(daisyCl->ioqueue);
    float * diffArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);

    error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBufferTrans, CL_TRUE,
                                0, coarseWidth * coarseHeight * rotationsNo * sizeof(float),
                                diffArray, 0, NULL, NULL);

    if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (diffBufferTrans)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);
    clFinish(daisyCl->ioqueue);
    for(int i = 1000; i < 1010; i++)
      printf("%.3f, ", diffArray[i]);
    char * fn = (char*) malloc(sizeof(char) * 200);
    sprintf(fn, "%s-diff%02d.bin", daisyTarget->filename, templatePointNo);
    saveBinary(diffArray,coarseWidth*coarseHeight*rotationsNo,fn);
    free(diffArray);*/

    for(int rotation = 0; rotation < rotationsNo; rotation++){

      const size_t wsoReduceMin = rotation * wsReduceMin;

      // Find maximum for HxW of each rotation
      error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->reduceMin, 1, 
                                     &wsoReduceMin, &wsReduceMin, &wgsReduceMin,
                                     0, NULL, NULL);

    }

    const size_t wsoReduceMinAll = templatePointNo;

    // Find minima for each rotation
    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->reduceMinAll, 1, 
                                   &wsoReduceMinAll, &wsReduceMinAll, &wgsReduceMinAll,
                                   0, NULL, (templatePointNo == templatePointsNo-1 ? &lastReduction : NULL));

    error = clFinish(daisyCl->ioqueue);

    if(oclErrorM("oclMatchDaisy","clFinish",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  }

  error = clEnqueueReadBuffer(daisyCl->ioqueue, argminBuffer, CL_TRUE,
                              0, argminBufferLength * sizeof(float), pinnedArgminArray,
                              1, &lastReduction, NULL);

  //
  // Process correspondences
  //

  // 1. Get mode of rot of min(minima) of all template points
  int votedRotation;
  int rotationVotes[ROTATIONS_NO+2];
  int rotationVotesArg[ROTATIONS_NO * templatePointsNo];
  for(int i = 0; i < rotationsNo; i++)
    rotationVotes[i] = 0;
  for(int i = 0; i < templatePointsNo; i++){

    int argrot;
    float mind = 9999;
    for(int r = 0; r < rotationsNo; r++){
      if(pinnedArgminArray[(r+rotationsNo) * templatePointsNo + i] < mind){
        mind = pinnedArgminArray[(r+rotationsNo) * templatePointsNo + i];
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

  // 2. Get the correspondences that result
  int * templateMatches = rotationVotesArg + votedRotation * templatePointsNo;
  int corrsNo = rotationVotes[votedRotation];
  int * targetMatches = (int*)malloc(sizeof(int) * corrsNo);
  for(int i = 0; i < corrsNo; i++)
    targetMatches[i] = (int)pinnedArgminArray[votedRotation * templatePointsNo + templateMatches[i]];

  // 3. Estimate object projection
  for(int i = 0; i < templatePointsNo; i++){
    templatePoints[i].x = templatePoints[i].x / gridSpacing;
    templatePoints[i].y = templatePoints[i].y / gridSpacing;
  }
  float * projectionErrors = (float*) malloc(sizeof(float) * corrsNo);
  transform * t = minimise2dProjection(templatePoints, templateMatches, targetMatches, corrsNo,
                                     coarseTargetSize, coarseTemplateSize, projectionErrors);

  // Filter out some of the correspondences with the projection errors
  float PROJERROR_THRESH = 0.15;
  int * filteredMatches = (int*)malloc(sizeof(int) * corrsNo);
  int matchesNo = 0;
  for(int i = 0; i < corrsNo; i++){
    if(projectionErrors[i] < PROJERROR_THRESH)
      filteredMatches[matchesNo++] = targetMatches[i];
  }

//  transform t;
//  point centre = estimateObjectCentre(templatePoints, templateMatches, targetMatches, corrsNo,
//                                      coarseTargetSize, coarseTemplateSize, &t);

  // 4. Get region of interest
  point p1 = projectPoint({ 0, 0 }, *t);
  point p2 = projectPoint({ coarseTemplateSize.x, 0 }, *t);
  point p3 = projectPoint({ coarseTemplateSize.x, coarseTemplateSize.y }, *t);
  point p4 = projectPoint({ 0, coarseTemplateSize.y }, *t);

  point topLeft = { max(min(min(min(p1.x, p2.x), p3.x), p4.x), 0),
                    max(min(min(min(p1.y, p2.y), p3.y), p4.y), 0)};
  point bottomRight = { min(max(max(max(p1.x, p2.x), p3.x), p4.x), coarseTargetSize.x-1),
                        min(max(max(max(p1.y, p2.y), p3.y), p4.y), coarseTargetSize.y-1)};

  // (pre) 5. Generate seed descriptors (512 of them)
  point * seedTemplatePoints = generateTemplatePoints(daisyTemplate, seedTemplatePointsNo, 0, 0);  

  // 5. Project them with the overall projection, then find 2 closest target descriptors
  point * seedTargetPoints = (point*)malloc(sizeof(point) * seedTemplatePointsNo);

  // Get local projection of those 2 target descriptors and re-project seed descriptor
  projectTargetSeeds(seedTemplatePoints, seedTargetPoints, seedTemplatePointsNo,
                     templatePoints,
                     filteredMatches, matchesNo, coarseTargetSize, t);

  printf("Xelloou9\n");
  // 6. Search for the descriptors in a 32x32 search region with subsample of 2 (so 64x64)

  //
  // !!!

//  3 rotations...

//  search region...

//  kernel...

/*kernel void diffMiddle( global   float * tmp,
                        global   float * trg,
                        global   float * diff,
                        global   float * corrs,
                        const    int     regionNo,
                        const    int     rotationNo)*/  

  for(int petalRegionNo = 2; petalRegionNo > -1; petalRegionNo--){

    int rotationNo = ((votedRotation-2) + ROTATIONS_NO) % ROTATIONS_NO;
    int regionNo = petalRegionNo;

    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 0, sizeof(templateBuffer), (void*)&templateBuffer);
    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 1, sizeof(targetBuffer), (void*)&targetBuffer);
    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 2, sizeof(diffBuffer), (void*)&diffBuffer);
    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 3, sizeof(corrsBuffer), (void*)&corrsBuffer);
    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 4, sizeof(int), (void*)&regionNo);
    clSetKernelArg(daisyTemplate->oclKernels->diffMiddle, 5, sizeof(int), (void*)&rotationNo);

    // Compute diffMiddle
    const size_t wgsDiffMiddle[2] = { 64, 1};
    int targetPixelsPerWorkgroup = 32;
    int workersPerTargetPixel = wgsDiffMiddle[0] / targetPixelsPerWorkgroup;
    const size_t wsDiffMiddle[2] = { searchWidthRefined * searchWidthRefined * workersPerTargetPixel, seedTemplatePointsNo };

    printf("\nTotal Workers = [%d,%d], WorkersPerTargetPixel = %d, Workgroup Size = [%d,%d]\n",wsDiffMiddle[0],wsDiffMiddle[1],
                                        workersPerTargetPixel,wgsDiffMiddle[0],wgsDiffMiddle[1]);

    error = clEnqueueNDRangeKernel(daisyCl->ioqueue, daisyTemplate->oclKernels->diffMiddle, 2, 
                                   NULL, wsDiffMiddle, wgsDiffMiddle, 
                                   0, NULL, NULL);

  }

  error = clFinish(daisyCl->ioqueue);

  if(oclErrorM("oclMatchDaisy","clFinish",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  gettimeofday(&times->endConv,NULL);

  times->difft = timeDiff(times->startConv,times->endConv);

  printf("Match: %.2f ms\n",times->difft);

  string fn = daisyTarget->filename;
  string sfx = "-coarseArgmin.bin";
  saveBinary(pinnedArgminArray, argminBufferLength, fn+sfx);
  sfx = "-templatePoints.bin";
  float * tp = (float*) malloc(sizeof(float) * templatePointsNo);
  for(int i = 0; i < templatePointsNo; i++)
    tp[i] = templatePoints[i].y * daisyTemplate->width + templatePoints[i].x;
  saveBinary(tp, templatePointsNo, fn+sfx);

#ifdef CPU_VERIFICATION
//
// VERIFICATION CODE
//

  float * argminArray = (float*)malloc(sizeof(float) * templatePointsNo * rotationsNo);
  float * diffArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);
  float * diffTransArray = (float*)malloc(sizeof(float) * coarseWidth * coarseHeight * rotationsNo);
  float * petalArray = (float*)malloc(sizeof(float) * 3 * REGION_PETALS_NO * GRADIENTS_NO);
  float * targetArray = (float*)malloc(sizeof(float) * (daisyTarget->paddedWidth * daisyTarget->paddedHeight * DESCRIPTOR_LENGTH));

  error = clEnqueueReadBuffer(daisyCl->ioqueue, diffBuffer, CL_TRUE,
                              0, coarseWidth * coarseHeight * rotationsNo * sizeof(float), 
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

  error = clEnqueueReadBuffer(daisyCl->ioqueue, petalBuffer, CL_TRUE,
                              0, 3 * REGION_PETALS_NO * GRADIENTS_NO * sizeof(float), 
                              petalArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (petalBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  error = clEnqueueReadBuffer(daisyCl->ioqueue, argminBuffer, CL_TRUE,
                              0, templatePointsNo * rotationsNo * sizeof(float), 
                              argminArray, 0, NULL, NULL);

  if(oclErrorM("oclMatchDaisy","clEnqueueReadBuffer (argminBuffer)",error)) return oclCleanUp(daisyTemplate->oclKernels,daisyCl,error);

  long int issues = verifyDiffCoarse(daisyTarget, petalArray, targetArray, diffArray);
  printf("diffCoarse verification: %ld issues (ignore if =512 and running reduceMin)\n",issues);

  issues = verifyTransposeRotations(daisyTarget, diffArray, diffTransArray);
  printf("transposeRotations verification: %ld issues (ignore if =512 and running reduceMin)\n",issues);

  issues = verifyReduceMinAll(daisyTarget, diffTransArray, argminArray, templatePointsNo);
  printf("reduceMin verification: %ld issues\n",issues);

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

  for(int targetY = subsample / 2 -1; targetY < daisyTarget->paddedHeight; targetY+=subsample){

    for(int targetX = subsample / 2 -1; targetX < daisyTarget->paddedWidth; targetX+=subsample){
      
      long int offsetToDescriptor = (targetY * daisyTarget->paddedWidth + targetX) * DESCRIPTOR_LENGTH;
      long int offsetToCoarseRegion = (TOTAL_PETALS_NO - REGION_PETALS_NO) * GRADIENTS_NO;
      float * targetDescriptor = targetArray + offsetToDescriptor;
      float * targetPetal = targetDescriptor + offsetToCoarseRegion;
      float * targetPetalMiddle = targetDescriptor + offsetToCoarseRegion - REGION_PETALS_NO * GRADIENTS_NO;
      float * targetPetalFine = targetDescriptor + 1 * GRADIENTS_NO;

      for(int rotation = 0; rotation < rotationsNo; rotation++){

        float diff = 0.0;

        int targetPetalNo = rotation;
        int targetGradientNo = rotation;

        for(int p = 0; p < REGION_PETALS_NO; p++){

          for(int g = 0; g < GRADIENTS_NO; g++){

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
        float gpudiff = diffArray[((targetY / subsample) * coarseWidth + targetX / subsample) * rotationsNo + rotation];
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

  for(int rotation = 0; rotation < rotationsNo; rotation++){

    for(int targetY = 0; targetY < coarseHeight; targetY++){

      for(int targetX = 0; targetX < coarseWidth; targetX++){

          float cpuTrans = diffArray[(targetY * coarseWidth + targetX) * rotationsNo + rotation];
          float gpuTrans = diffTransArray[rotation * coarseHeight * coarseWidth + targetY * coarseWidth + targetX];

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

  for(int rotation = 0; rotation < rotationsNo; rotation++){

    float min = 999;
    float argmin;

    for(int i = 0; i < diffArrayLengthPerRotation; i++){
      if(diffArray[rotation * diffArrayLengthPerRotation + i] < min){
        min = diffArray[rotation * diffArrayLengthPerRotation + i];
        argmin = i;
      }
    }

    // verify min
    float gpumin = minArray[rotation*templatePointsNo+templatePointsNo-1];
      printf("(%s) RotationNo = %d | CPU argmin = %f (%f)| GPU argmin = %f (%f)\n",(argmin==gpumin?"PASS":"FAIL"),
                      rotation,argmin,min,gpumin,diffArray[rotation*diffArrayLengthPerRotation+(int)gpumin]);
    if(argmin!=gpumin) issues++;

  }

  return issues; 

}
