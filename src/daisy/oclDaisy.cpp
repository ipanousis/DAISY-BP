#include "oclDaisy.h"
#include <sys/time.h>

long int verifyConvolutionY(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample);
long int verifyConvolutionX(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample);
long int verifyTransposeGradientsPartialNorm(float * inputData, float * outputData, int width, int height, int gradientsNo);

// transposition offsets case where petal is left over and cannot be paired
//
// maximum width that this TR_BLOCK_SIZE is effective on (assuming at least 
// TR_DATA_WIDTH rows should be allocated per block) is currently 16384
#define TR_BLOCK_SIZE 512*512
#define TR_DATA_WIDTH 16
#define TR_PAIRS_SINGLE_ONLY -999
#define TR_PAIRS_OFFSET_WIDTH 1000

// Verify all intermediate outputs
//#define DEBUG_ALL

int oclError(const char * function, const char * functionCall, int error){
  if(error){
    fprintf(stderr, "oclDaisy.cpp::%s %s failed: %d\n",function,functionCall,error);
    return error;
  }
  return 0;
}

int initOcl(ocl_daisy_programs * daisy, ocl_constructs * ocl){

  cl_int error;

  // Prepare/Reuse platform, device, context, command queue
  cl_bool recreateBuffers = 0;

  error = buildCachedConstructs(ocl, &recreateBuffers);
  oclError("initOcl", "buildCachedConstructs", error);

  // Pass preprocessor build options
  const char options[128] = "-cl-mad-enable -cl-fast-relaxed-math";

  // Build program
  error = buildCachedProgram(ocl, "daisyKernels.cl", options);
  oclError("initOcl", "buildCachedProgram", error);

  // Prepare kernels

  // Denoising filter
  daisy->kernel_f7x = clCreateKernel(ocl->program, "convolve_x7", &error);
  daisy->kernel_f7y = clCreateKernel(ocl->program, "convolve_y7", &error);  
  oclError("initOcl", "clCreateKernel (f7x)", error);

  // Gradients
  daisy->kernel_gAll = clCreateKernel(ocl->program, "gradient_all", &error);
  oclError("initOcl", "clCreateKernel (gAll)", error);

  // Convolve+Downsample
  daisy->kernel_fxds = clCreateKernel(ocl->program, "convolveDs_x", &error);
  daisy->kernel_fyds = clCreateKernel(ocl->program, "convolveDs_y", &error);
  oclError("initOcl", "clCreateKernel (fyds)", error);

  // Transpose A
  daisy->kernel_trans = clCreateKernel(ocl->program, "transposeGradients", &error);
  oclError("initOcl", "clCreateKernel (trans)", error);

  // Transpose B
  daisy->kernel_transd = clCreateKernel(ocl->program, "transposeDaisy", &error);
  oclError("initOcl", "clCreateKernel (transd)", error);

  // Search
  daisy->kernel_search = clCreateKernel(ocl->program, "searchDaisy", &error);
  oclError("initOcl", "clCreateKernel (search)", error);

  return error;
}

int oclDaisy(daisy_params * daisy, ocl_constructs * ocl, time_params * times){

  cl_int error;

  float * inputArray = (float*)malloc(sizeof(float) * daisy->paddedWidth * daisy->paddedHeight * 8);

  int windowHeight = TR_DATA_WIDTH;
  int windowWidth  = TR_DATA_WIDTH;
  int lclArrayPaddings[3] = {0,0,8};

  int * allPairOffsets[3];
  int allPairOffsetsLengths[3];
  cl_mem allPairOffsetBuffers[3];

  for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

    int petalsNo = daisy->petalsNo + (smoothingNo==0);

    float * petalOffsets = generatePetalOffsets(daisy->sigmas[smoothingNo], daisy->petalsNo, (smoothingNo==0));

    int pairOffsetsLength, actualPairs;
    int * pairOffsets = generateTranspositionOffsets(windowHeight, windowWidth,
                                                     petalOffsets, petalsNo,
                                                     &pairOffsetsLength, &actualPairs);


    cl_mem pairOffsetBuffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
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

  unsigned long int daisySectionSize = daisyBlockWidth * daisyBlockHeight * daisy->totalPetalsNo * daisy->gradientsNo * sizeof(float);

  //
  // End of parameter preparation
  // 

#ifdef DAISY_HOST_TRANSFER


  cl_mem hostPinnedDaisyDescriptors = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
                                                     daisySectionSize, NULL, &error);


  void * daisyDescriptorsSection = (void*)clEnqueueMapBuffer(ocl->queue, hostPinnedDaisyDescriptors, 0,
                                                             CL_MAP_WRITE, 0, daisySectionSize,
                                                             0, NULL, NULL, &error);

  if(totalSections == 1){

    // transfer only to pinned without doing the memcpy to non-pinned
    daisy->descriptors = (float*)daisyDescriptorsSection;

  }
  else{
  
    unsigned long int daisyDescriptorSize = daisy->paddedWidth * daisy->paddedHeight * daisy->totalPetalsNo * daisy->gradientsNo * sizeof(float);

    if(daisy->descriptors == NULL){
      daisy->descriptors = (float*)malloc(daisyDescriptorSize);
    }
    else{
      daisy->descriptors = (float*)realloc(daisy->descriptors, daisyDescriptorSize);
    }
    

  }
  //printf("\nBlock Size calculated (HxW): %dx%d\n",daisyBlockHeight,daisyBlockWidth);
  
#endif

  clFinish(ocl->queue);

  gettimeofday(&times->startFull,NULL);

  int paddedWidth  = daisy->paddedWidth;
  int paddedHeight = daisy->paddedHeight;

  // how much memory will be required to store the temporary output of convolveX for the first layer (the largest layer)
  long int tempPyramidSize = (daisy->paddedHeight * daisy->paddedWidth * daisy->gradientsNo) / 
                              pow(DOWNSAMPLE_RATE / 2, daisy->pyramidLayerSettings[0]->t_downsample);
  tempPyramidSize = max(tempPyramidSize, 2 * daisy->paddedHeight * daisy->paddedWidth);

  // either the non-downsampled gradients plus the first half-downsampled convolution output will be greater
  long int gradientsPlusTempSize = (daisy->paddedHeight * daisy->paddedWidth * daisy->gradientsNo) + tempPyramidSize;

  // or the whole pyramid plus the first half-downsampled convolution output,
  // it will normally be the former except for the case when the convolution outputs happen to not be downsampled enough
  long int pyramidBufferSize = max(daisy->pyramidLayerOffsets[daisy->smoothingsNo-1] + 
                                   daisy->pyramidLayerSizes[daisy->smoothingsNo-1] + 
                                   tempPyramidSize, gradientsPlusTempSize) * sizeof(float);

  cl_mem pyramidBuffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                        pyramidBufferSize, (void*)NULL, &error);

  oclError("oclDaisy","clCreateBuffer (1)",error);

  printf("pyramidBuffer size = %ld (%ldMB)\n", pyramidBufferSize, pyramidBufferSize / (1024 * 1024));
  printf("tempPyramid size = %ld (%ldMB)\n", tempPyramidSize, tempPyramidSize / (1024 * 1024));
  printf("paddedWidth = %d, paddedHeight = %d\n", paddedWidth, paddedHeight);

  cl_mem * filterBuffers = (cl_mem*) malloc(sizeof(cl_mem) * (daisy->smoothingsNo + 1));

  float denoiseSigma = 0.5;
  int denoiseSize = (int)((int)floor(denoiseSigma * 5) % 2 ? floor(denoiseSigma * 5) : floor(denoiseSigma * 5) + 1);
  float * denoiseFilter = (float*)malloc(sizeof(float) * denoiseSize);
  kutility::gaussian_1d(denoiseFilter,denoiseSize,denoiseSigma,0);

  filterBuffers[0] = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
                                    denoiseSize * sizeof(float),
                                    (void*)NULL, &error);

  clEnqueueWriteBuffer(ocl->queue, filterBuffers[0], CL_FALSE,
                       0, denoiseSize * sizeof(float), (void*)denoiseFilter,
                       0, NULL, NULL);

  // this does not have any impact on the pyramid layer, since the total sigma reached after the denoising + this new sigma
  // will be the same as the intended daisy->sigmas[0], but now we will have run a small denoising filter before we take the
  // gradient on the data
  daisy->pyramidLayerSettings[0]->sigma = sqrt(pow(daisy->sigmas[0],2) - pow(denoiseSigma,2));

  int * gaussianFilterSizes = (int*) malloc(sizeof(int) * daisy->smoothingsNo);
  float ** gaussianFilters = (float**) malloc(sizeof(float*) * daisy->smoothingsNo);
  for(int s = 0; s < daisy->smoothingsNo; s++){

    float sigma = daisy->pyramidLayerSettings[s]->sigma;
    gaussianFilterSizes[s] = (int)((int)floor(sigma * 5) % 2 ? floor(sigma * 5) : floor(sigma * 5) + 1);
    gaussianFilters[s] = (float*) malloc(sizeof(float) * gaussianFilterSizes[s]);
    kutility::gaussian_1d(gaussianFilters[s],gaussianFilterSizes[s],sigma,0);

    filterBuffers[s+1] = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
                                        gaussianFilterSizes[s] * sizeof(float),
                                        (void*)NULL, &error);

    clEnqueueWriteBuffer(ocl->queue, filterBuffers[s+1], CL_FALSE,
                         0, gaussianFilterSizes[s] * sizeof(float), (void*)gaussianFilters[s],
                         0, NULL, NULL);

#ifdef DEBUG_ALL
      printf("Displaying gaussian filter %d to reach total pyramid sigma %f with sigma %f\n",s,daisy->pyramidLayerSettings[s]->t_sigma,sigma);
      for(int i = 0; i < gaussianFilterSizes[s]; i++)
        printf("%f, ",gaussianFilters[s][i]);
      printf("\n\n");
#endif
  }

  oclError("oclDaisy","clEnqueueWriteBuffer (1)",error);


#ifdef DEBUG_ALL
  float * testArray;
  testArray = (float*)malloc(sizeof(float) * paddedWidth * 200 * 16);
#endif

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

  error = clEnqueueWriteBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                               0, paddedWidth * paddedHeight * sizeof(float),
                               (void*)inputArray,
                               0, NULL, NULL);

  oclError("oclDaisy","clEnqueueWriteBuffer (2)",error);

  cl_int k;

  gettimeofday(&times->startConvGrad,NULL);

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)
  size_t convWorkerSize7x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight};
  size_t convGroupSize7x[2] = {16,8};

  // convolve X - A.0 to A.1
  clSetKernelArg(daisy->oclPrograms->kernel_f7x, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
  clSetKernelArg(daisy->oclPrograms->kernel_f7x, 1, sizeof(filterBuffers[0]), (void*)&filterBuffers[0]);
  clSetKernelArg(daisy->oclPrograms->kernel_f7x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms->kernel_f7x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_f7x, 2, NULL, 
                                 convWorkerSize7x, convGroupSize7x, 0, 
                                 NULL, NULL);

  oclError("oclDaisy","clEnqueueNDRangeKernel (1)",error);

  error = clFinish(ocl->queue);
  oclError("oclDaisy","clFinish (1)",error);

#ifdef DEBUG_ALL

    // checked verified
    error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), testArray,
                        0, NULL, NULL);

    oclError("oclDaisy","clEnqueueReadBuffer (0)",error);

    printf("\nDenoising Input x: %f",inputArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[k]);
    printf("\n");
    printf("\nDenoising Output x: %f",testArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[k]);
    printf("\n");

#endif

  // convolve Y - A.1 to A.0
  size_t convWorkerSize7y[2] = {daisy->paddedWidth,daisy->paddedHeight / 4};
  size_t convGroupSize7y[2] = {16,8};

  clSetKernelArg(daisy->oclPrograms->kernel_f7y, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
  clSetKernelArg(daisy->oclPrograms->kernel_f7y, 1, sizeof(filterBuffers[0]), (void*)&filterBuffers[0]);
  clSetKernelArg(daisy->oclPrograms->kernel_f7y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms->kernel_f7y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_f7y, 2, 
                                 NULL, convWorkerSize7y, convGroupSize7y, 
                                 0, NULL, NULL);

  oclError("oclDaisy","clEnqueueNDRangeKernel (2)",error);

  error = clFinish(ocl->queue);
  oclError("oclDaisy","clFinish (2)",error);

#ifdef DEBUG_ALL
    // checked verified
    error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                0, 
                                paddedWidth * paddedHeight * sizeof(float), inputArray,
                                0, NULL, NULL);

    printf("\nDenoising Input y: %f",testArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[k*paddedWidth]);
    printf("\n");
    printf("\nDenoising Output y: %f",inputArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[k*paddedWidth]);
    printf("\n");
#endif

  // gradients for 8 orientations
  size_t gradWorkerSize = daisy->paddedWidth * daisy->paddedHeight;
  size_t gradGroupSize = 64;

  // gradient X,Y,all - A.0 to B+
  clSetKernelArg(daisy->oclPrograms->kernel_gAll, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
  clSetKernelArg(daisy->oclPrograms->kernel_gAll, 1, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms->kernel_gAll, 2, sizeof(int), (void*)&(daisy->paddedHeight));
  clSetKernelArg(daisy->oclPrograms->kernel_gAll, 3, sizeof(int), (void*)&(tempPyramidSize));

  error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_gAll, 1, NULL, 
                                 &gradWorkerSize, &gradGroupSize, 0, 
                                 NULL, NULL);

  oclError("oclDaisy","clEnqueueNDRangeKernel (3)",error);

  clFinish(ocl->queue);

#ifdef DEBUG_ALL
    error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                tempPyramidSize * sizeof(float), 
                                paddedWidth * paddedHeight * sizeof(float), testArray,
                                0, NULL, NULL);
    clFinish(ocl->queue);
    printf("\nBefore Gradient: %f",inputArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[k]);
    printf("\nAfter Gradient (X,X+22.5,X+45,Y): %f",testArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[k]);
    printf("\n");
#endif

  gettimeofday(&times->startConvGrad,NULL);

  // Smooth + Downsample all data - end up with a pyramid volume in pyramidBuffer    
  for(int layer = 0; layer < daisy->smoothingsNo; layer++){

    pyramid_layer_set* layerSettings = daisy->pyramidLayerSettings[layer];

    int totalDownsample = pow(DOWNSAMPLE_RATE / 2, (layer > 0?daisy->pyramidLayerSettings[layer-1]->t_downsample:0));

    int filterRadius = gaussianFilterSizes[layer] / 2;
    int filterDownsample = pow(DOWNSAMPLE_RATE / 2, layerSettings->downsample);

    int layerWidth = daisy->paddedWidth / totalDownsample;
    int layerHeight = daisy->paddedHeight / totalDownsample;

    int pyramidLayerOffset = tempPyramidSize + (layer > 0?daisy->pyramidLayerOffsets[layer-1]:0);

    // kernel X
    size_t convWorkerSizeX[2] = {layerWidth / 4, layerHeight * daisy->gradientsNo};
    size_t convGroupSizeX[2] = {16,4};

    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 1, sizeof(int), (void*)&layerWidth);
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 2, sizeof(int), (void*)&layerHeight);
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 3, sizeof(int), (void*)&pyramidLayerOffset);
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 4, sizeof(filterBuffers[layer+1]), (void*)&(filterBuffers[layer+1]));
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 5, sizeof(int), (void*)&filterRadius);
    clSetKernelArg(daisy->oclPrograms->kernel_fxds, 6, sizeof(int), (void*)&filterDownsample);

    error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_fxds, 2, NULL, 
                                   convWorkerSizeX, convGroupSizeX, 0, 
                                   NULL, NULL);

    oclError("oclDaisy","clEnqueueNDRangeKernel (4)",error);

    clFinish(ocl->queue);

    layerWidth = layerWidth / filterDownsample;

#ifdef DEBUG_ALL

    for(int g = 0; g < daisy->gradientsNo; g++){
      error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                  (pyramidLayerOffset + g * (layerWidth * filterDownsample) * layerHeight) * sizeof(float), 
                                  (layerWidth * filterDownsample) * layerHeight * sizeof(float), testArray,
                                  0, NULL, NULL);

      error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                  g * layerWidth * layerHeight * sizeof(float), 
                                  layerWidth * layerHeight * sizeof(float), inputArray,
                                  0, NULL, NULL);
      clFinish(ocl->queue);
    
      long int issues = verifyConvolutionX(testArray, inputArray, layerHeight, layerWidth * filterDownsample, gaussianFilters[layer], gaussianFilterSizes[layer], filterDownsample);

      if(issues > 0) printf("convolveDs_x has %ld issues with smoothing %d at g=%d\n",issues,layer,g);
      else if(g == daisy->gradientsNo-1) printf("convolveDs_x in smoothing %d is ok!\n",layer);
    }

#endif

    // kernelY
    size_t convWorkerSizeY[2] = {layerWidth, (layerHeight * daisy->gradientsNo) / 4};
    size_t convGroupSizeY[2] = {16,8};

    pyramidLayerOffset = tempPyramidSize + daisy->pyramidLayerOffsets[layer];

    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 1, sizeof(int), (void*)&layerWidth);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 2, sizeof(int), (void*)&layerHeight);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 3, sizeof(int), (void*)&pyramidLayerOffset);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 4, sizeof(filterBuffers[layer+1]), (void*)&filterBuffers[layer+1]);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 5, sizeof(int), (void*)&filterRadius);
    clSetKernelArg(daisy->oclPrograms->kernel_fyds, 6, sizeof(int), (void*)&filterDownsample);

    error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_fyds, 2, NULL, 
                                   convWorkerSizeY, convGroupSizeY, 0, 
                                   NULL, NULL);

    oclError("oclDaisy","clEnqueueNDRangeKernel (5)",error);

    clFinish(ocl->queue);

    layerHeight = layerHeight / filterDownsample;

#ifdef DEBUG_ALL

    for(int g = 0; g < daisy->gradientsNo; g++){
      error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                  g * layerWidth * (layerHeight * filterDownsample) * sizeof(float), 
                                  layerWidth * (layerHeight * filterDownsample) * sizeof(float), testArray,
                                  0, NULL, NULL);

      error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                  (pyramidLayerOffset + g * layerWidth * layerHeight) * sizeof(float), 
                                  layerWidth * layerHeight * sizeof(float), inputArray,
                                  0, NULL, NULL);
      clFinish(ocl->queue);
    
      //long int verifyConvolutionY(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample){
      long int issues = verifyConvolutionY(testArray, inputArray, layerHeight * filterDownsample, layerWidth, gaussianFilters[layer], gaussianFilterSizes[layer], filterDownsample);

      if(issues > 0) printf("convolveDs_y has %ld issues with smoothing %d at g=%d\n",issues,layer,g);
      else if(g == daisy->gradientsNo-1) printf("convolveDs_y in smoothing %d is ok!\n",layer);
    }

    printf("End of SMOOTH+DS %d, HxW = %dx%d --> %dx%d\n",layer,layerHeight*filterDownsample,layerWidth*filterDownsample,layerHeight,layerWidth);

#endif


  }

  // A) transpose pyramid from SxGxHxW to SxHxWxG

  gettimeofday(&times->startTransGrad,NULL);

  long int memorySize = (daisy->pyramidLayerOffsets[daisy->smoothingsNo-1] + 
                         daisy->pyramidLayerSizes[daisy->smoothingsNo-1]) * sizeof(float);

  cl_mem transBuffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                      memorySize, (void*)NULL, &error);

  oclError("oclDaisy","clCreateBuffer (3)",error);

  printf("\ntransBuffer size = %ld (%ldMB)\n",memorySize,memorySize/(1024*1024));


  for(int layer = 0; layer < daisy->smoothingsNo; layer++){
    
    int totalDownsample = pow(sqrt(DOWNSAMPLE_RATE), daisy->pyramidLayerSettings[layer]->t_downsample);
    int layerWidth = daisy->paddedWidth / totalDownsample;
    int layerHeight = daisy->paddedHeight / totalDownsample;

    size_t transWorkerSize[2] = {layerWidth, layerHeight * daisy->gradientsNo};
    size_t transGroupSize[2] = {32,8};

    int pyramidLayerOffset = daisy->pyramidLayerOffsets[layer];

    int srcOffset = tempPyramidSize + pyramidLayerOffset;
    int dstOffset = pyramidLayerOffset;

    printf("Layer Size HxW = %dx%d\n",layerHeight,layerWidth);

    clSetKernelArg(daisy->oclPrograms->kernel_trans, 0, sizeof(pyramidBuffer), (void*)&pyramidBuffer);
    clSetKernelArg(daisy->oclPrograms->kernel_trans, 1, sizeof(transBuffer), (void*)&transBuffer);
    clSetKernelArg(daisy->oclPrograms->kernel_trans, 2, sizeof(int), (void*)&layerWidth);
    clSetKernelArg(daisy->oclPrograms->kernel_trans, 3, sizeof(int), (void*)&layerHeight);
    clSetKernelArg(daisy->oclPrograms->kernel_trans, 4, sizeof(int), (void*)&srcOffset);
    clSetKernelArg(daisy->oclPrograms->kernel_trans, 5, sizeof(int), (void*)&dstOffset);

    error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_trans, 2, 
                                   NULL, transWorkerSize, transGroupSize,
                                   0, NULL, NULL);

    oclError("oclDaisy","clEnqueueNDRangeKernel (10)",error);

    clFinish(ocl->queue);



#ifdef DEBUG_ALL

    error = clEnqueueReadBuffer(ocl->queue, pyramidBuffer, CL_TRUE,
                                srcOffset * sizeof(float), 
                                layerWidth * layerHeight * daisy->gradientsNo * sizeof(float), inputArray,
                                0, NULL, NULL);

    error = clEnqueueReadBuffer(ocl->queue, transBuffer, CL_TRUE,
                                dstOffset * sizeof(float), 
                                layerWidth * layerHeight * daisy->gradientsNo * sizeof(float), testArray,
                                0, NULL, NULL);

    clFinish(ocl->queue);
  
    long int issues = verifyTransposeGradientsPartialNorm(inputArray, testArray, layerWidth, layerHeight, daisy->gradientsNo);

    if(issues > 0) printf("%ld issues with transposeGradients in layer %d\n",issues,layer);

#endif

  }

  clReleaseMemObject(pyramidBuffer);

  gettimeofday(&times->endTransGrad,NULL);
  gettimeofday(&times->endConvGrad,NULL);

#ifndef DAISY_NO_DESCRIPTORS

  // B) final transposition

  gettimeofday(&times->startTransDaisy,NULL);

  //printf("\nAllocated %ld bytes on GPU for daisy section buffer (%ldMB)\n",daisySectionSize,daisySectionSize/(1024*1024));

  //printf("\nAllocated %ld bytes on host for daisy descriptor array (%ldMB)\n",daisyDescriptorSize,daisyDescriptorSize/(1024*1024));
  
  cl_mem daisyBufferA, daisyBufferB;

  daisyBufferA = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
                                daisySectionSize,(void*)NULL, &error);

  oclError("oclDaisy","clCreateBuffer (daisybufferA)",error);

  if(totalSections > 1)
    daisyBufferB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
                                  daisySectionSize,(void*)NULL, &error);

  oclError("oclDaisy","clCreateBuffer (daisybufferB)",error);

  size_t daisyGroupSize[2] = {windowWidth,windowHeight};

  cl_event * memoryEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections);
  cl_event * kernelEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections * daisy->smoothingsNo);

  clFinish(ocl->queue);

  // for each 512x512 section... (sequentially first)
    // for each sigma...
  for(int sectionNo = 0; sectionNo < totalSections; sectionNo++){

    int sectionY = sectionNo;
    int sectionX = 0;

    int sectionWidth = daisyBlockWidth;
    int sectionHeight = (sectionNo < totalSections-1 ? daisyBlockHeight : 
                        (daisy->paddedHeight % daisyBlockHeight ? daisy->paddedHeight % daisyBlockHeight : daisyBlockHeight));

    size_t daisyWorkerOffsets[2] = {sectionX * daisyBlockWidth, sectionY * daisyBlockHeight};

    // undefined behaviour when sectionHeight is not a multiple of 16!!
    size_t daisyWorkerSize[2] = {sectionWidth + 2 * daisyGroupSize[0], sectionHeight + 2 * daisyGroupSize[1]};
      
    short int resourceContext = sectionNo%2;

    cl_event * prevMemoryEvents = NULL;
    cl_event * currMemoryEvents = &memoryEvents[sectionNo];
    cl_event * prevKernelEvents = NULL;
    cl_event * currKernelEvents = &kernelEvents[sectionNo * daisy->smoothingsNo];

    if(resourceContext != sectionNo){
      prevMemoryEvents = &memoryEvents[sectionNo-2];
      prevKernelEvents = &kernelEvents[(sectionNo-2) * daisy->smoothingsNo];
    }

    cl_mem daisyBuffer = (!resourceContext ? daisyBufferA : daisyBufferB);

    for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

      

      int pairOffsetsLength = allPairOffsetsLengths[smoothingNo];

      //printf("Running daisy section (%d,%d) out of %d - smoothing %d\n",sectionY,sectionX,totalSections,smoothingNo);

      pyramid_layer_set * layerSettings = daisy->pyramidLayerSettings[smoothingNo];

      //int srcGlobalOffset = daisy->paddedHeight * daisy->paddedWidth * daisy->gradientsNo * smoothingNo;
      int srcGlobalOffset = daisy->pyramidLayerOffsets[smoothingNo];
      int totalDownsample = pow(sqrt(DOWNSAMPLE_RATE), layerSettings->t_downsample);

      int petalStart = smoothingNo * 8 + (smoothingNo > 0);

      clSetKernelArg(daisy->oclPrograms->kernel_transd, 0, sizeof(transBuffer), (void*)&transBuffer);
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 1, sizeof(daisyBuffer), (void*)&daisyBuffer);
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 2, sizeof(allPairOffsetBuffers[smoothingNo]), (void*)&allPairOffsetBuffers[smoothingNo]);
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 3, sizeof(float) * (windowHeight * (windowWidth * daisy->gradientsNo)), 0);
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 4, sizeof(int), (void*)&(daisy->paddedWidth));
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 5, sizeof(int), (void*)&(daisy->paddedHeight));
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 6, sizeof(int), (void*)&(srcGlobalOffset));
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 7, sizeof(int), (void*)&(pairOffsetsLength));
      //clSetKernelArg(daisy->oclPrograms->kernel_transd, 8, sizeof(int), (void*)&(lclArrayPaddings[smoothingNo]));
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 8, sizeof(int), (void*)&totalDownsample);
      clSetKernelArg(daisy->oclPrograms->kernel_transd, 9, sizeof(int), (void*)&petalStart);

      error = clEnqueueNDRangeKernel(ocl->queue, daisy->oclPrograms->kernel_transd, 2,
                                     daisyWorkerOffsets, daisyWorkerSize, daisyGroupSize,
                                     (resourceContext!=sectionNo),
                                     prevMemoryEvents,
                                     &currKernelEvents[smoothingNo]);

      oclError("oclDaisy","clEnqueueNDRangeKernel (daisy block)",error);

    }

#ifdef DAISY_HOST_TRANSFER
    int y;

    if(sectionNo > 0){

      int descriptorsOffset = ((sectionY-1) * daisyBlockHeight * daisyBlockWidth + 
                               sectionX * daisyBlockWidth) * daisy->totalPetalsNo * daisy->gradientsNo;

      clWaitForEvents(1, currMemoryEvents-1);

      #pragma omp parallel for private(y)
      for(y = 0; y < daisyBlockHeight; y++)
        memcpy(daisy->descriptors + descriptorsOffset + y * daisyBlockWidth * 200, 
               (float*)daisyDescriptorsSection + y * daisyBlockWidth * 200, 
               daisyBlockWidth * 200 * sizeof(float));
        
    }

    error = clEnqueueReadBuffer(ocl->queue, daisyBuffer, 0,
                                0, daisySectionSize, daisyDescriptorsSection,
                                3, currKernelEvents, &currMemoryEvents[0]);

    oclError("oclDaisy","clEnqueueReadBuffer (async)",error);

    if(sectionNo == totalSections-1 && sectionNo > 0){

      int descriptorsOffset = (sectionY * daisyBlockHeight * daisyBlockWidth + 
                               sectionX * daisyBlockWidth) * daisy->totalPetalsNo * daisy->gradientsNo;

      clWaitForEvents(1, currMemoryEvents);

      #pragma omp parallel for private(y)
      for(y = 0; y < sectionHeight; y++)
        memcpy(daisy->descriptors + descriptorsOffset + y * daisyBlockWidth * 200, 
               (float*)daisyDescriptorsSection + y * daisyBlockWidth * 200, 
               daisyBlockWidth * 200 * sizeof(float));

    }

#else
    
    currMemoryEvents[0] = currKernelEvents[2];

#endif

  }

  error = clFinish(ocl->queue);
  oclError("oclDaisy","clFinish (daisy)",error);

  //
  // VERIFICATION CODE
  //

  int issues = 0;

  // Verify transposition b)
  int TESTING_TRANSD = times->measureDeviceHostTransfers;
  if(TESTING_TRANSD){


    for(int block = 0; block < totalSections; block++){

      int sectionY = block;
      int sectionX = 0;

      int sectionWidth = daisyBlockWidth;
      int sectionHeight = (block < totalSections-1 ? daisyBlockHeight : 
                          (daisy->paddedHeight % daisyBlockHeight ? daisy->paddedHeight % daisyBlockHeight : daisyBlockHeight));

      for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

        int pairOffsetsLength = allPairOffsetsLengths[smoothingNo];
        int * pairOffsets = allPairOffsets[smoothingNo];

        int petalStart = (smoothingNo > 0 ? smoothingNo * daisy->petalsNo + 1 : 0);

        //printf("\nPetalsNo = %d\n",petalsNo);
        float * daisyArray = daisy->descriptors + block * daisyBlockHeight * daisyBlockWidth * 200;

        clFinish(ocl->queue);
        /*for(int r = 0; r < 16; r+=16){
          for(k = petalStart * 8; k < (petalStart+2) * 8; k++)
            printf(", %f",daisyArray[r * daisyBlockWidth * 200 + k]);
          printf("\n\n");
        }*/

        error = clEnqueueReadBuffer(ocl->queue, transBuffer, CL_TRUE,
                                    paddedWidth * paddedHeight * 8 * smoothingNo * sizeof(float), 
                                    paddedWidth * paddedHeight * 8 * sizeof(float), inputArray,
                                    0, NULL, NULL);

        clFinish(ocl->queue);

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

  // Release and unmap buffers, free allocated space
#ifdef DAISY_HOST_TRANSFER

  // don't unmap the pinned memory if there was only one block
  //if(totalSections > 1){

  //error = clEnqueueUnmapMemObject(ocl->queue, hostPinnedDaisyDescriptors, daisyDescriptorsSection, 0, NULL, NULL);	
    oclError("oclDaisy","clEnqueueUnmapMemObject (hostPinnedSection)",error);
  //free(daisyDescriptorsSection);
  //}

  //clReleaseMemObject(hostPinnedDaisyDescriptors);

#endif

  clReleaseMemObject(filterBuffers[0]);

  for(int s = 0; s < daisy->smoothingsNo; s++){
    clReleaseMemObject(allPairOffsetBuffers[s]);
    free(allPairOffsets[s]);
  }

  clReleaseMemObject(daisyBufferA);
  if(totalSections > 1) clReleaseMemObject(daisyBufferB);

  free(memoryEvents);
  free(kernelEvents);

#endif

  gettimeofday(&times->endTransDaisy,NULL);

  gettimeofday(&times->endFull,NULL);

  times->startt = times->startFull.tv_sec+(times->startFull.tv_usec/1000000.0);
  times->endt = times->endFull.tv_sec+(times->endFull.tv_usec/1000000.0);
  times->difft = times->endt-times->startt;
  printf("\nDaisyFull: %.4fs (%.4f MPixel/sec)\n",times->difft,(daisy->paddedWidth*daisy->paddedHeight) / (1000000.0f*times->difft));

  times->startt = times->startConvGrad.tv_sec+(times->startConvGrad.tv_usec/1000000.0);
  times->endt = times->endConvGrad.tv_sec+(times->endConvGrad.tv_usec/1000000.0);
  times->difft = times->endt-times->startt;
  printf("\nconvds: %.4fs (%.4f MPixel/sec)\n",times->difft,(daisy->paddedWidth*daisy->paddedHeight*8*3) / (1000000.0f*times->difft));

  free(inputArray);

#ifdef DEBUG_ALL
  free(testArray);
#endif

#ifndef DAISY_SEARCH
  clReleaseMemObject(transBuffer);
#else
  daisy->oclBuffers->transBuffer = transBuffer;
#endif

  return error;
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

long int verifyConvolutionX(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample){

  long int issues = 0;
  int limit = 40;

  for(int y = 0; y < height; y++){

    for(int x = 0; x < width; x += downsample){

      float s = .0f;
      for(int f = -filterSize / 2; f < filterSize / 2 + 1; f++){
        if(x + f < 0) s += inputData[y * width] * filter[f + filterSize / 2];
        else if(x + f >= width) s += inputData[y * width + width-1] * filter[f + filterSize / 2];
        else s += inputData[y * width + x + f] * filter[f + filterSize / 2];
      }

      if(fabs(outputData[y * (width / downsample) + x / downsample] - s) > 0.0001f){
        issues++;
        if(issues < limit)
          printf("Issue at y,x %d,%d\n",y,x);
      }
    }

  }
  
  return issues;

}
long int verifyConvolutionY(float * inputData, float * outputData, int height, int width, float * filter, int filterSize, int downsample){

  long int issues = 0;
  int limit = 40;

  for(int x = 0; x < width; x++){

    for(int y = 0; y < height; y += downsample){

      float s = .0f;
      for(int f = -filterSize / 2; f < filterSize / 2 + 1; f++){
        if(y + f < 0) s += inputData[x] * filter[f + filterSize / 2];
        else if(y + f >= height) s += inputData[(height-1) * width + x] * filter[f + filterSize / 2];
        else s += inputData[(y + f) * width + x] * filter[f + filterSize / 2];
      }

      if(fabs(outputData[(y / downsample) * width + x] - s) > 0.0001f){
        issues++;
        if(issues < limit)
          printf("Issue at y,x %d,%d; found %f should be %f\n",y,x,outputData[(y / downsample) * width + x],s);
      }
    }
  }

  return issues;

}

long int verifyTransposeGradientsPartialNorm(float * inputData, float * outputData, int width, int height, int gradientsNo){

  long int issues = 0;
  int limit = 40;

  for(int g = 0; g < gradientsNo; g++){

    for(int y = 0; y < height; y++){

      for(int x = 0; x < width; x++){

        float inputValue = .0f;
        for(int i = 0; i < gradientsNo; i++){
          const float in = inputData[i * width * height + y * width + x];;
          inputValue += in*in;
        }
        float val = inputData[g * width * height + y * width + x];
        inputValue = (inputValue == 0.0 ? val : val / sqrt(inputValue));

//        inputValue = inputData[g * width * height + y * width + x];
        float outputValue = outputData[y * width * gradientsNo + x * gradientsNo + g];

        issues += fabs(inputValue - outputValue) > 0.0001f;
        if(issues > 0 && issues++ < limit)
          printf("Issue at y,x %d,%d; found %f should be %f\n",y,x,outputValue,inputValue);

      }

    }

  }

  return issues;
}
