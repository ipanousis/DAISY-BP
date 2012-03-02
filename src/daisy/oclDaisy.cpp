#include "oclDaisy.h"
#include <omp.h>
#include <sys/time.h>

float * generatePetalOffsets(float, int, short int);

int * generateTranspositionOffsets(int, int, float*, int, int*, int*);

// transposition offsets case where petal is left over and cannot be paired
//
// maximum width that this TR_BLOCK_SIZE is effective on (assuming at least 
// TR_DATA_WIDTH rows should be allocated per block) is currently 16384
#define ARRAY_PADDING 64
#define TR_BLOCK_SIZE 512*512
#define TR_DATA_WIDTH 16
#define TR_PAIRS_SINGLE_ONLY -999
#define TR_PAIRS_OFFSET_WIDTH 1000

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

int initOcl(ocl_daisy_programs * daisy, ocl_constructs * daisyCl){

  cl_int error;

  // Prepare/Reuse platform, device, context, command queue
  cl_bool recreateBuffers = 0;

  error = buildCachedConstructs(daisyCl, &recreateBuffers);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedConstructs returned %d, cannot continue\n",error);
    return 1;
  }

  // Pass preprocessor build options
  const char options[128] = "-cl-mad-enable -cl-fast-relaxed-math";

  // Build denoising filter
  error = buildCachedProgram(daisyCl, "daisyKernels.cl", options);
  
  if(daisyCl->program == NULL){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
    return 1;
  }

  // Prepare the kernel
  daisy->kernel_f7x = clCreateKernel(daisyCl->program, "convolve_x7", &error);
  daisy->kernel_f7y = clCreateKernel(daisyCl->program, "convolve_y7", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  // Build gradient kernel
  
  daisy->kernel_gAll = clCreateKernel(daisyCl->program, "gradient_8all", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->kernel_f11x = clCreateKernel(daisyCl->program, "convolve_x11", &error);
  daisy->kernel_f11y = clCreateKernel(daisyCl->program, "convolve_y11", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->kernel_f23x = clCreateKernel(daisyCl->program, "convolve_x23", &error);
  daisy->kernel_f23y = clCreateKernel(daisyCl->program, "convolve_y23", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->kernel_f29x = clCreateKernel(daisyCl->program, "convolve_x29", &error);
  daisy->kernel_f29y = clCreateKernel(daisyCl->program, "convolve_y29", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  daisy->kernel_trans = clCreateKernel(daisyCl->program, "transposeGradients", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  daisy->kernel_transd = clCreateKernel(daisyCl->program, "transposeDaisy", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  return error;

}

int oclError(const char * function, const char * functionCall, int error){
  if(error){
    fprintf(stderr, "oclDaisy.cpp::%s %s failed: %d\n",function,functionCall,error);
    return error;
  }
  return 0;
}

int oclDaisy(daisy_params * daisy, ocl_constructs * daisyCl, time_params * times){

  cl_int error;

  daisy->paddedWidth = daisy->width + (ARRAY_PADDING - daisy->width % ARRAY_PADDING) % ARRAY_PADDING;
  daisy->paddedHeight = daisy->height + (ARRAY_PADDING - daisy->height % ARRAY_PADDING) % ARRAY_PADDING;

  float * inputArray = (float*)malloc(sizeof(float) * daisy->paddedWidth * daisy->paddedHeight * 8);

  int windowHeight = TR_DATA_WIDTH;
  int windowWidth  = TR_DATA_WIDTH;

  float sigmas[3] = {2.5,5,7.5};
  int lclArrayPaddings[3] = {0,0,8};

  int * allPairOffsets[3];
  int allPairOffsetsLengths[3];
  cl_mem allPairOffsetBuffers[3];

  for(int smoothingNo = 0; smoothingNo < daisy->smoothingsNo; smoothingNo++){

    int petalsNo = daisy->petalsNo + (smoothingNo==0);

    float * petalOffsets = generatePetalOffsets(sigmas[smoothingNo], daisy->petalsNo, (smoothingNo==0));

    int pairOffsetsLength, actualPairs;
    int * pairOffsets = generateTranspositionOffsets(windowHeight, windowWidth,
                                                     petalOffsets, petalsNo,
                                                     &pairOffsetsLength, &actualPairs);


    cl_mem pairOffsetBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             pairOffsetsLength * 4 * sizeof(int), (void*)pairOffsets, &error);

    //oclError("oclDaisy","clCreateBuffer (pairOffset)",error);

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

  clFinish(daisyCl->queue);

  gettimeofday(&times->startFull,NULL);

  int paddedWidth  = daisy->paddedWidth;
  int paddedHeight = daisy->paddedHeight;

  long int memorySize = daisy->gradientsNo * (daisy->smoothingsNo+1) * 
                        paddedWidth * paddedHeight * sizeof(cl_float);

  cl_mem massBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                     memorySize, (void*)NULL, &error);

  //oclError("oclDaisy","clCreateBuffer (1)",error);

  //printf("massBuffer size = %ld (%ldMB)\n", memorySize, memorySize / (1024 * 1024));
  //printf("paddedWidth = %d, paddedHeight = %d\n", paddedWidth, paddedHeight);

  int filter5Size = 5;
  float * filter5 = (float*)malloc(sizeof(float)*filter5Size);
  kutility::gaussian_1d(filter5,filter5Size,0.5,0);
  int filter7Size = 7;
  /*float filter7[7] = {0.036633,0.11128,0.21675,
                      0.27068,
                      0.21675,0.11128,0.036633};*/
  int filter11Size = 11;
  float filter11[11] = {0.007024633,0.02381049,0.06153227,0.1212349,0.1821137,
                        0.2085680,
                        0.1821137,0.1212349,0.06153227,0.02381049,0.007024633};
  int filter23Size = 23;
  float filter23[23] = {0.00368473,0.00645096,0.01070729,0.01684890,0.02513619,0.03555197,0.04767209,0.06060396,0.07304224,0.08346096,0.09041261,
                        0.09285619,
                        0.09041261,0.08346096,0.07304224,0.06060396,0.04767209,0.03555197,0.02513619,0.01684890,0.01070729,0.00645096,0.00368473};
  int filter29Size = 29;
  float filter29[29] = {0.00316014,0.00486031,0.00724056,0.01044798,0.01460304,0.01976996,0.02592504,
                        0.03292945,0.04051361,0.04828015,0.05572983,0.06231004,0.06748073,0.07078688,
                        0.07192454,
                        0.07078688,0.06748073,0.06231004,0.05572983,0.04828015,0.04051361,0.03292945,
                        0.02592504,0.01976996,0.01460304,0.01044798,0.00724056,0.00486031,0.00316014};

  const int filterOffsets[4] = {0, filter7Size, filter7Size + filter11Size,
                                   filter7Size + filter11Size + filter23Size};

  cl_mem filterBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY,
                                       (filter7Size + filter11Size + filter23Size + filter29Size) * sizeof(float),
                                       (void*)NULL, &error);

  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[0] * sizeof(float), filter5Size * sizeof(float), (void*)filter5,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[1] * sizeof(float), filter11Size * sizeof(float), (void*)filter11,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[2] * sizeof(float), filter23Size * sizeof(float), (void*)filter23,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE,
                       filterOffsets[3] * sizeof(float), filter29Size * sizeof(float), (void*)filter29,
                       0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueWriteBuffer (1)",error);

  short int DEBUG_ALL = 0;

  float * testArray;

  if(DEBUG_ALL){
    testArray = (float*)malloc(sizeof(float) * paddedWidth * 200 * 16);
  }

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

  //oclError("oclDaisy","clEnqueueWriteBuffer (2)",error);

  cl_int k;

  gettimeofday(&times->startConvGrad,NULL);

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)
  size_t convWorkerSize7x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight};
  size_t convGroupSize7x[2] = {16,8};

  // convolve X - A.0 to A.1
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7x, 2, NULL, 
                                 convWorkerSize7x, convGroupSize7x, 0, 
                                 NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (1)",error);

  error = clFinish(daisyCl->queue);
  //oclError("oclDaisy","clFinish (1)",error);

  if(DEBUG_ALL){
    // checked verified
    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), testArray,
                        0, NULL, NULL);

    //oclError("oclDaisy","clEnqueueReadBuffer (0)",error);

    printf("\nDenoising Input x: %f",inputArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[k]);
    printf("\n");
    printf("\nDenoising Output x: %f",testArray[0]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[k]);
    printf("\n");
  }

  // convolve Y - A.1 to B.0
  size_t convWorkerSize7y[2] = {daisy->paddedWidth,daisy->paddedHeight / 4};
  size_t convGroupSize7y[2] = {16,8};

  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7y, 2, 
                                 NULL, convWorkerSize7y, convGroupSize7y, 
                                 0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (2)",error);

  error = clFinish(daisyCl->queue);
  //oclError("oclDaisy","clFinish (2)",error);

  if(DEBUG_ALL){
    // checked verified
    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * 8 * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), inputArray,
                        0, NULL, NULL);

    printf("\nDenoising Input y: %f",testArray[159+58*paddedWidth]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[(58+k)*paddedWidth+159]);
    printf("\n");
    printf("\nDenoising Output y: %f",inputArray[159+58*paddedWidth]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[(58+k)*paddedWidth+159]);
    printf("\n");
  }

  // gradients for 8 orientations
  size_t gradWorkerSize = daisy->paddedWidth * daisy->paddedHeight;
  size_t gradGroupSize = 64;

  // gradient X,Y,all - B.0 to A.0-7
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 1, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 2, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_gAll, 1, NULL, 
                                 &gradWorkerSize, &gradGroupSize, 0, 
                                 NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (3)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){

    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), testArray,
                        0, NULL, NULL);
    clFinish(daisyCl->queue);
    printf("\nBefore Gradient: %f",inputArray[512*paddedWidth+128]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[512*paddedWidth+128+k]);
    printf("\nAfter Gradient (X,X+22.5,X+45,Y): %f",testArray[512*paddedWidth+128]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[512*paddedWidth+128+k]);
    printf("\n");
  }
    
  // Smooth all to 2.5 - keep at massBuffer section A
  
  // convolve X - massBuffer sections: A to B
  size_t convWorkerSize11x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight * daisy->gradientsNo};
  size_t convGroupSize11x[2] = {16,4};

  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11x, 2, NULL, 
                                 convWorkerSize11x, convGroupSize11x, 0, 
                                 NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (4)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){
    // checek verified
    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * 11 * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), inputArray,
                        0, NULL, NULL);
    clFinish(daisyCl->queue);
    printf("\nBefore Smooth (11x): %f",testArray[(daisy->height-5)*paddedWidth+daisy->width]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[(daisy->height-5)*paddedWidth+daisy->width+k]);
    printf("\n");
    
    printf("\nAfter Smooth (11x): %f",inputArray[(daisy->height-5)*paddedWidth+daisy->width]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[(daisy->height-5)*paddedWidth+daisy->width+k]);
    printf("\n");
  }

  // convolve Y - massBuffer sections: B to A
  size_t convWorkerSize11y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 8};
  size_t convGroupSize11y[2] = {16,8};

  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11y, 2, NULL, 
                                 convWorkerSize11y, convGroupSize11y, 0, 
                                 NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (5)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){

    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * 7 * sizeof(float), paddedWidth * paddedHeight * sizeof(float), testArray,
                        0, NULL, NULL);
    clFinish(daisyCl->queue);
    printf("\nBefore Smooth (11y): %f",inputArray[200]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[k*paddedWidth+200]);
    printf("\n");
    printf("\nAfter Smooth (11y): %f",testArray[200]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[k*paddedWidth+200]);
    printf("\n");
  }

  // smooth all with size 23 - keep

  // convolve X - massBuffer sections: A to C
  size_t convWorkerSize23x[2] = {daisy->paddedWidth / 4, daisy->paddedHeight * daisy->gradientsNo};
  size_t convGroupSize23x[2] = {16,4};

  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f23x, 2, NULL, 
                                 convWorkerSize23x, convGroupSize23x, 0, 
                                 NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (6)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){

    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * (8 * 2 + 7) * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), inputArray,
                        0, NULL, NULL);
    clFinish(daisyCl->queue);
    printf("\nBefore Smooth (23x): %f",testArray[paddedWidth*daisy->height+50]);
    for(k = 1; k < 25; k++){
      int i = paddedWidth*daisy->height+50+k;
      printf(", %f", testArray[i]);
    }
    printf("\n");
    
    printf("\nAfter Smooth (23x): %f",inputArray[paddedWidth*daisy->height+50]);
    for(k = 1; k < 25; k++){
      int i = paddedWidth*daisy->height+50+k;
      printf(", %f", inputArray[i]);
    }
    printf("\n");
  }

  // convolve Y - massBuffer sections: C to B
  size_t convWorkerSize23y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 4};
  size_t convGroupSize23y[2]  = {16, 16};

  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f23y, 2, 
                                 NULL, convWorkerSize23y, convGroupSize23y,
                                 0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (7)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){  

    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                                paddedWidth * paddedHeight * (8 + 7) * sizeof(float), 
                                paddedWidth * paddedHeight * sizeof(float), testArray,
                                0, NULL, NULL);
    clFinish(daisyCl->queue);

    printf("\nBefore Smooth (23y): %f",inputArray[(paddedHeight-26)*paddedWidth+25]);
    for(k = 1; k < 25; k++)
      printf(", %f", inputArray[(k+paddedHeight-26)*paddedWidth+25]);
    printf("\n");
    printf("\nAfter Smooth (23y): %f",testArray[(paddedHeight-26)*paddedWidth+25]);
    for(k = 1; k < 25; k++)
      printf(", %f", testArray[(k+paddedHeight-26)*paddedWidth+25]);
    printf("\n");
  }

  // smooth all with size 29 - keep
  
  // convolve X - massBuffer sections: B to D
  size_t convWorkerSize29x[2] = {daisy->paddedWidth / 4, (daisy->paddedHeight * daisy->gradientsNo)};
  size_t convGroupSize29x[2]  = {16, 4};

  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f29x, 2, 
                                 NULL, convWorkerSize29x, convGroupSize29x, 
                                 0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (8)",error);

  clFinish(daisyCl->queue);

  if(DEBUG_ALL){

    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * (8 * 3 + 7) * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), inputArray,
                        0, NULL, NULL);

    if(error){
      fprintf(stderr, "oclDaisy.cpp::oclDaisy clEnqueueReadBuffer (29x) failed: %d\n",error);
      return 1;
    }
    clFinish(daisyCl->queue);

    printf("\nBefore Smooth (29x): %f",testArray[paddedWidth*daisy->height+50]);
    for(k = 1; k < 35; k++)
      printf(", %f", testArray[paddedWidth*daisy->height+50+k]);
    printf("\n");
    
    printf("\nAfter Smooth (29x): %f",inputArray[paddedWidth*daisy->height+50]);
    for(k = 1; k < 35; k++)
      printf(", %f", inputArray[paddedWidth*daisy->height+50+k]);
    printf("\n");
  }
  
  // convolve Y - massBuffer sections: D to C
  size_t convWorkerSize29y[2] = {daisy->paddedWidth, (daisy->paddedHeight * daisy->gradientsNo) / 4};
  size_t convGroupSize29y[2]  = {16, 16};

  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f29y, 2, 
                                 NULL, convWorkerSize29y, convGroupSize29y, 
                                 0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (9)",error);

  clFinish(daisyCl->queue);

  gettimeofday(&times->endConvGrad,NULL);

  if(DEBUG_ALL){
    error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                        paddedWidth * paddedHeight * (8 * 2 + 7) * sizeof(float), 
                        paddedWidth * paddedHeight * sizeof(float), testArray,
                        0, NULL, NULL);

    if(error){
      fprintf(stderr, "oclDaisy.cpp::oclDaisy clEnqueueReadBuffer (29y) failed: %d\n",error);
      return 1;
    }
    clFinish(daisyCl->queue);

    printf("\nBefore Smooth (29y): %f",inputArray[129+(paddedHeight-36)*paddedWidth]);
    for(k = 1; k < 35; k++)
      printf(", %f", inputArray[129+((paddedHeight-36)+k)*paddedWidth]);
    printf("\n");
    printf("\nAfter Smooth (29y): %f",testArray[129+(paddedHeight-36)*paddedWidth]);
    for(k = 1; k < 35; k++)
      printf(", %f", testArray[129+((paddedHeight-36)+k)*paddedWidth]);
    printf("\n");
  }

  // A) transpose SxGxHxW to SxHxWxG first

  gettimeofday(&times->startTransGrad,NULL);

  memorySize = daisy->gradientsNo * daisy->smoothingsNo * 
               paddedWidth * paddedHeight * sizeof(cl_float);

  cl_mem transBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                      memorySize, (void*)NULL, &error);

  //oclError("oclDaisy","clCreateBuffer (3)",error);

  //printf("\ntransBuffer size = %ld (%ldMB)\n",memorySize,memorySize/(1024*1024));

  
  size_t transWorkerSize[2] = {daisy->paddedWidth,daisy->paddedHeight * daisy->smoothingsNo * daisy->gradientsNo};
  size_t transGroupSize[2] = {32,8};

  clSetKernelArg(daisy->oclPrograms.kernel_trans, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 1, sizeof(transBuffer), (void*)&transBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_trans, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_trans, 2, 
                                 NULL, transWorkerSize, transGroupSize,
                                 0, NULL, NULL);

  //oclError("oclDaisy","clEnqueueNDRangeKernel (10)",error);

  clFinish(daisyCl->queue);

  clReleaseMemObject(massBuffer);

  gettimeofday(&times->endTransGrad,NULL);

  if(DEBUG_ALL){
    // Verify transposition A)
    error = clEnqueueReadBuffer(daisyCl->queue, transBuffer, CL_TRUE,
                                paddedWidth * paddedHeight * 8 * 1 * sizeof(float), 
                                paddedWidth * paddedHeight * 8 * sizeof(float), testArray,
                                0, NULL, NULL);

    clFinish(daisyCl->queue);

    int dstWidth = daisy->paddedWidth * daisy->gradientsNo;
    printf("\nTranspose:\n");
    int r;
    for(r = 0; r < 2; r++){
      printf("Row %d: %f",r,testArray[r * dstWidth]);
      for(k = 1; k < 25; k++)
        printf(", %f", testArray[r * dstWidth + k]);
      printf("\n\n");
    }
  }

  // B) final transposition

  gettimeofday(&times->startTransDaisy,NULL);

  //printf("\nAllocated %ld bytes on GPU for daisy section buffer (%ldMB)\n",daisySectionSize,daisySectionSize/(1024*1024));

  //printf("\nAllocated %ld bytes on host for daisy descriptor array (%ldMB)\n",daisyDescriptorSize,daisyDescriptorSize/(1024*1024));
  
  cl_mem daisyBufferA, daisyBufferB;

  daisyBufferA = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY,
                                daisySectionSize,(void*)NULL, &error);

  //oclError("oclDaisy","clCreateBuffer (daisybufferA)",error);

  if(totalSections > 1)
    daisyBufferB = clCreateBuffer(daisyCl->context, CL_MEM_WRITE_ONLY,
                                  daisySectionSize,(void*)NULL, &error);

  //oclError("oclDaisy","clCreateBuffer (daisybufferB)",error);

  size_t daisyGroupSize[2] = {windowWidth,windowHeight};

  cl_event * memoryEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections);
  cl_event * kernelEvents = (cl_event*)malloc(sizeof(cl_event) * totalSections * daisy->smoothingsNo);

  clFinish(daisyCl->queue);

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

      int srcGlobalOffset = daisy->paddedHeight * daisy->paddedWidth * daisy->gradientsNo * smoothingNo;

      clSetKernelArg(daisy->oclPrograms.kernel_transd, 0, sizeof(transBuffer), (void*)&transBuffer);
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 1, sizeof(daisyBuffer), (void*)&daisyBuffer);
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 2, sizeof(allPairOffsetBuffers[smoothingNo]), (void*)&allPairOffsetBuffers[smoothingNo]);
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 3, sizeof(float) * (windowHeight * (windowWidth * daisy->gradientsNo + lclArrayPaddings[smoothingNo])), 0);
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 4, sizeof(int), (void*)&(daisy->paddedWidth));
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 5, sizeof(int), (void*)&(daisy->paddedHeight));
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 6, sizeof(int), (void*)&(srcGlobalOffset));
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 7, sizeof(int), (void*)&(pairOffsetsLength));
      clSetKernelArg(daisy->oclPrograms.kernel_transd, 8, sizeof(int), (void*)&(lclArrayPaddings[smoothingNo]));

      error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_transd, 2,
                                     daisyWorkerOffsets, daisyWorkerSize, daisyGroupSize,
                                     (resourceContext!=sectionNo),
                                     prevMemoryEvents,
                                     &currKernelEvents[smoothingNo]);

      //oclError("oclDaisy","clEnqueueNDRangeKernel (daisy block)",error);

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

    error = clEnqueueReadBuffer(daisyCl->queue, daisyBuffer, 0,
                                0, daisySectionSize, daisyDescriptorsSection,
                                3, currKernelEvents, &currMemoryEvents[0]);

    //oclError("oclDaisy","clEnqueueReadBuffer (async)",error);

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

  error = clFinish(daisyCl->queue);
  //oclError("oclDaisy","clFinish (daisy)",error);

  gettimeofday(&times->endTransDaisy,NULL);

  gettimeofday(&times->endFull,NULL);

  times->startt = times->startFull.tv_sec+(times->startFull.tv_usec/1000000.0);
  times->endt = times->endFull.tv_sec+(times->endFull.tv_usec/1000000.0);
  times->difft = times->endt-times->startt;
  printf("\nDaisyFull: %.4fs (%.4f MPixel/sec)\n",times->difft,(daisy->paddedWidth*daisy->paddedHeight) / (1000000.0f*times->difft));

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

        int petalsNo = daisy->petalsNo + (smoothingNo==0);

        int petalStart = (smoothingNo > 0 ? smoothingNo * daisy->petalsNo + 1 : 0);

        //printf("\nPetalsNo = %d\n",petalsNo);
        float * daisyArray = daisy->descriptors + block * daisyBlockHeight * daisyBlockWidth * 200;

        clFinish(daisyCl->queue);
        /*for(int r = 0; r < 16; r+=16){
          for(k = petalStart * 8; k < (petalStart+2) * 8; k++)
            printf(", %f",daisyArray[r * daisyBlockWidth * 200 + k]);
          printf("\n\n");
        }*/

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

  // Release and unmap buffers, free allocated space
#ifdef DAISY_HOST_TRANSFER

  // don't unmap the pinned memory if there was only one block
  //if(totalSections > 1){

  error = clEnqueueUnmapMemObject(daisyCl->queue, hostPinnedDaisyDescriptors, daisyDescriptorsSection, 0, NULL, NULL);	
    //oclError("oclDaisy","clEnqueueUnmapMemObject (hostPinnedSection)",error);
  //free(daisyDescriptorsSection);
  //}

  clReleaseMemObject(hostPinnedDaisyDescriptors);

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

  if(DEBUG_ALL) free(testArray);

  free(filter5);
  free(memoryEvents);
  free(kernelEvents);

  return error | issues;
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
    petalOffsets[i*2]   = regionRadius * sin(i * (M_PI / 4));
    petalOffsets[i*2+1] = regionRadius * cos(i * (M_PI / 4));
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


