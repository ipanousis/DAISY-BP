#include "oclDaisy.h"
#include <sys/time.h>

daisy_params * newDaisyParams(unsigned char* array, int height, int width,
                              int orientationsNo, int smoothingsNo){

  daisy_params * params = (daisy_params*) malloc(sizeof(daisy_params));
  params->array = array;
  params->height = height;
  params->width = width;
  params->orientationsNo = orientationsNo;
  params->smoothingsNo = smoothingsNo;

  return params;
}

int initOcl(daisy_params * daisy, ocl_constructs * daisyCl){

  cl_int error;

  daisyCl->groupSize = 64; // need multiple of warp size (32 here)

  int extraHeight = (daisyCl->groupSize - (daisy->height % daisyCl->groupSize)) % daisyCl->groupSize;
  int extraWidth  = (daisyCl->groupSize - (daisy->width  % daisyCl->groupSize)) % daisyCl->groupSize;

  daisy->paddedWidth  = daisy->width  + extraWidth;
  daisy->paddedHeight = daisy->height + extraHeight;

  size_t groups = (daisy->paddedWidth * daisy->paddedHeight -1) / daisyCl->groupSize + 1;

  daisyCl->workerSize = groups * daisyCl->groupSize;

  printf("WorkerSize = %d, GroupSize = %d\n",daisyCl->workerSize,daisyCl->groupSize);

  // Prepare/Reuse platform, device, context, command queue
  cl_bool recreateBuffers = 0;

  error = buildCachedConstructs(daisyCl, &recreateBuffers);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedConstructs returned %d, cannot continue\n",error);
    return 1;
  }

  // Pass preprocessor build options
  char options[128];
  options[0] = '\0';
  //sprintf(options, "-DFILTER_SIZE=%d -DARRAY_HEIGHT=%d -DARRAY_WIDTH=%d", 7, daisy->height, daisy->width);
  //const char * opts = options;

  // Build denoising filter
  error = buildCachedProgram(daisyCl, "daisyKernels.cl", options);

  if(daisyCl->program == NULL){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
    return 1;
  }

  // Prepare the kernel
  daisy->oclPrograms.kernel_f7x = clCreateKernel(daisyCl->program, "convolve_7x", &error);
  daisy->oclPrograms.kernel_f7y = clCreateKernel(daisyCl->program, "convolve_7y", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  // Build gradient kernel
  
  daisy->oclPrograms.kernel_gAll = clCreateKernel(daisyCl->program, "gradient_8all", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->oclPrograms.kernel_f11x = clCreateKernel(daisyCl->program, "convolve_11x", &error);
  daisy->oclPrograms.kernel_f11y = clCreateKernel(daisyCl->program, "convolve_11y", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->oclPrograms.kernel_f23x = clCreateKernel(daisyCl->program, "convolve_23x", &error);
  daisy->oclPrograms.kernel_f23y = clCreateKernel(daisyCl->program, "convolve_23y", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  daisy->oclPrograms.kernel_f29x = clCreateKernel(daisyCl->program, "convolve_29x", &error);
  daisy->oclPrograms.kernel_f29y = clCreateKernel(daisyCl->program, "convolve_29y", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }
  
  return error;

}

int oclDaisy(daisy_params * daisy, ocl_constructs * daisyCl){

  // Time structures - measure down to microseconds
  struct timeval startParaTime;
  struct timeval endParaTime;
  double startt, endt, diffp;

  cl_int error;

  int largestKernelSize = 29;
  int largestKernelHalo = largestKernelSize / 2;

  int paddedWidth  = daisy->paddedWidth;
  int paddedHeight = daisy->paddedHeight;

  gettimeofday(&startParaTime,NULL);

  long int memorySize = daisy->orientationsNo * (daisy->smoothingsNo+1) * 
                        paddedWidth * paddedHeight * sizeof(cl_float);

  cl_mem massBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                     memorySize, (void*)NULL, &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateBuffer failed (0): %d\n",error);
    return 1;
  }

  printf("massBuffer size = %d (%dMB)\n", memorySize, memorySize / (1024 * 1024));
  printf("largestKernelSize = %d, paddedWidth = %d, paddedHeight = %d\n",largestKernelSize, paddedWidth, paddedHeight);

  int filter7Size = 7;
  float filter7[7] = {0.036633,0.11128,0.21675,
                      0.27068,
                      0.21675,0.11128,0.036633};
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
                       filterOffsets[0] * sizeof(float), filter7Size * sizeof(float), (void*)filter7,
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

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clEnqueueWriteBuffer failed (0): %d\n",error);
    return 1;
  }

  float * inputArray = (float*)malloc(sizeof(float) * paddedWidth * paddedHeight);

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

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clEnqueueWriteBuffer failed (1): %d\n",error);
    return 1;
  }

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)

  // convolve X - A.0 to A.1
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  // convolve Y - A.1 to B.0
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  printf("Convolution to 1.6 sent!\n");


  cl_uint k;
  float * testArray = (float*)malloc(sizeof(float) * paddedWidth * paddedHeight * 3);
  
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * sizeof(float), paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);

  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * sizeof(float), paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  

  printf("\nDenoising Input: %f",testArray[(daisy->height-25)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[(daisy->height-25+k)*paddedWidth]);
  printf("\n");
  printf("\nDenoising Output: %f",inputArray[(daisy->height-25)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[(daisy->height-25+k)*paddedWidth]);
  printf("\n");

  // gradients for 8 orientations

  // gradient X,Y,all - B.0 to A.0-7
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 1, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 2, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_gAll, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      0, paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Gradient: %f",inputArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[k]);
  printf("\nAfter Gradient (X): %f",testArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[k]);
  printf("\n");
  
  // Smooth all to 2.5 - keep at massBuffer section A
  
  // convolve X - massBuffer sections: A to B
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (11x): %f",testArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[k]);
  printf("\n");
  
  printf("\nAfter Smooth (11x): %f",inputArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[k]);
  printf("\n");
  // convolve Y - massBuffer sections: B to A

  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      0, paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (11y): %f",inputArray[(daisy->height-25)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[(daisy->height-25+k)*paddedWidth]);
  printf("\n");
  printf("\nAfter Smooth (11y): %f",testArray[(daisy->height-25)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[(daisy->height-25+k)*paddedWidth]);
  printf("\n");

  printf("Convolution to 2.5 sent!\n");

  // smooth all with size 23 - keep

  // Write filter-23 data

  // convolve X - massBuffer sections: A to C
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f23x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f23x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * 2 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (23x): %f",testArray[daisy->width-25]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[daisy->width-25+k]);
  printf("\n");
  
  printf("\nAfter Smooth (23x): %f",inputArray[daisy->width-25]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[daisy->width-25+k]);
  printf("\n");

  // convolve Y - massBuffer sections: C to B
  
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f23y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f23y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);

  printf("\nBefore Smooth (23y): %f",inputArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[k*paddedWidth]);
  printf("\n");
  printf("\nAfter Smooth (23y): %f",testArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[k*paddedWidth]);
  printf("\n");

  printf("Convolution to 5 sent!\n");

  // smooth all with size 29 - keep


  // Write filter-29 data
  
  // convolve X - massBuffer sections: B to D
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f29x, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f29x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * 3 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (29x): %f",testArray[daisy->width-25]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[daisy->width-25+k]);
  printf("\n");
  
  printf("\nAfter Smooth (29x): %f",inputArray[daisy->width-25]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[daisy->width-25+k]);
  printf("\n");

  // convolve Y - massBuffer sections: D to C
  
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 2, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f29y, 3, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f29y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * 2 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);

  printf("\nBefore Smooth (29y): %f",inputArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[k*paddedWidth]);
  printf("\n");
  printf("\nAfter Smooth (29y): %f",testArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[k*paddedWidth]);
  printf("\n");

  printf("Convolution to 7.5 sent!\n");


  // transpose

  // Buffers to release;
  // massBuffer
  // filterBuffer
  clReleaseMemObject(massBuffer);
  clReleaseMemObject(filterBuffer);

  gettimeofday(&endParaTime,NULL);

  startt = startParaTime.tv_sec+(startParaTime.tv_usec/1000000.0);
  endt = endParaTime.tv_sec+(endParaTime.tv_usec/1000000.0);

  diffp = endt-startt;
  printf("\nConvolutions: %.3fs\n",diffp);

  free(inputArray);
  free(testArray);

  return error;
}
