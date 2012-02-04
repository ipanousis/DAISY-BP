#include "oclDaisy.h"

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
  
  return error;

}

int oclDaisy(daisy_params * daisy, ocl_constructs * daisyCl){

  cl_int error;

  int largestKernelSize = 11;
  int largestKernelHalo = largestKernelSize / 2;

  int paddedWidth  = daisy->paddedWidth;
  int paddedHeight = daisy->paddedHeight;

  float * inputArray = (float*)malloc(sizeof(float) * paddedWidth * paddedHeight);

  // Pad edges of input array for i) to fit the workgroup size ii) convolution halo - resample nearest pixel
  int i;
  for(i = 0; i < daisy->height; i++){
    int j;
    for(j = 0; j < daisy->width; j++)
      inputArray[i * paddedWidth + j] = daisy->array[i * daisy->width + j];
  }
  free(daisy->array);

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)
  int filter7Size = 7;
  float filter7[7] = {0.036633,0.11128,0.21675,
                      0.27068,
                      0.21675,0.11128,0.036633};
  int filter11Size = 11;
  float filter11[11] = {0.007024633,0.02381049,0.06153227,0.1212349,0.1821137,
                        0.2085680,
                        0.1821137,0.1212349,0.06153227,0.02381049,0.007024633};

  cl_mem convBufferA = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      paddedWidth * paddedHeight * sizeof(cl_float),
                                      (void*)inputArray, &error);
                                          
  cl_mem convBufferB = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                      paddedWidth * paddedHeight * sizeof(cl_float),
                                      (void*)NULL, &error);

  cl_mem filterBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       largestKernelSize * sizeof(cl_float),
                                       (void*)filter7, &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateBuffer failed: %d\n",error);
    return 1;
  }

  // Prepare the kernel
  
  // convolve X - A to B
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 0, sizeof(convBufferA), (void*)&convBufferA);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 1, sizeof(convBufferB), (void*)&convBufferB);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 2, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 3, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 4, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 5, sizeof(int), (void*)&(daisy->paddedWidth));


  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  // convolve Y - B to A
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 0, sizeof(convBufferB), (void*)&convBufferB);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 1, sizeof(convBufferA), (void*)&convBufferA);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 2, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 3, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 4, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 5, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 6, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  // Write filter-11 data
  error = clEnqueueWriteBuffer(daisyCl->queue, filterBuffer, CL_FALSE, 
                               0, filter11Size * sizeof(float), (void*)filter11,
                               0, NULL, NULL);
                              

  printf("Convolution to 1.6 sent!\n");


  cl_uint k;
  float * testArray = (float*)malloc(sizeof(float) * paddedWidth * paddedHeight * 3);

  error = clEnqueueReadBuffer(daisyCl->queue, convBufferA, CL_TRUE,
                      0, paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  
  error = clEnqueueReadBuffer(daisyCl->queue, convBufferB, CL_TRUE,
                      0, paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  

  printf("\nDenoising Input: %f",testArray[daisy->width-20]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[daisy->width-20+k]);
  printf("\n");
  printf("\nDenoising Output: %f",inputArray[daisy->width-20]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[daisy->width-20+k]);
  printf("\n");

  // gradients for 8 orientations
  cl_mem massBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                     daisy->orientationsNo * daisy->smoothingsNo * 
                                     paddedWidth * paddedHeight * sizeof(cl_float),
                                     (void*)NULL, &error);

  printf("massBuffer size = %d (%dMB)\n",daisy->orientationsNo * daisy->smoothingsNo * 
                                     paddedWidth * paddedHeight * sizeof(cl_float), (daisy->orientationsNo * daisy->smoothingsNo * 
                                     paddedWidth * paddedHeight * sizeof(cl_float)) / (1024 * 1024));
  printf("largestKernelSize = %d, paddedWidth = %d, paddedHeight = %d\n",largestKernelSize, paddedWidth, paddedHeight);

  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 0, sizeof(convBufferA), (void*)&convBufferA);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 1, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 2, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 3, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 4, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 5, sizeof(int), (void*)&(daisy->paddedHeight));

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
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 2, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 3, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 4, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11x, 5, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      paddedWidth * paddedHeight * 8 * sizeof(float), 
                      paddedWidth * paddedHeight * sizeof(float), inputArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (11x): %f",testArray[paddedWidth*(daisy->height-3)]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[paddedWidth*(daisy->height-3)+k]);
  printf("\n");
  
  printf("\nAfter Smooth (11x): %f",inputArray[paddedWidth*(daisy->height-3)]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[paddedWidth*(daisy->height-3)+k]);
  printf("\n");
  // convolve Y - massBuffer sections: B to A

  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 0, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 1, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 2, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 3, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 4, sizeof(int), (void*)&(daisy->paddedWidth));
  clSetKernelArg(daisy->oclPrograms.kernel_f11y, 5, sizeof(int), (void*)&(daisy->paddedHeight));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f11y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      0, paddedWidth * paddedHeight * sizeof(float), testArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);
  printf("\nBefore Smooth (11y): %f",inputArray[(daisy->height-20)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[(daisy->height-20+k)*paddedWidth]);
  printf("\n");
  printf("\nAfter Smooth (11y): %f",testArray[(daisy->height-20)*paddedWidth]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[(daisy->height-20+k)*paddedWidth]);
  printf("\n");

  printf("Convolution to 2.5 sent!\n");

  // smooth all with size 23 - keep
  // smooth all with size 29 - keep
  // transpose

  return error;
}
