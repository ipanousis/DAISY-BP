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

  size_t groups = (daisy->width * daisy->height -1) / daisyCl->groupSize + 1;

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
  error = buildCachedProgram(daisyCl, "daisyFilter7.cl", options);

  if(daisyCl->program == NULL){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
    return 1;
  }

  daisy->oclPrograms.program_f7 = daisyCl->program;

  // Prepare the kernel
  daisy->oclPrograms.kernel_f7x = clCreateKernel(daisy->oclPrograms.program_f7, "convolve_x", &error);
  daisy->oclPrograms.kernel_f7y = clCreateKernel(daisy->oclPrograms.program_f7, "convolve_y", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  // Build gradient kernel
  daisyCl->program = NULL;

  error = buildCachedProgram(daisyCl, "daisyGradient8.cl", options);

  if(daisyCl->program == NULL){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy buildCachedProgram returned NULL, cannot continue\n");
    return 1;
  }

  daisy->oclPrograms.program_gAll = daisyCl->program;

  daisy->oclPrograms.kernel_gAll = clCreateKernel(daisy->oclPrograms.program_gAll, "gradient8", &error);

  if(error){
    fprintf(stderr, "oclDaisy.cpp::oclDaisy clCreateKernel failed: %d\n",error);
    return 1;
  }

  return error;

}

int oclDaisy(daisy_params * daisy, ocl_constructs * daisyCl){

  cl_int error;

  cl_uint largestKernelSize = 7;
  cl_uint largestKernelHalo = largestKernelSize / 2;

  cl_uint fullHeight = daisy->height + largestKernelSize - 1;
  cl_uint fullWidth  = daisy->width + largestKernelSize - 1;

  float * inputArray = (float*)malloc(sizeof(float) * fullWidth * fullHeight);

  // Pad edges of input array - resample nearest pixel
  cl_uint i;
  for(i = largestKernelHalo; i < largestKernelHalo + daisy->height; i++){
    cl_uint j;
    for(j = 0; j < largestKernelHalo; j++)
      inputArray[i * fullWidth + j] = daisy->array[(i - largestKernelHalo) * daisy->width];
    for(j = largestKernelHalo; j < largestKernelHalo + daisy->width; j++)
      inputArray[i * fullWidth + j] = daisy->array[(i - largestKernelHalo) * daisy->width + j - largestKernelHalo];
    for(j = largestKernelHalo + daisy->width; j < fullWidth; j++)
      inputArray[i * fullWidth + j] = daisy->array[(i - largestKernelHalo) * daisy->width + daisy->width - 1];
  }
  free(daisy->array);

  // smooth with kernel size 7 (achieve sigma 1.6 from 0.5)
  int filter7Size = 7;
  float filter7[7] = {0.036633,0.11128,0.21675,0.27068,0.21675,0.11128,0.036633};

  cl_mem convBufferA = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      fullWidth * fullHeight * sizeof(cl_float),
                                      (void*)inputArray, &error);
                                          
  cl_mem convBufferB = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                      fullWidth * fullHeight * sizeof(cl_float),
                                      (void*)NULL, &error);

  cl_mem filterBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       filter7Size * sizeof(cl_float),
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
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 3, sizeof(int), (void*)&filter7Size);
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 4, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f7x, 5, sizeof(int), (void*)&(daisy->height));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7x, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  // convolve Y - B to A
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 0, sizeof(convBufferB), (void*)&convBufferB);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 1, sizeof(convBufferA), (void*)&convBufferA);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 2, sizeof(filterBuffer), (void*)&filterBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 3, sizeof(int), (void*)&filter7Size);
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 4, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_f7y, 5, sizeof(int), (void*)&(daisy->height));

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_f7y, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);

  printf("Convolution to 1.6 sent!\n");

  /*float * testArray = (float*)malloc(sizeof(float) * fullWidth * fullHeight * 3);
  error = clEnqueueReadBuffer(daisyCl->queue, convBufferB, CL_TRUE,
                      3 * sizeof(float), fullWidth * 25 * sizeof(float), testArray,
                      0, NULL, NULL);*/


  cl_uint k;

  clFinish(daisyCl->queue);
  error = clEnqueueReadBuffer(daisyCl->queue, convBufferA, CL_TRUE,
                      0, 3 + fullWidth * 25 * sizeof(float), inputArray,
                      0, NULL, NULL);
  clFinish(daisyCl->queue);

  printf("\nDenoising Output: %f",inputArray[3]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[3 + k * fullWidth]);
  printf("\n");

  // gradients for 8 orientations
  cl_mem massBuffer = clCreateBuffer(daisyCl->context, CL_MEM_READ_WRITE,
                                     daisy->orientationsNo * daisy->smoothingsNo * 
                                     fullWidth * fullHeight * sizeof(cl_float),
                                     (void*)NULL, &error);
 
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 0, sizeof(convBufferA), (void*)&convBufferA);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 1, sizeof(massBuffer), (void*)&massBuffer);
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 2, sizeof(int), (void*)&(daisy->width));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 3, sizeof(int), (void*)&(daisy->height));
  clSetKernelArg(daisy->oclPrograms.kernel_gAll, 4, sizeof(int), (void*)&largestKernelHalo);

  error = clEnqueueNDRangeKernel(daisyCl->queue, daisy->oclPrograms.kernel_gAll, CL_TRUE, NULL, 
                                 &(daisyCl->workerSize), &(daisyCl->groupSize), 0, 
                                 NULL, NULL);


  float * testArray = (float*)malloc(sizeof(float) * fullWidth * fullHeight * 3);
  error = clEnqueueReadBuffer(daisyCl->queue, massBuffer, CL_TRUE,
                      fullWidth * 3 * sizeof(float), fullWidth * fullHeight * 3 * sizeof(float), testArray,
                      0, NULL, NULL);

  clFinish(daisyCl->queue);
  printf("\nBefore Gradient: %f",inputArray[fullWidth * 3]);
  for(k = 1; k < 25; k++)
    printf(", %f", inputArray[fullWidth * 3 + k]);
  printf("\nAfter Gradient (X): %f",testArray[0]);
  for(k = 1; k < 25; k++)
    printf(", %f", testArray[k]);
  printf("\n");
  
  // smooth all with size 13 - keep
  // smooth all with size 23 - keep
  // smooth all with size 29 - keep
  // transpose

  return error;
}
