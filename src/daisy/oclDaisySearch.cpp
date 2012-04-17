#include "oclDaisySearch.h"
#include <stdio.h>
#include <sys/time.h>

#define WORKGROUP_LENGTH 16
#define SEARCH_RADIUS 16
//#define DEBUG_ALL

long int verifyLayerSearch(float* transRef, float* transTar, float* dispDiff, int width, int height, int searchRadius, int* descBuild, int descBuildNo);

int oclErrorS(const char * function, const char * functionCall, int error){
  if(error){
    fprintf(stderr, "oclDaisySearch.cpp::%s %s failed: %d\n",function,functionCall,error);
    return error;
  }
  return 0;
}

int oclDaisySearch(ocl_constructs * ocl, daisy_params * daisyRef, daisy_params * daisyTar){

  int error = 0;

  int layers = daisyRef->smoothingsNo;

  int ** layerSizes = (int**)malloc(sizeof(int*) * layers);

  long int disparityPyramidSize = 0;

  for(int layer = 0; layer < layers; layer++){
    
    pyramid_layer_set * layerSettings = daisyRef->pyramidLayerSettings[layer];

    int downsample = pow(sqrt(DOWNSAMPLE_RATE),layerSettings->t_downsample);

    layerSizes[layer] = (int*)malloc(sizeof(int) * 2);
    layerSizes[layer][0] = daisyRef->paddedHeight / downsample;
    layerSizes[layer][1] = daisyRef->paddedWidth / downsample;

    int searchRadius = SEARCH_RADIUS / downsample;
    int gridSize = (searchRadius * 2 + 1) * (searchRadius * 2 + 1);

    disparityPyramidSize += layerSizes[layer][0] * layerSizes[layer][1] * gridSize;

  }

  cl_mem disparityBuffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                          disparityPyramidSize * sizeof(float),
                                          (void*)NULL, &error);

  oclErrorS("oclDaisySearch","clCreateBuffer (disparityBuffer)",error);

  printf("disparityBufferSize = %ld bytes (%ldMB)\n",disparityPyramidSize*4,(disparityPyramidSize*4)/(1024*1024));
  double totaltime = .0f;
  //
  int layerOffset = 0;
  int disparityLayerOffset = 0;
  struct timeval startConvGrad,endConvGrad;
  for(int layer = 0; layer < 3; layer++){
    pyramid_layer_set * layerSettings = daisyRef->pyramidLayerSettings[layer];
    int downsample = pow(sqrt(DOWNSAMPLE_RATE),layerSettings->t_downsample);

    int petalsNo = REGION_PETALS_NO + (layer == 0);

    float * daisyRegionOffsetsF = generatePetalOffsets(daisyRef->sigmas[layer],petalsNo,layer==0);
    int * daisyRegionOffsets = (int*)malloc(sizeof(int) * 2 * petalsNo);
    
    for(int i = 0; i < petalsNo; i++){
      daisyRegionOffsetsF[i*2] = round(daisyRegionOffsetsF[i*2] / downsample);
      daisyRegionOffsetsF[i*2+1] = round(daisyRegionOffsetsF[i*2+1] / downsample);
      daisyRegionOffsets[i*2] = (int)daisyRegionOffsetsF[i*2];
      daisyRegionOffsets[i*2+1] = (int)daisyRegionOffsetsF[i*2+1];
    }

    cl_mem buildBuffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        petalsNo * 2 * sizeof(float), (void*)daisyRegionOffsetsF, &error);

    int searchRadius = SEARCH_RADIUS / downsample;

    int searchGridSize = (2*searchRadius+1)*(2*searchRadius+1);

    int workgroupWidth = searchRadius;

    const size_t workerSize[2] = {layerSizes[layer][1],layerSizes[layer][0]}; // need to unwrap to 1D so to ensure 16-multiple in X for coalescence
    const size_t groupSize[2] = {workgroupWidth, workgroupWidth};

    const int BUILD_HALO = (int)(daisyRef->sigmas[layer] * 2); // is equal to max offset of region

    const int buildHalo = ceil(BUILD_HALO / (float)downsample);

    const int importBlockWidth = searchRadius + 2 * buildHalo;

    printf("Layer %d: sigma=%f, downsample=%d, searchRadius=%d, searchGridSize=%d, importBlockWidth=%d, buildHalo=%d\n",layer,daisyRef->sigmas[layer],downsample,searchRadius,searchGridSize,importBlockWidth,buildHalo);

/*
__kernel void searchDaisy(__global    float * refArray,
                          __global    float * tarArray,
                          __global    float * dspArray,
                          __local     float * lclRefArray,
                          __local     float * lclTarArray,
                          __constant  int   * buildArray,
                          const       int     pyramidOffset,
                          const       int     disparityOffset,
                          const       int     buildHalo)*/
    gettimeofday(&startConvGrad,NULL);
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 0, sizeof(daisyRef->oclBuffers->transBuffer), (void*)&(daisyRef->oclBuffers->transBuffer));
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 1, sizeof(daisyTar->oclBuffers->transBuffer), (void*)&(daisyTar->oclBuffers->transBuffer));
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 2, sizeof(disparityBuffer), (void*)&disparityBuffer);
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 3, sizeof(buildBuffer), (void*)&buildBuffer);
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 4, sizeof(int), (void*)&layerOffset);
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 5, sizeof(int), (void*)&disparityLayerOffset);
    clSetKernelArg(daisyRef->oclPrograms->kernel_search, 6, sizeof(int), (void*)&buildHalo);

    error = clEnqueueNDRangeKernel(ocl->queue, daisyRef->oclPrograms->kernel_search, 2, NULL, 
                                   workerSize, groupSize, 0, 
                                   NULL, NULL);
    oclErrorS("oclDaisySearch", "clEnqueueNDRangeKernel (search)", error);
  
    error = clFinish(ocl->queue);
    oclErrorS("oclDaisySearch", "clFinish (search)", error);

    gettimeofday(&endConvGrad,NULL);
    double startt = startConvGrad.tv_sec+(startConvGrad.tv_usec/1000000.0);
    double endt = endConvGrad.tv_sec+(endConvGrad.tv_usec/1000000.0);
    double difft = endt-startt;
    printf("\nlayersearch: %.4fs (%.4f MPixel/sec)\n",difft,(daisyRef->paddedWidth*daisyRef->paddedHeight*8*3) / (1000000.0f*difft));
    totaltime += difft;
#ifdef DEBUG_ALL

    float * transRefLayer = (float*)malloc(layerSizes[layer][0] * layerSizes[layer][1] * GRADIENTS_NO * sizeof(float));
    float * transTarLayer = (float*)malloc(layerSizes[layer][0] * layerSizes[layer][1] * GRADIENTS_NO * sizeof(float));
    float * dispDiffLayer = (float*)malloc(layerSizes[layer][0] * layerSizes[layer][1] * searchGridSize * sizeof(float));
    error = clEnqueueReadBuffer(ocl->queue, daisyRef->oclBuffers->transBuffer, CL_TRUE,
                                layerOffset * sizeof(float), 
                                layerSizes[layer][0] * layerSizes[layer][1] * GRADIENTS_NO * sizeof(float), transRefLayer,
                                0, NULL, NULL);

    error = clEnqueueReadBuffer(ocl->queue, daisyTar->oclBuffers->transBuffer, CL_TRUE,
                                layerOffset * sizeof(float), 
                                layerSizes[layer][0] * layerSizes[layer][1] * GRADIENTS_NO * sizeof(float), transTarLayer,
                                0, NULL, NULL);
    error = clEnqueueReadBuffer(ocl->queue, disparityBuffer, CL_TRUE,
                                disparityLayerOffset * sizeof(float), 
                                layerSizes[layer][0] * layerSizes[layer][1] * searchGridSize * sizeof(float), dispDiffLayer,
                                0, NULL, NULL);

    long int issues = verifyLayerSearch(transRefLayer,transTarLayer,dispDiffLayer,layerSizes[layer][1],layerSizes[layer][0],searchRadius,daisyRegionOffsets,REGION_PETALS_NO);
    printf("search issues: %ld\n",issues);
    free(transRefLayer);
    free(transTarLayer);
    free(dispDiffLayer);
#endif

  layerOffset += layerSizes[layer][0] * layerSizes[layer][1] * GRADIENTS_NO;
  disparityLayerOffset += layerSizes[layer][0] * layerSizes[layer][1] * searchGridSize;

  }

  printf("\ntotalsearch: %.4fs (%.4f MPixel/sec)\n",totaltime,(daisyRef->paddedWidth*daisyRef->paddedHeight*8*3*3) / (1000000.0f*totaltime));

  //

  return error;
}

long int verifyLayerSearch(float * transRef, float * transTar, float * dispDiff, int width, int height, int searchRadius, int * descBuild, int descBuildNo){

  long int issues = 0;
  long int limit = 100;

  int gridSize = (searchRadius*2+1)*(searchRadius*2+1);

  // per reference pixel
  for(int y = searchRadius*2; y < height-searchRadius*2; y++){
    for(int x = searchRadius*2; x < width-searchRadius*2; x++){
      
      // per target pixel around it
      for(int gy = -searchRadius; gy < searchRadius; gy++){
        for(int gx = -searchRadius; gx < searchRadius; gx++){

          // per descriptor element
          float thisDispDiffGpu = dispDiff[y * width * gridSize + x * gridSize + (gy+searchRadius) * (2*searchRadius+1) + (gx+searchRadius)];
          float thisDispDiffCpu = 0;
          for(int e = 0; e < descBuildNo; e++){

            float * thisTransRef = transRef + (y+descBuild[e*2]) * width * GRADIENTS_NO + (x+descBuild[e*2+1]) * GRADIENTS_NO;
            float * thisTransTar = transTar + (y+gy+descBuild[e*2]) * width * GRADIENTS_NO + (x+gx+descBuild[e*2+1]) * GRADIENTS_NO;

            for(int g = 0; g < GRADIENTS_NO; g++){
              thisDispDiffCpu += fabs(thisTransRef[g] - thisTransTar[g]);
            }

          }

          if(fabs(thisDispDiffGpu - thisDispDiffCpu) > 0.0001f && issues++ < limit){
            printf("search issue at (y,x) = (%d,%d), (sy,sx) = (%d,%d) should be %f but is %f\n",y,x,gy,gx,thisDispDiffCpu,thisDispDiffGpu);
          }

        }

      }


    }

  }

  return issues;

}
