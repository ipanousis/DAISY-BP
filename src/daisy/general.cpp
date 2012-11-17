#include "general.h"
#include <stdio.h>

pyramid_layer_set * newPyramidLayerSetting(float sigma, float newTotalSigma, int prevTotalDownsample){

  pyramid_layer_set * set = (pyramid_layer_set*) malloc(sizeof(pyramid_layer_set));
  set->phi = PHI_SIGMA_DOWNSAMPLE;
  set->sigma = sigma;
  set->downsample = (int)floor(set->sigma / set->phi);
  set->t_sigma = newTotalSigma;
  set->t_downsample = prevTotalDownsample + set->downsample;

  return set;
}

daisy_params * newDaisyParams(unsigned char* array, int height, int width,
                              int gradientsNo, int petalsNo, int smoothingsNo, float* sigmas){

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
  params->sigmas = sigmas;
  params->paddedWidth = params->width + (ARRAY_PADDING - params->width % ARRAY_PADDING) % ARRAY_PADDING;
  params->paddedHeight = params->height + (ARRAY_PADDING - params->height % ARRAY_PADDING) % ARRAY_PADDING;

  params->oclBuffers = (ocl_daisy_buffers*)malloc(sizeof(ocl_daisy_buffers));
  params->oclBuffers->transBuffer = NULL;

  generatePyramidSettings(params);

  return params;
}

void generatePyramidSettings(daisy_params * params){

  params->pyramidLayerSettings = (pyramid_layer_set**) malloc(sizeof(pyramid_layer_set*) * params->smoothingsNo);
  params->pyramidLayerSizes = (int*) malloc(sizeof(int) * params->smoothingsNo);
  params->pyramidLayerOffsets = (int*) malloc(sizeof(int) * params->smoothingsNo);

  for(int s = 0; s < params->smoothingsNo; s++){

    float sigma;
    if(s == 0)
      sigma = params->sigmas[0];
    else{
      sigma = sqrt(pow(params->sigmas[s],2) - pow(params->sigmas[s-1],2)) / pow(DOWNSAMPLE_RATE,params->pyramidLayerSettings[s-1]->t_downsample);
    }
  
    int prevTotalDownsample = (s > 0 ? params->pyramidLayerSettings[s-1]->t_downsample : 0);

    params->pyramidLayerSettings[s] = newPyramidLayerSetting(sigma,params->sigmas[s],prevTotalDownsample);

    int totalDownsample = pow(DOWNSAMPLE_RATE * 2,params->pyramidLayerSettings[s]->t_downsample);

    params->pyramidLayerSizes[s] = (params->paddedHeight * params->paddedWidth * params->gradientsNo) / totalDownsample;
    params->pyramidLayerOffsets[s] = (s > 0 ? params->pyramidLayerOffsets[s-1] + params->pyramidLayerSizes[s-1] : 0);

    //printf("\nsigma computed at level %d: %f",s,sigma);
    //printf("\ndownsample setting for level %d: %dx%d (%d)",s,(int)(params->paddedHeight / sqrt(totalDownsample)),(int)(params->paddedWidth / sqrt(totalDownsample)),(int)sqrt(totalDownsample));
  }

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
    petalOffsets[i*2]   = regionRadius * sin(i * (M_PI / 4)); // y
    petalOffsets[i*2+1] = regionRadius * cos(i * (M_PI / 4)); // x
  }

  return petalOffsets;
}

