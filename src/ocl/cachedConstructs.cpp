/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : December/2011

  File: cachedConstructs.cpp

*/
#include "ocl/cachedConstructs.h"
#include <stdio.h>

ocl_constructs * newOclConstructs(cl_uint workerSize, cl_uint groupSize, cl_bool clGlSharing){

  ocl_constructs * occs = (ocl_constructs*) malloc(sizeof(ocl_constructs));
  occs->platformId = NULL;
  occs->deviceId = NULL;
  occs->context = NULL;
  occs->ioqueue = NULL;
  occs->program = NULL;
  occs->buffers = NULL;

  occs->programsCount = 0;
  occs->programs = (cl_program*)malloc(sizeof(cl_program) * 10);

  if(clGlSharing && 0){
    occs->contextProperties = (cl_context_properties*) 
                                malloc(sizeof(cl_context_properties) * 7);
    occs->contextProperties[0] = CL_GL_CONTEXT_KHR;
    occs->contextProperties[1] = (cl_context_properties)glXGetCurrentContext();
    occs->contextProperties[2] = CL_GLX_DISPLAY_KHR;
    occs->contextProperties[3] = (cl_context_properties)glXGetCurrentDisplay();
    occs->contextProperties[4] = CL_CONTEXT_PLATFORM;
    //occs->contextProperties[5] = (cl_context_properties)cpPlatform;
    occs->contextProperties[6] = 0;
  }
  else occs->contextProperties = NULL;
  //occs->refreshCount = 30;

  return occs;
}

int buildCachedConstructs(ocl_constructs * occs, cl_bool * rebuildMemoryObjects){

  cl_int error = 0;
  *rebuildMemoryObjects = 0;

  if(occs == NULL)
    return 1;

  if(occs->platformId == NULL){
    error = clGetPlatformIDs(1, &(occs->platformId), NULL);

    if(error){
      fprintf(stderr, "cachedConstructs.cpp::%s %s failed: %d\n","buildCachedConstructs","clGetPlatformIDs",error);
      return error;
    }

    error = clGetDeviceIDs(occs->platformId, CL_DEVICE_TYPE_GPU, 1, 
                           &(occs->deviceId), NULL);

    if(error){
      fprintf(stderr, "cachedConstructs.cpp::%s %s failed: %d\n","buildCachedConstructs","clGetDeviceIDs",error);
      return error;
    }

    if(occs->contextProperties != NULL)
      occs->contextProperties[5] = (cl_context_properties)(occs->platformId);

    occs->context = clCreateContext(0, 1, &(occs->deviceId), NULL, NULL, &error);

    occs->ioqueue = clCreateCommandQueue(occs->context, occs->deviceId, 0, &error);

    occs->ooqueue = clCreateCommandQueue(occs->context, occs->deviceId, 
                                   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);

    *rebuildMemoryObjects = 1;

  }

  return error;
}

void cleanupConstructs(ocl_constructs * occs){

  clReleaseCommandQueue(occs->ioqueue);
  occs->ioqueue = NULL;
  clReleaseContext(occs->context);
  occs->context = NULL;
  occs->deviceId = NULL;
  occs->platformId = NULL;

}
