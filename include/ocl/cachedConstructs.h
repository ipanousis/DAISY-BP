/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : December/2011

  File: cachedConstructs.h

*/
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <GL/glx.h>

#ifndef OCL_CONSTRUCTS
#define OCL_CONSTRUCTS
typedef struct ocl_constructs_tag{
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_context context;
  cl_program program;
  cl_command_queue queue;
  cl_mem * buffers;
  cl_kernel kernel;
  cl_context_properties* contextProperties;
  size_t workerSize;
  size_t groupSize;
  //cl_uint refreshCount;
} ocl_constructs;
#endif

ocl_constructs * newOclConstructs(cl_uint, cl_uint, cl_bool);

int buildCachedConstructs(ocl_constructs*, cl_bool*);
