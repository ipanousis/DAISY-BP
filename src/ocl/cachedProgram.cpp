/*

  Project  : DAISY in OpenCL
  Author   : Ioannis Panousis - ip223@bath.ac.uk
  Creation : December/2011

  File: cachedProgram.cpp

*/
#include "ocl/cachedProgram.h"
#include <stdio.h>

cl_program CreateProgram(cl_context context, cl_device_id device, 
                         const char * clFilename, const char * options){

  // load kernel source from .cl file and generate program

  cl_int error = 0;

  cl_program program = NULL;

  FILE * fp = fopen(clFilename, "r");

  if(fp == NULL){

    fprintf(stderr, "cachedProgram.c::CreateProgram Failed to open %s for reading\n",clFilename);
    return NULL;

  }

  char srcStr[65536];

  error = (cl_int)fread(srcStr, 1, 65536, fp);

  fclose(fp);

  const char * srcStr2 = srcStr;

  program = clCreateProgramWithSource(context, 1, (const char**)&srcStr2, NULL, NULL);

  if(program == NULL){

    fprintf(stderr, "cachedProgram.c::CreateProgram failed to create program with source\n");
    return NULL;

  }

  error = clBuildProgram(program, 0, NULL, options, NULL, NULL);

  if(error != CL_SUCCESS){

    char* buildInfo = (char*)malloc(sizeof(char) * 16384);
    size_t buildInfoLength;

    fprintf(stderr, "OpenCL build FAILED : %d\n", error);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          16384,
                          buildInfo,
                          &buildInfoLength);

    fprintf(stderr, "Build output = '%s'\n", buildInfo);

    clReleaseProgram(program);

    return NULL;
  }

  return program;
}

cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, 
                                   const char * binaryName, const char * options){
  cl_int error;

  // load binary .cl.bin
  FILE * fp = fopen(binaryName, "rb");

  if(fp == NULL){
    fprintf(stderr, "cachedProgram.c::CreateProgramFromBinary failed to open %s for reading\n", binaryName);
    return NULL;
  }

  // Determine binary size
  size_t binarySize;
  fseek(fp, 0, SEEK_END);
  binarySize = ftell(fp);
  rewind(fp);

  // Load binary from disk
  unsigned char * programBinary = (unsigned char*) malloc(sizeof(unsigned char) 
                                                          * binarySize);
  error = (cl_int)fread(programBinary, 1, binarySize, fp);
  fclose(fp);

  cl_program program = NULL;
  cl_int binaryStatus;

  program = clCreateProgramWithBinary(context, 1, &device, &binarySize,
                                      (const unsigned char**)&programBinary,
                                      &binaryStatus, &error);

  free(programBinary);

  if(error != CL_SUCCESS){
    fprintf(stderr, "cachedProgram.c::CreateProgramFromBinary failed to create with binary (%d)\n", error);
    return NULL;
  }

  if(binaryStatus != CL_SUCCESS){
    fprintf(stderr, "Invalid binary for device\n");
    return NULL;
  }

  error = clBuildProgram(program, 0, NULL, options, NULL, NULL);

  if(error != CL_SUCCESS){

    char* buildInfo = (char*)malloc(sizeof(char) * 16384);
    size_t buildInfoLength;

    fprintf(stderr, "cachedProgram.c::CreateProgramFromBinary failed (%d)\n", error);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          16384,
                          buildInfo,
                          &buildInfoLength);

    fprintf(stderr, "Build output = '%s'\n", buildInfo);

    clReleaseProgram(program);

    return NULL;
  }

  return program;
}

cl_int SaveProgramBinary(cl_program program, cl_device_id device, const char * binaryName){

  cl_int error = 0;

  cl_uint numDevices = 0;

  // 1 - Query for number of devices attached to program
  error = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 
                           sizeof(cl_uint), &numDevices, NULL);

  if(error != CL_SUCCESS){
    fprintf(stderr, "cachedProgram.c::SaveProgramBinary Error querying for number of devices\n");
    return error;
  }

  // 2 - Get all if the device IDs
  cl_device_id * devices = (cl_device_id*) malloc(sizeof(cl_device_id) * numDevices);

  error = clGetProgramInfo(program, CL_PROGRAM_DEVICES, 
                           sizeof(cl_device_id) * numDevices,
                           devices, NULL);

  if(error != CL_SUCCESS){
    fprintf(stderr, "cachedProgram.c::SaveProgramBinary error querying for devices\n");
    return error;
  }

  // 3 - Determine the size of each program binary
  size_t * programBinarySizes = (size_t*) malloc(sizeof(size_t) * numDevices);
  
  error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 
                           sizeof(size_t) * numDevices,
                           programBinarySizes, NULL);

  if(error != CL_SUCCESS){
    fprintf(stderr, "cachedProgram.c::SaveProgramBinary Error querying for program binary sizes\n");
    return error;
  }

  unsigned char ** programBinaries = (unsigned char**)
                                      malloc(sizeof(unsigned char*) * numDevices);

  cl_uint i;
  for(i = 0; i < numDevices; i++)
    programBinaries[i] = (unsigned char*) malloc(sizeof(unsigned char) * 
                                                 programBinarySizes[i]);

  // 4 - Get all of the program binaries
  error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 
                           sizeof(unsigned char*) * numDevices,
                           programBinaries, NULL);

  if(error != CL_SUCCESS){
    fprintf(stderr, "cachedProgram.c::SaveProgramBinary Error querying for program binaries\n");
    return error;
  }

  // 5 - Store binaries for the device requested out to disk for future reading
  for(i = 0; i < numDevices; i++){

    if(devices[i] == device){

      FILE * fp = fopen(binaryName, "wb");

      fwrite(programBinaries[i], 1, programBinarySizes[i], fp);

      fclose(fp);
      break;

    }

  }

  free(devices);
  free(programBinarySizes);
  for(i = 0; i < numDevices; i++)
    free(programBinaries[i]);
  free(programBinaries);

  return error;

}

cl_int buildCachedProgram(ocl_constructs * occs, const char * filebase, const char * options){

  cl_int error = 0;

  if(occs->program != NULL)
    return error;

  char binaryName[256];
  binaryName[0] = '\0';
  strcat(binaryName, filebase);
  strcat(binaryName, ".bin");

  occs->program = CreateProgramFromBinary(occs->context, occs->deviceId, binaryName, options);

  if(occs->program == NULL){

    occs->program = CreateProgram(occs->context, occs->deviceId, filebase, options);

    if(occs->program == NULL){
      //Cleanup(context, commandQueue, program, kernel, memObjects);
      fprintf(stderr, "cachedProgram.c::buildCachedProgram failed to build program\n");
      return 1;
    }

    if(SaveProgramBinary(occs->program, occs->deviceId, binaryName)){
      fprintf(stderr, "cachedProgram.c::buildCachedProgram failed to save program binary\n");
      //CLeanup
      return 2;
    }

  }

  return error;

}
