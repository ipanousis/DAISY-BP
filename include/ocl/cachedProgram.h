#include <CL/cl.h>
#include <string.h>
#include "cachedConstructs.h"

#define CACHED_PROGRAM

cl_program CreateProgram(cl_context, cl_device_id, char *, const char *);

cl_program CreateProgramFromBinary(cl_context, cl_device_id, char *, const char*);

cl_int SaveProgramBinary(cl_program, cl_device_id, char*);

cl_int buildCachedProgram(ocl_constructs*, char*, const char *);
