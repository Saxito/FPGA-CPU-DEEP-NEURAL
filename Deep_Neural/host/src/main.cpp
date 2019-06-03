#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "preprocessing.h"

using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;



// Function prototypes
bool init();
void cleanup();
void create_kernel();

int main(int argc, char** argv)
{   
    double* matrix = preprocessing();

    return 1;
}


void create_kernel(){
  cl_int status;

  if(!init()) {
    return;
  }

  // Create the kernel using the name of the function in .cl file
  kernel = clCreateKernel(program, "hello_world"  , &status);
  checkError(status, "Failed to create kernel");
  // Set kernel arguments (one by argument)
  int year = 2017;
  status = clSetKernelArg(kernel, 0, sizeof(cl_int), &year);
  checkError(status, "Failed to set kernel arg 0");
  printf("\nKernel initialization is complete.\n");


  printf("Launching the kernel...\n\n");
  size_t size = 1000; // Not clear yet
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, &size, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  status = clFinish(queue);  // Wait the queue
  checkError(status, "Failed to finish");
  printf("\nKernel execution is complete.\n");

  // Free
  cleanup();
}

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform. TODO: More than one platform can be found
  cl_uint num_platforms;
  status = clGetPlatformIDs(1, &platform, &num_platforms);
  checkError(status, "Failed clGetPlatformIDs.");

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0]; // We'll just use the first device.

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program, using the name of aocx file
  std::string binary_file = getBoardBinaryFile("hello_world", device);
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
}
