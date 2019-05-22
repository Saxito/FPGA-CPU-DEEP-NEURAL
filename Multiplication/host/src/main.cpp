/*
 * Meant to be used as template for new applications...
 * Recommended to refer to OpenCL at Khronos:
 *    https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/
 * Recommended to refer to Intel Altera OpenCL SDK for FPGA Programming Guide
 * Recommended to refer to Intel Altera OpenCL SDK for FPGA Best Practices Guide 
 * 
 * Template HOST to call simple kernel.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <omp.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;
static cl_int status;

#define DIMENSION 8192
#define SIZE 8192
#define LOCAL_DIM 64

float start_time, spend_time_FPGA, spend_time_CPU;
int dimension; 
float* A;
float* B;
float* C;

cl_mem ABuffer, BBuffer, CBuffer;

// Function prototypes
bool init();
void cleanup();
void make_matrix(float* m, int type);
void setArgument();
void compute_matrix(float* a, float* b, float* c, int n);
void show_matrix(float* m);
void create_buffer();
void write_buffer();
void read_buffer();
void devices_info();


// Entry point.
int main() {

  cl_int status;

  if(!init()) {
    return -1;
  }

  printf("Begin load some matrix\n");
  //full matrix with not random number 
  A=(float*)malloc(sizeof(float)*DIMENSION*DIMENSION);
  B=(float*)malloc(sizeof(float)*DIMENSION*DIMENSION);
  C=(float*)malloc(sizeof(float)*DIMENSION*DIMENSION);

  make_matrix(A,1);
  make_matrix(B,1);
  make_matrix(C,0);
  printf("Matrix loaded\n");
  dimension = DIMENSION;

  // Create the kernel using the name of the function in .cl file
  kernel = clCreateKernel(program, "kmul2", &status);
  //kernel = clCreateKernel(program, "kmul_opti"  , &status);
  checkError(status, "Failed to create kernel");
  

  //FPGA Matrix Multiplication
  start_time = getCurrentTimestamp();
  create_buffer();
  write_buffer();
  setArgument();

  printf("Launching the kernel...\n\n");

  devices_info();
  
  //If kmul1
  size_t global_size[2]={SIZE,SIZE};
  size_t local_size[2]= {LOCAL_DIM,LOCAL_DIM}; 


  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  read_buffer();
  spend_time_FPGA += getCurrentTimestamp() - start_time;

  //reload the first matrix;
  make_matrix(C,0);

  //CPU Matrix multiplication
  start_time =getCurrentTimestamp();
  //compute_matrix(A,B,C, dimension);
  spend_time_CPU+= getCurrentTimestamp() -start_time;

  printf("FPGA time :%f\n", spend_time_FPGA);
  printf("CPU time%f\n", spend_time_CPU );

  status = clFinish(queue);  // Wait the queue
  checkError(status, "Failed to finish");
 
  printf("\nKernel execution is complete.\n");

  // Free
  cleanup();

  return 0;
}

void make_matrix(float* m, int type){
  for(int i =0; i<DIMENSION; i++){
    for(int j= 0; j<DIMENSION; j++){
      if(type != 0){
        if(i==j){
          m[i*DIMENSION+j]=1;
        }else{
          m[i*DIMENSION+j]=0;
       }
      }else{
        m[i*DIMENSION+j]=0;
      }
    }
  }
  //show_matrix(m);
  return;
}

void show_matrix(float* m){
  for(int i =0; i<DIMENSION; i++){
    printf("\n");
    for(int j =0; j<DIMENSION; j++){
      printf("%.1f ",m[i*DIMENSION+j]);
    }
  }
  printf("\n");
}

void compute_matrix(float* a, float* b ,float* c, int n){
  int i, j, k;
  //#pragma omp parallel private(i, j)
  #pragma unroll
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        // C(i, j) = sum(over k) A(i,k) * B(k,j)
        for (k = 0; k < n; k++) {
          c[i*n+j] += a[i*n+k] * b[k*n+j];
        }
      }
    }
}

void create_buffer(){

  printf("Let's create buffer\n");
  ABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * DIMENSION*DIMENSION, A, &status);
  BBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * DIMENSION*DIMENSION, B, &status);
  CBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * DIMENSION*DIMENSION, C, &status);
  printf("Buffer create\n");
}

void write_buffer(){
  printf("Let's write buffer\n");
  status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
          0, sizeof(float) * DIMENSION*DIMENSION, A, 0, NULL, NULL);
  status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
          0, sizeof(float) * DIMENSION*DIMENSION, B, 0, NULL, NULL);
  status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
          0, sizeof(float) * DIMENSION*DIMENSION, C, 0, NULL, NULL);
  printf("Buffer writen\n");
}

void read_buffer(){
  status  = clEnqueueReadBuffer(queue, ABuffer, CL_TRUE,
      0, sizeof(float) * DIMENSION*DIMENSION , A  , 0, NULL, NULL);
  status  = clEnqueueReadBuffer(queue, BBuffer, CL_TRUE,
      0, sizeof(float) * DIMENSION*DIMENSION , B , 0, NULL, NULL);
  status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(float) * DIMENSION*DIMENSION , C  , 0, NULL, NULL);

  //show_matrix(A);
  //show_matrix(B);
  //show_matrix(C);
}

void setArgument (){

  printf("Let's set argument\n");
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ABuffer);
  checkError(status, "Failed to set kernel arg 0");
  
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &BBuffer);
  checkError(status, "Failed to set kernel arg 1");
  
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &CBuffer);
  checkError(status, "Failed to set kernel arg 2");

  status=clSetKernelArg(kernel,3,sizeof(int), &dimension);
  printf("Finish set argument \n");
 
  printf("\nKernel initialization is complete.\n");
  

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
  std::string binary_file = getBoardBinaryFile("kmul", device);
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return true;
}

void devices_info(){

  size_t lenght;
  clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_GROUP_SIZE : %i\n",(int)lenght);

  cl_uint cl_lenght;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &cl_lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %i\n",(int)cl_lenght);

  size_t* size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t*), &size ,NULL);
  printf("CL_DEVICE_MAX_WORK_ITEM_SIZES : %i, %i,%i \n\n", (uint)size[0], (uint)size[1], (uint)size[2]);

  
}
void cleanup() {
// Free the resources allocated during initialization void cleanup() {
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
