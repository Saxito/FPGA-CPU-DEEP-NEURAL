/* 
In this matrix multiplication we only use square matrix.
We will compute spend time for multiplication with differentes way :
  -FPGA with naive implementation
  -FPGA with optimisation 
  -CPU with naive implementation
  -CPU with optimisation with OpenMP
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

#define DIMENSION_MIN 2
#define DIMENSION_MAX 512
#define MAX_SIZE_CPU 512

#define LOCAL_DIM 64
#define DEBUG 0

#define MAX_NUMBER_MATRIX 1000

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;
static cl_int status;

float* A;
float* B;
float* C;
float* result;
float total_time;

cl_mem ABuffer, BBuffer, CBuffer;

// Function 
float FPGA_matrix(int dimension_matrix, int optimisation);
float CPU_matrix(int dimension_matrix, int optimisation);


bool init();
void cleanup();
void make_matrix(float* m, int type, int dimension);
void setArgument(int dimension_matrix);
void compute_matrix(float* a, float* b, float* c, int n);
void show_matrix(float* m, int dimension_matrix);
void create_buffer(int dimension_matrix);
void write_buffer(int dimension_matrix);
void read_buffer(int dimension_matrix);
void devices_info();
void transpose(float* a, float* b);
void compute_matrix_opti(float* a, float* b, float* c, int n);
void show_result(float* r);


float FPGA_matrix(int dimension_matrix, int optimisation){

    float start_time =0.f, spend_time_FPGA=0.f;
    
    if(!init()) {
      return -1;
    }
    if(DEBUG)
      printf("Begin load some matrix, %i\n", dimension_matrix);
    
    //fill matrix with not random number 
    A=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);
    B=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);
    C=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);

    make_matrix(A,1,dimension_matrix);
    make_matrix(B,1,dimension_matrix);
    make_matrix(C,0,dimension_matrix);
    
    if(DEBUG)
      printf("Matrix loaded\n");

    // Choose optimisation or not;
    if(optimisation){
        kernel = clCreateKernel(program, "kmul2", &status);
    }else{
        kernel = clCreateKernel(program, "kmul1", &status);
    }
    checkError(status, "Failed to create kernel");
    

    //FPGA Matrix Multiplication
    start_time = getCurrentTimestamp();
   
    create_buffer(dimension_matrix);
    write_buffer(dimension_matrix);
    setArgument(dimension_matrix);

    if(DEBUG)
      printf("Launching the kernel...\n\n");

    //devices_info();
    
    size_t global_size[2]={(size_t)dimension_matrix,(size_t)dimension_matrix};

    size_t local_size[2]= {(size_t)dimension_matrix,(size_t)dimension_matrix}; 
    if(dimension_matrix>64){
      local_size[0]= (size_t)LOCAL_DIM;
      local_size[1]= (size_t)LOCAL_DIM; 
    }
    
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    read_buffer(dimension_matrix);

    if(DEBUG)
      show_matrix(C,dimension_matrix);

    spend_time_FPGA += getCurrentTimestamp() - start_time;

    status = clFinish(queue);  // Wait the queue
    checkError(status, "Failed to finish");
    
    if(DEBUG)
      printf("\nKernel execution is complete.\n");

    // Free
    cleanup(); 

    return spend_time_FPGA;
}

float CPU_matrix(int dimension_matrix, int optimisation){
    float start_time, spend_time_CPU=0.f;;

    //Fill matrix with random number;
    A=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);
    B=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);
    C=(float*)malloc(sizeof(float)*dimension_matrix*dimension_matrix);

    make_matrix(A,1,dimension_matrix);
    make_matrix(B,1,dimension_matrix);
    make_matrix(C,0,dimension_matrix);

    if(dimension_matrix<=MAX_SIZE_CPU){
      start_time =getCurrentTimestamp();
      if(optimisation){
        compute_matrix_opti(A,B,C, dimension_matrix);
      }else{
        compute_matrix(A,B,C, dimension_matrix);
      }
      if(DEBUG)
        show_matrix(C,dimension_matrix);
      spend_time_CPU+= getCurrentTimestamp() -start_time;
    }
    return spend_time_CPU;
}


int main() {

  int i=0;
  int dim = DIMENSION_MIN;
  result = (float*)malloc(5*sizeof(float)*32); // maximum 32 bits
  total_time=0.f;
  while(dim <= DIMENSION_MAX)
  { 
    result[i] = (float)dim;
    result[i+1] = FPGA_matrix(dim,0);
    result[i+2] = FPGA_matrix(dim,1);
    result[i+3] = CPU_matrix(dim,0);
    result[i+4] = CPU_matrix(dim,1);
    total_time+=result[i+1]+result[i+2]+result[i+3]+result[i+4];
    dim=dim*2;
    i+=5;
  }
  show_result(result);

  return 0;
}

void make_matrix(float* m, int type, int dim){
  for(int i =0; i<dim; i++){
    for(int j= 0; j<dim; j++){
      if(type != 0){
        m[i*dim+j]= random()%MAX_NUMBER_MATRIX;
      }else{
        m[i*dim+j]=0;
      }
    }
  }
  
  if(DEBUG)
    show_matrix(m, dim);
  
  return;
}

void show_matrix(float* m, int dim){
  for(int i =0; i<dim; i++){
    printf("\n");
    for(int j =0; j<dim; j++){
      printf("%.1f ",m[i*dim+j]);
    }
  }
  printf("\n");
}

void compute_matrix(float* a, float* b ,float* c, int n){
  int i, j, k;
  //#pragma omp parallel for 
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
          c[i*n+j] += a[i*n+k] * b[k*n+j];
        }
      }
    }
}

void transpose(float *a, float *b, int n) {
    int i,j;
    for(i=0; i<n; i++) {
      for(j=0; j<n; j++) {
        b[j*n+i] = a[i*n+j];
      }
    }
}

void compute_matrix_opti(float* a, float* b ,float* c, int n){
  float*b_T;
  b_T = (float*)malloc(sizeof(float)*n*n);
  transpose(b,b_T, n);
  #pragma omp parallel
  {
    int i, j, k;
    #pragma omp for
    for (i = 0; i < n; i++) { 
      for (j = 0; j < n; j++) {
        float dot  = 0;
        for (k = 0; k < n; k++) {
          dot += a[i*n+k]*b_T[j*n+k];
        } 
        c[i*n+j ] = dot;
      }
    }

  }
}

void create_buffer(int dim){
  if(DEBUG)
    printf("Let's create buffer\n");

  ABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * dim*dim, A, &status);
  BBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * dim*dim, B, &status);
  CBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * dim*dim, C, &status);
  
  if(DEBUG)
    printf("Buffer create\n");
}

void write_buffer(int dim){
  if(DEBUG)
    printf("Let's write buffer\n");

  status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
          0, sizeof(float) * dim*dim, A, 0, NULL, NULL);
  status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
          0, sizeof(float) * dim*dim, B, 0, NULL, NULL);
  status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
          0, sizeof(float) * dim*dim, C, 0, NULL, NULL);
  if(DEBUG)
    printf("Buffer writen\n");
}

void read_buffer(int dim){
  status  = clEnqueueReadBuffer(queue, ABuffer, CL_TRUE,
      0, sizeof(float) * dim*dim , A  , 0, NULL, NULL);
  status  = clEnqueueReadBuffer(queue, BBuffer, CL_TRUE,
      0, sizeof(float) * dim*dim , B , 0, NULL, NULL);
  status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(float) * dim*dim , C  , 0, NULL, NULL);

  //show_matrix(A);
  //show_matrix(B);
  //show_matrix(C);
}

void setArgument (int dim){

  if(DEBUG)
    printf("Let's set argument\n");
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ABuffer);
  checkError(status, "Failed to set kernel arg 0");
  
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &BBuffer);
  checkError(status, "Failed to set kernel arg 1");
  
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &CBuffer);
  checkError(status, "Failed to set kernel arg 2");

  status=clSetKernelArg(kernel,3,sizeof(int), &dim);
  if(DEBUG)
    printf("Finish set argument \n");
 
  

}

void show_result(float* result){

  printf("MATRIX MULTIPLICATION\n");
  printf("Matrix dimension;FPGA time;FPGA time Optimisation;CPUTIME; CPU tim Optimisation\n");
  for(int i=0, k=2; k<=DIMENSION_MAX; k=k*2, i+=5){
    printf("%f ;%f ;%f ;%f ;%f \n",result[i],result[i+1],result[i+2],result[i+3],result[i+4] );
  }
  printf("total_time; %f\n",total_time);
  return;
}


bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  srand(time(NULL));
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
