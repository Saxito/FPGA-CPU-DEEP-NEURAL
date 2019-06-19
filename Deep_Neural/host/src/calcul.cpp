#include "calcul.h"
#include <cstring>
#include "AOCLUtils/aocl_utils.h"



// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static int status;
//kernel

static cl_kernel matrix_sig = NULL;
static cl_kernel sig = NULL;

using namespace aocl_utils;


//Fonction utiliser dans le rnnset 
double sigmoid(double a){
  double res= 1 / (1 + exp(-a));
  return res;
}

void compute_matrix_sig(double* a, double* b ,double* c, double* d, int n, int m){
  int i, j;
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        c[i] += b[j*m+i]*(a[j]+d[j]);
      }
      c[i]= sigmoid(c[i]);
    }
}

void compute_matrix_sig_kernel(double* a, double* b ,double* c, double* d, int n, int m){
	//multiplication

	//create kernel 

	matrix_sig = clCreateKernel(program, "kvectormulmatrix", &status);
	checkError(status, "Failed to create kernel");
	
	// create buffer;

	cl_mem ABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*n, a, &status);
	cl_mem BBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n*m, b, &status);
	cl_mem CBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * m, c, &status);
	cl_mem DBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n, d, &status);

	//write buffer
	status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
	      0, sizeof(double) *n, a, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
	      0, sizeof(double) * n*m, b, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
	      0, sizeof(double) * m, c, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, DBuffer, CL_FALSE,
	      0, sizeof(double) * n, d, 0, NULL, NULL);

	//set argument
	status = clSetKernelArg(matrix_sig, 0, sizeof(cl_mem), &ABuffer);
	checkError(status, "Failed to set kernel arg 0");

	status |= clSetKernelArg(matrix_sig, 1, sizeof(cl_mem), &BBuffer);
	checkError(status, "Failed to set kernel arg 1");

	status = clSetKernelArg(matrix_sig, 2, sizeof(cl_mem), &CBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 3, sizeof(cl_mem), &DBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 4, sizeof(cl_int), &n);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 5, sizeof(cl_int), &m);
 	checkError(status, "Failed to set kernel arg 2");

	//lauch queue
	size_t global_size[2]={(size_t)m,(size_t)m};

   	size_t local_size[2]= {(size_t)m,(size_t)m};

   	status = clEnqueueNDRangeKernel(queue, matrix_sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	//sigmoide
   	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(double) * m , c  , 0, NULL, NULL);
   	
   
   	status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
	     0, sizeof(double) * m, c, 0, NULL, NULL);

	//create kernel 
	sig = clCreateKernel(program, "ksigmoide", &status);
	checkError(status, "Failed to create kernel");

	status = clSetKernelArg(sig, 0, sizeof(cl_mem), &CBuffer);
	checkError(status, "Failed to set kernel arg 0");

	status |= clSetKernelArg(sig, 1, sizeof(int), &m);
	checkError(status, "Failed to set kernel arg 1");

	status = clEnqueueNDRangeKernel(queue, sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

	//red buffer
	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(float) * m , c  , 0, NULL, NULL);

	clReleaseKernel(matrix_sig);
	clReleaseKernel(sig);
	//cleanup();
}

//fonction utiliser pour le learn

void soustraction_vector(double* a, double* b, double* c, int n){
	#pragma omp parallel
  	{
		#pragma omp for
		for (int i = 0; i < n; ++i)
		{
			c[i] = a[i]-b[i];
		}
	
	}
}

void compute_matrix(double* a, double* b, double *c, int n, int m){
  int i, j;

    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        c[i] += b[i*n+j]*a[j];
      }
    }
}

void compute_matrix_kernel(double* a, double* b, double *c, int n, int m){
	double * null =(double*)malloc(sizeof(double)*n);
	#pragma omp for
	for (int i = 0; i < n; ++i)
	{
		null[i]= 0.0;
	}

	matrix_sig = clCreateKernel(program, "kvectormulmatrix", &status);
	checkError(status, "Failed to create kernel");
	
	// create buffer;

	cl_mem ABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*n, a, &status);
	cl_mem BBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n*m, b, &status);
	cl_mem CBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * m, c, &status);
	cl_mem DBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n, null, &status);

	//write buffer
	status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
	      0, sizeof(double) *n, a, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
	      0, sizeof(double) * n*m, b, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
	      0, sizeof(double) * m, c, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, DBuffer, CL_FALSE,
	      0, sizeof(double) * n, null, 0, NULL, NULL);

	//set argument
	status = clSetKernelArg(matrix_sig, 0, sizeof(cl_mem), &ABuffer);
	checkError(status, "Failed to set kernel arg 0");

	status |= clSetKernelArg(matrix_sig, 1, sizeof(cl_mem), &BBuffer);
	checkError(status, "Failed to set kernel arg 1");

	status = clSetKernelArg(matrix_sig, 2, sizeof(cl_mem), &CBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 3, sizeof(cl_mem), &DBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 4, sizeof(cl_int), &n);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(matrix_sig, 5, sizeof(cl_int), &m);
 	checkError(status, "Failed to set kernel arg 2");

	//lauch queue
	size_t global_size[2]={(size_t)m,(size_t)m};

   	size_t local_size[2]= {(size_t)m,(size_t)m};

   	status = clEnqueueNDRangeKernel(queue, matrix_sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	//sigmoide
   	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(double) * m , c  , 0, NULL, NULL);
   
}

void change_weight_kernel(){
	
}

bool init() {
  cl_int status;
  printf("begin init\n");
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
  std::string binary_file = getBoardBinaryFile("calcul", device);
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
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

void devices_info(){

  size_t lenght;
  clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_GROUP_SIZE : %i\n",(int)lenght);

  cl_uint cl_lenght;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &cl_lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %i\n",(int)cl_lenght);
  
}




void vector_multiplication_constant(double* a, double b, double* c, double taille,int behaviour){
  for (int i = 0; i < taille; ++i)
  { 
    
    if(behaviour){
      c[i]*= a[i]*b;
    }
    else{
      c[i]=a[i]*b;
    }
  }
}
void vector_multiplication(double* wchange, double* value, int col, int raw){
  for (int i = 0; i < col; ++i)
  {
   for (int j = 0; j < raw; ++j)
    {
      wchange[i*raw+j]*=value[j];
    } 
  }
}

void change_matrix(double* matrix, double* change, int ligne, int colonne){
  for (int i = 0; i < ligne; ++i)
  {
    for (int j = 0; j < colonne; ++j)
    {
      matrix[i*colonne+j]-=change[i*colonne+j];
    }
  }
}

void normalisation(double* a, int intervalle, int taille){
  for (int i = 0; i < taille; ++i)
  {
    if(a[i]>intervalle){
      a[i]=intervalle;
    }else if(a[i]<-intervalle){
      a[i]=-intervalle;
    }
  }
}


void vector_multiplication_carre(double* value, double* derivative, int taille){
  for (int i = 0; i < taille; ++i)
  {
    derivative[i] = 1.0 - (value[i]*value[i]);
  }
}
