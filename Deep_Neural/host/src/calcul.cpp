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
static cl_kernel change_weight = NULL;
static cl_kernel soustraction = NULL;


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

 	int sigmoide = 1;

 	status = clSetKernelArg(matrix_sig, 6, sizeof(cl_int), &sigmoide);
 	checkError(status, "Failed to set kernel arg 2");

	//lauch queue
	size_t global_size[2]={(size_t)m,(size_t)m};

   	size_t local_size[2]= {(size_t)1,(size_t)1};

   	status = clEnqueueNDRangeKernel(queue, matrix_sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	//sigmoide
   	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(double) * m , c  , 0, NULL, NULL);
   		
	//cleanup();
}


void soustraction_vector_kernel(double* a, double* b, double* c, int n){
	
	// create buffer;
	cl_mem ABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*n, a, &status);
	cl_mem BBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n, b, &status);
	cl_mem CBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double) * n, c, &status);


	//write buffer
	status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
	      0, sizeof(double) *n, a, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
	      0, sizeof(double) * n, b, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
	      0, sizeof(double) * n, c, 0, NULL, NULL);

	//set argument
	status = clSetKernelArg(soustraction, 0, sizeof(cl_mem), &ABuffer);
	checkError(status, "Failed to set kernel arg 0");

	status |= clSetKernelArg(soustraction, 1, sizeof(cl_mem), &BBuffer);
	checkError(status, "Failed to set kernel arg 1");

	status = clSetKernelArg(soustraction, 2, sizeof(cl_mem), &CBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(soustraction, 3, sizeof(cl_int), &n);
	checkError(status, "Failed to set kernel arg 2");

	//lauch queue
	size_t global_size[2]={(size_t)n,(size_t)n};

   	size_t local_size[2]= {(size_t)1,(size_t)1};

   	status = clEnqueueNDRangeKernel(queue, soustraction, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	//sigmoide
   	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(double) * n , c  , 0, NULL, NULL);
}


void soustraction_vector(double* a, double* b, double* c, int n){
	for (int i = 0; i < n; ++i)
	{
		c[i] = a[i]-b[i];
	}
}

void compute_matrix(double* a, double* b, double *c, int n, int m){
  int i, j;
  	{	
	    for (i = 0; i < m; i++) {
	      for (j = 0; j < n; j++) {
	        c[i] += b[i*n+j]*a[j];
	      }
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

 	int sigmoide = 0;

 	status = clSetKernelArg(matrix_sig, 6, sizeof(cl_int), &sigmoide);
 	checkError(status, "Failed to set kernel arg 2");

	//lauch queue
	size_t global_size[2]={(size_t)m,(size_t)m};

   	size_t local_size[2]= {(size_t)1,(size_t)1};

   	status = clEnqueueNDRangeKernel(queue, matrix_sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	//sigmoide
   	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
      0, sizeof(double) * m , c  , 0, NULL, NULL);
   
}


void change_weight_CPU(double * weight, double* biais, double* error_next, double* value_next, double leanintegrate, 
						int raw, int col, int isoutput){

	int i,j;
	for (j = 0; j < col; ++j)
	{	
		double tmp= 0.0;
  		
		for (i = 0; i < raw; ++i)
		{
			tmp = error_next[j]*leanintegrate;
			if(isoutput){
				weight[i*col+j] -=tmp;
				if(weight[i*col+j]>5.0){
					weight[i*col+j]=5.0;
				}else if(weight[i*col+j]<-5.0){
					weight[i*col+j]=-5.0;
				}
			}
			else{
				tmp*= (1-(value_next[j]*value_next[j]));
				weight[i*col+j] -=tmp;
			}
		}
		if(isoutput){
			biais[j]-=tmp;
			if(biais[j]>5.0){
				biais[j]=5.0;
			}
			else if(biais[j]<-5.0){
				biais[j]=-5.0;
			}
		}else{
			biais[j]-=tmp;

			
		}
	}
}

void change_weight_kernel(double * weight, double* biais, double* error_next, double* value_next, double leanintegrate, 
						int raw, int col, int isoutpout){
	
	// create buffer;

	cl_mem WeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*raw*col, weight, &status);
	cl_mem ErrorNextBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*col, error_next, &status);
	cl_mem ValueNextBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      sizeof(double)*col , value_next, &status);
	//write buffer
	status  = clEnqueueWriteBuffer(queue, WeightBuffer, CL_FALSE,
	      0, sizeof(double)*raw*col, weight, 0, NULL, NULL);;
	status  = clEnqueueWriteBuffer(queue, ErrorNextBuffer, CL_FALSE,
	      0, sizeof(double)*col, error_next, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, ValueNextBuffer, CL_FALSE,
	      0, sizeof(double)*col, value_next, 0, NULL, NULL);

	//set argument
	status = clSetKernelArg(change_weight, 0, sizeof(cl_mem), &WeightBuffer);
	checkError(status, "Failed to set kernel arg 0");

	status |= clSetKernelArg(change_weight, 1, sizeof(cl_mem), &ErrorNextBuffer);
	checkError(status, "Failed to set kernel arg 1");

	status = clSetKernelArg(change_weight, 2, sizeof(cl_mem), &ValueNextBuffer);
	checkError(status, "Failed to set kernel arg 2");

	status = clSetKernelArg(change_weight, 3, sizeof(cl_double), &leanintegrate);
	checkError(status, "Failed to set kernel arg 4");

	status = clSetKernelArg(change_weight, 4, sizeof(cl_int), &raw);
 	checkError(status, "Failed to set kernel arg 5");

 	status = clSetKernelArg(change_weight, 5, sizeof(cl_int), &col);
 	checkError(status, "Failed to set kernel arg 6");

 	status = clSetKernelArg(change_weight, 6, sizeof(cl_int), &isoutpout);
 	checkError(status, "Failed to set kernel arg 6");


	//lauch queue
	size_t global_size[2]={(size_t)raw,(size_t)col};

   	size_t local_size[2]= {(size_t)1,(size_t)1};

   	status = clEnqueueNDRangeKernel(queue, change_weight, 2, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

   	status  = clEnqueueReadBuffer(queue, WeightBuffer, CL_TRUE,
      0, sizeof(double)*raw*col, weight, 0, NULL, NULL);

   	
   	#pragma omp for
   	for (int i = 0; i < col; ++i)
   	{	
   		double tmp = error_next[i]*leanintegrate;
   		if (!isoutpout)
   		{
   			tmp *= (1-(value_next[i]*value_next[i]));
   			biais[i]=tmp;

   		}else{
   			biais[i]=tmp;
   			if(biais[i]<-5.0){
	   			biais[i]=-5.0;
	   		}
	   		if(biais[i]>5.0){
	   			biais[i]=5.0;
	   		}
   		}
   	}

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

  //create kernel 
  	matrix_sig = clCreateKernel(program, "kvectormulmatrix", &status);
	checkError(status, "Failed to create kernel");

	matrix_sig = clCreateKernel(program, "kvectormulmatrix", &status);
	checkError(status, "Failed to create kernel");

	change_weight = clCreateKernel(program, "kchange_weight", &status);
	checkError(status, "Failed to create kernel");

	soustraction = clCreateKernel(program, "ksoustraction_vector", &status);
	checkError(status, "Failed to create kernel");


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
  
  clReleaseKernel(matrix_sig);
  clReleaseKernel(soustraction);
  clReleaseKernel(change_weight);

}

void devices_info(){

  size_t lenght;
  clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_GROUP_SIZE : %i\n",(int)lenght);

  cl_uint cl_lenght;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &cl_lenght,NULL);
  printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %i\n",(int)cl_lenght);
  
}

