#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "preprocessing.h"
#include "utils.h"
#include "AOCLUtils/aocl_utils.h"

static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static int status;

// OpenCL runtime configuration
//kernel
static cl_kernel matrix_sig = NULL;
static cl_kernel change_weight = NULL;
static cl_kernel soustraction = NULL;

using namespace aocl_utils;


typedef struct _layer LAYER;

int layer_size[NB_LAYOUT];
int col_matrix;
int raw_matrix;
int nb_error;
double average_sure =0.0;
double overload = 0.0;

struct _layer {
	int typeLayer;
	int nbnode;
	double* value;
	cl_mem BufferValue;
	
	double* weight;
	cl_mem BufferWeight;

	double* biais;
	cl_mem BufferBiais;
	
	double* value_prev;
	cl_mem BufferValue_Prev;
	
	double* error;
	cl_mem BufferError;
	
	double* error_prev;
};



double geterror(LAYER* tab_layer, double* out){
	double res =0.0;
	int nb=0;
	double sum=0.0;

  	//#pragma omp for
	for (int i = 0; i < tab_layer->nbnode; ++i)
	{ 
		res=tab_layer->value[i]-out[i];
		if(res<0){
			sum+=-res ;
		}else{
			sum+=res;
		}
		nb++;
	}
	return sum/(nb+1);
}

double* gettab_result(LAYER* tab_layer){
	double* res= (double*)malloc(sizeof(double)*tab_layer->nbnode);
  	#pragma omp for
	for (int i = 0; i < tab_layer->nbnode; ++i)
	{
		res[i] = tab_layer->value[i];
	}
	return res;
}


void init_layer_size(){
	layer_size[0]= col_matrix;
	for (int i = 1; i <NB_LAYOUT-1; ++i){
		layer_size[i]=SIZE_HD;
	}
	layer_size[NB_LAYOUT-1]=nb_error;
}



void ajustError(LAYER * layer){
	int i;
	double sum = 0.0;
    #pragma omp for
	for (i = 0; i < layer->nbnode; i++){ 
		sum += layer->value[i];
	}

    #pragma omp for
	for (i = 0; i < layer->nbnode; i++){ 
		layer->value[i] /= sum;
	}
}



void compute_matrix_sig(double* a, double* b ,double* c, double* d, int n, int m){
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			c[i] += b[j*m+i]*(a[j]);
		}
		c[i]= sigmoid(c[i]);
	}
}

void compute_matrix_sig_kernel(double* a, cl_mem ABuffer, double* b , cl_mem BBuffer , double* c, cl_mem CBuffer, double* d, cl_mem DBuffer,
	int n, int m){
	//multiplication
	double start, end;

	start = getCurrentTimestamp();
	//write buffer
	status  = clEnqueueWriteBuffer(queue, CBuffer, CL_FALSE,
		0, sizeof(double)*m, c, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, ABuffer, CL_FALSE,
		0, sizeof(double) *n, a, 0, NULL, NULL);
	status  = clEnqueueWriteBuffer(queue, BBuffer, CL_FALSE,
		0, sizeof(double)*n*m, b, 0, NULL, NULL);	
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
	status |= clFinish(queue);

	end = getCurrentTimestamp();
	overload += end-start;
	//lauch queue
	size_t global_size[2]={(size_t)m,(size_t)m};

	size_t local_size[2]= {(size_t)1,(size_t)1};

	status = clEnqueueNDRangeKernel(queue, matrix_sig, 1, NULL, global_size, local_size, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
	status |= clFinish(queue);

   	//sigmoide
	start = getCurrentTimestamp();

	status  = clEnqueueReadBuffer(queue, CBuffer, CL_TRUE,
		0, sizeof(double) * m , c  , 0, NULL, NULL);

	end = getCurrentTimestamp();
	overload += end-start;

	//cleanup();
}


void soustraction_vector(double* a, double* b, double* c, int n){
	for (int i = 0; i < n; ++i)
	{	
		c[i] = (a[i]-b[i])*sigmoid(a[i])*sigmoid((1-a[i]));
	}
}

void compute_matrix(double* a, double* b, double *c, int n, int m, double* value){
	int i, j;
	{	
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				c[i] += b[i*n+j]*a[j];
			}
			c[i] *= value[i]*(1-value[i]);
		}
	}
}


void change_weight_CPU(double * weight, double* biais, double* error_next, double* value_next, double* value, double leanintegrate, 
	int raw, int col, int isoutput){
	int normalisation =1.0;
	int i,j;
	for (j = 0; j < col; ++j)
	{	
		double tmp= 0.0;

		for (i = 0; i < raw; ++i)
		{
			tmp = error_next[j]*leanintegrate*value[i];
			if(isoutput){
				weight[i*col+j] -=tmp;
			}
			else{
				//tmp*= value_next[j];
				weight[i*col+j] -=tmp;
				if(weight[i*col+j]>normalisation){
					weight[i*col+j]=normalisation;
				}else if(weight[i*col+j]<-normalisation){
					weight[i*col+j]=-normalisation;
				}
			}

			
		}
		tmp = error_next[j]*leanintegrate;
		if(isoutput){
			biais[j]-=tmp;
			// biais[j]=0;	
		}else{
			//tmp *= value_next[i];
			biais[j]-=tmp;
			if(biais[j]>normalisation){
				biais[j]=normalisation;
			}
			else if(biais[j]<-normalisation){
				biais[j]=-normalisation;
			}

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
	status |= clFinish(queue);

	//lauch queue
	size_t global_size[2]={(size_t)raw,(size_t)col};

	size_t local_size[2]= {(size_t)1,(size_t)1};

	status = clEnqueueNDRangeKernel(queue, change_weight, 2, NULL, global_size, local_size, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");


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

	status  = clEnqueueReadBuffer(queue, WeightBuffer, CL_TRUE,
		0, sizeof(double)*raw*col, weight, 0, NULL, NULL);

	clReleaseMemObject(WeightBuffer);
	clReleaseMemObject(ErrorNextBuffer);
	clReleaseMemObject(ValueNextBuffer);

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
 	
 	change_weight = clCreateKernel(program, "kchange_weight", &status);
 	checkError(status, "Failed to create kernel");

 	soustraction = clCreateKernel(program, "ksoustraction_vector", &status);
 	checkError(status, "Failed to create kernel");

 	//Create buffer for each Kernel


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

 void init_value(LAYER* l, float* matrix, int index){
 	for(int i=0; i<layer_size[0];i++){
 		l->value[i]=matrix[index*layer_size[0]+i];
 	}
 }

 void init_layer(LAYER* l, float* matrix, int index, int currentlayer){

 	l->typeLayer = currentlayer;
    //printf("%d\n", layer_size[currentlayer]);
 	if(currentlayer == 0){
 		l->nbnode = layer_size[currentlayer];
 		l->value = (double*)malloc(sizeof(double)*layer_size[currentlayer]);

    	#pragma omp for
 		for(int i=0; i<layer_size[currentlayer];i++){
 			l->value[i]=matrix[index*layer_size[currentlayer]+i];
      //printf("value in %f\n", l->value[i]);
 		}
 		l->weight = (double*)malloc(sizeof(double)*layer_size[currentlayer]*layer_size[currentlayer+1]);

    	#pragma omp for
 		for (int i = 0; i < layer_size[currentlayer+1]*layer_size[currentlayer]; ++i)
 		{
 			l->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
 		}
 	}else if(currentlayer == NB_LAYOUT-1){
 		l->nbnode = layer_size[currentlayer];
 		l->value = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
 	}else{
 		l->nbnode = layer_size[currentlayer];
 		l->value = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
 		l->weight = (double*)malloc(sizeof(double)*layer_size[currentlayer+1]*layer_size[currentlayer]);

   		#pragma omp for  
 		for (int i = 0; i < layer_size[currentlayer]*layer_size[currentlayer+1]; ++i)
 		{
 			l->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
 		}
 	}
 	l->biais = (double*)malloc(sizeof(double)*layer_size[currentlayer]);

    #pragma omp for
 	for (int i = 0; i < layer_size[currentlayer]; ++i)
 	{
 		l->biais[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
 	}
 	l->value_prev = (double*)malloc(sizeof(double)*layer_size[currentlayer]);

    #pragma omp for
 	for(int i=0; i<layer_size[currentlayer];i++){
 		l->value_prev[i]=0.0;
 	}
 	l->error= (double*)malloc(sizeof(double)*layer_size[currentlayer]);

    #pragma omp for
 	for(int i=0; i<layer_size[currentlayer];i++){
 		l->error[i]=0.0;
 	}
 	l->error_prev = (double*)malloc(sizeof(double)*layer_size[currentlayer]);

    #pragma omp for
 	for(int i=0; i<layer_size[currentlayer];i++){
 		l->error_prev[i]=0.0;
 	}

 	l->BufferValue = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 		sizeof(double)*layer_size[currentlayer], l->value, &status);
 	l->BufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 		sizeof(double)*layer_size[currentlayer]*layer_size[currentlayer+1], l->weight, &status);
 	l->BufferError = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 		sizeof(double)*l->nbnode, l->error, &status);
 	l->BufferBiais = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 		sizeof(double)*l->nbnode, l->biais, &status);
 	l->BufferValue_Prev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 		sizeof(double)*l->nbnode, l->value_prev, &status);



 }

 void free_layer(LAYER* l){
 	free(l->value);
 	free(l->weight);
 	free(l->biais);
 	free(l->value_prev);
 	free(l->error);
 	free(l->error_prev);
 	clReleaseMemObject(l->BufferValue);
 	clReleaseMemObject(l->BufferWeight);
 	clReleaseMemObject(l->BufferError);
 	clReleaseMemObject(l->BufferBiais);
 	clReleaseMemObject(l->BufferValue_Prev);


 }

 void rnnsetstart(LAYER* tab_layer[]){
 	for (int i=0; i<NB_LAYOUT;i++){
 		if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
 			for(int k =0; k<tab_layer[i]->nbnode ;k++){
 				tab_layer[i]->value_prev[k] = tanh(tab_layer[i]->value[k]);
 				tab_layer[i]->value[k] = 0.0;
 			} 
 		}else{
 			for(int k =0; k<tab_layer[i]->nbnode ;k++){
 				tab_layer[i]->value_prev[k] = tab_layer[i]->value[k];
 			}
 		}
 	}
 }


 void rnnset(LAYER* tab_layer[], double *out){
 	for (int i = 0; i < NB_LAYOUT; ++i)
 	{	
 		for (int j = 0; j < tab_layer[i]->nbnode; ++j){
 			tab_layer[i]->error[j]=0.0;
 		}
 		if(tab_layer[i]->typeLayer!=0){
 			for (int k = 0; k < tab_layer[i]->nbnode; ++k){
 				tab_layer[i]->value[k] = tab_layer[i]->biais[k];
 			}
 			if(KERNEL){
 				compute_matrix_sig_kernel(tab_layer[i-1]->value, tab_layer[i-1]->BufferValue, tab_layer[i-1]->weight, tab_layer[i-1]->BufferWeight,
 					tab_layer[i]->value, tab_layer[i]->BufferValue, tab_layer[i-1]->value_prev, tab_layer[i-1]->BufferValue_Prev,
 					tab_layer[i-1]->nbnode, tab_layer[i]->nbnode);     
 			}
 			else{
 				compute_matrix_sig(tab_layer[i-1]->value, tab_layer[i-1]->weight, tab_layer[i]->value, tab_layer[i-1]->value_prev, tab_layer[i-1]->nbnode, tab_layer[i]->nbnode);       
 			}
 		}
 	}
 	soustraction_vector(tab_layer[NB_LAYOUT-1]->value, out, tab_layer[NB_LAYOUT-1]->error, tab_layer[NB_LAYOUT-1]->nbnode);


 }

 void rnnlearn(LAYER* tab_layer[], double learningrate){
 	
 	// for (int i = NB_LAYOUT-2; i >= 0; i--){
  //   	//multiplication de matrice  error i = error i+1 * weight i
 	// 	compute_matrix(tab_layer[i+1]->error, tab_layer[i]->weight, tab_layer[i]->error, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode, tab_layer[i]->value);
 	// }
 	double normalize =5.0;
 	for(int i = NB_LAYOUT-2; i>=0; i--){
 		double* deltaW = (double*)malloc(sizeof(double)*tab_layer[i]->nbnode*tab_layer[i+1]->nbnode);
 		//if(i == NB_LAYOUT-2){
 		for (int j = 0; j < tab_layer[i]->nbnode; ++j)
 		{
 			for (int k = 0; k < tab_layer[i+1]->nbnode; ++k)
 			{
                    // printf("k : %d\n", k);
 				deltaW[j*tab_layer[i+1]->nbnode+k] = (tab_layer[i+1]->error[k]*(tab_layer[i]->value[j])*learningrate);
 				tab_layer[i]->error[j] += tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k]*tab_layer[i+1]->error[k];
 				tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k] -= deltaW[j*tab_layer[i+1]->nbnode+k];
 				

 				double tmp;
 				if(i != NB_LAYOUT-2){
 					tmp = (1 - tab_layer[i+1]->value[k])*tab_layer[i+1]->value[k];
 					tmp *= learningrate * tab_layer[i+1]->error[k];
 					tab_layer[i+1]->biais[k] -= tmp;
 				}else{
 					tab_layer[i+1]->biais[k] -= learningrate * tab_layer[i+1]->error[k];
 				}

 				if(tab_layer[i]->biais[k]>normalize){
 					tab_layer[i]->biais[k]=normalize;
 				}else if(tab_layer[i]->biais[k]<-normalize){
 					tab_layer[i]->biais[k]=-normalize;
 				} 

 				if(tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k]>normalize){
 					tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k]=normalize;
 				}else if(tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k]<(-normalize)){
 					tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k] = (-normalize);
 				}

 					// if (i == 0 && j == 0 && k==0){
 					// 	printf("%f",tab_layer[i]->weight[j*tab_layer[i+1]->nbnode+k] );
 					// 	printf("\nError rnnLearn%f\n", tab_layer[i+1]->error[k]);
 					// }

 			}
 			tab_layer[i]->error[j] *= (tab_layer[i]->value[j])*(1-tab_layer[i]->value[j]);
 			//tab_layer[i]->biais[j] -= learningrate *tab_layer[i]->error[j]*(tab_layer[i]->value[j])*(1-(tab_layer[i]->value[j]));



 		}
 		free(deltaW);
 	}

  #pragma omp for
 	for (int i = 0; i < NB_LAYOUT; ++i){
 		for (int j = 0; j < tab_layer[i]->nbnode; ++j){
 			tab_layer[i]->error_prev[j]=tab_layer[i]->error[j];
 		}
 	}
 }


 void learn_KDD(const char* file_name_learn, const char* file_name_test){
 	double start, end, pre;

 	LAYER* tab_layer[NB_LAYOUT];

 	start = getCurrentTimestamp();
 	float* matrix = preprocessing(file_name_learn,0);
 	pre = getCurrentTimestamp();
 	printf("Temps de preprocessing : %f\n", pre-start);
 	col_matrix = get_col_matrix();
 	raw_matrix = get_raw_matrix();
 	printf("%d\n",raw_matrix );
 	nb_error =get_nberror();
 	int* out_process = get_output();

 	int* out_compt = (int*)malloc(sizeof(int*)*nb_error);    
 	for (int i = 0; i < nb_error; i++)
 		out_compt[i] = 0;

 	if(KERNEL){
 		if(!init()){
 			printf("Erreur init KERNEL\n");;
 		}else{
 			devices_info();
 		}
 	}

 	init_layer_size();
 	double* out;

 	double jtot=0.0;
 	double learn=0.1;
 	double* tab_result;
 	for (int i = 0; i < NB_LAYOUT; ++i)
 	{
 		tab_layer[i] = (LAYER*)malloc(sizeof(LAYER));
 	} 

 	printf("Begin init Layer for processing\n");
 	for(int i=0; i<NB_LAYOUT;i++){
 		printf("Layer %d is initializing\n",i );
 		init_layer(tab_layer[i], matrix,0, i);
 		printf("Layer %d initialized\n",i );

 	}
 	int j=0;
 	printf("End init Layer for processing\n");

 	for (int i = 0; i < 1; ++i)
 	{
 		for(int i=0; i<raw_matrix;i++){
 			printf("%d\n",i );
 			out = choose_output(out_process,i);
 			//init_layer(tab_layer[0], matrix, i,0);
 			init_value(tab_layer[0],matrix,i);
 			double error = 10.0;
 			rnnsetstart(tab_layer);

 			while(error > 0.05 && j<100){
 				rnnset(tab_layer, out);
 				//ajustError(tab_layer[NB_LAYOUT-1]);
 				rnnlearn(tab_layer,learn);
 				// tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
 				// show_result(tab_result, layer_size[NB_LAYOUT-1]);


 				error = geterror(tab_layer[NB_LAYOUT-1],out);
 				j++;
 			}
 			jtot += j;
 			printf("j : %d\n",j );
 			j=0;


 			if (DEBUG)
 				printf("Error %f\n", error);

 			tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
 			compte_resultat(tab_result, out_compt, out, tab_layer[NB_LAYOUT-1]->nbnode, &average_sure);

 			if (DEBUG)
 				show_result(tab_result, layer_size[NB_LAYOUT-1]);

 			//free_layer(tab_layer[0]);
 			free(out);
 			free(tab_result);
 		}
 	}

 	

 	printf("Finish learning\n");
 	end = getCurrentTimestamp() ;
 	printf("Temps mis a l'éxécutionn : %fd\n", end-start );
 	printf("average sure : %f tour de boucle pour training moyen : %f \n", average_sure/raw_matrix, jtot/raw_matrix );
 	postprocessing(out_compt);
 	printf("overload FPGA for learn= %f\n",overload );
 	free(matrix);

 	float* matrix_test = preprocessing(file_name_test,1);
 	start = getCurrentTimestamp();
 	col_matrix = get_col_matrix();
 	raw_matrix = get_raw_matrix();
 	printf("raw_matrix : %d\n", raw_matrix );
 	printf("col matrix : %d \n", col_matrix);
 	out_process = get_output();

 	//#pragma omp for
 	for (int i = 0; i < nb_error; i++)
 		out_compt[i] = 0;

 	printf("End init Layer for processing\n");
 	for(int i=0; i<raw_matrix;i++){
 		out = choose_output(out_process,i);
 		double error= 10.0;
 		init_value(tab_layer[0],matrix_test,i);


 		rnnsetstart(tab_layer);
 		rnnset(tab_layer,out);
 		ajustError(tab_layer[NB_LAYOUT-1]);

 		//rnnlearn(tab_layer,learn);
 		error = geterror(tab_layer[NB_LAYOUT-1], out);


 		if (DEBUG)
 			printf("Error %f\n", error);

 		tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
 		compte_resultat(tab_result, out_compt,out, tab_layer[NB_LAYOUT-1]->nbnode, &average_sure);

 		if (DEBUG){
 			printf("Resulat  ligne : %d\n", i);
 			show_result(tab_result, layer_size[NB_LAYOUT-1]);
 		}

 		free(out);
 		free(tab_result);
 	}

 	printf("Finish Testing look into result.csv for result\n");
 	end = getCurrentTimestamp() ;
 	printf("Time to execute for Testing : %fd\n", end-start );
 	printf("overload FPGA tot = %f\n", overload);
 	postprocessing(out_compt);

 	free(out_compt);
 	free(matrix_test);

 }



