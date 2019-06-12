#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "preprocessing.h"

using namespace aocl_utils;

#define DEBUG 1
#define SIZE_HD 50
#define NB_LAYOUT 3

int layer_size[NB_LAYOUT];
int col_matrix;
int raw_matrix;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

typedef struct _layer LAYER;

struct _layer {
  int typeLayer;
  int nbnode;
  double* value;
  double* weight;
  double* biais;
  double* value_prev;
  double* error;
  double* error_prev;
};


// Function prototypes
bool init();
void cleanup();
void create_kernel();
void init_layer(LAYER* l, double* matrix, int index, int currentlayer);

void init_layer(LAYER* l, double* matrix, int index, int currentlayer){
    l->typeLayer = currentlayer;
    //printf("%d\n", layer_size[currentlayer]);
  if(currentlayer == 0){
    l->nbnode = col_matrix;
    l->value = (double*)malloc(sizeof(double)*col_matrix);
    for(int i=0; i<col_matrix;i++){
      l->value[i]=matrix[index*col_matrix+i];
    }
    l->weight = (double*)malloc(sizeof(double)*col_matrix*layer_size[currentlayer+1]);
    for (int i = 0; i < layer_size[currentlayer+1]*col_matrix; ++i)
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
    for (int i = 0; i < layer_size[currentlayer]*layer_size[currentlayer+1]; ++i)
    {
      l->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    }
  }
    l->biais = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
    for (int i = 0; i < layer_size[currentlayer]; ++i)
    {
      l->biais[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    }
    l->value_prev = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
    for(int i=0; i<layer_size[currentlayer];i++){
      l->value_prev[i]=0.0;
    }
      l->error= (double*)malloc(sizeof(double)*layer_size[currentlayer]);
    for(int i=0; i<layer_size[currentlayer];i++){
      l->error[i]=0.0;
    }
      l->error_prev = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
    for(int i=0; i<layer_size[currentlayer];i++){
      l->error_prev[i]=0.0;
    }

}

double sigmoid(double a){
  double res= 1 / (1 + exp(-a));
  return res;
}

void rnnsetstart(LAYER* tab_layer[]){
  for (int i=0; i<NB_LAYOUT;i++){
    if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
      for(int k =0; k<tab_layer[i]->nbnode ;k++){
        tab_layer[i]->value_prev[k] = tanh(tab_layer[i]->value[k]);
      } 
    }else{
      //printf("node : %d\n", tab_layer[i]->nbnode);
      for(int k =0; k<tab_layer[i]->nbnode ;k++){
        tab_layer[i]->value_prev[k] = tab_layer[i]->value[k];
      }
    }
  }
}


void rnnlearnstart(LAYER* tab_layer[]) {
  for (int i=0; i<NB_LAYOUT;i++){
    for(int k =0; k<tab_layer[i]->nbnode ;k++){
      tab_layer[i]->value_prev[k] = 0.0;
      tab_layer[i]->error_prev[k] = 0.0;
    } 
  }
}

void compute_matrix_tan(double* a, double* b ,double* c, double* biais, int n, int m){
  int i, j;
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        //printf("i: %d j: %d \n", i,j);
        c[i] += b[j*m+i]*a[j];
        if(j==n-1){
          //printf("mres %f\n",c[i] );
        }
      }
      //c[i] += biais[i];
      c[i]= sigmoid(c[i]);
      //printf("value calculer :%f\n", c[i]);
    }
}

void softmax(double *input, int input_len) {
  assert(input);
  float m = -INFINITY;
  for (int i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (int i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}


void rnnset(LAYER* tab_layer[]){
  //printf("Begin rnn set\n");
  for (int i = 0; i < NB_LAYOUT; ++i)
  {
    if(tab_layer[i]->typeLayer!=0){
      for (int k = 0; k < tab_layer[i]->nbnode; ++k){
        tab_layer[i]->value[k]= tab_layer[i]->biais[k];
      }
      compute_matrix_tan(tab_layer[i-1]->value, tab_layer[i-1]->weight, tab_layer[i]->value, tab_layer[i-1]->biais, tab_layer[i-1]->nbnode, tab_layer[i]->nbnode);
       
      if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
        // Fonction comme softmax 
      }
    }
  }
}

void soustraction_vector(double* a, double* b, double* c, int n){
  double res = 0.0;
  for (int i = 0; i < n; ++i)
  {
    c[i] = a[i]-b[i]; //+=
    //printf("a[i] %f \n",a[i]);
    //printf("b[i] %f \n",b[i]);
    if(c[i]<0){
      res -=c[i];
    }else{
      res+=c[i];
    }
   //printf("c[i] %f\n", c[i]);
  }
  //printf("error tot = %f\n", res);

}
void compute_matrix(double* a, double* b, double *c, int n, int m){
  int i, j;
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        c[i] += b[i*n+j]*a[j];
      }
    }
}

void vector_multiplication_constant(double* a, double b, double* c, double taille,int behaviour){
  for (int i = 0; i < taille; ++i)
  { 
    //printf("error %f\n",a[i]);
    if(behaviour){
      c[i]*= a[i]*b;
    }
    else{
      c[i]=a[i]*b;
    }
  }
}
void vector_multiplication(double* a, double* b, double* c, double taille){
  for (int i = 0; i < taille; ++i)
  {
    c[i]=a[i]*b[i];
  }
}

void change_weight(double* weight, double* wchange, int ligne, int colonne){
  for (int i = 0; i < ligne; ++i)
  {
    for (int j = 0; j < colonne; ++j)
    {
      //printf("%f //", weight[i*colonne+j]);
      weight[i*colonne+j]-=wchange[j];
      //printf("%f ", weight[i*colonne+j]);

    }
    //printf("\n");

  }
}

void normalisation(double* a, int intervalle, int taille){
  for (int i = 0; i < taille; ++i)
  {
    if(a[i]>intervalle){
      //printf("normalisation\n");
      a[i]=intervalle;
    }else if(a[i]<-intervalle){
      //printf("normalisation\n");
      a[i]=-intervalle;
    }
  }
}

void change_biais(double* biais, int taille, double* derivative, int behaviour){
  for (int i = 0; i < taille; ++i)
  { 
    if(behaviour){
      biais[i]-=derivative[i];
    }else{
      biais[i]=derivative[i];
    }
  }
}

void vector_multiplication_carre(double* value, double* derivative, int taille){
  for (int i = 0; i < taille; ++i)
  {
    derivative[i] = 1.0 - (value[i]*value[i]);
    //printf("derivative[i] : %f\n", derivative[i]);
  }
}


void rnnlearn(LAYER* tab_layer[], double* out, double learningrate){
  // Initialize error to zero for the output layer:
  // Compute the error for output neurons, and 
  // initialize it to 0 for the other neurons:
  // printf("init learn\n");
  for (int i = 0; i < NB_LAYOUT; ++i){
    for (int j = 0; j < tab_layer[i]->nbnode; ++j){
      tab_layer[i]->error[j]=0.0;
    }
    if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
      //soustraction de vecteur avec out qui est le résultat;
      // printf("begin soustraction\n");
      soustraction_vector(tab_layer[i]->value, out, tab_layer[i]->error, tab_layer[i]->nbnode);
    }

  }
  for (int i = NB_LAYOUT-2; i >= 0; i--){
    //multiplication de matrice  error i = error i+1 * weight i
    // printf("begin compute matrix\n");
    compute_matrix(tab_layer[i+1]->error, tab_layer[i]->weight, tab_layer[i]->error, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
  }
  double* derivative;
  double* wchange;
  // printf("begin for\n");
  for(int i = NB_LAYOUT-2; i>=0; i--){
    //printf("%d\n",tab_layer[i+1]->nbnode );
    //printf("begin wchange derivative\n");
    //printf("%d\n",tab_layer[i+1]->nbnode);
    derivative = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode);
    // printf("end derivative\n");
    wchange = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode);
    // printf("end wchange\n");

    for (int j = 0; j < tab_layer[i+1]->nbnode; ++j)
    {
      derivative[j]= 0.0;
      wchange[j]=0.0;
    }
    // printf("end wchange\n");
    if(i == NB_LAYOUT-2){
      //derivative = vector multiplication error i+1 * learnintegrate;
      vector_multiplication_constant(tab_layer[i+1]->error, learningrate, derivative, tab_layer[i+1]->nbnode, 0);
      //wchange=derivative;
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l)
      {
        wchange[l]=derivative[l];
      }
      //wchange *= value i+1
      vector_multiplication(wchange, tab_layer[i+1]->value, wchange, tab_layer[i+1]->nbnode);
      //weight i -= wchange 
      change_weight(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      //normalisation entre -5 et 5
      normalisation(tab_layer[i]->weight, 5, tab_layer[i+1]->nbnode*tab_layer[i]->nbnode);
      //biais i+1 -= derivative;
      change_biais(tab_layer[i+1]->biais, tab_layer[i+1]->nbnode, derivative,1);
      //normalisation du biais entre -5 et 5
      normalisation(tab_layer[i+1]->biais, 5,tab_layer[i+1]->nbnode);
    }else{
      //derivative = 1.0 - (value i+1)²
      vector_multiplication_carre(tab_layer[i+1]->value, derivative, tab_layer[i+1]->nbnode);
      //derivative *= error i+1* learningrate
      vector_multiplication_constant(tab_layer[i+1]->error,learningrate,derivative,tab_layer[i+1]->nbnode, 1);
      // printf("end constant\n");
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l)
      {
        wchange[l]=derivative[l];
      }
      //wchange *= value i+1
      vector_multiplication(wchange, tab_layer[i+1]->value, wchange, tab_layer[i+1]->nbnode);
      // printf("end mult\n");
      //weight i -= wchange
      change_weight(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      
      // printf("end changew\n");
      //biais i -= derivate 
      change_biais(tab_layer[i+1]->biais, tab_layer[i+1]->nbnode, derivative,1);
      // printf("end changeb\n");
    }
    free(derivative);
    free(wchange);

  }

  for (int i = 0; i < NB_LAYOUT; ++i){
    for (int j = 0; j < tab_layer[i]->nbnode; ++j){
      tab_layer[i]->error_prev[j]=tab_layer[i]->error[j];
    }
  }
}

double geterror(LAYER* tab_layer){
  double res =0.0;
  int nb=0;
  for (int i = 0; i < tab_layer->nbnode; ++i)
  { 
    if(tab_layer->error[i]<0){
      res -= tab_layer->error[i];
    }
    else{
      res += tab_layer->error[i];
    }
    nb++;
  }
  return res/nb;
}

double* gettab_result(LAYER* tab_layer){
  double* res= (double*)malloc(sizeof(double)*tab_layer->nbnode);
  for (int i = 0; i < tab_layer->nbnode; ++i)
  {
    res[i] = tab_layer->value[i];
  }
  return res;
}

void show_result(double* a, int size){
  for (int i = 0; i < size; ++i)
  {
    printf("result [%d]: %f\n", i,a[i]);
  }
}

double* choose_output(int* output_process, int i){
  double* res = (double*)malloc(sizeof(double)*NB_ERROR);
  int ind = output_process[i];
  for (int k = 0; k < NB_ERROR; ++k)
  {
    if(k==ind){
      res[k]=1.0;
    }else{
      res[k]=0.0;
    }
  }
  return res;
}

void init_layer_size(){
  layer_size[0]= col_matrix;
  for (int i = 1; i <NB_LAYOUT-1; ++i){
    layer_size[i]=SIZE_HD;
  }
  layer_size[NB_LAYOUT-1]=NB_ERROR;
}


int main(int argc, char** argv)
{   

    double* matrix = preprocessing();

    col_matrix = get_col_matrix();
    raw_matrix = get_raw_matrix();
    int* out_process = get_output();
    init_layer_size();
    double* out;


    double learn=0.03;
    double* tab_result;
    LAYER* tab_layer[NB_LAYOUT];
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
    printf("End init Layer for processing\n");
    for(int i=0; i<10;i++){
      out = choose_output(out_process,i);
      //printf("Begin of learn\n");
      init_layer(tab_layer[0], matrix, i,0);
      double error = 10.0;
      // printf("Learn Start\n");
      rnnlearnstart(tab_layer);
        // printf("Learn set start\n");
      rnnsetstart(tab_layer);
        // printf("Learn set \n");
      rnnset(tab_layer);
        // printf("Learn\n");
      rnnlearn(tab_layer,out,learn);
     
      error = geterror(tab_layer[NB_LAYOUT-1]);

      if (DEBUG)
        printf("Error %f\n", error);

      tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);

      if (DEBUG)
        show_result(tab_result, layer_size[NB_LAYOUT-1]);
    }


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
