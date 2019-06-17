#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "preprocessing.h"

using namespace aocl_utils;

#define DEBUG 0
#define SIZE_HD 120
#define NB_LAYOUT 3

int layer_size[NB_LAYOUT];
int col_matrix;
int raw_matrix;
int nb_error;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

const char* file_name_learn = "/home/guillaume/Documents/stage/Multiplication_Maxtrix/Deep_Neural/data/KDDTrain+.txt";
const char* file_name_test = "/home/guillaume/Documents/stage/Multiplication_Maxtrix/Deep_Neural/data/KDDTest+.txt";

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

LAYER* tab_layer[NB_LAYOUT];

// Function prototypes
bool init();
void cleanup();
void create_kernel();
void init_layer(LAYER* l, double* matrix, int index, int currentlayer);

void init_layer(LAYER* l, double* matrix, int index, int currentlayer){
    
    l->typeLayer = currentlayer;
    //printf("%d\n", layer_size[currentlayer]);
  if(currentlayer == 0){
    l->nbnode = layer_size[currentlayer];
    l->value = (double*)malloc(sizeof(double)*layer_size[currentlayer]);
    for(int i=0; i<layer_size[currentlayer];i++){
      l->value[i]=matrix[index*layer_size[currentlayer]+i];
      //printf("value in %f\n", l->value[i]);
    }
    l->weight = (double*)malloc(sizeof(double)*layer_size[currentlayer]*layer_size[currentlayer+1]);
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

void free_layer(LAYER* l){
  free(l->value);
  free(l->weight);
  free(l->biais);
  free(l->value_prev);
  free(l->error);
  free(l->error_prev);
}

double sigmoid(double a){
  double res= 1 / (1 + exp(-a));
  return res;
}

// void rnnsetstart(LAYER* tab_layer[]){
//   for (int i=0; i<NB_LAYOUT;i++){
//     if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
//       for(int k =0; k<tab_layer[i]->nbnode ;k++){
//         tab_layer[i]->value_prev[k] = tanh(tab_layer[i]->value[k]);
//       } 
//     }else{
//       //printf("node : %d\n", tab_layer[i]->nbnode);
//       for(int k =0; k<tab_layer[i]->nbnode ;k++){
//         if(tab_layer[i]->typeLayer == 0){
//           //printf("value in %f\n", tab_layer[i]->value[k] );
//         }
//         tab_layer[i]->value_prev[k] = tab_layer[i]->value[k];
//       }
//     }
//   }
// }


// void rnnlearnstart(LAYER* tab_layer[]) {
//   for (int i=0; i<NB_LAYOUT;i++){
//     for(int k =0; k<tab_layer[i]->nbnode ;k++){
//       tab_layer[i]->value_prev[k] = 0.0;
//       tab_layer[i]->error_prev[k] = 0.0;
//     } 
//   }
// }


void compute_matrix_tan(double* a, double* b ,double* c, double* biais, int n, int m){
  int i, j;
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        c[i] += b[j*m+i]*a[j];
      }
      c[i]= sigmoid(c[i]);
    }
}

static void softmax(double *input, int input_len)
{
    assert (input != NULL);
    assert (input_len != 0);
    int i;
    double m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    double sum = 0;
    for (i = 0; i < input_len; i++) {
        sum += expf(input[i]-m);
    }

    for (i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - m - log(sum));

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
  for (int i = 0; i < n; ++i)
  {
    c[i] = a[i]-b[i];
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

void change_weight(double* weight, double* wchange, int ligne, int colonne){
  for (int i = 0; i < ligne; ++i)
  {
    for (int j = 0; j < colonne; ++j)
    {
      weight[i*colonne+j]-=wchange[i*colonne+j];
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
  }
}


void rnnlearn(LAYER* tab_layer[], double* out, double learningrate){
  for (int i = 0; i < NB_LAYOUT; ++i){
    for (int j = 0; j < tab_layer[i]->nbnode; ++j){
      tab_layer[i]->error[j]=0.0;
    }
    if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
      //soustraction de vecteur avec out qui est le résultat;
      soustraction_vector(tab_layer[i]->value, out, tab_layer[i]->error, tab_layer[i]->nbnode);
    }

  }
  for (int i = NB_LAYOUT-2; i >= 0; i--){
    //multiplication de matrice  error i = error i+1 * weight i
    compute_matrix(tab_layer[i+1]->error, tab_layer[i]->weight, tab_layer[i]->error, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
  }
  double* derivative;
  double* wchange;
  // printf("begin for\n");
  for(int i = NB_LAYOUT-2; i>=0; i--){
    derivative = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode);
    // printf("end derivative\n");
    wchange = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode*tab_layer[i]->nbnode);

    // printf("end wchange\n");
    for (int j = 0; j < tab_layer[i+1]->nbnode; ++j)
    {
      derivative[j]= 0.0;
      for (int l = 0; l < tab_layer[i]->nbnode; ++l)
      {
        wchange[l*tab_layer[i+1]->nbnode+j]=0.0;
      }
    }
    // printf("end wchange\n");
    if(i == NB_LAYOUT-2){
      //derivative = vector multiplication error i+1 * learnintegrate;
      vector_multiplication_constant(tab_layer[i+1]->error, learningrate, derivative, tab_layer[i+1]->nbnode, 0);
      //wchange=derivative;
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l)
      {
        for (int m = 0; m < tab_layer[i]->nbnode; ++m)
        {
          wchange[m*tab_layer[i+1]->nbnode+l]=derivative[l];
        }
        
      }
      //wchange *= value i+1 
      vector_multiplication(wchange, tab_layer[i]->value, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
      //weight i -= wchange 
      change_weight(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      //normalisation entre -5 et 5
      normalisation(tab_layer[i]->weight, 5, tab_layer[i+1]->nbnode*tab_layer[i]->nbnode);
      //biais i+1 -= derivative;
      change_biais(tab_layer[i]->biais, tab_layer[i]->nbnode, derivative,1);
      //normalisation du biais entre -5 et 5
      normalisation(tab_layer[i]->biais, 5,tab_layer[i]->nbnode);

    }else{
      //derivative = 1.0 - (value i+1)²
      vector_multiplication_carre(tab_layer[i+1]->value, derivative, tab_layer[i+1]->nbnode);
      //derivative *= error i+1* learningrate
      vector_multiplication_constant(tab_layer[i+1]->error,learningrate,derivative,tab_layer[i+1]->nbnode, 1);
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l){
        for (int m = 0; m < tab_layer[i]->nbnode; ++m){
          wchange[m*tab_layer[i+1]->nbnode+l]=derivative[l];
        } 
      }
      //wchange *= value i+1
      vector_multiplication(wchange, tab_layer[i]->value, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
      //weight i -= wchange
      change_weight(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      //biais i -= derivate 
      change_biais(tab_layer[i]->biais, tab_layer[i]->nbnode, derivative,1);

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
  nb++;
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
  double* res = (double*)malloc(sizeof(double)*nb_error);
  int ind = output_process[i];
  for (int k = 0; k < nb_error; ++k)
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
  layer_size[NB_LAYOUT-1]=nb_error;
}



void ajustError(LAYER * layer){
    int i;
    double sum = 0.0;
    for (i = 0; i < layer->nbnode; i++){ 
        sum += layer->value[i];
    }

    for (i = 0; i < layer->nbnode; i++){ 
        layer->value[i] /= sum;
    }
}

void compte_resultat(double* tab, int* compt, int taille){
  int compteur=0;
  double res=0.0;
  for (int i = 0; i < taille; ++i)
  {
    if(res<tab[i]){
      compteur=i;
      res=tab[i];
    }
  }
  compt[compteur]++;
}

void test_KDD(){
    double start = getCurrentTimestamp();

    double* matrix = preprocessing(file_name_test);
    col_matrix = get_col_matrix();
    raw_matrix = get_raw_matrix();
    nb_error =get_nberror();

    int* out_process = get_output();
    int* out_compt = (int*)malloc(sizeof(int*)*nb_error);    
    for (int i = 0; i < nb_error; i++)
        out_compt[i] = 0;
   
    init_layer_size();
    double* out;

    double learn=0.1;
    double* tab_result;
    double error =1.0;
    // LAYER* tab_layer[NB_LAYOUT];
    // for (int i = 0; i < NB_LAYOUT; ++i)
    // {
    //    tab_layer[i] = (LAYER*)malloc(sizeof(LAYER));
    // } 
    // printf("Begin init Layer for our Network \n");
    // for(int i=0; i<NB_LAYOUT;i++){
    //   init_layer(tab_layer[i], matrix,0, i);
    // }
    printf("End init Layer for processing\n");
    for(int i=0; i<raw_matrix;i++){
      out = choose_output(out_process,i);
      //printf("Begin of learn\n");
      init_layer(tab_layer[0], matrix, i,0);

      rnnset(tab_layer);
      rnnlearn(tab_layer,out,learn);

      error = geterror(tab_layer[NB_LAYOUT-1]);

      ajustError(tab_layer[NB_LAYOUT-1]);

      if (DEBUG)
        printf("Error %f\n", error);

      tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
      compte_resultat(tab_result, out_compt, tab_layer[NB_LAYOUT-1]->nbnode);

      if (DEBUG)
        show_result(tab_result, layer_size[NB_LAYOUT-1]);

      free_layer(tab_layer[0]);
      free(out);
      free(tab_result);
    }

    printf("Finish Testing look into result.csv for result\n");
    double end = getCurrentTimestamp() ;
    printf("Time to execute for Testing : %fd\n", end-start );
    postprocessing(out_compt);
}

void learn_KDD(){
   double start = getCurrentTimestamp();
    double* matrix = preprocessing(file_name_learn);
    col_matrix = get_col_matrix();
    raw_matrix = get_raw_matrix();
    nb_error =get_nberror();

    int* out_process = get_output();
    int* out_compt = (int*)malloc(sizeof(int*)*nb_error);    
    for (int i = 0; i < nb_error; i++)
        out_compt[i] = 0;

    init_layer_size();
    double* out;


    double learn=0.03;
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
    for(int i=0; i<raw_matrix;i++){
      out = choose_output(out_process,i);
      //printf("Begin of learn\n");
      init_layer(tab_layer[0], matrix, i,0);
      double error = 10.0;
      // printf("Learn Start\n");
      while(error > 0.01 && j<1000){

        rnnset(tab_layer);
        rnnlearn(tab_layer,out,learn);

        error = geterror(tab_layer[NB_LAYOUT-1]);
        j++;
      }
      j=0;

      ajustError(tab_layer[NB_LAYOUT-1]);

      if (DEBUG)
        printf("Error %f\n", error);

      tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
      compte_resultat(tab_result, out_compt, tab_layer[NB_LAYOUT-1]->nbnode);

      if (DEBUG)
        show_result(tab_result, layer_size[NB_LAYOUT-1]);

      free_layer(tab_layer[0]);
      free(out);
      free(tab_result);
    }
    printf("Finish learning\n");
    double end = getCurrentTimestamp() ;
    printf("Temps mis a l'éxécutionn : %fd\n", end-start );
    postprocessing(out_compt);
}

int main(int argc, char** argv)
{   

    learn_KDD();
    test_KDD();
    return 0;
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
