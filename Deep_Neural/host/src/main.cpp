#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "preprocessing.h"
#include "calcul.h"
#include <omp.h>


using namespace aocl_utils;

#define DEBUG 0
#define KERNEL 0
#define SIZE_HD 120
#define NB_LAYOUT 5

int layer_size[NB_LAYOUT];
int col_matrix;
int raw_matrix;
int nb_error;

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
}

void free_layer(LAYER* l){
  free(l->value);
  free(l->weight);
  free(l->biais);
  free(l->value_prev);
  free(l->error);
  free(l->error_prev);
}

void rnnsetstart(LAYER* tab_layer[]){
  for (int i=0; i<NB_LAYOUT;i++){
    if(tab_layer[i]->typeLayer == NB_LAYOUT-1){
      for(int k =0; k<tab_layer[i]->nbnode ;k++){
        tab_layer[i]->value_prev[k] = tanh(tab_layer[i]->value[k]);
      } 
    }else{
      for(int k =0; k<tab_layer[i]->nbnode ;k++){
        if(tab_layer[i]->typeLayer == 0){
        }
        tab_layer[i]->value_prev[k] = tab_layer[i]->value[k];
      }
    }
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
      if(KERNEL){
        compute_matrix_sig_kernel(tab_layer[i-1]->value, tab_layer[i-1]->weight, tab_layer[i]->value, tab_layer[i-1]->value_prev, tab_layer[i-1]->nbnode, tab_layer[i]->nbnode);     
      }
      else{
        compute_matrix_sig(tab_layer[i-1]->value, tab_layer[i-1]->weight, tab_layer[i]->value, tab_layer[i-1]->value_prev, tab_layer[i-1]->nbnode, tab_layer[i]->nbnode);
              
      }
    }
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

  for(int i = NB_LAYOUT-2; i>=0; i--){
    derivative = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode);
    wchange = (double*)malloc(sizeof(double)*tab_layer[i+1]->nbnode*tab_layer[i]->nbnode);

    #pragma omp for
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
      
      #pragma omp for
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l)
      {
        for (int m = 0; m < tab_layer[i]->nbnode; ++m)
        {
          wchange[m*tab_layer[i+1]->nbnode+l]=derivative[l];
        }
        
      }
      //wchange *= value i
      vector_multiplication(wchange, tab_layer[i]->value, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
      //weight i -= wchange 
      change_matrix(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      //normalisation entre -5 et 5
      normalisation(tab_layer[i]->weight, 5, tab_layer[i+1]->nbnode*tab_layer[i]->nbnode);
      //biais i+1 -= derivative;
      change_matrix(tab_layer[i]->biais, derivative, 1 , tab_layer[i]->nbnode);
      //normalisation du biais entre -5 et 5
      normalisation(tab_layer[i]->biais, 5,tab_layer[i]->nbnode);

    }else{
      //derivative = 1.0 - (value i+1)²
      vector_multiplication_carre(tab_layer[i+1]->value, derivative, tab_layer[i+1]->nbnode);
      //derivative *= error i+1* learningrate
      vector_multiplication_constant(tab_layer[i+1]->error,learningrate,derivative,tab_layer[i+1]->nbnode, 1);

      #pragma omp for 
      for (int l = 0; l < tab_layer[i+1]->nbnode; ++l){
        for (int m = 0; m < tab_layer[i]->nbnode; ++m){
          wchange[m*tab_layer[i+1]->nbnode+l]=derivative[l];
        } 
      }
      //wchange *= value i+1
      vector_multiplication(wchange, tab_layer[i]->value, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
      //weight i -= wchange
      change_matrix(tab_layer[i]->weight, wchange, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode);
      //biais i -= derivate 
      change_matrix(tab_layer[i]->biais, derivative, 1, tab_layer[i]->nbnode);
    }
    free(derivative);
    free(wchange);

  }

  #pragma omp for
  for (int i = 0; i < NB_LAYOUT; ++i){
    for (int j = 0; j < tab_layer[i]->nbnode; ++j){
      tab_layer[i]->error_prev[j]=tab_layer[i]->error[j];
    }
  }
}

double geterror(LAYER* tab_layer){
  double res =0.0;
  int nb=0;
  
  #pragma omp for
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
  return res/(nb+1);
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

void show_result(double* a, int size){
  for (int i = 0; i < size; ++i)
  {
    printf("result [%d]: %f\n", i,a[i]);
  }
}

double* choose_output(int* output_process, int i){
  double* res = (double*)malloc(sizeof(double)*nb_error);
  int ind = output_process[i];
  
  #pragma omp for
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
    #pragma omp for
    for (i = 0; i < layer->nbnode; i++){ 
        sum += layer->value[i];
    }

    #pragma omp for
    for (i = 0; i < layer->nbnode; i++){ 
        layer->value[i] /= sum;
    }
}

void compte_resultat(double* tab, int* compt, int taille){
  int compteur=0;
  double res=0.0;
  
  #pragma omp for
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
    #pragma omp for
    for (int i = 0; i < nb_error; i++)
        out_compt[i] = 0;
   
    init_layer_size();
    double* out;

    double learn=0.1;
    double* tab_result;
    double error =1.0;

    printf("End init Layer for processing\n");
    for(int i=0; i<raw_matrix;i++){
      out = choose_output(out_process,i);
      //printf("Begin of learn\n");
      init_layer(tab_layer[0], matrix, i,0);
      
      rnnsetstart(tab_layer);
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
    free(out_compt);
    free(matrix);
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
      printf("%d\n",i );
      out = choose_output(out_process,i);
      //printf("Begin of learn\n");
      init_layer(tab_layer[0], matrix, i,0);
      double error = 10.0;
      // printf("Learn Start\n");
      while(error > 0.01 && j<1000){
        rnnsetstart(tab_layer);
        rnnset(tab_layer);
        rnnlearn(tab_layer,out,learn);

        error = geterror(tab_layer[NB_LAYOUT-1]);
        j++;
      }
      j=0;

      //ajustError(tab_layer[NB_LAYOUT-1]);

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
    
    free(out_compt);
    free(matrix);
}

int main(int argc, char** argv)
{   
  if(KERNEL){
    if(!init()){
      printf("Erreur init KERNEL\n");;
    }else{
      devices_info();
    }
  }
  learn_KDD();
  test_KDD();
  return 0;
}


