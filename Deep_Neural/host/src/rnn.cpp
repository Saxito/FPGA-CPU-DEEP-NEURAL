#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "preprocessing.h"
#include "calcul.h"
#include "utils.h"

using namespace aocl_utils;

typedef struct _layer LAYER;

int layer_size[NB_LAYOUT];
int col_matrix;
int raw_matrix;
int nb_error;
double average_sure =0.0;

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
      if(KERNEL){
        soustraction_vector_kernel(tab_layer[i]->value, out, tab_layer[i]->error, tab_layer[i]->nbnode);

      }else{
        soustraction_vector(tab_layer[i]->value, out, tab_layer[i]->error, tab_layer[i]->nbnode);
      }
    }

  }
  for (int i = NB_LAYOUT-2; i >= 0; i--){
    //multiplication de matrice  error i = error i+1 * weight i
    compute_matrix(tab_layer[i+1]->error, tab_layer[i]->weight, tab_layer[i]->error, tab_layer[i+1]->nbnode, tab_layer[i]->nbnode);
  }

  for(int i = NB_LAYOUT-2; i>=0; i--){
    if(i == NB_LAYOUT-2){
      if(KERNEL){
          change_weight_kernel(tab_layer[i]->weight, tab_layer[i+1]->biais, tab_layer[i+1]->error, tab_layer[i+1]->value, 
                                  learningrate, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode, 1);
      }else{
          change_weight_CPU(tab_layer[i]->weight, tab_layer[i+1]->biais, tab_layer[i+1]->error, tab_layer[i+1]->value, 
                             learningrate, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode,1);
      }
      

    }else{
      if(KERNEL){
        change_weight_kernel(tab_layer[i]->weight, tab_layer[i+1]->biais, tab_layer[i+1]->error, tab_layer[i+1]->value, 
                                learningrate, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode, 0);
      }else{

        change_weight_CPU(tab_layer[i]->weight, tab_layer[i+1]->biais, tab_layer[i+1]->error, tab_layer[i+1]->value, 
                              learningrate, tab_layer[i]->nbnode, tab_layer[i+1]->nbnode,0);

      }
    }
  }

  #pragma omp for
  for (int i = 0; i < NB_LAYOUT; ++i){
    for (int j = 0; j < tab_layer[i]->nbnode; ++j){
      tab_layer[i]->error_prev[j]=tab_layer[i]->error[j];
    }
  }
}

void test_KDD(const char* file_name_test){
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
      compte_resultat(tab_result, out_compt, tab_layer[NB_LAYOUT-1]->nbnode, &average_sure);

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

void learn_KDD(const char* file_name_learn){
    
    if(KERNEL){
      if(!init()){
        printf("Erreur init KERNEL\n");;
      }else{
        devices_info();
      }
    }
    
    double start = getCurrentTimestamp();
    double* matrix = preprocessing(file_name_learn);
    double pre = getCurrentTimestamp();
    printf("Temps de preprocessing : %f\n", pre-start);
    col_matrix = get_col_matrix();
    raw_matrix = get_raw_matrix();
    printf("%d\n",raw_matrix );
    nb_error =get_nberror();
    int* out_process = get_output();

    int* out_compt = (int*)malloc(sizeof(int*)*nb_error);    
    for (int i = 0; i < nb_error; i++)
        out_compt[i] = 0;

    init_layer_size();
    double* out;

    double jtot=0.0;
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
      init_layer(tab_layer[0], matrix, i,0);
      double error = 10.0;

      while(error > 0.05 && j<1000){
        rnnsetstart(tab_layer);
        rnnset(tab_layer);
        rnnlearn(tab_layer,out,learn);

        error = geterror(tab_layer[NB_LAYOUT-1]);
        j++;
      }
      jtot += j;
      printf("j : %d\n",j );
      j=0;
      ajustError(tab_layer[NB_LAYOUT-1]);

      if (DEBUG)
        printf("Error %f\n", error);

      tab_result = gettab_result(tab_layer[NB_LAYOUT-1]);
      compte_resultat(tab_result, out_compt, tab_layer[NB_LAYOUT-1]->nbnode, &average_sure);

      if (DEBUG)
        show_result(tab_result, layer_size[NB_LAYOUT-1]);

      free_layer(tab_layer[0]);
      free(out);
      free(tab_result);
    }
    printf("Finish learning\n");
    double end = getCurrentTimestamp() ;
    printf("Temps mis a l'éxécutionn : %fd\n", end-start );
    printf("average sure : %f jmoyen : %f \n", average_sure/raw_matrix, jtot/raw_matrix );
    postprocessing(out_compt);
    
    free(out_compt);
    free(matrix);
}



