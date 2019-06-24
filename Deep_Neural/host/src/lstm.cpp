#include "preprocessing.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace aocl_utils;

typedef struct _layer_LSTM LSTM_LAYER;

struct _layer_LSTM {
  int typeLayer;
  int nbnode;

  double* value_f;
  double* weight_f;

  double* value_i;
  double* weight_i;

  double* value_cbar;
  double* weight_cbar;

  double* value_c;

  double* value_o;
  double* weight_o;

  double* value_x;

  double* biais_f;
  double* biais_i;
  double* biais_cbar;
  double* biais_o;

  double* value_prev_x;
  double* value_prev_f;
  double* value_prev_i;
  double* value_prev_cbar;
  double* value_prev_o;

  double* error;
  double* error_prev;
};

LSTM_LAYER* layers[NB_LAYOUT];

int layer_size_lstm[NB_LAYOUT];

void init_layer_size_lstm(int col_matrix, int nb_error){
  layer_size_lstm[0]= col_matrix;
  for (int i = 1; i <NB_LAYOUT-1; ++i){
    layer_size_lstm[i]=SIZE_HD;
  }
  layer_size_lstm[NB_LAYOUT-1]=nb_error;
}

double geterror_LSTM(LSTM_LAYER* tab_layer){
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

void ajustError_LSTM(LSTM_LAYER * layer){
    int i;
    double sum = 0.0;
    #pragma omp for
    for (i = 0; i < layer->nbnode; i++){ 
        sum += layer->value_x[i];
    }

    #pragma omp for
    for (i = 0; i < layer->nbnode; i++){ 
        layer->value_x[i] /= sum;
    }
}

double* gettab_result(LSTM_LAYER* tab_layer){
  double* res= (double*)malloc(sizeof(double)*tab_layer->nbnode);
  #pragma omp for
  for (int i = 0; i < tab_layer->nbnode; ++i)
  {
    res[i] = tab_layer->value_x[i];
  }
  return res;
}

void free_LSTM_layer(LSTM_LAYER* l){
	free(l->value_f);
 	free(l->weight_f);
	free(l->value_i);
	free(l->weight_i);
	free(l->value_cbar);
	free(l->weight_cbar);
	free(l->value_c);
	free(l->value_o);
	free(l->weight_o);
	free(l->value_x);
	free(l->biais_f);
	free(l->biais_i);
	free(l->biais_cbar);
	free(l->biais_o);
	free(l->value_prev_x);
	free(l->value_prev_f);
	free(l->value_prev_i);
	free(l->value_prev_cbar);
	free(l->value_prev_o);
	free(l->error);
	free(l->error_prev);
}


void init_LSTM_layer(LSTM_LAYER* l, double* matrix, int index, int currentlayer){

   l->typeLayer = currentlayer;

  if(currentlayer == 0){
    l->nbnode = layer_size_lstm[currentlayer];
    l->value_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);

    
    #pragma omp for
    for(int i=0; i<layer_size_lstm[currentlayer];i++){
      l->value_x[i]=matrix[index*layer_size_lstm[currentlayer]+i];
      l->value_f[i]=0.0;
      l->value_i[i]=0.0;
      l->value_cbar[i]=0.0;
      l->value_o[i]=0.0;
    }
    l->weight_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);

    
    #pragma omp for
    for (int i = 0; i < layer_size_lstm[currentlayer+1]*layer_size_lstm[currentlayer]; ++i)
    {
      l->weight_f[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_i[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_cbar[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_o[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;

    }
  }else if(currentlayer == NB_LAYOUT-1){
    l->nbnode = layer_size_lstm[currentlayer];
    l->value_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);

  }else{
    l->nbnode = layer_size_lstm[currentlayer];
    l->value_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);

    l->weight_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
    l->weight_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);

    
    #pragma omp for
    for (int i = 0; i < layer_size_lstm[currentlayer+1]*layer_size_lstm[currentlayer]; ++i)
    {
      l->weight_f[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_i[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_cbar[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->weight_o[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;

    }
  }
    l->biais_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->biais_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->biais_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->biais_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);

    
    #pragma omp for
    for (int i = 0; i < layer_size_lstm[currentlayer]; ++i)
    {
      l->biais_f[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->biais_i[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->biais_cbar[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
      l->biais_o[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;

    }
    l->value_prev_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_prev_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_prev_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_prev_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    l->value_prev_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    
    #pragma omp for
    for(int i=0; i<layer_size_lstm[currentlayer];i++){
      l->value_prev_x[i]=0.0;
      l->value_prev_f[i]=0.0;
      l->value_prev_i[i]=0.0;
      l->value_prev_cbar[i]=0.0;
      l->value_prev_o[i]=0.0;
    }
     
    l->error= (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    
    #pragma omp for
    for(int i=0; i<layer_size_lstm[currentlayer];i++){
      l->error[i]=0.0;
    }
      l->error_prev = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
    
    #pragma omp for
    for(int i=0; i<layer_size_lstm[currentlayer];i++){
      l->error_prev[i]=0.0;
    }
}
void lstm_start(){

}

void lstm_foward(){

}

void lstm_backfoward(){

}

void lstm_train(const char* file_name_learn){

    double start = getCurrentTimestamp();
    double * matrix = preprocessing(file_name_learn);
    double pre = getCurrentTimestamp();
    printf("Temps de preprocessing : %f\n", pre-start);
    int col_matrix = get_col_matrix();
    int raw_matrix = get_raw_matrix();
    int nb_error =get_nberror();
    double average_sure=0.0;

    int* out_process = get_output();
    int* out_compt = (int*)malloc(sizeof(int*)*nb_error);    
    for (int i = 0; i < nb_error; i++)
        out_compt[i] = 0;

    double* out;
    double jtot=0.0;
    double learn=0.03;
    int j=0;
	
	for (int i = 0; i < NB_LAYOUT; ++i)
	{
		layers[i] =(LSTM_LAYER*)malloc(sizeof(LSTM_LAYER));
	}
	for (int i = 1; i < NB_LAYOUT-1; ++i)
	{
		init_LSTM_layer(layers[i],matrix,0,i);
	}

	for(int i=0; i<1;i++){
      printf("%d\n",i );
      
      out = choose_output(out_process,i);
      init_LSTM_layer(layers[0],matrix,i,0);
      double error = 10.0;

      while(error > 0.05 && j<1000){
        lstm_start();
		lstm_foward();
		lstm_backfoward();

        error = geterror_LSTM(layers[NB_LAYOUT-1]);
        j++;
      }
      jtot += j;
      printf("j : %d\n",j );
      j=0;
      ajustError_LSTM(layers[NB_LAYOUT-1]);

      if (DEBUG)
        printf("Error %f\n", error);

      double *tab_result = gettab_result(layers[NB_LAYOUT-1]);
      compte_resultat(tab_result, out_compt, layers[NB_LAYOUT-1]->nbnode, &average_sure);

      if (DEBUG)
        show_result(tab_result, layer_size_lstm[NB_LAYOUT-1]);

      free_LSTM_layer(layers[0]);
      free(out);
      free(tab_result);

	}
}



