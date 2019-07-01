#include "preprocessing.h"
#include "utils.h"
#include "calcul.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

typedef struct _layer_LSTM LSTM_LAYER;

struct _layer_LSTM {
	int typeLayer;
	int nbnode;
	int nbnode_next;

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
	double* value_prev_c;
	double* value_prev_o;

	double* value_out;

	double* error;
	double* error_prev;
};

LSTM_LAYER* layers[NB_LAYOUT_LSTM];

int jmax=0;


int layer_size_lstm[NB_LAYOUT_LSTM+1];

void init_layer_size_lstm(int col_matrix, int nb_error){
	layer_size_lstm[0]= col_matrix;
	for (int i = 1; i <NB_LAYOUT_LSTM; ++i){
		layer_size_lstm[i]=SIZE_HD;
	}
	layer_size_lstm[NB_LAYOUT_LSTM]=nb_error;
}

double geterror_LSTM(LSTM_LAYER* tab_layer){
	double res =0.0;
	int nb=0;

  #pragma omp for
	for (int i = 0; i < tab_layer->nbnode_next; ++i)
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
	for (i = 0; i < layer->nbnode_next; i++){
		sum += layer->value_out[i];
	}

    #pragma omp for
	for (i = 0; i < layer->nbnode_next; i++){ 
		layer->value_out[i] /= sum;
	}
}

double* gettab_result(LSTM_LAYER* tab_layer){
	double* res= (double*)malloc(sizeof(double)*tab_layer->nbnode_next);
  	#pragma omp for
	for (int i = 0; i < tab_layer->nbnode_next; ++i)
	{
		res[i] = tab_layer->value_out[i];
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
	free(l->value_out);
	free(l->value_prev_x);
	free(l->value_prev_f);
	free(l->value_prev_i);
	free(l->value_prev_c);
	free(l->value_prev_o); 
}

void init_LSTM_layer(LSTM_LAYER* l, float* matrix, int index, int currentlayer){

	l->typeLayer = currentlayer;
	l->nbnode = layer_size_lstm[currentlayer];
	l->nbnode_next = layer_size_lstm[currentlayer+1];
	l->value_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);  
	l->value_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);  
	l->value_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);  
	l->value_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);  
	l->value_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]); 
	l->value_out = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->value_c = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);  



    #pragma omp for
	for(int i=0; i<layer_size_lstm[currentlayer];i++){
		l->value_x[i]=matrix[index*layer_size_lstm[currentlayer]+i];
	}
	for(int i=0; i<layer_size_lstm[currentlayer+1];i++){
		l->value_f[i]=0.0;
		l->value_i[i]=0.0;
		l->value_cbar[i]=0.0;
		l->value_o[i]=0.0;
		l->value_out[i]=0.0;
		l->value_c[i]=0.0;
	}
	l->weight_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
	l->weight_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
	l->weight_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);
	l->weight_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]*layer_size_lstm[currentlayer+1]);


    	#pragma omp for
	for (int i = 0; i < layer_size_lstm[currentlayer+1]*layer_size_lstm[currentlayer]; ++i)
	{
		l->weight_f[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
			//printf("%f\n", l->weight_f[i]);
		l->weight_i[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
		l->weight_cbar[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
		l->weight_o[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;

	}

	l->biais_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->biais_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->biais_cbar = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->biais_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);


    #pragma omp for
	for (int i = 0; i < layer_size_lstm[currentlayer+1]; ++i)
	{
		l->biais_f[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
		l->biais_i[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
		l->biais_cbar[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
		l->biais_o[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;

	}
	l->value_prev_x = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);
	l->value_prev_f = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->value_prev_i = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->value_prev_c = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);
	l->value_prev_o = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);

    #pragma omp for
	for(int i=0; i<layer_size_lstm[currentlayer];i++){
		l->value_prev_x[i]=0.0;
	}
	for(int i=0; i<layer_size_lstm[currentlayer+1];i++){
		l->value_prev_f[i]=0.0;
		l->value_prev_i[i]=0.0;
		l->value_prev_c[i]=0.0;
		l->value_prev_o[i]=0.0;

	}
	
	l->error= (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer+1]);

    #pragma omp for
	for(int i=0; i<layer_size_lstm[currentlayer+1];i++){
		l->error[i]=0.0;
	}

	l->error_prev = (double*)malloc(sizeof(double)*layer_size_lstm[currentlayer]);

	 #pragma omp for
	for(int i=0; i<layer_size_lstm[currentlayer];i++){
		l->error_prev[i]=0.0;
	}
}

void lstm_start(){
	for (int i = 0; i < NB_LAYOUT_LSTM; ++i)
	{
		for(int k =0; k<layers[i]->nbnode ;k++){
			layers[i]->value_prev_x[k] = (layers[i]->value_x[k]);
		}
		for(int k =0; k<layers[i]->nbnode_next ;k++){
			layers[i]->value_prev_f[k] = (layers[i]->value_f[k]);
			layers[i]->value_prev_i[k] = (layers[i]->value_i[k]);
			layers[i]->value_prev_c[k] = sigmoid(layers[i]->value_c[k]);
			layers[i]->value_prev_o[k] = (layers[i]->value_o[k]);
		}
		
	}
}


void foward_cbar(LSTM_LAYER* l){
	// cbar = tanh(Wcbar * value_x + Ua * value_prev o + Biaiscbar);
	for (int i = 0; i < l->nbnode_next; ++i)
	{	
		double tmp =0.0;
		for (int j = 0; j < l->nbnode; ++j)
		{	
			tmp+=(l->value_prev_o[i]+l->value_x[j])*l->weight_cbar[j*l->nbnode_next+i];
		}
		l->value_cbar[i]=tanh(tmp+l->biais_cbar[i]);
	}
}

void foward_i(LSTM_LAYER* l){
	// i = sigmoide(Wi* value_x + Ui*value_prev o + Biais_i)
	for (int i = 0; i < l->nbnode_next; ++i)
	{	
		double tmp =0.0;
		for (int j = 0; j < l->nbnode; ++j)
		{	
			tmp+=(l->value_prev_o[i]+l->value_x[j])*l->weight_i[j*l->nbnode_next+i];
		}
		l->value_i[i]=sigmoid(tmp+l->biais_i[i]);
	}
}

void foward_f(LSTM_LAYER* l){
	// Ft = sigmoide(Wf * value_x + Uf*Value prev o + BiaisF)
	for (int i = 0; i < l->nbnode_next; ++i)
	{	
		double tmp =0.0;
		for (int j = 0; j < l->nbnode; ++j)
		{	
			tmp+=(l->value_prev_o[i]+l->value_x[j])*l->weight_f[j*l->nbnode_next+i];
		}

		//printf(" tmp f %f \n",tmp );
		l->value_f[i]=sigmoid(tmp+l->biais_f[i]);
	}
}

void foward_o(LSTM_LAYER* l){
	// o = sigmoid(Weighto * value_x + Uo* value prev o * BiaisO)
	for (int i = 0; i < l->nbnode_next; ++i)
	{	
		double tmp =0.0;
		for (int j = 0; j < l->nbnode; ++j)
		{	
			tmp+=(l->value_prev_o[i]+l->value_x[j])*l->weight_o[j*l->nbnode_next+i];
			// if (jmax==999&& i==0)
			// {
			// 	printf(" value prev o[%d] = %f\n",j,l->value_prev_o[j]);
			// 	printf(" value x[%d] = %f\n",j,l->value_x[j]);
			// 	printf(" weight o[%d] = %f\n",i,l->weight_o[j*l->nbnode_next+i]);	
			// }
		}
		// if(jmax == 999 && i==0){
		// 	printf("value sig [%d] = %f \n", i,sigmoid(tmp+l->biais_o[i]));
		// }
		l->value_o[i]=sigmoid(tmp+l->biais_o[i]);
	}	
}

void foward_state(LSTM_LAYER* l){
	// value c = i rond cbar + Ft rond ct-1
	for (int i = 0; i < l->nbnode_next; ++i)
	{
		l->value_c[i]= l->value_cbar[i] * l->value_i[i] + l->value_f[i] * l->value_prev_c[i];
		//l->value_c[i]= l->value_cbar[i] * (1-l->value_f[i] ) + l->value_f[i] * l->value_prev_c[i];
		
			// printf(" value c[%d] = %f\n",i,l->value_c[i]  );
			// printf(" value cbar[%d] = %f\n",i,l->value_cbar[i]  );
			// printf(" value i[%d] = %f\n",i,l->value_i[i]  );
			// printf(" value f[%d] = %f\n",i,l->value_f[i]  );
			// printf(" value prev c [%d] = %f\n",i,l->value_prev_c[i]  );
		
	}
}

void foward_output(LSTM_LAYER* l){
	//outpout = tanh(valuec)* o
	for (int i = 0; i < l->nbnode_next; ++i)
	{
		l->value_out[i]= sigmoid(l->value_c[i])* l->value_o[i];
			// printf(" value[o]: %f\n", l->value_o[i]);
			// printf("sig c %f \n", sigmoid(l->value_c[i]));
			// printf("out %f \n", l->value_out[i]);

		
	}
}

void fill_out(LSTM_LAYER* l_prev, LSTM_LAYER* l){
	//printf("fill value node :%d\n", l_prev->nbnode_next);
	for (int i = 0; i < l_prev->nbnode_next; ++i)
	{	
		l->value_out[i] = l_prev->value_out[i];
		//printf(" value : %f\n", l_prev->value_out[i]);
	}
}

void fill_value(LSTM_LAYER* l_prev, LSTM_LAYER* l){
	for (int i = 0; i < l_prev->nbnode_next; ++i)
	{	
		l->value_x[i] = l_prev->value_out[i];
		//l->value_prev_c[i] = l_prev->value_c[i];
	}
}

void lstm_foward(){
	for (int i = 0; i < NB_LAYOUT_LSTM; ++i)
	{
		foward_cbar(layers[i]);
		foward_i(layers[i]);
		foward_f(layers[i]);
		foward_o(layers[i]);
		foward_state(layers[i]);
		foward_output(layers[i]);
		if(i<NB_LAYOUT_LSTM-1){
			fill_value(layers[i],layers[i+1]);
		}
		//printf("layout %d\n", i);

	}
	// i = sigmoide(Wi* value_x + Ui*value_prev o + Biaisi)
	// Ft = sigmoide(Wf * value_x + Uf*Value prev o + BiaisF)
	// o = sigmoid(Weighto * value_x + Uo* value prev o * BiaisO)
	// value c = i rond cbar + Ft rond o
	//outpout = tanh(valuec)* o

}
double* calcul_error(double* out, LSTM_LAYER* l, LSTM_LAYER* l_next){
	//Δoutput = Δerreur + Δoutput+1
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next; ++i)
	{	
		if(l->typeLayer >= NB_LAYOUT_LSTM-1){
			l->error[i] = l->value_out[i]-out[i];
			//printf("value_out[%d] = %f\n", i, l->value_out[i]);
			//printf("out[%d] = %f\n", i, out[i]);
		}else{
			l->error[i]+=l_next->value_prev_x[i];
		}
		res[i]= l->error[i];
		//printf("error[%d] = %f\n",i, res[i] );

	}
	return res;
}


void calcul_error_next(LSTM_LAYER* l){
	for (int i = 0; i < l->nbnode ; ++i)
	{
		for (int j = 0; j < l->nbnode_next; ++j)
		{
			l->value_prev_x[i]+= l->weight_o[i*l->nbnode_next+j]*l->error[j];
		}
	}
}
double* cdelta_state(LSTM_LAYER* l, double* error){
	// Δc = Δoutpout * o * (1-tanh²(c)) +  state t+1 * ft+1
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next ; ++i)
	{
		res[i] = error[i] *0.02*  (1-tanh(l->value_c[i])*tanh(l->value_c[i])); // state t+1* ft+1???

	}
	return res;
}

double* cdelta_cbar(double* delta_state, LSTM_LAYER* l){
	// Δcbar = Δc * i * (1 - c²)
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next ; ++i)
	{
		res[i] = delta_state[i] * l->value_i[i] * (1-l->value_cbar[i]*l->value_cbar[i]);
	}
	return res;
}

double* cdelta_i(double* delta_state, double* delta_cbar, LSTM_LAYER* l){
	// Δi = Δc * Δcbar* i *(1-it)
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next ; ++i)
	{
		res[i] = delta_state[i] * l->value_cbar[i]* l->value_i[i] * (1-l->value_i[i]) ;

	}
	return res;
}

double* cdelta_f(double* delta_state, LSTM_LAYER* l){
	// Δf = Δc * c-1* f * (1-f)
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next ; ++i)
	{
		res[i] = delta_state[i] * l->value_prev_c[i] * l->value_f[i] * (1-l->value_f[i]);
	}
	return res;
}

double* cdelta_o(LSTM_LAYER* l, double* error){
	// Δo = Δoutput * tanh(c) * o* (1-o)
	double* res = (double*)malloc(sizeof(double)*l->nbnode_next);
	for (int i = 0; i < l->nbnode_next ; ++i)
	{
		res[i] = error[i] * 0.3 ;//tanh(l->value_c[i]) * l->value_o[i] * (1-l->value_o[i]);
		//res[i] = error[i] * tanh (l->value_c[i]);
		// if(jmax==999){
		// 	printf("deslta o[%d] = %f \n",i, res[i] );
		// }
	}
	return res;
}

void change_weight_inside(double* delta_state, double* delta_cbar, double* delta_i, double* delta_f, double* delta_o, LSTM_LAYER* l, double learn){
	int normalisation = 1;
	for (int i = 0; i < l->nbnode; ++i)
	{	
		for (int j = 0; j < l->nbnode_next; ++j)
		{	
			l->weight_cbar[i*l->nbnode_next+j] -= learn * delta_cbar[j] * l->value_x[i];
			if(l->weight_cbar[i*l->nbnode_next+j] > normalisation){
				l->weight_cbar[i*l->nbnode_next+j] =normalisation;
			}else if (l->weight_cbar[i*l->nbnode_next+j] <-normalisation){
				l->weight_cbar[i*l->nbnode_next+j] =-normalisation;
			}
			//printf(" change : %f\n", learn * delta_cbar[j] * l->value_x[i]);
			l->weight_i[i*l->nbnode_next+j] -= learn * delta_i[j]* l->value_x[i];
			if(l->weight_i[i*l->nbnode_next+j] > normalisation){
				l->weight_i[i*l->nbnode_next+j] =normalisation;
			}else if (l->weight_i[i*l->nbnode_next+j] <-normalisation){
				l->weight_i[i*l->nbnode_next+j] =-normalisation;
			}

			l->weight_f[i*l->nbnode_next+j] -= learn * delta_f[j]* l->value_x[i];
			if(l->weight_f[i*l->nbnode_next+j] > normalisation){
				l->weight_f[i*l->nbnode_next+j] =normalisation;
			}else if (l->weight_f[i*l->nbnode_next+j] <-normalisation){
				l->weight_f[i*l->nbnode_next+j] =-normalisation;
			}

			l->weight_o[i*l->nbnode_next+j] -= learn * delta_o[j]* l->value_x[i];

			if(l->weight_o[i*l->nbnode_next+j] > normalisation){
				l->weight_o[i*l->nbnode_next+j] =normalisation;
			}else if (l->weight_o[i*l->nbnode_next+j] <-normalisation){
				l->weight_o[i*l->nbnode_next+j] =-normalisation;
			}
		}
	}
}

void change_biais(double* delta_state, double* delta_cbar, double* delta_i, double* delta_f, double* delta_o, LSTM_LAYER* l, double learn){
	int normalisation =1;
	for (int i = 0; i < l->nbnode_next; ++i)
	{
		l->biais_f[i] -= delta_f[i];
		if(l->biais_f[i]>normalisation){
			l->biais_f[i]=normalisation;
		}else if (l->biais_f[i]<-normalisation){
			l->biais_f[i]= -normalisation;
		}
		
		l->biais_i[i] -=  delta_i[i];
		if(l->biais_i[i]>normalisation){
			l->biais_i[i]=normalisation;
		}else if (l->biais_i[i]<-normalisation){
			l->biais_i[i]= -normalisation;
		}
		
		l->biais_o[i] -=  delta_o[i];
		if(l->biais_o[i]>normalisation){
			l->biais_o[i]=normalisation;
		}else if (l->biais_o[i]<-normalisation){
			l->biais_o[i]= -normalisation;
		}
		
		l->biais_cbar[i] -= delta_cbar[i];
		if(l->biais_cbar[i]>normalisation){
			l->biais_cbar[i]=normalisation;
		}else if (l->biais_cbar[i]<-normalisation){
			l->biais_cbar[i]= -normalisation;
		}	
	}

}

void lstm_backfoward(double learn, double* out){
	// for (int i = NB_LAYOUT_LSTM-1; i>=0; i--)
	// {

	//
	for (int i = NB_LAYOUT_LSTM-1; i >=0; i--)
	{	
		double* error;
		if(i==NB_LAYOUT_LSTM-1){
			error = calcul_error(out,layers[i],NULL);

		}
		else{
			error = calcul_error(out,layers[i],layers[i+1]);

		}

		double* delta_state = cdelta_state(layers[i], error);
		//printf("ok2\n");

		double* delta_cbar = cdelta_cbar(delta_state,layers[i]);
		//printf("ok3\n");

		double* delta_i = cdelta_i(delta_state, delta_cbar,layers[i]);
		//printf("ok4\n");

		double* delta_f = cdelta_f(delta_state,layers[i]);
		//printf("ok5\n");

		double* delta_o = cdelta_o(layers[i], error);

		//change weight
		change_weight_inside(delta_state,delta_cbar,delta_i,delta_f,delta_o, layers[i],learn);
		change_biais(delta_state,delta_cbar,delta_i,delta_f,delta_o, layers[i],learn);

		calcul_error_next(layers[i]);


		free(error);
		free(delta_state);
		free(delta_i);
		free(delta_f);
		free(delta_o);
		free(delta_cbar);
	}
	
	
	// Δoutput = Δerreur + Δoutput+1
	// Δc = Δoutpout * o * (1-tanh²(c)) +  state t+1 * ft+1
	// Δcbar = Δc * i * (1 - c²)
	// Δi = Δc * Δcbar* i *(1-it)
	// Δf = Δc * c-1* f * (1-f)
	// Δo = Δoutput * tanh(c) * o* (1-o)
	// Δx = Wtranspose * Δgate

	//Δb = Δgate
	//ΔW = Δgate*x

}

void lstm_train(const char* file_name_learn){

	double start = getCurrentTimestamp();
	float * matrix = preprocessing(file_name_learn);
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
	double learn=1;

	init_layer_size_lstm(col_matrix,nb_error);
	
	for (int i = 0; i < NB_LAYOUT_LSTM; ++i)
	{
		layers[i] =(LSTM_LAYER*)malloc(sizeof(LSTM_LAYER));
	}
	for (int i = 1; i < NB_LAYOUT_LSTM; ++i)
	{
		init_LSTM_layer(layers[i],matrix,0,i);
	}
	int ok = raw_matrix;
	for (int b = 0; b < 1; ++b)
	{
		for(int i=0; i<ok;i++){
		printf("%d\n",i );

		out = choose_output(out_process,i);
		init_LSTM_layer(layers[0],matrix,i,0);
		double error = 10.0;
		lstm_start();

		while(error > 0.05 && jmax<1000){
			//lstm_start();
			lstm_foward();
			ajustError_LSTM(layers[NB_LAYOUT_LSTM-1]);

			//softmax(layers[NB_LAYOUT_LSTM-1]->value_out,5);

			lstm_backfoward(learn, out);
			error = geterror_LSTM(layers[NB_LAYOUT_LSTM-1]);
			//show_result(layers[NB_LAYOUT_LSTM-1]->error, 5);
			jmax++;
		}
		//ajustError_LSTM(layers[NB_LAYOUT_LSTM-1]);

		jtot += jmax;
		printf("j : %d\n",jmax);
		jmax=0;

		//ajustError_LSTM(layers[NB_LAYOUT_LSTM-1]);

		if (DEBUG)
			printf("Error %f\n", error);

		double *tab_result = gettab_result(layers[NB_LAYOUT_LSTM-1]);
		compte_resultat(tab_result, out_compt, layers[NB_LAYOUT_LSTM-1]->nbnode_next, &average_sure);

		if (DEBUG)
			show_result(tab_result, nb_error);

		free_LSTM_layer(layers[0]);
		free(out);
		free(tab_result);
	}
	}
	
	printf("Finish learning\n");
	double end = getCurrentTimestamp() ;
	printf("Temps mis a l'éxécutionn : %fd\n", end-start );
	printf("average sure : %f tour de boucle pour training moyen : %f \n", average_sure/ok, jtot/ok );
	postprocessing(out_compt);
}


void lstm_test(const char* file_name_test){

	double start = getCurrentTimestamp();
	float* matrix = preprocessing(file_name_test);
	int col_matrix = get_col_matrix();
	int raw_matrix = get_raw_matrix();
	int nb_error =get_nberror();
	double average_sure=0.0;

	int* out_process = get_output();
	int* out_compt = (int*)malloc(sizeof(int*)*nb_error);

    #pragma omp for
	for (int i = 0; i < nb_error; i++)
		out_compt[i] = 0;

	init_layer_size_lstm(col_matrix,nb_error);
	
	double* out;

	double learn=1.0;
	double error =1.0;

	printf("End init Layer for processing\n");
	for(int i=0; i<raw_matrix;i++){
		
		init_LSTM_layer(layers[0],matrix,i,0);
		
		lstm_start();

		out = choose_output(out_process,i);
        //printf("Begin of learn\n");

		lstm_foward();
		ajustError_LSTM(layers[NB_LAYOUT_LSTM-1]);

		//softmax(layers[NB_LAYOUT_LSTM-1]->value_out,5);

			// printf("foward\n");
		lstm_backfoward(learn, out);
			// printf("lstm_backfoward\n");
		error = geterror_LSTM(layers[NB_LAYOUT_LSTM-1]);
			//show_result(layers[NB_LAYOUT_LSTM-1]->error, 5);
		
		ajustError_LSTM(layers[NB_LAYOUT_LSTM-1]);
		if (DEBUG)
			printf("Error %f\n", error);

		double *tab_result = gettab_result(layers[NB_LAYOUT_LSTM-1]);
		compte_resultat(tab_result, out_compt, layers[NB_LAYOUT_LSTM-1]->nbnode_next, &average_sure);

		if (DEBUG)
			show_result(tab_result, nb_error);

		free_LSTM_layer(layers[0]);
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

