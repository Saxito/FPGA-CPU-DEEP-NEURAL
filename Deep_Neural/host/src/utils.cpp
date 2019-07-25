#include "utils.h"
#include <stdio.h>
#include <stdlib.h>


double* choose_output(int* output_process, int i){
  double* res = (double*)malloc(sizeof(double)*NB_ERROR_MAX);
  int ind = output_process[i];
  #pragma omp for
  for (int k = 0; k < NB_ERROR_MAX; ++k)
  {
    if(k==ind){
      res[k]=1.0;
    }else{
      res[k]=0.0;
    }
    printf("%f ",res[k]);
  }
  printf("\n");
  return res;
}

void compte_resultat(double* tab, int* compt, double* out, int taille, double* average_sure){
  int compteur=0;
  int result=0;
  double res=0.0;
  
  for (int i = 0; i < taille; ++i)
  {
    if(res<tab[i]){
      compteur=i;
      res=tab[i];
    }
    if(out[i]==1.0){
      result=i;
    }
  }
  *average_sure += res;
  if(compteur == result || (compteur>0 && result >0))
  compt[compteur]++;
}


void show_result(double* a, int size){
  for (int i = 0; i < size; ++i)
  {
    printf("result [%d]: %f\n", i,a[i]);
  }
}


double sigmoid(double a){
  double res= 1 / (1 + exp(-a));
  return res;
}

