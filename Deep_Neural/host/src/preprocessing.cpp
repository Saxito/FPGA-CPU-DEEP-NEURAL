#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "preprocessing.h"

#define DEBUG 0


const char* resul_name = "/home/guillaume/Documents/stage/Multiplication_Maxtrix/Deep_Neural/data/result.txt";


char** tab_protocol;
char** tab_flag;
char** tab_service;
char** tab_out;
int lenght_protocol;
int lenght_service;
int lenght_flag;
int lenght_out;

const char* Dos[11] = {"back", "land", "neptune", "pod","smurf", "teardrop", "mailbomb", "processtable", "udpstorm", "apache2", "worm"};
const char* R2L[15] = {"warezclient","ftp_write", "guess_passwd","imap", "multihop", "phf", "spy", "warezmaster", "xlock", "xsnoop", "snmpguess", 
                      "snmpgetattack", "httptunnel", "sendmail", "named"};
const char* U2R[7] = {"buffer_overflow", "loadmodule", "perl", "rootkit", "sqlattack", "xterm", "ps"};
const char* Probe[6] = {"ipsweep", "nmap", "portsweep","satan", "mscan", "saint"};
const char* Normal[1] = {"normal"};

const char* Name_ERROR[5]= {"Normal","Probe","U2R","Dos","R2L"};

int raw,col;
int nb_col_matrix;
int nb_raw_matrix;
double* matrix;
int* out;
int* out_compt;

char* getfield(char* line, int num)
{
    char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

int is_present(char* element, char** tab, int n){
  for(int i=0; i<n; i++){
    if(strlen(element)==strlen(tab[i])){
      if(!memcmp(element,tab[i],strlen(element))){
        return 1;
      }
    }
  }
  return 0;
}

void show_tab(char ** tab, int n){
  printf("\n[");
  int i=0;
  for(; i<n-1;i++){
    printf("%s;",tab[i]);
  }

  printf("%s",tab[i]);
  printf("]\n");
}

void reading_file(const char* file){
  tab_protocol = (char**)malloc(sizeof(char*)*SIZE_PRE_PROC);
  #pragma omp for
  for (int i = 0; i < SIZE_PRE_PROC; i++)
        tab_protocol[i] = (char*)malloc(SIZE_PRE_PROC * sizeof(char));
  tab_flag = (char**)malloc(sizeof(char*)*SIZE_PRE_PROC);
  #pragma omp for
  for (int i = 0; i < SIZE_PRE_PROC; i++)
        tab_flag[i] = (char*)malloc(SIZE_PRE_PROC * sizeof(char));
  tab_service= (char**)malloc(sizeof(char*)*SIZE_PRE_PROC);    
  #pragma omp for
  for (int i = 0; i < SIZE_PRE_PROC; i++)
        tab_service[i] = (char*)malloc(SIZE_PRE_PROC * sizeof(char));
  tab_out= (char**)malloc(sizeof(char*)*50);    
  #pragma omp for
  for (int i = 0; i < 50; i++)
        tab_out[i] = (char*)malloc(100 * sizeof(char));
  out_compt = (int*)malloc(sizeof(int*)*NB_ERROR_MAX);    
  #pragma omp for
  for (int i = 0; i < NB_ERROR_MAX; i++)
        out_compt[i] = 0;

  //Preprocessing on protocol;
  lenght_protocol = 0;
  lenght_flag = 0;
  lenght_service = 0;
  lenght_out = 0;
  char* element;
 
  FILE* stream = fopen(file, "r");
  
  if(stream == NULL){
    printf("Failed to open File\n");
    return;
  }

  nb_raw_matrix=0;
  char line[1024];
  while (fgets(line, 1024, stream))
  {   
      char* tmp = strdup(line);

      element = getfield(tmp, 2);
      if(!is_present(element,tab_protocol,lenght_protocol)){
        strcpy(tab_protocol[lenght_protocol],element);
        lenght_protocol++;
      }

      tmp = strdup(line);
      element = getfield(tmp, 3);
      if(!is_present(element,tab_service,lenght_service)){
        strcpy(tab_service[lenght_service],element);
        lenght_service++;
      }

      tmp = strdup(line);
      element = getfield(tmp, 4);
      if(!is_present(element,tab_flag,lenght_flag)){
        strcpy(tab_flag[lenght_flag],element);
        lenght_flag++;
      }

      tmp = strdup(line);
      element = getfield(tmp, 42);
      if(!is_present(element,tab_out,lenght_out)){
        strcpy(tab_out[lenght_out],element);
        lenght_out++;
      }

      free(tmp);
      nb_raw_matrix++;
  }
  fclose(stream);
  printf("end reading \n");
  nb_col_matrix = NB_COL_NSL+lenght_protocol+lenght_flag+lenght_service-3;

  if(DEBUG){
    show_tab(tab_protocol,lenght_protocol);
    printf("%d\n", lenght_protocol);
    show_tab(tab_service,lenght_service);
    printf("%d\n", lenght_service);
    show_tab(tab_flag,lenght_flag);
    printf("%d\n", lenght_flag );
    show_tab(tab_out, lenght_out);
    printf("%d\n", lenght_out );
  }

}
void show_matrix(){
  for (int i = 0; i < 5; ++i)
  {
    for (int j = 0; j < nb_col_matrix; ++j)
    {
      printf("%.2f",matrix[nb_raw_matrix*i+j] );
    }
    printf("\n");
  }
}

void make_vector(int i, char* element){
  if(i==1){
    for(int k=0; k<lenght_protocol;k++){
      if(!strcmp(element,tab_protocol[k])){
        matrix[raw*nb_col_matrix+col]=1.0;
      }else{
        matrix[raw*nb_col_matrix+col]=0.0;
      }
      col++;
    }
  }
  else if(i==2){
    for(int k=0; k<lenght_service;k++){
      if(!strcmp(element,tab_service[k])){
        matrix[raw*nb_col_matrix+col]=1.0;
      }else{
        matrix[raw*nb_col_matrix+col]=0.0;
      }
      col++;

    }
  }
  else if(i==3){
      for(int k=0; k<lenght_flag;k++){
      if(!strcmp(element,tab_flag[k])){
        matrix[raw*nb_col_matrix+col]=1.0;
      }else{
        matrix[raw*nb_col_matrix+col]=0.0;
      }
      col++;

    }
  }
  else if(i==41){
    fill_output(element, raw);
  }else{
    matrix[raw*nb_col_matrix+col] = tanh(atof(element));
    col++;

  }
}

void make_matrix(const char* file){
  matrix =(double*)malloc(sizeof(double)*nb_col_matrix*nb_raw_matrix);
  out=(int*)malloc(sizeof(int)*nb_raw_matrix);

  FILE* stream = fopen(file, "r");
  char* element;
  
  if(stream == NULL){
    printf("Failed to open File\n");
    return;
  }

  char line[1024];
  char* tmp;
  raw = 0;
  while (fgets(line, 1024, stream)){
    col= 0;
    for(int i=0; i < NB_COL_NSL+1; i++){
      tmp = strdup(line);
      element = getfield(tmp,i+1);
      make_vector(i, element);

    }
    free(tmp);
    raw++;
  }
  if(DEBUG)
    show_matrix();

  free(tab_protocol);
  free(tab_flag);
  free(tab_service);

}

double* preprocessing(const char* file){
  printf("Preprocessing in charge\n");
  printf("Reading file\n");
  reading_file(file);
  printf("Creation of matrix for Input Layer\n");
  make_matrix(file);
  printf("Matrix created\n");
  return matrix;
}

void postprocessing(int* out){
  FILE *stream = fopen(resul_name, "w");
  fprintf(stream, "Nom de l'attaque;nombre dans le fichier;nombre trouvé;différence;precision\n");
  for (int i = 0; i < NB_ERROR_MAX; ++i)
  {
    
    fputs(Name_ERROR[i],stream);
    fprintf(stream, " %d   ",out_compt[i] );
    fprintf(stream, " %d   ",out[i] );
    fprintf(stream, " %d   ",out_compt[i]-out[i]);
    fprintf(stream, " %f   \n",1.0-(double)((double)out_compt[i]-(double)out[i])/(double)out_compt[i]);
    printf(" %f   \n",1.0-(double)((double)out_compt[i]-(double)out[i])/(double)out_compt[i]);
  }
  fclose(stream);
}

void fill_output(char* element, int k){
  for (int i = 0; i < 15; ++i)
  {
    if(i<1){
      if(!strcmp(element,Normal[i])){
        out[k]= 0;
        out_compt[0]++;
      }
    } 
    if(i<6){
      if(!strcmp(element,Probe[i])){
        out[k]= 1;
        out_compt[1]++;
      }
    } 
    if(i<7){
      if(!strcmp(element,U2R[i])){
        out[k]= 2;
        out_compt[2]++;
      }
    } 
    if(i<11){
      if(!strcmp(element,Dos[i])){
        out[k]= 3;
        out_compt[3]++;
      }
    } 
    if(i<15){
      if(!strcmp(element,R2L[i])){
        out[k]= 4;
        out_compt[4]++;
      }
    } 
  }

}

int* get_output(){
  return out;
}

int get_col_matrix(){
  return col;
}

int get_raw_matrix(){
  return nb_raw_matrix;
}

int get_nberror(){
  return NB_ERROR_MAX;
}


