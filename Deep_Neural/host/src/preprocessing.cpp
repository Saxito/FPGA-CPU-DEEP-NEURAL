#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "preprocessing.h"

#define DEBUG 1


const char* resul_name = "./data/result.txt";


const char* tab_protocol[3] = {"tcp","udp","icmp"};
const char* tab_flag[11] ={"SF","S0","REJ","RSTR","SH","RSTO","S1","RSTOS0","S3","S2","OTH"};
const char* tab_service[70] = {"ftp_data","other","private","http","remote_job","name","netbios_ns","eco_i","mtp","telnet","finger","domain_u",
                        "supdup","uucp_path","Z39_50","smtp","csnet_ns","uucp","netbios_dgm","urp_i","auth","domain","ftp","bgp","ldap",
                        "ecr_i","gopher","vmnet","systat","http_443","efs","whois","imap4","iso_tsap","echo","klogin","link","sunrpc","login",
                        "kshell","sql_net","time","hostnames","exec","ntp_u","discard","nntp","courier","ctf","ssh","daytime",
                        "shell","netstat","pop_3","nnsp","IRC","pop_2","printer","tim_i","pm_dump","red_i","netbios_ssn","rje","X11","urh_i",
                        "http_8001","aol","http_2784","tftp_u","harvest"};
char** tab_out;
int lenght_protocol=3;
int lenght_service=70;
int lenght_flag=11;
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
int nb_col_prev=0.0;
int nb_raw_matrix;
float* matrix;
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
  tab_out= (char**)malloc(sizeof(char*)*50);    
  #pragma omp for
  for (int i = 0; i < 50; i++)
    tab_out[i] = (char*)malloc(100 * sizeof(char));
  out_compt = (int*)malloc(sizeof(int*)*NB_ERROR_MAX);    
  #pragma omp for
  for (int i = 0; i < NB_ERROR_MAX; i++)
    out_compt[i] = 0;
  
  lenght_out=0;

  FILE* stream = fopen(file, "r");
  
  if(stream == NULL){
    printf("Failed to open File\n");
    return;
  }

  nb_raw_matrix=0;
  char line[1024];
  char* element;

  while (fgets(line, 1024, stream))
  {   
    char* tmp = strdup(line);

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
    // show_tab(tab_protocol,lenght_protocol);
    // printf("%d\n", lenght_protocol);
    // show_tab(tab_service,lenght_service);
    // printf("%d\n", lenght_service);
    // show_tab(tab_flag,lenght_flag);
    // printf("%d\n", lenght_flag );
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
    matrix[raw*nb_col_matrix+col] = (float)tanh(atof(element));
    col++;

  }
}

void make_matrix(const char* file){
  matrix =(float*)malloc(sizeof(float)*nb_col_matrix*nb_raw_matrix);
  out=(int*)malloc(sizeof(int)*nb_raw_matrix);

  FILE* stream = fopen(file, "r");
  char* element;
  
  if(stream == NULL){
    printf("Failed to open File\n");
    return;
  }

  char line[1024];
  raw = 0;
  while (fgets(line, 1024, stream)){
    col= 0;
    for(int i=0; i < NB_COL_NSL+1; i++){
      char * tmp = strdup(line);
      element = getfield(tmp,i+1);
      make_vector(i, element);
      free(tmp);
    }
    raw++;
  }
  if(DEBUG)
    show_matrix();

  fclose(stream);

}

float* preprocessing(const char* file, int istest){
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


