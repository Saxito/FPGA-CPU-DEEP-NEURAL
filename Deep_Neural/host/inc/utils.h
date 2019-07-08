#include <math.h>

#define DEBUG 1
#define KERNEL 0
#define SIZE_HD 120
#define SIZE_HD_LSTM 500
#define NB_LAYOUT 5
#define NB_LAYOUT_LSTM 2
#define NB_ERROR_MAX 5

double* choose_output(int* output_process, int i);
void compte_resultat(double* tab, int* compt, int taille, double* average);
void show_result(double* a, int size);

double sigmoid(double a);

