#include <math.h>

//Fonction Normales
double sigmoid(double a);
void compute_matrix_sig(double* a, double* b ,double* c, double* d, int n, int m);
void soustraction_vector(double* a, double* b, double* c, int n);
void compute_matrix(double* a, double* b, double *c, int n, int m);
void vector_multiplication_constant(double* a, double b, double* c, double taille,int behaviour);
void vector_multiplication(double* wchange, double* value, int col, int raw);
void change_matrix(double* matrix, double* change, int ligne, int colonne);
void normalisation(double* a, int intervalle, int taille);
void vector_multiplication_carre(double* value, double* derivative, int taille);

//Fonction Kernel
bool init();
void devices_info();
void compute_matrix_sig_kernel(double* a, double* b ,double* c, double* d, int n, int m);
void change_weight_kernel();

