#ifndef PRE_H
#define PRE_H 

	#include "CL/opencl.h"
	#include "AOCLUtils/aocl_utils.h"


	#define SIZE_PRE_PROC 100
	#define NB_ERROR 23
	#define NB_COL_NSL 41


	double* preprocessing();
	int is_present(char* element, char** tab, int n);
	void show_tab(char** tab, int n);
	void reading_file();
	void make_matrix();
	void make_vector(int i, char* element);
	void show_matrix();
	void fill_output(char* element, int k);
	int* get_output();
	int get_col_matrix();
	int get_raw_matrix();


#endif

