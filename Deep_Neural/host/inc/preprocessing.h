#ifndef PRE_H
#define PRE_H 

	#define SIZE_PRE_PROC 100
	#define NB_ERROR_MAX 5
	#define NB_COL_NSL 41


	float* preprocessing(const char *, int);
	void make_matrix(const char*);
	void make_vector(int i, char* element);
	void show_matrix();
	void fill_output(char* element, int k);
	int* get_output();
	int get_nberror();
	int get_col_matrix();
	int get_raw_matrix();
	void postprocessing(int* out);	


#endif

