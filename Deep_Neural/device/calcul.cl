

__kernel void kvectormulmatrix(__global double* restrict A, __global double* restrict B, __global double* restrict C, __global double* restrict D, 
	const int n, const int m){
	
	const int i = get_global_id(0); //== m (taille du vector input);
	double tmp =0.0;
	
	#pragma omp for
	for (int j = 0; j < n; ++j)
	{
    	tmp+= B[j*m+i]*(A[j]+D[j]);

	}
    C[i]+=tmp;
}

__kernel void ksigmoide( __global double* restrict C, const int size){
	
	const int i = get_global_id(0);
	double tmp= C[i];
	C[i] = 1/(1+exp(-tmp));
}

