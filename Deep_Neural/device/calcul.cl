#pragma OPENCL EXTENSION cl_intel_channels : enable
channel double c_value;


__kernel void kvectormulmatrix(__global double* restrict A, __global double* restrict B, __global double* restrict C, __global double* restrict D, 
	const int n, const int m, const int sigmoide, const int isinside){
	
	const int i = get_global_id(0); //== m (taille du vector input);

	double tmp =0.0;
	__local double value[122]; 
	if(isinside){
		for (int j = 0; j < n; ++j)
		{
			value[j] = read_channel_intel(c_value);
		}
	}else{
		for (int j = 0; j < n; ++j)
		{
			value[j] = A[j];
		}
	}

	#pragma unroll
	for (int j = 0; j < n; ++j)
	{
    	tmp+= B[j*m+i]*(value[j]+D[j]);

	}
	if(sigmoide){
		C[i] = 1/(1+exp(-tmp));
	}else{
		C[i]= tmp;
	}

	write_channel_intel(c_value,C[i]);
}


__kernel void kchange_weight(__global double* restrict weight, __global double* restrict error_next,__global double* restrict value_next ,
						 	const double leanintegrate, const int raw, const int col, const int isoutput){
	
	int i = get_global_id(0);
	int j = get_global_id(1);
	double tmp= 0.0;

	tmp = error_next[j]*leanintegrate; 
	
	if(isoutput){
		weight[i*col+j] -=tmp;
		if(weight[i*col+j]>5){
			weight[i*col+j]=5.0;
		}else if(weight[i*col+j]<-5){
			weight[i*col+j]=-5.0;
		}
	}
	else{
		tmp*= (1-(value_next[j]*value_next[j]));
		weight[i*col+j] -=tmp;
	}
	
	
}

__kernel void ksoustraction_vector(__global double* restrict A, __global double* restrict B, __global double* restrict C, const int taille){

	int i = get_global_id(0);
	C[i] = A[i] - B[i];
}