//OPENCL this function turn on Intel FPGA accelerator  
#define block_size 64

//First function naive implementation
__kernel void kmul1(__global float* restrict A, __global float* restrict B, __global float* restrict C, const int dimension) {
    int i = get_global_id(0);
    int j = get_global_id(2);
    float tmp =0.0f;

    for(int k=0;k<dimension; k++){
    	tmp+= A[i*dimension+k]*B[k*dimension+j];
    }
    
    C[i*dimension+j] =tmp;

}


// Second function using division of the matrix. We using a block size of 64
__kernel void kmul2(__global float* restrict A, __global float* restrict B, __global float* restrict C, const int dimension) {
   
	const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    const int globalRow = block_size*get_group_id(0) + row; 
    const int globalCol = block_size*get_group_id(1) + col; 
 
    __local float A_local[block_size][block_size];
    __local float B_local[block_size][block_size];
 
    float tmp = 0.0f;
    
    const int numTiles = dimension/block_size;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = block_size*t + row;
        const int tiledCol = block_size*t + col;
        A_local[col][row] = A[tiledCol*dimension + globalRow];
        B_local[col][row] = B[globalCol*dimension + tiledRow];
 
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<block_size; k++) {
            tmp += A_local[k][row] * B_local[col][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    C[globalCol*dimension + globalRow] = tmp;
    

}

