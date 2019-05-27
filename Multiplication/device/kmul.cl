//OPENCL this function turn on Intel FPGA accelerator  
#define block_size 32

//First function naive implementation
__kernel void kmul1(__global float* restrict A, __global float* restrict B, __global float* restrict C, const int dimension) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    float tmp =0.0f;

    for(int k=0;k<dimension; k++){
    	tmp+= A[i*dimension+k]*B[k*dimension+j];
    }
    
    C[i*dimension+j]=tmp;

}


// Second function using division of the matrix. We using a block size of 32
__kernel void kmul2(__global float* restrict A, __global float* restrict B, __global float* restrict C, const int dimension) {
   
	const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    int globalRow, globalCol;
    
    if(dimension<=block_size){
		globalRow =  row; 
   	 	globalCol =  col; 
    }else{
    	globalRow = block_size*get_group_id(0) + row; 
    	globalCol = block_size*get_group_id(1) + col; 
    }
    
 
    __local float A_local[block_size][block_size];
    __local float B_local[block_size][block_size];
 
    float tmp = 0.0f;
    int numTiles=0 ;
    
    if(dimension<block_size){
    	numTiles = 1;
    }else{
    	numTiles = dimension/block_size;
    }
    
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = block_size*t + row;
        const int tiledCol = block_size*t + col;
        B_local[col][row] = B[tiledCol*dimension + globalRow];
        A_local[col][row] = A[globalCol*dimension + tiledRow];
 
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<block_size; k++) {
            tmp += B_local[k][row] * A_local[col][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    C[globalCol*dimension + globalRow] = tmp;
    

}

