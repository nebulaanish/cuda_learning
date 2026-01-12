// Macro to index 1d memory array with 2d row-major order. 
// Row Major => Fill contigous memory locations with 1st row, then second etc. 
// ld is the leading dimension which is the number of columns in the matrix.

#define INDX(row, col, ld)(((row) * (ld)) + (col))

// Naive matrix transpose
__global__ void cuda_mat_transpose(int m, float *a, float *c)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < m){
        c[INDX(col, row, m)] = a[INDX(row, col, m)];
    }
    return ;
}

