#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define INDX(row, col, ld) ((row)*(ld) + (col))


__global__ void smem_transpose(int m, float* a, float* c){
    // declared statically allocated shared memory
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    // determine, my row tile and col tile index
    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;

    // Read from global memory into shared memory array
    smemArray[threadIdx.x][threadIdx.y] = a[INDX(tileRow + threadIdx.y, tileCol + threadIdx.x, m)];

    __syncthreads();

    // Write result from shared memory to global memory; 
    c[INDX(tileCol + threadIdx.y, tileRow + threadIdx.x, m)] = smemArray[threadIdx.y][threadIdx.x];
    return;
}