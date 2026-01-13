// https://tensara.org/problems/matrix-vector

#include <cuda_runtime.h>

__global__ void mat_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ output, size_t m, size_t k){

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row > m) return;

    float sum = 0.0000000000f;
    const float* row_ptr = &a[row*k];

    for (size_t i=0; i<k; i++){
        sum += row_ptr[i] * b[i];
    }
    output[row] = sum; 
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int threads_per_block = 256;
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;

    mat_mul<<<blocks_per_grid, threads_per_block>>>(input_a, input_b, output_c, m, k);
}