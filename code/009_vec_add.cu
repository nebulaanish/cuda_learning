// https://tensara.org/problems/vector-addition

#include <cuda_runtime.h>

__global__ void vec_add(const float* input1, const float* input2, float* output, size_t n){
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index <= n){
    output[index] = input1[index] + input2[index];
    }
}

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threads_per_block = 256;  

    size_t blocks_per_grid = (n + threads_per_block - 1)/threads_per_block;
    vec_add<<<blocks_per_grid, threads_per_block>>>(d_input1, d_input2, d_output, n);

}