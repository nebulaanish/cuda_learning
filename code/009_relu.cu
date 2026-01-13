// https://tensara.org/problems/relu
#include <cuda_runtime.h>

__global__ void relu(const float* input, float* output, size_t M, size_t N){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < M*N){
        float val = input[index];
        output[index] = fmaxf(0.0f, val);
    }

}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int threads_per_block = 256;
    size_t size_total_elements = n * m; 
    int blocker_per_grid = (size_total_elements + threads_per_block -1)/threads_per_block;
    relu<<<blocker_per_grid, threads_per_block>>>(input, output, n, m);
}

