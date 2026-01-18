#include <cuda_runtime.h>

__global__ void leaky_relu(const float* input, float alpha, float* output, size_t n, size_t m){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    size_t total = n * m;

    if (index < total){
    float x = input[index];
    output[index] = (x > 0.0f) ? x : alpha * x;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    size_t total = n * m;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    leaky_relu<<<blocks, threads>>>(input, alpha, output, n, m);
}