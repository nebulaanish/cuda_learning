// Perform 1D convolution between an input signal and a kernel:

#include <cuda_runtime.h>

__global__ void conv(const float* A, const float* B, float* C, size_t N, size_t K) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    int r = K / 2; 
    float sum = 0.0f;

    for (int j = 0; j < K; j++) {
        int a_idx = index + j - r;
        if (a_idx >= 0 && a_idx < (int)N) {
            sum += A[a_idx] * B[j];
        }
    }
    C[index] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    conv<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}