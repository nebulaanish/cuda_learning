#include <stdio.h>

// This is the "Kernel" - it runs on the GPU
__global__ void hello_from_gpu() {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    printf("Hello from GPU! Block: %d, Thread: %d\n", block_id, thread_id);
}

int main() {
    printf("Hello from CPU!\n");

    hello_from_gpu<<<2, 4>>>();

    // Capture the error status
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}