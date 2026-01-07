
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/cmath>
#include <chrono> 

void initArray(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<float>(i);
    }
}

void serialVecAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int n, float eps = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength){
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (workIndex < vectorLength){
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void unifiedMemExample(int vectorLength){
    //Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Unified memory for allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize the vectors on host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // --- GPU TIMING START ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);

    cudaEventRecord(start); // Start recording
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    cudaEventRecord(stop);  // Stop recording

    // Wait for the kernel to finish execution
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // --- GPU TIMING END ---

    auto cpu_start = std::chrono::high_resolution_clock::now();
    serialVecAdd(A, B, comparisonResult, vectorLength);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> cpu_ms = cpu_stop - cpu_start;

    // Results
    printf("Vector Length: %d elements\n", vectorLength);
    printf("GPU Time: %f ms\n", milliseconds);
    printf("CPU Time: %f ms\n", cpu_ms.count());
    printf("Speedup:  %fx\n", cpu_ms.count() / milliseconds);

    if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match \n");
    }
    else{
        printf("Unified Memory: Error - CPU and GPU answers don't match\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    free(comparisonResult);
}

int main() {
    int vectorLength = 1 << 28;  // Increased to ~16M for better comparison
    unifiedMemExample(vectorLength);
    return 0;
}