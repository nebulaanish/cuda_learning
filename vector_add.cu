
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/cmath>

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

    // Launch kernel -> unified makes sure: A, B and C are accessible to the GPU. 
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);

    // Wait for the kernel to finish execution
    cudaDeviceSynchronize();

    // Perform computation serialliy on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);


    // Confirm that both CPu and GPu got the answers
    if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match \n");
    }
    else{
        printf("Unified Memory: Error - CPU and GPU answers don't match\n");
    }

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}



int main() {
    int vectorLength = 1 << 20;  // ~1M elements
    unifiedMemExample(vectorLength);
    return 0;
}