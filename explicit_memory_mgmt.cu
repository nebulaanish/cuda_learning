#include <cstdio>
#include <cstdlib>
#include <cuda/cmath>
#include <cuda_runtime.h>
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

__global__ void vectorAdd(float* A, float* B, float* C, int vectorLength){
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (workIndex < vectorLength){
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void explicitMemoryExample(int vectorLength){
    float* A  = nullptr;
    float* B  = nullptr;
    float* C = nullptr;
    float* comparisonResult  = (float*)malloc(vectorLength*sizeof(float));

    // these will use memory in device
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    // Allocate Host(cpu) memory as this buffer will be used for copies between CPU and GPU memory. 
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize the memory on host. 
    initArray(A, vectorLength);
    initArray(B, vectorLength);



    // Allocate memory on GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault); // destination_pointer, source_pointer, size_in_bytes, cudaMemcpyKind_t 
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));

    // Launch kernel in GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vectorAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    cudaDeviceSynchronize();


    // After GPU is done computing, copy results back to CPU. 
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}

int main(){
    int vectorLength = 1 << 25;
    explicitMemoryExample(vectorLength);
    return 0;
}