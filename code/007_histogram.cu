#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); \
    if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); return -1; }} while (0)

__global__ void histogram_kernel(const int *input, int *hist, int array_size, int nbins)
{
    extern __shared__ int s_hist[];
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        s_hist[i] = 0;

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < array_size) {
        int val = input[idx];
        if (val >= 0 && val < nbins)
            atomicAdd(&s_hist[val], 1);
        idx += stride;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        atomicAdd(&hist[i], s_hist[i]);
}

int main()
{
    const int array_size = 1 << 20;
    const int nbins = 256;
    const int threads = 256;
    const int blocks = 256;

    int *h_input = new int[array_size];
    int *h_hist  = new int[nbins];

    for (int i = 0; i < array_size; i++)
        h_input[i] = rand() % nbins;

    int *d_input, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_input, array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist,  nbins      * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, array_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0,            nbins      * sizeof(int)));

    histogram_kernel<<<blocks, threads, nbins * sizeof(int)>>>(
        d_input, d_hist, array_size, nbins);

    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, nbins * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++)
        printf("bin[%d] = %d\n", i, h_hist[i]);


    cudaFree(d_input);
    cudaFree(d_hist);
    delete[] h_input;
    delete[] h_hist;

    return 0;
}
