__global__ void syncthreadsexample(int* input_data, int* output_data){
    __shared__ int shared_data[128];  // assuming blockDim.x is 128
    // thread indices now range from 0 to blockDim.x-1. So shared_data can hold all of it. 
    shared_data[threadIdx.x] = input_data[threadIdx.x]; 

    __syncthreads();

    if (threadIdx.x==0){
        int sum = 0;
        for (int i=0, i<blockDim.x; ++i){
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }

}