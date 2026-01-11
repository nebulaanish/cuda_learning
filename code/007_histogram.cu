#include <cooperative_groups.h>

__global__ void clusterHistogram(int* bins, cons int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size){

    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Initialize cluster, size and calculate local bin offsets. 
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i=threadIdx.x; i< bins_per_block; i+=blockDim.x){
        smem[i] = 0; // initializes shared memory histogram to zeros. 
    }

    // cluster synchronization ensures that shared memory is intialized to zero in all
    // thread blocks in the cluster. It also ensures that all thread blocks have 
    // started executing and they exist concurrently.
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        // find the right histogram bin
        int binid = ldata;
        if (ldata<0)
            binid = 0;
        else if (ldata>=nbins)
            binid = nbins -1;

        // Find destination block rank and offset for computing DSM histogram
        int dst_block_rank = (int)(binid/bins_per_block);
        int dst_offset = binid % bins_per_block;

        // Pointer to target block shared memory
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        // Perform atomic update of the histogram bin
        atomicAdd(dst_smem + dst_offset, 1);
    }

    cluster.sync(); // Why required? research further.

    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i+= blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }

}

