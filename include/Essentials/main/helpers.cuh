
#pragma once 

using distanceArray = unsigned long long int;

__device__ __host__
int getDistance(distanceArray word) {
    return static_cast<int>(word >> 32);
}

__device__ __host__
int getParent(distanceArray word) {
    return static_cast<int>(static_cast<uint32_t>(word));
}

__device__ __host__
distanceArray getWord(int distance, int parent) {
    return (static_cast<distanceArray>(static_cast<uint32_t>(distance)) << 32)
    | static_cast<uint32_t>(parent);
}

//############################################################################################
//                                  HELPER KERNELS
//############################################################################################
template <typename VertexT>
__global__ void initZero(VertexT *array, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        array[tid] = 0;
    }
}
template <typename VertexT>
__global__ void generate_frontier(VertexT *vkeep, VertexT *prefix_vkeep, int V, VertexT *unvisited_frontier)
{
    // launch V
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < V)
    {
        if (vkeep[idx])
        {
            unvisited_frontier[prefix_vkeep[idx]] = idx;
        }
    }
}
template <typename VertexT>
__global__ void printKernel(VertexT *currentFrontier, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        printf("%d : %d \n", tid, currentFrontier[tid]);
    }
}
//############################################################################################
//                                  HELPER FUNCTIONS
//############################################################################################
template <typename VertexT>
int getNumberSlabs(VertexT *prefix_sum_in, VertexT *prefix_sum_out, int N)
{
    if (N <= 0)
        return 0;
    int a1, a2;
    cudaMemcpy(&a1, &prefix_sum_in[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    thrust::exclusive_scan(thrust::device, prefix_sum_in, prefix_sum_in + N, prefix_sum_out);
    cudaMemcpy(&a2, &prefix_sum_out[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    return a1 + a2;
}
template<typename p>
int generateFrontierWrapper(p* currentFrontier, p* bitmask, p* prefixKeep, int N){
    int cf_n = getNumberSlabs<p>(bitmask, prefixKeep, N);
    int grid = (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<p><<<grid, THREADSPERBLOCK>>>(bitmask, prefixKeep, N, currentFrontier);
    initZero<p><<<grid, THREADSPERBLOCK>>>(bitmask, N);

    return cf_n; 
}
__device__ void inValidate(distanceArray &word){
    word = getWord(INT_MAX,-1); 
}
__device__ bool isValidate(distanceArray &word){
    return getDistance(word) == INT_MAX ;
}
// TEMPLATE MISSING -> FOR NOW NO ERROR AS THE TESTS ARE IN INT
__device__ int degree(int *row_offsets, int vertex, int V, int E)
{
    int sub = (vertex == V - 1) ? E : row_offsets[vertex + 1];
    return sub - row_offsets[vertex];
}
__global__ void getDegree(int *current_frontier, int *row_indices, int *degree_array, int N, int V, int E)
{ //-> number of vertex in "current fronteir" (N)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        degree_array[idx] = degree(row_indices, current_frontier[idx], V, E);
}
