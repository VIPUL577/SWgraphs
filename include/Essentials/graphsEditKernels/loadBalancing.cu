// ##################################################################
//      TODO       |                 Requirements                   |
// ##################################################################
// - 1) launch 1 thread per vertex make a slab count array, -> iterator is there, make an array of N, do(first!=last) ++;
// - 2) prefix sum on it and load balance, -> thrust
// - 3) launch 1 thread per vertex make a slab address array-> no need directly be handled in advance kernel
// - 4) advnace and the rest of the algo continues?

#ifndef LOAD_BALANCING_CU_
#define LOAD_BALANCING_CU_
void exclusiveScanGPU(int *d_data, int N)
{
    thrust::exclusive_scan(thrust::device, d_data, d_data + N, d_data);
}
template <typename edgeHashCxt, typename VertexT>
__global__ void countSlabsKernel(edgeHashCxt *graphVertex, VertexT *currentFrontier, int *FrontierSlabs, int N)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < N)
    {
        int n = 0;
        auto node = currentFrontier[threadId];
        auto it = graphVertex[node].Begin();
        auto end = graphVertex[node].End();
        while (it != end)
        {
            n++;
            it++;
        }
        FrontierSlabs[node] = n;
    }
}
template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, false>::countSlabs(DynamicSlabGraph<EdgesObj, false>::VertexT *currentFrontier, int *FrontierSlabs, int N)
{
    typename DynamicSlabGraph<EdgesObj, false>::edgeHashCxt *vertices = this->graphVertex;
    int blocks = (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    countSlabsKernel<DynamicSlabGraph<EdgesObj, false>::edgeHashCxt, DynamicSlabGraph<EdgesObj, false>::VertexT><<<blocks, THREADSPERBLOCK>>>(vertices, currentFrontier, FrontierSlabs, N);
}
template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, true>::countSlabs(DynamicSlabGraph<EdgesObj, true>::VertexT *currentFrontier, int *FrontierSlabs, int N)
{
    typename DynamicSlabGraph<EdgesObj, true>::edgeHashCxt *vertices = this->graphVertex;
    int blocks = (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    countSlabsKernel<DynamicSlabGraph<EdgesObj, true>::edgeHashCxt, DynamicSlabGraph<EdgesObj, true>::VertexT><<<blocks, THREADSPERBLOCK>>>(vertices, currentFrontier, FrontierSlabs, N);
}
#endif
