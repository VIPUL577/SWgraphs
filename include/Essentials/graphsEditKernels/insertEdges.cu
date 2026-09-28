#ifndef INSERT_EDGES_CU_
#define INSERT_EDGES_CU_


// #######################INSERT FUNCTION############################
template <typename EdgesObj>
__device__ __forceinline__ void DynamicSlabGraph<EdgesObj, true>::insertEdge(bool toInsert, VertexT &src, VertexT &dst, EdgeValueT &weight, int lane, DynamicSlabGraph<EdgesObj, true>::EdgeDynAllocator &localctxt)
{
    int workQueue = 0;
    int DestinationVertexBucket = toInsert ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;
    while ((workQueue = __ballot_sync(0xFFFFFFFF, toInsert)) != 0)
    {
        int currentLane = __ffs(workQueue) - 1;
        VertexT currentSrc = __shfl_sync(0xFFFFFFFF, src, currentLane, 32);
        bool sameSrc = (src == currentSrc);
        bool green = toInsert && sameSrc;
        graphVertex[currentSrc].insertPair(green, lane, dst, weight, DestinationVertexBucket, localctxt);
        int InsertionCount = __popc(__ballot_sync(0xFFFFFFFF, green));
        if (lane == 0)
            atomicAdd(d_vertexDegree + currentSrc, InsertionCount);
        if (green)
        {
            atomicAdd(&d_EdgesPerBucket[d_BucketsPrefixSum[currentSrc] + DestinationVertexBucket], 1);
            toInsert = false;
        }
    }
}
template <typename EdgesObj>
__device__ __forceinline__ void DynamicSlabGraph<EdgesObj, false>::insertEdge(bool toInsert, VertexT &src, VertexT &dst, int lane, DynamicSlabGraph<EdgesObj, false>::EdgeDynAllocator &localctxt)
{
    int workQueue = 0;
    int DestinationVertexBucket = toInsert ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;
    while ((workQueue = __ballot_sync(0xFFFFFFFF, toInsert)) != 0)
    {
        int currentLane = __ffs(workQueue) - 1;
        VertexT currentSrc = __shfl_sync(0xFFFFFFFF, src, currentLane, 32);
        bool sameSrc = (src == currentSrc);
        bool green = toInsert && sameSrc;
        graphVertex[currentSrc].insertPair(green, lane, dst, DestinationVertexBucket, localctxt);
        int InsertionCount = __popc(__ballot_sync(0xFFFFFFFF, green));
        if (lane == 0)
            atomicAdd(d_vertexDegree + currentSrc, InsertionCount);
        if (green)
        {
            atomicAdd(&d_EdgesPerBucket[d_BucketsPrefixSum[currentSrc] + DestinationVertexBucket], 1);
            toInsert = false;
        }
    }
}
// ##################################################################


// ##########################CUDA KERNELS############################
template <typename EdgesObj>
__global__ void InsertEdgesKernel(typename DynamicSlabGraph<EdgesObj, true>::VertexT *sourceVertex, typename DynamicSlabGraph<EdgesObj, true>::VertexT *dstVertex, typename DynamicSlabGraph<EdgesObj, true>::EdgeValueT *weigths, int countN, DynamicSlabGraph<EdgesObj, true> theGraph)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadID % 32;

    if (threadID - lane >= countN)
        return;
    typename DynamicSlabGraph<EdgesObj, true>::VertexT src{}; 
    typename DynamicSlabGraph<EdgesObj, true>::VertexT dst{} ; 
    typename DynamicSlabGraph<EdgesObj, true>::EdgeValueT weight{} ;
    bool insert = false;
    if (threadID < countN)
    {
        src = sourceVertex[threadID];
        dst = dstVertex[threadID];
        weight = weigths[threadID];
        insert = (src!=dst); 
    }
    typename DynamicSlabGraph<EdgesObj, true>::EdgeDynAllocator localctxt(theGraph.GetDynCtxt()); 
    localctxt.initAllocator(threadID, lane);
    theGraph.insertEdge(insert, src , dst , weight , lane ,localctxt) ; 

}
template <typename EdgesObj>
__global__ void InsertEdgesKernel(typename DynamicSlabGraph<EdgesObj, false>::VertexT *sourceVertex, typename DynamicSlabGraph<EdgesObj, false>::VertexT *dstVertex, int countN, DynamicSlabGraph<EdgesObj, false> theGraph)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadID % 32;

    if (threadID - lane >= countN)
        return;
    typename DynamicSlabGraph<EdgesObj, false>::VertexT src{}; 
    typename DynamicSlabGraph<EdgesObj, false>::VertexT dst{} ; 
    bool insert = false;
    if (threadID < countN)
    {
        src = sourceVertex[threadID];
        dst = dstVertex[threadID];
        insert = (src!=dst); 
    }
    typename DynamicSlabGraph<EdgesObj, false>::EdgeDynAllocator localctxt(theGraph.GetDynCtxt()); 
    localctxt.initAllocator(threadID, lane);
    theGraph.insertEdge(insert, src , dst , lane ,localctxt); 

}
// ##################################################################

template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, true>::insertEdges(VertexT *sourceVertex, VertexT *dstVertex, EdgeValueT *weigths, int countN)
{
    int blocks = (countN+THREADSPERBLOCK-1)/THREADSPERBLOCK ; 
    InsertEdgesKernel<EdgesObj><<<blocks , THREADSPERBLOCK>>>(sourceVertex , dstVertex , weigths , countN , *this);  
}
template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, false>::insertEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN)
{
    int blocks = (countN+THREADSPERBLOCK-1)/THREADSPERBLOCK ; 
    InsertEdgesKernel<EdgesObj><<<blocks , THREADSPERBLOCK>>>(sourceVertex , dstVertex , countN , *this);  
}

#endif