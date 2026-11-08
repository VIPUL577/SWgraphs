// #ifndef DELETE_EDGES_CU_
// #define DELETE_EDGES_CU_

// // #######################delete FUNCTION############################
// template <typename EdgesObj>
// __device__ __forceinline__ void DynamicSlabGraph<EdgesObj, true>::deleteEdge(bool toDelete, VertexT &src, VertexT &dst, int lane)
// {
//     int workQueue = 0;
//     int DestinationVertexBucket = toDelete ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;
//     while ((workQueue = __ballot_sync(0xFFFFFFFF, toDelete)) != 0)
//     {
//         int currentLane = __ffs(workQueue) - 1;
//         VertexT currentSrc = __shfl_sync(0xFFFFFFFF, src, currentLane, 32);
//         bool sameSrc = (src == currentSrc);
//         bool green = toDelete && sameSrc;
//         bool status = graphVertex[currentSrc].deleteKey(green, lane, dst, DestinationVertexBucket);
//         int deletionCount = __popc(__ballot_sync(0xFFFFFFFF, status));
//         if (lane == 0)
//             atomicSub(d_vertexDegree + currentSrc, deletionCount);
//         if (status)
//             atomicSub(&d_EdgesPerBucket[d_BucketsPrefixSum[currentSrc] + DestinationVertexBucket], 1);
//         if (toDelete && sameSrc)
//             toDelete = false;
//     }
// }
// template <typename EdgesObj>
// __device__ __forceinline__ void DynamicSlabGraph<EdgesObj, false>::deleteEdge(bool toDelete, VertexT &src, VertexT &dst, int lane)
// {
//     int workQueue = 0;
//     int DestinationVertexBucket = toDelete ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;
//     while ((workQueue = __ballot_sync(0xFFFFFFFF, toDelete)) != 0)
//     {
//         int currentLane = __ffs(workQueue) - 1;
//         VertexT currentSrc = __shfl_sync(0xFFFFFFFF, src, currentLane, 32);
//         bool sameSrc = (src == currentSrc);
//         bool green = toDelete && sameSrc;
//         bool status = graphVertex[currentSrc].deleteKey(green, lane, dst, DestinationVertexBucket);
//         int deletionCount = __popc(__ballot_sync(0xFFFFFFFF, status));
//         if (lane == 0)
//             atomicSub(d_vertexDegree + currentSrc, deletionCount);
//         if (status)
//             atomicSub(&d_EdgesPerBucket[d_BucketsPrefixSum[currentSrc] + DestinationVertexBucket], 1);
//         if (toDelete && sameSrc)
//             toDelete = false;
//     }
// }
// // ##################################################################

// // ##########################CUDA KERNELS############################
// template <typename EdgesObj>
// __global__ void DeleteEdgesKernel(typename DynamicSlabGraph<EdgesObj, true>::VertexT *sourceVertex, typename DynamicSlabGraph<EdgesObj, true>::VertexT *dstVertex, int countN, DynamicSlabGraph<EdgesObj, true> &theGraph)
// {
//     int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//     int lane = threadID % 32;

//     if (threadID - lane >= countN)
//         return;
//     typename DynamicSlabGraph<EdgesObj, true>::VertexT src{};
//     typename DynamicSlabGraph<EdgesObj, true>::VertexT dst{};
//     bool todelete = false;
//     if (threadID < countN)
//     {
//         src = sourceVertex[threadID];
//         dst = dstVertex[threadID];
//         todelete = (src != dst);
//     }
//     theGraph.deleteEdge(todelete, src, dst, lane);
// }
// template <typename EdgesObj>
// __global__ void DeleteEdgesKernel(typename DynamicSlabGraph<EdgesObj, false>::VertexT *sourceVertex, typename DynamicSlabGraph<EdgesObj, false>::VertexT *dstVertex, int countN, DynamicSlabGraph<EdgesObj, false> &theGraph)
// {
//     int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//     int lane = threadID % 32;

//     if (threadID - lane >= countN)
//         return;
//     typename DynamicSlabGraph<EdgesObj, false>::VertexT src{};
//     typename DynamicSlabGraph<EdgesObj, false>::VertexT dst{};
//     bool todelete = false;
//     if (threadID < countN)
//     {
//         src = sourceVertex[threadID];
//         dst = dstVertex[threadID];
//         todelete = (src != dst);
//     }
//     theGraph.deleteEdge(todelete, src, dst, lane);
// }
// // ##################################################################

// template <typename EdgesObj>
// void DynamicSlabGraph<EdgesObj, true>::deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN)
// {
//     int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
//     DeleteEdgesKernel<EdgesObj><<<blocks, THREADSPERBLOCK>>>(sourceVertex, dstVertex, countN, *this);
// }
// template <typename EdgesObj>
// void DynamicSlabGraph<EdgesObj, false>::deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN)
// {
//     int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
//     DeleteEdgesKernel<EdgesObj><<<blocks, THREADSPERBLOCK>>>(sourceVertex, dstVertex, countN, *this);
// }

// #endif

#ifndef DELETE_EDGES_CU_
#define DELETE_EDGES_CU_

// ##########################CUDA KERNELS############################
// Takes raw device pointers only. Never pass DynamicSlabGraph by value here:
// destroying the host-side copy runs ~DynamicSlabGraph and frees the graph.
template <typename VertexT, typename EdgeHashCtx>
__global__ void DeleteEdgesKernel(VertexT *sourceVertex, VertexT *dstVertex, int countN,
                                  EdgeHashCtx *graphVertex, int *vertexDegree,
                                  int *edgesPerBucket, int *bucketsPrefixSum)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadID % 32;

    if (threadID - lane >= countN)
        return;

    VertexT src{};
    VertexT dst{};
    bool toDelete = false;
    if (threadID < countN)
    {
        src = sourceVertex[threadID];
        dst = dstVertex[threadID];
        toDelete = (src != dst);
    }

    int bucket = toDelete ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;
    int workQueue = 0;
    while ((workQueue = __ballot_sync(0xFFFFFFFF, toDelete)) != 0)
    {
        int currentLane = __ffs(workQueue) - 1;
        VertexT currentSrc = __shfl_sync(0xFFFFFFFF, src, currentLane, 32);
        bool sameSrc = (src == currentSrc);
        bool green = toDelete && sameSrc;
        bool status = graphVertex[currentSrc].deleteKey(green, lane, dst, bucket);
        int deletionCount = __popc(__ballot_sync(0xFFFFFFFF, status));
        if (lane == 0)
            atomicSub(vertexDegree + currentSrc, deletionCount);
        if (status)
            atomicSub(&edgesPerBucket[bucketsPrefixSum[currentSrc] + bucket], 1);
        if (toDelete && sameSrc)
            toDelete = false;
    }
}
// ##################################################################

template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, true>::deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN)
{
    int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    DeleteEdgesKernel<VertexT, edgeHashCxt><<<blocks, THREADSPERBLOCK>>>(sourceVertex, dstVertex, countN, graphVertex, d_vertexDegree, d_EdgesPerBucket, d_BucketsPrefixSum);
}

template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, false>::deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN)
{
    int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    DeleteEdgesKernel<VertexT, edgeHashCxt><<<blocks, THREADSPERBLOCK>>>(sourceVertex, dstVertex, countN, graphVertex, d_vertexDegree, d_EdgesPerBucket, d_BucketsPrefixSum);
}

#endif