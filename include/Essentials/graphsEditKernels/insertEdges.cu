#ifndef INSERT_EDGES_CU_
#define INSERT_EDGES_CU_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ================================================================
// WEIGHTED EDGE INSERTION
// ================================================================

template <
    typename GraphVertexPtrT,
    typename VertexT,
    typename EdgeValueT,
    typename EdgeDynContextT>
__device__ __forceinline__
void insertWeightedEdge(
    bool toInsert,
    VertexT& src,
    VertexT& dst,
    EdgeValueT& weight,
    int lane,
    GraphVertexPtrT graphVertex,
    int* d_vertexDegree,
    int* d_BucketsPrefixSum,
    int* d_EdgesPerBucket,
    EdgeDynContextT& localctxt)
{
    int workQueue = 0;

    int destinationBucket =
        toInsert ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;

    while ((workQueue = __ballot_sync(0xFFFFFFFF, toInsert)) != 0)
    {
        int currentLane = __ffs(workQueue) - 1;

        VertexT currentSrc =
            __shfl_sync(0xFFFFFFFF, src, currentLane, 32);

        bool sameSrc = (src == currentSrc);
        bool green = toInsert && sameSrc;

        // insertPair takes bool& and clears it on success.
        // Give it a copy so 'green' survives for counting.
        bool pending = green;

        graphVertex[currentSrc].insertPair(
            pending,
            lane,
            dst,
            weight,
            destinationBucket,
            localctxt);

        bool inserted = green && !pending;

        int insertionCount =
            __popc(__ballot_sync(0xFFFFFFFF, inserted));

        if (lane == 0 && insertionCount > 0)
        {
            atomicAdd(d_vertexDegree + currentSrc, insertionCount);
        }

        if (inserted)
        {
            atomicAdd(
                &d_EdgesPerBucket[
                    d_BucketsPrefixSum[currentSrc] + destinationBucket],
                1);
        }

        // Retire lanes using the ORIGINAL variables (same as reference).
        if (toInsert && sameSrc)
            toInsert = false;
    }
}

// ================================================================
// UNWEIGHTED EDGE INSERTION
// ================================================================

template <
    typename GraphVertexPtrT,
    typename VertexT,
    typename EdgeDynContextT>
__device__ __forceinline__
void insertUnweightedEdge(
    bool toInsert,
    VertexT& src,
    VertexT& dst,
    int lane,
    GraphVertexPtrT graphVertex,
    int* d_vertexDegree,
    int* d_BucketsPrefixSum,
    int* d_EdgesPerBucket,
    EdgeDynContextT& localctxt)
{
    int workQueue = 0;

    int destinationBucket =
        toInsert ? graphVertex[src].computeBucket(dst) : 0xFFFFFFFF;

    while ((workQueue = __ballot_sync(0xFFFFFFFF, toInsert)) != 0)
    {
        int currentLane = __ffs(workQueue) - 1;

        VertexT currentSrc =
            __shfl_sync(0xFFFFFFFF, src, currentLane, 32);

        bool sameSrc = (src == currentSrc);
        bool green = toInsert && sameSrc;

        // insertKey clears 'green' itself and RETURNS true only when
        // the key was newly inserted (duplicates return false).
        bool inserted = graphVertex[currentSrc].insertKey(
            green,
            lane,
            dst,
            destinationBucket,
            localctxt);

        int insertionCount =
            __popc(__ballot_sync(0xFFFFFFFF, inserted));

        if (lane == 0 && insertionCount > 0)
        {
            atomicAdd(d_vertexDegree + currentSrc, insertionCount);
        }

        if (inserted)
        {
            atomicAdd(
                &d_EdgesPerBucket[
                    d_BucketsPrefixSum[currentSrc] + destinationBucket],
                1);
        }

        // Retire lanes using the ORIGINAL variables (same as reference).
        if (toInsert && sameSrc)
            toInsert = false;
    }
}

// ================================================================
// WEIGHTED INSERTION KERNEL
// ================================================================

template <typename EdgesObj, typename GraphVertexPtrT>
__global__
void InsertEdgesKernel(
    typename DynamicSlabGraph<EdgesObj, true>::VertexT* sourceVertex,
    typename DynamicSlabGraph<EdgesObj, true>::VertexT* dstVertex,
    typename DynamicSlabGraph<EdgesObj, true>::EdgeValueT* weights,
    int countN,
    GraphVertexPtrT graphVertex,
    int* d_vertexDegree,
    int* d_BucketsPrefixSum,
    int* d_EdgesPerBucket,
    typename DynamicSlabGraph<EdgesObj, true>::EdgeDynContext graphAlloc)
{
    using Graph = DynamicSlabGraph<EdgesObj, true>;
    using VertexT = typename Graph::VertexT;
    using EdgeValueT = typename Graph::EdgeValueT;
    using EdgeDynContext = typename Graph::EdgeDynContext;

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadID & 31;

    // Entire warps return together, preserving full-mask warp operations.
    if (threadID - lane >= countN)
        return;

    VertexT src{};
    VertexT dst{};
    EdgeValueT weight{};
    bool insert = false;

    if (threadID < countN)
    {
        src = sourceVertex[threadID];
        dst = dstVertex[threadID];
        weight = weights[threadID];

        insert = (src != dst);
    }

    EdgeDynContext localctxt(graphAlloc);

    uint32_t tid = static_cast<uint32_t>(threadID);
    uint32_t laneId = static_cast<uint32_t>(lane);

    localctxt.initAllocator(tid, laneId);

    insertWeightedEdge(
        insert,
        src,
        dst,
        weight,
        lane,
        graphVertex,
        d_vertexDegree,
        d_BucketsPrefixSum,
        d_EdgesPerBucket,
        localctxt);
}

// ================================================================
// UNWEIGHTED INSERTION KERNEL
// ================================================================

template <typename EdgesObj, typename GraphVertexPtrT>
__global__
void InsertEdgesKernel(
    typename DynamicSlabGraph<EdgesObj, false>::VertexT* sourceVertex,
    typename DynamicSlabGraph<EdgesObj, false>::VertexT* dstVertex,
    int countN,
    GraphVertexPtrT graphVertex,
    int* d_vertexDegree,
    int* d_BucketsPrefixSum,
    int* d_EdgesPerBucket,
    typename DynamicSlabGraph<EdgesObj, false>::EdgeDynContext graphAlloc)
{
    using Graph = DynamicSlabGraph<EdgesObj, false>;
    using VertexT = typename Graph::VertexT;
    using EdgeDynContext = typename Graph::EdgeDynContext;

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadID & 31;

    if (threadID - lane >= countN)
        return;

    VertexT src{};
    VertexT dst{};
    bool insert = false;

    if (threadID < countN)
    {
        src = sourceVertex[threadID];
        dst = dstVertex[threadID];

        insert = (src != dst);
    }

    EdgeDynContext localctxt(graphAlloc);

    uint32_t tid = static_cast<uint32_t>(threadID);
    uint32_t laneId = static_cast<uint32_t>(lane);

    localctxt.initAllocator(tid, laneId);

    insertUnweightedEdge(
        insert,
        src,
        dst,
        lane,
        graphVertex,
        d_vertexDegree,
        d_BucketsPrefixSum,
        d_EdgesPerBucket,
        localctxt);
}

// ================================================================
// HOST METHODS
// ================================================================

template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, true>::insertEdges(
    VertexT* sourceVertex,
    VertexT* dstVertex,
    EdgeValueT* weights,
    int countN)
{
    if (countN <= 0)
        return;

    int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    InsertEdgesKernel<EdgesObj, decltype(graphVertex)>
        <<<blocks, THREADSPERBLOCK>>>(
            sourceVertex,
            dstVertex,
            weights,
            countN,
            graphVertex,
            d_vertexDegree,
            d_BucketsPrefixSum,
            d_EdgesPerBucket,
            GetDynCtxt());
}

template <typename EdgesObj>
void DynamicSlabGraph<EdgesObj, false>::insertEdges(
    VertexT* sourceVertex,
    VertexT* dstVertex,
    int countN)
{
    if (countN <= 0)
        return;

    int blocks = (countN + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    InsertEdgesKernel<EdgesObj, decltype(graphVertex)>
        <<<blocks, THREADSPERBLOCK>>>(
            sourceVertex,
            dstVertex,
            countN,
            graphVertex,
            d_vertexDegree,
            d_BucketsPrefixSum,
            d_EdgesPerBucket,
            GetDynCtxt());
}

#endif