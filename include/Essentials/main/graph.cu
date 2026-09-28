#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include "slab_hash.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#define THREADSPERBLOCK 512 

template <bool IsWeighted, typename vertexTy, typename valueTy, typename ContainerPolicyT>
struct EdgesObj
{
    using VertexT = vertexTy;
    using EdgeValueT = valueTy;
    using containerPolicy = ContainerPolicyT;

    const bool isWeighted = IsWeighted;
};
// #######################HELPER FUNCTIONS###########################
template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator ExclusiveScan(InputIterator First, InputIterator Last,
                             OutputIterator Out, T Init)
{
    return thrust::exclusive_scan(thrust::host, First, Last, Out, Init);
}
// ##################################################################
template <typename SlabAllocPolicyTy, typename CountTy = int>
using WeightedEdgesObj = EdgesObj<true,int,int,ConcurrentMapPolicy<int, int, SlabAllocPolicyTy>>;

template <typename SlabAllocPolicyTy, typename CountTy = int>
using UnweightedEdgesObj = EdgesObj<false,int,int,ConcurrentSetPolicy<int, SlabAllocPolicyTy>>;

template <typename EdgesObj, bool Weighted>
class DynamicSlabGraph;

template <typename EdgesObj>
class DynamicSlabGraph<EdgesObj, true>
{
public:
    using EdgesContainer = typename EdgesObj::ContainerPolicy;
    using Edgesalloc = typename EdgesContainer::AllocPolicyT;
    using EdgeDynAllocator = typename Edgesalloc::DynamicAllocatorT;
    using EdgeDynContext = typename Edgesalloc::AllocatorContextT;

    using VertexT = typename EdgesObj::VertexT;
    using EdgeValueT = typename EdgesObj::EdgeValueT;

    using edgeHash = typename EdgesContainer::SlabHashT;
    using edgeHashCxt = typename EdgesContainer::SlabHashContextT;
    using SlabInfoT = typename EdgesContainer::SlabInfoT;

private:
    std::vector<edgeHash> VertexHost;
    std::vector<edgeHashCxt> tempVertexHost;

    EdgeDynAllocator dynallocator;

    int *h_BucketsPerVertex;
    int *h_BucketsPrefixSum;

    uint32_t       *d_firstUpdatedSlab;
    uint8_t        *d_firstUpdatedLaneId;
    int            *d_BucketsPrefixSum;
    int            *d_EdgesPerBucket;    
    int            *d_vertexDegree;
    bool           *d_isSlablistUpdated;
    uint32_t       *d_headptr;           // buckets head pointers
    edgeHashCxt    *graphVertex;         // gpu
    EdgeDynContext graphAlloc;        // gpu

public:
    DynamicSlabGraph(uint32_t N, int *h_vertexDegree, EdgeDynAllocator &allocator, float loadFactor, uint32_t device_index)
    {
        cudaSetDevice(device_index);
        h_BucketsPerVertex = new int[N];
        h_BucketsPrefixSum = new int[N];

        dynallocator = allocator; 

        int totalBuckets = 0;
        for (int i = 0; i < N; i++)
        {
            int temp = std::ceil((float)h_vertexDegree[i] / (32 * loadFactor));
            h_BucketsPerVertex[i] = (temp == 0) ? 1 : temp;
            totalBuckets += h_BucketsPerVertex[i];
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_firstUpdatedSlab, sizeof(uint32_t) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_firstUpdatedSlab, sizeof(uint32_t) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_EdgesPerBucket, sizeof(int) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_isSlablistUpdated, sizeof(bool) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_BucketsPrefixSum, sizeof(int) * N));
        CHECK_CUDA_ERROR(cudaMalloc(&d_vertexDegree, sizeof(int) * N));
        CHECK_CUDA_ERROR(cudaMalloc(&d_headptr, 32 * sizeof(uint32_t) * totalBuckets));

        thrust::fill(thrust::device, d_firstUpdatedSlab, d_firstUpdatedSlab + totalBuckets, static_cast<uint32_t>(SlabInfoT::A_INDEX_POINTER));
        thrust::fill(thrust::device, d_firstUpdatedLaneId, d_firstUpdatedLaneId + totalBuckets, 0);
        thrust::fill(thrust::device, d_isSlablistUpdated, d_isSlablistUpdated + totalBuckets, false);

        ExclusiveScan(h_BucketsPerVertex, h_BucketsPerVertex + N, h_BucketsPrefixSum, 0);

        for (int i = 0; i < N; i++)
            VertexHost.emplace_back(reinterpret_cast<int8_t * >(d_headptr) + 128 * h_BucketsPrefixSum[i], d_firstUpdatedSlab + h_BucketsPrefixSum[i], d_firstUpdatedLaneId + h_BucketsPrefixSum[i], h_BucketsPerVertex[i], &allocator, device_index);

        auto GetContainerHashCtxt = [](auto &Container)
        {
            return Container.getSlabHashContext();
        };

        std::transform(VertexHost.begin(), VertexHost.end(), std::back_inserter(tempVertexHost), GetContainerHashCtxt);

        CHECK_CUDA_ERROR(cudaMalloc(&graphVertex, sizeof(edgeHashCxt) * N));
        CHECK_CUDA_ERROR(cudaMemcpy(graphVertex, tempVertexHost.data(), sizeof(edgeHashCxt) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_BucketsPrefixSum, h_BucketsPrefixSum, sizeof(int) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_vertexDegree, h_vertexDegree, sizeof(int) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_EdgesPerBucket,0,sizeof(int) * totalBuckets));
        graphAlloc = *dynallocator.getContextPtr();
    }
    ~DynamicSlabGraph()
    {
        delete h_BucketsPerVertex;
        delete h_BucketsPrefixSum;
        CHECK_CUDA_ERROR(cudaFree(d_firstUpdatedSlab));
        CHECK_CUDA_ERROR(cudaFree(d_firstUpdatedLaneId));
        CHECK_CUDA_ERROR(cudaFree(d_isSlablistUpdated));
        CHECK_CUDA_ERROR(cudaFree(d_BucketsPrefixSum));
        CHECK_CUDA_ERROR(cudaFree(d_vertexDegree));
        CHECK_CUDA_ERROR(cudaFree(d_EdgesPerBucket));
        CHECK_CUDA_ERROR(cudaFree(graphVertex));
        CHECK_CUDA_ERROR(cudaFree(d_headptr));
    }
    EdgeDynAllocator GetDynCtxt() {
        return graphAlloc; 
    }
    __device__ __forceinline__ void insertEdge(bool toInsert, VertexT &src, VertexT &dst, EdgeValueT &weight, int lane, DynamicSlabGraph<EdgesObj, true>::EdgeDynAllocator &localctxt); 
    __device__ __forceinline__ void deleteEdge(bool toDelete, VertexT &src, VertexT &dst, int lane); 
    void updateEdge(); 
    void insertEdges(VertexT *sourceVertex, VertexT *dstVertex, EdgeValueT *weigths, int countN); 
    void deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN); 
    void updateEdges(VertexT* sourceVertex, VertexT* dstVertex, EdgeValueT* weigths, uint32_t countN); 
    // have to write getter functions, destructor, and insert , deletion & update on edges code max(2 hours),
};
template <typename EdgesObj>
class DynamicSlabGraph<EdgesObj, false>
{
public:
    using EdgesContainer = typename EdgesObj::ContainerPolicy;
    using Edgesalloc = typename EdgesContainer::AllocPolicyT;
    using EdgeDynAllocator = typename Edgesalloc::DynamicAllocatorT;
    using EdgeDynContext = typename Edgesalloc::AllocatorContextT;

    using VertexT = typename EdgesObj::VertexT;

    using edgeHash = typename EdgesContainer::SlabHashT;
    using edgeHashCxt = typename EdgesContainer::SlabHashContextT;
    using SlabInfoT = typename EdgesContainer::SlabInfoT;

private:
    std::vector<edgeHash> VertexHost;
    std::vector<edgeHashCxt> tempVertexHost;

    EdgeDynAllocator dynallocator;

    int *h_BucketsPerVertex;
    int *h_BucketsPrefixSum;

    uint32_t       *d_firstUpdatedSlab;
    uint8_t        *d_firstUpdatedLaneId;
    int            *d_BucketsPrefixSum;
    int            *d_EdgesPerBucket;    
    int            *d_vertexDegree;
    bool           *d_isSlablistUpdated;
    uint32_t       *d_headptr;           // buckets head pointers
    edgeHashCxt    *graphVertex;         // gpu
    EdgeDynContext graphAlloc;           // gpu

public:
    DynamicSlabGraph(uint32_t N, int *h_vertexDegree, EdgeDynAllocator &allocator, float loadFactor, uint32_t device_index)
    {
        cudaSetDevice(device_index);
        h_BucketsPerVertex = new int[N];
        h_BucketsPrefixSum = new int[N];

        dynallocator = allocator; 

        int totalBuckets = 0;
        for (int i = 0; i < N; i++)
        {
            int temp = std::ceil((float)h_vertexDegree[i] / (32 * loadFactor));
            h_BucketsPerVertex[i] = (temp == 0) ? 1 : temp;
            totalBuckets += h_BucketsPerVertex[i];
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_firstUpdatedSlab, sizeof(uint32_t) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_firstUpdatedLaneId, sizeof(uint8_t) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_EdgesPerBucket, sizeof(int) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_isSlablistUpdated, sizeof(bool) * totalBuckets));
        CHECK_CUDA_ERROR(cudaMalloc(&d_BucketsPrefixSum, sizeof(int) * N));
        CHECK_CUDA_ERROR(cudaMalloc(&d_vertexDegree, sizeof(int) * N));
        CHECK_CUDA_ERROR(cudaMalloc(&d_headptr, 32 * sizeof(uint32_t) * totalBuckets));

        thrust::fill(thrust::device, d_firstUpdatedSlab, d_firstUpdatedSlab + totalBuckets, static_cast<uint32_t>(SlabInfoT::A_INDEX_POINTER));
        thrust::fill(thrust::device, d_firstUpdatedLaneId, d_firstUpdatedLaneId + totalBuckets, 0);
        thrust::fill(thrust::device, d_isSlablistUpdated, d_isSlablistUpdated + totalBuckets, false);

        ExclusiveScan(h_BucketsPerVertex, h_BucketsPerVertex + N, h_BucketsPrefixSum, 0);

        for (int i = 0; i < N; i++)
            VertexHost.emplace_back(reinterpret_cast<int8_t * >(d_headptr) + 128 * h_BucketsPrefixSum[i], d_firstUpdatedSlab + h_BucketsPrefixSum[i], d_firstUpdatedLaneId + h_BucketsPrefixSum[i], h_BucketsPerVertex[i], &allocator, device_index);

        auto GetContainerHashCtxt = [](auto &Container)
        {
            return Container.getSlabHashContext();
        };

        std::transform(VertexHost.begin(), VertexHost.end(), std::back_inserter(tempVertexHost), GetContainerHashCtxt);

        CHECK_CUDA_ERROR(cudaMalloc(&graphVertex, sizeof(edgeHashCxt) * N));
        CHECK_CUDA_ERROR(cudaMemcpy(graphVertex, tempVertexHost.data(), sizeof(edgeHashCxt) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_BucketsPrefixSum, h_BucketsPrefixSum, sizeof(int) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_vertexDegree, h_vertexDegree, sizeof(int) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_EdgesPerBucket,0,sizeof(int) * totalBuckets));
        graphAlloc = *dynallocator.getContextPtr();
    }
    ~DynamicSlabGraph()
    {
        delete h_BucketsPerVertex;
        delete h_BucketsPrefixSum;
        CHECK_CUDA_ERROR(cudaFree(d_firstUpdatedSlab));
        CHECK_CUDA_ERROR(cudaFree(d_firstUpdatedLaneId));
        CHECK_CUDA_ERROR(cudaFree(d_isSlablistUpdated));
        CHECK_CUDA_ERROR(cudaFree(d_BucketsPrefixSum));
        CHECK_CUDA_ERROR(cudaFree(d_vertexDegree));
        CHECK_CUDA_ERROR(cudaFree(d_EdgesPerBucket));
        CHECK_CUDA_ERROR(cudaFree(graphVertex));
        CHECK_CUDA_ERROR(cudaFree(d_headptr));
    }
    EdgeDynAllocator GetDynCtxt() {
        return graphAlloc; 
    }
    __device__ __forceinline__ void insertEdge(bool toInsert, VertexT &src, VertexT &dst, int lane, DynamicSlabGraph<EdgesObj, false>::EdgeDynAllocator &localctxt); 
    __device__ __forceinline__ void deleteEdge(bool toDelete, VertexT &src, VertexT &dst, int lane); 
    void insertEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN); 
    void deleteEdges(VertexT *sourceVertex, VertexT *dstVertex, int countN); 
};
#include "../graphsEditKernels/insertEdges.cu"
#include "../graphsEditKernels/deleteEdges.cu"
// #include "../graphsEditKernels/updateEdges.cu"

int main(){}