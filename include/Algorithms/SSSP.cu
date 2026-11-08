#pragma once
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "Essentials/main/graph.cu"
#include "Essentials/main/helpers.cuh"
#include "Essentials/main/gunrockBFS.cuh"

// ############################################################################################
//                                        KERNELS
// ############################################################################################
template <typename VertexT, typename EdgeValueT, typename edgeHashCtxt>
__global__ void SSSPAdvance(VertexT *currentFrontier, VertexT *outgoingFrontier, distanceArray *minDist, int *SlabPrefixSum, edgeHashCtxt *graph, int cf_n, int noThreads, int slabsperwarp, int totalSlabs)
{ /*
    1) ADD MASK IF THE VERTEX IS VALID OR NOT, OR DELETED OR THE VERTEX HAS ZERO NEIGHBOURS(WHILE LOOP)
    2) TAKE CARE END POINTS
    3) CAN USE SHARED MEMORY, BUT IN LATER STAGES.
    4) TAKE CARE OF TOOMBSTONE VERTICES. done
  */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < noThreads)
    {
        int warpId = tid / 32;
        int begin = warpId * slabsperwarp;
        int end = (warpId + 1) * slabsperwarp;
        int lane = tid & 0x1F;

        int binaryStart = 0;
        int binaryEnd = cf_n - 1;
        if (lane == 0)
        {
            while (binaryEnd > binaryStart)
            {
                int mid = (binaryStart + binaryEnd + 1) / 2; // was (s+e)/2
                if (SlabPrefixSum[mid] <= begin)
                    binaryStart = mid;
                else
                    binaryEnd = mid - 1;
            }
        }
        binaryStart = __shfl_sync(0xFFFFFFFF, binaryStart, 0);
        auto currentVertex = currentFrontier[binaryStart];
        int offset = begin - SlabPrefixSum[binaryStart];
        auto startIt = graph[currentVertex].Begin();
        for (int i = 0; i < offset; ++i, ++startIt)
            ;
        auto endIt = graph[currentVertex].End();
        int stop = min(end, totalSlabs);
        for (int i = begin; i < stop; i++)
        {
            VertexT tentativeV = *startIt.GetPointer(lane);
            EdgeValueT edgeWeight = __shfl_down_sync(0xffffffff, tentativeV, 1);
            if (((1u << lane) & 0x15555555) && (tentativeV != EMPTY_KEY) && (tentativeV != TOMBSTONE_KEY) && (currentVertex != tentativeV))
            {
                EdgeValueT newDist = getDistance(minDist[currentVertex]) + edgeWeight;
                distanceArray newPD = getWord(newDist, currentVertex);
                distanceArray old = atomicMin(&minDist[tentativeV], newPD);
                if (newPD < old)
                {
                    outgoingFrontier[tentativeV] = 1;
                }
            }
            startIt++;
            while (startIt == endIt && binaryStart + 1 < cf_n && i + 1 < stop)
            {
                binaryStart++;

                currentVertex = currentFrontier[binaryStart];
                startIt = graph[currentVertex].Begin();
                endIt = graph[currentVertex].End();
            }
        }
    }
}
template <typename VertexT, typename EdgeValueT, typename edgeHashCtxt>
__global__ void InvalidationAdvance(VertexT *currentFrontier, VertexT *outgoingFrontier, distanceArray *minDist, int *SlabPrefixSum, edgeHashCtxt *graph, int cf_n, int noThreads, int slabsperwarp, int totalSlabs)
{ /*
    1) ADD MASK IF THE VERTEX IS VALID OR NOT, OR DELETED OR THE VERTEX HAS ZERO NEIGHBOURS(WHILE LOOP)
    2) TAKE CARE END POINTS
    3) CAN USE SHARED MEMORY, BUT IN LATER STAGES.
    4) TAKE CARE OF TOOMBSTONE VERTICES. done
  */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < noThreads)
    {
        int warpId = tid / 32;
        int begin = warpId * slabsperwarp;
        int end = (warpId + 1) * slabsperwarp;
        int lane = tid & 0x1F;

        int binaryStart = 0;
        int binaryEnd = cf_n - 1;
        if (lane == 0)
        {
            while (binaryEnd > binaryStart)
            {
                int mid = (binaryStart + binaryEnd + 1) / 2; // was (s+e)/2
                if (SlabPrefixSum[mid] <= begin)
                    binaryStart = mid;
                else
                    binaryEnd = mid - 1;
            }
        }
        binaryStart = __shfl_sync(0xFFFFFFFF, binaryStart, 0);
        auto currentVertex = currentFrontier[binaryStart];
        int offset = begin - SlabPrefixSum[binaryStart];
        auto startIt = graph[currentVertex].Begin();
        for (int i = 0; i < offset; ++i, ++startIt)
            ;
        auto endIt = graph[currentVertex].End();
        int stop = min(end, totalSlabs);
        for (int i = begin; i < stop; i++)
        {
            VertexT tentativeV = *startIt.GetPointer(lane);
            EdgeValueT edgeWeight = __shfl_down_sync(0xffffffff, tentativeV, 1);
            if (((1u << lane) & 0x15555555) && (tentativeV != EMPTY_KEY) && (tentativeV != TOMBSTONE_KEY) && (currentVertex != tentativeV))
            {
                if (isValidate(minDist[tentativeV]) && !isValidate(minDist[currentVertex]))
                {
                    outgoingFrontier[currentVertex] = 1;
                }
            }
            startIt++;
            while (startIt == endIt && binaryStart + 1 < cf_n && i + 1 < stop)
            {
                binaryStart++;

                currentVertex = currentFrontier[binaryStart];
                startIt = graph[currentVertex].Begin();
                endIt = graph[currentVertex].End();
            }
        }
    }
}
// ############################################################################################
//                                GENERATING CHILD CSR
// ############################################################################################
template <typename VertexT>
__global__ void countChild(distanceArray *minDist, int *count, int V)
{
    // launch V
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V)
    {
        auto parent = getParent(minDist[tid]);
        if (parent == -1)
            return;
        atomicAdd(&count[parent], 1);
    }
}
template <typename VertexT>
__global__ void populateChild(VertexT *child, int *childOfft, int *cursor, distanceArray *minDist, int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V)
    {
        auto parent = getParent(minDist[tid]);
        if (parent != -1)
            child[atomicAdd(&cursor[parent], 1)] = static_cast<VertexT>(tid);
    }
}
template <typename VertexT>
int generateChildCSR(VertexT *child, int *childOfft, int *vkeep, distanceArray *minDist, int V)
{

    int grid = (V + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    initZero<int><<<grid, THREADSPERBLOCK>>>(childOfft, V);
    countChild<VertexT><<<grid, THREADSPERBLOCK>>>(minDist, childOfft, V);
    int numberChildern = getNumberSlabs<VertexT>(childOfft, childOfft, V);
    thrust::copy(thrust::device, childOfft, childOfft + V, vkeep);
    populateChild<VertexT><<<grid, THREADSPERBLOCK>>>(child, childOfft, vkeep, minDist, V);
    return numberChildern;
    // std::cout<<"DONE"<<std::endl;
}
// ############################################################################################
//                           GENERATING FRONTIER INCREMENTAL
// ############################################################################################
template <typename VertexT, typename EdgeValueT>
__global__ void incrementalRelax(VertexT *src, VertexT *dst, EdgeValueT *weights, distanceArray *minDist, VertexT *outgoingFrontier, int E)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < E)
    {
        VertexT u = src[tid];
        VertexT v = dst[tid];

        if (getDistance(minDist[u]) == INT_MAX)
            return;

        int newDist = getDistance(minDist[u]) + weights[tid];

        distanceArray candidate = getWord(newDist, u);
        distanceArray old = atomicMin(&minDist[v], candidate);

        if (candidate < old)
            outgoingFrontier[v] = 1;
    }
}
// ############################################################################################
//                           GENERATING FRONTIER DECREMENTAL
// ############################################################################################
template <typename VertexT>
__global__ void invalidateKernel(VertexT *src, VertexT *dst, int *mask, distanceArray *minDist, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        if (getParent(minDist[dst[tid]]) == src[tid])
        {
            inValidate(minDist[dst[tid]]);
            mask[dst[tid]] = 1;
        }
    }
}

// ############################################################################################
//                                   WRAPPER FUNCTIONS
// ############################################################################################
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
int SSSPInvalidationAdvanceWrapper(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier, VertexT *prefixKeep, int *slabPrefixSum, distanceArray *minDist, int slabsperwarp)
{

    // thrust::sequence(
    //     thrust::device,
    //     currentFrontier,
    //     currentFrontier + graphClass.getNumVertices(), 0);

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // printf("SEQUENCE OK\n");

    // graphClass.countSlabs(
    //     currentFrontier,
    //     slabPrefixSum,
    //     graphClass.getNumVertices());

    // CHECK_CUDA_ERROR(cudaGetLastError());
    // printf("COUNT SLABS LAUNCHED\n");

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // printf("COUNT SLABS EXECUTED\n");

    // std::cout << "AFTER COUNT SLABS" << std::endl;

    // int noOfSlabs = getNumberSlabs<int>(slabPrefixSum, slabPrefixSum, graphClass.getNumVertices());

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // printf("PREFIX SUM OK: %d\n", noOfSlabs);
    thrust::sequence(thrust::device, currentFrontier, currentFrontier + graphClass.getNumVertices(), 0);
    graphClass.countSlabs(currentFrontier, slabPrefixSum, graphClass.getNumVertices());
    int noOfSlabs = getNumberSlabs<int>(slabPrefixSum, slabPrefixSum, graphClass.getNumVertices());
    int noOfThreads = 32 * ceil((double)noOfSlabs / slabsperwarp);
    int grid = (noOfThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    InvalidationAdvance<VertexT, EdgeValueT, typename DynamicSlabGraph::edgeHashCxt><<<grid, THREADSPERBLOCK>>>(currentFrontier, outgoingFrontier, minDist, slabPrefixSum, graphClass.getVertexContexts(), graphClass.getNumVertices(), noOfThreads, slabsperwarp, noOfSlabs);
    return generateFrontierWrapper<VertexT>(currentFrontier, outgoingFrontier, prefixKeep, graphClass.getNumVertices());
}
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
int SSSPAdvanceWrapper(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier, VertexT *prefixKeep, int *slabPrefixSum, distanceArray *minDist, int cf_n, int slabsperwarp)
{
    graphClass.countSlabs(currentFrontier, slabPrefixSum, cf_n);
    //<DEBUG>
    // int grida = (cf_n + THREADSPERBLOCK -1)/THREADSPERBLOCK;
    // printKernel<int><<<grida,THREADSPERBLOCK>>>(slabPrefixSum,cf_n);
    // thrust::device_ptr<int> arr(slabPrefixSum);
    // auto result = thrust::minmax_element(arr, arr + cf_n);
    // int minVal = *result.first;
    // int maxVal = *result.second;
    // std::cout <<"cf_n: "<<cf_n<<" Avg: "<<(noOfSlabs / cf_n) << " Max: "<< maxVal<<" Min: "<<minVal<< std::endl; // DEBUG
    //</DEBUG>
    int noOfSlabs = getNumberSlabs<int>(slabPrefixSum, slabPrefixSum, cf_n);
    // std::cout << "[DEBUG] 2" << std::endl;

    int noOfThreads = 32 * ceil((double)noOfSlabs / slabsperwarp);
    int grid = (noOfThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    SSSPAdvance<VertexT, EdgeValueT, typename DynamicSlabGraph::edgeHashCxt><<<grid, THREADSPERBLOCK>>>(currentFrontier, outgoingFrontier, minDist, slabPrefixSum, graphClass.getVertexContexts(), cf_n, noOfThreads, slabsperwarp, noOfSlabs);
    // std::cout << "[DEBUG] 2" << std::endl;
    
    return generateFrontierWrapper<VertexT>(currentFrontier, outgoingFrontier, prefixKeep, graphClass.getNumVertices());
}
// ############################################################################################
//                                   MAIN FUNCTIONS
// ############################################################################################
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
void SSSP(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier, VertexT *prefixKeep, int *slabPrefixSum, distanceArray *minDist, int cf_n, int slabsperwarp)
{
    while (cf_n > 0)
    {
        // clock_t starttt = clock();
        cf_n = SSSPAdvanceWrapper<DynamicSlabGraph, VertexT, EdgeValueT>(graphClass, currentFrontier, outgoingFrontier, prefixKeep, slabPrefixSum, minDist, cf_n, slabsperwarp);
        // clock_t enddd = clock();
        // printf("Time Taken (GPU)   : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
    }
    // std::cout<<"DONE"<<std::endl;
}
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
void DecrementalSSSP(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier, VertexT *src, VertexT *dst, VertexT *prefixKeep, int *slabPrefixSum, distanceArray *minDist, VertexT *child, int *childOfft, int cf_n, int slabsperwarp)
{
    int Etree = generateChildCSR(child, childOfft, slabPrefixSum, minDist, graphClass.getNumVertices());
    int grid = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    invalidateKernel<VertexT><<<grid, THREADSPERBLOCK>>>(src, dst, outgoingFrontier, minDist, cf_n);
    int numInvalid = generateFrontierWrapper<VertexT>(currentFrontier, outgoingFrontier, prefixKeep, graphClass.getNumVertices());
    BFSgunrockChild(currentFrontier, outgoingFrontier, minDist, child, childOfft, slabPrefixSum, prefixKeep, numInvalid, graphClass.getNumVertices(), Etree);
    int cf_n2 = SSSPInvalidationAdvanceWrapper<DynamicSlabGraph, VertexT, EdgeValueT>(graphClass, currentFrontier, outgoingFrontier, prefixKeep, slabPrefixSum, minDist, slabsperwarp);
    graphClass.deleteEdges(src, dst, cf_n);
    SSSP<DynamicSlabGraph, VertexT, EdgeValueT>(graphClass, currentFrontier, outgoingFrontier, prefixKeep, slabPrefixSum, minDist, cf_n2, slabsperwarp);
}

template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
void IncrementalSSSP(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier, VertexT *src, VertexT *dst,
                     EdgeValueT *weights, VertexT *prefixKeep, int *slabPrefixSum, distanceArray *minDist, int numInsertedEdges, int slabsperwarp)
{
    int V = graphClass.getNumVertices();
    graphClass.insertEdges(src, dst, weights, numInsertedEdges);
    int grid = (numInsertedEdges + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    incrementalRelax<VertexT, EdgeValueT><<<grid, THREADSPERBLOCK>>>(src, dst, weights, minDist, outgoingFrontier, numInsertedEdges);
    int cf_n = generateFrontierWrapper<VertexT>(currentFrontier, outgoingFrontier, prefixKeep, V);
    SSSP<DynamicSlabGraph, VertexT, EdgeValueT>(graphClass, currentFrontier, outgoingFrontier, prefixKeep, slabPrefixSum, minDist, cf_n, slabsperwarp);
}