#pragma once
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "Essentials/main/graph.cu"

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
template <typename VertexT, typename EdgeValueT, typename edgeHashCtxt>
__global__ void SSSPAdvance(VertexT *currentFrontier, VertexT *outgoingFrontier,VertexT *parent, EdgeValueT *minDist, int *SlabPrefixSum, edgeHashCtxt *graph, int cf_n, int noThreads, int slabsperwarp, int totalSlabs)
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
                EdgeValueT newDist = minDist[currentVertex] + edgeWeight;
                EdgeValueT old = atomicMin(&minDist[tentativeV], newDist);
                if (newDist < old){
                    outgoingFrontier[tentativeV] = 1;
                    parent[tentativeV] = currentVertex; 
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
template<typename p>
int generateFrontierWrapper(p* currentFrontier, p* bitmask, p* prefixKeep, int N){
    int cf_n = getNumberSlabs<p>(bitmask, prefixKeep, N);
    int grid = (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<p><<<grid, THREADSPERBLOCK>>>(bitmask, prefixKeep, N, currentFrontier);
    initZero<p><<<grid, THREADSPERBLOCK>>>(bitmask, N);

    return cf_n; 
}
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
int SSSPAdvanceWrapper(DynamicSlabGraph &graphClass, VertexT *currentFrontier, VertexT *outgoingFrontier,VertexT *parent, VertexT *prefixKeep, int *slabPrefixSum, EdgeValueT *minDist, int cf_n, int slabsperwarp)
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
    int noOfThreads = 32 * ceil((double)noOfSlabs / slabsperwarp);
    int grid = (noOfThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    SSSPAdvance<VertexT, EdgeValueT, typename DynamicSlabGraph::edgeHashCxt><<<grid, THREADSPERBLOCK>>>(currentFrontier, outgoingFrontier, parent,minDist, slabPrefixSum, graphClass.getVertexContexts(), cf_n, noOfThreads, slabsperwarp, noOfSlabs);
    return generateFrontierWrapper<VertexT>(currentFrontier,outgoingFrontier,prefixKeep,graphClass.getNumVertices()); 
}
template <typename DynamicSlabGraph, typename VertexT, typename EdgeValueT>
void SSSP(DynamicSlabGraph &graphClass, VertexT *currentFrontier,VertexT *parent, VertexT *outgoingFrontier, VertexT *prefixKeep, int *slabPrefixSum, EdgeValueT *minDist, int cf_n, int slabsperwarp)
{
    while (cf_n > 0)
    {
        cf_n = SSSPAdvanceWrapper<DynamicSlabGraph, VertexT, EdgeValueT>(graphClass, currentFrontier, outgoingFrontier,parent, prefixKeep,slabPrefixSum, minDist, cf_n, slabsperwarp);
    }
}