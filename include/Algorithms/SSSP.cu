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

template <typename VertexT,typename EdgeValueT,typename edgeHashCtxt>
__global__ void SSSPAdvance(VertexT *currentFrontier,VertexT *outgoingFrontier,EdgeValueT *minDist, int *SlabPrefixSum, edgeHashCtxt * graph , int cf_n, int noThreads , int slabsPerWarp )
{ /*
    1) ADD MASK IF THE VERTEX IS VALID OR NOT, OR DELETED OR THE VERTEX HAS ZERO NEIGHBOURS(WHILE LOOP)
    2) TAKE CARE END POINTS
    3) CAN USE SHARED MEMORY, BUT IN LATER STAGES. 
    4) TAKE CARE OF TOOMBSTONE VERTICES. 
  */  
    int tid = blockIdx.x*blockDim.x + threadIdx.x ; 
    if(tid < noThreads){
        using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
        using SlabInfoT = typename ContainerPolicy::SlabInfoT;
        int warpId = tid/32; 
        int begin = warpId * slabsPerWarp ; 
        int end = (warpId+1) * slabsPerWarp; 
        int lane = tid & 0xFF; 

        int binaryStart = 0; 
        int binaryEnd = cf_n -1 ; 
        if(lane == 0){
            while(binaryEnd>binaryStart){
                int mid = (binaryStart+binaryEnd)/2; 
                if(SlabPrefixSum[mid] <= begin) binaryStart = mid ;
                else binaryEnd = mid - 1;  
            }
        }
        binaryStart = __shfl_sync(0xFFFFFFFF , binaryStart , 0); 
        auto currentVertex = currentFrontier[binaryStart];  
        int offset  = begin - SlabPrefixSum[binaryStart]; 
        auto startIt = graph[currentVertex].Begin();
        for(int i = 0 ; i < offset ; ++i , ++startIt);
        auto endIt = graph[currentVertex].End();
        for(int i = start ; i<end ; i++){ 
            auto tentativeV = *startIt.GetPointer(lane); 
            auto edgeWeight = __shfl_down(tentativeV,1); 
            if(((1 << Warp.thread_rank()) & 0x15555555) && (tentativeV != EMPTY_KEY) &&(tentativeV != TOMBSTONE_KEY)&&(currentVertex != tentativeV)){
            EdgeValueT newDist = minDist[currentVertex] + edgeWeight ; 
            EdgeValueT old = atomicMin(&minDist[tentativeV],newDist); 
            if(newDist < old)
                outgoingFrontier[tentativeV] = 1; 
            }
            startIt++ ; 
            if(startIt == endIt){
                binaryStart++ ; 
                currentVertex = currentFrontier[binaryStart]; 
                startIt = graph[currentVertex].Begin();
                endIt = graph[currentVertex].End();
            }
        }
    }
}

template<typname DynamicSlabGraph,typename VertexT, typename EdgeValueT>
int SSSPAdvanceWrapper(DynamicSlabGraph graphClass,VertexT* currentFrontier,VertexT* outgoingFrontier,VertexT* prefixKeep, int* slabPrefixSum,EdgeValueT *minDist ,int cf_n, int slabsperwarp){
    graphClass.countSlabs(currentFrontier , slabPrefixSum , cf_n); 
    int noOfSlabs = getNumberSlabs<int>(slabPrefixSum,slabPrefixSum,cf_n); 
    int noOfThreads = 32*ceil((double)noOfSlabs/slabsPerWarp); 
    int grid = (noOfThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    SSSPAdvance<VertexT, EdgeValueT, DynamicSlabGraph::edgeHashCxt><<<grid, THREADSPERBLOCK>>>(currentFrontier, outgoingFrontier, minDist, slabPrefixSum, graphClass.graphVertex, cf_n, noOfThreads, slabsPerWarp ); 
    int cf_n = getNumberSlabs<VertexT>(outgoingFrontier, prefixKeep , graphClass.maxN); 
    grid = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<VertexT><<<grid, THREADSPERBLOCK>>>(outgoingFrontier, prefixKeep, graphClass.maxN, currentFrontier);
    initZero<VertexT><<<grid, THREADSPERBLOCK>>>(outgoingFrontier, graphClass.maxN);
    cudaDeviceSynchronize();
    return cf_n; 
    
}