#pragma once
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "Essentials/main/graph.cu"

// no of threads = (noofslabs/slabsperwarp)*32
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
template <typename VertexT,typename edgeHashCtxt>
__global__ void BFSAdvance(VertexT *currentFrontier,VertexT *outgoingFrontier, int *SlabPrefixSum, edgeHashCtxt *graph, int cf_n, int noThreads , int slabsPerWarp )
{ /*
    1) ADD MASK IF THE VERTEX IS VALID OR NOT, OR DELETED OR THE VERTEX HAS ZERO NEIGHBOURS(WHILE LOOP)
    2) TAKE CARE END POINTS
    3) CAN USE SHARED MEMORY, BUT IN LATER STAGES. 
    4) TAKE CARE OF TOOMBSTONE VERTICES. 
  */  
    int tid = blockIdx.x*blockDim.x + threadIdx.x ; 
    if(tid < noThreads){
        
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
        int offset  = begin - SlabPrefixSum[binaryStart]; 
        auto startIt = graph[currentFrontier[binaryStart]].Begin();
        for(int i = 0 ; i < offset ; ++i , ++startIt);
        auto endIt = graph[currentFrontier[binaryStart]].End();
        for(int i = start ; i<end ; i++){ 
            outgoingFrontier[*startIt.GetPointer(lane)] = 1; 
            startIt++ ; 
            if(startIt == endIt){
                binaryStart++ ; 
                startIt = graph[currentFrontier[binaryStart]].Begin();
                endIt = graph[currentFrontier[binaryStart]].End();
            }
        }
    }
}

template<typename DynamicSlabGraph, typename VertexT>
int BFSAdvanceWrapper(DynamicSlabGraph graphClass , VertexT * currentFrontier, VertexT * outgoingFrontier ,VertexT* prefixKeep, int cf_n, int* slabPrefixSum, int slabsperwarp){
    graphClass.countSlabs(currentFrontier , slabPrefixSum , cf_n); 
    int noOfSlabs = getNumberSlabs<int>(slabPrefixSum,slabPrefixSum,cf_n); 
    int noOfThreads = 32*ceil((double)noOfSlabs/slabsPerWarp); 
    int grid = (noOfThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    BFSAdvance<VertexT,DynamicSlabGraph::edgeHashCxt><<<grid,THREADSPERBLOCK>>>(currentFrontier,outgoingFrontier,slabPrefixSum,graphClass.graphVertex,cf_n,noOfThreads,slabsperwarp);
    int cf_n = getNumberSlabs<VertexT>(outgoingFrontier, prefixKeep , graphClass.maxN); 
    grid = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<VertexT><<<grid, THREADSPERBLOCK>>>(outgoingFrontier, prefixKeep, graphClass.maxN, currentFrontier);
    initZero<VertexT><<<grid, THREADSPERBLOCK>>>(outgoingFrontier, graphClass.maxN);
    cudaDeviceSynchronize();
    return cf_n; 
}

void BFS()
{
    
}