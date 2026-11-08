#pragma once
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "Essentials/main/graph.cu"
#include "Essentials/main/helpers.cuh"
#define EDGESPERTHREAD 4
// THREADSPERBLOCK2 512

template <typename VertexT>
__global__ void Advance_push(VertexT *current_frontier,
                             int *mask, distanceArray *minDist,
                             VertexT *col, int *row_indices,
                             int *prefix_sum, int edges,
                             int nothreads, int N) // N == current frontier size.
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nothreads)
    {
        // Advance_push — same fix, using edges/nothreads
        int start = (idx == 0) ? 0 : (int)floor(idx * ((double)edges / nothreads));
        int end = (idx == nothreads - 1) ? edges : (int)floor((idx + 1) * ((double)edges / nothreads));
        int bstart = 0;
        int benf = N - 1;
        while (benf > bstart)
        {
            int mid = (bstart + benf + 1) / 2;
            if (prefix_sum[mid] <= start)
                bstart = mid;
            else
                benf = mid - 1;
        }

        int offset = start - prefix_sum[bstart];
        int next_ps = (bstart + 1 < N) ? prefix_sum[bstart + 1] : edges;
        int degree = next_ps - prefix_sum[bstart];
        int colIndex = row_indices[current_frontier[bstart]] + offset;
        int counter = 0;
        int counter2 = 0;
        for (int i = start; i < end; i++)
        {
            // if(!isValidate(minDist[col[colIndex]]))
            {
                mask[col[colIndex]] = 1;
                inValidate(minDist[col[colIndex]]);
            }
            colIndex++;
            counter++;
            counter2++;
            if (counter >= degree - offset)
            {
                bstart++;
                while (bstart < N)
                {
                    int np = (bstart + 1 < N) ? prefix_sum[bstart + 1] : edges;
                    degree = np - prefix_sum[bstart];
                    if (degree > 0)
                        break;
                    bstart++;
                }
                if (bstart >= N)
                    break;
                colIndex = row_indices[current_frontier[bstart]];
                counter = 0;
                offset = 0;
            }
        }
    }
}
template <typename VertexT>
int advancePush(VertexT *d_current_frontier, int *d_outgoing_frontier, distanceArray *minDist, VertexT *child, int *childOfft, int *d_degree_array,
                int *d_prekeep, int cf_n, int V, int E)
{
    int blocks = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    getDegree<<<blocks, THREADSPERBLOCK>>>(d_current_frontier, childOfft, d_degree_array, cf_n, V, E);
    int number_of_edges = getNumberSlabs<int>(d_degree_array, d_degree_array, cf_n);
    int totalThreads = (number_of_edges + EDGESPERTHREAD - 1) / EDGESPERTHREAD;
    int grid = (totalThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    Advance_push<<<grid, THREADSPERBLOCK>>>(d_current_frontier, d_outgoing_frontier, minDist, child, childOfft, d_degree_array, number_of_edges, totalThreads, cf_n);
    return generateFrontierWrapper<VertexT>(d_current_frontier, d_outgoing_frontier, d_prekeep, V);
}
template <typename VertexT>
void BFSgunrockChild(VertexT *d_current_frontier, int *d_outgoing_frontier, distanceArray *minDist, VertexT *child, int *childOfft, int *d_degree_array,
                     int *d_prekeep, int cf_n, int V, int E)
{
    while (cf_n > 0)
    {
        cf_n = advancePush<VertexT>(d_current_frontier,d_outgoing_frontier,minDist,child,childOfft,d_degree_array,d_prekeep,cf_n,V,E);
    }
}
