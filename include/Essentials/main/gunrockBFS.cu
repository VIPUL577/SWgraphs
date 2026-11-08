#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#define THREADSPERBLOCK2 512


__global__ void Advance_push(int *current_frontier,
                             int *mask,
                             int *col, int *row_indices,
                             int *prefix_sum, int edges,
                             int nothreads, int N, int EPT) // N == current frontier size.
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nothreads)
    {
        // Advance_push — same fix, using edges/nothreads
        __shared__ int localOutgoingFrontier[EPT * THREADSPERBLOCK2];
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
            mask[col[colIndex]] = 1;
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



int main(){
    
}