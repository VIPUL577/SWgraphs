#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "../include/Algorithms/SSSP.cu"

using namespace std; 

int INF = INT_MAX; 
using AllocPolicy = FullAllocatorPolicy<8, 32, 1>;
using WeightedObj = WeightedEdgesObj<AllocPolicy>;
using Graph = DynamicSlabGraph<WeightedObj, true>;

bool file_to_array(string filepath,
                   std::vector<int> &src, std::vector<int> &dst, std::vector<int> &weights,
                   int &V, int &E)
{
    std::ifstream file(filepath);

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filepath << '\n';
        return false;
    }

    std::string line;
    E = 0;
    V = 0;

    while (std::getline(file, line))
    {
        // Skip comments and blank lines.
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);

        int u, v, w;
        if (!(iss >> u >> v >> w))
            continue; // Skip headers or invalid lines.

        src.push_back(u);
        dst.push_back(v);
        weights.push_back(w);
        ++E;

        V = std::max(V, std::max(u, v) + 1);
    }

    return true;
}
static void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// src[i] -> dst[i], with edge weight weights[i].
// The three input arrays remain CPU arrays.
void runSSSP(
    const int* src,
    const int* dst,
    const int* weights,
    int V,
    int E,
    int source)
{
    if (V <= 0 || E < 0 || source < 0 || source >= V)
        throw std::invalid_argument("Invalid graph dimensions/source");
    std::cout<<"startting SSSP"<<std::endl; 
    std::vector<int> degree(V, 0);

    for (int i = 0; i < E; ++i)
    {
        if (src[i] < 0 || src[i] >= V ||
            dst[i] < 0 || dst[i] >= V)
            throw std::invalid_argument("Edge endpoint out of range");

        if (weights[i] < 0)
            throw std::invalid_argument(
                "SSSP requires non-negative edge weights");

        ++degree[src[i]];
    }
    std::cout<<"graphclass"<<std::endl; 

    // Construct the weighted dynamic graph.
    AllocPolicy::DynamicAllocatorT allocator;

    Graph graph(
        V,
        degree.data(),
        allocator,
        1.0f,  // load factor
        0      // CUDA device
    );
    std::cout<<"graphclass done"<<std::endl; 
    std::cout<<"E: "<<E<<"V: "<<V<<std::endl; 
    
    // The insertion kernel accesses these arrays on the GPU.
    int *d_src = nullptr, *d_dst = nullptr, *d_weights = nullptr;
    std::cout<<"1"<<std::endl; 

    checkCuda(cudaMalloc(&d_src, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_dst, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_weights, E * sizeof(int)));

    checkCuda(cudaMemcpy(d_src, src, E * sizeof(int),
                        cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dst, dst, E * sizeof(int),
                        cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weights, weights, E * sizeof(int),
                        cudaMemcpyHostToDevice));

    std::cout<<"2"<<std::endl; 
    graph.insertEdges(d_src, d_dst, d_weights, E);
    checkCuda(cudaDeviceSynchronize());

    // Validate before allocating vertex arrays.
    if (V <= 0 || source < 0 || source >= V)
    {
        // Handle invalid graph/source.
        return;
    }
    std::cout<<"3"<<std::endl; 

    int *d_current = nullptr;
    int *d_next = nullptr;
    int *d_prefix = nullptr;
    int *d_slabPrefix = nullptr;
    int *d_distance = nullptr;
    std::cout<<"4"<<std::endl; 

    checkCuda(cudaMalloc(&d_current, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_next, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_prefix, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_slabPrefix, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_distance, V * sizeof(int)));

    checkCuda(cudaMemset(d_current, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_next, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_prefix, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_slabPrefix, 0, V * sizeof(int)));

    thrust::fill(thrust::device, d_distance, d_distance + V, INF);
    std::cout<<"5"<<std::endl; 

    const int zero = 0;
    checkCuda(cudaMemcpy(d_current, &source, sizeof(int),
                        cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_distance + source, &zero, sizeof(int),
                        cudaMemcpyHostToDevice));
    // Run ONLY SSSP. Initial frontier contains the source.
    // One slab per warp.
    std::cout<<"SSSP"<<std::endl; 

    clock_t starttt = clock();
    SSSP<Graph, int, int>(
        graph,
        d_current,
        d_next,
        d_prefix,
        d_slabPrefix,
        d_distance,
        1,
        8);
    clock_t enddd = clock();
    std::cout<<"SSSPdone "<<std::endl; 

    // checkCuda(cudaDeviceSynchronize());

    std::vector<int> distance(V);
    checkCuda(cudaMemcpy(
        distance.data(),
        d_distance,
        V * sizeof(int),
        cudaMemcpyDeviceToHost));

    std::cout << "SSSP distances from source " << source << ":\n";
    
    int maxDist =  0 ; 
    for (int v = 0; v < V; ++v)
    {
        if (distance[v] != INF) maxDist = max(maxDist,distance[v] );
    }
    cout << "#########SSSP GUNROCK########" << endl;
    cout << "vertices           : " << V << '\n';
    cout << "edges              : " << E << '\n';
    cout << "source             : " << source << endl;
    cout << "MAX Distance       : " << maxDist << endl;
    printf( "Time Taken (GPU)   : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_weights);

    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_prefix);
    cudaFree(d_slabPrefix);
    cudaFree(d_distance);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input.txt>\n";
        return 1;
    }
    string filepath = argv[1];
    int V = 0 ; 
    int E = 0 ; 
    std::vector<int> src; 
    std::vector<int> dst; 
    std::vector<int> weights; 

    if(file_to_array(filepath,src,dst,weights,V,E))    
        runSSSP(src.data(), dst.data(), weights.data(), V, E, 100);

    return 0;
}
