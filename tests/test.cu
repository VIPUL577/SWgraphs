#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <algorithm>
#include <stdio.h>
#include <bits/stdc++.h>

#include <cuda.h>
#include <math.h>
#include <time.h>
#include "../include/Algorithms/SSSP.cu"

using namespace std;

long long int INF = getWord(INT_MAX,-1);
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
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);

        int u, v, w;
        if (!(iss >> u >> v >> w))
            continue; 

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
void getFirst32Bits(distanceArray* arr, std::vector<int> &parent,std::vector<int> &result,int n) {
    for (int i = 0; i < n; i++) {
        result[i] = getDistance(arr[i]); 
        parent[i] = getParent(arr[i]); 
    }
}
void runSSSP(
    const int *src,
    const int *dst,
    const int *weights,
    int V,
    int E,
    int source,
    std::vector<int> &CPUdistance,
    std::vector<int> &CPUparent)
{
    if (V <= 0 || E < 0 || source < 0 || source >= V)
        throw std::invalid_argument("Invalid graph dimensions/source");
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

    AllocPolicy::DynamicAllocatorT allocator;

    Graph graph(
        V,
        degree.data(),
        allocator,
        1.0f, // load factor
        0     // CUDA device
    );

    // The insertion kernel accesses these arrays on the GPU.
    int *d_src = nullptr, *d_dst = nullptr, *d_weights = nullptr;
    // cout<<"[DEBUG] 1"<<endl; 
    checkCuda(cudaMalloc(&d_src, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_dst, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_weights, E * sizeof(int)));

    checkCuda(cudaMemcpy(d_src, src, E * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dst, dst, E * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weights, weights, E * sizeof(int),
                         cudaMemcpyHostToDevice));
    // cout<<"[DEBUG] 2"<<endl; 

    graph.insertEdges(d_src, d_dst, d_weights, E);
    checkCuda(cudaDeviceSynchronize());
    // cout<<"[DEBUG] 3"<<endl; 

    // Validate before allocating vertex arrays.
    if (V <= 0 || source < 0 || source >= V)
    {
        // Handle invalid graph/source.
        return;
    }

    int *d_current = nullptr;
    int *d_next = nullptr;
    int *d_prefix = nullptr;
    int *d_slabPrefix = nullptr;
    // int *d_parent = nullptr;

    distanceArray *d_distance = nullptr;
    // cout<<"[DEBUG] 4"<<endl; 

    checkCuda(cudaMalloc(&d_current, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_next, V * sizeof(int)));
    // checkCuda(cudaMalloc(&d_parent, V * sizeof(int)));

    checkCuda(cudaMalloc(&d_prefix, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_slabPrefix, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_distance, V * sizeof(distanceArray)));

    checkCuda(cudaMemset(d_current, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_next, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_prefix, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_slabPrefix, 0, V * sizeof(int)));
    // checkCuda(cudaMemset(d_parent, 0, V * sizeof(int)));


    thrust::fill(thrust::device, d_distance, d_distance + V, INF);
    // cout<<"[DEBUG] 5"<<endl; 

    distanceArray zero = getWord(0,-1);
    checkCuda(cudaMemcpy(&d_current[0], &source, sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_distance + source, &zero, sizeof(distanceArray),cudaMemcpyHostToDevice));

    clock_t starttt = clock();
    SSSP<Graph, int, int>(graph,d_current,d_next,d_prefix,d_slabPrefix,d_distance,1,16);
    clock_t enddd = clock();
    // cout<<"[DEBUG] 6"<<endl; 

    // checkCuda(cudaDeviceSynchronize());
    std::vector<distanceArray> arr(V);
    std::vector<int> distance(V,0);
    std::vector<int> parent(V,-1);

    checkCuda(cudaMemcpy(arr.data(),d_distance,V * sizeof(distanceArray),cudaMemcpyDeviceToHost));
    // cout<<"[DEBUG] 7"<<endl; 

    getFirst32Bits(arr.data(),parent,distance,V); 
    // cout<<"[DEBUG] 8"<<endl; 

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_weights);

    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_prefix);
    cudaFree(d_slabPrefix);
    cudaFree(d_distance);
    int maxDist = 0;
    int maxDistCPU = 0;

    int mismatch = 0;
    int parentmismatch = 0 ; 
    for (int v = 0; v < V; ++v)
    {
        if (distance[v] != INT_MAX)
        {
            maxDist = max(maxDist, distance[v]);
        }
        if (CPUdistance[v] != INT_MAX)
            maxDistCPU = max(maxDistCPU, CPUdistance[v]);

        if (CPUdistance[v] != INT_MAX && distance[v] != INT_MAX && distance[v] != CPUdistance[v])
            mismatch++;
            // cout<<"GPU : "<<distance[v]<<" CPU : "<<CPUdistance[v]<<endl;
        if (v!=source && CPUdistance[v] != INT_MAX && distance[v] != INT_MAX && parent[v] != CPUparent[v])
            parentmismatch++;
        
    }
    cout << "#########DYNAMIC SSSP WITH SLABHASH########" << endl;
    cout << "vertices           : " << V << '\n';
    cout << "edges              : " << E << '\n';
    cout << "source             : " << source << endl;
    cout << "MAX Distance       : " << maxDist << endl;
    cout << "MAX Distance CPU   : " << maxDistCPU << endl;
    cout << "misMatched         : " << mismatch << endl;
    cout << "Parnet misMatched  : " << parentmismatch << endl;


    printf("Time Taken (GPU)   : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
}

void dijkstra(
    int V,
    int E,
    const std::vector<int> &src,
    const std::vector<int> &dst,
    const std::vector<int> &weights,
    std::vector<int> &parent,
    std::vector<int> &dist,

    int source)
{
    std::vector<std::vector<std::pair<int, int>>> adj(V);

    for (int i = 0; i < E; i++)
        adj[src[i]].push_back({dst[i], weights[i]});

    // std::vector<int> dist(V, INT_MAX);
    parent.assign(V, -1);

    dist[source] = 0;

    std::priority_queue<
        std::pair<int, int>,
        std::vector<std::pair<int, int>>,
        std::greater<std::pair<int, int>>> pq;

    pq.push({0, source});

    while (!pq.empty())
    {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d != dist[u])
            continue;

        for (const auto &edge : adj[u])
        {
            int v = edge.first;
            int w = edge.second;

            long long candidate = static_cast<long long>(d) + w;

            if (candidate > INT_MAX - 1)
                continue;

            int newDist = static_cast<int>(candidate);

            if (newDist < dist[v])
            {
                dist[v] = newDist;
                parent[v] = u;
                pq.push({newDist, v});
            }
            else if (newDist == dist[v] &&
                     v != source &&
                     (parent[v] == -1 || u < parent[v]))
            {
                parent[v] = u;
            }
        }
    }


}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input.txt>\n";
        return 1;
    }
    string filepath = argv[1];
    int V = 0;
    int E = 0;
    std::vector<int> src;
    std::vector<int> dst;
    std::vector<int> weights;
    
    int source = 2677;
    if (file_to_array(filepath, src, dst, weights, V, E))
    {
        std::vector<int> parent(V, -1);
        std::vector<int> CPUdist(V, INT_MAX);

        clock_t starttt = clock();
        dijkstra(V, E, src, dst, weights, parent,CPUdist,source);
        clock_t enddd = clock();
        printf("Time Taken (CPU)   : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
        runSSSP(src.data(), dst.data(), weights.data(), V, E, source, CPUdist,parent);
        cudaDeviceSynchronize(); 

    }

    return 0;
}
