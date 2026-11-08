
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "../include/Algorithms/SSSP.cu"

using namespace std;

using AllocPolicy = FullAllocatorPolicy<8, 32, 1>;
using WeightedObj = WeightedEdgesObj<AllocPolicy>;
using Graph = DynamicSlabGraph<WeightedObj, true>;

static void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

bool file_to_array(
    const string &filepath,
    vector<int> &src,
    vector<int> &dst,
    vector<int> &weights,
    int &V)
{
    ifstream file(filepath);

    if (!file.is_open())
        return false;

    string line;
    V = 0;

    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        istringstream iss(line);

        int u, v, w;

        if (!(iss >> u >> v >> w))
            continue;

        src.push_back(u);
        dst.push_back(v);
        weights.push_back(w);

        V = max(V, max(u, v) + 1);
    }

    return true;
}

void dijkstra(
    int V,
    const vector<int> &src,
    const vector<int> &dst,
    const vector<int> &weights,
    int source,
    vector<int> &dist)
{
    vector<vector<pair<int, int>>> adj(V);

    for (size_t i = 0; i < src.size(); i++)
        adj[src[i]].push_back(make_pair(dst[i], weights[i]));

    dist.assign(V, INT_MAX);
    dist[source] = 0;

    priority_queue<
        pair<int, int>,
        vector<pair<int, int>>,
        greater<pair<int, int>>> pq;

    pq.push(make_pair(0, source));

    while (!pq.empty())
    {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d != dist[u])
            continue;

        for (size_t i = 0; i < adj[u].size(); i++)
        {
            int v = adj[u][i].first;
            int w = adj[u][i].second;

            long long candidate = (long long)d + w;

            if (candidate >= INT_MAX)
                continue;

            if (candidate < dist[v])
            {
                dist[v] = (int)candidate;
                pq.push(make_pair(dist[v], v));
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input.txt>\n";
        return 1;
    }

    vector<int> src, dst, weights;
    int V = 0;

    if (!file_to_array(argv[1], src, dst, weights, V))
    {
        cerr << "Could not read graph\n";
        return 1;
    }

    const int E = (int)src.size();
    const int source = 2677;
    const int slabsperwarp = 16;

    if (V <= source)
    {
        cerr << "Source is outside graph\n";
        return 1;
    }

    for (int w : weights)
    {
        if (w < 0)
        {
            cerr << "Dijkstra/SSSP requires non-negative weights\n";
            return 1;
        }
    }

    // Delete the first 10% of edges.
    const int deletedE = E / 10;
    const int remainingE = E - deletedE;

    vector<int> remainingSrc(src.begin() + deletedE, src.end());
    vector<int> remainingDst(dst.begin() + deletedE, dst.end());
    vector<int> remainingWeights(weights.begin() + deletedE, weights.end());

    cout << "Vertices             : " << V << endl;
    cout << "Total edges          : " << E << endl;
    cout << "Edges to delete      : " << deletedE << endl;
    cout << "Edges remaining      : " << remainingE << endl;

    // Allocate graph capacity using the complete graph's degrees.
    vector<int> degree(V, 0);

    for (int u : src)
        degree[u]++;

    AllocPolicy::DynamicAllocatorT allocator;
    Graph graph(V, degree.data(), allocator, 1.0f, 0);

    int *d_src = nullptr;
    int *d_dst = nullptr;
    int *d_weights = nullptr;

    int *d_current = nullptr;
    int *d_next = nullptr;
    int *d_prefix = nullptr;
    int *d_slabPrefix = nullptr;

    int *d_childOffsets = nullptr;
    int *d_degreeArray = nullptr;
    int *d_child = nullptr;

    distanceArray *d_distance = nullptr;

    checkCuda(cudaMalloc(&d_src, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_dst, E * sizeof(int)));
    checkCuda(cudaMalloc(&d_weights, E * sizeof(int)));

    checkCuda(cudaMalloc(&d_current, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_next, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_prefix, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_slabPrefix, V * sizeof(int)));

    checkCuda(cudaMalloc(&d_childOffsets, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_degreeArray, V * sizeof(int)));
    checkCuda(cudaMalloc(&d_child, V * sizeof(int)));

    checkCuda(cudaMalloc(&d_distance, V * sizeof(distanceArray)));

    checkCuda(cudaMemcpy(d_src, src.data(), E * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dst, dst.data(), E * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weights, weights.data(), E * sizeof(int),
                         cudaMemcpyHostToDevice));

    checkCuda(cudaMemset(d_current, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_next, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_prefix, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_slabPrefix, 0, V * sizeof(int)));

    checkCuda(cudaMemset(d_childOffsets, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_degreeArray, 0, V * sizeof(int)));
    checkCuda(cudaMemset(d_child, 0, V * sizeof(int)));

    thrust::fill(
        thrust::device,
        d_distance,
        d_distance + V,
        getWord(INT_MAX, -1));

    // Insert the complete graph.
    graph.insertEdges(d_src, d_dst, d_weights, E);
    checkCuda(cudaDeviceSynchronize());

    // Initialize source.
    checkCuda(cudaMemcpy(
        d_current,
        &source,
        sizeof(int),
        cudaMemcpyHostToDevice));

    distanceArray zero = getWord(0, -1);

    checkCuda(cudaMemcpy(
        d_distance + source,
        &zero,
        sizeof(distanceArray),
        cudaMemcpyHostToDevice));

    // Initial full-graph SSSP.
    SSSP<Graph, int, int>(
        graph,
        d_current,
        d_next,
        d_prefix,
        d_slabPrefix,
        d_distance,
        1,
        slabsperwarp);

    checkCuda(cudaDeviceSynchronize());

    cout << "\nInitial full-graph SSSP completed." << endl;

    // Decremental update: delete the first 10%.
    cudaEvent_t start, stop;

    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    checkCuda(cudaEventRecord(start));

    DecrementalSSSP<Graph, int, int>(
        graph,
        d_current,
        d_next,
        d_src,
        d_dst,
        d_prefix,
        d_slabPrefix,
        d_distance,
        d_child,
        d_childOffsets,
        deletedE,
        slabsperwarp);

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float gpuMs = 0.0f;

    checkCuda(cudaEventElapsedTime(&gpuMs, start, stop));

    cout << "Decremental GPU time : " << gpuMs << " ms" << endl;

    // Full CPU Dijkstra on the remaining graph.
    // This is only the correctness reference.
    vector<int> cpuDist;

    dijkstra(
        V,
        remainingSrc,
        remainingDst,
        remainingWeights,
        source,
        cpuDist);

    vector<distanceArray> gpuPacked(V);

    checkCuda(cudaMemcpy(
        gpuPacked.data(),
        d_distance,
        V * sizeof(distanceArray),
        cudaMemcpyDeviceToHost));

    int mismatch = 0;
    int reachabilityMismatch = 0;

    for (int v = 0; v < V; v++)
    {
        int gpuDist = getDistance(gpuPacked[v]);

        if ((gpuDist == INT_MAX) != (cpuDist[v] == INT_MAX))
        {
            reachabilityMismatch++;
        }
        else if (gpuDist != cpuDist[v])
        {
            mismatch++;
        }
    }

    cout << "Distance mismatches  : " << mismatch << endl;
    cout << "Reachability mismatch: " << reachabilityMismatch << endl;

    if (mismatch == 0 && reachabilityMismatch == 0)
        cout << "DECREMENTAL SSSP: PASS" << endl;
    else
        cout << "DECREMENTAL SSSP: FAIL" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_weights);

    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_prefix);
    cudaFree(d_slabPrefix);

    cudaFree(d_childOffsets);
    cudaFree(d_degreeArray);
    cudaFree(d_child);

    cudaFree(d_distance);

    return 0;
}
