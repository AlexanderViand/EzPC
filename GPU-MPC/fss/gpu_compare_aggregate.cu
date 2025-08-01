// GPU Compare-Aggregate algorithms using SCMP
// Based on "Secure sorting and selection via function secret sharing" by Agarwal et al. (2024)

#include "gpu_compare_aggregate.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

// Kernel to compute local ranks from comparison results
__global__ void computeLocalRanksKernel(u64 *localRanks, u32 *comparisonResults, 
                                       int *edgeList, int numEdges, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        localRanks[tid] = 0;
    }
    __syncthreads();
    
    if (tid < numEdges) {
        int i = edgeList[2 * tid];
        int j = edgeList[2 * tid + 1];
        u32 cmp = comparisonResults[tid];
        
        // If element i >= element j, increment rank of i
        if (cmp == 1) {
            atomicAdd((unsigned long long*)&localRanks[i], 1ULL);
        } else {
            atomicAdd((unsigned long long*)&localRanks[j], 1ULL);
        }
    }
}

// Helper function to create edge list on GPU
int* createEdgeListGPU(const std::vector<std::pair<int, int>>& edges) {
    int numEdges = edges.size();
    int *h_edges = new int[2 * numEdges];
    for (int i = 0; i < numEdges; i++) {
        h_edges[2 * i] = edges[i].first;
        h_edges[2 * i + 1] = edges[i].second;
    }
    
    int *d_edges;
    checkCudaErrors(cudaMalloc(&d_edges, 2 * numEdges * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_edges, h_edges, 2 * numEdges * sizeof(int), 
                               cudaMemcpyHostToDevice));
    delete[] h_edges;
    return d_edges;
}

// Perform batch comparisons and aggregate ranks
CompareAggregateResult compareAggregate(int party, u64 *d_elements, int n,
                                       const std::vector<std::pair<int, int>>& edges,
                                       GPUScmpKey *keys, int& keyIdx,
                                       AESGlobalContext *gaes) {
    CompareAggregateResult result;
    result.numElements = n;
    result.localRanks = (u64*)gpuMalloc(n * sizeof(u64));
    
    if (edges.empty()) {
        checkCudaErrors(cudaMemset(result.localRanks, 0, n * sizeof(u64)));
        return result;
    }
    
    int numEdges = edges.size();
    
    // Create edge list on GPU
    int *d_edges = createEdgeListGPU(edges);
    
    // Allocate memory for all comparison results
    u32 *d_allCompResults = (u32*)gpuMalloc(numEdges * sizeof(u32));
    
    // Perform comparisons for each edge
    for (int e = 0; e < numEdges; e++) {
        int i = edges[e].first;
        int j = edges[e].second;
        
        // Get elements to compare
        u64 h_xi, h_xj;
        checkCudaErrors(cudaMemcpy(&h_xi, d_elements + i, sizeof(u64), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&h_xj, d_elements + j, sizeof(u64), cudaMemcpyDeviceToHost));
        
        // Perform secure comparison using pre-generated key
        u64 h_result;
        evalGPUSCMP(party, &h_result, h_xi, h_xj, keys[keyIdx++], 1, gaes);
        
        // Store result
        u32 compResult = (u32)(h_result & 1);
        checkCudaErrors(cudaMemcpy(d_allCompResults + e, &compResult, sizeof(u32), 
                                   cudaMemcpyHostToDevice));
    }
    
    // Compute local ranks using GPU kernel
    int blockSize = 256;
    int gridSize = (std::max(n, numEdges) + blockSize - 1) / blockSize;
    computeLocalRanksKernel<<<gridSize, blockSize>>>(result.localRanks, d_allCompResults, 
                                                      d_edges, numEdges, n);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup
    checkCudaErrors(cudaFree(d_edges));
    gpuFree(d_allCompResults);
    
    return result;
}

// Two-iteration maximum finding algorithm
void runTwoIterationMaximum(int party, u64 *d_elements, int n, 
                            GPUScmpKey *keys, int numKeys,
                            AESGlobalContext *gaes) {
    printf("\n=== Running Two-Iteration Maximum Algorithm (Party %d) ===\n", party);
    printf("Number of elements: %d\n", n);
    
    if (n == 0) {
        printf("Empty array, no maximum\n");
        return;
    }
    if (n == 1) {
        u64 h_elem;
        checkCudaErrors(cudaMemcpy(&h_elem, d_elements, sizeof(u64), cudaMemcpyDeviceToHost));
        printf("Single element: %lu\n", h_elem);
        return;
    }
    
    // Line 1: Let t = n^(2/3) / 2^(1/3)
    int t = std::max(1, (int)(std::pow(n, 2.0/3.0) / std::pow(2, 1.0/3.0)));
    if (t > n) t = n;
    printf("Number of partitions (t): %d\n", t);
    
    // Iteration 1: Split into t partitions and find max in each
    printf("\n--- Iteration 1: Finding max in each partition ---\n");
    
    // Create partitions
    std::vector<std::vector<int>> partitions(t);
    int baseSize = n / t;
    int remainder = n % t;
    int currentIdx = 0;
    
    for (int i = 0; i < t; i++) {
        int partSize = baseSize + (i < remainder ? 1 : 0);
        for (int j = 0; j < partSize; j++) {
            partitions[i].push_back(currentIdx++);
        }
    }
    
    // Create edges for cliques within each partition
    std::vector<std::pair<int, int>> H1_edges;
    for (const auto& part : partitions) {
        for (size_t i = 0; i < part.size(); i++) {
            for (size_t j = i + 1; j < part.size(); j++) {
                H1_edges.push_back({part[i], part[j]});
            }
        }
    }
    printf("Number of edges in H1: %lu\n", H1_edges.size());
    
    // Perform Compare-Aggregate for iteration 1
    int keyIdx = 0;
    CompareAggregateResult iter1Result = compareAggregate(party, d_elements, n, 
                                                          H1_edges, keys, keyIdx, gaes);
    
    // Find element with max rank in each partition
    u64 *h_localRanks = new u64[n];
    checkCudaErrors(cudaMemcpy(h_localRanks, iter1Result.localRanks, n * sizeof(u64), 
                               cudaMemcpyDeviceToHost));
    
    std::vector<int> maxIndices;
    for (const auto& part : partitions) {
        if (part.empty()) continue;
        
        int maxIdx = part[0];
        u64 maxRank = h_localRanks[part[0]];
        
        for (int idx : part) {
            if (h_localRanks[idx] > maxRank) {
                maxRank = h_localRanks[idx];
                maxIdx = idx;
            }
        }
        maxIndices.push_back(maxIdx);
    }
    
    printf("Selected %lu elements for iteration 2\n", maxIndices.size());
    
    // Iteration 2: Find max among partition maxima
    printf("\n--- Iteration 2: Finding global maximum ---\n");
    
    // Copy selected elements to new array
    u64 *d_iter2Elements = (u64*)gpuMalloc(maxIndices.size() * sizeof(u64));
    u64 *h_iter2Elements = new u64[maxIndices.size()];
    
    for (size_t i = 0; i < maxIndices.size(); i++) {
        checkCudaErrors(cudaMemcpy(&h_iter2Elements[i], d_elements + maxIndices[i], 
                                   sizeof(u64), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaMemcpy(d_iter2Elements, h_iter2Elements, 
                               maxIndices.size() * sizeof(u64), cudaMemcpyHostToDevice));
    
    // Create complete graph for iteration 2
    std::vector<std::pair<int, int>> H2_edges;
    for (size_t i = 0; i < maxIndices.size(); i++) {
        for (size_t j = i + 1; j < maxIndices.size(); j++) {
            H2_edges.push_back({i, j});
        }
    }
    printf("Number of edges in H2: %lu\n", H2_edges.size());
    
    // Perform Compare-Aggregate for iteration 2
    CompareAggregateResult iter2Result = compareAggregate(party, d_iter2Elements, 
                                                          maxIndices.size(), H2_edges, 
                                                          keys, keyIdx, gaes);
    
    // Find element with maximum rank
    u64 *h_iter2Ranks = new u64[maxIndices.size()];
    checkCudaErrors(cudaMemcpy(h_iter2Ranks, iter2Result.localRanks, 
                               maxIndices.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    
    int finalMaxIdx = 0;
    u64 maxFinalRank = h_iter2Ranks[0];
    for (size_t i = 1; i < maxIndices.size(); i++) {
        if (h_iter2Ranks[i] > maxFinalRank) {
            maxFinalRank = h_iter2Ranks[i];
            finalMaxIdx = i;
        }
    }
    
    // Get the actual maximum element
    u64 maxElement = h_iter2Elements[finalMaxIdx];
    int originalIdx = maxIndices[finalMaxIdx];
    
    printf("\n=== Results ===\n");
    printf("Maximum element (Party %d share): %lu\n", party, maxElement);
    printf("Original index: %d\n", originalIdx);
    printf("Total SCMP keys used: %d out of %d\n", keyIdx, numKeys);
    
    // Cleanup
    delete[] h_localRanks;
    delete[] h_iter2Elements;
    delete[] h_iter2Ranks;
    gpuFree(d_iter2Elements);
    gpuFree(iter1Result.localRanks);
    gpuFree(iter2Result.localRanks);
}