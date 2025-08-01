// GPU Compare-Aggregate algorithms using SCMP
// Based on "Secure sorting and selection via function secret sharing" by Agarwal et al. (2024)

#pragma once

#include "utils/gpu_data_types.h"
#include "gpu_scmp.h"
#include <vector>
#include <cmath>

// Structure to hold comparison results for a batch
struct CompareAggregateResult {
    u64 *localRanks;    // Local rank for each element
    int numElements;
};

// Two-iteration maximum finding algorithm
void runTwoIterationMaximum(int party, u64 *elements, int n, 
                            GPUScmpKey *keys, int numKeys,
                            AESGlobalContext *gaes);

// Helper to perform batch comparisons and aggregate ranks
CompareAggregateResult compareAggregate(int party, u64 *elements, int n,
                                       const std::vector<std::pair<int, int>>& edges,
                                       GPUScmpKey *keys, int& keyIdx,
                                       AESGlobalContext *gaes);

#include "gpu_compare_aggregate.cu"