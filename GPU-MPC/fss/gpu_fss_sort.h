// Author: AI Assistant
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "utils/gpu_data_types.h"
#include "utils/misc_utils.h"
#include "dcf/gpu_dcf.h"
#include <vector>
#include <cassert>

namespace fss_sort
{
    // Structure to hold sorting keys and data
    struct SortKeyValue
    {
        u64 key;
        u64 value;
        u64 index;  // Original position for tracking
    };

    // Structure for FSS sorting configuration
    struct FSSSortConfig
    {
        int num_elements;
        int key_bitlength;
        int value_bitlength;
        int num_threads;
        int block_size;
        bool ascending;  // true for ascending, false for descending
    };

    // Structure for comparison results
    struct ComparisonResult
    {
        u32 comparison_bit;  // 0 or 1 indicating comparison result
        u64 aggregated_count;  // Count of elements less than current
    };

    // GPU kernel declarations
    __global__ void compareElementsKernel(
        u64* keys,
        u64* values,
        u64* indices,
        u32* comparison_results,
        int num_elements,
        int key_bitlength,
        bool ascending
    );

    __global__ void aggregateComparisonsKernel(
        u32* comparison_results,
        u64* aggregated_counts,
        int num_elements
    );

    __global__ void permuteElementsKernel(
        u64* input_keys,
        u64* input_values,
        u64* input_indices,
        u64* output_keys,
        u64* output_values,
        u64* output_indices,
        u64* aggregated_counts,
        int num_elements
    );

    // Host functions for FSS sorting
    void fssSort(
        u64* keys,
        u64* values,
        u64* indices,
        int num_elements,
        int key_bitlength,
        int value_bitlength,
        bool ascending = true,
        int num_threads = 256,
        int block_size = 256
    );

    // Helper functions
    void generateComparisonKeys(
        u64* keys,
        int num_elements,
        std::vector<dcf::GPUDCFKey>& dcf_keys
    );

    void evaluateComparisons(
        u64* keys,
        u32* comparison_results,
        const std::vector<dcf::GPUDCFKey>& dcf_keys,
        int num_elements
    );

    // Utility functions
    __device__ u32 compareElements(u64 a, u64 b, bool ascending);
    __device__ void atomicAddU64(u64* address, u64 val);
    
    // Memory management
    void allocateSortBuffers(
        int num_elements,
        u64** keys,
        u64** values,
        u64** indices,
        u32** comparison_results,
        u64** aggregated_counts
    );

    void freeSortBuffers(
        u64* keys,
        u64* values,
        u64* indices,
        u32* comparison_results,
        u64* aggregated_counts
    );
}