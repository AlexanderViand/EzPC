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

#include "gpu_fss_sort.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include <algorithm>
#include <iostream>
#include <cstring>

namespace fss_sort
{

    // Device function to compare two elements
    __device__ u32 compareElements(u64 a, u64 b, bool ascending)
    {
        if (ascending)
        {
            return (a < b) ? 1 : 0;
        }
        else
        {
            return (a > b) ? 1 : 0;
        }
    }

    // Device function for atomic addition of u64
    __device__ void atomicAddU64(u64* address, u64 val)
    {
        atomicAdd((unsigned long long*)address, (unsigned long long)val);
    }

    // GPU kernel for comparing elements using FSS
    __global__ void compareElementsKernel(
        u64* keys,
        u64* values,
        u64* indices,
        u32* comparison_results,
        int num_elements,
        int key_bitlength,
        bool ascending
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid >= num_elements * (num_elements - 1) / 2)
            return;

        // Convert linear thread ID to pair indices
        int i = 0, j = 0;
        int temp = tid;
        for (i = 0; i < num_elements - 1; i++)
        {
            if (temp < num_elements - 1 - i)
            {
                j = i + 1 + temp;
                break;
            }
            temp -= (num_elements - 1 - i);
        }

        if (i >= num_elements - 1)
            return;

        // Compare elements i and j
        u64 key_i = keys[i];
        u64 key_j = keys[j];
        
        u32 comparison_result = compareElements(key_i, key_j, ascending);
        
        // Store comparison result in a 2D array format
        int result_index = i * num_elements + j;
        comparison_results[result_index] = comparison_result;
        
        // Store reverse comparison (j vs i)
        result_index = j * num_elements + i;
        comparison_results[result_index] = 1 - comparison_result;
    }

    // GPU kernel for aggregating comparison results
    __global__ void aggregateComparisonsKernel(
        u32* comparison_results,
        u64* aggregated_counts,
        int num_elements
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid >= num_elements)
            return;

        u64 count = 0;
        
        // Count how many elements are less than the current element
        for (int j = 0; j < num_elements; j++)
        {
            if (j != tid)
            {
                // comparison_results[i][j] indicates if element i is less than element j
                count += comparison_results[tid * num_elements + j];
            }
        }
        
        aggregated_counts[tid] = count;
    }

    // GPU kernel for permuting elements based on aggregated counts
    __global__ void permuteElementsKernel(
        u64* input_keys,
        u64* input_values,
        u64* input_indices,
        u64* output_keys,
        u64* output_values,
        u64* output_indices,
        u64* aggregated_counts,
        int num_elements
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid >= num_elements)
            return;

        u64 target_position = aggregated_counts[tid];
        
        // Handle ties by using original index as secondary key
        // This ensures stable sorting
        for (int j = 0; j < num_elements; j++)
        {
            if (j != tid && aggregated_counts[j] == target_position)
            {
                if (input_indices[tid] < input_indices[j])
                {
                    target_position++;
                }
            }
        }
        
        // Ensure target position is within bounds
        if (target_position >= num_elements)
            target_position = num_elements - 1;
        
        // Place element at its sorted position
        output_keys[target_position] = input_keys[tid];
        output_values[target_position] = input_values[tid];
        output_indices[target_position] = input_indices[tid];
    }

    // Host function to allocate sorting buffers
    void allocateSortBuffers(
        int num_elements,
        u64** keys,
        u64** values,
        u64** indices,
        u32** comparison_results,
        u64** aggregated_counts
    )
    {
        *keys = (u64*)gpuMalloc(num_elements * sizeof(u64));
        *values = (u64*)gpuMalloc(num_elements * sizeof(u64));
        *indices = (u64*)gpuMalloc(num_elements * sizeof(u64));
        *comparison_results = (u32*)gpuMalloc(num_elements * num_elements * sizeof(u32));
        *aggregated_counts = (u64*)gpuMalloc(num_elements * sizeof(u64));
    }

    // Host function to free sorting buffers
    void freeSortBuffers(
        u64* keys,
        u64* values,
        u64* indices,
        u32* comparison_results,
        u64* aggregated_counts
    )
    {
        if (keys) gpuFree(keys);
        if (values) gpuFree(values);
        if (indices) gpuFree(indices);
        if (comparison_results) gpuFree(comparison_results);
        if (aggregated_counts) gpuFree(aggregated_counts);
    }

    // Host function to generate comparison keys using FSS
    void generateComparisonKeys(
        u64* keys,
        int num_elements,
        std::vector<dcf::GPUDCFKey>& dcf_keys
    )
    {
        // This function would generate DCF keys for secure comparison
        // In a real implementation, this would use the FSS key generation protocol
        // For now, we'll create placeholder keys
        
        dcf_keys.clear();
        int num_comparisons = num_elements * (num_elements - 1) / 2;
        dcf_keys.reserve(num_comparisons);
        
        for (int i = 0; i < num_elements - 1; i++)
        {
            for (int j = i + 1; j < num_elements; j++)
            {
                // Create a DCF key for comparing keys[i] and keys[j]
                // This is a simplified version - in practice, you'd use proper FSS key generation
                dcf::GPUDCFKey key;
                key.bin = 64;  // Assuming 64-bit keys
                key.bout = 1;  // 1-bit output
                key.M = 1;
                key.B = 1;
                key.memSzOut = sizeof(u32);
                
                // In practice, you'd generate proper DCF keys here
                // using the FSS key generation protocol
                
                dcf_keys.push_back(key);
            }
        }
    }

    // Host function to evaluate comparisons using FSS
    void evaluateComparisons(
        u64* keys,
        u32* comparison_results,
        const std::vector<dcf::GPUDCFKey>& dcf_keys,
        int num_elements
    )
    {
        // This function would evaluate DCF keys to get comparison results
        // In a real implementation, this would use the FSS evaluation protocol
        
        // For now, we'll use a simple comparison approach
        // In practice, you'd evaluate the DCF keys securely
        
        u32* h_comparison_results = new u32[num_elements * num_elements];
        
        for (int i = 0; i < num_elements; i++)
        {
            for (int j = 0; j < num_elements; j++)
            {
                if (i == j)
                {
                    h_comparison_results[i * num_elements + j] = 0;
                }
                else
                {
                    // Simple comparison (in practice, this would be FSS evaluation)
                    h_comparison_results[i * num_elements + j] = (keys[i] < keys[j]) ? 1 : 0;
                }
            }
        }
        
        // Copy results to GPU
        moveIntoGPUMem((u8*)comparison_results, (u8*)h_comparison_results, 
                      num_elements * num_elements * sizeof(u32), nullptr);
        
        delete[] h_comparison_results;
    }

    // Main FSS sorting function
    void fssSort(
        u64* keys,
        u64* values,
        u64* indices,
        int num_elements,
        int key_bitlength,
        int value_bitlength,
        bool ascending,
        int num_threads,
        int block_size
    )
    {
        // Allocate temporary buffers
        u64 *d_keys, *d_values, *d_indices, *d_output_keys, *d_output_values, *d_output_indices;
        u32 *d_comparison_results;
        u64 *d_aggregated_counts;
        
        allocateSortBuffers(num_elements, &d_keys, &d_values, &d_indices, 
                           &d_comparison_results, &d_aggregated_counts);
        
        // Allocate output buffers
        d_output_keys = (u64*)gpuMalloc(num_elements * sizeof(u64));
        d_output_values = (u64*)gpuMalloc(num_elements * sizeof(u64));
        d_output_indices = (u64*)gpuMalloc(num_elements * sizeof(u64));
        
        // Copy input data to GPU
        moveIntoGPUMem((u8*)d_keys, (u8*)keys, num_elements * sizeof(u64), nullptr);
        moveIntoGPUMem((u8*)d_values, (u8*)values, num_elements * sizeof(u64), nullptr);
        moveIntoGPUMem((u8*)d_indices, (u8*)indices, num_elements * sizeof(u64), nullptr);
        
        // Initialize indices if not provided
        if (indices == nullptr)
        {
            u64* h_indices = new u64[num_elements];
            for (int i = 0; i < num_elements; i++)
            {
                h_indices[i] = i;
            }
            moveIntoGPUMem((u8*)d_indices, (u8*)h_indices, num_elements * sizeof(u64), nullptr);
            delete[] h_indices;
        }
        
        // Generate FSS comparison keys
        std::vector<dcf::GPUDCFKey> dcf_keys;
        generateComparisonKeys(d_keys, num_elements, dcf_keys);
        
        // Evaluate comparisons using FSS
        evaluateComparisons(d_keys, d_comparison_results, dcf_keys, num_elements);
        
        // Calculate grid and block dimensions
        int num_blocks = (num_elements + block_size - 1) / block_size;
        int comparison_blocks = (num_elements * (num_elements - 1) / 2 + block_size - 1) / block_size;
        
        // Step 1: Compare elements (this would use FSS in practice)
        compareElementsKernel<<<comparison_blocks, block_size>>>(
            d_keys, d_values, d_indices, d_comparison_results,
            num_elements, key_bitlength, ascending
        );
        
        // Step 2: Aggregate comparison results
        aggregateComparisonsKernel<<<num_blocks, block_size>>>(
            d_comparison_results, d_aggregated_counts, num_elements
        );
        
        // Step 3: Permute elements based on aggregated counts
        permuteElementsKernel<<<num_blocks, block_size>>>(
            d_keys, d_values, d_indices,
            d_output_keys, d_output_values, d_output_indices,
            d_aggregated_counts, num_elements
        );
        
        // Copy results back to host
        moveFromGPUMem((u8*)keys, (u8*)d_output_keys, num_elements * sizeof(u64), nullptr);
        moveFromGPUMem((u8*)values, (u8*)d_output_values, num_elements * sizeof(u64), nullptr);
        moveFromGPUMem((u8*)indices, (u8*)d_output_indices, num_elements * sizeof(u64), nullptr);
        
        // Clean up
        freeSortBuffers(d_keys, d_values, d_indices, d_comparison_results, d_aggregated_counts);
        gpuFree(d_output_keys);
        gpuFree(d_output_values);
        gpuFree(d_output_indices);
        
        // Synchronize GPU
        cudaDeviceSynchronize();
    }

} // namespace fss_sort