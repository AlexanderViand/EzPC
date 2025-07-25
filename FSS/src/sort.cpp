/*
Authors: AI Assistant
Copyright:
Copyright (c) 2024 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "sort.h"
#include "add.h"
#include "mult.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <cstring>

// Helper function to convert linear index to pair indices
void linearToPairIndices(int linear_index, int num_elements, int& i, int& j)
{
    // Convert linear index to (i,j) pair where i < j
    // For n elements, we have n*(n-1)/2 comparisons
    int temp = linear_index;
    for (i = 0; i < num_elements - 1; i++)
    {
        if (temp < num_elements - 1 - i)
        {
            j = i + 1 + temp;
            return;
        }
        temp -= (num_elements - 1 - i);
    }
    // Should not reach here
    i = num_elements - 2;
    j = num_elements - 1;
}

// Helper function to convert pair indices to linear index
int pairToLinearIndex(int i, int j, int num_elements)
{
    if (i >= j) return -1; // Invalid pair
    return i * (num_elements - 1) - (i * (i + 1)) / 2 + (j - i - 1);
}

// Memory management functions
void allocateSortBuffers(
    int num_elements,
    GroupElement **comparison_results,
    GroupElement **aggregated_counts,
    GroupElement **output_keys,
    GroupElement **output_values,
    GroupElement **output_indices
)
{
    *comparison_results = new GroupElement[num_elements * num_elements];
    *aggregated_counts = new GroupElement[num_elements];
    *output_keys = new GroupElement[num_elements];
    *output_values = new GroupElement[num_elements];
    *output_indices = new GroupElement[num_elements];
}

void freeSortBuffers(
    GroupElement *comparison_results,
    GroupElement *aggregated_counts,
    GroupElement *output_keys,
    GroupElement *output_values,
    GroupElement *output_indices
)
{
    if (comparison_results) delete[] comparison_results;
    if (aggregated_counts) delete[] aggregated_counts;
    if (output_keys) delete[] output_keys;
    if (output_values) delete[] output_values;
    if (output_indices) delete[] output_indices;
}

// Utility functions
void initializeIndices(int num_elements, MASK_PAIR(GroupElement *indices))
{
    for (int i = 0; i < num_elements; i++)
    {
        indices[i] = GroupElement(i, 64);
        indices_mask[i] = GroupElement(0, 64);
    }
}

void copyArray(int size, MASK_PAIR(GroupElement *src), MASK_PAIR(GroupElement *dst))
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = src[i];
        dst_mask[i] = src_mask[i];
    }
}

// Generate DCF keys for all pairwise comparisons
std::vector<std::pair<DCFKeyPack, DCFKeyPack>> generateComparisonKeys(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    bool ascending
)
{
    std::vector<std::pair<DCFKeyPack, DCFKeyPack>> key_pairs;
    int num_comparisons = num_elements * (num_elements - 1) / 2;
    key_pairs.reserve(num_comparisons);
    
    for (int i = 0; i < num_elements - 1; i++)
    {
        for (int j = i + 1; j < num_elements; j++)
        {
            // Create DCF keys for comparing keys[i] and keys[j]
            // The DCF will output 1 if keys[i] < keys[j] (for ascending)
            // or 1 if keys[i] > keys[j] (for descending)
            
            GroupElement threshold = keys[j] - keys[i];
            if (!ascending)
            {
                threshold = keys[i] - keys[j];
            }
            
            // Generate DCF keys for the comparison
            GroupElement payload = GroupElement(1, 1); // 1-bit output
            auto key_pair = keyGenDCF(key_bitlength, 1, threshold, payload);
            key_pairs.push_back(key_pair);
        }
    }
    
    return key_pairs;
}

// Evaluate DCF keys to get comparison results
void evaluateComparisonKeys(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    const std::vector<std::pair<DCFKeyPack, DCFKeyPack>>& key_pairs,
    GroupElement *comparison_results,
    bool ascending
)
{
    int key_index = 0;
    
    for (int i = 0; i < num_elements; i++)
    {
        for (int j = 0; j < num_elements; j++)
        {
            if (i == j)
            {
                // Element compared to itself
                comparison_results[i * num_elements + j] = GroupElement(0, 1);
            }
            else if (i < j)
            {
                // Use the DCF key for this pair
                GroupElement result;
                evalDCF(party, &result, keys[i], key_pairs[key_index].first);
                comparison_results[i * num_elements + j] = result;
                key_index++;
            }
            else
            {
                // j < i, use the reverse comparison
                GroupElement result;
                evalDCF(party, &result, keys[j], key_pairs[pairToLinearIndex(j, i, num_elements)].first);
                comparison_results[i * num_elements + j] = GroupElement(1, 1) - result;
            }
        }
    }
}

// Compare phase: Generate and evaluate DCF keys for all pairwise comparisons
void comparePhase(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    GroupElement *comparison_results,
    bool ascending
)
{
    // Generate DCF keys for all pairwise comparisons
    auto key_pairs = generateComparisonKeys(num_elements, key_bitlength, keys, ascending);
    
    // Evaluate the DCF keys to get comparison results
    evaluateComparisonKeys(num_elements, key_bitlength, keys, key_pairs, comparison_results, ascending);
}

// Aggregate phase: Count how many elements are less than each element
void aggregatePhase(
    int num_elements,
    GroupElement *comparison_results,
    GroupElement *aggregated_counts
)
{
    for (int i = 0; i < num_elements; i++)
    {
        GroupElement count = GroupElement(0, 64);
        
        // Count how many elements are less than element i
        for (int j = 0; j < num_elements; j++)
        {
            if (j != i)
            {
                // comparison_results[i][j] indicates if element i is less than element j
                // We want to count elements less than i, so we look at comparison_results[j][i]
                count = count + comparison_results[j * num_elements + i];
            }
        }
        
        aggregated_counts[i] = count;
    }
}

// Permute phase: Rearrange elements based on their aggregated counts
void permutePhase(
    int num_elements,
    int key_bitlength,
    int value_bitlength,
    MASK_PAIR(GroupElement *input_keys),
    MASK_PAIR(GroupElement *input_values),
    MASK_PAIR(GroupElement *input_indices),
    MASK_PAIR(GroupElement *output_keys),
    MASK_PAIR(GroupElement *output_values),
    MASK_PAIR(GroupElement *output_indices),
    GroupElement *aggregated_counts,
    bool stable_sort
)
{
    // Create a mapping from original positions to sorted positions
    std::vector<std::pair<GroupElement, int>> position_mapping;
    position_mapping.reserve(num_elements);
    
    for (int i = 0; i < num_elements; i++)
    {
        GroupElement target_position = aggregated_counts[i];
        
        // Handle ties for stable sorting
        if (stable_sort)
        {
            for (int j = 0; j < num_elements; j++)
            {
                if (j != i && aggregated_counts[j] == target_position)
                {
                    // Use original index as tie-breaker
                    if (input_indices[i] < input_indices[j])
                    {
                        target_position = target_position + GroupElement(1, 64);
                    }
                }
            }
        }
        
        position_mapping.push_back({target_position, i});
    }
    
    // Sort the mapping by target position
    std::sort(position_mapping.begin(), position_mapping.end(),
              [](const std::pair<GroupElement, int>& a, const std::pair<GroupElement, int>& b) {
                  return a.first < b.first;
              });
    
    // Rearrange elements based on the mapping
    for (int i = 0; i < num_elements; i++)
    {
        int original_index = position_mapping[i].second;
        output_keys[i] = input_keys[original_index];
        output_values[i] = input_values[original_index];
        output_indices[i] = input_indices[original_index];
        
        // Also copy masks
        output_keys_mask[i] = input_keys_mask[original_index];
        output_values_mask[i] = input_values_mask[original_index];
        output_indices_mask[i] = input_indices_mask[original_index];
    }
}

// Main FSS sorting function using compare-and-aggregate approach
void fssSort(
    int num_elements,
    int key_bitlength,
    int value_bitlength,
    MASK_PAIR(GroupElement *keys),
    MASK_PAIR(GroupElement *values),
    MASK_PAIR(GroupElement *indices),
    bool ascending,
    bool stable_sort
)
{
    // Initialize indices if not provided
    if (indices == nullptr)
    {
        indices = new GroupElement[num_elements];
        indices_mask = new GroupElement[num_elements];
        initializeIndices(num_elements, MASK_PAIR(indices));
    }
    
    // Allocate temporary buffers
    GroupElement *comparison_results, *aggregated_counts;
    GroupElement *output_keys, *output_values, *output_indices;
    
    allocateSortBuffers(num_elements, &comparison_results, &aggregated_counts,
                       &output_keys, &output_values, &output_indices);
    
    // Step 1: Compare phase - Generate and evaluate DCF keys
    comparePhase(num_elements, key_bitlength, MASK_PAIR(keys), comparison_results, ascending);
    
    // Step 2: Aggregate phase - Count elements less than each element
    aggregatePhase(num_elements, comparison_results, aggregated_counts);
    
    // Step 3: Permute phase - Rearrange elements based on aggregated counts
    permutePhase(num_elements, key_bitlength, value_bitlength,
                MASK_PAIR(keys), MASK_PAIR(values), MASK_PAIR(indices),
                MASK_PAIR(output_keys), MASK_PAIR(output_values), MASK_PAIR(output_indices),
                aggregated_counts, stable_sort);
    
    // Copy results back to input arrays
    copyArray(num_elements, MASK_PAIR(output_keys), MASK_PAIR(keys));
    copyArray(num_elements, MASK_PAIR(output_values), MASK_PAIR(values));
    copyArray(num_elements, MASK_PAIR(output_indices), MASK_PAIR(indices));
    
    // Clean up
    freeSortBuffers(comparison_results, aggregated_counts, output_keys, output_values, output_indices);
}