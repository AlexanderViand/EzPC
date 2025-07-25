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

#pragma once

#include "group_element.h"
#include "dcf.h"
#include <vector>
#include <cassert>

// FSS-based sorting using compare-and-aggregate approach
// This implements the approach from the paper: "Compare-and-Aggregate: A Framework for Secure Sorting"

// Structure for sorting configuration
struct SortConfig
{
    int num_elements;
    int key_bitlength;
    int value_bitlength;
    bool ascending;
    bool stable_sort;
};

// Structure for comparison result
struct ComparisonResult
{
    GroupElement comparison_bit;  // 0 or 1 indicating comparison result
    GroupElement aggregated_count;  // Count of elements less than current
};

// Main FSS sorting function using compare-and-aggregate approach
void fssSort(
    int num_elements,
    int key_bitlength,
    int value_bitlength,
    MASK_PAIR(GroupElement *keys),
    MASK_PAIR(GroupElement *values),
    MASK_PAIR(GroupElement *indices),
    bool ascending = true,
    bool stable_sort = true
);

// Compare phase: Generate and evaluate DCF keys for all pairwise comparisons
void comparePhase(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    GroupElement *comparison_results,
    bool ascending
);

// Aggregate phase: Count how many elements are less than each element
void aggregatePhase(
    int num_elements,
    GroupElement *comparison_results,
    GroupElement *aggregated_counts
);

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
);

// Generate DCF keys for all pairwise comparisons
std::vector<std::pair<DCFKeyPack, DCFKeyPack>> generateComparisonKeys(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    bool ascending
);

// Evaluate DCF keys to get comparison results
void evaluateComparisonKeys(
    int num_elements,
    int key_bitlength,
    MASK_PAIR(GroupElement *keys),
    const std::vector<std::pair<DCFKeyPack, DCFKeyPack>>& key_pairs,
    GroupElement *comparison_results,
    bool ascending
);

// Helper function to convert linear index to pair indices
void linearToPairIndices(int linear_index, int num_elements, int& i, int& j);

// Helper function to convert pair indices to linear index
int pairToLinearIndex(int i, int j, int num_elements);

// Memory management functions
void allocateSortBuffers(
    int num_elements,
    GroupElement **comparison_results,
    GroupElement **aggregated_counts,
    GroupElement **output_keys,
    GroupElement **output_values,
    GroupElement **output_indices
);

void freeSortBuffers(
    GroupElement *comparison_results,
    GroupElement *aggregated_counts,
    GroupElement *output_keys,
    GroupElement *output_values,
    GroupElement *output_indices
);

// Utility functions for sorting
void initializeIndices(int num_elements, MASK_PAIR(GroupElement *indices));
void copyArray(int size, MASK_PAIR(GroupElement *src), MASK_PAIR(GroupElement *dst));