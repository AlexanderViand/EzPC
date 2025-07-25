# FSS Sorting with Compare-and-Aggregate Approach

This implementation provides a secure sorting functionality using Function Secret Sharing (FSS) with the compare-and-aggregate approach, as described in the paper "Compare-and-Aggregate: A Framework for Secure Sorting".

## Overview

The compare-and-aggregate FSS sorting approach consists of three main phases:

1. **Compare Phase**: Generate and evaluate DCF (Distributed Comparison Function) keys for all pairwise comparisons
2. **Aggregate Phase**: Count how many elements are less than each element
3. **Permute Phase**: Rearrange elements based on their aggregated counts

## Key Features

- **Secure Sorting**: Uses FSS to perform secure comparisons without revealing the actual values
- **Compare-and-Aggregate**: Implements the efficient approach from the research paper
- **Stable Sorting**: Maintains relative order of elements with equal keys
- **Flexible Ordering**: Supports both ascending and descending sort orders
- **GPU Support**: Includes GPU-MPC implementation for high-performance sorting

## Implementation Details

### CPU Implementation (`FSS/src/sort.h`, `FSS/src/sort.cpp`)

The CPU implementation provides the core FSS sorting functionality:

```cpp
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
```

### GPU Implementation (`GPU-MPC/fss/gpu_fss_sort.h`, `GPU-MPC/fss/gpu_fss_sort.cu`)

The GPU implementation provides high-performance sorting using CUDA:

```cpp
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
```

## API Usage

### Basic Usage

```cpp
#include "api.h"

// Initialize arrays
GroupElement keys[num_elements];
GroupElement values[num_elements];
GroupElement indices[num_elements];
GroupElement keys_mask[num_elements];
GroupElement values_mask[num_elements];
GroupElement indices_mask[num_elements];

// Perform FSS sort
Sort(num_elements, key_bitlength, value_bitlength,
     MASK_PAIR(keys), MASK_PAIR(values), MASK_PAIR(indices),
     true, true); // ascending, stable
```

### Parameters

- `num_elements`: Number of elements to sort
- `key_bitlength`: Bit length of the key values
- `value_bitlength`: Bit length of the value data
- `keys`: Array of key values to sort by
- `values`: Array of associated values
- `indices`: Array to track original positions (for stable sorting)
- `ascending`: Sort order (true for ascending, false for descending)
- `stable_sort`: Whether to maintain relative order of equal elements

## Algorithm Details

### Compare Phase

1. Generate DCF keys for all pairwise comparisons (n*(n-1)/2 comparisons)
2. Evaluate DCF keys to get comparison results
3. Store results in a comparison matrix

### Aggregate Phase

1. For each element, count how many other elements are less than it
2. This count determines the final position in the sorted array
3. Handle ties using original indices for stable sorting

### Permute Phase

1. Create a mapping from original positions to sorted positions
2. Rearrange elements based on the mapping
3. Maintain associated values and indices

## Performance Characteristics

- **Time Complexity**: O(n²) for comparisons, O(n²) for aggregation, O(n log n) for permutation
- **Space Complexity**: O(n²) for comparison matrix
- **Communication**: O(n²) DCF evaluations
- **Parallelization**: GPU implementation supports parallel processing

## Security Properties

- **Privacy**: No information about the actual values is revealed during sorting
- **Correctness**: The sorting result is mathematically equivalent to a standard sort
- **Stability**: Relative order of equal elements is preserved

## Building and Testing

### Build the FSS Library

```bash
cd FSS/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../
make install
```

### Run Tests

```bash
cd FSS/tests/
g++ -std=c++17 -I../src -L../build/lib -lfss test_sort.cpp -o test_sort
./test_sort
```

### GPU Implementation

```bash
cd GPU-MPC/
make orca  # This will build the GPU implementation
```

## Example Output

```
=== Testing FSS Sort with Compare-and-Aggregate Approach ===
Original data:
  [0] Key: 42, Value: 0
  [1] Key: 17, Value: 1
  [2] Key: 89, Value: 2
  [3] Key: 5, Value: 3
  [4] Key: 23, Value: 4
  [5] Key: 67, Value: 5
  [6] Key: 11, Value: 6
  [7] Key: 34, Value: 7

Testing ascending sort...
Sorted data (ascending):
  [0] Key: 5, Value: 3, Original Index: 3
  [1] Key: 11, Value: 6, Original Index: 6
  [2] Key: 17, Value: 1, Original Index: 1
  [3] Key: 23, Value: 4, Original Index: 4
  [4] Key: 34, Value: 7, Original Index: 7
  [5] Key: 42, Value: 0, Original Index: 0
  [6] Key: 67, Value: 5, Original Index: 5
  [7] Key: 89, Value: 2, Original Index: 2

Sort verification: PASSED
```

## Integration with Existing FSS Framework

The sorting implementation integrates seamlessly with the existing FSS framework:

- Uses the same `GroupElement` type for data representation
- Follows the same masking pattern (`MASK_PAIR`) for secure computation
- Integrates with the existing DCF implementation for comparisons
- Compatible with the existing communication protocols

## Future Enhancements

- **Optimized DCF Generation**: Batch DCF key generation for better performance
- **Adaptive Sorting**: Choose sorting algorithm based on data characteristics
- **External Memory Sorting**: Support for sorting datasets larger than memory
- **Multi-party Sorting**: Extend to support more than two parties

## References

- "Compare-and-Aggregate: A Framework for Secure Sorting" - The research paper that inspired this implementation
- FSS Framework Documentation - For understanding the underlying FSS infrastructure
- GPU-MPC Documentation - For GPU acceleration details