// Author: Neha Jawalkar
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

#include "gpu_select.h"
#include "gpu_dpf.h"

// using GPUDReluKey = GPUMaskedDCFKey;
using u32 = uint32_t;

/**
 * @brief FSS key structure for DReLU (comparison part of ReLU)
 * 
 * DReLU (Distributed ReLU) computes the comparison x > 0 using a DPF
 * to determine which values should be zeroed in the ReLU operation.
 */
struct GPUDReluKey
{
    GPUDPFKey dpfKey;  ///< DPF key for secure comparison with zero
    u32 *mask;         ///< Mask values for the comparison result
};

/**
 * @brief Complete FSS key structure for GPU ReLU operation
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * ReLU(x) = max(0, x) is implemented using:
 * 1. DReLU to compute comparison mask (x > 0)
 * 2. Secure selection to multiply x by the mask
 */
template <typename T>
struct GPUReluKey
{
    int bin;         ///< Input bit width
    int bout;        ///< Output bit width
    int numRelus;    ///< Number of ReLU operations in batch
    
    GPUDReluKey dreluKey;      ///< Key for comparison (x > 0)
    GPUSelectKey<T> selectKey;  ///< Key for secure selection based on comparison
};

/**
 * @brief Read a DReLU key from a byte stream
 * 
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUDReluKey Deserialized DReLU key
 * 
 * This function deserializes a DReLU key, extracting the DPF key
 * and mask values needed for the comparison operation.
 */
GPUDReluKey readGPUDReluKey(uint8_t **key_as_bytes)
{
    GPUDReluKey k;
    k.dpfKey = readGPUDPFKey(key_as_bytes);
    int N = k.dpfKey.M;
    k.mask = (uint32_t *)*key_as_bytes;
    // number of 32-bit integers * sizeof(int)
    // only works for bout = 1
    *key_as_bytes += ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

// const auto readGPUDReluWithDCFKey = readGPUMaskedDCFKey;

/**
 * @brief Read a complete ReLU key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUReluKey<T> Deserialized ReLU key
 * 
 * This function deserializes a complete ReLU key, including:
 * - Metadata (bit widths, number of operations)
 * - DReLU key for comparison
 * - Select key for conditional multiplication
 */
template <typename T>
GPUReluKey<T> readReluKey(uint8_t **key_as_bytes)
{
    GPUReluKey<T> k;
    memcpy(&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.dreluKey = readGPUDReluKey(key_as_bytes);
    k.selectKey = readGPUSelectKey<T>(key_as_bytes, k.numRelus);
    return k;
}

#include "gpu_relu.cu"