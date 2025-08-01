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

#include "gpu_relu.h"
#include "gpu_lut.h"
#include "gpu_truncate.h"

/**
 * @brief FSS key structure for GELU multiplexer operation
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * This structure contains secret shares for the multiplexer (MUX)
 * operation used in GELU activation, which selects between two
 * precomputed values based on the input range.
 */
template <typename T>
struct GPUGeluMuxKey
{
    T *c0;  ///< Secret share of first set of MUX values
    T *c1;  ///< Secret share of second set of MUX values
};

/**
 * @brief Read a GPU GELU multiplexer key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @param N Number of elements in the GELU operation
 * @return GPUGeluMuxKey<T> Deserialized GELU MUX key
 * 
 * This function deserializes a GELU MUX key containing secret shares
 * for the multiplexer operation. The factor of 4 in memory size
 * accounts for multiple values per element needed in the MUX.
 */
template <typename T>
GPUGeluMuxKey<T> readGPUGeluMuxKey(uint8_t **key_as_bytes, int N)
{
    GPUGeluMuxKey<T> k;
    u64 memSz = 4 * N * sizeof(T); // num bytes
    k.c0 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    k.c1 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

/**
 * @brief FSS key structure for GPU GELU (Gaussian Error Linear Unit) activation
 * 
 * @tparam T Data type for the main computation (e.g., u32, u64)
 * @tparam TClip Data type for clipped values in MUX operation
 * 
 * GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * This structure contains all FSS keys needed for secure GELU computation:
 * 1. Truncation for scaling operations
 * 2. DReLU for sign detection
 * 3. Interval check mask for range detection
 * 4. MUX key for selecting precomputed values
 * 5. LUT key for non-linear function approximation
 * 6. Select key for final conditional multiplication
 */
template <typename T, typename TClip>
struct GPUGeluKey
{
    int bw;                         ///< Bit width of the computation
    GPUTruncateKey<T> trKey;        ///< Key for truncation operations
    GPUDReluKey dReluKey;           ///< Key for sign detection (x > 0)
    u32 *icMask;                    ///< Interval check mask for range detection
    GPUGeluMuxKey<TClip> muxKey;    ///< Key for multiplexer operation
    GPULUTKey<T> lutKey;            ///< Key for lookup table evaluation
    GPUSelectKey<T> reluSelectKey;  ///< Key for final selection/multiplication
};

/**
 * @brief Read a GPU GELU key from a byte stream
 * 
 * @tparam T Data type for the main computation
 * @tparam TClip Data type for clipped values in MUX operation
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUGeluKey<T, TClip> Deserialized GELU key
 * 
 * This function deserializes a complete GELU key from a byte stream,
 * extracting all component keys needed for secure GELU activation:
 * 
 * 1. Bit width configuration
 * 2. Truncation key with slack for intermediate computations
 * 3. DReLU key for sign detection
 * 4. Interval check mask for input range classification
 * 5. MUX key for conditional value selection
 * 6. LUT key for non-linear function approximation
 * 7. Select key for final output computation
 * 
 * The GELU activation is approximated using a combination of interval
 * checks and lookup tables for efficiency while maintaining accuracy.
 */
template <typename T, typename TClip>
GPUGeluKey<T, TClip> readGpuGeluKey(uint8_t **key_as_bytes)
{
    GPUGeluKey<T, TClip> k;
    k.bw = *((int *)*key_as_bytes);
    *key_as_bytes += sizeof(int);
    k.trKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    k.dReluKey = readGPUDReluKey(key_as_bytes);
    int N = k.trKey.N;
    // printf("###### Gelu N=%d\n", N);
    auto icMaskMemSize = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    k.icMask = (u32 *)*key_as_bytes;
    *key_as_bytes += icMaskMemSize;
    k.muxKey = readGPUGeluMuxKey<TClip>(key_as_bytes, N);
    k.lutKey = readGPULUTKey<T>(key_as_bytes);
    k.reluSelectKey = readGPUSelectKey<T>(key_as_bytes, N);
    return k;
}


#include "gpu_gelu.cu"