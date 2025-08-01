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

#include "utils/gpu_random.h"

#include "gpu_truncate.h"
#include "gpu_window.h"
#include "gpu_lut.h"

/**
 * @brief FSS key structure for secure squaring operation
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * This structure contains the secret shares needed for computing x²
 * using the identity: x² = ((x+a)² - a²) / 2 - ax
 */
template <typename T>
struct GPUSqKey
{
    T *a;  ///< Secret share of random mask 'a'
    T *c;  ///< Secret share of precomputed value for squaring
};

/**
 * @brief Read a GPU squaring key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param N Number of elements to square
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUSqKey<T> Deserialized squaring key
 * 
 * This function deserializes a squaring key containing the secret shares
 * needed for secure multiplication of a value with itself.
 */
template <typename T>
GPUSqKey<T> readGPUSqKey(int N, u8** key_as_bytes)
{
    GPUSqKey<T> k;
    u64 memSz = N * sizeof(T);
    k.a = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    k.c = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}


/**
 * @brief FSS key structure for GPU layer normalization
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * Layer normalization computes: y = (x - μ) / σ * γ + β
 * where μ is the mean and σ is the standard deviation.
 * 
 * This structure contains all FSS keys needed for secure layer norm:
 * 1. Truncation key for computing the mean
 * 2. Squaring key for variance computation
 * 3. Window multiplication keys for normalization
 */
template <typename T>
struct GPULayerNormKey
{
    GPUTruncateKey<T> muTrKey;  ///< Key for truncating mean computation
    GPUSqKey<T> sqKey;          ///< Key for squaring (x - μ) for variance
    // GPULUTKey<T> invSqrtKey; ///< Key for computing 1/√σ (commented out)
    GPUMulKey<T> wMulKey1;      ///< Key for first window multiplication
    GPUMulKey<T> wMulKey2;      ///< Key for second window multiplication (transposed)
};

/**
 * @brief Transpose window parameters for layer normalization
 * 
 * @param p1 Original average pooling parameters
 * @return AvgPoolParams Transposed parameters
 * 
 * This function transposes the window dimensions used in layer normalization.
 * It converts a horizontal window (1 x W) to a vertical window (H x 1),
 * which is needed for the second pass of layer normalization.
 * 
 * @note Requires p1.FH == 1 (single row window)
 * @note Requires p1.FW == p1.imgW (full width window)
 * @note Requires p1.strideH == 1 and p1.strideW == p1.imgW
 */
AvgPoolParams transposeWindow(AvgPoolParams p1) {
    assert(p1.FH == 1 && p1.FW == p1.imgW && p1.strideH == 1 && p1.strideW == p1.imgW);
    AvgPoolParams p2;
    memcpy(&p2, &p1, sizeof(AvgPoolParams));
    p2.FH = p1.imgH;
    p2.FW = 1;
    p2.strideH = p1.imgH;
    p2.strideW = 1;
    initPoolParams(p2);
    return p2;
}

/**
 * @brief Read a GPU layer normalization key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param p Average pooling parameters defining the normalization window
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @param computeMu Whether to read the mean computation key (default: true)
 * @return GPULayerNormKey<T> Deserialized layer normalization key
 * 
 * This function deserializes a layer normalization key from a byte stream.
 * Layer normalization is typically applied over the feature dimension,
 * with the window parameters defining which dimensions to normalize over.
 * 
 * The function reads:
 * 1. Truncation key for mean computation (if computeMu is true)
 * 2. Squaring key for variance computation
 * 3. Window multiplication keys for both passes of normalization
 * 
 * @note The second window multiplication uses transposed parameters
 * to handle different normalization dimensions.
 */
template <typename T>
GPULayerNormKey<T> readGPULayerNormKey(AvgPoolParams p, u8** key_as_bytes, bool computeMu = true)
{
    GPULayerNormKey<T> k;
    auto inSz = getInSz(p);
    auto mSz = getMSz(p);
    if(computeMu) {
        k.muTrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
        // printf("Num Truncations=%d\n", k.muTrKey.N);
    }
    k.sqKey = readGPUSqKey<T>(inSz, key_as_bytes);
    // printf("Num sq=%ld\n", inSz);
    // k.invSqrtKey = readGPULUTKey<T>(key_as_bytes);
    k.wMulKey1 = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    auto p2 = transposeWindow(p);
    k.wMulKey2 = readGPUWindowMulKey<T>(p2, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    return k;
}

#include "gpu_layernorm.cu"
