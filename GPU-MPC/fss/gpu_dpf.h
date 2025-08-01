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
#include "gpu_aes_shm.h"
#include "gpu_sstab.h"

// using u32 = u32;

/**
 * @brief GPU Distributed Point Function (DPF) tree key structure
 * 
 * This structure contains the keys needed for evaluating a DPF using
 * a tree-based construction. DPF allows two parties to secret-share
 * a point function that evaluates to 1 at a single point and 0 everywhere else.
 */
struct GPUDPFTreeKey
{
    int bin;      ///< Input bit width (domain is 2^bin)
    int N;        ///< Number of DPF evaluations
    int evalAll;  ///< If true, evaluate on all domain points; otherwise selective
    
    AESBlock *scw;  ///< Seed correction words for tree traversal
    AESBlock *l0;   ///< Left child seeds
    AESBlock *l1;   ///< Right child seeds
    u32 *tR;        ///< Right child control bits
    
    u64 szScw;      ///< Number of seed correction words
    u64 memSzScw;   ///< Memory size for seed correction words
    u64 memSzL;     ///< Memory size for child seeds (each)
    u64 memSzT;     ///< Memory size for control bits
    u64 memSzOut;   ///< Memory size for output
};

/**
 * @brief Main GPU DPF key structure
 * 
 * This structure encapsulates either a tree-based DPF key (for large domains)
 * or a table-based key (for small domains where bin <= 7).
 */
struct GPUDPFKey
{
    // if bin <= 7, populate ss, else ss = NULL
    int bin;  ///< Input bit width
    int M;    ///< Number of DPF evaluations
    int B;    ///< Number of batches
    
    u64 memSzOut;  ///< Total memory size for output
    
    GPUDPFTreeKey *dpfTreeKey;  ///< Tree-based keys (if bin > 7)
    GPUSSTabKey ssKey;          ///< Table-based key (if bin <= 7)
};

/**
 * @brief Read a GPU DPF tree key from a byte stream
 * 
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUDPFTreeKey Deserialized DPF tree key
 * 
 * This function deserializes a DPF tree key, extracting all the correction
 * words and control bits needed for tree-based DPF evaluation.
 */
GPUDPFTreeKey readGPUDPFTreeKey(u8 **key_as_bytes)
{
    GPUDPFTreeKey k;

    std::memcpy((char *)&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.szScw = k.N * (k.bin - LOG_AES_BLOCK_LEN);
    k.memSzScw = k.szScw * sizeof(AESBlock);
    k.scw = (AESBlock *)*key_as_bytes;

    *key_as_bytes += k.memSzScw;
    k.memSzL = k.N * sizeof(AESBlock);
    k.l0 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;
    k.l1 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;

    if (k.evalAll)
        k.memSzT = k.N * sizeof(u32);
    else
        k.memSzT = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (k.bin - LOG_AES_BLOCK_LEN);
    k.tR = (u32 *)*key_as_bytes;
    *key_as_bytes += k.memSzT;
    k.memSzOut = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

/**
 * @brief Read a GPU DPF key from a byte stream
 * 
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUDPFKey Deserialized DPF key
 * 
 * This function deserializes a DPF key, automatically choosing between
 * table-based (for small domains) and tree-based (for large domains)
 * representations based on the input bit width.
 * 
 * @note For bin <= 7, uses efficient table-based implementation
 * @note For bin > 7, uses tree-based construction with AES-based PRG
 */
GPUDPFKey readGPUDPFKey(u8 **key_as_bytes)
{
    GPUDPFKey k;
    k.bin = *((int *)*key_as_bytes);
    if (k.bin <= 7)
    {
        k.ssKey = readGPUSSTabKey(key_as_bytes);
        k.M = k.ssKey.N;
        k.B = 1;
        k.memSzOut = k.ssKey.memSzOut;
    }
    else
    {
        memcpy(&k, *key_as_bytes, 3 * sizeof(int));
        *key_as_bytes += (3 * sizeof(int));

        k.dpfTreeKey = new GPUDPFTreeKey[k.B];
        k.memSzOut = 0;
        for (int b = 0; b < k.B; b++)
        {
            k.dpfTreeKey[b] = readGPUDPFTreeKey(key_as_bytes);
            k.memSzOut += k.dpfTreeKey[b].memSzOut;
        }
    }
    return k;
}

/**
 * @brief Alias for reading DCF keys (DCF uses same structure as DPF)
 * 
 * Distributed Comparison Function (DCF) keys use the same underlying
 * structure as DPF keys, as DCF is built on top of DPF.
 */
const auto readGPUDcfKey = readGPUDPFKey;

#include "gpu_dpf.cu"