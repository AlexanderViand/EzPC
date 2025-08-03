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
#include "utils/gpu_mem.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_random.h"
#include <cassert>
#include <omp.h>

struct GPUAndKey {
    int N;
    uint32_t *b0, *b1, *b2;
};

// Non-template function declaration (implementation in gpu_and.cu)
GPUAndKey readGPUAndKey(uint8_t** key_as_bytes);

// Forward declaration of template function
template <typename T>
void writeAndKey(u8 **key_as_bytes, int party, int N, T *d_b0, T *d_b1, T *d_maskOut, int bout);

// Template function implementations (must be in header)
template <typename T>
void gpuAndKeyGen(int N, u8 **key_as_bytes, int party, int bin, AESGlobalContext *gaes)
{
    auto d_b0 = randomGEOnGpu<u32>(N, 1);
    auto d_b1 = randomGEOnGpu<u32>(N, 1);
    auto d_maskOut = randomGEOnGpu<u32>(N, 1);
    
    writeAndKey<T>(key_as_bytes, party, N, d_b0, d_b1, d_maskOut, 1);
    
    gpuFree(d_b0);
    gpuFree(d_b1);
    gpuFree(d_maskOut);
}

template <typename T>
void writeAndKey(u8 **key_as_bytes, int party, int N, T *d_b0, T *d_b1, T *d_maskOut, int bout)
{
    auto h_b0 = (T *)moveToCPU(d_b0, N * sizeof(T), NULL);
    auto h_b1 = (T *)moveToCPU(d_b1, N * sizeof(T), NULL);
    auto h_maskOut = (T *)moveToCPU(d_maskOut, N * sizeof(T), NULL);
    
    auto num_ints = (N - 1) / PACKING_SIZE + 1;
    auto size_in_bytes = sizeof(int) + 3 * num_ints * sizeof(uint32_t);
    
    auto key_as_u8 = (u8 *)malloc(size_in_bytes);
    *((int*)key_as_u8) = N;
    auto key_as_u32 = (uint32_t*)(key_as_u8 + sizeof(int));
    
    memcpy(key_as_u32, h_b0, num_ints * sizeof(uint32_t));
    key_as_u32 += num_ints;
    memcpy(key_as_u32, h_b1, num_ints * sizeof(uint32_t));
    key_as_u32 += num_ints;
    memcpy(key_as_u32, h_maskOut, num_ints * sizeof(uint32_t));
    
    key_as_bytes[party] = key_as_u8;
    
    free(h_b0);
    free(h_b1);
    free(h_maskOut);
}

template <typename T>
void gpuAnd(int party, int N, u8 *key_as_bytes, T *d_a, T *d_b, T *d_out, Stats *s)
{
    GPUAndKey key = readGPUAndKey(&key_as_bytes);
    
    auto d_b0 = (T *)moveToGPU((u8*)key.b0, (N - 1) / PACKING_SIZE + 1, s);
    auto d_b1 = (T *)moveToGPU((u8*)key.b1, (N - 1) / PACKING_SIZE + 1, s);
    auto d_b2 = (T *)moveToGPU((u8*)key.b2, (N - 1) / PACKING_SIZE + 1, s);
    
    // Perform AND operation
    gpuXor(d_out, d_a, d_b0, N, s);
    gpuXor(d_out, d_out, d_b, N, s);
    gpuXor(d_out, d_out, d_b1, N, s);
    gpuXor(d_out, d_out, d_b2, N, s);
    
    gpuFree(d_b0);
    gpuFree(d_b1);
    gpuFree(d_b2);
}