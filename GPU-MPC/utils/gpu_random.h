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

#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "misc_utils.cuh"
#include <curand.h>

// Non-template function declarations (implementations in gpu_random.cu)
void randomUIntsOnGpu(const u64 n, u32 *d_data);
void randomUIntsOnCpu(const u64 n, u32 *h_data);
AESBlock *randomAESBlockOnGpu(const int n);
void initGPURandomness();
void destroyGPURandomness();
void initCPURandomness();
void destroyCPURandomness();

// Template function implementations (must be in header for visibility)
template <typename T>
T *randomGEOnGpu(const u64 n, int bw)
{
  u64 numUInts = (n * sizeof(T) - 1) / (sizeof(u32)) + 1;
  auto d_data = (u32 *)gpuMalloc(numUInts * sizeof(u32));
  randomUIntsOnGpu(numUInts, d_data);
  modKernel<<<(n - 1) / 256 + 1, 256>>>(n, (T *)d_data, bw);
  return (T *)d_data;
}

template <typename T>
void randomGEOnCpu(const u64 n, int bw, T *h_data)
{
  u64 numUInts = (n * sizeof(T)) / (sizeof(u32));
  assert((n * sizeof(T)) % sizeof(u32) == 0);
  randomUIntsOnCpu(numUInts, (u32 *)h_data);
  if (bw < sizeof(T) * 8)
  {
    for (u64 i = 0; i < n; i++)
    {
      h_data[i] &= ((T(1) << bw) - 1);
    }
  }
}

template <typename T>
T *randomGEOnCpu(const u64 n, int bw)
{
  auto h_data = (T *)cpuMalloc(n * sizeof(T));
  randomGEOnCpu(n, bw, h_data);
  return h_data;
}

template <typename TIn, typename TOut>
void writeShares(u8 **key_as_bytes, int party, u64 N, TIn *d_A, int bw, bool randomShares = true)
{
  assert(bw <= 8 * sizeof(TOut));
  TOut *d_A0 = NULL;
  if (randomShares)
    d_A0 = randomGEOnGpu<TOut>(N, bw);
  
  size_t memSzA;
  if (bw == 1 || bw == 2)
  {
    auto numInts = ((bw * N - 1) / PACKING_SIZE + 1);
    memSzA = numInts * PACKING_SIZE / 8;
    auto d_A_packed = (u8 *)gpuMalloc(memSzA);
    packKernel<<<(numInts - 1) / 256 + 1, 256>>>(bw, N, (TOut *)d_A, d_A_packed);
    auto h_A_packed = (u8 *)moveToCPU(d_A_packed, memSzA, NULL);
    gpuFree(d_A_packed);
    key_as_bytes[party] = h_A_packed;
  }
  else
  {
    memSzA = N * sizeof(TOut);
    key_as_bytes[party] = (u8 *)moveToCPU((u8 *)d_A, memSzA, NULL);
  }
  
  if (randomShares)
    gpuFree(d_A0);
}

template <typename T>
T *getMaskedInputOnGpu(int N, int bw, T *d_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0)
{
  auto d_I = (T *)moveToGPU((u8 *)*h_I, N * sizeof(T), NULL);
  gpuLinearComb(bw, N, d_I, T(1), d_I, T(-1), d_mask_I);
  return d_I;
}

template <typename T>
T *getMaskedInputOnCpu(int N, int bw, T *h_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0)
{
  auto h_maskedI = (T *)cpuMalloc(N * sizeof(T));
  for (int i = 0; i < N; i++)
  {
    h_maskedI[i] = (*h_I)[i] - h_mask_I[i];
    if (bw < sizeof(T) * 8)
    {
      gpuMod(h_maskedI[i], bw);
    }
  }
  return h_maskedI;
}