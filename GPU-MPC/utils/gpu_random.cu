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

#include "gpu_random.h"
#include "helper_cuda.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

// Global variables definitions (moved from original file)
curandGenerator_t gpuGen[8];
curandGenerator_t cpuGen[8];
curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW;
curandOrdering_t order = CURAND_ORDERING_PSEUDO_BEST;

#define CURAND_CHECK(call) do { \
    curandStatus_t err = call; \
    if (err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "CURAND error at %s:%d - %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// Non-template function implementations only
void randomUIntsOnGpu(const u64 n, u32 *d_data)
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandGenerate(gpuGen[device], d_data, n));
}

void randomUIntsOnCpu(const u64 n, u32 *h_data)
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandGenerate(cpuGen[device], h_data, n));
}

AESBlock *randomAESBlockOnGpu(const int n)
{
  AESBlock *d_data = (AESBlock *)gpuMalloc(n * sizeof(AESBlock));
  randomUIntsOnGpu(4 * n, (u32 *)d_data);
  return d_data;
}

void initGPURandomness()
{
  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 12345ULL;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandCreateGenerator(&(gpuGen[device]), CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetGeneratorOffset(gpuGen[device], offset));
  CURAND_CHECK(curandSetGeneratorOrdering(gpuGen[device], order));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gpuGen[device], seed));
}

void initCPURandomness()
{
  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 1234567890ULL;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  printf("CPU randomness, seed: %llu, offset: %llu\n", seed, offset);
  CURAND_CHECK(curandCreateGeneratorHost(&(cpuGen[device]), CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetGeneratorOffset(cpuGen[device], offset));
  CURAND_CHECK(curandSetGeneratorOrdering(cpuGen[device], order));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cpuGen[device], seed));
}

void destroyGPURandomness()
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandDestroyGenerator(gpuGen[device]));
}

void destroyCPURandomness()
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandDestroyGenerator(cpuGen[device]));
}