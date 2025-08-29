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

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"
#include "gpu_stats.h"
#include "secure_error.h"
#include <cassert>
#include <string.h>

/*
 * SECURITY HARDENING MIGRATION GUIDE
 * ==================================
 * 
 * This file demonstrates how to migrate from standard CUDA error checking
 * to secure error handling that doesn't leak implementation details.
 * 
 * Migration Steps:
 * 1. Replace checkCudaErrors(call) with checkCudaErrorsSecure(call)
 * 2. Replace getLastCudaError(msg) with getLastCudaErrorSecure(msg)  
 * 3. Replace printLastCudaError(msg) with printLastCudaErrorSecure(msg)
 * 4. Add initSecureErrorLogging() at program start
 * 5. Add cleanupSecureErrorLogging() at program end
 * 
 * Security Benefits:
 * - Error details logged to syslog (admin-only access)
 * - Generic error messages shown to users/attackers
 * - Secure shutdown with GPU memory cleanup on critical errors
 * - Prevents information leakage through error messages
 * 
 * See examples in initGPUMemPool() function below.
 */

// #include <sys/types.h>

cudaMemPool_t mempool;

extern "C" void initGPUMemPool()
{
    // Initialize secure error logging
    initSecureErrorLogging("gpu-mpc-mempool");
    
    int isMemPoolSupported = 0;
    int device = 0;
    // is it okay to use device=0?
    
    // MIGRATION EXAMPLE 1: Replace checkCudaErrors with checkCudaErrorsSecure
    // OLD: checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
    //                                           cudaDevAttrMemoryPoolsSupported, device));
    // NEW: Use secure version that doesn't leak implementation details
    checkCudaErrorsSecure(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, device));
    // printf("%d\n", isMemPoolSupported);
    assert(isMemPoolSupported);
    /* implicitly assumes that the device is 0 */

    // MIGRATION EXAMPLE 2: Replace checkCudaErrors with checkCudaErrorsSecure
    // OLD: checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
    // NEW: Secure version
    checkCudaErrorsSecure(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    
    // MIGRATION EXAMPLE 3: Replace checkCudaErrors with checkCudaErrorsSecure
    // OLD: checkCudaErrors(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    // NEW: Secure version
    checkCudaErrorsSecure(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    uint64_t *d_dummy_ptr;
    
    // Get available GPU memory instead of hardcoding 20GB
    size_t free_mem, total_mem;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
    // Use 90% of available memory
    uint64_t bytes = (uint64_t)(free_mem * 0.9);
    printf("GPU Memory: Total=%zuGB, Free=%zuGB, Allocating=%zuGB\n", 
           total_mem >> 30, free_mem >> 30, bytes >> 30);
    
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
    uint64_t reserved_read, threshold_read;
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_read));
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold_read));
    printf("reserved memory: %lu %lu\n", reserved_read, threshold_read);
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}


extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}

// CUDA kernel for secure memory zeroing (defined here to avoid multiple definitions)
__global__ void secureZeroKernel(uint8_t* ptr, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ptr[idx] = 0;
    }
}

// Secure versions of memory management functions
extern "C" uint8_t *gpuMallocSecure(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrorsSecure(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}

extern "C" uint8_t *cpuMallocSecure(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    if (err != 0) {
        secureLogError("cpuMallocSecure::posix_memalign", err);
        return nullptr;
    }
    
    if (pin) {
        checkCudaErrorsSecure(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    }
    return h_a;
}

extern "C" void gpuFreeSecure(void *d_a, size_t size_bytes)
{
    if (d_a == nullptr) {
        return;
    }
    
    // Zero out the memory before freeing
    const int block_size = 256;
    int grid_size = (size_bytes + block_size - 1) / block_size;
    
    secureZeroKernel<<<grid_size, block_size>>>((uint8_t*)d_a, size_bytes);
    checkCudaErrorsSecure(cudaDeviceSynchronize());
    
    // Now free the memory
    checkCudaErrorsSecure(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFreeSecure(void *h_a, size_t size_bytes, bool pinned)
{
    if (h_a == nullptr) {
        return;
    }
    
    // Zero out the memory before freeing
    memset(h_a, 0, size_bytes);
    
    if (pinned) {
        checkCudaErrorsSecure(cudaHostUnregister(h_a));
    }
    free(h_a);
}

extern "C" uint8_t *moveToCPUSecure(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMallocSecure(size_in_bytes, true);
    if (h_a == nullptr) {
        return nullptr;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrorsSecure(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMemSecure(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrorsSecure(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMemSecure(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrorsSecure(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPUSecure(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMallocSecure(size_in_bytes);
    if (d_a == nullptr) {
        return nullptr;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrorsSecure(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
