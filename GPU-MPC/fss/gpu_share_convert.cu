// Authors: GPU-MPC Authors
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

// Share conversion implementations

#include "gpu_lss.h"
// Don't include DCF headers directly to avoid duplicate symbols
// #include "dcf/gpu_dcf.cuh"
// #include "gpu_dpf.cuh"
// #include "gpu_scmp.h"

// ============== A2B (Arithmetic to Binary) Conversion ==============

template <typename T>
__global__ void extractBitsKernel(int bw, u64 N, T* d_values, u32* d_bits, int bitPos) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N) {
        T bit = (d_values[i] >> bitPos) & 1;
        // Pack bits into u32 array
        int packIdx = i / 32;
        int packBit = i % 32;
        atomicOr(&d_bits[packIdx], (u32(bit) << packBit));
    }
}

template <typename T>
__global__ void combineBitsKernel(u64 N, u32** d_bitArrays, int numBits, T* d_result) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N) {
        T value = 0;
        for (int b = 0; b < numBits; b++) {
            int packIdx = i / 32;
            int packBit = i % 32;
            u32 bit = (d_bitArrays[b][packIdx] >> packBit) & 1;
            value |= (T(bit) << b);
        }
        d_result[i] = value;
    }
}

// A2B using parallel prefix adder approach with DCF
template <typename T>
u32* gpuA2B(SigmaPeer* peer, int party, int bw, T* d_arithShares, 
            u64 N, u8* a2bKey, AESGlobalContext* gaes, Stats* s) {
    
    // For now, implement a simple bit-by-bit extraction approach
    // This can be optimized later using DCF-based parallel prefix adder
    
    u64 numPackedInts = (N - 1) / 32 + 1;
    u32** d_bitShares = (u32**)gpuMalloc(bw * sizeof(u32*));
    u32** h_bitPtrs = (u32**)cpuMalloc(bw * sizeof(u32*));
    
    // Extract each bit position
    for (int b = 0; b < bw; b++) {
        u32* d_bits = (u32*)gpuMalloc(numPackedInts * sizeof(u32));
        checkCudaErrors(cudaMemset(d_bits, 0, numPackedInts * sizeof(u32)));
        
        // Extract bit b from all values
        extractBitsKernel<<<(N - 1) / 256 + 1, 256>>>(bw, N, d_arithShares, d_bits, b);
        
        h_bitPtrs[b] = d_bits;
    }
    
    // Copy array of pointers to GPU
    checkCudaErrors(cudaMemcpy(d_bitShares, h_bitPtrs, bw * sizeof(u32*), cudaMemcpyHostToDevice));
    
    // Combine bits into final binary representation
    u32* d_binShares = (u32*)gpuMalloc(numPackedInts * sizeof(u32));
    checkCudaErrors(cudaMemset(d_binShares, 0, numPackedInts * sizeof(u32)));
    
    // For simplicity, just copy the bit arrays for now
    // A full implementation would use DCF to properly convert
    for (int b = 0; b < bw && b < 32; b++) {
        // Simple approach: just OR the bits together  
        xorKernel<<<(numPackedInts - 1) / 256 + 1, 256>>>(d_binShares, h_bitPtrs[b], numPackedInts);
    }
    
    // Cleanup
    for (int b = 0; b < bw; b++) {
        gpuFree(h_bitPtrs[b]);
    }
    cpuFree(h_bitPtrs);
    gpuFree(d_bitShares);
    
    return d_binShares;
}

// ============== B2A (Binary to Arithmetic) Conversion ==============

template <typename T>
__global__ void binaryToArithKernel(u64 N, u32* d_binShares, T* d_arithShares, int targetBw) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N) {
        // Extract bit from packed representation
        int packIdx = i / 32;
        int packBit = i % 32;
        T bit = (d_binShares[packIdx] >> packBit) & 1;
        
        // For simple B2A, the arithmetic share is just the bit value
        // A full implementation would use DPF evaluation
        d_arithShares[i] = bit;
        
        // Apply modulo for target bit width
        if (targetBw < sizeof(T) * 8) {
            d_arithShares[i] &= ((T(1) << targetBw) - 1);
        }
    }
}

// B2A using DPF evaluation
template <typename T>
T* gpuB2A(SigmaPeer* peer, int party, int bw, u32* d_binShares, 
          u64 N, int targetBw, u8* b2aKey, AESGlobalContext* gaes, Stats* s) {
    
    T* d_arithShares = (T*)gpuMalloc(N * sizeof(T));
    
    // Simple implementation for now
    // Full implementation would use DPF to convert binary to arithmetic domain
    binaryToArithKernel<<<(N - 1) / 256 + 1, 256>>>(N, d_binShares, d_arithShares, targetBw);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // In a complete implementation, we would:
    // 1. Use DPF keys to evaluate the conversion function
    // 2. Apply the DPF evaluation to each binary share
    // 3. Combine results to get arithmetic shares
    
    return d_arithShares;
}

// ============== Key Generation Functions ==============

template <typename T>
u8* genGPUA2BKey(u8** key_as_bytes, int party, int bw, u64 N, AESGlobalContext* gaes) {
    // Generate keys for A2B conversion
    // This would involve generating DCF keys for bit extraction
    
    size_t keySize = bw * N * sizeof(T); // Simplified estimate
    u8* key = (u8*)cpuMalloc(keySize);
    
    // Generate random masks for each bit position
    for (int b = 0; b < bw; b++) {
        T* d_mask = randomGEOnGpu<T>(N, 1); // 1-bit masks
        u8* h_mask = (u8*)moveToCPU((u8*)d_mask, N * sizeof(T), nullptr);
        memcpy(key + b * N * sizeof(T), h_mask, N * sizeof(T));
        cpuFree(h_mask);
        gpuFree(d_mask);
    }
    
    *key_as_bytes = key;
    return key;
}

template <typename T>
u8* genGPUB2AKey(u8** key_as_bytes, int party, int bw, u64 N, AESGlobalContext* gaes) {
    // Generate keys for B2A conversion
    // This would involve generating DPF keys
    
    size_t keySize = N * sizeof(T); // Simplified estimate
    u8* key = (u8*)cpuMalloc(keySize);
    
    // Generate random mask for conversion
    T* d_mask = randomGEOnGpu<T>(N, bw);
    u8* h_mask = (u8*)moveToCPU((u8*)d_mask, N * sizeof(T), nullptr);
    memcpy(key, h_mask, N * sizeof(T));
    cpuFree(h_mask);
    gpuFree(d_mask);
    
    *key_as_bytes = key;
    return key;
}

// Explicit template instantiations
// These are now in gpu_lss.cu
// template u32* gpuA2B<u32>(SigmaPeer*, int, int, u32*, u64, u8*, AESGlobalContext*, Stats*);
// template u32* gpuA2B<u64>(SigmaPeer*, int, int, u64*, u64, u8*, AESGlobalContext*, Stats*);

// template u32* gpuB2A<u32>(SigmaPeer*, int, int, u32*, u64, int, u8*, AESGlobalContext*, Stats*);
// template u64* gpuB2A<u64>(SigmaPeer*, int, int, u32*, u64, int, u8*, AESGlobalContext*, Stats*);

// template u8* genGPUA2BKey<u32>(u8**, int, int, u64, AESGlobalContext*);
// template u8* genGPUA2BKey<u64>(u8**, int, int, u64, AESGlobalContext*);

// template u8* genGPUB2AKey<u32>(u8**, int, int, u64, AESGlobalContext*);
// template u8* genGPUB2AKey<u64>(u8**, int, int, u64, AESGlobalContext*);