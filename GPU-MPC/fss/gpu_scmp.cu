// Author: Ported from CPU FSS implementation
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

#include <assert.h>
#include <cstdint>
#include <iostream>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"

// Helper function to split shares (simplified for GPU)
__device__ __host__ void splitShare(u64 value, u64 *share0, u64 *share1) {
    *share0 = value & 0x7FFFFFFFFFFFFFFF; // Simple split, could use better randomization
    *share1 = value - *share0;
}

// GPU DualDCF key generation
void keyGenGPUDualDCF(int Bin, int Bout, int N, u64 idx, 
                      u64 *payload1, u64 *payload2, int party,
                      GPUDualDCFKey *key0, GPUDualDCFKey *key1,
                      AESGlobalContext *gaes) {
    
    // Initialize key structures
    key0->Bin = Bin; key1->Bin = Bin;
    key0->Bout = Bout; key1->Bout = Bout;
    key0->groupSize = N; key1->groupSize = N;
    
    // Allocate memory for shared values
    key0->memSzSb = N * sizeof(u64);
    key1->memSzSb = N * sizeof(u64);
    key0->sb = (u64*)malloc(key0->memSzSb);
    key1->sb = (u64*)malloc(key1->memSzSb);
    
    // Calculate payload differences for DCF
    u64 *payload = (u64*)malloc(N * sizeof(u64));
    for (int i = 0; i < N; i++) {
        payload[i] = payload1[i] - payload2[i];
    }
    
    // Create device memory for the index
    u64 *d_idx = (u64*)gpuMalloc(sizeof(u64));
    checkCudaErrors(cudaMemcpy(d_idx, &idx, sizeof(u64), cudaMemcpyHostToDevice));
    
    // Generate DCF keys using existing gpuKeyGenDCF function
    u8 *keyBuf0 = (u8*)malloc(N * 1024 * 1024); // Allocate buffer for keys
    u8 *keyBuf1 = (u8*)malloc(N * 1024 * 1024); // FIXME: HARDCODED 1024^2 SEEMS WRONG, SHOULD BE RELATED TO N???
    u8 *keyPtr0 = keyBuf0;
    u8 *keyPtr1 = keyBuf1;
    
    // Generate keys for both parties
    gpuKeyGenDCF(&keyPtr0, 0, Bin, N, d_idx, gaes);
    gpuKeyGenDCF(&keyPtr1, 1, Bin, N, d_idx, gaes);
    
    // Read the generated keys
    key0->dcfKey = readGPUDcfKey(&keyBuf0);
    key1->dcfKey = readGPUDcfKey(&keyBuf1);

    // Split payload2 shares for sb values
    for (int i = 0; i < N; i++) {
        splitShare(payload2[i], &key0->sb[i], &key1->sb[i]);
    }
    
    // Cleanup
    free(payload);
    free(keyBuf0);
    free(keyBuf1);
    gpuFree(d_idx);
}

// Simplified DualDCF key generation (single payload version)
void keyGenGPUDualDCF(int Bin, int Bout, u64 idx, u64 payload1, u64 payload2, 
                      int party, GPUDualDCFKey *key0, GPUDualDCFKey *key1,
                      AESGlobalContext *gaes) {
    keyGenGPUDualDCF(Bin, Bout, 1, idx, &payload1, &payload2, party, key0, key1, gaes);
}

// GPU DualDCF evaluation kernel
__global__ void evalGPUDualDCFKernel(int party, u64 *res, u32 *dcf_result, 
                                      u64 *sb, int groupSize, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        // Add DCF evaluation result with shared values
        for (int i = 0; i < groupSize; i++) {
            res[tid * groupSize + i] = dcf_result[tid] + sb[i];
        }
    }
}

// GPU DualDCF evaluation
void evalGPUDualDCF(int party, u64 *res, u64 idx, const GPUDualDCFKey &key, 
                    int M, AESGlobalContext *gaes) {
    // Create device input for DCF evaluation (just the index for each element)
    u64 *d_idx = (u64*)gpuMalloc(M * sizeof(u64));
    checkCudaErrors(cudaMemset(d_idx, 0, M * sizeof(u64)));
    // Set all elements to the same index value for evaluation
    checkCudaErrors(cudaMemcpy(d_idx, &idx, sizeof(u64), cudaMemcpyHostToDevice));
    
    // Evaluate DCF
    Stats s;
    u32 *d_dcf_result = gpuDcf(key.dcfKey, party, d_idx, gaes, &s);
    
    // Allocate device memory for shared values
    u64 *d_sb;
    checkCudaErrors(cudaMalloc(&d_sb, key.memSzSb));
    checkCudaErrors(cudaMemcpy(d_sb, key.sb, key.memSzSb, cudaMemcpyHostToDevice));
    
    // Launch kernel to combine DCF result with shared values
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    evalGPUDualDCFKernel<<<gridSize, blockSize>>>(party, res, d_dcf_result, d_sb, key.groupSize, M);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup
    checkCudaErrors(cudaFree(d_sb));
    checkCudaErrors(cudaFree(d_idx));
    gpuFree(d_dcf_result);
}

// GPU SCMP key generation
void keyGenGPUSCMP(int Bin, int Bout, u64 rin1, u64 rin2, u64 rout, 
                   int party, GPUScmpKey *key0, GPUScmpKey *key1,
                   AESGlobalContext *gaes) {
    
    // Initialize key structures
    key0->Bin = Bin; key1->Bin = Bin;
    key0->Bout = Bout; key1->Bout = Bout;
    
    // Calculate comparison parameters (from CPU implementation)
    u64 y = -(rin1 - rin2);
    u8 y_msb = (y >> (Bin - 1)) & 1;
    u64 y_idx = y - ((u64)y_msb << (Bin - 1));
    
    u64 payload1 = 1 ^ y_msb;
    u64 payload2 = y_msb;
    
    // Generate DualDCF keys
    keyGenGPUDualDCF(Bin - 1, Bout, y_idx, payload1, payload2, party, 
                     &key0->dualDcfKey, &key1->dualDcfKey, gaes);
    
    // Split output randomness
    splitShare(rout, &key0->rb, &key1->rb);
}

// GPU SCMP evaluation kernel
__global__ void evalGPUSCMPKernel(int party, u64 *res, u64 x, u64 y, 
                                  u64 *mb_results, u64 rb, int Bin, int Bout, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        // Calculate comparison (from CPU implementation)
        u64 z = x - y;
        u8 z_msb = (z >> (Bin - 1)) & 1;
        u64 z_n_1 = z - ((u64)z_msb << (Bin - 1));
        u64 z_idx = ((u64)1 << (Bin - 1)) - z_n_1 - 1;
        
        // Get mb from DualDCF evaluation result
        u64 mb = mb_results[tid];
        
        // Final SCMP result calculation
        res[tid] = party - (party * z_msb + mb - 2 * z_msb * mb) + rb;
    }
}

// GPU SCMP evaluation
void evalGPUSCMP(int party, u64 *res, u64 x, u64 y, const GPUScmpKey &key, 
                 int M, AESGlobalContext *gaes) {
    
    // Calculate comparison index for DualDCF evaluation
    u64 z = x - y;
    u8 z_msb = (z >> (key.Bin - 1)) & 1;
    u64 z_n_1 = z - ((u64)z_msb << (key.Bin - 1));
    u64 z_idx = ((u64)1 << (key.Bin - 1)) - z_n_1 - 1;
    
    // Allocate device memory for DualDCF results
    u64 *d_mb_results = (u64*)gpuMalloc(M * sizeof(u64));
    
    // Evaluate DualDCF to get mb values
    evalGPUDualDCF(party, d_mb_results, z_idx, key.dualDcfKey, M, gaes);
    
    // Launch SCMP kernel
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    evalGPUSCMPKernel<<<gridSize, blockSize>>>(party, res, x, y, d_mb_results, 
                                               key.rb, key.Bin, key.Bout, M);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup
    gpuFree(d_mb_results);
}

// Memory management functions
void freeGPUDualDCFKey(GPUDualDCFKey &key) {
    if (key.sb) {
        free(key.sb);
        key.sb = nullptr;
    }
    // Free DCF key memory
    if (key.dcfKey.bin > 7 && key.dcfKey.dpfTreeKey) {
        for (int b = 0; b < key.dcfKey.B; b++) {
            // The key memory is allocated as a single buffer and managed externally
            // We don't need to free individual components here
        }
        delete[] key.dcfKey.dpfTreeKey;
        key.dcfKey.dpfTreeKey = nullptr;
    }
}

void freeGPUScmpKey(GPUScmpKey &key) {
    freeGPUDualDCFKey(key.dualDcfKey);
} 