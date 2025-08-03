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

#include "gpu_scmp.h"
#include <assert.h>
#include <cstdint>
#include <iostream>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_mem.h"
#include "dcf/gpu_dcf.cuh"
#include "dcf/gpu_dcf_templates.h"

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
    u64 *d_idx = (u64*)gpuMalloc(N * sizeof(u64));
    // Set all N elements to idx for batched processing
    u64 *h_idx = (u64*)malloc(N * sizeof(u64));
    for (int i = 0; i < N; i++) {
        h_idx[i] = idx;
    }
    checkCudaErrors(cudaMemcpy(d_idx, h_idx, N * sizeof(u64), cudaMemcpyHostToDevice));
    
    // Allocate reasonable key buffer size (estimate based on DCF key structure)
    size_t keyBufSize = N * (Bin + 10) * sizeof(AESBlock) + 1024 * 1024; // Conservative estimate
    u8 *keyBuf0 = (u8*)malloc(keyBufSize);
    u8 *keyBuf1 = (u8*)malloc(keyBufSize);
    u8 *keyPtr0 = keyBuf0;
    u8 *keyPtr1 = keyBuf1;
    
    // Generate keys for both parties using the correct signature
    // For now, we use payload[0] as the single payload value - this needs to be updated
    // to support multiple payloads properly
    dcf::gpuKeyGenDCF(&keyPtr0, 0, Bin, Bout, N, d_idx, u64(payload[0]), gaes);
    dcf::gpuKeyGenDCF(&keyPtr1, 1, Bin, Bout, N, d_idx, u64(payload[0]), gaes);
    
    // Reset pointers for reading
    keyPtr0 = keyBuf0;
    keyPtr1 = keyBuf1;
    
    // Read the generated keys
    key0->dcfKey = dcf::readGPUDCFKey(&keyPtr0);
    key1->dcfKey = dcf::readGPUDCFKey(&keyPtr1);

    // Split payload2 shares for sb values
    for (int i = 0; i < N; i++) {
        splitShare(payload2[i], &key0->sb[i], &key1->sb[i]);
    }
    
    // Cleanup
    free(h_idx);
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
void evalGPUDualDCF(int party, u64 *d_res, u64 idx, const GPUDualDCFKey &key, 
                    int M, AESGlobalContext *gaes) {
    // Create device input for DCF evaluation
    u64 *d_idx = (u64*)gpuMalloc(M * sizeof(u64));
    u64 *h_idx = (u64*)malloc(M * sizeof(u64));
    for (int i = 0; i < M; i++) {
        h_idx[i] = idx;
    }
    checkCudaErrors(cudaMemcpy(d_idx, h_idx, M * sizeof(u64), cudaMemcpyHostToDevice));
    
    // Evaluate DCF
    Stats s;
    // Use the dcf namespace and proper template parameters
    u32 *d_dcf_result = dcf::gpuDcf<u64, 1, dcf::idPrologue, dcf::idEpilogue>(
        key.dcfKey, party, d_idx, gaes, &s);
    
    // Allocate device memory for shared values
    u64 *d_sb;
    // Debug: Check if key.sb is valid and memSzSb is reasonable
    if (key.sb == nullptr || key.memSzSb == 0) {
        printf("ERROR: Invalid key.sb pointer or memSzSb. sb=%p, memSzSb=%lu, groupSize=%d\n", 
               key.sb, key.memSzSb, key.groupSize);
        return;
    }
    checkCudaErrors(cudaMalloc(&d_sb, key.memSzSb));
    checkCudaErrors(cudaMemcpy(d_sb, key.sb, key.memSzSb, cudaMemcpyHostToDevice));
    
    // Launch kernel to combine DCF result with shared values
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    evalGPUDualDCFKernel<<<gridSize, blockSize>>>(party, d_res, d_dcf_result, d_sb, key.groupSize, M);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup
    free(h_idx);
    checkCudaErrors(cudaFree(d_sb));
    gpuFree(d_idx);
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
        // u64 z_idx = ((u64)1 << (Bin - 1)) - z_n_1 - 1; // Not used in GPU kernel
        
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
    if (key.dcfKey.bin > 8 && key.dcfKey.dcfTreeKey) {
        for (int b = 0; b < key.dcfKey.B; b++) {
            // The key memory is allocated as a single buffer and managed externally
            // We don't need to free individual components here
        }
        delete[] key.dcfKey.dcfTreeKey;
        key.dcfKey.dcfTreeKey = nullptr;
    }
}

void freeGPUScmpKey(GPUScmpKey &key) {
    freeGPUDualDCFKey(key.dualDcfKey);
}

// GPU Integer Less-Than Comparison (x < y)
// This is implemented as NOT(x >= y), which is NOT(SCMP(x, y))
void keyGenGPULessThan(int Bin, int Bout, u64 rin1, u64 rin2, u64 rout,
                       int party, GPUScmpKey *key0, GPUScmpKey *key1,
                       AESGlobalContext *gaes) {
    // Use SCMP key generation (which computes x >= y)
    keyGenGPUSCMP(Bin, Bout, rin1, rin2, rout, party, key0, key1, gaes);
}

// GPU Less-Than evaluation kernel
__global__ void evalGPULessThanKernel(int party, u64 *res, u64 *scmp_results, 
                                      int Bout, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        // x < y is equivalent to NOT(x >= y)
        // In arithmetic shares: 1 - scmp_result
        res[tid] = ((1ULL << Bout) - 1) - scmp_results[tid];
    }
}

// GPU Less-Than evaluation
void evalGPULessThan(int party, u64 *res, u64 x, u64 y, const GPUScmpKey &key,
                     int M, AESGlobalContext *gaes) {
    // Allocate device memory for SCMP results
    u64 *d_scmp_results = (u64*)gpuMalloc(M * sizeof(u64));
    
    // First evaluate SCMP (x >= y)
    evalGPUSCMP(party, d_scmp_results, x, y, key, M, gaes);
    
    // Allocate device memory for final results
    u64 *d_res = (u64*)gpuMalloc(M * sizeof(u64));
    
    // Launch kernel to compute NOT(SCMP) = (x < y)
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    evalGPULessThanKernel<<<gridSize, blockSize>>>(party, d_res, d_scmp_results, key.Bout, M);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy results back to host
    checkCudaErrors(cudaMemcpy(res, d_res, M * sizeof(u64), cudaMemcpyDeviceToHost));
    
    // Cleanup
    gpuFree(d_scmp_results);
    checkCudaErrors(cudaFree(d_res));
} 