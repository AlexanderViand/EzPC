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

#pragma once

#include "utils/gpu_data_types.h"
#include "utils/gpu_random.h"
#include "dcf/gpu_dcf.cuh"

// GPU DualDCF Key structure (required for SCMP)
struct GPUDualDCFKey
{
    int Bin, Bout, groupSize;
    dcf::GPUDCFKey dcfKey;
    u64 *sb;   // shared values, size: groupSize
    u64 memSzSb;
};

// GPU SCMP Key structure
struct GPUScmpKey
{
    int Bin, Bout;
    GPUDualDCFKey dualDcfKey;
    u64 rb;  // random blinding value
};

// Function declarations for DualDCF operations
void keyGenGPUDualDCF(int Bin, int Bout, int groupSize, u64 idx, 
                      u64 *payload1, u64 *payload2, int party,
                      GPUDualDCFKey *key0, GPUDualDCFKey *key1,
                      AESGlobalContext *gaes);

void keyGenGPUDualDCF(int Bin, int Bout, u64 idx, u64 payload1, u64 payload2, 
                      int party, GPUDualDCFKey *key0, GPUDualDCFKey *key1,
                      AESGlobalContext *gaes);

void evalGPUDualDCF(int party, u64 *d_res, u64 idx, const GPUDualDCFKey &key, 
                    int M, AESGlobalContext *gaes);

// Function declarations for SCMP operations
void keyGenGPUSCMP(int Bin, int Bout, u64 rin1, u64 rin2, u64 rout, 
                   int party, GPUScmpKey *key0, GPUScmpKey *key1,
                   AESGlobalContext *gaes);

void evalGPUSCMP(int party, u64 *res, u64 x, u64 y, const GPUScmpKey &key, 
                 int M, AESGlobalContext *gaes);

// Memory management functions
void freeGPUDualDCFKey(GPUDualDCFKey &key);
void freeGPUScmpKey(GPUScmpKey &key);

// Integer comparison functions (x < y)
void keyGenGPULessThan(int Bin, int Bout, u64 rin1, u64 rin2, u64 rout,
                       int party, GPUScmpKey *key0, GPUScmpKey *key1,
                       AESGlobalContext *gaes);

void evalGPULessThan(int party, u64 *res, u64 x, u64 y, const GPUScmpKey &key,
                     int M, AESGlobalContext *gaes);

// Implementation is in gpu_scmp.cu 