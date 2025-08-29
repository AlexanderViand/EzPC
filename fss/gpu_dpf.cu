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

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_mem.h"

#include "gpu_linear_helper.cuh"
#include "gpu_dpf_templates.h"
#include "gpu_dpf.cuh"

typedef void (*treeTraversal)(int party, int bin, int N,
                              u64 x,
                              AESBlock *scw, AESBlock *l0, AESBlock *l1,
                              u32 *tR, AESSharedContext *c, u32 *out, u64 oStride);


// Moved doDpf and doDcf to gpu_dpf.cuh to comply with RULE 2

// think about when to pass pointers to large amounts of data like AESBlocks
/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
// Moved dpfTreeEval to gpu_dpf.cuh to comply with RULE 2

template <typename T, treeTraversal t>
void gpuDpfTreeEval(GPUDPFTreeKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, u32 *d_out, u64 oStride)
{
    // auto d_out = moveMasks(k.memSzOut, h_masks, s);
    assert(k.memSzScw % (k.bin - LOG_AES_BLOCK_LEN) == 0);

    AESBlock *d_scw = (AESBlock *)moveToGPU((u8 *)k.scw, k.memSzScw, s);
    AESBlock *d_l0 = (AESBlock *)moveToGPU((u8 *)k.l0, k.memSzL, s);
    AESBlock *d_l1 = (AESBlock *)moveToGPU((u8 *)k.l1, k.memSzL, s);
    u32 *d_tR = (u32 *)moveToGPU((u8 *)k.tR, k.memSzT, s);

    const int tbSz = 256;
    int tb = (k.N - 1) / tbSz + 1;
    // auto start = std::chrono::high_resolution_clock::now();
    // kernel launch
    dpfTreeEval<T, t><<<tb, tbSz>>>(party, k.bin, k.N, d_in, d_scw, d_l0, d_l1, d_tR, d_out, oStride, *g);
    checkCudaErrors(cudaDeviceSynchronize());
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // printf("Time taken by dpf kernel=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    gpuFree(d_scw);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);
}

// no memory leak
template <typename T>
u32 *gpuDpf(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s)
{
    u32 *d_out;
    if (k.bin <= 7)
        d_out = gpuLookupSSTable<T, 1, idPrologue, idEpilogue>(k.ssKey, party, d_in, s);
    else
    {
        d_out = moveMasks(k.memSzOut, NULL, s);
        int n = k.dpfTreeKey[0].N;
        size_t gIntSzOut = k.memSzOut / sizeof(PACK_TYPE);
        size_t bIntSzOut = k.dpfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
        for (int b = 0; b < k.B; b++)
        {
            gpuDpfTreeEval<T, doDpf>(k.dpfTreeKey[b], party, d_in + b * n, g, s, d_out + b * bIntSzOut, (u64)gIntSzOut);
        }
    }
    return d_out;
}

template <typename T, int E, dpfPrologue pr, dpfEpilogue ep>
u32 *gpuDcf_(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks = NULL)
{
    // printf("Started gpu dcf\n");
    u32 *d_out;
    if (k.bin <= 7)
        d_out = gpuLookupSSTable<T, E, pr, ep>(k.ssKey, party, d_in, s, h_masks);
    else
    {
        d_out = moveMasks(k.memSzOut, h_masks, s);
        size_t gIntSzOut = k.memSzOut / sizeof(PACK_TYPE);
        int n = k.dpfTreeKey[0].N;
        size_t bIntSzOut = k.dpfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
        // printf("outSz=%lu\n", bIntSzOut);
        for (int b = 0; b < k.B; b++)
        {
            gpuDpfTreeEval<T, doDcf<E, pr, ep>>(k.dpfTreeKey[b], party, d_in + b * n, g, s, d_out + b * bIntSzOut, (u64)gIntSzOut);
        }
    }
    return d_out;
}

// Real Endpoints - kernel implementation removed (moved to header)

// Moved to gpu_dpf.cuh

// Explicit template instantiations
template u32 *gpuDpf<u32>(GPUDPFKey k, int party, u32 *d_in, AESGlobalContext *g, Stats *s);
template u32 *gpuDpf<u64>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s);

template u32 *gpuDcf_<u32, 1, idPrologue, idEpilogue>(GPUDPFKey k, int party, u32 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
template u32 *gpuDcf_<u64, 1, idPrologue, idEpilogue>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

template u32 *gpuDcf_<u32, 1, idPrologue, maskEpilogue>(GPUDPFKey k, int party, u32 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
template u32 *gpuDcf_<u64, 1, idPrologue, maskEpilogue>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

// dRelu specializations
template u32 *gpuDcf_<u32, 1, dReluPrologue<65535UL>, dReluEpilogue<65535UL, false>>(GPUDPFKey k, int party, u32 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
template u32 *gpuDcf_<u64, 1, dReluPrologue<65535UL>, dReluEpilogue<65535UL, false>>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
template u32 *gpuDcf_<u32, 1, dReluPrologue<65535UL>, dReluEpilogue<65535UL, true>>(GPUDPFKey k, int party, u32 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
template u32 *gpuDcf_<u64, 1, dReluPrologue<65535UL>, dReluEpilogue<65535UL, true>>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

// Additional specializations for gelu
template u32 *gpuDcf_<u64, 3, geluPrologue<255UL, 18446744073709551361UL>, geluEpilogue<255UL, 18446744073709551361UL>>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

// Additional specializations for softmax/relu
template u32 *gpuDcf_<u64, 1, dReluPrologue<0UL>, dReluEpilogue<0UL, false>>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

// Additional specializations for truncate
template u32 *gpuDcf_<u16, 1, idPrologue, maskEpilogue>(GPUDPFKey k, int party, u16 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);

// Additional specializations for silu
template u32 *gpuDcf_<u64, 3, geluPrologue<1023UL, 18446744073709550593UL>, geluEpilogue<1023UL, 18446744073709550593UL>>(GPUDPFKey k, int party, u64 *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks);
