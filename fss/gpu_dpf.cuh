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
#include "gpu_aes_shm.cuh"
#include "gpu_sstab.h"
#include <cstring>

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
inline GPUDPFTreeKey readGPUDPFTreeKey(u8 **key_as_bytes)
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
inline GPUDPFKey readGPUDPFKey(u8 **key_as_bytes)
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

// Forward declarations
template <typename T>
u32 *gpuDpf(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s);

// Special DPF variant that accepts prologue/epilogue functions
template <typename T, int E, dpfPrologue pr, dpfEpilogue ep>
u32 *gpuDcf_(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks = NULL);

// Key generation function (defined in gpu_dpf.cu)
template <typename T>
void gpuKeyGenDCF_(u8 **key_as_bytes, int party, int bin, int N, T *d_in, AESGlobalContext *g);

// Include device function implementations
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

typedef void (*treeTraversal)(int party, int bin, int N,
                              u64 x,
                              AESBlock *scw, AESBlock *l0, AESBlock *l1,
                              u32 *tR, AESSharedContext *c, u32 *out, u64 oStride);

// only supports one bit output
// can avoid returning AESBlock to reduce copying
__device__ inline AESBlock expandDPFTreeNode(int bin, int party,
                                      const AESBlock s,
                                      const AESBlock cw,
                                      const AESBlock l0,
                                      const AESBlock l1,
                                      u32 tR,
                                      const u8 keep,
                                      int i,
                                      AESSharedContext *c)
{
    const AESBlock notOneAESBlock = ~1;
    const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};
    const AESBlock OneAESBlock = 1;

    AESBlock tau = 0, stcw;
    u8 t_previous = lsb(s);
    /* remove the last two bits from the AES seed */
    auto ss = s & notOneAESBlock;

    /* get the seed for this level (tau) */
    // apply aes to either 0 or 2 based on keep (is what is hopefully happening)
    applyAESPRG(c, (u32 *)&ss, keep * 2, (u32 *)&tau);
    AESBlock scw = 0;
    if (i < bin - LOG_AES_BLOCK_LEN - 1)
    {
        /* zero out the last two bits of the correction word for s because
    they must contain the corrections for t0 and t1 */
        scw = (cw & notOneAESBlock);
        /* separate the correction bits for t0 and t1 and place them
    in the lsbs of two AES blocks */
        // u32 ds1 = tR_l;
        // if (evalAll)
        // tR_l = ((tR_l >> i) & 1);
        // else
        // ds1 = tR; /*getVCW(1, tR, N, i);*/
        AESBlock ds[2] = {cw & OneAESBlock, AESBlock(tR)};
        scw ^= ds[keep];
    }
    else
    {
        AESBlock ds[2] = {l0, l1};
        scw = ds[keep];
    }

    const auto mask = zeroAndAllOne[t_previous];

    /* correct the seed for the next level if necessary */
    // tau is completely pseudorandom and is being xored with (scw || 0 || tcw)*keep
    stcw = tau ^ (scw & mask);
    return stcw;
}

__device__ inline u8 getDPFOutput(AESBlock *s, u64 x)
{
    gpuMod(x, LOG_AES_BLOCK_LEN);
    return u8(*s >> x) & 1;
}

// Device functions for DPF tree traversal - correct versions matching treeTraversal typedef
__device__ inline void doDpf(int party, int bin, int N,
                      u64 x,
                      AESBlock *scw, AESBlock *l0, AESBlock *l1,
                      u32 *tR, AESSharedContext *c, u32 *out, u64 oStride)
{
    AESBlock s = scw[0];
    auto x1 = __brevll(x) >> (64 - bin);
    for (int i = 0; i < bin - LOG_AES_BLOCK_LEN; ++i)
    {
        const u8 keep = lsb(x1);
        if (i < bin - LOG_AES_BLOCK_LEN - 1)
        {
            u32 tR_l = u32(getVCW(1, tR, N, i));
            s = expandDPFTreeNode(bin, party, s, scw[(i + 1) * N], 0, 0, tR_l, keep, i, c);
        }
        else
        {

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            s = expandDPFTreeNode(bin, party, s, 0, l0[tid], l1[tid], 0, keep, i, c);
        }
        x1 >>= 1;
    }
    auto o = getDPFOutput(&s, x);
    writePackedOp(out, u64(o), 1, N);
}

template <int E, dpfPrologue pr, dpfEpilogue ep>
__device__ inline void doDcf(int party, int bin, int N,
                      u64 x,
                      AESBlock *scw, AESBlock *l0, AESBlock *l1,
                      u32 *tR,
                      AESSharedContext *c, u32 *out, u64 oStride)
{
    AESBlock s[E];
    u64 x0[E], x1[E];
    // populate the input
    pr(party, bin, N, x, x0);
    u8 p[E], oldDir[E], keep[E];
    for (int e = 0; e < E; e++)
    {
        s[e] = scw[0];
        x1[e] = __brevll(x0[e]) >> (64 - bin);
        p[e] = 0;
        oldDir[e] = 0;
        keep[e] = 0;
    }
    for (int i = 0; i < bin - LOG_AES_BLOCK_LEN; ++i)
    {
        AESBlock curScw = 0, l0_l = 0, l1_l = 0;
        u32 tR_l;
        if (i < bin - LOG_AES_BLOCK_LEN - 1)
        {
            curScw = scw[(i + 1) * N];
            tR_l = u32(getVCW(1, tR, N, i));
        }
        else
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            l0_l = l0[tid];
            l1_l = l1[tid];
        }

        for (int e = 0; e < E; e++)
        {

            keep[e] = lsb(x1[e]);
            // the direction changed
            if (oldDir[e] != keep[e])
                p[e] ^= lsb(s[e]);
            // need to keep track of all the current seeds separately
            if (i < bin - LOG_AES_BLOCK_LEN - 1)
            {
                s[e] = expandDPFTreeNode(bin, party, s[e], curScw, 0, 0, tR_l, keep[e], i, c);
            }
            else
            {
                s[e] = expandDPFTreeNode(bin, party, s[e], 0, l0_l, l1_l, 0, keep[e], i, c);
                int ub;
                int pos = x0[e] & 127;
                if (keep[e] == 1)
                {
                    // xor with the complement of the prefix substring
                    // get rid of the lower order bits
                    // Neha: need to change this later
                    // can need to xor anywhere from 127 bits to 0 bits
                    ub = 127 - pos;
                    s[e] >>= (pos + 1);
                }
                else
                {
                    ub = pos + 1; // x0[e] & 127;
                    // don't get rid of the lower order bits
                }
                for (int i = 0; i < ub; i++)
                {
                    // extract the lsb of s
                    p[e] ^= lsb(s[e]) /*((u32)(*s) & 1)*/;
                    s[e] >>= 1;
                }
            }
            oldDir[e] = keep[e];
            x1[e] >>= 1;
        }
    }
    // add loop here as well
    if (party == SERVER1)
    {
        for (int e = 0; e < E; e++)
        {
            p[e] ^= u8(1);
        }
    }
    ep(party, bin, N, x, p, out, oStride);
}

// Template kernel for DPF tree evaluation using function pointer
template <typename T, treeTraversal t>
__global__ void dpfTreeEval(int party, int bin, int N, T *in, AESBlock *scw,
                            AESBlock *l0, AESBlock *l1, u32 *tR, u32 *out, u64 oStride, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        scw = &scw[tid];
        auto x = u64(in[tid]);
        t(party, bin, N, x, scw, l0, l1, tR, &saes, out, oStride);
    }
}

// REMOVED: Unused dcfTreeEval with evalAll parameter

template <typename T>
__global__ void keyGenDPFTreeKernel(int party, int bin, int N, T *rinArr, AESBlock *s0, AESBlock *s1, AESBlock *k0, AESBlock *l0, AESBlock *l1, u32 *tR, AESGlobalContext gaes, bool evalAll)
{
    AESGlobalContext *g = &gaes;
    AESSharedContext s;
    loadSbox(g, &s);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        AESBlock sl = s0[j], sr = s1[j];
        u32 tl = lsb(sl), tr = lsb(sr);

        AESBlock scw;
        for (int i = 0; i < bin - LOG_AES_BLOCK_LEN; i++)
        {
            // int i = level;
            u64 rin = rinArr[j];

            // compute the seeds for the next level
            applyAESPRGTwoTimes(&s, (u32 *)&sl, 0, (u32 *)&s0[j], (u32 *)&s1[j]);
            applyAESPRGTwoTimes(&s, (u32 *)&sr, 0, (u32 *)&sl, (u32 *)&sr);

            // get the bit we want to condition on
            u8 keep = (rin >> (bin - i - 1)) & 1;

            // control bit correction
            u32 t0L = lsb(s0[j]), t1L = lsb(s1[j]);
            u32 t0R = lsb(sl), t1R = lsb(sr);

            // if keep == 0: t_{CW} = t_0^L \oplus t_0^R \oplus keep \oplus 1
            // if keep == 1: t_{CW} = t_1^L \oplus t_1^R \oplus keep \oplus 1

            u32 tCW = keep ^ 1;
            if (keep == 0)
                tCW ^= t0L ^ t0R;
            else
                tCW ^= t1L ^ t1R;

            // seed correction
            // s_{CW} = s_0^L \oplus s_0^R if keep == 0
            // s_{CW} = s_1^L \oplus s_1^R if keep == 1
            if (keep == 0)
                scw = s0[j] ^ sl;
            else
                scw = s1[j] ^ sr;

            // convert
            // sl = s_{keep}^L, sr = s_{keep}^R
            if (keep == 0)
            {
                sr = sl;
                sl = s0[j];
            }
            else
            {
                sl = s1[j];
                // sr = sr;
            }

            // tl = t_{keep}^L, tr = t_{keep}^R
            if (keep == 0)
            {
                tr = t0R;
                tl = t0L;
            }
            else
            {
                tl = t1L;
                tr = t1R;
            }

            if (i < bin - LOG_AES_BLOCK_LEN - 1)
            {
                // set the control bit
                scw = (scw & (~1)) | tCW;
                // store the correction word for level i in the DPF key
                k0[j * (bin - LOG_AES_BLOCK_LEN) + i] = scw;
                if (evalAll)
                    tR[j] |= (tCW << i);
                else
                    tR[j * (bin - LOG_AES_BLOCK_LEN) + i] = tCW;

                // apply the correction to the seeds
                sl = sl ^ ((scw & (~1)) * tl);
                sr = sr ^ ((scw & (~1)) * tr);
                tl = tl ^ (tCW * tl);
                tr = tr ^ (tCW * tr);
            }
            else
            {
                // special case: last level
                l0[j] = scw;
                if (evalAll)
                    tR[j] |= (tCW << i);
                else
                    tR[j * (bin - LOG_AES_BLOCK_LEN) + i] = tCW;
            }
        }
    }
}

// Template functions that need to be in header for visibility
template <typename T>
void doDpfTreeKeyGen(u8 **key_as_bytes, int party, int bin, int N,
                     T *d_rin, AESGlobalContext *gaes, bool evalAll)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, N);
    writeInt(key_as_bytes, evalAll);
    assert(bin > LOG_AES_BLOCK_LEN);

    u64 memSizeK = N * (bin - LOG_AES_BLOCK_LEN) * sizeof(AESBlock);
    AESBlock *d_k0 = (AESBlock *)gpuMalloc(memSizeK);
    u64 memSizeL = N * sizeof(AESBlock);
    AESBlock *d_l0 = (AESBlock *)gpuMalloc(memSizeL);
    AESBlock *d_l1 = (AESBlock *)gpuMalloc(memSizeL);
    u64 memSizeT;
    if (evalAll)
        memSizeT = N * sizeof(u32);
    else
        memSizeT = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (bin - LOG_AES_BLOCK_LEN);
    u32 *d_tR = (u32 *)gpuMalloc(memSizeT);

    auto d_s0 = randomAESBlockOnGpu(N);
    auto d_s1 = randomAESBlockOnGpu(N);
    keyGenDPFTreeKernel<<<(N - 1) / 256 + 1, 256>>>(party, bin, N, d_rin, d_s0, d_s1, d_k0, d_l0, d_l1, d_tR, *gaes, evalAll);
    checkCudaErrors(cudaDeviceSynchronize());
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_k0, memSizeK, NULL);

    *key_as_bytes += memSizeK;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_l0, memSizeL, NULL);
    *key_as_bytes += memSizeL;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_l1, memSizeL, NULL);
    *key_as_bytes += memSizeL;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_tR, memSizeT, NULL);
    *key_as_bytes += memSizeT;

    gpuFree(d_s0);
    gpuFree(d_s1);
    gpuFree(d_k0);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);
}

template <typename T>
void gpuKeyGenBatchedDPF(u8 **key_as_bytes, int party, int bin, int N,
                         T *d_rin, AESGlobalContext *gaes, bool evalAll)
{
    u64 memSzOneK = (bin - LOG_AES_BLOCK_LEN + 2) * sizeof(AESBlock);
    int m = (24 * OneGB) / memSzOneK;
    m -= (m % 32);
    int B = (N - 1) / m + 1;
    // printf("N=%d, m=%d, B=%d, evalAll=%d\n", N, m, B, evalAll);
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, N);
    writeInt(key_as_bytes, B);
    for (int b = 0; b < B; b++)
        doDpfTreeKeyGen(key_as_bytes, party, bin, std::min(m, N - b * m), d_rin + b * m, gaes, evalAll);
}

template <typename T>
void gpuKeyGenDCF_(u8 **key_as_bytes, int party, int bin, int N,
                  T *d_rin, AESGlobalContext *gaes)
{
    // printf("Bin inside keygenDCF=%d\n", bin);
    if (bin <= 7)
    {
        genSSTable<T, dcfShares>(key_as_bytes, party, bin, N, d_rin);
    }
    else
    {
        gpuKeyGenBatchedDPF(key_as_bytes, party, bin, N, d_rin, gaes, false);
    }
}