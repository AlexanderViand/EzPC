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

#include <cstdint>

#include "fss/gpu_truncate.cuh"
#include "fss/gpu_softmax.cuh"
#include "fss/gpu_matmul.cuh"

/**
 * @brief Parameters for Multi-Head Attention (MHA) operations
 * 
 * This structure defines the configuration for secure multi-head attention
 * computation, a key component of transformer models.
 */
struct MHAParams
{
    int n_seq;      ///< Sequence length (number of tokens)
    int n_embed;    ///< Embedding dimension
    int n_heads;    ///< Number of attention heads
    int dim_W;      ///< Dimension per head (typically n_embed / n_heads)
    
    bool selfAttn;   ///< True for self-attention (uses causal mask)
    bool doNormQKt;  ///< True to normalize Q*K^T by sqrt(dim_W)
    bool rotEmb;     ///< True to apply rotary position embeddings
};

/**
 * @brief Collection of matrix multiplication parameters for MHA
 * 
 * This structure groups all the matrix multiplication configurations
 * needed for the different stages of multi-head attention.
 */
struct MHAMulParams
{
    MatmulParams pQKV;      ///< Parameters for Q, K, V projection
    MatmulParams pQKt;      ///< Parameters for Q * K^T computation
    MatmulParams pSmQKtV;   ///< Parameters for softmax(Q*K^T) * V
    MatmulParams pProj;     ///< Parameters for output projection
    MaxpoolParams pMPool;   ///< Parameters for maxpool in softmax
};

/**
 * @brief FSS key structure for GPU Multi-Head Attention
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * This structure contains all FSS keys needed for secure MHA:
 * 1. Matrix multiplication keys for Q/K/V projections
 * 2. Softmax key for attention weights
 * 3. Truncation keys for normalization and rotary embeddings
 * 
 * The MHA computation flow:
 * 1. Project input to Q, K, V using mmKeyQKV
 * 2. Apply rotary embeddings if enabled (reQTrKey, reKTrKey)
 * 3. Compute attention scores Q * K^T using mmKeyQKt
 * 4. Normalize scores if enabled (normQKtTrKey)
 * 5. Apply softmax using softmaxKey
 * 6. Compute attention output using mmKeySmQKtV
 * 7. Project back to original dimension using mmKeyProj
 */
template <typename T>
struct GPUMHAKey
{
    GPUMatmulKey<T> mmKeyQKV;       ///< Key for Q, K, V projections
    GPUMatmulKey<T> mmKeyQKt;       ///< Key for Q * K^T attention scores
    GPUMatmulKey<T> mmKeySmQKtV;    ///< Key for attention * V
    GPUMatmulKey<T> mmKeyProj;      ///< Key for output projection
    
    GPUSoftMaxKey<T> softmaxKey;    ///< Key for softmax over attention scores
    
    GPUTruncateKey<T> reQTrKey;     ///< Key for rotary embedding on Q
    GPUTruncateKey<T> reKTrKey;     ///< Key for rotary embedding on K
    GPUTruncateKey<T> normQKtTrKey; ///< Key for normalizing attention scores
};

/**
 * @brief Lookup tables for MHA non-linear operations
 * 
 * @tparam T Data type for the computation
 * 
 * This structure contains precomputed lookup tables for efficient
 * evaluation of non-linear functions in multi-head attention.
 */
template <typename T>
struct MHATables
{
    T *d_nExpMsbTab = NULL;  ///< LUT for negative exponential MSB part
    T *d_nExpLsbTab = NULL;  ///< LUT for negative exponential LSB part
    T *d_invTab = NULL;      ///< LUT for inverse operation (1/x)
};

/**
 * @brief Initialize lookup tables for MHA operations
 * 
 * @tparam T Data type for the computation
 * @param n_seq Sequence length
 * @param scale Scale factor for fixed-point arithmetic
 * @return MHATables<T> Initialized lookup tables
 * 
 * This function generates lookup tables for:
 * 1. Negative exponential (split into MSB and LSB for accuracy)
 * 2. Inverse function for softmax normalization
 * 
 * The table sizes and precisions are optimized for the given
 * sequence length and scale factor.
 */
template <typename T>
inline MHATables<T> initMHATables(int n_seq, int scale)
{
    MHATables<T> mhaTab;
    mhaTab.d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, scale);
    mhaTab.d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, scale);
    mhaTab.d_invTab = genLUT<T, inv<T>>(int(ceil(log2(n_seq))) + 6, 6, scale);
    return mhaTab;
}

inline MatmulParams initPQKV(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pQKV;
    pQKV.M = pMHA.n_seq;
    pQKV.K = pMHA.n_embed;
    pQKV.N = pMHA.dim_W;
    pQKV.batchSz = 3 * pMHA.n_heads;
    stdInit(pQKV, bw, scale);
    pQKV.size_A = pQKV.M * pQKV.K;
    pQKV.stride_A = 0;
    pQKV.ld_B = 3 * pMHA.n_embed;
    pQKV.stride_B = pMHA.dim_W;
    return pQKV;
}

inline MatmulParams initPQKt(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pQKt;
    pQKt.M = pMHA.n_seq;
    pQKt.K = pMHA.dim_W;
    pQKt.N = pMHA.n_seq;
    pQKt.batchSz = pMHA.n_heads;
    if (pMHA.selfAttn)
        pQKt.cIsLowerTriangular = true;

    stdInit(pQKt, bw, scale);
    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 0)
    {
        // assert(int(log2(dim_W)) % 2 == 0);
        // printf("Shift=%d\n", int(log2(pMHA.dim_W) / 2));
        pQKt.shift += int(log2(pMHA.dim_W) / 2);
    }
    // K is stored in column-major form
    pQKt.rowMaj_B = false;
    pQKt.ld_B = pQKt.K;
    return pQKt;
}

inline MatmulParams initPSmQKtV(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pSmQKtV;
    pSmQKtV.M = pMHA.n_seq;
    pSmQKtV.K = pMHA.n_seq;
    pSmQKtV.N = pMHA.dim_W;
    pSmQKtV.batchSz = pMHA.n_heads;
    stdInit(pSmQKtV, bw, scale);
    pSmQKtV.ld_C = pMHA.n_heads * pMHA.dim_W;
    pSmQKtV.stride_C = pMHA.dim_W;
    return pSmQKtV;
}

inline MatmulParams initPProj(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pProj;
    pProj.M = pMHA.n_seq;
    pProj.K = pMHA.n_embed;
    pProj.N = pMHA.n_embed;
    pProj.batchSz = 1;
    stdInit(pProj, bw, scale);
    return pProj;
}

inline MaxpoolParams initPMaxpool(MHAParams pMHA, int bw, int scale)
{
    MaxpoolParams pMPool;
    pMPool.N = pMHA.n_heads;
    pMPool.imgH = pMHA.n_seq;
    pMPool.imgW = pMHA.n_seq;
    pMPool.C = 1;
    pMPool.FH = 1;
    pMPool.FW = pMHA.n_seq;
    pMPool.strideH = 1;
    pMPool.strideW = pMHA.n_seq;
    pMPool.zPadHLeft = 0;
    pMPool.zPadWLeft = 0;
    pMPool.zPadHRight = 0;
    pMPool.zPadWRight = 0;
    pMPool.bw = bw;
    pMPool.bin = bw - scale;
    pMPool.scale = scale;
    pMPool.scaleDiv = 0;
    initPoolParams(pMPool);
    pMPool.bwBackprop = 0;
    if (pMHA.selfAttn)
        pMPool.isLowerTriangular = true;
    return pMPool;
}

MHAMulParams initMHAMulParams(MHAParams pMHA, int bw, int scale) {
    MHAMulParams pMHAMul;
    pMHAMul.pQKV = initPQKV(pMHA, bw, scale);
    pMHAMul.pQKt = initPQKt(pMHA, bw, scale);
    pMHAMul.pMPool = initPMaxpool(pMHA, bw, scale);
    pMHAMul.pSmQKtV = initPSmQKtV(pMHA, bw, scale);
    pMHAMul.pProj = initPProj(pMHA, bw, scale);
    return pMHAMul;
}

/**
 * @brief Read a GPU Multi-Head Attention key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param pMHA MHA parameters defining the attention configuration
 * @param pMHAMul Matrix multiplication parameters for MHA stages
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUMHAKey<T> Deserialized MHA key
 * 
 * This function deserializes a complete MHA key from a byte stream,
 * extracting all component keys in the correct order:
 * 
 * 1. Q/K/V projection keys
 * 2. Rotary embedding truncation keys (if enabled)
 * 3. Attention score computation key
 * 4. Normalization key (if needed based on dim_W)
 * 5. Softmax key for attention weights
 * 6. Attention output computation key
 * 7. Output projection key
 * 
 * The truncation types are chosen to maintain numerical stability:
 * - TrFloor for matrix multiplications
 * - TrWithSlack for rotary embeddings to preserve precision
 */
template <typename T>
GPUMHAKey<T> readGPUMHAKey(MHAParams pMHA, MHAMulParams pMHAMul, u8 **key_as_bytes)
{
    GPUMHAKey<T> k;
    k.mmKeyQKV = readGPUMatmulKey<T>(pMHAMul.pQKV, TruncateType::TrFloor, key_as_bytes);
    if (pMHA.rotEmb)
    {
        k.reQTrKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
        k.reKTrKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    }
    k.mmKeyQKt = readGPUMatmulKey<T>(pMHAMul.pQKt, TruncateType::TrFloor, key_as_bytes);
    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 1)
        k.normQKtTrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    k.softmaxKey = readGPUSoftMaxKey<T>(pMHAMul.pMPool, key_as_bytes);
    k.mmKeySmQKtV = readGPUMatmulKey<T>(pMHAMul.pSmQKtV, TruncateType::TrFloor, key_as_bytes);
    k.mmKeyProj = readGPUMatmulKey<T>(pMHAMul.pProj, TruncateType::TrFloor, key_as_bytes);
    return k;
}


// Implementation from gpu_mha.cu
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>

#include "gpu_mha.cuh"

#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_random.h"

#include "fss/gpu_scalarmul.h"
#include "fss/gpu_truncate.cuh"

template <typename T>
__global__ void rotEmbKernel(MHAParams pMHA, int bw, int scale, u64 N, T *X, T *Y)
{
    // the vectors are N x dim_W
    assert(pMHA.dim_W % 2 == 0);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int N = n_seq * dim_W;
    int dim_W_half = pMHA.dim_W / 2;
    if (tid < N)
    {
        // n_seq * dim_W * n_heads
        int temp = tid;
        int head = tid / (pMHA.n_seq * pMHA.dim_W);
        temp = temp % (pMHA.n_seq * pMHA.dim_W);
        int i = temp / pMHA.dim_W;
        int j = temp % pMHA.dim_W;
        auto k = j - (j >= dim_W_half) * dim_W_half;

        double scalar = 1.0 / std::pow(10000.0, (2 * k / (double)pMHA.dim_W));
        float scalarInt = T((i * scalar) * (1ULL << scale)) / (float) (1ULL << scale); 
        const auto uLim = T(1ULL << (scale - 3));
        T sinxi = (T)(i64)(std::sin(scalarInt) * uLim);
        T cosxi = (T)(i64)(std::cos(scalarInt) * uLim);
        if (sinxi == uLim)
            sinxi -= 1;
        if (cosxi == uLim)
            cosxi -= 1;

        auto l = (j + dim_W_half) % pMHA.dim_W;
        T m1 = 2 * (j >= dim_W_half) - 1;
        Y[tid] = cosxi * X[tid] + m1 * sinxi * X[head * pMHA.n_seq * pMHA.dim_W + i * pMHA.dim_W + l];
        gpuMod(Y[tid], bw);
    }
}

template <typename T>
T *gpuKeygenRotEmb(u8 **key_as_bytes, int party, int bw, int scale, MHAParams pMHA, T *d_mask_X, AESGlobalContext *g)
{
    size_t size_X = pMHA.n_heads * (u64)pMHA.n_seq * pMHA.dim_W;
    auto d_mask_X1 = (T *)gpuMalloc(size_X * sizeof(T));
    rotEmbKernel<<<(size_X - 1) / 128 + 1, 128>>>(pMHA, bw, scale, size_X, d_mask_X, d_mask_X1);
    // gpuFree(d_mask_X);
    auto d_mask_truncated_X = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrWithSlack, bw, bw, scale - 3, size_X, d_mask_X1, g);
    gpuFree(d_mask_X1);
    return d_mask_truncated_X;
}

template <typename T>
T *gpuRotEmb(SigmaPeer *peer, int party, int bw, int scale, MHAParams pMHA, GPUTruncateKey<T> trKey, T *d_X, AESGlobalContext *g, Stats *s)
{
    u64 b0 = peer->bytesSent() + peer->bytesReceived();

    size_t size_X = pMHA.n_heads * (u64)pMHA.n_seq * pMHA.dim_W;
    auto d_X1 = (T *)gpuMalloc(size_X * sizeof(T));
    rotEmbKernel<<<(size_X - 1) / 128 + 1, 128>>>(pMHA, bw, scale, size_X, d_X, d_X1);
    // don't free this because QKV is one long array
    // gpuFree(d_X);
    auto d_truncated_X = gpuTruncate<T, T>(bw, bw, TruncateType::TrWithSlack, trKey, scale - 3, peer, party, size_X, d_X1, g, s); //, true);
    gpuFree(d_X1);

    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    s->linear_comm_bytes += (b1 - b0);
    return d_truncated_X;
}

// neha: to fix: maxpool, and make it so the conv2d output is 40 bits???? (bout == 40????)
template <typename T>
T *gpuKeygenMHA(u8 **key_as_bytes, int party, int bw, int scale, MHAParams pMHA, MHAMulParams pMHAMul, T *WQKV, T *YQKV, T *WProj, T *YProj, T *d_mask_X, AESGlobalContext *g)
{
    auto d_mask_QKV = gpuKeygenMatmul(key_as_bytes, party, pMHAMul.pQKV, d_mask_X, WQKV, YQKV, TruncateType::TrFloor, g);
    // this->activation.d_data = d_mask_QKV;

    int QKSz = pMHAMul.pQKV.size_C / 3;
    auto d_mask_Q = d_mask_QKV;
    auto d_mask_K = d_mask_QKV + QKSz;
    auto d_mask_V = d_mask_K + QKSz;
    // this->activation.d_data = d_mask_V;
    if (pMHA.rotEmb)
    {
        d_mask_Q = gpuKeygenRotEmb(key_as_bytes, party, bw, scale, pMHA, d_mask_Q, g);
        d_mask_K = gpuKeygenRotEmb(key_as_bytes, party, bw, scale, pMHA, d_mask_K, g);
    }

    auto d_mask_QKt = gpuKeygenMatmul(key_as_bytes, party, pMHAMul.pQKt, d_mask_Q, d_mask_K, (T *)NULL, TruncateType::TrFloor, g, true);
    if (pMHA.rotEmb)
    {
        gpuFree(d_mask_Q);
        gpuFree(d_mask_K);
    }

    T *d_mask_normQKt = d_mask_QKt;
    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 1)
    {
        T invSqrtDimW = T((1.0f / sqrt(double(pMHA.dim_W))) * (1LL << scale));
        d_mask_normQKt = gpuKeygenScalarMul(key_as_bytes, party, bw, pMHAMul.pQKt.size_C, invSqrtDimW, d_mask_QKt, TruncateType::TrFloor, scale, g);
        gpuFree(d_mask_QKt);
    }

    auto d_mask_smQKt = gpuKeygenSoftmax(key_as_bytes, party, pMHAMul.pMPool, d_mask_normQKt, g);
    gpuFree(d_mask_normQKt);
    // this->activation.d_data = d_mask_smQKt;
    auto d_mask_smQKtV = gpuKeygenMatmul(key_as_bytes, party, pMHAMul.pSmQKtV, d_mask_smQKt, d_mask_V, (T *)NULL, TruncateType::TrFloor, g, true);
    gpuFree(d_mask_smQKt);
    gpuFree(d_mask_QKV);
    // this->activation.d_data = d_mask_smQKtV;

    auto d_mask_proj = gpuKeygenMatmul(key_as_bytes, party, pMHAMul.pProj, d_mask_smQKtV, WProj, YProj, TruncateType::TrFloor, g);
    gpuFree(d_mask_smQKtV);
    // free gpu memory
    // this->activation.d_data = d_mask_proj;
    return d_mask_proj;
}

template <typename T>
T *gpuMHA(SigmaPeer *peer, int party, int bw, int scale, MHAParams pMHA, MHAMulParams pMHAMul, GPUMHAKey<T> k, T *WQKV, T *YQKV, T *WProj, T *YProj, T *d_X, MHATables<T> t, AESGlobalContext *g, Stats *s)
{
    auto b0 = peer->bytesSent() + peer->bytesReceived();

    auto d_QKV = gpuMatmul(peer, party, pMHAMul.pQKV, k.mmKeyQKV, d_X, WQKV, YQKV, TruncateType::TrFloor, g, s);
    // this->activation.d_data = d_QKV;
    size_t QKSz = pMHAMul.pQKV.size_C / 3;
    auto d_Q = d_QKV;
    auto d_K = d_QKV + QKSz;
    auto d_V = d_K + QKSz;
    // this->activation.d_data = d_V;

    if (pMHA.rotEmb)
    {
        d_Q = gpuRotEmb(peer, party, bw, scale, pMHA, k.reQTrKey, d_Q, g, s);
        d_K = gpuRotEmb(peer, party, bw, scale, pMHA, k.reKTrKey, d_K, g, s);
    }

    auto d_QKt = gpuMatmul(peer, party, pMHAMul.pQKt, k.mmKeyQKt, d_Q, d_K, (T *)NULL, TruncateType::TrFloor, g, s, true);
    if (pMHA.rotEmb)
    {
        gpuFree(d_Q);
        gpuFree(d_K);
    }
    // this->activation.d_data = d_QKt;

    T *d_normQKt = d_QKt;

    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 1)
    {
        T invSqrtDimW = T((1.0f / sqrt(double(pMHA.dim_W))) * (1LL << scale));
        d_normQKt = gpuScalarMul(peer, party, bw, pMHAMul.pQKt.size_C, k.normQKtTrKey, invSqrtDimW, d_QKt, TruncateType::TrFloor, scale, g, s);
        gpuFree(d_QKt);
    }

    // assert(d_nExpMsbTab);
    // assert(d_nExpLsbTab);
    // assert(d_invTab);
    // this->activation.d_data = d_normQKt;

    auto d_smQKt = gpuSoftmax(peer, party, pMHAMul.pMPool, k.softmaxKey, d_normQKt, t.d_nExpMsbTab, t.d_nExpLsbTab, t.d_invTab, g, s);
    gpuFree(d_normQKt);
    // this->activation.d_data = d_smQKt;
    auto d_smQKtV = gpuMatmul(peer, party, pMHAMul.pSmQKtV, k.mmKeySmQKtV, d_smQKt, d_V, (T *)NULL, TruncateType::TrFloor, g, s, true);
    gpuFree(d_smQKt);
    gpuFree(d_QKV);
    // // this->activation.d_data = d_smQKtV;
    auto d_proj = gpuMatmul(peer, party, pMHAMul.pProj, k.mmKeyProj, d_smQKtV, WProj, YProj, TruncateType::TrFloor, g, s);
    gpuFree(d_smQKtV);
    auto b1 = peer->bytesSent() + peer->bytesReceived();
    return d_proj;
}
