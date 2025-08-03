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

#include "gpu_relu.cuh"
#include "gpu_lut.cuh"
#include "gpu_truncate.cuh"

/**
 * @brief FSS key structure for GELU multiplexer operation
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * This structure contains secret shares for the multiplexer (MUX)
 * operation used in GELU activation, which selects between two
 * precomputed values based on the input range.
 */
template <typename T>
struct GPUGeluMuxKey
{
    T *c0;  ///< Secret share of first set of MUX values
    T *c1;  ///< Secret share of second set of MUX values
};

/**
 * @brief Read a GPU GELU multiplexer key from a byte stream
 * 
 * @tparam T Data type for the computation
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @param N Number of elements in the GELU operation
 * @return GPUGeluMuxKey<T> Deserialized GELU MUX key
 * 
 * This function deserializes a GELU MUX key containing secret shares
 * for the multiplexer operation. The factor of 4 in memory size
 * accounts for multiple values per element needed in the MUX.
 */
template <typename T>
GPUGeluMuxKey<T> readGPUGeluMuxKey(uint8_t **key_as_bytes, int N)
{
    GPUGeluMuxKey<T> k;
    u64 memSz = 4 * N * sizeof(T); // num bytes
    k.c0 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    k.c1 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

/**
 * @brief FSS key structure for GPU GELU (Gaussian Error Linear Unit) activation
 * 
 * @tparam T Data type for the main computation (e.g., u32, u64)
 * @tparam TClip Data type for clipped values in MUX operation
 * 
 * GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * This structure contains all FSS keys needed for secure GELU computation:
 * 1. Truncation for scaling operations
 * 2. DReLU for sign detection
 * 3. Interval check mask for range detection
 * 4. MUX key for selecting precomputed values
 * 5. LUT key for non-linear function approximation
 * 6. Select key for final conditional multiplication
 */
template <typename T, typename TClip>
struct GPUGeluKey
{
    int bw;                         ///< Bit width of the computation
    GPUTruncateKey<T> trKey;        ///< Key for truncation operations
    GPUDReluKey dReluKey;           ///< Key for sign detection (x > 0)
    u32 *icMask;                    ///< Interval check mask for range detection
    GPUGeluMuxKey<TClip> muxKey;    ///< Key for multiplexer operation
    GPULUTKey<T> lutKey;            ///< Key for lookup table evaluation
    GPUSelectKey<T> reluSelectKey;  ///< Key for final selection/multiplication
};

/**
 * @brief Read a GPU GELU key from a byte stream
 * 
 * @tparam T Data type for the main computation
 * @tparam TClip Data type for clipped values in MUX operation
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUGeluKey<T, TClip> Deserialized GELU key
 * 
 * This function deserializes a complete GELU key from a byte stream,
 * extracting all component keys needed for secure GELU activation:
 * 
 * 1. Bit width configuration
 * 2. Truncation key with slack for intermediate computations
 * 3. DReLU key for sign detection
 * 4. Interval check mask for input range classification
 * 5. MUX key for conditional value selection
 * 6. LUT key for non-linear function approximation
 * 7. Select key for final output computation
 * 
 * The GELU activation is approximated using a combination of interval
 * checks and lookup tables for efficiency while maintaining accuracy.
 */
template <typename T, typename TClip>
GPUGeluKey<T, TClip> readGpuGeluKey(uint8_t **key_as_bytes)
{
    GPUGeluKey<T, TClip> k;
    k.bw = *((int *)*key_as_bytes);
    *key_as_bytes += sizeof(int);
    k.trKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    k.dReluKey = readGPUDReluKey(key_as_bytes);
    int N = k.trKey.N;
    // printf("###### Gelu N=%d\n", N);
    auto icMaskMemSize = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    k.icMask = (u32 *)*key_as_bytes;
    *key_as_bytes += icMaskMemSize;
    k.muxKey = readGPUGeluMuxKey<TClip>(key_as_bytes, N);
    k.lutKey = readGPULUTKey<T>(key_as_bytes);
    k.reluSelectKey = readGPUSelectKey<T>(key_as_bytes, N);
    return k;
}



// Implementation from gpu_gelu.cu
#include "gpu_maxpool.cuh"
#include "gpu_gelu.cuh"

template <typename TIn, typename TOut>
__global__ void keyGenGeluMuxKernel(int party, int bin, int bout, TOut *linFunc, int N, TIn *b0Mask, TIn *b1Mask, TIn *mask_X, TOut *outMask, TOut *c0, TOut *c1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auto muxMask = 2 * b1Mask[i] + b0Mask[i];
        for (int j = 0; j < 4; j++)
        {
            auto idx = 4 * i + j ^ muxMask;
            auto temp = -linFunc[4 + j] * mask_X[i];
            gpuMod(temp, bin);
            c0[idx] = linFunc[j] + /*- linFunc[4 + j] * mask_X[i]*/ temp + outMask[i];
            c1[idx] = linFunc[4 + j];

            gpuMod(c0[idx], bout);
            gpuMod(c1[idx], bout);
        }
        // if(i == 0) printf("MuxKernel %d: Input=%ld, Output=%ld\n", i, mask_X[i], outMask[i]);
    }
}

template <typename TIn, typename TOut>
TOut *keyGenGeluMux(u8 **key_as_bytes, int party, int bin, int bout, const TOut linFunc[2][4], int N, TIn *d_b0Mask, TIn *d_b1Mask, TIn *d_mask_X)
{
    assert(bin <= 8 * sizeof(TIn));
    assert(bout <= 8 * sizeof(TOut));
    auto d_outMask = randomGEOnGpu<TOut>(N, bout);
    // checkCudaErrors(cudaMemset(d_outMask, 0, N * sizeof(TOut)));
    u64 memSzC = 4 * N * sizeof(TOut);
    auto d_c0 = (TOut *)gpuMalloc(memSzC);
    auto d_c1 = (TOut *)gpuMalloc(memSzC);
    auto d_linFunc = (TOut *)moveToGPU((u8 *)linFunc, 8 * sizeof(TOut), NULL);
    keyGenGeluMuxKernel<TIn, TOut><<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, d_linFunc, N, d_b0Mask, d_b1Mask, d_mask_X, d_outMask, d_c0, d_c1);
    writeShares<TOut, TOut>(key_as_bytes, party, 4 * N, d_c0, bout);
    writeShares<TOut, TOut>(key_as_bytes, party, 4 * N, d_c1, bout);
    gpuFree(d_c0);
    gpuFree(d_c1);
    gpuFree(d_linFunc);
    return d_outMask;
}

template <typename TIn, typename TOut>
__global__ void geluMuxKernel(int party, int bin, int bout, int N, u32 *drelu_g, u32 *ic_g, TIn *Xt, TOut *out, TOut *c0_g, TOut *c1_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // assert(bout == 8);
        auto drelu = (drelu_g[i / 32] >> (i % 32)) & u32(1);
        auto ic = (ic_g[i / 32] >> (i % 32)) & u32(1);
        // if(i == 0) printf("dReluBit=%u, ze bit=%u\n", ic, drelu);
        auto c = 4 * i + 2 * ic + drelu;
        auto c0 = TIn(c0_g[c]);
        auto c1 = TIn(c1_g[c]);
        out[i] = c1 * Xt[i] + c0;
        gpuMod(out[i], bout);
        // if(i == 0) printf("MuxKernel %d: Input=%ld, Output=%ld\n", i, Xt[i], out[i]);
    }
}

template <typename TIn, typename TOut>
TOut *geluMux(SigmaPeer *peer, int party, GPUGeluMuxKey<TOut> k, int bin, int bout, int N, u32 *d_drelu, u32 *d_ic, TIn *d_Xt, Stats *s)
{
    // assert(bout == 8);
    assert(bout <= 8 * sizeof(TOut));
    auto d_out = (TOut *)gpuMalloc(N * sizeof(TOut));
    u64 memSzC = 4 * N * sizeof(TOut);
    auto d_c0 = (TOut *)moveToGPU((u8 *)k.c0, memSzC, s);
    auto d_c1 = (TOut *)moveToGPU((u8 *)k.c1, memSzC, s);
    geluMuxKernel<TIn, TOut><<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, N, d_drelu, d_ic, d_Xt, d_out, d_c0, d_c1);
    gpuFree(d_c0);
    gpuFree(d_c1);
    peer->reconstructInPlace(d_out, bout, N, s);
    return d_out;
}

template <typename T, typename TClip, int clipBw>
T *gpuKeyGenGelu(uint8_t **key_as_bytes, int party, int bw, int bin, int scale, int N, T *d_mask_X, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bw);
    int bwXt = bin - scale + 6 + 1;
    // truncated X = Xt
    auto d_mask_Xt = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrWithSlack, bw, bwXt, scale - 6, N, d_mask_X, gaes);
    auto d_dReluMask = gpuKeyGenDRelu(key_as_bytes, party, bwXt, N, d_mask_Xt, gaes);
    assert(bwXt > 7);
    // printf("ClipBW=%d\n", clipBw);
    assert(8 * sizeof(TClip) >= clipBw);
    const u64 max = (1ULL << clipBw) - 1;
    auto d_icMask = gpuKeyGenIc<T, max, -max>(key_as_bytes, party, bwXt, N, d_mask_Xt, false, gaes);
    const TClip linFunc[2][4] = {
        {TClip(max), TClip(max), 0, 0},
        {0, 0, TClip(-1), 1}};
    auto d_clipMask = keyGenGeluMux<T, TClip>(key_as_bytes, party, bwXt, clipBw, linFunc, N, d_dReluMask, d_icMask, d_mask_Xt);
    auto d_lutMask = gpuKeyGenLUT<TClip, T>(key_as_bytes, party, clipBw, bw, N, d_clipMask, gaes);
    gpuFree(d_clipMask);

    // auto d_reluMask = randomGEOnGpu<T>(N, bw);
    auto d_reluMask = gpuKeyGenSelect<T, T>(key_as_bytes, party, N, d_mask_X, d_dReluMask, bw);

    gpuLinearComb(bw, N, d_reluMask, T(1), d_reluMask, -T(1), d_lutMask);
    gpuFree(d_lutMask);
    return d_reluMask;
}

// clip happens in place
template <typename T, typename TClip, int clipBw>
T *gpuGelu(SigmaPeer *peer, int party, GPUGeluKey<T, TClip> &k, int bw, int bin, int scale, int N, T *d_X, T *d_geluSubRelu, AESGlobalContext *gaes, Stats *s)
{
    assert(8 * sizeof(TClip) >= clipBw);
    assert(bin > scale - 6);
    int bwXt = bin - scale + 6 + 1;
    // do a truncate reduce
    auto d_Xt = gpuTruncate(bw, bwXt, TruncateType::TrWithSlack, k.trKey, scale - 6, peer, party, N, d_X, gaes, s);
    // the -1 doesn't matter because anything larger is anyway set to (1 << clipBw) - 1
    const u64 clipVal = (1ULL << clipBw) - 1;
    std::vector<u32 *> h_masks({k.dReluKey.mask, k.icMask});
    u32 *d_res = gpuDcf_<T, 3, geluPrologue<clipVal, -clipVal>, geluEpilogue<clipVal, -clipVal>>(k.dReluKey.dpfKey, party, d_Xt, gaes, s, &h_masks);
    int numInts = ((N - 1) / PACKING_SIZE + 1);
    peer->reconstructInPlace(d_res, 1, 2 * numInts * 32, s);

    u32 *d_dRelu = d_res;
    u32 *d_ic = d_res + numInts;
    auto d_clippedX = geluMux<T, TClip>(peer, party, k.muxKey, bwXt, clipBw, N, d_dRelu, d_ic, d_Xt, s);
    gpuFree(d_Xt);
    auto d_reluSubGelu = gpuDpfLUT<TClip, T>(k.lutKey, peer, party, d_clippedX, d_geluSubRelu, gaes, s, false);
    gpuFree(d_clippedX);
    T *d_relu = gpuSelect<T, T, 0, 0>(peer, party, bw, k.reluSelectKey, d_dRelu, d_X, s, false);
    gpuFree(d_res);
    gpuLinearComb(bw, N, d_relu, T(1), d_relu, -T(1), d_reluSubGelu);
    gpuFree(d_reluSubGelu);
    peer->reconstructInPlace(d_relu, bw, N, s);
    return d_relu;
}