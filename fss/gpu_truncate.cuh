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

#include <cuda.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_stats.h"

#include "gpu_relu.cuh"
#include "gpu_dpf_templates.h"

using GPUMaskedDPFKey = GPUDReluKey;
constexpr auto readGPUMaskedDPFKey = readGPUDReluKey;

template <typename T>
struct GPUTrCorrKey
{
    GPUMaskedDPFKey mDpfKey;
    T *corr;
};

template <typename T>
struct GPUTruncateKey
{
    int bin, shift, bout, N;
    GPUTrCorrKey<T> lsbKey, msbKey;
};

enum TruncateType
{
    None,
    LocalLRS,
    LocalARS,
    TrWithSlack,
    TrFloor
};

template <typename T>
GPUTrCorrKey<T> readGPUTrCorrKey(u8 **key_as_bytes)
{
    GPUTrCorrKey<T> k;
    k.mDpfKey = readGPUMaskedDPFKey(key_as_bytes);
    size_t memSz = k.mDpfKey.dpfKey.M * sizeof(T);
    k.corr = (T *)*key_as_bytes;
    *key_as_bytes += 2 * memSz;
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTrWithSlackKey(uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    memcpy(&k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);
    k.lsbKey = readGPUTrCorrKey<T>(key_as_bytes);
    // correct the msb only if needed
    size_t memSz = k.N * sizeof(T);
    k.msbKey.corr = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTrFloorKey(uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    memcpy(&k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);

    k.lsbKey = readGPUTrCorrKey<T>(key_as_bytes);
    if (k.bout > k.bin - k.shift)
        k.msbKey = readGPUTrCorrKey<T>(key_as_bytes);
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTruncateKey(TruncateType t, uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    switch (t)
    {
    case TruncateType::TrWithSlack:
        k = readGPUTrWithSlackKey<T>(key_as_bytes);
        break;
    case TruncateType::TrFloor:
        k = readGPUTrFloorKey<T>(key_as_bytes);
        break;
    default:
        assert(t == TruncateType::None || t == TruncateType::LocalARS || t == TruncateType::LocalLRS);
    }
    return k;
}

template <typename T>
void checkTrFloor(int bin, int bout, int shift, int N, T *h_masked_A, T *h_mask_A, T *h_A_ct)
{
    // printf("N=%d\n", N);
    for (int i = 0; i < N; i++)
    {
        auto truncated_A = cpuArs(h_A_ct[i], bin, shift);
        cpuMod(truncated_A, bout);
        auto output = h_masked_A[i] - h_mask_A[i];
        cpuMod(output, bout);
        auto diff = output - truncated_A;
        if (i < 10 || diff != T(0))
            printf("%d: %ld %ld %ld %ld, %ld, %ld\n", i, u64(output), u64(truncated_A), u64(h_A_ct[i]), h_masked_A[i], h_mask_A[i], h_A_ct[i]);
        assert(diff == T(0));
    }
}

template <typename T>
void checkTrStochastic(int bin, int bout, int shift, int N, T *h_masked_O, T *h_mask_O, T *h_I, T *h_r)
{
    for (int i = 0; i < N; i++)
    {
        auto unmasked_o = h_masked_O[i] - h_mask_O[i];
        cpuMod(unmasked_o, bout);
        auto trInp = (h_I[i] + (1ULL << (bin - 1))) >> shift;
        cpuMod(trInp, bin - shift);
        T temp = h_I[i];
        cpuMod(temp, shift);
        if (h_r[i] < temp)
        {
            trInp += 1;
            cpuMod(trInp, bin - shift);
        }
        trInp -= (1ULL << (bin - shift - 1));
        cpuMod(trInp, bout);
        if (i < 10 || unmasked_o != trInp)
            printf("%d=%lu %lu %lu %lu %u\n", i, unmasked_o, trInp, h_r[i], temp, h_r[i] < temp);
        assert(unmasked_o == trInp); // <= 1);
    }
}


// Implementation from gpu_truncate.cu
#pragma once

#include <cassert>

#include "utils/gpu_data_types.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_comms.cuh"

#include "gpu_truncate.cuh"
#include "gpu_local_truncate.h"

template <typename TIn, typename TOut>
using trFunc = TOut (*)(int party, int bin, int shift, int i, TIn x, u8 *bytes);

template <typename TIn, typename TOut>
using keygenTrFunc = void (*)(int party, int bin, int shift, int bout, int N, int i, TIn x, TOut y, TIn z, TOut *trKey, u8 *bytes);



template <typename TIn, typename TOut>
__device__ TOut trReduce(int party, int bin, int shift, int i, TIn x, u8 *bytes)
{
    return (party == SERVER1) * TOut(x >> shift);
}

template <typename TIn, typename TOut>
__device__ TOut signExtend(int party, int bin, int shift, int i, TIn x, u8 *bytes)
{
    // if(i == 0) printf("sign extend x=%lu\n", x);
    return (party == SERVER1) * (TOut(x) - (1ULL << (bin - 1)));
}

template <typename TIn, typename TOut>
__device__ TOut trWithSlack(int party, int bin, int shift, int i, TIn x, u8 *bytes)
{
    auto x1 = (x + (1ULL << (bin - 2)));
    gpuMod(x1, bin);
    auto msb_x1 = gpuMsb(x1, bin);
    return (party == SERVER1) * TOut((x1 >> shift) - (1ULL << (bin - shift - 2))) + ((TOut *)bytes)[i] * (!msb_x1);
}

template <typename TIn, typename TOut, trFunc<TIn, TOut> tf>
__global__ void trCorrKernel(int party, int bin, int shift, int bout, int N, TIn *x, u32 *z_g, TOut *corr, TOut *y, u8 *bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        u32 z = (z_g[i / 32] >> (threadIdx.x & 0x1f)) & 1;
        auto y_l = (TOut)tf(party, bin, shift, i, x[i], bytes) + corr[2 * i + z];
        gpuMod(y_l, bout);
        y[i] = y_l;
    }
}

template <typename TIn, typename TOut>
__device__ void keygenTrReduce(int party, int bin, int shift, int bout, int N, int i, TIn x, TOut y, TIn z, TOut *trKey, u8 *bytes)
{
    auto corr = trKey;
    auto q = -TOut(x >> shift) + y;
    gpuMod(q, bout);
    auto qM1 = q - 1;
    gpuMod(qM1, bout);
    corr[2 * i + z] = q;
    corr[2 * i + (z ^ 1)] = qM1;
}

template <typename TIn, typename TOut>
__device__ void keygenSignExtend(int party, int bin, int shift, int bout, int N, int i, TIn x, TOut y, TIn z, TOut *trKey, u8 *bytes)
{
    auto corr = trKey;
    auto q = -TOut(x) + y;
    gpuMod(q, bout);
    auto r = q + (1ULL << bin);
    gpuMod(r, bout);
    corr[2 * i + z] = q;
    corr[2 * i + (z ^ 1)] = r;
    // if(i == 0) printf("sign extend mask=%lu, %lu, %lu, %lu\n", y, q, r, z);
}

template <typename TIn, typename TOut>
__device__ void keygenTrWithSlack(int party, int bin, int shift, int bout, int N, int i, TIn x, TOut y, TIn z, TOut *trKey, u8 *bytes)
{
    keygenTrReduce(party, bin, shift, bout, N, i, x, y, z, trKey, bytes);
    trKey[2 * N + i] = TOut(gpuMsb(x, bin) * (1ULL << (bin - shift)));
}

template <typename TIn, typename TOut, keygenTrFunc<TIn, TOut> tf>
__global__ void keygenTrFuncKernel(int party, int bin, int shift, int bout, int N, TIn *x, TOut *y, TIn *z, TOut *trKey, u8 *bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        tf(party, bin, shift, bout, N, i, x[i], y[i], z[i], trKey, bytes);
    }
}

template <typename TIn, typename TOut, keygenTrFunc<TIn, TOut> tf, u64 m>
TOut *gpuKeygenTrFunc(u8 **key_as_bytes, int party, int bin, int shift, int bout, int N, int bwToCmp, TIn *d_inputMask, AESGlobalContext *gaes, u8 *bytes = NULL)
{
    assert(bin >= shift);
    gpuKeyGenDCF_(key_as_bytes, party, bwToCmp, N, d_inputMask, gaes);
    auto d_dcfMask = randomGEOnGpu<TIn>(N, 1);
    TOut *d_outMask = randomGEOnGpu<TOut>(N, bout);
    // cudaMemset(d_outMask, 0, N * sizeof(TOut));
    auto d_trKey = (TOut *)gpuMalloc(m * N * sizeof(TOut));
    keygenTrFuncKernel<TIn, TOut, tf><<<(N - 1) / 128 + 1, 128>>>(party, bin, shift, bout, N, d_inputMask, d_outMask, d_dcfMask, d_trKey, bytes);
    checkCudaErrors(cudaDeviceSynchronize());

    writeShares<TIn, TIn>(key_as_bytes, party, N, d_dcfMask, 1);
    writeShares<TOut, TOut>(key_as_bytes, party, m * N, (TOut *)d_trKey, bout);
    gpuFree(d_dcfMask);
    gpuFree(d_trKey);
    return d_outMask;
}

template <typename TIn, typename TOut>
TOut *genGPUTrWithSlackKey(uint8_t **key_as_bytes, int party, int bin, int shift, int bout, int N, TIn *d_inputMask, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, shift);
    writeInt(key_as_bytes, bout);
    writeInt(key_as_bytes, N);

    auto d_outputMask = gpuKeygenTrFunc<TIn, TOut, keygenTrWithSlack<TIn, TOut>, 3>(key_as_bytes, party, bin, shift, bout, N, shift, d_inputMask, gaes);
    return d_outputMask;
}

template <typename TIn, typename TOut>
TOut *genGPUSignExtendKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, TIn *d_inputMask, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, bout);
    writeInt(key_as_bytes, N);
    auto d_outMask = gpuKeygenTrFunc<TIn, TOut, keygenSignExtend<TIn, TOut>, 2>(key_as_bytes, party, bin, 0, bout, N, bin, d_inputMask, gaes);
    return d_outMask;
}

template <typename TIn, typename TOut>
TOut *genGPUTrFloorKey(uint8_t **key_as_bytes, int party, int bin, int shift, int bout, int N, TIn *d_inputMask, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, shift);
    writeInt(key_as_bytes, bout);
    writeInt(key_as_bytes, N);

    auto d_trMask = gpuKeygenTrFunc<TIn, TOut, keygenTrReduce<TIn, TOut>, 2>(key_as_bytes, party, bin, shift, bin - shift, N, shift, d_inputMask, gaes);
    auto d_outMask = d_trMask;
    if (bout > bin - shift)
    {
        d_outMask = gpuKeygenTrFunc<TOut, TOut, keygenSignExtend<TOut, TOut>, 2>(key_as_bytes, party, bin - shift, 0, bout, N, bin - shift, d_trMask, gaes);
        gpuFree(d_trMask);
    }
    return d_outMask;
}

template <typename TIn, typename TOut>
TOut *genGPUTruncateKey(uint8_t **key_as_bytes, int party, TruncateType t, int bin, int bout, int shift, int N, TIn *d_inMask, AESGlobalContext *gaes)
{
    assert(shift > 0 || t == TruncateType::None);
    TOut *d_outMask;
    switch (t)
    {
    case TruncateType::TrWithSlack:
        // printf("%d, %d, %d\n", bout, bin, shift, bin - shift);
        // assert(bout > bin - shift);
        d_outMask = genGPUTrWithSlackKey<TIn, TOut>(key_as_bytes, party, bin, shift, bout, N, d_inMask, gaes);
        break;
    case TruncateType::TrFloor:
        d_outMask = genGPUTrFloorKey<TIn, TOut>(key_as_bytes, party, bin, shift, bout, N, d_inMask, gaes);
        break;
    case TruncateType::LocalARS:
        d_outMask = (TOut *)gpuLocalTr<TIn, TOut, ars>(party, bin, shift, N, d_inMask);
        break;
    case TruncateType::LocalLRS:
        d_outMask = (TOut *)gpuLocalTr<TIn, TOut, lrs>(party, bin, shift, N, d_inMask);
        break;
    default:
        d_outMask = (TOut *)d_inMask;
        assert(t == TruncateType::None);
    }
    return d_outMask;
}

template <typename TIn, typename TOut, trFunc<TIn, TOut> tf>
TOut *gpuTrHelper(int party, SigmaPeer *peer, int bin, int shift, int bout, int N, GPUTrCorrKey<TOut> k, TIn *d_X, AESGlobalContext *g, Stats *s, u8 *d_bytes = NULL)
{
    // printf("%d, %d, %d, %d, %d, %d\n", bin, shift, bout, N, k.mDpfKey.dpfKey.bin, k.mDpfKey.dpfKey.N);
    std::vector<u32 *> mask({k.mDpfKey.mask});
    // get a masked dcf as output
    auto d_b = gpuDcf_<TIn, 1, idPrologue, maskEpilogue>(k.mDpfKey.dpfKey, party, d_X, g, s, &mask);
    peer->reconstructInPlace(d_b, 1, N, s);
    size_t memSz = N * sizeof(TOut);
    auto d_corr = (TOut *)moveToGPU((u8 *)k.corr, 2 * memSz, s);
    auto d_O = (TOut *)gpuMalloc(memSz);
    trCorrKernel<TIn, TOut, tf><<<(N - 1) / 128 + 1, 128>>>(party, bin, shift, bout, N, d_X, d_b, d_corr, d_O, d_bytes);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_corr);
    gpuFree(d_b);

    peer->reconstructInPlace(d_O, bout, N, s);

    return d_O;
}

template <typename TIn, typename TOut>
TOut *gpuSignExtend(int party, SigmaPeer *peer, int bin, int bout, int N, GPUTrCorrKey<TOut> k, TIn *d_X, AESGlobalContext *g, Stats *s)
{
    auto d_Y = (TIn *)gpuMalloc(N * sizeof(TOut));
    gpuLinearComb(bin, N, d_Y, TIn(1), d_X, TIn(1ULL << (bin - 1)));
    auto d_O = gpuTrHelper<TIn, TOut, signExtend<TIn, TOut>>(party, peer, bin, 0, bout, N, k, d_Y, g, s);
    gpuFree(d_Y);
    return d_O;
}

template <typename TIn, typename TOut>
TOut *gpuTrFloor(GPUTruncateKey<TOut> k, int party, SigmaPeer *peer, TIn *d_X, AESGlobalContext *g, Stats *s)
{
    auto d_trX = gpuTrHelper<TIn, TOut, trReduce<TIn, TOut>>(party, peer, k.bin, k.shift, k.bin - k.shift, k.N, k.lsbKey, d_X, g, s);
    auto d_O = d_trX;
    if (k.bout > k.bin - k.shift)
    {
        d_O = gpuSignExtend<TOut, TOut>(party, peer, k.bin - k.shift, k.bout, k.N, k.msbKey, d_trX, g, s);
        gpuFree(d_trX);
    }
    return d_O;
}

template <typename TIn, typename TOut>
TOut *gpuTrWithSlack(GPUTruncateKey<TOut> k, int party, SigmaPeer *peer, TIn *d_X, AESGlobalContext *g, Stats *s)
{
    auto d_msbCorr = (u8 *)moveToGPU((u8 *)k.msbKey.corr, k.N * sizeof(TOut), s);
    auto d_O = gpuTrHelper<TIn, TOut, trWithSlack>(party, peer, k.bin, k.shift, k.bout, k.N, k.lsbKey, d_X, g, s, (u8 *)d_msbCorr);
    gpuFree(d_msbCorr);
    return d_O;
}

template <typename TIn, typename TOut>
TOut *gpuTruncate(int bin, int bout, TruncateType t, GPUTruncateKey<TOut> k, int shift, SigmaPeer *peer, int party, int N, TIn *d_I, AESGlobalContext *gaes, Stats *s)
{
    TOut *d_O;
    switch (t)
    {
    case TruncateType::LocalLRS:
        // static_assert(std::is_same<TIn, TOut>::value);
        d_O = (TOut *)gpuLocalTr<TIn, TOut, lrs>(party, bin, shift, N, d_I);
        break;
    case TruncateType::LocalARS:
        // static_assert(std::is_same<TIn, TOut>::value);
        d_O = (TOut *)gpuLocalTr<TIn, TOut, ars>(party, bin, shift, N, d_I);
        break;
    case TruncateType::TrWithSlack:
        // assert(bout > bin - shift);
        d_O = gpuTrWithSlack(k, party, peer, d_I, gaes, s);
        break;
    case TruncateType::TrFloor:
        d_O = gpuTrFloor(k, party, peer, d_I, gaes, s);
        break;
    case TruncateType::None:
        // static_assert(std::is_same<TIn, TOut>::value);
        d_O = (TOut *)d_I;
        break;
    default:
        assert(0 && "unknown truncate type!");
    }
    return d_O;
}
