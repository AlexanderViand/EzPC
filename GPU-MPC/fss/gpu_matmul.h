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
#include "gpu_truncate.h"

/**
 * @brief Parameters for GPU matrix multiplication operations
 * 
 * This structure defines all necessary parameters for performing secure
 * matrix multiplication on GPU using Function Secret Sharing (FSS).
 */
struct MatmulParams
{
    // multiply two matrices of input bitwidth bw
    // truncate by shift and return the output in output bitwidth bw
    int bw;     ///< Bit width of the computation
    int shift;  ///< Number of bits to shift for truncation after multiplication
    
    int batchSz;  ///< Batch size for batched matrix multiplication
    int M;        ///< Number of rows in matrix A and result matrix C
    int K;        ///< Number of columns in A / rows in B (inner dimension)
    int N;        ///< Number of columns in matrix B and result matrix C
    
    int size_A;    ///< Total number of elements in matrix A
    int size_B;    ///< Total number of elements in matrix B
    int size_C;    ///< Total number of elements in result matrix C
    
    int ld_A;      ///< Leading dimension of matrix A
    int ld_B;      ///< Leading dimension of matrix B
    int ld_C;      ///< Leading dimension of matrix C
    
    int stride_A;  ///< Stride between consecutive matrices in batch for A
    int stride_B;  ///< Stride between consecutive matrices in batch for B
    int stride_C;  ///< Stride between consecutive matrices in batch for C
    
    bool rowMaj_A;  ///< True if matrix A is in row-major format
    bool rowMaj_B;  ///< True if matrix B is in row-major format
    bool rowMaj_C;  ///< True if matrix C is in row-major format
    
    bool cIsLowerTriangular = false;  ///< True if output matrix C should be lower triangular
};

/**
 * @brief FSS key structure for GPU matrix multiplication
 * 
 * @tparam T Data type for the matrices (e.g., u32, u64)
 * 
 * This structure contains the secret-shared matrices and truncation keys
 * needed for secure matrix multiplication.
 */
template <typename T>
struct GPUMatmulKey
{
    // MatmulParams p;
    u64 mem_size_A;  ///< Memory size in bytes for matrix A
    u64 mem_size_B;  ///< Memory size in bytes for matrix B
    u64 mem_size_C;  ///< Memory size in bytes for matrix C
    
    T *A;  ///< Secret share of matrix A
    T *B;  ///< Secret share of matrix B
    T *C;  ///< Secret share of matrix C (used for masking)
    
    GPUTruncateKey<T> trKey;  ///< Truncation key for post-multiplication truncation
};

/**
 * @brief Read a GPU matrix multiplication key from a byte stream
 * 
 * @tparam T Data type for the matrices
 * @param p Matrix multiplication parameters
 * @param t Truncation type to use
 * @param key_as_bytes Pointer to byte stream containing the serialized key
 * @return GPUMatmulKey<T> Deserialized matrix multiplication key
 * 
 * This function deserializes a matrix multiplication key from a byte stream,
 * extracting the secret-shared matrices and truncation keys.
 */
template <typename T>
GPUMatmulKey<T> readGPUMatmulKey(MatmulParams p, TruncateType t, uint8_t **key_as_bytes)
{
    GPUMatmulKey<T> k;
    k.mem_size_A = p.size_A * sizeof(T);
    k.mem_size_B = p.size_B * sizeof(T);
    k.mem_size_C = p.size_C * sizeof(T);
    k.A = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_A;
    k.B = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_B;
    k.C = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_C;
    k.trKey = readGPUTruncateKey<T>(t, key_as_bytes);
    return k;
}

/**
 * @brief Initialize standard parameters for matrix multiplication
 * 
 * @param p Reference to MatmulParams structure to initialize
 * @param bw Bit width for the computation
 * @param scale Scale factor (used as shift amount for truncation)
 * 
 * This function sets up standard parameters for matrix multiplication,
 * including leading dimensions, strides, and sizes based on the matrix
 * dimensions already set in the structure.
 * 
 * @note Assumes p.M, p.K, p.N, and p.batchSz are already set
 * @note Sets all matrices to row-major format by default
 */
void stdInit(MatmulParams &p, int bw, int scale)
{
    p.bw = bw;
    p.shift = scale;

    p.ld_A = p.K;
    p.ld_B = p.N;
    p.ld_C = p.N;

    p.stride_A = p.M * p.K;
    p.stride_B = p.K * p.N;
    p.stride_C = p.M * p.N;

    p.size_A = p.batchSz * p.M * p.K;
    p.size_B = p.batchSz * p.K * p.N;

    if (p.cIsLowerTriangular)
    {
        assert(p.M == p.N);
        p.size_C = p.batchSz * (p.M * (p.M + 1)) / 2;
    }
    else
    {
        p.size_C = p.batchSz * p.M * p.N;
    }

    p.rowMaj_A = true;
    p.rowMaj_B = true;
    p.rowMaj_C = true;
}

#include "gpu_matmul.cu"