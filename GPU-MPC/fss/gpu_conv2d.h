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

/**
 * @brief Parameters for 2D convolution operations on GPU
 * 
 * This structure defines all necessary parameters for performing secure
 * 2D convolution operations using Function Secret Sharing (FSS).
 */
struct Conv2DParams
{
    int bin;   ///< Input bit width
    int bout;  ///< Output bit width
    int N;     ///< Batch size (number of images)
    int H;     ///< Height of input images
    int W;     ///< Width of input images
    int CI;    ///< Number of input channels
    int FH;    ///< Filter height (kernel height)
    int FW;    ///< Filter width (kernel width)
    int CO;    ///< Number of output channels
    
    int zPadHLeft;   ///< Zero padding on left side (height dimension)
    int zPadHRight;  ///< Zero padding on right side (height dimension)
    int zPadWLeft;   ///< Zero padding on left side (width dimension)
    int zPadWRight;  ///< Zero padding on right side (width dimension)
    
    int strideH;  ///< Stride in height dimension
    int strideW;  ///< Stride in width dimension
    int OH;       ///< Output height (computed by fillConv2DParams)
    int OW;       ///< Output width (computed by fillConv2DParams)
    
    size_t size_I;  ///< Total number of elements in input tensor
    size_t size_F;  ///< Total number of elements in filter tensor
    size_t size_O;  ///< Total number of elements in output tensor
};

/**
 * @brief FSS key structure for GPU 2D convolution
 * 
 * @tparam T Data type for the computation (e.g., u32, u64)
 * 
 * This structure contains the secret-shared tensors needed for secure
 * convolution computation. The tensors are laid out in NHWC format
 * (batch, height, width, channels) for optimal GPU performance.
 */
template <typename T>
struct GPUConv2DKey
{
    Conv2DParams p;  ///< Convolution parameters
    
    size_t mem_size_I;  ///< Memory size in bytes for input tensor
    size_t mem_size_F;  ///< Memory size in bytes for filter tensor
    size_t mem_size_O;  ///< Memory size in bytes for output tensor
    
    T *I;  ///< Secret share of input tensor (NHWC format)
    T *F;  ///< Secret share of filter/kernel tensor
    T *O;  ///< Secret share of output tensor (used for masking)
};

/**
 * @brief Initialize convolution parameters with computed output dimensions
 * 
 * @param p Pointer to Conv2DParams structure to initialize
 * 
 * This function computes the output dimensions (OH, OW) and tensor sizes
 * based on the input parameters, padding, and stride settings.
 * 
 * @note Must be called after setting all input parameters but before
 * using the Conv2DParams structure for convolution operations.
 * 
 * Output dimensions are computed as:
 * - OH = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1
 * - OW = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1
 */
void fillConv2DParams(Conv2DParams *p)
{
    p->OH = ((p->H - p->FH + (p->zPadHLeft + p->zPadHRight)) / p->strideH) + 1;
    p->OW = ((p->W - p->FW + (p->zPadWLeft + p->zPadWRight)) / p->strideW) + 1;
    p->size_I = p->N * p->H * p->W * p->CI;
    p->size_F = p->CO * p->FH * p->FW * p->CI;
    p->size_O = p->N * p->OH * p->OW * p->CO;
}

#include "gpu_conv2d.cu"
