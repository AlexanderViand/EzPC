// Author: GPU-MPC
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

#include "gpu_comms.cuh"

// Non-template kernel implementation moved from header
__global__ void addMod4(int numInts, u32 *A, u32 *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numInts)
    {
        u32 x = A[j];
        u32 y = B[j];
        u32 z = 0;
        for (int i = 0; i < 32; i += 2)
        {
            u32 a = (x >> i) & 3;
            u32 b = (y >> i) & 3;
            u32 c = ((a + b) & 3) << i;
            z |= c;
        }
        A[j] = z;
    }
}