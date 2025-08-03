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

#include "gpu_aes_shm.cuh"

void initAESContext(AESGlobalContext *g)
{
	g->t0_g = (u32 *)moveToGPU((u8 *)T0, AES_128_TABLE_SIZE * sizeof(u32), NULL);
	g->Sbox_g = (u8 *)moveToGPU((u8 *)Sbox_g, 256 * sizeof(u8), NULL);
	g->t4_0G = (u32 *)moveToGPU((u8 *)T4_0, AES_128_TABLE_SIZE * sizeof(u32), NULL);
	g->t4_1G = (u32 *)moveToGPU((u8 *)T4_1, AES_128_TABLE_SIZE * sizeof(u32), NULL);
	g->t4_2G = (u32 *)moveToGPU((u8 *)T4_2, AES_128_TABLE_SIZE * sizeof(u32), NULL);
	g->t4_3G = (u32 *)moveToGPU((u8 *)T4_3, AES_128_TABLE_SIZE * sizeof(u32), NULL);
}

void freeAESGlobalContext(AESGlobalContext *g)
{
	gpuFree(g->t0_g);
	gpuFree(g->Sbox_g);
	gpuFree(g->t4_0G);
	gpuFree(g->t4_1G);
	gpuFree(g->t4_2G);
	gpuFree(g->t4_3G);
}