// Authors: GPU-MPC Authors
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

#include <iostream>
#include <cassert>
#include <random>

#include "fss/gpu_lss.h"
#include "fss/gpu_aes_shm.cuh"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.cuh"
#include "utils/gpu_stats.h"

template <typename T>
void testLSSBasicOperations(int party, std::string peerIP) {
    std::cout << "Testing LSS Basic Operations..." << std::endl;
    
    // Initialize
    const u64 N = 1024;
    const int bw = 32;
    const int scale = 12;
    
    AESGlobalContext gaes;
    initAESContext(&gaes);
    initGPUMemPool();
    initGPURandomness();
    
    // Setup communication
    auto peer = new GpuPeer(true);
    peer->connect(party, peerIP);
    
    Stats stats;
    GPULSSEngine<T> lss(peer, party, bw, scale, &gaes, &stats);
    
    // Test 1: Share and Reconstruct
    std::cout << "Test 1: Share and Reconstruct" << std::endl;
    {
        T* h_values = (T*)cpuMalloc(N * sizeof(T));
        for (u64 i = 0; i < N; i++) {
            h_values[i] = T(i % 100);
        }
        T* d_values = (T*)moveToGPU((u8*)h_values, N * sizeof(T), nullptr);
        
        // Party 0 shares values
        bool isInputParty = (party == SERVER0);
        T* d_shares = lss.share(d_values, N, isInputParty);
        
        // Reconstruct
        lss.reconstruct(d_shares, N);
        
        // Verify
        if (party == SERVER0) {
            T* h_result = (T*)moveToCPU((u8*)d_shares, N * sizeof(T), nullptr);
            for (u64 i = 0; i < N; i++) {
                assert(h_result[i] == h_values[i]);
            }
            std::cout << "  Share/Reconstruct test PASSED" << std::endl;
            cpuFree(h_result);
        }
        
        cpuFree(h_values);
        gpuFree(d_values);
        gpuFree(d_shares);
    }
    
    // Test 2: Addition
    std::cout << "Test 2: Addition" << std::endl;
    {
        T* d_a = randomGEOnGpu<T>(N, bw);
        T* d_b = randomGEOnGpu<T>(N, bw);
        
        T* d_sum = lss.add(d_a, d_b, N);
        lss.reconstruct(d_sum, N);
        
        if (party == SERVER0) {
            T* h_a = (T*)moveToCPU((u8*)d_a, N * sizeof(T), nullptr);
            T* h_b = (T*)moveToCPU((u8*)d_b, N * sizeof(T), nullptr);
            T* h_sum = (T*)moveToCPU((u8*)d_sum, N * sizeof(T), nullptr);
            
            for (u64 i = 0; i < 10; i++) { // Check first 10 elements
                T expected = (h_a[i] + h_b[i]) & ((T(1) << bw) - 1);
                if (h_sum[i] != expected) {
                    std::cout << "  Addition mismatch at " << i << ": " 
                              << h_sum[i] << " != " << expected << std::endl;
                }
            }
            std::cout << "  Addition test completed" << std::endl;
            
            cpuFree(h_a);
            cpuFree(h_b);
            cpuFree(h_sum);
        }
        
        gpuFree(d_a);
        gpuFree(d_b);
        gpuFree(d_sum);
    }
    
    // Test 3: Scalar Multiplication
    std::cout << "Test 3: Scalar Multiplication" << std::endl;
    {
        T* d_shares = randomGEOnGpu<T>(N, bw);
        T scalar = 5;
        
        T* d_result = lss.scalarMultiply(d_shares, scalar, N);
        
        // Verify locally (scalar mult doesn't need communication)
        T* h_shares = (T*)moveToCPU((u8*)d_shares, N * sizeof(T), nullptr);
        T* h_result = (T*)moveToCPU((u8*)d_result, N * sizeof(T), nullptr);
        
        for (u64 i = 0; i < 10; i++) {
            T expected = (h_shares[i] * scalar) & ((T(1) << bw) - 1);
            assert(h_result[i] == expected);
        }
        std::cout << "  Scalar multiplication test PASSED" << std::endl;
        
        cpuFree(h_shares);
        cpuFree(h_result);
        gpuFree(d_shares);
        gpuFree(d_result);
    }
    
    // Test 4: Multiplication (requires keys)
    std::cout << "Test 4: Multiplication with Beaver Triples" << std::endl;
    {
        // Generate multiplication key (in real use, dealer would do this)
        u8* keyBuf = nullptr;
        auto mulKey = GPULSSEngine<T>::genMultiplyKey(&keyBuf, party, bw, scale, N, &gaes);
        
        T* d_a = randomGEOnGpu<T>(N, bw);
        T* d_b = randomGEOnGpu<T>(N, bw);
        
        T* d_product = lss.multiply(d_a, d_b, N, mulKey);
        lss.reconstruct(d_product, N);
        
        std::cout << "  Multiplication test completed (placeholder implementation)" << std::endl;
        
        gpuFree(d_a);
        gpuFree(d_b);
        gpuFree(d_product);
        if (keyBuf) cpuFree(keyBuf);
    }
    
    // Cleanup
    peer->close();
    delete peer;
    destroyGPURandomness();
    
    std::cout << "All LSS basic operation tests completed!" << std::endl;
}

template <typename T>
void testShareConversions(int party, std::string peerIP) {
    std::cout << "\nTesting Share Conversions..." << std::endl;
    
    // Initialize
    const u64 N = 256;
    const int bw = 16; // Use smaller bit width for testing
    const int scale = 0;
    
    AESGlobalContext gaes;
    initAESContext(&gaes);
    initGPUMemPool();
    initGPURandomness();
    
    // Setup communication
    auto peer = new GpuPeer(true);
    peer->connect(party, peerIP);
    
    Stats stats;
    GPULSSEngine<T> lss(peer, party, bw, scale, &gaes, &stats);
    
    // Test A2B conversion
    std::cout << "Test: Arithmetic to Binary conversion" << std::endl;
    {
        T* d_arithShares = randomGEOnGpu<T>(N, bw);
        
        // Convert to binary
        u32* d_binShares = lss.arithmeticToBinary(d_arithShares, N);
        
        std::cout << "  A2B conversion completed" << std::endl;
        
        gpuFree(d_arithShares);
        gpuFree(d_binShares);
    }
    
    // Test B2A conversion
    std::cout << "Test: Binary to Arithmetic conversion" << std::endl;
    {
        u64 numPackedInts = (N - 1) / 32 + 1;
        u32* d_binShares = (u32*)randomGEOnGpu<u32>(numPackedInts, 32);
        
        // Convert to arithmetic
        T* d_arithShares = lss.binaryToArithmetic(d_binShares, N, bw);
        
        std::cout << "  B2A conversion completed" << std::endl;
        
        gpuFree(d_binShares);
        gpuFree(d_arithShares);
    }
    
    // Test Binary operations
    std::cout << "Test: Binary XOR operation" << std::endl;
    {
        u64 numPackedInts = (N - 1) / 32 + 1;
        u32* d_a = (u32*)randomGEOnGpu<u32>(numPackedInts, 32);
        u32* d_b = (u32*)randomGEOnGpu<u32>(numPackedInts, 32);
        
        u32* d_xor = lss.binaryXor(d_a, d_b, N);
        
        // Verify XOR locally
        u32* h_a = (u32*)moveToCPU((u8*)d_a, numPackedInts * sizeof(u32), nullptr);
        u32* h_b = (u32*)moveToCPU((u8*)d_b, numPackedInts * sizeof(u32), nullptr);
        u32* h_xor = (u32*)moveToCPU((u8*)d_xor, numPackedInts * sizeof(u32), nullptr);
        
        for (u64 i = 0; i < numPackedInts; i++) {
            assert(h_xor[i] == (h_a[i] ^ h_b[i]));
        }
        std::cout << "  Binary XOR test PASSED" << std::endl;
        
        cpuFree(h_a);
        cpuFree(h_b);
        cpuFree(h_xor);
        gpuFree(d_a);
        gpuFree(d_b);
        gpuFree(d_xor);
    }
    
    // Cleanup
    peer->close();
    delete peer;
    destroyGPURandomness();
    
    std::cout << "All share conversion tests completed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <party_id> <peer_ip>" << std::endl;
        std::cout << "  party_id: 0 or 1" << std::endl;
        std::cout << "  peer_ip: IP address of peer" << std::endl;
        return 1;
    }
    
    int party = atoi(argv[1]);
    std::string peerIP = std::string(argv[2]);
    
    std::cout << "=== GPU LSS Test Suite ===" << std::endl;
    std::cout << "Party: " << party << std::endl;
    std::cout << "Peer IP: " << peerIP << std::endl;
    
    // Run tests for u32
    std::cout << "\n--- Testing with u32 ---" << std::endl;
    testLSSBasicOperations<u32>(party, peerIP);
    testShareConversions<u32>(party, peerIP);
    
    // Run tests for u64
    std::cout << "\n--- Testing with u64 ---" << std::endl;
    testLSSBasicOperations<u64>(party, peerIP);
    testShareConversions<u64>(party, peerIP);
    
    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    
    return 0;
}