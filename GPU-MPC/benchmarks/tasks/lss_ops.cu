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

#include "../mpc_benchmark.h"
#include "fss/gpu_lss.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.cuh"
#include "fss/gpu_aes_shm.cuh"
#include <chrono>
#include <iomanip>

namespace {

// LSS Operations Benchmark Task
void runLSSBenchmark(int party, const std::string& peer_ip, int threads) {
    // Initialize
    AESGlobalContext gaes;
    initAESContext(&gaes);
    initGPUMemPool();
    initGPURandomness();
    
    auto peer = new GpuPeer(true);
    peer->connect(party, peer_ip);
    
    Stats stats;
    const int bw = 64;
    const int scale = 12;
    auto lss = new GPULSSEngine<u64>(peer, party, bw, scale, &gaes, &stats);
    
    // Benchmark parameters
    std::vector<u64> sizes = {1024, 10240, 102400};
    const int warmup_runs = 2;
    const int benchmark_runs = 10;
    
    std::cout << "\n=== LSS Operations Benchmark ===" << std::endl;
    std::cout << "Party: " << party << ", Peer: " << peer_ip << std::endl;
    std::cout << "Bit width: " << bw << ", Scale: " << scale << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::setw(15) << "Size" 
              << std::setw(15) << "Add (ms)"
              << std::setw(15) << "ScalarMul (ms)"
              << std::setw(15) << "XOR (ms)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (u64 N : sizes) {
        // Allocate test data
        u64* d_a = randomGEOnGpu<u64>(N, bw);
        u64* d_b = randomGEOnGpu<u64>(N, bw);
        u64 scalar = 7;
        
        // Warmup
        for (int i = 0; i < warmup_runs; i++) {
            u64* tmp = lss->add(d_a, d_b, N);
            gpuFree(tmp);
        }
        
        // Benchmark Addition
        double add_time = 0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < benchmark_runs; i++) {
                u64* d_sum = lss->add(d_a, d_b, N);
                gpuFree(d_sum);
            }
            auto end = std::chrono::high_resolution_clock::now();
            add_time = std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
        }
        
        // Benchmark Scalar Multiplication
        double scalar_time = 0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < benchmark_runs; i++) {
                u64* d_scaled = lss->scalarMultiply(d_a, scalar, N);
                gpuFree(d_scaled);
            }
            auto end = std::chrono::high_resolution_clock::now();
            scalar_time = std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
        }
        
        // Benchmark Binary XOR
        double xor_time = 0;
        {
            u64 numPackedInts = (N - 1) / 32 + 1;
            u32* d_bin_a = (u32*)randomGEOnGpu<u32>(numPackedInts, 32);
            u32* d_bin_b = (u32*)randomGEOnGpu<u32>(numPackedInts, 32);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < benchmark_runs; i++) {
                u32* d_xor = lss->binaryXor(d_bin_a, d_bin_b, N);
                gpuFree(d_xor);
            }
            auto end = std::chrono::high_resolution_clock::now();
            xor_time = std::chrono::duration<double, std::milli>(end - start).count() / benchmark_runs;
            
            gpuFree(d_bin_a);
            gpuFree(d_bin_b);
        }
        
        // Print results for this size
        std::cout << std::setw(15) << N
                  << std::setw(15) << std::fixed << std::setprecision(3) << add_time
                  << std::setw(15) << scalar_time
                  << std::setw(15) << xor_time
                  << std::endl;
        
        // Cleanup
        gpuFree(d_a);
        gpuFree(d_b);
    }
    
    // Print communication stats
    std::cout << "\n=== Communication Statistics ===" << std::endl;
    std::cout << "Bytes sent: " << peer->bytesSent() << std::endl;
    std::cout << "Bytes received: " << peer->bytesReceived() << std::endl;
    std::cout << "Total communication: " << (peer->bytesSent() + peer->bytesReceived()) << std::endl;
    
    // Cleanup
    delete lss;
    peer->close();
    delete peer;
    destroyGPURandomness();
}

// Register the task
bool lss_registered = registerTask("lss", runLSSBenchmark);

} // namespace