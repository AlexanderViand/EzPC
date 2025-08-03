// Author: SCMP Benchmark for GPU-MPC
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

#include "mpc_benchmark.h"
#include "fss/gpu_scmp.h"
#include <fstream>
#include <chrono>

namespace {

// Named function for the actual MPC work
void runSCMPComparisonBenchmark(int party, const std::string& peer_ip, int threads) {
  // Default configuration
  const int bin = 32;
  const int bout = 1;
  const T element1 = 42;
  const T element2 = 35;
  
  printf("\n=== GPU SCMP Comparison Test ===\n");
  printf("Testing secure comparison: %lu >= %lu\n", element1, element2);
  printf("  Party: %d\n", party);
  printf("  Peer IP: %s\n", peer_ip.c_str());
  printf("  CPU threads: %d\n", threads);
  
  // Initialize AES context for crypto operations
  AESGlobalContext gaes;
  initAESContext(&gaes);
  
  // Test parameters
  int M = 1; // Number of comparisons
  u64 rin1 = 12345; // Random input mask 1
  u64 rin2 = 67890; // Random input mask 2  
  u64 rout = 1; // Random output mask (single bit for comparison)
  
  printf("\n--- SCMP Key Generation ---\n");
  GPUScmpKey key0, key1;
  
  auto start = std::chrono::high_resolution_clock::now();
  keyGenGPUSCMP(bin, bout, rin1, rin2, rout, 
                0, &key0, &key1, &gaes);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("SCMP key generation completed in %ld microseconds\n", elapsed.count());
  
  printf("\n--- SCMP Evaluation ---\n");
  GPUScmpKey *key = (party == 0) ? &key0 : &key1;
  
  // Allocate memory for results
  u64 h_results[M];
  
  start = std::chrono::high_resolution_clock::now();
  evalGPUSCMP(party, h_results, element1, element2, *key, M, &gaes);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("SCMP evaluation completed in %ld microseconds\n", elapsed.count());
  
  printf("\nSCMP Results for Party %d:\n", party);
  for (int i = 0; i < M; i++) {
    printf("  Share[%d]: %lu\n", i, h_results[i]);
  }
  
  // Expected result (for demonstration)
  bool expected = (element1 >= element2);
  printf("\nExpected comparison result: %lu >= %lu = %s\n", 
         element1, element2, expected ? "true" : "false");
  printf("Note: To get actual result, XOR shares from both parties\n");
  
  printf("\n--- Testing Less-Than Comparison ---\n");
  printf("Testing: %lu < %lu\n", element1, element2);
  
  // Test less-than function as well
  u64 h_lt_results[M];
  start = std::chrono::high_resolution_clock::now();
  evalGPULessThan(party, h_lt_results, element1, element2, *key, M, &gaes);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("Less-than evaluation completed in %ld microseconds\n", elapsed.count());
  printf("Less-than share for Party %d: %lu\n", party, h_lt_results[0]);
  
  bool expected_lt = (element1 < element2);
  printf("Expected less-than result: %lu < %lu = %s\n", 
         element1, element2, expected_lt ? "true" : "false");
  
  // Save results
  std::string outputDir = "./output/P" + std::to_string(party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream scmpFile(outputDir + "scmp_results.txt");
  scmpFile << "SCMP Results for Party " << party << std::endl;
  scmpFile << "Elements: " << element1 << " vs " << element2 << std::endl;
  scmpFile << "Key generation time: " << elapsed.count() << " us" << std::endl;
  scmpFile << "SCMP (>=) share: " << h_results[0] << std::endl;
  scmpFile << "Less-than (<) share: " << h_lt_results[0] << std::endl;
  scmpFile.close();
  
  // Cleanup
  freeGPUScmpKey(key0);
  freeGPUScmpKey(key1);
  
  printf("SCMP test completed successfully!\n");
}

// Lambda just handles registration
static bool registered = registerTask("scmp", [](int party, const std::string& peer_ip, int threads) {
    // Set up number of threads
    omp_set_num_threads(threads);
    
    // Create config for compatibility with existing code
    BenchmarkConfig config;
    config.party = party;
    config.peer_ip = peer_ip;
    config.cpu_threads = threads;
    config.bin = 32;
    config.bout = 1;
    config.element1 = 42;
    config.element2 = 35;
    config.task = "scmp";
    
    runSCMPComparisonBenchmark(party, peer_ip, threads);
});

} // anonymous namespace