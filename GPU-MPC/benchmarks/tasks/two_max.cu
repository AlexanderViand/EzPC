// Author: Two-Iteration Maximum Benchmark for GPU-MPC
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
#include "fss/gpu_compare_aggregate.cuh"
#include "fss/gpu_scmp.h"
#include <fstream>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>

namespace {

// Named function for the actual MPC work
void runTwoIterationMaximumBenchmark(int party, const std::string& peer_ip, int threads) {
  printf("\n=== Testing Two-Iteration Maximum Algorithm ===\n");
  printf("  Party: %d\n", party);
  printf("  Peer IP: %s\n", peer_ip.c_str());
  printf("  CPU threads: %d\n", threads);
  
  // Set thread count
  omp_set_num_threads(threads);
  
  // Initialize AES context
  AESGlobalContext gaes;
  initAESContext(&gaes);
  
  // Test parameters
  const int TEST_SIZE = 50;  // Number of elements to test
  const int MAX_VALUE = 1000;  // Maximum value for random elements
  const int bin = 32;  // 32-bit elements
  const int bout = 1;  // 1-bit comparison output
  
  // Generate random test data
  u64 *h_elements = new u64[TEST_SIZE];
  srand(time(NULL) + party);  // Different seed for each party
  
  u64 expectedMax = 0;
  printf("Test data: ");
  for (int i = 0; i < TEST_SIZE; i++) {
    h_elements[i] = rand() % MAX_VALUE;
    if (i < 10) printf("%lu ", h_elements[i]);  // Print first 10 elements
    if (h_elements[i] > expectedMax) {
      expectedMax = h_elements[i];
    }
  }
  printf("...\n");
  printf("Expected maximum (cleartext): %lu\n", expectedMax);
  
  // Copy data to GPU
  u64 *d_elements = (u64*)gpuMalloc(TEST_SIZE * sizeof(u64));
  checkCudaErrors(cudaMemcpy(d_elements, h_elements, TEST_SIZE * sizeof(u64), 
                             cudaMemcpyHostToDevice));
  
  // Calculate number of SCMP keys needed
  // For two-iteration maximum, we need keys for:
  // 1. All comparisons within partitions (iteration 1)
  // 2. All comparisons in the complete graph of partition maxima (iteration 2)
  int t = std::max(1, (int)(std::pow(TEST_SIZE, 2.0/3.0) / std::pow(2, 1.0/3.0)));
  if (t > TEST_SIZE) t = TEST_SIZE;
  
  // Estimate number of keys needed (conservative upper bound)
  int numKeys = TEST_SIZE * (TEST_SIZE - 1) / 2;  // Upper bound: complete graph
  
  printf("\nGenerating %d SCMP keys for two-iteration maximum...\n", numKeys);
  
  // Generate SCMP keys
  GPUScmpKey *keys = new GPUScmpKey[numKeys];
  for (int i = 0; i < numKeys; i++) {
    // Use random masks for demonstration
    u64 rin1 = rand() % MAX_VALUE;
    u64 rin2 = rand() % MAX_VALUE;
    u64 rout = rand() & 1;
    
    GPUScmpKey tempKey0, tempKey1;
    keyGenGPUSCMP(bin, bout, rin1, rin2, rout, 0, &tempKey0, &tempKey1, &gaes);
    
    // Use the appropriate key for this party
    keys[i] = (party == 0) ? tempKey0 : tempKey1;
    
    // Free the unused key
    if (party == 0) {
      freeGPUScmpKey(tempKey1);
    } else {
      freeGPUScmpKey(tempKey0);
    }
  }
  
  printf("Keys generated successfully\n");
  
  // Run the two-iteration maximum algorithm
  auto start = std::chrono::high_resolution_clock::now();
  runTwoIterationMaximum(party, d_elements, TEST_SIZE, keys, numKeys, &gaes);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("\nTwo-iteration maximum completed in %ld microseconds\n", elapsed.count());
  
  // Save results to file
  std::string outputDir = "./output/P" + std::to_string(party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream maxFile(outputDir + "two_iteration_max_results.txt");
  maxFile << "Two-Iteration Maximum Results for Party " << party << std::endl;
  maxFile << "Number of elements: " << TEST_SIZE << std::endl;
  maxFile << "Execution time: " << elapsed.count() << " us" << std::endl;
  maxFile << "Expected maximum (cleartext): " << expectedMax << std::endl;
  maxFile.close();
  
  // Cleanup
  for (int i = 0; i < numKeys; i++) {
    freeGPUScmpKey(keys[i]);
  }
  delete[] keys;
  delete[] h_elements;
  gpuFree(d_elements);
  
  printf("Two-iteration maximum test completed!\n");
}

// Lambda just handles registration
static bool registered = registerTask("twomax", [](int party, const std::string& peer_ip, int threads) {
    runTwoIterationMaximumBenchmark(party, peer_ip, threads);
});

} // anonymous namespace