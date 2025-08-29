// Author: SCMP MPC Benchmark for GPU-MPC
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
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <random>

namespace {

// SCMP Key Generation Backend
template <typename T> class SCMPKeygen {
public:
  int party;
  GPUScmpKey key0_gte, key1_gte;  // Keys for >= comparison
  GPUScmpKey key0_lt, key1_lt;    // Keys for < comparison
  AESGlobalContext g;
  
  SCMPKeygen(int party) : party(party) {
    initAESContext(&g);
  }
  
  void generateSCMPKeys(int bin, int bout) {
    // Generate random masks for >= comparison
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<u64> dist(0, (1ULL << bin) - 1);
    
    u64 rin1_gte = dist(gen);
    u64 rin2_gte = dist(gen);
    u64 rout_gte = dist(gen) & ((1ULL << bout) - 1);
    
    // Generate keys for >= comparison
    keyGenGPUSCMP(bin, bout, rin1_gte, rin2_gte, rout_gte, 0, &key0_gte, &key1_gte, &g);
    printf("Generated SCMP keys for >= comparison\n");
    
    // Generate different random masks for < comparison
    u64 rin1_lt = dist(gen);
    u64 rin2_lt = dist(gen);
    u64 rout_lt = dist(gen) & ((1ULL << bout) - 1);
    
    // Generate keys for < comparison
    keyGenGPUSCMP(bin, bout, rin1_lt, rin2_lt, rout_lt, 0, &key0_lt, &key1_lt, &g);
    printf("Generated SCMP keys for < comparison\n");
  }
  
  GPUScmpKey* getGTEKey() {
    return (party == 0) ? &key0_gte : &key1_gte;
  }
  
  GPUScmpKey* getLTKey() {
    return (party == 0) ? &key0_lt : &key1_lt;
  }
  
  void close() {
    freeGPUScmpKey(key0_gte);
    freeGPUScmpKey(key1_gte);
    freeGPUScmpKey(key0_lt);
    freeGPUScmpKey(key1_lt);
  }
};

// SCMP MPC Backend with proper network communication
template <typename T> class SCMPMPC {
public:
  int party;
  std::string peer_ip;
  GpuPeer *peer = NULL;
  AESGlobalContext g;
  Stats s;
  LlamaTransformer<T> *llama;
  
  SCMPMPC(int party, std::string ip, int numThreads) 
      : party(party), peer_ip(ip) {
    initAESContext(&g);
    
    // Initialize LLAMA config for network communication
    LlamaConfig::bitlength = 32;
    LlamaConfig::party = party + 2;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    
    llama = new LlamaTransformer<T>();
    u8* keyBuf = nullptr;
    if (party == SERVER0)
      llama->initServer(ip, (char **)&keyBuf);
    else
      llama->initClient(ip, (char **)&keyBuf);
    
    peer = new GpuPeer(true);
    peer->peer = LlamaConfig::peer;
    
    omp_set_num_threads(numThreads);
  }
  
  void sync() {
    peer->sync();
    printf("Party %d synchronized with peer\n", party);
  }
  
  void runSCMPComparison(int bin, int bout, T x_share, T y_share, 
                         GPUScmpKey *key, bool lessThan = false) {
    printf("Party %d running SCMP comparison over network\n", party);
    printf("Using shares: x_share=%lu, y_share=%lu\n", x_share, y_share);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate memory for results
    u64 h_results[1];
    
    // Phase 2: Perform SCMP evaluation with SHARES
    if (lessThan) {
      evalGPULessThan(party, h_results, x_share, y_share, *key, 1, &g);
    } else {
      evalGPUSCMP(party, h_results, x_share, y_share, *key, 1, &g);
    }
    
    auto eval_end = std::chrono::high_resolution_clock::now();
    auto eval_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        eval_end - start);
    
    printf("SCMP evaluation completed in %ld microseconds\n", 
           eval_elapsed.count());
    printf("Party %d share: %lu\n", party, h_results[0]);
    
    // Phase 3: Reconstruct/Reveal Results
    printf("\n--- Phase 3: Result Reconstruction ---\n");
    auto reconstruct_start = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory for reconstruction
    u64 *d_results = (u64*)gpuMalloc(sizeof(u64));
    checkCudaErrors(cudaMemcpy(d_results, h_results, sizeof(u64), 
                               cudaMemcpyHostToDevice));
    
    // Reconstruct the comparison result using XOR
    peer->reconstructInPlace(d_results, bout, 1, &s);
    
    // Get reconstructed result
    checkCudaErrors(cudaMemcpy(h_results, d_results, sizeof(u64), 
                               cudaMemcpyDeviceToHost));
    
    auto reconstruct_end = std::chrono::high_resolution_clock::now();
    auto reconstruct_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        reconstruct_end - reconstruct_start);
    
    printf("Reconstruction completed in %ld microseconds\n", 
           reconstruct_elapsed.count());
    
    // Display final result
    bool result = h_results[0] & 1;
    
    printf("\n=== SCMP Comparison Results (Reconstructed) ===\n");
    printf("Result: %s\n", result ? "true" : "false");
    printf("Expected (42 %s 35): %s\n", 
           lessThan ? "<" : ">=", 
           lessThan ? "false" : "true");
    
    if ((lessThan && !result) || (!lessThan && result)) {
      printf("Status: [PASS]\n");
    } else {
      printf("Status: [FAIL] MISMATCH!\n");
    }
    
    // Cleanup
    gpuFree(d_results);
    
    // Save timing stats
    s.compute_time = eval_elapsed.count();
    s.comm_time = reconstruct_elapsed.count();
  }
  
  void close() {
    peer->close();
    printf("Party %d closed connection\n", party);
  }
};

// Helper function to generate secret shares
void generateShares(T value, T& share0, T& share1) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<T> dist(0, UINT64_MAX);
  
  share0 = dist(gen);
  share1 = value - share0;  // Additive secret sharing
}

// Main benchmark function
void runSCMPComparisonBenchmark(int party, const std::string& peer_ip, int threads) {
  // Configuration
  const int bin = 32;
  const int bout = 1;
  const T element1 = 42;  // First value to compare
  const T element2 = 35;  // Second value to compare
  
  printf("\n=== GPU-MPC SCMP Benchmark (with MPC) ===\n");
  printf("Configuration:\n");
  printf("  Protocol: SCMP (Secure Comparison)\n");
  printf("  Comparing: %lu >= %lu and %lu < %lu\n", 
         element1, element2, element1, element2);
  printf("  Party: %d\n", party);
  printf("  Peer IP: %s\n", peer_ip.c_str());
  printf("  CPU threads: %d\n", threads);
  
  // For this demo, hardcode the shares
  // In a real system, these would be distributed during setup
  T x_share, y_share;
  if (party == 0) {
    x_share = 25;  // Party 0's share of 42
    y_share = 20;  // Party 0's share of 35
  } else {
    x_share = 17;  // Party 1's share of 42 (25 + 17 = 42)
    y_share = 15;  // Party 1's share of 35 (20 + 15 = 35)
  }
  
  printf("\nParty %d shares: x=%lu, y=%lu\n", party, x_share, y_share);
  printf("Reconstructed values: x=%lu+%lu=%d, y=%lu+%lu=%d\n", 
         25UL, 17UL, 42, 20UL, 15UL, 35);
  
  // Phase 1: Key Generation (offline phase)
  printf("\n--- Phase 1: SCMP Key Generation ---\n");
  auto keygen = new SCMPKeygen<T>(party);
  
  auto start = std::chrono::high_resolution_clock::now();
  keygen->generateSCMPKeys(bin, bout);
  auto end = std::chrono::high_resolution_clock::now();
  auto keygen_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      end - start);
  
  printf("Key generation completed in %.3f ms\n", 
         keygen_elapsed.count() / 1000.0);
  
  // Phase 2 & 3: Network MPC and Reconstruction
  printf("\n--- Phase 2: Network MPC Evaluation ---\n");
  auto scmpMPC = new SCMPMPC<T>(party, peer_ip, threads);
  
  // Synchronize with peer
  scmpMPC->sync();
  
  // Test >= comparison with proper keys
  printf("\n*** Testing >= comparison ***\n");
  scmpMPC->runSCMPComparison(bin, bout, x_share, y_share, 
                             keygen->getGTEKey(), false);
  
  // Test < comparison with different keys
  printf("\n*** Testing < comparison ***\n");
  scmpMPC->runSCMPComparison(bin, bout, x_share, y_share, 
                             keygen->getLTKey(), true);
  
  // Performance summary
  printf("\n=== Performance Summary ===\n");
  printf("Phase 1 (Key Generation): %.3f ms\n", 
         keygen_elapsed.count() / 1000.0);
  printf("Phase 2 (Evaluation): %.3f ms\n", 
         scmpMPC->s.compute_time / 1000.0);
  printf("Phase 3 (Reconstruction): %.3f ms\n", 
         scmpMPC->s.comm_time / 1000.0);
  u64 total_time = keygen_elapsed.count() + scmpMPC->s.compute_time + scmpMPC->s.comm_time;
  printf("Total time: %.3f ms\n", total_time / 1000.0);
  
  // Save results
  std::string outputDir = "./output/P" + std::to_string(party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  
  std::ofstream statsFile(outputDir + "scmp_mpc.txt");
  statsFile << "SCMP MPC Results for Party " << party << std::endl;
  statsFile << "Values: " << element1 << " and " << element2 << std::endl;
  statsFile << "Shares: x=" << x_share << ", y=" << y_share << std::endl;
  statsFile << "Key generation time: " << keygen_elapsed.count() << " us" << std::endl;
  statsFile << "Evaluation time: " << scmpMPC->s.compute_time << " us" << std::endl;
  statsFile << "Reconstruction time: " << scmpMPC->s.comm_time << " us" << std::endl;
  statsFile << "Total time: " << total_time << " us" << std::endl;
  statsFile.close();
  
  printf("\nResults saved to %s\n", outputDir.c_str());
  
  // Cleanup
  keygen->close();
  scmpMPC->close();
  delete keygen;
  delete scmpMPC;
}

// Register the task
static bool registered = registerTask("scmp", [](int party, const std::string& peer_ip, int threads) {
    runSCMPComparisonBenchmark(party, peer_ip, threads);
});

} // anonymous namespace