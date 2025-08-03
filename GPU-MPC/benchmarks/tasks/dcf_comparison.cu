// Author: DCF Benchmark for GPU-MPC
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
#include "fss/dcf/gpu_dcf.cuh"
#include <fstream>
#include <sstream>
#include <chrono>

namespace {

// DCF Key Generation Backend (similar to SIGMAKeygen)
template <typename T> class TestKeygen {
public:
  int party;
  u8 *startPtr = NULL;
  u8 *keyBuf = NULL;
  size_t keySize = 0;
  AESGlobalContext g;

  TestKeygen(int party, u64 keyBufSz) : party(party) {
    initAESContext(&g);
    u8 *ptr1, *ptr2;
    getKeyBuf(&ptr1, &ptr2, keyBufSz);
    startPtr = keyBuf = (party == 0) ? ptr1 : ptr2;
  }

  void generateDCFKeys(int bin, int bout, int N, T *d_threshold) {
    dcf::gpuKeyGenDCF(&keyBuf, party, bin, bout, N, d_threshold, T(1), &g);
    keySize = keyBuf - startPtr;
    printf("Generated DCF keys for party %d, size: %lu bytes\n", party,
           keySize);
  }

  void close() {
    // Keys are in memory, ready for use
  }
};

// DCF MPC Backend (similar to SIGMA)
template <typename T> class DCFMPC {
public:
  int party;
  std::string peer_ip;
  u8 *startPtr = NULL;
  u8 *keyBuf = NULL;
  size_t keySize = 0;
  GpuPeer *peer = NULL;
  AESGlobalContext g;
  Stats s;
  LlamaTransformer<T> *llama;

  DCFMPC(int party, std::string ip, int numThreads)
      : party(party), peer_ip(ip) {
    initAESContext(&g);

    // Initialize LLAMA config for network communication
    LlamaConfig::bitlength = 32;    // Using 32-bit for our DCF test
    LlamaConfig::party = party + 2; // LLAMA uses party+2 convention
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;

    llama = new LlamaTransformer<T>();
    if (party == SERVER0)
      llama->initServer(ip, (char **)&keyBuf);
    else
      llama->initClient(ip, (char **)&keyBuf);

    peer = new GpuPeer(true);
    peer->peer = LlamaConfig::peer;

    omp_set_num_threads(numThreads);
  }

  void setKeys(u8 *keyPtr, size_t kSize) {
    startPtr = keyBuf = keyPtr;
    keySize = kSize;
  }

  void sync() {
    peer->sync();
    printf("Party %d synchronized with peer\n", party);
  }

  void runDCFComparison(int bin, int bout, int N, T *d_elements, T *h_elements,
                        T *h_threshold) {
    printf("Party %d running DCF comparison over network\n", party);

    auto start = std::chrono::high_resolution_clock::now();

    // Read DCF key from buffer
    auto k = dcf::readGPUDCFKey(&keyBuf);

    // Step 2: Perform DCF evaluation with network communication
    auto d_result = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(
        k, party, d_elements, &g, &s);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Time taken for P%d: %lu microseconds\n", party, elapsed.count());
    printf("Transfer time: %lu microseconds\n", s.transfer_time);

    // Phase 3: Reconstruct/Reveal Results
    printf("\n--- Phase 3: Result Reconstruction ---\n");
    auto reconstruct_start = std::chrono::high_resolution_clock::now();
    
    // Reconstruct the comparison results in place
    peer->reconstructInPlace((u32*)d_result, k.bout, (N + 31) / 32, &s);
    
    auto reconstruct_end = std::chrono::high_resolution_clock::now();
    auto reconstruct_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(reconstruct_end - reconstruct_start);
    printf("Reconstruction completed in %lu microseconds\n", reconstruct_elapsed.count());

    // Get results
    auto h_result = (u32 *)moveToCPU((u8 *)d_result, k.memSzOut, NULL);

    printf("\n=== DCF Comparison Results (Reconstructed) ===\n");
    for (int i = 0; i < N; i++) {
      auto result_bit = (h_result[i / 32] >> (i & 31)) & T(1);
      bool expected = (h_elements[i] < h_threshold[i]);
      
      printf("Test %d: %lu < %lu\n", i, h_elements[i], h_threshold[i]);
      printf("  Expected: %s\n", expected ? "true" : "false");
      printf("  Actual: %s %s\n", result_bit ? "true" : "false", 
             (result_bit == expected) ? "[PASS]" : "[FAIL] MISMATCH!");
    }

    gpuFree(d_result);
    free(h_result);
  }

  void close() {
    peer->close();
    printf("Party %d closed connection\n", party);
  }
};

// Named function for the actual MPC work
void runDCFComparisonBenchmark(int party, const std::string& peer_ip, int threads) {
  // Default configuration for DCF test
  const int bin = 32;   // 32-bit input elements
  const int bout = 1;   // 1-bit output (comparison result)
  const T element1 = 42;
  const T element2 = 35;
  
  printf("\n=== GPU-MPC DCF Benchmark ===\n");
  printf("Configuration:\n");
  printf("  Protocol: DCF (Distributed Comparison Function)\n");
  printf("  Comparing: %lu < %lu\n", element1, element2);
  printf("  Party: %d\n", party);
  printf("  Peer IP: %s\n", peer_ip.c_str());
  printf("  CPU threads: %d\n", threads);
  printf("  Input bitwidth: %d\n", bin);
  printf("  Output bitwidth: %d\n", bout);

  const int N = 2;                // We're comparing 2 elements
  const u64 keyBufSz = 1 * OneGB; // 1GB should be enough for simple DCF

  // Create test data
  auto d_elements = (T *)gpuMalloc(N * sizeof(T));
  T h_elements[N] = {element1, element2};
  checkCudaErrors(cudaMemcpy(d_elements, h_elements, N * sizeof(T),
                             cudaMemcpyHostToDevice));

  // Create threshold for comparison - we'll check if element1 < element2
  auto d_threshold = (T *)gpuMalloc(N * sizeof(T));
  T h_threshold[N] = {element2, element2};
  checkCudaErrors(cudaMemcpy(d_threshold, h_threshold, N * sizeof(T),
                             cudaMemcpyHostToDevice));

  // Phase 1: Key Generation (offline phase)
  printf("\n--- Phase 1: DCF Key Generation ---\n");
  auto keygen = new TestKeygen<T>(party, keyBufSz);

  auto start = std::chrono::high_resolution_clock::now();
  keygen->generateDCFKeys(bin, bout, N, d_threshold);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  keygen->close();
  
  auto keygen_time_us = elapsed.count();
  printf("Key generation completed in %.3f ms\n", keygen_time_us / 1000.0);
  printf("Key size: %.2f MB\n", keygen->keySize / (1024.0 * 1024.0));

  // Save key generation stats
  std::stringstream ss;
  ss << "Key generation time=" << keygen_time_us << " us" << std::endl;
  ss << "Key size=" << keygen->keySize << " bytes" << std::endl;

  std::string outputDir = "./output/P" + std::to_string(party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream statsFile(outputDir + "keygen.txt");
  statsFile << ss.rdbuf();
  statsFile.close();

  // Phase 2: Network MPC (online phase)
  printf("\n--- Phase 2: Network MPC Evaluation ---\n");
  auto dcfMPC = new DCFMPC<T>(party, peer_ip, threads);
  dcfMPC->setKeys(keygen->startPtr, keygen->keySize);

  // Synchronize with peer before starting MPC
  dcfMPC->sync();

  start = std::chrono::high_resolution_clock::now();
  dcfMPC->runDCFComparison(bin, bout, N, d_elements, h_elements,
                           h_threshold);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto mpc_time_us = elapsed.count();

  // Save MPC stats
  ss.str("");
  ss.clear();
  ss << "MPC time=" << mpc_time_us << " us" << std::endl;
  ss << "Communication time=" << dcfMPC->s.comm_time << " us" << std::endl;
  ss << "Transfer time=" << dcfMPC->s.transfer_time << " us" << std::endl;

  std::ofstream mpcStatsFile(outputDir + "mpc.txt");
  mpcStatsFile << ss.rdbuf();
  mpcStatsFile.close();

  dcfMPC->close();

  // Performance summary
  printf("\n=== Performance Summary ===\n");
  printf("Phase 1 (Key Generation):\n");
  printf("  Time: %.3f ms\n", keygen_time_us / 1000.0);
  printf("  Key size: %.2f MB\n", keygen->keySize / (1024.0 * 1024.0));
  
  printf("\nPhase 2 (Online MPC):\n");
  printf("  Computation: %.3f ms\n", mpc_time_us / 1000.0);
  
  // Fix: Show communication in MB/GB instead of time
  double comm_mb = dcfMPC->s.linear_comm_bytes / (1024.0 * 1024.0);
  if (comm_mb >= 1024.0) {
    printf("  Communication: %.2f GB\n", comm_mb / 1024.0);
  } else {
    printf("  Communication: %.2f MB\n", comm_mb);
  }
  
  printf("  Transfer: %.3f ms\n", dcfMPC->s.transfer_time / 1000.0);
  
  double total_ms = (keygen_time_us + mpc_time_us) / 1000.0;
  printf("\nTotal time: %.3f ms\n", total_ms);
  
  // Write JSON results using simplified config
  BenchmarkConfig config;
  config.party = party;
  config.peer_ip = peer_ip;
  config.cpu_threads = threads;
  config.bin = bin;
  config.bout = bout;
  config.element1 = element1;
  config.element2 = element2;
  config.task = "dcf";
  
  writeJSONResult("dcf_comparison", config, dcfMPC->s, true);
  
  printf("\nExpected result: %lu < %lu = %s\n", element1, element2,
         (element1 < element2) ? "true" : "false");

  printf("\nResults saved to %s\n", outputDir.c_str());

  // Cleanup
  gpuFree(d_elements);
  gpuFree(d_threshold);
  delete keygen;
  delete dcfMPC;
}

// Lambda just handles registration
static bool registered = registerTask("dcf", [](int party, const std::string& peer_ip, int threads) {
    runDCFComparisonBenchmark(party, peer_ip, threads);
});

} // anonymous namespace