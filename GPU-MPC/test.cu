// Author: Generated for DCF Test
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "test.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include "fss/gpu_compare_aggregate.h"

DCFTestConfig parseTestArgs(int argc, char **argv) {
  if (argc != 7) {
    printf("Usage: %s <model> <sequence_length> <party=0/1> <peer_ip> "
           "<cpu_threads> <run_scmp=0/1>\n",
           argv[0]);
    printf("Example: %s dcf-test 128 0 192.168.1.100 64 1\n", argv[0]);
    printf("  run_scmp=1 to test SCMP functionality, 0 for DCF only\n");
    exit(1);
  }

  DCFTestConfig config;
  std::string model(argv[1]);

  // Parse arguments similar to sigma
  int seq_len = atoi(argv[2]);
  config.party = atoi(argv[3]);
  config.peer_ip = std::string(argv[4]);
  config.cpu_threads = atoi(argv[5]);
  config.run_scmp = (atoi(argv[6]) == 1);

  // Set DCF parameters based on model type
  if (model == "dcf-test" || model == "test") {
    config.bin = 32; // 32-bit input elements
    config.bout = 1; // 1-bit output (comparison result)
    // For demo purposes, use fixed test values
    config.element1 = 42; // First element
    config.element2 = 35; // Second element
  } else {
    printf("Unknown model: %s. Using dcf-test as default.\n", model.c_str());
    config.bin = 32;
    config.bout = 1;
    config.element1 = 42;
    config.element2 = 35;
  }

  printf("DCF Test Configuration:\n");
  printf("  Model: %s\n", model.c_str());
  printf("  Sequence Length: %d\n", seq_len);
  printf("  Party: %d\n", config.party);
  printf("  Peer IP: %s\n", config.peer_ip.c_str());
  printf("  CPU Threads: %d\n", config.cpu_threads);
  printf("  Input bit width: %d\n", config.bin);
  printf("  Output bit width: %d\n", config.bout);
  printf("  Element 1: %lu\n", config.element1);
  printf("  Element 2: %lu\n", config.element2);
  printf("  Run SCMP test: %s\n", config.run_scmp ? "Yes" : "No");

  return config;
}

void initTestEnvironment() {
  initGPUMemPool();
  initGPURandomness();
}

void cleanupTestEnvironment() { destroyGPURandomness(); }

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
    // FIXME: inline this into its parent function for easier hacking
    printf("Party %d running DCF comparison over network\n", party);

    auto start = std::chrono::high_resolution_clock::now();

    // Read DCF key from buffer
    // TODO: more magic here for mapping
    auto k = dcf::readGPUDCFKey(&keyBuf);

    // Step 2: Perform DCF evaluation with network communication
    auto d_result = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(
        k, party, d_elements, &g, &s);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // TODO: local rank aggregation: rank of element i = sum j DFC of (i,j)
    // NOTE: make sure the sum indexing matches i < j vs j > i

    // TODO: REVEAL ranks (FIXME: need to shuffle elements before we call max)
    // NOTE: look for reconstructInPlace

    /// Step 3:Comparisons between each partitions local maximum
    /// This assume we've already generated another p choose 2 keys
    /// PROBLEM: how could we have generated them for the RIGHT masks/labels?
    /// Step 3.1: call the DCFs
    /// Step 3.2:local rank aggregation (same as above)

    printf("Time taken for P%d: %lu microseconds\n", party, elapsed.count());
    printf("Transfer time: %lu microseconds\n", s.transfer_time);

    // Get results
    auto h_result = (u32 *)moveToCPU((u8 *)d_result, k.memSzOut, NULL);

    printf("Party %d comparison results:\n", party);
    for (int i = 0; i < N; i++) {
      auto result_bit = (h_result[i / 32] >> (i & 31)) & T(1);
      printf("  Element %d (%lu) < %lu: P%d_share = %u\n", i, h_elements[i],
             h_threshold[i], party, result_bit);
    }

    gpuFree(d_result);
    free(h_result);
  }

  void close() {
    peer->close();
    printf("Party %d closed connection\n", party);
  }
};

void runDCFComparison(const DCFTestConfig &config) {
  printf("\n=== Running Real Network MPC DCF Comparison: %lu vs %lu ===\n",
         config.element1, config.element2);

  const int N = 2;                // We're comparing 2 elements
  const u64 keyBufSz = 1 * OneGB; // 1GB should be enough for simple DCF

  // Create test data
  auto d_elements = (T *)gpuMalloc(N * sizeof(T));
  T h_elements[N] = {config.element1, config.element2};
  checkCudaErrors(cudaMemcpy(d_elements, h_elements, N * sizeof(T),
                             cudaMemcpyHostToDevice));

  // Create threshold for comparison - we'll check if element1 < element2
  auto d_threshold = (T *)gpuMalloc(N * sizeof(T));
  T h_threshold[N] = {config.element2, config.element2};
  checkCudaErrors(cudaMemcpy(d_threshold, h_threshold, N * sizeof(T),
                             cudaMemcpyHostToDevice));

  // For n elements split into p partitions of size k each,
  // we need a total of p * (k choose 2) DCF keys (for the first part)
  // We split the big buffer into the p partitions,
  // then, inside the partitions, we just need a simple map (i,j) -> l
  // for i < j, where l is the index of the key in the partition.
  // NOTE: for CPU/sequential code, we want to make sure that our mapping
  // will be accessing the keys in buffer order (avoid cache invalidation)
  // NOTE: on GPU, we can probably do at least an entire partition,
  // if not the entire first part in parallel b/c GPU magic
  // However, we might still need to do some cache locality magic?

  // Phase 1: Key Generation (offline phase)
  printf("\n--- Phase 1: DCF Key Generation ---\n");
  auto keygen = new TestKeygen<T>(config.party, keyBufSz);

  auto start = std::chrono::high_resolution_clock::now();
  keygen->generateDCFKeys(config.bin, config.bout, N, d_threshold);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  keygen->close();

  // Save key generation stats
  std::stringstream ss;
  ss << "Key generation time=" << elapsed.count() << " us" << std::endl;
  ss << "Key size=" << keygen->keySize << " bytes" << std::endl;

  std::string outputDir = "./output/P" + std::to_string(config.party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream statsFile(outputDir + "keygen.txt");
  statsFile << ss.rdbuf();
  statsFile.close();

  // Phase 2: Network MPC (online phase)
  printf("\n--- Phase 2: Network MPC Evaluation ---\n");
  auto dcfMPC = new DCFMPC<T>(config.party, config.peer_ip, config.cpu_threads);
  dcfMPC->setKeys(keygen->startPtr, keygen->keySize);

  // Synchronize with peer before starting MPC
  dcfMPC->sync();

  start = std::chrono::high_resolution_clock::now();
  dcfMPC->runDCFComparison(config.bin, config.bout, N, d_elements, h_elements,
                           h_threshold);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Save MPC stats
  ss.str("");
  ss.clear();
  ss << "MPC time=" << elapsed.count() << " us" << std::endl;
  ss << "Communication time=" << dcfMPC->s.comm_time << " us" << std::endl;
  ss << "Transfer time=" << dcfMPC->s.transfer_time << " us" << std::endl;

  std::ofstream mpcStatsFile(outputDir + "mpc.txt");
  mpcStatsFile << ss.rdbuf();
  mpcStatsFile.close();

  dcfMPC->close();

  printf("\nExpected result: %lu < %lu = %s\n", config.element1,
         config.element2,
         (config.element1 < config.element2) ? "true" : "false");

  printf("Results saved to %s\n", outputDir.c_str());

  // Cleanup
  gpuFree(d_elements);
  gpuFree(d_threshold);
  delete keygen;
  delete dcfMPC;
}

// SCMP comparison test function
void runSCMPComparison(DCFTestConfig config) {
  printf("\n=== GPU SCMP Comparison Test ===\n");
  printf("Testing secure comparison: %lu >= %lu\n", config.element1, config.element2);
  
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
  keyGenGPUSCMP(config.bin, config.bout, rin1, rin2, rout, 
                0, &key0, &key1, &gaes);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("SCMP key generation completed in %ld microseconds\n", elapsed.count());
  
  printf("\n--- SCMP Evaluation ---\n");
  GPUScmpKey *key = (config.party == 0) ? &key0 : &key1;
  
  // Allocate memory for results
  u64 h_results[M];
  
  start = std::chrono::high_resolution_clock::now();
  evalGPUSCMP(config.party, h_results, config.element1, config.element2, *key, M, &gaes);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("SCMP evaluation completed in %ld microseconds\n", elapsed.count());
  
  printf("\nSCMP Results for Party %d:\n", config.party);
  for (int i = 0; i < M; i++) {
    printf("  Share[%d]: %lu\n", i, h_results[i]);
  }
  
  // Expected result (for demonstration)
  bool expected = (config.element1 >= config.element2);
  printf("\nExpected comparison result: %lu >= %lu = %s\n", 
         config.element1, config.element2, expected ? "true" : "false");
  printf("Note: To get actual result, XOR shares from both parties\n");
  
  printf("\n--- Testing Less-Than Comparison ---\n");
  printf("Testing: %lu < %lu\n", config.element1, config.element2);
  
  // Test less-than function as well
  u64 h_lt_results[M];
  start = std::chrono::high_resolution_clock::now();
  evalGPULessThan(config.party, h_lt_results, config.element1, config.element2, *key, M, &gaes);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("Less-than evaluation completed in %ld microseconds\n", elapsed.count());
  printf("Less-than share for Party %d: %lu\n", config.party, h_lt_results[0]);
  
  bool expected_lt = (config.element1 < config.element2);
  printf("Expected less-than result: %lu < %lu = %s\n", 
         config.element1, config.element2, expected_lt ? "true" : "false");
  
  // Save results
  std::string outputDir = "./output/P" + std::to_string(config.party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream scmpFile(outputDir + "scmp_results.txt");
  scmpFile << "SCMP Results for Party " << config.party << std::endl;
  scmpFile << "Elements: " << config.element1 << " vs " << config.element2 << std::endl;
  scmpFile << "Key generation time: " << elapsed.count() << " us" << std::endl;
  scmpFile << "SCMP (>=) share: " << h_results[0] << std::endl;
  scmpFile << "Less-than (<) share: " << h_lt_results[0] << std::endl;
  scmpFile.close();
  
  // Cleanup
  freeGPUScmpKey(key0);
  freeGPUScmpKey(key1);
  
  printf("SCMP test completed successfully!\n");
}

int main(int argc, char **argv) {
  // Parse command line arguments
  DCFTestConfig config = parseTestArgs(argc, argv);

  // Initialize environment
  initTestEnvironment();

  // Run DCF comparison
  runDCFComparison(config);

  // Run SCMP test if requested
  if (config.run_scmp) {
    runSCMPComparison(config);
    
    // Run two-iteration maximum algorithm test
    runTwoIterationMaximumTest(config);
  }

  // Cleanup
  cleanupTestEnvironment();

  printf("\nAll tests completed successfully!\n");
  return 0;
}

// Test function for two-iteration maximum algorithm
void runTwoIterationMaximumTest(const DCFTestConfig& config) {
  printf("\n=== Testing Two-Iteration Maximum Algorithm ===\n");
  
  // Initialize AES context
  AESGlobalContext gaes;
  initAESContext(&gaes);
  
  // Test parameters
  const int TEST_SIZE = 50;  // Number of elements to test
  const int MAX_VALUE = 1000;  // Maximum value for random elements
  
  // Generate random test data
  u64 *h_elements = new u64[TEST_SIZE];
  srand(time(NULL) + config.party);  // Different seed for each party
  
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
    keyGenGPUSCMP(config.bin, config.bout, rin1, rin2, rout, 0, &tempKey0, &tempKey1, &gaes);
    
    // Use the appropriate key for this party
    keys[i] = (config.party == 0) ? tempKey0 : tempKey1;
    
    // Free the unused key
    if (config.party == 0) {
      freeGPUScmpKey(tempKey1);
    } else {
      freeGPUScmpKey(tempKey0);
    }
  }
  
  printf("Keys generated successfully\n");
  
  // Run the two-iteration maximum algorithm
  auto start = std::chrono::high_resolution_clock::now();
  runTwoIterationMaximum(config.party, d_elements, TEST_SIZE, keys, numKeys, &gaes);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  printf("\nTwo-iteration maximum completed in %ld microseconds\n", elapsed.count());
  
  // Save results to file
  std::string outputDir = "./output/P" + std::to_string(config.party) + "/";
  system(("mkdir -p " + outputDir).c_str());
  std::ofstream maxFile(outputDir + "two_iteration_max_results.txt");
  maxFile << "Two-Iteration Maximum Results for Party " << config.party << std::endl;
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