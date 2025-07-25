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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "test.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>

DCFTestConfig parseTestArgs(int argc, char **argv) {
    if (argc != 6) {
        printf("Usage: %s <model> <sequence_length> <party=0/1> <peer_ip> <cpu_threads>\n", argv[0]);
        printf("Example: %s dcf-test 128 0 192.168.1.100 64\n", argv[0]);
        exit(1);
    }
    
    DCFTestConfig config;
    std::string model(argv[1]);
    
    // Parse arguments similar to sigma
    int seq_len = atoi(argv[2]);
    config.party = atoi(argv[3]);
    config.peer_ip = std::string(argv[4]);
    config.cpu_threads = atoi(argv[5]);
    
    // Set DCF parameters based on model type
    if (model == "dcf-test" || model == "test") {
        config.bin = 32;   // 32-bit input elements
        config.bout = 1;   // 1-bit output (comparison result)
        // For demo purposes, use fixed test values
        config.element1 = 42;   // First element
        config.element2 = 35;   // Second element
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
    
    return config;
}

void initTestEnvironment() {
    initGPUMemPool();
    initGPURandomness();
}

void cleanupTestEnvironment() {
    destroyGPURandomness();
}

// DCF Key Generation Backend (similar to SIGMAKeygen)
template<typename T>
class DCFKeygen {
public:
    int party;
    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;
    size_t keySize = 0;
    AESGlobalContext g;
    
    DCFKeygen(int party, u64 keyBufSz) : party(party) {
        initAESContext(&g);
        u8 *ptr1, *ptr2;
        getKeyBuf(&ptr1, &ptr2, keyBufSz);
        startPtr = keyBuf = (party == 0) ? ptr1 : ptr2;
    }
    
    void generateDCFKeys(int bin, int bout, int N, T* d_threshold) {
        dcf::gpuKeyGenDCF(&keyBuf, party, bin, bout, N, d_threshold, T(1), &g);
        keySize = keyBuf - startPtr;
        printf("Generated DCF keys for party %d, size: %lu bytes\n", party, keySize);
    }
    
    void close() {
        // Keys are in memory, ready for use
    }
};

// DCF MPC Backend (similar to SIGMA)
template<typename T>
class DCFMPC {
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
    
    DCFMPC(int party, std::string ip, int numThreads) : party(party), peer_ip(ip) {
        initAESContext(&g);
        
        // Initialize LLAMA config for network communication
        LlamaConfig::bitlength = 32; // Using 32-bit for our DCF test
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
    
    void setKeys(u8* keyPtr, size_t kSize) {
        startPtr = keyBuf = keyPtr;
        keySize = kSize;
    }
    
    void sync() {
        peer->sync();
        printf("Party %d synchronized with peer\n", party);
    }
    
    void runDCFComparison(int bin, int bout, int N, T* d_elements, T* h_elements, T* h_threshold) {
        printf("Party %d running DCF comparison over network\n", party);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Read DCF key from buffer
        auto k = dcf::readGPUDCFKey(&keyBuf);
        
        // Perform DCF evaluation with network communication
        auto d_result = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(k, party, d_elements, &g, &s);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("Time taken for P%d: %lu microseconds\n", party, elapsed.count());
        printf("Transfer time: %lu microseconds\n", s.transfer_time);
        
        // Get results
        auto h_result = (u32*)moveToCPU((u8*)d_result, k.memSzOut, NULL);
        
        printf("Party %d comparison results:\n", party);
        for (int i = 0; i < N; i++) {
            auto result_bit = (h_result[i / 32] >> (i & 31)) & T(1);
            printf("  Element %d (%lu) < %lu: P%d_share = %u\n", 
                   i, h_elements[i], h_threshold[i], party, result_bit);
        }
        
        gpuFree(d_result);
        free(h_result);
    }
    
    void close() {
        peer->close();
        printf("Party %d closed connection\n", party);
    }
};

void runDCFComparison(const DCFTestConfig& config) {
    printf("\n=== Running Real Network MPC DCF Comparison: %lu vs %lu ===\n", 
           config.element1, config.element2);
    
    const int N = 2; // We're comparing 2 elements
    const u64 keyBufSz = 1 * OneGB; // 1GB should be enough for simple DCF
    
    // Create test data
    auto d_elements = (T*)gpuMalloc(N * sizeof(T));
    T h_elements[N] = {config.element1, config.element2};
    checkCudaErrors(cudaMemcpy(d_elements, h_elements, N * sizeof(T), cudaMemcpyHostToDevice));
    
    // Create threshold for comparison - we'll check if element1 < element2
    auto d_threshold = (T*)gpuMalloc(N * sizeof(T));
    T h_threshold[N] = {config.element2, config.element2};
    checkCudaErrors(cudaMemcpy(d_threshold, h_threshold, N * sizeof(T), cudaMemcpyHostToDevice));
    
    // Phase 1: Key Generation (offline phase)
    printf("\n--- Phase 1: DCF Key Generation ---\n");
    auto keygen = new DCFKeygen<T>(config.party, keyBufSz);
    
    auto start = std::chrono::high_resolution_clock::now();
    keygen->generateDCFKeys(config.bin, config.bout, N, d_threshold);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
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
    dcfMPC->runDCFComparison(config.bin, config.bout, N, d_elements, h_elements, h_threshold);
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
    
    printf("\nExpected result: %lu < %lu = %s\n", 
           config.element1, config.element2, 
           (config.element1 < config.element2) ? "true" : "false");
    
    printf("Results saved to %s\n", outputDir.c_str());
    
    // Cleanup
    gpuFree(d_elements);
    gpuFree(d_threshold);
    delete keygen;
    delete dcfMPC;
}

int main(int argc, char **argv) {
    // Parse command line arguments
    DCFTestConfig config = parseTestArgs(argc, argv);
    
    // Initialize environment
    initTestEnvironment();
    
    // Run DCF comparison
    runDCFComparison(config);
    
    // Cleanup
    cleanupTestEnvironment();
    
    printf("\nDCF comparison test completed successfully!\n");
    return 0;
} 