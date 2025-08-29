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

#pragma once

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.cuh"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.cuh"
#include <sytorch/tensor.h>
#include <sytorch/backend/llama_transformer.h>
#include <llama/comms.h>
#include <llama/api.h>
#include <omp.h>
#include <chrono>
#include <string>
#include <functional>
#include <map>

using T = u64;

// MPC benchmark configuration
struct BenchmarkConfig {
    int bin;            // Input bit width
    int bout;           // Output bit width  
    int party;          // Party number (0/1)
    std::string peer_ip; // Peer IP address
    int cpu_threads;    // Number of CPU threads
    T element1;         // First element to compare
    T element2;         // Second element to compare
    std::string task;   // Task to run (dcf, scmp, twomax)
};

// Task registration system
using TaskFunction = std::function<void(int party, const std::string& peer_ip, int threads)>;

// Robust registration pattern to avoid static initialization order issues
inline std::map<std::string, TaskFunction>& getTaskRegistry() {
    static std::map<std::string, TaskFunction>* tasks = nullptr;
    if (!tasks) tasks = new std::map<std::string, TaskFunction>();
    return *tasks;
}

inline bool registerTask(const std::string& name, TaskFunction func) {
    getTaskRegistry()[name] = func;
    return true;
}

// Helper function to initialize test environment
void initTestEnvironment();

// Helper function to cleanup test environment  
void cleanupTestEnvironment();

// Simple share reconstruction utilities
template<typename T>
inline bool reconstructBoolean(T share0, T share1) {
    return (share0 ^ share1) & 1;
}

template<typename T>
inline T reconstructArithmetic(T share0, T share1) {
    return share0 + share1;
}

// Verify boolean comparison result
template<typename T>
inline bool verifyBooleanResult(T share0, T share1, bool expected, const char* test_name = nullptr) {
    bool result = reconstructBoolean(share0, share1);
    bool passed = (result == expected);
    
    if (!passed && test_name) {
        printf("VERIFICATION FAILED for %s: expected %s, got %s\n", 
               test_name, expected ? "true" : "false", result ? "true" : "false");
    }
    
    return passed;
}

// Simple JSON output
void writeJSONResult(const char* test_name, const BenchmarkConfig& config, 
                    const Stats& stats, bool passed); 