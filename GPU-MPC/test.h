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
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"
#include "fss/dcf/gpu_dcf.h"
#include "fss/gpu_scmp.h"
#include <sytorch/tensor.h>
#include <sytorch/backend/llama_transformer.h>
#include <llama/comms.h>
#include <llama/api.h>
#include <omp.h>
#include <chrono>
#include <string>

using T = u64;

// DCF comparison test configuration
struct DCFTestConfig {
    int bin;            // Input bit width
    int bout;           // Output bit width  
    int party;          // Party number (0/1)
    std::string peer_ip; // Peer IP address
    int cpu_threads;    // Number of CPU threads
    T element1;         // First element to compare
    T element2;         // Second element to compare
    bool run_scmp;      // Whether to run SCMP test
};

// Function to parse command line arguments similar to sigma
DCFTestConfig parseTestArgs(int argc, char **argv);

// Function to run DCF-based comparison between two elements
void runDCFComparison(const DCFTestConfig& config);

// Helper function to initialize test environment
void initTestEnvironment();

// Helper function to cleanup test environment  
void cleanupTestEnvironment(); 