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

#include "mpc_benchmark.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>


void initTestEnvironment() {
  initGPUMemPool();
  initGPURandomness();
}

void cleanupTestEnvironment() { destroyGPURandomness(); }



int main(int argc, char **argv) {
  // Simple argument parsing
  std::string task;
  int party = -1;
  std::string peer_ip;
  int threads = std::thread::hardware_concurrency();  // Default to all cores
  
  // Parse --task/-t, --party/-p, --peer/-i, --threads/-n
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "--task" || arg == "-t") && i + 1 < argc) {
      task = argv[++i];
    } else if ((arg == "--party" || arg == "-p") && i + 1 < argc) {
      party = atoi(argv[++i]);
    } else if ((arg == "--peer" || arg == "-i") && i + 1 < argc) {
      peer_ip = argv[++i];
    } else if ((arg == "--threads" || arg == "-n") && i + 1 < argc) {
      threads = atoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s --task <name> --party <0/1> --peer <ip> [--threads <n>]\n", argv[0]);
      printf("  -t, --task     Task to run\n");
      printf("  -p, --party    Party number (0 or 1)\n");
      printf("  -i, --peer     Peer IP address\n");
      printf("  -n, --threads  Number of threads (default: %d)\n", threads);
      printf("\nAvailable tasks:\n");
      for (const auto& [name, _] : getTaskRegistry()) {
        printf("  %s\n", name.c_str());
      }
      return 0;
    }
  }
  
  // Validate arguments
  if (task.empty() || party == -1 || peer_ip.empty()) {
    printf("Error: Missing required arguments\n");
    printf("Usage: %s --task <name> --party <0/1> --peer <ip> [--threads <n>]\n", argv[0]);
    printf("Run with --help for more information\n");
    return 1;
  }
  
  if (party != 0 && party != 1) {
    printf("Error: Party must be 0 or 1\n");
    return 1;
  }
  
  if (threads < 1) {
    printf("Error: Threads must be >= 1\n");
    return 1;
  }
  
  // Find and run the task
  auto& tasks = getTaskRegistry();
  auto it = tasks.find(task);
  if (it == tasks.end()) {
    printf("Error: Unknown task '%s'\n", task.c_str());
    printf("Available tasks:\n");
    for (const auto& [name, _] : tasks) {
      printf("  %s\n", name.c_str());
    }
    return 1;
  }
  
  // Initialize GPU environment
  initTestEnvironment();
  
  // Run the task
  it->second(party, peer_ip, threads);
  
  // Cleanup
  cleanupTestEnvironment();
  
  return 0;
}

// Simple JSON output implementation
void writeJSONResult(const char* test_name, const BenchmarkConfig& config, 
                    const Stats& stats, bool passed) {
    char filename[256];
    sprintf(filename, "output/P%d/%s_results.json", config.party, test_name);
    
    FILE* f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "{\n");
    fprintf(f, "  \"test\": \"%s\",\n", test_name);
    fprintf(f, "  \"party\": %d,\n", config.party);
    fprintf(f, "  \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(f, "  \"configuration\": {\n");
    fprintf(f, "    \"elements\": [%lu, %lu],\n", config.element1, config.element2);
    fprintf(f, "    \"bit_width\": %d,\n", config.bin);
    fprintf(f, "    \"cpu_threads\": %d\n", config.cpu_threads);
    fprintf(f, "  },\n");
    fprintf(f, "  \"timing_us\": {\n");
    fprintf(f, "    \"compute\": %lu,\n", stats.compute_time);
    fprintf(f, "    \"communication\": %lu,\n", stats.comm_time);
    fprintf(f, "    \"transfer\": %lu,\n", stats.transfer_time);
    fprintf(f, "    \"total\": %lu\n", stats.compute_time + stats.comm_time);
    fprintf(f, "  },\n");
    fprintf(f, "  \"network\": {\n");
    fprintf(f, "    \"comm_bytes\": %lu,\n", stats.linear_comm_bytes);
    double mb = stats.linear_comm_bytes / (1024.0 * 1024.0);
    fprintf(f, "    \"comm_mb\": %.2f,\n", mb);
    if (mb >= 1024.0) {
        fprintf(f, "    \"comm_gb\": %.2f\n", mb / 1024.0);
    } else {
        fprintf(f, "    \"comm_gb\": 0\n");
    }
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
}

