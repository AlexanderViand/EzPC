# GPU-MPC Benchmarking Framework Plan (Simplified)

## Overview
Transform the current `test.h/test.cu` into a simple, well-organized benchmarking system that provides real results, basic metrics, and clean output for GPU-MPC protocols.

## Current Issues
1. **Poor naming**: `test.h/test.cu` doesn't reflect its purpose
2. **Limited output**: Only shows where shares are stored, not actual results
3. **No verification**: Can't see if MPC computation produced correct results
4. **Basic metrics**: Only captures total time, not detailed phase breakdowns
5. **Poor location**: test.cu sits at the root of GPU-MPC directory

## Proposed Architecture (Minimal)

```
GPU-MPC/
├── benchmarks/
│   ├── mpc_benchmark.cu      # Main runner (renamed from test.cu)
│   ├── mpc_benchmark.h       # Minimal shared utilities
│   ├── protocols/
│   │   ├── dcf_bench.cu      # DCF-specific benchmarks
│   │   ├── scmp_bench.cu     # SCMP-specific benchmarks
│   │   └── sorting_bench.cu  # Compare-aggregate sorting
│   └── CMakeLists.txt
```

## Key Changes (Not Features)

### 1. Simple Result Verification
Add inline functions for share reconstruction - no classes needed:

```cpp
// In mpc_benchmark.h - simple utilities
template<typename T>
inline bool verifyBooleanShare(T share0, T share1, T expected) {
    return ((share0 ^ share1) & 1) == expected;
}

template<typename T>
inline T reconstructArithmetic(T share0, T share1) {
    return share0 + share1;
}
```

### 2. Enhanced Existing Stats
Extend the existing Stats struct (from utils/gpu_stats.h) with a few fields:

```cpp
// Just add to existing Stats:
uint64_t keygen_time = 0;
uint64_t verification_time = 0;
size_t key_size_bytes = 0;
```

### 3. Better Console Output
Replace current printf statements with clearer formatting:

```cpp
// Instead of: "Party 0 comparison results: Element 0 (42) < 35: P0_share = 1"
printf("=== DCF Comparison Results ===\n");
printf("Test: %lu < %lu\n", a, b);
printf("Expected: %s\n", (a < b) ? "true" : "false");
printf("Reconstructed: %s %s\n", result ? "true" : "false",
       result == (a < b) ? "✓" : "✗ MISMATCH!");
```

### 4. Simple JSON Output (Optional)
One function to write results - no framework needed:

```cpp
void writeJSONResult(const char* test_name, const BenchmarkConfig& config, 
                    const Stats& stats, bool passed) {
    char filename[256];
    sprintf(filename, "output/P%d/%s_results.json", config.party, test_name);
    
    FILE* f = fopen(filename, "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"test\": \"%s\",\n", test_name);
    fprintf(f, "  \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(f, "  \"timing_us\": {\n");
    fprintf(f, "    \"keygen\": %lu,\n", stats.keygen_time);
    fprintf(f, "    \"compute\": %lu,\n", stats.compute_time);
    fprintf(f, "    \"total\": %lu\n", stats.keygen_time + stats.compute_time);
    fprintf(f, "  },\n");
    fprintf(f, "  \"network\": {\n");
    fprintf(f, "    \"bytes_sent\": %lu,\n", stats.bytes_sent);
    fprintf(f, "    \"bytes_received\": %lu\n", stats.bytes_received);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
}
```

## Implementation Steps

### Step 1: Reorganize (30 minutes)
1. Create `benchmarks/` directory
2. Move `test.cu` → `benchmarks/mpc_benchmark.cu`
3. Move `test.h` → `benchmarks/mpc_benchmark.h`
4. Update CMakeLists.txt

### Step 2: Enhance Output (1 hour)
1. Add share reconstruction after computation
2. Improve console output formatting
3. Show actual vs expected results

### Step 3: Extract Protocol Code (1 hour)
1. Move DCF-specific code to `protocols/dcf_bench.cu`
2. Move SCMP-specific code to `protocols/scmp_bench.cu`
3. Keep main runner generic

### Step 4: Add Basic Metrics (30 minutes)
1. Capture keygen time separately
2. Add key size tracking
3. Simple JSON output function

### Step 5: Update SkyPilot Config (15 minutes)
1. Update binary path in sky.yaml
2. Update any scripts that reference old binary

## What We're NOT Doing

- No abstract base classes
- No complex frameworks
- No metrics collectors
- No output formatters
- No multi-week implementation
- No CUPTI integration
- No cost estimation
- No regression detection

## Benefits

1. **Clean organization**: Benchmarks in logical location with good names
2. **See real results**: Immediate verification of correctness
3. **Simple implementation**: ~200 lines of changes total
4. **Maintains existing patterns**: Follows GPU-MPC style
5. **Easy to extend**: Just add new files to protocols/

## Example Enhanced Output

```
=== GPU-MPC DCF Benchmark ===
Configuration: 1M comparisons, 32-bit, 4 threads

Phase 1: Key Generation... done (12.5ms, 2.4MB keys)
Phase 2: Online Computation... done (3.2ms)

Results:
  Sample verification: 42 < 35 = false ✓
  Batch verification: 1,000,000 comparisons ✓
  
Performance:
  Total: 15.7ms (799kHz)
  Network: 1.2MB transferred
  
Output saved to: output/P0/dcf_results.json
```

This approach gives us better organization and output with minimal changes to the existing codebase.