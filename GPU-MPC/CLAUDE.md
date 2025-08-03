# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GPU-MPC is an academic proof-of-concept implementation of secure multi-party computation (MPC) protocols for GPUs, implementing Orca (secure training) and SIGMA (secure GPT inference) protocols. This is a CUDA-based system with CMake build system.

## Build System

### Quick Build Commands
```bash
# Fast build (core tests only, ~10 min)
cmake --preset test-only
cmake --build build --parallel

# Full build (includes Orca with SEAL, ~30 min)  
cmake --preset full
cmake --build build --parallel

# Single GPU architecture (faster compilation)
cmake --preset single-gpu
cmake --build build --parallel

# Debug build
cmake --preset debug
cmake --build build --parallel
```

### Manual Configuration
```bash
cmake -B build -S . \
    -DGPU_MPC_BUILD_TESTS=ON \
    -DGPU_MPC_BUILD_ORCA=OFF \
    -DGPU_MPC_BUILD_SIGMA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --parallel
```

### Build Options
- `GPU_MPC_BUILD_TESTS=ON/OFF` - Build test programs
- `GPU_MPC_BUILD_ORCA=ON/OFF` - Build Orca (requires SEAL, adds ~20 min)
- `GPU_MPC_BUILD_SIGMA=ON/OFF` - Build SIGMA GPT inference
- `GPU_MPC_BUILD_PIRANHA=ON/OFF` - Build Piranha fast inference
- `GPU_MPC_DOWNLOAD_DATA=ON/OFF` - Download CIFAR-10 dataset

## Running Tests and Benchmarks

### MPC Benchmarks (Multi-party)
```bash
# DCF (Distributed Comparison Function) - run on two machines
./build/benchmarks/mpc_benchmark --task dcf --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task dcf --party 1 --peer <peer_ip> --threads 4

# SCMP (Secure Comparison)
./build/benchmarks/mpc_benchmark --task scmp --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task scmp --party 1 --peer <peer_ip> --threads 4

# Two-iteration maximum
./build/benchmarks/mpc_benchmark --task twomax --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task twomax --party 1 --peer <peer_ip> --threads 4
```

### Individual FSS Tests (Single machine)
```bash
./build/tests/relu
./build/tests/dcf  
./build/tests/softmax
./build/tests/gelu
./build/tests/layernorm
./build/tests/mha
./build/tests/truncate
```

### Experiment Scripts
```bash
# Orca experiments (requires config.json setup)
cd experiments/orca
python run_experiment.py --party 0 --figure 5a
python run_experiment.py --party 0 --table 3

# SIGMA experiments (requires config.json setup)
cd experiments/sigma  
python run_experiment.py --party 0 --perf true
./sigma gpt2 128 0 <client_ip> 64
```

## Architecture

### Core Components
- **fss/**: Function Secret Sharing implementations (DCF, DPF, AES operations)
- **utils/**: GPU memory management, communications, file utilities
- **nn/**: Neural network layers for Orca
- **experiments/**: High-level protocol implementations (Orca, SIGMA)
- **benchmarks/**: MPC benchmark suite with task registration system
- **tests/**: Individual component tests

### External Dependencies
- **ext/sytorch/**: GPU-enhanced MPC framework with cryptoTools, LLAMA, bitpack, SCI
- **CUTLASS**: NVIDIA GPU math (headers only, fetched automatically)
- **SEAL**: Only for Orca builds (fetched automatically)

### Key Files
- `benchmarks/mpc_benchmark.cu`: Main benchmark executable with task system
- `benchmarks/mpc_benchmark.h`: Task registration and configuration framework  
- `CMakePresets.json`: Predefined build configurations
- `sky.yaml`: SkyPilot cloud deployment configuration

## Development Workflow

### Dependencies
- CUDA Toolkit (tested with 11.7+)
- CMake >= 3.18
- GCC 9+, OpenSSL, Eigen3, OpenMP, GMP, MPFR

### Cloud Development
```bash
# Launch 2-node MPC cluster
sky launch sky.yaml --env TASK=dcf

# Development mode (single node) 
sky launch sky.yaml --env NODES=1

# Rebuild and test
sky exec <cluster> sky.yaml --env REBUILD=1 --env TASK=scmp
```

### Configuration Files
- `experiments/orca/config.json`: Orca dealer/evaluator GPU and network config
- `experiments/sigma/config.json`: SIGMA party configuration

## Important Notes

- Use `--parallel` with cmake --build, not `-jX`
- Orca builds require ~500GB free space for FSS keys
- SIGMA only supports power-of-2 sequence lengths  
- All protocols require exactly 2 parties
- Results saved to `./output/P{party}/` directories
- This is academic prototype code, not production-ready