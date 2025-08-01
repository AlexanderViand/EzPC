# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EzPC is a comprehensive framework for easy secure multiparty computation (MPC), particularly focused on machine learning applications. The repository contains multiple components that work together to enable privacy-preserving computation:

- **GPU-MPC**: GPU-accelerated Function Secret Sharing (FSS) protocols for high-performance secure computation
- **FSS (LLAMA)**: Low Latency Math Library for Secure Inference using Function Secret Sharing
- **SCI**: Semi-honest 2-party computation library for secure neural network inference
- **Athos**: End-to-end compiler from TensorFlow/ONNX to various MPC protocols
- **Porthos**: Semi-honest 3-party computation protocol optimized for ML workloads
- **Beacon**: Framework for secure floating-point neural network training

## Development Commands

### Initial Setup
```bash
# Quick setup with default paths (recommended)
./setup_env_and_build.sh quick

# Manual setup with custom paths
./setup_env_and_build.sh

# Activate the virtual environment
source mpc_venv/bin/activate
```

### GPU-MPC Commands
```bash
# Build GPU-MPC (requires CUDA, tested with CUDA 11.7)
export CUDA_VERSION=11.7
export GPU_ARCH=86  # Set based on your GPU architecture
cd GPU-MPC
sh setup.sh         # Initial setup
make orca          # Build Orca protocols
make sigma         # Build SIGMA protocols

# Run experiments
cd experiments/orca
python run_experiment.py

cd experiments/sigma  
python run_experiment.py
```

### FSS/LLAMA Commands
```bash
# Build FSS
cd FSS
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../
make install

# Compile EzPC programs to FSS
fssc --bitlen <bitlen> program.ezpc

# Run FSS programs
./program.out r=1 file=1              # Dealer (generates keys)
./program.out r=2 file=1 port=8000    # Server
./program.out r=3 file=1 server=<ip> port=8000  # Client

# Run tests
cd tests
./run.sh <test_name>
./runall.sh  # Run all tests
```

### EzPC Compiler Commands
```bash
cd EzPC/EzPC
eval `opam env`
make  # Build the compiler

# Compile EzPC programs
./ezpc.sh <backend> <input.ezpc> <bitlen>
# Backends: CPP, CPPRING, ABY, EMP, OBLIVC, PORTHOS, SCI, SECFLOAT, FSS
```

### Running Tests
```bash
# SCI tests
cd SCI/build
ctest

# Athos tests  
cd Athos
pytest -v tests/

# FSS tests
cd FSS/tests
./runall.sh
```

## Architecture Overview

### GPU-MPC Structure
GPU-MPC implements GPU-accelerated protocols for secure computation:
- `backend/`: Protocol definitions (Orca, Piranha, SIGMA)
- `fss/`: GPU implementations of FSS primitives (DPF, DCF, math functions)
- `nn/`: Neural network layers for Orca
- `experiments/`: Benchmarking code and datasets
- `utils/`: GPU utilities, memory management, communication

Key protocols:
- **Orca**: FSS-based secure training with GPUs
- **SIGMA**: Secure GPT inference with Function Secret Sharing
- **Piranha**: GPU-accelerated secure inference

### FSS/LLAMA Structure
FSS provides low-latency implementations of secure computation primitives:
- `src/`: Core FSS implementations (DPF, DCF, arithmetic/boolean operations)
- `src/api.cpp`: Main API for EzPC integration
- `microbenchmarks/`: Individual operation benchmarks
- `benchmarks/`: Full model benchmarks (ResNet, MiniONN, etc.)

The FSS compiler (`fssc`) compiles EzPC programs to use FSS protocols, performing Three-Address-Code optimization for efficiency.

### Communication Architecture
- GPU-MPC uses custom GPU-aware communication with support for NVLink/PCIe
- FSS uses socket-based communication between dealer, server, and client
- Both support multi-threaded execution for parallelism

## Important Notes

1. **GPU Requirements**: GPU-MPC requires NVIDIA GPUs with appropriate CUDA drivers. Set `GPU_ARCH` based on your GPU compute capability.

2. **Memory Allocation**: GPU-MPC allocates up to 20GB of GPU memory by default. Adjust in code if needed.

3. **FSS Configuration**: For FSS math functions, edit `src/config.h` to match the required (bitlength, scale) configuration before recompiling.

4. **Python Environment**: Always activate the virtual environment before using Athos or running experiments:
   ```bash
   source mpc_venv/bin/activate
   ```

5. **Thread Count**: For FSS/GPU-MPC, ensure server and client use the same thread count (`nt` parameter).

6. **Git Branches**: The repository uses `master` as the main branch (not `main`) for pull requests.