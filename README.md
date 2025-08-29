
# GPU-MPC

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires NVIDIA GPUs, and assumes that GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) are already installed. The following has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2 and g++-9. 

GPU-MPC requires CMake version >= 3.18 for the modern build system.

### Quick Start (10-30 minutes)

```bash
sudo apt update && sudo apt install gcc-9 g++-9 libssl-dev libeigen3-dev libgmp-dev libmpfr-dev libomp-dev

cmake -S . -B build

cmake --build build
```

### Build Options

| Preset | Build Time | Includes | Use Case |
|--------|------------|----------|----------|
| `test-only` | ~10 min | Core FSS, DCF, SCMP, SIGMA | Testing & Development |
| `full` | ~30 min | Everything + Orca + SEAL | Full deployment |
| `single-gpu` | Varies | Current GPU arch only | Faster compilation |
| `debug` | Varies | Debug symbols | Debugging |

### Manual Configuration

```bash
cmake -B build \
    -DGPU_MPC_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86   # Or "native" for current GPU
    
cmake --build build --parallel
```

### Build Components

| Option | Default | Description | Additional Dependencies |
|--------|---------|-------------|------------------------|
| `GPU_MPC_BUILD_TESTS` | ON | Build test programs | None |


### Dependencies

The build system automatically manages dependencies:

#### Core Dependencies (Always Required)
- **CUTLASS**: NVIDIA GPU math (fetched automatically, headers only)
- **Sytorch-GPU**: GPU-enhanced MPC framework in `sytorch-gpu/` with cryptoTools, LLAMA, bitpack, SCI
- **System**: GCC 9+, OpenSSL, Eigen3, OpenMP, GMP, MPFR

### Build Performance
The CMake build system is optimized for speed:
- **CUTLASS**: Headers only, no compilation needed
- **Parallel fetching**: Dependencies downloaded concurrently
- **Conditional compilation**: Only builds what you need

## Running Benchmarks and Tests

### MPC Benchmarks
```bash
# DCF (Distributed Comparison Function) benchmark
./build/benchmarks/mpc_benchmark --task dcf --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task dcf --party 1 --peer <peer_ip> --threads 4

# SCMP (Secure Comparison) benchmark
./build/benchmarks/mpc_benchmark --task scmp --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task scmp --party 1 --peer <peer_ip> --threads 4

# Two-iteration maximum benchmark
./build/benchmarks/mpc_benchmark --task twomax --party 0 --peer <peer_ip> --threads 4
./build/benchmarks/mpc_benchmark --task twomax --party 1 --peer <peer_ip> --threads 4

# Show help and available tasks
./build/benchmarks/mpc_benchmark --help
```

### Individual FSS Tests
```bash
./build/tests/relu
./build/tests/dcf
./build/tests/softmax
./build/tests/gelu
./build/tests/layernorm
./build/tests/mha
./build/tests/truncate
# And many more in build/tests/
```

### Benchmark Output
The benchmarks save results in both human-readable and JSON formats to `./output/P{party}/`:
- `results_summary.txt` - Human-readable summary
- `results.json` - Machine-readable JSON with detailed metrics
- `phase_*.json` - Per-phase timing breakdowns

## Troubleshooting

### Missing Dependencies
CMake will tell you exactly what's missing:
```
CMake Error: OpenSSL not found.
Install with:
  sudo apt update
  sudo apt install libssl-dev
```

## Cloud Deployment with SkyPilot

GPU-MPC can be easily deployed to cloud GPUs using [SkyPilot](https://skypilot.co/). This automates cluster provisioning, dependency installation, and multi-node MPC execution.

### Quick Start

```bash
# Install SkyPilot
pip install skypilot[gcp]  # or [aws], [azure]

# Launch 2-node MPC cluster (auto-terminates after 60 min idle by default)
cd GPU-MPC
sky launch sky.yaml

# Run SCMP protocol
sky launch sky.yaml --env TASK=scmp

# Development mode (single node)
sky launch sky.yaml --env NODES=1

# Override auto-termination (e.g., for long experiments)
sky launch sky.yaml --idle-minutes-to-autostop 180  # 3 hours
sky launch sky.yaml -i 0  # Disable auto-termination
```

### Development Workflow

After making code changes:

```bash
# Sync and rebuild without restarting cluster
sky exec <cluster-name> sky.yaml --env REBUILD=1

# Run updated test
sky exec <cluster-name> sky.yaml --env TASK=dcf
```

### Configuration Options

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `TASK` | `dcf`, `scmp`, `twomax` | `dcf` | MPC protocol to run |
| `GPU` | `T4`, `V100`, `A100`, `L4` | `T4` | GPU type (auto-sets CUDA arch) |
| `NODES` | `1`, `2` | `2` | 1=dev mode, 2=MPC mode |
| `CPU_THREADS` | Any integer | `4` | Number of CPU threads |

### Cost Optimization

- Uses spot instances by default (3-6x cheaper)
- T4 GPUs recommended for best cost/performance
- Remember to terminate clusters: `sky down <cluster-name>`

See `sky.yaml` for full configuration details.
