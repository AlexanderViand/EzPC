
# GPU-MPC

Implementation of protocols from the papers [Orca](https://eprint.iacr.org/2023/206) and [SIGMA](https://eprint.iacr.org/2023/1269).

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires NVIDIA GPUs, and assumes that GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) are already installed. The following has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2 and g++-9. 

GPU-MPC requires CMake version >= 3.18 for the modern build system.

### Quick Start (10-30 minutes)

```bash
# 1. CMake will check system dependencies and tell you what's missing
cmake --preset test-only

# 2. Install any missing packages (example)
sudo apt update
sudo apt install gcc-9 g++-9 libssl-dev libeigen3-dev libgmp-dev libmpfr-dev

# 3. Build (choose one)
cmake --preset test-only    # Fast: Core tests only (~10 min)
cmake --preset full         # Full: Everything including Orca (~30 min)
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
    -DGPU_MPC_BUILD_ORCA=OFF \     # OFF = skip SEAL, save 20 min
    -DGPU_MPC_BUILD_SIGMA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86   # Or "native" for current GPU
    
cmake --build build -j
```

### Build Components

| Option | Default | Description | Additional Dependencies |
|--------|---------|-------------|------------------------|
| `GPU_MPC_BUILD_TESTS` | ON | Build test programs | None |
| `GPU_MPC_BUILD_ORCA` | OFF | Build Orca training | SEAL, SCI-FloatML (~20 min) |
| `GPU_MPC_BUILD_SIGMA` | ON | Build SIGMA GPT inference | None |
| `GPU_MPC_BUILD_PIRANHA` | ON | Build Piranha inference | None |
| `GPU_MPC_DOWNLOAD_DATA` | OFF | Download CIFAR-10 & setup | Python3, matplotlib |

### Dependencies

The build system automatically manages dependencies:

#### Core Dependencies (Always Required)
- **CUTLASS**: NVIDIA GPU math (fetched automatically, headers only)
- **Sytorch-GPU**: GPU-enhanced MPC framework in `sytorch-gpu/` with cryptoTools, LLAMA, bitpack, SCI
- **System**: GCC 9+, OpenSSL, Eigen3, OpenMP, GMP, MPFR

#### Component-Specific
- **Orca only**: SEAL (fetched automatically), SCI float libraries
- **SIGMA/Test**: No additional dependencies

#### Datasets (Optional)
- **MNIST & CIFAR-10**: Downloaded to `data/` when `-DGPU_MPC_DOWNLOAD_DATA=ON`
- **No git submodules required!**

### Build Performance

The CMake build system is optimized for speed:
- **CUTLASS**: Headers only, no compilation needed
- **SEAL**: Only built when Orca is enabled (saves ~20 min)
- **Parallel fetching**: Dependencies downloaded concurrently
- **Conditional compilation**: Only builds what you need

## Running Tests

### Basic DCF/SCMP Test
```bash
./build/tests/test test 128 0 <peer_ip> 64 1
# Arguments: <model> <sequence_length> <party> <peer_ip> <cpu_threads> <run_scmp>
```

### Individual FSS Tests
```bash
./build/tests/relu
./build/tests/dcf_dcf
./build/tests/softmax
```

## Run Orca

Please see the [Orca README](experiments/orca/README.md).

## Run SIGMA

Please see the [SIGMA README](experiments/sigma/README.md)

## Troubleshooting

### Missing Dependencies
CMake will tell you exactly what's missing:
```
CMake Error: OpenSSL not found.
Install with:
  sudo apt update
  sudo apt install libssl-dev
```

### Slow Compilation
- Use `cmake --preset single-gpu` to build for your GPU only
- Limit parallel jobs: `cmake --build build -j 4`

### SEAL/Orca Errors
- If you don't need Orca: `-DGPU_MPC_BUILD_ORCA=OFF`
- SEAL takes ~15-20 minutes to build on first run

### IDE Integration
The CMake build generates `compile_commands.json` for:
- VS Code (with CMake Tools extension)
- CLion
- Vim/Neovim (with clangd LSP)

## Docker Build

You can also build the docker image using the provided Dockerfile_Gen for building the Environment. 

### Install Nvidia Container Toolkit
- Configure the repository:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
```

- Install the NVIDIA Container Toolkit packages:
```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
### Build the Docker Image / pull the image from Docker Hub
```
# Local Build
docker build -t gpu_mpc -f Dockerfile_Gen .

# Pull from Docker Hub (Cuda 11.8)
docker pull trajore/gpu_mpc
```
### Run the Docker Container
```
sudo docker run --gpus all --network host -v /home/$USER/path_to_GPU-MPC/:/home -it container_name /bin/bash
```

Then build using CMake:
```bash
cmake --preset test-only  # or 'full' for Orca
cmake --build build
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
sky launch sky.yaml --env TEST=scmp

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
sky exec <cluster-name> sky.yaml --env TEST=dcf
```

### Configuration Options

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `TEST` | `basic`, `dcf`, `scmp` | `basic` | MPC protocol to run |
| `GPU` | `T4`, `V100`, `A100`, `L4` | `T4` | GPU type (auto-sets CUDA arch) |
| `NODES` | `1`, `2` | `2` | 1=dev mode, 2=MPC mode |
| `MODEL` | `dcf-test`, etc. | `dcf-test` | Model to test |
| `SEQ_LEN` | Any integer | `128` | Sequence length |
| `CPU_THREADS` | Any integer | `4` | Number of CPU threads |

### Cost Optimization

- Uses spot instances by default (3-6x cheaper)
- T4 GPUs recommended for best cost/performance
- Remember to terminate clusters: `sky down <cluster-name>`

See `sky.yaml` for full configuration details.

