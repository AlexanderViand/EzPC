# GPU-MPC SkyPilot Setup

Simple one-file configuration for running GPU-MPC secure multi-party computation in the cloud.

## 🚀 Quick Start

```bash
# Navigate to GPU-MPC directory first
cd GPU-MPC

# Basic 2-party MPC
sky launch skypilot/sky.yaml

# SCMP protocol on L4 GPUs
sky launch skypilot/sky.yaml --env TEST=scmp --env GPU=L4

# Development mode (single node)
sky launch skypilot/sky.yaml --env NODES=1
```

## 🔧 Environment Variables

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `TEST` | `basic`, `dcf`, `scmp` | `basic` | MPC protocol to run |
| `GPU` | `T4`, `V100`, `A100`, `L4` | `T4` | GPU type (auto-sets CUDA arch) |
| `NODES` | `1`, `2` | `2` | 1=dev mode, 2=MPC mode |
| `MODEL` | `dcf-test`, etc. | `dcf-test` | Model to test |
| `SEQ_LEN` | Any integer | `128` | Sequence length for test |
| `CPU_THREADS` | Any integer | `4` | Number of CPU threads |

## 💡 GPU Recommendations

- **T4**: Most cost-effective ($0.35/hour, CUDA arch 75)
- **L4**: Best performance/cost ($0.60/hour, CUDA arch 89)  
- **A100**: High performance ($2.93/hour, CUDA arch 80)

## 📋 Examples

```bash
# From GPU-MPC directory
cd GPU-MPC

# Basic FSS test
sky launch skypilot/sky.yaml

# SCMP on A100
sky launch skypilot/sky.yaml --env TEST=scmp --env GPU=A100

# Compare-and-aggregate
sky launch skypilot/sky.yaml --env TEST=compare --env GPU=L4

# Development workflow
sky launch skypilot/sky.yaml --env NODES=1
sky ssh gpu-mpc
# Then: ./test 0 & ./test 1 localhost
```

## 🔍 Management

```bash
# Check status
sky status

# View logs  
sky logs gpu-mpc

# SSH into cluster
sky ssh gpu-mpc

# Terminate
sky down gpu-mpc -y
```

## 🔄 Update Code Without Restarting

After making code changes locally:

```bash
# Option 1: Sync and re-run (keeps cluster running)
sky exec gpu-mpc skypilot/sky.yaml

# Option 2: Just sync files without running
sky exec gpu-mpc --run 'echo "Files synced"'

# Option 3: Sync and rebuild manually
sky exec gpu-mpc --run 'cd /workspace/GPU-MPC && \
  rm -rf build && mkdir build && cd build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 .. && \
  cmake --build . --parallel --target test && \
  cp test ../test'

# Option 4: Sync and run specific test
sky exec gpu-mpc skypilot/sky.yaml --env TEST=scmp
```

This saves time by:
- Keeping instances running (no provisioning delay)
- Only syncing changed files (respects .skyignore)
- Reusing existing build artifacts when possible

## ⚡ What Makes This Fast

✅ **Modern CMake build**: 3-5x faster than old makefiles  
✅ **Parallel compilation**: Uses all CPU cores  
✅ **Smart dependencies**: Only fetches what's needed  
✅ **Auto-GPU detection**: Sets CUDA arch automatically  

## 🛠️ Development Workflow

```bash
# From GPU-MPC directory
cd GPU-MPC

# Launch dev environment
sky launch skypilot/sky.yaml --env NODES=1

# SSH in
sky ssh gpu-mpc

# Manual testing
cd /workspace/GPU-MPC
./test 0 &           # Start server
./test 1 localhost   # Connect client

# Test different protocols
./test 0 --run-scmp &
./test 1 localhost --run-scmp
```

## 🐛 Troubleshooting

**Build fails**: All dependencies installed automatically  
**CUDA errors**: GPU type auto-detected from `GPU` env var  
**Connection issues**: Both nodes start automatically  
**Test failures**: Check `sky logs gpu-mpc` for details