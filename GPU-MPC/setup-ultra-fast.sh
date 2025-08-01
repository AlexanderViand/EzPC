#!/bin/bash
# Ultra-fast setup that downloads only CUTLASS headers
set -e

echo "=== Ultra-Fast GPU-MPC Setup ==="
echo "Downloads only necessary components"

# Install dependencies
echo "1. Installing system dependencies..."
sudo apt update && sudo apt install -y \
    gcc-9 g++-9 libssl-dev cmake python3-pip \
    libgmp-dev libmpfr-dev libeigen3-dev wget

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Download CUTLASS headers only (no submodule update)
echo "2. Getting CUTLASS headers..."
if [ ! -d "ext/cutlass/include" ]; then
    echo "   Downloading CUTLASS v2.11.0 headers..."
    mkdir -p ext/cutlass
    cd ext/cutlass
    # Download just the include directory from GitHub
    wget -q https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.tar.gz
    tar -xzf v2.11.0.tar.gz --strip-components=1 cutlass-2.11.0/include
    rm v2.11.0.tar.gz
    cd ../..
else
    echo "   CUTLASS headers already present"
fi

# Update only sytorch submodule
echo "3. Setting up Sytorch..."
git submodule update --init ext/sytorch

# Build sytorch
echo "4. Building Sytorch..."
mkdir -p ext/sytorch/build && cd ext/sytorch/build
cmake -DCMAKE_BUILD_TYPE=Release ../ -DCUDAToolkit_ROOT="/usr/local/cuda-$CUDA_VERSION/"
make sytorch -j$(nproc)
cd ../../..

# Quick dataset setup
echo "5. Creating directories..."
mkdir -p experiments/orca/datasets/cifar-10
mkdir -p experiments/orca/output/P{0,1}/{training,inference}
mkdir -p experiments/sigma/output/P{0,1}

# Skip data download for now (can be done later if needed)
echo "   Skipping dataset download (run experiments/orca/datasets/cifar-10/download-cifar10.sh if needed)"

echo "6. Installing Python dependencies..."
pip3 install matplotlib

echo "=== Ultra-fast setup completed! ==="
echo "Setup time: ~5-10 minutes (vs 45-60 minutes for full setup)"
echo "Run 'make test' to build the test executable"