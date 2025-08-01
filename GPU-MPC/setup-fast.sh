#!/bin/bash
# Fast setup script that avoids building unnecessary CUTLASS components
set -e

echo "=== Fast GPU-MPC Setup ==="
echo "This script skips building CUTLASS tests and libraries"

# Set environment variables
export NVCC_PATH="/usr/local/cuda-$CUDA_VERSION/bin/nvcc"

echo "1. Updating submodules..."
git submodule update --init --recursive

# Install dependencies
echo "2. Installing system dependencies..."
sudo apt update
if ! apt-cache show gcc-9 g++-9 > /dev/null 2>&1; then
    echo "Adding repository for gcc-9/g++-9"
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt update
fi
sudo apt install -y gcc-9 g++-9 libssl-dev cmake python3-pip libgmp-dev libmpfr-dev libeigen3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

echo "3. Setting up CUTLASS (headers only)..."
# CUTLASS is header-only, we just need to make sure it's checked out
cd ext/cutlass
if [ -n "$1" ]; then 
   git checkout $1
fi
cd ../..

# We DON'T build CUTLASS - GPU-MPC only needs the headers!
echo "   Skipping CUTLASS build (header-only library)"

# Build sytorch (this is actually needed)
echo "4. Building Sytorch..."
mkdir -p ext/sytorch/build && cd ext/sytorch/build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../ -DCUDAToolkit_ROOT="/usr/local/cuda-$CUDA_VERSION/"
make sytorch -j$(nproc)
cd ../../..

# Download CIFAR-10 if needed
echo "5. Setting up datasets..."
cd experiments/orca/datasets/cifar-10
if [ ! -f "downloaded" ]; then
    sh download-cifar10.sh
    touch downloaded
fi
cd ../../../..

# Make shares of data
echo "6. Creating data shares..."
make share_data
cd experiments/orca
./share_data
cd ../..

# Create output directories
echo "7. Creating output directories..."
mkdir -p experiments/orca/output/P0/training
mkdir -p experiments/orca/output/P0/inference
mkdir -p experiments/orca/output/P1/training
mkdir -p experiments/orca/output/P1/inference
mkdir -p experiments/sigma/output/P0
mkdir -p experiments/sigma/output/P1

# Install Python dependencies
echo "8. Installing Python dependencies..."
pip3 install matplotlib

echo "=== Fast setup completed! ==="
echo "Time saved by skipping CUTLASS build: ~30-45 minutes"
echo "You can now run: make test"