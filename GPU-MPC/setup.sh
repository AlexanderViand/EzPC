#!/bin/bash
set -e  # Exit on any error

# Set environment variables
export NVCC_PATH="/usr/local/cuda-$CUDA_VERSION/bin/nvcc"

echo "Updating submodules"
git submodule update --init --recursive

# Install dependencies
echo "Installing g++-9"
# Add repository for g++9 if not available in default repos
sudo apt update
if ! apt-cache show gcc-9 g++-9 > /dev/null 2>&1; then
    echo "Adding repository for gcc-9/g++-9"
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt update
fi
sudo apt install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --config gcc

# Installing dependencies
echo "Installing system dependencies"
sudo apt install -y libssl-dev cmake python3-pip libgmp-dev libmpfr-dev cmake make libeigen3-dev;


echo "Building CUTLASS"
mkdir -p ext/cutlass/build && cd ext/cutlass/build
# Build CUTLASS
if [ -n "$1" ]; then 
   git checkout $1
fi
cmake .. -DCUTLASS_NVCC_ARCHS=$GPU_ARCH -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=$NVCC_PATH
make -j$(nproc)
cd ../../..

# Build sytorch
echo "Building Sytorch"
mkdir -p ext/sytorch/build && cd ext/sytorch/build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../ -DCUDAToolkit_ROOT="/usr/local/cuda-$CUDA_VERSION/bin/"
make sytorch -j$(nproc)
cd ../../..

# Download CIFAR-10
echo "Downloading CIFAR-10 dataset"
cd experiments/orca/datasets/cifar-10
if [ ! -f "downloaded" ]; then
    sh download-cifar10.sh
    touch downloaded  # Mark as downloaded to avoid re-downloading
fi
cd ../../../..

# Make shares of data
echo "Creating data shares"
make share_data
cd experiments/orca
./share_data
cd ../..

# Build the orca codebase
# make orca; 

# Make output directories with -p flag
echo "Creating output directories"
# Orca
mkdir -p experiments/orca/output/P0/training
mkdir -p experiments/orca/output/P0/inference
mkdir -p experiments/orca/output/P1/training
mkdir -p experiments/orca/output/P1/inference

# Sigma
mkdir -p experiments/sigma/output/P0
mkdir -p experiments/sigma/output/P1

# install matplotlib
echo "Installing Python dependencies"
pip3 install matplotlib

echo "Setup completed successfully!"
