# CUDA Configuration

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Common GPU architectures:
    # 70: V100
    # 75: T4, RTX 2080
    # 80: A100
    # 86: RTX 3090, RTX 3080
    # 89: RTX 4090 (Ada Lovelace)
    # 90: H100
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    message(STATUS "CUDA architectures not specified, using: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp,-march=native,-fpermissive,-fpic")

# Add warning suppression for Eigen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20015")

# Build type specific flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

# Create interface library for CUDA dependencies
add_library(gpu_mpc_cuda INTERFACE)
target_link_libraries(gpu_mpc_cuda INTERFACE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::curand
)

# Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(WARNING "CUDA ${CUDAToolkit_VERSION} detected. GPU-MPC is tested with CUDA 11.0+")
endif()

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")