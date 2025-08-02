cmake_minimum_required(VERSION 3.18)
project(GPU-MPC VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# System dependencies with helpful errors
include(cmake/SystemDependencies.cmake)

# CUDA configuration
include(cmake/CUDAConfig.cmake)

# Build options
include(cmake/BuildOptions.cmake)

# External dependencies via FetchContent
include(cmake/ExternalDependencies.cmake)

# Compiler flags
include(cmake/CompilerFlags.cmake)

# Dataset management
include(cmake/Datasets.cmake)

# Core library with FSS implementations
add_subdirectory(fss)

# Utility functions
add_subdirectory(utils)

# Neural network layers (if needed)
if(GPU_MPC_BUILD_ORCA OR GPU_MPC_BUILD_SIGMA)
    add_subdirectory(nn)
endif()

# Test programs
if(GPU_MPC_BUILD_TESTS)
    add_subdirectory(tests)
    
    # Main test program in root directory
    add_executable(test test.cu)
    target_link_libraries(test PRIVATE
        gpu_mpc_fss
        gpu_mpc_utils
        gpu_mpc_nn
        gpu_mpc_external
    )
    set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Experiments
if(GPU_MPC_BUILD_ORCA OR GPU_MPC_BUILD_SIGMA OR GPU_MPC_BUILD_PIRANHA)
    add_subdirectory(experiments)
endif()

# Export targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GPU-MPCConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Status messages
message(STATUS "")
message(STATUS "GPU-MPC Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests: ${GPU_MPC_BUILD_TESTS}")
message(STATUS "  Build Orca: ${GPU_MPC_BUILD_ORCA}")
message(STATUS "  Build SIGMA: ${GPU_MPC_BUILD_SIGMA}")
message(STATUS "  Build Piranha: ${GPU_MPC_BUILD_PIRANHA}")
message(STATUS "")