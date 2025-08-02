# Utility functions

set(UTILS_SOURCES
    # GPU utilities
    gpu_mem.cu
    # Note: gpu_random.cu is included by gpu_random.h, don't compile separately
    
    # File and communication utilities
    gpu_file_utils.cpp
    sigma_comms.cpp
)

add_library(gpu_mpc_utils STATIC ${UTILS_SOURCES})

target_include_directories(gpu_mpc_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include/gpu-mpc>
)

target_link_libraries(gpu_mpc_utils
    PUBLIC
        gpu_mpc_external  # This includes sytorch and other dependencies
        CUDA::cudart
        CUDA::nvml
)

# Set CUDA properties
set_target_properties(gpu_mpc_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
)