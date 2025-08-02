# FSS (Function Secret Sharing) library
# This is a header-only library with template implementations

add_library(gpu_mpc_fss INTERFACE)

target_include_directories(gpu_mpc_fss
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utils>
        $<INSTALL_INTERFACE:include/gpu-mpc>
)

target_link_libraries(gpu_mpc_fss
    INTERFACE
        gpu_mpc_external
        CUDA::cudart
        CUDA::nvml
)