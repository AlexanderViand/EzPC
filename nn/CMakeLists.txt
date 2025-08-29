# Neural Network layers
# This is a header-only library with template implementations

add_library(gpu_mpc_nn INTERFACE)

target_include_directories(gpu_mpc_nn
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include/gpu-mpc>
)

target_link_libraries(gpu_mpc_nn
    INTERFACE
        gpu_mpc_fss
        gpu_mpc_utils
        gpu_mpc_external
        CUDA::cudart
        CUDA::cublas
)