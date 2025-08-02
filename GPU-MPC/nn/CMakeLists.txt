# Neural Network layers

# Orca neural network layers
set(NN_SOURCES
    orca/avg_pool_layer.cu
    orca/conv2d_layer.cu
    orca/fc_layer.cu
    orca/maxpool_layer.cu
    orca/relu_extend_layer.cu
    orca/relu_layer.cu
)

add_library(gpu_mpc_nn STATIC ${NN_SOURCES})

target_include_directories(gpu_mpc_nn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include/gpu-mpc>
)

target_link_libraries(gpu_mpc_nn
    PUBLIC
        gpu_mpc_fss
        gpu_mpc_utils
        gpu_mpc_external
        CUDA::cudart
        CUDA::cublas
)

# Set CUDA properties
set_target_properties(gpu_mpc_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
)