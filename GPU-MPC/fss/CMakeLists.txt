# FSS (Function Secret Sharing) library
# Now includes both headers and implementation files

add_library(gpu_mpc_fss
    gpu_scmp.cu
    gpu_aes_shm_host.cu
    gpu_aes_table.cu
    gpu_dpf.cu
    gpu_compare_aggregate.cu
    dcf/gpu_dcf_host.cu
    dcf/gpu_relu_host.cu
)

target_include_directories(gpu_mpc_fss
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utils>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/optimizations>
        $<INSTALL_INTERFACE:include/gpu-mpc>
)

target_link_libraries(gpu_mpc_fss
    PUBLIC
        gpu_mpc_external
        gpu_mpc_utils
        CUDA::cudart
        CUDA::nvml
)

# Enable CUDA separable compilation for this library
set_target_properties(gpu_mpc_fss PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)