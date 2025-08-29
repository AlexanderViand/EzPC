# GPU-MPC Benchmarks

# Main benchmark executable
# Due to GPU-MPC's header-only pattern, we compile without separable compilation
add_executable(mpc_benchmark 
    mpc_benchmark.cu
    tasks/dcf_comparison.cu
    tasks/scmp_comparison.cu
    tasks/two_max.cu
    tasks/lss_ops.cu
    fss_optimization_test.cu
)

target_include_directories(mpc_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(mpc_benchmark PRIVATE
    gpu_mpc_fss
    gpu_mpc_utils
    gpu_mpc_nn
    gpu_mpc_external
    gpu_mpc_cuda
)

# Disable separable compilation to avoid multiple definition errors
set_target_properties(mpc_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Protocol-specific benchmarks can be added here later
# add_library(benchmark_protocols STATIC
#     protocols/dcf_bench.cu
#     protocols/scmp_bench.cu
#     protocols/sorting_bench.cu
# )