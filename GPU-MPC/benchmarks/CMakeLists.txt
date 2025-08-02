# GPU-MPC Benchmarks

# Main benchmark executable
add_executable(mpc_benchmark mpc_benchmark.cu)
target_link_libraries(mpc_benchmark PRIVATE
    gpu_mpc_fss
    gpu_mpc_utils
    gpu_mpc_nn
    gpu_mpc_external
    gpu_mpc_cuda
)
set_target_properties(mpc_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Protocol-specific benchmarks can be added here later
# add_library(benchmark_protocols STATIC
#     protocols/dcf_bench.cu
#     protocols/scmp_bench.cu
#     protocols/sorting_bench.cu
# )