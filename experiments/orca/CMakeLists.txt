# Orca training experiments

# Dataset utilities
add_library(orca_datasets STATIC
    datasets/gpu_data.cu
    datasets/mnist.cpp
)
target_include_directories(orca_datasets PUBLIC datasets)

# Common sources
add_library(orca_common STATIC
    share_data.cpp
)

# Orca executables
add_executable(orca_dealer orca_dealer.cu)
add_executable(orca_evaluator orca_evaluator.cu)
add_executable(orca_inference orca_inference.cu)
add_executable(model_accuracy model_accuracy.cu)

# Piranha inference (part of Orca)
add_executable(piranha piranha.cu)

# Common dependencies for Orca
set(ORCA_LIBS
    gpu_mpc_fss
    gpu_mpc_utils
    gpu_mpc_nn
    gpu_mpc_external
    orca_datasets
    orca_common
)

# Link libraries for all executables
foreach(target orca_dealer orca_evaluator orca_inference model_accuracy piranha)
    target_link_libraries(${target} PRIVATE ${ORCA_LIBS})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    # Include datasets directory
    target_include_directories(${target} PRIVATE datasets)
endforeach()