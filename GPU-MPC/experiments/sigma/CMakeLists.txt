# SIGMA GPT inference experiments

# SIGMA executables
add_executable(sigma sigma.cu)
add_executable(sigma_offline_online sigma_offline_online.cu)

# Common dependencies
set(SIGMA_LIBS
    gpu_mpc_fss
    gpu_mpc_utils
    gpu_mpc_nn
    gpu_mpc_external
)

# Link libraries and set properties
foreach(target sigma sigma_offline_online)
    target_link_libraries(${target} PRIVATE ${SIGMA_LIBS})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endforeach()