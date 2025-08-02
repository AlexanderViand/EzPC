# Test programs

# FSS tests
add_executable(dpf fss/dpf.cu)
add_executable(relu fss/relu.cu)
add_executable(gelu fss/gelu.cu)
add_executable(softmax fss/softmax.cu)
add_executable(truncate fss/truncate.cu)
add_executable(mha fss/mha.cu)
add_executable(layernorm fss/layernorm.cu)
add_executable(silu fss/silu.cu)
add_executable(rmsnorm fss/rmsnorm.cu)
add_executable(rotary_embedding fss/rotary_embedding.cu)
add_executable(secfloat_softmax fss/secfloat_softmax.cu)
add_executable(piranha_softmax fss/piranha_softmax.cu)

# DCF tests
add_executable(dcf fss/dcf/dcf.cu)
add_executable(dcf_relu fss/dcf/relu.cu)
add_executable(dcf_maxpool fss/dcf/maxpool.cu)
add_executable(aes fss/dcf/aes.cu)
add_executable(relu_extend fss/dcf/relu_extend.cu)
add_executable(stochastic_truncate fss/dcf/stochastic_truncate.cu)

# Common link libraries for all tests
set(TEST_LIBS
    gpu_mpc_fss
    gpu_mpc_utils
    gpu_mpc_nn
    gpu_mpc_external
)

# Link all FSS tests
foreach(test_target dpf relu gelu softmax truncate mha layernorm silu rmsnorm 
                   rotary_embedding secfloat_softmax piranha_softmax)
    target_link_libraries(${test_target} PRIVATE ${TEST_LIBS})
    set_target_properties(${test_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# Link all DCF tests
foreach(test_target dcf dcf_relu dcf_maxpool aes relu_extend stochastic_truncate)
    target_link_libraries(${test_target} PRIVATE ${TEST_LIBS})
    set_target_properties(${test_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()