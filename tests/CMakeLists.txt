# Test programs

# FSS tests
add_executable(dpf fss/dpf.cu)
add_executable(truncate fss/truncate.cu)
# Commented out - depends on sytorch ClearText
# add_executable(rotary_embedding fss/rotary_embedding.cu)
add_executable(lss_test fss/lss_test.cu)

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
    gpu_mpc_external
)

# Link all FSS tests
foreach(test_target dpf truncate lss_test)
    target_link_libraries(${test_target} PRIVATE ${TEST_LIBS})
    set_target_properties(${test_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# Link all DCF tests
foreach(test_target dcf dcf_relu dcf_maxpool aes relu_extend stochastic_truncate)
    target_link_libraries(${test_target} PRIVATE ${TEST_LIBS})
    set_target_properties(${test_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()