# Compiler Flags Configuration

# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-write-strings -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fpic")

# Enable AES instructions if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-maes" HAS_AES)
if(HAS_AES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
endif()

# Suppress specific warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations")

# Build type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Verbose build
if(GPU_MPC_VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# Color diagnostics - only for C++ compilation, not CUDA
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
endif()

# Create interface library for common flags
add_library(gpu_mpc_flags INTERFACE)
target_compile_features(gpu_mpc_flags INTERFACE cxx_std_17)
target_compile_options(gpu_mpc_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-write-strings>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wno-write-strings>
)