# External Dependencies via FetchContent

include(FetchContent)

# Set download and build options
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# CUTLASS - Header-only, no build needed
message(STATUS "Configuring CUTLASS (headers only)...")
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v2.11.0
    GIT_SHALLOW TRUE
)

# Make CUTLASS available but skip its build
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
    
    # Create interface library for CUTLASS headers
    add_library(cutlass INTERFACE)
    target_include_directories(cutlass INTERFACE 
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    
    # Add CUTLASS definitions
    target_compile_definitions(cutlass INTERFACE
        CUTLASS_NAMESPACE=cutlass
    )
endif()


# bitpack - Bit packing library
message(STATUS "Configuring bitpack...")
file(GLOB BITPACK_SOURCES
    ${CMAKE_SOURCE_DIR}/ext/bitpack/src/bitpack/*.cpp
)
add_library(bitpack STATIC ${BITPACK_SOURCES})
target_include_directories(bitpack
    PUBLIC
        ${CMAKE_SOURCE_DIR}/ext/bitpack/include
)

# cryptoTools - Cryptographic utilities
message(STATUS "Configuring cryptoTools...")
file(GLOB CRYPTOTOOLS_SOURCES
    ${CMAKE_SOURCE_DIR}/ext/cryptoTools/cryptoTools/Common/*.cpp
    ${CMAKE_SOURCE_DIR}/ext/cryptoTools/cryptoTools/Crypto/*.cpp
)
add_library(cryptoTools STATIC ${CRYPTOTOOLS_SOURCES})
target_include_directories(cryptoTools
    PUBLIC
        ${CMAKE_SOURCE_DIR}/ext/cryptoTools
        ${CMAKE_SOURCE_DIR}/ext/cryptoTools/cryptoTools
)
target_compile_features(cryptoTools PUBLIC cxx_std_17)

# LLAMA - MPC library
message(STATUS "Configuring LLAMA...")

# Build LLAMA library from sources
file(GLOB LLAMA_SOURCES 
    ${CMAKE_SOURCE_DIR}/ext/llama/src/llama/*.cpp
)

add_library(llama STATIC ${LLAMA_SOURCES})
target_include_directories(llama 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/ext/llama/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/ext/llama
)

# LLAMA needs OpenMP, cryptoTools, bitpack, Eigen and other system libraries
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(llama 
    PUBLIC
        cryptoTools
        bitpack
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${CMAKE_DL_LIBS}
        pthread
)

# Create combined external dependencies target
add_library(gpu_mpc_external INTERFACE)
target_link_libraries(gpu_mpc_external INTERFACE
    cutlass
    llama
)

# Status message
message(STATUS "External dependencies configured:")
message(STATUS "  CUTLASS: Headers from ${cutlass_SOURCE_DIR}")
message(STATUS "  bitpack: From ${CMAKE_SOURCE_DIR}/ext/bitpack")
message(STATUS "  cryptoTools: From ${CMAKE_SOURCE_DIR}/ext/cryptoTools")
message(STATUS "  LLAMA: From ${CMAKE_SOURCE_DIR}/ext/llama")