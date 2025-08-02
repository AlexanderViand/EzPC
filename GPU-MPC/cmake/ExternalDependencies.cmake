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

# Sytorch configuration based on build options
set(BUILD_NETWORKS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_ALL_TESTS OFF CACHE BOOL "" FORCE)

if(GPU_MPC_BUILD_ORCA)
    # These will trigger SEAL build
    set(SCI_BUILD_FLOAT_ML ON CACHE BOOL "" FORCE)
    set(SCI_BUILD_LINEAR_HE ON CACHE BOOL "" FORCE)
else()
    # Skip SEAL and float libraries
    set(SCI_BUILD_FLOAT_ML OFF CACHE BOOL "" FORCE)
    set(SCI_BUILD_LINEAR_HE OFF CACHE BOOL "" FORCE)
endif()

# Sytorch - Core MPC library
if(NOT GPU_MPC_USE_SYSTEM_SYTORCH)
    message(STATUS "Configuring Sytorch...")
    
    # Sytorch-GPU is the GPU-enhanced version of Sytorch
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/sytorch/CMakeLists.txt")
        message(FATAL_ERROR 
            "Sytorch not found in ext/sytorch/.\n"
            "This should be part of the GPU-MPC repository.")
    endif()
    
    add_subdirectory(ext/sytorch)
else()
    # Use system-installed Sytorch
    find_package(sytorch REQUIRED)
endif()

# Create a unified interface library for all external dependencies
add_library(gpu_mpc_external INTERFACE)
target_link_libraries(gpu_mpc_external INTERFACE
    cutlass
    sytorch
    cryptoTools
    LLAMA
    bitpack
)

# Add SCI float libraries only if Orca is enabled
if(GPU_MPC_BUILD_ORCA)
    if(TARGET SCI-FloatML)
        target_link_libraries(gpu_mpc_external INTERFACE
            SCI-FloatML
            SCI-FloatingPoint
        )
    endif()
endif()

# Always link these core SCI libraries
target_link_libraries(gpu_mpc_external INTERFACE
    SCI-BuildingBlocks
    SCI-LinearOT
    SCI-GC
)

# Status message
message(STATUS "External dependencies configured:")
message(STATUS "  CUTLASS: Headers from ${cutlass_SOURCE_DIR}")
if(GPU_MPC_USE_SYSTEM_SYTORCH)
    message(STATUS "  Sytorch: System")
else()
    message(STATUS "  Sytorch: Built from source")
endif()
if(GPU_MPC_BUILD_ORCA)
    message(STATUS "  SEAL: Will be built with SCI")
endif()