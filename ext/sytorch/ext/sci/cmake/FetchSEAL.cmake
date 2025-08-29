# FetchContent configuration for SEAL

include(FetchContent)

message(STATUS "Fetching SEAL 3.3.2 for LinearHE...")

FetchContent_Declare(
    seal
    GIT_REPOSITORY https://github.com/microsoft/SEAL.git
    GIT_TAG v3.3.2
    GIT_SHALLOW TRUE
)

# SEAL configuration options
set(SEAL_USE_CXX17 OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# Disable msgsl to avoid compatibility issues with modern GSL
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL_SPAN OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL_MULTISPAN OFF CACHE BOOL "" FORCE)

# Use FetchContent_Populate to download first, then apply patch, then configure
FetchContent_GetProperties(seal)
if(NOT seal_POPULATED)
    FetchContent_Populate(seal)
    
    # Apply patch after fetching but before configuring
    if(EXISTS "${PROJECT_SOURCE_DIR}/cmake/seal.patch")
        message(STATUS "Applying SEAL patch...")
        execute_process(
            COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${seal_SOURCE_DIR}"
            RESULT_VARIABLE patch_result
            ERROR_VARIABLE patch_error
        )
        if(NOT patch_result EQUAL 0)
            message(WARNING "Failed to apply SEAL patch: ${patch_error}")
            # Try to reset and apply with different options
            execute_process(
                COMMAND git checkout -- .
                WORKING_DIRECTORY "${seal_SOURCE_DIR}"
            )
        endif()
    endif()
    
    # Now add SEAL's CMake project (CMakeLists.txt is in native/src)
    add_subdirectory(${seal_SOURCE_DIR}/native/src ${seal_BINARY_DIR})
endif()

# Create alias if it doesn't exist (SEAL 3.3.2 creates seal target, not SEAL::seal)
if(TARGET seal AND NOT TARGET SEAL::seal)
    add_library(SEAL::seal ALIAS seal)
endif()