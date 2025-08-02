# FetchContent configuration for SEAL

include(FetchContent)

# Only fetch SEAL if LinearHE is being built
if(SCI_BUILD_LINEAR_HE)
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
    
    FetchContent_MakeAvailable(seal)
    
    # Apply patch if needed
    if(EXISTS "${PROJECT_SOURCE_DIR}/cmake/seal.patch")
        execute_process(
            COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${seal_SOURCE_DIR}"
            RESULT_VARIABLE patch_result
        )
        if(NOT patch_result EQUAL 0)
            message(WARNING "Failed to apply SEAL patch, continuing anyway...")
        endif()
    endif()
endif()