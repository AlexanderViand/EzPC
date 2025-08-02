# Build Options for GPU-MPC

# Core build options
option(GPU_MPC_BUILD_TESTS "Build test programs" ON)
option(GPU_MPC_BUILD_EXAMPLES "Build example programs" ON)

# Protocol-specific options
option(GPU_MPC_BUILD_ORCA "Build Orca secure training (requires SEAL)" OFF)
option(GPU_MPC_BUILD_SIGMA "Build SIGMA GPT inference" ON)
option(GPU_MPC_BUILD_PIRANHA "Build Piranha fast inference" ON)

# Data and utility options
option(GPU_MPC_DOWNLOAD_DATA "Download CIFAR-10 dataset" OFF)
option(GPU_MPC_BUILD_BENCHMARKS "Build benchmark programs" OFF)

# Advanced options
option(GPU_MPC_USE_SYSTEM_SYTORCH "Use system-installed Sytorch instead of FetchContent" OFF)
option(GPU_MPC_VERBOSE_BUILD "Enable verbose build output" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# Print configuration summary
message(STATUS "GPU-MPC Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${GPU_MPC_BUILD_TESTS}")
message(STATUS "  Build Examples: ${GPU_MPC_BUILD_EXAMPLES}")
message(STATUS "  Build Orca (Training): ${GPU_MPC_BUILD_ORCA}")
message(STATUS "  Build SIGMA (GPT): ${GPU_MPC_BUILD_SIGMA}")
message(STATUS "  Build Piranha: ${GPU_MPC_BUILD_PIRANHA}")
message(STATUS "  Download Datasets: ${GPU_MPC_DOWNLOAD_DATA}")

# Conditional messages
if(GPU_MPC_BUILD_ORCA)
    message(STATUS "")
    message(STATUS "NOTE: Orca build enabled - this will build SEAL and SCI float libraries")
    message(STATUS "      First build may take 20-30 minutes due to SEAL compilation")
endif()

if(NOT GPU_MPC_BUILD_ORCA)
    message(STATUS "")
    message(STATUS "NOTE: Orca build disabled - skipping SEAL and SCI float libraries")
    message(STATUS "      This significantly reduces build time")
endif()

# Validate options
if(GPU_MPC_BUILD_BENCHMARKS AND NOT GPU_MPC_BUILD_TESTS)
    message(WARNING "Benchmarks require tests to be built. Enabling GPU_MPC_BUILD_TESTS.")
    set(GPU_MPC_BUILD_TESTS ON CACHE BOOL "Build test programs" FORCE)
endif()