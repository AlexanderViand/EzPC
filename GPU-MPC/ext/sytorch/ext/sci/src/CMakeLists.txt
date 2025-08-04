# set(CMAKE_FIND_DEBUG_MODE 1)
include_directories(${CMAKE_CURRENT_LIST_DIR})

# Use FetchContent for Eigen3
include(FetchContent)

find_package(Eigen3 3.3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen3 3.3 was not found: fetching from GitHub")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.3.9
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
    # Eigen3 doesn't create an alias target in older versions
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

add_subdirectory(utils)
add_subdirectory(OT)
add_subdirectory(GC)
add_subdirectory(Millionaire)
add_subdirectory(BuildingBlocks)
add_subdirectory(LinearOT)
if(SCI_BUILD_LINEAR_HE)
    # Configure SEAL before building LinearHE
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FetchSEAL.cmake)
    add_subdirectory(LinearHE) # Kanav: Removed for now
endif()
add_subdirectory(NonLinear)
add_subdirectory(Math)
if(SCI_BUILD_FLOAT_ML)
    add_subdirectory(FloatingPoint)
endif()

add_library(SCI-common INTERFACE)

find_package(Threads REQUIRED)

target_link_libraries(SCI-common
    INTERFACE
    SCI-NonLinear
    Threads::Threads
)

target_include_directories(SCI-common
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)

add_library(SCI-OT library_fixed_uniform.cpp library_fixed.cpp
    globals.cpp cleartext_library_fixed.cpp)

target_link_libraries(SCI-OT PUBLIC SCI-common SCI-LinearOT SCI-GC
	SCI-Math SCI-BuildingBlocks Eigen3::Eigen)
target_compile_definitions(SCI-OT PUBLIC SCI_OT=1)

if(SCI_BUILD_LINEAR_HE)
    add_library(SCI-HE library_fixed_uniform.cpp globals.cpp)
    target_link_libraries(SCI-HE PUBLIC SCI-common SCI-LinearHE)
    target_compile_definitions(SCI-HE PUBLIC SCI_HE=1)
endif()

if(SCI_BUILD_FLOAT_ML)
    add_library(SCI-FloatML library_float.cpp globals_float.cpp cleartext_library_float.cpp)
    target_link_libraries(SCI-FloatML PUBLIC SCI-FloatingPoint) 
    target_include_directories(SCI-FloatML PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ) # Kanav: HACK
endif()

# Core targets that are always built
set(SCI_INSTALL_TARGETS
    SCI-OT
    SCI-common
    SCI-BuildingBlocks
    SCI-LinearOT
    SCI-NonLinear
    SCI-Math
    SCI-Millionaire
    SCI-GC
    SCI-OTPrimitive
    SCI-utils
)

# Conditionally add targets based on build options
if(SCI_BUILD_LINEAR_HE)
    list(APPEND SCI_INSTALL_TARGETS SCI-HE SCI-LinearHE)
endif()

if(SCI_BUILD_FLOAT_ML)
    list(APPEND SCI_INSTALL_TARGETS SCI-FloatML SCI-FloatingPoint)
endif()

install(TARGETS
        ${SCI_INSTALL_TARGETS}
        EXPORT SCITargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(EXPORT SCITargets
        FILE SCITargets.cmake
        NAMESPACE SCI::
        DESTINATION lib/cmake/SCI)

install(DIRECTORY
        utils
        OT
        GC
        Millionaire
        NonLinear
        BuildingBlocks
        LinearOT
        LinearHE
        Math
        FloatingPoint
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

install(FILES
        "defines.h"
        "defines_uniform.h"
        "defines_float.h"
        "globals.h"
        "globals_float.h"
        "library_fixed.h"
        "library_fixed_uniform.h"
        "library_float.h"
        "cleartext_library_fixed.h"
        "cleartext_library_fixed_uniform.h"
        "cleartext_library_float.h"
        DESTINATION include)

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/cmake/FindGMP.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/cmake/source_of_randomness.cmake"
        DESTINATION lib/cmake/SCI)
