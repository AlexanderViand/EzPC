# System Dependencies Check with Helpful Error Messages

# Compiler requirements
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        message(FATAL_ERROR 
            "GCC 9 or higher required. Current version: ${CMAKE_CXX_COMPILER_VERSION}\n"
            "Install with:\n"
            "  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test\n"
            "  sudo apt update\n"
            "  sudo apt install gcc-9 g++-9\n"
            "  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9\n"
            "  sudo update-alternatives --config gcc\n")
    endif()
endif()

# OpenSSL
find_package(OpenSSL)
if(NOT OpenSSL_FOUND)
    message(FATAL_ERROR 
        "OpenSSL not found.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install libssl-dev\n")
endif()
message(STATUS "Found OpenSSL: ${OPENSSL_VERSION}")

# Eigen3
find_package(Eigen3 3.3 NO_MODULE)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR 
        "Eigen3 (>= 3.3) not found.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install libeigen3-dev\n")
endif()
message(STATUS "Found Eigen3: ${Eigen3_VERSION}")

# OpenMP
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR 
        "OpenMP not found.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install libomp-dev\n")
endif()
message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")

# GMP
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY gmp)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR 
        "GMP not found.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install libgmp-dev\n")
endif()
message(STATUS "Found GMP: ${GMP_LIBRARY}")

# MPFR
find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIBRARY mpfr)
if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY)
    message(FATAL_ERROR 
        "MPFR not found.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install libmpfr-dev\n")
endif()
message(STATUS "Found MPFR: ${MPFR_LIBRARY}")

# Python3 (for dataset download scripts)
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    message(WARNING 
        "Python3 not found. Dataset download scripts will not work.\n"
        "Install with:\n"
        "  sudo apt update\n"
        "  sudo apt install python3 python3-pip\n")
else()
    message(STATUS "Found Python3: ${Python3_VERSION}")
endif()

# Check for matplotlib if data download is enabled
if(GPU_MPC_DOWNLOAD_DATA AND Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
        RESULT_VARIABLE MATPLOTLIB_CHECK
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT MATPLOTLIB_CHECK EQUAL 0)
        message(WARNING 
            "Python matplotlib not found. Plotting scripts will not work.\n"
            "Install with:\n"
            "  pip3 install matplotlib\n")
    endif()
endif()

# Create interface library for system dependencies
add_library(gpu_mpc_system_deps INTERFACE)
target_link_libraries(gpu_mpc_system_deps INTERFACE
    OpenSSL::SSL
    OpenSSL::Crypto
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
target_include_directories(gpu_mpc_system_deps INTERFACE
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
)