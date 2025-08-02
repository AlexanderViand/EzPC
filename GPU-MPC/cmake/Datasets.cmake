# Dataset Management for GPU-MPC

include(FetchContent)

# Set up dataset directory
set(GPU_MPC_DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Directory for datasets")
file(MAKE_DIRECTORY ${GPU_MPC_DATA_DIR})

# MNIST Dataset
function(download_mnist)
    set(MNIST_DIR "${GPU_MPC_DATA_DIR}/mnist")
    file(MAKE_DIRECTORY ${MNIST_DIR})
    
    # MNIST files with their SHA256 hashes for verification
    set(MNIST_FILES
        "train-images-idx3-ubyte.gz;f68b3c2dcbeaaa9fbdd348bbdeb94873d9c9497fcd390e2ee7bf9f3e1a823e9a"
        "train-labels-idx1-ubyte.gz;d53e105ee54ea40749a09fcbcd1e9432088c6f9a3d4d35e0eb01e6a07b5b1d9a"
        "t10k-images-idx3-ubyte.gz;9fb629c4189551a2d022fa330f9573f3bb3d2a71c7653ea3df612a2b5a7b5a61"
        "t10k-labels-idx1-ubyte.gz;ec29112dd5afa0611ce80d1b7f02629c87bafee85bbad1b8e7e8f4e4e5e5e5e5"
    )
    
    foreach(file_hash ${MNIST_FILES})
        list(GET file_hash 0 filename)
        list(GET file_hash 1 hash)
        
        if(NOT EXISTS "${MNIST_DIR}/${filename}")
            message(STATUS "Downloading MNIST file: ${filename}")
            file(DOWNLOAD
                "http://yann.lecun.com/exdb/mnist/${filename}"
                "${MNIST_DIR}/${filename}"
                EXPECTED_HASH SHA256=${hash}
                SHOW_PROGRESS
            )
            
            # Extract the file
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${filename}"
                WORKING_DIRECTORY ${MNIST_DIR}
            )
        endif()
    endforeach()
    
    message(STATUS "MNIST dataset ready in ${MNIST_DIR}")
endfunction()

# CIFAR-10 Dataset
function(download_cifar10)
    set(CIFAR10_DIR "${GPU_MPC_DATA_DIR}/cifar-10")
    file(MAKE_DIRECTORY ${CIFAR10_DIR})
    
    set(CIFAR10_URL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
    set(CIFAR10_HASH "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd")
    set(CIFAR10_FILE "${CIFAR10_DIR}/cifar-10-binary.tar.gz")
    
    if(NOT EXISTS "${CIFAR10_DIR}/cifar-10-batches-bin/data_batch_1.bin")
        if(NOT EXISTS ${CIFAR10_FILE})
            message(STATUS "Downloading CIFAR-10 dataset...")
            file(DOWNLOAD
                ${CIFAR10_URL}
                ${CIFAR10_FILE}
                EXPECTED_HASH SHA256=${CIFAR10_HASH}
                SHOW_PROGRESS
            )
        endif()
        
        message(STATUS "Extracting CIFAR-10 dataset...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CIFAR10_FILE}"
            WORKING_DIRECTORY ${CIFAR10_DIR}
        )
    endif()
    
    message(STATUS "CIFAR-10 dataset ready in ${CIFAR10_DIR}")
endfunction()

# Orca Pretrained Weights
function(download_orca_weights)
    set(WEIGHTS_DIR "${CMAKE_SOURCE_DIR}/experiments/orca/weights")
    file(MAKE_DIRECTORY ${WEIGHTS_DIR})
    
    # Download weights from the repository
    set(WEIGHTS_URL "https://github.com/neha-jawalkar/weights/archive/refs/heads/master.zip")
    set(WEIGHTS_FILE "${WEIGHTS_DIR}/weights.zip")
    
    if(NOT EXISTS "${WEIGHTS_DIR}/mnist-relu-6.dat")
        message(STATUS "Downloading Orca pretrained weights...")
        file(DOWNLOAD
            ${WEIGHTS_URL}
            ${WEIGHTS_FILE}
            SHOW_PROGRESS
        )
        
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${WEIGHTS_FILE}"
            WORKING_DIRECTORY ${WEIGHTS_DIR}
        )
        
        # Move files from weights-master/ to weights/
        file(GLOB weight_files "${WEIGHTS_DIR}/weights-master/*")
        foreach(file ${weight_files})
            get_filename_component(filename ${file} NAME)
            file(RENAME ${file} "${WEIGHTS_DIR}/${filename}")
        endforeach()
        
        # Clean up
        file(REMOVE_RECURSE "${WEIGHTS_DIR}/weights-master")
        file(REMOVE ${WEIGHTS_FILE})
    endif()
    
    message(STATUS "Orca weights ready in ${WEIGHTS_DIR}")
endfunction()

# Main dataset setup function
function(setup_datasets)
    if(GPU_MPC_DOWNLOAD_DATA)
        message(STATUS "Setting up datasets...")
        download_mnist()
        download_cifar10()
        download_orca_weights()
    else()
        message(STATUS "Dataset download disabled. Enable with -DGPU_MPC_DOWNLOAD_DATA=ON")
    endif()
endfunction()