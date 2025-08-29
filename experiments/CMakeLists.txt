# Experiments for different protocols

if(GPU_MPC_BUILD_ORCA)
    add_subdirectory(orca)
endif()

if(GPU_MPC_BUILD_SIGMA)
    add_subdirectory(sigma)
endif()

if(GPU_MPC_BUILD_PIRANHA)
    # Piranha is part of Orca experiments
    # add_subdirectory(piranha)
endif()