find_package(OpenMP REQUIRED)

# SEAL should already be configured by parent CMakeLists.txt

add_library(SCI-LinearHE
    conv-field.cpp
    fc-field.cpp
    elemwise-prod-field.cpp
    utils-HE.cpp
)

target_link_libraries(SCI-LinearHE
    PUBLIC
    SCI-utils
    seal
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)
