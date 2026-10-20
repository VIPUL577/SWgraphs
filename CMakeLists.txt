# cmake_minimum_required(VERSION 3.25)

# project(SWGR LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_ARCHITECTURES 89)

# add_executable(graph include/Essentials/main/graph.cu)


# target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/external/SlabHash/src)
# target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/external/SlabHash/SlabAlloc/src)
# target_include_directories(graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../thrust/)

# cmake_minimum_required(VERSION 3.25)

cmake_minimum_required(VERSION 3.25)

project(SWGR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/SlabHash/src
    ${CMAKE_SOURCE_DIR}/external/SlabHash/SlabAlloc/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../thrust/
)

# add_executable(graph include/Essentials/main/graph.cu)
# target_include_directories(graph PRIVATE ${COMMON_INCLUDES})

add_executable(test_sssp tests/test.cu)
target_include_directories(test_sssp PRIVATE ${COMMON_INCLUDES})