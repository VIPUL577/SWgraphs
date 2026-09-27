cmake_minimum_required(VERSION 3.25)

project(SWGR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(graph include/Essentials/main/graph.cu)


target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/SlabHash/src)
target_include_directories(graph PRIVATE ${CMAKE_SOURCE_DIR}/SlabHash/SlabAlloc/src)
target_include_directories(graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../thrust/)

