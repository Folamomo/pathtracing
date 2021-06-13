cmake_minimum_required(VERSION 3.17)
project(CUDATest)

#set(CMAKE_C_COMPILER /usr/bin/gcc-9)
#set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.3/bin/nvcc)

#set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/gcc-9")

set(CMAKE_CUDA_STANDARD 20)
enable_language(CXX)
enable_language(CUDA)

add_executable(CUDATest main.cu Screen.cu Screen.cuh ColorF.cu ColorF.cuh ColorC.cu ColorC.cuh Camera.cu Camera.cuh Matrix4.cu Matrix4.cuh Vector3.cu Vector3.cuh Renderer.cu Renderer.cuh Scene.cu Scene.cuh Triangle.cpp Triangle.h Material.cu Material.cuh loader.cuh Mesh.cu Mesh.cuh)

set_target_properties(
        CUDATest
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

