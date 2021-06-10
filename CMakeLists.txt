cmake_minimum_required(VERSION 3.17)
project(CUDATest CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/gcc-9")

set(CMAKE_CUDA_STANDARD 20)


add_executable(CUDATest main.cu Screen.cu Screen.cuh Vector2.cu Vector2.cuh ColorF.cu ColorF.cuh ColorC.cu ColorC.cuh Camera.cu Camera.cuh Matrix4.cu Matrix4.cuh Vector3.cu Vector3.cuh Sphere.cu Sphere.cuh ConvMask.cu ConvMask.cuh Renderer.cu Renderer.cuh Scene.cu Scene.cuh Triangle.cpp Triangle.h Material.cu Material.cuh Texture.cu Texture.cuh loader.cuh)

set_target_properties(
        CUDATest
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

