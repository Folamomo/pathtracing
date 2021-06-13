//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_SCREEN_CUH
#define CUDATEST_SCREEN_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ColorF.cuh"
#include "ColorC.cuh"
#include "Vector3.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <deque>

#define BLOCK_X 16
#define BLOCK_Y 16

class Screen {
private:
    class ScreenImp{
    public:
        unsigned int sizeX, sizeY;
        ColorC* image;
        ColorF* d_image;
        ColorC* d_imageC;

        ScreenImp(unsigned int sizeX, unsigned int sizeY);
        ScreenImp(const ScreenImp& screenImp) = delete;
        ~ScreenImp();
    };
    static std::deque<Screen::ScreenImp> implementations;

public:
    unsigned int sizeX, sizeY;
    ColorC* image;
    ColorF* d_image;
    ColorC* d_imageC;

    Screen(unsigned int x, unsigned int y);


    __device__ ColorF* operator[](size_t row);

    //copies memory from/to device
    void copy(cudaMemcpyKind dir);

    void cast();

    //Outputs contents of d_image to .ppm
    void copyAndSave(const char *path);

    //Outputs content of image to .ppm file denoted by path
    void save(const char* path) const;

    //frees resources if needed
    static void removeAll();
};




#endif //CUDATEST_SCREEN_CUH
