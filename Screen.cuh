//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_SCREEN_CUH
#define CUDATEST_SCREEN_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Color.cuh"
#include "Vector2.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define BLOCK_X 32
#define BLOCK_Y 16

class Screen {
public:
    unsigned int sizeX, sizeY;
    Color* image;
    Color* d_image;

    Screen(unsigned int x, unsigned int y);

    Screen(const Screen&) = delete;
    Screen& operator=(const Screen&) = delete;

    ~Screen();

    template<typename F, typename... A>
    auto cudaExecute(F f, A... a) {
        f<<<dim3(sizeX/BLOCK_X + 1, sizeY/BLOCK_Y + 1), dim3(BLOCK_X, BLOCK_Y)>>>(*this, a...);
    }

    //copies memory from/to device
    void copy(cudaMemcpyKind dir);

    //Outputs content of image to .ppm file denoted by path
    void save(const char* path) const;

    class ScreenRef {
    public:
        unsigned int sizeX, sizeY;
        Color* image;
        Color* d_image;

        ScreenRef(Screen& s): sizeX(s.sizeX), sizeY(s.sizeY), image(s.image), d_image(s.d_image){}
    };

    ScreenRef makeRef();
};



#endif //CUDATEST_SCREEN_CUH
