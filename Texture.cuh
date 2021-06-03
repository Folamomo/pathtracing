//
// Created by igor on 03.06.2021.
//

#ifndef CUDATEST_TEXTURE_CUH
#define CUDATEST_TEXTURE_CUH


#include "ColorF.cuh"

class Texture {
public:
    ColorF* img;
    uint max;
    uint x, y;
    Texture(const char* path);
    ~Texture();
};


#endif //CUDATEST_TEXTURE_CUH
