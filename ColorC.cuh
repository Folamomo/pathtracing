//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_COLOR_CUH
#define CUDATEST_COLOR_CUH


#include "ColorF.cuh"

class ColorC {
public:
    unsigned char r, g, b;

    __host__ __device__ ColorC(unsigned char r, unsigned char g, unsigned char b);
    __host__ __device__ ColorC(ColorF f);

};

__host__ __device__ ColorC operator * (const ColorC& c, float f);

__host__ __device__ ColorC operator + (const ColorC& l, const ColorC& r);


#endif //CUDATEST_COLOR_CUH
