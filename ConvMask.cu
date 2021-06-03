//
// Created by igor on 11.05.2021.
//

#include "ConvMask.cuh"

__device__ __host__ double* ConvMask::operator[](std::size_t at){
    return data[at];
}
__device__ __host__ ConvMask::ConvMask(std::initializer_list<double> list) noexcept : data() {
    int i = 0;
    double sum = 0;
    for(double d: list){
        data[0][i] = d;
        sum += d;
        ++i;
    }

    for (int j = 0; j < 9; ++j){
        data[0][j] /= sum;
    }
}

