//
// Created by igor on 11.05.2021.
//

#ifndef CUDATEST_CONVMASK_CUH
#define CUDATEST_CONVMASK_CUH


#include <initializer_list>

class ConvMask {
public:
    double data[3][3];
    __device__ __host__ double*  operator[](std::size_t at);
    __device__ __host__ ConvMask(std::initializer_list<double> data) noexcept;
};


#endif //CUDATEST_CONVMASK_CUH
