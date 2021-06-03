//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_COLORF_CUH
#define CUDATEST_COLORF_CUH

struct ColorF {
    float r, g, b;
    __host__ __device__ ColorF operator += (const ColorF& r);
};

__host__ __device__ ColorF operator * (const ColorF& c, float f);

__host__ __device__ ColorF operator + (const ColorF& l, const ColorF& r);




#endif //CUDATEST_COLORF_CUH
