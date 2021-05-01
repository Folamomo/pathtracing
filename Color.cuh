//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_COLOR_CUH
#define CUDATEST_COLOR_CUH



struct Color {unsigned char r, g, b;};

__host__ __device__ Color operator * (const Color& c, float f);

__host__ __device__ Color operator + (const Color& l, const Color& r);


#endif //CUDATEST_COLOR_CUH
