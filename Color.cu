//
// Created by igor on 26.03.2021.
//

#include "Color.cuh"

__host__ __device__ Color operator*(const Color &c, float f) {
    return {(unsigned char)(c.r * f),
            (unsigned char)(c.g * f),
            (unsigned char)(c.b * f)};
}

__host__ __device__ Color operator+(const Color &l, const Color &r) {
    return {(unsigned char)(l.r + r.r),
            (unsigned char)(l.g + r.g),
            (unsigned char)(l.b + r.b)};
}
