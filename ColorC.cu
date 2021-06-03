//
// Created by igor on 26.03.2021.
//

#include "ColorC.cuh"

__host__ __device__ ColorC operator* (const ColorC &c, float f) {
    return {(unsigned char)(c.r * f),
            (unsigned char)(c.g * f),
                    (unsigned char)(c.b * f)};
}

__host__ __device__ ColorC operator+ (const ColorC &l, const ColorC &r) {
    return {(unsigned char)(l.r + r.r),
            (unsigned char)(l.g + r.g),
            (unsigned char)(l.b + r.b)};
}

__host__ __device__ ColorC::ColorC(unsigned char r, unsigned char g, unsigned char b) : r(r), g(g), b(b) {}

__host__ __device__ ColorC::ColorC(ColorF f): r(f.r), g(f.g), b(f.b) {

}
