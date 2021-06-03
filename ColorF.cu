//
// Created by igor on 26.03.2021.
//

#include "ColorF.cuh"

__host__ __device__ ColorF operator* (const ColorF &c, float f) {
    return {c.r * f,
            c.g * f,
            c.b * f};
}

__host__ __device__ ColorF operator+ (const ColorF &l, const ColorF &r) {
    return {l.r + r.r,
            l.g + r.g,
            l.b + r.b};
}

__host__ __device__ ColorF ColorF::operator+=(const ColorF &o) {
    r += o.r;
    g += o.g;
    b += o.b;
    return *this;
}
