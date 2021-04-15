//
// Created by igor on 26.03.2021.
//

#include "Vector2.cuh"

__host__ __device__ Vector2 operator-(const Vector2 &l, const Vector2 &r) {
    return {l.x - r.x,
            l.y - r.y};
}
