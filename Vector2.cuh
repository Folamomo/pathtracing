//
// Created by igor on 26.03.2021.
//

#ifndef CUDATEST_VECTOR2_CUH
#define CUDATEST_VECTOR2_CUH

struct Vector2 {float x, y;};

__host__ __device__ Vector2 operator - (const Vector2& l, const Vector2& r);
#endif //CUDATEST_VECTOR2_CUH
