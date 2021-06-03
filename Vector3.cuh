//
// Created by igor on 10.04.2021.
//

#ifndef CUDATEST_VECTOR3_CUH
#define CUDATEST_VECTOR3_CUH


class Vector3 {
public:
    float x, y, z;

    __device__ __host__ Vector3(float x, float y, float z);
    Vector3() = default;
    __device__ __host__ Vector3& normalize();
    __device__ __host__ float norm() const;
    __device__ __host__ float dot(Vector3 other) const;
    __device__ __host__ Vector3 cross(Vector3 other) const;
    __device__ __host__ float squared() const;


};

__device__ __host__ Vector3 operator *(float s, const Vector3& v);
__device__ __host__ Vector3 operator +(const Vector3& l, const Vector3& v);
__device__ __host__ Vector3 operator -(const Vector3& l, const Vector3& v);
__device__ __host__ Vector3 operator -(const Vector3& l);

#endif //CUDATEST_VECTOR3_CUH
