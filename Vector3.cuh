//
// Created by igor on 10.04.2021.
//

#ifndef CUDATEST_VECTOR3_CUH
#define CUDATEST_VECTOR3_CUH


class Vector3 {
public:
    double x, y, z;

    __device__ __host__ Vector3(double x, double y, double z);

    Vector3();

    __device__ __host__ Vector3& normalize();
    __device__ __host__ double norm() const;
    __device__ __host__ double dot(Vector3 other) const;
    __device__ __host__ double squared() const;


};

__device__ __host__ Vector3 operator *(double s, const Vector3& v);
__device__ __host__ Vector3 operator +(const Vector3& l, const Vector3& v);
__device__ __host__ Vector3 operator -(const Vector3& l, const Vector3& v);
__device__ __host__ Vector3 operator -(const Vector3& l);

#endif //CUDATEST_VECTOR3_CUH
