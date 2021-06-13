//
// Created by igor on 28.03.2021.
//

#ifndef CUDATEST_MATRIX4_CUH
#define CUDATEST_MATRIX4_CUH


#include <initializer_list>
#include "Vector3.cuh"

class Matrix4 {
public:
    double data [4][4];
    Matrix4(std::initializer_list<double> data) noexcept;

    Matrix4();

    __host__ __device__ double* operator[](unsigned long y);
    __host__ __device__ Vector3 operator * (const Vector3& other);

    static Matrix4 fromAxisAngle(Vector3 axis, double angle);
    static Matrix4 fromScaleFactor(double x, double y, double z);
    static Matrix4 fromTranslationVector(const Vector3& v);
    static const Matrix4 IDENTITY;
};




#endif //CUDATEST_MATRIX4_CUH
