//
// Created by igor on 28.03.2021.
//

#include "Matrix4.cuh"

__host__ __device__ double *Matrix4::operator[](unsigned long y) {
    return data[y];
}

Matrix4::Matrix4(std::initializer_list<double> list) noexcept : data(){
    int i = 0;
    for(double d: list){
        data[0][i]=d;
        ++i;
    }
}

const Matrix4 Matrix4::IDENTITY = {1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1};

Matrix4 Matrix4::fromAxisAngle(Vector3 axis, double angle) {
    return {};
    //TODO implement
}

Matrix4 Matrix4::fromScaleFactor(double x, double y, double z) {
    return {x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1};
}

Matrix4 Matrix4::fromTranslationVector(const Vector3 &v) {
    return {1, 0, 0, v.x,
            0, 1, 0, v.y,
            0, 0, 1, v.z,
            0, 0, 0, 1};
}

Matrix4::Matrix4() :Matrix4{1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1}{}

__host__ __device__ Vector3 Matrix4::operator*(const Vector3 &other) {
    //TODO implement
    return other;
}

