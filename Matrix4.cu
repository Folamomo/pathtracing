//
// Created by igor on 28.03.2021.
//

#include "Matrix4.cuh"

__host__ __device__ double* Matrix4::operator[](unsigned long y) {
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

//https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
Matrix4 Matrix4::fromAxisAngle(Vector3 u, double t) {
    return {cos(t) + u.x*u.x * (1-cos(t)),          u.x*u.y * (1-cos(t)) - u.z * sin(t),    u.x*u.z * (1-cos(t)) + u.y * sin(t),    0,
            u.y*u.x * (1-cos(t)) + u.z * sin(t),    cos(t) + u.y*u.y * (1-cos(t)),          u.y*u.z * (1-cos(t)) - u.x * sin(t),    0,
            u.z*u.x * (1-cos(t)) - u.y * sin(t),    u.z*u.y * (1-cos(t)) + u.x * sin(t),    cos(t) + u.z*u.z * (1-cos(t)),          0,
            0,                                      0,                                      0,                                      1};
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

__host__ __device__ Matrix4::Matrix4() :data{0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0}{}

__host__ __device__ Vector3 Matrix4::operator*(const Vector3 &vector3) {
    float v[4] = {vector3.x, vector3.y, vector3.z, 1};
    Vector3 result{0, 0, 0};
    for (int i = 0; i < 4; ++i){
        result.x += data[0][i] * v[i];
        result.y += data[1][i] * v[i];
        result.z += data[2][i] * v[i];
    }
    return result;
}

__host__ __device__ Matrix4 Matrix4::operator*(Matrix4 other) {
    Matrix4 result;
    for(int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            for (int k = 0; k < 4; ++k){
                result[i][j] += data[i][k] * other[k][j];
            }
        }
    }
    return result;
}

