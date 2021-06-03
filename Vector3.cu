//
// Created by igor on 10.04.2021.
//

#include "Vector3.cuh"

__device__ __host__ Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

__device__ __host__ Vector3& Vector3::normalize() {
    float norm = this->norm();
    x /= norm;
    y /= norm;
    z /= norm;
    return *this;
}

__device__ __host__ float Vector3::norm() const{
    return sqrt(x*x+y*y+z*z);
}

__device__ __host__ float Vector3::dot(const Vector3 r) const {
    return x * r.x + y * r.y + z * r.z;
}

__device__ __host__ float Vector3::squared() const {
    return x*x+y*y+z*z;
}

__device__ __host__ Vector3 Vector3::cross(Vector3 b) const {
    const Vector3& a = *this;
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ __host__ Vector3 operator*(float s, const Vector3& v) {
    return {s*v.x, s*v.y, s*v.z};
}

__device__ __host__ Vector3 operator+(const Vector3 &l, const Vector3 &r) {
    return {l.x + r.x, l.y + r.y, l.z + r.z};
}

__device__ __host__ Vector3 operator-(const Vector3 &l, const Vector3 &r) {
    return {l.x - r.x, l.y - r.y, l.z - r.z};
}

__device__ __host__ Vector3 operator-(const Vector3 &l) {
    return -1 * l;
}
