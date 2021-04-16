//
// Created by igor on 10.04.2021.
//

#include "Vector3.cuh"

__device__ __host__ Vector3::Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

__device__ __host__ Vector3& Vector3::normalize() {
    double norm = this->norm();
    x /= norm;
    y /= norm;
    z /= norm;
    return *this;
}

__device__ __host__ double Vector3::norm() const{
    return sqrt(x*x+y*y+z*z);
}

__device__ __host__ double Vector3::dot(const Vector3 r) const {
    return x * r.x + y * r.y + z * r.z;
}

__device__ __host__ double Vector3::squared() const {
    return x*x+y*y+z*z;
}

Vector3::Vector3(): x(0), y(0), z(0){}

__device__ __host__ Vector3 operator*(double s, const Vector3& v) {
    return {s*v.x, s*v.y, s*v.z};
}

__device__ __host__ Vector3 operator+(const Vector3 &l, const Vector3 &r) {
    return {l.x + r.x, l.y + r.y, l.z + r.z};
}

__device__ __host__ Vector3 operator-(const Vector3 &l, const Vector3 &r) {
    return {l.x - r.x, l.y - r.y, l.z - r.z};
}
