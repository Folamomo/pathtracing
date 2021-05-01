//
// Created by igor on 10.04.2021.
//

#ifndef CUDATEST_SPHERE_CUH
#define CUDATEST_SPHERE_CUH


#include "Vector3.cuh"
#include "Color.cuh"

class Sphere {
public:
    Sphere(const Vector3 &center, double radius, const Color &color);
    double radius;
    Vector3 center;
    Color color;
};


#endif //CUDATEST_SPHERE_CUH
