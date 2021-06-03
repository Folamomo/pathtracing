//
// Created by igor on 10.04.2021.
//

#ifndef CUDATEST_SPHERE_CUH
#define CUDATEST_SPHERE_CUH


#include "Vector3.cuh"
#include "ColorF.cuh"

class Sphere {
public:
    Sphere(const Vector3 &center, double radius, const ColorF &color);
    double radius;
    Vector3 center;
    ColorF color;
};


#endif //CUDATEST_SPHERE_CUH
