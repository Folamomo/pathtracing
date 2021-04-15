//
// Created by igor on 10.04.2021.
//

#ifndef CUDATEST_SPHERE_CUH
#define CUDATEST_SPHERE_CUH


#include "Vector3.cuh"

class Sphere {
public:
    Sphere(const Vector3 &center, double radius);
    double radius;
    Vector3 center;
};


#endif //CUDATEST_SPHERE_CUH
