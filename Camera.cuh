//
// Created by igor on 28.03.2021.
//

#ifndef CUDATEST_CAMERA_CUH
#define CUDATEST_CAMERA_CUH


#include "Matrix4.cuh"
#include "Screen.cuh"

class Camera {
public:
    Matrix4 position;
    double fov;
    const unsigned int x, y;
    Vector3 topLeft;
    Vector3 pixelDx, pixelDy;


};


#endif //CUDATEST_CAMERA_CUH
