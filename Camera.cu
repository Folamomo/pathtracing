//
// Created by igor on 28.03.2021.
//

#include "Camera.cuh"


Camera::Camera(float fov, const unsigned int x, const unsigned int y) : fov(fov), x(x), y(y) {
    position = Matrix4::IDENTITY;
    origin = Vector3{0, 0, 0};
    float pixelDxLen = tan(fov/2)/x*2;
    pixelDx = {pixelDxLen, 0, 0};
    pixelDy = {0, -pixelDxLen, 0};
    Vector3 left = x / -2.0 * pixelDx;
    Vector3 top = y / -2.0 * pixelDy;
    topLeft = left + top + Vector3{0, 0, -1};
}
