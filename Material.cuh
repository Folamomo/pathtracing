//
// Created by igor on 23.05.2021.
//

#ifndef CUDATEST_MATERIAL_CUH
#define CUDATEST_MATERIAL_CUH


#include "ColorC.cuh"

class Material {
public:
    ColorF diffuse_color;
    ColorF emit_color;
    float emit;
    ColorF specular_color;
    float specular;
    ColorF transparent_color;
    float transparent;
};


#endif //CUDATEST_MATERIAL_CUH
