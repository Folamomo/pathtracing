//
// Created by igor on 23.05.2021.
//

#ifndef CUDATEST_SCENE_CUH
#define CUDATEST_SCENE_CUH

#include <vector>
#include "Vector3.cuh"
#include "Triangle.h"
#include "Material.cuh"
#include "Mesh.cuh"


class Scene {
public:
//    Scene(const char *skybox);

    std::vector<Mesh> meshes;

    void loadObj(const char* path);
//    Texture skybox;
};


#endif //CUDATEST_SCENE_CUH
