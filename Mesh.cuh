//
// Created by igor on 12.06.2021.
//

#ifndef CUDATEST_MESH_CUH
#define CUDATEST_MESH_CUH


#include <vector>
#include <string>
#include "Material.cuh"
#include "Vector3.cuh"
#include "Matrix4.cuh"

class Mesh {
public:
    std::string name;
    Material material;
    std::vector<unsigned int> indices;
    std::vector<Vector3> positions;
    Matrix4 transform;
};


#endif //CUDATEST_MESH_CUH
