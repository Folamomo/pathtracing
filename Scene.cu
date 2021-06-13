//
// Created by igor on 23.05.2021.
//

#include "Scene.cuh"
#include "loader.cuh"

//Scene::Scene(const char *skybox) : skybox(skybox), triangles(16), vertices(16), materials(16) {}

Mesh convertMesh(const objl::Mesh& from){
    Mesh result;
    result.name = from.MeshName;
    result.indices = std::vector<unsigned int>(from.Indices);
    result.positions = std::vector<Vector3>();
    for (const auto& v : from.Vertices){
        result.positions.emplace_back(v.Position.x, v.Position.y, v.Position.z);
    }
    return result;
}

void Scene::loadObj(const char *path) {
    objl::Loader Loader;
    // Load .obj File
    if (Loader.LoadFile(path)) {
        for(const objl::Mesh& mesh: Loader.LoadedMeshes){
            meshes.push_back(convertMesh(mesh));
        }
    }


}
