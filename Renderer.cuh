//
// Created by igor on 17.05.2021.
//

#ifndef CUDATEST_RENDERER_CUH
#define CUDATEST_RENDERER_CUH


#include <vector>
#include "Screen.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Scene.cuh"

class Renderer {
public:
    Screen screen;
    Scene scene;
    Camera camera;

    void render();
    void uploadScene();

    Renderer(Screen &&screen, Scene &&scene, Camera &&camera);

    Vector3* d_vectors;
    Material* d_materials;
    Triangle* d_triangles;
    ColorF* d_skybox_tex;
};


#endif //CUDATEST_RENDERER_CUH
