//
// Created by igor on 17.05.2021.
//

#ifndef CUDATEST_RENDERER_CUH
#define CUDATEST_RENDERER_CUH


#include <vector>
#include <curand_kernel.h>
#include "Screen.cuh"
#include "Camera.cuh"
#include "Scene.cuh"

class Renderer {
public:
    Renderer();

    ~Renderer();

    void render();

    void uploadScene(Scene& scene);
    void uploadCamera(Camera& camera);
    void uploadScreen(Screen& screen);


    struct RendererConstants {
        unsigned int sizeX, sizeY;
        Vector3 topLeft;
        Vector3 pixelDx, pixelDy;
        Vector3 origin;
        Vector3* vertices;
        Material* materials;
        unsigned int* indices;
        unsigned int* material_index;
        ColorF* d_image;
        unsigned int n_triangles;
        curandState *dev_random;
    };

private:
    RendererConstants constants;
};



#endif //CUDATEST_RENDERER_CUH
