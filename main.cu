#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ColorF.cuh"
#include "Screen.cuh"
#include "Matrix4.cuh"
#include "Camera.cuh"
#include "Renderer.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "loader.cuh"

using namespace std;


#define prune(number) ( number = number < 0 ? 0 : number > 1 ? 1 : number )



int main(){

    Renderer renderer{{1920, 1080},
                      {},
                      {120.0/180.0 * 3.1415, 1920, 1080}};

    objl::Loader Loader;
    std::ofstream file("out.txt");
    // Load .obj File
    bool loadout = Loader.LoadFile("boxes.obj");
    if (loadout) {

        // Material
        renderer.scene.materials.push_back({{}, {255, 0, 0}, 1, {255,255,255}, 0});
        renderer.scene.materials.push_back({{}, {0, 255, 0}, 0.5, {255,255,255}, 0.5});
        renderer.scene.materials.push_back({{}, {255,105,180}, 0.5, {255,255,255}, 0.5});
        // Go through each loaded mesh and out its contents
        unsigned int offset = 0;
        for (int i = 0; i < Loader.LoadedMeshes.size(); i++) {

            objl::Mesh curMesh = Loader.LoadedMeshes[i];

            // Vertices
            for (int j = 0; j < curMesh.Vertices.size(); j++) {
                renderer.scene.vertices.push_back({curMesh.Vertices[j].Position.X, curMesh.Vertices[j].Position.Y, curMesh.Vertices[j].Position.Z});
            }

            // Faces
            // iterate over indices, 3 indices create triangle face
            for (int j = 0; j < curMesh.Indices.size(); j += 3)
            {
                int a = curMesh.Indices[j] + offset;
                int b = curMesh.Indices[j + 1] + offset;
                int c = curMesh.Indices[j + 2] + offset;
                renderer.scene.triangles.push_back({static_cast<unsigned short>(a), static_cast<unsigned short>(b), static_cast<unsigned short>(c), 0,0,0, static_cast<unsigned short>(i)});
            }
            offset += curMesh.Vertices.size();
        }
    }


//    renderer.scene.vertices.push_back({0.23, 2.08, 0.23-5});
//    renderer.scene.vertices.push_back({0.23, 2.55, 0.23-5});
//    renderer.scene.vertices.push_back({-0.23, 2.08, 0.23-5});
//    renderer.scene.vertices.push_back({-0.23, 2.55, 0.23-5});
//    renderer.scene.vertices.push_back({0.23, 2.08, -0.23-5});
//    renderer.scene.vertices.push_back({0.23, 2.55, -0.23-5});
//    renderer.scene.vertices.push_back({-0.23, 2.08, -0.23-5});
//    renderer.scene.vertices.push_back({-0.23, 2.55, -0.23-5});
//    renderer.scene.materials.push_back({{}, {255, 0, 0}, 0.9, {}, 0.1});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
//    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});

    renderer.render();
//    ConvMask c{1, 1, 1,
//               1, 1, 1,
//               1, 1, 1};

    cudaDeviceSynchronize();
    //convolution<<<dim3((renderer.screen.sizeX + BLOCK_X - 3)/(BLOCK_X-2), (renderer.screen.sizeY + BLOCK_Y - 3)/(BLOCK_Y-2)), dim3(BLOCK_X, BLOCK_Y)>>>(renderer.screen, c);

//    renderer.screen.copy(cudaMemcpyDeviceToHost);
//    cudaMemcpy(renderer.screen.d_image, sky.img, sizeof(ColorF) * renderer.screen.sizeX * renderer.screen.sizeY, cudaMemcpyHostToDevice);
    renderer.screen.copyAndSave("out.ppm");


    file.close();

    return 0;
}