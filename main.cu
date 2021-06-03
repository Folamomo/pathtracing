#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector2.cuh"
#include "ColorF.cuh"
#include "Screen.cuh"
#include "Matrix4.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "ConvMask.cuh"
#include "Renderer.cuh"
#include "Texture.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace std;


#define prune(number) ( number = number < 0 ? 0 : number > 1 ? 1 : number )



int main(){

    Renderer renderer{{1920, 1080},
                      {},
                      {120.0/180.0 * 3.1415, 1920, 1080}};


    renderer.scene.vertices.push_back({0.1, 0, -10});
    renderer.scene.vertices.push_back({1, 3, -10});
    renderer.scene.vertices.push_back({3, 3, -10});
    renderer.scene.vertices.push_back({3, 0, -8});
    renderer.scene.vertices.push_back({0, 1, -10.1});
    renderer.scene.vertices.push_back({-4, -4, -10});
    renderer.scene.vertices.push_back({4, -4, -13});
    renderer.scene.vertices.push_back({4, 4, -13});
    renderer.scene.vertices.push_back({-4, 4, -10});
    renderer.scene.materials.push_back({{}, {255, 0, 0}, 0.9, {}, 0.1});
    renderer.scene.materials.push_back({{}, {255, 255, 0}, 0.9, {}, 0.1});
    renderer.scene.materials.push_back({{}, {}, 0.3, {}, 0.7});
    renderer.scene.triangles.push_back({0, 1, 2, 0,0,0,0});
    renderer.scene.triangles.push_back({4, 3, 2, 0,0,0,1});
    renderer.scene.triangles.push_back({5, 6, 7, 0,0,0,2});
    renderer.scene.triangles.push_back({8, 5, 7, 0,0,0, 2});
    renderer.render();
//    ConvMask c{1, 1, 1,
//               1, 1, 1,
//               1, 1, 1};

    cudaDeviceSynchronize();
    //convolution<<<dim3((renderer.screen.sizeX + BLOCK_X - 3)/(BLOCK_X-2), (renderer.screen.sizeY + BLOCK_Y - 3)/(BLOCK_Y-2)), dim3(BLOCK_X, BLOCK_Y)>>>(renderer.screen, c);

//    renderer.screen.copy(cudaMemcpyDeviceToHost);
//    cudaMemcpy(renderer.screen.d_image, sky.img, sizeof(ColorF) * renderer.screen.sizeX * renderer.screen.sizeY, cudaMemcpyHostToDevice);
    renderer.screen.copyAndSave("out.ppm");



    return 0;
}