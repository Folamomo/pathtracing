#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector2.cuh"
#include "Color.cuh"
#include "Screen.cuh"
#include "Matrix4.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
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
#define clip(number) (unsigned char)( number = number < 0 ? 0 : number > 255 ? 255 : number )


__global__  void gradient(Screen::ScreenRef screen, const Vector2 left, const Vector2 right,
                         const Color leftColor, const Color rightColor){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= screen.sizeX || y >= screen.sizeY) return;

    float x1 = left.x, y1 = left.y, x2 = right.x, y2 = right.y;

    float color_p = ((x2 - x1) * (x - x1) + (y2 - y1) * (y - y1))/
                    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    prune(color_p);

    screen.d_image[x + screen.sizeX * y] = leftColor * (1-color_p) + rightColor * color_p;
}

__global__  void gradient_triangle(Screen::ScreenRef screen, const Vector2 A, const Vector2 B, const Vector2 C,
                          const Color color_a, const Color color_b, const Color color_c){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= screen.sizeX || y >= screen.sizeY) return;

    float acx = C.x - A.x;
    float acy = C.y - A.y;
    float bx = x - B.x;
    float by = y - B.y;
    float bax = A.x - B.x;
    float bay = A.y - B.y;

    float den = acx * by - acy - bx;
    if (den == 0.0) return;
    float ac = (bay * bx - bax * by) / den;
    float b = (acx * bay - acy * bax) / den;

//    if ( 0 < ac && 1 > ac && 1 < b) {
    float b_inv = 1/b;
    prune(b_inv);
    prune(ac);

        screen.d_image[x + screen.sizeX * y] =
                color_b * (1 - b_inv) + (color_c * ac + color_a * (1-ac) ) * b_inv;
//    }
}






__device__ __host__ Vector3 directionFromXY(Camera c, unsigned int x, unsigned int y){
    return (c.topLeft + x * c.pixelDx + y * c.pixelDy).normalize();
}

struct convMask{
    double data[3][3];
    __device__ __host__ double*  operator[](std::size_t at);
    __device__ __host__ convMask(std::initializer_list<double> data) noexcept;
};

__device__ __host__ double* convMask::operator[](std::size_t at){
    return data[at];
}
__device__ __host__ convMask::convMask(std::initializer_list<double> list) noexcept : data() {
    int i = 0;
    double sum = 0;
    for(double d: list){
        data[0][i] = d;
        sum += d;
        ++i;
    }

    for (int j = 0; j < 9; ++j){
        data[0][j] /= sum;
    }
}



__global__ void convolution(Screen::ScreenRef screen, convMask mask, Color* result){
    __shared__ Color old[BLOCK_X * BLOCK_Y];
    int x = threadIdx.x - 1 + blockIdx.x * (blockDim.x - 2);
    int y = threadIdx.y - 1 + blockIdx.y * (blockDim.y - 2);

    if (x >= screen.sizeX || y >= screen.sizeY) return;

    x = x < 0 ? 0 : x >= screen.sizeX ? screen.sizeX - 1 : x;
    y = y < 0 ? 0 : y >= screen.sizeY ? screen.sizeY - 1 : y;
    unsigned int shared_pos = threadIdx.x + threadIdx.y * BLOCK_X;
    old[shared_pos] = screen.d_image[y * screen.sizeX + x];

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == BLOCK_X - 1 || threadIdx.y == BLOCK_Y - 1 ) return;

    __syncthreads();

    double r = old[shared_pos - 1 - BLOCK_X].r * mask[0][0] +
               old[shared_pos     - BLOCK_X].r * mask[0][1] +
               old[shared_pos + 1 - BLOCK_X].r * mask[0][2] +
               old[shared_pos - 1          ].r * mask[1][0] +
               old[shared_pos              ].r * mask[1][1] +
               old[shared_pos + 1          ].r * mask[1][2] +
               old[shared_pos - 1 + BLOCK_X].r * mask[2][0] +
               old[shared_pos     + BLOCK_X].r * mask[2][1] +
               old[shared_pos + 1 + BLOCK_X].r * mask[2][2];

    double g = old[shared_pos - 1 - BLOCK_X].g * mask[0][0] +
               old[shared_pos     - BLOCK_X].g * mask[0][1] +
               old[shared_pos + 1 - BLOCK_X].g * mask[0][2] +
               old[shared_pos - 1          ].g * mask[1][0] +
               old[shared_pos              ].g * mask[1][1] +
               old[shared_pos + 1          ].g * mask[1][2] +
               old[shared_pos - 1 + BLOCK_X].g * mask[2][0] +
               old[shared_pos     + BLOCK_X].g * mask[2][1] +
               old[shared_pos + 1 + BLOCK_X].g * mask[2][2];

    double b = old[shared_pos - 1 - BLOCK_X].b * mask[0][0] +
               old[shared_pos     - BLOCK_X].b * mask[0][1] +
               old[shared_pos + 1 - BLOCK_X].b * mask[0][2] +
               old[shared_pos - 1          ].b * mask[1][0] +
               old[shared_pos              ].b * mask[1][1] +
               old[shared_pos + 1          ].b * mask[1][2] +
               old[shared_pos - 1 + BLOCK_X].b * mask[2][0] +
               old[shared_pos     + BLOCK_X].b * mask[2][1] +
               old[shared_pos + 1 + BLOCK_X].b * mask[2][2];

    result[y * screen.sizeX + x] = {(unsigned char)r, (unsigned char)g, (unsigned char)b};

}

__global__ void calculatePixel(Screen::ScreenRef screen, Sphere* d_spheres, unsigned int spheresSize, Camera camera) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= screen.sizeX || y >= screen.sizeY) return;

    Vector3 direction = directionFromXY(camera, x, y);

//    Sphere *closest = nullptr;
//    double minDistance;
//    for (int i = 0; i < spheresSize; ++i) {
//        Sphere *sphere = d_spheres + i;
//        double delta = pow((direction.dot(camera.origin - sphere->center)), 2) - ((camera.origin -
//                                                                                   sphere->center).squared() -
//                                                                                  sphere->radius * sphere->radius);
//        if (delta < 0) continue;
//        double distance = -(direction.dot(camera.origin - sphere->center)) - sqrt(delta);
//        if (distance < minDistance) {
//            closest = sphere;
//            minDistance = distance;
//        }
//    }
//    if (closest == nullptr) return;

    for (int i = 0; i < spheresSize; ++i) {
        Sphere *sphere = d_spheres + i;
        double delta = pow((direction.dot(camera.origin - sphere->center)), 2) - ((camera.origin -
                                                                                   sphere->center).squared() -
                                                                                  sphere->radius * sphere->radius);
        if (delta < 0) continue;

        double distance = -(direction.dot(camera.origin - sphere->center)) - sqrt(delta);
        Vector3 normal = ((camera.origin + distance * direction) - sphere->center).normalize();

        Vector3 lightDir = Vector3{1, 1, 1}.normalize();
        double color = lightDir.dot(normal);
        if (color < 0) color = 0;
        else if (color > 1) {
            color = 1;
        }

        double shiny = pow(color , 7) * color * 255;
        double red = 10 + color * sphere->color.r + shiny;
        double green = 10 + color * sphere->color.g + shiny;
        double blue = 10 + color * sphere->color.b + shiny;
        screen.d_image[x + screen.sizeX * y] = {clip(red), clip(green), clip(blue)};
    }

}

int main(){
    Screen screen {1920, 1080};

//    Vector2 A = {100, 500}, B  = {100, 1000}, C = {2000, 2000}, D = {1900, 100};
////    screen.cudaExecute(gradient_triangle, A, C, D, Color {255, 0, 0}, Color {0, 255, 0}, Color {0, 0, 255});
//    screen.cudaExecute(gradient_triangle, A, B, C, Color {255, 0, 0}, Color {0, 255, 0}, Color {0, 0, 255});
//    screen.copy(cudaMemcpyDeviceToHost);
//    screen.save("out.ppm");
    std::vector<Sphere> spheres{{{0, 0, -100}, 10.0, {255, 0, 0}},
                                {{-5, 1, -90},  7.0, {0, 0, 255}}};
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(Sphere) * spheres.size());
    cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

    Camera camera {90.0/180.0 * 3.1415, 1920, 1080};

    screen.cudaExecute(calculatePixel, d_spheres, spheres.size(), camera);

    convMask c{1, 1, 1,
               1, 1, 1,
               1, 1, 1};

    Color* result;
    cudaMalloc(&result, sizeof(Color) * screen.sizeX * screen.sizeY);
    convolution<<<dim3((screen.sizeX + BLOCK_X - 3)/(BLOCK_X-2), (screen.sizeY + BLOCK_Y - 3)/(BLOCK_Y-2)), dim3(BLOCK_X, BLOCK_Y)>>>(screen, c, result);
    cudaFree(screen.d_image);
    screen.d_image = result;

    screen.copy(cudaMemcpyDeviceToHost);
    screen.save("out.ppm");

    cudaDeviceSynchronize();
    return 0;
}