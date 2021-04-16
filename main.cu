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

using namespace std;


#define prune(number) (number = number < 0 ? 0 : number > 1 ? 1 : number)


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

    Sphere *sphere = d_spheres;
    double delta = pow((direction.dot(camera.origin - sphere->center)), 2) - ((camera.origin -
                                                                               sphere->center).squared() -
                                                                              sphere->radius * sphere->radius);
    if (delta < 0) return;



    screen.d_image[x + screen.sizeX * y] = {(unsigned char)(delta / (sphere->radius) / (sphere->radius) * 128), 0, 0};


}

int main(){
    Screen screen {1920, 1080};

//    Vector2 A = {100, 500}, B  = {100, 1000}, C = {2000, 2000}, D = {1900, 100};
////    screen.cudaExecute(gradient_triangle, A, C, D, Color {255, 0, 0}, Color {0, 255, 0}, Color {0, 0, 255});
//    screen.cudaExecute(gradient_triangle, A, B, C, Color {255, 0, 0}, Color {0, 255, 0}, Color {0, 0, 255});
//    screen.copy(cudaMemcpyDeviceToHost);
//    screen.save("out.ppm");
    std::vector<Sphere> spheres{{{0, 0, -100}, 10.0}};
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(Sphere) * spheres.size());
    cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

    Camera camera {60.0/180.0 * 3.1415, 1920, 1080};

    screen.cudaExecute(calculatePixel, d_spheres, spheres.size(), camera);
    screen.copy(cudaMemcpyDeviceToHost);
    screen.save("out.ppm");
//    for (unsigned int y = 0; y < 100; ++y) {
//        for (unsigned int x = 0; x < 300; ++x) {
//            Vector3 direction = directionFromXY(camera, x, y);
//            Sphere *sphere = spheres.data();
//            double delta = pow((direction.dot(camera.origin - sphere->center)), 2) - ((camera.origin -
//                                                                                       sphere->center).squared() -
//                                                                                      sphere->radius * sphere->radius);
//            if (delta > 0) cout << "#";
//            else cout << " ";
//        }
//        cout << "\n";
//    }



    return 0;
}