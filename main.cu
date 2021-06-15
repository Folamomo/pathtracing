#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ColorF.cuh"
#include "Screen.cuh"
#include "Matrix4.cuh"
#include "Camera.cuh"
#include "Renderer.cuh"


using namespace std;


#define prune(number) ( number = number < 0 ? 0 : number > 1 ? 1 : number )



int main(){
    Screen screen{1920, 1080};
    Camera camera{120.0/180.0 * 3.1415, 1920, 1080};
    Scene scene;
    scene.loadObj("boxes.obj");

    scene.meshes[0].material = {{}, {255, 0, 0}, 1, {255,255,255}, 0};
    scene.meshes[1].material = {{}, {0, 255, 0}, 0.5, {255,255,255}, 0.5};
    scene.meshes[2].material = {{}, {255,105,180}, 0.5, {255,255,255}, 0.5};

    scene.loadObj("box.obj");

    scene.meshes[3].transform = Matrix4::fromTranslationVector({0, 0, -6}) * Matrix4::fromScaleFactor(2, 1, 1) * Matrix4::fromAxisAngle({1, 0, 0}, 1);
    scene.meshes[3].material = {{}, {0,200,200}, 0.1, {255,255,255}, 0.9};

    Renderer renderer;
    renderer.uploadScene(scene);
    renderer.uploadScreen(screen);
    renderer.uploadCamera(camera);
    renderer.render();

    cudaDeviceSynchronize();
    screen.copyAndSave("out.ppm");



    return 0;
}