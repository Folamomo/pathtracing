//
// Created by igor on 17.05.2021.
//

#define BLOCK_X 32
#define BLOCK_Y 32
#define EPSILON 0.0000001
#define BATCH_SIZE 1024
#define raysPerPixel 100
#include "Renderer.cuh"
#include <curand_kernel.h>
#include <curand.h>
#define clip(number) ( (number) = (number) < 0 ? 0 : (number) > 255 ? 255 : (number) )



__constant__ struct C_Camera {
    unsigned int x, y;
    Vector3 topLeft;
    Vector3 pixelDx, pixelDy;
    Vector3 origin;
} c_camera;


__device__ Vector3 directionFromXY(C_Camera c, unsigned int x, unsigned int y, curandState *state){
    return (c.topLeft + (x + curand_uniform (state)) * c.pixelDx + (y + curand_uniform (state)) * c.pixelDy).normalize();
}

__device__ int cast_ray(const Vector3 direction,
                         const Vector3 origin,
                         const Vector3* as,
                         const Vector3* edges1,
                         const Vector3* edges2,
                         const unsigned int triangles_n,
                         float* closest_distance){
    *closest_distance = 10000000.0f;
    int closest_id = -1;
    for (int t = 0; t < triangles_n; ++t) {
        Vector3 a = as[t];
        Vector3 edge1 = edges1[t];
        Vector3 edge2 = edges2[t];
        Vector3 h = direction.cross(edge2);
        float a_ = edge1.dot(h);
        // The ray is parallel to this triangle.
        if (a_ > -EPSILON && a_ < EPSILON) continue;
        float f = 1.0f / a_;
        Vector3 s = origin - a;
        float u = f * s.dot(h);
        Vector3 q = s.cross(edge1);
        float v = f * direction.dot(q);
        // The ray intercepts the plane outside the triangle;
        if (v < 0.0 || u + v > 1.0 || u < 0.0 || u > 1.0) continue;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float distance = f * edge2.dot(q);
        if (distance > EPSILON && distance < *closest_distance) {
            closest_id = t;
            *closest_distance = distance;
        }
    }
    return closest_id;
}

void Renderer::uploadScene() {
    cudaMalloc(&d_vectors, scene.vertices.size() * sizeof(Vector3));
    cudaMalloc(&d_materials, scene.materials.size() * sizeof(Material));
    cudaMalloc(&d_triangles, scene.triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_vectors, scene.vertices.data(), scene.vertices.size() * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, scene.triangles.data(), scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);


//    cudaMalloc(&d_skybox_tex, scene.skybox.y * scene.skybox.x * sizeof(ColorF));
//    cudaMemcpy(d_skybox_tex, scene.skybox.img, scene.skybox.y * scene.skybox.x * sizeof(ColorF), cudaMemcpyHostToDevice);
}

__global__ void random_init(curandState *dev_random){
    unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;
    curand_init(id, id, 0, &dev_random[id]);
}

__global__  void calculatePixel(
        const Screen screen,
        const Vector3* d_vertices,
        const Material* d_materials,
        const Triangle* d_triangles,
        const unsigned int triangles_n,
        curandState *dev_random
        ){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;


    extern __shared__ Vector3 triangles[];
    Vector3* as = &triangles[0];
    Vector3* edges1 = &triangles[BATCH_SIZE];
    Vector3* edges2 = &triangles[BATCH_SIZE * 2];

    if(id < triangles_n){
        as[id] = d_vertices[d_triangles[id].a];
        edges1[id] = d_vertices[d_triangles[id].b] - as[id];
        edges2[id] = d_vertices[d_triangles[id].c] - as[id];
    }
    __syncthreads();
    ColorF final_color{};
    for(int i = 0; i < raysPerPixel; ++i) {
        float closest_distance;
        Vector3 ray = directionFromXY(c_camera, x, y, &dev_random[id]);
        int closest_id = cast_ray(ray, c_camera.origin, as, edges1, edges2, triangles_n, &closest_distance);
        const Material& material = d_materials[d_triangles[closest_id].material];
        if (closest_id >= 0) {
            //emission color
            final_color += material.emit_color * material.emit;
            //specular color
            Vector3 normal = edges1[closest_id].cross(edges2[closest_id]).normalize();
            Vector3 reflection_dir = -2.0f * ray.dot(normal) * normal + ray;
            Vector3 reflection_origin = c_camera.origin + closest_distance * ray;
            int reflection_id = cast_ray(reflection_dir, reflection_origin, as, edges1, edges2, triangles_n, &closest_distance);
            if (reflection_id >= 0) {
                const Material &ref_material = d_materials[d_triangles[reflection_id].material];
                final_color += ref_material.emit_color * material.specular * ref_material.emit;
            } else {
                final_color += ColorF{0, 0, 255.0f * (0.5f + asin(reflection_dir.y) / 3.1415f)} * material.specular;
            }
        } else {
          final_color +=  ColorF{0, 0, 255.0f * (0.5f + asin(ray.y)/3.1415f)};
        }
    }
    screen.d_image[y * screen.sizeX + x] = final_color * (1.0f/raysPerPixel);
}

void Renderer::render() {
    uploadScene();
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, BLOCK_X * BLOCK_Y * sizeof(curandState));
    C_Camera c {
        camera.x,
        camera.y,
        camera.topLeft,
        camera.pixelDx,
        camera.pixelDy,
        camera.origin
    };
    cudaMemcpyToSymbol(c_camera, &c, sizeof(C_Camera));

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess) std::cout << "Error before cudaExecute: " << cudaGetErrorString(error);
    random_init<<<1, dim3(BLOCK_X, BLOCK_Y)>>>(dev_random);
    calculatePixel<<<dim3((screen.sizeX + BLOCK_X - 1)/BLOCK_X,
                          (screen.sizeY + BLOCK_Y - 1)/BLOCK_Y),
                          dim3(BLOCK_X, BLOCK_Y), BATCH_SIZE * sizeof(Vector3) * 3>>>
    (
            screen,
             d_vectors,
             d_materials,
             d_triangles,
             scene.triangles.size(),
             dev_random
     );
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) std::cout << "Error in cudaExecute: " << cudaGetErrorString(error);

    cudaFree(&dev_random);
    cudaFree(&d_vectors);
    cudaFree(&d_materials);
    cudaFree(&d_triangles);
//    cudaFree(&d_skybox_tex);

}

Renderer::Renderer(Screen &&screen, Scene &&scene, Camera &&camera) : screen(screen), scene(scene),
                                                                      camera(camera) {}

